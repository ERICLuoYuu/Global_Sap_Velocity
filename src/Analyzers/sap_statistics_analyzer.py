"""
Sap Flow Statistics Analyzer
=============================
A script to investigate and summarize statistics of sap flow values for each site.

Generates:
1. Per-site summary statistics (mean, median, std, min, max, percentiles, etc.)
2. Data availability metrics (coverage, gaps, time range)
3. Distribution characteristics (skewness, kurtosis)
4. Visualizations (distributions, boxplots, time series overview)
5. Aggregated summary table and export to CSV

Usage:
    python sap_statistics_analyzer.py --time-scale daily --output-dir ./outputs/statistics
    python sap_statistics_analyzer.py --time-scale hourly --plot
"""

from pathlib import Path
import sys
import os
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from path_config import PathConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SapFlowStatisticsAnalyzer:
    """
    Comprehensive analyzer for sap flow statistics across all sites.
    Uses raw SAPFLUXNET data files from data/raw/0.1.5/0.1.5/csv/plant.
    Also analyzes sapwood area from plant metadata files (*_plant_md.csv).
    """
    
    # Default target column names to look for (for merged data)
    TARGET_COLUMNS = [
        'sapflow_mean', 'sap_flow', 'sapflow', 'vs', 'v_s', 
        'sap_velocity', 'sapwood_area_sap_flow', 'js'
    ]
    
    # Plant metadata columns of interest
    PLANT_METADATA_COLUMNS = [
        'pl_sapw_area',  # Sapwood area (cm²)
        'pl_dbh',        # Diameter at breast height (cm)
        'pl_height',     # Tree height (m)
        'pl_age',        # Tree age (years)
        'pl_bark_thick', # Bark thickness (mm)
        'pl_sapw_depth', # Sapwood depth (cm)
        'pl_leaf_area',  # Leaf area (m²)
        'pl_species',    # Species name
    ]
    
    def __init__(
        self, 
        data_dir: Union[str, Path] = None,
        time_scale: str = 'daily',
        target_column: str = 'sapflow_mean',
        output_dir: Union[str, Path] = None,
        use_raw_data: bool = True,
        use_outliers_removed: bool = False,
        scale: str = 'sapwood'
    ):
        """
        Initialize the analyzer.
        
        Args:
            data_dir: Directory containing site CSV files
            time_scale: 'daily' or 'hourly' (only used for merged data)
            target_column: Name of the sap flow column to analyze (for merged data)
            output_dir: Directory to save outputs
            use_raw_data: If True, use raw SAPFLUXNET files (*_sapf_data.csv)
            use_outliers_removed: If True, use outlier-removed data from 
                                  outputs/processed_data/sapwood/sap/outliers_removed
            scale: 'sapwood' or 'plant' - determines which raw data directory to use
        """
        self.time_scale = time_scale
        self.target_column = target_column
        self.use_raw_data = use_raw_data
        self.use_outliers_removed = use_outliers_removed
        self.scale = scale
        
        # Set up paths
        paths = PathConfig(scale=scale)
        
        if data_dir is None:
            if use_outliers_removed:
                # Use outlier-removed data
                self.data_dir = paths.sap_outliers_removed_dir
                self.file_pattern = '*_sapf_data_outliers_removed.csv'
            elif use_raw_data:
                # Use raw SAPFLUXNET data files from data/raw/0.1.5/0.1.5/csv/sapwood
                self.data_dir = paths.raw_csv_dir
                self.file_pattern = '*_sapf_data.csv'
            else:
                # Use merged/processed data
                self.data_dir = paths.merged_data_root / time_scale
                self.file_pattern = f'*_{time_scale}.csv'
        else:
            self.data_dir = Path(data_dir)
            # Determine file pattern based on data type
            if use_outliers_removed:
                self.file_pattern = '*_sapf_data_outliers_removed.csv'
            elif use_raw_data:
                self.file_pattern = '*_sapf_data.csv'
            else:
                self.file_pattern = f'*_{time_scale}.csv'
            
        if output_dir is None:
            if use_outliers_removed:
                self.output_dir = paths.figures_root / 'sap_flow_statistics' / 'outliers_removed'
            else:
                self.output_dir = paths.figures_root / 'sap_flow_statistics'
        else:
            self.output_dir = Path(output_dir)
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for results
        self.site_stats: Dict[str, Dict] = {}
        self.summary_df: Optional[pd.DataFrame] = None
        
    def find_target_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the sap flow target column in the DataFrame."""
        # First check the specified target column
        if self.target_column in df.columns:
            return self.target_column
        
        # Try alternative names
        for col in self.TARGET_COLUMNS:
            if col in df.columns:
                return col
                
        # Try case-insensitive match
        df_cols_lower = {c.lower(): c for c in df.columns}
        for col in self.TARGET_COLUMNS:
            if col.lower() in df_cols_lower:
                return df_cols_lower[col.lower()]
                
        return None
    
    def load_plant_metadata(self, site_id: str) -> Optional[pd.DataFrame]:
        """
        Load plant metadata for a site from the *_plant_md.csv file.
        
        Args:
            site_id: Site identifier (e.g., 'ARG_MAZ')
            
        Returns:
            DataFrame with plant metadata or None if not found
        """
        # Look for plant metadata file in the same directory as sap flow data
        plant_md_files = list(self.data_dir.glob(f'{site_id}_plant_md.csv'))
        
        if len(plant_md_files) == 0:
            return None
        
        try:
            df = pd.read_csv(plant_md_files[0])
            return df
        except Exception as e:
            logger.warning(f"Error loading plant metadata for {site_id}: {e}")
            return None
    
    def compute_sapwood_area_statistics(self, plant_md: pd.DataFrame) -> Dict:
        """
        Compute statistics for sapwood area and other plant metadata.
        
        Args:
            plant_md: DataFrame with plant metadata
            
        Returns:
            Dictionary with sapwood area and plant statistics
        """
        stats = {}
        
        # Sapwood area statistics (primary focus)
        if 'pl_sapw_area' in plant_md.columns:
            sapw_area = plant_md['pl_sapw_area'].dropna()
            if len(sapw_area) > 0:
                stats['sapw_area_n'] = len(sapw_area)
                stats['sapw_area_mean'] = sapw_area.mean()
                stats['sapw_area_median'] = sapw_area.median()
                stats['sapw_area_std'] = sapw_area.std()
                stats['sapw_area_min'] = sapw_area.min()
                stats['sapw_area_max'] = sapw_area.max()
                stats['sapw_area_cv'] = (sapw_area.std() / sapw_area.mean() * 100) if sapw_area.mean() != 0 else np.nan
                stats['sapw_area_sum'] = sapw_area.sum()  # Total sapwood area for site
        
        # DBH statistics
        if 'pl_dbh' in plant_md.columns:
            dbh = plant_md['pl_dbh'].dropna()
            if len(dbh) > 0:
                stats['dbh_mean'] = dbh.mean()
                stats['dbh_median'] = dbh.median()
                stats['dbh_std'] = dbh.std()
                stats['dbh_min'] = dbh.min()
                stats['dbh_max'] = dbh.max()
        
        # Tree height statistics
        if 'pl_height' in plant_md.columns:
            height = plant_md['pl_height'].dropna()
            if len(height) > 0:
                stats['height_mean'] = height.mean()
                stats['height_median'] = height.median()
                stats['height_min'] = height.min()
                stats['height_max'] = height.max()
        
        # Tree age statistics
        if 'pl_age' in plant_md.columns:
            age = plant_md['pl_age'].dropna()
            if len(age) > 0:
                stats['age_mean'] = age.mean()
                stats['age_median'] = age.median()
                stats['age_min'] = age.min()
                stats['age_max'] = age.max()
        
        # Sapwood depth statistics
        if 'pl_sapw_depth' in plant_md.columns:
            sapw_depth = plant_md['pl_sapw_depth'].dropna()
            if len(sapw_depth) > 0:
                stats['sapw_depth_mean'] = sapw_depth.mean()
                stats['sapw_depth_median'] = sapw_depth.median()
                stats['sapw_depth_min'] = sapw_depth.min()
                stats['sapw_depth_max'] = sapw_depth.max()
        
        # Species information
        if 'pl_species' in plant_md.columns:
            species = plant_md['pl_species'].dropna()
            if len(species) > 0:
                stats['n_species'] = species.nunique()
                stats['species_list'] = ', '.join(species.unique()[:5])  # First 5 species
                if species.nunique() > 5:
                    stats['species_list'] += f'... (+{species.nunique() - 5} more)'
        
        return stats
    
    def load_env_data(self, site_id: str) -> Optional[pd.DataFrame]:
        """
        Load environmental data from the *_env_data.csv file.
        
        Args:
            site_id: Site identifier (e.g., 'ARG_MAZ')
            
        Returns:
            DataFrame with environmental data or None if not found
        """
        env_files = list(self.data_dir.glob(f'{site_id}_env_data.csv'))
        
        if len(env_files) == 0:
            return None
        
        try:
            df = pd.read_csv(env_files[0])
            return df
        except Exception as e:
            logger.warning(f"Error loading env data for {site_id}: {e}")
            return None
    
    def compute_env_statistics(self, env_df: pd.DataFrame) -> Dict:
        """
        Compute statistics for environmental variables.
        
        Args:
            env_df: DataFrame with environmental data
            
        Returns:
            Dictionary with environmental statistics
        """
        stats = {}
        
        # Key environmental variables
        env_vars = {
            'ta': 'Air Temperature (°C)',
            'vpd': 'Vapor Pressure Deficit (kPa)',
            'sw_in': 'Shortwave Radiation (W/m²)',
            'rh': 'Relative Humidity (%)',
            'ppfd_in': 'PPFD (µmol/m²/s)',
            'ws': 'Wind Speed (m/s)',
            'precip': 'Precipitation (mm)',
            'swc_shallow': 'Soil Water Content (%)',
        }
        
        for var, description in env_vars.items():
            if var in env_df.columns:
                data = env_df[var].dropna()
                if len(data) > 0:
                    stats[f'{var}_mean'] = data.mean()
                    stats[f'{var}_median'] = data.median()
                    stats[f'{var}_std'] = data.std()
                    stats[f'{var}_min'] = data.min()
                    stats[f'{var}_max'] = data.max()
                    stats[f'{var}_n'] = len(data)
        
        return stats
    
    def load_site_metadata(self, site_id: str) -> Optional[pd.DataFrame]:
        """
        Load site metadata from the *_site_md.csv file.
        
        Args:
            site_id: Site identifier (e.g., 'ARG_MAZ')
            
        Returns:
            DataFrame with site metadata or None if not found
        """
        site_md_files = list(self.data_dir.glob(f'{site_id}_site_md.csv'))
        
        if len(site_md_files) == 0:
            return None
        
        try:
            df = pd.read_csv(site_md_files[0])
            return df
        except Exception as e:
            logger.warning(f"Error loading site metadata for {site_id}: {e}")
            return None
    
    def get_site_info(self, site_id: str) -> Dict:
        """
        Get site-level information including biome, location, climate.
        
        Args:
            site_id: Site identifier
            
        Returns:
            Dictionary with site information
        """
        site_md = self.load_site_metadata(site_id)
        info = {}
        
        if site_md is not None and len(site_md) > 0:
            row = site_md.iloc[0]
            
            # Biome
            if 'si_biome' in site_md.columns:
                info['biome'] = row['si_biome']
            
            # IGBP classification
            if 'si_igbp' in site_md.columns:
                info['igbp'] = row['si_igbp']
            
            # Location
            if 'si_lat' in site_md.columns:
                info['lat'] = row['si_lat']
            if 'si_long' in site_md.columns:
                info['lon'] = row['si_long']
            if 'si_elev' in site_md.columns:
                info['elevation'] = row['si_elev']
            if 'si_country' in site_md.columns:
                info['country'] = row['si_country']
            
            # Climate
            if 'si_mat' in site_md.columns:
                info['mat'] = row['si_mat']  # Mean annual temperature
            if 'si_map' in site_md.columns:
                info['map'] = row['si_map']  # Mean annual precipitation
            
            # Site name
            if 'si_name' in site_md.columns:
                info['site_name'] = row['si_name']
        
        return info

    
    def get_sap_flow_columns(self, df: pd.DataFrame, site_id: str) -> List[str]:
        """
        Get all sap flow measurement columns from raw SAPFLUXNET data.
        These are columns that contain the site_id and are not TIMESTAMP columns.
        
        Args:
            df: DataFrame with raw SAPFLUXNET data
            site_id: Site identifier (e.g., 'ARG_MAZ')
            
        Returns:
            List of column names with sap flow measurements
        """
        sap_cols = []
        for col in df.columns:
            # Skip timestamp columns
            if 'TIMESTAMP' in col.upper():
                continue
            # Check if column contains site_id (these are tree measurements)
            if site_id in col:
                sap_cols.append(col)

        # If no columns found, try shorter prefixes (handles sub-site suffixes,
        # e.g. site_id 'COL_MAC_SAF_RAD' -> columns use base 'COL_MAC_SAF')
        if not sap_cols:
            parts = site_id.split('_')
            for n in range(len(parts) - 1, 1, -1):
                prefix = '_'.join(parts[:n])
                sap_cols = [
                    col for col in df.columns
                    if 'TIMESTAMP' not in col.upper() and prefix in col
                ]
                if sap_cols:
                    logger.debug(f"Site {site_id}: matched columns using prefix '{prefix}'")
                    break

        return sap_cols
    
    def compute_raw_site_statistics(self, df: pd.DataFrame, site_id: str) -> Dict:
        """
        Compute statistics for raw SAPFLUXNET data (multiple tree columns).
        
        Args:
            df: DataFrame with raw site data
            site_id: Site identifier
            
        Returns:
            Dictionary with all computed statistics
        """
        # Find all sap flow measurement columns for this site
        sap_cols = self.get_sap_flow_columns(df, site_id)
        
        if len(sap_cols) == 0:
            logger.warning(f"No sap flow columns found for site {site_id}")
            return {'site_id': site_id, 'error': 'No sap flow columns found'}
        
        # Compute mean across all trees for each timestamp
        sap_values_all = df[sap_cols].values.flatten()
        sap_values_all = sap_values_all[~np.isnan(sap_values_all)]
        
        if len(sap_values_all) == 0:
            logger.warning(f"No valid sap flow data for site {site_id}")
            return {'site_id': site_id, 'error': 'No valid data'}
        
        # Also compute mean sap flow per timestamp (across trees)
        sap_mean_per_time = df[sap_cols].mean(axis=1).dropna()
        
        # Number of trees
        n_trees = len(sap_cols)
        
        # Basic statistics on ALL measurements (across all trees and times)
        stats_dict = {
            'site_id': site_id,
            'n_trees': n_trees,
            'tree_columns': ', '.join(sap_cols[:5]) + ('...' if len(sap_cols) > 5 else ''),
            
            # Sample size
            'n_timestamps': len(df),
            'n_total_measurements': len(sap_values_all),
            'n_valid_per_tree_avg': len(sap_values_all) / n_trees,
            'n_missing': np.isnan(df[sap_cols].values).sum(),
            'missing_pct': np.isnan(df[sap_cols].values).sum() / df[sap_cols].values.size * 100,
            
            # Central tendency (all measurements)
            'mean': np.mean(sap_values_all),
            'median': np.median(sap_values_all),
            
            # Per-timestamp mean statistics
            'mean_per_time_mean': sap_mean_per_time.mean(),
            'mean_per_time_std': sap_mean_per_time.std(),
            
            # Dispersion
            'std': np.std(sap_values_all),
            'var': np.var(sap_values_all),
            'cv': (np.std(sap_values_all) / np.mean(sap_values_all) * 100) if np.mean(sap_values_all) != 0 else np.nan,
            'iqr': np.percentile(sap_values_all, 75) - np.percentile(sap_values_all, 25),
            'range': np.max(sap_values_all) - np.min(sap_values_all),
            
            # Extremes
            'min': np.min(sap_values_all),
            'max': np.max(sap_values_all),
            
            # Percentiles
            'p01': np.percentile(sap_values_all, 1),
            'p05': np.percentile(sap_values_all, 5),
            'p10': np.percentile(sap_values_all, 10),
            'p25': np.percentile(sap_values_all, 25),
            'p50': np.percentile(sap_values_all, 50),
            'p75': np.percentile(sap_values_all, 75),
            'p90': np.percentile(sap_values_all, 90),
            'p95': np.percentile(sap_values_all, 95),
            'p99': np.percentile(sap_values_all, 99),
            
            # Distribution shape
            'skewness': stats.skew(sap_values_all),
            'kurtosis': stats.kurtosis(sap_values_all),
            
            # Zero/negative value analysis
            'n_zeros': (sap_values_all == 0).sum(),
            'pct_zeros': (sap_values_all == 0).sum() / len(sap_values_all) * 100,
            'n_negative': (sap_values_all < 0).sum(),
            'pct_negative': (sap_values_all < 0).sum() / len(sap_values_all) * 100,
            
            # Extreme value analysis (potential outliers)
            'n_above_mean_3std': ((sap_values_all > np.mean(sap_values_all) + 3 * np.std(sap_values_all))).sum(),
            'n_below_mean_3std': ((sap_values_all < np.mean(sap_values_all) - 3 * np.std(sap_values_all))).sum(),
            
            # Tree-level statistics
            'tree_mean_min': df[sap_cols].mean().min(),
            'tree_mean_max': df[sap_cols].mean().max(),
            'tree_mean_std': df[sap_cols].mean().std(),
        }
        
        # Time range
        timestamp_cols = ['TIMESTAMP', 'solar_TIMESTAMP']
        for ts_col in timestamp_cols:
            if ts_col in df.columns:
                try:
                    timestamps = pd.to_datetime(df[ts_col])
                    stats_dict['start_date'] = timestamps.min()
                    stats_dict['end_date'] = timestamps.max()
                    stats_dict['date_range_days'] = (timestamps.max() - timestamps.min()).days
                    break
                except:
                    pass
        
        # Load and add plant metadata statistics (sapwood area, DBH, etc.)
        plant_md = self.load_plant_metadata(site_id)
        if plant_md is not None:
            sapwood_stats = self.compute_sapwood_area_statistics(plant_md)
            stats_dict.update(sapwood_stats)
        
        # Load and add site metadata (biome, location, climate)
        site_info = self.get_site_info(site_id)
        stats_dict.update(site_info)
        
        # Load and add environmental data statistics
        env_df = self.load_env_data(site_id)
        if env_df is not None:
            env_stats = self.compute_env_statistics(env_df)
            stats_dict.update(env_stats)
        
        return stats_dict
    
    def compute_site_statistics(self, df: pd.DataFrame, site_id: str) -> Dict:
        """
        Compute comprehensive statistics for a single site.
        
        Args:
            df: DataFrame with site data
            site_id: Site identifier
            
        Returns:
            Dictionary with all computed statistics
        """
        target_col = self.find_target_column(df)
        
        if target_col is None:
            logger.warning(f"No sap flow column found for site {site_id}")
            return {'site_id': site_id, 'error': 'No sap flow column found'}
        
        # Get the sap flow values
        sap_values = df[target_col].dropna()
        
        if len(sap_values) == 0:
            logger.warning(f"No valid sap flow data for site {site_id}")
            return {'site_id': site_id, 'error': 'No valid data'}
        
        # Basic statistics
        stats_dict = {
            'site_id': site_id,
            'target_column': target_col,
            
            # Sample size
            'n_total_rows': len(df),
            'n_valid_values': len(sap_values),
            'n_missing': df[target_col].isna().sum(),
            'missing_pct': df[target_col].isna().sum() / len(df) * 100,
            
            # Central tendency
            'mean': sap_values.mean(),
            'median': sap_values.median(),
            'mode': sap_values.mode().iloc[0] if len(sap_values.mode()) > 0 else np.nan,
            
            # Dispersion
            'std': sap_values.std(),
            'var': sap_values.var(),
            'cv': (sap_values.std() / sap_values.mean() * 100) if sap_values.mean() != 0 else np.nan,
            'iqr': sap_values.quantile(0.75) - sap_values.quantile(0.25),
            'range': sap_values.max() - sap_values.min(),
            
            # Extremes
            'min': sap_values.min(),
            'max': sap_values.max(),
            
            # Percentiles
            'p01': sap_values.quantile(0.01),
            'p05': sap_values.quantile(0.05),
            'p10': sap_values.quantile(0.10),
            'p25': sap_values.quantile(0.25),
            'p50': sap_values.quantile(0.50),
            'p75': sap_values.quantile(0.75),
            'p90': sap_values.quantile(0.90),
            'p95': sap_values.quantile(0.95),
            'p99': sap_values.quantile(0.99),
            
            # Distribution shape
            'skewness': stats.skew(sap_values),
            'kurtosis': stats.kurtosis(sap_values),
            
            # Zero/negative value analysis
            'n_zeros': (sap_values == 0).sum(),
            'pct_zeros': (sap_values == 0).sum() / len(sap_values) * 100,
            'n_negative': (sap_values < 0).sum(),
            'pct_negative': (sap_values < 0).sum() / len(sap_values) * 100,
            
            # Extreme value analysis (potential outliers)
            'n_above_mean_3std': ((sap_values > sap_values.mean() + 3 * sap_values.std())).sum(),
            'n_below_mean_3std': ((sap_values < sap_values.mean() - 3 * sap_values.std())).sum(),
        }
        
        # Time range (if timestamp available)
        timestamp_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'datetime', 'date', 'time']
        for ts_col in timestamp_cols:
            if ts_col in df.columns:
                try:
                    timestamps = pd.to_datetime(df[ts_col])
                    stats_dict['start_date'] = timestamps.min()
                    stats_dict['end_date'] = timestamps.max()
                    stats_dict['date_range_days'] = (timestamps.max() - timestamps.min()).days
                    break
                except:
                    pass
        
        # Site metadata (if available)
        metadata_cols = {
            'lat': ['lat', 'latitude', 'latitude_x'],
            'lon': ['lon', 'longitude', 'longitude_x'],
            'pft': ['pft', 'plant_functional_type', 'biome'],
            'elevation': ['elevation', 'elev', 'altitude'],
            'canopy_height': ['canopy_height', 'height'],
        }
        
        for meta_name, possible_cols in metadata_cols.items():
            for col in possible_cols:
                if col in df.columns:
                    value = df[col].dropna()
                    if len(value) > 0:
                        if meta_name in ['pft']:
                            stats_dict[meta_name] = value.mode().iloc[0] if len(value.mode()) > 0 else None
                        else:
                            stats_dict[meta_name] = value.median()
                    break
        
        return stats_dict
    
    def analyze_all_sites(self, max_sites: int = None) -> pd.DataFrame:
        """
        Analyze all site files and compute statistics.
        
        Args:
            max_sites: Maximum number of sites to process (for testing)
            
        Returns:
            DataFrame with statistics for all sites
        """
        logger.info(f"Scanning for site files in: {self.data_dir}")
        logger.info(f"Using raw data: {self.use_raw_data}")
        logger.info(f"Using outliers removed: {self.use_outliers_removed}")
        logger.info(f"File pattern: {self.file_pattern}")
        
        # Find all site files using the file pattern
        site_files = sorted(list(self.data_dir.glob(self.file_pattern)))
        
        # Filter out merged files for non-raw data
        if not self.use_raw_data and not self.use_outliers_removed:
            site_files = [f for f in site_files if 'all_biomes_merged' not in f.name]
        
        if max_sites:
            site_files = site_files[:max_sites]
        
        logger.info(f"Found {len(site_files)} site files to process")
        
        all_stats = []
        
        for i, file_path in enumerate(site_files):
            if self.use_outliers_removed:
                # Extract site_id from filename like 'ARG_MAZ_sapf_data_outliers_removed.csv'
                site_id = file_path.stem.replace('_sapf_data_outliers_removed', '')
            elif self.use_raw_data:
                # Extract site_id from filename like 'ARG_MAZ_sapf_data.csv'
                site_id = file_path.stem.replace('_sapf_data', '')
            else:
                site_id = file_path.stem.replace(f'_{self.time_scale}', '')
            
            if (i + 1) % 20 == 0:
                logger.info(f"Processing site {i + 1}/{len(site_files)}: {site_id}")
            
            try:
                # Load data
                df = pd.read_csv(file_path)
                
                # Try to parse TIMESTAMP if present
                if 'TIMESTAMP' in df.columns:
                    try:
                        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
                    except:
                        pass
                
                # Compute statistics - use raw method for both raw and outliers_removed data
                if self.use_raw_data or self.use_outliers_removed:
                    site_stats = self.compute_raw_site_statistics(df, site_id)
                else:
                    site_stats = self.compute_site_statistics(df, site_id)
                site_stats['file_path'] = str(file_path)
                
                all_stats.append(site_stats)
                self.site_stats[site_id] = site_stats
                
            except Exception as e:
                logger.error(f"Error processing {site_id}: {e}")
                all_stats.append({
                    'site_id': site_id,
                    'error': str(e),
                    'file_path': str(file_path)
                })
        
        # Create summary DataFrame
        self.summary_df = pd.DataFrame(all_stats)
        
        # Reorder columns for better readability
        if self.use_raw_data or self.use_outliers_removed:
            priority_cols = [
                'site_id', 'biome', 'igbp', 'country', 'lat', 'lon', 'elevation', 'mat', 'map',
                'n_trees', 'n_timestamps', 'n_total_measurements', 
                'mean', 'median', 'std', 'min', 'max',
                'p05', 'p25', 'p50', 'p75', 'p95', 'cv', 'skewness', 'kurtosis',
                'pct_zeros', 'pct_negative', 'missing_pct',
                'tree_mean_min', 'tree_mean_max', 'tree_mean_std',
                'start_date', 'end_date', 'date_range_days',
                # Sapwood area and plant metadata columns
                'sapw_area_n', 'sapw_area_mean', 'sapw_area_median', 'sapw_area_std',
                'sapw_area_min', 'sapw_area_max', 'sapw_area_cv', 'sapw_area_sum',
                'dbh_mean', 'dbh_median', 'dbh_min', 'dbh_max',
                'height_mean', 'height_median', 'height_min', 'height_max',
                'age_mean', 'age_median', 'age_min', 'age_max',
                'sapw_depth_mean', 'sapw_depth_median',
                'n_species', 'species_list'
            ]
        else:
            priority_cols = [
                'site_id', 'n_valid_values', 'mean', 'median', 'std', 'min', 'max',
                'p05', 'p25', 'p50', 'p75', 'p95', 'cv', 'skewness', 'kurtosis',
                'pct_zeros', 'pct_negative', 'missing_pct', 'pft', 'lat', 'lon',
                'start_date', 'end_date', 'date_range_days'
            ]
        
        existing_priority = [c for c in priority_cols if c in self.summary_df.columns]
        other_cols = [c for c in self.summary_df.columns if c not in priority_cols]
        self.summary_df = self.summary_df[existing_priority + other_cols]
        
        logger.info(f"Completed analysis for {len(all_stats)} sites")
        
        return self.summary_df
    
    def get_summary_statistics(self) -> Dict:
        """
        Get aggregated summary statistics across all sites.
        
        Returns:
            Dictionary with cross-site summary
        """
        if self.summary_df is None:
            raise ValueError("No analysis results. Run analyze_all_sites() first.")
        
        # Filter to valid sites only
        valid_df = self.summary_df[self.summary_df['error'].isna()] if 'error' in self.summary_df.columns else self.summary_df
        
        # Determine the observation count column based on data type
        obs_col = 'n_total_measurements' if self.use_raw_data else 'n_valid_values'
        
        summary = {
            'total_sites': len(self.summary_df),
            'valid_sites': len(valid_df),
            'failed_sites': len(self.summary_df) - len(valid_df),
            
            'total_observations': valid_df[obs_col].sum() if obs_col in valid_df.columns else None,
            
            # Cross-site statistics of site means
            'mean_of_means': valid_df['mean'].mean() if 'mean' in valid_df.columns else None,
            'std_of_means': valid_df['mean'].std() if 'mean' in valid_df.columns else None,
            'min_site_mean': valid_df['mean'].min() if 'mean' in valid_df.columns else None,
            'max_site_mean': valid_df['mean'].max() if 'mean' in valid_df.columns else None,
            
            # Cross-site statistics of site medians
            'mean_of_medians': valid_df['median'].mean() if 'median' in valid_df.columns else None,
            'std_of_medians': valid_df['median'].std() if 'median' in valid_df.columns else None,
            'min_site_median': valid_df['median'].min() if 'median' in valid_df.columns else None,
            'max_site_median': valid_df['median'].max() if 'median' in valid_df.columns else None,
            
            # Range of max values
            'max_value_overall': valid_df['max'].max() if 'max' in valid_df.columns else None,
            'min_value_overall': valid_df['min'].min() if 'min' in valid_df.columns else None,
            
            # Data availability
            'median_observations_per_site': valid_df[obs_col].median() if obs_col in valid_df.columns else None,
            'median_missing_pct': valid_df['missing_pct'].median() if 'missing_pct' in valid_df.columns else None,
        }
        
        # Add raw-data specific statistics
        if self.use_raw_data:
            summary['total_trees'] = valid_df['n_trees'].sum() if 'n_trees' in valid_df.columns else None
            summary['median_trees_per_site'] = valid_df['n_trees'].median() if 'n_trees' in valid_df.columns else None
            
            # Sapwood area summary statistics
            if 'sapw_area_mean' in valid_df.columns:
                sapw_data = valid_df['sapw_area_mean'].dropna()
                if len(sapw_data) > 0:
                    summary['sapwood_area_stats'] = {
                        'sites_with_sapw_area': len(sapw_data),
                        'sites_missing_sapw_area': len(valid_df) - len(sapw_data),
                        'mean_of_site_means': sapw_data.mean(),
                        'std_of_site_means': sapw_data.std(),
                        'min_site_mean': sapw_data.min(),
                        'max_site_mean': sapw_data.max(),
                        'median_site_mean': sapw_data.median(),
                    }
            
            # Total sapwood area across all sites
            if 'sapw_area_sum' in valid_df.columns:
                total_sapw = valid_df['sapw_area_sum'].dropna().sum()
                summary['total_sapwood_area_cm2'] = total_sapw
            
            # DBH summary statistics
            if 'dbh_mean' in valid_df.columns:
                dbh_data = valid_df['dbh_mean'].dropna()
                if len(dbh_data) > 0:
                    summary['dbh_stats'] = {
                        'sites_with_dbh': len(dbh_data),
                        'mean_of_site_means': dbh_data.mean(),
                        'min_site_mean': dbh_data.min(),
                        'max_site_mean': dbh_data.max(),
                    }
            
            # Tree height summary statistics
            if 'height_mean' in valid_df.columns:
                height_data = valid_df['height_mean'].dropna()
                if len(height_data) > 0:
                    summary['height_stats'] = {
                        'sites_with_height': len(height_data),
                        'mean_of_site_means': height_data.mean(),
                        'min_site_mean': height_data.min(),
                        'max_site_mean': height_data.max(),
                    }
            
            # Tree age summary statistics
            if 'age_mean' in valid_df.columns:
                age_data = valid_df['age_mean'].dropna()
                if len(age_data) > 0:
                    summary['age_stats'] = {
                        'sites_with_age': len(age_data),
                        'mean_of_site_means': age_data.mean(),
                        'min_site_mean': age_data.min(),
                        'max_site_mean': age_data.max(),
                    }
            
            # Biome distribution
            if 'biome' in valid_df.columns:
                biome_data = valid_df['biome'].dropna()
                if len(biome_data) > 0:
                    summary['biome_distribution'] = biome_data.value_counts().to_dict()
        else:
            # PFT distribution (only for merged data)
            summary['pft_distribution'] = valid_df['pft'].value_counts().to_dict() if 'pft' in valid_df.columns else None
        
        return summary
    
    def get_biome_statistics(self) -> pd.DataFrame:
        """
        Get sap flow statistics grouped by biome.
        
        Returns:
            DataFrame with statistics by biome
        """
        if self.summary_df is None:
            raise ValueError("No analysis results. Run analyze_all_sites() first.")
        
        # Filter to valid sites only
        valid_df = self.summary_df[self.summary_df['error'].isna()].copy() if 'error' in self.summary_df.columns else self.summary_df.copy()
        
        # Filter out inf values
        valid_df = valid_df[~np.isinf(valid_df['mean'])]
        
        if 'biome' not in valid_df.columns:
            logger.warning("No biome column found in data")
            return pd.DataFrame()
        
        # Group by biome and compute statistics
        biome_stats = valid_df.groupby('biome').agg({
            'site_id': 'count',
            'n_trees': 'sum',
            'n_total_measurements': 'sum',
            'mean': ['mean', 'std', 'min', 'max', 'median'],
            'median': ['mean', 'std', 'min', 'max'],
            'std': 'mean',
            'cv': 'mean',
            'pct_zeros': 'mean',
            'pct_negative': 'mean',
            'missing_pct': 'mean',
        }).round(4)
        
        # Flatten column names
        biome_stats.columns = ['_'.join(col).strip('_') for col in biome_stats.columns.values]
        biome_stats = biome_stats.rename(columns={
            'site_id_count': 'n_sites',
            'n_trees_sum': 'total_trees',
            'n_total_measurements_sum': 'total_observations',
            'mean_mean': 'sapflow_mean_of_means',
            'mean_std': 'sapflow_std_of_means',
            'mean_min': 'sapflow_min_site_mean',
            'mean_max': 'sapflow_max_site_mean',
            'mean_median': 'sapflow_median_of_means',
            'median_mean': 'sapflow_mean_of_medians',
            'median_std': 'sapflow_std_of_medians',
            'median_min': 'sapflow_min_site_median',
            'median_max': 'sapflow_max_site_median',
            'std_mean': 'mean_site_std',
            'cv_mean': 'mean_cv',
            'pct_zeros_mean': 'mean_pct_zeros',
            'pct_negative_mean': 'mean_pct_negative',
            'missing_pct_mean': 'mean_missing_pct',
        })
        
        # Sort by number of sites
        biome_stats = biome_stats.sort_values('n_sites', ascending=False)
        
        # Add sapwood area statistics by biome if available
        if 'sapw_area_mean' in valid_df.columns:
            sapw_by_biome = valid_df.groupby('biome').agg({
                'sapw_area_mean': ['mean', 'std', 'count'],
                'sapw_area_sum': 'sum',
            }).round(4)
            sapw_by_biome.columns = ['sapw_area_mean_of_means', 'sapw_area_std', 'sites_with_sapw_area', 'total_sapw_area']
            biome_stats = biome_stats.join(sapw_by_biome)
        
        return biome_stats.reset_index()
    
    def save_results(self, prefix: str = None):
        """Save analysis results to CSV files."""
        if self.summary_df is None:
            raise ValueError("No analysis results. Run analyze_all_sites() first.")
        
        if prefix is None:
            if self.use_raw_data:
                prefix = f"sap_flow_stats_{self.scale}_raw"
            else:
                prefix = f"sap_flow_stats_{self.scale}_{self.time_scale}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed statistics
        detailed_path = self.output_dir / f"{prefix}_detailed_{timestamp}.csv"
        self.summary_df.to_csv(detailed_path, index=False)
        logger.info(f"Saved detailed statistics to: {detailed_path}")
        
        # Save summary
        summary = self.get_summary_statistics()
        summary_path = self.output_dir / f"{prefix}_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SAP FLOW STATISTICS SUMMARY\n")
            f.write(f"Time Scale: {self.time_scale}\n")
            f.write(f"Generated: {timestamp}\n")
            f.write("=" * 60 + "\n\n")
            
            for key, value in summary.items():
                if isinstance(value, dict):
                    f.write(f"\n{key}:\n")
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                elif isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        logger.info(f"Saved summary to: {summary_path}")
        
        return detailed_path, summary_path
    
    def plot_distributions(self, top_n: int = 20, save: bool = True):
        """
        Create distribution plots for sap flow values.
        
        Args:
            top_n: Number of sites to highlight
            save: Whether to save the plots
        """
        if self.summary_df is None:
            raise ValueError("No analysis results. Run analyze_all_sites() first.")
        
        valid_df = self.summary_df[self.summary_df['mean'].notna()].copy()
        # Filter out infinite values
        valid_df = valid_df[~np.isinf(valid_df['mean'])]
        valid_df = valid_df[~np.isinf(valid_df['max'])]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Distribution of site means
        ax = axes[0, 0]
        ax.hist(valid_df['mean'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(valid_df['mean'].mean(), color='red', linestyle='--', label=f"Mean: {valid_df['mean'].mean():.2f}")
        ax.axvline(valid_df['mean'].median(), color='orange', linestyle='--', label=f"Median: {valid_df['mean'].median():.2f}")
        ax.set_xlabel('Site Mean Sap Flow')
        ax.set_ylabel('Number of Sites')
        ax.set_title('Distribution of Site Mean Values')
        ax.legend()
        
        # 2. Distribution of site max values
        ax = axes[0, 1]
        ax.hist(valid_df['max'], bins=30, edgecolor='black', alpha=0.7, color='coral')
        ax.axvline(valid_df['max'].mean(), color='red', linestyle='--', label=f"Mean: {valid_df['max'].mean():.2f}")
        ax.set_xlabel('Site Maximum Sap Flow')
        ax.set_ylabel('Number of Sites')
        ax.set_title('Distribution of Site Maximum Values')
        ax.legend()
        
        # 3. Distribution of CV (coefficient of variation)
        ax = axes[0, 2]
        cv_data = valid_df['cv'].dropna()
        ax.hist(cv_data, bins=30, edgecolor='black', alpha=0.7, color='green')
        ax.axvline(cv_data.median(), color='red', linestyle='--', label=f"Median: {cv_data.median():.1f}%")
        ax.set_xlabel('Coefficient of Variation (%)')
        ax.set_ylabel('Number of Sites')
        ax.set_title('Variability Across Sites (CV)')
        ax.legend()
        
        # 4. Boxplot of means by PFT (if available)
        ax = axes[1, 0]
        if 'pft' in valid_df.columns and valid_df['pft'].notna().sum() > 0:
            pft_data = valid_df[valid_df['pft'].notna()]
            pft_order = pft_data.groupby('pft')['mean'].median().sort_values(ascending=False).index
            sns.boxplot(data=pft_data, x='pft', y='mean', order=pft_order, ax=ax, palette='Set2')
            ax.set_xlabel('Plant Functional Type')
            ax.set_ylabel('Site Mean Sap Flow')
            ax.set_title('Mean Sap Flow by PFT')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No PFT data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Mean Sap Flow by PFT')
        
        # 5. Sample size distribution
        ax = axes[1, 1]
        # Use appropriate observation count column
        obs_col = 'n_total_measurements' if self.use_raw_data else 'n_valid_values'
        if obs_col in valid_df.columns:
            ax.hist(valid_df[obs_col], bins=30, edgecolor='black', alpha=0.7, color='purple')
            ax.axvline(valid_df[obs_col].median(), color='red', linestyle='--', 
                       label=f"Median: {valid_df[obs_col].median():.0f}")
        ax.set_xlabel('Number of Valid Observations')
        ax.set_ylabel('Number of Sites')
        ax.set_title('Data Availability per Site')
        ax.legend()
        
        # 6. Skewness distribution
        ax = axes[1, 2]
        skew_data = valid_df['skewness'].dropna()
        ax.hist(skew_data, bins=30, edgecolor='black', alpha=0.7, color='teal')
        ax.axvline(0, color='black', linestyle='-', linewidth=2, label='Normal (0)')
        ax.axvline(skew_data.median(), color='red', linestyle='--', label=f"Median: {skew_data.median():.2f}")
        ax.set_xlabel('Skewness')
        ax.set_ylabel('Number of Sites')
        ax.set_title('Distribution Skewness Across Sites')
        ax.legend()
        
        data_type = 'Raw' if self.use_raw_data else self.time_scale.title()
        plt.suptitle(f'Sap Flow Statistics Overview ({data_type} Data)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save:
            suffix = 'raw' if self.use_raw_data else self.time_scale
            plot_path = self.output_dir / f'sap_flow_distributions_{suffix}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved distribution plot to: {plot_path}")
        
        plt.show()
        
    def plot_site_comparison(self, top_n: int = 30, sort_by: str = 'mean', save: bool = True):
        """
        Create bar plot comparing sites.
        
        Args:
            top_n: Number of sites to show
            sort_by: Column to sort by ('mean', 'max', 'n_valid_values')
            save: Whether to save the plot
        """
        if self.summary_df is None:
            raise ValueError("No analysis results. Run analyze_all_sites() first.")
        
        valid_df = self.summary_df[self.summary_df['mean'].notna()].copy()
        valid_df = valid_df.sort_values(sort_by, ascending=False).head(top_n)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Mean values with error bars (std)
        ax = axes[0]
        y_pos = np.arange(len(valid_df))
        ax.barh(y_pos, valid_df['mean'], xerr=valid_df['std'], 
                align='center', color='steelblue', alpha=0.7, capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(valid_df['site_id'], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Mean Sap Flow ± Std Dev')
        ax.set_title(f'Top {top_n} Sites by Mean Sap Flow')
        ax.grid(axis='x', alpha=0.3)
        
        # Right: Max values
        ax = axes[1]
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(valid_df)))
        ax.barh(y_pos, valid_df['max'], align='center', color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(valid_df['site_id'], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Maximum Sap Flow')
        ax.set_title(f'Top {top_n} Sites by Maximum Value')
        ax.grid(axis='x', alpha=0.3)
        
        data_type = 'Raw' if self.use_raw_data else self.time_scale.title()
        plt.suptitle(f'Site Comparison ({data_type} Data)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save:
            suffix = 'raw' if self.use_raw_data else self.time_scale
            plot_path = self.output_dir / f'site_comparison_{suffix}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved site comparison plot to: {plot_path}")
        
        plt.show()
    
    def identify_outlier_sites(self, threshold_std: float = 3.0) -> pd.DataFrame:
        """
        Identify sites with unusual statistics (potential data quality issues).
        
        Args:
            threshold_std: Number of standard deviations to consider as outlier
            
        Returns:
            DataFrame with flagged sites
        """
        if self.summary_df is None:
            raise ValueError("No analysis results. Run analyze_all_sites() first.")
        
        valid_df = self.summary_df[self.summary_df['mean'].notna()].copy()
        
        flags = []
        
        for idx, row in valid_df.iterrows():
            site_flags = []
            
            # Check for unusually high mean
            mean_z = (row['mean'] - valid_df['mean'].mean()) / valid_df['mean'].std()
            if abs(mean_z) > threshold_std:
                site_flags.append(f"unusual_mean (z={mean_z:.1f})")
            
            # Check for high percentage of zeros
            if row.get('pct_zeros', 0) > 50:
                site_flags.append(f"high_zeros ({row['pct_zeros']:.1f}%)")
            
            # Check for negative values
            if row.get('pct_negative', 0) > 5:
                site_flags.append(f"negative_values ({row['pct_negative']:.1f}%)")
            
            # Check for very high skewness
            if abs(row.get('skewness', 0)) > 5:
                site_flags.append(f"high_skewness ({row['skewness']:.1f})")
            
            # Check for very high max relative to mean
            if row['mean'] > 0:
                max_ratio = row['max'] / row['mean']
                if max_ratio > 20:
                    site_flags.append(f"extreme_max (ratio={max_ratio:.1f})")
            
            # Check for low data availability
            if row.get('missing_pct', 0) > 50:
                site_flags.append(f"high_missing ({row['missing_pct']:.1f}%)")
            
            if site_flags:
                # Determine observation count column based on data type
                obs_col = 'n_total_measurements' if self.use_raw_data else 'n_valid_values'
                flags.append({
                    'site_id': row['site_id'],
                    'n_trees': row.get('n_trees', None) if self.use_raw_data else None,
                    'mean': row['mean'],
                    'median': row['median'],
                    'max': row['max'],
                    'n_valid': row.get(obs_col, None),
                    'flags': '; '.join(site_flags)
                })
        
        flagged_df = pd.DataFrame(flags)
        
        if len(flagged_df) > 0:
            suffix = 'raw' if self.use_raw_data else self.time_scale
            flagged_path = self.output_dir / f'flagged_sites_{suffix}.csv'
            flagged_df.to_csv(flagged_path, index=False)
            logger.info(f"Identified {len(flagged_df)} sites with potential issues")
            logger.info(f"Saved flagged sites to: {flagged_path}")
        else:
            logger.info("No sites with potential issues identified")
        
        return flagged_df
    
    def plot_biome_boxplots(self, save: bool = True):
        """
        Create box plots of sap flow values grouped by biome.
        
        Args:
            save: Whether to save the plots
        """
        if self.summary_df is None:
            raise ValueError("No analysis results. Run analyze_all_sites() first.")
        
        # Filter to valid sites
        valid_df = self.summary_df[self.summary_df['error'].isna()].copy() if 'error' in self.summary_df.columns else self.summary_df.copy()
        valid_df = valid_df[~np.isinf(valid_df['mean'])]
        
        if 'biome' not in valid_df.columns or valid_df['biome'].isna().all():
            logger.warning("No biome data available for box plots")
            return
        
        # Remove NaN biomes
        valid_df = valid_df[valid_df['biome'].notna()]
        
        # Order biomes by median sap flow
        biome_order = valid_df.groupby('biome')['mean'].median().sort_values(ascending=False).index.tolist()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Color palette for biomes
        n_biomes = len(biome_order)
        colors = sns.color_palette("husl", n_biomes)
        
        # 1. Box plot of sap flow mean by biome
        ax = axes[0, 0]
        sns.boxplot(data=valid_df, x='biome', y='mean', order=biome_order, 
                    ax=ax, palette=colors, showfliers=True)
        ax.set_xlabel('Biome', fontsize=12)
        ax.set_ylabel('Site Mean Sap Flow', fontsize=12)
        ax.set_title('Sap Flow Mean by Biome', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        # Add sample size annotations
        for i, biome in enumerate(biome_order):
            n = len(valid_df[valid_df['biome'] == biome])
            ax.annotate(f'n={n}', xy=(i, valid_df[valid_df['biome'] == biome]['mean'].max()),
                       ha='center', va='bottom', fontsize=9, color='gray')
        
        # 2. Box plot of sap flow median by biome
        ax = axes[0, 1]
        sns.boxplot(data=valid_df, x='biome', y='median', order=biome_order, 
                    ax=ax, palette=colors, showfliers=True)
        ax.set_xlabel('Biome', fontsize=12)
        ax.set_ylabel('Site Median Sap Flow', fontsize=12)
        ax.set_title('Sap Flow Median by Biome', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # 3. Box plot of sapwood area by biome (if available)
        ax = axes[1, 0]
        if 'sapw_area_mean' in valid_df.columns and valid_df['sapw_area_mean'].notna().sum() > 0:
            sapw_df = valid_df[valid_df['sapw_area_mean'].notna()]
            sns.boxplot(data=sapw_df, x='biome', y='sapw_area_mean', order=biome_order, 
                        ax=ax, palette=colors, showfliers=True)
            ax.set_xlabel('Biome', fontsize=12)
            ax.set_ylabel('Mean Sapwood Area (cm²)', fontsize=12)
            ax.set_title('Sapwood Area by Biome', fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, 'No sapwood area data available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Sapwood Area by Biome', fontsize=14, fontweight='bold')
        
        # 4. Box plot of CV by biome
        ax = axes[1, 1]
        sns.boxplot(data=valid_df, x='biome', y='cv', order=biome_order, 
                    ax=ax, palette=colors, showfliers=True)
        ax.set_xlabel('Biome', fontsize=12)
        ax.set_ylabel('Coefficient of Variation (%)', fontsize=12)
        ax.set_title('Sap Flow Variability (CV) by Biome', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        data_type = 'Raw' if self.use_raw_data else self.time_scale.title()
        scale_label = self.scale.title() if hasattr(self, 'scale') else 'Sapwood'
        plt.suptitle(f'Sap Flow Statistics by Biome ({data_type} Data - {scale_label} Scale)', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            suffix = f'{self.scale}_raw' if self.use_raw_data else f'{self.scale}_{self.time_scale}'
            plot_path = self.output_dir / f'sap_flow_biome_boxplots_{suffix}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved biome box plots to: {plot_path}")
        
        plt.show()
        
        return fig
    
    def plot_biome_comparison(self, save: bool = True):
        """
        Create additional comparison plots by biome.
        
        Args:
            save: Whether to save the plots
        """
        if self.summary_df is None:
            raise ValueError("No analysis results. Run analyze_all_sites() first.")
        
        valid_df = self.summary_df[self.summary_df['error'].isna()].copy() if 'error' in self.summary_df.columns else self.summary_df.copy()
        valid_df = valid_df[~np.isinf(valid_df['mean'])]
        valid_df = valid_df[valid_df['biome'].notna()]
        
        biome_order = valid_df.groupby('biome')['mean'].median().sort_values(ascending=False).index.tolist()
        n_biomes = len(biome_order)
        colors = sns.color_palette("husl", n_biomes)
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Number of trees by biome
        ax = axes[0, 0]
        if 'n_trees' in valid_df.columns:
            sns.boxplot(data=valid_df, x='biome', y='n_trees', order=biome_order, 
                        ax=ax, palette=colors)
            ax.set_ylabel('Number of Trees', fontsize=11)
        ax.set_xlabel('Biome', fontsize=11)
        ax.set_title('Number of Trees per Site by Biome', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # 2. DBH by biome
        ax = axes[0, 1]
        if 'dbh_mean' in valid_df.columns and valid_df['dbh_mean'].notna().sum() > 0:
            sns.boxplot(data=valid_df[valid_df['dbh_mean'].notna()], 
                        x='biome', y='dbh_mean', order=biome_order, ax=ax, palette=colors)
            ax.set_ylabel('Mean DBH (cm)', fontsize=11)
        ax.set_xlabel('Biome', fontsize=11)
        ax.set_title('DBH by Biome', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # 3. Tree height by biome
        ax = axes[0, 2]
        if 'height_mean' in valid_df.columns and valid_df['height_mean'].notna().sum() > 0:
            sns.boxplot(data=valid_df[valid_df['height_mean'].notna()], 
                        x='biome', y='height_mean', order=biome_order, ax=ax, palette=colors)
            ax.set_ylabel('Mean Tree Height (m)', fontsize=11)
        ax.set_xlabel('Biome', fontsize=11)
        ax.set_title('Tree Height by Biome', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # 4. Mean Annual Temperature by biome
        ax = axes[1, 0]
        if 'mat' in valid_df.columns and valid_df['mat'].notna().sum() > 0:
            sns.boxplot(data=valid_df[valid_df['mat'].notna()], 
                        x='biome', y='mat', order=biome_order, ax=ax, palette=colors)
            ax.set_ylabel('MAT (°C)', fontsize=11)
        ax.set_xlabel('Biome', fontsize=11)
        ax.set_title('Mean Annual Temperature by Biome', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # 5. Mean Annual Precipitation by biome
        ax = axes[1, 1]
        if 'map' in valid_df.columns and valid_df['map'].notna().sum() > 0:
            sns.boxplot(data=valid_df[valid_df['map'].notna()], 
                        x='biome', y='map', order=biome_order, ax=ax, palette=colors)
            ax.set_ylabel('MAP (mm)', fontsize=11)
        ax.set_xlabel('Biome', fontsize=11)
        ax.set_title('Mean Annual Precipitation by Biome', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        # 6. Missing data % by biome
        ax = axes[1, 2]
        sns.boxplot(data=valid_df, x='biome', y='missing_pct', order=biome_order, 
                    ax=ax, palette=colors)
        ax.set_xlabel('Biome', fontsize=11)
        ax.set_ylabel('Missing Data (%)', fontsize=11)
        ax.set_title('Data Availability by Biome', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        
        data_type = 'Raw' if self.use_raw_data else self.time_scale.title()
        scale_label = self.scale.title() if hasattr(self, 'scale') else 'Sapwood'
        plt.suptitle(f'Site Characteristics by Biome ({data_type} Data - {scale_label} Scale)', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            suffix = f'{self.scale}_raw' if self.use_raw_data else f'{self.scale}_{self.time_scale}'
            plot_path = self.output_dir / f'site_characteristics_biome_{suffix}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved site characteristics plots to: {plot_path}")
        
        plt.show()
        
        return fig
    
    def plot_env_by_biome(self, save: bool = True):
        """
        Create box plots of environmental variables (ta, vpd, sw_in) grouped by biome.
        
        Args:
            save: Whether to save the plots
        """
        if self.summary_df is None:
            raise ValueError("No analysis results. Run analyze_all_sites() first.")
        
        # Filter to valid sites
        valid_df = self.summary_df[self.summary_df['error'].isna()].copy() if 'error' in self.summary_df.columns else self.summary_df.copy()
        valid_df = valid_df[~np.isinf(valid_df['mean'])]
        
        if 'biome' not in valid_df.columns or valid_df['biome'].isna().all():
            logger.warning("No biome data available for environmental plots")
            return
        
        # Check if we have environmental data
        env_cols = ['ta_mean', 'vpd_mean', 'sw_in_mean', 'rh_mean', 'ppfd_in_mean', 'ws_mean']
        available_env = [col for col in env_cols if col in valid_df.columns and valid_df[col].notna().sum() > 0]
        
        if len(available_env) == 0:
            logger.warning("No environmental data available for plots")
            return
        
        # Remove NaN biomes
        valid_df = valid_df[valid_df['biome'].notna()]
        
        # Order biomes by median sap flow
        biome_order = valid_df.groupby('biome')['mean'].median().sort_values(ascending=False).index.tolist()
        
        # Color palette for biomes
        n_biomes = len(biome_order)
        colors = sns.color_palette("husl", n_biomes)
        
        # Create figure with subplots (2 rows, 3 columns for 6 env variables)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Environmental variables to plot with their labels and units
        env_vars = [
            ('ta_mean', 'Air Temperature', '°C'),
            ('vpd_mean', 'Vapor Pressure Deficit', 'kPa'),
            ('sw_in_mean', 'Shortwave Radiation', 'W/m²'),
            ('rh_mean', 'Relative Humidity', '%'),
            ('ppfd_in_mean', 'PPFD', 'µmol/m²/s'),
            ('ws_mean', 'Wind Speed', 'm/s'),
        ]
        
        for idx, (var, label, unit) in enumerate(env_vars):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            if var in valid_df.columns and valid_df[var].notna().sum() > 0:
                plot_df = valid_df[valid_df[var].notna()]
                sns.boxplot(data=plot_df, x='biome', y=var, order=biome_order, 
                           ax=ax, hue='biome', palette=colors, legend=False)
                ax.set_ylabel(f'{label} ({unit})', fontsize=11)
                # Add sample size annotations
                for i, biome in enumerate(biome_order):
                    n = len(plot_df[plot_df['biome'] == biome])
                    if n > 0:
                        y_max = plot_df[plot_df['biome'] == biome][var].max()
                        ax.annotate(f'n={n}', xy=(i, y_max), ha='center', va='bottom', fontsize=8, color='gray')
            else:
                ax.text(0.5, 0.5, f'No {label} data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=11)
            
            ax.set_xlabel('Biome', fontsize=11)
            ax.set_title(f'{label} by Biome', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
        
        data_type = 'Raw' if self.use_raw_data else self.time_scale.title()
        scale_label = self.scale.title() if hasattr(self, 'scale') else 'Sapwood'
        plt.suptitle(f'Environmental Variables by Biome ({data_type} Data - {scale_label} Scale)', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            suffix = f'{self.scale}_raw' if self.use_raw_data else f'{self.scale}_{self.time_scale}'
            plot_path = self.output_dir / f'env_variables_biome_{suffix}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved environmental variables plot to: {plot_path}")
        
        plt.show()
        
        return fig
    
    def plot_env_vs_sapflow(self, save: bool = True):
        """
        Create scatter plots of sap flow vs environmental variables by biome.
        
        Args:
            save: Whether to save the plots
        """
        if self.summary_df is None:
            raise ValueError("No analysis results. Run analyze_all_sites() first.")
        
        valid_df = self.summary_df[self.summary_df['error'].isna()].copy() if 'error' in self.summary_df.columns else self.summary_df.copy()
        valid_df = valid_df[~np.isinf(valid_df['mean'])]
        valid_df = valid_df[valid_df['biome'].notna()]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        env_vars = [
            ('ta_mean', 'Air Temperature (°C)'),
            ('vpd_mean', 'VPD (kPa)'),
            ('sw_in_mean', 'Shortwave Radiation (W/m²)'),
            ('rh_mean', 'Relative Humidity (%)'),
            ('mat', 'Mean Annual Temperature (°C)'),
            ('map', 'Mean Annual Precipitation (mm)'),
        ]
        
        # Get unique biomes for coloring
        biomes = valid_df['biome'].unique()
        colors = sns.color_palette("husl", len(biomes))
        biome_colors = dict(zip(biomes, colors))
        
        for idx, (var, label) in enumerate(env_vars):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            if var in valid_df.columns and valid_df[var].notna().sum() > 0:
                plot_df = valid_df[valid_df[var].notna()]
                for biome in biomes:
                    biome_data = plot_df[plot_df['biome'] == biome]
                    if len(biome_data) > 0:
                        ax.scatter(biome_data[var], biome_data['mean'], 
                                  label=biome, alpha=0.7, color=biome_colors[biome], s=50)
                ax.set_xlabel(label, fontsize=11)
                ax.set_ylabel('Mean Sap Flow', fontsize=11)
                ax.set_title(f'Sap Flow vs {label.split("(")[0].strip()}', fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f'No {label} data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=11)
        
        # Add legend to the last subplot
        handles, labels = axes[1, 2].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), 
                      title='Biome', fontsize=9)
        
        data_type = 'Raw' if self.use_raw_data else self.time_scale.title()
        scale_label = self.scale.title() if hasattr(self, 'scale') else 'Sapwood'
        plt.suptitle(f'Sap Flow vs Environmental Variables ({data_type} Data - {scale_label} Scale)', 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save:
            suffix = f'{self.scale}_raw' if self.use_raw_data else f'{self.scale}_{self.time_scale}'
            plot_path = self.output_dir / f'sapflow_vs_env_{suffix}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved sap flow vs env plot to: {plot_path}")
        
        plt.show()
        
        return fig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze sap flow statistics for all sites'
    )
    parser.add_argument(
        '--time-scale', type=str, default='daily',
        choices=['daily', 'hourly'],
        help='Time scale of data to analyze (only used when --use-merged-data is set)'
    )
    parser.add_argument(
        '--target-column', type=str, default='sapflow_mean',
        help='Name of the sap flow column to analyze (only used when --use-merged-data is set)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--data-dir', type=str, default=None,
        help='Directory containing site data files'
    )
    parser.add_argument(
        '--max-sites', type=int, default=None,
        help='Maximum number of sites to process (for testing)'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--identify-outliers', action='store_true',
        help='Identify sites with potential data quality issues'
    )
    parser.add_argument(
        '--use-merged-data', action='store_true',
        help='Use merged/processed data instead of raw SAPFLUXNET files. '
             'By default, uses raw data from data/raw/0.1.5/0.1.5/csv/sapwood'
    )
    parser.add_argument(
        '--use-outliers-removed', action='store_true',
        help='Use outlier-removed data from outputs/processed_data/sapwood/sap/outliers_removed. '
             'This takes precedence over --use-merged-data if both are specified.'
    )
    parser.add_argument(
        '--scale', type=str, default='sapwood',
        choices=['sapwood', 'plant'],
        help='Scale of raw data to use: sapwood or plant (default: sapwood)'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Determine data source
    use_outliers_removed = args.use_outliers_removed
    use_raw_data = not args.use_merged_data and not use_outliers_removed
    
    logger.info("=" * 60)
    logger.info("SAP FLOW STATISTICS ANALYZER")
    logger.info("=" * 60)
    
    if use_outliers_removed:
        logger.info("Using OUTLIERS-REMOVED data")
    elif use_raw_data:
        logger.info("Using RAW SAPFLUXNET data")
    else:
        logger.info("Using MERGED data")
        logger.info(f"Time scale: {args.time_scale}")
        logger.info(f"Target column: {args.target_column}")
    
    # Initialize analyzer
    analyzer = SapFlowStatisticsAnalyzer(
        data_dir=args.data_dir,
        time_scale=args.time_scale,
        target_column=args.target_column,
        output_dir=args.output_dir,
        use_raw_data=use_raw_data,
        use_outliers_removed=use_outliers_removed,
        scale=args.scale
    )
    
    logger.info(f"Scale: {args.scale}")
    logger.info(f"Data directory: {analyzer.data_dir}")
    
    # Run analysis
    logger.info("\nAnalyzing all sites...")
    summary_df = analyzer.analyze_all_sites(max_sites=args.max_sites)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 60)
    
    summary = analyzer.get_summary_statistics()
    for key, value in summary.items():
        if not isinstance(value, dict):
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
    
    # Save results
    logger.info("\nSaving results...")
    analyzer.save_results()
    
    # Generate plots if requested
    if args.plot:
        logger.info("\nGenerating plots...")
        analyzer.plot_distributions()
        analyzer.plot_site_comparison()
        analyzer.plot_biome_boxplots()
        analyzer.plot_biome_comparison()
        analyzer.plot_env_by_biome()
        analyzer.plot_env_vs_sapflow()
    
    # Identify outlier sites if requested
    if args.identify_outliers:
        logger.info("\nIdentifying outlier sites...")
        flagged = analyzer.identify_outlier_sites()
        if len(flagged) > 0:
            print("\nFlagged Sites:")
            print(flagged.to_string(index=False))
    
    logger.info("\n" + "=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    
    return summary_df


if __name__ == '__main__':
    main()
