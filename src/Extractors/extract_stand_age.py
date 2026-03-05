"""
Extract Stand Age Data using h5py

This script extracts forest stand age from the MPI-BGC Forest Age Dataset
using h5py directly (since xarray has issues with this specific HDF5 structure).

Dataset variables:
- ForestAge_TC000: Forest age with 0% tree cover threshold
- ForestAge_TC010: Forest age with 10% tree cover threshold  
- ForestAge_TC020: Forest age with 20% tree cover threshold
- ForestAge_TC030: Forest age with 30% tree cover threshold
- TCloss_intensity: Tree cover loss intensity
- LastTimeTCloss_std: Standard deviation of last time of tree cover loss

Reference:
Besnard et al. (2021). Mapping global forest age from forest inventories, 
biomass and climate data. Earth System Science Data, 13, 4881-4896.
https://doi.org/10.17871/ForestAgeBGI.2021
"""

import os
import sys
import shutil
import tempfile
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Optional, Dict, List

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from path_config import PathConfig, get_default_paths
    USE_PATH_CONFIG = True
except ImportError:
    USE_PATH_CONFIG = False
    print("Note: path_config not found. Using manual paths.")


class MPIBGCForestAgeExtractor:
    """
    Extract forest stand age from MPI-BGC dataset using h5py.
    """
    
    # Available forest age variables (different tree cover thresholds)
    AGE_VARIABLES = {
        'TC000': 'ForestAge_TC000',  # 0% tree cover threshold
        'TC010': 'ForestAge_TC010',  # 10% tree cover threshold
        'TC020': 'ForestAge_TC020',  # 20% tree cover threshold
        'TC030': 'ForestAge_TC030',  # 30% tree cover threshold
    }
    
    def __init__(self, nc_path: str, use_temp_copy: bool = True):
        """
        Initialize the extractor.
        
        Parameters:
        -----------
        nc_path : str
            Path to the MPI-BGC forest age NetCDF/HDF5 file
        use_temp_copy : bool
            Copy to temp directory before opening (fixes Windows path issues)
        """
        self.original_path = nc_path
        self.use_temp_copy = use_temp_copy
        self.temp_path = None
        self.file_handle = None
        
        # Data arrays (loaded lazily)
        self._lats = None
        self._lons = None
        self._lat_indices = None
        self._lon_indices = None
        
        # Open the file
        self._open_file()
        self._load_coordinates()
    
    def _open_file(self):
        """Open the HDF5 file."""
        working_path = self.original_path
        
        if self.use_temp_copy:
            # Copy to temp location
            self.temp_path = os.path.join(tempfile.gettempdir(), "mpi_bgc_forest_age.nc")
            print(f"Copying file to temp location...")
            print(f"  Source: {self.original_path}")
            print(f"  Destination: {self.temp_path}")
            shutil.copy2(self.original_path, self.temp_path)
            working_path = self.temp_path
            print("  Copy complete!")
        
        print(f"\nOpening HDF5 file with h5py...")
        self.file_handle = h5py.File(working_path, 'r')
        print(f"  Available datasets: {list(self.file_handle.keys())}")
    
    def _load_coordinates(self):
        """Load latitude and longitude arrays."""
        print("\nLoading coordinates...")
        
        self._lats = self.file_handle['latitude'][:]
        self._lons = self.file_handle['longitude'][:]
        
        # Handle 1D or 2D coordinate arrays
        if len(self._lats.shape) > 1:
            # If 2D, take the first column/row
            self._lats = self._lats[:, 0] if self._lats.shape[1] == 1 else self._lats[0, :]
        if len(self._lons.shape) > 1:
            self._lons = self._lons[0, :] if self._lons.shape[0] == 1 else self._lons[:, 0]
        
        # Flatten if needed
        self._lats = np.asarray(self._lats).flatten()
        self._lons = np.asarray(self._lons).flatten()
        
        print(f"  Latitude: {len(self._lats)} values, range [{self._lats.min():.2f}, {self._lats.max():.2f}]")
        print(f"  Longitude: {len(self._lons)} values, range [{self._lons.min():.2f}, {self._lons.max():.2f}]")
        
        # Calculate resolution
        if len(self._lats) > 1:
            lat_res = np.abs(np.mean(np.diff(self._lats)))
            print(f"  Latitude resolution: ~{lat_res:.4f}° ({lat_res * 111:.2f} km)")
        if len(self._lons) > 1:
            lon_res = np.abs(np.mean(np.diff(self._lons)))
            print(f"  Longitude resolution: ~{lon_res:.4f}° ({lon_res * 111:.2f} km at equator)")
    
    def _find_nearest_index(self, arr: np.ndarray, value: float) -> int:
        """Find index of nearest value in array."""
        return int(np.abs(arr - value).argmin())
    
    def get_data_info(self, variable: str = 'TC000') -> Dict:
        """Get information about a data variable."""
        var_name = self.AGE_VARIABLES.get(variable, variable)
        
        if var_name not in self.file_handle:
            return {'error': f"Variable '{var_name}' not found"}
        
        data = self.file_handle[var_name]
        
        return {
            'name': var_name,
            'shape': data.shape,
            'dtype': str(data.dtype),
            'chunks': data.chunks,
            'compression': data.compression,
        }
    
    def extract_value(self, 
                      lon: float, 
                      lat: float, 
                      variable: str = 'TC000',
                      buffer_pixels: int = 0) -> Optional[float]:
        """
        Extract forest age value at a given location.
        
        Parameters:
        -----------
        lon : float
            Longitude in decimal degrees
        lat : float
            Latitude in decimal degrees
        variable : str
            Which tree cover threshold to use: 'TC000', 'TC010', 'TC020', 'TC030'
        buffer_pixels : int
            Number of pixels for spatial averaging (0 = nearest neighbor)
        
        Returns:
        --------
        float or None
            Forest age in years, or None if no data
        """
        var_name = self.AGE_VARIABLES.get(variable, variable)
        
        if var_name not in self.file_handle:
            print(f"Warning: Variable '{var_name}' not found")
            return None
        
        try:
            # Find nearest indices
            lat_idx = self._find_nearest_index(self._lats, lat)
            lon_idx = self._find_nearest_index(self._lons, lon)
            
            data = self.file_handle[var_name]
            
            if buffer_pixels == 0:
                # Direct extraction
                # Try both dimension orders
                try:
                    if data.shape[0] == len(self._lats):
                        value = data[lat_idx, lon_idx]
                    else:
                        value = data[lon_idx, lat_idx]
                except IndexError:
                    # Try the other order
                    value = data[lon_idx, lat_idx] if data.shape[0] == len(self._lons) else data[lat_idx, lon_idx]
            else:
                # Spatial averaging
                lat_start = max(0, lat_idx - buffer_pixels)
                lat_end = min(len(self._lats), lat_idx + buffer_pixels + 1)
                lon_start = max(0, lon_idx - buffer_pixels)
                lon_end = min(len(self._lons), lon_idx + buffer_pixels + 1)
                
                if data.shape[0] == len(self._lats):
                    window = data[lat_start:lat_end, lon_start:lon_end]
                else:
                    window = data[lon_start:lon_end, lat_start:lat_end]
                
                # Calculate mean of valid values
                valid_mask = ~np.isnan(window) & (window > 0) & (window < 500)
                if np.any(valid_mask):
                    value = np.mean(window[valid_mask])
                else:
                    return None
            
            # Convert to Python float
            if hasattr(value, 'item'):
                value = value.item()
            
            # Check for nodata values
            if np.isnan(value) or value <= 0 or value > 500:
                return None
            
            return float(value)
            
        except Exception as e:
            print(f"  Error extracting at ({lon:.4f}, {lat:.4f}): {e}")
            return None
    
    def extract_all_thresholds(self, 
                                lon: float, 
                                lat: float,
                                buffer_pixels: int = 0) -> Dict[str, Optional[float]]:
        """
        Extract forest age for all tree cover thresholds.
        
        Returns dict with keys: stand_age_TC000, stand_age_TC010, etc.
        """
        results = {}
        
        for tc_key in self.AGE_VARIABLES.keys():
            value = self.extract_value(lon, lat, variable=tc_key, buffer_pixels=buffer_pixels)
            results[f'stand_age_{tc_key}'] = value
        
        return results
    
    def close(self):
        """Close file and clean up temp file."""
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None
        
        if self.temp_path and os.path.exists(self.temp_path):
            try:
                os.remove(self.temp_path)
                print(f"\nCleaned up temp file: {self.temp_path}")
            except Exception as e:
                print(f"\nCould not remove temp file: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def extract_stand_age_for_sites(input_csv: str,
                                 output_csv: str,
                                 nc_path: str,
                                 lon_col: str = 'lon',
                                 lat_col: str = 'lat',
                                 site_col: str = 'site_name',
                                 tree_cover_threshold: str = 'TC000',
                                 extract_all_thresholds: bool = False,
                                 buffer_pixels: int = 0) -> pd.DataFrame:
    """
    Extract stand age data for all sites in a CSV file.
    
    Parameters:
    -----------
    input_csv : str
        Path to input CSV with site coordinates
    output_csv : str
        Path to save output CSV
    nc_path : str
        Path to MPI-BGC forest age NetCDF file
    lon_col : str
        Column name for longitude
    lat_col : str
        Column name for latitude
    site_col : str
        Column name for site identifier
    tree_cover_threshold : str
        Which threshold to use: 'TC000', 'TC010', 'TC020', 'TC030'
    extract_all_thresholds : bool
        If True, extract all 4 thresholds
    buffer_pixels : int
        Pixels for spatial averaging (0 = nearest)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with extracted stand age values
    """
    # Read input
    print(f"\nReading input CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Found {len(df)} sites")
    
    # Initialize extractor
    with MPIBGCForestAgeExtractor(nc_path, use_temp_copy=True) as extractor:
        
        # Print data info
        print(f"\nData variable info ({tree_cover_threshold}):")
        info = extractor.get_data_info(tree_cover_threshold)
        for k, v in info.items():
            print(f"  {k}: {v}")
        
        # Extract values
        print("\n" + "="*60)
        print("EXTRACTING STAND AGE VALUES")
        print("="*60 + "\n")
        
        results = []
        
        for idx, row in df.iterrows():
            site_name = row.get(site_col, f"Site_{idx}")
            lon = row[lon_col]
            lat = row[lat_col]
            
            # Progress update
            if (idx + 1) % 20 == 0 or idx == 0:
                print(f"Processing {idx+1}/{len(df)}: {site_name}")
            
            if extract_all_thresholds:
                # Extract all thresholds
                site_results = extractor.extract_all_thresholds(lon, lat, buffer_pixels)
            else:
                # Extract single threshold
                value = extractor.extract_value(lon, lat, tree_cover_threshold, buffer_pixels)
                site_results = {f'stand_age_{tree_cover_threshold}': value}
            
            results.append(site_results)
    
    # Merge results with original data
    results_df = pd.DataFrame(results)
    df_final = pd.concat([df, results_df], axis=1)
    
    # Save output
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    df_final.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    return df_final


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    age_cols = [col for col in df.columns if col.startswith('stand_age_')]
    
    for col in age_cols:
        print(f"\n{col}:")
        print(df[col].describe())
        missing = df[col].isna().sum()
        print(f"Missing: {missing} ({missing/len(df)*100:.1f}%)")
    
    print("="*60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    if USE_PATH_CONFIG:
        paths = get_default_paths()
        input_csv = paths.site_info_path
        output_csv = os.path.join(
            os.path.dirname(paths.terrain_attributes_data_path),
            "stand_age_data.csv"
        )
    else:
        # Manual paths - UPDATE THESE
        input_csv = r".\data\raw\0.1.5\0.1.5\csv\sapwood\site_info.csv"
        output_csv = r".\data\processed\stand_age_data.csv"
    
    # Path to your MPI-BGC forest age file
    NC_PATH = r".\data\raw\grided\stand_age\2026113115224222_BGIForestAgeMPIBGC1.0.0.nc"
    
    # Tree cover threshold to use:
    # - TC000: No minimum tree cover (most inclusive)
    # - TC010: 10% minimum tree cover
    # - TC020: 20% minimum tree cover
    # - TC030: 30% minimum tree cover (most restrictive)
    TREE_COVER_THRESHOLD = 'TC000'
    
    # Set to True to extract all 4 thresholds
    EXTRACT_ALL_THRESHOLDS = True
    
    # Buffer for spatial averaging (in pixels)
    # 0 = exact pixel, 1 = 3x3 window, 2 = 5x5 window
    BUFFER_PIXELS = 0
    
    # ========================================================================
    # RUN EXTRACTION
    # ========================================================================
    
    print("="*60)
    print("MPI-BGC FOREST AGE EXTRACTION")
    print("="*60)
    print(f"\nDataset reference:")
    print("  Besnard et al. (2021). Earth System Science Data, 13, 4881-4896")
    print("  https://doi.org/10.17871/ForestAgeBGI.2021")
    
    # Validate paths
    if not os.path.exists(NC_PATH):
        print(f"\nERROR: NetCDF file not found: {NC_PATH}")
        sys.exit(1)
    
    if not os.path.exists(input_csv):
        print(f"\nERROR: Input CSV not found: {input_csv}")
        sys.exit(1)
    
    # Run extraction
    df_final = extract_stand_age_for_sites(
        input_csv=input_csv,
        output_csv=output_csv,
        nc_path=NC_PATH,
        lon_col='lon',
        lat_col='lat',
        site_col='site_name',
        tree_cover_threshold=TREE_COVER_THRESHOLD,
        extract_all_thresholds=EXTRACT_ALL_THRESHOLDS,
        buffer_pixels=BUFFER_PIXELS
    )
    
    # Print summary
    print_summary(df_final)
    
    print(f"\n✓ Done! Extracted stand age for {len(df_final)} sites.")
    print(f"  Output saved to: {output_csv}")