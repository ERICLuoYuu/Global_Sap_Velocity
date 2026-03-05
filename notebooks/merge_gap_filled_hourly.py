"""
Merge Sap Flow and Environmental Data with Scientific Growing Season Filtering

This script merges sap flow velocity data with environmental data and applies
scientifically validated growing season filters.

Growing season detection methods implemented:
1. GSI (Growing Season Index) - Jolly et al. (2005) Global Change Biology
2. Thermal Growing Season - WMO definition
3. Phenology-based - Following MODIS phenology algorithms  
4. Combined multi-factor approach (recommended)

Usage:
    python merge_growing_season_scientific.py
"""

import pandas as pd
from pathlib import Path
import numpy as np
from typing import Optional, Dict, List
from path_config import PathConfig, get_default_paths
from growing_season_detector import (
    GrowingSeasonDetector, 
    filter_growing_season_scientific,
    get_growing_season_mask
)

paths = get_default_paths()


def merge_sap_env_data_site(output_file: Path,
                            growing_season_only: bool = True,
                            growing_season_method: str = 'combined',
                            gsi_threshold: float = 0.5,
                            save_diagnostics: bool = True,
                            skip_sites: Optional[List[str]] = None,
                            skip_biomes: Optional[List[str]] = None,
                            skip_pfts: Optional[List[str]] = None,
                            skip_tropical: bool = False,
                            tropical_lat_threshold: float = 23.5) -> pd.DataFrame:
    """
    Merge sap flow and environmental data with optional growing season filtering.
    
    Parameters:
    -----------
    output_file : Path
        Output directory path
    growing_season_only : bool
        If True, filter data to growing season only (default: True)
    growing_season_method : str
        Scientific method for growing season detection:
        - 'gsi': Growing Season Index (Jolly et al., 2005)
          Based on temperature, VPD, and photoperiod
        - 'thermal': Temperature-based (WMO definition)
          Period with 5+ consecutive days > 5°C
        - 'phenology': LAI-based phenology detection
          Based on seasonal LAI dynamics
        - 'combined': Multi-factor approach (recommended)
          Integrates GSI + phenology + moisture constraints
    gsi_threshold : float
        GSI threshold for growing season (default: 0.5)
        Only used for 'gsi' and 'combined' methods
    save_diagnostics : bool
        If True, save diagnostic plots and statistics
    skip_sites : List[str], optional
        List of site names to skip from growing season filtering.
        These sites will retain all data without filtering.
        Example: ['US-Ha1', 'AU-Tum']
    skip_biomes : List[str], optional
        List of biome types to skip from filtering.
        Example: ['Tropical rainforest', 'Tropical seasonal forest']
    skip_pfts : List[str], optional
        List of Plant Functional Types (IGBP) to skip.
        Example: ['Evergreen Broadleaf Forests', 'Woody Savannas']
    skip_tropical : bool
        If True, automatically skip sites within tropical latitudes
        (default: False)
    tropical_lat_threshold : float
        Latitude threshold for tropical classification (default: 23.5°)
        Sites with |latitude| < threshold are considered tropical
        
    Returns:
    --------
    pd.DataFrame
        Merged and filtered data
    """
    
    # Initialize skip lists if None
    skip_sites = skip_sites or []
    skip_biomes = skip_biomes or []
    skip_pfts = skip_pfts or []
    
    def should_skip_site(site_name: str, biome: str, pft: str, lat: float) -> tuple:
        """
        Check if a site should be skipped from growing season filtering.
        
        Returns:
        --------
        tuple: (should_skip: bool, reason: str)
        """
        # Check explicit site skip list
        if site_name in skip_sites:
            return True, f"Site in skip list"
        
        # Check biome skip list
        if biome in skip_biomes:
            return True, f"Biome '{biome}' in skip list"
        
        # Check PFT skip list
        if pft in skip_pfts:
            return True, f"PFT '{pft}' in skip list"
        
        # Check tropical latitude
        if skip_tropical and abs(lat) < tropical_lat_threshold:
            return True, f"Tropical site (|lat|={abs(lat):.1f}° < {tropical_lat_threshold}°)"
        
        return False, ""
    
    # Print skip configuration
    if growing_season_only:
        print("\n" + "="*60)
        print("GROWING SEASON FILTER CONFIGURATION")
        print("="*60)
        print(f"Method: {growing_season_method}")
        print(f"GSI threshold: {gsi_threshold}")
        if skip_sites:
            print(f"Skip sites: {skip_sites}")
        if skip_biomes:
            print(f"Skip biomes: {skip_biomes}")
        if skip_pfts:
            print(f"Skip PFTs: {skip_pfts}")
        if skip_tropical:
            print(f"Skip tropical: Yes (|lat| < {tropical_lat_threshold}°)")
        print("="*60 + "\n")
    
    if not output_file.exists():
        output_file.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store dataframes by biome
    biome_merged_data = {}
    plant_merged_data = {}
    all_pft_types = set()

    # Load reference data
    site_info = pd.read_csv(paths.env_extracted_data_path)
    site_biome_mapping = {}
    
    era5_data = pd.read_csv(paths.era5_discrete_data_path)
    era5_data['prcip/PET'] = era5_data['total_precipitation_hourly'] / era5_data['potential_evaporation_hourly']
    lai_data = pd.read_csv(paths.globmap_lai_data_path)
    pft_data = pd.read_csv(paths.pft_data_path)
    
    # PFT mapping
    pft_mapping = {
        1: 'Evergreen Needleleaf Forests',
        2: 'Evergreen Broadleaf Forests',
        3: 'Deciduous Needleleaf Forests',
        4: 'Deciduous Broadleaf Forests',
        5: 'Mixed Forests',
        6: 'Closed Shrublands',
        7: 'Open Shrublands',
        8: 'Woody Savannas',
        9: 'Savannas',
        10: 'Grasslands',
        11: 'Permanent Wetlands',
        12: 'Cropland',
        13: 'Urban and Built-up Lands',
        14: 'Cropland/Natural Vegetation Mosaics',
        15: 'Permanent Snow and Ice',
        16: 'Areas with less than 10% vegetation',
        17: 'Water Bodies'
    }
    pft_data['pft'] = pft_data['landcover_type'].map(pft_mapping)

    # Standardize timestamp columns
    for df in [pft_data, lai_data, era5_data]:
        df.rename(columns={'timestamp': 'TIMESTAMP', 'datetime': 'TIMESTAMP'}, inplace=True)
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True)
    
    pft_data = pft_data.sort_values('TIMESTAMP')
    lai_data = lai_data.sort_values('TIMESTAMP')
    era5_data = era5_data.sort_values('TIMESTAMP')
    
    # Track growing season statistics
    growing_season_stats = []
    growing_season_components = {}  # Store component data for diagnostics
    count = 0
    
    for sap_data_file in paths.sap_outliers_removed_dir.glob("*.csv"):
        try:
            # Parse location and plant type
            parts = sap_data_file.stem.split("_")
            location_type = '_'.join(parts[:-4])
            
            # Load data files
            env_data_file = paths.env_outliers_removed_dir / f"{location_type}_env_data_outliers_removed.csv"
            biome_data_file = paths.raw_csv_dir / f'{location_type}_site_md.csv'
            
            try:
                biome_info = pd.read_csv(biome_data_file)
                biome_type = biome_info['si_biome'][0]
                igbp_type = biome_info['si_igbp'][0]
                lat = biome_info['si_lat'][0]
                lon = biome_info['si_long'][0]
                
                # Extract site characteristics
                site_row = site_info[site_info['site_name'] == location_type]
                ele = site_row['elevation_m'].values[0]
                slope = site_row['slope_deg'].values[0]
                aspect_sin = site_row['aspect_sin'].values[0]
                aspect_cos = site_row['aspect_cos'].values[0]
                mean_annual_temp = site_row['bio1'].values[0]
                mean_annual_precip = site_row['bio12'].values[0]
                temp_seasonality = site_row['bio4'].values[0]
                precip_seasonality = site_row['bio15'].values[0]
                canopy_height = site_row['canopy_height_m'].values[0]
                
                site_biome_mapping[location_type] = biome_type
                all_pft_types.add(igbp_type)
                
            except Exception as be:
                print(f"Error reading site info data for {location_type}: {str(be)}")
                continue
            
            env_data = pd.read_csv(env_data_file, parse_dates=['TIMESTAMP'])
            
            count += 1
            print(f"\n{'='*60}")
            print(f"Processing {location_type} (Biome: {biome_type}, Lat: {lat:.2f})")
            print(f"Site {count}")
            print(f"{'='*60}")

            sap_data = pd.read_csv(sap_data_file, parse_dates=['TIMESTAMP'])
            sap_data.drop(columns=[col for col in sap_data.columns if 'Unnamed' in col], inplace=True)
            
            env_data.set_index('TIMESTAMP', inplace=True)
            
            # Calculate site-level sap velocity
            col_names = [col for col in sap_data.columns 
                        if col not in ['solar_TIMESTAMP', 'TIMESTAMP', 'TIMESTAMP_LOCAL']]
            sap_data['site'] = sap_data[col_names].mean(axis=1)
            
            plant_sap_data = pd.DataFrame({
                'TIMESTAMP': sap_data['TIMESTAMP'],
                'TIMESTAMP_LOCAL': sap_data.get('TIMESTAMP_LOCAL'),
                'sap_velocity': sap_data['site'],
            })
            plant_sap_data.set_index('TIMESTAMP', inplace=True)
            
            # Merge sap and environmental data
            df = pd.merge(plant_sap_data, env_data, left_index=True, right_index=True)
            
            if ele is None or np.isnan(ele):
                print(f"Warning: Missing elevation data for {location_type}")

            # Resample to hourly
            sum_cols = [col for col in ['precip'] if col in df.columns]
            mean_cols = [col for col in ['vpd', 'ta', 'ws', 'sw_in', 'rh', 
                                         'swc_shallow', 'swc_deep', 'netrad', 
                                         'ext_rad', 'ppfd_in', 'sap_velocity'] 
                        if col in df.columns]
            first_cols = [col for col in df.columns if 'solar_TIMESTAMP' in col]

            agg_dict = {
                **{col: 'sum' for col in sum_cols},
                **{col: 'mean' for col in mean_cols},
                **{col: 'first' for col in first_cols}
            }

            df = df.resample('h').agg(agg_dict)

            if 'solar_TIMESTAMP_y' in df.columns:
                df.rename(columns={'solar_TIMESTAMP_y': 'solar_TIMESTAMP'}, inplace=True)

            # Add site metadata
            df['biome'] = biome_type
            df['pft'] = igbp_type
            df['site_name'] = location_type
            df['latitude'] = lat
            df['longitude'] = lon
            df['elevation'] = ele
            df['slope'] = slope
            df['aspect_sin'] = aspect_sin
            df['aspect_cos'] = aspect_cos
            df['mean_annual_temp'] = mean_annual_temp
            df['mean_annual_precip'] = mean_annual_precip
            df['temp_seasonality'] = temp_seasonality
            df['precip_seasonality'] = precip_seasonality
            df['canopy_height'] = canopy_height
            
            df = df.reset_index()

            # Merge with LAI and ERA5 data
            df = pd.merge_asof(df, lai_data, on='TIMESTAMP', by='site_name', direction='nearest')
            df = pd.merge_asof(df, era5_data, on='TIMESTAMP', by='site_name', direction='nearest')
            
            print(f"After merging - shape: {df.shape}")
            
            # ============================================================
            # SCIENTIFIC GROWING SEASON FILTER
            # ============================================================
            original_rows = len(df)
            
            if growing_season_only:
                # Check if this site should be skipped
                skip_filter, skip_reason = should_skip_site(
                    site_name=location_type,
                    biome=biome_type,
                    pft=igbp_type,
                    lat=lat
                )
                
                if skip_filter:
                    print(f"\n*** SKIPPING growing season filter: {skip_reason} ***")
                    growing_season_stats.append({
                        'site': location_type,
                        'biome': biome_type,
                        'pft': igbp_type,
                        'latitude': lat,
                        'hemisphere': 'Northern' if lat >= 0 else 'Southern',
                        'original_rows': original_rows,
                        'filtered_rows': original_rows,
                        'pct_retained': 100.0,
                        'method': 'skipped',
                        'skip_reason': skip_reason
                    })
                else:
                    print(f"\nApplying {growing_season_method.upper()} growing season filter...")
                    
                    # Determine which columns are available for filtering
                    temp_col = 'ta' if 'ta' in df.columns else None
                    vpd_col = 'vpd' if 'vpd' in df.columns else None
                    lai_col = 'LAI' if 'LAI' in df.columns else None
                    swc_col = 'swc_shallow' if 'swc_shallow' in df.columns else None
                    
                    available = [f"temp={temp_col}", f"vpd={vpd_col}", 
                                f"lai={lai_col}", f"swc={swc_col}"]
                    print(f"Available columns: {', '.join(available)}")
                    
                    # Check minimum requirements
                    if temp_col is None or vpd_col is None:
                        print("Warning: Missing required columns (ta, vpd). Skipping growing season filter.")
                        growing_season_stats.append({
                            'site': location_type,
                            'biome': biome_type,
                            'latitude': lat,
                            'hemisphere': 'Northern' if lat >= 0 else 'Southern',
                            'original_rows': original_rows,
                            'filtered_rows': original_rows,
                            'pct_retained': 100.0,
                            'method': 'skipped_missing_data'
                        })
                    else:
                        try:
                            # Apply scientific growing season filter
                            df_filtered, components = filter_growing_season_scientific(
                                df=df,
                                latitude=lat,
                                method=growing_season_method,
                                timestamp_col='TIMESTAMP',
                                temp_col=temp_col,
                                vpd_col=vpd_col,
                                lai_col=lai_col,  # Will be None if not available
                                swc_col=swc_col,  # Will be None if not available
                                gsi_threshold=gsi_threshold,
                                return_components=True
                            )
                            
                            df = df_filtered
                            filtered_rows = len(df)
                            pct_retained = (filtered_rows / original_rows * 100) if original_rows > 0 else 0
                            
                            print(f"Growing season filter result: {original_rows} -> {filtered_rows} rows "
                                  f"({pct_retained:.1f}% retained)")
                            
                            # Store statistics
                            stat_entry = {
                                'site': location_type,
                                'biome': biome_type,
                                'latitude': lat,
                                'longitude': lon,
                                'hemisphere': 'Northern' if lat >= 0 else 'Southern',
                                'original_rows': original_rows,
                                'filtered_rows': filtered_rows,
                                'pct_retained': pct_retained,
                                'method': growing_season_method,
                                'gsi_threshold': gsi_threshold,
                                'had_lai': lai_col is not None,
                                'had_swc': swc_col is not None
                            }
                            
                            # Add component statistics if available
                            if 'gsi' in components.columns:
                                stat_entry['mean_gsi'] = components['gsi'].mean()
                                stat_entry['median_gsi'] = components['gsi'].median()
                            
                            growing_season_stats.append(stat_entry)
                            growing_season_components[location_type] = components
                            
                        except Exception as gs_error:
                            print(f"Warning: Growing season filter failed: {gs_error}")
                            print("Falling back to thermal method...")
                            
                            # Fallback to simple thermal method
                            if 'ta' in df.columns:
                                df = df[df['ta'] >= 5.0]
                                filtered_rows = len(df)
                                pct_retained = (filtered_rows / original_rows * 100) if original_rows > 0 else 0
                                print(f"Thermal fallback: {original_rows} -> {filtered_rows} rows")
                                
                                growing_season_stats.append({
                                    'site': location_type,
                                    'biome': biome_type,
                                    'latitude': lat,
                                    'hemisphere': 'Northern' if lat >= 0 else 'Southern',
                                    'original_rows': original_rows,
                                    'filtered_rows': filtered_rows,
                                    'pct_retained': pct_retained,
                                    'method': 'thermal_fallback'
                                })
            # ============================================================
            
            # Save individual site data
            df.to_csv(output_file / f'{location_type}_merged.csv', index=True)
            
            plant_merged_data[location_type] = df
            
            if biome_type not in biome_merged_data:
                biome_merged_data[biome_type] = []
            biome_merged_data[biome_type].append(df)
            
        except Exception as e:
            print(f"Error processing {sap_data_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total sites processed: {count}")
    print(f"PFT types encountered: {all_pft_types}")
    print(f"{'='*60}")

    # Save growing season diagnostics
    if growing_season_only and growing_season_stats and save_diagnostics:
        _save_growing_season_diagnostics(
            output_file, 
            growing_season_stats, 
            growing_season_method
        )

    # Concatenate by biome and save
    if biome_merged_data:
        biome_output_dir = output_file / 'by_biome'
        biome_output_dir.mkdir(parents=True, exist_ok=True)
        
        all_biome_dfs = []
        for biome, dfs in biome_merged_data.items():
            if dfs:
                biome_df = pd.concat(dfs)
                safe_biome_name = ''.join(c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in biome)
                safe_biome_name = safe_biome_name.replace(' ', '_')
                
                biome_df.to_csv(biome_output_dir / f'{safe_biome_name}_merged_data.csv')
                all_biome_dfs.append(biome_df)
                print(f"Saved merged data for biome: {biome} with {len(dfs)} sites")
        
        if all_biome_dfs:
            all_data = pd.concat(all_biome_dfs)
            all_data.to_csv(output_file / 'all_biomes_merged_data.csv')
            
            biome_summary = pd.DataFrame({
                'site': list(site_biome_mapping.keys()),
                'biome': list(site_biome_mapping.values())
            })
            biome_summary.to_csv(output_file / 'site_biome_mapping.csv', index=False)
            
            return all_data
        else:
            raise ValueError("No data was successfully processed after biome merging")
    else:
        raise ValueError("No data was successfully processed")


def _save_growing_season_diagnostics(output_file: Path, 
                                     stats: List[Dict], 
                                     method: str):
    """Save growing season filter diagnostics and summary statistics."""
    
    diagnostics_dir = output_file / 'growing_season_diagnostics'
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed statistics
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(diagnostics_dir / 'filter_statistics.csv', index=False)
    
    # Generate summary report
    summary_lines = [
        "=" * 70,
        "GROWING SEASON FILTER SUMMARY REPORT",
        "=" * 70,
        f"\nMethod: {method}",
        f"Total sites processed: {len(stats_df)}",
    ]
    
    # Report on skipped sites
    if 'method' in stats_df.columns:
        skipped_df = stats_df[stats_df['method'] == 'skipped']
        filtered_df = stats_df[stats_df['method'] != 'skipped']
        
        if len(skipped_df) > 0:
            summary_lines.extend([
                f"\nSkipped Sites: {len(skipped_df)}",
            ])
            if 'skip_reason' in skipped_df.columns:
                for reason in skipped_df['skip_reason'].unique():
                    count = len(skipped_df[skipped_df['skip_reason'] == reason])
                    summary_lines.append(f"  - {reason}: {count} sites")
            
            summary_lines.append(f"\nFiltered Sites: {len(filtered_df)}")
        else:
            filtered_df = stats_df
    else:
        filtered_df = stats_df
    
    # Overall statistics (only for filtered sites)
    if len(filtered_df) > 0:
        summary_lines.extend([
            f"\nOverall Statistics (filtered sites only):",
            f"  - Mean data retention: {filtered_df['pct_retained'].mean():.1f}%",
            f"  - Median data retention: {filtered_df['pct_retained'].median():.1f}%",
            f"  - Min retention: {filtered_df['pct_retained'].min():.1f}%",
            f"  - Max retention: {filtered_df['pct_retained'].max():.1f}%",
            f"\nBy Hemisphere:",
        ])
        
        for hemisphere in ['Northern', 'Southern']:
            hem_data = filtered_df[filtered_df['hemisphere'] == hemisphere]
            if len(hem_data) > 0:
                summary_lines.append(
                    f"  {hemisphere}: {len(hem_data)} sites, "
                    f"mean retention {hem_data['pct_retained'].mean():.1f}%"
                )
        
        if 'biome' in filtered_df.columns:
            summary_lines.append("\nBy Biome:")
            for biome in filtered_df['biome'].unique():
                biome_data = filtered_df[filtered_df['biome'] == biome]
                summary_lines.append(
                    f"  {biome}: {len(biome_data)} sites, "
                    f"mean retention {biome_data['pct_retained'].mean():.1f}%"
                )
        
        if 'mean_gsi' in filtered_df.columns and not filtered_df['mean_gsi'].isna().all():
            summary_lines.extend([
                f"\nGSI Statistics:",
                f"  - Mean GSI across sites: {filtered_df['mean_gsi'].mean():.3f}",
                f"  - Median GSI across sites: {filtered_df['median_gsi'].median():.3f}"
            ])
    
    summary_lines.extend([
        "\n" + "=" * 70,
        "Method Description:",
        "=" * 70
    ])
    
    method_descriptions = {
        'gsi': """
Growing Season Index (GSI) - Jolly et al. (2005)
  
GSI is calculated as the product of three limiting factors:
  - Temperature index (iTmin): Scaled 0-1 based on minimum temperature
  - VPD index (iVPD): Scaled 0-1 based on vapor pressure deficit (inverted)
  - Photoperiod index (iPhoto): Scaled 0-1 based on day length

GSI = iTmin × iVPD × iPhoto

A 21-day moving average is applied, and growing season is defined
as periods where GSI > threshold (default 0.5).

Reference: Jolly, W.M., et al. (2005). A generalized, bioclimatic index 
to predict foliar phenology in response to climate. Global Change Biology,
11(4), 619-632.
""",
        'thermal': """
Thermal Growing Season - WMO Definition

Growing season is defined as the period between:
  - Start: First occurrence of 5+ consecutive days with Tmean > 5°C
  - End: Last occurrence of 5+ consecutive days with Tmean > 5°C

This is the standard meteorological definition used by the World 
Meteorological Organization.
""",
        'phenology': """
Phenology-based Growing Season

Growing season is detected from LAI (Leaf Area Index) dynamics:
  - LAI data is smoothed using a 15-day moving average
  - Growing season = periods where LAI > baseline + 20% of amplitude

This approach follows methods used in MODIS Land Cover Dynamics products.
""",
        'combined': """
Combined Multi-factor Growing Season (Recommended)

This integrates multiple constraints:
  1. GSI (temperature, VPD, photoperiod) - primary constraint
  2. Phenology (LAI dynamics) - if LAI data available
  3. Moisture (soil water content) - if SWC data available

Growing season is only when ALL available constraints are satisfied.
This provides the most ecologically meaningful definition.
"""
    }
    
    summary_lines.append(method_descriptions.get(method, "Custom method"))
    
    # Write summary report
    with open(diagnostics_dir / 'summary_report.txt', 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"\nDiagnostics saved to: {diagnostics_dir}")
    print(f"  - filter_statistics.csv")
    print(f"  - summary_report.txt")


def main():
    output_file = paths.merged_data_root
    
    # ========================================
    # CONFIGURATION
    # ========================================
    # Choose growing season method:
    #   - 'gsi': Growing Season Index (Jolly et al., 2005)
    #   - 'thermal': WMO temperature-based definition
    #   - 'phenology': LAI-based phenology
    #   - 'combined': Multi-factor approach (RECOMMENDED)
    
    # ========================================
    # SKIP OPTIONS (optional)
    # ========================================
    # You can skip certain sites from the growing season filter.
    # These sites will retain ALL their data without filtering.
    
    # Option 1: Skip specific sites by name
    sites_to_skip = [
        'CHN_ARG_GWD',   
        'CHN_ARG_GWS',    
    ]
    
    # Option 2: Skip entire biomes
    biomes_to_skip = [
        # 'Tropical rainforest',
        # 'Tropical seasonal forest',
        # 'Tropical deciduous forest',
    ]
    
    # Option 3: Skip specific Plant Functional Types (IGBP)
    pfts_to_skip = [
        # 'Evergreen Broadleaf Forests',  # Often tropical, year-round growing
        # 'Woody Savannas',
        # 'Savannas',
    ]
    
    # Option 4: Automatically skip tropical sites
    skip_tropical_sites = False  # Set to True to skip |lat| < 23.5°
    
    # ========================================
    # RUN MERGE
    # ========================================
    merged_data = merge_sap_env_data_site(
        output_file,
        growing_season_only=True,
        growing_season_method='phenology',  # RECOMMENDED
        gsi_threshold=0.5,                 # GSI threshold (0-1)
        save_diagnostics=True,             # Save diagnostic reports
        # Skip options
        skip_sites=sites_to_skip if sites_to_skip else None,
        skip_biomes=biomes_to_skip if biomes_to_skip else None,
        skip_pfts=pfts_to_skip if pfts_to_skip else None,
        skip_tropical=skip_tropical_sites,
        tropical_lat_threshold=23.5        # Tropic of Cancer/Capricorn
    )
    
    print(f"\nData merged and saved to {output_file}")
    print(f"Total records: {len(merged_data)}")


if __name__ == "__main__":
    main()