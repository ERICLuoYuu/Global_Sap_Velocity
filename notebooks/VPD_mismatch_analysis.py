import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Create output directory if it doesn't exist
os.makedirs('./outputs/figures/mismatch_analysis', exist_ok=True)

def analyze_vpd_mismatches(era5_data, env_files, threshold=0.5, ratio_threshold=2.0):
    """
    Analyze mismatches between ERA5 and local environmental VPD measurements using ratio-based detection.
    
    Parameters:
    -----------
    era5_data : pandas.DataFrame
        ERA5 data with 'site_name' and 'vpd' columns, indexed by timestamp
    env_files : list
        List of Path objects pointing to environmental data files
    threshold : float
        Threshold for defining a mismatch in VPD values based on absolute difference (in kPa)
        (Only used for historical reference in plots - not used for detection)
    ratio_threshold : float
        Threshold for VPD ratio between sources (values > ratio_threshold or < 1/ratio_threshold are considered mismatches)
    """
    
    # Set plot style
    sns.set_style("whitegrid")
    
    # Process each site's environmental data
    for env_file in env_files:
        # Extract site name from filename
        parts = env_file.stem.split('_')
        site_name = '_'.join(parts[:-3])
        
        print(f"Processing site: {site_name}")
        
        # Read environmental data
        env_data = pd.read_csv(env_file, parse_dates=['TIMESTAMP'])
        
        # Convert column names to lowercase for consistency
        env_data.columns = [col.lower() for col in env_data.columns]
        
        # Skip files without VPD data
        if 'vpd' not in env_data.columns:
            print(f"Skipping {site_name}: No VPD data")
            continue
            
        # Skip files without precipitation or rh data
        required_columns = ['vpd', 'rh']
        if not all(col in env_data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in env_data.columns]
            print(f"Skipping {site_name}: Missing columns {missing}")
            continue
            
        # Check if precipitation data exists (might be named differently)
        precip_col = None
        for col in env_data.columns:
            if 'preci' in col.lower() or 'rain' in col.lower() or 'precip' in col.lower():
                precip_col = col
                break
                
        if precip_col is None:
            print(f"Warning: No precipitation column found for {site_name}")
            
        # Check for total_precipitation_hourly in ERA5 data
        has_era5_precip = 'total_precipitation_hourly' in era5_data.columns
        
        # Process environmental data
        env_data['timestamp'] = pd.to_datetime(env_data['timestamp'])
        env_data.set_index('timestamp', inplace=True)
        
        # Filter for June 1-21 (summer period)
        env_data = env_data[
            (env_data.index.month == 6) & 
            (env_data.index.day >= 1) & 
            (env_data.index.day <= 21)
        ]
        
        # Round timestamps to nearest hour and resample
        env_data.index = env_data.index.round('H')
        if 'solar_TIMESTAMP' in env_data.columns:
            env_data = env_data.drop(columns='solar_TIMESTAMP')
        env_data = env_data.resample('H').mean(numeric_only=True)
        
        # Filter ERA5 data for this site
        era5_site = era5_data[era5_data['site_name'] == site_name].copy()
        
        if era5_site.empty:
            print(f"Skipping {site_name}: No matching ERA5 data")
            continue
            
        # Filter for June 1-21 (summer period)
        era5_site = era5_site[
            (era5_site.index.month == 6) & 
            (era5_site.index.day >= 1) & 
            (era5_site.index.day <= 21)
        ]
        
        # Find overlapping time period
        common_start = max(env_data.index.min(), era5_site.index.min())
        common_end = min(env_data.index.max(), era5_site.index.max())
        
        # Filter both datasets to this period
        env_filtered = env_data.loc[common_start:common_end]
        era5_filtered = era5_site.loc[common_start:common_end]
        
        # Get matching timestamps
        matching_times = env_filtered.index.intersection(era5_filtered.index)
        
        # Use these matching times for both datasets
        env_matched = env_filtered.loc[matching_times].copy()
        era5_matched = era5_filtered.loc[matching_times].copy()
        
        # Skip if no matching data
        if env_matched.empty or era5_matched.empty:
            print(f"Skipping {site_name}: No matching timestamps")
            continue
            
        # Calculate VPD differences and prepare for ratio-based mismatch detection
        env_matched['vpd_diff'] = abs(env_matched['vpd'] - era5_matched['vpd'])
        
        # Calculate ratio between ENV and ERA5 VPD values
        # Handle division by zero or near-zero values
        env_matched['vpd_ratio'] = np.where(
            era5_matched['vpd'] > 1e-6,  # Check for near-zero values
            env_matched['vpd'] / era5_matched['vpd'],
            np.inf  # Set to infinity if dividing by near-zero
        )
        
        # Remove NaN values that could cause regression issues
        valid_mask = ~np.isnan(env_matched['vpd']) & ~np.isnan(era5_matched['vpd'])
        if sum(valid_mask) < 2:
            print(f"Skipping {site_name}: Insufficient valid data points (less than 2)")
            continue
            
        env_matched_valid = env_matched[valid_mask]
        era5_matched_valid = era5_matched[valid_mask]
        
        # Identify mismatches using the ratio-based method only
        # A point is a mismatch if ENV/ERA5 > ratio_threshold or ENV/ERA5 < 1/ratio_threshold
        valid_ratio_mask = ~np.isnan(env_matched['vpd_ratio']) & ~np.isinf(env_matched['vpd_ratio'])
        if sum(valid_ratio_mask) > 0:
            mismatches = env_matched[
                (valid_ratio_mask) & 
                ((env_matched['vpd_ratio'] > ratio_threshold) | (env_matched['vpd_ratio'] < 1/ratio_threshold))
            ].copy()
        else:
            mismatches = pd.DataFrame()
        
        print(f"Found {len(mismatches)} mismatched points out of {len(env_matched)} total points ({len(env_matched_valid)} valid points)")
        print(f"Using ratio-based detection method with threshold {ratio_threshold} (identifying points where ENV/ERA5 > {ratio_threshold} or < {1/ratio_threshold:.2f})")
        
        if mismatches.empty:
            print(f"No significant mismatches found for {site_name}")
            continue
            
        # Create summary dataframe for visualization
        summary = pd.DataFrame({
            'timestamp': mismatches.index,
            'env_vpd': mismatches['vpd'],
            'era5_vpd': era5_matched.loc[mismatches.index, 'vpd'],
            'env_rh': mismatches['rh'],
            'era5_rh': era5_matched.loc[mismatches.index, 'rh'] if 'rh' in era5_matched.columns else np.nan
        })
        
        if precip_col:
            summary['env_precip'] = mismatches[precip_col]
            if any(col in era5_matched.columns for col in ['tp', 'precip', 'precipitation']):
                precip_era5_col = next(col for col in era5_matched.columns if col in ['tp', 'precip', 'precipitation'])
                summary['era5_precip'] = era5_matched.loc[mismatches.index, precip_era5_col]
            else:
                summary['era5_precip'] = np.nan
                
        # Calculate correlation and stats with error handling
        try:
            vpd_corr = np.corrcoef(env_matched_valid['vpd'], era5_matched_valid['vpd'])[0, 1]
            vpd_r2 = vpd_corr**2
        except Exception as e:
            print(f"Warning: Could not calculate correlation for {site_name}: {str(e)}")
            vpd_corr = np.nan
            vpd_r2 = np.nan
        
        rh_corr = np.nan
        if 'rh' in era5_matched.columns:
            try:
                # Get valid rh measurements from both sources
                valid_rh_idx = env_matched['rh'].notna() & era5_matched['rh'].notna()
                if sum(valid_rh_idx) >= 2:  # Need at least 2 points for correlation
                    rh_env_valid = env_matched.loc[valid_rh_idx, 'rh']
                    rh_era5_valid = era5_matched.loc[valid_rh_idx, 'rh']
                    
                    # Check for constant values which would cause correlation issues
                    if np.std(rh_env_valid) > 1e-10 and np.std(rh_era5_valid) > 1e-10:
                        rh_corr = np.corrcoef(rh_env_valid, rh_era5_valid)[0, 1]
                    else:
                        print(f"Warning: Cannot calculate rh correlation for {site_name} - values have near-zero standard deviation")
                else:
                    print(f"Warning: Insufficient valid rh data points for {site_name}")
            except Exception as e:
                print(f"Warning: Could not calculate rh correlation for {site_name}: {str(e)}")
        
        # Generate visualizations
        # 1. VPD comparison scatter plot with mismatches highlighted
        plt.figure(figsize=(10, 8))
        
        # Plot all points
        plt.scatter(env_matched_valid['vpd'], era5_matched_valid['vpd'], alpha=0.5, label='All points')
        
        # Highlight the ratio-based mismatches
        plt.scatter(summary['env_vpd'], summary['era5_vpd'], color='red', alpha=0.7, 
                   label=f'Ratio mismatches (ENV/ERA5 > {ratio_threshold} or < {1/ratio_threshold:.2f})')
        
        # Add reference line for y = x (perfect agreement)
        min_val = min(env_matched_valid['vpd'].min(), era5_matched_valid['vpd'].min())
        max_val = max(env_matched_valid['vpd'].max(), era5_matched_valid['vpd'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='1:1 line')
        
        # Add reference lines for ratio thresholds
        x_range = np.linspace(min_val, max_val, 100)
        plt.plot(x_range, x_range * ratio_threshold, 'r--', alpha=0.3, label=f'Ratio = {ratio_threshold}')
        plt.plot(x_range, x_range / ratio_threshold, 'r--', alpha=0.3, label=f'Ratio = {1/ratio_threshold:.2f}')
        
        # Fit line to valid points with error handling
        try:
            m, b = np.polyfit(env_matched_valid['vpd'], era5_matched_valid['vpd'], 1)
            vpd_min = env_matched_valid['vpd'].min()
            vpd_max = env_matched_valid['vpd'].max()
            vpd_range = np.linspace(vpd_min, vpd_max, 100)
            plt.plot(vpd_range, m*vpd_range + b, 'b--', alpha=0.7, label=f'Best fit (slope={m:.2f})')
        except Exception as e:
            print(f"Warning: Could not fit regression line for {site_name}: {str(e)}")
        
        # Calculate correlation and add to title
        try:
            vpd_corr = np.corrcoef(env_matched_valid['vpd'], era5_matched_valid['vpd'])[0, 1]
            vpd_r2 = vpd_corr**2
            plt.title(f"VPD Comparison for {site_name} (R² = {vpd_r2:.3f})")
        except Exception as e:
            print(f"Warning: Could not calculate correlation for {site_name}: {str(e)}")
            plt.title(f"VPD Comparison for {site_name}")
        
        plt.xlabel('ENV VPD (kPa)')
        plt.ylabel('ERA5 VPD (kPa)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.text(0.05, 0.95, 
                f"Total points: {len(env_matched)}\nMismatches: {len(mismatches)} ({len(mismatches)/len(env_matched)*100:.1f}%)", 
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        plt.savefig(f'./outputs/figures/mismatch_analysis/{site_name}_vpd_mismatches.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Time series of VPD with mismatches highlighted
        plt.figure(figsize=(12, 6))
        plt.plot(env_matched.index, env_matched['vpd'], 'b-', alpha=0.6, label='ENV VPD')
        plt.plot(era5_matched.index, era5_matched['vpd'], 'g-', alpha=0.6, label='ERA5 VPD')
        plt.scatter(mismatches.index, mismatches['vpd'], color='red', alpha=0.7, label='Mismatch points (ENV)')
        plt.scatter(mismatches.index, era5_matched.loc[mismatches.index, 'vpd'], color='orange', alpha=0.7, label='Mismatch points (ERA5)')
        
        plt.title(f"VPD Time Series for {site_name}")
        plt.xlabel('Date')
        plt.ylabel('VPD (kPa)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f'./outputs/figures/mismatch_analysis/{site_name}_vpd_timeseries.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Relative Humidity comparison for mismatches
        if 'rh' in era5_matched.columns:
            plt.figure(figsize=(10, 8))
            plt.scatter(env_matched['rh'], era5_matched['rh'], alpha=0.5, label='All points')
            plt.scatter(summary['env_rh'], summary['era5_rh'], color='red', alpha=0.7, label='VPD mismatch points')
            
            # Fit line with robust error handling
            valid_idx = ~np.isnan(env_matched['rh']) & ~np.isnan(era5_matched['rh'])
            if sum(valid_idx) > 1:
                try:
                    rh_env = env_matched.loc[valid_idx, 'rh']
                    rh_era5 = era5_matched.loc[valid_idx, 'rh']
                    
                    # Check for constant values which would cause fitting problems
                    if np.std(rh_env) > 1e-10 and np.std(rh_era5) > 1e-10:
                        m, b = np.polyfit(rh_env, rh_era5, 1)
                        rh_min = rh_env.min()
                        rh_max = rh_env.max()
                        rh_range = np.linspace(rh_min, rh_max, 100)
                        plt.plot(rh_range, m*rh_range + b, 'b--', alpha=0.7)
                    else:
                        print(f"Warning: Cannot fit line for {site_name} rh - values have near-zero standard deviation")
                except Exception as e:
                    print(f"Warning: Could not fit rh regression line for {site_name}: {str(e)}")
            
            plt.title(f"Relative Humidity Comparison for {site_name} (r = {rh_corr:.3f})")
            plt.xlabel('ENV rh (%)')
            plt.ylabel('ERA5 rh (%)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(f'./outputs/figures/mismatch_analysis/{site_name}_rh_mismatches.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Precipitation comparison if available
        precip_era5_col = None
        if 'total_precipitation_hourly' in era5_matched.columns:
            precip_era5_col = 'total_precipitation_hourly'
        elif any(col in era5_matched.columns for col in ['tp', 'precip', 'precipitation']):
            precip_era5_col = next(col for col in era5_matched.columns if col in ['tp', 'precip', 'precipitation'])
            
        if precip_col or precip_era5_col:
            plt.figure(figsize=(12, 6))
            
            # Plot environmental precipitation if available
            if precip_col:
                plt.plot(env_matched.index, env_matched[precip_col], 'b-', alpha=0.6, label='ENV Precip')
                # Mark mismatch points for environmental data
                plt.scatter(mismatches.index, mismatches[precip_col], 
                           color='red', alpha=0.7, label='ENV Mismatch points')
            
            # Plot ERA5 precipitation if available
            if precip_era5_col:
                plt.plot(era5_matched.index, era5_matched[precip_era5_col], 'g-', alpha=0.6, label='ERA5 Precip')
                # Mark mismatch points for ERA5 data
                plt.scatter(mismatches.index, era5_matched.loc[mismatches.index, precip_era5_col], 
                           color='orange', alpha=0.7, label='ERA5 Mismatch points')
            
            plt.title(f"Precipitation Time Series for {site_name}")
            plt.xlabel('Date')
            plt.ylabel('Precipitation (mm)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(f'./outputs/figures/mismatch_analysis/{site_name}_precip_timeseries.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create precipitation scatter plot if both sources are available
            if precip_col and precip_era5_col:
                plt.figure(figsize=(10, 8))
                # Filter out NaN values for the scatter plot
                valid_precip = ~np.isnan(env_matched[precip_col]) & ~np.isnan(era5_matched[precip_era5_col])
                
                if sum(valid_precip) > 0:
                    plt.scatter(env_matched.loc[valid_precip, precip_col], 
                               era5_matched.loc[valid_precip, precip_era5_col], 
                               alpha=0.5, label='All points')
                    
                    # Highlight mismatch points
                    valid_mismatch_precip = mismatches.index.intersection(env_matched.loc[valid_precip].index)
                    if len(valid_mismatch_precip) > 0:
                        plt.scatter(env_matched.loc[valid_mismatch_precip, precip_col],
                                   era5_matched.loc[valid_mismatch_precip, precip_era5_col],
                                   color='red', alpha=0.7, label='VPD mismatch points')
                    
                    # Add regression line if enough points
                    if sum(valid_precip) > 2:
                        try:
                            m, b = np.polyfit(env_matched.loc[valid_precip, precip_col], 
                                             era5_matched.loc[valid_precip, precip_era5_col], 1)
                            x_range = np.linspace(env_matched.loc[valid_precip, precip_col].min(),
                                                 env_matched.loc[valid_precip, precip_col].max(), 100)
                            plt.plot(x_range, m*x_range + b, 'b--', alpha=0.7)
                            
                            # Calculate correlation
                            precip_corr = np.corrcoef(env_matched.loc[valid_precip, precip_col],
                                                     era5_matched.loc[valid_precip, precip_era5_col])[0, 1]
                            
                            plt.title(f"Precipitation Comparison for {site_name} (r = {precip_corr:.3f})")
                        except Exception as e:
                            print(f"Warning: Could not fit precipitation regression line for {site_name}: {str(e)}")
                            plt.title(f"Precipitation Comparison for {site_name}")
                    else:
                        plt.title(f"Precipitation Comparison for {site_name}")
                        
                    plt.xlabel('ENV Precipitation (mm)')
                    plt.ylabel('ERA5 Precipitation (mm)')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.savefig(f'./outputs/figures/mismatch_analysis/{site_name}_precip_scatter.png', 
                              dpi=300, bbox_inches='tight')
                else:
                    print(f"Warning: No valid precipitation data points for {site_name}")
                plt.close()
        
        # 5. Create detailed mismatch analysis table
        mismatch_details = summary.copy()
        mismatch_details['vpd_diff'] = abs(mismatch_details['env_vpd'] - mismatch_details['era5_vpd'])
        mismatch_details['rh_diff'] = abs(mismatch_details['env_rh'] - mismatch_details['era5_rh']) if 'era5_rh' in mismatch_details else np.nan
        
        # Sort by VPD difference magnitude
        mismatch_details = mismatch_details.sort_values('vpd_diff', ascending=False)
        
        # Save to CSV
        mismatch_details.to_csv(f'./outputs/figures/mismatch_analysis/{site_name}_mismatch_details.csv')
        
        print(f"Analysis completed for {site_name}")
        
        

# Example usage
if __name__ == "__main__":
    # Load ERA5 data
    era5_data = pd.read_csv('data/raw/era5_extracted_data1.csv')
    era5_data['TIMESTAMP'] = pd.to_datetime(era5_data['TIMESTAMP'])
    era5_data.set_index('TIMESTAMP', inplace=True)
    # After loading ERA5 data (around line 420)
    era5_data['vpd'] = era5_data['vpd'] / 10  # Converting from hPa to kPa

    # Add precipitation unit conversion
    if 'total_precipitation_hourly' in era5_data.columns:
        era5_data['total_precipitation_hourly'] *= 1000  # Convert from m to mm

    # Handle alternative precipitation column names
    for col in era5_data.columns:
        if col in ['tp', 'precip', 'precipitation'] and col != 'total_precipitation_hourly':
            era5_data[col] *= 1000  # Convert from m to mm
    
    # Get environmental data files
    env_files = list(Path('./outputs/processed_data/env/filtered').glob("*_env_data_filtered.csv"))
    
    # Run analysis with ratio-based detection method
    analyze_vpd_mismatches(
        era5_data, 
        env_files, 
        threshold=0.5,         # Absolute difference threshold (only for reference in plots)
        ratio_threshold=2.0    # Ratio threshold (ENV/ERA5) for mismatch detection
    )