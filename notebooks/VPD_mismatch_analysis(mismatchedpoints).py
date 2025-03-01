import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Create output directories if they don't exist
os.makedirs('./outputs/figures/mismatch_analysis/mismatched_points', exist_ok=True)
os.makedirs('./outputs/statistics', exist_ok=True)

def analyze_vpd_mismatches(era5_data, env_files, threshold=0.5, ratio_threshold=2.0):
    """
    Analyze mismatches between ERA5 and local environmental VPD measurements using threshold-based detection.
    
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
        Threshold for VPD ratio between sources
        (Not used in current implementation but kept for compatibility)
    """
    
    # Set plot style
    sns.set_style("whitegrid")
    
    # Create a DataFrame to store site statistics
    all_site_stats = pd.DataFrame(columns=[
        'site_name', 'total_valid_points', 'total_mismatches', 'mismatch_percentage',
        'env_low_era5_high_count', 'env_low_era5_high_percentage', 
        'era5_low_env_high_count', 'era5_low_env_high_percentage',
        'env_low_era5_high_mean_env_vpd', 'env_low_era5_high_mean_era5_vpd',
        'era5_low_env_high_mean_env_vpd', 'era5_low_env_high_mean_era5_vpd',
        'env_low_era5_high_mean_vpd_diff', 'era5_low_env_high_mean_vpd_diff',
        'env_low_era5_high_mean_env_rh', 'env_low_era5_high_mean_era5_rh',
        'era5_low_env_high_mean_env_rh', 'era5_low_env_high_mean_era5_rh'
    ])
    
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
        
        # Process environmental data - note lowercase timestamp due to column name conversion
        env_data.set_index('timestamp', inplace=True)
        
        # Filter for June 1-21 (summer period)
        env_data = env_data[
            (env_data.index.month == 6) & 
            (env_data.index.day >= 1) & 
            (env_data.index.day <= 21)
        ]
        
        # Round timestamps to nearest hour and resample
        env_data.index = env_data.index.round('H')
        if 'solar_timestamp' in env_data.columns:  # Changed from solar_TIMESTAMP to solar_timestamp
            env_data = env_data.drop(columns='solar_timestamp')
            
        # Identify any datetime columns that should be excluded from mean calculation
        datetime_cols = [col for col in env_data.columns if 
                         env_data[col].dtype == 'object' and 
                         isinstance(env_data[col].iloc[0] if len(env_data) > 0 else '', str) and
                         ('time' in col.lower() or 'date' in col.lower())]
        
        # Drop datetime columns before resampling
        if datetime_cols:
            env_data = env_data.drop(columns=datetime_cols)
        
        # Now resample with only numeric columns
        env_data = env_data.resample('H').mean()
        
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
        
        # Identify mismatches based on specific thresholds:
        # 1. ENV VPD close to 0 (< 0.3) while ERA5 VPD > 1
        # 2. ERA5 VPD close to 0 (< 0.3) while ENV VPD > 1
        valid_mask = ~np.isnan(env_matched['vpd']) & ~np.isnan(era5_matched['vpd'])
        if sum(valid_mask) > 0:
            mismatches = env_matched[
                ((env_matched['vpd'] < 0.3) & (era5_matched['vpd'] > 1)) | 
                ((era5_matched['vpd'] < 0.3) & (env_matched['vpd'] > 1))
            ].copy()
        else:
            mismatches = pd.DataFrame()
        
        total_valid_points = len(env_matched_valid)
        total_mismatches = len(mismatches)
        mismatch_percentage = (total_mismatches / total_valid_points * 100) if total_valid_points > 0 else 0
        
        print(f"Found {total_mismatches} mismatched points out of {total_valid_points} total points ({mismatch_percentage:.1f}%)")
        print(f"Using threshold-based detection: ENV VPD < 0.3 & ERA5 VPD > 1 OR ERA5 VPD < 0.3 & ENV VPD > 1")
        
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
            if any(col in era5_matched.columns for col in ['tp', 'precip', 'precipitation', 'total_precipitation_hourly']):
                precip_era5_col = next(col for col in era5_matched.columns if col in ['tp', 'precip', 'precipitation', 'total_precipitation_hourly'])
                summary['era5_precip'] = era5_matched.loc[mismatches.index, precip_era5_col]
            else:
                summary['era5_precip'] = np.nan
        
        # Calculate statistics for each type of mismatch
        # Type 1: ENV VPD < 0.3 & ERA5 VPD > 1
        env_low_era5_high = ((mismatches['vpd'] < 0.3) & (era5_matched.loc[mismatches.index, 'vpd'] > 1))
        env_low_era5_high_count = env_low_era5_high.sum()
        env_low_era5_high_percentage = (env_low_era5_high_count / total_mismatches * 100) if total_mismatches > 0 else 0
        
        # Type 2: ERA5 VPD < 0.3 & ENV VPD > 1
        era5_low_env_high = ((era5_matched.loc[mismatches.index, 'vpd'] < 0.3) & (mismatches['vpd'] > 1))
        era5_low_env_high_count = era5_low_env_high.sum()
        era5_low_env_high_percentage = (era5_low_env_high_count / total_mismatches * 100) if total_mismatches > 0 else 0
        
        # Calculate mean values for each type of mismatch
        env_low_era5_high_mean_env_vpd = mismatches.loc[env_low_era5_high, 'vpd'].mean() if env_low_era5_high_count > 0 else np.nan
        env_low_era5_high_mean_era5_vpd = era5_matched.loc[mismatches.loc[env_low_era5_high].index, 'vpd'].mean() if env_low_era5_high_count > 0 else np.nan
        env_low_era5_high_mean_vpd_diff = abs(env_low_era5_high_mean_env_vpd - env_low_era5_high_mean_era5_vpd) if env_low_era5_high_count > 0 else np.nan
        
        era5_low_env_high_mean_env_vpd = mismatches.loc[era5_low_env_high, 'vpd'].mean() if era5_low_env_high_count > 0 else np.nan
        era5_low_env_high_mean_era5_vpd = era5_matched.loc[mismatches.loc[era5_low_env_high].index, 'vpd'].mean() if era5_low_env_high_count > 0 else np.nan
        era5_low_env_high_mean_vpd_diff = abs(era5_low_env_high_mean_env_vpd - era5_low_env_high_mean_era5_vpd) if era5_low_env_high_count > 0 else np.nan
        
        # Calculate mean RH values for each type of mismatch
        env_low_era5_high_mean_env_rh = np.nan
        env_low_era5_high_mean_era5_rh = np.nan
        era5_low_env_high_mean_env_rh = np.nan
        era5_low_env_high_mean_era5_rh = np.nan
        
        if 'rh' in era5_matched.columns:
            env_low_era5_high_mean_env_rh = mismatches.loc[env_low_era5_high, 'rh'].mean() if env_low_era5_high_count > 0 else np.nan
            env_low_era5_high_mean_era5_rh = era5_matched.loc[mismatches.loc[env_low_era5_high].index, 'rh'].mean() if env_low_era5_high_count > 0 else np.nan
            
            era5_low_env_high_mean_env_rh = mismatches.loc[era5_low_env_high, 'rh'].mean() if era5_low_env_high_count > 0 else np.nan
            era5_low_env_high_mean_era5_rh = era5_matched.loc[mismatches.loc[era5_low_env_high].index, 'rh'].mean() if era5_low_env_high_count > 0 else np.nan
        
        # Calculate mean precipitation values for each type of mismatch
        env_low_era5_high_mean_env_precip = np.nan
        env_low_era5_high_mean_era5_precip = np.nan
        era5_low_env_high_mean_env_precip = np.nan
        era5_low_env_high_mean_era5_precip = np.nan
        
        if precip_col and any(col in era5_matched.columns for col in ['tp', 'precip', 'precipitation', 'total_precipitation_hourly']):
            precip_era5_col = next(col for col in era5_matched.columns if col in ['tp', 'precip', 'precipitation', 'total_precipitation_hourly'])
            
            env_low_era5_high_mean_env_precip = mismatches.loc[env_low_era5_high, precip_col].mean() if env_low_era5_high_count > 0 else np.nan
            env_low_era5_high_mean_era5_precip = era5_matched.loc[mismatches.loc[env_low_era5_high].index, precip_era5_col].mean() * 1000 if env_low_era5_high_count > 0 else np.nan
            
            era5_low_env_high_mean_env_precip = mismatches.loc[era5_low_env_high, precip_col].mean() if era5_low_env_high_count > 0 else np.nan
            era5_low_env_high_mean_era5_precip = era5_matched.loc[mismatches.loc[era5_low_env_high].index, precip_era5_col].mean() * 1000 if era5_low_env_high_count > 0 else np.nan
        
        # Print statistics to console
        print("\nMismatch Statistics:")
        print(f"Total valid points: {total_valid_points}")
        print(f"Total mismatches: {total_mismatches} ({mismatch_percentage:.1f}%)")
        print(f"\nType 1 - ENV VPD < 0.3 & ERA5 VPD > 1:")
        print(f"  Count: {env_low_era5_high_count} ({env_low_era5_high_percentage:.1f}% of mismatches)")
        print(f"  Mean ENV VPD: {env_low_era5_high_mean_env_vpd:.3f} kPa")
        print(f"  Mean ERA5 VPD: {env_low_era5_high_mean_era5_vpd:.3f} kPa")
        print(f"  Mean VPD difference: {env_low_era5_high_mean_vpd_diff:.3f} kPa")
        if not np.isnan(env_low_era5_high_mean_env_rh) and not np.isnan(env_low_era5_high_mean_era5_rh):
            print(f"  Mean ENV RH: {env_low_era5_high_mean_env_rh:.1f}%")
            print(f"  Mean ERA5 RH: {env_low_era5_high_mean_era5_rh:.1f}%")
        if not np.isnan(env_low_era5_high_mean_env_precip) and not np.isnan(env_low_era5_high_mean_era5_precip):
            print(f"  Mean ENV Precip: {env_low_era5_high_mean_env_precip:.2f} mm")
            print(f"  Mean ERA5 Precip: {env_low_era5_high_mean_era5_precip:.2f} mm")
        
        print(f"\nType 2 - ERA5 VPD < 0.3 & ENV VPD > 1:")
        print(f"  Count: {era5_low_env_high_count} ({era5_low_env_high_percentage:.1f}% of mismatches)")
        print(f"  Mean ENV VPD: {era5_low_env_high_mean_env_vpd:.3f} kPa")
        print(f"  Mean ERA5 VPD: {era5_low_env_high_mean_era5_vpd:.3f} kPa")
        print(f"  Mean VPD difference: {era5_low_env_high_mean_vpd_diff:.3f} kPa")
        if not np.isnan(era5_low_env_high_mean_env_rh) and not np.isnan(era5_low_env_high_mean_era5_rh):
            print(f"  Mean ENV RH: {era5_low_env_high_mean_env_rh:.1f}%")
            print(f"  Mean ERA5 RH: {era5_low_env_high_mean_era5_rh:.1f}%")
        if not np.isnan(era5_low_env_high_mean_env_precip) and not np.isnan(era5_low_env_high_mean_era5_precip):
            print(f"  Mean ENV Precip: {era5_low_env_high_mean_env_precip:.2f} mm")
            print(f"  Mean ERA5 Precip: {era5_low_env_high_mean_era5_precip:.2f} mm")
        
        # Temporal distribution of mismatches
        print("\nTemporal Distribution of Mismatches:")
        early_june = (mismatches.index.day >= 1) & (mismatches.index.day <= 7)
        mid_june = (mismatches.index.day >= 8) & (mismatches.index.day <= 14)
        late_june = (mismatches.index.day >= 15) & (mismatches.index.day <= 21)
        
        early_june_count = early_june.sum()
        mid_june_count = mid_june.sum()
        late_june_count = late_june.sum()
        
        print(f"  Early June (1-7): {early_june_count} ({early_june_count/total_mismatches*100:.1f}%)")
        print(f"  Mid June (8-14): {mid_june_count} ({mid_june_count/total_mismatches*100:.1f}%)")
        print(f"  Late June (15-21): {late_june_count} ({late_june_count/total_mismatches*100:.1f}%)")
        
        # Add statistics to the site_stats DataFrame
        site_stats = pd.DataFrame({
            'site_name': [site_name],
            'total_valid_points': [total_valid_points],
            'total_mismatches': [total_mismatches],
            'mismatch_percentage': [mismatch_percentage],
            'env_low_era5_high_count': [env_low_era5_high_count],
            'env_low_era5_high_percentage': [env_low_era5_high_percentage],
            'era5_low_env_high_count': [era5_low_env_high_count],
            'era5_low_env_high_percentage': [era5_low_env_high_percentage],
            'env_low_era5_high_mean_env_vpd': [env_low_era5_high_mean_env_vpd],
            'env_low_era5_high_mean_era5_vpd': [env_low_era5_high_mean_era5_vpd],
            'era5_low_env_high_mean_env_vpd': [era5_low_env_high_mean_env_vpd],
            'era5_low_env_high_mean_era5_vpd': [era5_low_env_high_mean_era5_vpd],
            'env_low_era5_high_mean_vpd_diff': [env_low_era5_high_mean_vpd_diff],
            'era5_low_env_high_mean_vpd_diff': [era5_low_env_high_mean_vpd_diff],
            'env_low_era5_high_mean_env_rh': [env_low_era5_high_mean_env_rh],
            'env_low_era5_high_mean_era5_rh': [env_low_era5_high_mean_era5_rh],
            'era5_low_env_high_mean_env_rh': [era5_low_env_high_mean_env_rh],
            'era5_low_env_high_mean_era5_rh': [era5_low_env_high_mean_era5_rh],
            'env_low_era5_high_mean_env_precip': [env_low_era5_high_mean_env_precip],
            'env_low_era5_high_mean_era5_precip': [env_low_era5_high_mean_era5_precip],
            'era5_low_env_high_mean_env_precip': [era5_low_env_high_mean_env_precip],
            'era5_low_env_high_mean_era5_precip': [era5_low_env_high_mean_era5_precip],
            'early_june_count': [early_june_count],
            'mid_june_count': [mid_june_count],
            'late_june_count': [late_june_count]
        })
        
        # Append to the all_site_stats DataFrame
        all_site_stats = pd.concat([all_site_stats, site_stats], ignore_index=True)
        
        # Save detailed mismatch statistics to CSV
        detailed_stats = {
            'Statistic': [
                'Total valid points', 'Total mismatches', 'Mismatch percentage (%)',
                'ENV low ERA5 high count', 'ENV low ERA5 high percentage (%)',
                'ERA5 low ENV high count', 'ERA5 low ENV high percentage (%)',
                'ENV low ERA5 high mean ENV VPD (kPa)', 'ENV low ERA5 high mean ERA5 VPD (kPa)',
                'ENV low ERA5 high mean VPD difference (kPa)',
                'ERA5 low ENV high mean ENV VPD (kPa)', 'ERA5 low ENV high mean ERA5 VPD (kPa)',
                'ERA5 low ENV high mean VPD difference (kPa)',
                'ENV low ERA5 high mean ENV RH (%)', 'ENV low ERA5 high mean ERA5 RH (%)',
                'ERA5 low ENV high mean ENV RH (%)', 'ERA5 low ENV high mean ERA5 RH (%)',
                'ENV low ERA5 high mean ENV Precipitation (mm)', 'ENV low ERA5 high mean ERA5 Precipitation (mm)',
                'ERA5 low ENV high mean ENV Precipitation (mm)', 'ERA5 low ENV high mean ERA5 Precipitation (mm)',
                'Early June mismatches (1-7)', 'Mid June mismatches (8-14)', 'Late June mismatches (15-21)'
            ],
            'Value': [
                total_valid_points, total_mismatches, mismatch_percentage,
                env_low_era5_high_count, env_low_era5_high_percentage,
                era5_low_env_high_count, era5_low_env_high_percentage,
                env_low_era5_high_mean_env_vpd, env_low_era5_high_mean_era5_vpd,
                env_low_era5_high_mean_vpd_diff,
                era5_low_env_high_mean_env_vpd, era5_low_env_high_mean_era5_vpd,
                era5_low_env_high_mean_vpd_diff,
                env_low_era5_high_mean_env_rh, env_low_era5_high_mean_era5_rh,
                era5_low_env_high_mean_env_rh, era5_low_env_high_mean_era5_rh,
                env_low_era5_high_mean_env_precip, env_low_era5_high_mean_era5_precip,
                era5_low_env_high_mean_env_precip, era5_low_env_high_mean_era5_precip,
                early_june_count, mid_june_count, late_june_count
            ]
        }
        
        detailed_stats_df = pd.DataFrame(detailed_stats)
        detailed_stats_df.to_csv(f'./outputs/statistics/{site_name}_mismatch_statistics.csv', index=False)
        
        # Generate visualizations
        # 1. VPD comparison scatter plot showing only mismatches
        plt.figure(figsize=(10, 8))
        
        # Only plot mismatched points
        plt.scatter(summary['env_vpd'], summary['era5_vpd'], color='red', alpha=0.7, 
                   label='Mismatched points')
        
        # Add reference line for y = x (perfect agreement)
        # Calculate min/max values from the full dataset for proper scaling
        min_val = min(env_matched_valid['vpd'].min(), era5_matched_valid['vpd'].min())
        max_val = max(env_matched_valid['vpd'].max(), era5_matched_valid['vpd'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='1:1 line')
        
        # Add reference lines for threshold values
        x_range = np.linspace(min_val, max_val, 100)
        plt.axhline(y=1.0, color='g', linestyle='--', alpha=0.3, label='ERA5 VPD = 1.0')
        plt.axvline(x=1.0, color='b', linestyle='--', alpha=0.3, label='ENV VPD = 1.0')
        plt.axhline(y=0.3, color='g', linestyle=':', alpha=0.3, label='ERA5 VPD = 0.3')
        plt.axvline(x=0.3, color='b', linestyle=':', alpha=0.3, label='ENV VPD = 0.3')
        
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
        
        # Add statistics to the plot
        stats_text = (
            f"Total valid points: {total_valid_points}\n"
            f"Total mismatches: {total_mismatches} ({mismatch_percentage:.1f}%)\n\n"
            f"Type 1 (ENV < 0.3, ERA5 > 1): {env_low_era5_high_count} ({env_low_era5_high_percentage:.1f}%)\n"
            f"Type 2 (ERA5 < 0.3, ENV > 1): {era5_low_env_high_count} ({era5_low_env_high_percentage:.1f}%)"
        )
        
        plt.xlabel('ENV VPD (kPa)')
        plt.ylabel('ERA5 VPD (kPa)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.text(0.05, 0.95, stats_text,
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        plt.savefig(f'./outputs/figures/mismatch_analysis/mismatched_points/{site_name}_vpd_mismatches.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Create a dual-axis plot for VPD and precipitation (if precipitation data exists)
        if precip_col or precip_era5_col:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot VPD on primary y-axis (only mismatched points)
            ax1.scatter(mismatches.index, mismatches['vpd'], color='red', alpha=0.7, 
                      marker='o', s=50, label='ENV VPD')
            ax1.scatter(mismatches.index, era5_matched.loc[mismatches.index, 'vpd'], 
                      color='orange', alpha=0.7, marker='s', s=50, label='ERA5 VPD')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('VPD (kPa)', color='red')
            ax1.tick_params(axis='y', labelcolor='red')
            ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
            ax1.axhline(y=0.3, color='black', linestyle=':', alpha=0.3)
            
            # Create secondary y-axis for precipitation
            ax2 = ax1.twinx()
            
            # Plot precipitation on secondary y-axis (only mismatched points)
            if precip_col:
                ax2.scatter(mismatches.index, mismatches[precip_col], color='blue', alpha=0.5, 
                          marker='^', s=40, label=f'ENV {precip_col}')
            
            if precip_era5_col:
                era5_mismatch_precip = era5_matched.loc[mismatches.index, precip_era5_col] * 1000
                ax2.scatter(mismatches.index, era5_mismatch_precip, color='green', alpha=0.5, 
                          marker='v', s=40, label='ERA5 Precip')
            
            ax2.set_ylabel('Precipitation (mm)', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            
            # Create combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            # Add statistics to the plot
            stats_text = (
                f"Type 1 (ENV < 0.3, ERA5 > 1): {env_low_era5_high_count}\n"
                f"Type 2 (ERA5 < 0.3, ENV > 1): {era5_low_env_high_count}"
            )
            ax1.text(0.02, 0.98, stats_text,
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            plt.title(f"VPD and Precipitation at Mismatch Points for {site_name}")
            plt.tight_layout()
            plt.savefig(f'./outputs/figures/mismatch_analysis/mismatched_points/{site_name}_vpd_precip_combined.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Create a focused VPD-only plot for mismatched points with statistics
        plt.figure(figsize=(12, 6))
        
        # Create a custom marker style for each mismatch type
        env_low_era5_high = ((mismatches['vpd'] < 0.3) & (era5_matched.loc[mismatches.index, 'vpd'] > 1))
        era5_low_env_high = ((era5_matched.loc[mismatches.index, 'vpd'] < 0.3) & (mismatches['vpd'] > 1))
        
        # Plot different mismatch types with different markers
        plt.scatter(mismatches.index[env_low_era5_high], mismatches.loc[env_low_era5_high, 'vpd'], 
                   color='red', alpha=0.7, marker='o', s=60, label='ENV VPD < 0.3')
        plt.scatter(mismatches.index[env_low_era5_high], era5_matched.loc[mismatches.loc[env_low_era5_high].index, 'vpd'], 
                   color='orange', alpha=0.7, marker='o', s=60, label='ERA5 VPD > 1')
        
        plt.scatter(mismatches.index[era5_low_env_high], mismatches.loc[era5_low_env_high, 'vpd'], 
                   color='darkred', alpha=0.7, marker='s', s=60, label='ENV VPD > 1')
        plt.scatter(mismatches.index[era5_low_env_high], era5_matched.loc[mismatches.loc[era5_low_env_high].index, 'vpd'], 
                   color='darkorange', alpha=0.7, marker='s', s=60, label='ERA5 VPD < 0.3')
        
        # Add reference lines
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='VPD = 1.0 kPa')
        plt.axhline(y=0.3, color='black', linestyle=':', alpha=0.5, label='VPD = 0.3 kPa')
        
        # Add detailed statistics to the plot
        stats_text = (
            f"Type 1 (ENV < 0.3, ERA5 > 1): {env_low_era5_high_count} ({env_low_era5_high_percentage:.1f}%)\n"
            f"  Mean ENV VPD: {env_low_era5_high_mean_env_vpd:.3f} kPa\n"
            f"  Mean ERA5 VPD: {env_low_era5_high_mean_era5_vpd:.3f} kPa\n\n"
            f"Type 2 (ERA5 < 0.3, ENV > 1): {era5_low_env_high_count} ({era5_low_env_high_percentage:.1f}%)\n"
            f"  Mean ENV VPD: {era5_low_env_high_mean_env_vpd:.3f} kPa\n"
            f"  Mean ERA5 VPD: {era5_low_env_high_mean_era5_vpd:.3f} kPa"
        )
        
        plt.xlabel('Date')
        plt.ylabel('VPD (kPa)')
        plt.title(f"VPD Mismatch Analysis for {site_name}")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.text(0.02, 0.98, stats_text,
                transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        plt.tight_layout()
        plt.savefig(f'./outputs/figures/mismatch_analysis/mismatched_points/{site_name}_vpd_focused.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Combined time series plot with multiple subplots for VPD, RH, and Precipitation
        # Only showing mismatched points (no full time series)
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

        # VPD subplot - only mismatched points
        axes[0].scatter(mismatches.index, mismatches['vpd'], color='red', alpha=0.7, 
                      marker='o', s=50, label='ENV VPD at mismatch points')
        axes[0].scatter(mismatches.index, era5_matched.loc[mismatches.index, 'vpd'], 
                      color='orange', alpha=0.7, marker='s', s=50, label='ERA5 VPD at mismatch points')
        # Add reference lines for the thresholds
        axes[0].axhline(y=1.0, color='orange', linestyle='--', alpha=0.3, label='ERA5 VPD = 1.0')
        axes[0].axhline(y=0.3, color='orange', linestyle=':', alpha=0.3, label='ERA5 VPD = 0.3')  
        axes[0].axhline(y=1.0, color='red', linestyle='--', alpha=0.3, label='ENV VPD = 1.0')
        axes[0].axhline(y=0.3, color='red', linestyle=':', alpha=0.3, label='ENV VPD = 0.3')
        axes[0].set_ylabel('VPD (kPa)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='best')

        # Add mismatch type statistics
        axes[0].text(0.02, 0.98, 
                 f"Type 1: {env_low_era5_high_count} | Type 2: {era5_low_env_high_count}",
                 transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # RH subplot (if available) - only mismatched points
        if 'rh' in era5_matched.columns:
            axes[1].scatter(mismatches.index, mismatches['rh'], color='red', alpha=0.7, 
                          marker='o', s=50, label='ENV RH at mismatch points')
            axes[1].scatter(mismatches.index, era5_matched.loc[mismatches.index, 'rh'], 
                          color='orange', alpha=0.7, marker='s', s=50, label='ERA5 RH at mismatch points')
            axes[1].set_ylabel('Relative Humidity (%)')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc='best')
        else:
            axes[1].text(0.5, 0.5, 'No Relative Humidity Data Available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[1].transAxes)

        # Precipitation subplot (if available) - only mismatched points
        precip_era5_col = None
        if 'total_precipitation_hourly' in era5_matched.columns:
            precip_era5_col = 'total_precipitation_hourly'
        elif any(col in era5_matched.columns for col in ['tp', 'precip', 'precipitation']):
            precip_era5_col = next(col for col in era5_matched.columns if col in ['tp', 'precip', 'precipitation'])

        if precip_col and precip_era5_col:
            # Only show precipitation at mismatch points
            axes[2].scatter(mismatches.index, mismatches[precip_col], color='red', alpha=0.7, 
                          marker='o', s=50, label=f'ENV {precip_col} at mismatch points')
            era5_mismatch_precip = era5_matched.loc[mismatches.index, precip_era5_col] * 1000
            axes[2].scatter(mismatches.index, era5_mismatch_precip, 
                          color='orange', alpha=0.7, marker='s', s=50, label='ERA5 Precip at mismatch points')
            axes[2].set_ylabel('Precipitation (mm)')
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(loc='best')
        elif precip_col:
            axes[2].scatter(mismatches.index, mismatches[precip_col], color='red', alpha=0.7, 
                          marker='o', s=50, label=f'ENV {precip_col} at mismatch points')
            axes[2].set_ylabel('Precipitation (mm)')
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(loc='best')
        elif precip_era5_col:
            era5_mismatch_precip = era5_matched.loc[mismatches.index, precip_era5_col] * 1000
            axes[2].scatter(mismatches.index, era5_mismatch_precip, 
                          color='orange', alpha=0.7, marker='s', s=50, label='ERA5 Precip at mismatch points')
            axes[2].set_ylabel('Precipitation (mm)')
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(loc='best')
        else:
            axes[2].text(0.5, 0.5, 'No Precipitation Data Available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[2].transAxes)

        # Common x-axis label and title
        axes[2].set_xlabel('Date')
        fig.suptitle(f"Mismatched Points Analysis for {site_name}\nType 1 (ENV < 0.3, ERA5 > 1): {env_low_era5_high_count} | Type 2 (ERA5 < 0.3, ENV > 1): {era5_low_env_high_count}", fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(f'./outputs/figures/mismatch_analysis/mismatched_points/{site_name}_combined_timeseries.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Relative Humidity comparison for mismatches - only showing mismatched points
        if 'rh' in era5_matched.columns:
            plt.figure(figsize=(10, 8))
            
            # Plot different mismatch types with different markers
            plt.scatter(mismatches.loc[env_low_era5_high, 'rh'], 
                      era5_matched.loc[mismatches.loc[env_low_era5_high].index, 'rh'], 
                      color='red', alpha=0.7, marker='o', s=60, 
                      label=f'Type 1 (ENV VPD < 0.3, ERA5 VPD > 1): {env_low_era5_high_count}')
            
            plt.scatter(mismatches.loc[era5_low_env_high, 'rh'], 
                      era5_matched.loc[mismatches.loc[era5_low_env_high].index, 'rh'], 
                      color='blue', alpha=0.7, marker='s', s=60, 
                      label=f'Type 2 (ERA5 VPD < 0.3, ENV VPD > 1): {era5_low_env_high_count}')
            
            # Add reference line for y = x (perfect agreement)
            rh_min = min(mismatches['rh'].min(), era5_matched.loc[mismatches.index, 'rh'].min())
            rh_max = max(mismatches['rh'].max(), era5_matched.loc[mismatches.index, 'rh'].max())
            plt.plot([rh_min, rh_max], [rh_min, rh_max], 'k--', alpha=0.5, label='1:1 line')
            
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
            
            # Add statistics
            stats_text = ""
            if env_low_era5_high_count > 0:
                stats_text += f"Type 1 RH (mean):\n  ENV: {env_low_era5_high_mean_env_rh:.1f}%\n  ERA5: {env_low_era5_high_mean_era5_rh:.1f}%\n\n"
            if era5_low_env_high_count > 0:
                stats_text += f"Type 2 RH (mean):\n  ENV: {era5_low_env_high_mean_env_rh:.1f}%\n  ERA5: {era5_low_env_high_mean_era5_rh:.1f}%"
            
            plt.text(0.02, 0.98, stats_text,
                    transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            plt.title(f"RH at VPD Mismatch Points for {site_name}")
            plt.xlabel('ENV RH (%)')
            plt.ylabel('ERA5 RH (%)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(f'./outputs/figures/mismatch_analysis/mismatched_points/{site_name}_rh_mismatches.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 6. Precipitation comparison scatter plot - only showing mismatched points
        if precip_col and precip_era5_col:
            plt.figure(figsize=(10, 8))
            # Apply unit conversion for ERA5 precipitation
            era5_matched_with_conversion = era5_matched.copy()
            era5_matched_with_conversion[precip_era5_col] *= 1000  # m to mm
            
            # Plot different mismatch types with different markers
            # Type 1: ENV VPD < 0.3 & ERA5 VPD > 1
            type1_valid = env_low_era5_high & ~np.isnan(mismatches[precip_col]) & ~np.isnan(era5_matched_with_conversion.loc[mismatches.index, precip_era5_col])
            if type1_valid.sum() > 0:
                plt.scatter(mismatches.loc[type1_valid, precip_col],
                          era5_matched_with_conversion.loc[mismatches.loc[type1_valid].index, precip_era5_col],
                          color='red', alpha=0.7, marker='o', s=60,
                          label=f'Type 1 (ENV VPD < 0.3, ERA5 VPD > 1): {type1_valid.sum()}')
            
            # Type 2: ERA5 VPD < 0.3 & ENV VPD > 1
            type2_valid = era5_low_env_high & ~np.isnan(mismatches[precip_col]) & ~np.isnan(era5_matched_with_conversion.loc[mismatches.index, precip_era5_col])
            if type2_valid.sum() > 0:
                plt.scatter(mismatches.loc[type2_valid, precip_col],
                          era5_matched_with_conversion.loc[mismatches.loc[type2_valid].index, precip_era5_col],
                          color='blue', alpha=0.7, marker='s', s=60,
                          label=f'Type 2 (ERA5 VPD < 0.3, ENV VPD > 1): {type2_valid.sum()}')
            
            # Add reference line for y = x (perfect agreement)
            if type1_valid.sum() > 0 or type2_valid.sum() > 0:
                valid_mask = ~np.isnan(mismatches[precip_col]) & ~np.isnan(era5_matched_with_conversion.loc[mismatches.index, precip_era5_col])
                if valid_mask.sum() > 0:
                    precip_min = min(mismatches.loc[valid_mask, precip_col].min(), 
                                  era5_matched_with_conversion.loc[mismatches.loc[valid_mask].index, precip_era5_col].min())
                    precip_max = max(mismatches.loc[valid_mask, precip_col].max(), 
                                   era5_matched_with_conversion.loc[mismatches.loc[valid_mask].index, precip_era5_col].max())
                    plt.plot([precip_min, precip_max], [precip_min, precip_max], 'k--', alpha=0.5, label='1:1 line')
            
            # Add statistics
            stats_text = ""
            if env_low_era5_high_count > 0 and not np.isnan(env_low_era5_high_mean_env_precip) and not np.isnan(env_low_era5_high_mean_era5_precip):
                stats_text += f"Type 1 Precip (mean):\n  ENV: {env_low_era5_high_mean_env_precip:.2f} mm\n  ERA5: {env_low_era5_high_mean_era5_precip:.2f} mm\n\n"
            if era5_low_env_high_count > 0 and not np.isnan(era5_low_env_high_mean_env_precip) and not np.isnan(era5_low_env_high_mean_era5_precip):
                stats_text += f"Type 2 Precip (mean):\n  ENV: {era5_low_env_high_mean_env_precip:.2f} mm\n  ERA5: {era5_low_env_high_mean_era5_precip:.2f} mm"
            
            if stats_text:
                plt.text(0.02, 0.98, stats_text,
                        transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            plt.title(f"Precipitation at VPD Mismatch Points for {site_name}")
            plt.xlabel('ENV Precipitation (mm)')
            plt.ylabel('ERA5 Precipitation (mm)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(f'./outputs/figures/mismatch_analysis/mismatched_points/{site_name}_precip_scatter.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 7. Distribution of VPD differences by mismatch type
        plt.figure(figsize=(10, 6))
        
        # Calculate VPD differences for each type
        if env_low_era5_high_count > 0:
            type1_diffs = abs(mismatches.loc[env_low_era5_high, 'vpd'] - era5_matched.loc[mismatches.loc[env_low_era5_high].index, 'vpd'])
            sns.histplot(type1_diffs, kde=True, color='red', label=f'Type 1 (ENV < 0.3, ERA5 > 1): {env_low_era5_high_count}')
        
        if era5_low_env_high_count > 0:
            type2_diffs = abs(mismatches.loc[era5_low_env_high, 'vpd'] - era5_matched.loc[mismatches.loc[era5_low_env_high].index, 'vpd'])
            sns.histplot(type2_diffs, kde=True, color='blue', label=f'Type 2 (ERA5 < 0.3, ENV > 1): {era5_low_env_high_count}')
        
        plt.title(f"Distribution of VPD Differences by Mismatch Type for {site_name}")
        plt.xlabel('|ENV VPD - ERA5 VPD| (kPa)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig(f'./outputs/figures/mismatch_analysis/mismatched_points/{site_name}_vpd_diff_by_type.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save overall statistics for all sites
    if not all_site_stats.empty:
        all_site_stats.to_csv('./outputs/statistics/all_sites_mismatch_statistics.csv', index=False)
        
        # Generate summary plots across all sites
        
        # 1. Bar chart of mismatch counts by site
        plt.figure(figsize=(12, 6))
        sites = all_site_stats['site_name'].tolist()
        type1_counts = all_site_stats['env_low_era5_high_count'].tolist()
        type2_counts = all_site_stats['era5_low_env_high_count'].tolist()
        
        x = np.arange(len(sites))
        width = 0.35
        
        plt.bar(x - width/2, type1_counts, width, label='Type 1 (ENV < 0.3, ERA5 > 1)')
        plt.bar(x + width/2, type2_counts, width, label='Type 2 (ERA5 < 0.3, ENV > 1)')
        
        plt.xlabel('Site')
        plt.ylabel('Number of Mismatches')
        plt.title('Mismatch Counts by Site and Type')
        plt.xticks(x, sites, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig('./outputs/statistics/mismatch_counts_by_site.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Bar chart of mismatch percentages by site
        plt.figure(figsize=(12, 6))
        type1_percentages = all_site_stats['env_low_era5_high_percentage'].tolist()
        type2_percentages = all_site_stats['era5_low_env_high_percentage'].tolist()
        
        plt.bar(x - width/2, type1_percentages, width, label='Type 1 (ENV < 0.3, ERA5 > 1)')
        plt.bar(x + width/2, type2_percentages, width, label='Type 2 (ERA5 < 0.3, ENV > 1)')
        
        plt.xlabel('Site')
        plt.ylabel('Percentage of Mismatches (%)')
        plt.title('Mismatch Percentages by Site and Type')
        plt.xticks(x, sites, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig('./outputs/statistics/mismatch_percentages_by_site.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Stacked bar chart of mismatch temporal distribution
        plt.figure(figsize=(12, 6))
        early_june = all_site_stats['early_june_count'].tolist()
        mid_june = all_site_stats['mid_june_count'].tolist()
        late_june = all_site_stats['late_june_count'].tolist()
        
        plt.bar(x, early_june, width, label='Early June (1-7)')
        plt.bar(x, mid_june, width, bottom=early_june, label='Mid June (8-14)')
        bottom = [early + mid for early, mid in zip(early_june, mid_june)]
        plt.bar(x, late_june, width, bottom=bottom, label='Late June (15-21)')
        
        plt.xlabel('Site')
        plt.ylabel('Number of Mismatches')
        plt.title('Temporal Distribution of Mismatches by Site')
        plt.xticks(x, sites, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig('./outputs/statistics/mismatch_temporal_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

# Example usage
if __name__ == "__main__":
    # Load ERA5 data
    era5_data = pd.read_csv('data/raw/era5_extracted_data1.csv')
    era5_data['TIMESTAMP'] = pd.to_datetime(era5_data['TIMESTAMP'])
    era5_data.set_index('TIMESTAMP', inplace=True)
    era5_data['vpd'] = era5_data['vpd'] / 10  # Converting from hPa to kPa
    
    # Convert precipitation from m to mm if it exists
    if 'total_precipitation_hourly' in era5_data.columns:
        era5_data['total_precipitation_hourly'] *= 1000  # Convert from m to mm
        
    # Handle alternative precipitation column names
    for col in era5_data.columns:
        if col in ['tp', 'precip', 'precipitation'] and col != 'total_precipitation_hourly':
            era5_data[col] *= 1000  # Convert from m to mm
    
    # Get environmental data files
    env_files = list(Path('./outputs/processed_data/env/filtered').glob("*_env_data_filtered.csv"))
    
    # Run analysis with threshold-based detection method focusing on specific VPD ranges
    analyze_vpd_mismatches(
        era5_data, 
        env_files, 
        threshold=0.5,         # Not used in current implementation but kept for compatibility
        ratio_threshold=2.0    # Not used in current implementation but kept for compatibility
    )