import xarray as xr
import pandas as pd
import sys
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
parent_dir = str(Path(__file__).parent.parent)
print(parent_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from src.derive_climate_data.climate_data_calculator import ClimateDataCalculator
era5_data = pd.read_csv('data/raw/extracted_data/era5land_site_data/sapwood/era5_extracted_data.csv')
site_info = pd.read_csv('data/raw/0.1.5/0.1.5/csv/sapwood/site_info.csv')
print(era5_data.head())
print(era5_data.shape)
print(era5_data['site_id'].nunique())
# set the column name for an unmaned column
""""
site_info['site_id'] = site_info.index + 1
site_info.columns = ['site_name', 'start_date', 'end_date', 'lat', 'lon', 'site_id']
site_info['site_name'] = site_info['site_name'].str[:-10]
print(site_info.head())
"""
# convert the format of time from 19/06/2006  00:00:00 to 2006-06-19T00:00:00Z
# era5_data['TIMESTAMP'] = pd.to_datetime(era5_data['datetime'], format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%dT%H:%M:%SZ')
"""
# Change the column name datetime to TIMESTAMP
era5_data.rename(columns={'datetime': 'TIMESTAMP'}, inplace=True)
era5_data['site_id'] = era5_data['site_id']+1
era5_data = era5_data.drop(columns=['site_name', 'year', 'month', 'day', 'hour'])
# asign the site from the site_info to the era5_data
# to drop a column in a dataframe
era5_data = era5_data.merge(site_info.loc[:, ['site_id', 'site_name']], on='site_id', how='left')
print(era5_data.head())

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(era5_data.loc[era5_data['site_id'] == 1,'TIMESTAMP'], era5_data.loc[era5_data['site_id'] == 1, 'temperature_2m'])
plt.show()

"""
# save the data
# era5_data.to_csv('data/raw/era5_extracted_data1.csv', index=False)
# calculate relative humidity and save the data

calculator = ClimateDataCalculator()
"""
@staticmethod
def calculate_rh(air_temp_k, dew_temp_k):
    
    Calculate Relative Humidity using air temperature and dew point temperature
    Args:
        air_temp_k (float): Air temperature in Kelvin
        dew_temp_k (float): Dew point temperature in Kelvin
    Returns:
        float: Relative humidity in percentage
    
    # Convert temperatures to Celsius
    air_temp_c = ClimateDataCalculator.kelvin_to_celsius(air_temp_k)
    dew_temp_c = ClimateDataCalculator.kelvin_to_celsius(dew_temp_k)
    
    # Ensure dew point temperature is not higher than air temperature
    if dew_temp_c > air_temp_c:
        dew_temp_c = air_temp_c
    
    # Calculate saturated vapor pressure (es) at air temperature
    es = ClimateDataCalculator.calculate_vapor_pressure(air_temp_c)
    
    # Calculate actual vapor pressure (ea) at dew point temperature
    ea = ClimateDataCalculator.calculate_vapor_pressure(dew_temp_c)
    
    # Calculate relative humidity
    rh = (ea / es) * 100
    
    # Ensure RH doesn't exceed 100%
    return min(rh, 100.0)
    
"""
"""
# calculate relative humidity
calculator = ClimateDataCalculator()
# Apply the calculate_rh function element-wise
era5_data['rh'] = era5_data.apply(
lambda row: calculator.calculate_rh(row['temperature_2m'], row['dewpoint_temperature_2m']), 
axis=1
)
# save the data
era5_data.to_csv('data/raw/era5_extracted_data1.csv', index=False)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Optional, Union

def compare_era5_env_data(
    variable: str,
    era5_file: str = 'data/raw/era5_extracted_data1.csv',
    env_data_dir: str = './outputs/processed_data/env/timezone_adjusted',
    output_dir: str = './outputs',
    era5_unit_conversion: Optional[float] = None,
    env_unit_conversion: Optional[float] = None,
    correlation_threshold: float = 0.5,
    min_data_points: int = 24,  # 30 days of hourly data
    min_days: int = 1,
    create_plots: bool = True,
    save_results: bool = True,
    scale: str = '1h',
    raw: bool = False
) -> Dict:
    """
    Compare ERA5 reanalysis data with environmental sensor data for a given variable.
    
    Parameters:
    -----------
    variable : str
        Variable name to compare (e.g., 'vpd', 'temperature', 'humidity')
    era5_file : str
        Path to ERA5 CSV file
    env_data_dir : str
        Directory containing environmental data files
    output_dir : str
        Base directory for outputs
    era5_unit_conversion : float, optional
        Factor to multiply ERA5 data (e.g., 0.1 to convert hPa to kPa)
    env_unit_conversion : float, optional
        Factor to multiply environmental data
    correlation_threshold : float
        R² threshold to classify sites as good (>= threshold) or bad (< threshold)
    min_data_points : int
        Minimum number of valid data points required
    min_days : int
        Minimum number of days of data required
    create_plots : bool
        Whether to create time series and scatter plots
    save_results : bool
        Whether to save results to CSV files
        
    Returns:
    --------
    dict: Dictionary containing analysis results with keys:
        - 'good_sites': Dict of sites with high correlation
        - 'bad_sites': Dict of sites with low correlation
        - 'summary_stats': Summary statistics for all sites
        - 'processing_log': List of processing messages
    """
    
    # Initialize results containers
    good_sites = {}
    bad_sites = {}
    summary_stats = []
    processing_log = []
    # variable name mapping for ERA5
    variable_mapping = {
        'vpd': 'vpd',
        'surface_solar_radiation_downwards_hourly': 'sw_in',
    }
    
    # Create output directories
    if create_plots:
        if raw:
            fig_dir = Path(output_dir) / 'figures' / f'env_era5_raw_{variable}_{scale}'
        else:
            fig_dir = Path(output_dir) / 'figures' / f'env_era5_{variable}_{scale}'
        fig_dir.mkdir(parents=True, exist_ok=True)
    
    if save_results:
        if raw:
            table_dir = Path(output_dir) / 'tables' / f'env_era5_raw_comparison_{variable}_{scale}'
        else:
            table_dir = Path(output_dir) / 'tables' / f'env_era5_comparison_{variable}_{scale}'
        table_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load ERA5 data
        processing_log.append(f"Loading ERA5 data from {era5_file}")
        era5_data = pd.read_csv(era5_file)
        era5_data = era5_data.rename(columns=variable_mapping)
        # print(era5_data.columns)
        era5_data['TIMESTAMP'] = pd.to_datetime(era5_data['TIMESTAMP'])
        era5_data.set_index('TIMESTAMP', inplace=True)
        
        # Round timestamps to nearest hour for consistent matching
        era5_data.index = era5_data.index.round(scale)
        
        # Apply unit conversion if specified
        if era5_unit_conversion is not None:
            era5_data[variable] = era5_data[variable] * era5_unit_conversion
            processing_log.append(f"Applied unit conversion to ERA5 {variable}: factor = {era5_unit_conversion}")
        
        # Check if variable exists in ERA5 data
        if variable not in era5_data.columns:
            raise ValueError(f"Variable '{variable}' not found in ERA5 data. Available columns: {era5_data.columns.tolist()}")
        
    except Exception as e:
        processing_log.append(f"Error loading ERA5 data: {str(e)}")
        return {
            'good_sites': {},
            'bad_sites': {},
            'summary_stats': [],
            'processing_log': processing_log
        }
    
    # Find environmental data files
    env_files = list(Path(env_data_dir).glob("*_env_data.csv"))
    processing_log.append(f"Found {len(env_files)} environmental data files")
    
    # Process each site
    for env_file in env_files:
        try:
            # Extract site name
            parts = env_file.stem.split('_')
            if raw:

                site_name = '_'.join(parts[:-2])
            else:
                site_name = '_'.join(parts[:-3])
            
            # Load environmental data
            env_data = pd.read_csv(env_file, parse_dates=['TIMESTAMP'])
            
            # Check if variable exists
            if variable not in env_data.columns:
                processing_log.append(f"Skipping {site_name}: '{variable}' column not found")
                continue
            
            # Process environmental data
            env_data['TIMESTAMP'] = pd.to_datetime(env_data['TIMESTAMP'])
            env_data.set_index('TIMESTAMP', inplace=True)
            
            # Round timestamps to nearest hour for consistent matching
            env_data.index = env_data.index.round('h')
 
            
            # Resample to hourly means
            env_data = env_data.loc[:, [variable]]  # Keep only the variable of interest
            # print(env_data.head())
            env_data = env_data.resample(scale).mean()
            print('Resampled environmental data for site:', site_name)
            # Apply unit conversion if specified
            if env_unit_conversion is not None:
                env_data = env_data * env_unit_conversion
            
            # Filter ERA5 data for this specific site
            site_era5 = era5_data.loc[era5_data['site_name'] == site_name].copy()
            
            if len(site_era5) == 0:
                processing_log.append(f"No ERA5 data found for site {site_name}")
                continue
            site_era5 = site_era5.loc[:, ['latitude', 'longitude', variable]]  # Keep only the variable of interest
            # Resample ERA5 to hourly means (for consistency)
            site_era5 = site_era5.resample(scale).mean()
            
            # Find overlapping timestamps
            common_timestamps = env_data.index.intersection(site_era5.index)
            
            if len(common_timestamps) == 0:
                processing_log.append(f"No overlapping timestamps for site {site_name}")
                continue
            
            # Filter both datasets to common timestamps
            env_matched = env_data.loc[common_timestamps]
            era5_matched = site_era5.loc[common_timestamps]
            
            # Remove rows with NaN values
            valid_mask = ~(env_matched[variable].isna() | era5_matched[variable].isna())
            env_matched = env_matched.loc[valid_mask]
            era5_matched = era5_matched.loc[valid_mask]
            
            # Data quality checks
            if len(env_matched) < min_data_points:
                processing_log.append(f"Insufficient data points for site {site_name}: {len(env_matched)} < {min_data_points}")
                continue
            
            # Check temporal coverage
            date_range = (env_matched.index.max() - env_matched.index.min()).days
            if date_range < min_days:
                processing_log.append(f"Insufficient temporal coverage for site {site_name}: {date_range} days < {min_days} days")
                continue
            
            # Calculate statistics
            r_value = np.corrcoef(env_matched[variable], era5_matched[variable])[0, 1]
            r2 = r_value ** 2
            bias = era5_matched[variable].mean() - env_matched[variable].mean()
            rmse = np.sqrt(((era5_matched[variable] - env_matched[variable])**2).mean())
            mae = np.abs(era5_matched[variable] - env_matched[variable]).mean()
            
            # Get coordinates (assuming they're the same for all timestamps)
            coords = (era5_matched['latitude'].iloc[0], era5_matched['longitude'].iloc[0])
            
            # Store site information
            site_info = {
                'coordinates': coords,
                'r2': r2,
                'correlation': r_value,
                'bias': bias,
                'rmse': rmse,
                'mae': mae,
                'n_points': len(env_matched),
                'date_range_days': date_range,
                'env_mean': env_matched[variable].mean(),
                'era5_mean': era5_matched[variable].mean(),
                'env_std': env_matched[variable].std(),
                'era5_std': era5_matched[variable].std()
            }
            
            # Classify site
            if r2 >= correlation_threshold:
                good_sites[site_name] = site_info
                processing_log.append(f"Good correlation for site {site_name}: R² = {r2:.3f}")
            else:
                bad_sites[site_name] = site_info
                processing_log.append(f"Low correlation for site {site_name}: R² = {r2:.3f}")
            
            # Add to summary stats
            summary_stats.append({
                'site_name': site_name,
                'r2': r2,
                'correlation': r_value,
                'bias': bias,
                'rmse': rmse,
                'mae': mae,
                'n_points': len(env_matched),
                'date_range_days': date_range,
                'classification': 'good' if r2 >= correlation_threshold else 'bad'
            })
            
            # Create plots if requested
            if create_plots:
                create_comparison_plots(
                    env_matched, era5_matched, variable, site_name, 
                    r2, bias, rmse, fig_dir
                )
                
        except Exception as e:
            processing_log.append(f"Error processing site {site_name}: {str(e)}")
            continue
    
    # Summary
    processing_log.append(f"Analysis complete: {len(good_sites)} good sites, {len(bad_sites)} bad sites")
    
    # Save results if requested
    if save_results and (good_sites or bad_sites):
        save_comparison_results(good_sites, bad_sites, summary_stats, table_dir, variable)
    
    return {
        'good_sites': good_sites,
        'bad_sites': bad_sites,
        'summary_stats': summary_stats,
        'processing_log': processing_log
    }

def create_comparison_plots(env_matched, era5_matched, variable, site_name, r2, bias, rmse, fig_dir):
    """Create time series and scatter plots for site comparison."""
    
    # plt.ioff()
    
    # Time series plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    ax.plot(era5_matched.index, era5_matched[variable], label='ERA5', alpha=0.7, linewidth=1)
    ax.plot(env_matched.index, env_matched[variable], label='Environmental', alpha=0.7, linewidth=1)
    ax.set_title(f"{variable.upper()} Time Series - {site_name}")
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{variable.upper()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(fig_dir / f'{site_name}_{variable}_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Scatter plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(env_matched[variable], era5_matched[variable], alpha=0.6, s=20)
    ax.set_title(f"{variable.upper()} Correlation - {site_name}")
    ax.set_xlabel(f'Environmental {variable.upper()}')
    ax.set_ylabel(f'ERA5 {variable.upper()}')
    
    # Add 1:1 line
    min_val = min(env_matched[variable].min(), era5_matched[variable].min())
    max_val = max(env_matched[variable].max(), era5_matched[variable].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='1:1 line')
    
    # Add regression line
    m, b = np.polyfit(env_matched[variable], era5_matched[variable], 1)
    ax.plot(env_matched[variable], m*env_matched[variable] + b, 'r-', alpha=0.8, label='Regression')
    
    # Add statistics text
    stats_text = f'R² = {r2:.3f}\nBias = {bias:.3f}\nRMSE = {rmse:.3f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(fig_dir / f'{site_name}_{variable}_scatter.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def save_comparison_results(good_sites, bad_sites, summary_stats, table_dir, variable):
    """Save comparison results to CSV files."""
    
    if good_sites:
        good_df = pd.DataFrame(good_sites).T
        good_df.to_csv(table_dir / f'good_sites_{variable}.csv')
    
    if bad_sites:
        bad_df = pd.DataFrame(bad_sites).T
        bad_df.to_csv(table_dir / f'bad_sites_{variable}.csv')
    
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(table_dir / f'summary_stats_{variable}.csv', index=False)

# Example usage functions for common variables
def compare_vpd(era5_unit_conversion, era5_file='data/raw/era5_extracted_data1.csv', **kwargs):
    """Compare VPD data with default unit conversion (ERA5 hPa to kPa)."""
    return compare_era5_env_data(
        variable='vpd',
        era5_file=era5_file,
        era5_unit_conversion=era5_unit_conversion,  # Convert hPa to kPa
        **kwargs
    )

def compare_temperature(era5_file='data/raw/era5_extracted_data1.csv', **kwargs):
    """Compare temperature data (assuming both in same units)."""
    return compare_era5_env_data(
        variable='temperature',
        era5_file=era5_file,
        **kwargs
    )

def compare_humidity(era5_file='data/raw/era5_extracted_data1.csv', **kwargs):
    """Compare humidity data (assuming both in same units)."""
    return compare_era5_env_data(
        variable='humidity',
        era5_file=era5_file,
        **kwargs
    )
def compare_shortwave_radiation(era5_unit_conversion, era5_file='data/raw/extracted_data/era5land_site_data/sapwood/era5_extracted_data.csv', **kwargs):
    """Compare shortwave incoming radiation data (assuming both in same units)."""
    return compare_era5_env_data(
        variable='sw_in',
        era5_file=era5_file,
        era5_unit_conversion=era5_unit_conversion,  # Convert W/m² to kW/m²
        **kwargs
    )
# Example usage:
if __name__ == "__main__":
    '''
    print('-'*50)
    print('Comparing ERA5 VPD data with environmental sensor data after timezone adjustment')
    print('-'*50)
    # Compare VPD data
    vpd_results = compare_vpd(
        correlation_threshold=0.5,
        min_data_points=2,  # 30 days
        era5_unit_conversion=0.1,  # Convert hPa to kPa
        scale='1h',  # Hourly data
        create_plots=True
    )
    # Print the processing log to see what went wrong
    print("\nProcessing Log:")
    for message in vpd_results['processing_log']:
        print(f"  {message}")
    print(f"VPD Analysis Results:")
    print(f"Good sites: {len(vpd_results['good_sites'])}")
    print(f"Bad sites: {len(vpd_results['bad_sites'])}")
    
    print('-'*50)
    print('Comparing ERA5 VPD data with environmental sensor data beofre timezone adjustment')
    print('-'*50)
     # Compare VPD data
    vpd_results = compare_vpd(
        correlation_threshold=0.5,
        min_data_points=2,  # 30 days
        env_data_dir='data/raw/0.1.5/0.1.5/csv/sapwood',
        era5_unit_conversion=0.1,  # Convert hPa to kPa
        scale='1h',  # Hourly data
        create_plots=True,
        raw=True  # Use raw data without timezone adjustment
    )
    # Print the processing log to see what went wrong
    print("\nProcessing Log:")
    for message in vpd_results['processing_log']:
        print(f"  {message}")
    print(f"VPD Analysis Results:")
    print(f"Good sites: {len(vpd_results['good_sites'])}")
    print(f"Bad sites: {len(vpd_results['bad_sites'])}")
    '''
    
    # comapre shortwave incoming radiation data
    print('-'*50)
    print('Comparing ERA5 Shortwave Radiation data with environmental sensor data before filtering')
    print('-'*50)
    sw_results = compare_shortwave_radiation(
        correlation_threshold=0.5,
        min_data_points=2,  # 1 day
        create_plots=True,
        env_data_dir='data/raw/0.1.5/0.1.5/csv/sapwood',
        scale ='1h',  # Hourly data
        raw=True,  # Use raw data without filtering
        era5_unit_conversion=1/3600  
    )
    for message in sw_results['processing_log']:
        print(f"  {message}")
    # Print the results
    print(f"Shortwave Radiation Analysis Results:")
    print(f"Good sites: {len(sw_results['good_sites'])}")
    print(f"Bad sites: {len(sw_results['bad_sites'])}")
    '''
    print('-'*50)
    print('Comparing ERA5 Shortwave Radiation data with environmental sensor data before timezone adjustment')
    print('-'*50)
    # comapre shortwave incoming radiation data
    sw_results = compare_shortwave_radiation(
        correlation_threshold=0.5,
        min_data_points=2,  # 1 day
        create_plots=True,
        env_data_dir='outputs/processed_data/env/filtered',
        scale ='1h',  # Hourly data
        # raw=True,  # Use raw data without timezone adjustment
        era5_unit_conversion=1/3600  
    )
    for message in sw_results['processing_log']:
        print(f"  {message}")
    # Print the results
    print(f"Shortwave Radiation Analysis Results:")
    print(f"Good sites: {len(sw_results['good_sites'])}")
    print(f"Bad sites: {len(sw_results['bad_sites'])}")
    '''

    