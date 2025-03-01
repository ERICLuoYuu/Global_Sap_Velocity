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
era5_data = pd.read_csv('data/raw/era5_extracted_data1.csv')
site_info = pd.read_csv('data/raw/0.1.5/0.1.5/csv/sapwood/site_info1.csv')
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
# Load data once before the loop
era5_data = pd.read_csv('data/raw/era5_extracted_data1.csv')
era5_data['TIMESTAMP'] = pd.to_datetime(era5_data['TIMESTAMP'])
era5_data.set_index('TIMESTAMP', inplace=True)
era5_data['vpd'] = era5_data['vpd'] / 10  # Convert hPa to kPa

env_files = list(Path('./outputs/processed_data/env/filtered').glob("*_env_data_filtered.csv"))
good_sites = {}
bad_sites = {}
for env_file in env_files:
    try:
        # Extract site name
        parts = env_file.stem.split('_')
        site_name = '_'.join(parts[:-3])
        
        # Load and prepare environment data
        env_data = pd.read_csv(env_file, parse_dates=['TIMESTAMP'])
        
        # Skip if no VPD column
        if 'vpd' not in env_data.columns:
            print(f"Skipping {site_name}: 'vpd' column not found")
            continue
        
        # Process env data
        
        env_data['TIMESTAMP'] = pd.to_datetime(env_data['TIMESTAMP'])
        env_data.set_index('TIMESTAMP', inplace=True)
        env_data.index = env_data.index.round('H')
        
        if 'solar_TIMESTAMP' in env_data.columns:
            env_data = env_data.drop(columns='solar_TIMESTAMP')
            
        env_data = env_data.resample('H').mean()
        
        # Filter ERA5 data for this specific site and same date range
        site_era5 = era5_data.loc[(era5_data['site_name'] == site_name)]
        
        # Find overlapping timestamps
        common_timestamps = env_data.index.intersection(site_era5.index)
        
        if len(common_timestamps) == 0:
            print(f"No overlapping data for site {site_name}")
            continue
            
        # Filter both datasets to only include common timestamps
        env_matched = env_data.loc[common_timestamps]
        era5_matched = site_era5.loc[common_timestamps]
        
        # Remove rows with NaN values in either dataset
        valid_mask = ~(env_matched['vpd'].isna() | era5_matched['vpd'].isna())
        env_matched = env_matched.loc[valid_mask]
        era5_matched = era5_matched.loc[valid_mask]
        
        if len(env_matched) < 2:
            print(f"Insufficient valid data points for site {site_name}")
            continue
        
        # Time series plot
        plt.ioff()
        fig, ax = plt.subplots()
        sns.set_style("white")
        
        ax.plot(era5_matched.index, era5_matched['vpd'], label='ERA5', alpha=0.5)
        ax.plot(env_matched.index, env_matched['vpd'], label='ENV', alpha=0.5)
        ax.set_title(f"VPD for site {site_name}")
        ax.set_xlabel('Date')
        ax.set_ylabel('VPD (kPa)')
        ax.legend()
        
        fig.savefig(f'./outputs/figures/env_era5/{site_name}_vpd.png')
        plt.close(fig)
        
        # Scatter plot
        fig, ax = plt.subplots()
        
        # Calculate correlation
        r2 = np.corrcoef(env_matched['vpd'], era5_matched['vpd'])[0, 1] ** 2
        if r2 < 0.5:
            print(f"Low correlation for site {site_name}: {r2:.3f}")
            if site_name not in bad_sites:
                bad_sites[site_name] = {}
                bad_sites[site_name]['coordinates'] = (era5_matched['latitude'].iloc[0], era5_matched['longitude'].iloc[0]) 

        else:
            print(f"High correlation for site {site_name}: {r2:.3f}")
            if site_name not in good_sites:
                good_sites[site_name] = {}
                good_sites[site_name]['coordinates'] = (era5_matched['latitude'].iloc[0], era5_matched['longitude'].iloc[0])
        
        ax.scatter(env_matched['vpd'], era5_matched['vpd'])
        ax.set_title(f"VPD correlation for site {site_name}")
        ax.set_xlabel('ENV VPD (kPa)')
        ax.set_ylabel('ERA5 VPD (kPa)')
        
        # Add regression line
        m, b = np.polyfit(env_matched['vpd'], era5_matched['vpd'], 1)
        ax.plot(env_matched['vpd'], m*env_matched['vpd'] + b, color='lightskyblue')
        ax.text(0.05, 0.95, f"R²: {r2:.3f}", horizontalalignment='left',
                verticalalignment='top', transform=ax.transAxes)
        
        fig.savefig(f'./outputs/figures/env_era5/{site_name}_vpd_scatter.png')
        plt.close(fig)
        
    except Exception as e:
        print(f"Error processing {site_name}: {str(e)}")
print(f"Good sites: {len(good_sites)}")
print(f"Bad sites: {len(bad_sites)}")  
save_dir = Path('./outputs/tables/env_era5_comparison')
if not save_dir.exists():
    save_dir.mkdir(parents=True)
# Save good and bad sites to CSV
good_sites_df = pd.DataFrame(good_sites).T
good_sites_df.to_csv(save_dir/'good_sites.csv')    
bad_sites_df = pd.DataFrame(bad_sites).T
bad_sites_df.to_csv(save_dir/'bad_sites.csv')