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
era5_data = pd.read_csv('data/raw/era5_extracted_data_parallel.csv')
# era5_data.rename(columns={'datetime': 'TIMESTAMP'}, inplace=True)

# Convert TIMESTAMP to datetime object
era5_data['TIMESTAMP'] = pd.to_datetime(era5_data['datetime'], utc=True)

# No need to convert to string format and then back
# Keep as datetime objects for easier merging
# era5_data['TIMESTAMP'] = era5_data['TIMESTAMP'].dt.strftime('%Y-%m-%d %H:%M:%S+00:00')

# Create a copy of TIMESTAMP as a separate column for merging
# This allows you to set the index for other operations while keeping a column for merging


        # Create a copy of TIMESTAMP for merging
era5_data.set_index('TIMESTAMP', inplace=True)
era5_data = era5_data[[col for col in era5_data.columns if col != 'datetime' and col != 'site_id' and col != 'latitude' and col != 'longitude' ]]
# for 'surface_solar_radiation_downwards_hourly', 'surface_net_solar_radiation_hourly' and 'total_precipitation_hourly' assign 0 to negtive values

env_files = list(Path('./outputs/processed_data/env/gap_filled_size1/add_era5_data').glob("*_era5.csv"))

for env_file in env_files:
    try:
        save_path = Path('./outputs/processed_data/env/gap_filled_size1/add_era5_data/')
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Extract site name
        parts = env_file.stem.split('_')
        site_name = '_'.join(parts[:-2])
        print(f"Processing {site_name}")
        
        # Load and prepare environment data
        env_data = pd.read_csv(env_file, parse_dates=['TIMESTAMP'])
        
        
        env_data.set_index('TIMESTAMP', inplace=True)
        
        # Filter ERA5 data for this specific site
        site_era5 = era5_data[era5_data['site_name'] == site_name].copy()
        
        # Merge on the TIMESTAMP_for_merge column (both are now datetime64 type)
        # Reset index first to avoid duplicate index issues
        merged_data = env_data.reset_index().merge(
            site_era5.reset_index(),
            on='TIMESTAMP',
            how='left'
        )
        
        
        
        # If there are duplicate TIMESTAMP columns from the merge
        if 'TIMESTAMP_x' in merged_data.columns and 'TIMESTAMP_y' in merged_data.columns:
            merged_data.drop('TIMESTAMP_y', axis=1, inplace=True)
            merged_data.rename(columns={'TIMESTAMP_x': 'TIMESTAMP'}, inplace=True)
        
        # Save the merged data
        output_filename = f'{site_name}_with_era5.csv'
        merged_data.to_csv(save_path / output_filename, index=False)
        print(f"Successfully processed {site_name}")
        
    except Exception as e:
        print(f"Error processing {site_name}: {e}")
        continue