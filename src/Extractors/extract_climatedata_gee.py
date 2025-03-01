import ee

# Initialize Earth Engine with your new project ID
project_id = 'era5download-447713'  # Replace with the Project ID from step 2
ee.Authenticate()
ee.Initialize(project=project_id)

print("Authentication successful!")
ee.Authenticate(auth_mode='notebook')
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def get_monthly_chunks(start_date, end_date):
    """Break up a date range into monthly chunks"""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    chunks = []
    chunk_start = start

    while chunk_start < end:
        # Get end of current month
        if chunk_start.month == 12:
            chunk_end = chunk_start.replace(year=chunk_start.year + 1, month=1, day=1)
        else:
            chunk_end = chunk_start.replace(month=chunk_start.month + 1, day=1)

        # If chunk_end exceeds end_date, use end_date instead
        chunk_end = min(chunk_end, end)

        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end

    return chunks

def extract_era5_data(sites_df, variables=['temperature_2m']):
    """Extract ERA5 data for sites in monthly chunks"""
    era5_dataset = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
    all_results = []

    for idx, site in sites_df.iterrows():
        print(f"Processing site {idx + 1} of {len(sites_df)}")

        # Get monthly chunks for this site
        chunks = get_monthly_chunks(site['start_date'], site['end_date'])
        point = ee.Geometry.Point([site['lon'], site['lat']])

        for chunk_start, chunk_end in chunks:
            print(f"  Processing chunk: {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")

            try:
                # Filter collection for current chunk
                filtered_collection = era5_dataset.filterDate(
                    chunk_start.strftime('%Y-%m-%d'),
                    chunk_end.strftime('%Y-%m-%d')
                )

                # Create point feature collection
                point_fc = ee.FeatureCollection([
                    ee.Feature(point, {'site_id': str(idx)})
                ])

                # Extract values
                def extract_values(image):
                    timestamp = ee.Date(image.get('system:time_start'))
                    values = image.reduceRegions(
                        collection=point_fc,
                        reducer=ee.Reducer.first(),
                        scale=10000
                    )
                    return values.map(lambda f: f.set('timestamp', timestamp))

                values = filtered_collection.map(extract_values).flatten()
                data = values.getInfo()

                if data and 'features' in data:
                    for feature in data['features']:
                        props = feature['properties']
                        dt = pd.to_datetime(props.get('timestamp', {}).get('value', 0), unit='ms')

                        row = {
                            'site_id': idx + 1,
                            'site_name': site['site_name'],
                            'latitude': site['lat'],
                            'longitude': site['lon'],
                            'datetime': dt,
                            
                        }
                        # Add variables
                        for var in variables:
                            row[var] = props.get(var)
                        all_results.append(row)

            except Exception as e:
                print(f"  Error processing chunk: {str(e)}")
                continue

        print(f"Completed site {idx + 1}")

    if not all_results:
        print("No data was successfully retrieved")
        return pd.DataFrame()

    # Convert to DataFrame and sort
    result_df = pd.DataFrame(all_results)
    if not result_df.empty:
        result_df = result_df.sort_values(['site_id', 'datetime'])

    return result_df

# Example usage:
# Your DataFrame should look like this:
example_df = pd.read_csv('data/raw/0.1.5/0.1.5/csv/sapwood/site_info1.csv')

# Define which ERA5 variables you need
variables =  [
                'u_component_of_wind_10m',
                'v_component_of_wind_10m',
                'surface_solar_radiation_downwards_hourly',
                'surface_net_solar_radiation_hourly',
                'surface_net_thermal_radiation_hourly',
                'total_precipitation_hourly',
                'volumetric_soil_water_layer_1',
                'volumetric_soil_water_layer_4',
                'temperature_2m',
                'dewpoint_temperature_2m'

            ]


# Extract data
data = extract_era5_data(example_df, variables)

# Save to CSV
data.to_csv('data/raw/era5_extracted_data1.csv', index=False)