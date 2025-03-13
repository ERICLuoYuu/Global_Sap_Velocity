import ee
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import concurrent.futures
import os
from tqdm import tqdm

# Initialize Earth Engine with your project ID
def initialize_ee(project_id):
    ee.Authenticate()
    ee.Initialize(project=project_id)
    print("Authentication successful!")

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

def process_site(site_data, variables, era5_dataset):
    """Process a single site - this function will run in parallel"""
    idx, site = site_data
    site_results = []
    
    # Get monthly chunks for this site
    chunks = get_monthly_chunks(site['start_date'], site['end_date'])
    point = ee.Geometry.Point([site['lon'], site['lat']])
    
    for chunk_start, chunk_end in chunks:
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
                    site_results.append(row)

        except Exception as e:
            print(f"Error processing site {idx+1}, chunk {chunk_start.strftime('%Y-%m-%d')}: {str(e)}")
            continue
            
    return site_results

def extract_era5_data_parallel(sites_df, variables=['temperature_2m'], max_workers=4):
    """Extract ERA5 data for sites in parallel"""
    # Initialize Earth Engine once before parallelization
    project_id = 'era5download-447713'  # Replace with your project ID
    initialize_ee(project_id)
    
    era5_dataset = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
    all_results = []
    
    # Create a list of (idx, site) tuples for parallel processing
    site_data_list = list(sites_df.iterrows())
    
    # Process sites in parallel using ThreadPoolExecutor
    # (ProcessPoolExecutor would not work with Earth Engine due to authentication issues)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and get futures
        futures = {executor.submit(process_site, site_data, variables, era5_dataset): site_data[0] 
                  for site_data in site_data_list}
        
        # Process results as they complete using tqdm for progress tracking
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), 
                          desc="Processing sites"):
            site_idx = futures[future]
            try:
                site_results = future.result()
                all_results.extend(site_results)
                print(f"Completed site {site_idx + 1}")
            except Exception as e:
                print(f"Site {site_idx + 1} generated an exception: {e}")

    if not all_results:
        print("No data was successfully retrieved")
        return pd.DataFrame()

    # Convert to DataFrame and sort
    result_df = pd.DataFrame(all_results)
    if not result_df.empty:
        result_df = result_df.sort_values(['site_id', 'datetime'])

    return result_df

# Usage example
if __name__ == "__main__":
    # Read your site information
    example_df = pd.read_csv('data/raw/0.1.5/0.1.5/csv/sapwood/site_info1.csv')

    # Define which ERA5 variables you need
    variables = [
        'leaf_area_index_high_vegetation', 
        'leaf_area_index_low_vegetation',
        'soil_temperature_level_1',
        'soil_temperature_level_2',
        'soil_temperature_level_3',
        'surface_latent_heat_flux_hourly',
        'evaporation_from_vegetation_transpiration_hourly',
        'volumetric_soil_water_layer_2',
        'volumetric_soil_water_layer_3',
    ]

    # Extract data in parallel (adjust max_workers based on your system capabilities)
    data = extract_era5_data_parallel(example_df, variables, max_workers=16)

    # Save to CSV
    data.to_csv('data/raw/era5_extracted_data_parallel.csv', index=False)