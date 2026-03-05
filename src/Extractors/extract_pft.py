import os
import pandas as pd
import ee

from pathlib import Path
import sys
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from path_config import PathConfig, get_default_paths
def get_landcover_for_sites(csv_path, output_path):
    """
    Extracts landcover data for sites from a CSV file using Google Earth Engine.

    Args:
        csv_path (str): The file path for the input CSV file.
        output_path (str): The file path for the output CSV file.
    """
    try:
        # Initialize the Earth Engine library.
        ee.Authenticate()
        ee.Initialize(project='era5download-447713')
        print("Google Earth Engine initialized successfully!")
        print("Successfully initialized Google Earth Engine.")

        # Read the site information from the CSV file.
        sites_df = pd.read_csv(csv_path)
        print(f"Reading site information from {csv_path}.")

        all_sites_data = []
        
        # Iterate over each site in the DataFrame.
        for index, row in sites_df.iterrows():
            site_name = row['site_name']
            
            # Keep datetime objects for calculations
            start_date_dt = pd.to_datetime(row['start_date'])
            end_date_dt = pd.to_datetime(row['end_date'])
            
            # Create formatted strings for Earth Engine
            start_date = start_date_dt.strftime('%Y-%m-%d')
            end_date = end_date_dt.strftime('%Y-%m-%d')
            
            lon = row['lon']
            lat = row['lat']
            site_years_data = []

            print(f"Processing site: {site_name}")

            # Define the point of interest.
            point = ee.Geometry.Point(lon, lat)
            
            # Use datetime objects for year calculations
            if end_date_dt.year > start_date_dt.year:
                download_years = range(start_date_dt.year, end_date_dt.year + 1)
            else:
                download_years = [start_date_dt.year]

            for year in download_years:
                download_year = year
                if year < 2001:
                    download_year = 2001  # MODIS data starts from 2001
                year_start_date = f"{year}-01-01"
                year_end_date = f"{year}-12-31"
                
                # Use datetime objects for comparisons
                if year == start_date_dt.year:
                    year_start_date = start_date  # Use formatted string
                if year == end_date_dt.year:
                    year_end_date = end_date      # Use formatted string
                # Select the MODIS landcover dataset.
                # This product is updated yearly.
                landcover_collection = ee.ImageCollection('MODIS/061/MCD12Q1').filterDate(f'{download_year}-01-01', f'{download_year}-01-02')

                # Get the landcover image for the period. We take the first available image.
                landcover_image = ee.Image(landcover_collection.first())

                # Extract the landcover value at the specified point.
                # We select the 'LC_Type1' band which is the primary land cover classification.
                landcover_band = landcover_image.select('LC_Type1')
                landcover_value = landcover_band.reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=point,
                    scale=500  # The resolution of the MODIS data is 500 meters.
                ).get('LC_Type1').getInfo()

                # Create a date range with hourly frequency.
                hourly_dates = pd.date_range(start=year_start_date, end=year_end_date, freq='h')

                # Create a DataFrame for the current site with hourly data.
                # The landcover is assumed to be constant over the period.
                site_year_data = pd.DataFrame({
                    'site_name': site_name,
                    'timestamp': hourly_dates,
                    'lon': lon,
                    'lat': lat,
                    'landcover_type': landcover_value
                })

                site_years_data.append(site_year_data)

            all_sites_data.append(pd.concat(site_years_data, ignore_index=True))

        # Concatenate the data for all sites.
        final_df = pd.concat(all_sites_data, ignore_index=True)

        # Save the results to a new CSV file.
        final_df.to_csv(output_path, index=False)
        print(f"Successfully exported landcover data to {output_path}.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    paths = get_default_paths()
    # Define the path to your input CSV file.
    # Make sure to replace 'your_sites.csv' with the actual file name.
    input_csv_file = paths.site_info_path

    # Define the path for the output CSV file.
    output_csv_file = paths.pft_data_path
    if not os.path.exists(os.path.dirname(output_csv_file)):
        os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)

    get_landcover_for_sites(input_csv_file, output_csv_file)