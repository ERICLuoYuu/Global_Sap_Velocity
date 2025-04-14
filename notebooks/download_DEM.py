


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ASTER GDEM Terrain Extraction Tool

This script extracts elevation, slope, and aspect from ASTER GDEM v003 data
for a list of geographic coordinates provided in a CSV file.
"""

import os
import math
import tempfile
import zipfile
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import rasterio
from rasterio.warp import transform_bounds
from osgeo import gdal
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("terrain_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TerrainExtractor")


def setup_earthdata_session():
    """
    Set up a requests session with NASA Earthdata authentication.
    
    Returns:
    --------
    requests.Session
        Authenticated session for NASA Earthdata
    """
    # Get credentials
    print("Please enter your NASA Earthdata credentials for tile downloads:")
    username = input("Enter your NASA Earthdata username: ")
    password = input("Enter your NASA Earthdata password: ")
    
    # Create session
    session = requests.Session()
    session.auth = (username, password)
    
    # Create a user agent for identification
    session.headers.update({
        'User-Agent': 'Python/3.x NASA ASTGTM Downloader (Contact: your@email.com)'
    })
    
    # Test the authentication with a real NASA Earthdata URL
    test_url = "https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01/"
    
    try:
        logger.info("Testing NASA Earthdata authentication...")
        response = session.get(test_url)
        
        # Check for successful authentication
        if response.status_code == 200:
            logger.info("NASA Earthdata authentication successful")
            return session
        else:
            logger.error(f"NASA Earthdata authentication failed with status code {response.status_code}")
            logger.info("Response content preview: " + response.text[:200] + "...")
            return None
    except Exception as e:
        logger.error(f"Error during authentication: {e}")
        return None


def download_aster_tile_for_location(lat, lon, output_dir, session):
    """
    Download the ASTER GDEM tile that contains the specified lat/lon coordinates.
    
    Parameters:
    -----------
    lat : float
        Latitude in decimal degrees
    lon : float
        Longitude in decimal degrees
    output_dir : str
        Directory to save the downloaded tile
    session : requests.Session
        Authenticated session for NASA Earthdata
    
    Returns:
    --------
    str
        Path to the downloaded DEM file or None if download failed
    """
    # Determine the ASTER tile code for the given coordinates
    # IMPORTANT: ASTER tile naming follows a convention where the label refers to the 
    # southwest corner (bottom-left) of the tile
    # For example, N30E120 covers the area from 30-31°N and 120-121°E
    
    # Get the integer degree of the tile's SW corner (floor of the coordinates)
    lat_floor = int(math.floor(lat))
    lon_floor = int(math.floor(lon))
    
    lat_dir = 'N' if lat_floor >= 0 else 'S'
    lon_dir = 'E' if lon_floor >= 0 else 'W'
    
    # Absolute values and proper formatting
    lat_abs = abs(lat_floor)
    lon_abs = abs(lon_floor)
    
    tile_code = f"{lat_dir}{lat_abs:02d}{lon_dir}{lon_abs:03d}"
    
    # Print debugging information
    print(f"Coordinates: {lat}, {lon}")
    print(f"Mapped to tile: {tile_code}")
    print(f"This tile covers the area from {lat_floor} to {lat_floor+1}° latitude and {lon_floor} to {lon_floor+1}° longitude")
    
    # Construct URL for the ASTER tile
    base_url = "https://e4ftl01.cr.usgs.gov/ASTT/ASTGTM.003/2000.03.01"
    zip_filename = f"ASTGTMV003_{tile_code}.zip"
    url = f"{base_url}/{zip_filename}"
    
    # Path for the downloaded and extracted files
    zip_path = os.path.join(output_dir, zip_filename)
    dem_filename = f"ASTGTMV003_{tile_code}_dem.tif"
    dem_path = os.path.join(output_dir, dem_filename)
    
    # Check if the DEM file already exists
    if os.path.exists(dem_path):
        print(f"DEM file {dem_filename} already exists. Using existing file.")
        return dem_path
    
    try:
        print(f"Downloading tile for {tile_code}...")
        response = session.get(url, stream=True)
        response.raise_for_status()
        
        # Save the zip file
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract the DEM file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract only the DEM file
            for file in zip_ref.namelist():
                if file.endswith('_dem.tif'):
                    zip_ref.extract(file, output_dir)
                    # Rename if needed
                    extracted_path = os.path.join(output_dir, file)
                    if extracted_path != dem_path:
                        os.rename(extracted_path, dem_path)
        
        print(f"Successfully downloaded and extracted {dem_filename}")
        return dem_path
    
    except Exception as e:
        print(f"Error downloading or extracting DEM file: {e}")
        return None


def calculate_terrain_attributes(dem_path, lat, lon):
    """
    Calculate elevation, slope, and aspect at the specified coordinates.
    If data is missing at the exact location, search neighboring cells.
    
    Parameters:
    -----------
    dem_path : str
        Path to the DEM file
    lat : float
        Latitude in decimal degrees
    lon : float
        Longitude in decimal degrees
    
    Returns:
    --------
    tuple
        (elevation, slope, aspect) values at the specified location
    """
    try:
        # Open the DEM file
        dem = rasterio.open(dem_path)
        
        # Transform the lat/lon to the DEM's coordinate system
        # For ASTER GDEM, the x-coordinate corresponds to longitude and y to latitude
        x, y = lon, lat
        
        # Convert geographic coordinates to pixel coordinates
        # Using rasterio's transform method for correct coordinate conversion
        py, px = rasterio.transform.rowcol(dem.transform, x, y)
        
        # Verify that the pixel coordinates are within the raster bounds
        if py < 0 or py >= dem.height or px < 0 or px >= dem.width:
            print(f"Warning: Coordinates ({lat}, {lon}) fall outside the DEM boundaries")
            print(f"Pixel coordinates ({py}, {px}) out of bounds for raster size {dem.height}x{dem.width}")
            
            # Clip to the nearest valid pixel
            py = max(0, min(py, dem.height - 1))
            px = max(0, min(px, dem.width - 1))
            print(f"Using nearest valid pixel ({py}, {px}) instead")
        
        # Get the elevation value at the specific pixel
        elevation_data = dem.read(1)[py, px]
        
        # Check for no data value in elevation
        dem_nodata = dem.nodata
        if dem_nodata is not None and elevation_data == dem_nodata:
            print(f"Warning: No valid elevation data at coordinates ({lat}, {lon})")
            elevation = None
        else:
            elevation = elevation_data
        
        # Create temporary files for slope and aspect
        temp_dir = tempfile.mkdtemp()
        slope_path = os.path.join(temp_dir, 'slope.tif')
        aspect_path = os.path.join(temp_dir, 'aspect.tif')
        
        # Calculate slope
        gdal.DEMProcessing(slope_path, dem_path, 'slope')
        
        # Calculate aspect
        gdal.DEMProcessing(aspect_path, dem_path, 'aspect')
        
        # Open the slope and aspect files
        with rasterio.open(slope_path) as slope_raster:
            # Ensure we're using the same coordinate transformation
            slope_py, slope_px = rasterio.transform.rowcol(slope_raster.transform, x, y)
            slope_py = max(0, min(slope_py, slope_raster.height - 1))
            slope_px = max(0, min(slope_px, slope_raster.width - 1))
            slope_data = slope_raster.read(1)[slope_py, slope_px]
            
            # Check for no data value in slope
            slope_nodata = slope_raster.nodata
            
            # Check if the slope value is valid
            slope_is_valid = not ((slope_nodata is not None and slope_data == slope_nodata) or 
                                 slope_data < -90 or slope_data > 90)
            
            # If slope value is invalid, search neighboring cells
            if not slope_is_valid:
                print(f"Warning: No valid slope data at coordinates ({lat}, {lon}). Searching neighboring cells...")
                
                # Define search radius (in pixels)
                search_radius = 3
                
                # Initialize variables to track best value found
                valid_slope_found = False
                slope = None
                
                # Search in expanding window
                for radius in range(1, search_radius + 1):
                    if valid_slope_found:
                        break
                        
                    # Check cells in a square around the target point
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            # Skip cells we've already checked
                            if abs(dy) < radius and abs(dx) < radius:
                                continue
                                
                            # Calculate neighboring cell coordinates
                            neighbor_py = slope_py + dy
                            neighbor_px = slope_px + dx
                            
                            # Check if neighbor is within bounds
                            if (0 <= neighbor_py < slope_raster.height and 
                                0 <= neighbor_px < slope_raster.width):
                                
                                # Get slope value at neighbor
                                neighbor_slope = slope_raster.read(1)[neighbor_py, neighbor_px]
                                
                                # Check if neighbor has valid slope
                                if not ((slope_nodata is not None and neighbor_slope == slope_nodata) or 
                                       neighbor_slope < -90 or neighbor_slope > 90):
                                    slope = neighbor_slope
                                    valid_slope_found = True
                                    print(f"Found valid slope value {slope} at offset ({dy}, {dx}) from target pixel")
                                    break
                        if valid_slope_found:
                            break
                
                if not valid_slope_found:
                    print(f"No valid slope data found within {search_radius} pixel radius")
                    slope = None
            else:
                slope = slope_data
            
        with rasterio.open(aspect_path) as aspect_raster:
            # Ensure we're using the same coordinate transformation
            aspect_py, aspect_px = rasterio.transform.rowcol(aspect_raster.transform, x, y)
            aspect_py = max(0, min(aspect_py, aspect_raster.height - 1))
            aspect_px = max(0, min(aspect_px, aspect_raster.width - 1))
            aspect_data = aspect_raster.read(1)[aspect_py, aspect_px]
            
            # Check for no data value in aspect
            aspect_nodata = aspect_raster.nodata
            
            # Check if the aspect value is valid
            aspect_is_valid = not ((aspect_nodata is not None and aspect_data == aspect_nodata) or 
                                  aspect_data < 0 or aspect_data > 360)
            
            # If aspect value is invalid and we have a valid slope, search neighboring cells
            if not aspect_is_valid and slope is not None:
                print(f"Warning: No valid aspect data at coordinates ({lat}, {lon}). Searching neighboring cells...")
                
                # Define search radius (in pixels)
                search_radius = 3
                
                # Initialize variables to track best value found
                valid_aspect_found = False
                aspect = None
                
                # Search in expanding window
                for radius in range(1, search_radius + 1):
                    if valid_aspect_found:
                        break
                        
                    # Check cells in a square around the target point
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            # Skip cells we've already checked
                            if abs(dy) < radius and abs(dx) < radius:
                                continue
                                
                            # Calculate neighboring cell coordinates
                            neighbor_py = aspect_py + dy
                            neighbor_px = aspect_px + dx
                            
                            # Check if neighbor is within bounds
                            if (0 <= neighbor_py < aspect_raster.height and 
                                0 <= neighbor_px < aspect_raster.width):
                                
                                # Get aspect value at neighbor
                                neighbor_aspect = aspect_raster.read(1)[neighbor_py, neighbor_px]
                                
                                # Check if neighbor has valid aspect
                                if not ((aspect_nodata is not None and neighbor_aspect == aspect_nodata) or 
                                       neighbor_aspect < 0 or neighbor_aspect > 360):
                                    aspect = neighbor_aspect
                                    valid_aspect_found = True
                                    print(f"Found valid aspect value {aspect} at offset ({dy}, {dx}) from target pixel")
                                    break
                        if valid_aspect_found:
                            break
                
                if not valid_aspect_found:
                    print(f"No valid aspect data found within {search_radius} pixel radius")
                    aspect = None
            else:
                aspect = aspect_data
        
        # Clean up temporary files
        for path in [slope_path, aspect_path]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        
        return elevation, slope, aspect
    
    except Exception as e:
        print(f"Error calculating terrain attributes: {e}")
        return None, None, None


def standardize_aspect(aspect, slope):
    """
    Standardize aspect values to handle the circular nature of aspect data.
    Account for flat terrain (slope = 0) where aspect is undefined.
    
    Parameters:
    -----------
    aspect : float or None
        Raw aspect value in degrees
    slope : float or None
        Slope value in degrees
        
    Returns:
    --------
    tuple
        (standardized_aspect, aspect_category, sin_aspect, cos_aspect)
    """
    # For flat terrain (slope = 0 or very close to 0), aspect is undefined
    if slope is None or np.isnan(slope) or abs(slope) < 0.1:  # Using 0.1° threshold for "flat"
        return None, None, 0.0, 0.0  # Aspect=null, Category=null, sin=0, cos=0
    
    if aspect is None or np.isnan(aspect):
        return None, None, None, None
    
    # Ensure aspect is in 0-359.99 range (360 becomes 0)
    standardized_aspect = aspect % 360
    
    # Calculate the sine and cosine components for circular analysis
    sin_aspect = np.sin(np.radians(aspect))
    cos_aspect = np.cos(np.radians(aspect))
    
    # Convert to compass direction categories
    if standardized_aspect >= 337.5 or standardized_aspect < 22.5:
        category = "N"
    elif standardized_aspect >= 22.5 and standardized_aspect < 67.5:
        category = "NE"
    elif standardized_aspect >= 67.5 and standardized_aspect < 112.5:
        category = "E"
    elif standardized_aspect >= 112.5 and standardized_aspect < 157.5:
        category = "SE"
    elif standardized_aspect >= 157.5 and standardized_aspect < 202.5:
        category = "S"
    elif standardized_aspect >= 202.5 and standardized_aspect < 247.5:
        category = "SW"
    elif standardized_aspect >= 247.5 and standardized_aspect < 292.5:
        category = "W"
    elif standardized_aspect >= 292.5 and standardized_aspect < 337.5:
        category = "NW"
    
    return standardized_aspect, category, sin_aspect, cos_aspect


def process_sites_csv(csv_path, output_dir):
    """
    Process a CSV file with site information and add elevation, slope, and aspect columns.
    
    Parameters:
    -----------
    csv_path : str
        Path to the input CSV file with lat/lon columns
    output_dir : str
        Directory to save DEM tiles and output CSV
    
    Returns:
    --------
    str
        Path to the output CSV file with added terrain attributes
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Add columns for the terrain attributes
    df['elevation_m'] = np.nan
    df['slope_degrees'] = np.nan
    df['aspect_degrees'] = np.nan
    df['aspect_category'] = None
    df['aspect_sin'] = np.nan  # For circular statistics
    df['aspect_cos'] = np.nan  # For circular statistics
    
    # Keep track of downloaded tiles to avoid redundant downloads
    dem_tile_cache = {}
    
    # Authenticate once for all downloads
    session = setup_earthdata_session()
    
    # Check if authentication was successful
    if session is None:
        logger.error("Authentication failed. Cannot proceed with download.")
        logger.info("Troubleshooting tips:")
        logger.info("1. Verify your NASA Earthdata username and password are correct")
        logger.info("2. Ensure you've accepted all data usage agreements at https://urs.earthdata.nasa.gov")
        logger.info("3. Try manually downloading a file from the NASA Earthdata website first")
        logger.info("4. Make sure your NASA Earthdata account has approved access to LP DAAC data")
        return None
    
    # Process each site
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing sites"):
        try:
            lat = row['lat']
            lon = row['lon']
            
            print(f"\nProcessing site {i+1}/{len(df)}: lat={lat}, lon={lon}")
            
            # Determine the correct tile based on floor of coordinates
            lat_floor = int(math.floor(lat))
            lon_floor = int(math.floor(lon))
            
            lat_dir = 'N' if lat_floor >= 0 else 'S'
            lon_dir = 'E' if lon_floor >= 0 else 'W'
            
            lat_abs = abs(lat_floor)
            lon_abs = abs(lon_floor)
            
            tile_code = f"{lat_dir}{lat_abs:02d}{lon_dir}{lon_abs:03d}"
            
            # Check if we already have this tile
            if tile_code in dem_tile_cache:
                dem_path = dem_tile_cache[tile_code]
            else:
                # Download the tile using the authenticated session
                dem_path = download_aster_tile_for_location(lat, lon, output_dir, session)
                if dem_path:
                    dem_tile_cache[tile_code] = dem_path
                else:
                    print(f"Skipping site at row {i} due to download failure")
                    continue
            
            # Calculate terrain attributes
            elevation, slope, aspect = calculate_terrain_attributes(dem_path, lat, lon)
            
            # Standardize aspect values for circular consistency
            std_aspect, aspect_cat, sin_aspect, cos_aspect = standardize_aspect(aspect, slope)
            
            # Update the dataframe
            df.at[i, 'elevation_m'] = elevation
            df.at[i, 'slope_degrees'] = slope
            df.at[i, 'aspect_degrees'] = std_aspect
            df.at[i, 'aspect_category'] = aspect_cat
            df.at[i, 'aspect_sin'] = sin_aspect
            df.at[i, 'aspect_cos'] = cos_aspect
            
        except Exception as e:
            print(f"Error processing site at row {i}: {e}")
    
    # Save the updated dataframe
    output_path = os.path.join(output_dir, 'site_info_with_terrain.csv')
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    return output_path


def main():
    """Main function to execute the terrain extraction process"""
    import argparse
    
    site_info = 'data/raw/0.1.5/0.1.5/csv/sapwood/site_info1.csv'
    output_dir = 'data/raw/0.1.5/0.1.5/csv/sapwood'
    
    # Log the start of processing
    print(f"Starting terrain extraction for sites in {site_info}")
    print(f"Output will be saved to {output_dir}")
    
    # Process the CSV file
    output_path = process_sites_csv(site_info, output_dir)
    
    # Summarize results
    if output_path:
        print("\nProcessing complete!")
        print(f"Terrain attributes have been calculated and saved to: {output_path}")
        
        # Load the result file to display statistics
        try:
            result_df = pd.read_csv(output_path)
            
            # Count non-null values
            valid_elevations = result_df['elevation_m'].notna().sum()
            valid_slopes = result_df['slope_degrees'].notna().sum()
            valid_aspects = result_df['aspect_degrees'].notna().sum()
            
            print(f"\nSummary statistics:")
            print(f"Total sites processed: {len(result_df)}")
            print(f"Sites with valid elevation data: {valid_elevations} ({valid_elevations/len(result_df)*100:.1f}%)")
            
            if valid_elevations > 0:
                print(f"Elevation range: {result_df['elevation_m'].min():.1f} to {result_df['elevation_m'].max():.1f} meters")
                print(f"Average elevation: {result_df['elevation_m'].mean():.1f} meters")
            
            if valid_slopes > 0:
                print(f"Slope range: {result_df['slope_degrees'].min():.1f} to {result_df['slope_degrees'].max():.1f} degrees")
                print(f"Average slope: {result_df['slope_degrees'].mean():.1f} degrees")
            
            if valid_aspects > 0:
                # For aspect, we need to calculate the mean differently due to its circular nature
                if result_df['aspect_sin'].notna().sum() > 0 and result_df['aspect_cos'].notna().sum() > 0:
                    mean_sin = result_df['aspect_sin'].mean()
                    mean_cos = result_df['aspect_cos'].mean()
                    mean_aspect = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360
                    print(f"Mean aspect direction: {mean_aspect:.1f} degrees")
                
                # Show distribution of aspect categories
                if 'aspect_category' in result_df.columns:
                    aspect_counts = result_df['aspect_category'].value_counts()
                    print("\nAspect distribution:")
                    for category, count in aspect_counts.items():
                        if pd.notna(category):
                            print(f"  {category}: {count} sites ({count/len(result_df)*100:.1f}%)")
        
        except Exception as e:
            print(f"Error generating summary statistics: {e}")
    else:
        print("Processing failed. Please check the error messages above.")
        
    print("\nAdditional usage notes:")
    print("- For circular statistics, use the aspect_sin and aspect_cos columns")
    print("- To calculate mean aspect direction: arctan2(mean_sin, mean_cos)")
    print("- aspect_sin represents 'eastness' and aspect_cos represents 'northness'")


if __name__ == "__main__":
    main()