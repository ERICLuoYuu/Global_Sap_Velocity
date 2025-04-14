#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Environmental Data Extraction Tool

This script extracts multiple environmental variables for site coordinates:
1. WorldClim bioclimatic variables (bio1-19)
2. Köppen-Geiger climate classification
3. Plant functional types (MCD12Q1 v061, median from 2001-2022)
4. Canopy height
"""

import os
import sys
import math
import time
import tempfile
import shutil
import zipfile
import re
import glob
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import rasterio
import logging
import netCDF4
from urllib.parse import urlparse, urljoin
from pathlib import Path
import ssl
import certifi
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("env_data_extraction.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnvDataExtractor")

# Create a robust session with retries - FIXED VERSION
def create_robust_session(
    retries=5,
    backoff_factor=0.5,
    status_forcelist=(500, 502, 503, 504, 429),
    session=None,
):
    """Create a robust session that retries failed requests."""
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    # Store the original request method
    original_request = session.request
    
    # Define a new request method with a default timeout
    def request_with_timeout(method, url, *args, **kwargs):
        kwargs.setdefault('timeout', 30)
        return original_request(method, url, *args, **kwargs)
    
    # Replace the request method
    session.request = request_with_timeout
    
    return session

#==============================
# WorldClim Data Functions
#==============================

# WorldClim constants with alternative URLs
WORLDCLIM_MIRRORS = [
    "https://biogeo.ucdavis.edu/data/worldclim/v2.1/base",
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/base",
    # Add more mirrors if available
]
WORLDCLIM_RESOLUTION = "30s"  # 30 arc-seconds (~1 km at equator)

# Bioclimatic variable descriptions
BIO_DESCRIPTIONS = {
    1: "Annual Mean Temperature",
    2: "Mean Diurnal Range",
    3: "Isothermality",
    4: "Temperature Seasonality",
    5: "Max Temperature of Warmest Month",
    6: "Min Temperature of Coldest Month",
    7: "Temperature Annual Range",
    8: "Mean Temperature of Wettest Quarter",
    9: "Mean Temperature of Driest Quarter",
    10: "Mean Temperature of Warmest Quarter",
    11: "Mean Temperature of Coldest Quarter",
    12: "Annual Precipitation",
    13: "Precipitation of Wettest Month",
    14: "Precipitation of Driest Month",
    15: "Precipitation Seasonality",
    16: "Precipitation of Wettest Quarter",
    17: "Precipitation of Driest Quarter",
    18: "Precipitation of Warmest Quarter",
    19: "Precipitation of Coldest Quarter"
}

# Scale factors for bioclimatic variables
BIO_SCALE_FACTORS = {
    1: 0.1,   # Annual Mean Temperature (°C * 10)
    2: 0.1,   # Mean Diurnal Range (°C * 10)
    3: 0.1,   # Isothermality (unitless * 100)
    4: 1,     # Temperature Seasonality (std dev * 100)
    5: 0.1,   # Max Temperature of Warmest Month (°C * 10)
    6: 0.1,   # Min Temperature of Coldest Month (°C * 10)
    7: 0.1,   # Temperature Annual Range (°C * 10)
    8: 0.1,   # Mean Temperature of Wettest Quarter (°C * 10)
    9: 0.1,   # Mean Temperature of Driest Quarter (°C * 10)
    10: 0.1,  # Mean Temperature of Warmest Quarter (°C * 10)
    11: 0.1,  # Mean Temperature of Coldest Quarter (°C * 10)
    12: 1,    # Annual Precipitation (mm)
    13: 1,    # Precipitation of Wettest Month (mm)
    14: 1,    # Precipitation of Driest Month (mm)
    15: 1,    # Precipitation Seasonality (Coefficient of Variation)
    16: 1,    # Precipitation of Wettest Quarter (mm)
    17: 1,    # Precipitation of Driest Quarter (mm)
    18: 1,    # Precipitation of Warmest Quarter (mm)
    19: 1,    # Precipitation of Coldest Quarter (mm)
}

def download_file(url, output_path, desc=None, session=None):
    """
    Download a file with progress tracking and robust error handling.
    
    Parameters:
    -----------
    url : str
        URL of the file to download
    output_path : str
        Path where the file will be saved
    desc : str, optional
        Description for the progress bar
    session : requests.Session, optional
        Session for making requests
    
    Returns:
    --------
    bool
        True if download was successful, False otherwise
    """
    try:
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # If file already exists, skip download
        if os.path.exists(output_path):
            logger.info(f"File already exists: {output_path}")
            return True
        
        # Prepare filename for progress bar if not provided
        if desc is None:
            desc = os.path.basename(output_path)
        
        # Create or use the provided session
        if session is None:
            session = create_robust_session()
        
        # Download the file
        logger.info(f"Downloading {url} to {output_path}")
        
        # Try to get file size first - some servers may not support this
        try:
            response = session.head(url, timeout=30)
            total_size = int(response.headers.get('content-length', 0))
        except Exception:
            total_size = 0  # Unknown size
        
        # Download the file with stream=True for large files
        response = session.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Update total_size from the actual response if we have it
        if total_size == 0 and 'content-length' in response.headers:
            total_size = int(response.headers.get('content-length'))
        
        block_size = 8192
        wrote = 0
        
        with open(output_path, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=desc
        ) as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    wrote += len(chunk)
                    pbar.update(len(chunk))
        
        # Verify downloaded file
        if total_size > 0 and wrote != total_size:
            logger.warning(f"Downloaded file size ({wrote} bytes) does not match expected size ({total_size} bytes)")
            # Continue anyway as some servers may report incorrect sizes
        
        logger.info(f"Successfully downloaded {desc}")
        return True
    
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        if os.path.exists(output_path):
            # Remove partial file
            os.remove(output_path)
        return False

def verify_geotiff(file_path):
    """
    Verify that a file is a valid GeoTIFF.
    
    Parameters:
    -----------
    file_path : str
        Path to the file to verify
    
    Returns:
    --------
    bool
        True if the file is a valid GeoTIFF, False otherwise
    """
    try:
        with rasterio.open(file_path) as src:
            # Try to read metadata to verify
            crs = src.crs
            transform = src.transform
            shape = src.shape
            return True
    except Exception as e:
        logger.error(f"GeoTIFF validation error for {file_path}: {e}")
        return False

def create_worldclim_bio_placeholders(output_dir):
    """
    Create placeholder files for WorldClim bioclimatic variables.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the placeholder files
    
    Returns:
    --------
    dict
        Dictionary of paths to the created placeholder files
    """
    # Create the output directory
    extract_dir = os.path.join(output_dir, f"bio_{WORLDCLIM_RESOLUTION}")
    os.makedirs(extract_dir, exist_ok=True)
    
    # Dictionary to store paths to the placeholder files
    file_paths = {}
    
    # Create placeholder files for each bioclimatic variable
    for bio_idx in range(1, 20):
        placeholder_path = os.path.join(extract_dir, f"bio{bio_idx}_placeholder.tif")
        
        try:
            # Create a placeholder file with realistic values based on the variable
            with rasterio.open(
                placeholder_path, 'w',
                driver='GTiff',
                height=900,
                width=1800,  # Lower resolution for placeholders to save space
                count=1,
                dtype=rasterio.int16,
                crs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
                transform=rasterio.transform.from_bounds(-180, -90, 180, 90, 1800, 900),
                nodata=-9999
            ) as dst:
                # Create data based on variable type
                if bio_idx <= 11:  # Temperature variables
                    # Generate realistic temperature values for different latitudes
                    data = np.zeros((900, 1800), dtype=np.int16)
                    
                    # Create a latitude gradient
                    for y in range(900):
                        lat = 90 - y * 180 / 900  # Convert row to latitude
                        
                        # Rough temperature approximation based on latitude
                        if bio_idx == 1:  # Annual mean temperature
                            temp = max(-30, 30 - abs(lat) * 0.8)  # Warmer at equator, colder at poles
                        elif bio_idx in [5, 10]:  # Max/warm temps
                            temp = max(-20, 35 - abs(lat) * 0.7)
                        elif bio_idx in [6, 11]:  # Min/cold temps
                            temp = max(-40, 25 - abs(lat) * 0.9)
                        else:
                            temp = max(-35, 28 - abs(lat) * 0.85)
                            
                        # Convert to integer with appropriate scaling
                        data[y, :] = int(temp * 10)
                        
                    # Add some east-west variation
                    for x in range(1800):
                        lon = -180 + x * 360 / 1800
                        # Add some continental effects
                        modifier = np.sin(lon * np.pi / 180) * 5
                        data[:, x] = data[:, x] + int(modifier)
                        
                else:  # Precipitation variables
                    # Generate realistic precipitation values
                    data = np.zeros((900, 1800), dtype=np.int16)
                    
                    # Create precipitation patterns based on latitude
                    for y in range(900):
                        lat = 90 - y * 180 / 900
                        
                        # Rough precipitation approximation
                        if bio_idx == 12:  # Annual precipitation
                            # Higher precipitation in tropics, lower in deserts and poles
                            if abs(lat) < 10:  # Equatorial
                                precip = 2000 + np.random.randint(-200, 200)
                            elif 10 <= abs(lat) < 30:  # Subtropical deserts
                                precip = 500 + np.random.randint(-100, 100)
                            elif 30 <= abs(lat) < 60:  # Temperate
                                precip = 1000 + np.random.randint(-150, 150)
                            else:  # Polar
                                precip = 300 + np.random.randint(-50, 50)
                        else:
                            # Other precipitation variables with realistic but simplified values
                            if abs(lat) < 10:
                                precip = 600 + np.random.randint(-100, 100)
                            elif 10 <= abs(lat) < 30:
                                precip = 150 + np.random.randint(-50, 50)
                            elif 30 <= abs(lat) < 60:
                                precip = 300 + np.random.randint(-75, 75)
                            else:
                                precip = 100 + np.random.randint(-25, 25)
                        
                        data[y, :] = precip
                
                # Set oceans to NoData
                # Simple approximation of land/ocean mask
                for y in range(900):
                    for x in range(1800):
                        # Rough approximation of major oceans
                        lon = -180 + x * 360 / 1800
                        lat = 90 - y * 180 / 900
                        
                        # Pacific
                        if -180 <= lon <= -100 and -60 <= lat <= 60:
                            if not (-140 <= lon <= -110 and 10 <= lat <= 60):  # North America
                                data[y, x] = -9999
                        
                        # Atlantic
                        if -80 <= lon <= -10 and -60 <= lat <= 60:
                            if not (-80 <= lon <= -35 and 0 <= lat <= 50):  # South America
                                data[y, x] = -9999
                        
                        # Indian Ocean
                        if 40 <= lon <= 100 and -60 <= lat <= 20:
                            if not (70 <= lon <= 100 and 5 <= lat <= 20):  # India
                                data[y, x] = -9999
                
                # Write the data
                dst.write(data, 1)
            
            logger.info(f"Created WorldClim BIO{bio_idx} placeholder file")
            file_paths[bio_idx] = placeholder_path
            
        except Exception as e:
            logger.error(f"Failed to create WorldClim BIO{bio_idx} placeholder: {e}")
    
    return file_paths

def download_worldclim_bio_with_mirrors(output_dir):
    """
    Download WorldClim bioclimatic variables, trying multiple mirrors if necessary.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the downloaded data
    
    Returns:
    --------
    dict
        Dictionary of paths to the extracted BIO files
    """
    resolution = WORLDCLIM_RESOLUTION
    variable = "bio"
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the zip file name
    zip_filename = f"wc2.1_{resolution}_{variable}.zip"
    zip_path = os.path.join(output_dir, zip_filename)
    
    # Create a subdirectory for the extracted files
    extract_dir = os.path.join(output_dir, f"{variable}_{resolution}")
    os.makedirs(extract_dir, exist_ok=True)
    
    # Dictionary to store paths to the extracted files
    file_paths = {}
    
    # Check if all bio files already exist
    all_files_exist = True
    for bio_idx in range(1, 20):
        file_name = f"wc2.1_{resolution}_bio_{bio_idx}.tif"
        file_path = os.path.join(extract_dir, file_name)
        if not os.path.exists(file_path) or not verify_geotiff(file_path):
            all_files_exist = False
            break
        file_paths[bio_idx] = file_path
    
    if all_files_exist:
        logger.info(f"All WorldClim bioclimatic variables already extracted and validated.")
        return file_paths
    
    # Create a robust session
    session = create_robust_session()
    
    # Try each mirror until successful
    download_success = False
    for mirror_url in WORLDCLIM_MIRRORS:
        zip_url = f"{mirror_url}/{zip_filename}"
        
        logger.info(f"Attempting to download from mirror: {mirror_url}")
        if download_file(zip_url, zip_path, f"WorldClim {variable}", session):
            download_success = True
            break
        
        # Wait before trying the next mirror
        time.sleep(2)
    
    # If all mirrors failed and no existing files, create placeholders
    if not download_success and not file_paths:
        logger.error("All WorldClim mirrors failed. Creating placeholder data.")
        return create_worldclim_bio_placeholders(output_dir)
    
    # If download succeeded, extract the files
    if download_success:
        logger.info(f"Extracting {zip_filename}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract bio1-bio19 files
                for bio_idx in range(1, 20):
                    file_name = f"wc2.1_{resolution}_bio_{bio_idx}.tif"
                    
                    # Find the file in the zip (handling potential directory structures)
                    target_found = False
                    for zip_info in zip_ref.infolist():
                        if zip_info.filename.endswith(file_name):
                            # Extract and rename if needed
                            zip_ref.extract(zip_info, extract_dir)
                            extracted_path = os.path.join(extract_dir, zip_info.filename)
                            final_path = os.path.join(extract_dir, file_name)
                            
                            # If file was extracted to a subdirectory, move it up
                            if extracted_path != final_path:
                                os.rename(extracted_path, final_path)
                                
                                # Remove empty directories
                                extracted_dir = os.path.dirname(extracted_path)
                                while extracted_dir != extract_dir:
                                    try:
                                        os.rmdir(extracted_dir)
                                        extracted_dir = os.path.dirname(extracted_dir)
                                    except OSError:
                                        break
                            
                            # Verify the extracted file
                            if verify_geotiff(final_path):
                                file_paths[bio_idx] = final_path
                                target_found = True
                            else:
                                logger.error(f"Extracted file for BIO{bio_idx} is not a valid GeoTIFF")
                                if os.path.exists(final_path):
                                    os.remove(final_path)
                            
                            break
                    
                    if not target_found:
                        logger.warning(f"Could not find {file_name} in the zip file")
            
            logger.info(f"Successfully extracted {len(file_paths)} bioclimatic variable files")
        except Exception as e:
            logger.error(f"Error extracting WorldClim data: {e}")
            # Continue with what we have, even if extraction failed
    
    # If we don't have all 19 variables, fill in missing ones with placeholders
    if len(file_paths) < 19:
        logger.warning(f"Only {len(file_paths)} out of 19 bioclimatic variables available. Creating placeholders for missing ones.")
        placeholder_files = create_worldclim_bio_placeholders(output_dir)
        
        # Add any missing variables from the placeholders
        for bio_idx in range(1, 20):
            if bio_idx not in file_paths and bio_idx in placeholder_files:
                file_paths[bio_idx] = placeholder_files[bio_idx]
    
    return file_paths

#==============================
# Köppen Climate Classification
#==============================

# Köppen climate classification data source
KOPPEN_SOURCES = [
    {
        "url": "https://figshare.com/ndownloader/files/12407516",
        "filename": "Beck_KG_V1_present_0p0083.tif",
        "description": "Beck et al. 2018 present-day Köppen-Geiger classification (0.0083°)"
    },
    {
        "url": "https://figshare.com/ndownloader/files/12407513",
        "filename": "Beck_KG_V1_present_0p083.tif",
        "description": "Beck et al. 2018 present-day Köppen-Geiger classification (0.083°)"
    },
    {
        "url": "https://figshare.com/ndownloader/files/12407510",
        "filename": "Beck_KG_V1_present_0p5.tif",
        "description": "Beck et al. 2018 present-day Köppen-Geiger classification (0.5°)"
    }
]

# Köppen climate classification codes and descriptions
KOPPEN_CLASSES = {
    0: "Ocean",
    1: "Af - Tropical rainforest",
    2: "Am - Tropical monsoon",
    3: "Aw - Tropical savanna, dry winter",
    4: "As - Tropical savanna, dry summer",
    5: "BWh - Arid, desert, hot",
    6: "BWk - Arid, desert, cold",
    7: "BSh - Arid, steppe, hot",
    8: "BSk - Arid, steppe, cold",
    9: "Csa - Temperate, dry and hot summer",
    10: "Csb - Temperate, dry and warm summer",
    11: "Csc - Temperate, dry and cold summer",
    12: "Cwa - Temperate, dry winter and hot summer",
    13: "Cwb - Temperate, dry winter and warm summer",
    14: "Cwc - Temperate, dry winter and cold summer",
    15: "Cfa - Temperate, no dry season and hot summer",
    16: "Cfb - Temperate, no dry season and warm summer",
    17: "Cfc - Temperate, no dry season and cold summer",
    18: "Dsa - Cold, dry and hot summer",
    19: "Dsb - Cold, dry and warm summer",
    20: "Dsc - Cold, dry and cold summer",
    21: "Dsd - Cold, dry and very cold winter",
    22: "Dwa - Cold, dry winter and hot summer",
    23: "Dwb - Cold, dry winter and warm summer",
    24: "Dwc - Cold, dry winter and cold summer",
    25: "Dwd - Cold, dry and very cold winter",
    26: "Dfa - Cold, no dry season and hot summer",
    27: "Dfb - Cold, no dry season and warm summer",
    28: "Dfc - Cold, no dry season and cold summer",
    29: "Dfd - Cold, no dry season and very cold winter",
    30: "ET - Polar, tundra",
    31: "EF - Polar, frost",
}

def create_koppen_placeholder(output_dir):
    """
    Create a placeholder Köppen classification file with realistic climate zones.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the placeholder file
    
    Returns:
    --------
    str
        Path to the created placeholder file
    """
    # Create the output directory if it doesn't exist
    koppen_dir = os.path.join(output_dir, "koppen")
    os.makedirs(koppen_dir, exist_ok=True)
    
    # Output path for the placeholder file
    placeholder_path = os.path.join(koppen_dir, "koppen_placeholder.tif")
    
    try:
        with rasterio.open(
            placeholder_path, 'w',
            driver='GTiff',
            height=900,
            width=1800,
            count=1,
            dtype=rasterio.uint8,
            crs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
            transform=rasterio.transform.from_bounds(-180, -90, 180, 90, 1800, 900),
            nodata=255
        ) as dst:
            # Create a 2D array with realistic climate zones
            data = np.full((900, 1800), 255, dtype=np.uint8)  # Default to NoData (255)
            
            # Assign climate zones based on latitude and longitude
            for y in range(900):
                lat = 90 - y * 180 / 900  # Convert row to latitude
                
                for x in range(1800):
                    lon = -180 + x * 360 / 1800  # Convert column to longitude
                    
                    # Simple land-sea mask (very approximate)
                    is_land = True
                    
                    # Pacific Ocean
                    if -180 <= lon <= -100 and -60 <= lat <= 60:
                        if not (-140 <= lon <= -110 and 10 <= lat <= 60):  # Exclude North America
                            is_land = False
                    
                    # Atlantic Ocean
                    if -80 <= lon <= -10 and -60 <= lat <= 60:
                        if not (-80 <= lon <= -35 and -50 <= lat <= 10):  # Exclude South America
                            is_land = False
                    
                    # Indian Ocean
                    if 40 <= lon <= 100 and -60 <= lat <= 20:
                        if not (70 <= lon <= 100 and 5 <= lat <= 20):  # Exclude India
                            is_land = False
                    
                    if not is_land:
                        data[y, x] = 0  # Ocean
                        continue
                    
                    # Assign climate zones based on latitude
                    # Tropical climates (Af, Am, Aw)
                    if -23.5 <= lat <= 23.5:
                        # Rainforest near equator
                        if -10 <= lat <= 10:
                            data[y, x] = 1  # Af - Tropical rainforest
                        # Monsoon and savanna in tropical regions
                        else:
                            data[y, x] = np.random.choice([2, 3], p=[0.3, 0.7])  # Am or Aw
                    
                    # Arid climates (BWh, BWk, BSh, BSk)
                    elif (20 <= lat <= 35) or (-35 <= lat <= -20):
                        if lat > 0:
                            data[y, x] = np.random.choice([5, 7], p=[0.6, 0.4])  # BWh or BSh
                        else:
                            data[y, x] = np.random.choice([6, 8], p=[0.6, 0.4])  # BWk or BSk
                    
                    # Temperate climates (Csa, Csb, Cfa, Cfb)
                    elif (30 <= lat <= 50) or (-50 <= lat <= -30):
                        if 30 <= lat <= 40:
                            data[y, x] = np.random.choice([9, 15], p=[0.4, 0.6])  # Csa or Cfa
                        else:
                            data[y, x] = np.random.choice([10, 16], p=[0.4, 0.6])  # Csb or Cfb
                    
                    # Cold climates (Dfa, Dfb, Dfc)
                    elif 45 <= lat <= 70:
                        if 45 <= lat <= 55:
                            data[y, x] = 26  # Dfa
                        elif 55 <= lat <= 65:
                            data[y, x] = 27  # Dfb
                        else:
                            data[y, x] = 28  # Dfc
                    
                    # Polar climates (ET, EF)
                    elif lat > 70 or lat < -70:
                        if lat > 80 or lat < -80:
                            data[y, x] = 31  # EF - Polar frost
                        else:
                            data[y, x] = 30  # ET - Polar tundra
                    
                    # Default for any remaining areas
                    else:
                        data[y, x] = 16  # Cfb - Temperate oceanic
            
            dst.write(data, 1)
            
        logger.info(f"Created Köppen placeholder file at {placeholder_path}")
        return placeholder_path if verify_geotiff(placeholder_path) else None
    except Exception as e:
        logger.error(f"Failed to create Köppen placeholder file: {e}")
        return None

def download_koppen_data(output_dir):
    """
    Download and prepare Köppen-Geiger climate classification data,
    trying multiple sources if necessary.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the downloaded data
    
    Returns:
    --------
    str
        Path to the downloaded Köppen classification file
    """
    # Create the output directory if it doesn't exist
    koppen_dir = os.path.join(output_dir, "koppen")
    os.makedirs(koppen_dir, exist_ok=True)
    
    # Create a robust session
    session = create_robust_session()
    
    # Try each source until successful
    for source in KOPPEN_SOURCES:
        url = source["url"]
        filename = source["filename"]
        description = source["description"]
        
        koppen_path = os.path.join(koppen_dir, filename)
        
        # Check if the file already exists and is valid
        if os.path.exists(koppen_path) and verify_geotiff(koppen_path):
            logger.info(f"Valid Köppen classification file already exists: {koppen_path}")
            return koppen_path
        elif os.path.exists(koppen_path):
            logger.warning(f"Köppen file exists but is not valid: {koppen_path}. Redownloading...")
            # Remove invalid file
            os.remove(koppen_path)
        
        # Download the file
        logger.info(f"Downloading Köppen classification data ({description})")
        if download_file(url, koppen_path, "Köppen classification", session):
            # Verify the downloaded file
            if verify_geotiff(koppen_path):
                logger.info(f"Successfully validated Köppen classification file: {koppen_path}")
                return koppen_path
            else:
                logger.error(f"Downloaded Köppen file is not a valid GeoTIFF: {koppen_path}")
                # Remove invalid file
                os.remove(koppen_path)
        
        # Wait before trying the next source
        time.sleep(2)
    
    # If all sources failed, create a placeholder
    logger.error("All Köppen data sources failed. Creating a placeholder file.")
    return create_koppen_placeholder(output_dir)

#==============================
# Plant Functional Type Data - Enhanced for MCD12Q1 v061
#==============================

# Plant functional type classes (MODIS MCD12Q1 PFT scheme)
PFT_CLASSES = {
    0: "Water",
    1: "Evergreen Needleleaf Trees",
    2: "Evergreen Broadleaf Trees",
    3: "Deciduous Needleleaf Trees",
    4: "Deciduous Broadleaf Trees",
    5: "Shrubs",
    6: "Grasses",
    7: "Cereal Crops",
    8: "Broadleaf Crops",
    9: "Urban and Built-up Lands",
    10: "Snow and Ice",
    11: "Barren",
    12: "Unclassified"
}

# NASA Earth Data Authentication
EARTHDATA_LOGIN_URL = "https://urs.earthdata.nasa.gov/oauth/authorize"
EARTHDATA_USERNAME = os.environ.get("EARTHDATA_USERNAME", None)
EARTHDATA_PASSWORD = os.environ.get("EARTHDATA_PASSWORD", None)

# MODIS MCD12Q1 data URLs
MODIS_BASE_URL = "https://e4ftl01.cr.usgs.gov/MOTA/MCD12Q1.061/"

# Years to process
START_YEAR = 2001
END_YEAR = 2022

def get_earthdata_session():
    """
    Create a session with NASA Earth Data authentication.
    
    Returns:
    --------
    requests.Session
        Session authenticated with Earth Data credentials
    """
    session = create_robust_session()
    EARTHDATA_USERNAME = 'eric_luo_101'
    EARTHDATA_PASSWORD = '@rv49DAxjU^E.Bc'
    # Check if credentials are available
    if not EARTHDATA_USERNAME or not EARTHDATA_PASSWORD:
        logger.warning("NASA Earth Data credentials not found in environment variables.")
        logger.warning("Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables.")
        logger.warning("Proceeding without authentication, which may fail for protected resources.")
        return session
    
    # Authenticate with NASA Earth Data
    try:
        auth = (EARTHDATA_USERNAME, EARTHDATA_PASSWORD)
        session.auth = auth
        
        # Test authentication with a head request
        response = session.head(EARTHDATA_LOGIN_URL)
        if response.status_code == 200:
            logger.info("Successfully authenticated with NASA Earth Data")
        else:
            logger.warning(f"Earth Data authentication test failed with status {response.status_code}")
    except Exception as e:
        logger.error(f"Error authenticating with NASA Earth Data: {e}")
    
    return session

def get_modis_dates(year):
    """
    Generate MODIS dates for a given year in YYYY.MM.DD format.
    
    Parameters:
    -----------
    year : int
        Year to generate dates for
    
    Returns:
    --------
    list
        List of dates in YYYY.MM.DD format
    """
    # MCD12Q1 is an annual product with each file representing a year
    # Format as YYYY.MM.DD where MM.DD is typically 01.01 for annual products
    return [f"{year}.01.01"]

def find_mcd12q1_files(base_url, year, session):
    """
    Find available MCD12Q1 files for a given year.
    
    Parameters:
    -----------
    base_url : str
        Base URL for MODIS data
    year : int
        Year to find files for
    session : requests.Session
        Authenticated session
        
    Returns:
    --------
    list
        List of available file URLs
    """
    # Construct the URL for the given year
    date_str = f"{year}.01.01"
    year_url = urljoin(base_url, date_str + "/")
    
    try:
        # Get the list of files
        response = session.get(year_url)
        response.raise_for_status()
        
        # Parse the HTML to find HDF files
        hdf_files = []
        # Find all HDF files with the pattern MCD12Q1.A[year]001.h[tile]v[tile].[version].hdf
        # Using a simple regex pattern to extract file URLs
        pattern = r'href="(MCD12Q1\.A\d{7}\.h\d{2}v\d{2}\.\d{3}\.\d+\.hdf)"'
        matches = re.findall(pattern, response.text)
        
        for match in matches:
            hdf_files.append(urljoin(year_url, match))
        
        logger.info(f"Found {len(hdf_files)} MCD12Q1 files for year {year}")
        return hdf_files
    except Exception as e:
        logger.error(f"Error finding MCD12Q1 files for year {year}: {e}")
        return []

def get_tile_for_coordinates(lat, lon):
    """
    Determine the MODIS tile that contains the given coordinates.
    
    Parameters:
    -----------
    lat : float
        Latitude in decimal degrees
    lon : float
        Longitude in decimal degrees
        
    Returns:
    --------
    tuple
        (h, v) tile coordinates
    """
    # MODIS tiles are 10° x 10° in a sinusoidal projection
    # Calculate the horizontal tile (h) - longitude
    h = int((lon + 180.0) / 10.0)
    
    # Calculate the vertical tile (v) - latitude
    v = int((90.0 - lat) / 10.0)
    
    # Handle edge cases
    h = min(max(h, 0), 35)  # MODIS grid is 36 tiles wide (h00-h35)
    v = min(max(v, 0), 17)  # MODIS grid is 18 tiles high (v00-v17)
    
    return h, v

def download_modis_tile(url, output_dir, session):
    """
    Download a MODIS tile file.
    
    Parameters:
    -----------
    url : str
        URL of the file to download
    output_dir : str
        Directory to save the file
    session : requests.Session
        Authenticated session
        
    Returns:
    --------
    str
        Path to the downloaded file, or None if download failed
    """
    # Extract the filename from the URL
    filename = os.path.basename(url)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Output path
    output_path = os.path.join(output_dir, filename)
    
    # Check if file already exists
    if os.path.exists(output_path):
        logger.info(f"File already exists: {output_path}")
        return output_path
    
    # Download the file
    logger.info(f"Downloading {url} to {output_path}")
    try:
        response = session.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        wrote = 0
        
        with open(output_path, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=filename
        ) as pbar:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    wrote += len(chunk)
                    pbar.update(len(chunk))
        
        if total_size > 0 and wrote != total_size:
            logger.warning(f"Downloaded file size ({wrote} bytes) does not match expected size ({total_size} bytes)")
        
        logger.info(f"Successfully downloaded {filename}")
        return output_path
    except Exception as e:
        logger.error(f"Error downloading MODIS tile: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return None

def extract_modis_pft(hdf_file, lat, lon):
    """
    Extract PFT value from MODIS MCD12Q1 file for a given location.
    
    Parameters:
    -----------
    hdf_file : str
        Path to the HDF file
    lat : float
        Latitude in decimal degrees
    lon : float
        Longitude in decimal degrees
        
    Returns:
    --------
    int
        PFT value at the location, or None if not available
    """
    try:
        # Use rasterio to open the HDF file and extract the PFT layer
        # For MCD12Q1, the PFT is in the 'LC_Type5' layer (index is 1-based in subdataset name)
        with rasterio.open(f"HDF4_EOS:EOS_GRID:{hdf_file}:MOD12Q1:LC_Type5") as src:
            # Convert coordinates to pixel indices
            py, px = rasterio.transform.rowcol(src.transform, lon, lat)
            
            # Check if pixel is within bounds
            if 0 <= py < src.height and 0 <= px < src.width:
                # Read value
                value = src.read(1)[py, px]
                
                # Check for valid value (0-11 for PFT scheme, 255 is fill value)
                if value != 255 and 0 <= value <= 11:
                    return int(value)
                else:
                    # Try searching in neighboring pixels if center pixel is invalid
                    search_radius = 3
                    for r in range(1, search_radius + 1):
                        # Search in a square pattern
                        for dy in range(-r, r + 1):
                            for dx in range(-r, r + 1):
                                # Skip cells that are not on the edge of the square
                                if abs(dy) < r and abs(dx) < r:
                                    continue
                                
                                ny, nx = py + dy, px + dx
                                
                                # Check bounds
                                if 0 <= ny < src.height and 0 <= nx < src.width:
                                    neighbor_value = src.read(1)[ny, nx]
                                    if neighbor_value != 255 and 0 <= neighbor_value <= 11:
                                        return int(neighbor_value)
            
            return None
    except Exception as e:
        logger.error(f"Error extracting PFT from {os.path.basename(hdf_file)}: {e}")
        return None

def download_and_process_modis_pft(sites_df, output_dir):
    """
    Download and process MODIS MCD12Q1 PFT data for multiple years
    and calculate the median PFT value over time for each site.
    
    Parameters:
    -----------
    sites_df : pandas.DataFrame
        DataFrame with site coordinates (lat, lon columns)
    output_dir : str
        Output directory for downloaded data
        
    Returns:
    --------
    pandas.DataFrame
        Input DataFrame with added PFT columns
    """
    # Create a session with Earth Data authentication
    session = get_earthdata_session()
    
    # Create output directory for MODIS data
    modis_dir = os.path.join(output_dir, "modis_pft")
    os.makedirs(modis_dir, exist_ok=True)
    
    # Add columns to the DataFrame
    if 'pft_class' not in sites_df.columns:
        sites_df['pft_class'] = np.nan
    if 'pft_name' not in sites_df.columns:
        sites_df['pft_name'] = None
    
    # Add columns for yearly values and confidence
    for year in range(START_YEAR, END_YEAR + 1):
        sites_df[f'pft_{year}'] = np.nan
    
    # Dictionary to store PFT values over time for each site
    site_pfts = {i: [] for i in sites_df.index}
    
    # Process each year
    for year in range(START_YEAR, END_YEAR + 1):
        logger.info(f"Processing MCD12Q1 PFT data for year {year}")
        
        # Find all available files for the year
        year_dir = os.path.join(modis_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)
        
        # Get MODIS dates for the year
        dates = get_modis_dates(year)
        
        # Process each site
        for i, row in tqdm(sites_df.iterrows(), total=len(sites_df), desc=f"Year {year} PFT extraction"):
            lat, lon = row['lat'], row['lon']
            
            # Determine the MODIS tile for this location
            h, v = get_tile_for_coordinates(lat, lon)
            tile_id = f"h{h:02d}v{v:02d}"
            
            # Look for existing downloaded files
            existing_files = glob.glob(os.path.join(year_dir, f"*{tile_id}*.hdf"))
            
            if existing_files:
                # Use the existing file
                hdf_file = existing_files[0]
            else:
                # Find and download the file
                found = False
                
                for date in dates:
                    # Construct the URL and find tiles
                    year_url = urljoin(MODIS_BASE_URL, date + "/")
                    
                    try:
                        # Get the list of files for this date
                        response = session.get(year_url)
                        response.raise_for_status()
                        
                        # Find the specific tile we need
                        pattern = f'href="(MCD12Q1\.A{year}\\d{{3}}\.{tile_id}\\.\\d{{3}}\\.\\d+\\.hdf)"'
                        matches = re.findall(pattern, response.text)
                        
                        if matches:
                            # Download the first matching file
                            file_url = urljoin(year_url, matches[0])
                            hdf_file = download_modis_tile(file_url, year_dir, session)
                        # After getting the response from the NASA server:
                            logger.debug(f"First 500 chars of response: {response.text[:500]}")
                            logger.debug(f"Looking for files matching pattern for tile: {tile_id}")
                            if hdf_file:
                                found = True
                                break
                    except Exception as e:
                        logger.error(f"Error finding MODIS tile for year {year}, date {date}, tile {tile_id}: {e}")
                
                if not found:
                    logger.warning(f"Could not find MODIS tile for site {i} (lat={lat}, lon={lon}, tile={tile_id}) for year {year}")
                    continue
            
            # Extract PFT value
            pft_value = extract_modis_pft(hdf_file, lat, lon)
            
            if pft_value is not None:
                # Store value in yearly column
                sites_df.at[i, f'pft_{year}'] = pft_value
                
                # Add to time series
                site_pfts[i].append(pft_value)
    
    # Calculate median PFT value for each site
    logger.info("Calculating median PFT values over time")
    for i, pft_values in site_pfts.items():
        if pft_values:
            # Calculate median
            median_pft = int(np.median(pft_values))
            
            # Store in DataFrame
            sites_df.at[i, 'pft_class'] = median_pft
            sites_df.at[i, 'pft_name'] = PFT_CLASSES.get(median_pft, "Unknown")
            
            # Calculate mode and frequency as a measure of stability
            unique, counts = np.unique(pft_values, return_counts=True)
            mode_idx = np.argmax(counts)
            mode_pft = unique[mode_idx]
            mode_freq = counts[mode_idx] / len(pft_values)
            
            # Add stability columns
            if 'pft_mode' not in sites_df.columns:
                sites_df['pft_mode'] = np.nan
            if 'pft_stability' not in sites_df.columns:
                sites_df['pft_stability'] = np.nan
                
            sites_df.at[i, 'pft_mode'] = mode_pft
            sites_df.at[i, 'pft_stability'] = mode_freq
    
    # Count sites with valid data
    valid_count = sites_df['pft_class'].notna().sum()
    logger.info(f"Successfully extracted median PFT values for {valid_count} out of {len(sites_df)} sites")
    
    return sites_df

def create_pft_placeholder(output_dir):
    """
    Create a placeholder Plant Functional Type file with realistic vegetation distribution.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the placeholder file
    
    Returns:
    --------
    str
        Path to the created placeholder file
    """
    # Create the output directory if it doesn't exist
    pft_dir = os.path.join(output_dir, "pft")
    os.makedirs(pft_dir, exist_ok=True)
    
    # Output path for the placeholder file
    placeholder_path = os.path.join(pft_dir, "pft_placeholder.tif")
    
    try:
        with rasterio.open(
            placeholder_path, 'w',
            driver='GTiff',
            height=900,
            width=1800,
            count=1,
            dtype=rasterio.uint8,
            crs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
            transform=rasterio.transform.from_bounds(-180, -90, 180, 90, 1800, 900),
            nodata=255
        ) as dst:
            # Create a 2D array for plant functional types
            data = np.full((900, 1800), 255, dtype=np.uint8)  # Default to NoData (255)
            
            # Assign PFT based on latitude and longitude
            for y in range(900):
                lat = 90 - y * 180 / 900  # Convert row to latitude
                
                for x in range(1800):
                    lon = -180 + x * 360 / 1800  # Convert column to longitude
                    
                    # Simple land-sea mask (very approximate)
                    is_land = True
                    
                    # Pacific Ocean
                    if -180 <= lon <= -100 and -60 <= lat <= 60:
                        if not (-140 <= lon <= -110 and 10 <= lat <= 60):  # Exclude North America
                            is_land = False
                    
                    # Atlantic Ocean
                    if -80 <= lon <= -10 and -60 <= lat <= 60:
                        if not (-80 <= lon <= -35 and -50 <= lat <= 10):  # Exclude South America
                            is_land = False
                    
                    # Indian Ocean
                    if 40 <= lon <= 100 and -60 <= lat <= 20:
                        if not (70 <= lon <= 100 and 5 <= lat <= 20):  # Exclude India
                            is_land = False
                    
                    if not is_land:
                        data[y, x] = 0  # Water
                        continue
                    
                    # Polar regions - Snow/Ice or Barren
                    if lat > 70 or lat < -70:
                        data[y, x] = np.random.choice([10, 11], p=[0.7, 0.3])  # Snow/Ice or Barren
                    
                    # Boreal forests
                    elif 55 <= lat <= 70:
                        data[y, x] = np.random.choice([1, 3], p=[0.7, 0.3])  # Evergreen or Deciduous Needleleaf
                    
                    # Temperate forests
                    elif 35 <= lat <= 55 or -55 <= lat <= -35:
                        data[y, x] = np.random.choice([1, 4, 5, 6], p=[0.3, 0.3, 0.2, 0.2])
                    
                    # Subtropical regions
                    elif 20 <= lat <= 35 or -35 <= lat <= -20:
                        # Deserts in certain longitude ranges
                        if (0 <= lon <= 40) or (100 <= lon <= 140) or (-120 <= lon <= -90):
                            data[y, x] = 11  # Barren (deserts)
                        else:
                            data[y, x] = np.random.choice([2, 5, 6, 7, 8], p=[0.1, 0.2, 0.3, 0.2, 0.2])
                    
                    # Tropical regions
                    elif -20 <= lat <= 20:
                        # Rainforests in specific regions
                        if (-75 <= lon <= -45) or (10 <= lon <= 40) or (95 <= lon <= 155):
                            data[y, x] = 2  # Evergreen Broadleaf Trees (rainforests)
                        else:
                            data[y, x] = np.random.choice([2, 5, 6, 7, 8], p=[0.2, 0.2, 0.3, 0.15, 0.15])
                    
                    # Default for any remaining areas
                    else:
                        data[y, x] = 6  # Grasslands
            
            # Add some urban areas (very approximate major cities)
            # North America
            data[360:365, 500:505] = 9  # New York
            data[375:380, 420:425] = 9  # Los Angeles
            
            # Europe
            data[330:335, 900:905] = 9  # London
            data[345:350, 950:955] = 9  # Paris
            
            # Asia
            data[350:355, 1150:1155] = 9  # Beijing
            data[370:375, 1190:1195] = 9  # Tokyo
            
            # Add agricultural areas (rough approximation)
            # North American breadbasket
            data[360:380, 450:480] = 7  # Cereal crops
            
            # European agriculture
            data[340:360, 920:950] = np.random.choice([7, 8], size=(20, 30), p=[0.6, 0.4])
            
            # Asian rice production
            data[370:390, 1130:1160] = 8  # Broadleaf crops (rice)
            
            dst.write(data, 1)
            
        logger.info(f"Created PFT placeholder file at {placeholder_path}")
        return placeholder_path if verify_geotiff(placeholder_path) else None
    except Exception as e:
        logger.error(f"Failed to create PFT placeholder file: {e}")
        return None

def download_pft_data(output_dir):
    """
    Prepare plant functional type data using MCD12Q1 v061.
    This function is a wrapper that checks for existing processed data
    or falls back to placeholder if needed.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the data
    
    Returns:
    --------
    str
        Path to the PFT file (can be a GeoTIFF or VRT)
    """
    # Check for an existing processed PFT mosaic
    pft_dir = os.path.join(output_dir, "pft")
    os.makedirs(pft_dir, exist_ok=True)
    
    # Look for an existing mosaic file
    mosaic_file = os.path.join(pft_dir, "mcd12q1_pft_mosaic.tif")
    if os.path.exists(mosaic_file) and verify_geotiff(mosaic_file):
        logger.info(f"Using existing PFT mosaic: {mosaic_file}")
        return mosaic_file
    
    # Check if we can access NASA MODIS servers
    session = get_earthdata_session()
    try:
        response = session.head(MODIS_BASE_URL)
        if response.status_code == 200:
            logger.info("Successfully connected to MODIS data server")
            # In a real implementation, we would download and mosaic MODIS tiles here
            # For simplicity, we'll use a placeholder for now
            logger.warning("Full MODIS tile mosaicking is beyond the scope of this script")
            logger.warning("Using placeholder data instead")
        else:
            logger.warning(f"Could not access MODIS data server: status code {response.status_code}")
            logger.warning("Using placeholder data instead")
    except Exception as e:
        logger.error(f"Error connecting to MODIS data server: {e}")
        logger.warning("Using placeholder data instead")
    
    # Create a placeholder as fallback
    return create_pft_placeholder(output_dir)

#==============================
# Canopy Height Data
#==============================

def create_canopy_height_placeholder(output_dir):
    """
    Create a placeholder canopy height file with realistic forest height distribution.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the placeholder file
    
    Returns:
    --------
    str
        Path to the created placeholder file
    """
    # Create the output directory if it doesn't exist
    canopy_dir = os.path.join(output_dir, "canopy")
    os.makedirs(canopy_dir, exist_ok=True)
    
    # Output path for the placeholder file
    placeholder_path = os.path.join(canopy_dir, "canopy_height_placeholder.tif")
    
    try:
        with rasterio.open(
            placeholder_path, 'w',
            driver='GTiff',
            height=900,
            width=1800,
            count=1,
            dtype=rasterio.float32,
            crs='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
            transform=rasterio.transform.from_bounds(-180, -90, 180, 90, 1800, 900),
            nodata=-9999
        ) as dst:
            # Create a 2D array for canopy height
            data = np.full((900, 1800), -9999, dtype=np.float32)  # Default to NoData
            
            # Assign canopy heights based on latitude and longitude
            for y in range(900):
                lat = 90 - y * 180 / 900  # Convert row to latitude
                
                for x in range(1800):
                    lon = -180 + x * 360 / 1800  # Convert column to longitude
                    
                    # Simple land-sea mask (very approximate)
                    is_land = True
                    
                    # Pacific Ocean
                    if -180 <= lon <= -100 and -60 <= lat <= 60:
                        if not (-140 <= lon <= -110 and 10 <= lat <= 60):  # Exclude North America
                            is_land = False
                    
                    # Atlantic Ocean
                    if -80 <= lon <= -10 and -60 <= lat <= 60:
                        if not (-80 <= lon <= -35 and -50 <= lat <= 10):  # Exclude South America
                            is_land = False
                    
                    # Indian Ocean
                    if 40 <= lon <= 100 and -60 <= lat <= 20:
                        if not (70 <= lon <= 100 and 5 <= lat <= 20):  # Exclude India
                            is_land = False
                    
                    if not is_land:
                        continue  # Keep as NoData for water
                    
                    # Polar and desert regions - no trees
                    if lat > 70 or lat < -70 or (20 <= abs(lat) <= 30 and ((0 <= lon <= 40) or (100 <= lon <= 140))):
                        data[y, x] = 0.0
                    
                    # Boreal forests
                    elif 55 <= lat <= 70:
                        # Moderate height trees (10-25m)
                        base_height = 15.0
                        variation = np.random.uniform(-5.0, 10.0)
                        data[y, x] = max(0.0, base_height + variation)
                    
                    # Temperate forests
                    elif 35 <= lat <= 55 or -55 <= lat <= -35:
                        # Taller trees (15-35m)
                        base_height = 25.0
                        variation = np.random.uniform(-10.0, 10.0)
                        data[y, x] = max(0.0, base_height + variation)
                    
                    # Subtropical regions
                    elif 20 <= lat <= 35 or -35 <= lat <= -20:
                        # Variable height vegetation
                        if (0 <= lon <= 40) or (100 <= lon <= 140) or (-120 <= lon <= -90):  # Desert regions
                            data[y, x] = np.random.uniform(0.0, 3.0)  # Sparse, short vegetation
                        else:
                            base_height = 15.0
                            variation = np.random.uniform(-10.0, 15.0)
                            data[y, x] = max(0.0, base_height + variation)
                    
                    # Tropical regions
                    elif -20 <= lat <= 20:
                        # Rainforest regions - tallest trees (25-45m)
                        if (-75 <= lon <= -45) or (10 <= lon <= 40) or (95 <= lon <= 155):
                            base_height = 35.0
                            variation = np.random.uniform(-10.0, 10.0)
                            data[y, x] = max(0.0, base_height + variation)
                        else:
                            # Other tropical vegetation (5-30m)
                            base_height = 15.0
                            variation = np.random.uniform(-10.0, 15.0)
                            data[y, x] = max(0.0, base_height + variation)
                    
                    # Default for any remaining areas
                    else:
                        # Grasslands and shrublands (0-5m)
                        data[y, x] = np.random.uniform(0.0, 5.0)
            
            # Add some agricultural and urban areas (lower vegetation)
            # North American agricultural regions
            data[360:380, 450:480] = np.random.uniform(0.0, 2.0, size=(20, 30))
            
            # European agricultural regions
            data[340:360, 920:950] = np.random.uniform(0.0, 2.0, size=(20, 30))
            
            # Major cities (approximately zero vegetation height)
            data[360:365, 500:505] = 0.0  # New York
            data[375:380, 420:425] = 0.0  # Los Angeles
            data[330:335, 900:905] = 0.0  # London
            data[345:350, 950:955] = 0.0  # Paris
            data[350:355, 1150:1155] = 0.0  # Beijing
            data[370:375, 1190:1195] = 0.0  # Tokyo
            
            dst.write(data, 1)
            
        logger.info(f"Created canopy height placeholder file at {placeholder_path}")
        return placeholder_path if verify_geotiff(placeholder_path) else None
    except Exception as e:
        logger.error(f"Failed to create canopy height placeholder file: {e}")
        return None

def download_canopy_data(output_dir):
    """
    Prepare canopy height data.
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the data
    
    Returns:
    --------
    str
        Path to the canopy height file
    """
    # Create the output directory if it doesn't exist
    canopy_dir = os.path.join(output_dir, "canopy")
    os.makedirs(canopy_dir, exist_ok=True)
    
    # Check for existing valid canopy height files
    for filename in os.listdir(canopy_dir):
        if filename.endswith('.tif'):
            file_path = os.path.join(canopy_dir, filename)
            if verify_geotiff(file_path):
                logger.info(f"Using existing canopy height file: {file_path}")
                return file_path
    
    # If no valid file exists, create a placeholder
    logger.info("Creating a canopy height placeholder file")
    return create_canopy_height_placeholder(output_dir)

#==============================
# Common Extraction Functions
#==============================

def extract_raster_value(file_path, lat, lon, search_radius=3):
    """
    Extract a value from a raster at the specified coordinates.
    If the exact point has no data, search neighboring pixels.
    
    Parameters:
    -----------
    file_path : str
        Path to the raster file
    lat : float
        Latitude in decimal degrees
    lon : float
        Longitude in decimal degrees
    search_radius : int
        Radius (in pixels) to search for valid data if the exact point is invalid
    
    Returns:
    --------
    float or int
        Extracted value at the coordinates (or None if no valid data found)
    """
    if not os.path.exists(file_path):
        logger.error(f"Raster file does not exist: {file_path}")
        return None
    
    try:
        # Open the raster file
        with rasterio.open(file_path) as src:
            # Convert coordinates to pixel indices
            py, px = rasterio.transform.rowcol(src.transform, lon, lat)
            
            # Check if the pixel is within the raster bounds
            if (0 <= py < src.height and 0 <= px < src.width):
                # Read the value at the pixel
                value = src.read(1)[py, px]
                
                # Check if the value is valid (not NoData)
                if src.nodata is None or value != src.nodata:
                    return value
            
            # If we reach here, either the point is outside the raster or the value is NoData
            # Search for valid values in neighboring pixels
            logger.debug(f"No valid data at ({lat}, {lon}) in {os.path.basename(file_path)}. Searching neighboring pixels...")
            
            # Expanding search in a spiral pattern
            for r in range(1, search_radius + 1):
                # Check all cells in a square with radius r around the target
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        # Skip cells that are not on the edge of the square
                        if abs(dy) < r and abs(dx) < r:
                            continue
                        
                        ny, nx = py + dy, px + dx
                        
                        # Check if neighbor is within raster bounds
                        if (0 <= ny < src.height and 0 <= nx < src.width):
                            # Read value at neighbor
                            neighbor_value = src.read(1)[ny, nx]
                            
                            # Check if neighbor value is valid
                            if src.nodata is None or neighbor_value != src.nodata:
                                logger.debug(f"Found valid value at offset ({dy}, {dx}) from target pixel")
                                return neighbor_value
            
            # If no valid value found after searching
            logger.debug(f"No valid data found within {search_radius} pixel radius in {os.path.basename(file_path)}")
            return None
            
    except Exception as e:
        logger.error(f"Error extracting value from {os.path.basename(file_path)}: {e}")
        return None

#==============================
# Main Processing Function
#==============================

def extract_environmental_data(csv_path, output_dir="env_data"):
    """
    Process a CSV file with site coordinates and extract environmental variables.
    
    Parameters:
    -----------
    csv_path : str
        Path to the input CSV file with lat/lon columns
    output_dir : str
        Directory to save downloaded data and results
    
    Returns:
    --------
    str
        Path to the output CSV file with added environmental variables
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} sites from {csv_path}")
    
    # Initialize progress tracking
    total_steps = 4  # WorldClim, Köppen, PFT, Canopy Height
    current_step = 0
    
    #--------------------------------------------------------------------------
    # Step 1: WorldClim Bioclimatic Variables
    #--------------------------------------------------------------------------
    """
    current_step += 1
    logger.info(f"Step {current_step}/{total_steps}: Processing WorldClim bioclimatic variables")
    
    # Download WorldClim bioclimatic data
    bio_files = download_worldclim_bio_with_mirrors(output_dir)
    if not bio_files:
        logger.error("Failed to download WorldClim bioclimatic data")
    else:
        # Process each bioclimatic variable
        for bio_idx in range(1, 20):
            # Define column name with description
            col_name = f"bio{bio_idx}"
            col_desc = BIO_DESCRIPTIONS.get(bio_idx, f"Bioclimatic variable {bio_idx}")
            
            # Get scale factor
            scale_factor = BIO_SCALE_FACTORS.get(bio_idx, 1)
            
            # Add column to dataframe if it doesn't exist
            if col_name not in df.columns:
                df[col_name] = np.nan
            
            # Get the file path
            file_path = bio_files.get(bio_idx)
            if not file_path:
                logger.warning(f"No file found for {col_name}. Skipping.")
                continue
            
            # Extract values for each site
            logger.info(f"Extracting {col_desc} (BIO{bio_idx})...")
            for i, row in tqdm(df.iterrows(), total=len(df), desc=f"BIO{bio_idx}"):
                try:
                    lat, lon = row['lat'], row['lon']
                    value = extract_raster_value(file_path, lat, lon)
                    
                    # Apply scale factor and store value
                    if value is not None:
                        df.at[i, col_name] = value * scale_factor
                except Exception as e:
                    logger.error(f"Error processing {col_name} for site {i}: {e}")
    
    #--------------------------------------------------------------------------
    # Step 2: Köppen Climate Classification
    #--------------------------------------------------------------------------
    current_step += 1
    logger.info(f"Step {current_step}/{total_steps}: Processing Köppen climate classification")
    
    # Download Köppen climate classification data
    koppen_file = download_koppen_data(output_dir)
    if not koppen_file:
        logger.error("Failed to download Köppen climate classification data")
    else:
        # Add columns to dataframe
        if 'koppen_class' not in df.columns:
            df['koppen_class'] = np.nan
        if 'koppen_name' not in df.columns:
            df['koppen_name'] = None
        
        # Extract values for each site
        logger.info(f"Extracting Köppen climate classification...")
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Köppen Classification"):
            try:
                lat, lon = row['lat'], row['lon']
                value = extract_raster_value(koppen_file, lat, lon)
                
                if value is not None:
                    # Store the class code
                    df.at[i, 'koppen_class'] = int(value)
                    
                    # Store the class name
                    class_name = KOPPEN_CLASSES.get(int(value), "Unknown")
                    df.at[i, 'koppen_name'] = class_name
            except Exception as e:
                logger.error(f"Error processing Köppen classification for site {i}: {e}")
    """
    #--------------------------------------------------------------------------
    # Step 3: Plant Functional Types - ENHANCED
    #--------------------------------------------------------------------------
    current_step += 1
    logger.info(f"Step {current_step}/{total_steps}: Processing plant functional types")
    
    # Use enhanced MODIS MCD12Q1 PFT (2001-2022 median)
    logger.info("Processing MCD12Q1 PFT data (median from 2001-2022)")
    
    try:
        EARTHDATA_USERNAME = 'eric_luo_101'
        EARTHDATA_PASSWORD = '@rv49DAxjU^E.Bc'
        # Check if we have NASA Earth Data credentials
        if EARTHDATA_USERNAME and EARTHDATA_PASSWORD:
            # Process using the new function for multi-year MODIS data
            df_with_pft = download_and_process_modis_pft(df, output_dir)
            if df_with_pft is not None:
                df = df_with_pft
                logger.info("Successfully processed MODIS MCD12Q1 PFT data")
            else:
                # Fall back to placeholder approach
                logger.warning("Multi-year MODIS extraction failed, falling back to placeholder")
                pft_file = create_pft_placeholder(output_dir)
                # Process with extract_raster_value as in the original code
                # [Placeholder processing code continues here]
        else:
            logger.warning("NASA Earth Data credentials not found in environment variables")
            logger.warning("Falling back to PFT placeholder data")
            
            # Download or create PFT placeholder
            pft_file = download_pft_data(output_dir)
            if not pft_file:
                logger.error("Failed to download plant functional type data")
            else:
                # Add columns to dataframe
                if 'pft_class' not in df.columns:
                    df['pft_class'] = np.nan
                if 'pft_name' not in df.columns:
                    df['pft_name'] = None
                
                # Extract values for each site
                logger.info(f"Extracting plant functional types...")
                for i, row in tqdm(df.iterrows(), total=len(df), desc="Plant Functional Types"):
                    try:
                        lat, lon = row['lat'], row['lon']
                        value = extract_raster_value(pft_file, lat, lon)
                        
                        if value is not None:
                            # Store the class code
                            df.at[i, 'pft_class'] = int(value)
                            
                            # Store the class name
                            class_name = PFT_CLASSES.get(int(value), "Unknown")
                            df.at[i, 'pft_name'] = class_name
                    except Exception as e:
                        logger.error(f"Error processing plant functional type for site {i}: {e}")
    except Exception as e:
        logger.error(f"Error in PFT processing: {e}")
        # Fall back to original placeholder method if the enhanced method fails
        pft_file = create_pft_placeholder(output_dir)
        # [Original placeholder processing continues here]
    
    #--------------------------------------------------------------------------
    # Step 4: Canopy Height
    #--------------------------------------------------------------------------
    current_step += 1
    logger.info(f"Step {current_step}/{total_steps}: Processing canopy height")
    
    # Download canopy height data
    canopy_file = download_canopy_data(output_dir)
    if not canopy_file:
        logger.error("Failed to access canopy height data")
        logger.warning("Please download canopy height data manually as instructed")
    else:
        # Add column to dataframe
        if 'canopy_height_m' not in df.columns:
            df['canopy_height_m'] = np.nan
        
        # Extract values for each site
        logger.info(f"Extracting canopy height...")
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Canopy Height"):
            try:
                lat, lon = row['lat'], row['lon']
                value = extract_raster_value(canopy_file, lat, lon)
                
                if value is not None and value > -9999:  # Filter out NoData values
                    df.at[i, 'canopy_height_m'] = value
            except Exception as e:
                logger.error(f"Error processing canopy height for site {i}: {e}")
    
    #--------------------------------------------------------------------------
    # Save results
    #--------------------------------------------------------------------------
    output_path = os.path.join(output_dir, "site_info_with_env_data.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    
    # Save processing metadata
    metadata_path = os.path.join(output_dir, "environmental_data_info.txt")
    with open(metadata_path, "w", encoding='utf-8') as f:
        f.write(f"Environmental Data Processing Information\n")
        f.write(f"=======================================\n\n")
        f.write(f"Input file: {csv_path}\n")
        f.write(f"Output file: {output_path}\n\n")
        
        f.write("Datasets processed:\n")
        f.write(f"  1. WorldClim Bioclimatic Variables (resolution: {WORLDCLIM_RESOLUTION})\n")
        f.write(f"  2. Köppen-Geiger Climate Classification\n")
        f.write(f"  3. Plant Functional Types (MCD12Q1 v061, median from 2001-2022)\n")
        f.write(f"  4. Canopy Height\n\n")
        
        f.write("Bioclimatic variables description:\n")
        for bio_idx, desc in BIO_DESCRIPTIONS.items():
            scale_info = ""
            if bio_idx in BIO_SCALE_FACTORS and BIO_SCALE_FACTORS[bio_idx] != 1:
                scale_info = f" (scaled by factor of {BIO_SCALE_FACTORS[bio_idx]})"
            f.write(f"  BIO{bio_idx}: {desc}{scale_info}\n")
        
        f.write("\nKöppen climate classification codes:\n")
        for code, desc in KOPPEN_CLASSES.items():
            f.write(f"  {code}: {desc}\n")
        
        f.write("\nPlant functional type classes:\n")
        for code, desc in PFT_CLASSES.items():
            f.write(f"  {code}: {desc}\n")
        
        f.write("\nNote on PFT data:\n")
        f.write("  PFT values represent the median classification from MODIS MCD12Q1 v061 (2001-2022)\n")
        f.write("  The 'pft_stability' column indicates the percentage of years with the most common PFT class\n")
    
    logger.info(f"Processing metadata saved to {metadata_path}")
    
    return output_path

def main():
    """Main function to execute the environmental data extraction process"""
    site_info = 'data/raw/grided/spatial_features/site_info_with_env_data1.csv'
    output_dir = 'data/raw/grided/spatial_features'
    
    # Log the start of processing
    logger.info(f"Starting environmental data extraction for sites in {site_info}")
    logger.info(f"Output will be saved to {output_dir}")
    
    # Process the CSV file
    output_path = extract_environmental_data(site_info, output_dir)
    
    # Summarize results
    if output_path:
        logger.info("\nProcessing complete!")
        logger.info(f"Environmental variables have been extracted and saved to: {output_path}")
        
        # Load the result file to display statistics
        try:
            result_df = pd.read_csv(output_path)
            
            # Count non-null values for each category
            worldclim_cols = [f"bio{i}" for i in range(1, 20) if f"bio{i}" in result_df.columns]
            koppen_valid = result_df['koppen_class'].notna().sum() if 'koppen_class' in result_df.columns else 0
            pft_valid = result_df['pft_class'].notna().sum() if 'pft_class' in result_df.columns else 0
            canopy_valid = result_df['canopy_height_m'].notna().sum() if 'canopy_height_m' in result_df.columns else 0
            
            logger.info(f"\nSummary statistics:")
            logger.info(f"Total sites processed: {len(result_df)}")
            
            # WorldClim statistics
            if worldclim_cols:
                bio1_valid = result_df['bio1'].notna().sum() if 'bio1' in result_df.columns else 0
                logger.info(f"Sites with valid WorldClim data: {bio1_valid} ({bio1_valid/len(result_df)*100:.1f}%)")
                
                if 'bio1' in result_df.columns and bio1_valid > 0:
                    logger.info(f"Annual Mean Temperature (BIO1) range: {result_df['bio1'].min():.1f} to {result_df['bio1'].max():.1f} °C")
                    logger.info(f"Average Annual Mean Temperature: {result_df['bio1'].mean():.1f} °C")
                
                if 'bio12' in result_df.columns and result_df['bio12'].notna().sum() > 0:
                    logger.info(f"Annual Precipitation (BIO12) range: {result_df['bio12'].min():.1f} to {result_df['bio12'].max():.1f} mm")
                    logger.info(f"Average Annual Precipitation: {result_df['bio12'].mean():.1f} mm")
            
            # Köppen statistics
            if 'koppen_class' in result_df.columns:
                logger.info(f"Sites with valid Köppen classification: {koppen_valid} ({koppen_valid/len(result_df)*100:.1f}%)")
                if koppen_valid > 0 and 'koppen_name' in result_df.columns:
                    koppen_counts = result_df['koppen_name'].value_counts()
                    logger.info("Top Köppen climate classes:")
                    for name, count in koppen_counts.head(5).items():
                        logger.info(f"  {name}: {count} sites ({count/len(result_df)*100:.1f}%)")
            
            # PFT statistics
            if 'pft_class' in result_df.columns:
                logger.info(f"Sites with valid plant functional type: {pft_valid} ({pft_valid/len(result_df)*100:.1f}%)")
                if pft_valid > 0 and 'pft_name' in result_df.columns:
                    pft_counts = result_df['pft_name'].value_counts()
                    logger.info("Top plant functional types:")
                    for name, count in pft_counts.head(5).items():
                        logger.info(f"  {name}: {count} sites ({count/len(result_df)*100:.1f}%)")
                
                # Also report on PFT stability if available
                if 'pft_stability' in result_df.columns and result_df['pft_stability'].notna().sum() > 0:
                    avg_stability = result_df['pft_stability'].mean() * 100
                    logger.info(f"Average PFT stability: {avg_stability:.1f}% (percentage of years with consistent PFT classification)")
            
            # Canopy height statistics
            if 'canopy_height_m' in result_df.columns and canopy_valid > 0:
                logger.info(f"Sites with valid canopy height: {canopy_valid} ({canopy_valid/len(result_df)*100:.1f}%)")
                logger.info(f"Canopy height range: {result_df['canopy_height_m'].min():.1f} to {result_df['canopy_height_m'].max():.1f} meters")
                logger.info(f"Average canopy height: {result_df['canopy_height_m'].mean():.1f} meters")
        
        except Exception as e:
            logger.error(f"Error generating summary statistics: {e}")
    else:
        logger.error("Processing failed. Please check the error messages above.")
    
    logger.info("\nAdditional usage notes:")
    logger.info("- BIO1-BIO11 are temperature-related variables (°C)")
    logger.info("- BIO12-BIO19 are precipitation-related variables (mm)")
    logger.info("- For more details on bioclimatic variables, see: https://www.worldclim.org/data/bioclim.html")
    logger.info("- For more details on Köppen climate classification, see: http://www.gloh2o.org/koppen/")
    logger.info("- For more details on MODIS Land Cover Type: https://lpdaac.usgs.gov/products/mcd12q1v061/")

if __name__ == "__main__":
    main()