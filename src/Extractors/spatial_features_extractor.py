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
from sklearn.preprocessing import scale
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
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
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(sys.path)
from path_config import PathConfig, get_default_paths
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / "env_data_extraction.log", encoding='utf-8'),
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
# The download below is `wc2.1_{resolution}_bio_{n}.tif` -- WorldClim **v2.1**,
# whose bioclimatic layers are float32 in NATIVE units: degC for the
# temperature variables, mm for precipitation, percent for isothermality.  The
# `degC * 10` integer encoding belonged to WorldClim **v1**, and applying its
# 0.1 factors to v2.1 rasters made every temperature variable 10x too small.
#
# Measured over the 226-site corpus before the correction: bio1 spanned
# -1.159 .. 2.837 (x10 -> -11.6 .. 28.4 degC, boreal to tropical), bio3
# 1.58 .. 9.07 for a quantity defined on 0..100, bio5 1.32 .. 3.95 for the
# maximum temperature of the warmest month.  bio1 reaches the model as
# `mean_annual_temp`, which the global prediction pipeline supplies in native
# degC -- so the mismatch was a 10x train/serve skew, not only a unit label.
#
# bio4 keeps factor 1: WorldClim publishes temperature seasonality as the
# standard deviation x 100, and that is the conventional form.  Divide by 100
# downstream if degC are wanted.
BIO_SCALE_FACTORS = {
    1: 1,   # Annual Mean Temperature (°C)
    2: 1,   # Mean Diurnal Range (°C)
    3: 1,   # Isothermality (percent, 0-100)
    4: 1,   # Temperature Seasonality (std dev × 100, as published)
    5: 1,   # Max Temperature of Warmest Month (°C)
    6: 1,   # Min Temperature of Coldest Month (°C)
    7: 1,   # Temperature Annual Range (°C)
    8: 1,   # Mean Temperature of Wettest Quarter (°C)
    9: 1,   # Mean Temperature of Driest Quarter (°C)
    10: 1,  # Mean Temperature of Warmest Quarter (°C)
    11: 1,  # Mean Temperature of Coldest Quarter (°C)
    12: 1,  # Annual Precipitation (mm)
    13: 1,  # Precipitation of Wettest Month (mm)
    14: 1,  # Precipitation of Driest Month (mm)
    15: 1,  # Precipitation Seasonality (Coefficient of Variation)
    16: 1,  # Precipitation of Wettest Quarter (mm)
    17: 1,  # Precipitation of Driest Quarter (mm)
    18: 1,  # Precipitation of Warmest Quarter (mm)
    19: 1,  # Precipitation of Coldest Quarter (mm)
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
        return None
    
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
        return None
    
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
# Beck et al. (2018), Sci. Data 5:180214 -- the legend of
# Beck_KG_V1_present_*.tif, which is what download_koppen_data fetches.  THIRTY
# classes, 1..30, with 0 for ocean; the raster's distinct values are exactly
# 0..30, which is the check that settles it.
#
# The previous version of this dict inserted a 31st class, "As - Tropical
# savanna, dry summer", at code 4.  Beck's A group is Af/Am/Aw only, so every
# label from 4 upward named the class BELOW the one the pixel actually held.
# 216 of the 226 corpus sites were relabelled when this was corrected.  Four
# cases show it plainly:
#   code 11  CHN_YUN_YUN (23.9 N, coastal Fujian)  "Csc", a rare alpine
#            class, for what is really Cwa -- subtropical, dry winter, hot summer
#   code 15  GBR_DEV_*   (56.0 N, Scotland)        "Cfa", hot summer, for Cfb
#   code 27  RUS_CHE_*   (68.7 N, annual mean -11.6 C)  "Dfb", WARM summer,
#            for Dfc -- the cold-summer subarctic class
#   code 29  ITA_MAT_S21 / ITA_RUN_N20 (46.7 N Alps, annual mean 0.5-0.9 C)
#            and ARG_MAZ (51.6 S)  "Dfd", a Siberian-winter class, for ET,
#            polar tundra -- which is what alpine treeline actually is
#: Beck's rasters declare ``nodata=None`` and encode ocean as 0.  Because the
#: validity test in ``extract_raster_value`` short-circuits on ``nodata is None``,
#: a raster that declares nothing accepts everything -- so the sentinel has to be
#: named here and passed in explicitly.  Three corpus sites sat on this value with
#: land 0.93 km away, one pixel.
KOPPEN_OCEAN_CODE = 0

KOPPEN_CLASSES = {
    0: "Ocean",
    1: "Af - Tropical rainforest",
    2: "Am - Tropical monsoon",
    3: "Aw - Tropical savanna",
    4: "BWh - Arid, desert, hot",
    5: "BWk - Arid, desert, cold",
    6: "BSh - Arid, steppe, hot",
    7: "BSk - Arid, steppe, cold",
    8: "Csa - Temperate, dry and hot summer",
    9: "Csb - Temperate, dry and warm summer",
    10: "Csc - Temperate, dry and cold summer",
    11: "Cwa - Temperate, dry winter and hot summer",
    12: "Cwb - Temperate, dry winter and warm summer",
    13: "Cwc - Temperate, dry winter and cold summer",
    14: "Cfa - Temperate, no dry season and hot summer",
    15: "Cfb - Temperate, no dry season and warm summer",
    16: "Cfc - Temperate, no dry season and cold summer",
    17: "Dsa - Cold, dry and hot summer",
    18: "Dsb - Cold, dry and warm summer",
    19: "Dsc - Cold, dry and cold summer",
    20: "Dsd - Cold, dry and very cold winter",
    21: "Dwa - Cold, dry winter and hot summer",
    22: "Dwb - Cold, dry winter and warm summer",
    23: "Dwc - Cold, dry winter and cold summer",
    24: "Dwd - Cold, dry winter and very cold winter",
    25: "Dfa - Cold, no dry season and hot summer",
    26: "Dfb - Cold, no dry season and warm summer",
    27: "Dfc - Cold, no dry season and cold summer",
    28: "Dfd - Cold, no dry season and very cold winter",
    29: "ET - Polar, tundra",
    30: "EF - Polar, frost",
}


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
    return None

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







#==============================
# Common Extraction Functions
#==============================

#==============================
# Main Processing Function
#==============================

def extract_raster_value(file_path, lat, lon, search_radius=3, sentinel_values=None):
    """
    Extract a value from a raster at the specified coordinates.
    If the exact point has no data, return the NEAREST valid pixel within
    search_radius.

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
    sentinel_values : iterable, optional
        Values that mean "no data" even though the raster does not DECLARE them
        as nodata. Beck's Koppen rasters ship ``nodata=None`` and encode ocean as
        0; the validity test below short-circuits on ``nodata is None``, so
        without this a raster that declares nothing accepts everything and the
        sea is returned as a climate class. Three corpus sites read the sea that
        way, with land one pixel (0.93 km) off.

    Returns:
    --------
    float or int
        Value at the coordinates, or the nearest valid neighbour, or None.
    """
    if not os.path.exists(file_path):
        logger.error(f"Raster file does not exist: {file_path}")
        return None

    try:
        with rasterio.open(file_path) as src:
            py, px = rasterio.transform.rowcol(src.transform, lon, lat)

            invalid = set()
            if src.nodata is not None:
                invalid.add(src.nodata)
            if sentinel_values:
                invalid.update(sentinel_values)

            def is_valid(value):
                # NaN never equals anything, a NaN nodata included, so it has to
                # be rejected explicitly rather than by comparison.
                if value != value:
                    return False
                return not any(value == bad for bad in invalid)

            # ONE windowed read covering the centre pixel and its whole search
            # neighbourhood. The previous implementation called src.read(1) --
            # the ENTIRE band -- for the centre pixel and again for every
            # neighbour probe: 933 MB and 0.687 s per lookup on the 0.0083 degree
            # Koppen raster against 0.003 s windowed, repeated up to 48 times.
            row_start = max(py - search_radius, 0)
            row_stop = min(py + search_radius + 1, src.height)
            col_start = max(px - search_radius, 0)
            col_stop = min(px + search_radius + 1, src.width)
            if row_stop <= row_start or col_stop <= col_start:
                logger.debug(
                    f"({lat}, {lon}) falls outside {os.path.basename(file_path)}"
                )
                return None

            block = src.read(
                1,
                window=Window(col_start, row_start, col_stop - col_start, row_stop - row_start),
            )

            # Nearest by actual distance, with the centre pixel at distance 0 so
            # it wins whenever it is valid. The previous loop scanned dy then dx
            # from -r and returned the FIRST hit, which is the ring's CORNER
            # (distance r*sqrt(2)) before its edge (distance r): at CHN_YUN_YUN
            # it took the pixel at (-1,-1) while valid land sat at (0,-1).
            best_distance = None
            best_value = None
            for ny in range(row_start, row_stop):
                for nx in range(col_start, col_stop):
                    value = block[ny - row_start, nx - col_start]
                    if not is_valid(value):
                        continue
                    distance = math.hypot(ny - py, nx - px)
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_value = value

            if best_value is None:
                logger.debug(
                    f"No valid data within {search_radius} pixels of ({lat}, {lon}) "
                    f"in {os.path.basename(file_path)}"
                )
                return None
            if best_distance > 0:
                logger.debug(
                    f"No valid data at ({lat}, {lon}) in {os.path.basename(file_path)}; "
                    f"using the nearest valid pixel, {best_distance:.2f} px away"
                )
            return best_value

    except Exception as e:
        logger.error(f"Error extracting value from {os.path.basename(file_path)}: {e}")
        return None


def extract_environmental_data(csv_path, output_dir="env_data", worldclim_data_dir="worldclim_data", koppen_data_dir="koppen_data"):
    """
    Process a CSV file with site coordinates and extract environmental variables.
    
    Parameters:
    -----------
    csv_path : str
        Path to the input CSV file with lat/lon columns
    output_dir : str
        Directory to save downloaded data and results
    worldclim_data_dir : str
        Directory for WorldClim data
    koppen_data_dir : str
        Directory for Köppen data

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
    
    current_step += 1
    logger.info(f"Step {current_step}/{total_steps}: Processing WorldClim bioclimatic variables")
    
    # Download WorldClim bioclimatic data
    bio_files = download_worldclim_bio_with_mirrors(worldclim_data_dir)
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
    koppen_file = download_koppen_data(koppen_data_dir)
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
        unresolved = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Köppen Classification"):
            site = row.get('site_name', i)
            try:
                lat, lon = row['lat'], row['lon']
                # The ocean sentinel must be named: the raster declares
                # nodata=None, so without this it is accepted as a climate class.
                value = extract_raster_value(
                    koppen_file, lat, lon, sentinel_values=(KOPPEN_OCEAN_CODE,)
                )

                if value is None:
                    unresolved.append((site, "no valid pixel within the search radius"))
                    continue

                code = int(value)
                if code == KOPPEN_OCEAN_CODE or code not in KOPPEN_CLASSES:
                    # A code outside the legend is a failure of this extractor's
                    # vocabulary, not a property of the site. Writing "Unknown"
                    # into the corpus buries it in a column nobody reads.
                    unresolved.append((site, f"code {code} is not a land class"))
                    continue

                df.at[i, 'koppen_class'] = code
                df.at[i, 'koppen_name'] = KOPPEN_CLASSES[code]
            except Exception as e:
                unresolved.append((site, repr(e)))

        if unresolved:
            raise ValueError(
                f"Köppen classification failed at {len(unresolved)} of {len(df)} sites: "
                f"{unresolved[:10]}"
            )
    

    
    #--------------------------------------------------------------------------
    # Save results
    #--------------------------------------------------------------------------
    output_path = output_dir / f"site_info_with_env_data.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    
    # Save processing metadata
    metadata_path = output_dir / "environmental_data_info.txt"
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
        
       
    
    logger.info(f"Processing metadata saved to {metadata_path}")
    
    return output_path

def main():
    """Main function to execute the environmental data extraction process"""
    paths = PathConfig()
    site_info = paths.terrain_attributes_data_path
    output_dir = paths.env_extracted_data_dir
    worldclim_data_dir = paths.worldclim_data_dir
    koppen_data_dir = paths.koppen_data_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(worldclim_data_dir):
        os.makedirs(worldclim_data_dir, exist_ok=True)
    if not os.path.exists(koppen_data_dir):
        os.makedirs(koppen_data_dir, exist_ok=True)
    # Log the start of processing
    logger.info(f"Starting environmental data extraction for sites in {site_info}")
    logger.info(f"Output will be saved to {output_dir}")
    
    # Process the CSV file
    output_path = extract_environmental_data(site_info, output_dir, worldclim_data_dir, koppen_data_dir)
    
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