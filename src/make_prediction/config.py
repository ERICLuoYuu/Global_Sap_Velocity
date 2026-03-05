"""
Configuration settings for sap velocity prediction system.
Contains paths, parameters, and settings for the entire workflow.
"""

import os
from pathlib import Path
GEE_PROJECT = 'ee-yuluo-2' 
# Base directory structure
BASE_DIR = Path(__file__).parent.parent.parent.absolute()
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "grided"
OUTPUT_DIR = DATA_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

# ERA5-Land data settings
ERA5LAND_DIR = Path('D:/Temp/era5land_vars')
#ERA5LAND_TEMP_DIR = DATA_DIR / "temp" / "era5land_extracted"
ERA5LAND_TEMP_DIR = Path('D:/Temp/era5land_extracted/')
BIOMES_FILE = RAW_DATA_DIR / "spatial_features" / "biomes" / "biomes.shp"

# Create directories if they don't exist
for directory in [ERA5LAND_TEMP_DIR, MODEL_DIR, PREDICTION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model settings
DEFAULT_MODEL = "xgb/default_daily_without_coordinates/FINAL_xgb_default_daily_without_coordinates.joblib"
FEATURE_SCALER = "xgb/default_daily_without_coordinates/FINAL_scaler_default_daily_without_coordinates.pkl"
LABEL_SCALER = "label_scaler.pkl"

# Prediction settings
PREDICTION_WINDOW_SIZE = 8  # Number of time steps to use as input (matching training)
DEFAULT_OUTPUT_FORMAT = "csv"  # Default output format (csv, json)

# Coordinate settings for study area
# Default to a broad region (can be overridden by user input)
# set global extent as default, exclude antarctica and arctic regions
DEFAULT_LAT_MIN = -60
DEFAULT_LAT_MAX = 78
DEFAULT_LON_MIN = -180
DEFAULT_LON_MAX = 180
# Processing settings
DASK_MEMORY_LIMIT = "8GB"
DASK_CREATE_CLIENT = False
CHUNK_SIZE = {
    'time': 32,
    'step': 24,
    'latitude': 1,
    'longitude': 1
}

# Feature generation settings
TIME_FEATURES = True  # Whether to generate cyclical time features
POINT_MODE = False    # Whether to predict for specific points vs. a spatial grid
BIOME_FEATURES = True  # Whether to include biome-related features

# List of all possible biome types (must match training data)
BIOME_TYPES = [
    'Boreal forest', 
    'Subtropical desert', 
    'Temperate forest', 
    'Temperate grassland desert', 
    'Temperate rain forest', 
    'Tropical forest savanna', 
    'Tropical rain forest', 
    'Tundra', 
    'Woodland/Shrubland'
]

# Climate data file paths
TEMP_CLIMATE_FILE = Path('data/raw/grided/spatial_features/wc2.1_2.5m_bio/wc2.1_2.5m_bio_1.tif')
PRECIP_CLIMATE_FILE = Path('data/raw/grided/spatial_features/wc2.1_2.5m_bio/wc2.1_2.5m_bio_12.tif')

# Define biome types for one-hot encoding
BIOME_TYPES = [
    'Boreal forest', 'Subtropical desert', 'Temperate forest', 
    'Temperate grassland desert', 'Temperate rain forest', 
    'Tropical forest savanna', 'Tropical rain forest', 
    'Tundra', 'Woodland/Shrubland'
]

# Add these parameters to your config.py file

# Grid processing limitation (prevents memory overflows)
MAX_GRID_CELLS = 1000  # Maximum number of grid cells to process at once





# LAI data directory
LAI_DATA_DIR = Path('./data/raw/grided/globmap_lai/')

# Dask settings
DASK_CREATE_CLIENT = False
DASK_MEMORY_LIMIT = '4GB'

# Required ERA5-Land variables
REQUIRED_VARIABLES = [
    'temperature_2m',
    'dewpoint_temperature_2m',
    'total_precipitation',
    'surface_solar_radiation_downwards',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'volumetric_soil_water_layer_1',
    'soil_temperature_level_1',
    'potential_evaporation',  # For PET calculation
]

# Variable renaming mapping (source name -> model feature name)
VARIABLE_RENAME = {
    'surface_solar_radiation_downwards': 'sw_in',
    'ppfd_in': 'ppfd_in',
    'ext_rad': 'ext_rad',
    'wind_speed': 'ws',
    'temperature_2m': 'ta',
    'vpd': 'vpd',
    'LAI': 'LAI',
    'precip_pet_ratio': 'prcip/PET',
    'pft': 'pft',
    'canopy_height': 'canopy_height',
    'elevation': 'elevation',
    'annual_mean_temperature': 'mean_annual_temp',
    'annual_precipitation': 'mean_annual_precip',
    'volumetric_soil_water_layer_1': 'volumetric_soil_water_layer_1',
    'soil_temperature_level_1': 'soil_temperature_level_1',
}

# Time features flag
TIME_FEATURES = True  # Set to True if you want cyclical time features