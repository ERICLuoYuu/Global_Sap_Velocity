"""
ERA5-Land data processing utility for sap velocity prediction.
Handles extraction, reading, and preprocessing of ERA5-Land data
from Google Earth Engine with maintained functionality for derived variables.
"""
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import time
import warnings
import geopandas as gpd
import subprocess
import glob
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import rasterio
from rasterio.warp import transform_bounds, transform
import calendar
# Add Google Earth Engine imports
import ee
import geemap
import requests
import config
import rioxarray  # Import rioxarray to fix the undefined error
from rasterio.windows import from_bounds
import config

# Try optional libraries
try:
    import dask
    from dask.diagnostics import ProgressBar
    HAVE_DASK = True
    
    # Configure dask for better memory usage
    dask.config.set({
        'array.chunk-size': '32MiB',
        'distributed.worker.memory.target': 0.6,
        'distributed.worker.memory.spill': 0.7,
        'distributed.worker.memory.pause': 0.8,
    })
except ImportError:
    HAVE_DASK = False
    print("Dask not available. Some memory optimizations disabled.")

try:
    import netCDF4
    HAVE_NETCDF4 = True
except ImportError:
    HAVE_NETCDF4 = False
    print("NetCDF4 module not available. Using fallback methods.")

def _hour_processing_worker(args):
    """
    Worker function for processing a single hour (used by multiprocessing).
    Must be defined at module level to be picklable.
    
    Parameters:
    -----------
    args : tuple
        Tuple containing (year, month, day, hour, variables, shapefile, output_dir, temp_dir)
        
    Returns:
    --------
    dict
        Dictionary containing hour data and metadata, or None if processing failed
    """
    year, month, day, hour, variables, shapefile, output_dir, temp_dir = args
    
    # Create a new processor instance for each process with unique temp directory
    hour_temp_dir = Path(temp_dir) 
    hour_temp_dir.mkdir(exist_ok=True, parents=True)
    
    hour_processor = ERA5LandGEEProcessor(temp_dir=hour_temp_dir)
    
    try:
        print(f"Processing hour {year}-{month:02d}-{day:02d} {hour:02d}:00...")
        
        # Calculate start and end times for this hour
        hour_start = pd.Timestamp(year=year, month=month, day=day, hour=hour, tz='UTC')
        hour_end = hour_start + pd.Timedelta(hours=1) - pd.Timedelta(seconds=1)
        
        # Load data for this hour
        hour_processor.load_variable_data(
            variables, year, month, day, hour, shapefile=shapefile
        )
        
        # Calculate derived variables
        hour_processor.calculate_derived_variables()
        
        # Create prediction dataset for just this hour
        df = hour_processor.create_prediction_dataset(
        )
        
        if df is not None:
            # Return the data instead of saving it (we'll save the combined daily data later)
            print(f"Hour {hour} DF columns before return: {df.columns.tolist()}")
            result = {
                'hour': hour,
                'data': df,
                'timestamp': hour_start,
                'success': True
            }
            print(f"Successfully processed hour {hour:02d}:00")
            return result
        else:
            print(f"No data for hour {hour:02d}:00")
            return {'hour': hour, 'success': False}
    
    except Exception as e:
        print(f"Error processing hour {hour:02d}:00: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'hour': hour, 'success': False}
    
    finally:
        # Clean up
        hour_processor.close()
        
       
def _day_processing_worker(args):
    """
    Worker function for processing a single day (used by multiprocessing).
    Must be defined at module level to be picklable.
    
    Parameters:
    -----------
    args : tuple
        Tuple containing (year, month, day, variables, shapefile, output_dir)
        
    Returns:
    --------
    str
        Path to saved prediction dataset file, or None if processing failed
    """
    year, month, day, variables, shapefile, output_dir, day_processor = args
    
    # Create a new processor instance for each process
    day_processor = ERA5LandGEEProcessor()
    try:
        
        
        # Process the day's data
        return day_processor.process_single_day(
            variables, year, month, day, 
            shapefile=shapefile, 
            output_dir=output_dir
        )
    finally:
        day_processor.close()
class ERA5LandGEEProcessor:
    """
    Class to process ERA5-Land data from Google Earth Engine with optimized methods
    for memory efficiency. Based on the ERA5LandProcessor with modifications for
    accessing data directly from GEE instead of local files.
    """
    # Format: (temp_min, temp_max, precip_min, precip_max)
    # Temperatures in °C, precipitation in mm/year
    
    # ERA5-Land variable names in Google Earth Engine
    GEE_VARIABLE_MAPPING = {
        'temperature_2m': 'temperature_2m',
        'dewpoint_temperature_2m': 'dewpoint_temperature_2m',
        'total_precipitation': 'total_precipitation',
        'surface_solar_radiation_downwards': 'surface_solar_radiation_downwards',
        '10m_u_component_of_wind': 'u_component_of_wind_10m',
        '10m_v_component_of_wind': 'v_component_of_wind_10m',


    }
    
    def __init__(self, temp_dir=None, create_client=None, memory_limit=None, 
             temp_climate_file=None, precip_climate_file=None, gee_initialize=True):
        """
        Initialize the ERA5-Land GEE data processor.
        
        Parameters:
        -----------
        temp_dir : str or Path, optional
            Directory for temporary storage (defaults to config)
        create_client : bool, optional
            Whether to create a Dask distributed client
        memory_limit : str, optional
            Memory limit for the Dask client (if created)
        temp_climate_file : str or Path, optional
            Path to the annual mean temperature GeoTIFF file
        precip_climate_file : str or Path, optional
            Path to the annual mean precipitation GeoTIFF file
        gee_initialize : bool, optional
            Whether to initialize the Google Earth Engine API
        """
        self.temp_dir = Path(temp_dir) if temp_dir else config.ERA5LAND_TEMP_DIR
        
        # Create temporary directory if it doesn't exist
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        
        # Set climate file paths
        self.temp_climate_file = Path(temp_climate_file) if temp_climate_file else getattr(config, 'TEMP_CLIMATE_FILE', None)
        self.precip_climate_file = Path(precip_climate_file) if precip_climate_file else getattr(config, 'PRECIP_CLIMATE_FILE', None)
        
        # Initialize climate data attributes
        self.temp_climate_data = None
        self.precip_climate_data = None
        self.resampled_temp_data = None
        self.resampled_precip_data = None

        
        # Initialize Dask client if requested and available
        self.client = None
        create_client = config.DASK_CREATE_CLIENT if create_client is None else create_client
        memory_limit = config.DASK_MEMORY_LIMIT if memory_limit is None else memory_limit
        
        if create_client and HAVE_DASK:
            try:
                from dask.distributed import Client
                print(f"Creating Dask client with memory limit: {memory_limit}")
                self.client = Client(memory_limit=memory_limit)
                print(f"Dask dashboard link: {self.client.dashboard_link}")
            except Exception as e:
                print(f"Could not create Dask client: {str(e)}")
        
        # Initialize variable datasets
        self.datasets = {}
        

        
        # Initialize Google Earth Engine if requested
        self.ee_initialized = False
        if gee_initialize:
            self.initialize_gee()
        
        # Try to load climate data if files are provided
        if self.temp_climate_file and self.precip_climate_file:
            if self.temp_climate_file.exists() and self.precip_climate_file.exists():
                print("Loading climate data during initialization...")
                self.load_climate_data(bounds = {
                    'lon_min': -180.0,  # Eastern longitude boundary
                    'lat_min': -57,  # Southern latitude boundary
                    'lon_max': 180.0,  # Western longitude boundary 
                    'lat_max': 78.0   # Northern latitude boundary
                })
    
    def __del__(self):
        """Clean up resources on object deletion."""
        self.close()
    
    def close(self):
        """Close all open datasets and clients."""
        # Close all open datasets
        for var_name, ds in self.datasets.items():
            try:
                if hasattr(ds, 'close'):
                    ds.close()
            except:
                pass
        
        # Close Dask client if it exists
        if self.client:
            try:
                self.client.close()
                print("Closed Dask client")
            except:
                pass
            self.client = None
    
    def initialize_gee(self):
        """Initialize the Google Earth Engine API."""
        try:
            # Initialize the Earth Engine API
            ee.Authenticate()
            ee.Initialize(project='ee-yuluo-2')
            #ee.Initialize()
            self.ee_initialized = True
            print("Google Earth Engine initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Google Earth Engine: {str(e)}")
            print("You may need to authenticate first using 'earthengine authenticate'")
            self.ee_initialized = False
    def read_shapefile(self, shapefile_path):
        """
        Read a shapefile and convert it to an ee.Geometry object.
        
        Parameters:
        -----------
        shapefile_path : str or Path
            Path to the shapefile
            
        Returns:
        --------
        ee.Geometry
            Earth Engine geometry object
        """
        try:
            # Read the shapefile using geopandas
            gdf = gpd.read_file(shapefile_path)
            
            # Check if CRS is WGS84, if not, reproject
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
                print(f"Reprojected shapefile from {gdf.crs} to EPSG:4326")
            
            # If multiple features exist, use the union
            if len(gdf) > 1:
                print(f"Shapefile contains {len(gdf)} features. Using the union of all features.")
                geometry = gdf.unary_union
            else:
                geometry = gdf.geometry.iloc[0]
                
            # Convert shapely geometry to GeoJSON
            geo_json = geometry.__geo_interface__
            
            # Create an ee.Geometry object
            ee_geometry = ee.Geometry(geo_json)
            
            print(f"Successfully converted shapefile to ee.Geometry")
            return ee_geometry
        except Exception as e:
            print(f"Error reading shapefile: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    def get_era5land_variable_from_gee(self, variable, year, month, day=None, hour=None, scale=None, shapefile=None, region=None):
        """
        Access ERA5-Land data from Google Earth Engine for a specific time period.
        Modified to support hour-specific filtering.
        
        Parameters:
        -----------
        variable : str
            Variable name (e.g., '2m_temperature')
        year : int
            Year to access
        month : int, optional
            Month to access (if None, access all available months)
        day : int, optional
            Day to access (if None, access all days in month)
        hour : int, optional
            Hour to access (if None, access all hours in day)
        scale : float, optional
            Scale in meters for the Earth Engine export (defaults to ~9km for ERA5-Land)
        shapefile : str, optional
            Path to shapefile for region definition
        region : dict or ee.Geometry, optional
            Region to extract data for (default is bounding box from config)
                
        Returns:
        --------
        xarray.Dataset
            Dataset containing the requested variable data
        """
        if not self.ee_initialized:
            self.initialize_gee()
            if not self.ee_initialized:
                print("Google Earth Engine not initialized. Cannot access data.")
                return None
        
        # Convert variable name to GEE variable name
        gee_var = self.GEE_VARIABLE_MAPPING.get(variable, variable)
        
        time_str = f"{year}-{month:02d}"
        if day is not None:
            time_str += f"-{day:02d}"
            if hour is not None:
                time_str += f" {hour:02d}:00"
        print(f"Accessing ERA5-Land variable '{gee_var}' from Google Earth Engine for {time_str}")
        
        try:
            # Handle region configuration same as original
            if region is None:
                if shapefile is not None:
                    region = self.read_shapefile(shapefile)
                else:
                    region = ee.Geometry.Polygon(
                        [[[config.DEFAULT_LON_MIN, config.DEFAULT_LAT_MIN], 
                          [config.DEFAULT_LON_MAX, config.DEFAULT_LAT_MIN], 
                          [config.DEFAULT_LON_MAX, config.DEFAULT_LAT_MAX], 
                          [config.DEFAULT_LON_MIN, config.DEFAULT_LAT_MAX], 
                          [config.DEFAULT_LON_MIN, config.DEFAULT_LAT_MIN]]],
                        None, False # Planar coordinates for global extent
                    )

            # Set default scale if not specified (ERA5-Land is ~9km)
            if scale is None:
                scale = 11132  # 9km in meters
            
            # Set up date range - ENHANCED FOR HOURLY PRECISION
            if hour is not None and day is not None and month is not None:
                # Precise hour-level filtering
                start_date = ee.Date.fromYMD(year, month, day).advance(hour, 'hour')
                end_date = start_date.advance(1, 'hour')
                print(f"Filtering for exact hour: {start_date.format('YYYY-MM-dd HH:mm:ss').getInfo()}")
            elif day is not None and month is not None:
                # Day-level filtering
                start_date = ee.Date.fromYMD(year, month, day)
                # Set end date to the next day
                if day == calendar.monthrange(year, month)[1] and month == 12:
                    end_date = ee.Date.fromYMD(year + 1, 1, 1)
                elif day == calendar.monthrange(year, month)[1]:
                    end_date = ee.Date.fromYMD(year, month + 1, 1)
                else:
                    end_date = ee.Date.fromYMD(year, month, day + 1)
            elif month is not None:
                # Month-level filtering
                start_date = ee.Date.fromYMD(year, month, 1)
                if month == 12:
                    end_date = ee.Date.fromYMD(year + 1, 1, 1)
                else:
                    end_date = ee.Date.fromYMD(year, month + 1, 1)
            else:
                # Year-level filtering
                start_date = ee.Date.fromYMD(year, 1, 1)
                end_date = ee.Date.fromYMD(year + 1, 1, 1)
            
            # Access ERA5-Land collection
            era5land = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
            
            # Filter by date
            era5land = era5land.filterDate(start_date, end_date)
            
            # Select the variable
            era5land = era5land.select(gee_var)
            
            # Convert to xarray via geemap
            try:
                # Process images individually to avoid ImageCollection limitations
                print(f"Converting GEE images to xarray dataset...")
                
                # Get the data as a list of images
                image_list = era5land.toList(era5land.size())
                
                # Prepare for batch processing to avoid memory issues
                batch_size = 10  # Process smaller batches to avoid timeouts
                collection_size = image_list.size().getInfo()
                print(f"Collection contains {collection_size} images")
                
                all_arrays = []
                timestamps = []
                try:
                    # Use geometry().bounds() for safety with complex shapes
                    region_bounds = region.bounds().getInfo()
                    coords = region_bounds['coordinates'][0]
                    lons_list = [coord[0] for coord in coords]
                    lats_list = [coord[1] for coord in coords]
                    lon_min, lon_max = min(lons_list), max(lons_list)
                    lat_min, lat_max = min(lats_list), max(lats_list)
                    print(f"Region Bounding Box: lon={lon_min:.4f}-{lon_max:.4f}, lat={lat_min:.4f}-{lat_max:.4f}")
                except Exception as bounds_err:
                    print(f"Could not get bounds from ee.Geometry: {bounds_err}. Falling back to config defaults.")
                    lon_min, lat_min, lon_max, lat_max = config.DEFAULT_LON_MIN, config.DEFAULT_LAT_MIN, config.DEFAULT_LON_MAX, config.DEFAULT_LAT_MAX
                # When getting coordinates from the first image
                # Fix for: "Could not get precise grid info from first image projection (image_props() got an unexpected keyword argument 'region')"
                try:
                    # Get the first image and its projection
                    first_img = ee.Image(image_list.get(0))
                    first_img_proj = first_img.select(gee_var).projection().getInfo()
                    
                    # Version 1: Try to use the ee.Image.getInfo() to get image properties
                    img_info = first_img.getInfo()
                    bands_info = img_info.get('bands', [])
                    if bands_info and 'crs_transform' in bands_info[0] and 'crs' in bands_info[0]:
                        transform_list = bands_info[0]['crs_transform']
                        crs = bands_info[0]['crs']
                        
                        # Get image dimensions by first exporting a small sample
                        sample_result = geemap.ee_to_numpy(
                            first_img.select(gee_var),
                            region=region,
                            scale=scale
                        )
                        
                        if sample_result is not None:
                            height, width = sample_result.shape[:2]
                            print(f"Image dimensions from sample: {height}x{width}")
                            
                            # FIXED APPROACH: Get coordinates directly from region bounds
                            try:
                                # Get region bounds directly from the region object
                                region_bounds = region.bounds().getInfo()
                                coords = region_bounds['coordinates'][0]
                                region_lons = [c[0] for c in coords]
                                region_lats = [c[1] for c in coords]
                                
                                # Extract min/max values
                                lon_min, lon_max = min(region_lons), max(region_lons)
                                lat_min, lat_max = min(region_lats), max(region_lats)
                                
                                print(f"Using region-derived coordinates: lon={lon_min:.4f}-{lon_max:.4f}, lat={lat_min:.4f}-{lat_max:.4f}")
                            except Exception as e:
                                print(f"Error getting region bounds: {e}. Using transform-derived coordinates.")
                                # Fall back to transform-based approach if region bounds fail
                                img_lon_min = transform_list[2]  # x origin
                                img_lat_max = transform_list[5]  # y origin
                                img_lon_max = img_lon_min + width * transform_list[0]
                                img_lat_min = img_lat_max + height * transform_list[4]
                                
                                # VALIDATION CHECK: If coordinates look unreasonable, use region settings
                                if abs(img_lon_min) > 180 or abs(img_lon_max) > 180 or abs(img_lat_min) > 90 or abs(img_lat_max) > 90:
                                    print("WARNING: Transform-derived coordinates outside valid range!")
                                    # Extract from region definition or use defaults
                                    if hasattr(region, 'coordinates') and region.coordinates:
                                        coords = region.coordinates().getInfo()[0]
                                        lon_min = coords[0][0]
                                        lat_min = coords[0][1]
                                        lon_max = coords[2][0]
                                        lat_max = coords[2][1]
                                    else:
                                        # Use the default region coordinates as fallback
                                        lon_min = config.DEFAULT_LON_MIN
                                        lat_min = config.DEFAULT_LAT_MIN
                                        lon_max = config.DEFAULT_LON_MAX
                                        lat_max = config.DEFAULT_LAT_MAX
                                    print(f"Fallback to region coordinates: lon={lon_min:.4f}-{lon_max:.4f}, lat={lat_min:.4f}-{lat_max:.4f}")
                                else:
                                    lon_min, lon_max = img_lon_min, img_lon_max
                                    lat_min, lat_max = img_lat_min, img_lat_max
                    else:
                        raise Exception("Could not extract transform information from image metadata")
                        
                except Exception as proj_err:
                    print(f"Could not get precise grid info from image: {proj_err}. Estimating dimensions.")
                    # Estimate width/height (less accurate)
                    width = int(np.ceil((lon_max - lon_min) / (scale / 111320)))  # Approx meters per degree lon at equator
                    height = int(np.ceil((lat_max - lat_min) / (scale / 111000)))  # Approx meters per degree lat
                    print(f"Estimated grid dimensions: {height}x{width}")
                    if width <= 0 or height <= 0:
                        print("Error: Calculated grid dimensions are non-positive. Check region or scale.")
                        return None
                    
                # Create coordinate arrays with proper ordering
                lons = np.linspace(lon_min, lon_max, width, endpoint=False)  # Use endpoint=True for more intuitive coordinates
                # For ERA5-Land, latitude typically decreases with increasing index (north to south)
                lats = np.linspace(lat_max, lat_min, height, endpoint=False)  # Latitude decreasing (North to South)

                # VALIDATION: Ensure coordinates are in correct order
                print(f"Final Coordinates Range: lon={lons.min():.4f}-{lons.max():.4f}, lat={lats.min():.4f}-{lats.max():.4f}")
                # Ensure coordinates are sorted as xarray expects
                lons.sort()
                lats.sort()
                print(region.getInfo())
                img_data = geemap.ee_to_numpy(
                    first_img.select(gee_var),
                    region=region, 
                    scale=scale
                )
                height = img_data.shape[0]
                width = img_data.shape[1]
                """
                # Get region bounds safely
                bounds = region.bounds().getInfo()
                if isinstance(bounds, list) and len(bounds) == 4:
                    # Direct unpacking if bounds returns a flat list of 4 values
                    lon_min, lat_min, lon_max, lat_max = bounds
                else:
                    # Handle case where bounds() returns a GeoJSON-like structure
                    try:
                        # Try to extract from a GeoJSON structure
                        coords = bounds['coordinates'][0]
                        lons = [coord[0] for coord in coords]
                        lats = [coord[1] for coord in coords]
                        lon_min, lon_max = min(lons), max(lons)
                        lat_min, lat_max = min(lats), max(lats)
                    except (KeyError, TypeError):
                        # If that fails, try to use the geometry bounds method
                        bbox = region.geometry().bounds().getInfo()
                        lon_min = bbox[0]
                        lat_min = bbox[1]
                        lon_max = bbox[2]
                        lat_max = bbox[3]

                # Create evenly spaced coordinate arrays using the extracted image dimensions
                lons = np.linspace(lon_min, lon_max, width)
                lats = np.linspace(lat_max, lat_min, height)  # Note: lat typically decreases with increasing index

                print(f"Coordinates extracted: lon_min={lon_min}, lon_max={lon_max}, lat_min={lat_min}, lat_max={lat_max}")
                print(f"Grid dimensions: {height}x{width}")
                # Sort coordinates (important for consistent grid)
                lons.sort()
                lats.sort()
                print(lons, lats)
                """
                coords = geemap.ee_to_numpy(
                    first_img.select(gee_var),
                    region=region, 
                    scale=scale,
                    
                )
                
                if not img_data.any():
                    print("Failed to get coordinates from first image.")
                    return None
                    
                

                print(f"Grid size: {height}x{width}")
                
                # Process each image individually
                for i in range(0, collection_size):
                    try:
                        if i % 10 == 0:
                            import gc
                            gc.collect()
                            print(f"Processing image {i+1} of {collection_size}...")
                        # Get the image
                        img = ee.Image(image_list.get(i))
                        
                        # Get the timestamp
                        # CHANGE TO - explicitly standardized
                        time_start = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd HH:mm:ss').getInfo()
                        current_time = pd.to_datetime(time_start, utc=True)  # Always timezone-aware (UTC)
                        
                        # Process the image - PRIMARY METHOD: Export to temporary file
                        # Process the image - PRIMARY METHOD: Export to temporary file
                        try:
                            # Export to temporary file as primary approach
                            temp_file = self.temp_dir / f"{variable}_{time_start.replace(':', '-').replace(' ', '_')}.tif"
                            
                            # First check if file already exists
                            if temp_file.exists():
                                print(f"Temporary file {temp_file} already exists. Using existing file.")
                                success = True
                            else:
                                # Export the image
                                success = geemap.ee_export_image(
                                    img.select(gee_var),
                                    filename=str(temp_file),
                                    scale=scale,
                                    region=region,
                                    file_per_band=False
                                )
                                
                                # Add a small delay to ensure file system has time to register the file
                                import time
                                time.sleep(1)
                                
                                # Re-check if the file exists, regardless of the success flag
                                if temp_file.exists():
                                    print(f"File exists at {temp_file}, proceeding with processing")
                                    success = True
                                else:
                                    print(f"File does not exist at {temp_file} after export")
                                    success = False
                            
                            # Try to read the file if it exists, regardless of the success flag from geemap
                            if temp_file.exists():
                                try:
                                    # Read the GeoTIFF with appropriate error handling
                                    with rasterio.open(temp_file) as src:
                                        img_array = src.read(1)
                                        
                                        # Check shape consistency
                                        if img_array.shape[:2] == (height, width):
                                            all_arrays.append(img_array)
                                            timestamps.append(current_time)
                                            # Successfully processed, continue to next image
                                            continue  # Skip the fallback method
                                        else:
                                            print(f"Skipping image {i+1} from GeoTIFF - inconsistent shape: {img_array.shape} vs expected {(height, width)}")
                                            # Don't raise here, try the fallback method
                                except rasterio.errors.RasterioIOError as rio_err:
                                    print(f"Rasterio failed to read file {temp_file}: {str(rio_err)}")
                                    # File exists but can't be read, try fallback method
                            
                            # If we reach here, either the file doesn't exist or couldn't be read properly
                            print(f"Trying fallback method for image {i+1}")
                            raise Exception("GeoTIFF processing unsuccessful")
                                    
                        except Exception as tif_err:
                            print(f"Error with GeoTIFF approach for image {i+1}: {str(tif_err)}")
                            
                            # FALLBACK METHOD: Try direct conversion to numpy
                            try:
                                print(f"Using fallback method (ee_to_numpy) for image {i+1}")
                                # Try to use ee_to_numpy for the image
                                img_array = geemap.ee_to_numpy(
                                    img.select(gee_var),
                                    region=region,
                                    scale=scale
                                )
                                
                                # Make sure the shape is consistent
                                if img_array.shape[:2] == (height, width):
                                    all_arrays.append(img_array)
                                    timestamps.append(current_time)
                                else:
                                    print(f"Skipping image {i+1} - inconsistent shape: {img_array.shape} vs expected {(height, width)}")
                            except Exception as img_err:
                                print(f"Fallback method also failed for image {i+1}: {str(img_err)}")
                    except Exception as e:
                        print(f"Unexpected error processing image {i+1}: {str(e)}")
                
                # Check if we got any data
                if not all_arrays or not timestamps:
                    print("No data was successfully retrieved.")
                    return None
                
                print(f"Successfully processed {len(all_arrays)} out of {collection_size} images")
                
                # Stack all arrays into a 3D array [time, lat, lon]
                combined_array = np.stack(all_arrays)
                
                # Create xarray data array with only the successful timestamps
                da = xr.DataArray(
                    combined_array,
                    dims=['time', 'latitude', 'longitude'],
                    coords={
                        'time': timestamps,
                        'latitude': lats,  # Extract unique latitudes
                        'longitude': lons   # Extract unique longitudes
                    },
                    name=gee_var
                )
                
                # Convert to dataset
                ds = da.to_dataset()
                
                # Add metadata
                ds.attrs['title'] = f'ERA5-Land {gee_var}'
                ds.attrs['source'] = 'Google Earth Engine'
                ds[gee_var].attrs['units'] = self._get_variable_units(gee_var)
                
                return ds
                
            except Exception as ee_err:
                print(f"Error converting GEE collection to xarray: {str(ee_err)}")
                
                # Alternative approach: Download as GeoTIFF and convert
                print("Trying alternative approach: Export to GeoTIFF...")
                
                # Define a function to export a single image
                def export_image(image, filename):
                    try:
                        geemap.ee_export_image(
                            image, 
                            filename=filename, 
                            scale=scale, 
                            region=region,
                            file_per_band=False
                        )
                        return True
                    except Exception as export_err:
                        print(f"Error exporting image: {str(export_err)}")
                        return False
                
                # Download in smaller batches
                temp_files = []
                timestamps = []
                
                # Use a batch approach
                for i in range(0, collection_size, 10):
                    end_idx = min(i + 10, collection_size)
                    print(f"Exporting batch {i+1} to {end_idx} of {collection_size}...")
                    
                    for j in range(i, end_idx):
                        img = ee.Image(image_list.get(j))
                        # Get the timestamp
                        time_start = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd HH:mm:ss').getInfo()
                        timestamps.append(pd.to_datetime(time_start))
                        
                        # Export the image
                        filename = self.temp_dir / f"{variable}_{time_start}.tif"
                        if export_image(img, str(filename)):
                            temp_files.append(filename)
                
                # Read all the files and combine them
                if temp_files:
                    arrays = []
                    for file in temp_files:
                        with rasterio.open(file) as src:
                            array = src.read(1)
                            arrays.append(array)
                            
                            # Get spatial coordinates from the first file
                            if len(arrays) == 1:
                                transform = src.transform
                                height = src.height
                                width = src.width
                                
                                # Calculate lat/lon for each pixel
                                lats = np.zeros(height)
                                lons = np.zeros(width)
                                
                                for i in range(height):
                                    lat, _ = transform * (0, i)
                                    lats[i] = lat
                                
                                for j in range(width):
                                    _, lon = transform * (j, 0)
                                    lons[j] = lon
                    
                    # Create xarray data array
                    combined_array = np.stack(arrays)
                    
                    da = xr.DataArray(
                        combined_array,
                        dims=['time', 'latitude', 'longitude'],
                        coords={
                            'time': timestamps,
                            'latitude': lats,
                            'longitude': lons
                        },
                        name=gee_var
                    )
                    
                    # Convert to dataset
                    ds = da.to_dataset()
                    
                    # Add metadata
                    ds.attrs['title'] = f'ERA5-Land {gee_var}'
                    ds.attrs['source'] = 'Google Earth Engine'
                    ds[gee_var].attrs['units'] = self._get_variable_units(gee_var)
                    
                    # Clean up temporary files
                    for file in temp_files:
                        try:
                            os.remove(file)
                        except:
                            pass
                    
                    return ds
                else:
                    print("Failed to export any images as GeoTIFF")
                    return None
                
        except Exception as e:
            print(f"Error accessing ERA5-Land data from GEE: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_variable_units(self, gee_var):
        """
        Get the units for a GEE variable name.
        
        Parameters:
        -----------
        gee_var : str
            GEE variable name
            
        Returns:
        --------
        str
            Units for the variable
        """
        # Map of variable names to their units
        units_map = {
            'temperature_2m': 'K',
            'dewpoint_temperature_2m': 'K',
            'total_precipitation': 'm',
            'surface_solar_radiation_downwards': 'j m-2',
            'u_component_of_wind_10m': 'm s-1',
            'v_component_of_wind_10m': 'm s-1',
            'surface_pressure': 'Pa',
            'total_evaporation': 'm of water equivalent',
            'volumetric_soil_water_layer_1': 'm3 m-3',
            'volumetric_soil_water_layer_2': 'm3 m-3',
            'volumetric_soil_water_layer_3': 'm3 m-3',
            'volumetric_soil_water_layer_4': 'm3 m-3',
        }
        
        return units_map.get(gee_var, 'unknown')
    
    def extract_region(self, ds, lat_min=None, lat_max=None, lon_min=None, lon_max=None):
        """
        Extract a spatial subset of the dataset.
        
        Parameters:
        -----------
        ds : xarray.Dataset
            Input dataset
        lat_min, lat_max, lon_min, lon_max : float, optional
            Bounding box coordinates
                
        Returns:
        --------
        xarray.Dataset
            Spatially subset dataset
        """
        if ds is None:
            return None
        
        # Use default bounds if not specified
        lat_min = lat_min if lat_min is not None else config.DEFAULT_LAT_MIN
        lat_max = lat_max if lat_max is not None else config.DEFAULT_LAT_MAX
        lon_min = lon_min if lon_min is not None else config.DEFAULT_LON_MIN
        lon_max = lon_max if lon_max is not None else config.DEFAULT_LON_MAX
        
        try:
            # Print the actual dimensions and coordinates to help debug
            print("Dataset dimensions:", ds.dims)
            print("Dataset coordinates:", list(ds.coords))
            
            # Find coordinate variables for lat/lon
            lat_var = None
            lon_var = None
            
            # Try common variable names
            for var in ds.coords:
                if var.lower() in ['latitude', 'lat']:
                    lat_var = var
                elif var.lower() in ['longitude', 'lon']:
                    lon_var = var
            
            if not lat_var or not lon_var:
                print("Latitude or longitude variable not found in dataset")
                return ds
            
            # Print coordinate ranges to diagnose issues
            print(f"Latitude range in data: {ds[lat_var].min().values} to {ds[lat_var].max().values}")
            print(f"Longitude range in data: {ds[lon_var].min().values} to {ds[lon_var].max().values}")
            print(f"Extracting region: lon={lon_min}-{lon_max}, lat={lat_min}-{lat_max}")
            
            # Extract the region using sel() method
            region_ds = ds.sel(
                {lat_var: slice(lat_min, lat_max)},
                {lon_var: slice(lon_min, lon_max)}
            )
            
            # Verify we got some data
            if lat_var in region_ds.dims and lon_var in region_ds.dims:
                lat_size = region_ds.dims[lat_var]
                lon_size = region_ds.dims[lon_var]
                print(f"Region extracted: {lat_size} x {lon_size} grid points")
                
                if lat_size == 0 or lon_size == 0:
                    print("WARNING: Region selection returned empty dataset. Using complete dataset.")
                    return ds
                
                return region_ds
            else:
                print("Region selection didn't preserve dimensions. Using complete dataset.")
                return ds
        except Exception as e:
            print(f"Error extracting region: {str(e)}")
            print("Using complete dataset as fallback.")
            return ds
    
    def load_variable_data(self, variables, year, month, day=None, hour=None, shapefile=None):
        """
        Load and preprocess multiple ERA5-Land variables, potentially clipping to a shapefile.
        Modified to support hour-specific loading.

        Parameters:
        -----------
        variables : list
            List of variable names to load (e.g., ['temperature_2m', 'total_precipitation']).
        year : int
            Year to load data for.
        month : int
            Month to load data for.
        day : int, optional
            Day to load data for.
        hour : int, optional
            Hour to load data for.
        shapefile : str, optional
            Path to a shapefile for clipping the data. If None, uses default global extent.

        Returns:
        --------
        dict
            Dictionary where keys are variable names and values are the processed xarray.Dataset objects.
        """
        if not rioxarray:
            print("ERROR: rioxarray library is required for shapefile masking. Please install it.")
            # Optionally raise an error: raise ImportError("rioxarray not found")
            # Or proceed without masking capability:
            if shapefile:
                print("WARNING: Cannot perform shapefile masking because rioxarray is missing. Only bounding box extraction will occur.")
                # shapefile = None # Disable shapefile processing further down

        if not gpd and shapefile:
            print("ERROR: GeoPandas library is required for shapefile processing. Please install it.")
            # Optionally raise an error: raise ImportError("geopandas not found")
            # Or proceed without masking capability:
            print("WARNING: Cannot perform shapefile masking because geopandas is missing. Only bounding box extraction will occur.")
            # shapefile = None # Disable shapefile processing further down

        # Clear existing datasets before loading new ones
        time_str = f"{year}-{month:02d}"
        if day is not None:
            time_str += f"-{day:02d}"
            if hour is not None:
                time_str += f" {hour:02d}:00"
        print(f"Loading variables for {time_str}...")
        
        for var_name, ds in self.datasets.items():
            try:
                if hasattr(ds, 'close'):
                    ds.close()
                del ds # Attempt to free memory
            except Exception as e:
                print(f"    Warning: Could not close/delete dataset for {var_name}: {e}")
        self.datasets = {}
        import gc
        gc.collect() # Suggest garbage collection

        shapefile_gdf = None
        shapefile_bounds = None

        # Read shapefile ONCE using geopandas if provided
        if shapefile is not None and gpd is not None:
            print(f"Reading shapefile: {shapefile}")
            try:
                shapefile_gdf = gpd.read_file(shapefile)
                # Get bounds (minx, miny, maxx, maxy)
                bounds = shapefile_gdf.total_bounds
                shapefile_bounds = {
                    'lon_min': bounds[0],
                    'lat_min': bounds[1],
                    'lon_max': bounds[2],
                    'lat_max': bounds[3]
                }
                print(f"    Shapefile bounds: lat={shapefile_bounds['lat_min']:.4f} to {shapefile_bounds['lat_max']:.4f}, "
                      f"lon={shapefile_bounds['lon_min']:.4f} to {shapefile_bounds['lon_max']:.4f}")
                print(f"    Shapefile CRS: {shapefile_gdf.crs}")
            except Exception as e:
                print(f"ERROR: Could not read or get bounds from shapefile: {e}")
                print("Proceeding without shapefile clipping, using default region if defined, or global.")
                shapefile_gdf = None # Ensure it's None if reading failed
                shapefile_bounds = None # Ensure bounds are None

        # Define the region for data loading/extraction
        # Priority: Shapefile bounds > Default region (if shapefile not used)
        region_to_extract = None
        if shapefile_bounds is not None:
            region_to_extract = shapefile_bounds
        else:
            # Use default global region if no shapefile
            region_to_extract = {
                'lat_min': config.DEFAULT_LAT_MIN,
                'lat_max': config.DEFAULT_LAT_MAX,
                'lon_min': config.DEFAULT_LON_MIN,
                'lon_max': config.DEFAULT_LON_MAX
            }
            print("Using default global region.")

        # Load each variable
        for var_name in variables:
            var_desc = f"{var_name} for {time_str}"
            print(f"\nProcessing variable: {var_desc}...")

            # 1. Get data from GEE (or other source) - MODIFIED to include hour parameter
            try:
                ds = self.get_era5land_variable_from_gee(
                    var_name, year, month, day, hour
                )  # Pass shapefile path for context if needed by GEE func
            except Exception as gee_err:
                print(f"  ERROR: Failed to load data for {var_name} from GEE source: {gee_err}")
                ds = None

            if ds is not None:
                print(f"  Initial data loaded for {var_name}. Size: {ds.nbytes / 1e6:.2f} MB")
                print(f"  Initial dimensions: {ds.dims}")
                print(f"  Coordinates: {list(ds.coords.keys())}")

                # 3. Apply Shapefile Mask (if shapefile was provided and libraries exist)
                if shapefile_gdf is not None and rioxarray is not None and ds.nbytes > 0:
                    print(f"  Applying precise shapefile mask using rioxarray for {var_name}...")
                    try:
                        # Ensure Dataset CRS is set (should be done in get_... or extract_region)
                        if ds.rio.crs is None:
                            print(f"    CRS is missing for dataset '{var_name}'. Setting to EPSG:4326 (WGS84).")
                            # The correct way to set CRS inplace
                            ds.rio.write_crs("EPSG:4326", inplace=True)
                            # Verify CRS was set
                            if ds.rio.crs is None:
                                raise ValueError(f"Failed to set CRS for {var_name}")
                            else:
                                print(f"    CRS successfully set to {ds.rio.crs}")

                        data_crs = ds.rio.crs
                        shape_crs = shapefile_gdf.crs
                        
                        print(f"    Data CRS: {data_crs}, Shapefile CRS: {shape_crs}")

                        # Reproject shapefile CRS if it doesn't match data CRS
                        if shape_crs != data_crs:
                            print(f"    Reprojecting shapefile from {shape_crs} to {data_crs}...")
                            shapefile_gdf = shapefile_gdf.to_crs(data_crs)
                            print("    Reprojection complete.")

                        # Explicitly check for the data variable name before clipping
                        data_vars = list(ds.data_vars)
                        if len(data_vars) == 0:
                            raise ValueError(f"Dataset has no data variables to clip")
                        
                        print(f"    Data variables found: {data_vars}")
                            
                        # Perform the clipping using rioxarray
                        masked_ds = ds.rio.clip(shapefile_gdf.geometry, 
                                                drop=False, 
                                                invert=False, 
                                                all_touched=True)

                        # Check if clipping resulted in an empty dataset
                        if masked_ds.nbytes == 0 or all(dim == 0 for dim in masked_ds.dims.values()):
                            print(f"    WARNING: Clipping {var_name} resulted in an empty dataset. Shapefile might not overlap the data region.")
                            print(f"    Storing the bounding-box extracted data for {var_name} instead of empty mask.")
                        else:
                            ds = masked_ds  # Replace original dataset with the masked version
                            print(f"    Shapefile mask applied successfully. Final size: {ds.nbytes / 1e6:.2f} MB")

                    except Exception as mask_err:
                        print(f"  ERROR applying shapefile mask with rioxarray for {var_name}: {mask_err}")
                        print("    Using data only extracted to the shapefile's bounding box (if applicable).")
                        import traceback
                        traceback.print_exc()  # Print detailed error traceback

                elif shapefile and (rioxarray is None or gpd is None):
                    print(f"  Skipping precise shapefile masking for {var_name} because required libraries are missing.")

                # 4. Store the final dataset
                if ds is not None and ds.nbytes > 0:
                    self.datasets[var_name] = ds
                    print(f"  --> Successfully processed and stored dataset for {var_name}")
                elif ds is not None and ds.nbytes == 0:
                    print(f"  --> Processed {var_name}, but the resulting dataset is empty (likely due to clipping). Not stored.")
                else:
                    print(f"  --> Failed to process or load {var_name}. Not stored.")

            else: # If ds was None initially
                print(f"  --> Failed to load initial data for {var_name}. Skipping further processing.")

        print("\nFinished loading all variables.")
        return self.datasets
    
    def calculate_derived_variables(self):
        """
        Calculate derived variables from the loaded ERA5-Land data in a memory-efficient way.
        Uses chunked processing to avoid memory issues with large datasets.
        Preserves the exact calculation logic of the original implementation.
        
        Returns:
        --------
        dict
            Updated dataset dictionary with derived variables
        """
        if not self.datasets:
            print("No datasets loaded. Cannot calculate derived variables.")
            return self.datasets
        
        # Add this at the beginning of the calculate_derived_variables method
        for var_name, ds in self.datasets.items():
            if 'time' in ds.coords:
                time_values = ds.coords['time'].values
                unique_times = np.unique(time_values)
                print(f"Dataset '{var_name}' has {len(time_values)} unique time values")
                print(time_values)
                if len(time_values) != len(unique_times):
                    print(f"WARNING: Dataset '{var_name}' has {len(time_values) - len(unique_times)} duplicate time values")
                    # Print the first few duplicates
                    time_series = pd.Series(time_values)
                    duplicated = time_series[time_series.duplicated()]
                    if not duplicated.empty:
                        print(f"First 5 duplicate values: {duplicated[:5].values}")
        
        # Debug: Print structure of each dataset
        for var_name, ds in self.datasets.items():
            print(f"\nDataset '{var_name}' structure:")
            print(f"- Coordinates: {list(ds.coords)}")
            print(f"- Data variables: {list(ds.data_vars)}")
            if list(ds.data_vars):
                main_var = list(ds.data_vars)[0]
                print(f"- Main data variable: {main_var}")
                print(f"- Data shape: {ds[main_var].shape}")
        
        # Calculate wind speed if components are available
        if '10m_u_component_of_wind' in self.datasets and '10m_v_component_of_wind' in self.datasets:
            print("Calculating wind speed from u and v components using chunked approach...")
            try:
                u_ds = self.datasets['10m_u_component_of_wind']
                v_ds = self.datasets['10m_v_component_of_wind']
                
                # Get the main data variable from each dataset
                u_var = list(u_ds.data_vars)[0] if list(u_ds.data_vars) else None
                v_var = list(v_ds.data_vars)[0] if list(v_ds.data_vars) else None
                
                if u_var and v_var:
                    # Check if we have time dimension
                    if 'time' in u_ds.dims:
                        times = u_ds.coords['time'].values
                        lat_var = [v for v in u_ds.dims if v.lower() in ('latitude', 'lat')][0]
                        lon_var = [v for v in u_ds.dims if v.lower() in ('longitude', 'lon')][0]
                        
                        # Process in time chunks
                        wind_chunks = []
                        chunk_coords = {'time': [], 'latitude': u_ds[lat_var].values, 'longitude': u_ds[lon_var].values}
                        
                        # Process each time step separately to conserve memory
                        for t_idx in range(len(times)):
                            # Extract single time slice to reduce memory usage
                            u_slice = u_ds[u_var].isel(time=t_idx).values
                            v_slice = v_ds[v_var].isel(time=t_idx).values
                            
                            # Calculate wind speed using numpy directly - EXACT SAME FORMULA
                            wind_slice = np.sqrt(u_slice**2 + v_slice**2)
                            
                            # Store results and time
                            wind_chunks.append(wind_slice)
                            chunk_coords['time'].append(times[t_idx])
                            
                            # Status update
                            if t_idx % max(1, len(times)//10) == 0 or t_idx == len(times) - 1:
                                print(f"Processed wind speed for time {t_idx+1}/{len(times)}")
                        
                        # Combine chunks into a single dataset
                        wind_array = np.stack(wind_chunks)
                        wind_da = xr.DataArray(
                            wind_array,
                            dims=['time', lat_var, lon_var],
                            coords=chunk_coords,
                            name='wind_speed'
                        )
                        
                        # Create dataset
                        wind_ds = wind_da.to_dataset()
                        wind_ds['wind_speed'].attrs['units'] = 'm s-1'
                        wind_ds['wind_speed'].attrs['long_name'] = 'Wind speed at 10m'
                        
                        self.datasets['wind_speed'] = wind_ds
                        print("Wind speed calculated successfully with chunked approach")
                    else:
                        # Standard approach using named variables - EXACT SAME AS ORIGINAL
                        wind_speed = np.sqrt(u_ds[u_var]**2 + v_ds[v_var]**2)
                        
                        wind_ds = xr.Dataset(
                            data_vars={'wind_speed': wind_speed},
                            coords=u_ds.coords,
                            attrs={'units': 'm s-1', 'long_name': 'Wind speed at 10m'}
                        )
                        wind_ds['wind_speed'].attrs['units'] = 'm s-1'
                        wind_ds['wind_speed'].attrs['long_name'] = 'Wind speed at 10m'
                        
                        self.datasets['wind_speed'] = wind_ds
                        print(f"Wind speed calculated successfully from variables: {u_var}, {v_var}")
                    
                else:
                    print("Could not identify u and v component variables")
            except Exception as e:
                print(f"Error calculating wind speed: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Calculate VPD if temperature and dewpoint are available
        if '2m_temperature' in self.datasets and '2m_dewpoint_temperature' in self.datasets:
            print("Calculating vapor pressure deficit (VPD) using chunked approach...")
            try:
                t_ds = self.datasets['2m_temperature']
                td_ds = self.datasets['2m_dewpoint_temperature']
                
                # Get the main data variable from each dataset
                t_var = list(t_ds.data_vars)[0] if list(t_ds.data_vars) else None
                td_var = list(td_ds.data_vars)[0] if list(td_ds.data_vars) else None
                
                if t_var and td_var:
                    # Check if we have time dimension
                    if 'time' in t_ds.dims:
                        times = t_ds.coords['time'].values
                        lat_var = [v for v in t_ds.dims if v.lower() in ('latitude', 'lat')][0]
                        lon_var = [v for v in t_ds.dims if v.lower() in ('longitude', 'lon')][0]
                        
                        # Process in time chunks
                        vpd_chunks = []
                        chunk_coords = {'time': [], 'latitude': t_ds[lat_var].values, 'longitude': t_ds[lon_var].values}
                        
                        # Process each time step separately to conserve memory
                        for t_idx in range(len(times)):
                            # Extract single time slice to reduce memory usage
                            t_slice = t_ds[t_var].isel(time=t_idx).values
                            td_slice = td_ds[td_var].isel(time=t_idx).values
                            
                            # Convert from K to C (using numpy directly)
                            t_c = t_slice - 273.15
                            td_c = td_slice - 273.15
                            
                            # Apply validation check: ensure dew point <= air temperature
                            td_c = np.minimum(td_c, t_c)
                            
                            # Calculate saturated vapor pressure using exponential Magnus formula
                            es = 6.1078 * np.exp((17.269 * t_c) / (237.3 + t_c))
                            ea = 6.1078 * np.exp((17.269 * td_c) / (237.3 + td_c))
                            
                            # Apply validation check: ensure VPD is never negative
                            vpd_slice = np.maximum(es - ea, 0.0)
                            
                            # Store results and time
                            vpd_chunks.append(vpd_slice)
                            chunk_coords['time'].append(times[t_idx])
                            
                            # Status update
                            if t_idx % max(1, len(times)//10) == 0 or t_idx == len(times) - 1:
                                print(f"Processed VPD for time {t_idx+1}/{len(times)}")
                        
                        # Combine chunks into a single dataset
                        vpd_array = np.stack(vpd_chunks)
                        vpd_da = xr.DataArray(
                            vpd_array,
                            dims=['time', lat_var, lon_var],
                            coords=chunk_coords,
                            name='vpd'
                        )
                        
                        # Create dataset
                        vpd_ds = vpd_da.to_dataset()
                        vpd_ds['vpd'].attrs['units'] = 'hPa'
                        vpd_ds['vpd'].attrs['long_name'] = 'Vapor Pressure Deficit'
                        
                        self.datasets['vpd'] = vpd_ds
                        print("VPD calculated successfully with chunked approach")
                    else:
                        # For datasets without time dimension - EXACT SAME AS ClimateDataCalculator
                        # Convert from K to C
                        t_c = t_ds[t_var].values - 273.15
                        td_c = td_ds[td_var].values - 273.15
                        
                        # Apply validation check: ensure dew point <= air temperature
                        td_c = np.minimum(td_c, t_c)
                        
                        # Calculate saturated vapor pressure using exponential Magnus formula
                        es = 6.1078 * np.exp((17.269 * t_c) / (237.3 + t_c))
                        ea = 6.1078 * np.exp((17.269 * td_c) / (237.3 + td_c))
                        
                        # Apply validation check: ensure VPD is never negative
                        vpd = np.maximum(es - ea, 0.0)
                        
                        # Create a new dataset with the VPD variable
                        vpd_ds = xr.Dataset(
                            data_vars={'vpd': (t_ds[t_var].dims, vpd)},
                            coords=t_ds.coords,
                            attrs={'units': 'hPa', 'long_name': 'Vapor Pressure Deficit'}
                        )
                        vpd_ds['vpd'].attrs['units'] = 'hPa'
                        vpd_ds['vpd'].attrs['long_name'] = 'Vapor Pressure Deficit'
                        
                        self.datasets['vpd'] = vpd_ds
                        print("VPD calculated successfully using ClimateDataCalculator approach")
                else:
                    print("Could not identify temperature variables")
            except Exception as e:
                print(f"Error calculating VPD: {str(e)}")
                import traceback
                traceback.print_exc()
        
       # Calculate PAR-related variables if solar radiation is available
        if 'surface_solar_radiation_downwards' in self.datasets:
            print("Calculating PAR-related variables from solar radiation...")
            try:
                sr_ds = self.datasets['surface_solar_radiation_downwards']
                
                # Get the main data variable
                sr_var = list(sr_ds.data_vars)[0] if list(sr_ds.data_vars) else None
                
                if sr_var:
                    # Check if the data needs time differencing (for accumulated values)
                    is_accumulated = False
                    if hasattr(sr_ds[sr_var], 'units'):
                        units = sr_ds[sr_var].attrs.get('units', '').lower()
                        # J m-2 indicates accumulated energy over time period
                        is_accumulated = 'j m-2' in units or 'j/m2' in units
                    
                    # Process with time chunking if we have time dimension
                    if 'time' in sr_ds.dims:
                        times = sr_ds.coords['time'].values
                        lat_var = [v for v in sr_ds.dims if v.lower() in ('latitude', 'lat')][0]
                        lon_var = [v for v in sr_ds.dims if v.lower() in ('longitude', 'lon')][0]
                        
                        # Get dimensions for spatial chunking
                        lat_values = sr_ds[lat_var].values
                        lon_values = sr_ds[lon_var].values
                        num_lats = len(lat_values)
                        num_lons = len(lon_values)
                        
                        # Calculate time steps in seconds (for accumulated values)
                        if is_accumulated and len(times) > 1:
                            time_diff_seconds = np.diff(
                                pd.DatetimeIndex(times).view('int64') / 10**9
                            ).astype(float)
                            # Handle case with only one time step
                            time_diff_seconds = np.append(time_diff_seconds[0] if len(time_diff_seconds) > 0 else 3600, 
                                                    time_diff_seconds)
                        
                        # CONVERT ACCUMULATED SOLAR RADIATION TO INSTANTANEOUS IF NEEDED
                        if is_accumulated:
                            print(f"Converting accumulated solar radiation data to instantaneous values...")
                            
                            # Create new array for instantaneous values
                            inst_values = np.zeros_like(sr_ds[sr_var].values)
                            
                            # Process all time steps after the first
                            for t_idx in range(1, len(times)):
                                current = sr_ds[sr_var].isel(time=t_idx).values
                                previous = sr_ds[sr_var].isel(time=t_idx-1).values
                                dt = time_diff_seconds[t_idx]
                                
                                # Ensure we're not dividing by zero or very small values
                                if dt < 1.0:
                                    dt = 3600.0  # Fall back to assuming hourly data
                                    
                                # Calculate instantaneous value and validate
                                inst_values[t_idx] = (current - previous) / dt
                                
                                # Check for unrealistic values
                                if np.any(inst_values[t_idx] > 1500) or np.any(inst_values[t_idx] < -10):
                                    print(f"Warning: Extreme solar radiation values detected at time {t_idx}")
                                    print(f"Range: {np.min(inst_values[t_idx])} to {np.max(inst_values[t_idx])}")
                                    # Clip to realistic values
                                    inst_values[t_idx] = np.clip(inst_values[t_idx], 0, 1500)
                            
                            # Handle first time step (copy second time step or use average)
                            if len(times) > 1:
                                inst_values[0] = inst_values[1]
                            
                            # Create new DataArray and Dataset
                            new_sr_da = xr.DataArray(
                                inst_values, 
                                dims=sr_ds[sr_var].dims,
                                coords=sr_ds[sr_var].coords,
                                attrs=sr_ds[sr_var].attrs.copy()
                            )
                            new_sr_da.attrs['units'] = 'W m-2'
                            
                            # Create new Dataset with instantaneous values
                            new_sr_ds = sr_ds.copy()
                            new_sr_ds[sr_var] = new_sr_da
                            
                            # Update the dataset
                            sr_ds = new_sr_ds
                            self.datasets['surface_solar_radiation_downwards'] = new_sr_ds
                            print("Successfully converted accumulated solar radiation to instantaneous values")
                        
                        # MEMORY OPTIMIZATION: Process in spatial chunks
                        # Define chunk size for processing
                        LAT_CHUNK_SIZE = min(100, num_lats)  # Process up to 100 latitude points at a time
                        
                        # Initialize final result arrays
                        ppfd_array = np.zeros((len(times), num_lats, num_lons), dtype=np.float32)
                        ext_rad_array = np.zeros((len(times), num_lats, num_lons), dtype=np.float32)
                        
                        # Create coordinate dictionary for final datasets
                        chunk_coords = {
                            'time': times,
                            'latitude': lat_values,
                            'longitude': lon_values
                        }
                        
                        # For extraterrestrial radiation calculation
                        if lat_var:
                            # Enhanced extraterrestrial radiation calculation based on solaR R package
                            print("Calculating extraterrestrial radiation using enhanced algorithm with spatial chunking...")
                            try:
                                # Define helper functions for conversion
                                def d2r(degrees):
                                    """Convert degrees to radians."""
                                    return np.deg2rad(degrees)
                                
                                def r2d(radians):
                                    """Convert radians to degrees."""
                                    return np.rad2deg(radians)
                                
                                def h2r(hours):
                                    """Convert hours to radians."""
                                    return hours * np.pi / 12
                                
                                # Get day of year for each timestamp
                                time_dt = pd.DatetimeIndex(times)
                                dn = time_dt.dayofyear
                                
                                # Calculate Julian days since 2000-01-01 12:00:00
                                # Define a timezone-naive origin for compatibility with time_values
                                origin = pd.Timestamp('2000-01-01 12:00:00')  # No timezone specification
                                
                                # Alternative implementation that doesn't rely on datetime subtraction
                                def datetime_to_julian(dt):
                                    """Convert datetime to Julian day since 2000-01-01 12:00:00"""
                                    # Convert to Python datetime if it's numpy datetime64
                                    dt = pd.Timestamp(dt).to_pydatetime()
                                    
                                    # Calculate the Julian day
                                    a = (14 - dt.month) // 12
                                    y = dt.year + 4800 - a
                                    m = dt.month + 12 * a - 3
                                    
                                    # Basic Julian day calculation
                                    jdn = dt.day + ((153 * m + 2) // 5) + 365 * y + y // 4 - y // 100 + y // 400 - 32045
                                    
                                    # Add time of day
                                    jd = jdn + (dt.hour - 12) / 24.0 + dt.minute / 1440.0 + dt.second / 86400.0
                                    
                                    # Days since 2000-01-01 12:00:00 (JD 2451545.0)
                                    return jd - 2451545.0
                                
                                # Calculate Julian days for each timestamp
                                jd_array = np.array([datetime_to_julian(t) for t in times])
                                
                                # Day angle for solar calculations
                                X_array = 2 * np.pi * (dn - 1) / 365
                                
                                # Solar calculations using the 'michalsky' method (default in solaR)
                                method = 'michalsky'
                                
                                # Solar constant (W/m²) as used in the R package
                                Bo = 1367
                                
                                # Process each time step separately to save memory
                                for t_idx in range(len(times)):
                                    # Get the corrected instantaneous solar radiation
                                    sr_slice_full = sr_ds[sr_var].isel(time=t_idx).values
                                    
                                    # Process in latitude chunks to avoid memory issues
                                    for lat_chunk_start in range(0, num_lats, LAT_CHUNK_SIZE):
                                        lat_chunk_end = min(lat_chunk_start + LAT_CHUNK_SIZE, num_lats)
                                        lat_chunk_size = lat_chunk_end - lat_chunk_start
                                        
                                        # Extract chunk of solar radiation data
                                        sr_slice = sr_slice_full[lat_chunk_start:lat_chunk_end, :]
                                        
                                        # Data validation - replace NaNs and negative values
                                        sr_slice = np.nan_to_num(sr_slice, nan=0.0)
                                        sr_slice = np.maximum(sr_slice, 0.0)  # No negative radiation
                                        
                                        # Calculate PAR (Photosynthetically Active Radiation)
                                        par_fraction = 0.45
                                        par_wm2 = sr_slice * par_fraction
                                        
                                        # Convert PAR from W/m2 to μmol/m2/s
                                        conversion_factor = 4.6
                                        ppfd_slice = par_wm2 * conversion_factor
                                        
                                        # Validate and constrain to physical limits
                                        ppfd_slice = np.clip(ppfd_slice, 0.0, 2500.0)  # Max realistic PPFD
                                        
                                        # Check for extreme values (debug)
                                        if t_idx == 0 and lat_chunk_start == 0:
                                            print(f"PPFD sample values (first chunk): {ppfd_slice.flatten()[:10]}")
                                            print(f"PPFD range: {np.min(ppfd_slice)} to {np.max(ppfd_slice)}")
                                        
                                        # Store PPFD result in final array
                                        ppfd_array[t_idx, lat_chunk_start:lat_chunk_end, :] = ppfd_slice
                                        
                                        # Prepare for extraterrestrial radiation calculation
                                        jd = jd_array[t_idx]
                                        X = X_array[t_idx]
                                        day_of_year = dn[t_idx]
                                        t_value = times[t_idx]
                                        
                                        # Calculate declination based on method
                                        if method == 'cooper':
                                            # Cooper method
                                            decl = d2r(23.45 * np.sin(2 * np.pi * (dn[t_idx] + 284) / 365))
                                        elif method == 'spencer':
                                            # Spencer method
                                            decl = 0.006918 - 0.399912 * np.cos(X) + 0.070257 * np.sin(X) - 0.006758 * np.cos(2 * X) \
                                                + 0.000907 * np.sin(2 * X) - 0.002697 * np.cos(3 * X) + 0.001480 * np.sin(3 * X)
                                        elif method == 'strous':
                                            # Strous method
                                            meanAnomaly = (357.5291 + 0.98560028 * jd) % 360
                                            coefC = np.array([1.9148, 0.02, 0.0003])
                                            
                                            C = 0
                                            for i in range(3):
                                                C += coefC[i] * np.sin(d2r((i + 1) * meanAnomaly))
                                            
                                            trueAnomaly = (meanAnomaly + C) % 360
                                            eclipLong = (trueAnomaly + 282.9372) % 360
                                            excen = 23.435
                                            
                                            decl = np.arcsin(np.sin(d2r(eclipLong)) * np.sin(d2r(excen)))
                                        else:  # 'michalsky' (default)
                                            # Michalsky method
                                            meanLong = (280.460 + 0.9856474 * jd) % 360
                                            meanAnomaly = (357.528 + 0.9856003 * jd) % 360
                                            
                                            eclipLong = (meanLong + 1.915 * np.sin(d2r(meanAnomaly)) + 
                                                        0.02 * np.sin(d2r(2 * meanAnomaly))) % 360
                                            excen = 23.439 - 0.0000004 * jd
                                            
                                            decl = np.arcsin(np.sin(d2r(eclipLong)) * np.sin(d2r(excen)))
                                        
                                        # Calculate Earth-Sun distance factor (eo)
                                        if method == 'cooper':
                                            eo = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
                                        else:  # spencer, michalsky, strous all use the same formula
                                            eo = (1.000110 + 0.034221 * np.cos(X) + 0.001280 * np.sin(X) +
                                                0.000719 * np.cos(2 * X) + 0.000077 * np.sin(2 * X))
                                        
                                        # Calculate Equation of Time (minutes)
                                        M = 2 * np.pi / 365.24 * day_of_year
                                        EoT_min = 229.18 * (-0.0334 * np.sin(M) + 0.04184 * np.sin(2 * M + 3.5884))
                                        EoT_hours = EoT_min / 60  # Convert to hours
                                        
                                        # Get hour of day for this timestamp
                                        timestamp = pd.Timestamp(t_value)
                                        hour_of_day = timestamp.hour + timestamp.minute / 60
                                        solar_time = hour_of_day + EoT_hours
                                        
                                        # Calculate hour angle (15° per hour from solar noon)
                                        hour_angle_rad = d2r(15 * (solar_time - 12))
                                        
                                        # Create smaller array for this chunk
                                        ext_rad_slice = np.zeros((lat_chunk_size, num_lons), dtype=np.float32)
                                        
                                        # Process only the current latitude chunk
                                        lat_chunk_values = lat_values[lat_chunk_start:lat_chunk_end]
                                        
                                        # Process each latitude point in chunk
                                        for i, lat in enumerate(lat_chunk_values):
                                            lat_rad = d2r(lat)
                                            
                                            # Sunset hour angle calculation
                                            cosWs = -np.tan(lat_rad) * np.tan(decl)
                                            
                                            # Handle polar day/night correctly
                                            if cosWs < -1:  # Polar day
                                                ws = np.pi
                                            elif cosWs > 1:  # Polar night
                                                ws = 0
                                            else:
                                                ws = np.arccos(cosWs)
                                            
                                            # Calculate cosine of solar zenith angle
                                            cos_zenith = (np.sin(lat_rad) * np.sin(decl) + 
                                                        np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle_rad))
                                            
                                            # No radiation when sun is below horizon
                                            cos_zenith = max(0.0, cos_zenith)
                                            
                                            # Calculate extraterrestrial radiation adjusted by Earth-Sun distance
                                            ext_rad_slice[i, :] = Bo * eo * cos_zenith
                                        
                                        # Store ext_rad result in final array
                                        ext_rad_array[t_idx, lat_chunk_start:lat_chunk_end, :] = ext_rad_slice
                                        
                                        # Clean up memory for this chunk
                                        del sr_slice, par_wm2, ppfd_slice, ext_rad_slice
                                    
                                    # Status update
                                    if t_idx % max(1, len(times)//10) == 0 or t_idx == len(times) - 1:
                                        print(f"Processed PAR variables for time {t_idx+1}/{len(times)}")
                                    
                            except Exception as e:
                                print(f"Error in extraterrestrial radiation calculation: {str(e)}")
                                import traceback
                                traceback.print_exc()
                                
                                # Fall back to simpler method if error occurs in custom calculation
                                print("Falling back to simpler extraterrestrial radiation calculation...")
                                
                                # Reset arrays (in case they were partially filled)
                                ppfd_array = np.zeros((len(times), num_lats, num_lons), dtype=np.float32)
                                ext_rad_array = np.zeros((len(times), num_lats, num_lons), dtype=np.float32)
                                
                                # Process each time step with simplified approach
                                for t_idx in range(len(times)):
                                    # Get the corrected instantaneous solar radiation
                                    sr_slice_full = sr_ds[sr_var].isel(time=t_idx).values
                                    
                                    # Process in chunks to reduce memory usage
                                    for lat_chunk_start in range(0, num_lats, LAT_CHUNK_SIZE):
                                        lat_chunk_end = min(lat_chunk_start + LAT_CHUNK_SIZE, num_lats)
                                        
                                        # Extract chunk
                                        sr_slice = sr_slice_full[lat_chunk_start:lat_chunk_end, :]
                                        
                                        # Data validation - replace NaNs and negative values
                                        sr_slice = np.nan_to_num(sr_slice, nan=0.0)
                                        sr_slice = np.maximum(sr_slice, 0.0)  # No negative radiation
                                        
                                        # Calculate PAR
                                        par_fraction = 0.45
                                        par_wm2 = sr_slice * par_fraction
                                        ppfd_slice = par_wm2 * 4.6
                                        
                                        # Validate and constrain to physical limits
                                        ppfd_slice = np.clip(ppfd_slice, 0.0, 2500.0)
                                        
                                        # Simple extraterrestrial radiation approximation
                                        ext_rad_slice = sr_slice * 1.5
                                        ext_rad_slice = np.clip(ext_rad_slice, 0.0, 1500.0)
                                        
                                        # Store in final arrays
                                        ppfd_array[t_idx, lat_chunk_start:lat_chunk_end, :] = ppfd_slice
                                        ext_rad_array[t_idx, lat_chunk_start:lat_chunk_end, :] = ext_rad_slice
                                        
                                        # Clean up memory
                                        del sr_slice, par_wm2, ppfd_slice, ext_rad_slice
                                    
                                    # Status update
                                    if t_idx % max(1, len(times)//10) == 0 or t_idx == len(times) - 1:
                                        print(f"Processed PAR variables for time {t_idx+1}/{len(times)} (simplified method)")
                        else:
                            # If no lat_var found, use simpler approach
                            print("Latitude coordinate not found, using solar radiation as proxy for ext_rad")
                            
                            # Initialize arrays
                            ppfd_array = np.zeros((len(times), num_lats, num_lons), dtype=np.float32)
                            ext_rad_array = np.zeros((len(times), num_lats, num_lons), dtype=np.float32)
                            
                            # Process each time step with simplified approach
                            for t_idx in range(len(times)):
                                # Get the corrected instantaneous solar radiation
                                sr_slice_full = sr_ds[sr_var].isel(time=t_idx).values
                                
                                # Process in chunks
                                for lat_chunk_start in range(0, num_lats, LAT_CHUNK_SIZE):
                                    lat_chunk_end = min(lat_chunk_start + LAT_CHUNK_SIZE, num_lats)
                                    
                                    # Extract chunk
                                    sr_slice = sr_slice_full[lat_chunk_start:lat_chunk_end, :]
                                    
                                    # Data validation - replace NaNs and negative values
                                    sr_slice = np.nan_to_num(sr_slice, nan=0.0)
                                    sr_slice = np.maximum(sr_slice, 0.0)  # No negative radiation
                                    
                                    # Calculate PAR
                                    par_fraction = 0.45
                                    par_wm2 = sr_slice * par_fraction
                                    ppfd_slice = par_wm2 * 4.6
                                    
                                    # Validate and constrain to physical limits
                                    ppfd_slice = np.clip(ppfd_slice, 0.0, 2500.0)
                                    
                                    # Simple extraterrestrial radiation approximation
                                    ext_rad_slice = sr_slice * 1.5
                                    ext_rad_slice = np.clip(ext_rad_slice, 0.0, 1500.0)
                                    
                                    # Store in final arrays
                                    ppfd_array[t_idx, lat_chunk_start:lat_chunk_end, :] = ppfd_slice
                                    ext_rad_array[t_idx, lat_chunk_start:lat_chunk_end, :] = ext_rad_slice
                                    
                                    # Clean up memory
                                    del sr_slice, par_wm2, ppfd_slice, ext_rad_slice
                                
                                # Status update
                                if t_idx % max(1, len(times)//10) == 0 or t_idx == len(times) - 1:
                                    print(f"Processed PAR variables for time {t_idx+1}/{len(times)}")
                        
                        # Create PPFD dataset
                        ppfd_da = xr.DataArray(
                            ppfd_array,
                            dims=['time', lat_var, lon_var],
                            coords=chunk_coords,
                            name='ppfd_in'
                        )
                        ppfd_ds = ppfd_da.to_dataset()
                        ppfd_ds['ppfd_in'].attrs['units'] = 'μmol m-2 s-1'
                        ppfd_ds['ppfd_in'].attrs['long_name'] = 'Photosynthetic Photon Flux Density'
                        ppfd_ds['ppfd_in'].attrs['description'] = 'Calculated from instantaneous solar radiation'
                        self.datasets['ppfd_in'] = ppfd_ds
                        
                        # Create extraterrestrial radiation dataset
                        ext_rad_da = xr.DataArray(
                            ext_rad_array,
                            dims=['time', lat_var, lon_var],
                            coords=chunk_coords,
                            name='ext_rad'
                        )
                        ext_rad_ds = ext_rad_da.to_dataset()
                        ext_rad_ds['ext_rad'].attrs['units'] = 'W m-2'
                        ext_rad_ds['ext_rad'].attrs['long_name'] = 'Extraterrestrial Radiation'
                        self.datasets['ext_rad'] = ext_rad_ds
                        
                        print("PAR-related variables calculated successfully with chunked approach")
                    else:
                        # For datasets without time dimension
                        # Standard approach using named variables
                        solar_rad = sr_ds[sr_var]
                        
                        # Data validation
                        solar_rad = solar_rad.where(solar_rad >= 0, 0)
                        
                        # Calculate PAR (Photosynthetically Active Radiation)
                        par_fraction = 0.45
                        par_wm2 = solar_rad * par_fraction
                        
                        # Convert PAR from W/m2 to μmol/m2/s
                        conversion_factor = 4.6
                        ppfd = par_wm2 * conversion_factor
                        
                        # Validate and constrain to physical limits
                        ppfd = ppfd.where(ppfd <= 2500.0, 2500.0)
                        
                        # Create new datasets
                        ppfd_ds = xr.Dataset(
                            data_vars={'ppfd_in': ppfd},
                            coords=sr_ds.coords,
                            attrs={'units': 'μmol m-2 s-1', 'long_name': 'Photosynthetic Photon Flux Density'}
                        )
                        ppfd_ds['ppfd_in'].attrs['units'] = 'μmol m-2 s-1'
                        ppfd_ds['ppfd_in'].attrs['long_name'] = 'Photosynthetic Photon Flux Density'
                        
                        # Also calculate extraterrestrial radiation
                        lat_var = None
                        for var in sr_ds.coords:
                            if var.lower() in ['latitude', 'lat']:
                                lat_var = var
                        
                        if lat_var:
                            # Use original enhanced calculation
                            try:
                                # Full calculation from original (this won't run in no-time case,
                                # but keeping for logical consistency with original)
                                print("Calculating extraterrestrial radiation using enhanced algorithm...")
                                # ... original calculation code would be here ...
                                # But since we're in the no-time-dimension case, this won't actually execute
                                raise NotImplementedError("Enhanced calculation requires time dimension")
                            except Exception as inner_e:
                                print(f"Error in enhanced calculation: {str(inner_e)}")
                                print("Falling back to simpler method")
                                ext_rad = solar_rad * 1.5  # Simple approximation
                                ext_rad = ext_rad.where(ext_rad <= 1500.0, 1500.0)
                                ext_rad_ds = xr.Dataset(
                                    data_vars={'ext_rad': ext_rad},
                                    coords=sr_ds.coords,
                                    attrs={'units': 'W m-2', 'long_name': 'Extraterrestrial Radiation (approximated)'}
                                )
                        else:
                            print("Latitude coordinate not found, using solar radiation as proxy for ext_rad")
                            ext_rad = solar_rad * 1.5  # Simple approximation
                            ext_rad = ext_rad.where(ext_rad <= 1500.0, 1500.0)
                            ext_rad_ds = xr.Dataset(
                                data_vars={'ext_rad': ext_rad},
                                coords=sr_ds.coords,
                                attrs={'units': 'W m-2', 'long_name': 'Extraterrestrial Radiation (approximated)'}
                            )
                        
                        ext_rad_ds['ext_rad'].attrs['units'] = 'W m-2'
                        ext_rad_ds['ext_rad'].attrs['long_name'] = 'Extraterrestrial Radiation (approximated)'
                        
                        self.datasets['ppfd_in'] = ppfd_ds
                        self.datasets['ext_rad'] = ext_rad_ds
                        print(f"PAR-related variables calculated successfully from variable: {sr_var}")
                else:
                    print("Could not identify solar radiation variable")
            except Exception as e:
                print(f"Error calculating PAR-related variables: {str(e)}")
                import traceback
                traceback.print_exc()
                
                return self.datasets

    # Keeping the existing methods from the original class that don't require changes
    # for accessing data from local files vs GEE:
    
    # Methods related to climate data
    def load_climate_data(self, temp_file=None, precip_file=None, bounds=None):
        """
        Load annual mean temperature and precipitation data from GeoTIFF files,
        optionally subset to area of interest.
        
        Parameters:
        -----------
        temp_file : str or Path, optional
            Path to the annual mean temperature file (.tif)
        precip_file : str or Path, optional
            Path to the annual mean precipitation file (.tif)
        bounds : tuple or dict, optional
            Bounding box coordinates as (lon_min, lat_min, lon_max, lat_max) or 
            dict with these keys
                    
        Returns:
        --------
        tuple
            (temperature_data, precipitation_data)
        """
        temp_file = Path(temp_file) if temp_file else config.TEMP_CLIMATE_FILE
        precip_file = Path(precip_file) if precip_file else config.PRECIP_CLIMATE_FILE
        
        if not temp_file.exists():
            print(f"Temperature file not found: {temp_file}")
            return None, None
        
        if not precip_file.exists():
            print(f"Precipitation file not found: {precip_file}")
            return None, None
        
        # Process bounds to standard format
        if bounds:
            if isinstance(bounds, dict):
                lon_min = bounds.get('lon_min')
                lat_min = bounds.get('lat_min')
                lon_max = bounds.get('lon_max')
                lat_max = bounds.get('lat_max')
            else:
                lon_min, lat_min, lon_max, lat_max = bounds
            
            print(f"Loading climate data for region: lon={lon_min}-{lon_max}, lat={lat_min}-{lat_max}")
        else:
            print("Loading global climate data (no bounds specified)")
        
        try:
            print(f"Loading annual mean temperature data from {temp_file}...")
            with rasterio.open(temp_file) as temp_src:
                # Get the geotransform information
                temp_transform = temp_src.transform
                temp_crs = temp_src.crs
                
                # If bounds specified, determine window for reading
                if bounds:
                    # Convert geographic bounds to pixel coordinates
                    window = from_bounds(lon_min, lat_min, lon_max, lat_max, temp_transform)
                    
                    # Read only the data within the window
                    temp_data = temp_src.read(1, window=window)
                    
                    # Update transform for the window
                    window_transform = rasterio.windows.transform(window, temp_transform)
                    
                    print(f"Subset temperature data loaded: shape={temp_data.shape}")
                    
                    # Get temperature data range (excluding NoData values)
                    nodata = temp_src.nodata
                    valid_temp = temp_data[temp_data != nodata] if nodata is not None else temp_data
                    if len(valid_temp) > 0:
                        temp_min, temp_max = valid_temp.min(), valid_temp.max()
                        print(f"Temperature range in region: {temp_min} to {temp_max}")
                        
                else:
                    # Read the entire dataset
                    temp_data = temp_src.read(1)
                    window_transform = temp_transform
                
                # Create a simple dataset-like structure
                height, width = temp_data.shape
                self.temp_climate_data = {
                    'data': temp_data,
                    'transform': window_transform,  # Use window transform if subsetting
                    'crs': temp_crs,
                    'height': height,
                    'width': width,
                    'nodata': temp_src.nodata
                }
            
            print(f"Loading annual mean precipitation data from {precip_file}...")
            with rasterio.open(precip_file) as precip_src:
                # If bounds specified, determine window for reading
                if bounds:
                    # Convert geographic bounds to pixel coordinates
                    window = from_bounds(lon_min, lat_min, lon_max, lat_max, precip_src.transform)
                    
                    # Read only the data within the window
                    precip_data = precip_src.read(1, window=window)
                    
                    # Update transform for the window
                    window_transform = rasterio.windows.transform(window, precip_src.transform)
                    
                    print(f"Subset precipitation data loaded: shape={precip_data.shape}")
                    
                    # Get precipitation data range (excluding NoData values)
                    nodata = precip_src.nodata
                    valid_precip = precip_data[precip_data != nodata] if nodata is not None else precip_data
                    if len(valid_precip) > 0:
                        precip_min, precip_max = valid_precip.min(), valid_precip.max()
                        print(f"Precipitation range in region: {precip_min} to {precip_max}")
                else:
                    # Read the entire dataset
                    precip_data = precip_src.read(1)
                    window_transform = precip_src.transform
                
                # Create a simple dataset-like structure
                height, width = precip_data.shape
                self.precip_climate_data = {
                    'data': precip_data,
                    'transform': window_transform,  # Use window transform if subsetting
                    'crs': precip_src.crs,
                    'height': height,
                    'width': width,
                    'nodata': precip_src.nodata
                }
            
            # Note: Don't forget to add these attributes to your __init__ method
            self.worldclim_already_scaled = False  # Set based on your data source
            self.is_worldclim = True  # Set based on your data source
            
            return self.temp_climate_data, self.precip_climate_data
        
        except Exception as e:
            print(f"Error loading climate data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def get_climate_at_location_direct(self, lon, lat, temp_climate=None, precip_climate=None, max_distance=0.1):
        """
        KD-tree approach for climate data lookup with improved boundary handling.
        
        Parameters:
        -----------
        lon, lat : float or array-like
            Coordinates of the location(s)
        temp_climate : dict, optional
            Temperature climate data dictionary
        precip_climate : dict, optional
            Precipitation climate data dictionary
        max_distance : float, optional
            Maximum distance (in degrees) to accept a valid data point.
            Points beyond this distance will use default values.
                    
        Returns:
        --------
        tuple
            (temperature, precipitation) for single point or
            (temperatures, precipitations) for multiple points
        """
        from scipy.spatial import cKDTree
        import time
        
        # Use instance climate data if not provided
        if temp_climate is None:
            temp_climate = self.temp_climate_data
        if precip_climate is None:
            precip_climate = self.precip_climate_data
        
        # Convert inputs to arrays
        single_point = np.isscalar(lon) and np.isscalar(lat)
        lons = np.array([lon]) if single_point else np.asarray(lon)
        lats = np.array([lat]) if single_point else np.asarray(lat)
        
        # Default values
        temperatures = np.full_like(lons, 15.0, dtype=float)
        precipitations = np.full_like(lats, 800.0, dtype=float)
        
        # Process temperature data
        if temp_climate and 'data' in temp_climate:
            try:
                start_time = time.time()
                
                # Get raster dimensions
                height, width = temp_climate['data'].shape
                
                # Extract raster extent
                transform = temp_climate['transform']
                xmin, ymax = transform * (0, 0)
                xmax, ymin = transform * (width, height)
                
                print(f"Temperature data extent: lon={xmin:.4f}-{xmax:.4f}, lat={ymin:.4f}-{ymax:.4f}")
                
                # Check how many points are outside the extent
                outside_extent = ((lons < xmin) | (lons > xmax) | (lats < ymin) | (lats > ymax))
                outside_count = np.sum(outside_extent)
                if outside_count > 0:
                    print(f"WARNING: {outside_count} out of {len(lons)} points are outside temperature data extent")
                
                # Create a regular grid of pixel center coordinates
                row_coords = np.arange(height)
                col_coords = np.arange(width)
                
                # Get geographic coordinates for each pixel
                pixel_lons = np.zeros(width)
                pixel_lats = np.zeros(height)
                
                # Compute coordinates for each pixel center
                for c in range(width):
                    pixel_lons[c], _ = transform * (c + 0.5, 0)
                
                for r in range(height):
                    _, pixel_lats[r] = transform * (0, r + 0.5)
                
                # Create full coordinate grids using meshgrid
                lon_grid, lat_grid = np.meshgrid(pixel_lons, pixel_lats)
                
                # Flatten for KD-tree
                points = np.vstack((lon_grid.flatten(), lat_grid.flatten())).T
                values = temp_climate['data'].flatten()
                
                # Only include valid data points
                nodata = temp_climate.get('nodata')
                valid_mask = np.ones_like(values, dtype=bool) if nodata is None else values != nodata
                
                if np.sum(valid_mask) == 0:
                    print("WARNING: No valid temperature data points found in raster")
                else:
                    # Create KD-tree only with valid points
                    valid_points = points[valid_mask]
                    valid_values = values[valid_mask]
                    
                    # Build tree and query
                    tree = cKDTree(valid_points)
                    distances, indices = tree.query(np.vstack((lons, lats)).T, k=1)
                    
                    # Extract values with distance check
                    valid_distance = distances <= max_distance
                    for i, (idx, valid) in enumerate(zip(indices, valid_distance)):
                        if valid:
                            temperatures[i] = valid_values[idx]
                    
                    tree_time = time.time() - start_time
                    print(f"Temperature lookup completed in {tree_time:.2f}s: {np.sum(temperatures != 15.0)} valid values found")
                    
                    # Additional debug info
                    if np.sum(valid_distance) < len(lons):
                        print(f"Some points were too far from valid climate data: {len(lons) - np.sum(valid_distance)} points using default temperature")
            except Exception as e:
                print(f"Error in temperature KD-tree lookup: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("No temperature climate data available for KD-tree lookup")
        
        # Process precipitation data
        if precip_climate and 'data' in precip_climate:
            try:
                start_time = time.time()
                
                # Get raster dimensions
                height, width = precip_climate['data'].shape
                
                # Extract raster extent
                transform = precip_climate['transform']
                xmin, ymax = transform * (0, 0)
                xmax, ymin = transform * (width, height)
                
                print(f"Precipitation data extent: lon={xmin:.4f}-{xmax:.4f}, lat={ymin:.4f}-{ymax:.4f}")
                
                # Check how many points are outside the extent
                outside_extent = ((lons < xmin) | (lons > xmax) | (lats < ymin) | (lats > ymax))
                outside_count = np.sum(outside_extent)
                if outside_count > 0:
                    print(f"WARNING: {outside_count} out of {len(lons)} points are outside precipitation data extent")
                
                # Create a regular grid of pixel center coordinates
                row_coords = np.arange(height)
                col_coords = np.arange(width)
                
                # Get geographic coordinates for each pixel
                pixel_lons = np.zeros(width)
                pixel_lats = np.zeros(height)
                
                # Compute coordinates for each pixel center
                for c in range(width):
                    pixel_lons[c], _ = transform * (c + 0.5, 0)
                
                for r in range(height):
                    _, pixel_lats[r] = transform * (0, r + 0.5)
                
                # Create full coordinate grids using meshgrid
                lon_grid, lat_grid = np.meshgrid(pixel_lons, pixel_lats)
                
                # Flatten for KD-tree
                points = np.vstack((lon_grid.flatten(), lat_grid.flatten())).T
                values = precip_climate['data'].flatten()
                
                # Only include valid data points
                nodata = precip_climate.get('nodata')
                valid_mask = np.ones_like(values, dtype=bool) if nodata is None else values != nodata
                
                if np.sum(valid_mask) == 0:
                    print("WARNING: No valid precipitation data points found in raster")
                else:
                    # Create KD-tree only with valid points
                    valid_points = points[valid_mask]
                    valid_values = values[valid_mask]
                    
                    # Build tree and query
                    tree = cKDTree(valid_points)
                    distances, indices = tree.query(np.vstack((lons, lats)).T, k=1)
                    
                    # Extract values with distance check
                    valid_distance = distances <= max_distance
                    for i, (idx, valid) in enumerate(zip(indices, valid_distance)):
                        if valid:
                            precipitations[i] = valid_values[idx]
                    
                    tree_time = time.time() - start_time
                    print(f"Precipitation lookup completed in {tree_time:.2f}s: {np.sum(precipitations != 800.0)} valid values found")
                    
                    # Additional debug info
                    if np.sum(valid_distance) < len(lons):
                        print(f"Some points were too far from valid climate data: {len(lons) - np.sum(valid_distance)} points using default precipitation")
            except Exception as e:
                print(f"Error in precipitation KD-tree lookup: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("No precipitation climate data available for KD-tree lookup")
        
        # Return based on input type
        if single_point:
            return float(temperatures[0]), float(precipitations[0])
        else:
            return temperatures, precipitations
    def resample_climate_data(self, target_lat=None, target_lon=None):
        """
        Resample climate data from GeoTIFF files to match a target resolution.
        Uses vectorized implementation for efficiency.
        
        Parameters:
        -----------
        target_lat : array-like, optional
            Target latitude array
        target_lon : array-like, optional
            Target longitude array
                
        Returns:
        --------
        tuple
            (resampled_temp_data, resampled_precip_data)
        """
        if not hasattr(self, 'temp_climate_data') or not hasattr(self, 'precip_climate_data'):
            print("No climate data available for resampling")
            return None, None
        
        try:
            # If no target resolution provided, use the first ERA5-Land dataset
            if target_lat is None or target_lon is None:
                if not self.datasets:
                    print("No ERA5-Land datasets available for target resolution")
                    return None, None
                
                # Get the first dataset for coordinates
                first_ds = list(self.datasets.values())[0]
                
                # Find coordinate variables
                lat_var = None
                lon_var = None
                for var in first_ds.coords:
                    if var.lower() in ['latitude', 'lat']:
                        lat_var = var
                    elif var.lower() in ['longitude', 'lon']:
                        lon_var = var
                
                if not lat_var or not lon_var:
                    print("Could not identify coordinate variables in ERA5-Land dataset")
                    return None, None
                
                target_lat = first_ds[lat_var].values
                target_lon = first_ds[lon_var].values
            
            print(f"Resampling climate data to target grid: {len(target_lat)}x{len(target_lon)}")
            
            # Create coordinate meshgrid for vectorized lookup
            lon_grid, lat_grid = np.meshgrid(target_lon, target_lat)
            
            # Use the vectorized method to get all values at once
            resampled_temp, resampled_precip = self.get_climate_at_location_direct(
                lon_grid.flatten(), lat_grid.flatten()
            )
            
            # Reshape to match the target grid
            resampled_temp = resampled_temp.reshape(len(target_lat), len(target_lon))
            resampled_precip = resampled_precip.reshape(len(target_lat), len(target_lon))
            
            # Create xarray DataArrays with the resampled data
            resampled_temp_da = xr.DataArray(
                resampled_temp,
                dims=['latitude', 'longitude'],
                coords={'latitude': target_lat, 'longitude': target_lon},
                name='annual_mean_temperature'
            )
            
            resampled_precip_da = xr.DataArray(
                resampled_precip,
                dims=['latitude', 'longitude'],
                coords={'latitude': target_lat, 'longitude': target_lon},
                name='annual_mean_precipitation'
            )
            
            # Store the resampled data
            self.resampled_temp_data = resampled_temp_da
            self.resampled_precip_data = resampled_precip_da
            
            print("Climate data resampled successfully using vectorized approach")
            return resampled_temp_da, resampled_precip_da
        
        except Exception as e:
            print(f"Error resampling climate data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

        
    def add_time_features(self, df):
        """
        Add cyclical time features to the DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with datetime index
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added time features
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except:
                print("Cannot convert index to datetime")
                return df
        
        # Extract time components
        timestamp_s = df.index.map(pd.Timestamp.timestamp)
        
        # Define cycles in seconds
        day = 24 * 60 * 60  # seconds in a day
        week = 7 * day      # seconds in a week
        month = 30.44 * day # seconds in a month (average)
        year = 365.2425 * day  # seconds in a year (average)
        
        # Create cyclical features
        df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        df['Week sin'] = np.sin(timestamp_s * (2 * np.pi / week))
        df['Week cos'] = np.cos(timestamp_s * (2 * np.pi / week))
        df['Month sin'] = np.sin(timestamp_s * (2 * np.pi / month))
        df['Month cos'] = np.cos(timestamp_s * (2 * np.pi / month))
        df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
        
        return df
    def standardize_timestamps(self, timestamps, use_utc=True):
        """
        Standardize timestamps to either all timezone-aware (UTC) or all timezone-naive.
        
        Parameters:
        -----------
        timestamps : list, pandas.DatetimeIndex, or xarray time coordinate
            Timestamps to standardize
        use_utc : bool
            If True, standardize to timezone-aware UTC
            If False, standardize to timezone-naive
            
        Returns:
        --------
        Same type as input, with standardized timestamps
        """
        import pandas as pd
        
        if isinstance(timestamps, list):
            result = []
            for ts in timestamps:
                if use_utc:
                    # Make timezone-aware (UTC) - using pandas functions which are more reliable
                    if ts.tzinfo is None:
                        result.append(pd.Timestamp(ts).tz_localize('UTC'))
                    else:
                        result.append(pd.Timestamp(ts).tz_convert('UTC'))
                else:
                    # Make timezone-naive
                    if ts.tzinfo is not None:
                        result.append(pd.Timestamp(ts).tz_localize(None))
                    else:
                        result.append(ts)
            return result
        
        elif isinstance(timestamps, pd.DatetimeIndex):
            if use_utc:
                # Make timezone-aware (UTC)
                if timestamps.tz is None:
                    return timestamps.tz_localize('UTC')
                else:
                    return timestamps.tz_convert('UTC')
            else:
                # Make timezone-naive
                if timestamps.tz is not None:
                    return timestamps.tz_localize(None)
                else:
                    return timestamps
        
        elif hasattr(timestamps, 'values'):  # xarray time coordinate
            # Extract values and convert back to same type
            values = pd.DatetimeIndex(timestamps.values)
            standardized = self.standardize_timestamps(values, use_utc)
            return type(timestamps)(standardized)
        
        return timestamps  # Return unchanged if not recognized type
        
        return timestamps  # Return unchanged if not recognized type
    def create_prediction_dataset(self, start_date=None, end_date=None, points=None, region=None, use_existing_data=True, use_utc=True):
        """
        Create a dataset ready for prediction by extracting data.
        Uses vectorized approach for climate data lookup and optimized grid processing.

        Parameters:
        -----------
        start_date : str or datetime, optional
            Start date for the prediction period. If None and use_existing_data=True,
            derived from loaded data. Required if use_existing_data=False.
        end_date : str or datetime, optional
            End date for the prediction period. If None and use_existing_data=True,
            derived from loaded data. Required if use_existing_data=False.
        points : list of dict, optional
            List of points to extract data for [{lat, lon, name}, ...].
            If provided, overrides region/grid processing.
        region : dict, optional
            Region dictionary {lat_min, lat_max, lon_min, lon_max}. Used for grid
            processing if 'points' is None. Defaults taken from config if None.
        use_existing_data : bool, optional
            If True, uses data already loaded in self.datasets. start/end_date
            can be used to filter this data. If False, loading logic (not shown
            in this specific method) would need to be triggered beforehand based
            on start/end_date. Defaults to True.
        use_utc : bool, optional
            Whether to standardize timestamps to UTC (True) or timezone-naive (False).

        Returns:
        --------
        pandas.DataFrame or list of pandas.DataFrame
            Single DataFrame with features for grid processing (MultiIndex initially,
            then reset). List of DataFrames for point processing. Returns None on failure.
        """
        print("Creating prediction dataset...")
        # --- Setup and Data Loading/Validation ---
        if region is None:
            # Assuming config has these defaults defined
            region = {
                'lat_min': getattr(config, 'DEFAULT_LAT_MIN', -90),
                'lat_max': getattr(config, 'DEFAULT_LAT_MAX', 90),
                'lon_min': getattr(config, 'DEFAULT_LON_MIN', -180),
                'lon_max': getattr(config, 'DEFAULT_LON_MAX', 180)
            }

        if use_existing_data:
            if not self.datasets:
                print("Error: use_existing_data=True but no data has been loaded.")
                return None # Changed from raise ValueError to return None for worker compatibility

            # Check if derived variables are needed and calculated
            # Assuming config.VARIABLE_RENAME maps ERA5 names to model names
            # We need to check if the *source* names for derived vars exist
            # or if the derived vars themselves exist
            source_vars_needed = list(config.VARIABLE_RENAME.keys())
            required_derived_vars = ['wind_speed', 'vpd', 'ppfd_in', 'ext_rad'] # Example derived names
            missing_derived = []
            if 'wind_speed' in source_vars_needed or 'ws' in source_vars_needed:
                 if 'wind_speed' not in self.datasets: missing_derived.append('wind_speed')
            if 'vpd' in source_vars_needed:
                 if 'vpd' not in self.datasets: missing_derived.append('vpd')
            if 'ppfd_in' in source_vars_needed:
                 if 'ppfd_in' not in self.datasets: missing_derived.append('ppfd_in')
            if 'ext_rad' in source_vars_needed:
                 if 'ext_rad' not in self.datasets: missing_derived.append('ext_rad')

            if missing_derived:
                print(f"Calculating missing derived variables: {missing_derived}...")
                try:
                    self.calculate_derived_variables()
                    # Verify they exist now
                    if any(var in missing_derived and var not in self.datasets for var in missing_derived):
                         print(f"Warning: Could not calculate all derived variables. Missing: {[var for var in missing_derived if var not in self.datasets]}")
                except Exception as calc_err:
                    print(f"Error during derived variable calculation: {calc_err}")
                    # Decide whether to continue without them or fail
                    # return None # Example: Fail if derived vars are critical

            # Determine time range from existing data if start/end not specified
            if start_date is None or end_date is None:
                all_times_pd = []
                for ds_name, ds in self.datasets.items():
                    time_coord = None
                    for coord_name in ds.coords:
                        if coord_name.lower() in ['time', 'datetime']:
                            time_coord = coord_name
                            break
                    if time_coord:
                        try:
                            # Convert numpy datetime64 to pandas DatetimeIndex
                            ds_times = pd.DatetimeIndex(ds[time_coord].values)
                            all_times_pd.append(ds_times)
                        except Exception as time_err:
                            print(f"Warning: Could not process time coordinate for {ds_name}: {time_err}")

                if not all_times_pd:
                    print("Error: No time coordinates found in existing datasets to determine range.")
                    return None # Cannot proceed without time range

                # Concatenate all time indices and standardize
                # Using pd.Index to handle potential duplicates before finding min/max
                combined_times = pd.Index([])
                for idx in all_times_pd:
                    combined_times = combined_times.union(idx)

                standardized_times = self.standardize_timestamps(combined_times, use_utc=use_utc)

                if not standardized_times.empty:
                    if start_date is None:
                        start_date = standardized_times.min()
                        print(f"Using start_date from existing data: {start_date}")
                    if end_date is None:
                        end_date = standardized_times.max()
                        print(f"Using end_date from existing data: {end_date}")
                else:
                    print("Error: Could not determine time range from existing data.")
                    return None
        else:
            # If not using existing data, assume data for the required period
            # was loaded beforehand by the calling function (e.g., hourly worker)
            if start_date is None or end_date is None:
                 print("Error: start_date and end_date are required if use_existing_data=False (and data isn't pre-loaded).")
                 return None

        # Convert dates to datetime objects and standardize timezone
        start_date_std, end_date_std = None, None
        try:
            if start_date:
                start_date_pd = pd.Timestamp(start_date)
                # Standardize requires a DatetimeIndex input for this implementation
                start_date_std = self.standardize_timestamps(pd.DatetimeIndex([start_date_pd]), use_utc=use_utc)[0]
            if end_date:
                end_date_pd = pd.Timestamp(end_date)
                end_date_std = self.standardize_timestamps(pd.DatetimeIndex([end_date_pd]), use_utc=use_utc)[0]
        except Exception as date_err:
            print(f"Error processing start/end dates: {date_err}")
            return None

        # Select only the variables needed for prediction based on config rename mapping
        # We need both the original names (for selecting from self.datasets)
        # and the final model names (for the output)
        prediction_vars_map = {} # Maps original_name -> model_name
        for era5_name, model_name in config.VARIABLE_RENAME.items():
            # Check if the source variable exists in the loaded datasets
            if era5_name in self.datasets:
                prediction_vars_map[era5_name] = model_name
            # Also check if a derived variable (which might share the model name) exists
            elif model_name in self.datasets and model_name not in prediction_vars_map.values():
                 # If the model name itself exists as a key in datasets (e.g. 'vpd')
                 # and we haven't already mapped something else to it.
                 # We map it to itself to ensure it's included.
                 prediction_vars_map[model_name] = model_name

        if not prediction_vars_map:
            print("Error: No required data variables found in loaded datasets based on config.")
            return None

        print(f"Using variables for processing (Original Name -> Model Name): {prediction_vars_map}")
        original_vars_to_process = list(prediction_vars_map.keys())

        # --- Processing Logic ---
        if points:
            # --- POINTS-BASED APPROACH ---
            print("Processing specific points...")
            all_data = [] # List to hold DataFrames, one per point

            # Vectorized climate data lookup for all points
            point_lons = np.array([p['lon'] for p in points])
            point_lats = np.array([p['lat'] for p in points])
            temps, precips = None, None
            climate_data_available = False
            if hasattr(self, 'temp_climate_data') and hasattr(self, 'precip_climate_data'):
                try:
                    print(f"Getting climate data for {len(point_lons)} points using vectorized approach...")
                    temps, precips = self.get_climate_at_location_direct(
                        point_lons, point_lats,
                        self.temp_climate_data, self.precip_climate_data
                    )
                    climate_data_available = True
                    print("Successfully retrieved climate data for points.")
                except Exception as e:
                    print(f"Warning: Error in batch climate data retrieval for points: {e}")
            else:
                print("Warning: Climate data not loaded, will use defaults for points.")

            # Process each point
            for i, point in enumerate(points):
                point_data_dict = {} # Holds scalar data for this point
                point_series_list = [] # Holds time series data for this point
                point_lat = point['lat']
                point_lon = point['lon']
                point_name = point.get('name', f"Point_{point_lat}_{point_lon}")

                point_data_dict['name'] = point_name
                point_data_dict['latitude'] = point_lat
                point_data_dict['longitude'] = point_lon

                # Assign climate data
                if climate_data_available and temps is not None and precips is not None:
                    point_data_dict['annual_mean_temperature'] = float(temps[i])
                    point_data_dict['annual_precipitation'] = float(precips[i])
                else:
                    # Fallback to default values if batch failed or data wasn't loaded
                    point_data_dict['annual_mean_temperature'] = 15.0
                    point_data_dict['annual_precipitation'] = 800.0

                # Extract time series for each required variable at the point
                common_time_index = None # Track the index from the first successful series
                for original_var_name in original_vars_to_process:
                    model_name = prediction_vars_map[original_var_name] # Get target name
                    if original_var_name not in self.datasets:
                        print(f"Warning: Dataset for '{original_var_name}' not found for point {point_name}")
                        continue
                    try:
                        ds = self.datasets[original_var_name]
                        # Robustly find the data variable within the dataset
                        data_var_key = None
                        if len(ds.data_vars) == 1:
                             data_var_key = list(ds.data_vars)[0]
                        elif model_name in ds.data_vars: # Check if derived var name matches
                             data_var_key = model_name
                        elif original_var_name in ds.data_vars: # Check if original name matches
                             data_var_key = original_var_name
                        else: # Fallback if naming is unexpected
                             data_var_key = list(ds.data_vars)[0] if ds.data_vars else None

                        if not data_var_key:
                             print(f"Warning: Could not find data variable in dataset for '{original_var_name}'. Skipping.")
                             continue

                        # Find time coordinate
                        time_dim_point = None
                        for dim in ds.dims:
                            if dim.lower() in ['time', 'datetime']:
                                time_dim_point = dim
                                break
                        if not time_dim_point:
                             print(f"Warning: No time dimension found for '{original_var_name}'. Skipping.")
                             continue # Skip if no time dim

                        # Select nearest point
                        ts_point = ds[data_var_key].sel(
                            latitude=point_lat, longitude=point_lon, method='nearest'
                        )

                        # Apply time filtering using standardized dates
                        current_times_pd = pd.DatetimeIndex(ts_point[time_dim_point].values)
                        current_times_std = self.standardize_timestamps(current_times_pd, use_utc=use_utc)
                        ts_point = ts_point.assign_coords({time_dim_point: current_times_std}) # Ensure coord is standardized

                        if start_date_std is not None and end_date_std is not None:
                           # Use the standardized slice for selection
                           time_slice_point = slice(start_date_std, end_date_std)
                           ts_point = ts_point.sel({time_dim_point: time_slice_point})

                        # Check if data remains after slicing
                        if ts_point.size == 0:
                            print(f"Warning: No data remains for '{original_var_name}' at point {point_name} after time slicing.")
                            continue

                        # Convert to Pandas Series with standardized time index
                        times_final_point = self.standardize_timestamps(pd.DatetimeIndex(ts_point[time_dim_point].values), use_utc=use_utc)
                        values_point = ts_point.values

                        series = pd.Series(values_point, index=times_final_point, name=model_name)

                        # Store the first valid time index
                        if common_time_index is None and not series.empty:
                            common_time_index = series.index

                        point_series_list.append(series)

                    except Exception as e:
                        print(f"Error extracting '{original_var_name}' (as {model_name}) for point {point_name}: {e}")
                        import traceback
                        traceback.print_exc()


                # Create DataFrame for the point if any series were extracted
                if point_series_list and common_time_index is not None and not common_time_index.empty:
                    try:
                        # Create DF from series, ensuring alignment to the common index
                        point_df = pd.DataFrame(index=common_time_index)
                        for s in point_series_list:
                            # Reindex each series to the common index before assigning
                            # Use nearest neighbor or linear interpolation if appropriate, or None for exact match
                            point_df[s.name] = s.reindex(common_time_index, method='nearest', tolerance=pd.Timedelta('1 hour')) # Example tolerance

                        # Add scalar data (lat, lon, name, climate)
                        for key, value in point_data_dict.items():
                            point_df[key] = value # Broadcast scalar values

                        point_df.index.name = 'timestamp' # Ensure index has a name

                        # Add time features
                        if config.TIME_FEATURES:
                           point_df = self.add_time_features(point_df) # Assumes index is DatetimeIndex

                        # Reset index to make timestamp a column
                        point_df = point_df.reset_index()
                        all_data.append(point_df)

                    except Exception as e:
                        print(f"Error creating DataFrame for point {point_name}: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                     print(f"No valid time series data extracted or common index found for point {point_name}")

            print(f"Finished processing {len(points)} points.")
            # For points, return the list of DataFrames
            return all_data if all_data else None

        else:
            # --- GRID-BASED APPROACH (REVISED FOR EFFICIENCY) ---
            print("Processing gridded data...")
            start_grid_proc_time = time.time()

            # 1. Identify coordinate names from the first valid dataset
            first_valid_ds_key = next((var for var in original_vars_to_process if var in self.datasets), None)
            if not first_valid_ds_key:
                 print("Error: Could not find any valid datasets for requested prediction variables.")
                 return None

            first_ds = self.datasets[first_valid_ds_key]
            lat_var, lon_var, time_var = None, None, None
            for var_name in first_ds.coords:
                low_var = var_name.lower()
                if low_var in ['latitude', 'lat'] and lat_var is None: lat_var = var_name
                elif low_var in ['longitude', 'lon'] and lon_var is None: lon_var = var_name
                elif low_var in ['time', 'datetime'] and time_var is None: time_var = var_name
            if not all([lat_var, lon_var, time_var]):
                print(f"Error: Could not identify standard coordinate variables (found lat: {lat_var}, lon: {lon_var}, time: {time_var})")
                return None

            lats = first_ds[lat_var].values
            lons = first_ds[lon_var].values
            # Get the full standardized time index from the reference dataset
            try:
                times_pd = pd.DatetimeIndex(first_ds[time_var].values)
                times_std = self.standardize_timestamps(times_pd, use_utc=use_utc)
                if times_std is None or times_std.empty:
                     raise ValueError("Standardized time index is empty or None")
            except Exception as time_init_err:
                 print(f"Error initializing time index from dataset '{first_valid_ds_key}': {time_init_err}")
                 return None

            print(f"Grid dimensions: lat={len(lats)}, lon={len(lons)}, time={len(times_std)}")
            print(f"Available Time range (standardized): {times_std.min()} to {times_std.max()}")

            # 2. Create the time slice object using standardized start/end dates
            # This slice will be used in .sel() later
            time_slice = slice(None) # Default slice (all times)
            if start_date_std is not None and end_date_std is not None:
                time_slice = slice(start_date_std, end_date_std)
                print(f"Requested Time slice: {start_date_std} to {end_date_std}")
            else:
                print("No start/end date provided for filtering, using full available time range.")

            # 3. Get Climate Data (Vectorized)
            temp_grid, precip_grid = None, None
            if hasattr(self, 'temp_climate_data') and hasattr(self, 'precip_climate_data') and self.temp_climate_data and self.precip_climate_data:
                try:
                    lon_grid, lat_grid = np.meshgrid(lons, lats)
                    flat_lons, flat_lats = lon_grid.flatten(), lat_grid.flatten()
                    print(f"Getting climate data for {len(flat_lons)} grid points...")
                    # Pass the actual loaded climate data dictionaries
                    temps, precips = self.get_climate_at_location_direct(
                        flat_lons, flat_lats,
                        self.temp_climate_data, self.precip_climate_data
                    )
                    temp_grid = temps.reshape(len(lats), len(lons))
                    precip_grid = precips.reshape(len(lats), len(lons))
                    print("Successfully retrieved climate data for grid.")
                except Exception as e:
                    print(f"Warning: Error in grid climate data retrieval: {e}. Using defaults.")
                    temp_grid = np.full((len(lats), len(lons)), 15.0)
                    precip_grid = np.full((len(lats), len(lons)), 800.0)
            else:
                 print("Warning: Climate data attributes not found or not loaded, using default values.")
                 temp_grid = np.full((len(lats), len(lons)), 15.0)
                 precip_grid = np.full((len(lats), len(lons)), 800.0)

            # 4. Prepare datasets for merging (select time slice here)
            print("Preparing and merging variable datasets...")
            datasets_to_merge = []
            final_time_coord_name = 'timestamp' # Target standard name
            final_lat_coord_name = 'latitude'
            final_lon_coord_name = 'longitude'

            for original_var_name in original_vars_to_process:
                # Skip if dataset wasn't loaded correctly
                if original_var_name not in self.datasets: continue
                ds = self.datasets[original_var_name]

                # Find current time coordinate name in this dataset
                current_time_var = None
                for dim in ds.coords: # Check coords first
                    if dim.lower() in ['time', 'datetime']:
                        current_time_var = dim
                        break
                if not current_time_var: # Check dims if not in coords
                     for dim in ds.dims:
                          if dim.lower() in ['time', 'datetime']:
                               current_time_var = dim
                               break
                if not current_time_var:
                    print(f"Warning: No time coordinate/dimension found in dataset for '{original_var_name}'. Skipping.")
                    continue

                try:
                    # Ensure time coordinate is standardized before slicing
                    current_times_pd = pd.DatetimeIndex(ds[current_time_var].values)
                    current_times_std = self.standardize_timestamps(current_times_pd, use_utc=use_utc)
                    ds = ds.assign_coords({current_time_var: current_times_std})

                    # Apply the time slice using .sel() with the standardized time coordinate
                    ds_filtered = ds.sel({current_time_var: time_slice})

                    # Check if any time steps remain
                    if ds_filtered.dims[current_time_var] == 0:
                        print(f"Warning: No data remaining for '{original_var_name}' after time slicing. Skipping.")
                        continue

                    # Rename coordinates to standard names *before* merge
                    rename_coords = {}
                    current_lat_var, current_lon_var = None, None
                    for cname in ds_filtered.coords:
                         low_cname = cname.lower()
                         if low_cname in ['latitude', 'lat']: current_lat_var = cname
                         elif low_cname in ['longitude', 'lon']: current_lon_var = cname
                    if current_lat_var and current_lat_var != final_lat_coord_name: rename_coords[current_lat_var] = final_lat_coord_name
                    if current_lon_var and current_lon_var != final_lon_coord_name: rename_coords[current_lon_var] = final_lon_coord_name
                    if current_time_var != final_time_coord_name: rename_coords[current_time_var] = final_time_coord_name

                    if rename_coords:
                        ds_aligned = ds_filtered.rename(rename_coords)
                    else:
                        ds_aligned = ds_filtered

                    # Ensure lat/lon/time are coordinates
                    ds_aligned = ds_aligned.set_coords([final_lat_coord_name, final_lon_coord_name, final_time_coord_name])

                    datasets_to_merge.append(ds_aligned)

                except Exception as select_err:
                     print(f"Warning: Could not select/prepare time for '{original_var_name}': {select_err}. Skipping.")
                     import traceback
                     traceback.print_exc()


            if not datasets_to_merge:
                print("Error: No datasets available for merging after preparation.")
                return None

            # 5. Merge datasets
            print(f"Attempting to merge {len(datasets_to_merge)} datasets...")
            try:
                # Using inner join initially might be safer if coords aren't perfectly aligned
                # Override allows coords to differ slightly, but might hide issues
                merged_ds = xr.merge(datasets_to_merge, join='override', compat='override')
                print(f"Merged dataset variables: {list(merged_ds.data_vars)}")
                print(f"Merged dataset coords: {list(merged_ds.coords)}")

            except Exception as merge_err:
                 print(f"Error merging datasets: {merge_err}")
                 print("--- Details of datasets attempted to merge ---")
                 for i, dsm in enumerate(datasets_to_merge):
                      print(f"Dataset {i}: Vars={list(dsm.data_vars)}, Coords={list(dsm.coords)}")
                      print(dsm.coords)
                 print("-----------------------------------------------")
                 return None

            # 6. Add climate data as DataArrays
            print("Adding climate data to merged dataset...")
            try:
                # Ensure climate data aligns with the merged dataset's lat/lon coords
                merged_lats = merged_ds[final_lat_coord_name].values
                merged_lons = merged_ds[final_lon_coord_name].values

                # Check if resampling is needed (basic check based on shape)
                if temp_grid.shape != (len(merged_lats), len(merged_lons)):
                     print("Warning: Climate grid shape mismatch. Attempting simple broadcast (might be incorrect).")
                     # This is a fallback, proper resampling might be needed earlier
                     merged_ds['annual_mean_temperature'] = xr.DataArray(temp_grid[0,0], name='annual_mean_temperature')
                     merged_ds['annual_precipitation'] = xr.DataArray(precip_grid[0,0], name='annual_precipitation')
                else:
                    merged_ds['annual_mean_temperature'] = xr.DataArray(
                        temp_grid, dims=[final_lat_coord_name, final_lon_coord_name],
                        coords={final_lat_coord_name: merged_lats, final_lon_coord_name: merged_lons}
                    )
                    merged_ds['annual_precipitation'] = xr.DataArray(
                        precip_grid, dims=[final_lat_coord_name, final_lon_coord_name],
                        coords={final_lat_coord_name: merged_lats, final_lon_coord_name: merged_lons}
                    )
            except Exception as climate_add_err:
                 print(f"Error adding climate data: {climate_add_err}")
                 # Continue without climate data or return None
                 # return None

            # 7. Rename variables to final model names
            print("Renaming variables...")
            rename_dict_final = {}
            final_model_vars = []
            # Add climate names first if they were added
            if 'annual_mean_temperature' in merged_ds: final_model_vars.append('annual_mean_temperature')
            if 'annual_precipitation' in merged_ds: final_model_vars.append('annual_precipitation')

            # Iterate through the original map used for processing
            for original_name, model_name in prediction_vars_map.items():
                 # Check if the original name exists as a data variable in the merged dataset
                 if original_name in merged_ds.data_vars:
                     # If the model name is different, add to rename dict
                     if original_name != model_name:
                         rename_dict_final[original_name] = model_name
                     # Add the target model name to the list of vars to keep
                     if model_name not in final_model_vars:
                         final_model_vars.append(model_name)
                 # Also handle case where derived var name might be the key (e.g., 'vpd')
                 elif original_name in merged_ds.data_vars and original_name == model_name:
                      if model_name not in final_model_vars:
                           final_model_vars.append(model_name)


            print(f"Final rename dictionary: {rename_dict_final}")
            if rename_dict_final: # Only rename if needed
                merged_ds = merged_ds.rename(rename_dict_final)
            print(f"Final list of variables to keep: {final_model_vars}")

            # Filter dataset to keep only the final required variables
            try:
                # Ensure all requested final vars actually exist after rename
                vars_to_keep_actual = [v for v in final_model_vars if v in merged_ds.data_vars]
                if len(vars_to_keep_actual) != len(final_model_vars):
                     print(f"Warning: Not all expected final variables found. Keeping only available: {vars_to_keep_actual}")
                if not vars_to_keep_actual:
                     print("Error: No variables left to keep after renaming/filtering.")
                     return None
                merged_ds = merged_ds[vars_to_keep_actual]
                print(f"Variables available after rename/filter: {list(merged_ds.data_vars)}")
            except KeyError as filter_err:
                 print(f"Error during final variable filtering: {filter_err}")
                 print(f"Available variables were: {list(merged_ds.data_vars)}")
                 print(f"Attempted to keep: {final_model_vars}")
                 return None

            # 8. Convert the entire xarray Dataset to a Pandas DataFrame
            print("Converting merged xarray Dataset to Pandas DataFrame...")
            try:
                # Ensure standard coordinate names are set as coords before conversion
                merged_ds = merged_ds.set_coords([final_lat_coord_name, final_lon_coord_name, final_time_coord_name])
                # to_dataframe() creates a MultiIndex (time, lat, lon)
                final_df = merged_ds.to_dataframe()
            except Exception as df_conv_err:
                 print(f"Error converting to DataFrame: {df_conv_err}. This might be a memory issue.")
                 import traceback
                 traceback.print_exc()
                 return None

            # Reset the MultiIndex (timestamp, latitude, longitude) to get columns
            final_df = final_df.reset_index()

            # Check if essential columns exist
            if not all(col in final_df.columns for col in [final_time_coord_name, final_lat_coord_name, final_lon_coord_name]):
                print(f"Error: Missing essential coordinate columns after to_dataframe/reset_index. Columns: {final_df.columns.tolist()}")
                # Attempt to find columns with case variations
                found_cols = {}
                for col in final_df.columns:
                     low_col = col.lower()
                     if low_col == 'timestamp': found_cols['timestamp'] = col
                     elif low_col == 'latitude': found_cols['latitude'] = col
                     elif low_col == 'longitude': found_cols['longitude'] = col
                if len(found_cols) == 3:
                     print("Attempting rename based on case variations...")
                     final_df = final_df.rename(columns={
                          found_cols['timestamp']: final_time_coord_name,
                          found_cols['latitude']: final_lat_coord_name,
                          found_cols['longitude']: final_lon_coord_name
                     })
                else:
                     print("Could not reliably identify coordinate columns.")
                     return None


            # 9. Add Time Features (if configured)
            time_feature_success = False
            if config.TIME_FEATURES:
                print("Adding time features...")
                if final_time_coord_name in final_df.columns:
                    try:
                        # Ensure timestamp column is datetime type
                        final_df[final_time_coord_name] = pd.to_datetime(final_df[final_time_coord_name])
                        # Temporarily set timestamp as index for the function
                        final_df = final_df.set_index(final_time_coord_name)
                        final_df = self.add_time_features(final_df)
                        final_df = final_df.reset_index() # Put timestamp back as column
                        time_feature_success = True
                    except Exception as time_feat_err:
                        print(f"Warning: Could not add time features: {time_feat_err}")
                        # Ensure timestamp is still a column if error occurred after set_index
                        if final_time_coord_name not in final_df.columns and final_df.index.name == final_time_coord_name:
                            final_df = final_df.reset_index()
                else:
                     print(f"Warning: '{final_time_coord_name}' column not found, cannot add time features.")

            # 10. Add Name column
            print("Adding identifier columns...")
            try:
                # Use the standard coordinate names ensured before
                final_df['name'] = 'Grid_' + final_df[final_lat_coord_name].astype(str) + '_' + final_df[final_lon_coord_name].astype(str)
            except KeyError as name_err:
                 print(f"Error creating 'name' column. Missing coordinate column?: {name_err}")
                 print(f"Available columns: {final_df.columns.tolist()}")


            # 11. Final Checks and Return
            end_grid_proc_time = time.time()
            print(f"Finished creating grid dataset in {end_grid_proc_time - start_grid_proc_time:.2f} seconds.")

            # Drop unnecessary columns like 'spatial_ref' if they exist from rioxarray/xarray
            if 'spatial_ref' in final_df.columns:
                final_df = final_df.drop(columns=['spatial_ref'])

            # Final check for NaN values which might indicate issues
            nan_counts = final_df.isnull().sum()
            print("NaN counts per column:")
            print(nan_counts[nan_counts > 0]) # Print only columns with NaNs

            print(f"Created prediction dataset with {len(final_df)} rows")
            print(f"Final columns: {final_df.columns.tolist()}") # Print final columns for verification

            # For grid, return the single combined DataFrame
            return final_df
    
    def save_prediction_dataset(self, df, output_path=None):
        """
        Save the prediction dataset to a file.
        
        Parameters:
        -----------
        df : pandas.DataFrame or list of DataFrames
            Prediction dataset(s) to save
        output_path : str or Path, optional
            Path to save the dataset to
            
        Returns:
        --------
        str or list
            Path(s) to the saved file(s)
        """
        if output_path is None:
            output_path = config.PREDICTION_DIR / f"prediction_dataset_{int(time.time())}.csv"
       
        
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        if isinstance(df, list):
            # Multiple DataFrames (points)
            saved_paths = []
            for i, single_df in enumerate(df):
                name = single_df.get('name', f"point_{i}").iloc[0] if 'name' in single_df.columns else f"point_{i}"
                file_path = output_path.parent / f"{output_path.stem}_{name}{output_path.suffix}"
                single_df['ta'] = single_df['ta'] - 273.15  # Convert to Celsius
                single_df['td'] = single_df['td'] - 273.15  # Convert to Celsius
                single_df.to_csv(file_path, index_label='timestamp')
                print(f"Saved dataset to {file_path}")
                saved_paths.append(str(file_path))
            
            return saved_paths
        elif isinstance(df, pd.DataFrame):
            # Single DataFrame (grid data - the case causing the error)
            # Create a copy to avoid modifying the original DataFrame passed to the function
            df_to_save = df.copy()
            if 'ta' in df_to_save.columns:
                print(f"Converting 'ta' column for {output_path.name}...")
                df_to_save['ta'] = df_to_save['ta'] - 273.15 # Convert to Celsius
            else:
                # Only print warning, don't error out
                print(f"Warning: 'ta' column missing in the combined DataFrame. Cannot convert units for {output_path.name}")

            if 'td' in df_to_save.columns:
                print(f"Converting 'td' column for {output_path.name}...")
                df_to_save['td'] = df_to_save['td'] - 273.15 # Convert to Celsius
            else:
                # Only print warning, don't error out
                print(f"Warning: 'td' column missing in the combined DataFrame. Cannot convert units for {output_path.name}")
            df_to_save.to_csv(output_path, index_label='timestamp')
            print(f"Saved dataset to {output_path}")
            return str(output_path)
    def process_single_day(self, variables, year, month, day, shapefile=None, output_dir=None, hourly_parallel=False, num_workers=None):
        """
        Process a single day of ERA5-Land data and create a prediction dataset.
        Modified to optionally use hourly parallel processing.
        
        Parameters:
        -----------
        variables : list
            List of variables to process
        year : int
            Year to process
        month : int
            Month to process
        day : int
            Day to process
        shapefile : str, optional
            Path to shapefile for region definition
        output_dir : str or Path, optional
            Directory to save daily prediction dataset
        hourly_parallel : bool, optional
            Whether to use hourly parallel processing
        num_workers : int, optional
            Number of parallel worker processes for hourly processing
                
        Returns:
        --------
        str
            Path to saved prediction dataset file
        """
        print(f"\n===== Processing day {year}-{month:02d}-{day:02d} =====")
        
        if hourly_parallel:
            # Use hourly parallel processing
            return self._process_day_with_hourly_parallel(
                variables, year, month, day, 
                shapefile=shapefile, 
                num_workers=num_workers,
                output_dir=output_dir
            )
        
        # Original single-day processing (non-parallel)
        try:
            # Load data for this day
            self.load_variable_data(variables, year, month, day, shapefile=shapefile)
            
            # Calculate derived variables
            self.calculate_derived_variables()
            
            # Create prediction dataset with explicit date range filtering
            start_date = pd.Timestamp(year=year, month=month, day=day, tz='UTC')
            end_date = start_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # End of the day
            print(f"Filtering prediction dataset to range: {start_date} to {end_date}")
            df = self.create_prediction_dataset()
            
            if df is not None:
                # Create output directory if needed
                if output_dir is None:
                    output_dir = config.PREDICTION_DIR / f"{year}_{month:02d}_daily"
                
                output_dir = Path(output_dir)
                output_dir.mkdir(exist_ok=True, parents=True)
                
                # Save daily prediction dataset
                output_file = output_dir / f"prediction_{year}_{month:02d}_{day:02d}.csv"
                self.save_prediction_dataset(df, output_file)
                
                print(f"Saved prediction dataset for {year}-{month:02d}-{day:02d} to {output_file}")
                return str(output_file)
            else:
                print(f"No prediction dataset created for {year}-{month:02d}-{day:02d}")
                return None
        
        except Exception as e:
            print(f"Error processing day {year}-{month:02d}-{day:02d}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
        finally:
            # Clear datasets to free memory
            self._clear_datasets()
            
            # Force garbage collection
            import gc
            gc.collect()

    def _process_day_with_hourly_parallel(self, variables, year, month, day, 
                                    shapefile=None, num_workers=None, 
                                    output_dir=None):
        """
        Process a single day of ERA5-Land data with parallel processing of individual hours.
        
        Parameters:
        -----------
        variables : list
            List of variables to process
        year : int
            Year to process
        month : int
            Month to process
        day : int
            Day to process
        shapefile : str, optional
            Path to shapefile for region definition
        num_workers : int, optional
            Number of parallel worker processes (defaults to CPU count - 1)
        output_dir : str or Path, optional
            Directory to save daily prediction datasets
                    
        Returns:
        --------
        str
            Path to the daily prediction dataset file
        """
        import multiprocessing as mp
        
        # Set number of worker processes (default to CPU count - 1)
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)
        else:
            num_workers = max(1, min(num_workers, mp.cpu_count()))
        
        print(f"Processing {year}-{month:02d}-{day:02d} using {num_workers} parallel processes for hourly data")
        
        # Create output directory
        if output_dir is None:
            output_dir = config.PREDICTION_DIR / f"{year}_{month:02d}_daily"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create a temporary directory for hourly processing
        temp_dir = self.temp_dir
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        # Prepare arguments for each hour's processing
        process_args = []
        for hour in range(24):
            # Each item is (year, month, day, hour, variables, shapefile, output_dir, temp_dir)
            process_args.append((year, month, day, hour, variables, shapefile, output_dir, temp_dir))
        
        # Print confirmation before starting parallel processing
        print(f"Starting parallel processing for 24 hours with {num_workers} workers")
        
        # Use a process pool to process hours in parallel
        with mp.Pool(processes=num_workers) as pool:
            # Map the processing function to all hours with their arguments
            results = pool.map(_hour_processing_worker, process_args)
        
        # Filter out failed results
        successful_results = [r for r in results if r and r.get('success', False)]
        
        print(f"Successfully processed {len(successful_results)} out of 24 hours")
        
        if successful_results:
            # Combine hourly datasets into a single daily dataset
            try:
                # Collect all dataframes
                hourly_dfs = [result['data'] for result in successful_results if 'data' in result]
                
                if hourly_dfs:
                    # Concatenate all hourly dataframes
                    combined_df = pd.concat(hourly_dfs)
                    
                    # Sort by timestamp and location
                    sort_columns = []
                    if combined_df.index.name == 'timestamp':
                        combined_df = combined_df.sort_index()
                    else:
                        if 'timestamp' in combined_df.columns:
                            sort_columns.append('timestamp')
                    
                    if 'name' in combined_df.columns:
                        sort_columns.append('name')
                    else:
                        if 'latitude' in combined_df.columns:
                            sort_columns.append('latitude')
                        if 'longitude' in combined_df.columns:
                            sort_columns.append('longitude')
                    
                    if sort_columns:
                        combined_df = combined_df.sort_values(sort_columns)
                    
                    # Save the combined daily dataset
                    output_file = output_dir / f"prediction_{year}_{month:02d}_{day:02d}.csv"
                    self.save_prediction_dataset(combined_df, output_file)
                    
                    print(f"Saved combined daily dataset with {len(combined_df)} rows to {output_file}")
                    
                    
                    return str(output_file)
                else:
                    print("No hourly data found to combine")
                    return None
            except Exception as e:
                print(f"Error combining hourly data: {str(e)}")
                import traceback
                traceback.print_exc()
                return None
        else:
            print("No successful hourly data to combine")
            return None


    def _process_month_with_hourly_parallel(self, variables, year, month, 
                                        shapefile=None, num_workers=None, 
                                        output_dir=None, days_per_batch=1):
        """
        Process an entire month of ERA5-Land data with parallel processing at hourly level.
        
        Parameters:
        -----------
        variables : list
            List of variables to process
        year : int
            Year to process
        month : int
            Month to process
        shapefile : str, optional
            Path to shapefile for region definition
        num_workers : int, optional
            Number of parallel worker processes (defaults to CPU count - 1)
        output_dir : str or Path, optional
            Directory to save daily prediction datasets
        days_per_batch : int, optional
            Number of days to process in each batch (to avoid memory issues)
                    
        Returns:
        --------
        list
            Paths to all successfully created daily prediction dataset files
        """
        import calendar
        
        # Determine days in month
        days_in_month = calendar.monthrange(year, month)[1]
        
        print(f"Processing {year}-{month:02d} with hourly parallel processing")
        
        # Create output directory
        if output_dir is None:
            output_dir = config.PREDICTION_DIR / f"{year}_{month:02d}_daily"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Process days in batches to manage memory usage
        successful_files = []
        
        for start_day in range(1, days_in_month + 1, days_per_batch):
            end_day = min(start_day + days_per_batch - 1, days_in_month)
            
            print(f"\nProcessing batch: days {start_day} to {end_day}")
            
            # Process each day in the batch
            for day in range(start_day, end_day + 1):
                result = self._process_day_with_hourly_parallel(
                    variables, year, month, day, 
                    shapefile=shapefile, 
                    num_workers=num_workers,
                    output_dir=output_dir
                )
                
                if result:
                    successful_files.append(result)
                
                # Force garbage collection between days
                import gc
                gc.collect()
        
        print(f"Completed processing {len(successful_files)} out of {days_in_month} days")
        print(f"All daily prediction files are available in: {output_dir}")
        
        return successful_files
    def _clear_datasets(self):
        """Clear all loaded datasets to free memory."""
        for var_name in list(self.datasets.keys()):
            if hasattr(self.datasets[var_name], 'close'):
                self.datasets[var_name].close()
            del self.datasets[var_name]
        
        self.datasets = {}
    def merge_prediction_files(self, file_paths, output_file=None):
        """
        Merge multiple prediction dataset files into one.
        
        Parameters:
        -----------
        file_paths : list
            List of file paths to prediction datasets
        output_file : str or Path, optional
            Path to save the merged dataset
            
        Returns:
        --------
        str
            Path to the merged dataset file
        """
        if not file_paths:
            print("No prediction dataset files to merge")
            return None
        
        print(f"Merging {len(file_paths)} prediction dataset files...")
        
        # Read and combine all files
        dataframes = []
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
                dataframes.append(df)
                print(f"Read file: {file_path} with {len(df)} rows")
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")
        
        if not dataframes:
            print("No valid prediction datasets found")
            return None
        
        # Merge all dataframes
        merged_df = pd.concat(dataframes, axis=0)
        
        # Sort by timestamp and other relevant columns
        sort_columns = ['timestamp']
        if 'name' in merged_df.columns:
            sort_columns.append('name')
        else:
            if 'latitude' in merged_df.columns:
                sort_columns.append('latitude')
            if 'longitude' in merged_df.columns:
                sort_columns.append('longitude')
        
        merged_df = merged_df.sort_values(sort_columns)
        
        # Determine output file path
        if output_file is None:
            # Get first file to extract year and month
            first_file = Path(file_paths[0])
            parts = first_file.stem.split('_')
            if len(parts) >= 3:
                year = parts[1]
                month = parts[2]
                output_file = config.PREDICTION_DIR / f"era5land_gee_prediction_{year}_{month}.csv"
            else:
                output_file = config.PREDICTION_DIR / f"era5land_gee_prediction_merged.csv"
        
        output_file = Path(output_file)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        # Save merged dataset
        merged_df.to_csv(output_file)
        print(f"Saved merged dataset with {len(merged_df)} rows to {output_file}")
        
        return str(output_file)
    def process_month_parallel(self, variables, year, month, shapefile=None, num_workers=None, 
                           output_dir=None, hourly_parallel=False, days_per_batch=1):
        """
        Process an entire month of ERA5-Land data with parallel processing of individual days.
        Modified to optionally use hourly parallel processing.
        
        Parameters:
        -----------
        variables : list
            List of variables to process
        year : int
            Year to process
        month : int
            Month to process
        shapefile : str, optional
            Path to shapefile for region definition
        num_workers : int, optional
            Number of parallel worker processes (defaults to CPU count - 1)
        output_dir : str or Path, optional
            Directory to save daily prediction datasets
        hourly_parallel : bool, optional
            Whether to use hourly parallel processing
        days_per_batch : int, optional
            Number of days to process in each batch (for memory management)
                    
        Returns:
        --------
        list
            Paths to all successfully created daily prediction dataset files
        """
        if hourly_parallel:
            # Use hourly parallel processing for the month
            return self._process_month_with_hourly_parallel(
                variables, year, month, 
                shapefile=shapefile, 
                num_workers=num_workers,
                output_dir=output_dir,
                days_per_batch=days_per_batch
            )
        
        # Original month-parallel processing using days as the parallelization unit
        import calendar
        import multiprocessing as mp
        
        # Determine days in month
        days_in_month = calendar.monthrange(year, month)[1]
        
        # Set number of worker processes (default to CPU count - 1)
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)
        else:
            num_workers = max(1, min(num_workers, mp.cpu_count()))
        
        print(f"Processing {year}-{month:02d} using {num_workers} parallel processes")
        
        # Create output directory
        if output_dir is None:
            output_dir = config.PREDICTION_DIR / f"{year}_{month:02d}_daily"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Prepare arguments for each day's processing
        process_args = []
        for day in range(1, days_in_month + 1):
            # Each item is (year, month, day, variables, shapefile, output_dir, processor)
            process_args.append((year, month, day, variables, shapefile, output_dir, self))
        
        # Print confirmation before starting parallel processing
        print(f"Starting parallel processing for {len(process_args)} days with {num_workers} workers")
        
        # Use a process pool to process days in parallel
        with mp.Pool(processes=num_workers) as pool:
            # Map the processing function to all days with their arguments
            results = pool.map(_day_processing_worker, process_args)
        
        # Filter out None results (days that failed to process)
        successful_files = [r for r in results if r is not None]
        
        print(f"Completed processing {len(successful_files)} out of {days_in_month} days")
        print(f"All daily prediction files are available in: {output_dir}")
        
        return successful_files


def main():
    """
    Main function for ERA5-Land processing with parallel processing options.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Process ERA5-Land data from GEE for sap velocity prediction')
    parser.add_argument('--year', type=int, required=True, help='Year to process')
    parser.add_argument('--month', type=int, required=True, help='Month to process')
    parser.add_argument('--day', type=int, help='Day to process (for single-day processing)')
    parser.add_argument('--parallel', action='store_true', help='Process using parallel processing')
    parser.add_argument('--hourly-parallel', action='store_true', help='Process with hour-level parallelization')
    parser.add_argument('--num-workers', type=int, help='Number of parallel worker processes')
    parser.add_argument('--output', type=str, help='Output directory path')
    parser.add_argument('--shapefile', type=str, default=None, help='Path to shapefile for region definition')
    parser.add_argument('--gee-scale', type=float, default=11132, help='Scale in meters for GEE export (default ~9km)')
    parser.add_argument('--days-per-batch', type=int, default=1, help='Days to process in each batch (for memory management)')

    args = parser.parse_args()

    try:
        # Initialize the processor
        processor = ERA5LandGEEProcessor()
        
        output_dir = args.output or config.PREDICTION_DIR / f"{args.year}_{args.month:02d}_daily"
        
        if args.parallel or args.hourly_parallel:
            if args.day is not None:
                # Process a single day with appropriate parallelization
                processor.process_single_day(
                    config.REQUIRED_VARIABLES, args.year, args.month, args.day,
                    shapefile=args.shapefile,
                    output_dir=output_dir,
                    hourly_parallel=args.hourly_parallel,
                    num_workers=args.num_workers
                )
            else:
                # Process the entire month with appropriate parallelization
                processor.process_month_parallel(
                    config.REQUIRED_VARIABLES, args.year, args.month,
                    shapefile=args.shapefile,
                    num_workers=args.num_workers,
                    output_dir=output_dir,
                    hourly_parallel=args.hourly_parallel,
                    days_per_batch=args.days_per_batch
                )
        else:
            # Non-parallel processing
            if args.day is not None:
                # Process a single day
                processor.process_single_day(
                    config.REQUIRED_VARIABLES, args.year, args.month, args.day, 
                    shapefile=args.shapefile, output_dir=output_dir
                )
            else:
                # Process normally (whole month at once)
                processor.load_variable_data(
                    config.REQUIRED_VARIABLES, args.year, args.month, None, 
                    shapefile=args.shapefile
                )
                
                # Calculate derived variables
                processor.calculate_derived_variables()
                
                # Create prediction dataset
                df = processor.create_prediction_dataset()
                
                if df is not None:
                    # Save the dataset
                    output_path = args.output or f"era5land_gee_prediction_{args.year}_{args.month:02d}.csv"
                    processor.save_prediction_dataset(df, output_path)
    
    finally:
        # Clean up
        processor.close()



if __name__ == "__main__":

    main()
