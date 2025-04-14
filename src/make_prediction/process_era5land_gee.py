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
def _biome_batch_processor(args):
    """
    Process a batch of locations for biome determination.
    This function must be at module level to be picklable for multiprocessing.
    
    Parameters:
    -----------
    args : tuple
        (processor_settings, points_batch)
        processor_settings: dict containing temp_climate_file, precip_climate_file
        points_batch: list of (lon, lat) tuples
        
    Returns:
    --------
    dict
        Dictionary mapping location keys to biome types
    """
    # Recreate a minimal processor just for biome determination
    temp_file, precip_file = args[0]
    points_batch = args[1]
    
    # Create a minimal processor
    processor = ERA5LandGEEProcessor(
        gee_initialize=False,  # Don't need GEE for biome calculation
        temp_climate_file=temp_file,
        precip_climate_file=precip_file
    )
    
    # Process each point and build the cache
    result_cache = {}
    try:
        for lon, lat in points_batch:
            # Create a key for the location
            loc_key = (round(lon, 5), round(lat, 5))
            
            # Get climate values at the location
            temperature, precipitation = processor.get_climate_at_location(lon, lat)
            
            # Determine biome from climate
            biome_type = processor.determine_biome_from_climate(temperature, precipitation)
            
            # Store in local result cache
            result_cache[loc_key] = biome_type
            
    except Exception as e:
        print(f"Error in biome batch processor: {str(e)}")
    finally:
        # Clean up resources
        processor.close()
    
    return result_cache
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
    year, month, day, variables, shapefile, output_dir = args
    
    # Create a new processor instance for each process
    day_processor = ERA5LandGEEProcessor()
    try:
        # Load climate data first for biome determination
        day_processor.load_climate_data()
        
        # Try to load pre-calculated biome cache if it exists
        biome_cache_file = Path(output_dir) / "biome_cache.pkl"
        if biome_cache_file.exists():
            try:
                import pickle
                with open(biome_cache_file, 'rb') as f:
                    day_processor.biome_cache = pickle.load(f)
                print(f"Process {os.getpid()}: Loaded {len(day_processor.biome_cache)} biomes from cache")
            except Exception as e:
                print(f"Process {os.getpid()}: Error loading biome cache: {e}")
        
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
    # Define Whittaker biome classification thresholds
    # Format: (temp_min, temp_max, precip_min, precip_max)
    # Temperatures in °C, precipitation in mm/year
    BIOME_CLIMATE_THRESHOLDS = {
        'Tropical rain forest': (20, 30, 2000, 10000),
        'Tropical forest savanna': (20, 30, 1000, 2000),
        'Subtropical desert': (15, 30, 0, 250),
        'Temperate rain forest': (5, 18, 1500, 3000),
        'Temperate forest': (5, 20, 750, 1500),
        'Woodland/Shrubland': (8, 20, 250, 750),
        'Temperate grassland desert': (0, 20, 0, 500),
        'Boreal forest': (-5, 5, 300, 750),
        'Tundra': (-15, 0, 0, 250)
    }
    
    # ERA5-Land variable names in Google Earth Engine
    GEE_VARIABLE_MAPPING = {
        '2m_temperature': 'temperature_2m',
        '2m_dewpoint_temperature': 'dewpoint_temperature_2m',
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
        
        # Add biome cache
        self.biome_cache = {}
        
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
        
        # Store biome data
        self.biome_data = None
        self.biome_kdtree = None
        
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
            ee.Initialize(project='era5download-447713')
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
    def get_era5land_variable_from_gee(self, variable, year, month, day, scale=None, shapefile=None, region=None):
        """
        Access ERA5-Land data from Google Earth Engine for a specific variable and time period.
        
        Parameters:
        -----------
        variable : str
            Variable name (e.g., '2m_temperature')
        year : int
            Year to access
        month : int, optional
            Month to access (if None, access all available months)
        region : dict or ee.Geometry, optional
            Region to extract data for (default is bounding box from config)
        scale : float, optional
            Scale in meters for the Earth Engine export (defaults to ~9km for ERA5-Land)
            
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
        print(f"Accessing ERA5-Land variable '{gee_var}' from Google Earth Engine")
        
        try:
            # Set default region if not specified
            if region is None:
                region = ee.Geometry.Polygon(
                [[[config.DEFAULT_LON_MIN, config.DEFAULT_LAT_MIN], [config.DEFAULT_LON_MAX, config.DEFAULT_LAT_MIN], [config.DEFAULT_LON_MAX, config.DEFAULT_LAT_MAX], [config.DEFAULT_LON_MIN, config.DEFAULT_LAT_MAX], [config.DEFAULT_LON_MIN, config.DEFAULT_LAT_MIN]]],
                None, False # Planar coordinates for global extent
            )
                print(f"Using default region: {region.getInfo()}")
                print(f"default settings: {config.DEFAULT_LON_MIN}, {config.DEFAULT_LAT_MIN}, {config.DEFAULT_LON_MAX}, {config.DEFAULT_LAT_MAX}")
                print(f"Using specified region: {region.getInfo()}")

            # Set default scale if not specified (ERA5-Land is ~9km)
            if scale is None:
                scale = 11132  # 9km in meters
            
            # Set up date range
            if month is not None:
                if day is not None:
                    start_date = ee.Date.fromYMD(year, month, 1)
                    # if readch the last day of the month, set to next month
                    if day == calendar.monthrange(year, month)[1]:
                        if month == 12:
                            end_date = ee.Date.fromYMD(year + 1, 1, 1)
                        else:
                            end_date = ee.Date.fromYMD(year, month + 1, 1)
                    else:
                        end_date = ee.Date.fromYMD(year, month, day + 1)
                else:
                    start_date = ee.Date.fromYMD(year, month, 1)
                    if month == 12:
                        end_date = ee.Date.fromYMD(year + 1, 1, 1)
                    else:
                        end_date = ee.Date.fromYMD(year, month + 1, 1)
            else:
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
                            
                            # Calculate coordinates using the transform
                            img_lon_min = transform_list[2]  # x origin
                            img_lat_max = transform_list[5]  # y origin
                            img_lon_max = img_lon_min + width * transform_list[0]  # x origin + width * pixel width
                            img_lat_min = img_lat_max + height * transform_list[4]  # y origin + height * pixel height (usually negative)
                            
                            # Use image bounds for more accurate coordinate generation
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
                    
                # Create coordinate arrays (ensure latitude decreases if scale_y is negative)
                lons = np.linspace(lon_min, lon_max, width, endpoint=False) # endpoint=False often matches raster conventions
                if 'transform_list' in locals() and transform_list[4] < 0:
                    lats = np.linspace(lat_max, lat_min, height, endpoint=False) # Lat decreases
                else:
                    lats = np.linspace(lat_min, lat_max, height, endpoint=False) # Lat increases (less common)

                print(f"Final Coordinates: lon={lons.min():.4f}-{lons.max():.4f}, lat={lats.min():.4f}-{lats.max():.4f}")
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
                        timestamps.append(pd.to_datetime(time_start), utc=True)
                        
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
            'surface_solar_radiation_downwards': 'W m-2',
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
    
    def load_variable_data(self, variables, year, month, day, shapefile=None):
        """
        Load and preprocess multiple ERA5-Land variables, potentially clipping to a shapefile.

        Parameters:
        -----------
        variables : list
            List of variable names to load (e.g., ['temperature_2m', 'total_precipitation']).
        year : int
            Year to load data for.
        month : int
            Month to load data for.
        day : int
            Day to load data for.
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
        print("Clearing previously loaded datasets...")
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
            print(f"\nProcessing variable: {var_name} for {year}-{month:02d}...")

            # 1. Get data from GEE (or other source)
            # Ideally, pass region_to_extract to GEE function if it supports server-side filtering
            try:
                ds = self.get_era5land_variable_from_gee(var_name, year, month, day, shapefile=shapefile) # Pass shapefile path for context if needed by GEE func
            except Exception as gee_err:
                 print(f"  ERROR: Failed to load data for {var_name} from GEE source: {gee_err}")
                 ds = None

            if ds is not None:
                print(f"  Initial data loaded for {var_name}. Size: {ds.nbytes / 1e6:.2f} MB")
                print(f"  Initial dimensions: {ds.dims}")
                print(f"  Coordinates: {list(ds.coords.keys())}")


                # 3. Apply Shapefile Mask (if shapefile was provided and libraries exist)
                # Replace the problematic section with this:
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
                                    # Extract single time slice for solar radiation
                                    if is_accumulated and t_idx > 0:
                                        # For accumulated values, compute difference between consecutive time steps
                                        current = sr_ds[sr_var].isel(time=t_idx).values
                                        previous = sr_ds[sr_var].isel(time=t_idx-1).values
                                        sr_slice_full = (current - previous) / time_diff_seconds[t_idx]
                                    else:
                                        # For non-accumulated or first step, use the value directly
                                        sr_slice_full = sr_ds[sr_var].isel(time=t_idx).values
                                    
                                    # Process in latitude chunks to avoid memory issues
                                    for lat_chunk_start in range(0, num_lats, LAT_CHUNK_SIZE):
                                        lat_chunk_end = min(lat_chunk_start + LAT_CHUNK_SIZE, num_lats)
                                        lat_chunk_size = lat_chunk_end - lat_chunk_start
                                        
                                        # Extract chunk of solar radiation data
                                        sr_slice = sr_slice_full[lat_chunk_start:lat_chunk_end, :]
                                        
                                        # Calculate PAR (Photosynthetically Active Radiation)
                                        par_fraction = 0.45
                                        par_wm2 = sr_slice * par_fraction
                                        
                                        # Convert PAR from W/m2 to μmol/m2/s
                                        conversion_factor = 4.6
                                        ppfd_slice = par_wm2 * conversion_factor
                                        
                                        # Store PPFD result in final array
                                        ppfd_array[t_idx, lat_chunk_start:lat_chunk_end, :] = ppfd_slice
                                        
                                        # Prepare for extraterrestrial radiation calculation
                                        jd = jd_array[t_idx]
                                        X = X_array[t_idx]
                                        day_of_year = dn[t_idx]
                                        t_value = times[t_idx]
                                        
                                        # Calculate declination based on method - ORIGINAL LOGIC
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
                                    
                                    # Clean up time slice memory
                                    del sr_slice_full
                                    
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
                                    # Extract single time slice for solar radiation
                                    if is_accumulated and t_idx > 0:
                                        current = sr_ds[sr_var].isel(time=t_idx).values
                                        previous = sr_ds[sr_var].isel(time=t_idx-1).values
                                        sr_slice_full = (current - previous) / time_diff_seconds[t_idx]
                                    else:
                                        sr_slice_full = sr_ds[sr_var].isel(time=t_idx).values
                                    
                                    # Process in chunks to reduce memory usage
                                    for lat_chunk_start in range(0, num_lats, LAT_CHUNK_SIZE):
                                        lat_chunk_end = min(lat_chunk_start + LAT_CHUNK_SIZE, num_lats)
                                        
                                        # Extract chunk
                                        sr_slice = sr_slice_full[lat_chunk_start:lat_chunk_end, :]
                                        
                                        # Calculate PAR
                                        par_fraction = 0.45
                                        par_wm2 = sr_slice * par_fraction
                                        ppfd_slice = par_wm2 * 4.6
                                        
                                        # Simple extraterrestrial radiation approximation
                                        ext_rad_slice = sr_slice * 1.5
                                        
                                        # Store in final arrays
                                        ppfd_array[t_idx, lat_chunk_start:lat_chunk_end, :] = ppfd_slice
                                        ext_rad_array[t_idx, lat_chunk_start:lat_chunk_end, :] = ext_rad_slice
                                        
                                        # Clean up memory
                                        del sr_slice, par_wm2, ppfd_slice, ext_rad_slice
                                    
                                    # Clean up time slice memory
                                    del sr_slice_full
                                    
                                    # Status update
                                    if t_idx % max(1, len(times)//10) == 0 or t_idx == len(times) - 1:
                                        print(f"Processed PAR variables for time {t_idx+1}/{len(times)} (simplified method)")
                        else:
                            # If no lat_var found, use simpler approach - SAME AS ORIGINAL FALLBACK
                            print("Latitude coordinate not found, using solar radiation as proxy for ext_rad")
                            
                            # Initialize arrays
                            ppfd_array = np.zeros((len(times), num_lats, num_lons), dtype=np.float32)
                            ext_rad_array = np.zeros((len(times), num_lats, num_lons), dtype=np.float32)
                            
                            # Process each time step with simplified approach
                            for t_idx in range(len(times)):
                                # Extract single time slice
                                if is_accumulated and t_idx > 0:
                                    current = sr_ds[sr_var].isel(time=t_idx).values
                                    previous = sr_ds[sr_var].isel(time=t_idx-1).values
                                    sr_slice_full = (current - previous) / time_diff_seconds[t_idx]
                                else:
                                    sr_slice_full = sr_ds[sr_var].isel(time=t_idx).values
                                
                                # Process in chunks
                                for lat_chunk_start in range(0, num_lats, LAT_CHUNK_SIZE):
                                    lat_chunk_end = min(lat_chunk_start + LAT_CHUNK_SIZE, num_lats)
                                    
                                    # Extract chunk
                                    sr_slice = sr_slice_full[lat_chunk_start:lat_chunk_end, :]
                                    
                                    # Calculate PAR
                                    par_fraction = 0.45
                                    par_wm2 = sr_slice * par_fraction
                                    ppfd_slice = par_wm2 * 4.6
                                    
                                    # Simple extraterrestrial radiation approximation
                                    ext_rad_slice = sr_slice * 1.5
                                    
                                    # Store in final arrays
                                    ppfd_array[t_idx, lat_chunk_start:lat_chunk_end, :] = ppfd_slice
                                    ext_rad_array[t_idx, lat_chunk_start:lat_chunk_end, :] = ext_rad_slice
                                    
                                    # Clean up memory
                                    del sr_slice, par_wm2, ppfd_slice, ext_rad_slice
                                
                                # Clean up time slice memory
                                del sr_slice_full
                                
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
                        # For datasets without time dimension - EXACT SAME AS ORIGINAL
                        # Standard approach using named variables
                        solar_rad = sr_ds[sr_var]
                        
                        # Calculate PAR (Photosynthetically Active Radiation)
                        par_fraction = 0.45
                        par_wm2 = solar_rad * par_fraction
                        
                        # Convert PAR from W/m2 to μmol/m2/s
                        conversion_factor = 4.6
                        ppfd = par_wm2 * conversion_factor
                        
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
                                ext_rad_ds = xr.Dataset(
                                    data_vars={'ext_rad': ext_rad},
                                    coords=sr_ds.coords,
                                    attrs={'units': 'W m-2', 'long_name': 'Extraterrestrial Radiation (approximated)'}
                                )
                        else:
                            print("Latitude coordinate not found, using solar radiation as proxy for ext_rad")
                            ext_rad = solar_rad * 1.5  # Simple approximation
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

    def determine_biome_from_climate(self, temp, precip):
        """
        Determine biome type based on temperature and precipitation values
        using Whittaker biome classification with curved boundaries.
        
        Parameters:
        -----------
        temp : float
            Annual mean temperature in °C
        precip : float
            Annual mean precipitation in mm/year
                
        Returns:
        --------
        str
            Biome type
        """
        # Default biome if no match is found
        default_biome = 'Temperate forest'
        
        # Handle missing values
        if np.isnan(temp) or np.isnan(precip):
            print(f"Missing climate data: temp={temp}, precip={precip}")
            return default_biome
        
        # Define curved boundary functions for each biome transition
        # These functions more accurately represent the Whittaker diagram
        
        # Tropical rain forest - typically high rainfall, consistently warm
        def is_in_tropical_rainforest(t, p):
            if 20 <= t <= 30:
                # Boundary with tropical savanna, increases with temperature
                min_precip = 2300 + 45 * (t - 20)
                return p >= min_precip
            return False
        
        # Tropical forest savanna - seasonally dry tropics
        def is_in_tropical_savanna(t, p):
            if 18 <= t <= 30:
                min_precip = 500 + 42 * (t - 18)  # Lower boundary with subtropical desert
                max_precip = 2300 + 45 * (t - 20)    # Upper boundary with rainforest
                return min_precip <= p <= max_precip
            return False
        
        # Subtropical desert - hot, very dry
        def is_in_subtropical_desert(t, p):
            if 15 <= t <= 30:
                # Upper boundary with savanna, increases with temperature
                max_precip = 500 + 42 * (t - 18) 
                return 0 <= p <= max_precip
            return False
        
        # Temperate rain forest - mild temperatures, very wet
        def is_in_temperate_rainforest(t, p):
            if 3 <= t <= 19:
                # Curved lower boundary, decreases as it gets warmer
                min_precip = 400 + (t - 3) * 50
                max_precip = 1750 + (t - 3) * 35  # Upper boundary with temperate forest
                return min_precip <= p <= max_precip
            return False
        
        # Temperate forest - mild temperature, moderate rainfall
        def is_in_temperate_forest(t, p):
            if 4 <= t <= 20:
                
                min_precip = 1600 + 50 * ( t - 4 )
                # Upper boundary with temperate rainforest
                max_precip = 2000 + (t - 4) * 82 
                return min_precip <= p <= max_precip
            return False
        
        # Woodland/Shrubland - warm, relatively dry
        def is_in_woodland_shrubland(t, p):
            if -9 <= t <= 20:
                min_precip = 30 + (t + 9) * 17  # Curved lower boundary
                max_precip = 50 + ( t + 9) * 42
                return min_precip <= p <= max_precip
            return False
        
        # Temperate grassland/desert - cold to warm, very dry
        def is_in_temperate_grassland_desert(t, p):
            if -9 <= t <= 20:
                max_precip = 0 + (t - 10) * 16  # Curved upper boundary
                return 0 <= p <= max_precip
            return False
        
        # Boreal forest - cold, moderate precipitation
        def is_in_boreal_forest(t, p):
            if -7 <= t <= 5:
                min_precip = 150 + (t + 7) * 30  # Curved lower boundary
                max_precip = 1000 + (t + 7) * 83  # Curved upper boundary
                return min_precip <= p <= max_precip
            return False
        
        # Tundra - very cold, low precipitation
        def is_in_tundra(t, p):
            if -15 <= t <= -4:
                max_precip = 0 + (t + 15) * 91  # Precipitation increases with temperature
                return 0 <= p <= max_precip
            return False
        
        # Check each biome using the curved boundary functions
        # Order matters - check from wettest to driest within temperature zones
        if is_in_tropical_rainforest(temp, precip):
            return 'Tropical rain forest'
        elif is_in_tropical_savanna(temp, precip):
            return 'Tropical forest savanna'
        elif is_in_subtropical_desert(temp, precip):
            return 'Subtropical desert'
        elif is_in_temperate_rainforest(temp, precip):
            return 'Temperate rain forest'
        elif is_in_temperate_forest(temp, precip):
            return 'Temperate forest'
        elif is_in_woodland_shrubland(temp, precip):
            return 'Woodland/Shrubland'
        elif is_in_temperate_grassland_desert(temp, precip):
            return 'Temperate grassland desert'
        elif is_in_boreal_forest(temp, precip):
            return 'Boreal forest'
        elif is_in_tundra(temp, precip):
            return 'Tundra'
        
        # If no match found with curved boundaries, use distance-based approach as fallback
        # This is similar to the existing code but with some improvements
        min_distance = float('inf')
        closest_biome = default_biome
        
        # Define representative points for each biome (more accurate than rectangle center)
        biome_representative_points = {
            'Tropical rain forest': (25, 3300),          # °C, mm/year
            'Tropical forest savanna': (25, 1500),
            'Subtropical desert': (23, 400),
            'Temperate rain forest': (15, 2300),
            'Temperate forest': (12, 1500),
            'Woodland/Shrubland': (12, 500),
            'Temperate grassland desert': (10, 250),
            'Boreal forest': (0, 1000),
            'Tundra': (-10, 250)
        }
        
        # Calculate distance to representative points
        for biome, (biome_temp, biome_precip) in biome_representative_points.items():
            # Normalize temperature to 0-1 range over -15 to 30°C
            temp_range = 45  # -15 to 30°C
            temp_min = -15
            temp_norm = (temp - temp_min) / temp_range
            biome_temp_norm = (biome_temp - temp_min) / temp_range
            
            # Normalize precipitation using log scale to better represent its effect
            # Log scale makes more sense for precipitation differences
            precip_log = np.log1p(max(0, precip))  # log(1+x) to handle zero
            biome_precip_log = np.log1p(max(0, biome_precip))
            
            # Scale factors - precipitation often has less weight than temperature in biome determination
            temp_weight = 0.6
            precip_weight = 0.4
            
            # Calculate weighted distance
            distance = np.sqrt(
                temp_weight * (temp_norm - biome_temp_norm)**2 + 
                precip_weight * (precip_log / 8 - biome_precip_log / 8)**2  # Scaled to similar range
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_biome = biome
        
        print(f"No exact biome match for temp={temp}, precip={precip}. Using closest: {closest_biome}")
        return closest_biome

    def get_climate_at_location(self, lon, lat):
        """
        Get annual mean temperature and precipitation at a specific location
        using direct raster lookup on GeoTIFF data.
        
        Parameters:
        -----------
        lon, lat : float
            Coordinates of the location
            
        Returns:
        --------
        tuple
            (temperature, precipitation)
        """
        if not hasattr(self, 'temp_climate_data') or not hasattr(self, 'precip_climate_data'):
            # Try to load climate data if not already loaded
            if self.load_climate_data(bounds = {
                    'lon_min': -180.0,  # Eastern longitude boundary
                    'lat_min': -57,  # Southern latitude boundary
                    'lon_max': 180.0,  # Western longitude boundary 
                    'lat_max': 78.0   }) == (None, None):
                print("Could not load climate data, returning default values")
                return 15.0, 800.0  # Default values for temperate climate
        
        try:
            # Get temperature using direct raster lookup
            # Convert geographic coordinates to raster row/col
            row, col = ~self.temp_climate_data['transform'] * (lon, lat)
            row, col = int(row), int(col)
            
            # Make sure the indices are within bounds
            if 0 <= row < self.temp_climate_data['height'] and 0 <= col < self.temp_climate_data['width']:
                temperature = float(self.temp_climate_data['data'][row, col])
                
                # Check for nodata value
                if self.temp_climate_data['nodata'] is not None and temperature == self.temp_climate_data['nodata']:
                    temperature = 15.0  # Default value
                else:
                    temperature = float(temperature)  # Convert from tenths of degree to degrees Celsius
            else:
                temperature = 15.0  # Default value
            
            # Get precipitation using direct raster lookup
            # Convert geographic coordinates to raster row/col
            row, col = ~self.precip_climate_data['transform'] * (lon, lat)
            row, col = int(row), int(col)
            
            # Make sure the indices are within bounds
            if 0 <= row < self.precip_climate_data['height'] and 0 <= col < self.precip_climate_data['width']:
                precipitation = float(self.precip_climate_data['data'][row, col])
                
                # Check for nodata value
                if self.precip_climate_data['nodata'] is not None and precipitation == self.precip_climate_data['nodata']:
                    precipitation = 800.0  # Default value
            else:
                precipitation = 800.0  # Default value
            
            return temperature, precipitation
        
        except Exception as e:
            print(f"Error getting climate at location ({lon}, {lat}): {str(e)}")
            import traceback
            traceback.print_exc()
            return 15.0, 800.0  # Default values for temperate climate

    def resample_climate_data(self, target_lat=None, target_lon=None):
        """
        Resample climate data from GeoTIFF files to match a target resolution.
        
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
            
            # Extract data from source
            temp_data = self.temp_climate_data['data']
            temp_transform = self.temp_climate_data['transform']
            temp_nodata = self.temp_climate_data['nodata']
            
            precip_data = self.precip_climate_data['data']
            precip_transform = self.precip_climate_data['transform']
            precip_nodata = self.precip_climate_data['nodata']
            
            # Create target grid
            resampled_temp = np.zeros((len(target_lat), len(target_lon)), dtype=np.float32)
            resampled_precip = np.zeros((len(target_lat), len(target_lon)), dtype=np.float32)
            
            # For each point in the target grid, sample the source data
            for i, lat in enumerate(target_lat):
                for j, lon in enumerate(target_lon):
                    # Get temperature using direct raster lookup
                    try:
                        # Convert geographic coordinates to raster row/col
                        row, col = ~temp_transform * (lon, lat)
                        row, col = int(row), int(col)
                        
                        # Make sure the indices are within bounds
                        if 0 <= row < self.temp_climate_data['height'] and 0 <= col < self.temp_climate_data['width']:
                            value = temp_data[row, col]
                            
                            # Check for nodata value
                            if temp_nodata is not None and value == temp_nodata:
                                resampled_temp[i, j] = 15.0  # Default value
                            else:
                                resampled_temp[i, j] = value
                        else:
                            # Out of bounds
                            resampled_temp[i, j] = 15.0  # Default value
                    except Exception as e:
                        print(f"Error resampling temperature at ({lat}, {lon}): {str(e)}")
                        resampled_temp[i, j] = 15.0  # Default value
                    
                    # Get precipitation using direct raster lookup
                    try:
                        # Convert geographic coordinates to raster row/col
                        row, col = ~precip_transform * (lon, lat)
                        row, col = int(row), int(col)
                        
                        # Make sure the indices are within bounds
                        if 0 <= row < self.precip_climate_data['height'] and 0 <= col < self.precip_climate_data['width']:
                            value = precip_data[row, col]
                            
                            # Check for nodata value
                            if precip_nodata is not None and value == precip_nodata:
                                resampled_precip[i, j] = 800.0  # Default value
                            else:
                                resampled_precip[i, j] = value
                        else:
                            # Out of bounds
                            resampled_precip[i, j] = 800.0  # Default value
                    except Exception as e:
                        print(f"Error resampling precipitation at ({lat}, {lon}): {str(e)}")
                        resampled_precip[i, j] = 800.0  # Default value
            
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
            
            print("Climate data resampled successfully using direct sampling")
            return resampled_temp_da, resampled_precip_da
        
        except Exception as e:
            print(f"Error resampling climate data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def get_biome_at_location(self, lon, lat):
        """
        Get the biome type at a specific location using climate data.
        Now with caching to avoid redundant calculations.
        
        Parameters:
        -----------
        lon, lat : float
            Coordinates of the location
            
        Returns:
        --------
        str
            Biome type at the location
        """
        # Create a key for the location (round to reduce floating point issues)
        loc_key = (round(lon, 5), round(lat, 5))
        
        # Check if we already have this location in the cache
        if loc_key in self.biome_cache:
            return self.biome_cache[loc_key]
        
        # First try using climate data to determine biome
        try:
            # Get climate values at the location
            temperature, precipitation = self.get_climate_at_location(lon, lat)
            
            # Determine biome from climate
            biome_type = self.determine_biome_from_climate(temperature, precipitation)
            
            # Cache the result
            self.biome_cache[loc_key] = biome_type
            
            return biome_type
        
        except Exception as e:
            print(f"Error determining biome from climate at ({lon}, {lat}): {str(e)}")
            
            # Fall back to the original method if biome_data is available
            if self.biome_data is not None:
                try:
                    # Use existing shapefile lookup method from the original implementation
                    # First try the fast KDTree approach
                    if self.biome_kdtree is not None:
                        # Find nearest biome centroid
                        dist, idx = self.biome_kdtree.query((lon, lat))
                        biome_type = self.biome_data.iloc[idx]['biome_type']
                        
                        # Verify with point-in-polygon for the nearest and its neighbors
                        if 'biome_type' in self.biome_data.columns:
                            # Create a point geometry
                            from shapely.geometry import Point
                            point = Point(lon, lat)
                            
                            # Check if point is in the nearest polygon
                            if self.biome_data.iloc[idx].geometry.contains(point):
                                biome_type = self.biome_data.iloc[idx]['biome_type']
                                self.biome_cache[loc_key] = biome_type
                                return biome_type
                            
                            # If not in the nearest, check neighbors
                            distances, indices = self.biome_kdtree.query((lon, lat), k=5)
                            for i in indices:
                                if self.biome_data.iloc[i].geometry.contains(point):
                                    biome_type = self.biome_data.iloc[i]['biome_type']
                                    self.biome_cache[loc_key] = biome_type
                                    return biome_type
                        
                        # If no polygon contains the point, cache and return the nearest biome type
                        self.biome_cache[loc_key] = biome_type
                        return biome_type
                    
                    # Fallback to direct spatial query (slower)
                    from shapely.geometry import Point
                    point = Point(lon, lat)
                    
                    # Find biomes that contain the point
                    contains_point = self.biome_data.contains(point)
                    if contains_point.any():
                        biome_type = self.biome_data[contains_point].iloc[0]['biome_type']
                        self.biome_cache[loc_key] = biome_type
                        return biome_type
                    
                    # If point is not in any polygon, find the nearest
                    distances = self.biome_data.distance(point)
                    nearest_idx = distances.idxmin()
                    biome_type = self.biome_data.loc[nearest_idx, 'biome_type']
                    self.biome_cache[loc_key] = biome_type
                    return biome_type
                
                except Exception as inner_e:
                    print(f"Error in shapefile fallback: {str(inner_e)}")
            
            # If all else fails, return a default biome type
            default_biome = "Temperate forest"
            print(f"Using default biome for location ({lon}, {lat})")
            self.biome_cache[loc_key] = default_biome
            return default_biome
        
    def precalculate_biomes(self, points=None, region=None, num_workers=None):
        """
        Pre-calculate biomes for all points or a grid in the region of interest.
        Uses parallel processing for efficiency.
        
        Parameters:
        -----------
        points : list of dict, optional
            List of specific points [{lat, lon, name}, ...]
        region : dict, optional
            Region boundaries {lat_min, lat_max, lon_min, lon_max}
        resolution : int, optional
            Number of grid points to calculate for each dimension if using region
        num_workers : int, optional
            Number of worker processes to use (defaults to CPU count - 1)
        """
        import multiprocessing as mp
        
        print("Pre-calculating biomes for all locations using parallel processing...")
        
        # Set number of worker processes (default to CPU count - 1)
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)
        else:
            num_workers = max(1, min(num_workers, mp.cpu_count()))
        
        print(f"Using {num_workers} worker processes for biome determination")
        
        # Ensure climate data is loaded
        if not hasattr(self, 'temp_climate_data') or not hasattr(self, 'precip_climate_data'):
            self.load_climate_data(bounds = {
                    'lon_min': -180.0,  # Eastern longitude boundary
                    'lat_min': -57,  # Southern latitude boundary
                    'lon_max': 180.0,  # Western longitude boundary 
                    'lat_max': 78.0   })
        
        # Create list of points to process
        location_points = []
        
        if points:
            # Use specific points
            for point in points:
                location_points.append((point['lon'], point['lat']))
            print(f"Processing {len(location_points)} specific points")
        
        elif region:
            # Create a grid for the region
            lats = np.linspace(region['lat_min'], region['lat_max'], 1351)
            lons = np.linspace(region['lon_min'], region['lon_max'], 3601)
            
            for lat in lats:
                for lon in lons:
                    location_points.append((lon, lat))
            
            print(f"Processing {len(location_points)} grid points in region:")
            print(f"  Region: lat={region['lat_min']}-{region['lat_max']}, lon={region['lon_min']}-{region['lon_max']}")
            print(f"  Resolution: {1351}x{3601}")
        
        else:
            print("No points or region specified for biome precalculation")
            return
        
        # Split the points into batches for parallel processing
        batch_size = max(100, len(location_points) // (num_workers * 10))  # Aim for ~10 batches per worker
        batches = []
        
        for i in range(0, len(location_points), batch_size):
            batch = location_points[i:i+batch_size]
            batches.append(batch)
        
        print(f"Split {len(location_points)} points into {len(batches)} batches of ~{batch_size} points each")
        
        # Prepare processor settings to pass to worker processes
        processor_settings = (str(self.temp_climate_file), str(self.precip_climate_file))
        
        # Process batches in parallel
        print("Starting parallel biome determination...")
        start_time = time.time()
        
        with mp.Pool(processes=num_workers) as pool:
            # Each worker gets (processor_settings, batch_of_points)
            worker_args = [(processor_settings, batch) for batch in batches]
            results = pool.map(_biome_batch_processor, worker_args)
        
        # Merge all results into the main cache
        total_biomes = 0
        for result_cache in results:
            self.biome_cache.update(result_cache)
            total_biomes += len(result_cache)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Biome pre-calculation complete in {elapsed_time:.2f} seconds")
        print(f"Processed {total_biomes} locations, found {len(self.biome_cache)} unique biome locations")
        
        # Print biome distribution
        biome_counts = {}
        for biome in self.biome_cache.values():
            biome_counts[biome] = biome_counts.get(biome, 0) + 1
        
        print("Biome distribution:")
        for biome, count in sorted(biome_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {biome}: {count} locations ({count/len(self.biome_cache)*100:.1f}%)")
    def create_biome_one_hot(self, biome_type):
            """
            Create one-hot encoding for a biome type.
            
            Parameters:
            -----------
            biome_type : str
                Biome type to encode
                
            Returns:
            --------
            dict
                Dictionary with one-hot encoded biome features
            """
            result = {}
            
            # Initialize all biome types to 0
            for biome in config.BIOME_TYPES:
                result[biome] = 0.0
            
            # Set the matching biome to 1
            if biome_type in result:
                result[biome_type] = 1.0
            
            return result
        
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
        
        Parameters:
        -----------
        start_date : str or datetime, optional
            Start date for the prediction period (required if use_existing_data=False)
        end_date : str or datetime, optional
            End date for the prediction period (required if use_existing_data=False)
        points : list of dict, optional
            List of points to extract data for [{lat, lon, name}, ...]
        region : dict, optional
            Region to extract data for, if no points are specified
        use_existing_data : bool, optional
            If True, will use all data already loaded in self.datasets instead of 
            loading new data. When True, start_date and end_date are optional and
            only used to filter the existing data if provided.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with features ready for prediction
        """
        # Set default region if not specified
        if region is None:
            region = {
                'lat_min': config.DEFAULT_LAT_MIN,
                'lat_max': config.DEFAULT_LAT_MAX,
                'lon_min': config.DEFAULT_LON_MIN,
                'lon_max': config.DEFAULT_LON_MAX
            }
        
        # Make sure climate data is loaded for biome determination
        if hasattr(self, 'temp_climate_file') and hasattr(self, 'precip_climate_file'):
            if not hasattr(self, 'temp_climate_data') or not hasattr(self, 'precip_climate_data'):
                self.load_climate_data(bounds = {
                    'lon_min': -180.0,  # Eastern longitude boundary
                    'lat_min': -57,  # Southern latitude boundary
                    'lon_max': 180.0,  # Western longitude boundary 
                    'lat_max': 78.0   })
        
        # Check if we're using existing data
        if use_existing_data:
            if not self.datasets:
                raise ValueError("use_existing_data=True but no data has been loaded. Call load_variable_data() first.")
            
            # Determine time range from existing data if not specified
            if start_date is None or end_date is None:
                # Find min and max dates across all datasets
                # With this standardized version
                all_times = []
                for ds_name, ds in self.datasets.items():
                    if 'time' in ds.coords:
                        times = pd.to_datetime(ds.coords['time'].values)
                        all_times.extend(times)
                    elif 'datetime' in ds.coords:
                        times = pd.to_datetime(ds.coords['datetime'].values)
                        all_times.extend(times)

                # Standardize all timestamps before comparison
                use_utc = True  # Choose whichever standard you prefer
                all_times = self.standardize_timestamps(all_times, use_utc=use_utc)
                
                if not all_times:
                    raise ValueError("No time coordinates found in existing datasets")
                
                # Use entire date range from existing data if not specified
                if start_date is None:
                    start_date = min(all_times)
                    print(f"Using start_date from existing data: {start_date}")
                if end_date is None:
                    end_date = max(all_times)
                    print(f"Using end_date from existing data: {end_date}")
            
            # Check required derived variables
            required_derived_vars = ['wind_speed', 'vpd', 'ppfd_in', 'ext_rad']
            if not all(var in self.datasets for var in required_derived_vars):
                print("Calculating missing derived variables...")
                self.calculate_derived_variables()
        else:
            # Traditional mode - we need to load data for each month in the range
            if start_date is None or end_date is None:
                raise ValueError("start_date and end_date are required when use_existing_data=False")
                
            # Convert dates to datetime objects if they're strings
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            # Make consistent with your chosen standard
            if use_utc:
                # Make timezone-aware (UTC)
                if start_date.tzinfo is None:
                    start_date = start_date.replace(tzinfo=datetime.timezone.utc)
                if end_date.tzinfo is None:
                    end_date = end_date.replace(tzinfo=datetime.timezone.utc)
            else:
                # Make timezone-naive
                if start_date.tzinfo is not None:
                    start_date = start_date.replace(tzinfo=None)
                if end_date.tzinfo is not None:
                    end_date = end_date.replace(tzinfo=None)
            # Make consistent with your chosen standard
            if use_utc:
                # Make timezone-aware (UTC)
                if start_date.tzinfo is None:
                    start_date = start_date.replace(tzinfo=datetime.timezone.utc)
                if end_date.tzinfo is None:
                    end_date = end_date.replace(tzinfo=datetime.timezone.utc)
            else:
                # Make timezone-naive
                if start_date.tzinfo is not None:
                    start_date = start_date.replace(tzinfo=None)
                if end_date.tzinfo is not None:
                    end_date = end_date.replace(tzinfo=None)
            # Load data for each month in the range
            current_date = start_date
            while current_date <= end_date:
                year = current_date.year
                month = current_date.month
                
                print(f"\nProcessing data for {year}-{month:02d}...")
                
                # Load variable data for this month
                self.load_variable_data(config.REQUIRED_VARIABLES, year, month, None, region)
                
                # Calculate derived variables
                self.calculate_derived_variables()
                
                # Move to the next month
                if current_date.month == 12:
                    current_date = pd.Timestamp(current_date.year + 1, 1, 1)
                else:
                    current_date = pd.Timestamp(current_date.year, current_date.month + 1, 1)
        
        # Convert dates to datetime objects if they're strings
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Process all data to create prediction dataset
        all_data = []
        
        # Select only the variables needed for prediction
        prediction_vars = []
        for era5_name, model_name in config.VARIABLE_RENAME.items():
            if era5_name in self.datasets:
                prediction_vars.append(era5_name)
        
        if not prediction_vars:
            print("No valid data variables found")
            return None
        
        print(f"Processing variables: {prediction_vars}")
        
        # Create DataArrays for each point or grid cell
        if points:
            # Extract data for specific points
            for point in points:
                point_data = {}
                point_data['name'] = point.get('name', f"Point_{point['lat']}_{point['lon']}")
                point_data['latitude'] = point['lat']
                point_data['longitude'] = point['lon']
                
                # Get biome for this point using climate-based method
                if config.BIOME_FEATURES:
                    biome_type = self.get_biome_at_location(point['lon'], point['lat'])
                    biome_features = self.create_biome_one_hot(biome_type)
                    point_data.update(biome_features)
                
                # Extract time series for each variable
                for var_name in prediction_vars:
                    try:
                        ds = self.datasets[var_name]
                        
                        # Find the data variable in the dataset
                        data_var = None
                        for var in ds.data_vars:
                            data_var = var
                            break
                        
                        if data_var:
                            # Extract the nearest grid cell
                            ts = ds[data_var].sel(
                                latitude=point['lat'],
                                longitude=point['lon'],
                                method='nearest'
                            )
                            
                            # Get the time coordinate name
                            time_dim = None
                            for dim in ts.dims:
                                if dim.lower() in ['time', 'datetime']:
                                    time_dim = dim
                                    break
                            
                            if time_dim is None:
                                print(f"Warning: No time dimension found in {var_name}")
                                continue
                            
                            # Filter to the requested date range if specified
                            if start_date is not None and end_date is not None:
                                ts = ts.sel({time_dim: slice(start_date, end_date)})
                            
                            # Convert to pandas Series with proper time index
                            times = ts[time_dim].values
                            values = ts.values
                            series = pd.Series(values, index=times)
                            
                            # Rename to model variable name
                            model_name = config.VARIABLE_RENAME.get(var_name, var_name)
                            point_data[model_name] = series
                        else:
                            print(f"No data variable found in {var_name} dataset")
                    except Exception as e:
                        print(f"Error extracting {var_name} for point {point_data['name']}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                
                # Create DataFrame from the point data
                try:
                    # Find a common time index across all series
                    all_series = [s for s in point_data.values() if isinstance(s, pd.Series)]
                    if all_series:
                        # Create a DataFrame with all data
                        df = pd.DataFrame(point_data)
                        
                        # Set datetime index if available
                        time_cols = [col for col in df.columns if col.lower() in ['time', 'datetime']]
                        if time_cols:
                            df = df.set_index(time_cols[0])
                        
                        # Add time features if configured
                        if config.TIME_FEATURES:
                            df = self.add_time_features(df)
                        
                        # Add to the list
                        all_data.append(df)
                    else:
                        print(f"No valid time series data for point {point_data['name']}")
                except Exception as e:
                    print(f"Error creating DataFrame for point {point_data['name']}: {str(e)}")
                    import traceback
                    traceback.print_exc()
        else:
            # Extract data for entire region grid cells
            # First, build a common grid
            lat_var = None
            lon_var = None
            time_var = None
            
            # Use the first dataset to get coordinate variables
            first_ds = self.datasets[prediction_vars[0]]
            for var in first_ds.coords:
                if var.lower() in ['latitude', 'lat']:
                    lat_var = var
                elif var.lower() in ['longitude', 'lon']:
                    lon_var = var
                elif var.lower() in ['time', 'datetime']:
                    time_var = var
            
            if not all([lat_var, lon_var, time_var]):
                print("Could not identify coordinate variables")
                return None
            
            # Get the grid coordinates
            lats = first_ds[lat_var].values
            lons = first_ds[lon_var].values
            times = first_ds[time_var].values
            
            print(f"Grid dimensions: {len(lats)}x{len(lons)} with {len(times)} time steps")
            
            # For each grid cell, create a DataFrame
            for lat_idx, lat in enumerate(lats):
                for lon_idx, lon in enumerate(lons):
                    grid_data = {}
                    grid_data['name'] = f"Grid_{lat}_{lon}"
                    grid_data['latitude'] = lat
                    grid_data['longitude'] = lon
                    
                    # Get biome for this grid cell using climate-based method
                    if config.BIOME_FEATURES:
                        biome_type = self.get_biome_at_location(lon, lat)
                        biome_features = self.create_biome_one_hot(biome_type)
                        grid_data.update(biome_features)
                    
                    # Extract time series for each variable
                    for var_name in prediction_vars:
                        try:
                            ds = self.datasets[var_name]
                            
                            # Find the data variable in the dataset
                            data_var = None
                            for var in ds.data_vars:
                                data_var = var
                                break
                            
                            if data_var:
                                # Extract the grid cell
                                # Try setting Dask to use memory instead of disk if available
                                if HAVE_DASK:
                                    with dask.config.set({'temporary_directory': None, 'local_directory': None}):
                                        ts = ds[data_var].sel(
                                            latitude=lat,
                                            longitude=lon,
                                            method='nearest'
                                        )
                                        
                                        # Get the time coordinate name
                                        time_dim = None
                                        for dim in ts.dims:
                                            if dim.lower() in ['time', 'datetime']:
                                                time_dim = dim
                                                break
                                        
                                        if time_dim is None:
                                            print(f"Warning: No time dimension found in {var_name}")
                                            continue
                                        
                                        # Filter to the requested date range if specified
                                        if start_date is not None and end_date is not None:
                                            ts = ts.sel({time_dim: slice(start_date, end_date)})
                                        
                                        # Compute the result
                                        result = ts.compute()
                                        series = pd.Series(result.values, index=ts[time_dim].values)
                                else:
                                    # Without Dask, use direct selection
                                    ts = ds[data_var].sel(
                                        latitude=lat,
                                        longitude=lon,
                                        method='nearest'
                                    )
                                    
                                    # Get the time coordinate name
                                    time_dim = None
                                    for dim in ts.dims:
                                        if dim.lower() in ['time', 'datetime']:
                                            time_dim = dim
                                            break
                                    
                                    if time_dim is None:
                                        print(f"Warning: No time dimension found in {var_name}")
                                        continue
                                    
                                    # Filter to the requested date range if specified
                                    if start_date is not None and end_date is not None:
                                        ts = ts.sel({time_dim: slice(start_date, end_date)})
                                    
                                    # Convert to pandas Series
                                    series = pd.Series(ts.values, index=ts[time_dim].values)
                                
                                # Rename to model variable name
                                model_name = config.VARIABLE_RENAME.get(var_name, var_name)
                                grid_data[model_name] = series
                            else:
                                print(f"No data variable found in {var_name} dataset")
                        except Exception as e:
                            print(f"Error extracting {var_name} for grid cell ({lat}, {lon}): {str(e)}")
                            import traceback
                            traceback.print_exc()
                    
                    # Create DataFrame from the grid data
                    try:
                        # Find a common time index across all series
                        all_series = [s for s in grid_data.values() if isinstance(s, pd.Series)]
                        if all_series:
                            # Create a DataFrame with all data
                            df = pd.DataFrame(grid_data)
                            
                            # Set datetime index if available
                            time_cols = [col for col in df.columns if col.lower() in ['time', 'datetime']]
                            if time_cols:
                                df = df.set_index(time_cols[0])
                            
                            # Add time features if configured
                            if config.TIME_FEATURES:
                                df = self.add_time_features(df)
                            
                            # Add to the list
                            all_data.append(df)
                        else:
                            print(f"No valid time series data for grid cell ({lat}, {lon})")
                    except Exception as e:
                        print(f"Error creating DataFrame for grid cell ({lat}, {lon}): {str(e)}")
                        import traceback
                        traceback.print_exc()
        
        # Combine all DataFrames
        if all_data:
            print(f"Combining {len(all_data)} datasets...")
            # Combine by concatenating (for points) or by merging with multi-index (for grid)
            if points:
                # For points, keep each point separate
                result = all_data

            else:
                # For grid, combine with multi-index
                result = pd.concat(all_data)
                
            
            print(f"Created prediction dataset with {len(result)} rows")
            return result
        else:
            print("No data found for the specified time period")
            return None
    
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
        else:
            # Single DataFrame
            df['ta'] = df['ta'] - 273.15  # Convert to Celsius
            df['td'] = df['td'] - 273.15  # Convert to Celsius
            df.to_csv(output_path, index_label='timestamp')
            print(f"Saved dataset to {output_path}")
            return str(output_path)
    def process_single_day(self, variables, year, month, day, shapefile=None, output_dir=None):
        """
        Process a single day of ERA5-Land data and create a prediction dataset.
        
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
            
        Returns:
        --------
        str
            Path to saved prediction dataset file
        """
        print(f"\n===== Processing day {year}-{month:02d}-{day:02d} =====")
        
        try:
            # Load data for this day
            self.load_variable_data(variables, year, month, day, shapefile=shapefile)
            
            # Calculate derived variables
            self.calculate_derived_variables()
            
            # Create prediction dataset
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
    def process_month_parallel(self, variables, year, month, shapefile=None, num_workers=None, output_dir=None, precalculate_biomes=True):
        """
        Process an entire month of ERA5-Land data with parallel processing of individual days.
        
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
        precalculate_biomes : bool, optional
            Whether to precalculate biomes for the region before processing
                
        Returns:
        --------
        list
            Paths to all successfully created daily prediction dataset files
        """
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
        
        # Pre-calculate biomes if requested
        if precalculate_biomes:
            print("Pre-calculating biomes for the region before parallel processing...")
            # Define region based on shapefile or default configuration
            if shapefile is not None:
                try:
                    import geopandas as gpd
                    shapefile_gdf = gpd.read_file(shapefile)
                    bounds = shapefile_gdf.total_bounds
                    region = {
                        'lon_min': bounds[0],
                        'lat_min': bounds[1],
                        'lon_max': bounds[2],
                        'lat_max': bounds[3]
                    }
                except Exception as e:
                    print(f"Error reading shapefile for biome precalculation: {e}")
                    print("Using default region instead")
                    region = {
                        'lat_min': config.DEFAULT_LAT_MIN,
                        'lat_max': config.DEFAULT_LAT_MAX,
                        'lon_min': config.DEFAULT_LON_MIN,
                        'lon_max': config.DEFAULT_LON_MAX
                    }
            else:
                region = {
                    'lat_min': config.DEFAULT_LAT_MIN,
                    'lat_max': config.DEFAULT_LAT_MAX,
                    'lon_min': config.DEFAULT_LON_MIN,
                    'lon_max': config.DEFAULT_LON_MAX
                }
            
            # Pre-calculate biomes using parallel processing
            # Use fewer workers for biome calculation to avoid resource contention
  
            self.precalculate_biomes(region=region)
            
            # Save the biome cache to a file so it can be loaded by worker processes
            biome_cache_file = output_dir / "biome_cache.pkl"
            try:
                import pickle
                with open(biome_cache_file, 'wb') as f:
                    pickle.dump(self.biome_cache, f)
                print(f"Saved biome cache to {biome_cache_file}")
            except Exception as e:
                print(f"Error saving biome cache: {e}")
        
        # Prepare arguments for each day's processing
        process_args = []
        for day in range(1, days_in_month + 1):
            # Each item is (year, month, day, variables, shapefile, output_dir)
            process_args.append((year, month, day, variables, shapefile, output_dir))
        
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
def visualize_biome_map(climate_processor, region=None, resolution=100, output_file=None):
    """
    Create a biome map visualization based on climate data.
    
    Parameters:
    -----------
    climate_processor : ERA5LandGEEProcessor
        Processor instance with loaded climate data
    region : dict, optional
        Region to visualize (lat_min, lat_max, lon_min, lon_max)
    resolution : int, optional
        Number of points in each dimension
    output_file : str, optional
        File path to save the map image
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    
    # Make sure climate data is loaded
    if not hasattr(climate_processor, 'temp_climate_data') or not hasattr(climate_processor, 'precip_climate_data'):
        climate_processor.load_climate_data(bounds = {
                    'lon_min': -180.0,  # Eastern longitude boundary
                    'lat_min': -57,  # Southern latitude boundary
                    'lon_max': 180.0,  # Western longitude boundary 
                    'lat_max': 78.0   })
    
    # Set default region if not specified
    if region is None:
        region = {
            'lat_min': config.DEFAULT_LAT_MIN,
            'lat_max': config.DEFAULT_LAT_MAX,
            'lon_min': config.DEFAULT_LON_MIN,
            'lon_max': config.DEFAULT_LON_MAX
        }
    
    # Create a grid of points for the region
    lats = np.linspace(region['lat_min'], region['lat_max'], resolution)
    lons = np.linspace(region['lon_min'], region['lon_max'], resolution)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Create arrays to hold biome data
    biome_map = np.empty(lon_grid.shape, dtype=object)
    biome_map.fill("Unknown")
    
    # Define colors for each biome
    biome_colors = {
        'Tropical rain forest': '#004529',  # Dark green
        'Tropical forest savanna': '#238443',  # Medium green
        'Subtropical desert': '#ffe0b2',  # Light orange/tan
        'Temperate rain forest': '#7fcdbb',  # Teal
        'Temperate forest': '#41ab5d',  # Medium-light green
        'Woodland/Shrubland': '#a1d99b',  # Light green
        'Temperate grassland desert': '#fdae61',  # Light orange
        'Boreal forest': '#2166ac',  # Blue
        'Tundra': '#d1e5f0',  # Light blue/gray
        'Unknown': '#f7f7f7'  # White/light gray
    }
    
    # Numerical mapping for biomes
    unique_biomes = list(climate_processor.BIOME_CLIMATE_THRESHOLDS.keys()) + ['Unknown']
    biome_to_index = {biome: i for i, biome in enumerate(unique_biomes)}
    biome_index_map = np.zeros(lon_grid.shape, dtype=int)
    
    # For each point in the grid, determine the biome
    for i in range(resolution):
        for j in range(resolution):
            lat = lat_grid[i, j]
            lon = lon_grid[i, j]
            
            # Get the biome at this location
            biome = climate_processor.get_biome_at_location(lon, lat)
            biome_map[i, j] = biome
            biome_index_map[i, j] = biome_to_index.get(biome, len(unique_biomes) - 1)  # Default to 'Unknown'
    
    # Create a colormap from the biome colors
    colors = [biome_colors.get(biome, '#f7f7f7') for biome in unique_biomes]
    cmap = ListedColormap(colors)
    
    # Plot the biome map
    plt.figure(figsize=(12, 10))
    plt.pcolormesh(lon_grid, lat_grid, biome_index_map, cmap=cmap, shading='auto')
    
    # Add coastlines or borders if available
    try:
        from cartopy import crs as ccrs
        from cartopy.feature import COASTLINE, BORDERS
        
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(COASTLINE, linewidth=0.5)
        ax.add_feature(BORDERS, linewidth=0.3, linestyle=':')
    except ImportError:
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
    
    # Create a legend
    legend_patches = []
    for biome, color in biome_colors.items():
        if biome in climate_processor.BIOME_CLIMATE_THRESHOLDS:
            patch = mpatches.Patch(color=color, label=biome)
            legend_patches.append(patch)
    
    plt.legend(handles=legend_patches, loc='lower left', bbox_to_anchor=(1.01, 0),
               title='Whittaker Biomes', frameon=True)
    
    plt.title('Climate-based Biome Classification')
    plt.tight_layout()
    
    # Save the figure if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Biome map saved to {output_file}")
    
    plt.show()


def main():
    """
    Main function for ERA5-Land processing with parallel day processing.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Process ERA5-Land data from GEE for sap velocity prediction')
    parser.add_argument('--year', type=int, required=True, help='Year to process')
    parser.add_argument('--month', type=int, required=True, help='Month to process')
    parser.add_argument('--day', type=int, help='Day to process (for single-day processing)')
    parser.add_argument('--parallel', action='store_true', help='Process the entire month using parallel processing')
    parser.add_argument('--num-workers', type=int, help='Number of parallel worker processes')
    parser.add_argument('--output', type=str, help='Output directory path')
    parser.add_argument('--shapefile', type=str, default=None, help='Path to shapefile for region definition')
    parser.add_argument('--gee-scale', type=float, default=11132, help='Scale in meters for GEE export (default 27km)')
    parser.add_argument('--precalculate-biomes', action='store_true', 
                        help='Pre-calculate biomes for the region before processing')
    
    args = parser.parse_args()

    # Create ERA5-Land GEE processor
    processor = ERA5LandGEEProcessor()
    
    try:
        # Load climate data for biome determination
        processor.load_climate_data(bounds = {
                    'lon_min': -180.0,  # Eastern longitude boundary
                    'lat_min': -57,  # Southern latitude boundary
                    'lon_max': 180.0,  # Western longitude boundary 
                    'lat_max': 78.0   })
        
        # Pre-calculate biomes if requested and not using parallel processing
        # (parallel processing has its own precalculation)
        if args.precalculate_biomes and not args.parallel:
            if args.shapefile:
                try:
                    import geopandas as gpd
                    shapefile_gdf = gpd.read_file(args.shapefile)
                    bounds = shapefile_gdf.total_bounds
                    region = {
                        'lon_min': bounds[0],
                        'lat_min': bounds[1],
                        'lon_max': bounds[2],
                        'lat_max': bounds[3]
                    }
                except Exception as e:
                    print(f"Error reading shapefile for biome precalculation: {e}")
                    print("Using default region instead")
                    region = {
                        'lat_min': config.DEFAULT_LAT_MIN,
                        'lat_max': config.DEFAULT_LAT_MAX,
                        'lon_min': config.DEFAULT_LON_MIN,
                        'lon_max': config.DEFAULT_LON_MAX
                    }
            else:
                region = {
                    'lat_min': config.DEFAULT_LAT_MIN,
                    'lat_max': config.DEFAULT_LAT_MAX,
                    'lon_min': config.DEFAULT_LON_MIN,
                    'lon_max': config.DEFAULT_LON_MAX
                }
            
            processor.precalculate_biomes(region=region)
        
        if args.parallel:
            # Process the entire month using parallel processing
            output_dir = args.output or config.PREDICTION_DIR / f"{args.year}_{args.month:02d}_daily"
            processor.process_month_parallel(
                config.REQUIRED_VARIABLES, args.year, args.month, 
                shapefile=args.shapefile, 
                num_workers=args.num_workers,
                output_dir=output_dir,
                precalculate_biomes=args.precalculate_biomes
            )
            
            print(f"Parallel processing complete. All daily files saved to {output_dir}")
            
        elif args.day is not None:
            # Process a single day
            output_dir = args.output or config.PREDICTION_DIR / f"{args.year}_{args.month:02d}_daily"
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