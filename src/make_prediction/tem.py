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
# Add multiprocessing imports
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# Import configuration
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
             temp_climate_file=None, precip_climate_file=None, gee_initialize=True,
             num_processes=None):
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
      num_processes : int, optional
          Number of processes to use for multiprocessing (defaults to CPU count - 1)
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

      # Set up multiprocessing configuration
      if num_processes is None:
          # Default to CPU count - 1 (leave one CPU for system tasks)
          self.num_processes = max(1, mp.cpu_count() - 1)
      else:
          self.num_processes = num_processes
      
      print(f"Using {self.num_processes} processes for parallel operations")
      
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
              self.load_climate_data()
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
    
    def get_era5land_variable_from_gee(self, variable, year, month, day, region=None, scale=None):
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
      day : int, optional
          Day to access (if None, access all days in the month)
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
              region = ee.Geometry.Rectangle([
                  config.DEFAULT_LON_MIN, 
                  config.DEFAULT_LAT_MIN, 
                  config.DEFAULT_LON_MAX, 
                  config.DEFAULT_LAT_MAX
              ])
          elif isinstance(region, dict):
              # Convert dict to ee.Geometry
              region = ee.Geometry.Rectangle([
                  region.get('lon_min', config.DEFAULT_LON_MIN),
                  region.get('lat_min', config.DEFAULT_LAT_MIN),
                  region.get('lon_max', config.DEFAULT_LON_MAX),
                  region.get('lat_max', config.DEFAULT_LAT_MAX)
              ])
          
          # Set default scale if not specified (ERA5-Land is ~9km)
          if scale is None:
              scale = 9000  # 9km in meters
          
          # Set up date range
          if month is not None:
              if day is not None:
                  start_date = ee.Date.fromYMD(year, month, day)
                  # if reach the last day of the month, set to next month
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
              
              # When getting coordinates from the first image
              first_img = ee.Image(image_list.get(0))
              img_data = geemap.ee_to_numpy(
                  first_img.select(gee_var),
                  region=region, 
                  scale=scale
              )
              height = img_data.shape[0]
              width = img_data.shape[1]

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
              
              if not img_data.any():
                  print("Failed to get coordinates from first image.")
                  return None
              
              print(f"Grid size: {height}x{width}")
              
              # Define a function to process a single image that can be run in parallel
              def process_single_image(i):
                  try:
                      # Get the image
                      img = ee.Image(image_list.get(i))
                      
                      # Get the timestamp
                      time_start = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd HH:mm:ss').getInfo()
                      current_time = pd.to_datetime(time_start)
                      
                      # Create a unique temp file name based on index and timestamp
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
                          time.sleep(1)
                          
                          # Re-check if the file exists
                          if temp_file.exists():
                              print(f"File exists at {temp_file}, proceeding with processing")
                              success = True
                          else:
                              print(f"File does not exist at {temp_file} after export")
                              success = False
                      
                      # Try to read the file if it exists
                      if temp_file.exists():
                          try:
                              # Read the GeoTIFF with appropriate error handling
                              with rasterio.open(temp_file) as src:
                                  img_array = src.read(1)
                                  
                                  # Check shape consistency
                                  if img_array.shape[:2] == (height, width):
                                      return (i, current_time, img_array, None)
                                  else:
                                      print(f"Skipping image {i+1} from GeoTIFF - inconsistent shape: {img_array.shape} vs expected {(height, width)}")
                          except rasterio.errors.RasterioIOError as rio_err:
                              print(f"Rasterio failed to read file {temp_file}: {str(rio_err)}")
                      
                      # Fallback method: Try direct conversion to numpy
                      print(f"Using fallback method (ee_to_numpy) for image {i+1}")
                      img_array = geemap.ee_to_numpy(
                          img.select(gee_var),
                          region=region,
                          scale=scale
                      )
                      
                      # Make sure the shape is consistent
                      if img_array.shape[:2] == (height, width):
                          return (i, current_time, img_array, None)
                      else:
                          print(f"Skipping image {i+1} - inconsistent shape: {img_array.shape} vs expected {(height, width)}")
                          return (i, None, None, f"Inconsistent shape: {img_array.shape}")
                      
                  except Exception as e:
                      print(f"Error processing image {i+1}: {str(e)}")
                      return (i, None, None, str(e))
              
              # Process images in parallel using ThreadPoolExecutor
              all_arrays = []
              timestamps = []
              
              # Split processing into batches to avoid memory issues
              for batch_start in range(0, collection_size, batch_size):
                  batch_end = min(batch_start + batch_size, collection_size)
                  print(f"Processing batch from {batch_start+1} to {batch_end} of {collection_size}...")
                  
                  # For GEE, we use ThreadPoolExecutor instead of ProcessPoolExecutor
                  # because the EE API has global state that doesn't work well with process forking
                  with ThreadPoolExecutor(max_workers=self.num_processes) as executor:
                      # Submit all tasks
                      future_to_idx = {executor.submit(process_single_image, i): i 
                                      for i in range(batch_start, batch_end)}
                      
                      # Process results as they complete
                      for future in as_completed(future_to_idx):
                          idx = future_to_idx[future]
                          try:
                              result = future.result()
                              if result[1] is not None and result[2] is not None:
                                  # Successful processing: (i, timestamp, array, None)
                                  _, timestamp, array, _ = result
                                  all_arrays.append(array)
                                  timestamps.append(timestamp)
                              else:
                                  # Error occurred: (i, None, None, error_message)
                                  _, _, _, error = result
                                  print(f"Failed to process image {idx+1}: {error}")
                          except Exception as exc:
                              print(f"Image {idx+1} generated an exception: {exc}")
              
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
                      time_start = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd_HH-mm-ss').getInfo()
                      timestamps.append(pd.to_datetime(time_start.replace('_', ' ')))
                      
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
    
    def load_variable_data(self, variables, year, month, day, region=None):
        """
        Load and preprocess multiple ERA5-Land variables from GEE for a specific time period.
        
        Parameters:
        -----------
        variables : list
            List of variable names to load
        year : int
            Year to load data for
        month : int
            Month to load data for
        region : dict, optional
            Region boundaries (lat_min, lat_max, lon_min, lon_max)
            
        Returns:
        --------
        dict
            Dictionary of datasets for each variable
        """
        if region is None:
            region = {
                'lat_min': config.DEFAULT_LAT_MIN,
                'lat_max': config.DEFAULT_LAT_MAX, 
                'lon_min': config.DEFAULT_LON_MIN,
                'lon_max': config.DEFAULT_LON_MAX
            }
        
        # Clear existing datasets
        for var_name, ds in self.datasets.items():
            try:
                if hasattr(ds, 'close'):
                    ds.close()
            except:
                pass
        self.datasets = {}
        
        # Load each variable
        for var_name in variables:
            print(f"\nLoading {var_name} data for {year}-{month}...")
            ds = self.get_era5land_variable_from_gee(var_name, year, month, day, region)
            
            if ds is not None:
                """
                # Extract region of interest if needed
                ds = self.extract_region(
                    ds, 
                    lat_min=region['lat_min'],
                    lat_max=region['lat_max'],
                    lon_min=region['lon_min'],
                    lon_max=region['lon_max']
                )
                """
                self.datasets[var_name] = ds
                print(f"Successfully loaded {var_name}")
            else:
                print(f"Failed to load {var_name}")
        
        return self.datasets
    
    def calculate_derived_variables(self):
        """
        Calculate derived variables from the loaded ERA5-Land data.
        
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
            print("Calculating wind speed from u and v components...")
            try:
                u_ds = self.datasets['10m_u_component_of_wind']
                v_ds = self.datasets['10m_v_component_of_wind']
                
                # Get the main data variable from each dataset
                u_var = list(u_ds.data_vars)[0] if list(u_ds.data_vars) else None
                v_var = list(v_ds.data_vars)[0] if list(v_ds.data_vars) else None
                
                if u_var and v_var:
                    # Standard approach using named variables
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
            print("Calculating vapor pressure deficit (VPD)...")
            try:
                t_ds = self.datasets['2m_temperature']
                td_ds = self.datasets['2m_dewpoint_temperature']
                
                # Get the main data variable from each dataset
                t_var = list(t_ds.data_vars)[0] if list(t_ds.data_vars) else None
                td_var = list(td_ds.data_vars)[0] if list(td_ds.data_vars) else None
                
                if t_var and td_var:
                    # Convert from K to C
                    t_c = t_ds[t_var] - 273.15
                    td_c = td_ds[td_var] - 273.15
                    
                    # Calculate saturated vapor pressure (hPa) using Magnus-Tetens formula
                    es = 6.1078 * 10.0**(7.5 * t_c / (237.3 + t_c))
                    
                    # Calculate actual vapor pressure (hPa)
                    ea = 6.1078 * 10.0**(7.5 * td_c / (237.3 + td_c))
                    
                    # Calculate VPD (hPa)
                    vpd = es - ea
                    
                    # Create a new dataset with the VPD variable
                    vpd_ds = xr.Dataset(
                        data_vars={'vpd': vpd},
                        coords=t_ds.coords,
                        attrs={'units': 'hPa', 'long_name': 'Vapor Pressure Deficit'}
                    )
                    vpd_ds['vpd'].attrs['units'] = 'hPa'
                    vpd_ds['vpd'].attrs['long_name'] = 'Vapor Pressure Deficit'
                    
                    self.datasets['vpd'] = vpd_ds
                    print(f"VPD calculated successfully from variables: {t_var}, {td_var}")
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
                    
                    if is_accumulated and 'time' in sr_ds.dims:
                        # Need to compute the difference between consecutive time steps
                        # Calculate instantaneous radiation in W/m2
                        diff = sr_ds[sr_var].diff('time')
                        time_step_seconds = np.diff(sr_ds.coords['time'].values).astype('timedelta64[s]').astype(float)[0]
                        solar_rad_diff = diff / time_step_seconds

                        # Get the original time coordinates
                        original_times = sr_ds.coords['time'].values

                        # Create a new time array with first time unchanged and all others shifted by 1 hour
                        new_times = np.copy(original_times)
                        new_times[1:] = original_times[1:] + np.timedelta64(1, 'h')  # Shift all but first by 1 hour

                        # Create the solar_rad DataArray with the first value being the first diff value
                        # and the same new time coordinates
                        solar_rad = xr.DataArray(
                            data=np.zeros_like(sr_ds[sr_var].values),
                            dims=sr_ds[sr_var].dims,
                            coords={**sr_ds[sr_var].coords, 'time': new_times}
                        )

                        # Set values - first value from first diff, rest from remaining diffs
                        solar_rad.values[0] = solar_rad_diff.values[0]  # Use the first diff value for the first time step
                        solar_rad.values[1:] = solar_rad_diff.values    # Use the rest of the diff values for remaining time steps
                    else:
                        # Data is already in instantaneous form (W/m2)
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
                        # Enhanced extraterrestrial radiation calculation based on solaR R package
                        print("Calculating extraterrestrial radiation using enhanced algorithm...")
                        try:
                            # Get dimension information without loading all data
                            lat_values = sr_ds[lat_var].values
                            time_values = sr_ds.coords['time'].values
                            
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
                            time_dt = pd.DatetimeIndex(time_values)
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
                            jd = np.array([datetime_to_julian(t) for t in time_values])
                            
                            # Day angle for solar calculations
                            X = 2 * np.pi * (dn - 1) / 365
                            
                            # Solar calculations using the 'michalsky' method (default in solaR)
                            # Can be modified to use 'cooper', 'spencer', or 'strous' methods
                            method = 'michalsky'
                            
                            # Calculating declination based on method
                            if method == 'cooper':
                                # Cooper method
                                decl = d2r(23.45 * np.sin(2 * np.pi * (dn + 284) / 365))
                            elif method == 'spencer':
                                # Spencer method
                                decl = 0.006918 - 0.399912 * np.cos(X) + 0.070257 * np.sin(X) - 0.006758 * np.cos(2 * X) \
                                    + 0.000907 * np.sin(2 * X) - 0.002697 * np.cos(3 * X) + 0.001480 * np.sin(3 * X)
                            elif method == 'strous':
                                # Strous method
                                meanAnomaly = (357.5291 + 0.98560028 * jd) % 360
                                coefC = np.array([1.9148, 0.02, 0.0003])
                                
                                C = np.zeros_like(meanAnomaly)
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
                                eo = 1 + 0.033 * np.cos(2 * np.pi * dn / 365)
                            else:  # spencer, michalsky, strous all use the same formula
                                eo = (1.000110 + 0.034221 * np.cos(X) + 0.001280 * np.sin(X) +
                                    0.000719 * np.cos(2 * X) + 0.000077 * np.sin(2 * X))
                            
                            # Calculate Equation of Time (minutes)
                            M = 2 * np.pi / 365.24 * dn
                            EoT_min = 229.18 * (-0.0334 * np.sin(M) + 0.04184 * np.sin(2 * M + 3.5884))
                            EoT_hours = EoT_min / 60  # Convert to hours
                            
                            # Solar constant (W/m²) as used in the R package
                            Bo = 1367
                            
                            # Calculate sunset hour angle for each latitude/day combination
                            rad_values_2d = np.zeros((len(lat_values), len(time_values)))
                            
                            for i, lat in enumerate(lat_values):
                                lat_rad = d2r(lat)
                                # Sunset hour angle calculation
                                cosWs = -np.tan(lat_rad) * np.tan(decl)
                                ws = np.zeros_like(cosWs)
                                
                                # Handle polar day/night correctly
                                for j in range(len(cosWs)):
                                    if cosWs[j] < -1:  # Polar day
                                        ws[j] = np.pi
                                    elif cosWs[j] > 1:  # Polar night
                                        ws[j] = 0
                                    else:
                                        ws[j] = np.arccos(cosWs[j])
                                
                                for j, t in enumerate(time_values):
                                    # Calculate solar time including Equation of Time correction
                                    hour_of_day = pd.Timestamp(t).hour + pd.Timestamp(t).minute / 60
                                    solar_time = hour_of_day + EoT_hours[j]
                                    
                                    # Calculate hour angle (15° per hour from solar noon)
                                    hour_angle_rad = d2r(15 * (solar_time - 12))
                                    
                                    # Calculate cosine of solar zenith angle
                                    cos_zenith = (np.sin(lat_rad) * np.sin(decl[j]) + 
                                                np.cos(lat_rad) * np.cos(decl[j]) * np.cos(hour_angle_rad))
                                    
                                    # No radiation when sun is below horizon
                                    cos_zenith = max(0.0, cos_zenith)
                                    
                                    # Calculate extraterrestrial radiation adjusted by Earth-Sun distance
                                    rad_values_2d[i, j] = Bo * eo[j] * cos_zenith
                            
                            # Create a 2D DataArray for lat/time dimensions
                            ext_rad_2d = xr.DataArray(
                                rad_values_2d,
                                dims=[lat_var, 'time'],
                                coords={lat_var: lat_values, 'time': time_values}
                            )
                            
                            # Broadcast to full dimensionality using template
                            ones_template = xr.ones_like(sr_ds[sr_var])
                            ext_rad_full = ones_template * ext_rad_2d
                            
                            # Create the dataset with proper attributes
                            ext_rad_ds = xr.Dataset(
                                data_vars={'ext_rad': ext_rad_full},
                                coords=sr_ds.coords,
                                attrs={'units': 'W m-2', 'long_name': f'Extraterrestrial Radiation ({method} method)'}
                            )
                            ext_rad_ds['ext_rad'].attrs['units'] = 'W m-2'
                            ext_rad_ds['ext_rad'].attrs['long_name'] = f'Extraterrestrial Radiation ({method} method)'
                            
                            # Optional: Also calculate daily total extraterrestrial radiation
                            # This matches the Bo0d calculation in the R code
                            if 'day' in sr_ds.coords or 'dayofyear' in sr_ds.coords:
                                print("Also calculating daily total extraterrestrial radiation...")
                                day_coord = 'day' if 'day' in sr_ds.coords else 'dayofyear'
                                
                                # Daily total (Bo0d) as in the R code:
                                # Bo0d = -24/pi*Bo*eo*(ws*sin(lat)*sin(decl)+cos(lat)*cos(decl)*sin(ws))
                                daily_totals = {}
                                
                                for i, lat in enumerate(lat_values):
                                    lat_rad = d2r(lat)
                                    for j, d in enumerate(np.unique(dn)):
                                        day_idx = np.where(dn == d)[0][0]  # Get first occurrence
                                        day_decl = decl[day_idx]
                                        day_eo = eo[day_idx]
                                        
                                        # Calculate sunset hour angle for this lat/day
                                        cosWs = -np.tan(lat_rad) * np.tan(day_decl)
                                        if cosWs < -1:  # Polar day
                                            day_ws = np.pi
                                        elif cosWs > 1:  # Polar night
                                            day_ws = 0
                                        else:
                                            day_ws = np.arccos(cosWs)
                                        
                                        # Calculate daily total extraterrestrial radiation (MJ/m²/day)
                                        Bo0d = (-24/np.pi * Bo * day_eo * 
                                            (day_ws * np.sin(lat_rad) * np.sin(day_decl) + 
                                                np.cos(lat_rad) * np.cos(day_decl) * np.sin(day_ws)))
                                        
                                        # Convert from W/m² to MJ/m²/day
                                        Bo0d = Bo0d * 0.0864  # 24*60*60/1e6
                                        
                                        daily_totals[(lat, d)] = Bo0d
                        
                        except Exception as e:
                            print(f"Error in enhanced extraterrestrial radiation calculation: {str(e)}")
                            import traceback
                            traceback.print_exc()
                            
                            # Fall back to simpler method
                            print("Falling back to simpler extraterrestrial radiation calculation...")
                            # Original calculation code...
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
    def load_climate_data(self, temp_file=None, precip_file=None):
            """
            Load annual mean temperature and precipitation data from GeoTIFF files.
            
            Parameters:
            -----------
            temp_file : str or Path, optional
                Path to the annual mean temperature file (.tif)
            precip_file : str or Path, optional
                Path to the annual mean precipitation file (.tif)
                
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
            
            try:
                print(f"Loading annual mean temperature data from {temp_file}...")
                with rasterio.open(temp_file) as temp_src:
                    # Read temperature data
                    temp_data = temp_src.read(1)  # Read the first band
                    
                    # Get the geotransform information
                    temp_transform = temp_src.transform
                    temp_crs = temp_src.crs
                    
                    # Create a simple dataset-like structure
                    height, width = temp_data.shape
                    self.temp_climate_data = {
                        'data': temp_data,
                        'transform': temp_transform,
                        'crs': temp_crs,
                        'height': height,
                        'width': width,
                        'nodata': temp_src.nodata
                    }
                    
                    print(f"Temperature data loaded: shape={temp_data.shape}, crs={temp_crs}")
                
                print(f"Loading annual mean precipitation data from {precip_file}...")
                with rasterio.open(precip_file) as precip_src:
                    # Read precipitation data
                    precip_data = precip_src.read(1)  # Read the first band
                    
                    # Get the geotransform information
                    precip_transform = precip_src.transform
                    precip_crs = precip_src.crs
                    
                    # Create a simple dataset-like structure
                    height, width = precip_data.shape
                    self.precip_climate_data = {
                        'data': precip_data,
                        'transform': precip_transform,
                        'crs': precip_crs,
                        'height': height,
                        'width': width,
                        'nodata': precip_src.nodata
                    }
                    
                    print(f"Precipitation data loaded: shape={precip_data.shape}, crs={precip_crs}")
                
                return self.temp_climate_data, self.precip_climate_data
            
            except Exception as e:
                print(f"Error loading climate data: {str(e)}")
                import traceback
                traceback.print_exc()
                return None, None

    def determine_biome_from_climate(self, temp, precip):
        """
        Determine biome type based on temperature and precipitation values
        using Whittaker biome classification.
        
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
        
        # Check each biome's climate thresholds
        for biome, (temp_min, temp_max, precip_min, precip_max) in self.BIOME_CLIMATE_THRESHOLDS.items():
            if temp_min <= temp <= temp_max and precip_min <= precip <= precip_max:
                return biome
        
        # If no exact match, find the closest biome by climate distance
        min_distance = float('inf')
        closest_biome = default_biome
        
        for biome, (temp_min, temp_max, precip_min, precip_max) in self.BIOME_CLIMATE_THRESHOLDS.items():
            # Use the midpoint of each range
            biome_temp = (temp_min + temp_max) / 2
            biome_precip = (precip_min + precip_max) / 2
            
            # Calculate Euclidean distance in normalized climate space
            # Normalize temperature to 0-1 range over -15 to 30°C
            temp_range = 45  # -15 to 30°C
            temp_min = -15
            temp_norm = (temp - temp_min) / temp_range
            biome_temp_norm = (biome_temp - temp_min) / temp_range
            
            # Normalize precipitation to 0-1 range over 0 to 3000mm
            precip_max = 3000
            precip_norm = min(precip / precip_max, 1.0)  # Cap at 1.0 for very high precipitation
            biome_precip_norm = min(biome_precip / precip_max, 1.0)
            
            # Calculate distance in this normalized space
            distance = np.sqrt((temp_norm - biome_temp_norm)**2 + (precip_norm - biome_precip_norm)**2)
            
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
            if self.load_climate_data() == (None, None):
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
        
        Parameters:
        -----------
        lon, lat : float
            Coordinates of the location
            
        Returns:
        --------
        str
            Biome type at the location
        """
        # First try using climate data to determine biome
        try:
            # Get climate values at the location
            temperature, precipitation = self.get_climate_at_location(lon, lat)
            
            # Determine biome from climate
            biome_type = self.determine_biome_from_climate(temperature, precipitation)
            
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
                                return self.biome_data.iloc[idx]['biome_type']
                            
                            # If not in the nearest, check neighbors
                            distances, indices = self.biome_kdtree.query((lon, lat), k=5)
                            for i in indices:
                                if self.biome_data.iloc[i].geometry.contains(point):
                                    return self.biome_data.iloc[i]['biome_type']
                        
                        # If no polygon contains the point, return the nearest biome type
                        return biome_type
                    
                    # Fallback to direct spatial query (slower)
                    from shapely.geometry import Point
                    point = Point(lon, lat)
                    
                    # Find biomes that contain the point
                    contains_point = self.biome_data.contains(point)
                    if contains_point.any():
                        biome_type = self.biome_data[contains_point].iloc[0]['biome_type']
                        return biome_type
                    
                    # If point is not in any polygon, find the nearest
                    distances = self.biome_data.distance(point)
                    nearest_idx = distances.idxmin()
                    return self.biome_data.loc[nearest_idx, 'biome_type']
                
                except Exception as inner_e:
                    print(f"Error in shapefile fallback: {str(inner_e)}")
            
            # If all else fails, return a default biome type
            print(f"Using default biome for location ({lon}, {lat})")
            return "Temperate forest"
        

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
    def flatten_time_dimensions(self, preserve_original=False):
        """
        Flatten time and step dimensions in ERA5-Land datasets to create a single datetime dimension.
        
        This function processes all datasets in self.datasets to combine the time (date) and 
        step (time in a day) dimensions into a single datetime dimension, making the data easier
        to work with for time series analysis.
        
        Parameters:
        -----------
        preserve_original : bool, optional
            If True, preserves the original datasets in a separate attribute
            
        Returns:
        --------
        dict
            Dictionary containing the flattened datasets
        """
        if not self.datasets:
            print("No datasets to flatten")
            return {}
        
        # Save original datasets if requested
        if preserve_original:
            self.original_datasets = self.datasets.copy()
        
        # Process each dataset
        flattened_datasets = {}
        for var_name, ds in self.datasets.items():
            print(f"\nFlattening time dimensions for {var_name}...")
            
            # Check if dataset has both time and step dimensions
            has_time = 'time' in ds.dims
            has_step = 'step' in ds.dims
            
            if not (has_time and has_step):
                print(f"Dataset {var_name} does not have both time and step dimensions - skipping")
                flattened_datasets[var_name] = ds
                continue
            
            try:
                # Find the main data variable in the dataset
                data_vars = list(ds.data_vars)
                if not data_vars:
                    print(f"No data variables found in {var_name} dataset - using dataset directly")
                    # For GRIB format, data might be directly in dataset
                    da = ds.to_array().squeeze()
                else:
                    # Use the first data variable
                    main_var = data_vars[0]
                    da = ds[main_var]
                
                # Create valid_time coordinate if it doesn't exist
                if 'valid_time' not in da.coords:
                    print("Creating valid_time coordinate")
                    # Create valid_time by adding step to time
                    times = da.coords['time'].values[:, None]
                    steps = da.coords['step'].values[None, :]
                    valid_time = times + steps
                    
                    # Assign the new coordinate
                    da = da.assign_coords(valid_time=(('time', 'step'), valid_time))
                
                # Stack the time and step dimensions
                print("Stacking time and step dimensions")
                stacked_da = da.stack(datetime=('time', 'step'))
                
                # Fix for FutureWarning: first drop the variables then assign new coordinate
                stacked_da = stacked_da.drop_vars(['datetime', 'time', 'step'], errors='ignore')
                
                # Use the valid_time values as the datetime coordinate
                flat_time = da.valid_time.stack(datetime=('time', 'step'))
                stacked_da = stacked_da.assign_coords(datetime=flat_time.values)
                
                # Sort by the datetime coordinate for time series consistency
                stacked_da = stacked_da.sortby('datetime')
                
                # Create a new dataset with the same metadata
                if data_vars:
                    flat_ds = xr.Dataset(
                        data_vars={main_var: stacked_da},
                        attrs=ds.attrs
                    )
                else:
                    # If original dataset had no data variables, create with proper name
                    flat_ds = xr.Dataset(
                        data_vars={var_name: stacked_da},
                        attrs=ds.attrs
                    )
                
                # Add to the result dictionary
                flattened_datasets[var_name] = flat_ds
                print(f"Successfully flattened {var_name} dataset")
                
                # Calculate the size of the flattened variable in MB
                if data_vars:
                    size_mb = flat_ds[main_var].nbytes / 1024**2
                    flat_ds[main_var].attrs['size'] = size_mb
                    print(f"Flattened {var_name} size: {size_mb:.2f} MB")
                
            except Exception as e:
                print(f"Error flattening {var_name} dataset: {str(e)}")
                import traceback
                traceback.print_exc()
                # Keep the original dataset
                flattened_datasets[var_name] = ds
        
        # Update the class datasets
        self.datasets = flattened_datasets
        
        return flattened_datasets
    def save_datasets_to_file(self, year, month, output_dir=None):
        """
        Save the complete datasets to files with multiple fallback approaches
        and enhanced error handling.
        
        Parameters:
        -----------
        year : int
            Year to save data for
        month : int
            Month to save data for
        output_dir : str or Path, optional
            Directory to save data to (defaults to temp_dir)
                
        Returns:
        --------
        list
            List of saved file paths
        """
        import gc
        
        if not self.datasets:
            print("No datasets to save")
            return []
        
        def sanitize_filename(name):
            """Sanitize a string to be used as a filename."""
            # Replace invalid characters with underscores
            invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']
            for char in invalid_chars:
                name = name.replace(char, '_')
            return name
        

        # Create the base output directory if it doesn't exist
        output_base_dir = Path(output_dir).absolute() if output_dir else Path(self.temp_dir).absolute()
        output_base_dir.mkdir(parents=True, exist_ok=True)
        
        
        saved_files = []
        derived_variables = ['wind_speed', 'vpd', 'ppfd_in', 'ext_rad']
        
        for var_name, ds in self.datasets.items():
            if var_name not in derived_variables:
                print(f"Skipping {var_name} as it's not in the save list")
                continue
                    
            try:
                # Sanitize variable name for file path
                safe_var_name = sanitize_filename(var_name)
                
                # Create simplified output path
                output_path = output_base_dir / f"{safe_var_name}_{year}_{month:02d}"
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Define output file
                nc_file = output_path / f"{safe_var_name}.grib"
                
                print(f"Saving {var_name} data to {output_path}...")
                print(f"Dataset variables: {list(ds.data_vars)}")
                print(f"Dataset dimensions: {dict(ds.dims)}")
                print(f"Dataset chunks: {getattr(ds, 'chunks', None)}")
                
                # Force garbage collection before heavy operations
                gc.collect()
                
                # APPROACH 0: Try a simple test path first
                test_path = Path('/tmp') if os.name != 'nt' else Path(os.environ.get('TEMP', 'C:/temp'))
                test_path.mkdir(exist_ok=True)
                test_file = test_path / f"{safe_var_name}_test.grib"
                
                try:
                    print(f"Attempting test save to {test_file}...")
                    to_grib(ds, test_file)
                    print(f"Test save succeeded to {test_file}")
                    # Clean up test file
                    if os.path.exists(test_file):
                        os.remove(test_file)
                except Exception as e:
                    print(f"Test save failed: {str(e)}")
                    print("This suggests a problem with the dataset structure rather than the path")
                
                # APPROACH 1: Save as CSV first (most robust format)
                try:
                    print("Attempting to save as CSV first...")
                    csv_dir = output_path / "csv_data"
                    csv_dir.mkdir(exist_ok=True, parents=True)
                    
                    for var in ds.data_vars:
                        csv_file = csv_dir / f"{sanitize_filename(var)}.csv"
                        # Convert to dataframe and save as CSV
                        df = ds[var].to_dataframe()
                        df.to_csv(csv_file)
                        print(f"Saved data for {var} as CSV: {csv_file}")
                        saved_files.append(csv_file)
                    
                    # Create a manifest file to document the CSV export
                    with open(csv_dir / "README.txt", "w") as f:
                        f.write(f"Dataset: {var_name}\n")
                        f.write(f"Year: {year}, Month: {month}\n")
                        f.write(f"Variables: {', '.join(ds.data_vars)}\n")
                        f.write(f"Dimensions: {dict(ds.dims)}\n")
                        f.write("Export date: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
                    
                    print("CSV backup saved successfully. Now trying NetCDF formats...")
                except Exception as e:
                    print(f"CSV save failed: {str(e)}")
                
                # APPROACH 2: Try with explicit chunking
                try:
                    print("Attempting to save with explicit chunking...")
                    
                    # Calculate reasonable chunk sizes (e.g., 1 time step, full lat/lon)
                    chunk_dict = {'time': 1}
                    for dim in ds.dims:
                        if dim != 'time':
                            chunk_dict[dim] = ds.dims[dim]
                    
                    print(f"Using explicit chunk sizes: {chunk_dict}")
                    chunked_ds = ds.chunk(chunk_dict)
                    
                    # Try to save with explicit encoding
                    encoding = {}
                    for var in chunked_ds.data_vars:
                        # Set up encoding for each variable
                        encoding[var] = {
                            'zlib': True,
                            'complevel': 1,  # Lower compression for speed
                            'chunksizes': [chunk_dict.get(dim, 1) for dim in chunked_ds[var].dims],
                            '_FillValue': np.nan
                        }
                    
                    print("Saving with explicit encoding...")
                    chunked_ds.to_netcdf(
                        nc_file,
                        mode='w',
                        engine='netcdf4',
                        encoding=encoding,
                        compute=True
                    )
                    print(f"Successfully saved with explicit chunking to {nc_file}")
                    saved_files.append(nc_file)
                    continue
                except Exception as e:
                    print(f"Explicit chunking save failed: {str(e)}")
                
                # APPROACH 3: Use optimized chunking (saves complete data)
                try:
                    print("Attempting to save with default settings...")
                    
                    # Save with basic settings - full dataset
                    ds.to_netcdf(
                        nc_file,
                        mode='w',
                        engine='netcdf4',
                        compute=True  # Ensures the operation completes
                    )
                    print(f"Successfully saved complete dataset to {nc_file}")
                    saved_files.append(nc_file)
                    continue
                except Exception as e:
                    print(f"Default save failed: {str(e)}")
                
                # APPROACH 4: Try to save as zarr directory (more robust for large datasets)
                try:
                    print("Attempting to save as zarr directory...")
                    zarr_dir = output_path / f"{safe_var_name}.zarr"
                    
                    # Remove directory if it exists (zarr requires this)
                    if os.path.exists(zarr_dir):
                        import shutil
                        shutil.rmtree(zarr_dir)
                    
                    # Save as zarr
                    ds.to_zarr(zarr_dir)
                    print(f"Successfully saved as zarr directory: {zarr_dir}")
                    saved_files.append(zarr_dir)
                    continue
                except Exception as e:
                    print(f"Zarr save failed: {str(e)}")
                
                # APPROACH 5: Use scipy engine (simpler but reliable)
                try:
                    print("Attempting to save with scipy engine...")
                    
                    # Try saving with scipy engine
                    simple_nc = output_path / f"{safe_var_name}_scipy.nc"
                    ds.to_netcdf(simple_nc, engine='scipy')
                    print(f"Successfully saved with scipy engine to {simple_nc}")
                    saved_files.append(simple_nc)
                    continue
                except Exception as e:
                    print(f"Scipy engine save failed: {str(e)}")
                
                # APPROACH 6: Split by variable for complex datasets
                try:
                    print("Attempting to save by splitting variables...")
                    success = False
                    
                    for var in ds.data_vars:
                        # Extract single variable dataset
                        single_var_ds = ds[var].to_dataset()
                        var_file = output_path / f"{sanitize_filename(var)}.nc"
                        
                        # Try different engines
                        for engine in ['netcdf4', 'h5netcdf', 'scipy']:
                            try:
                                print(f"Trying to save {var} with {engine} engine...")
                                single_var_ds.to_netcdf(var_file, engine=engine)
                                print(f"Saved variable {var} to {var_file} with {engine} engine")
                                saved_files.append(var_file)
                                success = True
                                break
                            except Exception as e_inner:
                                print(f"  Failed with {engine} engine: {str(e_inner)}")
                    
                    if success:
                        continue
                except Exception as e:
                    print(f"Split variable save failed: {str(e)}")
                
                # APPROACH 7: Last resort - save in a different format
                try:
                    print("Attempting to save in HDF5 format...")
                    import h5py
                    
                    h5_file = output_path / f"{safe_var_name}.h5"
                    
                    with h5py.File(h5_file, 'w') as f:
                        # Create a group for this dataset
                        grp = f.create_group(safe_var_name)
                        
                        # Add metadata
                        grp.attrs['year'] = year
                        grp.attrs['month'] = month
                        
                        # Save coordinates
                        coords_grp = grp.create_group('coordinates')
                        for coord_name, coord_data in ds.coords.items():
                            coords_grp.create_dataset(coord_name, data=coord_data.values)
                        
                        # Save variables
                        vars_grp = grp.create_group('variables')
                        for var in ds.data_vars:
                            # Get the data array
                            data_array = ds[var]
                            
                            # Convert to numpy and handle missing values
                            data = data_array.values
                            
                            # Create the dataset
                            dset = vars_grp.create_dataset(var, data=data)
                            
                            # Add variable attributes
                            for attr_name, attr_val in data_array.attrs.items():
                                if attr_val is not None:
                                    try:
                                        dset.attrs[attr_name] = attr_val
                                    except:
                                        # Some attributes might not be convertible to HDF5
                                        pass
                    
                    print(f"Successfully saved as HDF5: {h5_file}")
                    saved_files.append(h5_file)
                    continue
                except Exception as e:
                    print(f"HDF5 save failed: {str(e)}")
                
                print("All attempts to save this dataset have failed")
                
            except Exception as e:
                print(f"Error processing {var_name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Final garbage collection
        gc.collect()
        
        return saved_files
    def create_prediction_dataset(self, start_date=None, end_date=None, points=None, region=None, use_existing_data=True):
      """
      Create a dataset ready for prediction by extracting data with multiprocessing support.
      
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
              self.load_climate_data()
      
      # Check if we're using existing data
      if use_existing_data:
          if not self.datasets:
              raise ValueError("use_existing_data=True but no data has been loaded. Call load_variable_data() first.")
          
          # Determine time range from existing data if not specified
          if start_date is None or end_date is None:
              # Find min and max dates across all datasets
              all_times = []
              for ds_name, ds in self.datasets.items():
                  if 'time' in ds.coords:
                      times = pd.to_datetime(ds.coords['time'].values)
                      all_times.extend(times)
                  elif 'datetime' in ds.coords:
                      times = pd.to_datetime(ds.coords['datetime'].values)
                      all_times.extend(times)
              
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
          # Extract data for specific points with multiprocessing
          def process_point(point):
              try:
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
                  
                  # Create DataFrame from the point data
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
                      
                      return df
                  else:
                      print(f"No valid time series data for point {point_data['name']}")
                      return None
              except Exception as e:
                  print(f"Error processing point {point.get('name', 'unknown')}: {str(e)}")
                  import traceback
                  traceback.print_exc()
                  return None
          
          # Process points in parallel
          with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
              results = list(executor.map(process_point, points))
          
          # Filter out None results
          all_data = [df for df in results if df is not None]
          
      else:
          # Extract data for entire region grid cells using multiprocessing
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
          
          # Create a list of grid cells to process
          grid_cells = []
          for lat_idx, lat in enumerate(lats):
              for lon_idx, lon in enumerate(lons):
                  grid_cells.append((lat, lon))
          
          # Function to process a single grid cell
          def process_grid_cell(cell):
              lat, lon = cell
              try:
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
                      ds = self.datasets[var_name]
                      
                      # Find the data variable in the dataset
                      data_var = None
                      for var in ds.data_vars:
                          data_var = var
                          break
                      
                      if data_var:
                          # Extract the grid cell
                          # Use direct selection without Dask for multiprocessing
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
                  
                  # Create DataFrame from the grid data
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
                      
                      return df
                  else:
                      print(f"No valid time series data for grid cell ({lat}, {lon})")
                      return None
              except Exception as e:
                  print(f"Error processing grid cell ({lat}, {lon}): {str(e)}")
                  return None
          
          # Process grid cells in parallel
          # Use a chunksize to avoid spawning too many processes at once
          chunksize = max(1, len(grid_cells) // (self.num_processes * 10))
          with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
              results = list(executor.map(process_grid_cell, grid_cells, chunksize=chunksize))
          
          # Filter out None results
          all_data = [df for df in results if df is not None]
      
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
    from concurrent.futures import ProcessPoolExecutor
    
    # Make sure climate data is loaded
    if not hasattr(climate_processor, 'temp_climate_data') or not hasattr(climate_processor, 'precip_climate_data'):
        climate_processor.load_climate_data()
    
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
    
    # Function to process a chunk of grid points
    def process_grid_chunk(chunk_indices):
        chunk_results = []
        for idx in chunk_indices:
            i, j = idx // resolution, idx % resolution
            lat = lat_grid[i, j]
            lon = lon_grid[i, j]
            
            # Get the biome at this location
            biome = climate_processor.get_biome_at_location(lon, lat)
            biome_index = biome_to_index.get(biome, len(unique_biomes) - 1)  # Default to 'Unknown'
            
            chunk_results.append((i, j, biome, biome_index))
        return chunk_results
    
    # Create a list of all grid indices
    all_indices = list(range(resolution * resolution))
    
    # Split into chunks for parallel processing
    num_processes = climate_processor.num_processes
    chunk_size = max(1, len(all_indices) // (num_processes * 4))  # Adjust chunk size based on total work
    
    print(f"Processing biome map using {num_processes} processes with chunk size {chunk_size}...")
    
    # Process chunks in parallel
    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit chunks for processing
        futures = []
        for i in range(0, len(all_indices), chunk_size):
            chunk = all_indices[i:i+chunk_size]
            future = executor.submit(process_grid_chunk, chunk)
            futures.append(future)
        
        # Gather results
        for future in futures:
            chunk_results = future.result()
            results.extend(chunk_results)
    
    # Process results
    for i, j, biome, biome_index in results:
        biome_map[i, j] = biome
        biome_index_map[i, j] = biome_index
    
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
    Main function to demonstrate ERA5-Land processing from GEE with climate-based biome determination.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Process ERA5-Land data from GEE for sap velocity prediction')
    parser.add_argument('--year', type=int, required=True, help='Year to process')
    parser.add_argument('--month', type=int, required=False, help='Month to process')
    parser.add_argument('--day', type=int, default=None, help='Day to process')
    parser.add_argument('--lat-min', type=float, default=None, help='Minimum latitude')
    parser.add_argument('--lat-max', type=float, default=None, help='Maximum latitude')
    parser.add_argument('--lon-min', type=float, default=None, help='Minimum longitude')
    parser.add_argument('--lon-max', type=float, default=None, help='Maximum longitude')
    parser.add_argument('--output', type=str, default=None, help='Output file path')
    parser.add_argument('--point-lat', type=float, default=None, help='Point latitude')
    parser.add_argument('--point-lon', type=float, default=None, help='Point longitude')
    parser.add_argument('--point-name', type=str, default=None, help='Point name')
    parser.add_argument('--temp-climate', type=str, default=None, help='Temperature climate GeoTIFF file path')
    parser.add_argument('--precip-climate', type=str, default=None, help='Precipitation climate GeoTIFF file path')
    parser.add_argument('--create-biome-map', action='store_true', help='Create a biome map visualization')
    parser.add_argument('--gee-scale', type=float, default=9000, help='Scale in meters for GEE export (default 9km)')
    parser.add_argument('--num-processes', type=int, default=None, 
                        help='Number of parallel processes to use (default: CPU count - 1)')
    
    args = parser.parse_args()

    # Create ERA5-Land GEE processor with climate data files
    processor = ERA5LandGEEProcessor(
        temp_climate_file=args.temp_climate,
        precip_climate_file=args.precip_climate,
        num_processes=args.num_processes
    )
    
    try:
        # Set up region
        region = {
            'lat_min': args.lat_min if args.lat_min is not None else config.DEFAULT_LAT_MIN,
            'lat_max': args.lat_max if args.lat_max is not None else config.DEFAULT_LAT_MAX,
            'lon_min': args.lon_min if args.lon_min is not None else config.DEFAULT_LON_MIN,
            'lon_max': args.lon_max if args.lon_max is not None else config.DEFAULT_LON_MAX
        }
        
        # Create biome map if requested
        if args.create_biome_map:
            biome_map_file = args.output or f"biome_map_{int(time.time())}.png"
            visualize_biome_map(processor, region, output_file=biome_map_file)
            return
        
        # Define points if specified
        points = None
        if args.point_lat is not None and args.point_lon is not None:
            points = [{
                'lat': args.point_lat,
                'lon': args.point_lon,
                'name': args.point_name or f"Point_{args.point_lat}_{args.point_lon}"
            }]
        
        # Load data from GEE
        processor.load_variable_data(config.REQUIRED_VARIABLES, args.year, args.month, args.day, region)
        
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