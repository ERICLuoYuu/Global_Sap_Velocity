"""
ERA5-Land data processing utility for sap velocity prediction.

Handles extraction, reading, and preprocessing of ERA5-Land data
from Google Earth Engine with maintained functionality for derived variables.

Refactored for improved performance and resource utilization using Dask and Xarray.
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
from datetime import datetime, timedelta, timezone # Added timezone
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import rasterio
from rasterio.warp import transform_bounds, transform
from rasterio.windows import from_bounds
import calendar
import gc # Keep for strategic use

# Google Earth Engine imports
import ee
import geemap
import requests
import rioxarray # Use rioxarray for geospatial operations

# Configuration (assuming a config.py file exists)
try:
    import config
except ImportError:
    # Define default config if config.py is missing
    class Config:
        ERA5LAND_TEMP_DIR = Path("./era5land_temp")
        PREDICTION_DIR = Path("./prediction_output")
        TEMP_CLIMATE_FILE = Path("./data/worldclim_temp.tif") # Example path
        PRECIP_CLIMATE_FILE = Path("./data/worldclim_precip.tif") # Example path
        DEFAULT_LAT_MIN = -90.0
        DEFAULT_LAT_MAX = 90.0
        DEFAULT_LON_MIN = -180.0
        DEFAULT_LON_MAX = 180.0
        DASK_CREATE_CLIENT = True # Default to True if Dask available
        DASK_MEMORY_LIMIT = '4GB' # Example limit
        REQUIRED_VARIABLES = [ # Variables needed for calculations
             '2m_temperature', '2m_dewpoint_temperature',
             'total_precipitation', 'surface_solar_radiation_downwards',
             '10m_u_component_of_wind', '10m_v_component_of_wind'
        ]
        VARIABLE_RENAME = { # Mapping from ERA5 names to model names
            '2m_temperature': 'ta',
            '2m_dewpoint_temperature': 'td',
            'total_precipitation': 'precip',
            'surface_solar_radiation_downwards': 'sw_down',
            'wind_speed': 'ws', # Derived
            'vpd': 'vpd',       # Derived
            'ppfd_in': 'ppfd_in', # Derived
            'ext_rad': 'ext_rad', # Derived
            'annual_mean_temperature': 'annual_mean_temp',
            'annual_precipitation': 'annual_precip',
            'latitude': 'latitude',
            'longitude': 'longitude',
            'name': 'name'
        }
        TIME_FEATURES = True # Whether to add cyclical time features
        # Add other necessary config variables here
    config = Config()

# Try optional libraries
try:
    import dask
    from dask.diagnostics import ProgressBar
    from dask.distributed import Client, LocalCluster
    HAVE_DASK = True
    print("Dask is available.")
    # Configure dask for better memory usage (can be adjusted)
    dask.config.set({
        'array.chunk-size': '128MiB', # Increased chunk size
        'distributed.worker.memory.target': 0.7, # Allow workers to use more memory
        'distributed.worker.memory.spill': 0.8,
        'distributed.worker.memory.pause': 0.9,
        'distributed.scheduler.worker-saturation': '1.1', # Allow slightly more tasks than cores
    })
except ImportError:
    HAVE_DASK = False
    print("Dask not available. Parallel processing and memory optimizations will be limited.")
    # Define dummy ProgressBar if Dask is not available
    class ProgressBar:
        def __enter__(self): return self
        def __exit__(self, type, value, traceback): pass
        def compute(self, *args, **kwargs):
            # Fallback to sequential computation if dask is missing
            if len(args) == 1: return args[0]
            return args

try:
    import netCDF4
    HAVE_NETCDF4 = True
except ImportError:
    HAVE_NETCDF4 = False
    print("NetCDF4 module not available. Saving to NetCDF disabled.")

# --- Helper Functions ---

def _memory_usage_mb(obj):
    """Estimate memory usage of an object in MB."""
    if isinstance(obj, pd.DataFrame):
        return obj.memory_usage(deep=True).sum() / 1e6
    elif isinstance(obj, xr.Dataset):
        return obj.nbytes / 1e6
    elif isinstance(obj, xr.DataArray):
        return obj.nbytes / 1e6
    elif isinstance(obj, np.ndarray):
        return obj.nbytes / 1e6
    # Add more types if needed
    return sys.getsizeof(obj) / 1e6

# --- Main Processor Class ---

class ERA5LandGEEProcessor:
    """
    Class to process ERA5-Land data from Google Earth Engine with optimized methods
    for performance and memory efficiency using Dask and Xarray.
    """
    # ERA5-Land variable names in Google Earth Engine
    GEE_VARIABLE_MAPPING = {
        '2m_temperature': 'temperature_2m',
        '2m_dewpoint_temperature': 'dewpoint_temperature_2m',
        'total_precipitation': 'total_precipitation_hourly', # Use hourly for consistency
        'surface_solar_radiation_downwards': 'surface_solar_radiation_downwards_hourly', # Use hourly
        '10m_u_component_of_wind': 'u_component_of_wind_10m',
        '10m_v_component_of_wind': 'v_component_of_wind_10m',
        # Add other variables if needed
    }
    # Units mapping
    GEE_UNITS = {
        'temperature_2m': 'K',
        'dewpoint_temperature_2m': 'K',
        'total_precipitation_hourly': 'm', # Accumulated meter since previous step
        'surface_solar_radiation_downwards_hourly': 'J m**-2', # Accumulated Joules / m2
        'u_component_of_wind_10m': 'm s**-1',
        'v_component_of_wind_10m': 'm s**-1',
    }
    # Conversion factors (from ERA5 units to desired units if needed)
    # Example: Precipitation from m/hr to mm/hr
    UNIT_CONVERSION = {
         'total_precipitation_hourly': 1000, # m to mm
         'surface_solar_radiation_downwards_hourly': 1 / 3600 # J m-2 hr-1 to W m-2 (avg over hour)
    }
    # Desired output units after conversion (for metadata)
    OUTPUT_UNITS = {
        'temperature_2m': 'K',
        'dewpoint_temperature_2m': 'K',
        'total_precipitation_hourly': 'mm', # mm per hour
        'surface_solar_radiation_downwards_hourly': 'W m**-2', # W/m2
        'u_component_of_wind_10m': 'm s**-1',
        'v_component_of_wind_10m': 'm s**-1',
    }

    def __init__(self, temp_dir=None, memory_limit=None, num_workers=None,
                 temp_climate_file=None, precip_climate_file=None, gee_initialize=True,
                 dask_client=None):
        """
        Initialize the ERA5-Land GEE data processor.

        Parameters:
        -----------
        temp_dir : str or Path, optional
            Directory for temporary storage (defaults to config)
        memory_limit : str, optional
            Memory limit per Dask worker (e.g., '4GB'). Overrides config.
        num_workers : int, optional
            Number of Dask workers. Overrides default based on CPU count.
        temp_climate_file : str or Path, optional
            Path to the annual mean temperature GeoTIFF file
        precip_climate_file : str or Path, optional
            Path to the annual mean precipitation GeoTIFF file
        gee_initialize : bool, optional
            Whether to initialize the Google Earth Engine API
        dask_client : dask.distributed.Client, optional
            An existing Dask client to use. If None, a new one may be created.
        """
        self.temp_dir = Path(temp_dir) if temp_dir else config.ERA5LAND_TEMP_DIR
        self.temp_dir.mkdir(exist_ok=True, parents=True)

        # Climate data attributes
        self.temp_climate_file = Path(temp_climate_file) if temp_climate_file else getattr(config, 'TEMP_CLIMATE_FILE', None)
        self.precip_climate_file = Path(precip_climate_file) if precip_climate_file else getattr(config, 'PRECIP_CLIMATE_FILE', None)
        self._temp_climate_data = None # Store raw rasterio-like dict
        self._precip_climate_data = None
        self._temp_kdtree = None # Cached KDTree
        self._precip_kdtree = None
        self._climate_points = None # Cached climate grid points for KDTree

        # Datasets store (will hold xarray Datasets)
        self.datasets = {}

        # Initialize GEE
        self.ee_initialized = False
        if gee_initialize:
            self.initialize_gee()

        # Dask Client Setup
        self.client = dask_client
        self._cluster = None # Track if we created the cluster
        if self.client:
             print(f"Using provided Dask client: {self.client}")
        elif HAVE_DASK and config.DASK_CREATE_CLIENT:
            try:
                mem_limit = memory_limit or config.DASK_MEMORY_LIMIT
                n_workers = num_workers or max(1, os.cpu_count() - 1)
                print(f"Creating Dask LocalCluster with {n_workers} workers and memory limit: {mem_limit} per worker")
                # Use LocalCluster for more control
                self._cluster = LocalCluster(
                    n_workers=n_workers,
                    memory_limit=mem_limit,
                    # Consider adding threads_per_worker=1 if tasks are GIL-bound
                )
                self.client = Client(self._cluster) # Connect client to the cluster
                print(f"Dask client created: {self.client}")
                print(f"Dask dashboard link: {self.client.dashboard_link}")
            except Exception as e:
                print(f"Could not create Dask client: {str(e)}. Proceeding without Dask parallelism.")
                if self._cluster: # Clean up cluster if client creation failed
                    try: self._cluster.close()
                    except: pass
                self.client = None
                self._cluster = None
        else:
             self.client = None
             self._cluster = None
             if HAVE_DASK:
                 print("Dask client creation disabled by config.")
             else:
                 print("Dask not available, cannot create client.")

        # Load climate data eagerly if files exist
        # Determine bounds based on a sample ERA5 dataset if possible, otherwise global
        # This requires GEE to be initialized
        initial_bounds = None
        if self.ee_initialized:
             try:
                  # Get bounds from a sample GEE image (more relevant than global)
                  sample_coll = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').limit(1)
                  sample_info = sample_coll.first().geometry().bounds().getInfo()
                  coords = sample_info['coordinates'][0]
                  lons = [c[0] for c in coords]
                  lats = [c[1] for c in coords]
                  initial_bounds = {
                       'lon_min': min(lons), 'lat_min': min(lats),
                       'lon_max': max(lons), 'lat_max': max(lats)
                  }
                  print(f"Using initial bounds from sample GEE data: {initial_bounds}")
             except Exception as e:
                  print(f"Could not get initial bounds from GEE ({e}), using default global.")
                  initial_bounds = None # Fallback

        if self.temp_climate_file and self.precip_climate_file:
            if self.temp_climate_file.exists() and self.precip_climate_file.exists():
                print("Loading and caching climate data during initialization...")
                self.load_climate_data(bounds=initial_bounds) # Load potentially subsetted climate data
            else:
                 print("Climate files specified but not found.")

    def __del__(self):
        """Clean up resources on object deletion."""
        self.close()

    def close(self):
        """Close all open datasets and clients."""
        print("Closing ERA5LandGEEProcessor...")
        # Close datasets
        for var_name in list(self.datasets.keys()):
             try:
                  if hasattr(self.datasets[var_name], 'close'):
                       self.datasets[var_name].close()
                  del self.datasets[var_name]
             except Exception as e:
                  print(f"  Warning: Could not close/delete dataset for {var_name}: {e}")
        self.datasets = {}

        # Close Dask client and cluster if owned by this instance
        if self.client and self._cluster: # Check if we created the cluster
             try:
                  self.client.close()
                  self._cluster.close()
                  print("Closed owned Dask client and cluster.")
             except Exception as e:
                  print(f"  Warning: Error closing Dask client/cluster: {e}")
        elif self.client:
             print("External Dask client provided, not closing it.")

        self.client = None
        self._cluster = None
        self._temp_climate_data = None
        self._precip_climate_data = None
        self._temp_kdtree = None
        self._precip_kdtree = None
        self._climate_points = None
        gc.collect()
        print("Processor closed.")

    def initialize_gee(self):
        """Initialize the Google Earth Engine API."""
        if self.ee_initialized:
            print("Google Earth Engine already initialized.")
            return
        try:
            # Attempt initialization without authentication first
            try:
                 # Specify project if needed (replace 'your-gee-project' or get from config)
                 gee_project = getattr(config, 'GEE_PROJECT', None)
                 if gee_project:
                      ee.Initialize(project=gee_project, opt_url='https://earthengine-highvolume.googleapis.com')
                 else:
                      ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
                 self.ee_initialized = True
                 print("Google Earth Engine initialized successfully (high-volume endpoint).")
            except Exception as init_err:
                 print(f"GEE Initialization failed ({init_err}). Trying authentication...")
                 # Fallback to authentication if initialization fails
                 ee.Authenticate()
                 gee_project = getattr(config, 'GEE_PROJECT', None)
                 if gee_project:
                      ee.Initialize(project=gee_project, opt_url='https://earthengine-highvolume.googleapis.com')
                 else:
                      ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
                 self.ee_initialized = True
                 print("Google Earth Engine authenticated and initialized successfully (high-volume endpoint).")

        except Exception as e:
            print(f"Failed to initialize Google Earth Engine: {str(e)}")
            print("You may need to run 'earthengine authenticate' in your terminal.")
            self.ee_initialized = False

    def _get_ee_region(self, shapefile=None, region_dict=None):
        """Helper to get ee.Geometry from shapefile or dictionary."""
        if shapefile:
            try:
                print(f"Reading shapefile: {shapefile}")
                gdf = gpd.read_file(shapefile)
                # Ensure WGS84 projection for GEE
                if gdf.crs and gdf.crs != "EPSG:4326":
                    print(f"Reprojecting shapefile from {gdf.crs} to EPSG:4326")
                    gdf = gdf.to_crs("EPSG:4326")

                # Use unary_union if multiple features
                if len(gdf) > 1:
                    print(f"Shapefile contains {len(gdf)} features. Using the union.")
                    geometry = gdf.unary_union
                else:
                    geometry = gdf.geometry.iloc[0]

                # Convert to GeoJSON and then to ee.Geometry
                geo_json = geometry.__geo_interface__
                ee_region = ee.Geometry(geo_json)
                print(f"Successfully created ee.Geometry from shapefile.")
                # Get bounds for potential use elsewhere
                bounds = geometry.bounds # (minx, miny, maxx, maxy)
                ee_bounds_dict = {'lon_min': bounds[0], 'lat_min': bounds[1],
                                  'lon_max': bounds[2], 'lat_max': bounds[3]}
                return ee_region, ee_bounds_dict
            except Exception as e:
                print(f"Error reading shapefile {shapefile}: {e}. Falling back to default region.")
                # Fallback to default region
                ee_region = ee.Geometry.Rectangle(
                    [config.DEFAULT_LON_MIN, config.DEFAULT_LAT_MIN,
                     config.DEFAULT_LON_MAX, config.DEFAULT_LAT_MAX],
                    proj='EPSG:4326', geodesic=False
                )
                ee_bounds_dict = {'lon_min': config.DEFAULT_LON_MIN, 'lat_min': config.DEFAULT_LAT_MIN,
                                  'lon_max': config.DEFAULT_LON_MAX, 'lat_max': config.DEFAULT_LAT_MAX}
                return ee_region, ee_bounds_dict

        elif region_dict:
             try:
                  ee_region = ee.Geometry.Rectangle(
                       [region_dict['lon_min'], region_dict['lat_min'],
                        region_dict['lon_max'], region_dict['lat_max']],
                       proj='EPSG:4326', geodesic=False # Use planar for rectangles
                  )
                  print(f"Using region dictionary: {region_dict}")
                  return ee_region, region_dict
             except Exception as e:
                  print(f"Error creating ee.Geometry from region_dict: {e}. Falling back.")

        # Default fallback: Global or configured default
        print("Using default region.")
        ee_region = ee.Geometry.Rectangle(
            [config.DEFAULT_LON_MIN, config.DEFAULT_LAT_MIN,
             config.DEFAULT_LON_MAX, config.DEFAULT_LAT_MAX],
            proj='EPSG:4326', geodesic=False
        )
        ee_bounds_dict = {'lon_min': config.DEFAULT_LON_MIN, 'lat_min': config.DEFAULT_LAT_MIN,
                          'lon_max': config.DEFAULT_LON_MAX, 'lat_max': config.DEFAULT_LAT_MAX}
        return ee_region, ee_bounds_dict

    def get_era5land_variable_from_gee(self, variable, start_date, end_date,
                                       shapefile=None, region_dict=None, scale=11132,
                                       target_chunks={'time': 24, 'latitude': 100, 'longitude': 100}):
        """
        Access ERA5-Land data from Google Earth Engine for a specific time period and region.
        Uses geemap.ee_to_xarray for efficient data retrieval.

        Parameters:
        -----------
        variable : str
            Standard variable name (e.g., '2m_temperature')
        start_date : str or datetime
            Start date/time for data retrieval
        end_date : str or datetime
            End date/time for data retrieval (exclusive)
        shapefile : str, optional
            Path to shapefile for region definition
        region_dict : dict, optional
             Dictionary with {'lat_min', 'lat_max', 'lon_min', 'lon_max'}
        scale : float, optional
            Spatial resolution in meters for the GEE export (default ~9km)
        target_chunks : dict, optional
            Desired Dask chunking for the output xarray Dataset.

        Returns:
        --------
        xarray.Dataset or None
            Dataset containing the requested variable data, or None if failed.
        """
        if not self.ee_initialized:
            print("Error: Google Earth Engine is not initialized.")
            return None

        try:
            gee_var = self.GEE_VARIABLE_MAPPING.get(variable)
            if not gee_var:
                print(f"Warning: Variable '{variable}' not found in GEE_VARIABLE_MAPPING. Trying original name.")
                gee_var = variable # Try the original name

            # Ensure dates are timezone-aware (UTC is standard for GEE)
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            if start_dt.tzinfo is None:
                start_dt = start_dt.tz_localize('UTC')
            else:
                start_dt = start_dt.tz_convert('UTC')
            if end_dt.tzinfo is None:
                end_dt = end_dt.tz_localize('UTC')
            else:
                end_dt = end_dt.tz_convert('UTC')

            start_str = start_dt.strftime('%Y-%m-%dT%H:%M:%S')
            end_str = end_dt.strftime('%Y-%m-%dT%H:%M:%S')

            print(f"Accessing GEE: Variable={gee_var}, Time={start_str} to {end_str}")

            # Get region geometry
            ee_region, bounds_dict = self._get_ee_region(shapefile, region_dict)
            print(f"GEE Region Bounds: {bounds_dict}")

            # Access ERA5-Land collection
            era5land = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY') \
                         .filterDate(start_str, end_str) \
                         .select(gee_var)

            # Check collection size
            collection_size = era5land.size().getInfo()
            print(f"Found {collection_size} images in GEE for {gee_var}.")
            if collection_size == 0:
                print("No images found for the specified time period and variable.")
                return None

            # Use geemap.ee_to_xarray for efficient conversion
            print(f"Converting GEE ImageCollection to xarray Dataset (Scale: {scale}m)...")
            start_conv = time.time()

            # Attempt direct conversion
            try:
                ds = geemap.ee_to_xarray(
                    image=era5land,
                    region=ee_region,
                    scale=scale,
                    # crs='EPSG:4326' # Usually default
                )
            except Exception as e:
                 print(f"Direct ee_to_xarray failed: {e}. Trying export to temporary GeoTIFF...")
                 # Fallback: Export to temporary GeoTIFF and read with rioxarray
                 temp_tif = self.temp_dir / f"temp_{variable}_{start_str}_{end_str}.tif".replace(":", "-")
                 try:
                      # Export the *first* image of the collection as a multi-band GeoTIFF
                      # This is less ideal than ee_to_xarray but a necessary fallback
                      # Note: This exports *all bands (time steps)* into one file
                      print(f"Exporting to {temp_tif}...")
                      geemap.ee_export_image_collection(
                           era5land,
                           out_dir=str(self.temp_dir),
                           filenames=temp_tif.stem, # Will append band names
                           scale=scale,
                           region=ee_region,
                           file_per_band=True, # Export each time step as a separate file
                           # timeout=300 # Increase timeout if needed
                      )

                      # Find the exported files (they might have band names appended)
                      exported_files = sorted(list(self.temp_dir.glob(f"{temp_tif.stem}*.tif")))
                      if not exported_files:
                           raise IOError("GEE export seemed to succeed, but no TIF files found.")

                      print(f"Reading {len(exported_files)} exported GeoTIFF files...")
                      # Open multiple files as a single xarray dataset
                      ds_list = [rioxarray.open_rasterio(f, chunks=target_chunks) for f in exported_files]

                      # Extract timestamps from filenames or metadata if possible
                      # Assuming filenames might contain time info like '..._20200101T000000.tif'
                      timestamps = []
                      for f in exported_files:
                           try:
                                # Attempt to parse time from filename (adjust format as needed)
                                time_part = f.stem.split('_')[-1] # Assuming time is last part
                                ts = pd.to_datetime(time_part, format='%Y%m%dT%H%M%S', utc=True)
                                timestamps.append(ts)
                           except:
                                # Fallback: Use file modification time or sequence number? Less reliable.
                                print(f"Warning: Could not parse timestamp from filename {f.name}. Timestamps might be inaccurate.")
                                timestamps.append(pd.Timestamp.now(tz='UTC')) # Placeholder

                      # Combine datasets along a new time dimension
                      ds = xr.concat(ds_list, dim=pd.Index(timestamps, name='time'))
                      ds = ds.rename({'band': gee_var}) # Rename band dimension if needed
                      # Ensure spatial coords are named correctly
                      ds = ds.rename({'y': 'latitude', 'x': 'longitude'})
                      print("Successfully loaded data via temporary GeoTIFF files.")

                 except Exception as export_err:
                      print(f"Fallback GeoTIFF export/read also failed: {export_err}")
                      return None
                 finally:
                      # Clean up temporary files
                      temp_files = list(self.temp_dir.glob(f"{temp_tif.stem}*.tif"))
                      for f in temp_files:
                           try:
                                f.unlink()
                           except OSError:
                                pass

            end_conv = time.time()
            print(f"GEE data conversion took {end_conv - start_conv:.2f} seconds.")
            print(f"Initial Dataset size: {_memory_usage_mb(ds):.2f} MB")

            # --- Post-processing ---
            # Rename variable to the GEE name used
            if list(ds.data_vars)[0] != gee_var:
                 ds = ds.rename({list(ds.data_vars)[0]: gee_var})

            # Ensure coordinates are standard ('latitude', 'longitude', 'time')
            coord_mapping = {'y': 'latitude', 'x': 'longitude'}
            ds = ds.rename({k: v for k, v in coord_mapping.items() if k in ds.coords})

            # Ensure time coordinate is datetime
            if 'time' in ds.coords and ds['time'].dtype != 'datetime64[ns]':
                try:
                    ds['time'] = pd.to_datetime(ds['time'].values)
                except Exception as time_err:
                    print(f"Warning: Could not convert time coordinate to datetime: {time_err}")

            # Apply chunking if not already chunked or chunks are different
            if HAVE_DASK and target_chunks:
                current_chunks = ds[gee_var].chunks
                # Check if chunking needs to be applied/updated
                needs_rechunk = False
                if not current_chunks: # Not chunked
                    needs_rechunk = True
                else:
                    # Compare target chunks (handle None for full dimension chunk)
                    for dim, size in target_chunks.items():
                         if dim in ds.dims:
                              # Handle cases where chunks might be None or tuples
                              current_dim_chunks_tuple = current_chunks[ds.dims.index(dim)]
                              if current_dim_chunks_tuple:
                                   current_dim_chunk = current_dim_chunks_tuple[0]
                              else: # If chunks tuple is empty/None for a dim, it's not chunked along it
                                   current_dim_chunk = ds.dims[dim]

                              target_dim_chunk = size if size is not None else ds.dims[dim]
                              if current_dim_chunk != target_dim_chunk:
                                   needs_rechunk = True
                                   break
                         else:
                              print(f"Warning: Target chunk dimension '{dim}' not found in dataset.")

                if needs_rechunk:
                    print(f"Rechunking dataset to {target_chunks}...")
                    # Filter target_chunks to only include dimensions present in the dataset
                    valid_target_chunks = {dim: size for dim, size in target_chunks.items() if dim in ds.dims}
                    ds = ds.chunk(valid_target_chunks)
                    print(f"Dataset rechunked. New chunking: {ds[gee_var].chunksizes}")


            # Add units metadata
            ds[gee_var].attrs['units'] = self.GEE_UNITS.get(gee_var, 'unknown')
            ds[gee_var].attrs['long_name'] = gee_var.replace('_', ' ').title()
            ds.attrs['source'] = 'Google Earth Engine (ECMWF/ERA5_LAND/HOURLY)'
            ds.attrs['retrieval_timestamp'] = datetime.now(timezone.utc).isoformat()

            print(f"Successfully retrieved and processed {gee_var}.")
            return ds

        except ee.EEException as e:
            print(f"Google Earth Engine Error for variable {variable}: {e}")
            # Consider adding retries here for transient GEE errors
            return None
        except Exception as e:
            print(f"Unexpected error getting variable {variable} from GEE: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_variable_data(self, variables, start_date, end_date, shapefile=None, region_dict=None,
                           data_dir=None, gee_scale=11132,
                           target_chunks={'time': 24, 'latitude': 100, 'longitude': 100}):
        """
        Load and preprocess multiple ERA5-Land variables for a given period and region.
        Prioritizes loading from local files if available, otherwise fetches from GEE.

        Parameters:
        -----------
        variables : list
            List of standard variable names to load (e.g., '2m_temperature').
        start_date, end_date : str or datetime
            Start and end dates for the data period.
        shapefile : str, optional
            Path to shapefile for clipping the data.
        region_dict : dict, optional
             Dictionary with {'lat_min', 'lat_max', 'lon_min', 'lon_max'}
        data_dir : str or Path, optional
            Path to directory containing local data files (NetCDF or GeoTIFF).
        gee_scale : float, optional
            Spatial resolution in meters for GEE export (if used).
        target_chunks : dict, optional
            Desired Dask chunking for the output xarray Datasets.

        Returns:
        --------
        dict
            Dictionary where keys are variable names and values are the loaded xarray Datasets.
        """
        print(f"\n--- Loading Variables: {start_date} to {end_date} ---")
        # Clear existing datasets before loading new ones for a different period
        self._clear_datasets()

        # Determine region bounds (used for both GEE and local file loading if possible)
        _, bounds_dict = self._get_ee_region(shapefile, region_dict)

        loaded_datasets = {}
        for var_name in variables:
            print(f"\nProcessing variable: {var_name}")
            ds = None
            used_local = False

            # 1. Attempt to load from local files first
            if data_dir:
                data_dir = Path(data_dir)
                if data_dir.is_dir():
                    # Look for NetCDF first, then GeoTIFF
                    # Construct a pattern that might match monthly/yearly files
                    # This might need refinement based on actual file naming conventions
                    year_start = pd.to_datetime(start_date).year
                    month_start = pd.to_datetime(start_date).month
                    patterns = [
                        f"*{var_name}*{year_start}*{month_start:02d}*.nc", # Monthly NetCDF
                        f"*{var_name}*{year_start}*.nc", # Yearly NetCDF
                        f"*{var_name}*.nc", # Any NetCDF
                        f"*{var_name}*{year_start}*{month_start:02d}*.tif", # Monthly TIF
                        f"*{var_name}*{year_start}*.tif", # Yearly TIF
                        f"*{var_name}*.tif" # Any TIF
                    ]
                    local_file = None
                    for pattern in patterns:
                        found_files = sorted(list(data_dir.glob(pattern)))
                        if found_files:
                            local_file = found_files[0] # Take the first match
                            print(f"Found potential local file: {local_file}")
                            break

                    if local_file:
                        try:
                            print(f"Attempting to load from local file: {local_file}")
                            # Use rioxarray for robust opening of GeoTIFF/NetCDF
                            # Apply chunking during load
                            ds = rioxarray.open_rasterio(local_file, chunks=target_chunks)

                            # Ensure dataset structure is consistent (e.g., variable name)
                            # Find the likely variable name in the file
                            possible_var_names = [v for v in ds.data_vars if var_name in v]
                            actual_var_name = list(ds.data_vars)[0] # Default to first
                            if possible_var_names:
                                actual_var_name = possible_var_names[0]
                            elif len(ds.data_vars) == 1:
                                actual_var_name = list(ds.data_vars)[0]
                            else:
                                print(f"Warning: Could not definitively identify variable for {var_name} in {local_file}. Using '{actual_var_name}'.")

                            # Rename if necessary and select only that variable
                            if actual_var_name != var_name:
                                ds = ds.rename({actual_var_name: var_name})
                            ds = ds[[var_name]] # Select only the target variable


                            # Ensure standard coordinate names
                            coord_mapping = {'y': 'latitude', 'x': 'longitude'}
                            ds = ds.rename({k: v for k, v in coord_mapping.items() if k in ds.coords})

                            # Select time slice
                            if 'time' in ds.coords:
                                 ds = ds.sel(time=slice(start_date, end_date))
                            else:
                                 print(f"Warning: No time coordinate found in local file {local_file}")

                            # Clip to bounds if possible (rioxarray handles CRS)
                            if bounds_dict and hasattr(ds, 'rio'):
                                 print("Clipping local file data to bounds...")
                                 try:
                                      # Create geometry for clipping
                                      from shapely.geometry import box
                                      clip_geom = gpd.GeoSeries([box(bounds_dict['lon_min'], bounds_dict['lat_min'],
                                                                     bounds_dict['lon_max'], bounds_dict['lat_max'])],
                                                                crs="EPSG:4326") # Assume bounds are WGS84
                                      # Ensure dataset has CRS, default to WGS84 if missing
                                      if ds.rio.crs is None:
                                           ds.rio.write_crs("EPSG:4326", inplace=True)
                                      ds = ds.rio.clip(clip_geom.geometry, drop=False, all_touched=True)
                                 except Exception as clip_err:
                                      print(f"Warning: Failed to clip local data: {clip_err}")

                            print(f"Successfully loaded {var_name} from local file.")
                            used_local = True
                        except Exception as e:
                            print(f"Error loading local file {local_file}: {e}")
                            ds = None # Ensure ds is None if local loading fails

            # 2. If local loading failed or wasn't attempted, fetch from GEE
            if ds is None:
                print(f"Fetching {var_name} from Google Earth Engine...")
                ds = self.get_era5land_variable_from_gee(
                    var_name, start_date, end_date,
                    shapefile=shapefile, region_dict=region_dict, # Pass region info
                    scale=gee_scale, target_chunks=target_chunks
                )

            # 3. Store the loaded dataset
            if ds is not None:
                 # --- Unit Conversion (Example for Precip/Solar Rad) ---
                 # Use the *original* var_name key to check for conversion, but operate on the actual data var name
                 actual_var_name = list(ds.data_vars)[0] # Should be var_name if renaming worked
                 gee_equiv_var = self.GEE_VARIABLE_MAPPING.get(var_name, var_name) # Find GEE equivalent for unit lookup

                 if gee_equiv_var in self.UNIT_CONVERSION:
                      current_units = ds[actual_var_name].attrs.get('units', '')
                      target_units = self.OUTPUT_UNITS.get(gee_equiv_var, current_units) # Get target unit based on GEE var
                      factor = self.UNIT_CONVERSION[gee_equiv_var]

                      print(f"Applying unit conversion for {actual_var_name} (from {current_units} to {target_units})...")

                      # Check if data is accumulated (e.g., J/m2 or m)
                      is_accumulated = 'J m**-2' in current_units or current_units == 'm'

                      if is_accumulated and 'time' in ds.dims and len(ds['time']) > 1:
                           # Calculate time difference in hours for rate conversion
                           print(f"  Detected accumulated units ({current_units}). Converting to rate.")
                           with dask.config.set(scheduler='synchronous'): # Avoid dask errors during diff
                                time_diff_seconds = ds['time'].diff(dim='time').astype('timedelta64[s]')
                                # Prepend the first time step's difference (assume it's the same as the second)
                                first_diff = time_diff_seconds[0] if len(time_diff_seconds) > 0 else np.timedelta64(3600, 's') # Default to 1 hour
                                time_diff_full = xr.concat([first_diff, time_diff_seconds], dim='time')
                                time_diff_hours = time_diff_full / np.timedelta64(3600, 's') # Diff in hours

                           # Divide accumulated value by time difference (in hours) and apply factor
                           # Ensure time_diff_hours aligns with the data array's time coordinate
                           time_diff_hours = time_diff_hours.reindex_like(ds['time'], method='nearest')
                           ds[actual_var_name] = (ds[actual_var_name] / time_diff_hours) * factor
                           print(f"  Applied rate conversion and factor {factor}.")
                      else:
                           # Apply simple factor if not rate conversion or single time step
                           ds[actual_var_name] = ds[actual_var_name] * factor
                           print(f"  Applied simple factor {factor}.")

                      ds[actual_var_name].attrs['units'] = target_units
                      print(f"  Converted {actual_var_name} units to {target_units}")


                 loaded_datasets[var_name] = ds
                 print(f"Stored dataset for {var_name}. Size: {_memory_usage_mb(ds):.2f} MB, Chunks: {ds[var_name].chunksizes if HAVE_DASK and ds[var_name].chunks else 'Not Chunked'}")
            else:
                 print(f"Failed to load data for {var_name}.")

        self.datasets = loaded_datasets
        print("\n--- Finished loading variables ---")
        return self.datasets

    def calculate_derived_variables(self):
        """
        Calculate derived variables (Wind Speed, VPD, PAR) using vectorized
        Xarray and Dask operations.
        """
        print("\n--- Calculating Derived Variables ---")
        if not self.datasets:
            print("No datasets loaded. Cannot calculate derived variables.")
            return self.datasets

        # --- Calculate Wind Speed ---
        u_var_key = '10m_u_component_of_wind'
        v_var_key = '10m_v_component_of_wind'
        if u_var_key in self.datasets and v_var_key in self.datasets:
            print("Calculating Wind Speed...")
            try:
                u_ds = self.datasets[u_var_key]
                v_ds = self.datasets[v_var_key]

                # Ensure datasets are aligned (important if loaded separately)
                # Use outer join to keep all times/coords, fill missing with NaN
                u_ds_aligned, v_ds_aligned = xr.align(u_ds, v_ds, join='outer')

                # Get actual variable names within the datasets
                u_var_name = list(u_ds_aligned.data_vars)[0]
                v_var_name = list(v_ds_aligned.data_vars)[0]

                # Vectorized calculation (Dask handles parallelism if chunked)
                wind_speed_da = np.sqrt(u_ds_aligned[u_var_name]**2 + v_ds_aligned[v_var_name]**2)
                wind_speed_da.name = 'wind_speed'
                wind_speed_da.attrs['units'] = 'm s**-1'
                wind_speed_da.attrs['long_name'] = '10m Wind Speed'

                self.datasets['wind_speed'] = wind_speed_da.to_dataset()
                print(f"Wind Speed calculated. Size: {_memory_usage_mb(self.datasets['wind_speed']):.2f} MB")
            except Exception as e:
                print(f"Error calculating wind speed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Skipping Wind Speed calculation (missing u or v component).")

        # --- Calculate Vapor Pressure Deficit (VPD) ---
        t_var_key = '2m_temperature'
        td_var_key = '2m_dewpoint_temperature'
        if t_var_key in self.datasets and td_var_key in self.datasets:
            print("Calculating Vapor Pressure Deficit (VPD)...")
            try:
                t_ds = self.datasets[t_var_key]
                td_ds = self.datasets[td_var_key]

                # Align datasets
                t_ds_aligned, td_ds_aligned = xr.align(t_ds, td_ds, join='outer')

                t_var_name = list(t_ds_aligned.data_vars)[0]
                td_var_name = list(td_ds_aligned.data_vars)[0]

                # Ensure units are Kelvin
                if t_ds_aligned[t_var_name].attrs.get('units', '').upper() != 'K':
                     raise ValueError(f"Temperature units are not K: {t_ds_aligned[t_var_name].attrs.get('units')}")
                if td_ds_aligned[td_var_name].attrs.get('units', '').upper() != 'K':
                     raise ValueError(f"Dewpoint Temperature units are not K: {td_ds_aligned[td_var_name].attrs.get('units')}")

                # Convert K to C
                t_c = t_ds_aligned[t_var_name] - 273.15
                td_c = td_ds_aligned[td_var_name] - 273.15

                # Ensure td <= t
                td_c = xr.where(td_c > t_c, t_c, td_c)

                # Calculate saturation vapor pressure (es) and actual vapor pressure (ea)
                # Using Magnus formula (units: hPa)
                es = 6.1078 * xr.ufuncs.exp((17.269 * t_c) / (237.3 + t_c))
                ea = 6.1078 * xr.ufuncs.exp((17.269 * td_c) / (237.3 + td_c))

                # Calculate VPD, ensuring it's non-negative
                vpd_da = xr.where(es > ea, es - ea, 0.0)
                vpd_da.name = 'vpd'
                vpd_da.attrs['units'] = 'hPa'
                vpd_da.attrs['long_name'] = 'Vapor Pressure Deficit'

                self.datasets['vpd'] = vpd_da.to_dataset()
                print(f"VPD calculated. Size: {_memory_usage_mb(self.datasets['vpd']):.2f} MB")
            except Exception as e:
                print(f"Error calculating VPD: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Skipping VPD calculation (missing temperature or dewpoint).")

        # --- Calculate PAR and Extraterrestrial Radiation ---
        sr_var_key = 'surface_solar_radiation_downwards'
        if sr_var_key in self.datasets:
            print("Calculating PAR and Extraterrestrial Radiation...")
            try:
                sr_ds = self.datasets[sr_var_key]
                sr_var_name = list(sr_ds.data_vars)[0]

                # Ensure units are W m-2 (may require conversion if loaded from GEE hourly)
                if sr_ds[sr_var_name].attrs.get('units', '') != 'W m**-2':
                     # This assumes the conversion from J m-2 hr-1 happened during load
                     if 'J m**-2' in sr_ds[sr_var_name].attrs.get('units', ''):
                          print(f"Warning: Solar radiation units ({sr_ds[sr_var_name].attrs.get('units', '')}) suggest accumulation but target is W m-2. Conversion should have happened during load.")
                          # If conversion didn't happen, attempt it now (less ideal)
                          # factor = 1 / 3600
                          # solar_rad_wm2 = sr_ds[sr_var_name] * factor
                     else:
                          raise ValueError(f"Unexpected solar radiation units for PAR calculation: {sr_ds[sr_var_name].attrs.get('units')}. Expected W m**-2.")
                else:
                     solar_rad_wm2 = sr_ds[sr_var_name]


                # Calculate PPFD (Photosynthetic Photon Flux Density)
                # Conversion factor: ~4.6 μmol/s/W for PAR range (approximate)
                par_fraction = 0.45 # Fraction of solar radiation that is PAR
                conversion_factor_par = 4.6
                ppfd_da = solar_rad_wm2 * par_fraction * conversion_factor_par
                ppfd_da.name = 'ppfd_in'
                ppfd_da.attrs['units'] = 'umol m**-2 s**-1'
                ppfd_da.attrs['long_name'] = 'Incoming Photosynthetic Photon Flux Density'
                self.datasets['ppfd_in'] = ppfd_da.to_dataset()
                print(f"PPFD calculated. Size: {_memory_usage_mb(self.datasets['ppfd_in']):.2f} MB")

                # Calculate Extraterrestrial Radiation (more complex)
                # Requires latitude and time coordinates
                if 'latitude' in sr_ds.coords and 'time' in sr_ds.coords:
                    print("Calculating Extraterrestrial Radiation...")
                    lat = sr_ds['latitude']
                    time_coord = sr_ds['time']

                    # Ensure time is datetime64
                    if not np.issubdtype(time_coord.dtype, np.datetime64):
                         time_coord_dt = pd.to_datetime(time_coord.values)
                    else:
                         time_coord_dt = pd.DatetimeIndex(time_coord.values)

                    # Calculate day of year (doy) and fractional year
                    doy = xr.DataArray(time_coord_dt.dayofyear, coords={'time': time_coord}, name='doy')
                    year_frac = 2 * np.pi / 365.0 * (doy - 1 + (time_coord_dt.hour / 24.0))

                    # Solar declination (radians) - Using Spencer (1971) approx.
                    decl = 0.006918 - 0.399912 * np.cos(year_frac) + 0.070257 * np.sin(year_frac) \
                           - 0.006758 * np.cos(2 * year_frac) + 0.000907 * np.sin(2 * year_frac) \
                           - 0.002697 * np.cos(3 * year_frac) + 0.00148 * np.sin(3 * year_frac)

                    # Equation of time (minutes) - Using Spencer (1971) approx.
                    eq_time = 229.18 * (0.000075 + 0.001868 * np.cos(year_frac) - 0.032077 * np.sin(year_frac) \
                                        - 0.014615 * np.cos(2 * year_frac) - 0.040849 * np.sin(2 * year_frac))

                    # Time correction (minutes) - Simplified, assumes standard time meridian
                    # For more accuracy, longitude difference from standard meridian is needed
                    time_offset = eq_time # minutes
                    local_solar_time_hr = time_coord_dt.hour + time_coord_dt.minute / 60.0 + time_offset / 60.0

                    # Hour angle (radians) - relative to solar noon
                    hour_angle = (local_solar_time_hr - 12.0) * (np.pi / 12.0) # 15 degrees = pi/12 radians per hour

                    # Latitude in radians
                    lat_rad = np.deg2rad(lat)

                    # Cosine of solar zenith angle
                    cos_zenith = (np.sin(lat_rad) * np.sin(decl) +
                                  np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle))

                    # Clip to horizon (max(0, cos_zenith))
                    cos_zenith = xr.where(cos_zenith < 0, 0, cos_zenith)

                    # Earth-Sun distance factor (eccentricity correction) - Using Spencer (1971) approx.
                    earth_sun_dist_factor = 1.000110 + 0.034221 * np.cos(year_frac) + 0.001280 * np.sin(year_frac) \
                                     + 0.000719 * np.cos(2 * year_frac) + 0.000077 * np.sin(2 * year_frac)

                    # Solar constant (W/m^2)
                    solar_constant = 1367.0

                    # Extraterrestrial radiation on horizontal plane
                    ext_rad_da = solar_constant * earth_sun_dist_factor * cos_zenith
                    ext_rad_da.name = 'ext_rad'
                    ext_rad_da.attrs['units'] = 'W m**-2'
                    ext_rad_da.attrs['long_name'] = 'Extraterrestrial Solar Radiation'

                    self.datasets['ext_rad'] = ext_rad_da.to_dataset()
                    print(f"Extraterrestrial Radiation calculated. Size: {_memory_usage_mb(self.datasets['ext_rad']):.2f} MB")
                else:
                    print("Skipping Extraterrestrial Radiation (missing latitude or time coords).")

            except Exception as e:
                print(f"Error calculating PAR/ExtRad variables: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Skipping PAR/ExtRad calculation (missing surface solar radiation).")

        print("--- Finished calculating derived variables ---")
        return self.datasets

    def load_climate_data(self, bounds=None):
        """
        Load annual mean temperature and precipitation data from GeoTIFF files,
        optionally subset to bounds, and build KD-Trees for efficient lookup.

        Parameters:
        -----------
        bounds : dict, optional
            Bounding box dictionary {'lon_min', 'lat_min', 'lon_max', 'lat_max'}
        """
        if self._temp_kdtree and self._precip_kdtree:
             print("Climate data and KD-Trees already loaded.")
             return

        if not self.temp_climate_file or not self.precip_climate_file or \
           not self.temp_climate_file.exists() or not self.precip_climate_file.exists():
            print("Climate files not found or not specified. Cannot load climate data.")
            return

        print("Loading climate data and building KD-Trees...")
        start_time = time.time()

        def _load_and_build_kdtree(filepath, data_attr_name, kdtree_attr_name, bounds):
            """Internal helper to load raster, filter points, build KDTree."""
            try:
                # chunks=False loads into memory, which is needed for KDTree build
                with rioxarray.open_rasterio(filepath, chunks=False) as src:
                    # Clip if bounds are provided
                    clipped_src = src # Start with original
                    if bounds:
                         try:
                              from shapely.geometry import box
                              clip_geom = gpd.GeoSeries([box(bounds['lon_min'], bounds['lat_min'],
                                                             bounds['lon_max'], bounds['lat_max'])],
                                                        crs="EPSG:4326") # Assume bounds WGS84
                              # Ensure dataset has CRS, default to WGS84 if missing
                              if clipped_src.rio.crs is None:
                                   print(f"Warning: CRS missing for {filepath.name}. Assuming EPSG:4326.")
                                   clipped_src.rio.write_crs("EPSG:4326", inplace=True)
                              # Clip requires matching CRS
                              if clipped_src.rio.crs != clip_geom.crs:
                                   print(f"Reprojecting clip geometry to {clipped_src.rio.crs}...")
                                   clip_geom = clip_geom.to_crs(clipped_src.rio.crs)

                              clipped_src = clipped_src.rio.clip(clip_geom.geometry, drop=True, all_touched=True) # Drop points outside bounds
                              print(f"Clipped {filepath.name} to bounds. New shape: {clipped_src.shape}")
                              if clipped_src.size == 0:
                                   print(f"Warning: Clipping resulted in empty data for {filepath.name}")
                                   return None # No valid points
                         except Exception as clip_err:
                              print(f"Warning: Failed to clip {filepath.name}: {clip_err}. Using full extent.")
                              clipped_src = src # Revert to original if clip fails

                    # Ensure standard coordinate names
                    coord_mapping = {'y': 'latitude', 'x': 'longitude'}
                    renamed_src = clipped_src.rename({k: v for k, v in coord_mapping.items() if k in clipped_src.coords})

                    # Extract data and coordinates
                    data = renamed_src.values.squeeze() # Remove single band dim
                    lats = renamed_src['latitude'].values
                    lons = renamed_src['longitude'].values
                    nodata = renamed_src.rio.nodata

                    # Create meshgrid and flatten
                    lon_grid, lat_grid = np.meshgrid(lons, lats)
                    points = np.vstack((lon_grid.flatten(), lat_grid.flatten())).T
                    values = data.flatten()

                    # Filter out nodata values for KDTree
                    # Handle potential NaN as nodata as well
                    if nodata is not None:
                         valid_mask = (values != nodata) & (~np.isnan(values))
                    else:
                         valid_mask = ~np.isnan(values)

                    if np.sum(valid_mask) == 0:
                         print(f"Warning: No valid data points found in {filepath.name} after nodata/nan filtering.")
                         setattr(self, data_attr_name, None)
                         setattr(self, kdtree_attr_name, None)
                         return None # Return None for points if no valid data

                    valid_points = points[valid_mask]
                    valid_values = values[valid_mask]

                    # Store the raw data info (optional, maybe not needed if KDTree is primary access)
                    # setattr(self, data_attr_name, {'data': data, 'lats': lats, 'lons': lons, 'nodata': nodata})

                    # Build and store KDTree
                    print(f"Building KD-Tree for {filepath.name} with {len(valid_points)} points...")
                    kdtree = cKDTree(valid_points)
                    setattr(self, kdtree_attr_name, (kdtree, valid_values)) # Store tree and corresponding values
                    print(f"KD-Tree built for {filepath.name}.")
                    return valid_points # Return points for potential reuse

            except Exception as e:
                print(f"Error loading climate file {filepath}: {e}")
                import traceback
                traceback.print_exc()
                setattr(self, data_attr_name, None)
                setattr(self, kdtree_attr_name, None)
                return None

        # Load Temp Data and Build Tree
        temp_points = _load_and_build_kdtree(self.temp_climate_file, '_temp_climate_data', '_temp_kdtree', bounds)

        # Load Precip Data and Build Tree
        # Reuse points if grids match, otherwise build separately
        precip_points = _load_and_build_kdtree(self.precip_climate_file, '_precip_climate_data', '_precip_kdtree', bounds)

        # Store points if they are consistent (optional, mainly for debugging)
        # if temp_points is not None and precip_points is not None and np.array_equal(temp_points, precip_points):
        #      self._climate_points = temp_points

        end_time = time.time()
        print(f"Climate data loading and KD-Tree building took {end_time - start_time:.2f} seconds.")

    def get_climate_at_location_kdtree(self, lons, lats, max_distance=0.1):
        """
        Get climate data for given locations using pre-built KD-Trees.

        Parameters:
        -----------
        lons, lats : array-like
            Coordinates of the location(s)
        max_distance : float, optional
            Maximum distance (in degrees) to accept a valid data point.

        Returns:
        --------
        tuple
            (temperatures, precipitations) numpy arrays for the locations.
        """
        if self._temp_kdtree is None or self._precip_kdtree is None:
            print("Warning: Climate KD-Trees not built. Loading climate data first.")
            # Attempt to load climate data now (using default bounds)
            self.load_climate_data()
            if self._temp_kdtree is None or self._precip_kdtree is None:
                 print("Error: Failed to load climate data. Returning default values.")
                 # Return default values matching input shape
                 default_temp = np.full_like(np.asarray(lons), 15.0, dtype=float)
                 default_precip = np.full_like(np.asarray(lats), 800.0, dtype=float)
                 return default_temp, default_precip

        lons = np.asarray(lons)
        lats = np.asarray(lats)
        query_points = np.vstack((lons.flatten(), lats.flatten())).T

        # Default values
        temperatures = np.full(query_points.shape[0], 15.0, dtype=float) # Default Temp: 15 C
        precipitations = np.full(query_points.shape[0], 800.0, dtype=float) # Default Precip: 800 mm/yr

        # Query Temperature Tree
        temp_tree, temp_values = self._temp_kdtree
        distances, indices = temp_tree.query(query_points, k=1, distance_upper_bound=max_distance)
        # Indices can be out of bounds if distance_upper_bound is exceeded
        valid_indices_mask = indices < len(temp_values)
        valid_mask = np.isfinite(distances) & valid_indices_mask
        temperatures[valid_mask] = temp_values[indices[valid_mask]]
        print(f"Temperature lookup: Found {np.sum(valid_mask)} valid points within {max_distance} degrees.")

        # Query Precipitation Tree
        precip_tree, precip_values = self._precip_kdtree
        distances, indices = precip_tree.query(query_points, k=1, distance_upper_bound=max_distance)
        # Indices can be out of bounds if distance_upper_bound is exceeded
        valid_indices_mask = indices < len(precip_values)
        valid_mask = np.isfinite(distances) & valid_indices_mask
        precipitations[valid_mask] = precip_values[indices[valid_mask]]
        print(f"Precipitation lookup: Found {np.sum(valid_mask)} valid points within {max_distance} degrees.")


        # Reshape results to match input shape
        output_shape = lons.shape
        return temperatures.reshape(output_shape), precipitations.reshape(output_shape)

    def add_time_features(self, df):
        """Add cyclical time features (sin/cos of day, week, month, year) to the DataFrame."""
        if not isinstance(df.index, pd.DatetimeIndex):
            print("Warning: DataFrame index is not DatetimeIndex. Cannot add time features.")
            return df

        print("Adding cyclical time features...")
        timestamp_s = df.index.astype(np.int64) // 10**9 # Convert to Unix timestamp (seconds)

        day = 24 * 60 * 60
        week = 7 * day
        month = 30.44 * day # Average month
        year = 365.2425 * day

        df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        df['Week sin'] = np.sin(timestamp_s * (2 * np.pi / week))
        df['Week cos'] = np.cos(timestamp_s * (2 * np.pi / week))
        # Month features might be less useful due to varying length
        # df['Month sin'] = np.sin(timestamp_s * (2 * np.pi / month))
        # df['Month cos'] = np.cos(timestamp_s * (2 * np.pi / month))
        df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
        df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
        print("Time features added.")
        return df

    def create_prediction_dataset(self, points=None):
        """
        Create a dataset ready for prediction by combining loaded ERA5 variables
        and climate data. Operates on the data currently held in self.datasets.

        Parameters:
        -----------
        points : list of dict, optional
            List of points [{lat, lon, name}, ...] to extract data for.
            If None, returns the full grid data as a DataFrame with multi-index.

        Returns:
        --------
        pandas.DataFrame or list of pandas.DataFrames
            DataFrame(s) with features ready for prediction.
            If points is provided, returns a list of DataFrames, one per point.
            If points is None, returns a single DataFrame with a multi-index
            (time, latitude, longitude), potentially filtered to remove all-NaN rows.
        """
        print("\n--- Creating Prediction Dataset ---")
        if not self.datasets:
            print("Error: No datasets loaded. Cannot create prediction dataset.")
            return None

        # --- Combine ERA5 Variables ---
        # Align all datasets to a common grid and time index
        # Use an outer join to keep all data points, filling missing with NaN
        print("Aligning loaded ERA5 datasets...")
        # Filter out None datasets before aligning
        datasets_to_align = [ds for ds in self.datasets.values() if ds is not None]
        if not datasets_to_align:
            print("Error: No valid datasets available for alignment.")
            return None
        aligned_datasets = xr.align(*datasets_to_align, join='outer')
        # Merge into a single dataset
        combined_ds = xr.merge(aligned_datasets, compat='override') # compat='override' might be needed if coords differ slightly
        print(f"Combined ERA5 dataset size: {_memory_usage_mb(combined_ds):.2f} MB")

        # --- Add Climate Data ---
        if 'latitude' in combined_ds.coords and 'longitude' in combined_ds.coords:
            print("Adding climate data...")
            lats = combined_ds['latitude'].values
            lons = combined_ds['longitude'].values
            lon_grid, lat_grid = np.meshgrid(lons, lats)

            # Get climate data using KD-Tree lookup
            temps, precips = self.get_climate_at_location_kdtree(lon_grid, lat_grid)

            # Add as static variables (same value for all times)
            # Ensure correct shape (lat, lon) matching the grid
            temp_da = xr.DataArray(temps, coords={'latitude': lats, 'longitude': lons}, dims=['latitude', 'longitude'])
            precip_da = xr.DataArray(precips, coords={'latitude': lats, 'longitude': lons}, dims=['latitude', 'longitude'])

            # Use the standard renamed variable names directly
            temp_var_name = config.VARIABLE_RENAME.get('annual_mean_temperature', 'annual_mean_temp')
            precip_var_name = config.VARIABLE_RENAME.get('annual_precipitation', 'annual_precip')
            combined_ds[temp_var_name] = temp_da
            combined_ds[precip_var_name] = precip_da
            print("Climate data added.")
        else:
            print("Warning: Latitude/Longitude coordinates missing. Cannot add climate data.")

        # --- Rename Variables to Model Names ---
        # Create the rename map based on the original keys used for loading/calculation
        rename_map = {}
        for orig_name in list(self.datasets.keys()): # Iterate over original keys
             # Find the corresponding variable name in the combined dataset (might be the same)
             actual_var_name = list(self.datasets[orig_name].data_vars)[0] if self.datasets[orig_name] and len(self.datasets[orig_name].data_vars)==1 else orig_name
             target_name = config.VARIABLE_RENAME.get(orig_name, actual_var_name) # Get target name from config based on original key
             if actual_var_name in combined_ds.data_vars and actual_var_name != target_name:
                  rename_map[actual_var_name] = target_name

        # Add climate vars if they exist and need renaming (already added with target name, so skip)
        # temp_var_name = config.VARIABLE_RENAME.get('annual_mean_temperature', 'annual_mean_temp')
        # precip_var_name = config.VARIABLE_RENAME.get('annual_precipitation', 'annual_precip')
        # if 'annual_mean_temperature' in combined_ds.data_vars and 'annual_mean_temperature' != temp_var_name:
        #      rename_map['annual_mean_temperature'] = temp_var_name
        # if 'annual_precipitation' in combined_ds.data_vars and 'annual_precipitation' != precip_var_name:
        #      rename_map['annual_precipitation'] = precip_var_name

        if rename_map:
             print(f"Renaming variables: {rename_map}")
             combined_ds = combined_ds.rename(rename_map)
        print(f"Final variables in dataset: {list(combined_ds.data_vars)}")


        # --- Extract Points or Convert Full Grid ---
        if points:
             # --- Points Extraction ---
             print(f"Extracting data for {len(points)} points...")
             all_point_dfs = []
             for i, point in enumerate(points):
                  lat, lon = point['lat'], point['lon']
                  name = point.get('name', f"Point_{i}_{lat:.4f}_{lon:.4f}")
                  print(f"  Processing point: {name} ({lat}, {lon})")
                  try:
                       # Select nearest neighbor for the point
                       point_ds = combined_ds.sel(latitude=lat, longitude=lon, method='nearest')

                       # Convert to DataFrame
                       # Drop dimensions that become scalar after selection
                       df = point_ds.to_dataframe().reset_index() # Reset index to easily add cols
                       # Remove original lat/lon columns if they exist after reset_index
                       df = df.drop(columns=['latitude', 'longitude'], errors='ignore')

                       # Add point metadata back (lat, lon, name)
                       df['latitude'] = lat
                       df['longitude'] = lon
                       df['name'] = name

                       # Set time index
                       if 'time' in df.columns:
                            df = df.set_index('time')
                       else:
                            print(f"Warning: No 'time' column found for point {name}")


                       # Add time features if configured
                       if config.TIME_FEATURES:
                            df = self.add_time_features(df)

                       # Remove rows where all original ERA5/derived vars are NaN
                       # Identify the columns to check for NaNs (exclude lat, lon, name, time features)
                       cols_to_check = [col for col in df.columns if col not in ['latitude', 'longitude', 'name'] and ' sin' not in col and ' cos' not in col]
                       # Drop rows where all these specific columns are NaN
                       df.dropna(subset=cols_to_check, how='all', inplace=True)

                       if df.empty:
                            print(f"  Warning: No valid data found for point {name} after NaN check.")
                       else:
                            all_point_dfs.append(df)

                  except Exception as e:
                       print(f"  Error processing point {name}: {e}")
                       import traceback
                       traceback.print_exc()


             print(f"Finished extracting points. {len(all_point_dfs)} successful.")
             # Return list of DataFrames, one per point
             return all_point_dfs # Already filtered for non-empty
        else:
             # --- Full Grid Conversion ---
             print("Converting full grid to DataFrame...")
             # Convert the entire dataset to a DataFrame
             # This will create a MultiIndex (time, latitude, longitude)
             df = combined_ds.to_dataframe()

             # Drop rows where all data variables are NaN
             # Identify the actual data variable columns after potential renaming
             data_cols = list(combined_ds.data_vars.keys())
             print(f"Dropping rows where all columns in {data_cols} are NaN...")
             initial_rows = len(df)
             df.dropna(subset=data_cols, how='all', inplace=True)
             final_rows = len(df)
             print(f"Dropped {initial_rows - final_rows} all-NaN rows.")


             # Add time features if configured and time is in index
             if config.TIME_FEATURES and 'time' in df.index.names:
                  # Need to handle multi-index carefully
                  # Option 1: Reset index, add features, set index back
                  idx_names = df.index.names
                  df_reset = df.reset_index()
                  # Temporarily set time as index for add_time_features
                  df_time_idx = df_reset.set_index('time')
                  df_time_idx = self.add_time_features(df_time_idx)
                  # Merge features back and restore original index
                  # Need to handle potential duplicate columns if time features already exist
                  time_feature_cols = [col for col in df_time_idx.columns if ' sin' in col or ' cos' in col]
                  df = df_reset.set_index('time').join(df_time_idx[time_feature_cols]).reset_index().set_index(idx_names)

             elif config.TIME_FEATURES and 'time' in df.columns:
                  # If time is a column after reset_index
                  df = df.set_index('time')
                  df = self.add_time_features(df)
                  df = df.reset_index() # Put time back as column if needed

             print(f"Full grid DataFrame created with shape {df.shape}")
             return df

    def save_prediction_dataset(self, df_or_list, output_prefix, format='netcdf'):
        """
        Save the prediction dataset(s) to file(s).

        Parameters:
        -----------
        df_or_list : pandas.DataFrame or list of pandas.DataFrames
            Prediction dataset(s) to save.
        output_prefix : str or Path
            Prefix for the output file(s) (e.g., 'prediction_2020_01').
            Directory will be created if it doesn't exist.
        format : str, optional
             Output format: 'netcdf' (recommended), 'csv', 'zarr'.
        """
        output_prefix = Path(output_prefix)
        output_prefix.parent.mkdir(exist_ok=True, parents=True)
        print(f"\n--- Saving Prediction Dataset(s) (Prefix: {output_prefix}, Format: {format}) ---")

        if isinstance(df_or_list, list):
            # --- Save Multiple Point DataFrames ---
            saved_paths = []
            for i, df in enumerate(df_or_list):
                if df is None or df.empty:
                    print(f"Skipping empty DataFrame for point {i}.")
                    continue

                # Try to get name, lat, lon for filename
                # Ensure we handle cases where these might not be columns (e.g., if index wasn't reset)
                name_val = df['name'].iloc[0] if 'name' in df.columns and not df['name'].empty else f"point_{i}"
                # Sanitize name for filename
                safe_name = "".join(c if c.isalnum() else "_" for c in str(name_val)) # Ensure name is string
                filename = f"{output_prefix.name}_{safe_name}"

                output_path = output_prefix.parent / filename

                try:
                    print(f"Saving {name_val} to {output_path}.{format}...")
                    if format == 'netcdf':
                         if not HAVE_NETCDF4:
                              print("NetCDF4 library not found. Cannot save to NetCDF.")
                              continue
                         # Convert DataFrame back to xarray Dataset for saving
                         # Best effort: reset index, set time index, then convert
                         df_to_save = df.reset_index()
                         if 'time' in df_to_save.columns:
                              df_to_save = df_to_save.set_index('time')
                         else:
                              print("Warning: 'time' column not found for NetCDF conversion. Index may be incorrect.")

                         ds = df_to_save.to_xarray()

                         # Add metadata
                         ds.attrs['description'] = f"Prediction dataset for point {name_val}"
                         ds.attrs['creation_date'] = datetime.now(timezone.utc).isoformat()
                         if 'latitude' in ds.coords or 'latitude' in ds.data_vars:
                              ds.attrs['point_latitude'] = float(df['latitude'].iloc[0]) if 'latitude' in df.columns else 'unknown'
                         if 'longitude' in ds.coords or 'longitude' in ds.data_vars:
                              ds.attrs['point_longitude'] = float(df['longitude'].iloc[0]) if 'longitude' in df.columns else 'unknown'


                         # Save with compression
                         encoding = {var: {'zlib': True, 'complevel': 5} for var in ds.data_vars}
                         ds.to_netcdf(f"{output_path}.nc", encoding=encoding, mode='w') # Ensure overwrite
                         saved_paths.append(f"{output_path}.nc")

                    elif format == 'zarr':
                         # Convert DataFrame back to xarray Dataset for saving
                         df_to_save = df.reset_index()
                         if 'time' in df_to_save.columns:
                              df_to_save = df_to_save.set_index('time')
                         ds = df_to_save.to_xarray()
                         ds.attrs['description'] = f"Prediction dataset for point {name_val}"
                         ds.attrs['creation_date'] = datetime.now(timezone.utc).isoformat()
                         if 'latitude' in ds.coords or 'latitude' in ds.data_vars:
                              ds.attrs['point_latitude'] = float(df['latitude'].iloc[0]) if 'latitude' in df.columns else 'unknown'
                         if 'longitude' in ds.coords or 'longitude' in ds.data_vars:
                              ds.attrs['point_longitude'] = float(df['longitude'].iloc[0]) if 'longitude' in df.columns else 'unknown'

                         ds.to_zarr(f"{output_path}.zarr", mode='w') # Overwrite if exists
                         saved_paths.append(f"{output_path}.zarr")
                    elif format == 'csv':
                         # Ensure index is included if it's meaningful (like time)
                         df.to_csv(f"{output_path}.csv", index=isinstance(df.index, pd.DatetimeIndex))
                         saved_paths.append(f"{output_path}.csv")
                    else:
                         print(f"Unsupported format: {format}")
                         continue
                    print(f"  Successfully saved {name_val}.")
                except Exception as e:
                    print(f"  Error saving {name_val}: {e}")
                    import traceback
                    traceback.print_exc()
            return saved_paths

        elif isinstance(df_or_list, pd.DataFrame):
            # --- Save Single Grid DataFrame ---
            df = df_or_list
            if df.empty:
                 print("DataFrame is empty. Nothing to save.")
                 return None

            output_path = Path(f"{output_prefix}.{format}")
            try:
                print(f"Saving grid data to {output_path}...")
                if format == 'netcdf':
                    if not HAVE_NETCDF4:
                        print("NetCDF4 library not found. Cannot save to NetCDF.")
                        return None
                    # Convert DataFrame (potentially with MultiIndex) to Dataset
                    if isinstance(df.index, pd.MultiIndex):
                         ds = df.to_xarray()
                    else: # Assume index needs to be reset/set
                         # This might need adjustment based on how df was created
                         index_cols = ['time', 'latitude', 'longitude'] # Expected index cols
                         df_reset = df.reset_index()
                         if all(col in df_reset.columns for col in index_cols):
                              ds = df_reset.set_index(index_cols).to_xarray()
                         elif 'time' in df_reset.columns: # Fallback if only time is index/column
                              print("Warning: Creating NetCDF with only 'time' index.")
                              ds = df_reset.set_index('time').to_xarray()
                         else:
                              print("Warning: Could not determine standard index columns for NetCDF conversion.")
                              ds = df.to_xarray() # Best effort conversion

                    ds.attrs['description'] = "Prediction dataset grid"
                    ds.attrs['creation_date'] = datetime.now(timezone.utc).isoformat()
                    encoding = {var: {'zlib': True, 'complevel': 5} for var in ds.data_vars}
                    ds.to_netcdf(output_path, encoding=encoding, mode='w') # Ensure overwrite
                elif format == 'zarr':
                    # Similar conversion to xarray needed
                    if isinstance(df.index, pd.MultiIndex):
                         ds = df.to_xarray()
                    else:
                         index_cols = ['time', 'latitude', 'longitude']
                         df_reset = df.reset_index()
                         if all(col in df_reset.columns for col in index_cols):
                              ds = df_reset.set_index(index_cols).to_xarray()
                         elif 'time' in df_reset.columns:
                              ds = df_reset.set_index('time').to_xarray()
                         else:
                              ds = df.to_xarray()

                    ds.attrs['description'] = "Prediction dataset grid"
                    ds.attrs['creation_date'] = datetime.now(timezone.utc).isoformat()
                    ds.to_zarr(output_path, mode='w')
                elif format == 'csv':
                     # Save with index if it's a MultiIndex or DatetimeIndex
                    save_index = isinstance(df.index, (pd.MultiIndex, pd.DatetimeIndex))
                    df.to_csv(output_path, index=save_index)
                else:
                    print(f"Unsupported format: {format}")
                    return None
                print(f"Successfully saved grid data.")
                return str(output_path)
            except Exception as e:
                print(f"Error saving grid data: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            print("Invalid input type for saving. Expected DataFrame or list of DataFrames.")
            return None

    def process_period(self, variables, start_date, end_date, output_prefix,
                       shapefile=None, region_dict=None, points=None,
                       data_dir=None, gee_scale=11132,
                       target_chunks={'time': 24*7, 'latitude': 100, 'longitude': 100}, # Process weekly chunks
                       save_format='netcdf'):
        """
        High-level function to process data for a given period, calculate derived
        variables, create the prediction dataset, and save it.

        Parameters:
        -----------
        variables : list
            List of standard variable names to load (e.g., '2m_temperature').
        start_date, end_date : str or datetime
            Start and end dates for the data period.
        output_prefix : str or Path
            Prefix for the output file(s).
        shapefile : str, optional
            Path to shapefile for region definition.
        region_dict : dict, optional
             Dictionary with {'lat_min', 'lat_max', 'lon_min', 'lon_max'}
        points : list of dict, optional
            List of points [{lat, lon, name}, ...] to extract data for.
        data_dir : str or Path, optional
            Path to directory containing local data files.
        gee_scale : float, optional
            Spatial resolution in meters for GEE export (if used).
        target_chunks : dict, optional
            Desired Dask chunking for loading and processing.
        save_format : str, optional
             Output format: 'netcdf', 'csv', 'zarr'.

        Returns:
        --------
        str or list or None
            Path(s) to the saved file(s), or None if processing failed.
        """
        start_proc = time.time()
        try:
            # 1. Load Data (Handles local files and GEE fallback)
            self.load_variable_data(
                variables=variables,
                start_date=start_date,
                end_date=end_date,
                shapefile=shapefile,
                region_dict=region_dict,
                data_dir=data_dir,
                gee_scale=gee_scale,
                target_chunks=target_chunks
            )

            if not self.datasets:
                 print("Failed to load any variable data for the period.")
                 return None

            # 2. Calculate Derived Variables (Uses Dask/Xarray internally)
            self.calculate_derived_variables()

            # 3. Create Prediction Dataset (DataFrame or list of DataFrames)
            pred_data = self.create_prediction_dataset(points=points)

            # Check if pred_data is None or empty (handles both list and DataFrame cases)
            is_empty = False
            if pred_data is None:
                is_empty = True
            elif isinstance(pred_data, list):
                # Check if the list itself is empty OR if all DataFrames within it are empty
                if not pred_data:
                    is_empty = True
                else: # Check if all contained dataframes are empty
                    is_empty = all(df is None or df.empty for df in pred_data)
            elif isinstance(pred_data, pd.DataFrame):
                is_empty = pred_data.empty

            if is_empty:
                print("Prediction dataset creation failed or resulted in empty data.")
                return None


            # 4. Save the final dataset(s)
            saved_paths = self.save_prediction_dataset(
                df_or_list=pred_data,
                output_prefix=output_prefix,
                format=save_format
            )

            end_proc = time.time()
            print(f"\n--- Processing complete for {start_date} to {end_date} ---")
            print(f"Total time: {end_proc - start_proc:.2f} seconds")
            return saved_paths

        except Exception as e:
            print(f"\n--- ERROR during processing period {start_date} to {end_date} ---")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
             # Clean up datasets for the next potential run
             self._clear_datasets()
             gc.collect()

    def _clear_datasets(self):
        """Clear loaded ERA5 datasets to free memory."""
        print("Clearing loaded ERA5 datasets...")
        cleared_count = 0
        for var_name in list(self.datasets.keys()):
             try:
                  if hasattr(self.datasets[var_name], 'close'):
                       self.datasets[var_name].close()
                  del self.datasets[var_name]
                  cleared_count += 1
             except Exception as e:
                  print(f"  Warning: Could not clear dataset for {var_name}: {e}")
        self.datasets = {}
        print(f"Cleared {cleared_count} datasets.")
        gc.collect()


# --- Main Execution Example ---

def main():
    """
    Example main function demonstrating the use of the optimized processor.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Process ERA5-Land data using optimized processor')
    parser.add_argument('--year', type=int, required=True, help='Start year to process')
    parser.add_argument('--month', type=int, required=True, help='Start month to process')
    # Allow specifying end year/month for ranges
    parser.add_argument('--end-year', type=int, help='End year for processing range (inclusive, defaults to start year)')
    parser.add_argument('--end-month', type=int, help='End month for processing range (inclusive, defaults to start month)')

    parser.add_argument('--output-prefix', type=str, help='Prefix for output file(s) (e.g., ./output/era5_pred)')
    parser.add_argument('--format', type=str, default='netcdf', choices=['netcdf', 'csv', 'zarr'], help='Output file format')

    parser.add_argument('--shapefile', type=str, default=None, help='Path to shapefile for region definition')
    parser.add_argument('--lat-min', type=float, default=None, help='Minimum latitude (overrides shapefile/default)')
    parser.add_argument('--lat-max', type=float, default=None, help='Maximum latitude (overrides shapefile/default)')
    parser.add_argument('--lon-min', type=float, default=None, help='Minimum longitude (overrides shapefile/default)')
    parser.add_argument('--lon-max', type=float, default=None, help='Maximum longitude (overrides shapefile/default)')
    parser.add_argument('--points-csv', type=str, help='CSV file with points (columns: lat, lon, name)')


    parser.add_argument('--gee-scale', type=float, default=11132, help='Scale in meters for GEE export (default ~9km)')
    parser.add_argument('--data-dir', type=str, help='Directory containing local NetCDF/TIF data files (optional)')

    # Dask/Parallelism options
    parser.add_argument('--disable-dask', action='store_true', help='Disable Dask client creation')
    parser.add_argument('--num-workers', type=int, help='Number of Dask worker processes')
    parser.add_argument('--memory-limit', type=str, help='Memory limit per Dask worker (e.g., 4GB)')


    args = parser.parse_args()

    # Determine processing period
    start_date = pd.Timestamp(f"{args.year}-{args.month:02d}-01")
    end_year = args.end_year or args.year
    end_month = args.end_month or args.month
    # End date for filtering should be the start of the *next* month after the desired end month
    if end_month == 12:
         end_date_exclusive = pd.Timestamp(f"{end_year + 1}-01-01")
    else:
         end_date_exclusive = pd.Timestamp(f"{end_year}-{end_month + 1:02d}-01")

    print(f"Processing Period: {start_date.strftime('%Y-%m-%d')} to {end_date_exclusive.strftime('%Y-%m-%d')} (exclusive end)")

    # Define region dictionary if lat/lon args are provided (overrides shapefile/defaults)
    region = None
    if args.lat_min is not None and args.lat_max is not None and \
       args.lon_min is not None and args.lon_max is not None:
        region = {
            'lat_min': args.lat_min, 'lat_max': args.lat_max,
            'lon_min': args.lon_min, 'lon_max': args.lon_max
        }
        print(f"Using specified lat/lon bounds: {region}")


    # Load points if provided
    points_list = None
    if args.points_csv:
        try:
            points_df = pd.read_csv(args.points_csv)
            # Ensure required columns exist
            if not {'lat', 'lon'}.issubset(points_df.columns):
                 raise ValueError("Points CSV must contain 'lat' and 'lon' columns.")
            if 'name' not in points_df.columns:
                 points_df['name'] = [f"Point_{i}" for i in range(len(points_df))]
            points_list = points_df[['lat', 'lon', 'name']].to_dict('records')
            print(f"Loaded {len(points_list)} points from {args.points_csv}")
        except Exception as e:
            print(f"Error loading points CSV: {e}. Proceeding without point extraction.")
            points_list = None

    # Determine output prefix
    output_prefix = args.output_prefix
    if not output_prefix:
         # Create a default prefix based on the period
         start_str = start_date.strftime('%Y%m')
         end_dt_inclusive = end_date_exclusive - pd.Timedelta(days=1) # Get last day of period
         end_str = end_dt_inclusive.strftime('%Y%m')
         output_prefix = config.PREDICTION_DIR / f"era5_pred_{start_str}_{end_str}"
    else:
         output_prefix = Path(output_prefix) # Ensure it's a Path object

    # Initialize the processor
    # Explicitly disable Dask if requested
    dask_create = not args.disable_dask if HAVE_DASK else False

    processor = None # Ensure processor is defined outside try block
    try:
        processor = ERA5LandGEEProcessor(
            memory_limit=args.memory_limit,
            num_workers=args.num_workers,
            # Pass other relevant init args like climate file paths if needed
            temp_climate_file=config.TEMP_CLIMATE_FILE,
            precip_climate_file=config.PRECIP_CLIMATE_FILE,
        )

        # Run the processing for the period
        processor.process_period(
            variables=config.REQUIRED_VARIABLES,
            start_date=start_date,
            end_date=end_date_exclusive, # Use exclusive end date for loading
            output_prefix=output_prefix,
            shapefile=args.shapefile,
            region_dict=region, # Pass explicitly defined region if available
            points=points_list,
            data_dir=args.data_dir,
            gee_scale=args.gee_scale,
            save_format=args.format
            # Pass target_chunks if needed
        )

    except Exception as e:
        print(f"\n--- Main execution failed ---")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup happens
        if processor:
            processor.close()
        print("Main execution finished.")


if __name__ == "__main__":
    main()
