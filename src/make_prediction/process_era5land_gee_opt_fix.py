"""
ERA5-Land data processing utility for sap velocity prediction.
Handles extraction, reading, and preprocessing of ERA5-Land data
from Google Earth Engine with maintained functionality for derived variables.
"""

import os
import logging
import sys

logger = logging.getLogger(__name__)



# Find and set PROJ_LIB path
def setup_proj():
    """Set up PROJ_LIB environment variable."""
    # Common locations for proj.db
    possible_paths = []

    # Check virtual environment first
    if hasattr(sys, "prefix"):
        venv_paths = [
            os.path.join(sys.prefix, "Library", "share", "proj"),  # Windows conda/venv
            os.path.join(sys.prefix, "share", "proj"),  # Linux/Mac
            os.path.join(sys.prefix, "Lib", "site-packages", "pyproj", "proj_dir", "share", "proj"),  # pyproj bundled
        ]
        possible_paths.extend(venv_paths)

    # Check site-packages for pyproj bundled proj
    try:
        import pyproj

        pyproj_path = os.path.dirname(pyproj.__file__)
        possible_paths.extend(
            [
                os.path.join(pyproj_path, "proj_dir", "share", "proj"),
                os.path.join(os.path.dirname(pyproj_path), "pyproj.libs"),
            ]
        )

        # pyproj >= 3.0 has datadir
        if hasattr(pyproj, "datadir"):
            proj_dir = pyproj.datadir.get_data_dir()
            if proj_dir:
                possible_paths.insert(0, proj_dir)
    except ImportError:
        pass

    # Check for OSGEO4W installation (common on Windows)
    osgeo_paths = [
        r"C:\OSGeo4W64\share\proj",
        r"C:\OSGeo4W\share\proj",
        r"C:\Program Files\QGIS 3.28\share\proj",
        r"C:\Program Files\QGIS 3.34\share\proj",
    ]
    possible_paths.extend(osgeo_paths)

    # Find the first valid path containing proj.db
    for path in possible_paths:
        proj_db = os.path.join(path, "proj.db")
        if os.path.exists(proj_db):
            os.environ["PROJ_LIB"] = path
            os.environ["PROJ_DATA"] = path  # For newer PROJ versions
            logger.info("PROJ_LIB set to: %s", path)
            return True

    # If not found, try to find proj.db anywhere in site-packages
    try:
        import site

        for site_path in site.getsitepackages():
            for root, dirs, files in os.walk(site_path):
                if "proj.db" in files:
                    os.environ["PROJ_LIB"] = root
                    os.environ["PROJ_DATA"] = root
                    logger.info("PROJ_LIB set to: %s", root)
                    return True
    except Exception:
        pass

    logger.warning("Could not find proj.db. CRS operations may fail.")
    return False


# Set up PROJ before importing geo libraries
setup_proj()

import calendar
import glob
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import config

# Add Google Earth Engine imports
import ee
import geemap
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.windows import from_bounds
from scipy.spatial import cKDTree

# Try optional libraries
try:
    import dask
    from dask.diagnostics import ProgressBar

    HAVE_DASK = True

    # Configure dask for better memory usage
    dask.config.set(
        {
            "array.chunk-size": "32MiB",
            "distributed.worker.memory.target": 0.6,
            "distributed.worker.memory.spill": 0.7,
            "distributed.worker.memory.pause": 0.8,
        }
    )
except ImportError:
    HAVE_DASK = False
    print("Dask not available. Some memory optimizations disabled.")

try:
    import netCDF4

    HAVE_NETCDF4 = True
except ImportError:
    HAVE_NETCDF4 = False
    print("NetCDF4 module not available. Using fallback methods.")
# --- ADD THIS AT THE TOP LEVEL OF YOUR FILE (After imports) ---


def global_worker_init():
    """
    Initializer that runs inside every worker process immediately upon creation.
    It sets up the GEE session using credentials cached by the main process.
    """
    import ee

    try:
        # Re-initialize using the project ID.
        # This reads the credentials from disk/environment variables.
        # It does NOT open a browser or ask for interactive auth.
        ee.Initialize(project="ee-yuluo-2")
    except Exception as e:
        # On Windows, printing might get swallowed depending on the IDE,
        # but it's good to try.
        print(f"!!! Worker Initialization Failed: {e}")


def _hour_processing_worker(args):
    """
    Worker function for processing a single hour.
    Saves results to a temporary file to avoid Multiprocessing Data Leaks.
    """
    year, month, day, hour, variables, shapefile, output_dir, temp_dir = args

    # Create a unique temp directory for this specific hour
    hour_id = f"{year}{month:02d}{day:02d}_{hour:02d}"
    hour_temp_dir = Path(temp_dir) / f"worker_{hour_id}"
    hour_temp_dir.mkdir(exist_ok=True, parents=True)

    # Initialize processor for this worker
    # We set gee_initialize=True to ensure each process has its own GEE session
    hour_processor = ERA5LandGEEProcessor(temp_dir=hour_temp_dir, gee_initialize=False)

    try:
        print(f"--- Process Start: Hour {hour:02d}:00 ---")

        # Determine timestamps
        hour_start = pd.Timestamp(year=year, month=month, day=day, hour=hour, tz="UTC")

        # LOGIC FIX: Solar Radiation in ERA5-Land is accumulated.
        # To get the value for Hour H, we often need Hour H and Hour H-1.
        # We load a small buffer if needed, or handle it in the GEE call.
        hour_processor.load_variable_data(variables, year, month, day, hour, shapefile=shapefile)

        # Calculate derived variables (VPD, Wind Speed, etc.)
        hour_processor.calculate_derived_variables()

        # Create the dataset for this specific hour
        # This will be a small DataFrame (Lat x Lon for 1 timestamp)
        df = hour_processor.create_prediction_dataset(use_existing_data=True)

        if df is not None and not df.empty:
            # FIX #7: Save to disk immediately.
            # We use a temporary CSV. Index=False makes it easier to merge later.
            temp_file_path = hour_temp_dir / f"hour_{hour:02d}_raw.csv"
            df.to_csv(temp_file_path, index=False)

            print(f"--- Process Success: Hour {hour:02d}:00 (Saved to Temp) ---")
            return {"hour": hour, "file_path": str(temp_file_path), "success": True, "timestamp": hour_start}
        else:
            return {"hour": hour, "success": False, "error": "Empty DataFrame"}

    except Exception as e:
        print(f"!!! Process Error Hour {hour:02d}: {str(e)} !!!")
        traceback.print_exc()
        return {"hour": hour, "success": False, "error": str(e)}

    finally:
        # Crucial for memory management in multiprocessing
        hour_processor.close()
        import gc

        gc.collect()


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
        # Process the day's data
        return day_processor.process_single_day(variables, year, month, day, shapefile=shapefile, output_dir=output_dir)
    finally:
        day_processor.close()


class ERA5LandGEEProcessor:
    """
    Class to process ERA5-Land data from Google Earth Engine with optimized methods
    for memory efficiency. Based on the ERA5LandProcessor with modifications for
    accessing data directly from GEE instead of local files.
    """

    # Fix 5.4: Named physical constants (previously magic numbers)
    MAGNUS_A = 6.1078  # hPa, Tetens formula saturation vapor pressure coefficient
    MAGNUS_B = 17.269  # dimensionless, Tetens formula exponent numerator
    MAGNUS_C = 237.3  # degC, Tetens formula exponent denominator
    PAR_FRACTION = 0.45  # fraction of shortwave radiation that is PAR
    PPFD_CONVERSION = 4.6  # umol photons per J conversion factor for PAR
    SOLAR_CONSTANT = 1367.0  # W/m2, total solar irradiance (Gsc)
    KELVIN_OFFSET = 273.15  # K, offset between Kelvin and Celsius

    # Format: (temp_min, temp_max, precip_min, precip_max)
    # Temperatures in degC, precipitation in mm/year

    def __init__(
        self,
        temp_dir=None,
        create_client=None,
        memory_limit=None,
        temp_climate_file=None,
        precip_climate_file=None,
        gee_initialize=True,
        time_scale="daily",
    ):
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
        self.ee_initialized = False
        import ee

        self.time_scale = time_scale
        if self.time_scale not in ["hourly", "daily"]:
            raise ValueError("time_scale must be either 'hourly' or 'daily'")
        if self.time_scale == "hourly":
            self.GEE_VARIABLE_MAPPING = {
                "temperature_2m": "temperature_2m",
                "dewpoint_temperature_2m": "dewpoint_temperature_2m",
                "total_precipitation": "total_precipitation_hourly",
                "surface_solar_radiation_downwards": "surface_solar_radiation_downwards_hourly",
                "10m_u_component_of_wind": "u_component_of_wind_10m",
                "10m_v_component_of_wind": "v_component_of_wind_10m",
                "volumetric_soil_water_layer_1": "volumetric_soil_water_layer_1",
                "soil_temperature_level_1": "soil_temperature_level_1",
                "potential_evaporation": "potential_evaporation_hourly",  # For PET calculation
            }
        elif self.time_scale == "daily":
            self.GEE_VARIABLE_MAPPING = {
                "temperature_2m": "temperature_2m",
                "temperature_2m_min": "temperature_2m_min",
                "temperature_2m_max": "temperature_2m_max",
                "dewpoint_temperature_2m": "dewpoint_temperature_2m",
                "dewpoint_temperature_2m_min": "dewpoint_temperature_2m_min",
                "dewpoint_temperature_2m_max": "dewpoint_temperature_2m_max",
                "total_precipitation": "total_precipitation_sum",
                "surface_solar_radiation_downwards": "surface_solar_radiation_downwards_sum",
                "10m_u_component_of_wind": "u_component_of_wind_10m",
                "10m_u_component_of_wind_min": "u_component_of_wind_10m_min",
                "10m_u_component_of_wind_max": "u_component_of_wind_10m_max",
                "10m_v_component_of_wind": "v_component_of_wind_10m",
                "10m_v_component_of_wind_min": "v_component_of_wind_10m_min",
                "10m_v_component_of_wind_max": "v_component_of_wind_10m_max",
                "volumetric_soil_water_layer_1": "volumetric_soil_water_layer_1",
                "soil_temperature_level_1": "soil_temperature_level_1",
                "potential_evaporation": "potential_evaporation_sum",
            }
        # Check if a session is already active (set by global_worker_init)
        if ee.data._credentials:
            self.ee_initialized = True
        elif gee_initialize:
            # Only explicitly initialize if requested AND not already done
            self.initialize_gee()
        self.temp_dir = Path(temp_dir) if temp_dir else config.ERA5LAND_TEMP_DIR

        # Create temporary directory if it doesn't exist
        self.temp_dir.mkdir(exist_ok=True, parents=True)

        # Set climate file paths
        self.temp_climate_file = (
            Path(temp_climate_file) if temp_climate_file else getattr(config, "TEMP_CLIMATE_FILE", None)
        )
        self.precip_climate_file = (
            Path(precip_climate_file) if precip_climate_file else getattr(config, "PRECIP_CLIMATE_FILE", None)
        )

        # Initialize climate data attributes
        self.temp_climate_data = None
        self.precip_climate_data = None
        self.resampled_temp_data = None
        self.resampled_precip_data = None
        # Initialize static data attributes
        # Initialize static data attributes (raw data with transform info)
        self.elevation_raw = None
        self.pft_raw = None
        self.canopy_height_raw = None
        self.lai_raw = None

        # Resampled static data (aligned to ERA5 grid)
        self.elevation_data = None
        self.pft_data = None
        self.canopy_height_data = None
        self.lai_data = None

        # Path for local LAI data
        self.lai_data_dir = Path(getattr(config, "LAI_DATA_DIR", "./data/raw/grided/globmap_lai/"))

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

        # Fix 1.3: Only initialize GEE if not already initialized by worker or earlier code
        if not self.ee_initialized and gee_initialize:
            self.initialize_gee()

        # Try to load climate data if files are provided
        if self.temp_climate_file and self.precip_climate_file:
            if self.temp_climate_file.exists() and self.precip_climate_file.exists():
                print("Loading climate data during initialization...")
                self.load_climate_data(
                    bounds={
                        "lon_min": -180.0,  # Eastern longitude boundary
                        "lat_min": -57,  # Southern latitude boundary
                        "lon_max": 180.0,  # Western longitude boundary
                        "lat_max": 78.0,  # Northern latitude boundary
                    }
                )

    def __del__(self):
        """Clean up resources on object deletion."""
        self.close()

    def close(self):
        """Close all open datasets and clients."""
        # Close all open datasets
        for var_name, ds in self.datasets.items():
            try:
                if hasattr(ds, "close"):
                    ds.close()
            except Exception:
                pass

        # Close Dask client if it exists
        if self.client:
            try:
                self.client.close()
                print("Closed Dask client")
            except Exception:
                pass
            self.client = None

    def initialize_gee(self):
        """Initialize the Google Earth Engine API.

        Strategy: read stored credentials from ~/.config/earthengine/credentials
        and build an OAuth2 Credentials object directly. This avoids gcloud ADC
        flow which requires interactive browser auth on HPC compute nodes.
        """
        import json
        from pathlib import Path

        # Try multiple credential sources: EE credentials, then gcloud ADC
        cred_paths = [
            Path.home() / ".config" / "earthengine" / "credentials",
            Path.home() / ".config" / "gcloud" / "application_default_credentials.json",
        ]
        gee_project = getattr(config, "GEE_PROJECT", "era5download-447713")
        try:
            cred_data = None
            for cred_path in cred_paths:
                if cred_path.exists():
                    cred_data = json.loads(cred_path.read_text())
                    break
            if cred_data and "refresh_token" in cred_data:
                from google.oauth2.credentials import Credentials

                credentials = Credentials(
                    token=None,
                    refresh_token=cred_data["refresh_token"],
                    token_uri="https://oauth2.googleapis.com/token",
                    client_id=cred_data["client_id"],
                    client_secret=cred_data["client_secret"],
                )
                ee.Initialize(credentials=credentials, project=gee_project)
                self.ee_initialized = True
                print("GEE initialized with stored credentials from", cred_path)
            else:
                ee.Initialize(project="ee-yuluo-2")
                self.ee_initialized = True
                print("GEE initialized via default credentials")
        except Exception as e:
            print(f"Failed to initialize GEE: {e}")
            print("Run 'earthengine authenticate' on the login node first")
            self.ee_initialized = False

    def mask_zero_values(self, variables_to_check=None, threshold=1e-10, require_all_zero=True, apply_to_all=True):
        """
        Set grid cells to NaN based on zero values in specified variables.

        Parameters:
        -----------
        variables_to_check : list, optional
            List of variable names to check for zeros. If None, uses all raw ERA5 variables.
        threshold : float, optional
            Values below this threshold are considered "zero" (default 1e-10)
        require_all_zero : bool, optional
            If True, mask only where ALL specified variables are zero.
            If False, mask where ANY specified variable is zero.
        apply_to_all : bool, optional
            If True, apply the mask to all datasets.
            If False, only apply to the variables that were checked.

        Returns:
        --------
        tuple
            (mask_array, masked_cell_count)
        """
        if not self.datasets:
            print("No datasets loaded. Cannot apply zero mask.")
            return None, 0

        print("\n" + "=" * 60)
        print("MASKING ZERO VALUES")
        print("=" * 60)

        # Default raw ERA5-Land variables
        default_raw_variables = [
            "temperature_2m",
            "dewpoint_temperature_2m",
            "total_precipitation",
            "surface_solar_radiation_downwards",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "volumetric_soil_water_layer_1",
            "soil_temperature_level_1",
            "potential_evaporation",
        ]

        # Determine which variables to check
        if variables_to_check is None:
            variables_to_check = [v for v in default_raw_variables if v in self.datasets]
        else:
            variables_to_check = [v for v in variables_to_check if v in self.datasets]

        if not variables_to_check:
            print("No specified variables found in datasets.")
            print(f"Available datasets: {list(self.datasets.keys())}")
            return None, 0

        print(f"Variables to check: {variables_to_check}")
        print(f"Mode: {'ALL must be zero' if require_all_zero else 'ANY can be zero'}")

        # Get reference for dimensions
        ref_var = variables_to_check[0]
        ref_ds = self.datasets[ref_var]
        ref_data_var = list(ref_ds.data_vars)[0]
        ref_shape = ref_ds[ref_data_var].shape

        # Find coordinates
        lat_var, lon_var, time_var = None, None, None
        for coord in ref_ds.coords:
            coord_lower = coord.lower()
            if coord_lower in ("latitude", "lat"):
                lat_var = coord
            elif coord_lower in ("longitude", "lon"):
                lon_var = coord
            elif coord_lower in ("time", "datetime"):
                time_var = coord

        print(f"Reference shape: {ref_shape}")

        # Build mask
        if require_all_zero:
            # Start with True (assume all zeros), set to False where ANY var is non-zero
            combined_mask = np.ones(ref_shape, dtype=bool)
        else:
            # Start with False (assume all non-zero), set to True where ANY var is zero
            combined_mask = np.zeros(ref_shape, dtype=bool)

        for var_name in variables_to_check:
            ds = self.datasets[var_name]
            data_var_name = list(ds.data_vars)[0]
            data = ds[data_var_name].values

            if data.shape != ref_shape:
                print(f"  Warning: {var_name} shape {data.shape} doesn't match reference {ref_shape}, skipping")
                continue

            is_zero = np.abs(data) <= threshold

            if require_all_zero:
                # Mask only where ALL variables are zero
                combined_mask = combined_mask & is_zero
            else:
                # Mask where ANY variable is zero
                combined_mask = combined_mask | is_zero

            zero_count = np.sum(is_zero)
            total_count = data.size
            print(f"  {var_name}: {zero_count}/{total_count} cells are zero ({100 * zero_count / total_count:.2f}%)")

        # Summary
        masked_cell_count = np.sum(combined_mask)
        total_cell_count = combined_mask.size
        mask_percentage = (masked_cell_count / total_cell_count) * 100

        print("\nMask summary:")
        print(f"  Cells to mask: {masked_cell_count}/{total_cell_count} ({mask_percentage:.2f}%)")

        if masked_cell_count == 0:
            print("  No cells to mask.")
            return combined_mask, 0

        # Apply mask
        datasets_to_mask = self.datasets.keys() if apply_to_all else variables_to_check

        print(f"\nApplying mask to {len(list(datasets_to_mask))} datasets...")

        for var_name in list(datasets_to_mask):
            ds = self.datasets[var_name]
            try:
                data_var_name = list(ds.data_vars)[0]
                data = ds[data_var_name].values.copy()

                if data.shape == combined_mask.shape:
                    data[combined_mask] = np.nan
                    ds[data_var_name].values = data
                    print(f"  {var_name}: Masked")
                else:
                    print(f"  {var_name}: Shape mismatch, skipped")

            except Exception as e:
                print(f"  {var_name}: Error - {e}")

        self.zero_mask = combined_mask
        print("\nMasking complete.")
        print("=" * 60 + "\n")

        return combined_mask, masked_cell_count

    def process_month_daily(self, variables, year, month, shapefile=None, output_dir=None):
        """
        Process an entire month of daily ERA5-Land data and save to a single file.
        Optimized for daily time scale where data volume is much smaller.

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
        output_dir : str or Path, optional
            Directory to save the monthly prediction dataset

        Returns:
        --------
        str
            Path to saved prediction dataset file
        """
        import calendar

        print(f"\n{'=' * 60}")
        print(f"PROCESSING MONTH: {year}-{month:02d} (DAILY TIME SCALE)")
        print(f"{'=' * 60}\n")

        # Validate time scale
        if self.time_scale != "daily":
            print(f"Warning: time_scale is '{self.time_scale}', but this method is optimized for 'daily'.")
            print("Consider using process_month_parallel() for hourly data.")

        # Get number of days in month
        days_in_month = calendar.monthrange(year, month)[1]
        print(f"Days in month: {days_in_month}")

        # Create output directory
        if output_dir is None:
            output_dir = config.PREDICTION_DIR / f"{year}_{month:02d}_monthly"

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # ============================================================
        # OPTION 1: Load entire month at once (simpler, works for daily)
        # ============================================================
        try:
            print(f"\nLoading all daily data for {year}-{month:02d}...")

            # Load data for the entire month (day=None loads all days)
            self.load_variable_data(variables, year, month, day=None, shapefile=shapefile)

            # Apply zero mask
            print("\nApplying zero-value mask...")
            self.mask_zero_values(threshold=1e-10)

            # Calculate derived variables
            print("\nCalculating derived variables...")
            self.calculate_derived_variables()

            # Create prediction dataset for the entire month
            print("\nCreating prediction dataset for entire month...")
            start_date = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
            if month == 12:
                end_date = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC") - pd.Timedelta(seconds=1)
            else:
                end_date = pd.Timestamp(year=year, month=month + 1, day=1, tz="UTC") - pd.Timedelta(seconds=1)

            df = self.create_prediction_dataset(start_date=start_date, end_date=end_date, use_existing_data=True)

            if df is not None and not df.empty:
                # Save monthly prediction dataset
                output_file = output_dir / f"prediction_{year}_{month:02d}_daily.csv"
                saved_path = self.save_prediction_dataset(df, output_file)

                print(f"\n{'=' * 60}")
                print("SUCCESS: Monthly dataset saved")
                print(f"  File: {output_file}")
                print(f"  Rows: {len(df)}")
                print(f"  Columns: {len(df.columns)}")
                print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
                print(f"{'=' * 60}\n")

                return saved_path
            else:
                print(f"ERROR: No data created for {year}-{month:02d}")
                return None

        except Exception as e:
            print(f"ERROR processing month {year}-{month:02d}: {str(e)}")
            traceback.print_exc()
            return None

        finally:
            # Clean up
            self._clear_datasets()
            import gc

            gc.collect()

    def process_month_daily_chunked(self, variables, year, month, shapefile=None, output_dir=None, days_per_chunk=10):
        """
        Process an entire month of daily ERA5-Land data in chunks to manage memory.
        Accumulates daily data and saves to a single monthly file.

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
        output_dir : str or Path, optional
            Directory to save the monthly prediction dataset
        days_per_chunk : int, optional
            Number of days to process at once (default 10)

        Returns:
        --------
        str
            Path to saved prediction dataset file
        """
        import calendar
        import gc

        print(f"\n{'=' * 60}")
        print(f"PROCESSING MONTH (CHUNKED): {year}-{month:02d}")
        print(f"Time scale: {self.time_scale}")
        print(f"Days per chunk: {days_per_chunk}")
        print(f"{'=' * 60}\n")

        # Get number of days in month
        days_in_month = calendar.monthrange(year, month)[1]
        print(f"Total days in month: {days_in_month}")

        # Create output directory
        if output_dir is None:
            output_dir = config.PREDICTION_DIR / f"{year}_{month:02d}_monthly"

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Fix 2.1: Month-level checkpoint — skip if monthly file already exists
        monthly_output = output_dir / f"prediction_{year}_{month:02d}_daily.csv"
        if monthly_output.exists() and monthly_output.stat().st_size > 1000:
            print(f"Month {year}-{month:02d} already processed ({monthly_output.name}), skipping")
            return str(monthly_output)

        # Temporary directory for chunk files
        temp_chunk_dir = self.temp_dir / f"month_{year}_{month:02d}_chunks"
        temp_chunk_dir.mkdir(exist_ok=True, parents=True)

        # ============================================================
        # PROCESS IN CHUNKS
        # ============================================================
        chunk_files = []
        chunk_dataframes = []

        for chunk_start_day in range(1, days_in_month + 1, days_per_chunk):
            chunk_end_day = min(chunk_start_day + days_per_chunk - 1, days_in_month)

            print(f"\n--- Processing chunk: days {chunk_start_day} to {chunk_end_day} ---")

            try:
                # Process each day in the chunk
                chunk_dfs = []

                for day in range(chunk_start_day, chunk_end_day + 1):
                    print(f"\n  Processing day {day}...")

                    # Clear previous data
                    self._clear_datasets()
                    gc.collect()

                    # Load data for this specific day
                    self.load_variable_data(variables, year, month, day, shapefile=shapefile)

                    # Apply zero mask
                    self.mask_zero_values(threshold=1e-10)

                    # Calculate derived variables
                    self.calculate_derived_variables()

                    # Create prediction dataset for this day
                    start_date = pd.Timestamp(year=year, month=month, day=day, tz="UTC")
                    end_date = start_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

                    df = self.create_prediction_dataset(
                        start_date=start_date, end_date=end_date, use_existing_data=True
                    )

                    if df is not None and not df.empty:
                        chunk_dfs.append(df)
                        print(f"    Day {day}: {len(df)} rows")
                    else:
                        print(f"    Day {day}: No data")

                # Combine chunk dataframes
                if chunk_dfs:
                    chunk_df = pd.concat(chunk_dfs, ignore_index=True)

                    # Option A: Keep in memory (for smaller datasets)
                    chunk_dataframes.append(chunk_df)

                    # Option B: Save to temp file (for larger datasets)
                    # chunk_file = temp_chunk_dir / f"chunk_{chunk_start_day:02d}_{chunk_end_day:02d}.csv"
                    # chunk_df.to_csv(chunk_file, index=False)
                    # chunk_files.append(chunk_file)

                    print(f"  Chunk complete: {len(chunk_df)} total rows")

                    # Clean up chunk_dfs
                    del chunk_dfs
                    gc.collect()

            except Exception as e:
                print(f"  ERROR processing chunk {chunk_start_day}-{chunk_end_day}: {str(e)}")
                traceback.print_exc()

        # ============================================================
        # COMBINE ALL CHUNKS INTO FINAL MONTHLY FILE
        # ============================================================
        print(f"\n{'=' * 60}")
        print("COMBINING CHUNKS INTO MONTHLY FILE")
        print(f"{'=' * 60}")

        try:
            # Option A: Combine from memory
            if chunk_dataframes:
                print(f"Combining {len(chunk_dataframes)} chunks from memory...")
                monthly_df = pd.concat(chunk_dataframes, ignore_index=True)

            # Option B: Combine from temp files
            # elif chunk_files:
            #     print(f"Combining {len(chunk_files)} chunks from temp files...")
            #     chunk_dfs = [pd.read_csv(f) for f in chunk_files]
            #     monthly_df = pd.concat(chunk_dfs, ignore_index=True)

            else:
                print("ERROR: No chunk data available")
                return None

            # Sort by timestamp and location
            sort_cols = []
            if "timestamp" in monthly_df.columns:
                monthly_df["timestamp"] = pd.to_datetime(monthly_df["timestamp"])
                sort_cols.append("timestamp")
            if "name" in monthly_df.columns:
                sort_cols.append("name")
            elif "latitude" in monthly_df.columns and "longitude" in monthly_df.columns:
                sort_cols.extend(["latitude", "longitude"])

            if sort_cols:
                monthly_df = monthly_df.sort_values(sort_cols)

            # Remove duplicates if any
            before_dedup = len(monthly_df)
            monthly_df = monthly_df.drop_duplicates()
            after_dedup = len(monthly_df)
            if before_dedup != after_dedup:
                print(f"Removed {before_dedup - after_dedup} duplicate rows")

            # Save monthly file
            output_file = output_dir / f"prediction_{year}_{month:02d}_daily.csv"
            saved_path = self.save_prediction_dataset(monthly_df, output_file)

            # Print summary
            print(f"\n{'=' * 60}")
            print("SUCCESS: Monthly dataset saved")
            print(f"  File: {output_file}")
            print(f"  Total rows: {len(monthly_df)}")
            print(f"  Columns: {len(monthly_df.columns)}")
            if "timestamp" in monthly_df.columns:
                print(f"  Date range: {monthly_df['timestamp'].min()} to {monthly_df['timestamp'].max()}")
            if "latitude" in monthly_df.columns and "longitude" in monthly_df.columns:
                print(f"  Lat range: {monthly_df['latitude'].min():.4f} to {monthly_df['latitude'].max():.4f}")
                print(f"  Lon range: {monthly_df['longitude'].min():.4f} to {monthly_df['longitude'].max():.4f}")
            print(f"{'=' * 60}\n")

            return saved_path

        except Exception as e:
            print(f"ERROR combining chunks: {str(e)}")
            traceback.print_exc()
            return None

        finally:
            # Clean up
            self._clear_datasets()

            # Clean up temp files
            try:
                for f in chunk_files:
                    if Path(f).exists():
                        Path(f).unlink()
                if temp_chunk_dir.exists():
                    temp_chunk_dir.rmdir()
            except Exception:
                pass

            # Clean up memory
            del chunk_dataframes
            gc.collect()

    def process_year(self, variables, year, shapefile=None, output_dir=None, days_per_chunk=10, save_monthly=True):
        """
        Process an entire year of daily ERA5-Land data.

        Parameters:
        -----------
        variables : list
            List of variables to process
        year : int
            Year to process
        shapefile : str, optional
            Path to shapefile for region definition
        output_dir : str or Path, optional
            Directory to save the prediction datasets
        days_per_chunk : int, optional
            Days per chunk for memory management (default 10)
        save_monthly : bool, optional
            If True, save separate file for each month.
            If False, combine entire year into single file.

        Returns:
        --------
        list or str
            List of monthly file paths, or path to yearly file
        """
        import gc

        print(f"\n{'=' * 70}")
        print(f"PROCESSING YEAR: {year}")
        print(f"Time scale: {self.time_scale}")
        print(f"Save monthly: {save_monthly}")
        print(f"{'=' * 70}\n")

        # Setup output directory
        if output_dir is None:
            output_dir = config.PREDICTION_DIR / f"{year}_{self.time_scale}"
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        monthly_files = []
        monthly_dataframes = []

        # Process each month
        for month in range(1, 13):
            print(f"\n{'=' * 50}")
            print(f"Processing {year}-{month:02d} ({month}/12)")
            print(f"{'=' * 50}")

            try:
                if save_monthly:
                    # Save each month separately
                    result = self.process_month_daily_chunked(
                        variables,
                        year,
                        month,
                        shapefile=shapefile,
                        output_dir=output_dir,
                        days_per_chunk=days_per_chunk,
                    )
                    if result:
                        monthly_files.append(result)
                        print(f"✓ Month {month:02d} saved: {result}")
                    else:
                        print(f"✗ Month {month:02d} failed")
                else:
                    # Accumulate for yearly file
                    month_df = self._process_month_to_dataframe(
                        variables, year, month, shapefile=shapefile, days_per_chunk=days_per_chunk
                    )
                    if month_df is not None and not month_df.empty:
                        monthly_dataframes.append(month_df)
                        print(f"✓ Month {month:02d} loaded: {len(month_df)} rows")
                    else:
                        print(f"✗ Month {month:02d} no data")

                # Force garbage collection between months
                gc.collect()

            except Exception as e:
                print(f"ERROR processing {year}-{month:02d}: {e}")
                traceback.print_exc()

        # Handle yearly file output
        if not save_monthly and monthly_dataframes:
            print(f"\n{'=' * 50}")
            print(f"Combining {len(monthly_dataframes)} months into yearly file...")
            print(f"{'=' * 50}")

            try:
                yearly_df = pd.concat(monthly_dataframes, ignore_index=True)

                # Sort
                sort_cols = []
                if "timestamp" in yearly_df.columns:
                    yearly_df["timestamp"] = pd.to_datetime(yearly_df["timestamp"])
                    sort_cols.append("timestamp")
                if "name" in yearly_df.columns:
                    sort_cols.append("name")
                if sort_cols:
                    yearly_df = yearly_df.sort_values(sort_cols)

                # Remove duplicates
                yearly_df = yearly_df.drop_duplicates()

                # Save yearly file
                output_file = output_dir / f"prediction_{year}_{self.time_scale}.csv"
                saved_path = self.save_prediction_dataset(yearly_df, output_file)

                print(f"\n✓ Yearly file saved: {output_file}")
                print(f"  Total rows: {len(yearly_df)}")

                # Clean up
                del monthly_dataframes, yearly_df
                gc.collect()

                return saved_path

            except Exception as e:
                print(f"ERROR combining yearly data: {e}")
                traceback.print_exc()
                return monthly_files if monthly_files else None

        # Summary for monthly files
        if save_monthly:
            print(f"\n{'=' * 70}")
            print(f"YEAR {year} COMPLETE")
            print(f"Successfully processed: {len(monthly_files)}/12 months")
            print(f"Output directory: {output_dir}")
            print(f"{'=' * 70}\n")

            return monthly_files

        return None

    def _process_month_to_dataframe(self, variables, year, month, shapefile=None, days_per_chunk=10):
        """
        Process a month and return as DataFrame (without saving to file).
        Helper method for yearly processing.

        Parameters:
        -----------
        variables : list
            List of variables to process
        year : int
            Year to process
        month : int
            Month to process
        shapefile : str, optional
            Path to shapefile
        days_per_chunk : int, optional
            Days per chunk

        Returns:
        --------
        pandas.DataFrame or None
            Monthly data as DataFrame
        """
        import calendar
        import gc

        days_in_month = calendar.monthrange(year, month)[1]
        chunk_dataframes = []

        for chunk_start_day in range(1, days_in_month + 1, days_per_chunk):
            chunk_end_day = min(chunk_start_day + days_per_chunk - 1, days_in_month)

            try:
                for day in range(chunk_start_day, chunk_end_day + 1):
                    self._clear_datasets()
                    gc.collect()

                    self.load_variable_data(variables, year, month, day, shapefile=shapefile)
                    self.mask_zero_values(threshold=1e-10)
                    self.calculate_derived_variables()

                    start_date = pd.Timestamp(year=year, month=month, day=day, tz="UTC")
                    end_date = start_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

                    df = self.create_prediction_dataset(
                        start_date=start_date, end_date=end_date, use_existing_data=True
                    )

                    if df is not None and not df.empty:
                        chunk_dataframes.append(df)

            except Exception as e:
                print(f"    Error processing days {chunk_start_day}-{chunk_end_day}: {e}")

        if chunk_dataframes:
            monthly_df = pd.concat(chunk_dataframes, ignore_index=True)
            del chunk_dataframes
            gc.collect()
            return monthly_df

        return None

    def process_multiple_years(
        self, variables, years, months=None, shapefile=None, output_dir=None, days_per_chunk=10, save_monthly=True
    ):
        """
        Process multiple years of daily ERA5-Land data.

        Parameters:
        -----------
        variables : list
            List of variables to process
        years : list of int
            List of years to process
        months : list of int, optional
            List of months to process (1-12). If None, processes all 12 months.
        shapefile : str, optional
            Path to shapefile for region definition
        output_dir : str or Path, optional
            Base directory to save the prediction datasets
        days_per_chunk : int, optional
            Days per chunk for memory management
        save_monthly : bool, optional
            If True, save separate file for each month

        Returns:
        --------
        dict
            Dictionary mapping year -> list of output files
        """
        print(f"\n{'#' * 70}")
        print(f"BATCH PROCESSING: {len(years)} YEARS")
        print(f"Years: {years}")
        print(f"Months: {months if months else 'All (1-12)'}")
        print(f"Time scale: {self.time_scale}")
        print(f"{'#' * 70}\n")

        # Setup base output directory
        if output_dir is None:
            output_dir = config.PREDICTION_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        # Default to all months if not specified
        if months is None:
            months = list(range(1, 13))

        results = {}
        total_years = len(years)

        for year_idx, year in enumerate(years, 1):
            print(f"\n{'#' * 70}")
            print(f"YEAR {year} ({year_idx}/{total_years})")
            print(f"{'#' * 70}")

            year_output_dir = output_dir / f"{year}_{self.time_scale}"
            year_output_dir.mkdir(exist_ok=True, parents=True)

            year_files = []

            for month in months:
                try:
                    print(f"\n--- Processing {year}-{month:02d} ---")

                    result = self.process_month_daily_chunked(
                        variables,
                        year,
                        month,
                        shapefile=shapefile,
                        output_dir=year_output_dir,
                        days_per_chunk=days_per_chunk,
                    )

                    if result:
                        year_files.append(result)
                        print(f"✓ {year}-{month:02d} complete")
                    else:
                        print(f"✗ {year}-{month:02d} failed")

                except Exception as e:
                    print(f"ERROR {year}-{month:02d}: {e}")
                    traceback.print_exc()

            results[year] = year_files

            # Summary for this year
            print(f"\n--- Year {year} Summary ---")
            print(f"  Months processed: {len(year_files)}/{len(months)}")
            print(f"  Output directory: {year_output_dir}")

        # Final summary
        print(f"\n{'#' * 70}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'#' * 70}")

        total_files = sum(len(files) for files in results.values())
        total_expected = len(years) * len(months)

        print(f"Total files created: {total_files}/{total_expected}")
        print(f"Output base directory: {output_dir}")

        for year, files in results.items():
            print(f"  {year}: {len(files)} files")

        return results


    def _download_tile_geotiff(self, image, region_list, scale, band_name=None, timeout=300):
        """Download a single GEE image tile as GeoTIFF via getDownloadURL.

        Unlike geemap.ee_to_numpy, this guarantees that the returned array
        dimensions match the requested geographic extent because rasterio
        reads the GeoTIFF with its embedded affine transform.

        Parameters
        ----------
        image : ee.Image
        region_list : list
            [lon_min, lat_min, lon_max, lat_max]
        scale : float
            Pixel size in metres (at equator for EPSG:4326).
        band_name : str, optional
            Band to select before download.
        timeout : int
            HTTP timeout in seconds (default 300).

        Returns
        -------
        np.ndarray or None
            2-D array (height, width).  None on failure.
        """
        import tempfile
        import requests
        import rasterio

        tile_img = image.select(band_name) if band_name else image
        region = ee.Geometry.Rectangle(region_list, None, False)

        url = tile_img.getDownloadURL({
            "region": region,
            "scale": scale,
            "format": "GEO_TIFF",
            "crs": "EPSG:4326",
        })

        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name

            with rasterio.open(tmp_path) as src:
                data = src.read(1)
            return data.astype(np.float32)
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def _download_gee_image_tiled(self, image, region, scale, band_name=None, max_tile_size=1024):
        """
        Download a large GEE image by splitting it into smaller tiles.
        Robust to 1-pixel edge cases and geometry types.

        Parameters:
        -----------
        image : ee.Image
            The image to download
        region : ee.Geometry
            Region to download
        scale : float
            Scale in meters
        band_name : str, optional
            Band to select (if None, uses first band)
        max_tile_size : int
            Maximum tile size in pixels (default 1024)

        Returns:
        --------
        tuple
            (numpy_array, bounds_dict) where bounds_dict has lon_min, lon_max, lat_min, lat_max
        """
        import math

        # Constant for degree-to-meter conversion at equator (WGS84)
        METERS_PER_DEGREE = 111320.0

        try:
            # --- 1. Robust Bounds Parsing ---
            bounds_info = region.bounds().getInfo()
            if "coordinates" in bounds_info:
                coords = bounds_info["coordinates"][0]
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                lon_min, lon_max = min(lons), max(lons)
                lat_min, lat_max = min(lats), max(lats)
            elif "bbox" in bounds_info:
                lon_min, lat_min, lon_max, lat_max = bounds_info["bbox"]
            else:
                lon_min, lat_min, lon_max, lat_max = bounds_info[0], bounds_info[1], bounds_info[2], bounds_info[3]

            print(f"  Region bounds: lon=[{lon_min:.4f}, {lon_max:.4f}], lat=[{lat_min:.4f}, {lat_max:.4f}]")

            # Safeguard: reject bounds outside valid geographic range
            if lon_min > 180 or lon_max > 180 or lon_min < -180 or lon_max < -180:
                raise ValueError(
                    f"Region bounds have invalid longitudes: [{lon_min}, {lon_max}]. "
                    f"This usually means the input shapefile extends past -180/180. "
                    f"Clamp your shapefile bounds to [-180, 180] before passing to GEE."
                )
            if lat_min > 90 or lat_max > 90 or lat_min < -90 or lat_max < -90:
                raise ValueError(
                    f"Region bounds have invalid latitudes: [{lat_min}, {lat_max}]. "
                    f"Clamp your shapefile bounds to [-90, 90] before passing to GEE."
                )

            # --- 2. Resolution Setup (Correct for EPSG:4326) ---
            deg_per_pixel = scale / METERS_PER_DEGREE

            # --- 3. Array Allocation (Use CEIL for safety) ---
            total_width = int(math.ceil((lon_max - lon_min) / deg_per_pixel))
            total_height = int(math.ceil((lat_max - lat_min) / deg_per_pixel))

            print(f"  Master Grid: {total_height} x {total_width} pixels")
            print(f"  Resolution: {deg_per_pixel:.6f} deg/pixel ({scale}m at equator)")

            # --- 4. Small Region Bypass (Optimization) ---
            if total_height <= max_tile_size and total_width <= max_tile_size:
                print("  Small region - downloading directly via GeoTIFF...")
                try:
                    result = self._download_tile_geotiff(
                        image, [lon_min, lat_min, lon_max, lat_max], scale, band_name=band_name
                    )
                except Exception as e:
                    print(f"  GeoTIFF download failed ({e}), falling back to geemap")
                    tile_img = image.select(band_name) if band_name else image
                    bbox_region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max], None, False)
                    result = geemap.ee_to_numpy(tile_img, region=bbox_region, scale=scale)

                if result is not None:
                    if result.ndim > 2:
                        result = result.squeeze()
                    if result.ndim == 0:
                        result = np.array([[result]])
                    elif result.ndim == 1:
                        if total_height == 1:
                            result = result.reshape(1, -1)
                        else:
                            result = result.reshape(-1, 1)
                    print(f"  Download complete: {result.shape}")

                return result, {"lon_min": lon_min, "lon_max": lon_max, "lat_min": lat_min, "lat_max": lat_max}

            # --- 5. Tiling Setup ---
            output_array = np.full((total_height, total_width), np.nan, dtype=np.float32)

            n_tiles_y = int(math.ceil(total_height / max_tile_size))
            n_tiles_x = int(math.ceil(total_width / max_tile_size))
            total_tiles = n_tiles_y * n_tiles_x
            tile_count = 0
            successful_tiles = 0

            print(f"  Downloading in {n_tiles_y} x {n_tiles_x} tiles ({total_tiles} total)...")

            # --- 6. Tiling Loop ---
            for row_start in range(0, total_height, max_tile_size):
                for col_start in range(0, total_width, max_tile_size):
                    tile_count += 1

                    row_end = min(row_start + max_tile_size, total_height)
                    col_end = min(col_start + max_tile_size, total_width)

                    # Calculate geographic bounds for this tile
                    tile_lon_min = lon_min + (col_start * deg_per_pixel)
                    tile_lon_max = lon_min + (col_end * deg_per_pixel)
                    tile_lat_max = lat_max - (row_start * deg_per_pixel)
                    tile_lat_min = lat_max - (row_end * deg_per_pixel)

                    tile_region = ee.Geometry.Rectangle(
                        [tile_lon_min, tile_lat_min, tile_lon_max, tile_lat_max], None, False
                    )

                    try:
                        tile_data = self._download_tile_geotiff(
                            image,
                            [tile_lon_min, tile_lat_min, tile_lon_max, tile_lat_max],
                            scale,
                            band_name=band_name,
                        )

                        if tile_data is None:
                            continue

                        # --- 7. Robust Shape Handling ---
                        if tile_data.ndim > 2:
                            tile_data = tile_data.squeeze()
                        if tile_data.ndim == 0:
                            tile_data = np.array([[tile_data]])
                        elif tile_data.ndim == 1:
                            expected_h = row_end - row_start
                            expected_w = col_end - col_start
                            if expected_h == 1:
                                tile_data = tile_data.reshape(1, -1)
                            else:
                                tile_data = tile_data.reshape(-1, 1)

                        actual_h, actual_w = tile_data.shape

                        # Calculate safe insertion range (protects against ±1 pixel GEE rounding)
                        ins_h = min(actual_h, total_height - row_start)
                        ins_w = min(actual_w, total_width - col_start)

                        output_array[row_start : row_start + ins_h, col_start : col_start + ins_w] = tile_data[
                            :ins_h, :ins_w
                        ]
                        successful_tiles += 1

                    except Exception as tile_err:
                        print(f"    Skip Tile [{row_start}, {col_start}]: {tile_err}")

                    # Progress logging
                    if tile_count % 10 == 0 or tile_count == total_tiles:
                        print(f"    Processed {tile_count}/{total_tiles} tiles ({successful_tiles} successful)")

            # --- 8. Final Summary ---
            valid_pixels = np.sum(~np.isnan(output_array))
            total_pixels = output_array.size
            coverage = (valid_pixels / total_pixels) * 100

            print(f"  Tiled download complete: {output_array.shape}")
            print(f"  Coverage: {valid_pixels}/{total_pixels} pixels ({coverage:.1f}%)")

            if successful_tiles == 0:
                print("  WARNING: No tiles were successfully downloaded!")
                return None, None

            return output_array, {"lon_min": lon_min, "lon_max": lon_max, "lat_min": lat_min, "lat_max": lat_max}

        except Exception as e:
            print(f"Tiling failure: {e}")
            traceback.print_exc()
            return None, None

    def read_shapefile(self, shapefile_path, simplify=True, use_bounds=False):
        """
        Read a shapefile and convert it to an ee.Geometry object.

        Parameters:
        -----------
        shapefile_path : str or Path
            Path to the shapefile
        simplify : bool
            Whether to simplify the geometry (reduce vertices)
        use_bounds : bool
            If True, just use the bounding box instead of the actual geometry

        Returns:
        --------
        ee.Geometry
            Earth Engine geometry object
        """
        try:
            gdf = gpd.read_file(shapefile_path)

            # Check CRS
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
                print("Reprojected shapefile to EPSG:4326")

            n_features = len(gdf)
            print(f"Shapefile contains {n_features} features.")

            # For very complex shapefiles, use bounding box
            if use_bounds or n_features > 1000:
                print(f"Using bounding box due to complexity ({n_features} features)")
                bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
                if bounds[0] < -180 or bounds[2] > 180 or bounds[1] < -90 or bounds[3] > 90:
                    raise ValueError(
                        f"Shapefile bounds [{bounds[0]:.4f}, {bounds[1]:.4f}, "
                        f"{bounds[2]:.4f}, {bounds[3]:.4f}] exceed [-180, -90, 180, 90]. "
                        f"Clamp your shapefile to valid geographic bounds before use."
                    )
                ee_geometry = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]], None, False)
                return ee_geometry

            # Union all features
            if n_features > 1:
                print(f"Merging {n_features} features into single geometry...")
                geometry = gdf.unary_union
            else:
                geometry = gdf.geometry.iloc[0]

            # Simplify if requested
            if simplify:
                # Simplify with tolerance (in degrees, ~1km)
                original_vertices = self._count_vertices(geometry)
                tolerance = 0.01  # About 1km
                geometry = geometry.simplify(tolerance, preserve_topology=True)
                new_vertices = self._count_vertices(geometry)
                print(f"Simplified geometry: {original_vertices} -> {new_vertices} vertices")

                # If still too complex, use bounding box
                if new_vertices > 5000:
                    print("Geometry still too complex, using bounding box")
                    bounds = gdf.total_bounds
                    ee_geometry = ee.Geometry.Rectangle([bounds[0], bounds[1], bounds[2], bounds[3]], None, False)
                    return ee_geometry

            # Convert to GeoJSON
            geo_json = geometry.__geo_interface__

            # Create ee.Geometry
            ee_geometry = ee.Geometry(geo_json)

            print("Successfully converted shapefile to ee.Geometry")
            return ee_geometry

        except Exception as e:
            print(f"Error reading shapefile: {str(e)}")
            traceback.print_exc()
            return None

    def _count_vertices(self, geometry):
        """Count total vertices in a geometry."""
        try:
            if hasattr(geometry, "geoms"):  # MultiPolygon
                return sum(len(g.exterior.coords) + sum(len(r.coords) for r in g.interiors) for g in geometry.geoms)
            elif hasattr(geometry, "exterior"):  # Polygon
                return len(geometry.exterior.coords) + sum(len(r.coords) for r in geometry.interiors)
            else:
                return 0
        except Exception:
            return 0

    def debug_coordinate_alignment(self):
        """Debug coordinate alignment issues by checking specific locations."""

        test_locations = [
            (0, 10),  # Your problematic location
            (10, 10),  # Another test
            (-10, 10),  # Southern hemisphere test
            (0, 0),  # Equator/Prime Meridian
        ]

        for var_name, ds in self.datasets.items():
            print(f"\n=== Debugging {var_name} ===")
            var = list(ds.data_vars)[0]

            # Check coordinate ordering
            lats = ds.latitude.values
            lons = ds.longitude.values

            print(f"Latitude range: {lats.min():.4f} to {lats.max():.4f}")
            print(f"Longitude range: {lons.min():.4f} to {lons.max():.4f}")
            print(f"Latitude ordering: {'descending' if lats[0] > lats[-1] else 'ascending'}")

            # Check specific locations
            for lat, lon in test_locations:
                try:
                    # Find nearest indices
                    lat_idx = np.argmin(np.abs(lats - lat))
                    lon_idx = np.argmin(np.abs(lons - lon))

                    actual_lat = lats[lat_idx]
                    actual_lon = lons[lon_idx]

                    # Extract time series
                    values = ds[var].isel(latitude=lat_idx, longitude=lon_idx).values

                    print(f"\nLocation ({lat}, {lon}):")
                    print(f"  Nearest grid point: ({actual_lat:.4f}, {actual_lon:.4f})")
                    print(f"  Indices: lat_idx={lat_idx}, lon_idx={lon_idx}")
                    print(f"  Value range: {values.min()} to {values.max()}")
                    print(f"  Unique values: {len(np.unique(values))}")

                    # Check for constant values
                    if len(np.unique(values)) <= 3:
                        print("  WARNING: Very few unique values!")
                        print(f"  First 10 values: {values[:10]}")

                        # Check adjacent grid points
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue
                                try:
                                    adj_lat_idx = lat_idx + di
                                    adj_lon_idx = lon_idx + dj
                                    if 0 <= adj_lat_idx < len(lats) and 0 <= adj_lon_idx < len(lons):
                                        adj_values = ds[var].isel(latitude=adj_lat_idx, longitude=adj_lon_idx).values
                                        print(f"  Adjacent ({di}, {dj}): {adj_values[:5]}...")
                                except Exception:
                                    pass

                except Exception as e:
                    print(f"  Error checking location ({lat}, {lon}): {str(e)}")

    def diagnose_coordinate_system(self, variable_name=None):
        """Comprehensive diagnostic tool for coordinate system issues."""

        if variable_name:
            datasets_to_check = {variable_name: self.datasets[variable_name]}
        else:
            datasets_to_check = self.datasets

        for var_name, ds in datasets_to_check.items():
            print(f"\n{'=' * 20} {var_name} {'=' * 20}")

            # Check basic structure
            print(f"Dimensions: {ds.dims}")
            print(f"Coordinates: {list(ds.coords)}")
            print(f"Data variables: {list(ds.data_vars)}")

            # Check coordinate properties
            if "latitude" in ds.coords and "longitude" in ds.coords:
                lats = ds.latitude.values
                lons = ds.longitude.values

                print("\nLatitude properties:")
                print(f"  Range: {lats.min():.4f} to {lats.max():.4f}")
                print(f"  Number of points: {len(lats)}")
                print(f"  Ordering: {'descending' if lats[0] > lats[-1] else 'ascending'}")
                print(f"  Resolution: {np.mean(np.diff(lats)):.6f}")

                print("\nLongitude properties:")
                print(f"  Range: {lons.min():.4f} to {lons.max():.4f}")
                print(f"  Number of points: {len(lons)}")
                print(f"  Resolution: {np.mean(np.diff(lons)):.6f}")

                # Check for problematic regions
                var = list(ds.data_vars)[0]
                data_array = ds[var]

                # Find regions with constant values
                print("\nChecking for constant value regions:")
                for t_idx in range(min(3, len(ds.time))):  # Check first 3 timestamps
                    data_slice = data_array.isel(time=t_idx).values

                    # Check for rows with constant values
                    constant_rows = np.where(np.all(data_slice == data_slice[:, 0:1], axis=1))[0]
                    if len(constant_rows) > 0:
                        print(f"  Timestamp {t_idx}: {len(constant_rows)} rows with constant values")
                        print(f"    Row indices: {constant_rows[:10]}...")
                        print(f"    Corresponding latitudes: {lats[constant_rows[:10]]}")

                    # Check for columns with constant values
                    constant_cols = np.where(np.all(data_slice == data_slice[0:1, :], axis=0))[0]
                    if len(constant_cols) > 0:
                        print(f"  Timestamp {t_idx}: {len(constant_cols)} columns with constant values")
                        print(f"    Column indices: {constant_cols[:10]}...")
                        print(f"    Corresponding longitudes: {lons[constant_cols[:10]]}")

                # Test coordinate mapping - use center of actual data region
                # Define region bounds if not already available
                lat_min = config.DEFAULT_LAT_MIN if not hasattr(config, "DEFAULT_LAT_MIN") else config.DEFAULT_LAT_MIN
                lat_max = config.DEFAULT_LAT_MAX if not hasattr(config, "DEFAULT_LAT_MAX") else config.DEFAULT_LAT_MAX
                lon_min = config.DEFAULT_LON_MIN if not hasattr(config, "DEFAULT_LON_MIN") else config.DEFAULT_LON_MIN
                lon_max = config.DEFAULT_LON_MAX if not hasattr(config, "DEFAULT_LON_MAX") else config.DEFAULT_LON_MAX

                test_lat = (lat_min + lat_max) / 2
                test_lon = (lon_min + lon_max) / 2
                lat_idx = np.argmin(np.abs(lats - test_lat))
                lon_idx = np.argmin(np.abs(lons - test_lon))
                print(
                    f"Test location ({test_lat:.2f}, {test_lon:.2f}) maps to indices: lat_idx={lat_idx}, lon_idx={lon_idx}"
                )
                print(f"Actual coordinates at indices: ({lats[lat_idx]:.4f}, {lons[lon_idx]:.4f})")

                print(f"  Actual coordinates: ({lats[lat_idx]:.4f}, {lons[lon_idx]:.4f})")

                # Check values at this location
                values = data_array.isel(latitude=lat_idx, longitude=lon_idx).values
                print(f"  Value range: {values.min()} to {values.max()}")
                print(f"  Unique values: {len(np.unique(values))}")
                if len(np.unique(values)) <= 5:
                    print(f"  All values: {values}")

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
        units_map = {
            "temperature_2m": "K",
            "dewpoint_temperature_2m": "K",
            "total_precipitation_hourly": "m",
            "surface_solar_radiation_downwards": "J m-2",
            "u_component_of_wind_10m": "m s-1",
            "v_component_of_wind_10m": "m s-1",
            "surface_pressure": "Pa",
            "total_evaporation": "m of water equivalent",
            "potential_evaporation": "m",
            "volumetric_soil_water_layer_1": "Volume fraction (m3 m-3)",
            "volumetric_soil_water_layer_2": "Volume fraction (m3 m-3)",
            "volumetric_soil_water_layer_3": "Volume fraction (m3 m-3)",
            "volumetric_soil_water_layer_4": "Volume fraction (m3 m-3)",
            "soil_temperature_level_1": "K",
            "soil_temperature_level_2": "K",
            "soil_temperature_level_3": "K",
            "soil_temperature_level_4": "K",
        }
        return units_map.get(gee_var, "unknown")

    def _align_timezone_for_slice(self, ds, time_var, start_date, end_date, use_utc=True):
        """
        Ensure dataset time coordinate and slice bounds have matching timezone awareness.

        Parameters:
        -----------
        ds : xarray.Dataset
            Dataset with time coordinate
        time_var : str
            Name of the time coordinate variable
        start_date : pd.Timestamp or None
            Start date for slicing
        end_date : pd.Timestamp or None
            End date for slicing
        use_utc : bool
            If True, standardize to UTC-aware; if False, standardize to tz-naive

        Returns:
        --------
        tuple
            (modified_ds, aligned_start, aligned_end)
        """
        import pandas as pd

        # Get the dataset's time coordinate
        ds_times = pd.DatetimeIndex(ds[time_var].values)

        # Determine current timezone state of dataset
        ds_is_tz_aware = ds_times.tz is not None

        # Determine timezone state of slice bounds
        start_is_tz_aware = start_date is not None and start_date.tzinfo is not None
        end_is_tz_aware = end_date is not None and end_date.tzinfo is not None

        # Decide target timezone state
        if use_utc:
            # Target: everything UTC-aware
            target_tz = "UTC"

            # Align dataset times
            if not ds_is_tz_aware:
                # Assume naive times are UTC, localize them
                ds_times_aligned = ds_times.tz_localize("UTC")
                ds = ds.assign_coords({time_var: ds_times_aligned})
            elif str(ds_times.tz) != "UTC":
                # Convert to UTC
                ds_times_aligned = ds_times.tz_convert("UTC")
                ds = ds.assign_coords({time_var: ds_times_aligned})

            # Align start_date
            if start_date is not None:
                if not start_is_tz_aware:
                    start_date = pd.Timestamp(start_date).tz_localize("UTC")
                else:
                    start_date = pd.Timestamp(start_date).tz_convert("UTC")

            # Align end_date
            if end_date is not None:
                if not end_is_tz_aware:
                    end_date = pd.Timestamp(end_date).tz_localize("UTC")
                else:
                    end_date = pd.Timestamp(end_date).tz_convert("UTC")

        else:
            # Target: everything tz-naive

            # Align dataset times
            if ds_is_tz_aware:
                # Convert to UTC first (if not already), then remove timezone
                if str(ds_times.tz) != "UTC":
                    ds_times = ds_times.tz_convert("UTC")
                ds_times_aligned = ds_times.tz_localize(None)
                ds = ds.assign_coords({time_var: ds_times_aligned})

            # Align start_date
            if start_date is not None and start_is_tz_aware:
                start_date = pd.Timestamp(start_date).tz_convert("UTC").tz_localize(None)

            # Align end_date
            if end_date is not None and end_is_tz_aware:
                end_date = pd.Timestamp(end_date).tz_convert("UTC").tz_localize(None)

        return ds, start_date, end_date

    def _safe_time_slice(self, ds, time_var, start_date, end_date, use_utc=True, dates_already_standardized=False):
        """
        Safely slice a dataset by time, handling timezone mismatches.
        """
        import pandas as pd

        if start_date is None and end_date is None:
            return ds

        # Convert to pandas Timestamps if not already
        if start_date is not None and not isinstance(start_date, pd.Timestamp):
            start_date = pd.Timestamp(start_date)
        if end_date is not None and not isinstance(end_date, pd.Timestamp):
            end_date = pd.Timestamp(end_date)

        # --- Standardize dataset time coordinate (always needed) ---
        ds_times_pd = pd.DatetimeIndex(ds[time_var].values)
        ds_times_std = self.standardize_timestamps(ds_times_pd, use_utc=use_utc)
        if ds_times_std is None or ds_times_std.empty:
            print("Warning: Time standardization failed for dataset.")
            return ds
        ds = ds.assign_coords({time_var: ds_times_std})

        # --- Standardize slice bounds (skip if already done) ---
        if not dates_already_standardized:
            if start_date is not None:
                start_idx = pd.DatetimeIndex([start_date])
                start_date = self.standardize_timestamps(start_idx, use_utc=use_utc)[0]

            if end_date is not None:
                end_idx = pd.DatetimeIndex([end_date])
                end_date = self.standardize_timestamps(end_idx, use_utc=use_utc)[0]

        # Create and apply the slice
        time_slice = slice(start_date, end_date)

        try:
            ds_filtered = ds.sel({time_var: time_slice})

            if ds_filtered[time_var].size == 0:
                print("Warning: Time slice returned empty dataset.")
                print(f"  Dataset time range: {ds[time_var].values.min()} to {ds[time_var].values.max()}")
                print(f"  Requested slice: {start_date} to {end_date}")

            return ds_filtered

        except Exception as e:
            print(f"Error during time slicing: {e}")
            return self._fallback_time_slice(ds, time_var, start_date, end_date)

    def _fallback_time_slice(self, ds, time_var, start_date, end_date):
        """Fallback time slicing using boolean indexing."""
        import pandas as pd

        ds_times = pd.DatetimeIndex(ds[time_var].values)
        mask = np.ones(len(ds_times), dtype=bool)

        if start_date is not None:
            mask = mask & (ds_times >= start_date)
        if end_date is not None:
            mask = mask & (ds_times <= end_date)

        return ds.isel({time_var: mask})

    def get_elevation_from_gee(self, region=None, shapefile=None, scale=None):
        """
        Get elevation data from ASTER DEM via Google Earth Engine.
        Uses same storage format as climate data for consistent lookup.
        """
        if not self.ee_initialized:
            self.initialize_gee()
            if not self.ee_initialized:
                print("Google Earth Engine not initialized. Cannot access elevation data.")
                return None

        print("Accessing ASTER DEM elevation data from Google Earth Engine...")

        try:
            # Get bounds from shapefile or defaults
            if shapefile is not None:
                gdf = gpd.read_file(shapefile)
                if gdf.crs != "EPSG:4326":
                    gdf = gdf.to_crs("EPSG:4326")
                bounds = gdf.total_bounds
                lon_min, lat_min, lon_max, lat_max = bounds
                print(f"  Shapefile bounds: lon=[{lon_min:.2f}, {lon_max:.2f}], lat=[{lat_min:.2f}, {lat_max:.2f}]")
            else:
                lon_min = config.DEFAULT_LON_MIN
                lon_max = config.DEFAULT_LON_MAX
                lat_min = config.DEFAULT_LAT_MIN
                lat_max = config.DEFAULT_LAT_MAX
                print(f"  Default bounds: lon=[{lon_min:.2f}, {lon_max:.2f}], lat=[{lat_min:.2f}, {lat_max:.2f}]")

            region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max], None, False)

            if scale is None:
                scale = 11132  # ~0.1 degree, match ERA5-Land resolution

            # Access ASTER DEM
            aster_dem = ee.Image("NASA/NASADEM_HGT/001").select("elevation")

            # Use tiled download
            elev_array, bounds_returned = self._download_gee_image_tiled(
                aster_dem, region, scale, band_name="elevation", max_tile_size=1024
            )
            elev_array = elev_array.astype(float)
            # Filter NoData values
            elev_array = np.where(elev_array <= -32000, np.nan, elev_array)  # -32768 is NoData
            elev_array = np.where(elev_array > 9000, np.nan, elev_array)  # >9000m is invalid (Everest is ~8849m)
            if elev_array is None:
                print("Failed to download elevation data")
                return None

            height, width = elev_array.shape

            # Create transform EXACTLY like climate data uses
            # from_bounds(west, south, east, north, width, height)
            from rasterio.transform import from_bounds

            transform = from_bounds(
                bounds_returned["lon_min"],
                bounds_returned["lat_min"],
                bounds_returned["lon_max"],
                bounds_returned["lat_max"],
                width,
                height,
            )

            # Store in SAME FORMAT as climate data
            self.elevation_raw = {
                "data": elev_array.astype(float),
                "transform": transform,
                "crs": "EPSG:4326",
                "height": height,
                "width": width,
                "nodata": None,
            }

            # Validation output
            valid_data = elev_array[~np.isnan(elev_array)]
            print("Successfully loaded raw elevation data:")
            print(f"  Shape: {height} x {width}")
            print(f"  Range: [{np.nanmin(valid_data):.1f}, {np.nanmax(valid_data):.1f}] m")
            print(f"  Unique values: {len(np.unique(valid_data))}")

            return self.elevation_raw

        except Exception as e:
            print(f"Error accessing elevation data from GEE: {str(e)}")
            traceback.print_exc()
            return None

    def get_pft_from_gee(self, year, region=None, shapefile=None, scale=None):
        """
        Get Plant Functional Type (PFT) data from MODIS Land Cover via Google Earth Engine.
        Uses same storage format as climate data for consistent lookup.
        """
        if not self.ee_initialized:
            self.initialize_gee()
            if not self.ee_initialized:
                print("Google Earth Engine not initialized. Cannot access PFT data.")
                return None

        print(f"Accessing MODIS Land Cover (PFT) data for year {year} from Google Earth Engine...")

        try:
            # Get bounds from shapefile or defaults
            if shapefile is not None:
                gdf = gpd.read_file(shapefile)
                if gdf.crs != "EPSG:4326":
                    gdf = gdf.to_crs("EPSG:4326")
                bounds = gdf.total_bounds
                lon_min, lat_min, lon_max, lat_max = bounds
                print(f"  Shapefile bounds: lon=[{lon_min:.2f}, {lon_max:.2f}], lat=[{lat_min:.2f}, {lat_max:.2f}]")
            else:
                lon_min = config.DEFAULT_LON_MIN
                lon_max = config.DEFAULT_LON_MAX
                lat_min = config.DEFAULT_LAT_MIN
                lat_max = config.DEFAULT_LAT_MAX
                print(f"  Default bounds: lon=[{lon_min:.2f}, {lon_max:.2f}], lat=[{lat_min:.2f}, {lat_max:.2f}]")

            region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max], None, False)

            if scale is None:
                scale = 11132

            # Access MODIS Land Cover
            modis_lc = (
                ee.ImageCollection("MODIS/061/MCD12Q1")
                .filterDate(f"{year}-01-01", f"{year}-12-31")
                .first()
                .select("LC_Type1")
            )
            # .resample('mode')

            # Use tiled download
            pft_array, bounds_returned = self._download_gee_image_tiled(
                modis_lc, region, scale, band_name="LC_Type1", max_tile_size=1024
            )

            if pft_array is None:
                print("Failed to download PFT data")
                return None

            height, width = pft_array.shape

            # Create transform EXACTLY like climate data uses
            from rasterio.transform import from_bounds

            transform = from_bounds(
                bounds_returned["lon_min"],
                bounds_returned["lat_min"],
                bounds_returned["lon_max"],
                bounds_returned["lat_max"],
                width,
                height,
            )

            # Store in SAME FORMAT as climate data
            self.pft_raw = {
                "data": pft_array.astype(float),
                "transform": transform,
                "crs": "EPSG:4326",
                "height": height,
                "width": width,
                "nodata": 255,
                "year": year,
                "class_descriptions": {
                    1: "ENF",
                    2: "EBF",
                    3: "DNF",
                    4: "DBF",
                    5: "MF",
                    6: "CSH",
                    7: "OSH",
                    8: "WSA",
                    9: "SAV",
                    10: "GRA",
                    11: "WET",
                    12: "CRO",
                    13: "URB",
                    14: "CNV",
                    15: "SNO",
                    16: "BSV",
                    17: "WAT",
                },
            }

            # Validation output
            valid_data = pft_array[~np.isnan(pft_array) & (pft_array != 255)]
            unique_classes = np.unique(valid_data).astype(int)
            print("Successfully loaded raw PFT data:")
            print(f"  Shape: {height} x {width}")
            print(f"  PFT classes present: {unique_classes.tolist()}")
            print(f"  Unique values: {len(unique_classes)}")

            return self.pft_raw

        except Exception as e:
            print(f"Error accessing PFT data from GEE: {str(e)}")
            traceback.print_exc()
            return None

    def get_canopy_height_from_gee(self, region=None, shapefile=None, scale=None):
        """
        Get canopy height data from Meta/Facebook dataset via Google Earth Engine.
        Uses same storage format as climate data for consistent lookup.
        """
        if not self.ee_initialized:
            self.initialize_gee()
            if not self.ee_initialized:
                print("Google Earth Engine not initialized. Cannot access canopy height data.")
                return None

        print("Accessing Meta canopy height data from Google Earth Engine...")

        try:
            # Get bounds from shapefile or defaults
            if shapefile is not None:
                gdf = gpd.read_file(shapefile)
                if gdf.crs != "EPSG:4326":
                    gdf = gdf.to_crs("EPSG:4326")
                bounds = gdf.total_bounds
                lon_min, lat_min, lon_max, lat_max = bounds
                print(f"  Shapefile bounds: lon=[{lon_min:.2f}, {lon_max:.2f}], lat=[{lat_min:.2f}, {lat_max:.2f}]")
            else:
                lon_min = config.DEFAULT_LON_MIN
                lon_max = config.DEFAULT_LON_MAX
                lat_min = config.DEFAULT_LAT_MIN
                lat_max = config.DEFAULT_LAT_MAX
                print(f"  Default bounds: lon=[{lon_min:.2f}, {lon_max:.2f}], lat=[{lat_min:.2f}, {lat_max:.2f}]")

            region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max], None, False)

            if scale is None:
                scale = 11132

            # Try different dataset paths
            dataset_paths = [
                ("ImageCollection", "projects/meta-forest-monitoring-okw37/assets/CanopyHeight"),
                ("Image", "projects/sat-io/open-datasets/META/GLOBCANOPYHEIGHT"),
            ]

            canopy_height = None
            successful_path = None

            for ds_type, ds_path in dataset_paths:
                try:
                    print(f"  Trying {ds_type}: {ds_path}")
                    if ds_type == "ImageCollection":
                        canopy_height = ee.ImageCollection(ds_path).mosaic()
                    else:
                        canopy_height = ee.Image(ds_path)

                    _ = canopy_height.bandNames().getInfo()
                    successful_path = ds_path
                    print(f"  Successfully connected to: {ds_path}")
                    break
                except Exception as e:
                    print(f"  Failed: {str(e)[:50]}...")
                    canopy_height = None

            if canopy_height is None:
                print("Error: Could not access any canopy height dataset")
                return None

            # Select band
            band_names = canopy_height.bandNames().getInfo()
            if "canopy_height" in band_names:
                band_name = "canopy_height"
            elif "b1" in band_names:
                band_name = "b1"
            else:
                band_name = band_names[0]

            # Use tiled download
            ch_array, bounds_returned = self._download_gee_image_tiled(
                canopy_height, region, scale, band_name=band_name, max_tile_size=1024
            )

            if ch_array is None:
                print("Failed to download canopy height data")
                return None

            # Handle invalid values
            ch_array = ch_array.astype(float)
            ch_array = np.where((ch_array < 0) | (ch_array > 100), np.nan, ch_array)

            height, width = ch_array.shape

            # Create transform EXACTLY like climate data uses
            from rasterio.transform import from_bounds

            transform = from_bounds(
                bounds_returned["lon_min"],
                bounds_returned["lat_min"],
                bounds_returned["lon_max"],
                bounds_returned["lat_max"],
                width,
                height,
            )

            # Store in SAME FORMAT as climate data
            self.canopy_height_raw = {
                "data": ch_array,
                "transform": transform,
                "crs": "EPSG:4326",
                "height": height,
                "width": width,
                "nodata": np.nan,
                "source": successful_path,
            }

            # Validation output
            valid_data = ch_array[~np.isnan(ch_array)]
            print("Successfully loaded raw canopy height data:")
            print(f"  Shape: {height} x {width}")
            if len(valid_data) > 0:
                print(f"  Range: [{valid_data.min():.1f}, {valid_data.max():.1f}] m")
                print(f"  Unique values: {len(np.unique(valid_data))}")
            else:
                print("  WARNING: No valid data points!")

            return self.canopy_height_raw

        except Exception as e:
            print(f"Error accessing canopy height data from GEE: {str(e)}")
            traceback.print_exc()
            return None

    def load_lai_data_raw(self, year, month, day=None, region=None, shapefile=None):
        """
        Load LAI data from local GLOBMAP files as raw data.
        Uses same storage format as climate data for consistent lookup.

        Parameters:
        -----------
        year : int
            Year of data
        month : int
            Month of data
        day : int, optional
            Day of data. If None, loads data for entire month
        region : dict, optional
            Region bounds
        shapefile : str, optional
            Path to shapefile for region definition

        Returns:
        --------
        dict
            Raw LAI data dictionary with 'data', 'times', and 'transform'
        """
        if day is not None:
            print(f"Loading raw GLOBMAP LAI data for {year}-{month:02d}-{day:02d}")
            start_date = datetime(year, month, day)
            end_date = start_date + timedelta(days=1)
        else:
            print(f"Loading raw GLOBMAP LAI data for entire month {year}-{month:02d}")
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1)
            else:
                end_date = datetime(year, month + 1, 1)

        try:
            # Get region bounds
            if shapefile:
                gdf = gpd.read_file(shapefile)
                if gdf.crs != "EPSG:4326":
                    gdf = gdf.to_crs("EPSG:4326")
                bounds = gdf.total_bounds
                lon_min, lat_min, lon_max, lat_max = bounds
                print(f"  Shapefile bounds: lon=[{lon_min:.2f}, {lon_max:.2f}], lat=[{lat_min:.2f}, {lat_max:.2f}]")
            elif region:
                lon_min = region.get("lon_min", config.DEFAULT_LON_MIN)
                lon_max = region.get("lon_max", config.DEFAULT_LON_MAX)
                lat_min = region.get("lat_min", config.DEFAULT_LAT_MIN)
                lat_max = region.get("lat_max", config.DEFAULT_LAT_MAX)
            else:
                lon_min = config.DEFAULT_LON_MIN
                lon_max = config.DEFAULT_LON_MAX
                lat_min = config.DEFAULT_LAT_MIN
                lat_max = config.DEFAULT_LAT_MAX

            # Find LAI files
            lai_files = list(glob.glob(str(self.lai_data_dir / "GlobMapLAIV3.A*.Global.LAI.tif")))
            if not lai_files:
                lai_files = list(glob.glob(str(self.lai_data_dir / "*LAI*.tif")))

            if not lai_files:
                print(f"Error: No LAI files found in {self.lai_data_dir}")
                self.lai_raw = None
                return None

            print(f"  Found {len(lai_files)} LAI files in {self.lai_data_dir}")

            # Parse filenames
            def parse_globmap_filename(filepath):
                filename = Path(filepath).name
                try:
                    parts = filename.split(".")
                    for part in parts:
                        if part.startswith("A") and len(part) == 8:
                            file_year = int(part[1:5])
                            file_doy = int(part[5:8])
                            file_date = datetime(file_year, 1, 1) + timedelta(days=file_doy - 1)
                            return file_year, file_doy, file_date
                except (ValueError, IndexError):
                    pass
                return None, None, None

            # Find relevant files with buffer
            buffer_days = 16
            search_start = start_date - timedelta(days=buffer_days)
            search_end = end_date + timedelta(days=buffer_days)

            relevant_files = []
            for lai_file in lai_files:
                file_year, file_doy, file_date = parse_globmap_filename(lai_file)
                if file_date is None:
                    continue
                if search_start <= file_date <= search_end:
                    relevant_files.append({"path": lai_file, "date": file_date})

            relevant_files.sort(key=lambda x: x["date"])

            if not relevant_files:
                # Find nearest file
                target_mid = start_date + (end_date - start_date) / 2
                min_diff = float("inf")
                best_file = None

                for lai_file in lai_files:
                    _, _, file_date = parse_globmap_filename(lai_file)
                    if file_date is None:
                        continue
                    diff = abs((file_date - target_mid).days)
                    if diff < min_diff:
                        min_diff = diff
                        best_file = {"path": lai_file, "date": file_date}

                if best_file:
                    relevant_files = [best_file]
                    print(f"  Using nearest file: {Path(best_file['path']).name} ({min_diff} days from target)")
                else:
                    print("Error: Could not find any suitable LAI files")
                    self.lai_raw = None
                    return None

            print(f"  Loading {len(relevant_files)} LAI files:")

            # Load all files
            lai_arrays = []
            lai_dates = []
            transform_ref = None

            for file_info in relevant_files:
                lai_file = file_info["path"]
                file_date = file_info["date"]

                try:
                    with rasterio.open(lai_file) as src:
                        # Create window from bounds
                        window = from_bounds(lon_min, lat_min, lon_max, lat_max, src.transform)
                        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

                        if window.width <= 0 or window.height <= 0:
                            print(f"    Skipping {Path(lai_file).name}: No overlap with region")
                            continue

                        # Read data within window
                        lai_array = src.read(1, window=window)

                        # Get transform for this window - SAME AS CLIMATE DATA
                        window_transform = rasterio.windows.transform(window, src.transform)

                        if transform_ref is None:
                            transform_ref = window_transform

                        nodata = src.nodata

                        # Handle nodata
                        lai_array = lai_array.astype(float)
                        if nodata is not None:
                            lai_array = np.where(lai_array == nodata, np.nan, lai_array)

                        # Apply scaling if needed
                        valid_data = lai_array[~np.isnan(lai_array)]
                        if len(valid_data) > 0:
                            data_max = np.nanmax(valid_data)
                            if data_max > 700:
                                lai_array = lai_array * 0.01
                            elif data_max > 70:
                                lai_array = lai_array * 0.1

                        # Validate range (LAI typically 0-10)
                        lai_array = np.where((lai_array < 0) | (lai_array > 10), np.nan, lai_array)

                        lai_arrays.append(lai_array)
                        lai_dates.append(file_date)

                        valid_after = lai_array[~np.isnan(lai_array)]
                        print(
                            f"    - {Path(lai_file).name}: shape={lai_array.shape}, "
                            f"valid={len(valid_after)}, range=[{valid_after.min():.2f}, {valid_after.max():.2f}]"
                            if len(valid_after) > 0
                            else "no valid data"
                        )

                except Exception as e:
                    print(f"    Error loading {Path(lai_file).name}: {e}")
                    continue

            if not lai_arrays:
                print("Error: Failed to load any LAI data")
                self.lai_raw = None
                return None

            # Stack into 3D array (n_times, height, width)
            lai_stack = np.stack(lai_arrays, axis=0)

            # Store in SAME FORMAT as climate data
            self.lai_raw = {
                "data": lai_stack,
                "times": lai_dates,
                "transform": transform_ref,
                "crs": "EPSG:4326",
                "height": lai_stack.shape[1],
                "width": lai_stack.shape[2],
                "nodata": np.nan,
                "source": "GLOBMAP LAI V3",
            }

            # Validation output
            valid_lai = lai_stack[~np.isnan(lai_stack)]
            print("Successfully loaded raw LAI data:")
            print(f"  Shape: {lai_stack.shape} (times, lat, lon)")
            print(f"  Time range: {min(lai_dates)} to {max(lai_dates)}")
            if len(valid_lai) > 0:
                print(f"  Value range: [{valid_lai.min():.2f}, {valid_lai.max():.2f}]")
                print(f"  Unique values: {len(np.unique(valid_lai))}")

            return self.lai_raw

        except Exception as e:
            print(f"Error loading LAI data: {str(e)}")
            traceback.print_exc()
            self.lai_raw = None
            return None

    def load_static_datasets(self, year, shapefile=None, region=None):
        """
        Load all static datasets (elevation, PFT, canopy height, LAI).

        Parameters:
        -----------
        year : int
            Year for time-varying static data (PFT, LAI)
        shapefile : str, optional
            Path to shapefile for region definition
        region : dict, optional
            Region bounds

        Returns:
        --------
        dict
            Dictionary of loaded static datasets
        """
        print("\n===== Loading Static Datasets =====")

        static_data = {}

        # Load elevation
        elev = self.get_elevation_from_gee(shapefile=shapefile)
        if elev is not None:
            static_data["elevation"] = elev

        # Load PFT
        pft = self.get_pft_from_gee(year, shapefile=shapefile)
        if pft is not None:
            static_data["pft"] = pft

        # Load canopy height
        ch = self.get_canopy_height_from_gee(shapefile=shapefile)
        if ch is not None:
            static_data["canopy_height"] = ch

        # Load LAI (will need month as well for temporal matching)
        # This is loaded separately per time period in load_variable_data

        print(f"Loaded {len(static_data)} static datasets")
        return static_data

    def get_era5land_variable_from_gee(
        self, variable, year, month=None, day=None, hour=None, scale=None, shapefile=None, region=None
    ):
        """
        Access ERA5-Land data from Google Earth Engine for a specific time period.
        Modified to properly handle region bounds and coordinate extraction.

        Parameters:
        -----------
        variable : str
            Variable name (e.g., '2m_temperature')
        year : int
            Year to access
        month : int, optional
            Month to access
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

        # Create time string for logging
        time_str = f"{year}-{month:02d}"
        if day is not None:
            time_str += f"-{day:02d}"
            if hour is not None:
                time_str += f" {hour:02d}:00"
        print(f"Accessing ERA5-Land variable '{gee_var}' from Google Earth Engine for {time_str}")

        try:
            # Handle region configuration
            if region is None:
                if shapefile is not None:
                    region = self.read_shapefile(shapefile)
                else:
                    region = ee.Geometry.Polygon(
                        [
                            [
                                [config.DEFAULT_LON_MIN, config.DEFAULT_LAT_MIN],
                                [config.DEFAULT_LON_MAX, config.DEFAULT_LAT_MIN],
                                [config.DEFAULT_LON_MAX, config.DEFAULT_LAT_MAX],
                                [config.DEFAULT_LON_MIN, config.DEFAULT_LAT_MAX],
                                [config.DEFAULT_LON_MIN, config.DEFAULT_LAT_MIN],
                            ]
                        ],
                        None,
                        False,  # Planar coordinates for global extent
                    )
                    print(
                        f"Created region with bounds: lon={config.DEFAULT_LON_MIN}-{config.DEFAULT_LON_MAX}, lat={config.DEFAULT_LAT_MIN}-{config.DEFAULT_LAT_MAX}"
                    )

            # Set default scale if not specified (ERA5-Land is ~9km)
            if scale is None:
                scale = 11132  # 9km in meters

            # Set up date range
            if hour is not None and day is not None and month is not None:
                start_date = ee.Date.fromYMD(year, month, day).advance(hour, "hour")
                end_date = start_date.advance(1, "hour")
                print(f"Filtering for exact hour: {start_date.format('YYYY-MM-dd HH:mm:ss').getInfo()}")
            elif day is not None and month is not None:
                start_date = ee.Date.fromYMD(year, month, day)
                if day == calendar.monthrange(year, month)[1] and month == 12:
                    end_date = ee.Date.fromYMD(year + 1, 1, 1)
                elif day == calendar.monthrange(year, month)[1]:
                    end_date = ee.Date.fromYMD(year, month + 1, 1)
                else:
                    end_date = ee.Date.fromYMD(year, month, day + 1)
            elif month is not None:
                start_date = ee.Date.fromYMD(year, month, 1)
                if month == 12:
                    end_date = ee.Date.fromYMD(year + 1, 1, 1)
                else:
                    end_date = ee.Date.fromYMD(year, month + 1, 1)
            else:
                start_date = ee.Date.fromYMD(year, 1, 1)
                end_date = ee.Date.fromYMD(year + 1, 1, 1)

            # Access ERA5-Land collection
            if self.time_scale == "hourly":
                era5land = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
            else:
                era5land = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")

            # Filter by date and select the variable
            era5land = era5land.filterDate(start_date, end_date).select(gee_var)

            # Convert to xarray via geemap
            print("Converting GEE images to xarray dataset...")
            image_list = era5land.toList(era5land.size())
            collection_size = image_list.size().getInfo()
            print(f"Collection contains {collection_size} images")

            # Get the first image for coordinate extraction
            first_img = ee.Image(image_list.get(0))

            # Extract coordinates using the region bounds
            print("Extracting coordinates from region bounds...")

            # Use the specified region bounds for coordinate arrays
            if region and hasattr(region, "bounds"):
                try:
                    bounds_info = region.bounds().getInfo()

                    # Extract min/max coordinates from bounds
                    if "coordinates" in bounds_info:
                        coords = bounds_info["coordinates"][0]
                        lons_list = [coord[0] for coord in coords]
                        lats_list = [coord[1] for coord in coords]
                        lon_min, lon_max = min(lons_list), max(lons_list)
                        lat_min, lat_max = min(lats_list), max(lats_list)
                    else:
                        # Fallback to direct bounds
                        try:
                            if "bbox" in bounds_info:
                                lon_min, lat_min, lon_max, lat_max = bounds_info["bbox"]
                            else:
                                raise ValueError("Could not extract bounds from bounds_info")
                        except Exception:
                            # Try direct attribute access
                            lon_min = bounds_info[0]
                            lat_min = bounds_info[1]
                            lon_max = bounds_info[2]
                            lat_max = bounds_info[3]
                except Exception as e:
                    print(f"Error extracting bounds from region: {e}")
                    # Use default bounds if extraction fails
                    if shapefile:
                        # Get bounds from shapefile if provided
                        gdf = gpd.read_file(shapefile)
                        bounds = gdf.total_bounds
                        lon_min, lat_min, lon_max, lat_max = bounds
                    else:
                        # Use config defaults
                        lon_min = config.DEFAULT_LON_MIN
                        lon_max = config.DEFAULT_LON_MAX
                        lat_min = config.DEFAULT_LAT_MIN
                        lat_max = config.DEFAULT_LAT_MAX
            else:
                # Direct fallback to config
                lon_min = config.DEFAULT_LON_MIN
                lon_max = config.DEFAULT_LON_MAX
                lat_min = config.DEFAULT_LAT_MIN
                lat_max = config.DEFAULT_LAT_MAX

            print(f"Using region bounds: lon={lon_min}-{lon_max}, lat={lat_min}-{lat_max}")

            # Get dimensions from the exported image
            sample_array = geemap.ee_to_numpy(first_img.select(gee_var), region=region, scale=scale)

            # Handle potential extra dimensions
            if sample_array.ndim > 2:
                if sample_array.shape[0] == 1:
                    sample_array = sample_array.squeeze(axis=0)
                elif sample_array.ndim == 3 and sample_array.shape[2] == 1:
                    sample_array = sample_array.squeeze(axis=2)
                else:
                    print(f"Warning: Sample array has unexpected dimensions: {sample_array.shape}")
                    sample_array = (
                        sample_array[0] if sample_array.shape[0] < sample_array.shape[1] else sample_array[:, :, 0]
                    )

            height, width = sample_array.shape[:2]
            print(f"Image dimensions: {height}x{width}")

            # Generate coordinate arrays based on the region bounds and image dimensions
            # Fix 1.1: Use pixel-center coordinates, not pixel-edge
            pixel_size_lon = (lon_max - lon_min) / width
            pixel_size_lat = (lat_max - lat_min) / height
            lons = np.linspace(lon_min + pixel_size_lon / 2, lon_max - pixel_size_lon / 2, width)
            lats = np.linspace(lat_max - pixel_size_lat / 2, lat_min + pixel_size_lat / 2, height)

            # Validate that latitude is in descending order
            lat_needs_flip = False
            if lats[0] < lats[-1]:
                lats = lats[::-1]
                lat_needs_flip = True
                print("Latitude array flipped to ensure descending order")

            print(f"Generated coordinates: lon={lons[0]:.4f}-{lons[-1]:.4f}, lat={lats[0]:.4f}-{lats[-1]:.4f}")
            print(f"Latitude ordering: {'descending' if lats[0] > lats[-1] else 'ascending'}")

            # Test coordinate mapping
            test_lat, test_lon = 0, 10
            lat_idx = np.argmin(np.abs(lats - test_lat))
            lon_idx = np.argmin(np.abs(lons - test_lon))
            print(f"Test location ({test_lat}, {test_lon}) maps to indices: lat_idx={lat_idx}, lon_idx={lon_idx}")
            print(f"Actual coordinates at indices: ({lats[lat_idx]:.4f}, {lons[lon_idx]:.4f})")

            # Process all images
            all_arrays = []
            timestamps = []

            # Fix 3.1: Batch-extract all timestamps in a single getInfo() call
            all_dates = image_list.map(
                lambda img: ee.Date(ee.Image(img).get("system:time_start")).format("YYYY-MM-dd HH:mm:ss")
            ).getInfo()
            all_timestamps = [pd.to_datetime(d, utc=True) if d is not None else None for d in all_dates]

            for i in range(collection_size):
                try:
                    img = ee.Image(image_list.get(i))
                    current_time = all_timestamps[i]
                    if current_time is None:
                        print(f"  Skipping image {i}: missing timestamp metadata")
                        continue

                    # Fix 2.2: Extract array with retry logic for transient GEE failures
                    img_array = None
                    for _attempt in range(3):
                        try:
                            img_array = geemap.ee_to_numpy(img.select(gee_var), region=region, scale=scale)
                            break
                        except Exception as retry_err:
                            if _attempt < 2:
                                _wait = 5 * (3**_attempt)  # 5s, 15s
                                print(f"  Retry {_attempt + 1}/3 for image {i} after {_wait}s: {retry_err}")
                                time.sleep(_wait)
                            else:
                                raise

                    # Handle ERA5-Land NoData values
                    # Common NoData sentinels: 9999, 9.999e20, -9999
                    nodata_values = [9999, 9.999e20, -9999, 9999.0]
                    for nodata in nodata_values:
                        if np.any(np.isclose(img_array, nodata, rtol=1e-5)):
                            img_array = np.where(np.isclose(img_array, nodata, rtol=1e-5), np.nan, img_array)
                            print(f"  Replaced NoData value ~{nodata} with NaN")
                    # Check and handle extra dimensions
                    if img_array.ndim > 2:
                        if img_array.shape[0] == 1:
                            # Squeeze out the band dimension if it's singleton
                            img_array = img_array.squeeze(axis=0)
                        elif img_array.ndim == 3 and img_array.shape[2] == 1:
                            # Handle case where band dimension is last
                            img_array = img_array.squeeze(axis=2)
                        else:
                            # If multiple bands exist, select the first one
                            print(f"Warning: Multiple bands detected. Original shape: {img_array.shape}")
                            img_array = img_array[0] if img_array.shape[0] < img_array.shape[1] else img_array[:, :, 0]
                            print(f"Selected first band. New shape: {img_array.shape}")

                    # Ensure we have a 2D array for lat/lon
                    if img_array.ndim != 2:
                        print(f"Error: After processing, array still has {img_array.ndim} dimensions. Expected 2.")
                        continue

                    # Flip latitude if needed
                    if lat_needs_flip:
                        img_array = np.flip(img_array, axis=0)

                    # Validate data at test location
                    if i % 5 == 0:  # Check every 5th image
                        value_at_test = img_array[lat_idx, lon_idx]
                        print(f"Image {i}: Value at ({test_lat}, {test_lon}): {value_at_test}")

                    # Check for constant values
                    unique_values = np.unique(img_array)
                    if len(unique_values) <= 3:
                        print(f"WARNING: Image {i} has very few unique values!")
                        print(f"  Unique values: {unique_values}")

                        # Try alternative extraction method if constant values detected
                        time_start_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
                        temp_file = self.temp_dir / f"{variable}_{time_start_str}.tif"
                        success = geemap.ee_export_image(
                            img.select(gee_var),
                            filename=str(temp_file),
                            scale=scale,
                            region=region,
                            file_per_band=False,
                        )

                        if success and temp_file.exists():
                            with rasterio.open(temp_file) as src:
                                alt_array = src.read(1)
                                if lat_needs_flip:
                                    alt_array = np.flip(alt_array, axis=0)

                                # Check if alternative method gives different results
                                alt_unique = np.unique(alt_array)
                                if len(alt_unique) > len(unique_values):
                                    print(
                                        f"  Using alternative extraction method (more unique values: {len(alt_unique)})"
                                    )
                                    img_array = alt_array

                    all_arrays.append(img_array)
                    timestamps.append(current_time)

                    if i % 10 == 0:
                        print(f"Processed {i + 1}/{collection_size} images")

                except Exception as e:
                    print(f"Error processing image {i}: {str(e)}")

            if not all_arrays or not timestamps:
                print("No data successfully retrieved")
                return None

            print(f"Successfully processed {len(all_arrays)} out of {collection_size} images")

            # Create the 3D array for time series
            combined_array = np.stack(all_arrays)

            # Final dimension check
            expected_dims = 3  # time, lat, lon
            if combined_array.ndim != expected_dims:
                print(f"Warning: Combined array has {combined_array.ndim} dimensions, expected {expected_dims}")
                print(f"Shape: {combined_array.shape}")
                # If the combined array has extra dimensions, attempt to handle them
                if combined_array.ndim == 4 and combined_array.shape[1] == 1:
                    combined_array = combined_array.squeeze(axis=1)
                    print(f"Squeezed axis 1. New shape: {combined_array.shape}")

            # Create xarray DataArray with proper dimensions
            da = xr.DataArray(
                combined_array,
                dims=["time", "latitude", "longitude"],
                coords={"time": timestamps, "latitude": lats, "longitude": lons},
                name=gee_var,
            )

            # Convert to dataset and add metadata
            ds = da.to_dataset()
            ds.attrs["title"] = f"ERA5-Land {gee_var}"
            ds.attrs["source"] = "Google Earth Engine"
            ds[gee_var].attrs["units"] = self._get_variable_units(gee_var)

            # Final validation
            test_value = ds[gee_var].sel(latitude=test_lat, longitude=test_lon, method="nearest")
            print(f"Final validation - Value at ({test_lat}, {test_lon}): {test_value.values}")

            return ds

        except Exception as e:
            print(f"Error accessing ERA5-Land data from GEE: {str(e)}")
            import traceback

            traceback.print_exc()
            return None

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
                if var.lower() in ["latitude", "lat"]:
                    lat_var = var
                elif var.lower() in ["longitude", "lon"]:
                    lon_var = var

            if not lat_var or not lon_var:
                print("Latitude or longitude variable not found in dataset")
                return ds

            # Print coordinate ranges to diagnose issues
            print(f"Latitude range in data: {ds[lat_var].min().values} to {ds[lat_var].max().values}")
            print(f"Longitude range in data: {ds[lon_var].min().values} to {ds[lon_var].max().values}")
            print(f"Extracting region: lon={lon_min}-{lon_max}, lat={lat_min}-{lat_max}")

            # Extract the region using sel() method
            # Fix 1.7 & 1.8: Correct sel() syntax (single dict) and handle descending lat
            lat_vals = ds[lat_var].values
            if lat_vals[0] > lat_vals[-1]:  # descending order
                lat_slice = slice(lat_max, lat_min)
            else:
                lat_slice = slice(lat_min, lat_max)
            region_ds = ds.sel({lat_var: lat_slice, lon_var: slice(lon_min, lon_max)})

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

    def _build_forest_mask(self, shapefile_path, lats, lons):
        """Rasterise a shapefile onto the ERA5 lat/lon grid.

        Returns a boolean ndarray (nlat, nlon) — True inside the
        shapefile geometry, False outside.  If the shapefile cannot be
        read, returns an all-True mask (no masking).
        """
        import geopandas as gpd
        import rasterio.features
        import rasterio.transform

        try:
            gdf = gpd.read_file(shapefile_path)
            if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs("EPSG:4326")

            nlat, nlon = len(lats), len(lons)
            lat_res = abs(lats[1] - lats[0]) if nlat > 1 else 0.1
            lon_res = abs(lons[1] - lons[0]) if nlon > 1 else 0.1

            west = float(lons.min()) - lon_res / 2
            north = float(lats.max()) + lat_res / 2

            transform = rasterio.transform.from_origin(west, north, lon_res, lat_res)

            shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
            mask = rasterio.features.rasterize(
                shapes,
                out_shape=(nlat, nlon),
                transform=transform,
                fill=0,
                dtype=np.uint8,
                all_touched=True,
            )
            n_inside = int(mask.sum())
            print(
                f"  Forest mask: {n_inside} / {nlat * nlon} pixels "
                f"({100.0 * n_inside / max(nlat * nlon, 1):.1f}%) inside shapefile"
            )
            return mask.astype(bool)

        except Exception as e:
            print(f"  WARNING: Failed to build forest mask ({e}) — skipping mask")
            import traceback

            traceback.print_exc()
            return np.ones((len(lats), len(lons)), dtype=bool)

    def _apply_mask_to_ds(self, ds, mask_2d):
        """Set all data variables to NaN where *mask_2d* is False."""
        import xarray as xr

        # Determine coordinate names used by this dataset
        lat_dim = "latitude" if "latitude" in ds.dims else "lat"
        lon_dim = "longitude" if "longitude" in ds.dims else "lon"

        nan_mask = xr.DataArray(
            ~mask_2d,
            dims=[lat_dim, lon_dim],
            coords={lat_dim: ds[lat_dim], lon_dim: ds[lon_dim]},
        )
        for var in ds.data_vars:
            ds[var] = ds[var].where(~nan_mask)
        return ds

    def load_variable_data(self, variables, year, month, day=None, hour=None, shapefile=None):
        """
        Load and preprocess multiple ERA5-Land variables, potentially clipping to a shapefile.

        The shapefile is rasterised **once** into a boolean mask, then applied
        via ``xr.where`` to every loaded dataset — much faster than the previous
        per-variable ``rioxarray.clip`` approach.

        Parameters:
        -----------
        variables : list
            List of variable names to load.
        year : int
            Year to load data for.
        month : int
            Month to load data for.
        day : int, optional
            Day to load data for.
        hour : int, optional
            Hour to load data for.
        shapefile : str, optional
            Path to a shapefile for forest masking. If None, uses default global extent.
        """
        import gc

        if not shapefile:
            raise ValueError("A shapefile path must be provided for region clipping.")

        # Clear existing datasets before loading new ones
        time_str = f"{year}-{month:02d}"
        if day is not None:
            time_str += f"-{day:02d}"
            if hour is not None:
                time_str += f" {hour:02d}:00"
        print(f"Loading variables for {time_str}...")

        for var_name, ds in self.datasets.items():
            try:
                if hasattr(ds, "close"):
                    ds.close()
            except Exception as e:
                print(f"    Warning: Could not close dataset for {var_name}: {e}")
        self.datasets = {}
        gc.collect()

        # Load each ERA5 variable from GEE
        for var_name in variables:
            var_desc = f"{var_name} for {time_str}"
            print(f"\nProcessing variable: {var_desc}...")

            try:
                ds = self.get_era5land_variable_from_gee(var_name, year, month, day, hour)
            except Exception as gee_err:
                print(f"  ERROR: Failed to load data for {var_name} from GEE: {gee_err}")
                ds = None

            if ds is not None and ds.nbytes > 0:
                print(f"  Loaded {var_name}. Size: {ds.nbytes / 1e6:.2f} MB")
                self.datasets[var_name] = ds
            elif ds is not None:
                print(f"  Loaded {var_name} but dataset is empty. Skipping.")
            else:
                print(f"  Failed to load {var_name}. Skipping.")

        # Build forest mask ONCE from shapefile and apply to ALL datasets
        if shapefile is not None and self.datasets:
            ref_ds = next(iter(self.datasets.values()))
            lat_dim = "latitude" if "latitude" in ref_ds.dims else "lat"
            lon_dim = "longitude" if "longitude" in ref_ds.dims else "lon"
            lats = ref_ds[lat_dim].values
            lons = ref_ds[lon_dim].values

            # Cache mask so repeated calls in the same run reuse it
            if not hasattr(self, "_forest_mask") or self._forest_mask is None:
                print(f"\nBuilding forest mask from {shapefile}...")
                self._forest_mask = self._build_forest_mask(shapefile, lats, lons)

            print("Applying forest mask to all loaded datasets...")
            for var_name in list(self.datasets):
                self.datasets[var_name] = self._apply_mask_to_ds(self.datasets[var_name], self._forest_mask)

        # At the END of load_variable_data, AFTER all ERA5 variables are loaded:
        # ============================================================
        # MASK ZERO VALUES (NEW - Add this section)
        # ============================================================
        print("\nApplying zero-value mask to ERA5-Land data...")
        self.mask_zero_values(threshold=1e-10)
        # Load raw static datasets (reload when year changes for year-dependent PFT)
        if (
            not hasattr(self, "_static_raw_loaded")
            or not self._static_raw_loaded
            or getattr(self, "_static_year", None) != year
        ):
            print(f"\nLoading raw static datasets for year {year}...")
            self.load_static_datasets(year, shapefile=shapefile)
            self._static_raw_loaded = True
            self._static_year = year

        # Load raw LAI data
        self.load_lai_data_raw(year, month, day, shapefile=shapefile)

        # Now resample all static data to match ERA5 grid
        # Get target coordinates from loaded ERA5 data
        target_lats = None
        target_lons = None
        target_times = None

        if self.datasets:
            ref_ds_key = next(iter(self.datasets))
            ref_ds = self.datasets[ref_ds_key]

            for coord in ref_ds.coords:
                coord_lower = coord.lower()
                if coord_lower in ("latitude", "lat") and target_lats is None:
                    target_lats = ref_ds[coord].values
                elif coord_lower in ("longitude", "lon") and target_lons is None:
                    target_lons = ref_ds[coord].values
                elif coord_lower in ("time", "datetime") and target_times is None:
                    target_times = ref_ds[coord].values

        # Resample static data to ERA5 grid
        self.resample_static_data(target_lats=target_lats, target_lons=target_lons, target_times=target_times)

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
            if "time" in ds.coords:
                time_values = ds.coords["time"].values
                unique_times = np.unique(time_values)
                print(f"Dataset '{var_name}' has {len(time_values)} unique time values")
                print(time_values)
                if len(time_values) != len(unique_times):
                    print(
                        f"WARNING: Dataset '{var_name}' has {len(time_values) - len(unique_times)} duplicate time values"
                    )
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
        if "10m_u_component_of_wind" in self.datasets and "10m_v_component_of_wind" in self.datasets:
            print("Calculating wind speed from u and v components using chunked approach...")
            try:
                u_ds = self.datasets["10m_u_component_of_wind"]
                v_ds = self.datasets["10m_v_component_of_wind"]

                # Get the main data variable from each dataset
                u_var = list(u_ds.data_vars)[0] if list(u_ds.data_vars) else None
                v_var = list(v_ds.data_vars)[0] if list(v_ds.data_vars) else None

                if u_var and v_var:
                    # Check if we have time dimension
                    if "time" in u_ds.dims:
                        times = u_ds.coords["time"].values
                        lat_var = [v for v in u_ds.dims if v.lower() in ("latitude", "lat")][0]
                        lon_var = [v for v in u_ds.dims if v.lower() in ("longitude", "lon")][0]

                        # Process in time chunks
                        wind_chunks = []
                        chunk_coords = {"time": [], "latitude": u_ds[lat_var].values, "longitude": u_ds[lon_var].values}

                        # Process each time step separately to conserve memory
                        for t_idx in range(len(times)):
                            # Extract single time slice to reduce memory usage
                            u_slice = u_ds[u_var].isel(time=t_idx).values
                            v_slice = v_ds[v_var].isel(time=t_idx).values

                            # Calculate wind speed using numpy directly - EXACT SAME FORMULA
                            wind_slice = np.sqrt(u_slice**2 + v_slice**2)

                            # Store results and time
                            wind_chunks.append(wind_slice)
                            chunk_coords["time"].append(times[t_idx])

                            # Status update
                            if t_idx % max(1, len(times) // 10) == 0 or t_idx == len(times) - 1:
                                print(f"Processed wind speed for time {t_idx + 1}/{len(times)}")

                        # Combine chunks into a single dataset
                        wind_array = np.stack(wind_chunks)
                        wind_da = xr.DataArray(
                            wind_array, dims=["time", lat_var, lon_var], coords=chunk_coords, name="wind_speed"
                        )

                        # Create dataset
                        wind_ds = wind_da.to_dataset()
                        wind_ds["wind_speed"].attrs["units"] = "m s-1"
                        wind_ds["wind_speed"].attrs["long_name"] = "Wind speed at 10m"

                        self.datasets["wind_speed"] = wind_ds
                        print("Wind speed calculated successfully with chunked approach")
                    else:
                        # Standard approach using named variables - EXACT SAME AS ORIGINAL
                        wind_speed = np.sqrt(u_ds[u_var] ** 2 + v_ds[v_var] ** 2)

                        wind_ds = xr.Dataset(
                            data_vars={"wind_speed": wind_speed},
                            coords=u_ds.coords,
                            attrs={"units": "m s-1", "long_name": "Wind speed at 10m"},
                        )
                        wind_ds["wind_speed"].attrs["units"] = "m s-1"
                        wind_ds["wind_speed"].attrs["long_name"] = "Wind speed at 10m"

                        self.datasets["wind_speed"] = wind_ds
                        print(f"Wind speed calculated successfully from variables: {u_var}, {v_var}")

                else:
                    print("Could not identify u and v component variables")
            except Exception as e:
                print(f"Error calculating wind speed: {str(e)}")
                import traceback

                traceback.print_exc()

        # ============================================================
        # Fix 1.2C: Compute wind speed min/max from u/v component extremes
        # NOTE: This is an APPROXIMATION. Daily u_min and v_min are temporally
        # independent (they occur at different hours). sqrt(u_min²+v_min²) gives
        # a LOWER BOUND that may OVERESTIMATE the true daily minimum wind speed.
        # Similarly, sqrt(u_max²+v_max²) gives an UPPER BOUND. The training
        # pipeline computes ws=sqrt(u²+v²) hourly then takes min/max, which is
        # exact. This approximation is consistent with how other gridded products
        # handle daily wind speed extremes from component aggregates.
        # ============================================================
        if "10m_u_component_of_wind_max" in self.datasets and "10m_v_component_of_wind_max" in self.datasets:
            print("Calculating wind speed max...")
            try:
                u_max_ds = self.datasets["10m_u_component_of_wind_max"]
                v_max_ds = self.datasets["10m_v_component_of_wind_max"]
                u_max_var = list(u_max_ds.data_vars)[0]
                v_max_var = list(v_max_ds.data_vars)[0]
                ws_max = np.sqrt(u_max_ds[u_max_var].values ** 2 + v_max_ds[v_max_var].values ** 2)
                ws_max_ds = xr.Dataset(
                    data_vars={"wind_speed_max": (u_max_ds[u_max_var].dims, ws_max)},
                    coords=u_max_ds.coords,
                    attrs={"units": "m s-1", "long_name": "Maximum wind speed at 10m"},
                )
                self.datasets["wind_speed_max"] = ws_max_ds
                print("Wind speed max calculated successfully")
            except Exception as e:
                print(f"Error calculating wind speed max: {e}")

        if "10m_u_component_of_wind_min" in self.datasets and "10m_v_component_of_wind_min" in self.datasets:
            print("Calculating wind speed min...")
            try:
                u_min_ds = self.datasets["10m_u_component_of_wind_min"]
                v_min_ds = self.datasets["10m_v_component_of_wind_min"]
                u_min_var = list(u_min_ds.data_vars)[0]
                v_min_var = list(v_min_ds.data_vars)[0]
                ws_min = np.sqrt(u_min_ds[u_min_var].values ** 2 + v_min_ds[v_min_var].values ** 2)
                ws_min_ds = xr.Dataset(
                    data_vars={"wind_speed_min": (u_min_ds[u_min_var].dims, ws_min)},
                    coords=u_min_ds.coords,
                    attrs={"units": "m s-1", "long_name": "Minimum wind speed at 10m"},
                )
                self.datasets["wind_speed_min"] = ws_min_ds
                print("Wind speed min calculated successfully")
            except Exception as e:
                print(f"Error calculating wind speed min: {e}")

        # Calculate VPD if temperature and dewpoint are available
        if "temperature_2m" in self.datasets and "dewpoint_temperature_2m" in self.datasets:
            print("Calculating vapor pressure deficit (VPD) using chunked approach...")
            try:
                t_ds = self.datasets["temperature_2m"]
                td_ds = self.datasets["dewpoint_temperature_2m"]

                # Get the main data variable from each dataset
                t_var = list(t_ds.data_vars)[0] if list(t_ds.data_vars) else None
                td_var = list(td_ds.data_vars)[0] if list(td_ds.data_vars) else None

                if t_var and td_var:
                    # Check if we have time dimension
                    if "time" in t_ds.dims:
                        times = t_ds.coords["time"].values
                        lat_var = [v for v in t_ds.dims if v.lower() in ("latitude", "lat")][0]
                        lon_var = [v for v in t_ds.dims if v.lower() in ("longitude", "lon")][0]

                        # Process in time chunks
                        vpd_chunks = []
                        chunk_coords = {"time": [], "latitude": t_ds[lat_var].values, "longitude": t_ds[lon_var].values}

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
                            # Fix 1.4: Convert hPa to kPa (divide by 10) to match training data
                            vpd_slice = np.maximum(es - ea, 0.0) / 10.0

                            # Store results and time
                            vpd_chunks.append(vpd_slice)
                            chunk_coords["time"].append(times[t_idx])

                            # Status update
                            if t_idx % max(1, len(times) // 10) == 0 or t_idx == len(times) - 1:
                                print(f"Processed VPD for time {t_idx + 1}/{len(times)}")

                        # Combine chunks into a single dataset
                        vpd_array = np.stack(vpd_chunks)
                        vpd_da = xr.DataArray(
                            vpd_array, dims=["time", lat_var, lon_var], coords=chunk_coords, name="vpd"
                        )

                        # Create dataset
                        vpd_ds = vpd_da.to_dataset()
                        vpd_ds["vpd"].attrs["units"] = "kPa"
                        vpd_ds["vpd"].attrs["long_name"] = "Vapor Pressure Deficit"

                        self.datasets["vpd"] = vpd_ds
                        print("VPD calculated successfully with chunked approach (kPa)")
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
                        # Fix 1.4: Convert hPa to kPa to match training data
                        vpd = np.maximum(es - ea, 0.0) / 10.0

                        # Create a new dataset with the VPD variable
                        vpd_ds = xr.Dataset(
                            data_vars={"vpd": (t_ds[t_var].dims, vpd)},
                            coords=t_ds.coords,
                            attrs={"units": "kPa", "long_name": "Vapor Pressure Deficit"},
                        )
                        vpd_ds["vpd"].attrs["units"] = "kPa"
                        vpd_ds["vpd"].attrs["long_name"] = "Vapor Pressure Deficit"

                        self.datasets["vpd"] = vpd_ds
                        print("VPD calculated successfully using ClimateDataCalculator approach")
                else:
                    print("Could not identify temperature variables")
            except Exception as e:
                print(f"Error calculating VPD: {str(e)}")
                import traceback

                traceback.print_exc()

        # ============================================================
        # Fix 1.2B: Compute VPD min/max from temperature/dewpoint extremes
        # vpd_max = VPD(ta_max, td_min) — hottest + driest moment
        # vpd_min = VPD(ta_min, td_max) — coolest + most humid moment
        # ============================================================
        if "temperature_2m_max" in self.datasets and "dewpoint_temperature_2m_min" in self.datasets:
            print("Calculating VPD max (from ta_max, td_min)...")
            try:
                t_max_ds = self.datasets["temperature_2m_max"]
                td_min_ds = self.datasets["dewpoint_temperature_2m_min"]
                t_max_var = list(t_max_ds.data_vars)[0]
                td_min_var = list(td_min_ds.data_vars)[0]

                t_max_c = t_max_ds[t_max_var].values - 273.15
                td_min_c = td_min_ds[td_min_var].values - 273.15
                td_min_c = np.minimum(td_min_c, t_max_c)  # physical: Td ≤ T

                es = 6.1078 * np.exp((17.269 * t_max_c) / (237.3 + t_max_c))
                ea = 6.1078 * np.exp((17.269 * td_min_c) / (237.3 + td_min_c))
                vpd_max_vals = np.maximum(es - ea, 0.0) / 10.0  # hPa → kPa

                vpd_max_ds = xr.Dataset(
                    data_vars={"vpd_max": (t_max_ds[t_max_var].dims, vpd_max_vals)},
                    coords=t_max_ds.coords,
                    attrs={"units": "kPa", "long_name": "Maximum Vapor Pressure Deficit"},
                )
                self.datasets["vpd_max"] = vpd_max_ds
                print("VPD max calculated successfully (kPa)")
            except Exception as e:
                print(f"Error calculating VPD max: {e}")

        if "temperature_2m_min" in self.datasets and "dewpoint_temperature_2m_max" in self.datasets:
            print("Calculating VPD min (from ta_min, td_max)...")
            try:
                t_min_ds = self.datasets["temperature_2m_min"]
                td_max_ds = self.datasets["dewpoint_temperature_2m_max"]
                t_min_var = list(t_min_ds.data_vars)[0]
                td_max_var = list(td_max_ds.data_vars)[0]

                t_min_c = t_min_ds[t_min_var].values - 273.15
                td_max_c = td_max_ds[td_max_var].values - 273.15
                td_max_c = np.minimum(td_max_c, t_min_c)  # physical: Td ≤ T

                es = 6.1078 * np.exp((17.269 * t_min_c) / (237.3 + t_min_c))
                ea = 6.1078 * np.exp((17.269 * td_max_c) / (237.3 + td_max_c))
                vpd_min_vals = np.maximum(es - ea, 0.0) / 10.0  # hPa → kPa

                vpd_min_ds = xr.Dataset(
                    data_vars={"vpd_min": (t_min_ds[t_min_var].dims, vpd_min_vals)},
                    coords=t_min_ds.coords,
                    attrs={"units": "kPa", "long_name": "Minimum Vapor Pressure Deficit"},
                )
                self.datasets["vpd_min"] = vpd_min_ds
                print("VPD min calculated successfully (kPa)")
            except Exception as e:
                print(f"Error calculating VPD min: {e}")

        # ============================================================
        # Fix 1.2E: Rename ta_min/ta_max from Kelvin to Celsius
        # These are loaded directly from GEE in Kelvin; convert now
        # ============================================================
        for ta_var in ["temperature_2m_min", "temperature_2m_max"]:
            if ta_var in self.datasets:
                ds = self.datasets[ta_var]
                dv = list(ds.data_vars)[0]
                # Idempotency guard: only convert if still in Kelvin
                if ds[dv].attrs.get("units", "K") != "°C":
                    ds[dv].values[:] = ds[dv].values - 273.15
                    ds[dv].attrs["units"] = "°C"

        # ============================================================
        # Fix 1.2D: Compute day_length from latitude and day of year
        # Uses CBM model: day_length = f(latitude, declination)
        # ============================================================
        if self.datasets:
            print("Calculating day length...")
            try:
                ref_ds = next(iter(self.datasets.values()))
                ref_var = list(ref_ds.data_vars)[0]
                lat_name = [d for d in ref_ds.dims if d.lower() in ("latitude", "lat")][0]
                lon_name = [d for d in ref_ds.dims if d.lower() in ("longitude", "lon")][0]
                lat_vals = ref_ds[lat_name].values
                lon_vals = ref_ds[lon_name].values

                if "time" in ref_ds.dims:
                    times = ref_ds.coords["time"].values
                    day_length_array = np.zeros((len(times), len(lat_vals), len(lon_vals)), dtype=np.float32)
                    for t_idx, t in enumerate(times):
                        ts = pd.Timestamp(t)
                        doy = ts.dayofyear
                        days_in_year = 366 if ts.is_leap_year else 365
                        lat_rad = np.deg2rad(lat_vals)
                        declination = 0.4093 * np.sin(2 * np.pi * (doy - 81) / days_in_year)
                        cos_ha = -np.tan(lat_rad) * np.tan(declination)
                        cos_ha = np.clip(cos_ha, -1, 1)
                        dl = 2 * np.arccos(cos_ha) * 12 / np.pi  # hours
                        day_length_array[t_idx, :, :] = dl[:, np.newaxis]  # broadcast to lon

                    dl_ds = xr.Dataset(
                        data_vars={"day_length": (["time", lat_name, lon_name], day_length_array)},
                        coords={"time": times, lat_name: lat_vals, lon_name: lon_vals},
                        attrs={"units": "hours", "long_name": "Day length"},
                    )
                else:
                    # Single timestep — use Jan 1 as default
                    doy = 1
                    lat_rad = np.deg2rad(lat_vals)
                    declination = 0.4093 * np.sin(2 * np.pi * (doy - 81) / 365)
                    cos_ha = np.clip(-np.tan(lat_rad) * np.tan(declination), -1, 1)
                    dl = 2 * np.arccos(cos_ha) * 12 / np.pi
                    dl_2d = dl[:, np.newaxis] * np.ones((1, len(lon_vals)))
                    dl_ds = xr.Dataset(
                        data_vars={"day_length": ([lat_name, lon_name], dl_2d.astype(np.float32))},
                        coords={lat_name: lat_vals, lon_name: lon_vals},
                        attrs={"units": "hours", "long_name": "Day length"},
                    )

                self.datasets["day_length"] = dl_ds
                print("Day length calculated successfully")
            except Exception as e:
                print(f"Error calculating day length: {e}")
                import traceback

                traceback.print_exc()

        # Calculate Precipitation / PET ratio if both are available
        if "total_precipitation" in self.datasets and "potential_evaporation" in self.datasets:
            print("Calculating precipitation/PET ratio...")
            try:
                precip_ds = self.datasets["total_precipitation"]
                pet_ds = self.datasets["potential_evaporation"]

                precip_var = list(precip_ds.data_vars)[0]
                pet_var = list(pet_ds.data_vars)[0]

                if "time" in precip_ds.dims:
                    times = precip_ds.coords["time"].values
                    lat_var = [v for v in precip_ds.dims if v.lower() in ("latitude", "lat")][0]
                    lon_var = [v for v in precip_ds.dims if v.lower() in ("longitude", "lon")][0]

                    precip_pet_chunks = []
                    chunk_coords = {
                        "time": [],
                        "latitude": precip_ds[lat_var].values,
                        "longitude": precip_ds[lon_var].values,
                    }

                    for t_idx in range(len(times)):
                        precip_slice = precip_ds[precip_var].isel(time=t_idx).values
                        pet_slice = pet_ds[pet_var].isel(time=t_idx).values

                        # PET from ERA5 is negative (evaporation is loss)
                        # Convert to positive values
                        pet_slice = np.abs(pet_slice)

                        # Avoid division by zero
                        pet_slice = np.where(pet_slice < 1e-10, 1e-10, pet_slice)

                        # Calculate ratio
                        ratio_slice = precip_slice / pet_slice

                        # Clip to reasonable range
                        ratio_slice = np.clip(ratio_slice, 0, 10)

                        precip_pet_chunks.append(ratio_slice)
                        chunk_coords["time"].append(times[t_idx])

                    precip_pet_array = np.stack(precip_pet_chunks)
                    precip_pet_da = xr.DataArray(
                        precip_pet_array, dims=["time", lat_var, lon_var], coords=chunk_coords, name="precip_pet_ratio"
                    )

                    precip_pet_ds = precip_pet_da.to_dataset()
                    precip_pet_ds["precip_pet_ratio"].attrs["units"] = "ratio"
                    precip_pet_ds["precip_pet_ratio"].attrs["long_name"] = "Precipitation to PET ratio"

                    self.datasets["precip_pet_ratio"] = precip_pet_ds
                    print("Precipitation/PET ratio calculated successfully")

            except Exception as e:
                print(f"Error calculating precipitation/PET ratio: {str(e)}")
                traceback.print_exc()
        elif "total_precipitation" in self.datasets:
            # If no PET available, estimate it using Hargreaves method
            print("PET not available, estimating using temperature-based method...")
            try:
                if "temperature_2m" in self.datasets:
                    precip_ds = self.datasets["total_precipitation"]
                    temp_ds = self.datasets["temperature_2m"]

                    precip_var = list(precip_ds.data_vars)[0]
                    temp_var = list(temp_ds.data_vars)[0]

                    if "time" in precip_ds.dims:
                        times = precip_ds.coords["time"].values
                        lat_var = [v for v in precip_ds.dims if v.lower() in ("latitude", "lat")][0]
                        lon_var = [v for v in precip_ds.dims if v.lower() in ("longitude", "lon")][0]

                        lats = precip_ds[lat_var].values

                        precip_pet_chunks = []
                        chunk_coords = {"time": [], "latitude": lats, "longitude": precip_ds[lon_var].values}

                        for t_idx in range(len(times)):
                            precip_slice = precip_ds[precip_var].isel(time=t_idx).values
                            temp_slice = temp_ds[temp_var].isel(time=t_idx).values - 273.15  # K to C

                            # Get day of year for solar radiation estimate
                            timestamp = pd.Timestamp(times[t_idx])
                            doy = timestamp.dayofyear

                            # Simple Hargreaves-Samani PET estimate (mm/day)
                            # PET = 0.0023 * Ra * (T + 17.8) * sqrt(TD)
                            # Simplified version using temperature only
                            lat_rad = np.deg2rad(lats)[:, np.newaxis]

                            # Extraterrestrial radiation estimate
                            dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
                            delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
                            ws = np.arccos(-np.tan(lat_rad) * np.tan(delta))
                            ws = np.clip(ws, 0, np.pi)

                            Ra = (
                                (24 * 60 / np.pi)
                                * 0.082
                                * dr
                                * (ws * np.sin(lat_rad) * np.sin(delta) + np.cos(lat_rad) * np.cos(delta) * np.sin(ws))
                            )

                            # Simplified PET (mm/hour, assuming hourly data)
                            pet_slice = 0.0023 * Ra * (temp_slice + 17.8) * np.sqrt(np.maximum(temp_slice, 0) + 1) / 24
                            pet_slice = np.maximum(pet_slice, 1e-10)

                            # Precipitation is in meters, convert to mm for ratio
                            precip_mm = precip_slice * 1000

                            ratio_slice = precip_mm / pet_slice
                            ratio_slice = np.clip(ratio_slice, 0, 10)

                            precip_pet_chunks.append(ratio_slice)
                            chunk_coords["time"].append(times[t_idx])

                        precip_pet_array = np.stack(precip_pet_chunks)
                        precip_pet_da = xr.DataArray(
                            precip_pet_array,
                            dims=["time", lat_var, lon_var],
                            coords=chunk_coords,
                            name="precip_pet_ratio",
                        )

                        precip_pet_ds = precip_pet_da.to_dataset()
                        precip_pet_ds["precip_pet_ratio"].attrs["units"] = "ratio"
                        precip_pet_ds["precip_pet_ratio"].attrs["long_name"] = "Precipitation to PET ratio (estimated)"

                        self.datasets["precip_pet_ratio"] = precip_pet_ds
                        print("Precipitation/PET ratio estimated successfully using Hargreaves method")

            except Exception as e:
                print(f"Error estimating precipitation/PET ratio: {str(e)}")
                traceback.print_exc()

        # Calculate PAR-related variables if solar radiation is available
        if "surface_solar_radiation_downwards" in self.datasets:
            print("Calculating PAR-related variables from solar radiation...")
            try:
                sr_ds = self.datasets["surface_solar_radiation_downwards"]

                # Get the main data variable
                sr_var = list(sr_ds.data_vars)[0] if list(sr_ds.data_vars) else None

                if sr_var:
                    # Check if the data needs time differencing (for accumulated values)
                    is_accumulated = False

                    if self.time_scale == "daily":
                        is_accumulated = False
                    elif self.time_scale == "hourly":
                        # if variable name contains 'hourly', assume instantaneous
                        # otherwise, assume accumulated
                        if "hourly" in sr_var.lower():
                            is_accumulated = False
                            print(
                                "  > Detected 'hourly' in variable name, treating solar radiation as instantaneous values."
                            )
                        else:
                            is_accumulated = True
                            print(
                                "  > Treating solar radiation as accumulated values needing conversion to instantaneous."
                            )

                    # Process with time chunking if we have time dimension
                    if "time" in sr_ds.dims:
                        times = sr_ds.coords["time"].values
                        lat_var = [v for v in sr_ds.dims if v.lower() in ("latitude", "lat")][0]
                        lon_var = [v for v in sr_ds.dims if v.lower() in ("longitude", "lon")][0]

                        # Get dimensions for spatial chunking
                        lat_values = sr_ds[lat_var].values
                        lon_values = sr_ds[lon_var].values
                        num_lats = len(lat_values)
                        num_lons = len(lon_values)

                        # Calculate time steps in seconds (for accumulated values)
                        if is_accumulated and len(times) > 1:
                            time_diff_seconds = np.diff(pd.DatetimeIndex(times).view("int64") / 10**9).astype(float)
                            # Handle case with only one time step
                            time_diff_seconds = np.append(
                                time_diff_seconds[0] if len(time_diff_seconds) > 0 else 3600, time_diff_seconds
                            )

                        # ============================================================
                        # HANDLE ACCUMULATED -> INSTANTANEOUS CONVERSION (HOURLY ONLY)
                        # ============================================================
                        if is_accumulated:
                            print("Converting accumulated solar radiation to instantaneous values...")

                            inst_values = np.zeros_like(sr_ds[sr_var].values)

                            for t_idx in range(len(times)):
                                current = sr_ds[sr_var].isel(time=t_idx).values

                                if t_idx == 0:
                                    # First timestep: accumulated value is energy since 00:00
                                    dt = time_diff_seconds[0]
                                    inst_values[0] = current / dt
                                else:
                                    previous = sr_ds[sr_var].isel(time=t_idx - 1).values
                                    dt = time_diff_seconds[t_idx]
                                    diff = current - previous

                                    # Reset detection: if diff < 0, we crossed 00:00 UTC
                                    # In that case, 'current' is the fresh accumulation for this hour
                                    inst_values[t_idx] = np.where(diff < 0, current, diff) / dt

                                # Check for unrealistic values
                                if np.any(inst_values[t_idx] > 1500) or np.any(inst_values[t_idx] < -10):
                                    print(f"Warning: Extreme solar radiation values detected at time {t_idx}")
                                    print(f"Range: {np.min(inst_values[t_idx])} to {np.max(inst_values[t_idx])}")
                                    # Clip to realistic values
                                    inst_values[t_idx] = np.clip(inst_values[t_idx], 0, 1500)

                            # Create new DataArray and Dataset
                            new_sr_da = xr.DataArray(
                                inst_values,
                                dims=sr_ds[sr_var].dims,
                                coords=sr_ds[sr_var].coords,
                                attrs=sr_ds[sr_var].attrs.copy(),
                            )
                            new_sr_da.attrs["units"] = "W m-2"

                            # Create new Dataset with instantaneous values
                            new_sr_ds = sr_ds.copy()
                            new_sr_ds[sr_var] = new_sr_da

                            # Update the dataset
                            sr_ds = new_sr_ds
                            self.datasets["surface_solar_radiation_downwards"] = new_sr_ds
                            print("Successfully converted accumulated solar radiation to instantaneous values")

                        else:
                            # ============================================================
                            # CONVERT J/m² TO W/m² (NON-ACCUMULATED DATA)
                            # ============================================================
                            if self.time_scale == "daily":
                                # Fix 1.6: Daily sum J/m2 / 86400s = daily mean W/m2
                                # This is consistent with training data's sw_in (daily mean irradiance)
                                dt_seconds = 86400.0
                            elif self.time_scale == "hourly":
                                dt_seconds = 3600.0
                            else:
                                dt_seconds = 3600.0  # Default fallback

                            print(f"  > Converting Energy (J/m²) to Power (W/m²) using dt={dt_seconds}s")

                            # Create a new dataset to hold the converted values
                            new_sr_ds = sr_ds.copy()
                            new_sr_ds[sr_var] = sr_ds[sr_var] / dt_seconds
                            new_sr_ds[sr_var].attrs["units"] = "W m-2"

                            # Update the stored dataset
                            self.datasets["surface_solar_radiation_downwards"] = new_sr_ds
                            sr_ds = new_sr_ds
                            print("Successfully converted solar radiation from J/m² to W/m²")

                        # ============================================================
                        # MEMORY OPTIMIZATION: Process in spatial chunks
                        # ============================================================
                        LAT_CHUNK_SIZE = min(100, num_lats)  # Process up to 100 latitude points at a time

                        # Initialize final result arrays
                        ppfd_array = np.zeros((len(times), num_lats, num_lons), dtype=np.float32)
                        ext_rad_array = np.zeros((len(times), num_lats, num_lons), dtype=np.float32)

                        # Create coordinate dictionary for final datasets
                        chunk_coords = {"time": times, "latitude": lat_values, "longitude": lon_values}

                        # PAR conversion constants
                        PAR_FRACTION = 0.45
                        PPFD_CONVERSION = 4.6

                        # Solar constant (W/m²)
                        Gsc = 1367.0

                        # ============================================================
                        # BRANCH: DAILY vs HOURLY CALCULATION
                        # ============================================================
                        if self.time_scale == "daily":
                            # ========================================================
                            # DAILY CALCULATION: FAO-56 integrated formula for ext_rad
                            # ========================================================
                            print("Calculating DAILY extraterrestrial radiation using FAO-56 formula...")

                            try:
                                for t_idx in range(len(times)):
                                    # Get the converted solar radiation (now in W/m²)
                                    sr_slice_full = sr_ds[sr_var].isel(time=t_idx).values

                                    # Get day of year
                                    timestamp = pd.Timestamp(times[t_idx])
                                    doy = timestamp.dayofyear

                                    # ----- Earth-Sun distance correction (dr) -----
                                    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)

                                    # ----- Solar declination (Spencer's formula) -----
                                    day_angle = 2 * np.pi * (doy - 1) / 365
                                    declination = (
                                        0.006918
                                        - 0.399912 * np.cos(day_angle)
                                        + 0.070257 * np.sin(day_angle)
                                        - 0.006758 * np.cos(2 * day_angle)
                                        + 0.000907 * np.sin(2 * day_angle)
                                        - 0.002697 * np.cos(3 * day_angle)
                                        + 0.001480 * np.sin(3 * day_angle)
                                    )

                                    # Process in latitude chunks for memory efficiency
                                    for lat_chunk_start in range(0, num_lats, LAT_CHUNK_SIZE):
                                        lat_chunk_end = min(lat_chunk_start + LAT_CHUNK_SIZE, num_lats)
                                        lat_chunk_size = lat_chunk_end - lat_chunk_start

                                        # Extract chunk of solar radiation data
                                        sr_slice = sr_slice_full[lat_chunk_start:lat_chunk_end, :]

                                        # Data validation
                                        sr_slice = np.where(sr_slice < 0, 0.0, sr_slice)

                                        # ----- PPFD Calculation -----
                                        ppfd_slice = sr_slice * PAR_FRACTION * PPFD_CONVERSION
                                        ppfd_slice = np.clip(ppfd_slice, 0.0, 2500.0)
                                        ppfd_array[t_idx, lat_chunk_start:lat_chunk_end, :] = ppfd_slice

                                        # ----- Daily ext_rad using FAO-56 -----
                                        ext_rad_slice = np.zeros((lat_chunk_size, num_lons), dtype=np.float32)
                                        lat_chunk_values = lat_values[lat_chunk_start:lat_chunk_end]

                                        for i, lat in enumerate(lat_chunk_values):
                                            lat_rad = np.deg2rad(lat)

                                            # Sunset hour angle (ωs)
                                            tan_product = -np.tan(lat_rad) * np.tan(declination)

                                            # Handle polar day/night
                                            if tan_product < -1:
                                                ws = np.pi  # Polar day: sun never sets
                                            elif tan_product > 1:
                                                ws = 0  # Polar night: sun never rises
                                            else:
                                                ws = np.arccos(np.clip(tan_product, -1, 1))

                                            # Daily extraterrestrial radiation (FAO-56 Eq. 21)
                                            # Ra = (24×60/π) × Gsc × dr × [ωs×sin(φ)×sin(δ) + cos(φ)×cos(δ)×sin(ωs)]
                                            # Result is in W·min/m²/day
                                            Ra = (
                                                (24 * 60 / np.pi)
                                                * Gsc
                                                * dr
                                                * (
                                                    ws * np.sin(lat_rad) * np.sin(declination)
                                                    + np.cos(lat_rad) * np.cos(declination) * np.sin(ws)
                                                )
                                            )

                                            # Convert to daily mean W/m² (divide by minutes per day)
                                            Ra_mean_wm2 = Ra / (24 * 60)

                                            # Store (broadcast to all longitudes in this chunk)
                                            ext_rad_slice[i, :] = Ra_mean_wm2

                                        ext_rad_array[t_idx, lat_chunk_start:lat_chunk_end, :] = ext_rad_slice

                                        # Clean up memory
                                        del sr_slice, ppfd_slice, ext_rad_slice

                                    # Progress update
                                    if t_idx % max(1, len(times) // 10) == 0 or t_idx == len(times) - 1:
                                        print(f"    Processed daily PAR variables for day {t_idx + 1}/{len(times)}")

                                print("Daily PAR-related variables calculated successfully")

                            except Exception as daily_err:
                                print(f"Error in daily PAR calculation: {str(daily_err)}")
                                traceback.print_exc()

                                # Fallback: use simple approximation
                                print("Falling back to simple approximation for daily ext_rad...")
                                for t_idx in range(len(times)):
                                    sr_slice = sr_ds[sr_var].isel(time=t_idx).values
                                    sr_slice = np.maximum(sr_slice, 0.0)

                                    ppfd_array[t_idx, :, :] = np.clip(
                                        sr_slice * PAR_FRACTION * PPFD_CONVERSION, 0.0, 2500.0
                                    )
                                    ext_rad_array[t_idx, :, :] = np.clip(sr_slice * 1.5, 0.0, 1500.0)

                        else:
                            # ========================================================
                            # HOURLY CALCULATION: Instantaneous values (ORIGINAL CODE)
                            # ========================================================
                            print(
                                "Calculating extraterrestrial radiation using enhanced algorithm with spatial chunking..."
                            )

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
                                def datetime_to_julian(dt):
                                    """Convert datetime to Julian day since 2000-01-01 12:00:00"""
                                    dt = pd.Timestamp(dt).to_pydatetime()
                                    a = (14 - dt.month) // 12
                                    y = dt.year + 4800 - a
                                    m = dt.month + 12 * a - 3
                                    jdn = dt.day + ((153 * m + 2) // 5) + 365 * y + y // 4 - y // 100 + y // 400 - 32045
                                    jd = jdn + (dt.hour - 12) / 24.0 + dt.minute / 1440.0 + dt.second / 86400.0
                                    return jd - 2451545.0

                                # Calculate Julian days for each timestamp
                                jd_array = np.array([datetime_to_julian(t) for t in times])

                                # Day angle for solar calculations
                                X_array = 2 * np.pi * (dn - 1) / 365

                                # Solar calculations using the 'michalsky' method (default in solaR)
                                method = "michalsky"

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
                                        sr_slice = np.where(sr_slice < 0, 0.0, sr_slice)

                                        # Calculate PAR (Photosynthetically Active Radiation)
                                        par_wm2 = sr_slice * PAR_FRACTION

                                        # Convert PAR from W/m2 to μmol/m2/s
                                        ppfd_slice = par_wm2 * PPFD_CONVERSION

                                        # Validate and constrain to physical limits
                                        ppfd_slice = np.clip(ppfd_slice, 0.0, 2500.0)

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
                                        if method == "cooper":
                                            decl = d2r(23.45 * np.sin(2 * np.pi * (dn[t_idx] + 284) / 365))
                                        elif method == "spencer":
                                            decl = (
                                                0.006918
                                                - 0.399912 * np.cos(X)
                                                + 0.070257 * np.sin(X)
                                                - 0.006758 * np.cos(2 * X)
                                                + 0.000907 * np.sin(2 * X)
                                                - 0.002697 * np.cos(3 * X)
                                                + 0.001480 * np.sin(3 * X)
                                            )
                                        elif method == "strous":
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
                                            meanLong = (280.460 + 0.9856474 * jd) % 360
                                            meanAnomaly = (357.528 + 0.9856003 * jd) % 360
                                            eclipLong = (
                                                meanLong
                                                + 1.915 * np.sin(d2r(meanAnomaly))
                                                + 0.02 * np.sin(d2r(2 * meanAnomaly))
                                            ) % 360
                                            excen = 23.439 - 0.0000004 * jd
                                            decl = np.arcsin(np.sin(d2r(eclipLong)) * np.sin(d2r(excen)))

                                        # Calculate Earth-Sun distance factor (eo)
                                        if method == "cooper":
                                            eo = 1 + 0.033 * np.cos(2 * np.pi * day_of_year / 365)
                                        else:
                                            eo = (
                                                1.000110
                                                + 0.034221 * np.cos(X)
                                                + 0.001280 * np.sin(X)
                                                + 0.000719 * np.cos(2 * X)
                                                + 0.000077 * np.sin(2 * X)
                                            )

                                        # Calculate Equation of Time (minutes)
                                        M = 2 * np.pi / 365.24 * day_of_year
                                        EoT_min = 229.18 * (-0.0334 * np.sin(M) + 0.04184 * np.sin(2 * M + 3.5884))
                                        EoT_hours = EoT_min / 60

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
                                            cos_zenith = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(
                                                decl
                                            ) * np.cos(hour_angle_rad)

                                            # No radiation when sun is below horizon
                                            cos_zenith = max(0.0, cos_zenith)

                                            # Calculate extraterrestrial radiation adjusted by Earth-Sun distance
                                            ext_rad_slice[i, :] = Bo * eo * cos_zenith

                                        # Store ext_rad result in final array
                                        ext_rad_array[t_idx, lat_chunk_start:lat_chunk_end, :] = ext_rad_slice

                                        # Clean up memory for this chunk
                                        del sr_slice, par_wm2, ppfd_slice, ext_rad_slice

                                    # Status update
                                    if t_idx % max(1, len(times) // 10) == 0 or t_idx == len(times) - 1:
                                        print(f"    Processed PAR variables for time {t_idx + 1}/{len(times)}")

                            except Exception as e:
                                print(f"Error in extraterrestrial radiation calculation: {str(e)}")
                                traceback.print_exc()

                                # Fall back to simpler method if error occurs in custom calculation
                                print("Falling back to simpler extraterrestrial radiation calculation...")

                                # Reset arrays (in case they were partially filled)
                                ppfd_array = np.zeros((len(times), num_lats, num_lons), dtype=np.float32)
                                ext_rad_array = np.zeros((len(times), num_lats, num_lons), dtype=np.float32)

                                # Process each time step with simplified approach
                                for t_idx in range(len(times)):
                                    sr_slice_full = sr_ds[sr_var].isel(time=t_idx).values

                                    for lat_chunk_start in range(0, num_lats, LAT_CHUNK_SIZE):
                                        lat_chunk_end = min(lat_chunk_start + LAT_CHUNK_SIZE, num_lats)

                                        sr_slice = sr_slice_full[lat_chunk_start:lat_chunk_end, :]
                                        sr_slice = np.nan_to_num(sr_slice, nan=0.0)
                                        sr_slice = np.maximum(sr_slice, 0.0)

                                        par_wm2 = sr_slice * PAR_FRACTION
                                        ppfd_slice = par_wm2 * PPFD_CONVERSION
                                        ppfd_slice = np.clip(ppfd_slice, 0.0, 2500.0)

                                        ext_rad_slice = sr_slice * 1.5
                                        ext_rad_slice = np.clip(ext_rad_slice, 0.0, 1500.0)

                                        ppfd_array[t_idx, lat_chunk_start:lat_chunk_end, :] = ppfd_slice
                                        ext_rad_array[t_idx, lat_chunk_start:lat_chunk_end, :] = ext_rad_slice

                                        del sr_slice, par_wm2, ppfd_slice, ext_rad_slice

                                    if t_idx % max(1, len(times) // 10) == 0 or t_idx == len(times) - 1:
                                        print(
                                            f"    Processed PAR variables for time {t_idx + 1}/{len(times)} (simplified method)"
                                        )

                        # ============================================================
                        # CREATE OUTPUT DATASETS (common for both daily and hourly)
                        # ============================================================

                        # Create PPFD dataset
                        ppfd_da = xr.DataArray(
                            ppfd_array, dims=["time", lat_var, lon_var], coords=chunk_coords, name="ppfd_in"
                        )
                        ppfd_ds = ppfd_da.to_dataset()
                        ppfd_ds["ppfd_in"].attrs["units"] = "μmol m-2 s-1"
                        ppfd_ds["ppfd_in"].attrs["long_name"] = "Photosynthetic Photon Flux Density"
                        ppfd_ds["ppfd_in"].attrs["description"] = f"Calculated from {self.time_scale} solar radiation"
                        self.datasets["ppfd_in"] = ppfd_ds

                        # Create extraterrestrial radiation dataset
                        ext_rad_da = xr.DataArray(
                            ext_rad_array, dims=["time", lat_var, lon_var], coords=chunk_coords, name="ext_rad"
                        )
                        ext_rad_ds = ext_rad_da.to_dataset()
                        ext_rad_ds["ext_rad"].attrs["units"] = "W m-2"
                        if self.time_scale == "daily":
                            ext_rad_ds["ext_rad"].attrs["long_name"] = "Daily Mean Extraterrestrial Radiation (FAO-56)"
                        else:
                            ext_rad_ds["ext_rad"].attrs["long_name"] = "Instantaneous Extraterrestrial Radiation"
                        self.datasets["ext_rad"] = ext_rad_ds

                        print(f"PAR-related variables calculated successfully ({self.time_scale} mode)")

                        # Validation output
                        valid_ppfd = ppfd_array[~np.isnan(ppfd_array)]
                        valid_ext = ext_rad_array[~np.isnan(ext_rad_array)]
                        if len(valid_ppfd) > 0:
                            print(f"  PPFD range: [{valid_ppfd.min():.1f}, {valid_ppfd.max():.1f}] μmol/m²/s")
                        if len(valid_ext) > 0:
                            print(f"  ext_rad range: [{valid_ext.min():.1f}, {valid_ext.max():.1f}] W/m²")

                    else:
                        # ============================================================
                        # NO TIME DIMENSION: Single snapshot (original fallback)
                        # ============================================================
                        print("Processing single-time solar radiation data (no time dimension)...")
                        solar_rad = sr_ds[sr_var]

                        # Data validation
                        solar_rad = solar_rad.where(solar_rad >= 0, 0)

                        # Calculate PAR
                        par_wm2 = solar_rad * 0.45
                        ppfd = par_wm2 * 4.6
                        ppfd = ppfd.where(ppfd <= 2500.0, 2500.0)

                        ppfd_ds = xr.Dataset(
                            data_vars={"ppfd_in": ppfd},
                            coords=sr_ds.coords,
                            attrs={"units": "μmol m-2 s-1", "long_name": "Photosynthetic Photon Flux Density"},
                        )
                        ppfd_ds["ppfd_in"].attrs["units"] = "μmol m-2 s-1"
                        ppfd_ds["ppfd_in"].attrs["long_name"] = "Photosynthetic Photon Flux Density"

                        # Simple ext_rad approximation (no time info available for proper calculation)
                        ext_rad = solar_rad * 1.5
                        ext_rad = ext_rad.where(ext_rad <= 1500.0, 1500.0)
                        ext_rad_ds = xr.Dataset(
                            data_vars={"ext_rad": ext_rad},
                            coords=sr_ds.coords,
                            attrs={"units": "W m-2", "long_name": "Extraterrestrial Radiation (approximated)"},
                        )
                        ext_rad_ds["ext_rad"].attrs["units"] = "W m-2"
                        ext_rad_ds["ext_rad"].attrs["long_name"] = "Extraterrestrial Radiation (approximated)"

                        self.datasets["ppfd_in"] = ppfd_ds
                        self.datasets["ext_rad"] = ext_rad_ds
                        print("PAR-related variables calculated successfully (single-time mode, ext_rad approximated)")
                else:
                    print("Could not identify solar radiation variable")

            except Exception as e:
                print(f"Error calculating PAR-related variables: {str(e)}")
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
                lon_min = bounds.get("lon_min")
                lat_min = bounds.get("lat_min")
                lon_max = bounds.get("lon_max")
                lat_max = bounds.get("lat_max")
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
                    "data": temp_data,
                    "transform": window_transform,  # Use window transform if subsetting
                    "crs": temp_crs,
                    "height": height,
                    "width": width,
                    "nodata": temp_src.nodata,
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
                    "data": precip_data,
                    "transform": window_transform,  # Use window transform if subsetting
                    "crs": precip_src.crs,
                    "height": height,
                    "width": width,
                    "nodata": precip_src.nodata,
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

    def get_climate_at_location_direct(self, lon, lat, temp_climate=None, precip_climate=None, max_distance=0.05):
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
        temperatures = np.full_like(lons, np.nan, dtype=float)
        precipitations = np.full_like(lats, np.nan, dtype=float)

        # Process temperature data
        if temp_climate and "data" in temp_climate:
            try:
                start_time = time.time()

                # Get raster dimensions
                height, width = temp_climate["data"].shape

                # Extract raster extent
                transform = temp_climate["transform"]
                xmin, ymax = transform * (0, 0)
                xmax, ymin = transform * (width, height)

                print(f"Temperature data extent: lon={xmin:.4f}-{xmax:.4f}, lat={ymin:.4f}-{ymax:.4f}")

                # Check how many points are outside the extent
                outside_extent = (lons < xmin) | (lons > xmax) | (lats < ymin) | (lats > ymax)
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
                values = temp_climate["data"].flatten()

                # Only include valid data points
                nodata = temp_climate.get("nodata")
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
                    print(
                        f"Temperature lookup completed in {tree_time:.2f}s: {np.sum(temperatures != 15.0)} valid values found"
                    )

                    # Additional debug info
                    if np.sum(valid_distance) < len(lons):
                        print(
                            f"Some points were too far from valid climate data: {len(lons) - np.sum(valid_distance)} points using default temperature"
                        )
            except Exception as e:
                print(f"Error in temperature KD-tree lookup: {str(e)}")
                import traceback

                traceback.print_exc()
        else:
            print("No temperature climate data available for KD-tree lookup")

        # Process precipitation data
        if precip_climate and "data" in precip_climate:
            try:
                start_time = time.time()

                # Get raster dimensions
                height, width = precip_climate["data"].shape

                # Extract raster extent
                transform = precip_climate["transform"]
                xmin, ymax = transform * (0, 0)
                xmax, ymin = transform * (width, height)

                print(f"Precipitation data extent: lon={xmin:.4f}-{xmax:.4f}, lat={ymin:.4f}-{ymax:.4f}")

                # Check how many points are outside the extent
                outside_extent = (lons < xmin) | (lons > xmax) | (lats < ymin) | (lats > ymax)
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
                values = precip_climate["data"].flatten()

                # Only include valid data points
                nodata = precip_climate.get("nodata")
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
                    print(
                        f"Precipitation lookup completed in {tree_time:.2f}s: {np.sum(precipitations != 800.0)} valid values found"
                    )

                    # Additional debug info
                    if np.sum(valid_distance) < len(lons):
                        print(
                            f"Some points were too far from valid climate data: {len(lons) - np.sum(valid_distance)} points using default precipitation"
                        )
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

    def get_raster_at_location_direct(self, lon, lat, raster_data, max_distance=0.1, default_value=np.nan):
        """
        Generic KD-tree approach for raster data lookup - same approach as climate data.
        Works for elevation, PFT, canopy height, LAI, etc.

        Parameters:
        -----------
        lon, lat : float or array-like
            Coordinates of the location(s)
        raster_data : dict
            Raster data dictionary with 'data', 'transform', and optionally 'nodata'
        max_distance : float, optional
            Maximum distance (in degrees) to accept a valid data point.
        default_value : float, optional
            Default value for points outside valid data range.

        Returns:
        --------
        numpy.ndarray or float
            Values at the specified locations
        """

        if raster_data is None or "data" not in raster_data:
            print("Warning: No raster data available for lookup")
            single_point = np.isscalar(lon) and np.isscalar(lat)
            if single_point:
                return default_value
            else:
                return np.full_like(np.asarray(lon), default_value, dtype=float)

        # Convert inputs to arrays
        single_point = np.isscalar(lon) and np.isscalar(lat)
        lons = np.array([lon]) if single_point else np.asarray(lon).flatten()
        lats = np.array([lat]) if single_point else np.asarray(lat).flatten()

        # Default values
        values = np.full_like(lons, default_value, dtype=float)

        try:
            # Get raster dimensions
            data = raster_data["data"]
            height, width = data.shape

            # Extract raster extent using transform (SAME AS CLIMATE DATA)
            transform = raster_data["transform"]
            xmin, ymax = transform * (0, 0)
            xmax, ymin = transform * (width, height)

            print(f"    Raster extent: lon=[{xmin:.4f}, {xmax:.4f}], lat=[{ymin:.4f}, {ymax:.4f}]")
            print(
                f"    Query points: lon=[{lons.min():.4f}, {lons.max():.4f}], lat=[{lats.min():.4f}, {lats.max():.4f}]"
            )

            # Check for overlap
            if lons.max() < xmin or lons.min() > xmax or lats.max() < ymin or lats.min() > ymax:
                print("    WARNING: No overlap between raster and query points!")
                return values[0] if single_point else values

            # Create a regular grid of pixel center coordinates (SAME AS CLIMATE DATA)
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
            data_values = data.flatten().astype(float)

            # Only include valid data points
            nodata = raster_data.get("nodata")
            if nodata is not None:
                valid_mask = (data_values != nodata) & (~np.isnan(data_values))
            else:
                valid_mask = ~np.isnan(data_values)

            num_valid = np.sum(valid_mask)
            if num_valid == 0:
                print("    WARNING: No valid data points found in raster")
                return values[0] if single_point else values

            # Create KD-tree only with valid points
            valid_points = points[valid_mask]
            valid_values = data_values[valid_mask]

            print(f"    Valid data points: {num_valid}, unique values: {len(np.unique(valid_values))}")

            # Build tree and query
            tree = cKDTree(valid_points)
            query_points = np.vstack((lons, lats)).T
            distances, indices = tree.query(query_points, k=1)

            # Extract values with distance check
            valid_distance = distances <= max_distance
            values[valid_distance] = valid_values[indices[valid_distance]]

            num_within_distance = np.sum(valid_distance)
            print(f"    Points within max_distance ({max_distance}): {num_within_distance} / {len(lons)}")

            if num_within_distance > 0:
                result_unique = len(np.unique(values[valid_distance]))
                print(f"    Result unique values: {result_unique}")

            if num_within_distance < len(lons):
                print(f"    Points using default value: {len(lons) - num_within_distance}")

        except Exception as e:
            print(f"Error in raster KD-tree lookup: {str(e)}")
            traceback.print_exc()

        if single_point:
            return float(values[0])
        else:
            return values

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
        if not hasattr(self, "temp_climate_data") or not hasattr(self, "precip_climate_data"):
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
                    if var.lower() in ["latitude", "lat"]:
                        lat_var = var
                    elif var.lower() in ["longitude", "lon"]:
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
                dims=["latitude", "longitude"],
                coords={"latitude": target_lat, "longitude": target_lon},
                name="mean_annual_temp",
            )

            resampled_precip_da = xr.DataArray(
                resampled_precip,
                dims=["latitude", "longitude"],
                coords={"latitude": target_lat, "longitude": target_lon},
                name="annual_mean_precipitation",
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

    def resample_static_data(self, target_lats=None, target_lons=None, target_times=None):
        """
        Resample all static data to match ERA5 grid coordinates.
        Uses EXACT same approach as resample_climate_data() and get_climate_at_location_direct().

        Parameters:
        -----------
        target_lats : array-like, optional
            Target latitude array from ERA5
        target_lons : array-like, optional
            Target longitude array from ERA5
        target_times : array-like, optional
            Target time array from ERA5 (for LAI temporal alignment)

        Returns:
        --------
        dict
            Dictionary of resampled static DataArrays
        """
        # Get target coordinates from ERA5 if not provided
        if target_lats is None or target_lons is None:
            if not self.datasets:
                print("Error: No ERA5 datasets loaded and no target coordinates provided")
                return None

            ref_ds_key = next(iter(self.datasets))
            ref_ds = self.datasets[ref_ds_key]

            for coord in ref_ds.coords:
                coord_lower = coord.lower()
                if coord_lower in ("latitude", "lat") and target_lats is None:
                    target_lats = ref_ds[coord].values
                elif coord_lower in ("longitude", "lon") and target_lons is None:
                    target_lons = ref_ds[coord].values
                elif coord_lower in ("time", "datetime") and target_times is None:
                    target_times = ref_ds[coord].values

        if target_lats is None or target_lons is None:
            print("Error: Could not determine target coordinates")
            return None

        print(f"\n{'=' * 60}")
        print("RESAMPLING STATIC DATA TO ERA5 GRID")
        print(f"Target grid: {len(target_lats)} lat x {len(target_lons)} lon")
        print(f"Target lat range: [{target_lats.min():.4f}, {target_lats.max():.4f}]")
        print(f"Target lon range: [{target_lons.min():.4f}, {target_lons.max():.4f}]")
        print(f"{'=' * 60}")

        # Create coordinate meshgrid - SAME AS CLIMATE DATA
        lon_grid, lat_grid = np.meshgrid(target_lons, target_lats)
        flat_lons = lon_grid.flatten()
        flat_lats = lat_grid.flatten()

        print(f"Total query points: {len(flat_lons)}")

        resampled = {}

        # ========== ELEVATION ==========
        if self.elevation_raw is not None:
            print("\n--- Resampling ELEVATION using climate data approach ---")
            elev_values = self.get_raster_at_location_direct(
                flat_lons,
                flat_lats,
                self.elevation_raw,
                max_distance=0.05,
            )
            elev_grid = elev_values.reshape(len(target_lats), len(target_lons))

            self.elevation_data = xr.DataArray(
                elev_grid,
                dims=["latitude", "longitude"],
                coords={"latitude": target_lats, "longitude": target_lons},
                name="elevation",
            )
            self.elevation_data.attrs["units"] = "m"
            self.elevation_data.attrs["long_name"] = "Elevation from ASTER DEM"
            resampled["elevation"] = self.elevation_data

            valid_elev = elev_grid[~np.isnan(elev_grid)]
            print(
                f"  RESULT: shape={elev_grid.shape}, range=[{valid_elev.min():.1f}, {valid_elev.max():.1f}] m, "
                f"unique={len(np.unique(valid_elev))}"
            )
        else:
            print("\n--- ELEVATION: Raw data not available, skipping ---")

        # ========== PFT ==========
        if self.pft_raw is not None:
            print("\n--- Resampling PFT using climate data approach ---")
            pft_values = self.get_raster_at_location_direct(
                flat_lons,
                flat_lats,
                self.pft_raw,
                max_distance=0.05,
            )
            pft_grid = pft_values.reshape(len(target_lats), len(target_lons))

            self.pft_data = xr.DataArray(
                pft_grid.astype(int),
                dims=["latitude", "longitude"],
                coords={"latitude": target_lats, "longitude": target_lons},
                name="pft",
            )
            self.pft_data.attrs["units"] = "class"
            self.pft_data.attrs["long_name"] = "Plant Functional Type from MODIS"
            resampled["pft"] = self.pft_data

            valid_pft = pft_grid[~np.isnan(pft_grid)]
            unique_classes = np.unique(valid_pft).astype(int)
            print(f"  RESULT: shape={pft_grid.shape}, classes={unique_classes.tolist()}, unique={len(unique_classes)}")
        else:
            print("\n--- PFT: Raw data not available, skipping ---")

        # ========== CANOPY HEIGHT ==========
        if self.canopy_height_raw is not None:
            print("\n--- Resampling CANOPY HEIGHT using climate data approach ---")
            ch_values = self.get_raster_at_location_direct(
                flat_lons,
                flat_lats,
                self.canopy_height_raw,
                max_distance=0.05,
            )
            ch_grid = ch_values.reshape(len(target_lats), len(target_lons))

            self.canopy_height_data = xr.DataArray(
                ch_grid,
                dims=["latitude", "longitude"],
                coords={"latitude": target_lats, "longitude": target_lons},
                name="canopy_height",
            )
            self.canopy_height_data.attrs["units"] = "m"
            self.canopy_height_data.attrs["long_name"] = "Canopy Height from Meta"
            resampled["canopy_height"] = self.canopy_height_data

            valid_ch = ch_grid[~np.isnan(ch_grid)]
            if len(valid_ch) > 0:
                print(
                    f"  RESULT: shape={ch_grid.shape}, range=[{valid_ch.min():.1f}, {valid_ch.max():.1f}] m, "
                    f"unique={len(np.unique(valid_ch))}"
                )
            else:
                print(f"  RESULT: shape={ch_grid.shape}, WARNING: no valid data!")
        else:
            print("\n--- CANOPY HEIGHT: Raw data not available, skipping ---")

        # ========== LAI ==========
        if self.lai_raw is not None:
            print("\n--- Resampling LAI using climate data approach ---")
            lai_data = self.lai_raw["data"]  # Shape: (n_times, height, width)
            lai_times = self.lai_raw["times"]
            n_lai_times = len(lai_times)

            print(f"  LAI has {n_lai_times} time slices")

            # Resample each LAI time slice spatially using same approach
            lai_resampled_times = []
            for t_idx in range(n_lai_times):
                # Create temporary dict for this time slice (same format as climate data)
                lai_slice_dict = {
                    "data": lai_data[t_idx],
                    "transform": self.lai_raw["transform"],
                    "nodata": self.lai_raw.get("nodata"),
                }

                if t_idx == 0:
                    print(f"  Processing LAI time slice {t_idx} ({lai_times[t_idx]}):")

                lai_values = self.get_raster_at_location_direct(
                    flat_lons, flat_lats, lai_slice_dict, max_distance=0.05, default_value=np.nan
                )
                lai_grid = lai_values.reshape(len(target_lats), len(target_lons))
                lai_resampled_times.append(lai_grid)

                if t_idx == 0:
                    valid_lai_slice = lai_grid[~np.isnan(lai_grid)]
                    if len(valid_lai_slice) > 0:
                        print(
                            f"    First slice result: range=[{valid_lai_slice.min():.2f}, {valid_lai_slice.max():.2f}], "
                            f"unique={len(np.unique(valid_lai_slice))}"
                        )

            lai_resampled_stack = np.stack(lai_resampled_times, axis=0)
            print(f"  Stacked LAI shape: {lai_resampled_stack.shape}")

            # Temporal alignment to ERA5 times
            if target_times is not None and len(target_times) > 0:
                print(f"  Aligning LAI temporally to {len(target_times)} ERA5 timestamps...")
                target_times_pd = pd.DatetimeIndex(target_times)

                # Make timezone naive for comparison
                if target_times_pd.tz is not None:
                    target_times_pd = target_times_pd.tz_localize(None)

                lai_times_pd = pd.DatetimeIndex(lai_times)

                # Find nearest LAI time for each target time
                lai_indices = []
                for target_time in target_times_pd:
                    time_diffs = np.abs((lai_times_pd - target_time).total_seconds())
                    nearest_idx = np.argmin(time_diffs)
                    lai_indices.append(nearest_idx)

                # Create aligned LAI array
                lai_aligned = lai_resampled_stack[lai_indices, :, :]

                self.lai_data = xr.DataArray(
                    lai_aligned,
                    dims=["time", "latitude", "longitude"],
                    coords={"time": target_times_pd, "latitude": target_lats, "longitude": target_lons},
                    name="LAI",
                )
                print(f"  Temporally aligned LAI shape: {lai_aligned.shape}")
            else:
                # No target times - keep original LAI times
                self.lai_data = xr.DataArray(
                    lai_resampled_stack,
                    dims=["time", "latitude", "longitude"],
                    coords={"time": pd.DatetimeIndex(lai_times), "latitude": target_lats, "longitude": target_lons},
                    name="LAI",
                )

            self.lai_data.attrs["units"] = "m2/m2"
            self.lai_data.attrs["long_name"] = "Leaf Area Index from GLOBMAP"
            resampled["LAI"] = self.lai_data

            # Final LAI stats
            valid_lai = self.lai_data.values[~np.isnan(self.lai_data.values)]
            if len(valid_lai) > 0:
                print(
                    f"  RESULT: shape={self.lai_data.shape}, range=[{valid_lai.min():.2f}, {valid_lai.max():.2f}], "
                    f"unique={len(np.unique(valid_lai))}"
                )
            else:
                print(f"  RESULT: shape={self.lai_data.shape}, WARNING: no valid LAI data!")
        else:
            print("\n--- LAI: Raw data not available, skipping ---")

        print(f"\n{'=' * 60}")
        print("STATIC DATA RESAMPLING COMPLETE")
        print(f"Resampled {len(resampled)} datasets: {list(resampled.keys())}")
        print(f"{'=' * 60}\n")

        return resampled

    def add_time_features(self, df):
        """
        Add Local Solar Time (LST) and derived cyclical features.

        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with 'timestamp' and 'longitude' columns

        Returns:
        --------
        pandas.DataFrame
            DataFrame with 'solar_timestamp' and cyclical features
        """
        # 1. Validate Input
        if "timestamp" not in df.columns:
            # If timestamp is the index, reset it to a column
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df = df.rename(columns={"index": "timestamp"})
            else:
                print("Error: 'timestamp' missing for solar time calculation.")
                return df

        if "longitude" not in df.columns:
            print("Error: 'longitude' column missing. Cannot calculate solar time.")
            return df

        try:
            # Ensure proper types
            times_utc = pd.to_datetime(df["timestamp"], utc=True)
            lons = df["longitude"].astype(float)

            # --- 2. Calculate Equation of Time (EoT) ---
            # EoT accounts for Earth's elliptical orbit and axial tilt.
            # Formula approximation (minutes): E = 9.87*sin(2B) - 7.53*cos(B) - 1.5*sin(B)
            # Where B = 360/365 * (d - 81), d is day of year.

            doy = times_utc.dt.dayofyear
            B = (360.0 / 365.0) * (doy - 81) * (np.pi / 180.0)  # Convert degrees to radians

            eot_minutes = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)

            # --- 3. Calculate Local Solar Time (LST) ---
            # Formula: LST = UTC + (Longitude * 4 min/deg) + EoT_minutes

            lon_offset_minutes = lons * 4.0
            total_offset_minutes = lon_offset_minutes + eot_minutes

            # Create the solar timestamp
            # We add the offset to the UTC time
            df["solar_timestamp"] = times_utc + pd.to_timedelta(total_offset_minutes, unit="m")

            # --- 4. Generate Cyclical Features based on SOLAR Time ---

            # A. Daily Cycle (Based on Solar Hour)
            # We convert solar time to "seconds from solar midnight"
            solar_dt = df["solar_timestamp"].dt
            solar_seconds = solar_dt.hour * 3600 + solar_dt.minute * 60 + solar_dt.second
            seconds_in_day = 24 * 60 * 60.0

            df["Day sin"] = np.sin(2 * np.pi * solar_seconds / seconds_in_day)
            # df['Day cos'] = np.cos(2 * np.pi * solar_seconds / seconds_in_day)

            # B. Yearly Cycle (Seasonality)
            # Using Solar Day of Year ensures seasonality aligns with sun position
            solar_doy = solar_dt.dayofyear
            # Include fraction of day for smoother transition
            solar_year_fraction = solar_doy + (solar_seconds / seconds_in_day)
            days_in_year = 365.25

            df["Year sin"] = np.sin(2 * np.pi * solar_year_fraction / days_in_year)
            # df['Year cos'] = np.cos(2 * np.pi * solar_year_fraction / days_in_year)

            # Optional: Remove the intermediate solar_timestamp if not needed for output
            # df = df.drop(columns=['solar_timestamp'])

            print("Successfully added Solar Time features.")

        except Exception as e:
            print(f"Error calculating solar time features: {e}")
            import traceback

            traceback.print_exc()

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
                        result.append(pd.Timestamp(ts).tz_localize("UTC"))
                    else:
                        result.append(pd.Timestamp(ts).tz_convert("UTC"))
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
                    return timestamps.tz_localize("UTC")
                else:
                    return timestamps.tz_convert("UTC")
            else:
                # Make timezone-naive
                if timestamps.tz is not None:
                    return timestamps.tz_localize(None)
                else:
                    return timestamps

        elif hasattr(timestamps, "values"):  # xarray time coordinate
            # Extract values and convert back to same type
            values = pd.DatetimeIndex(timestamps.values)
            standardized = self.standardize_timestamps(values, use_utc)
            return type(timestamps)(standardized)

        return timestamps  # Return unchanged if not recognized type

        return timestamps  # Return unchanged if not recognized type

    def create_prediction_dataset(
        self, start_date=None, end_date=None, points=None, region=None, use_existing_data=True, use_utc=True
    ):
        """
        Create a dataset ready for prediction by extracting data.
        Enhanced with coordinate validation, debugging improvements, and xr.merge optimization.

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
                "lat_min": getattr(config, "DEFAULT_LAT_MIN", -90),
                "lat_max": getattr(config, "DEFAULT_LAT_MAX", 90),
                "lon_min": getattr(config, "DEFAULT_LON_MIN", -180),
                "lon_max": getattr(config, "DEFAULT_LON_MAX", 180),
            }

        if use_existing_data:
            if not self.datasets:
                print("Error: use_existing_data=True but no data has been loaded.")
                return None  # Changed from raise ValueError to return None for worker compatibility

            # Check if derived variables are needed and calculated
            source_vars_needed = list(getattr(config, "VARIABLE_RENAME", {}).keys())  # Use getattr for safety
            # Example derived names - adjust based on your actual derived variables
            required_derived_vars = ["wind_speed", "vpd", "ppfd_in", "ext_rad"]
            missing_derived = []
            # Improved check using VARIABLE_RENAME and required_derived_vars
            model_names_needed = list(getattr(config, "VARIABLE_RENAME", {}).values())
            for derived_var in required_derived_vars:
                if derived_var in model_names_needed and derived_var not in self.datasets:
                    missing_derived.append(derived_var)
                # Handle cases where source name might be used if rename isn't perfect
                elif derived_var in source_vars_needed and derived_var not in self.datasets:
                    # Check if the *renamed* version exists instead
                    renamed_var = getattr(config, "VARIABLE_RENAME", {}).get(derived_var)
                    if not renamed_var or renamed_var not in self.datasets:
                        # Only add if neither original nor renamed derived var exists
                        if derived_var not in missing_derived:
                            missing_derived.append(derived_var)

            if missing_derived:
                print(f"Calculating missing derived variables: {missing_derived}...")
                try:
                    self.calculate_derived_variables()
                    # Verify they exist now
                    still_missing = [var for var in missing_derived if var not in self.datasets]
                    if still_missing:
                        print(f"Warning: Could not calculate all derived variables. Missing: {still_missing}")
                except Exception as calc_err:
                    print(f"Error during derived variable calculation: {calc_err}")
                    traceback.print_exc()  # Print stack trace for debugging calc issues

            # Determine time range from existing data if start/end not specified
            if start_date is None or end_date is None:
                all_times_pd = []
                for ds_name, ds in self.datasets.items():
                    time_coord = None
                    for coord_name in ds.coords:
                        if coord_name.lower() in ["time", "datetime"]:
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
                    return None  # Cannot proceed without time range

                # Concatenate all time indices and standardize
                # Standardize each time index BEFORE combining to avoid tz-aware/naive mixing
                combined_times = pd.DatetimeIndex([])
                for idx in all_times_pd:
                    # Standardize this index first
                    standardized_idx = self.standardize_timestamps(idx, use_utc=use_utc)
                    if standardized_idx is not None and not standardized_idx.empty:
                        if combined_times.empty:
                            combined_times = standardized_idx
                        else:
                            combined_times = combined_times.union(standardized_idx)

                standardized_times = combined_times  # Already standardized

                if standardized_times is not None and not standardized_times.empty:
                    if start_date is None:
                        start_date = standardized_times.min()
                        print(f"Using start_date from existing data: {start_date}")
                    if end_date is None:
                        end_date = standardized_times.max()
                        print(f"Using end_date from existing data: {end_date}")
                else:
                    print("Error: Could not determine time range from existing data (standardization failed or empty).")
                    return None
        else:
            # If not using existing data, assume data for the required period
            # was loaded beforehand by the calling function (e.g., hourly worker)
            if start_date is None or end_date is None:
                print(
                    "Error: start_date and end_date are required if use_existing_data=False (and data isn't pre-loaded)."
                )
                return None

        # Convert dates to datetime objects and standardize timezone
        start_date_std, end_date_std = None, None
        try:
            if start_date:
                start_date_pd = pd.Timestamp(start_date)
                # Standardize requires a DatetimeIndex input for this implementation
                temp_start_index = pd.DatetimeIndex([start_date_pd])
                standardized_start = self.standardize_timestamps(temp_start_index, use_utc=use_utc)
                if standardized_start is not None and not standardized_start.empty:
                    start_date_std = standardized_start[0]
            if end_date:
                end_date_pd = pd.Timestamp(end_date)
                temp_end_index = pd.DatetimeIndex([end_date_pd])
                standardized_end = self.standardize_timestamps(temp_end_index, use_utc=use_utc)
                if standardized_end is not None and not standardized_end.empty:
                    end_date_std = standardized_end[0]

            # Ensure start is before end if both are defined
            if start_date_std and end_date_std and start_date_std > end_date_std:
                print(f"Warning: start_date ({start_date_std}) is after end_date ({end_date_std}). Swapping them.")
                start_date_std, end_date_std = end_date_std, start_date_std

        except Exception as date_err:
            print(f"Error processing start/end dates: {date_err}")
            traceback.print_exc()
            return None

        # Select only the variables needed for prediction based on config rename mapping
        prediction_vars_map = {}  # Maps original_name -> model_name
        variable_rename_config = getattr(config, "VARIABLE_RENAME", {})
        for source_name, model_name in variable_rename_config.items():
            # Check if the source variable exists in the loaded datasets
            if source_name in self.datasets:
                prediction_vars_map[source_name] = model_name
            # Also check if a derived variable (which might share the model name) exists
            elif model_name in self.datasets and model_name not in prediction_vars_map.values():
                # Use the model_name as the 'original' key if the source_name wasn't found
                # but the derived variable (matching model_name) exists.
                prediction_vars_map[model_name] = model_name

        if not prediction_vars_map:
            print("Error: No required data variables found in loaded datasets based on config.")
            print(f"Looked for sources/derived vars based on: {variable_rename_config}")
            print(f"Available datasets: {list(self.datasets.keys())}")
            return None

        print(f"Using variables for processing (Original/Source Name -> Model Name): {prediction_vars_map}")
        original_vars_to_process = list(prediction_vars_map.keys())

        if points:
            # --- POINTS-BASED APPROACH with static data extraction ---
            print("Processing specific points...")
            all_data = []

            # Vectorized lookups for all points at once
            point_lons = np.array([p["lon"] for p in points])
            point_lats = np.array([p["lat"] for p in points])
            n_points = len(points)

            # ============================================
            # CLIMATE DATA LOOKUP (existing)
            # ============================================
            temps, precips = None, None
            climate_data_available = False

            if (
                hasattr(self, "temp_climate_data")
                and self.temp_climate_data is not None
                and hasattr(self, "precip_climate_data")
                and self.precip_climate_data is not None
            ):
                try:
                    print(f"Getting climate data for {n_points} points...")
                    temps, precips = self.get_climate_at_location_direct(
                        point_lons, point_lats, self.temp_climate_data, self.precip_climate_data
                    )
                    climate_data_available = True
                    print("  Climate data retrieved successfully")
                except Exception as e:
                    print(f"  Warning: Error in climate data retrieval: {e}")
            else:
                print("Warning: Climate data not loaded, will use defaults")

            # ============================================
            # STATIC DATA LOOKUP (using same approach as climate data)
            # ============================================
            elevations = None
            pfts = None
            canopy_heights = None

            # Elevation lookup - using same method as climate data
            if hasattr(self, "elevation_raw") and self.elevation_raw is not None:
                try:
                    print(f"Getting elevation data for {n_points} points...")
                    elevations = self.get_raster_at_location_direct(
                        point_lons, point_lats, self.elevation_raw, max_distance=0.05
                    )
                    valid_elev = elevations[~np.isnan(elevations)]
                    if len(valid_elev) > 0:
                        print(f"  Elevation range: [{valid_elev.min():.1f}, {valid_elev.max():.1f}] m")
                        print(f"  Unique values: {len(np.unique(valid_elev))}")
                except Exception as e:
                    print(f"  Warning: Error in elevation lookup: {e}")
                    traceback.print_exc()
            else:
                print("Warning: Elevation data not loaded")

            # PFT lookup - using same method as climate data
            if hasattr(self, "pft_raw") and self.pft_raw is not None:
                try:
                    print(f"Getting PFT data for {n_points} points...")
                    pfts = self.get_raster_at_location_direct(point_lons, point_lats, self.pft_raw, max_distance=0.05)
                    pfts = pfts.astype(int)
                    print(f"  PFT classes: {np.unique(pfts).tolist()}")
                    print(f"  Unique values: {len(np.unique(pfts))}")
                except Exception as e:
                    print(f"  Warning: Error in PFT lookup: {e}")
                    traceback.print_exc()
            else:
                print("Warning: PFT data not loaded")

            # Canopy height lookup - using same method as climate data
            if hasattr(self, "canopy_height_raw") and self.canopy_height_raw is not None:
                try:
                    print(f"Getting canopy height data for {n_points} points...")
                    canopy_heights = self.get_raster_at_location_direct(
                        point_lons,
                        point_lats,
                        self.canopy_height_raw,
                        max_distance=0.05,
                    )
                    valid_ch = canopy_heights[~np.isnan(canopy_heights)]
                    if len(valid_ch) > 0:
                        print(f"  Canopy height range: [{valid_ch.min():.1f}, {valid_ch.max():.1f}] m")
                        print(f"  Unique values: {len(np.unique(valid_ch))}")
                except Exception as e:
                    print(f"  Warning: Error in canopy height lookup: {e}")
                    traceback.print_exc()
            else:
                print("Warning: Canopy height data not loaded")

            # ============================================
            # LAI DATA LOOKUP (time-varying) - using same approach as climate data
            # ============================================
            # LAI needs special handling because it varies with time
            # We'll extract LAI time series for each point
            lai_point_data = {}  # Dict mapping point index to LAI time series

            if hasattr(self, "lai_raw") and self.lai_raw is not None:
                try:
                    print(f"Getting LAI data for {n_points} points...")
                    lai_data = self.lai_raw["data"]  # Shape: (n_times, height, width)
                    lai_times = self.lai_raw["times"]
                    n_lai_times = len(lai_times)

                    print(f"  LAI has {n_lai_times} time slices")

                    # Extract LAI at each point for each LAI time
                    for t_idx in range(n_lai_times):
                        lai_slice_dict = {
                            "data": lai_data[t_idx],
                            "transform": self.lai_raw["transform"],
                            "nodata": self.lai_raw.get("nodata", np.nan),
                        }

                        # Use same method as climate data
                        lai_values_t = self.get_raster_at_location_direct(
                            point_lons, point_lats, lai_slice_dict, max_distance=0.05, default_value=np.nan
                        )

                        # Store for each point
                        for p_idx in range(n_points):
                            if p_idx not in lai_point_data:
                                lai_point_data[p_idx] = {"times": [], "values": []}
                            lai_point_data[p_idx]["times"].append(lai_times[t_idx])
                            lai_point_data[p_idx]["values"].append(lai_values_t[p_idx])

                        # Print debug info for first time slice
                        if t_idx == 0:
                            valid_lai = lai_values_t[~np.isnan(lai_values_t)]
                            if len(valid_lai) > 0:
                                print(
                                    f"    First time slice: range=[{valid_lai.min():.2f}, {valid_lai.max():.2f}], "
                                    f"unique={len(np.unique(valid_lai))}"
                                )

                    print(f"  LAI data extracted for {n_lai_times} time slices")

                except Exception as e:
                    print(f"  Warning: Error in LAI lookup: {e}")
                    traceback.print_exc()
            else:
                print("Warning: LAI data not loaded")

            # ============================================
            # PROCESS EACH POINT
            # ============================================
            for i, point in enumerate(points):
                point_data_dict = {}
                point_series_list = []
                point_lat = point["lat"]
                point_lon = point["lon"]
                point_name = point.get("name", f"Point_{point_lat}_{point_lon}")

                print(f"\nProcessing point {i + 1}/{n_points}: ({point_lat}, {point_lon}) - {point_name}")

                # Basic info
                point_data_dict["name"] = point_name
                point_data_dict["latitude"] = point_lat
                point_data_dict["longitude"] = point_lon

                # Climate data
                if climate_data_available and temps is not None and precips is not None:
                    point_data_dict["mean_annual_temp"] = float(temps[i])
                    point_data_dict["mean_annual_precip"] = float(precips[i])
                else:
                    point_data_dict["mean_annual_temp"] = 15.0  # Default
                    point_data_dict["mean_annual_precip"] = 800.0  # Default

                # Static data
                if elevations is not None:
                    point_data_dict["elevation"] = float(elevations[i])
                else:
                    point_data_dict["elevation"] = 0.0  # Default

                if pfts is not None:
                    point_data_dict["pft"] = int(pfts[i])
                else:
                    point_data_dict["pft"] = 11  # Barren as default

                if canopy_heights is not None:
                    point_data_dict["canopy_height"] = float(canopy_heights[i])
                else:
                    point_data_dict["canopy_height"] = 0.0  # Default

                # ============================================
                # Extract ERA5 time series for this point
                # ============================================
                common_time_index = None

                for original_var_name in original_vars_to_process:
                    model_name = prediction_vars_map[original_var_name]
                    if original_var_name not in self.datasets:
                        print(f"  Warning: Dataset for '{original_var_name}' not found. Skipping.")
                        continue

                    try:
                        ds = self.datasets[original_var_name]

                        # Find data variable
                        data_var_key = None
                        potential_keys = [k for k in ds.data_vars if k not in ds.coords]
                        if len(potential_keys) == 1:
                            data_var_key = potential_keys[0]
                        elif model_name in potential_keys:
                            data_var_key = model_name
                        elif original_var_name in potential_keys:
                            data_var_key = original_var_name
                        else:
                            if potential_keys:
                                data_var_key = potential_keys[0]

                        if not data_var_key:
                            print(f"  Warning: Could not find data variable for '{original_var_name}'. Skipping.")
                            continue

                        # Find coordinates
                        lat_coord_point, lon_coord_point, time_dim_point = None, None, None
                        for name in list(ds.coords) + list(ds.dims):
                            lname = name.lower()
                            if lname in ["lat", "latitude"] and lat_coord_point is None:
                                lat_coord_point = name
                            if lname in ["lon", "longitude"] and lon_coord_point is None:
                                lon_coord_point = name
                            if lname in ["time", "datetime"] and time_dim_point is None:
                                time_dim_point = name

                        if not all([lat_coord_point, lon_coord_point, time_dim_point]):
                            print(f"  Warning: Missing coordinates for '{original_var_name}'. Skipping.")
                            continue

                        # Extract time series at point using nearest neighbor
                        ts_point = ds[data_var_key].sel(
                            {lat_coord_point: point_lat, lon_coord_point: point_lon}, method="nearest"
                        )

                        # Standardize times
                        current_times_pd = pd.DatetimeIndex(ts_point[time_dim_point].values)
                        current_times_std = self.standardize_timestamps(current_times_pd, use_utc=use_utc)

                        if current_times_std is None or current_times_std.empty:
                            print(f"  Warning: Time standardization failed for '{original_var_name}'. Skipping.")
                            continue

                        ts_point = ts_point.assign_coords({time_dim_point: current_times_std})

                        # Apply time slice
                        time_slice_point = slice(
                            start_date_std if start_date_std else None, end_date_std if end_date_std else None
                        )

                        if time_slice_point.start is not None or time_slice_point.stop is not None:
                            ts_point_filtered = ts_point.sel({time_dim_point: time_slice_point})
                        else:
                            ts_point_filtered = ts_point

                        if ts_point_filtered.size == 0:
                            print(f"  Warning: No data for '{original_var_name}' after time slicing. Skipping.")
                            continue

                        # Convert to Series
                        times_final_point = self.standardize_timestamps(
                            pd.DatetimeIndex(ts_point_filtered[time_dim_point].values), use_utc=use_utc
                        )
                        values_point = ts_point_filtered.values

                        series = pd.Series(values_point, index=times_final_point, name=model_name)

                        if common_time_index is None and not series.empty:
                            common_time_index = series.index

                        point_series_list.append(series)

                    except Exception as e:
                        print(f"  Error extracting '{original_var_name}' for point {point_name}: {e}")
                        traceback.print_exc()

                # ============================================
                # Create LAI time series for this point
                # ============================================
                if i in lai_point_data and common_time_index is not None:
                    try:
                        lai_times = lai_point_data[i]["times"]
                        lai_values = lai_point_data[i]["values"]

                        # Create LAI series aligned to common_time_index using nearest neighbor
                        lai_times_pd = pd.DatetimeIndex(lai_times)

                        # Make timezone consistent
                        if common_time_index.tz is not None and lai_times_pd.tz is None:
                            lai_times_pd = lai_times_pd.tz_localize(common_time_index.tz)
                        elif common_time_index.tz is None and lai_times_pd.tz is not None:
                            lai_times_pd = lai_times_pd.tz_localize(None)

                        # Create temporary series with LAI times
                        lai_series_raw = pd.Series(lai_values, index=lai_times_pd)

                        # Align to common time index using nearest neighbor
                        lai_aligned = []
                        for target_time in common_time_index:
                            # Find nearest LAI time
                            time_diffs = np.abs((lai_times_pd - target_time).total_seconds())
                            nearest_idx = np.argmin(time_diffs)
                            lai_aligned.append(lai_values[nearest_idx])

                        lai_series = pd.Series(lai_aligned, index=common_time_index, name="LAI")
                        point_series_list.append(lai_series)

                    except Exception as e:
                        print(f"  Warning: Could not create LAI series for point {point_name}: {e}")

                # ============================================
                # Create DataFrame for this point
                # ============================================
                if point_series_list and common_time_index is not None and not common_time_index.empty:
                    try:
                        point_df = pd.DataFrame(index=common_time_index)

                        for s in point_series_list:
                            if s.name in point_df.columns:
                                print(f"  Warning: Duplicate column '{s.name}'. Overwriting.")
                            point_df[s.name] = s.reindex(
                                common_time_index, method="nearest", tolerance=pd.Timedelta("1 hour")
                            )

                        # Add scalar data (broadcast to all rows)
                        for key, value in point_data_dict.items():
                            point_df[key] = value

                        point_df.index.name = "timestamp"

                        # Add time features if configured
                        time_feature_config = getattr(config, "TIME_FEATURES", False)
                        if time_feature_config and isinstance(point_df.index, pd.DatetimeIndex):
                            point_df = self.add_time_features(point_df)

                        # Reset index
                        point_df = point_df.reset_index()
                        all_data.append(point_df)

                        print(f"  Created DataFrame with {len(point_df)} rows, columns: {point_df.columns.tolist()}")

                    except Exception as e:
                        print(f"  Error creating DataFrame for point {point_name}: {e}")
                        traceback.print_exc()
                else:
                    print(f"  No valid data for point {point_name}")

            print(f"\nFinished processing {len(points)} points. Created {len(all_data)} DataFrames.")
            return all_data if all_data else None

        else:
            # --- GRID-BASED APPROACH with xr.merge optimization ---
            print("Processing gridded data...")
            start_grid_proc_time = time.time()

            # --- 1. Find a reference dataset and validate coordinates ---
            reference_ds_key = next((var for var in original_vars_to_process if var in self.datasets), None)
            if not reference_ds_key:
                print("Error: Could not find any valid datasets for requested prediction variables.")
                return None

            ref_ds_initial = self.datasets[reference_ds_key]

            # Validate coordinate names and properties robustly
            ref_lat_var, ref_lon_var, ref_time_var = None, None, None
            for var_name in list(ref_ds_initial.coords) + list(ref_ds_initial.dims):
                low_var = var_name.lower()
                if low_var in ["latitude", "lat"] and ref_lat_var is None:
                    ref_lat_var = var_name
                elif low_var in ["longitude", "lon"] and ref_lon_var is None:
                    ref_lon_var = var_name
                elif low_var in ["time", "datetime"] and ref_time_var is None:
                    ref_time_var = var_name

            if not all([ref_lat_var, ref_lon_var, ref_time_var]):
                print(
                    f"Error: Could not identify standard coordinate variables in reference dataset '{reference_ds_key}' (found lat: {ref_lat_var}, lon: {ref_lon_var}, time: {ref_time_var})"
                )
                return None

            # Get coordinates from the reference dataset
            try:
                ref_lats = ref_ds_initial[ref_lat_var].values
                ref_lons = ref_ds_initial[ref_lon_var].values
                ref_times_pd = pd.DatetimeIndex(ref_ds_initial[ref_time_var].values)
                ref_times_std = self.standardize_timestamps(ref_times_pd, use_utc=use_utc)

                if ref_times_std is None or ref_times_std.empty:
                    raise ValueError("Standardized time index for reference dataset is empty or None")

            except Exception as ref_coord_err:
                print(f"Error accessing coordinates from reference dataset '{reference_ds_key}': {ref_coord_err}")
                traceback.print_exc()
                return None

            print(f"\nReference Grid Coordinates ({reference_ds_key}):")
            print(
                f"  Latitude ({ref_lat_var}): {len(ref_lats)} points, range [{ref_lats.min():.4f}, {ref_lats.max():.4f}], order: {'descending' if ref_lats[0] > ref_lats[-1] else 'ascending'}"
            )
            print(
                f"  Longitude ({ref_lon_var}): {len(ref_lons)} points, range [{ref_lons.min():.4f}, {ref_lons.max():.4f}], order: {'ascending' if ref_lons[0] < ref_lons[-1] else 'descending'}"
            )
            print(
                f"  Time ({ref_time_var}): {len(ref_times_std)} points, range [{ref_times_std.min()}, {ref_times_std.max()}] (standardized)"
            )

            if ref_lats[0] < ref_lats[-1]:  # Check for ascending latitude
                print(
                    "  WARNING: Reference latitude is in ascending order. Ensure all datasets are handled consistently."
                )

            # --- 2. Define Time Slice ---
            # Use standardized dates determined earlier
            time_slice = slice(start_date_std if start_date_std else None, end_date_std if end_date_std else None)
            print(f"Requested Time slice: {time_slice.start} to {time_slice.stop}")

            # --- 3. Get Climate Data (Vectorized) ---
            temp_grid_da, precip_grid_da = None, None  # Store as DataArrays
            # Standard coordinate names for climate data
            final_lat_coord_name = "latitude"
            final_lon_coord_name = "longitude"

            if (
                hasattr(self, "temp_climate_data")
                and self.temp_climate_data is not None
                and hasattr(self, "precip_climate_data")
                and self.precip_climate_data is not None
            ):
                try:
                    # Use the reference coordinates to create the meshgrid
                    lon_grid, lat_grid = np.meshgrid(ref_lons, ref_lats)
                    flat_lons, flat_lats = lon_grid.flatten(), lat_grid.flatten()
                    print(f"Getting climate data for {len(flat_lons)} grid points...")

                    temps_flat, precips_flat = self.get_climate_at_location_direct(
                        flat_lons, flat_lats, self.temp_climate_data, self.precip_climate_data
                    )
                    # Reshape back to grid matching reference lat/lon order
                    temp_grid = temps_flat.reshape(len(ref_lats), len(ref_lons))
                    precip_grid = precips_flat.reshape(len(ref_lats), len(ref_lons))

                    # Create DataArrays using reference coordinates
                    temp_grid_da = xr.DataArray(
                        temp_grid,
                        dims=[final_lat_coord_name, final_lon_coord_name],
                        coords={final_lat_coord_name: ref_lats, final_lon_coord_name: ref_lons},
                        name="mean_annual_temp",
                    )
                    precip_grid_da = xr.DataArray(
                        precip_grid,
                        dims=[final_lat_coord_name, final_lon_coord_name],
                        coords={final_lat_coord_name: ref_lats, final_lon_coord_name: ref_lons},
                        name="mean_annual_precip",
                    )
                    print("Successfully created climate DataArrays aligned with reference grid.")

                except Exception as e:
                    print(
                        f"Warning: Error in grid climate data retrieval or DataArray creation: {e}. Climate data will be missing."
                    )
                    traceback.print_exc()
            else:
                print("Warning: Climate data attributes not found or not loaded. Climate data will be missing.")

            # --- 4. Prepare, Align, and Filter Datasets for Merging ---
            print("\nPreparing and aligning variable datasets...")
            aligned_datasets_to_merge = []
            reference_ds_for_alignment = None  # Store the first fully processed dataset for alignment target
            final_time_coord_name = "timestamp"  # Target standard time name

            for original_var_name in original_vars_to_process:
                if original_var_name not in self.datasets:
                    print(f"Info: Dataset for '{original_var_name}' not found. Skipping.")
                    continue

                ds = self.datasets[original_var_name].copy()  # Work on a copy

                # Find current coordinate names in this dataset
                current_lat_var, current_lon_var, current_time_var = None, None, None
                # (Use robust finding logic similar to reference dataset identification)
                for var_name in list(ds.coords) + list(ds.dims):
                    low_var = var_name.lower()
                    if low_var in ["latitude", "lat"] and current_lat_var is None:
                        current_lat_var = var_name
                    elif low_var in ["longitude", "lon"] and current_lon_var is None:
                        current_lon_var = var_name
                    elif low_var in ["time", "datetime"] and current_time_var is None:
                        current_time_var = var_name

                if not all([current_lat_var, current_lon_var, current_time_var]):
                    print(
                        f"Warning: Could not identify standard coordinates in dataset '{original_var_name}'. Skipping."
                    )
                    continue

                try:
                    # --- 4b. Apply Time Slice ---
                    ds_filtered = self._safe_time_slice(
                        ds,
                        current_time_var,
                        start_date_std,  # These come from earlier in the method
                        end_date_std,
                        use_utc=use_utc,
                    )
                    if ds_filtered[current_time_var].size == 0:
                        print(f"Info: No data remaining for '{original_var_name}' after time slicing. Skipping.")
                        continue

                    # --- 4c. Rename Coordinates to Standard Names ---
                    rename_coords = {}
                    if current_lat_var != final_lat_coord_name:
                        rename_coords[current_lat_var] = final_lat_coord_name
                    if current_lon_var != final_lon_coord_name:
                        rename_coords[current_lon_var] = final_lon_coord_name
                    if current_time_var != final_time_coord_name:
                        rename_coords[current_time_var] = final_time_coord_name
                    if rename_coords:
                        ds_renamed = ds_filtered.rename(rename_coords)
                    else:
                        ds_renamed = ds_filtered

                    # --- 4d. Select ONLY the data variable needed ---
                    model_name = prediction_vars_map[original_var_name]
                    data_var_key = None
                    potential_keys = [k for k in ds_renamed.data_vars if k not in ds_renamed.coords]
                    if len(potential_keys) == 1:
                        data_var_key = potential_keys[0]
                    # Check if the *original* name (before potential rename in prediction_vars_map) exists
                    elif original_var_name in potential_keys:
                        data_var_key = original_var_name
                    # Check if the *model* name exists (could be a derived variable)
                    elif model_name in potential_keys:
                        data_var_key = model_name
                    else:  # Fallback
                        if potential_keys:
                            data_var_key = potential_keys[0]
                            print(
                                f"Warning: Could not definitively identify data variable for '{original_var_name}', using first found: '{data_var_key}'"
                            )

                    if not data_var_key:
                        print(
                            f"Warning: Could not find data variable key for '{original_var_name}' after processing. Skipping."
                        )
                        continue

                    # Ensure standard coords are set, keep only the target data var
                    ds_final_var = ds_renamed[[data_var_key]].set_coords(
                        [final_lat_coord_name, final_lon_coord_name, final_time_coord_name]
                    )

                    # --- 4e. Align to Reference Dataset ---
                    if reference_ds_for_alignment is None:
                        # This is the first dataset, use it as the reference for alignment
                        print(f"Using processed '{original_var_name}' as reference for grid alignment.")
                        reference_ds_for_alignment = ds_final_var
                        aligned_datasets_to_merge.append(reference_ds_for_alignment)
                    else:
                        # Align this dataset to the reference grid/time
                        print(f"Aligning '{original_var_name}' to reference grid...")
                        ds_aligned = ds_final_var.reindex_like(
                            reference_ds_for_alignment,
                            method="nearest",  # Use 'nearest' or 'linear' for interpolation if needed
                            tolerance=1e-6,  # Adjust tolerance as needed for float coords
                        )
                        aligned_datasets_to_merge.append(ds_aligned)

                except Exception as prep_err:
                    print(f"Warning: Could not prepare/align dataset '{original_var_name}': {prep_err}. Skipping.")
                    traceback.print_exc()

            # --- 5. Perform the Optimized Merge ---
            if not aligned_datasets_to_merge:
                print("Error: No datasets available for merging after preparation and alignment.")
                return None

            print(f"\nAttempting to merge {len(aligned_datasets_to_merge)} aligned datasets using 'exact' join...")
            merged_ds = None
            try:
                # Use 'exact' join since coordinates should now match perfectly
                merged_ds = xr.merge(aligned_datasets_to_merge, join="exact", compat="equals")  # Be strict
                print(f"Successfully merged datasets. Variables: {list(merged_ds.data_vars)}")
                print(f"Merged dataset coords: {list(merged_ds.coords)}")

            except Exception as merge_err:
                print(f"Error merging datasets: {merge_err}")
                print("--- Details of datasets attempted to merge (after alignment) ---")
                for i, dsm in enumerate(aligned_datasets_to_merge):
                    print(f"Dataset {i}: Vars={list(dsm.data_vars)}")
                    # Print coordinate details to help diagnose mismatch
                    try:
                        print(
                            f"  Lat coords: {dsm[final_lat_coord_name].values.min()} to {dsm[final_lat_coord_name].values.max()}, size {dsm[final_lat_coord_name].size}"
                        )
                        print(
                            f"  Lon coords: {dsm[final_lon_coord_name].values.min()} to {dsm[final_lon_coord_name].values.max()}, size {dsm[final_lon_coord_name].size}"
                        )
                        print(
                            f"  Time coords: {dsm[final_time_coord_name].values.min()} to {dsm[final_time_coord_name].values.max()}, size {dsm[final_time_coord_name].size}"
                        )
                    except Exception as coord_err:
                        print(f"  Error printing coords for dataset {i}: {coord_err}")
                print("-----------------------------------------------")
                traceback.print_exc()
                return None

            # --- 6. Add Climate Data (if available) ---
            if temp_grid_da is not None and precip_grid_da is not None:
                print("Adding climate data to merged dataset...")
                try:
                    # Climate DAs were already created using reference coords, should align
                    merged_ds["mean_annual_temp"] = temp_grid_da
                    merged_ds["mean_annual_precip"] = precip_grid_da
                except Exception as climate_add_err:
                    print(f"Warning: Error adding climate data after merge: {climate_add_err}")
                    traceback.print_exc()
            # --- 6b. Add Static Datasets (elevation, PFT, canopy height, LAI) ---
            print("\nAdding static datasets to merged dataset...")

            time_coord = "timestamp" if "timestamp" in merged_ds.dims else "time"
            has_time = time_coord in merged_ds.dims
            n_times = len(merged_ds[time_coord]) if has_time else 1

            # Get the EXACT coordinates from merged dataset
            merged_lats = merged_ds["latitude"].values
            merged_lons = merged_ds["longitude"].values
            n_lats = len(merged_lats)
            n_lons = len(merged_lons)

            print(f"  Merged dataset dimensions: time={n_times}, lat={n_lats}, lon={n_lons}")
            print(f"  Merged lat range: [{merged_lats.min():.4f}, {merged_lats.max():.4f}]")
            print(f"  Merged lon range: [{merged_lons.min():.4f}, {merged_lons.max():.4f}]")

            # Track which static variables were successfully added
            static_vars_added = []

            def add_static_var_to_merged(raw_data_dict, var_name, default_value=np.nan, is_categorical=False):
                """
                Add static data to merged dataset using the SAME lookup approach as climate data.
                This ensures coordinates match exactly.

                Parameters:
                -----------
                raw_data_dict : dict
                    Raw data dictionary with 'data', 'transform', etc. (same format as climate data)
                var_name : str
                    Name of the variable
                default_value : float
                    Default value for missing data
                is_categorical : bool
                    If True, convert to int after lookup

                Returns:
                --------
                bool
                    True if successfully added
                """
                if raw_data_dict is None:
                    print(f"  Skipping {var_name}: raw data not available")
                    return False

                try:
                    print(f"  Processing {var_name}...")

                    # Create meshgrid from merged dataset coordinates (SAME AS CLIMATE DATA APPROACH)
                    lon_grid, lat_grid = np.meshgrid(merged_lons, merged_lats)
                    flat_lons = lon_grid.flatten()
                    flat_lats = lat_grid.flatten()

                    # Use the same lookup method as climate data
                    values = self.get_raster_at_location_direct(
                        flat_lons, flat_lats, raw_data_dict, max_distance=0.05, default_value=default_value
                    )

                    # Reshape to grid
                    values_grid = values.reshape(n_lats, n_lons)

                    if is_categorical:
                        values_grid = np.nan_to_num(values_grid, nan=default_value).astype(int)

                    # Broadcast to time dimension if needed
                    if has_time:
                        broadcast_values = np.broadcast_to(
                            values_grid[np.newaxis, :, :], (n_times, n_lats, n_lons)
                        ).copy()  # .copy() to make it writable

                        data_array = xr.DataArray(
                            broadcast_values,
                            dims=[time_coord, "latitude", "longitude"],
                            coords={
                                time_coord: merged_ds[time_coord],
                                "latitude": merged_lats,
                                "longitude": merged_lons,
                            },
                            name=var_name,
                        )
                    else:
                        data_array = xr.DataArray(
                            values_grid,
                            dims=["latitude", "longitude"],
                            coords={"latitude": merged_lats, "longitude": merged_lons},
                            name=var_name,
                        )

                    # Add to merged dataset
                    merged_ds[var_name] = data_array

                    # Print validation info
                    valid_values = values_grid[~np.isnan(values_grid)] if not is_categorical else values_grid.flatten()
                    if len(valid_values) > 0:
                        if is_categorical:
                            unique_vals = np.unique(valid_values)
                            print(
                                f"    Added {var_name}: shape={data_array.shape}, classes={unique_vals.tolist()[:10]}{'...' if len(unique_vals) > 10 else ''}"
                            )
                        else:
                            print(
                                f"    Added {var_name}: shape={data_array.shape}, range=[{valid_values.min():.2f}, {valid_values.max():.2f}], unique={len(np.unique(valid_values))}"
                            )
                    else:
                        print(f"    Added {var_name}: shape={data_array.shape}, WARNING: no valid values!")

                    return True

                except Exception as e:
                    print(f"  Warning: Could not add {var_name}: {e}")
                    traceback.print_exc()
                    return False

            def add_lai_to_merged(lai_raw_dict):
                """
                Add LAI data to merged dataset with temporal alignment.
                Uses the SAME lookup approach as climate data.

                Parameters:
                -----------
                lai_raw_dict : dict
                    Raw LAI data dictionary with 'data' (3D: time, lat, lon), 'times', 'transform'

                Returns:
                --------
                bool
                    True if successfully added
                """
                if lai_raw_dict is None:
                    print("  Skipping LAI: raw data not available")
                    return False

                try:
                    print("  Processing LAI (time-varying)...")

                    lai_data = lai_raw_dict["data"]  # Shape: (n_lai_times, height, width)
                    lai_times = lai_raw_dict["times"]
                    n_lai_times = len(lai_times)

                    print(f"    LAI has {n_lai_times} time slices")

                    # Create meshgrid from merged dataset coordinates
                    lon_grid, lat_grid = np.meshgrid(merged_lons, merged_lats)
                    flat_lons = lon_grid.flatten()
                    flat_lats = lat_grid.flatten()

                    # Resample each LAI time slice spatially
                    lai_resampled_times = []
                    for t_idx in range(n_lai_times):
                        # Create temporary dict for this time slice
                        lai_slice_dict = {
                            "data": lai_data[t_idx],
                            "transform": lai_raw_dict["transform"],
                            "nodata": lai_raw_dict.get("nodata", np.nan),
                        }

                        # Use same lookup method as climate data
                        lai_values = self.get_raster_at_location_direct(
                            flat_lons, flat_lats, lai_slice_dict, max_distance=0.05, default_value=np.nan
                        )
                        lai_grid = lai_values.reshape(n_lats, n_lons)
                        lai_resampled_times.append(lai_grid)

                        # Debug first slice
                        if t_idx == 0:
                            valid_lai = lai_grid[~np.isnan(lai_grid)]
                            if len(valid_lai) > 0:
                                print(
                                    f"    First time slice: range=[{valid_lai.min():.2f}, {valid_lai.max():.2f}], unique={len(np.unique(valid_lai))}"
                                )

                    lai_resampled_stack = np.stack(lai_resampled_times, axis=0)
                    print(f"    LAI resampled stack shape: {lai_resampled_stack.shape}")

                    # Temporal alignment to merged dataset times
                    if has_time:
                        merged_times = merged_ds[time_coord].values
                        merged_times_pd = pd.DatetimeIndex(merged_times)

                        # Make timezone naive for comparison
                        if merged_times_pd.tz is not None:
                            merged_times_pd = merged_times_pd.tz_localize(None)

                        lai_times_pd = pd.DatetimeIndex(lai_times)
                        if lai_times_pd.tz is not None:
                            lai_times_pd = lai_times_pd.tz_localize(None)

                        print(f"    Aligning LAI to {len(merged_times_pd)} merged timestamps...")

                        # Find nearest LAI time for each merged time
                        lai_indices = []
                        for target_time in merged_times_pd:
                            time_diffs = np.abs((lai_times_pd - target_time).total_seconds())
                            nearest_idx = np.argmin(time_diffs)
                            lai_indices.append(nearest_idx)

                        # Create aligned LAI array
                        lai_aligned = lai_resampled_stack[lai_indices, :, :]

                        lai_da = xr.DataArray(
                            lai_aligned,
                            dims=[time_coord, "latitude", "longitude"],
                            coords={
                                time_coord: merged_ds[time_coord],
                                "latitude": merged_lats,
                                "longitude": merged_lons,
                            },
                            name="LAI",
                        )
                    else:
                        # No time dimension - use mean of all LAI times
                        lai_mean = np.nanmean(lai_resampled_stack, axis=0)
                        lai_da = xr.DataArray(
                            lai_mean,
                            dims=["latitude", "longitude"],
                            coords={"latitude": merged_lats, "longitude": merged_lons},
                            name="LAI",
                        )

                    lai_da.attrs["units"] = "m2/m2"
                    lai_da.attrs["long_name"] = "Leaf Area Index"

                    # Add to merged dataset
                    merged_ds["LAI"] = lai_da

                    # Print validation info
                    valid_lai = lai_da.values[~np.isnan(lai_da.values)]
                    if len(valid_lai) > 0:
                        print(
                            f"    Added LAI: shape={lai_da.shape}, range=[{valid_lai.min():.2f}, {valid_lai.max():.2f}], unique={len(np.unique(valid_lai))}"
                        )
                    else:
                        print(f"    Added LAI: shape={lai_da.shape}, WARNING: no valid values!")

                    return True

                except Exception as e:
                    print(f"  Warning: Could not add LAI: {e}")
                    traceback.print_exc()
                    return False

            # ========== ADD ELEVATION ==========
            if hasattr(self, "elevation_raw") and self.elevation_raw is not None:
                if add_static_var_to_merged(self.elevation_raw, "elevation", default_value=0.0):
                    static_vars_added.append("elevation")
            else:
                print("  Skipping elevation: raw data not loaded")

            # ========== ADD PFT ==========
            if hasattr(self, "pft_raw") and self.pft_raw is not None:
                if add_static_var_to_merged(self.pft_raw, "pft", default_value=0, is_categorical=True):
                    static_vars_added.append("pft")
            else:
                print("  Skipping pft: raw data not loaded")

            # ========== ADD CANOPY HEIGHT ==========
            if hasattr(self, "canopy_height_raw") and self.canopy_height_raw is not None:
                if add_static_var_to_merged(self.canopy_height_raw, "canopy_height", default_value=0.0):
                    static_vars_added.append("canopy_height")
            else:
                print("  Skipping canopy_height: raw data not loaded")

            # ========== ADD LAI ==========
            if hasattr(self, "lai_raw") and self.lai_raw is not None:
                if add_lai_to_merged(self.lai_raw):
                    static_vars_added.append("LAI")
            else:
                print("  Skipping LAI: raw data not loaded")

            print(f"\nStatic variables added to merged dataset: {static_vars_added}")
            # --- 6c. Propagate NaN mask from ERA5 variables to static variables ---
            print("\nPropagating NaN mask from ERA5 data to static variables...")

            # Find a reference ERA5 variable that has the NaN mask (from shapefile clipping)
            era5_reference_var = None
            for var_name in [
                "ta",
                "temperature_2m",
                "ws",
                "wind_speed",
                "sw_in",
                "surface_solar_radiation_downwards_hourly",
            ]:
                if var_name in merged_ds.data_vars:
                    era5_reference_var = var_name
                    break

            if era5_reference_var is not None:
                # Get the NaN mask from ERA5 data
                era5_data = merged_ds[era5_reference_var].values
                nan_mask = np.isnan(era5_data)
                nan_count = np.sum(nan_mask)
                print(f"  Reference variable: {era5_reference_var}")
                print(f"  NaN pixels to propagate: {nan_count}")

                # Apply mask to static variables
                static_vars_to_mask = ["elevation", "pft", "canopy_height", "LAI"]

                for static_var in static_vars_to_mask:
                    if static_var in merged_ds.data_vars:
                        # Get current values
                        static_data = merged_ds[static_var].values.copy()

                        # Handle dimension mismatch (static may be 2D, ERA5 is 3D with time)
                        if static_data.ndim == 2 and nan_mask.ndim == 3:
                            # Use mask from first timestep (mask should be same for all times)
                            mask_2d = nan_mask[0, :, :]
                            static_data = np.where(mask_2d, np.nan, static_data)
                        elif static_data.ndim == 3 and nan_mask.ndim == 3:
                            # Same dimensions - apply directly
                            static_data = np.where(nan_mask, np.nan, static_data)
                        else:
                            # Broadcast mask to match static data shape
                            static_data = np.where(nan_mask, np.nan, static_data)

                        # Update the dataset
                        merged_ds[static_var].values = static_data
                        new_nan_count = np.sum(np.isnan(static_data))
                        print(f"  {static_var}: Applied mask, now has {new_nan_count} NaN values")
            else:
                print("  WARNING: Could not find ERA5 reference variable for NaN mask propagation")
            # --- 7. Rename variables to final model names ---
            print("Renaming variables to final model names...")
            rename_dict_final = {}
            final_expected_vars = []
            checked_var = []
            # Add climate variables
            if "mean_annual_temp" in merged_ds.data_vars:
                final_expected_vars.append("mean_annual_temp")
            if "mean_annual_precip" in merged_ds.data_vars:
                final_expected_vars.append("mean_annual_precip")

            # Add static variables that were successfully added
            for static_var in static_vars_added:
                if static_var not in final_expected_vars:
                    final_expected_vars.append(static_var)

            # Process ERA5 variables from prediction_vars_map
            for original_name, model_name in prediction_vars_map.items():
                possible_names = [original_name, f"{original_name}_hourly", f"{original_name}_sum"]

                for var_to_check in possible_names:
                    if var_to_check in merged_ds.data_vars:
                        checked_var.append(var_to_check)

                        if var_to_check != model_name:
                            rename_dict_final[var_to_check] = model_name
                        if model_name not in final_expected_vars:
                            final_expected_vars.append(model_name)
                        break
                print(f"Renamed {checked_var} to {final_expected_vars}'")

            print(f"Final rename dictionary: {rename_dict_final}")
            if rename_dict_final:
                merged_ds = merged_ds.rename(rename_dict_final)

            print(f"Expected final variables: {final_expected_vars}")

            # --- 8. Filter dataset to keep only the final required variables ---
            try:
                # Get all data variables currently in the dataset
                all_current_vars = list(merged_ds.data_vars)
                print(f"All variables in merged dataset: {all_current_vars}")

                # Keep variables that are in final_expected_vars
                vars_to_keep_actual = [v for v in final_expected_vars if v in merged_ds.data_vars]

                # Also keep static variables even if not in final_expected_vars
                # (this is a safety check)
                for static_var in ["elevation", "pft", "canopy_height", "LAI"]:
                    if static_var in merged_ds.data_vars and static_var not in vars_to_keep_actual:
                        vars_to_keep_actual.append(static_var)
                        print(f"Note: Adding {static_var} to keep list (was missing from expected)")

                missing_final_vars = [v for v in final_expected_vars if v not in vars_to_keep_actual]
                if missing_final_vars:
                    print(f"Warning: Not all expected final variables found. Missing: {missing_final_vars}")

                if not vars_to_keep_actual:
                    print("Error: No variables left to keep after renaming/filtering.")
                    return None

                merged_ds = merged_ds[vars_to_keep_actual]
                print(f"Variables after final filter: {list(merged_ds.data_vars)}")

            except Exception as filter_err:
                print(f"Error during final variable filtering: {filter_err}")
                print(f"Available variables were: {list(merged_ds.data_vars)}")
                print(f"Attempted to keep: {final_expected_vars}")
                traceback.print_exc()
                return None

            # --- 9. Convert to Pandas DataFrame ---
            print("Converting merged xarray Dataset to Pandas DataFrame...")
            final_df = None
            try:
                # Ensure standard coordinate names are set as coords before conversion
                merged_ds = merged_ds.set_coords([final_lat_coord_name, final_lon_coord_name, final_time_coord_name])

                # Check for potential memory issues before conversion
                estimated_size = merged_ds.nbytes / (1024**3)  # GiB
                print(f"Estimated size of xarray dataset before DataFrame conversion: {estimated_size:.2f} GiB")
                if estimated_size > 4:  # Threshold for warning (adjust as needed)
                    print(
                        "Warning: Dataset size is large, conversion to DataFrame might consume significant memory and time."
                    )

                # to_dataframe() creates a MultiIndex (time, lat, lon) by default if coords are set
                final_df = merged_ds.to_dataframe()
                print("Successfully converted to DataFrame.")

            except Exception as df_conv_err:
                print(
                    f"Error converting to DataFrame: {df_conv_err}. This might be a memory issue or data type problem."
                )
                traceback.print_exc()
                # Consider returning the merged_ds xarray object here if conversion fails
                # return merged_ds
                return None  # Or return None as per original logic

            # Reset the MultiIndex (timestamp, latitude, longitude) to get columns
            final_df = final_df.reset_index()

            # Check if essential columns exist after reset_index
            essential_cols = [final_time_coord_name, final_lat_coord_name, final_lon_coord_name]
            if not all(col in final_df.columns for col in essential_cols):
                print(
                    f"Error: Missing essential coordinate columns after to_dataframe/reset_index. Columns: {final_df.columns.tolist()}"
                )
                # Attempt to find columns with case variations - less likely now with explicit renaming
                # ... (fallback logic as in original code if needed, but should be less necessary) ...
                return None


            # --- 9b. Drop rows with any NaN (ocean / non-forest masked pixels) ---
            rows_before = len(final_df)
            feature_cols = [c for c in final_df.columns if c not in [final_time_coord_name, final_lat_coord_name, final_lon_coord_name]]
            final_df = final_df.dropna(subset=feature_cols, how='any')
            rows_after = len(final_df)
            # Drop rows with negative elevation (SRTM no-data sentinel)
            if "elevation" in final_df.columns:
                final_df = final_df[final_df["elevation"] >= 0]
            print(f'Dropped {rows_before - rows_after} NaN rows ({100*(rows_before - rows_after)/rows_before:.1f}%). Remaining: {rows_after} rows.')

            # --- 10. Add Time Features (if configured) ---
            time_feature_config = getattr(config, "TIME_FEATURES", True)

            if time_feature_config:
                print("Adding solar time features...")

                # Check for required columns BEFORE any index manipulation
                # final_time_coord_name = 'timestamp', final_lon_coord_name = 'longitude'
                has_timestamp = final_time_coord_name in final_df.columns
                has_longitude = final_lon_coord_name in final_df.columns

                if has_timestamp and has_longitude:
                    try:
                        # Ensure timestamp column is datetime type
                        final_df[final_time_coord_name] = pd.to_datetime(final_df[final_time_coord_name])

                        # Call the time features method (it handles index internally)
                        final_df = self.add_time_features(final_df)

                        print("Successfully added solar time features.")

                    except Exception as time_feat_err:
                        print(f"Warning: Could not add time features: {time_feat_err}")
                        traceback.print_exc()
                else:
                    print(
                        f"Warning: Cannot add solar features. "
                        f"timestamp present: {has_timestamp}, longitude present: {has_longitude}"
                    )
                    print(f"Available columns: {final_df.columns.tolist()[:10]}...")  # Debug info

            # --- 11. Add Name column ---
            print("Adding identifier 'name' column...")
            try:
                # Use the standard coordinate names ensured before
                final_df["name"] = (
                    "Grid_"
                    + final_df[final_lat_coord_name].round(4).astype(str)
                    + "_"
                    + final_df[final_lon_coord_name].round(4).astype(str)
                )
            except KeyError as name_err:
                print(f"Error creating 'name' column. Missing coordinate column?: {name_err}")
                print(f"Available columns: {final_df.columns.tolist()}")
                # Don't fail the whole process, just proceed without the name column
            except Exception as general_name_err:
                print(f"Unexpected error creating 'name' column: {general_name_err}")
                traceback.print_exc()

            # --- 12. One-Hot Encode Specific PFT Classes ---
            if "pft" in final_df.columns:
                print("One-hot encoding PFT variable (Target classes only)...")

                # 1. Define ONLY the classes you need
                target_pft_classes = {
                    1: "ENF",  # Evergreen Needleleaf Forests
                    2: "EBF",  # Evergreen Broadleaf Forests
                    3: "DNF",  # Deciduous Needleleaf Forests
                    4: "DBF",  # Deciduous Broadleaf Forests
                    5: "MF",  # Mixed Forests
                    8: "WSA",  # Woody Savannas
                    9: "SAV",  # Savannas
                    11: "WET",  # Permanent Wetlands
                }

                # 2. Clean the PFT data
                # Convert to numeric, turn NaNs to 0, round to nearest int (safety for interpolation artifacts), convert to int
                # Any class NOT in your list (like Urban, Water, Barren) effectively becomes "0" here or just doesn't match below.
                pft_series = pd.to_numeric(final_df["pft"], errors="coerce").fillna(0).round().astype(int)

                # 3. Create One-Hot Columns Manually
                # This guarantees that ALL these columns exist in every single output file,
                # even if a specific day/location doesn't contain that specific tree type.
                for code, label in target_pft_classes.items():
                    col_name = label  # Result: ENF, EBF, etc.
                    # Create binary column: 1 if match, 0 if not
                    final_df[col_name] = (pft_series == code).astype(int)

                # 4. Drop the original raw 'pft' column
                final_df = final_df.drop(columns=["pft"])

                print(
                    f"PFT encoding complete. Created {len(target_pft_classes)} columns: {[f'{v}' for v in target_pft_classes.values()]}"
                )
            # --- 12b. Remove non-forest rows (PFT sum != 1) ---
            pft_cols = ['ENF', 'EBF', 'DNF', 'DBF', 'MF', 'WSA', 'SAV', 'WET']
            existing_pft = [c for c in pft_cols if c in final_df.columns]
            if existing_pft:
                pft_sum = final_df[existing_pft].sum(axis=1)
                rows_before_pft = len(final_df)
                final_df = final_df[pft_sum == 1]
                rows_after_pft = len(final_df)
                print(f'PFT filter: dropped {rows_before_pft - rows_after_pft} non-forest rows (PFT sum != 1). Remaining: {rows_after_pft} rows.')

            # --- 13. Final Checks and Return ---
            end_grid_proc_time = time.time()
            print(f"\nFinished creating grid dataset in {end_grid_proc_time - start_grid_proc_time:.2f} seconds.")

            # Drop unnecessary columns like 'spatial_ref' if they exist from rioxarray/xarray merge artifacts
            if "spatial_ref" in final_df.columns:
                print("Dropping 'spatial_ref' column.")
                final_df = final_df.drop(columns=["spatial_ref"])

            # Final check for NaN values which might indicate issues
            nan_counts = final_df.isnull().sum()
            nan_cols = nan_counts[nan_counts > 0]
            if not nan_cols.empty:
                print("\nNaN counts per column (showing columns with NaNs):")
                print(nan_cols)
                total_rows = len(final_df)
                print(f"Total rows: {total_rows}")
                # Optionally, add checks for percentage of NaNs
                # nan_percentages = (nan_cols / total_rows) * 100
                # print("NaN percentages:\n", nan_percentages)
            else:
                print("\nNo NaN values found in the final DataFrame.")

            print(f"\nCreated prediction dataset with {len(final_df)} rows and {len(final_df.columns)} columns.")
            print(f"Final columns: {final_df.columns.tolist()}")  # Print final columns for verification

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
                single_df = single_df.copy()  # avoid mutating caller's data
                name = single_df.get("name", f"point_{i}").iloc[0] if "name" in single_df.columns else f"point_{i}"
                file_path = output_path.parent / f"{output_path.stem}_{name}{output_path.suffix}"
                if "ta" in single_df.columns:
                    single_df["ta"] = single_df["ta"] - 273.15  # Convert to Celsius
                if "td" in single_df.columns:
                    single_df["td"] = single_df["td"] - 273.15  # Convert to Celsius
                single_df.to_csv(file_path, index=False)
                print(f"Saved dataset to {file_path}")
                saved_paths.append(str(file_path))

            return saved_paths
        elif isinstance(df, pd.DataFrame):
            # Single DataFrame (grid data - the case causing the error)
            # Create a copy to avoid modifying the original DataFrame passed to the function
            df_to_save = df.copy()
            if "ta" in df_to_save.columns:
                print(f"Converting 'ta' column for {output_path.name}...")
                df_to_save["ta"] = df_to_save["ta"] - 273.15  # Convert to Celsius
            else:
                # Only print warning, don't error out
                print(
                    f"Warning: 'ta' column missing in the combined DataFrame. Cannot convert units for {output_path.name}"
                )

            if "td" in df_to_save.columns:
                print(f"Converting 'td' column for {output_path.name}...")
                df_to_save["td"] = df_to_save["td"] - 273.15  # Convert to Celsius
            else:
                # Only print warning, don't error out
                print(
                    f"Warning: 'td' column missing in the combined DataFrame. Cannot convert units for {output_path.name}"
                )
            df_to_save.to_csv(output_path, index=False)
            print(f"Saved dataset to {output_path}")
            return str(output_path)

    def process_single_day(
        self, variables, year, month, day, shapefile=None, output_dir=None, hourly_parallel=False, num_workers=None
    ):
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
                variables, year, month, day, shapefile=shapefile, num_workers=num_workers, output_dir=output_dir
            )

        # Original single-day processing (non-parallel)
        try:
            # Load data for this day
            self.load_variable_data(variables, year, month, day, shapefile=shapefile)

            # Calculate derived variables
            self.calculate_derived_variables()

            # Create prediction dataset with explicit date range filtering
            start_date = pd.Timestamp(year=year, month=month, day=day, tz="UTC")
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

    def _process_day_with_hourly_parallel(
        self, variables, year, month, day, shapefile=None, num_workers=None, output_dir=None
    ):
        """
        Parallel orchestrator for a single day.
        Fixes Fix #4 (No resizing) and Fix #7 (Disk-based transfer).
        """
        import gc
        import multiprocessing as mp

        # 1. Setup Parallelism
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)

        print(f"\n{'=' * 60}")
        print(f"PARALLEL DAY PROCESSOR: {year}-{month:02d}-{day:02d}")
        print(f"Workers: {num_workers} | Temp Dir: {self.temp_dir}")
        print(f"{'=' * 60}\n")

        # 2. Prepare Worker Arguments
        process_args = []
        for hour in range(24):
            process_args.append((year, month, day, hour, variables, shapefile, str(output_dir), str(self.temp_dir)))

        # 3. Execute Pool
        # We use 'spawn' or 'forkserver' on Linux for cleaner memory if available
        try:
            ctx = mp.get_context("spawn")  # Enforce isolation (Good for GEE)
        except AttributeError:
            ctx = mp  # Fallback for older Pythons

        # PASS global_worker_init HERE
        with ctx.Pool(processes=num_workers, initializer=global_worker_init) as pool:
            results = pool.map(_hour_processing_worker, process_args)

        # 4. Collect successfully produced file paths
        # We sort by hour to ensure chronological order before merging
        successful_results = sorted([r for r in results if r.get("success")], key=lambda x: x["hour"])
        temp_files = [r["file_path"] for r in successful_results]

        if not temp_files:
            print("ERROR: No hourly files were produced. Check GEE authentication/connectivity.")
            return None

        print(f"\nMerging {len(temp_files)} hourly segments...")

        try:
            # 5. Merge DataFrames from Disk
            # We use a generator to keep memory usage low during concatenation
            def frame_generator(files):
                for f in files:
                    yield pd.read_csv(f)

            combined_df = pd.concat(frame_generator(temp_files), ignore_index=True)

            # Ensure timestamp is datetime type for sorting/saving
            if "timestamp" in combined_df.columns:
                combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"])

            # 6. Final Sort: Primary by timestamp, secondary by location
            sort_cols = []
            if "timestamp" in combined_df.columns:
                sort_cols.append("timestamp")
            if "name" in combined_df.columns:
                sort_cols.append("name")

            if sort_cols:
                combined_df = combined_df.sort_values(sort_cols)

            # 7. Save Final Daily File
            if output_dir is None:
                output_dir = config.PREDICTION_DIR / f"{year}_{month:02d}_daily"

            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)

            output_file = output_dir / f"prediction_{year}_{month:02d}_{day:02d}.csv"

            # Use our standard save utility (handles Kelvin to Celsius conversion)
            self.save_prediction_dataset(combined_df, output_file)

            # 8. CLEANUP: Delete hourly temp files and their folders
            print("Cleaning up temporary worker files...")
            for f_path in temp_files:
                p = Path(f_path)
                try:
                    if p.exists():
                        p.unlink()  # Delete the file
                        p.parent.rmdir()  # Delete the worker's folder
                except Exception as e:
                    print(f"Warning: Could not delete temp file/folder {p.parent}: {e}")

            # Explicitly clear memory
            del combined_df
            gc.collect()

            print(f"SUCCESS: Day {year}-{month:02d}-{day:02d} complete.")
            print(f"Result saved to: {output_file}\n")

            return str(output_file)

        except Exception as e:
            print(f"CRITICAL ERROR during day aggregation: {e}")
            traceback.print_exc()
            return None

    def _process_month_with_hourly_parallel(
        self, variables, year, month, shapefile=None, num_workers=None, output_dir=None, days_per_batch=1
    ):
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
                    variables, year, month, day, shapefile=shapefile, num_workers=num_workers, output_dir=output_dir
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
            if hasattr(self.datasets[var_name], "close"):
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
                df = pd.read_csv(file_path, parse_dates=["timestamp"], index_col="timestamp")
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
        sort_columns = ["timestamp"]
        if "name" in merged_df.columns:
            sort_columns.append("name")
        else:
            if "latitude" in merged_df.columns:
                sort_columns.append("latitude")
            if "longitude" in merged_df.columns:
                sort_columns.append("longitude")

        merged_df = merged_df.sort_values(sort_columns)

        # Determine output file path
        if output_file is None:
            # Get first file to extract year and month
            first_file = Path(file_paths[0])
            parts = first_file.stem.split("_")
            if len(parts) >= 3:
                year = parts[1]
                month = parts[2]
                output_file = config.PREDICTION_DIR / f"era5land_gee_prediction_{year}_{month}.csv"
            else:
                output_file = config.PREDICTION_DIR / "era5land_gee_prediction_merged.csv"

        output_file = Path(output_file)
        output_file.parent.mkdir(exist_ok=True, parents=True)

        # Save merged dataset
        merged_df.to_csv(output_file)
        print(f"Saved merged dataset with {len(merged_df)} rows to {output_file}")

        return str(output_file)

    def process_month_parallel(
        self,
        variables,
        year,
        month,
        shapefile=None,
        num_workers=None,
        output_dir=None,
        hourly_parallel=False,
        days_per_batch=1,
    ):
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
                variables,
                year,
                month,
                shapefile=shapefile,
                num_workers=num_workers,
                output_dir=output_dir,
                days_per_batch=days_per_batch,
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


def main():
    """
    Main function for ERA5-Land processing with multi-year support.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Process ERA5-Land data from GEE for sap velocity prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single month
  python script.py --year 2023 --month 6 --time-scale daily
  
  # Process entire year (all 12 months)
  python script.py --year 2023 --time-scale daily
  
  # Process multiple years
  python script.py --years 2020 2021 2022 2023 --time-scale daily
  
  # Process specific months across multiple years
  python script.py --years 2020 2021 --months 6 7 8 --time-scale daily
  
  # Process year range
  python script.py --year-range 2018 2023 --time-scale daily
  
  # Save yearly files instead of monthly
  python script.py --year 2023 --time-scale daily --yearly-output
        """,
    )

    # Year arguments (mutually exclusive group)
    year_group = parser.add_mutually_exclusive_group(required=True)
    year_group.add_argument("--year", type=int, help="Single year to process")
    year_group.add_argument(
        "--years", type=int, nargs="+", help="List of years to process (e.g., --years 2020 2021 2022)"
    )
    year_group.add_argument(
        "--year-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Range of years to process (e.g., --year-range 2018 2023)",
    )

    # Month arguments
    parser.add_argument("--month", type=int, help="Single month to process (1-12). If omitted, processes all months.")
    parser.add_argument(
        "--months", type=int, nargs="+", help="List of months to process (e.g., --months 1 2 3 or --months 6 7 8)"
    )

    # Day argument (for single day processing)
    parser.add_argument("--day", type=int, help="Day to process (for single-day processing)")

    # Processing options
    parser.add_argument(
        "--time-scale",
        type=str,
        default="daily",
        choices=["hourly", "daily"],
        help="Time scale of the data: hourly or daily (default: daily)",
    )
    parser.add_argument("--parallel", action="store_true", help="Process using parallel processing")
    parser.add_argument("--hourly-parallel", action="store_true", help="Process with hour-level parallelization")
    parser.add_argument("--num-workers", type=int, help="Number of parallel worker processes")
    parser.add_argument(
        "--days-per-chunk", type=int, default=15, help="Days per chunk for memory management (default: 15)"
    )

    # Output options
    parser.add_argument("--output", type=str, help="Output directory path")
    parser.add_argument(
        "--yearly-output", action="store_true", help="Save entire year to single file instead of monthly files"
    )
    parser.add_argument(
        "--monthly-output", action="store_true", help="Explicitly save monthly files (default behavior)"
    )

    # Region options
    parser.add_argument("--shapefile", type=str, default=None, help="Path to shapefile for region definition")
    parser.add_argument(
        "--gee-scale", type=float, default=11132, help="Scale in meters for GEE export (default: ~10km)"
    )

    args = parser.parse_args()

    # ============================================================
    # RESOLVE YEARS
    # ============================================================
    if args.year is not None:
        years = [args.year]
    elif args.years is not None:
        years = sorted(args.years)
    elif args.year_range is not None:
        start_year, end_year = args.year_range
        if start_year > end_year:
            start_year, end_year = end_year, start_year
        years = list(range(start_year, end_year + 1))
    else:
        print("Error: Must specify --year, --years, or --year-range")
        return

    # ============================================================
    # RESOLVE MONTHS
    # ============================================================
    if args.month is not None:
        months = [args.month]
    elif args.months is not None:
        months = sorted(args.months)
    else:
        months = None  # Will process all 12 months

    # Validate months
    if months:
        invalid_months = [m for m in months if m < 1 or m > 12]
        if invalid_months:
            print(f"Error: Invalid month(s): {invalid_months}. Months must be 1-12.")
            return

    # ============================================================
    # PRINT PROCESSING PLAN
    # ============================================================
    print(f"\n{'=' * 70}")
    print("PROCESSING PLAN")
    print(f"{'=' * 70}")
    print(f"Years: {years}")
    print(f"Months: {months if months else 'All (1-12)'}")
    print(f"Time scale: {args.time_scale}")
    print(f"Output mode: {'Yearly files' if args.yearly_output else 'Monthly files'}")
    if args.shapefile:
        print(f"Shapefile: {args.shapefile}")
    if args.output:
        print(f"Output directory: {args.output}")
    print(f"{'=' * 70}\n")

    # Confirm for large jobs
    total_months = len(years) * (len(months) if months else 12)
    if total_months > 12:
        print(f"This will process {total_months} month(s) of data.")

    # ============================================================
    # INITIALIZE PROCESSOR
    # ============================================================
    try:
        processor = ERA5LandGEEProcessor(time_scale=args.time_scale)

        output_dir = Path(args.output) if args.output else config.PREDICTION_DIR

        # ============================================================
        # SINGLE DAY PROCESSING
        # ============================================================
        if args.day is not None:
            if len(years) > 1 or (months and len(months) > 1):
                print("Error: --day can only be used with a single year and month")
                return

            year = years[0]
            month = months[0] if months else 1

            print(f"Processing single day: {year}-{month:02d}-{args.day:02d}")

            processor.process_single_day(
                config.REQUIRED_VARIABLES,
                year,
                month,
                args.day,
                shapefile=args.shapefile,
                output_dir=output_dir / f"{year}_{args.time_scale}",
                hourly_parallel=args.hourly_parallel,
                num_workers=args.num_workers,
            )

        # ============================================================
        # SINGLE YEAR PROCESSING
        # ============================================================
        elif len(years) == 1:
            year = years[0]

            if months and len(months) == 1:
                # Single month
                month = months[0]
                print(f"Processing single month: {year}-{month:02d}")

                processor.process_month_daily_chunked(
                    config.REQUIRED_VARIABLES,
                    year,
                    month,
                    shapefile=args.shapefile,
                    output_dir=output_dir / f"{year}_{args.time_scale}",
                    days_per_chunk=args.days_per_chunk,
                )
            else:
                # Multiple months or full year
                if months:
                    print(f"Processing {year} months: {months}")
                    # Process specific months
                    processor.process_multiple_years(
                        config.REQUIRED_VARIABLES,
                        years=[year],
                        months=months,
                        shapefile=args.shapefile,
                        output_dir=output_dir,
                        days_per_chunk=args.days_per_chunk,
                        save_monthly=not args.yearly_output,
                    )
                else:
                    print(f"Processing full year: {year}")
                    processor.process_year(
                        config.REQUIRED_VARIABLES,
                        year,
                        shapefile=args.shapefile,
                        output_dir=output_dir / f"{year}_{args.time_scale}",
                        days_per_chunk=args.days_per_chunk,
                        save_monthly=not args.yearly_output,
                    )

        # ============================================================
        # MULTI-YEAR PROCESSING
        # ============================================================
        else:
            print(f"Processing {len(years)} years: {years}")

            processor.process_multiple_years(
                config.REQUIRED_VARIABLES,
                years=years,
                months=months,
                shapefile=args.shapefile,
                output_dir=output_dir,
                days_per_chunk=args.days_per_chunk,
                save_monthly=not args.yearly_output,
            )

        print("\n" + "=" * 70)
        print("PROCESSING COMPLETE")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
    finally:
        try:
            processor.close()
        except Exception:
            pass


if __name__ == "__main__":
    import json
    from pathlib import Path

    import ee

    cred_paths = [
        Path.home() / ".config" / "earthengine" / "credentials",
        Path.home() / ".config" / "gcloud" / "application_default_credentials.json",
    ]
    gee_project = "era5download-447713"
    try:
        cred_data = None
        for cred_path in cred_paths:
            if cred_path.exists():
                cred_data = json.loads(cred_path.read_text())
                break
        if cred_data and "refresh_token" in cred_data:
            from google.oauth2.credentials import Credentials

            credentials = Credentials(
                token=None,
                refresh_token=cred_data["refresh_token"],
                token_uri="https://oauth2.googleapis.com/token",
                client_id=cred_data["client_id"],
                client_secret=cred_data["client_secret"],
            )
            ee.Initialize(credentials=credentials, project=gee_project)
        else:
            ee.Initialize(project="ee-yuluo-2")
        print("GEE Initialized successfully.")
    except Exception as e:
        print(f"GEE Initialization failed: {e}")
        print("Run 'earthengine authenticate' on the login node first, then resubmit.")
        import sys

        sys.exit(1)
    main()
