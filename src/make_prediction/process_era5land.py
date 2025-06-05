"""
ERA5-Land data processing utility for sap velocity prediction.
Handles extraction, reading, and preprocessing of ERA5-Land NetCDF files
with enhanced robustness for handling file access issues.
"""
from cfgrib.xarray_to_grib import to_grib
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import time
import tempfile
import shutil
import warnings
import geopandas as gpd
import subprocess
import glob
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import rasterio
from rasterio.warp import transform_bounds, transform
import gc
# Import configuration
import config
import traceback

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
    import gzip
    HAVE_GZIP = True
except ImportError:
    HAVE_GZIP = False
    print("Gzip module not available. Decompression may be limited.")

try:
    import netCDF4
    HAVE_NETCDF4 = True
except ImportError:
    HAVE_NETCDF4 = False
    print("NetCDF4 module not available. Using fallback methods.")


class ERA5LandProcessor:
    """
    Class to process ERA5-Land data files with optimized methods for memory efficiency.
    Based on the ERA5LandCompressedReader with modifications for sap velocity prediction.
    """
    # Define Whittaker biome classification thresholds
    # Format: (temp_min, temp_max, precip_min, precip_max)
    # Temperatures in °C, precipitation in mm/year
    
    def __init__(self, data_dir=None, temp_dir=None, create_client=None, memory_limit=None, 
                 temp_climate_file=None, precip_climate_file=None):
        """
        Initialize the ERA5-Land data processor.
        
        Parameters:
        -----------
        data_dir : str or Path, optional
            Directory where ERA5-Land data is stored (defaults to config)
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
        """
        self.data_dir = Path(data_dir) if data_dir else config.ERA5LAND_DIR
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
        

        
        # Check available system utilities
        self.has_gunzip = self._check_command("gunzip")
        self.has_file = self._check_command("file")
        self.has_cdo = self._check_command("cdo")
        self.has_nccopy = self._check_command("nccopy")
        
        print("\nSystem utilities available:")
        print(f"gunzip: {self.has_gunzip}")
        print(f"file: {self.has_file}")
        print(f"cdo: {self.has_cdo}")
        print(f"nccopy: {self.has_nccopy}")
        
        # Try to load climate data if files are provided
        if self.temp_climate_file and self.precip_climate_file:
            if self.temp_climate_file.exists() and self.precip_climate_file.exists():
                print("Loading climate data during initialization...")
                self.load_climate_data()
    
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
    
    def _check_command(self, command):
        """Check if a command is available on the system."""
        try:
            # On Windows, use 'where' instead of 'which'
            if os.name == 'nt':  # Windows
                result = os.system(f"where {command} >nul 2>&1")
                return result == 0
            else:  # Unix/Linux/MacOS
                result = os.system(f"which {command} >/dev/null 2>&1")
                return result == 0
        except:
            return False
    
    def identify_file_type(self, filepath):
        """
        Identify the type of a file (compressed, NetCDF, etc.).
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the file
            
        Returns:
        --------
        str
            File type description
        """
        filepath = Path(filepath)
        if not filepath.exists():
            return "File not found"
        
        # Use 'file' command if available
        if self.has_file:
            try:
                result = subprocess.run(["file", "-b", str(filepath)], 
                                       stdout=subprocess.PIPE, 
                                       stderr=subprocess.PIPE, 
                                       universal_newlines=True,
                                       check=False)
                if result.returncode == 0:
                    return result.stdout.strip()
            except:
                pass
        
        # Fall back to checking magic bytes
        try:
            with open(filepath, 'rb') as f:
                magic_bytes = f.read(4)
                
                # Check for common formats
                if magic_bytes.startswith(b'\x1f\x8b'):
                    return "gzip compressed data"
                elif magic_bytes.startswith(b'PK\x03\x04'):
                    return "Zip archive data"
                elif magic_bytes.startswith(b'CDF'):
                    return "NetCDF data"
                elif magic_bytes.startswith(b'HDF'):
                    return "HDF data"
                elif magic_bytes.startswith(b'GRIB'):
                    return "GRIB data"
        except:
            pass
        
        # Check file extension as a last resort
        suffix = filepath.suffix.lower()
        if suffix == '.gz':
            return "gzip compressed data (by extension)"
        elif suffix == '.zip':
            return "ZIP archive data (by extension)"
        elif suffix == '.nc':
            return "NetCDF data (by extension)"
        elif suffix in ['.grb', '.grib', '.grib2']:
            return "GRIB data (by extension)"
        elif suffix == '.hdf' or suffix == '.h5':
            return "HDF data (by extension)"
        elif suffix == '.tif' or suffix == '.tiff':
            return "GeoTIFF data (by extension)"
        
        return "Unknown file type"
    
    def _decompress_file(self, filepath):
        """
        Decompress a file if compressed, with thorough path checking.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the file
            
        Returns:
        --------
        Path
            Path to the decompressed file (original path if not compressed)
        """
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return None
        
        # Define possible paths where the decompressed file might be found
        # Direct path in temp directory
        flat_path = self.temp_dir / filepath.stem
        
        # Sub-directory path (seen in error logs)
        sub_dir = self.temp_dir / filepath.stem
        sub_path = sub_dir / filepath.name
        
        # Check all possible existing decompressed versions
        if flat_path.exists() and flat_path.is_file():
            print(f"Using existing decompressed file: {flat_path}")
            return flat_path
        
        if sub_path.exists() and sub_path.is_file():
            print(f"Using existing decompressed file: {sub_path}")
            return sub_path
        
        # If the subdirectory exists but the expected file doesn't, check for other files
        if sub_dir.exists() and sub_dir.is_dir():
            # Look for any *.nc files in this directory
            nc_files = list(sub_dir.glob("*.nc"))
            if nc_files:
                print(f"Found existing decompressed file with different name: {nc_files[0]}")
                return nc_files[0]
            
            # Look for files without extension
            other_files = list(sub_dir.glob("*"))
            if other_files:
                for f in other_files:
                    if f.is_file():
                        print(f"Found potential decompressed file: {f}")
                        return f
        
        # If we reach here, no decompressed version exists, check if we need to decompress
        
        # Identify file type
        file_type = self.identify_file_type(filepath)
        print(f"File type: {file_type}")
        
        # Check if the file needs decompression
        if 'gzip' in file_type.lower() or filepath.suffix.lower() == '.gz':
            # Create a temporary output filename
            temp_file = self.temp_dir / filepath.stem
            
            print(f"Decompressing gzip file to: {temp_file}")
            
            # Try using system gunzip first
            if self.has_gunzip:
                try:
                    # Copy the file to temp directory first
                    temp_gz = self.temp_dir / filepath.name
                    shutil.copy2(filepath, temp_gz)
                    
                    # Run gunzip on the copied file
                    subprocess.run(["gunzip", "-f", str(temp_gz)], 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   check=True)
                    
                    # If the output path exists, decompression was successful
                    if temp_file.exists():
                        print(f"Successfully decompressed using gunzip command")
                        return temp_file
                except Exception as e:
                    print(f"Error using gunzip: {str(e)}")
            
            # Fall back to Python's gzip module
            if HAVE_GZIP:
                try:
                    with gzip.open(filepath, 'rb') as f_in:
                        with open(temp_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    print(f"Successfully decompressed using Python's gzip module")
                    return temp_file
                except Exception as e:
                    print(f"Error using Python's gzip module: {str(e)}")
            
            print("Failed to decompress file")
            return None
        
        elif 'zip' in file_type.lower() or filepath.suffix.lower() == '.zip':
            # Handle zip files
            import zipfile
            
            # Create a temporary directory for extraction
            extract_dir = self.temp_dir / filepath.stem
            extract_dir.mkdir(exist_ok=True)
            
            print(f"Extracting zip file to: {extract_dir}")
            
            try:
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Look for NetCDF files in the extracted directory
                grib_files = list(extract_dir.glob("**/*.grib"))
                if grib_files:
                    print(f"Found {len(grib_files)} grib files in the archive")
                    return grib_files[0]  # Return the first one
                else:
                    print("No .grib files found in the zip archive")
                    return None
            except Exception as e:
                print(f"Error extracting zip file: {str(e)}")
                return None
        
        # File doesn't appear to be compressed
        return filepath
    
    def _clean_cfgrib_index_files(self, filepath):
        """
        Remove any cfgrib index files that might cause issues.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the NetCDF file
        """
        try:
            filepath = str(filepath)
            index_pattern = f"{filepath}.*idx"
            
            # Find and remove any index files
            index_files = glob.glob(index_pattern)
            if index_files:
                print(f"Found {len(index_files)} cfgrib index files to clean")
                for idx_file in index_files:
                    try:
                        os.remove(idx_file)
                        print(f"Removed index file: {idx_file}")
                    except Exception as e:
                        print(f"Failed to remove index file {idx_file}: {str(e)}")
        except Exception as e:
            print(f"Error cleaning cfgrib index files: {str(e)}")
    
    def _find_variable_files(self, variable, year, month=None):
        """
        Find ERA5-Land files for a specific variable and time period.
        
        Parameters:
        -----------
        variable : str
            Variable name (e.g., '2m_temperature')
        year : int
            Year to search for
        month : int, optional
            Month to search for (if None, search for all months)
            
        Returns:
        --------
        list
            List of file paths matching the criteria
        """
        var_dir = self.data_dir / variable
        print(f"Searching for variable files in: {var_dir}")
        if not var_dir.exists():
            print(f"Variable directory not found: {var_dir}")
            return []
        
        # Create year subdirectory path
        year_dir = var_dir / str(year)
        if not year_dir.exists():
            print(f"Year directory not found: {year_dir}")
            return []
        
        # Find files
        if month is not None:
            # Look for specific month
            month_pattern = f"{month:02d}"
            pattern = f"{variable}_{year}_{month_pattern}_hourly.nc"
            month_dir = year_dir / str(month)
            
            files = []
            # Check if month subdirectory exists
            if month_dir.exists():
                files.extend(list(month_dir.glob(pattern)))
            
            # Also check directly in year directory
            files.extend(list(year_dir.glob(pattern)))
            
            # Filter out .idx files
            files = [f for f in files if not str(f).endswith('.idx')]
            
            print(f"Found {len(files)} files after filtering out index files")
            return files
        else:
            # Find all months
            files = []
            # Look in each month subdirectory
            for month_dir in year_dir.glob("[0-9][0-9]"):
                if month_dir.is_dir():
                    month_pattern = month_dir.name
                    pattern = f"{variable}_{year}_{month_pattern}_hourly.nc*"
                    files.extend(list(month_dir.glob(pattern)))
            
            # Also check directly in year directory
            pattern = f"{variable}_{year}_[0-9][0-9]_hourly.nc*"
            files.extend(list(year_dir.glob(pattern)))
            
            # Filter out .idx files
            files = [f for f in files if not str(f).endswith('.idx')]
            
            print(f"Found {len(files)} files after filtering out index files")
            return files
    
    def read_era5land_variable(self, variable, year, month=None, chunks=None):
        """
        Read ERA5-Land data for a specific variable and time period with robust error handling.
        
        Parameters:
        -----------
        variable : str
            Variable name (e.g., '2m_temperature')
        year : int
            Year to read
        month : int, optional
            Month to read (if None, read all available months)
        chunks : dict, optional
            Chunk sizes for each dimension
            
        Returns:
        --------
        xarray.Dataset
            Dataset containing the requested variable data
        """
        # Set default chunking if not provided
        if chunks is None:
            chunks = config.CHUNK_SIZE
        
        # Find files for the requested variable and time period
        files = self._find_variable_files(variable, year, month)
        
        if not files:
            print(f"No files found for variable {variable}, year {year}, month {month}")
            return None
        
        print(f"Found {len(files)} files for {variable}, year {year}, month {month if month else 'all'}")
        
        # Process each file and collect datasets
        datasets = []
        for file_path in sorted(files):
            print(f"Processing file: {file_path}")
            
            # Decompress if needed, and save to temp directory
            decompressed_path = self._decompress_file(file_path)
            if decompressed_path is None:
                print(f"Failed to access or decompress file: {file_path}")
                continue
            
            # Verify the file actually exists
            if not os.path.exists(decompressed_path):
                print(f"Decompressed path doesn't exist: {decompressed_path}")
                # Try alternate locations
                alternate_paths = [
                    file_path,  # Original file may not need decompression
                    self.temp_dir / file_path.name,
                    self.temp_dir / file_path.stem,
                    self.temp_dir / file_path.stem / file_path.name,
                    self.temp_dir / file_path.stem / file_path.stem
                ]
                
                found = False
                for alt_path in alternate_paths:
                    if os.path.exists(alt_path):
                        print(f"Found file at alternate location: {alt_path}")
                        decompressed_path = alt_path
                        found = True
                        break
                
                if not found:
                    print("Could not locate the file anywhere, skipping...")
                    continue
            
            # Clean any existing cfgrib index files
            self._clean_cfgrib_index_files(decompressed_path)
            
            # Try to read the file using multiple methods
            ds = self._read_netcdf_with_fallbacks(decompressed_path, chunks=chunks)
            
            if ds is not None:
                datasets.append(ds)
            else:
                print(f"Failed to read file with all methods: {decompressed_path}")
        
        if not datasets:
            print(f"Failed to read any data for variable {variable}")
            return None
        
        # Combine datasets if multiple files were read
        if len(datasets) > 1:
            try:
                print(f"Combining {len(datasets)} datasets...")
                combined_ds = xr.concat(datasets, dim='time')
                print("Successfully combined datasets")
                
                # Close individual datasets to free memory
                for ds in datasets:
                    ds.close()
                
                return combined_ds
            except Exception as e:
                print(f"Error combining datasets: {str(e)}")
                # Return the first dataset if combination fails
                for i in range(1, len(datasets)):
                    datasets[i].close()
                return datasets[0]
        else:
            # Return the single dataset
            return datasets[0]
    
    def _read_netcdf_with_fallbacks(self, filepath, chunks=None, variables=None):
        """
        Read a NetCDF file with multiple fallback methods for improved robustness.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the NetCDF file
        chunks : dict, optional
            Chunk sizes for each dimension
        variables : list, optional
            Specific variables to load
            
        Returns:
        --------
        xarray.Dataset or None
            Dataset containing the file data, or None if all methods fail
        """
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return None
        
        print(f"Reading file using multiple methods: {filepath}")
        try:
            print("Trying xarray with cfgrib engine...")
            
            # Set environment variable to disable index
            os.environ['DISABLE_CFGRIB_INDEX'] = '1'
            # Method 1: Try with cfgrib but disable indexing
            try:
                ds = xr.open_dataset(filepath, engine='cfgrib', chunks=chunks)
                print("Successfully read file with cfgrib engine")
                return ds
            finally:
                # Reset the environment variable
                os.environ.pop('DISABLE_CFGRIB_INDEX', None)
        except Exception as e:
            print(f"Error with cfgrib engine: {str(e)}")
        # Method 2: Try direct netCDF4 engine first - often the most reliable
        try:
            print("Trying xarray with netcdf4 engine...")
            ds = xr.open_dataset(filepath, engine='netcdf4', chunks=chunks)
            print("Successfully read file with netcdf4 engine")
            return ds
        except Exception as e:
            print(f"Error with netcdf4 engine: {str(e)}")
        
        # Method 3: Try h5netcdf engine
        try:
            print("Trying xarray with h5netcdf engine...")
            ds = xr.open_dataset(filepath, engine='h5netcdf', chunks=chunks)
            print("Successfully read file with h5netcdf engine")
            return ds
        except Exception as e:
            print(f"Error with h5netcdf engine: {str(e)}")
        
        # Method 4: Try scipy engine
        try:
            print("Trying xarray with scipy engine...")
            ds = xr.open_dataset(filepath, engine='scipy', chunks=chunks)
            print("Successfully read file with scipy engine")
            return ds
        except Exception as e:
            print(f"Error with scipy engine: {str(e)}")
        
        
        
        
        # Method 5: Use netCDF4 directly for more control
        if HAVE_NETCDF4:
            try:
                print("Trying to use netCDF4 library directly...")
                
                nc = netCDF4.Dataset(filepath)
                
                # Extract coordinates
                coords = {}
                for dim_name in nc.dimensions:
                    if dim_name in nc.variables:
                        coord_var = nc.variables[dim_name]
                        coords[dim_name] = (dim_name, np.array(coord_var[:]))
                
                # Extract data variables
                data_vars = {}
                for var_name, var in nc.variables.items():
                    if var_name not in nc.dimensions:  # Skip coordinate variables
                        if variables is None or var_name in variables:
                            dims = var.dimensions
                            data = np.array(var[:])
                            attrs = {attr: getattr(var, attr) for attr in var.ncattrs()}
                            data_vars[var_name] = (dims, data, attrs)
                
                # Create xarray Dataset
                ds = xr.Dataset(data_vars=data_vars, coords=coords)
                print("Successfully created xarray Dataset from netCDF4")
                
                # Apply chunking if specified and dask is available
                if chunks and HAVE_DASK:
                    ds = ds.chunk(chunks)
                
                nc.close()
                return ds
            except Exception as e:
                print(f"Error with direct netCDF4: {str(e)}")
        
        # Method 6: Try using CDO if available
        if self.has_cdo:
            try:
                print("Trying to use CDO to convert file...")
                converted_path = self.temp_dir / f"cdo_converted_{filepath.stem}.nc"
                
                # Use CDO to copy the file (which can sometimes fix corrupted files)
                result = subprocess.run(
                    ["cdo", "copy", str(filepath), str(converted_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False
                )
                
                if result.returncode == 0 and converted_path.exists():
                    try:
                        print("Reading CDO-converted file...")
                        ds = xr.open_dataset(converted_path, engine='netcdf4', chunks=chunks)
                        print("Successfully read CDO-converted file")
                        return ds
                    except Exception as e:
                        print(f"Error reading CDO-converted file: {str(e)}")
            except Exception as e:
                print(f"Error using CDO: {str(e)}")
        
        # Method 7: Try using nccopy if available
        if self.has_nccopy:
            try:
                print("Trying to use nccopy to repair file...")
                repaired_path = self.temp_dir / f"nccopy_repaired_{filepath.stem}.nc"
                
                # Use nccopy to create a fresh copy
                result = subprocess.run(
                    ["nccopy", "-k", "netCDF4", str(filepath), str(repaired_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False
                )
                
                if result.returncode == 0 and repaired_path.exists():
                    try:
                        print("Reading nccopy-repaired file...")
                        ds = xr.open_dataset(repaired_path, engine='netcdf4', chunks=chunks)
                        print("Successfully read nccopy-repaired file")
                        return ds
                    except Exception as e:
                        print(f"Error reading nccopy-repaired file: {str(e)}")
            except Exception as e:
                print(f"Error using nccopy: {str(e)}")
        
        # All methods failed
        print("All methods failed to read the file")
        return None
    
    def extract_region(self, ds, lat_min=None, lat_max=None, lon_min=None, lon_max=None):
        """
        Extract a spatial subset of the dataset with improved GRIB file support.
        
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
            
            # Handle different longitude conventions (0-360 vs -180 to 180)
            if ds[lon_var].max() > 180 and lon_min < 0:
                # Data uses 0-360 convention but we're using -180 to 180
                print("Converting longitude range from -180/180 to 0/360 convention")
                lon_min_360 = (lon_min + 360) % 360
                lon_max_360 = (lon_max + 360) % 360
                
                # Using a direct filtering approach instead of sel()
                # First filter latitude
                lat_mask = (ds[lat_var] >= lat_min) & (ds[lat_var] <= lat_max)
                
                # Then filter longitude based on the 0-360 conversion
                if lon_min_360 > lon_max_360:  # Crossing the 0/360 boundary
                    lon_mask = (ds[lon_var] >= lon_min_360) | (ds[lon_var] <= lon_max_360)
                else:
                    lon_mask = (ds[lon_var] >= lon_min_360) & (ds[lon_var] <= lon_max_360)
                
                # Create masks for each dimension
                masks = {}
                if lat_var in ds.dims:
                    masks[lat_var] = lat_mask
                if lon_var in ds.dims:
                    masks[lon_var] = lon_mask
                
                # Apply all masks
                region_ds = ds
                for dim, mask in masks.items():
                    if mask.any():  # Only apply non-empty masks
                        region_ds = region_ds.isel({dim: xr.where(mask, True, False)})
            else:
                # Using direct filtering for standard coordinates too
                lat_mask = (ds[lat_var] >= lat_min) & (ds[lat_var] <= lat_max)
                lon_mask = (ds[lon_var] >= lon_min) & (ds[lon_var] <= lon_max)
                
                # Create masks for each dimension
                masks = {}
                if lat_var in ds.dims:
                    masks[lat_var] = lat_mask
                if lon_var in ds.dims:
                    masks[lon_var] = lon_mask
                
                # Apply all masks
                region_ds = ds
                for dim, mask in masks.items():
                    if mask.any():  # Only apply non-empty masks
                        region_ds = region_ds.isel({dim: xr.where(mask, True, False)})
            
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
    
    def load_variable_data(self, variables, year, month, region=None):
        """
        Load and preprocess multiple ERA5-Land variables for a specific time period.
        
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
            ds = self.read_era5land_variable(var_name, year, month)
            
            if ds is not None:
                # Extract region of interest
                ds = self.extract_region(
                    ds, 
                    lat_min=region['lat_min'],
                    lat_max=region['lat_max'],
                    lon_min=region['lon_min'],
                    lon_max=region['lon_max']
                )
                
                self.datasets[var_name] = ds
                print(f"Successfully loaded {var_name}")
            else:
                print(f"Failed to load {var_name}")
        
        return self.datasets
    
    def calculate_derived_variables(self):
        """
        Calculate derived variables from the loaded ERA5-Land data.
        With special handling for GRIB file format.
        
        Returns:
        --------
        dict
            Updated dataset dictionary with derived variables
        """
        if not self.datasets:
            print("No datasets loaded. Cannot calculate derived variables.")
            return self.datasets
        
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
                
                # Get the main data variable from each dataset (for GRIB files, there's typically only one)
                u_var = list(u_ds.data_vars)[0] if list(u_ds.data_vars) else None
                v_var = list(v_ds.data_vars)[0] if list(v_ds.data_vars) else None
                
                # If no data variables found, the data may be directly in the dataset (GRIB specific)
                if u_var is None or v_var is None:
                    print("No named data variables found - using dataset directly (GRIB format)")
                    # Create synthetic variables for the calculation
                    u_wind = u_ds.to_array().squeeze()
                    v_wind = v_ds.to_array().squeeze()
                    
                    # Calculate wind speed
                    wind_speed = np.sqrt(u_wind**2 + v_wind**2)
                    
                    # Create a new dataset with the wind speed variable
                    wind_ds = xr.Dataset(
                        data_vars={'wind_speed': wind_speed},
                        coords=u_ds.coords,
                        attrs={'units': 'm s**-1', 'long_name': 'Wind speed at 10m'}
                    )
                    
                    self.datasets['wind_speed'] = wind_ds
                    print("Wind speed calculated successfully from dataset arrays")
                elif u_var and v_var:
                    # Standard approach using named variables
                    wind_speed = np.sqrt(u_ds[u_var]**2 + v_ds[v_var]**2)
                    
                    wind_ds = xr.Dataset(
                        data_vars={'wind_speed': wind_speed},
                        coords=u_ds.coords,
                        attrs={'units': 'm s**-1', 'long_name': 'Wind speed at 10m'}
                    )
                    # calculate the size of the wind speed variable in mb
                    wind_ds['wind_speed'].attrs['size'] = wind_ds['wind_speed'].nbytes / 1024**2  # Convert bytes to MB
                    print(f"Wind speed variable size: {wind_ds['wind_speed'].attrs['size']} mb")
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
                
                # If no data variables found, the data may be directly in the dataset (GRIB specific)
                if t_var is None or td_var is None:
                    print("No named temperature variables found - using dataset directly (GRIB format)")
                    # Create synthetic variables for the calculation
                    temperature = t_ds.to_array().squeeze()
                    dewpoint = td_ds.to_array().squeeze()
                    
                    # Convert from K to C
                    t_c = temperature - 273.15
                    td_c = dewpoint - 273.15
                    
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
                    
                    self.datasets['vpd'] = vpd_ds
                    print("VPD calculated successfully from dataset arrays")
                elif t_var and td_var:
                    # Standard approach using named variables
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
                    # calculate the size of the VPD variable in mb
                    vpd_ds['vpd'].attrs['size'] = vpd_ds['vpd'].nbytes / 1024**2  # Convert bytes to MB
                    print(f"VPD variable size: {vpd_ds['vpd'].attrs['size']} mb")
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
                
                # If no data variables found, the data may be directly in the dataset (GRIB specific)
                if sr_var is None:
                    print("No named radiation variable found - using dataset directly (GRIB format)")
                    # Create synthetic variable for the calculation
                    solar_rad_data = sr_ds.to_array().squeeze()
                    
                    # Check if it's accumulated (using metadata or name)
                    is_accumulated = False
                    if hasattr(solar_rad_data, 'long_name'):
                        is_accumulated = 'accum' in solar_rad_data.long_name.lower()
                    elif hasattr(sr_ds, 'typeOfStatisticalProcessing'):
                        is_accumulated = sr_ds.typeOfStatisticalProcessing == 1  # Accumulation in GRIB
                    elif 'step' in sr_ds.coords:
                        # ERA5 with forecast steps typically has accumulated values
                        is_accumulated = True
                    
                    if is_accumulated and 'time' in sr_ds.dims:
                        print("Processing accumulated radiation data")
                        # Handle accumulated data
                        diff = solar_rad_data.diff('time')
                        
                        # Get time step in seconds
                        time_var = sr_ds.coords['time']
                        time_step_seconds = np.diff(time_var.values).astype('timedelta64[s]').astype(float)[0]
                        
                        # Calculate instantaneous radiation in W/m2
                        solar_rad = diff / time_step_seconds
                        
                        # Handle first time step
                        first_step_rate = solar_rad.isel(time=0)
                        solar_rad = xr.concat([first_step_rate.expand_dims('time'), solar_rad], dim='time')
                    else:
                        solar_rad = solar_rad_data
                    
                    # Calculate PAR (Photosynthetically Active Radiation)
                    par_fraction = 0.45
                    par_wm2 = solar_rad * par_fraction
                    
                    # Convert PAR from W/m2 to μmol/m2/s
                    conversion_factor = 4.6
                    ppfd = par_wm2 * conversion_factor
                    
                    # Also calculate extraterrestrial radiation
                    # Extract latitude
                    lat_var = None
                    for var in sr_ds.coords:
                        if var.lower() in ['latitude', 'lat']:
                            lat_var = var
                    
                    if lat_var:
                        # Get dimension information without loading all data
                        lat_values = sr_ds[lat_var].values
                        time_values = sr_ds.coords['time'].values
                        
                        # Create proper DataArrays for broadcasting
                        lat_rad = np.deg2rad(lat_values)
                        
                        # Extract time and convert to day of year
                        time_dt = pd.DatetimeIndex(time_values)
                        doy = time_dt.dayofyear
                        
                        # Solar constant (W/m2)
                        solar_constant = 1361.0
                        
                        # Calculate solar declination
                        declination = 0.409 * np.sin(2 * np.pi * (doy - 82) / 365)
                        
                        # Create a smaller 2D grid for easier computation
                        rad_values_2d = np.zeros((len(lat_values), len(time_values)))
                        
                        # Calculate the radiation values for each lat/time point
                        for i, lat in enumerate(lat_values):
                            lat_rad_val = np.deg2rad(lat)
                            for j, t in enumerate(time_values):
                                t_idx = np.where(time_values == t)[0][0]
                                decl = declination[t_idx]
                                
                                # Calculate cos(zenith) at solar noon (hour_angle = 0)
                                cos_zenith = np.sin(lat_rad_val) * np.sin(decl) + np.cos(lat_rad_val) * np.cos(decl)
                                cos_zenith = max(0.001, cos_zenith)  # Ensure no negative values
                                
                                # Calculate extraterrestrial radiation
                                rad_values_2d[i, j] = solar_constant * cos_zenith
                        
                        # Create a 2D DataArray with these values
                        ext_rad_2d = xr.DataArray(
                            rad_values_2d,
                            dims=[lat_var, 'time'],
                            coords={lat_var: lat_values, 'time': time_values}
                        )
                        
                        # Create a dataset with the standard structure by broadcasting
                        # Use the original dataset as a template and broadcast the 2D values
                        ext_rad_ds = xr.Dataset(
                            data_vars={'ext_rad': ext_rad_2d},
                            coords=sr_ds.coords
                        )
                        
                        # Add attributes to the dataset
                        ext_rad_ds.attrs['units'] = 'W m-2'
                        ext_rad_ds.attrs['long_name'] = 'Extraterrestrial Radiation'
                    else:
                        print("Latitude coordinate not found, using solar radiation as proxy for ext_rad")
                        ext_rad_ds = xr.Dataset(
                            data_vars={'ext_rad': solar_rad * 1.5},  # Simple approximation 
                            coords=sr_ds.coords,
                            attrs={'units': 'W m-2', 'long_name': 'Extraterrestrial Radiation'}
                        )
                    
                    # Create new datasets
                    ppfd_ds = xr.Dataset(
                        data_vars={'ppfd_in': ppfd},
                        coords=sr_ds.coords,
                        attrs={'units': 'μmol m-2 s-1', 'long_name': 'Photosynthetic Photon Flux Density'}
                    )
                    
                    self.datasets['ppfd_in'] = ppfd_ds
                    self.datasets['ext_rad'] = ext_rad_ds
                    print("PAR-related variables calculated successfully from dataset arrays")
                elif sr_var:
                    # Standard approach using named variable
                    # Check if the data needs time differencing (for accumulated values)
                    is_accumulated = False
                    if 'long_name' in sr_ds[sr_var].attrs:
                        is_accumulated = 'accum' in sr_ds[sr_var].attrs['long_name'].lower()
                    elif 'standard_name' in sr_ds[sr_var].attrs:
                        is_accumulated = 'accum' in sr_ds[sr_var].attrs['standard_name'].lower()
                    
                    if is_accumulated and 'time' in sr_ds.dims:
                        # Need to compute the difference between consecutive time steps
                        # and divide by the time step in seconds
                        diff = sr_ds[sr_var].diff('time')
                        
                        # Get time step in seconds
                        time_var = sr_ds.coords['time']
                        time_step_seconds = np.diff(time_var.values).astype('timedelta64[s]').astype(float)[0]
                        
                        # Calculate instantaneous radiation in W/m2
                        solar_rad = diff / time_step_seconds
                        
                        # Need to handle the first time step (no diff available)
                        first_step = sr_ds[sr_var].isel(time=0)
                        # Assume the rate for the first time step is the same as the second
                        first_step_rate = solar_rad.isel(time=0)
                        
                        # Concatenate the first step with the rest
                        solar_rad = xr.concat([first_step_rate.expand_dims('time'), solar_rad], dim='time')
                    else:
                        # Data is already in instantaneous form (W/m2)
                        solar_rad = sr_ds[sr_var]
                    
                    # Calculate PAR (Photosynthetically Active Radiation)
                    par_fraction = 0.45
                    par_wm2 = solar_rad * par_fraction
                    
                    # Convert PAR from W/m2 to μmol/m2/s
                    conversion_factor = 4.6
                    ppfd = par_wm2 * conversion_factor
                    
                    # Also calculate extraterrestrial radiation
                    lat_var = None
                    for var in sr_ds.coords:
                        if var.lower() in ['latitude', 'lat']:
                            lat_var = var
                    
                    if lat_var:
                        # Get dimension information without loading all data
                        dims = sr_ds[sr_var].dims
                        shape = sr_ds[sr_var].shape
                        print(f"Template dimensions: {dims}")
                        print(f"Template shape: {shape}")
                        
                        # Extract coordinates without loading full data
                        time_values = sr_ds.coords['time'].values
                        time_dt = pd.DatetimeIndex(time_values)
                        lat_values = sr_ds[lat_var].values
                        
                        # Calculate day of year for each time
                        doy = time_dt.dayofyear
                        
                        # Solar constant (W/m2)
                        solar_constant = 1361.0
                        
                        # Calculate solar declination for each day
                        declination = 0.409 * np.sin(2 * np.pi * (doy - 82) / 365)
                        
                        # Create a 2D array for latitude and time calculations
                        rad_values_2d = np.zeros((len(lat_values), len(time_values)))
                        
                        # Calculate extraterrestrial radiation for each lat/time point
                        for i, lat in enumerate(lat_values):
                            lat_rad = np.deg2rad(lat)
                            for j, t in enumerate(time_values):
                                day_idx = np.where(time_values == t)[0][0]
                                day_decl = declination[day_idx]
                                
                                # Calculate cos(zenith) at solar noon (hour_angle = 0)
                                cos_zenith = np.sin(lat_rad) * np.sin(day_decl) + np.cos(lat_rad) * np.cos(day_decl)
                                
                                # Ensure no negative values
                                cos_zenith = max(0.001, cos_zenith)
                                
                                # Calculate extraterrestrial radiation
                                rad_values_2d[i, j] = solar_constant * cos_zenith
                        
                        # Create a 2D DataArray
                        ext_rad_2d = xr.DataArray(
                            rad_values_2d,
                            dims=[lat_var, 'time'],
                            coords={lat_var: lat_values, 'time': time_values}
                        )
                        
                        # Create a 1D DataArray for each dimension that will be broadcast
                        # This approach uses broadcasting to expand the 2D array to all dimensions
                        # Create a template with the same structure as the original data
                        # Important: This doesn't load the data into memory
                        ones_template = xr.ones_like(sr_ds[sr_var])
                        
                        # Now use the 2D array and broadcast it to the full dimensionality
                        # by multiplying with the template
                        ext_rad_full = ones_template * ext_rad_2d
                        
                        # Create the dataset with the same coordinates as the source
                        ext_rad_ds = xr.Dataset(
                            data_vars={'ext_rad': ext_rad_full},
                            coords=sr_ds.coords,
                            attrs={'units': 'W m-2', 'long_name': 'Extraterrestrial Radiation'}
                        )
                        
                        print(f"Created ext_rad with shape: {ext_rad_full.shape}")
                        print(f"Dimensions: {ext_rad_full.dims}")
                        
                    else:
                        print("Latitude coordinate not found, using solar radiation as proxy for ext_rad")
                        ext_rad = solar_rad * 1.5  # Simple approximation
                        ext_rad_ds = xr.Dataset(
                            data_vars={'ext_rad': ext_rad},
                            coords=sr_ds.coords,
                            attrs={'units': 'W m-2', 'long_name': 'Extraterrestrial Radiation'}
                        )
                    
                    # Create new datasets
                    ppfd_ds = xr.Dataset(
                        data_vars={'ppfd_in': ppfd},
                        coords=sr_ds.coords,
                        attrs={'units': 'μmol m-2 s-1', 'long_name': 'Photosynthetic Photon Flux Density'}
                    )
                    
                    # calculate the size of the ppfd variable in mb
                    ppfd_ds['ppfd_in'].attrs['size'] = ppfd_ds['ppfd_in'].nbytes / 1024**2  # Convert bytes to MB
                    print(f"PPFD variable size: {ppfd_ds['ppfd_in'].attrs['size']} mb")
                    
                    # Calculate the size of the ext_rad variable in mb
                    ext_rad_ds['ext_rad'].attrs['size'] = ext_rad_ds['ext_rad'].nbytes / 1024**2  # Convert bytes to MB
                    print(f"Extraterrestrial radiation variable size: {ext_rad_ds['ext_rad'].attrs['size']} mb")
                    
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


    def create_prediction_dataset(self, start_date, end_date, points=None, region=None, output_dir=None, lat_chunk_size=50, lon_chunk_size=100, time_chunk_size=96):
        """
        Create a dataset ready for prediction by extracting data for the required time period.
        Processes large regions in spatio-temporal-longitude chunks to avoid memory errors.

        Parameters:
        -----------
        start_date : str or datetime
            Start date for the prediction period
        end_date : str or datetime
            End date for the prediction period
        points : list of dict, optional
            List of points to extract data for [{lat, lon, name}, ...]
        region : dict, optional
            Region to extract data for, if no points are specified
        output_dir : str or Path, optional
            Base directory to save output files (defaults to config.PREDICTION_DIR)
        lat_chunk_size : int, optional
            Number of latitude rows to process in each spatial chunk (default: 50)
        lon_chunk_size : int, optional
            Number of longitude columns per spatial chunk (default: 100)
        time_chunk_size : int, optional
            Number of time steps per temporal chunk (default: 96)

        Returns:
        --------
        list of str or list of pandas.DataFrame
            - If processing a region: Returns a list of file paths where each latitude chunk DataFrame was saved.
            - If processing points: Returns a list of pandas DataFrames, one for each point.
            Returns None if no data can be processed or if processing fails.
        """
        # --- Initial Setup ---
        start_time_total = time.time()
        print(f"Starting dataset creation for {start_date} to {end_date}")

        # Convert dates to datetime objects
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            # Ensure end_date includes the whole day if just a date is given
            if end_date.normalize() == end_date: # Checks if time part is midnight
                 end_date = end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            print(f"Processing time range: {start_date} to {end_date}")
        except Exception as e:
            print(f"Error parsing start/end dates: {e}")
            return None

        # Determine processing mode and define region if needed
        if points:
            processing_mode = 'points'
            print(f"Processing mode: Specific points ({len(points)})")
            if region is None:
                 all_lats = [p['lat'] for p in points]
                 all_lons = [p['lon'] for p in points]
                 region = {
                     'lat_min': min(all_lats) - 0.1, 'lat_max': max(all_lats) + 0.1,
                     'lon_min': min(all_lons) - 0.1, 'lon_max': max(all_lons) + 0.1
                 }
                 print(f"Region automatically determined from points: {region}")
            else:
                 print(f"Using provided region for point data loading: {region}")
        else:
            processing_mode = 'region'
            print("Processing mode: Region")
            if region is None:
                 region = {
                     'lat_min': getattr(config, 'DEFAULT_LAT_MIN', 0), 'lat_max': getattr(config, 'DEFAULT_LAT_MAX', 0),
                     'lon_min': getattr(config, 'DEFAULT_LON_MIN', 0), 'lon_max': getattr(config, 'DEFAULT_LON_MAX', 0)
                 }
            print(f"Processing for region: Lat({region.get('lat_min', 'N/A')} to {region.get('lat_max', 'N/A')}), Lon({region.get('lon_min', 'N/A')} to {region.get('lon_max', 'N/A')})")

        # Set default output directory
        if output_dir is None:
            output_dir = getattr(config, 'PREDICTION_DIR', Path('./prediction_output'))
        output_dir = Path(output_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Using output directory: {output_dir.resolve()}")
        except Exception as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return None

        # --- End Initial Setup ---


        # --- Monthly Data Loading and Combination ---
        all_monthly_data = []
        # Generate date range covering the start and end months
        month_iterator = pd.date_range(start=start_date.replace(day=1), end=end_date.replace(day=1), freq='MS')

        required_vars = getattr(config, 'REQUIRED_VARIABLES', [])
        if not required_vars:
            print("Error: config.REQUIRED_VARIABLES is not defined or empty.")
            return None

        for current_month_start in month_iterator:
            year = current_month_start.year
            month = current_month_start.month
            print(f"\nProcessing data loading for {year}-{month:02d}...")

            # Ensure load_variable_data exists and works
            try:
                 # Load variable data for the current month
                 monthly_datasets = self.load_variable_data(required_vars, year, month, region)
            except AttributeError:
                 print("Error: self.load_variable_data method not found.")
                 return None
            except Exception as e_load:
                 print(f"Error during load_variable_data for {year}-{month}: {e_load}")
                 monthly_datasets = None # Ensure it's None if loading fails

            if not monthly_datasets:
                print(f"No primary data loaded for {year}-{month:02d}. Skipping month.")
                continue

            # Set self.datasets for the methods below to operate on
            self.datasets = monthly_datasets

            # Calculate derived variables
            print(f"Calculating derived variables for {year}-{month:02d}...")
            try: self.calculate_derived_variables() # Operates on self.datasets
            except AttributeError: print("Warning: calculate_derived_variables method not found.")
            except Exception as e_deriv: print(f"Warning: Error calculating derived variables: {e_deriv}")

            # Flatten time dimensions
            print(f"Flattening time dimensions for {year}-{month:02d}...")
            try: self.flatten_time_dimensions() # Operates on self.datasets
            except AttributeError: print("Warning: flatten_time_dimensions method not found.")
            except Exception as e_flat: print(f"Warning: Error flattening time dimensions: {e_flat}")

            # Store the processed datasets for this month
            all_monthly_data.append(self.datasets.copy())
            print(f"Finished processing for {year}-{month:02d}")


        if not all_monthly_data:
            print("No data could be processed for the entire date range.")
            return None

        # --- Combine Monthly Data ---
        print("\nCombining data across all months...")
        combined_datasets = {}
        all_vars = sorted(list(set(k for month_data in all_monthly_data for k in month_data.keys())))

        for var_name in all_vars:
            # ... (Combination logic with two attempts - 'no_conflicts' then attribute cleaning) ...
            # (Copied from previous correct version - ensure it's placed here)
            print(f"--- Combining variable: {var_name} ---")
            datasets_to_concat = [
                month_data[var_name]
                for month_data in all_monthly_data
                if var_name in month_data and month_data[var_name] is not None
            ]

            if datasets_to_concat:
                print(f"Found {len(datasets_to_concat)} monthly datasets for {var_name}")
                combined_success = False
                try:
                    # Attempt 1: Concatenate with 'no_conflicts'
                    print(f"Attempting xr.concat for {var_name} (compat='no_conflicts')...")
                    uses_dask = any(hasattr(ds, 'chunks') and ds.chunks for ds in datasets_to_concat)
                    if uses_dask:
                         first_chunks = datasets_to_concat[0].chunks
                         if first_chunks and any(ds.chunks != first_chunks for ds in datasets_to_concat[1:] if hasattr(ds, 'chunks')):
                              print(f"  Detected inconsistent chunks for {var_name}. Rechunking to match first dataset.")
                              datasets_to_concat = [ds.chunk(first_chunks) if hasattr(ds,'chunks') and ds.chunks != first_chunks else ds for ds in datasets_to_concat]

                    combined_ds = xr.concat(
                        datasets_to_concat, dim='datetime', join='inner',
                        compat='no_conflicts', coords='minimal'
                    )
                    combined_datasets[var_name] = combined_ds
                    print(f"Combined {var_name} successfully (Attempt 1).")
                    combined_success = True

                except TypeError as te:
                    if "unhashable type: 'Frozen'" not in str(te):
                        print(f"TypeError combining {var_name} (compat='no_conflicts'): {te}. Skipping.")
                        continue # Skip this variable
                    print(f"Caught 'unhashable type: Frozen' with compat='no_conflicts'. Will try cleaning attributes.")
                except Exception as e:
                    print(f"Error combining {var_name} with compat='no_conflicts': {type(e).__name__} - {e}. Will try cleaning attributes.")

                if not combined_success:
                    # Attempt 2: Clean Attributes and Retry
                    print(f"Attempting xr.concat for {var_name} after cleaning attributes...")
                    try:
                        cleaned_datasets = []
                        for ds in datasets_to_concat:
                            ds_copy = ds.copy(deep=False); ds_copy.attrs = {}
                            for var in ds_copy.variables: ds_copy[var].attrs = {}
                            cleaned_datasets.append(ds_copy)

                        combined_ds_cleaned = xr.concat(
                            cleaned_datasets, dim='datetime', join='inner',
                            compat='override', coords='minimal'
                        )
                        combined_datasets[var_name] = combined_ds_cleaned
                        print(f"Combined {var_name} successfully after cleaning attributes (Attempt 2).")

                    except Exception as e_clean:
                        print(f"Error combining {var_name} even after cleaning attributes: {type(e_clean).__name__} - {e_clean}. Skipping this variable.")
            else:
                print(f"No monthly datasets found for {var_name} to combine.")
        # --- End Combination Loop ---

        if not combined_datasets:
            print("Failed to combine any variables across months.")
            self.datasets = {}
            return None

        self.datasets = combined_datasets
        print("Finished combining monthly data.")
        # --- End Monthly Data Loading and Combination ---


        # --- Prepare for Prediction Data Extraction ---
        # Use getattr for safe access to VARIABLE_RENAME
        var_rename_map = getattr(config, 'VARIABLE_RENAME', {})
        prediction_vars = [era5_name for era5_name in var_rename_map if era5_name in self.datasets]
        # Include variables that might not be renamed but are in datasets
        for var_name in self.datasets:
             if var_name not in prediction_vars and var_name not in var_rename_map.values():
                  prediction_vars.append(var_name) # Add variables not in rename map

        if not prediction_vars:
            print("No variables available for prediction after combining months.")
            return None
        print(f"Variables prepared for extraction: {prediction_vars}")
        # --- End Prepare ---


        # --- Branch based on processing mode ---
        if processing_mode == 'points':
            # --- Process Specific Points ---
            print(f"\nExtracting data for {len(points)} specified points...")
            final_data_frames = []
            for point_idx, point in enumerate(points):
                # ... (Point processing logic - copied from previous correct version) ...
                point_name = point.get('name', f"Point_{point['lat']:.4f}_{point['lon']:.4f}")
                print(f"Processing point {point_idx + 1}/{len(points)}: {point_name} ({point['lat']:.4f}, {point['lon']:.4f})")
                point_timeseries_static = {'name': point_name, 'latitude': point['lat'], 'longitude': point['lon']}
                

                valid_point_data = True
                first_var_time_coord = None
                lazy_arrays = {}

                for var_name in prediction_vars:
                     if var_name not in self.datasets: continue # Skip if var somehow not in combined set
                     try:
                          ds = self.datasets[var_name]
                          data_var_name = list(ds.data_vars)[0] if list(ds.data_vars) else None
                          if not data_var_name: continue

                          # Use appropriate coordinate names found earlier if possible
                          # Assuming 'latitude', 'longitude' are standard now after processing
                          ts_lazy = ds[data_var_name].sel(latitude=point['lat'], longitude=point['lon'], method='nearest')

                          if first_var_time_coord is None and 'datetime' in ts_lazy.coords:
                               first_var_time_coord = ts_lazy['datetime'].copy()

                          model_name = var_rename_map.get(var_name, var_name)
                          lazy_arrays[model_name] = ts_lazy

                     except Exception as e_sel:
                          print(f"  Error selecting {var_name} for point {point_name}: {e_sel}")
                          valid_point_data = False; break

                if valid_point_data and first_var_time_coord is not None:
                     print(f"  Computing data for point {point_name}...")
                     try:
                          computed_ds = xr.Dataset(lazy_arrays).compute()
                          # Use computed time coord if needed, or the stored one
                          time_index_vals = computed_ds['datetime'].values if 'datetime' in computed_ds else first_var_time_coord.values
                          df_data = {k: computed_ds[k].values for k in computed_ds.data_vars if k != 'datetime'} # Exclude time if it's a data var
                          df = pd.DataFrame(df_data, index=time_index_vals)
                          df.index.name = 'time' # Ensure index is named 'time'

                          for static_key, static_val in point_timeseries_static.items():
                              df[static_key] = static_val

                          if getattr(config, 'TIME_FEATURES', False): df = self.add_time_features(df)

                          df = df[(df.index >= start_date) & (df.index <= end_date)].sort_index()

                          if not df.empty:
                              final_data_frames.append(df)
                              print(f"  Successfully processed point {point_name}. DataFrame shape: {df.shape}")
                          else:
                              print(f"  Warning: No data remained for point {point_name} after date filtering.")
                     except Exception as e_compute:
                          print(f"  Error computing or creating DataFrame for point {point_name}: {e_compute}")
                          traceback.print_exc()
                else:
                     print(f"  Skipping point {point_name} due to data extraction errors or missing time coordinate.")

            print(f"Finished processing points. Returning {len(final_data_frames)} DataFrames.")
            return final_data_frames if final_data_frames else None
            # --- End Process Specific Points ---

        elif processing_mode == 'region':
            # --- Process Entire Region in Spatio-Temporal-Longitudinal Chunks ---
            print("\nProcessing entire region in spatio-temporal-longitude chunks...")
            saved_chunk_files = [] # Store paths to saved chunk files

            # --- Prepare Merged Dataset ---
            print("Merging all variables into a single dataset for chunking...")
            merged_ds = xr.Dataset()
            lat_var, lon_var, time_var = None, None, None
            try:
                 # Use first available variable to find coordinate names
                 first_var_name = next(iter(self.datasets.keys())) # Get first key safely
                 first_ds = self.datasets[first_var_name]
                 lat_var = next((v for v in first_ds.coords if v.lower() in ['latitude', 'lat']), None)
                 lon_var = next((v for v in first_ds.coords if v.lower() in ['longitude', 'lon']), None)
                 time_var = next((v for v in first_ds.coords if v.lower() in ['datetime']), None)

                 if not all([lat_var, lon_var, time_var]):
                      print("Could not identify required coordinate variables (latitude, longitude, datetime).")
                      return None

                 # Merge variables using found coordinate names
                 for var_name in prediction_vars: # Use filtered prediction_vars
                      if var_name not in self.datasets: continue # Ensure var exists
                      ds = self.datasets[var_name]
                      data_var_name = list(ds.data_vars)[0] if list(ds.data_vars) else None
                      if not data_var_name: continue
                      model_name = var_rename_map.get(var_name, var_name) # Use rename map
                      merged_ds[model_name] = ds[data_var_name]

                 print(f"Merged dataset variables for chunking: {list(merged_ds.data_vars)}")
                 print(f"Merged dataset coords for chunking: {list(merged_ds.coords)}")
            except Exception as e:
                 print(f"Error merging variables into a single dataset: {e}")
                 traceback.print_exc()
                 return None

            # --- End Prepare Merged Dataset ---

            # --- ****** NEW: Rechunk Merged Dataset ****** ---
            print("\nRechunking merged dataset for optimized selection...")
            # Tune these based on memory and testing
            spatial_rechunk_size = max(1, lat_chunk_size // 2, lon_chunk_size // 2) # e.g., half of processing chunk
            time_rechunk = time_chunk_size

            target_chunks = {
                lat_var: spatial_rechunk_size,
                lon_var: spatial_rechunk_size,
                time_var: time_rechunk
            }
            print(f"Target rechunk sizes: {target_chunks}")
            try:
                rechunk_start = time.time()
                # Ensure merged_ds has data before rechunking
                if not merged_ds.data_vars:
                     raise ValueError("Merged dataset has no data variables to rechunk.")
                merged_ds = merged_ds.chunk(target_chunks)
                print(f"Rechunked merged_ds chunks: {merged_ds.chunks} ({time.time() - rechunk_start:.2f}s)")
                gc.collect()
            except Exception as e_rechunk:
                print(f"Error rechunking merged_ds: {e_rechunk}")
                print("Try adjusting target_chunks or skipping rechunking if error persists.")
                traceback.print_exc()
                return None # Stop if rechunking fails
            # --- ****** End Rechunk Merged Dataset ****** ---


            # --- Get Coordinates for Chunking ---
            try:
                 lats_coords = merged_ds[lat_var]
                 lons_coords = merged_ds[lon_var]
                 time_coords = merged_ds[time_var]
                 num_lats = len(lats_coords)
                 num_lons = len(lons_coords)
                 num_times = len(time_coords)
                 print(f"Using coordinate variables: Lat='{lat_var}', Lon='{lon_var}', Time='{time_var}'")
            except Exception as e_coord:
                 print(f"Error accessing coordinates for chunking: {e_coord}")
                 return None

            print(f"Starting chunked processing loops for {num_lats} lat, {num_lons} lon, {num_times} time steps...")

            # --- Outer Loop: Latitude Chunks ---
            for lat_start_idx in range(0, num_lats, lat_chunk_size):
                lat_end_idx = min(lat_start_idx + lat_chunk_size, num_lats)
                lat_chunk_indices = range(lat_start_idx, lat_end_idx)
                lat_min_chunk = lats_coords.isel({lat_var: lat_start_idx}).item()
                lat_max_chunk = lats_coords.isel({lat_var: lat_end_idx - 1}).item()
                print(f"\nProcessing Latitude Chunk {lat_start_idx // lat_chunk_size + 1}/{ (num_lats + lat_chunk_size - 1) // lat_chunk_size }: Indices {lat_start_idx}-{lat_end_idx-1} (Lat {lat_min_chunk:.2f} to {lat_max_chunk:.2f})")

                dfs_for_current_lat_chunk = []

                # --- Middle Loop: Time Chunks ---
                for time_start_idx in range(0, num_times, time_chunk_size):
                    time_end_idx = min(time_start_idx + time_chunk_size, num_times)
                    time_chunk_indices = range(time_start_idx, time_end_idx)
                    time_min_chunk_ts = pd.Timestamp(time_coords.isel({time_var: time_start_idx}).values)
                    time_max_chunk_ts = pd.Timestamp(time_coords.isel({time_var: time_end_idx - 1}).values)
                    print(f"  Processing Time Chunk {time_start_idx // time_chunk_size + 1}/{ (num_times + time_chunk_size - 1) // time_chunk_size }: Indices {time_start_idx}-{time_end_idx-1} ({time_min_chunk_ts} to {time_max_chunk_ts})")

                    dfs_for_current_st_chunk = []

                    # --- Inner Loop: Longitude Chunks ---
                    for lon_start_idx in range(0, num_lons, lon_chunk_size):
                        lon_end_idx = min(lon_start_idx + lon_chunk_size, num_lons)
                        lon_chunk_indices = range(lon_start_idx, lon_end_idx)
                        print(f"    Processing Lon Chunk {lon_start_idx // lon_chunk_size + 1}/{ (num_lons + lon_chunk_size - 1) // lon_chunk_size }: Indices {lon_start_idx}-{lon_end_idx-1}")

                        # Initialize variables used in try/finally
                        stl_chunk_ds_lazy = None
                        computed_stl_chunk_ds = None
                        df_stl_chunk = None
                        stacked_stl_chunk = None

                        try:
                            # Select spatio-temporal-longitude chunk (LAZY selection on RECHUNKED data)
                            print("      Selecting chunk indices...")
                            stl_chunk_ds_lazy = merged_ds.isel({
                                lat_var: lat_chunk_indices,
                                time_var: time_chunk_indices,
                                lon_var: lon_chunk_indices
                            })

                            # Compute only this smallest chunk
                            print("      Computing chunk data...")
                            start_compute = time.time()
                            # Add progress bar if dask client exists
                            if hasattr(self, 'client') and self.client:
                                 from dask.diagnostics import ProgressBar
                                 with ProgressBar():
                                     computed_stl_chunk_ds = stl_chunk_ds_lazy.compute()
                            else:
                                 computed_stl_chunk_ds = stl_chunk_ds_lazy.compute()
                            print(f"      Computation complete ({time.time() - start_compute:.2f}s).")

                            # Convert smallest chunk to DataFrame
                            print("      Converting chunk to DataFrame...")
                            start_convert = time.time()
                            # Stack spatial dimensions, keep time as dimension initially
                            # create_index=False might help memory slightly
                            stacked_stl_chunk = computed_stl_chunk_ds.stack(location=(lat_var, lon_var), create_index=False).reset_index(time_var)
                            df_stl_chunk = stacked_stl_chunk.to_dataframe()
                            df_stl_chunk = df_stl_chunk.reset_index() # Get time, lat, lon as columns
                            print(f"      Conversion complete ({time.time() - start_convert:.2f}s).")


                            # Process smallest DataFrame Chunk
                            df_stl_chunk = df_stl_chunk.rename(columns={time_var: 'time', lat_var: 'latitude', lon_var: 'longitude'})
                            if 'location' in df_stl_chunk.columns: df_stl_chunk = df_stl_chunk.drop(columns=['location'])
                            df_stl_chunk.dropna(axis=1, how='all', inplace=True) # Drop empty columns

                            if df_stl_chunk.empty:
                                print("      Chunk DataFrame is empty. Skipping.")
                                continue

                            # Set time index
                            if 'time' in df_stl_chunk.columns:
                                df_stl_chunk = df_stl_chunk.set_index('time')
                            else:
                                print("      Warning: 'time' column not found. Cannot set index.")
                                continue

                            # Add time features
                            if getattr(config, 'TIME_FEATURES', False):
                                print("      Adding time features...")
                                df_stl_chunk = self.add_time_features(df_stl_chunk) # Assumes method exists

                            # Append the processed smallest DataFrame
                            dfs_for_current_st_chunk.append(df_stl_chunk)

                        except MemoryError as me_stl:
                            print(f"! MemoryError processing chunk (Lat: {lat_start_idx}-{lat_end_idx-1}, Time: {time_start_idx}-{time_end_idx-1}, Lon: {lon_start_idx}-{lon_end_idx-1}): {me_stl}")
                            print("! This chunk is STILL too large during compute/convert. Try reducing ALL chunk sizes ('lat_chunk_size', 'lon_chunk_size', 'time_chunk_size') and rechunk sizes ('spatial_rechunk_size', 'time_rechunk').")
                            raise MemoryError("Stopping due to MemoryError in smallest chunk.") from me_stl # Stop all processing
                        except Exception as e_stl:
                            print(f"! Error processing chunk (Lat: {lat_start_idx}-{lat_end_idx-1}, Time: {time_start_idx}-{time_end_idx-1}, Lon: {lon_start_idx}-{lon_end_idx-1}): {e_stl}")
                            traceback.print_exc()
                            print("      Skipping current smallest chunk due to error.")
                        finally:
                            # --- Updated Finally Block ---
                            print("      Cleaning up chunk memory...")
                            # Check existence before deleting
                            if 'stl_chunk_ds_lazy' in locals() and stl_chunk_ds_lazy is not None: del stl_chunk_ds_lazy
                            if 'computed_stl_chunk_ds' in locals() and computed_stl_chunk_ds is not None: del computed_stl_chunk_ds
                            if 'df_stl_chunk' in locals() and df_stl_chunk is not None: del df_stl_chunk
                            if 'stacked_stl_chunk' in locals() and stacked_stl_chunk is not None: del stacked_stl_chunk
                            gc.collect() # Force garbage collection
                            print("      Memory cleanup for smallest chunk complete.")
                            # --- End Updated Finally Block ---
                    # --- End Inner Loop: Longitude Chunks ---

                    # --- Combine Longitude Chunks for the current Spatio-Temporal Chunk ---
                    if dfs_for_current_st_chunk:
                         print(f"    Concatenating {len(dfs_for_current_st_chunk)} longitude chunks for time chunk {time_start_idx}-{time_end_idx-1}...")
                         st_concat_start = time.time()
                         try:
                              df_st_chunk_combined = pd.concat(dfs_for_current_st_chunk, axis=0)
                              dfs_for_current_lat_chunk.append(df_st_chunk_combined)
                              print(f"    Appended combined ST chunk. Shape: {df_st_chunk_combined.shape} ({time.time() - st_concat_start:.2f}s)")
                         except Exception as e_concat_lon:
                              print(f"    Error concatenating longitude chunks: {e_concat_lon}")
                         finally:
                              del dfs_for_current_st_chunk, df_st_chunk_combined
                              gc.collect()
                    else:
                         print(f"    No longitude chunks processed for time chunk {time_start_idx}-{time_end_idx-1}.")
                    # --- End Combine Longitude Chunks ---
                    print(f"  Finished Time Chunk {time_start_idx // time_chunk_size + 1}.")
                # --- End Middle Loop: Time Chunks ---

                # --- Combine Time Chunks and Save Full Latitude Chunk ---
                if dfs_for_current_lat_chunk:
                    print(f"\nConcatenating {len(dfs_for_current_lat_chunk)} time chunks for latitude chunk {lat_start_idx}-{lat_end_idx-1}...")
                    lat_chunk_concat_start = time.time()
                    df_lat_chunk = None # Initialize for finally block
                    try:
                        df_lat_chunk = pd.concat(dfs_for_current_lat_chunk, axis=0)
                        print(f"  Concatenated latitude chunk DataFrame shape: {df_lat_chunk.shape} ({time.time() - lat_chunk_concat_start:.2f}s)")

                        # Apply final filtering by date range
                        print("  Filtering final date range for latitude chunk...")
                        # Ensure index is DatetimeIndex before filtering
                        if not isinstance(df_lat_chunk.index, pd.DatetimeIndex):
                             print("  Warning: Index is not DatetimeIndex before final filtering. Attempting conversion...")
                             try: df_lat_chunk.index = pd.to_datetime(df_lat_chunk.index)
                             except Exception as e_conv:
                                  print(f"  Error converting index to DatetimeIndex: {e_conv}. Cannot filter/save chunk.")
                                  continue # Skip to next latitude chunk

                        df_lat_chunk = df_lat_chunk[
                             (df_lat_chunk.index >= start_date) & (df_lat_chunk.index <= end_date)
                        ]
                        # Sorting index ensures time is sequential, might be required by models
                        print("  Sorting index for latitude chunk...")
                        sort_start = time.time()
                        df_lat_chunk.sort_index(inplace=True)
                        print(f"  Sorting complete ({time.time() - sort_start:.2f}s)")


                        if df_lat_chunk.empty:
                             print("  Latitude chunk is empty after final date filtering.")
                        else:
                            # Define output file path
                            chunk_filename = Path(output_dir) / f"prediction_chunk_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_lat_{lat_min_chunk:.2f}_{lat_max_chunk:.2f}.parquet"
                            print(f"  Saving latitude chunk to {chunk_filename}...")
                            save_start = time.time()
                            # Ensure lat/lon are standard columns before saving
                            if 'latitude' not in df_lat_chunk.columns or 'longitude' not in df_lat_chunk.columns:
                                 df_lat_chunk = df_lat_chunk.reset_index()

                            df_lat_chunk.to_parquet(chunk_filename, index=True) # Save with time index
                            saved_chunk_files.append(str(chunk_filename))
                            print(f"  Latitude chunk saved successfully ({time.time() - save_start:.2f}s).")

                    except Exception as e_concat_time:
                        print(f"Error concatenating time chunks or saving latitude chunk {lat_start_idx}-{lat_end_idx-1}: {e_concat_time}")
                        traceback.print_exc()
                    finally:
                         # Clean up list and combined DataFrame for the latitude chunk
                         del dfs_for_current_lat_chunk # List of DFs
                         if 'df_lat_chunk' in locals() and df_lat_chunk is not None: del df_lat_chunk
                         gc.collect()
                else:
                     print(f"No data processed for latitude chunk {lat_start_idx}-{lat_end_idx-1}.")
                # --- End Combine and Save Full Latitude Chunk ---

                # No explicit lazy latitude dataset to delete here as merged_ds is used directly
                gc.collect()
                print(f"--- Finished Latitude Chunk {lat_start_idx // lat_chunk_size + 1} ---")
            # --- End Outer Loop: Latitude Chunks ---

            print(f"\nFinished chunked processing. Saved {len(saved_chunk_files)} chunk files.")
            print(f"Total time for dataset creation: {time.time() - start_time_total:.2f} seconds")
            return saved_chunk_files # Return the list of file paths
            # --- End Region Processing ---

        # Fallback return if neither points nor region processing path completed
        return None



def main():
    """
    Main function to demonstrate ERA5-Land processing with climate-based biome determination.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Process ERA5-Land data for sap velocity prediction')
    parser.add_argument('--year', type=int, required=True, help='Year to process')
    parser.add_argument('--month', type=int, required=True, help='Month to process')
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
    
    parser.add_argument('--disable-cfgrib-index', action='store_true', help='Disable cfgrib indexing')
    parser.add_argument('--lat-chunk-size', type=int, default=1, help='Number of latitude rows per spatial chunk for region processing')
    parser.add_argument('--lon-chunk-size', type=int, default=1, help='Number of longitude columns per spatial chunk for region processing') # Added
    parser.add_argument('--time-chunk-size', type=int, default=24, help='Number of time steps per temporal chunk for region processing') # Added
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save output chunk files for region processing')
    args = parser.parse_args()

    # Set environment variable if requested
    if args.disable_cfgrib_index:
        os.environ['CFGRIB_DISABLE_INDEX'] = '1'
        print("cfgrib indexing disabled via environment variable")
    output_dir = args.output_dir or config.PREDICTION_DIR

    # Create ERA5-Land processor with climate data files
    processor = ERA5LandProcessor(
        temp_climate_file=args.temp_climate,
        precip_climate_file=args.precip_climate
    )
    
    try:
        # Set up region
        region = {
            'lat_min': args.lat_min if args.lat_min is not None else config.DEFAULT_LAT_MIN,
            'lat_max': args.lat_max if args.lat_max is not None else config.DEFAULT_LAT_MAX,
            'lon_min': args.lon_min if args.lon_min is not None else config.DEFAULT_LON_MIN,
            'lon_max': args.lon_max if args.lon_max is not None else config.DEFAULT_LON_MAX
        }
        
 
        
        # Define points if specified
        points = None
        if args.point_lat is not None and args.point_lon is not None:
            points = [{
                'lat': args.point_lat,
                'lon': args.point_lon,
                'name': args.point_name or f"Point_{args.point_lat}_{args.point_lon}"
            }]
        
        # Set the start and end dates
        start_date = pd.Timestamp(args.year, args.month, 1)
        if args.month == 12:
            end_date = pd.Timestamp(args.year + 1, 1, 1) - pd.Timedelta(days=1)
        else:
            end_date = pd.Timestamp(args.year, args.month + 1, 1) - pd.Timedelta(days=1)
        
        # Create prediction dataset
        df = processor.create_prediction_dataset(start_date, end_date, points, region, output_dir=output_dir, lat_chunk_size=args.lat_chunk_size, lon_chunk_size=args.lon_chunk_size, time_chunk_size=args.time_chunk_size)
        
        if df is not None:
            # Save the dataset
            saved_files = processor.save_prediction_dataset(df, output_path=Path(output_dir) / (args.output or f"prediction_output_{args.year}_{args.month:02d}") ) # Provide a base name/path for point files
            print("\nProcessing complete.")
            if isinstance(saved_files, list):
                print(f"Generated {len(saved_files)} output files:")
                for f in saved_files: print(f" - {f}")
            elif isinstance(saved_files, str):
                print(f"Generated output file: {saved_files}")
    
    finally:
        # Reset environment variable if set
        if args.disable_cfgrib_index:
            os.environ.pop('CFGRIB_DISABLE_INDEX', None)
        
        # Clean up
        processor.close()

if __name__ == "__main__":
    main()