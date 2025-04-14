import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import shutil
import subprocess
import glob
import warnings

# Dask configuration for improved memory management
try:
    import dask
    from dask.diagnostics import ProgressBar
    HAVE_DASK = True
    
    # Configure dask for better memory usage
    dask.config.set({
        'array.chunk-size': '64MiB',
        'distributed.worker.memory.target': 0.6,  # Target fraction of memory to use
        'distributed.worker.memory.spill': 0.7,   # When to spill to disk
        'distributed.worker.memory.pause': 0.8,   # When to pause processing
    })
except ImportError:
    HAVE_DASK = False

# Try to import compression-related libraries
try:
    import gzip
    HAVE_GZIP = True
except ImportError:
    HAVE_GZIP = False

try:
    import tarfile
    HAVE_TARFILE = True
except ImportError:
    HAVE_TARFILE = False

try:
    import xarray as xr
    HAVE_XARRAY = True
except ImportError:
    HAVE_XARRAY = False

try:
    import netCDF4
    HAVE_NETCDF4 = True
except ImportError:
    HAVE_NETCDF4 = False


class ERA5LandCompressedReader:
    """
    Class to read compressed ERA5-Land data files with chunked processing for memory efficiency.
    """
    def __init__(self, data_dir='era5land_data', temp_dir=None, create_client=False, memory_limit='4GB'):
        """
        Initialize the ERA5-Land data reader.
        
        Parameters:
        -----------
        data_dir : str
            Directory where ERA5-Land data is stored
        temp_dir : str, optional
            Directory for temporary storage of decompressed files
        create_client : bool, optional
            Whether to create a Dask distributed client
        memory_limit : str, optional
            Memory limit for the Dask client (if created)
        """
        self.data_dir = Path(data_dir)
        
        # Set up temporary directory for decompressed files
        if temp_dir:
            self.temp_dir = Path(temp_dir)
            self.temp_dir.mkdir(exist_ok=True, parents=True)
        else:
            # Use system temporary directory
            self.temp_dir = Path(tempfile.gettempdir()) / "era5land_temp"
            self.temp_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Using temporary directory: {self.temp_dir}")
        
        # Check available libraries
        self._check_available_libraries()
        
        # Initialize Dask client if requested and available
        self.client = None
        if create_client and HAVE_DASK:
            try:
                from dask.distributed import Client
                print(f"Creating Dask client with memory limit: {memory_limit}")
                self.client = Client(memory_limit=memory_limit)
                print(f"Dask dashboard link: {self.client.dashboard_link}")
            except (ImportError, Exception) as e:
                print(f"Could not create Dask client: {str(e)}")
    
    def _check_available_libraries(self):
        """Print information about available libraries."""
        print("\nAvailable libraries:")
        print(f"xarray: {HAVE_XARRAY}")
        print(f"netCDF4: {HAVE_NETCDF4}")
        print(f"gzip: {HAVE_GZIP}")
        print(f"tarfile: {HAVE_TARFILE}")
        print(f"dask: {HAVE_DASK}")
        
        # Check if system utilities are available
        self.has_gunzip = self._check_command("gunzip")
        self.has_file = self._check_command("file")
        self.has_cdo = self._check_command("cdo")
        self.has_nccopy = self._check_command("nccopy")
        
        print(f"System utilities:")
        print(f"gunzip: {self.has_gunzip}")
        print(f"file: {self.has_file}")
        print(f"cdo: {self.has_cdo}")
        print(f"nccopy: {self.has_nccopy}")
    
    def _check_command(self, command):
        """Check if a command is available on the system."""
        try:
            # On Windows, use 'where' instead of 'which'
            if os.name == 'nt':  # Windows
                subprocess.run(["where", command], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               check=True)
            else:  # Unix/Linux/MacOS
                subprocess.run(["which", command], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               check=True)
            return True
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
        
        return "Unknown file type"
    
    def decompress_file(self, filepath):
        """
        Decompress a file if compressed.
        
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
    
    def read_file(self, filepath, variables=None, chunks=None, engine=None, sample_only=False):
        """
        Read a NetCDF file with chunked processing, decompressing if necessary.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the file
        variables : list, optional
            List of specific variables to load (to reduce memory usage)
        chunks : dict, optional
            Chunk sizes for each dimension (e.g., {'time': 1, 'step': 1, 'latitude': 100, 'longitude': 100})
            If None, automatic chunking will be applied
        engine : str, optional
            Specific engine to use with xarray
        sample_only : bool, optional
            If True, only load a small sample of the data for exploration
            
        Returns:
        --------
        xarray.Dataset
            The dataset containing the ERA5-Land data
        """
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return None
        
        # Set default chunking if not provided - optimized for spatial operations
        if chunks is None:
            # These are more memory-friendly defaults for ERA5-Land data
            chunks = {'time': 1, 'step': 1, 'latitude': 100, 'longitude': 100}
            print(f"Using default chunking: {chunks}")
        
        # Try to decompress if needed
        decompressed_path = self.decompress_file(filepath)
        if decompressed_path is None:
            print(f"Failed to decompress or access file: {filepath}")
            return None
        
        print(f"Reading file: {decompressed_path}")
        
        # For sample only mode, use smaller chunks and load limited data
        if sample_only:
            print("Loading sample data only for exploration")
            try:
                if HAVE_XARRAY:
                    engines_to_try = [engine] if engine else ['cfgrib', 'netcdf4', 'h5netcdf', 'scipy']
                    
                    for eng in engines_to_try:
                        try:
                            # Create subsetting parameters to load a small sample
                            # For example, take first few time steps and a low-resolution spatial grid
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                
                                # First, open dataset to check dimensions without loading data
                                sample_ds = xr.open_dataset(
                                    decompressed_path,
                                    engine=eng,
                                    chunks=None  # Don't use chunking for initial inspection
                                )
                                
                                # Get coordinate names for dimensions
                                lat_var = next((var for var in sample_ds.coords if 'lat' in var.lower()), None)
                                lon_var = next((var for var in sample_ds.coords if 'lon' in var.lower()), None)
                                time_var = next((var for var in sample_ds.coords if 'time' in var.lower()), None)
                                
                                # Create subsetting slices
                                subset_dict = {}
                                if time_var:
                                    subset_dict[time_var] = slice(0, 2)  # First 2 time steps
                                if lat_var and lat_var in sample_ds.dims:
                                    subset_dict[lat_var] = slice(None, None, 10)  # Take every 10th latitude
                                if lon_var and lon_var in sample_ds.dims:
                                    subset_dict[lon_var] = slice(None, None, 10)  # Take every 10th longitude
                                
                                # Close temporary dataset
                                sample_ds.close()
                                
                                # Now load the subset with chunking
                                ds = xr.open_dataset(
                                    decompressed_path,
                                    engine=eng,
                                    chunks=chunks
                                ).isel(**subset_dict)
                                
                                print(f"Successfully loaded sample data with xarray (engine: {eng})")
                                return ds
                                
                        except Exception as e:
                            print(f"Error with sample mode using engine {eng}: {str(e)}")
            except Exception as e:
                print(f"Error in sample mode: {str(e)}")
                
            print("Sample mode failed, trying normal load...")
        
        # Attempt different reading methods with chunking
        if HAVE_XARRAY and HAVE_DASK:
            try:
                engines_to_try = [engine] if engine else ['cfgrib', 'netcdf4', 'h5netcdf', 'scipy']
                
                for eng in engines_to_try:
                    try:
                        print(f"Trying xarray with engine: {eng} and chunking")
                        # Use selective variable loading if variables are specified
                        if variables:
                            print(f"Loading only variables: {variables}")
                            ds = xr.open_dataset(
                                decompressed_path, 
                                engine=eng,
                                chunks=chunks,
                                variables=variables
                            )
                        else:
                            ds = xr.open_dataset(
                                decompressed_path, 
                                engine=eng,
                                chunks=chunks
                            )
                        # inspect the dataset
                        # data dimensions and variables
                        print("Dataset dimensions:")
                        print(ds.dims)
                        print("Dataset variables:")
                        print(ds.data_vars)
                        print(ds)
                        
                        print(f"Successfully read file with xarray (engine: {eng}, chunked)")
                        
                        # Print chunk information
                        if hasattr(ds, 'chunks') and ds.chunks:
                            print("Dataset is chunked with the following chunk sizes:")
                            for var_name, var_chunks in ds.chunks.items():
                                print(f"  {var_name}: {var_chunks}")
                        
                        return ds
                    except Exception as e:
                        print(f"Error with xarray engine {eng}: {str(e)}")
            except Exception as e:
                print(f"Error with xarray: {str(e)}")
        
        # Try with netCDF4 directly (no chunking support in this fallback)
        if HAVE_NETCDF4:
            try:
                print("Trying netCDF4 directly (note: no chunking support)")
                nc = netCDF4.Dataset(decompressed_path)
                print("Successfully opened with netCDF4")
                
                # Create a simple xarray-like Dataset
                variables_dict = {}
                for var_name, variable in nc.variables.items():
                    # Skip dimension variables for simplicity
                    if var_name not in nc.dimensions:
                        if variables is None or var_name in variables:
                            variables_dict[var_name] = variable
                
                class SimpleDataset:
                    def __init__(self, nc_dataset, variables):
                        self.nc = nc_dataset
                        self.data_vars = variables
                    
                    def close(self):
                        self.nc.close()
                    
                    def to_dataframe(self):
                        # Very basic conversion
                        data = {}
                        for var_name, var in self.data_vars.items():
                            try:
                                # Read limited data to prevent memory issues
                                if len(var.shape) > 2:
                                    # For multi-dimensional data, read first slice only
                                    first_slice = tuple([0] * (len(var.shape) - 2)) + (slice(None), slice(None))
                                    data[var_name] = var[first_slice]
                                    print(f"Warning: Only reading first slice of {var_name} to conserve memory")
                                else:
                                    data[var_name] = var[:]
                            except Exception as e:
                                print(f"Error reading {var_name}: {str(e)}")
                        return pd.DataFrame(data)
                
                return SimpleDataset(nc, variables_dict)
            except Exception as e:
                print(f"Error with netCDF4: {str(e)}")
        
        # If all methods failed, try using CDO if available
        if self.has_cdo:
            try:
                print("Attempting to use CDO to convert the file")
                cdo_output = self.temp_dir / f"cdo_converted_{filepath.stem}.nc"
                
                # Use CDO with subsetting if variables are specified
                if variables:
                    var_list = ",".join(variables)
                    subprocess.run(["cdo", "selname," + var_list, str(decompressed_path), str(cdo_output)], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE, 
                                  check=True)
                else:
                    subprocess.run(["cdo", "copy", str(decompressed_path), str(cdo_output)], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE, 
                                  check=True)
                
                if cdo_output.exists() and HAVE_XARRAY:
                    print(f"Trying to read CDO-converted file with xarray and chunking")
                    ds = xr.open_dataset(cdo_output, chunks=chunks)
                    print("Successfully read CDO-converted file")
                    return ds
            except Exception as e:
                print(f"Error using CDO: {str(e)}")
        
        print("All methods failed to read the file")
        print("\nRecommendations to fix this issue:")
        print("1. Install dask to enable chunked processing:")
        print("   pip install dask distributed")
        print("2. Check if the file is actually a NetCDF file or needs special processing")
        print("3. Install required dependencies:")
        print("   pip install xarray netCDF4 dask")
        print("4. For GRIB files, install cfgrib:")
        print("   pip install cfgrib")
        print("5. For advanced file operations, install CDO (Climate Data Operators):")
        print("   - On Linux: sudo apt-get install cdo")
        print("   - On macOS: brew install cdo")
        print("   - On Windows: Check https://code.mpimet.mpg.de/projects/cdo/")
        
        return None
    
    def extract_region(self, ds, lon_min=None, lon_max=None, lat_min=None, lat_max=None):
        """
        Extract a spatial subset of the dataset.
        
        Parameters:
        -----------
        ds : xarray.Dataset
            Dataset from read_file
        lon_min, lon_max, lat_min, lat_max : float, optional
            Bounding box coordinates for the region of interest
            
        Returns:
        --------
        xarray.Dataset
            Spatially subset dataset
        """
        if ds is None:
            print("Cannot extract region: dataset is None")
            return None
        
        if not (lon_min and lon_max and lat_min and lat_max):
            print("No region specified, returning full dataset")
            return ds
        
        try:
            # Find coordinate variables for lat/lon
            lat_names = ['latitude', 'lat']
            lon_names = ['longitude', 'lon']
            
            lat_var = None
            lon_var = None
            
            # Find the lat/lon variable names in the dataset
            for var in ds.coords:
                if var.lower() in lat_names:
                    lat_var = var
                elif var.lower() in lon_names:
                    lon_var = var
            
            if not lat_var:
                print("Warning: Latitude coordinate not found. Available coords:", list(ds.coords))
                return ds
                
            if not lon_var:
                print("Warning: Longitude coordinate not found. Available coords:", list(ds.coords))
                return ds
                
            print(f"Extracting region: lon={lon_min}-{lon_max}, lat={lat_min}-{lat_max}")
            
            # For chunked data, we'll use .sel() which maintains chunking
            region_ds = ds.sel({
                lon_var: slice(lon_min, lon_max), 
                lat_var: slice(lat_min, lat_max)
            })
            
            print(f"Region extracted: {region_ds[lon_var].size} x {region_ds[lat_var].size} grid points")
            return region_ds
        except Exception as e:
            print(f"Error extracting region: {str(e)}")
            return ds
    
    def extract_time_series(self, ds, variable=None, lat=None, lon=None, method='mean', compute=True):
        """
        Extract a time series from the dataset using dask-optimized operations.
        
        Parameters:
        -----------
        ds : xarray.Dataset
            Dataset from read_file
        variable : str, optional
            Specific variable to extract
        lat, lon : float, optional
            Latitude and longitude for point extraction (instead of spatial averaging)
        method : str, optional
            Method for spatial aggregation ('mean', 'max', 'min', 'sum')
        compute : bool, optional
            Whether to compute the result immediately or return a dask array
            
        Returns:
        --------
        pd.DataFrame or xarray.DataArray
            Time series data
        """
        if ds is None:
            print("Cannot extract time series: dataset is None")
            return None
        
        try:
            # Identify coordinate names
            lat_names = ['latitude', 'lat']
            lon_names = ['longitude', 'lon']
            time_names = ['time', 't']
            step_names = ['step', 'forecast_time']
            
            lat_var = None
            lon_var = None
            time_var = None
            step_var = None
            
            # Find the coordinate variable names
            for var in ds.coords:
                if var.lower() in lat_names:
                    lat_var = var
                elif var.lower() in lon_names:
                    lon_var = var
                elif var.lower() in time_names:
                    time_var = var
                elif var.lower() in step_names:
                    step_var = var
            
            # If coordinate variables weren't found, print a warning
            if not all([lat_var, lon_var, time_var]):
                print(f"Warning: Some coordinate variables not found. Using available dimensions.")
                print(f"Available coordinates: {list(ds.coords)}")
                print(f"Available dimensions: {list(ds.dims)}")
                
                # Try to use dimensions directly if coordinates weren't found
                dims = list(ds.dims)
                if not lat_var and any(d.lower() in lat_names for d in dims):
                    lat_var = next(d for d in dims if d.lower() in lat_names)
                if not lon_var and any(d.lower() in lon_names for d in dims):
                    lon_var = next(d for d in dims if d.lower() in lon_names)
                if not time_var and any(d.lower() in time_names for d in dims):
                    time_var = next(d for d in dims if d.lower() in time_names)
                if not step_var and any(d.lower() in step_names for d in dims):
                    step_var = next(d for d in dims if d.lower() in step_names)
            
            # Select specific variable if requested
            if variable and variable in ds:
                ds_var = ds[variable]
                print(f"Using variable: {variable}")
            else:
                # Use first data variable if none specified
                data_vars = list(ds.data_vars)
                if not data_vars:
                    print("Error: No data variables found in dataset")
                    return None
                    
                ds_var = ds[data_vars[0]]
                variable = data_vars[0]
                print(f"No specific variable requested, using: {variable}")
            
            # If both lat/lon were provided, extract a point time series
            if lat is not None and lon is not None and lat_var and lon_var:
                print(f"Extracting point time series at lat={lat}, lon={lon}")
                point_ds = ds_var.sel({lat_var: lat, lon_var: lon}, method='nearest')
                
                # Print coordinates of the selected point
                sel_lat = float(point_ds[lat_var].values)
                sel_lon = float(point_ds[lon_var].values)
                print(f"Nearest grid point: lat={sel_lat}, lon={sel_lon}")
                
                # Convert to DataFrame for point time series
                if compute:
                    if time_var in point_ds.coords:
                        point_df = point_ds.to_dataframe()
                        # Keep only the variable column
                        if isinstance(point_df, pd.DataFrame) and variable in point_df.columns:
                            return point_df[[variable]]
                        return point_df
                    else:
                        # No time dimension, just return the value
                        return pd.DataFrame({variable: [float(point_ds.values)]})
                else:
                    # Return the DataArray without computing
                    return point_ds
            
            # Improved approach for spatial aggregation to reduce memory usage
            # First, reduce spatial dimensions
            print(f"Extracting time series with spatial {method} (memory-optimized)")
            
            # Get spatial dimensions (exclude time and step dimensions)
            spatial_dims = [dim for dim in ds_var.dims 
                           if dim not in [time_var, step_var] 
                           and dim is not None]
            
            print(f"Spatial dimensions to reduce: {spatial_dims}")
            
            # For large grids, we'll use coarsen first to reduce memory pressure
            if (lat_var in ds_var.dims and ds_var[lat_var].size > 500) or \
               (lon_var in ds_var.dims and ds_var[lon_var].size > 500):
                print("Large spatial grid detected, using coarsening before aggregation")
                
                # Calculate coarsening factor - aim for around 100-200 grid points in each dimension
                lat_coarsen = max(1, ds_var[lat_var].size // 150) if lat_var in ds_var.dims else 1
                lon_coarsen = max(1, ds_var[lon_var].size // 150) if lon_var in ds_var.dims else 1
                
                print(f"Coarsening factors: lat={lat_coarsen}, lon={lon_coarsen}")
                
                # Create coarsening dictionary
                coarsen_dict = {}
                if lat_var in ds_var.dims:
                    coarsen_dict[lat_var] = lat_coarsen
                if lon_var in ds_var.dims:
                    coarsen_dict[lon_var] = lon_coarsen
                
                # Apply coarsening
                if coarsen_dict:
                    ds_var = ds_var.coarsen(dim=coarsen_dict, boundary='trim').mean()
                    print(f"Coarsened grid size: {ds_var[lat_var].size if lat_var in ds_var.dims else 'N/A'} x " 
                          f"{ds_var[lon_var].size if lon_var in ds_var.dims else 'N/A'}")
            
            # Apply the requested aggregation method
            try:
                if method == 'mean':
                    ts = ds_var.mean(dim=spatial_dims)
                elif method == 'max':
                    ts = ds_var.max(dim=spatial_dims)
                elif method == 'min':
                    ts = ds_var.min(dim=spatial_dims)
                elif method == 'sum':
                    ts = ds_var.sum(dim=spatial_dims)
                else:
                    print(f"Unknown method: {method}, using mean")
                    ts = ds_var.mean(dim=spatial_dims)
            except Exception as e:
                print(f"Error in spatial aggregation: {str(e)}")
                print("Falling back to progressive computation...")
                # Fall back to progressive computation if aggregation fails
                return self.extract_time_series_progressive(ds, variable, method)
            
            # Convert to DataFrame if requested
            if compute:
                try:
                    print("Computing result...")
                    if HAVE_DASK:
                        with ProgressBar():
                            result = ts.compute()
                    else:
                        result = ts.compute()
                    
                    print("Computation complete")
                    
                    # Convert to DataFrame
                    try:
                        df = result.to_dataframe()
                        return df
                    except Exception as e:
                        print(f"Error converting to DataFrame: {str(e)}")
                        return result
                except Exception as e:
                    print(f"Error during computation: {str(e)}")
                    print("Falling back to progressive computation...")
                    return self.extract_time_series_progressive(ds, variable, method)
            else:
                print("Returning lazy dask array (not computed)")
                return ts
                
        except Exception as e:
            print(f"Error extracting time series: {str(e)}")
            print("Tip: Try using a specific variable and/or region to reduce memory usage")
            return None
    
    def extract_time_series_progressive(self, ds, variable=None, method='mean'):
        """
        Extract a time series using progressive computation to handle large datasets.
        Process one time step at a time to reduce memory usage.
        
        Parameters:
        -----------
        ds : xarray.Dataset
            Dataset from read_file
        variable : str, optional
            Specific variable to extract
        method : str, optional
            Method for spatial aggregation ('mean', 'max', 'min', 'sum')
            
        Returns:
        --------
        pd.DataFrame
            Time series data
        """
        if ds is None:
            print("Cannot extract time series: dataset is None")
            return None
        
        print("Using progressive time series extraction...")
        
        try:
            # Identify coordinate names
            lat_names = ['latitude', 'lat']
            lon_names = ['longitude', 'lon']
            time_names = ['time', 't']
            step_names = ['step', 'forecast_time']
            
            # Find the coordinate variables
            lat_var = next((var for var in ds.coords if var.lower() in lat_names), None)
            lon_var = next((var for var in ds.coords if var.lower() in lon_names), None)
            time_var = next((var for var in ds.coords if var.lower() in time_names), None)
            step_var = next((var for var in ds.coords if var.lower() in step_names), None)
            
            # Select variable
            if variable and variable in ds:
                var_data = ds[variable]
            else:
                # Use first data variable if none specified
                data_vars = list(ds.data_vars)
                if not data_vars:
                    print("Error: No data variables found in dataset")
                    return None
                variable = data_vars[0]
                var_data = ds[variable]
            
            print(f"Processing variable: {variable}")
            
            # Find spatial dimensions
            spatial_dims = [dim for dim in var_data.dims 
                           if dim not in [time_var, step_var] 
                           and dim is not None]
            
            print(f"Spatial dimensions to aggregate: {spatial_dims}")
            
            # Prepare result container
            result_list = []
            
            # Process one time step at a time
            num_times = len(ds[time_var]) if time_var else 1
            num_steps = len(ds[step_var]) if step_var else 1
            
            total_iterations = num_times * num_steps
            print(f"Total iterations: {total_iterations} (times: {num_times}, steps: {num_steps})")
            
            for t_idx in range(num_times):
                # Get current time value
                if time_var:
                    current_time = ds[time_var].isel({time_var: t_idx}).values
                    print(f"Processing time {t_idx+1}/{num_times}: {pd.to_datetime(current_time)}")
                else:
                    current_time = None
                
                for s_idx in range(num_steps):
                    # Get current step value
                    if step_var:
                        current_step = ds[step_var].isel({step_var: s_idx}).values
                        if t_idx == 0:  # Only print steps for the first time to avoid too much output
                            print(f"  Processing step {s_idx+1}/{num_steps}")
                    else:
                        current_step = None
                    
                    # Extract single time/step slice
                    slice_dict = {}
                    if time_var:
                        slice_dict[time_var] = t_idx
                    if step_var:
                        slice_dict[step_var] = s_idx
                    
                    # Extract the slice
                    try:
                        slice_data = var_data.isel(slice_dict)
                        
                        # Apply aggregation
                        if method == 'mean':
                            value = float(slice_data.mean(dim=spatial_dims).values)
                        elif method == 'max':
                            value = float(slice_data.max(dim=spatial_dims).values)
                        elif method == 'min':
                            value = float(slice_data.min(dim=spatial_dims).values)
                        elif method == 'sum':
                            value = float(slice_data.sum(dim=spatial_dims).values)
                        else:
                            value = float(slice_data.mean(dim=spatial_dims).values)
                        
                        # Build result entry
                        result_entry = {}
                        if time_var:
                            result_entry['time'] = current_time
                        if step_var:
                            result_entry['step'] = current_step
                        result_entry[variable] = value
                        
                        result_list.append(result_entry)
                    except Exception as e:
                        print(f"  Error processing slice {slice_dict}: {str(e)}")
            
            # Convert results to DataFrame
            if result_list:
                df = pd.DataFrame(result_list)
                
                # Convert time column to datetime if present
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                
                # Set index if appropriate
                if 'time' in df.columns and 'step' in df.columns:
                    # Multi-index with time and step
                    df = df.set_index(['time', 'step'])
                elif 'time' in df.columns:
                    # Index by time only
                    df = df.set_index('time')
                elif 'step' in df.columns:
                    # Index by step only
                    df = df.set_index('step')
                
                print(f"Successfully extracted time series: {len(df)} data points")
                return df
            else:
                print("No data extracted")
                return None
                
        except Exception as e:
            print(f"Error in progressive extraction: {str(e)}")
            return None
    
    def downsample_dataset(self, ds, spatial_factor=5, time_factor=1):
        """
        Downsample a dataset to reduce its size for analysis.
        
        Parameters:
        -----------
        ds : xarray.Dataset
            Dataset to downsample
        spatial_factor : int, optional
            Factor by which to reduce spatial resolution
        time_factor : int, optional
            Factor by which to reduce temporal resolution
            
        Returns:
        --------
        xarray.Dataset
            Downsampled dataset
        """
        if ds is None:
            return None
            
        try:
            print(f"Downsampling dataset: spatial={spatial_factor}x, time={time_factor}x")
            
            # Identify coordinate dimensions
            lat_names = ['latitude', 'lat']
            lon_names = ['longitude', 'lon']
            time_names = ['time', 't']
            
            lat_var = None
            lon_var = None
            time_var = None
            
            # Find the coordinate variable names
            for var in ds.coords:
                if var.lower() in lat_names:
                    lat_var = var
                elif var.lower() in lon_names:
                    lon_var = var
                elif var.lower() in time_names:
                    time_var = var
            
            # Create coarsening dictionary
            coarsen_dict = {}
            
            if lat_var and lat_var in ds.dims and spatial_factor > 1:
                coarsen_dict[lat_var] = spatial_factor
            
            if lon_var and lon_var in ds.dims and spatial_factor > 1:
                coarsen_dict[lon_var] = spatial_factor
            
            if time_var and time_var in ds.dims and time_factor > 1:
                coarsen_dict[time_var] = time_factor
            
            if not coarsen_dict:
                print("No dimensions to downsample, returning original dataset")
                return ds
            
            # Apply coarsening
            downsampled = ds.coarsen(dim=coarsen_dict, boundary='pad').mean()
            
            # Log the new dimensions
            print("Original dimensions:", {dim: size for dim, size in ds.dims.items()})
            print("Downsampled dimensions:", {dim: size for dim, size in downsampled.dims.items()})
            
            return downsampled
            
        except Exception as e:
            print(f"Error downsampling dataset: {str(e)}")
            return ds
    
    def calculate_wind_speed(self, ds, u_var=None, v_var=None):
        """
        Calculate wind speed from u and v components if present.
        
        Parameters:
        -----------
        ds : xarray.Dataset
            Dataset containing u and v wind components
        u_var, v_var : str, optional
            Names of the u and v component variables
            
        Returns:
        --------
        xarray.Dataset
            Dataset with added wind speed variable
        """
        if ds is None:
            return None
        
        try:
            # Find u and v component variables if not specified
            if u_var is None or v_var is None:
                u_candidates = []
                v_candidates = []
                
                for var_name in ds.data_vars:
                    if 'u_component' in var_name.lower() or 'u-component' in var_name.lower():
                        u_candidates.append(var_name)
                    elif 'v_component' in var_name.lower() or 'v-component' in var_name.lower():
                        v_candidates.append(var_name)
                
                if u_candidates and not u_var:
                    u_var = u_candidates[0]
                
                if v_candidates and not v_var:
                    v_var = v_candidates[0]
            
            if u_var in ds and v_var in ds:
                print(f"Found wind components: {u_var}, {v_var}")
                
                # Create a copy if working with data in memory, or use the original for dask arrays
                if hasattr(ds, 'chunks') and ds.chunks:
                    # Working with chunked data, use the original
                    ds_copy = ds
                else:
                    # Working with in-memory data, create a copy
                    ds_copy = ds.copy()
                
                # Calculate wind speed (uses dask arrays if inputs are dask arrays)
                ds_copy['wind_speed'] = np.sqrt(ds_copy[u_var]**2 + ds_copy[v_var]**2)
                
                # Add metadata
                ds_copy['wind_speed'].attrs['long_name'] = 'Wind speed'
                ds_copy['wind_speed'].attrs['units'] = 'm/s'
                
                print("Added wind_speed variable to dataset")
                return ds_copy
            else:
                print(f"Wind components not found in dataset. Available variables: {list(ds.data_vars)}")
                return ds
                
        except Exception as e:
            print(f"Error calculating wind speed: {str(e)}")
            return ds
    
    def plot_time_series(self, data, variable=None, title=None, figsize=(12, 6), compute_if_lazy=True):
        """
        Plot a time series from a DataFrame or DataArray.
        
        Parameters:
        -----------
        data : pd.DataFrame or xarray.DataArray
            Data containing time series to plot
        variable : str, optional
            Variable to plot (if None, plot first numeric column)
        title : str, optional
            Plot title
        figsize : tuple, optional
            Figure size
        compute_if_lazy : bool, optional
            Whether to compute dask arrays before plotting
            
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if data is None:
            print("Cannot plot: Data is None")
            return None
        
        # Handle lazy dask arrays
        if hasattr(data, 'compute') and compute_if_lazy:
            print("Computing lazy data before plotting...")
            try:
                if HAVE_DASK:
                    with ProgressBar():
                        data = data.compute()
                else:
                    data = data.compute()
                print("Computation complete")
            except Exception as e:
                print(f"Error computing data: {str(e)}")
                print("Try using a smaller region or single variable to reduce memory usage")
                return None
        
        # Handle DataArray
        if hasattr(data, 'to_dataframe'):
            try:
                print("Converting DataArray to DataFrame for plotting")
                data = data.to_dataframe()
            except Exception as e:
                print(f"Error converting to DataFrame: {str(e)}")
                return None
        
        if isinstance(data, pd.DataFrame) and data.empty:
            print("Cannot plot: DataFrame is empty")
            return None
        
        # If no variable specified, find the first numeric column
        if variable is None:
            if isinstance(data, pd.DataFrame):
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    variable = numeric_cols[0]
                    print(f"No variable specified, using: {variable}")
                else:
                    print("No numeric variables found in DataFrame")
                    return None
            else:
                # If it's not a DataFrame, we might be working with a Series
                variable = data.name if hasattr(data, 'name') else 'value'
        
        # Check if the variable exists
        if isinstance(data, pd.DataFrame) and variable not in data.columns:
            print(f"Variable {variable} not found. Available: {data.columns.tolist()}")
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        if isinstance(data, pd.DataFrame):
            # Plot from DataFrame
            data[variable].plot(ax=ax)
        else:
            # Direct plot for series or other objects
            data.plot(ax=ax)
        
        # Set title and labels
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Time series of {variable}")
        
        ax.set_xlabel("Time/Index")
        ax.set_ylabel(variable)
        
        plt.tight_layout()
        return fig
    
    def clean_temp_files(self):
        """Remove temporary files."""
        try:
            # Recursively delete the entire temp directory
            shutil.rmtree(self.temp_dir)
            print(f"Removed temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"Error cleaning temporary files: {str(e)}")

    def find_era5_files(self):
        """Find all potential ERA5-Land data files."""
        # Look for files with various relevant extensions
        extensions = ['.nc', '.nc.gz', '.grb', '.grib', '.grib2']
        
        all_files = []
        for ext in extensions:
            files = list(self.data_dir.glob(f"**/*{ext}"))
            all_files.extend(files)
        
        # If no files found with those extensions, look for any files in expected structure
        if not all_files:
            # Try looking in directories that match ERA5 variable patterns
            wind_dirs = list(self.data_dir.glob("*wind*"))
            temp_dirs = list(self.data_dir.glob("*temp*"))
            
            potential_dirs = wind_dirs + temp_dirs
            
            for d in potential_dirs:
                if d.is_dir():
                    # Look for any files in year/month subdirectories
                    for year_dir in d.glob("20??"):
                        if year_dir.is_dir():
                            for month_dir in year_dir.glob("[0-9]") + year_dir.glob("[0-9][0-9]"):
                                if month_dir.is_dir():
                                    # Get all files in this directory
                                    files = list(month_dir.glob("*"))
                                    all_files.extend(files)
        
        return all_files


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ERA5-Land Compressed Data Reader with Chunked Processing')
    parser.add_argument('--data-dir', type=str, default='./data/raw/grided/era5land_vars',
                        help='Directory containing ERA5-Land data')
    parser.add_argument('--file', type=str, default=None,
                        help='Specific file to read (optional)')
    parser.add_argument('--var', type=str, default=None,
                        help='Specific variable to extract (optional)')
    parser.add_argument('--lat-min', type=float, default=None,
                        help='Minimum latitude for region extraction (optional)')
    parser.add_argument('--lat-max', type=float, default=None,
                        help='Maximum latitude for region extraction (optional)')
    parser.add_argument('--lon-min', type=float, default=None,
                        help='Minimum longitude for region extraction (optional)')
    parser.add_argument('--lon-max', type=float, default=None,
                        help='Maximum longitude for region extraction (optional)')
    parser.add_argument('--point-lat', type=float, default=None,
                        help='Latitude for point extraction (optional)')
    parser.add_argument('--point-lon', type=float, default=None,
                        help='Longitude for point extraction (optional)')
    parser.add_argument('--chunk-time', type=int, default=1,
                        help='Chunk size for time dimension (default: 1)')
    parser.add_argument('--chunk-step', type=int, default=1,
                        help='Chunk size for step dimension (default: 1)')
    parser.add_argument('--chunk-lat', type=int, default=100,
                        help='Chunk size for latitude dimension (default: 100)')
    parser.add_argument('--chunk-lon', type=int, default=100,
                        help='Chunk size for longitude dimension (default: 100)')
    parser.add_argument('--downsample', type=int, default=0,
                        help='Downsample factor for spatial dimensions (default: 0, no downsampling)')
    parser.add_argument('--sample-only', action='store_true',
                        help='Load only a sample of the data for exploration')
    parser.add_argument('--no-compute', action='store_true',
                        help='Do not compute results (return lazy dask arrays)')
    parser.add_argument('--create-client', action='store_true',
                        help='Create a Dask distributed client')
    parser.add_argument('--memory-limit', type=str, default='4GB',
                        help='Memory limit for Dask client (default: 4GB)')
    parser.add_argument('--progressive', action='store_true',
                        help='Use progressive time series computation')
    args = parser.parse_args()
    
    # Set up chunking
    chunks = {
        'time': args.chunk_time,
        'step': args.chunk_step,
        'latitude': args.chunk_lat, 
        'longitude': args.chunk_lon
    }
    
    # Create the reader
    reader = ERA5LandCompressedReader(
        data_dir=args.data_dir,
        create_client=args.create_client,
        memory_limit=args.memory_limit
    )
    
    try:
        # If a specific file is provided, read it
        if args.file:
            filepath = Path(args.file)
            if filepath.exists():
                print(f"\nReading specified file: {filepath}")
                
                # Print file info
                print(f"File size: {filepath.stat().st_size / (1024*1024):.2f} MB")
                file_type = reader.identify_file_type(filepath)
                print(f"File type: {file_type}")
                
                # Read the file with chunks
                ds = reader.read_file(
                    filepath, 
                    variables=[args.var] if args.var else None,
                    chunks=chunks,
                    sample_only=args.sample_only
                )
                
                if ds is not None:
                    print("\nDataset information:")
                    if hasattr(ds, 'dims'):
                        print(f"Dimensions: {ds.dims}")
                    if hasattr(ds, 'data_vars'):
                        print(f"Variables: {list(ds.data_vars)}")
                    
                    # Downsample if requested
                    if args.downsample > 0:
                        ds = reader.downsample_dataset(ds, spatial_factor=args.downsample)
                    
                    # Extract region if specified
                    if any([args.lat_min, args.lat_max, args.lon_min, args.lon_max]):
                        ds = reader.extract_region(
                            ds, 
                            lon_min=args.lon_min, 
                            lon_max=args.lon_max, 
                            lat_min=args.lat_min, 
                            lat_max=args.lat_max
                        )
                    
                    # Extract time series
                    if args.progressive:
                        # Use progressive computation
                        ts = reader.extract_time_series_progressive(
                            ds,
                            variable=args.var,
                            method='mean'
                        )
                    elif args.point_lat and args.point_lon:
                        # Extract point time series
                        ts = reader.extract_time_series(
                            ds, 
                            variable=args.var,
                            lat=args.point_lat,
                            lon=args.point_lon,
                            compute=not args.no_compute
                        )
                    else:
                        # Extract area-averaged time series
                        ts = reader.extract_time_series(
                            ds, 
                            variable=args.var,
                            method='mean',
                            compute=not args.no_compute
                        )
                    
                    if ts is not None:
                        if not args.no_compute or args.progressive:
                            print("\nTime series sample:")
                            if hasattr(ts, 'head'):
                                print(ts.head())
                            
                            # Plot
                            var_to_plot = args.var if args.var else None
                            fig = reader.plot_time_series(ts, var_to_plot)
                            output_file = f"era5land_ts_{Path(args.file).stem}.png"
                            plt.savefig(output_file)
                            print(f"Plot saved as: {output_file}")
                        else:
                            print("\nLazy computation object returned (not computed)")
                            print("To compute and plot, remove --no-compute flag")
            else:
                print(f"Specified file not found: {args.file}")
        
        # Otherwise, find and process all files
        else:
            # Find ERA5-Land files
            files = reader.find_era5_files()
            
            if not files:
                print("No ERA5-Land files found")
            else:
                print(f"\nFound {len(files)} potential ERA5-Land files")
                print("First few files:")
                for i, file in enumerate(files[:5]):
                    print(f"  {i+1}. {file}")
                
                # Try the first file
                first_file = files[0]
                print(f"\nAttempting to read: {first_file}")
                
                # Print file info
                print(f"File size: {first_file.stat().st_size / (1024*1024):.2f} MB")
                file_type = reader.identify_file_type(first_file)
                print(f"File type: {file_type}")
                
                # Read the file with chunks
                ds = reader.read_file(
                    first_file, 
                    variables=[args.var] if args.var else None,
                    chunks=chunks,
                    sample_only=args.sample_only
                )
                
                if ds is not None:
                    # Extract and plot
                    if args.progressive:
                        ts = reader.extract_time_series_progressive(
                            ds,
                            variable=args.var,
                            method='mean'
                        )
                    else:
                        ts = reader.extract_time_series(
                            ds, 
                            variable=args.var,
                            compute=not args.no_compute
                        )
                    
                    if ts is not None and (not args.no_compute or args.progressive):
                        var_to_plot = args.var if args.var else None
                        fig = reader.plot_time_series(ts, var_to_plot)
                        output_file = f"era5land_ts_{Path(first_file).stem}.png"
                        plt.savefig(output_file)
                        print(f"Plot saved as: {output_file}")
    
    finally:
        # Clean up temporary files
        reader.clean_temp_files()
        
        # Clean up any Dask client
        if hasattr(reader, 'client') and reader.client is not None:
            reader.client.close()
            print("Closed Dask client")