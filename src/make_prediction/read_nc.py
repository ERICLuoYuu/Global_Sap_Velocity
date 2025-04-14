import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import warnings

# Try to import required packages
try:
    import xarray as xr
    HAVE_XARRAY = True
except ImportError:
    HAVE_XARRAY = False
    print("xarray not installed. Please install with: pip install xarray")

try:
    import cfgrib
    HAVE_CFGRIB = True
except ImportError:
    HAVE_CFGRIB = False
    print("cfgrib not installed. Please install with: pip install cfgrib")


class ERA5LandGribReader:
    """
    A specialized reader for ERA5-Land data in GRIB format (potentially with .nc extension).
    """
    def __init__(self, data_dir='./data/raw/grided/era5land_vars'):
        """
        Initialize the ERA5-Land GRIB reader.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing ERA5-Land data
        """
        self.data_dir = Path(data_dir)
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Check dependencies
        self._check_dependencies()
    
    def _setup_logging(self):
        """Set up logging configuration."""
        logger = logging.getLogger("ERA5LandGribReader")
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(ch)
        
        return logger
    
    def _check_dependencies(self):
        """Check if required packages are installed."""
        self.logger.info("Checking required packages:")
        
        if HAVE_XARRAY:
            self.logger.info("✓ xarray is installed")
        else:
            self.logger.error("✗ xarray is not installed - required")
            self.logger.info("Install with: pip install xarray")
        
        if HAVE_CFGRIB:
            self.logger.info("✓ cfgrib is installed")
        else:
            self.logger.error("✗ cfgrib is not installed - required for GRIB files")
            self.logger.info("Install with: pip install cfgrib eccodes")
        
        try:
            import dask
            self.logger.info("✓ dask is installed")
        except ImportError:
            self.logger.warning("✗ dask is not installed - recommended for large files")
            self.logger.info("Install with: pip install dask")
    
    def find_era5_files(self, variable=None, year=None, month=None):
        """
        Find ERA5-Land data files in the directory structure.
        
        Parameters:
        -----------
        variable : str, optional
            Variable name to filter by
        year : int, optional
            Year to filter by
        month : int, optional
            Month to filter by
        
        Returns:
        --------
        list
            List of file paths
        """
        # Build path based on filters
        search_path = self.data_dir
        
        if variable:
            search_path = search_path / variable
        
        if year:
            search_path = search_path / str(year)
        
        if month:
            search_path = search_path / str(month)
        
        self.logger.info(f"Searching for data files in: {search_path}")
        
        # Search for files with different extensions
        extensions = ['.nc', '.grib', '.grb', '.grib2', '.grb2']
        all_files = []
        
        for ext in extensions:
            pattern = f"**/*{ext}" if variable is None else f"**/*{variable}*{ext}"
            files = list(search_path.glob(pattern))
            all_files.extend(files)
        
        # If no files found with extensions, look for any files
        if not all_files and search_path.exists():
            all_files = [f for f in search_path.glob("**/*") if f.is_file()]
        
        if all_files:
            self.logger.info(f"Found {len(all_files)} files")
            for i, f in enumerate(all_files[:5]):
                file_size = f.stat().st_size / (1024*1024)  # Size in MB
                self.logger.info(f"  {i+1}. {f} ({file_size:.2f} MB)")
            
            if len(all_files) > 5:
                self.logger.info(f"  ... and {len(all_files) - 5} more")
        else:
            self.logger.warning(f"No files found in {search_path}")
        
        return all_files
    
    def list_variables(self):
        """
        List all variable directories in the data directory.
        
        Returns:
        --------
        list
            List of variable names
        """
        try:
            var_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
            var_names = [d.name for d in var_dirs]
            
            self.logger.info(f"Found {len(var_names)} variables:")
            for var in var_names:
                self.logger.info(f"  - {var}")
            
            return var_names
        except Exception as e:
            self.logger.error(f"Error listing variables: {str(e)}")
            return []
    
    def identify_file_type(self, filepath):
        """
        Try to identify the format of a file.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the file
        
        Returns:
        --------
        str
            Description of the file type
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            return "File not found"
        
        if filepath.is_dir():
            return "Directory"
        
        # Check file extension
        ext = filepath.suffix.lower()
        if ext in ['.nc']:
            file_type = "NetCDF (by extension)"
        elif ext in ['.grib', '.grb', '.grib2', '.grb2']:
            file_type = "GRIB (by extension)"
        else:
            file_type = "Unknown format (by extension)"
        
        # Try to read the first few bytes to confirm
        try:
            with open(filepath, 'rb') as f:
                header = f.read(8)
                
                if header.startswith(b'GRIB'):
                    file_type = "GRIB (confirmed by header)"
                elif header.startswith(b'CDF'):
                    file_type = "NetCDF (confirmed by header)"
        except Exception as e:
            self.logger.warning(f"Could not read file header: {str(e)}")
        
        return file_type
    
    def read_file(self, filepath):
        """
        Read an ERA5-Land data file, automatically detecting the format.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the file
        
        Returns:
        --------
        xarray.Dataset
            The dataset containing the ERA5-Land data
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            self.logger.error(f"File not found: {filepath}")
            return None
        
        if filepath.is_dir():
            self.logger.error(f"The path is a directory, not a file: {filepath}")
            return None
        
        # Check file size
        file_size = filepath.stat().st_size / (1024*1024)  # Size in MB
        self.logger.info(f"Reading file: {filepath} ({file_size:.2f} MB)")
        
        # Identify file type
        file_type = self.identify_file_type(filepath)
        self.logger.info(f"File type: {file_type}")
        
        # Determine which engine to use
        if "GRIB" in file_type:
            engine = 'cfgrib'
        else:
            engine = None  # Let xarray choose the appropriate engine
        
        self.logger.info(f"Using engine: {engine}")
        
        # Try to read the file
        try:
            if engine == 'cfgrib':
                if not HAVE_CFGRIB:
                    self.logger.error("cfgrib package is required to read GRIB files")
                    self.logger.info("Install with: pip install cfgrib eccodes")
                    return None
                
                self.logger.info("Trying to open with cfgrib engine...")
                ds = xr.open_dataset(filepath, engine='cfgrib')
                self.logger.info("Successfully read file with cfgrib")
            else:
                self.logger.info("Trying to open with default xarray engine...")
                ds = xr.open_dataset(filepath)
                self.logger.info("Successfully read file with default xarray engine")
            
            # Display basic dataset info
            self.logger.info(f"Dataset dimensions: {dict(ds.dims)}")
            self.logger.info(f"Variables: {list(ds.data_vars)}")
            
            return ds
        except Exception as e:
            self.logger.error(f"Error reading file: {str(e)}")
            
            # Try alternate engines if the first one failed
            if engine == 'cfgrib':
                try:
                    self.logger.info("Trying fallback to default xarray engine...")
                    ds = xr.open_dataset(filepath)
                    self.logger.info("Successfully read file with default xarray engine")
                    return ds
                except Exception as e2:
                    self.logger.error(f"Fallback also failed: {str(e2)}")
            else:
                try:
                    if HAVE_CFGRIB:
                        self.logger.info("Trying fallback to cfgrib engine...")
                        ds = xr.open_dataset(filepath, engine='cfgrib')
                        self.logger.info("Successfully read file with cfgrib engine")
                        return ds
                except Exception as e2:
                    self.logger.error(f"Fallback also failed: {str(e2)}")
            
            return None
    
    def extract_time_series(self, ds, lat=None, lon=None, variable=None):
        """
        Extract a time series for a specific location and variable.
        
        Parameters:
        -----------
        ds : xarray.Dataset
            Dataset containing ERA5-Land data
        lat : float, optional
            Latitude
        lon : float, optional
            Longitude
        variable : str, optional
            Variable name to extract
        
        Returns:
        --------
        xarray.DataArray or pandas.DataFrame
            The time series data
        """
        if ds is None:
            self.logger.error("Cannot extract time series: dataset is None")
            return None
        
        self.logger.info("Extracting time series from dataset")
        
        # If variable is not specified, use the first data variable
        if variable is None and len(ds.data_vars) > 0:
            variable = list(ds.data_vars)[0]
            self.logger.info(f"No variable specified, using: {variable}")
        
        # Check if the variable exists
        if variable not in ds.data_vars:
            self.logger.error(f"Variable '{variable}' not found in dataset")
            self.logger.info(f"Available variables: {list(ds.data_vars)}")
            return None
        
        # Get the variable data array
        data_array = ds[variable]
        self.logger.info(f"Extracted variable: {variable}")
        
        # Extract for specific location if coordinates provided
        if lat is not None and lon is not None:
            # Find coordinate names (they can vary in different datasets)
            lat_name = None
            lon_name = None
            
            for dim in data_array.dims:
                if dim.lower() in ['latitude', 'lat']:
                    lat_name = dim
                elif dim.lower() in ['longitude', 'lon']:
                    lon_name = dim
            
            if lat_name is None or lon_name is None:
                for coord in data_array.coords:
                    if coord.lower() in ['latitude', 'lat']:
                        lat_name = coord
                    elif coord.lower() in ['longitude', 'lon']:
                        lon_name = coord
            
            if lat_name and lon_name:
                self.logger.info(f"Extracting at lat={lat}, lon={lon}")
                
                try:
                    # Get the point data using the nearest method
                    point_data = data_array.sel({
                        lat_name: lat,
                        lon_name: lon
                    }, method='nearest')
                    
                    # Print the actual coordinates used
                    actual_lat = float(point_data[lat_name].values)
                    actual_lon = float(point_data[lon_name].values)
                    self.logger.info(f"Actual coordinates: lat={actual_lat}, lon={actual_lon}")
                    
                    return point_data
                except Exception as e:
                    self.logger.error(f"Error selecting location: {str(e)}")
                    return data_array
            else:
                self.logger.warning("Could not identify latitude/longitude coordinates")
                return data_array
        else:
            self.logger.info("No specific location provided, returning the full data array")
            return data_array
    
    def convert_to_dataframe(self, data_array):
        """
        Convert a DataArray to a DataFrame with proper error handling.
        
        Parameters:
        -----------
        data_array : xarray.DataArray
            The data array to convert
        
        Returns:
        --------
        pandas.DataFrame
            The converted DataFrame
        """
        if data_array is None:
            self.logger.error("Cannot convert: data array is None")
            return None
        
        try:
            self.logger.info("Converting DataArray to DataFrame")
            df = data_array.to_dataframe()
            
            # Reset index for simpler handling
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
                
                # Try to identify time column
                time_cols = [col for col in df.columns if col.lower() in ['time', 'valid_time']]
                if time_cols:
                    time_col = time_cols[0]
                    df = df.set_index(time_col)
                    self.logger.info(f"Set index to {time_col}")
            
            self.logger.info(f"DataFrame shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error converting to DataFrame: {str(e)}")
            return None
    
    def calculate_wind_speed(self, ds):
        """
        Calculate wind speed from u and v components.
        
        Parameters:
        -----------
        ds : xarray.Dataset
            Dataset containing u and v wind components
        
        Returns:
        --------
        xarray.Dataset
            Dataset with wind_speed variable added
        """
        if ds is None:
            self.logger.error("Cannot calculate wind speed: dataset is None")
            return None
        
        # Find u and v component variables
        u_var = None
        v_var = None
        
        for var in ds.data_vars:
            if 'u_component_of_wind' in var or 'u10' in var.lower():
                u_var = var
            elif 'v_component_of_wind' in var or 'v10' in var.lower():
                v_var = var
        
        if u_var and v_var:
            self.logger.info(f"Found wind components: {u_var}, {v_var}")
            
            try:
                # Create a copy of the dataset
                ds_copy = ds.copy()
                
                # Calculate wind speed
                ds_copy['wind_speed'] = np.sqrt(ds_copy[u_var]**2 + ds_copy[v_var]**2)
                
                # Add metadata
                ds_copy['wind_speed'].attrs['long_name'] = 'Wind speed at 10m'
                ds_copy['wind_speed'].attrs['units'] = 'm s-1'
                
                self.logger.info("Successfully calculated wind speed")
                return ds_copy
            except Exception as e:
                self.logger.error(f"Error calculating wind speed: {str(e)}")
                return ds
        else:
            self.logger.warning("Wind components not found in the dataset")
            return ds
    
    def convert_temperature_kelvin_to_celsius(self, ds, variable=None):
        """
        Convert temperature from Kelvin to Celsius.
        
        Parameters:
        -----------
        ds : xarray.Dataset
            Dataset containing temperature data
        variable : str, optional
            Temperature variable name. If None, try to identify automatically.
        
        Returns:
        --------
        xarray.Dataset
            Dataset with temperature in Celsius
        """
        if ds is None:
            self.logger.error("Cannot convert temperature: dataset is None")
            return None
        
        # If variable not specified, try to identify a temperature variable
        if variable is None:
            temp_vars = []
            for var in ds.data_vars:
                if var.lower() in ['t2m', '2t', 'temperature', '2m_temperature']:
                    temp_vars.append(var)
                elif 'temp' in var.lower() and 'dewpoint' not in var.lower():
                    temp_vars.append(var)
            
            if temp_vars:
                variable = temp_vars[0]
                self.logger.info(f"Identified temperature variable: {variable}")
            else:
                self.logger.warning("Could not identify temperature variable")
                return ds
        elif variable not in ds.data_vars:
            self.logger.error(f"Variable '{variable}' not found in dataset")
            self.logger.info(f"Available variables: {list(ds.data_vars)}")
            return ds
        
        # Check if conversion is needed
        units = None
        if 'units' in ds[variable].attrs:
            units = ds[variable].attrs['units']
        
        if units is None:
            # If no units specified, check values range
            if np.mean(ds[variable].values) > 100:
                self.logger.info("Temperature values appear to be in Kelvin (no units specified)")
                needs_conversion = True
            else:
                self.logger.info("Temperature values appear to already be in Celsius (no units specified)")
                needs_conversion = False
        elif units.lower() in ['k', 'kelvin']:
            self.logger.info(f"Temperature units are {units}, conversion needed")
            needs_conversion = True
        else:
            self.logger.info(f"Temperature units are {units}, no conversion needed")
            needs_conversion = False
        
        if needs_conversion:
            try:
                # Create a copy of the dataset
                ds_copy = ds.copy()
                
                # Convert Kelvin to Celsius
                ds_copy[variable] = ds[variable] - 273.15
                
                # Update units attribute
                ds_copy[variable].attrs['units'] = 'C'
                if 'units' in ds[variable].attrs:
                    ds_copy[variable].attrs['original_units'] = ds[variable].attrs['units']
                
                self.logger.info(f"Successfully converted {variable} from Kelvin to Celsius")
                return ds_copy
            except Exception as e:
                self.logger.error(f"Error converting temperature: {str(e)}")
                return ds
        else:
            # No conversion needed
            return ds
    
    def plot_data_array(self, data_array, title=None, figsize=(12, 6)):
        """
        Plot a DataArray time series.
        
        Parameters:
        -----------
        data_array : xarray.DataArray
            The data array to plot
        title : str, optional
            Plot title
        figsize : tuple, optional
            Figure size
        
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if data_array is None:
            self.logger.error("Cannot plot: data array is None")
            return None
        
        try:
            self.logger.info("Creating time series plot from DataArray")
            
            # Check for time dimension
            time_dim = None
            for dim in data_array.dims:
                if dim.lower() in ['time', 'valid_time']:
                    time_dim = dim
                    break
            
            if time_dim is None:
                self.logger.warning("No time dimension found, cannot create time series plot")
                return None
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Get the variable name
            var_name = data_array.name
            
            # Plot the data - different approach depending on dimensions
            if len(data_array.dims) == 1 and time_dim in data_array.dims:
                # 1D time series
                data_array.plot(ax=ax)
            else:
                # For multi-dimensional data, try to plot mean across spatial dimensions
                plot_data = data_array
                for dim in data_array.dims:
                    if dim != time_dim and dim.lower() not in ['latitude', 'longitude', 'lat', 'lon']:
                        plot_data = plot_data.mean(dim=dim)
                
                plot_data.plot(ax=ax)
            
            # Set title
            if title:
                ax.set_title(title)
            else:
                ax.set_title(f"Time series of {var_name}")
            
            # Set labels
            ax.set_xlabel("Time")
            
            # Try to get units
            if 'units' in data_array.attrs:
                y_label = f"{var_name} ({data_array.attrs['units']})"
            else:
                y_label = var_name
            
            ax.set_ylabel(y_label)
            
            # Add grid
            ax.grid(True)
            
            plt.tight_layout()
            
            self.logger.info(f"Created time series plot for {var_name}")
            return fig
        except Exception as e:
            self.logger.error(f"Error plotting DataArray: {str(e)}")
            return None
    
    def plot_dataframe(self, df, variable=None, title=None, figsize=(12, 6)):
        """
        Plot a DataFrame time series.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame to plot
        variable : str, optional
            Variable to plot
        title : str, optional
            Plot title
        figsize : tuple, optional
            Figure size
        
        Returns:
        --------
        matplotlib.figure.Figure
            The created figure
        """
        if df is None or df.empty:
            self.logger.error("Cannot plot: DataFrame is None or empty")
            return None
        
        # If variable not specified, use the first data column
        if variable is None:
            # Get the data columns (exclude coordinate columns)
            exclude_cols = ['latitude', 'lat', 'longitude', 'lon', 'x', 'y']
            data_cols = [col for col in df.columns if col.lower() not in exclude_cols]
            
            if not data_cols:
                self.logger.error("No data columns found in DataFrame")
                return None
            
            variable = data_cols[0]
            self.logger.info(f"No variable specified, using: {variable}")
        
        # Check if the variable exists
        if variable not in df.columns:
            self.logger.error(f"Variable '{variable}' not found in DataFrame")
            self.logger.info(f"Available columns: {list(df.columns)}")
            return None
        
        try:
            self.logger.info("Creating time series plot from DataFrame")
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot the data
            df[variable].plot(ax=ax)
            
            # Set title
            if title:
                ax.set_title(title)
            else:
                ax.set_title(f"Time series of {variable}")
            
            # Set labels
            ax.set_xlabel("Time")
            ax.set_ylabel(variable)
            
            # Add grid
            ax.grid(True)
            
            plt.tight_layout()
            
            self.logger.info(f"Created time series plot for {variable}")
            return fig
        except Exception as e:
            self.logger.error(f"Error plotting DataFrame: {str(e)}")
            return None
    
    def save_data_array_to_csv(self, data_array, output_path):
        """
        Save a DataArray to a CSV file.
        
        Parameters:
        -----------
        data_array : xarray.DataArray
            The data array to save
        output_path : str or Path
            Path to save the CSV file
        
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if data_array is None:
            self.logger.error("Cannot save: data array is None")
            return False
        
        try:
            # Convert to DataFrame first
            df = self.convert_to_dataframe(data_array)
            
            if df is None:
                self.logger.error("Failed to convert DataArray to DataFrame for CSV export")
                return False
            
            # Save to CSV
            df.to_csv(output_path)
            self.logger.info(f"Data saved to CSV: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving to CSV: {str(e)}")
            return False


# Command-line interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ERA5-Land GRIB Format Reader")
    parser.add_argument("--data-dir", type=str, default="./data/raw/grided/era5land_vars",
                        help="Directory containing ERA5-Land data")
    parser.add_argument("--list-vars", action="store_true",
                        help="List available variables")
    parser.add_argument("--variable", type=str, default=None,
                        help="Variable to process (e.g., '2m_temperature')")
    parser.add_argument("--year", type=int, default=None,
                        help="Year to process (e.g., 2016)")
    parser.add_argument("--month", type=int, default=None,
                        help="Month to process (e.g., 1)")
    parser.add_argument("--lat", type=float, default=None,
                        help="Latitude for point extraction")
    parser.add_argument("--lon", type=float, default=None,
                        help="Longitude for point extraction")
    parser.add_argument("--convert-temp", action="store_true",
                        help="Convert temperature from Kelvin to Celsius")
    parser.add_argument("--output-plot", type=str, default="era5_plot.png",
                        help="Output plot filename")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Output CSV filename (optional)")
    
    args = parser.parse_args()
    
    # Create the reader
    reader = ERA5LandGribReader(data_dir=args.data_dir)
    
    # List variables if requested
    if args.list_vars:
        reader.list_variables()
        return
    
    # Find data files based on filters
    files = reader.find_era5_files(
        variable=args.variable,
        year=args.year,
        month=args.month
    )
    
    if not files:
        print("No files found matching the criteria")
        return
    
    # Process the first file
    first_file = files[0]
    print(f"Processing file: {first_file}")
    
    # Read the file
    ds = reader.read_file(first_file)
    
    if ds is None:
        print("Failed to read the file")
        return
    
    # Convert temperature if requested
    if args.convert_temp:
        ds = reader.convert_temperature_kelvin_to_celsius(ds, variable=args.variable)
    
    # Extract time series
    data_array = reader.extract_time_series(
        ds,
        lat=args.lat,
        lon=args.lon,
        variable=args.variable
    )
    
    if data_array is None:
        print("Failed to extract time series")
        return
    
    # Plot the data
    fig = reader.plot_data_array(
        data_array,
        title=f"{args.variable} ({args.year}-{args.month})" if args.variable else None
    )
    
    if fig:
        # Save the plot
        plt.savefig(args.output_plot)
        print(f"Plot saved to: {args.output_plot}")
    
    # Save to CSV if requested
    if args.output_csv:
        reader.save_data_array_to_csv(data_array, args.output_csv)


if __name__ == "__main__":
    main()