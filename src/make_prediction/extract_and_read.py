import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob
from datetime import datetime
import cartopy.crs as ccrs
from metpy.units import units
from metpy.calc import relative_humidity_from_dewpoint, wind_speed, heat_index
from metpy.calc import apparent_temperature
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from dateutil.relativedelta import relativedelta

class ERA5Analyzer:
    """
    Class for analyzing ERA5-Land data with advanced operations
    """
    def __init__(self, data_dir):
        """
        Initialize the analyzer with a directory containing ERA5 data
        
        Parameters:
        -----------
        data_dir : str
            Directory containing extracted .nc files
        """
        self.data_dir = Path(data_dir)
        self.datasets = {}
        
    def load_variable(self, variable_name, year=None, month=None):
        """
        Load data for a specific variable, optionally filtered by year and month
        
        Parameters:
        -----------
        variable_name : str
            Name of the variable to load (folder name)
        year : int, optional
            Year to filter by
        month : int, optional
            Month to filter by
            
        Returns:
        --------
        xarray.Dataset
            Combined dataset for the specified variable
        """
        # Build path pattern based on inputs
        path_pattern = f"{variable_name}"
        if year is not None:
            path_pattern = f"{path_pattern}/{year}"
            if month is not None:
                path_pattern = f"{path_pattern}/{month:02d}"
        
        path_pattern = f"{path_pattern}/**/*.nc"
        
        # Find all matching files
        file_paths = sorted(self.data_dir.glob(path_pattern))
        
        if not file_paths:
            print(f"No files found for {variable_name}" + 
                  (f" in year {year}" if year else "") + 
                  (f", month {month}" if month else ""))
            return None
        
        print(f"Loading {len(file_paths)} files for {variable_name}" + 
              (f" in year {year}" if year else "") + 
              (f", month {month}" if month else ""))
        
        # Open and combine datasets
        datasets = []
        for file_path in file_paths:
            try:
                ds = xr.open_dataset(file_path)
                datasets.append(ds)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        if not datasets:
            return None
        
        # Combine datasets along time dimension
        combined_ds = xr.concat(datasets, dim='time')
        
        # Sort by time to ensure chronological order
        combined_ds = combined_ds.sortby('time')
        
        # Store in instance cache
        key = f"{variable_name}_{year if year else 'all'}_{month if month else 'all'}"
        self.datasets[key] = combined_ds
        
        return combined_ds
    
    def load_multiple_variables(self, variable_names, year=None, month=None):
        """
        Load multiple variables and merge them into a single dataset
        
        Parameters:
        -----------
        variable_names : list
            List of variable names to load
        year : int, optional
            Year to filter by
        month : int, optional
            Month to filter by
            
        Returns:
        --------
        xarray.Dataset
            Merged dataset containing all variables
        """
        datasets = []
        
        for var_name in variable_names:
            ds = self.load_variable(var_name, year, month)
            if ds is not None:
                datasets.append(ds)
        
        if not datasets:
            return None
        
        # Merge datasets
        merged_ds = xr.merge(datasets)
        
        return merged_ds
    
    def extract_time_series(self, dataset, variable, lat, lon, method='nearest'):
        """
        Extract a time series for a specific location
        
        Parameters:
        -----------
        dataset : xarray.Dataset
            Dataset containing the variable
        variable : str
            Variable name
        lat : float
            Latitude
        lon : float
            Longitude
        method : str
            Method for selecting location ('nearest' or 'linear')
            
        Returns:
        --------
        xarray.DataArray
            Time series for the specified location
        """
        # Adjust longitude to match ERA5 conventions if needed
        if lon > 180:
            lon = lon - 360
        
        # Select the point
        if method == 'nearest':
            point_data = dataset[variable].sel(latitude=lat, longitude=lon, method='nearest')
        elif method == 'linear':
            point_data = dataset[variable].interp(latitude=lat, longitude=lon)
        else:
            raise ValueError(f"Method {method} not supported. Use 'nearest' or 'linear'.")
        
        return point_data
    
    def calculate_daily_statistics(self, dataset, variable, stat='mean'):
        """
        Calculate daily statistics for a variable
        
        Parameters:
        -----------
        dataset : xarray.Dataset
            Dataset containing the variable
        variable : str
            Variable name
        stat : str
            Statistic to calculate ('mean', 'min', 'max', or 'sum')
            
        Returns:
        --------
        xarray.Dataset
            Dataset with daily statistics
        """
        # Ensure dataset has time dimension
        if 'time' not in dataset.dims:
            print("Dataset doesn't have a time dimension")
            return None
        
        # Make sure variable exists
        if variable not in dataset:
            print(f"Variable {variable} not found in dataset")
            return None
        
        # Calculate daily statistics
        if stat == 'mean':
            daily_stat = dataset[variable].resample(time='1D').mean()
        elif stat == 'min':
            daily_stat = dataset[variable].resample(time='1D').min()
        elif stat == 'max':
            daily_stat = dataset[variable].resample(time='1D').max()
        elif stat == 'sum':
            daily_stat = dataset[variable].resample(time='1D').sum()
        else:
            raise ValueError(f"Statistic {stat} not supported. Use 'mean', 'min', 'max', or 'sum'.")
        
        # Create a new dataset with the daily statistics
        daily_ds = xr.Dataset({f"{variable}_{stat}_daily": daily_stat})
        
        return daily_ds
    
    def calculate_monthly_climatology(self, dataset, variable):
        """
        Calculate monthly climatology for a variable
        
        Parameters:
        -----------
        dataset : xarray.Dataset
            Dataset containing the variable
        variable : str
            Variable name
            
        Returns:
        --------
        xarray.Dataset
            Dataset with monthly climatology
        """
        # Group by month and calculate mean
        monthly_climatology = dataset[variable].groupby('time.month').mean()
        
        # Create a new dataset
        climatology_ds = xr.Dataset({f"{variable}_climatology": monthly_climatology})
        
        return climatology_ds
    
    def calculate_anomalies(self, dataset, variable, reference_period=None):
        """
        Calculate anomalies relative to a reference period
        
        Parameters:
        -----------
        dataset : xarray.Dataset
            Dataset containing the variable
        variable : str
            Variable name
        reference_period : tuple, optional
            Start and end years for reference period (e.g., (1981, 2010))
            
        Returns:
        --------
        xarray.Dataset
            Dataset with anomalies
        """
        # Calculate climatology
        if reference_period:
            start_year, end_year = reference_period
            
            # Filter dataset to reference period
            mask = ((dataset.time.dt.year >= start_year) & 
                    (dataset.time.dt.year <= end_year))
            reference_data = dataset.sel(time=mask)
            
            # Calculate monthly climatology for reference period
            climatology = reference_data[variable].groupby('time.month').mean('time')
        else:
            # Use the entire dataset as reference
            climatology = dataset[variable].groupby('time.month').mean('time')
        
        # Calculate anomalies
        anomalies = dataset[variable].groupby('time.month') - climatology
        
        # Create a new dataset
        anomalies_ds = xr.Dataset({f"{variable}_anomaly": anomalies})
        
        return anomalies_ds
    
    def calculate_regional_mean(self, dataset, variable, lat_min, lat_max, lon_min, lon_max):
        """
        Calculate area-weighted mean over a region
        
        Parameters:
        -----------
        dataset : xarray.Dataset
            Dataset containing the variable
        variable : str
            Variable name
        lat_min, lat_max, lon_min, lon_max : float
            Boundaries of the region
            
        Returns:
        --------
        xarray.DataArray
            Area-weighted mean time series
        """
        # Adjust longitudes if needed
        if lon_min > 180:
            lon_min = lon_min - 360
        if lon_max > 180:
            lon_max = lon_max - 360
        
        # Select the region
        region = dataset[variable].sel(
            latitude=slice(lat_min, lat_max),
            longitude=slice(lon_min, lon_max)
        )
        
        # Calculate weights (proportional to grid cell area, approximated by cosine of latitude)
        weights = np.cos(np.deg2rad(region.latitude))
        
        # Calculate weighted mean
        regional_mean = region.weighted(weights).mean(dim=('latitude', 'longitude'))
        
        return regional_mean
    
    def plot_map(self, dataset, variable, time_idx=0, title=None, 
                 vmin=None, vmax=None, cmap='viridis', projection='PlateCarree',
                 save_path=None):
        """
        Create a map plot of a variable
        
        Parameters:
        -----------
        dataset : xarray.Dataset
            Dataset containing the variable
        variable : str
            Variable name
        time_idx : int
            Time index to plot
        title : str, optional
            Plot title
        vmin, vmax : float, optional
            Color scale limits
        cmap : str
            Colormap name
        projection : str
            Map projection
        save_path : str, optional
            Path to save the figure
        """
        # Select data at specified time
        if 'time' in dataset[variable].dims:
            time_value = dataset.time.values[time_idx]
            data = dataset[variable].isel(time=time_idx)
            time_str = np.datetime_as_string(time_value, unit='h')
        else:
            data = dataset[variable]
            time_str = ""
        
        # Create figure with cartopy projection
        proj = getattr(ccrs, projection)()
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': proj})
        
        # Add map features
        ax.coastlines(resolution='50m')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.gridlines(draw_labels=True)
        
        # Plot data
        im = data.plot(
            ax=ax, 
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=True
        )
        
        # Set title
        if title:
            plt.title(title)
        else:
            var_name = dataset[variable].attrs.get('long_name', variable)
            plt.title(f"{var_name} - {time_str}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_time_series(self, data, title=None, ylabel=None, 
                         resample=None, save_path=None):
        """
        Plot a time series
        
        Parameters:
        -----------
        data : xarray.DataArray
            Time series data
        title : str, optional
            Plot title
        ylabel : str, optional
            Y-axis label
        resample : str, optional
            Resample frequency (e.g., 'D', 'M', 'Y')
        save_path : str, optional
            Path to save the figure
        """
        # Resample if requested
        if resample:
            if resample == 'D':
                plot_data = data.resample(time='1D').mean()
                resample_label = 'Daily'
            elif resample == 'M':
                plot_data = data.resample(time='1M').mean()
                resample_label = 'Monthly'
            elif resample == 'Y':
                plot_data = data.resample(time='1Y').mean()
                resample_label = 'Annual'
            else:
                print(f"Resample frequency {resample} not recognized, using original data")
                plot_data = data
                resample_label = ''
        else:
            plot_data = data
            resample_label = ''
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot time series
        plt.plot(plot_data.time, plot_data.values)
        
        # Set labels
        if title:
            plt.title(title)
        else:
            var_name = data.attrs.get('long_name', data.name)
            plt.title(f"{resample_label} {var_name}")
        
        if ylabel:
            plt.ylabel(ylabel)
        elif 'units' in data.attrs:
            plt.ylabel(data.attrs['units'])
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

# Example usage
if __name__ == "__main__":
    # Replace with your actual data directory
    era5_data_dir = "./data/raw/grided/era5land_vars"
    
    analyzer = ERA5Analyzer(era5_data_dir)
    
    # Example 1: Load temperature data for 2017
    temp_data = analyzer.load_variable('2m_temperature', year=2017)
    
    if temp_data is not None:
        # Example 2: Extract time series for a specific location
        location_ts = analyzer.extract_time_series(
            temp_data, '2m_temperature', 
            lat=40.7128, lon=-74.0060  # New York City
        )
        
        # Convert from Kelvin to Celsius
        location_ts_celsius = location_ts - 273.15
        location_ts_celsius.attrs['units'] = '°C'
        
        # Example 3: Plot daily mean temperatures
        analyzer.plot_time_series(
            location_ts_celsius, 
            title="Daily Mean Temperature in New York (2017)",
            resample='D',
            save_path="nyc_temperature_2017.png"
        )
        
        # Example 4: Calculate and plot monthly climatology
        monthly_clim = analyzer.calculate_monthly_climatology(temp_data, '2m_temperature')
        
        # Example 5: Create a map for a specific time
        analyzer.plot_map(
            temp_data, '2m_temperature', 
            time_idx=0,  # First time step
            title="ERA5-Land 2m Temperature",
            save_path="temperature_map.png"
        )
        
    # Example 6: Load and process wind data
    wind_u = analyzer.load_variable('10m_u_component_of_wind', year=2017, month=7)
    wind_v = analyzer.load_variable('10m_v_component_of_wind', year=2017, month=7)
    
    if wind_u is not None and wind_v is not None:
        # Example 7: Calculate wind speed
        wind_speed = np.sqrt(wind_u['10m_u_component_of_wind']**2 + 
                            wind_v['10m_v_component_of_wind']**2)
        wind_speed.name = 'wind_speed'
        wind_speed.attrs['units'] = 'm s**-1'
        wind_speed.attrs['long_name'] = '10m Wind Speed'
        
        # Example 8: Create a combined dataset
        wind_ds = xr.Dataset({
            'wind_speed': wind_speed,
            'u_component': wind_u['10m_u_component_of_wind'],
            'v_component': wind_v['10m_v_component_of_wind']
        })
        
        # Example 9: Map of wind speed
        analyzer.plot_map(
            wind_ds, 'wind_speed',
            time_idx=0,
            title="ERA5-Land 10m Wind Speed (July 2017)",
            cmap='YlOrRd',
            save_path="wind_speed_map.png"
        )
class ERA5DerivedVariables:
    """
    Class for calculating derived variables from ERA5-Land data
    """
    def __init__(self, era5_analyzer=None):
        """
        Initialize with an optional ERA5Analyzer instance
        
        Parameters:
        -----------
        era5_analyzer : ERA5Analyzer, optional
            Instance of ERA5Analyzer for loading data
        """
        self.era5_analyzer = era5_analyzer
    
    def _ensure_same_dimensions(self, *datasets):
        """
        Ensure all datasets have the same dimensions and coordinates
        
        Returns:
        --------
        list of xarray.Dataset
            List of aligned datasets
        """
        if len(datasets) <= 1:
            return datasets
        
        # Find common dimensions
        common_dims = set(datasets[0].dims)
        for ds in datasets[1:]:
            common_dims = common_dims.intersection(ds.dims)
        
        # Align datasets along common dimensions
        aligned_datasets = xr.align(*datasets, join='inner')
        
        return aligned_datasets
    
    def calculate_relative_humidity(self, temperature_ds, dewpoint_ds):
        """
        Calculate relative humidity from temperature and dewpoint
        
        Parameters:
        -----------
        temperature_ds : xarray.Dataset
            Dataset containing temperature variable
        dewpoint_ds : xarray.Dataset
            Dataset containing dewpoint temperature variable
            
        Returns:
        --------
        xarray.Dataset
            Dataset with relative humidity
        """
        # Find variable names
        temp_var = [v for v in temperature_ds.data_vars 
                    if '2m_temperature' in v or 'temperature' in v][0]
        dewp_var = [v for v in dewpoint_ds.data_vars 
                    if 'dewpoint' in v or 'dew_point' in v][0]
        
        # Align datasets
        temp_aligned, dewp_aligned = self._ensure_same_dimensions(
            temperature_ds, dewpoint_ds
        )
        
        # Convert to units for MetPy
        temp_data = temp_aligned[temp_var].values * units.kelvin
        dewp_data = dewp_aligned[dewp_var].values * units.kelvin
        
        # Calculate relative humidity using MetPy
        rh_data = relative_humidity_from_dewpoint(temp_data, dewp_data) * 100
        
        # Create DataArray with the same coordinates
        coords = {dim: temp_aligned[dim] for dim in temp_aligned[temp_var].dims}
        rh = xr.DataArray(
            rh_data.magnitude, 
            coords=coords,
            dims=temp_aligned[temp_var].dims,
            name='relative_humidity',
            attrs={
                'units': '%',
                'long_name': 'Relative Humidity at 2m'
            }
        )
        
        # Create dataset
        rh_ds = xr.Dataset({'relative_humidity': rh})
        
        return rh_ds
    
    def calculate_heat_stress_indices(self, temperature_ds, humidity_ds=None, dewpoint_ds=None):
        """
        Calculate heat stress indices (heat index and apparent temperature)
        
        Parameters:
        -----------
        temperature_ds : xarray.Dataset
            Dataset containing temperature variable
        humidity_ds : xarray.Dataset, optional
            Dataset containing relative humidity variable
        dewpoint_ds : xarray.Dataset, optional
            Dataset containing dewpoint temperature variable
            
        Returns:
        --------
        xarray.Dataset
            Dataset with heat stress indices
        """
        # Find temperature variable name
        temp_var = [v for v in temperature_ds.data_vars 
                    if '2m_temperature' in v or 'temperature' in v][0]
        
        # Convert temperature from Kelvin to Celsius
        temperature_C = temperature_ds[temp_var] - 273.15
        
        # Calculate relative humidity if not provided
        if humidity_ds is None and dewpoint_ds is not None:
            humidity_ds = self.calculate_relative_humidity(temperature_ds, dewpoint_ds)
        
        if humidity_ds is None:
            raise ValueError("Either humidity_ds or dewpoint_ds must be provided")
        
        # Find humidity variable name
        rh_var = [v for v in humidity_ds.data_vars 
                  if 'relative_humidity' in v or 'humidity' in v][0]
        
        # Align datasets
        temp_aligned, rh_aligned = self._ensure_same_dimensions(
            xr.Dataset({'temperature_C': temperature_C}),
            humidity_ds
        )
        
        # Convert to units for MetPy
        temp_data = temp_aligned['temperature_C'].values * units.degC
        rh_data = rh_aligned[rh_var].values * units.percent
        
        # Calculate heat index
        heat_idx = heat_index(temp_data, rh_data)
        
        # Calculate apparent temperature (feels like)
        apparent_temp = apparent_temperature(temp_data, rh_data)
        
        # Create DataArrays
        coords = {dim: temp_aligned[dim] for dim in temp_aligned['temperature_C'].dims}
        
        heat_idx_da = xr.DataArray(
            heat_idx.magnitude, 
            coords=coords,
            dims=temp_aligned['temperature_C'].dims,
            name='heat_index',
            attrs={
                'units': 'degC',
                'long_name': 'Heat Index'
            }
        )
        
        apparent_temp_da = xr.DataArray(
            apparent_temp.magnitude, 
            coords=coords,
            dims=temp_aligned['temperature_C'].dims,
            name='apparent_temperature',
            attrs={
                'units': 'degC',
                'long_name': 'Apparent Temperature (Feels Like)'
            }
        )
        
        # Create dataset
        heat_stress_ds = xr.Dataset({
            'heat_index': heat_idx_da,
            'apparent_temperature': apparent_temp_da
        })
        
        return heat_stress_ds
    
    def calculate_wind_power_density(self, u_wind_ds, v_wind_ds, height_adjustment=True):
        """
        Calculate wind power density (W/m²)
        
        Parameters:
        -----------
        u_wind_ds : xarray.Dataset
            Dataset containing u-component wind variable
        v_wind_ds : xarray.Dataset
            Dataset containing v-component wind variable
        height_adjustment : bool
            Whether to adjust wind speed from 10m to 80m (typical turbine height)
            
        Returns:
        --------
        xarray.Dataset
            Dataset with wind power density
        """
        # Find variable names
        u_var = [v for v in u_wind_ds.data_vars 
                 if 'u_component' in v or 'u_wind' in v or 'u10' in v.lower()][0]
        v_var = [v for v in v_wind_ds.data_vars 
                 if 'v_component' in v or 'v_wind' in v or 'v10' in v.lower()][0]
        
        # Align datasets
        u_aligned, v_aligned = self._ensure_same_dimensions(u_wind_ds, v_wind_ds)
        
        # Calculate wind speed
        wind_speed_10m = np.sqrt(u_aligned[u_var]**2 + v_aligned[v_var]**2)
        
        # Adjust wind speed to turbine height (80m) using power law
        # Wind speed increases with height: V2 = V1 * (h2/h1)^alpha
        # where alpha is ~0.143 for neutral stability conditions over flat terrain
        if height_adjustment:
            alpha = 0.143
            height_ratio = (80/10) ** alpha
            wind_speed_80m = wind_speed_10m * height_ratio
        else:
            wind_speed_80m = wind_speed_10m
        
        # Calculate wind power density: P = 0.5 * air_density * V^3
        # Standard air density at sea level: 1.225 kg/m³
        air_density = 1.225  # kg/m³
        wind_power_density = 0.5 * air_density * wind_speed_80m**3
        
        # Create dataset
        wind_power_ds = xr.Dataset({
            'wind_speed_10m': wind_speed_10m.assign_attrs(
                units='m/s',
                long_name='Wind Speed at 10m'
            ),
            'wind_speed_80m': wind_speed_80m.assign_attrs(
                units='m/s',
                long_name='Wind Speed at 80m (adjusted)'
            ),
            'wind_power_density': wind_power_density.assign_attrs(
                units='W/m²',
                long_name='Wind Power Density at 80m'
            )
        })
        
        return wind_power_ds
    
    def calculate_growing_degree_days(self, temperature_ds, base_temp=10.0, start_month=4, end_month=10):
        """
        Calculate growing degree days (GDD) for agricultural applications
        
        Parameters:
        -----------
        temperature_ds : xarray.Dataset
            Dataset containing temperature variable
        base_temp : float
            Base temperature in Celsius
        start_month : int
            Start month of growing season
        end_month : int
            End month of growing season
            
        Returns:
        --------
        xarray.Dataset
            Dataset with daily and accumulated GDD
        """
        # Find temperature variable name
        temp_var = [v for v in temperature_ds.data_vars 
                    if '2m_temperature' in v or 'temperature' in v][0]
        
        # Get maximum and minimum daily temperatures
        tmax_daily = temperature_ds[temp_var].resample(time='1D').max() - 273.15  # K to °C
        tmin_daily = temperature_ds[temp_var].resample(time='1D').min() - 273.15
        
        # Calculate daily mean temperature
        tmean_daily = (tmax_daily + tmin_daily) / 2
        
        # Calculate GDD with base temperature
        # GDD = max(0, mean_temp - base_temp)
        daily_gdd = tmean_daily - base_temp
        daily_gdd = daily_gdd.where(daily_gdd > 0, 0)
        
        # Filter to growing season
        if start_month <= end_month:
            season_mask = ((daily_gdd.time.dt.month >= start_month) & 
                           (daily_gdd.time.dt.month <= end_month))
        else:  # Handle Southern Hemisphere (e.g., start_month=9, end_month=4)
            season_mask = ((daily_gdd.time.dt.month >= start_month) | 
                           (daily_gdd.time.dt.month <= end_month))
        
        seasonal_gdd = daily_gdd.where(season_mask)
        
        # Calculate accumulated GDD
        # Group by year to reset accumulation each year
        years = seasonal_gdd.time.dt.year
        cumulative_gdd = seasonal_gdd.groupby(years).cumsum()
        
        # Create dataset
        gdd_ds = xr.Dataset({
            'daily_gdd': daily_gdd.assign_attrs(
                units='°C',
                long_name=f'Daily Growing Degree Days (base {base_temp}°C)'
            ),
            'seasonal_gdd': seasonal_gdd.assign_attrs(
                units='°C',
                long_name=f'Seasonal Daily Growing Degree Days (base {base_temp}°C)'
            ),
            'cumulative_gdd': cumulative_gdd.assign_attrs(
                units='°C',
                long_name=f'Cumulative Growing Degree Days (base {base_temp}°C)'
            )
        })
        
        return gdd_ds
    
    def plot_wind_power_potential_map(self, wind_power_ds, time_idx=0, 
                                      threshold=200, title=None, save_path=None):
        """
        Create a map of wind power potential with areas above threshold highlighted
        
        Parameters:
        -----------
        wind_power_ds : xarray.Dataset
            Dataset containing wind power density
        time_idx : int
            Time index to plot
        threshold : float
            Threshold for viable wind power (W/m²)
        title : str, optional
            Title for the plot
        save_path : str, optional
            Path to save the plot
        """
        # Select data at specified time
        if 'time' in wind_power_ds['wind_power_density'].dims:
            time_value = wind_power_ds.time.values[time_idx]
            data = wind_power_ds['wind_power_density'].isel(time=time_idx)
            time_str = np.datetime_as_string(time_value, unit='h')
        else:
            data = wind_power_ds['wind_power_density']
            time_str = ""
        
        # Create figure with projection
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Add map features
        ax.coastlines(resolution='50m')
        ax.add_feature(ccrs.feature.BORDERS, linestyle=':')
        ax.gridlines(draw_labels=True)
        
        # Plot data
        im = data.plot(
            ax=ax, 
            transform=ccrs.PlateCarree(),
            cmap='viridis',
            vmin=0,
            add_colorbar=True
        )
        
        # Create a mask for areas above threshold
        viable_areas = data >= threshold
        
        # Create a stippling effect for viable areas
        if np.any(viable_areas):
            lats = data.latitude
            lons = data.longitude
            
            # Create a meshgrid for stippling (less dense than the data)
            step = max(1, len(lats) // 50)  # Adjust density of stippling
            lats_sub = lats[::step]
            lons_sub = lons[::step]
            
            lons_mesh, lats_mesh = np.meshgrid(lons_sub, lats_sub)
            
            # Sample the viable areas mask at the stippling points
            viable_stipple = viable_areas.interp(
                latitude=lats_mesh.flatten(), 
                longitude=lons_mesh.flatten()
            ).values.reshape(lons_mesh.shape)
            
            # Add stippling for viable areas
            ax.scatter(
                lons_mesh[viable_stipple], 
                lats_mesh[viable_stipple],
                color='red', 
                s=10, 
                alpha=0.8, 
                transform=ccrs.PlateCarree(),
                label=f'Viable Areas (>{threshold} W/m²)'
            )
            
            plt.legend(loc='lower left')
        
        # Set title
        if title:
            plt.title(title)
        else:
            plt.title(f"Wind Power Density at 80m - {time_str}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

# Example usage
if __name__ == "__main__":
    # This assumes the ERA5Analyzer class from the previous example is available
    #from era5_analyzer import ERA5Analyzer
    
    # Replace with your actual data directory
    era5_data_dir = "./data/raw/grided/era5land_vars"
    
    # Create analyzer instances
    analyzer = ERA5Analyzer(era5_data_dir)
    derived_vars = ERA5DerivedVariables(analyzer)
    
    # Example 1: Calculate relative humidity
    temp_data = analyzer.load_variable('2m_temperature', year=2017, month=1)
    dewp_data = analyzer.load_variable('2m_dewpoint_temperature', year=2017, month=1)
    
    if temp_data is not None and dewp_data is not None:
        rh_ds = derived_vars.calculate_relative_humidity(temp_data, dewp_data)
        
        # Example map of relative humidity
        analyzer.plot_map(
            rh_ds, 'relative_humidity',
            time_idx=0,
            title="ERA5-Land Relative Humidity at 2m (July 2017)",
            cmap='Blues',
            save_path="relative_humidity_map.png"
        )
    
    # Example 2: Calculate wind power density
    u_wind = analyzer.load_variable('10m_u_component_of_wind', year=2017, month=7)
    v_wind = analyzer.load_variable('10m_v_component_of_wind', year=2017, month=7)
    
    if u_wind is not None and v_wind is not None:
        wind_power_ds = derived_vars.calculate_wind_power_density(u_wind, v_wind)
        
        # Plot wind power potential
        derived_vars.plot_wind_power_potential_map(
            wind_power_ds,
            time_idx=0,
            threshold=200,  # 200 W/m² is often considered viable for wind power
            title="Wind Power Potential (July 2017)",
            save_path="wind_power_potential_map.png"
        )