from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
import plotly.graph_objects as go
import plotly.io as pio

class GermanEnvironmentalAnalyzer:
    """
    Analyzer for German (DEU) Environmental data with metadata handling
    """
    def __init__(self, base_dir: str = "./data/raw/0.1.5/0.1.5/csv/sapwood"):
        self.data_dir = Path(base_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Directory not found: {self.data_dir}")
        
        self.env_data = {}
        self._load_environmental_data()
        self._standardize_all_data()
         
    
    def _load_environmental_data(self):
        """Load all German environmental data files and their corresponding metadata"""
        env_files = list(self.data_dir.glob("DEU_*_env_data.csv"))
        print(f"Found {len(env_files)} German environmental files")
        
        for file in env_files:
            parts = file.stem.split('_')
            location = parts[1]
            plant_type = parts[2]
            try:
                # Load environmental data
                df = pd.read_csv(file, parse_dates=['TIMESTAMP'])
                
                # Load corresponding metadata file
                flag_file = file.parent / f"{file.stem.replace('_data', '_flags')}.csv"
                if flag_file.exists():
                    print(f"Loading flags: {flag_file}")
                    flags = pd.read_csv(flag_file, parse_dates=['TIMESTAMP'])
                else:
                    print(f"Warning: No flag file found for {file.name}")
                    flags = None
                
                # Process data
                print(f"\nProcessing {file.name}")
                # get cooridnates information from "DEU_*_site_md.csv"
                site_md_file = file.parent / f"{file.stem.replace('_env_data', '_site_md')}.csv"
                # load the site_md_file
                site_md = pd.read_csv(site_md_file)
                # get the latitude and longitude from columns "si_lat" and "si_long"
                latitude = site_md['si_lat'].values[0]
                longitude = site_md['si_long'].values[0]
                df = self._process_environmental_data(df, location, plant_type, flags)
                # add coordinates to the metadata
                df['lat'] = latitude
                df['long'] = longitude
                if location not in self.env_data:
                    self.env_data[location] = {}
                self.env_data[location][plant_type] = df
            
            except Exception as e:
                print(f"Error loading {file.name}: {str(e)}")
    
    def _process_environmental_data(self, df: pd.DataFrame, location: str = None, plant_type: str = None, flags: pd.DataFrame = None) -> pd.DataFrame:
        """
        Process environment data with debugging information
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with environmental data
        flags : pd.DataFrame
            Input DataFrame with flag data

        Returns: pd.DataFrame
            Processed DataFrame with daily resampled data
        --------
        """
        print("\nProcessing data shape:", df.shape)
        print("Data types before processing:")
        print(df.dtypes)
        print("\nSample of raw data:")
        print(df.head())
        
        # Convert string "NA" to numpy NaN
        df = df.replace("NA", np.nan)
        
        # Convert numeric columns to float, with error handling
        numeric_cols = [col for col in df.columns if col != 'TIMESTAMP' and col != 'solar_TIMESTAMP' and col != 'lat' and col != 'long']
        for col in numeric_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"\nColumn {col} range: {df[col].min()} to {df[col].max()}")
            except Exception as e:
                print(f"Error converting column {col}: {str(e)}")

        # Set index to TIMESTAMP and sort
        df = df.set_index('TIMESTAMP')
        df = df.sort_index()
        
        # Remove duplicates
        if df.index.duplicated().any():
            print(f"Removing {df.index.duplicated().sum()} duplicate timestamps")
            df = df[~df.index.duplicated(keep='first')]
        
        # Apply flag-based filtering
        if flags is not None:
            flags = flags.set_index('TIMESTAMP').sort_index()
            if flags.index.duplicated().any():
                flags = flags[~flags.index.duplicated(keep='first')]
            
            data_cols = [col for col in df.columns if col != 'solar_TIMESTAMP' and col != 'lat' and col != 'long']
            for col in data_cols:
                flag_col = [fcol for fcol in flags.columns if col in fcol]
                if flag_col:
                    warn_mask = flags[flag_col[0]].str.contains('OUT_WARN|RANGE_WARN', na=False)
                    if warn_mask.any():
                        print(f"\nFiltering {warn_mask.sum()} values in {col} due to warnings")
                    df.loc[warn_mask, col] = np.nan
            
            # Export filtered data immediately
            output_dir = Path('./data/processed/env/filtered')
            output_dir.mkdir(parents=True, exist_ok=True)
            filter_path = output_dir / f"{location}_{plant_type}_filtered.csv"
            df.to_csv(filter_path)
            print(f"Saved filtered data to {filter_path}")


        try:
            # Create processed directory for outliers
            outlier_dir = Path('./data/processed/env/outliers')
            outlier_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each column for outliers
            data_cols = [col for col in df.columns if col != 'solar_TIMESTAMP' and col != 'lat' and col != 'long']
            for col in data_cols:
                if df[col].isna().all():
                    print(f"Skipping {col} - all values are NA")
                    continue
                    
                # Detect outliers using method B (rolling window)
                outliers = self._detect_outliers_monthly_sd(
                    series=df[col],
                    n_std=3,
                    time_window=1440,  # 24 hours
                    method='B'
                )
                
                # Save outlier information
                outlier_df = pd.DataFrame({
                    'timestamp': df.index,
                    'value': df[col],
                    'is_outlier': outliers
                })
                
                outlier_path = outlier_dir / f"{col}_outliers.csv"
                outlier_df.to_csv(outlier_path)
                print(f"Saved outliers for {col} to {outlier_path}")
                
                # Set outliers to NaN in original data
                df.loc[outliers, col] = np.nan
        except Exception as e:
            print(f"Error processing outliers: {str(e)}")

        try:
            # Create processed directory
            output_dir = Path('./data/processed/env/daily')
            output_dir.mkdir(parents=True, exist_ok=True)

            # Get columns to resample (exclude solar_TIMESTAMP)
            data_cols = [col for col in df.columns if col != 'solar_TIMESTAMP' and col != 'lat' and col != 'long']
            
            # Get columns that need sum vs mean
            precip_rad_cols = [col for col in data_cols if 'precip' in col.lower() or 'rad' in col.lower()]
            mean_cols = [col for col in data_cols if col not in precip_rad_cols]

            # Resample separately
            daily_sums = df[precip_rad_cols].resample('D').sum() if precip_rad_cols else None
            daily_means = df[mean_cols].resample('D').mean() if mean_cols else None

            # Combine results
            if daily_sums is not None and daily_means is not None:
                daily_df = pd.concat([daily_sums, daily_means], axis=1)
            elif daily_sums is not None:
                daily_df = daily_sums
            else:
                daily_df = daily_means

            # Add appropriate suffixes
            daily_df.columns = [f"{col}_sum" if col in precip_rad_cols else f"{col}_mean" 
                               for col in daily_df.columns]
            
            # Save daily data
            start_date = df.index[0].strftime('%Y%m%d')
            end_date = df.index[-1].strftime('%Y%m%d')
            output_path = output_dir / f"daily_{start_date}_{end_date}.csv"
            daily_df.to_csv(output_path)
            print(f"Saved daily resampled data to {output_path}")

            return daily_df

        except Exception as e:
            print(f"Error during daily resampling: {str(e)}")
            return df, None

    def _detect_outliers_monthly_sd(self, series: pd.Series, n_std: float = 3, time_window:int = 1440 ,method: str = 'A') -> pd.Series:
        """
        Detect outliers using standard deviation within monthly windows
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
        n_std : float
            Number of standard deviations for outlier threshold
                
        Returns:
        --------
        pd.Series : Boolean mask where True indicates outliers
        """
        # Initialize outlier mask with same index as input series
        outliers = pd.Series(False, index=series.index)
        if method == 'A':
            # Group by year and month
            grouped = series.groupby([series.index.year, series.index.month])
            
            
            # Process each month separately
            for (year, month), group in grouped:
                if len(group) > 0:  # Only process if there's data
                    # Calculate mean and std for the month
                    mean = group.mean()
                    std = group.std()
                    
                    # Find outliers for this month
                    month_mask = (series.index.year == year) & (series.index.month == month)
                    monthly_data = series[month_mask]
                    
                    # Detect outliers
                    monthly_outliers = (np.abs(monthly_data - mean) > n_std * std)
                    outliers[monthly_data.index] = monthly_outliers
                    
                    # Print monthly summary
                    n_outliers = monthly_outliers.sum()
                    print(f"Month {month}/{year}: "
                        f"{n_outliers} outliers out of {len(monthly_data)} points "
                        f"({(n_outliers/len(monthly_data)*100):.1f}%)")
                    
        elif method == 'B':
            # Calculate rolling statistics
            rolling_mean = series.rolling(
                window=time_window,
                center=True,
                min_periods=int(time_window/2)
            ).mean()
            
            rolling_std = series.rolling(
                window=time_window,
                center=True,
                min_periods=int(time_window/2)
            ).std()
            
            # Handle edge cases with forward/backward fill
            rolling_mean = rolling_mean.fillna(method='ffill').fillna(method='bfill')
            rolling_std = rolling_std.fillna(method='ffill').fillna(method='bfill')
            
            # Detect outliers
            outliers = np.abs(series - rolling_mean) > (n_std * rolling_std)
            
            print(f"Outliers detected: {outliers.sum()} out of {len(series)} points")
        return outliers

    def _standardize_env_data(self, series: pd.Series, variable: str) -> pd.Series:
        """
        Standardizes a data series using pre-calculated min-max scaling values.
        
        Parameters:
        -----------
        series : pd.Series
            Input series to standardize
        variable : str
            Name of the variable being standardized
                
        Returns:
        --------
        pd.Series
            Standardized values for the specified variable
        """
        try:
            if variable not in self._minmax_cache:
                raise ValueError(f"No pre-calculated range found for {variable}")
                
            max_val = self._minmax_cache[variable]['max']
            min_val = self._minmax_cache[variable]['min']
            
            # Check for zero range
            if max_val == min_val:
                print(f"Warning: {variable} has constant value {max_val}")
                return None
            
            # Standardize data
            standardized_data = (series - min_val) / (max_val - min_val)
            print(f"Standardized range for {variable}: {standardized_data.min():.3f} to {standardized_data.max():.3f}")
            
            return standardized_data
            
        except Exception as e:
            print(f"Error standardizing {variable}: {str(e)}")
            return None

    def plot_environmental_variables(self, location: str, plant_type: str, figsize=(12, 6), 
                                  save_dir: str = None):
        """Create individual plots for each environmental variable"""
        if location not in self.env_data or plant_type not in self.env_data[location]:
            raise ValueError(f"No data found for {location}_{plant_type}")
        
        data = self.env_data[location][plant_type].copy()
        plot_columns = [col for col in data.columns if col.endswith(('_mean', '_sum'))]
        
        print(f"\nPlotting environmental data for {location}_{plant_type}")
        print(f"Data shape: {data.shape}")
        print(f"Variables to plot: {plot_columns}")
        
        if save_dir:
            save_path = Path(save_dir) / f"DEU_{location}_{plant_type}_env"
            save_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving plots to {save_path}")
        
        for variable in plot_columns:
            
            # Print column statistics
            print(f"\nColumn: {variable}")
            print(f"Non-null values: {data[variable].count()}")
            print(f"Value range: {data[variable].min()} to {data[variable].max()}")
            fig, ax = plt.subplots(figsize=figsize)
            
            # Get valid data
            valid_data = data[variable].dropna()
            
            if len(valid_data) > 0:
                ax.plot(valid_data.index, valid_data.values,
                       '-b.', alpha=0.5, label='Measurements', linewidth=1,
                       markersize=3)
                
                # Customize plot
                ax.set_title(f'Environmental Variable: {variable}\nLocation: {location}{plant_type}', 
                           pad=20)
                ax.set_xlabel('Date')
                ax.set_ylabel(variable)
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                ax.legend()
                
                # Add data quality info
                valid_percent = (len(valid_data) / len(data)) * 100
                ax.text(0.02, 0.98, 
                       f'Valid data: {valid_percent:.1f}%\n'
                       f'Range: {valid_data.min():.3f} to {valid_data.max():.3f}', 
                       transform=ax.transAxes, va='top')
                
                plt.tight_layout()
                
                if save_dir:
                    filename = f"{variable}.png"
                    fig.savefig(save_path / filename, bbox_inches='tight')
                    plt.close(fig)
                    print(f"Saved {filename}")
                else:
                    plt.show()
            else:
                plt.close(fig)
                print(f"No valid data to plot for {variable}")

    def plot_all(self, figsize=(12, 6), save_dir: str = None):
        """Plot variables for all sites"""
        for location in self.env_data:
            for plant_type in self.env_data[location]:
                try:
                    print(f"\nProcessing DEU_{location}_{plant_type}")
                    self.plot_environmental_variables(
                        location=location,
                        plant_type=plant_type,
                        figsize=figsize,
                        save_dir=save_dir
                    )
                except Exception as e:
                    print(f"Error plotting {location}_{plant_type}: {str(e)}")

    def get_summary(self, location: str, plant_type: str) -> Dict:
        """Get summary statistics for a specific location"""
        if location not in self.env_data or plant_type not in self.env_data[location]:
            raise ValueError(f"No data found for {location}")
        
        data = self.env_data[location][plant_type]
        variables = [col for col in data.columns if col.endswith(('_mean', '_sum'))]
        
        summary = {
            'location': location,
            'plant_type': plant_type,
            'time_range': {
                'start': data.index.min().strftime('%Y-%m-%d'),
                'end': data.index.max().strftime('%Y-%m-%d'),
                'duration_days': (data.index.max() - data.index.min()).days
            },
            'variables': len(variables),
            'measurements': len(data),
            'variables_list': variables,
            'missing_data': {var: (data[var].isna().sum() / len(data) * 100) 
                           for var in variables}
        }
        
        return summary
    def _standardize_all_data(self):
        """Standardize environmental data using only common columns across all datasets with error handling"""
        try:
            print("\nStarting data standardization process...")
            
            # Verify data is loaded
            if not self.env_data:
                raise ValueError("No environmental data has been loaded")
            
            # Create processed directory for standardized data
            try:
                std_dir = Path('./data/processed/env/standardized')
                std_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise IOError(f"Failed to create standardization directory: {str(e)}")
            
            # Get all datasets with validation
            try:
                all_dataframes = []
                for location in self.env_data:
                    for plant_type in self.env_data[location]:
                        df = self.env_data[location][plant_type]
                        if not isinstance(df, pd.DataFrame):
                            print(f"Warning: Invalid data type for {location}_{plant_type}")
                            continue
                        if df.empty:
                            print(f"Warning: Empty dataset for {location}_{plant_type}")
                            continue
                        all_dataframes.append(df)
                
                if not all_dataframes:
                    raise ValueError("No valid DataFrames found for standardization")
                    
            except Exception as e:
                raise ValueError(f"Error collecting datasets: {str(e)}")
            
            # Find common columns
            try:
                common_columns = self.get_common_columns_multiple(all_dataframes)
                common_columns = [col for col in common_columns if col != 'solar_TIMESTAMP' and col != 'lat' and col != 'long']
                
                if not common_columns:
                    raise ValueError("No common columns found across datasets")
                    
                print(f"\nFound {len(common_columns)} common columns to standardize: {common_columns}")
                
            except Exception as e:
                raise ValueError(f"Error identifying common columns: {str(e)}")
            
            # Calculate global min/max for common columns
            self._minmax_cache = {}
            for variable in common_columns:
                try:
                    temp_data = []
                    for location in self.env_data:
                        for plant_type in self.env_data[location]:
                            data = self.env_data[location][plant_type]
                            valid_data = data[variable].dropna().tolist()
                            if valid_data:  # Only add if there's valid data
                                temp_data.extend(valid_data)
                    
                    if not temp_data:
                        print(f"Warning: No valid data found for {variable}")
                        continue
                        
                    self._minmax_cache[variable] = {
                        'max': np.max(temp_data),
                        'min': np.min(temp_data)
                    }
                    print(f"\nGlobal range for {variable}: "
                          f"{self._minmax_cache[variable]['min']} to "
                          f"{self._minmax_cache[variable]['max']}")
                          
                except Exception as e:
                    print(f"Error calculating range for {variable}: {str(e)}")
                    continue
            
            # Standardize common columns for each dataset
            standardization_results = []
            for location in self.env_data:
                for plant_type in self.env_data[location]:
                    try:
                        print(f"\nStandardizing data for {location}_{plant_type}")
                        data = self.env_data[location][plant_type].copy()
                        
                        # Track successful standardizations
                        standardized_cols = []
                        
                        for variable in common_columns:
                            try:
                                if variable not in self._minmax_cache:
                                    print(f"Skipping {variable} - no global range available")
                                    continue
                                    
                                standardized = self._standardize_env_data(
                                    data[variable], variable)
                                if standardized is not None:
                                    data[variable] = standardized
                                    standardized_cols.append(variable)
                                    print(f"Standardized {variable}")
                                    
                            except Exception as e:
                                print(f"Error standardizing {variable}: {str(e)}")
                                continue
                        
                        # Only save if some columns were standardized
                        print(df.head())
                        if standardized_cols:
                            output_path = std_dir / f"{location}_{plant_type}_standardized.csv"
                            # for data[standardized_cols] keep 3 decimal points
                            data[standardized_cols] = data[standardized_cols].apply(lambda x: round(x, 3))
                            # Save standardized data with lat and long
                            saved_cols = standardized_cols + ['lat', 'long']     
                            data = data[saved_cols]
                            
                            data.to_csv(output_path)
                            print(f"Saved standardized data to {output_path}")
                            
                            # Update stored data
                            self.env_data[location][plant_type] = data
                            
                            standardization_results.append({
                                'location': location,
                                'plant_type': plant_type,
                                'standardized_columns': standardized_cols,
                                'success': True
                            })
                        else:
                            print(f"No columns were standardized for {location}_{plant_type}")
                            standardization_results.append({
                                'location': location,
                                'plant_type': plant_type,
                                'standardized_columns': [],
                                'success': False
                            })
                            
                    except Exception as e:
                        print(f"Error processing {location}_{plant_type}: {str(e)}")
                        standardization_results.append({
                            'location': location,
                            'plant_type': plant_type,
                            'error': str(e),
                            'success': False
                        })
                        continue
            
            # Summary report
            print("\nStandardization Summary:")
            successful = sum(1 for result in standardization_results if result['success'])
            print(f"Successfully standardized {successful} out of {len(standardization_results)} datasets")
            
            return standardization_results
            
        except Exception as e:
            print(f"Critical error during standardization process: {str(e)}")
            return None

    def get_common_columns_multiple(self, dataframes: list) -> list:
        """Get list of common columns across multiple DataFrames"""
        try:
            # Check if list is empty
            if not dataframes:
                print("No DataFrames provided")
                return []
                
            # Convert first DataFrame columns to set
            common_cols = set(dataframes[0].columns)
            
            # Intersect with remaining DataFrames
            for df in dataframes[1:]:
                common_cols = common_cols.intersection(set(df.columns))
                
            # Convert back to list
            common_cols = list(common_cols)
            print(f"Found {len(common_cols)} common columns across {len(dataframes)} DataFrames:")
            print(f"Common columns: {common_cols}")
            
            return common_cols
            
        except Exception as e:
            print(f"Error finding common columns: {str(e)}")
            return []
    # get all columns
    def get_all_columns(self, dataframe: list) -> List:
        """Get list of all columns in a DataFrame"""
        try:
            # Check if list is empty
            if not dataframe:
                print("No DataFrame provided")
                return []
            colums = []
            for df in dataframe:
                colums.extend(df.columns)
            # remove duplicates
            colums = list(set(colums))
            print(f"Found {len(colums)} columns in the DataFrame:")
            return colums
        except Exception as e:
            print(f"Error finding columns: {str(e)}")
            return []
            
            