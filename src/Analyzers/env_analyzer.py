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
                df = self._process_environmental_data(df, flags, location, plant_type)
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
        numeric_cols = [col for col in df.columns if col != 'TIMESTAMP' and col != 'solar_TIMESTAMP']
        for col in numeric_cols:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                print(f"\nColumn {col} range: {df[col].min()} to {df[col].max()}")
            except Exception as e:
                print(f"Error converting column {col}: {str(e)}")

        # Set index
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
            
            data_cols = [col for col in df.columns if col != 'solar_TIMESTAMP']
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
            data_cols = [col for col in df.columns if col != 'solar_TIMESTAMP']
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
            # Create processed directory for standardized data
            std_dir = Path('./data/processed/env/standardized')
            std_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each column for standardization
            data_cols = [col for col in df.columns if col != 'solar_TIMESTAMP']
            # only use common columns
            for col in data_cols:
                common_cols = self.get_common_columns_multiple([df])
                if col not in common_cols:
                    print(f"Skipping {col} - not in common columns")
                    continue
                if df[col].isna().all():
                    print(f"Skipping {col} - all values are NA")
                    continue
                
                # Standardize data
                standardized_data = self._standardize_env_data(df[col], col)
                if standardized_data is not None:
                    df[col] = standardized_data
                    print(f"Standardized {col} data")
                else:
                    print(f"Skipping {col} due to zero range")
            df.to_csv(std_dir / f"{location}_{plant_type}_standardized.csv")
        except Exception as e:
            print(f"Error processing standardization: {str(e)}")

        try:
            # Create processed directory
            output_dir = Path('./data/processed/env/daily')
            output_dir.mkdir(parents=True, exist_ok=True)

            # Get columns to resample (exclude solar_TIMESTAMP)
            data_cols = [col for col in df.columns if col != 'solar_TIMESTAMP']
            
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
        Standardize environmental data for comparison using min-max scaling.
        Caches global max/min values for each variable to avoid recalculation.
        
        Parameters:
        -----------
        series : pd.Series
            Input series to standardize
        variable : str
            Name of the variable to standardize
            
        Returns:
        --------
        pd.Series
            Standardized values for the specified variable
        """
        try:
            print(f"\nStandardizing variable: {variable}")
            
            # Create cache for min/max values if it doesn't exist
            if not hasattr(self, '_minmax_cache'):
                self._minmax_cache = {}
                
            # Calculate global max/min only if not already cached
            if variable not in self._minmax_cache:
                temp = []
                for location in self.env_data:
                    for plant_type in self.env_data[location]:
                        data = self.env_data[location][plant_type].copy()
                        if variable in data.columns:
                            data = data[variable].to_list()
                            temp.extend(data)
                        else:
                            continue
                
                if not temp:
                    raise ValueError(f"No data found for variable {variable}")
                    
                self._minmax_cache[variable] = {
                    'max': np.max(temp),
                    'min': np.min(temp)
                }
                print(f"Calculated and cached global range for {variable}")
            
            max_val = self._minmax_cache[variable]['max']
            min_val = self._minmax_cache[variable]['min']
            print(f"Global range for {variable}: {min_val} to {max_val}")
            
            # Check for zero range
            if max_val == min_val:
                print(f"Warning: {variable} has constant value {max_val}")
                return None
            
            # Standardize data
            standardized_data = (series - min_val) / (max_val - min_val)
            print(f"Standardized range: {standardized_data.min()} to {standardized_data.max()}")
            
            return standardized_data
            
        except Exception as e:
            print(f"Error during standardization: {str(e)}")
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
            
            