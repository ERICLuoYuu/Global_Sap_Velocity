from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
import plotly.graph_objects as go
import plotly.io as pio
class GermanSapFlowAnalyzer:
    """
    Simplified analyzer for German (DEU) SapFlow data with flag handling
    """
    def __init__(self, base_dir: str = "./data/raw/0.1.5/0.1.5/csv/sapwood"):
        self.data_dir = Path(base_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Directory not found: {self.data_dir}")
        
        self.sapflow_data = {}
        self._load_sapflow_data()
    
    def _load_sapflow_data(self):
        """Load all German sapflow data files and their corresponding flags"""
        sapf_files = list(self.data_dir.glob("DEU_*_sapf_data.csv"))
        print(f"Found {len(sapf_files)} German sapflow files")
        
        for file in sapf_files:
            parts = file.stem.split('_')
            location = parts[1]
            plant_type = parts[2]
            
            try:
                # Load sap flow data
                df = pd.read_csv(file, parse_dates=['TIMESTAMP'])
                
                # Load corresponding flags file
                flag_file = file.parent / f"{file.stem.replace('_data', '_flags')}.csv"
                if flag_file.exists():
                    print(flag_file)
                    flags = pd.read_csv(flag_file, parse_dates=['TIMESTAMP'])
                else:
                    print(f"Warning: No flags file found for {file.name}")
                    flags = None
                
                # Process data with flags
                df = self._process_sapflow_data(location, plant_type, df, flags)
                
                if location not in self.sapflow_data:
                    self.sapflow_data[location] = {}
                self.sapflow_data[location][plant_type] = df
            
            except Exception as e:
                print(f"Error loading {file.name}: {str(e)}")
        
    

    def _process_sapflow_data(self, location: str, plant_type: str, df: pd.DataFrame, flags: pd.DataFrame = None) -> pd.DataFrame:
        """
        Process sapflow data with debugging information
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
            output_dir = Path('./data/processed/sap/filtered')
            output_dir.mkdir(parents=True, exist_ok=True)
            filter_path = output_dir / f"{df.index[0].strftime('%Y%m%d')}_{df.index[-1].strftime('%Y%m%d')}_filtered.csv"
            df.to_csv(filter_path)
            print(f"Saved filtered data to {filter_path}")


        try:
            # Create processed directory for outliers
            outlier_dir = Path('./data/processed/sap/outliers')
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
            # Create processed directory
            output_dir = Path('./data/processed/sap/daily')
            output_dir.mkdir(parents=True, exist_ok=True)

            # Get columns to resample (exclude solar_TIMESTAMP)
            data_cols = [col for col in df.columns if col != 'solar_TIMESTAMP']
            
            # Simple daily mean resampling
            daily_df = df[data_cols].resample('D').mean()
            
            # Add '_mean' suffix to column names
            daily_df.columns = [f"{col}_mean" for col in daily_df.columns]
            
            # Save daily data
            output_path = output_dir / f"{location}_{plant_type}_daily.csv"
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

    def plot_individual_plants(self, location: str, plant_type: str, 
                            figsize=(12, 6), save_dir: str = None):
        """Create individual plots for each plant/tree with outlier detection"""
        if location not in self.sapflow_data or plant_type not in self.sapflow_data[location]:
            raise ValueError(f"No data found for {location}_{plant_type}")
        
        data = self.sapflow_data[location][plant_type].copy()
        plot_columns = [col for col in data.columns if col != 'solar_TIMESTAMP']
        
        print(f"\nPlotting {location}_{plant_type}")
        print(f"Data shape: {data.shape}")
        print(f"Columns to plot: {plot_columns}")
        
        if save_dir:
            save_path = Path(save_dir) / f"DEU_{location}_{plant_type}"
            save_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving plots to {save_path}")
        
        for column in plot_columns:
            # Print column statistics
            print(f"\nColumn: {column}")
            print(f"Non-null values: {data[column].count()}")
            print(f"Value range: {data[column].min()} to {data[column].max()}")
            
            # Skip if all values are NaN
            if data[column].isna().all():
                print(f"Skipping {column} - all values are NA")
                continue
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Get valid data
            valid_data = data[column].dropna()
            
            if len(valid_data) > 0:
                
                ax.plot(valid_data.index, valid_data.values,
                    '-b.', alpha=0.5, label='Normal data', linewidth=1,
                    markersize=3)
                
              
                # Customize plot
                ax.set_title(f'Sap Flow Time Series - {location} {plant_type}\nTree {column}', pad=20)
                ax.set_xlabel('Date')
                ax.set_ylabel('Sap Flow')
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
                    filename = f"tree_{column}.png"
                    fig.savefig(save_path / filename, bbox_inches='tight')
                    plt.close(fig)
                    print(f"Saved {filename}")
                else:
                    plt.show()
            else:
                plt.close(fig)
                print(f"No valid data to plot for {column}")


    
    def plot_all_plants(self, figsize=(12, 6), save_dir: str = None):
        """Plot individual trees for all sites"""
        for location in self.sapflow_data:
            for plant_type in self.sapflow_data[location]:
                try:
                    print(f"\nProcessing DEU_{location}_{plant_type}")
                    self.plot_individual_plants(
                        location=location,
                        plant_type=plant_type,
                        figsize=figsize,
                        save_dir=save_dir
                    )
                except Exception as e:
                    print(f"Error plotting {location}_{plant_type}: {str(e)}")
    
    def get_summary(self, location: str, plant_type: str) -> Dict:
        """Get summary statistics for a specific site"""
        if location not in self.sapflow_data or plant_type not in self.sapflow_data[location]:
            raise ValueError(f"No data found for {location}_{plant_type}")
        
        data = self.sapflow_data[location][plant_type]
        plot_columns = [col for col in data.columns if col != 'solar_TIMESTAMP']
        
        summary = {
            'location': location,
            'plant_type': plant_type,
            'time_range': {
                'start': data.index.min().strftime('%Y-%m-%d'),
                'end': data.index.max().strftime('%Y-%m-%d'),
                'duration_days': (data.index.max() - data.index.min()).days
            },
            'trees': len(plot_columns),
            'measurements': len(data),
            'missing_data': (data[plot_columns].isna().sum() / len(data) * 100).mean()
        }
        
        return summary