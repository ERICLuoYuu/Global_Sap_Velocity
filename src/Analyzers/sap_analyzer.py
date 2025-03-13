from pathlib import Path
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
import plotly.graph_objects as go
import plotly.io as pio
import copy
import concurrent.futures
parent_dir = str(Path(__file__).parent.parent.parent)
print(parent_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from src.tools import adjust_time_to_utc, create_timezone_mapping
class SapFlowAnalyzer:
    """
    Analyzer for Global SapFlow data with flag handling
    """
    def __init__(self, base_dir: str = "./data/raw/0.1.5/0.1.5/csv/sapwood"):
        self.data_dir = Path(base_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Directory not found: {self.data_dir}")
        self.sapf_raw_data = {}
        self.sapflow_data = {}
        self.outlier_removed_data = {}
        self._load_sapflow_data()
    
    def _load_sapflow_data(self):
        """Load all German sapflow data files and their corresponding flags"""
        sapf_files = list(self.data_dir.glob("*_sapf_data.csv"))
        print(f"Found {len(sapf_files)} sapflow files")
        column_mapping = {
                'TIMESTAMP_solar': 'solar_TIMESTAMP',  # Map alternate name to standard name
                }
        for file in sapf_files:
            parts = file.stem.split('_')
            location = '_'.join(parts[:2])
            
            plant_type = '_'.join(parts[2:])
            
            try:
                # Load sap flow data
                df = pd.read_csv(file, parse_dates=['TIMESTAMP'])
                df = df.rename(columns=column_mapping)
                if location not in self.sapf_raw_data:
                    self.sapf_raw_data[location] = {}
                self.sapf_raw_data[location][plant_type] = df
                # load corresponding evn_md data
                time_zone_file = file.parent / f"{file.stem.replace('sapf_data', 'env_md')}.csv"
                tz_df = pd.read_csv(time_zone_file)
                time_zone = tz_df['env_time_zone'].iloc[0]
                time_zone_map = create_timezone_mapping()
                # Apply the time zone adjustment to the entire column at once
                df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
                df['solar_TIMESTAMP'] = pd.to_datetime(df['solar_TIMESTAMP'])
                df['TIMESTAMP'] = df['TIMESTAMP'].apply(lambda x: adjust_time_to_utc(x, time_zone, time_zone_map))
                df['solar_TIMESTAMP'] = df['solar_TIMESTAMP'].apply(lambda x: adjust_time_to_utc(x, time_zone, time_zone_map))
                output_dir = Path('./outputs/processed_data/sap/timezone_adjusted')
                if not output_dir.exists():
                    output_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_dir / f'{file.stem}_adjusted.csv')
                # Load corresponding flags file
                flag_file = file.parent / f"{file.stem.replace('_data', '_flags')}.csv"
                if flag_file.exists():
                    print(flag_file)
                    flags = pd.read_csv(flag_file, parse_dates=['TIMESTAMP'])
                    flags = flags.rename(columns=column_mapping)
                    flags['TIMESTAMP'] = pd.to_datetime(flags['TIMESTAMP'])
                    flags['solar_TIMESTAMP'] = pd.to_datetime(flags['solar_TIMESTAMP'])
                    flags['TIMESTAMP'] = flags['TIMESTAMP'].apply(lambda x: adjust_time_to_utc(x, time_zone, time_zone_map))
                    flags['solar_TIMESTAMP'] = flags['solar_TIMESTAMP'].apply(lambda x: adjust_time_to_utc(x, time_zone, time_zone_map))
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

        # Set index with error checking
        try:
            if 'TIMESTAMP' not in df.columns:
                print("Error: TIMESTAMP column not found")
                return None, None
                
            # Check if TIMESTAMP is already datetime (should be from parse_dates)
            is_datetime = pd.api.types.is_datetime64_any_dtype(df['TIMESTAMP'])
            print(f"TIMESTAMP column is already datetime: {is_datetime}")
            
            # Set index 
            df = df.set_index('TIMESTAMP')
            df = df.sort_index()
            
            # Check for all-NaN index
            if df.index.isna().all():
                print("Error: All timestamp values are NaN")
                return None, None
            
            # REMOVE THIS LINE - it's causing the issue
            # df.index = pd.to_datetime(df.index)  # This second conversion is unnecessary
                
        except Exception as e:
            print(f"Error setting timestamp index: {str(e)}")
            return None, None

        # Check for duplicates properly
        dup_mask = df.index.duplicated()
        dup_count = dup_mask.sum()

        if dup_count > 0:
            print(f"Found {dup_count} duplicate timestamps")
            
            # Export duplicates for analysis
            dup_dir = Path(f"./outputs/processed_data/sap/duplicates")
            dup_dir.mkdir(parents=True, exist_ok=True)
            
            # Export duplicate rows
            df[dup_mask].to_csv(dup_dir / f"{location}_{plant_type}_duplicates.csv")
            
            print(f"Removing {dup_count} duplicate timestamps ({dup_count/len(df)*100:.1f}%)")
            df = df[~dup_mask]  # Remove duplicates
        else:
            print("No duplicates found in timestamps")
        
        # Apply flag-based filtering
        if flags is not None:
            flags = flags.set_index('TIMESTAMP').sort_index()
            if flags.index.duplicated().any():
                flags = flags[~flags.index.duplicated(keep='first')]
            
            data_cols = [col for col in df.columns if col != 'solar_TIMESTAMP']
            # Export filtered data immediately
            output_dir = Path('./outputs/processed_data/sap/filtered')
            output_dir.mkdir(parents=True, exist_ok=True)
            for col in data_cols:
                flag_col = [fcol for fcol in flags.columns if col in fcol]
                if flag_col:
                    warn_mask = flags[flag_col[0]].astype(str).str.contains('OUT_WARN|RANGE_WARN', na=False)
                    if warn_mask.any():
                        print(f"\nFiltering {warn_mask.sum()} values in {col} due to warnings")
                        df.loc[warn_mask, col].to_csv(output_dir / f"{col}_flaged.csv")
                    df.loc[warn_mask, col] = np.nan
            
            
            filter_path = output_dir / f"{location}_{plant_type}_filtered.csv"
            df.to_csv(filter_path, index=False)
            print(f"Saved filtered data to {filter_path}")

        
        try:
            # Initialize the nested dictionary if location doesn't exist
            if location not in self.outlier_removed_data:
                self.outlier_removed_data[location] = {}
            # Create processed directory for outliers
            outlier_dir = Path('./outputs/processed_data/sap/outliers/MAD')
            outlier_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each column for outliers
            data_cols = [col for col in df.columns if col != 'solar_TIMESTAMP' and 'Unnamed:'not in str(col)]
            for col in data_cols:
                if df[col].isna().all():
                    print(f"Skipping {col} - all values are NA")
                    continue
                    
                # Detect outliers using method B (rolling window)
                outliers = self._detect_outliers(
                    series=df[col],
                    n_std=7,
                    time_window= len(df[col]),  # 24 hours
                    method='C'
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
            self.outlier_removed_data[location][plant_type] = df
            
        except Exception as e:
            print(f"Error processing outliers: {str(e)}")
            return None
        
        try:
            # Create processed directory
            output_dir = Path('./outputs/processed_data/sap/daily')
            output_dir.mkdir(parents=True, exist_ok=True)

            # Get columns to resample (exclude solar_TIMESTAMP)
            data_cols = [col for col in df.columns if col != 'solar_TIMESTAMP']
            
            # Simple daily mean resampling
            daily_df = df[data_cols].resample('D').mean()
            
            # Add '_mean' suffix to column names
            daily_df.columns = [f"{col}_mean" for col in daily_df.columns]
            
            # Save daily data
            output_path = output_dir / f"{location}_{plant_type}_daily.csv"
            daily_df.to_csv(output_path, index=False)
            print(f"Saved daily resampled data to {output_path}")

            return daily_df

        except Exception as e:
            print(f"Error during daily resampling: {str(e)}")
        
        

        

    
    def _detect_outliers(self, series: pd.Series, n_std: float = 3, time_window:int = 1440 ,method: str = 'A') -> pd.Series:
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
            # Updated code using preferred methods
            rolling_mean = rolling_mean.ffill().bfill()
            rolling_std = rolling_std.ffill().bfill()
            
            # Detect outliers
            outliers = np.abs(series - rolling_mean) > (n_std * rolling_std)
            
            print(f"Outliers detected: {outliers.sum()} out of {len(series)} points")
        elif method == 'C':
            # Apply MAD-based spike detection using 30-min windows
            window_size = time_window  # Number of points in window (e.g., 1440 for 30 min at 20 Hz)
            
            # Process each window
            for start_idx in range(0, len(series), window_size):
                end_idx = min(start_idx + window_size, len(series))
                window_data = series.iloc[start_idx:end_idx]
                
                if len(window_data) > 0:
                    # Calculate median
                    x_median = window_data.median()
                    
                    # Calculate MAD
                    mad = np.median(np.abs(window_data - x_median))
                    
                    # Calculate threshold
                    threshold = n_std * mad / 0.6745
                    
                    # Detect spikes
                    window_outliers = (window_data < (x_median - threshold)) | (window_data > (x_median + threshold))
                    
                    # Update outlier mask
                    outliers.iloc[start_idx:end_idx] = window_outliers
                    
            
            print(f"MAD-based outliers detected: {outliers.sum()} out of {len(series)} points")
        return outliers

    def plot_all_plants(self, figsize=(12, 6), save_dir: str = None, skip_empty: bool = True, 
                       plot_limit: int = None, progress_update: bool = True):
        """
        Plot individual trees for all sites with enhanced error handling and progress tracking
        
        Args:
            figsize (tuple): Figure size for plots
            save_dir (str): Directory to save plots
            skip_empty (bool): Skip locations with no valid data
            plot_limit (int): Maximum number of plots per location (None for all)
            progress_update (bool): Print progress updates
        """
        # Create summary dictionary to store processing results
        summary = {
            'total_locations': len(self.outlier_removed_data),
            'processed_locations': 0,
            'successful_plots': 0,
            'failed_plots': 0,
            'skipped_empty': 0,
            'errors': []
        }
        
        # Create save directory if specified
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        # Process each location
        total_locations = len(self.outlier_removed_data)
        for loc_idx, location in enumerate(self.outlier_removed_data, 1):
            if progress_update:
                print(f"\nProcessing location {loc_idx}/{total_locations}: {location}")
            
            for plant_type in self.outlier_removed_data[location]:
                try:
                    # Get data for current location/plant type
                    data = self.outlier_removed_data[location][plant_type]
                    plot_columns = [col for col in data.columns if col != 'solar_TIMESTAMP']
                    
                    # Check if data is empty
                    if skip_empty and all(data[col].isna().all() for col in plot_columns):
                        if progress_update:
                            print(f"Skipping {location}_{plant_type} - no valid data")
                        summary['skipped_empty'] += 1
                        continue
                    
                    # Limit number of plots if specified
                    if plot_limit:
                        plot_columns = plot_columns[:plot_limit]
                    
                    if progress_update:
                        print(f"\nProcessing {location}_{plant_type}")
                        print(f"Plotting {len(plot_columns)} trees")
                    
                    # Create location-specific save directory
                    if save_dir:
                        location_save_dir = save_path / f"{location}_{plant_type}"
                        location_save_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        location_save_dir = None
                    
                    # Plot individual plants
                    self.plot_individual_plants(
                        location=location,
                        plant_type=plant_type,
                        figsize=figsize,
                        save_dir=location_save_dir
                    )
                    
                    summary['successful_plots'] += len(plot_columns)
                    
                except Exception as e:
                    error_msg = f"Error processing {location}_{plant_type}: {str(e)}"
                    print(f"Error: {error_msg}")
                    summary['errors'].append(error_msg)
                    summary['failed_plots'] += 1
            
            summary['processed_locations'] += 1
        
        # Print final summary
        if progress_update:
            print("\nProcessing Summary:")
            print(f"Total locations processed: {summary['processed_locations']}/{summary['total_locations']}")
            print(f"Successful plots: {summary['successful_plots']}")
            print(f"Failed plots: {summary['failed_plots']}")
            print(f"Skipped empty locations: {summary['skipped_empty']}")
            
            if summary['errors']:
                print("\nErrors encountered:")
                for error in summary['errors']:
                    print(f"- {error}")
        
        return summary
    
    def plot_individual_plants(self, location: str, plant_type: str, 
                             figsize=(12, 6), save_dir: str = None):
        """
        Create individual plots for each plant/tree with outlier and flag detection
        
        Args:
            location (str): Location identifier
            plant_type (str): Plant type identifier
            figsize (tuple): Figure size
            save_dir (str): Directory to save plots
        """
        if location not in self.sapflow_data or plant_type not in self.sapflow_data[location]:
            raise ValueError(f"No data found for {location}_{plant_type}")
        
        data = self.outlier_removed_data[location][plant_type].copy()
        plot_columns = [col for col in data.columns if col != 'solar_TIMESTAMP' and col != 'TIMESTAMP' and len(col) > 0 and col not in ['lat', 'long'] and 'Unnamed:'not in str(col)]
        
        print(f"\nPlotting {location}_{plant_type}")
        print(f"Data shape: {data.shape}")
        print(f"Columns to plot: {plot_columns}")
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving plots to {save_path}")
        
        
        
        for column in plot_columns:
            # Load flagged data
            flag_path = Path('./outputs/processed_data/sap/filtered') / f"{column}_flaged.csv"
            if flag_path.exists():
                flagged_data = pd.read_csv(flag_path, parse_dates=['TIMESTAMP']).set_index('TIMESTAMP')
            else:
                flagged_data = None
                print("No flagged data found")
            # Print column statistics
            print(f"\nColumn: {column}")
            print(f"Non-null values: {data[column].count()}")
            print(f"Value range: {data[column].min()} to {data[column].max()}")
            
            # Skip if all values are NaN
            if data[column].isna().all():
                print(f"Skipping {column} - all values are NA")
                continue
            
            # Load outliers data
            outlier_path = Path('./outputs/processed_data/sap/outliers/MAD') / f"{column}_outliers.csv"
            if outlier_path.exists():
                outliers_df = pd.read_csv(outlier_path, parse_dates=['timestamp']).set_index('timestamp')
            else:
                outliers_df = None
                print(f"No outliers data found for {column}")
            
            # Create figure with two subplots: main plot and monthly boxplot
            fig = plt.figure(figsize=(figsize[0], figsize[1] * 1.5))
            gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
            
            # Main time series plot
            ax1 = fig.add_subplot(gs[0])
            
            # Get valid data
            valid_data = data[column].dropna()
            
            if len(valid_data) > 0:
                # Plot base data
                ax1.plot(valid_data.index, valid_data.values,
                        '-b.', alpha=0.5, label='Normal data', linewidth=1,
                        markersize=3)
                
                # Plot flagged data if available
                if flagged_data is not None:
                    ax1.scatter(flagged_data.index, flagged_data.values,
                                  color='red', alpha=0.5, label='Flagged data',
                                  marker='x', s=50)
                
                # Plot outliers if available
                if outliers_df is not None:
                    outlier_mask = outliers_df['is_outlier']
                    outlier_points = outliers_df[outlier_mask]['value']
                    if len(outlier_points) > 0:
                        ax1.scatter(outlier_points.index, outlier_points.values,
                                  color='orange', alpha=0.5, label='Outliers',
                                  marker='o', s=50)
                
                # Customize main plot
                ax1.set_title(f'Sap Flow Time Series - {location} {plant_type}\nTree {column}', pad=20)
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Sap Flow')
                ax1.grid(True, alpha=0.3)
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
                ax1.legend()
                
                # Add data quality info
                valid_percent = (len(valid_data) / len(data)) * 100
                flagged_count = (
    (~flagged_data[column].isna() &  # Not NaN/None
     (flagged_data[column].astype(str).str.strip() != '') &  # Not empty string
     pd.to_numeric(flagged_data[column], errors='coerce').notna()  # Valid number
    ).sum() if flagged_data is not None else 0
)
                outlier_count = len(outlier_points) if outliers_df is not None else 0
                
                info_text = (
                    f'Valid data: {valid_percent:.1f}%\n'
                    f'Flagged points: {flagged_count}\n'
                    f'Outliers: {outlier_count}\n'
                    f'Range: {valid_data.min():.3f} to {valid_data.max():.3f}'
                )
                ax1.text(0.02, 0.98, info_text,
                        transform=ax1.transAxes, va='top',
                        bbox=dict(facecolor='white', alpha=0.8))
                
        
                
                plt.tight_layout()
                
                if save_dir:
                    filename = f"tree_{column}.png"
                    fig.savefig(save_path / filename, bbox_inches='tight', dpi=300)
                    plt.close(fig)
                    print(f"Saved {filename}")
                else:
                    plt.show()
            else:
                plt.close(fig)
                print(f"No valid data to plot for {column}")
    
    def plot_histogram(self, save_dir: str = None):
        """
        Plot histogram of sapflow data
        """
        for location in self.sapf_raw_data:
            for plant_type in self.sapf_raw_data[location]:
                df = self.sapf_raw_data[location][plant_type]
                plot_columns = [col for col in df.columns if col != 'solar_TIMESTAMP' and col != 'TIMESTAMP' and len(col) > 0 and col not in ['lat', 'long'] and 'Unnamed:'not in str(col)]
                # Convert "NA" strings to np.nan
                df = df.replace("NA", np.nan)
                for column in plot_columns:
                    try:
                        # Convert to numeric, coerce errors to NaN
                        series = pd.to_numeric(df[column], errors='coerce')
                        
                        # Remove NaN values
                        series = series.dropna()
                        
                        # Only plot if we have valid data
                        if len(series) > 0:
                            fig = plt.figure(figsize=(12, 6))
                            ax = fig.add_subplot(111)
                            
                            # Plot histogram with statistics
                            sns.histplot(series, bins=50, kde=True, ax=ax)
                            ax.set_title(f'Histogram of {location}_{plant_type}_{column}\n'
                                       f'Mean: {series.mean():.2f}, Std: {series.std():.2f}\n'
                                       f'Valid points: {len(series)}')
                            ax.set_xlabel('Sap Flow')
                            ax.set_ylabel('Frequency')
                            
                            if save_dir:
                                save_path = Path(save_dir)
                                save_path.mkdir(parents=True, exist_ok=True)
                                plt.savefig(save_path / f"{column}_histogram.png")
                                print(f"Saved histogram to {save_path / f'{column}_histogram.png'}")
                            plt.close()  # Close the figure to free memory
                        else:
                            print(f"No valid data for {column}")
                        
                    except Exception as e:
                        print(f"Error plotting histogram for {column}: {str(e)}")
                        continue
    
    
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