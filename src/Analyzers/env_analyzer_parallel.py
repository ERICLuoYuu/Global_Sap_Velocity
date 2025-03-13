from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
import plotly.graph_objects as go
import plotly.io as pio
import concurrent.futures
import multiprocessing
import os
from functools import partial
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings("ignore")
parent_dir = str(Path(__file__).parent.parent.parent)
print(parent_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from src.tools import adjust_time_to_utc, create_timezone_mapping


def _process_env_histogram_column(args):
    """
    Process a single histogram column for environmental data - this must be at module level to be picklable
    
    Args:
        args: Tuple containing (file_path, column_name, location, plant_type, output_path)
    
    Returns:
        Status message string
    """
    file_path, column_name, location, plant_type, output_path = args
    
    try:
        # Load only this column's data
        try:
            col_data = pd.read_csv(file_path, usecols=[column_name])[column_name]
        except ValueError:
            # If column not found in file
            return f"Column {column_name} not found in {Path(file_path).name}"
        
        # Process the column
        col_data = col_data.replace("NA", np.nan)
        series = pd.to_numeric(col_data, errors='coerce').dropna()
        
        if len(series) == 0:
            return f"No valid data for {column_name}"
        
        # Create and save histogram
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        sns.histplot(series, bins=50, kde=True, ax=ax)
        
        ax.set_title(f'Histogram of {location}_{plant_type}_{column_name}\n'
                   f'Mean: {series.mean():.2f}, Std: {series.std():.2f}\n'
                   f'Valid points: {len(series)}')
        ax.set_xlabel(f'{column_name}')
        ax.set_ylabel('Frequency')
        
        if output_path:
            out_dir = Path(output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_dir / f"{location}_{plant_type}_{column_name}_histogram.png")
            plt.close(fig)
            
            # Free memory
            del col_data, series, fig, ax
            gc.collect()
            
            return f"Saved histogram for {column_name}"
        else:
            plt.close(fig)
            
            # Free memory
            del col_data, series, fig, ax
            gc.collect()
            
            return f"Processed histogram for {column_name}"
        
    except Exception as e:
        return f"Error processing {column_name}: {str(e)}"


def _process_env_variable_plot(args):
    """
    Plot a single environmental variable in parallel
    
    Args:
        args: Tuple containing (location, plant_type, variable, data_path, figsize, save_dir)
        
    Returns:
        Status message string
    """
    location, plant_type, variable, data_path, figsize, save_dir = args
    
    try:
        # Load the saved data file
        data = pd.read_csv(data_path, parse_dates=['TIMESTAMP']).set_index('TIMESTAMP')
        
        # Load flagged data
        flag_path = Path('./outputs/processed_data/env/filtered') / f"{location}_{plant_type}_{variable}_flagged.csv"
        if flag_path.exists():
            flagged_data = pd.read_csv(flag_path, parse_dates=['TIMESTAMP']).set_index('TIMESTAMP')
        else:
            flagged_data = None
            
        # Skip if all values are NaN
        if data[variable].isna().all():
            return f"Skipping {variable} - all values are NA"
            
        # Load outliers data
        outlier_path = Path('./outputs/processed_data/env/outliers') / f"{location}{plant_type}_{variable}_outliers.csv"
        if outlier_path.exists():
            outliers_df = pd.read_csv(outlier_path, parse_dates=['timestamp']).set_index('timestamp')
        else:
            outliers_df = None
            
        # Create figure with two subplots: main plot and monthly boxplot
        fig = plt.figure(figsize=(figsize[0], figsize[1] * 1.5))
        gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
        
        # Main time series plot
        ax1 = fig.add_subplot(gs[0])
        
        # Get valid data
        valid_data = data[variable].dropna()
        
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
            ax1.set_title(f'Environmental Variable: {variable}\nLocation: {location} {plant_type}', pad=20)
            ax1.set_xlabel('Date')
            ax1.set_ylabel(variable)
            ax1.grid(True, alpha=0.3)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            ax1.legend()
            
            # Add data quality info
            valid_percent = (len(valid_data) / len(data)) * 100
            flagged_count = (
                (~flagged_data[variable].isna() if flagged_data is not None else 0) &
                (flagged_data[variable].astype(str).str.strip() != '' if flagged_data is not None else 0) &
                (pd.to_numeric(flagged_data[variable], errors='coerce').notna() if flagged_data is not None else 0)
            ).sum() if flagged_data is not None else 0
            
            outlier_count = len(outlier_points) if outliers_df is not None and 'outlier_points' in locals() else 0
            
            info_text = (
                f'Valid data: {valid_percent:.1f}%\n'
                f'Flagged points: {flagged_count}\n'
                f'Outliers: {outlier_count}\n'
                f'Range: {valid_data.min():.3f} to {valid_data.max():.3f}'
            )
            ax1.text(0.02, 0.98, info_text,
                    transform=ax1.transAxes, va='top',
                    bbox=dict(facecolor='white', alpha=0.8))
            
            # Monthly boxplot in second subplot - grouped by month name only
            ax2 = fig.add_subplot(gs[1])

            # Group by month name (Jan, Feb, etc.) regardless of year
            valid_data_with_month = valid_data.copy()
            month_indices = valid_data.index.month  # Extract just the month number (1-12)

            # Define month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            # Group the data by month number
            monthly_boxplot_data = []
            month_labels = []

            # Process each month (1-12)
            for month_num in range(1, 13):  # 1 to 12
                # Get data for this month across all years
                month_data = valid_data[month_indices == month_num].values
                month_data = [v for v in month_data if pd.notna(v)]
                
                if len(month_data) > 0:  # Only add months that have data
                    monthly_boxplot_data.append(month_data)
                    month_labels.append(month_names[month_num-1])

            # Only proceed if we have data to plot
            if monthly_boxplot_data:
                # Create the boxplot with improved visual settings
                bp = ax2.boxplot(monthly_boxplot_data, labels=month_labels, 
                            patch_artist=True, showfliers=False)
                
                # Customize boxplot appearance
                for box in bp['boxes']:
                    box.set(color='blue', linewidth=1)
                    box.set(facecolor='lightblue', alpha=0.7)
                for whisker in bp['whiskers']:
                    whisker.set(color='blue', linewidth=1)
                for cap in bp['caps']:
                    cap.set(color='blue', linewidth=1)
                for median in bp['medians']:
                    median.set(color='red', linewidth=1.5)
                
                ax2.set_title('Monthly Distribution (all years combined)')
                ax2.set_xlabel('Month')
                ax2.set_ylabel(variable)
                
                # Set ylim to match the main plot for consistency
                ax2.set_ylim(ax1.get_ylim())
                
                # Add grid for easier reading
                ax2.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            if save_dir:
                filename = f"{variable}.png"
                fig.savefig(Path(save_dir) / filename, bbox_inches='tight', dpi=300)
                plt.close(fig)
                return f"Saved {filename}"
            else:
                plt.close(fig)
                return "Plot completed (not saved)"
        else:
            plt.close(fig)
            return f"No valid data to plot for {variable}"
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"Error plotting {variable}: {str(e)}\n{error_detail}"


def _process_env_file(args):
    """
    Process a single environmental data file with its corresponding flags
    
    Args:
        args: Tuple containing (file_path, time_zone_map)
                
    Returns:
        Tuple of (location, plant_type, df, flags)
    """
    file_path, time_zone_map = args
    
    try:
        # Extract file information
        parts = file_path.stem.split('_')
        
        # Better handling of file name parsing
        if 'env' not in file_path.stem:
            return None, None, None, None
                
        location = '_'.join(parts[:2])  # First two components typically form the location
        plant_type = '_'.join(parts[2:])
        
        print(f"\nProcessing file: {file_path}")
        print(f"Extracted location: {location}, plant_type: {plant_type}")
        
        column_mapping = {
            'TIMESTAMP_solar': 'solar_TIMESTAMP',  # Map alternate name to standard name
        }
        
        # Load environmental data with more robust error handling
        try:
            df = pd.read_csv(file_path, parse_dates=['TIMESTAMP'])
            print(f"Successfully loaded {file_path.name}, shape: {df.shape}")
        except Exception as e:
            print(f"Failed to load {file_path.name}: {str(e)}")
            return None, None, None, None
        
        # Check for required columns
        required_cols = ['TIMESTAMP']
        has_required_cols = all(col in df.columns or col in column_mapping for col in required_cols)
        
        if not has_required_cols:
            print(f"Missing required columns in {file_path.name}. Available columns: {list(df.columns)}")
            return None, None, None, None
        
        # Rename columns as needed
        df = df.rename(columns=column_mapping)
        
        # Try to locate corresponding env_md data
        time_zone_file = file_path.parent / f"{file_path.stem.replace('env_data', 'env_md')}.csv"
        
        # Check if time zone file exists
        if not time_zone_file.exists():
            print(f"Warning: Time zone file not found: {time_zone_file}")
            print("Using UTC as default timezone")
            time_zone = "UTC"
        else:
            try:
                tz_df = pd.read_csv(time_zone_file)
                if 'env_time_zone' not in tz_df.columns:
                    print(f"Warning: env_time_zone column not found in {time_zone_file}")
                    print(f"Available columns: {list(tz_df.columns)}")
                    time_zone = "UTC"
                else:
                    time_zone = tz_df['env_time_zone'].iloc[0]
            except Exception as e:
                print(f"Error reading time zone file {time_zone_file}: {str(e)}")
                time_zone = "UTC"
                
        print(f"Using time zone: {time_zone}")
        
        # Make timestamps into datetime objects with better error handling
        for col in ['TIMESTAMP', 'solar_TIMESTAMP']:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    null_timestamps = df[col].isna().sum()
                    if null_timestamps > 0:
                        print(f"Warning: {null_timestamps} null values in {col} after conversion")
                except Exception as e:
                    print(f"Error converting {col} to datetime: {str(e)}")
                    if col == 'TIMESTAMP':  # Only TIMESTAMP is required
                        return None, None, None, None
         
        # Apply time zone adjustment with better error handling
        try:
            df['TIMESTAMP'] = df['TIMESTAMP'].apply(
                lambda x: adjust_time_to_utc(x, time_zone, time_zone_map) if pd.notna(x) else x
            )
            
            if 'solar_TIMESTAMP' in df.columns:
                df['solar_TIMESTAMP'] = df['solar_TIMESTAMP'].apply(
                    lambda x: adjust_time_to_utc(x, time_zone, time_zone_map) if pd.notna(x) else x
                )
            
            # Check for null values after timezone adjustment
            null_timestamps = df['TIMESTAMP'].isna().sum()
            if null_timestamps > 0:
                print(f"Warning: {null_timestamps} null values in TIMESTAMP after timezone adjustment")
        except Exception as e:
            print(f"Error during timezone adjustment: {str(e)}")
            return None, None, None, None
        
        # Save timezone adjusted data
        output_dir = Path('./outputs/processed_data/env/timezone_adjusted')
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        try:
            df.to_csv(output_dir / f'{file_path.stem}_adjusted.csv')
            print(f"Saved timezone adjusted data to {output_dir / f'{file_path.stem}_adjusted.csv'}")
        except Exception as e:
            print(f"Error saving timezone adjusted data: {str(e)}")
            # Continue anyway as this is not critical
        
        # Get site coordinates if available
        try:
            site_md_file = file_path.parent / f"{file_path.stem.replace('_env_data', '_site_md')}.csv"
            if site_md_file.exists():
                site_md = pd.read_csv(site_md_file)
                if 'si_lat' in site_md.columns and 'si_long' in site_md.columns:
                    latitude = site_md['si_lat'].values[0]
                    longitude = site_md['si_long'].values[0]
                    df['lat'] = latitude
                    df['long'] = longitude
                    print(f"Added coordinates: lat={latitude}, long={longitude}")
                else:
                    print(f"Missing coordinate columns in {site_md_file.name}")
            else:
                print(f"No site metadata file found at {site_md_file}")
        except Exception as e:
            print(f"Error loading site coordinates: {str(e)}")
            # Continue without coordinates
        
        # Find and load flags file
        flags = None
        flag_file = file_path.parent / f"{file_path.stem.replace('_data', '_flags')}.csv"
        if flag_file.exists():
            try:
                flags = pd.read_csv(flag_file, parse_dates=['TIMESTAMP'])
                flags = flags.rename(columns=column_mapping)
                for col in ['TIMESTAMP']:
                    if col in flags.columns:
                        flags[col] = pd.to_datetime(flags[col], errors='coerce')
                        flags[col] = flags[col].apply(
                            lambda x: adjust_time_to_utc(x, time_zone, time_zone_map) if pd.notna(x) else x
                        )
                print(f"Successfully loaded flags from {flag_file}")
            except Exception as e:
                print(f"Error loading flags file {flag_file}: {str(e)}")
                # Continue without flags
                flags = None
        else:
            print(f"No flags file found at {flag_file}")
        
        return location, plant_type, df, flags
                
    except Exception as e:
        import traceback
        print(f"Unhandled error processing {file_path.name}:")
        print(traceback.format_exc())
        return None, None, None, None


def _detect_env_outliers_parallel(series, time_windows, n_std=7, method='C'):
    """
    Detect outliers in parallel for different time windows in environmental data
    
    Args:
        series: Time series data
        time_windows: List of (start_idx, end_idx) tuples defining windows
        n_std: Number of standard deviations for outlier threshold
        method: Outlier detection method
        
    Returns:
        Series of boolean outlier flags
    """
    # Initialize outlier mask
    outliers = pd.Series(False, index=series.index)
    
    def process_window(window_data):
        start_idx, end_idx = window_data
        window_series = series.iloc[start_idx:end_idx]
        
        if len(window_series) == 0:
            return start_idx, end_idx, None
        
        # Calculate median
        x_median = window_series.median()
        
        # Calculate MAD
        mad = np.median(np.abs(window_series - x_median))
        
        # Handle case where mad is 0
        if mad == 0:
            mad = np.mean(np.abs(window_series - x_median)) or 1e-10
        
        # Calculate threshold
        threshold = n_std * mad / 0.6745
        
        # Detect spikes
        window_outliers = (window_series < (x_median - threshold)) | (window_series > (x_median + threshold))
        
        return start_idx, end_idx, window_outliers
    
    # Process windows in parallel (using up to 4 workers for this sub-task)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_window, time_windows))
    
    # Combine results
    for start_idx, end_idx, window_outliers in results:
        if window_outliers is not None:
            outliers.iloc[start_idx:end_idx] = window_outliers
    
    return outliers


def _process_numeric_column(args):
    """Process a single numeric column, converting to appropriate data type"""
    col, df = args
    try:
        if col in df.columns:  # Check if column exists
            series = pd.to_numeric(df[col], errors='coerce')
            non_null = series.count()
            if non_null == 0:
                return col, series, (None, None), 0
            return col, series, (series.min(), series.max()), non_null
        else:
            return col, None, (None, None), 0
    except Exception as e:
        print(f"Error converting column {col}: {str(e)}")
        return col, df[col] if col in df.columns else None, (None, None), 0


def _process_column_for_flags(args):
    """Process a single column for flag-based filtering"""
    col, df, flags, output_dir = args
    flag_col = [fcol for fcol in flags.columns if col in fcol]
    if not flag_col:
        return col, None
    
    warn_mask = flags[flag_col[0]].astype(str).str.contains('OUT_WARN|RANGE_WARN', na=False)
    if warn_mask.any():
        # Export filtered points
        filtered_data = df.loc[warn_mask, col]
        filtered_data.to_csv(output_dir / f"{col}_flaged.csv")
        return col, warn_mask
    return col, None


def _process_outliers_for_column(args):
    """Process outlier detection for a single column"""
    col, df, location, plant_type, outlier_dir, n_std = args
    try:
        if col not in df.columns:
            return col, None
            
        if df[col].isna().all():
            print(f"Skipping outlier detection for {col} - all values are NA")
            return col, None
        
        # Skip columns with too few values
        non_null_count = df[col].count()
        if non_null_count < 10:  # Minimum threshold for meaningful outlier detection
            print(f"Skipping outlier detection for {col} - only {non_null_count} non-null values")
            return col, None
        
        # Prepare time windows for parallel processing
        window_size = len(df[col])  # Cap window size
        time_windows = []
        for start_idx in range(0, len(df[col]), window_size):
            end_idx = min(start_idx + window_size, len(df[col]))
            time_windows.append((start_idx, end_idx))
        
        # Detect outliers using parallel MAD-based method
        outliers = _detect_env_outliers_parallel(
            series=df[col],
            time_windows=time_windows,
            n_std=n_std,
            method='C'
        )
        
        # Save outlier information
        outlier_df = pd.DataFrame({
            'timestamp': df.index,
            'value': df[col],
            'is_outlier': outliers
        })
        
        outlier_path = outlier_dir / f"{location}{plant_type}_{col}_outliers.csv"
        outlier_df.to_csv(outlier_path)
        print(f"Saved outliers for {col} to {outlier_path}")
        
        return col, outliers
    except Exception as e:
        import traceback
        print(f"Error processing outliers for column {col}:")
        print(traceback.format_exc())
        return col, None


def _calculate_column_range(args):
    """Calculate global min/max for a column across all datasets"""
    variable, env_data = args
    try:
        temp_data = []
        for location in env_data:
            for plant_type in env_data[location]:
                data = env_data[location][plant_type]
                if variable not in data.columns:
                    continue
                valid_data = data[variable].dropna().tolist()
                if valid_data:  # Only add if there's valid data
                    temp_data.extend(valid_data)
        
        if not temp_data:
            return variable, None
            
        min_val = np.min(temp_data)
        max_val = np.max(temp_data)
        
        return variable, {'min': min_val, 'max': max_val}
        
    except Exception as e:
        print(f"Error calculating range for {variable}: {str(e)}")
        return variable, None


def _standardize_dataset_for_file(args):
    """Standardize a dataset using min/max cache"""
    df, location, plant_type, minmax_cache, std_dir = args
    try:
        
        print(f"\nStandardizing data for {location}_{plant_type}")
        
        # Track successful standardizations
        standardized_cols = []
        
        for variable in df.columns:
            try:
                if variable not in minmax_cache or variable in ['lat', 'long', 'solar_TIMESTAMP']:
                    continue
                    
                # Standardize the column
                min_val = minmax_cache[variable]['min']
                max_val = minmax_cache[variable]['max']
                
                # Check for zero range
                if max_val == min_val:
                    continue
                
                # Standardize data
                df[variable] = (df[variable] - min_val) / (max_val - min_val)
                standardized_cols.append(variable)
                
            except Exception as e:
                print(f"Error standardizing {variable}: {str(e)}")
                continue
        
        # Only return if some columns were standardized
        if standardized_cols:
            # Round to 3 decimal places
            df[standardized_cols] = df[standardized_cols].apply(lambda x: round(x, 3))
            # Add lat/long columns if they exist
            saved_cols = standardized_cols.copy()
            if 'lat' in df.columns and 'long' in df.columns:
                saved_cols.extend(['lat', 'long'])
            
            # Save standardized data
            output_path = Path(std_dir) / f"{location}_{plant_type}_standardized.csv"
            df[saved_cols].to_csv(output_path)
            print(f"Saved standardized data to {output_path}")
            
            return location, plant_type, len(standardized_cols), saved_cols, str(output_path)
        
        return location, plant_type, 0, [], None
        
    except Exception as e:
        print(f"Error standardizing dataset: {str(e)}")
        return "unknown", "unknown", 0, [], None


class EnvironmentalAnalyzer:
    """
    Analyzer for Environmental data with parallel processing capabilities
    """
    def __init__(self, base_dir: str = "./data/raw/0.1.5/0.1.5/csv/sapwood", max_workers: int = None):
        self.data_dir = Path(base_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Directory not found: {self.data_dir}")
        
        # Data storage
        self.env_raw_data = {}
        self.env_data = {}
        self.outlier_removed_data = {}
        self._minmax_cache = {}
        
        # Set default max_workers to number of CPU cores
        self.max_workers = max_workers if max_workers is not None else max(1, multiprocessing.cpu_count() - 1)
        print(f"Using {self.max_workers} workers for parallel processing")
        
        # Initialize time zone mapping once to avoid repeated creation
        self.time_zone_map = create_timezone_mapping()
        
        # Load data on initialization
        self._load_environmental_data_parallel()
    
    def _load_environmental_data_parallel(self):
        """Load all environmental data files in parallel"""
        try:
            env_files = list(self.data_dir.glob("*_env_data.csv"))
            print(f"Found {len(env_files)} environmental files")
            
            # For debugging, you can limit to a few files
            # env_files = env_files[:20]  # Process only first 5 files
            
            # Setup the multiprocessing parameters
            if self.max_workers > multiprocessing.cpu_count():
                print(f"Warning: Requested {self.max_workers} workers but only {multiprocessing.cpu_count()} CPUs available")
                self.max_workers = max(1, multiprocessing.cpu_count() - 1)
                
            print(f"Using {self.max_workers} workers for parallel processing")
            
            # Prepare args for each file
            process_args = [(file, self.time_zone_map) for file in env_files]
            
            # Process files in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Use tqdm to show progress bar
                results = list(tqdm(
                    executor.map(_process_env_file, process_args),
                    total=len(process_args),
                    desc="Loading environmental files"
                ))
                
            # Count successful and failed files
            successful = 0
            failed = 0
            
            # Process results and update data dictionaries
            for location, plant_type, df, flags in results:
                if location is not None and plant_type is not None and df is not None:
                    # Initialize nested dictionaries if needed
                    if location not in self.env_raw_data:
                        self.env_raw_data[location] = {}
                    
                    # Store raw data
                    self.env_raw_data[location][plant_type] = df
                    
                    # Process the data
                    processed_df = self._process_environmental_data_parallel(df, location, plant_type, flags)
                    
                    if processed_df is not None:
                        # Initialize nested dictionaries if needed
                        if location not in self.env_data:
                            self.env_data[location] = {}
                        if location not in self.outlier_removed_data:
                            self.outlier_removed_data[location] = {}
                        
                        # Store processed data
                        self.env_data[location][plant_type] = processed_df
                        self.outlier_removed_data[location][plant_type] = processed_df.copy()
                        
                        # Debug print
                        print(f"Successfully processed {location}_{plant_type}")
                        successful += 1
                    else:
                        print(f"Failed to process {location}_{plant_type}")
                        failed += 1
                else:
                    failed += 1
                    
            print(f"\nProcessing summary: {successful} successful, {failed} failed")
            print(f"self.env_data has {len(self.env_data)} locations")
            print(f"self.outlier_removed_data has {len(self.outlier_removed_data)} locations")
            
            # After loading data, standardize it
            self._standardize_all_data_parallel()
            
        except Exception as e:
            import traceback
            print(f"Unhandled error in _load_environmental_data_parallel:")
            print(traceback.format_exc())
    
    def _process_environmental_data_parallel(self, df, location=None, plant_type=None, flags=None):
        """
        Process environmental data with parallel processing for flags and outliers
        
        Returns:
            Processed DataFrame
        """
        try:
            print("\nProcessing data shape:", df.shape)
            
            # Convert string "NA" to numpy NaN
            df = df.replace("NA", np.nan)
            
            # Convert numeric columns to float, with error handling
            numeric_cols = [col for col in df.columns if col not in ['TIMESTAMP', 'solar_TIMESTAMP', 'lat', 'long']]
            
            # Process numeric columns in parallel
            process_args = [(col, df) for col in numeric_cols]
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(_process_numeric_column, process_args))
            
            # Track columns with data
            valid_columns = []
            
            # Update DataFrame with processed columns
            for col, series, (min_val, max_val), non_null in results:
                if series is not None:
                    df[col] = series
                    if min_val is not None and max_val is not None:
                        print(f"Column {col}: {non_null} non-null values, range: {min_val} to {max_val}")
                        valid_columns.append(col)
                    else:
                        print(f"Column {col}: All null values or conversion error")
            
            if not valid_columns:
                print(f"No valid data columns found for {location}_{plant_type}")
                return None
            
            print(f"Found {len(valid_columns)} valid data columns with non-null values")
            
            # Set index and sort
            df = df.set_index('TIMESTAMP')
            df = df.sort_index()
            
            # Remove duplicates
            if df.index.duplicated().any():
                print(f"Removing {df.index.duplicated().sum()} duplicate timestamps")
                df = df[~df.index.duplicated(keep='first')]
            
            # Apply flag-based filtering in parallel
            if flags is not None:
                try:
                    flags = flags.set_index('TIMESTAMP').sort_index()
                    if flags.index.duplicated().any():
                        flags = flags[~flags.index.duplicated(keep='first')]
                    
                    data_cols = [col for col in df.columns if col != 'solar_TIMESTAMP' and col not in ['lat', 'long']]
                    
                    # Create output directory
                    output_dir = Path('./outputs/processed_data/env/filtered')
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Process columns in parallel
                    process_args = [(col, df, flags, output_dir) for col in data_cols]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                        results = list(executor.map(_process_column_for_flags, process_args))
                    
                    # Apply filter masks
                    for col, mask in results:
                        if mask is not None:
                            df.loc[mask, col] = np.nan
                    
                    # Save filtered data
                    filter_path = output_dir / f"{location}_{plant_type}_filtered.csv"
                    df.to_csv(filter_path)
                    print(f"Saved filtered data to {filter_path}")
                    
                except Exception as e:
                    print(f"Error during flag processing: {str(e)}")
                    # Continue without flag filtering
            
            # Process outliers in parallel
            try:
                # Create processed directory for outliers
                outlier_dir = Path('./outputs/processed_data/env/outliers')
                outlier_dir.mkdir(parents=True, exist_ok=True)
                outlier_removed_dir = Path('./outputs/processed_data/env/outliers_removed')
                outlier_removed_dir.mkdir(parents=True, exist_ok=True)
                
                # Process each column for outliers in parallel
                data_cols = [col for col in df.columns if col != 'solar_TIMESTAMP' and col not in ['lat', 'long'] and 'Unnamed:' not in str(col)]
                
                # Process in parallel
                process_args = [(col, df, location, plant_type, outlier_dir, 7) for col in data_cols]
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    outlier_results = list(executor.map(_process_outliers_for_column, process_args))
                
                # Apply outlier masks
                for col, outliers in outlier_results:
                    if outliers is not None:
                        df.loc[outliers, col] = np.nan
                # save outlier removed data
                outlier_removed_path = outlier_removed_dir / f"{location}_{plant_type}_outliers_removed.csv"
                df.to_csv(outlier_removed_path)
                print(f"Saved outlier removed data to {outlier_removed_path}")

            
            except Exception as e:
                import traceback
                print(f"Error in outlier processing:")
                print(traceback.format_exc())
                # Continue without outlier processing
            
            # Create daily dataframe
            try:
                # Create processed directory
                output_dir = Path('./outputs/processed_data/env/daily')
                output_dir.mkdir(parents=True, exist_ok=True)

                # Get columns to resample
                data_cols = [col for col in df.columns if col != 'solar_TIMESTAMP' and col not in ['lat', 'long']]
                
                # Identify columns that need sum vs mean
                sum_cols = [col for col in data_cols if 'precip' in col.lower()]
                mean_cols = [col for col in data_cols if col not in sum_cols]

                # Resample separately
                daily_sums = df[sum_cols].resample('D').sum() if sum_cols else None
                daily_means = df[mean_cols].resample('D').mean() if mean_cols else None

                # Combine results
                if daily_sums is not None and daily_means is not None:
                    daily_df = pd.concat([daily_sums, daily_means], axis=1)
                elif daily_sums is not None:
                    daily_df = daily_sums
                else:
                    daily_df = daily_means

                # Add appropriate suffixes
                daily_df.columns = [f"{col}_sum" if col in sum_cols else f"{col}_mean" 
                                  for col in daily_df.columns]
                
                # Save daily data
                output_path = output_dir / f"daily_{location}_{plant_type}.csv"
                daily_df.to_csv(output_path)
                print(f"Saved daily resampled data to {output_path}")

                # Return the original processed data (not the daily resampled)
                return df
                
            except Exception as e:
                import traceback
                print(f"Error during daily resampling:")
                print(traceback.format_exc())
                # Return the filtered/outlier-processed data even if daily resampling fails
                return df
                
        except Exception as e:
            import traceback
            print(f"Unhandled error in _process_environmental_data_parallel:")
            print(traceback.format_exc())
            return None
    
    def _detect_outliers(self, series, n_std=3, time_window=1440, method='C'):
        """
        Detect outliers using different methods
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
        n_std : float
            Number of standard deviations for outlier threshold
        time_window : int
            Number of points in each window for method B and C
        method : str
            Method for outlier detection: 'A' (monthly), 'B' (rolling), 'C' (MAD-based)
                
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
            rolling_mean = rolling_mean.ffill().bfill()
            rolling_std = rolling_std.ffill().bfill()
            
            # Detect outliers
            outliers = np.abs(series - rolling_mean) > (n_std * rolling_std)
            
            print(f"Outliers detected: {outliers.sum()} out of {len(series)} points")
            
        elif method == 'C':
            # For method C, use the parallel implementation
            # Prepare time windows
            window_size = min(time_window, len(series))  # Cap window size
            time_windows = []
            for start_idx in range(0, len(series), window_size):
                end_idx = min(start_idx + window_size, len(series))
                time_windows.append((start_idx, end_idx))
            
            # Use the parallel version
            outliers = _detect_env_outliers_parallel(series, time_windows, n_std, method)
            
            print(f"MAD-based outliers detected: {outliers.sum()} out of {len(series)} points")
            
        return outliers
    
    def _standardize_all_data_parallel(self):
        """Standardize environmental data using parallel processing"""
        try:
            print("\nStarting parallel data standardization process...")
            
            # Verify data is loaded
            if not self.env_data:
                raise ValueError("No environmental data has been loaded")
            
            # Create processed directory for standardized data
            std_dir = Path('./outputs/processed_data/env/standardized')
            std_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all datasets with validation
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
                    
                    all_dataframes.append((location, plant_type, df))
            
            if not all_dataframes:
                raise ValueError("No valid DataFrames found for standardization")
                
            # Find all columns across datasets
            all_columns = set()
            for location in self.env_data:
                for plant_type in self.env_data[location]:
                    df = self.env_data[location][plant_type]
                    all_columns.update(df.columns)
            
            # Remove special columns
            all_columns = [col for col in all_columns if col not in ['solar_TIMESTAMP', 'lat', 'long']]
            
            if not all_columns:
                raise ValueError("No columns found across datasets")
                
            print(f"\nFound {len(all_columns)} columns to standardize: {all_columns}")
            
            # Calculate global min/max for all columns in parallel
            process_args = [(variable, self.env_data) for variable in all_columns]
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(_calculate_column_range, process_args))
            
            # Update cache with results
            for variable, range_data in results:
                if range_data is not None:
                    self._minmax_cache[variable] = range_data
                    print(f"\nGlobal range for {variable}: "
                          f"{range_data['min']} to {range_data['max']}")
            
            # Standardize datasets in parallel using saved files
            process_args = [(df, location, plant_type, self._minmax_cache, str(std_dir)) 
                          for location, plant_type, df in all_dataframes]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(tqdm(
                    executor.map(_standardize_dataset_for_file, process_args),
                    total=len(process_args),
                    desc="Standardizing datasets"
                ))
            
            # Process results
            successful = 0
            for location, plant_type, stdz_count, stdz_cols, output_path in results:
                if output_path and stdz_count > 0:
                    # Update stored data if possible
                    if location in self.env_data and plant_type in self.env_data[location]:
                        try:
                            standardized_df = pd.read_csv(output_path, parse_dates=['TIMESTAMP']).set_index('TIMESTAMP')
                            self.env_data[location][plant_type] = standardized_df
                            successful += 1
                        except Exception as e:
                            print(f"Error updating data for {location}_{plant_type}: {str(e)}")
            
            
            # Summary report
            print("\nStandardization Summary:")
            print(f"Successfully standardized {successful} out of {len(all_dataframes)} datasets")
            
            return successful
            
        except Exception as e:
            import traceback
            print(f"Critical error during standardization process: {str(e)}")
            print(traceback.format_exc())
            return 0
    
    def plot_environmental_variables_parallel(self, location: str, plant_type: str, 
                                     figsize=(12, 6), save_dir: str = None):
        """
        Create individual plots for environmental variables in parallel
        
        Args:
            location (str): Location identifier
            plant_type (str): Plant type identifier
            figsize (tuple): Figure size
            save_dir (str): Directory to save plots
        """
        if location not in self.outlier_removed_data or plant_type not in self.outlier_removed_data[location]:
            raise ValueError(f"No data found for {location}_{plant_type}")
        
        data = self.outlier_removed_data[location][plant_type].copy()
        plot_columns = [col for col in data.columns if col != 'solar_TIMESTAMP' and col != 'TIMESTAMP' 
                      and len(col) > 0 and col not in ['lat', 'long'] and 'Unnamed:' not in str(col)]
        
        print(f"\nPlotting environmental data for {location}_{plant_type}")
        print(f"Data shape: {data.shape}")
        print(f"Variables to plot: {len(plot_columns)}")
        
        if save_dir:
            save_path = Path(save_dir) / f"{location}_{plant_type}_env"
            save_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving plots to {save_path}")
        
        # Save data to temporary file to be loaded in parallel process
        temp_data_path = Path('./outputs/processed_data/env/temp') / f"{location}_{plant_type}_plot_data.csv"
        temp_data_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(temp_data_path)
        
        # Prepare plot data
        plot_data = [(location, plant_type, variable, str(temp_data_path), figsize, str(save_path)) for variable in plot_columns]
        
        # Plot in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(tqdm(
                executor.map(_process_env_variable_plot, plot_data),
                total=len(plot_data),
                desc=f"Plotting {location}_{plant_type}"
            ))
        
        # Clean up temp file
        try:
            if temp_data_path.exists():
                temp_data_path.unlink()
        except:
            pass
        
        # Print results
        for result in results:
            print(result)
    
    def plot_all_parallel(self, figsize=(12, 6), save_dir: str = None, skip_empty: bool = True, 
                 plot_limit: int = None, progress_update: bool = True):
        """
        Plot all environmental variables for all sites in parallel
        
        Args:
            figsize (tuple): Figure size for plots
            save_dir (str): Directory to save plots
            skip_empty (bool): Skip locations with no valid data
            plot_limit (int): Maximum number of variables to plot per location
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
        
        # Prepare plotting tasks
        plot_tasks = []
        for location in self.outlier_removed_data:
            for plant_type in self.outlier_removed_data[location]:
                try:
                    # Get data for current location/plant type
                    data = self.outlier_removed_data[location][plant_type]
                    plot_columns = [col for col in data.columns if col != 'solar_TIMESTAMP' and col != 'TIMESTAMP'
                                  and col not in ['lat', 'long'] and 'Unnamed:' not in str(col)]
                    
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
                        print(f"\nAdding {location}_{plant_type} to plot queue")
                        print(f"Will plot {len(plot_columns)} variables")
                    
                    # Create location-specific save directory
                    if save_dir:
                        location_save_dir = save_path / f"{location}_{plant_type}"
                        location_save_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        location_save_dir = None
                    
                    # Add task to queue
                    plot_tasks.append((location, plant_type, figsize, location_save_dir))
                    
                except Exception as e:
                    error_msg = f"Error preparing {location}_{plant_type}: {str(e)}"
                    print(f"Error: {error_msg}")
                    summary['errors'].append(error_msg)
                    summary['failed_plots'] += 1
        
        # Process tasks in sequential batches (not fully parallel to prevent memory issues)
        batch_size = min(4, len(plot_tasks))
        for i in range(0, len(plot_tasks), batch_size):
            batch = plot_tasks[i:i+batch_size]
            
            # Process this batch
            for loc, pt, fs, sd in tqdm(batch, desc=f"Processing batch {i//batch_size + 1}/{(len(plot_tasks) + batch_size - 1)//batch_size}"):
                try:
                    self.plot_environmental_variables_parallel(loc, pt, fs, sd)
                    summary['processed_locations'] += 1
                    
                    # Count successful plots
                    data = self.outlier_removed_data[loc][pt]
                    plot_columns = [col for col in data.columns if col != 'solar_TIMESTAMP' and col != 'TIMESTAMP'
                                  and col not in ['lat', 'long'] and 'Unnamed:' not in str(col)]
                                  
                    if plot_limit:
                        plot_columns = plot_columns[:plot_limit]
                    summary['successful_plots'] += len(plot_columns)
                    
                except Exception as e:
                    error_msg = f"Error processing {loc}_{pt}: {str(e)}"
                    print(f"Error: {error_msg}")
                    summary['errors'].append(error_msg)
                    summary['failed_plots'] += 1
        
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
    
    def plot_histogram_parallel(self, save_dir: str = None, max_workers=2, chunk_size=10):
        """
        Plot histograms with memory-efficient parallel processing
        
        Args:
            save_dir: Directory to save histograms
            max_workers: Maximum number of parallel workers (use small number)
            chunk_size: Number of columns to process in each parallel batch
        """
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        # Find all environmental data files
        env_files = list(self.data_dir.glob("*_env_data.csv"))
        print(f"Found {len(env_files)} environmental data files")
        
        # Process each file directly
        for file_path in env_files:
            # Extract location and plant type from filename
            parts = file_path.stem.split('_')
            if len(parts) < 4:  # Need at least location, plant type, and env_data
                print(f"Skipping file with unexpected name format: {file_path.name}")
                continue
                
            # Extract location and plant type 
            location = '_'.join(parts[:2])  # First two components typically form the location
            
            # Everything between location and env_data is plant type
            try:
                env_idx = next(i for i, part in enumerate(parts) if 'env' in part)
                plant_type = '_'.join(parts[2:env_idx])
            except StopIteration:
                # If 'env' not found, use default approach
                plant_type = '_'.join(parts[2:-1])
            
            print(f"\nProcessing histograms for {location}_{plant_type} from file {file_path.name}")
            
            # Get column names from the first few rows
            try:
                data_sample = pd.read_csv(file_path, nrows=5)
            except Exception as e:
                print(f"Error reading file {file_path.name}: {str(e)}")
                continue
            
            plot_columns = [col for col in data_sample.columns 
                            if col != 'solar_TIMESTAMP' and col != 'TIMESTAMP' 
                            and len(col) > 0 and col not in ['lat', 'long'] and 'Unnamed:' not in str(col)]
            
            # Create location-specific save directory
            if save_dir:
                location_save_dir = Path(save_dir) / f"{location}_{plant_type}"
                location_save_dir.mkdir(parents=True, exist_ok=True)
            
            # Split columns into chunks to reduce memory pressure
            column_chunks = [plot_columns[i:i + chunk_size] for i in range(0, len(plot_columns), chunk_size)]
            
            # Process each chunk of columns
            for chunk_idx, columns_chunk in enumerate(column_chunks):
                print(f"Processing chunk {chunk_idx+1}/{len(column_chunks)} with {len(columns_chunk)} columns")
                
                # Prepare arguments for the picklable function
                process_args = [(str(file_path), column, location, plant_type, 
                               str(location_save_dir) if save_dir else None) 
                            for column in columns_chunk]
                
                # Process this chunk in parallel
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    results = list(tqdm(
                        executor.map(_process_env_histogram_column, process_args),
                        total=len(process_args),
                        desc=f"Processing chunk {chunk_idx+1}"
                    ))
                
                # Force garbage collection between chunks
                gc.collect()
                
                # Print results
                for result in results:
                    print(result)
                
                print(f"Completed chunk {chunk_idx+1}/{len(column_chunks)}")
    
    def get_all_columns(self, dataframe_list):
        """Get list of all unique columns across all DataFrames"""
        unique_columns = set()
        for df in dataframe_list:
            unique_columns.update(df.columns)
        return list(unique_columns)
    
    def get_common_columns(self, dataframe_list):
        """Get list of columns common to all DataFrames"""
        if not dataframe_list:
            return []
        
        # Start with all columns from first DataFrame
        common_cols = set(dataframe_list[0].columns)
        
        # Find intersection with all other DataFrames
        for df in dataframe_list[1:]:
            common_cols = common_cols.intersection(set(df.columns))
            
        return list(common_cols)
    
    def get_summary(self, location: str = None, plant_type: str = None):
        """
        Get summary statistics for environmental data
        
        Args:
            location: Specific location (or None for all)
            plant_type: Specific plant type (or None for all)
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_locations': len(self.env_data),
            'total_datasets': sum(len(plants) for plants in self.env_data.values()),
            'locations': {},
        }
        
        # Get locations to process
        if location is not None:
            if location not in self.env_data:
                raise ValueError(f"Location {location} not found")
            locations = [location]
        else:
            locations = list(self.env_data.keys())
            
        # Process each location
        for loc in locations:
            summary['locations'][loc] = {'plant_types': {}}
            
            # Get plant types to process
            if plant_type is not None:
                if plant_type not in self.env_data[loc]:
                    raise ValueError(f"Plant type {plant_type} not found for location {loc}")
                plant_types = [plant_type]
            else:
                plant_types = list(self.env_data[loc].keys())
                
            # Process each plant type
            for pt in plant_types:
                data = self.env_data[loc][pt]
                
                # Get variables
                variables = [col for col in data.columns if col not in ['solar_TIMESTAMP', 'lat', 'long']]
                
                # Calculate statistics
                time_range = None
                if isinstance(data.index, pd.DatetimeIndex):
                    time_range = {
                        'start': data.index.min().strftime('%Y-%m-%d'),
                        'end': data.index.max().strftime('%Y-%m-%d'),
                        'duration_days': (data.index.max() - data.index.min()).days
                    }
                
                # Add to summary
                summary['locations'][loc]['plant_types'][pt] = {
                    'variables': len(variables),
                    'time_range': time_range,
                    'measurements': len(data),
                    'missing_data_pct': data[variables].isna().sum().sum() / (len(data) * len(variables)) * 100
                }
                
                # Add coordinates if available
                if 'lat' in data.columns and 'long' in data.columns:
                    coords = {
                        'lat': data['lat'].iloc[0] if not data['lat'].isna().all() else None,
                        'long': data['long'].iloc[0] if not data['long'].isna().all() else None
                    }
                    summary['locations'][loc]['plant_types'][pt]['coordinates'] = coords
                    
        return summary 