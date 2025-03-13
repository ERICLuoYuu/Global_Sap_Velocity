from pathlib import Path
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
import plotly.graph_objects as go
import plotly.io as pio
import copy
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


def _process_histogram_column(args):
    """
    Process a single histogram column - this must be at module level to be picklable
    
    Args:
        args: Tuple containing (file_path, column_name, output_path)
    
    Returns:
        Status message string
    """
    file_path, column_name, output_path = args
    
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
        
        # Extract location and plant type from filename
        file_stem = Path(file_path).stem
        parts = file_stem.split('_')
        location_plant = '_'.join(parts)
        
        
        ax.set_title(f'Histogram of {location_plant}_{column_name}\n'
                   f'Mean: {series.mean():.2f}, Std: {series.std():.2f}\n'
                   f'Valid points: {len(series)}')
        ax.set_xlabel('Sap Flow')
        ax.set_ylabel('Frequency')
        
        if output_path:
            out_dir = Path(output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_dir / f"{column_name}_histogram.png")
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



class SapFlowAnalyzer:
    """
    Analyzer for Global SapFlow data with flag handling and parallel processing
    """
    def __init__(self, base_dir: str = "./data/raw/0.1.5/0.1.5/csv/sapwood", max_workers: int = None):
        self.data_dir = Path(base_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Directory not found: {self.data_dir}")
        self.sapf_raw_data = {}
        self.sapflow_data = {}
        self.outlier_removed_data = {}
        # Set default max_workers to number of CPU cores
        self.max_workers = max_workers if max_workers is not None else max(1, multiprocessing.cpu_count() - 1)
        print(f"Using {self.max_workers} workers for parallel processing")
        print(f"Using {self.max_workers} workers for parallel processing")
        self._load_sapflow_data()
    
    
    def _load_sapflow_data(self):
        """Load all sapflow data files in parallel and their corresponding flags"""
        try:
            sapf_files = list(self.data_dir.glob("*_sapf_data.csv"))
            print(f"Found {len(sapf_files)} sapflow files")
            
            # For debugging, you can limit to a few files
            # Uncomment this to test with fewer files
            # sapf_files = sapf_files[:5]  # Process only first 3 files
            # list of file chunks
           
            # Setup the multiprocessing parameters
            if self.max_workers > multiprocessing.cpu_count():
                print(f"Warning: Requested {self.max_workers} workers but only {multiprocessing.cpu_count()} CPUs available")
                self.max_workers = max(1, multiprocessing.cpu_count() - 1)
                
            print(f"Using {self.max_workers} workers for parallel processing")
            
            # Process files in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Use tqdm to show progress bar
                results = list(tqdm(
                    executor.map(self._process_file, sapf_files),
                    total=len(sapf_files),
                    desc="Loading sapflow files"
                ))
                
            # Count successful and failed files
            successful = 0
            failed = 0
            
            # Process results and update data dictionaries
            for location, plant_type, daily_df, outlier_removed_df in results:
                if location is not None and plant_type is not None and daily_df is not None and outlier_removed_df is not None:
                    # Update raw data dictionary
                    if location not in self.sapf_raw_data:
                        self.sapf_raw_data[location] = {}
                    
                    # Find the corresponding raw file to get raw data
                    for file in sapf_files:
                        if location in file.stem and plant_type in file.stem:
                            df = pd.read_csv(file, parse_dates=['TIMESTAMP'])
                            self.sapf_raw_data[location][plant_type] = df
                            break
                    
                    # Update processed data dictionaries
                    if location not in self.sapflow_data:
                        self.sapflow_data[location] = {}
                    self.sapflow_data[location][plant_type] = daily_df
                    
                    # Update outlier_removed_data dictionary
                    if location not in self.outlier_removed_data:
                        self.outlier_removed_data[location] = {}
                    self.outlier_removed_data[location][plant_type] = outlier_removed_df
                    
                    # Debug print
                    print(f"Successfully processed {location}_{plant_type}")
                    successful += 1
                else:
                    failed += 1
                    
            print(f"\nProcessing summary: {successful} successful, {failed} failed")
            print(f"self.sapflow_data has {len(self.sapflow_data)} locations")
            print(f"self.outlier_removed_data has {len(self.outlier_removed_data)} locations")
            
        except Exception as e:
            import traceback
            print(f"Unhandled error in _load_sapflow_data:")
            print(traceback.format_exc())

    def _process_sapflow_data_parallel(self, data_cols: List[str], df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
        """
        Process flag-based filtering in parallel
        
        Args:
            data_cols: List of data columns to process
            df: DataFrame to filter
            flags: DataFrame containing flags
            
        Returns:
            Filtered DataFrame
        """
        output_dir = Path('./outputs/processed_data/sap/filtered')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        def process_column(col):
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
        
        # Process columns in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_column, data_cols))
        
        # Apply filter masks
        df_copy = df.copy()
        for col, mask in results:
            if mask is not None:
                df_copy.loc[mask, col] = np.nan
        
        return df_copy

    def _process_sapflow_data(self, location: str, plant_type: str, df: pd.DataFrame, flags: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process sapflow data with parallel flag handling and outlier detection
        
        Returns:
            Tuple of (daily_df, outlier_removed_df)
        """
        try:
            print(f"\nProcessing {location}_{plant_type}")
            print(f"Data shape: {df.shape}")
            print("Data types before processing:")
            print(df.dtypes)
            print("\nSample of raw data (first 3 rows):")
            print(df.head(3))
            
            # Convert string "NA" to numpy NaN
            df = df.replace("NA", np.nan)
            
            # Convert numeric columns to float, with error handling
            numeric_cols = [col for col in df.columns if col not in ['TIMESTAMP', 'solar_TIMESTAMP']]
            print(f"Found {len(numeric_cols)} numeric columns to process")
            
            # Process numeric columns in parallel
            def process_numeric_column(col):
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
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(process_numeric_column, numeric_cols))
            
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
                return None, None
                
            print(f"Found {len(valid_columns)} valid data columns with non-null values")

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
            # count the number of unique timestamps
            utiimestamp_count = len(df.index.unique())
            time_stamp_count = len(df.index)

            if dup_count > 0:
                
                print(f"Found {dup_count} duplicate timestamps")
                # log the number of duplicates and save as a .txt, even if the file is not existent
                with open(f"./outputs/processed_data/sap/duplicates/{location}_{plant_type}_duplicates.txt", "w") as f:
                    f.write(f"Found {dup_count} duplicate timestamps and there are {time_stamp_count} timestamps and {utiimestamp_count} unique timestamps\n")



                
                # Export duplicates for analysis
                dup_dir = Path(f"./outputs/processed_data/sap/duplicates")
                dup_dir.mkdir(parents=True, exist_ok=True)
                
                # Export duplicate rows
                df[dup_mask].to_csv(dup_dir / f"{location}_{plant_type}_duplicates.csv")
                
                print(f"Removing {dup_count} duplicate timestamps ({dup_count/len(df)*100:.1f}%)")
                df = df[~dup_mask]  # Remove duplicates
            else:
                print("No duplicates found in timestamps")
            
            # Apply flag-based filtering in parallel
            if flags is not None:
                try:
                    flags = flags.set_index('TIMESTAMP').sort_index()
                    if flags.index.duplicated().any():
                        flags = flags[~flags.index.duplicated(keep='first')]
                    
                    data_cols = [col for col in df.columns if col != 'solar_TIMESTAMP']
                    
                    # Process flag-based filtering in parallel
                    df = self._process_sapflow_data_parallel(data_cols, df, flags)
                    
                    # Save filtered data
                    filter_path = Path('./outputs/processed_data/sap/filtered') / f"{location}_{plant_type}_filtered.csv"
                    df.to_csv(filter_path)
                    print(f"Saved filtered data to {filter_path}")
                except Exception as e:
                    print(f"Error during flag processing: {str(e)}")
                    # Continue without flag filtering
            
            # Create outlier removed dataframe
            outlier_removed_df = None
            try:
                # Create processed directory for outliers
                outlier_dir = Path('./outputs/processed_data/sap/outliers/MAD')
                outlier_dir.mkdir(parents=True, exist_ok=True)
                outlier_removed_dir = Path('./outputs/processed_data/sap/outlier_removed')
                outlier_removed_dir.mkdir(parents=True, exist_ok=True)
                
                # Process each column for outliers in parallel
                data_cols = [col for col in df.columns if col != 'solar_TIMESTAMP' and 'Unnamed:' not in str(col)]
                
                def process_outliers(col):
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
                        
                        # Detect outliers using method C (MAD-based)
                        outliers = self._detect_outliers(
                            series=df[col],
                            n_std=7,
                            time_window=len(df[col]),  # Cap window size
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
                        
                        return col, outliers
                    except Exception as e:
                        import traceback
                        print(f"Error processing outliers for column {col}:")
                        print(traceback.format_exc())
                        return col, None
                
                # Process outliers in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    outlier_results = list(executor.map(process_outliers, data_cols))
                
                # Apply outlier masks
                df_copy = df.copy()
                outlier_columns_processed = 0
                
                for col, outliers in outlier_results:
                    if outliers is not None:
                        df_copy.loc[outliers, col] = np.nan
                        outlier_columns_processed += 1
                
                print(f"Applied outlier detection to {outlier_columns_processed} columns")
                outlier_removed_df = df_copy
                # Save outlier removed data
                output_path = outlier_removed_dir / f"{location}_{plant_type}_outlier_removed.csv"
                outlier_removed_df.to_csv(output_path)
                print(f"Saved outlier removed data to {output_path}")

            except Exception as e:
                import traceback
                print(f"Error in outlier processing:")
                print(traceback.format_exc())
                # Continue with original data if outlier processing fails
                outlier_removed_df = df.copy()
            
            # Create daily dataframe
            daily_df = None
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
                daily_df.to_csv(output_path)
                print(f"Saved daily resampled data to {output_path}")

            except Exception as e:
                import traceback
                print(f"Error during daily resampling:")
                print(traceback.format_exc())
                return None, None
            
            print(f"Successfully processed {location}_{plant_type}")
            return daily_df, outlier_removed_df

        except Exception as e:
            import traceback
            print(f"Unhandled error in _process_sapflow_data for {location}_{plant_type}:")
            print(traceback.format_exc())
            return None, None
    def _process_file(self, file: Path) -> Tuple[str, str, pd.DataFrame, pd.DataFrame]:
        """
        Process a single sapflow file with its corresponding flags
        
        Args:
            file: Path to sapflow data file
                
        Returns:
            Tuple of (location, plant_type, daily_df, outlier_removed_df)
        """
        try:
            # Extract file information
            parts = file.stem.split('_')
            
            # Better handling of file name parsing
            if 'sapf' not in file.stem:
                return None, None, None, None
                
            # Find the index where 'sapf' appears
            
            location = '_'.join(parts[:2])  # First two components typically form the location
            plant_type = '_'.join(parts[2:])
            
            
            print(f"\nProcessing file: {file}")
            print(f"Extracted location: {location}, plant_type: {plant_type}")
            
            column_mapping = {
                'TIMESTAMP_solar': 'solar_TIMESTAMP',  # Map alternate name to standard name
            }
            
            # Load sap flow data with more robust error handling
            try:
                df = pd.read_csv(file, parse_dates=['TIMESTAMP'])
                print(f"Successfully loaded {file.name}, shape: {df.shape}")
            except Exception as e:
                print(f"Failed to load {file.name}: {str(e)}")
                return None, None, None, None
            
            # Check for required columns
            required_cols = ['TIMESTAMP']
            has_required_cols = all(col in df.columns or col in column_mapping for col in required_cols)
            
            if not has_required_cols:
                print(f"Missing required columns in {file.name}. Available columns: {list(df.columns)}")
                # Try to adapt if only one timestamp column exists
                if 'TIMESTAMP' in df.columns and 'solar_TIMESTAMP' not in df.columns:
                    print("Only TIMESTAMP column found, adding solar_TIMESTAMP as a copy")
                    df['solar_TIMESTAMP'] = df['TIMESTAMP']
                elif 'solar_TIMESTAMP' in df.columns and 'TIMESTAMP' not in df.columns:
                    print("Only solar_TIMESTAMP column found, adding TIMESTAMP as a copy")
                    df['TIMESTAMP'] = df['solar_TIMESTAMP']
                else:
                    print("Cannot adapt missing time columns")
                    return None, None, None, None
            
            # Rename columns as needed
            df = df.rename(columns=column_mapping)
            # drop solar_TIMESTAMP column if it exists
            if 'solar_TIMESTAMP' in df.columns:
                df = df.drop(columns=['solar_TIMESTAMP'])
            
            # Try to locate corresponding env_md data
            time_zone_file = file.parent / f"{file.stem.replace('sapf_data', 'env_md')}.csv"
            
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
            time_zone_map = create_timezone_mapping()
            
            # Make timestamps into datetime objects with better error handling
            for col in ['TIMESTAMP']:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    null_timestamps = df[col].isna().sum()
                    if null_timestamps > 0:
                        print(f"Warning: {null_timestamps} null values in {col} after conversion")
                except Exception as e:
                    print(f"Error converting {col} to datetime: {str(e)}")
                    return None, None, None, None
             
            # Apply time zone adjustment with better error handling
            try:
                df['TIMESTAMP'] = df['TIMESTAMP'].apply(
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
            output_dir = Path('./outputs/processed_data/sap/timezone_adjusted')
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            try:
                df.to_csv(output_dir / f'{file.stem}_adjusted.csv')
                print(f"Saved timezone adjusted data to {output_dir / f'{file.stem}_adjusted.csv'}")
            except Exception as e:
                print(f"Error saving timezone adjusted data: {str(e)}")
                # Continue anyway as this is not critical
            
            # Find and load flags file
            flags = None
            flag_file = file.parent / f"{file.stem.replace('_data', '_flags')}.csv"
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
            
            # Process data with flags
            try:
                daily_df, outlier_removed_df = self._process_sapflow_data(location, plant_type, df, flags)
                if daily_df is None or outlier_removed_df is None:
                    print(f"Processing returned None for {location}_{plant_type}")
                    return None, None, None, None
                    
                return location, plant_type, daily_df, outlier_removed_df
                
            except Exception as e:
                import traceback
                print(f"Error in _process_sapflow_data for {file.name}:")
                print(traceback.format_exc())
                return None, None, None, None
                
        except Exception as e:
            import traceback
            print(f"Unhandled error processing {file.name}:")
            print(traceback.format_exc())
            return None, None, None, None
    def _detect_outliers_parallel(self, series: pd.Series, time_windows: List[Tuple[int, int]], 
                                 n_std: float = 7, method: str = 'C') -> pd.Series:
        """
        Detect outliers in parallel for different time windows
        
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
        
        # Process windows in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_window, time_windows))
        
        # Combine results
        for start_idx, end_idx, window_outliers in results:
            if window_outliers is not None:
                outliers.iloc[start_idx:end_idx] = window_outliers
        
        return outliers
    
    def _detect_outliers(self, series: pd.Series, n_std: float = 3, time_window: int = 1440, method: str = 'C') -> pd.Series:
        """
        Detect outliers using different methods with parallel processing for method C
        
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
            # Apply MAD-based spike detection using time windows
            window_size = min(time_window, len(series))  # Make sure window size is not larger than series length
            
            # Create list of time windows
            time_windows = []
            for start_idx in range(0, len(series), window_size):
                end_idx = min(start_idx + window_size, len(series))
                time_windows.append((start_idx, end_idx))
            
            # Process windows in parallel
            outliers = self._detect_outliers_parallel(
                series=series,
                time_windows=time_windows,
                n_std=n_std,
                method=method
            )
            
            print(f"MAD-based outliers detected: {outliers.sum()} out of {len(series)} points")
            
        return outliers

    def plot_plant_parallel(self, plot_data):
        """
        Plot a single plant in parallel
        
        Args:
            plot_data: Tuple containing (location, plant_type, column, figsize, save_dir)
        """
        location, plant_type, column, figsize, save_dir = plot_data
        
        try:
            
            data = self.outlier_removed_data[location][plant_type].copy()
            
            # Load flagged data
            flag_path = Path('./outputs/processed_data/sap/filtered') / f"{column}_flaged.csv"
            if flag_path.exists():
                flagged_data = pd.read_csv(flag_path, parse_dates=['TIMESTAMP']).set_index('TIMESTAMP')
            else:
                flagged_data = None
            
            # Skip if all values are NaN
            if data[column].isna().all():
                print(f"Skipping {column} - all values are NA")
                return None
            
            # Load outliers data
            outlier_path = Path('./outputs/processed_data/sap/outliers/MAD') / f"{column}_outliers.csv"
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
                    (~flagged_data[column].isna() if flagged_data is not None else 0) &
                    (flagged_data[column].astype(str).str.strip() != '' if flagged_data is not None else 0) &
                    (pd.to_numeric(flagged_data[column], errors='coerce').notna() if flagged_data is not None else 0)
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
                
                plt.tight_layout()
                
                if save_dir:
                    filename = f"tree_{column}.png"
                    fig.savefig(save_dir / filename, bbox_inches='tight', dpi=300)
                    plt.close(fig)
                    return f"Saved {filename}"
                else:
                    plt.close(fig)
                    return "Plot completed (not saved)"
            else:
                plt.close(fig)
                return f"No valid data to plot for {column}"
                
        except Exception as e:
            return f"Error plotting {column}: {str(e)}"
    
    def plot_individual_plants_parallel(self, location: str, plant_type: str, 
                             figsize=(12, 6), save_dir: str = None):
        """
        Create individual plots for each plant/tree in parallel
        
        Args:
            location (str): Location identifier
            plant_type (str): Plant type identifier
            figsize (tuple): Figure size
            save_dir (str): Directory to save plots
        """
        if location not in self.sapflow_data or plant_type not in self.sapflow_data[location]:
            raise ValueError(f"No data found for {location}_{plant_type}")
        
        data = self.outlier_removed_data[location][plant_type].copy()
        plot_columns = [col for col in data.columns if col != 'solar_TIMESTAMP' and col != 'TIMESTAMP' and len(col) > 0 
                        and col not in ['lat', 'long'] and 'Unnamed:' not in str(col)]
        
        print(f"\nPlotting {location}_{plant_type}")
        print(f"Data shape: {data.shape}")
        print(f"Columns to plot: {len(plot_columns)}")
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving plots to {save_path}")
        
        # Prepare plot data
        plot_data = [(location, plant_type, column, figsize, save_path) for column in plot_columns]
        
        # Plot in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(tqdm(
                executor.map(self.plot_plant_parallel, plot_data),
                total=len(plot_data),
                desc=f"Plotting {location}_{plant_type}"
            ))
        
        for result in results:
            if result:
                print(result)
    
    def plot_all_plants_parallel(self, figsize=(12, 6), save_dir: str = None, skip_empty: bool = True, 
                       plot_limit: int = None, progress_update: bool = True):
        """
        Plot individual trees for all sites in parallel
        
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
        print(len(self.outlier_removed_data), self.outlier_removed_data)
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
                        print(f"\nAdding {location}_{plant_type} to plot queue")
                        print(f"Will plot {len(plot_columns)} trees")
                    
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
        
        # Plot locations in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(self.max_workers, len(plot_tasks))) as executor:
            future_to_loc = {
                executor.submit(self.plot_individual_plants, loc, pt, fs, sd): (loc, pt) 
                for loc, pt, fs, sd in plot_tasks
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_loc), 
                              total=len(future_to_loc), 
                              desc="Processing locations"):
                location, plant_type = future_to_loc[future]
                try:
                    # Wait for result
                    future.result()
                    summary['processed_locations'] += 1
                    
                    # Count successful plots
                    data = self.outlier_removed_data[location][plant_type]
                    plot_columns = [col for col in data.columns if col != 'solar_TIMESTAMP']
                    if plot_limit:
                        plot_columns = plot_columns[:plot_limit]
                    summary['successful_plots'] += len(plot_columns)
                    
                except Exception as e:
                    error_msg = f"Error processing {location}_{plant_type}: {str(e)}"
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
    
   
    def plot_histogram_parallel(self, save_dir: str = None, max_workers=2, chunk_size=10):
        """
        Plot histograms with memory-efficient parallel processing
        
        Args:
            save_dir: Directory to save histograms
            max_workers: Maximum number of parallel workers (use small number)
            chunk_size: Number of columns to process in each parallel batch
        """
        import gc  # Import garbage collection
        from pathlib import Path
        import concurrent.futures
        from tqdm import tqdm
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        # Find all sapflow data files directly
        sapf_files = list(self.data_dir.glob("*_sapf_data.csv"))
        print(f"Found {len(sapf_files)} sapflow data files")
        
        # Process each file directly instead of using the sapf_raw_data dictionary
        for file_path in sapf_files:
            # Extract location and plant type from filename
            parts = file_path.stem.split('_')
            if len(parts) < 4:  # Need at least location, plant type, and sapf_data
                print(f"Skipping file with unexpected name format: {file_path.name}")
                continue
                
            # Extract location and plant type 
            # Typical format: COUNTRY_SITE_SPECIES_sapf_data.csv or similar
            location = '_'.join(parts[:2])  # First two components typically form the location
            
            # Everything between location and sapf_data is plant type
            # Find the index where 'sapf' appears
            try:
                sapf_idx = next(i for i, part in enumerate(parts) if 'sapf' in part)
                plant_type = '_'.join(parts[2:sapf_idx])
            except StopIteration:
                # If 'sapf' not found, use default approach
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
                process_args = [(str(file_path), column, str(location_save_dir) if save_dir else None) 
                            for column in columns_chunk]
                
                # Process this chunk in parallel
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    results = list(tqdm(
                        executor.map(_process_histogram_column, process_args),
                        total=len(process_args),
                        desc=f"Processing chunk {chunk_idx+1}"
                    ))
                
                # Force garbage collection between chunks
                gc.collect()
                
                # Print results
                for result in results:
                    print(result)
                
                print(f"Completed chunk {chunk_idx+1}/{len(column_chunks)}")
    
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