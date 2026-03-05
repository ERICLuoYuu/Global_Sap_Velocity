import gc
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
import pvlib
from typing import Union, List, Optional, Tuple, Callable
parent_dir = str(Path(__file__).parent.parent.parent)
from path_config import PathConfig, get_default_paths
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# from src.tools import adjust_time_to_utc, create_timezone_mapping

class EnvironmentalAnalyzer:
    """
    Analyzer for German (DEU) Environmental data with metadata handling
    """
    def __init__(self, paths: PathConfig = None, scale: str = 'sapwood'):
        self.paths = paths if paths is not None else PathConfig(scale=scale)
        self.data_dir = self.paths.sap_data_dir
        self.data_outlier_removed_dir = self.paths.sap_outliers_removed_dir
        self.env_raw_data = {}
        self.env_data = {}
        self.outlier_removed_data = {}

         
    # NEW: A method to control the batch processing
    def run_analysis_in_batches(self, batch_size: int = 20, switch = 'both'):
        """
        Orchestrates the analysis by processing files in batches to manage memory.
        """
        all_env_files = list(self.data_dir.glob("*_env_data.csv"))
        print(f"Found {len(all_env_files)} total files to process in batches of {batch_size}.")

        # Split the file list into chunks (batches)
        file_batches = [all_env_files[i:i + batch_size] for i in range(0, len(all_env_files), batch_size)]

        if switch == 'both':
            for i, batch in enumerate(file_batches):
                print("-" * 80)
                print(f"Processing Batch {i+1}/{len(file_batches)} (files {i*batch_size + 1} to {i*batch_size + len(batch)})")
                
                # 1. Load data for the current batch
                self._load_environmental_data(files_to_load=batch)

                self._standardize_all_data()
                # 2. Run plotting or other analysis on the loaded batch data
                #    The plot_all_plants function will now only plot the 20 files in memory.
                print("\nGenerating plots for the current batch...")
                self.plot_all(save_dir=self.paths.env_cleaned_figures_dir, progress_update=False)
                
                # 3. CRITICAL: Clear data from memory before the next batch
                self._clear_batch_data()
        elif switch == 'load':
            for i, batch in enumerate(file_batches):
                print("-" * 80)
                print(f"Loading Batch {i+1}/{len(file_batches)} (files {i*batch_size + 1} to {i*batch_size + len(batch)})")
                
                # 1. Load data for the current batch
                self._load_environmental_data(files_to_load=batch)
                self._standardize_all_data()
                # 2. CRITICAL: Clear data from memory before the next batch
                self._clear_batch_data()
        elif switch == 'plot':
            if not self.data_outlier_removed_dir.exists():
                print("No outlier removed data found. Please run the 'load' switch first to load data.")
                raise ValueError("No outlier removed data found. Please run the 'load' switch first to load data.")
            else:
                # Load outlier removed data from saved files
                all_sapf_files_outliersremoved = list(self.data_outlier_removed_dir.glob("*_outliers_removed.csv"))
                print("Plotting outlier removed data in batches...")
                # Ensure the outlier removed data is loaded
                file_batches_outliersremoved = [all_sapf_files_outliersremoved[i:i + batch_size] for i in range(0, len(all_sapf_files_outliersremoved), batch_size)]
                for i, batch in enumerate(file_batches_outliersremoved):
                    print("-" * 80)
                    print(f"Plotting Batch {i+1}/{len(file_batches_outliersremoved)} (files {i*batch_size + 1} to {i*batch_size + len(batch)})")
                    
                    # 1. Run plotting or other analysis on the loaded batch data
                    #    The plot_all_plants function will now only plot the 20 files in memory.
                    print("\nGenerating plots for the current batch...")
                    self.plot_all(save_dir=self.paths.env_cleaned_figures_dir, files_to_load=batch, progress_update=False)

        print("-" * 80)
        print("All batches processed successfully.")

    # NEW: A helper method to clear the dictionaries
    def _clear_batch_data(self):
        """
        Resets data dictionaries and calls the garbage collector to free memory.
        """
        print("\nClearing data from memory...")
        self.env_raw_data.clear()
        self.env_data.clear()
        self.outlier_removed_data.clear()
        gc.collect()
        print("Memory cleared.")
    def _load_environmental_data(self, files_to_load: List[Path]):
        """Load all German environmental data files and their corresponding metadata"""
        print(f"Loading {len(files_to_load)} env files for this batch...")
        column_mapping = {
                'TIMESTAMP_solar': 'solar_TIMESTAMP',  # Map alternate name to standard name
                }
        for file in files_to_load:
            parts = file.stem.split('_') 
            location = '_'.join(parts[:2])
            
            plant_type = '_'.join(parts[2:])
            try:
                # Load environmental data
                df = pd.read_csv(file, parse_dates=['TIMESTAMP'])
                df = df.rename(columns=column_mapping)
                if location not in self.env_raw_data:
                    self.env_raw_data[location] = {}
                self.env_raw_data[location][plant_type] = df
                # load corresponding evn_md data
                time_zone_file = file.parent / f"{file.stem.replace('env_data', 'env_md')}.csv"
                tz_df = pd.read_csv(time_zone_file)
                time_zone = tz_df['env_time_zone'].iloc[0][2:]
                # time_zone_map = create_timezone_mapping()
                # Apply the time zone adjustment to the entire column at once
                
                df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
                df['solar_TIMESTAMP'] = pd.to_datetime(df['solar_TIMESTAMP'])
                '''
                df['TIMESTAMP'] = df['TIMESTAMP'].apply(lambda x: adjust_time_to_utc(x, time_zone, time_zone_map))
                df['solar_TIMESTAMP'] = df['solar_TIMESTAMP'].apply(lambda x: adjust_time_to_utc(x, time_zone, time_zone_map))
                output_dir = Path(f'./outputs/processed_data/{SCALE}/env/timezone_adjusted')
                if not output_dir.exists():
                    output_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_dir / f'{file.stem}_adjusted.csv')
                '''
                # Load corresponding metadata file
                flag_file = file.parent / f"{file.stem.replace('_data', '_flags')}.csv"
                if flag_file.exists():
                    print(f"Loading flags: {flag_file}")
                    flags = pd.read_csv(flag_file, parse_dates=['TIMESTAMP'])
                    flags = flags.rename(columns=column_mapping)
                    flags['TIMESTAMP'] = pd.to_datetime(flags['TIMESTAMP'])
                    flags['solar_TIMESTAMP'] = pd.to_datetime(flags['solar_TIMESTAMP'])
                    '''
                    flags['TIMESTAMP'] = flags['TIMESTAMP'].apply(lambda x: adjust_time_to_utc(x, time_zone, time_zone_map))
                    flags['solar_TIMESTAMP'] = flags['solar_TIMESTAMP'].apply(lambda x: adjust_time_to_utc(x, time_zone, time_zone_map))
                    '''
                else:
                    print(f"Warning: No flag file found for {file.name}")
                    flags = None
                
                # Process data
                print(f"\nProcessing {file.name}")
                # get cooridnates information from "*_site_md.csv"
                site_md_file = file.parent / f"{file.stem.replace('_env_data', '_site_md')}.csv"
                env_md_file = file.parent / f"{file.stem.replace('_env_data', '_env_md')}.csv"
                # load the site_md_file
                site_md = pd.read_csv(site_md_file)
                env_md = pd.read_csv(env_md_file)
                # get the latitude and longitude from columns "si_lat" and "si_long"
                latitude = site_md['si_lat'].values[0]
                longitude = site_md['si_long'].values[0]
                elevation = site_md['si_elev'].values[0]
                timestep = env_md['env_timestep'].values[0]
                df = self._process_environmental_data(df, latitude, longitude, timestep, time_zone, elevation, location, plant_type, flags, )
                # add coordinates to the metadata
                df['lat'] = latitude
                df['long'] = longitude
                if location not in self.env_data:
                    self.env_data[location] = {}
                self.env_data[location][plant_type] = df
            
            except Exception as e:
                print(f"Error loading {file.name}: {str(e)}")
    
    def _process_environmental_data(self, df: pd.DataFrame, latitude: float, longitude: float, timestep: float, time_zone: str, elevation: float, location: str = None, plant_type: str = None, flags: pd.DataFrame = None) -> pd.DataFrame:
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
        print(flags.dtypes)
        print("\nSample of raw data:")
        print(df.head())
        # Standardize column names - handle different solar timestamp formats
        
        # Convert string "NA" to numpy NaN
        df = df.replace("NA", np.nan)

        # --- Define your radiation variables ---
        NONNEGATIVE_VARIABLES = ['ppfd_in', 'sw_in', 'precip', 'ws', 'rh', 'vpd', 'swc_deep', 'swc_shallow', 'ext_rad'] # Variables that should not have negative values
        DIURNAL_VARIABLES = ['ppfd_in', 'sw_in', 'ext_rad', 'netrad'] # Variables that should be diurnal
        RADIATION_VARIABLES = ['ppfd_in', 'sw_in', 'ext_rad', 'netrad'] # Variables that are radiation-related
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
            # Export filtered data immediately
            env_filtered_dir = self.paths.env_filtered_dir
            env_flagged_dir = self.paths.env_flagged_dir
            if not env_filtered_dir.exists():
                env_filtered_dir.mkdir(parents=True, exist_ok=True)
            if not env_flagged_dir.exists():
                env_flagged_dir.mkdir(parents=True, exist_ok=True)
            for col in data_cols:
                flag_col = [fcol for fcol in flags.columns if col in fcol]
                if flag_col:
                    warn_mask = flags[flag_col[0]].astype(str).str.contains('RANGE_WARN', na=False)
                    if col in  NONNEGATIVE_VARIABLES:
                        # get mask for radiation columns where values are less than 0, and exclude them from original mask
                        negative_radiation_mask = (df[col] < 0)
                        warn_mask = warn_mask & ~ negative_radiation_mask
                        # set negative values to 0
                        df.loc[df[col] < 0, col] = 0
                    
                    # Export flagged values to CSV
                    if warn_mask.any():
                        print(f"\nFiltering {warn_mask.sum()} values in {col} due to warnings")
                    df.loc[warn_mask, col].to_csv(env_flagged_dir / f"{location}_{plant_type}_{col}_flagged.csv")
                    df.loc[warn_mask, col] = np.nan


            filter_path = env_filtered_dir / f"{location}_{plant_type}_filtered.csv"
            df.to_csv(filter_path, index=True)
            print(f"Saved filtered data to {filter_path}")
        try: 
            # --- Default parameters for non-radiation variables ---
            # set one month time window
            # pre_timewindow = self._get_timewindow(30*24*60*60, df.index)
            pre_timewindow = self._get_timewindow(30*24*60*60, df.index) # 30 days in seconds
            DEFAULT_WINDOW_POINTS = pre_timewindow if pre_timewindow < len(df) else len(df)  # 30 days in seconds
            DEFAULT_N_STD = 4
            # --- Parameters for DAYTIME radiation ---
            DAY_WINDOW_POINTS = DEFAULT_WINDOW_POINTS
            print(f"Default window points: {DEFAULT_WINDOW_POINTS} and {DAY_WINDOW_POINTS} for timestep {timestep} minutes")
            DAY_N_STD = 4
            NIGHT_N_STD = 7 

           
            # Alternatively, use _detect_outliers with specific params:
            # NIGHT_RADIATION_WINDOW_POINTS = 3
            # NIGHT_RADIATION_N_STD = 3 
            if location not in self.outlier_removed_data:
                self.outlier_removed_data[location] = {}
            outlier_dir = self.paths.env_outliers_dir
            outlier_dir.mkdir(parents=True, exist_ok=True)
            day_mask_dir = self.paths.env_day_masks_dir
            day_mask_dir.mkdir(parents=True, exist_ok=True)
            outliers_removed_dir = self.paths.env_outliers_removed_dir
            outliers_removed_dir.mkdir(parents=True, exist_ok=True)
            current_df_latitude = latitude
            current_df_longitude = longitude

            data_cols = [col for col in df.columns if col not in ['solar_TIMESTAMP', 'lat', 'long'] and 'Unnamed:' not in str(col)]
            
            for col in data_cols:
                NIGHT_N_STD = 4 # Default for non-radiation variables
                if df[col].isna().all():
                    print(f"Skipping {col} at {location}_{plant_type} - all values are NA")
                    continue

                print(f"Processing {col} for outliers at {location}_{plant_type}...")
                column_series = df[col].copy() # Work on a copy for safety before modifying df
                final_col_outliers = pd.Series(False, index=column_series.index)

                if col in DIURNAL_VARIABLES:
                    if col in RADIATION_VARIABLES:
                        NIGHT_N_STD = 3 # For radiation variables, we can use a lower threshold for nighttime
                    print(f"Applying diurnal-specific outlier detection for {col}...")
                    try:
                        
                        day_mask = self._calculate_daytime_mask(df.index, current_df_latitude, current_df_longitude, elevation, elevation_threshold=0)
                        # Extract the boolean values. These correspond positionally to the solar_TIMESTAMP sequence.
                        day_mask_raw_values = day_mask.values
                        # Sanity check for length. After DataFrame processing (set_index on 'TIMESTAMP',
                        # sort, drop duplicates), df.index and df['solar_TIMESTAMP'] (as a column)
                        # should have the same length. pvlib's handling of NaT in solar_TIMESTAMP
                        # (typically resulting in NaN elevation, then False in mask) should maintain length.
                        if len(day_mask_raw_values) != len(df.index):
                            message = (
                                f"Length mismatch for {location}_{plant_type}, column {col}: "
                                f"Solar mask values length ({len(day_mask_raw_values)}) "
                                f"does not match df.index length ({len(df.index)}). "
                                "This may be due to NaT handling in solar_TIMESTAMP or pvlib. "
                                "Cannot reliably align day/night mask."
                            )
                            # This custom error will be caught by the broader except block below,
                            # leading to the global fallback.
                            raise ValueError(message)

                        # Create the final day_mask Series.
                        # Its *values* are determined by solar_TIMESTAMP.
                        # Its *index* is df.index (i.e., 'TIMESTAMP').
                        # This correctly aligns the solar-time-based day/night determination
                        # with the primary 'TIMESTAMP' index of the DataFrame.
                        
                        day_mask = pd.Series(day_mask_raw_values, index=df.index)
                        day_mask_with_extrad = {
                            'day_mask': day_mask,
                            'ext_rad': df['ext_rad'],
                            col: df[col],
                            'lat': current_df_latitude,
                            'long': current_df_longitude

                        }
                        # save it to csv
                        day_mask_df = pd.DataFrame(day_mask_with_extrad)
                        day_mask_path = day_mask_dir / f"{location}_{plant_type}_{col}_day_mask.csv"
                        day_mask_df.to_csv(day_mask_path, index=True)
                        print(f"  Day/night mask saved to {day_mask_path}")
                        night_mask = ~day_mask # Also indexed by df.index
                        
                    except Exception as e:
                        print(f"  Could not calculate day/night mask for {col} at {location}_{plant_type}: {e}. Processing column globally.")
                        # Fallback to default global processing if day/night calculation fails
                        final_col_outliers = self._detect_outliers(
                            series=column_series,
                            n_std=DEFAULT_N_STD, # Or your previous n_std=7 if preferred as fallback
                            time_window=len(column_series), # Fallback to global as before if this is intended
                            method='C'
                        )
                        df.loc[final_col_outliers, col] = np.nan
                        continue # Next column


                    day_series = column_series.copy()
                    night_series = column_series.copy()
                    # Apply the day/night mask to separate series
                    day_series.loc[night_mask] = np.nan
                    night_series.loc[day_mask] = np.nan

                    # Process Daytime Radiation
                    if not day_series.empty:
                        # print(f"  Processing daytime for {col} ({len(day_series)} points)...")
                        day_outliers_detected = self._detect_outliers(
                            series=day_series,
                            n_std=DAY_N_STD,
                            time_window=DAY_WINDOW_POINTS,
                            method='B' # Or 'B'
                        )
                        final_col_outliers.loc[day_outliers_detected[day_outliers_detected].index] = True

                    '''
                    # Process Nighttime Radiation
                    if not night_series.empty:
                        # print(f"  Processing nighttime for {col} ({len(night_series)} points)...")
                        night_outliers_detected = self._detect_outliers(
                            series=night_series,
                            n_std=NIGHT_N_STD, # Use same n_std as daytime
                            time_window=DAY_WINDOW_POINTS, # Use same window as daytime
                            method='B' # Or 'B'
                        )
                        final_col_outliers.loc[night_outliers_detected[night_outliers_detected].index] = True
                    '''

                else: # For non-diunal variables
                    NIGHT_N_STD = 7 # Default for non-radiation variables
                    # set 0 values to NaN if the variable is precip
                    if col == 'precip':
                        NIGHT_N_STD = 5 # For precipitation, we can use a higher threshold
                        column_series[column_series <= 0] = np.nan
                    final_col_outliers = self._detect_outliers(
                        series=column_series,
                        n_std=NIGHT_N_STD,
                        time_window=DEFAULT_WINDOW_POINTS, # A fixed, reasonable window
                        method='B' # Or 'B'
                    )
                
                # Save outlier information (using final_col_outliers)
                # This part needs the original values before they are set to NaN
                outlier_info_df = pd.DataFrame({
                    'timestamp': df.index, # Timestamps of outliers
                    'value': df.loc[:, col],    # Original values of outliers
                    'is_outlier': final_col_outliers                          # Redundant but can be explicit
                })
                
                if not outlier_info_df.empty:
                    outlier_path = outlier_dir / f"{location}_{plant_type}_{col}_outliers.csv" # Ensure plant_type is defined
                    outlier_info_df.to_csv(outlier_path, index=False)
                    # print(f"Saved {final_col_outliers.sum()} outliers for {col} to {outlier_path}")

                # Set outliers to NaN in the original DataFrame
                df.loc[final_col_outliers, col] = np.nan
                # print(f"Applied {final_col_outliers.sum()} NaNs for {col} at {location}_{plant_type}")

            self.outlier_removed_data[location][plant_type] = df
            outliers_removed_path = outliers_removed_dir / f"{location}_{plant_type}_outliers_removed.csv"
            #save outliers removed df
            df.to_csv(outliers_removed_path)
            # print(f"Finished outlier processing for {location}_{plant_type}")

        except Exception as e:
            print(f"Critical error during outlier processing for {location}_{plant_type}: {str(e)}")
            # return None # Or handle as appropriate
        
        try:
            # Create processed directory
            env_daily_resampled_dir = self.paths.env_daily_resampled_dir
            if not env_daily_resampled_dir.exists():
                env_daily_resampled_dir.mkdir(parents=True, exist_ok=True)

            # Get columns to resample (exclude solar_TIMESTAMP)
            data_cols = [col for col in df.columns if col != 'solar_TIMESTAMP' and col != 'lat' and col != 'long']
            
            # Get columns that need sum vs mean
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
            output_path = env_daily_resampled_dir / f"daily_{location}_{plant_type}.csv"
            daily_df.to_csv(output_path, index=False)
            print(f"Saved daily resampled data to {output_path}")

            return daily_df

        except Exception as e:
            print(f"Error during daily resampling: {str(e)}")
            return None
    def _get_timewindow(self, time_window: int, timesteps_data: Union[pd.Index, pd.Series, pd.DataFrame]) -> int:
        """
        Calculate the number of points in a time window based on the determined timestep.

        Parameters:
        -----------
        time_window : int
            Time window in seconds.
        timesteps_data : Union[pd.Index, pd.Series, pd.DataFrame]
            A pandas Index, Series, or DataFrame containing 'TIMESTAMP' information.
            The method will infer the timestep from this data.

        Returns:
        --------
        int : Number of points in the time window.
        """

        # Determine the timestamp Index from the input data
        if isinstance(timesteps_data, pd.DataFrame):
            if 'TIMESTAMP' in timesteps_data.columns:
                timesteps = timesteps_data['TIMESTAMP']
            elif timesteps_data.index.name == 'TIMESTAMP':
                timesteps = timesteps_data.index
            else:
                raise ValueError("DataFrame must have a 'TIMESTAMP' column or its index named 'TIMESTAMP' to determine the timestep.")
        elif isinstance(timesteps_data, pd.Series):
            if timesteps_data.index.name == 'TIMESTAMP':
                timesteps = timesteps_data.index
            else:
                # If Series itself is datetime series, use it directly, otherwise try its index
                if pd.api.types.is_datetime64_any_dtype(timesteps_data):
                    timesteps = timesteps_data
                else:
                    raise ValueError("Series must have a 'TIMESTAMP' index or contain datetime values directly to determine the timestep.")
        elif isinstance(timesteps_data, pd.Index):
            timesteps = timesteps_data
        else:
            raise TypeError("Input 'timesteps_data' must be a pandas Index, Series, or DataFrame with a 'TIMESTAMP' column or index.")

        # Ensure the extracted timesteps are datetime objects
        if not pd.api.types.is_datetime64_any_dtype(timesteps):
            try:
                timesteps = pd.to_datetime(timesteps)
            except Exception as e:
                raise ValueError(f"Could not convert 'timesteps_data' to datetime objects: {e}")

        if len(timesteps) < 2:
            raise ValueError("Timestamp data must contain at least two points to calculate the difference and determine the timestep.")

        # Calculate the median timestep in minutes
        # Using .to_series().diff() handles both Series and Index objects
        timestep_seconds = timesteps.to_series().diff().median().total_seconds()

        if timestep_seconds <= 0:
            raise ValueError(f"Calculated timestep is non-positive ({timestep_seconds} minutes). "
                             "This can happen if timestamps are identical or not in increasing order.")

        return int(time_window / timestep_seconds)
    def _calculate_daytime_mask(self, timestamps: pd.DatetimeIndex, lat: float, lon: float, elevation: float, elevation_threshold: float = 0) -> pd.Series:
        """
        Calculates a boolean mask indicating daytime based on solar elevation.

        Parameters:
        -----------
        timestamps : pd.DatetimeIndex
            Timestamps for which to calculate daytime.
        lat : float
            Latitude of the location.
        lon : float
            Longitude of the location.
        elevation_threshold : float, optional
            Solar elevation threshold (in degrees) to consider it daytime. 
            Defaults to 0 (sun above the horizon).

        Returns:
        --------
        pd.Series : Boolean mask, True if daytime.
        """
        if not isinstance(timestamps, pd.DatetimeIndex):
            raise ValueError("Timestamps must be a pandas DatetimeIndex.")
        if timestamps.tzinfo is None:
            # pvlib requires timezone-aware timestamps. Assuming UTC if not specified.
            # You might need to adjust this based on your data's timezone.
            print("Warning: Timestamps are timezone-naive. Assuming UTC for solar position calculation.")
            timestamps = timestamps.tz_localize('UTC')

            
        solar_position = pvlib.solarposition.get_solarposition(timestamps, lat, lon, altitude=elevation)
        return solar_position['elevation'] > elevation_threshold
    def adaptive_centered_rolling(self, series, window_size, func=np.mean):
        """
        Apply rolling function with adaptive centered windows.
        Near edges, borrows points from available side to maintain window size.
        
        Parameters:
        -----------
        series : pd.Series
            Input time series
        window_size : int
            Size of rolling window
        func : callable
            Function to apply (np.mean, np.std, np.median, etc.)
            
        Returns:
        --------
        pd.Series : Result with same index as input
        """
        if window_size >= len(series):
            # If window is larger than series, use entire series for all points
            return pd.Series([func(series)] * len(series), index=series.index)
        
        result = pd.Series(index=series.index, dtype=float)
        half_window = window_size // 2
        
        for i in range(len(series)):
            # Calculate ideal centered window bounds
            ideal_start = i - half_window
            ideal_end = i + half_window + 1  # +1 for inclusive end
            
            # Adjust bounds to stay within series limits
            actual_start = max(0, ideal_start)
            actual_end = min(len(series), ideal_end)
            
            # If window is smaller than desired, extend to the available side
            current_window_size = actual_end - actual_start
            if current_window_size < window_size:
                # Try to extend to the left
                if actual_start > 0:
                    extend_left = min(actual_start, window_size - current_window_size)
                    actual_start -= extend_left
                    current_window_size = actual_end - actual_start
                
                # Try to extend to the right
                if current_window_size < window_size and actual_end < len(series):
                    extend_right = min(len(series) - actual_end, window_size - current_window_size)
                    actual_end += extend_right
            
            # Apply function to the window
            window_data = series.iloc[actual_start:actual_end]
            result.iloc[i] = func(window_data)
        
        return result
    def _detect_outliers(self, series: pd.Series, n_std: float = 3, time_window: int = 1440, method: str = 'B') -> pd.Series:
        """
        Improved outlier detection with better edge handling
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
        n_std : float
            Number of standard deviations for outlier threshold
        time_window : int
            Rolling window size
        method : str
            'A': Monthly grouping
            'B': Adaptive centered rolling (standard deviation)
            'C': Adaptive centered rolling (MAD-based)
            'D': Standard pandas rolling (for comparison)
                    
        Returns:
        --------
        pd.Series : Boolean mask where True indicates outliers
        """
        outliers = pd.Series(False, index=series.index)
        
        if method == 'A':
            # Original monthly method (unchanged)
            grouped = series.groupby([series.index.year, series.index.month])
            
            for (year, month), group in grouped:
                if len(group) > 0:
                    mean = group.mean()
                    std = group.std()
                    
                    month_mask = (series.index.year == year) & (series.index.month == month)
                    monthly_data = series[month_mask]
                    
                    monthly_outliers = (np.abs(monthly_data - mean) > n_std * std)
                    outliers[monthly_data.index] = monthly_outliers
                    
                    n_outliers = monthly_outliers.sum()
                    print(f"Month {month}/{year}: "
                        f"{n_outliers} outliers out of {len(monthly_data)} points "
                        f"({(n_outliers/len(monthly_data)*100):.1f}%)")
        
        elif method == 'B':
            # Improved rolling with adaptive centering
            print(f"Using adaptive centered rolling (window={time_window})")
            
            # Calculate adaptive rolling statistics
            rolling_mean = self.adaptive_centered_rolling(series, time_window, np.mean)
            rolling_std = self.adaptive_centered_rolling(series, time_window, np.std)
            
            # Handle potential zero std (constant values in window)
            rolling_std = rolling_std.replace(0, rolling_std.mean())
            
            # Detect outliers
            outliers = np.abs(series - rolling_mean) > (n_std * rolling_std)
            
            print(f"Adaptive rolling outliers detected: {outliers.sum()} out of {len(series)} points")
            print(f"No NaN values in rolling statistics")
        
        elif method == 'C':
            # Improved MAD with adaptive centering
            print(f"Using adaptive centered MAD (window={time_window})")
            
            def rolling_mad(window_data):
                """Calculate MAD for a window"""
                if len(window_data) <= 1:
                    return 0
                median_val = np.median(window_data)
                return np.median(np.abs(window_data - median_val))
            
            # Calculate adaptive rolling statistics
            rolling_median = self.adaptive_centered_rolling(series, time_window, np.median)
            rolling_mad_values = self.adaptive_centered_rolling(series, time_window, rolling_mad)
            
            # Scale MAD to be comparable to standard deviation
            scaled_mad = rolling_mad_values / 0.6745
            
            # Add small epsilon to prevent division by zero
            epsilon = 1e-10
            scaled_mad = scaled_mad + epsilon
            
            # Calculate deviations and detect outliers
            deviations = np.abs(series - rolling_median)
            outliers = deviations > (n_std * scaled_mad)
            
            print(f"Adaptive MAD outliers detected: {outliers.sum()} out of {len(series)} points")
            print(f"No NaN values in rolling statistics")
        
        elif method == 'D':
            # Standard pandas rolling (for comparison)
            print(f"Using standard pandas rolling (window={time_window})")
            
            rolling_mean = series.rolling(
                window=time_window,
                center=True,
                min_periods=max(1, time_window//2)  # More conservative min_periods
            ).mean()
            
            rolling_std = series.rolling(
                window=time_window,
                center=True,
                min_periods=max(1, time_window//2)
            ).std()
            
            # Handle NaN values more carefully
            rolling_mean = rolling_mean.fillna(series.mean())
            rolling_std = rolling_std.fillna(series.std())
            
            # Detect outliers
            outliers = np.abs(series - rolling_mean) > (n_std * rolling_std)
            
            nan_count = series.rolling(time_window, center=True).mean().isna().sum()
            print(f"Standard rolling outliers detected: {outliers.sum()} out of {len(series)} points")
            print(f"Original NaN values in rolling: {nan_count}")
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return outliers
    '''
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
                min_periods=int(time_window)
            ).mean()
            
            rolling_std = series.rolling(
                window=time_window,
                center=True,
                min_periods=int(time_window)
            ).std()
            
            # Handle edge cases with forward/backward fill
            # Updated code using preferred methods
            rolling_mean = rolling_mean.ffill().bfill()
            rolling_std = rolling_std.ffill().bfill()
            
            # Detect outliers
            outliers = np.abs(series - rolling_mean) > (n_std * rolling_std)
            
            print(f"Outliers detected: {outliers.sum()} out of {len(series)} points")
        elif method == 'C':
            
            # 1. Calculate the rolling median
            rolling_median = series.rolling(
                window=time_window,
                center=True,
                min_periods=int(time_window)
            ).median()

            # Fill NaNs at the edges of the series
            rolling_median = rolling_median.ffill().bfill()

            # 2. Calculate the absolute deviation from the rolling median
            deviation = abs(series - rolling_median)

            # 3. Calculate the rolling MAD of the deviation
            rolling_mad = deviation.rolling(
                window=time_window,
                center=True,
                min_periods=int(time_window)
            ).median()

            # Fill NaNs for the MAD series as well
            rolling_mad = rolling_mad.ffill().bfill()

            # 4. (Recommended) Scale the MAD to be comparable to standard deviation
            # The scaling factor 1.4826 makes the MAD an unbiased estimator of the
            # standard deviation for normal data.
            scaled_mad = rolling_mad / 0.6745

            # Add a small epsilon to prevent division by zero or issues with zero MAD
            # This prevents flagging points as outliers when the signal is constant.
            epsilon = 1e-10

            # 5. Identify outliers
            # Outliers are points where the deviation is greater than the threshold
            # multiplied by the scaled MAD.
            outliers = deviation > (n_std * scaled_mad + epsilon)
            """
            # Process each window
            for start_idx in range(0, len(series), time_window):
                end_idx = min(start_idx + time_window, len(series))
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
            """
            
            print(f"MAD-based outliers detected: {outliers.sum()} out of {len(series)} points")
        return outliers
    '''
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

    def plot_environmental_variables(self, location: str, plant_type: str, 
                              figsize=(12, 6), save_dir: str = None):
        """
        Create individual plots for each environmental variable with flags and outliers
        
        Args:
            location (str): Location identifier
            plant_type (str): Plant type identifier
            figsize (tuple): Figure size
            save_dir (str): Directory to save plots
        """
        if location not in self.outlier_removed_data or plant_type not in self.outlier_removed_data[location]:
            raise ValueError(f"No data found for {location}_{plant_type}")
        
        data = self.outlier_removed_data[location][plant_type].copy()
        plot_columns = [col for col in data.columns if col != 'solar_TIMESTAMP' and col != 'TIMESTAMP' and len(col) > 0 and col not in ['lat', 'long'] and 'Unnamed:'not in str(col)]
        
        print(f"\nPlotting environmental data for {location}_{plant_type}")
        print(f"Data shape: {data.shape}")
        print(f"Variables to plot: {plot_columns}")
        
        if save_dir:
            save_path = Path(save_dir) / f"{location}_{plant_type}_env"
            save_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving plots to {save_path}")
        
        
        
        for variable in plot_columns:
            # Load flagged data
            flag_path = self.paths.env_flagged_dir / f"{location}_{plant_type}_{variable}_flagged.csv"
            if flag_path.exists():
                flagged_data = pd.read_csv(flag_path, parse_dates=['TIMESTAMP']).set_index('TIMESTAMP')
            else:
                flagged_data = None
                print("No flagged data found")
            # Print column statistics
            print(f"\nVariable: {variable}")
            print(f"Non-null values: {data[variable].count()}")
            print(f"Value range: {data[variable].min()} to {data[variable].max()}")
            
            # Skip if all values are NaN
            if data[variable].isna().all():
                print(f"Skipping {variable} - all values are NA")
                continue
            
            # Load outliers data
            outlier_path = self.paths.env_outliers_dir / f"{location}_{plant_type}_{variable}_outliers.csv"
            if outlier_path.exists():
                outliers_df = pd.read_csv(outlier_path, parse_dates=['timestamp']).set_index('timestamp')
            else:
                outliers_df = None
                print(f"No outliers data found for {variable}")
            
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
                if flagged_data is not None and not flagged_data.dropna().empty:
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
                flagged_count =     (
                                    (~flagged_data[variable].isna() &  # Not NaN/None
                                    (flagged_data[variable].astype(str).str.strip() != '') &  # Not empty string
                                    pd.to_numeric(flagged_data[variable], errors='coerce').notna()  # Valid number
                                    ).sum() if flagged_data is not None else 0 )
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
                    filename = f"{variable}.png"
                    fig.savefig(save_path / filename, bbox_inches='tight', dpi=300)
                    plt.close(fig)
                    print(f"Saved {filename}")
                else:
                    plt.show()
            else:
                plt.close(fig)
                print(f"No valid data to plot for {variable}")
    
    def plot_all(self, files_to_load: Optional[List[Path]] = None, figsize=(12, 6), save_dir: str = None, skip_empty: bool = True, 
                 plot_limit: int = None, progress_update: bool = True):
        """
        Plot all environmental variables for all sites with enhanced error handling and progress tracking
        
        Args:
            figsize (tuple): Figure size for plots
            save_dir (str): Directory to save plots
            skip_empty (bool): Skip locations with no valid data
            plot_limit (int): Maximum number of variables to plot per location
            progress_update (bool): Print progress updates
        """
        # release memory by empty some dictionaries
        self.env_raw_data = {}
        self.env_data = {}
        if not self.outlier_removed_data:
           
            print("Loading outlier removed data from saved files...")
            for file in files_to_load:
                parts = file.stem.split('_')
                location = '_'.join(parts[:2])
                plant_type = '_'.join(parts[2:-2])
                df = pd.read_csv(file, parse_dates=['TIMESTAMP'])
                df = df.set_index('TIMESTAMP')
                if location not in self.outlier_removed_data:
                    self.outlier_removed_data[location] = {}
                self.outlier_removed_data[location][plant_type] = df
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
                    plot_columns = [col for col in data.columns if col != 'solar_TIMESTAMP' and col != 'TIMESTAMP' and len(col) > 0 and col not in ['lat', 'long'] and 'Unnamed:'not in str(col)]
                    
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
                        print(f"Plotting {len(plot_columns)} variables")
                    
                    # Create location-specific save directory
                    if save_dir:
                        location_save_dir = save_path / f"{location}_{plant_type}"
                        location_save_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        location_save_dir = None
                    
                    # Plot environmental variables
                    self.plot_environmental_variables(
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
    def plot_histogram(self, save_dir: str = None):
        """
        Plot histogram of env data
        """
        for location in self.env_raw_data:
            for plant_type in self.env_raw_data[location]:
                df = self.env_raw_data[location][plant_type]
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
                            ax.set_xlabel(f'{column}')
                            ax.set_ylabel('Frequency')
                            
                            if save_dir:
                                save_path = Path(save_dir)
                                save_path.mkdir(parents=True, exist_ok=True)
                                plt.savefig(save_path / f"{location}_{plant_type}_{column}_histogram.png")
                                print(f"Saved histogram to {save_path / f'{location}_{plant_type}_{column}_histogram.png'}")
                            plt.close()  # Close the figure to free memory
                        else:
                            print(f"No valid data for {location}_{plant_type}_{column}")
                        
                    except Exception as e:
                        print(f"Error plotting histogram for {column}: {str(e)}")
                        continue

    def _standardize_all_data(self):
        """Standardize environmental data using only common columns across all datasets with error handling"""
        try:
            print("\nStarting data standardization process...")
            
            # Verify data is loaded
            if not self.env_data:
                raise ValueError("No environmental data has been loaded")
            
            # Create processed directory for standardized data
            try:
                std_dir = self.paths.env_standardized_dir
                if not std_dir.exists():
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
            
            # Find all columns
            try:
                all_columns = self.get_all_columns(all_dataframes)
                all_columns = [col for col in all_columns if col != 'solar_TIMESTAMP' and col != 'lat' and col != 'long']
                
                if not all_columns:
                    raise ValueError("No columns found across datasets")
                    
                print(f"\nFound {len(all_columns)} columns to standardize: {all_columns}")
                
            except Exception as e:
                raise ValueError(f"Error identifying all columns: {str(e)}")
            
            # Calculate global min/max for common columns
            self._minmax_cache = {}
            for variable in all_columns:
                try:
                    temp_data = []
                    for location in self.env_data:
                        for plant_type in self.env_data[location]:
                            data = self.env_data[location][plant_type]
                            if variable not in data.columns:
                                print(f"{variable} not in {location}{plant_type}")
                                continue
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
            
            # Standardize columns for each dataset
            standardization_results = []
            for location in self.env_data:
                for plant_type in self.env_data[location]:
                    try:
                        print(f"\nStandardizing data for {location}_{plant_type}")
                        data = self.env_data[location][plant_type].copy()
                        
                        # Track successful standardizations
                        standardized_cols = []
                        
                        for variable in data.columns:
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