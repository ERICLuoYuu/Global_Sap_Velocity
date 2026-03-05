from pathlib import Path
import sys
import os
import logging

# Set environment variables for determinism BEFORE importing TensorFlow
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED'] = '42'

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the randomization control module first
from src.utils.random_control import (
    set_seed, get_seed, deterministic,
)

# Set the master seed at the very beginning
set_seed(42)

# Now import all other dependencies
import tensorflow as tf
import numpy as np
import pandas as pd


class TimeSeriesSegmenter:
    """
    Utility class for segmenting time series data with gaps.
    """
    
    @staticmethod
    @deterministic
    def segment_time_series(
        df: pd.DataFrame, 
        gap_threshold: float = 2, 
        unit: str = 'hours', 
        min_segment_length: int = None,
        timestamp_column: str = None
    ):
        """
        Segments a time series into continuous blocks based on time gaps.
        
        Parameters:
        ----------
        df : pandas.DataFrame
            Time series data with datetime index or timestamp column
        gap_threshold : float
            Size of gap that defines a new segment
        unit : str
            Time unit for gap_threshold ('seconds', 'minutes', 'hours', 'days')
        min_segment_length : int, optional
            Minimum number of points required for a valid segment
        timestamp_column : str, optional
            The name of the timestamp column. If None, assumes the DataFrame is indexed by timestamp
            
        Returns:
        -------
        list of pandas.DataFrame
            List of continuous segments
        """
        # Use timestamp column or index
        if timestamp_column is not None:
            if timestamp_column not in df.columns:
                raise ValueError(f"Timestamp column '{timestamp_column}' not found in DataFrame")
            time_values = pd.to_datetime(df[timestamp_column])
            df_indexed = df.set_index(time_values)
        else:
            # Check if index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df_indexed = df.set_index(pd.to_datetime(df.index))
                except:
                    raise ValueError("DataFrame index cannot be converted to datetime")
            else:
                df_indexed = df
        
        # Calculate time differences in the specified unit
        if unit == 'seconds':
            time_diffs = df_indexed.index.to_series().diff().dt.total_seconds()
        elif unit == 'minutes':
            time_diffs = df_indexed.index.to_series().diff().dt.total_seconds() / 60
        elif unit == 'hours':
            time_diffs = df_indexed.index.to_series().diff().dt.total_seconds() / 3600
        elif unit == 'days':
            time_diffs = df_indexed.index.to_series().diff().dt.total_seconds() / 86400
        else:
            raise ValueError(f"Unsupported time unit: {unit}")
        
        # Find indices where gaps exceed threshold
        gap_indices = np.where(time_diffs > gap_threshold)[0]
        
        # Create segment boundaries
        boundaries = [0] + list(gap_indices) + [len(df)]
        
        # Extract segments
        segments = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            # Use original DataFrame to maintain all columns
            segment = df.iloc[start:end].copy()
            
            # Only include segments that meet minimum length requirement
            if min_segment_length is None or len(segment) >= min_segment_length:
                segments.append(segment)
        
        return segments


class WindowGenerator:
    """
    Generates windowed datasets from time series for machine learning models.
    Handles input windows and label (target) windows with configurable offsets.
    """
    
    def __init__(
        self, 
        input_width: int, 
        label_width: int, 
        shift: int,
        data_df: pd.DataFrame,
        label_columns=None,
        batch_size: int = 32,
        shuffle: bool = True,
        exclude_targets_from_features: bool = True,
        exclude_labels_from_inputs: bool = True
    ):
        """
        Initialize the window generator with the specified parameters.
        """
        # Store the raw data
        self.data_df = data_df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.exclude_targets_from_features = exclude_targets_from_features

        # Work out the label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                         enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(data_df.columns)}

        # Work out the window parameters
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift
        if exclude_labels_from_inputs:
            self.input_slice = slice(0, input_width)
            self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        else:
            # If labels are included in inputs, input slice includes the entire window
            self.input_slice = slice(0, self.total_window_size)
            self.input_indices = np.arange(self.total_window_size)[self.input_slice]
            self.input_width = self.total_window_size  # Adjust input width to total window size

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
        # Set example as None initially
        self._example = None

    def __repr__(self):
        """String representation of the WindowGenerator."""
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'
        ])
    
    def split_window(self, features):
        """
        Split a window of features into inputs and labels.
        Excludes 'source_file' column from model inputs.
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        
        if self.label_columns is not None:
            # Extract the label columns
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
                
            # Remove label columns and metadata columns from inputs
            if hasattr(self, 'exclude_targets_from_features') and self.exclude_targets_from_features:
                # Create a list of indices for columns to keep
                feature_indices = []
                for i, name in enumerate(self.data_df.columns):
                    # Skip label columns and source_file column
                    if name not in self.label_columns and name != 'source_file':
                        feature_indices.append(i)
                
                # Apply the filtering to keep only desired feature columns
                if feature_indices:  # Only if we have features to keep
                    inputs = tf.gather(inputs, feature_indices, axis=2)
                else:
                    logging.warning("Warning: No feature columns remain after excluding labels and metadata.")

        # Slicing doesn't preserve static shape information, so set the shapes manually
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None if self.label_columns is None else len(self.label_columns)])

        return inputs, labels
    
    def make_dataset(self, data=None):
        """
        Create a windowed tf.data.Dataset from a pandas DataFrame.
        """
        if data is None:
            data = self.data_df
        # Exclude source_file column before conversion to NumPy array
        if 'source_file' in data.columns:
            data = data.drop(columns=['source_file'])
            data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            seed=get_seed())  # Use deterministic seed

        ds = ds.map(self.split_window)
        
        # Apply deterministic options to dataset in a version-compatible way
        try:
            options = tf.data.Options()
            options.experimental_deterministic = True
            # Try setting auto shard policy if available
            try:
                options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
            except:
                pass
            ds = ds.with_options(options)
        except:
            # If options aren't available, rely on seed for determinism
            pass
        
        return ds
    
    @property
    def dataset(self):
        """Get the windowed dataset."""
        return self.make_dataset()

    @property
    def example(self):
        """
        Get and cache an example batch of `inputs, labels` for plotting.
        """
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the dataset
            result = next(iter(self.dataset))
            # And cache it for next time
            self._example = result
        return result


class TemporalWindowGenerator:
    """
    Creates windows from segmented time series data and splits them in temporal order
    by processing each timeseries (CSV file) independently.
    """
    
    @staticmethod
    @deterministic
    def create_windowed_dataset_from_segments(
        segments, 
        input_width, 
        label_width, 
        shift,
        label_columns=None,
        batch_size=32,
        min_segment_length=None,
        exclude_targets_from_features=True,
        train_val_test_split=(0.7, 0.15, 0.15)
    ):
        """
        Creates windowed datasets from segmented time series by processing each timeseries
        separately. Windows from the same timeseries are split temporally into train/val/test.
        
        Parameters:
        ----------
        segments : list of pandas.DataFrame
            List of time series segments, each segment should have a 'source_file' column
        input_width : int
            Width of input window
        label_width : int
            Width of label window
        shift : int
            Offset between input and label
        label_columns : list of str, optional
            Columns to use as labels
        batch_size : int
            Batch size for datasets
        min_segment_length : int, optional
            Minimum length required for a segment to be used
        exclude_targets_from_features : bool
            Whether to exclude label columns from inputs
        train_val_test_split : tuple
            Proportions for train, validation, and test splits (e.g., (0.7, 0.15, 0.15))
            
        Returns:
        -------
        dict
            Dictionary containing train, validation, and test datasets
        """
        # Validate split ratio to ensure it sums to 1.0
        if abs(sum(train_val_test_split) - 1.0) > 1e-10:
            raise ValueError(f"Split ratios must sum to 1.0, got {train_val_test_split} with sum {sum(train_val_test_split)}")
            
        # Calculate minimum required length to create at least one complete window
        window_size = input_width + shift
        
        # Set min_segment_length to window_size if not provided or too small
        if min_segment_length is None or min_segment_length < window_size:
            min_segment_length = window_size

        logging.info(f"Minimum segment length required: {min_segment_length} (window size: {window_size})")
        logging.info(f"Using train/val/test split ratio: {train_val_test_split}")

        # Group segments by timeseries (source file)
        timeseries_segments = {}
        
        for segment in segments:
            # Get the source file from the column
            if 'source_file' in segment.columns:
                source_file = segment['source_file'].iloc[0]  # Get first value
            else:
                # Use segment index as a fallback identifier
                source_file = f"unknown_source_{len(timeseries_segments)}"
            
            # Add to the appropriate group
            if source_file not in timeseries_segments:
                timeseries_segments[source_file] = []
            timeseries_segments[source_file].append(segment)

        logging.info(f"Grouped {len(segments)} segments into {len(timeseries_segments)} distinct timeseries")

        # Track processed timeseries to prevent duplicates
        processed_timeseries = set()
        
        # Process each timeseries individually
        all_train_windows = []
        all_val_windows = []
        all_test_windows = []
        
        for source_file, source_segments in list(timeseries_segments.items()):
            # Skip if already processed (prevent duplicates)
            if source_file in processed_timeseries:
                logging.warning(f"Skipping duplicate timeseries {source_file}")
                continue
                
            processed_timeseries.add(source_file)
            logging.info(f"Processing timeseries: {source_file} with {len(source_segments)} segments")

            # Collect all windows from all segments of this timeseries
            timeseries_windows = []
            
            for segment in source_segments:
                # Skip segments that are too short
                if len(segment) < min_segment_length:
                    continue
                
                # Create a WindowGenerator for this segment
                window_gen = WindowGenerator(
                    input_width=input_width,
                    label_width=label_width,
                    shift=shift,
                    data_df=segment,
                    label_columns=label_columns,
                    batch_size=1,  # Use batch size of 1 to extract individual windows
                    shuffle=False,  # Don't shuffle to maintain temporal order
                    exclude_targets_from_features=exclude_targets_from_features
                )
                
                # Collect all windows from this segment
                segment_ds = window_gen.dataset
                for inputs, labels in segment_ds:
                    timeseries_windows.append((inputs, labels))
            
            # Skip if no windows were created for this timeseries
            if not timeseries_windows:
                logging.warning(f"No valid windows created for timeseries {source_file}")
                continue
            
            # Split windows for this timeseries temporally
            n_windows = len(timeseries_windows)
            train_end = int(n_windows * train_val_test_split[0])
            val_end = train_end + int(n_windows * train_val_test_split[1])
            
            # Add to the respective collections
            ts_train_windows = timeseries_windows[:train_end]
            ts_val_windows = timeseries_windows[train_end:val_end]
            ts_test_windows = timeseries_windows[val_end:]
            
            all_train_windows.extend(ts_train_windows)
            all_val_windows.extend(ts_val_windows)
            all_test_windows.extend(ts_test_windows)

            logging.info(f"  Windows from {source_file}: {len(ts_train_windows)} train, {len(ts_val_windows)} val, {len(ts_test_windows)} test")
        # Check if we have enough windows for each split
        if not all_train_windows or not all_val_windows or not all_test_windows:
            raise ValueError("Not enough valid windows for at least one of the splits. Consider adjusting split ratios or window size.")
        
        # Convert window collections to TensorFlow datasets
        train_ds = TemporalWindowGenerator._windows_to_dataset(all_train_windows)
        val_ds = TemporalWindowGenerator._windows_to_dataset(all_val_windows)
        test_ds = TemporalWindowGenerator._windows_to_dataset(all_test_windows)
        
        # Apply batching and shuffling
        buffer_size = 1000
        train_ds = train_ds.shuffle(buffer_size, seed=get_seed()).batch(batch_size)
        val_ds = val_ds.batch(batch_size)
        test_ds = test_ds.batch(batch_size)
        
        # Print final dataset sizes
        logging.info(f"Train windows: {len(all_train_windows)} ({train_val_test_split[0]*100:.1f}%)")
        logging.info(f"Validation windows: {len(all_val_windows)} ({train_val_test_split[1]*100:.1f}%)")
        logging.info(f"Test windows: {len(all_test_windows)} ({train_val_test_split[2]*100:.1f}%)")

        return {
            'train': train_ds,
            'val': val_ds,
            'test': test_ds
        }
    
    @staticmethod
    def _windows_to_dataset(windows):
        """Helper method to convert windows to a TensorFlow dataset."""
        # Early exit if empty
        if not windows:
            return None
            
        # Extract inputs and labels from complete windows
        inputs = tf.concat([w[0] for w in windows], axis=0)
        labels = tf.concat([w[1] for w in windows], axis=0)
        
        # Create dataset with complete windows
        ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
        
        # Apply deterministic options in a version-compatible way
        try:
            options = tf.data.Options()
            options.experimental_deterministic = True
            try:
                options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
            except:
                pass
            ds = ds.with_options(options)
        except:
            pass
            
        return ds