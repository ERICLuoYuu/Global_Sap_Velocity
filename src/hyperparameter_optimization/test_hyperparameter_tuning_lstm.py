"""
Deterministic LSTM model for time series prediction with comprehensive
randomness control for reproducible results.
"""
import os
import sys
from pathlib import Path

# Set environment variables for determinism BEFORE importing TensorFlow
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED'] = '42'

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import frameworks with global seed control
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
from tensorflow import keras
from typing import List, Optional, Union, Dict, Tuple, Any

# Set global seeds
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Apply additional TensorFlow determinism settings - version compatible
try:
    # For TensorFlow 2.8+
    tf.config.experimental.enable_op_determinism()
except AttributeError:
    # For older TensorFlow versions
    # Set environment variables and session config instead
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    # Try to set operation determinism at session level if available
    try:
        from tensorflow.compat.v1 import ConfigProto
        from tensorflow.compat.v1 import InteractiveSession
        config = ConfigProto()
        config.intra_op_parallelism_threads = 1
        config.inter_op_parallelism_threads = 1
        session = InteractiveSession(config=config)
    except:
        print("Warning: Could not configure session-level determinism")

# Limit TensorFlow to use only one thread for CPU operations
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Configure GPU for determinism if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except:
            print(f"Warning: Could not set memory growth for GPU device {device}")

# Import the hyperparameter optimizer
from src.hyperparameter_optimization.hyper_tuner import DLOptimizer


class TimeSeriesSegmenter:
    """
    Utility class for segmenting time series data with gaps.
    """
    
    @staticmethod
    def segment_time_series(
        df: pd.DataFrame, 
        gap_threshold: float = 2, 
        unit: str = 'hours', 
        min_segment_length: Optional[int] = None,
        timestamp_column: Optional[str] = None
    ) -> List[pd.DataFrame]:
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
    
    @staticmethod
    def segment_irregular_time_series(
        df: pd.DataFrame, 
        relative_gap_threshold: float = 5,
        timestamp_column: Optional[str] = None,
        min_segment_length: Optional[int] = None
    ) -> List[pd.DataFrame]:
        """
        Segments an irregular time series based on relative gaps.
        
        Parameters:
        ----------
        df : pandas.DataFrame
            Time series data with datetime index or timestamp column
        relative_gap_threshold : float
            A gap is identified when the time difference is X times larger
            than the median time difference
        timestamp_column : str, optional
            The name of the timestamp column. If None, assumes the DataFrame is indexed by timestamp
        min_segment_length : int, optional
            Minimum number of points required for a valid segment
            
        Returns:
        -------
        list of pandas.DataFrame
            List of continuous segments
        """
        # Handle timestamp column or index
        if timestamp_column is not None:
            if timestamp_column not in df.columns:
                raise ValueError(f"Timestamp column '{timestamp_column}' not found in DataFrame")
            time_values = pd.to_datetime(df[timestamp_column])
        else:
            # Check if index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    time_values = pd.to_datetime(df.index)
                except:
                    raise ValueError("DataFrame index cannot be converted to datetime")
            else:
                time_values = df.index
        
        # Calculate time differences
        time_diffs = pd.Series(time_values).diff().dt.total_seconds()
        
        # Calculate median time difference as a reference
        median_diff = time_diffs.median()
        
        # Identify gaps as points where the time difference is significantly larger than median
        gap_indices = np.where(time_diffs > median_diff * relative_gap_threshold)[0]
        
        # Create segment boundaries
        boundaries = [0] + list(gap_indices) + [len(df)]
        
        # Extract segments
        segments = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            
            segment = df.iloc[start:end].copy()
            
            # Only include segments that meet minimum length requirement
            if min_segment_length is None or len(segment) >= min_segment_length:
                segments.append(segment)
        
        return segments
    
    @staticmethod
    def plot_segments(
        segments: List[pd.DataFrame], 
        value_column: str,
        timestamp_column: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Visualize segmented time series data.
        
        Parameters:
        ----------
        segments : list of pandas.DataFrame
            List of time series segments
        value_column : str
            Column name containing the values to plot
        timestamp_column : str, optional
            Column name for timestamps. If None, uses DataFrame index
        figsize : tuple of int
            Figure size
        """
        plt.figure(figsize=figsize)
        # Use deterministic colors and fixed order
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, segment in enumerate(segments):
            color = colors[i % len(colors)]
            
            if timestamp_column is not None:
                x = segment[timestamp_column]
            else:
                x = segment.index
                
            plt.plot(x, segment[value_column], 'o-', 
                     color=color, label=f'Segment {i+1}')
        
        plt.xlabel('Time')
        plt.ylabel(value_column)
        plt.title('Time Series Segmented by Gaps')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


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
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        label_columns: Optional[List[str]] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        exclude_labels_from_inputs: bool = True
    ):
        """
        Initialize the window generator with the specified parameters.
        
        Parameters:
        ----------
        input_width : int
            Width of the input window (number of time steps)
        label_width : int
            Width of the label window (number of time steps)
        shift : int
            Offset between the end of the input window and the end of the label window
        train_df : pandas.DataFrame
            Training data
        val_df : pandas.DataFrame
            Validation data
        test_df : pandas.DataFrame
            Test data
        label_columns : list of str, optional
            Which columns to use as labels. If None, all columns are used as features and labels
        batch_size : int
            Batch size for datasets
        shuffle : bool
            Whether to shuffle the training dataset
        """
        # Store the raw data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.exclude_labels_from_inputs = exclude_labels_from_inputs

        # Work out the label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                         enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
        # Set example as None initially
        self._example = None

    def __repr__(self) -> str:
        """String representation of the WindowGenerator."""
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'
        ])
    
    def split_window(self, features: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Split a window of features into inputs and labels.
        Ensures that label columns are not included in input features.
        
        Parameters:
        ----------
        features : tf.Tensor
            Window of features
            
        Returns:
        -------
        tuple of tf.Tensor
            (inputs, labels)
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        
        if self.label_columns is not None:
            # Extract the label columns
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
                
            # Remove label columns from inputs if requested
            # Create a mask to select non-label columns
            if hasattr(self, 'exclude_labels_from_inputs') and self.exclude_labels_from_inputs:
                feature_indices = [i for i, name in enumerate(self.train_df.columns) 
                                if name not in self.label_columns]
                inputs = tf.gather(inputs, feature_indices, axis=2)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None if self.label_columns is None else len(self.label_columns)])

        return inputs, labels
    
    def make_dataset(self, data: pd.DataFrame) -> tf.data.Dataset:
        """
        Create a windowed tf.data.Dataset from a pandas DataFrame.
        
        Parameters:
        ----------
        data : pandas.DataFrame
            Data to create windows from
            
        Returns:
        -------
        tf.data.Dataset
            Dataset of (input_window, label_window) pairs
        """
        data = np.array(data, dtype=np.float32)
        
        # Use deterministic seed for dataset creation
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            seed=SEED)  # Add seed for determinism

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
    def train(self) -> tf.data.Dataset:
        """Get the training dataset."""
        return self.make_dataset(self.train_df)

    @property
    def val(self) -> tf.data.Dataset:
        """Get the validation dataset."""
        return self.make_dataset(self.val_df)

    @property
    def test(self) -> tf.data.Dataset:
        """Get the test dataset."""
        return self.make_dataset(self.test_df)

    @property
    def example(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get and cache an example batch of `inputs, labels` for plotting.
        """
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result
    
    def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
        """
        Plot the input and label windows with optional model predictions.
        
        Parameters:
        ----------
        model : tf.keras.Model, optional
            Model to generate predictions, if provided
        plot_col : str
            Column to plot
        max_subplots : int
            Maximum number of examples to plot
        """
        # Set matplotlib to deterministic mode
        plt.rcParams['agg.path.chunksize'] = 10000
        
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                       edgecolors='k', label='Labels', c='#2ca02c', s=64)
                       
            if model is not None:
                # Use deterministic prediction
                try:
                    tf.config.run_functions_eagerly(True)
                    predictions = model(inputs)
                    tf.config.run_functions_eagerly(False)
                except:
                    predictions = model(inputs)
                
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                          marker='X', edgecolors='k', label='Predictions',
                          c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        plt.tight_layout()


class SegmentedWindowGenerator:
    """
    Handles creating and combining WindowGenerator objects from segmented time series data.
    """
    
    @staticmethod
    def create_window_generators_from_segments(
        segments: List[pd.DataFrame], 
        input_width: int, 
        label_width: int, 
        shift: int,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        label_columns: Optional[List[str]] = None,
        batch_size: int = 32,
        min_segment_length: Optional[int] = None,
        exclude_labels_from_inputs: bool = True
    ) -> List[WindowGenerator]:
        """
        Creates WindowGenerator objects for each valid segment.
        
        Parameters:
        ----------
        segments : list of pandas.DataFrame
            List of continuous time series segments
        input_width : int
            Width of the input window
        label_width : int
            Width of the label window
        shift : int
            Offset between input and label windows
        train_val_test_split : tuple of float
            Proportions for train, validation, and test splits
        label_columns : list of str, optional
            Columns to use as labels
        batch_size : int
            Batch size for datasets
        min_segment_length : int, optional
            Minimum length required for a segment to be used
            
        Returns:
        -------
        list of WindowGenerator
            A WindowGenerator for each segment that's long enough
        """
        window_generators = []
        min_required_length = input_width + shift
        
        # Calculate minimum samples needed for each split to create valid windows
        # Each split needs at least the window size (input_width + shift)
        min_train_samples = min_required_length
        min_val_samples = min_required_length
        min_test_samples = min_required_length
        
        # Calculate total minimum segment length based on split ratios
        # This ensures each split will have enough samples
        min_total_length = max(
            min_required_length,
            int(min_train_samples / train_val_test_split[0]),
            int(min_val_samples / train_val_test_split[1]),
            int(min_test_samples / train_val_test_split[2])
        )
        
        # Ensure min_segment_length is at least min_total_length
        if min_segment_length is None or min_segment_length < min_total_length:
            min_segment_length = min_total_length
        
        print(f"Minimum segment length required: {min_segment_length}")
        
        for segment in segments:
            # Skip segments that are too short
            if len(segment) < min_segment_length:
                # print(f"Skipping segment with length {len(segment)} < {min_segment_length}")
                continue
                
            # Split the segment into train, validation, and test sets
            segment_len = len(segment)
            train_len = int(segment_len * train_val_test_split[0])
            val_len = int(segment_len * train_val_test_split[1])
            
            # Verify splits have minimum required samples
            train_df = segment.iloc[:train_len]
            val_df = segment.iloc[train_len:train_len+val_len]
            test_df = segment.iloc[train_len+val_len:]
            
            # Double-check that each split has minimum required length
            if len(train_df) < min_train_samples or len(val_df) < min_val_samples or len(test_df) < min_test_samples:
                # print(f"Skipping segment: splits too small - train:{len(train_df)}, val:{len(val_df)}, test:{len(test_df)}")
                continue
            
            # Create a WindowGenerator for this segment
            window_gen = WindowGenerator(
                input_width=input_width,
                label_width=label_width,
                shift=shift,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                label_columns=label_columns,
                batch_size=batch_size,
                exclude_labels_from_inputs=exclude_labels_from_inputs
            )
            
            window_generators.append(window_gen)
        
        return window_generators

    @staticmethod
    def combine_datasets(
        window_generators: List[WindowGenerator],
        buffer_size: int = 1000
    ) -> Dict[str, tf.data.Dataset]:
        """
        Combines datasets from multiple WindowGenerator objects.
        
        Parameters:
        ----------
        window_generators : list of WindowGenerator
            List of window generators for different segments
        buffer_size : int
            Size of the shuffle buffer for the training dataset
            
        Returns:
        -------
        dict of tf.data.Dataset
            Combined train, val, and test datasets
        """
        if not window_generators:
            raise ValueError("No valid window generators found. All segments might be too small for the chosen parameters.")
        
        # Initialize with first window generator's datasets
        combined = {
            'train': window_generators[0].train,
            'val': window_generators[0].val,
            'test': window_generators[0].test
        }
        
        # Add datasets from remaining window generators
        for wg in window_generators[1:]:
            combined['train'] = combined['train'].concatenate(wg.train)
            combined['val'] = combined['val'].concatenate(wg.val)
            combined['test'] = combined['test'].concatenate(wg.test)
        
        # Set deterministic options for each dataset in a version-compatible way
        try:
            options = tf.data.Options()
            options.experimental_deterministic = True
            # Try setting auto shard policy if available
            try:
                options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
            except:
                pass
                
            # Apply options if available
            combined['train'] = combined['train'].shuffle(buffer_size, seed=SEED).with_options(options)
            combined['val'] = combined['val'].with_options(options)
            combined['test'] = combined['test'].with_options(options)
        except:
            # If options aren't available, at least use seed for shuffle
            combined['train'] = combined['train'].shuffle(buffer_size, seed=SEED)
            # And continue with datasets as they are
        
        return combined
    
    @staticmethod
    def create_complete_dataset_from_segments(
        segments: List[pd.DataFrame],
        input_width: int,
        label_width: int,
        shift: int,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        label_columns: Optional[List[str]] = None,
        batch_size: int = 32,
        buffer_size: int = 1000,
        exclude_labels_from_inputs: bool = True
    ) -> Dict[str, tf.data.Dataset]:
        """
        Creates a complete set of datasets from segmented time series.
        Combines the creation of window generators and merging of datasets.
        
        Parameters:
        ----------
        segments : list of pandas.DataFrame
            List of continuous time series segments
        input_width : int
            Width of the input window
        label_width : int
            Width of the label window
        shift : int
            Offset between input and label windows
        train_val_test_split : tuple of float
            Proportions for train, validation, and test splits
        label_columns : list of str, optional
            Columns to use as labels
        batch_size : int
            Batch size for datasets
        buffer_size : int
            Size of the shuffle buffer for the training dataset
            
        Returns:
        -------
        dict of tf.data.Dataset
            Combined train, val, and test datasets
        """
        # Create window generators for each segment
        window_generators = SegmentedWindowGenerator.create_window_generators_from_segments(
            segments=segments,
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            train_val_test_split=train_val_test_split,
            label_columns=label_columns,
            batch_size=batch_size,
            exclude_labels_from_inputs=exclude_labels_from_inputs
        )
        
        # Combine the datasets
        return SegmentedWindowGenerator.combine_datasets(
            window_generators=window_generators,
            buffer_size=buffer_size
        )


def create_lstm_model(input_shape, output_shape, n_layers, units, dropout_rate):
    """
    Create an improved LSTM model with regularization and bidirectional layers.
    
    Parameters:
    -----------
    input_shape: tuple, shape of input data (timesteps, features)
    output_shape: int, number of output values
    n_layers: int, number of LSTM layers
    units: int, number of LSTM units per layer
    dropout_rate: float, dropout rate for regularization
    
    Returns:
    --------
    model: keras.Model, compiled LSTM model
    """
    # Create a deterministic model with seeds for all random operations
    tf.random.set_seed(SEED)
    
    # Create incrementing seeds for each layer to ensure independence
    kernel_seeds = [SEED + i for i in range(100)]
    seed_idx = 0
    
    model = keras.Sequential()
    
    # Input layer
    model.add(keras.layers.Input(shape=input_shape))
    
    # Add a time-distributed attention mechanism (optional)
    if n_layers > 1:
        # First layer can be bidirectional with deterministic seed
        model.add(keras.layers.Bidirectional(
            keras.layers.LSTM(
                units,
                return_sequences=True,
                activation='tanh',
                recurrent_activation='sigmoid',
                dropout=0.0,  # Dropout on input units
                recurrent_dropout=0.2,  # Dropout on recurrent units
                kernel_regularizer=regularizers.l2(0.001),
                recurrent_regularizer=regularizers.l2(0.001),
                bias_regularizer=regularizers.l2(0.001),
                unroll=False,
                use_bias=True,
                # Add deterministic initializers
                kernel_initializer=keras.initializers.GlorotUniform(seed=kernel_seeds[seed_idx]),
                recurrent_initializer=keras.initializers.Orthogonal(seed=kernel_seeds[seed_idx+1]),
                bias_initializer=keras.initializers.Zeros()
            )
        ))
        seed_idx += 2
        
        # Add seeded dropout
        model.add(keras.layers.Dropout(dropout_rate, seed=kernel_seeds[seed_idx]))
        seed_idx += 1
        
        # Middle layers
        for i in range(1, n_layers - 1):
            return_sequences = True
            model.add(keras.layers.Bidirectional(keras.layers.LSTM(
                units,
                return_sequences=return_sequences,
                activation='tanh',
                recurrent_activation='sigmoid',
                dropout=0.0,
                recurrent_dropout=0.1,
                # kernel_regularizer=regularizers.l2(0.001),
                # recurrent_regularizer=regularizers.l2(0.001),
                # bias_regularizer=regularizers.l2(0.001),
                unroll=False,
                use_bias=True,
                # Add deterministic initializers
                kernel_initializer=keras.initializers.GlorotUniform(seed=kernel_seeds[seed_idx]),
                recurrent_initializer=keras.initializers.Orthogonal(seed=kernel_seeds[seed_idx+1]),
                bias_initializer=keras.initializers.Zeros()
            )))
            seed_idx += 2
            
            # Add seeded dropout
            model.add(keras.layers.Dropout(dropout_rate, seed=kernel_seeds[seed_idx]))
            seed_idx += 1
        
        # Final LSTM layer
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(
            units,
            return_sequences=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            dropout=0.0,
            recurrent_dropout=0.1,
            # kernel_regularizer=regularizers.l2(0.001),
            # recurrent_regularizer=regularizers.l2(0.001),
            # bias_regularizer=regularizers.l2(0.001),
            unroll=False,
            use_bias=True,
            # Add deterministic initializers
            kernel_initializer=keras.initializers.GlorotUniform(seed=kernel_seeds[seed_idx]),
            recurrent_initializer=keras.initializers.Orthogonal(seed=kernel_seeds[seed_idx+1]),
            bias_initializer=keras.initializers.Zeros()
        )))
        seed_idx += 2
        
        # Add seeded dropout
        model.add(keras.layers.Dropout(dropout_rate, seed=kernel_seeds[seed_idx]))
        seed_idx += 1
    else:
        # If only one layer, make it bidirectional for better performance
        model.add(keras.layers.Bidirectional(
            keras.layers.LSTM(
                units,
                return_sequences=False,
                activation='tanh',
                recurrent_activation='sigmoid',
                dropout=0.0,
                recurrent_dropout=0.1,
                # kernel_regularizer=regularizers.l2(0.001),
                # recurrent_regularizer=regularizers.l2(0.001),
                # bias_regularizer=regularizers.l2(0.001),
                unroll=False,
                use_bias=True,
                # Add deterministic initializers
                kernel_initializer=keras.initializers.GlorotUniform(seed=kernel_seeds[seed_idx]),
                recurrent_initializer=keras.initializers.Orthogonal(seed=kernel_seeds[seed_idx+1]),
                bias_initializer=keras.initializers.Zeros()
            )
        ))
        seed_idx += 2
        
        # Add seeded dropout
        model.add(keras.layers.Dropout(dropout_rate, seed=kernel_seeds[seed_idx]))
        seed_idx += 1
    
    # Add a dense layer before the output for better representation
    model.add(keras.layers.Dense(
        units=max(units // 2, output_shape * 2),
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
        # Add deterministic initializers
        kernel_initializer=keras.initializers.GlorotUniform(seed=kernel_seeds[seed_idx]),
        bias_initializer=keras.initializers.Zeros()
    ))
    seed_idx += 1
    
    # Output layer
    model.add(keras.layers.Dense(
        output_shape,
        # Add deterministic initializers
        kernel_initializer=keras.initializers.GlorotUniform(seed=kernel_seeds[seed_idx]),
        bias_initializer=keras.initializers.Zeros()
    ))
    
   
    
    
    return model


def get_predictions(model, dataset, scaler):
    """Get predictions from a windowed dataset and inverse transform them."""
    # Set evaluation mode to be deterministic - version-compatible approach
    try:
        # For newer TensorFlow versions
        tf.config.run_functions_eagerly(True)
    except:
        # For older versions, use an alternative approach
        pass
    
    all_predictions = []
    all_labels = []
    
    # Use a fixed batch size for prediction to ensure consistency
    for features, labels in dataset:
        batch_predictions = model.predict(features, verbose=0)
        
        # Make sure predictions and labels have compatible shapes before flattening
        # For single-step prediction (LABEL_WIDTH=1), reshape appropriately
        if len(batch_predictions.shape) == 3:  # Shape is [batch, time_steps, features]
            batch_predictions = batch_predictions[:, -1, :]  # Get last time step
            
        if len(labels.shape) == 3:  # Shape is [batch, time_steps, features]
            labels = labels[:, -1, :]  # Get last time step
            
        # Now flatten and add to our collections
        all_predictions.extend(batch_predictions.flatten())
        all_labels.extend(labels.numpy().flatten())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions).reshape(-1, 1)
    all_labels = np.array(all_labels).reshape(-1, 1)
    
    # Inverse transform to get original scale
    all_predictions = scaler.inverse_transform(all_predictions)
    all_labels = scaler.inverse_transform(all_labels)
    
    # Ensure both arrays have the same length
    min_length = min(len(all_predictions), len(all_labels))
    all_predictions = all_predictions[:min_length]
    all_labels = all_labels[:min_length]
    
    try:
        # Reset eager execution when done for newer TF versions
        tf.config.run_functions_eagerly(False)
    except:
        pass
    
    return all_predictions.flatten(), all_labels.flatten()


def add_time_features(df, datetime_column=None):
    """
    Create cyclical time features from a datetime column or index.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the datetime column or with datetime index
    datetime_column : str, optional
        Name of the datetime column. If None, uses the DataFrame index
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with additional cyclical time features
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Get the datetime series from column or index
    if datetime_column is not None:
        if datetime_column not in df.columns:
            raise ValueError(f"Column '{datetime_column}' not found in DataFrame")
        date_time = pd.to_datetime(df[datetime_column])
    else:
        # Use index if no column specified
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                date_time = pd.to_datetime(df.index)
            except:
                raise ValueError("DataFrame index cannot be converted to datetime and no datetime_column specified")
        else:
            date_time = df.index
    
    # Convert datetime to timestamp in seconds
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    
    # Define day and year in seconds
    day = 24 * 60 * 60  # seconds in a day
    year = 365.2425 * day  # seconds in a year
    
    # Create cyclical features for day and year
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    
    # Add week features (7 day cycle)
    week = 7 * day
    df['Week sin'] = np.sin(timestamp_s * (2 * np.pi / week))
    df['Week cos'] = np.cos(timestamp_s * (2 * np.pi / week))
    
    # Add month features (30.44 day cycle)
    month = 30.44 * day
    df['Month sin'] = np.sin(timestamp_s * (2 * np.pi / month))
    df['Month cos'] = np.cos(timestamp_s * (2 * np.pi / month))
    
    return df


def main():
    # Set parameters for the model
    INPUT_WIDTH = 4   # Use 4 time steps as input
    LABEL_WIDTH = 1    # Predict 1 time step ahead
    SHIFT = 1          # Predict 1 step ahead
    BATCH_SIZE = 64   # Batch size for training
    EXCLUDE_LABEL = True  # Exclude labels in input features
    
    # Set matplotlib to deterministic mode
    import matplotlib as mpl
    mpl.rcParams['agg.path.chunksize'] = 10000
    np.random.seed(SEED)  # Ensure matplotlib's random operations use our seed
    
    # Load data
    print("Loading data...")
    data_dir = Path('./outputs/processed_data/merged/site/gap_filled_size1_with_era5')
    data_list = list(data_dir.glob('*_merged.csv'))
    
    if not data_list:
        print(f"No CSV files found in {data_dir}")
        sys.exit(1)
    
    # Sort data files for deterministic processing order
    data_list = sorted(data_list)
    
    print(f"Found {len(data_list)} data files")
    
    # Process data files
    all_segments = []
    used_cols = ['sap_velocity','soil_temperature_level_1','soil_temperature_level_3','surface_net_solar_radiation', 'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_1','ppfd_in','sw_in', 'ta', 'precip', 'u_component_of_wind_10m', 'v_component_of_wind_10m', 'vpd', 'leaf_area_index_high_vegetation', 'leaf_area_index_low_vegetation', 'biome']
    all_biome_types = set()  # Will collect all unique biome types

    # Sort data files for deterministic processing order
    data_list = sorted(data_list)

    for data_file in data_list:
        print(f"Processing {data_file.name}")
        try:
            df = pd.read_csv(data_file, parse_dates=['TIMESTAMP'])
            df.set_index('TIMESTAMP', inplace=True)
            df = add_time_features(df)
            
            # Verify columns exist
            missing_cols = [col for col in used_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns in {data_file.name}: {missing_cols}")
                continue
                
            # Fix potential case/whitespace issues
            df.columns = [col.strip().lower() for col in df.columns]
            used_cols_lower = [col.lower() for col in used_cols]
            df = df[used_cols_lower]
            # Process biome column
            if 'biome' in used_cols_lower:
                if 'biome' in df.columns:
                    # Store biome values to collect all types
                    all_biome_types.update(df['biome'].unique())
                    
                    # Get non-biome columns 
                    orig_cols = [col for col in used_cols_lower if col != 'biome']
                    
                    # Create dummy variables for biome with fixed random state and no dropping of first column
                    biome_df = pd.get_dummies(df['biome'], prefix='', prefix_sep='', dtype=float, drop_first=False)
                    
                    # Join with original data
                    df = df[orig_cols].join(biome_df)
                else:
                    print(f"Warning: Missing biome column in {data_file.name}")
                    continue
            
            # Clean data (only once)
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            
            if len(df) < INPUT_WIDTH + SHIFT:
                print(f"Warning: {data_file.name} has too few records after cleaning: {len(df)}")
                continue
            
            # Segment the data
            segments = TimeSeriesSegmenter.segment_time_series(
                df, 
                gap_threshold=2, 
                unit='days', 
                min_segment_length=INPUT_WIDTH + SHIFT + 7
            )
            
            print(f"  Created {len(segments)} segments")
            all_segments.extend(segments)
            
        except Exception as e:
            print(f"Error processing {data_file.name}: {e}")

    if not all_segments:
        print("No valid segments found. Check your data files.")
        sys.exit(1)

    print(f"Total segments collected: {len(all_segments)}")
    print(f"All biome types found: {all_biome_types}")

    # Sort biome types for consistent order
    all_biome_types = sorted(all_biome_types)

    # Ensure all segments have all biome types as columns in the same order
    for segment in all_segments:
        for biome_type in all_biome_types:
            if biome_type not in segment.columns:
                segment[biome_type] = 0.0

    # Now standardize the data
    all_data = pd.concat(all_segments)

    # Create scalers
    feature_scaler = StandardScaler()
    label_scaler = StandardScaler()

    # Get feature columns (excluding biome dummy columns and target)
    base_feature_columns = [col for col in used_cols_lower if col != 'biome' and col != 'sap_velocity']
    biome_columns = list(all_biome_types)
    feature_columns = base_feature_columns + biome_columns
    feature_scaler.fit(all_data[feature_columns])
    label_scaler.fit(all_data[['sap_velocity']])
    
    # Apply scaling to each segment
    scaled_segments = []
    for segment in all_segments:
        segment_copy = segment.copy()
        segment_copy[feature_columns] = feature_scaler.transform(segment[feature_columns])
        segment_copy['sap_velocity'] = label_scaler.transform(segment[['sap_velocity']])
        scaled_segments.append(segment_copy)
    
    # Create windowed datasets - Try with different split ratios if the first one fails
    print("Creating windowed datasets...")
    try:
        # First attempt with regular split
        datasets = SegmentedWindowGenerator.create_complete_dataset_from_segments(
            segments=scaled_segments,
            input_width=INPUT_WIDTH,
            label_width=LABEL_WIDTH,
            shift=SHIFT,
            label_columns=['sap_velocity'],
            train_val_test_split=(0.8, 0.1, 0.1),
            batch_size=BATCH_SIZE,
            exclude_labels_from_inputs=EXCLUDE_LABEL
        )
    except ValueError as e:
        print(f"Error with default split ratio: {e}")
        print("Trying with an alternative split ratio...")
        
        # Second attempt with more balanced split
        datasets = SegmentedWindowGenerator.create_complete_dataset_from_segments(
            segments=scaled_segments,
            input_width=INPUT_WIDTH,
            label_width=LABEL_WIDTH,
            shift=SHIFT,
            label_columns=['sap_velocity'],
            train_val_test_split=(0.8, 0.1, 0.1),  # More data for validation/test
            batch_size=BATCH_SIZE,
            exclude_labels_from_inputs=EXCLUDE_LABEL  # Include labels in inputs
        )
    
    train_ds = datasets['train']
    val_ds = datasets['val']
    test_ds = datasets['test']
    
    # Check dataset properties
    if train_ds is None:
        print("ERROR: Could not create training dataset.")
        sys.exit(1)
    
    # Print dataset shapes for debugging
    print("Dataset shapes:")
    for features_batch, labels_batch in train_ds.take(1):
        print(f"Features batch shape: {features_batch.shape}")
        print(f"Labels batch shape: {labels_batch.shape}")
        n_features = features_batch.shape[2]

    
    # Define parameter grid for LSTM with deterministic settings
    param_grid = {
        'architecture': {
            'n_layers': [1],
            'units': [64],
            'dropout_rate': [0.2]
        },
        'optimizer': {
            'name': ['Adam'],
            'learning_rate': [0.001]
        },
        'training': {
            'batch_size': BATCH_SIZE,
            'epochs': 100,
            'patience': 50
        }
    }
    
    # Create and fit optimizer with deterministic behavior
    print("Starting hyperparameter optimization...")
    optimizer = DLOptimizer(
        base_architecture=create_lstm_model,
        task='regression',
        model_type='LSTM',
        param_grid=param_grid,
        input_shape=(INPUT_WIDTH, n_features),  # (timesteps, features)
        output_shape=LABEL_WIDTH,  # Output prediction length
        scoring='val_loss',
        random_state=SEED,  # Add random state for deterministic behavior
    )
    
    # Convert TensorFlow dataset to X and y numpy arrays in a deterministic way
    def convert_tf_dataset_to_xy(dataset):
        features_list = []
        labels_list = []
        
        # Iterate through all batches in the dataset
        for features_batch, labels_batch in dataset:
            # Convert TensorFlow tensors to numpy arrays
            features_list.append(features_batch.numpy())
            labels_list.append(labels_batch.numpy())
        
        # Concatenate batches
        X = np.concatenate(features_list, axis=0)
        y = np.concatenate(labels_list, axis=0)
        
        # If labels are shape (batch, seq_len, features) and you need (batch, features)
        if y.ndim > 2:
            y = y.reshape(y.shape[0], -1)
        
        return X, y

    # Using the function with deterministic behavior
    X_train, y_train = convert_tf_dataset_to_xy(train_ds)
    X_val, y_val = convert_tf_dataset_to_xy(val_ds)

    # Set additional TensorFlow determinism options before fitting
    # Use version-compatible approach
    try:
        # For newer TensorFlow versions
        tf.keras.utils.set_random_seed(SEED)
    except:
        # For older TensorFlow versions
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
    
    # Custom fitting for time series data
    optimizer.fit(
        X_train, 
        y_train,
        is_cv=False,  # We're not using cross-validation
        X_val=X_val,  # Validation data
        y_val=y_val,  # Validation labels
        split_type='random',  # We've already created our datasets
    )
    
    # Get the best model
    best_model = optimizer.get_best_model()
    
    # Get predictions with deterministic settings
    print("Generating predictions...")
    test_predictions, test_labels_actual = get_predictions(best_model, test_ds, label_scaler)
    train_predictions, train_labels_actual = get_predictions(best_model, train_ds, label_scaler)
    
    # Create plots directory if it doesn't exist
    plot_dir = Path('./plots')
    plot_dir.mkdir(exist_ok=True)
    
    # Plot predictions vs actual for test data
    plt.figure(figsize=(10, 10))
    plt.scatter(test_labels_actual, test_predictions, alpha=0.5)
    plt.xlabel('True Values [Sap Velocity]')
    plt.ylabel('Predictions [Sap Velocity]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, max(plt.xlim()[1], plt.ylim()[1])])
    plt.ylim([0, max(plt.xlim()[1], plt.ylim()[1])])
    plt.plot([-100, 100], [-100, 100])
    plt.title('LSTM Model: Test Predictions vs Actual')
    plt.savefig(plot_dir / 'lstm_test_predictions.png')
    plt.close()
    
    # Plot predictions vs actual for training data
    plt.figure(figsize=(10, 10))
    plt.scatter(train_labels_actual, train_predictions, alpha=0.5)
    plt.xlabel('True Values [Sap Velocity]')
    plt.ylabel('Predictions [Sap Velocity]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, max(plt.xlim()[1], plt.ylim()[1])])
    plt.ylim([0, max(plt.xlim()[1], plt.ylim()[1])])
    plt.plot([-100, 100], [-100, 100])
    plt.title('LSTM Model: Training Predictions vs Actual')
    plt.savefig(plot_dir / 'lstm_train_predictions.png')
    plt.close()
    
    # Plot error distribution for test data
    error = test_predictions - test_labels_actual
    plt.figure(figsize=(10, 7))
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [Sap Velocity]')
    plt.ylabel('Count')
    plt.title('LSTM Model: Test Error Distribution')
    plt.savefig(plot_dir / 'lstm_test_error_distribution.png')
    plt.close()
    
    # Plot error distribution for training data
    error_train = train_predictions - train_labels_actual
    plt.figure(figsize=(10, 7))
    plt.hist(error_train, bins=25)
    plt.xlabel('Prediction Error [Sap Velocity]')
    plt.ylabel('Count')
    plt.title('LSTM Model: Training Error Distribution')
    plt.savefig(plot_dir / 'lstm_train_error_distribution.png')
    plt.close()
    
    # Calculate and print metrics
    r2 = r2_score(test_labels_actual, test_predictions)
    r2_train = r2_score(train_labels_actual, train_predictions)
    
    test_mse = mean_squared_error(test_labels_actual, test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(test_labels_actual, test_predictions)
    
    train_mse = mean_squared_error(train_labels_actual, train_predictions)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(train_labels_actual, train_predictions)
    
    print("\nModel Evaluation:")
    print("=================")
    print(f"R² Score (test): {r2:.3f}")
    print(f"R² Score (train): {r2_train:.3f}")
    
    print("\nTest Metrics:")
    print(f"MSE: {test_mse:.3f}")
    print(f"RMSE: {test_rmse:.3f}")
    print(f"MAE: {test_mae:.3f}")
    
    print("\nTrain Metrics:")
    print(f"MSE: {train_mse:.3f}")
    print(f"RMSE: {train_rmse:.3f}")
    print(f"MAE: {train_mae:.3f}")
    
    # Print optimization summary
    print("\nOptimization Summary:")
    print(f"Best parameters: {optimizer.best_params_}")
    print(f"Best validation score: {optimizer.best_score_}")
    
    # Plot training history if available with deterministic rendering
    if hasattr(optimizer, 'history_') and optimizer.history_ is not None:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(optimizer.history_['loss'])
        plt.plot(optimizer.history_['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.subplot(1, 2, 2)
        plt.plot(optimizer.history_['mae'])
        plt.plot(optimizer.history_['val_mae'])
        plt.title('Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'lstm_training_history.png')
        plt.close()
    
    # Save the best model
    model_dir = Path('./models')
    model_dir.mkdir(exist_ok=True)
    best_model.save(model_dir / 'lstm_sap_velocity_model')
    
    print("\nAnalysis complete. Model and visualizations saved.")
    
    # Return some key metric for determinism testing
    return test_rmse


def verify_determinism():
    """Test if the pipeline produces deterministic results."""
    # Save current random state
    current_seed = SEED
    
    # First run with fixed seed
    np.random.seed(42)
    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    print("\nRunning first determinism test...")
    result1 = main()
    
    # Second run with same seed
    np.random.seed(42)
    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    print("\nRunning second determinism test...")
    result2 = main()
    
    # Check if results match
    print("\nDeterminism Test Results:")
    print(f"First run RMSE: {result1:.6f}")
    print(f"Second run RMSE: {result2:.6f}")
    print(f"Difference: {abs(result1 - result2):.10f}")
    print(f"Results identical: {np.isclose(result1, result2, rtol=1e-10, atol=1e-10)}")
    
    # Restore original seed
    np.random.seed(current_seed)
    tf.random.set_seed(current_seed)
    
    return result1, result2


if __name__ == "__main__":
    # Run the main function with deterministic controls
    print("\nRunning main function with enhanced determinism...")
    main()
    
    # Uncomment to verify pipeline determinism
    # print("\nVerifying pipeline determinism...")
    # result1, result2 = verify_determinism()