import numpy as np
import pandas as pd
from pathlib import Path




import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Dict, Tuple, Any


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Dict, Tuple, Any


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
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=self.shuffle,
            batch_size=self.batch_size)

        ds = ds.map(self.split_window)
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
        
        # Override min_segment_length if it's less than required for windowing
        if min_segment_length is None or min_segment_length < min_required_length:
            min_segment_length = min_required_length
        
        for segment in segments:
            # Skip segments that are too short
            if len(segment) < min_segment_length:
                continue
                
            # Split the segment into train, validation, and test sets
            segment_len = len(segment)
            train_len = int(segment_len * train_val_test_split[0])
            val_len = int(segment_len * train_val_test_split[1])
            
            train_df = segment.iloc[:train_len]
            val_df = segment.iloc[train_len:train_len+val_len]
            test_df = segment.iloc[train_len+val_len:]
            
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
            return {'train': None, 'val': None, 'test': None}
        
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
        
        # Shuffle training data
        combined['train'] = combined['train'].shuffle(buffer_size)
        
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





 

if __name__ == "__main__":
    # Test the function
    """
    np.random.seed(0)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    values = np.random.randn(100)
    df = pd.DataFrame({'value': values}, index=dates)
    # Introduce a gap
    df.loc['2020-01-10':'2020-01-20'] = np.nan
    """
    # Load data
    data_dir = Path('./outputs\processed_data\merged\site\gap_filled_size1')
    data_list = list(data_dir.rglob('*.csv'))
    all_segments = []
    for data_file in data_list[:10]:
        df = pd.read_csv(data_file, parse_dates=['TIMESTAMP'])
        # Set index
        df.set_index('TIMESTAMP', inplace=True)

        segments = TimeSeriesSegmenter.segment_time_series(df, gap_threshold=2, unit='days', min_segment_length=7)
        all_segments.extend(segments)
    TimeSeriesSegmenter.plot_segments(all_segments, value_column='sap_velocity')
    
    for i, segment in enumerate(all_segments):
        print(f"Segment {i + 1}: {segment.index[0]} to {segment.index[-1]}, length={len(segment)}")
        print(segment.head())
    print(f"Total segments: {len(all_segments)}", all_segments)