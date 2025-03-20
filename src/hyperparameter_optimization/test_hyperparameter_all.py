import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import argparse

# Add parent directory to Python path if needed
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import required modules
try:
    from src.hyperparameter_optimization.hyper_tuner import DLOptimizer, MLOptimizer
except ImportError:
    print("Warning: Unable to import hyper_tuner module. Using placeholder implementations.")
    
    # Placeholder implementation if modules can't be imported
    class DLOptimizer:
        def __init__(self, **kwargs):
            self.best_params_ = {}
            self.best_score_ = 0
            self.history_ = None
            
        def fit(self, X, y, **kwargs):
            pass
            
        def get_best_model(self):
            return None
    
    class MLOptimizer:
        def __init__(self, **kwargs):
            self.best_params_ = {}
            self.best_estimator_ = None
            self.best_score_ = 0
            
        def fit(self, X, y, **kwargs):
            pass

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Global plotting settings
plt.style.use('seaborn-v0_8-whitegrid')
PLOT_DPI = 100
FIGSIZE_STANDARD = (10, 6)
FIGSIZE_COMPARISON = (12, 8)
FIGSIZE_SCATTER = (8, 8)

# Define paths
PLOT_DIR = Path('./plots')
DATA_DIR = Path('./outputs/processed_data/merged/site/gap_filled_size1_with_era5')

# Ensure directories exist
PLOT_DIR.mkdir(exist_ok=True)


# Colors for different models
MODEL_COLORS = {
    'XGBoost': '#2C8ECF',
    'RandomForest': '#7EB26D',
    'SVM': '#EAB839',
    'Neural Network': '#E24D42',
    'LSTM': '#1F78C1',
    'Transformer': '#BA43A9',
    'CNN-LSTM': '#705DA0'
}

# Model results storage
model_results = {
    'model_name': [],
    'train_r2': [],
    'test_r2': [],
    'train_rmse': [],
    'test_rmse': [],
    'train_mae': [],
    'test_mae': []
}

# Utility functions
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

class TimeSeriesSegmenter:
    """
    Utility class for segmenting time series data with gaps.
    """
    
    @staticmethod
    def segment_time_series(
        df, 
        gap_threshold=2, 
        unit='hours', 
        min_segment_length=None,
        timestamp_column=None
    ):
        """
        Segments a time series into continuous blocks based on time gaps.
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
        input_width, 
        label_width, 
        shift,
        train_df,
        val_df,
        test_df,
        label_columns=None,
        batch_size=64,
        shuffle=True,
        exclude_labels_from_inputs=True
    ):
        """
        Initialize the window generator with the specified parameters.
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
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        
        if self.label_columns is not None:
            # Extract the label columns
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
                
            # Remove label columns from inputs if requested
            if hasattr(self, 'exclude_labels_from_inputs') and self.exclude_labels_from_inputs:
                feature_indices = [i for i, name in enumerate(self.train_df.columns) 
                                if name not in self.label_columns]
                inputs = tf.gather(inputs, feature_indices, axis=2)

        # Slicing doesn't preserve static shape information, so set the shapes manually
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None if self.label_columns is None else len(self.label_columns)])

        return inputs, labels
    
    def make_dataset(self, data):
        """
        Create a windowed tf.data.Dataset from a pandas DataFrame.
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            seed=RANDOM_SEED)

        ds = ds.map(self.split_window)
        return ds
    
    @property
    def train(self):
        """Get the training dataset."""
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        """Get the validation dataset."""
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        """Get the test dataset."""
        return self.make_dataset(self.test_df)

    @property
    def example(self):
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

class SegmentedWindowGenerator:
    """
    Handles creating and combining WindowGenerator objects from segmented time series data.
    """
    
    @staticmethod
    def create_window_generators_from_segments(
        segments, 
        input_width, 
        label_width, 
        shift,
        train_val_test_split=(0.7, 0.15, 0.15),
        label_columns=None,
        batch_size=32,
        min_segment_length=None,
        exclude_labels_from_inputs=True
    ):
        """
        Creates WindowGenerator objects for each valid segment.
        """
        window_generators = []
        min_required_length = input_width + shift
        
        # Calculate minimum samples needed for each split to create valid windows
        min_train_samples = min_required_length
        min_val_samples = min_required_length
        min_test_samples = min_required_length
        
        # Calculate total minimum segment length based on split ratios
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
                continue
                
            # Split the segment into train, validation, and test sets
            segment_len = len(segment)
            train_len = int(segment_len * train_val_test_split[0])
            val_len = int(segment_len * train_val_test_split[1])
            
            train_df = segment.iloc[:train_len]
            val_df = segment.iloc[train_len:train_len+val_len]
            test_df = segment.iloc[train_len+val_len:]
            
            # Double-check that each split has minimum required length
            if len(train_df) < min_train_samples or len(val_df) < min_val_samples or len(test_df) < min_test_samples:
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
    def combine_datasets(window_generators, buffer_size=1000):
        """
        Combines datasets from multiple WindowGenerator objects.
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
        
        # Shuffle training data
        combined['train'] = combined['train'].shuffle(buffer_size, seed=RANDOM_SEED)
        
        # Optionally rebatch to ensure consistent batch sizes
        batch_size = window_generators[0].batch_size
        combined['train'] = combined['train'].unbatch().batch(batch_size)
        combined['val'] = combined['val'].unbatch().batch(batch_size) 
        combined['test'] = combined['test'].unbatch().batch(batch_size)
        
        return combined
    
    @staticmethod
    def create_complete_dataset_from_segments(
        segments,
        input_width,
        label_width,
        shift,
        train_val_test_split=(0.7, 0.15, 0.15),
        label_columns=None,
        batch_size=32,
        buffer_size=1000,
        exclude_labels_from_inputs=True
    ):
        """
        Creates a complete set of datasets from segmented time series.
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

def get_predictions(model, dataset, scaler=None):
    """
    Get predictions from a windowed dataset and inverse transform them.
    
    Parameters:
    -----------
    model : tf.keras.Model
        The trained model to make predictions
    dataset : tf.data.Dataset
        The windowed dataset with features and labels
    scaler : sklearn.preprocessing.StandardScaler, optional
        Scaler to inverse transform the predictions and labels
        
    Returns:
    --------
    tuple
        (predictions, actual_values) both as flattened numpy arrays
    """
    all_predictions = []
    all_labels = []
    
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
    if scaler is not None:
        all_predictions = scaler.inverse_transform(all_predictions)
        all_labels = scaler.inverse_transform(all_labels)
    
    # Ensure both arrays have the same length
    min_length = min(len(all_predictions), len(all_labels))
    all_predictions = all_predictions[:min_length]
    all_labels = all_labels[:min_length]
    
    return all_predictions.flatten(), all_labels.flatten()

def load_and_preprocess_data(file_path=None, biome_column='biome', target_column='sap_velocity', training_portion=0.8):
    """
    Load and preprocess the data for modeling.
    
    Parameters:
    -----------
    file_path : str or Path, optional
        Path to the data file. If None, uses default path
    biome_column : str
        Name of the categorical biome column
    target_column : str
        Name of the target column to predict
    training_portion : float
        Portion of data to use for training (0-1)
        
    Returns:
    --------
    tuple
        (X_train, y_train, X_test, y_test, train_data_orig, test_data_orig)
    """
    if file_path is None:
        file_path = DATA_DIR / 'all_biomes_merged_data.csv'
    
    # Load data
    data = pd.read_csv(file_path).set_index('TIMESTAMP').sort_index().dropna()
    
    # Select relevant columns - using a standardized set across all models
    data = data[['sap_velocity', 'vpd', 'ta', 'swc_shallow', 'ppfd_in', 
                 'volumetric_soil_water_layer_4', 'leaf_area_index_low_vegetation', 
                 'soil_temperature_level_3', 'volumetric_soil_water_layer_3', 'biome', 
                 'swc_deep', 'u_component_of_wind_10m', 'v_component_of_wind_10m']]
    
    # Convert biome to categorical and create dummy variables
    if biome_column in data.columns:
        data[biome_column] = data[biome_column].astype('category')
        data = pd.get_dummies(data, columns=[biome_column])
    
    # Add time features
    data = add_time_features(data)
    
    # Split into training and test sets
    training_data = data.iloc[:int(len(data)*training_portion)]
    test_data = data.iloc[int(len(data)*training_portion):]
    
    # Separate features and target
    X_train = training_data.drop(columns=[target_column])
    y_train = training_data[target_column].values
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column].values
    
    return X_train, y_train, X_test, y_test, training_data, test_data

def plot_results(y_true, y_pred, model_name, set_name, plot_dir=PLOT_DIR, figsize=FIGSIZE_STANDARD):
    """
    Create standard plots for model evaluation.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    model_name : str
        Name of the model
    set_name : str
        Name of the dataset ('train' or 'test')
    plot_dir : Path
        Directory to save plots
    figsize : tuple
        Figure size (width, height)
    """
    # Create directory for this model if it doesn't exist
    model_plot_dir = plot_dir / model_name.lower().replace(' ', '_')
    model_plot_dir.mkdir(exist_ok=True)
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # 1. Time Series Plot
    if hasattr(y_true, 'index'):
        plt.figure(figsize=figsize)
        plt.plot(y_true.index, y_true, label='Actual', color='blue', alpha=0.7)
        plt.plot(y_true.index, y_pred, label='Predicted', color='red', alpha=0.7)
        plt.title(f'{model_name}: Actual vs Predicted Sap Velocity ({set_name})')
        plt.xlabel('Time')
        plt.ylabel('Sap Velocity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(model_plot_dir / f'{set_name}_timeseries.png', dpi=PLOT_DPI)
        plt.close()
    
    # 2. Scatter Plot
    plt.figure(figsize=FIGSIZE_SCATTER)
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    
    # Add regression line
    m, b = np.polyfit(y_true, y_pred, 1)
    plt.plot(y_true, m*y_true + b, color='red', lw=2)
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name}: Predicted vs Actual ({set_name})')
    plt.axis('equal')
    plt.axis('square')
    
    # Add metrics text
    plt.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}',
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(model_plot_dir / f'{set_name}_scatter.png', dpi=PLOT_DPI)
    plt.close()
    
    # 3. Error Distribution
    errors = y_pred - y_true
    plt.figure(figsize=figsize)
    plt.hist(errors, bins=30, alpha=0.7, color=MODEL_COLORS.get(model_name, 'blue'))
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(f'{model_name}: Error Distribution ({set_name})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(model_plot_dir / f'{set_name}_error_distribution.png', dpi=PLOT_DPI)
    plt.close()
    
    return r2, rmse, mae

def store_model_results(model_name, train_results, test_results):
    """
    Store model evaluation results for later comparison.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    train_results : tuple
        (r2, rmse, mae) for training data
    test_results : tuple
        (r2, rmse, mae) for test data
    """
    train_r2, train_rmse, train_mae = train_results
    test_r2, test_rmse, test_mae = test_results
    
    model_results['model_name'].append(model_name)
    model_results['train_r2'].append(train_r2)
    model_results['test_r2'].append(test_r2)
    model_results['train_rmse'].append(train_rmse)
    model_results['test_rmse'].append(test_rmse)
    model_results['train_mae'].append(train_mae)
    model_results['test_mae'].append(test_mae)
    
    # Print results
    print(f"\n{model_name} Results:")
    print(f"Training - R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
    print(f"Test     - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

def plot_model_comparison():
    """
    Create comparison plots for all models.
    """
    # Convert results to DataFrame for easier plotting
    results_df = pd.DataFrame(model_results)
    
    # Sort by test R²
    results_df = results_df.sort_values(by='test_r2', ascending=False)
    
    # 1. R² Comparison
    plt.figure(figsize=FIGSIZE_COMPARISON)
    bar_width = 0.35
    index = np.arange(len(results_df))
    
    plt.bar(index, results_df['train_r2'], bar_width, label='Training', alpha=0.7, color='blue')
    plt.bar(index + bar_width, results_df['test_r2'], bar_width, label='Test', alpha=0.7, color='red')
    
    plt.xlabel('Model')
    plt.ylabel('R² Score')
    plt.title('Model Comparison: R² Score')
    plt.xticks(index + bar_width/2, results_df['model_name'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'model_comparison_r2.png', dpi=PLOT_DPI)
    plt.close()
    
    # 2. RMSE Comparison
    plt.figure(figsize=FIGSIZE_COMPARISON)
    plt.bar(index, results_df['train_rmse'], bar_width, label='Training', alpha=0.7, color='blue')
    plt.bar(index + bar_width, results_df['test_rmse'], bar_width, label='Test', alpha=0.7, color='red')
    
    plt.xlabel('Model')
    plt.ylabel('RMSE')
    plt.title('Model Comparison: RMSE')
    plt.xticks(index + bar_width/2, results_df['model_name'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'model_comparison_rmse.png', dpi=PLOT_DPI)
    plt.close()
    
    # 3. Combined Metric Plot with Seaborn
    plt.figure(figsize=FIGSIZE_COMPARISON)
    
    # Reshape data for seaborn
    r2_data = pd.melt(results_df, 
                      id_vars=['model_name'], 
                      value_vars=['train_r2', 'test_r2'],
                      var_name='Dataset', 
                      value_name='R²')
    r2_data['Dataset'] = r2_data['Dataset'].map({'train_r2': 'Training', 'test_r2': 'Test'})
    
    # Create plot
    sns.barplot(x='model_name', y='R²', hue='Dataset', data=r2_data)
    plt.xlabel('Model')
    plt.ylabel('R² Score')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'model_comparison_combined.png', dpi=PLOT_DPI)
    plt.close()
    
    # 4. Create a summary table
    fig, ax = plt.subplots(figsize=(12, len(results_df)*0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = results_df[['model_name', 'train_r2', 'test_r2', 'train_rmse', 'test_rmse']]
    table_data.columns = ['Model', 'Train R²', 'Test R²', 'Train RMSE', 'Test RMSE']
    
    # Format values
    for col in table_data.columns[1:]:
        table_data[col] = table_data[col].map(lambda x: f"{x:.4f}")
    
    # Create table
    table = ax.table(cellText=table_data.values, colLabels=table_data.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Header styling
    for i, key in enumerate(table._cells):
        cell = table._cells[key]
        if key[0] == 0:  # Header row
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4472C4')
        else:  # Data rows
            if i % 2 == 0:
                cell.set_facecolor('#D9E1F2')
            if key[1] == 0:  # Model name column
                cell.set_text_props(weight='bold')
    
    plt.title('Model Performance Summary', pad=20)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'model_performance_table.png', dpi=PLOT_DPI)
    plt.close()
    
    # Save results to CSV
    results_df.to_csv(PLOT_DIR / 'model_results.csv', index=False)
    
    print("\nModel comparison plots saved to:", PLOT_DIR)

# Model definition functions
def create_transformer_model(input_shape, output_shape, d_model=128, num_heads=8, 
                            num_encoder_layers=4, dff=512, dropout_rate=0.1):
    """
    Create a Transformer model for time series prediction.
    """
    def positional_encoding(length, depth):
        """Generate positional encoding for transformer."""
        depth = depth / 2
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :] / depth
        
        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates
        
        pos_encoding = np.zeros((length, int(depth) * 2))
        pos_encoding[:, 0::2] = np.sin(angle_rads)
        pos_encoding[:, 1::2] = np.cos(angle_rads)
        
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    class TransformerEncoderLayer(tf.keras.layers.Layer):
        """Custom Transformer encoder layer."""
        def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
            super(TransformerEncoderLayer, self).__init__()
            
            self.mha = tf.keras.layers.MultiHeadAttention(
                key_dim=d_model // num_heads, num_heads=num_heads, dropout=dropout_rate)
            
            self.ffn = tf.keras.Sequential([
                tf.keras.layers.Dense(dff, activation='relu'),
                tf.keras.layers.Dense(d_model)
            ])
            
            self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            
            self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
            self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
            
        def call(self, inputs, training=False, mask=None):
            # Multi-head attention with residual connection and layer normalization
            attn_output = self.mha(inputs, inputs, attention_mask=mask)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            
            # Feed forward network with residual connection and layer normalization
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Project inputs to d_model dimensions
    x = tf.keras.layers.Dense(d_model, activation=None)(inputs)
    
    # Add positional encoding
    seq_length = input_shape[0]
    pos_encoding = positional_encoding(seq_length, d_model)
    x = x + pos_encoding
    
    # Dropout
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Transformer encoder layers
    for i in range(num_encoder_layers):
        x = TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)(x)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Dense layers for prediction
    x = tf.keras.layers.Dense(d_model // 2, activation='relu', 
                            kernel_regularizer=regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(output_shape)(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def create_lstm_model(input_shape, output_shape, n_layers, units, dropout_rate):
    """Create an LSTM model for time series prediction."""
    model = keras.Sequential()
    
    # Input layer
    model.add(keras.layers.Input(shape=input_shape))
    
    # LSTM layers
    for i in range(n_layers):
        return_sequences = i < n_layers - 1
        
        model.add(keras.layers.Bidirectional(
            keras.layers.LSTM(
                units,
                return_sequences=return_sequences,
                activation='tanh',
                recurrent_activation='sigmoid',
                dropout=0.0,
                recurrent_dropout=0.2,
                kernel_regularizer=regularizers.l2(0.001),
                recurrent_regularizer=regularizers.l2(0.001),
                bias_regularizer=regularizers.l2(0.001)
            )
        ))
        model.add(keras.layers.Dropout(dropout_rate))
    
    # Dense layer before output
    model.add(keras.layers.Dense(
        units=max(units // 2, output_shape * 2),
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001)
    ))
    
    # Output layer
    model.add(keras.layers.Dense(output_shape))
    
    return model

def create_cnn_lstm_model(input_shape, output_shape, cnn_layers=2, lstm_layers=2, 
                         cnn_filters=64, lstm_units=128, dropout_rate=0.3):
    """Create a hybrid CNN-LSTM model for time series prediction."""
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Reshape for CNN
    x = tf.keras.layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
    
    # CNN Block with explicit seeds
    for i in range(cnn_layers):
        kernel_size = 5 if i == 0 else 3
        
        x = tf.keras.layers.Conv2D(
            filters=cnn_filters * (2**i),
            kernel_size=(kernel_size, 1),
            padding='same',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(x)
        
        x = tf.keras.layers.BatchNormalization()(x)
        
        if i < cnn_layers - 1:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1))(x)
    
    # Reshape calculations
    new_timesteps = input_shape[0] // (2**(cnn_layers-1)) if cnn_layers > 1 else input_shape[0]
    new_features = cnn_filters * (2**(cnn_layers-1))
    
    # Reshape for LSTM
    x = tf.keras.layers.Reshape((new_timesteps, new_features * input_shape[1]))(x)
    
    # LSTM Block
    for i in range(lstm_layers):
        return_sequences = i < lstm_layers - 1
        
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                lstm_units,
                return_sequences=return_sequences,
                dropout=0.0,
                recurrent_dropout=0.2,
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            )
        )(x)
        
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Dense layer
    x = tf.keras.layers.Dense(
        lstm_units // 2,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(output_shape)(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def create_nn(input_shape, output_shape, n_layers, units, dropout_rate):
    """Create a neural network for time series prediction."""
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    
    for i in range(n_layers):
        model.add(keras.layers.Dense(
            units,
            activation='relu',
            kernel_initializer='glorot_uniform',
            kernel_regularizer=regularizers.l2(0.001)
        ))
        if dropout_rate > 0:
            model.add(keras.layers.Dropout(dropout_rate))
    
    model.add(keras.layers.Dense(output_shape))
    return model

# Model training functions
def train_xgboost_model(X_train, y_train, X_test, y_test):
    """
    Train an XGBoost model for sap velocity prediction.
    
    Parameters:
    -----------
    X_train, y_train : training data
    X_test, y_test : test data
    
    Returns:
    --------
    tuple
        (model, train_predictions, test_predictions)
    """
    print("\nTraining XGBoost model...")
    
    # Define parameter grid
    param_grid = {
        'reg_alpha': [1.0],
        'reg_lambda': [1.0],
        'max_depth': [9],
        'min_child_weight': [1],
        'gamma': [0.1],
        'subsample': [0.5],
        'colsample_bytree': [0.6],
        'colsample_bylevel': [0.9],
        'learning_rate': [0.01],
        'n_estimators': [500],
        'enable_categorical': [True]
    }
    
    # Create and train optimizer
    optimizer = MLOptimizer(
        param_grid=param_grid, 
        scoring='neg_mean_squared_error', 
        model_type='xgb', 
        task='regression'
    )
    
    optimizer.fit(X_train, y_train, split_type='temporal', is_shuffle=False)
    
    # Print results
    print(f"Best parameters: {optimizer.best_params_}")
    print(f"Best score: {optimizer.best_score_}")
    
    # Make predictions
    model = optimizer.best_estimator_
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Feature importance analysis
    feature_importance = model.feature_importances_
    feature_names = X_train.columns
    sorted_idx = np.argsort(feature_importance)
    
    plt.figure(figsize=FIGSIZE_STANDARD)
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'xgboost_feature_importance.png', dpi=PLOT_DPI)
    plt.close()
    
    
    
    return model, train_preds, test_preds

def train_random_forest_model(X_train, y_train, X_test, y_test):
    """
    Train a Random Forest model for sap velocity prediction.
    
    Parameters:
    -----------
    X_train, y_train : training data
    X_test, y_test : test data
    
    Returns:
    --------
    tuple
        (model, train_predictions, test_predictions)
    """
    print("\nTraining Random Forest model...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [500],
        'max_depth': [7],
        'min_samples_split': [5],
        'min_samples_leaf': [2],
        'max_features': ['sqrt'],
        'bootstrap': [True],
        'random_state': [RANDOM_SEED],
        'oob_score': [True]
    }
    
    # Create and train optimizer
    optimizer = MLOptimizer(
        param_grid=param_grid, 
        scoring='r2', 
        model_type='rf', 
        task='regression'
    )
    
    optimizer.fit(X_train, y_train, split_type='temporal')
    
    # Print results
    print(f"Best parameters: {optimizer.best_params_}")
    print(f"Best score: {optimizer.best_score_}")
    
    # Make predictions
    model = optimizer.best_estimator_
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    # Feature importance analysis
    feature_importance = model.feature_importances_
    feature_names = X_train.columns
    sorted_idx = np.argsort(feature_importance)
    
    plt.figure(figsize=FIGSIZE_STANDARD)
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'random_forest_feature_importance.png', dpi=PLOT_DPI)
    plt.close()
    

    
    return model, train_preds, test_preds

def train_svm_model(X_train, y_train, X_test, y_test):
    """
    Train an SVM model for sap velocity prediction.
    
    Parameters:
    -----------
    X_train, y_train : training data
    X_test, y_test : test data
    
    Returns:
    --------
    tuple
        (model, train_predictions, test_predictions)
    """
    print("\nTraining SVM model...")
    
    # Define parameter grid
    param_grid = [
        {
            'kernel': ['poly'],
            'C': [10],
            'gamma': ['scale'],
            'degree': [4],
            'coef0': [1],
            'epsilon': [0.5]
        }
    ]
    
    # Create and train optimizer
    optimizer = MLOptimizer(
        param_grid=param_grid, 
        scoring='r2', 
        model_type='svm', 
        task='regression'
    )
    
    optimizer.fit(X_train, y_train, split_type='temporal')
    
    # Print results
    print(f"Best parameters: {optimizer.best_params_}")
    print(f"Best score: {optimizer.best_score_}")
    
    # Make predictions
    model = optimizer.best_estimator_
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    

    
    return model, train_preds, test_preds

def train_neural_network_model(X_train, y_train, X_test, y_test):
    """
    Train a Neural Network model for sap velocity prediction.
    
    Parameters:
    -----------
    X_train, y_train : training data
    X_test, y_test : test data
    
    Returns:
    --------
    tuple
        (model, train_predictions, test_predictions)
    """
    print("\nTraining Neural Network model...")
    
    # Scale data
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    X_test_scaled = scaler_x.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Define parameter grid
    param_grid = {
        'architecture': {
            'n_layers': [2],
            'units': [64],
            'dropout_rate': [0.2] 
        },
        'optimizer': {   
            'name': ['Adam'],
            'learning_rate': [0.001]
        },
        'training': {
            'batch_size': 64,
            'epochs': 100,
            'patience': 20
        }
    }
    
    # Create and train optimizer
    optimizer = DLOptimizer(
        base_architecture=create_nn,
        task='regression',
        model_type='nn',
        param_grid=param_grid,
        input_shape=(X_train_scaled.shape[1],),
        output_shape=1,
        scoring='val_loss'
    )
    
    optimizer.fit(
        X_train_scaled,
        y_train_scaled,
        split_type='random',
        groups=None
    )
    
    # Get the best model
    model = optimizer.get_best_model()
    
    # Make predictions
    train_preds_scaled = model.predict(X_train_scaled).flatten()
    test_preds_scaled = model.predict(X_test_scaled).flatten()
    
    # Inverse transform predictions
    train_preds = scaler_y.inverse_transform(train_preds_scaled.reshape(-1, 1)).flatten()
    test_preds = scaler_y.inverse_transform(test_preds_scaled.reshape(-1, 1)).flatten()
    
    # Plot training history if available
    if hasattr(optimizer, 'history_') and optimizer.history_ is not None:
        plt.figure(figsize=FIGSIZE_STANDARD)
        plt.subplot(1, 2, 1)
        plt.plot(optimizer.history_['loss'])
        plt.plot(optimizer.history_['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.subplot(1, 2, 2)
        if 'mae' in optimizer.history_:
            plt.plot(optimizer.history_['mae'])
            plt.plot(optimizer.history_['val_mae'])
            plt.title('Model MAE')
            plt.ylabel('MAE')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(PLOT_DIR / 'neural_network_training_history.png', dpi=PLOT_DPI)
        plt.close()
    
    
    
    
    return model, train_preds, test_preds

def prepare_time_series_data(X_train, y_train, X_test, y_test, scaled=True, input_width=4, label_width=1, shift=1):
    """
    Prepare data for time series models (LSTM, Transformer, CNN-LSTM).
    
    Parameters:
    -----------
    X_train, y_train : training data
    X_test, y_test : test data
    scaled : bool
        Whether to scale the data
    input_width : int
        Number of time steps to use as input
    label_width : int
        Number of time steps to predict
    shift : int
        Offset between input and label windows
        
    Returns:
    --------
    tuple
        (window_generator, label_scaler)
    """
    # Create segments for time series models
    # First, reconstruct DataFrames
    train_df = pd.DataFrame(X_train.copy())
    train_df['sap_velocity'] = y_train
    
    test_df = pd.DataFrame(X_test.copy())
    test_df['sap_velocity'] = y_test
    
    # Split for validation
    val_size = int(len(train_df) * 0.2)
    val_df = train_df.iloc[-val_size:].copy()
    train_df = train_df.iloc[:-val_size].copy()
    
    if scaled:
        # Scale data
        feature_scaler = StandardScaler()
        label_scaler = StandardScaler()
        
        # Get feature and label columns
        feature_cols = [col for col in train_df.columns if col != 'sap_velocity']
        
        # Fit scalers on training data
        feature_scaler.fit(train_df[feature_cols])
        label_scaler.fit(train_df[['sap_velocity']])
        
        # Transform data
        for df in [train_df, val_df, test_df]:
            df[feature_cols] = feature_scaler.transform(df[feature_cols])
            df['sap_velocity'] = label_scaler.transform(df[['sap_velocity']])
    else:
        label_scaler = None
    
    # Create window generator
    window_generator = WindowGenerator(
        input_width=input_width,
        label_width=label_width,
        shift=shift,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_columns=['sap_velocity'],
        batch_size=64,
        exclude_labels_from_inputs=True
    )
    
    return window_generator, label_scaler

def train_lstm_model(X_train, y_train, X_test, y_test):
    """
    Train an LSTM model for sap velocity prediction.
    
    Parameters:
    -----------
    X_train, y_train : training data
    X_test, y_test : test data
    
    Returns:
    --------
    tuple
        (model, train_predictions, test_predictions)
    """
    print("\nTraining LSTM model...")
    
    # Prepare windowed data
    window_generator, label_scaler = prepare_time_series_data(
        X_train, y_train, X_test, y_test, 
        scaled=True, input_width=3, label_width=1, shift=1
    )
    
    # Get feature dimensions from a batch
    for features_batch, _ in window_generator.train.take(1):
        n_features = features_batch.shape[2]
        break
    
    # Define parameter grid
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
            'batch_size': 64,
            'epochs': 100,
            'patience': 50
        }
    }
    
    # Convert TensorFlow dataset to X and y numpy arrays
    def convert_tf_dataset_to_xy(dataset):
        features_list = []
        labels_list = []
        
        for features_batch, labels_batch in dataset:
            features_list.append(features_batch.numpy())
            labels_list.append(labels_batch.numpy())
        
        # Concatenate batches
        X = np.concatenate(features_list, axis=0)
        y = np.concatenate(labels_list, axis=0)
        
        # If labels are shape (batch, seq_len, features) and you need (batch, features)
        if y.ndim > 2:
            y = y.reshape(y.shape[0], -1)
        
        return X, y
    
    # Prepare data for optimizer
    X_train_lstm, y_train_lstm = convert_tf_dataset_to_xy(window_generator.train)
    X_val_lstm, y_val_lstm = convert_tf_dataset_to_xy(window_generator.val)
    
    # Create and train optimizer
    optimizer = DLOptimizer(
        base_architecture=create_lstm_model,
        task='regression',
        model_type='LSTM',
        param_grid=param_grid,
        input_shape=(3, n_features),  # (timesteps, features)
        output_shape=1,  # Output prediction length
        scoring='val_loss'
    )
    
    optimizer.fit(
        X_train_lstm, 
        y_train_lstm,
        is_cv=False,
        X_val=X_val_lstm,
        y_val=y_val_lstm,
        split_type='temporal'
    )
    
    # Get the best model
    model = optimizer.get_best_model()
    
    # Get predictions
    train_preds, train_y = get_predictions(model, window_generator.train, label_scaler)
    test_preds, test_y = get_predictions(model, window_generator.test, label_scaler)
    
    # Plot training history
    if hasattr(optimizer, 'history_') and optimizer.history_ is not None:
        plt.figure(figsize=FIGSIZE_STANDARD)
        plt.subplot(1, 2, 1)
        plt.plot(optimizer.history_['loss'])
        plt.plot(optimizer.history_['val_loss'])
        plt.title('LSTM Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.subplot(1, 2, 2)
        if 'mae' in optimizer.history_:
            plt.plot(optimizer.history_['mae'])
            plt.plot(optimizer.history_['val_mae'])
            plt.title('LSTM Model MAE')
            plt.ylabel('MAE')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(PLOT_DIR / 'lstm_training_history.png', dpi=PLOT_DPI)
        plt.close()
    
  
    
    return model, train_preds, test_y

def train_transformer_model(X_train, y_train, X_test, y_test):
    """
    Train a Transformer model for sap velocity prediction.
    
    Parameters:
    -----------
    X_train, y_train : training data
    X_test, y_test : test data
    
    Returns:
    --------
    tuple
        (model, train_predictions, test_predictions)
    """
    print("\nTraining Transformer model...")
    
    # Prepare windowed data
    window_generator, label_scaler = prepare_time_series_data(
        X_train, y_train, X_test, y_test, 
        scaled=True, input_width=4, label_width=1, shift=1
    )
    
    # Get feature dimensions from a batch
    for features_batch, _ in window_generator.train.take(1):
        n_features = features_batch.shape[2]
        break
    
    # Define parameter grid
    param_grid = {
        'architecture': {
            'd_model': [128],
            'num_heads': [8],
            'num_encoder_layers': [6],
            'dff': [256],
            'dropout_rate': [0.3]
        },
        'optimizer': {
            'name': ['Adam'],
        },
        'training': {
            'batch_size': 64,
            'epochs': 100,
            'patience': 15
        }
    }
    
    # Convert TensorFlow dataset to X and y numpy arrays
    def convert_tf_dataset_to_xy(dataset):
        features_list = []
        labels_list = []
        
        for features_batch, labels_batch in dataset:
            features_list.append(features_batch.numpy())
            labels_list.append(labels_batch.numpy())
        
        # Concatenate batches
        X = np.concatenate(features_list, axis=0)
        y = np.concatenate(labels_list, axis=0)
        
        # If labels are shape (batch, seq_len, features) and you need (batch, features)
        if y.ndim > 2:
            y = y.reshape(y.shape[0], -1)
        
        return X, y
    
    # Prepare data for optimizer
    X_train_tf, y_train_tf = convert_tf_dataset_to_xy(window_generator.train)
    X_val_tf, y_val_tf = convert_tf_dataset_to_xy(window_generator.val)
    
    # Create and train optimizer
    optimizer = DLOptimizer(
        base_architecture=create_transformer_model,
        task='regression',
        model_type='Transformer',
        param_grid=param_grid,
        input_shape=(4, n_features),
        output_shape=1,
        scoring='val_loss',
    )
    
    optimizer.fit(
        X_train_tf, 
        y_train_tf,
        is_cv=False,
        X_val=X_val_tf,
        y_val=y_val_tf,
        split_type='temporal'
    )
    
    # Get the best model
    model = optimizer.get_best_model()
    
    # Get predictions
    train_preds, train_y = get_predictions(model, window_generator.train, label_scaler)
    test_preds, test_y = get_predictions(model, window_generator.test, label_scaler)
    
    # Plot training history
    if hasattr(optimizer, 'history_') and optimizer.history_ is not None:
        plt.figure(figsize=FIGSIZE_STANDARD)
        plt.subplot(1, 2, 1)
        plt.plot(optimizer.history_['loss'])
        plt.plot(optimizer.history_['val_loss'])
        plt.title('Transformer Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.subplot(1, 2, 2)
        if 'mae' in optimizer.history_:
            plt.plot(optimizer.history_['mae'])
            plt.plot(optimizer.history_['val_mae'])
            plt.title('Transformer Model MAE')
            plt.ylabel('MAE')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(PLOT_DIR / 'transformer_training_history.png', dpi=PLOT_DPI)
        plt.close()
    
    
    
    return model, train_preds, test_y

def train_cnn_lstm_model(X_train, y_train, X_test, y_test):
    """
    Train a CNN-LSTM model for sap velocity prediction.
    
    Parameters:
    -----------
    X_train, y_train : training data
    X_test, y_test : test data
    
    Returns:
    --------
    tuple
        (model, train_predictions, test_predictions)
    """
    print("\nTraining CNN-LSTM model...")
    
    # Prepare windowed data
    window_generator, label_scaler = prepare_time_series_data(
        X_train, y_train, X_test, y_test, 
        scaled=True, input_width=4, label_width=1, shift=1
    )
    
    # Get feature dimensions from a batch
    for features_batch, _ in window_generator.train.take(1):
        n_features = features_batch.shape[2]
        break
    
    # Define parameter grid
    param_grid = {
        'architecture': {
            'cnn_layers': [1],
            'lstm_layers': [1],
            'cnn_filters': [8],
            'lstm_units': [6],
            'dropout_rate': [0.3]
        },
        'optimizer': {
            'name': ['Adam'],
        },
        'training': {
            'batch_size': 64,
            'epochs': 100,
            'patience': 10
        }
    }
    
    # Convert TensorFlow dataset to X and y numpy arrays
    def convert_tf_dataset_to_xy(dataset):
        features_list = []
        labels_list = []
        
        for features_batch, labels_batch in dataset:
            features_list.append(features_batch.numpy())
            labels_list.append(labels_batch.numpy())
        
        # Concatenate batches
        X = np.concatenate(features_list, axis=0)
        y = np.concatenate(labels_list, axis=0)
        
        # If labels are shape (batch, seq_len, features) and you need (batch, features)
        if y.ndim > 2:
            y = y.reshape(y.shape[0], -1)
        
        return X, y
    
    # Prepare data for optimizer
    X_train_cnn, y_train_cnn = convert_tf_dataset_to_xy(window_generator.train)
    X_val_cnn, y_val_cnn = convert_tf_dataset_to_xy(window_generator.val)
    
    # Create and train optimizer
    optimizer = DLOptimizer(
        base_architecture=create_cnn_lstm_model,
        task='regression',
        model_type='CNN-LSTM',
        param_grid=param_grid,
        input_shape=(4, n_features),
        output_shape=1,
        scoring='val_loss',
    )
    
    optimizer.fit(
        X_train_cnn, 
        y_train_cnn,
        is_cv=False,
        X_val=X_val_cnn,
        y_val=y_val_cnn,
        split_type='temporal'
    )
    
    # Get the best model
    model = optimizer.get_best_model()
    
    # Get predictions
    train_preds, train_y = get_predictions(model, window_generator.train, label_scaler)
    test_preds, test_y = get_predictions(model, window_generator.test, label_scaler)
    
    # Plot training history
    if hasattr(optimizer, 'history_') and optimizer.history_ is not None:
        plt.figure(figsize=FIGSIZE_STANDARD)
        plt.subplot(1, 2, 1)
        plt.plot(optimizer.history_['loss'])
        plt.plot(optimizer.history_['val_loss'])
        plt.title('CNN-LSTM Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.subplot(1, 2, 2)
        if 'mae' in optimizer.history_:
            plt.plot(optimizer.history_['mae'])
            plt.plot(optimizer.history_['val_mae'])
            plt.title('CNN-LSTM Model MAE')
            plt.ylabel('MAE')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(PLOT_DIR / 'cnn_lstm_training_history.png', dpi=PLOT_DPI)
        plt.close()
    
   
    
    return model, train_preds, test_y

def main(args):
    """
    Main function to run the model comparison.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    """
    print(f"Loading data from {args.data_path}")
    X_train, y_train, X_test, y_test, train_df, test_df = load_and_preprocess_data(
        file_path=args.data_path,
        training_portion=args.train_portion
    )
    
    # Dictionary to store all model results
    all_models = {}
    
    # Train models based on command line arguments
    if args.all or args.xgboost:
        model, train_preds, test_preds = train_xgboost_model(X_train, y_train, X_test, y_test)
        all_models['XGBoost'] = (model, train_preds, test_preds)
        
        # Plot and store results
        train_results = plot_results(y_train, train_preds, 'XGBoost', 'train')
        test_results = plot_results(y_test, test_preds, 'XGBoost', 'test')
        store_model_results('XGBoost', train_results, test_results)
    
    if args.all or args.random_forest:
        model, train_preds, test_preds = train_random_forest_model(X_train, y_train, X_test, y_test)
        all_models['RandomForest'] = (model, train_preds, test_preds)
        
        # Plot and store results
        train_results = plot_results(y_train, train_preds, 'RandomForest', 'train')
        test_results = plot_results(y_test, test_preds, 'RandomForest', 'test')
        store_model_results('RandomForest', train_results, test_results)
    
    if args.all or args.svm:
        model, train_preds, test_preds = train_svm_model(X_train, y_train, X_test, y_test)
        all_models['SVM'] = (model, train_preds, test_preds)
        
        # Plot and store results
        train_results = plot_results(y_train, train_preds, 'SVM', 'train')
        test_results = plot_results(y_test, test_preds, 'SVM', 'test')
        store_model_results('SVM', train_results, test_results)
    
    if args.all or args.neural_network:
        model, train_preds, test_preds = train_neural_network_model(X_train, y_train, X_test, y_test)
        all_models['Neural Network'] = (model, train_preds, test_preds)
        
        # Plot and store results
        train_results = plot_results(y_train, train_preds, 'Neural Network', 'train')
        test_results = plot_results(y_test, test_preds, 'Neural Network', 'test')
        store_model_results('Neural Network', train_results, test_results)
    
    if args.all or args.lstm:
        model, train_preds, test_y = train_lstm_model(X_train, y_train, X_test, y_test)
        all_models['LSTM'] = (model, train_preds, test_y)
        
        # Plot and store results
        train_results = plot_results(train_preds, train_preds, 'LSTM', 'train')  # Using predictions for both since original y is lost in windowing
        test_results = plot_results(test_y, test_y, 'LSTM', 'test')
        store_model_results('LSTM', train_results, test_results)
    
    if args.all or args.transformer:
        model, train_preds, test_y = train_transformer_model(X_train, y_train, X_test, y_test)
        all_models['Transformer'] = (model, train_preds, test_y)
        
        # Plot and store results
        train_results = plot_results(train_preds, train_preds, 'Transformer', 'train')
        test_results = plot_results(test_y, test_y, 'Transformer', 'test')
        store_model_results('Transformer', train_results, test_results)
    
    if args.all or args.cnn_lstm:
        model, train_preds, test_y = train_cnn_lstm_model(X_train, y_train, X_test, y_test)
        all_models['CNN-LSTM'] = (model, train_preds, test_y)
        
        # Plot and store results
        train_results = plot_results(train_preds, train_preds, 'CNN-LSTM', 'train')
        test_results = plot_results(test_y, test_y, 'CNN-LSTM', 'test')
        store_model_results('CNN-LSTM', train_results, test_results)
    
    # Create comparison plots
    if len(model_results['model_name']) > 1:
        plot_model_comparison()
    
    print("Model comparison completed!")
    return all_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sap Velocity Prediction Model Comparison')
    
    # Data options
    parser.add_argument('--data_path', type=str, default=None, 
                        help='Path to the data file (CSV)')
    parser.add_argument('--train_portion', type=float, default=0.8,
                        help='Portion of data to use for training (0-1)')
    
    # Model selection options
    parser.add_argument('--all', action='store_true',
                        help='Train all models')
    parser.add_argument('--xgboost', action='store_true',
                        help='Train XGBoost model')
    parser.add_argument('--random_forest', action='store_true',
                        help='Train Random Forest model')
    parser.add_argument('--svm', action='store_true',
                        help='Train SVM model')
    parser.add_argument('--neural_network', action='store_true',
                        help='Train Neural Network model')
    parser.add_argument('--lstm', action='store_true',
                        help='Train LSTM model')
    parser.add_argument('--transformer', action='store_true',
                        help='Train Transformer model')
    parser.add_argument('--cnn_lstm', action='store_true',
                        help='Train CNN-LSTM model')
    
    args = parser.parse_args()
    
    # If no model is specified, train all models
    if not (args.all or args.xgboost or args.random_forest or args.svm or 
            args.neural_network or args.lstm or args.transformer or args.cnn_lstm):
        args.all = True
    
    main(args)