"""
Deep Learning model with spatial cross-validation approach for site-based prediction.
Implements group-based spatial cross-validation with proper time windowing.
"""
from pathlib import Path
import sys
import os
import argparse
from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap

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
    set_seed, get_seed, get_initializer, deterministic,
    random_state_manager, test_determinism
)

# Import the time series windowing modules
from src.hyperparameter_optimization.timeseries_processor1 import (
    TimeSeriesSegmenter, WindowGenerator, TemporalWindowGenerator
)
from path_config import PathConfig, get_default_paths
# Set the master seed at the very beginning
set_seed(42)

# Now import all other dependencies
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import regularizers
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.model_selection import StratifiedGroupKFold
from typing import Union, List, Tuple, Dict, Any
from collections import Counter

import matplotlib as mpl
import logging
import warnings
import json
import random

# Apply additional TensorFlow determinism settings
tf.random.set_seed(42)
np.random.seed(42)

# Force TensorFlow to use deterministic algorithms (TF 2.6+)
try:
    tf.config.experimental.enable_op_determinism()
except:
    # Fallback for older TF versions
    logging.warning("Warning: Your TensorFlow version doesn't support enable_op_determinism()")
    # Try alternative approaches
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Limit TensorFlow to use only one thread for CPU operations
# n_physical_cores = os.cpu_count()
# tf.config.threading.set_inter_op_parallelism_threads(n_physical_cores)
# tf.config.threading.set_intra_op_parallelism_threads(n_physical_cores)


# Import the hyperparameter optimizer
from src.hyperparameter_optimization.hyper_tuner import DLOptimizer
from joblib import Parallel, delayed
import joblib
# Simple logging setup function
def setup_logging(logger_name,log_dir=None):
    """Set up basic logging configuration"""
    # set the logger name


    if log_dir is None:
        log_dir = Path('./outputs/logs')
    
    log_dir = Path(log_dir)
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f'{logger_name}_optimizer.log'

    # Configure logging to file and console with basic format
    handlers = [
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    return logging.getLogger(logger_name)
def process_date_file(data_file, args):
    logging.info(f"Processing {data_file.name}")
    try:
        # Unpack arguments
        used_cols = args['USED_COLS']
        all_possible_pft_types = args['ALL_POSSIBLE_PFT_TYPES']
        INPUT_WIDTH = args['INPUT_WIDTH']
        SHIFT = args['SHIFT']
        site_data_dict = {}
        site_info_dict = {}
        pft_types = set()
        df = pd.read_csv(data_file, parse_dates=['TIMESTAMP'])
        site_id = df['site_name'].iloc[0]
        time_scale = args['TIME_SCALE']
        # Extract lat/lon if available
        latitude = None
        longitude = None
        
        lat_col = next((col for col in df.columns if col.lower() in ['lat', 'latitude_x']), None)
        lon_col = next((col for col in df.columns if col.lower() in ['lon', 'longitude_x']), None)
        
        # Find the PFT column for stratification
        pft_col = next((col for col in df.columns if col.lower() in ['pft', 'plant_functional_type', 'biome']), None)
        if not (lat_col and lon_col and pft_col):
                logging.warning(f"  Warning: Lat/Lon/PFT not found for site {site_id}. Required for spatial stratified CV. Skipping.")
                return {}, {}, set()
        latitude = df[lat_col].median()
        longitude = df[lon_col].median()
        # skip CZE sites
        if site_id.startswith('CZE'):
            logging.info(f"  Skipping site {site_id} as it starts with CZE.")
            return {}, {}, set()
        # Add coordinates as features (constant per site)
        df['latitude'] = latitude
        df['longitude'] = longitude
        # Get the single PFT value for the site (using mode is robust)
        pft_value = df[pft_col].mode()[0] 
        
        # If lat/lon not found, raise an error
        if latitude is None or longitude is None:
            raise ValueError(f"Latitude or longitude not found in {data_file.name}. Please ensure the file contains lat/lon columns.")
        
        # Set index and add time features
        df.set_index('TIMESTAMP', inplace=True)
        df = add_time_features(df)
        # Fix potential case/whitespace issues
        df.columns = [col.strip().lower() for col in df.columns]
        used_cols = [col.strip().lower() for col in used_cols]
        # Verify columns exist
        missing_cols = [col for col in used_cols if col not in df.columns]
        if missing_cols:
            logging.warning(f"Warning: Missing columns in {data_file.name}: {missing_cols}")
            return {}, {}, set()
            

        
        # Select only the columns we need
        df = df[used_cols].copy()
        
        if 'pft' in used_cols:
            if 'pft' in df.columns:
                # Get non-pft columns
                orig_cols = [col for col in used_cols if col != 'pft']

                # Track which pft types are found in the data
                pft_types = df['pft'].unique()

                # Create a categorical type to ensure all possible pfts are represented
                pft_cat = pd.Categorical(df['pft'], categories=all_possible_pft_types)
                pft_df = pd.get_dummies(pft_cat)
                pft_df.index = df.index

                # Join with original data
                df = df[orig_cols].join(pft_df)
            else:
                logging.warning(f"Warning: Missing pft column in {data_file.name}")
                # skip the processing of this file
                return {}, {}, set()
        print(f"\n=== DEBUG for {site_id} ===")
        print("DataFrame dtypes:")
        print(df.dtypes)
        print("\nNon-numeric columns:")
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        print(non_numeric)
        if non_numeric:
            print("\nSample values from non-numeric columns:")
            for col in non_numeric:
                print(f"{col}: {df[col].head()}")
                print(f"  Unique values: {df[col].unique()[:10]}")
        # Now all columns should be numeric
        df = df.astype(np.float32)
        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # Calculate minimum required segment length
        window_size = INPUT_WIDTH + SHIFT
        min_segment_length = window_size + 7
        
        # Skip files that are too small
        if len(df) < min_segment_length:
            logging.warning(f"Warning: {data_file.name} has too few records after cleaning: {len(df)}")
            return {}, {}, set()
        
        # Segment the data
        segments = TimeSeriesSegmenter.segment_time_series(
            df, 
            gap_threshold=2, 
            unit='hours' if time_scale == 'hourly' else 'days', 
            min_segment_length=min_segment_length
        )
        
        # Count total data points (considering segments)
        data_count = sum(len(segment) for segment in segments)
        logging.info(f"  Created {len(segments)} segments for site {site_id} with {data_count} total points")
        # Add site data to dictionary
        if site_id not in site_data_dict:
            site_data_dict[site_id] = []
            site_info_dict[site_id] = {
                'latitude': latitude,
                'longitude': longitude,
                'data_count': 0,
                'segments': 0,
                'pft': pft_value # Store the PFT value
            }
        site_data_dict[site_id].extend(segments)
        site_info_dict[site_id]['data_count'] += data_count
        site_info_dict[site_id]['segments'] += len(segments)
        return site_data_dict, site_info_dict, pft_types
        
    except Exception as e:
        logging.error(f"Error processing {data_file.name}: {e}")

def determine_model_creation_function(model_type: str):
    """
    Determine the model creation function based on the model type.
    
    Parameters:
    -----------
    model_type : str
        Type of model to create ('ANN', 'CNN-LSTM', 'LSTM', etc.)

    Returns:
    --------
    function
        The appropriate model creation function
    """
    if model_type == 'ANN':
        return create_windowed_nn
    elif model_type == 'CNN-LSTM':
        return create_cnn_lstm_model
    elif model_type == 'LSTM':
        return create_lstm_model
    else:
        raise ValueError(f"Unknown model type: {model_type}")
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
    seed = get_seed()
    tf.random.set_seed(seed)
    
    # Create incrementing seeds for each layer to ensure independence
    kernel_seeds = [seed + i for i in range(100)]
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
@deterministic
def create_cnn_lstm_model(input_shape, output_shape, cnn_layers=2, lstm_layers=2, 
                         cnn_filters=64, lstm_units=128, dropout_rate=0.3):
    """Create a hybrid CNN-LSTM model with explicit seeds for all random operations."""
    # Force deterministic behavior for this function
    seed = get_seed()
    
    # Create incrementing seeds for each layer to ensure independence
    kernel_seeds = [seed + i for i in range(100)]
    seed_idx = 0
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Reshape for CNN
    x = tf.keras.layers.Reshape((input_shape[0], input_shape[1], 1))(inputs)
    
    # CNN Block with explicit seeds
    for i in range(cnn_layers):
        kernel_size = 5 if i == 0 else 3
        
        # Use explicit seed for each layer
        x = tf.keras.layers.Conv2D(
            filters=cnn_filters * (2**i),
            kernel_size=(kernel_size, 1),
            padding='same',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seeds[seed_idx]),
            bias_initializer=tf.keras.initializers.Zeros()
        )(x)
        seed_idx += 1
        
        x = tf.keras.layers.BatchNormalization()(x)
        
        if i < cnn_layers - 1:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), padding='same')(x)

    cnn_output_shape = x.shape
    
    # We flatten the original features and the new CNN-generated filters into a single feature vector.
    new_lstm_features = cnn_output_shape[2] * cnn_output_shape[3]
    
    # Reshape to (batch, new_timesteps, new_lstm_features) which is what the LSTM expects.
    x = tf.keras.layers.Reshape((cnn_output_shape[1], new_lstm_features))(x)
    
    # LSTM Block with explicit seeds
    for i in range(lstm_layers):
        return_sequences = i < lstm_layers - 1
        
        # Use explicit seed for each LSTM layer
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                lstm_units,
                return_sequences=return_sequences,
                dropout=0.0,
                recurrent_dropout=0.2,
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seeds[seed_idx]),
                recurrent_initializer=tf.keras.initializers.Orthogonal(seed=kernel_seeds[seed_idx+1]),
                bias_initializer=tf.keras.initializers.Zeros()
            )
        )(x)
        seed_idx += 2
        
        # Dropout with explicit seed
        x = tf.keras.layers.Dropout(
            dropout_rate, 
            seed=kernel_seeds[seed_idx]
        )(x)
        seed_idx += 1
    
    # Dense layer with explicit seed
    x = tf.keras.layers.Dense(
        lstm_units // 2,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seeds[seed_idx]),
        bias_initializer=tf.keras.initializers.Zeros()
    )(x)
    seed_idx += 1
    
    # Output layer with explicit seed
    outputs = tf.keras.layers.Dense(
        output_shape,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seeds[seed_idx]),
        bias_initializer=tf.keras.initializers.Zeros()
    )(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    
    
    return model
def create_windowed_nn(input_shape, output_shape, n_layers, units, dropout_rate):
    """
    Create a neural network model that takes a time window as input,
    with explicit seeds for all random operations.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of the input data (timesteps, features)
    output_shape : int
        Number of output neurons
    n_layers : int
        Number of hidden layers
    units : int
        Number of neurons per hidden layer
    dropout_rate : float
        Dropout rate for regularization
        
    Returns:
    --------
    model : keras.Model
        The compiled neural network model
    """
    # Force deterministic behavior by setting seed
    seed = get_seed()
    
    # Create incrementing seeds for each layer to ensure independence
    kernel_seeds = [seed + i for i in range(100)]
    seed_idx = 0
    
    # Input layer with deterministic settings
    # Input shape is now (timesteps, features) instead of just (features)
    inputs = keras.layers.Input(shape=input_shape)
    
    # Flatten the time window 
    x = keras.layers.Flatten()(inputs)
    
    # Hidden layers with explicit seeds for weight initialization
    for i in range(n_layers):
        x = keras.layers.Dense(
            units,
            activation='relu',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seeds[seed_idx]),
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(x)
        seed_idx += 1
        
        # Add dropout with explicit seed
        if dropout_rate > 0:
            x = keras.layers.Dropout(dropout_rate, seed=kernel_seeds[seed_idx])(x)
            seed_idx += 1
    
    # Output layer with explicit seed
    outputs = keras.layers.Dense(
        output_shape,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seeds[seed_idx]),
        bias_initializer=tf.keras.initializers.Zeros()
    )(x)
    
    # Create and return the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def add_time_features(df, datetime_column='solar_TIMESTAMP'):
    """
    Create cyclical time features from a datetime column or index.
    """
    df = df.copy()
    
    if datetime_column is not None:
        if datetime_column not in df.columns:
            raise ValueError(f"Column '{datetime_column}' not found in DataFrame")
        
        # Drop rows with NaN in the datetime column
        if df[datetime_column].isna().any():
            n_nulls = df[datetime_column].isna().sum()
            logging.warning(f"Dropping {n_nulls} rows with NaN in {datetime_column}")
            df = df.dropna(subset=[datetime_column])
        
        date_time = pd.to_datetime(df[datetime_column])
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                date_time = pd.to_datetime(df.index)
            except:
                raise ValueError("DataFrame index cannot be converted to datetime")
        else:
            date_time = df.index

    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = 365.2425 * day
    week = 7 * day
    month = 30.44 * day
    
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    df['Week sin'] = np.sin(timestamp_s * (2 * np.pi / week))
    df['Week cos'] = np.cos(timestamp_s * (2 * np.pi / week))
    df['Month sin'] = np.sin(timestamp_s * (2 * np.pi / month))
    df['Month cos'] = np.cos(timestamp_s * (2 * np.pi / month))
    
    return df



@deterministic
def create_windows_from_segments(
    segments: List[pd.DataFrame],
    input_width: int,
    label_width: int,
    shift: int,
    label_columns: List[str] = None,
    exclude_targets_from_features: bool = True,
    exclude_labels_from_inputs: bool = True
) -> List[Tuple[tf.Tensor, tf.Tensor]]:
    """
    Create windows from multiple segments without splitting them.
    Each segment is processed independently to maintain time series integrity.
    
    Parameters:
    -----------
    segments : list
        List of DataFrame segments
    input_width : int
        Width of input window
    label_width : int
        Width of label window
    shift : int
        Offset between input and label
    label_columns : list of str, optional
        Columns to use as labels
    exclude_targets_from_features : bool
        Whether to exclude label columns from inputs
    exclude_labels_from_inputs : bool
        Whether to exclude labels from inputs 
        
    Returns:
    --------
    list
        List of (inputs, labels) windows
    """
    all_windows = []
    
    # Process each segment independently
    for segment in segments:
        # Skip segments that are too short
        min_segment_length = input_width + shift
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
            exclude_targets_from_features=exclude_targets_from_features,
            exclude_labels_from_inputs=exclude_labels_from_inputs
        )
        
        # Collect all windows from this segment
        segment_ds = window_gen.dataset
        for inputs, labels in segment_ds:
            all_windows.append((inputs, labels))
    
    return all_windows



@deterministic
def get_predictions(model, dataset, scaler):
    """Get predictions from a windowed dataset and inverse transform them."""
    
    # 1. Extract all features and labels from the dataset
    all_features = np.concatenate([features.numpy() for features, _ in dataset], axis=0)
    all_labels = np.concatenate([labels.numpy() for _, labels in dataset], axis=0)
    
    # 2. Pass the entire dataset to model.predict() once
    all_predictions = model.predict(all_features, verbose=0)
    
    # 3. Handle shapes for single-step or multi-step predictions
    if len(all_predictions.shape) == 3:  # Shape is [batch, time_steps, features]
        all_predictions = all_predictions[:, -1, :]  # Get the last time step
        
    if len(all_labels.shape) == 3:  # Shape is [batch, time_steps, features]
        all_labels = all_labels[:, -1, :]  # Get the last time step
        
    # 4. Reshape for inverse transformation
    all_predictions = all_predictions.reshape(-1, 1)
    all_labels = all_labels.reshape(-1, 1)
    
    # 5. Inverse transform to get the original scale.
    all_predictions = scaler.inverse_transform(all_predictions)
    all_labels = scaler.inverse_transform(all_labels)
    
    # 6. Ensure both arrays have the same length
    min_length = min(len(all_predictions), len(all_labels))
    all_predictions = all_predictions[:min_length]
    all_labels = all_labels[:min_length]
    
    try:
        # This is generally not needed when not in a debugging loop,
        # but kept for consistency with the original function.
        tf.config.run_functions_eagerly(False)
    except:
        pass
    
    return all_predictions.flatten(), all_labels.flatten()

@deterministic
def create_spatial_groups(
    lat: Union[np.ndarray, List],
    lon: Union[np.ndarray, List],
    data_counts: Union[np.ndarray, List] = None,
    n_groups: int = None,
    balanced: bool = True,
    method: str = 'grid',
    lat_grid_size: float = 0.05,
    lon_grid_size: float = 0.05,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Group geographical data points using various spatial grouping methods.
    Enhanced to consider data counts for balanced grouping.
    
    Parameters
    ----------
    lat : array-like
        Latitude values
    lon : array-like
        Longitude values
    data_counts : array-like
        Number of data points at each (lat, lon) location
    n_groups : int, optional
        Number of groups to form
    balanced : bool, default=True
        Whether to balance groups based on data counts
    method : str, default='grid'
        Clustering method to use: 'grid'
    lat_grid_size : float, default=5.0
        Size of latitude intervals for grid method (in degrees)
    lon_grid_size : float, default=5.0
        Size of longitude intervals for grid method (in degrees)
    random_state : int, default=42
        Random state for reproducibility
        
    Returns
    -------
    groups : np.ndarray
        Array of group labels
    stats : pd.DataFrame
        DataFrame containing statistics for each group
    """
    # Convert inputs to numpy arrays
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    
    
    if method == 'grid':
        # Create grid cells
        lat_bins = np.arange(
            np.floor(lat.min()),
            np.ceil(lat.max()) + lat_grid_size,
            lat_grid_size
        )
        lon_bins = np.arange(
            np.floor(lon.min()),
            np.ceil(lon.max()) + lon_grid_size,
            lon_grid_size
        )
        
        # Assign points to grid cells
        lat_indices = np.digitize(lat, lat_bins) - 1
        lon_indices = np.digitize(lon, lon_bins) - 1
        
        # Create unique group numbers for each grid cell
        n_lon_bins = len(lon_bins)
        grid_cell_ids = lat_indices * n_lon_bins + lon_indices
        if data_counts is None or n_groups is None or not balanced:
            # If data counts are provided, we can try to balance groups
            unique_groups = np.unique(grid_cell_ids)
            group_map = {g: n for n, g in enumerate(unique_groups)}
            groups = np.array([group_map[g] for g in grid_cell_ids])
        else:
        
            # Calculate the total data count per grid cell
            unique_cells, cell_indices = np.unique(grid_cell_ids, return_inverse=True)
            cell_data_counts = np.bincount(cell_indices, weights=data_counts)

            # Sort cells by data count in descending order
            sorted_cell_indices = np.argsort(cell_data_counts)[::-1]

            # Greedily assign cells to folds to balance data counts
            fold_data_counts = np.zeros(n_groups)
            cell_to_fold = np.zeros(len(unique_cells), dtype=int)
            for cell_idx in sorted_cell_indices:
                # Assign the current cell to the fold with the minimum total data count
                target_fold = np.argmin(fold_data_counts)
                cell_to_fold[cell_idx] = target_fold
                fold_data_counts[target_fold] += cell_data_counts[cell_idx]

            # Map the fold assignment back to each site
            groups = cell_to_fold[cell_indices]
    elif method == 'default':
        # Randomly assign points into groups, ensuring close points are in same group
        if not len(lat) == len(lon):
            raise ValueError("Latitude and longitude arrays must have the same length.")
        if data_counts is None or n_groups is None or not balanced:
            groups = np.arange(len(lat))
        else:
            # Sort points by data counts (descending)
            sorted_indices = np.argsort(data_counts)[::-1]
            groups = np.zeros(len(lat), dtype=int) - 1  # Initialize all to -1 (unassigned)
            fold_data_counts = np.zeros(n_groups)
            
            for idx in sorted_indices:
                # Assign to the group with the least total data count
                target_group = np.argmin(fold_data_counts)
                groups[idx] = target_group
                fold_data_counts[target_group] += data_counts[idx]
        '''
        n_points = len(lat)
        
        # Calculate distances between points
        from scipy.spatial.distance import cdist
        coords = np.column_stack((lat, lon))
        distances = cdist(coords, coords, metric='euclidean')
        
        # Create adjacency matrix for close points
        close_mask = distances < 0.05
        
        # Use connected components to find groups of close points
        from scipy.sparse.csgraph import connected_components
        n_components, labels = connected_components(close_mask, directed=False)
        
        # Randomly assign cluster IDs to each connected component
        component_to_cluster = np.random.randint(0, n_clusters, size=n_components)
        groups = component_to_cluster[labels]
        '''
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods are 'kmeans', 'dbscan', 'grid', or 'default'.")

    # Calculate group statistics including data counts
    stats = []
    for group in np.unique(groups):
        if group == -1:  # Skip DBSCAN noise points
            continue
            
        mask = groups == group
        stats.append({
            'group': group,
            'size': np.sum(mask),
            'mean_lat': np.mean(lat[mask]),
            'mean_lon': np.mean(lon[mask]),
            'std_lat': np.std(lat[mask]),
            'std_lon': np.std(lon[mask]),
            'min_lat': np.min(lat[mask]),
            'max_lat': np.max(lat[mask]),
            'min_lon': np.min(lon[mask]),
            'max_lon': np.max(lon[mask])
        })
    
    stats_df = pd.DataFrame(stats)
    
    #Log summary of data distribution
    if len(stats_df) > 0:
        logging.info("Spatial Group Statistics:")
        logging.info(f"Total groups: {len(stats_df)}")
        
        logging.info(f"Group details:\n{stats_df[['group', 'size', 'mean_lat', 'mean_lon']].to_string(index=False)}")
    return groups, stats_df


@deterministic
def convert_windows_to_numpy(windows: List[Tuple[tf.Tensor, tf.Tensor]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of windows to numpy arrays for features and labels.
    
    Parameters:
    -----------
    windows : list
        List of window tuples (inputs, labels) 
        
    Returns:
    --------
    X : np.ndarray
        Features array with shape [n_samples, input_width, n_features]
    y : np.ndarray
        Labels array with shape [n_samples, label_width] or [n_samples, label_width, n_label_features]
    """
    if not windows:
        return np.array([]), np.array([])
    
    # Extract features and labels from windows
    features = []
    labels = []
    
    for inputs, lbls in windows:
        # Convert to numpy
        features.append(inputs.numpy())
        labels.append(lbls.numpy())
    
    # Concatenate along batch dimension (dim 0)
    X = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)
    
    # If labels are 3D and we only have one label feature, flatten to 2D
    if y.ndim == 3 and y.shape[2] == 1:
        y = y.reshape(y.shape[0], y.shape[1])
    
    return X, y

def parse_args():
    """
    Parse command line arguments for the script.
    
    Returns:
    -------- 
    argparse.Namespace
        Parsed command line arguments
    """
    
    
    parser = argparse.ArgumentParser(description="Train a Deep Learning model with leave-fold-out cross-validation.")
    parser.add_argument('--model', type=str, default='CNN-LSTM', help='Model type to train (default: ANN)')
    parser.add_argument('--RANDOM_SEED', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--BATCH_SIZE', type=int, default=32, help='Batch size for training')
    parser.add_argument('--TARGET_COL', type=str, default='sap_velocity', help='Target column for prediction')
    parser.add_argument('--INPUT_WIDTH', type=int, default=1, help='Input width for time series windows')
    parser.add_argument('--LABEL_WIDTH', type=int, default=1, help='Label width for time series windows')
    parser.add_argument('--SHIFT', type=int, default=1, help='Shift for time series windows')
    parser.add_argument('--EXCLUDE_LABELS', type=bool, default=True, help='Exclude labels from input features')
    parser.add_argument('--EXCLUDE_TARGETS', type=bool, default=True, help='Exclude targets from input features')
    parser.add_argument('--TUNING_WORKERS', type=int, default=1, help='Number of workers for hyperparameter tuning')
    parser.add_argument('--n_groups', type=int, default=10, help='Number of spatial groups for cross-validation')
    parser.add_argument('--spatial_split_method', type=str, default='default', help='Method for spatial splitting of data')
    parser.add_argument('--IS_CV', type=bool, default=False, help='Whether to use cross-validation')
    parser.add_argument('--IS_SHUFFLE', type=bool, default=True, help='Whether to shuffle the data')
    parser.add_argument('--IS_STRATIFIED', type=bool, default=True, help='Whether to use stratified sampling')
    parser.add_argument('--BALANCED', type=bool, default=False, help='Whether to balance the spatial groups')
    # add arguments for hyperparameters
    parser.add_argument('--hyperparameters', type=str, help='Path to the JSON file of hyperparameters') 
    parser.add_argument('--additional_features', nargs='*', default=[], help='List of features')
    parser.add_argument('--TIME_SCALE', type=str, default='daily', help='Time scale of the data: hourly or daily')
    parser.add_argument('--run_id', type=str, default='default_daily', help='Unique identifier for this run (default: {model_type}_run_{RANDOM_SEED})')
    return parser.parse_args()

@deterministic
def main():
    """
    Main function implementing the paper's leave-fold-out CV approach.
    Each fold serves as test set exactly once, with remaining folds split 8:1 for train:validation.
    """
    # unpack arguments
    args = parse_args()
    RANDOM_SEED = args.RANDOM_SEED
    BATCH_SIZE = args.BATCH_SIZE
    INPUT_WIDTH = args.INPUT_WIDTH
    LABEL_WIDTH = args.LABEL_WIDTH
    TARGET_COL = args.TARGET_COL
    SHIFT = args.SHIFT
    EXCLUDE_LABELS = args.EXCLUDE_LABELS
    EXCLUDE_TARGETS = args.EXCLUDE_TARGETS
    TUNING_WORKERS = args.TUNING_WORKERS
    IS_CV = args.IS_CV
    IS_SHUFFLE = args.IS_SHUFFLE
    n_groups = args.n_groups
    spatial_split_method = args.spatial_split_method
    model_type = args.model
    IS_STRATIFIED = args.IS_STRATIFIED
    BALANCED = args.BALANCED
    
    # Use provided run_id or generate default
    if args.run_id is not None:
        run_id = args.run_id
    else:
        run_id = f"{model_type}_run_{RANDOM_SEED}"
    
    paths = PathConfig(scale='sapwood')
    TIME_SCALE = args.TIME_SCALE
    additional_features = args.additional_features
    # Open the file and load the JSON directly from it
    with open(args.hyperparameters, 'r') as f:
        hyperparameters = json.load(f)
    
    # Set deterministic rendering for matplotlib
    mpl.rcParams['agg.path.chunksize'] = 10000
    np.random.seed(RANDOM_SEED)
    # setup logging
    setup_logging(logger_name=model_type)
    logging.info(f"Starting {model_type} model training with leave-fold-out CV, seed {RANDOM_SEED}")
    logging.info(f"Run ID: {run_id}")
    logging.info(f"Using {n_groups} spatial groups with leave-fold-out cross-validation")
    logging.info(f"batch_size={BATCH_SIZE}, input_width={INPUT_WIDTH}, label_width={LABEL_WIDTH}, shift={SHIFT}, exclude_labels={EXCLUDE_LABELS}, exclude_targets={EXCLUDE_TARGETS}, shuffle strategy={IS_SHUFFLE}, spatial_split_method={spatial_split_method}, IS_STRATIFIED={IS_STRATIFIED}, model_type={model_type}")

    # Create output directories
    plot_dir = paths.hyper_tuning_plots_dir / model_type / run_id
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = paths.models_root / model_type / run_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    data_dir = paths.merged_data_root / TIME_SCALE
    data_list = list(data_dir.glob(f'*{TIME_SCALE}.csv'))
    
    if not data_list:
        logging.critical("No data files found in the specified directory. Attempting fallback to single file mode.")
        raise FileNotFoundError("No data files found in the specified directory. Please check the path.")
    else:
        logging.info(f"Found {len(data_list)} data files")

    # Process data files
    site_data_dict = {}  # Dictionary to store data segments by site
    site_info_dict = {}  # Dictionary to store site metadata
    base_features = [
        TARGET_COL, 
        'sw_in', 'ppfd_in', 'ext_rad', 'ws', 'ta', 'vpd', 
        'LAI', 'prcip/PET', 'pft', 'canopy_height', 
        'elevation', 'latitude', 'longitude', 'Day sin', 'Year sin'
    ]
    # Define the columns we want to use
    used_cols = list(set(base_features + additional_features))
    logging.info(f"Using columns: {used_cols}")
    all_pft_types = set()
    all_possible_pft_types = ['MF', 'DNF', 'ENF', 'EBF', 'WSA', 'WET', 'DBF', 'SAV']
    
    # Sort data files for deterministic processing order
    data_list = sorted(data_list)
    args_dict = {
        'USED_COLS': used_cols,
        'ALL_POSSIBLE_PFT_TYPES': all_possible_pft_types,
        'INPUT_WIDTH': INPUT_WIDTH,
        'SHIFT': SHIFT,
        'ALL_PFT_TYPES': all_pft_types,
        'TUNING_WORKERS': TUNING_WORKERS,
        'TIME_SCALE': TIME_SCALE,
    }
    results = []  # Store results from each file
    results = Parallel(n_jobs=5)(delayed(process_date_file)(file, args_dict) for file in data_list)
    # Now, aggregate the collected results in the main process
    site_data_dict = {}
    site_info_dict = {}
    all_pft_types = set()

    for site_data, site_info, pft_types in results:
        site_data_dict.update(site_data)
        site_info_dict.update(site_info)
        all_pft_types.update(pft_types)
    if not site_data_dict:
        logging.critical("No valid sites found. Please check your input files.")
        raise ValueError("No valid site data found. Please check your input files.")

    # Create a DataFrame of site information
    site_info_df = pd.DataFrame.from_dict(site_info_dict, orient='index')
    logging.info(f"\nSite information:")
    logging.info(f"{site_info_df}")

    # Extract spatial information for grouping
    site_ids = list(site_data_dict.keys())
    latitudes = [site_info_dict[site]['latitude'] for site in site_ids]
    longitudes = [site_info_dict[site]['longitude'] for site in site_ids]
    data_counts = [site_info_dict[site]['data_count'] for site in site_ids]
    # Create spatial groups
    logging.info("\nCreating spatial groups...")
    if IS_STRATIFIED:
        spatial_groups, group_stats = create_spatial_groups(
            lat=latitudes,
            lon=longitudes,
            method=spatial_split_method,
        )
    else:
        spatial_groups, group_stats = create_spatial_groups(
            lat=latitudes,
            lon=longitudes,
            data_counts=data_counts,
            n_groups=n_groups,
            method=spatial_split_method,
            balanced=BALANCED,
        )
    

    logging.info(f"Spatial groups assigned: {np.unique(spatial_groups)}")

    # Create a mapping for easy lookup
    site_to_group = {site_id: group for site_id, group in zip(site_ids, spatial_groups)}
    site_to_pft = {site_id: site_info_dict[site_id]['pft'] for site_id in site_ids}
    # plot spatial grouping
    plt.figure(figsize=(20, 6))
  
    # Extract colors from the colormap first, then create a new colormap
    colors = plt.cm.tab20.colors[:n_groups]  # Access the colors attribute
    color_map = ListedColormap(colors)  # Create new colormap with just those colors
    scatter = plt.scatter(longitudes, latitudes, c=spatial_groups, s=100, edgecolor='k', cmap=color_map)
    plt.colorbar(scatter, label='Spatial Group')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Spatial Grouping of Sites (Run: {run_id})')
    plt.grid(True)
    plt.show()
    plt.savefig(plot_dir / f'spatial_grouping_{run_id}.png')
    plt.close()
    # Logging mapping for reference
    logging.info("\nSite to group mapping:")
    for site, group in sorted(site_to_group.items()):
        logging.info(f"  Site {site}: Group {group}")

    segment_cols = list(site_data_dict.values())[0][0].columns.tolist()

    # Get feature columns
    if 'pft' in used_cols:
        numerical_cols = [col for col in segment_cols if col != 'pft' and col != TARGET_COL]
        categorical_cols = all_possible_pft_types
        feature_columns = numerical_cols + categorical_cols
        # Get the indices
        numerical_indices = [feature_columns.index(col) for col in numerical_cols]
        categorical_indices = [feature_columns.index(col) for col in categorical_cols]
    else:
        numerical_cols = [col for col in segment_cols if col != TARGET_COL]
        feature_columns = numerical_cols
        numerical_indices = [feature_columns.index(col) for col in numerical_cols]
        categorical_indices = []
    
    logging.info(f"Identified {len(numerical_indices)} numerical and {len(categorical_indices)} categorical feature indices.")

    
    # This block processes all data into unified, record-level numpy arrays.
    # This is the core change to enable record-level splitting.
    logging.info("Preparing final RECORD-LEVEL data structures for all sites...")
    
    # Use lists for efficient appending
    list_X, list_y, list_groups, list_pfts = [], [], [], []

    # This single loop creates all the data needed for the entire CV process.
    for site_id, raw_data in site_data_dict.items():
        windows = create_windows_from_segments(
            segments=raw_data, input_width=INPUT_WIDTH, label_width=LABEL_WIDTH, shift=SHIFT,
            label_columns=[TARGET_COL], exclude_targets_from_features=EXCLUDE_TARGETS,
            exclude_labels_from_inputs=EXCLUDE_LABELS
        )
        if not windows:
            continue
        X_site, y_site = convert_windows_to_numpy(windows)
        

        num_records = len(y_site)
        if num_records == 0:
            continue
            
        list_X.append(X_site)
        list_y.append(y_site)
        
        # For each record from this site, store its spatial group and PFT
        list_groups.append(np.full(num_records, site_to_group[site_id]))
        list_pfts.append(np.full(num_records, site_to_pft[site_id]))

    # Release memory
    del site_data_dict

    # Concatenate lists into final NumPy arrays
    if not list_X:
        logging.critical("ERROR: No data records were generated after processing. Exiting.")
        return [], []
        
    X_all_records = np.vstack(list_X)
    y_all_records = np.concatenate(list_y)
    groups_all_records = np.concatenate(list_groups)
    pfts_all_records = np.concatenate(list_pfts)

    logging.info(f"Total records processed and ready for CV: {len(y_all_records)}")
    # Encode the PFT strings into integers
    pfts_encoded, pft_categories = pd.factorize(pfts_all_records)
    logging.info(f"Encoded PFTs into {len(pft_categories)} integer classes for stratification.")
    # --- Stratified Group K-Fold Cross-Validation ---
    logging.info("\nInitializing K-Fold Cross-Validation at RECORD LEVEL...")
    if IS_STRATIFIED:
        outer_cv = StratifiedGroupKFold(n_splits=n_groups, shuffle=True, random_state=RANDOM_SEED)
        inner_cv = StratifiedGroupKFold(n_splits=n_groups - 1, shuffle=True, random_state=RANDOM_SEED)
        y_all_stratified = pfts_encoded  # Use PFTs for stratification
        logging.info(f"Using StratifiedGroupKFold with {n_groups} splits, stratifying by PFTs.")
    else:
        outer_cv = GroupKFold(n_splits=n_groups)
        inner_cv = GroupKFold(n_splits=n_groups - 1)
        y_all_stratified = y_all_records  # Use the target variable for grouping
        logging.info(f"Using GroupKFold with {n_groups} splits, grouping by spatial groups.")
    # Get unique group numbers


    # Storage for results from all iterations
    all_test_r2_scores = []
    all_test_rmse_scores = []
    all_test_mae_scores = []
    fold_results = []
    all_predictions = []
    all_actuals = []


    # Prepare a list of arguments for each fold
    # The splitter is now called with the record-level arrays. It will respect groups
    split_generator = outer_cv.split(X_all_records, y_all_stratified, groups_all_records)

    for test_fold_idx, (train_val_indices, test_indices) in enumerate(split_generator):
        logging.info(f"\n=== Fold {test_fold_idx + 1}/{n_groups} ===")

        # --- Simplified Data Collection via Direct Slicing ---
        # The splitter gives us indices that directly slice our master arrays.

        X_train_val = X_all_records[train_val_indices]
        y_train_val = y_all_records[train_val_indices]
        pfts_train_val = pfts_all_records[train_val_indices]
        groups_train_val = groups_all_records[train_val_indices]
        
        X_test = X_all_records[test_indices]
        y_test = y_all_records[test_indices]

        if IS_SHUFFLE:
            shuffled_indices = np.random.permutation(len(X_train_val))
            X_train_val, y_train_val = X_train_val[shuffled_indices], y_train_val[shuffled_indices]
            pfts_train_val, groups_train_val = pfts_train_val[shuffled_indices], groups_train_val[shuffled_indices]

        logging.info(f"  Train/Val records: {len(y_train_val)}, Test records: {len(y_test)}")
        train_idx, val_idx = next(inner_cv.split(X_train_val, pfts_train_val, groups_train_val))
        
        X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
        X_val, y_val = X_train_val[val_idx], y_train_val[val_idx]
        # --- SCALING (Fit on Train, Transform All, Numerical features Only) ---
        # Reshape 3D data to 2D for scaler
        # 1. Separate numerical and categorical features
        X_train_num = X_train[:, :, numerical_indices]
        X_train_cat = X_train[:, :, categorical_indices]

        X_val_num = X_val[:, :, numerical_indices]
        X_val_cat = X_val[:, :, categorical_indices]

        X_test_num = X_test[:, :, numerical_indices]
        X_test_cat = X_test[:, :, categorical_indices]

        # 2. Reshape numerical data for the scaler (from 3D to 2D)
        _, _, n_num_features = X_train_num.shape
        X_train_num_reshaped = X_train_num.reshape(-1, n_num_features)

        # 3. Fit the scaler ONLY on the training numerical data
        feature_scaler = StandardScaler()
        X_train_num_scaled_reshaped = feature_scaler.fit_transform(X_train_num_reshaped)

        # 4. Reshape the scaled numerical data back to 3D
        X_train_num_scaled = X_train_num_scaled_reshaped.reshape(X_train_num.shape)

        # 5. Transform val and test numerical data (and reshape back)
        X_val_num_scaled = feature_scaler.transform(X_val_num.reshape(-1, n_num_features)).reshape(X_val_num.shape)
        X_test_num_scaled = feature_scaler.transform(X_test_num.reshape(-1, n_num_features)).reshape(X_test_num.shape)

        # 6. Recombine scaled numerical features with original categorical features
        X_train_scaled = np.concatenate([X_train_num_scaled, X_train_cat], axis=2)
        X_val_scaled = np.concatenate([X_val_num_scaled, X_val_cat], axis=2)
        X_test_scaled = np.concatenate([X_test_num_scaled, X_test_cat], axis=2)

        # Scale labels (this part remains the same)
        label_scaler = StandardScaler()
        y_train_scaled = label_scaler.fit_transform(y_train)
        y_val_scaled = label_scaler.transform(y_val)
        y_test_scaled = label_scaler.transform(y_test)
        
        logging.info(f"  Data scaled. Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
        feature_scaler_path = paths.feature_scaler_path
        label_scaler_path = paths.label_scaler_path
        # Create directory if it doesn't exist
        if not feature_scaler_path.parent.exists():
            feature_scaler_path.parent.mkdir(parents=True, exist_ok=True)
        if not label_scaler_path.parent.exists():
            label_scaler_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(feature_scaler, feature_scaler_path)
        joblib.dump(label_scaler, label_scaler_path)
        logging.info(f"  Feature scaler saved to: {feature_scaler_path}")
        logging.info(f"  Label scaler saved to: {label_scaler_path}")


        input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
        test_ds = tf.data.Dataset.from_tensor_slices((X_test_scaled, y_test_scaled)).batch(32)
        
        # 7. Train the model for this fold
        logging.info(f"  Training model for fold {test_fold_idx + 1}...")
        optimizer = DLOptimizer(
            base_architecture=determine_model_creation_function(model_type),
            task='regression',
            model_type=model_type,
            param_grid=hyperparameters,
            input_shape=input_shape,
            output_shape=LABEL_WIDTH,
            scoring='val_loss',
            random_state=RANDOM_SEED + test_fold_idx
        )

        optimizer.fit(X_train_scaled, y_train_scaled, is_cv=IS_CV, X_val=X_val_scaled, y_val=y_val_scaled)
        best_model = optimizer.get_best_model()

        # 8. Get predictions and inverse-transform them
        test_predictions, test_labels_actual = get_predictions(best_model, test_ds, label_scaler)
        
        # 9. Calculate and report performance metrics
        fold_test_r2 = r2_score(test_labels_actual, test_predictions)
        fold_test_rmse = np.sqrt(mean_squared_error(test_labels_actual, test_predictions))
        fold_test_mae = mean_absolute_error(test_labels_actual, test_predictions)
        logging.info(f"  Fold {test_fold_idx + 1} Results: R²={fold_test_r2:.4f}, RMSE={fold_test_rmse:.4f}, MAE={fold_test_mae:.4f}")

        logging.info(f"  Generating plots for Fold {test_fold_idx + 1}...")

        # --- Plot 1: Predictions vs Actuals + Residuals (side by side) ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Predictions vs Actuals
        axes[0].scatter(test_labels_actual, test_predictions, alpha=0.5, s=20, 
                       color='steelblue', edgecolors='black', linewidth=0.5)
        axes[0].set_xlabel('Observed Sap Velocity', fontsize=13)
        axes[0].set_ylabel('Predicted Sap Velocity', fontsize=13)
        axes[0].set_title(f'Fold {test_fold_idx + 1}: Predictions vs Actuals\n'
                         f'$R^2 = {fold_test_r2:.3f}$, RMSE = ${fold_test_rmse:.3f}$', 
                         fontsize=13, fontweight='bold')

        # Add 1:1 line
        min_val = min(test_labels_actual.min(), test_predictions.min())
        max_val = max(test_labels_actual.max(), test_predictions.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r-', alpha=0.8, 
                    linewidth=2, label='Perfect Prediction')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect('equal', 'box')

        # Right: Residual Plot
        residuals = test_labels_actual - test_predictions
        axes[1].scatter(test_predictions, residuals, alpha=0.5, s=20, 
                       color='coral', edgecolors='black', linewidth=0.5)
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
        axes[1].set_xlabel('Predicted Sap Velocity', fontsize=13)
        axes[1].set_ylabel('Residuals (Observed - Predicted)', fontsize=13)
        axes[1].set_title(f'Fold {test_fold_idx + 1}: Residual Plot\n'
                         f'MAE = ${fold_test_mae:.3f}$', 
                         fontsize=13, fontweight='bold')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fold_plot_path = plot_dir / f'fold_{test_fold_idx + 1}_performance_{run_id}.png'
        plt.savefig(fold_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"    Saved: {fold_plot_path}")

        # --- Plot 2: Distribution Comparison + Q-Q Plot ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Distribution comparison
        axes[0].hist(test_labels_actual, bins=50, alpha=0.6, label='Observed', 
                    color='blue', edgecolor='black')
        axes[0].hist(test_predictions, bins=50, alpha=0.6, label='Predicted', 
                    color='orange', edgecolor='black')
        axes[0].set_xlabel('Sap Velocity', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'Fold {test_fold_idx + 1}: Distribution Comparison', 
                         fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # Right: Q-Q plot of residuals
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title(f'Fold {test_fold_idx + 1}: Q-Q Plot of Residuals', 
                         fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        dist_plot_path = plot_dir / f'fold_{test_fold_idx + 1}_distributions_{run_id}.png'
        plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"    Saved: {dist_plot_path}")

        # --- Save fold predictions to CSV ---
        fold_predictions_df = pd.DataFrame({
            'Observed': test_labels_actual,
            'Predicted': test_predictions,
            'Residual': residuals,
            'Absolute_Error': np.abs(residuals)
        })
        fold_csv_path = plot_dir / f'fold_{test_fold_idx + 1}_predictions_{run_id}.csv'
        fold_predictions_df.to_csv(fold_csv_path, index=False)
        logging.info(f"    Saved: {fold_csv_path}")
        # 10. Save model and assemble results
        best_model.save(model_dir / f'spatial_{model_type}_fold_{test_fold_idx + 1}_{run_id}.keras')

        fold_results_summary = {
            'fold': test_fold_idx + 1,
            'test_r2': fold_test_r2,
            'test_rmse': fold_test_rmse,
            'test_mae': fold_test_mae,
            'n_train_records': len(X_train),
            'n_val_records': len(X_val),
            'n_test_records': len(X_test),
        }
        
        # 11. Append results directly to the main lists
        fold_results.append(fold_results_summary)
        all_predictions.extend(test_predictions)
        all_actuals.extend(test_labels_actual)
        all_test_r2_scores.append(fold_test_r2)
        all_test_rmse_scores.append(fold_test_rmse)
        all_test_mae_scores.append(fold_test_mae)

    # --- End of Integrated Sequential Cross-Validation Loop ---

    if not fold_results:
        logging.critical("No folds completed successfully.")
        return [], []

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)
    all_residuals = all_actuals - all_predictions

    # Calculate overall statistics
    mean_test_r2 = np.mean(all_test_r2_scores)
    std_test_r2 = np.std(all_test_r2_scores)
    mean_test_rmse = np.mean(all_test_rmse_scores)
    std_test_rmse = np.std(all_test_rmse_scores)
    mean_test_mae = np.mean(all_test_mae_scores)
    std_test_mae = np.std(all_test_mae_scores)

    logging.info(f"\n=== OVERALL CROSS-VALIDATION RESULTS ===")
    logging.info(f"Test R²: {mean_test_r2:.4f} ± {std_test_r2:.4f}")
    logging.info(f"Test RMSE: {mean_test_rmse:.4f} ± {std_test_rmse:.4f}")
    logging.info(f"Test MAE: {mean_test_mae:.4f} ± {std_test_mae:.4f}")

    # --- Plot 1: Publication-quality scatter plot (all folds) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Predictions vs Actuals
    axes[0].scatter(all_actuals, all_predictions, alpha=0.3, s=10, 
                   color='steelblue', edgecolors='none')
    axes[0].set_xlabel('Observed Sap Velocity', fontsize=14)
    axes[0].set_ylabel('Predicted Sap Velocity', fontsize=14)
    axes[0].set_title(f'All Folds Combined: Predictions vs Actuals\n'
                     f'$R^2 = {mean_test_r2:.3f} \\pm {std_test_r2:.3f}$, '
                     f'RMSE = ${mean_test_rmse:.3f} \\pm {std_test_rmse:.3f}$', 
                     fontsize=13, fontweight='bold')

    min_val = min(all_actuals.min(), all_predictions.min())
    max_val = max(all_actuals.max(), all_predictions.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r-', 
                alpha=0.8, linewidth=2, label='1:1 Line')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal', 'box')

    # Right: Residuals
    axes[1].scatter(all_predictions, all_residuals, alpha=0.3, s=10, 
                   color='coral', edgecolors='none')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Sap Velocity', fontsize=14)
    axes[1].set_ylabel('Residuals (Observed - Predicted)', fontsize=14)
    axes[1].set_title(f'All Folds Combined: Residual Plot\n'
                     f'MAE = ${mean_test_mae:.3f} \\pm {std_test_mae:.3f}$', 
                     fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir / f'ALL_FOLDS_scatter_{run_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 2: Metrics by Fold (Bar Charts) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fold_numbers = range(1, len(all_test_r2_scores) + 1)

    # R² by fold
    bars1 = axes[0].bar(fold_numbers, all_test_r2_scores, color='steelblue', edgecolor='black')
    axes[0].axhline(y=mean_test_r2, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_test_r2:.3f}')
    axes[0].fill_between([0.5, len(fold_numbers) + 0.5], 
                        mean_test_r2 - std_test_r2, mean_test_r2 + std_test_r2, 
                        alpha=0.2, color='red')
    axes[0].set_xlabel('Fold', fontsize=12)
    axes[0].set_ylabel('Test R²', fontsize=12)
    axes[0].set_title('R² Score by Fold', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_xticks(fold_numbers)

    # RMSE by fold
    bars2 = axes[1].bar(fold_numbers, all_test_rmse_scores, color='coral', edgecolor='black')
    axes[1].axhline(y=mean_test_rmse, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_test_rmse:.3f}')
    axes[1].fill_between([0.5, len(fold_numbers) + 0.5], 
                        mean_test_rmse - std_test_rmse, mean_test_rmse + std_test_rmse, 
                        alpha=0.2, color='red')
    axes[1].set_xlabel('Fold', fontsize=12)
    axes[1].set_ylabel('Test RMSE', fontsize=12)
    axes[1].set_title('RMSE by Fold', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_xticks(fold_numbers)

    # MAE by fold
    bars3 = axes[2].bar(fold_numbers, all_test_mae_scores, color='mediumseagreen', edgecolor='black')
    axes[2].axhline(y=mean_test_mae, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_test_mae:.3f}')
    axes[2].fill_between([0.5, len(fold_numbers) + 0.5], 
                        mean_test_mae - std_test_mae, mean_test_mae + std_test_mae, 
                        alpha=0.2, color='red')
    axes[2].set_xlabel('Fold', fontsize=12)
    axes[2].set_ylabel('Test MAE', fontsize=12)
    axes[2].set_title('MAE by Fold', fontsize=13, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_xticks(fold_numbers)

    plt.tight_layout()
    plt.savefig(plot_dir / f'ALL_FOLDS_metrics_by_fold_{run_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 3: Distribution + Q-Q (all folds combined) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Distribution comparison
    axes[0].hist(all_actuals, bins=100, alpha=0.6, label='Observed', 
                color='blue', edgecolor='black', linewidth=0.5)
    axes[0].hist(all_predictions, bins=100, alpha=0.6, label='Predicted', 
                color='orange', edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('Sap Velocity', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution: Observed vs Predicted', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Error distribution
    axes[1].hist(all_residuals, bins=100, alpha=0.7, color='coral', edgecolor='black', linewidth=0.5)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].axvline(x=np.mean(all_residuals), color='blue', linestyle='-', linewidth=2, 
                   label=f'Mean: {np.mean(all_residuals):.3f}')
    axes[1].set_xlabel('Residual (Observed - Predicted)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Residual Distribution', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    # Q-Q plot
    from scipy import stats
    stats.probplot(all_residuals, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot of Residuals', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir / f'ALL_FOLDS_distributions_{run_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Save combined predictions CSV ---
    combined_df = pd.DataFrame({
        'Observed': all_actuals,
        'Predicted': all_predictions,
        'Residual': all_residuals,
        'Absolute_Error': np.abs(all_residuals)
    })
    combined_df.to_csv(plot_dir / f'ALL_FOLDS_predictions_{run_id}.csv', index=False)

    # --- Save results summary ---
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(plot_dir / f'cv_results_summary_{run_id}.csv', index=False)

    logging.info(f"\nAll plots saved to: {plot_dir}")
    logging.info("\n" + "="*60)
    logging.info("=== TRAINING FINAL MODEL ON ALL DATA ===")
    logging.info("="*60)
    
    # --- 1. Prepare ALL data (no train/test split) ---
    logging.info("Preparing all data for final model training...")
    
    # Use the same arrays we created earlier: X_all_records, y_all_records
    # But we need to re-split for scaling (fit scaler on a portion, or fit on all)
    
    # Option A: Split into train/val for hyperparameter tuning, then refit on all
    # This is the recommended approach
    
    if IS_STRATIFIED:
        final_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        y_stratify_final = pfts_encoded
    else:
        final_cv = GroupKFold(n_splits=5)
        y_stratify_final = y_all_records
    
    # Get one split for validation during tuning
    train_idx_final, val_idx_final = next(final_cv.split(
        X_all_records, y_stratify_final, groups_all_records
    ))
    
    X_train_final = X_all_records[train_idx_final]
    y_train_final = y_all_records[train_idx_final]
    X_val_final = X_all_records[val_idx_final]
    y_val_final = y_all_records[val_idx_final]
    
    logging.info(f"Final training split: Train={len(X_train_final)}, Val={len(X_val_final)}")
    
    # --- 2. Scale the data ---
    # Separate numerical and categorical features (same logic as in CV loop)
    X_train_final_num = X_train_final[:, :, numerical_indices]
    X_train_final_cat = X_train_final[:, :, categorical_indices]
    X_val_final_num = X_val_final[:, :, numerical_indices]
    X_val_final_cat = X_val_final[:, :, categorical_indices]
    X_all_num = X_all_records[:, :, numerical_indices]
    X_all_cat = X_all_records[:, :, categorical_indices]
    
    # Fit scaler on ALL numerical data (since this is the final model)
    _, _, n_num_features = X_all_num.shape
    X_all_num_reshaped = X_all_num.reshape(-1, n_num_features)
    
    final_feature_scaler = StandardScaler()
    final_feature_scaler.fit(X_all_num_reshaped)  # Fit on ALL data
    
    # Transform all sets
    X_train_final_num_scaled = final_feature_scaler.transform(
        X_train_final_num.reshape(-1, n_num_features)
    ).reshape(X_train_final_num.shape)
    
    X_val_final_num_scaled = final_feature_scaler.transform(
        X_val_final_num.reshape(-1, n_num_features)
    ).reshape(X_val_final_num.shape)
    
    X_all_num_scaled = final_feature_scaler.transform(
        X_all_num_reshaped
    ).reshape(X_all_num.shape)
    
    # Recombine with categorical features
    X_train_final_scaled = np.concatenate([X_train_final_num_scaled, X_train_final_cat], axis=2)
    X_val_final_scaled = np.concatenate([X_val_final_num_scaled, X_val_final_cat], axis=2)
    X_all_scaled = np.concatenate([X_all_num_scaled, X_all_cat], axis=2)
    
    # Scale labels
    final_label_scaler = StandardScaler()
    final_label_scaler.fit(y_all_records.reshape(-1, 1))  # Fit on ALL labels
    
    y_train_final_scaled = final_label_scaler.transform(y_train_final.reshape(-1, 1)).reshape(y_train_final.shape)
    y_val_final_scaled = final_label_scaler.transform(y_val_final.reshape(-1, 1)).reshape(y_val_final.shape)
    y_all_scaled = final_label_scaler.transform(y_all_records.reshape(-1, 1)).reshape(y_all_records.shape)
    
    logging.info("Data scaling complete.")
    
    # --- 3. Hyperparameter tuning on train/val split ---
    logging.info("Starting hyperparameter tuning for final model...")
    
    input_shape = (X_train_final_scaled.shape[1], X_train_final_scaled.shape[2])
    
    final_optimizer = DLOptimizer(
        base_architecture=determine_model_creation_function(model_type),
        task='regression',
        model_type=f'{model_type}_FINAL',
        param_grid=hyperparameters,
        input_shape=input_shape,
        output_shape=LABEL_WIDTH,
        scoring='val_loss',
        random_state=RANDOM_SEED,
        n_splits=5
    )
    
    # Tune hyperparameters using train/val split
    final_optimizer.fit(
        X_train_final_scaled, 
        y_train_final_scaled, 
        is_cv=IS_CV,  # Use same CV setting as main loop
        X_val=X_val_final_scaled, 
        y_val=y_val_final_scaled
    )
    
    best_params = final_optimizer.best_params_
    best_score = final_optimizer.best_score_
    
    logging.info(f"Best hyperparameters found: {best_params}")
    logging.info(f"Best validation score: {best_score:.4f}")
    
    # --- 4. Refit final model on ALL data with best hyperparameters ---
    logging.info("Refitting final model on ALL data with best hyperparameters...")
    
    # Clear keras backend
    keras.backend.clear_session()
    tf.random.set_seed(RANDOM_SEED)
    
    # Create final model with best architecture params
    final_model = determine_model_creation_function(model_type)(
        input_shape=input_shape,
        output_shape=LABEL_WIDTH,
        **best_params.get('architecture', {})
    )
    
    # Compile with best optimizer params
    opt_name = best_params.get('optimizer_name', 'adam')
    opt_params = best_params.get('optimizer', {})
    optimizer_class = getattr(keras.optimizers, opt_name.capitalize() if opt_name == 'adam' else opt_name)
    optimizer = optimizer_class(**opt_params)
    
    final_model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )
    
    # Get training params
    train_params = best_params.get('training', {})
    batch_size = train_params.get('batch_size', BATCH_SIZE)
    epochs = train_params.get('epochs', 100)
    
    # Create callbacks for final training
    final_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='loss',  # Monitor training loss since we're using all data
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            patience=5,
            factor=0.5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            model_dir / f'FINAL_{model_type}_best_{run_id}.keras',
            monitor='loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train on ALL data
    logging.info(f"Training final model: {len(X_all_scaled)} samples, {epochs} max epochs, batch_size={batch_size}")
    
    history = final_model.fit(
        X_all_scaled,
        y_all_scaled,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=final_callbacks,
        verbose=1
    )
    
    final_epochs_trained = len(history.history['loss'])
    final_loss = min(history.history['loss'])
    logging.info(f"Final model trained for {final_epochs_trained} epochs, final loss: {final_loss:.4f}")
    
    # --- 5. Save final model and artifacts ---
    logging.info("Saving final model and artifacts...")
    
    # Save model
    final_model_path = model_dir / f'FINAL_{model_type}_{run_id}.keras'
    final_model.save(final_model_path)
    logging.info(f"Final model saved to: {final_model_path}")
    
    # Save scalers
    final_feature_scaler_path = model_dir / f'FINAL_feature_scaler_{run_id}.pkl'
    final_label_scaler_path = model_dir / f'FINAL_label_scaler_{run_id}.pkl'
    joblib.dump(final_feature_scaler, final_feature_scaler_path)
    joblib.dump(final_label_scaler, final_label_scaler_path)
    logging.info(f"Feature scaler saved to: {final_feature_scaler_path}")
    logging.info(f"Label scaler saved to: {final_label_scaler_path}")
    
    # Save config/metadata
    final_config = {
        'model_type': model_type,
        'run_id': run_id,
        'best_params': {
            'architecture': best_params.get('architecture', {}),
            'optimizer_name': best_params.get('optimizer_name', 'adam'),
            'optimizer': best_params.get('optimizer', {}),
            'training': best_params.get('training', {})
        },
        'tuning_best_score': float(best_score),
        'final_training_loss': float(final_loss),
        'final_epochs_trained': final_epochs_trained,
        'n_total_samples': len(X_all_records),
        'input_shape': list(input_shape),
        'output_shape': LABEL_WIDTH,
        'cv_results': {
            'mean_r2': float(mean_test_r2),
            'std_r2': float(std_test_r2),
            'mean_rmse': float(mean_test_rmse),
            'std_rmse': float(std_test_rmse),
            'mean_mae': float(mean_test_mae),
            'std_mae': float(std_test_mae),
        },
        'feature_columns': feature_columns,
        'numerical_indices': numerical_indices,
        'categorical_indices': categorical_indices,
        'random_seed': RANDOM_SEED,
    }
    
    config_path = model_dir / f'FINAL_config_{run_id}.json'
    with open(config_path, 'w') as f:
        json.dump(final_config, f, indent=2, default=str)
    logging.info(f"Config saved to: {config_path}")
    
    # --- 6. Plot training history ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title(f'Final Model Training Loss\nFinal Loss: {final_loss:.4f}', 
                     fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE curve
    axes[1].plot(history.history['mean_absolute_error'], label='Training MAE', 
                linewidth=2, color='orange')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title(f'Final Model Training MAE\nFinal MAE: {history.history["mean_absolute_error"][-1]:.4f}', 
                     fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / f'FINAL_training_history_{run_id}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- 7. Quick sanity check: predict on all data ---
    logging.info("Running sanity check predictions on all data...")
    
    all_preds_scaled = final_model.predict(X_all_scaled, verbose=0)
    all_preds = final_label_scaler.inverse_transform(all_preds_scaled.reshape(-1, 1)).flatten()
    
    sanity_r2 = r2_score(y_all_records.flatten(), all_preds)
    sanity_rmse = np.sqrt(mean_squared_error(y_all_records.flatten(), all_preds))
    sanity_mae = mean_absolute_error(y_all_records.flatten(), all_preds)
    
    logging.info(f"Sanity check (training data): R²={sanity_r2:.4f}, RMSE={sanity_rmse:.4f}, MAE={sanity_mae:.4f}")
    logging.info("(Note: This is on training data, expect higher performance than CV)")
    
    # Plot sanity check
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_all_records.flatten(), all_preds, alpha=0.3, s=10, color='steelblue')
    ax.set_xlabel('Observed Sap Velocity', fontsize=14)
    ax.set_ylabel('Predicted Sap Velocity', fontsize=14)
    ax.set_title(f'Final Model: Training Data Fit\n'
                f'$R^2 = {sanity_r2:.3f}$, RMSE = ${sanity_rmse:.3f}$\n'
                f'(CV estimate: $R^2 = {mean_test_r2:.3f} \\pm {std_test_r2:.3f}$)', 
                fontsize=12)
    
    min_val = min(y_all_records.min(), all_preds.min())
    max_val = max(y_all_records.max(), all_preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label='1:1 Line')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', 'box')
    
    plt.tight_layout()
    plt.savefig(plot_dir / f'FINAL_sanity_check_{run_id}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("\n" + "="*60)
    logging.info("=== FINAL MODEL TRAINING COMPLETE ===")
    logging.info("="*60)
    logging.info(f"Model: {final_model_path}")
    logging.info(f"Feature Scaler: {final_feature_scaler_path}")
    logging.info(f"Label Scaler: {final_label_scaler_path}")
    logging.info(f"Config: {config_path}")
    logging.info(f"Expected generalization (from CV): R²={mean_test_r2:.4f} ± {std_test_r2:.4f}")

    return all_test_r2_scores, all_test_rmse_scores

if __name__ == "__main__":
    # Run the main function with spatial cross-validation
    main()