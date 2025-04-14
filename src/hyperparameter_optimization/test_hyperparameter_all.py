"""
Unified implementation of multiple time series prediction models with ensemble capabilities.

This script provides a comprehensive framework for:
1. ANN with time window approach
2. Transformer model 
3. CNN-LSTM model
4. LSTM model with bidirectional layers
5. Traditional ML models (XGBoost, SVR, Random Forest)
6. Ensemble model combining all approaches

All models implement deterministic behavior to ensure reproducible results
across multiple runs with the same seed.
"""
import os
import sys
from pathlib import Path
import argparse

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
from src.hyperparameter_optimization.timeseries_processor import TimeSeriesSegmenter, SegmentedWindowGenerator

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
import seaborn as sns
from datetime import datetime

# ML models
from src.hyperparameter_optimization.hyper_tuner import MLOptimizer, DLOptimizer

# Apply additional TensorFlow determinism settings
tf.random.set_seed(42)
np.random.seed(42)

# Force TensorFlow to use deterministic algorithms (TF 2.6+)
try:
    tf.config.experimental.enable_op_determinism()
except:
    # Fallback for older TF versions
    print("Warning: Your TensorFlow version doesn't support enable_op_determinism()")
    # Try alternative approaches
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Limit TensorFlow to use only one thread for CPU operations
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Configure GPU for determinism if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

#=============================================================================
# Part 1: Common Utility Functions
#=============================================================================

@deterministic
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

@deterministic
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

@deterministic
def convert_tf_dataset_to_xy(dataset):
    """
    Convert TensorFlow dataset to X and y numpy arrays with deterministic behavior.
    """
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

def convert_tf_dataset_to_xy_flat(dataset):
    """
    Convert TensorFlow dataset to flattened X and y arrays for traditional ML models.
    """
    X, y = convert_tf_dataset_to_xy(dataset)
    
    # Reshape for traditional ML (flatten window dimension)
    X_reshaped = X.reshape(X.shape[0], -1)
    
    return X_reshaped, y.flatten()

#=============================================================================
# Part 2: Model Implementations
#=============================================================================

# 1. ANN Model
@deterministic
def create_windowed_nn(input_shape, output_shape, n_layers, units, dropout_rate):
    """
    Create a neural network model that takes a time window as input,
    with explicit seeds for all random operations.
    """
    # Force deterministic behavior by setting seed
    seed = get_seed()
    
    # Create incrementing seeds for each layer to ensure independence
    kernel_seeds = [seed + i for i in range(100)]
    seed_idx = 0
    
    # Input layer with deterministic settings
    inputs = keras.layers.Input(shape=input_shape)
    
    # Flatten the time window for the ANN
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

# 2. Transformer Model
@deterministic
def positional_encoding(length, depth):
    """Generate positional encoding for transformer models."""
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]    # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)
    
    angle_rates = 1 / (10000**depths)               # (1, depth)
    angle_rads = positions * angle_rates            # (seq, depth)
    
    # Apply sin to even indices, cos to odd indices
    pos_encoding = np.zeros((length, int(depth) * 2))
    pos_encoding[:, 0::2] = np.sin(angle_rads)
    pos_encoding[:, 1::2] = np.cos(angle_rads)
    
    return tf.cast(pos_encoding, dtype=tf.float32)

class TransformerEncoderLayer(tf.keras.layers.Layer):
    """Custom Transformer encoder layer with MultiHeadAttention and feed-forward network."""
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, seed=None):
        super(TransformerEncoderLayer, self).__init__()
        
        # Use seed for initialization if provided, otherwise get from global seed
        self.seed = seed if seed is not None else get_seed()
        
        # Create incremental seeds for each component
        kernel_seeds = [self.seed + i for i in range(10)]
        
        self.mha = tf.keras.layers.MultiHeadAttention(
            key_dim=d_model // num_heads, 
            num_heads=num_heads, 
            dropout=dropout_rate,
            # Use seed for weight initialization if possible
            **({"seed": kernel_seeds[0]} if tf.__version__ >= "2.5.0" else {})
        )
        
        # Use sequential for FFN with deterministic initialization
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(
                dff, 
                activation='relu',
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seeds[1]),
                bias_initializer=tf.keras.initializers.Zeros()
            ),
            tf.keras.layers.Dense(
                d_model,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seeds[2]),
                bias_initializer=tf.keras.initializers.Zeros()
            )
        ])
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate, seed=kernel_seeds[3])
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate, seed=kernel_seeds[4])
        
    def call(self, inputs, training=False, mask=None):
        # Multi-head attention with residual connection and layer normalization
        attn_output = self.mha(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed forward network with residual connection and layer normalization
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

@deterministic
def create_transformer_model(input_shape, output_shape, d_model=128, num_heads=8, 
                            num_encoder_layers=4, dff=512, dropout_rate=0.1):
    """
    Create a Transformer model for time series prediction with explicit seeds for determinism.
    """
    # Get master seed for this function
    seed = get_seed()
    
    # Create incremental seeds for each layer to ensure independence
    kernel_seeds = [seed + i for i in range(100)]
    seed_idx = 0
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Project inputs to d_model dimensions with deterministic initialization
    x = tf.keras.layers.Dense(
        d_model, 
        activation=None,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seeds[seed_idx]),
        bias_initializer=tf.keras.initializers.Zeros()
    )(inputs)
    seed_idx += 1
    
    # Add positional encoding
    seq_length = input_shape[0]
    pos_encoding = positional_encoding(seq_length, d_model)
    x = x + pos_encoding
    
    # Dropout with explicit seed
    x = tf.keras.layers.Dropout(
        dropout_rate,
        seed=kernel_seeds[seed_idx]
    )(x)
    seed_idx += 1
    
    # Transformer encoder layers with explicit seeds
    for i in range(num_encoder_layers):
        # Create a new seed for this encoder layer
        encoder_seed = kernel_seeds[seed_idx]
        seed_idx += 1
        
        x = TransformerEncoderLayer(
            d_model, 
            num_heads, 
            dff, 
            dropout_rate,
            seed=encoder_seed
        )(x)
    
    # Global average pooling (alternative to using [CLS] token)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Dense layers for prediction with deterministic initialization
    x = tf.keras.layers.Dense(
        d_model // 2, 
        activation='relu', 
        kernel_regularizer=regularizers.l2(0.001),
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seeds[seed_idx]),
        bias_initializer=tf.keras.initializers.Zeros()
    )(x)
    seed_idx += 1
    
    # Dropout with explicit seed
    x = tf.keras.layers.Dropout(
        dropout_rate,
        seed=kernel_seeds[seed_idx]
    )(x)
    seed_idx += 1
    
    # Output layer with deterministic initialization
    outputs = tf.keras.layers.Dense(
        output_shape,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seeds[seed_idx]),
        bias_initializer=tf.keras.initializers.Zeros()
    )(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# 3. CNN-LSTM Model
@deterministic
def create_cnn_lstm_model(input_shape, output_shape, cnn_layers=2, lstm_layers=2, 
                         cnn_filters=64, lstm_units=128, dropout_rate=0.3):
    """
    Create a hybrid CNN-LSTM model with explicit seeds for all random operations.
    """
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
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1))(x)
    
    # Reshape calculations
    new_timesteps = input_shape[0] // (2**(cnn_layers-1)) if cnn_layers > 1 else input_shape[0]
    new_features = cnn_filters * (2**(cnn_layers-1))
    
    # Reshape for LSTM
    x = tf.keras.layers.Reshape((new_timesteps, new_features * input_shape[1]))(x)
    
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

# 4. LSTM Model
@deterministic
def create_lstm_model(input_shape, output_shape, n_layers, units, dropout_rate):
    """
    Create an improved LSTM model with regularization and bidirectional layers.
    """
    # Create a deterministic model with seeds for all random operations
    seed = get_seed()
    
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
                kernel_regularizer=regularizers.l2(0.001),
                recurrent_regularizer=regularizers.l2(0.001),
                bias_regularizer=regularizers.l2(0.001),
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
            kernel_regularizer=regularizers.l2(0.001),
            recurrent_regularizer=regularizers.l2(0.001),
            bias_regularizer=regularizers.l2(0.001),
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

#=============================================================================
# Part 3: Training and Evaluation Functions
#=============================================================================

def load_and_prepare_data(input_width=8, label_width=1, shift=1, batch_size=32, exclude_label=True):
    """
    Load and prepare data for time series prediction.
    
    Returns:
    --------
    dict
        Dictionary containing datasets, scalers, and other info needed for training and evaluation
    """
    # Set seed for reproducibility
    np.random.seed(get_seed())
    
    # Load data
    print("Loading data...")
    data_dir = Path('./outputs/processed_data/merged/site/gap_filled_size1_hourly_after_filter')
    data_list = list(data_dir.glob('*merged.csv'))
    
    if not data_list:
        print(f"No CSV files found in {data_dir}")
        try:
            # Fallback to single file if no directory is found
            data = pd.read_csv('./outputs/processed_data/merged/site/gap_filled_size1_with_era5/test.csv')
            print("Using fallback test.csv file")
        except:
            print("ERROR: Could not find any data files")
            return None
    else:
        print(f"Found {len(data_list)} data files")
    
    # Define the columns we want to use
    used_cols = ['sap_velocity', 'ext_rad', 'ta', 'ws', 'vpd', 'ppfd_in', 'sw_in', 'biome', 'Day sin', 'Week sin', 'Month sin', 'Year sin']
    all_biome_types = set()  # Will collect all unique biome types

    # Sort data files for deterministic processing order
    data_list = sorted(data_list)
    
    # Process each data file
    all_segments = []
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
            
            # Select only the columns we need
            available_cols = [col for col in used_cols_lower if col in df.columns]
            df = df[available_cols]
            
            # Process biome column
            if 'biome' in available_cols:
                # Store biome values to collect all types
                all_biome_types.update(df['biome'].unique())
                
                # Get non-biome columns 
                orig_cols = [col for col in available_cols if col != 'biome']
                
                # Create dummy variables for biome with fixed random state and no dropping of first column
                biome_df = pd.get_dummies(df['biome'], prefix='', prefix_sep='', dtype=float, drop_first=False)
                
                # Join with original data
                df = df[orig_cols].join(biome_df)
            
            # Clean data (only once)
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            
            if len(df) < input_width + shift:
                print(f"Warning: {data_file.name} has too few records after cleaning: {len(df)}")
                continue
            
            # Segment the data
            segments = TimeSeriesSegmenter.segment_time_series(
                df, 
                gap_threshold=2, 
                unit='hours', 
                min_segment_length=input_width + shift
            )
            
            print(f"  Created {len(segments)} segments")
            all_segments.extend(segments)
            
        except Exception as e:
            print(f"Error processing {data_file.name}: {e}")

    if not all_segments:
        print("No valid segments found. Falling back to single file mode.")
        try:
            # Fallback to single file processing
            data = pd.read_csv('./outputs/processed_data/merged/site/gap_filled_size1_with_era5/test.csv')
            data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
            data = data.sort_values('TIMESTAMP').set_index('TIMESTAMP')
            data = add_time_features(data)
            
            # Select only needed columns
            columns = ['sap_velocity', 'sw_in', 'ext_rad', 'ta', 'ws', 'vpd', 'rh', 'ppfd_in']
            # Use available columns
            columns = [col for col in columns if col in data.columns]
            data = data[columns].dropna()
            
            # Create a single segment
            all_segments = [data]
        except:
            print("ERROR: Could not process any data.")
            return None

    print(f"Total segments collected: {len(all_segments)}")
    
    if 'biome' in used_cols:
        print(f"All biome types found: {all_biome_types}")
        # Sort all biome types for consistent column order
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

    # Get feature columns (excluding target)
    if 'biome' in used_cols:
        base_feature_columns = [col for col in all_data.columns if col != 'biome' and col != 'sap_velocity']
        biome_columns = list(all_biome_types)
        feature_columns = base_feature_columns + biome_columns
    else:
        feature_columns = [col for col in all_data.columns if col != 'sap_velocity']
    
    feature_scaler.fit(all_data[feature_columns])
    label_scaler.fit(all_data[['sap_velocity']])
    
    # Apply scaling to each segment
    scaled_segments = []
    for segment in all_segments:
        segment_copy = segment.copy()
        segment_copy[feature_columns] = feature_scaler.transform(segment[feature_columns])
        segment_copy['sap_velocity'] = label_scaler.transform(segment[['sap_velocity']])
        scaled_segments.append(segment_copy)
    
    # Create windowed datasets with deterministic options
    print("Creating windowed datasets...")
    try:
        # First attempt with regular split
        datasets = SegmentedWindowGenerator.create_complete_dataset_from_segments(
            segments=scaled_segments,
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            label_columns=['sap_velocity'],
            train_val_test_split=(0.8, 0.1, 0.1),
            batch_size=batch_size,
            exclude_labels_from_inputs=exclude_label
        )
    except ValueError as e:
        print(f"Error with default split ratio: {e}")
        print("Trying with an alternative split ratio...")
        
        # Second attempt with more balanced split
        datasets = SegmentedWindowGenerator.create_complete_dataset_from_segments(
            segments=scaled_segments,
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            label_columns=['sap_velocity'],
            train_val_test_split=(0.8, 0.1, 0.1),
            batch_size=batch_size,
            exclude_labels_from_inputs=False  # Include labels in inputs as a fallback
        )
    
    train_ds = datasets['train']
    val_ds = datasets['val']
    test_ds = datasets['test']

    # Check dataset properties
    if train_ds is None:
        print("ERROR: Could not create training dataset.")
        return None
    
    # Print dataset shapes for debugging
    print("Dataset shapes:")
    for features_batch, labels_batch in train_ds.take(1):
        print(f"Features batch shape: {features_batch.shape}")
        print(f"Labels batch shape: {labels_batch.shape}")
        input_shape = (features_batch.shape[1], features_batch.shape[2])
        n_features = features_batch.shape[2]
        break
        
    # Prepare traditional ML datasets (flattened)
    X_train_flat, y_train_flat = convert_tf_dataset_to_xy_flat(train_ds)
    X_val_flat, y_val_flat = convert_tf_dataset_to_xy_flat(val_ds)
    X_test_flat, y_test_flat = convert_tf_dataset_to_xy_flat(test_ds)
    
    print(f"Traditional ML shapes: X_train={X_train_flat.shape}, y_train={y_train_flat.shape}")
        
    # Return as a dictionary for easy reference
    return {
        'train_ds': train_ds,
        'val_ds': val_ds,
        'test_ds': test_ds,
        'feature_scaler': feature_scaler,
        'label_scaler': label_scaler,
        'input_shape': input_shape,
        'n_features': n_features,
        'X_train_flat': X_train_flat,
        'y_train_flat': y_train_flat,
        'X_val_flat': X_val_flat, 
        'y_val_flat': y_val_flat,
        'X_test_flat': X_test_flat,
        'y_test_flat': y_test_flat,
        'feature_columns': feature_columns,
        'all_biome_types': all_biome_types
    }

def train_dl_model(model_type, data_dict, output_shape=1, run_id="default", model_dir=None):
    """
    Train a deep learning model with the specified architecture.
    
    Parameters:
    -----------
    model_type : str
        Type of model to train ('ann', 'transformer', 'cnn_lstm', 'lstm')
    data_dict : dict
        Dictionary containing datasets and other info returned by load_and_prepare_data()
    output_shape : int
        Number of output values to predict
    run_id : str
        Identifier for the run
    model_dir : Path, optional
        Directory to save the trained model
        
    Returns:
    --------
    dict
        Dictionary containing the trained model, predictions, and evaluation metrics
    """
    # Set random seed for reproducibility
    set_seed(42)
    RANDOM_SEED = get_seed()
    
    # Extract necessary data from data_dict
    train_ds = data_dict['train_ds']
    val_ds = data_dict['val_ds']
    test_ds = data_dict['test_ds']
    feature_scaler = data_dict['feature_scaler']
    label_scaler = data_dict['label_scaler']
    input_shape = data_dict['input_shape']
    
    # Convert TensorFlow datasets to numpy arrays for optimizer
    X_train, y_train = convert_tf_dataset_to_xy(train_ds)
    X_val, y_val = convert_tf_dataset_to_xy(val_ds)
    
    # Model-specific batch sizes
    if model_type == 'transformer':
        BATCH_SIZE = 64
    else:
        BATCH_SIZE = 32
    
    # Setup model-specific parameter grid and architecture function
    if model_type == 'ann':
        # ANN parameters
        param_grid = {
            'architecture': {
                'n_layers': [2, 3],
                'units': [32, 64],
                'dropout_rate': [0.2] 
            },
            'optimizer': {   
                'name': ['Adam'],
                'learning_rate': [0.001]
            },
            'training': {
                'batch_size': BATCH_SIZE,
                'epochs': 100,
                'patience': 10
            }
        }
        model_architecture = create_windowed_nn
        model_type_name = 'windowed_nn'
        
    elif model_type == 'transformer':
        # Transformer parameters
        param_grid = {
        'architecture': {
            'd_model': [16, 32],
            'num_heads': [2, 4, 8],
            'num_encoder_layers': [4],
            'dff': [32, 64],
            'dropout_rate': [0.3]
        },
        'optimizer': {
            'name': ['Adam'],
        },
        'training': {
            'batch_size': [BATCH_SIZE],
            'epochs': 100, 
            'patience': [20]
        }
        }
        model_architecture = create_transformer_model
        model_type_name = 'Transformer'
        
    elif model_type == 'cnn_lstm':
        # CNN-LSTM parameters
        param_grid = {
            'architecture': {
                'cnn_layers': [1],
                'lstm_layers': [1],
                'cnn_filters': [10],
                'lstm_units': [10],
                'dropout_rate': [0.3],
            },
            'optimizer': {
                'name': ['Adam'],
            },
            'training': {
                'batch_size': [BATCH_SIZE],
                'epochs': 100,
                'patience': [10]
            }
        }
        model_architecture = create_cnn_lstm_model
        model_type_name = 'CNN-LSTM'
        
    elif model_type == 'lstm':
        # LSTM parameters
        param_grid = {
            'architecture': {
                'n_layers': [2],
                'units': [16],
                'dropout_rate': [0.2]
            },
            'optimizer': {
                'name': ['Adam'],
                'learning_rate': [0.001]
            },
            'training': {
                'batch_size': [BATCH_SIZE],
                'epochs': 100,
                'patience': [20]
            }
        }
        model_architecture = create_lstm_model
        model_type_name = 'LSTM'
        
    else:
        print(f"ERROR: Unknown model type '{model_type}'")
        return None
    
    print(f"Training {model_type.upper()} model...")
    
    # Create and fit optimizer with deterministic settings
    optimizer = DLOptimizer(
        base_architecture=model_architecture,
        task='regression',
        model_type=model_type_name,
        param_grid=param_grid,
        input_shape=input_shape,
        output_shape=output_shape,
        scoring='val_loss',
        random_state=RANDOM_SEED,
        use_distribution=False  # Use deterministic behavior
    )
    
    # Fit the optimizer with time series data
    optimizer.fit(
        X_train, 
        y_train,
        is_cv=False,  # We're not using cross-validation
        X_val=X_val,  # Validation data
        y_val=y_val,  # Validation labels
        split_type='temporal'  # We've already created our datasets
    )
    
    # Get the best model
    best_model = optimizer.get_best_model()
    
    # Get predictions using the same function for all models
    print("Generating predictions...")
    test_predictions, test_labels_actual = get_predictions(best_model, test_ds, label_scaler)
    train_predictions, train_labels_actual = get_predictions(best_model, train_ds, label_scaler)
    
    # Calculate metrics
    r2 = r2_score(test_labels_actual, test_predictions)
    r2_train = r2_score(train_labels_actual, train_predictions)
    
    test_mse = mean_squared_error(test_labels_actual, test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(test_labels_actual, test_predictions)
    
    train_mse = mean_squared_error(train_labels_actual, train_predictions)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(train_labels_actual, train_predictions)
    
    # Print metrics
    print("\nModel Evaluation:")
    print("=================")
    print(f"R² Score (test): {r2:.6f}")
    print(f"R² Score (train): {r2_train:.6f}")
    
    print("\nTest Metrics:")
    print(f"MSE: {test_mse:.6f}")
    print(f"RMSE: {test_rmse:.6f}")
    print(f"MAE: {test_mae:.6f}")
    
    # Save the model if directory is provided
    if model_dir is not None:
        model_dir.mkdir(exist_ok=True)
        best_model.save(model_dir / f'{model_type}_model_seed_{RANDOM_SEED}_{run_id}.h5')
    
    # Return results
    return {
        'model': best_model,
        'model_type': model_type,
        'test_predictions': test_predictions,
        'test_labels': test_labels_actual,
        'train_predictions': train_predictions,
        'train_labels': train_labels_actual,
        'metrics': {
            'r2_test': r2,
            'r2_train': r2_train,
            'rmse_test': test_rmse,
            'rmse_train': train_rmse,
            'mae_test': test_mae,
            'mae_train': train_mae
        },
        'best_params': optimizer.best_params_,
        'history': optimizer.history_ if hasattr(optimizer, 'history_') else None
    }

def train_ml_model(model_type, data_dict, run_id="default"):
    """
    Train a traditional machine learning model.
    
    Parameters:
    -----------
    model_type : str
        Type of model to train ('xgboost', 'svr')
    data_dict : dict
        Dictionary containing datasets and other info returned by load_and_prepare_data()
    run_id : str
        Identifier for the run
        
    Returns:
    --------
    dict
        Dictionary containing the trained model, predictions, and evaluation metrics
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Extract necessary data from data_dict
    X_train = data_dict['X_train_flat']
    y_train = data_dict['y_train_flat']
    X_val = data_dict['X_val_flat']
    y_val = data_dict['y_val_flat']
    X_test = data_dict['X_test_flat']
    y_test = data_dict['y_test_flat']
    label_scaler = data_dict['label_scaler']
    
    # Define parameter grid based on model type
    if model_type == 'xgb':
        param_grid = {
            'reg_alpha': [1.0],
            'reg_lambda': [1.0],
            'max_depth': [7],
            'min_child_weight': [1],
            'gamma': [0.1],
            'subsample': [0.5],
            'colsample_bytree': [0.6],
            'colsample_bylevel': [0.9],
            'learning_rate': [0.01],
            'n_estimators': [800],
        }
        score_metric = 'neg_mean_squared_error'
        
    elif model_type == 'svr':
        param_grid = {
            'kernel': ['poly'],
            'C': [10],
            'gamma': ['scale'],
            'degree': [4],
            'coef0': [1],
            'epsilon': [0.5]
        }
        score_metric = 'r2'
        
    elif model_type == 'rf':
        param_grid = {
        'n_estimators': [500],  # Fewer trees can sometimes help reduce overfitting
        'max_depth': [7],  # Reduce max_depth to prevent trees from becoming too specific
        'min_samples_split': [5],  # Higher values prevent creating too many small splits
        'min_samples_leaf': [2,],  # Higher values ensure terminal nodes aren't too specific
        'max_features': ['sqrt',],  # Controls feature randomness in tree building
        'bootstrap': [True],
        'random_state': [42],
        'oob_score': [True]  # Enable out-of-bag evaluation for better generalization assessment
    }
        score_metric = 'neg_mean_squared_error'
        
    else:
        print(f"ERROR: Unknown model type '{model_type}'")
        return None
    
    print(f"Training {model_type.upper()} model...")
    
    # Create and fit optimizer
    optimizer = MLOptimizer(
        param_grid=param_grid, 
        scoring=score_metric, 
        model_type=model_type, 
        task='regression',
        random_state=42
    )
    
    # Fit with validation data
    optimizer.fit(
        X_train, 
        y_train,
        X_val=X_val,
        y_val=y_val,
        split_type='predefined',
        is_shuffle=False  # Don't shuffle time series data
    )
    
    # Get best model
    best_model = optimizer.best_estimator_
    print(f"Best parameters: {optimizer.best_params_}")
    
    # Predict on test and train data
    y_pred_scaled = best_model.predict(X_test)
    y_train_pred_scaled = best_model.predict(X_train)
    
    # Inverse transform predictions to original scale
    y_pred = label_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_train_pred = label_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
    
    # Inverse transform labels to original scale for evaluation
    y_test_actual = label_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_train_actual = label_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    r2 = r2_score(y_test_actual, y_pred)
    r2_train = r2_score(y_train_actual, y_train_pred)
    
    test_mse = mean_squared_error(y_test_actual, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test_actual, y_pred)
    
    train_mse = mean_squared_error(y_train_actual, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train_actual, y_train_pred)
    
    # Print metrics
    print("\nModel Evaluation:")
    print("=================")
    print(f"R² Score (test): {r2:.6f}")
    print(f"R² Score (train): {r2_train:.6f}")
    
    print("\nTest Metrics:")
    print(f"MSE: {test_mse:.6f}")
    print(f"RMSE: {test_rmse:.6f}")
    print(f"MAE: {test_mae:.6f}")
    
    # Return results
    return {
        'model': best_model,
        'model_type': model_type,
        'test_predictions': y_pred,
        'test_labels': y_test_actual,
        'train_predictions': y_train_pred,
        'train_labels': y_train_actual,
        'metrics': {
            'r2_test': r2,
            'r2_train': r2_train,
            'rmse_test': test_rmse,
            'rmse_train': train_rmse,
            'mae_test': test_mae,
            'mae_train': train_mae
        },
        'best_params': optimizer.best_params_
    }

def create_ensemble_predictions(model_results_list):
    """
    Create ensemble predictions by taking the median of individual model predictions.
    
    Parameters:
    -----------
    model_results_list : list
        List of dictionaries containing model results
        
    Returns:
    --------
    dict
        Dictionary containing ensemble predictions and evaluation metrics
    """
    # Check if we have any models
    if not model_results_list:
        print("No models to ensemble.")
        return None
    
    # Extract predictions
    test_predictions = []
    train_predictions = []
    
    # Reference labels (should be the same for all models)
    test_labels = model_results_list[0]['test_labels']
    train_labels = model_results_list[0]['train_labels']
    
    # Collect predictions from all models
    for result in model_results_list:
        test_predictions.append(result['test_predictions'])
        train_predictions.append(result['train_predictions'])
    
    # Convert to numpy arrays
    test_predictions = np.array(test_predictions)
    train_predictions = np.array(train_predictions)
    
    # Create ensemble predictions (median)
    ensemble_test_pred = np.median(test_predictions, axis=0)
    ensemble_train_pred = np.median(train_predictions, axis=0)
    
    # Calculate metrics
    r2 = r2_score(test_labels, ensemble_test_pred)
    r2_train = r2_score(train_labels, ensemble_train_pred)
    
    test_mse = mean_squared_error(test_labels, ensemble_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(test_labels, ensemble_test_pred)
    
    train_mse = mean_squared_error(train_labels, ensemble_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(train_labels, ensemble_train_pred)
    
    # Print metrics
    print("\nEnsemble Model Evaluation:")
    print("=========================")
    print(f"R² Score (test): {r2:.6f}")
    print(f"R² Score (train): {r2_train:.6f}")
    
    print("\nTest Metrics:")
    print(f"MSE: {test_mse:.6f}")
    print(f"RMSE: {test_rmse:.6f}")
    print(f"MAE: {test_mae:.6f}")
    
    # Return results
    return {
        'model_type': 'ensemble',
        'test_predictions': ensemble_test_pred,
        'test_labels': test_labels,
        'train_predictions': ensemble_train_pred,
        'train_labels': train_labels,
        'metrics': {
            'r2_test': r2,
            'r2_train': r2_train,
            'rmse_test': test_rmse,
            'rmse_train': train_rmse,
            'mae_test': test_mae,
            'mae_train': train_mae
        },
        'component_models': [result['model_type'] for result in model_results_list]
    }

#=============================================================================
# Part 4: Visualization Functions
#=============================================================================

def plot_predictions(result, plot_dir=None, run_id="default"):
    """
    Plot predictions vs actual values for a model.
    
    Parameters:
    -----------
    result : dict
        Dictionary containing model results
    plot_dir : Path, optional
        Directory to save the plots
    run_id : str
        Identifier for the run
    """
    model_type = result['model_type']
    test_predictions = result['test_predictions']
    test_labels = result['test_labels']
    train_predictions = result['train_predictions']
    train_labels = result['train_labels']
    
    # Set deterministic plotting
    plt.rcParams['agg.path.chunksize'] = 10000
    np.random.seed(42)
    
    # Create plot directory if it doesn't exist
    if plot_dir is not None:
        plot_dir.mkdir(exist_ok=True)
    
    # Plot predictions vs actual for test data
    plt.figure(figsize=(10, 10))
    plt.scatter(test_labels, test_predictions, alpha=0.5)
    plt.xlabel('True Values [Sap Velocity]')
    plt.ylabel('Predictions [Sap Velocity]')
    plt.title(f'{model_type.upper()} Model: Test Predictions vs Actual')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    plt.plot([-100, 100], [-100, 100])
    if plot_dir is not None:
        plt.savefig(plot_dir / f'{model_type}_test_predictions_{run_id}.png')
    plt.close()
    
    # Plot predictions vs actual for training data
    plt.figure(figsize=(10, 10))
    plt.scatter(train_labels, train_predictions, alpha=0.5)
    plt.xlabel('True Values [Sap Velocity]')
    plt.ylabel('Predictions [Sap Velocity]')
    plt.title(f'{model_type.upper()} Model: Training Predictions vs Actual')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    plt.plot([-100, 100], [-100, 100])
    if plot_dir is not None:
        plt.savefig(plot_dir / f'{model_type}_train_predictions_{run_id}.png')
    plt.close()
    
    # Plot error distribution for test data
    error = test_predictions - test_labels
    plt.figure(figsize=(10, 7))
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [Sap Velocity]')
    plt.ylabel('Count')
    plt.title(f'{model_type.upper()} Model: Test Error Distribution')
    if plot_dir is not None:
        plt.savefig(plot_dir / f'{model_type}_test_error_distribution_{run_id}.png')
    plt.close()
    
    # Plot error distribution for training data
    error_train = train_predictions - train_labels
    plt.figure(figsize=(10, 7))
    plt.hist(error_train, bins=25)
    plt.xlabel('Prediction Error [Sap Velocity]')
    plt.ylabel('Count')
    plt.title(f'{model_type.upper()} Model: Training Error Distribution')
    if plot_dir is not None:
        plt.savefig(plot_dir / f'{model_type}_train_error_distribution_{run_id}.png')
    plt.close()
    
def plot_model_comparison(results_list, plot_dir=None, run_id="default"):
    """
    Create comparative plots of model performance.
    
    Parameters:
    -----------
    results_list : list
        List of dictionaries containing model results
    plot_dir : Path, optional
        Directory to save the plots
    run_id : str
        Identifier for the run
    """
    # Create plot directory if it doesn't exist
    if plot_dir is not None:
        plot_dir.mkdir(exist_ok=True)
    
    # Extract model names and metrics
    model_names = []
    r2_test_values = []
    r2_train_values = []
    rmse_test_values = []
    rmse_train_values = []
    mae_test_values = []
    mae_train_values = []
    
    for result in results_list:
        model_names.append(result['model_type'].upper())
        r2_test_values.append(result['metrics']['r2_test'])
        r2_train_values.append(result['metrics']['r2_train'])
        rmse_test_values.append(result['metrics']['rmse_test'])
        rmse_train_values.append(result['metrics']['rmse_train'])
        mae_test_values.append(result['metrics']['mae_test'])
        mae_train_values.append(result['metrics']['mae_train'])
    
    # Set a consistent style
    sns.set(style="whitegrid")
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'Model': model_names,
        'R² (Test)': r2_test_values,
        'R² (Train)': r2_train_values,
        'RMSE (Test)': rmse_test_values,
        'RMSE (Train)': rmse_train_values,
        'MAE (Test)': mae_test_values,
        'MAE (Train)': mae_train_values
    })
    
    # Create a timestamp for the plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Plot R² values
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Model', y='value', hue='variable', 
                    data=pd.melt(df, id_vars=['Model'], value_vars=['R² (Test)', 'R² (Train)']))
    plt.title('Model Comparison: R² Scores', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('R² Score', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Metric')
    plt.tight_layout()
    if plot_dir is not None:
        plt.savefig(plot_dir / f'model_comparison_r2_{run_id}_{timestamp}.png')
    plt.close()
    
    # 2. Plot RMSE values
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Model', y='value', hue='variable', 
                    data=pd.melt(df, id_vars=['Model'], value_vars=['RMSE (Test)', 'RMSE (Train)']))
    plt.title('Model Comparison: RMSE', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('RMSE', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Metric')
    plt.tight_layout()
    if plot_dir is not None:
        plt.savefig(plot_dir / f'model_comparison_rmse_{run_id}_{timestamp}.png')
    plt.close()
    
    # 3. Plot MAE values
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Model', y='value', hue='variable', 
                    data=pd.melt(df, id_vars=['Model'], value_vars=['MAE (Test)', 'MAE (Train)']))
    plt.title('Model Comparison: MAE', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('MAE', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Metric')
    plt.tight_layout()
    if plot_dir is not None:
        plt.savefig(plot_dir / f'model_comparison_mae_{run_id}_{timestamp}.png')
    plt.close()
    
    # 4. Comprehensive comparison chart with all test metrics
    plt.figure(figsize=(16, 10))
    
    # Normalize metrics to make them comparable on the same scale
    max_rmse = max(df['RMSE (Test)'])
    max_mae = max(df['MAE (Test)'])
    
    df['RMSE (Test) Normalized'] = df['RMSE (Test)'] / max_rmse
    df['MAE (Test) Normalized'] = df['MAE (Test)'] / max_mae
    
    # For R², higher is better, so we invert it for visualization (1 - R²)
    # This way, lower values are better for all metrics
    df['R² (Test) Inverted'] = 1 - df['R² (Test)']
    
    # Plot comprehensive comparison
    ax = sns.barplot(x='Model', y='value', hue='variable', 
                    data=pd.melt(df, id_vars=['Model'], 
                                value_vars=['R² (Test) Inverted', 'RMSE (Test) Normalized', 'MAE (Test) Normalized']))
    
    plt.title('Comprehensive Model Comparison (Test Set Metrics)', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Normalized Metric (Lower is Better)', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Metric', labels=['1 - R²', 'RMSE (Normalized)', 'MAE (Normalized)'])
    plt.tight_layout()
    if plot_dir is not None:
        plt.savefig(plot_dir / f'model_comparison_comprehensive_{run_id}_{timestamp}.png')
    plt.close()
    
    # Create and save a summary table
    summary_df = df[['Model', 'R² (Test)', 'RMSE (Test)', 'MAE (Test)']]
    summary_df = summary_df.sort_values('R² (Test)', ascending=False)
    
    if plot_dir is not None:
        summary_df.to_csv(plot_dir / f'model_comparison_summary_{run_id}_{timestamp}.csv', index=False)
    
    print("\nModel Comparison Summary:")
    print("========================")
    print(summary_df.to_string(index=False))
    
    return summary_df

#=============================================================================
# Part 5: Main Execution Functions
#=============================================================================

def train_all_models(run_id="default", output_dir=None):
    """
    Train and evaluate all models, including ensemble.
    
    Parameters:
    -----------
    run_id : str
        Identifier for the run
    output_dir : Path, optional
        Directory to save models and plots
        
    Returns:
    --------
    list
        List of dictionaries containing results for all models
    """
    # Set up directories
    if output_dir is not None:
        output_dir = Path(output_dir)
        model_dir = output_dir / "models"
        plot_dir = output_dir / "plots"
        model_dir.mkdir(exist_ok=True, parents=True)
        plot_dir.mkdir(exist_ok=True, parents=True)
    else:
        model_dir = None
        plot_dir = None
    
    # Load and prepare data
    data_dict = load_and_prepare_data()
    if data_dict is None:
        print("Error loading data.")
        return None
    
    # Train all models
    model_results = []
    
    # Deep learning models
    dl_models = ['ann', 'transformer', 'cnn_lstm', 'lstm']
    for model_type in dl_models:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*50}")
        
        set_seed(42)  # Reset seed for each model
        result = train_dl_model(model_type, data_dict, run_id=run_id, model_dir=model_dir)
        
        if result is not None:
            model_results.append(result)
            # Plot individual model results
            plot_predictions(result, plot_dir=plot_dir, run_id=run_id)
    
    # ML models
    ml_models = ['xgboost', 'svr', 'random_forest']
    for model_type in ml_models:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*50}")
        
        np.random.seed(42)  # Reset seed for each model
        result = train_ml_model(model_type, data_dict, run_id=run_id)
        
        if result is not None:
            model_results.append(result)
            # Plot individual model results
            plot_predictions(result, plot_dir=plot_dir, run_id=run_id)
    
    # Create ensemble model
    print(f"\n{'='*50}")
    print(f"Creating ensemble model")
    print(f"{'='*50}")
    
    ensemble_result = create_ensemble_predictions(model_results)
    
    if ensemble_result is not None:
        model_results.append(ensemble_result)
        # Plot ensemble model results
        plot_predictions(ensemble_result, plot_dir=plot_dir, run_id=run_id)
    
    # Plot model comparison
    summary_df = plot_model_comparison(model_results, plot_dir=plot_dir, run_id=run_id)
    
    return model_results, summary_df

def main():
    """Main function to run the time series prediction framework."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate time series prediction models.')
    parser.add_argument('--model', type=str, default='ensemble', 
                        choices=['ann', 'transformer', 'cnn_lstm', 'lstm', 'xgboost', 'svr', 'random_forest', 'ensemble', 'all'],
                        help='Model type to train or "all" for all models')
    parser.add_argument('--run-id', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"),
                        help='Identifier for the run')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Directory to save models and plots')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Train models based on argument
    if args.model == 'all' or args.model == 'ensemble':
        # Train all models and create ensemble
        results, summary = train_all_models(run_id=args.run_id, output_dir=output_dir)
    else:
        # Train a single model
        data_dict = load_and_prepare_data()
        if data_dict is None:
            print("Error loading data.")
            return
        
        # Set up directories
        model_dir = output_dir / "models"
        plot_dir = output_dir / "plots"
        model_dir.mkdir(exist_ok=True, parents=True)
        plot_dir.mkdir(exist_ok=True, parents=True)
        
        if args.model in ['ann', 'transformer', 'cnn_lstm', 'lstm']:
            # Train a deep learning model
            set_seed(42)
            result = train_dl_model(args.model, data_dict, run_id=args.run_id, model_dir=model_dir)
        else:
            # Train a traditional ML model
            np.random.seed(42)
            result = train_ml_model(args.model, data_dict, run_id=args.run_id)
        
        if result is not None:
            # Plot individual model results
            plot_predictions(result, plot_dir=plot_dir, run_id=args.run_id)
            
            # Create a list with a single result for summary
            results = [result]
            summary = plot_model_comparison(results, plot_dir=plot_dir, run_id=args.run_id)
    
    print("\nTraining and evaluation complete.")

if __name__ == "__main__":
    main()