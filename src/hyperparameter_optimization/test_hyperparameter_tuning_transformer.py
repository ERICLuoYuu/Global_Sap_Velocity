"""
Modified main script with enhanced deterministic behavior for Transformer models.
This script implements comprehensive controls to ensure reproducible results
across multiple runs with the same seed.
"""
from pathlib import Path
import sys
import os
import joblib

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

# Apply additional TensorFlow determinism settings
tf.random.set_seed(42)
np.random.seed(42)

# Force TensorFlow to use deterministic algorithms (TF 2.6+)
tf.config.experimental.enable_op_determinism()

# Limit TensorFlow to use only one thread for CPU operations
# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)

# Configure GPU for determinism if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# Import the hyperparameter optimizer
from src.hyperparameter_optimization.hyper_tuner import DLOptimizer



# Transformer positional encoding with deterministic behavior
@deterministic
def positional_encoding(length, depth):
    """
    Generate positional encoding for transformer models.
    
    Parameters:
    -----------
    length: int, sequence length
    depth: int, dimension of the model
    
    Returns:
    --------
    pos_encoding: tf.Tensor, positional encoding matrix of shape (1, length, depth)
    """
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


# Custom Transformer Encoder Layer with deterministic behavior
class TransformerEncoderLayer(tf.keras.layers.Layer):
    """
    Custom Transformer encoder layer with MultiHeadAttention and feed-forward network.
    """
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
            # Not all TF versions support this, so using a try/except
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
    
    Parameters:
    -----------
    input_shape: tuple, shape of input data (timesteps, features)
    output_shape: int, number of output values
    d_model: int, dimension of the model
    num_heads: int, number of attention heads
    num_encoder_layers: int, number of transformer encoder layers
    dff: int, dimension of feed-forward network
    dropout_rate: float, dropout rate
    
    Returns:
    --------
    model: keras.Model, Transformer model for time series prediction
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
def add_time_features(df, datetime_column=None):
    """
    Create cyclical time features from a datetime column or index.
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


# Visualization function for Transformer attention patterns with deterministic rendering
@deterministic
def visualize_attention_weights(model, dataset, layer_idx=0, head_idx=0):
    """
    Visualize attention weights from a Transformer model with deterministic behavior.
    
    Parameters:
    -----------
    model: keras.Model, trained Transformer model
    dataset: tf.data.Dataset, dataset to extract samples from
    layer_idx: int, index of the transformer layer to visualize
    head_idx: int, index of the attention head to visualize
    """
    # Set deterministic rendering for matplotlib
    import matplotlib as mpl
    mpl.rcParams['agg.path.chunksize'] = 10000
    np.random.seed(get_seed())  # Ensure matplotlib's random operations use our seed
    
    # Get a batch of input data
    for features, _ in dataset.take(1):
        # Create a model that outputs attention weights
        inputs = tf.keras.layers.Input(shape=features.shape[1:])
        
        # Get the transformer layer
        transformer_layer = model.layers[layer_idx+2]  # Adjust index as needed
        
        # Call the layer and extract attention weights
        _, attention_weights = transformer_layer(inputs, return_attention_weights=True)
        
        # Create a model to extract attention weights
        attention_model = tf.keras.Model(inputs=inputs, outputs=attention_weights)
        
        # Get attention weights for the batch
        batch_attention = attention_model.predict(features)
        
        # Extract weights for the specified head
        head_attention = batch_attention[:, head_idx, :, :]
        
        # Plot attention heatmap for the first sample
        plt.figure(figsize=(10, 8))
        plt.imshow(head_attention[0], cmap='viridis')
        plt.title(f'Transformer Attention Weights (Layer {layer_idx}, Head {head_idx})')
        plt.xlabel('Token Position (Output)')
        plt.ylabel('Token Position (Input)')
        plt.colorbar()
        plt.tight_layout()
        
        return


@deterministic
def main():
    """
    Main function to train and evaluate the ANN model with deterministic behavior.
    Modified to use time windows for prediction similar to CNN-LSTM.
    
    Parameters:
    -----------
    run_id : str
        Identifier for the run, used for saving models and plots
        
    Returns:
    --------
    tuple
        R² scores (test, train) and RMSE scores (test, train)
    """
    # Set parameters
    RANDOM_SEED = get_seed()
    BATCH_SIZE = 32
    
    # Define window parameters (matching CNN-LSTM implementation)
    INPUT_WIDTH = 8   # Use 8 time steps as input
    LABEL_WIDTH = 1   # Predict 1 time step ahead
    SHIFT = 1         # Predict 1 step ahead
    EXCLUDE_LABEL = True  # Exclude labels in input features
    
    # Set deterministic rendering for matplotlib
    import matplotlib as mpl
    mpl.rcParams['agg.path.chunksize'] = 10000
    np.random.seed(RANDOM_SEED)  # Ensure matplotlib's random operations use our seed
    
    print(f"Starting windowed ANN model training with seed {RANDOM_SEED} (run_id: {run_id})")
    
    # Load and preprocess data in a deterministic manner
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
            return (0, 0), (0, 0)  # Return dummy values
    else:
        print(f"Found {len(data_list)} data files")
    
    # Process data files (similar to CNN-LSTM implementation)
    all_segments = []
    # Define the columns we want to use
    used_cols = ['sap_velocity', 'ext_rad', 'sw_in', 'ta', 'ws', 'vpd','ppfd_in', 'biome', 'Day sin', 'Week sin', 'Month sin', 'Year sin']
    all_biome_types = set()  # Will collect all unique biome types
    all_possible_biome_types = ['Boreal forest', 'Subtropical desert', 'Temperate forest', 'Temperate grassland desert', 'Temperate rain forest', 'Tropical forest savanna', 'Tropical rain forest', 'Tundra', 'Woodland/Shrubland']
    # Sort data files for deterministic processing order
    data_list = sorted(data_list)

    # Process each data file
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
            
            if 'biome' in used_cols_lower:
                if 'biome' in df.columns:
                    # Get non-biome columns 
                    orig_cols = [col for col in used_cols_lower if col != 'biome']
                    
                    # Track which biome types are found in the data
                    all_biome_types.update(df['biome'].unique())
                    
                    # Create a temporary biome column Series
                    biome_series = df['biome']
                    
                    # Initialize a DataFrame with zeros for all possible biomes
                    biome_df = pd.DataFrame(0.0, index=df.index, columns=all_possible_biome_types)
                    
                    # For each row, set the corresponding biome column to 1.0
                    for idx, biome in biome_series.items():
                        if biome in all_possible_biome_types:
                            biome_df.loc[idx, biome] = 1.0
                    
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
            
            # Segment the data (just like in CNN-LSTM)
            segments = TimeSeriesSegmenter.segment_time_series(
                df, 
                gap_threshold=2, 
                unit='hours', 
                min_segment_length=INPUT_WIDTH + SHIFT
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
            return (0, 0), (0, 0)  # Return dummy values

    print(f"Total segments collected: {len(all_segments)}")
    
    if 'biome' in used_cols:
        # Sort all biome types for consistent column order
        all_biome_types = sorted(all_biome_types)
        
        # Log what biome types were actually found in the data
        print(f"Biome types found in data: {all_biome_types}")
        print(f"Missing biome types: {set(all_possible_biome_types) - all_biome_types}")

    # Now standardize the data
    all_data = pd.concat(all_segments)

    # Create scalers
    feature_scaler = StandardScaler()
    label_scaler = StandardScaler()

     # Get numerical and categorical feature columns
    if 'biome' in used_cols:
        numerical_features = [col for col in all_data.columns 
                            if col != 'sap_velocity' and col not in all_possible_biome_types]
        categorical_features = [col for col in all_possible_biome_types if col in all_data.columns]
    else:
        numerical_features = [col for col in all_data.columns if col != 'sap_velocity']
        categorical_features = []

    # Only fit scaler on numerical features
    feature_scaler.fit(all_data[numerical_features])
    label_scaler.fit(all_data[['sap_velocity']])
    # save the scalers for later use
    feature_scaler_path = Path('./outputs/scalers/feature_scaler.pkl')
    label_scaler_path = Path('./outputs/scalers/label_scaler.pkl')
    # Create directory if it doesn't exist
    Path('./outputs/scalers').mkdir(parents=True, exist_ok=True)


    joblib.dump(feature_scaler, feature_scaler_path)
    joblib.dump(label_scaler, label_scaler_path)
    # Apply scaling to each segment
    scaled_segments = []
    for segment in all_segments:
        segment_copy = segment.copy()
        # Only transform numerical features
        segment_copy[numerical_features] = feature_scaler.transform(segment[numerical_features])
        # Leave categorical features as is
        segment_copy['sap_velocity'] = label_scaler.transform(segment[['sap_velocity']])
        print(segment_copy.columns)
        scaled_segments.append(segment_copy)
    
    # Create windowed datasets
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
            train_val_test_split=(0.8, 0.1, 0.1),
            batch_size=BATCH_SIZE,
            exclude_labels_from_inputs=False  # Include labels in inputs as a fallback
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

    # Define parameter grid for Transformer with deterministic settings
    param_grid = {
        'architecture': {
            'd_model': [32, 64],
            'num_heads': [4, 8],
            'num_encoder_layers': [2, 4],
            'dff': [64],
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
    
    # Create and fit optimizer with deterministic behavior
    print("Starting hyperparameter optimization...")
    optimizer = DLOptimizer(
        base_architecture=create_transformer_model,
        task='regression',
        model_type='transformer',
        param_grid=param_grid,
        input_shape=(INPUT_WIDTH, n_features),
        output_shape=LABEL_WIDTH,
        scoring='val_loss',
        # Pass random_state for determinism
        random_state=get_seed(),
    )
    
    # Convert TensorFlow dataset to X and y numpy arrays with deterministic behavior
    @deterministic
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
        tf.keras.utils.set_random_seed(get_seed())
    except:
        # For older TensorFlow versions
        tf.random.set_seed(get_seed())
        np.random.seed(get_seed())
    
    # Custom fitting for time series data
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
    plt.title('Transformer Model: Test Predictions vs Actual')
    plt.savefig(plot_dir / 'transformer_test_predictions.png')
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
    plt.title('Transformer Model: Training Predictions vs Actual')
    plt.savefig(plot_dir / 'transformer_train_predictions.png')
    plt.close()
    
    # Plot error distribution for test data
    error = test_predictions - test_labels_actual
    plt.figure(figsize=(10, 7))
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [Sap Velocity]')
    plt.ylabel('Count')
    plt.title('Transformer Model: Test Error Distribution')
    plt.savefig(plot_dir / 'transformer_test_error_distribution.png')
    plt.close()
    
    # Plot error distribution for training data
    error_train = train_predictions - train_labels_actual
    plt.figure(figsize=(10, 7))
    plt.hist(error_train, bins=25)
    plt.xlabel('Prediction Error [Sap Velocity]')
    plt.ylabel('Count')
    plt.title('Transformer Model: Training Error Distribution')
    plt.savefig(plot_dir / 'transformer_train_error_distribution.png')
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
        plt.savefig(plot_dir / 'transformer_training_history.png')
        plt.close()
    
    return test_rmse


def verify_determinism():
    """Test if the pipeline produces deterministic results."""
    # Save current random state
    current_seed = get_seed()
    
    # First run with fixed seed
    set_seed(42)
    print("\nRunning first determinism test...")
    result1 = main()
    
    # Second run with same seed
    set_seed(42)
    print("\nRunning second determinism test...")
    result2 = main()
    
    # Check if results match
    print("\nDeterminism Test Results:")
    print(f"First run RMSE: {result1:.6f}")
    print(f"Second run RMSE: {result2:.6f}")
    print(f"Difference: {abs(result1 - result2):.10f}")
    print(f"Results identical: {np.isclose(result1, result2, rtol=1e-10, atol=1e-10)}")
    
    # Restore original seed
    set_seed(current_seed)
    
    return result1, result2


if __name__ == "__main__":
    # Run the main function
    main()
    
