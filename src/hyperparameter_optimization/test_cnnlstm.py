
"""
Modified main script with enhanced deterministic behavior for CNN-LSTM models.
This script implements comprehensive controls to ensure reproducible results
across multiple runs with the same seed.
"""
from pathlib import Path
import sys
import os

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
def create_sliding_windows(segments, input_width, label_width, shift, 
                          feature_columns, label_columns):
    """
    Create sliding windows from segments using direct approach from second script.
    
    Args:
        segments: List of DataFrame segments
        input_width: Size of the input window
        label_width: Size of the prediction window
        shift: Offset for the prediction window
        feature_columns: Columns to use as features
        label_columns: Columns to predict
        
    Returns:
        Tuple of X and y numpy arrays
    """
    all_X, all_y = [], []
    
    # Track statistics for debugging
    total_windows = 0
    skipped_segments = 0
    used_segments = 0
    
    for segment in segments:
        # Skip segments that are too short
        min_length = input_width + shift + label_width - 1
        if len(segment) < min_length:
            skipped_segments += 1
            continue
        
        used_segments += 1
        segment_windows = 0
        
        # Get feature and label indices for faster array access
        feature_indices = [list(segment.columns).index(col) for col in feature_columns]
        label_indices = [list(segment.columns).index(col) for col in label_columns]
        
        # Convert segment to numpy array for faster indexing
        segment_array = segment.values
        
        # Create windows for this segment
        for i in range(len(segment_array) - (input_width + shift + label_width - 1)):
            # Input window
            input_slice = segment_array[i:i+input_width, :][:, feature_indices]
            
            # Label window (shifted)
            label_start = i + input_width + shift - 1
            label_slice = segment_array[label_start:label_start+label_width, :][:, label_indices]
            
            # Flatten label if it's a single value
            if label_width == 1 and len(label_columns) == 1:
                label_slice = label_slice.flatten()[0]
            
            all_X.append(input_slice)
            all_y.append(label_slice)
            segment_windows += 1
        
        total_windows += segment_windows
    
    # Print statistics
    print(f"Created {total_windows} sliding windows from {used_segments} segments")
    print(f"Skipped {skipped_segments} segments (too short)")
    
    if not all_X:
        raise ValueError("No windows could be created. Check your data and window parameters.")
    
    # Convert to numpy arrays
    X = np.array(all_X)
    y = np.array(all_y)
    
    return X, y

@deterministic
def create_datasets_with_sliding_windows(segments, input_width, label_width, shift,
                                       feature_columns, label_columns, 
                                       train_val_test_split, batch_size):
    """
    Create train, validation, and test datasets using sliding windows.
    
    Args:
        segments: List of DataFrame segments
        input_width: Size of the input window
        label_width: Size of the prediction window
        shift: Offset for the prediction window
        feature_columns: Columns to use as features
        label_columns: Columns to predict
        train_val_test_split: Tuple of (train_fraction, val_fraction, test_fraction)
        batch_size: Batch size for the datasets
        
    Returns:
        Dictionary with 'train', 'val', and 'test' TensorFlow datasets
    """
    # Ensure split fractions sum to 1
    if sum(train_val_test_split) != 1.0:
        raise ValueError("Split fractions must sum to 1.0")
    
    # Shuffle segments deterministically
    np.random.seed(get_seed())
    shuffled_segments = segments.copy()
    np.random.shuffle(shuffled_segments)
    
    # Calculate split indices
    n_segments = len(shuffled_segments)
    train_end = int(n_segments * train_val_test_split[0])
    val_end = train_end + int(n_segments * train_val_test_split[1])
    
    # Split segments
    train_segments = shuffled_segments[:train_end]
    val_segments = shuffled_segments[train_end:val_end]
    test_segments = shuffled_segments[val_end:]
    
    print(f"Split into {len(train_segments)} train, {len(val_segments)} validation, and {len(test_segments)} test segments")
    
    # Create windows for each split
    print("Creating training windows...")
    X_train, y_train = create_sliding_windows(train_segments, input_width, label_width, shift,
                                            feature_columns, label_columns)
    
    print("Creating validation windows...")
    X_val, y_val = create_sliding_windows(val_segments, input_width, label_width, shift,
                                        feature_columns, label_columns)
    
    print("Creating test windows...")
    X_test, y_test = create_sliding_windows(test_segments, input_width, label_width, shift,
                                         feature_columns, label_columns)
    
    # Ensure y has correct shape for TensorFlow (add dimension if needed)
    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)
    if len(y_val.shape) == 1:
        y_val = y_val.reshape(-1, 1)
    if len(y_test.shape) == 1:
        y_test = y_test.reshape(-1, 1)
    
    # Create TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    
    # Shuffle training dataset
    buffer_size = len(X_train)
    train_ds = train_ds.shuffle(buffer_size=buffer_size, seed=get_seed())
    
    # Batch all datasets
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)
    
    # Print dataset sizes
    print(f"Train windows: {len(X_train)} batched in {len(X_train) // batch_size + (1 if len(X_train) % batch_size else 0)} batches")
    print(f"Validation windows: {len(X_val)} batched in {len(X_val) // batch_size + (1 if len(X_val) % batch_size else 0)} batches")
    print(f"Test windows: {len(X_test)} batched in {len(X_test) // batch_size + (1 if len(X_test) % batch_size else 0)} batches")
    
    # Print sample shapes
    for features_batch, labels_batch in train_ds.take(1):
        print(f"Features batch shape: {features_batch.shape}")
        print(f"Labels batch shape: {labels_batch.shape}")
    
    return {
        'train': train_ds,
        'val': val_ds,
        'test': test_ds
    }

@deterministic
def select_features_by_correlation(df, target_column='sap_velocity', threshold=0.1):
    """
    Select features based on correlation with target variable.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of the target column
        threshold: Minimum absolute correlation to include a feature
        
    Returns:
        Tuple of (filtered DataFrame, list of selected feature names)
    """
    from scipy.stats import pearsonr
    
    correlations = []
    features = []
    
    # Calculate correlation for each feature with the target
    for col in df.columns:
        if col != target_column:
            # Handle potential missing values
            valid_data = df[[col, target_column]].dropna()
            if len(valid_data) > 10:  # Ensure enough data for correlation
                corr, _ = pearsonr(valid_data[target_column], valid_data[col])
                correlations.append(abs(corr))
                features.append(col)
    
    # Create a correlation dataframe for logging
    corr_df = pd.DataFrame({'feature': features, 'correlation': correlations})
    corr_df = corr_df.sort_values('correlation', ascending=False)
    
    print("Feature correlations with sap_velocity:")
    for i, row in corr_df.iterrows():
        print(f"  {row['feature']}: {row['correlation']:.4f}")
    
    # Select features with correlation >= threshold
    selected_features = [feat for feat, corr in zip(features, correlations) 
                         if abs(corr) >= threshold]
    
    if not selected_features:
        print(f"Warning: No features with correlation >= {threshold}. Using all features.")
        selected_features = features
    else:
        print(f"Selected {len(selected_features)} features with correlation >= {threshold}:")
        print(selected_features)
    
    # Return filtered dataframe and list of selected features
    return selected_features

def create_cgru_model(input_shape, output_shape, cnn_layers=2, gru_layers=2, 
                     cnn_filters=32, gru_units=128, dropout_rate=0.2,
                     kernel_size=3, pool_size=2):
    """
    Create a hybrid CNN-GRU model (CGRU) matching the architecture from the second script.
    """
    # Force deterministic behavior
    seed = get_seed()
    
    # Create incrementing seeds for each layer
    kernel_seeds = [seed + i for i in range(100)]
    seed_idx = 0
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_layer")
    
    # CNN Block
    x = inputs
    
    for i in range(cnn_layers):
        # CNN layers with kernel size 3 as in the second script
        x = tf.keras.layers.Conv1D(
            filters=cnn_filters * (2 if i == 1 else 1),  # 32 for first layer, 64 for second
            kernel_size=kernel_size,  # Now uses the parameter
            padding='same',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seeds[seed_idx]),
            bias_initializer=tf.keras.initializers.Zeros()
        )(x)
        seed_idx += 1
        
        # MaxPooling with pool size parameter
        x = tf.keras.layers.MaxPooling1D(pool_size=pool_size)(x)
        
        # Dropout as in the second script
        x = tf.keras.layers.Dropout(
            dropout_rate,
            seed=kernel_seeds[seed_idx]
        )(x)
        seed_idx += 1
    
    # GRU Block
    for i in range(gru_layers):
        return_sequences = i < gru_layers - 1
        
        # GRU layer with explicit seed
        x = tf.keras.layers.GRU(
            gru_units,  # Both layers use same units (128)
            return_sequences=return_sequences,
            activation='tanh',
            recurrent_activation='sigmoid',
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seeds[seed_idx]),
            recurrent_initializer=tf.keras.initializers.Orthogonal(seed=kernel_seeds[seed_idx+1]),
            bias_initializer=tf.keras.initializers.Zeros()
        )(x)
        seed_idx += 2
        
        # Dropout with higher rate as in the second script
        x = tf.keras.layers.Dropout(
            dropout_rate,
            seed=kernel_seeds[seed_idx]
        )(x)
        seed_idx += 1
    
    # Dense layers - using 64 units as in the second script
    x = tf.keras.layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seeds[seed_idx]),
        bias_initializer=tf.keras.initializers.Zeros()
    )(x)
    seed_idx += 1
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        output_shape,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seeds[seed_idx]),
        bias_initializer=tf.keras.initializers.Zeros()
    )(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model with lower learning rate (0.001) as in the second script
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    
    # Print model summary
    model.summary()
    
    return model

@deterministic
def main():
    # Set parameters for the model
    INPUT_WIDTH = 4   # Use 16 time steps as input, as in the paper
    LABEL_WIDTH = 1    # Predict 1 time step ahead
    SHIFT = 1          # Predict 1 step ahead
    BATCH_SIZE = 64    # Match batch size from second script
    
    # Set deterministic rendering for matplotlib
    import matplotlib as mpl
    mpl.rcParams['agg.path.chunksize'] = 10000
    np.random.seed(get_seed())  # Ensure matplotlib's random operations use our seed
    
    # Load data
    print("Loading data...")
    data_dir = Path('./outputs/processed_data/merged/site/gap_filled_size1_with_era5')
    data_list = list(data_dir.glob('test.csv'))
    
    if not data_list:
        print(f"No CSV files found in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(data_list)} data files")
    
    # Process data files
    all_segments = []
    # Use these columns as features (based on paper)
    # 'sap_velocity' is the target
    # 'sw_in', 'ext_rad', 'ta', 'ws', 'vpd', 'rh', 'ppfd_in' are features from the paper
    used_cols = ['sap_velocity', 'sw_in', 'ext_rad', 'ta', 'ws', 'vpd', 'rh', 'ppfd_in']
    all_biome_types = set()  # Will collect all unique biome types

    # Sort data files for deterministic processing order
    data_list = sorted(data_list)

    for data_file in data_list:
        print(f"Processing {data_file.name}")
        try:
            df = pd.read_csv(data_file, parse_dates=['TIMESTAMP'])
            df.set_index('TIMESTAMP', inplace=True)
            df = df.resample('h').mean()  # Resample to 30-minute intervals as in the paper
            #df = df[df.index.year == 2012]  # Filter out data before 2003
            
            # Verify columns exist
            missing_cols = [col for col in used_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns in {data_file.name}: {missing_cols}")
                continue
                
            # Fix potential case/whitespace issues
            df.columns = [col.strip().lower() for col in df.columns]
            used_cols_lower = [col.lower() for col in used_cols]
            df = df[used_cols_lower]
            
            # Process biome column if present
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
            
            # Segment the data - handling gaps properly for time series
            segments = TimeSeriesSegmenter.segment_time_series(
                df, 
                gap_threshold=2, 
                unit='hours',  # Changed to hours since we're using 30-minute data
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

    # Sort all biome types for consistent column order
    all_biome_types = sorted(all_biome_types)

    # Ensure all segments have all biome types as columns in the same order
    for segment in all_segments:
        for biome_type in all_biome_types:
            if biome_type not in segment.columns:
                segment[biome_type] = 0.0

    # Now standardize the data
    all_data = pd.concat(all_segments)
    
    # NEW: Apply feature selection based on correlation (from second script)
    print("Selecting features based on correlation...")
    selected_features = select_features_by_correlation(all_data, target_column='sap_velocity', threshold=0.1)
    
    # Get base feature columns and add time features
    base_feature_columns = [col for col in selected_features if not col.startswith('Day') and 
                           not col.startswith('Year') and not col.startswith('Week') and
                           not col.startswith('Month')]
    
    # Add time features which are likely important
    time_features = [col for col in all_data.columns if col.endswith('sin') or col.endswith('cos')]
    feature_columns = base_feature_columns + time_features
    
    # Add biome columns if they exist
    biome_columns = list(all_biome_types)
    if biome_columns:
        feature_columns = feature_columns + biome_columns
    
    print(f"Final feature set: {feature_columns}")
    
    # Create scalers
    from sklearn.preprocessing import MinMaxScaler  # From second script
    feature_scaler = MinMaxScaler()  # Changed from StandardScaler to MinMaxScaler
    label_scaler = MinMaxScaler()    # Changed from StandardScaler to MinMaxScaler
    
    # Fit scalers
    feature_scaler.fit(all_data[feature_columns])
    label_scaler.fit(all_data[['sap_velocity']])
    
    # Apply scaling to each segment
    scaled_segments = []
    for segment in all_segments:
        segment_copy = segment.copy()
        segment_copy[feature_columns] = feature_scaler.transform(segment[feature_columns])
        segment_copy['sap_velocity'] = label_scaler.transform(segment[['sap_velocity']])
        scaled_segments.append(segment_copy)
    
    # NEW: Create datasets using our sliding window approach
    print("Creating datasets with sliding windows...")
    datasets = create_datasets_with_sliding_windows(
        segments=scaled_segments,
        input_width=INPUT_WIDTH,
        label_width=LABEL_WIDTH,
        shift=SHIFT,
        feature_columns=feature_columns,
        label_columns=['sap_velocity'],
        train_val_test_split=(0.8, 0.1, 0.1),
        batch_size=BATCH_SIZE
    )

    train_ds = datasets['train']
    val_ds = datasets['val']
    test_ds = datasets['test']

    # Check dataset properties
    if train_ds is None:
        print("ERROR: Could not create training dataset.")
        sys.exit(1)
    
    # Get the first batch to determine input shape
    for features_batch, labels_batch in train_ds.take(1):
        print(f"Features batch shape: {features_batch.shape}")
        print(f"Labels batch shape: {labels_batch.shape}")
        n_features = features_batch.shape[2]
        break

    # Add early stopping callback (from second script)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Parameter grid matching the second script
    param_grid = {
        'architecture': {
            'cnn_layers': [2],           # Second script uses 2 CNN layers
            'gru_layers': [2],           # Second script uses 2 GRU layers
            'cnn_filters': [32],         # First CNN layer uses 32 filters
            'gru_units': [128],          # Both GRU layers use 128 units 
            'dropout_rate': [0.2],       # Second script uses 0.2 dropout (vs 0.01)
            'kernel_size': [3],          # Second script uses kernel_size=3 (vs 1)
            'pool_size': [2]             # Both scripts use pool_size=2
        },
        'optimizer': {
            'name': ['Adam'],
            'learning_rate': [0.001]     # Second script uses 0.001 (vs 0.01)
        },
        'training': {
            'batch_size': [BATCH_SIZE],
            'epochs': [100],
            'patience': [10]             # Second script uses early stopping with patience=10
        }
    }
    
    # Create and fit optimizer with deterministic behavior
    print("Starting hyperparameter optimization for CGRU model...")
    optimizer = DLOptimizer(
        base_architecture=create_cgru_model,  # Use our CGRU model function
        task='regression',
        model_type='CGRU',                   # Update model type
        param_grid=param_grid,
        input_shape=(INPUT_WIDTH, n_features),
        output_shape=LABEL_WIDTH,
        scoring='val_loss',
        # Pass random_state 
        random_state=get_seed(),
        use_distribution=False  # Use deterministic behavior
    )
    
    # Set additional TensorFlow determinism options before fitting
    try:
        # For newer TensorFlow versions
        tf.keras.utils.set_random_seed(get_seed())
    except:
        # For older TensorFlow versions
        tf.random.set_seed(get_seed())
        np.random.seed(get_seed())
    
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
    
    # Custom fitting for time series data with early stopping callback
    optimizer.fit(
        X_train, 
        y_train,
        is_cv=False,  # We're not using cross-validation
        X_val=X_val,  # Validation data
        y_val=y_val,  # Validation labels
        split_type='temporal',  # We've already created our datasets
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
    plt.title('CGRU Model: Test Predictions vs Actual')
    plt.savefig(plot_dir / 'cgru_test_predictions.png')
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
    plt.title('CGRU Model: Training Predictions vs Actual')
    plt.savefig(plot_dir / 'cgru_train_predictions.png')
    plt.close()
    
    # Plot error distribution for test data
    error = test_predictions - test_labels_actual
    plt.figure(figsize=(10, 7))
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [Sap Velocity]')
    plt.ylabel('Count')
    plt.title('CGRU Model: Test Error Distribution')
    plt.savefig(plot_dir / 'cgru_test_error_distribution.png')
    plt.close()
    
    # Plot error distribution for training data
    error_train = train_predictions - train_labels_actual
    plt.figure(figsize=(10, 7))
    plt.hist(error_train, bins=25)
    plt.xlabel('Prediction Error [Sap Velocity]')
    plt.ylabel('Count')
    plt.title('CGRU Model: Training Error Distribution')
    plt.savefig(plot_dir / 'cgru_train_error_distribution.png')
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
        plt.title('CGRU Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.subplot(1, 2, 2)
        plt.plot(optimizer.history_['mae'])
        plt.plot(optimizer.history_['val_mae'])
        plt.title('CGRU Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'cgru_training_history.png')
        plt.close()
    
    return test_rmse

if __name__ == '__main__':
    main()