"""
Deterministic LSTM model for time series prediction with comprehensive
randomness control for reproducible results.
"""
import os
import sys
from pathlib import Path
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

# Import frameworks with global seed control
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers
from tensorflow import keras

# Set global seeds
tf.random.set_seed(get_seed())
np.random.seed(get_seed())

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
        #config.intra_op_parallelism_threads = 1
        #config.inter_op_parallelism_threads = 1
        session = InteractiveSession(config=config)
    except:
        print("Warning: Could not configure session-level determinism")

# Limit TensorFlow to use only one thread for CPU operations
#tf.config.threading.set_inter_op_parallelism_threads(1)
#tf.config.threading.set_intra_op_parallelism_threads(1)

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
    
    print(f"Starting windowed ANN model training with seed {RANDOM_SEED} (run_id: )")
    
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
        print(f"Missing biome types: {set(all_possible_biome_types) - set(all_biome_types)}")

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

    
    # Define parameter grid for LSTM with deterministic settings
    param_grid = {
        'architecture': {
            'n_layers': [2, 3],
            'units': [16, 32, 64],
            'dropout_rate': [0.2]
        },
        'optimizer': {
            'name': ['Adam'],
            'learning_rate': [0.001]
        },
        'training': {
            'batch_size': [ BATCH_SIZE],
            'epochs': 100,
            'patience': [20]
        }
    }
    
    # Create and fit optimizer with deterministic behavior
    print("Starting hyperparameter optimization...")
    optimizer = DLOptimizer(
        base_architecture=create_lstm_model,
        task='regression',
        model_type='lstm',
        param_grid=param_grid,
        input_shape=(INPUT_WIDTH, n_features),  # (timesteps, features)
        output_shape=LABEL_WIDTH,  # Output prediction length
        scoring='val_loss',
        use_distribution=False,  # Use deterministic behavior
        random_state=get_seed(),  # Add random state for deterministic behavior
    )
    
    # Convert TensorFlow dataset to X and y numpy arrays in a deterministic way
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
    
    
    
    print("\nAnalysis complete, visualizations saved.")
    
    # Return some key metric for determinism testing
    return test_rmse



if __name__ == "__main__":
    # Run the main function with deterministic controls
    print("\nRunning main function with enhanced determinism...")
    main()