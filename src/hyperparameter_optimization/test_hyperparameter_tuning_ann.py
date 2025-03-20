"""
ANN model with comprehensive random control for full determinism.
Following the pattern from the CNN-LSTM implementation to ensure
consistent results across multiple runs.
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
    set_seed, get_seed, get_initializer, deterministic,
    random_state_manager, test_determinism
)

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

# Import the hyperparameter optimizer
from src.hyperparameter_optimization.hyper_tuner import DLOptimizer

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
def create_nn(input_shape, output_shape, n_layers, units, dropout_rate):
    """
    Create a neural network model with explicit seeds for all random operations.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of the input data
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
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    
    # Hidden layers with explicit seeds for weight initialization
    for i in range(n_layers):
        model.add(keras.layers.Dense(
            units,
            activation='relu',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seeds[seed_idx]),
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ))
        seed_idx += 1
        
        # Add batch normalization for better training stability
        model.add(keras.layers.BatchNormalization())
        
        # Add dropout with explicit seed
        if dropout_rate > 0:
            model.add(keras.layers.Dropout(dropout_rate, seed=kernel_seeds[seed_idx]))
            seed_idx += 1
    
    # Output layer with explicit seed
    model.add(keras.layers.Dense(
        output_shape,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seeds[seed_idx]),
        bias_initializer=tf.keras.initializers.Zeros()
    ))
    
    return model

@deterministic
def verify_determinism_of_ann_model():
    """Test if the ANN model produces deterministic results across runs."""
    # Run the model twice with the same seed
    set_seed(42)
    r2_scores1, rmse_scores1 = main(run_id="run1")
    
    set_seed(42)
    r2_scores2, rmse_scores2 = main(run_id="run2")
    
    # Check if results match
    r2_diff = abs(r2_scores1[0] - r2_scores2[0])
    rmse_diff = abs(rmse_scores1[0] - rmse_scores2[0])
    
    print("\nDeterminism Verification Results:")
    print(f"Run 1 Test R²: {r2_scores1[0]:.6f}, RMSE: {rmse_scores1[0]:.6f}")
    print(f"Run 2 Test R²: {r2_scores2[0]:.6f}, RMSE: {rmse_scores2[0]:.6f}")
    print(f"Difference - R²: {r2_diff:.10f}, RMSE: {rmse_diff:.10f}")
    
    is_deterministic = r2_diff < 1e-10 and rmse_diff < 1e-10
    print(f"Model is deterministic: {is_deterministic}")
    
    return is_deterministic

@deterministic
def main(run_id="default"):
    """
    Main function to train and evaluate the ANN model with deterministic behavior.
    
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
    BATCH_SIZE = 64
    
    # Set deterministic rendering for matplotlib
    import matplotlib as mpl
    mpl.rcParams['agg.path.chunksize'] = 10000
    np.random.seed(RANDOM_SEED)  # Ensure matplotlib's random operations use our seed
    
    print(f"Starting ANN model training with seed {RANDOM_SEED} (run_id: {run_id})")
    
    # Load and preprocess data in a deterministic manner
    data = pd.read_csv('./outputs/processed_data/merged/site/gap_filled_size1_with_era5/all_biomes_merged_data.csv')
    
    # Sort data by timestamp to ensure deterministic ordering
    data = data.sort_values('TIMESTAMP').set_index('TIMESTAMP')
    
    # Select columns - use exact order for determinism
    columns = [
        'sap_velocity',
        'vpd', 
        'ta',
        'swc_shallow', 
        'ppfd_in', 
        'volumetric_soil_water_layer_4', 
        'leaf_area_index_low_vegetation', 
        'soil_temperature_level_3', 
        'volumetric_soil_water_layer_3', 
        'biome', 
        'swc_deep', 
        'u_component_of_wind_10m', 
        'v_component_of_wind_10m',
    ]
    
    # Check columns exist
    missing_cols = [col for col in columns if col not in data.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        # Use available columns
        columns = [col for col in columns if col in data.columns]
    
    data = data[columns].dropna()
    
    # Convert biome to categorical
    if 'biome' in data.columns:
        # Sort unique values for consistent ordering
        unique_biomes = sorted(data['biome'].unique())
        # Convert to categorical with ordered categories
        data['biome'] = pd.Categorical(data['biome'], categories=unique_biomes)
    
    # Get dummy variables for biome with consistent order
    if 'biome' in data.columns:
        # Create dummies with fixed parameters
        biome_dummies = pd.get_dummies(data['biome'], drop_first=False, prefix='biome')
        # Drop original biome column and join dummies
        data = data.drop(columns=['biome'])
        data = pd.concat([data, biome_dummies], axis=1)
    
    print("Data columns after preprocessing:", data.columns.tolist())
    
    # Add time features
    data = add_time_features(data)
    
    # Use a fixed, deterministic data split
    training_portion = 0.8
    
    # Use chronological split for time series data
    data_indices = np.arange(len(data))
    split_idx = int(len(data) * training_portion)
    
    # Use fixed indices for the split
    train_indices = data_indices[:split_idx]
    test_indices = data_indices[split_idx:]
    
    # Split using indices
    training_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]
    
    print(f"Training data: {len(training_data)} samples")
    print(f"Test data: {len(test_data)} samples")
    
    # Select features and target
    train_features = training_data.drop(columns=['sap_velocity'])
    train_labels = training_data['sap_velocity']
    test_features = test_data.drop(columns=['sap_velocity'])
    test_labels = test_data['sap_velocity']
    
    # Clean data and handle NaN/inf values
    train_features = train_features.replace([np.inf, -np.inf], np.nan)
    train_labels = train_labels.replace([np.inf, -np.inf], np.nan)
    test_features = test_features.replace([np.inf, -np.inf], np.nan)
    test_labels = test_labels.replace([np.inf, -np.inf], np.nan)
    
    # Remove rows with NaN values
    mask_train = ~(train_features.isna().any(axis=1) | train_labels.isna())
    train_features = train_features[mask_train]
    train_labels = train_labels[mask_train]
    
    mask_test = ~(test_features.isna().any(axis=1) | test_labels.isna())
    test_features = test_features[mask_test]
    test_labels = test_labels[mask_test]
    
    # Standardize the data with fixed parameters
    scaler = StandardScaler()
    train_labels_reshaped = train_labels.values.reshape(-1, 1)
    scaler.fit(train_labels_reshaped)  # Fit only on training data
    
    train_labels_scaled = scaler.transform(train_labels_reshaped)
    test_labels_scaled = scaler.transform(test_labels.values.reshape(-1, 1))
    
    feature_scaler = StandardScaler()
    feature_scaler.fit(train_features)  # Fit only on training data
    
    train_features_scaled = feature_scaler.transform(train_features)
    test_features_scaled = feature_scaler.transform(test_features)
    
    # Define parameter grid for hyperparameter tuning
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
            'batch_size': BATCH_SIZE,
            'epochs': 100,
            'patience': 20
        }
    }
    
    # Create custom callbacks for training
    callbacks = [
        # Early stopping with deterministic behavior
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=param_grid['training']['patience'],
            restore_best_weights=True,
            verbose=1
        ),
        # Reset seed before each epoch
        tf.keras.callbacks.LambdaCallback(
            on_epoch_begin=lambda epoch, logs: tf.random.set_seed(RANDOM_SEED + epoch)
        )
    ]
    
    # Create and fit optimizer with deterministic settings
    optimizer = DLOptimizer(
        base_architecture=create_nn,
        task='regression',
        model_type='nn',
        param_grid=param_grid,
        input_shape=(train_features_scaled.shape[1],),
        output_shape=1,
        scoring='val_loss',
        random_state=RANDOM_SEED  # Pass our random state to the optimizer
    )
    
    # Set additional determinism controls before fitting
    tf.keras.utils.set_random_seed(RANDOM_SEED)  # TF 2.7+ comprehensive seeding
    
    # Fit the optimizer with deterministic validation split
    optimizer.fit(
        train_features_scaled,
        train_labels_scaled,
        split_type='random',  # Use split_type='temporal' for time series
        groups=None,
        random_state=RANDOM_SEED,  # Use consistent random state for any splits
        
    )
    
    # Get the best model
    best_model = optimizer.get_best_model()
    
    # Reset seed before prediction to ensure consistency
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Predict with deterministic behavior
    test_predictions = best_model.predict(test_features_scaled, verbose=0, batch_size=BATCH_SIZE).flatten()
    train_predictions = best_model.predict(train_features_scaled, verbose=0, batch_size=BATCH_SIZE).flatten()
    
    # Reverse the normalization
    test_predictions = scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
    train_predictions = scaler.inverse_transform(train_predictions.reshape(-1, 1)).flatten()
    test_labels_actual = test_labels.values
    train_labels_actual = train_labels.values
    
    # Create plots directory if it doesn't exist
    plot_dir = Path('./plots')
    plot_dir.mkdir(exist_ok=True)
    
    # Set deterministic plotting
    plt.rcParams['agg.path.chunksize'] = 10000
    np.random.seed(RANDOM_SEED)
    
    # Plot predictions vs actual for test data
    plt.figure(figsize=(10, 10))
    plt.scatter(test_labels_actual, test_predictions, alpha=0.5)
    plt.xlabel('ANN True Values [Sap Velocity] on Test Data')
    plt.ylabel('ANN Predictions [Sap Velocity] on Test Data')
    plt.title('ANN Predictions vs True Values on Test Data')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    plt.plot([-100, 100], [-100, 100])
    plt.savefig(plot_dir / f'predictions_vs_actual_{run_id}.png')
    plt.close()
    
    # Plot predictions vs actual for training data
    plt.figure(figsize=(10, 10))
    plt.scatter(train_labels_actual, train_predictions, alpha=0.5)
    plt.xlabel('ANN True Values [Sap Velocity] on Training Data')
    plt.ylabel('ANN Predictions [Sap Velocity] on Training Data')
    plt.title('ANN Predictions vs True Values on Training Data')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    plt.plot([-100, 100], [-100, 100])
    plt.savefig(plot_dir / f'train_predictions_vs_actual_{run_id}.png')
    plt.close()
    
    # Plot error distribution for test data
    error = test_predictions - test_labels_actual
    plt.figure(figsize=(10, 7))
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [Sap Velocity]')
    plt.ylabel('Count')
    plt.title('Test Error Distribution')
    plt.savefig(plot_dir / f'error_distribution_{run_id}.png')
    plt.close()
    
    # Plot error distribution for training data
    error_train = train_predictions - train_labels_actual
    plt.figure(figsize=(10, 7))
    plt.hist(error_train, bins=25)
    plt.xlabel('Prediction Error [Sap Velocity]')
    plt.ylabel('Count')
    plt.title('Training Error Distribution')
    plt.savefig(plot_dir / f'error_distribution_train_{run_id}.png')
    plt.close()
    
    # Calculate metrics with fixed random state
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
    print(f"R² Score (test): {r2:.6f}")
    print(f"R² Score (train): {r2_train:.6f}")
    
    print("\nTest Metrics:")
    print(f"MSE: {test_mse:.6f}")
    print(f"RMSE: {test_rmse:.6f}")
    print(f"MAE: {test_mae:.6f}")
    
    print("\nTrain Metrics:")
    print(f"MSE: {train_mse:.6f}")
    print(f"RMSE: {train_rmse:.6f}")
    print(f"MAE: {train_mae:.6f}")
    
    # Print optimization summary
    print("\nOptimization Summary:")
    print(f"Best parameters: {optimizer.best_params_}")
    print(f"Best validation score: {optimizer.best_score_:.6f}")
    
    # Plot training history if available
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
        plt.savefig(plot_dir / f'training_history_{run_id}.png')
        plt.close()
    
    # Save the model with deterministic filename
    model_dir = Path('./models')
    model_dir.mkdir(exist_ok=True)
    best_model.save(model_dir / f'ann_model_seed_{RANDOM_SEED}_{run_id}.h5')
    
    # Return metrics for determinism testing
    return (r2, r2_train), (test_rmse, train_rmse)


if __name__ == "__main__":
    # Verify determinism by running the entire pipeline twice
    is_deterministic = verify_determinism_of_ann_model()
    
    if is_deterministic:
        print("\nSuccess! The ANN model produces deterministic results.")
        # Run the final model with full determinism
        main(run_id="final")
    else:
        print("\nWarning: The ANN model is NOT fully deterministic. Review the implementation.")
        # Additional debugging information could be added here