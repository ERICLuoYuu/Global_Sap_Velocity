import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import sys
from pathlib import Path
# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from src.hyperparameter_optimization.hyper_tuner import DLOptimizer
# Define the ANN model architecture function for sap flow prediction
def create_ann_architecture(
    input_shape, 
    output_shape=1,
    num_layers=4,
    layer_sizes=[512, 1024, 512, 256],
    activation='relu',
    dropout_rates=[0.3, 0.4, 0.3, 0.2],
    use_batch_norm=True,
    l2_regularization=0.0005,
    **kwargs
):
    """
    Create an ANN architecture for sap flow prediction.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (excluding batch dimension)
    output_shape : int, default=1
        Number of output units
    num_layers : int, default=4
        Number of hidden layers
    layer_sizes : list, default=[512, 1024, 512, 256]
        Size of each hidden layer
    activation : str, default='relu'
        Activation function for hidden layers
    dropout_rates : list, default=[0.3, 0.4, 0.3, 0.2]
        Dropout rate for each hidden layer
    use_batch_norm : bool, default=True
        Whether to use batch normalization
    l2_regularization : float, default=0.0005
        L2 regularization strength
    **kwargs : dict
        Additional arguments for model architecture
        
    Returns:
    --------
    model : tf.keras.Model
        Compiled Keras model
    """
    # Input layer
    inputs = Input(shape=input_shape, name="input_layer")
    x = inputs
    
    # Create hidden layers
    for i in range(min(num_layers, len(layer_sizes))):
        x = Dense(
            layer_sizes[i],
            activation=activation,
            kernel_regularizer=tf.keras.regularizers.l2(l2_regularization)
        )(x)
        
        if use_batch_norm:
            x = BatchNormalization()(x)
            
        if i < len(dropout_rates) and dropout_rates[i] > 0:
            x = Dropout(dropout_rates[i])(x)
    
    # Output layer
    outputs = Dense(output_shape)(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Functions for data preprocessing
def load_and_prepare_data(file_path, correlation_threshold=0.0):
    """
    Load, preprocess, and split the sap flow data.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file with sap flow data
    correlation_threshold : float, default=0.0
        Threshold for feature selection based on correlation with sap velocity
        
    Returns:
    --------
    dict : A dictionary containing the processed data and metadata
    """
    from scipy.stats import pearsonr
    
    # Load data
    df = pd.read_csv(file_path, parse_dates=['TIMESTAMP'])
    df = df[['TIMESTAMP','sap_velocity', 'sw_in', 'ext_rad', 'ta', 'ws', 'vpd', 'rh', 'ppfd_in']].dropna()
    df.set_index('TIMESTAMP', inplace=True)
    
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        df = df.ffill().bfill()
    
    # Select features based on correlation
    correlations = []
    features = []
    
    for col in df.columns:
        if col != 'sap_velocity':
            corr, _ = pearsonr(df['sap_velocity'], df[col])
            correlations.append(abs(corr))
            features.append(col)
    
    selected_features = [feat for feat, corr in zip(features, correlations) if abs(corr) >= correlation_threshold]
    data_selected = df[['sap_velocity'] + selected_features]
    
    # Normalize data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data_selected)
    data_norm = pd.DataFrame(normalized_data, columns=data_selected.columns, index=data_selected.index)
    
    # Split data temporally
    test_ratio = 0.2
    n = len(data_norm)
    test_size = int(n * test_ratio)
    
    train_val_data = data_norm.iloc[:-test_size]
    test_data = data_norm.iloc[-test_size:]
    
    # Further split train_val into train and validation (temporal split)
    train_size = int(len(train_val_data) * 0.8)
    train_data = train_val_data.iloc[:train_size]
    val_data = train_val_data.iloc[train_size:]
    
    return {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data,
        'scaler': scaler,
        'selected_features': selected_features,
        'feature_indices': [list(data_selected.columns).index(col) for col in selected_features],
        'target_index': list(data_selected.columns).index('sap_velocity')
    }

def create_sliding_windows(data, window_size, feature_indices, target_index=0):
    """Create sliding windows for time series prediction"""
    X, y = [], []
    data_values = data.values
    
    for i in range(window_size, len(data_values)):
        window = data_values[i - window_size:i, :]
        features = window[:, feature_indices]
        target = data_values[i, target_index]
        X.append(features)
        y.append(target)
        
    return np.array(X), np.array(y)

def flatten_input_data(X):
    """Flatten 3D input data for ANN model"""
    samples, time_steps, features = X.shape
    X_flat = X.reshape(samples, time_steps * features)
    return X_flat

# Function to run the optimizer for sap flow prediction
def run_ann_optimizer(
    file_path,
    window_size=12,
    correlation_threshold=0.0,
    n_splits=5,
    param_grid=None,
    save_model_dir=None
):
    """
    Run hyperparameter optimization for sap flow prediction with ANN.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file with sap flow data
    window_size : int, default=12
        Number of time steps to use for prediction window
    correlation_threshold : float, default=0.0
        Threshold for feature selection
    n_splits : int, default=5
        Number of cross-validation splits
    param_grid : dict, default=None
        Parameter grid for optimization. If None, uses default parameter grid.
    save_model_dir : str, default=None
        Directory to save the best model
        
    Returns:
    --------
    tuple : (optimizer, data_info)
        The optimizer object and data information dictionary
    """

    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'architecture': {
                'num_layers': [3, 4, 5],
                'layer_sizes': [
                    [256, 512, 256],
                    [512, 1024, 512, 256],
                    [256, 512, 1024, 512, 256]
                ],
                'activation': ['relu', 'elu'],
                'dropout_rates': [
                    [0.2, 0.3, 0.2],
                    [0.3, 0.4, 0.3, 0.2],
                    [0.2, 0.3, 0.4, 0.3, 0.2]
                ],
                'use_batch_norm': [True],
                'l2_regularization': [0.0001, 0.0005, 0.001]
            },
            'optimizer': {
                'name': ['adam'],
                'learning_rate': [0.001, 0.0005, 0.0001]
            },
            'training': {
                'batch_size': [32, 64, 128],
                'epochs': [200],
                'patience': [20]
            }
        }
    
    # Load and prepare data
    data_info = load_and_prepare_data(file_path, correlation_threshold)
    
    # Create sliding windows for each split
    X_train, y_train = create_sliding_windows(
        data_info['train_data'], 
        window_size, 
        data_info['feature_indices'], 
        data_info['target_index']
    )
    
    X_val, y_val = create_sliding_windows(
        data_info['val_data'], 
        window_size, 
        data_info['feature_indices'], 
        data_info['target_index']
    )
    
    X_test, y_test = create_sliding_windows(
        data_info['test_data'], 
        window_size, 
        data_info['feature_indices'], 
        data_info['target_index']
    )
    
    # Flatten input data for ANN
    X_train_flat = flatten_input_data(X_train)
    X_val_flat = flatten_input_data(X_val)
    X_test_flat = flatten_input_data(X_test)
    
    # Print dataset information
    print(f"Dataset shapes:")
    print(f"  X_train: {X_train_flat.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val: {X_val_flat.shape}")
    print(f"  y_val: {y_val.shape}")
    print(f"  X_test: {X_test_flat.shape}")
    print(f"  y_test: {y_test.shape}")
    
    # Configure DLOptimizer
    input_dim = X_train_flat.shape[1]  # window_size * num_features
    
    # Create optimizer
    optimizer = DLOptimizer(
        model_type="ann",
        base_architecture=create_ann_architecture,
        task="regression",
        param_grid=param_grid,
        input_shape=(input_dim,),
        output_shape=1,
        scoring="val_loss",
        n_splits=n_splits,
        random_state=42,
        verbose=1,
        use_distribution=False,  # Set to True if using GPU or distributed training
        # save_model_dir=save_model_dir
    )
    
    # Perform hyperparameter optimization
    # Using direct validation set instead of cross-validation for time series data
    optimizer.fit(
        X_train_flat, y_train,
        is_cv=False,  # Use direct validation instead of CV for time series
        X_val=X_val_flat,
        y_val=y_val
    )
    
    # Get best model and evaluate on test set
    best_model = optimizer.get_best_model()
    y_pred = best_model.predict(X_test_flat).flatten()
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    r2_train = r2_score(y_train, best_model.predict(X_train_flat).flatten())
    
    # Safe MAPE calculation
    mask = y_test != 0
    if np.sum(mask) == 0:
        mape = float('inf')
    else:
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    
    # Print results
    print("\n===== Best Model Results =====")
    print(f"Test MSE: {mse:.5f}")
    print(f"Test RMSE: {rmse:.5f}")
    print(f"Test R²: {r2:.5f}")
    print(f"Train R²: {r2_train:.5f}")
    print(f"Test MAPE: {mape:.2f}%")
    print(f"Improvement over baseline R² ({0.35344}): {((r2 - 0.35344) / 0.35344) * 100:.2f}%")
    
    # Plot predictions vs actual
    plt.figure(figsize=(15, 8))
    
    # Main plot - first 500 points
    n_points = min(500, len(y_test))
    plt.plot(y_test[:n_points], 'r-', label='Actual')
    plt.plot(y_pred[:n_points], 'b-', label='Predicted')
    plt.title(f'Sap Flow: Actual vs Predicted (Best Model, R² = {r2:.4f})')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Sap Flow')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('best_model_predictions.png')
    
    # Store all relevant information
    data_info.update({
        'X_train': X_train_flat,
        'y_train': y_train,
        'X_val': X_val_flat,
        'y_val': y_val,
        'X_test': X_test_flat,
        'y_test': y_test,
        'y_pred': y_pred,
        'metrics': {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    })
    
    return optimizer, data_info

# Example usage
# Define a smaller parameter grid for testing
test_param_grid = {
        'architecture': {
            'n_layers': [1],
            'units': [16],
            'dropout_rate': [0.2] 
        },
        'optimizer': {   
            'name': ['Adam'],
            'learning_rate': [0.001]
        },
        'training': {
            'batch_size': 64,
            'epochs': 200,
            'patience': 10
        }
    }

# To run the optimizer with your data:
optimizer, data_info = run_ann_optimizer(
     "./outputs/processed_data/merged/site/gap_filled_size1_with_era5/NZL_HUA_HUA_merged.csv",
     window_size=4,
     param_grid=test_param_grid
 )