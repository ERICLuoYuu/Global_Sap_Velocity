import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, GRU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Load and prepare data
# Assuming you have a CSV file with timestamp, sap flow, and environmental factors
def load_data(file_path):
    """
    Load the sap flow dataset.
    Expected columns: timestamp, sap_flow, and environmental factors 
    (Ta, VPD, WS, PPFD, SW, RH, RAD)
    """
    df = pd.read_csv(file_path, parse_dates=['TIMESTAMP'])
    df = df[['TIMESTAMP','sap_velocity', 'sw_in', 'ext_rad', 'ta', 'ws', 'vpd', 'rh', 'ppfd_in']].dropna()
    df.set_index('TIMESTAMP', inplace=True)
    
    # Ensure 30-minute intervals (if needed)
    df = df.resample('h').mean().dropna()
    
    # Select one year of data (as in the paper)
    # df = df[df.index.year == 2012]
    print(f"Loaded data for {len(df)} time steps")
    
    return df

# Step 2: Feature selection using correlation analysis
def select_features(df, threshold=0):
    """
    Select features based on correlation with sap flow.
    Return dataframe with selected features.
    """
    correlations = []
    features = []
    
    # Calculate correlation for each environmental factor
    for col in df.columns:
        if col != 'sap_velocity':
            corr, _ = pearsonr(df['sap_velocity'], df[col])
            correlations.append(abs(corr))
            features.append(col)
    
    # Create a correlation dataframe for visualization
    corr_df = pd.DataFrame({'feature': features, 'correlation': correlations})
    corr_df = corr_df.sort_values('correlation', ascending=False)
    
    # Plot correlations
    plt.figure(figsize=(10, 6))
    sns.barplot(x='feature', y='correlation', data=corr_df)
    plt.axhline(y=threshold, color='r', linestyle='--')
    plt.title('Correlation with Sap Flow')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('feature_correlations.png')
    
    # Select features with correlation >= threshold
    selected_features = [feat for feat, corr in zip(features, correlations) if abs(corr) >= threshold]
    
    return df[['sap_velocity'] + selected_features], selected_features

# Step 3: Data normalization
def normalize_data(df):
    """
    Normalize data using min-max scaling to [0,1] range.
    """
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df)
    
    # Create a new dataframe with normalized values
    normalized_df = pd.DataFrame(normalized_data, columns=df.columns, index=df.index)
    
    return normalized_df, scaler

# Step 4: Split data into train, validation, test sets
def split_data(df, test_ratio=0.2):
    """
    Split data into training and testing sets temporally.
    """
    n = len(df)
    test_size = int(n * test_ratio)
    
    train_val_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]
    
    return train_val_df, test_df

# Step 5: Create sliding windows
def create_sliding_windows(data, window_size, feature_indices, target_index=0):
    """
    Create sliding windows for time series prediction.
    Parameters:
        data: numpy array of shape [samples, features]
        window_size: number of time steps to include in each window
        feature_indices: list of indices for the features to use
        target_index: index of the target variable (default=0 for sap_flow)
    Returns:
        X: input sequences [samples, time steps, features]
        y: target values [samples, 1]
    """
    X, y = [], []
    
    for i in range(window_size, len(data)):
        # Extract window of observations
        window = data[i - window_size:i, :]
        features = window[:, feature_indices]
        target = data[i, target_index]
        
        X.append(features)
        y.append(target)
    
    return np.array(X), np.array(y)

# Step 6: Create CGRU model
def create_cgru_model(input_shape, window_size, learning_rate=0.001):
    """
    Create the CGRU hybrid model combining CNN and GRU.
    Adapted to work with various window sizes.
    """
    # Input layer
    inputs = Input(shape=input_shape, name="input_layer")
    
    # Calculate appropriate pool size based on window size
    pool_size = 2
    
    # CNN layers with more filters
    x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Dropout(0.2)(x)
    
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    x = Dropout(0.2)(x)
    
    # GRU layers with higher capacity
    x = GRU(128, activation='tanh', return_sequences=True, recurrent_dropout=0.0)(x)
    x = Dropout(0.2)(x)
    
    x = GRU(128, activation='tanh')(x)
    x = Dropout(0.2)(x)
    
    # Dense layers with more regularization
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    outputs = Dense(1)(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model with adjusted learning rate
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    
    # Print model summary for debugging
    model.summary()
    
    return model

# Step 7: Model training and evaluation
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, 
                           epochs=100, batch_size=32):
    """
    Train the model and evaluate its performance.
    """
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.savefig('training_history.png')
    
    # Evaluate on test set
    y_pred = model.predict(X_test).flatten()
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    train_r2 = r2_score(y_train, model.predict(X_train).flatten())

    
    # Safe MAPE calculation that avoids division by zero
    mask = y_test != 0
    if np.sum(mask) == 0:
        mape = float('inf')
    else:
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    
    print(f"Test MSE: {mse:.5f}")
    print(f"Test R²: {r2:.5f}")
    print(f"Test MAPE: {mape:.2f}%")
    print(f"Train R²: {train_r2:.5f}")
    # Additional diagnostics
    print(f"\nFirst 10 predictions vs actual values:")
    for i in range(10):
        print(f"Actual: {y_test[i]:.4f}, Predicted: {y_pred[i]:.4f}, Error: {((y_pred[i] - y_test[i])/max(y_test[i], 1e-10)):.2%}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:500], 'r', label='Actual')
    plt.plot(y_pred[:500], 'b', label='Predicted')
    plt.title('Sap Flow: Actual vs Predicted')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Sap Flow')
    plt.legend()
    plt.savefig('predictions_vs_actual.png')
    
    return {
        'mse': mse,
        'r2': r2,
        'mape': mape,
        'y_pred': y_pred
    }

# Step 8: Run the full pipeline
def run_sap_flow_prediction(file_path, window_size=16, correlation_threshold=0, learning_rate=0.001):
    """
    Run the complete sap flow prediction pipeline.
    
    Args:
        file_path: Path to the CSV file with sap flow data
        window_size: Number of previous time steps to use
        correlation_threshold: Threshold for feature selection
        learning_rate: Learning rate for the model
    """
    # Load data
    print("Loading data...")
    data = load_data(file_path)
    
    # Print basic dataset info
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    
    # Check for missing values
    missing = data.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values:\n{missing[missing > 0]}")
        print("Filling missing values with forward fill followed by backward fill")
        data = data.ffill().bfill()
    
    # Select features with lower threshold to include more features
    print(f"Selecting features with correlation threshold: {correlation_threshold}...")
    data_selected, selected_features = select_features(data, threshold=correlation_threshold)
    
    print(f"Selected features: {selected_features}")
    
    # Normalize data
    print("Normalizing data...")
    data_norm, scaler = normalize_data(data_selected)
    
    # Split data temporally 
    print("Splitting data...")
    train_val_data, test_data = split_data(data_norm)
    
    # Further split train_val into train and validation (using temporal split)
    # This is different from the paper which uses random split, but temporal is better for time series
    train_size = int(len(train_val_data) * 0.8)
    train_data = train_val_data.iloc[:train_size]
    val_data = train_val_data.iloc[train_size:]
    
    # Convert to numpy arrays
    train_data_values = train_data.values
    val_data_values = val_data.values
    
    # Create sliding windows
    print(f"Creating sliding windows with size {window_size}...")
    
    # Get indices of features and target
    feature_indices = [list(data_selected.columns).index(col) for col in selected_features]
    target_index = list(data_selected.columns).index('sap_velocity')
    
    X_train, y_train = create_sliding_windows(
        train_data_values, window_size, feature_indices, target_index
    )
    X_val, y_val = create_sliding_windows(
        val_data_values, window_size, feature_indices, target_index
    )
    X_test, y_test = create_sliding_windows(
        test_data.values, window_size, feature_indices, target_index
    )
    
    # Print dataset shapes
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Create and train model
    print("Creating CGRU model...")
    input_shape = (window_size, len(selected_features))
    model = create_cgru_model(input_shape, window_size, learning_rate=learning_rate)
    
    print("Training model...")
    results = train_and_evaluate_model(
        model, X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    return model, results, scaler

# Example usage:
#model, results, scaler = run_sap_flow_prediction("./outputs/processed_data/merged/site/gap_filled_size1_with_era5/test.csv", window_size=16)
model, results, scaler = run_sap_flow_prediction("./outputs/processed_data/merged/site/gap_filled_size1_with_era5/NZL_HUA_HUA_merged.csv", window_size=4)

# If you want to save the model for later use:
# model.save("sap_flow_cgru_model.h5")