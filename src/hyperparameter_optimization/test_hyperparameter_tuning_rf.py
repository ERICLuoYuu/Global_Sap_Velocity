import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.hyperparameter_optimization.hyper_tuner import MLOptimizer
from src.tools import create_spatial_groups
# Import time series processing modules for windowing
from src.hyperparameter_optimization.timeseries_processor import TimeSeriesSegmenter, SegmentedWindowGenerator

# Set seed for reproducibility
np.random.seed(42)

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

def main():
    # Define window parameters (matching CNN-LSTM implementation)
    INPUT_WIDTH = 8   # Use 8 time steps as input
    LABEL_WIDTH = 1   # Predict 1 time step ahead
    SHIFT = 1         # Predict 1 step ahead
    BATCH_SIZE = 32   # Batch size (used by window generator)
    
    # Set parameters for matplotlib
    plt.rcParams['agg.path.chunksize'] = 10000
    np.random.seed(42)  # Ensure matplotlib's random operations use seed
    
    print("Starting XGBoost model training")
    
    # Load data - exactly as in CNN-LSTM
    print("Loading data...")
    data_dir = Path('./outputs/processed_data/merged/site/gap_filled_size1_hourly_after_filter')
    data_list = list(data_dir.glob('*merged.csv'))
    
    if not data_list:
        print(f"No CSV files found in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(data_list)} data files")
    
    # Process data files
    all_segments = []
    used_cols = ['sap_velocity', 'ext_rad', 'sw_in', 'ta', 'ws', 'vpd','ppfd_in', 'mean_annual_temp', 'mean_annual_precip', 'Day sin', 'Week sin', 'Month sin', 'Year sin', ]
    ['sap_velocity', 'ext_rad', 'ta', 'vpd', 'biome', 'Day sin', 'Week sin', 'Month sin', 'Year sin']
  
    all_biome_types = set()  # Will collect all unique biome types
    all_possible_biome_types = ['Boreal forest', 'Subtropical desert', 'Temperate forest', 'Temperate grassland desert', 'Temperate rain forest', 'Tropical forest savanna', 'Tropical rain forest', 'Tundra', 'Woodland/Shrubland']
    # Sort data files for deterministic processing order
    data_list = sorted(data_list)

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
            df = df[used_cols_lower]
            
            # Define all possible biomes in your desired order


            # Replace the current biome processing code with this:
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
            
            # Segment the data
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
        print("No valid segments found. Check your data files.")
        sys.exit(1)

    print(f"Total segments collected: {len(all_segments)}")
    print(f"All biome types found: {all_biome_types}")

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
    
    # Create windowed datasets with deterministic options
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
            exclude_labels_from_inputs=True
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
    for features_batch, labels_batch in train_ds:
        print(f"Features batch shape: {features_batch.shape}")
        print(f"Labels batch shape: {labels_batch.shape}")
        break
    
    # Convert TensorFlow datasets to numpy arrays for rf
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
        
        # Reshape for rf (flatten window dimension)
        X_reshaped = X.reshape(X.shape[0], -1)
        
        # If labels are shape (batch, seq_len, features) and you need (batch, features)
        if y.ndim > 2:
            y = y.reshape(y.shape[0], -1)
        
        return X_reshaped, y.flatten()

    # Using the function to prepare data for rf
    X_train, y_train = convert_tf_dataset_to_xy(train_ds)
    X_val, y_val = convert_tf_dataset_to_xy(val_ds)
    X_test, y_test = convert_tf_dataset_to_xy(test_ds)
    
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")

    # do optimatization for ML model
    # Define parameter grid

    param_grid = {
    'n_estimators': [800],  # More trees to capture complex patterns
    'max_depth': [15],  # Deeper trees to model more complex relationships
    'min_samples_split': [3],  # Lower values allow more splits
    'min_samples_leaf': [1],  # Minimum value allows leaves to be as specific as possible
    'max_features': [0.7],  # More features per split ('auto' is same as 'sqrt' for classification, all for regression)
    'bootstrap': [True],
    'random_state': [42],
    'oob_score': [True],
    'max_samples': [0.9]  # Use more of the data in each tree
}

    # Train the model with optimization
    print("Training RandomForest model with hyperparameter optimization...")
    optimizer = MLOptimizer(
        param_grid=param_grid, 
        scoring='r2', 
        model_type='rf', 
        task='regression',
        random_state=42  # For reproducibility
    )

    # Fit with validation data
    optimizer.fit(
        X_train, 
        y_train,
        X_val=X_val,
        y_val=y_val,
        is_cv=False,  # Don't use cross-validation
        is_shuffle=False  # Don't shuffle time series data
    )

    # Get best model
    best_model = optimizer.best_estimator_
    print(f"Best parameters: {optimizer.best_params_}")
    print(f"Best validation score: {optimizer.best_score_}")

    # Predict on test data
    y_pred_scaled = best_model.predict(X_test)

    # Get predictions on train data
    y_train_pred_scaled = best_model.predict(X_train)

    # Inverse transform predictions to original scale
    y_pred = label_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_train_pred = label_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()

    # Inverse transform labels to original scale for evaluation
    y_test_actual = label_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_train_actual = label_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()

    # Calculate metrics
    r2 = r2_score(y_test_actual, y_pred)
    mse = mean_squared_error(y_test_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, y_pred)

    r2_train = r2_score(y_train_actual, y_train_pred)
    mse_train = mean_squared_error(y_train_actual, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    mae_train = mean_absolute_error(y_train_actual, y_train_pred)

    print("\nModel Evaluation:")
    print("=================")
    print(f"R² Score (test): {r2:.6f}")
    print(f"R² Score (train): {r2_train:.6f}")

    print("\nTest Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")

    print("\nTrain Metrics:")
    print(f"MSE: {mse_train:.6f}")
    print(f"RMSE: {rmse_train:.6f}")
    print(f"MAE: {mae_train:.6f}")

    # Create plots directory if it doesn't exist
    plot_dir = Path('./plots')
    plot_dir.mkdir(exist_ok=True)

    # Plot scatter plots for predictions vs actual
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test_actual, y_pred, alpha=0.5)
    plt.xlabel('True Values [Sap Velocity]')
    plt.ylabel('Predictions [Sap Velocity]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, max(plt.xlim()[1], plt.ylim()[1])])
    plt.ylim([0, max(plt.xlim()[1], plt.ylim()[1])])
    plt.plot([-100, 100], [-100, 100])
    plt.title('rf Model: Test Predictions vs Actual')
    plt.savefig(plot_dir / 'rf_test_predictions.png')
    plt.close()

    # Plot predictions vs actual for training data
    plt.figure(figsize=(10, 10))
    plt.scatter(y_train_actual, y_train_pred, alpha=0.5)
    plt.xlabel('True Values [Sap Velocity]')
    plt.ylabel('Predictions [Sap Velocity]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, max(plt.xlim()[1], plt.ylim()[1])])
    plt.ylim([0, max(plt.xlim()[1], plt.ylim()[1])])
    plt.plot([-100, 100], [-100, 100])
    plt.title('rf Model: Training Predictions vs Actual')
    plt.savefig(plot_dir / 'rf_train_predictions.png')
    plt.close()

    # Plot error distribution for test data
    error = y_pred - y_test_actual
    plt.figure(figsize=(10, 7))
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error [Sap Velocity]')
    plt.ylabel('Count')
    plt.title('rf Model: Test Error Distribution')
    plt.savefig(plot_dir / 'rf_test_error_distribution.png')
    plt.close()

    # Plot error distribution for training data
    error_train = y_train_pred - y_train_actual
    plt.figure(figsize=(10, 7))
    plt.hist(error_train, bins=25)
    plt.xlabel('Prediction Error [Sap Velocity]')
    plt.ylabel('Count')
    plt.title('rf Model: Training Error Distribution')
    plt.savefig(plot_dir / 'rf_train_error_distribution.png')
    plt.close()


if __name__ == "__main__":
    main()