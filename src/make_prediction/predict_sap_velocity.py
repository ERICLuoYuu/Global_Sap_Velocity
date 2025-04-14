#!/usr/bin/env python
"""
Simplified Sap Velocity Prediction Script

This script works directly with preprocessed ERA5-Land data,
organizes it into sequences, and generates predictions.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import logging
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import time
import warnings
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sap_velocity_prediction.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("sap_prediction")



# Default parameters if config is not available
DEFAULT_PARAMS = {
    'MODEL_TYPES': ['xgb', 'rf', 'cnn_lstm'],
    'INPUT_WIDTH': 8,
    'LABEL_WIDTH': 1,
    'SHIFT': 1
}

try:
    # Import configuration - adapt this to your config structure
    from config import (
        BASE_DIR, MODELS_DIR, OUTPUT_DIR, SCALER_DIR,
        MODEL_TYPES, BIOME_TYPES
    )
except ImportError:
    logger.warning("Config module not found. Using default parameters.")
    BASE_DIR = Path('.')
    MODELS_DIR = BASE_DIR / 'models'
    OUTPUT_DIR = BASE_DIR / 'outputs'
    SCALER_DIR = BASE_DIR / 'scalers'
    MODEL_TYPES = DEFAULT_PARAMS['MODEL_TYPES']
    BIOME_TYPES = []


def load_models(models_dir=None, model_types=None):
    """
    Load saved models for each model type.
    
    Parameters:
    -----------
    models_dir : str, optional
        Directory containing model files. If None, uses './outputs/models/'
    model_types : list, optional
        List of model types to load. If None, loads all available models.
    
    Returns:
    --------
    dict
        Dictionary mapping model types to loaded models
    """
    if models_dir is None:
        models_dir = './outputs/models'
    
    if model_types is None:
        model_types = ['ann', 'lstm', 'transformer', 'cnn_lstm', 'rf', 'svr', 'xgb']
    
    models_dir = Path(models_dir)
    loaded_models = {}
    
    for model_type in model_types:
        model_path = None
        
        # Define model directory
        type_dir = models_dir / (model_type + '_regression')
        
        # Check if directory exists
        if not type_dir.exists():
            print(f"Warning: Directory for {model_type} not found at {type_dir}")
            continue
        
        # Find the most recent model file
        if model_type in ['rf', 'svr', 'xgb']:
            # Traditional ML models use .joblib format
            model_files = list(type_dir.glob('*.joblib'))
            
            if model_files:
                # Sort by modification time (most recent first)
                model_path = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                
                # Load the model
                try:
                    print(f"Loading {model_type} model from {model_path}")
                    model = joblib.load(model_path)
                    loaded_models[model_type] = model
                except Exception as e:
                    print(f"Error loading {model_type} model: {e}")
            else:
                print(f"No .joblib files found for {model_type}")
                
        else:
            # Deep learning models use .keras format
            model_files = list(type_dir.glob('*.keras'))
            
            if not model_files:
                # Check for saved model directories (TensorFlow format)
                model_dirs = [d for d in type_dir.iterdir() if d.is_dir() and (d/'saved_model.pb').exists()]
                if model_dirs:
                    model_path = sorted(model_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                else:
                    print(f"No .keras files or SavedModel directories found for {model_type}")
                    continue
            else:
                # Use the most recent .keras file
                model_path = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            
            # Load the model
            try:
                print(f"Loading {model_type} model from {model_path}")
                model = tf.keras.models.load_model(model_path)
                loaded_models[model_type] = model
            except Exception as e:
                print(f"Error loading {model_type} model: {e}")
    
    print(f"Successfully loaded {len(loaded_models)} models")
    return loaded_models


def load_scalers(scaler_dir=SCALER_DIR):
    """
    Load feature and label scalers.
    
    Parameters:
    -----------
    scaler_dir : Path
        Directory containing scalers
        
    Returns:
    --------
    tuple
        (feature_scaler, label_scaler)
    """
    try:
        feature_scaler = joblib.load(scaler_dir / 'feature_scaler.pkl')
        label_scaler = joblib.load(scaler_dir / 'label_scaler.pkl')
        logger.info("Loaded scalers")
    except FileNotFoundError as e:
        logger.warning(f"Scaler not found: {e}")
        logger.warning("Creating new standard scalers (this may affect prediction accuracy)")
        from sklearn.preprocessing import StandardScaler
        feature_scaler = StandardScaler()
        label_scaler = StandardScaler()
    
    return feature_scaler, label_scaler


def load_preprocessed_data(data_file):
    """
    Load preprocessed ERA5-Land data from CSV.
    
    Parameters:
    -----------
    data_file : str or Path
        Path to preprocessed CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data
    """
    try:
        logger.info(f"Loading preprocessed data from {data_file}")
        
        # Attempt to find the timestamp column format
        with open(data_file, 'r') as f:
            header = f.readline().strip().split(',')
            first_row = f.readline().strip().split(',')
        
        # Check if the first column looks like a datetime
        timestamp_col = None
        datetime_format = None
        
        for i, (col_name, val) in enumerate(zip(header, first_row)):
            if any(time_word in col_name.lower() for time_word in ['time', 'date', 'timestamp']):
                timestamp_col = col_name
                break
            
            # Try to parse the first column as a datetime if no obvious datetime column
            if i == 0 and '/' in val and ':' in val:
                timestamp_col = col_name
                break
        
        # Load the data
        if timestamp_col:
            logger.info(f"Using {timestamp_col} as timestamp column")
            df = pd.read_csv(data_file, parse_dates=[timestamp_col])
            df.set_index(timestamp_col, inplace=True)
        else:
            # Try the first column as a fallback
            logger.warning("No timestamp column identified in header. Trying first column.")
            df = pd.read_csv(data_file)
            try:
                # Try to parse the first column as datetime
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                df.set_index(df.columns[0], inplace=True)
                logger.info(f"Using {df.index.name} as timestamp index")
            except:
                logger.warning("Could not parse first column as datetime. Using as-is.")
                df = pd.read_csv(data_file)
        
        logger.info(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    except Exception as e:
        logger.error(f"Error loading preprocessed data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def map_windowed_predictions_to_df(df, predictions, input_width=8, shift=1):
    """
    Maps predictions for future forecasting with proper window handling
    """
    # Initialize result Series
    result = pd.Series(np.nan, index=df.index)
    
    # Track current prediction index
    pred_idx = 0
    mapped_count = 0
    
    # Process each location independently
    if 'name' in df.columns:
        locations = sorted(df['name'].unique())
        
        for location in locations:
            # Get indices for this location
            loc_mask = df['name'] == location
            loc_indices = np.where(loc_mask)[0]
            
            # Sort indices by timestamp if needed
            if isinstance(df.index, pd.DatetimeIndex):
                timestamps = df.index[loc_indices]
                sorted_pos = np.argsort(timestamps)
                loc_indices = loc_indices[sorted_pos]
            
            # Skip if insufficient data
            if len(loc_indices) < input_width:
                logger.debug(f"Skipping location {location}: not enough data points")
                continue
                
            # Calculate windows for prediction mode
            num_windows = len(loc_indices) - input_width + 1
            
            # Map predictions for each window to future timestamps
            for i in range(num_windows):
                if pred_idx >= len(predictions):
                    break
                    
                # Get target position (window end + shift)
                target_pos = i + input_width - 1 + shift
                
                # Map to actual row if target position exists in data
                if target_pos < len(loc_indices):
                    row_idx = loc_indices[target_pos]
                    result.iloc[row_idx] = predictions[pred_idx]
                    mapped_count += 1
                # For future predictions beyond data, we can't map them
                
                pred_idx += 1
    
    logger.info(f"Mapped {mapped_count} out of {len(predictions)} predictions")
    return result


def create_prediction_windows(df, feature_columns, input_width=8, batch_size=32):
    """
    Create time series windows for prediction with consistent location ordering.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with time series data
    feature_columns : list
        List of feature column names to include
    input_width : int
        Window size for input sequence
    batch_size : int
        Batch size for TensorFlow dataset
        
    Returns:
    --------
    tf.data.Dataset
        Dataset with windowed features ready for prediction
    """
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    
    # Create windows
    windows = []
    
    # Create location groups with consistent ordering
    location_groups = []
    
    if 'name' in df.columns:
        # Group by name and sort by name for consistency
        for name in sorted(df['name'].unique()):
            group = df[df['name'] == name]
            # Sort by time index
            if isinstance(group.index, pd.DatetimeIndex):
                group = group.sort_index()
            location_groups.append((name, group))
    else:
        # Group by lat/lon and sort by lat/lon for consistency
        latlons = []
        for lat in sorted(df['latitude'].unique()):
            for lon in sorted(df['longitude'].unique()):
                latlons.append((lat, lon))
                
        for latlon in latlons:
            lat, lon = latlon
            group = df[(df['latitude'] == lat) & (df['longitude'] == lon)]
            # Sort by time index
            if isinstance(group.index, pd.DatetimeIndex):
                group = group.sort_index()
            location_groups.append((latlon, group))
    
    logger.info(f"Processing {len(location_groups)} locations for windowing")
    
    # Process each location in consistent order
    for location, group in location_groups:
        # Get feature data for this location
        location_data = group[feature_columns].values
        
        # Skip if not enough data points
        if len(location_data) < input_width:
            logger.warning(f"Skipping location {location} - not enough data points (<{input_width})")
            continue
            
        # Create sliding windows
        num_windows = len(location_data) - input_width + 1
        logger.info(f"Creating {num_windows} windows for location {location}")
        
        for i in range(num_windows):
            # Get window
            window = location_data[i:i+input_width]
            windows.append(window)
    
    # Convert to numpy array
    windows = np.array(windows)
    
    logger.info(f"Created {len(windows)} windows of shape {windows.shape}")
    
    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(windows)
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    return dataset


def prepare_features_from_preprocessed(df, feature_scaler=None, 
                                     input_width=8, label_width=1, shift=1):
    """
    Prepare features from preprocessed data with a specific feature order.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed data
    feature_scaler : sklearn.preprocessing.StandardScaler, optional
        Scaler for features
    for_cnn_lstm : bool
        Whether to prepare features for sequence models (CNN-LSTM, LSTM, etc.)
    input_width : int
        Number of time steps to use as input for sequence models
    label_width : int
        Number of time steps to predict for sequence models
    shift : int
        Number of time steps to shift predictions for sequence models
        
    Returns:
    --------
    tuple
        (df_features, feature_columns) if for_cnn_lstm=False
        (windowed_dataset, feature_columns) if for_cnn_lstm=True
    """
    try:
        logger.info("Preparing features from preprocessed data...")
        
        # Create a copy of the DataFrame to avoid modifying the original
        
        # covert 'Day sin', 'Week sin', 'Month sin', 'Year sin' to lowercase
        # Convert the specific cyclical time features to lowercase
        time_features = ['Day sin', 'Week sin', 'Month sin', 'Year sin']
        lowercase_mapping = {}
        
        for col in time_features:
            if col in df.columns:
                lowercase_col = col.lower()
                # Create mapping from capitalized to lowercase
                lowercase_mapping[col] = lowercase_col
                # Rename the column in the DataFrame
                df.rename(columns={col: lowercase_col}, inplace=True)
        
        df_features = df.copy()
        # Define the exact order of features as specified
        ordered_features = [
            'ext_rad', 'sw_in', 'ta', 'ws', 'vpd', 'ppfd_in', 
            'day sin', 'week sin', 'month sin', 'year sin',
            'Boreal forest', 'Subtropical desert', 'Temperate forest', 
            'Temperate grassland desert', 'Temperate rain forest', 
            'Tropical forest savanna', 'Tropical rain forest', 
            'Tundra', 'Woodland/Shrubland'
        ]
        biomes = ['Boreal forest', 'Subtropical desert', 'Temperate forest', 
            'Temperate grassland desert', 'Temperate rain forest', 
            'Tropical forest savanna', 'Tropical rain forest', 
            'Tundra', 'Woodland/Shrubland']
        # Filter to only include features that are present in the dataframe
        feature_columns = [col for col in ordered_features if col in df.columns]
        scale_columns = [col for col in feature_columns if col not in biomes]
        # Check if any specified features are missing
        missing_features = [col for col in ordered_features if col not in df.columns]
        if missing_features:
            logger.warning(f"Some specified features are missing from the dataset: {missing_features}")
        
        # Check if we have any features at all
        if not feature_columns:
            logger.error("None of the specified features are present in the dataset")
            # Fall back to automatic feature detection
            metadata_cols = ['name', 'latitude', 'longitude', 'TIMESTAMP', 'timestamp', 'datetime']
            feature_columns = [col for col in df.columns if col not in metadata_cols and not col.startswith('sap_velocity')]
            
            
            logger.info(f"Falling back to automatically detected features: {feature_columns}")
        
        logger.info(f"Using {len(feature_columns)} features in specified order")
        
        # Apply feature scaling if needed
        if feature_scaler is not None:
            # Scale features
            try:
                # Check if feature_scaler has been fit
                if hasattr(feature_scaler, 'n_features_in_'):
                    # print all fit features
                    logger.info(f"Feature scaler fitted with {feature_scaler.n_features_in_} features")
                    print(feature_columns)
                    # Scale features
                    df_features[scale_columns] = feature_scaler.transform(df_features[scale_columns])

                else:
                    logger.warning("Feature scaler not fitted. Using raw features.")
            except Exception as e:
                logger.warning(f"Error applying feature scaling: {e}")
                logger.warning("Using raw features.")
        

        logger.info(f"Creating prediction windows for sequence models with input_width={input_width}")
        
        try:
            # Create windows directly without segmentation or train/val/test splitting
            windowed_dataset = create_prediction_windows(
                df_features,
                feature_columns,
                input_width=input_width,
                batch_size=32
            )
            
            logger.info("Successfully created prediction windows for sequence models")
            return windowed_dataset, df_features, feature_columns
        except Exception as e:
            logger.error(f"Error creating prediction windows: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.warning("Falling back to standard features.")
            return df_features, feature_columns
    
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return df, []


def make_predictions(df, feature_columns, models, label_scaler, windowed_dataset=None, input_width=8, shift=1):
    """
    Make predictions using multiple model types with proper handling for each.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with features
    feature_columns : list
        List of feature column names
    models : dict
        Dictionary of loaded models
    label_scaler : sklearn.preprocessing.StandardScaler
        Scaler for labels
    windowed_dataset : tf.data.Dataset, optional
        Windowed dataset for sequence models
    input_width : int
        Width of input window used for sequence models
    shift : int
        Prediction shift used for sequence models (usually 1)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with predictions from all models
    """
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from collections import defaultdict
    
    logger.info("Making predictions with multiple model types...")
    
    # Create copy of DataFrame for results
    df_result = df.copy()

    
    # Categorize models by type
    deep_sequence_models = {}  # Models that take sequence input (windows) directly
    traditional_ml_models = {}  # Models that need flattened inputs
    
    for model_type, model in models.items():
        if model_type in ['cnn_lstm_config']:
            continue  # Skip config entries
            
        if model_type.lower() in ['cnn_lstm', 'lstm', 'transformer', 'ann']:
            deep_sequence_models[model_type] = model
            logger.info(f"Categorized {model_type} as deep sequence model")
        elif model_type.lower() in ['svr', 'rf', 'xgb', 'xgboost', 'random_forest', 'svm']:
            traditional_ml_models[model_type] = model
            logger.info(f"Categorized {model_type} as traditional ML model")
        else:
            # For any uncategorized model, treat as traditional for safety
            traditional_ml_models[model_type] = model
            logger.info(f"Categorized {model_type} as traditional ML model (default)")
    
    # Part 1: Make predictions with traditional ML models using flattened windows
    if traditional_ml_models and windowed_dataset is not None:
        logger.info("Preparing flattened windows for traditional ML models...")
        
        # Extract and flatten window features
        flattened_windows = []
        
        # Capture windows in a list
        for window_batch in windowed_dataset:
            if isinstance(window_batch, tuple):
                windows = window_batch[0].numpy()  # Extract features only
            else:
                windows = window_batch.numpy()
                
            for window in windows:
                # Flatten window: from (time_steps, features) to (time_steps * features)
                flattened = window.flatten()
                flattened_windows.append(flattened)
        
        # Convert to numpy array
        X_flattened = np.array(flattened_windows)
        
        if len(X_flattened) == 0:
            logger.warning("No windows extracted from dataset for traditional ML models")
        else:
            logger.info(f"Created {len(X_flattened)} flattened windows of shape {X_flattened.shape}")
            
            # Make predictions with each traditional model
            for model_type, model in traditional_ml_models.items():
                try:
                    # Get predictions
                    preds_scaled = model.predict(X_flattened)
                    preds = label_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
                    
                    # Map predictions back to original data
                    # This is challenging because windowing creates overlapping sequences
                    # We'll map predictions to rows based on the position in the original data
                    
                    # Map windowes predictions to original dataframe
                    prediction_map = map_windowed_predictions_to_df(
                        df_result, 
                        preds, 
                        input_width=input_width, 
                        shift=shift
                    )
                    
                    # Add to results
                    df_result[f'sap_velocity_{model_type}'] = prediction_map
                    logger.info(f"Made predictions with {model_type} model using flattened windows")
                    
                except Exception as e:
                    logger.error(f"Error making predictions with {model_type} model: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
    elif traditional_ml_models:
        logger.info("Making predictions with traditional ML models using raw features...")
        
        # Fallback to standard feature vectors without windowing
        X = df[feature_columns].values
        
        # Make predictions with each traditional model
        for model_type, model in traditional_ml_models.items():
            try:
                preds_scaled = model.predict(X)
                preds = label_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
                df_result[f'sap_velocity_{model_type}'] = preds
                logger.info(f"Made predictions with {model_type} model using raw features")
            except Exception as e:
                logger.error(f"Error making predictions with {model_type} model: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # Part 2: Make predictions with deep sequence models (CNN-LSTM, LSTM, Transformer, ANN)
    if deep_sequence_models and windowed_dataset is not None:
        logger.info("Making predictions with deep sequence models...")
        
        for model_type, model in deep_sequence_models.items():
            try:
                # Helper function to extract predictions from windowed dataset
                def get_sequence_model_predictions(model, dataset, scaler):
                    """Extract predictions from a sequence model using windowed data."""
                    all_predictions = []
                    window_positions = []
                    window_count = 0
                    
                    # Process each batch of windows
                    for features_batch in dataset:
                        # Handle both (features, labels) and just features
                        if isinstance(features_batch, tuple):
                            features = features_batch[0]
                        else:
                            features = features_batch
                        
                        # Get predictions for this batch
                        batch_predictions = model.predict(features, verbose=0)
                        
                        # Reshape if needed (depending on model output shape)
                        if len(batch_predictions.shape) == 3:  # [batch, time_steps, features]
                            batch_predictions = batch_predictions[:, -1, :]  # Get last time step
                        
                        # Add to results
                        all_predictions.extend(batch_predictions.flatten())
                        
                        # Track batch size for window position calculation
                        batch_size = features.shape[0]
                        for i in range(batch_size):
                            window_positions.append(window_count)
                            window_count += 1
                    
                    # Convert to numpy arrays
                    all_predictions = np.array(all_predictions).reshape(-1, 1)
                    
                    # Inverse transform predictions
                    if scaler is not None:
                        all_predictions = scaler.inverse_transform(all_predictions).flatten()
                    else:
                        all_predictions = all_predictions.flatten()
                    
                    return all_predictions, np.array(window_positions)
                
                # Get predictions
                model_preds, window_positions = get_sequence_model_predictions(
                    model, windowed_dataset, label_scaler
                )
                
                # Map predictions back to original data
                prediction_map = map_windowed_predictions_to_df(
                    df_result, 
                    model_preds, 
                    input_width=input_width, 
                    shift=shift
                )
                
                # Add to results
                df_result[f'sap_velocity_{model_type}'] = prediction_map
                logger.info(f"Made predictions with {model_type} model")
                sequence_predictions_made = True
                
            except Exception as e:
                logger.error(f"Error making predictions with {model_type} model: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # Create ensemble prediction from all available models
    pred_cols = [col for col in df_result.columns if col.startswith('sap_velocity_') and col != 'sap_velocity_ensemble']
    
    if pred_cols:
        # Create ensemble by taking median of all predictions
        df_result['sap_velocity_ensemble'] = df_result[pred_cols].median(axis=1)
        logger.info(f"Created ensemble prediction from {len(pred_cols)} models: {', '.join(pred_cols)}")
    else:
        logger.warning("No predictions generated, cannot create ensemble")
    
    return df_result


def visualize_predictions(df_predictions, output_dir, include_feature_importance=True, models=None):
    """
    Create visualizations of the predictions.
    
    Parameters:
    -----------
    df_predictions : pandas.DataFrame
        DataFrame with predictions
    output_dir : Path
        Directory to save visualizations
    include_feature_importance : bool
        Whether to include feature importance plots
    models : dict, optional
        Dictionary of loaded models for feature importance
    """
    logger.info("Creating visualizations...")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get different types of columns
    metadata_cols = ['name', 'latitude', 'longitude']
    pred_cols = [col for col in df_predictions.columns if col.startswith('sap_velocity_')]
    
    # Remaining columns are feature columns
    feature_cols = [col for col in df_predictions.columns 
                    if col not in metadata_cols and col not in pred_cols]
    
    # Reset index if it's a DatetimeIndex to get a datetime column
    if isinstance(df_predictions.index, pd.DatetimeIndex):
        df_predictions = df_predictions.reset_index()
        datetime_col = df_predictions.columns[0]  # First column is the former index
    else:
        # Find a datetime column
        datetime_col = None
        for col in df_predictions.columns:
            if col.lower() in ['time', 'timestamp', 'datetime', 'date']:
                datetime_col = col
                break
                
        if datetime_col is None:
            logger.warning("No datetime column found. Using index for time series plots.")
            df_predictions['plot_index'] = df_predictions.index
            datetime_col = 'plot_index'
    
    # Get unique locations
    if 'name' in df_predictions.columns:
        # Use name column for locations
        location_col = 'name'
        unique_locations = df_predictions['name'].unique()
    else:
        # Use latitude/longitude for locations
        location_col = None
        unique_lats = df_predictions['latitude'].unique()
        unique_lons = df_predictions['longitude'].unique()
        
        # Create list of (lat, lon) tuples
        unique_locations = [(lat, lon) for lat in unique_lats for lon in unique_lons]
    
    logger.info(f"Creating visualizations for {len(unique_locations)} locations")
    
    # Limit to a maximum number of plots to avoid creating too many files
    max_plots = 10
    if len(unique_locations) > max_plots:
        logger.info(f"Too many locations ({len(unique_locations)}), limiting to {max_plots} plots")
        unique_locations = unique_locations[:max_plots]
    
    # Plot time series for each location
    for location in unique_locations:
        # Filter data for this location
        if location_col:
            # Filter by name
            point_data = df_predictions[df_predictions[location_col] == location]
        else:
            # Filter by lat/lon tuple
            lat, lon = location
            point_data = df_predictions[
                (df_predictions['latitude'] == lat) & 
                (df_predictions['longitude'] == lon)
            ]
        
        if point_data.empty:
            continue
        
        # Create figure with 2 subplots (prediction and features)
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Sort by datetime
        point_data = point_data.sort_values(datetime_col)
        
        # Plot ensemble prediction
        if 'sap_velocity_ensemble' in point_data.columns:
            axes[0].plot(
                point_data[datetime_col], 
                point_data['sap_velocity_ensemble'], 
                label='Ensemble', 
                linewidth=2, 
                color='black'
            )
        
        # Plot individual model predictions
        other_pred_cols = [col for col in pred_cols if col != 'sap_velocity_ensemble']
        colors = plt.cm.tab10(np.linspace(0, 1, len(other_pred_cols)))
        
        for i, col in enumerate(other_pred_cols):
            model_type = col.replace('sap_velocity_', '')
            axes[0].plot(
                point_data[datetime_col], 
                point_data[col], 
                label=f'{model_type.upper()}', 
                alpha=0.7,
                color=colors[i]
            )
        
        # Set title and labels
        location_str = location if location_col else f"Lat={location[0]:.2f}, Lon={location[1]:.2f}"
        axes[0].set_title(f'Sap Velocity Predictions for {location_str}')
        axes[0].set_ylabel('Sap Velocity')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot key features
        # Select top climate features and time features
        climate_features = ['ta', 'vpd', 'ppfd_in', 'ws', 'sw_in']
        selected_features = [col for col in climate_features if col in feature_cols][:4]  # Limit to 4
        
        # Add a few time features if available
        time_features = [col for col in feature_cols if 'sin' in col or 'cos' in col]
        if time_features:
            selected_time_features = [col for col in time_features if 'sin' in col][:2]  # Limit to 2 sin features
            selected_features.extend(selected_time_features)
        
        # Limit to 6 features total to avoid overcrowding
        selected_features = selected_features[:6]
        
        if selected_features:
            # Normalize features for comparison
            normed_features = point_data[selected_features].copy()
            for col in selected_features:
                # Min-max scaling
                min_val = normed_features[col].min()
                max_val = normed_features[col].max()
                if max_val > min_val:
                    normed_features[col] = (normed_features[col] - min_val) / (max_val - min_val)
                else:
                    normed_features[col] = 0
            
            # Plot normalized features
            for col in selected_features:
                axes[1].plot(
                    point_data[datetime_col], 
                    normed_features[col], 
                    label=col
                )
            
            axes[1].set_title('Normalized Features')
            axes[1].set_xlabel('Time')
            axes[1].set_ylabel('Normalized Value')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].set_visible(False)
        
        plt.tight_layout()
        
        # Format filename to avoid special characters
        if location_col:
            # Use name for filename
            safe_name = str(location).replace(' ', '_').replace('/', '_').replace('\\', '_')
            filename = f'sap_velocity_{safe_name}.png'
        else:
            # Use lat/lon for filename
            lat, lon = location
            filename = f'sap_velocity_lat{lat:.2f}_lon{lon:.2f}.png'.replace('-', 'neg')
        
        plt.savefig(output_dir / filename)
        plt.close()
    
    # Create feature importance plots if requested
    if include_feature_importance and models is not None:
        try:
            # Plot feature importance for models that support it
            for model_type, model in models.items():
                if model_type == 'cnn_lstm' or model_type == 'cnn_lstm_config':
                    continue
                
                # Check if model has feature_importances_ attribute
                if hasattr(model, 'feature_importances_'):
                    # Create feature importance DataFrame
                    importance_df = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    # Plot feature importance
                    plt.figure(figsize=(10, 8))
                    plt.barh(
                        importance_df['Feature'],
                        importance_df['Importance'],
                        color='skyblue'
                    )
                    plt.xlabel('Importance')
                    plt.title(f'Feature Importance - {model_type.upper()}')
                    plt.gca().invert_yaxis()  # Display highest importance at top
                    plt.tight_layout()
                    
                    plt.savefig(output_dir / f'feature_importance_{model_type}.png')
                    plt.close()
                    
                    # Save feature importance to CSV
                    importance_df.to_csv(output_dir / f'feature_importance_{model_type}.csv', index=False)
        except Exception as e:
            logger.error(f"Error creating feature importance plots: {e}")
    
    logger.info(f"Visualizations saved to {output_dir}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Predict sap velocity from preprocessed data')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file with preprocessed data')
    parser.add_argument('--output', type=str, default=None, help='Output directory for predictions')
    parser.add_argument('--models-dir', type=str, default='./outputs/models', help='Directory containing trained models')
    parser.add_argument('--scaler-dir', type=str, default='./outputs/scalers', help='Directory containing scalers')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations of predictions')
    parser.add_argument('--model-types', type=list, default=['cnn_lstm'], help='Model types to use')
    parser.add_argument('--feature-importance', action='store_true', help='Include feature importance plots')
    parser.add_argument('--input_width', type=int, default=DEFAULT_PARAMS['INPUT_WIDTH'], 
                      help='Number of time steps for CNN-LSTM input window')
    parser.add_argument('--label_width', type=int, default=DEFAULT_PARAMS['LABEL_WIDTH'], 
                      help='Number of time steps for CNN-LSTM label window')
    parser.add_argument('--shift', type=int, default=DEFAULT_PARAMS['SHIFT'], 
                      help='Number of time steps to shift for CNN-LSTM')
    return parser.parse_args()


def main():
    """Main function"""
    start_time = time.time()
    
    # Parse arguments
    args = parse_args()
    
    # Set directories
    models_dir = Path(args.models_dir) if args.models_dir else MODELS_DIR
    scaler_dir = Path(args.scaler_dir) if args.scaler_dir else SCALER_DIR
    input_file = Path(args.input)
    
    # Set output directory based on input file name if not specified
    if args.output:
        output_dir = Path(args.output)
    else:
        stem = input_file.stem
        output_dir = OUTPUT_DIR / stem
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Log basic information
    logger.info(f"Starting prediction using preprocessed data: {input_file}")
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Output directory: {output_dir}")
    input_width = args.input_width
    label_width = args.label_width
    shift = args.shift
    logger.info(f"Input width: {input_width}, Label width: {label_width}, Shift: {shift}")
    # Load models
    print(f"model_types: {args.model_types}")
    models = load_models(models_dir, args.model_types)
    if not models:
        logger.error("No models found. Please check the model directory.")
        sys.exit(1)
    
    # Load scalers
    feature_scaler, label_scaler = load_scalers(scaler_dir)
    
    # Load preprocessed data
    df = load_preprocessed_data(input_file)
    if df is None or len(df) == 0:
        logger.error("Failed to load preprocessed data.")
        sys.exit(1)
    
    # Prepare features
    windowed_dataset = None
    

    # Prepare features for CNN-LSTM
    windowed_dataset, df_features, feature_columns = prepare_features_from_preprocessed(
        df, 
        feature_scaler, 
        input_width=input_width,
        label_width=label_width,
        shift=shift
    )
    
    # Make predictions
    df_predictions = make_predictions(
        df_features, 
        feature_columns, 
        models, 
        label_scaler,
        windowed_dataset
    )
    # Save predictions
    output_path = output_dir / 'sap_velocity_predictions.csv'
    df_predictions.to_csv(output_path, index=True)
    logger.info(f"Predictions saved to {output_path}")
    
    # Create visualizations if requested
    if args.visualize:
        visualize_predictions(df_predictions, output_dir, args.feature_importance, models)
    
    # Calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time
    
    logger.info(f"Done! Execution time: {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)