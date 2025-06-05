#!/usr/bin/env python
"""
Improved Sap Velocity Prediction Script

This script works with preprocessed ERA5-Land data to predict sap velocity,
using an improved index mapping approach for more reliable time series predictions.
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
import multiprocessing
from functools import partial
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
        MODEL_TYPES
    )
except ImportError:
    logger.warning("Config module not found. Using default parameters.")
    BASE_DIR = Path('.')
    MODELS_DIR = BASE_DIR / 'models'
    OUTPUT_DIR = BASE_DIR / 'outputs'
    SCALER_DIR = BASE_DIR / 'scalers'
    MODEL_TYPES = DEFAULT_PARAMS['MODEL_TYPES']






# Add these imports at the top of the script alongside the other imports

def process_location_chunk(location_data, models, feature_columns, label_scaler, 
                          input_width=8, shift=1, location_id=None):
    """
    Process a single location chunk for prediction.
    
    Parameters:
    -----------
    location_data : pandas.DataFrame
        DataFrame containing data for a single location
    models : dict
        Dictionary of loaded models {model_type: model_object}
    feature_columns : list
        List of feature column names
    label_scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler for the target variable
    input_width : int
        Width of the input window
    shift : int
        Prediction shift
    location_id : str or tuple
        Identifier for the location (for logging)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added prediction columns for this location
    """
    # Create a logger for this process
    process_logger = logging.getLogger(f"sap_prediction.worker.{multiprocessing.current_process().name}")
    process_logger.info(f"Processing location {location_id} with {len(location_data)} rows")
    
    # Skip if not enough data
    if len(location_data) < input_width:
        process_logger.warning(f"Location {location_id} has insufficient data points ({len(location_data)} < {input_width})")
        return location_data
    
    # Create windows for this location only
    try:
        windows_dataset, window_metadata, _, total_windows = create_prediction_windows_improved(
            location_data, feature_columns, input_width=input_width, shift=shift
        )
        
        if windows_dataset is None or not window_metadata:
            process_logger.warning(f"Failed to create prediction windows for location {location_id}")
            return location_data
            
        process_logger.info(f"Created {total_windows} windows for location {location_id}")
        
        # Store results
        results_df = location_data.copy()
        
        # Make predictions with each model
        for model_type, model in models.items():
            process_logger.info(f"Making predictions with {model_type} model for location {location_id}")
            
            try:
                # Handle different model types
                is_deep_model = model_type.lower() in ['cnn_lstm', 'lstm', 'transformer', 'ann', 'gru']
                
                if is_deep_model and isinstance(model, tf.keras.Model):
                    # For sequence models, use the windowed dataset directly
                    preds_scaled = model.predict(windows_dataset, verbose=0)
                elif hasattr(model, 'predict'):
                    # For traditional ML models, flatten the windows
                    windows_np = np.concatenate([batch.numpy() for batch in windows_dataset], axis=0)
                    X_flattened = windows_np.reshape(windows_np.shape[0], -1)  # Flatten each window
                    preds_scaled = model.predict(X_flattened)
                else:
                    process_logger.error(f"Model {model_type} lacks a predict method. Skipping.")
                    continue
                
                # Standardize prediction shape
                if len(preds_scaled.shape) == 3:  # Sequence output (batch, timesteps, features)
                    preds_scaled = preds_scaled[:, -1, :].reshape(-1, 1)  # Take the last timestep
                elif len(preds_scaled.shape) == 2 and preds_scaled.shape[1] > 1:
                    preds_scaled = preds_scaled[:, 0].reshape(-1, 1)  # Take first feature
                elif len(preds_scaled.shape) == 1:
                    preds_scaled = preds_scaled.reshape(-1, 1)  # Ensure 2D for inverse transform
                
                # Inverse transform to get actual sap velocity values
                preds_unscaled = label_scaler.inverse_transform(preds_scaled).flatten()
                
                # Map predictions back to DataFrame with explicit metadata
                mapped_results = map_predictions_to_df_improved(
                    location_data, preds_unscaled, window_metadata, model_type
                )
                
                # Add mapped predictions to results DataFrame
                results_df[f'sap_velocity_{model_type}'] = mapped_results[f'sap_velocity_{model_type}']
                
                # Log prediction statistics
                pred_col = f'sap_velocity_{model_type}'
                non_nan_count = results_df[pred_col].notna().sum()
                process_logger.info(f"Added {non_nan_count} predictions for {location_id} with {model_type}")
                
            except Exception as e:
                process_logger.error(f"Error making predictions with {model_type} for location {location_id}: {e}")
                import traceback
                process_logger.error(traceback.format_exc())
        
        # Create ensemble prediction if multiple models succeeded for this location
        pred_cols = [col for col in results_df.columns if col.startswith('sap_velocity_')]
        if len(pred_cols) > 1:
            process_logger.info(f"Creating ensemble prediction for location {location_id}")
            results_df['sap_velocity_ensemble'] = results_df[pred_cols].median(axis=1)
        elif len(pred_cols) == 1:
            process_logger.info(f"Only one model available for location {location_id}. Using it as the ensemble.")
            results_df['sap_velocity_ensemble'] = results_df[pred_cols[0]]
        
        return results_df
        
    except Exception as e:
        process_logger.error(f"Error processing location {location_id}: {e}")
        import traceback
        process_logger.error(traceback.format_exc())
        return location_data


# Inside the make_predictions_parallel function, replace the problematic part with this:

def make_predictions_parallel(df, models, feature_columns, label_scaler, input_width=8, shift=1, n_processes=None):
    """
    Make predictions in parallel across different locations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the features
    models : dict
        Dictionary of loaded models {model_type: model_object}
    feature_columns : list
        List of feature column names
    label_scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler for the target variable
    input_width : int
        Width of the input window
    shift : int
        Prediction shift
    n_processes : int, optional
        Number of processes to use. If None, uses available CPU cores.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added prediction columns
    """
    logger.info("Making predictions with parallel processing across locations...")
    
    # Validate inputs
    if df is None or df.empty:
        logger.error("Input DataFrame is empty or None")
        return pd.DataFrame()
    
    if not models:
        logger.error("No models provided")
        return df
    
    if not feature_columns:
        logger.error("No feature columns specified")
        return df
    
    if label_scaler is None:
        logger.error("Label scaler is required for predictions")
        return df
    
    # Set number of processes
    if n_processes is None:
        n_processes = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
    
    logger.info(f"Using {n_processes} processes for parallel prediction")
    
    # Identify locations for grouping
    location_groups = []
    location_id_col = None
    
    if 'name' in df.columns:
        logger.info("Grouping data by 'name' column.")
        location_ids = sorted(df['name'].unique())
        location_groups = [(name, df[df['name'] == name].copy()) for name in location_ids]
        location_id_col = 'name'
    elif 'latitude' in df.columns and 'longitude' in df.columns:
        logger.info("Grouping data by 'latitude' and 'longitude' columns.")
        latlon_pairs = sorted(df[['latitude', 'longitude']].drop_duplicates().values.tolist())
        location_groups = [((lat, lon), df[(df['latitude'] == lat) & (df['longitude'] == lon)].copy()) 
                          for lat, lon in latlon_pairs]
        location_id_col = ('latitude', 'longitude')
    else:
        logger.warning("No location columns found. Treating data as a single sequence.")
        # If only a single sequence, no need for multiprocessing
        logger.info("Only one location found, reverting to single-process prediction.")
        return make_predictions_improved(df, models, feature_columns, label_scaler, 
                                       input_width=input_width, shift=shift)
    
    logger.info(f"Found {len(location_groups)} locations for parallel processing")
    
    # If very few locations, may not be worth the overhead of multiprocessing
    if len(location_groups) < 3:
        logger.info(f"Only {len(location_groups)} locations found, reverting to single-process prediction.")
        return make_predictions_improved(df, models, feature_columns, label_scaler, 
                                      input_width=input_width, shift=shift)
    
    # Prepare parameter tuples for starmap - THIS IS THE FIXED PART
    param_tuples = [
        (location_df, models, feature_columns, label_scaler, input_width, shift, location_id)
        for location_id, location_df in location_groups
    ]
    
    # Use multiprocessing to process chunks in parallel
    try:
        # Initialize multiprocessing with a context manager
        ctx = multiprocessing.get_context('spawn')  # Use 'spawn' for better TensorFlow compatibility
        
        with ctx.Pool(processes=n_processes) as pool:
            # Apply the function to each chunk and collect results
            # We directly use process_location_chunk without partial
            results = pool.starmap(process_location_chunk, param_tuples)
            
        # Combine results from all processes
        result_df = pd.concat(results, axis=0)
        
        # Create ensemble prediction across all locations if multiple models succeeded
        pred_cols = [col for col in result_df.columns if col.startswith('sap_velocity_') 
                    and not col.endswith('_ensemble')]
        
        if len(pred_cols) > 1:
            logger.info(f"Creating global ensemble prediction from {len(pred_cols)} models")
            result_df['sap_velocity_ensemble'] = result_df[pred_cols].median(axis=1)
            ensemble_count = result_df['sap_velocity_ensemble'].notna().sum()
            logger.info(f"Ensemble has {ensemble_count} predictions ({ensemble_count/len(result_df)*100:.1f}% coverage)")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in parallel processing: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Fallback to non-parallel version
        logger.warning("Falling back to single-process prediction due to error in parallel processing")
        return make_predictions_improved(df, models, feature_columns, label_scaler, 
                                      input_width=input_width, shift=shift)



def load_models(models_dir=None, model_types=None):
    """
    Load saved models for each model type.

    Parameters:
    -----------
    models_dir : str or Path, optional
        Directory containing model files. If None, uses './outputs/models/'
    model_types : list, optional
        List of model types to load. If None, loads all available models.

    Returns:
    --------
    dict
        Dictionary mapping model types to loaded models
    """
    if models_dir is None:
        models_dir = Path('./outputs/models')
    else:
        models_dir = Path(models_dir)  # Ensure it's a Path object

    if model_types is None:
        # Default list if none provided
        model_types = ['ann', 'lstm', 'transformer', 'cnn_lstm', 'rf', 'svr', 'xgb']
        logger.info(f"No model types specified, attempting to load default types: {model_types}")

    loaded_models = {}

    for model_type in model_types:
        model_path = None
        # Define model directory based on convention
        type_dir = models_dir / (model_type + '_regression')

        # Check if directory exists
        if not type_dir.exists():
            logger.warning(f"Directory for {model_type} not found at {type_dir}")
            continue

        # Find the most recent model file based on type
        if model_type in ['rf', 'svr', 'xgb']:
            # Traditional ML models use .joblib format
            model_files = list(type_dir.glob('*.joblib'))
            if model_files:
                # Sort by modification time (most recent first)
                model_path = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                # Load the model
                try:
                    logger.info(f"Loading {model_type} model from {model_path}")
                    model = joblib.load(model_path)
                    loaded_models[model_type] = model
                except Exception as e:
                    logger.error(f"Error loading {model_type} model from {model_path}: {e}")
            else:
                logger.warning(f"No .joblib files found for {model_type} in {type_dir}")

        else:
            # Deep learning models
            model_files = list(type_dir.glob('*.keras'))
            if model_files:
                # Use the most recent .keras file
                model_path = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            else:
                # Check for saved model directories
                model_dirs = [d for d in type_dir.iterdir() if d.is_dir() and (d / 'saved_model.pb').exists()]
                if model_dirs:
                    model_path = sorted(model_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                else:
                    logger.warning(f"No model files found for {model_type} in {type_dir}")
                    continue

            # Load the model
            if model_path:
                try:
                    logger.info(f"Loading {model_type} model from {model_path}")
                    model = tf.keras.models.load_model(model_path)
                    loaded_models[model_type] = model
                except Exception as e:
                    logger.error(f"Error loading {model_type} model from {model_path}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

    if not loaded_models:
        logger.error(f"Failed to load any models from the specified types: {model_types}")
    else:
        logger.info(f"Successfully loaded {len(loaded_models)} models: {list(loaded_models.keys())}")

    return loaded_models


def load_scalers(scaler_dir=None):
    """
    Load feature and label scalers.

    Parameters:
    -----------
    scaler_dir : str or Path, optional
        Directory containing scalers. If None, uses './outputs/scalers/'.

    Returns:
    --------
    tuple
        (feature_scaler, label_scaler) or (None, None) if not found
    """
    if scaler_dir is None:
        scaler_dir = Path('./outputs/scalers')
    else:
        scaler_dir = Path(scaler_dir)  # Ensure it's a Path object

    feature_scaler = None
    label_scaler = None
    try:
        feature_scaler_path = scaler_dir / 'feature_scaler.pkl'
        label_scaler_path = scaler_dir / 'label_scaler.pkl'

        if feature_scaler_path.exists():
            feature_scaler = joblib.load(feature_scaler_path)
            logger.info(f"Loaded feature scaler from {feature_scaler_path}")
        else:
            logger.warning(f"Feature scaler not found at {feature_scaler_path}")

        if label_scaler_path.exists():
            label_scaler = joblib.load(label_scaler_path)
            logger.info(f"Loaded label scaler from {label_scaler_path}")
        else:
            logger.warning(f"Label scaler not found at {label_scaler_path}")

    except FileNotFoundError as e:
        logger.error(f"Scaler file not found: {e}")
    except Exception as e:
        logger.error(f"Error loading scalers from {scaler_dir}: {e}")

    if feature_scaler is None or label_scaler is None:
        logger.warning("One or both scalers could not be loaded. Predictions might be inaccurate or fail.")

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
    pandas.DataFrame or None
        Loaded data, or None if loading fails
    """
    try:
        data_file_path = Path(data_file)
        logger.info(f"Loading preprocessed data from {data_file_path}")

        if not data_file_path.exists():
            logger.error(f"Input data file not found: {data_file_path}")
            return None
        
        # Read header to check for timestamp column
        try:
            with open(data_file_path, 'r') as f:
                header = f.readline().strip().split(',')
                # Find potential timestamp columns
                time_cols = [col for col in header if 'time' in col.lower() or 'date' in col.lower()]
                
                # Try to read first data row
                first_row_line = f.readline()
                if not first_row_line:
                    logger.error(f"Data file {data_file_path} appears to be empty or only has a header.")
                    return None
        except Exception as e:
            logger.error(f"Error reading file header: {e}")
            return None

        # Load the data using pandas
        df = pd.read_csv(data_file_path, low_memory=False).dropna()
        df = df.iloc[:, 1:]
        # Try to set index using timestamp columns if they exist
        # check if there are multiple timestamp columns
        # and set the second one as index
        timestamp_col = 'timestamp.1'
        # sort by timestamp if it exists
        df = df.sort_values(by=timestamp_col) if timestamp_col in df.columns else df

        logger.info(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
        
        # Basic data validation
        if df.empty:
            logger.error("Loaded DataFrame is empty.")
            return None
            
        # Check for important columns
        required_features = ['ta', 'vpd', 'sw_in', 'ppfd_in', 'ext_rad', 'ws', 'annual_mean_temperature', 'annual_precipitation']
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            logger.warning(f"Some important features are missing: {missing_features}")
        
        return df

    except Exception as e:
        logger.error(f"Error loading preprocessed data from {data_file}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def create_prediction_windows_improved(df, feature_columns, input_width=8, shift=1, batch_size=32):
    """
    Create time series windows for prediction with explicit metadata tracking.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with time series data
    feature_columns : list
        List of feature column names to include in the windows
    input_width : int
        Window size for input sequence
    shift : int
        Number of steps to shift for prediction (default: 1)
    batch_size : int
        Batch size for the TensorFlow dataset
        
    Returns:
    --------
    tuple: (tf.data.Dataset, list, dict, int)
        - Dataset with windowed features ready for prediction
        - List of metadata dictionaries for each window
        - Dictionary mapping location identifier to sorted indices
        - Total number of windows created
    """
    import tensorflow as tf
    
    logger.info(f"Creating prediction windows with input_width={input_width}, shift={shift}, batch_size={batch_size}")
    logger.info(f"Using features: {feature_columns}")
    
    # Ensure unique index for reliable mapping
    df_processed = df.copy()
    if not df_processed.index.is_unique:
        logger.warning("DataFrame index is not unique. Resetting index for reliable mapping.")
        df_processed = df_processed.reset_index(drop=True)
    
    # Store all windows with their metadata
    window_data = []
    window_metadata = []
    location_indices_map = {}
    
    # Identify time column for sorting
    time_col = None
    if 'timestamp.1' in df_processed.columns:
        time_col = 'timestamp.1'
        logger.info(f"Using 'timestamp.1' column for time reference.")
    elif 'timestamp' in df_processed.columns:
        time_col = 'timestamp'
        logger.info(f"Using 'timestamp' column for time reference.")
    else:
        # Try to find other time columns
        time_cols = [col for col in df_processed.columns 
                    if 'time' in col.lower() or 'date' in col.lower()]
        if time_cols:
            time_col = time_cols[0]
            logger.info(f"Using '{time_col}' column for time reference.")
        else:
            logger.warning("No time column found. Sorting will use index only.")
    
    # Identify locations for grouping
    if 'name' in df_processed.columns:
        logger.info("Grouping data by 'name' column.")
        location_groups = [(name, df_processed[df_processed['name'] == name].copy()) 
                           for name in sorted(df_processed['name'].unique())]
        location_id_col = 'name'
    elif 'latitude' in df_processed.columns and 'longitude' in df_processed.columns:
        logger.info("Grouping data by 'latitude' and 'longitude' columns.")
        latlon_pairs = sorted(df_processed[['latitude', 'longitude']].drop_duplicates().values.tolist())
        location_groups = [((lat, lon), df_processed[(df_processed['latitude'] == lat) & 
                                                    (df_processed['longitude'] == lon)].copy()) 
                          for lat, lon in latlon_pairs]
        location_id_col = ('latitude', 'longitude')
    else:
        logger.warning("No location columns found. Treating data as a single sequence.")
        location_groups = [('single_sequence', df_processed.copy())]
        location_id_col = None
    
    total_windows = 0
    
    # Process each location
    for location_id, group_df in location_groups:
        # Sort by timestamp if available
        if time_col and time_col in group_df.columns:
            try:
                group_df = group_df.sort_values(time_col)
                logger.debug(f"Sorted location {location_id} data by {time_col}")
            except Exception as e:
                logger.warning(f"Error sorting by {time_col}: {e}. Using index order.")
                group_df = group_df.sort_index()
        else:
            # Fall back to index sorting for consistency
            group_df = group_df.sort_index()
        
        # Store sorted indices for this location
        location_indices_map[location_id] = group_df.index.tolist()
        
        # Check for required features
        missing_cols = [col for col in feature_columns if col not in group_df.columns]
        if missing_cols:
            logger.error(f"Missing required feature columns for location {location_id}: {missing_cols}")
            continue
            
        # Get feature data as numpy array
        feature_data = group_df[feature_columns].values
        
        # Skip if not enough data for a single window
        if len(feature_data) < input_width:
            logger.warning(f"Location {location_id} has insufficient data points: {len(feature_data)} < {input_width}")
            continue
            
        # Create windows for this location
        for i in range(len(feature_data) - input_width + 1):
            # Extract the window
            window = feature_data[i:i+input_width]
            
            # Store window data
            window_data.append(window)
            
            # Calculate target index for this prediction based on shift
            target_idx = None
            if i + input_width - 1 + shift < len(group_df):
                target_idx = group_df.index[i + input_width - 1 + shift]
            
            # Store comprehensive metadata for each window
            metadata = {
                'location_id': location_id,
                'window_start_idx': group_df.index[i],
                'window_end_idx': group_df.index[i + input_width - 1],
                'prediction_target_idx': target_idx,
                'window_position': i,
                'global_window_index': total_windows,
                'shift': shift
            }
            window_metadata.append(metadata)
            total_windows += 1
    
    if not window_data:
        logger.error("No windows were created. Cannot proceed with prediction.")
        return None, [], {}, 0
    
    # Convert window data to numpy array for TensorFlow
    try:
        windows_array = np.array(window_data)
        logger.info(f"Created a total of {total_windows} windows with shape {windows_array.shape}")
    except ValueError as ve:
        logger.error(f"Could not convert windows to numpy array: {ve}")
        return None, [], {}, 0
    
    # Create TensorFlow dataset
    try:
        windows_dataset = tf.data.Dataset.from_tensor_slices(windows_array)
        windows_dataset = windows_dataset.batch(batch_size)
        windows_dataset = windows_dataset.prefetch(tf.data.AUTOTUNE)
        logger.info("Successfully created TensorFlow dataset for prediction.")
    except Exception as e:
        logger.error(f"Failed to create TensorFlow dataset: {e}")
        return None, [], {}, 0
    
    return windows_dataset, window_metadata, location_indices_map, total_windows


def map_predictions_to_df_improved(df, predictions, window_metadata, model_name):
    """
    Maps predictions back to the original DataFrame using explicit metadata.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The original DataFrame
    predictions : numpy.ndarray
        Array of model predictions
    window_metadata : list
        List of metadata dictionaries for each window
    model_name : str
        Name of the model for the prediction column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with predictions mapped back to original indices
    """
    logger.info(f"Mapping {len(predictions)} {model_name} predictions back to DataFrame")
    
    # Validate inputs
    if len(predictions) != len(window_metadata):
        logger.error(f"Prediction count ({len(predictions)}) doesn't match metadata count ({len(window_metadata)})")
        n_predictions = min(len(predictions), len(window_metadata))
        logger.warning(f"Will map only {n_predictions} predictions")
    else:
        n_predictions = len(predictions)
    
    # Pre-allocate target indices and values for better performance
    target_indices = []
    pred_values = []
    
    # Extract valid target indices and corresponding predictions
    for i in range(n_predictions):
        target_idx = window_metadata[i]['prediction_target_idx']
        if target_idx is not None:
            target_indices.append(target_idx)
            pred_values.append(predictions[i])
    
    # Check for duplicate target indices
    if len(target_indices) != len(set(target_indices)):
        logger.warning(f"Found duplicate target indices. Some predictions may overwrite others.")
        # Count duplicates for detailed logging
        from collections import Counter
        dup_counts = Counter(target_indices)
        duplicates = {idx: count for idx, count in dup_counts.items() if count > 1}
        logger.debug(f"Duplicate indices: {duplicates}")
    
    # Create a series with predictions indexed by their target indices
    pred_series = pd.Series(pred_values, index=target_indices, dtype=float)
    mapped_count = len(pred_series)
    
    logger.info(f"Successfully mapped {mapped_count} {model_name} predictions")
    
    # Add the predictions to a copy of the original DataFrame
    df_with_preds = df.copy()
    column_name = f'sap_velocity_{model_name}'
    
    # Check that all target indices exist in the DataFrame
    missing_indices = [idx for idx in target_indices if idx not in df.index]
    if missing_indices:
        logger.warning(f"Found {len(missing_indices)} target indices that don't exist in the DataFrame index")
        logger.debug(f"First few missing indices: {missing_indices[:5]}")
    
    # Align the predictions with the DataFrame
    df_with_preds[column_name] = pred_series
    
    # Verify mapping success
    filled_count = df_with_preds[column_name].notna().sum()
    if filled_count != mapped_count:
        logger.warning(f"Only {filled_count} of {mapped_count} predictions were added to DataFrame. Check index alignment.")
    
    # Check for potential mapping issues
    mapped_pct = mapped_count / n_predictions * 100 if n_predictions > 0 else 0
    if mapped_pct < 80:  # Warn if less than 80% mapped
        logger.warning(f"Only {mapped_pct:.1f}% of {model_name} predictions were mapped. Check index alignment and shift value.")
    
    return df_with_preds


def prepare_features_from_preprocessed(df, feature_scaler=None, input_width=8, label_width=1, shift=1):
    """
    Prepare features from preprocessed data with improved error handling.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed data
    feature_scaler : sklearn.preprocessing.StandardScaler, optional
        Fitted scaler for features
    input_width : int
        Number of time steps for input window
    label_width : int
        Number of time steps for output window
    shift : int
        Number of steps to shift for prediction
        
    Returns:
    --------
    tuple: (pandas.DataFrame, list)
        - Prepared DataFrame with features scaled
        - List of actual feature columns used
    """
    try:
        logger.info("Preparing features from preprocessed data...")
        if df is None or df.empty:
            logger.error("Input DataFrame is None or empty.")
            return None, []
        
        # Create a copy to avoid modifying the original DataFrame
        df_processed = df.copy()
        
        # Convert temperature to Celsius if 'ta' exists
        if 'ta' in df_processed.columns:
            try:
                # Check if values are likely in Kelvin (most values > 100)
                if df_processed['ta'].median() > 100:
                    logger.info("Converting temperature from Kelvin to Celsius")
                    df_processed['ta'] = df_processed['ta'] - 273.15
            except Exception as e:
                logger.error(f"Error converting temperature: {e}")
        
        # Handle column name standardization
        if 'annual_mean_temperature' in df_processed.columns:
            df_processed.rename(columns={'annual_mean_temperature': 'mean_annual_temp'}, inplace=True)
        if 'annual_precipitation' in df_processed.columns:
            df_processed.rename(columns={'annual_precipitation': 'mean_annual_precip'}, inplace=True)
        
        # Standardize time feature names
        time_features_caps = ['Day sin', 'Week sin', 'Month sin', 'Year sin']
        for col in time_features_caps:
            if col in df_processed.columns:
                lowercase_col = col.lower()
                if lowercase_col != col:
                    logger.info(f"Renaming column '{col}' to '{lowercase_col}'")
                    df_processed.rename(columns={col: lowercase_col}, inplace=True)
        
        # Define the desired order of features
        ordered_features = [
            'ext_rad', 'sw_in', 'ta', 'ws', 'vpd', 'ppfd_in',
            'mean_annual_temp', 'mean_annual_precip',
            'day sin', 'week sin', 'month sin', 'year sin'
        ]
        
        # Filter to get features actually present
        available_features = [col for col in ordered_features if col in df_processed.columns]
        
        # Log missing features
        missing_features = [col for col in ordered_features if col not in df_processed.columns]
        if missing_features:
            logger.warning(f"Some specified features are missing: {missing_features}")
        
        # Check if we have enough features
        if len(available_features) < 3:  # Arbitrary minimum
            logger.warning(f"Very few features available ({len(available_features)}). Results may be unreliable.")
            # Try to find more usable numeric features
            numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
            excluded_cols = ['sap_velocity', 'latitude', 'longitude']
            additional_features = [col for col in numeric_cols 
                                  if col not in available_features 
                                  and col not in excluded_cols]
            if additional_features:
                logger.info(f"Adding {len(additional_features)} additional numeric features.")
                available_features.extend(additional_features)
        
        logger.info(f"Using {len(available_features)} features: {available_features}")
        feature_columns = available_features
        
        # Apply scaling if scaler is provided
        if feature_scaler is not None:
            try:
                if hasattr(feature_scaler, 'n_features_in_'):
                    logger.info(f"Applying feature scaling (scaler was trained on {feature_scaler.n_features_in_} features)")
                    
                    # Ensure we only scale features the scaler was trained on
                    if hasattr(feature_scaler, 'feature_names_in_'):
                        scaler_features = feature_scaler.feature_names_in_
                        common_features = [f for f in scaler_features if f in df_processed.columns]
                        if common_features:
                            logger.info(f"Scaling {len(common_features)} features: {common_features}")
                            df_processed[common_features] = feature_scaler.transform(df_processed[common_features])
                        else:
                            logger.warning("No common features found between scaler and data. Skipping scaling.")
                    else:
                        # Assume the scaler was trained on the features in the same order
                        if len(feature_columns) == feature_scaler.n_features_in_:
                            df_processed[feature_columns] = feature_scaler.transform(df_processed[feature_columns])
                        else:
                            logger.warning(f"Feature count mismatch: data has {len(feature_columns)}, "
                                          f"scaler expects {feature_scaler.n_features_in_}. Skipping scaling.")
                else:
                    logger.warning("Feature scaler doesn't appear to be properly fitted. Skipping scaling.")
            except Exception as e:
                logger.error(f"Error applying feature scaling: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.info("No feature scaler provided. Using unscaled features.")
        
        return df_processed, feature_columns
        
    except Exception as e:
        logger.error(f"Error in prepare_features_from_preprocessed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, []


def make_predictions_improved(df, models, feature_columns, label_scaler, input_width=8, shift=1):
    """
    Make predictions with improved mapping approach.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the features
    models : dict
        Dictionary of loaded models {model_type: model_object}
    feature_columns : list
        List of feature column names
    label_scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler for the target variable
    input_width : int
        Width of the input window
    shift : int
        Prediction shift
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added prediction columns
    """
    logger.info("Making predictions with improved mapping...")
    
    # Validate inputs
    if df is None or df.empty:
        logger.error("Input DataFrame is empty or None")
        return pd.DataFrame()
    
    if not models:
        logger.error("No models provided")
        return df
    
    if not feature_columns:
        logger.error("No feature columns specified")
        return df
    
    if label_scaler is None:
        logger.error("Label scaler is required for predictions")
        return df
    
    # Create windows with explicit metadata, passing the shift parameter
    windows_dataset, window_metadata, location_map, total_windows = create_prediction_windows_improved(
        df, feature_columns, input_width=input_width, shift=shift
    )
    
    if windows_dataset is None or not window_metadata:
        logger.error("Failed to create prediction windows")
        return df
    
    # Store results
    results_df = df.copy()
    
    # Make predictions with each model
    for model_type, model in models.items():
        logger.info(f"Making predictions with {model_type} model")
        
        try:
            # Handle different model types
            is_deep_model = model_type.lower() in ['cnn_lstm', 'lstm', 'transformer', 'ann', 'gru']
            
            if is_deep_model and isinstance(model, tf.keras.Model):
                # For sequence models, use the windowed dataset directly
                preds_scaled = model.predict(windows_dataset, verbose=0)
            elif hasattr(model, 'predict'):
                # For traditional ML models, flatten the windows
                windows_np = np.concatenate([batch.numpy() for batch in windows_dataset], axis=0)
                X_flattened = windows_np.reshape(windows_np.shape[0], -1)  # Flatten each window
                preds_scaled = model.predict(X_flattened)
            else:
                logger.error(f"Model {model_type} lacks a predict method. Skipping.")
                continue
            
            # Standardize prediction shape
            if len(preds_scaled.shape) == 3:  # Sequence output (batch, timesteps, features)
                logger.info(f"Model {model_type} produced 3D output with shape {preds_scaled.shape}.")
                preds_scaled = preds_scaled[:, -1, :].reshape(-1, 1)  # Take the last timestep
            elif len(preds_scaled.shape) == 2 and preds_scaled.shape[1] > 1:
                logger.info(f"Model {model_type} produced 2D output with multiple features: {preds_scaled.shape}.")
                preds_scaled = preds_scaled[:, 0].reshape(-1, 1)  # Take first feature
            elif len(preds_scaled.shape) == 1:
                preds_scaled = preds_scaled.reshape(-1, 1)  # Ensure 2D for inverse transform
            
            # Inverse transform to get actual sap velocity values
            preds_unscaled = label_scaler.inverse_transform(preds_scaled).flatten()
            
            # Map predictions back to DataFrame with explicit metadata
            mapped_results = map_predictions_to_df_improved(
                df, preds_unscaled, window_metadata, model_type
            )
            
            # Add mapped predictions to results DataFrame
            results_df[f'sap_velocity_{model_type}'] = mapped_results[f'sap_velocity_{model_type}']
            
            # Log prediction statistics
            pred_col = f'sap_velocity_{model_type}'
            non_nan_count = results_df[pred_col].notna().sum()
            logger.info(f"Added {non_nan_count} predictions for {model_type} ({non_nan_count/len(results_df)*100:.1f}% coverage)")
            
            if non_nan_count > 0:
                logger.info(f"{model_type} predictions range: [{results_df[pred_col].min():.2f}, {results_df[pred_col].max():.2f}]")
            
        except Exception as e:
            logger.error(f"Error making predictions with {model_type}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Create ensemble prediction if multiple models succeeded
    pred_cols = [col for col in results_df.columns if col.startswith('sap_velocity_')]
    if len(pred_cols) > 1:
        logger.info(f"Creating ensemble prediction from {len(pred_cols)} models")
        results_df['sap_velocity_ensemble'] = results_df[pred_cols].median(axis=1)
        ensemble_count = results_df['sap_velocity_ensemble'].notna().sum()
        logger.info(f"Ensemble has {ensemble_count} predictions ({ensemble_count/len(results_df)*100:.1f}% coverage)")
    elif len(pred_cols) == 1:
        logger.info(f"Only one model available ({pred_cols[0]}). Using it as the ensemble.")
        results_df['sap_velocity_ensemble'] = results_df[pred_cols[0]]
    else:
        logger.warning("No predictions were generated by any model.")
    
    return results_df


def validate_predictions(df_predictions, feature_columns):
    """
    Validate predictions to detect potential mapping issues.
    
    Parameters:
    -----------
    df_predictions : pandas.DataFrame
        DataFrame with predictions
    feature_columns : list
        List of feature columns used
        
    Returns:
    --------
    bool
        True if all validations pass, False otherwise
    """
    logger.info("Validating predictions...")
    
    # Get prediction columns
    pred_cols = [col for col in df_predictions.columns if col.startswith('sap_velocity_')]
    if not pred_cols:
        logger.warning("No prediction columns found to validate")
        return False
    
    validation_results = {}
    validation_passed = True
    
    # 1. Check for reasonable value ranges for sap velocity
    for col in pred_cols:
        pred_data = df_predictions[col].dropna()
        if len(pred_data) == 0:
            logger.error(f"No predictions in column {col}")
            validation_results[f"{col}_empty"] = False
            validation_passed = False
            continue
            
        min_val, max_val = pred_data.min(), pred_data.max()
        if min_val < -10 or max_val > 100:  # Reasonable range for sap velocity
            logger.warning(f"Column {col} has potentially unreasonable values: [{min_val:.2f}, {max_val:.2f}]")
            validation_results[f"{col}_range_issue"] = False
            validation_passed = False
    
    # 2. Check for temporal consistency if time column exists
    time_col = None
    if isinstance(df_predictions.index, pd.DatetimeIndex):
        time_col = df_predictions.index
    else:
        time_cols = [col for col in df_predictions.columns 
                    if 'time' in col.lower() or 'date' in col.lower()]
        if time_cols and time_cols[0] in df_predictions.columns:
            try:
                time_col = pd.to_datetime(df_predictions[time_cols[0]])
            except:
                pass
    
    if time_col is not None:
        for col in pred_cols:
            # Check if predictions change too abruptly over time
            if len(df_predictions[col].dropna()) > 10:  # Need reasonable sample
                pred_series = df_predictions[col].dropna()
                # Create temporary DataFrame with time index for proper diff calculation
                if isinstance(time_col, pd.DatetimeIndex):
                    temp_df = pd.DataFrame({col: pred_series}, index=time_col)
                else:
                    temp_df = pd.DataFrame({col: pred_series, 'time': time_col})
                    temp_df.set_index('time', inplace=True)
                
                # Sort by time to ensure proper diff calculation
                temp_df = temp_df.sort_index()
                abs_diff = temp_df[col].diff().abs()
                
                # Skip empty diff results
                if not abs_diff.empty:
                    max_diff = abs_diff.max()
                    if max_diff > 50:  # Threshold for suspiciously large changes
                        logger.warning(f"Column {col} shows potentially abrupt changes (max diff: {max_diff:.2f})")
                        validation_results[f"{col}_temporal_consistency"] = False
                        validation_passed = False
    
    # 3. Check correlation with expected driving variables
    key_features = [f for f in ['ta', 'vpd', 'ppfd_in'] if f in df_predictions.columns]
    for feat in key_features:
        for col in pred_cols:
            # Create a DataFrame with only the feature and prediction
            valid_data = df_predictions[[feat, col]].dropna()
            if len(valid_data) > 10:  # Need reasonable sample
                try:
                    corr = valid_data.corr().iloc[0, 1]
                    # Check for suspicious lack of correlation with known important features
                    if abs(corr) < 0.05:  # Threshold for suspiciously low correlation
                        logger.warning(f"Column {col} shows very low correlation with {feat}: {corr:.3f}")
                        validation_results[f"{col}_{feat}_correlation"] = False
                        validation_passed = False
                except:
                    # Ignore correlation calculation errors
                    pass
    
    # Log overall validation results
    if validation_passed:
        logger.info("All prediction validations passed.")
    else:
        logger.warning("Some prediction validations failed. Check mapping and model outputs.")
        logger.warning(f"Failed validations: {[k for k, v in validation_results.items() if not v]}")
    
    return validation_passed


def visualize_predictions(df_predictions, output_dir, feature_columns, include_feature_importance=True, models=None):
    """
    Create visualizations of the predictions and optionally feature importance.
    
    Parameters:
    -----------
    df_predictions : pandas.DataFrame
        DataFrame with original features and added prediction columns
    output_dir : Path
        Directory to save visualizations
    feature_columns : list
        List of feature columns used in the model
    include_feature_importance : bool
        Whether to generate feature importance plots
    models : dict, optional
        Dictionary of loaded models
    """
    logger.info("Creating visualizations...")
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('seaborn-v0_8-whitegrid')

    output_dir.mkdir(exist_ok=True, parents=True)
    df_plot = df_predictions.copy()

    # Get prediction columns
    pred_cols = [col for col in df_plot.columns if col.startswith('sap_velocity_')]
    ensemble_col = 'sap_velocity_ensemble' if 'sap_velocity_ensemble' in pred_cols else None
    individual_pred_cols = [col for col in pred_cols if col != ensemble_col]

    if not pred_cols:
        logger.warning("No prediction columns found for visualization.")
        return

    # Find or create time column for plotting
    time_col = None
    if isinstance(df_plot.index, pd.DatetimeIndex):
        time_col = df_plot.index.name or 'datetime'
        df_plot[time_col] = df_plot.index
        logger.info(f"Using DatetimeIndex for plotting.")
    else:
        # Try to find a time column
        time_cols = [c for c in df_plot.columns if 'time' in c.lower() or 'date' in c.lower()]
        for col in time_cols:
            try:
                df_plot[col] = pd.to_datetime(df_plot[col])
                time_col = col
                logger.info(f"Using column '{time_col}' as datetime for plotting.")
                break
            except:
                continue
        
        if time_col is None:
            logger.warning("No suitable time column found. Using row number for x-axis.")
            time_col = 'row_number'
            df_plot[time_col] = range(len(df_plot))

    # Identify locations
    location_col = None
    if 'name' in df_plot.columns:
        location_col = 'name'
        unique_locations = sorted(df_plot['name'].unique())
        logger.info(f"Found {len(unique_locations)} locations using 'name' column.")
    elif 'latitude' in df_plot.columns and 'longitude' in df_plot.columns:
        location_col = ('latitude', 'longitude')
        # Round coordinates for display
        df_plot['lat_rounded'] = df_plot['latitude'].round(2)
        df_plot['lon_rounded'] = df_plot['longitude'].round(2)
        unique_locations = sorted(df_plot[['lat_rounded', 'lon_rounded']].drop_duplicates().values.tolist())
        logger.info(f"Found {len(unique_locations)} locations using lat/lon coordinates.")
    else:
        logger.info("No location columns found. Plotting data as a single sequence.")
        unique_locations = ['single_sequence']

    # Limit number of plots
    max_plots = 10
    if len(unique_locations) > max_plots:
        logger.warning(f"Limiting visualization to {max_plots} locations.")
        locations_to_plot = unique_locations[:max_plots]
    else:
        locations_to_plot = unique_locations

    # Create plots for each location
    for i, location_id in enumerate(locations_to_plot):
        try:
            # Extract data for this location
            if location_col == 'name':
                location_data = df_plot[df_plot['name'] == location_id].copy()
                loc_str = str(location_id)
            elif location_col == ('latitude', 'longitude'):
                lat, lon = location_id
                location_data = df_plot[(df_plot['lat_rounded'] == lat) & 
                                       (df_plot['lon_rounded'] == lon)].copy()
                loc_str = f"Lat={lat:.2f}_Lon={lon:.2f}"
            else:
                location_data = df_plot.copy()
                loc_str = "All_Data"
            
            # Skip if no data
            if location_data.empty:
                logger.warning(f"No data for location {loc_str}. Skipping plot.")
                continue
            
            # Sort by time for proper plotting
            location_data = location_data.sort_values(by=time_col)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True, 
                                     gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot 1: Predictions
            ax = axes[0]
            
            # Plot ensemble prediction if available
            if ensemble_col and ensemble_col in location_data.columns:
                ensemble_data = location_data[[time_col, ensemble_col]].dropna()
                if not ensemble_data.empty:
                    ax.plot(ensemble_data[time_col], ensemble_data[ensemble_col], 
                           label='Ensemble', color='black', linewidth=2.5, zorder=10)
            
            # Plot individual model predictions
            palette = sns.color_palette("tab10", len(individual_pred_cols))
            for j, col in enumerate(individual_pred_cols):
                if col in location_data.columns:
                    model_data = location_data[[time_col, col]].dropna()
                    if not model_data.empty:
                        model_type = col.replace('sap_velocity_', '')
                        ax.plot(model_data[time_col], model_data[col], 
                               label=f'{model_type.upper()}', color=palette[j], 
                               alpha=0.8, linewidth=1.5)
            
            ax.set_title(f'Sap Velocity Predictions - {loc_str}')
            ax.set_ylabel('Sap Velocity')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            
            if isinstance(location_data[time_col].iloc[0], pd.Timestamp):
                ax.tick_params(axis='x', rotation=45)
            
            # Plot 2: Key Features
            ax = axes[1]
            key_features = ['ta', 'vpd', 'ppfd_in', 'sw_in']
            plot_features = [f for f in key_features if f in location_data.columns]
            
            if plot_features:
                # Normalize features for comparison
                norm_data = location_data[plot_features].copy()
                for col in plot_features:
                    min_val, max_val = norm_data[col].min(), norm_data[col].max()
                    if max_val > min_val:
                        norm_data[col] = (norm_data[col] - min_val) / (max_val - min_val)
                    else:
                        norm_data[col] = 0.5
                
                # Plot normalized features
                for j, col in enumerate(plot_features):
                    ax.plot(location_data[time_col], norm_data[col], 
                           label=col, linewidth=1)
                
                ax.set_title('Key Environmental Variables (Normalized)')
                ax.set_ylabel('Normalized Value')
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            else:
                ax.set_visible(False)
            
            # Set x-axis label
            axes[-1].set_xlabel(time_col.replace('_', ' ').title())
            
            # Save the figure
            plt.tight_layout()
            safe_loc_str = loc_str.replace(' ', '_').replace('=', '').replace('.', 'p')
            safe_loc_str = safe_loc_str.replace(',', '').replace('/', '_').replace('\\', '_')
            filename = f'predictions_{safe_loc_str}.png'
            filepath = output_dir / filename
            
            plt.savefig(filepath, bbox_inches='tight', dpi=150)
            logger.info(f"Saved plot: {filepath}")
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error creating plot for location {location_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # Feature Importance Plots
    if include_feature_importance and models is not None and feature_columns:
        logger.info("Generating feature importance plots...")
        
        for model_type, model in models.items():
            try:
                importances = None
                
                # Extract feature importance based on model type
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_') and hasattr(model.coef_, 'shape'):
                    if len(model.coef_.shape) == 1:
                        importances = np.abs(model.coef_)
                    elif len(model.coef_.shape) == 2:
                        importances = np.mean(np.abs(model.coef_), axis=0)
                
                if importances is not None:
                    # Check if dimensions match
                    if len(importances) == len(feature_columns):
                        # Create DataFrame for plotting
                        imp_df = pd.DataFrame({
                            'Feature': feature_columns,
                            'Importance': importances
                        })
                        imp_df = imp_df.sort_values('Importance', ascending=False)
                        
                        # Limit to top 20 features
                        if len(imp_df) > 20:
                            imp_df = imp_df.head(20)
                        
                        # Plot feature importance
                        plt.figure(figsize=(10, max(6, len(imp_df) * 0.3)))
                        ax = sns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis')
                        plt.title(f'Feature Importance - {model_type.upper()}')
                        plt.tight_layout()
                        
                        # Add value labels
                        for p, importance in zip(ax.patches, imp_df['Importance']):
                            width = p.get_width()
                            plt.text(width + width * 0.02, p.get_y() + p.get_height()/2, 
                                    f'{importance:.3f}', ha='left', va='center')
                        
                        # Save plot and data
                        filepath = output_dir / f'feature_importance_{model_type}.png'
                        plt.savefig(filepath, bbox_inches='tight', dpi=150)
                        logger.info(f"Saved feature importance plot: {filepath}")
                        plt.close()
                        
                        # Save as CSV
                        imp_df.to_csv(output_dir / f'feature_importance_{model_type}.csv', index=False)
                    else:
                        logger.warning(f"Feature importance dimensions mismatch for {model_type}: "
                                       f"{len(importances)} importances vs {len(feature_columns)} features")
            except Exception as e:
                logger.error(f"Error creating feature importance plot for {model_type}: {e}")

    logger.info(f"Visualization complete. Results saved to {output_dir}")
# Modify the argument parser to include processes parameter
# Modify the argument parser to include processes parameter
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Predict sap velocity from preprocessed data using parallel processing.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file with preprocessed data.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory. Defaults to ./outputs/<input_file_stem>/')
    parser.add_argument('--models-dir', type=str, default='./outputs/models',
                        help='Directory containing trained model subdirectories.')
    parser.add_argument('--scaler-dir', type=str, default='./outputs/scalers',
                        help='Directory containing feature and label scalers.')
    parser.add_argument('--model-types', type=str, nargs='+', default=['cnn_lstm'],
                        help='List of model types to load and predict with.')
    parser.add_argument('--input-width', type=int, default=DEFAULT_PARAMS['INPUT_WIDTH'],
                        help='Time steps for sequence model input window.')
    parser.add_argument('--shift', type=int, default=DEFAULT_PARAMS['SHIFT'],
                        help='Prediction shift steps ahead.')
    parser.add_argument('--label-width', type=int, default=DEFAULT_PARAMS['LABEL_WIDTH'],
                        help='Time steps in the label window (informational).')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations of predictions.')
    parser.add_argument('--feature-importance', action='store_true',
                        help='Generate feature importance plots for applicable models.')
    parser.add_argument('--validate', action='store_true',
                        help='Perform validation checks on predictions.')
    parser.add_argument('--processes', type=int, default=None,
                        help='Number of processes to use. If None, uses CPU count - 1.')

    return parser.parse_args()

# Update the main function to use parallel prediction
def main():
    """Main execution function"""
    start_time = time.time()
    args = parse_args()

    # Setup directories
    input_file = Path(args.input)
    if not input_file.is_file():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    models_dir = Path(args.models_dir)
    scaler_dir = Path(args.scaler_dir)

    if args.output:
        output_dir = Path(args.output)
    else:
        output_base = Path('./outputs')
        output_dir = output_base / input_file.stem
    output_dir.mkdir(exist_ok=True, parents=True)

    # Log setup information
    logger.info("="*60)
    logger.info(" Sap Velocity Prediction Script (Parallel Processing) ")
    logger.info("="*60)
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Models: {args.model_types}")
    logger.info(f"Window params: width={args.input_width}, shift={args.shift}")
    logger.info(f"Processes: {args.processes if hasattr(args, 'processes') else 'auto'}")

    # Load models
    models = load_models(models_dir, args.model_types)
    if not models:
        logger.error("No models were loaded. Exiting.")
        sys.exit(1)
    logger.info(f"Successfully loaded {len(models)} models: {list(models.keys())}")

    # Load scalers
    feature_scaler, label_scaler = load_scalers(scaler_dir)
    if label_scaler is None:
        logger.error("Label scaler is required but not found. Exiting.")
        sys.exit(1)

    # Load and prepare data
    df_raw = load_preprocessed_data(input_file)
    if df_raw is None or df_raw.empty:
        logger.error("Failed to load input data. Exiting.")
        sys.exit(1)
    logger.info(f"Loaded data with {len(df_raw)} rows and {len(df_raw.columns)} columns")

    # Save original index
    original_index = df_raw.index

    # Prepare features
    df_prepared, feature_columns = prepare_features_from_preprocessed(
        df_raw,
        feature_scaler,
        input_width=args.input_width
    )
    
    if df_prepared is None or not feature_columns:
        logger.error("Failed to prepare features. Exiting.")
        sys.exit(1)
    logger.info(f"Prepared features: {feature_columns}")

    # Make predictions with parallel processing
    df_predictions = make_predictions_parallel(
        df_prepared,
        models,
        feature_columns,
        label_scaler,
        input_width=args.input_width,
        shift=args.shift,
        n_processes=args.processes
    )
    
    if df_predictions.empty:
        logger.error("Prediction failed or returned empty DataFrame. Exiting.")
        sys.exit(1)

    # Restore original index if possible
    if len(original_index) == len(df_predictions):
        try:
            df_final = df_predictions.set_index(original_index)
            logger.info("Restored original index to predictions DataFrame.")
        except Exception as e:
            logger.warning(f"Could not restore original index: {e}. Using current index.")
            df_final = df_predictions
    else:
        logger.warning(f"Length mismatch between original ({len(original_index)}) and "
                      f"predictions ({len(df_predictions)}). Using current index.")
        df_final = df_predictions

    # Validate predictions
    if args.validate:
        validation_passed = validate_predictions(df_final, feature_columns)
        logger.info(f"Prediction validation {'passed' if validation_passed else 'failed'}")

    # Save predictions
    output_file = output_dir / f"{input_file.stem}_predictions_parallel.csv"
    try:
        df_final.to_csv(output_file)
        logger.info(f"Saved predictions to {output_file}")
    except Exception as e:
        logger.error(f"Error saving predictions: {e}")

    # Create visualizations
    if args.visualize:
        visualize_predictions(
            df_final, 
            output_dir, 
            feature_columns,
            include_feature_importance=args.feature_importance,
            models=models
        )

    # Finish
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info("="*60)
    logger.info(" Parallel Prediction Script Completed ")
    logger.info(f" Execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    logger.info("="*60)





if __name__ == "__main__":
    try:
        main()
    except SystemExit as se:
        if se.code != 0:
            logger.info(f"Script exited with code: {se.code}")
        sys.exit(se.code)
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)