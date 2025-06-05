#!/usr/bin/env python
"""
Improved Sap Velocity Prediction Script with Parallel Processing

This script works with preprocessed ERA5-Land data to predict sap velocity,
using an improved index mapping approach for more reliable time series predictions.
Parallel processing is implemented for handling different spatial chunks
and window creation to improve performance. Includes optimized DataFrame splitting.
Version 3: Re-implements --parallel-windows flag for parallel window creation
           followed by sequential prediction, using NumPy arrays.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
# Conditional import for plotting
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    # Add logger warning if plotting is requested but unavailable later
from datetime import datetime
from pathlib import Path
import logging
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(6)  # Parallel operations
tf.config.threading.set_intra_op_parallelism_threads(8)  # Within operations
from sklearn.preprocessing import StandardScaler
import time
import warnings
import json
import traceback
import concurrent.futures
from functools import partial
from itertools import chain
from tqdm import tqdm # Import tqdm for progress bars

# Set up logging
# Use a new log file name for this version
log_file_name = "sap_velocity_prediction_optimized_v3.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_name),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("sap_prediction")

# Default parameters if config is not available
DEFAULT_PARAMS = {
    'MODEL_TYPES': ['xgb', 'rf', 'cnn_lstm'],
    'INPUT_WIDTH': 8,
    'LABEL_WIDTH': 1,
    'SHIFT': 1,
    'N_WORKERS': None  # None means use available CPU count
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
    MODELS_DIR = BASE_DIR / 'outputs/models' # Adjusted default path based on logs
    OUTPUT_DIR = BASE_DIR / 'outputs/predictions' # Adjusted default path based on logs
    SCALER_DIR = BASE_DIR / 'outputs/scalers' # Adjusted default path based on logs
    MODEL_TYPES = DEFAULT_PARAMS['MODEL_TYPES']


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
                # Check for saved model directories (.pb format)
                model_dirs = [d for d in type_dir.iterdir() if d.is_dir() and (d / 'saved_model.pb').exists()]
                if model_dirs:
                    model_path = sorted(model_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                else:
                     # Check for older HDF5 format (.h5) as a fallback
                     h5_files = list(type_dir.glob('*.h5'))
                     if h5_files:
                          model_path = sorted(h5_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                     else:
                          logger.warning(f"No .keras, saved_model.pb, or .h5 files found for {model_type} in {type_dir}")
                          continue

            # Load the model
            if model_path:
                try:
                    logger.info(f"Loading {model_type} model from {model_path}")
                    # Suppress optimizer warning if needed and avoid recompiling
                    with warnings.catch_warnings():
                         warnings.simplefilter("ignore", category=UserWarning)
                         # Set compile=False for Keras 3+ compatibility when optimizer state isn't needed for inference
                         model = tf.keras.models.load_model(model_path, compile=False)
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

        logger.info("Reading CSV file (this might take a while for large files)...")
        start_read = time.time()
        # Using low_memory=False is a fallback if dtypes aren't specified
        df = pd.read_csv(data_file_path, low_memory=False)
        read_time = time.time() - start_read
        logger.info(f"CSV read complete in {read_time:.2f} seconds.")
        logger.info(f"Initial shape: {df.shape}")


        # --- Data Cleaning/Preparation ---
        # Remove potential unnamed index column if it exists after loading
        if 'Unnamed: 0' in df.columns:
             df = df.drop(columns=['Unnamed: 0'])
             logger.info("Removed 'Unnamed: 0' column.")

        # Drop rows with NaN values - **CRITICAL WARNING FROM LOGS**
        initial_rows = len(df)
        df.dropna(inplace=True) # This dropped ~90M rows in the user's log!
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
             logger.warning(f"CRITICAL: Dropped {dropped_rows} rows ({dropped_rows/initial_rows*100:.1f}%) containing NaN values. Check input data quality and preprocessing steps.")
             if len(df) == 0:
                  logger.error("DataFrame is empty after dropping NaN values. Cannot proceed.")
                  return None

        # Try to identify and sort by timestamp column
        timestamp_col = None
        potential_time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        # Prioritize 'timestamp.1' based on user logs
        if 'timestamp.1' in df.columns:
             timestamp_col = 'timestamp.1'
        elif potential_time_cols:
             timestamp_col = potential_time_cols[0] # Pick the first likely candidate

        if timestamp_col:
             logger.info(f"Attempting to sort data by column: {timestamp_col}")
             try:
                 # Convert to datetime if not already, handle potential errors
                 df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
                 # Drop rows where conversion failed BEFORE sorting
                 conversion_nan_rows = df[timestamp_col].isnull().sum()
                 if conversion_nan_rows > 0:
                      logger.warning(f"Dropping {conversion_nan_rows} rows where '{timestamp_col}' could not be converted to datetime.")
                      df.dropna(subset=[timestamp_col], inplace=True)

                 if len(df) > 0:
                      df.sort_values(by=timestamp_col, inplace=True)
                      logger.info(f"Successfully sorted data by {timestamp_col}.")
                 else:
                      logger.error("DataFrame became empty after handling timestamp conversion errors.")
                      return None

             except Exception as e:
                 logger.error(f"Could not sort by timestamp column {timestamp_col}: {e}. Proceeding without sorting.")
                 timestamp_col = None # Reset if sorting failed
        else:
             logger.info("No clear timestamp column found for sorting.")

        logger.info(f"Loaded and preprocessed data with {len(df)} rows and {len(df.columns)} columns")

        # Basic data validation
        if df.empty:
            logger.error("Loaded DataFrame is empty after preprocessing.")
            return None

        # Check for important columns (using standardized names)
        required_features = ['ta', 'vpd', 'sw_in', 'ppfd_in', 'ext_rad', 'ws', 'mean_annual_temp', 'mean_annual_precip']
        # Check against standardized names expected later
        current_cols = df.columns
        missing_features = [f for f in required_features
                             if not any(col_name in current_cols for col_name in [
                                 f, f.replace('_', ' '), f.replace('mean_', 'annual_')
                             ])]
        if missing_features:
            logger.warning(f"Some potentially important features might be missing or named differently: {missing_features}")

        return df

    except Exception as e:
        logger.error(f"Error loading preprocessed data from {data_file}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def process_location_group(location_id, group_df, feature_columns, input_width, shift, global_offset=0):
    """
    Process a single location group to create prediction windows.
    This function is designed to be used with parallel processing.

    Parameters:
    -----------
    location_id : str or tuple
        Identifier for the location group
    group_df : pandas.DataFrame
        DataFrame containing data for this location (should be a copy)
    feature_columns : list
        List of feature column names to include
    input_width : int
        Width of input window
    shift : int
        Prediction shift
    global_offset : int
        Offset to add to window indices for global indexing (approximate)

    Returns:
    --------
    tuple: (list, list, list, int)
        - List of window data arrays (numpy)
        - List of metadata dictionaries
        - List of sorted indices for this location
        - Count of windows created
    """
    try:
        # Prepare results
        window_data = []
        window_metadata = []

        # Data should already be sorted by time within the group if possible
        # Store original indices for metadata
        location_indices = group_df.index.tolist()

        # Check for required features
        missing_cols = [col for col in feature_columns if col not in group_df.columns]
        if missing_cols:
            # This should ideally not happen if prepare_features handles it, but check defensively
            logger.warning(f"Location {location_id} missing columns during windowing: {missing_cols}. Skipping.")
            return [], [], location_indices, 0

        # Get feature data as numpy array for efficiency
        try:
             feature_data = group_df[feature_columns].values
        except KeyError as ke:
             logger.error(f"KeyError accessing feature columns for location {location_id}: {ke}. Columns: {group_df.columns}. Required: {feature_columns}")
             return [], [], location_indices, 0


        # Skip if not enough data for a single window
        if len(feature_data) < input_width:
            # logger.debug(f"Location {location_id} has insufficient data ({len(feature_data)}) for window width {input_width}. Skipping.")
            return [], [], location_indices, 0

        # Create windows for this location using array slicing (efficient)
        window_count = 0
        num_possible_windows = len(feature_data) - input_width + 1
        for i in range(num_possible_windows):
            # Extract the window
            window = feature_data[i : i + input_width]

            # Store window data
            window_data.append(window)

            # Calculate target index for this prediction based on shift
            # Use the original DataFrame index stored in location_indices
            target_idx_pos = i + input_width - 1 + shift
            target_idx = location_indices[target_idx_pos] if target_idx_pos < len(location_indices) else None

            # Store comprehensive metadata for each window using original indices
            metadata = {
                'location_id': location_id,
                'window_start_idx': location_indices[i],
                'window_end_idx': location_indices[i + input_width - 1],
                'prediction_target_idx': target_idx,
                'window_position': i, # Local position within the group
                'global_window_index': global_offset + window_count, # Approximate global index
                'shift': shift
            }
            window_metadata.append(metadata)
            window_count += 1

        return window_data, window_metadata, location_indices, window_count

    except Exception as e:
        # Log error but don't halt parallel processing
        logger.error(f"Error processing location {location_id}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return [], [], [], 0


def create_prediction_windows_parallel(df, feature_columns, input_width=8, shift=1, n_workers=None):
    """
    Create time series windows (as NumPy array) and metadata in parallel by location.
    Returns NumPy array directly, not tf.data.Dataset.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with time series data
    feature_columns : list
        List of feature column names to include in the windows
    input_width : int
        Window size for input sequence
    shift : int
        Number of steps to shift for prediction
    n_workers : int or None
        Number of worker processes to use for parallel processing.
        If None, uses available CPU count.

    Returns:
    --------
    tuple: (np.ndarray | None, list, dict, int)
        - NumPy array of windowed features [n_windows, input_width, n_features], or None if error
        - List of metadata dictionaries for each window
        - Dictionary mapping location identifier to sorted indices
        - Total number of windows created
    """
    logger.info(f"Creating prediction windows (NumPy output) PARALLELLY with input_width={input_width}, shift={shift}, n_workers={n_workers or 'auto'}")
    start_time = time.time()

    # Ensure unique index for reliable mapping if not already
    df_processed = df.copy()
    if not df_processed.index.is_unique:
        logger.warning("DataFrame index is not unique. Resetting index for reliable mapping.")
        df_processed = df_processed.reset_index(drop=True)

    # --- Grouping using efficient groupby ---
    location_col = None
    if 'name' in df_processed.columns:
        logger.info("Grouping data by 'name' column.")
        location_col = 'name'
    elif 'latitude' in df_processed.columns and 'longitude' in df_processed.columns:
        logger.info("Grouping data by 'latitude' and 'longitude' columns.")
        location_col = ['latitude', 'longitude']
    else:
        logger.warning("No location columns found. Treating data as a single sequence.")
        location_groups_iter = [('single_sequence', df_processed.copy())]
        num_groups = 1

    if location_col:
        try:
            grouped = df_processed.groupby(location_col, sort=False, observed=True)
            num_groups = grouped.ngroups
            logger.info(f"Identified {num_groups} location groups using groupby.")
            # Create iterator of (id, group_df_copy) for parallel processing
            # Copying here ensures data isolation for each worker
            location_groups_iter = ((name, group.copy()) for name, group in grouped)
        except Exception as e:
            logger.error(f"Error during groupby: {e}. Treating as single sequence.")
            location_groups_iter = [('single_sequence', df_processed.copy())]
            num_groups = 1

    logger.info(f"Preparing {num_groups} location groups for parallel window creation")

    # Prepare for parallel processing
    if n_workers is None:
        import multiprocessing
        n_workers = max(1, multiprocessing.cpu_count() - 2)
    n_workers = min(n_workers, num_groups) # Cannot use more workers than groups

    logger.info(f"Using {n_workers} workers for parallel window creation")

    # Process locations in parallel
    all_window_data = []
    all_window_metadata = []
    location_indices_map = {}
    total_windows = 0

    # --- Parallel Execution ---
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        current_offset = 0 # Keep track of approximate global index offset

        # Submit tasks
        for location_id, group_df in location_groups_iter:
            task_offset = current_offset
            # Rough estimate for next offset - actual offset determined by results
            current_offset += max(0, len(group_df) - input_width + 1)

            futures.append(executor.submit(
                process_location_group,
                location_id,
                group_df, # Pass the copy
                feature_columns,
                input_width,
                shift,
                task_offset
            ))

        # Process results as they complete
        logger.info(f"Submitted {len(futures)} tasks to ProcessPoolExecutor. Waiting for results...")
        processed_count = 0
        # Recalculate global indices accurately as results come in
        actual_global_window_index = 0
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Creating windows"):
            try:
                window_data, window_metadata, location_indices, window_count = future.result()
                processed_count += 1

                if window_count > 0 and window_metadata:
                    location_id = window_metadata[0]['location_id']
                    # --- Correct global window index ---
                    for i in range(len(window_metadata)):
                         window_metadata[i]['global_window_index'] = actual_global_window_index + i
                    actual_global_window_index += window_count
                    # ------------------------------------
                    all_window_data.extend(window_data)
                    all_window_metadata.extend(window_metadata)
                    location_indices_map[location_id] = location_indices
                    total_windows += window_count
                # else: # Log skipped locations if needed
                #     logger.debug(f"Skipped location (no windows created or error). Processed {processed_count}/{len(futures)}")

            except Exception as e:
                processed_count += 1
                logger.error(f"Error processing a future task ({processed_count}/{len(futures)}): {e}")

    if not all_window_data:
        logger.error("No windows were created across all locations. Cannot proceed with prediction.")
        return None, [], {}, 0

    # --- Convert to NumPy array ---
    logger.info(f"Aggregated results from parallel processing. Total windows: {total_windows}")
    logger.info("Converting window data to NumPy array...")
    try:
        # Using np.stack assumes all window_data elements are already numpy arrays of the same shape
        windows_array = np.stack(all_window_data, axis=0)
        logger.info(f"Created NumPy array of shape {windows_array.shape}")
    except ValueError as ve:
        logger.error(f"Could not convert windows to NumPy array: {ve}. Check window creation logic.")
        # Check shapes if possible
        shapes = set(arr.shape for arr in all_window_data if hasattr(arr, 'shape'))
        logger.error(f"Unique window shapes found: {shapes}")
        return None, [], {}, 0
    except Exception as e:
        logger.error(f"Unexpected error during NumPy array conversion: {e}")
        return None, [], {}, 0

    # --- NO TensorFlow Dataset Creation Here ---

    end_time = time.time()
    logger.info(f"Parallel window creation (NumPy output) finished in {end_time - start_time:.2f} seconds.")

    # Return NumPy array instead of tf.data.Dataset
    return windows_array, all_window_metadata, location_indices_map, total_windows


def create_prediction_windows_improved(df, feature_columns, input_width=8, shift=1):
    """
    Create time series windows (as NumPy array) and metadata sequentially by location.
    Returns NumPy array directly, not tf.data.Dataset.

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

    Returns:
    --------
    tuple: (np.ndarray | None, list, dict, int)
        - NumPy array of windowed features [n_windows, input_width, n_features], or None if error
        - List of metadata dictionaries for each window
        - Dictionary mapping location identifier to sorted indices
        - Total number of windows created
    """
    logger.info(f"Creating prediction windows (NumPy output) SEQUENTIALLY with input_width={input_width}, shift={shift}")
    logger.info(f"Using features: {feature_columns}")
    start_time = time.time()

    # Ensure unique index for reliable mapping
    df_processed = df.copy()
    if not df_processed.index.is_unique:
        logger.warning("DataFrame index is not unique. Resetting index for reliable mapping.")
        df_processed = df_processed.reset_index(drop=True)

    # Store all windows with their metadata
    window_data = []
    window_metadata = []
    location_indices_map = {}

    # --- Grouping using efficient groupby ---
    location_col = None
    if 'name' in df_processed.columns:
        logger.info("Grouping data by 'name' column.")
        location_col = 'name'
    elif 'latitude' in df_processed.columns and 'longitude' in df_processed.columns:
        logger.info("Grouping data by 'latitude' and 'longitude' columns.")
        location_col = ['latitude', 'longitude']
    else:
        logger.warning("No location columns found. Treating data as a single sequence.")
        location_groups_iter = [('single_sequence', df_processed.copy())]
        num_groups = 1

    if location_col:
        try:
            grouped = df_processed.groupby(location_col, sort=False, observed=True)
            num_groups = grouped.ngroups
            logger.info(f"Identified {num_groups} location groups using groupby.")
            location_groups_iter = grouped # Iterate directly over the groupby object
        except Exception as e:
            logger.error(f"Error during groupby: {e}. Treating as single sequence.")
            location_groups_iter = [('single_sequence', df_processed.copy())]
            num_groups = 1

    total_windows = 0

    # Process each location sequentially
    logger.info(f"Processing {num_groups} location groups sequentially...")
    for location_id, group_df in tqdm(location_groups_iter, total=num_groups, desc="Processing locations"):

        # Data should already be sorted by time if done during loading
        location_indices = group_df.index.tolist()
        location_indices_map[location_id] = location_indices

        # Check for required features
        missing_cols = [col for col in feature_columns if col not in group_df.columns]
        if missing_cols:
            logger.error(f"Missing required feature columns for location {location_id}: {missing_cols}")
            continue

        # Get feature data as numpy array
        try:
            feature_data = group_df[feature_columns].values
        except KeyError: # Should not happen if prepare_features worked correctly
             logger.error(f"KeyError getting features for {location_id}. Skipping.")
             continue


        # Skip if not enough data for a single window
        if len(feature_data) < input_width:
            # logger.warning(f"Location {location_id} has insufficient data points: {len(feature_data)} < {input_width}")
            continue

        # Create windows for this location
        num_possible_windows = len(feature_data) - input_width + 1
        for i in range(num_possible_windows):
            window = feature_data[i : i + input_width]
            window_data.append(window)

            target_idx_pos = i + input_width - 1 + shift
            target_idx = location_indices[target_idx_pos] if target_idx_pos < len(location_indices) else None

            metadata = {
                'location_id': location_id,
                'window_start_idx': location_indices[i],
                'window_end_idx': location_indices[i + input_width - 1],
                'prediction_target_idx': target_idx,
                'window_position': i,
                'global_window_index': total_windows, # Index across all windows generated so far
                'shift': shift
            }
            window_metadata.append(metadata)
            total_windows += 1

    if not window_data:
        logger.error("No windows were created. Cannot proceed with prediction.")
        return None, [], {}, 0

    # Convert window data to numpy array
    logger.info(f"Created a total of {total_windows} windows sequentially.")
    logger.info("Converting window data to NumPy array...")
    try:
        windows_array = np.stack(window_data, axis=0)
        logger.info(f"Created NumPy array of shape {windows_array.shape}")
    except ValueError as ve:
        logger.error(f"Could not convert windows to numpy array: {ve}")
        shapes = set(arr.shape for arr in window_data if hasattr(arr, 'shape'))
        logger.error(f"Unique window shapes found: {shapes}")
        return None, [], {}, 0
    except Exception as e:
        logger.error(f"Unexpected error during NumPy array conversion: {e}")
        return None, [], {}, 0

    # --- NO TensorFlow Dataset Creation Here ---

    end_time = time.time()
    logger.info(f"Sequential window creation (NumPy output) finished in {end_time - start_time:.2f} seconds.")
    # Return NumPy array instead of tf.data.Dataset
    return windows_array, window_metadata, location_indices_map, total_windows


def process_spatial_chunk(chunk_df, model_type, model, feature_columns, label_scaler, input_width, shift, chunk_id):
    """
    Process a spatial chunk for a single model. (Used by parallel prediction)
    MODIFIED: Creates NumPy windows, predicts directly on NumPy array.

    Parameters:
    -----------
    chunk_df : pandas.DataFrame
        Chunk of data to process (should be a copy)
    model_type : str
        Type of model being used
    model : object
        Loaded model
    feature_columns : list
        List of feature columns
    label_scaler : object
        Scaler for label values
    input_width : int
        Window size for input
    shift : int
        Prediction shift
    chunk_id : int
        ID of this chunk for logging

    Returns:
    --------
    tuple: (list, list, dict)
        - List of prediction values (unscaled)
        - List of metadata dictionaries corresponding to predictions
        - Dictionary mapping location ID to indices within this chunk (less relevant now)
    """
    try:
        # logger.debug(f"Processing chunk {chunk_id} with {len(chunk_df)} rows for model {model_type}")

        # Create windows (as NumPy array) using the sequential version for this chunk
        windows_np, window_metadata, location_map, total_windows = create_prediction_windows_improved(
            chunk_df, feature_columns, input_width=input_width, shift=shift
        )

        if windows_np is None or not window_metadata:
            # logger.warning(f"No prediction windows (NumPy) were created for chunk {chunk_id}")
            return [], [], {}

        # Make predictions based on model type using the NumPy array
        is_deep_model = model_type.lower() in ['cnn_lstm', 'lstm', 'transformer', 'ann', 'gru']

        # Define a reasonable batch size for model.predict on NumPy arrays
        # This prevents potential memory issues if windows_np is huge
        predict_batch_size = 1024 # Adjust as needed based on model size and memory

        if is_deep_model and isinstance(model, tf.keras.Model):
            # logger.debug(f"Chunk {chunk_id}: Predicting with DL model {model_type} on NumPy array shape {windows_np.shape}...")
            # Use model.predict directly on the NumPy array, specifying batch size
            preds_scaled = model.predict(windows_np, batch_size=predict_batch_size, verbose=0)
        elif hasattr(model, 'predict'):
            # For traditional ML models, flatten the windows if needed
            # logger.debug(f"Chunk {chunk_id}: Predicting with ML model {model_type} on NumPy array shape {windows_np.shape}...")
            if windows_np.ndim == 3: # (num_windows, timesteps, features)
                 X_flattened = windows_np.reshape(windows_np.shape[0], -1) # Flatten each window
            else: # Already 2D or unexpected shape
                 X_flattened = windows_np
                 logger.warning(f"Chunk {chunk_id}: Input shape for ML model {model_type} is {X_flattened.shape}, using as is.")

            # Note: Scikit-learn models typically don't have a batch_size argument in predict
            # If X_flattened is too large, prediction might need manual batching loop here
            # For now, assume it fits in memory for the ML models
            preds_scaled = model.predict(X_flattened)
        else:
            logger.error(f"Model {model_type} lacks a predict method. Skipping chunk {chunk_id}.")
            return [], [], {}

        # Standardize prediction shape (ensure 2D for scaler)
        if preds_scaled.ndim == 3: preds_scaled = preds_scaled[:, -1, :] # Take last timestep
        if preds_scaled.ndim == 1: preds_scaled = preds_scaled.reshape(-1, 1)
        elif preds_scaled.ndim == 2 and preds_scaled.shape[1] > 1:
             logger.warning(f"Chunk {chunk_id}, Model {model_type}: Output multiple features ({preds_scaled.shape[1]}). Using first.")
             preds_scaled = preds_scaled[:, 0].reshape(-1, 1)
        elif preds_scaled.ndim > 2 or preds_scaled.ndim == 0:
             logger.error(f"Chunk {chunk_id}, Model {model_type}: Unexpected prediction shape {preds_scaled.shape}. Skipping.")
             return [], [], {}

        # Check consistency between predictions and metadata
        if len(preds_scaled) != len(window_metadata):
             logger.error(f"Chunk {chunk_id}, Model {model_type}: Mismatch predictions ({len(preds_scaled)}) vs metadata ({len(window_metadata)}). Skipping.")
             return [], [], {}

        # Inverse transform
        try:
            preds_unscaled = label_scaler.inverse_transform(preds_scaled).flatten()
        except ValueError as e:
            logger.error(f"Chunk {chunk_id}, Model {model_type}: Error inverse transform: {e}. Scaled preds shape: {preds_scaled.shape}")
            return [], [], {}

        # logger.debug(f"Generated {len(preds_unscaled)} predictions for chunk {chunk_id}")
        return preds_unscaled.tolist(), window_metadata, location_map

    except Exception as e:
        logger.error(f"Error processing chunk {chunk_id} with model {model_type}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return [], [], {}


# --- OPTIMIZED split_dataframe_by_location ---
def split_dataframe_by_location(df, max_chunks=None):
    """
    Split DataFrame into chunks based on spatial locations using groupby for efficiency.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    max_chunks : int, optional
        Maximum number of chunks to create. If None, each location becomes a chunk.
        If specified, locations might be grouped into larger chunks.

    Returns:
    --------
    list
        List of DataFrame chunks (copies)
    """
    logger.info("Splitting data by location using optimized groupby method...")
    start_split_time = time.time()
    chunks = []
    location_col = None

    # Identify location columns
    if 'name' in df.columns:
        logger.info("Grouping by 'name' column")
        location_col = 'name'
    elif 'latitude' in df.columns and 'longitude' in df.columns:
        logger.info("Grouping by latitude/longitude coordinates")
        location_col = ['latitude', 'longitude']
    else:
        logger.warning("No location columns ('name' or 'latitude'/'longitude') found. Treating data as a single chunk.")
        return [df.copy()] # Return a copy to be consistent

    # Group by the identified location column(s)
    try:
        # Using observed=True can be faster if the location column is categorical or has many groups
        # Using sort=False might speed up grouping but changes chunk order (acceptable here)
        logger.info(f"Performing groupby on column(s): {location_col}...")
        group_start_time = time.time()
        # Ensure the grouping column(s) exist
        if isinstance(location_col, list):
             missing_group_cols = [c for c in location_col if c not in df.columns]
        else:
             missing_group_cols = [location_col] if location_col not in df.columns else []

        if missing_group_cols:
             logger.error(f"Cannot group by non-existent columns: {missing_group_cols}. Returning single chunk.")
             return [df.copy()]

        grouped = df.groupby(location_col, observed=True, sort=False)
        num_groups = grouped.ngroups
        group_end_time = time.time()
        logger.info(f"Groupby identified {num_groups} unique location groups in {group_end_time - group_start_time:.2f} seconds.")

        # Decide whether to return individual groups or regroup them
        if max_chunks is None or num_groups <= max_chunks:
            logger.info(f"Creating {num_groups} chunks, one per location group.")
            # Iterate through groups and append a copy of each group's DataFrame
            # Use tqdm for progress bar if there are many groups
            desc = f"Extracting {num_groups} location chunks"
            for name, group_df in tqdm(grouped, total=num_groups, desc=desc, disable=num_groups < 100):
                 # Make a copy to ensure independence, crucial for parallel processing later
                chunks.append(group_df.copy())
        else:
            logger.info(f"Regrouping {num_groups} locations into approximately {max_chunks} target chunks.")
            # Strategy: Assign each group to one of the `max_chunks` bins.

            # Get group identifiers (keys)
            group_indices = list(grouped.groups.keys())
            # Shuffle to potentially balance chunk sizes (optional but can help)
            np.random.shuffle(group_indices)

            # Create bins for chunks
            chunk_bins = [[] for _ in range(max_chunks)]
            for i, group_key in enumerate(group_indices):
                chunk_bins[i % max_chunks].append(group_key) # Distribute keys round-robin

            logger.info(f"Concatenating groups into {max_chunks} final chunks...")
            # Create the final chunks by concatenating DataFrames corresponding to keys in each bin
            for i, bin_keys in enumerate(tqdm(chunk_bins, desc="Creating final chunks")):
                if not bin_keys: continue # Skip empty bins

                # Efficiently select rows for all keys in the current bin using boolean indexing
                if isinstance(location_col, list): # Multi-column case (lat/lon)
                    # Create MultiIndex for efficient check
                    multi_index_keys = pd.MultiIndex.from_tuples(bin_keys, names=location_col)
                    # Check if DataFrame index matches location columns
                    if df.index.names == location_col:
                        # Use .loc for potentially faster index-based selection
                        chunk_df = df.loc[df.index.isin(multi_index_keys)].copy()
                    else:
                        # Fallback: Create temporary multi-index on the fly for comparison
                        try:
                            chunk_df = df[pd.MultiIndex.from_frame(df[location_col]).isin(multi_index_keys)].copy()
                        except KeyError as e:
                             logger.error(f"KeyError during multi-index filtering for chunk {i}: {e}. Bin keys: {bin_keys[:5]}...")
                             chunk_df = pd.DataFrame() # Create empty df to avoid error propagation
                else: # Single column case ('name')
                     # Use isin for efficient filtering on the column
                     chunk_df = df[df[location_col].isin(bin_keys)].copy()

                if not chunk_df.empty:
                    chunks.append(chunk_df)
                # else: # Log if a chunk ends up empty after filtering
                #     logger.debug(f"Chunk bin {i} resulted in an empty DataFrame.")

            logger.info(f"Successfully created {len(chunks)} chunks by regrouping.")

    except Exception as e:
        logger.error(f"Error during optimized groupby splitting: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.error("Falling back to returning the entire DataFrame as one chunk.")
        return [df.copy()]

    if not chunks:
         logger.warning("Splitting resulted in zero chunks. Returning the original DataFrame as one chunk.")
         return [df.copy()]

    split_end_time = time.time()
    logger.info(f"Finished splitting data into {len(chunks)} chunks in {split_end_time - start_split_time:.2f} seconds.")
    # --- Memory Usage Check (Optional) ---
    try:
        total_memory = sum(chunk.memory_usage(deep=True).sum() for chunk in chunks) / (1024**2) # MB
        logger.info(f"Approximate total memory usage of {len(chunks)} chunks: {total_memory:.2f} MB")
    except Exception as mem_e:
        logger.warning(f"Could not calculate chunk memory usage: {mem_e}")
    # -------------------------------------
    return chunks
# --- End of OPTIMIZED function ---


def map_predictions_to_df_improved(df, predictions, window_metadata, model_name):
    """
    Maps predictions back to the original DataFrame using explicit metadata.

    Parameters:
    -----------
    df : pandas.DataFrame
        The original DataFrame (or a copy) to map predictions onto
    predictions : numpy.ndarray or list
        Array/list of model predictions
    window_metadata : list
        List of metadata dictionaries for each window, MUST correspond to predictions
    model_name : str
        Name of the model for the prediction column

    Returns:
    --------
    pandas.Series
        A Series containing the predictions, indexed by the target index from the original df.
        NaN where no prediction exists for an index.
    """
    logger.info(f"Mapping {len(predictions)} {model_name} predictions back to DataFrame structure")
    map_start_time = time.time()

    # Validate inputs
    if len(predictions) != len(window_metadata):
        logger.error(f"CRITICAL: Prediction count ({len(predictions)}) doesn't match metadata count ({len(window_metadata)}) for {model_name}. Cannot map accurately.")
        # Return an empty series aligned with df's index
        return pd.Series(dtype=float, name=f'sap_velocity_{model_name}', index=df.index)

    # --- Efficient Mapping using Series ---
    # Create lists of target indices and corresponding prediction values
    target_indices = [meta['prediction_target_idx'] for meta in window_metadata]
    pred_values = list(predictions) # Ensure it's a list

    # Filter out entries where target index is None (e.g., windows at the very end)
    valid_map = [(idx, val) for idx, val in zip(target_indices, pred_values) if idx is not None]

    if not valid_map:
        logger.warning(f"No valid target indices found for mapping {model_name} predictions.")
        return pd.Series(dtype=float, name=f'sap_velocity_{model_name}', index=df.index)

    # Unzip into separate lists
    valid_indices, valid_preds = zip(*valid_map)

    # Create a pandas Series with predictions indexed by their target indices
    # This automatically handles alignment and potential duplicates (keeps last by default)
    pred_series = pd.Series(valid_preds, index=valid_indices, dtype=float, name=f'sap_velocity_{model_name}')

    # --- Handle Duplicate Target Indices ---
    if pred_series.index.has_duplicates:
        num_duplicates = pred_series.index.duplicated().sum()
        logger.warning(f"Found {num_duplicates} duplicate target indices for {model_name}. The last prediction for each index was kept.")
        # If averaging duplicates is desired:
        # pred_series = pred_series.groupby(pred_series.index).mean()
        # pred_series.name = f'sap_velocity_{model_name}' # Ensure name is preserved after groupby

    # Reindex the series to match the original DataFrame's index
    # This ensures the output Series has the same index as df, with NaNs where no prediction exists.
    try:
        pred_series_aligned = pred_series.reindex(df.index)
    except ValueError as e:
         logger.error(f"Error reindexing prediction series for {model_name}: {e}")
         logger.error(f"Original df index type: {type(df.index)}, length: {len(df.index)}")
         logger.error(f"Prediction series index type: {type(pred_series.index)}, length: {len(pred_series.index)}")
         # Fallback: return an empty series aligned with df's index
         return pd.Series(dtype=float, name=f'sap_velocity_{model_name}', index=df.index)


    mapped_count = pred_series_aligned.notna().sum()
    map_end_time = time.time()
    logger.info(f"Successfully mapped {mapped_count} {model_name} predictions in {map_end_time - map_start_time:.2f} seconds.")

    # Sanity check mapping percentage
    total_possible_targets = len([m for m in window_metadata if m['prediction_target_idx'] is not None])
    if total_possible_targets > 0:
        mapped_pct = mapped_count / total_possible_targets * 100
        if mapped_pct < 99: # Allow for minor discrepancies
             logger.warning(f"Mapped percentage for {model_name} is {mapped_pct:.1f}%. Check for index issues if this is unexpected.")
    elif mapped_count > 0:
         logger.warning(f"Mapped {mapped_count} predictions but calculated 0 possible targets. Metadata issue?")


    return pred_series_aligned


def make_predictions_parallel(df, models, feature_columns, label_scaler, input_width=8, shift=1, n_workers=None):
    """
    Make predictions with parallel processing for spatial chunks.
    MODIFIED: Uses process_spatial_chunk that predicts on NumPy arrays.

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
    n_workers : int or None
        Number of worker processes, if None uses CPU count

    Returns:
    --------
    pandas.DataFrame
        DataFrame with added prediction columns
    """
    logger.info("Making predictions with PARALLEL processing for spatial chunks...")
    parallel_start_time = time.time()

    # Validate inputs
    if df is None or df.empty: logger.error("Input DataFrame is empty"); return pd.DataFrame()
    if not models: logger.error("No models provided"); return df.copy()
    if not feature_columns: logger.error("No feature columns specified"); return df.copy()
    if label_scaler is None: logger.error("Label scaler required"); return df.copy()

    # Determine optimal number of workers
    if n_workers is None:
        import multiprocessing
        n_workers = max(1, multiprocessing.cpu_count() - 2) # Leave some cores free

    logger.info(f"Using {n_workers} worker processes for parallel prediction")

    # Split data into spatial chunks using the OPTIMIZED function
    max_chunks = n_workers * 4 # Aim for a few chunks per worker
    chunks = split_dataframe_by_location(df, max_chunks)
    if not chunks:
         logger.error("Data splitting resulted in no chunks. Cannot proceed.")
         return df.copy()

    # Store results - start with a copy of the original df
    results_df = df.copy()

    # Make predictions with each model - process chunks in parallel for each model
    for model_type, model in models.items():
        model_start_time = time.time()
        logger.info(f"--- Starting predictions for model: {model_type} ---")

        try:
            # Process spatial chunks in parallel for this model
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                for i, chunk_df in enumerate(chunks):
                    # Submit task to process one chunk (predicts on NumPy inside)
                    future = executor.submit(
                        process_spatial_chunk,
                        chunk_df, # Pass the chunk (which is already a copy)
                        model_type,
                        model,
                        feature_columns,
                        label_scaler,
                        input_width,
                        shift,
                        i # Chunk ID for logging
                    )
                    futures.append(future)

                # Collect all predictions and metadata from completed futures
                all_predictions = []
                all_metadata = []

                logger.info(f"Submitted {len(futures)} chunk processing tasks for {model_type}. Waiting for results...")
                # Process results as they complete with progress bar
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc=f"Predicting {model_type}"
                ):
                    try:
                        # Get results: predictions_list, metadata_list, location_map_dict
                        predictions, metadata, _ = future.result() # Ignore location map
                        if predictions and metadata:
                             all_predictions.extend(predictions)
                             all_metadata.extend(metadata)
                        # else: logger.debug("A chunk processing task returned no predictions/metadata.")
                    except concurrent.futures.process.BrokenProcessPool as bpe:
                         logger.error(f"CRITICAL: BrokenProcessPool error encountered for model {model_type}. This often indicates a worker process died unexpectedly (e.g., out of memory, segfault). Error: {bpe}")
                         # Decide how to handle: re-raise, try to continue, exit?
                         # For now, log and attempt to continue, but results will be incomplete.
                         break # Stop processing futures for this model
                    except Exception as e:
                        logger.error(f"Error processing a chunk future for model {model_type}: {e}")
                        # import traceback # Uncomment for detailed debugging
                        # logger.error(traceback.format_exc())

            # Check if we have any predictions for this model AFTER the loop
            if not all_predictions or not all_metadata:
                logger.warning(f"No predictions were generated or collected for model {model_type}. Skipping mapping.")
                continue

            logger.info(f"Collected {len(all_predictions)} total predictions for {model_type} across all chunks.")

            # Map predictions back to the original DataFrame structure
            mapped_series = map_predictions_to_df_improved(
                results_df, # Use results_df for correct index alignment
                all_predictions,
                all_metadata,
                model_type
            )

            # Add the mapped prediction Series to the results DataFrame
            results_df[mapped_series.name] = mapped_series

            # Log prediction statistics
            pred_col = mapped_series.name
            non_nan_count = results_df[pred_col].notna().sum()
            total_rows = len(results_df)
            coverage = (non_nan_count / total_rows * 100) if total_rows > 0 else 0
            logger.info(f"Added {non_nan_count} predictions for {model_type} ({coverage:.1f}% coverage)")

            if non_nan_count > 0:
                logger.info(f"{model_type} predictions range: [{results_df[pred_col].min():.4f}, {results_df[pred_col].max():.4f}]")
            model_end_time = time.time()
            logger.info(f"--- Finished predictions for model: {model_type} in {model_end_time - model_start_time:.2f} seconds ---")

        except Exception as e:
            logger.error(f"Critical error during parallel prediction processing for model {model_type}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Continue to the next model if possible

    # Create ensemble prediction if multiple models succeeded
    pred_cols = [col for col in results_df.columns if col.startswith('sap_velocity_') and col != 'sap_velocity_ensemble']
    if len(pred_cols) > 1:
        logger.info(f"Creating ensemble prediction (median) from {len(pred_cols)} models: {pred_cols}")
        results_df['sap_velocity_ensemble'] = results_df[pred_cols].median(axis=1)
        ensemble_count = results_df['sap_velocity_ensemble'].notna().sum()
        total_rows = len(results_df)
        coverage = (ensemble_count / total_rows * 100) if total_rows > 0 else 0
        logger.info(f"Ensemble prediction created with {ensemble_count} values ({coverage:.1f}% coverage)")
        if ensemble_count > 0:
             logger.info(f"Ensemble predictions range: [{results_df['sap_velocity_ensemble'].min():.4f}, {results_df['sap_velocity_ensemble'].max():.4f}]")
    elif len(pred_cols) == 1:
        logger.info(f"Only one model succeeded ({pred_cols[0]}). Using it as the 'ensemble'.")
        results_df['sap_velocity_ensemble'] = results_df[pred_cols[0]]
    else:
        logger.warning("No individual model predictions were generated. Cannot create ensemble.")

    parallel_end_time = time.time()
    logger.info(f"Parallel prediction process completed in {parallel_end_time - parallel_start_time:.2f} seconds.")
    return results_df


def make_predictions_improved(df, models, feature_columns, label_scaler, input_width=8, shift=1):
    """
    Make predictions with improved mapping approach (Sequential Version).
    MODIFIED: Creates NumPy windows, predicts directly on NumPy array.

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
    logger.info("Making predictions with improved mapping (SEQUENTIAL)...")
    sequential_start_time = time.time()

    # Validate inputs
    if df is None or df.empty: logger.error("Input DataFrame is empty"); return pd.DataFrame()
    if not models: logger.error("No models provided"); return df.copy()
    if not feature_columns: logger.error("No feature columns specified"); return df.copy()
    if label_scaler is None: logger.error("Label scaler required"); return df.copy()

    # Create windows (as NumPy array) and metadata sequentially
    windows_np, window_metadata, location_map, total_windows = create_prediction_windows_improved(
        df, feature_columns, input_width=input_width, shift=shift
    )

    if windows_np is None or not window_metadata:
        logger.error("Failed to create prediction windows (NumPy) sequentially.")
        return df.copy()

    # Store results
    results_df = df.copy()
    predict_batch_size = 1024 # Batch size for TF model predict on NumPy

    # Make predictions with each model sequentially
    for model_type, model in models.items():
        model_start_time = time.time()
        logger.info(f"--- Starting sequential predictions for model: {model_type} ---")

        try:
            # Handle different model types, predict on NumPy array
            is_deep_model = model_type.lower() in ['cnn_lstm', 'lstm', 'transformer', 'ann', 'gru']

            if is_deep_model and isinstance(model, tf.keras.Model):
                logger.debug(f"Predicting with DL model {model_type} on NumPy array...")
                preds_scaled = model.predict(windows_np, batch_size=predict_batch_size, verbose=0)
            elif hasattr(model, 'predict'):
                logger.debug(f"Predicting with ML model {model_type} on NumPy array...")
                X_input = windows_np.reshape(windows_np.shape[0], -1) if windows_np.ndim == 3 else windows_np
                preds_scaled = model.predict(X_input)
            else:
                logger.error(f"Model {model_type} lacks a predict method. Skipping.")
                continue

            # Standardize prediction shape
            if preds_scaled.ndim == 3: preds_scaled = preds_scaled[:, -1, :]
            if preds_scaled.ndim == 1: preds_scaled = preds_scaled.reshape(-1, 1)
            elif preds_scaled.ndim == 2 and preds_scaled.shape[1] > 1:
                 preds_scaled = preds_scaled[:, 0].reshape(-1, 1)
            elif preds_scaled.ndim > 2 or preds_scaled.ndim == 0: continue # Skip bad shapes

            if len(preds_scaled) != len(window_metadata):
                 logger.error(f"Mismatch preds/metadata for {model_type}. Skipping map.")
                 continue

            # Inverse transform
            preds_unscaled = label_scaler.inverse_transform(preds_scaled).flatten()

            # Map predictions back using the improved function
            mapped_series = map_predictions_to_df_improved(
                results_df, # Use results_df for index alignment
                preds_unscaled,
                window_metadata,
                model_type
            )

            # Add mapped predictions Series to results DataFrame
            results_df[mapped_series.name] = mapped_series

            # Log prediction statistics
            pred_col = mapped_series.name
            non_nan_count = results_df[pred_col].notna().sum()
            total_rows = len(results_df)
            coverage = (non_nan_count / total_rows * 100) if total_rows > 0 else 0
            logger.info(f"Added {non_nan_count} predictions for {model_type} ({coverage:.1f}% coverage)")
            if non_nan_count > 0:
                logger.info(f"{model_type} predictions range: [{results_df[pred_col].min():.4f}, {results_df[pred_col].max():.4f}]")
            model_end_time = time.time()
            logger.info(f"--- Finished sequential predictions for {model_type} in {model_end_time - model_start_time:.2f} sec ---")

        except Exception as e:
            logger.error(f"Error making sequential predictions with {model_type}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # Create ensemble prediction
    pred_cols = [col for col in results_df.columns if col.startswith('sap_velocity_') and col != 'sap_velocity_ensemble']
    if len(pred_cols) > 1:
        logger.info(f"Creating ensemble prediction (median) from {len(pred_cols)} models.")
        results_df['sap_velocity_ensemble'] = results_df[pred_cols].median(axis=1)
        ensemble_count = results_df['sap_velocity_ensemble'].notna().sum()
        total_rows = len(results_df)
        coverage = (ensemble_count / total_rows * 100) if total_rows > 0 else 0
        logger.info(f"Ensemble created with {ensemble_count} values ({coverage:.1f}% coverage)")
    elif len(pred_cols) == 1:
        results_df['sap_velocity_ensemble'] = results_df[pred_cols[0]]
    else:
        logger.warning("No predictions generated sequentially.")

    sequential_end_time = time.time()
    logger.info(f"Sequential prediction process completed in {sequential_end_time - sequential_start_time:.2f} seconds.")
    return results_df


def prepare_features_from_preprocessed(df, feature_scaler=None, input_width=8, label_width=1, shift=1):
    """
    Prepare features from preprocessed data with improved error handling and standardization.

    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed data
    feature_scaler : sklearn.preprocessing.StandardScaler, optional
        Fitted scaler for features
    input_width : int
        Number of time steps for input window (informational)
    label_width : int
        Number of time steps for output window (informational)
    shift : int
        Number of steps to shift for prediction (informational)

    Returns:
    --------
    tuple: (pandas.DataFrame, list)
        - Prepared DataFrame with features potentially scaled
        - List of actual feature column names used
    """
    try:
        logger.info("Preparing features from preprocessed data...")
        if df is None or df.empty:
            logger.error("Input DataFrame is None or empty.")
            return None, []

        # Create a copy to avoid modifying the original DataFrame
        df_processed = df.copy()

        # --- Temperature Conversion ---
        if 'ta' in df_processed.columns:
            try:
                temp_median = df_processed['ta'].median()
                if pd.notna(temp_median) and temp_median > 200: # Heuristic for Kelvin
                    logger.info(f"Column 'ta' median ({temp_median:.1f}) suggests Kelvin. Converting to Celsius.")
                    df_processed['ta'] = df_processed['ta'] - 273.15
                # else: logger.info(f"Column 'ta' median ({temp_median:.1f}) suggests Celsius. No conversion needed.")
            except Exception as e:
                logger.error(f"Error converting temperature 'ta': {e}")

        # --- Column Name Standardization ---
        rename_map = {
            'annual_mean_temperature': 'mean_annual_temp',
            'annual_precipitation': 'mean_annual_precip',
            'Day sin': 'day sin',
            'Week sin': 'week sin',
            'Month sin': 'month sin',
            'Year sin': 'year sin'
        }
        cols_to_rename = {k: v for k, v in rename_map.items() if k in df_processed.columns}
        if cols_to_rename:
            logger.info(f"Renaming columns: {cols_to_rename}")
            df_processed.rename(columns=cols_to_rename, inplace=True)

        # --- Define Feature Set ---
        # This list should ideally come from model training configuration
        # Using the list from user logs as the target feature set
        ordered_features = [
            'ext_rad', 'sw_in', 'ta', 'ws', 'vpd', 'ppfd_in',
            'mean_annual_temp', 'mean_annual_precip',
            'day sin', 'week sin', 'month sin', 'year sin'
        ]

        # Identify features available in the loaded data AFTER renaming
        available_features_in_df = df_processed.columns.tolist()
        final_feature_columns = [col for col in ordered_features if col in available_features_in_df]

        # Log missing features from the desired list
        missing_desired_features = [col for col in ordered_features if col not in final_feature_columns]
        if missing_desired_features:
            logger.warning(f"Desired features missing from data: {missing_desired_features}")

        # Check if the scaler expects specific features
        scaler_features = None
        if feature_scaler and hasattr(feature_scaler, 'feature_names_in_'):
            scaler_features = list(feature_scaler.feature_names_in_)
            logger.info(f"Feature scaler expects {len(scaler_features)} features: {scaler_features}")

            # Crucial Check: Do the final features match what the scaler expects?
            if set(final_feature_columns) != set(scaler_features):
                 logger.error(f"CRITICAL MISMATCH: Final features in data {final_feature_columns} "
                              f"do not match scaler's expected features {scaler_features}.")
                 # Option: Try using only scaler_features if they exist in df_processed? Risky.
                 # For safety, fail here.
                 return None, []
            else:
                 # Ensure the final list uses the scaler's order
                 final_feature_columns = scaler_features
                 logger.info("Data features match scaler's expected features and order.")
        else:
            # No scaler or scaler has no feature names: use features found in data
            logger.info("Using features found in data matching the ordered list (scaler feature names not available or no scaler).")


        # Check if we have any features left
        if not final_feature_columns:
            logger.error("No usable feature columns identified after filtering and checks.")
            return None, []

        logger.info(f"Final feature set ({len(final_feature_columns)} features): {final_feature_columns}")

        # --- Apply Scaling ---
        if feature_scaler is not None:
            logger.info("Applying feature scaling...")
            try:
                # Ensure only the required columns exist before scaling
                missing_cols_for_scaling = [col for col in final_feature_columns if col not in df_processed.columns]
                if missing_cols_for_scaling:
                    logger.error(f"Cannot apply scaling. Missing required columns: {missing_cols_for_scaling}")
                    return None, []

                # Select only the feature columns in the correct order for the scaler
                features_to_scale = df_processed[final_feature_columns]

                # Apply the transform
                scaled_values = feature_scaler.transform(features_to_scale)

                # Put scaled values back into the DataFrame
                df_processed[final_feature_columns] = scaled_values
                logger.info(f"Successfully scaled {len(final_feature_columns)} features.")

            except ValueError as ve:
                 logger.error(f"ValueError during feature scaling: {ve}. Check data/scaler column mismatch.")
                 logger.error(f"Data columns provided to scaler: {final_feature_columns}")
                 logger.error(f"First few rows of data provided:\n{features_to_scale.head()}")
                 if hasattr(feature_scaler, 'n_features_in_'): logger.error(f"Scaler expected {feature_scaler.n_features_in_} features.")
                 if hasattr(feature_scaler, 'feature_names_in_'): logger.error(f"Scaler feature names: {list(feature_scaler.feature_names_in_)}")
                 return None, [] # Return error state
            except Exception as e:
                logger.error(f"Unexpected error applying feature scaling: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None, [] # Return error state
        else:
            logger.info("No feature scaler provided. Using unscaled features.")

        # Return the processed DataFrame and the list of features used
        return df_processed, final_feature_columns

    except Exception as e:
        logger.error(f"Error in prepare_features_from_preprocessed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, []


def validate_predictions(df_predictions, feature_columns):
    """
    Validate predictions to detect potential mapping issues or unreasonable values.

    Parameters:
    -----------
    df_predictions : pandas.DataFrame
        DataFrame with predictions
    feature_columns : list
        List of feature columns used

    Returns:
    --------
    bool
        True if basic validations pass, False otherwise
    """
    logger.info("Validating predictions...")
    validation_passed = True

    # Get prediction columns
    pred_cols = [col for col in df_predictions.columns if col.startswith('sap_velocity_')]
    if not pred_cols:
        logger.warning("No prediction columns found to validate.")
        return True # No predictions to fail validation

    # 1. Check for NaN/Inf values
    for col in pred_cols:
        if df_predictions[col].isnull().all():
            logger.error(f"Validation FAIL: Prediction column '{col}' is entirely NaN.")
            validation_passed = False
        elif np.isinf(df_predictions[col]).any():
            inf_count = np.isinf(df_predictions[col]).sum()
            logger.warning(f"Validation WARN: Prediction column '{col}' contains {inf_count} Inf values.")
            # Decide if this is a failure or just a warning
            # validation_passed = False

    # 2. Check for reasonable value ranges for sap velocity (adjust as needed)
    min_reasonable = -5
    max_reasonable = 50
    for col in pred_cols:
        pred_data = df_predictions[col].dropna()
        # Filter out inf values before checking min/max
        pred_data = pred_data[~np.isinf(pred_data)]
        if len(pred_data) > 0:
            min_val, max_val = pred_data.min(), pred_data.max()
            if min_val < min_reasonable or max_val > max_reasonable:
                logger.warning(f"Validation WARN: Column '{col}' has values outside expected range "
                               f"[{min_reasonable}, {max_reasonable}]: Found [{min_val:.2f}, {max_val:.2f}]")
                # validation_passed = False

    # 3. Check temporal consistency if time index exists
    if isinstance(df_predictions.index, pd.DatetimeIndex):
        logger.info("Checking temporal consistency (requires DatetimeIndex)...")
        df_sorted = df_predictions.sort_index()
        for col in pred_cols:
            pred_series = df_sorted[col].dropna()
            pred_series = pred_series[~np.isinf(pred_series)] # Exclude inf
            if len(pred_series) > 1:
                abs_diff = pred_series.diff().abs()
                max_diff = abs_diff.max()
                max_diff_threshold = 20 # Example threshold
                if pd.notna(max_diff) and max_diff > max_diff_threshold:
                    logger.warning(f"Validation WARN: Column '{col}' shows potentially abrupt changes "
                                   f"(max difference between consecutive points: {max_diff:.2f})")
                    # validation_passed = False

    # 4. Check correlation with key driving environmental variables (if present)
    key_features = [f for f in ['ta', 'vpd', 'ppfd_in', 'sw_in'] if f in df_predictions.columns]
    if key_features:
        logger.info(f"Checking correlation with key features: {key_features}...")
        for feat in key_features:
            for col in pred_cols:
                # Calculate correlation on valid (non-NaN, non-Inf) pairs
                temp_df = df_predictions[[feat, col]].copy()
                temp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                valid_data = temp_df.dropna()

                if len(valid_data) > 10: # Need sufficient points
                    try:
                        corr = valid_data.corr().iloc[0, 1]
                        # Basic checks for expected directionality (adjust as needed)
                        expected_positive = True
                        if feat == 'ws': expected_positive = False

                        if expected_positive and corr < -0.1:
                             logger.warning(f"Validation WARN: Column '{col}' has unexpected negative correlation with {feat}: {corr:.3f}")
                        elif not expected_positive and corr > 0.1:
                             logger.warning(f"Validation WARN: Column '{col}' has unexpected positive correlation with {feat}: {corr:.3f}")
                        elif abs(corr) < 0.05:
                             logger.info(f"Validation INFO: Column '{col}' shows very weak correlation with {feat}: {corr:.3f}")

                    except Exception as e:
                        logger.warning(f"Could not calculate correlation between {col} and {feat}: {e}")

    if validation_passed:
        logger.info("Basic prediction validations passed.")
    else:
        logger.error("One or more prediction validations failed or raised warnings. Review logs.")

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
        Dictionary of loaded models {type: model_object}
    """
    if not PLOT_AVAILABLE:
        logger.warning("Matplotlib/Seaborn not installed. Skipping visualization.")
        return

    logger.info("Creating visualizations...")
    plt.style.use('seaborn-v0_8-whitegrid')

    output_dir.mkdir(exist_ok=True, parents=True)
    df_plot = df_predictions.copy()

    # Get prediction columns
    pred_cols = [col for col in df_plot.columns if col.startswith('sap_velocity_')]
    ensemble_col = 'sap_velocity_ensemble' if 'sap_velocity_ensemble' in df_plot.columns else None
    individual_pred_cols = [col for col in pred_cols if col != ensemble_col]

    if not pred_cols:
        logger.warning("No prediction columns found for visualization.")
        return

    # --- Time Axis Setup ---
    time_axis = None
    time_col_name = 'index' # Default if no time column found
    if isinstance(df_plot.index, pd.DatetimeIndex):
        time_axis = df_plot.index
        time_col_name = df_plot.index.name or 'datetime'
        logger.info(f"Using DatetimeIndex ('{time_col_name}') for plotting.")
    else:
        potential_time_cols = [c for c in df_plot.columns if 'time' in c.lower() or 'date' in c.lower()]
        for col in potential_time_cols:
            try:
                time_series = pd.to_datetime(df_plot[col], errors='coerce')
                if time_series.notna().any():
                     df_plot[col] = time_series # Update column
                     time_axis = df_plot[col]
                     time_col_name = col
                     logger.info(f"Using column '{time_col_name}' as datetime for plotting.")
                     break
            except Exception: continue

        if time_axis is None:
            logger.warning("No suitable time column found or convertible. Using row number for x-axis.")
            time_axis = np.arange(len(df_plot))
            time_col_name = 'row_number'

    # --- Location Identification ---
    location_col_name = None
    unique_locations = ['single_sequence']
    loc_str_map = {'single_sequence': 'All_Data'}
    location_col_id = None # Column used for filtering plots

    if 'name' in df_plot.columns:
        location_col_name = 'name'
        location_col_id = 'name'
        unique_locations = sorted(df_plot[location_col_name].unique())
        loc_str_map = {loc: str(loc) for loc in unique_locations}
        logger.info(f"Found {len(unique_locations)} locations using 'name' column.")
    elif 'latitude' in df_plot.columns and 'longitude' in df_plot.columns:
        location_col_name = ['latitude', 'longitude']
        df_plot['latlon_id'] = df_plot.apply(lambda row: f"Lat{row['latitude']:.2f}_Lon{row['longitude']:.2f}", axis=1)
        location_col_id = 'latlon_id'
        unique_locations = sorted(df_plot[location_col_id].unique())
        loc_str_map = {loc_id: loc_id for loc_id in unique_locations}
        logger.info(f"Found {len(unique_locations)} locations using lat/lon coordinates.")
    else:
        logger.info("No location columns found. Plotting data as a single sequence.")

    # --- Plotting Loop ---
    max_plots = 10
    locations_to_plot = unique_locations
    if len(unique_locations) > max_plots:
        logger.warning(f"Plotting only the first {max_plots} locations.")
        locations_to_plot = unique_locations[:max_plots]

    logger.info(f"Generating time series plots for {len(locations_to_plot)} locations...")
    for i, location_id in enumerate(tqdm(locations_to_plot, desc="Generating plots")):
        try:
            # Extract data for this location
            if location_col_id:
                location_data = df_plot[df_plot[location_col_id] == location_id].copy()
            else:
                location_data = df_plot.copy()

            loc_str = loc_str_map[location_id]

            if location_data.empty: continue

            # Get the corresponding time axis for this subset
            current_time_axis = time_axis[location_data.index] if isinstance(time_axis, (pd.Series, pd.DatetimeIndex)) else np.arange(len(location_data))

            # Sort data by time axis for correct line plotting
            try:
                 sort_order = np.argsort(current_time_axis)
                 location_data = location_data.iloc[sort_order]
                 current_time_axis = current_time_axis[sort_order]
            except TypeError as te:
                 logger.warning(f"Could not sort time axis for plot {loc_str} (TypeError: {te}). Plot may be unordered.")
            except IndexError as ie:
                 logger.warning(f"Could not sort time axis for plot {loc_str} (IndexError: {ie}). Plot may be unordered.")


            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1]})
            fig.suptitle(f'Sap Velocity Predictions - {loc_str}', fontsize=16)

            # --- Plot 1: Predictions ---
            ax1 = axes[0]
            palette = sns.color_palette("tab10", len(individual_pred_cols))

            for j, col in enumerate(individual_pred_cols):
                if col in location_data.columns:
                    model_data = location_data[col].dropna()
                    if not model_data.empty:
                        time_axis_subset = current_time_axis[model_data.index]
                        model_type = col.replace('sap_velocity_', '')
                        ax1.plot(time_axis_subset, model_data,
                               label=f'{model_type.upper()}', color=palette[j],
                               alpha=0.7, linewidth=1.5)

            if ensemble_col and ensemble_col in location_data.columns:
                ensemble_data = location_data[ensemble_col].dropna()
                if not ensemble_data.empty:
                     time_axis_subset = current_time_axis[ensemble_data.index]
                     ax1.plot(time_axis_subset, ensemble_data,
                           label='Ensemble', color='black', linewidth=2.5, zorder=10)

            ax1.set_ylabel('Sap Velocity')
            ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
            ax1.grid(True, which='major', linestyle='--', alpha=0.6)

            # --- Plot 2: Key Features ---
            ax2 = axes[1]
            key_features = ['ta', 'vpd', 'ppfd_in', 'sw_in']
            plot_features = [f for f in key_features if f in location_data.columns]

            if plot_features:
                feature_palette = sns.color_palette("Greys", len(plot_features) + 2)
                norm_data = location_data[plot_features].copy()
                for col in plot_features:
                    series = norm_data[col].dropna()
                    min_val, max_val = series.min(), series.max()
                    if pd.notna(min_val) and pd.notna(max_val) and max_val > min_val:
                        norm_data[col] = (series - min_val) / (max_val - min_val)
                    elif pd.notna(min_val): norm_data[col] = 0.5
                    else: norm_data[col] = np.nan

                for j, col in enumerate(plot_features):
                    feature_series = norm_data[col].dropna()
                    if not feature_series.empty:
                         time_axis_subset = current_time_axis[feature_series.index]
                         ax2.plot(time_axis_subset, feature_series,
                                label=col, linewidth=1.0, color=feature_palette[j+1], alpha=0.8)

                ax2.set_ylabel('Normalized Value')
                ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
                ax2.grid(True, which='major', linestyle=':', alpha=0.5)
            else:
                ax2.set_visible(False)

            # Set shared x-axis label and format ticks if datetime
            axes[-1].set_xlabel(time_col_name.replace('_', ' ').title())
            if isinstance(current_time_axis, (pd.DatetimeIndex, pd.Series)) and pd.api.types.is_datetime64_any_dtype(current_time_axis):
                try:
                     # Improve date formatting
                     import matplotlib.dates as mdates
                     axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
                     plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha='right')
                except ImportError:
                     plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha='right')


            # Save the figure
            plt.tight_layout(rect=[0, 0, 0.9, 0.96])
            safe_loc_str = loc_str.replace(' ', '_').replace('=', '').replace('.', 'p')
            safe_loc_str = "".join(c for c in safe_loc_str if c.isalnum() or c in ('_', '-')).rstrip()
            filename = f'predictions_{safe_loc_str}.png'
            filepath = output_dir / filename

            plt.savefig(filepath, bbox_inches='tight', dpi=150)
            plt.close(fig)

        except Exception as e:
            logger.error(f"Error creating plot for location {location_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            plt.close('all') # Close any potentially open figures on error

    # --- Feature Importance Plots ---
    if include_feature_importance and models is not None and feature_columns:
        logger.info("Generating feature importance plots...")
        if not feature_columns:
             logger.warning("Cannot generate feature importance: feature_columns list is empty.")
             return

        for model_type, model in models.items():
            try:
                importances = None
                importance_type = None

                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    importance_type = 'feature_importances_'
                elif hasattr(model, 'coef_'):
                    if model.coef_.ndim == 1:
                        importances = np.abs(model.coef_)
                        importance_type = 'abs(coef_)'
                    elif model.coef_.ndim == 2:
                        importances = np.mean(np.abs(model.coef_), axis=0)
                        importance_type = 'mean(abs(coef_))'

                if importances is not None:
                    # Assume importance corresponds directly to feature_columns
                    if len(importances) == len(feature_columns):
                        imp_df = pd.DataFrame({'Feature': feature_columns, 'Importance': importances})
                        imp_df = imp_df.sort_values('Importance', ascending=False).reset_index(drop=True)
                        top_n = 20
                        imp_df = imp_df.head(top_n)

                        plt.figure(figsize=(10, max(6, len(imp_df) * 0.3)))
                        ax = sns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis', orient='h')
                        plt.title(f'Feature Importance ({importance_type}) - {model_type.upper()}')
                        plt.xlabel('Importance Score')
                        plt.ylabel('Feature')
                        plt.tight_layout()
                        for i, v in enumerate(imp_df['Importance']):
                             ax.text(v * 1.01, i, f'{v:.3f}', color='black', va='center')

                        filepath = output_dir / f'feature_importance_{model_type}.png'
                        plt.savefig(filepath, bbox_inches='tight', dpi=150)
                        logger.info(f"Saved feature importance plot: {filepath}")
                        plt.close()
                        imp_df.to_csv(output_dir / f'feature_importance_{model_type}.csv', index=False)
                    else:
                        logger.warning(f"Feature importance dimension mismatch for {model_type}: "
                                       f"{len(importances)} scores vs {len(feature_columns)} features. Plot skipped.")
                # else: logger.info(f"Feature importance not directly available for model type: {model_type}")

            except Exception as e:
                logger.error(f"Error creating feature importance plot for {model_type}: {e}")
                plt.close('all')

    logger.info(f"Visualization generation complete. Results saved to {output_dir}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Predict sap velocity from preprocessed data using optimized parallel processing.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file with preprocessed data.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory. Defaults to ./outputs/predictions/<input_file_stem>/')
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
                        help='Time steps in the label window (informational only for prediction).')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations of predictions.')
    parser.add_argument('--feature-importance', action='store_true',
                        help='Generate feature importance plots for applicable models.')
    parser.add_argument('--validate', action='store_true',
                        help='Perform basic validation checks on predictions.')

    # --- Modified Parallelism Flags ---
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--parallel', action='store_true', default=False, # Default to parallel prediction by chunk
                        help='Use parallel processing for prediction across spatial chunks (Default).')
    group.add_argument('--parallel-windows', action='store_true', default=False,
                        help='Use parallel processing ONLY for window creation, then predict sequentially.')
    group.add_argument('--sequential', action='store_true', default=False,
                       help='Run both window creation and prediction sequentially.')

    parser.add_argument('--n-workers', type=int, default=None,
                        help='Number of worker processes for parallel execution (used by --parallel and --parallel-windows). '
                             'Defaults to CPU count - 2 if not specified.')

    return parser.parse_args()


def main():
    """Main execution function"""
    script_start_time = time.time()
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
        output_base = Path('./outputs/predictions')
        output_dir = output_base / input_file.stem
    output_dir.mkdir(exist_ok=True, parents=True)
    # Update log file handler path
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)
    log_filepath = output_dir / log_file_name # Place log in output dir
    file_handler = logging.FileHandler(log_filepath)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Log file: {log_filepath}")


    logger.info(f"Output will be saved to: {output_dir}")

    # Configure parallel processing workers
    import multiprocessing
    if args.n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 2)
    else:
        n_workers = max(1, args.n_workers)
    available_cores = multiprocessing.cpu_count()
    if n_workers >= available_cores:
         logger.warning(f"Requested workers ({n_workers}) >= available cores ({available_cores}). Reducing to {available_cores - 1}.")
         n_workers = max(1, available_cores - 1)
    logger.info(f"Maximum worker processes set to: {n_workers}")


    # Log setup information
    logger.info("="*60)
    logger.info(" Sap Velocity Prediction Script (Optimized V3) ")
    logger.info("="*60)
    logger.info(f"Input File: {input_file}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Models Directory: {models_dir}")
    logger.info(f"Scaler Directory: {scaler_dir}")
    logger.info(f"Model Types: {args.model_types}")
    logger.info(f"Window Params: width={args.input_width}, shift={args.shift}, label_width={args.label_width}")
    # Determine execution mode based on flags
    if args.parallel_windows:
         exec_mode = "Parallel Windows Creation, Sequential Prediction"
    elif args.sequential:
         exec_mode = "Fully Sequential"
         args.parallel = False # Ensure parallel flag is off if sequential is chosen
    else: # Default is parallel by chunk
         exec_mode = "Parallel Prediction by Chunk"
         args.parallel = True # Ensure parallel flag is on if it's the default/chosen mode

    logger.info(f"Execution Mode: {exec_mode}")
    if args.parallel or args.parallel_windows:
        logger.info(f"Number of Workers: {n_workers}")
    logger.info(f"Visualize: {args.visualize}")
    logger.info(f"Feature Importance: {args.feature_importance}")
    logger.info(f"Validate Predictions: {args.validate}")
    logger.info("-"*60)


    # --- Load Models ---
    logger.info("Loading models...")
    models = load_models(models_dir, args.model_types)
    if not models:
        logger.error("No models were loaded. Exiting.")
        sys.exit(1)

    # --- Load Scalers ---
    logger.info("Loading scalers...")
    feature_scaler, label_scaler = load_scalers(scaler_dir)
    if label_scaler is None:
        logger.error("Label scaler is required but not found or failed to load. Exiting.")
        sys.exit(1)
    if feature_scaler is None:
         logger.warning("Feature scaler not found or failed to load. Proceeding without feature scaling.")

    # --- Load and Prepare Data ---
    logger.info("Loading and preparing input data...")
    load_start = time.time()
    df_raw = load_preprocessed_data(input_file)
    if df_raw is None or df_raw.empty:
        logger.error("Failed to load or preprocess input data. Exiting.")
        sys.exit(1)
    load_end = time.time()
    logger.info(f"Data loading finished in {load_end - load_start:.2f} seconds.")

    # Store original index before any potential resets
    original_index = df_raw.index.copy()
    logger.info(f"Original index type: {type(original_index)}")


    logger.info("Preparing features...")
    prep_start = time.time()
    df_prepared, feature_columns = prepare_features_from_preprocessed(
        df_raw,
        feature_scaler, # Pass the loaded feature scaler (can be None)
        input_width=args.input_width,
        label_width=args.label_width,
        shift=args.shift
    )
    prep_end = time.time()

    if df_prepared is None or not feature_columns:
        logger.error("Failed to prepare features. Exiting.")
        sys.exit(1)
    logger.info(f"Feature preparation finished in {prep_end - prep_start:.2f} seconds.")
    logger.info(f"Using {len(feature_columns)} features for prediction: {feature_columns}")


    # --- Make Predictions ---
    logger.info("Starting prediction phase...")
    pred_start = time.time()
    df_predictions = pd.DataFrame() # Initialize empty DataFrame

    # --- Execution Logic based on Flags ---
    if args.parallel:
        # Default: Parallel prediction by chunk
        df_predictions = make_predictions_parallel(
            df_prepared, models, feature_columns, label_scaler,
            input_width=args.input_width, shift=args.shift, n_workers=n_workers
        )
    elif args.parallel_windows:
        # Mode: Parallel windows creation, sequential prediction
        logger.info("--- Mode: Parallel Window Creation ---")
        logger.warning("This mode creates all windows upfront and may require significant memory.")
        windows_np, window_metadata, _, total_windows = create_prediction_windows_parallel(
            df_prepared, feature_columns,
            input_width=args.input_width, shift=args.shift, n_workers=n_workers
        )

        if windows_np is None or not window_metadata:
            logger.error("Parallel window creation failed. Exiting.")
            sys.exit(1)

        logger.info(f"Successfully created {total_windows} windows in parallel.")
        logger.info(f"Window array shape: {windows_np.shape}, Approx Size: {windows_np.nbytes / (1024**2):.2f} MB")

        logger.info("--- Mode: Sequential Prediction ---")
        results_df = df_prepared.copy() # Start with prepared data for mapping
        predict_batch_size = 4096 # Batch size for TF predict

        for model_type, model in models.items():
            model_pred_start = time.time()
            logger.info(f"--- Predicting sequentially with {model_type} ---")
            try:
                is_deep_model = model_type.lower() in ['cnn_lstm', 'lstm', 'transformer', 'ann', 'gru']
                if is_deep_model and isinstance(model, tf.keras.Model):
                    preds_scaled = model.predict(windows_np, batch_size=predict_batch_size, verbose=1) # Show progress for sequential
                elif hasattr(model, 'predict'):
                    X_input = windows_np.reshape(windows_np.shape[0], -1) if windows_np.ndim == 3 else windows_np
                    preds_scaled = model.predict(X_input)
                else: continue

                # Standardize shape & inverse transform
                if preds_scaled.ndim == 3: preds_scaled = preds_scaled[:, -1, :]
                if preds_scaled.ndim == 1: preds_scaled = preds_scaled.reshape(-1, 1)
                elif preds_scaled.ndim == 2 and preds_scaled.shape[1] > 1: preds_scaled = preds_scaled[:, 0].reshape(-1, 1)
                elif preds_scaled.ndim > 2 or preds_scaled.ndim == 0: continue

                if len(preds_scaled) != len(window_metadata): continue

                preds_unscaled = label_scaler.inverse_transform(preds_scaled).flatten()

                # Map back
                mapped_series = map_predictions_to_df_improved(results_df, preds_unscaled, window_metadata, model_type)
                results_df[mapped_series.name] = mapped_series
                non_nan_count = results_df[mapped_series.name].notna().sum()
                logger.info(f"Mapped {non_nan_count} predictions for {model_type}.")
                model_pred_end = time.time()
                logger.info(f"--- Finished {model_type} in {model_pred_end - model_pred_start:.2f} sec ---")

            except Exception as e: logger.error(f"Error predicting with {model_type}: {e}", exc_info=True)

        # Create ensemble
        pred_cols = [c for c in results_df.columns if c.startswith('sap_velocity_') and c != 'sap_velocity_ensemble']
        if len(pred_cols) > 1: results_df['sap_velocity_ensemble'] = results_df[pred_cols].median(axis=1)
        elif len(pred_cols) == 1: results_df['sap_velocity_ensemble'] = results_df[pred_cols[0]]

        df_predictions = results_df # Assign final result

    else: # Fully sequential mode (--sequential flag)
        logger.info("--- Mode: Sequential Window Creation & Prediction ---")
        df_predictions = make_predictions_improved(
            df_prepared, models, feature_columns, label_scaler,
            input_width=args.input_width, shift=args.shift
        )

    pred_end = time.time()
    logger.info(f"Prediction phase finished in {pred_end - pred_start:.2f} seconds.")

    if df_predictions.empty:
        logger.error("Prediction failed or returned an empty DataFrame. Exiting.")
        sys.exit(1)

    # --- Final Steps ---
    # Restore original index if lengths match
    df_final = df_predictions # Default to predictions df
    if len(original_index) == len(df_predictions):
        try:
            if df_predictions.index.equals(original_index):
                 logger.info("Prediction DataFrame index already matches original index.")
                 df_final = df_predictions
            else:
                 logger.info("Attempting to restore original index...")
                 df_final = df_predictions.set_index(original_index)
                 logger.info("Successfully restored original index to the final DataFrame.")
        except Exception as e:
            logger.warning(f"Could not restore original index (Error: {e}). Using the current index of the predictions DataFrame.")
            df_final = df_predictions
    else:
        logger.warning(f"Length mismatch between original data ({len(original_index)}) and "
                      f"predictions DataFrame ({len(df_predictions)}). Cannot restore original index reliably. Using current index.")
        df_final = df_predictions

    # Validate predictions
    if args.validate:
        logger.info("Performing final prediction validation...")
        validation_passed = validate_predictions(df_final, feature_columns)
        logger.info(f"Prediction validation {'PASSED' if validation_passed else 'FAILED or raised warnings'}")

    # Save predictions
    output_file = output_dir / f"{input_file.stem}_predictions.csv" # Simpler name
    logger.info(f"Saving final predictions to {output_file}...")
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(output_file, index=True) # Save with index
        logger.info(f"Successfully saved predictions.")
    except Exception as e:
        logger.error(f"Error saving predictions to CSV: {e}")
        logger.error(traceback.format_exc())


    # Create visualizations
    if args.visualize:
        if not PLOT_AVAILABLE:
             logger.warning("Visualization requested but Matplotlib/Seaborn not found. Skipping.")
        else:
             logger.info("Generating visualizations...")
             vis_start = time.time()
             visualize_predictions(
                 df_final,
                 output_dir,
                 feature_columns,
                 include_feature_importance=args.feature_importance,
                 models=models # Pass loaded models for importance plots
             )
             vis_end = time.time()
             logger.info(f"Visualization finished in {vis_end - vis_start:.2f} seconds.")

    # --- Finish ---
    script_end_time = time.time()
    execution_time = script_end_time - script_start_time
    logger.info("="*60)
    logger.info(" Prediction Script Completed Successfully ")
    logger.info(f" Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    logger.info(f" Predictions saved in: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    # Optional: Set TensorFlow GPU memory growth if using GPU and experiencing OOM errors
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         logger.info(f"Enabled memory growth for {len(gpus)} GPU(s)")
    #     except RuntimeError as e:
    #         logger.error(f"Could not set memory growth: {e}")

    try:
        main()
    except SystemExit as se:
        if se.code != 0: logger.critical(f"Script exited with error code: {se.code}")
        else: logger.info("Script exited normally.")
        sys.exit(se.code)
    except Exception as e:
        logger.critical(f"Unhandled critical error in main execution: {e}", exc_info=True)
        sys.exit(1)
