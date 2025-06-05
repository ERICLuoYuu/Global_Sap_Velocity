#!/usr/bin/env python
"""
Simplified Sap Velocity Prediction Script

This script works directly with preprocessed ERA5-Land data,
organizes it into sequences, and generates predictions using a specified feature order.
Biome features have been removed.
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
    # Note: BIOME_TYPES is no longer expected or used from config
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
    # BIOME_TYPES list removed


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
        models_dir = Path(models_dir) # Ensure it's a Path object

    if model_types is None:
        # Default list if none provided
        model_types = ['ann', 'lstm', 'transformer', 'cnn_lstm', 'rf', 'svr', 'xgb']
        logger.info(f"No model types specified, attempting to load default types: {model_types}")

    loaded_models = {}

    for model_type in model_types:
        model_path = None
        # Define model directory based on convention (e.g., 'xgb_regression')
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
            # Deep learning models use .keras format or SavedModel directory
            model_files = list(type_dir.glob('*.keras'))
            if model_files:
                # Use the most recent .keras file
                 model_path = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            else:
                 # Check for saved model directories (TensorFlow format)
                 model_dirs = [d for d in type_dir.iterdir() if d.is_dir() and (d / 'saved_model.pb').exists()]
                 if model_dirs:
                     model_path = sorted(model_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                 else:
                    logger.warning(f"No .keras files or SavedModel directories found for {model_type} in {type_dir}")
                    continue # Skip to next model type if no file/dir found

            # Load the model
            if model_path:
                try:
                    logger.info(f"Loading {model_type} model from {model_path}")
                    # Suppress TensorFlow warnings during model loading if desired
                    # tf.get_logger().setLevel('ERROR')
                    model = tf.keras.models.load_model(model_path)
                    # tf.get_logger().setLevel('INFO') # Restore logging level
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
        scaler_dir = Path(scaler_dir) # Ensure it's a Path object

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
        
        # Read header and first data row to infer timestamp column
        with open(data_file_path, 'r') as f:
            header = f.readline().strip().split(',')
            # Handle potential empty lines or files
            try:
                first_row_line = f.readline()
                if not first_row_line:
                     logger.error(f"Data file {data_file_path} appears to be empty or only has a header.")
                     return None
                first_row = first_row_line.strip().split(',')
            except StopIteration:
                 logger.error(f"Could not read the first data row from {data_file_path}.")
                 return None
       
        # Load the data using pandas
        df = pd.read_csv(data_file_path).dropna()
        # Set index if a timestamp column was identified and exists
        if 'timestamp.1' in df.columns:
            try:
                df['timestamp.1'] = pd.to_datetime(df['timestamp.1'])
                df.set_index('timestamp.1', inplace=True)
                logger.info(f"Successfully set '{'timestamp.1'}' as DatetimeIndex.")
            except Exception as e:
                logger.warning(f"Could not convert column 'timestamp.1' to datetime or set as index: {e}. Proceeding without DatetimeIndex.")
        elif 'timestamp.1':
             logger.warning(f"Identified potential timestamp column '{'timestamp.1'}', but it wasn't found in the loaded DataFrame columns: {df.columns.tolist()}")
        else:
             logger.warning("No timestamp column identified or used. Ensure time ordering if necessary.")


        logger.info(f"Loaded data with {len(df)} rows and {len(df.columns)} columns: {df.columns.tolist()}")
        # Basic data validation
        if df.empty:
            logger.error("Loaded DataFrame is empty.")
            return None
        if df.isnull().all().all():
             logger.error("Loaded DataFrame contains only NaN values.")
             return None

        return df

    except Exception as e:
        logger.error(f"Error loading preprocessed data from {data_file}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def map_windowed_predictions_to_df(df_shape, num_predictions, location_indices_map, input_width=8, shift=1):
    """
    Maps predictions made on windows back to the original DataFrame structure.

    Parameters:
    -----------
    df_shape : tuple
        Shape of the original DataFrame (used for index size).
    num_predictions : int
        Total number of predictions generated (should match number of windows).
    location_indices_map : dict
        Dictionary mapping location identifier (name or lat/lon tuple) to
        a sorted list of original DataFrame row indices for that location.
    input_width : int
        The width of the input window used for predictions.
    shift : int
        The prediction shift (how many steps ahead the prediction is for).

    Returns:
    --------
    np.array
        An array of the same length as the original DataFrame, containing
        prediction indices mapped to their corresponding future time steps, or NaN
        where no prediction applies.
    """
    # Initialize result array with NaNs
    result = np.full(df_shape[0], np.nan)
    pred_idx = 0  # Index for the predictions array
    mapped_count = 0

    logger.info(f"Mapping {num_predictions} predictions back to DataFrame shape {df_shape} using {len(location_indices_map)} locations.")
    logger.info(f"Window parameters: input_width={input_width}, shift={shift}")

    # Iterate through locations in the same sorted order used for window creation
    # Check if location_indices_map is not empty before sorting
    if not location_indices_map:
        logger.warning("Location indices map is empty. Cannot map predictions.")
        return result # Return array of NaNs

    sorted_locations = sorted(location_indices_map.keys())

    for location in sorted_locations:
        loc_indices = location_indices_map[location]
        num_loc_points = len(loc_indices)

        # Calculate the number of windows generated for this location
        num_windows_for_loc = max(0, num_loc_points - input_width + 1)

        # Map predictions for each window of this location
        for i in range(num_windows_for_loc):
            if pred_idx >= num_predictions:
                logger.warning(f"Reached end of predictions ({pred_idx}/{num_predictions}) while processing location {location}, window {i}. Check window creation consistency.")
                break # Stop if we run out of predictions

            # Calculate the index in the original DataFrame where this prediction belongs
            # The prediction corresponds to 'shift' steps after the *end* of the input window.
            # The end of the i-th window is at position i + input_width - 1 within the location's data.
            target_loc_pos = i + input_width - 1 + shift

            # Check if this target position is within the bounds of the location's data
            if target_loc_pos < num_loc_points:
                # Get the actual DataFrame row index corresponding to this target position
                row_idx = loc_indices[target_loc_pos]

                # Find the original position of row_idx in the full DataFrame index
                # This step is crucial if df_shape[0] represents the original unsorted/unfiltered DataFrame
                # However, if df_shape[0] comes from the potentially filtered/sorted df used in prepare_features,
                # then row_idx might be directly usable if it's an integer position.
                # Assuming row_idx is the original index value (could be integer or datetime):
                # We need the integer position of this index in the *original* DataFrame shape.
                # This mapping logic assumes the indices in loc_indices are the original index *values*.
                # Let's refine this: the result array corresponds to the original df's integer positions.
                # We need to map the original index value `row_idx` back to its integer position.
                # This is complex without the original index.
                # SAFER APPROACH: Assume df_shape is from the DataFrame passed to prepare_features,
                # and loc_indices contains integer positions *within that DataFrame*.
                # Let's adjust the function signature or assume df_shape matches the input df to prepare_features.

                # Assuming result array matches the DataFrame structure used for windowing:
                # Find the integer position corresponding to the original index `row_idx`
                # This requires having the index object available.
                # Let's simplify: Assume `result` has the same index as the input `df` to `prepare_features`.
                # Then `row_idx` (which is an index value) can be used directly with .loc if needed,
                # but `result` is a numpy array, so we need integer positions.

                # REVISED LOGIC: Pass the actual index object to this function or return a Series.
                # Sticking with numpy array: Assume df_shape[0] is the number of rows in the *processed* df.
                # And assume loc_indices contains the *integer positions* within that processed df.
                # This seems consistent with how create_prediction_windows works if df is indexed 0..N-1.

                # If loc_indices contains the *original* index values, we need a map back to 0..N-1.
                # Let's assume loc_indices contains integer positions relative to the start of the *processed* df.

                # The index `row_idx` here IS the integer position within the location's subset of data.
                # We need the position in the *overall* processed DataFrame.
                # The `location_indices_map` should store these overall integer positions.
                # Let's assume `location_indices_map` stores integer positions from the DataFrame passed to `create_prediction_windows`.

                # Check if row_idx (which is an integer position from loc_indices) is valid
                if 0 <= row_idx < df_shape[0]:
                     result[row_idx] = pred_idx # Store prediction index at the correct integer position
                     mapped_count += 1
                else:
                     logger.warning(f"Calculated row index {row_idx} is out of bounds for DataFrame shape {df_shape}. Skipping mapping for prediction {pred_idx}.")

            # else: prediction is for a time step beyond the available data for this location

            pred_idx += 1 # Move to the next prediction

    logger.info(f"Finished mapping window indices. {mapped_count} potential prediction slots identified.")
    if pred_idx != num_predictions:
         # This warning is common if the last few points don't form a full window + shift
         logger.info(f"Final prediction index ({pred_idx}) does not match total number of predictions ({num_predictions}). This may be normal if predictions extend beyond data.")

    return result # Return the array with prediction indices (or NaN)


def create_prediction_windows(df, feature_columns, input_width=8, batch_size=32):
    """
    Create time series windows for prediction with consistent location ordering.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with time series data, potentially including 'name' or 'latitude'/'longitude'.
        Must contain the specified feature_columns. Index should be unique.
    feature_columns : list
        List of feature column names to include in the windows.
    input_width : int
        Window size for input sequence.
    batch_size : int
        Batch size for the TensorFlow dataset.

    Returns:
    --------
    tuple: (tf.data.Dataset, dict, int)
        - Dataset with windowed features ready for prediction.
        - Dictionary mapping location identifier to sorted *integer position* indices.
        - Total number of windows created.
        Returns (None, {}, 0) if window creation fails.
    """
    import tensorflow as tf
    import numpy as np
    import pandas as pd

    logger.info(f"Creating prediction windows with input_width={input_width}, batch_size={batch_size}")
    logger.info(f"Using features: {feature_columns}")

    if not df.index.is_unique:
        logger.warning("DataFrame index is not unique. Resetting index for reliable integer position mapping.")
        df = df.reset_index(drop=True) # Ensure 0..N-1 index

    all_windows = []
    location_indices_map = {} # Map location -> sorted list of original df *integer position* indices

    # --- Identify and sort locations consistently ---
    location_groups = []
    if 'name' in df.columns:
        logger.info("Grouping data by 'name' column.")
        for name in sorted(df['name'].unique()):
            group = df[df['name'] == name].copy()
            if isinstance(group.index, pd.DatetimeIndex): # Should not happen if index reset, but check
                group.sort_index(inplace=True)
            else:
                 group.sort_index(inplace=True) # Sort by integer index now
            location_groups.append((name, group))
            # Store integer positions (index values after potential reset)
            location_indices_map[name] = group.index.tolist()
    elif 'latitude' in df.columns and 'longitude' in df.columns:
        logger.info("Grouping data by 'latitude' and 'longitude' columns.")
        latlon_pairs = sorted(df[['latitude', 'longitude']].drop_duplicates().values.tolist())
        for lat, lon in latlon_pairs:
            group = df[(df['latitude'] == lat) & (df['longitude'] == lon)].copy()
            group.sort_index(inplace=True) # Sort by integer index
            location_groups.append(((lat, lon), group))
            location_indices_map[(lat, lon)] = group.index.tolist()
    else:
        logger.warning("No 'name' or 'latitude'/'longitude' columns found. Treating data as a single sequence.")
        df_sorted = df.sort_index() # Sort by integer index
        location_groups.append(('single_sequence', df_sorted))
        location_indices_map['single_sequence'] = df_sorted.index.tolist()

    logger.info(f"Processing {len(location_groups)} locations/sequences for windowing.")

    # --- Create windows for each location ---
    total_windows = 0
    for location_id, group_df in location_groups:
        missing_cols = [col for col in feature_columns if col not in group_df.columns]
        if missing_cols:
            logger.error(f"Missing required feature columns for location {location_id}: {missing_cols}. Skipping.")
            continue

        location_data = group_df[feature_columns].values
        if len(location_data) < input_width:
            logger.warning(f"Skipping location {location_id} - not enough data points ({len(location_data)} < {input_width})")
            continue

        try:
            # Create windows for this group's data
            # Use stride_tricks for potentially faster windowing than tf.keras.preprocessing
            num_windows_loc = len(location_data) - input_width + 1
            shape = (num_windows_loc, input_width, location_data.shape[1])
            strides = (location_data.strides[0], location_data.strides[0], location_data.strides[1])
            windows_np = np.lib.stride_tricks.as_strided(location_data, shape=shape, strides=strides)

            all_windows.extend(windows_np) # Add list of numpy arrays

            if num_windows_loc > 0:
                 logger.info(f"Created {num_windows_loc} windows for location {location_id}")
                 total_windows += num_windows_loc
            else:
                 logger.warning(f"Calculation resulted in zero windows for location {location_id}.")

        except Exception as e:
             logger.error(f"Error creating windows with stride_tricks for location {location_id}: {e}. Skipping.")
             continue


    if not all_windows:
        logger.error("No windows were created across all locations. Cannot proceed with prediction.")
        return None, {}, 0

    # Convert the list of windows to a single numpy array
    try:
        windows_array = np.array(all_windows)
        logger.info(f"Created a total of {total_windows} windows with shape {windows_array.shape}")
    except ValueError as ve:
         logger.error(f"Could not convert list of windows to numpy array: {ve}. Check window shapes.")
         # Log shapes of first few windows for debugging
         for i in range(min(5, len(all_windows))):
             logger.error(f"Window {i} shape: {all_windows[i].shape}")
         return None, {}, 0


    # Create the final TensorFlow dataset from the numpy array
    try:
        final_dataset = tf.data.Dataset.from_tensor_slices(windows_array)
        final_dataset = final_dataset.batch(batch_size)
        final_dataset = final_dataset.prefetch(tf.data.AUTOTUNE)
        logger.info("Successfully created batched TensorFlow dataset for prediction.")
    except Exception as e:
        logger.error(f"Failed to create final TensorFlow dataset: {e}")
        return None, {}, 0


    return final_dataset, location_indices_map, total_windows


def prepare_features_from_preprocessed(df, feature_scaler=None,
                                       input_width=8, label_width=1, shift=1):
    """
    Prepare features from preprocessed data using a specific feature order,
    handles scaling, and creates windowed dataset for sequence models.
    Biome features are excluded.

    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed data, potentially including target, features, location info.
    feature_scaler : sklearn.preprocessing.StandardScaler, optional
        Fitted scaler for features. If None, scaling is skipped.
    input_width : int
        Number of time steps for input window (for sequence models).
    label_width : int
        (Currently unused in prediction context but kept for consistency).
    shift : int
        (Currently unused in prediction context but kept for consistency).

    Returns:
    --------
    tuple: (tf.data.Dataset, pd.DataFrame, list, dict, int)
        - windowed_dataset: TensorFlow dataset with windowed features.
        - df_prepared: DataFrame with features potentially scaled and index reset.
        - feature_columns: List of actual feature columns used (in order).
        - location_indices_map: Map from location ID to original *integer position* indices.
        - total_windows: Total number of windows created.
        Returns (None, df, [], {}, 0) if preparation fails.
    """
    try:
        logger.info("Preparing features from preprocessed data (Biomes excluded)...")
        if df is None or df.empty:
             logger.error("Input DataFrame is None or empty.")
             return None, pd.DataFrame(), [], {}, 0

        # --- Define Feature Order and Identify Columns ---
        # Create a copy to avoid modifying the original DataFrame
        # drop the first column
        df['ta'] = df['ta'] - 273.15 # Convert temperature to Celsius if needed
        df_processed = df.copy()
        # summarize missing values
        missing_values = df_processed.isnull().sum()
        missing_summary = missing_values[missing_values > 0]
        if not missing_summary.empty:
            logger.warning(f"Missing values in the DataFrame:\n{missing_summary}")

        # Ensure unique index for reliable integer position mapping later
        if not df_processed.index.is_unique:
            logger.warning("DataFrame index is not unique. Resetting index.")
            df_processed.reset_index(drop=True, inplace=True)

        # rename annual_mean_temperature and annual_mean_precipitation to mean_annual_temp and mean_annual_precip
        if 'annual_mean_temperature' in df_processed.columns:
            df_processed.rename(columns={'annual_mean_temperature': 'mean_annual_temp'}, inplace=True)
        if 'annual_precipitation' in df_processed.columns:
            df_processed.rename(columns={'annual_precipitation': 'mean_annual_precip'}, inplace=True)
        # Convert specific cyclical time features to lowercase if they exist
        time_features_caps = ['Day sin', 'Week sin', 'Month sin', 'Year sin']
        for col in time_features_caps:
            if col in df_processed.columns:
                lowercase_col = col.lower()
                if lowercase_col != col: # Avoid renaming if already lowercase
                    logger.info(f"Renaming column '{col}' to '{lowercase_col}'")
                    df_processed.rename(columns={col: lowercase_col}, inplace=True)

        # Define the exact desired order of features (excluding target and biomes)
        ordered_features = [
            'ext_rad', 'sw_in', 'ta', 'ws', 'vpd', 'ppfd_in',
            'mean_annual_temp', 'mean_annual_precip', # Added features
            'day sin', 'week sin', 'month sin', 'year sin', # Time features (lowercase)
            # Biome types REMOVED
        ]

        # Filter to get features actually present in the DataFrame, maintaining the order
        available_features = [col for col in ordered_features if col in df_processed.columns]

        # Check if any specified features are missing
        missing_features = [col for col in ordered_features if col not in df_processed.columns]
        if missing_features:
            logger.warning(f"Specified features missing from the dataset: {missing_features}")

        # Check if we have *any* features left
        if not available_features:
            logger.error("None of the specified features (excluding biomes) are present in the dataset columns.")
            logger.error(f"Available columns: {df_processed.columns.tolist()}")
             # Attempt fallback (as before, but ensure no biomes included if they exist)
            potential_features = df_processed.select_dtypes(include=np.number).columns.tolist()
            excluded_cols = ['sap_velocity', 'latitude', 'longitude'] # Add others if needed
            # Also exclude known biome names if they somehow exist
            known_biomes_lower = [b.lower() for b in [
                'Boreal forest', 'Subtropical desert', 'Temperate forest',
                'Temperate grassland desert', 'Temperate rain forest',
                'Tropical forest savanna', 'Tropical rain forest',
                'Tundra', 'Woodland/Shrubland']]
            available_features = [
                col for col in potential_features
                if col not in excluded_cols
                and col.lower() not in excluded_cols
                and col.lower() not in known_biomes_lower
            ]
            if not available_features:
                 logger.error("Fallback failed: No suitable numeric non-biome feature columns found.")
                 return None, df, [], {}, 0
            else:
                 logger.warning(f"Falling back to using automatically detected numeric non-biome features: {available_features}")


        logger.info(f"Using {len(available_features)} features in specified order: {available_features}")
        feature_columns = available_features # Final list of features to use

        # --- Apply Feature Scaling ---
        df_scaled = df_processed.copy() # Work on a new copy for scaling
        if feature_scaler is not None:
            # Define columns to scale (exclude potentially mean annual stats)
    

            scale_columns = [col for col in feature_columns]
            missing_scale_cols = [col for col in scale_columns if col not in df_scaled.columns]

            if missing_scale_cols:
                 logger.warning(f"Columns intended for scaling are missing from the DataFrame: {missing_scale_cols}")
                 scale_columns = [col for col in scale_columns if col in df_scaled.columns]


            if not scale_columns:
                 logger.warning("No columns identified for scaling.")
            else:
                logger.info(f"Applying feature scaling to columns: {scale_columns}")
                try:
                    if not hasattr(feature_scaler, 'n_features_in_'):
                         logger.warning("Feature scaler does not appear to be fitted. Skipping scaling.")
                    else:
                        logger.info(f"Feature scaler was fitted with {feature_scaler.n_features_in_} features.")
                        if all(col in df_scaled.columns for col in scale_columns):
                            # Important: Ensure the scaler is applied ONLY to the columns it was fitted on, in the correct order.
                            # This requires the loaded scaler to match the `scale_columns` derived here.
                            # A robust implementation would store/load the column names with the scaler.
                            # Assuming the scaler was fitted on columns equivalent to `scale_columns`:
                            try:
                                df_scaled[scale_columns] = feature_scaler.transform(df_scaled[scale_columns])
                                logger.info("Successfully applied feature scaling.")
                            except ValueError as ve:
                                 logger.error(f"ValueError during scaling transform: {ve}. Check feature mismatch.")
                                 logger.error(f"Columns attempted: {scale_columns}")
                                 logger.error(f"Scaler expected features (if available): {getattr(feature_scaler, 'feature_names_in_', 'Not Available')}")
                                 logger.warning("Using unscaled features due to scaling error.")
                                 df_scaled = df_processed.copy() # Revert to unscaled
                        else:
                            logger.error(f"Cannot apply scaling. Not all scale_columns exist in DataFrame: {scale_columns}")
                            logger.warning("Using unscaled features due to missing columns.")

                except Exception as e:
                    logger.error(f"Unexpected error applying feature scaling: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    logger.warning("Using unscaled features due to scaling error.")
                    df_scaled = df_processed.copy() # Revert to unscaled
        else:
            logger.warning("No feature scaler provided. Using raw features.")

        # df_prepared now holds the data ready for windowing (scaled or not, index reset)
        df_prepared = df_scaled

        # --- Create Windowed Dataset ---
        logger.info(f"Creating prediction windows for sequence models using input_width={input_width}")
        try:
            # Pass the prepared DataFrame (df_prepared) which has a unique integer index
            windowed_dataset, location_indices_map, total_windows = create_prediction_windows(
                df_prepared,
                feature_columns,
                input_width=input_width,
                batch_size=32
            )

            if windowed_dataset is None or total_windows == 0:
                 logger.error("Window creation failed or resulted in zero windows.")
                 return None, df_prepared, feature_columns, {}, 0

            logger.info(f"Successfully created {total_windows} prediction windows.")
            # Return df_prepared as it contains the state before windowing
            return windowed_dataset, df_prepared, feature_columns, location_indices_map, total_windows

        except Exception as e:
            logger.error(f"Error creating prediction windows: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, df_prepared, feature_columns, {}, 0

    except Exception as e:
        logger.error(f"Error in prepare_features_from_preprocessed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Return safely
        return None, df if 'df' in locals() else pd.DataFrame(), [], {}, 0


def make_predictions(df_features, feature_columns, models, label_scaler,
                     windowed_dataset=None, location_indices_map=None, total_windows=0,
                     input_width=8, shift=1):
    """
    Make predictions using multiple model types. Handles sequence models
    (using windowed_dataset) and traditional models (using flattened windows).

    Parameters:
    -----------
    df_features : pandas.DataFrame
        DataFrame containing the features (potentially scaled) and original index/metadata.
        Should have a unique integer index if prepared by `prepare_features_from_preprocessed`.
    feature_columns : list
        Ordered list of feature column names used for modeling.
    models : dict
        Dictionary of loaded models {model_type: model_object}.
    label_scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler for the target variable (sap_velocity). Required for inverse transform.
    windowed_dataset : tf.data.Dataset, optional
        Windowed dataset created by `create_prediction_windows` for sequence models.
    location_indices_map : dict, optional
        Map from location ID to *integer position* indices, needed for mapping predictions.
    total_windows : int, optional
        Total number of windows created, used for consistency checks.
    input_width : int
        Width of the input window used for sequence models.
    shift : int
        Prediction shift used when mapping windowed predictions back.

    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with added prediction columns ('sap_velocity_{model_type}').
    """
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from collections import defaultdict

    logger.info("Making predictions with loaded models...")

    if df_features is None or df_features.empty:
        logger.error("Input DataFrame for prediction is empty or None.")
        return pd.DataFrame()
    if not models:
        logger.error("No models provided for prediction.")
        return df_features # Return original features df
    if label_scaler is None:
         logger.error("Label scaler is required for inverse transforming predictions. Cannot proceed.")
         return df_features

    # Create copy of DataFrame to add results to
    # Preserve the original index if it exists and is meaningful
    df_result = df_features.copy()

    # --- Categorize models ---
    deep_sequence_models = {} # Models that take sequence input (windows)
    traditional_ml_models = {} # Models that need flattened inputs

    for model_type, model in models.items():
        mt_lower = model_type.lower()
        if mt_lower in ['cnn_lstm', 'lstm', 'transformer', 'ann', 'gru']:
            if isinstance(model, tf.keras.Model):
                deep_sequence_models[model_type] = model
                logger.info(f"Categorized '{model_type}' as deep sequence model.")
            else:
                logger.warning(f"Model '{model_type}' expected Keras model, got {type(model)}. Treating as traditional.")
                if hasattr(model, 'predict'): traditional_ml_models[model_type] = model
        elif mt_lower in ['svr', 'rf', 'xgb', 'xgboost', 'randomforest', 'svm', 'linearregression', 'ridge', 'lasso']:
            if hasattr(model, 'predict'):
                 traditional_ml_models[model_type] = model
                 logger.info(f"Categorized '{model_type}' as traditional ML model.")
            else:
                 logger.warning(f"Model '{model_type}' lacks 'predict' method. Skipping.")
        else:
            logger.warning(f"Unrecognized model type '{model_type}'. Trying as traditional.")
            if hasattr(model, 'predict'):
                 traditional_ml_models[model_type] = model
            else:
                 logger.warning(f"Model '{model_type}' lacks 'predict' method. Skipping.")


    # --- Generate Predictions ---
    all_predictions = {} # Store raw predictions {model_type: np.array[total_windows]}

    # 1. Deep Sequence Models (using windowed_dataset)
    if deep_sequence_models:
        if windowed_dataset is None or total_windows == 0:
            logger.warning("Windowed dataset unavailable/empty. Skipping deep sequence models.")
        else:
            logger.info(f"Predicting with {len(deep_sequence_models)} deep sequence models...")
            for model_type, model in deep_sequence_models.items():
                try:
                    logger.info(f"Predicting with {model_type}...")
                    preds_scaled = model.predict(windowed_dataset, verbose=0)

                    # Handle output shape variations
                    if len(preds_scaled.shape) == 3:
                         preds_scaled = preds_scaled[:, -1, :]
                    if len(preds_scaled.shape) == 2 and preds_scaled.shape[1] > 1:
                         logger.warning(f"{model_type} output shape {preds_scaled.shape}. Using first column.")
                         preds_scaled = preds_scaled[:, 0].reshape(-1, 1)
                    elif not (len(preds_scaled.shape) == 2 and preds_scaled.shape[1] == 1):
                         logger.error(f"Unexpected prediction shape from {model_type}: {preds_scaled.shape}. Skipping.")
                         continue

                    if len(preds_scaled) != total_windows:
                         logger.warning(f"Prediction count mismatch for {model_type}: {len(preds_scaled)} vs {total_windows} windows.")

                    preds_unscaled = label_scaler.inverse_transform(preds_scaled).flatten()
                    all_predictions[model_type] = preds_unscaled
                    logger.info(f"Generated {len(preds_unscaled)} predictions for {model_type}.")

                except Exception as e:
                    logger.error(f"Error predicting with deep sequence model {model_type}: {e}", exc_info=True)

    # 2. Traditional ML Models (need flattened windows)
    if traditional_ml_models:
        if windowed_dataset is None or total_windows == 0:
            logger.warning("Windowed dataset unavailable/empty. Skipping traditional models.")
        else:
            logger.info(f"Preparing flattened windows for {len(traditional_ml_models)} traditional ML models...")
            try:
                # Efficiently get all windows as a single numpy array from the dataset
                all_windows_np = np.concatenate([batch.numpy() for batch in windowed_dataset], axis=0)
                if all_windows_np.shape[0] != total_windows:
                     logger.warning(f"Concatenated windows count mismatch: {all_windows_np.shape[0]} vs {total_windows}")
                     # Adjust total_windows if needed? Or proceed with caution.
                     # total_windows = all_windows_np.shape[0] # Risky

                # Flatten the windows
                num_features = len(feature_columns)
                expected_flattened_size = input_width * num_features
                X_flattened = all_windows_np.reshape(total_windows, expected_flattened_size)

                logger.info(f"Created {len(X_flattened)} flattened windows of shape {X_flattened.shape}")

                for model_type, model in traditional_ml_models.items():
                    try:
                        logger.info(f"Predicting with {model_type}...")
                        preds_scaled = model.predict(X_flattened)
                        preds_unscaled = label_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

                        if len(preds_unscaled) != total_windows:
                             logger.warning(f"Prediction count mismatch for {model_type}: {len(preds_unscaled)} vs {total_windows} windows.")

                        all_predictions[model_type] = preds_unscaled
                        logger.info(f"Generated {len(preds_unscaled)} predictions for {model_type}.")

                    except Exception as e:
                        logger.error(f"Error predicting with traditional model {model_type}: {e}", exc_info=True)

            except Exception as e:
                logger.error(f"Error creating/processing flattened windows: {e}", exc_info=True)


    # --- Map all generated predictions back to the DataFrame ---
    if not all_predictions:
        logger.warning("No predictions were generated by any model.")
        return df_result

    logger.info(f"Mapping predictions from {len(all_predictions)} models back to DataFrame...")

    if location_indices_map is None or total_windows == 0:
         logger.error("Location map or window count missing. Cannot map windowed predictions accurately.")
         return df_result

    # Create the mapping array (prediction indices for each row)
    prediction_idx_map = map_windowed_predictions_to_df(
        df_shape=df_features.shape, # Shape of the DataFrame used for windowing
        num_predictions=total_windows,
        location_indices_map=location_indices_map, # Map with integer positions
        input_width=input_width,
        shift=shift
    )

    # Add prediction columns using the map
    for model_type, preds in all_predictions.items():
        pred_col_name = f'sap_velocity_{model_type}'
        # Initialize column with NaNs using the result DataFrame's index
        df_result[pred_col_name] = pd.Series(np.nan, index=df_result.index)

        num_preds_to_map = min(len(preds), total_windows)
        if len(preds) != total_windows:
             logger.warning(f"Mapping {num_preds_to_map} predictions for {model_type} due to length mismatch ({len(preds)} vs {total_windows}).")

        mapped_count = 0
        # Iterate through the integer positions and the prediction index map
        for int_pos, pred_map_val in enumerate(prediction_idx_map):
             if not np.isnan(pred_map_val):
                 pred_idx = int(pred_map_val)
                 if pred_idx < num_preds_to_map:
                     # Use integer position (int_pos) with iloc for assignment
                     try:
                         df_result.iloc[int_pos, df_result.columns.get_loc(pred_col_name)] = preds[pred_idx]
                         mapped_count += 1
                     except IndexError:
                          logger.error(f"IndexError: Cannot access iloc[{int_pos}] for DataFrame with shape {df_result.shape}. Skipping mapping for prediction {pred_idx}.")
                     except Exception as e:
                          logger.error(f"Error mapping prediction {pred_idx} to {pred_col_name} at position {int_pos}: {e}")

        logger.info(f"Mapped {mapped_count} predictions for {model_type} to column '{pred_col_name}'.")


    # --- Create Ensemble Prediction ---
    pred_cols = [col for col in df_result.columns if col.startswith('sap_velocity_') and col != 'sap_velocity_ensemble']
    if len(pred_cols) > 1:
        logger.info(f"Creating ensemble prediction (median) from: {pred_cols}")
        df_result['sap_velocity_ensemble'] = df_result[pred_cols].median(axis=1)
        non_nan_ensemble = df_result['sap_velocity_ensemble'].notna().sum()
        logger.info(f"Computed {non_nan_ensemble} non-NaN ensemble predictions.")
    elif len(pred_cols) == 1:
         logger.info(f"Only one prediction column ('{pred_cols[0]}'). Using it as the 'ensemble'.")
         df_result['sap_velocity_ensemble'] = df_result[pred_cols[0]]
    else:
        logger.warning("No individual model predictions available to create an ensemble.")

    return df_result


def visualize_predictions(df_predictions, output_dir, feature_columns, include_feature_importance=True, models=None):
    """
    Create visualizations of the predictions and optionally feature importance.

    Parameters:
    -----------
    df_predictions : pandas.DataFrame
        DataFrame with original features and added prediction columns.
        Should have index restored or a time column available.
    output_dir : Path
        Directory to save visualizations.
    feature_columns : list
        List of feature columns used in the model (for importance plots).
    include_feature_importance : bool
        Whether to attempt generating feature importance plots.
    models : dict, optional
        Dictionary of loaded models, needed for feature importance extraction.
    """
    logger.info("Creating visualizations...")
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir.mkdir(exist_ok=True, parents=True)
    df_plot = df_predictions.copy()

    pred_cols = [col for col in df_plot.columns if col.startswith('sap_velocity_')]
    ensemble_col = 'sap_velocity_ensemble' if 'sap_velocity_ensemble' in pred_cols else None
    individual_pred_cols = [col for col in pred_cols if col != ensemble_col]

    # --- Find or Create Time Column for Plotting ---
    time_col = None
    if isinstance(df_plot.index, pd.DatetimeIndex):
        time_col = df_plot.index.name if df_plot.index.name else 'datetime'
        df_plot[time_col] = df_plot.index # Add index as column
        # Keep original index for now, reset later if needed for grouping
        logger.info(f"Using DatetimeIndex column '{time_col}' for plotting.")
    else:
        # Try to find a time-like column
        potential_time_cols = [c for c in df_plot.columns if any(w in c.lower() for w in ['time', 'date'])]
        if potential_time_cols:
             for col in potential_time_cols:
                 try:
                     df_plot[col] = pd.to_datetime(df_plot[col])
                     time_col = col
                     logger.info(f"Using existing column '{time_col}' converted to datetime for plotting.")
                     break
                 except Exception:
                     logger.warning(f"Could not convert potential time column '{col}' to datetime.")
        if time_col is None:
            logger.warning("No suitable time column found. Time series plots will use row number (index).")
            time_col = 'row_number'
            df_plot[time_col] = range(len(df_plot)) # Use simple range

    # --- Identify Locations ---
    location_col = None
    unique_locations = []
    group_by_cols = []
    if 'name' in df_plot.columns:
        location_col = 'name'
        group_by_cols = ['name']
        unique_locations = sorted(df_plot['name'].unique())
        logger.info("Identifying locations using 'name' column.")
    elif 'latitude' in df_plot.columns and 'longitude' in df_plot.columns:
        location_col = ('latitude', 'longitude')
        group_by_cols = ['latitude', 'longitude']
        unique_locations = sorted(df_plot[group_by_cols].drop_duplicates().values.tolist())
        logger.info("Identifying locations using 'latitude' and 'longitude' columns.")
    else:
        logger.info("No location columns found. Plotting data as a single sequence.")
        unique_locations = ['single_sequence']

    logger.info(f"Found {len(unique_locations)} unique locations/sequences to plot.")

    max_plots = 10
    locations_to_plot = unique_locations[:max_plots]
    if len(unique_locations) > max_plots:
        logger.warning(f"Limiting visualization to the first {max_plots} locations.")

    # --- Plot Time Series for Each Location ---
    if not group_by_cols: # Handle single sequence case
         groups = [('single_sequence', df_plot)]
    else:
         groups = df_plot.groupby(group_by_cols)

    plot_count = 0
    for location_id, point_data_full in groups:
        if plot_count >= max_plots: break

        point_data = point_data_full.copy() # Work on a copy for each group
        point_data.sort_values(by=time_col, inplace=True) # Sort group by time

        if isinstance(location_id, tuple):
            loc_str = f"Lat={location_id[0]:.2f}_Lon={location_id[1]:.2f}"
        else:
            loc_str = str(location_id)

        if point_data.empty:
            logger.warning(f"No data for location {loc_str}. Skipping plot.")
            continue

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        # Plot 1: Predictions
        ax = axes[0]
        if ensemble_col and ensemble_col in point_data.columns:
            ax.plot(point_data[time_col], point_data[ensemble_col], label='Ensemble', color='black', linewidth=2.5, zorder=10)
        palette = sns.color_palette("tab10", len(individual_pred_cols))
        for i, col in enumerate(individual_pred_cols):
            if col in point_data.columns:
                model_type = col.replace('sap_velocity_', '')
                ax.plot(point_data[time_col], point_data[col], label=f'{model_type.upper()}', color=palette[i], alpha=0.8, linewidth=1.5)
        ax.set_title(f'Sap Velocity Predictions - {loc_str}')
        ax.set_ylabel('Sap Velocity')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.tick_params(axis='x', rotation=45)

        # Plot 2: Key Features (Normalized)
        ax = axes[1]
        key_features_to_plot = ['ta', 'vpd', 'ppfd_in', 'sw_in', 'ws', 'day sin', 'month sin']
        plot_features = [f for f in key_features_to_plot if f in point_data.columns]
        if plot_features:
            norm_data = point_data[plot_features].copy()
            for col in plot_features:
                 min_val, max_val = norm_data[col].min(), norm_data[col].max()
                 norm_data[col] = (norm_data[col] - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            palette_feat = sns.color_palette("Greys", len(plot_features) + 2)
            for i, col in enumerate(plot_features):
                 ax.plot(point_data[time_col], norm_data[col], label=col, color=palette_feat[i+1], linewidth=1)
            ax.set_title('Selected Normalized Features')
            ax.set_ylabel('Normalized Value')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        else:
             ax.set_visible(False)

        ax.set_xlabel(time_col.replace('_', ' ').title())
        plt.tight_layout(rect=[0, 0, 0.9, 1])

        safe_loc_str = loc_str.replace(' ', '_').replace('=', '').replace('.', 'p').replace(',', '').replace('/', '_').replace('\\', '_')
        filename = f'predictions_{safe_loc_str}.png'
        filepath = output_dir / filename
        try:
            plt.savefig(filepath, bbox_inches='tight')
            logger.info(f"Saved time series plot: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save plot {filepath}: {e}")
        plt.close(fig)
        plot_count += 1


    # --- Plot Feature Importance ---
    if include_feature_importance and models is not None and feature_columns:
        logger.info("Attempting to generate feature importance plots...")
        for model_type, model in models.items():
            importances = None
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_') and len(model.coef_.shape) == 1:
                importances = np.abs(model.coef_)

            if importances is not None:
                 if len(importances) == len(feature_columns):
                     try:
                         imp_df = pd.DataFrame({'Feature': feature_columns, 'Importance': importances})
                         imp_df = imp_df.sort_values('Importance', ascending=False).head(20)
                         plt.figure(figsize=(10, max(6, len(imp_df) * 0.4)))
                         sns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis')
                         plt.title(f'Feature Importance - {model_type.upper()}')
                         plt.tight_layout()
                         filepath = output_dir / f'feature_importance_{model_type}.png'
                         plt.savefig(filepath)
                         logger.info(f"Saved feature importance plot: {filepath}")
                         plt.close()
                         imp_df.to_csv(output_dir / f'feature_importance_{model_type}.csv', index=False)
                     except Exception as e:
                          logger.error(f"Error plotting/saving importance for {model_type}: {e}")
                 else:
                      logger.warning(f"Feature importance length mismatch for {model_type}: {len(importances)} vs {len(feature_columns)} features.")

    logger.info(f"Visualizations saved to {output_dir}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Predict sap velocity from preprocessed data using specified models and feature order (Biomes excluded).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file with preprocessed data.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory. Defaults to ./outputs/<input_file_stem>/')
    parser.add_argument('--models-dir', type=str, default='./outputs/models',
                        help='Directory containing trained model subdirectories.')
    parser.add_argument('--scaler-dir', type=str, default='./outputs/scalers',
                        help='Directory containing fitted feature_scaler.pkl and label_scaler.pkl.')
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
                        help='Attempt feature importance plots.')

    return parser.parse_args()


def main():
    """Main execution function"""
    start_time = time.time()
    args = parse_args()

    # --- Setup Directories ---
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

    # --- Log Setup ---
    logger.info("="*60)
    logger.info(" Starting Sap Velocity Prediction Script (Biomes Excluded) ")
    logger.info("="*60)
    logger.info(f"Input Data: {input_file}")
    logger.info(f"Models Directory: {models_dir}")
    logger.info(f"Scalers Directory: {scaler_dir}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Model Types: {args.model_types}")
    logger.info(f"Sequence Params: Input Width={args.input_width}, Shift={args.shift}")
    logger.info(f"Visualize: {args.visualize}, Feature Importance: {args.feature_importance}")

    # --- Load Models ---
    models = load_models(models_dir, args.model_types)
    if not models: sys.exit("No models loaded. Exiting.")

    # --- Load Scalers ---
    feature_scaler, label_scaler = load_scalers(scaler_dir)
    if label_scaler is None: sys.exit("Label scaler not found. Exiting.")
    if feature_scaler is None: logger.warning("Feature scaler not found. Features will be unscaled.")

    # --- Load Data ---
    df_raw = load_preprocessed_data(input_file)
    if df_raw is None or df_raw.empty: sys.exit("Failed to load input data. Exiting.")

    # Store original index before potentially resetting it in prepare_features
    original_index = df_raw.index

    # --- Prepare Features & Windows ---
    windowed_dataset, df_prepared, feature_columns, location_map, n_windows = prepare_features_from_preprocessed(
        df_raw,
        feature_scaler,
        input_width=args.input_width,
        label_width=args.label_width,
        shift=args.shift
    )
    if windowed_dataset is None or not feature_columns:
        sys.exit("Failed to prepare features or create windows. Exiting.")

    # --- Make Predictions ---
    # Pass df_prepared which has the correct structure (potentially reset index) for mapping
    df_predictions_mapped = make_predictions(
        df_prepared,
        feature_columns,
        models,
        label_scaler,
        windowed_dataset=windowed_dataset,
        location_indices_map=location_map,
        total_windows=n_windows,
        input_width=args.input_width,
        shift=args.shift
    )
    if df_predictions_mapped.empty:
        sys.exit("Prediction failed or returned empty DataFrame. Exiting.")

    # --- Restore Original Index ---
    # The df_predictions_mapped has the integer index from df_prepared.
    # We need to align it back to the original index for saving and visualization.
    if len(original_index) == len(df_predictions_mapped):
         df_final_predictions = df_predictions_mapped.set_index(original_index)
         logger.info("Restored original index to the predictions DataFrame.")
    else:
         logger.warning(f"Length mismatch between original index ({len(original_index)}) and predictions ({len(df_predictions_mapped)}). Cannot restore original index reliably.")
         df_final_predictions = df_predictions_mapped # Keep the potentially reset index


    # --- Save Predictions ---
    output_path = output_dir / f'{input_file.stem}_predictions.csv'
    try:
        df_final_predictions.to_csv(output_path, index=True)
        logger.info(f"Predictions saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save predictions to {output_path}: {e}")

    # --- Visualize (Optional) ---
    if args.visualize:
        visualize_predictions(
            df_final_predictions, # Use the DataFrame with the potentially restored index
            output_dir,
            feature_columns,
            args.feature_importance,
            models
            )

    # --- Finish ---
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info("="*60)
    logger.info(" Prediction Script Finished ")
    logger.info(f" Execution Time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as se:
         logger.info(f"Script exited with code: {se}") # Log system exits as info
         sys.exit(se.code) # Propagate exit code
    except FileNotFoundError as fnf:
         logger.error(f"File not found error: {fnf}", exc_info=True)
         sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled error during execution: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Script execution completed.")
        # Any cleanup or final logging can be done here