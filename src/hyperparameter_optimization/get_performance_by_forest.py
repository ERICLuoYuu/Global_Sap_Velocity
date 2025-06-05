"""
Multi-Model Evaluation Script with IGBP forest type analysis.
Supports CNN-LSTM, LSTM, ANN, XGBoost, and Random Forest models.
This script loads pre-trained models to make predictions and analyze performance
across different IGBP forest types using MODIS land cover data.
"""
from pathlib import Path
import sys
import os
import joblib
import time
import warnings

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the randomization control module
from src.utils.random_control import (
    set_seed, get_seed, deterministic,
)
from src.hyperparameter_optimization.timeseries_processor import TimeSeriesSegmenter, SegmentedWindowGenerator

# Set the master seed for reproducibility
set_seed(42)

# Import all dependencies
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import ee
from tqdm import tqdm

# Configure GPU for determinism if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# Suppress warnings
warnings.filterwarnings("ignore")

#############################################################################
# Google Earth Engine Integration Functions
#############################################################################

def initialize_earth_engine():
    """Initialize Google Earth Engine with appropriate authentication."""
    try:
        ee.Initialize(project = 'era5download-447713')
        print("Google Earth Engine initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing Google Earth Engine: {e}")
        print("Proceeding without GEE functionality")
        return False

def get_igbp_landcover(lat, lon, year=2020):
    """
    Retrieve IGBP landcover classification from MODIS for given coordinates.
    
    Parameters:
    -----------
    lat : float
        Latitude
    lon : float
        Longitude
    year : int
        Year for landcover data (default: 2020)
        
    Returns:
    --------
    str
        IGBP classification name
    """
    # IGBP classification mapping
    igbp_classes = {
        0: "Water",
        1: "Evergreen Needleleaf Forest",
        2: "Evergreen Broadleaf Forest", 
        3: "Deciduous Needleleaf Forest",
        4: "Deciduous Broadleaf Forest",
        5: "Mixed Forest",
        6: "Closed Shrublands",
        7: "Open Shrublands",
        8: "Woody Savannas",
        9: "Savannas",
        10: "Grasslands",
        11: "Permanent Wetlands",
        12: "Croplands",
        13: "Urban and Built-up",
        14: "Cropland/Natural Vegetation Mosaic",
        15: "Snow and Ice",
        16: "Barren or Sparsely Vegetated",
        17: "Unclassified"
    }
    
    try:
        # Define point of interest
        point = ee.Geometry.Point([lon, lat])
        
        # Get MODIS landcover dataset
        # Using the MCD12Q1.006 MODIS Land Cover Type Yearly Global 500m
        modis_lc = ee.ImageCollection("MODIS/006/MCD12Q1")
        
        # Filter by year
        modis_year = modis_lc.filter(ee.Filter.calendarRange(year, year, 'year')).first()
        
        # Extract the IGBP classification band
        igbp = modis_year.select('LC_Type1')
        
        # Sample the value at the point
        value = igbp.sample(point, 500).first().get('LC_Type1').getInfo()
        
        # Return the class name
        return igbp_classes.get(value, "Unknown")
    
    except Exception as e:
        print(f"Error getting landcover for coordinates ({lat}, {lon}): {e}")
        return "Unknown"

def batch_process_coordinates(coordinates_df, cache_file="igbp_cache.csv"):
    """
    Process multiple coordinates and cache results to avoid redundant API calls.
    
    Parameters:
    -----------
    coordinates_df : pandas.DataFrame
        DataFrame with 'latitude' and 'longitude' columns
    cache_file : str
        File to store cached results
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added 'igbp_class' column
    """
    # Create cache directory if it doesn't exist
    cache_dir = Path('./outputs/cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_file
    
    # Check if cache exists
    try:
        cache = pd.read_csv(cache_path)
        cache_dict = dict(zip(zip(cache['latitude'], cache['longitude']), cache['igbp_class']))
        print(f"Loaded {len(cache_dict)} cached IGBP classifications")
    except:
        cache_dict = {}
        print("No cache found, creating new cache")
    
    # Round coordinates to reduce redundant API calls (0.01° ≈ 1km)
    coordinates_df['lat_rounded'] = coordinates_df['latitude'].round(4)
    coordinates_df['lon_rounded'] = coordinates_df['longitude'].round(4)
    
    # Create unique coordinate pairs
    unique_coords = set(zip(coordinates_df['lat_rounded'], coordinates_df['lon_rounded']))
    new_coords = [coord for coord in unique_coords if coord not in cache_dict]
    
    # Process new coordinates
    if new_coords:
        print(f"Processing {len(new_coords)} new coordinate pairs...")
        for lat, lon in tqdm(new_coords):
            cache_dict[(lat, lon)] = get_igbp_landcover(lat, lon)
            # Add small delay to avoid rate limiting
            time.sleep(0.2)
    
        # Update cache file
        cache_df = pd.DataFrame([
            {'latitude': lat, 'longitude': lon, 'igbp_class': cls}
            for (lat, lon), cls in cache_dict.items()
        ])
        cache_df.to_csv(cache_path, index=False)
        print(f"Updated cache with {len(cache_dict)} entries")
    
    # Map IGBP classifications to original dataframe
    coordinates_df['igbp_class'] = [
        cache_dict.get((lat, lon), "Unknown") 
        for lat, lon in zip(coordinates_df['lat_rounded'], coordinates_df['lon_rounded'])
    ]
    
    return coordinates_df

#############################################################################
# Model Functions
#############################################################################

def load_pretrained_model(model_path):
    """
    Load a pre-trained model from disk, handling different format types.
    
    Parameters:
    -----------
    model_path : str or Path
        Path to the saved model
        
    Returns:
    --------
    tuple
        (model, model_type, expects_sequence)
    """
    # Keep original path string for debugging and direct access if needed
    original_path = model_path
    
    # Ensure model_path is a string first for analysis
    if not isinstance(model_path, str):
        model_path = str(model_path)
    
    # Print debug info about the path
    print(f"Original model path: {model_path}")
    print(f"Path exists: {os.path.exists(model_path)}")
    
    # Check file extension directly from string
    if model_path.endswith('.joblib'):
        print("Detected .joblib extension in string path")
        is_joblib = True
    elif model_path.endswith('.keras') or model_path.endswith('.h5'):
        print("Detected .keras or .h5 extension in string path")
        is_keras = True
    else:
        print("No recognized extension found in string path")
        is_joblib = False
        is_keras = False
    
    # Now convert to Path object for easier handling
    model_path = Path(model_path)
    
    print(f"Path object: {model_path}")
    print(f"Path suffix: '{model_path.suffix}'")
    print(f"Path stem: '{model_path.stem}'")
    
    # Check if suffix is missing but extension is in the name
    if not model_path.suffix and is_joblib:
        print("Path object missing suffix but .joblib detected in string - will treat as joblib")
        model_type = None
        expects_sequence = False
        
        try:
            print(f"Loading machine learning model directly: {original_path}")
            model = joblib.load(original_path)
            
            # Determine model type from filename or model class
            model_name = model_path.name.lower()
            if 'rf' in model_name or 'randomforest' in model_name or 'random_forest' in model_name:
                model_type = 'rf'
            elif 'xgb' in model_name or 'xgboost' in model_name:
                model_type = 'xgb'
            else:
                # Try to infer from model type
                model_class = str(type(model)).lower()
                if 'forest' in model_class:
                    model_type = 'rf'
                elif 'xgb' in model_class or 'xgboost' in model_class:
                    model_type = 'xgb'
                else:
                    model_type = 'ml_model'
            
            print(f"Detected model type: {model_type}")
            return model, model_type, expects_sequence
        except Exception as e:
            print(f"Error loading as joblib despite extension: {e}")
            # Continue to other methods if this fails
    
    # Check if suffix is missing but extension is in the name
    if not model_path.suffix and is_keras:
        print("Path object missing suffix but .keras or .h5 detected in string - will treat as keras")
        model_type = None
        expects_sequence = False
        
        try:
            print(f"Loading deep learning model directly: {original_path}")
            model = tf.keras.models.load_model(original_path)
            
            # Determine model type from layers - improved to detect bidirectional LSTM
            has_bidirectional = any(isinstance(layer, tf.keras.layers.Bidirectional) for layer in model.layers)
            has_lstm = any(isinstance(layer, tf.keras.layers.LSTM) for layer in model.layers)
            has_conv = any(isinstance(layer, tf.keras.layers.Conv1D) for layer in model.layers)
            
            # Check for bidirectional LSTM specifically
            if has_bidirectional:
                # Find all bidirectional layers
                bidirectional_layers = [layer for layer in model.layers 
                                        if isinstance(layer, tf.keras.layers.Bidirectional)]
                
                # Check if any bidirectional layer wraps an LSTM
                has_bidirectional_lstm = any(
                    isinstance(layer.forward_layer, tf.keras.layers.LSTM) or
                    isinstance(layer.backward_layer, tf.keras.layers.LSTM)
                    for layer in bidirectional_layers
                )
                
                if has_bidirectional_lstm:
                    if has_conv:
                        model_type = 'cnn_lstm'
                    else:
                        model_type = 'lstm'  # New model type for bidirectional LSTM
                    expects_sequence = True
            elif has_lstm:
                if has_conv:
                    model_type = 'cnn_lstm'
                else:
                    model_type = 'lstm'
                expects_sequence = True
            else:
                model_type = 'ann'
                # Check if ANN expects sequential data
                try:
                    input_shape = model.input_shape
                    if len(input_shape) > 2:  # More than 2 dims suggests sequential
                        expects_sequence = True
                except:
                    expects_sequence = False
            
            print(f"Detected model type: {model_type}")
            return model, model_type, expects_sequence
        except Exception as e:
            print(f"Error loading as keras despite extension: {e}")
            # Continue to other methods if this fails
    
    # Determine model type from file extension and name normally
    if model_path.suffix.lower() in ['.keras', '.h5']:
        try:
            print(f"Loading deep learning model from {model_path}")
            model = tf.keras.models.load_model(model_path)
            
            # Try different methods to determine the input shape
            try:
                # Get the model's input shape (safer than checking the first layer)
                if hasattr(model, 'input_shape'):
                    input_shape = model.input_shape
                    print(f"Model input shape from model.input_shape: {input_shape}")
                elif hasattr(model, 'inputs') and model.inputs:
                    input_shape = model.inputs[0].shape
                    print(f"Model input shape from model.inputs[0].shape: {input_shape}")
                else:
                    # Try to get it from the first layer
                    first_layer = model.layers[0]
                    if hasattr(first_layer, 'input_shape'):
                        input_shape = first_layer.input_shape
                    elif hasattr(first_layer, 'input') and hasattr(first_layer.input, 'shape'):
                        input_shape = first_layer.input.shape
                    elif hasattr(first_layer, 'get_config'):
                        config = first_layer.get_config()
                        if 'batch_input_shape' in config:
                            input_shape = config['batch_input_shape']
                        else:
                            # Default fallback - assume it's sequential based on model type
                            input_shape = None
                    else:
                        input_shape = None
                    
                    print(f"Model input shape from first layer: {input_shape}")
            except Exception as shape_error:
                print(f"Error getting input shape: {shape_error}")
                input_shape = None
            
            # If input shape has 3 dimensions (batch, timesteps, features), it expects sequential data
            if input_shape is not None:
                # Handle different input shape formats
                if isinstance(input_shape, tuple):
                    expects_sequence = len(input_shape) > 2
                elif isinstance(input_shape, list):
                    # For multiple inputs, check the first one
                    first_input_shape = input_shape[0]
                    expects_sequence = len(first_input_shape) > 2
                else:
                    # If we can't determine, fall back to model type logic
                    expects_sequence = False
            
            # Check if model type can be determined from filename
            model_name = model_path.stem.lower()
            if 'cnn_lstm' in model_name:
                model_type = 'cnn_lstm'
                expects_sequence = True
            elif 'lstm' in model_name:
                model_type = 'lstm'
                expects_sequence = True
            elif 'ann' in model_name:
                model_type = 'ann'
                # For ANN, we already determined if it expects sequences above
            else:
                # Enhanced detection for Bidirectional LSTM layers
                has_bidirectional = any(isinstance(layer, tf.keras.layers.Bidirectional) for layer in model.layers)
                has_lstm = any(isinstance(layer, tf.keras.layers.LSTM) for layer in model.layers)
                has_conv = any(isinstance(layer, tf.keras.layers.Conv1D) for layer in model.layers)
                
                # Check for bidirectional LSTM specifically
                if has_bidirectional:
                    # Find all bidirectional layers
                    bidirectional_layers = [layer for layer in model.layers 
                                            if isinstance(layer, tf.keras.layers.Bidirectional)]
                    
                    # Check if any bidirectional layer wraps an LSTM
                    has_bidirectional_lstm = any(
                        isinstance(layer.forward_layer, tf.keras.layers.LSTM) or
                        isinstance(layer.backward_layer, tf.keras.layers.LSTM)
                        for layer in bidirectional_layers
                    )
                    
                    if has_bidirectional_lstm:
                        if has_conv:
                            model_type = 'cnn_lstm'
                        else:
                            model_type = 'lstm'  # Specific type for bidirectional LSTM
                        expects_sequence = True
                elif has_lstm:
                    if has_conv:
                        model_type = 'cnn_lstm'
                    else:
                        model_type = 'lstm'
                    expects_sequence = True
                else:
                    model_type = 'ann'
                    # For ANN, we already determined if it expects sequences above
            
            print(f"Detected model type: {model_type} (Expects sequence: {expects_sequence})")
            return model, model_type, expects_sequence
        except Exception as e:
            print(f"Error loading deep learning model: {e}")
            print(f"Error details: {str(e)}")
            sys.exit(1)
    
    elif model_path.suffix.lower() == '.joblib':
        try:
            print(f"Loading machine learning model from {model_path}")
            model = joblib.load(model_path)
            expects_sequence = False  # ML models don't expect sequences
            
            # Determine if it's XGBoost or Random Forest from filename
            model_name = model_path.stem.lower()
            if 'xgb' in model_name:
                model_type = 'xgb'
            elif 'rf' in model_name or 'randomforest' in model_name:
                model_type = 'rf'
            else:
                # Try to determine from model class
                model_class = str(type(model)).lower()
                if 'xgb' in model_class:
                    model_type = 'xgb'
                elif 'forest' in model_class:
                    model_type = 'rf'
                else:
                    model_type = 'ml_model'  # Generic ML model
                
            print(f"Detected model type: {model_type} (Expects sequence: {expects_sequence})")
            return model, model_type, expects_sequence
        except Exception as e:
            print(f"Error loading machine learning model: {e}")
            sys.exit(1)
    
    # If we get here, let's try loading the file directly as a last resort
    print("File extension not recognized. Attempting direct loading...")
    
    # Try joblib first
    try:
        print("Attempting to load as joblib...")
        model = joblib.load(original_path)
        print("Successfully loaded as joblib!")
        
        # Determine model type
        model_class = str(type(model)).lower()
        if 'forest' in model_class:
            model_type = 'rf'
        elif 'xgb' in model_class:
            model_type = 'xgb'
        else:
            model_type = 'ml_model'
            
        expects_sequence = False
        print(f"Detected model type: {model_type} (Expects sequence: {expects_sequence})")
        return model, model_type, expects_sequence
    except Exception as joblib_err:
        print(f"Failed to load as joblib: {joblib_err}")
    
    # Then try keras
    try:
        print("Attempting to load as keras...")
        model = tf.keras.models.load_model(original_path)
        print("Successfully loaded as keras!")
        
        # Enhanced detection for Bidirectional LSTM layers
        has_bidirectional = any(isinstance(layer, tf.keras.layers.Bidirectional) for layer in model.layers)
        has_lstm = any(isinstance(layer, tf.keras.layers.LSTM) for layer in model.layers)
        has_conv = any(isinstance(layer, tf.keras.layers.Conv1D) for layer in model.layers)
        
        # Check for bidirectional LSTM specifically
        if has_bidirectional:
            # Find all bidirectional layers
            bidirectional_layers = [layer for layer in model.layers 
                                    if isinstance(layer, tf.keras.layers.Bidirectional)]
            
            # Check if any bidirectional layer wraps an LSTM
            has_bidirectional_lstm = any(
                isinstance(layer.forward_layer, tf.keras.layers.LSTM) or
                isinstance(layer.backward_layer, tf.keras.layers.LSTM)
                for layer in bidirectional_layers
            )
            
            if has_bidirectional_lstm:
                if has_conv:
                    model_type = 'cnn_lstm'
                else:
                    model_type = 'lstm'  # Specific type for bidirectional LSTM
                expects_sequence = True
        elif has_lstm:
            if has_conv:
                model_type = 'cnn_lstm'
            else:
                model_type = 'lstm'
            expects_sequence = True
        else:
            model_type = 'ann'
            # Check if ANN expects sequential data
            try:
                input_shape = model.input_shape
                expects_sequence = len(input_shape) > 2  # More than 2 dims suggests sequence
            except:
                expects_sequence = False
            
        print(f"Detected model type: {model_type} (Expects sequence: {expects_sequence})")
        return model, model_type, expects_sequence
    except Exception as keras_err:
        print(f"Failed to load as keras: {keras_err}")
    
    # If we get here, all attempts failed
    print(f"All loading attempts failed for path: {original_path}")
    print("Please check if the file exists and has the correct format.")
    
    if os.path.isdir(original_path):
        print(f"The path is a directory. You might need to specify the model file within this directory.")
        # List files in directory to help user
        print("Files in this directory:")
        for file in os.listdir(original_path):
            print(f"  - {file}")
    
    raise ValueError(f"Could not load model from {original_path}. Please ensure file exists and has correct format (.keras, .h5, or .joblib).")

def prepare_data_for_model(datasets, model_type, expects_sequence=False):
    """
    Prepare data according to model type requirement.
    
    Parameters:
    -----------
    datasets : dict
        Dictionary containing 'train', 'val', and 'test' datasets
    model_type : str
        Type of model ('cnn_lstm', 'lstm', 'ann', 'xgb', 'rf')
    expects_sequence : bool
        Whether the model expects sequential data regardless of model type
    
    Returns:
    --------
    dict
        Processed datasets ready for the specific model type
    """
    processed_datasets = {}
    
    if model_type in ['cnn_lstm', 'lstm'] or expects_sequence:
        # These models can use the windowed datasets directly
        return datasets
    
    # For ANN (that don't expect sequences), XGBoost, and RF, we need to flatten the data
    for key, ds in datasets.items():
        X_list = []
        y_list = []
        
        # Extract and flatten all batches
        for features, labels in ds:
            # Convert to numpy if they're tensors
            if isinstance(features, tf.Tensor):
                features = features.numpy()
            if isinstance(labels, tf.Tensor):
                labels = labels.numpy()
            
            # For each batch item
            for i in range(features.shape[0]):
                # Flatten the features - from (timesteps, features) to (timesteps*features)
                flattened_features = features[i].flatten()
                X_list.append(flattened_features)
                
                # Get the target value (last timestep's label if multiple)
                if len(labels.shape) == 3:  # Shape is [batch, time_steps, features]
                    y_value = labels[i, -1, 0]  # Last timestep, first (only) feature
                else:
                    y_value = labels[i, 0]  # First (only) feature
                
                y_list.append(y_value)
        
        # Convert lists to numpy arrays
        X = np.array(X_list)
        y = np.array(y_list)
        
        processed_datasets[key] = (X, y)
    
    return processed_datasets

def get_predictions(model, dataset, scaler, model_type, expects_sequence=False):
    """
    Get predictions using the appropriate method for the model type.
    
    Parameters:
    -----------
    model : Model or Estimator
        The machine learning model
    dataset : tf.data.Dataset or tuple
        For deep learning: windowed dataset
        For ML models: tuple of (X, y)
    scaler : StandardScaler
        Scaler for the target variable
    model_type : str
        Type of model ('cnn_lstm', 'lstm', 'ann', 'xgb', 'rf')
    expects_sequence : bool
        Whether the model expects sequential data regardless of model type
        
    Returns:
    --------
    tuple
        (predictions, actual_values)
    """
    if model_type in ['cnn_lstm', 'lstm'] or (model_type == 'ann' and expects_sequence):
        try:
            tf.config.run_functions_eagerly(True)
        except:
            pass
        
        all_predictions = []
        all_labels = []
        
        # Use a fixed batch size for prediction to ensure consistency
        for features, labels in dataset:
            batch_predictions = model.predict(features, verbose=0)
            
            # Make sure predictions and labels have compatible shapes before flattening
            if len(batch_predictions.shape) == 3:  # Shape is [batch, time_steps, features]
                batch_predictions = batch_predictions[:, -1, :]  # Get last time step
                
            if len(labels.shape) == 3:  # Shape is [batch, time_steps, features]
                labels = labels[:, -1, :]  # Get last time step
                
            # Handle different types of predictions objects
            if isinstance(batch_predictions, tf.Tensor):
                batch_preds_np = batch_predictions.numpy()
            else:
                # Already a NumPy array
                batch_preds_np = batch_predictions
                
            # Handle different types of label objects
            if isinstance(labels, tf.Tensor):
                labels_np = labels.numpy()
            else:
                labels_np = labels
                
            # Now flatten and add to our collections
            all_predictions.extend(batch_preds_np.flatten())
            all_labels.extend(labels_np.flatten())
        
        try:
            tf.config.run_functions_eagerly(False)
        except:
            pass
            
    elif model_type == 'ann' and not expects_sequence:
        # For ANN that don't expect sequences, we use flattened data
        X, y = dataset
        all_predictions = model.predict(X, verbose=0).flatten()
        all_labels = y
    
    else:  # For XGBoost and Random Forest
        X, y = dataset
        
        # Handle scikit-learn version compatibility issues
        try:
            all_predictions = model.predict(X)
        except AttributeError as e:
            if "object has no attribute 'monotonic_cst'" in str(e):
                print("\nERROR: Scikit-learn version compatibility issue detected.")
                print("This model was trained with a newer version of scikit-learn (1.0.0 or later)")
                print("but you're running an older version that doesn't support 'monotonic_cst'.")
                print("\nPossible solutions:")
                print("1. Upgrade scikit-learn: pip install --upgrade scikit-learn>=1.0.0")
                print("2. Retrain your model with your current scikit-learn version")
                print("\nAttempting to use a workaround for prediction...")
                
                try:
                    # Workaround for RandomForest: manually call predict on estimators
                    if hasattr(model, 'estimators_'):
                        # For RandomForest, average predictions from individual trees
                        tree_preds = []
                        for tree in model.estimators_:
                            try:
                                # Try a direct prediction first without validation
                                tree_pred = tree.predict(X)
                                tree_preds.append(tree_pred)
                                continue
                            except Exception:
                                pass
                                
                            # If direct prediction fails, try a different approach:
                            # Monkey patch the _validate_X_predict method to accept the self argument
                            try:
                                if hasattr(tree, '_validate_X_predict'):
                                    orig_validate = tree._validate_X_predict
                                    
                                    # Create a replacement that accepts both self and X
                                    def new_validate(self, X):
                                        X = np.asarray(X)
                                        if X.ndim != 2:
                                            raise ValueError("Input must be 2D array")
                                        return X
                                    
                                    # Bind the method to the tree instance
                                    import types
                                    tree._validate_X_predict = types.MethodType(new_validate, tree)
                                    
                                    # Try prediction with patched method
                                    tree_pred = tree.predict(X)
                                    tree_preds.append(tree_pred)
                                    
                                    # Restore original method
                                    tree._validate_X_predict = orig_validate
                            except Exception as tree_err:
                                print(f"Error with tree prediction: {tree_err}")
                                continue
                        
                        if tree_preds:
                            all_predictions = np.mean(tree_preds, axis=0)
                            print(f"Successfully used tree-based workaround for prediction with {len(tree_preds)} trees.")
                        else:
                            # Last resort: use a simpler approach
                            print("Attempting simpler prediction approach...")
                            try:
                                # Skip validation by directly accessing tree prediction
                                predictions = []
                                for sample in X:
                                    sample_predictions = []
                                    for tree in model.estimators_:
                                        # Try to directly navigate the tree for prediction
                                        # This is a very simplified approach and might not work for all trees
                                        if hasattr(tree, 'tree_'):
                                            tree_struct = tree.tree_
                                            node_id = 0  # Start at root
                                            while tree_struct.children_left[node_id] != tree_struct.children_right[node_id]:
                                                # Navigate based on feature and threshold
                                                if sample[tree_struct.feature[node_id]] <= tree_struct.threshold[node_id]:
                                                    node_id = tree_struct.children_left[node_id]
                                                else:
                                                    node_id = tree_struct.children_right[node_id]
                                            # We've reached a leaf, get the prediction
                                            sample_predictions.append(tree_struct.value[node_id][0][0])
                                    if sample_predictions:
                                        predictions.append(np.mean(sample_predictions))
                                    else:
                                        # If we couldn't get predictions, use a fallback value
                                        predictions.append(0)
                                all_predictions = np.array(predictions)
                                print("Successfully used direct tree navigation approach.")
                            except Exception as direct_err:
                                print(f"Direct tree navigation failed: {direct_err}")
                                raise ValueError("No compatible workaround available for this model type")
                    
                    # Workaround for XGBoost: use raw model if available
                    elif hasattr(model, 'get_booster'):
                        try:
                            import xgboost as xgb
                            # Convert input to DMatrix format
                            dmatrix = xgb.DMatrix(X)
                            all_predictions = model.get_booster().predict(dmatrix)
                            print("Successfully used XGBoost booster for prediction.")
                        except ImportError:
                            print("XGBoost not available for direct booster access.")
                            raise ValueError("Could not access XGBoost booster")
                    
                    else:
                        raise ValueError("No compatible workaround available for this model type")
                        
                except Exception as workaround_err:
                    print(f"Workaround failed: {workaround_err}")
                    print("Unable to get predictions due to scikit-learn version incompatibility.")
                    # Return dummy predictions as a last resort
                    print("WARNING: Returning dummy predictions. Results will not be accurate!")
                    all_predictions = np.zeros(len(y))
            else:
                # If it's a different error, re-raise it
                raise
                
        all_labels = y
    
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
    
    return all_predictions.flatten(), all_labels.flatten()

@deterministic
def add_time_features(df, datetime_column=None):
    """
    Create cyclical time features from a datetime column or index.
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

#############################################################################
# IGBP Classification and Analysis Functions
#############################################################################

def extract_coordinates_from_files(data_files):
    """
    Extract coordinates from data files for IGBP classification.
    
    Parameters:
    -----------
    data_files : list
        List of Path objects for data files
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with file_id, latitude, and longitude columns
    """
    coords_data = []
    
    for file_idx, file_path in enumerate(data_files):
        try:
            # Read the file
            df = pd.read_csv(file_path)
            
            # Look for latitude and longitude columns
            lat_cols = [col for col in df.columns if 'lat' in col.lower()]
            lon_cols = [col for col in df.columns if 'lon' in col.lower()]
            
            if lat_cols and lon_cols:
                lat_col = lat_cols[0]
                lon_col = lon_cols[0]
                
                # Extract first non-null coordinate
                lat = df[lat_col].dropna().iloc[0] if not df[lat_col].dropna().empty else None
                lon = df[lon_col].dropna().iloc[0] if not df[lon_col].dropna().empty else None
                
                if lat is not None and lon is not None:
                    coords_data.append({
                        'file_id': file_idx,
                        'file_name': file_path.name,
                        'latitude': float(lat),
                        'longitude': float(lon)
                    })
            else:
                print(f"No coordinate columns found in {file_path.name}")
        except Exception as e:
            print(f"Error extracting coordinates from {file_path.name}: {e}")
    
    # Create DataFrame
    coords_df = pd.DataFrame(coords_data)
    print(f"Extracted coordinates from {len(coords_df)} files")
    
    return coords_df

def get_predictions_by_igbp(predictions, actuals, file_igbp_mapping):
    """
    Group predictions and actuals by IGBP class.
    
    Parameters:
    -----------
    predictions : numpy.ndarray
        Model predictions
    actuals : numpy.ndarray
        Actual values
    file_igbp_mapping : dict
        Mapping from file_id to IGBP class
        
    Returns:
    --------
    dict
        Results grouped by IGBP class
    """
    # Create a mapping of file indices to sample indices
    # This is a simplified approach - in a real scenario, you would need
    # to track which segments and predictions come from which files
    
    # For this example, we'll assume predictions are distributed evenly
    # across files in the same order as the files were processed
    n_files = len(file_igbp_mapping)
    samples_per_file = len(predictions) // n_files
    
    # Group predictions by IGBP class
    results_by_igbp = {}
    
    for file_idx, igbp_class in file_igbp_mapping.items():
        # Calculate sample indices for this file
        start_idx = file_idx * samples_per_file
        end_idx = start_idx + samples_per_file
        
        # Ensure indices are within bounds
        if start_idx >= len(predictions):
            continue
        
        end_idx = min(end_idx, len(predictions))
        
        file_predictions = predictions[start_idx:end_idx]
        file_actuals = actuals[start_idx:end_idx]
        
        if igbp_class not in results_by_igbp:
            results_by_igbp[igbp_class] = {
                'predictions': [],
                'actual': []
            }
        
        results_by_igbp[igbp_class]['predictions'].extend(file_predictions)
        results_by_igbp[igbp_class]['actual'].extend(file_actuals)
    
    # Convert lists to arrays
    for igbp_class in results_by_igbp:
        results_by_igbp[igbp_class]['predictions'] = np.array(results_by_igbp[igbp_class]['predictions'])
        results_by_igbp[igbp_class]['actual'] = np.array(results_by_igbp[igbp_class]['actual'])
    
    return results_by_igbp

def analyze_forest_type_performance(results_by_igbp, plot_dir, model_type, min_samples=10):
    """
    Analyze model performance by IGBP forest type.
    
    Parameters:
    -----------
    results_by_igbp : dict
        Results grouped by IGBP forest type
    plot_dir : Path
        Directory to save plots
    model_type : str
        Type of model for labeling plots
    min_samples : int
        Minimum number of samples required for analysis
        
    Returns:
    --------
    pandas.DataFrame
        Performance metrics by forest type
    """
    # Create directory for forest type plots
    forest_type_dir = plot_dir / f'{model_type}/forest_types'
    forest_type_dir.mkdir(parents=True, exist_ok=True)
    
    # Define forest type classes (IGBP classes 1-5 and 8)
    forest_type_classes = [
        'Evergreen Needleleaf Forest',
        'Evergreen Broadleaf Forest',
        'Deciduous Needleleaf Forest',
        'Deciduous Broadleaf Forest',
        'Mixed Forest',
        'Woody Savannas'
    ]
    
    # Calculate metrics for each IGBP class
    metrics = {}
    for igbp_class, data in results_by_igbp.items():
        sample_count = len(data['predictions'])
        if sample_count < min_samples:
            print(f"Skipping {igbp_class} (insufficient data: {sample_count} samples)")
            continue
        
        is_forest = igbp_class in forest_type_classes
        
        # Calculate performance metrics
        r2 = r2_score(data['actual'], data['predictions'])
        rmse = np.sqrt(mean_squared_error(data['actual'], data['predictions']))
        mae = mean_absolute_error(data['actual'], data['predictions'])
        
        metrics[igbp_class] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'sample_count': sample_count,
            'is_forest': is_forest
        }
        
        # Create scatter plot
        plt.figure(figsize=(10, 10))
        plt.scatter(data['actual'], data['predictions'], alpha=0.5)
        plt.xlabel('True Values [Sap Velocity]')
        plt.ylabel('Predictions [Sap Velocity]')
        plt.axis('equal')
        plt.axis('square')
        max_val = max(plt.xlim()[1], plt.ylim()[1])
        plt.xlim([0, max_val])
        plt.ylim([0, max_val])
        plt.plot([-100, 100], [-100, 100])
        plt.title(f'{model_type.upper()} - IGBP Class: {igbp_class}\nR² = {r2:.3f}, RMSE = {rmse:.3f}, n = {sample_count}')
        plt.savefig(forest_type_dir / f'igbp_{igbp_class.replace(" ", "_").replace("/", "_")}.png')
        plt.close()
    
    if not metrics:
        print("No IGBP classes had sufficient data for analysis")
        empty_metrics = pd.DataFrame(columns=['R²', 'RMSE', 'MAE', 'Samples', 'Is Forest'])
        model_dir = plot_dir / model_type
        model_dir.mkdir(exist_ok=True)
        empty_metrics.to_csv(model_dir / 'igbp_performance_metrics.csv')
        return empty_metrics
    
    # Create summary plots for forest types
    create_forest_type_summary_plots(metrics, plot_dir, model_type)
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame.from_dict(
        {k: {
            'R²': v['r2'], 
            'RMSE': v['rmse'], 
            'MAE': v['mae'], 
            'Samples': v['sample_count'],
            'Is Forest': v['is_forest']
        } for k, v in metrics.items()}, 
        orient='index'
    )
    metrics_df = metrics_df.sort_values('R²', ascending=False)
    model_dir = plot_dir / model_type
    model_dir.mkdir(exist_ok=True)
    
    # Save metrics to CSV
    metrics_file = model_dir / 'igbp_performance_metrics.csv'
    metrics_df.to_csv(metrics_file)
    print(f"Saved forest type metrics to {metrics_file}")
    
    return metrics_df

def create_forest_type_summary_plots(metrics, plot_dir, model_type):
    """Create summary plots for forest type performance"""
    # Create model-specific directory
    model_dir = plot_dir / model_type
    model_dir.mkdir(exist_ok=True)
    
    # Filter for forest types only
    forest_metrics = {k: v for k, v in metrics.items() if v['is_forest']}
    
    if not forest_metrics:
        print("No forest types with sufficient data for analysis")
        return
    
    # Create R² comparison plot
    fig, ax = plt.subplots(figsize=(12, 8))  # Create figure and axes explicitly
    forest_types = list(forest_metrics.keys())
    r2_values = [m['r2'] for m in forest_metrics.values()]
    sample_counts = [m['sample_count'] for m in forest_metrics.values()]
    
    # Sort by R² value
    sorted_indices = np.argsort(r2_values)
    sorted_forest_types = [forest_types[i] for i in sorted_indices]
    sorted_r2_values = [r2_values[i] for i in sorted_indices]
    sorted_sample_counts = [sample_counts[i] for i in sorted_indices]
    
    # Create bar chart
    colors = plt.cm.viridis(np.array(sorted_sample_counts)/max(sorted_sample_counts))
    bars = ax.barh(sorted_forest_types, sorted_r2_values, color=colors)
    
    ax.set_xlabel('R² Score')
    ax.set_title(f'{model_type.upper()} Model Performance by IGBP Forest Type')
    
    # Add value labels
    for i, (bar, value, count) in enumerate(zip(bars, sorted_r2_values, sorted_sample_counts)):
        ax.text(max(0.05, bar.get_width() - 0.1), bar.get_y() + bar.get_height()/2, 
                f'R² = {value:.3f} (n = {count})', 
                ha='right', va='center', color='white', fontweight='bold')
    
    # Add colorbar to indicate sample size
    norm = plt.Normalize(min(sample_counts), max(sample_counts))
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)  # Pass ax explicitly
    cbar.set_label('Sample Count')
    
    plt.tight_layout()
    plt.savefig(model_dir / 'forest_type_r2_comparison.png')
    plt.close(fig)
    
    # Create RMSE comparison plot
    fig, ax = plt.subplots(figsize=(12, 8))  # Create figure and axes explicitly
    rmse_values = [m['rmse'] for m in forest_metrics.values()]
    
    # Sort by RMSE value (lower is better)
    sorted_indices = np.argsort(rmse_values)[::-1]  # Reverse to show best at top
    sorted_forest_types = [forest_types[i] for i in sorted_indices]
    sorted_rmse_values = [rmse_values[i] for i in sorted_indices]
    sorted_sample_counts = [sample_counts[i] for i in sorted_indices]
    
    # Create bar chart
    colors = plt.cm.viridis(np.array(sorted_sample_counts)/max(sorted_sample_counts))
    bars = ax.barh(sorted_forest_types, sorted_rmse_values, color=colors)
    
    ax.set_xlabel('RMSE')
    ax.set_title(f'{model_type.upper()} Model Error by IGBP Forest Type')
    
    # Add value labels
    for i, (bar, value, count) in enumerate(zip(bars, sorted_rmse_values, sorted_sample_counts)):
        ax.text(max(0.05, bar.get_width() - 0.1), bar.get_y() + bar.get_height()/2, 
                f'RMSE = {value:.3f} (n = {count})', 
                ha='right', va='center', color='white', fontweight='bold')
    
    # Add colorbar
    norm = plt.Normalize(min(sample_counts), max(sample_counts))
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)  # Pass ax explicitly
    cbar.set_label('Sample Count')
    
    plt.tight_layout()
    plt.savefig(model_dir / 'forest_type_rmse_comparison.png')
    plt.close(fig)

#############################################################################
# Main Function
#############################################################################

@deterministic
def main(model_path=None):
    """
    Main function to evaluate a pre-trained model with forest type analysis.
    Supports CNN-LSTM, LSTM, ANN, XGBoost, and Random Forest models.
    
    Parameters:
    -----------
    model_path : str, optional
        Path to the model file. If None, uses a default path.
    """
    # Set parameters
    RANDOM_SEED = get_seed()
    BATCH_SIZE = 32
    
    # Define window parameters (must match what the model was trained with)
    INPUT_WIDTH = 8   # Use 8 time steps as input
    LABEL_WIDTH = 1   # Predict 1 time step ahead
    SHIFT = 1         # Predict 1 step ahead
    EXCLUDE_LABEL = True  # Exclude labels in input features
    
    # Path to pre-trained model
    if model_path is None:
        # Default model path
        MODEL_PATH = './outputs/models/cnn_lstm_regression/dl_regression_model_20250421_184644.keras'
    else:
        MODEL_PATH = model_path
    
    # Set deterministic rendering for matplotlib
    import matplotlib as mpl
    mpl.rcParams['agg.path.chunksize'] = 10000
    np.random.seed(RANDOM_SEED)
    
    # Load pre-trained model
    print(f"Loading pre-trained model from {MODEL_PATH}")
    model, model_type, expects_sequence = load_pretrained_model(MODEL_PATH)
    print(f"Starting model evaluation with IGBP forest type analysis using {model_type.upper()} model")
    print(f"Model expects sequential data: {expects_sequence}")
    
    # Load data files
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
    
    # Initialize Earth Engine and get coordinates
    gee_enabled = initialize_earth_engine()
    
    # Extract coordinates and get IGBP classifications if GEE is enabled
    if gee_enabled:
        print("Extracting coordinates from data files...")
        coordinates_df = extract_coordinates_from_files(data_list)
        
        if not coordinates_df.empty:
            print("Getting IGBP classifications from MODIS data...")
            coordinates_df = batch_process_coordinates(coordinates_df, "igbp_cache.csv")
            
            # Create mapping from file_id to IGBP class
            file_to_igbp = dict(zip(coordinates_df['file_id'], coordinates_df['igbp_class']))
            
            # Save this mapping for later analysis
            coordinates_df.to_csv('./outputs/metadata/file_igbp_mapping.csv', index=False)
        else:
            print("No coordinates found, skipping IGBP classification")
            gee_enabled = False
    
    # Process data files
    all_segments = []
    # Define the columns we want to use
    used_cols = ['sap_velocity', 'ext_rad', 'sw_in', 'ta', 'ws', 'vpd','ppfd_in', 
                 'mean_annual_temp', 'mean_annual_precip', 'Day sin', 'Week sin', 'Month sin', 'Year sin']
    
    # Sort data files for deterministic processing order
    data_list = sorted(data_list)

    # Track which segments come from which files
    segment_file_mapping = []
    
    # Process each data file
    for file_idx, data_file in enumerate(data_list):
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
            
            # Clean data
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
            
            # Track which file these segments came from
            for _ in range(len(segments)):
                segment_file_mapping.append(file_idx)
            
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
            segment_file_mapping = [0]  # All from the same file
        except:
            print("ERROR: Could not process any data.")
            return (0, 0), (0, 0)  # Return dummy values

    print(f"Total segments collected: {len(all_segments)}")
    
    # Now standardize the data
    all_data = pd.concat(all_segments)

    # Load existing scalers instead of creating new ones
    try:
        feature_scaler_path = Path('./outputs/scalers/feature_scaler.pkl')
        label_scaler_path = Path('./outputs/scalers/label_scaler.pkl')
        
        feature_scaler = joblib.load(feature_scaler_path)
        label_scaler = joblib.load(label_scaler_path)
        print("Loaded existing scalers")
    except:
        print("Creating new scalers (warning: this may affect model performance)")
        # Create new scalers
        feature_scaler = StandardScaler()
        label_scaler = StandardScaler()

        # Get feature columns
        numerical_features = [col for col in all_data.columns if col != 'sap_velocity']

        # Fit scalers
        feature_scaler.fit(all_data[numerical_features])
        label_scaler.fit(all_data[['sap_velocity']])
        
        # Save the new scalers
        Path('./outputs/scalers').mkdir(parents=True, exist_ok=True)
        joblib.dump(feature_scaler, './outputs/scalers/feature_scaler.pkl')
        joblib.dump(label_scaler, './outputs/scalers/label_scaler.pkl')
    
    # Get numerical features
    numerical_features = [col for col in all_data.columns if col != 'sap_velocity']
    
    # Apply scaling to each segment
    scaled_segments = []
    for segment in all_segments:
        segment_copy = segment.copy()
        # Transform numerical features
        segment_copy[numerical_features] = feature_scaler.transform(segment[numerical_features])
        segment_copy['sap_velocity'] = label_scaler.transform(segment[['sap_velocity']])
        scaled_segments.append(segment_copy)
    
    # Create windowed datasets
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
        
        # Process data based on model type and whether it expects sequential data
        datasets = prepare_data_for_model(datasets, model_type, expects_sequence)
        
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
        
        # Process data based on model type and whether it expects sequential data
        datasets = prepare_data_for_model(datasets, model_type, expects_sequence)
    
    test_ds = datasets['test']
    train_ds = datasets['train']

    # Get predictions using pre-trained model
    print("Generating predictions...")
    test_predictions, test_labels_actual = get_predictions(model, test_ds, label_scaler, model_type, expects_sequence)
    train_predictions, train_labels_actual = get_predictions(model, train_ds, label_scaler, model_type, expects_sequence)
    
    # Save predictions to file for later use
    predictions_dir = Path('./outputs/predictions')
    predictions_dir.mkdir(exist_ok=True)
    
    # Create model-specific directory
    model_pred_dir = predictions_dir / model_type
    model_pred_dir.mkdir(exist_ok=True)
    
    # Save test predictions
    predictions_df = pd.DataFrame({
        'actual': test_labels_actual,
        'predicted': test_predictions
    })
    predictions_df.to_csv(model_pred_dir / 'test_predictions.csv', index=False)
    
    # Create plots directory
    plot_dir = Path('./plots')
    plot_dir.mkdir(exist_ok=True)
    
    # Create model-specific plot directory
    model_plot_dir = plot_dir / model_type
    model_plot_dir.mkdir(exist_ok=True)
    
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
    plt.title(f'{model_type.upper()} Model: Test Predictions vs Actual')
    plt.savefig(model_plot_dir / f'{model_type}_test_predictions.png')
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
    
    print(f"\n{model_type.upper()} Model Evaluation:")
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
    
    # Save metrics to file
    metrics_df = pd.DataFrame({
        'Metric': ['R² (test)', 'R² (train)', 'RMSE (test)', 'RMSE (train)', 'MAE (test)', 'MAE (train)'],
        'Value': [r2, r2_train, test_rmse, train_rmse, test_mae, train_mae]
    })
    metrics_df.to_csv(model_pred_dir / 'model_metrics.csv', index=False)
    
    # IGBP Forest Type Analysis
    try:
        print(f"\nPerforming IGBP forest type analysis for {model_type.upper()} model...")
        
        # Check if GEE is enabled and we have forest type mapping
        if not gee_enabled:
            print("Google Earth Engine was not initialized, looking for cached mapping...")
            
            # Try to load the mapping from a previously saved file
            try:
                mapping_file = Path('./outputs/metadata/file_igbp_mapping.csv')
                if mapping_file.exists():
                    print(f"Found existing IGBP mapping file: {mapping_file}")
                    mapping_df = pd.read_csv(mapping_file)
                    file_to_igbp = dict(zip(mapping_df['file_id'], mapping_df['igbp_class']))
                    gee_enabled = True  # Enable forest type analysis
                else:
                    # For testing purposes: create dummy forest type data if no real data
                    print("No IGBP mapping found - creating test data for demonstration")
                    forest_types = [
                        "Evergreen Needleleaf Forest", 
                        "Evergreen Broadleaf Forest",
                        "Deciduous Needleleaf Forest",
                        "Deciduous Broadleaf Forest",
                        "Mixed Forest"
                    ]
                    
                    # Create a simple mapping of file indices to forest types
                    num_files = len(data_list) if data_list else 10
                    file_to_igbp = {i: forest_types[i % len(forest_types)] for i in range(num_files)}
                    gee_enabled = True  # Enable forest type analysis with test data
                    
                    # Save this test mapping
                    Path('./outputs/metadata').mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({
                        'file_id': list(file_to_igbp.keys()),
                        'igbp_class': list(file_to_igbp.values())
                    }).to_csv('./outputs/metadata/file_igbp_mapping_test.csv', index=False)
            except Exception as e:
                print(f"Error loading IGBP mapping: {e}")
        
        if gee_enabled and 'file_to_igbp' in locals():
            # Group predictions by IGBP forest type
            results_by_igbp = get_predictions_by_igbp(
                test_predictions, test_labels_actual, file_to_igbp
            )
            
            # Save results by forest type for later use
            igbp_results_dir = model_pred_dir / 'igbp_results'
            igbp_results_dir.mkdir(exist_ok=True)
            
            print(f"Saving forest type results to {igbp_results_dir}")
            
            for igbp_class, data in results_by_igbp.items():
                igbp_df = pd.DataFrame({
                    'actual': data['actual'],
                    'predicted': data['predictions']
                })
                output_file = igbp_results_dir / f'{igbp_class.replace(" ", "_").replace("/", "_")}.csv'
                igbp_df.to_csv(output_file, index=False)
                print(f"Saved {len(igbp_df)} samples for {igbp_class} to {output_file}")
            
            # Analyze performance by forest type
            print(f"Analyzing performance by forest type...")
            forest_metrics = analyze_forest_type_performance(
                results_by_igbp, plot_dir, model_type, min_samples=10
            )
            
            # Print performance by IGBP forest type
            print(f"\n{model_type.upper()} Performance by IGBP Forest Type:")
            print("===============================")
            for idx, row in forest_metrics.iterrows():
                if row['Is Forest']:
                    print(f"{idx}: R² = {row['R²']:.3f}, RMSE = {row['RMSE']:.3f}, n = {int(row['Samples'])}")
                    
            print(f"\nForest type analysis results saved to {plot_dir / model_type}")
        else:
            print("Skipping forest type analysis: no IGBP classification data available")
    except Exception as e:
        print(f"Error in forest type analysis: {e}")
        import traceback
        traceback.print_exc()
    return test_rmse, model_type


def evaluate_multiple_models(model_paths):
    """
    Evaluate multiple models and compare their performance.
    
    Parameters:
    -----------
    model_paths : list of str
        Paths to model files to evaluate
    """
    results = {}
    
    for model_path in model_paths:
        print(f"\n{'='*50}")
        print(f"Evaluating model: {model_path}")
        print(f"{'='*50}")
        
        try:
            rmse, model_type = main(model_path)
            results[model_path] = {
                'rmse': rmse,
                'model_type': model_type
            }
        except Exception as e:
            print(f"Error evaluating model {model_path}: {e}")
            import traceback
            traceback.print_exc()
            results[model_path] = {
                'rmse': None,
                'model_type': 'error',
                'error': str(e)
            }
    
    # Create comparison table
    results_df = pd.DataFrame({
        'Model File': list(results.keys()),
        'Model Type': [r['model_type'] for r in results.values()],
        'RMSE': [r['rmse'] for r in results.values()]
    })
    
    # Sort by RMSE (best first)
    results_df = results_df.sort_values('RMSE')
    
    # Save comparison
    results_dir = Path('./outputs/comparisons')
    results_dir.mkdir(exist_ok=True)
    
    results_df.to_csv(results_dir / 'model_comparison.csv', index=False)
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Filter valid results
    valid_results = results_df[~results_df['RMSE'].isna()]
    
    colors = {
        'cnn_lstm': 'blue',
        'lstm': 'green',
        'ann': 'red',
        'xgb': 'purple',
        'rf': 'orange',
        'ml_model': 'gray'
    }
    
    # Create list of colors based on model type
    bar_colors = [colors.get(model_type, 'black') for model_type in valid_results['Model Type']]
    
    # Create bar chart
    bars = plt.barh(valid_results['Model File'], valid_results['RMSE'], color=bar_colors)
    
    # Add labels
    for bar, rmse, model_type in zip(bars, valid_results['RMSE'], valid_results['Model Type']):
        plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                f'RMSE: {rmse:.3f}\nType: {model_type}', 
                va='center')
    
    plt.xlabel('RMSE (lower is better)')
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    
    # Create color legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=model_type) 
                      for model_type, color in colors.items() 
                      if model_type in valid_results['Model Type'].values]
    plt.legend(handles=legend_elements, title='Model Types')
    
    plt.savefig(results_dir / 'model_comparison.png')
    plt.close()
    
    print("\nModel Comparison Results:")
    print(results_df)
    print(f"\nComparison saved to {results_dir}")
    
    return results_df


if __name__ == "__main__":
    # Check if model path is provided as command line argument
    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        # Evaluate all models listed
        print("\nRunning evaluation on all model types")
        model_paths = []
        
        # Try to find models in common directories
        model_dirs = [
            #'./outputs/models/cnn_lstm_regression',
            './outputs/models/lstm_regression',
            #'./outputs/models/ann_regression',
            #'./outputs/models/xgb',
            #'./outputs/models/rf_regression',
        ]
        
        # Find all model files
        for model_dir in model_dirs:
            dir_path = Path(model_dir)
            if dir_path.exists():
                # Find Keras models
                keras_models = list(dir_path.glob('*.keras')) + list(dir_path.glob('*.h5'))
                # Find joblib models
                joblib_models = list(dir_path.glob('*.joblib'))
                
                model_paths.extend([str(m) for m in keras_models + joblib_models])
        
        if model_paths:
            print(f"Found {len(model_paths)} models to evaluate:")
            for path in model_paths:
                print(f"  - {path}")
            evaluate_multiple_models(model_paths)
        else:
            print("No models found in expected directories. Please specify a model path.")
            
    elif len(sys.argv) > 1:
        # Evaluate a single model
        model_path = sys.argv[1]
        print(f"\nRunning model evaluation with IGBP forest type analysis using model: {model_path}")
        main(model_path)
    elif len(sys.argv) == 1:
        # No arguments, run with default model
        print("\nRunning model evaluation with IGBP forest type analysis using default model path")
        main()
        
    # Example model paths for reference:
    # model_paths = [
    #     './outputs/models/cnn_lstm_regression/dl_regression_model.keras',
    #     './outputs/models/lstm/lstm_model.keras',
    #     './outputs/models/ann/ann_model.keras',
    #     './outputs/models/xgb/xgb_model.joblib',
    #     './outputs/models/rf/rf_model.joblib'
    # ]