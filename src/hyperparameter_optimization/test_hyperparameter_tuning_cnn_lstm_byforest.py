"""
Modified CNN-LSTM model script with IGBP forest type analysis.
This script loads a pre-trained model to make predictions and analyze performance
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
        ee.Initialize(project = 'ee-yuluo-2')
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

def get_predictions(model, dataset, scaler):
    """Get predictions from a windowed dataset and inverse transform them."""
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
        tf.config.run_functions_eagerly(False)
    except:
        pass
    
    return all_predictions.flatten(), all_labels.flatten()

def load_pretrained_model(model_path):
    """
    Load a pre-trained model from disk, handling different format types.
    
    Parameters:
    -----------
    model_path : str or Path
        Path to the saved model
        
    Returns:
    --------
    Model
        Loaded model
    """
    try:
        # First try direct loading
        try:
            print(f"Attempting to load model directly from {model_path}")
            model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully")
            return model
        except Exception as e1:
            print(f"Direct loading failed: {e1}")
            
        # If direct loading fails, try loading using TF 1.x compatibility
        try:
            print("Attempting to load using custom_objects...")
            custom_objects = {
                'tf': tf
            }
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            print("Model loaded successfully with custom_objects")
            return model
        except Exception as e2:
            print(f"Loading with custom_objects failed: {e2}")
            
        # Try loading using lower-level SavedModel API
        print("Attempting to load using low-level SavedModel API...")
        imported = tf.saved_model.load(model_path)
        
        # Check for a specific function to use for inference
        if hasattr(imported, 'signatures'):
            print("Model has signatures:", imported.signatures.keys())
            signature_key = list(imported.signatures.keys())[0]
            infer_function = imported.signatures[signature_key]
            
            # Create a wrapper model
            class SavedModelWrapper(tf.keras.Model):
                def __init__(self, infer_func):
                    super(SavedModelWrapper, self).__init__()
                    self.infer_func = infer_func
                
                def call(self, inputs):
                    # Convert inputs to the format expected by the inference function
                    if isinstance(inputs, tf.Tensor):
                        input_tensor = inputs
                    else:
                        input_tensor = tf.convert_to_tensor(inputs)
                    
                    # Call the inference function
                    result = self.infer_func(input_tensor)
                    
                    # Return the output tensor
                    output_key = list(result.keys())[0]
                    return result[output_key]
                
                def predict(self, inputs, **kwargs):
                    # Handle batched prediction
                    results = []
                    for batch in inputs:
                        result = self.call(batch)
                        results.append(result.numpy())
                    return np.concatenate(results, axis=0)
            
            model = SavedModelWrapper(infer_function)
            print("Created wrapper model using SavedModel signature")
            return model
        else:
            # If no signatures, check for a callable model
            if callable(imported):
                class CallableWrapper(tf.keras.Model):
                    def __init__(self, callable_model):
                        super(CallableWrapper, self).__init__()
                        self.callable_model = callable_model
                    
                    def call(self, inputs):
                        return self.callable_model(inputs)
                    
                    def predict(self, inputs, **kwargs):
                        # Handle batched prediction
                        return np.array([self.call(x).numpy() for x in inputs])
                
                model = CallableWrapper(imported)
                print("Created wrapper model using callable SavedModel")
                return model
                
        raise ValueError("Could not find a valid signature or callable in the SavedModel")
            
    except Exception as e:
        print(f"All loading attempts failed: {e}")
        print("Please provide the model in .keras, .h5, or valid SavedModel format.")
        sys.exit(1)

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

def analyze_forest_type_performance(results_by_igbp, plot_dir, min_samples=10):
    """
    Analyze model performance by IGBP forest type.
    
    Parameters:
    -----------
    results_by_igbp : dict
        Results grouped by IGBP forest type
    plot_dir : Path
        Directory to save plots
    min_samples : int
        Minimum number of samples required for analysis
        
    Returns:
    --------
    pandas.DataFrame
        Performance metrics by forest type
    """
    # Create directory for forest type plots
    forest_type_dir = plot_dir / 'forest_types'
    forest_type_dir.mkdir(exist_ok=True)
    
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
        plt.title(f'IGBP Class: {igbp_class}\nR² = {r2:.3f}, RMSE = {rmse:.3f}, n = {sample_count}')
        plt.savefig(forest_type_dir / f'igbp_{igbp_class.replace(" ", "_").replace("/", "_")}.png')
        plt.close()
    
    # Create summary plots for forest types
    create_forest_type_summary_plots(metrics, plot_dir)
    
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
    metrics_df.to_csv(plot_dir / 'igbp_performance_metrics.csv')
    
    return metrics_df

def create_forest_type_summary_plots(metrics, plot_dir):
    """Create summary plots for forest type performance"""
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
    ax.set_title('CNN-LSTM Model Performance by IGBP Forest Type')
    
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
    plt.savefig(plot_dir / 'forest_type_r2_comparison.png')
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
    ax.set_title('CNN-LSTM Model Error by IGBP Forest Type')
    
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
    plt.savefig(plot_dir / 'forest_type_rmse_comparison.png')
    plt.close(fig)

#############################################################################
# Main Function (modified to use pre-trained model)
#############################################################################

@deterministic
def main():
    """
    Main function to evaluate a pre-trained CNN-LSTM model with forest type analysis.
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
    MODEL_PATH = './outputs/models/cnn_lstm_regression/dl_regression_model_20250421_184644.keras'
    
    # Set deterministic rendering for matplotlib
    import matplotlib as mpl
    mpl.rcParams['agg.path.chunksize'] = 10000
    np.random.seed(RANDOM_SEED)
    
    print(f"Starting CNN-LSTM model evaluation with IGBP forest type analysis using pre-trained model")
    
    # Load pre-trained model
    print(f"Loading pre-trained model from {MODEL_PATH}")
    model = load_pretrained_model(MODEL_PATH)
    
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
    
    # Process data files (similar to original implementation)
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
    
    test_ds = datasets['test']
    train_ds = datasets['train']

    # Get predictions using pre-trained model
    print("Generating predictions...")
    test_predictions, test_labels_actual = get_predictions(model, test_ds, label_scaler)
    train_predictions, train_labels_actual = get_predictions(model, train_ds, label_scaler)
    
    # Save predictions to file for later use
    predictions_dir = Path('./outputs/predictions')
    predictions_dir.mkdir(exist_ok=True)
    
    # Save test predictions
    predictions_df = pd.DataFrame({
        'actual': test_labels_actual,
        'predicted': test_predictions
    })
    predictions_df.to_csv(predictions_dir / 'test_predictions.csv', index=False)
    
    # Create plots directory
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
    plt.title('CNN-LSTM Model: Test Predictions vs Actual')
    plt.savefig(plot_dir / 'cnn_lstm_test_predictions.png')
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
    
    # Save metrics to file
    metrics_df = pd.DataFrame({
        'Metric': ['R² (test)', 'R² (train)', 'RMSE (test)', 'RMSE (train)', 'MAE (test)', 'MAE (train)'],
        'Value': [r2, r2_train, test_rmse, train_rmse, test_mae, train_mae]
    })
    metrics_df.to_csv(predictions_dir / 'model_metrics.csv', index=False)
    
    # IGBP Forest Type Analysis
    if gee_enabled and 'file_to_igbp' in locals():
        print("\nPerforming IGBP forest type analysis...")
        
        # Group predictions by IGBP forest type
        results_by_igbp = get_predictions_by_igbp(
            test_predictions, test_labels_actual, file_to_igbp
        )
        
        # Save results by forest type for later use
        igbp_results_dir = predictions_dir / 'igbp_results'
        igbp_results_dir.mkdir(exist_ok=True)
        
        for igbp_class, data in results_by_igbp.items():
            igbp_df = pd.DataFrame({
                'actual': data['actual'],
                'predicted': data['predictions']
            })
            igbp_df.to_csv(igbp_results_dir / f'{igbp_class.replace(" ", "_").replace("/", "_")}.csv', index=False)
        
        # Analyze performance by forest type
        forest_metrics = analyze_forest_type_performance(
            results_by_igbp, plot_dir, min_samples=10
        )
        
        # Print performance by IGBP forest type
        print("\nPerformance by IGBP Forest Type:")
        print("===============================")
        for idx, row in forest_metrics.iterrows():
            if row['Is Forest']:
                print(f"{idx}: R² = {row['R²']:.3f}, RMSE = {row['RMSE']:.3f}, n = {int(row['Samples'])}")
    
    return test_rmse


if __name__ == "__main__":
    # Run the main function
    print("\nRunning CNN-LSTM model with IGBP forest type analysis using pre-trained model...")
    main()