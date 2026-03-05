"""
ANN model with spatial cross-validation approach for site-based prediction.
Implements group-based spatial cross-validation with proper time windowing.
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

# Import the time series windowing modules
from src.hyperparameter_optimization.timeseries_processor1 import (
    TimeSeriesSegmenter, WindowGenerator, TemporalWindowGenerator
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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import GroupKFold, train_test_split
from typing import Union, List, Tuple, Dict, Any
from collections import Counter

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



@deterministic
def create_windows_from_segments(
    segments: List[pd.DataFrame],
    input_width: int,
    label_width: int,
    shift: int,
    label_columns: List[str] = None,
    exclude_targets_from_features: bool = True,
    exclude_labels_from_inputs: bool = True
) -> List[Tuple[tf.Tensor, tf.Tensor]]:
    """
    Create windows from multiple segments without splitting them.
    Each segment is processed independently to maintain time series integrity.
    
    Parameters:
    -----------
    segments : list
        List of DataFrame segments
    input_width : int
        Width of input window
    label_width : int
        Width of label window
    shift : int
        Offset between input and label
    label_columns : list of str, optional
        Columns to use as labels
    exclude_targets_from_features : bool
        Whether to exclude label columns from inputs
    exclude_labels_from_inputs : bool
        Whether to exclude labels from inputs 
        
    Returns:
    --------
    list
        List of (inputs, labels) windows
    """
    all_windows = []
    
    # Process each segment independently
    for segment in segments:
        # Skip segments that are too short
        min_segment_length = input_width + shift
        if len(segment) < min_segment_length:
            continue
        
        # Create a WindowGenerator for this segment
        window_gen = WindowGenerator(
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            data_df=segment,
            label_columns=label_columns,
            batch_size=1,  # Use batch size of 1 to extract individual windows
            shuffle=False,  # Don't shuffle to maintain temporal order
            exclude_targets_from_features=exclude_targets_from_features,
            exclude_labels_from_inputs=exclude_labels_from_inputs
        )
        
        # Collect all windows from this segment
        segment_ds = window_gen.dataset
        for inputs, labels in segment_ds:
            all_windows.append((inputs, labels))
    
    return all_windows

@deterministic
def windows_to_tf_dataset(windows: List[Tuple[tf.Tensor, tf.Tensor]], batch_size: int = 32, shuffle: bool = True) -> tf.data.Dataset:
    """
    Convert a list of windows to a TensorFlow dataset.
    
    Parameters:
    -----------
    windows : list
        List of (inputs, labels) windows
    batch_size : int
        Batch size for the dataset
    shuffle : bool
        Whether to shuffle the dataset
        
    Returns:
    --------
    tf.data.Dataset
        TensorFlow dataset of windows
    """
    # Early exit if empty
    if not windows:
        return None
        
    # Extract inputs and labels from complete windows
    inputs = tf.concat([w[0] for w in windows], axis=0)
    labels = tf.concat([w[1] for w in windows], axis=0)
    
    # Create dataset with complete windows
    ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
    
    # Apply shuffling if requested
    if shuffle:
        ds = ds.shuffle(buffer_size=10000, seed=get_seed())
    
    # Apply batching
    ds = ds.batch(batch_size)
    
    # Apply deterministic options in a version-compatible way
    try:
        options = tf.data.Options()
        options.experimental_deterministic = True
        try:
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        except:
            pass
        ds = ds.with_options(options)
    except:
        pass
        
    return ds

@deterministic
def create_windowed_nn(input_shape, output_shape, n_layers, units, dropout_rate):
    """
    Create a neural network model that takes a time window as input,
    with explicit seeds for all random operations.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of the input data (timesteps, features)
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
    # Input shape is now (timesteps, features) instead of just (features)
    inputs = keras.layers.Input(shape=input_shape)
    
    # Flatten the time window for the ANN
    x = keras.layers.Flatten()(inputs)
    
    # Hidden layers with explicit seeds for weight initialization
    for i in range(n_layers):
        x = keras.layers.Dense(
            units,
            activation='relu',
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seeds[seed_idx]),
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(x)
        seed_idx += 1
        
        # Add dropout with explicit seed
        if dropout_rate > 0:
            x = keras.layers.Dropout(dropout_rate, seed=kernel_seeds[seed_idx])(x)
            seed_idx += 1
    
    # Output layer with explicit seed
    outputs = keras.layers.Dense(
        output_shape,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=kernel_seeds[seed_idx]),
        bias_initializer=tf.keras.initializers.Zeros()
    )(x)
    
    # Create and return the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

@deterministic
def get_predictions(model, dataset, scaler):
    """Get predictions from a windowed dataset and inverse transform them."""
    # Set evaluation mode to be deterministic - version-compatible approach
    try:
        # For newer TensorFlow versions
        tf.config.run_functions_eagerly(True)
    except:
        # For older versions, use an alternative approach
        pass
    
    all_predictions = []
    all_labels = []
    
    # Use a fixed batch size for prediction to ensure consistency
    for features, labels in dataset:
        batch_predictions = model.predict(features, verbose=0)
        
        # Make sure predictions and labels have compatible shapes before flattening
        # For single-step prediction (LABEL_WIDTH=1), reshape appropriately
        if len(batch_predictions.shape) == 3:  # Shape is [batch, time_steps, features]
            batch_predictions = batch_predictions[:, -1, :]  # Get last time step
            
        if len(labels.shape) == 3:  # Shape is [batch, time_steps, features]
            labels = labels[:, -1, :]  # Get last time step
            
        # Now flatten and add to our collections
        all_predictions.extend(batch_predictions.flatten())
        all_labels.extend(labels.numpy().flatten())
    
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
        # Reset eager execution when done for newer TF versions
        tf.config.run_functions_eagerly(False)
    except:
        pass
    
    return all_predictions.flatten(), all_labels.flatten()

@deterministic
def create_spatial_groups(
    lat: Union[np.ndarray, List],
    lon: Union[np.ndarray, List],
    data_counts: Union[np.ndarray, List] = None,
    method: str = 'grid',
    n_clusters: int = 10,
    eps: float = 0.5,
    min_samples: int = 5,
    lat_grid_size: float = 0.05,
    lon_grid_size: float = 0.05,
    random_state: int = 42
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Group geographical data points using various spatial grouping methods.
    Enhanced to consider data counts for balanced grouping.
    
    Parameters
    ----------
    lat : array-like
        Latitude values
    lon : array-like
        Longitude values
    data_counts : array-like, optional
        Count of data points for each site (for balancing)
    method : str, default='kmeans'
        Clustering method to use: 'kmeans', 'dbscan', or 'grid'
    n_clusters : int, default=10
        Number of clusters for KMeans
    eps : float, default=0.5
        Maximum distance between samples for DBSCAN
    min_samples : int, default=5
        Minimum number of samples in a neighborhood for DBSCAN
    lat_grid_size : float, default=5.0
        Size of latitude intervals for grid method (in degrees)
    lon_grid_size : float, default=5.0
        Size of longitude intervals for grid method (in degrees)
    random_state : int, default=42
        Random state for reproducibility
        
    Returns
    -------
    groups : np.ndarray
        Array of group labels
    stats : pd.DataFrame
        DataFrame containing statistics for each group
    """
    # Convert inputs to numpy arrays
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    
    # Use equal data counts if not provided
    if data_counts is None:
        data_counts = np.ones_like(lat)
    else:
        data_counts = np.asarray(data_counts)
    
    if method == 'grid':
        # Create grid cells
        lat_bins = np.arange(
            np.floor(lat.min()),
            np.ceil(lat.max()) + lat_grid_size,
            lat_grid_size
        )
        lon_bins = np.arange(
            np.floor(lon.min()),
            np.ceil(lon.max()) + lon_grid_size,
            lon_grid_size
        )
        
        # Assign points to grid cells
        lat_indices = np.digitize(lat, lat_bins) - 1
        lon_indices = np.digitize(lon, lon_bins) - 1
        
        # Create unique group numbers for each grid cell
        n_lon_bins = len(lon_bins)
        groups = lat_indices * n_lon_bins + lon_indices
        # randomly assign groups to n_clusters
        np.random.seed(random_state)
        unique_groups = np.unique(groups)
        group_map = {g: np.random.randint(0, n_clusters) for g in unique_groups}
        groups = np.array([group_map[g] for g in groups])
        
    elif method in ['kmeans', 'dbscan']:
        # Scale coordinates
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(np.column_stack([lat, lon]))
        
        # Perform clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
        else:  # dbscan
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        
        groups = clusterer.fit_predict(coords_scaled)
    
    elif method == 'default':
        # Randomly assign points into groups, ensuring close points are in same group
        np.random.seed(random_state)
        groups = np.random.randint(0, n_clusters, size=len(lat))
        '''
        n_points = len(lat)
        
        # Calculate distances between points
        from scipy.spatial.distance import cdist
        coords = np.column_stack((lat, lon))
        distances = cdist(coords, coords, metric='euclidean')
        
        # Create adjacency matrix for close points
        close_mask = distances < 0.05
        
        # Use connected components to find groups of close points
        from scipy.sparse.csgraph import connected_components
        n_components, labels = connected_components(close_mask, directed=False)
        
        # Randomly assign cluster IDs to each connected component
        component_to_cluster = np.random.randint(0, n_clusters, size=n_components)
        groups = component_to_cluster[labels]
        '''
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods are 'kmeans', 'dbscan', 'grid', or 'default'.")

    # Calculate group statistics including data counts
    stats = []
    for group in np.unique(groups):
        if group == -1:  # Skip DBSCAN noise points
            continue
            
        mask = groups == group
        stats.append({
            'group': group,
            'size': np.sum(mask),
            'total_data_points': np.sum(data_counts[mask]),
            'mean_lat': np.mean(lat[mask]),
            'mean_lon': np.mean(lon[mask]),
            'std_lat': np.std(lat[mask]),
            'std_lon': np.std(lon[mask]),
            'min_lat': np.min(lat[mask]),
            'max_lat': np.max(lat[mask]),
            'min_lon': np.min(lon[mask]),
            'max_lon': np.max(lon[mask])
        })
    
    stats_df = pd.DataFrame(stats)
    
    # Print summary of data distribution
    if len(stats_df) > 0:
        print("Spatial Group Statistics:")
        print(f"Total groups: {len(stats_df)}")
        print(f"Min data points in a group: {stats_df['total_data_points'].min()}")
        print(f"Max data points in a group: {stats_df['total_data_points'].max()}")
        print(f"Mean data points per group: {stats_df['total_data_points'].mean():.1f}")
        
    return groups, stats_df

@deterministic
def create_group_labels_for_windows(site_to_group: Dict[str, int], site_windows_count: Dict[str, int]) -> List[int]:
    """
    Create group labels for all windows based on site group assignments.
    
    Parameters:
    -----------
    site_to_group : dict
        Dictionary mapping site IDs to group numbers
    site_windows_count : dict
        Dictionary mapping site IDs to number of windows
        
    Returns:
    --------
    group_labels : list
        List containing group labels for each window
    """
    group_labels = []
    
    for site_id, window_count in site_windows_count.items():
        group = site_to_group.get(site_id, 0)  # Default to group 0 if not found
        # Assign the same group number to all windows from this site
        group_labels.extend([group] * window_count)
    
    return group_labels

@deterministic
def convert_windows_to_numpy(windows: List[Tuple[tf.Tensor, tf.Tensor]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of windows to numpy arrays for features and labels.
    
    Parameters:
    -----------
    windows : list
        List of window tuples (inputs, labels) 
        
    Returns:
    --------
    X : np.ndarray
        Features array with shape [n_samples, input_width, n_features]
    y : np.ndarray
        Labels array with shape [n_samples, label_width] or [n_samples, label_width, n_label_features]
    """
    if not windows:
        return np.array([]), np.array([])
    
    # Extract features and labels from windows
    features = []
    labels = []
    
    for inputs, lbls in windows:
        # Convert to numpy
        features.append(inputs.numpy())
        labels.append(lbls.numpy())
    
    # Concatenate along batch dimension (dim 0)
    X = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)
    
    # If labels are 3D and we only have one label feature, flatten to 2D
    if y.ndim == 3 and y.shape[2] == 1:
        y = y.reshape(y.shape[0], y.shape[1])
    
    return X, y

@deterministic
def verify_determinism_of_spatial_ann_model():
    """Test if the spatial ANN model produces deterministic results across runs."""
    # Run the model twice with the same seed
    set_seed(42)
    r2_scores1, rmse_scores1 = main(run_id="spatial_run1", n_groups=10, test_size=0.2)
    
    set_seed(42)
    r2_scores2, rmse_scores2 = main(run_id="spatial_run2", n_groups=10, test_size=0.2)
    
    # Check if results match
    r2_diff = np.max(np.abs(np.array(r2_scores1) - np.array(r2_scores2)))
    rmse_diff = np.max(np.abs(np.array(rmse_scores1) - np.array(rmse_scores2)))
    
    print("\nDeterminism Verification Results:")
    print(f"Run 1 Test R² (average): {np.mean(r2_scores1):.6f}, RMSE (average): {np.mean(rmse_scores1):.6f}")
    print(f"Run 2 Test R² (average): {np.mean(r2_scores2):.6f}, RMSE (average): {np.mean(rmse_scores2):.6f}")
    print(f"Maximum Difference - R²: {r2_diff:.10f}, RMSE: {rmse_diff:.10f}")
    
    is_deterministic = r2_diff < 1e-10 and rmse_diff < 1e-10
    print(f"Model is deterministic: {is_deterministic}")
    
    return is_deterministic

@deterministic
def main(run_id="default", n_groups=10, n_cv_folds=5):
    """
    Main function implementing the paper's leave-fold-out CV approach.
    Each fold serves as test set exactly once, with remaining folds split 8:1 for train:validation.
    """
    # Set parameters
    RANDOM_SEED = get_seed()
    BATCH_SIZE = 32
    
    # Define window parameters
    INPUT_WIDTH = 3
    LABEL_WIDTH = 1
    SHIFT = 1
    EXCLUDE_LABELS = True
    EXCLUDE_TARGETS = True
    # Set deterministic rendering for matplotlib
    import matplotlib as mpl
    mpl.rcParams['agg.path.chunksize'] = 10000
    np.random.seed(RANDOM_SEED)
    
    print(f"Starting ANN model training with leave-fold-out CV, seed {RANDOM_SEED} (run_id: {run_id})")
    print(f"Using {n_groups} spatial groups with leave-fold-out cross-validation")
    
    # Create output directories
    plot_dir = Path('./plots')
    plot_dir.mkdir(exist_ok=True)
    
    model_dir = Path('./models')
    model_dir.mkdir(exist_ok=True)
    
    # Load and preprocess data
    data_dir = Path('./outputs/processed_data/merged/site/hourly_after_outlier_removal')
    data_list = list(data_dir.glob('*merged.csv'))
    
    if not data_list:
        print(f"No CSV files found in {data_dir}")
        try:
            # Fallback to single file
            data = pd.read_csv('./outputs/processed_data/merged/site/hourly_after_outlier_removal/test.csv')
            print("Using fallback test.csv file")
        except:
            print("ERROR: Could not find any data files")
            return [], []  # Return empty lists
    else:
        print(f"Found {len(data_list)} data files")
    
    # Process data files
    site_data_dict = {}  # Dictionary to store data segments by site
    site_info_dict = {}  # Dictionary to store site metadata
    site_windows_dict = {}  # Dictionary to store windows per site
    # Define the columns we want to use
    used_cols = ['sap_velocity', 'sw_in', 'ext_rad', 'ta', 'ws', 'vpd', 'ppfd_in', 'Day sin', 'Week sin', 'Month sin', 'Year sin']
    all_biome_types = set()
    all_possible_biome_types = ['Boreal forest', 'Subtropical desert', 'Temperate forest', 'Temperate grassland desert', 'Temperate rain forest', 'Tropical forest savanna', 'Tropical rain forest', 'Tundra', 'Woodland/Shrubland']
    
    # Sort data files for deterministic processing order
    data_list = sorted(data_list)

    # Process each data file
    for data_file in data_list:
        print(f"Processing {data_file.name}")
        try:

            
            
            df = pd.read_csv(data_file, parse_dates=['TIMESTAMP'])
            site_id = df['site'].iloc[0]
            # Extract lat/lon if available
            latitude = None
            longitude = None
            
            # Try to find latitude and longitude columns (check for various naming conventions)
            lat_cols = ['lat', 'latitude', 'LAT', 'LATITUDE']
            lon_cols = ['lon', 'longitude', 'LON', 'LONGITUDE']
            
            # Find the first matching column for latitude
            for col in lat_cols:
                if col in df.columns:
                    # Take the median to handle potential variations
                    latitude = df[col].median()
                    break
            
            # Find the first matching column for longitude
            for col in lon_cols:
                if col in df.columns:
                    # Take the median to handle potential variations
                    longitude = df[col].median()
                    break
            
            # If lat/lon not found, use a generated position (this is just for testing)
            if latitude is None or longitude is None:
                # Generate a deterministic pseudo-random position based on site_id
                np.random.seed(hash(site_id) % 2**32)
                latitude = np.random.uniform(-90, 90)
                longitude = np.random.uniform(-180, 180)
                print(f"  Warning: Generated pseudo-random position for site {site_id}: ({latitude:.2f}, {longitude:.2f})")
            
            # Set index and add time features
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
            
            # Clean data
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            
            # Calculate minimum required segment length
            window_size = INPUT_WIDTH + SHIFT
            min_segment_length = window_size + 7
            
            # Skip segments that are too small
            if len(df) < min_segment_length:
                print(f"Warning: {data_file.name} has too few records after cleaning: {len(df)}")
                continue
            
            # Segment the data
            segments = TimeSeriesSegmenter.segment_time_series(
                df, 
                gap_threshold=2, 
                unit='hours', 
                min_segment_length=min_segment_length
            )
            
            # Count total data points (considering segments)
            data_count = sum(len(segment) for segment in segments)
            
            print(f"  Created {len(segments)} segments for site {site_id} with {data_count} total points")
            
            # Add site data to dictionary
            if site_id not in site_data_dict:
                site_data_dict[site_id] = []
                site_info_dict[site_id] = {
                    'latitude': latitude,
                    'longitude': longitude,
                    'data_count': 0,
                    'segments': 0
                }
            site_data_dict[site_id].extend(segments)
            site_info_dict[site_id]['data_count'] += data_count
            site_info_dict[site_id]['segments'] += len(segments)
            
        except Exception as e:
            print(f"Error processing {data_file.name}: {e}")

    if not site_data_dict:
        print("No valid sites found. Falling back to single file mode.")
        try:
            # Fallback to single file processing
            data = pd.read_csv('./outputs/processed_data/merged/site/gap_filled_size1_with_era5/test.csv')
            site_id = "default_site"
            
            data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
            data = data.sort_values('TIMESTAMP').set_index('TIMESTAMP')
            data = add_time_features(data)
            
            # Select only needed columns
            columns = ['sap_velocity', 'sw_in', 'ext_rad', 'ta', 'ws', 'vpd', 'rh', 'ppfd_in']
            # Use available columns
            columns = [col for col in columns if col in data.columns]
            data = data[columns].dropna()
            
            # Create a single segment
            site_data_dict[site_id] = [data]
            site_info_dict[site_id] = {
                'latitude': 0.0,
                'longitude': 0.0,
                'data_count': len(data),
                'segments': 1
            }
        except:
            print("ERROR: Could not process any data.")
            return [], []

    # Create a DataFrame of site information
    site_info_df = pd.DataFrame.from_dict(site_info_dict, orient='index')
    print(f"\nSite information:")
    print(site_info_df)
    
    # Extract spatial information for grouping
    site_ids = list(site_data_dict.keys())
    latitudes = [site_info_dict[site]['latitude'] for site in site_ids]
    longitudes = [site_info_dict[site]['longitude'] for site in site_ids]
    data_counts = [site_info_dict[site]['data_count'] for site in site_ids]
    
    # Create spatial groups
    print("\nCreating spatial groups...")
    spatial_groups, group_stats = create_spatial_groups(
        lat=latitudes,
        lon=longitudes,
        data_counts=data_counts,
        method='default',
        n_clusters=n_groups,
        random_state=RANDOM_SEED
    )
    
    # Create mapping from site IDs to group numbers
    site_to_group = {site_ids[i]: spatial_groups[i] for i in range(len(site_ids))}
    
    # Print mapping for reference
    print("\nSite to group mapping:")
    for site, group in sorted(site_to_group.items()):
        print(f"  Site {site}: Group {group}")
    
    # Combine all segments for standardization
    all_segments = []
    for segments in site_data_dict.values():
        all_segments.extend(segments)
    
    all_data = pd.concat(all_segments)

    # Create scalers
    feature_scaler = StandardScaler()
    label_scaler = StandardScaler()

    # Get feature columns
    if 'biome' in used_cols:
        base_feature_columns = [col for col in all_data.columns if col != 'biome' and col != 'sap_velocity']
        biome_columns = all_possible_biome_types
        feature_columns = base_feature_columns + biome_columns
    else:
        feature_columns = [col for col in all_data.columns if col != 'sap_velocity']
    
    # Fit scalers on all data
    feature_scaler.fit(all_data[feature_columns])
    label_scaler.fit(all_data[['sap_velocity']])
    
    # Apply scaling to each segment in each site
    for site_id, segments in site_data_dict.items():
        site_windows_dict[site_id] = []
        for i, segment in enumerate(segments):
            segment_copy = segment.copy()
            segment_copy[feature_columns] = feature_scaler.transform(segment[feature_columns])
            segment_copy['sap_velocity'] = label_scaler.transform(segment[['sap_velocity']])
            site_data_dict[site_id][i] = segment_copy
        # create windows for each site
        site_windows = create_windows_from_segments(
            segments=site_data_dict[site_id],
            input_width=INPUT_WIDTH,
            label_width=LABEL_WIDTH,
            shift=SHIFT,
            label_columns=['sap_velocity'],
            exclude_targets_from_features=EXCLUDE_TARGETS,
            exclude_labels_from_inputs=EXCLUDE_LABELS
        )
        
        # Store the windows in the site_data_dict
        site_windows_dict[site_id].extend(site_windows)
    # release memory
    site_data_dict = {}
    # Get unique group numbers
    unique_groups = np.unique(spatial_groups)
    n_folds = len(unique_groups)
    
    print(f"\nImplementing leave-fold-out cross-validation:")
    print(f"  Total folds (groups): {n_folds}")
    print(f"  Each fold will serve as test set once")
    print(f"  For each iteration: {n_folds-2} training + 1 validation + 1 test")
    
    # Storage for results from all iterations
    all_test_r2_scores = []
    all_test_rmse_scores = []
    all_test_mae_scores = []
    fold_results = []
    all_predictions = []
    all_actuals = []
    
    # Iterate through each fold as test set
    for test_fold_idx, test_group in enumerate(unique_groups):
        print(f"\n=== Fold {test_fold_idx + 1}/{n_folds}: Group {test_group} as test set ===")
        
        # Get remaining groups (exclude test group)
        remaining_groups = [g for g in unique_groups if g != test_group]
        
        # Skip if we don't have enough groups for proper split
        if len(remaining_groups) < 2:
            print(f"  Skipping fold {test_fold_idx + 1}: not enough groups for train/val split")
            continue
        
        # Split remaining groups into train and validation
        # Use deterministic approach to select validation group
        np.random.seed(RANDOM_SEED + test_fold_idx)  # Different seed for each fold
        val_group = np.random.choice(remaining_groups, size=1)[0]
        train_groups = [g for g in remaining_groups if g != val_group]
        
        print(f"  Test group: {test_group}")
        print(f"  Validation group: {val_group}")
        print(f"  Training groups: {sorted(train_groups)}")
        
        # Collect sites for training, validation, and testing
        train_sites = [site for site, group in site_to_group.items() if group in train_groups]
        val_sites = [site for site, group in site_to_group.items() if group == val_group]
        test_sites = [site for site, group in site_to_group.items() if group == test_group]
        
        print(f"  Training sites: {len(train_sites)}")
        print(f"  Validation sites: {len(val_sites)}")
        print(f"  Test sites: {len(test_sites)}")
        
        # Skip if any split is empty
        if len(train_sites) == 0 or len(val_sites) == 0 or len(test_sites) == 0:
            print(f"  Skipping fold {test_fold_idx + 1}: empty train/val/test split")
            continue
        
        # Collect segments for training, validation, and testing
        train_windows = []
        for site in train_sites:
            train_windows.extend(site_windows_dict[site])
        
        val_windows = []
        for site in val_sites:
            val_windows.extend(site_windows_dict[site])
        
        test_windows = []
        for site in test_sites:
            test_windows.extend(site_windows_dict[site])
        
       
        
        print(f"  Created {len(train_windows)} training windows")
        print(f"  Created {len(val_windows)} validation windows")
        print(f"  Created {len(test_windows)} test windows")
        
        # Skip if any set is empty
        if len(train_windows) == 0 or len(val_windows) == 0 or len(test_windows) == 0:
            print(f"  Skipping fold {test_fold_idx + 1} due to empty window set")
            continue
        
        # Convert windows to numpy arrays
        X_train, y_train = convert_windows_to_numpy(train_windows)
        X_val, y_val = convert_windows_to_numpy(val_windows)
        X_test, y_test = convert_windows_to_numpy(test_windows)
        
        # Create TensorFlow datasets
        train_ds = windows_to_tf_dataset(train_windows, batch_size=BATCH_SIZE)
        val_ds = windows_to_tf_dataset(val_windows, batch_size=BATCH_SIZE, shuffle=False)
        test_ds = windows_to_tf_dataset(test_windows, batch_size=BATCH_SIZE, shuffle=False)
        
        # Get input shape
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Combine train and validation data for optimizer
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.vstack([y_train, y_val])
        
        
        
        print(f"  Training model for fold {test_fold_idx + 1} using optimizer...")
        print(f"    Combined dataset: {len(X_train_val)} samples")
        print(f"    Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        
        # Define parameter grid for this fold
        param_grid = {
            'architecture': {
                'n_layers': [2],
                'units': [64],
                'dropout_rate': [0.1] 
            },
            'optimizer': {   
                'name': ['Adam'],
                'learning_rate': [0.001]
            },
            'training': {
                'batch_size': BATCH_SIZE,
                'epochs': 100,
                'patience': 10
            }
        }
        
        # Create optimizer for this fold
        optimizer = DLOptimizer(
            base_architecture=create_windowed_nn,
            task='regression',
            model_type='windowed_nn',
            param_grid=param_grid,
            input_shape=input_shape,
            output_shape=LABEL_WIDTH,
            scoring='val_loss',
            random_state=RANDOM_SEED + test_fold_idx,
            use_distribution=False,
        )
        
        # Fit the optimizer with train/val groups
        # This ensures train and val data are properly separated during CV
        optimizer.fit(
            X_train_val, 
            y_train_val,
            is_cv=False,  # Use cross-validation with train/val split
            X_val=X_val,
            y_val=y_val,
        )
        
        print(f"  Training completed using optimizer")
        best_model = optimizer.get_best_model()
        # Get predictions on test set only
        test_predictions, test_labels_actual = get_predictions(best_model, test_ds, label_scaler)
        
        # Calculate test metrics for this fold
        fold_test_r2 = r2_score(test_labels_actual, test_predictions)
        fold_test_rmse = np.sqrt(mean_squared_error(test_labels_actual, test_predictions))
        fold_test_mae = mean_absolute_error(test_labels_actual, test_predictions)
        
        # Store results
        all_test_r2_scores.append(fold_test_r2)
        all_test_rmse_scores.append(fold_test_rmse)
        all_test_mae_scores.append(fold_test_mae)
        
        # Store predictions for overall analysis
        all_predictions.extend(test_predictions)
        all_actuals.extend(test_labels_actual)
        
        # Store detailed fold results
        fold_results.append({
            'fold': test_fold_idx + 1,
            'test_group': test_group,
            'val_group': val_group,
            'train_groups': train_groups,
            'n_train_sites': len(train_sites),
            'n_val_sites': len(val_sites),
            'n_test_sites': len(test_sites),
            'n_train_windows': len(train_windows),
            'n_val_windows': len(val_windows),
            'n_test_windows': len(test_windows),
            'test_r2': fold_test_r2,
            'test_rmse': fold_test_rmse,
            'test_mae': fold_test_mae,
            'training_epochs': param_grid['training']['epochs'],
            #'final_train_loss': final_train_loss,
            #'final_val_loss': final_val_loss
        })
        
        print(f"  Fold {test_fold_idx + 1} Results:")
        print(f"    Test R²: {fold_test_r2:.6f}")
        print(f"    Test RMSE: {fold_test_rmse:.6f}")
        print(f"    Test MAE: {fold_test_mae:.6f}")
        
        # Save model for this fold
        best_model.save(model_dir / f'spatial_ann_fold_{test_fold_idx + 1}_{run_id}.h5')
    
    # Calculate overall statistics
    if len(all_test_r2_scores) > 0:
        mean_test_r2 = np.mean(all_test_r2_scores)
        std_test_r2 = np.std(all_test_r2_scores)
        mean_test_rmse = np.mean(all_test_rmse_scores)
        std_test_rmse = np.std(all_test_rmse_scores)
        mean_test_mae = np.mean(all_test_mae_scores)
        std_test_mae = np.std(all_test_mae_scores)
        
        print(f"\n=== OVERALL LEAVE-FOLD-OUT CROSS-VALIDATION RESULTS ===")
        print(f"Number of completed folds: {len(all_test_r2_scores)}")
        print(f"Test R² Score: {mean_test_r2:.6f} ± {std_test_r2:.6f}")
        print(f"Test RMSE: {mean_test_rmse:.6f} ± {std_test_rmse:.6f}")
        print(f"Test MAE: {mean_test_mae:.6f} ± {std_test_mae:.6f}")
        
        # Create results summary DataFrame
        results_df = pd.DataFrame(fold_results)
        results_df.to_csv(Path(__file__).parent / f'spatial_cv_results_{run_id}.csv', index=False)
        print(f"\nDetailed results saved to: {Path(__file__).parent / f'spatial_cv_results_{run_id}.csv'}")
        
        # Plot results across folds
        plt.figure(figsize=(15, 10))
        
        # Plot 1: R² scores by fold
        plt.subplot(2, 3, 1)
        plt.bar(range(1, len(all_test_r2_scores) + 1), all_test_r2_scores)
        plt.axhline(y=mean_test_r2, color='r', linestyle='--', label=f'Mean: {mean_test_r2:.3f}')
        plt.xlabel('Fold')
        plt.ylabel('Test R² Score')
        plt.title('Test R² Score by Fold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: RMSE by fold
        plt.subplot(2, 3, 2)
        plt.bar(range(1, len(all_test_rmse_scores) + 1), all_test_rmse_scores)
        plt.axhline(y=mean_test_rmse, color='r', linestyle='--', label=f'Mean: {mean_test_rmse:.3f}')
        plt.xlabel('Fold')
        plt.ylabel('Test RMSE')
        plt.title('Test RMSE by Fold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: MAE by fold
        plt.subplot(2, 3, 3)
        plt.bar(range(1, len(all_test_mae_scores) + 1), all_test_mae_scores)
        plt.axhline(y=mean_test_mae, color='r', linestyle='--', label=f'Mean: {mean_test_mae:.3f}')
        plt.xlabel('Fold')
        plt.ylabel('Test MAE')
        plt.title('Test MAE by Fold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Overall predictions vs actual
        plt.subplot(2, 3, 4)
        plt.scatter(all_actuals, all_predictions, alpha=0.5)
        plt.xlabel('True Values [Sap Velocity]')
        plt.ylabel('Predictions [Sap Velocity]')
        plt.title(f'All Folds: Predictions vs Actual\n(R² = {mean_test_r2:.3f})')
        plt.axis('equal')
        plt.axis('square')
        
        # Add 1:1 line
        min_val = min(min(all_actuals), min(all_predictions))
        max_val = max(max(all_actuals), max(all_predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Error distribution
        plt.subplot(2, 3, 5)
        errors = np.array(all_predictions) - np.array(all_actuals)
        plt.hist(errors, bins=30, alpha=0.7)
        plt.xlabel('Prediction Error [Sap Velocity]')
        plt.ylabel('Count')
        plt.title(f'Error Distribution\n(RMSE = {mean_test_rmse:.3f})')
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.8)
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Spatial distribution of sites
        plt.subplot(2, 3, 6)
        # Create a colormap for groups
        cmap = plt.cm.get_cmap('tab10', n_folds)
        
        # Plot each site with different colors for groups
        for i, site in enumerate(site_ids):
            group = spatial_groups[i]
            plt.scatter(
                longitudes[i],
                latitudes[i],
                s=50 + site_info_dict[site]['data_count'] / 1000,  # Size based on data count
                c=[cmap(group)],
                alpha=0.7,
                label=f"Group {group}" if i == list(spatial_groups).index(group) else ""
            )
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Spatial Distribution of Sites by Group')
        plt.grid(True, alpha=0.3)
        
        # Add legend with unique entries only
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if len(by_label) <= 10:  # Only show legend if not too many groups
            plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(plot_dir / f'spatial_cv_comprehensive_results_{run_id}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create individual plots for paper-quality figures
        # Plot predictions vs actual (publication quality)
        plt.figure(figsize=(8, 8))
        plt.scatter(all_actuals, all_predictions, alpha=0.6, s=20)
        plt.xlabel('Observed Sap Velocity', fontsize=14)
        plt.ylabel('Predicted Sap Velocity', fontsize=14)
        plt.title(f'Leave-Fold-Out Cross-Validation Results\nR² = {mean_test_r2:.3f} ± {std_test_r2:.3f}, RMSE = {mean_test_rmse:.3f} ± {std_test_rmse:.3f}', 
                 fontsize=12)
        
        # Add 1:1 line
        min_val = min(min(all_actuals), min(all_predictions))
        max_val = max(max(all_actuals), max(all_predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r-', alpha=0.8, linewidth=2)
        
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.axis('square')
        plt.tight_layout()
        plt.savefig(plot_dir / f'spatial_cv_predictions_vs_actual_{run_id}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return all_test_r2_scores, all_test_rmse_scores
    
    else:
        print("ERROR: No folds completed successfully")
        return [], []

if __name__ == "__main__":
    # Run the main function with spatial cross-validation
    main(run_id="spatial_ann_group_cv", n_groups=10)
    
    # Optionally run determinism test
    # verify_determinism_of_spatial_ann_model()