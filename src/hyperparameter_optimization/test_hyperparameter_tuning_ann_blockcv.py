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
from src.hyperparameter_optimization.blockcv import BlockCV
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
def extract_site_id_from_filename(filename):
    """
    Extract site ID from a filename.
    
    Parameters:
    -----------
    filename : Path or str
        File path or name
        
    Returns:
    --------
    str
        Site ID
    """
    # Convert to Path if it's a string
    if isinstance(filename, str):
        filename = Path(filename)
    
    # Extract site ID from filename
    # Assuming the site ID is the first part of the filename before an underscore
    # Adapt this pattern according to your actual filename structure
    site_id = filename.stem.split('_')[0]
    
    return site_id

@deterministic
def create_windows_from_segments(
    segments: List[pd.DataFrame],
    input_width: int,
    label_width: int,
    shift: int,
    label_columns: List[str] = None,
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
    exclude_labels_from_inputs : bool
        Whether to exclude label columns from inputs
        
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
def main(run_id="default", block_size=1.0, buffer_size=0.0, n_cv_folds=5, use_autocorrelation=False):
    """
    Main function to train and evaluate the ANN model with spatial cross-validation
    using BlockCV approach with hyperparameter optimization.
    
    Parameters:
    -----------
    run_id : str
        Identifier for the run, used for saving models and plots
    block_size : float or tuple of float
        Size of spatial blocks in degrees (ignored if use_autocorrelation=True)
    buffer_size : float
        Size of buffer zone between blocks in degrees
    n_cv_folds : int
        Number of cross-validation folds for the optimizer
    use_autocorrelation : bool
        Whether to use spatial autocorrelation to determine optimal block size
        
    Returns:
    --------
    tuple
        Lists of R² scores and RMSE scores for each test set
    """
    # Set parameters
    RANDOM_SEED = get_seed()
    BATCH_SIZE = 32
    
    # Define window parameters
    INPUT_WIDTH = 8
    LABEL_WIDTH = 1
    SHIFT = 1
    EXCLUDE_LABEL = True
    
    # Set deterministic rendering for matplotlib
    import matplotlib as mpl
    mpl.rcParams['agg.path.chunksize'] = 10000
    np.random.seed(RANDOM_SEED)
    
    print(f"Starting ANN model training with BlockCV, seed {RANDOM_SEED} (run_id: {run_id})")
    if use_autocorrelation:
        print(f"Using autocorrelation to determine block size, buffer_size={buffer_size}")
    else:
        print(f"Using block_size={block_size}, buffer_size={buffer_size}")
    
    # Create output directories
    plot_dir = Path('./plots')
    plot_dir.mkdir(exist_ok=True)
    
    model_dir = Path('./models')
    model_dir.mkdir(exist_ok=True)
    
    # Load and preprocess data
    data_dir = Path('./outputs/processed_data/merged/site/gap_filled_size1_hourly_after_filter')
    data_list = list(data_dir.glob('*merged.csv'))
    
    if not data_list:
        print(f"No CSV files found in {data_dir}")
        try:
            # Fallback to single file
            data = pd.read_csv('./outputs/processed_data/merged/site/gap_filled_size1_with_era5/test.csv')
            print("Using fallback test.csv file")
        except:
            print("ERROR: Could not find any data files")
            return [], []  # Return empty lists
    else:
        print(f"Found {len(data_list)} data files")
    
    # Process data files
    site_data_dict = {}  # Dictionary to store data segments by site
    site_info_dict = {}  # Dictionary to store site metadata
    
    # Define the columns we want to use
    used_cols = ['sap_velocity', 'ext_rad', 'sw_in', 'ta', 'ws', 'vpd','ppfd_in', 'Day sin', 'Week sin', 'Month sin', 'Year sin', 'mean_annual_temp', 'mean_annual_precip']

    all_possible_biome_types = ['Boreal forest', 'Subtropical desert', 'Temperate forest', 'Temperate grassland desert', 'Temperate rain forest', 'Tropical forest savanna', 'Tropical rain forest', 'Tundra', 'Woodland/Shrubland']
    
    # Sort data files for deterministic processing order
    data_list = sorted(data_list)

    # Process each data file
    for data_file in data_list:
        print(f"Processing {data_file.name}")
        try:
            # Extract site ID from filename
            site_id = extract_site_id_from_filename(data_file)
            
            df = pd.read_csv(data_file, parse_dates=['TIMESTAMP'])
            
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
        for i, segment in enumerate(segments):
            segment_copy = segment.copy()
            segment_copy[feature_columns] = feature_scaler.transform(segment[feature_columns])
            segment_copy['sap_velocity'] = label_scaler.transform(segment[['sap_velocity']])
            site_data_dict[site_id][i] = segment_copy
    
    # === USING BLOCKCV TO CREATE SPATIAL GROUPS ===
    # Extract spatial information
    site_ids = list(site_data_dict.keys())
    latitudes = [site_info_dict[site]['latitude'] for site in site_ids]
    longitudes = [site_info_dict[site]['longitude'] for site in site_ids]
    data_counts = [site_info_dict[site]['data_count'] for site in site_ids]
    
    # Determine block size using spatial autocorrelation if requested
    if use_autocorrelation:
        print("\nAnalyzing spatial autocorrelation to determine optimal block size...")
        
        # Collect site-level target values for autocorrelation analysis
        site_values = []
        for site in site_ids:
            if site in site_data_dict and site_data_dict[site]:
                # Use mean sap_velocity for each site
                site_mean = np.mean([segment['sap_velocity'].mean() for segment in site_data_dict[site]])
                site_values.append(site_mean)
            else:
                site_values.append(np.nan)
        
        # Calculate spatial autocorrelation
        autocorr_results = BlockCV.calculate_spatial_autocorrelation(
            lat=latitudes,
            lon=longitudes,
            values=site_values,
            n_bins=15
        )
        
        # Determine block size from autocorrelation with a multiplier of 1.5
        # This ensures blocks are large enough to reduce spatial dependence
        block_size = BlockCV.select_block_size_from_autocorrelation(
            autocorr_results,
            method='practical_range',
            multiplier=1.5
        )
        
        # Plot autocorrelation results
        BlockCV.plot_spatial_autocorrelation(
            autocorr_results,
            filename=plot_dir / f'spatial_autocorrelation_{run_id}.png',
            title='Spatial Autocorrelation Analysis for Sap Velocity'
        )
    
    # Create spatial blocks
    print("\nCreating spatial blocks using BlockCV approach...")
    block_assignments, block_info = BlockCV.create_spatial_blocks(
        lat=latitudes,
        lon=longitudes,
        data_counts=data_counts,
        block_size=block_size,
        buffer_size=buffer_size,
        random_state=RANDOM_SEED
    )
    
    # Plot spatial blocks
    BlockCV.plot_spatial_blocks(
        lat=latitudes,
        lon=longitudes,
        block_assignments=block_assignments,
        block_info=block_info,
        filename=plot_dir / f'spatial_blocks_{run_id}.png',
        title=f'Spatial Blocks for Cross-Validation (Buffer: {buffer_size})'
    )
    
    # Create site-to-block mapping
    site_to_block = {site_ids[i]: block_assignments[i] for i in range(len(site_ids))}
    
    # Get valid blocks (excluding buffer zones)
    valid_blocks = np.unique(block_assignments)
    valid_blocks = valid_blocks[valid_blocks >= 0]
    
    # Split blocks into training and testing
    np.random.seed(RANDOM_SEED)
    test_size = 0.2  # Use 20% of blocks for testing
    train_blocks, test_blocks = train_test_split(
        valid_blocks, 
        test_size=test_size,
        random_state=RANDOM_SEED
    )
    
    print(f"\nTrain-test split:")
    print(f"  Training blocks: {sorted(train_blocks)}")
    print(f"  Test blocks: {sorted(test_blocks)}")
    
    # Collect sites for training and testing
    train_sites = [site for site, block in site_to_block.items() 
                   if block in train_blocks]
    test_sites = [site for site, block in site_to_block.items() 
                  if block in test_blocks]
    buffer_sites = [site for site, block in site_to_block.items() 
                   if block == -1]  # Sites in buffer zones
    
    print(f"  Training sites: {len(train_sites)}")
    print(f"  Test sites: {len(test_sites)}")
    print(f"  Buffer zone sites: {len(buffer_sites)}")
    
    # Collect segments for training and testing
    train_segments = []
    for site in train_sites:
        train_segments.extend(site_data_dict[site])
    
    test_segments = []
    for site in test_sites:
        test_segments.extend(site_data_dict[site])
    
    print(f"  Training segments: {len(train_segments)}")
    print(f"  Test segments: {len(test_segments)}")
    
    # Create windows
    print("\nCreating windows from segments...")
    # Function to create windows from a segment list
    def create_windows_from_segments(segments, input_width, label_width, shift, label_columns, exclude_labels):
        all_windows = []
        
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
                batch_size=1,  # Use batch size of 1 to extract windows
                shuffle=False,  # Don't shuffle to maintain order
                exclude_labels_from_inputs=exclude_labels
            )
            
            # Collect windows from this segment
            segment_ds = window_gen.dataset
            for inputs, labels in segment_ds:
                all_windows.append((inputs, labels))
        
        return all_windows
    
    # Function to convert windows to TF dataset
    def windows_to_tf_dataset(windows, batch_size, shuffle=True):
        if not windows:
            return None
            
        # Concatenate all windows
        inputs = tf.concat([w[0] for w in windows], axis=0)
        labels = tf.concat([w[1] for w in windows], axis=0)
        
        # Create dataset
        ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
        
        # Shuffle if requested
        if shuffle:
            ds = ds.shuffle(buffer_size=10000, seed=get_seed())
        
        # Batch
        ds = ds.batch(batch_size)
        
        # Set deterministic options
        options = tf.data.Options()
        options.experimental_deterministic = True
        ds = ds.with_options(options)
        
        return ds
    
    # Create windows for training and testing
    train_windows = create_windows_from_segments(
        train_segments, 
        INPUT_WIDTH, 
        LABEL_WIDTH, 
        SHIFT, 
        ['sap_velocity'], 
        EXCLUDE_LABEL
    )
    
    test_windows = create_windows_from_segments(
        test_segments, 
        INPUT_WIDTH, 
        LABEL_WIDTH, 
        SHIFT, 
        ['sap_velocity'], 
        EXCLUDE_LABEL
    )
    
    print(f"  Created {len(train_windows)} training windows")
    print(f"  Created {len(test_windows)} test windows")
    
    # Count windows per site for group labeling
    site_windows_count = {}
    
    for site in train_sites:
        site_segments = site_data_dict[site]
        site_windows = create_windows_from_segments(
            site_segments, 
            INPUT_WIDTH, 
            LABEL_WIDTH, 
            SHIFT, 
            ['sap_velocity'], 
            EXCLUDE_LABEL
        )
        site_windows_count[site] = len(site_windows)
    
    # Create group labels for all training windows
    group_labels = BlockCV.create_group_labels_for_windows(site_to_block, site_windows_count)
    
    print(f"  Created {len(group_labels)} group labels for training windows")
    
    # Convert windows to numpy arrays for the optimizer
    def convert_windows_to_numpy(windows):
        if not windows:
            return np.array([]), np.array([])
            
        features = []
        labels = []
        
        for inputs, lbls in windows:
            features.append(inputs.numpy())
            labels.append(lbls.numpy())
        
        X = np.concatenate(features, axis=0)
        y = np.concatenate(labels, axis=0)
        
        # Reshape if needed
        if y.ndim == 3 and y.shape[2] == 1:
            y = y.reshape(y.shape[0], y.shape[1])
        
        return X, y
    
    X_train, y_train = convert_windows_to_numpy(train_windows)
    X_test, y_test = convert_windows_to_numpy(test_windows)
    
    print(f"  Training features shape: {X_train.shape}")
    print(f"  Training labels shape: {y_train.shape}")
    print(f"  Test features shape: {X_test.shape}")
    print(f"  Test labels shape: {y_test.shape}")
    
    # Create TF datasets for predictions
    train_ds = windows_to_tf_dataset(train_windows, BATCH_SIZE)
    test_ds = windows_to_tf_dataset(test_windows, BATCH_SIZE, shuffle=False)
    
    # Get input shape from data
    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
    print(f"  Input shape: {input_shape}")
    
    # Define parameter grid for ANN
    param_grid = {
        'architecture': {
            'n_layers': [3],
            'units': [128],
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
    
    # Create and fit optimizer with block-based CV
    print("\nTraining model with BlockCV spatial cross-validation...")
    optimizer = DLOptimizer(
        base_architecture=create_windowed_nn,
        task='regression',
        model_type='windowed_nn',
        param_grid=param_grid,
        input_shape=input_shape,
        output_shape=LABEL_WIDTH,
        scoring='val_loss',
        random_state=RANDOM_SEED,
        use_distribution=False,
        n_splits=n_cv_folds  # Use specified number of CV folds
    )
    
    # Filter out buffer zone sites from groups
    valid_group_indices = [i for i, g in enumerate(group_labels) if g >= 0]
    if valid_group_indices:
        X_train_filtered = X_train[valid_group_indices]
        y_train_filtered = y_train[valid_group_indices]
        groups_filtered = [group_labels[i] for i in valid_group_indices]
        
        # Fit the optimizer with block-based group labels
        optimizer.fit(
            X_train_filtered, 
            y_train_filtered,
            is_cv=True,  # Use cross-validation
            groups=groups_filtered,  # Pass block labels for grouped CV
            split_type='spatial'  # Indicate spatial splitting
        )
    else:
        # Fall back to regular CV if no valid groups
        print("Warning: No valid group labels. Falling back to regular CV.")
        optimizer.fit(
            X_train, 
            y_train,
            is_cv=True,
            split_type='random'
        )
    
    # Get the best model
    best_model = optimizer.get_best_model()
    
    # Get predictions
    test_predictions, test_labels_actual = get_predictions(best_model, test_ds, label_scaler)
    train_predictions, train_labels_actual = get_predictions(best_model, train_ds, label_scaler)
    
    # Calculate metrics
    test_r2 = r2_score(test_labels_actual, test_predictions)
    test_mse = mean_squared_error(test_labels_actual, test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(test_labels_actual, test_predictions)
    
    train_r2 = r2_score(train_labels_actual, train_predictions)
    train_mse = mean_squared_error(train_labels_actual, train_predictions)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(train_labels_actual, train_predictions)
    
    # Print results
    print("\nModel Evaluation Results:")
    print("\nTraining Set Metrics:")
    print(f"  R² Score: {train_r2:.6f}")
    print(f"  RMSE: {train_rmse:.6f}")
    print(f"  MAE: {train_mae:.6f}")
    
    print("\nTest Set Metrics:")
    print(f"  R² Score: {test_r2:.6f}")
    print(f"  RMSE: {test_rmse:.6f}")
    print(f"  MAE: {test_mae:.6f}")
    
    # Plot training predictions
    plt.figure(figsize=(10, 10))
    plt.scatter(train_labels_actual, train_predictions, alpha=0.5)
    plt.xlabel('True Values [Sap Velocity]')
    plt.ylabel('Predictions [Sap Velocity]')
    plt.title('Training Set: Predictions vs Actual')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    plt.plot([-100, 100], [-100, 100])
    plt.savefig(plot_dir / f'blockcv_train_predictions_{run_id}.png')
    plt.close()
    
    # Plot test predictions
    plt.figure(figsize=(10, 10))
    plt.scatter(test_labels_actual, test_predictions, alpha=0.5)
    plt.xlabel('True Values [Sap Velocity]')
    plt.ylabel('Predictions [Sap Velocity]')
    plt.title('Test Set: Predictions vs Actual')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    plt.plot([-100, 100], [-100, 100])
    plt.savefig(plot_dir / f'blockcv_test_predictions_{run_id}.png')
    plt.close()
    
    # Plot error distributions
    train_error = train_predictions - train_labels_actual
    plt.figure(figsize=(10, 7))
    plt.hist(train_error, bins=25)
    plt.xlabel('Prediction Error [Sap Velocity]')
    plt.ylabel('Count')
    plt.title('Training Set: Error Distribution')
    plt.savefig(plot_dir / f'blockcv_train_error_distribution_{run_id}.png')
    plt.close()
    
    test_error = test_predictions - test_labels_actual
    plt.figure(figsize=(10, 7))
    plt.hist(test_error, bins=25)
    plt.xlabel('Prediction Error [Sap Velocity]')
    plt.ylabel('Count')
    plt.title('Test Set: Error Distribution')
    plt.savefig(plot_dir / f'blockcv_test_error_distribution_{run_id}.png')
    plt.close()
    
    # Plot spatial distribution with train/test assignment
    try:
        plt.figure(figsize=(12, 8))
        
        # Define markers and colors
        train_marker = 'o'
        test_marker = 'X'
        buffer_marker = 's'
        
        # Plot sites by train/test assignment
        # Training sites
        for site in train_sites:
            idx = site_ids.index(site)
            block = block_assignments[idx]
            plt.scatter(
                longitudes[idx], 
                latitudes[idx],
                s=50 + site_info_dict[site]['data_count'] / 1000,
                c=plt.cm.tab10(block % 10),
                marker=train_marker,
                edgecolor='black',
                alpha=0.7,
                label=f"Train Block {block}" if site == train_sites[0] else ""
            )
        
        # Test sites
        for site in test_sites:
            idx = site_ids.index(site)
            block = block_assignments[idx]
            plt.scatter(
                longitudes[idx], 
                latitudes[idx],
                s=50 + site_info_dict[site]['data_count'] / 1000,
                c=plt.cm.tab10(block % 10),
                marker=test_marker,
                edgecolor='red',
                linewidth=1.5,
                alpha=0.7,
                label=f"Test Block {block}" if site == test_sites[0] else ""
            )
        
        # Buffer sites
        for site in buffer_sites:
            idx = site_ids.index(site)
            plt.scatter(
                longitudes[idx], 
                latitudes[idx],
                s=30,
                c='gray',
                marker=buffer_marker,
                alpha=0.5,
                label="Buffer Zone" if site == buffer_sites[0] else ""
            )
        
        # Add legend with unique entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='best')
        
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Spatial Distribution by Train/Test Assignment (BlockCV)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(plot_dir / f'blockcv_spatial_distribution_{run_id}.png')
        plt.close()
    except Exception as e:
        print(f"Error creating spatial distribution plot: {e}")
    
    # Save the model
    best_model.save(model_dir / f'blockcv_model_seed_{RANDOM_SEED}_{run_id}.h5')
    
    # Return metrics
    return [test_r2], [test_rmse]


if __name__ == "__main__":
    # Run the main function with BlockCV
    # First run: Using fixed block size

    
    # Second run: Using autocorrelation-determined block size
    main(
        run_id="blockcv_auto", 
        buffer_size=0, 
        n_cv_folds=10,
        use_autocorrelation=True
    )