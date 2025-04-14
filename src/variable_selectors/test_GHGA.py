import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import sys
from pathlib import Path
parent_dir = str(Path(__file__).parent.parent.parent)
print(parent_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# Import our optimized version instead of the original
from src.variable_selectors.GHGA_selector import GHGASelector

def add_time_features(df, datetime_column=None):
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Get the datetime series from column or index with better error handling
    try:
        if datetime_column is not None:
            if datetime_column not in df.columns:
                raise ValueError(f"Column '{datetime_column}' not found in DataFrame")
            date_time = pd.to_datetime(df[datetime_column], errors='coerce')
        else:
            # Use index if no column specified
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    date_time = pd.to_datetime(df.index, errors='coerce')
                except:
                    raise ValueError("DataFrame index cannot be converted to datetime and no datetime_column specified")
            else:
                date_time = df.index
                
        # Check for NaN values after conversion
        if date_time.isna().any():
            print(f"Warning: {date_time.isna().sum()} NaN datetime values detected")
            
        # Convert datetime to timestamp in seconds with error handling
        timestamp_s = date_time.map(lambda x: pd.Timestamp(x).timestamp() if pd.notna(x) else np.nan)
        
        # Check for NaN values in timestamps
        if pd.isna(timestamp_s).any():
            print(f"Warning: {pd.isna(timestamp_s).sum()} NaN timestamp values detected")
            
        # Define day and year in seconds
        day = 24 * 60 * 60  # seconds in a day
        year = 365.2425 * day  # seconds in a year
        week = 7 * day
        month = 30.44 * day
        
        # Create cyclical features with NaN handling
        for name, period in [ ('Year', year), ('Week', week), ('Month', month)]:
            # Calculate sine and cosine features, handling NaN values
            df[f'{name} sin'] = np.sin(timestamp_s * (2 * np.pi / period))
            df[f'{name} cos'] = np.cos(timestamp_s * (2 * np.pi / period))
            
            # Check for NaN values in created features
            if df[f'{name} sin'].isna().any() or df[f'{name} cos'].isna().any():
                print(f"Warning: NaN values detected in {name} cyclical features")
                
    except Exception as e:
        print(f"Error in add_time_features: {str(e)}")
        # Return original dataframe if there's an error
        return df
        
    return df
# Add this function to your main script
def robust_preprocess_data(df, target_column):
    """
    Robustly preprocess data for feature selection.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data frame
    target_column : str
        Name of the target column
        
    Returns:
    --------
    tuple
        (X_scaled, y, feature_names)
    """
    print("Starting robust data preprocessing...")
    
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Check for missing values
    missing_counts = df_clean.isna().sum()
    print(f"Missing values before cleaning: {missing_counts[missing_counts > 0].to_dict()}")
    
    # Drop rows with NaN values
    df_clean = df_clean.dropna()
    print(f"Rows after dropping NaNs: {len(df_clean)}")
    
    # Check for infinite values and replace with NaN, then drop
    inf_mask = np.isinf(df_clean.select_dtypes(include=np.number))
    if inf_mask.any().any():
        print(f"Found {inf_mask.sum().sum()} infinite values, replacing with NaN")
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
        print(f"Rows after removing infinites: {len(df_clean)}")
    
    # Separate target
    if target_column not in df_clean.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    y = df_clean[target_column]
    X = df_clean.drop(columns=[target_column])
    
    # Handle categorical variables
    cat_columns = X.select_dtypes(include=['category', 'object']).columns
    for col in cat_columns:
        X[col] = X[col].astype('category').cat.codes
    
    # Check for constant columns
    variances = X.var()
    constant_cols = variances[variances <= 1e-10].index.tolist()
    if constant_cols:
        print(f"Removing {len(constant_cols)} constant columns: {constant_cols}")
        X = X.drop(columns=constant_cols)
    
    # Check for highly correlated features
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    if high_corr_cols:
        print(f"Found {len(high_corr_cols)} highly correlated features (>0.95)")
        print(f"Consider removing: {high_corr_cols}")
        # Uncomment to actually remove them:
        # X = X.drop(columns=high_corr_cols)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X)
    
    # Convert back to DataFrame
    X_scaled = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)
    
    # Final check for NaN/Inf values after scaling
    if X_scaled.isna().any().any() or np.isinf(X_scaled.values).any():
        print("Warning: NaN or Inf values found after scaling, replacing with 0")
        X_scaled = X_scaled.fillna(0).replace([np.inf, -np.inf], 0)
    
    # Add a tiny amount of noise to prevent perfect correlations
    noise_level = 1e-8
    X_scaled = X_scaled + np.random.normal(0, noise_level, X_scaled.shape)
    
    print(f"Final preprocessed data shape: {X_scaled.shape}")
    return X_scaled, y, X.columns.tolist()
# Main example
def main():
    # Start timing
    start_time = time.time()
    
    all_data = pd.read_csv('outputs/processed_data/merged/site/gap_filled_size1_with_era5/all_biomes_merged_data.csv')
    all_data = all_data.set_index('TIMESTAMP').sort_index()
    selected_vars = [
    # Target variable
    'sap_velocity',
    # One from each radiation group
    'surface_net_solar_radiation', 'sw_in', 'surface_net_thermal_radiation', 'surface_latent_heat_flux',
    
    # Temperature variables
    'ta', 'soil_temperature_level_1', 'soil_temperature_level_2', 'soil_temperature_level_3',# Just one soil level
    
    # Water variables
    'precip', 'swc_shallow', 'swc_deep', 'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4',
    
    # Wind variables (vector components provide more info than speed)
    'u_component_of_wind_10m', 'v_component_of_wind_10m',
    
    # Other important variables
    'vpd', 'rh', 'leaf_area_index_high_vegetation', 'leaf_area_index_low_vegetation',
    'evaporation_from_vegetation_transpiration', 'ppfd_in',
    
    # bio-climatic variables
    'biome', 
]
    selected_vars = [
    # Target variable
    'sap_velocity',
    # One from each radiation group
    'surface_net_solar_radiation', 'sw_in', 'surface_net_thermal_radiation', 'surface_latent_heat_flux',
    
    # Temperature variables
    'ta', 'soil_temperature_level_1', 'soil_temperature_level_2', 'soil_temperature_level_3',# Just one soil level
    
    # Water variables
    'precip', 'swc_shallow', 'swc_deep', 'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4',
    
    # Wind variables (vector components provide more info than speed)
    'u_component_of_wind_10m', 'v_component_of_wind_10m',
    
    # Other important variables
    'vpd', 'rh', 'leaf_area_index_high_vegetation', 'leaf_area_index_low_vegetation',
    'evaporation_from_vegetation_transpiration', 'ppfd_in',
    
    # Categorical
    'biome',
    # Time features
    'Year sin', 'Year cos', 'Month sin', 'Month cos', 'Week sin', 'Week cos',
    # Topographic features
    'elevation', 'slope', 'aspect',
    # Water indices
     'Water availability index', 'Index of water availability',
    # plant trait
    'canopy_height',

]
    all_data = all_data[selected_vars]
    all_data = add_time_features(all_data)
    print(all_data.describe())
    # make use of first 5000 rows
    print(all_data.shape)
    train_rate = 0.8
    train_len = int(all_data.shape[0] * train_rate)
    training_data = all_data[:train_len]
    print(training_data.shape)
    # save the testing data and training data
    # make directories if they don't exist
    print("training shape:", training_data.shape)
    training_data = training_data[[col for col in training_data.columns if 'site' not in col ]]
    training_data = training_data.dropna()
    print("Dataset shape:", training_data.shape)
    # With:
    y = training_data['sap_velocity']
    X = training_data.drop(columns=['sap_velocity'])
    if 'biome' in X.columns:
        X['biome'] = X['biome'].astype('category').cat.codes
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )
    # X_train_scaled, y, feature_names = robust_preprocess_data(training_data, 'sap_velocity')
    print(X_train_scaled.describe())
    # Create output directories if they don't exist
    os.makedirs('plots', exist_ok=True)
    os.makedirs('outputs/checkpoints', exist_ok=True)
    
    # Configuration for supercomputer usage - choose one of the following:
    
    # Option 1: Multi-CPU configuration
    """
    selector = GHGASelector(
        population_size=50,         # Increased from 10 for better results
        generations=50,             # Increased from 10 for better convergence
        mutation_rate=0.1,
        local_search_prob=0.1,
        elite_size=1,
        random_state=42,
        n_jobs=8,                   # Use 8 CPU cores (adjust to your system)
        use_gpu=False,              # Don't use GPU
        checkpoint_interval=5,      # Save checkpoint every 5 generations
        checkpoint_dir='outputs/checkpoints',
        fitness_batch_size=10       # Process 10 chromosomes per batch
    )
    """
    # Option 2: GPU Configuration (commented out)
    
    selector = GHGASelector(
        population_size=100,        # Larger populations work well with GPU
        generations=50,            # More generations for better results
        mutation_rate=0.5,
        local_search_prob=0.1,
        elite_size=5,
        random_state=42,
        n_jobs=-1,                   # Still use some CPU cores
        use_gpu=False,               # Enable GPU acceleration                  # Use first GPU
        checkpoint_interval=5,
        checkpoint_dir='outputs/checkpoints'
    )
    
    
    # Option 3: Resume from checkpoint (commented out)
    """
    # First, create the selector with the same parameters
    selector = GHGASelector(
        population_size=50,
        generations=100,            # Set to total desired generations
        mutation_rate=0.1,
        local_search_prob=0.1,
        elite_size=1,
        random_state=42,
        n_jobs=8,
        checkpoint_interval=5,
        checkpoint_dir='outputs/checkpoints'
    )
    
    # Then pass the checkpoint file when fitting
    selector.fit(X_train_scaled, y, 
                resume_from='outputs/checkpoints/ghga_checkpoint_gen25_20250301_120145.pkl')
    """
    # make a directory for the outputs
    var_select_dir = Path('outputs/Var_select')
    if not var_select_dir.exists():
        var_select_dir.mkdir(parents=True, exist_ok=True)
    # Fit selector with regular approach
    print("\nStarting feature selection...")
    selector.fit(X_train_scaled, y)
    
    # Get selected features
    selected_features = selector.get_feature_names()
    print("\nSelected features:", selected_features)
    print("Number of selected features:", len(selected_features))
    
    # Get feature importance
    feature_importance = selector.get_feature_importance()
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x=feature_importance.index, y=feature_importance.values)
    plt.xticks(rotation=45)
    plt.title('Feature Importance Scores')
    plt.tight_layout()
    plt.savefig(var_select_dir /'feature_importance_plot.png')
    
    # Plot convergence curve with generation times
    selector.plot_convergence(save_path= var_select_dir/'convergence_plot.png')
    
    # Get detailed summary including runtime statistics
    summary = selector.summary()
    print("\nSelection Summary:")
    print("Total features:", summary['n_total_features'])
    print("Selected features:", summary['n_selected_features'])
    print("Selected feature names:", summary['selected_features'])
    print("Best fitness score:", round(summary['best_fitness'], 4))
    
    print("\nPerformance Info:")
    print(f"Total runtime: {summary['convergence']['total_runtime']:.2f} seconds")
    print(f"Average time per generation: {summary['convergence']['avg_generation_time']:.2f} seconds")
    print(f"Hardware used: {summary['hardware_info']['n_jobs']} CPU cores, GPU: {summary['hardware_info']['gpu']}")
    
    print("\nConvergence info:")
    print("Initial fitness:", round(summary['convergence']['initial_fitness'], 4))
    print("Final fitness:", round(summary['convergence']['final_fitness'], 4))
    
    # Transform data using selected features
    X_train_selected = selector.transform(X_train_scaled)
    
    print("\nTransformed data shapes:")
    print("Training data:", X_train_selected.shape)
    
    # Calculate total runtime
    total_runtime = time.time() - start_time
    print(f"\nTotal script runtime: {total_runtime:.2f} seconds")


if __name__ == "__main__":
    main()