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


# Main example
def main():
    # Start timing
    start_time = time.time()
    
    all_data = pd.read_csv('outputs/processed_data/merged/site/gap_filled_size1_with_era5/all_biomes_merged_data.csv')
    all_data = all_data.dropna().iloc[:10000]
    # make use of first 5000 rows
    print(all_data.shape)
    # randomly split the data into training and testing
    training_data = all_data.sample(frac=0.8, random_state=42)
    print(training_data.shape)
    testing_data = all_data.drop(training_data.index)
    # save the testing data and training data
    # make directories if they don't exist
    
    training_data = training_data.set_index('TIMESTAMP').sort_index()
    print("training shape:", training_data.shape)
    training_data = training_data[[col for col in training_data.columns if 'site' not in col ]]
    training_data = training_data.dropna()
    print("Dataset shape:", training_data.shape)
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
        population_size=50,        # Larger populations work well with GPU
        generations=50,            # More generations for better results
        mutation_rate=0.5,
        local_search_prob=0.1,
        elite_size=5,
        random_state=42,
        n_jobs=4,                   # Still use some CPU cores
        use_gpu=True,               # Enable GPU acceleration
        gpu_id=0,                   # Use first GPU
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