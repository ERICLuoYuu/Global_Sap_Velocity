import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.Analyzers import sap_analyzer_parallel
from src.Analyzers import env_analyzer_parallel

def create_spaced_gaps(n_gaps, min_spacing, max_idx):
    """
    Create a list of random gap indices with minimum spacing between them.
    
    Parameters:
        n_gaps (int): Number of gaps to create
        min_spacing (int): Minimum distance between gap indices
        max_idx (int): Maximum index value (exclusive)
        
    Returns:
        list: Sorted list of gap indices
    """
    gaps = []
    attempts = 0
    max_attempts = n_gaps * 10  # Prevent infinite loop
    
    while len(gaps) < n_gaps and attempts < max_attempts:
        new_gap = np.random.randint(0, max_idx)
        # Check if gap maintains minimum spacing
        if all(abs(new_gap - g) >= min_spacing for g in gaps):
            gaps.append(new_gap)
        attempts += 1
    
    return sorted(gaps)

def extend_gaps(gaps, size, max_idx):
    """
    Extend each gap to include consecutive points.
    
    Parameters:
        gaps (list): Original gap indices
        size (int): Gap size (number of consecutive points)
        max_idx (int): Maximum index value (exclusive)
        
    Returns:
        list: Extended gap indices
    """
    extended_gaps = []
    for gap in gaps:
        for i in range(size):
            if gap + i < max_idx:
                extended_gaps.append(gap + i)
    return extended_gaps

def evaluate_interpolation(original, interpolated, mask):
    """
    Calculate performance metrics for interpolation.
    
    Parameters:
        original (pd.Series): Original data
        interpolated (pd.Series): Interpolated data
        mask (np.array): Boolean mask where False indicates gap positions
        
    Returns:
        dict: Dictionary of performance metrics
    """
    # Extract values at gap positions
    orig_gaps = original[~mask]
    interp_gaps = interpolated[~mask]
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((orig_gaps - interp_gaps)**2))
    
    # Calculate R²
    ss_total = np.sum((orig_gaps - np.mean(orig_gaps))**2)
    ss_residual = np.sum((orig_gaps - interp_gaps)**2)
    r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    
    # Calculate Mean Absolute Error
    mae = np.mean(np.abs(orig_gaps - interp_gaps))
    
    # Calculate Mean Absolute Percentage Error
    non_zero_mask = orig_gaps != 0
    mape = np.mean(np.abs((orig_gaps[non_zero_mask] - interp_gaps[non_zero_mask]) / orig_gaps[non_zero_mask])) * 100 if any(non_zero_mask) else float('nan')
    
    return {
        'rmse': rmse,
        'r2': r2,
        'mae': mae,
        'mape': mape
    }

def test_gap_interpolation(data, gap_size, n_gaps=500, min_spacing=10):
    """
    Test interpolation performance for gaps of a specific size.
    
    Parameters:
        data (pd.Series): Original time series data
        gap_size (int): Size of gaps to create
        n_gaps (int): Number of gaps to create
        min_spacing (int): Minimum spacing between gaps
        
    Returns:
        dict: Performance metrics
    """
    # Create initial gaps
    gaps = create_spaced_gaps(n_gaps, min_spacing + gap_size - 1, len(data) - gap_size + 1)
    
    # Extend gaps to the specified size
    extended_gaps = extend_gaps(gaps, gap_size, len(data))
    
    # Create mask and apply to data
    mask = np.ones(len(data), dtype=bool)
    mask[extended_gaps] = False
    
    # Create data with gaps
    data_with_gaps = data.copy()
    data_with_gaps[~mask] = np.nan
    
    # Interpolate gaps
    interpolated_data = data_with_gaps.interpolate()
    
    # Evaluate interpolation
    metrics = evaluate_interpolation(data, interpolated_data, mask)
    
    return {
        'data_with_gaps': data_with_gaps,
        'interpolated_data': interpolated_data,
        'gaps': gaps,
        'extended_gaps': extended_gaps,
        'mask': mask,
        'metrics': metrics
    }

# Main code execution
if __name__ == "__main__":
    # Read the sap velocity data
    sap_velocity = pd.read_csv('./outputs/processed_data/sap/filtered/DEU_HIN_OAK_sapf_data_filtered.csv', parse_dates=['TIMESTAMP'])
    sap_velocity = sap_velocity.iloc[2904:6575, 3]
    print(f"Original data shape: {sap_velocity.shape}")
    print("Original data head:")
    print(sap_velocity.head())
    print(f"Original data mean: {sap_velocity.mean()}")
    
    # Test interpolation for gaps of size 1, 2, and 3
    gap_sizes = [1, 2, 3]
    results = {}
    
    for size in gap_sizes:
        print(f"\n{'='*50}")
        print(f"Testing gap size: {size}")
        print(f"{'='*50}")
        
        results[size] = test_gap_interpolation(sap_velocity, size)
        
        # Print results
        metrics = results[size]['metrics']
        gaps_created = len(results[size]['extended_gaps'])
        
        print(f"Total gaps created: {gaps_created} (from {len(results[size]['gaps'])} positions)")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
    # Save results
    results_dir = Path('./outputs/statistics/gap_interpolation')
    save_dir = Path('./outputs/figures/interpolation_results')
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    results_file = results_dir / 'gap_interpolation_results.csv'
    pd.DataFrame(results).to_csv(results_file)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    # Optional: Visualize results
    plt.figure(figsize=(12, 8))
    
    x_labels = [f"Size {size}" for size in gap_sizes]
    rmse_values = [results[size]['metrics']['rmse'] for size in gap_sizes]
    r2_values = [results[size]['metrics']['r2'] for size in gap_sizes]
    
    plt.subplot(2, 1, 1)
    plt.bar(x_labels, rmse_values)
    plt.title('RMSE by Gap Size')
    plt.ylabel('RMSE')
    
    plt.subplot(2, 1, 2)
    plt.bar(x_labels, r2_values)
    plt.title('R² by Gap Size')
    plt.ylabel('R²')
    
    plt.tight_layout()
    plt.savefig(save_dir/'gap_interpolation_performance.png')
    plt.close()
    
    print("\nAnalysis complete. Results saved to gap_interpolation_performance.png")