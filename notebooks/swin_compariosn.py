import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List
from sklearn.linear_model import LinearRegression

def compare_and_downscale_sw_in(
    era5_file: str,
    env_data_dir: str,
    output_dir: str = './outputs/figures',
    era5_variable: str = 'surface_solar_radiation_downwards_hourly',
    env_variable: str = 'sw_in',
    era5_unit_conversion: float = 1/3600,  # J/m^2 to W/m^2
    scale: str = '1h'
) -> None:
    """
    Loads, compares, downscales, and re-compares sw_in data from ERA5 and
    environmental sensors for all sites combined.
    """
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # --- 1. Load and Aggregate Data (Same as before) ---
    try:
        print(f"Loading ERA5 data from {era5_file}...")
        era5_data = pd.read_csv(era5_file)
        era5_data['TIMESTAMP'] = pd.to_datetime(era5_data['datetime'], utc=True)
        era5_data.set_index('TIMESTAMP', inplace=True)
        era5_data[era5_variable] *= era5_unit_conversion
    except Exception as e:
        print(f"An error occurred while loading ERA5 data: {e}")
        return

    env_files = list(Path(env_data_dir).glob("*_env_data_outliers_removed.csv"))
    print(f"Found {len(env_files)} environmental data files. Processing...")
    
    all_era5_matched, all_env_matched = [], []
    for env_file in env_files:
        try:
            site_name = '_'.join(env_file.stem.split('_')[:-4])
            env_data = pd.read_csv(env_file)
            if env_variable not in env_data.columns: continue
            
            env_data['TIMESTAMP'] = pd.to_datetime(env_data['TIMESTAMP'], utc=True)
            env_data.set_index('TIMESTAMP', inplace=True)
            env_data_resampled = env_data[[env_variable]].resample(scale).mean()
            
            site_era5 = era5_data[era5_data['site_name'] == site_name]
            if site_era5.empty: continue
            
            site_era5_resampled = site_era5[[era5_variable]].resample(scale).mean()
            merged_data = pd.merge(env_data_resampled, site_era5_resampled, left_index=True, right_index=True, how='inner').dropna()
            
            if not merged_data.empty:
                all_env_matched.append(merged_data[env_variable])
                all_era5_matched.append(merged_data[era5_variable])
        except Exception as e:
            print(f"Error processing {env_file.name}: {e}")
            continue

    if not all_env_matched:
        print("Error: No valid data could be aggregated.")
        return

    final_env_data = pd.concat(all_env_matched).abs()
    final_era5_data = pd.concat(all_era5_matched).abs()

    # --- 2. "Before" Comparison ---
    print("\n--- 1. Comparison Before Downscaling ---")
    plot_comparison(
        x_data=final_era5_data,
        y_data=final_env_data,
        x_label='Original ERA5 sw_in (W/m²)',
        y_label='Original Environmental Sensor sw_in (W/m²)',
        title='Comparison Before Downscaling (All Sites)',
        output_path=Path(output_dir) / 'sw_in_comparison_before_downscaling.png'
    )
    
    # --- 3. Train Linear Regression Model and Downscale ---
    print("\n--- 2. Training Linear Regression Model for Downscaling ---")
    
    # Reshape data for scikit-learn: X = predictor (ERA5), y = target (Sensor)
    X_era5 = final_era5_data.values.reshape(-1, 1)
    y_sensor = final_env_data.values

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_era5, y_sensor)

    # Get model parameters
    slope = model.coef_[0]
    intercept = model.intercept_
    print(f"Model trained. Equation: Sensor_sw_in = {slope:.4f} * ERA5_sw_in + {intercept:.4f}")

    # Apply the model to downscale the ERA5 data
    downscaled_era5_data = model.predict(X_era5)
    
    # --- 4. "After" Comparison ---
    print("\n--- 3. Comparison After Downscaling ---")
    plot_comparison(
        x_data=pd.Series(downscaled_era5_data, index=final_env_data.index),
        y_data=final_env_data,
        x_label='Downscaled ERA5 sw_in (W/m²)',
        y_label='Original Environmental Sensor sw_in (W/m²)',
        title='Comparison After Linear Regression Downscaling (All Sites)',
        output_path=Path(output_dir) / 'sw_in_comparison_after_downscaling.png'
    )


def plot_comparison(x_data: pd.Series, y_data: pd.Series, x_label: str, y_label: str, title: str, output_path: Path):
    """Helper function to create and save a scatter plot comparison."""
    
    # Calculate statistics
    r_value = np.corrcoef(x_data, y_data)[0, 1]
    r2 = r_value**2
    bias = (y_data - x_data).mean()
    rmse = np.sqrt(((y_data - x_data)**2).mean())
    n_points = len(x_data)

    print(f"Results for '{title}':")
    print(f"  R-squared (R²): {r2:.4f}")
    print(f"  Bias: {bias:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    # Create plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 10), layout='constrained')
    ax.scatter(x_data, y_data, alpha=0.3, s=15, label='Hourly Data')
    
    min_val = min(x_data.min(), y_data.min())
    max_val = max(x_data.max(), y_data.max())
    padding = (max_val - min_val) * 0.05
    
    ax.set_xlim(min_val - padding, max_val + padding)
    ax.set_ylim(min_val - padding, max_val + padding)
    
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='1:1 Line')
    
    m, b = np.polyfit(x_data, y_data, 1)
    ax.plot(x_data, m * x_data + b, 'r-', alpha=0.8, label='Regression Line')

    stats_text = (f'R² = {r2:.3f}\nBias = {bias:.3f}\nRMSE = {rmse:.3f}\nN = {n_points}')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    ERA5_DATA_FILE = 'data/raw/extracted_data/era5land_site_data/sapwood/era5_extracted_data.csv'
    ENV_DATA_DIRECTORY = 'outputs/processed_data/sapwood/env/outliers_removed'
    
    compare_and_downscale_sw_in(
        era5_file=ERA5_DATA_FILE,
        env_data_dir=ENV_DATA_DIRECTORY
    )