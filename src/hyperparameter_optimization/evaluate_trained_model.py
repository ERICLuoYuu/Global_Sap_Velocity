"""
Model Evaluation Script for XGBoost/ML models trained with spatial cross-validation.
Compatible with models trained using the spatial CV training script.

This script:
1. Auto-loads model, scaler, and config from the same directory
2. Applies the same preprocessing as training
3. Generates comprehensive evaluation plots
4. Supports evaluation by site and PFT
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
import sys
import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Suppress warnings
warnings.filterwarnings("ignore")

# Add parent directory to path if needed
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


def sanitize_filename(name):
    """
    Sanitize a string to be safe for use as a filename.
    Replaces or removes characters that are problematic on Windows/Linux.
    """
    # Replace problematic characters
    replacements = {
        '/': '_',
        '\\': '_',
        ':': '_',
        '*': '_',
        '?': '_',
        '"': '_',
        '<': '_',
        '>': '_',
        '|': '_',
        ' ': '_',
    }
    
    safe_name = str(name)
    for char, replacement in replacements.items():
        safe_name = safe_name.replace(char, replacement)
    
    # Remove any remaining non-alphanumeric characters except underscore and hyphen
    safe_name = ''.join(c if c.isalnum() or c in ('_', '-', '.') else '_' for c in safe_name)
    
    # Remove consecutive underscores
    while '__' in safe_name:
        safe_name = safe_name.replace('__', '_')
    
    # Strip leading/trailing underscores
    safe_name = safe_name.strip('_')
    
    return safe_name if safe_name else 'unnamed'


def setup_logging(log_dir=None):
    """Set up logging configuration."""
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / 'evaluation.log'
        handlers = [
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    else:
        handlers = [logging.StreamHandler()]
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def load_model_artifacts(model_path: str):
    """
    Load model, scaler, and config from the model directory.
    
    Automatically finds the corresponding scaler and config files
    based on the model filename pattern.
    
    Parameters:
    -----------
    model_path : str
        Path to the model file (e.g., FINAL_xgb_default_base_daily.json)
        
    Returns:
    --------
    tuple: (model, scaler, config, model_dir)
    """
    model_path = Path(model_path)
    model_dir = model_path.parent
    model_name = model_path.stem  # e.g., "FINAL_xgb_default_base_daily"
    
    # Extract run_id from model name
    # Pattern: FINAL_{model_type}_{run_id}
    parts = model_name.split('_')
    if parts[0] == 'FINAL' and len(parts) >= 3:
        model_type = parts[1]
        run_id = '_'.join(parts[2:])
    else:
        # Fallback: try to find matching files
        model_type = 'xgb'
        run_id = model_name
    
    logging.info(f"Loading model artifacts from: {model_dir}")
    logging.info(f"  Model type: {model_type}, Run ID: {run_id}")
    
    # Load model
    logging.info(f"Loading model: {model_path}")
    if model_path.suffix == '.json':
        model = xgb.Booster()
        model.load_model(str(model_path))
    elif model_path.suffix == '.joblib':
        model = joblib.load(model_path)
    else:
        raise ValueError(f"Unsupported model format: {model_path.suffix}")
    
    # Find and load scaler
    scaler_patterns = [
        f'FINAL_scaler_{run_id}.pkl',
        f'scaler_{run_id}.pkl',
        'scaler.pkl',
        'FINAL_scaler.pkl'
    ]
    
    scaler = None
    for pattern in scaler_patterns:
        scaler_path = model_dir / pattern
        if scaler_path.exists():
            logging.info(f"Loading scaler: {scaler_path}")
            scaler = joblib.load(scaler_path)
            break
    
    if scaler is None:
        logging.warning("No scaler found! Will skip feature scaling.")
    
    # Find and load config
    config_patterns = [
        f'FINAL_config_{run_id}.json',
        f'config_{run_id}.json',
        'config.json',
        'FINAL_config.json'
    ]
    
    config = None
    for pattern in config_patterns:
        config_path = model_dir / pattern
        if config_path.exists():
            logging.info(f"Loading config: {config_path}")
            with open(config_path, 'r') as f:
                config = json.load(f)
            break
    
    if config is None:
        logging.warning("No config found! Using default settings.")
        config = {
            'preprocessing': {'target_transform': 'log1p', 'feature_scaling': 'StandardScaler'},
            'feature_names': None
        }
    
    return model, scaler, config, model_dir


def add_time_features(df, datetime_column='TIMESTAMP'):
    """Create cyclical time features from a datetime column."""
    df = df.copy()
    
    if datetime_column in df.columns:
        date_time = pd.to_datetime(df[datetime_column])
    elif isinstance(df.index, pd.DatetimeIndex):
        date_time = df.index
    else:
        logging.warning("No datetime column found for time features")
        return df
    
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = 365.2425 * day
    
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    df['Week sin'] = np.sin(timestamp_s * (2 * np.pi / (7 * day)))
    df['Week cos'] = np.cos(timestamp_s * (2 * np.pi / (7 * day)))
    df['Month sin'] = np.sin(timestamp_s * (2 * np.pi / (30.44 * day)))
    df['Month cos'] = np.cos(timestamp_s * (2 * np.pi / (30.44 * day)))
    
    return df


def prepare_data(data_dir: str, config: dict, target_col: str = 'sap_velocity', 
                 time_scale: str = 'daily', is_only_day: bool = False):
    """
    Load and prepare data for evaluation using the same preprocessing as training.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing CSV data files
    config : dict
        Model configuration with feature_names
    target_col : str
        Name of target column
    time_scale : str
        Time scale of data ('daily' or 'hourly')
    is_only_day : bool
        Whether to filter for daytime only
        
    Returns:
    --------
    tuple: (X, y, site_ids, pfts, metadata_df)
    """
    data_dir = Path(data_dir)
    data_list = sorted(list(data_dir.glob(f'*{time_scale}.csv')))
    data_list = [f for f in data_list if 'all_biomes_merged' not in f.name]
    
    if not data_list:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    logging.info(f"Found {len(data_list)} data files in {data_dir}")
    
    # Get feature names from config
    feature_names = config.get('feature_names', None)
    if feature_names is None:
        raise ValueError("No feature_names found in config. Cannot proceed.")
    
    logging.info(f"Expected features ({len(feature_names)}): {feature_names[:10]}...")
    
    # Identify PFT dummy columns
    all_possible_pft_types = ['MF', 'DNF', 'ENF', 'EBF', 'WSA', 'WET', 'DBF', 'SAV']
    pft_dummy_cols = [col for col in feature_names if col in all_possible_pft_types]
    base_feature_cols = [col for col in feature_names if col not in all_possible_pft_types]
    
    logging.info(f"Base features: {len(base_feature_cols)}, PFT dummies: {pft_dummy_cols}")
    
    all_X, all_y, all_sites, all_pfts = [], [], [], []
    site_metadata = []
    
    for data_file in data_list:
        try:
            df = pd.read_csv(data_file, parse_dates=['TIMESTAMP'])
            
            # Get site info
            site_id = df['site_name'].iloc[0] if 'site_name' in df.columns else data_file.stem
            
            # Skip certain sites if needed
            if site_id.startswith('CZE'):
                logging.info(f"Skipping site {site_id}")
                continue
            
            # Filter for daytime if requested
            if is_only_day and 'sw_in' in df.columns:
                df = df[df['sw_in'] > 10]
                if df.empty:
                    continue
            
            # Get coordinates
            lat_col = next((col for col in df.columns if col.lower() in ['lat', 'latitude', 'latitude_x']), None)
            lon_col = next((col for col in df.columns if col.lower() in ['lon', 'longitude', 'longitude_x']), None)
            pft_col = next((col for col in df.columns if col.lower() in ['pft', 'plant_functional_type', 'biome']), None)
            
            if lat_col and lon_col:
                df['latitude'] = df[lat_col].median()
                df['longitude'] = df[lon_col].median()
            
            pft_value = df[pft_col].mode()[0] if pft_col and pft_col in df.columns else 'Unknown'
            
            # Add time features
            df.set_index('TIMESTAMP', inplace=True)
            df = add_time_features(df)
            
            # Check for missing base columns
            missing_cols = [col for col in base_feature_cols if col not in df.columns]
            if missing_cols:
                logging.warning(f"Site {site_id}: Missing columns {missing_cols}. Skipping.")
                continue
            
            # Check target column
            if target_col not in df.columns:
                logging.warning(f"Site {site_id}: Missing target column {target_col}. Skipping.")
                continue
            
            # Select base features
            df_features = df[base_feature_cols].copy()
            
            # One-hot encode PFT
            if pft_dummy_cols:
                for pft_type in all_possible_pft_types:
                    df_features[pft_type] = 1.0 if pft_value == pft_type else 0.0
            
            # Ensure EXACT column order matching config
            # Check all expected features are present
            missing_features = [col for col in feature_names if col not in df_features.columns]
            if missing_features:
                logging.warning(f"Site {site_id}: Missing features {missing_features}. Skipping.")
                continue
            
            # Reorder columns to match exactly the config order
            df_features = df_features[feature_names]
            
            # Get target
            y_site = df[target_col].values
            
            # Clean data
            df_features = df_features.astype(np.float32)
            df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Create mask for valid rows
            valid_mask = ~df_features.isna().any(axis=1) & ~np.isnan(y_site)
            
            X_site = df_features.values[valid_mask]
            y_site = y_site[valid_mask]
            
            if len(y_site) == 0:
                logging.warning(f"Site {site_id}: No valid data after cleaning. Skipping.")
                continue
            
            all_X.append(X_site)
            all_y.append(y_site)
            all_sites.extend([site_id] * len(y_site))
            all_pfts.extend([pft_value] * len(y_site))
            
            site_metadata.append({
                'site_id': site_id,
                'pft': pft_value,
                'n_records': len(y_site),
                'latitude': df['latitude'].iloc[0] if 'latitude' in df.columns else None,
                'longitude': df['longitude'].iloc[0] if 'longitude' in df.columns else None,
                'y_mean': np.mean(y_site),
                'y_std': np.std(y_site),
                'y_max': np.max(y_site)
            })
            
            logging.info(f"Site {site_id}: Loaded {len(y_site)} records (PFT: {pft_value})")
            
        except Exception as e:
            logging.error(f"Error processing {data_file.name}: {e}")
            continue
    
    if not all_X:
        raise ValueError("No valid data could be loaded!")
    
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    sites = np.array(all_sites)
    pfts = np.array(all_pfts)
    metadata_df = pd.DataFrame(site_metadata)
    
    logging.info(f"Total data loaded: {len(y)} records from {len(metadata_df)} sites")
    logging.info(f"Feature matrix shape: {X.shape}")
    logging.info(f"Features used: {feature_names}")
    
    return X, y, sites, pfts, metadata_df


def evaluate_model(model, X, y, scaler, config):
    """
    Make predictions and calculate metrics.
    
    Parameters:
    -----------
    model : xgb.Booster or sklearn model
        Trained model
    X : np.ndarray
        Features
    y : np.ndarray
        True target values (original scale)
    scaler : StandardScaler
        Feature scaler
    config : dict
        Model configuration
        
    Returns:
    --------
    tuple: (predictions, actuals, metrics_dict)
    """
    # Scale features
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    
    # Get feature names from config
    feature_names = config.get('feature_names', None)
    
    # Verify feature count matches
    if feature_names is not None:
        expected_n_features = len(feature_names)
        actual_n_features = X_scaled.shape[1]
        
        if expected_n_features != actual_n_features:
            logging.error(f"Feature count mismatch! Expected {expected_n_features}, got {actual_n_features}")
            logging.error(f"Expected features: {feature_names}")
            raise ValueError(f"Feature count mismatch: model expects {expected_n_features} features, data has {actual_n_features}")
        
        logging.info(f"Feature count verified: {actual_n_features} features")
    
    # Make predictions
    if isinstance(model, xgb.Booster):
        # Check what feature names the model was trained with
        model_feature_names = model.feature_names
        logging.info(f"Model feature names: {model_feature_names[:5] if model_feature_names else 'None'}...")
        
        # If model was trained with numeric feature names (e.g., '0', '1', '2', ...)
        # we need to pass those same string names to DMatrix
        if model_feature_names and all(name.isdigit() for name in model_feature_names):
            logging.info("Model was trained with numeric feature names - using numeric string indices")
            # Create list of string indices to match model's feature names
            numeric_feature_names = [str(i) for i in range(X_scaled.shape[1])]
            dmatrix = xgb.DMatrix(X_scaled, feature_names=numeric_feature_names)
        elif model_feature_names is not None:
            # Model was trained with actual feature names - use them
            logging.info("Model was trained with named features - using model's feature names")
            dmatrix = xgb.DMatrix(X_scaled, feature_names=model_feature_names)
        elif feature_names is not None:
            # Use config feature names
            logging.info("Using feature names from config")
            dmatrix = xgb.DMatrix(X_scaled, feature_names=feature_names)
        else:
            # Fallback: no feature names
            logging.info("No feature names available - using default")
            dmatrix = xgb.DMatrix(X_scaled)
        
        y_pred_transformed = model.predict(dmatrix)
    else:
        y_pred_transformed = model.predict(X_scaled)
    
    # Inverse transform predictions
    target_transform = config.get('preprocessing', {}).get('target_transform', 'log1p')
    
    if target_transform == 'log1p':
        y_pred = np.expm1(y_pred_transformed)
    elif target_transform == 'sqrt':
        y_pred = y_pred_transformed ** 2
    elif target_transform == 'none' or target_transform is None:
        y_pred = y_pred_transformed
    else:
        logging.warning(f"Unknown transform '{target_transform}', using raw predictions")
        y_pred = y_pred_transformed
    
    # Clip negative predictions
    y_pred = np.clip(y_pred, 0, None)
    
    # Calculate metrics
    # Filter out any NaN/inf
    valid_mask = np.isfinite(y) & np.isfinite(y_pred)
    y_valid = y[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    metrics = {
        'r2': r2_score(y_valid, y_pred_valid),
        'rmse': np.sqrt(mean_squared_error(y_valid, y_pred_valid)),
        'mae': mean_absolute_error(y_valid, y_pred_valid),
        'n_samples': len(y_valid),
        'n_filtered': len(y) - len(y_valid)
    }
    
    logging.info(f"Evaluation Metrics:")
    logging.info(f"  R²   = {metrics['r2']:.4f}")
    logging.info(f"  RMSE = {metrics['rmse']:.4f}")
    logging.info(f"  MAE  = {metrics['mae']:.4f}")
    logging.info(f"  N    = {metrics['n_samples']} ({metrics['n_filtered']} filtered)")
    
    return y_pred, y, metrics


def evaluate_by_group(y_true, y_pred, groups, group_name='Group'):
    """
    Calculate metrics for each group (site or PFT).
    
    Returns:
    --------
    pd.DataFrame with metrics per group
    """
    results = []
    
    for group in np.unique(groups):
        mask = groups == group
        y_t = y_true[mask]
        y_p = y_pred[mask]
        
        # Filter valid
        valid = np.isfinite(y_t) & np.isfinite(y_p)
        y_t, y_p = y_t[valid], y_p[valid]
        
        if len(y_t) < 10:
            continue
        
        results.append({
            group_name: group,
            'N': len(y_t),
            'R2': r2_score(y_t, y_p),
            'RMSE': np.sqrt(mean_squared_error(y_t, y_p)),
            'MAE': mean_absolute_error(y_t, y_p),
            'Mean_Observed': np.mean(y_t),
            'Mean_Predicted': np.mean(y_p),
            'Bias': np.mean(y_p - y_t)
        })
    
    return pd.DataFrame(results).sort_values('R2', ascending=False)


def plot_predictions_vs_actual(y_true, y_pred, metrics, output_path, title_suffix=''):
    """Create scatter plot of predictions vs actual values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Filter valid
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t, y_p = y_true[valid], y_pred[valid]
    
    # Left: Predictions vs Actuals
    axes[0].scatter(y_t, y_p, alpha=0.3, s=10, color='steelblue', edgecolors='none')
    axes[0].set_xlabel('Observed Sap Velocity (cm³ cm⁻² h⁻¹)', fontsize=12)
    axes[0].set_ylabel('Predicted Sap Velocity (cm³ cm⁻² h⁻¹)', fontsize=12)
    axes[0].set_title(f'Predictions vs Observations{title_suffix}\n'
                      f'$R^2 = {metrics["r2"]:.3f}$, RMSE = {metrics["rmse"]:.3f}, N = {metrics["n_samples"]}',
                      fontsize=12, fontweight='bold')
    
    # 1:1 line
    min_val = min(y_t.min(), y_p.min())
    max_val = max(y_t.max(), y_p.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label='1:1 Line')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal', 'box')
    
    # Right: Residuals
    residuals = y_t - y_p
    axes[1].scatter(y_p, residuals, alpha=0.3, s=10, color='coral', edgecolors='none')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Sap Velocity', fontsize=12)
    axes[1].set_ylabel('Residuals (Observed - Predicted)', fontsize=12)
    axes[1].set_title(f'Residual Plot\nMAE = {metrics["mae"]:.3f}', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved plot: {output_path}")


def plot_by_group_performance(metrics_df, group_name, output_path):
    """Create bar chart of R² by group."""
    if metrics_df.empty:
        logging.warning(f"No data to plot for {group_name}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(metrics_df) * 0.4)))
    
    # Sort by R²
    df_sorted = metrics_df.sort_values('R2', ascending=True)
    
    # Left: R² by group
    colors = plt.cm.RdYlGn((df_sorted['R2'].values - df_sorted['R2'].min()) / 
                           (df_sorted['R2'].max() - df_sorted['R2'].min() + 0.01))
    
    bars = axes[0].barh(df_sorted[group_name].astype(str), df_sorted['R2'], color=colors, edgecolor='black')
    axes[0].set_xlabel('R² Score', fontsize=12)
    axes[0].set_ylabel(group_name, fontsize=12)
    axes[0].set_title(f'Model Performance by {group_name}', fontsize=13, fontweight='bold')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, r2, n in zip(bars, df_sorted['R2'], df_sorted['N']):
        axes[0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{r2:.2f} (n={n})', va='center', fontsize=9)
    
    # Right: RMSE by group
    df_sorted_rmse = metrics_df.sort_values('RMSE', ascending=False)
    colors_rmse = plt.cm.RdYlGn_r((df_sorted_rmse['RMSE'].values - df_sorted_rmse['RMSE'].min()) / 
                                   (df_sorted_rmse['RMSE'].max() - df_sorted_rmse['RMSE'].min() + 0.01))
    
    bars = axes[1].barh(df_sorted_rmse[group_name].astype(str), df_sorted_rmse['RMSE'], 
                        color=colors_rmse, edgecolor='black')
    axes[1].set_xlabel('RMSE', fontsize=12)
    axes[1].set_ylabel(group_name, fontsize=12)
    axes[1].set_title(f'Model Error by {group_name}', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved plot: {output_path}")


def plot_distribution_comparison(y_true, y_pred, output_path):
    """Plot distribution comparison and Q-Q plot."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t, y_p = y_true[valid], y_pred[valid]
    residuals = y_t - y_p
    
    # Distribution comparison
    axes[0].hist(y_t, bins=50, alpha=0.6, label='Observed', color='blue', edgecolor='black')
    axes[0].hist(y_p, bins=50, alpha=0.6, label='Predicted', color='orange', edgecolor='black')
    axes[0].set_xlabel('Sap Velocity', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution Comparison', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Residual histogram
    axes[1].hist(residuals, bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].axvline(x=np.mean(residuals), color='blue', linestyle='-', linewidth=2, label=f'Mean: {np.mean(residuals):.2f}')
    axes[1].set_xlabel('Residual (Observed - Predicted)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Residual Distribution', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[2])
    axes[2].set_title('Q-Q Plot of Residuals', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved plot: {output_path}")


def plot_error_vs_magnitude(y_true, y_pred, output_path):
    """Plot error analysis vs observation magnitude."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t, y_p = y_true[valid], y_pred[valid]
    
    abs_error = np.abs(y_t - y_p)
    rel_error = abs_error / (y_t + 0.1) * 100  # Relative error in %
    
    # Absolute error vs magnitude
    axes[0].scatter(y_t, abs_error, alpha=0.3, s=10, color='steelblue')
    axes[0].set_xlabel('Observed Sap Velocity', fontsize=12)
    axes[0].set_ylabel('Absolute Error', fontsize=12)
    axes[0].set_title('Absolute Error vs Observation Magnitude', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Binned analysis
    bins = np.percentile(y_t, np.arange(0, 101, 10))
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_errors = []
    
    for i in range(len(bins) - 1):
        mask = (y_t >= bins[i]) & (y_t < bins[i+1])
        if mask.sum() > 0:
            bin_errors.append(np.median(abs_error[mask]))
        else:
            bin_errors.append(np.nan)
    
    ax2 = axes[0].twinx()
    ax2.plot(bin_centers, bin_errors, 'r-o', linewidth=2, markersize=8, label='Median Error (binned)')
    ax2.set_ylabel('Median Absolute Error', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')
    
    # Relative error distribution by magnitude bin
    axes[1].scatter(y_t, rel_error, alpha=0.3, s=10, color='coral')
    axes[1].set_xlabel('Observed Sap Velocity', fontsize=12)
    axes[1].set_ylabel('Relative Error (%)', fontsize=12)
    axes[1].set_title('Relative Error vs Observation Magnitude', fontsize=13, fontweight='bold')
    axes[1].set_ylim(0, min(200, np.percentile(rel_error, 95)))
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved plot: {output_path}")


def plot_scatter_by_pft(y_true, y_pred, pfts, output_dir, min_samples=10):
    """
    Create individual scatter plots for each PFT (Plant Functional Type).
    
    Parameters:
    -----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    pfts : np.ndarray
        PFT labels for each sample
    output_dir : Path
        Directory to save plots
    min_samples : int
        Minimum samples required to create a plot
    """
    output_dir = Path(output_dir)
    pft_plots_dir = output_dir / 'pft_scatter_plots'
    pft_plots_dir.mkdir(parents=True, exist_ok=True)
    
    unique_pfts = np.unique(pfts)
    logging.info(f"\nCreating scatter plots for {len(unique_pfts)} PFTs...")
    
    # Store metrics for summary plot
    pft_metrics = {}
    
    for pft in unique_pfts:
        mask = pfts == pft
        y_t = y_true[mask]
        y_p = y_pred[mask]
        
        # Filter valid values
        valid = np.isfinite(y_t) & np.isfinite(y_p)
        y_t, y_p = y_t[valid], y_p[valid]
        
        if len(y_t) < min_samples:
            logging.info(f"  Skipping {pft}: only {len(y_t)} samples (min: {min_samples})")
            continue
        
        # Calculate metrics
        r2 = r2_score(y_t, y_p)
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        mae = mean_absolute_error(y_t, y_p)
        n_samples = len(y_t)
        
        pft_metrics[pft] = {'r2': r2, 'rmse': rmse, 'mae': mae, 'n': n_samples}
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Scatter plot
        axes[0].scatter(y_t, y_p, alpha=0.5, s=20, color='steelblue', edgecolors='black', linewidth=0.3)
        axes[0].set_xlabel('Observed Sap Velocity (cm³ cm⁻² h⁻¹)', fontsize=12)
        axes[0].set_ylabel('Predicted Sap Velocity (cm³ cm⁻² h⁻¹)', fontsize=12)
        axes[0].set_title(f'PFT: {pft}\n$R^2 = {r2:.3f}$, RMSE = {rmse:.3f}, n = {n_samples}',
                          fontsize=13, fontweight='bold')
        
        # 1:1 line
        min_val = min(y_t.min(), y_p.min())
        max_val = max(y_t.max(), y_p.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label='1:1 Line')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect('equal', 'box')
        axes[0].set_xlim([min_val - 0.5, max_val + 0.5])
        axes[0].set_ylim([min_val - 0.5, max_val + 0.5])
        
        # Right: Residual plot
        residuals = y_t - y_p
        axes[1].scatter(y_p, residuals, alpha=0.5, s=20, color='coral', edgecolors='black', linewidth=0.3)
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted Sap Velocity', fontsize=12)
        axes[1].set_ylabel('Residuals (Observed - Predicted)', fontsize=12)
        axes[1].set_title(f'Residual Plot - {pft}\nMAE = {mae:.3f}, Bias = {np.mean(residuals):.3f}',
                          fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Sanitize PFT name for filename - replace all problematic characters
        safe_pft_name = sanitize_filename(pft)
        plot_path = pft_plots_dir / f'scatter_{safe_pft_name}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"  {pft}: R²={r2:.3f}, RMSE={rmse:.3f}, n={n_samples} -> {plot_path.name}")
    
    # Create summary grid plot with all PFTs
    if pft_metrics:
        create_pft_summary_grid(y_true, y_pred, pfts, pft_metrics, pft_plots_dir)
    
    return pft_metrics


def create_pft_summary_grid(y_true, y_pred, pfts, pft_metrics, output_dir):
    """
    Create a summary grid showing all PFT scatter plots in one figure.
    """
    n_pfts = len(pft_metrics)
    if n_pfts == 0:
        return
    
    # Calculate grid dimensions
    n_cols = min(3, n_pfts)
    n_rows = (n_pfts + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    # Flatten axes for easy iteration
    if n_pfts == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Sort PFTs by R² for consistent ordering
    sorted_pfts = sorted(pft_metrics.keys(), key=lambda x: pft_metrics[x]['r2'], reverse=True)
    
    for idx, pft in enumerate(sorted_pfts):
        ax = axes[idx]
        
        mask = pfts == pft
        y_t = y_true[mask]
        y_p = y_pred[mask]
        
        valid = np.isfinite(y_t) & np.isfinite(y_p)
        y_t, y_p = y_t[valid], y_p[valid]
        
        metrics = pft_metrics[pft]
        
        # Scatter plot
        ax.scatter(y_t, y_p, alpha=0.4, s=10, color='steelblue', edgecolors='none')
        
        # 1:1 line
        min_val = min(y_t.min(), y_p.min())
        max_val = max(y_t.max(), y_p.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=1.5)
        
        ax.set_xlabel('Observed', fontsize=10)
        ax.set_ylabel('Predicted', fontsize=10)
        ax.set_title(f'{pft}\n$R^2$={metrics["r2"]:.2f}, RMSE={metrics["rmse"]:.2f}, n={metrics["n"]}',
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', 'box')
    
    # Hide unused subplots
    for idx in range(n_pfts, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Model Performance by Plant Functional Type (PFT)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plot_path = output_dir / 'pft_summary_grid.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved PFT summary grid: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained ML model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model file')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing test data CSV files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save evaluation results')
    parser.add_argument('--target_col', type=str, default='sap_velocity',
                        help='Name of target column')
    parser.add_argument('--time_scale', type=str, default='hourly',
                        help='Time scale of data (daily or hourly)')
    parser.add_argument('--is_only_day', action='store_true',
                        help='Filter for daytime data only')
    
    args = parser.parse_args()
    
    # Set up paths
    model_path = Path(args.model)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = model_path.parent / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(output_dir)
    logging.info("=" * 60)
    logging.info("MODEL EVALUATION SCRIPT")
    logging.info("=" * 60)
    logging.info(f"Model: {model_path}")
    logging.info(f"Output: {output_dir}")
    
    # Load model artifacts
    model, scaler, config, model_dir = load_model_artifacts(str(model_path))
    
    # Determine data directory
    if args.data_dir:
        data_dir = args.data_dir
    else:
        # Try to infer from config or use default
        data_dir = f'./outputs/processed_data/sapwood/merged/{args.time_scale}'
        if not Path(data_dir).exists():
            data_dir = f'./outputs/processed_data/merged/site/{args.time_scale}'
    
    logging.info(f"Data directory: {data_dir}")
    
    # Load and prepare data
    X, y, sites, pfts, metadata_df = prepare_data(
        data_dir=data_dir,
        config=config,
        target_col=args.target_col,
        time_scale=args.time_scale,
        is_only_day=args.is_only_day
    )
    
    # Evaluate model
    y_pred, y_true, metrics = evaluate_model(model, X, y, scaler, config)
    
    # Save overall metrics
    metrics_path = output_dir / 'overall_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Saved metrics: {metrics_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'observed': y_true,
        'predicted': y_pred,
        'residual': y_true - y_pred,
        'site': sites,
        'pft': pfts
    })
    predictions_path = output_dir / 'predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    logging.info(f"Saved predictions: {predictions_path}")
    
    # Generate plots
    logging.info("\nGenerating plots...")
    
    # 1. Main predictions vs actual plot
    plot_predictions_vs_actual(
        y_true, y_pred, metrics,
        output_dir / 'predictions_vs_actual.png'
    )
    
    # 2. Distribution comparison
    plot_distribution_comparison(
        y_true, y_pred,
        output_dir / 'distribution_comparison.png'
    )
    
    # 3. Error vs magnitude
    plot_error_vs_magnitude(
        y_true, y_pred,
        output_dir / 'error_vs_magnitude.png'
    )
    
    # 4. Evaluate by site
    site_metrics = evaluate_by_group(y_true, y_pred, sites, 'Site')
    site_metrics.to_csv(output_dir / 'metrics_by_site.csv', index=False)
    logging.info(f"\nPerformance by Site (top 10):\n{site_metrics.head(10).to_string()}")
    
    plot_by_group_performance(
        site_metrics, 'Site',
        output_dir / 'performance_by_site.png'
    )
    
    # 5. Evaluate by PFT
    pft_metrics = evaluate_by_group(y_true, y_pred, pfts, 'PFT')
    pft_metrics.to_csv(output_dir / 'metrics_by_pft.csv', index=False)
    logging.info(f"\nPerformance by PFT:\n{pft_metrics.to_string()}")
    
    plot_by_group_performance(
        pft_metrics, 'PFT',
        output_dir / 'performance_by_pft.png'
    )
    
    # 6. Create individual scatter plots for each PFT
    logging.info("\nGenerating scatter plots by PFT...")
    plot_scatter_by_pft(y_true, y_pred, pfts, output_dir, min_samples=10)
    
    # 7. Individual site scatter plots (for worst performing sites)
    worst_sites = site_metrics.tail(5)['Site'].tolist()
    site_plots_dir = output_dir / 'site_plots'
    site_plots_dir.mkdir(exist_ok=True)
    
    for site_id in worst_sites:
        mask = sites == site_id
        if mask.sum() < 10:
            continue
        
        site_y_true = y_true[mask]
        site_y_pred = y_pred[mask]
        
        site_r2 = r2_score(site_y_true, site_y_pred)
        site_rmse = np.sqrt(mean_squared_error(site_y_true, site_y_pred))
        
        site_metrics_dict = {'r2': site_r2, 'rmse': site_rmse, 'mae': mean_absolute_error(site_y_true, site_y_pred), 'n_samples': len(site_y_true)}
        
        # Sanitize site_id for filename
        safe_site_id = sanitize_filename(site_id)
        
        plot_predictions_vs_actual(
            site_y_true, site_y_pred, site_metrics_dict,
            site_plots_dir / f'{safe_site_id}_predictions.png',
            title_suffix=f' - Site: {site_id}'
        )
    
    # Summary
    logging.info("\n" + "=" * 60)
    logging.info("EVALUATION COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Overall R²:   {metrics['r2']:.4f}")
    logging.info(f"Overall RMSE: {metrics['rmse']:.4f}")
    logging.info(f"Overall MAE:  {metrics['mae']:.4f}")
    logging.info(f"\nResults saved to: {output_dir}")
    
    return metrics


if __name__ == "__main__":
    main()