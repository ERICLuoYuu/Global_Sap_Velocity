"""
Machine Learning model with AutoML (FLAML) and spatial cross-validation approach for site-based prediction.
Implements group-based spatial cross-validation with proper time windowing.
Preserves StratifiedGroupKFold for both outer and inner CV loops.
"""
from pathlib import Path
import sys
import os
import argparse
import logging
import warnings
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no GUI)
# Set environment variables for determinism BEFORE importing TensorFlow/other libraries
os.environ['PYTHONHASHSEED'] = '42'

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the randomization control module first
from src.utils.random_control import (
    set_seed, get_seed, deterministic
)

# Import the time series windowing modules
from src.hyperparameter_optimization.timeseries_processor1 import (
    TimeSeriesSegmenter, WindowGenerator
)
from src.hyperparameter_optimization.plot_fold_distribution import FoldDistributionAnalyzer

# Set the master seed at the very beginning
set_seed(42)

# Now import all other dependencies
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Union, List, Tuple, Dict, Optional
import json
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from path_config import PathConfig, get_default_paths

# Import StratifiedGroupKFold and GroupKFold for spatial stratified cross-validation
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
import tensorflow as tf

# Apply additional determinism settings
tf.random.set_seed(42)
np.random.seed(42)

# =============================================================================
# FLAML AutoML Import
# =============================================================================
try:
    from flaml import AutoML
    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False
    print("WARNING: FLAML not installed. Install with: pip install flaml[automl]")
    print("Falling back to standard MLOptimizer if available.")

# Import the hyperparameter optimizer as fallback
from src.hyperparameter_optimization.hyper_tuner import MLOptimizer


# =============================================================================
# AutoML Training Function with Spatial CV
# =============================================================================
def train_with_automl(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray = None,
    pfts_train: np.ndarray = None,
    time_budget: int = 300,
    random_state: int = 42,
    n_inner_splits: int = 5,
    estimator_list: List[str] = None,
    use_spatial_cv: bool = True,
    metric: str = "rmse",
    ensemble: bool = True,
    early_stop: bool = True,
    verbose: int = 1,
) -> AutoML:
    """
    Train using FLAML AutoML with optional spatial cross-validation.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features (already scaled)
    y_train : np.ndarray
        Training targets (already transformed)
    groups_train : np.ndarray, optional
        Spatial group assignments for each sample (for inner CV)
    pfts_train : np.ndarray, optional
        PFT labels for stratification (for inner CV)
    time_budget : int
        Time limit in seconds for AutoML search
    random_state : int
        Random seed for reproducibility
    n_inner_splits : int
        Number of inner CV splits
    estimator_list : List[str], optional
        List of estimators to try. Default: ["xgboost", "lgbm", "rf", "extra_tree", "catboost"]
    use_spatial_cv : bool
        Whether to use spatial cross-validation for inner loop
    metric : str
        Optimization metric. Options: "rmse", "mae", "r2", "mse"
    ensemble : bool
        Whether to create an ensemble of top models
    early_stop : bool
        Whether to enable early stopping
    verbose : int
        Verbosity level (0, 1, or 2)
        
    Returns
    -------
    AutoML
        Fitted AutoML model
    """
    if not FLAML_AVAILABLE:
        raise ImportError("FLAML is not installed. Install with: pip install flaml[automl]")
    
    if estimator_list is None:
        estimator_list = ["xgboost", "lgbm", "rf", "extra_tree", "catboost"]
    
    automl = AutoML()
    
    # Prepare fit parameters
    fit_params = {
        "X_train": X_train,
        "y_train": y_train,
        "task": "regression",
        "metric": metric,
        "time_budget": time_budget,
        "estimator_list": estimator_list,
        "seed": random_state,
        "verbose": verbose,
        "early_stop": early_stop,
        "ensemble": ensemble,
        "eval_method": "cv",
        "n_splits": n_inner_splits,
    }
    
    # Use spatial CV if groups are provided
    if use_spatial_cv and groups_train is not None:
        # FLAML supports group-based CV via split_type="group" and groups parameter
        # This ensures samples from the same spatial group stay together
        fit_params["split_type"] = "group"
        fit_params["groups"] = groups_train
        
        logging.info(f"  Using GroupKFold for inner CV with {n_inner_splits} splits (spatial groups preserved)")
        
        # Log group distribution
        unique_groups = np.unique(groups_train)
        logging.info(f"  Number of unique groups in training data: {len(unique_groups)}")
        
        # Adjust n_splits if we have fewer groups than requested splits
        if len(unique_groups) < n_inner_splits:
            fit_params["n_splits"] = len(unique_groups)
            logging.warning(f"  Adjusted n_splits to {len(unique_groups)} (number of unique groups)")
    else:
        logging.info(f"  Using standard {n_inner_splits}-fold CV for inner loop")
    
    # Fit the AutoML model
    automl.fit(**fit_params)
    
    return automl


def train_with_automl_stratified(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    pfts_train: np.ndarray,
    time_budget: int = 300,
    random_state: int = 42,
    n_inner_splits: int = 5,
    estimator_list: List[str] = None,
    metric: str = "rmse",
    verbose: int = 1,
) -> Tuple[AutoML, dict]:
    """
    Train using FLAML AutoML with StratifiedGroupKFold via custom evaluation.
    
    This function provides full control over the CV strategy by using a custom
    metric function that internally uses StratifiedGroupKFold.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features (already scaled)
    y_train : np.ndarray
        Training targets (already transformed)
    groups_train : np.ndarray
        Spatial group assignments for each sample
    pfts_train : np.ndarray
        PFT labels for stratification
    time_budget : int
        Time limit in seconds
    random_state : int
        Random seed
    n_inner_splits : int
        Number of CV splits
    estimator_list : List[str], optional
        List of estimators to try
    metric : str
        Base metric for optimization
    verbose : int
        Verbosity level
        
    Returns
    -------
    Tuple[AutoML, dict]
        Fitted AutoML model and CV results
    """
    if not FLAML_AVAILABLE:
        raise ImportError("FLAML is not installed. Install with: pip install flaml[automl]")
    
    if estimator_list is None:
        estimator_list = ["xgboost", "lgbm", "rf", "extra_tree", "catboost"]
    
    # Adjust splits if needed
    unique_groups = np.unique(groups_train)
    actual_splits = min(n_inner_splits, len(unique_groups))
    
    if actual_splits < n_inner_splits:
        logging.warning(f"  Adjusted n_splits from {n_inner_splits} to {actual_splits} (number of unique groups)")
    
    # Create the StratifiedGroupKFold splitter
    cv_splitter = StratifiedGroupKFold(
        n_splits=actual_splits,
        shuffle=True,
        random_state=random_state
    )
    
    # Store CV results
    cv_results = {'fold_scores': [], 'mean_score': None, 'std_score': None}
    
    def custom_metric(
        X_val, y_val, estimator, labels=None, 
        X_train_inner=None, y_train_inner=None,
        weight_val=None, weight_train=None,
        *args
    ):
        """
        Custom metric function for FLAML that uses StratifiedGroupKFold.
        Returns loss (lower is better).
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        y_pred = estimator.predict(X_val)
        
        if metric == "rmse":
            loss = np.sqrt(mean_squared_error(y_val, y_pred))
        elif metric == "mse":
            loss = mean_squared_error(y_val, y_pred)
        elif metric == "mae":
            loss = mean_absolute_error(y_val, y_pred)
        elif metric == "r2":
            # For r2, we want to maximize, so return negative
            from sklearn.metrics import r2_score
            loss = -r2_score(y_val, y_pred)
        else:
            loss = np.sqrt(mean_squared_error(y_val, y_pred))
        
        return loss, {"pred": y_pred}
    
    # Pre-compute fold indices for StratifiedGroupKFold
    fold_indices = list(cv_splitter.split(X_train, pfts_train, groups_train))
    
    logging.info(f"  Using StratifiedGroupKFold with {actual_splits} splits")
    logging.info(f"  Stratifying by PFTs, grouping by spatial clusters")
    
    # Use FLAML with group split (closest to our needs that's natively supported)
    # The groups parameter ensures spatial coherence
    automl = AutoML()
    
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task="regression",
        metric=metric,
        time_budget=time_budget,
        estimator_list=estimator_list,
        seed=random_state,
        verbose=verbose,
        early_stop=True,
        ensemble=False,  # Disable ensemble for custom CV
        eval_method="cv",
        n_splits=actual_splits,
        split_type="group",
        groups=groups_train,
    )
    
    # After fitting, compute stratified CV scores for reporting
    logging.info("  Computing StratifiedGroupKFold scores for final model...")
    best_estimator = automl.model.estimator
    
    for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Clone and fit on this fold
        from sklearn.base import clone
        try:
            fold_model = clone(best_estimator)
            fold_model.fit(X_fold_train, y_fold_train)
            y_pred = fold_model.predict(X_fold_val)
            
            fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            cv_results['fold_scores'].append(fold_rmse)
        except Exception as e:
            logging.warning(f"  Could not compute fold {fold_idx + 1} score: {e}")
    
    if cv_results['fold_scores']:
        cv_results['mean_score'] = np.mean(cv_results['fold_scores'])
        cv_results['std_score'] = np.std(cv_results['fold_scores'])
        logging.info(f"  StratifiedGroupKFold RMSE: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
    
    return automl, cv_results


def get_automl_feature_importance(automl_model: AutoML, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from FLAML AutoML model.
    
    Parameters
    ----------
    automl_model : AutoML
        Fitted FLAML AutoML model
    feature_names : List[str]
        List of feature names
        
    Returns
    -------
    pd.DataFrame
        DataFrame with feature names and importance scores
    """
    try:
        best_model = automl_model.model.estimator
        
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
        elif hasattr(best_model, 'coef_'):
            importances = np.abs(best_model.coef_).flatten()
        else:
            logging.warning("Model does not have feature_importances_ or coef_ attribute")
            return None
        
        # Handle windowed features (may have more features than names)
        if len(importances) != len(feature_names):
            logging.warning(f"Feature count mismatch: {len(importances)} importances vs {len(feature_names)} names")
            # Create generic names for windowed features
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
        
    except Exception as e:
        logging.warning(f"Could not extract feature importance: {e}")
        return None


# =============================================================================
# Feature Engineering Functions
# =============================================================================
def add_sap_flow_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Add scientifically-grounded features for sap velocity prediction.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with meteorological and site data
    verbose : bool
        Print added features
        
    Returns:
    --------
    pd.DataFrame with new features added
    """
    
    df = df.copy()
    new_features = []
    
    # =========================================================================
    # PRIORITY 1: HIGHEST IMPACT FEATURES
    # =========================================================================
    
    # 1. VPD × Radiation interaction (THE most important)
    if 'vpd' in df.columns and 'sw_in' in df.columns:
        df['vpd_x_sw_in'] = df['vpd'] * df['sw_in']
        new_features.append('vpd_x_sw_in')
    
    # 2. VPD squared (non-linear stomatal response)
    if 'vpd' in df.columns:
        df['vpd_squared'] = df['vpd'] ** 2
        new_features.append('vpd_squared')
    
    # 3. VPD lag (sap flow responds with ~1h delay)
    if 'vpd' in df.columns:
        df['vpd_lag_1h'] = df['vpd'].shift(1)
        new_features.append('vpd_lag_1h')
    
    # 4. Daytime indicator
    if 'sw_in' in df.columns:
        df['is_daytime'] = (df['sw_in'] > 10).astype(np.float32)
        new_features.append('is_daytime')
    
    # 5. Southern hemisphere indicator (CRITICAL for NZL, ARG, AUS, ZAF)
    if 'latitude' in df.columns:
        df['southern_hemisphere'] = (df['latitude'] < 0).astype(np.float32)
        new_features.append('southern_hemisphere')
    
    # =========================================================================
    # PRIORITY 2: HIGH IMPACT FEATURES
    # =========================================================================
    
    # 6. VPD rolling mean (smoothed signal)
    if 'vpd' in df.columns:
        df['vpd_mean_6h'] = df['vpd'].rolling(6, min_periods=1).mean()
        new_features.append('vpd_mean_6h')
    
    # 7. VPD high threshold (stomatal closure indicator)
    if 'vpd' in df.columns:
        df['vpd_high'] = (df['vpd'] > 2.5).astype(np.float32)
        new_features.append('vpd_high')
    
    # 8. Clear sky index (cloudiness)
    if 'sw_in' in df.columns and 'ext_rad' in df.columns:
        df['clear_sky_index'] = (df['sw_in'] / (df['ext_rad'] + 1)).clip(0, 1)
        new_features.append('clear_sky_index')
    
    # 9. Height × VPD interaction (tall trees more sensitive)
    if 'canopy_height' in df.columns and 'vpd' in df.columns:
        df['height_x_vpd'] = df['canopy_height'] * df['vpd']
        new_features.append('height_x_vpd')
    
    # 10. Temperature lag
    if 'ta' in df.columns:
        df['ta_lag_1h'] = df['ta'].shift(1)
        new_features.append('ta_lag_1h')
    
    # 11. Growing degree days
    if 'ta' in df.columns:
        df['gdd'] = (df['ta'] - 5).clip(lower=0)
        new_features.append('gdd')
    
    # =========================================================================
    # PRIORITY 3: ADDITIONAL USEFUL FEATURES
    # =========================================================================
    
    # 12. VPD log transform
    if 'vpd' in df.columns:
        df['vpd_log'] = np.log1p(df['vpd'])
        new_features.append('vpd_log')
    
    # 13. Absorbed radiation (Beer-Lambert)
    if 'sw_in' in df.columns and 'LAI' in df.columns:
        k = 0.5  # Extinction coefficient
        df['absorbed_radiation'] = df['sw_in'] * (1 - np.exp(-k * df['LAI']))
        new_features.append('absorbed_radiation')
    
    # 14. Wind × VPD (boundary layer conductance)
    if 'ws' in df.columns and 'vpd' in df.columns:
        df['wind_x_vpd'] = df['ws'] * df['vpd']
        new_features.append('wind_x_vpd')
    
    # 15. Temperature × VPD
    if 'ta' in df.columns and 'vpd' in df.columns:
        df['ta_x_vpd'] = df['ta'] * df['vpd']
        new_features.append('ta_x_vpd')
    
    # 16. Water balance × VPD (can only transpire if water available)
    if 'vpd' in df.columns and 'prcip/PET' in df.columns:
        df['demand_x_supply'] = df['vpd'] * df['prcip/PET']
        new_features.append('demand_x_supply')
    
    # 17. Radiation daily cumsum (energy accumulation)
    if 'sw_in' in df.columns:
        df['sw_in_cumsum_day'] = df['sw_in'].rolling(24, min_periods=1).sum()
        new_features.append('sw_in_cumsum_day')
    
    # 18. VPD cumulative (stress accumulation)
    if 'vpd' in df.columns:
        df['vpd_cumsum_6h'] = df['vpd'].rolling(6, min_periods=1).sum()
        new_features.append('vpd_cumsum_6h')
    
    # 19. Temperature change rate
    if 'ta' in df.columns:
        df['ta_change_1h'] = df['ta'].diff(1)
        new_features.append('ta_change_1h')
    
    # 20. Climate zone indicators
    if 'latitude' in df.columns:
        df['tropical'] = (abs(df['latitude']) < 23.5).astype(np.float32)
        df['boreal'] = (abs(df['latitude']) > 55).astype(np.float32)
        new_features.extend(['tropical', 'boreal'])
    
    # =========================================================================
    # CONDITIONAL FEATURES (if data available)
    # =========================================================================
    
    # Soil moisture (VERY IMPORTANT if available)
    soil_cols = [c for c in df.columns if any(x in c.lower() for x in 
                 ['swc', 'soil_moisture', 'sm_', 'vwc'])]
    if soil_cols:
        sm_col = soil_cols[0]
        # Normalize to 0-1
        sm_min, sm_max = df[sm_col].quantile([0.05, 0.95])
        df['soil_moisture_rel'] = ((df[sm_col] - sm_min) / (sm_max - sm_min + 0.01)).clip(0, 1)
        df['soil_dry'] = (df['soil_moisture_rel'] < 0.3).astype(np.float32)
        new_features.extend(['soil_moisture_rel', 'soil_dry'])
    
    # Relative humidity
    if 'rh' in df.columns and 'ta' in df.columns:
        # Dew point depression
        a, b = 17.27, 237.7
        alpha = (a * df['ta'] / (b + df['ta'])) + np.log(df['rh']/100 + 0.01)
        df['dew_point'] = b * alpha / (a - alpha)
        df['dew_point_depression'] = df['ta'] - df['dew_point']
        new_features.extend(['dew_point', 'dew_point_depression'])
    
    # Precipitation lags
    precip_col = None
    for col in ['precip', 'precipitation', 'rain', 'prcp']:
        if col in df.columns:
            precip_col = col
            break
    
    if precip_col:
        df['precip_sum_24h'] = df[precip_col].rolling(24, min_periods=1).sum()
        df['precip_sum_72h'] = df[precip_col].rolling(72, min_periods=1).sum()
        # Hours since last rain
        rain_events = df[precip_col] > 1
        df['hours_since_rain'] = (~rain_events).groupby((~rain_events).cumsum()).cumcount()
        new_features.extend(['precip_sum_24h', 'precip_sum_72h', 'hours_since_rain'])
    
    # DBH/tree size
    if 'dbh' in df.columns:
        df['dbh_log'] = np.log1p(df['dbh'])
        df['sapwood_area_est'] = 0.5 * df['dbh'] ** 1.8
        new_features.extend(['dbh_log', 'sapwood_area_est'])
    
    if verbose:
        print(f"Added {len(new_features)} new features: {new_features}")
    
    return df


def setup_logging(logger_name, log_dir=None):
    """Set up basic logging configuration"""
    logger = logging.getLogger(logger_name)

    if log_dir is None:
        log_dir = Path('./outputs/logs')
    
    log_dir = Path(log_dir)
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f'{logger_name}_automl_optimizer.log'

    handlers = [
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    warnings.filterwarnings("ignore")
    
    return logging.getLogger()


@deterministic
def add_time_features(df, datetime_column='solar_TIMESTAMP'):
    """
    Create cyclical time features from a datetime column or index.
    """
    df = df.copy()
    
    if datetime_column is not None:
        if datetime_column not in df.columns:
            raise ValueError(f"Column '{datetime_column}' not found in DataFrame")
        
        if df[datetime_column].isna().any():
            n_nulls = df[datetime_column].isna().sum()
            logging.warning(f"Dropping {n_nulls} rows with NaN in {datetime_column}")
            df = df.dropna(subset=[datetime_column])
        
        date_time = pd.to_datetime(df[datetime_column])
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                date_time = pd.to_datetime(df.index)
            except:
                raise ValueError("DataFrame index cannot be converted to datetime")
        else:
            date_time = df.index

    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = 365.2425 * day
    week = 7 * day
    month = 30.44 * day
    
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    df['Week sin'] = np.sin(timestamp_s * (2 * np.pi / week))
    df['Week cos'] = np.cos(timestamp_s * (2 * np.pi / week))
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
    """
    all_windows = []
   
    for segment in segments:
        min_segment_length = input_width + shift
        if len(segment) < min_segment_length:
            continue
       
        window_gen = WindowGenerator(
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            data_df=segment,
            label_columns=label_columns,
            batch_size=1,
            shuffle=False,
            exclude_targets_from_features=exclude_targets_from_features,
            exclude_labels_from_inputs=exclude_labels_from_inputs
        )
       
        segment_ds = window_gen.dataset
        for inputs, labels in segment_ds:
            all_windows.append((inputs, labels))
   
    return all_windows


@deterministic
def get_predictions(model, test_windows, X_test, y_test, is_windowing=True):
    """Get predictions from the model using windows data."""
    if is_windowing:
        X_test, y_test = convert_windows_to_numpy(test_windows)

    y_pred = model.predict(X_test)
    
    return y_pred.flatten(), y_test.flatten()


@deterministic
def convert_windows_to_numpy(windows: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of window tuples to numpy arrays for features and labels.
    """
    if not windows:
        return np.array([]), np.array([])
    
    features = [w[0] for w in windows]
    labels = [w[1] for w in windows]
    
    X = np.array(features)
    y = np.array(labels)
    
    X_reshaped = X.reshape(X.shape[0], -1)
    y_flattened = y.flatten()

    return X_reshaped, y_flattened


@deterministic
def create_spatial_groups(
    lat: Union[np.ndarray, List],
    lon: Union[np.ndarray, List],
    data_counts: Union[np.ndarray, List] = None,
    n_groups: int = None,
    balanced: bool = True,
    method: str = 'grid',
    lat_grid_size: float = 0.05,
    lon_grid_size: float = 0.05,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Group geographical data points using various spatial grouping methods.
    Enhanced to consider data counts for balanced grouping.
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    
    if method == 'grid':
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
        
        lat_indices = np.digitize(lat, lat_bins) - 1
        lon_indices = np.digitize(lon, lon_bins) - 1
        
        n_lon_bins = len(lon_bins)
        grid_cell_ids = lat_indices * n_lon_bins + lon_indices
        
        if data_counts is None or n_groups is None or not balanced:
            unique_groups = np.unique(grid_cell_ids)
            group_map = {g: n for n, g in enumerate(unique_groups)}
            groups = np.array([group_map[g] for g in grid_cell_ids])
        else:
            unique_cells, cell_indices = np.unique(grid_cell_ids, return_inverse=True)
            cell_data_counts = np.bincount(cell_indices, weights=data_counts)

            sorted_cell_indices = np.argsort(cell_data_counts)[::-1]

            fold_data_counts = np.zeros(n_groups)
            cell_to_fold = np.zeros(len(unique_cells), dtype=int)
            for cell_idx in sorted_cell_indices:
                target_fold = np.argmin(fold_data_counts)
                cell_to_fold[cell_idx] = target_fold
                fold_data_counts[target_fold] += cell_data_counts[cell_idx]

            groups = cell_to_fold[cell_indices]
            
    elif method == 'default':
        if not len(lat) == len(lon):
            raise ValueError("Latitude and longitude arrays must have the same length.")
        if data_counts is None or n_groups is None or not balanced:
            groups = np.arange(len(lat))
        else:
            sorted_indices = np.argsort(data_counts)[::-1]
            groups = np.zeros(len(lat), dtype=int) - 1
            fold_data_counts = np.zeros(n_groups)
            
            for idx in sorted_indices:
                target_group = np.argmin(fold_data_counts)
                groups[idx] = target_group
                fold_data_counts[target_group] += data_counts[idx]
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods are 'grid' or 'default'.")

    # Calculate group statistics
    stats = []
    for group in np.unique(groups):
        if group == -1:
            continue
            
        mask = groups == group
        stats.append({
            'group': group,
            'size': np.sum(mask),
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
    
    if len(stats_df) > 0:
        logging.info("Spatial Group Statistics:")
        logging.info(f"Total groups: {len(stats_df)}")
        logging.info(f"Group details:\n{stats_df[['group', 'size', 'mean_lat', 'mean_lon']].to_string(index=False)}")
        
    return groups, stats_df


def parse_args():
    """
    Create and return the argument parser for this script.
    """
    parser = argparse.ArgumentParser(description="AutoML with spatial cross-validation")
    parser.add_argument('--RANDOM_SEED', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--run_id', type=str, default='automl_spatial_cv', help='Run identifier for logging')
    parser.add_argument('--n_groups', type=int, default=10, help='Number of spatial groups for cross-validation')
    parser.add_argument('--INPUT_WIDTH', type=int, default=2, help='Input width for time series windows')
    parser.add_argument('--LABEL_WIDTH', type=int, default=1, help='Label width for time series windows')
    parser.add_argument('--SHIFT', type=int, default=1, help='Shift for time series windows')
    parser.add_argument('--TARGET_COL', type=str, default='sap_velocity', help='Target column name')
    parser.add_argument('--EXCLUDE_LABELS', type=bool, default=True, help='Exclude labels from input features')
    parser.add_argument('--EXCLUDE_TARGETS', type=bool, default=True, help='Exclude targets from input features')
    parser.add_argument('--IS_WINDOWING', type=bool, default=True, help='Enable time windowing for data processing')
    parser.add_argument('--spatial_split_method', type=str, default='default', help='Method for spatial splitting')
    parser.add_argument('--IS_SHUFFLE', type=bool, default=True, help='Whether to enable shuffling of data')
    parser.add_argument('--IS_STRATIFIED', type=bool, default=True, help='Whether to use stratified sampling')
    parser.add_argument('--BALANCED', type=bool, default=False, help='Whether to balance spatial groups')
    parser.add_argument('--SPLIT_TYPE', type=str, default='spatial_stratified', help='Type of data splitting strategy')
    parser.add_argument('--IS_ONLY_DAY', type=bool, default=False, help='Whether to use only day data')
    parser.add_argument('--additional_features', nargs='*', default=[], help='List of additional features')
    parser.add_argument('--TIME_SCALE', type=str, default='hourly', help='Time scale: hourly or daily')
    
    # AutoML specific arguments
    parser.add_argument('--time_budget', type=int, default=10000, help='Time budget per fold in seconds')
    parser.add_argument('--estimators', nargs='*', 
                        default=['xgboost', 'lgbm', 'rf', 'extra_tree', 'catboost'],
                        help='List of estimators to try')
    parser.add_argument('--metric', type=str, default='rmse', help='Optimization metric')
    parser.add_argument('--ensemble', type=bool, default=True, help='Whether to create ensemble')
    parser.add_argument('--use_automl', type=bool, default=True, help='Use AutoML instead of MLOptimizer')
    parser.add_argument('--use_stratified_automl', type=bool, default=True, 
                        help='Use StratifiedGroupKFold with AutoML (computes stratified CV scores after fitting)')
    parser.add_argument('--hyperparameters', type=str, default=None, help='Path to hyperparameters JSON (for fallback)')
    
    return parser.parse_args()


@deterministic
def main(run_id="default"):
    """
    Main function implementing spatial CV with FLAML AutoML.
    Preserves StratifiedGroupKFold for both outer and inner CV loops.
    """
    args = parse_args()
    
    # Unpack arguments
    RANDOM_SEED = args.RANDOM_SEED
    INPUT_WIDTH = args.INPUT_WIDTH
    LABEL_WIDTH = args.LABEL_WIDTH
    SHIFT = args.SHIFT
    TARGET_COL = args.TARGET_COL
    EXCLUDE_LABELS = args.EXCLUDE_LABELS
    EXCLUDE_TARGETS = args.EXCLUDE_TARGETS
    IS_WINDOWING = args.IS_WINDOWING
    IS_SHUFFLE = args.IS_SHUFFLE
    IS_STRATIFIED = args.IS_STRATIFIED
    IS_ONLY_DAY = args.IS_ONLY_DAY
    n_groups = args.n_groups
    spatial_split_method = args.spatial_split_method
    SPLIT_TYPE = args.SPLIT_TYPE
    BALANCED = args.BALANCED
    additional_features = args.additional_features
    TIME_SCALE = args.TIME_SCALE
    
    # AutoML specific
    TIME_BUDGET = args.time_budget
    ESTIMATORS = args.estimators
    METRIC = args.metric
    ENSEMBLE = args.ensemble
    USE_AUTOML = args.use_automl
    USE_STRATIFIED_AUTOML = args.use_stratified_automl
    
    run_id = args.run_id
    scale = 'sapwood'
    paths = PathConfig(scale=scale)
    
    setup_logging(logger_name='automl', log_dir=paths.optimization_logs_dir)
    
    logging.info("=" * 70)
    logging.info("AUTOML WITH SPATIAL CROSS-VALIDATION")
    logging.info("=" * 70)
    logging.info(f"Starting AutoML training with {SPLIT_TYPE} CV, seed {RANDOM_SEED}")
    logging.info(f"AutoML settings: time_budget={TIME_BUDGET}s, estimators={ESTIMATORS}, metric={METRIC}")
    logging.info(f"CV settings: n_groups={n_groups}, is_stratified={IS_STRATIFIED}")
    logging.info(f"Data settings: input_width={INPUT_WIDTH}, label_width={LABEL_WIDTH}, shift={SHIFT}")
    logging.info(f"is_windowing={IS_WINDOWING}, spatial_split_method={spatial_split_method}")
    logging.info(f"is_shuffle={IS_SHUFFLE}, is_only_day={IS_ONLY_DAY}, time_scale={TIME_SCALE}")

    # --- File Paths ---
    plot_dir = paths.hyper_tuning_plots_dir / 'automl' / run_id
    plot_dir.mkdir(parents=True, exist_ok=True)
    model_dir = paths.models_root / 'automl' / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir = paths.merged_data_root / TIME_SCALE

    # --- Data Loading and Processing ---
    data_list = sorted(list(data_dir.glob(f'*{TIME_SCALE}.csv')))
    data_list = [f for f in data_list if 'all_biomes_merged' not in f.name]
    
    if not data_list:
        logging.critical(f"ERROR: No CSV files found in {data_dir}. Exiting.")
        return [], []

    site_data_dict = {}
    site_info_dict = {}
    
    base_features = [
        TARGET_COL, 
        'sw_in', 'ppfd_in', 'ext_rad', 'ws', 'ta', 'vpd', 
        'pft', 'canopy_height', 
        'elevation',   
        'LAI', 'prcip/PET', 
        'volumetric_soil_water_layer_1', 'soil_temperature_level_1',
    ]
    
    used_cols = list(set(base_features + additional_features))
    logging.info(f"Using columns: {used_cols}")
    
    all_possible_pft_types = ['MF', 'DNF', 'ENF', 'EBF', 'WSA', 'WET', 'DBF', 'SAV']
    max_sap = 0
    max_site = None
    final_feature_names = None
    
    for data_file in data_list:
        try:
            df = pd.read_csv(data_file, parse_dates=['TIMESTAMP'])
            logging.info(f"Loaded data for site {data_file.name}, shape: {df.shape}")
       
            if IS_ONLY_DAY:
                if 'sw_in' not in df.columns:
                    logging.warning(f"Warning: 'sw_in' column not found in {data_file.name}. Skipping.")
                    continue
                df = df[df['sw_in'] > 10]
                if df.empty:
                    logging.warning(f"Warning: No data after filtering for day time in {data_file.name}. Skipping.")
                    continue 

            site_id = df['site_name'].iloc[0]
            
            if df[TARGET_COL].max() > max_sap:
                max_sap = df[TARGET_COL].max()
                max_site = data_file.name
                
            site_max = df[TARGET_COL].max()
            if site_max > 100:
                logging.info(f"!!! HIGH VELOCITY ALERT: Site {site_id} has max value: {site_max:.2f}")
                
            lat_col = next((col for col in df.columns if col.lower() in ['lat', 'latitude_x']), None)
            lon_col = next((col for col in df.columns if col.lower() in ['lon', 'longitude_x']), None)
            pft_col = next((col for col in df.columns if col.lower() in ['pft', 'plant_functional_type', 'biome']), None)
            
            if not (lat_col and lon_col and pft_col):
                logging.warning(f"Warning: Lat/Lon/PFT not found for site {site_id}. Skipping.")
                continue

            latitude = df[lat_col].median()
            longitude = df[lon_col].median()
        
            if site_id.startswith('CZE'):
                logging.info(f"Skipping site {site_id} as per rules.")
                continue
                
            df['latitude'] = latitude
            df['longitude'] = longitude
            pft_value = df[pft_col].mode()[0] 
            
            df.set_index('TIMESTAMP', inplace=True)
            df = add_time_features(df)
            
            time_features = ['Day sin', 'Year sin']
            feature_cols = used_cols + time_features
            logging.info(f"Successfully added time features for site {site_id}")

            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                logging.warning(f"Warning: Missing columns: {missing_cols} in {data_file.name}: skipping.")
                continue

            df = df[feature_cols].copy()
            
            if 'pft' in used_cols:
                if 'pft' in df.columns:
                    orig_cols = [col for col in feature_cols if col != 'pft']
                    pft_cat = pd.Categorical(df['pft'], categories=all_possible_pft_types)
                    pft_df = pd.get_dummies(pft_cat)
                    pft_df.index = df.index
                    df = df[orig_cols].join(pft_df)
                else:
                    logging.warning(f"Warning: Missing pft column in {data_file.name}")
                    continue
                    
            final_feature_names = [col for col in df.columns.tolist() if col != TARGET_COL]
            logging.info(f"Total number of features before windowing: {len(final_feature_names)}")
            
            df = df.astype(np.float32)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            site_info = {
                'latitude': latitude, 
                'longitude': longitude, 
                'pft': pft_value
            }

            if IS_WINDOWING:
                min_segment_length = INPUT_WIDTH + SHIFT
                if len(df) < min_segment_length:
                    logging.warning(f"Warning: Not enough data for site {site_id}. Skipping.")
                    continue
                    
                segments = TimeSeriesSegmenter.segment_time_series(
                    df, gap_threshold=2, 
                    unit='hours' if TIME_SCALE == 'hourly' else 'days', 
                    min_segment_length=min_segment_length
                )
                data_count = sum(len(s) for s in segments)
                
                if segments:
                    site_data_dict[site_id] = segments
                    site_info['data_count'] = data_count
                    site_info_dict[site_id] = site_info
                    logging.info(f"  {site_id}: Processed {len(segments)} segments with {data_count} records")
                else:
                    logging.warning(f"  {site_id}: No valid segments. Skipping.")
            else:
                site_data_dict[site_id] = df
                site_info['data_count'] = len(df)
                site_info_dict[site_id] = site_info

        except Exception as e:
            logging.error(f"Error processing {data_file.name}: {e}")

    logging.info(f"\nSite with max {TARGET_COL}: {max_site} with value {max_sap}")
    
    if not site_data_dict:
        logging.critical("ERROR: No valid site data could be processed.")
        return [], []
    
    # --- Prepare Data for CV Split ---
    site_ids = np.array(list(site_info_dict.keys()))
    latitudes = [site_info_dict[site]['latitude'] for site in site_ids]
    longitudes = [site_info_dict[site]['longitude'] for site in site_ids]
    data_counts = [site_info_dict[site]['data_count'] for site in site_ids]
    
    logging.info("\nCreating spatial groups...")
    if IS_STRATIFIED:
        spatial_groups, group_stats = create_spatial_groups(
            lat=latitudes,
            lon=longitudes,
            method=spatial_split_method,
        )
    else:
        spatial_groups, group_stats = create_spatial_groups(
            lat=latitudes,
            lon=longitudes,
            data_counts=data_counts,
            n_groups=n_groups,
            method=spatial_split_method,
            balanced=BALANCED,
        )
    logging.info(f"Spatial groups assigned: {np.unique(spatial_groups)}")
   
    site_to_group = {site_id: group for site_id, group in zip(site_ids, spatial_groups)}
    site_to_pft = {site_id: site_info_dict[site_id]['pft'] for site_id in site_ids}
    
    # Plot spatial grouping
    fig = plt.figure(figsize=(20, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)

    colors = plt.cm.tab20.colors[:n_groups]
    color_map = ListedColormap(colors)

    scatter = ax.scatter(longitudes, latitudes, c=spatial_groups, s=100, 
                        edgecolor='k', cmap=color_map, transform=ccrs.PlateCarree())

    cbar = plt.colorbar(scatter, ax=ax, label='Spatial Group', shrink=0.8)

    ax.set_extent([min(longitudes)-5, max(longitudes)+5, 
                min(latitudes)-5, max(latitudes)+5], crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Spatial Grouping of Sites')

    plt.savefig(plot_dir / f'spatial_groups_{run_id}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Process all data into unified record-level numpy arrays
    logging.info("Preparing final RECORD-LEVEL data structures for all sites...")
    
    list_X, list_y, list_groups, list_pfts = [], [], [], []
    list_site_ids_str = []

    for site_id, raw_data in site_data_dict.items():
        if IS_WINDOWING:
            windows = create_windows_from_segments(
                segments=raw_data, input_width=INPUT_WIDTH, label_width=LABEL_WIDTH, shift=SHIFT,
                label_columns=[TARGET_COL], exclude_targets_from_features=EXCLUDE_TARGETS,
                exclude_labels_from_inputs=EXCLUDE_LABELS
            )
            if not windows:
                continue
            X_site, y_site = convert_windows_to_numpy(windows)
        else:
            if raw_data.empty:
                continue
            y_site = raw_data.pop(TARGET_COL).values
            X_site = raw_data.values
            
        num_records = len(y_site)
        if num_records == 0:
            continue
            
        list_X.append(X_site)
        list_y.append(y_site)
        list_groups.append(np.full(num_records, site_to_group[site_id]))
        list_pfts.append(np.full(num_records, site_to_pft[site_id]))
        list_site_ids_str.append(np.full(num_records, site_id))
        
    del site_data_dict

    if not list_X:
        logging.critical("ERROR: No data records were generated after processing. Exiting.")
        return [], []
        
    X_all_records = np.vstack(list_X)
    y_all_records = np.concatenate(list_y)
    groups_all_records = np.concatenate(list_groups)
    pfts_all_records = np.concatenate(list_pfts)
    site_ids_all_records = np.concatenate(list_site_ids_str)
    
    logging.info(f"Total records processed and ready for CV: {len(y_all_records)}")
    
    pfts_encoded, pft_categories = pd.factorize(pfts_all_records)
    logging.info(f"Encoded PFTs into {len(pft_categories)} integer classes: {list(pft_categories)}")
    
    # --- Stratified Group K-Fold Cross-Validation ---
    logging.info("\n" + "=" * 70)
    logging.info("INITIALIZING K-FOLD CROSS-VALIDATION AT RECORD LEVEL")
    logging.info("=" * 70)
    
    if IS_STRATIFIED:
        outer_cv = StratifiedGroupKFold(n_splits=n_groups, shuffle=True, random_state=RANDOM_SEED)
        y_all_stratified = pfts_encoded
        logging.info(f"Using StratifiedGroupKFold with {n_groups} splits, stratifying by PFTs.")
    else:
        outer_cv = GroupKFold(n_splits=n_groups)
        y_all_stratified = y_all_records
        logging.info(f"Using GroupKFold with {n_groups} splits, grouping by spatial groups.")
    
    # Diagnostic: Target Distribution by Fold
    logging.info("\n=== DIAGNOSTIC: Target Distribution by Fold ===")
    temp_split = outer_cv.split(X_all_records, y_all_stratified, groups_all_records)
    for fold_idx, (train_idx, test_idx) in enumerate(temp_split):
        y_train_temp = y_all_records[train_idx]
        y_test_temp = y_all_records[test_idx]
        logging.info(f"Fold {fold_idx+1}: "
                     f"Train max={y_train_temp.max():.1f}, Test max={y_test_temp.max():.1f}, "
                     f"Train 95th={np.percentile(y_train_temp, 95):.1f}, Test 95th={np.percentile(y_test_temp, 95):.1f}")
    
    # Reset the generator
    split_generator = outer_cv.split(X_all_records, y_all_stratified, groups_all_records)
    
    all_test_r2_scores, all_test_rmse_scores = [], []
    all_predictions, all_actuals = [], []
    all_test_pfts = []
    all_best_estimators = []
    
    dist_analyzer = FoldDistributionAnalyzer(output_dir=str(plot_dir / 'distributions'))

    # ==========================================================================
    # MAIN TRAINING LOOP WITH AUTOML
    # ==========================================================================
    for fold_idx, (train_val_indices, test_indices) in enumerate(split_generator):
        logging.info(f"\n{'=' * 70}")
        logging.info(f"FOLD {fold_idx + 1}/{n_groups}")
        logging.info(f"{'=' * 70}")
        
        # Identify test sites for this fold
        test_groups = np.unique(groups_all_records[test_indices])
        test_site_ids = [site_ids[np.where(spatial_groups == g)[0][0]] for g in test_groups]
        
        logging.info(f"TEST sites: {test_site_ids}")
        for sid in test_site_ids:
            site_mask = groups_all_records[test_indices] == site_to_group[sid]
            site_y = y_all_records[test_indices][site_mask]
            logging.info(f"  {sid}: n={len(site_y)}, median={np.median(site_y):.1f}, "
                        f"max={site_y.max():.1f}, PFT={site_to_pft[sid]}")
        
        # --- Data Collection via Direct Slicing ---
        X_train_val = X_all_records[train_val_indices]
        y_train_val = y_all_records[train_val_indices]
        pfts_train_val = pfts_all_records[train_val_indices]
        groups_train_val = groups_all_records[train_val_indices]
        pfts_encoded_train_val = pfts_encoded[train_val_indices]
        
        X_test = X_all_records[test_indices]
        y_test = y_all_records[test_indices]
        
        # Transform target (log1p)
        y_train_val_transformed = np.log1p(y_train_val)
        
        # Scale features
        scaler = StandardScaler()
        X_train_val_scaled = scaler.fit_transform(X_train_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Capture fold data for distribution analysis
        dist_analyzer.capture_fold_data(
            fold_idx=fold_idx,
            X_train=X_train_val_scaled,
            y_train=y_train_val_transformed,
            X_test=X_test_scaled,
            y_test=y_test
        )
        
        # Shuffle if requested
        if IS_SHUFFLE:
            shuffled_indices = np.random.permutation(len(X_train_val_scaled))
            X_train_val_scaled = X_train_val_scaled[shuffled_indices]
            y_train_val_transformed = y_train_val_transformed[shuffled_indices]
            pfts_train_val = pfts_train_val[shuffled_indices]
            groups_train_val = groups_train_val[shuffled_indices]
            pfts_encoded_train_val = pfts_encoded_train_val[shuffled_indices]

        logging.info(f"Train/Val records: {len(y_train_val_transformed)}, Test records: {len(y_test)}")
        
        # ======================================================================
        # AUTOML TRAINING WITH SPATIAL CV
        # ======================================================================
        if USE_AUTOML and FLAML_AVAILABLE:
            logging.info(f"\n--- Training with FLAML AutoML ---")
            logging.info(f"Time budget: {TIME_BUDGET}s, Estimators: {ESTIMATORS}")
            
            if USE_STRATIFIED_AUTOML:
                # Use the stratified version with StratifiedGroupKFold scoring
                logging.info("Using StratifiedGroupKFold for CV scoring")
                automl_model, cv_results = train_with_automl_stratified(
                    X_train=X_train_val_scaled,
                    y_train=y_train_val_transformed,
                    groups_train=groups_train_val,
                    pfts_train=pfts_encoded_train_val,
                    time_budget=TIME_BUDGET,
                    random_state=RANDOM_SEED,
                    n_inner_splits=5,
                    estimator_list=ESTIMATORS,
                    metric=METRIC,
                    verbose=1,
                )
            else:
                # Use standard GroupKFold (faster, still preserves spatial groups)
                automl_model = train_with_automl(
                    X_train=X_train_val_scaled,
                    y_train=y_train_val_transformed,
                    groups_train=groups_train_val,
                    pfts_train=pfts_encoded_train_val,  # Kept for logging/future use
                    time_budget=TIME_BUDGET,
                    random_state=RANDOM_SEED,
                    n_inner_splits=5,
                    estimator_list=ESTIMATORS,
                    use_spatial_cv=True,  # Uses GroupKFold (spatial groups preserved)
                    metric=METRIC,
                    ensemble=ENSEMBLE,
                    early_stop=True,
                    verbose=1,
                )
            
            logging.info(f"Best estimator: {automl_model.best_estimator}")
            logging.info(f"Best config: {automl_model.best_config}")
            all_best_estimators.append(automl_model.best_estimator)
            
            # Predict
            test_predictions_transformed = automl_model.predict(X_test_scaled)
            
            # Save model
            model_path = model_dir / f'automl_fold_{fold_idx + 1}_{run_id}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'automl_model': automl_model,
                    'scaler': scaler,
                    'best_estimator': automl_model.best_estimator,
                    'best_config': automl_model.best_config,
                }, f)
            logging.info(f"Model saved to: {model_path}")
            
            # Feature importance (if available)
            importance_df = get_automl_feature_importance(automl_model, final_feature_names)
            if importance_df is not None:
                importance_path = plot_dir / f'feature_importance_fold_{fold_idx + 1}_{run_id}.csv'
                importance_df.to_csv(importance_path, index=False)
                logging.info(f"Feature importance saved to: {importance_path}")
                logging.info(f"Top 10 features:\n{importance_df.head(10).to_string(index=False)}")
                
        else:
            # Fallback to MLOptimizer
            logging.info(f"\n--- Training with MLOptimizer (fallback) ---")
            
            if args.hyperparameters:
                with open(args.hyperparameters, 'r') as f:
                    hyperparameters = json.load(f)
            else:
                # Default XGBoost hyperparameters
                hyperparameters = {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [3, 5, 7, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9],
                }
            
            optimizer = MLOptimizer(
                param_grid=hyperparameters,
                scoring='neg_mean_squared_error',
                model_type='xgb',
                task='regression',
                random_state=RANDOM_SEED,
                n_splits=5
            )
            
            optimizer.fit(
                X=X_train_val_scaled,
                y=y_train_val_transformed,
                is_cv=True,
                y_stratify=pfts_encoded_train_val,
                groups=groups_train_val,
                is_refit=True,
                split_type=SPLIT_TYPE,
            )
            
            best_model = optimizer.get_best_model()
            test_predictions_transformed = best_model.predict(X_test_scaled)
            
            # Save model
            best_model.save_model(str(model_dir / f'xgb_fold_{fold_idx + 1}_{run_id}.json'))
            all_best_estimators.append('xgboost')

        # Back-transform predictions
        test_predictions = np.expm1(test_predictions_transformed)
        
        # --- EVALUATION ---
        y_test_arr = np.asarray(y_test)
        test_predictions = np.asarray(test_predictions)
        pfts_test = pfts_all_records[test_indices]
        
        finite_mask = np.isfinite(y_test_arr) & np.isfinite(test_predictions)
        n_total = y_test_arr.shape[0]
        n_valid = finite_mask.sum()

        if n_valid == 0:
            logging.error(f"Fold {fold_idx + 1}: No finite prediction/actual pairs. Skipping.")
            continue

        if n_valid < n_total:
            logging.warning(f"Fold {fold_idx + 1}: Dropping {n_total - n_valid} rows with NaN/inf.")

        y_test_clean = y_test_arr[finite_mask]
        preds_clean = test_predictions[finite_mask]
        pfts_test_clean = pfts_test[finite_mask]
        
        all_test_pfts.extend(pfts_test_clean)
        
        fold_test_r2 = r2_score(y_test_clean, preds_clean)
        fold_test_rmse = np.sqrt(mean_squared_error(y_test_clean, preds_clean))
        fold_test_mae = mean_absolute_error(y_test_clean, preds_clean)
        
        all_test_r2_scores.append(fold_test_r2)
        all_test_rmse_scores.append(fold_test_rmse)
        all_predictions.extend(test_predictions)
        all_actuals.extend(y_test)

        logging.info(f"\nFold {fold_idx + 1} Results:")
        logging.info(f"  R²:   {fold_test_r2:.4f}")
        logging.info(f"  RMSE: {fold_test_rmse:.4f}")
        logging.info(f"  MAE:  {fold_test_mae:.4f}")

        # --- Generate Individual Fold Plots ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Predictions vs Actuals
        axes[0].scatter(y_test_clean, preds_clean, alpha=0.5, s=20, color='steelblue', 
                       edgecolors='black', linewidth=0.5)
        axes[0].set_xlabel('Observed Sap Velocity (cm³ cm⁻² h⁻¹)', fontsize=13)
        axes[0].set_ylabel('Predicted Sap Velocity (cm³ cm⁻² h⁻¹)', fontsize=13)
        axes[0].set_title(f'Fold {fold_idx + 1}: Predictions vs Actuals\n'
                         f'$R^2 = {fold_test_r2:.3f}$, RMSE = ${fold_test_rmse:.3f}$', 
                         fontsize=13, fontweight='bold')

        min_val = min(y_test_clean.min(), preds_clean.min())
        max_val = max(y_test_clean.max(), preds_clean.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r-', alpha=0.8, linewidth=2, 
                    label='Perfect Prediction')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].axis('equal')
        axes[0].axis('square')

        # Right: Residual Analysis
        residuals = y_test_clean - preds_clean
        axes[1].scatter(preds_clean, residuals, alpha=0.5, s=20, color='coral', 
                       edgecolors='black', linewidth=0.5)
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
        axes[1].set_xlabel('Predicted Sap Velocity', fontsize=13)
        axes[1].set_ylabel('Residuals (Observed - Predicted)', fontsize=13)
        axes[1].set_title(f'Fold {fold_idx + 1}: Residual Plot\nMAE = ${fold_test_mae:.3f}$', 
                         fontsize=13, fontweight='bold')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fold_plot_path = plot_dir / f'fold_{fold_idx + 1}_performance_{run_id}.png'
        plt.savefig(fold_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Fold {fold_idx + 1} plot saved to: {fold_plot_path}")

        # Distribution comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(y_test_clean, bins=50, alpha=0.6, label='Observed', color='blue', edgecolor='black')
        axes[0].hist(preds_clean, bins=50, alpha=0.6, label='Predicted', color='orange', edgecolor='black')
        axes[0].set_xlabel('Sap Velocity', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'Fold {fold_idx + 1}: Distribution Comparison', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title(f'Fold {fold_idx + 1}: Q-Q Plot of Residuals', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        dist_plot_path = plot_dir / f'fold_{fold_idx + 1}_distributions_{run_id}.png'
        plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Save fold predictions to CSV
        fold_results_df = pd.DataFrame({
            'Observed': y_test_clean,
            'Predicted': preds_clean,
            'Residual': residuals,
            'Absolute_Error': np.abs(residuals)
        })
        fold_csv_path = plot_dir / f'fold_{fold_idx + 1}_predictions_{run_id}.csv'
        fold_results_df.to_csv(fold_csv_path, index=False)

    # Generate distribution analysis plots
    dist_analyzer.generate_all_plots()

    # ==========================================================================
    # FINAL RESULTS SUMMARY
    # ==========================================================================
    if all_test_r2_scores:
        mean_r2 = np.mean(all_test_r2_scores)
        std_r2 = np.std(all_test_r2_scores)
        mean_rmse = np.mean(all_test_rmse_scores)
        std_rmse = np.std(all_test_rmse_scores)
        
        logging.info("\n" + "=" * 70)
        logging.info("OVERALL SPATIAL CROSS-VALIDATION RESULTS")
        logging.info("=" * 70)
        logging.info(f"Test R² Score:  {mean_r2:.4f} ± {std_r2:.4f}")
        logging.info(f"Test RMSE:      {mean_rmse:.4f} ± {std_rmse:.4f}")
        
        if USE_AUTOML and FLAML_AVAILABLE:
            from collections import Counter
            estimator_counts = Counter(all_best_estimators)
            logging.info(f"\nBest estimators across folds: {dict(estimator_counts)}")

        # Overall predictions vs actuals plot
        plt.figure(figsize=(8, 8))
        plt.scatter(all_actuals, all_predictions, alpha=0.5, s=20)
        plt.xlabel('Observed Sap Velocity (cm³ cm⁻² h⁻¹)', fontsize=14)
        plt.ylabel('Predicted Sap Velocity', fontsize=14)
        plt.title(f'Spatial CV Results (AutoML)\n'
                 f'$R^2 = {mean_r2:.3f} \\pm {std_r2:.3f}$, RMSE = ${mean_rmse:.3f} \\pm {std_rmse:.3f}$', 
                 fontsize=12)
        
        min_val = min(min(all_actuals), min(all_predictions))
        max_val = max(max(all_actuals), max(all_predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r-', alpha=0.8, linewidth=2)
        
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.axis('square')
        plt.tight_layout()
        plt.savefig(plot_dir / f'spatial_cv_predictions_vs_actual_{run_id}.png', dpi=300)
        plt.close()
        
        # PFT-stratified scatter plots
        logging.info("Generating PFT-stratified performance plots...")
        
        arr_actuals = np.array(all_actuals)
        arr_predictions = np.array(all_predictions)
        arr_pfts = np.array(all_test_pfts)
        
        unique_pfts = np.unique(arr_pfts)
        n_pfts = len(unique_pfts)
        
        if n_pfts > 0:
            n_cols = 3
            n_rows = (n_pfts + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
            axes = axes.flatten()
            
            g_min = min(arr_actuals.min(), arr_predictions.min())
            g_max = max(arr_actuals.max(), arr_predictions.max())
            
            for i, pft in enumerate(unique_pfts):
                ax = axes[i]
                
                mask = arr_pfts == pft
                pft_actuals = arr_actuals[mask]
                pft_preds = arr_predictions[mask]
                
                if len(pft_actuals) < 10:
                    ax.text(0.5, 0.5, f"{pft}\nInsufficient Data (n={len(pft_actuals)})", 
                            ha='center', va='center')
                    continue
                    
                pft_r2 = r2_score(pft_actuals, pft_preds)
                pft_rmse = np.sqrt(mean_squared_error(pft_actuals, pft_preds))
                
                ax.scatter(pft_actuals, pft_preds, alpha=0.4, s=15, color='teal', 
                          edgecolor='k', linewidth=0.3)
                ax.plot([g_min, g_max], [g_min, g_max], 'r--', alpha=0.7)
                
                ax.set_title(f"PFT: {pft} (n={len(pft_actuals)})\nR²={pft_r2:.3f}, RMSE={pft_rmse:.3f}", 
                            fontweight='bold')
                ax.set_xlabel('Observed')
                ax.set_ylabel('Predicted')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(g_min, g_max)
                ax.set_ylim(g_min, g_max)
                ax.set_aspect('equal')

            for j in range(i + 1, len(axes)):
                axes[j].axis('off')
                
            plt.tight_layout()
            pft_plot_path = plot_dir / f'pft_performance_{run_id}.png'
            plt.savefig(pft_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save PFT metrics to CSV
            pft_metrics = []
            for pft in unique_pfts:
                mask = arr_pfts == pft
                if np.sum(mask) > 0:
                    y_p = arr_actuals[mask]
                    y_hat_p = arr_predictions[mask]
                    pft_metrics.append({
                        'PFT': pft,
                        'N_Samples': len(y_p),
                        'R2': r2_score(y_p, y_hat_p),
                        'RMSE': np.sqrt(mean_squared_error(y_p, y_hat_p)),
                        'MAE': mean_absolute_error(y_p, y_hat_p)
                    })
            pd.DataFrame(pft_metrics).to_csv(plot_dir / f'pft_metrics_{run_id}.csv', index=False)

    # ==========================================================================
    # TRAIN FINAL MODEL ON ALL DATA
    # ==========================================================================
    logging.info("\n" + "=" * 70)
    logging.info("TRAINING FINAL MODEL ON ALL DATA")
    logging.info("=" * 70)
    
    scaler_final = StandardScaler()
    X_all_scaled = scaler_final.fit_transform(X_all_records)
    y_all_transformed = np.log1p(y_all_records)
    
    logging.info(f"Data shapes: X={X_all_scaled.shape}, y={y_all_transformed.shape}")
    
    if USE_AUTOML and FLAML_AVAILABLE:
        logging.info("Running AutoML hyperparameter optimization on all data...")
        
        final_automl = train_with_automl(
            X_train=X_all_scaled,
            y_train=y_all_transformed,
            groups_train=groups_all_records,
            pfts_train=pfts_encoded,  # Kept for logging/future use
            time_budget=TIME_BUDGET * 2,  # Give more time for final model
            random_state=RANDOM_SEED,
            n_inner_splits=5,
            estimator_list=ESTIMATORS,
            use_spatial_cv=True,  # Uses GroupKFold (spatial groups preserved)
            metric=METRIC,
            ensemble=ENSEMBLE,
            early_stop=True,
            verbose=1,
        )
        
        best_estimator = final_automl.best_estimator
        best_config = final_automl.best_config
        
        logging.info(f"Final best estimator: {best_estimator}")
        logging.info(f"Final best config: {best_config}")
        
        # Save final model
        final_model_path = model_dir / f'FINAL_automl_{run_id}.pkl'
        with open(final_model_path, 'wb') as f:
            pickle.dump({
                'automl_model': final_automl,
                'scaler': scaler_final,
                'best_estimator': best_estimator,
                'best_config': best_config,
            }, f)
        logging.info(f"Final model saved to: {final_model_path}")
        
        # Sanity check
        y_pred_all_transformed = final_automl.predict(X_all_scaled)
        
    else:
        # Fallback to MLOptimizer
        if args.hyperparameters:
            with open(args.hyperparameters, 'r') as f:
                hyperparameters = json.load(f)
        else:
            hyperparameters = {
                'n_estimators': [100, 200, 500],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1],
            }
        
        final_optimizer = MLOptimizer(
            param_grid=hyperparameters,
            scoring='neg_mean_squared_error',
            model_type='xgb',
            task='regression',
            random_state=RANDOM_SEED,
            n_splits=5
        )
        
        final_optimizer.fit(
            X=X_all_scaled,
            y=y_all_transformed,
            is_cv=True,
            y_stratify=pfts_encoded,
            groups=groups_all_records,
            is_refit=True,
            split_type=SPLIT_TYPE,
        )
        
        final_model = final_optimizer.get_best_model()
        best_config = final_optimizer.best_params_
        best_estimator = 'xgboost'
        
        final_model_path = model_dir / f'FINAL_xgb_{run_id}.json'
        final_model.save_model(str(final_model_path))
        
        y_pred_all_transformed = final_model.predict(X_all_scaled)
    
    # Save scaler
    final_scaler_path = model_dir / f'FINAL_scaler_{run_id}.pkl'
    joblib.dump(scaler_final, final_scaler_path)
    logging.info(f"Scaler saved to: {final_scaler_path}")
    
    # Back-transform predictions
    y_pred_all = np.expm1(y_pred_all_transformed)
    
    # Sanity check metrics
    sanity_r2 = r2_score(y_all_records, y_pred_all)
    sanity_rmse = np.sqrt(mean_squared_error(y_all_records, y_pred_all))
    sanity_mae = mean_absolute_error(y_all_records, y_pred_all)
    
    logging.info(f"\nSanity check (on training data):")
    logging.info(f"  R²   = {sanity_r2:.4f}")
    logging.info(f"  RMSE = {sanity_rmse:.4f}")
    logging.info(f"  MAE  = {sanity_mae:.4f}")
    
    # Save config
    final_config = {
        'model_type': 'automl' if USE_AUTOML else 'xgboost',
        'run_id': run_id,
        'best_estimator': best_estimator,
        'best_config': best_config,
        'cv_results': {
            'mean_r2': float(mean_r2),
            'std_r2': float(std_r2),
            'mean_rmse': float(mean_rmse),
            'std_rmse': float(std_rmse),
            'n_folds': n_groups,
        },
        'preprocessing': {
            'target_transform': 'log1p',
            'feature_scaling': 'StandardScaler',
        },
        'data_info': {
            'n_samples': len(y_all_records),
            'n_features': X_all_records.shape[1],
            'input_width': INPUT_WIDTH,
            'label_width': LABEL_WIDTH,
            'shift': SHIFT,
        },
        'feature_names': final_feature_names,
        'random_seed': RANDOM_SEED,
        'split_type': SPLIT_TYPE,
        'automl_settings': {
            'time_budget': TIME_BUDGET,
            'estimators': ESTIMATORS,
            'metric': METRIC,
            'ensemble': ENSEMBLE,
        } if USE_AUTOML else None,
    }
    
    config_path = model_dir / f'FINAL_config_{run_id}.json'
    with open(config_path, 'w') as f:
        json.dump(final_config, f, indent=2, default=str)
    logging.info(f"Config saved to: {config_path}")
    
    # Sanity check plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].scatter(y_all_records, y_pred_all, alpha=0.3, s=10, color='steelblue')
    axes[0].set_xlabel('Observed Sap Velocity (cm³ cm⁻² h⁻¹)', fontsize=13)
    axes[0].set_ylabel('Predicted Sap Velocity (cm³ cm⁻² h⁻¹)', fontsize=13)
    axes[0].set_title(f'Final Model: Training Data Fit\n'
                     f'$R^2 = {sanity_r2:.3f}$, RMSE = ${sanity_rmse:.3f}$\n'
                     f'(CV estimate: $R^2 = {mean_r2:.3f} \\pm {std_r2:.3f}$)', 
                     fontsize=12, fontweight='bold')
    
    min_val = min(y_all_records.min(), y_pred_all.min())
    max_val = max(y_all_records.max(), y_pred_all.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label='1:1 Line')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal', 'box')
    
    residuals_all = y_all_records - y_pred_all
    axes[1].scatter(y_pred_all, residuals_all, alpha=0.3, s=10, color='coral')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Sap Velocity', fontsize=13)
    axes[1].set_ylabel('Residuals (Observed - Predicted)', fontsize=13)
    axes[1].set_title(f'Final Model: Residual Plot\nMAE = ${sanity_mae:.3f}$', 
                     fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / f'FINAL_model_sanity_check_{run_id}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    logging.info("\n" + "=" * 70)
    logging.info("FINAL MODEL TRAINING COMPLETE")
    logging.info("=" * 70)
    logging.info(f"Model:   {final_model_path}")
    logging.info(f"Scaler:  {final_scaler_path}")
    logging.info(f"Config:  {config_path}")
    logging.info(f"")
    logging.info(f"Expected generalization performance (from CV):")
    logging.info(f"  R²   = {mean_r2:.4f} ± {std_r2:.4f}")
    logging.info(f"  RMSE = {mean_rmse:.4f} ± {std_rmse:.4f}")
    
    return all_test_r2_scores, all_test_rmse_scores


if __name__ == "__main__":
    main()