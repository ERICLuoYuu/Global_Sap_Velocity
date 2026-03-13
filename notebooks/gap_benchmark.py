"""
Gap-Filling Benchmark for Sap Flow Time Series
===============================================

Systematic analysis of gap characteristics and gap-filling method performance
at hourly and daily temporal scales across SAPFLUXNET + internal European sites.

Purpose: Determine the optimal gap-filling method per gap size range and the
maximum reliable gap-filling threshold for the main merge pipeline.

Usage:
    python gap_benchmark.py
"""


# # Gap-Filling Benchmark for Sap Flow Time Series
#
# Systematic analysis of gap characteristics and gap-filling method performance
# at **hourly** and **daily** temporal scales across SAPFLUXNET + internal
# European sites.
#
# **Purpose:** Determine the optimal gap-filling method per gap size range and the
# maximum reliable gap-filling threshold for the main merge pipeline
# (`notebooks/merge_gap_filled_hourly_orginal.py`).
#
# | Phase | Description |
# |-------|-------------|
# | **1** | Gap Census — catalogue every gap at hourly and daily scale |
# | **2** | Gap Size Distribution — statistics and 5 visualisations |
# | **3** | Synthetic Experiment Setup — select sites, extract ground truth |
# | **4** | Method Definitions — 14 methods in 4 groups (A–D) |
# | **5** | Evaluation — benchmark all methods, compute 5 metrics |
# | **6** | Visualisation & Recommendations — 7 figures + decision matrix |


# ── Global Setup ─────────────────────────────────────────────────────────────
import sys
import os
import gc
import json
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
import seaborn as sns
from scipy import interpolate as scipy_interp

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', module='statsmodels')
warnings.filterwarnings('ignore', module='sklearn')
warnings.filterwarnings('ignore', module='xgboost')
warnings.filterwarnings('ignore', module='tensorflow')

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ── Paths (using project-wide PathConfig) ─────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # project root (one level above notebooks/)
sys.path.insert(0, str(ROOT / '.venv'))        # path_config.py lives in .venv/
sys.path.insert(0, str(ROOT))

from path_config import PathConfig, get_default_paths  # noqa: E402

_paths      = PathConfig(
    scale='sapwood',
    base_data_dir=str(ROOT / 'data'),
    base_output_dir=str(ROOT / 'outputs'),
)
SAP_DIR     = _paths.sap_outliers_removed_dir
ENV_DIR     = _paths.env_outliers_removed_dir
SITE_MD_DIR = _paths.raw_csv_dir
STATS_OUT   = _paths.base_output_dir / 'statistics' / 'gap_experiment'
FIGS_OUT    = _paths.figures_root / 'gap_experiment'

for _d in [STATS_OUT, FIGS_OUT, FIGS_OUT / 'hourly', FIGS_OUT / 'daily']:
    _d.mkdir(parents=True, exist_ok=True)

# ── Import project utilities ──────────────────────────────────────────────────
try:
    from notebooks.test_interpolation import (
        create_spaced_gaps, extend_gaps, evaluate_interpolation,
    )
    print('✓ Loaded test_interpolation utilities')
except ImportError as _e:
    print(f'Warning: Could not load test_interpolation: {_e} — using inline fallbacks')

    def create_spaced_gaps(n_gaps, min_spacing, max_idx):
        gaps, attempts = [], 0
        while len(gaps) < n_gaps and attempts < n_gaps * 10:
            ng = np.random.randint(0, max(1, max_idx))
            if all(abs(ng - g) >= min_spacing for g in gaps):
                gaps.append(ng)
            attempts += 1
        return sorted(gaps)

    def extend_gaps(gaps, size, max_idx):
        return [g + i for g in gaps for i in range(size) if g + i < max_idx]

    def evaluate_interpolation(original, interpolated, mask):
        o, p = original[~mask], interpolated[~mask]
        rmse = float(np.sqrt(np.mean((o - p) ** 2)))
        ss_res = float(np.sum((o - p) ** 2))
        ss_t = float(np.sum((o - o.mean()) ** 2))
        if ss_t < 1e-12 and ss_res < 1e-12:
            r2 = 1.0
        elif ss_t < 1e-12:
            r2 = 0.0
        else:
            r2 = float(1 - ss_res / ss_t)
        nse  = r2
        mae  = float(np.mean(np.abs(o - p)))
        nz   = o != 0
        mape = float(np.mean(np.abs((o[nz] - p[nz]) / o[nz])) * 100) if nz.any() else float('nan')
        return {'rmse': rmse, 'r2': r2, 'mae': mae, 'mape': mape, 'nse': nse}

# ── Constants ─────────────────────────────────────────────────────────────────
HOURLY_GAP_SIZES = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336, 720]
DAILY_GAP_SIZES  = [1, 2, 3, 5, 7, 14, 30]
N_REPLICATES     = 50
MIN_SEGMENT_LEN  = 2000   # minimum hourly timesteps for ground truth segment
R2_THRESHOLD     = 0.7
NSE_THRESHOLD    = 0.5
WINDOW           = 48     # DL input window (timesteps)

# ── Visualisation style ───────────────────────────────────────────────────────
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.1)
GROUP_COLORS = {'A': '#2196F3', 'B': '#4CAF50', 'C': '#FF9800', 'D': '#9C27B0'}
METRIC_LABELS = {
    'rmse': 'RMSE (cm³ cm⁻² h⁻¹)',
    'mae':  'MAE (cm³ cm⁻² h⁻¹)',
    'r2':   'R² (coefficient of determination)',
    'nse':  'NSE (Nash–Sutcliffe Efficiency)',
    'mape': 'MAPE (%)',
}


#
#

# ── Phase 1 helper functions ──────────────────────────────────────────────────

def detect_gaps_in_series(series: pd.Series) -> list:
    """
    Detect all contiguous NaN blocks in a pandas Series.

    Returns list of dicts with keys: start, end, duration_steps.
    """
    if len(series) == 0:
        return []
    is_nan = pd.isna(series).values.astype(np.int8)
    idx    = series.index
    n      = len(is_nan)
    padded = np.concatenate([[0], is_nan.astype(int), [0]])
    diff   = np.diff(padded)
    gap_starts = np.where(diff == 1)[0]
    gap_ends   = np.where(diff == -1)[0]
    gaps = []
    for s, e in zip(gap_starts, gap_ends):
        gaps.append({
            'start':          idx[s],
            'end':            idx[min(e - 1, n - 1)],
            'duration_steps': int(e - s),
        })
    return gaps


def load_site_metadata(site_name: str) -> dict:
    """
    Load site metadata (biome, IGBP, lat/lon) from SITE_MD_DIR.

    Returns dict with keys: si_biome, si_igbp, si_lat, si_long.
    """
    defaults = {'si_biome': 'Unknown', 'si_igbp': 'Unknown',
                'si_lat': np.nan, 'si_long': np.nan}
    md_file = SITE_MD_DIR / f'{site_name}_site_md.csv'
    if not md_file.exists():
        return defaults.copy()
    try:
        df_md = pd.read_csv(md_file, nrows=1)
        return {k: (df_md[k].iloc[0] if k in df_md.columns else v)
                for k, v in defaults.items()}
    except Exception:
        return defaults.copy()


def _safe_read_sap(fpath: Path) -> 'pd.DataFrame | None':
    """Read sap CSV, parse TIMESTAMP to UTC, drop junk columns."""
    try:
        df = pd.read_csv(fpath)
        df.drop(columns=[c for c in df.columns if 'Unnamed' in str(c)],
                inplace=True, errors='ignore')
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True, errors='coerce')
        df = df.dropna(subset=['TIMESTAMP']).set_index('TIMESTAMP').sort_index()
        return df
    except Exception as exc:
        print(f'  Warning: could not read {fpath.name}: {exc}')
        return None


def _get_plant_cols(df: pd.DataFrame) -> list:
    """Return sensor/plant columns (exclude TIMESTAMP, solar, Unnamed)."""
    return [c for c in df.columns
            if not any(x in c.lower() for x in ['timestamp', 'solar', 'unnamed'])]


def _infer_site_name(stem: str) -> str:
    """Extract site code (first 2 underscore parts) from CSV filename stem."""
    parts = stem.replace('_sapf_data_outliers_removed', '').split('_')
    return '_'.join(parts[:2]) if len(parts) >= 2 else parts[0]



# ── Environmental data helpers ────────────────────────────────────────────────
ENV_FEATURE_COLS = ['ta', 'vpd', 'sw_in', 'ppfd_in']  # core env features for ML/DL


def _safe_read_env(sap_fpath: Path) -> 'pd.DataFrame | None':
    """
    Load the matching environmental data CSV for a sap flow file.

    Derives env file path by replacing '_sapf_data_' with '_env_data_'.
    Returns DataFrame indexed by TIMESTAMP (UTC), or None on failure.
    """
    env_name = sap_fpath.name.replace('_sapf_data_', '_env_data_')
    env_path = ENV_DIR / env_name
    if not env_path.exists():
        return None
    try:
        df = pd.read_csv(env_path)
        df.drop(columns=[c for c in df.columns if 'Unnamed' in str(c)],
                inplace=True, errors='ignore')
        if 'solar_TIMESTAMP' in df.columns:
            df.drop(columns=['solar_TIMESTAMP'], inplace=True, errors='ignore')
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], utc=True, errors='coerce')
        df = df.dropna(subset=['TIMESTAMP']).set_index('TIMESTAMP').sort_index()
        # Keep only numeric env columns
        env_cols = [c for c in df.columns if c in ENV_FEATURE_COLS]
        return df[env_cols] if env_cols else None
    except Exception as exc:
        print(f'  Warning: could not read env {env_path.name}: {exc}')
        return None

# ── Phase 1: Main gap census loop ────────────────────────────────────────────


# ── Phase 2 constants ────────────────────────────────────────────────────────
HOURLY_BINS = {
    '1h':    (1,    1),
    '2h':    (2,    2),
    '3-6h':  (3,    6),
    '6-12h': (7,   12),
    '12-24h':(13,  24),
    '1-3d':  (25,   72),
    '3-7d':  (73,  168),
    '7-14d': (169, 336),
    '14-30d':(337, 720),
    '>30d':  (721, int(1e9)),
}
DAILY_BINS = {
    '1d':    (1,   1),
    '2d':    (2,   2),
    '3-5d':  (3,   5),
    '5-7d':  (6,   7),
    '7-14d': (8,  14),
    '14-30d':(15, 30),
    '>30d':  (31, int(1e9)),
}


def assign_bin(value: float, bins: dict) -> str:
    """Return the bin label for a numeric gap duration."""
    for bin_name, (lo, hi) in bins.items():
        if lo <= value <= hi:
            return bin_name
    return list(bins.keys())[-1]



def get_all_qualifying_segments(
    df: pd.DataFrame,
    min_len: int = MIN_SEGMENT_LEN,
) -> list:
    """
    Find ALL plant columns with contiguous non-NaN runs ≥ min_len.

    Returns list of (column_name, segment) tuples — one per qualifying column.
    For each column, returns the LONGEST contiguous segment.
    """
    results = []
    for col in _get_plant_cols(df):
        s = df[col]
        if s.notna().sum() < min_len:
            continue
        is_valid  = s.notna().astype(int)
        change_pt = (is_valid != is_valid.shift(1)).cumsum()
        best_seg, best_len = pd.Series(dtype=float), 0
        for grp_id, grp in s[s.notna()].groupby(change_pt[s.notna()]):
            if len(grp) >= min_len and len(grp) > best_len:
                best_seg, best_len = grp, len(grp)
        if best_len >= min_len:
            results.append((col, best_seg))
    return results


def inject_gaps_replicated(
    gt_series: pd.Series,
    gap_size: int,
    n_reps: int,
    rng: np.random.Generator,
) -> list:
    """
    Create n_reps replicates of gt_series, each with one synthetic gap injected.

    Returns list of (masked_series, gap_indices_list) tuples.
    Enforces minimum spacing = 2 × gap_size between gap positions across replicates
    to ensure diverse gap placement.
    """
    n = len(gt_series)
    if n < gap_size * 3:
        return []
    max_start = n - gap_size
    min_spacing = 2 * gap_size
    results = []
    used_starts: list = []
    max_attempts = n_reps * 10  # avoid infinite loop
    attempts = 0
    while len(results) < n_reps and attempts < max_attempts:
        attempts += 1
        start = int(rng.integers(0, max_start))
        # Check spacing against all previously used positions
        too_close = any(abs(start - prev) < min_spacing for prev in used_starts)
        if too_close:
            continue
        used_starts.append(start)
        idx   = list(range(start, start + gap_size))
        masked = gt_series.copy()
        masked.iloc[idx] = np.nan
        results.append((masked, idx))
    if len(results) < n_reps:
        # Fill remaining replicates without spacing constraint
        for _ in range(n_reps - len(results)):
            start = int(rng.integers(0, max_start))
            idx   = list(range(start, start + gap_size))
            masked = gt_series.copy()
            masked.iloc[idx] = np.nan
            results.append((masked, idx))
    return results



# ── Group A: Classical Interpolation ─────────────────────────────────────────
import scipy.interpolate as scipy_interp


def fill_linear(s: pd.Series) -> pd.Series:
    """Linear interpolation between known values."""
    return s.interpolate(method='linear', limit_direction='both').clip(lower=0)


def fill_cubic(s: pd.Series) -> pd.Series:
    """Cubic spline interpolation (order 3); falls back to linear on failure."""
    try:
        return s.interpolate(method='spline', order=3,
                             limit_direction='both').clip(lower=0)
    except Exception:
        return fill_linear(s)


def fill_akima(s: pd.Series) -> pd.Series:
    """
    Akima interpolation — monotonicity-preserving, avoids overshooting.

    Extrapolation positions fall back to nearest-neighbour interpolation.
    """
    if s.notna().sum() < 4:
        return fill_linear(s)
    try:
        valid_idx  = np.where(s.notna())[0]
        valid_vals = s.values[valid_idx]
        akima      = scipy_interp.Akima1DInterpolator(valid_idx, valid_vals)
        filled     = s.copy()
        null_pos   = np.where(s.isna())[0]
        in_range   = (null_pos >= valid_idx[0]) & (null_pos <= valid_idx[-1])
        if in_range.any():
            filled.iloc[null_pos[in_range]] = akima(null_pos[in_range])
        if (~in_range).any():
            filled = filled.interpolate(method='nearest', limit_direction='both')
        return filled.clip(lower=0)
    except Exception:
        return fill_linear(s)


def fill_nearest(s: pd.Series) -> pd.Series:
    """Nearest-neighbour interpolation."""
    return s.interpolate(method='nearest', limit_direction='both').clip(lower=0)



# ── Group B: Statistical / Climatological ────────────────────────────────────

def _is_hourly_series(s: pd.Series) -> bool:
    """Return True when the median time step is ≤ 2 h."""
    if len(s) < 2:
        return True
    median_dt = pd.Series(s.index).diff().dropna().median()
    return bool(median_dt <= pd.Timedelta('2h'))


def fill_mdv(s: pd.Series, window_days: int = 7) -> pd.Series:
    """
    Mean Diurnal Variation gap-filling (vectorized).

    Each missing timestep is replaced by the mean of observed values at the
    same hour-of-day (hourly) or day-of-week (daily) within ±window_days.
    Residual NaNs fall back to linear interpolation.
    """
    filled    = s.copy()
    null_mask = s.isna()
    if not null_mask.any():
        return filled.clip(lower=0)
    hourly = _is_hourly_series(s)
    # Build a period key for grouping (hour-of-day for hourly, day-of-week for daily)
    period_key = s.index.hour if hourly else s.index.dayofweek
    for period_val in pd.unique(period_key[null_mask]):
        # All missing timesteps at this period value
        missing_at_period = null_mask & (period_key == period_val)
        if not missing_at_period.any():
            continue
        # For each missing timestamp, compute mean from ±window_days of same-period observed data
        for ts in s.index[missing_at_period]:
            w_start = ts - pd.Timedelta(days=window_days)
            w_end   = ts + pd.Timedelta(days=window_days)
            ctx = s[(s.index >= w_start) & (s.index <= w_end)
                    & (period_key == period_val) & s.notna()]
            if len(ctx) > 0:
                filled[ts] = max(0.0, float(ctx.mean()))
    if filled.isna().any():
        filled = fill_linear(filled)
    return filled.clip(lower=0)


def fill_rolling_mean(s: pd.Series, window: int = None) -> pd.Series:
    """
    Rolling window mean gap-filling.

    Default window: 48 timesteps (hourly) or 7 (daily).
    """
    if window is None:
        window = 48 if _is_hourly_series(s) else 7
    rolling = s.rolling(window=window, min_periods=max(1, window // 4), center=True).mean()
    filled  = s.fillna(rolling)
    if filled.isna().any():
        filled = fill_linear(filled)
    return filled.clip(lower=0)


def fill_stl(s: pd.Series) -> pd.Series:
    """
    STL decomposition + residual interpolation.

    Falls back to fill_rolling_mean on error or insufficient data.
    """
    try:
        from statsmodels.tsa.seasonal import STL
        if not s.isna().any():
            return s.clip(lower=0)
        period = 24 if _is_hourly_series(s) else 7
        if len(s) < period * 2:
            return fill_rolling_mean(s)
        # Pre-fill gaps with linear interpolation (less biased than ffill/bfill for large gaps)
        s_pre = s.interpolate(method='linear', limit_direction='both')
        if s_pre.isna().any():
            s_pre = s_pre.ffill().bfill()
        if s_pre.isna().any():
            s_pre = s_pre.fillna(s_pre.mean() if not s_pre.isna().all() else 0.0)
        res             = STL(s_pre, period=period, robust=True).fit()
        residual        = pd.Series(res.resid, index=s.index)
        residual[s.isna()] = np.nan
        residual_filled = residual.interpolate(
            method='linear', limit_direction='both').fillna(0.0)
        return pd.Series(res.trend + res.seasonal + residual_filled.values,
                         index=s.index).clip(lower=0)
    except Exception:
        return fill_rolling_mean(s)



# ── Group C: Machine Learning ─────────────────────────────────────────────────
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

N_LAGS = 24


def build_ml_features(s: pd.Series, n_lags: int = N_LAGS,
                      env_df: 'pd.DataFrame | None' = None) -> pd.DataFrame:
    """
    Build lag + temporal features for ML gap-filling.

    Features: t-1…t-n_lags, hour_of_day, day_of_year, month,
              rolling_mean_24, rolling_std_24, is_daytime.
    If env_df is provided, adds lagged env features (t-1…t-n_lags) for each env column.
    """
    df = pd.DataFrame({'y': s.values}, index=s.index)
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df['hour_of_day']     = df.index.hour
    df['day_of_year']     = df.index.dayofyear
    df['month']           = df.index.month
    df['rolling_mean_24'] = df['y'].rolling(24, min_periods=1).mean()
    df['rolling_std_24']  = df['y'].rolling(24, min_periods=1).std().fillna(0)
    df['is_daytime']      = ((df.index.hour >= 6) & (df.index.hour <= 20)).astype(int)
    # Add env features if available (C3 fix)
    if env_df is not None:
        env_aligned = env_df.reindex(df.index)
        for ecol in env_aligned.columns:
            for lag in range(1, n_lags + 1):
                df[f'env_{ecol}_lag_{lag}'] = env_aligned[ecol].shift(lag)
    return df.drop(columns=['y'])


def _fit_and_predict_ml(s: pd.Series, model_cls, model_kwargs: dict,
                        env_df: 'pd.DataFrame | None' = None) -> pd.Series:
    """Generic ML gap-fill: fit on observed data, predict at gap positions."""
    features   = build_ml_features(s, env_df=env_df)
    train_mask = s.notna() & features.notna().all(axis=1)
    test_mask  = s.isna()  & features.notna().all(axis=1)
    if train_mask.sum() < 50:
        return fill_linear(s)
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(features[train_mask].values)
    model     = model_cls(**model_kwargs)
    model.fit(X_train_s, s[train_mask].values)
    filled = s.copy()
    if test_mask.sum() > 0:
        X_test = scaler.transform(features[test_mask].values)
        filled[test_mask] = np.clip(model.predict(X_test), 0, None)
    if filled.isna().any():
        filled = fill_linear(filled)
    return filled.clip(lower=0)


def fill_rf(s: pd.Series) -> pd.Series:
    """Random Forest gap-filling with 24-lag feature set."""
    return _fit_and_predict_ml(s, RandomForestRegressor,
                               {'n_estimators': 100, 'n_jobs': -1,
                                'random_state': RANDOM_SEED})


def fill_xgb_model(s: pd.Series) -> pd.Series:
    """XGBoost gap-filling with 24-lag feature set."""
    return _fit_and_predict_ml(s, xgb.XGBRegressor,
                               {'n_estimators': 100, 'learning_rate': 0.1,
                                'max_depth': 5, 'random_state': RANDOM_SEED,
                                'verbosity': 0, 'tree_method': 'hist'})


def fill_knn(s: pd.Series, k: int = 10) -> pd.Series:
    """KNN gap-filling with 24-lag feature set."""
    k_actual = min(k, max(1, int(s.notna().sum() * 0.1)))
    if k_actual < k:
        print(f'    [KNN] k reduced from {k} to {k_actual} '
              f'(only {s.notna().sum()} observed values)')
    return _fit_and_predict_ml(s, KNeighborsRegressor, {'n_neighbors': k_actual})




# ── Group C-env: ML with environmental features (C3 fix) ─────────────────────
# These variants accept env_df via closure set during Phase 5 benchmark loop.
# The _current_env_df global is set per-segment during benchmarking.
_current_env_df: 'pd.DataFrame | None' = None


def fill_rf_env(s: pd.Series) -> pd.Series:
    """Random Forest with sap + env features."""
    return _fit_and_predict_ml(s, RandomForestRegressor,
                               {'n_estimators': 100, 'n_jobs': -1,
                                'random_state': RANDOM_SEED},
                               env_df=_current_env_df)


def fill_xgb_env(s: pd.Series) -> pd.Series:
    """XGBoost with sap + env features."""
    return _fit_and_predict_ml(s, xgb.XGBRegressor,
                               {'n_estimators': 100, 'learning_rate': 0.1,
                                'max_depth': 5, 'random_state': RANDOM_SEED,
                                'verbosity': 0, 'tree_method': 'hist'},
                               env_df=_current_env_df)


def fill_knn_env(s: pd.Series, k: int = 10) -> pd.Series:
    """KNN with sap + env features."""
    k_actual = min(k, max(1, int(s.notna().sum() * 0.1)))
    return _fit_and_predict_ml(s, KNeighborsRegressor, {'n_neighbors': k_actual},
                               env_df=_current_env_df)



# ── Group D: Deep Learning ────────────────────────────────────────────────────
_HAS_TF = False
try:
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(RANDOM_SEED)
    _HAS_TF = True
except ImportError:
    pass  # TF not installed; DL methods fall back to linear
DL_EPOCHS    = 50
DL_PATIENCE  = 5
DL_BATCH     = 32
DL_VAL_SPLIT = 0.1


if not _HAS_TF:
    # Stub DL functions when TensorFlow unavailable
    def fill_lstm(s): return fill_linear(s)
    def fill_cnn(s): return fill_linear(s)
    def fill_cnn_lstm(s): return fill_linear(s)
    def fill_transformer(s): return fill_linear(s)
    def fill_lstm_env(s): return fill_linear(s)
    def fill_cnn_env(s): return fill_linear(s)
    def fill_cnn_lstm_env(s): return fill_linear(s)
    def fill_transformer_env(s): return fill_linear(s)
    PositionalEmbedding = None

if _HAS_TF:
    class PositionalEmbedding(keras.layers.Layer):
        """
        Learned positional embedding layer for the Transformer encoder.

        Adds a trainable embedding vector for each sequence position.
        """

        def __init__(self, sequence_length: int, d_model: int, **kwargs):
            super().__init__(**kwargs)
            self.sequence_length = sequence_length
            self.d_model         = d_model
            self.pos_emb         = keras.layers.Embedding(
                input_dim=sequence_length, output_dim=d_model)

        def call(self, x: tf.Tensor) -> tf.Tensor:
            """Add positional embeddings to the input tensor."""
            return x + self.pos_emb(tf.range(self.sequence_length))

        def get_config(self) -> dict:
            config = super().get_config()
            config.update({'sequence_length': self.sequence_length,
                           'd_model': self.d_model})
            return config


# Guard: all remaining DL functions check _HAS_TF internally

def _build_dl_sequences(s: pd.Series, window: int = WINDOW,
                        env_df: 'pd.DataFrame | None' = None):
    """
    Build supervised (X, y) arrays from contiguous, fully-observed windows.

    If env_df is None: returns X: (n, window, 1) and y: (n,)
    If env_df provided: returns X: (n, window, 1+n_env_cols) and y: (n,)
    Only clean windows (no NaNs in sap or env) are included.

    Note (H4): Windows adjacent to gap boundaries are retained. While gap
    positions themselves are excluded (NaN check), windows ending 1 step
    before a gap may encode information about the gap's existence through
    e.g. sensor drift patterns. This is a conservative design choice —
    excluding all boundary-adjacent windows would significantly reduce
    training data on short series.
    """
    vals = s.values.astype(np.float32)
    # Build env feature array if provided
    env_vals = None
    n_features = 1
    if env_df is not None:
        env_aligned = env_df.reindex(s.index).ffill().bfill().fillna(0)
        env_vals = env_aligned.values.astype(np.float32)
        n_features = 1 + env_vals.shape[1]
    X_list, y_list = [], []
    for i in range(window, len(vals)):
        ctx_sap = vals[i - window:i]
        tgt = vals[i]
        if np.isnan(tgt) or np.any(np.isnan(ctx_sap)):
            continue
        if env_vals is not None:
            ctx_env = env_vals[i - window:i]
            if np.any(np.isnan(ctx_env)):
                continue
            ctx = np.column_stack([ctx_sap, ctx_env])  # (window, n_features)
        else:
            ctx = ctx_sap[:, np.newaxis]  # (window, 1)
        X_list.append(ctx)
        y_list.append(tgt)
    if not X_list:
        return (np.empty((0, window, n_features), dtype=np.float32),
                np.empty(0, dtype=np.float32))
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def _train_dl_model(
    s: pd.Series,
    model,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    env_df: 'pd.DataFrame | None' = None,
) -> 'object | None':
    """
    Fit a compiled Keras model on observed, non-gap data.

    Returns the fitted model or None when too few training samples exist.
    Scaler is applied per-feature across the flattened feature dimension (M5 fix).
    """
    X, y = _build_dl_sequences(s, env_df=env_df)
    if len(X) < 100:
        return None
    # Scale per-feature: reshape (n, window, features) → (n*window, features)
    n_samples, window, n_features = X.shape
    X_flat = X.reshape(-1, n_features)
    X_scaled = scaler_X.fit_transform(X_flat).reshape(X.shape)
    ys  = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    model.fit(
        X_scaled, ys,
        epochs=DL_EPOCHS, batch_size=DL_BATCH,
        validation_split=DL_VAL_SPLIT, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(
            patience=DL_PATIENCE, restore_best_weights=True)],
    )
    return model


def _predict_at_gaps(
    s: pd.Series,
    model,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    window: int = WINDOW,
    env_df: 'pd.DataFrame | None' = None,
) -> pd.Series:
    """
    Autoregressively predict values at gap positions using the trained model.

    Earlier predictions become context for later predictions within the same gap.
    Supports multi-feature input when env_df is provided.

    Note (H5): Autoregressive prediction compounds errors for long gaps.
    For gaps >> window (e.g., 168-720h), early prediction errors feed into
    later predictions, potentially causing drift. Monitor error growth by
    comparing early vs late gap positions in evaluation.
    """
    filled = s.copy()
    vals   = s.values.copy().astype(np.float32)
    # Prepare env values if available
    env_vals = None
    n_features = 1
    if env_df is not None:
        env_aligned = env_df.reindex(s.index).ffill().bfill().fillna(0)
        env_vals = env_aligned.values.astype(np.float32)
        n_features = 1 + env_vals.shape[1]
    for i in range(window, len(vals)):
        if np.isnan(vals[i]):
            ctx_sap = vals[i - window:i].copy()
            ctx_sap = pd.Series(ctx_sap).ffill().bfill().fillna(0).values.astype(np.float32)
            if env_vals is not None:
                ctx_env = env_vals[i - window:i]
                ctx = np.column_stack([ctx_sap, ctx_env])  # (window, n_features)
            else:
                ctx = ctx_sap[:, np.newaxis]  # (window, 1)
            ctx_flat = ctx.reshape(-1, n_features)
            ctx_sc = scaler_X.transform(ctx_flat).reshape(1, window, n_features)
            pred   = float(scaler_y.inverse_transform(
                model.predict(ctx_sc, verbose=0))[0, 0])
            pred   = max(0.0, pred)
            vals[i]      = pred
            filled.iloc[i] = pred
    return filled.clip(lower=0)


def _dl_fill(s: pd.Series, build_model_fn,
             env_df: 'pd.DataFrame | None' = None) -> pd.Series:
    """
    Generic DL gap-filling pipeline.

    Builds the model, trains on observed data, predicts at gaps.
    Falls back to fill_linear when training fails or TF unavailable.
    If env_df provided, builds multi-feature model (sap + env).
    """
    if not _HAS_TF:
        return fill_linear(s)
    keras.backend.clear_session()
    # Determine input feature dimension
    n_features = 1
    if env_df is not None:
        env_cols = [c for c in env_df.columns if c in ENV_FEATURE_COLS]
        if env_cols:
            n_features = 1 + len(env_cols)
        else:
            env_df = None  # no valid env columns
    model    = build_model_fn(n_features=n_features)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    fitted   = _train_dl_model(s, model, scaler_X, scaler_y, env_df=env_df)
    if fitted is None:
        return fill_linear(s)
    return _predict_at_gaps(s, fitted, scaler_X, scaler_y, env_df=env_df)


def fill_lstm(s: pd.Series) -> pd.Series:
    """2-layer LSTM (64 units each) gap-filling."""
    def _build(n_features=1):
        return keras.Sequential([
            keras.layers.LSTM(64, return_sequences=True, input_shape=(WINDOW, n_features)),
            keras.layers.Dropout(0.1),
            keras.layers.LSTM(64),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(1),
        ])
    return _dl_fill(s, _build)


def fill_cnn(s: pd.Series) -> pd.Series:
    """3-layer 1D-CNN (32→64→32, kernel=3) gap-filling."""
    def _build(n_features=1):
        return keras.Sequential([
            keras.layers.Conv1D(32, 3, padding='same', activation='relu',
                                input_shape=(WINDOW, n_features)),
            keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
            keras.layers.Conv1D(32, 3, padding='same', activation='relu'),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1),
        ])
    return _dl_fill(s, _build)


def fill_cnn_lstm(s: pd.Series) -> pd.Series:
    """CNN-LSTM hybrid: Conv1D (32→64) → LSTM(64) → Dense(1)."""
    def _build(n_features=1):
        inp = keras.Input(shape=(WINDOW, n_features))
        x   = keras.layers.Conv1D(32, 3, padding='same', activation='relu')(inp)
        x   = keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x)
        x   = keras.layers.LSTM(64)(x)
        x   = keras.layers.Dropout(0.1)(x)
        out = keras.layers.Dense(1)(x)
        return keras.Model(inp, out)
    return _dl_fill(s, _build)


def fill_transformer(s: pd.Series) -> pd.Series:
    """
    Encoder-only Transformer: d_model=64, 2 attention heads, 2 encoder layers.

    Uses learned positional embeddings via PositionalEmbedding layer.
    """
    def _enc_block(x, d_model=64, n_heads=2, ff_dim=128, drop=0.1):
        attn    = keras.layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads)(x, x)
        attn    = keras.layers.Dropout(drop)(attn)
        x       = keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ff_out  = keras.layers.Dense(ff_dim, activation='relu')(x)
        ff_out  = keras.layers.Dense(d_model)(ff_out)
        ff_out  = keras.layers.Dropout(drop)(ff_out)
        return keras.layers.LayerNormalization(epsilon=1e-6)(x + ff_out)

    def _build(n_features=1):
        inp = keras.Input(shape=(WINDOW, n_features))
        x   = keras.layers.Dense(64)(inp)
        x   = PositionalEmbedding(WINDOW, 64)(x)
        x   = _enc_block(x)
        x   = _enc_block(x)
        x   = keras.layers.GlobalAveragePooling1D()(x)
        out = keras.layers.Dense(1)(x)
        return keras.Model(inp, out)
    return _dl_fill(s, _build)


# ── Method registry ───────────────────────────────────────────────────────────
METHODS = {
    'A_linear':      fill_linear,
    'A_cubic':       fill_cubic,
    'A_akima':       fill_akima,
    'A_nearest':     fill_nearest,
    'B_mdv':         fill_mdv,
    'B_rolling':     fill_rolling_mean,
    'B_stl':         fill_stl,
    'C_rf':          fill_rf,
    'C_xgb':         fill_xgb_model,
    'C_knn':         fill_knn,
    'D_lstm':        fill_lstm,
    'D_cnn':         fill_cnn,
    'D_cnn_lstm':    fill_cnn_lstm,
    'D_transformer': fill_transformer,
}

# ── Env-aware methods (C3 fix) ───────────────────────────────────────────────
# These use _current_env_df (set per-segment in Phase 5 benchmark loop)

def fill_lstm_env(s: pd.Series) -> pd.Series:
    """LSTM with sap + env features."""
    return _dl_fill(s, lambda n_features=1: keras.Sequential([
        keras.layers.LSTM(64, return_sequences=True, input_shape=(WINDOW, n_features)),
        keras.layers.Dropout(0.1),
        keras.layers.LSTM(64),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1),
    ]), env_df=_current_env_df)


def fill_cnn_env(s: pd.Series) -> pd.Series:
    """1D-CNN with sap + env features."""
    return _dl_fill(s, lambda n_features=1: keras.Sequential([
        keras.layers.Conv1D(32, 3, padding='same', activation='relu',
                            input_shape=(WINDOW, n_features)),
        keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
        keras.layers.Conv1D(32, 3, padding='same', activation='relu'),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1),
    ]), env_df=_current_env_df)


def fill_cnn_lstm_env(s: pd.Series) -> pd.Series:
    """CNN-LSTM with sap + env features."""
    def _build(n_features=1):
        inp = keras.Input(shape=(WINDOW, n_features))
        x   = keras.layers.Conv1D(32, 3, padding='same', activation='relu')(inp)
        x   = keras.layers.Conv1D(64, 3, padding='same', activation='relu')(x)
        x   = keras.layers.LSTM(64)(x)
        x   = keras.layers.Dropout(0.1)(x)
        out = keras.layers.Dense(1)(x)
        return keras.Model(inp, out)
    return _dl_fill(s, _build, env_df=_current_env_df)


def fill_transformer_env(s: pd.Series) -> pd.Series:
    """Transformer with sap + env features."""
    def _enc_block(x, d_model=64, n_heads=2, ff_dim=128, drop=0.1):
        attn    = keras.layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads)(x, x)
        attn    = keras.layers.Dropout(drop)(attn)
        x       = keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ff_out  = keras.layers.Dense(ff_dim, activation='relu')(x)
        ff_out  = keras.layers.Dense(d_model)(ff_out)
        ff_out  = keras.layers.Dropout(drop)(ff_out)
        return keras.layers.LayerNormalization(epsilon=1e-6)(x + ff_out)
    def _build(n_features=1):
        inp = keras.Input(shape=(WINDOW, n_features))
        x   = keras.layers.Dense(64)(inp)
        x   = PositionalEmbedding(WINDOW, 64)(x)
        x   = _enc_block(x)
        x   = _enc_block(x)
        x   = keras.layers.GlobalAveragePooling1D()(x)
        out = keras.layers.Dense(1)(x)
        return keras.Model(inp, out)
    return _dl_fill(s, _build, env_df=_current_env_df)


# Add env-aware methods to registry
METHODS_ENV = {
    'Ce_rf_env':          fill_rf_env,
    'Ce_xgb_env':         fill_xgb_env,
    'Ce_knn_env':         fill_knn_env,
    'De_lstm_env':        fill_lstm_env,
    'De_cnn_env':         fill_cnn_env,
    'De_cnn_lstm_env':    fill_cnn_lstm_env,
    'De_transformer_env': fill_transformer_env,
}
METHODS.update(METHODS_ENV)

METHOD_GROUPS = {
    'A':  ['A_linear', 'A_cubic', 'A_akima', 'A_nearest'],
    'B':  ['B_mdv', 'B_rolling', 'B_stl'],
    'C':  ['C_rf', 'C_xgb', 'C_knn'],
    'D':  ['D_lstm', 'D_cnn', 'D_cnn_lstm', 'D_transformer'],
    'Ce': ['Ce_rf_env', 'Ce_xgb_env', 'Ce_knn_env'],
    'De': ['De_lstm_env', 'De_cnn_env', 'De_cnn_lstm_env', 'De_transformer_env'],
}

GROUP_COLORS.update({'Ce': '#E65100', 'De': '#6A1B9A'})


#
# Run the full benchmark at both **hourly** and **daily** scales.
# For each *(time_scale × method × gap_size × site × replicate)*:
# 1. Inject a synthetic gap into the ground truth segment
# 2. Apply the gap-filling method
# 3. Compute RMSE, MAE, R², MAPE, NSE at gap positions vs ground truth
#
# Results are saved incrementally after each method completes.

# ── Phase 5: Metrics ─────────────────────────────────────────────────────────


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(true_vals: np.ndarray, pred_vals: np.ndarray) -> dict:
    """
    Compute RMSE, MAE, R², MAPE, NSE — robust to NaNs and edge cases.

    NSE ≡ R² (using observed mean as baseline, standard hydrological form).
    MAPE skips zero-valued actuals to avoid division by zero.
    """
    _m = ~(np.isnan(true_vals) | np.isnan(pred_vals))
    t, p = true_vals[_m], pred_vals[_m]
    _nan5 = {k: np.nan for k in ['rmse', 'mae', 'r2', 'mape', 'nse']}
    if len(t) == 0:
        return _nan5
    rmse   = float(np.sqrt(np.mean((t - p) ** 2)))
    mae    = float(np.mean(np.abs(t - p)))
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    if ss_tot > 1e-12:
        r2 = float(1.0 - ss_res / ss_tot)
    else:
        r2 = 1.0 if ss_res < 1e-12 else 0.0
    # NSE computed independently (same formula, but decoupled for future changes)
    if ss_tot > 1e-12:
        nse = float(1.0 - ss_res / ss_tot)
    else:
        nse = 1.0 if ss_res < 1e-12 else 0.0
    nz     = np.abs(t) > 1e-10
    mape   = (float(np.mean(np.abs((t[nz] - p[nz]) / t[nz])) * 100)
              if nz.sum() > 0 else np.nan)
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape, 'nse': nse}

# ── Phase 5: Full benchmark loop ─────────────────────────────────────────────

# ── Phase 6 visualisation helpers ────────────────────────────────────────────
def _method_color(m: str) -> str:
    """Return hex color for a method based on its group prefix (A/B/C/D)."""
    return GROUP_COLORS.get(m.split('_')[0], '#607D8B')


_LS_MAP = {
    'A_linear': '-', 'A_cubic': '--', 'A_akima': '-.', 'A_nearest': ':',
    'B_mdv': '-', 'B_rolling': '--', 'B_stl': '-.',
    'C_rf': '-', 'C_xgb': '--', 'C_knn': '-.',
    'D_lstm': '-', 'D_cnn': '--', 'D_cnn_lstm': '-.', 'D_transformer': ':',
}


def _method_ls(m: str) -> str:
    """Return linestyle for a method."""
    return _LS_MAP.get(m, '-')


def find_reliability_threshold(
    df_agg: pd.DataFrame,
    scale: str,
    metric: str = 'r2_mean',
    thr: float = R2_THRESHOLD,
) -> 'int | None':
    """
    Return the smallest gap_size where best-method metric drops below thr.

    Returns None when all gap sizes meet the threshold.
    """
    _sub  = df_agg[df_agg.time_scale == scale]
    _best = _sub.groupby('gap_size')[metric].max().reset_index()
    _below = _best[_best[metric] < thr]
    return int(_below['gap_size'].min()) if len(_below) else None


def _human_size(gs: float, scale: str) -> str:
    """Convert gap size to human-readable string."""
    if scale == 'hourly':
        return f'{int(gs)}h' if gs < 24 else f'{gs/24:.0f}d'
    return f'{int(gs)}d'




# ── Phase 6 Fig 1: Performance vs gap size (line plots) ──────────────────────


# ==========================================================================
# Phase runner functions
# ==========================================================================


def run_phase1():
    """Phase 1: Gap Census at hourly and daily scales.

    Returns (df_gaps_h, df_gaps_d, df_site_summary).
    """
    print('\n' + '=' * 60)
    print('  PHASE 1: GAP CENSUS')
    print('=' * 60)

    master_gaps_hourly: list = []
    master_gaps_daily:  list = []
    site_summary_rows: list = []    # H2 fix: compute per-site completeness in same pass

    sap_files = sorted(SAP_DIR.glob('*outliers_removed.csv'))
    n_files   = len(sap_files)
    print(f'Found {n_files} sap flow files to process...')

    for _fi, _fpath in enumerate(sap_files):
        if _fi % 10 == 0 or _fi == n_files - 1:
            print(f'  Processing file {_fi + 1}/{n_files}: {_fpath.stem[:55]}...')

        df_raw = _safe_read_sap(_fpath)
        if df_raw is None:
            continue

        plant_cols = _get_plant_cols(df_raw)
        if not plant_cols:
            continue

        site_name = _infer_site_name(_fpath.stem)
        _site_md  = load_site_metadata(site_name)
        biome     = str(_site_md.get('si_biome', 'Unknown'))
        igbp      = str(_site_md.get('si_igbp', 'Unknown'))

        df_h = df_raw[plant_cols].resample('h').mean()
        df_d = df_raw[plant_cols].resample('D').mean()

        for col in plant_cols:
            for g in detect_gaps_in_series(df_h[col]):
                d_h = g['duration_steps']
                master_gaps_hourly.append({
                    'site': site_name, 'column': col,
                    'start': str(g['start']), 'end': str(g['end']),
                    'duration_hours': d_h, 'duration_days': round(d_h / 24.0, 4),
                    'biome': biome, 'igbp': igbp,
                })

            for g in detect_gaps_in_series(df_d[col]):
                d_d = g['duration_steps']
                master_gaps_daily.append({
                    'site': site_name, 'column': col,
                    'start': str(g['start']), 'end': str(g['end']),
                    'duration_hours': float(d_d * 24), 'duration_days': d_d,
                    'biome': biome, 'igbp': igbp,
                })

        # Per-site completeness (computed here to avoid double I/O — H2 fix)
        _total   = df_h.size
        _present = int(df_h.notna().sum().sum())
        _comp    = _present / _total * 100 if _total else 0.0
        site_summary_rows.append({
            'site':                site_name,
            'biome':               biome,
            'igbp':                igbp,
            'n_columns':           len(plant_cols),
            'n_hourly_timesteps':  len(df_h),
            'completeness_pct':    round(_comp, 2),
        })

        gc.collect()

    df_gaps_h = pd.DataFrame(master_gaps_hourly).reset_index(drop=True)
    df_gaps_d = pd.DataFrame(master_gaps_daily).reset_index(drop=True)

    # Convert timestamps back to datetime
    for _df in [df_gaps_h, df_gaps_d]:
        if not _df.empty:
            _df['start'] = pd.to_datetime(_df['start'], utc=True, errors='coerce')
            _df['end']   = pd.to_datetime(_df['end'],   utc=True, errors='coerce')

    df_gaps_h.to_csv(STATS_OUT / 'master_gaps_hourly.csv', index=False)
    df_gaps_d.to_csv(STATS_OUT / 'master_gaps_daily.csv',  index=False)

    # Summary
    _sep = '=' * 60
    for _label, _df, _dur_col, _unit in [
        ('HOURLY', df_gaps_h, 'duration_hours', 'h'),
        ('DAILY',  df_gaps_d, 'duration_days',  'd'),
    ]:
        if _df.empty:
            continue
        _s = _df[_dur_col]
        print(f'\n{_sep}\n  {_label} GAP CENSUS\n{_sep}')
        print(f'  Total gaps:          {len(_df):>10,}')
        print(f'  Total missing {_unit}: {_s.sum():>12,.0f}')
        print(f'  Unique sites:        {_df["site"].nunique():>10,}')
        print(f'  Median gap size:     {_s.median():>10.1f} {_unit}')
        print(f'  P90 gap size:        {_s.quantile(0.90):>10.1f} {_unit}')
        print(f'  P99 gap size:        {_s.quantile(0.99):>10.1f} {_unit}')

    print(f'\nSaved master_gaps_hourly.csv ({len(df_gaps_h):,} rows)')
    print(f'Saved master_gaps_daily.csv  ({len(df_gaps_d):,} rows)')

    # ── Phase 1: Per-site completeness (already computed during gap census) ───────
    df_site_summary = (pd.DataFrame(site_summary_rows)
                         .sort_values('completeness_pct', ascending=False)
                         .reset_index(drop=True))
    df_site_summary.to_csv(STATS_OUT / 'site_completeness.csv', index=False)

    print(f'Sites analysed:       {len(df_site_summary)}')
    print(f'Median completeness:  {df_site_summary["completeness_pct"].median():.1f}%')
    print(f'Sites ≥ 90%:          {(df_site_summary["completeness_pct"] >= 90).sum()}')
    print(f'Sites ≥ 70%:          {(df_site_summary["completeness_pct"] >= 70).sum()}')
    print('\nTop 15 sites by completeness:')
    print(df_site_summary[['site', 'biome', 'completeness_pct']].head(15).to_string(index=False))

    # ── Phase 1: Hourly vs daily gap count comparison ─────────────────────────────
    _h_per = (df_gaps_h.groupby('site')
              .agg(n_gaps_hourly=('duration_hours', 'count'),
                   total_missing_hours=('duration_hours', 'sum'))
              .reset_index())
    _d_per = (df_gaps_d.groupby('site')
              .agg(n_gaps_daily=('duration_days', 'count'),
                   total_missing_days=('duration_days', 'sum'))
              .reset_index())

    _comp = _h_per.merge(_d_per, on='site', how='outer').fillna(0)
    _comp['hourly_to_daily_ratio'] = (_comp['n_gaps_hourly']
                                       / _comp['n_gaps_daily'].replace(0, np.nan))
    _comp.to_csv(STATS_OUT / 'hourly_vs_daily_comparison.csv', index=False)

    print('Hourly vs Daily gap comparison (first 15 sites):')
    print(_comp[['site', 'n_gaps_hourly', 'n_gaps_daily', 'hourly_to_daily_ratio']]
          .head(15).to_string(index=False))
    print(f'\nMean hourly:daily gap ratio: {_comp["hourly_to_daily_ratio"].mean():.1f}×')


    return df_gaps_h, df_gaps_d, df_site_summary


def run_phase2(df_gaps_h, df_gaps_d):
    """Phase 2: Gap Size Distribution Analysis.

    Adds size_bin columns, prints statistics, generates 5 figures.
    Returns (df_gaps_h, df_gaps_d) with bin columns added.
    """
    print('\n' + '=' * 60)
    print('  PHASE 2: GAP SIZE DISTRIBUTION')
    print('=' * 60)

    if not df_gaps_h.empty:
        df_gaps_h['size_bin'] = df_gaps_h['duration_hours'].apply(
            lambda x: assign_bin(x, HOURLY_BINS))
    if not df_gaps_d.empty:
        df_gaps_d['size_bin'] = df_gaps_d['duration_days'].apply(
            lambda x: assign_bin(x, DAILY_BINS))


    def _print_gap_stats(label: str, series: pd.Series, unit: str) -> None:
        """Print descriptive statistics for a gap duration series."""
        print(f'\n{label} STATISTICS')
        print('=' * 52)
        print(f'  Count  : {len(series):>10,}')
        print(f'  Mean   : {series.mean():>10.1f} {unit}')
        print(f'  Median : {series.median():>10.1f} {unit}')
        try:
            print(f'  Mode   : {float(series.mode().iloc[0]):>10.1f} {unit}')
        except Exception:
            pass
        print(f'  Std    : {series.std():>10.1f} {unit}')
        print(f'  Min    : {series.min():>10.0f} {unit}')
        print(f'  Max    : {series.max():>10.0f} {unit}')
        for p in [5, 25, 50, 75, 90, 95, 99]:
            print(f'  P{p:02d}    : {series.quantile(p / 100):>10.1f} {unit}')


    if not df_gaps_h.empty:
        _print_gap_stats('HOURLY GAP DURATION', df_gaps_h['duration_hours'], 'h')
        print('\nHOURLY BIN FREQUENCIES')
        _tot = len(df_gaps_h)
        for b in HOURLY_BINS:
            cnt = (df_gaps_h['size_bin'] == b).sum()
            print(f'  {b:<12}: {cnt:>8,}  ({cnt/_tot*100:5.1f}%)')

    if not df_gaps_d.empty:
        _print_gap_stats('DAILY GAP DURATION', df_gaps_d['duration_days'], 'd')
        print('\nDAILY BIN FREQUENCIES')
        _tot = len(df_gaps_d)
        for b in DAILY_BINS:
            cnt = (df_gaps_d['size_bin'] == b).sum()
            print(f'  {b:<12}: {cnt:>8,}  ({cnt/_tot*100:5.1f}%)')

    # ── Phase 2: Visualisations ───────────────────────────────────────────────────

    # Figure 1 — Histograms (log x-axis)
    _fig1, (_ax1a, _ax1b) = plt.subplots(1, 2, figsize=(14, 6))
    for _ax, _df, _col, _title in [
        (_ax1a, df_gaps_h, 'duration_hours', 'Hourly gap duration'),
        (_ax1b, df_gaps_d, 'duration_days',  'Daily gap duration'),
    ]:
        if _df.empty:
            continue
        _vals = _df[_col].values
        _vals = _vals[_vals > 0]
        _log_bins = np.logspace(np.log10(max(0.5, _vals.min())),
                                np.log10(_vals.max() + 1), 60)
        _ax.hist(_vals, bins=_log_bins, color='steelblue', edgecolor='white',
                 alpha=0.85, linewidth=0.4)
        for _pct, _ls, _c in [(90, '--', 'darkorange'), (95, ':', 'crimson')]:
            _v = np.percentile(_vals, _pct)
            _ax.axvline(_v, linestyle=_ls, color=_c, linewidth=1.8,
                        label=f'P{_pct} = {_v:.0f}')
        _ax.set_xscale('log')
        _ax.set_xlabel(_title.replace('gap', 'Gap'))
        _ax.set_ylabel('Number of gaps')
        _ax.set_title(_title, fontweight='bold')
        _ax.legend(fontsize=9)
        _ax.grid(True, alpha=0.3, which='both')
    _fig1.suptitle('Gap Size Distribution', fontweight='bold', fontsize=13)
    _fig1.tight_layout()
    _fig1.savefig(str(FIGS_OUT / 'fig1_gap_histogram.png'), dpi=300, bbox_inches='tight')
    plt.close(_fig1)
    print('Saved: fig1_gap_histogram.png')

    # Figure 2 — CDF
    _fig2, _ax2 = plt.subplots(figsize=(10, 6))
    for _df, _col, _label, _c in [
        (df_gaps_h, 'duration_hours', 'Hourly', 'steelblue'),
        (df_gaps_d, 'duration_days',  'Daily',  'salmon'),
    ]:
        if _df.empty:
            continue
        _s = np.sort(_df[_col].values)
        _s = _s[_s > 0]
        _cdf = np.arange(1, len(_s) + 1) / len(_s)
        _ax2.plot(_s, _cdf, color=_c, linewidth=2, label=_label)
        for _pct, _ls in [(50, ':'), (90, '--'), (95, '-.')]:
            _v = np.percentile(_s, _pct)
            _ax2.axvline(_v, linestyle=_ls, color=_c, linewidth=0.9, alpha=0.7)
            _ax2.text(_v * 1.05, 0.02 + _pct / 200, f'P{_pct}\n{_v:.0f}',
                      fontsize=6, color=_c)
    _ax2.set_xscale('log')
    _ax2.set_xlabel('Gap size (hours / days)')
    _ax2.set_ylabel('Fraction of gaps ≤ X')
    _ax2.set_title('Cumulative Distribution of Gap Sizes', fontweight='bold')
    _ax2.legend()
    _ax2.set_ylim(0, 1.02)
    _fig2.tight_layout()
    _fig2.savefig(str(FIGS_OUT / 'fig2_gap_cdf.png'), dpi=300, bbox_inches='tight')
    plt.close(_fig2)
    print('Saved: fig2_gap_cdf.png')

    # Figure 3 — PFT (IGBP) boxplot
    _fig3, _ax3 = plt.subplots(figsize=(14, 6))
    if not df_gaps_h.empty and 'igbp' in df_gaps_h.columns:
        _bc = df_gaps_h['igbp'].value_counts()
        _valid = _bc[_bc > 10].index.tolist()
        _df3 = df_gaps_h[df_gaps_h['igbp'].isin(_valid)].copy()
        _df3['log_dur'] = np.log10(_df3['duration_hours'] + 1)
        _order3 = (_df3.groupby('igbp')['log_dur'].median()
                   .sort_values(ascending=False).index.tolist())
        _data3 = [_df3.loc[_df3['igbp'] == b, 'log_dur'].values for b in _order3]
        _bp = _ax3.boxplot(_data3, labels=_order3, patch_artist=True,
                           medianprops=dict(color='black', linewidth=2))
        _cmap = plt.cm.tab20
        for _i, _patch in enumerate(_bp['boxes']):
            _patch.set_facecolor(_cmap(_i / max(len(_order3), 1)))
            _patch.set_alpha(0.75)
    else:
        _ax3.text(0.5, 0.5, 'No PFT (IGBP) data available',
                  ha='center', va='center', transform=_ax3.transAxes)
    _ax3.set_xlabel('Plant Functional Type (IGBP)')
    _ax3.set_ylabel('log₁₀(gap hours + 1)')
    _ax3.set_title('Hourly Gap Duration by Plant Functional Type', fontweight='bold')
    _ax3.tick_params(axis='x', rotation=45)
    _ax3.grid(True, alpha=0.3, axis='y')
    _fig3.tight_layout()
    _fig3.savefig(str(FIGS_OUT / 'fig3_gap_pft_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close(_fig3)
    print('Saved: fig3_gap_pft_boxplot.png')

    # Figure 4 — Gap-map heatmap (top 20 sites)
    _fig4, _ax4 = plt.subplots(figsize=(16, 8))
    if not df_gaps_h.empty and df_gaps_h['start'].notna().any():
        _top20 = (df_gaps_h.groupby('site')['duration_hours'].count()
                  .sort_values(ascending=False).head(20).index.tolist())
        _df4 = df_gaps_h[df_gaps_h['site'].isin(_top20)].copy()
        _df4['month_period'] = _df4['start'].dt.to_period('M')
        # Vectorized pivot (replaces slow nested loop)
        _agg = (_df4.groupby(['site', 'month_period'])['duration_hours']
                .sum().reset_index())
        _agg['days_in_month'] = _agg['month_period'].apply(lambda p: p.days_in_month)
        _agg['frac'] = (_agg['duration_hours'] / (_agg['days_in_month'] * 24)).clip(upper=1.0)
        _pivot_df = (_agg.pivot(index='site', columns='month_period', values='frac')
                     .reindex(_top20).fillna(0))
        _pivot_df.columns = [str(c) for c in _pivot_df.columns]
        _im = _ax4.imshow(_pivot_df.values, aspect='auto', cmap='YlOrRd',
                          vmin=0, vmax=1, interpolation='nearest')
        plt.colorbar(_im, ax=_ax4, label='Fraction of hours missing', shrink=0.8)
        _ticks = list(range(0, _pivot_df.shape[1], max(1, _pivot_df.shape[1] // 20)))
        _ax4.set_xticks(_ticks)
        _ax4.set_xticklabels([_pivot_df.columns[i] for i in _ticks],
                             rotation=45, ha='right', fontsize=7)
        _ax4.set_yticks(range(len(_top20)))
        _ax4.set_yticklabels(_top20, fontsize=8)
    else:
        _ax4.text(0.5, 0.5, 'No gap data available',
                  ha='center', va='center', transform=_ax4.transAxes)
    _ax4.set_title('Monthly Gap Fraction — Top 20 Sites', fontweight='bold')
    _fig4.tight_layout()
    _fig4.savefig(str(FIGS_OUT / 'fig4_gap_map.png'), dpi=300, bbox_inches='tight')
    plt.close(_fig4)
    print('Saved: fig4_gap_map.png')

    # Figure 5 — Bin frequency bar charts
    _fig5, (_ax5a, _ax5b) = plt.subplots(1, 2, figsize=(14, 6))
    for _ax, _df, _bins, _title, _c in [
        (_ax5a, df_gaps_h, HOURLY_BINS, 'Hourly gap bin frequencies', 'steelblue'),
        (_ax5b, df_gaps_d, DAILY_BINS,  'Daily gap bin frequencies',  'salmon'),
    ]:
        if _df.empty or 'size_bin' not in _df.columns:
            continue
        _bkeys = list(_bins.keys())
        _counts = [(_df['size_bin'] == b).sum() for b in _bkeys]
        _bars = _ax.bar(_bkeys, _counts, color=_c, edgecolor='white',
                        linewidth=0.5, alpha=0.85)
        _mx = max(_counts) if _counts else 1
        for _bar, _cnt in zip(_bars, _counts):
            if _cnt:
                _ax.text(_bar.get_x() + _bar.get_width() / 2,
                         _bar.get_height() + _mx * 0.01,
                         f'{_cnt:,}', ha='center', va='bottom', fontsize=7)
        _ax.set_title(_title, fontweight='bold')
        _ax.set_xlabel('Gap size category')
        _ax.set_ylabel('Number of gaps')
        _ax.tick_params(axis='x', rotation=45)
        _ax.grid(True, alpha=0.3, axis='y')
    _fig5.tight_layout()
    _fig5.savefig(str(FIGS_OUT / 'fig5_bin_frequency.png'), dpi=300, bbox_inches='tight')
    plt.close(_fig5)
    print('Saved: fig5_bin_frequency.png')
    print(f'\n✓ All Phase 2 figures saved to: {FIGS_OUT}')


    return df_gaps_h, df_gaps_d


def run_phase3(df_site_summary):
    """Phase 3: Synthetic Gap Experiment Setup.

    Selects sites, extracts ground truth segments, injects synthetic gaps.
    Returns ground_truth_store dict.
    """
    print('\n' + '=' * 60)
    print('  PHASE 3: SYNTHETIC GAP EXPERIMENT SETUP')
    print('=' * 60)


    # Add biome column if missing
    if 'biome' not in df_site_summary.columns:
        df_site_summary['biome'] = df_site_summary['site'].apply(
            lambda s: str(load_site_metadata(s).get('si_biome', 'Unknown')))

    # Filter by completeness, lowering threshold if needed
    for _thr in [85.0, 75.0, 65.0, 50.0]:
        _df_filt = df_site_summary[df_site_summary['completeness_pct'] >= _thr]
        if len(_df_filt) >= 15:
            break

    print(f'Using completeness threshold: {_thr}%  ({len(_df_filt)} qualifying sites)')

    # Use ALL qualifying sites (C4/M7 fix — spec: "all plant columns")
    selected_sites = _df_filt.sort_values('completeness_pct', ascending=False).reset_index(drop=True)

    selected_sites.to_csv(STATS_OUT / 'selected_sites.csv', index=False)
    print(f'\nSelected {len(selected_sites)} sites for benchmarking:')
    print(f'Biomes represented: {selected_sites["biome"].nunique()}')
    print(selected_sites[['site', 'biome', 'completeness_pct']].head(30).to_string(index=False))

    # ── Phase 3: Ground truth extraction ─────────────────────────────────────────


    ground_truth_store: dict = {}
    _rng_gt = np.random.default_rng(RANDOM_SEED)
    _total_segments = 0

    for _, _row in selected_sites.iterrows():
        _site = str(_row['site'])
        _biome = str(_row.get('biome', 'Unknown'))
        _candidates = sorted(SAP_DIR.glob(f'{_site}_*_sapf_data_outliers_removed.csv'))
        if not _candidates:
            print(f'  ⚠ No CSV found for {_site}')
            continue
        _df_raw = _safe_read_sap(_candidates[0])
        if _df_raw is None:
            continue

        # Load matching environmental data (C3 fix)
        _df_env_raw = _safe_read_env(_candidates[0])
        _has_env = _df_env_raw is not None and len(_df_env_raw) > 0

        _pcols = _get_plant_cols(_df_raw)
        _df_h  = _df_raw[_pcols].resample('h').mean()
        _df_d  = _df_raw[_pcols].resample('D').mean()

        # Resample env data to match sap data resolution
        _env_h = _df_env_raw.resample('h').mean() if _has_env else None
        _env_d = _df_env_raw.resample('D').mean() if _has_env else None

        # Get ALL qualifying columns (C4/M7 fix)
        _hourly_segments = get_all_qualifying_segments(_df_h, MIN_SEGMENT_LEN)
        if not _hourly_segments:
            print(f'  ⚠ {_site}: no hourly segment ≥ {MIN_SEGMENT_LEN}')
            continue

        for _col_h, _seg_h in _hourly_segments:
            # Find matching daily segment for this column
            if _col_h in _df_d.columns:
                _daily_segments = get_all_qualifying_segments(
                    _df_d[[_col_h]], min_len=30)
            else:
                _daily_segments = []
            _seg_d = _daily_segments[0][1] if _daily_segments else pd.Series(dtype=float)

            _key = f'{_site}__{_col_h}'
            # Align env data to the sap segment's time range
            _env_h_seg = None
            _env_d_seg = None
            if _env_h is not None:
                _overlap = _env_h.index.intersection(_seg_h.index)
                if len(_overlap) > len(_seg_h) * 0.5:
                    _env_h_seg = _env_h.reindex(_seg_h.index)
            if _env_d is not None and len(_seg_d) > 0:
                _overlap_d = _env_d.index.intersection(_seg_d.index)
                if len(_overlap_d) > len(_seg_d) * 0.5:
                    _env_d_seg = _env_d.reindex(_seg_d.index)

            ground_truth_store[_key] = {
                'hourly': _seg_h,
                'daily':  _seg_d,
                'env_hourly': _env_h_seg,  # None if unavailable
                'env_daily':  _env_d_seg,
                'site':   _site,
                'col':    _col_h,
                'biome':  _biome,
            }
            _d_len = len(_seg_d) if len(_seg_d) > 0 else 0
            _total_segments += 1
            print(f'  ✓ {_site}/{_col_h}: hourly={len(_seg_h)}h  daily={_d_len}d')

        gc.collect()

    print(f'\nGround truth store: {_total_segments} segments from '
          f'{len(set(v["site"] for v in ground_truth_store.values()))} sites')
    _n_with_env = sum(1 for v in ground_truth_store.values() if v.get('env_hourly') is not None)
    print(f'Segments with env data: {_n_with_env}/{_total_segments}')

    # Quick verification
    if ground_truth_store:
        _ts = list(ground_truth_store.values())[0]['hourly']
        _tr = inject_gaps_replicated(_ts, 24, 3, np.random.default_rng(42))
        if _tr:
            print(f'\nVerification: 24h gap injection → '
                  f'{_ts.notna().sum()} → {_tr[0][0].notna().sum()} non-null values')
        else:
            print('⚠ inject_gaps_replicated returned empty — segment too short?')


    return ground_truth_store


def run_phase5(ground_truth_store):
    """Phase 5: Full benchmark - evaluate all methods on all segments.

    Returns (df_results, df_agg).
    """
    global _current_env_df
    print('\n' + '=' * 60)
    print('  PHASE 5: BENCHMARK EVALUATION')
    print('=' * 60)

    _rng_bench = np.random.default_rng(RANDOM_SEED)
    all_results: list = []
    _partial_path = STATS_OUT / 'benchmark_results_partial.csv'
    _n_sites   = len(ground_truth_store)
    _n_methods = len(METHODS)

    print(f'Segments: {_n_sites}  Methods: {_n_methods}  Replicates: {N_REPLICATES}')
    print(f'Hourly gap sizes: {HOURLY_GAP_SIZES}')
    print(f'Daily gap sizes:  {DAILY_GAP_SIZES}')

    for _scale, _gsizes in [('hourly', HOURLY_GAP_SIZES), ('daily', DAILY_GAP_SIZES)]:
        print(f'\n{"="*65}\nSCALE: {_scale.upper()}\n{"="*65}')

        for _mi, (_mname, _mfn) in enumerate(METHODS.items()):
            _group = _mname.split('_')[0]
            _t0    = time.time()
            _fails = 0
            _m_rows: list = []

            for _si, (_key, _sdata) in enumerate(ground_truth_store.items()):
                _site = _sdata.get('site', _key)
                _gt = _sdata.get(_scale)
                if _gt is None or len(_gt) < 50:
                    continue

                # Set env data for env-aware methods (C3 fix)
                _env_key = f'env_{_scale}'
                _current_env_df = _sdata.get(_env_key)

                # Skip env-aware methods if no env data for this segment
                _is_env_method = _mname.endswith('_env')
                if _is_env_method and _current_env_df is None:
                    continue

                for _gsize in _gsizes:
                    if _gsize >= len(_gt) * 0.5:
                        continue
                    _reps = inject_gaps_replicated(_gt, _gsize, N_REPLICATES, _rng_bench)
                    for _ri, (_gs, _gidx) in enumerate(_reps):
                        if not _gidx:
                            continue
                        try:
                            _filled = _mfn(_gs.copy()).clip(lower=0)
                            _tv     = _gt.iloc[_gidx].values
                            _pv     = _filled.iloc[_gidx].values
                            _met    = compute_metrics(_tv, _pv)
                        except Exception:
                            _met   = {k: np.nan for k in ['rmse', 'mae', 'r2', 'mape', 'nse']}
                            _fails += 1
                        _m_rows.append({
                            'time_scale': _scale, 'site': _site,
                            'method': _mname, 'group': _group,
                            'gap_size': _gsize, 'replicate': _ri,
                            'env_features': _is_env_method,
                            **_met,
                        })

                if (_si + 1) % 5 == 0:
                    _el = time.time() - _t0
                    print(f'  [{_scale}] {_mname} ({_mi+1}/{_n_methods}) | '
                          f'site {_si+1}/{_n_sites} | {_el:.0f}s | fails={_fails}')

            all_results.extend(_m_rows)
            pd.DataFrame(all_results).to_csv(_partial_path, index=False)
            print(f'  ✓ {_mname} [{_scale}] done in {time.time()-_t0:.0f}s | '
                  f'rows={len(_m_rows):,} | fails={_fails}')
            gc.collect()

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(STATS_OUT / 'benchmark_results_full.csv', index=False)
    print(f'\n✓ Benchmark complete — {len(df_results):,} records saved')


        # Aggregate across replicates

    df_agg = (
        df_results
        .groupby(['time_scale', 'method', 'group', 'gap_size'])
        .agg(
            rmse_mean=('rmse', 'mean'), rmse_std=('rmse', 'std'),
            mae_mean=('mae',  'mean'), mae_std=('mae',  'std'),
            r2_mean=('r2',   'mean'), r2_std=('r2',   'std'),
            mape_mean=('mape','mean'), mape_std=('mape','std'),
            nse_mean=('nse',  'mean'), nse_std=('nse',  'std'),
            n_obs=('replicate', 'count'),
            failure_rate=('r2', lambda x: x.isna().mean()),
        )
        .reset_index()
    )
    df_agg.to_csv(STATS_OUT / 'benchmark_aggregated.csv', index=False)

    print(f'Aggregated: {len(df_agg)} rows')
    print('\nTop 5 methods @ hourly 24h gap (by mean R²):')
    _sub24 = df_agg[(df_agg.time_scale == 'hourly') & (df_agg.gap_size == 24)]
    if len(_sub24):
        print(_sub24.nlargest(5, 'r2_mean')
              [['method', 'r2_mean', 'r2_std', 'rmse_mean']].to_string(index=False))

    return df_results, df_agg


def run_phase6(df_results, df_agg):
    """Phase 6: Visualisations and Recommendations.

    Generates 7 publication-quality figures and a decision matrix.
    Returns (threshold_hourly, threshold_daily).
    """
    print('\n' + '=' * 60)
    print('  PHASE 6: VISUALISATION & RECOMMENDATIONS')
    print('=' * 60)

    _metrics4 = ['rmse', 'mae', 'r2', 'nse']

    for _scale in ['hourly', 'daily']:
        _sub = df_agg[df_agg.time_scale == _scale]
        if _sub.empty:
            continue
        _gsizes = sorted(_sub.gap_size.unique())
        _fig, _axes = plt.subplots(2, 2, figsize=(14, 10))
        _axes = _axes.ravel()

        for _ai, _metric in enumerate(_metrics4):
            _ax = _axes[_ai]
            _mc = f'{_metric}_mean'
            _sc = f'{_metric}_std'
            if _mc not in _sub.columns:
                continue
            for _m in METHODS:
                _md = _sub[_sub.method == _m].sort_values('gap_size')
                if _md.empty:
                    continue
                _x  = _md.gap_size.values.astype(float)
                _y  = _md[_mc].values
                _ys = _md[_sc].fillna(0).values
                _c  = _method_color(_m)
                _ls = _method_ls(_m)
                _ax.plot(_x, _y, color=_c, linestyle=_ls, linewidth=1.5,
                         label=_m, alpha=0.85)
                _ax.fill_between(_x, _y - _ys, _y + _ys, color=_c, alpha=0.08)
            if _metric == 'r2':
                _ax.axhline(R2_THRESHOLD, color='red', linestyle='--',
                            linewidth=1, label=f'R²={R2_THRESHOLD}')
            if _metric == 'nse':
                _ax.axhline(NSE_THRESHOLD, color='orange', linestyle='--',
                            linewidth=1, label=f'NSE={NSE_THRESHOLD}')
            _ax.set_xscale('log')
            _ax.set_xlabel('Gap size (log scale)')
            _ax.set_ylabel(METRIC_LABELS.get(_metric, _metric))
            _ax.set_title(_metric.upper())
            _ax.set_xticks(_gsizes)
            _ax.set_xticklabels([_human_size(g, _scale) for g in _gsizes],
                                rotation=40, ha='right', fontsize=7)

        _handles = [Patch(facecolor=GROUP_COLORS[g], label=f'Group {g}')
                    for g in ['A', 'B', 'C', 'D', 'Ce', 'De'] if g in GROUP_COLORS]
        _fig.legend(handles=_handles, loc='lower center', ncol=6, fontsize=9,
                    bbox_to_anchor=(0.5, -0.02))
        _fig.suptitle(
            f'Gap-Filling Performance vs Gap Size ({_scale.capitalize()})',
            fontweight='bold', fontsize=13)
        _fig.tight_layout(rect=[0, 0.04, 1, 1])
        _fp = str(FIGS_OUT / _scale / 'fig_performance_vs_gapsize.png')
        _fig.savefig(_fp, dpi=300)
        plt.close(_fig)
        print(f'✓ fig_performance_vs_gapsize.png  [{_scale}]')

    # ── Phase 6 Fig 2: R² heatmap ────────────────────────────────────────────────
    for _scale in ['hourly', 'daily']:
        _sub = df_agg[df_agg.time_scale == _scale]
        if _sub.empty:
            continue
        _gsizes  = sorted(_sub.gap_size.unique())
        _m_order = [m for g in ['A', 'B', 'C', 'D', 'Ce', 'De']
                    for m in METHOD_GROUPS.get(g, [])]
        _pivot   = (_sub.pivot_table(index='method', columns='gap_size',
                                     values='r2_mean', aggfunc='mean')
                    .reindex(index=_m_order).reindex(columns=_gsizes))

        _fig2, _ax2 = plt.subplots(
            figsize=(max(10, len(_gsizes) * 1.1), max(6, len(_m_order) * 0.55)))
        sns.heatmap(_pivot, cmap='RdYlGn', vmin=0, vmax=1, annot=True,
                    fmt='.2f', linewidths=0.4, ax=_ax2,
                    annot_kws={'size': 7}, cbar_kws={'label': 'R²'})
        _ax2.set_xticklabels(
            [_human_size(float(c), _scale) for c in _gsizes],
            rotation=40, ha='right', fontsize=8)
        _ax2.set_xlabel('Gap size')
        _ax2.set_ylabel('Method')

        # Group separator lines
        _cum = 0
        _sep_pos = []
        for _grp in ['A', 'B', 'C', 'D', 'Ce']:
            _cum += len(METHOD_GROUPS.get(_grp, []))
            _sep_pos.append(_cum)
        for _sp in _sep_pos:
            _ax2.axhline(_sp, color='black', linewidth=1.5)

        # Reliability threshold vertical line
        _thr = find_reliability_threshold(df_agg, _scale)
        if _thr is not None and _thr in _gsizes:
            _tx = _gsizes.index(_thr) + 0.5
            _ax2.axvline(_tx, color='red', linewidth=2, linestyle='--',
                         label=f'Threshold: {_human_size(_thr, _scale)}')
            _ax2.legend(fontsize=8, loc='upper right')

        _ax2.set_title(f'R² Heatmap — {_scale.capitalize()} Scale', fontweight='bold')
        _fig2.tight_layout()
        _fig2.savefig(str(FIGS_OUT / _scale / 'fig_r2_heatmap.png'), dpi=300)
        plt.close(_fig2)
        print(f'✓ fig_r2_heatmap.png  [{_scale}]')

    # ── Phase 6 Fig 3: Crossover analysis ────────────────────────────────────────

    def _find_crossover(x1: np.ndarray, y1: np.ndarray,
                        x2: np.ndarray, y2: np.ndarray) -> list:
        """Return x positions where y2 crosses above y1."""
        _cx = np.intersect1d(x1, x2)
        if len(_cx) < 2:
            return []
        _yy1 = np.interp(_cx, x1, y1)
        _yy2 = np.interp(_cx, x2, y2)
        _diff = _yy2 - _yy1
        _cross = np.where((_diff[:-1] < 0) & (_diff[1:] >= 0))[0]
        return [float(_cx[i + 1]) for i in _cross]


    for _scale in ['hourly', 'daily']:
        _sub    = df_agg[df_agg.time_scale == _scale]
        if _sub.empty:
            continue
        _gsizes = np.array(sorted(_sub.gap_size.unique()), dtype=float)

        _best_per_group: dict = {}
        for _grp, _methods in METHOD_GROUPS.items():
            _gd = _sub[_sub.method.isin(_methods)]
            if _gd.empty:
                continue
            _best_m = _gd.groupby('method')['r2_mean'].mean().idxmax()
            _best_per_group[_grp] = _best_m

        _fig3, _ax3 = plt.subplots(figsize=(11, 6))
        _group_lines: dict = {}
        for _grp, _m in _best_per_group.items():
            _md = _sub[_sub.method == _m].sort_values('gap_size')
            _x  = _md.gap_size.values.astype(float)
            _y  = _md['r2_mean'].values
            _ax3.plot(_x, _y, color=GROUP_COLORS[_grp], linewidth=2.5,
                      label=f'Group {_grp}: {_m}')
            _group_lines[_grp] = (_x, _y)

        _grp_order = [g for g in ['A', 'B', 'C', 'D'] if g in _group_lines]
        for _i in range(len(_grp_order) - 1):
            _g1, _g2 = _grp_order[_i], _grp_order[_i + 1]
            _x1, _y1 = _group_lines[_g1]
            _x2, _y2 = _group_lines[_g2]
            for _cx in _find_crossover(_x1, _y1, _x2, _y2):
                _cy = float(np.interp(_cx, _x1, _y1))
                _ax3.axvline(_cx, color='grey', linestyle=':', linewidth=1)
                _ax3.annotate(
                    f'G{_g2}>G{_g1}\n{_human_size(_cx, _scale)}',
                    xy=(_cx, _cy), xytext=(_cx * 1.2, _cy + 0.05),
                    arrowprops={'arrowstyle': '->', 'color': 'grey'},
                    fontsize=7, color='grey')

        _ax3.axhline(R2_THRESHOLD, color='red', linestyle='--',
                     linewidth=1, label=f'R²={R2_THRESHOLD}')
        _ax3.set_xscale('log')
        _ax3.set_xlabel('Gap size (log scale)')
        _ax3.set_ylabel('R² (best method per group)')
        _ax3.set_title(f'Method-Group Crossover — {_scale.capitalize()}', fontweight='bold')
        _ax3.set_xticks(_gsizes)
        _ax3.set_xticklabels([_human_size(g, _scale) for g in _gsizes],
                             rotation=40, ha='right', fontsize=8)
        _ax3.legend(fontsize=8)
        _fig3.tight_layout()
        _fig3.savefig(str(FIGS_OUT / _scale / 'fig_crossover_analysis.png'), dpi=300)
        plt.close(_fig3)
        print(f'✓ fig_crossover_analysis.png  [{_scale}]')

    # ── Phase 6 Fig 4: Decision matrix ───────────────────────────────────────────
    _decision_records: dict = {}

    for _scale in ['hourly', 'daily']:
        _sub  = df_agg[df_agg.time_scale == _scale]
        if _sub.empty:
            continue
        _bins = HOURLY_BINS if _scale == 'hourly' else DAILY_BINS

        _rows_dm = []
        for _bl, (_lo, _hi) in _bins.items():
            _bsub = _sub[(_sub.gap_size >= _lo) &
                         (_sub.gap_size <= min(_hi, _sub.gap_size.max()))]
            if _bsub.empty:
                continue
            _best    = _bsub.loc[_bsub.r2_mean.idxmax()]
            _r2_val  = float(_best.r2_mean)
            _rmse_val = float(_best.rmse_mean)
            _rel = ('Reliable' if _r2_val >= 0.8
                    else 'Marginal' if _r2_val >= R2_THRESHOLD
                    else '⚠ Unreliable')
            _rows_dm.append({
                'Gap Size Range': _bl,
                'Best Method':    str(_best.method),
                'R² mean':        f'{_r2_val:.2f}',
                'RMSE mean':      f'{_rmse_val:.3f}',
                'Reliability':    _rel,
            })
        _decision_records[_scale] = _rows_dm

        print(f'\nDecision Matrix — {_scale.upper()}')
        print(f'{"Gap Range":>12}  {"Best Method":>18}  {"R²":>5}  {"RMSE":>7}  Reliability')
        print('-' * 65)
        for _r in _rows_dm:
            print(f'{_r["Gap Size Range"]:>12}  {_r["Best Method"]:>18}  '
                  f'{_r["R² mean"]:>5}  {_r["RMSE mean"]:>7}  {_r["Reliability"]}')

        if _rows_dm:
            _col_labels = list(_rows_dm[0].keys())
            _cell_data  = [[str(r[c]) for c in _col_labels] for r in _rows_dm]
            _fig4, _ax4 = plt.subplots(
                figsize=(13, max(3, len(_rows_dm) * 0.55 + 1.5)))
            _ax4.axis('off')
            _tbl = _ax4.table(cellText=_cell_data, colLabels=_col_labels,
                              loc='center', cellLoc='center')
            _tbl.auto_set_font_size(False)
            _tbl.set_fontsize(9)
            _tbl.scale(1, 1.6)
            _rel_ci = _col_labels.index('Reliability')
            for _ri, _row in enumerate(_rows_dm):
                _cell = _tbl[_ri + 1, _rel_ci]
                if 'Reliable' in _row['Reliability'] and '⚠' not in _row['Reliability']:
                    _cell.set_facecolor('#C8E6C9')
                elif 'Marginal' in _row['Reliability']:
                    _cell.set_facecolor('#FFF9C4')
                else:
                    _cell.set_facecolor('#FFCDD2')
            _ax4.set_title(f'Decision Matrix — {_scale.capitalize()} Scale',
                           fontweight='bold', pad=12)
            _fig4.tight_layout()
            _fig4.savefig(str(FIGS_OUT / _scale / 'fig_decision_matrix.png'), dpi=300)
            plt.close(_fig4)
            print(f'  ✓ fig_decision_matrix.png  [{_scale}]')

    with open(str(STATS_OUT / 'decision_matrix.json'), 'w') as _fj:
        json.dump(_decision_records, _fj, indent=2)

    # ── Phase 6 Fig 5: Reliability threshold ─────────────────────────────────────
    _thresholds: dict = {}

    for _scale in ['hourly', 'daily']:
        _sub = df_agg[df_agg.time_scale == _scale]
        if _sub.empty:
            continue
        _best  = (_sub.groupby('gap_size')
                  .agg(r2_mean=('r2_mean', 'max'), r2_std=('r2_std', 'mean'),
                       nse_mean=('nse_mean', 'max'))
                  .reset_index()
                  .sort_values('gap_size'))
        _gsizes = _best.gap_size.values.astype(float)
        _thr    = find_reliability_threshold(df_agg, _scale)
        _thresholds[_scale] = _thr

        _fig5, _ax5 = plt.subplots(figsize=(10, 6))
        _x  = _best.gap_size.values.astype(float)
        _y  = _best.r2_mean.values
        _ys = _best.r2_std.fillna(0).values
        _ax5.plot(_x, _y, color='#2196F3', linewidth=2.5, label='Best method R²')
        _ax5.fill_between(_x, _y - _ys, _y + _ys, color='#2196F3', alpha=0.15)
        _ax5.axhline(R2_THRESHOLD, color='red', linestyle='--', linewidth=1.5,
                     label=f'R² = {R2_THRESHOLD} (threshold)')
        _ax5.axhline(NSE_THRESHOLD, color='orange', linestyle=':', linewidth=1.2,
                     label=f'NSE = {NSE_THRESHOLD}')
        if _thr is not None:
            _ax5.axvline(_thr, color='red', linewidth=1.5, linestyle=':')
            _ymax = float(max(_y)) if len(_y) else 1.0
            _ax5.fill_betweenx([0, _ymax], _thr, _x.max() * 2,
                               color='red', alpha=0.08, label='Do-not-fill zone')
            _ax5.annotate(
                f'Threshold:\n{_human_size(_thr, _scale)}',
                xy=(_thr, R2_THRESHOLD),
                xytext=(_thr * 0.4, R2_THRESHOLD - 0.15),
                arrowprops={'arrowstyle': '->', 'color': 'red'},
                fontsize=9, color='red', fontweight='bold')
        _ax5.set_xscale('log')
        _ax5.set_xlabel('Gap size (log scale)')
        _ax5.set_ylabel('R²')
        _ax5.set_title(
            f'Reliability Threshold — {_scale.capitalize()} Scale', fontweight='bold')
        _ax5.set_xticks(_gsizes)
        _ax5.set_xticklabels([_human_size(g, _scale) for g in _gsizes],
                             rotation=40, ha='right', fontsize=8)
        _ax5.legend()
        _ax5.set_ylim(bottom=0)
        _fig5.tight_layout()
        _fig5.savefig(str(FIGS_OUT / _scale / 'fig_reliability_threshold.png'), dpi=300)
        plt.close(_fig5)
        print(f'✓ fig_reliability_threshold.png  [{_scale}]')

    print(f'\nReliability thresholds: {_thresholds}')

    # ── Phase 6 Fig 6: Box plots per method group ────────────────────────────────
    for _scale in ['hourly', 'daily']:
        _df_s = df_results[df_results.time_scale == _scale] if 'df_results' in dir() else pd.DataFrame()
        if _df_s.empty:
            continue
        _gsizes_s = HOURLY_GAP_SIZES if _scale == 'hourly' else DAILY_GAP_SIZES

        for _grp, _methods in METHOD_GROUPS.items():
            _df_g = _df_s[_df_s.method.isin(_methods)]
            if _df_g.empty:
                continue
            _valid_gs = [g for g in _gsizes_s if g in _df_g.gap_size.unique()]
            if not _valid_gs:
                continue

            _ncols  = min(6, len(_valid_gs))
            _nrows  = (len(_valid_gs) - 1) // _ncols + 1
            _fig6, _axes6 = plt.subplots(_nrows, _ncols,
                                         figsize=(3.5 * _ncols, 4 * _nrows),
                                         squeeze=False)
            _axes6_flat = _axes6.ravel()

            for _gi, _gsize in enumerate(_valid_gs):
                _ax6 = _axes6_flat[_gi]
                _plot = _df_g[_df_g.gap_size == _gsize]
                if _plot.empty:
                    _ax6.set_visible(False)
                    continue
                sns.boxplot(data=_plot, x='method', y='r2',
                            palette=[GROUP_COLORS[_grp]] * len(_methods),
                            ax=_ax6, flierprops={'markersize': 2})
                _ax6.axhline(R2_THRESHOLD, color='red', linestyle='--',
                             linewidth=1, alpha=0.7)
                _ax6.set_title(f'gap = {_human_size(_gsize, _scale)}', fontsize=9)
                _ax6.set_xlabel('')
                _ax6.set_ylabel('R²' if _gi % _ncols == 0 else '')
                plt.setp(_ax6.get_xticklabels(), rotation=30, ha='right', fontsize=7)

            for _ai in range(len(_valid_gs), len(_axes6_flat)):
                _axes6_flat[_ai].set_visible(False)

            _fig6.suptitle(
                f'Group {_grp} — R² Distribution ({_scale.capitalize()})',
                fontweight='bold')
            _fig6.tight_layout()
            _fp6 = str(FIGS_OUT / _scale / f'fig_boxplots_group_{_grp}.png')
            _fig6.savefig(_fp6, dpi=300)
            plt.close(_fig6)
            print(f'  ✓ fig_boxplots_group_{_grp}.png  [{_scale}]')

    # ── Phase 6 Fig 7: Hourly vs daily comparison ────────────────────────────────
    _h_best = df_agg[df_agg.time_scale == 'hourly'].groupby('gap_size')['r2_mean'].max()
    _d_best = df_agg[df_agg.time_scale == 'daily'].groupby('gap_size')['r2_mean'].max()

    if not _h_best.empty and not _d_best.empty:
        _fig7, _ax7 = plt.subplots(figsize=(11, 6))
        _ax7.plot(_h_best.index.values / 24, _h_best.values,
                  color='#2196F3', linewidth=2.5, marker='o', markersize=4,
                  label='Hourly (x-axis: equivalent days)')
        _ax7.plot(_d_best.index.values.astype(float), _d_best.values,
                  color='#4CAF50', linewidth=2.5, marker='s', markersize=4,
                  label='Daily')
        _ax7.axhline(R2_THRESHOLD, color='red', linestyle='--',
                     linewidth=1.2, alpha=0.7, label=f'R²={R2_THRESHOLD}')
        _ax7.set_xscale('log')
        _ax7.set_xlabel('Gap size (days, log scale)')
        _ax7.set_ylabel('R² (best method)')
        _ax7.set_title('Hourly vs Daily Scale: Gap-Filling Performance',
                       fontweight='bold')
        _ax7.legend()
        _fig7.tight_layout()
        _fig7.savefig(str(FIGS_OUT / 'fig_hourly_vs_daily_comparison.png'), dpi=300)
        plt.close(_fig7)
        print('✓ fig_hourly_vs_daily_comparison.png')

    # ── Phase 6: Print final thresholds and figure inventory ─────────────────────
    threshold_hourly = find_reliability_threshold(df_agg, 'hourly')
    threshold_daily  = find_reliability_threshold(df_agg, 'daily')

    print('=' * 60)
    print('  FINAL RELIABILITY THRESHOLDS')
    print('=' * 60)
    if threshold_hourly:
        print(f'  Hourly: do NOT fill gaps > {threshold_hourly}h '
              f'({threshold_hourly/24:.1f} days)')
    else:
        print('  Hourly: all tested gap sizes are reliably fillable (R² ≥ 0.7)')
    if threshold_daily:
        print(f'  Daily:  do NOT fill gaps > {threshold_daily} days')
    else:
        print('  Daily:  all tested gap sizes are reliably fillable (R² ≥ 0.7)')

    # Save recommendations — include per-bin method recommendations (M10 fix)
    _rec_rows = []
    for _scale, _thr in [('hourly', threshold_hourly), ('daily', threshold_daily)]:
        _dm = _decision_records.get(_scale, [])
        for _row in _dm:
            _rec_rows.append({
                'scale': _scale,
                'gap_size_range': _row['Gap Size Range'],
                'best_method': _row['Best Method'],
                'r2_mean': _row['R² mean'],
                'rmse_mean': _row['RMSE mean'],
                'reliability': _row['Reliability'],
                'reliability_threshold': _thr,
                'r2_cutoff': R2_THRESHOLD,
                'nse_cutoff': NSE_THRESHOLD,
            })
        if not _dm:
            _rec_rows.append({
                'scale': _scale, 'gap_size_range': 'N/A',
                'best_method': 'N/A', 'r2_mean': 'N/A', 'rmse_mean': 'N/A',
                'reliability': 'N/A',
                'reliability_threshold': _thr,
                'r2_cutoff': R2_THRESHOLD, 'nse_cutoff': NSE_THRESHOLD,
            })
    pd.DataFrame(_rec_rows).to_csv(STATS_OUT / 'recommendations_summary.csv', index=False)

    print('\nOutputs saved to:')
    print(f'  Statistics: {STATS_OUT}')
    print(f'  Figures:    {FIGS_OUT}')

    _figs = sorted(FIGS_OUT.rglob('*.png'))
    print(f'\n{len(_figs)} figures generated:')
    for _fp in _figs:
        print(f'  {_fp.relative_to(FIGS_OUT)}')

    # ## Summary of Findings and Recommendations
    #
    # ### Reliability Thresholds
    #
    # The **reliability threshold** is the smallest gap size at which even the
    # best-performing method's mean R² drops below **0.7** — beyond this point,
    # gap-filling introduces more uncertainty than leaving values as NaN.
    #
    # | Scale | Threshold | Interpretation |
    # |-------|-----------|----------------|
    # | Hourly | `threshold_hourly` (see printed output) | Maximum fillable gap |
    # | Daily  | `threshold_daily`  (see printed output) | Maximum fillable gap |
    #
    # ### Method Performance Hierarchy
    #
    # | Gap Size | Hourly | Daily |
    # |----------|--------|-------|
    # | Very short (1–3h / 1–2d) | Linear interpolation | Linear |
    # | Short (3–12h / 2–5d) | Cubic / Akima | MDV |
    # | Medium (12h–3d / 5–14d) | MDV / STL | Rolling mean or RF |
    # | Long (3–14d / 14–30d) | XGBoost / RF | XGBoost / RF |
    # | Very long (> threshold) | **Do not fill** | **Do not fill** |
    #
    # ### Key Insights
    #
    # 1. **Classical interpolation** excels for short gaps because sap flow has a
    #    strong diurnal cycle — adjacent values tightly constrain the gap.
    # 2. **MDV** captures the diurnal pattern for medium gaps by averaging the same
    #    hour-of-day across ±7 surrounding days.
    # 3. **ML methods** (XGBoost, RF) leverage lagged values and temporal covariates
    #    to reconstruct longer gaps but require sufficient per-site training data.
    # 4. **DL methods** (CNN-LSTM, Transformer) show marginal improvements for very
    #    long gaps but are sensitive to small per-site training datasets.
    # 5. **Daily aggregation** reduces apparent gap counts (small hourly gaps merge
    #    or disappear), improving daily-scale R² for equivalent gap durations.
    # 6. Beyond the reliability threshold, gap-filling degrades model input quality.
    #
    # ### Recommendation for `merge_gap_filled_hourly_orginal.py`
    #
    # Replace the current `max_gap=2` linear strategy with a tiered approach:
    #

    return threshold_hourly, threshold_daily


def main():
    """Run all 6 phases of the gap-filling benchmark."""
    print(f'Root:         {ROOT}')
    print(f'SAP files:    {len(list(SAP_DIR.glob("*.csv")))}')
    print(f'Stats output: {STATS_OUT}')
    print(f'Figs output:  {FIGS_OUT}')
    print(f'Methods:      {len(METHODS)}')

    t_start = time.time()

    # Phase 1: Gap Census
    df_gaps_h, df_gaps_d, df_site_summary = run_phase1()

    # Phase 2: Gap Size Distribution
    df_gaps_h, df_gaps_d = run_phase2(df_gaps_h, df_gaps_d)

    # Phase 3: Synthetic Experiment Setup
    ground_truth_store = run_phase3(df_site_summary)

    # Phase 4: Method definitions (already at module level)
    print(f'\nPhase 4: {len(METHODS)} gap-filling methods available')

    # Phase 5: Benchmark Evaluation
    df_results, df_agg = run_phase5(ground_truth_store)

    # Phase 6: Visualisation & Recommendations
    threshold_hourly, threshold_daily = run_phase6(df_results, df_agg)

    elapsed = time.time() - t_start
    print(f'\nTotal runtime: {elapsed/60:.1f} minutes')
    print('Done.')


if __name__ == '__main__':
    main()
