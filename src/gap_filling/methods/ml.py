"""Group C/Ce: Machine Learning gap-filling methods.

Extracted from notebooks/gap_benchmark.py. Key change: _current_env_df global
removed; env_df passed explicitly as parameter. fit/predict separated for
the gap filler's two-step pattern.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from src.gap_filling.methods.interpolation import fill_linear

RANDOM_SEED = 42
N_LAGS = 24


def build_ml_features(s: pd.Series, n_lags: int = N_LAGS, env_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build lag + temporal features for ML gap-filling.

    Features: t-1...t-n_lags, hour_of_day, day_of_year, month,
              rolling_mean_24, rolling_std_24, is_daytime.
    If env_df is provided, adds lagged env features for each env column.
    """
    df = pd.DataFrame({"y": s.values}, index=s.index)
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["hour_of_day"] = df.index.hour
    df["day_of_year"] = df.index.dayofyear
    df["month"] = df.index.month
    df["rolling_mean_24"] = df["y"].rolling(24, min_periods=1).mean()
    df["rolling_std_24"] = df["y"].rolling(24, min_periods=1).std().fillna(0)
    df["is_daytime"] = ((df.index.hour >= 6) & (df.index.hour <= 20)).astype(int)
    if env_df is not None:
        env_aligned = env_df.reindex(df.index)
        for ecol in env_aligned.columns:
            for lag in range(1, n_lags + 1):
                df[f"env_{ecol}_lag_{lag}"] = env_aligned[ecol].shift(lag)
    return df.drop(columns=["y"])


# ---------------------------------------------------------------------------
# Fit / predict split (for GapFiller two-step pattern)
# ---------------------------------------------------------------------------


def _fit_ml_on_ground_truth(
    gt_series: pd.Series,
    model_cls,
    model_kwargs: dict,
    env_df: pd.DataFrame | None = None,
):
    """Pre-train an ML model on the full ground truth (no gaps).

    Returns (model, scaler) or None if insufficient data.
    """
    features = build_ml_features(gt_series, env_df=env_df)
    train_mask = gt_series.notna() & features.notna().all(axis=1)
    if train_mask.sum() < 50:
        return None
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(features[train_mask].values)
    model = model_cls(**model_kwargs)
    model.fit(X_train_s, gt_series[train_mask].values)
    return model, scaler


def _predict_ml_at_gaps(s: pd.Series, cached, env_df: pd.DataFrame | None = None) -> pd.Series:
    """Predict at gap positions using a pre-trained ML model.

    If cached is None, falls back to linear interpolation.
    """
    if cached is None:
        return fill_linear(s)
    model, scaler = cached
    features = build_ml_features(s, env_df=env_df)
    test_mask = s.isna() & features.notna().all(axis=1)
    filled = s.copy()
    if test_mask.sum() > 0:
        X_test = scaler.transform(features[test_mask].values)
        filled[test_mask] = np.clip(model.predict(X_test), 0, None)
    if filled.isna().any():
        filled = fill_linear(filled)
    return filled.clip(lower=0)


# ---------------------------------------------------------------------------
# Combined fit+predict (for standalone use and benchmark compatibility)
# ---------------------------------------------------------------------------


def _fit_and_predict_ml(
    s: pd.Series,
    model_cls,
    model_kwargs: dict,
    env_df: pd.DataFrame | None = None,
) -> pd.Series:
    """Generic ML gap-fill: fit on observed data, predict at gap positions."""
    features = build_ml_features(s, env_df=env_df)
    train_mask = s.notna() & features.notna().all(axis=1)
    test_mask = s.isna() & features.notna().all(axis=1)
    if train_mask.sum() < 50:
        return fill_linear(s)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(features[train_mask].values)
    model = model_cls(**model_kwargs)
    model.fit(X_train_s, s[train_mask].values)
    filled = s.copy()
    if test_mask.sum() > 0:
        X_test = scaler.transform(features[test_mask].values)
        filled[test_mask] = np.clip(model.predict(X_test), 0, None)
    if filled.isna().any():
        filled = fill_linear(filled)
    return filled.clip(lower=0)


# ---------------------------------------------------------------------------
# Public API: fit_*/predict_* per model type
# ---------------------------------------------------------------------------

_RF_KWARGS = {"n_estimators": 100, "n_jobs": 1, "random_state": RANDOM_SEED}
_XGB_KWARGS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5,
    "random_state": RANDOM_SEED,
    "verbosity": 0,
    "tree_method": "hist",
}


def fit_rf(s: pd.Series, env_df: pd.DataFrame | None = None):
    """Fit Random Forest on observed data. Returns (model, scaler) or None."""
    return _fit_ml_on_ground_truth(s, RandomForestRegressor, _RF_KWARGS, env_df=env_df)


def predict_rf(s: pd.Series, cached, env_df: pd.DataFrame | None = None) -> pd.Series:
    """Predict gaps using pre-trained RF model."""
    return _predict_ml_at_gaps(s, cached, env_df=env_df)


def fit_xgb(s: pd.Series, env_df: pd.DataFrame | None = None):
    """Fit XGBoost on observed data. Returns (model, scaler) or None."""
    if xgb is None:
        return None
    return _fit_ml_on_ground_truth(s, xgb.XGBRegressor, _XGB_KWARGS, env_df=env_df)


def predict_xgb(s: pd.Series, cached, env_df: pd.DataFrame | None = None) -> pd.Series:
    """Predict gaps using pre-trained XGBoost model."""
    return _predict_ml_at_gaps(s, cached, env_df=env_df)


def fit_knn(s: pd.Series, env_df: pd.DataFrame | None = None, k: int = 10):
    """Fit KNN on observed data. Returns (model, scaler) or None."""
    k_actual = min(k, max(1, int(s.notna().sum() * 0.1)))
    return _fit_ml_on_ground_truth(s, KNeighborsRegressor, {"n_neighbors": k_actual}, env_df=env_df)


def predict_knn(s: pd.Series, cached, env_df: pd.DataFrame | None = None) -> pd.Series:
    """Predict gaps using pre-trained KNN model."""
    return _predict_ml_at_gaps(s, cached, env_df=env_df)


# ---------------------------------------------------------------------------
# Standalone fill functions (benchmark compatibility)
# ---------------------------------------------------------------------------


def fill_rf(s: pd.Series) -> pd.Series:
    """Random Forest gap-filling with 24-lag feature set."""
    return _fit_and_predict_ml(s, RandomForestRegressor, _RF_KWARGS)


def fill_xgb_model(s: pd.Series) -> pd.Series:
    """XGBoost gap-filling with 24-lag feature set."""
    if xgb is None:
        return fill_linear(s)
    return _fit_and_predict_ml(s, xgb.XGBRegressor, _XGB_KWARGS)


def fill_knn(s: pd.Series, k: int = 10) -> pd.Series:
    """KNN gap-filling with 24-lag feature set."""
    k_actual = min(k, max(1, int(s.notna().sum() * 0.1)))
    return _fit_and_predict_ml(s, KNeighborsRegressor, {"n_neighbors": k_actual})
