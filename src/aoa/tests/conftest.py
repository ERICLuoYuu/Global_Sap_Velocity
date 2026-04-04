"""Shared fixtures and pytest configuration for AOA tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# --- Constants ---
N_TRAIN = 50
N_NEW = 20
N_FEATURES_CONT = 5
N_FEATURES_PFT = 3
N_FEATURES = N_FEATURES_CONT + N_FEATURES_PFT  # 8
N_FOLDS = 3
SEED = 42
FEATURE_NAMES_CONT = ["ta", "vpd", "sw_in", "elevation", "LAI"]
FEATURE_NAMES_PFT = ["ENF", "EBF", "DBF"]
FEATURE_NAMES = FEATURE_NAMES_CONT + FEATURE_NAMES_PFT


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (> 10 seconds)")
    config.addinivalue_line("markers", "performance: HPC-only scalability tests")
    config.addinivalue_line("markers", "requires_rasterio: requires rasterio installed")


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def synthetic_X_train(rng):
    """50x8 training matrix: 5 continuous + 3 one-hot PFT."""
    X_cont = rng.standard_normal((N_TRAIN, N_FEATURES_CONT))
    X_pft = np.zeros((N_TRAIN, N_FEATURES_PFT))
    for i in range(N_TRAIN):
        X_pft[i, rng.integers(0, N_FEATURES_PFT)] = 1.0
    return np.hstack([X_cont, X_pft])


@pytest.fixture
def synthetic_fold_labels(rng):
    """3-fold assignments for 50 points (balanced)."""
    labels = np.tile(np.arange(N_FOLDS), N_TRAIN // N_FOLDS + 1)[:N_TRAIN]
    rng.shuffle(labels)
    return labels


@pytest.fixture
def synthetic_shap_weights(rng):
    """Raw |SHAP| importance for 8 features."""
    return np.abs(rng.standard_normal(N_FEATURES)) + 0.01


@pytest.fixture
def synthetic_X_new(rng):
    """20x8 prediction matrix."""
    X_cont = rng.standard_normal((N_NEW, N_FEATURES_CONT))
    X_pft = np.zeros((N_NEW, N_FEATURES_PFT))
    for i in range(N_NEW):
        X_pft[i, rng.integers(0, N_FEATURES_PFT)] = 1.0
    return np.hstack([X_cont, X_pft])


@pytest.fixture
def era5_like_df(rng):
    """Mock ERA5 CSV DataFrame with metadata + features + extras."""
    n = 100
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2015-01-01", periods=n, freq="D"),
            "latitude": rng.choice([50.0, 50.1, 50.2], n),
            "longitude": rng.choice([10.0, 10.1], n),
            "name": "test_site",
            "solar_timestamp": pd.date_range("2015-01-01", periods=n, freq="D"),
        }
    )
    for feat in FEATURE_NAMES_CONT:
        df[feat] = rng.standard_normal(n)
    for feat in FEATURE_NAMES_PFT:
        df[feat] = 0.0
    pft_idx = rng.integers(0, N_FEATURES_PFT, n)
    for i in range(n):
        df.loc[i, FEATURE_NAMES_PFT[pft_idx[i]]] = 1.0
    df["annual_mean_temperature"] = 15.0
    df["Year sin"] = np.sin(2 * np.pi * np.arange(n) / 365.25)
    return df
