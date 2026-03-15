"""Shared fixtures for prediction pipeline tests."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_POINTS: List[Dict] = [
    {"lat": -3.0, "lon": -60.0, "sap_xgb": 12.5, "sap_rf": 11.8, "label": "amazon"},
    {"lat": 48.5, "lon": 11.5, "sap_xgb": 4.2, "sap_rf": 4.5, "label": "germany"},
    {"lat": 61.0, "lon": 24.0, "sap_xgb": 2.1, "sap_rf": 2.3, "label": "finland"},
    {"lat": 1.0, "lon": 25.0, "sap_xgb": 18.7, "sap_rf": 17.9, "label": "congo"},
    {"lat": 47.0, "lon": -122.0, "sap_xgb": 6.3, "sap_rf": 6.1, "label": "pacific_nw"},
]

TIMESTAMPS = pd.date_range("2015-07-01", periods=5, freq="D")

RESOLUTION = 0.1
LAT_RANGE = (-60.0, 78.0)
LON_RANGE = (-180.0, 180.0)
EPS = 1e-9

# Feature names matching a realistic XGBoost model config
FEATURE_NAMES = [
    "ta", "vpd", "sw_in", "ppfd_in", "ext_rad", "ws", "precip",
    "volumetric_soil_water_layer_1", "soil_temperature_level_1",
    "LAI", "day_length", "year sin",
    "canopy_height", "elevation",
    "PFT_ENF", "PFT_DBF", "PFT_EBF", "PFT_MF",
]


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def coord_to_pixel(
    lat: float, lon: float,
    lat_range: Tuple[float, float] = LAT_RANGE,
    lon_range: Tuple[float, float] = LON_RANGE,
    resolution: float = RESOLUTION,
) -> Tuple[int, int]:
    """Convert (lat, lon) to (row, col) matching _write_geotiff logic."""
    max_lat = lat_range[1]
    min_lon = lon_range[0]
    row = int(np.floor((max_lat - lat + EPS) / resolution))
    col = int(np.floor((lon - min_lon + EPS) / resolution))
    return row, col


# ---------------------------------------------------------------------------
# Mock model and scalers
# ---------------------------------------------------------------------------

class MockXGBModel:
    """Deterministic mock: predict = sum(features) * 0.01.

    Same input always gives same output — essential for detecting
    row-shuffle bugs.
    """

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sum(X, axis=1) * 0.01


class IdentityScaler:
    """Feature scaler that returns input unchanged."""

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def fit(self, X: np.ndarray) -> "IdentityScaler":
        return self


class Expm1LabelScaler:
    """Label scaler whose inverse_transform applies expm1.

    Matches training with log1p target transform.
    """

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.expm1(X)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random generator."""
    return np.random.default_rng(42)


@pytest.fixture
def known_points() -> List[Dict]:
    """Fixed reference points with deterministic sap velocities."""
    return KNOWN_POINTS


@pytest.fixture
def synthetic_predictions(rng: np.random.Generator) -> pd.DataFrame:
    """~500-row synthetic prediction DataFrame.

    5 known points + 95 random points, each appearing in 5 timestamps.
    All coordinates snapped to 0.1 degree grid.
    Columns: latitude, longitude, timestamp, sap_velocity_xgb, sap_velocity_rf
    """
    rows = []

    # Generate 95 unique random coords (avoid colliding with known points)
    known_set = {(p["lat"], p["lon"]) for p in KNOWN_POINTS}
    random_coords = []
    while len(random_coords) < 95:
        lat = round(rng.uniform(-60, 78), 1)
        lon = round(rng.uniform(-180, 180), 1)
        if (lat, lon) not in known_set and (lat, lon) not in {
            (c[0], c[1]) for c in random_coords
        }:
            random_coords.append((lat, lon))

    for ts in TIMESTAMPS:
        # Known points (deterministic values)
        for pt in KNOWN_POINTS:
            rows.append({
                "latitude": pt["lat"],
                "longitude": pt["lon"],
                "timestamp": ts,
                "sap_velocity_xgb": pt["sap_xgb"],
                "sap_velocity_rf": pt["sap_rf"],
            })
        # Random points
        for lat, lon in random_coords:
            rows.append({
                "latitude": lat,
                "longitude": lon,
                "timestamp": ts,
                "sap_velocity_xgb": round(float(rng.uniform(0.5, 25.0)), 4),
                "sap_velocity_rf": round(float(rng.uniform(0.5, 25.0)), 4),
            })

    df = pd.DataFrame(rows)
    # Ensure timestamp column is string (matching real pipeline output)
    df["timestamp"] = df["timestamp"].astype(str)
    return df


@pytest.fixture
def synthetic_era5_input(rng: np.random.Generator) -> pd.DataFrame:
    """~300-row synthetic ERA5-Land input DataFrame.

    3 timestamps x 100 locations with realistic feature values.
    Has an extra index column at position 0 (matching load_preprocessed_data
    which does df.iloc[:, 1:]).
    """
    timestamps = pd.date_range("2015-07-01", periods=3, freq="D")
    lats = np.round(rng.uniform(-60, 78, 100), 1)
    lons = np.round(rng.uniform(-180, 180, 100), 1)

    rows = []
    for ts in timestamps:
        for lat, lon in zip(lats, lons):
            row = {
                "idx": 0,  # dummy index column (dropped by iloc[:, 1:])
                "latitude": lat,
                "longitude": lon,
                "timestamp": str(ts),
                "ta": float(rng.uniform(260, 310)),
                "vpd": float(rng.uniform(0, 5)),
                "sw_in": float(rng.uniform(0, 400)),
                "ppfd_in": float(rng.uniform(0, 2000)),
                "ext_rad": float(rng.uniform(0, 500)),
                "ws": float(rng.uniform(0, 15)),
                "precip": float(rng.uniform(0, 20)),
                "volumetric_soil_water_layer_1": float(rng.uniform(0.05, 0.5)),
                "soil_temperature_level_1": float(rng.uniform(270, 310)),
                "LAI": float(rng.uniform(0, 8)),
                "day_length": float(rng.uniform(8, 16)),
                "year sin": float(rng.uniform(-1, 1)),
                "canopy_height": float(rng.uniform(2, 40)),
                "elevation": float(rng.uniform(0, 3000)),
                "PFT_ENF": float(rng.choice([0, 1])),
                "PFT_DBF": float(rng.choice([0, 1])),
                "PFT_EBF": float(rng.choice([0, 1])),
                "PFT_MF": float(rng.choice([0, 1])),
            }
            rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def mock_model() -> MockXGBModel:
    """Deterministic mock XGBoost model."""
    return MockXGBModel()


@pytest.fixture
def mock_feature_scaler() -> IdentityScaler:
    """Identity feature scaler."""
    return IdentityScaler()


@pytest.fixture
def mock_label_scaler() -> Expm1LabelScaler:
    """Label scaler with expm1 inverse."""
    return Expm1LabelScaler()


@pytest.fixture
def mock_model_config() -> Dict:
    """Realistic model config dict for XGBoost."""
    from src.make_prediction.predict_sap_velocity_sequential import ModelConfig
    return ModelConfig(
        model_type="xgb",
        run_id="test_run",
        is_windowing=False,
        feature_names=FEATURE_NAMES,
        target_transform="log1p",
        feature_scaling="StandardScaler",
    )


@pytest.fixture
def prediction_csv_dir(
    tmp_path: Path, synthetic_predictions: pd.DataFrame,
) -> Path:
    """Write synthetic predictions as CSV."""
    csv_dir = tmp_path / "pred_csv"
    csv_dir.mkdir()
    csv_path = csv_dir / "prediction_2015_07_predictions_xgb_test.csv"
    synthetic_predictions.to_csv(csv_path, index=False)
    return csv_dir


@pytest.fixture
def prediction_parquet_dir(
    tmp_path: Path, synthetic_predictions: pd.DataFrame,
) -> Path:
    """Write synthetic predictions as Parquet with gzip."""
    pq_dir = tmp_path / "pred_parquet"
    pq_dir.mkdir()
    pq_path = pq_dir / "prediction_2015_07_predictions_xgb_test.parquet"
    synthetic_predictions.to_parquet(pq_path, compression="gzip", engine="pyarrow")
    return pq_dir


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Clean output directory for GeoTIFFs."""
    out = tmp_path / "geotiff_output"
    out.mkdir()
    return out
