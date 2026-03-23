"""Tests for windowed model support (AC-9)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from src.aoa import core
from src.aoa.apply import (
    compute_di_for_dataframe,
    create_windowed_features,
    load_model_config,
    parse_windowed_feature_names,
    process_files,
)
from src.aoa.config import AOAConfig
from src.aoa.prepare import build_aoa_reference, load_aoa_reference

# --- Fixtures ---

DYNAMIC_BASE = ["ta", "vpd", "sw_in"]
STATIC_FEATS = ["elevation", "ENF", "EBF"]
INPUT_WIDTH = 2  # lags: t-0, t-1
WINDOWED_NAMES = [f"{f}_t-0" for f in DYNAMIC_BASE] + [f"{f}_t-1" for f in DYNAMIC_BASE] + STATIC_FEATS
N_WINDOWED = len(WINDOWED_NAMES)  # 3*2 + 3 = 9


@pytest.fixture
def windowed_training_data():
    """Training data in windowed feature space (N=40, P=9)."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((40, N_WINDOWED))
    # Make PFT-like columns binary
    for i in range(40):
        X[i, -3:] = 0.0
        X[i, -3 + rng.integers(0, 3)] = 1.0
    return X


@pytest.fixture
def windowed_fold_labels():
    rng = np.random.default_rng(42)
    labels = np.tile(np.arange(3), 14)[:40]
    rng.shuffle(labels)
    return labels


@pytest.fixture
def windowed_shap_weights():
    rng = np.random.default_rng(99)
    return np.abs(rng.standard_normal(N_WINDOWED)) + 0.01


@pytest.fixture
def windowed_reference(windowed_training_data, windowed_fold_labels, windowed_shap_weights, tmp_path):
    """Build reference NPZ with windowed feature names."""
    path = build_aoa_reference(
        X_train=windowed_training_data,
        fold_labels=windowed_fold_labels,
        shap_importances=windowed_shap_weights,
        feature_names=WINDOWED_NAMES,
        output_dir=tmp_path,
        run_id="windowed_test",
        d_bar_method="full",
    )
    ref = load_aoa_reference(path)
    tree = core.build_kdtree(ref["reference_cloud_weighted"])
    return ref, tree, path


@pytest.fixture
def raw_era5_windowed_df():
    """Simulate raw ERA5 file with base feature columns (no _t-N suffixes).

    3 pixels, 5 timestamps each = 15 rows.
    """
    rng = np.random.default_rng(123)
    rows = []
    for lat, lon in [(50.0, 10.0), (50.1, 10.0), (50.0, 10.1)]:
        for day in range(5):
            row = {
                "latitude": lat,
                "longitude": lon,
                "timestamp": f"2015-01-{day + 1:02d}",
            }
            for feat in DYNAMIC_BASE:
                row[feat] = rng.standard_normal()
            for feat in STATIC_FEATS:
                row[feat] = 0.0
            row[STATIC_FEATS[rng.integers(0, len(STATIC_FEATS))]] = 1.0
            rows.append(row)
    return pd.DataFrame(rows)


# --- Tests: parse_windowed_feature_names ---


class TestParseWindowedFeatureNames:
    def test_separates_dynamic_static(self):
        dynamic, static = parse_windowed_feature_names(WINDOWED_NAMES)
        assert set(dynamic.keys()) == set(DYNAMIC_BASE)
        assert set(static) == set(STATIC_FEATS)

    def test_dynamic_lags_correct(self):
        dynamic, _ = parse_windowed_feature_names(WINDOWED_NAMES)
        for base in DYNAMIC_BASE:
            assert sorted(dynamic[base]) == [0, 1]

    def test_no_windowed_features(self):
        dynamic, static = parse_windowed_feature_names(["elevation", "LAI", "ENF"])
        assert dynamic == {}
        assert static == ["elevation", "LAI", "ENF"]

    def test_all_windowed(self):
        names = ["ta_t-0", "ta_t-1", "ta_t-2"]
        dynamic, static = parse_windowed_feature_names(names)
        assert list(dynamic.keys()) == ["ta"]
        assert sorted(dynamic["ta"]) == [0, 1, 2]
        assert static == []

    def test_preserves_order(self):
        names = ["vpd_t-0", "ta_t-0", "elevation", "vpd_t-1"]
        dynamic, static = parse_windowed_feature_names(names)
        assert static == ["elevation"]
        # dynamic keys insertion order
        assert list(dynamic.keys()) == ["vpd", "ta"]


# --- Tests: create_windowed_features ---


class TestCreateWindowedFeatures:
    def test_creates_lag_columns(self, raw_era5_windowed_df):
        dynamic = {"ta": [0, 1], "vpd": [0, 1], "sw_in": [0, 1]}
        static = ["elevation", "ENF", "EBF"]
        result = create_windowed_features(raw_era5_windowed_df, dynamic, static)

        # All windowed + static + meta columns present
        for name in WINDOWED_NAMES:
            assert name in result.columns, f"Missing column: {name}"
        assert "latitude" in result.columns
        assert "longitude" in result.columns
        assert "timestamp" in result.columns

    def test_t0_equals_raw(self, raw_era5_windowed_df):
        """t-0 columns should equal the raw base column values."""
        dynamic = {"ta": [0]}
        static = []
        result = create_windowed_features(raw_era5_windowed_df, dynamic, static)
        # After sorting, t-0 should match the base column
        df_sorted = raw_era5_windowed_df.sort_values(["latitude", "longitude", "timestamp"]).reset_index(drop=True)
        assert_allclose(result["ta_t-0"].values, df_sorted["ta"].values)

    def test_t1_is_shifted(self, raw_era5_windowed_df):
        """t-1 should be the previous timestamp value within each pixel."""
        dynamic = {"ta": [0, 1]}
        static = []
        result = create_windowed_features(raw_era5_windowed_df, dynamic, static)
        # Group by pixel, check shift
        for (lat, lon), grp in result.groupby(["latitude", "longitude"]):
            grp = grp.sort_values("timestamp")
            # First row of each pixel should have NaN for t-1
            assert np.isnan(grp["ta_t-1"].iloc[0]), f"First row at ({lat},{lon}) should be NaN"
            # Subsequent rows: t-1 should equal previous t-0
            for i in range(1, len(grp)):
                assert_allclose(
                    grp["ta_t-1"].iloc[i],
                    grp["ta_t-0"].iloc[i - 1],
                    err_msg=f"Lag mismatch at row {i} for pixel ({lat},{lon})",
                )

    def test_static_features_unchanged(self, raw_era5_windowed_df):
        dynamic = {"ta": [0, 1]}
        static = ["elevation"]
        result = create_windowed_features(raw_era5_windowed_df, dynamic, static)
        df_sorted = raw_era5_windowed_df.sort_values(["latitude", "longitude", "timestamp"]).reset_index(drop=True)
        assert_allclose(result["elevation"].values, df_sorted["elevation"].values)

    def test_missing_base_feature_raises(self, raw_era5_windowed_df):
        dynamic = {"nonexistent_feat": [0, 1]}
        static = []
        with pytest.raises(ValueError, match="Base feature"):
            create_windowed_features(raw_era5_windowed_df, dynamic, static)

    def test_row_count_preserved(self, raw_era5_windowed_df):
        """Windowing doesn't drop rows — NaN rows kept for downstream handling."""
        dynamic = {"ta": [0, 1]}
        static = []
        result = create_windowed_features(raw_era5_windowed_df, dynamic, static)
        assert len(result) == len(raw_era5_windowed_df)


# --- Tests: windowed model E2E ---


class TestWindowedModelE2E:
    def test_windowed_model_config(self, tmp_path):
        """load_model_config correctly reads IS_WINDOWING=True."""
        config = {
            "model_type": "xgb",
            "run_id": "windowed_test",
            "feature_names": WINDOWED_NAMES,
            "data_info": {"IS_WINDOWING": True, "input_width": 2, "shift": 1},
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))
        mc = load_model_config(path)
        assert mc["is_windowing"] is True
        assert mc["input_width"] == 2

    def test_di_on_windowed_dataframe(self, windowed_reference, windowed_training_data):
        """compute_di_for_dataframe works with windowed feature columns."""
        ref, tree, _ = windowed_reference
        # Create a DataFrame with windowed column names
        df = pd.DataFrame(windowed_training_data[:10], columns=WINDOWED_NAMES)
        di, aoa_mask, valid_mask = compute_di_for_dataframe(df, ref, tree, WINDOWED_NAMES)
        assert len(di) == 10
        assert np.all(di >= 0)

    def test_process_files_with_windowing(self, windowed_reference, raw_era5_windowed_df, tmp_path):
        """process_files correctly applies windowing to raw ERA5 data."""
        ref, _, ref_path = windowed_reference

        # Write mock ERA5 CSV with raw (un-windowed) column names
        era5_dir = tmp_path / "era5" / "2015_daily"
        era5_dir.mkdir(parents=True)
        raw_era5_windowed_df.to_csv(era5_dir / "prediction_2015_01_daily.csv", index=False)

        model_config = {
            "feature_names": WINDOWED_NAMES,
            "is_windowing": True,
        }
        config = AOAConfig(
            model_type="xgb",
            run_id="windowed_test",
            time_scale="daily",
            aoa_reference_path=ref_path,
            input_dir=tmp_path / "era5",
            model_config_path=Path("/tmp/config.json"),
            output_dir=tmp_path / "output",
            years=(2015,),
            months=(1,),
            save_per_timestamp=True,
        )
        process_files(config, ref, model_config)

        # Monthly summary should exist
        monthly = tmp_path / "output" / "daily" / "monthly" / "di_monthly_2015_01.parquet"
        assert monthly.exists()
        mdf = pd.read_parquet(monthly)
        assert len(mdf) > 0
        assert "median_DI" in mdf.columns

        # Per-timestamp output
        ts = tmp_path / "output" / "daily" / "per_timestamp" / "di_2015_01_daily.parquet"
        assert ts.exists()
        tdf = pd.read_parquet(ts)
        # 3 pixels × 5 timestamps, minus first row per pixel (lag NaN) = 3×4 = 12
        assert len(tdf) == 12

    def test_non_windowed_model_unchanged(self, windowed_training_data, windowed_reference, tmp_path):
        """When is_windowing=False, process_files doesn't modify data."""
        ref, _, ref_path = windowed_reference

        # Create ERA5 CSV that already has windowed column names
        era5_dir = tmp_path / "era5" / "2015_daily"
        era5_dir.mkdir(parents=True)
        df = pd.DataFrame(windowed_training_data[:6], columns=WINDOWED_NAMES)
        df["latitude"] = [50.0] * 3 + [50.1] * 3
        df["longitude"] = [10.0] * 6
        df["timestamp"] = pd.date_range("2015-01-01", periods=6)
        df.to_csv(era5_dir / "prediction_2015_01_daily.csv", index=False)

        model_config = {
            "feature_names": WINDOWED_NAMES,
            "is_windowing": False,  # Not windowed
        }
        config = AOAConfig(
            model_type="xgb",
            run_id="test",
            time_scale="daily",
            aoa_reference_path=ref_path,
            input_dir=tmp_path / "era5",
            model_config_path=Path("/tmp/config.json"),
            output_dir=tmp_path / "output",
            years=(2015,),
            months=(1,),
            save_per_timestamp=True,
        )
        process_files(config, ref, model_config)

        ts = tmp_path / "output" / "daily" / "per_timestamp" / "di_2015_01_daily.parquet"
        tdf = pd.read_parquet(ts)
        # All 6 rows should be valid (no windowing NaN)
        assert len(tdf) == 6
