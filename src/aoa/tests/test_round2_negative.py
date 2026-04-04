"""Round 2: Negative paths & error handling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.aoa import core
from src.aoa.apply import (
    compute_di_for_dataframe,
    create_windowed_features,
    load_model_config,
    process_files,
    select_and_validate_features,
)
from src.aoa.config import AOAConfig
from src.aoa.prepare import build_aoa_reference, load_aoa_reference
from src.aoa.tests.conftest import FEATURE_NAMES, N_FEATURES


class TestInvalidModelConfig:
    """Negative paths for model config loading."""

    def test_missing_config_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model_config(tmp_path / "nonexistent.json")

    def test_malformed_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{invalid json")
        with pytest.raises(json.JSONDecodeError):
            load_model_config(path)

    def test_missing_feature_names_key(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"model_type": "xgb", "run_id": "test"}))
        with pytest.raises(KeyError):
            load_model_config(path)


class TestInvalidReference:
    """Negative paths for reference loading."""

    def test_load_nonexistent_npz(self):
        with pytest.raises(FileNotFoundError):
            load_aoa_reference(Path("/nonexistent/ref.npz"))

    def test_load_truncated_npz(self, tmp_path):
        """Corrupted/truncated file should raise."""
        bad = tmp_path / "bad.npz"
        bad.write_bytes(b"PK\x03\x04corrupted data")
        with pytest.raises(Exception):
            load_aoa_reference(bad)

    def test_load_npz_missing_keys(self, tmp_path):
        """NPZ with missing required keys should raise ValueError."""
        np.savez(tmp_path / "incomplete.npz", d_bar=1.0)
        with pytest.raises(ValueError, match="missing keys"):
            load_aoa_reference(tmp_path / "incomplete.npz")


class TestFeatureValidationErrors:
    """Negative paths for feature selection and validation."""

    def test_completely_wrong_features(self):
        """Features that don't overlap at all."""
        df = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
        with pytest.raises(ValueError, match="Feature mismatch"):
            select_and_validate_features(df, ["x", "y", "z"], ["a", "b", "c"])

    def test_partial_feature_overlap(self):
        """Some features match, some don't."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        with pytest.raises(ValueError, match="Feature mismatch"):
            select_and_validate_features(df, ["a", "b", "c"], ["a", "b", "d"])


class TestWindowedNegativePaths:
    """Negative paths for windowed feature creation."""

    def test_missing_timestamp_column(self):
        """DataFrame without timestamp should fail during sort."""
        df = pd.DataFrame(
            {
                "latitude": [50.0],
                "longitude": [10.0],
                "ta": [1.0],
            }
        )
        with pytest.raises(KeyError):
            create_windowed_features(df, {"ta": [0, 1]}, [])

    def test_missing_lat_lon_columns(self):
        """DataFrame without lat/lon should fail during groupby."""
        df = pd.DataFrame(
            {
                "timestamp": ["2015-01-01"],
                "ta": [1.0],
            }
        )
        with pytest.raises(KeyError):
            create_windowed_features(df, {"ta": [0, 1]}, [])

    def test_windowed_process_with_missing_base_feature(self, tmp_path):
        """process_files with windowed model where ERA5 lacks a base feature."""
        rng = np.random.default_rng(42)
        # Build reference with windowed features
        names = ["ta_t-0", "ta_t-1", "elev"]
        X = rng.standard_normal((20, 3))
        folds = np.tile([0, 1], 10)
        shap = np.array([0.3, 0.3, 0.4])
        ref_path = build_aoa_reference(X, folds, shap, names, tmp_path, "test")
        ref = load_aoa_reference(ref_path)

        # ERA5 file missing 'ta' column
        era5_dir = tmp_path / "era5" / "2015_daily"
        era5_dir.mkdir(parents=True)
        df = pd.DataFrame(
            {
                "latitude": [50.0, 50.0],
                "longitude": [10.0, 10.0],
                "timestamp": ["2015-01-01", "2015-01-02"],
                "vpd": [1.0, 2.0],  # wrong feature
                "elev": [500.0, 500.0],
            }
        )
        df.to_csv(era5_dir / "prediction_2015_01_daily.csv", index=False)

        config = AOAConfig(
            model_type="xgb",
            run_id="test",
            time_scale="daily",
            aoa_reference_path=ref_path,
            input_dir=tmp_path / "era5",
            model_config_path=Path("/tmp/c.json"),
            output_dir=tmp_path / "out",
            years=(2015,),
            months=(1,),
        )
        model_config = {"feature_names": names, "is_windowing": True}
        # Should not crash — process_files catches exceptions per file
        process_files(config, ref, model_config)
        # No output should be written for the failed file
        monthly = tmp_path / "out" / "daily" / "monthly"
        assert not list(monthly.glob("*.parquet"))


class TestBuildReferenceNegative:
    """Negative paths for build_aoa_reference."""

    def test_nan_in_shap(self, tmp_path):
        X = np.ones((10, 2))
        folds = np.tile([0, 1], 5)
        shap = np.array([0.5, np.nan])
        with pytest.raises(ValueError, match="NaN"):
            build_aoa_reference(X, folds, shap, ["a", "b"], tmp_path, "test")

    def test_wrong_shap_length(self, tmp_path):
        X = np.ones((10, 2))
        folds = np.tile([0, 1], 5)
        shap = np.array([0.5, 0.3, 0.2])  # 3 != 2
        with pytest.raises(ValueError, match="shap_importances length"):
            build_aoa_reference(X, folds, shap, ["a", "b"], tmp_path, "test")

    def test_wrong_fold_length(self, tmp_path):
        X = np.ones((10, 2))
        folds = np.array([0, 1, 2])  # 3 != 10
        shap = np.array([0.5, 0.3])
        with pytest.raises(ValueError, match="fold_labels length"):
            build_aoa_reference(X, folds, shap, ["a", "b"], tmp_path, "test")


class TestAllowPickleSecurityNegative:
    """Verify pickle deserialization is blocked."""

    def test_load_aoa_reference_rejects_pickle(self, tmp_path):
        """NPZ with object arrays should fail with allow_pickle=False."""
        # Create an NPZ with a pickle-requiring object array
        obj_arr = np.array([{"malicious": "payload"}], dtype=object)
        path = tmp_path / "bad_reference.npz"
        np.savez(path, evil=obj_arr)
        with pytest.raises(ValueError):
            load_aoa_reference(path)

    def test_load_aoa_reference_rejects_missing_keys(self, tmp_path):
        """NPZ missing required keys should raise ValueError."""
        path = tmp_path / "incomplete.npz"
        np.savez(path, some_random_key=np.array([1, 2, 3]))
        with pytest.raises(ValueError, match="missing keys"):
            load_aoa_reference(path)


class TestNJobsNegative:
    """Negative tests for n_jobs parameter threading."""

    def test_compute_di_all_nan_with_n_jobs(self):
        """All-NaN input with n_jobs should return empty, not crash."""
        ref = {
            "feature_names": ["a", "b"],
            "feature_means": np.array([0.0, 0.0]),
            "feature_stds": np.array([1.0, 1.0]),
            "feature_weights": np.array([1.0, 1.0]),
            "d_bar": 1.0,
            "threshold": 1.0,
            "reference_cloud_weighted": np.array([[0.0, 0.0], [1.0, 1.0]]),
        }
        tree = core.build_kdtree(ref["reference_cloud_weighted"])
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
        di, mask, valid = compute_di_for_dataframe(df, ref, tree, ["a", "b"], n_jobs=2)
        assert len(di) == 0
        assert len(mask) == 0
