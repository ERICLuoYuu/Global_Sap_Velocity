"""
Tests for review fixes applied in Phase 4.

Covers:
  C1: safe_joblib_load (path containment before deserialization)
  C2: resolve_and_contain (path traversal prevention)
  H5: DataTransformer overflow guard (_guard_non_finite)
  H7: MAX_CSV_BYTES file size guard in load_preprocessed_data
  M6: ModelConfig allowlist validation for target_transform/feature_scaling
  L2: validate_path_component (alphanumeric enforcement)
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from predict_sap_velocity_sequential import (
    DataTransformer,
    ModelConfig,
    load_preprocessed_data,
    resolve_and_contain,
    safe_joblib_load,
    validate_path_component,
)

# -- C1: safe_joblib_load ----------------------------------------------------


class TestSafeJoblibLoad:
    def test_loads_file_inside_allowed_root(self, tmp_path):
        """Should load a joblib file that is inside the allowed root."""
        model_dir = tmp_path / "models" / "xgb"
        model_dir.mkdir(parents=True)
        model_file = model_dir / "model.joblib"
        joblib.dump({"test": 42}, model_file)

        result = safe_joblib_load(model_file, tmp_path)
        assert result == {"test": 42}

    def test_rejects_file_outside_allowed_root(self, tmp_path):
        """Should raise ValueError for files outside the allowed root."""
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside_file = outside_dir / "evil.joblib"
        joblib.dump({"evil": True}, outside_file)

        allowed_root = tmp_path / "models"
        allowed_root.mkdir()

        with pytest.raises(ValueError, match="outside allowed root"):
            safe_joblib_load(outside_file, allowed_root)

    def test_rejects_symlink_escape(self, tmp_path):
        """Should reject symlinks that escape the allowed root."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        real_file = real_dir / "data.joblib"
        joblib.dump("payload", real_file)

        jail = tmp_path / "jail"
        jail.mkdir()
        link = jail / "escape.joblib"

        try:
            link.symlink_to(real_file)
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

        with pytest.raises(ValueError, match="outside allowed root"):
            safe_joblib_load(link, jail)

    def test_nonexistent_file_raises(self, tmp_path):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            safe_joblib_load(tmp_path / "nope.joblib", tmp_path)


# -- C2: resolve_and_contain -------------------------------------------------


class TestResolveAndContain:
    def test_valid_path_inside_root(self, tmp_path):
        """Should return resolved path when inside the root."""
        subdir = tmp_path / "sub"
        subdir.mkdir()
        result = resolve_and_contain(str(subdir), tmp_path, "test")
        assert result == subdir.resolve()

    def test_rejects_path_outside_root(self, tmp_path):
        """Should raise ValueError for traversal attempts."""
        with pytest.raises(ValueError, match="escapes allowed root"):
            resolve_and_contain("/etc/passwd", tmp_path, "test")

    def test_rejects_dotdot_traversal(self, tmp_path):
        """Should catch ../ traversal."""
        subdir = tmp_path / "sub"
        subdir.mkdir()
        traversal = str(subdir / ".." / ".." / "etc")
        with pytest.raises(ValueError, match="escapes allowed root"):
            resolve_and_contain(traversal, tmp_path, "test")


# -- L2: validate_path_component ---------------------------------------------


class TestValidatePathComponent:
    @pytest.mark.parametrize(
        "value",
        [
            "xgb",
            "random_forest",
            "run-001",
            "model_v2.1",
            "CNN_LSTM",
        ],
    )
    def test_accepts_safe_values(self, value):
        assert validate_path_component(value, "test") == value

    @pytest.mark.parametrize(
        "value",
        [
            "../etc",
            "xgb/../../",
            "run;rm -rf",
            "model type",
            "run\nid",
            "",
        ],
    )
    def test_rejects_unsafe_values(self, value):
        with pytest.raises(ValueError, match="Invalid characters"):
            validate_path_component(value, "test")


# -- H5: DataTransformer overflow guard ---------------------------------------


class TestOverflowGuard:
    def test_expm1_extreme_positive_clipped(self):
        """Values > 20 in transformed space should be clipped, not inf."""
        config = ModelConfig(model_type="xgb", target_transform="log1p")
        dt = DataTransformer(config)
        extreme = np.array([100.0, 500.0, 1000.0])
        result = dt.transform_target(extreme, inverse=True)
        assert np.all(np.isfinite(result)), "All results must be finite"
        assert np.all(result < 1e9)

    def test_expm1_extreme_negative_clipped(self):
        """Values < -20 in transformed space should be clipped."""
        config = ModelConfig(model_type="xgb", target_transform="log1p")
        dt = DataTransformer(config)
        extreme = np.array([-100.0, -500.0])
        result = dt.transform_target(extreme, inverse=True)
        assert np.all(np.isfinite(result))

    def test_exp_overflow_guard(self):
        """exp transform should also be guarded."""
        config = ModelConfig(model_type="xgb", target_transform="log")
        dt = DataTransformer(config)
        extreme = np.array([1000.0])
        result = dt.transform_target(extreme, inverse=True)
        assert np.all(np.isfinite(result))

    def test_normal_values_unaffected(self):
        """Values within safe range should roundtrip correctly."""
        config = ModelConfig(model_type="xgb", target_transform="log1p")
        dt = DataTransformer(config)
        normal = np.array([0.0, 1.0, 10.0, 100.0])
        transformed = dt.transform_target(normal)
        recovered = dt.transform_target(transformed, inverse=True)
        np.testing.assert_allclose(recovered, normal, rtol=1e-6)


# -- H7: MAX_CSV_BYTES guard -------------------------------------------------


class TestMaxCsvBytesGuard:
    def test_small_file_loads(self, tmp_path):
        """Files under the limit should load normally."""
        csv_file = tmp_path / "small.csv"
        df = pd.DataFrame({"ta": [1.0, 2.0], "vpd": [0.5, 0.6]})
        df.to_csv(csv_file, index=False)
        result = load_preprocessed_data(csv_file)
        assert result is not None
        assert len(result) == 2

    def test_nonexistent_file_returns_none(self, tmp_path):
        """Missing file should return None."""
        result = load_preprocessed_data(tmp_path / "nope.csv")
        assert result is None

    def test_empty_csv_returns_none(self, tmp_path):
        """Empty CSV (header only) should return None."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("ta,vpd\n")
        result = load_preprocessed_data(csv_file)
        assert result is None


# -- M6: ModelConfig allowlist validation -------------------------------------


class TestModelConfigAllowlist:
    def test_valid_transform(self):
        config = ModelConfig.from_dict(
            {
                "model_type": "xgb",
                "preprocessing": {"target_transform": "log1p"},
            }
        )
        assert config.target_transform == "log1p"

    def test_valid_scaler(self):
        config = ModelConfig.from_dict(
            {
                "model_type": "xgb",
                "preprocessing": {"feature_scaling": "StandardScaler"},
            }
        )
        assert config.feature_scaling == "StandardScaler"

    def test_none_transform_allowed(self):
        config = ModelConfig.from_dict(
            {
                "model_type": "xgb",
                "preprocessing": {},
            }
        )
        assert config.target_transform is None

    def test_invalid_transform_raises(self):
        with pytest.raises(ValueError, match="Invalid target_transform"):
            ModelConfig.from_dict(
                {
                    "model_type": "xgb",
                    "preprocessing": {"target_transform": "exec(evil)"},
                }
            )

    def test_invalid_scaler_raises(self):
        with pytest.raises(ValueError, match="Invalid feature_scaling"):
            ModelConfig.from_dict(
                {
                    "model_type": "xgb",
                    "preprocessing": {"feature_scaling": "PickleInjection"},
                }
            )

    @pytest.mark.parametrize("transform", ["log1p", "log", "sqrt", None])
    def test_all_allowed_transforms(self, transform):
        config = ModelConfig.from_dict(
            {
                "model_type": "xgb",
                "preprocessing": {"target_transform": transform},
            }
        )
        assert config.target_transform == transform

    @pytest.mark.parametrize(
        "scaler",
        [
            "StandardScaler",
            "MinMaxScaler",
            "RobustScaler",
            "standard",
            "minmax",
            "robust",
            None,
        ],
    )
    def test_all_allowed_scalers(self, scaler):
        config = ModelConfig.from_dict(
            {
                "model_type": "xgb",
                "preprocessing": {"feature_scaling": scaler},
            }
        )
        assert config.feature_scaling == scaler
