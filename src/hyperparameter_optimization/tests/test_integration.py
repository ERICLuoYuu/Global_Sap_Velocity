"""Round 3: Integration & cross-module regression tests."""

import json
import pickle

import joblib
import numpy as np
import pytest

# run_shap_analysis_ml imports cartopy at module level — skip if unavailable
cartopy = pytest.importorskip("cartopy", reason="cartopy not installed")

from src.hyperparameter_optimization.run_shap_analysis_ml import load_artifacts  # noqa: E402
from src.hyperparameter_optimization.target_transformer import TargetTransformer  # noqa: E402


class TestLoadArtifacts:
    """Integration test: load_artifacts with synthetic model artifacts."""

    @pytest.fixture
    def synthetic_artifacts(self, tmp_path):
        """Create a minimal set of model artifacts on disk."""
        run_id = "test_run"
        model_type = "xgb"
        model_dir = tmp_path / "models"
        model_dir.mkdir()

        # Synthetic XGBoost-like model (just a dict for testing load)
        model = {"type": "xgb", "n_features": 3}
        joblib.dump(model, model_dir / f"FINAL_{model_type}_{run_id}.joblib")

        # Transformer
        transformer = TargetTransformer(method="log1p")
        transformer.fit(np.array([1.0, 2.0, 3.0]))

        # Config
        config = {
            "model_type": model_type,
            "run_id": run_id,
            "preprocessing": {
                "target_transform": "log1p",
                "target_transform_params": transformer.get_params(),
                "feature_scaling": "StandardScaler",
            },
            "data_info": {
                "n_samples": 100,
                "n_features": 3,
                "IS_WINDOWING": False,
                "input_width": None,
            },
            "feature_names": ["ta", "vpd", "sw_in"],
            "shap_feature_names": ["ta", "vpd", "sw_in"],
            "time_scale": "daily",
        }
        with open(model_dir / f"FINAL_config_{run_id}.json", "w") as f:
            json.dump(config, f)

        # SHAP context bundle
        rng = np.random.RandomState(42)
        np.savez_compressed(
            model_dir / f"SHAP_context_{run_id}.npz",
            X_all_scaled=rng.randn(100, 3),
            X_all_records=rng.randn(100, 3),
            y_all_records=rng.exponential(5.0, 100),
            site_ids=np.array(["site_a"] * 50 + ["site_b"] * 50),
            timestamps=np.arange(100),
            pfts=np.array(["ENF"] * 100),
        )

        # Site info
        with open(model_dir / f"site_info_{run_id}.json", "w") as f:
            json.dump({"site_a": {"lat": 51.0, "lon": 7.0}}, f)

        # Scaler
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(rng.randn(100, 3))
        with open(model_dir / f"FINAL_scaler_{run_id}_feature.pkl", "wb") as f:
            pickle.dump(scaler, f)

        return model_dir, run_id, model_type

    def test_loads_all_artifacts(self, synthetic_artifacts):
        model_dir, run_id, model_type = synthetic_artifacts
        model, config, context, site_info, scaler, transformer = load_artifacts(model_dir, run_id, model_type)
        assert model["type"] == "xgb"
        assert config["model_type"] == "xgb"
        assert context["X_all_scaled"].shape == (100, 3)
        assert "site_a" in site_info
        assert scaler is not None
        assert transformer is not None
        assert transformer.method == "log1p"

    def test_config_keys_match_training_output(self, synthetic_artifacts):
        """Verify config keys used by run_shap_analysis are present."""
        model_dir, run_id, model_type = synthetic_artifacts
        _, config, _, _, _, _ = load_artifacts(model_dir, run_id, model_type)

        # These are the keys run_shap_analysis reads
        data_info = config.get("data_info", {})
        assert "IS_WINDOWING" in data_info
        assert "input_width" in data_info
        assert "feature_names" in config
        assert "shap_feature_names" in config
        assert "time_scale" in config

    def test_loads_without_optional_files(self, synthetic_artifacts):
        """Missing site_info and scaler should not raise."""
        model_dir, run_id, model_type = synthetic_artifacts
        # Remove optional files
        (model_dir / f"site_info_{run_id}.json").unlink()
        (model_dir / f"FINAL_scaler_{run_id}_feature.pkl").unlink()

        model, config, context, site_info, scaler, transformer = load_artifacts(model_dir, run_id, model_type)
        assert site_info == {}
        assert scaler is None

    def test_missing_model_raises(self, synthetic_artifacts):
        """Missing model file should raise FileNotFoundError."""
        model_dir, run_id, model_type = synthetic_artifacts
        (model_dir / f"FINAL_{model_type}_{run_id}.joblib").unlink()
        with pytest.raises(FileNotFoundError):
            load_artifacts(model_dir, run_id, model_type)

    def test_transformer_none_when_no_transform(self, synthetic_artifacts):
        """Config with target_transform=none should yield transformer=None."""
        model_dir, run_id, model_type = synthetic_artifacts
        config_path = model_dir / f"FINAL_config_{run_id}.json"
        with open(config_path) as f:
            config = json.load(f)
        config["preprocessing"]["target_transform"] = "none"
        config["preprocessing"]["target_transform_params"] = None
        with open(config_path, "w") as f:
            json.dump(config, f)

        _, _, _, _, _, transformer = load_artifacts(model_dir, run_id, model_type)
        assert transformer is None


class TestCrossModuleImports:
    """Verify no circular imports and all modules are importable."""

    def test_import_target_transformer(self):
        from src.hyperparameter_optimization.target_transformer import TargetTransformer

        assert TargetTransformer.VALID_METHODS is not None

    def test_import_shap_constants(self):
        from src.hyperparameter_optimization.shap_constants import (
            PFT_COLUMNS,
            get_feature_unit,
        )

        assert len(PFT_COLUMNS) == 8
        assert callable(get_feature_unit)

    def test_import_shap_plotting(self):
        from src.hyperparameter_optimization.shap_plotting import (
            aggregate_pft_shap_values,
            generate_windowed_feature_names,
        )

        assert callable(aggregate_pft_shap_values)
        assert callable(generate_windowed_feature_names)

    def test_import_feature_engineering(self):
        from src.hyperparameter_optimization.feature_engineering import (
            add_sap_flow_features,
            apply_feature_engineering,
        )

        assert callable(add_sap_flow_features)
        assert callable(apply_feature_engineering)

    def test_import_training_utils(self):
        from src.hyperparameter_optimization.training_utils import (
            add_time_features,
            convert_windows_to_numpy,
            create_spatial_groups,
        )

        assert callable(add_time_features)
        assert callable(convert_windows_to_numpy)
        assert callable(create_spatial_groups)
