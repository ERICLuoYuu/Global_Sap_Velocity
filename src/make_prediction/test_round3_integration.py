"""
Round 3 -- Integration & regression tests.

Tests:
  1. Config import regression (C1 fix -- MODEL_DIR vs MODELS_DIR)
  2. Model loading + config loading integration
  3. Full flat prediction pipeline (load -> prepare -> predict -> map -> validate)
  4. DataTransformer + scaler integration
  5. Regression: hoisted model loading doesn't break per-file output dirs
  6. Config.py env var override (M5 fix)
  7. Window metadata completeness for windowed models
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from predict_sap_velocity_sequential import (
    DataTransformer,
    ModelConfig,
    _extract_year_from_filename,
    get_model_dir,
    load_model,
    load_model_config_for_run,
    load_model_configs,
    load_models,
    load_preprocessed_data,
    load_scalers,
    make_predictions_improved,
    prepare_features_from_preprocessed,
    validate_predictions,
)
from sklearn.preprocessing import StandardScaler

# Module-level MockModel so joblib can pickle it
class MockModel:
    def predict(self, X):
        import numpy as np
        return np.random.rand(len(X)) * 10

# -- Fixtures ----------------------------------------------------------------


@pytest.fixture
def feature_cols():
    return ["ta", "vpd", "sw_in", "ppfd_in", "ws"]


@pytest.fixture
def sample_df(feature_cols):
    np.random.seed(42)
    n = 50
    data = {col: np.random.rand(n) * 10 for col in feature_cols}
    data["timestamp"] = pd.date_range("2020-01-01", periods=n, freq="D")
    data["latitude"] = 51.0
    data["longitude"] = 7.0
    return pd.DataFrame(data)


@pytest.fixture
def xgb_config(feature_cols):
    return ModelConfig(
        model_type="xgb",
        run_id="test_run",
        is_windowing=False,
        feature_names=feature_cols,
        target_transform="log1p",
        feature_scaling="StandardScaler",
    )


@pytest.fixture
def model_artifacts(tmp_path, xgb_config, feature_cols):
    """Create a complete set of model artifacts for integration testing."""
    model_dir = tmp_path / "xgb" / "test_run"
    model_dir.mkdir(parents=True)

    model = MockModel()
    model_path = model_dir / "FINAL_xgb_test_run.joblib"
    joblib.dump(model, model_path)

    # Create config JSON
    config_path = model_dir / "FINAL_config_test_run.json"
    with open(config_path, "w") as f:
        json.dump(xgb_config.to_dict(), f)

    # Create feature scaler
    scaler = StandardScaler()
    scaler.fit(np.random.rand(100, len(feature_cols)))
    scaler_path = model_dir / "FINAL_scaler_test_run_feature.pkl"
    joblib.dump(scaler, scaler_path)

    # Create label scaler
    label_scaler = StandardScaler()
    label_scaler.fit(np.random.rand(100, 1))
    label_scaler_path = model_dir / "FINAL_scaler_test_run_label.pkl"
    joblib.dump(label_scaler, label_scaler_path)

    return tmp_path


# -- 1. Config import regression (C1 fix) ------------------------------------


class TestConfigImportRegression:
    """Test config.py content by reading the file directly."""

    @pytest.fixture
    def config_source(self):
        config_path = Path(__file__).parent / "config.py"
        return config_path.read_text()

    def test_config_defines_model_dir(self, config_source):
        """config.py should define MODEL_DIR."""
        assert "MODEL_DIR =" in config_source

    def test_config_no_models_dir(self, config_source):
        """config.py should NOT define MODELS_DIR."""
        assert "MODELS_DIR =" not in config_source

    def test_config_no_scaler_dir(self, config_source):
        """config.py should NOT define SCALER_DIR."""
        assert "SCALER_DIR =" not in config_source

    def test_config_no_duplicate_biome_types(self, config_source):
        """config.py should have BIOME_TYPES defined exactly once."""
        count = config_source.count("BIOME_TYPES = [")
        assert count == 1, f"BIOME_TYPES defined {count} times"

    def test_config_single_dask_memory_limit(self, config_source):
        """config.py should have DASK_MEMORY_LIMIT defined once."""
        count = config_source.count("DASK_MEMORY_LIMIT =")
        assert count == 1, f"DASK_MEMORY_LIMIT defined {count} times"


# -- 2. Model loading integration --------------------------------------------


class TestModelLoadingIntegration:
    def test_load_model_from_artifacts(self, model_artifacts):
        """Should load a real joblib model."""
        model = load_model(model_artifacts, "xgb", "test_run")
        assert model is not None
        assert hasattr(model, "predict")

    def test_load_models_returns_dict(self, model_artifacts):
        """load_models should return a dict with model_type key."""
        models = load_models(model_artifacts, "xgb", "test_run")
        assert isinstance(models, dict)
        assert "xgb" in models
        assert hasattr(models["xgb"], "predict")

    def test_load_config_from_artifacts(self, model_artifacts):
        """Should load config JSON into ModelConfig."""
        config = load_model_config_for_run(model_artifacts, "xgb", "test_run")
        assert config is not None
        assert config.model_type == "xgb"
        assert config.run_id == "test_run"

    def test_load_scalers_from_artifacts(self, model_artifacts):
        """Should load both feature and label scalers."""
        feat_scaler, label_scaler = load_scalers(model_artifacts, "xgb", "test_run")
        assert feat_scaler is not None
        assert label_scaler is not None
        assert hasattr(feat_scaler, "transform")
        assert hasattr(label_scaler, "transform")


# -- 3. Full flat prediction pipeline ----------------------------------------


class TestFullFlatPipeline:
    def test_end_to_end_flat_prediction(self, sample_df, model_artifacts, feature_cols):
        """Full pipeline: load model -> prepare features -> predict -> map."""
        # Load artifacts
        models = load_models(model_artifacts, "xgb", "test_run")
        configs = load_model_configs(model_artifacts, "xgb", "test_run")
        feat_scaler, label_scaler = load_scalers(model_artifacts, "xgb", "test_run")

        config = configs.get("xgb")

        # Prepare features
        df_prep, feat_cols = prepare_features_from_preprocessed(sample_df, feature_scaler=feat_scaler, config=config)
        assert df_prep is not None
        assert len(feat_cols) > 0

        # Make predictions
        df_preds = make_predictions_improved(df_prep, models, feat_cols, label_scaler, model_configs=configs)
        assert not df_preds.empty
        assert "sap_velocity_xgb" in df_preds.columns

        # Validate predictions
        result = validate_predictions(df_preds, feat_cols)
        # Predictions from random model may have negatives, so result varies

    def test_csv_roundtrip_integration(self, tmp_path, sample_df, model_artifacts, feature_cols):
        """Save data to CSV, reload, predict, save predictions."""
        # Save sample data
        csv_path = tmp_path / "test_input.csv"
        sample_df.to_csv(csv_path, index=False)

        # Load via the pipeline function
        df_loaded = load_preprocessed_data(csv_path)
        assert df_loaded is not None
        assert len(df_loaded) == len(sample_df)

        # Load model and predict
        models = load_models(model_artifacts, "xgb", "test_run")
        configs = load_model_configs(model_artifacts, "xgb", "test_run")
        config = configs.get("xgb")

        df_prep, feat_cols = prepare_features_from_preprocessed(df_loaded, config=config)
        assert df_prep is not None

        df_preds = make_predictions_improved(df_prep, models, feat_cols, None, model_configs=configs)
        assert "sap_velocity_xgb" in df_preds.columns

        # Save predictions
        out_path = tmp_path / "predictions.csv"
        df_preds.to_csv(out_path, index=False)
        assert out_path.exists()

        # Reload and verify
        df_reloaded = pd.read_csv(out_path)
        assert "sap_velocity_xgb" in df_reloaded.columns


# -- 4. DataTransformer + scaler integration ----------------------------------


class TestTransformerScalerIntegration:
    def test_transform_predict_inverse(self, xgb_config):
        """Full transform cycle: scale features, predict, inverse transform."""
        transformer = DataTransformer(xgb_config)

        # Simulate target values
        y_original = np.array([10.0, 20.0, 50.0, 100.0])

        # Forward transform
        y_transformed = transformer.transform_target(y_original)
        assert not np.array_equal(y_transformed, y_original)

        # Inverse transform
        y_recovered = transformer.transform_target(y_transformed, inverse=True)
        np.testing.assert_allclose(y_recovered, y_original, rtol=1e-6)

    def test_scaler_consistency(self, model_artifacts, feature_cols):
        """Scaler loaded from disk should match the one that was saved."""
        feat_scaler, _ = load_scalers(model_artifacts, "xgb", "test_run")

        # The scaler was fit on 100 samples of 5 features
        assert feat_scaler.n_features_in_ == len(feature_cols)
        assert feat_scaler.mean_ is not None
        assert len(feat_scaler.mean_) == len(feature_cols)


# -- 5. Regression: hoisted model loading -------------------------------------


class TestHoistedModelLoadingRegression:
    def test_year_extraction_for_output_dir(self):
        """_extract_year_from_filename should work for typical ERA5 filenames."""
        assert _extract_year_from_filename("era5land_daily_2015") == "2015"
        assert _extract_year_from_filename("preprocessed_2020_jan") == "2020"
        assert _extract_year_from_filename("no_year_here") == "unknown"

    def test_model_dir_path_construction(self, tmp_path):
        """get_model_dir should construct correct paths."""
        result = get_model_dir(tmp_path, "xgb", "run_01")
        expected = tmp_path / "xgb" / "run_01"
        assert result == expected


# -- 6. Config env var override (M5 fix) --------------------------------------


class TestConfigEnvVarOverride:
    def test_era5land_dir_env_override(self, monkeypatch, tmp_path):
        """GSV_ERA5LAND_DIR env var should override default path."""
        custom_path = str(tmp_path / "era5")
        monkeypatch.setenv("GSV_ERA5LAND_DIR", custom_path)
        monkeypatch.setenv("GSV_ERA5LAND_TEMP_DIR", str(tmp_path / "temp"))
        import importlib
        import config
        importlib.reload(config)
        assert str(config.ERA5LAND_DIR) == custom_path
        monkeypatch.delenv("GSV_ERA5LAND_DIR", raising=False)
        monkeypatch.delenv("GSV_ERA5LAND_TEMP_DIR", raising=False)
        importlib.reload(config)

    def test_era5land_temp_dir_env_override(self, monkeypatch, tmp_path):
        """GSV_ERA5LAND_TEMP_DIR env var should override default path."""
        custom_path = str(tmp_path / "temp")
        monkeypatch.setenv("GSV_ERA5LAND_DIR", str(tmp_path / "era5"))
        monkeypatch.setenv("GSV_ERA5LAND_TEMP_DIR", custom_path)
        import importlib
        import config
        importlib.reload(config)
        assert str(config.ERA5LAND_TEMP_DIR) == custom_path
        monkeypatch.delenv("GSV_ERA5LAND_DIR", raising=False)
        monkeypatch.delenv("GSV_ERA5LAND_TEMP_DIR", raising=False)
        importlib.reload(config)


# -- 7. Window metadata completeness -----------------------------------------


class TestWindowMetadataCompleteness:
    def test_metadata_keys_present(self, sample_df, feature_cols):
        """Each window metadata should have all required keys."""
        pytest.importorskip("tensorflow")
        from predict_sap_velocity_sequential import create_prediction_windows_improved

        _, metadata, _, _ = create_prediction_windows_improved(sample_df, feature_cols, input_width=4, shift=1)
        required_keys = {
            "location_id",
            "window_start_idx",
            "window_end_idx",
            "prediction_target_idx",
            "window_position",
            "global_window_index",
            "shift",
        }
        for m in metadata:
            assert required_keys.issubset(m.keys()), f"Missing keys: {required_keys - m.keys()}"

    def test_metadata_shift_matches_param(self, sample_df, feature_cols):
        """Shift value in metadata should match the input parameter."""
        pytest.importorskip("tensorflow")
        from predict_sap_velocity_sequential import create_prediction_windows_improved

        _, metadata, _, _ = create_prediction_windows_improved(sample_df, feature_cols, input_width=4, shift=3)
        for m in metadata:
            assert m["shift"] == 3
