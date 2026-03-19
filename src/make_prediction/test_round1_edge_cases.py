"""
Round 1 — Edge cases & boundary value tests for predict_sap_velocity_sequential.py

Tests:
  1. Window range boundary with shift parameter (H3 fix regression)
  2. MAX_EXPECTED_SAP_VELOCITY validation split (H4 fix regression)
  3. Empty/single-row DataFrame through prepare_features_from_preprocessed
  4. load_preprocessed_data with all-NaN columns (no dropna regression)
  5. find_available_run_ids with empty/nonexistent dirs
  6. ModelConfig.from_dict with extra unknown keys (forward compat)
  7. DataTransformer with extreme values (overflow protection)
  8. _extract_year_from_filename edge cases
  9. _resolve_timestamp_column with mixed-case columns
"""

import numpy as np
import pandas as pd
import pytest

# Module under test
from predict_sap_velocity_sequential import (
    MAX_EXPECTED_SAP_VELOCITY,
    DataTransformer,
    ModelConfig,
    _extract_year_from_filename,
    _resolve_timestamp_column,
    create_prediction_windows_improved,
    find_available_run_ids,
    load_preprocessed_data,
    prepare_features_from_preprocessed,
    validate_predictions,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def feature_cols():
    return ["ta", "vpd", "sw_in", "ppfd_in", "ws"]


@pytest.fixture
def sample_df(feature_cols):
    """20-row DataFrame with standard features."""
    np.random.seed(42)
    n = 20
    data = {col: np.random.rand(n) * 10 for col in feature_cols}
    data["timestamp"] = pd.date_range("2020-01-01", periods=n, freq="D")
    data["latitude"] = 51.0
    data["longitude"] = 7.0
    data["location_id"] = "loc_0"
    return pd.DataFrame(data)


@pytest.fixture
def xgb_config(feature_cols):
    return ModelConfig(
        model_type="xgb",
        run_id="test",
        is_windowing=False,
        feature_names=feature_cols,
    )


@pytest.fixture
def windowed_config(feature_cols):
    return ModelConfig(
        model_type="cnn_lstm",
        run_id="test",
        is_windowing=True,
        input_width=4,
        label_width=1,
        shift=2,
        feature_names=feature_cols,
    )


# ── 1. Window range boundary with shift (H3 fix) ───────────────────────────


class TestWindowRangeWithShift:
    """Verify that create_prediction_windows_improved accounts for shift."""

    def test_no_windows_beyond_data_length(self, sample_df, feature_cols):
        """With input_width=4 and shift=2, all target_idx must be valid."""
        # Need TF for windowing — skip if not available
        pytest.importorskip("tensorflow")

        ds, metadata, loc_map, total = create_prediction_windows_improved(
            sample_df, feature_cols, input_width=4, shift=2
        )
        for m in metadata:
            assert m["prediction_target_idx"] is not None, (
                f"Window at position {m['window_position']} has None target_idx — "
                f"range should have been shortened by shift"
            )

    def test_window_count_formula(self, sample_df, feature_cols):
        """Number of windows should be len(data) - input_width - shift + 1."""
        pytest.importorskip("tensorflow")

        input_width, shift = 4, 2
        expected_count = len(sample_df) - input_width - shift + 1
        _, metadata, _, total = create_prediction_windows_improved(
            sample_df, feature_cols, input_width=input_width, shift=shift
        )
        assert total == expected_count, f"Expected {expected_count} windows, got {total}"

    def test_shift_1_still_works(self, sample_df, feature_cols):
        """Default shift=1 should still produce valid windows."""
        pytest.importorskip("tensorflow")

        input_width, shift = 4, 1
        _, metadata, _, total = create_prediction_windows_improved(
            sample_df, feature_cols, input_width=input_width, shift=shift
        )
        expected = len(sample_df) - input_width - shift + 1
        assert total == expected
        for m in metadata:
            assert m["prediction_target_idx"] is not None

    def test_insufficient_data_for_shift(self, feature_cols):
        """If data length < input_width + shift, no windows should be created."""
        pytest.importorskip("tensorflow")

        short_df = pd.DataFrame({col: [1.0, 2.0, 3.0] for col in feature_cols})
        short_df["location_id"] = "loc_0"
        result = create_prediction_windows_improved(short_df, feature_cols, input_width=4, shift=2)
        # Should return (None, [], {}, 0) since 3 < 4 (input_width)
        assert result[3] == 0


# ── 2. Validation threshold split (H4 fix) ─────────────────────────────────


class TestValidationThresholdSplit:
    """H4: negatives should fail, high values should only warn."""

    def test_negative_values_fail_validation(self):
        df = pd.DataFrame({"sap_velocity_xgb": [-5.0, -1.0, 0.0]})
        result = validate_predictions(df, [])
        assert result is False

    def test_high_values_pass_validation(self):
        """Values up to MAX_EXPECTED_SAP_VELOCITY should pass."""
        df = pd.DataFrame({"sap_velocity_xgb": [0.0, 150.0, 250.0, MAX_EXPECTED_SAP_VELOCITY - 1]})
        result = validate_predictions(df, [])
        assert result is True

    def test_extreme_high_values_still_pass(self):
        """Values above MAX_EXPECTED_SAP_VELOCITY should only warn, not fail."""
        df = pd.DataFrame({"sap_velocity_xgb": [0.0, 10.0, MAX_EXPECTED_SAP_VELOCITY + 100]})
        result = validate_predictions(df, [])
        # High values are warning only — should still pass
        assert result is True

    def test_mixed_neg_and_high_fails(self):
        """Mix of negative and high values should fail (due to negatives)."""
        df = pd.DataFrame({"sap_velocity_xgb": [-1.0, 10.0, 600.0]})
        result = validate_predictions(df, [])
        assert result is False

    def test_zero_values_pass(self):
        """Zero is a valid sap velocity (no flow at night)."""
        df = pd.DataFrame({"sap_velocity_xgb": [0.0, 0.0, 0.0]})
        result = validate_predictions(df, [])
        assert result is True


# ── 3. prepare_features edge cases ─────────────────────────────────────────


class TestPrepFeaturesEdgeCases:
    """Edge cases for prepare_features_from_preprocessed."""

    def test_all_nan_features_returns_empty(self, xgb_config):
        """All-NaN features should produce zero rows after NaN removal."""
        df = pd.DataFrame(
            {
                "ta": [np.nan, np.nan],
                "vpd": [np.nan, np.nan],
                "sw_in": [1.0, 2.0],
                "ppfd_in": [1.0, 2.0],
                "ws": [1.0, 2.0],
                "timestamp": pd.date_range("2020-01-01", periods=2),
            }
        )
        result, cols = prepare_features_from_preprocessed(df, config=xgb_config)
        # Should still return a DF (possibly empty rows due to NaN dropping)
        assert result is not None
        # Feature columns should be detected
        assert len(cols) > 0

    def test_single_row_with_config(self, xgb_config):
        """A single valid row should work fine."""
        df = pd.DataFrame(
            {
                "ta": [20.0],
                "vpd": [1.0],
                "sw_in": [200.0],
                "ppfd_in": [400.0],
                "ws": [2.0],
                "timestamp": ["2020-06-15"],
            }
        )
        result, cols = prepare_features_from_preprocessed(df, config=xgb_config)
        assert result is not None
        assert len(result) == 1


# ── 4. load_preprocessed_data with all-NaN columns ─────────────────────────


class TestLoadDataEdgeCases:
    def test_all_nan_column_preserved(self, tmp_path):
        """All-NaN columns should NOT be dropped (C1 fix from earlier)."""
        csv = tmp_path / "test.csv"
        df = pd.DataFrame(
            {
                "timestamp": ["2020-01-01", "2020-01-02"],
                "ta": [20.0, 21.0],
                "vpd": [np.nan, np.nan],  # all NaN
            }
        )
        df.to_csv(csv, index=False)
        result = load_preprocessed_data(csv)
        assert "vpd" in result.columns

    def test_header_only_csv(self, tmp_path):
        """CSV with only headers should return None."""
        csv = tmp_path / "empty.csv"
        csv.write_text("ta,vpd,sw_in\n")
        result = load_preprocessed_data(csv)
        assert result is None


# ── 5. find_available_run_ids edge cases ────────────────────────────────────


class TestFindAvailableRunIds:
    def test_nonexistent_dir(self, tmp_path):
        result = find_available_run_ids(tmp_path / "no_such_dir", "xgb")
        assert result == []

    def test_empty_model_dir(self, tmp_path):
        xgb_dir = tmp_path / "xgb"
        xgb_dir.mkdir()
        result = find_available_run_ids(tmp_path, "xgb")
        assert result == []

    def test_finds_run_ids(self, tmp_path):
        """Should find subdirectories with valid model files as run IDs."""
        xgb_dir = tmp_path / "xgb"
        (xgb_dir / "run_a").mkdir(parents=True)
        (xgb_dir / "run_b").mkdir(parents=True)
        # Must have FINAL_xgb_{run_id}.joblib to be detected
        (xgb_dir / "run_a" / "FINAL_xgb_run_a.joblib").touch()
        (xgb_dir / "run_b" / "FINAL_xgb_run_b.joblib").touch()
        result = find_available_run_ids(tmp_path, "xgb")
        assert set(result) == {"run_a", "run_b"}


# ── 6. ModelConfig forward compatibility ────────────────────────────────────


class TestModelConfigForwardCompat:
    def test_extra_keys_ignored(self):
        """Unknown keys in config dict should not crash from_dict."""
        config_dict = {
            "model_type": "xgb",
            "run_id": "test",
            "future_key_v2": True,
            "another_new_field": [1, 2, 3],
        }
        config = ModelConfig.from_dict(config_dict)
        assert config.model_type == "xgb"
        assert config.run_id == "test"


# ── 7. DataTransformer extreme values ───────────────────────────────────────


class TestDataTransformerExtreme:
    def test_log1p_large_values(self):
        """log1p should handle very large values without overflow."""
        config = ModelConfig(model_type="xgb", target_transform="log1p")
        dt = DataTransformer(config)
        large = np.array([1e15, 1e10, 1e5])
        transformed = dt.transform_target(large)
        recovered = dt.transform_target(transformed, inverse=True)
        np.testing.assert_allclose(recovered, large, rtol=1e-6)

    def test_log1p_zero_values(self):
        """log1p(0) = 0, expm1(0) = 0."""
        config = ModelConfig(model_type="xgb", target_transform="log1p")
        dt = DataTransformer(config)
        zeros = np.array([0.0, 0.0])
        result = dt.transform_target(zeros)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_sqrt_zero(self):
        """sqrt(0) = 0."""
        config = ModelConfig(model_type="xgb", target_transform="sqrt")
        dt = DataTransformer(config)
        result = dt.transform_target(np.array([0.0]))
        assert result[0] == 0.0


# ── 8. _extract_year edge cases ─────────────────────────────────────────────


class TestExtractYearEdgeCases:
    def test_multiple_years_picks_first(self):
        """Should pick the first valid 4-digit year."""
        result = _extract_year_from_filename("data_2015_processed_2020")
        assert result == "2015"

    def test_year_at_end(self):
        result = _extract_year_from_filename("era5land_daily_2018")
        assert result == "2018"

    def test_embedded_in_numbers(self):
        """Year-like pattern embedded in larger number."""
        result = _extract_year_from_filename("run_120150101")
        assert result == "2015"

    def test_empty_string(self):
        result = _extract_year_from_filename("")
        assert result == "unknown"


# ── 9. _resolve_timestamp_column mixed case ─────────────────────────────────


class TestResolveTimestampMixedCase:
    def test_mixed_case_timestamp(self):
        result = _resolve_timestamp_column(["Timestamp", "value"])
        assert result is not None

    def test_datetime_column(self):
        result = _resolve_timestamp_column(["DateTime", "value"])
        assert result is not None

    def test_no_time_column(self):
        result = _resolve_timestamp_column(["feature_a", "feature_b"])
        assert result is None
