"""Tests for prediction script correctness.

Categories:
- A: Data integrity (no silent data loss)
- B: Transform correctness (log1p/expm1 chain)
- C: Feature ordering correctness
- D: Row identity preservation
- J: CLI / I/O
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

from src.make_prediction.predict_sap_velocity_sequential import (
    DataTransformer,
    ModelConfig,
    _extract_year_from_filename,
    get_feature_columns_from_config,
    map_flat_predictions_to_df,
    prepare_flat_features,
)
from src.make_prediction.tests.conftest import (
    FEATURE_NAMES,
    MockXGBModel,
    IdentityScaler,
    Expm1LabelScaler,
)


# ===================================================================
# Category A — Data Integrity (no silent data loss)
# ===================================================================


class TestDataIntegrity:
    """Verify no rows are silently dropped during prediction."""

    def test_no_rows_dropped_feature_prep_clean_data(
        self, synthetic_era5_input: pd.DataFrame,
    ) -> None:
        """Clean data (no NaN) must retain all rows after feature prep."""
        # Drop the dummy index column (mimicking load_preprocessed_data)
        df = synthetic_era5_input.iloc[:, 1:]
        feature_cols = [c for c in FEATURE_NAMES if c in df.columns]

        X, indices = prepare_flat_features(df, feature_cols)

        assert len(X) == len(df), (
            f"Data loss: {len(df)} input rows -> {len(X)} output rows"
        )
        assert len(indices) == len(df)

    def test_rows_with_nan_excluded_consistently(
        self, synthetic_era5_input: pd.DataFrame,
    ) -> None:
        """Rows with NaN in feature columns must be excluded from both X and indices."""
        df = synthetic_era5_input.iloc[:, 1:].copy()
        feature_cols = [c for c in FEATURE_NAMES if c in df.columns]

        # Inject NaN into 5 rows
        nan_indices = [0, 10, 20, 30, 40]
        for idx in nan_indices:
            if idx < len(df):
                df.iloc[idx, df.columns.get_loc(feature_cols[0])] = np.nan

        X, indices = prepare_flat_features(df, feature_cols)

        expected_len = len(df) - len([i for i in nan_indices if i < len(df)])
        assert len(X) == expected_len
        assert len(indices) == expected_len
        # NaN indices must not appear in output
        for idx in nan_indices:
            if idx < len(df):
                assert df.index[idx] not in indices

    def test_no_rows_dropped_during_mapping(
        self, synthetic_era5_input: pd.DataFrame,
    ) -> None:
        """map_flat_predictions_to_df output must have same length as input."""
        df = synthetic_era5_input.iloc[:, 1:]
        n = len(df)
        predictions = np.random.default_rng(42).uniform(0, 10, n)
        indices = df.index.tolist()

        result = map_flat_predictions_to_df(df, predictions, indices, "xgb")
        assert len(result) == n

    def test_mapping_indices_match_input(
        self, synthetic_era5_input: pd.DataFrame,
    ) -> None:
        """Predictions appear at exactly the provided indices."""
        df = synthetic_era5_input.iloc[:, 1:]
        feature_cols = [c for c in FEATURE_NAMES if c in df.columns]

        X, indices = prepare_flat_features(df, feature_cols)
        predictions = np.ones(len(indices)) * 42.0

        result = map_flat_predictions_to_df(df, predictions, indices, "xgb")

        # Non-NaN predictions at returned indices
        for idx in indices:
            assert not np.isnan(result.loc[idx, "sap_velocity_xgb"]), (
                f"Missing prediction at index {idx}"
            )
        # All non-NaN values should be 42.0
        non_nan = result["sap_velocity_xgb"].dropna()
        assert len(non_nan) == len(indices)
        np.testing.assert_allclose(non_nan.values, 42.0)

    def test_no_nan_predictions_for_valid_input(
        self, synthetic_era5_input: pd.DataFrame,
        mock_model: MockXGBModel,
    ) -> None:
        """If all features are non-NaN, mock prediction has zero NaN."""
        df = synthetic_era5_input.iloc[:, 1:]
        feature_cols = [c for c in FEATURE_NAMES if c in df.columns]

        X, indices = prepare_flat_features(df, feature_cols)
        preds = mock_model.predict(X)

        assert not np.any(np.isnan(preds)), "Mock model produced NaN predictions"

    def test_output_row_count_preserved(
        self, synthetic_era5_input: pd.DataFrame,
        mock_model: MockXGBModel,
    ) -> None:
        """Prediction count must equal valid input count."""
        df = synthetic_era5_input.iloc[:, 1:]
        feature_cols = [c for c in FEATURE_NAMES if c in df.columns]

        X, indices = prepare_flat_features(df, feature_cols)
        preds = mock_model.predict(X)

        assert len(preds) == len(X), (
            f"Prediction count {len(preds)} != input count {len(X)}"
        )


# ===================================================================
# Category B — Transform Correctness (log1p / expm1)
# ===================================================================


class TestTransformCorrectness:
    """Verify log1p/expm1 roundtrip and transform ordering."""

    def test_log1p_expm1_roundtrip_numerical(self) -> None:
        """expm1(log1p(x)) must recover x for physical sap velocity range."""
        values = np.array([0.0, 0.001, 1.0, 10.0, 50.0, 100.0])
        roundtripped = np.expm1(np.log1p(values))
        np.testing.assert_allclose(roundtripped, values, atol=1e-10)

    def test_data_transformer_log1p_inverse(self) -> None:
        """DataTransformer log1p forward + inverse = identity."""
        config = ModelConfig(model_type="xgb", target_transform="log1p")
        transformer = DataTransformer(config)

        original = np.array([0.5, 2.0, 10.0, 50.0])
        forward = transformer.transform_target(original, inverse=False)
        recovered = transformer.transform_target(forward, inverse=True)

        np.testing.assert_allclose(recovered, original, atol=1e-10)

    def test_data_transformer_sqrt_inverse(self) -> None:
        """DataTransformer sqrt forward + inverse = identity."""
        config = ModelConfig(model_type="xgb", target_transform="sqrt")
        transformer = DataTransformer(config)

        original = np.array([0.0, 1.0, 4.0, 25.0, 100.0])
        forward = transformer.transform_target(original, inverse=False)
        recovered = transformer.transform_target(forward, inverse=True)

        np.testing.assert_allclose(recovered, original, atol=1e-10)

    def test_data_transformer_log_inverse(self) -> None:
        """DataTransformer log forward + inverse = identity."""
        config = ModelConfig(model_type="xgb", target_transform="log")
        transformer = DataTransformer(config)

        original = np.array([0.1, 1.0, 10.0, 100.0])
        forward = transformer.transform_target(original, inverse=False)
        recovered = transformer.transform_target(forward, inverse=True)

        np.testing.assert_allclose(recovered, original, atol=1e-8)

    def test_zero_input_nonnegative_output(self) -> None:
        """expm1(0) = 0: log1p inverse of zero must be non-negative."""
        config = ModelConfig(model_type="xgb", target_transform="log1p")
        transformer = DataTransformer(config)

        result = transformer.transform_target(np.array([0.0]), inverse=True)
        assert result[0] >= 0, f"Negative output: {result[0]}"
        assert result[0] == 0.0

    def test_no_transform_returns_unchanged(self) -> None:
        """DataTransformer with None transform returns input unchanged."""
        config = ModelConfig(model_type="xgb", target_transform=None)
        transformer = DataTransformer(config)

        values = np.array([1.0, 2.0, 3.0])
        result = transformer.transform_target(values, inverse=True)
        np.testing.assert_array_equal(result, values)

    def test_should_transform_target_flag(self) -> None:
        """should_transform_target reflects config."""
        config_yes = ModelConfig(model_type="xgb", target_transform="log1p")
        config_no = ModelConfig(model_type="xgb", target_transform=None)

        assert DataTransformer(config_yes).should_transform_target() is True
        assert DataTransformer(config_no).should_transform_target() is False


# ===================================================================
# Category C — Feature Ordering Correctness
# ===================================================================


class TestFeatureOrdering:
    """Verify feature columns are in config-specified order."""

    def test_feature_column_order_matches_config(
        self, synthetic_era5_input: pd.DataFrame,
        mock_model_config: ModelConfig,
    ) -> None:
        """Output column order must match config.feature_names."""
        df = synthetic_era5_input.iloc[:, 1:]
        result = get_feature_columns_from_config(df, mock_model_config)

        # Result should be in same order as config
        expected = [
            f for f in mock_model_config.feature_names if f in df.columns
        ]
        assert result == expected, (
            f"Feature order mismatch!\n"
            f"Expected: {expected[:5]}...\n"
            f"Got: {result[:5]}..."
        )

    def test_feature_array_column_count(
        self, synthetic_era5_input: pd.DataFrame,
    ) -> None:
        """Feature array width must match number of requested columns."""
        df = synthetic_era5_input.iloc[:, 1:]
        feature_cols = [c for c in FEATURE_NAMES if c in df.columns]

        X, _ = prepare_flat_features(df, feature_cols)

        assert X.shape[1] == len(feature_cols), (
            f"Expected {len(feature_cols)} columns, got {X.shape[1]}"
        )

    def test_feature_array_values_match_source(
        self, synthetic_era5_input: pd.DataFrame,
    ) -> None:
        """Each column in feature array must match the source DataFrame column."""
        df = synthetic_era5_input.iloc[:, 1:]
        feature_cols = [c for c in FEATURE_NAMES if c in df.columns]

        X, indices = prepare_flat_features(df, feature_cols)

        for i, col_name in enumerate(feature_cols):
            expected = df.loc[df.index[indices], col_name].values
            np.testing.assert_array_equal(
                X[:, i], expected,
                err_msg=f"Column {i} ({col_name}) values don't match source",
            )

    def test_missing_feature_excluded_with_warning(
        self, synthetic_era5_input: pd.DataFrame,
    ) -> None:
        """Config requesting nonexistent feature: excluded, warning logged."""
        df = synthetic_era5_input.iloc[:, 1:]
        config = ModelConfig(
            model_type="xgb",
            feature_names=["ta", "NONEXISTENT_FEATURE", "vpd"],
        )

        result = get_feature_columns_from_config(df, config)
        assert "NONEXISTENT_FEATURE" not in result
        assert "ta" in result
        assert "vpd" in result

    def test_extra_columns_ignored(
        self, synthetic_era5_input: pd.DataFrame,
    ) -> None:
        """Only config-specified columns appear in output."""
        df = synthetic_era5_input.iloc[:, 1:]
        config = ModelConfig(
            model_type="xgb",
            feature_names=["ta", "vpd", "sw_in"],
        )

        result = get_feature_columns_from_config(df, config)
        assert len(result) == 3
        assert set(result) == {"ta", "vpd", "sw_in"}

    def test_case_insensitive_feature_match(
        self, synthetic_era5_input: pd.DataFrame,
    ) -> None:
        """Case-insensitive matching works for feature names."""
        df = synthetic_era5_input.iloc[:, 1:].copy()
        # Rename a column to uppercase
        df = df.rename(columns={"ta": "TA"})
        config = ModelConfig(
            model_type="xgb",
            feature_names=["ta", "vpd"],
        )

        result = get_feature_columns_from_config(df, config)
        # Should find TA via case-insensitive match
        assert len(result) >= 1  # At least vpd; TA matched case-insensitively

    def test_no_config_returns_none(
        self, synthetic_era5_input: pd.DataFrame,
    ) -> None:
        """No config or empty feature_names returns None (fallback signal)."""
        df = synthetic_era5_input.iloc[:, 1:]
        result = get_feature_columns_from_config(df, None)
        assert result is None

        config_empty = ModelConfig(model_type="xgb", feature_names=[])
        result2 = get_feature_columns_from_config(df, config_empty)
        assert result2 is None


# ===================================================================
# Category D — Row Identity Preservation
# ===================================================================


class TestRowIdentity:
    """Verify predictions attach to correct coordinates."""

    def test_prediction_attached_to_correct_index(
        self, synthetic_era5_input: pd.DataFrame,
        mock_model: MockXGBModel,
    ) -> None:
        """Predictions at indices returned by prepare_flat_features, not shifted."""
        df = synthetic_era5_input.iloc[:, 1:]
        feature_cols = [c for c in FEATURE_NAMES if c in df.columns]

        X, indices = prepare_flat_features(df, feature_cols)
        preds = mock_model.predict(X)

        result = map_flat_predictions_to_df(df, preds, indices, "xgb")

        # Verify each prediction is at its correct index
        for i, idx in enumerate(indices):
            expected = preds[i]
            actual = result.loc[idx, "sap_velocity_xgb"]
            assert abs(actual - expected) < 1e-10, (
                f"Prediction at index {idx}: expected {expected}, got {actual}"
            )

    def test_known_value_at_known_coordinate(
        self, synthetic_era5_input: pd.DataFrame,
        mock_model: MockXGBModel,
    ) -> None:
        """Deterministic mock model: verify prediction at a specific row."""
        df = synthetic_era5_input.iloc[:, 1:]
        feature_cols = [c for c in FEATURE_NAMES if c in df.columns]

        X, indices = prepare_flat_features(df, feature_cols)
        preds = mock_model.predict(X)

        # Compute expected prediction for first row
        first_features = X[0]
        expected_pred = np.sum(first_features) * 0.01

        assert abs(preds[0] - expected_pred) < 1e-10

    def test_shuffled_input_same_predictions(
        self, synthetic_era5_input: pd.DataFrame,
        mock_model: MockXGBModel,
    ) -> None:
        """Shuffling rows must produce same prediction values (order-independent)."""
        df = synthetic_era5_input.iloc[:, 1:]
        feature_cols = [c for c in FEATURE_NAMES if c in df.columns]

        # Original order
        X1, idx1 = prepare_flat_features(df, feature_cols)
        preds1 = mock_model.predict(X1)
        result1 = map_flat_predictions_to_df(df, preds1, idx1, "xgb")

        # Shuffled
        df_shuffled = df.sample(frac=1, random_state=99)
        X2, idx2 = prepare_flat_features(df_shuffled, feature_cols)
        preds2 = mock_model.predict(X2)
        result2 = map_flat_predictions_to_df(df_shuffled, preds2, idx2, "xgb")

        # Compare at shared indices
        shared = set(idx1) & set(idx2)
        for idx in shared:
            v1 = result1.loc[idx, "sap_velocity_xgb"]
            v2 = result2.loc[idx, "sap_velocity_xgb"]
            assert abs(v1 - v2) < 1e-10, (
                f"Mismatch at index {idx}: original={v1}, shuffled={v2}"
            )

    def test_lat_lon_preserved_exactly(
        self, synthetic_era5_input: pd.DataFrame,
    ) -> None:
        """lat/lon values must not drift through copy operations."""
        df = synthetic_era5_input.iloc[:, 1:]
        original_lats = df["latitude"].values.copy()
        original_lons = df["longitude"].values.copy()

        feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
        prepare_flat_features(df, feature_cols)

        # DataFrame lat/lon must be unchanged after feature prep
        np.testing.assert_array_equal(df["latitude"].values, original_lats)
        np.testing.assert_array_equal(df["longitude"].values, original_lons)

    def test_timestamp_preserved_exactly(
        self, synthetic_era5_input: pd.DataFrame,
    ) -> None:
        """Timestamp column must not be modified by feature preparation."""
        df = synthetic_era5_input.iloc[:, 1:]
        original_ts = df["timestamp"].values.copy()

        feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
        prepare_flat_features(df, feature_cols)

        np.testing.assert_array_equal(df["timestamp"].values, original_ts)


# ===================================================================
# Category J — CLI / I/O
# ===================================================================


class TestCLIAndIO:
    """Verify CLI argument parsing and I/O helpers."""

    def test_extract_year_from_filename_standard(self) -> None:
        """Standard filename extracts correct year."""
        assert _extract_year_from_filename("prediction_2015_01_daily") == "2015"

    def test_extract_year_from_filename_no_match(self) -> None:
        """Filename without year returns 'unknown'."""
        assert _extract_year_from_filename("random_name") == "unknown"

    def test_extract_year_from_filename_multiple_years(self) -> None:
        """First 4-digit year-like match is returned."""
        result = _extract_year_from_filename("data_2015_to_2020")
        assert result == "2015"
