"""Tests for feature_engineering extracted module."""

import numpy as np
import pandas as pd
import pytest

from src.hyperparameter_optimization.feature_engineering import (
    add_sap_flow_features,
    apply_feature_engineering,
)


class TestAddSapFlowFeatures:
    """Test add_sap_flow_features column creation."""

    @pytest.fixture
    def base_df(self):
        rng = np.random.RandomState(42)
        n = 100
        return pd.DataFrame(
            {
                "vpd": rng.uniform(0.1, 3.0, n),
                "sw_in": rng.uniform(0, 800, n),
                "ta": rng.uniform(5, 35, n),
                "precip": rng.uniform(0, 20, n),
                "ws": rng.uniform(0, 10, n),
                "LAI": rng.uniform(0.5, 6.0, n),
                "volumetric_soil_water_layer_1": rng.uniform(0.1, 0.5, n),
                "canopy_height": rng.uniform(5, 30, n),
                "soil_temperature_level_1": rng.uniform(273, 300, n),
                "ppfd_in": rng.uniform(0, 2000, n),
            }
        )

    def test_returns_dataframe(self, base_df):
        result = add_sap_flow_features(base_df)
        assert isinstance(result, pd.DataFrame)

    def test_adds_interaction_features(self, base_df):
        result = add_sap_flow_features(base_df)
        assert "vpd_x_sw_in" in result.columns

    def test_does_not_mutate_input(self, base_df):
        original_cols = set(base_df.columns)
        add_sap_flow_features(base_df)
        assert set(base_df.columns) == original_cols

    def test_handles_missing_columns(self):
        df = pd.DataFrame({"ta": [1.0, 2.0, 3.0]})
        result = add_sap_flow_features(df)
        assert isinstance(result, pd.DataFrame)


class TestApplyFeatureEngineering:
    """Test apply_feature_engineering with groups."""

    @pytest.fixture
    def grouped_df(self):
        rng = np.random.RandomState(42)
        n = 60
        df = pd.DataFrame(
            {
                "vpd": rng.uniform(0.1, 3.0, n),
                "sw_in": rng.uniform(0, 800, n),
                "ta": rng.uniform(5, 35, n),
                "precip": rng.uniform(0, 20, n),
                "ws": rng.uniform(0, 10, n),
                "LAI": rng.uniform(0.5, 6.0, n),
                "volumetric_soil_water_layer_1": rng.uniform(0.1, 0.5, n),
                "canopy_height": rng.uniform(5, 30, n),
                "soil_temperature_level_1": rng.uniform(273, 300, n),
                "ppfd_in": rng.uniform(0, 2000, n),
                "sap_velocity": rng.uniform(0, 50, n),
            }
        )
        groups = np.array([0] * 20 + [1] * 20 + [2] * 20)
        return df, groups

    def test_returns_tuple(self, grouped_df):
        df, groups = grouped_df
        result_df, new_features = apply_feature_engineering(df, groups)
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(new_features, list)

    def test_nan_handling(self, grouped_df):
        df, groups = grouped_df
        result_df, _ = apply_feature_engineering(df, groups)
        # Rolling features will have NaN at edges — that's expected
        # but no all-NaN columns should be created
        all_nan_cols = [c for c in result_df.columns if result_df[c].isna().all()]
        assert len(all_nan_cols) == 0
