"""Tests for shap_plotting extracted module — aggregation shape checks."""

import numpy as np
import pandas as pd
import pytest

from src.hyperparameter_optimization.shap_plotting import (
    aggregate_pft_shap_values,
    aggregate_static_feature_shap,
    generate_windowed_feature_names,
    get_dynamic_features_only,
    get_sample_pft_labels,
    group_pft_for_summary_plots,
)


class TestAggregatePFTShapValues:
    """Test PFT SHAP aggregation shapes and correctness."""

    def _make_data(self, n_samples: int = 100):
        rng = np.random.RandomState(42)
        feature_names = ["ta", "vpd", "MF", "DNF", "ENF", "EBF", "WSA", "WET", "DBF", "SAV"]
        shap_values = rng.randn(n_samples, len(feature_names))
        return shap_values, feature_names

    def test_output_shape_sum(self):
        shap_vals, names = self._make_data()
        agg, agg_names = aggregate_pft_shap_values(shap_vals, names)
        # 2 non-PFT + 1 aggregated PFT = 3
        assert agg.shape == (100, 3)
        assert agg_names == ["ta", "vpd", "PFT"]

    def test_output_shape_mean(self):
        shap_vals, names = self._make_data()
        agg, agg_names = aggregate_pft_shap_values(shap_vals, names, aggregation="mean")
        assert agg.shape == (100, 3)

    def test_sum_correctness(self):
        shap_vals, names = self._make_data()
        agg, _ = aggregate_pft_shap_values(shap_vals, names, aggregation="sum")
        # PFT sum should equal sum of columns 2-9
        expected_pft = shap_vals[:, 2:].sum(axis=1)
        np.testing.assert_allclose(agg[:, -1], expected_pft)

    def test_no_pft_columns(self):
        rng = np.random.RandomState(42)
        names = ["ta", "vpd", "sw_in"]
        shap_vals = rng.randn(50, 3)
        agg, agg_names = aggregate_pft_shap_values(shap_vals, names)
        assert agg.shape == (50, 3)
        assert agg_names == names

    def test_invalid_aggregation_raises(self):
        shap_vals, names = self._make_data()
        with pytest.raises(ValueError, match="Unknown aggregation"):
            aggregate_pft_shap_values(shap_vals, names, aggregation="max")


class TestGetSamplePFTLabels:
    """Test PFT label extraction from one-hot data."""

    def test_correct_labels(self):
        feature_names = ["ta", "MF", "ENF", "DBF"]
        # 3 samples: MF=1, ENF=1, DBF=1
        X = np.array(
            [
                [5.0, 1.0, 0.0, 0.0],
                [6.0, 0.0, 1.0, 0.0],
                [7.0, 0.0, 0.0, 1.0],
            ]
        )
        labels = get_sample_pft_labels(X, feature_names)
        np.testing.assert_array_equal(labels, ["MF", "ENF", "DBF"])

    def test_no_pft_columns(self):
        X = np.array([[1.0, 2.0]])
        labels = get_sample_pft_labels(X, ["ta", "vpd"])
        assert labels[0] == "Unknown"


class TestGroupPFTForSummaryPlots:
    """Test group_pft_for_summary_plots output structure."""

    def test_output_structure(self):
        rng = np.random.RandomState(42)
        feature_names = ["ta", "vpd", "MF", "ENF"]
        n = 50
        shap_values = rng.randn(n, 4)
        X_df = pd.DataFrame(rng.randn(n, 4), columns=feature_names)
        # Make PFT one-hot: half MF, half ENF
        X_df["MF"] = [1] * 25 + [0] * 25
        X_df["ENF"] = [0] * 25 + [1] * 25

        new_shap, new_X, new_names = group_pft_for_summary_plots(shap_values, X_df, feature_names, ["MF", "ENF"])
        assert new_shap.shape[1] == 3  # ta, vpd, PFT
        assert "PFT" in new_names
        assert isinstance(new_X, pd.DataFrame)


class TestGenerateWindowedFeatureNames:
    """Test windowed feature name generation."""

    def test_basic(self):
        # generate_windowed_feature_names iterates time steps, then features per step
        # For input_width=2: t=0 -> offset=1 (_t-1), t=1 -> offset=0 (_t-0)
        names = generate_windowed_feature_names(["ta", "vpd"], input_width=2)
        assert names == ["ta_t-1", "vpd_t-1", "ta_t-0", "vpd_t-0"]

    def test_width_1(self):
        names = generate_windowed_feature_names(["sw_in"], input_width=1)
        assert names == ["sw_in_t-0"]


class TestGetDynamicFeaturesOnly:
    """Test dynamic feature filtering."""

    def test_filters_static(self):
        rng = np.random.RandomState(42)
        feature_names = ["ta_t-0", "ta_t-1", "canopy_height", "MF"]
        shap_vals = rng.randn(10, 4)
        # Signature: (shap_values, feature_names, base_feature_names, input_width, static_features)
        dyn_shap, dyn_names = get_dynamic_features_only(
            shap_vals,
            feature_names,
            base_feature_names=["ta", "canopy_height", "MF"],
            input_width=2,
            static_features=["canopy_height", "MF"],
        )
        assert dyn_shap.shape == (10, 2)
        assert dyn_names == ["ta_t-0", "ta_t-1"]


class TestAggregateStaticFeatureShap:
    """Test static feature SHAP aggregation."""

    def test_basic_aggregation(self):
        rng = np.random.RandomState(42)
        base_features = ["ta", "canopy_height"]
        # Windowed: for input_width=2: ta_t-1, canopy_height_t-1, ta_t-0, canopy_height_t-0
        windowed_names = generate_windowed_feature_names(base_features, input_width=2)
        shap_vals = rng.randn(20, len(windowed_names))

        agg_shap, agg_names = aggregate_static_feature_shap(
            shap_vals,
            windowed_names,
            base_features,
            input_width=2,
            static_features=["canopy_height"],
        )
        assert "canopy_height" in agg_names
        assert agg_shap.shape[0] == 20
        # Dynamic features keep time suffixes, static get aggregated
        dynamic_names = [n for n in agg_names if n != "canopy_height"]
        assert len(dynamic_names) == 2  # ta_t-1, ta_t-0


class TestBoundaryValues:
    """Round 1: Edge cases and boundary values."""

    def test_aggregate_pft_single_sample(self):
        """Aggregation with a single sample."""
        shap_vals = np.array([[1.0, 2.0, 0.5, 0.3]])
        names = ["ta", "vpd", "MF", "ENF"]
        agg, agg_names = aggregate_pft_shap_values(shap_vals, names)
        assert agg.shape == (1, 3)
        assert agg_names == ["ta", "vpd", "PFT"]

    def test_aggregate_pft_all_pft_columns(self):
        """All columns are PFT — should aggregate to single PFT column."""
        rng = np.random.RandomState(42)
        names = ["MF", "DNF", "ENF"]
        shap_vals = rng.randn(10, 3)
        agg, agg_names = aggregate_pft_shap_values(shap_vals, names)
        assert agg.shape == (10, 1)
        assert agg_names == ["PFT"]

    def test_windowed_names_width_zero_empty(self):
        """Input width 0 should produce empty list."""
        names = generate_windowed_feature_names(["ta"], input_width=0)
        assert names == []

    def test_windowed_names_empty_features(self):
        """Empty feature list should produce empty output."""
        names = generate_windowed_feature_names([], input_width=3)
        assert names == []

    def test_get_sample_pft_labels_all_zeros(self):
        """All PFT columns zero — argmax picks first PFT (expected behavior)."""
        X = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        labels = get_sample_pft_labels(X, ["ta", "MF", "ENF"])
        # argmax on all-zero PFT row returns first PFT column
        assert all(label == "MF" for label in labels)


class TestNegativePaths:
    """Round 2: Negative paths and error handling."""

    def test_aggregate_pft_shap_shape_mismatch(self):
        """Feature names count != shap columns — function handles gracefully."""
        rng = np.random.RandomState(42)
        shap_vals = rng.randn(10, 3)
        names = ["ta", "vpd"]  # 2 names, 3 columns — no PFT found
        # Function logs warning and returns original data unchanged
        agg, agg_names = aggregate_pft_shap_values(shap_vals, names)
        assert agg.shape == (10, 3)
        assert agg_names == names

    def test_generate_windowed_names_negative_width(self):
        """Negative input_width produces empty list (range produces nothing)."""
        names = generate_windowed_feature_names(["ta"], input_width=-1)
        assert names == []

    def test_get_dynamic_features_no_static(self):
        """No static features — all features returned."""
        rng = np.random.RandomState(42)
        names = ["ta_t-0", "vpd_t-0"]
        shap_vals = rng.randn(5, 2)
        dyn_shap, dyn_names = get_dynamic_features_only(
            shap_vals,
            names,
            base_feature_names=["ta", "vpd"],
            input_width=1,
            static_features=[],
        )
        assert dyn_shap.shape == (5, 2)
        assert dyn_names == ["ta_t-0", "vpd_t-0"]

    def test_group_pft_empty_pft_list(self):
        """Empty PFT list — should return data unchanged."""
        rng = np.random.RandomState(42)
        names = ["ta", "vpd"]
        shap_vals = rng.randn(10, 2)
        X_df = pd.DataFrame(rng.randn(10, 2), columns=names)
        new_shap, new_X, new_names = group_pft_for_summary_plots(shap_vals, X_df, names, [])
        assert new_shap.shape == (10, 2)
        assert new_names == ["ta", "vpd"]
