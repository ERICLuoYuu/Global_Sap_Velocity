"""Tests for the AOA orchestrator."""

import numpy as np
import pytest
from src.aoa.aoa import compute_aoa


class TestComputeAOA:
    """Tests for the top-level compute_aoa function."""

    def test_returns_expected_keys(self):
        """Result dict should contain DI, AOA mask, and threshold."""
        rng = np.random.default_rng(42)
        train_X = rng.standard_normal((50, 5))
        new_X = rng.standard_normal((20, 5))
        weights = np.ones(5)
        fold_indices = [np.arange(0, 25), np.arange(25, 50)]

        result = compute_aoa(train_X, new_X, weights, fold_indices)

        assert "di" in result
        assert "aoa" in result
        assert "threshold" in result

    def test_aoa_mask_is_boolean(self):
        """AOA mask should be boolean array."""
        rng = np.random.default_rng(42)
        train_X = rng.standard_normal((50, 5))
        new_X = rng.standard_normal((20, 5))
        weights = np.ones(5)
        fold_indices = [np.arange(0, 25), np.arange(25, 50)]

        result = compute_aoa(train_X, new_X, weights, fold_indices)

        assert result["aoa"].dtype == bool
        assert result["aoa"].shape == (20,)

    def test_di_shape_matches_new_points(self):
        """DI array length should equal number of new points."""
        rng = np.random.default_rng(42)
        train_X = rng.standard_normal((30, 4))
        new_X = rng.standard_normal((15, 4))
        weights = np.ones(4)
        fold_indices = [np.arange(0, 15), np.arange(15, 30)]

        result = compute_aoa(train_X, new_X, weights, fold_indices)

        assert result["di"].shape == (15,)

    def test_aoa_consistent_with_threshold(self):
        """AOA mask should be True where DI <= threshold."""
        rng = np.random.default_rng(42)
        train_X = rng.standard_normal((60, 3))
        new_X = rng.standard_normal((30, 3))
        weights = np.ones(3)
        fold_indices = [np.arange(0, 20), np.arange(20, 40), np.arange(40, 60)]

        result = compute_aoa(train_X, new_X, weights, fold_indices)

        expected_mask = result["di"] <= result["threshold"]
        np.testing.assert_array_equal(result["aoa"], expected_mask)

    def test_training_point_inside_aoa(self):
        """A new point identical to a training point should be inside AOA."""
        rng = np.random.default_rng(42)
        train_X = rng.standard_normal((40, 3))
        # New point is identical to first training point
        new_X = train_X[:1].copy()
        weights = np.ones(3)
        fold_indices = [np.arange(0, 20), np.arange(20, 40)]

        result = compute_aoa(train_X, new_X, weights, fold_indices)

        assert result["aoa"][0] is np.True_

    def test_distant_point_outside_aoa(self):
        """A point very far from training should be outside AOA."""
        rng = np.random.default_rng(42)
        train_X = rng.standard_normal((40, 3))
        # Extremely distant point
        new_X = np.array([[1e6, 1e6, 1e6]])
        weights = np.ones(3)
        fold_indices = [np.arange(0, 20), np.arange(20, 40)]

        result = compute_aoa(train_X, new_X, weights, fold_indices)

        assert result["aoa"][0] is np.False_

    def test_does_not_mutate_inputs(self):
        """Input arrays must not be modified."""
        rng = np.random.default_rng(42)
        train_X = rng.standard_normal((30, 3))
        new_X = rng.standard_normal((10, 3))
        weights = np.ones(3)
        fold_indices = [np.arange(0, 15), np.arange(15, 30)]

        train_copy = train_X.copy()
        new_copy = new_X.copy()
        weights_copy = weights.copy()

        compute_aoa(train_X, new_X, weights, fold_indices)

        np.testing.assert_array_equal(train_X, train_copy)
        np.testing.assert_array_equal(new_X, new_copy)
        np.testing.assert_array_equal(weights, weights_copy)

    def test_chunked_matches_unchunked(self):
        """Chunked processing should produce identical results to single-batch."""
        rng = np.random.default_rng(42)
        train_X = rng.standard_normal((50, 4))
        new_X = rng.standard_normal((100, 4))
        weights = np.ones(4)
        fold_indices = [np.arange(0, 25), np.arange(25, 50)]

        result_single = compute_aoa(train_X, new_X, weights, fold_indices, chunk_size=0)
        result_chunked = compute_aoa(train_X, new_X, weights, fold_indices, chunk_size=30)

        np.testing.assert_allclose(result_single["di"], result_chunked["di"])
        np.testing.assert_array_equal(result_single["aoa"], result_chunked["aoa"])
        assert result_single["threshold"] == pytest.approx(result_chunked["threshold"])
