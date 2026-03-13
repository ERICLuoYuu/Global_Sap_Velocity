"""Tests for dissimilarity index computation."""

import numpy as np
import pytest

from src.aoa.dissimilarity import (
    compute_di,
    compute_mean_training_distance,
    standardize_predictors,
    weight_by_importance,
)


class TestStandardizePredictors:
    """Tests for Step 1: Standardization using training-only statistics."""

    def test_uses_training_statistics_only(self):
        """Standardization must use mean/std from training data, not new data."""
        train_X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        new_X = np.array([[100.0, 200.0]])

        train_scaled, new_scaled, stats = standardize_predictors(train_X, new_X)

        # Training mean should be [3, 4], std should be [~1.63, ~1.63]
        np.testing.assert_allclose(train_scaled.mean(axis=0), 0.0, atol=1e-10)
        # With ddof=1 standardization (matching R's scale()), sample std is 1.0
        np.testing.assert_allclose(train_scaled.std(axis=0, ddof=1), 1.0, atol=1e-10)

        # New data should be scaled with training stats, NOT centered at 0
        assert new_scaled[0, 0] > 10  # 100 is far from mean=3

    def test_zero_variance_predictor_handled(self):
        """Predictors with zero variance should not cause division by zero."""
        train_X = np.array([[1.0, 5.0], [1.0, 3.0], [1.0, 7.0]])
        new_X = np.array([[2.0, 4.0]])

        train_scaled, new_scaled, stats = standardize_predictors(train_X, new_X)

        # Zero-variance column should be set to 0
        np.testing.assert_allclose(train_scaled[:, 0], 0.0)
        np.testing.assert_allclose(new_scaled[:, 0], 0.0)

    def test_output_shapes(self):
        """Output shapes must match input shapes."""
        train_X = np.random.randn(100, 5)
        new_X = np.random.randn(20, 5)

        train_scaled, new_scaled, stats = standardize_predictors(train_X, new_X)

        assert train_scaled.shape == (100, 5)
        assert new_scaled.shape == (20, 5)

    def test_does_not_mutate_inputs(self):
        """Input arrays must not be modified."""
        train_X = np.array([[1.0, 2.0], [3.0, 4.0]])
        new_X = np.array([[5.0, 6.0]])
        train_copy = train_X.copy()
        new_copy = new_X.copy()

        standardize_predictors(train_X, new_X)

        np.testing.assert_array_equal(train_X, train_copy)
        np.testing.assert_array_equal(new_X, new_copy)

    def test_nan_in_train_raises(self):
        """Should raise ValueError if train_X contains NaN."""
        train_X = np.array([[1.0, np.nan], [3.0, 4.0]])
        new_X = np.array([[5.0, 6.0]])

        with pytest.raises(ValueError, match="train_X contains NaN"):
            standardize_predictors(train_X, new_X)

    def test_nan_in_new_raises(self):
        """Should raise ValueError if new_X contains NaN."""
        train_X = np.array([[1.0, 2.0], [3.0, 4.0]])
        new_X = np.array([[5.0, np.nan]])

        with pytest.raises(ValueError, match="new_X contains NaN"):
            standardize_predictors(train_X, new_X)


class TestWeightByImportance:
    """Tests for Step 2: Weighting by variable importance."""

    def test_correct_weighting(self):
        """Weighted values should equal importance * scaled values."""
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        weights = np.array([0.5, 0.3, 0.2])

        result = weight_by_importance(X, weights)

        expected = X * weights
        np.testing.assert_allclose(result, expected)

    def test_zero_weight_zeroes_column(self):
        """A feature with zero importance should be zeroed out."""
        X = np.array([[10.0, 20.0], [30.0, 40.0]])
        weights = np.array([0.0, 1.0])

        result = weight_by_importance(X, weights)

        np.testing.assert_allclose(result[:, 0], 0.0)
        np.testing.assert_allclose(result[:, 1], X[:, 1])

    def test_mismatched_dimensions_raises(self):
        """Weight vector length must match number of features."""
        X = np.array([[1.0, 2.0]])
        weights = np.array([0.5, 0.3, 0.2])

        with pytest.raises(ValueError):
            weight_by_importance(X, weights)

    def test_does_not_mutate_input(self):
        """Input array must not be modified."""
        X = np.array([[1.0, 2.0]])
        X_copy = X.copy()
        weights = np.array([0.5, 0.3])

        weight_by_importance(X, weights)
        np.testing.assert_array_equal(X, X_copy)


class TestComputeMeanTrainingDistance:
    """Tests for Step 3: Average pairwise training distance."""

    def test_known_distances(self):
        """Test with simple geometry where distances are known."""
        # Two points at (0,0) and (3,4) -> distance = 5
        train_weighted = np.array([[0.0, 0.0], [3.0, 4.0]])

        d_bar = compute_mean_training_distance(train_weighted)

        assert d_bar == pytest.approx(5.0)

    def test_identical_points_gives_zero(self):
        """All identical training points should give d_bar = 0."""
        train_weighted = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

        d_bar = compute_mean_training_distance(train_weighted)

        assert d_bar == pytest.approx(0.0)

    def test_positive_result(self):
        """d_bar should be non-negative for any input."""
        rng = np.random.default_rng(42)
        train_weighted = rng.standard_normal((50, 10))

        d_bar = compute_mean_training_distance(train_weighted)

        assert d_bar >= 0

    def test_subsampling_large_input(self):
        """When n > _MAX_PDIST_SAMPLES, should subsample and still return valid d_bar."""
        rng = np.random.default_rng(42)
        # 6000 > _MAX_PDIST_SAMPLES (5000)
        train_weighted = rng.standard_normal((6000, 3))

        d_bar = compute_mean_training_distance(train_weighted)

        assert d_bar > 0
        assert np.isfinite(d_bar)


class TestComputeDI:
    """Tests for Step 4: Dissimilarity Index for new points."""

    def test_identical_to_training_gives_zero(self):
        """DI should be 0 when new point is identical to a training point."""
        train_weighted = np.array([[0.0, 0.0], [3.0, 4.0]])
        new_weighted = np.array([[0.0, 0.0]])  # identical to first training point
        d_bar = 5.0  # known

        di = compute_di(new_weighted, train_weighted, d_bar)

        assert di[0] == pytest.approx(0.0)

    def test_di_non_negative(self):
        """DI values must always be non-negative."""
        rng = np.random.default_rng(42)
        train = rng.standard_normal((50, 5))
        new = rng.standard_normal((20, 5))
        d_bar = 3.0

        di = compute_di(new, train, d_bar)

        assert np.all(di >= 0)

    def test_farther_point_has_higher_di(self):
        """Points farther from training should have higher DI."""
        train_weighted = np.array([[0.0, 0.0]])
        near = np.array([[1.0, 0.0]])
        far = np.array([[100.0, 0.0]])
        d_bar = 1.0  # dummy

        di_near = compute_di(near, train_weighted, d_bar)
        di_far = compute_di(far, train_weighted, d_bar)

        assert di_far[0] > di_near[0]

    def test_output_shape(self):
        """DI output shape should match number of new points."""
        train = np.random.randn(30, 4)
        new = np.random.randn(15, 4)
        d_bar = 2.0

        di = compute_di(new, train, d_bar)

        assert di.shape == (15,)

    def test_d_bar_zero_handled(self):
        """When d_bar is 0 (all training identical), DI should handle gracefully."""
        train = np.array([[1.0, 1.0], [1.0, 1.0]])
        new = np.array([[2.0, 2.0]])
        d_bar = 0.0

        # d_bar=0, new point differs from training → expect inf
        di = compute_di(new, train, d_bar)
        assert np.isinf(di[0])

    def test_1d_input_raises(self):
        """A 1-D input should raise ValueError, not be silently reshaped."""
        train = np.array([[0.0, 0.0], [3.0, 4.0]])
        new = np.array([0.0, 0.0])  # 1-D, not 2-D
        d_bar = 5.0

        with pytest.raises(ValueError, match="2-D"):
            compute_di(new, train, d_bar)
