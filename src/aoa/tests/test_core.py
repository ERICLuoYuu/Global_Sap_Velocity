"""Known-answer unit tests for core.py (M1 — RED phase)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.aoa.core import (
    apply_importance_weights,
    build_kdtree,
    compute_d_bar_full,
    compute_d_bar_sample,
    compute_prediction_di,
    compute_threshold,
    compute_training_di,
    standardize_features,
)


# ---------------------------------------------------------------------------
# standardize_features
# ---------------------------------------------------------------------------
class TestStandardizeFeatures:
    def test_known_output(self):
        X = np.array([[2.0, 4.0], [4.0, 6.0]])
        means = np.array([3.0, 5.0])
        stds = np.array([1.0, 1.0])
        result = standardize_features(X, means, stds)
        expected = np.array([[-1.0, -1.0], [1.0, 1.0]])
        assert_allclose(result, expected)

    def test_zero_variance_col_becomes_zero(self):
        X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        means = np.array([2.0, 5.0])
        stds = np.array([1.0, 0.0])
        result = standardize_features(X, means, stds)
        assert_allclose(result[:, 1], [0.0, 0.0, 0.0])
        assert_allclose(result[:, 0], [-1.0, 0.0, 1.0])

    def test_new_data_uses_training_stats(self):
        X_train = np.array([[0.0], [10.0]])
        means = np.array([5.0])
        stds = np.array([5.0])
        X_new = np.array([[15.0]])
        result = standardize_features(X_new, means, stds)
        assert_allclose(result, [[2.0]])

    def test_single_row(self):
        X = np.array([[3.0, 7.0]])
        means = np.array([1.0, 2.0])
        stds = np.array([2.0, 5.0])
        result = standardize_features(X, means, stds)
        assert_allclose(result, [[1.0, 1.0]])


# ---------------------------------------------------------------------------
# apply_importance_weights
# ---------------------------------------------------------------------------
class TestApplyImportanceWeights:
    def test_weights_multiply_columns(self):
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        weights = np.array([2.0, 0.0, 1.0])
        result = apply_importance_weights(X, weights)
        expected = np.array([[2.0, 0.0, 3.0], [8.0, 0.0, 6.0]])
        assert_allclose(result, expected)

    def test_zero_weight_makes_invisible(self):
        X = np.array([[100.0, 200.0]])
        weights = np.array([0.0, 0.0])
        result = apply_importance_weights(X, weights)
        assert_allclose(result, [[0.0, 0.0]])

    def test_equal_weights(self):
        X = np.array([[1.0, 2.0]])
        weights = np.array([3.0, 3.0])
        result = apply_importance_weights(X, weights)
        assert_allclose(result, [[3.0, 6.0]])


# ---------------------------------------------------------------------------
# compute_d_bar_full
# ---------------------------------------------------------------------------
class TestComputeDBarFull:
    def test_equilateral_triangle(self):
        # 3 points forming equilateral triangle, side=2
        X = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, np.sqrt(3)]])
        d_bar = compute_d_bar_full(X)
        # 3 pairwise distances: 2, 2, 2 → mean = 2.0
        assert_allclose(d_bar, 2.0, atol=1e-10)

    def test_two_points(self):
        X = np.array([[0.0], [5.0]])
        d_bar = compute_d_bar_full(X)
        assert_allclose(d_bar, 5.0)

    def test_identical_points_raises(self):
        X = np.array([[1.0, 2.0], [1.0, 2.0]])
        with pytest.raises(ValueError, match="d_bar is zero"):
            compute_d_bar_full(X)

    def test_row_shuffle_invariance(self):
        rng = np.random.default_rng(123)
        X = rng.standard_normal((20, 3))
        d1 = compute_d_bar_full(X)
        perm = rng.permutation(len(X))
        d2 = compute_d_bar_full(X[perm])
        assert_allclose(d1, d2)


# ---------------------------------------------------------------------------
# compute_d_bar_sample
# ---------------------------------------------------------------------------
class TestComputeDBarSample:
    def test_sample_equals_n_gives_full(self):
        X = np.array([[0.0], [3.0], [6.0]])
        full = compute_d_bar_full(X)
        sampled = compute_d_bar_sample(X, sample_size=3)
        assert_allclose(full, sampled)

    def test_large_sample_fallback(self):
        X = np.array([[0.0], [1.0]])
        sampled = compute_d_bar_sample(X, sample_size=100)
        assert_allclose(sampled, 1.0)


# ---------------------------------------------------------------------------
# compute_training_di
# ---------------------------------------------------------------------------
class TestComputeTrainingDI:
    def test_worked_example_4_points(self):
        """4 points in 1D, 2 folds — hand-computed DI values."""
        X_w = np.array([[0.0], [10.0], [3.0], [12.0]])
        folds = np.array([0, 0, 1, 1])
        # d_bar = mean(10, 3, 12, 7, 2, 9) = 43/6
        d_bar = 43.0 / 6.0
        di = compute_training_di(X_w, folds, d_bar)
        # DI[0]: nearest in fold 1 = point 2, d=3 → 3/(43/6) = 18/43
        # DI[1]: nearest in fold 1 = point 3, d=2 → 2/(43/6) = 12/43
        # DI[2]: nearest in fold 0 = point 0, d=3 → 3/(43/6) = 18/43
        # DI[3]: nearest in fold 0 = point 1, d=2 → 2/(43/6) = 12/43
        expected = np.array([18 / 43, 12 / 43, 18 / 43, 12 / 43])
        assert_allclose(di, expected, atol=1e-10)

    def test_fold_constraint_verified(self):
        """Points close in same fold should NOT be each other's neighbor."""
        X_w = np.array([[0.0], [0.1], [100.0]])
        folds = np.array([0, 0, 1])
        d_bar = compute_d_bar_full(X_w)
        di = compute_training_di(X_w, folds, d_bar)
        # DI[0] and DI[1] must use point 2 (fold 1), not each other
        assert di[0] > 1.0  # far from fold 1
        assert di[1] > 1.0

    def test_same_d_bar_used(self):
        """DI should scale linearly with 1/d_bar."""
        X_w = np.array([[0.0], [5.0], [10.0]])
        folds = np.array([0, 1, 0])
        d_bar_a = 5.0
        d_bar_b = 10.0
        di_a = compute_training_di(X_w, folds, d_bar_a)
        di_b = compute_training_di(X_w, folds, d_bar_b)
        assert_allclose(di_a * d_bar_a, di_b * d_bar_b, atol=1e-10)

    def test_single_fold_raises(self):
        X_w = np.array([[0.0], [1.0]])
        folds = np.array([0, 0])
        with pytest.raises(ValueError, match="folds"):
            compute_training_di(X_w, folds, 1.0)


# ---------------------------------------------------------------------------
# compute_threshold
# ---------------------------------------------------------------------------
class TestComputeThreshold:
    def test_no_outliers_returns_max(self):
        di = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        # Q25=0.2, Q75=0.4, IQR=0.2, upper_whisker=0.7
        # All values <= 0.7 → threshold = 0.5
        assert_allclose(compute_threshold(di), 0.5)

    def test_with_outlier(self):
        di = np.array([0.1, 0.2, 0.3, 0.4, 2.0])
        # Q25=0.2, Q75=0.4, IQR=0.2, upper_whisker=0.7
        # Values <= 0.7: [0.1, 0.2, 0.3, 0.4] → threshold = 0.4
        assert_allclose(compute_threshold(di), 0.4)

    def test_all_identical(self):
        di = np.array([0.5, 0.5, 0.5, 0.5])
        # IQR=0, upper_whisker=0.5, threshold=0.5
        assert_allclose(compute_threshold(di), 0.5)

    def test_single_value(self):
        di = np.array([1.0])
        assert_allclose(compute_threshold(di), 1.0)


# ---------------------------------------------------------------------------
# compute_prediction_di
# ---------------------------------------------------------------------------
class TestComputePredictionDI:
    def test_identical_to_training_is_zero(self):
        X_train = np.array([[0.0], [5.0], [10.0]])
        tree = build_kdtree(X_train)
        X_new = np.array([[5.0]])
        di = compute_prediction_di(X_new, tree, d_bar=5.0)
        assert_allclose(di, [0.0], atol=1e-10)

    def test_far_point_greater_than_one(self):
        X_train = np.array([[0.0], [10.0]])
        tree = build_kdtree(X_train)
        d_bar = compute_d_bar_full(X_train)  # = 10.0
        X_new = np.array([[20.0]])  # distance to nearest = 10
        di = compute_prediction_di(X_new, tree, d_bar)
        assert_allclose(di, [1.0])

    def test_worked_example(self):
        """Match the plan's worked example."""
        X_train = np.array([[0.0], [10.0], [3.0], [12.0]])
        tree = build_kdtree(X_train)
        d_bar = 43.0 / 6.0
        # New point at x=20: nearest = 12, d=8 → DI = 8/(43/6) = 48/43
        # New point at x=3: nearest = 3, d=0 → DI = 0
        X_new = np.array([[20.0], [3.0]])
        di = compute_prediction_di(X_new, tree, d_bar)
        assert_allclose(di, [48 / 43, 0.0], atol=1e-10)


# ---------------------------------------------------------------------------
# build_kdtree
# ---------------------------------------------------------------------------
class TestBuildKDTree:
    def test_correct_nearest_neighbor(self):
        X = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 5.0]])
        tree = build_kdtree(X)
        dist, idx = tree.query([[0.0, 0.0]])
        assert_allclose(dist, [0.0], atol=1e-10)
        assert idx[0] == 0

    def test_exact_point_distance_zero(self):
        X = np.array([[1.0], [2.0], [3.0]])
        tree = build_kdtree(X)
        dist, _ = tree.query([[2.0]])
        assert_allclose(dist, [0.0], atol=1e-10)
