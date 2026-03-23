"""Property-based invariant tests for core.py (M1)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.aoa.core import (
    apply_importance_weights,
    build_kdtree,
    compute_d_bar_full,
    compute_prediction_di,
    compute_threshold,
    compute_training_di,
    standardize_features,
)

SEEDS = [42, 123, 456, 789, 1024]


class TestDINonNegative:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_training_di_non_negative(self, seed):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((30, 5))
        folds = np.tile([0, 1, 2], 10)
        d_bar = compute_d_bar_full(X)
        di = compute_training_di(X, folds, d_bar)
        assert np.all(di >= 0)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_prediction_di_non_negative(self, seed):
        rng = np.random.default_rng(seed)
        X_train = rng.standard_normal((20, 4))
        X_new = rng.standard_normal((10, 4))
        tree = build_kdtree(X_train)
        d_bar = compute_d_bar_full(X_train)
        di = compute_prediction_di(X_new, tree, d_bar)
        assert np.all(di >= 0)


class TestDIZeroForIdentical:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_identical_point_has_zero_di(self, seed):
        rng = np.random.default_rng(seed)
        X_train = rng.standard_normal((20, 3))
        tree = build_kdtree(X_train)
        d_bar = compute_d_bar_full(X_train)
        di = compute_prediction_di(X_train[:1], tree, d_bar)
        assert_allclose(di[0], 0.0, atol=1e-10)


class TestThresholdBounds:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_threshold_non_negative(self, seed):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((30, 5))
        folds = np.tile([0, 1, 2], 10)
        d_bar = compute_d_bar_full(X)
        di = compute_training_di(X, folds, d_bar)
        threshold = compute_threshold(di)
        assert threshold >= 0

    @pytest.mark.parametrize("seed", SEEDS)
    def test_threshold_leq_max_training_di(self, seed):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((30, 5))
        folds = np.tile([0, 1, 2], 10)
        d_bar = compute_d_bar_full(X)
        di = compute_training_di(X, folds, d_bar)
        threshold = compute_threshold(di)
        assert threshold <= np.max(di) + 1e-10


class TestFeaturePermutationInvariance:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_permute_features_same_di(self, seed):
        rng = np.random.default_rng(seed)
        X_train = rng.standard_normal((20, 4))
        X_new = rng.standard_normal((5, 4))
        means = X_train.mean(axis=0)
        stds = X_train.std(axis=0, ddof=1)
        weights = np.abs(rng.standard_normal(4)) + 0.1

        X_s = standardize_features(X_train, means, stds)
        X_sw = apply_importance_weights(X_s, weights)
        tree = build_kdtree(X_sw)
        d_bar = compute_d_bar_full(X_sw)
        X_new_s = standardize_features(X_new, means, stds)
        X_new_sw = apply_importance_weights(X_new_s, weights)
        di_original = compute_prediction_di(X_new_sw, tree, d_bar)

        perm = rng.permutation(4)
        X_train_p = X_train[:, perm]
        X_new_p = X_new[:, perm]
        means_p = means[perm]
        stds_p = stds[perm]
        weights_p = weights[perm]

        X_s_p = standardize_features(X_train_p, means_p, stds_p)
        X_sw_p = apply_importance_weights(X_s_p, weights_p)
        tree_p = build_kdtree(X_sw_p)
        d_bar_p = compute_d_bar_full(X_sw_p)
        X_new_s_p = standardize_features(X_new_p, means_p, stds_p)
        X_new_sw_p = apply_importance_weights(X_new_s_p, weights_p)
        di_permuted = compute_prediction_di(X_new_sw_p, tree_p, d_bar_p)

        assert_allclose(di_original, di_permuted, atol=1e-10)


class TestImportanceScaleInvariance:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_scale_weights_same_di(self, seed):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((20, 3))
        X_new = rng.standard_normal((5, 3))
        weights = np.abs(rng.standard_normal(3)) + 0.1

        X_sw1 = apply_importance_weights(X, weights)
        tree1 = build_kdtree(X_sw1)
        d_bar1 = compute_d_bar_full(X_sw1)
        X_new_sw1 = apply_importance_weights(X_new, weights)
        di1 = compute_prediction_di(X_new_sw1, tree1, d_bar1)

        scale = 7.5
        X_sw2 = apply_importance_weights(X, weights * scale)
        tree2 = build_kdtree(X_sw2)
        d_bar2 = compute_d_bar_full(X_sw2)
        X_new_sw2 = apply_importance_weights(X_new, weights * scale)
        di2 = compute_prediction_di(X_new_sw2, tree2, d_bar2)

        assert_allclose(di1, di2, atol=1e-10)


class TestMonotonicity:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_farther_point_higher_di(self, seed):
        rng = np.random.default_rng(seed)
        X_train = rng.standard_normal((20, 3))
        tree = build_kdtree(X_train)
        d_bar = compute_d_bar_full(X_train)
        center = X_train.mean(axis=0)
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        close_point = (center + 0.1 * direction).reshape(1, -1)
        far_point = (center + 100.0 * direction).reshape(1, -1)
        di_close = compute_prediction_di(close_point, tree, d_bar)
        di_far = compute_prediction_di(far_point, tree, d_bar)
        assert di_far[0] > di_close[0]


class TestRowOrderInvarianceOfDBar:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_shuffle_rows_same_d_bar(self, seed):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((25, 4))
        d1 = compute_d_bar_full(X)
        perm = rng.permutation(len(X))
        d2 = compute_d_bar_full(X[perm])
        assert_allclose(d1, d2)


class TestDuplicateTrainingPoint:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_adding_duplicate_doesnt_increase_nn_distance(self, seed):
        """Adding a duplicate doesn't increase nearest-neighbor distance.

        Note: DI = distance / d_bar CAN increase because duplicates reduce
        d_bar (zero-distance pairs). But raw NN distance cannot increase.
        """
        rng = np.random.default_rng(seed)
        X_train = rng.standard_normal((20, 3))
        X_new = rng.standard_normal((5, 3))
        tree1 = build_kdtree(X_train)
        dist1, _ = tree1.query(X_new, k=1)

        dup_idx = rng.integers(0, 20)
        X_train2 = np.vstack([X_train, X_train[dup_idx : dup_idx + 1]])
        tree2 = build_kdtree(X_train2)
        dist2, _ = tree2.query(X_new, k=1)

        assert np.all(dist2 <= dist1 + 1e-10)


class TestDIEqualsOneAtDBar:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_distance_equals_d_bar_gives_di_one(self, seed):
        # Construct a point exactly d_bar away from nearest training point
        rng = np.random.default_rng(seed)
        X_train = np.array([[0.0], [10.0]])
        tree = build_kdtree(X_train)
        d_bar = compute_d_bar_full(X_train)  # = 10.0
        # Point at d_bar distance from nearest: x = 10 + 10 = 20
        # Nearest = 10, distance = 10, DI = 10/10 = 1.0
        X_new = np.array([[20.0]])
        di = compute_prediction_di(X_new, tree, d_bar)
        assert_allclose(di[0], 1.0, atol=1e-10)
