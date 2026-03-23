"""Edge case tests for core.py (M1)."""

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


class TestSingleTrainingPoint:
    def test_d_bar_raises(self):
        X = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="Need >= 2"):
            compute_d_bar_full(X)


class TestAllSameFold:
    def test_training_di_raises(self):
        X = np.array([[0.0], [1.0], [2.0]])
        folds = np.array([0, 0, 0])
        with pytest.raises(ValueError, match="folds"):
            compute_training_di(X, folds, 1.0)


class TestAllWeightsZero:
    def test_d_bar_zero_raises(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        weights = np.array([0.0, 0.0])
        X_w = apply_importance_weights(X, weights)
        with pytest.raises(ValueError, match="d_bar is zero"):
            compute_d_bar_full(X_w)


class TestOneFeatureOnly:
    def test_1d_distances_hand_verifiable(self):
        X = np.array([[0.0], [5.0], [10.0]])
        d_bar = compute_d_bar_full(X)
        # distances: 5, 10, 5 → mean = 20/3
        assert_allclose(d_bar, 20.0 / 3.0)

        folds = np.array([0, 1, 0])
        di = compute_training_di(X, folds, d_bar)
        # DI[0] (fold 0): nearest in fold 1 = 5, d=5 → 5/(20/3) = 15/20 = 0.75
        # DI[1] (fold 1): nearest in fold 0 = 0 or 10, d=5 → 0.75
        # DI[2] (fold 0): nearest in fold 1 = 5, d=5 → 0.75
        assert_allclose(di, [0.75, 0.75, 0.75])


class TestHighDimensional:
    def test_100_features_works(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 100))
        d_bar = compute_d_bar_full(X)
        assert d_bar > 0
        folds = np.tile([0, 1], 10)
        di = compute_training_di(X, folds, d_bar)
        assert di.shape == (20,)
        assert np.all(di >= 0)


class TestAllTrainingDIIdentical:
    def test_threshold_equals_value(self):
        di = np.array([0.42, 0.42, 0.42, 0.42, 0.42])
        threshold = compute_threshold(di)
        assert_allclose(threshold, 0.42)


class TestPredictionAtExactTrainingLocation:
    def test_di_is_zero(self):
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((15, 4))
        tree = build_kdtree(X_train)
        d_bar = compute_d_bar_full(X_train)
        di = compute_prediction_di(X_train, tree, d_bar)
        assert_allclose(di, np.zeros(15), atol=1e-10)
