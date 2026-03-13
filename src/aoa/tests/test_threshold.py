"""Tests for CV-based threshold computation."""

import numpy as np
import pytest

from src.aoa.threshold import (
    compute_cv_training_di,
    compute_threshold,
)


class TestComputeCVTrainingDI:
    """Tests for Step 5: Training DI using cross-validation folds."""

    def test_respects_fold_boundaries(self):
        """Each training point's DI must use only out-of-fold neighbors."""
        rng = np.random.default_rng(42)
        train_weighted = rng.standard_normal((20, 3))
        # 2-fold split: first 10 vs last 10
        fold_indices = [np.arange(10), np.arange(10, 20)]
        d_bar = 2.0

        di_train = compute_cv_training_di(train_weighted, fold_indices, d_bar)

        assert di_train.shape == (20,)
        assert np.all(di_train >= 0)

    def test_single_fold_raises(self):
        """Must have at least 2 folds for meaningful CV DI."""
        train_weighted = np.random.randn(10, 3)
        fold_indices = [np.arange(10)]
        d_bar = 1.0

        with pytest.raises(ValueError):
            compute_cv_training_di(train_weighted, fold_indices, d_bar)

    def test_two_fold_known_geometry(self):
        """With known geometry, verify DI values are correct."""
        # Fold 0: point at origin; Fold 1: point at (3, 4)
        train_weighted = np.array([[0.0, 0.0], [3.0, 4.0]])
        fold_indices = [np.array([0]), np.array([1])]
        d_bar = 5.0  # distance between them

        di_train = compute_cv_training_di(train_weighted, fold_indices, d_bar)

        # Each point's nearest out-of-fold neighbor is the other point
        # min_dist / d_bar = 5 / 5 = 1.0
        np.testing.assert_allclose(di_train, [1.0, 1.0])

    def test_output_length_matches_training(self):
        """Output should have one DI per training point."""
        rng = np.random.default_rng(0)
        n = 50
        train_weighted = rng.standard_normal((n, 4))
        fold_indices = [np.arange(0, 25), np.arange(25, 50)]
        d_bar = 2.5

        di_train = compute_cv_training_di(train_weighted, fold_indices, d_bar)

        assert len(di_train) == n

    def test_d_bar_zero_handled(self):
        """When d_bar is 0, should handle without crashing."""
        train_weighted = np.array([[0.0, 0.0], [1.0, 1.0]])
        fold_indices = [np.array([0]), np.array([1])]
        d_bar = 0.0

        di_train = compute_cv_training_di(train_weighted, fold_indices, d_bar)
        # d_bar=0, points differ → expect inf
        assert np.all(np.isinf(di_train))


class TestComputeThreshold:
    """Tests for Step 6: AOA threshold from training DI distribution."""

    def test_iqr_formula(self):
        """Threshold = Q75 + 1.5 * IQR of training DI values."""
        # Use a simple array where quartiles are easy to compute
        di_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        q75 = np.percentile(di_train, 75)
        q25 = np.percentile(di_train, 25)
        expected = q75 + 1.5 * (q75 - q25)

        threshold = compute_threshold(di_train)

        assert threshold == pytest.approx(expected)

    def test_identical_di_gives_finite_threshold(self):
        """If all training DI are identical, threshold should still be finite."""
        di_train = np.array([2.0, 2.0, 2.0, 2.0])

        threshold = compute_threshold(di_train)

        assert np.isfinite(threshold)
        assert threshold == pytest.approx(2.0)  # IQR=0, so threshold=Q75=2.0

    def test_threshold_greater_than_median(self):
        """Threshold should be >= median for any reasonable distribution."""
        rng = np.random.default_rng(42)
        di_train = np.abs(rng.standard_normal(100))

        threshold = compute_threshold(di_train)

        assert threshold >= np.median(di_train)

    def test_empty_input_raises(self):
        """Empty training DI should raise an error."""
        with pytest.raises(ValueError):
            compute_threshold(np.array([]))

    def test_single_value(self):
        """Single training DI value should return that value as threshold."""
        di_train = np.array([3.5])

        threshold = compute_threshold(di_train)

        assert threshold == pytest.approx(3.5)
