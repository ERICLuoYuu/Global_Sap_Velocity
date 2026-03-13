"""Integration and scale tests for the AOA module."""

import numpy as np
import pytest
from src.aoa.aoa import compute_aoa


class TestEndToEnd:
    """End-to-end tests with synthetic data where AOA boundary is known."""

    def test_2d_cluster_separation(self):
        """Two well-separated clusters: points between them should be outside AOA."""
        rng = np.random.default_rng(42)
        # Training: tight cluster around (0, 0)
        train_X = rng.normal(0, 0.5, size=(100, 2))
        # New points: some near training, some far away
        near = rng.normal(0, 0.5, size=(20, 2))
        far = rng.normal(10, 0.5, size=(20, 2))
        new_X = np.vstack([near, far])

        weights = np.ones(2)
        fold_indices = [np.arange(0, 50), np.arange(50, 100)]

        result = compute_aoa(train_X, new_X, weights, fold_indices)

        # Near points should mostly be inside AOA
        assert result["aoa"][:20].sum() >= 15  # at least 75%
        # Far points should all be outside AOA
        assert result["aoa"][20:].sum() == 0

    def test_high_dimensional(self):
        """AOA should work correctly in high-dimensional feature space."""
        rng = np.random.default_rng(42)
        n_features = 30  # similar to real model feature count
        train_X = rng.standard_normal((200, n_features))
        new_X = rng.standard_normal((50, n_features))
        weights = np.abs(rng.standard_normal(n_features))
        fold_indices = [
            np.arange(0, 100),
            np.arange(100, 200),
        ]

        result = compute_aoa(train_X, new_X, weights, fold_indices)

        assert result["di"].shape == (50,)
        assert result["aoa"].shape == (50,)
        assert np.isfinite(result["threshold"])

    def test_weighted_features_change_aoa(self):
        """Different weight vectors should produce different AOA boundaries."""
        rng = np.random.default_rng(42)
        train_X = rng.standard_normal((80, 4))
        new_X = rng.standard_normal((40, 4))
        fold_indices = [np.arange(0, 40), np.arange(40, 80)]

        weights_uniform = np.ones(4)
        weights_skewed = np.array([10.0, 0.01, 0.01, 0.01])

        result_uniform = compute_aoa(train_X, new_X, weights_uniform, fold_indices)
        result_skewed = compute_aoa(train_X, new_X, weights_skewed, fold_indices)

        # DI values should differ with different weights
        assert not np.allclose(result_uniform["di"], result_skewed["di"])


class TestScalability:
    """Tests for memory and performance at moderate scale."""

    def test_moderate_scale(self):
        """Should handle 5000 training + 10000 new points without error."""
        rng = np.random.default_rng(42)
        n_train = 5000
        n_new = 10000
        n_features = 20

        train_X = rng.standard_normal((n_train, n_features))
        new_X = rng.standard_normal((n_new, n_features))
        weights = np.ones(n_features)
        fold_indices = [np.arange(i, n_train, 5) for i in range(5)]

        result = compute_aoa(train_X, new_X, weights, fold_indices, chunk_size=2000)

        assert result["di"].shape == (n_new,)
        assert result["aoa"].shape == (n_new,)
        assert np.isfinite(result["threshold"])

    def test_many_folds(self):
        """Should work with 10 CV folds matching the real pipeline."""
        rng = np.random.default_rng(42)
        n_train = 500
        train_X = rng.standard_normal((n_train, 10))
        new_X = rng.standard_normal((100, 10))
        weights = np.ones(10)
        # 10 folds like the real StratifiedGroupKFold
        fold_indices = [np.arange(i, n_train, 10) for i in range(10)]

        result = compute_aoa(train_X, new_X, weights, fold_indices)

        assert result["di"].shape == (100,)
        assert len(result["di"]) == 100
