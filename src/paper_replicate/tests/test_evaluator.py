"""Tests for metrics computation (Loritz et al. 2024 replication)."""

import numpy as np
import pytest

from src.paper_replicate.evaluator import (
    compute_kge,
    compute_mae,
    compute_nse,
)


class TestKGE:
    """Test KGE computation matches hydroeval."""

    def test_perfect_prediction(self):
        """KGE of identical pred and obs should be 1.0."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        kge, r, alpha, beta = compute_kge(obs, obs)
        assert abs(kge - 1.0) < 1e-6
        assert abs(r - 1.0) < 1e-6
        assert abs(alpha - 1.0) < 1e-6
        assert abs(beta - 1.0) < 1e-6

    def test_known_values(self):
        """Test KGE against manually computed values."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        kge, r, alpha, beta = compute_kge(pred, obs)
        # Correlation should be 1.0 (perfect linear relationship)
        assert abs(r - 1.0) < 1e-6
        # Beta = mean_pred/mean_obs = 3.5/3.0
        assert abs(beta - 3.5 / 3.0) < 1e-6

    def test_negative_correlation(self):
        """KGE should be negative for anti-correlated predictions."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        kge, r, alpha, beta = compute_kge(pred, obs)
        assert r < 0

    def test_empty_arrays(self):
        """Empty arrays should return NaN."""
        kge, r, alpha, beta = compute_kge(np.array([]), np.array([]))
        assert np.isnan(kge)


class TestNSE:
    """Test NSE computation."""

    def test_perfect_prediction(self):
        """NSE of identical pred and obs should be 1.0."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        nse = compute_nse(obs, obs)
        assert abs(nse - 1.0) < 1e-6

    def test_mean_prediction(self):
        """NSE of mean prediction should be 0.0."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.full_like(obs, obs.mean())
        nse = compute_nse(pred, obs)
        assert abs(nse - 0.0) < 1e-6

    def test_empty_arrays(self):
        """Empty arrays should return NaN."""
        nse = compute_nse(np.array([]), np.array([]))
        assert np.isnan(nse)


class TestMAE:
    """Test MAE computation."""

    def test_perfect_prediction(self):
        """MAE of identical pred and obs should be 0.0."""
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mae = compute_mae(obs, obs)
        assert abs(mae) < 1e-10

    def test_known_values(self):
        """MAE should be mean of absolute differences."""
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([2.0, 3.0, 4.0])
        mae = compute_mae(pred, obs)
        assert abs(mae - 1.0) < 1e-10

    def test_empty_arrays(self):
        """Empty arrays should return NaN."""
        mae = compute_mae(np.array([]), np.array([]))
        assert np.isnan(mae)
