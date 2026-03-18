"""Tests for pooled R²/RMSE/MAE metrics — validates against sklearn."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from notebooks.gap_benchmark import compute_metrics, compute_pooled_metrics


def _make_replicate_df(all_true, all_pred, gap_size=24):
    """Build a DataFrame of per-replicate rows from lists of arrays."""
    rows = []
    for i, (tv, pv) in enumerate(zip(all_true, all_pred)):
        met = compute_metrics(tv, pv)
        met.update(
            {
                "site": f"S{i % 10}",
                "replicate": i,
                "gap_size": gap_size,
                "time_scale": "hourly",
                "method": "test",
                "group": "D",
            }
        )
        rows.append(met)
    return pd.DataFrame(rows)


class TestPooledMetricsMatchSklearn:
    """Pooled R²/RMSE/MAE must match sklearn exactly."""

    def test_24h_gap_pooled_r2(self):
        np.random.seed(42)
        all_t, all_p = [], []
        for _ in range(200):
            t = np.clip(5 + 3 * np.sin(np.linspace(0, 2 * np.pi, 24)) + np.random.normal(0, 0.5, 24), 0, None)
            p = np.clip(t + np.random.normal(0, 0.3, 24), 0, None)
            all_t.append(t)
            all_p.append(p)
        df = _make_replicate_df(all_t, all_p)
        pooled = compute_pooled_metrics(df)
        flat_t, flat_p = np.concatenate(all_t), np.concatenate(all_p)
        assert abs(pooled["r2_pooled"] - r2_score(flat_t, flat_p)) < 1e-10

    def test_24h_gap_pooled_rmse(self):
        np.random.seed(42)
        all_t, all_p = [], []
        for _ in range(200):
            t = np.clip(5 + 3 * np.sin(np.linspace(0, 2 * np.pi, 24)) + np.random.normal(0, 0.5, 24), 0, None)
            p = np.clip(t + np.random.normal(0, 0.3, 24), 0, None)
            all_t.append(t)
            all_p.append(p)
        df = _make_replicate_df(all_t, all_p)
        pooled = compute_pooled_metrics(df)
        flat_t, flat_p = np.concatenate(all_t), np.concatenate(all_p)
        assert abs(pooled["rmse_pooled"] - np.sqrt(mean_squared_error(flat_t, flat_p))) < 1e-10

    def test_24h_gap_pooled_mae(self):
        np.random.seed(42)
        all_t, all_p = [], []
        for _ in range(200):
            t = np.clip(5 + 3 * np.sin(np.linspace(0, 2 * np.pi, 24)) + np.random.normal(0, 0.5, 24), 0, None)
            p = np.clip(t + np.random.normal(0, 0.3, 24), 0, None)
            all_t.append(t)
            all_p.append(p)
        df = _make_replicate_df(all_t, all_p)
        pooled = compute_pooled_metrics(df)
        flat_t, flat_p = np.concatenate(all_t), np.concatenate(all_p)
        assert abs(pooled["mae_pooled"] - mean_absolute_error(flat_t, flat_p)) < 1e-10


class TestPooled1hGap:
    """1h gaps (1 data point each) must now produce meaningful pooled R²."""

    def test_1h_gap_pooled_r2_sensible(self):
        np.random.seed(99)
        all_t, all_p = [], []
        for _ in range(500):
            t = np.array([np.random.uniform(0, 8)])
            p = np.clip(t + np.random.normal(0, 0.3, 1), 0, None)
            all_t.append(t)
            all_p.append(p)
        df = _make_replicate_df(all_t, all_p, gap_size=1)
        pooled = compute_pooled_metrics(df)
        # Pooled R² should be high (good predictions, small noise)
        assert pooled["r2_pooled"] > 0.9, f"Expected >0.9, got {pooled['r2_pooled']}"
        # Must match sklearn
        flat_t, flat_p = np.concatenate(all_t), np.concatenate(all_p)
        assert abs(pooled["r2_pooled"] - r2_score(flat_t, flat_p)) < 1e-10

    def test_1h_old_r2_was_broken(self):
        """Per-replicate R² for 1h gaps should be ~0 (broken denominator)."""
        np.random.seed(99)
        all_t, all_p = [], []
        for _ in range(500):
            t = np.array([np.random.uniform(0, 8)])
            p = np.clip(t + np.random.normal(0, 0.3, 1), 0, None)
            all_t.append(t)
            all_p.append(p)
        df = _make_replicate_df(all_t, all_p, gap_size=1)
        # Old per-replicate mean R² should be near 0 (1 point → ss_tot=0 → R²=0)
        assert abs(df["r2"].mean()) < 0.1, f"Expected ~0, got {df['r2'].mean()}"


class TestPooledEdgeCases:
    """Edge cases for pooled metric computation."""

    def test_all_nan_rows(self):
        df = pd.DataFrame(
            [
                {
                    "n_points": np.nan,
                    "ss_res": np.nan,
                    "sum_abs_err": np.nan,
                    "sum_true": np.nan,
                    "sum_true_sq": np.nan,
                },
            ]
        )
        result = compute_pooled_metrics(df)
        assert np.isnan(result["r2_pooled"])
        assert result["n_total"] == 0

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["n_points", "ss_res", "sum_abs_err", "sum_true", "sum_true_sq"])
        result = compute_pooled_metrics(df)
        assert np.isnan(result["r2_pooled"])

    def test_constant_true_values(self):
        """All true values identical → ss_tot=0 → R² = 0.0 (not -inf)."""
        all_t = [np.array([5.0, 5.0, 5.0])] * 100
        all_p = [np.array([5.1, 4.9, 5.0])] * 100
        df = _make_replicate_df(all_t, all_p, gap_size=3)
        pooled = compute_pooled_metrics(df)
        assert pooled["r2_pooled"] == 0.0  # ss_tot = 0, ss_res > 0

    def test_perfect_predictions(self):
        """Exact predictions → R² = 1.0."""
        all_t = [np.array([1.0, 5.0, 3.0])] * 100
        all_p = [np.array([1.0, 5.0, 3.0])] * 100
        df = _make_replicate_df(all_t, all_p, gap_size=3)
        pooled = compute_pooled_metrics(df)
        assert abs(pooled["r2_pooled"] - 1.0) < 1e-10
        assert abs(pooled["rmse_pooled"]) < 1e-10

    def test_n_total_correct(self):
        """n_total should equal sum of all gap points."""
        all_t = [np.random.randn(24) for _ in range(50)]
        all_p = [t + 0.1 for t in all_t]
        df = _make_replicate_df(all_t, all_p, gap_size=24)
        pooled = compute_pooled_metrics(df)
        assert pooled["n_total"] == 50 * 24


class TestComputeMetricsSufficientStats:
    """compute_metrics must return all required keys."""

    def test_returns_all_keys(self):
        t = np.array([1.0, 2.0, 3.0])
        p = np.array([1.1, 2.1, 3.1])
        met = compute_metrics(t, p)
        required = {"rmse", "mae", "r2", "mape", "nse", "n_points", "ss_res", "sum_abs_err", "sum_true", "sum_true_sq"}
        assert required == set(met.keys())

    def test_sufficient_stats_values(self):
        t = np.array([1.0, 2.0, 3.0, 4.0])
        p = np.array([1.5, 2.5, 3.5, 4.5])
        met = compute_metrics(t, p)
        assert met["n_points"] == 4
        assert abs(met["ss_res"] - 1.0) < 1e-10  # 4 × 0.5² = 1.0
        assert abs(met["sum_abs_err"] - 2.0) < 1e-10  # 4 × 0.5 = 2.0
        assert abs(met["sum_true"] - 10.0) < 1e-10  # 1+2+3+4 = 10
        assert abs(met["sum_true_sq"] - 30.0) < 1e-10  # 1+4+9+16 = 30

    def test_nan_input_returns_nan(self):
        t = np.array([np.nan])
        p = np.array([1.0])
        met = compute_metrics(t, p)
        assert np.isnan(met["r2"])
        assert np.isnan(met["n_points"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
