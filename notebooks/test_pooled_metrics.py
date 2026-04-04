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


class TestPooledLargeGaps:
    """Large gaps (168h, 720h) — pooled metrics on realistic gap sizes."""

    def test_720h_gap_matches_sklearn(self):
        """30-day gap: 720 points per replicate, 50 reps = 36000 pooled."""
        np.random.seed(77)
        all_t, all_p = [], []
        for _ in range(50):
            t = np.arange(720, dtype=float)
            t = np.clip(5 + 3 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.5, 720), 0, None)
            p = np.clip(t + np.random.normal(0, 1.0, 720), 0, None)
            all_t.append(t)
            all_p.append(p)
        df = _make_replicate_df(all_t, all_p, gap_size=720)
        pooled = compute_pooled_metrics(df)
        flat_t, flat_p = np.concatenate(all_t), np.concatenate(all_p)
        assert abs(pooled["r2_pooled"] - r2_score(flat_t, flat_p)) < 1e-9
        assert abs(pooled["rmse_pooled"] - np.sqrt(mean_squared_error(flat_t, flat_p))) < 1e-9
        assert pooled["n_total"] == 50 * 720

    def test_168h_gap_matches_sklearn(self):
        """7-day gap: realistic diurnal + weekly pattern."""
        np.random.seed(88)
        all_t, all_p = [], []
        for _ in range(100):
            hours = np.arange(168, dtype=float)
            diurnal = 3 * np.sin(2 * np.pi * hours / 24)
            weekly = 0.5 * np.sin(2 * np.pi * hours / 168)
            t = np.clip(5 + diurnal + weekly + np.random.normal(0, 0.4, 168), 0, None)
            p = np.clip(t + np.random.normal(0, 0.5, 168), 0, None)
            all_t.append(t)
            all_p.append(p)
        df = _make_replicate_df(all_t, all_p, gap_size=168)
        pooled = compute_pooled_metrics(df)
        flat_t, flat_p = np.concatenate(all_t), np.concatenate(all_p)
        assert abs(pooled["r2_pooled"] - r2_score(flat_t, flat_p)) < 1e-9


class TestPooledMixedNaN:
    """Some replicates fail (NaN rows) — pooled metrics should ignore them."""

    def test_mixed_valid_and_nan_rows(self):
        """50% of rows are NaN (method failures), pooled uses only valid ones."""
        np.random.seed(55)
        all_t, all_p = [], []
        rows = []
        for i in range(200):
            t = np.clip(5 + np.random.normal(0, 2, 24), 0, None)
            p = np.clip(t + np.random.normal(0, 0.3, 24), 0, None)
            if i % 2 == 0:
                # Valid row
                met = compute_metrics(t, p)
                all_t.append(t)
                all_p.append(p)
            else:
                # Simulated failure — all NaN
                met = {
                    k: np.nan
                    for k in [
                        "rmse",
                        "mae",
                        "r2",
                        "mape",
                        "nse",
                        "n_points",
                        "ss_res",
                        "sum_abs_err",
                        "sum_true",
                        "sum_true_sq",
                    ]
                }
            met.update(
                {
                    "site": f"S{i % 10}",
                    "replicate": i,
                    "gap_size": 24,
                    "time_scale": "hourly",
                    "method": "test",
                    "group": "D",
                }
            )
            rows.append(met)
        df = pd.DataFrame(rows)
        pooled = compute_pooled_metrics(df)
        flat_t, flat_p = np.concatenate(all_t), np.concatenate(all_p)
        assert abs(pooled["r2_pooled"] - r2_score(flat_t, flat_p)) < 1e-10
        assert pooled["n_total"] == 100 * 24  # only 100 valid rows

    def test_single_valid_row_among_nans(self):
        """Only 1 valid row out of many — should still compute valid metrics."""
        t = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
        p = np.array([1.1, 4.9, 3.2, 6.8, 2.1])
        rows = []
        # 1 valid
        met = compute_metrics(t, p)
        met.update(
            {"site": "S0", "replicate": 0, "gap_size": 5, "time_scale": "hourly", "method": "test", "group": "D"}
        )
        rows.append(met)
        # 9 NaN
        for i in range(1, 10):
            nan_met = {
                k: np.nan
                for k in [
                    "rmse",
                    "mae",
                    "r2",
                    "mape",
                    "nse",
                    "n_points",
                    "ss_res",
                    "sum_abs_err",
                    "sum_true",
                    "sum_true_sq",
                ]
            }
            nan_met.update(
                {"site": f"S{i}", "replicate": i, "gap_size": 5, "time_scale": "hourly", "method": "test", "group": "D"}
            )
            rows.append(nan_met)
        df = pd.DataFrame(rows)
        pooled = compute_pooled_metrics(df)
        # Should match per-replicate R² since only 1 valid row
        assert abs(pooled["r2_pooled"] - r2_score(t, p)) < 1e-10
        assert pooled["n_total"] == 5


class TestPooledSingleReplicate:
    """Single replicate — pooled should equal per-replicate metrics."""

    def test_single_rep_r2_matches(self):
        t = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 2.0, 4.0, 6.0])
        p = np.array([1.2, 2.8, 5.3, 6.7, 9.1, 2.2, 3.9, 6.2])
        df = _make_replicate_df([t], [p], gap_size=8)
        pooled = compute_pooled_metrics(df)
        per_rep = compute_metrics(t, p)
        # With single replicate, pooled R² == per-replicate R²
        assert abs(pooled["r2_pooled"] - per_rep["r2"]) < 1e-10
        assert abs(pooled["rmse_pooled"] - per_rep["rmse"]) < 1e-10
        assert abs(pooled["mae_pooled"] - per_rep["mae"]) < 1e-10


class TestPooledMultiSite:
    """Pooling across multiple sites with different value ranges."""

    def test_multi_site_different_scales(self):
        """Sites with very different sap flow magnitudes (tropical vs boreal)."""
        np.random.seed(33)
        all_t, all_p = [], []
        for _ in range(50):
            # "Tropical" site: high sap flow
            t_trop = np.clip(15 + np.random.normal(0, 3, 12), 0, None)
            p_trop = np.clip(t_trop + np.random.normal(0, 0.5, 12), 0, None)
            all_t.append(t_trop)
            all_p.append(p_trop)
        for _ in range(50):
            # "Boreal" site: low sap flow
            t_bor = np.clip(1.0 + np.random.normal(0, 0.3, 12), 0, None)
            p_bor = np.clip(t_bor + np.random.normal(0, 0.1, 12), 0, None)
            all_t.append(t_bor)
            all_p.append(p_bor)
        df = _make_replicate_df(all_t, all_p, gap_size=12)
        pooled = compute_pooled_metrics(df)
        flat_t, flat_p = np.concatenate(all_t), np.concatenate(all_p)
        assert abs(pooled["r2_pooled"] - r2_score(flat_t, flat_p)) < 1e-10
        # R² should be high since both sites predict well
        assert pooled["r2_pooled"] > 0.9


class TestPooledNumericalStability:
    """Large values / many replicates — check floating point accumulation."""

    def test_large_values_no_precision_loss(self):
        """True values around 1000 — sum_true_sq will be ~10^6 per point."""
        np.random.seed(11)
        all_t, all_p = [], []
        for _ in range(500):
            t = 1000 + np.random.normal(0, 50, 24)
            p = t + np.random.normal(0, 5, 24)
            all_t.append(t)
            all_p.append(p)
        df = _make_replicate_df(all_t, all_p, gap_size=24)
        pooled = compute_pooled_metrics(df)
        flat_t, flat_p = np.concatenate(all_t), np.concatenate(all_p)
        # Allow slightly larger tolerance for large values (catastrophic cancellation)
        assert abs(pooled["r2_pooled"] - r2_score(flat_t, flat_p)) < 1e-6

    def test_near_zero_values(self):
        """True values near zero (nighttime) — the original problem domain."""
        np.random.seed(22)
        all_t, all_p = [], []
        for _ in range(500):
            t = np.clip(np.random.exponential(0.1, 6), 0, None)  # mostly near 0
            p = np.clip(t + np.random.normal(0, 0.02, 6), 0, None)
            all_t.append(t)
            all_p.append(p)
        df = _make_replicate_df(all_t, all_p, gap_size=6)
        pooled = compute_pooled_metrics(df)
        flat_t, flat_p = np.concatenate(all_t), np.concatenate(all_p)
        assert abs(pooled["r2_pooled"] - r2_score(flat_t, flat_p)) < 1e-8
        # Even near-zero data should give meaningful pooled R²
        assert pooled["r2_pooled"] > 0.5


class TestPooledNSEConsistency:
    """NSE should always equal R² in our implementation."""

    def test_nse_equals_r2(self):
        np.random.seed(44)
        all_t = [np.random.uniform(1, 10, 24) for _ in range(100)]
        all_p = [t + np.random.normal(0, 0.5, 24) for t in all_t]
        df = _make_replicate_df(all_t, all_p, gap_size=24)
        pooled = compute_pooled_metrics(df)
        assert pooled["nse_pooled"] == pooled["r2_pooled"]


class TestComputeMetricsBackwardCompat:
    """Per-replicate metrics must remain unchanged by the refactor."""

    def test_rmse_formula(self):
        t = np.array([1.0, 2.0, 3.0])
        p = np.array([1.5, 2.5, 3.5])
        met = compute_metrics(t, p)
        expected_rmse = np.sqrt(np.mean((t - p) ** 2))
        assert abs(met["rmse"] - expected_rmse) < 1e-10

    def test_mae_formula(self):
        t = np.array([1.0, 2.0, 3.0])
        p = np.array([1.5, 2.5, 3.5])
        met = compute_metrics(t, p)
        expected_mae = np.mean(np.abs(t - p))
        assert abs(met["mae"] - expected_mae) < 1e-10

    def test_r2_matches_sklearn_per_replicate(self):
        t = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        p = np.array([1.2, 2.8, 5.3, 7.1, 8.7])
        met = compute_metrics(t, p)
        assert abs(met["r2"] - r2_score(t, p)) < 1e-10

    def test_mape_skips_zeros(self):
        t = np.array([0.0, 0.0, 5.0, 10.0])
        p = np.array([0.1, 0.1, 6.0, 9.0])
        met = compute_metrics(t, p)
        # MAPE should only use the 2 non-zero values: |1/5| + |1/10| = 0.3, /2 = 15%
        expected = (abs(1.0 / 5.0) + abs(1.0 / 10.0)) / 2 * 100
        assert abs(met["mape"] - expected) < 1e-10

    def test_empty_after_nan_filter(self):
        t = np.array([np.nan, np.nan])
        p = np.array([1.0, 2.0])
        met = compute_metrics(t, p)
        assert np.isnan(met["rmse"])
        assert np.isnan(met["ss_res"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
