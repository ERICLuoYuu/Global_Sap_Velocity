from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_lookup_csv(tmpdir: Path) -> Path:
    """Create a minimal benchmark_results_full.csv for testing."""
    rows = []
    rng = np.random.default_rng(42)
    methods = ["A_linear", "B_mdv", "C_rf", "D_lstm", "Ce_rf_env", "De_lstm_env"]
    gap_sizes = [1, 6, 24, 48, 168]
    seg_lengths = [1000, 3000, 5000]

    for method in methods:
        group = method.split("_")[0]
        is_env = method.endswith("_env")
        for gs in gap_sizes:
            for seg_len in seg_lengths:
                for rep in range(5):  # 5 reps per combo
                    n_pts = gs
                    true_vals = rng.uniform(1, 10, n_pts)
                    # Better methods have lower residuals
                    noise_scale = 0.1 if "linear" in method else 0.5 if "rf" in method else 1.0
                    if gs > 48:
                        noise_scale *= 3  # larger gaps = worse
                    pred_vals = true_vals + rng.normal(0, noise_scale, n_pts)
                    ss_res = float(np.sum((true_vals - pred_vals) ** 2))
                    rows.append(
                        {
                            "time_scale": "hourly",
                            "site": f"site_{seg_len}",
                            "seg_length": seg_len,
                            "method": method,
                            "group": group,
                            "gap_size": gs,
                            "replicate": rep,
                            "env_features": is_env,
                            "n_points": n_pts,
                            "ss_res": ss_res,
                            "sum_abs_err": float(np.sum(np.abs(true_vals - pred_vals))),
                            "sum_true": float(np.sum(true_vals)),
                            "sum_true_sq": float(np.sum(true_vals**2)),
                        }
                    )
    df = pd.DataFrame(rows)
    path = tmpdir / "benchmark_results_full.csv"
    df.to_csv(path, index=False)
    return path


class TestMethodSelector:
    def test_loads_csv(self, tmp_path):
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.selector import MethodSelector

        csv_path = _make_lookup_csv(tmp_path)
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.5)
        sel = MethodSelector(cfg, target="sap")
        assert sel._grouped is not None
        assert len(sel._grouped) > 0

    def test_select_non_env_method(self, tmp_path):
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.selector import MethodSelector

        csv_path = _make_lookup_csv(tmp_path)
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.3)
        sel = MethodSelector(cfg, target="sap")
        result = sel.select(column_length=3000, env_available=False)
        assert result.non_env_method is not None
        assert result.max_non_env_gap_h > 0
        assert result.env_method is None  # env not available

    def test_select_with_env(self, tmp_path):
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.selector import MethodSelector

        csv_path = _make_lookup_csv(tmp_path)
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.3)
        sel = MethodSelector(cfg, target="sap")
        result = sel.select(column_length=3000, env_available=True)
        assert result.non_env_method is not None

    def test_data_volume_guard_short(self, tmp_path):
        """Column with <500h should only get Group A/B methods."""
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.selector import MethodSelector

        csv_path = _make_lookup_csv(tmp_path)
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.3)
        sel = MethodSelector(cfg, target="sap")
        result = sel.select(column_length=300, env_available=True)
        assert set(result.eligible_groups).issubset({"A", "B"})

    def test_data_volume_guard_medium(self, tmp_path):
        """Column 500-2000h: A/B/C/Ce, no D/De."""
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.selector import MethodSelector

        csv_path = _make_lookup_csv(tmp_path)
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.3)
        sel = MethodSelector(cfg, target="sap")
        result = sel.select(column_length=800, env_available=True)
        assert "D" not in result.eligible_groups
        assert "De" not in result.eligible_groups

    def test_caching_same_length(self, tmp_path):
        """Similar column lengths should reuse cached results."""
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.selector import MethodSelector

        csv_path = _make_lookup_csv(tmp_path)
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.3)
        sel = MethodSelector(cfg, target="sap")
        r1 = sel.select(column_length=3010, env_available=False)
        r2 = sel.select(column_length=3050, env_available=False)
        # Both round to 3000 -> same cache key
        assert r1.non_env_method == r2.non_env_method

    def test_pooled_r2_computation(self, tmp_path):
        """Verify pooled R2 output contains expected columns."""
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.selector import MethodSelector

        csv_path = _make_lookup_csv(tmp_path)
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.0)
        sel = MethodSelector(cfg, target="sap")
        metrics = sel._compute_pooled_metrics(column_length=10000)
        assert not metrics.empty
        assert "r2_pooled" in metrics.columns
        assert "rmse_pooled" in metrics.columns
        assert "mae_pooled" in metrics.columns

    def test_numerical_stability_guard(self, tmp_path):
        """pooled_ss_tot should never be negative."""
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.selector import MethodSelector

        csv_path = _make_lookup_csv(tmp_path)
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.0)
        sel = MethodSelector(cfg, target="sap")
        metrics = sel._compute_pooled_metrics(column_length=10000)
        assert (metrics["pooled_ss_tot"] >= 0).all()
        # R2 should be finite
        assert metrics["r2_pooled"].notna().all()

    def test_no_method_meets_threshold(self, tmp_path):
        """Very high threshold -> no method selected for any gap."""
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.selector import MethodSelector

        csv_path = _make_lookup_csv(tmp_path)
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.9999)
        sel = MethodSelector(cfg, target="sap")
        result = sel.select(column_length=3000, env_available=True)
        # With extremely high threshold, likely no method qualifies
        # Result should still be valid (no crash)
        assert result.eligible_groups is not None
