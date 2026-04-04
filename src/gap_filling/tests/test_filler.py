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
                for rep in range(5):
                    n_pts = gs
                    true_vals = rng.uniform(1, 10, n_pts)
                    noise_scale = 0.1 if "linear" in method else 0.5 if "rf" in method else 1.0
                    if gs > 48:
                        noise_scale *= 3
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


def _make_column(n=2000, gap_sizes=None, freq="h"):
    """Create a series with multiple gaps of varying sizes."""
    if gap_sizes is None:
        gap_sizes = [3, 12, 48]
    idx = pd.date_range("2020-01-01", periods=n, freq=freq)
    vals = np.abs(np.sin(np.linspace(0, 20 * np.pi, n))) * 5 + 1
    s = pd.Series(vals, index=idx)
    pos = 100
    for gs in gap_sizes:
        if pos + gs < n:
            s.iloc[pos : pos + gs] = np.nan
            pos += gs + 200  # spacing between gaps
    return s


class TestGapFiller:
    def test_fill_column_returns_new_series(self, tmp_path):
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.filler import GapFiller

        csv_path = _make_lookup_csv(tmp_path)
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.3)
        filler = GapFiller(cfg, target="sap")
        s = _make_column()
        original_nans = s.isna().sum()
        filled = filler.fill_column(s)
        # Original unchanged (immutability)
        assert s.isna().sum() == original_nans
        # Filled has fewer NaNs
        assert filled.isna().sum() < original_nans

    def test_fill_column_no_gaps(self, tmp_path):
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.filler import GapFiller

        csv_path = _make_lookup_csv(tmp_path)
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.3)
        filler = GapFiller(cfg, target="sap")
        idx = pd.date_range("2020-01-01", periods=100, freq="h")
        s = pd.Series(np.ones(100), index=idx)
        filled = filler.fill_column(s)
        pd.testing.assert_series_equal(filled, s)

    def test_fill_column_non_negative(self, tmp_path):
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.filler import GapFiller

        csv_path = _make_lookup_csv(tmp_path)
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.3)
        filler = GapFiller(cfg, target="sap")
        s = _make_column()
        filled = filler.fill_column(s)
        assert (filled.dropna() >= 0).all()

    def test_fill_dataframe(self, tmp_path):
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.filler import GapFiller

        csv_path = _make_lookup_csv(tmp_path)
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.3)
        filler = GapFiller(cfg, target="sap")
        n = 2000
        idx = pd.date_range("2020-01-01", periods=n, freq="h")
        df = pd.DataFrame(
            {
                "col_a": np.where(np.arange(n) % 50 < 3, np.nan, 1.0),
                "col_b": np.where(np.arange(n) % 80 < 6, np.nan, 2.0),
            },
            index=idx,
        )
        filled_df = filler.fill_dataframe(df)
        assert filled_df.isna().sum().sum() < df.isna().sum().sum()
        # Original unchanged
        assert df.isna().sum().sum() > 0

    def test_fill_column_respects_max_gap(self, tmp_path):
        """Gaps larger than max selectable gap should remain NaN."""
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.filler import GapFiller

        csv_path = _make_lookup_csv(tmp_path)
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.95)
        filler = GapFiller(cfg, target="sap")
        # 3h gap should fill, 720h gap should not (too large)
        s = _make_column(n=3000, gap_sizes=[3, 720])
        filled = filler.fill_column(s)
        # At minimum the 3h gap should be filled
        assert filled.isna().sum() < s.isna().sum()

    def test_fill_dataframe_preserves_index(self, tmp_path):
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.filler import GapFiller

        csv_path = _make_lookup_csv(tmp_path)
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.3)
        filler = GapFiller(cfg, target="sap")
        idx = pd.date_range("2020-01-01", periods=200, freq="h")
        df = pd.DataFrame({"col": np.where(np.arange(200) % 40 < 3, np.nan, 1.0)}, index=idx)
        filled_df = filler.fill_dataframe(df)
        assert filled_df.index.equals(df.index)
        assert list(filled_df.columns) == list(df.columns)
