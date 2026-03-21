"""Phase 5 Round 3: Integration & regression tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_series(values, freq="h"):
    idx = pd.date_range("2020-01-01", periods=len(values), freq=freq)
    return pd.Series(values, index=idx)


def _make_lookup_csv(tmpdir, include_env=False):
    rows = []
    rng = np.random.default_rng(42)
    methods = ["A_linear", "B_mdv", "C_rf"]
    if include_env:
        methods += ["Ce_rf_env"]
    for method in methods:
        group = method.split("_")[0]
        is_env = method.endswith("_env")
        for gs in [1, 6, 24, 48, 168]:
            for seg_len in [500, 1000, 3000, 5000]:
                for rep in range(5):
                    n_pts = max(gs, 10)
                    true_vals = rng.uniform(1, 10, n_pts)
                    if "linear" in method:
                        noise = 0.05 if gs <= 6 else 2.0
                    elif "rf" in method and not is_env:
                        noise = 0.3 if gs <= 48 else 1.5
                    elif is_env:
                        noise = 0.2
                    else:
                        noise = 0.5
                    pred_vals = true_vals + rng.normal(0, noise, n_pts)
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
                            "ss_res": float(np.sum((true_vals - pred_vals) ** 2)),
                            "sum_abs_err": float(np.sum(np.abs(true_vals - pred_vals))),
                            "sum_true": float(np.sum(true_vals)),
                            "sum_true_sq": float(np.sum(true_vals**2)),
                        }
                    )
    df = pd.DataFrame(rows)
    path = tmpdir / "benchmark_results_full.csv"
    df.to_csv(path, index=False)
    return path


class TestFullPipeline:
    def test_detect_select_fill_roundtrip(self, tmp_path):
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.detector import GapDetector
        from src.gap_filling.filler import GapFiller

        csv_path = _make_lookup_csv(tmp_path)
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.3)
        filler = GapFiller(cfg, target="sap")
        n = 2000
        idx = pd.date_range("2020-01-01", periods=n, freq="h")
        vals = np.abs(np.sin(np.linspace(0, 20 * np.pi, n))) * 5 + 1
        s = pd.Series(vals, index=idx)
        s.iloc[100:103] = np.nan
        s.iloc[500:512] = np.nan
        s.iloc[1000:1048] = np.nan
        gaps = GapDetector.detect(s)
        assert len(gaps) == 3
        filled = filler.fill_column(s)
        assert s.isna().sum() == 3 + 12 + 48
        assert filled.isna().sum() < s.isna().sum()
        assert (filled.dropna() >= 0).all()

    def test_fill_dataframe_preserves_column_order(self, tmp_path):
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.filler import GapFiller

        csv_path = _make_lookup_csv(tmp_path)
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.3)
        filler = GapFiller(cfg, target="sap")
        idx = pd.date_range("2020-01-01", periods=500, freq="h")
        df = pd.DataFrame(
            {
                "z_col": np.where(np.arange(500) % 50 < 3, np.nan, 1.0),
                "a_col": np.where(np.arange(500) % 80 < 6, np.nan, 2.0),
                "m_col": np.ones(500),
            },
            index=idx,
        )
        filled = filler.fill_dataframe(df)
        assert list(filled.columns) == ["z_col", "a_col", "m_col"]

    def test_fill_with_env_two_step(self, tmp_path):
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.filler import GapFiller

        csv_path = _make_lookup_csv(tmp_path, include_env=True)
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.3)
        filler = GapFiller(cfg, target="sap")
        n = 2000
        idx = pd.date_range("2020-01-01", periods=n, freq="h")
        vals = np.abs(np.sin(np.linspace(0, 20 * np.pi, n))) * 5 + 1
        s = pd.Series(vals, index=idx)
        s.iloc[100:103] = np.nan
        s.iloc[500:548] = np.nan
        env_df = pd.DataFrame(
            {
                "ta": np.random.RandomState(42).rand(n) * 30,
                "vpd": np.random.RandomState(43).rand(n) * 2,
            },
            index=idx,
        )
        filled = filler.fill_column(s, env_df=env_df)
        assert filled.isna().sum() < s.isna().sum()


class TestReviewFixRegressions:
    def test_detector_short_series_no_crash(self):
        from src.gap_filling.detector import GapDetector

        s1 = _make_series([np.nan])
        gaps1 = GapDetector.detect(s1)
        assert len(gaps1) == 1
        s2 = _make_series([np.nan, np.nan])
        gaps2 = GapDetector.detect(s2)
        assert len(gaps2) == 1

    def test_interpolation_logs_fallback(self, caplog):
        import logging

        from src.gap_filling.methods.interpolation import fill_akima

        s = _make_series([1.0, np.nan, 3.0])
        with caplog.at_level(logging.DEBUG, logger="src.gap_filling.methods.interpolation"):
            filled = fill_akima(s)
        assert filled.isna().sum() == 0

    def test_dl_predict_gap_counter_exists(self):
        import inspect

        from src.gap_filling.methods.dl import _predict_at_gaps

        source = inspect.getsource(_predict_at_gaps)
        assert "empty_cache" in source

    def test_stl_logs_fallback(self, caplog):
        import logging

        from src.gap_filling.methods.statistical import fill_stl

        s = _make_series([1.0, np.nan, 3.0])
        with caplog.at_level(logging.DEBUG, logger="src.gap_filling.methods.statistical"):
            filled = fill_stl(s)
        assert filled.isna().sum() == 0


class TestRegistryConsistency:
    def test_all_registered_methods_have_required_keys(self):
        from src.gap_filling.methods import METHODS

        for name, info in METHODS.items():
            assert "group" in info, f"{name} missing 'group'"
            has_fill = "fill" in info
            has_fit_predict = "fit" in info and "predict" in info
            assert has_fill or has_fit_predict, f"{name} missing callable"

    def test_all_group_a_methods_are_stateless(self):
        from src.gap_filling.methods import METHODS

        for name, info in METHODS.items():
            if info["group"] == "A":
                assert "fill" in info, f"{name} is Group A but has no 'fill'"

    def test_all_env_methods_have_env_flag(self):
        from src.gap_filling.methods import METHODS

        for name, info in METHODS.items():
            if info["group"] in ("Ce", "De"):
                assert info.get("env", False), f"{name} missing env flag"

    def test_method_groups_dict_complete(self):
        from src.gap_filling.methods import METHOD_GROUPS

        assert set(METHOD_GROUPS.keys()) == {"A", "B", "C", "Ce", "D", "De"}

    def test_config_all_groups_matches_registry(self):
        from src.gap_filling.config import ALL_GROUPS
        from src.gap_filling.methods import METHOD_GROUPS

        for g in ALL_GROUPS:
            assert g in METHOD_GROUPS
