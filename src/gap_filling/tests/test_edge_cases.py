"""Phase 5 Round 1: Edge cases & boundary value tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_series(values, freq="h"):
    idx = pd.date_range("2020-01-01", periods=len(values), freq=freq)
    return pd.Series(values, index=idx)


class TestConfigBoundaries:
    def test_exactly_min_data_ml(self):
        from src.gap_filling.config import GapFillingConfig

        cfg = GapFillingConfig()
        assert cfg.eligible_groups(500) == ["A", "B", "C", "Ce"]

    def test_one_below_min_data_ml(self):
        from src.gap_filling.config import GapFillingConfig

        cfg = GapFillingConfig()
        assert cfg.eligible_groups(499) == ["A", "B"]

    def test_exactly_min_data_dl(self):
        from src.gap_filling.config import GapFillingConfig

        cfg = GapFillingConfig()
        assert cfg.eligible_groups(2000) == ["A", "B", "C", "Ce", "D", "De"]

    def test_one_below_min_data_dl(self):
        from src.gap_filling.config import GapFillingConfig

        cfg = GapFillingConfig()
        assert cfg.eligible_groups(1999) == ["A", "B", "C", "Ce"]

    def test_zero_column_length(self):
        from src.gap_filling.config import GapFillingConfig

        cfg = GapFillingConfig()
        assert cfg.eligible_groups(0) == ["A", "B"]

    def test_negative_column_length(self):
        from src.gap_filling.config import GapFillingConfig

        cfg = GapFillingConfig()
        assert cfg.eligible_groups(-100) == ["A", "B"]


class TestDetectorBoundaries:
    def test_single_nan(self):
        from src.gap_filling.detector import GapDetector

        s = _make_series([1.0, np.nan, 3.0])
        gaps = GapDetector.detect(s)
        assert len(gaps) == 1
        assert gaps[0].size_hours == 1

    def test_single_element_nan(self):
        from src.gap_filling.detector import GapDetector

        s = _make_series([np.nan])
        gaps = GapDetector.detect(s)
        assert len(gaps) == 1
        assert gaps[0].size_hours >= 1

    def test_single_element_valid(self):
        from src.gap_filling.detector import GapDetector

        s = _make_series([5.0])
        gaps = GapDetector.detect(s)
        assert gaps == []

    def test_alternating_nan_valid(self):
        from src.gap_filling.detector import GapDetector

        vals = [np.nan, 1.0, np.nan, 2.0, np.nan]
        s = _make_series(vals)
        gaps = GapDetector.detect(s)
        assert len(gaps) == 3
        for g in gaps:
            assert g.size_hours == 1

    def test_two_element_series_both_nan(self):
        from src.gap_filling.detector import GapDetector

        s = _make_series([np.nan, np.nan])
        gaps = GapDetector.detect(s)
        assert len(gaps) == 1
        assert gaps[0].size_hours == 2


class TestInterpolationEdgeCases:
    def test_fill_linear_all_nan(self):
        from src.gap_filling.methods.interpolation import fill_linear

        s = _make_series([np.nan, np.nan, np.nan])
        filled = fill_linear(s)
        assert filled.isna().all()

    def test_fill_linear_single_valid(self):
        from src.gap_filling.methods.interpolation import fill_linear

        s = _make_series([np.nan, 5.0, np.nan])
        filled = fill_linear(s)
        assert filled.isna().sum() == 0
        assert (filled >= 0).all()

    def test_fill_nearest_preserves_values(self):
        from src.gap_filling.methods.interpolation import fill_nearest

        vals = [1.0, np.nan, 3.0, np.nan, 5.0]
        s = _make_series(vals)
        filled = fill_nearest(s)
        assert filled.iloc[0] == 1.0
        assert filled.iloc[2] == 3.0
        assert filled.iloc[4] == 5.0

    def test_fill_cubic_large_gap(self):
        from src.gap_filling.methods.interpolation import fill_cubic

        vals = [1.0, 2.0] + [np.nan] * 200 + [3.0, 4.0]
        s = _make_series(vals)
        filled = fill_cubic(s)
        assert filled.isna().sum() == 0

    def test_fill_akima_two_points(self):
        from src.gap_filling.methods.interpolation import fill_akima

        s = _make_series([1.0, np.nan, np.nan, 4.0])
        filled = fill_akima(s)
        assert filled.isna().sum() == 0


class TestStatisticalEdgeCases:
    def test_mdv_all_missing_single_hour(self):
        from src.gap_filling.methods.statistical import fill_mdv

        idx = pd.date_range("2020-01-01", periods=72, freq="h")
        vals = np.ones(72)
        vals[0::24] = np.nan
        s = pd.Series(vals, index=idx)
        filled = fill_mdv(s)
        assert filled.isna().sum() == 0

    def test_rolling_mean_window_larger_than_series(self):
        from src.gap_filling.methods.statistical import fill_rolling_mean

        s = _make_series([1.0, np.nan, 3.0])
        filled = fill_rolling_mean(s, window=100)
        assert filled.isna().sum() == 0

    def test_stl_all_nan_series(self):
        from src.gap_filling.methods.statistical import fill_stl

        s = _make_series([np.nan] * 100)
        filled = fill_stl(s)
        assert len(filled) == 100


class TestMLEdgeCases:
    def test_build_features_very_short_series(self):
        from src.gap_filling.methods.ml import build_ml_features

        s = _make_series([1.0, 2.0, 3.0])
        features = build_ml_features(s, n_lags=24)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 3

    def test_fit_rf_insufficient_data(self):
        from src.gap_filling.methods.ml import fit_rf

        vals = [1.0] * 30 + [np.nan] * 20
        s = _make_series(vals)
        result = fit_rf(s)
        assert result is None

    def test_predict_rf_with_none_model(self):
        from src.gap_filling.methods.ml import predict_rf

        vals = [1.0, np.nan, 3.0, np.nan, 5.0]
        s = _make_series(vals)
        filled = predict_rf(s, cached=None)
        assert filled.isna().sum() == 0


class TestSelectorBoundaries:
    def _make_lookup_csv(self, tmpdir):
        rows = []
        rng = np.random.default_rng(42)
        methods = ["A_linear", "B_mdv", "C_rf"]
        for method in methods:
            group = method.split("_")[0]
            for gs in [1, 6, 24]:
                for seg_len in [1000, 3000]:
                    for rep in range(3):
                        n_pts = gs
                        true_vals = rng.uniform(1, 10, n_pts)
                        noise = 0.1 if "linear" in method else 0.5
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
                                "env_features": False,
                                "n_points": n_pts,
                                "ss_res": float(np.sum((true_vals - pred_vals) ** 2)),
                                "sum_abs_err": float(np.sum(np.abs(true_vals - pred_vals))),
                                "sum_true": float(np.sum(true_vals)),
                                "sum_true_sq": float(np.sum(true_vals**2)),
                            }
                        )
        df = pd.DataFrame(rows)
        path = tmpdir / "benchmark_test.csv"
        df.to_csv(path, index=False)
        return path

    def test_selector_zero_column_length(self, tmp_path):
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.selector import MethodSelector

        csv_path = self._make_lookup_csv(tmp_path)
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.3)
        sel = MethodSelector(cfg, target="sap")
        result = sel.select(column_length=0, env_available=False)
        assert result.non_env_method is None

    def test_selector_threshold_exactly_one(self, tmp_path):
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.selector import MethodSelector

        csv_path = self._make_lookup_csv(tmp_path)
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=1.0)
        sel = MethodSelector(cfg, target="sap")
        result = sel.select(column_length=5000, env_available=False)
        assert result.non_env_method is None or result.max_non_env_gap_h >= 0

    def test_selector_threshold_zero(self, tmp_path):
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.selector import MethodSelector

        csv_path = self._make_lookup_csv(tmp_path)
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.0)
        sel = MethodSelector(cfg, target="sap")
        result = sel.select(column_length=5000, env_available=False)
        assert result.non_env_method is not None
