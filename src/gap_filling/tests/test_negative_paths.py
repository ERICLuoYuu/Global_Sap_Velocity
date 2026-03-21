"""Phase 5 Round 2: Negative paths & error handling tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_series(values, freq="h"):
    idx = pd.date_range("2020-01-01", periods=len(values), freq=freq)
    return pd.Series(values, index=idx)


class TestSelectorNegativePaths:
    def test_missing_csv_raises(self):
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.selector import MethodSelector

        cfg = GapFillingConfig(lookup_csv_sap="/nonexistent/path.csv")
        with pytest.raises((FileNotFoundError, OSError)):
            MethodSelector(cfg, target="sap")

    def test_empty_csv(self, tmp_path):
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.selector import MethodSelector

        csv_path = tmp_path / "empty.csv"
        csv_path.write_text(
            "time_scale,site,seg_length,method,group,gap_size,replicate,"
            "env_features,n_points,ss_res,sum_abs_err,sum_true,sum_true_sq\n"
        )
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.3)
        sel = MethodSelector(cfg, target="sap")
        result = sel.select(column_length=1000, env_available=False)
        assert result.non_env_method is None
        assert result.max_non_env_gap_h == 0

    def test_csv_missing_columns(self, tmp_path):
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.selector import MethodSelector

        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("col_a,col_b\n1,2\n")
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.3)
        with pytest.raises(KeyError):
            MethodSelector(cfg, target="sap")


class TestFillerNegativePaths:
    def _make_lookup_csv(self, tmpdir):
        rows = []
        rng = np.random.default_rng(42)
        for gs in [1, 6, 24]:
            for seg_len in [1000, 3000]:
                for rep in range(3):
                    n_pts = gs
                    true_vals = rng.uniform(1, 10, n_pts)
                    pred_vals = true_vals + rng.normal(0, 0.1, n_pts)
                    rows.append(
                        {
                            "time_scale": "hourly",
                            "site": f"site_{seg_len}",
                            "seg_length": seg_len,
                            "method": "A_linear",
                            "group": "A",
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

    def test_filler_with_missing_benchmark_csv(self):
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.filler import GapFiller

        cfg = GapFillingConfig(lookup_csv_sap="/nonexistent/path.csv")
        with pytest.raises((FileNotFoundError, OSError)):
            GapFiller(cfg, target="sap")

    def test_filler_all_nan_column(self, tmp_path):
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.filler import GapFiller

        csv_path = self._make_lookup_csv(tmp_path)
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.0)
        filler = GapFiller(cfg, target="sap")
        s = _make_series([np.nan] * 100)
        filled = filler.fill_column(s)
        assert len(filled) == 100

    def test_filler_empty_dataframe(self, tmp_path):
        from src.gap_filling.config import GapFillingConfig
        from src.gap_filling.filler import GapFiller

        csv_path = self._make_lookup_csv(tmp_path)
        cfg = GapFillingConfig(lookup_csv_sap=csv_path, threshold=0.3)
        filler = GapFiller(cfg, target="sap")
        df = pd.DataFrame(index=pd.DatetimeIndex([]))
        result = filler.fill_dataframe(df)
        assert len(result) == 0


class TestInterpolationNegativePaths:
    def test_fill_linear_inf_values(self):
        from src.gap_filling.methods.interpolation import fill_linear

        vals = [1.0, float("inf"), np.nan, 4.0]
        s = _make_series(vals)
        filled = fill_linear(s)
        assert len(filled) == 4

    def test_fill_cubic_constant_series(self):
        from src.gap_filling.methods.interpolation import fill_cubic

        vals = [5.0] * 50 + [np.nan] * 10 + [5.0] * 50
        s = _make_series(vals)
        filled = fill_cubic(s)
        assert filled.isna().sum() == 0
        assert abs(filled.iloc[55] - 5.0) < 1.0


class TestMLNegativePaths:
    def test_build_features_with_empty_env_df(self):
        from src.gap_filling.methods.ml import build_ml_features

        s = _make_series(np.random.rand(100))
        env_df = pd.DataFrame(index=s.index)
        features = build_ml_features(s, env_df=env_df)
        env_cols = [c for c in features.columns if c.startswith("env_")]
        assert len(env_cols) == 0

    def test_build_features_with_misaligned_env_df(self):
        from src.gap_filling.methods.ml import build_ml_features

        s = _make_series(np.random.rand(100))
        env_idx = pd.date_range("2021-01-01", periods=50, freq="h")
        env_df = pd.DataFrame({"ta": np.random.rand(50)}, index=env_idx)
        features = build_ml_features(s, env_df=env_df)
        assert "env_ta_lag_1" in features.columns

    def test_fit_rf_all_same_value(self):
        from src.gap_filling.methods.ml import fit_rf

        vals = [5.0] * 200 + [np.nan] * 50
        s = _make_series(vals)
        result = fit_rf(s)
        assert result is None or len(result) == 2
