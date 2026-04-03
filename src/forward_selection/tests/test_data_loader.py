"""Tests for data_loader.py — rh extensions and cache roundtrip."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forward_selection.data_loader import _add_rh_rolling_and_lag, load_cache


class TestAddRhRollingAndLag:
    def _make_df(self, n: int = 30, include_rh: bool = True) -> pd.DataFrame:
        rng = np.random.RandomState(42)
        df = pd.DataFrame({"ta": rng.randn(n), "vpd": rng.rand(n)})
        if include_rh:
            df["rh"] = rng.uniform(30, 90, n)
        return df

    def test_no_rh_column_returns_empty(self) -> None:
        df = self._make_df(include_rh=False)
        result_df, new_feats = _add_rh_rolling_and_lag(df)
        assert new_feats == []
        assert list(result_df.columns) == list(df.columns)

    def test_with_rh_adds_7_features(self) -> None:
        df = self._make_df()
        _, new_feats = _add_rh_rolling_and_lag(df)
        # 3 windows × 2 stats (mean, std) + 1 lag = 7
        assert len(new_feats) == 7

    def test_feature_names_correct(self) -> None:
        df = self._make_df()
        _, new_feats = _add_rh_rolling_and_lag(df)
        expected = [
            "rh_roll3d_mean",
            "rh_roll3d_std",
            "rh_roll7d_mean",
            "rh_roll7d_std",
            "rh_roll14d_mean",
            "rh_roll14d_std",
            "rh_lag1d",
        ]
        assert new_feats == expected

    def test_does_not_mutate_input(self) -> None:
        df = self._make_df()
        original_cols = list(df.columns)
        _add_rh_rolling_and_lag(df)
        assert list(df.columns) == original_cols

    def test_rolling_mean_values(self) -> None:
        df = pd.DataFrame({"rh": [10.0, 20.0, 30.0, 40.0, 50.0]})
        result_df, _ = _add_rh_rolling_and_lag(df)
        # 3-day rolling mean at index 2: mean(10, 20, 30) = 20
        assert result_df["rh_roll3d_mean"].iloc[2] == pytest.approx(20.0)

    def test_lag_shifts_by_one(self) -> None:
        df = pd.DataFrame({"rh": [100.0, 200.0, 300.0]})
        result_df, _ = _add_rh_rolling_and_lag(df)
        assert np.isnan(result_df["rh_lag1d"].iloc[0])
        assert result_df["rh_lag1d"].iloc[1] == pytest.approx(100.0)
        assert result_df["rh_lag1d"].iloc[2] == pytest.approx(200.0)

    def test_empty_dataframe(self) -> None:
        df = pd.DataFrame({"rh": pd.Series([], dtype=float)})
        result_df, new_feats = _add_rh_rolling_and_lag(df)
        assert len(new_feats) == 7
        assert len(result_df) == 0


class TestLoadCache:
    def _make_cache(self, tmp_path: Path) -> Path:
        cache_path = tmp_path / "test_cache.npz"
        rng = np.random.RandomState(42)
        np.savez_compressed(
            cache_path,
            X=rng.randn(100, 5).astype(np.float32),
            y=rng.rand(100).astype(np.float32),
            groups=np.repeat([0, 1, 2, 3, 4], 20),
            pfts_encoded=np.repeat([0, 1, 2], [40, 30, 30]),
            feature_names=np.array(["f1", "f2", "f3", "f4", "f5"]),
            pft_categories=np.array(["ENF", "DBF", "EBF"]),
        )
        return cache_path

    def test_loads_all_keys(self, tmp_path: Path) -> None:
        cache_path = self._make_cache(tmp_path)
        data = load_cache(cache_path)
        expected_keys = {"X", "y", "groups", "pfts_encoded", "feature_names", "pft_categories"}
        assert set(data.keys()) == expected_keys

    def test_shapes_match(self, tmp_path: Path) -> None:
        cache_path = self._make_cache(tmp_path)
        data = load_cache(cache_path)
        assert data["X"].shape == (100, 5)
        assert data["y"].shape == (100,)
        assert data["groups"].shape == (100,)
        assert data["pfts_encoded"].shape == (100,)

    def test_feature_names_are_list(self, tmp_path: Path) -> None:
        cache_path = self._make_cache(tmp_path)
        data = load_cache(cache_path)
        assert isinstance(data["feature_names"], list)
        assert data["feature_names"] == ["f1", "f2", "f3", "f4", "f5"]

    def test_pft_categories_are_list(self, tmp_path: Path) -> None:
        cache_path = self._make_cache(tmp_path)
        data = load_cache(cache_path)
        assert isinstance(data["pft_categories"], list)
        assert data["pft_categories"] == ["ENF", "DBF", "EBF"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_cache(tmp_path / "nonexistent.npz")

    def test_dtypes_preserved(self, tmp_path: Path) -> None:
        cache_path = self._make_cache(tmp_path)
        data = load_cache(cache_path)
        assert data["X"].dtype == np.float32
        assert data["y"].dtype == np.float32


class TestAddRhRollingEdgeCases:
    """Negative paths and boundary values for rh rolling."""

    def test_single_row(self) -> None:
        df = pd.DataFrame({"rh": [50.0]})
        result_df, new_feats = _add_rh_rolling_and_lag(df)
        assert len(new_feats) == 7
        # Rolling mean of single value = that value (min_periods=1)
        assert result_df["rh_roll3d_mean"].iloc[0] == pytest.approx(50.0)
        # Lag is NaN for single row
        assert np.isnan(result_df["rh_lag1d"].iloc[0])

    def test_nan_values_in_rh(self) -> None:
        df = pd.DataFrame({"rh": [10.0, np.nan, 30.0, 40.0, 50.0]})
        result_df, _ = _add_rh_rolling_and_lag(df)
        # Rolling mean at index 2 with NaN at index 1: mean(NaN, 30) with min_periods=1 = 30
        # (pandas skips NaN in rolling mean with min_periods=1)
        assert not np.isnan(result_df["rh_roll3d_mean"].iloc[2])

    def test_constant_rh(self) -> None:
        df = pd.DataFrame({"rh": [60.0] * 20})
        result_df, _ = _add_rh_rolling_and_lag(df)
        # Std of constant values = 0 (or NaN for window < min_periods)
        assert result_df["rh_roll3d_std"].iloc[10] == pytest.approx(0.0)
        # Mean of constant values = the constant
        assert result_df["rh_roll7d_mean"].iloc[10] == pytest.approx(60.0)
