"""Tests for aggregate.py (M5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from src.aoa.aggregate import (
    _aggregate_group,
    aggregate_climatological,
    aggregate_overall,
    aggregate_yearly,
)


def _make_monthly_summary(year: int, month: int, lat: float = 50.0, lon: float = 10.0, **kwargs) -> pd.DataFrame:
    defaults = {
        "latitude": [lat],
        "longitude": [lon],
        "median_DI": [0.5],
        "mean_DI": [0.5],
        "std_DI": [0.1],
        "frac_inside_aoa": [0.8],
        "n_timestamps": [30],
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


def _write_monthly(tmp_path, year, month, **kwargs):
    d = tmp_path / "monthly"
    d.mkdir(parents=True, exist_ok=True)
    df = _make_monthly_summary(year, month, **kwargs)
    path = d / f"di_monthly_{year}_{month:02d}.parquet"
    df.to_parquet(path, index=False)
    return path


class TestAggregateYearly:
    def test_12_months_to_yearly(self, tmp_path):
        for m in range(1, 13):
            _write_monthly(tmp_path, 2015, m, median_DI=[m * 0.1])
        out_dir = tmp_path / "yearly"
        paths = aggregate_yearly(tmp_path / "monthly", out_dir, [2015])
        assert len(paths) == 1
        df = pd.read_parquet(paths[0])
        assert len(df) == 1
        assert_allclose(df["n_timestamps"].iloc[0], 30 * 12)

    def test_missing_year_skips(self, tmp_path):
        _write_monthly(tmp_path, 2015, 1)
        out_dir = tmp_path / "yearly"
        paths = aggregate_yearly(tmp_path / "monthly", out_dir, [2099])
        assert len(paths) == 0


class TestAggregateClimatological:
    def test_all_januaries(self, tmp_path):
        for y in [2015, 2016, 2017]:
            _write_monthly(tmp_path, y, 1, median_DI=[y - 2014.0])
        out_dir = tmp_path / "clim"
        paths = aggregate_climatological(tmp_path / "monthly", out_dir)
        assert len(paths) == 1  # only January exists
        df = pd.read_parquet(paths[0])
        # median of [1.0, 2.0, 3.0] = 2.0
        assert_allclose(df["median_DI"].iloc[0], 2.0)


class TestAggregateOverall:
    def test_single_summary(self, tmp_path):
        for y in [2015, 2016]:
            for m in range(1, 4):
                _write_monthly(tmp_path, y, m)
        out_path = tmp_path / "overall.parquet"
        result = aggregate_overall(tmp_path / "monthly", out_path)
        assert result.exists()
        df = pd.read_parquet(result)
        assert len(df) == 1
        assert_allclose(df["n_timestamps"].iloc[0], 30 * 6)


class TestFracInsideAOABoundaries:
    def test_all_inside(self, tmp_path):
        _write_monthly(tmp_path, 2015, 1, frac_inside_aoa=[1.0])
        out = tmp_path / "yearly"
        paths = aggregate_yearly(tmp_path / "monthly", out, [2015])
        df = pd.read_parquet(paths[0])
        assert_allclose(df["frac_inside_aoa"].iloc[0], 1.0)

    def test_all_outside(self, tmp_path):
        _write_monthly(tmp_path, 2015, 1, frac_inside_aoa=[0.0])
        out = tmp_path / "yearly"
        paths = aggregate_yearly(tmp_path / "monthly", out, [2015])
        df = pd.read_parquet(paths[0])
        assert_allclose(df["frac_inside_aoa"].iloc[0], 0.0)


class TestSingleMonth:
    def test_yearly_equals_monthly(self, tmp_path):
        _write_monthly(
            tmp_path,
            2015,
            6,
            median_DI=[0.42],
            mean_DI=[0.45],
            std_DI=[0.05],
        )
        out = tmp_path / "yearly"
        paths = aggregate_yearly(tmp_path / "monthly", out, [2015])
        df = pd.read_parquet(paths[0])
        assert_allclose(df["median_DI"].iloc[0], 0.42, atol=1e-5)
        assert_allclose(df["mean_DI"].iloc[0], 0.45, atol=1e-5)
