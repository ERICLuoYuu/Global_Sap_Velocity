"""Shared fixtures for transpiration comparison tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def small_domain():
    """A tiny 5x5 spatial domain for fast tests."""
    lat = np.arange(45.05, 50.0, 1.0)  # 5 points
    lon = np.arange(5.05, 10.0, 1.0)  # 5 points
    return lat, lon


@pytest.fixture
def monthly_time():
    """36 months: 2016-01 to 2018-12."""
    return pd.date_range("2016-01-01", "2018-12-31", freq="MS")


@pytest.fixture
def daily_time():
    """3 years of daily data."""
    return pd.date_range("2016-01-01", "2018-12-31", freq="D")


@pytest.fixture
def make_dataset(small_domain, monthly_time):
    """Factory to create synthetic xr.Dataset with (time, lat, lon)."""

    def _make(
        var_name: str = "transpiration",
        seed: int = 42,
        add_seasonal: bool = True,
        scale: float = 1.0,
    ) -> xr.Dataset:
        lat, lon = small_domain
        rng = np.random.default_rng(seed)
        nt, nlat, nlon = len(monthly_time), len(lat), len(lon)

        data = rng.normal(loc=0.5, scale=0.2, size=(nt, nlat, nlon)).astype(np.float32)
        if add_seasonal:
            months = monthly_time.month.values
            seasonal = 0.3 * np.sin(2 * np.pi * (months - 3) / 12)
            data += seasonal[:, None, None] * scale

        ds = xr.Dataset(
            {var_name: (["time", "lat", "lon"], data)},
            coords={"time": monthly_time, "lat": lat, "lon": lon},
        )
        ds.attrs["product"] = var_name
        ds.attrs["units"] = "mm/day"
        return ds

    return _make


@pytest.fixture
def pft_map(small_domain):
    """Categorical PFT map on the small domain."""
    lat, lon = small_domain
    pft_codes = ["ENF", "EBF", "DBF", "MF", "WSA"]
    data = np.array(pft_codes * 5).reshape(len(lat), len(lon))
    return xr.DataArray(data, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
