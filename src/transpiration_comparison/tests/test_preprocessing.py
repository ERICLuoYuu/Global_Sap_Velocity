"""Tests for preprocessing modules: normalize, temporal, regrid."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.transpiration_comparison.config import SpatialDomain
from src.transpiration_comparison.preprocessing.normalize import (
    compute_anomaly,
    compute_climatology,
    normalize_seasonal_amplitude,
    zscore_normalize,
)
from src.transpiration_comparison.preprocessing.regrid import (
    _estimate_resolution,
    _fallback_interp,
    regrid_to_common,
)
from src.transpiration_comparison.preprocessing.temporal import (
    aggregate_to_daily,
    compute_annual_mean,
    compute_monthly_mean,
)

# ── Z-score normalize ──────────────────────────────────────────────


class TestZscoreNormalize:
    def test_basic_zscore(self, make_dataset):
        ds = make_dataset(var_name="transpiration", seed=1)
        result = zscore_normalize(ds, "transpiration")
        zvar = "transpiration_zscore"
        assert zvar in result.data_vars
        # Mean should be ~0 and std ~1 per pixel (within tolerance for 36 months)
        mean_per_pixel = result[zvar].mean(dim="time")
        std_per_pixel = result[zvar].std(dim="time")
        assert float(mean_per_pixel.mean()) == pytest.approx(0.0, abs=0.1)
        assert float(std_per_pixel.mean()) == pytest.approx(1.0, abs=0.15)

    def test_min_valid_mask(self):
        """Pixels with too few valid values should be NaN."""
        rng = np.random.default_rng(99)
        time = pd.date_range("2016-01-01", periods=10, freq="MS")
        # Use varying data so std > 0 for valid pixels
        data = rng.normal(1.0, 0.5, (10, 2, 2)).astype(np.float32)
        data[:, 0, 0] = np.nan  # pixel (0,0) has zero valid
        data[:5, 1, 1] = np.nan  # pixel (1,1) has 5 valid
        ds = xr.Dataset(
            {"v": (["time", "lat", "lon"], data)},
            coords={"time": time, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
        )
        result = zscore_normalize(ds, "v", min_valid=8)
        assert np.all(np.isnan(result["v_zscore"].values[:, 0, 0]))
        assert np.all(np.isnan(result["v_zscore"].values[:, 1, 1]))
        # pixel (0,1) and (1,0) have 10 valid with variance -> should be valid
        assert not np.all(np.isnan(result["v_zscore"].values[:, 0, 1]))

    def test_zero_variance_masked(self):
        """Constant time series should produce NaN (zero std)."""
        time = pd.date_range("2016-01-01", periods=36, freq="MS")
        data = np.full((36, 2, 2), 5.0, dtype=np.float32)
        ds = xr.Dataset(
            {"v": (["time", "lat", "lon"], data)},
            coords={"time": time, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
        )
        result = zscore_normalize(ds, "v", min_valid=30)
        assert np.all(np.isnan(result["v_zscore"].values))

    def test_attrs_set(self, make_dataset):
        ds = make_dataset()
        result = zscore_normalize(ds, "transpiration")
        attrs = result["transpiration_zscore"].attrs
        assert attrs["units"] == "dimensionless"
        assert "min_valid_days" in attrs


# ── Climatology ────────────────────────────────────────────────────


class TestComputeClimatology:
    def test_shape(self, make_dataset):
        ds = make_dataset()
        result = compute_climatology(ds, "transpiration")
        assert "transpiration_clim" in result.data_vars
        assert "month" in result.dims
        assert result.dims["month"] == 12

    def test_values_reasonable(self, make_dataset):
        ds = make_dataset(add_seasonal=True)
        result = compute_climatology(ds, "transpiration")
        clim = result["transpiration_clim"]
        # Seasonal cycle should show variation across months
        assert float(clim.max() - clim.min()) > 0.1


# ── Anomaly ────────────────────────────────────────────────────────


class TestComputeAnomaly:
    def test_anomaly_adds_variable(self, make_dataset):
        ds = make_dataset()
        result = compute_anomaly(ds, "transpiration")
        assert "transpiration_anomaly" in result.data_vars

    def test_anomaly_mean_near_zero(self, make_dataset):
        ds = make_dataset()
        result = compute_anomaly(ds, "transpiration")
        # Grouped anomaly should average ~0 per month
        monthly_mean = result["transpiration_anomaly"].groupby("time.month").mean(dim="time")
        assert float(abs(monthly_mean).max()) < 1e-5


# ── Seasonal amplitude normalization ──────────────────────────────


class TestNormalizeSeasonalAmplitude:
    def test_range_0_1(self, make_dataset):
        ds = make_dataset(add_seasonal=True)
        result = normalize_seasonal_amplitude(ds, "transpiration")
        norm = result["transpiration_norm"]
        assert float(norm.min()) >= -0.01  # allow float tolerance
        assert float(norm.max()) <= 1.01


# ── Temporal aggregation ──────────────────────────────────────────


class TestAggregateToDaily:
    def test_daily_passthrough(self, make_dataset):
        ds = make_dataset()
        result = aggregate_to_daily(ds, "daily")
        assert result is ds  # should return same object

    def test_hourly_to_daily(self):
        time = pd.date_range("2016-01-01", periods=48, freq="h")
        data = np.random.default_rng(0).random((48, 2, 2)).astype(np.float32)
        ds = xr.Dataset(
            {"transpiration": (["time", "lat", "lon"], data)},
            coords={"time": time, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
        )
        result = aggregate_to_daily(ds, "hourly")
        assert len(result.time) == 2  # 48h -> 2 days

    def test_3hourly_to_daily(self):
        time = pd.date_range("2016-01-01", periods=24, freq="3h")
        data = np.ones((24, 2, 2), dtype=np.float32)
        ds = xr.Dataset(
            {"transpiration": (["time", "lat", "lon"], data)},
            coords={"time": time, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
        )
        result = aggregate_to_daily(ds, "3-hourly")
        assert len(result.time) == 3  # 24 * 3h = 72h = 3 days

    def test_8day_interpolation(self):
        time = pd.date_range("2016-01-01", periods=5, freq="8D")
        data = np.arange(5, dtype=np.float32)[:, None, None] * np.ones((1, 2, 2))
        ds = xr.Dataset(
            {"transpiration": (["time", "lat", "lon"], data)},
            coords={"time": time, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
        )
        result = aggregate_to_daily(ds, "8-day")
        assert len(result.time) > len(ds.time)

    def test_unknown_freq_raises(self, make_dataset):
        ds = make_dataset()
        with pytest.raises(ValueError, match="Unknown source frequency"):
            aggregate_to_daily(ds, "weekly")


class TestComputeMonthlyMean:
    def test_monthly_reduces_time(self):
        time = pd.date_range("2016-01-01", periods=365, freq="D")
        data = np.ones((365, 2, 2), dtype=np.float32)
        ds = xr.Dataset(
            {"transpiration": (["time", "lat", "lon"], data)},
            coords={"time": time, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
        )
        result = compute_monthly_mean(ds)
        assert len(result.time) == 12


class TestComputeAnnualMean:
    def test_annual_reduces_time(self, make_dataset):
        ds = make_dataset()  # 36 months
        result = compute_annual_mean(ds)
        assert len(result.time) == 3  # 2016, 2017, 2018


# ── Regridding ────────────────────────────────────────────────────


class TestEstimateResolution:
    def test_regular_grid(self):
        ds = xr.Dataset(coords={"lat": np.arange(0, 10, 0.25), "lon": np.arange(0, 10, 0.25)})
        assert _estimate_resolution(ds) == pytest.approx(0.25)

    def test_single_point_default(self):
        ds = xr.Dataset(coords={"lat": [5.0], "lon": [10.0]})
        assert _estimate_resolution(ds) == 0.1


class TestRegridToCommon:
    def test_already_on_grid_skips(self):
        """0.1-degree input should skip regridding."""
        domain = SpatialDomain(lat_min=0, lat_max=5, lon_min=0, lon_max=5, resolution=0.1)
        lat = np.arange(0.05, 5.0, 0.1)
        lon = np.arange(0.05, 5.0, 0.1)
        ds = xr.Dataset(
            {"v": (["lat", "lon"], np.ones((len(lat), len(lon))))},
            coords={"lat": lat, "lon": lon},
        )
        result = regrid_to_common(ds, target_resolution=0.1, domain=domain)
        assert result.dims["lat"] > 0

    def test_fallback_interp(self):
        """Test the xarray fallback interpolation path."""
        domain = SpatialDomain(lat_min=0, lat_max=5, lon_min=0, lon_max=5, resolution=1.0)
        ds = xr.Dataset(
            {"v": (["lat", "lon"], np.ones((10, 10)))},
            coords={"lat": np.arange(0.25, 5.0, 0.5), "lon": np.arange(0.25, 5.0, 0.5)},
        )
        result = _fallback_interp(ds, domain, 1.0)
        assert "lat" in result.coords
        assert "lon" in result.coords
