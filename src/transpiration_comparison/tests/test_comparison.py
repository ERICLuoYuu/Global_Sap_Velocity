"""Tests for comparison modules: spatial, temporal, agreement, collocation, regional."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.transpiration_comparison.comparison.agreement import (
    consensus_map,
    product_ranking,
    unique_signal_detection,
)
from src.transpiration_comparison.comparison.collocation import (
    compute_tc_weights,
    triple_collocation,
)
from src.transpiration_comparison.comparison.regional import (
    all_regional_analyses,
    regional_analysis,
)
from src.transpiration_comparison.comparison.spatial import (
    _manual_correlation,
    pft_stratified_metrics,
    pixel_temporal_correlation,
    spatial_rmse_map,
    zonal_mean_profile,
)
from src.transpiration_comparison.comparison.temporal_comp import (
    interannual_correlation,
    seasonal_cycle_comparison,
    trend_agreement,
)
from src.transpiration_comparison.config import PFT_COLUMNS, REGIONS, Region

# ── Helpers ────────────────────────────────────────────────────────


def _make_corr_ds(lat, lon, r_value: float = 0.6) -> xr.Dataset:
    """Create a Dataset with a uniform pearson_r map."""
    data = np.full((len(lat), len(lon)), r_value, dtype=np.float32)
    return xr.Dataset(
        {"pearson_r": (["lat", "lon"], data)},
        coords={"lat": lat, "lon": lon},
    )


def _make_ts_dataset(
    var_name: str,
    lat: np.ndarray,
    lon: np.ndarray,
    time: pd.DatetimeIndex,
    seed: int = 0,
    scale: float = 1.0,
) -> xr.Dataset:
    """Create a synthetic time-series dataset with seasonal signal."""
    rng = np.random.default_rng(seed)
    nt, nlat, nlon = len(time), len(lat), len(lon)
    months = time.month.values
    seasonal = scale * np.sin(2 * np.pi * (months - 3) / 12)
    data = seasonal[:, None, None] + rng.normal(0, 0.1, (nt, nlat, nlon))
    return xr.Dataset(
        {var_name: (["time", "lat", "lon"], data.astype(np.float32))},
        coords={"time": time, "lat": lat, "lon": lon},
    )


# ── Consensus map ─────────────────────────────────────────────────


class TestConsensusMap:
    def test_all_high_agreement(self, small_domain):
        lat, lon = small_domain
        corr_maps = {
            "gleam": _make_corr_ds(lat, lon, 0.8),
            "era5": _make_corr_ds(lat, lon, 0.7),
            "gldas": _make_corr_ds(lat, lon, 0.9),
            "pmlv2": _make_corr_ds(lat, lon, 0.6),
        }
        result = consensus_map(corr_maps)
        assert "mean_r" in result.data_vars
        assert "consensus_zone" in result.data_vars
        assert "n_agree" in result.data_vars
        # All r > 0.5, so all pixels should agree
        assert int(result["n_agree"].min()) == 4

    def test_mixed_agreement(self, small_domain):
        lat, lon = small_domain
        corr_maps = {
            "gleam": _make_corr_ds(lat, lon, 0.8),
            "era5": _make_corr_ds(lat, lon, 0.3),  # below 0.5
            "gldas": _make_corr_ds(lat, lon, 0.9),
        }
        result = consensus_map(corr_maps)
        # 2 out of 3 agree (67%) -> medium consensus
        assert int(result["n_agree"].values[0, 0]) == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No correlation maps"):
            consensus_map({})

    def test_consensus_zones_categorical(self, small_domain):
        lat, lon = small_domain
        corr_maps = {
            "a": _make_corr_ds(lat, lon, 0.8),
            "b": _make_corr_ds(lat, lon, 0.8),
            "c": _make_corr_ds(lat, lon, 0.8),
            "d": _make_corr_ds(lat, lon, 0.8),
        }
        result = consensus_map(corr_maps)
        zones = np.unique(result["consensus_zone"].values[~np.isnan(result["consensus_zone"].values)])
        assert all(z in [1, 2, 3] for z in zones)


# ── Product ranking ───────────────────────────────────────────────


class TestProductRanking:
    def test_ranking_order(self):
        metrics_a = pd.DataFrame([{"pft": "ENF", "pearson_r": 0.9, "rmse": 0.1, "n_pixels": 100}])
        metrics_b = pd.DataFrame([{"pft": "ENF", "pearson_r": 0.5, "rmse": 0.3, "n_pixels": 100}])
        result = product_ranking({"prod_a": metrics_a, "prod_b": metrics_b})
        assert len(result) == 2
        best = result[result["rank_by_r"] == 1].iloc[0]
        assert best["product"] == "prod_a"

    def test_single_product_skipped(self):
        metrics = pd.DataFrame([{"pft": "ENF", "pearson_r": 0.9, "rmse": 0.1, "n_pixels": 100}])
        result = product_ranking({"only_one": metrics})
        assert len(result) == 0


# ── Unique signal detection ───────────────────────────────────────


class TestUniqueSignalDetection:
    def test_returns_dataframe(self, small_domain, monthly_time, pft_map):
        lat, lon = small_domain
        datasets = {
            "sap_velocity": _make_ts_dataset("sv_zscore", lat, lon, monthly_time, seed=0),
            "gleam": _make_ts_dataset("gl_zscore", lat, lon, monthly_time, seed=1),
            "era5": _make_ts_dataset("e5_zscore", lat, lon, monthly_time, seed=2),
        }
        result = unique_signal_detection(datasets, pft_map)
        assert isinstance(result, pd.DataFrame)

    def test_insufficient_products_returns_empty(self, small_domain, monthly_time, pft_map):
        lat, lon = small_domain
        datasets = {
            "sap_velocity": _make_ts_dataset("sv_zscore", lat, lon, monthly_time),
        }
        result = unique_signal_detection(datasets, pft_map)
        assert len(result) == 0


# ── Triple collocation ────────────────────────────────────────────


class TestTripleCollocation:
    def _make_tc_data(self, n_time: int = 100, n_lat: int = 5, n_lon: int = 5):
        """Create 3 synthetic products: truth + independent noise."""
        rng = np.random.default_rng(42)
        time = pd.date_range("2016-01-01", periods=n_time, freq="MS")
        lat = np.arange(n_lat, dtype=float)
        lon = np.arange(n_lon, dtype=float)

        truth = rng.normal(0, 1, (n_time, n_lat, n_lon)).astype(np.float32)
        noise_a = rng.normal(0, 0.3, truth.shape).astype(np.float32)
        noise_b = rng.normal(0, 0.5, truth.shape).astype(np.float32)
        noise_c = rng.normal(0, 0.7, truth.shape).astype(np.float32)

        def _to_ds(data, var):
            return xr.Dataset(
                {var: (["time", "lat", "lon"], data)},
                coords={"time": time, "lat": lat, "lon": lon},
            )

        return (
            _to_ds(truth + noise_a, "a"),
            _to_ds(truth + noise_b, "b"),
            _to_ds(truth + noise_c, "c"),
        )

    def test_returns_three_error_variances(self):
        ds_a, ds_b, ds_c = self._make_tc_data()
        result = triple_collocation(ds_a, ds_b, ds_c, "a", "b", "c", min_valid=24)
        assert "err_var_a" in result
        assert "err_var_b" in result
        assert "err_var_c" in result

    def test_error_ordering(self):
        """Product A (noise=0.3) should have smallest error, C (noise=0.7) largest."""
        ds_a, ds_b, ds_c = self._make_tc_data(n_time=200)
        result = triple_collocation(ds_a, ds_b, ds_c, "a", "b", "c", min_valid=24)
        med_a = float(result["err_var_a"].median())
        med_c = float(result["err_var_c"].median())
        assert med_a < med_c

    def test_min_valid_filter(self):
        """With min_valid > n_time, all pixels should be NaN."""
        ds_a, ds_b, ds_c = self._make_tc_data(n_time=10)
        result = triple_collocation(ds_a, ds_b, ds_c, "a", "b", "c", min_valid=50)
        assert np.all(np.isnan(result["err_var_a"].values))


class TestComputeTCWeights:
    def test_weights_sum_to_one(self):
        err = {
            "err_var_a": xr.DataArray([0.1, 0.2]),
            "err_var_b": xr.DataArray([0.3, 0.4]),
            "err_var_c": xr.DataArray([0.5, 0.6]),
        }
        weights = compute_tc_weights(err)
        total = weights["err_var_a"] + weights["err_var_b"] + weights["err_var_c"]
        np.testing.assert_allclose(total.values, 1.0, atol=1e-5)

    def test_lower_error_gets_higher_weight(self):
        err = {
            "err_var_a": xr.DataArray([0.1]),
            "err_var_b": xr.DataArray([1.0]),
            "err_var_c": xr.DataArray([1.0]),
        }
        weights = compute_tc_weights(err)
        assert float(weights["err_var_a"]) > float(weights["err_var_b"])


# ── Seasonal cycle comparison ─────────────────────────────────────


class TestSeasonalCycleComparison:
    def test_basic_output(self, small_domain, monthly_time, pft_map):
        lat, lon = small_domain
        ds1 = _make_ts_dataset("t_clim", lat, lon, monthly_time, seed=0)
        ds1["t_clim"] = ds1["t_clim"].groupby("time.month").mean("time")
        ds2 = _make_ts_dataset("t_clim", lat, lon, monthly_time, seed=1)
        ds2["t_clim"] = ds2["t_clim"].groupby("time.month").mean("time")

        result = seasonal_cycle_comparison({"prod_a": ds1, "prod_b": ds2}, pft_map, var_suffix="_clim")
        assert "correlations" in result
        assert "peak_months" in result
        assert "seasonal_profiles" in result

    def test_insufficient_products_raises(self, small_domain, monthly_time, pft_map):
        lat, lon = small_domain
        ds1 = _make_ts_dataset("t_clim", lat, lon, monthly_time)
        ds1["t_clim"] = ds1["t_clim"].groupby("time.month").mean("time")
        with pytest.raises(ValueError, match="Need"):
            seasonal_cycle_comparison({"only": ds1}, pft_map)


# ── Interannual correlation ───────────────────────────────────────


class TestInterannualCorrelation:
    def test_perfect_correlation(self, small_domain, monthly_time):
        lat, lon = small_domain
        ds = _make_ts_dataset("v", lat, lon, monthly_time, seed=0)
        result = interannual_correlation(ds, ds, "v", "v")
        # Self-correlation should be ~1
        r = result["interannual_r"]
        valid = r.values[np.isfinite(r.values)]
        if len(valid) > 0:
            assert np.mean(valid) > 0.8


# ── Trend agreement ───────────────────────────────────────────────


class TestTrendAgreement:
    def test_same_trend_full_agreement(self, small_domain, monthly_time):
        lat, lon = small_domain
        # Both products with identical positive trend
        ds1 = _make_ts_dataset("v", lat, lon, monthly_time, seed=0, scale=1.0)
        ds2 = _make_ts_dataset("v", lat, lon, monthly_time, seed=0, scale=1.0)
        result = trend_agreement({"a": ds1, "b": ds2}, var_key="v")
        assert "trend_agreement" in result.data_vars
        # Identical data -> same trend sign -> agreement = 1.0
        assert float(result["trend_agreement"].mean()) == pytest.approx(1.0)

    def test_single_product_returns_empty(self, small_domain, monthly_time):
        lat, lon = small_domain
        ds = _make_ts_dataset("v", lat, lon, monthly_time)
        result = trend_agreement({"only": ds}, var_key="v")
        assert len(result.data_vars) == 0


# ── Regional analysis ─────────────────────────────────────────────


class TestRegionalAnalysis:
    def test_basic_output(self, small_domain, monthly_time, pft_map):
        lat, lon = small_domain
        ds = _make_ts_dataset("v_zscore", lat, lon, monthly_time)
        region = Region("test", float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max()))
        result = regional_analysis({"prod": ds}, region, pft_map)
        assert "time_series" in result
        assert "seasonal_overlay" in result
        assert "pft_breakdown" in result
        assert "stats" in result

    def test_empty_region(self, small_domain, monthly_time, pft_map):
        lat, lon = small_domain
        ds = _make_ts_dataset("v_zscore", lat, lon, monthly_time)
        # Region that doesn't overlap with data
        region = Region("empty", -90, -80, -180, -170)
        result = regional_analysis({"prod": ds}, region, pft_map)
        assert len(result["time_series"]) == 0
        assert result["stats"] == {}

    def test_all_regional_analyses_runs(self, small_domain, monthly_time, pft_map):
        lat, lon = small_domain
        ds = _make_ts_dataset("v_zscore", lat, lon, monthly_time)
        result = all_regional_analyses({"prod": ds}, pft_map)
        assert len(result) == len(REGIONS)


# ── Pixel temporal correlation ────────────────────────────────────


class TestPixelTemporalCorrelation:
    def test_self_correlation(self, small_domain, monthly_time):
        lat, lon = small_domain
        ds = _make_ts_dataset("v", lat, lon, monthly_time, seed=0)
        result = pixel_temporal_correlation(ds, ds, "v", "v", min_overlap=10)
        assert "pearson_r" in result.data_vars
        # Self-correlation should be ~1
        r = result["pearson_r"]
        valid = r.values[np.isfinite(r.values)]
        assert np.mean(valid) > 0.99

    def test_min_overlap_filter(self, small_domain):
        lat, lon = small_domain
        time = pd.date_range("2016-01-01", periods=5, freq="MS")
        ds = _make_ts_dataset("v", lat, lon, time, seed=0)
        result = pixel_temporal_correlation(ds, ds, "v", "v", min_overlap=10)
        # Only 5 timesteps < 10 min_overlap => all NaN
        assert np.all(np.isnan(result["pearson_r"].values))

    def test_uncorrelated_low_r(self, small_domain, monthly_time):
        lat, lon = small_domain
        ds1 = _make_ts_dataset("v", lat, lon, monthly_time, seed=0)
        # Pure noise — no seasonal signal — so uncorrelated with ds1
        rng = np.random.default_rng(999)
        noise = rng.normal(0, 1, (len(monthly_time), len(lat), len(lon))).astype(np.float32)
        ds2 = xr.Dataset(
            {"v": (["time", "lat", "lon"], noise)},
            coords={"time": monthly_time, "lat": lat, "lon": lon},
        )
        result = pixel_temporal_correlation(ds1, ds2, "v", "v", min_overlap=10)
        r = result["pearson_r"]
        valid = r.values[np.isfinite(r.values)]
        assert abs(np.mean(valid)) < 0.5


# ── Spatial RMSE ──────────────────────────────────────────────────


class TestSpatialRmseMap:
    def test_zero_rmse_for_identical(self, small_domain, monthly_time):
        lat, lon = small_domain
        ds = _make_ts_dataset("v", lat, lon, monthly_time)
        rmse = spatial_rmse_map(ds, ds, "v", "v")
        np.testing.assert_allclose(rmse.values, 0.0, atol=1e-6)

    def test_nonzero_rmse_for_different(self, small_domain, monthly_time):
        lat, lon = small_domain
        ds1 = _make_ts_dataset("v", lat, lon, monthly_time, seed=0)
        ds2 = _make_ts_dataset("v", lat, lon, monthly_time, seed=42)
        rmse = spatial_rmse_map(ds1, ds2, "v", "v")
        assert float(rmse.mean()) > 0


# ── PFT stratified metrics ───────────────────────────────────────


class TestPftStratifiedMetrics:
    def _make_large_grid(self, monthly_time):
        """20x20 grid to satisfy n_pixels >= 10 and valid >= 100."""
        lat = np.arange(45.05, 65.0, 1.0)  # 20 points
        lon = np.arange(5.05, 25.0, 1.0)  # 20 points
        pft_codes = list(PFT_COLUMNS.keys())
        # Tile PFTs across the grid
        pft_data = np.array([[pft_codes[(i + j) % len(pft_codes)] for j in range(len(lon))] for i in range(len(lat))])
        pft_map = xr.DataArray(pft_data, dims=["lat", "lon"], coords={"lat": lat, "lon": lon})
        ds_ref = _make_ts_dataset("ref_zscore", lat, lon, monthly_time, seed=0)
        ds_comp = _make_ts_dataset("comp_zscore", lat, lon, monthly_time, seed=1)
        return ds_ref, ds_comp, pft_map

    def test_returns_dataframe(self, monthly_time):
        ds_ref, ds_comp, pft_map = self._make_large_grid(monthly_time)
        result = pft_stratified_metrics(ds_ref, ds_comp, pft_map, "ref_zscore", "comp_zscore")
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert "pearson_r" in result.columns
            assert "rmse" in result.columns
            assert "pft" in result.columns


# ── Zonal mean profile ───────────────────────────────────────────


class TestZonalMeanProfile:
    def test_basic_output(self, small_domain, monthly_time):
        lat, lon = small_domain
        ds = _make_ts_dataset("v_zscore", lat, lon, monthly_time)
        result = zonal_mean_profile({"prod": ds}, var_suffix="_zscore")
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert "lat_center" in result.columns
            assert "product" in result.columns

    def test_no_zscore_var_warns(self, small_domain, monthly_time):
        lat, lon = small_domain
        ds = _make_ts_dataset("temperature", lat, lon, monthly_time)
        result = zonal_mean_profile({"prod": ds}, var_suffix="_zscore")
        assert len(result) == 0


# ── Manual correlation ────────────────────────────────────────────


class TestManualCorrelation:
    def test_self_correlation_is_one(self, small_domain, monthly_time):
        lat, lon = small_domain
        ds = _make_ts_dataset("v", lat, lon, monthly_time, seed=0)
        result = _manual_correlation(ds["v"], ds["v"])
        assert "pearson_r" in result.data_vars
        r = result["pearson_r"]
        valid = r.values[np.isfinite(r.values)]
        np.testing.assert_allclose(valid, 1.0, atol=1e-5)

    def test_p_value_is_nan(self, small_domain, monthly_time):
        lat, lon = small_domain
        ds = _make_ts_dataset("v", lat, lon, monthly_time)
        result = _manual_correlation(ds["v"], ds["v"])
        assert np.all(np.isnan(result["p_value"].values))
