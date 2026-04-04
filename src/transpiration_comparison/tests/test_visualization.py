"""Tests for visualization modules (matplotlib-based, use Agg backend)."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# matplotlib is installed but broken in this env (cp312 .pyd with py310 runtime,
# missing __init__.py). Skip entire module if pyplot is not importable.
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot  # noqa: F401
except (ImportError, AttributeError):
    pytest.skip("matplotlib not functional in this environment", allow_module_level=True)

from src.transpiration_comparison.visualization.timeseries import (  # noqa: E402
    PRODUCT_COLORS,
    plot_pft_heatmap,
    plot_regional_timeseries,
    plot_seasonal_overlay,
    plot_zonal_mean,
)

# ── Timeseries plots ─────────────────────────────────────────────


class TestProductColors:
    def test_all_products_have_colors(self):
        expected = {"sap_velocity", "gleam", "era5land", "pmlv2", "gldas"}
        assert set(PRODUCT_COLORS.keys()) == expected


class TestPlotSeasonalOverlay:
    def test_creates_file(self, tmp_path):
        profiles = {
            "sap_velocity": {"ENF": np.sin(np.linspace(0, 2 * np.pi, 12))},
            "gleam": {"ENF": np.cos(np.linspace(0, 2 * np.pi, 12))},
        }
        out = tmp_path / "seasonal.png"
        plot_seasonal_overlay(profiles, "ENF", out)
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_missing_pft_no_error(self, tmp_path):
        profiles = {"sap_velocity": {"DBF": np.zeros(12)}}
        out = tmp_path / "seasonal_enf.png"
        plot_seasonal_overlay(profiles, "ENF", out)
        assert out.exists()


class TestPlotRegionalTimeseries:
    def test_creates_file(self, tmp_path):
        df = pd.DataFrame(
            {
                "time": pd.date_range("2016-01-01", periods=100, freq="D"),
                "product": ["gleam"] * 100,
                "zscore": np.random.default_rng(0).normal(0, 1, 100),
            }
        )
        out = tmp_path / "regional_ts.png"
        plot_regional_timeseries(df, "Amazon", out)
        assert out.exists()

    def test_empty_df_no_error(self, tmp_path):
        out = tmp_path / "empty_ts.png"
        plot_regional_timeseries(pd.DataFrame(), "Empty", out)
        assert not out.exists()  # should skip on empty


class TestPlotZonalMean:
    def test_creates_file(self, tmp_path):
        df = pd.DataFrame(
            {
                "lat_center": np.arange(-55, 75, 5),
                "product": ["gleam"] * 26,
                "seasonal_amplitude": np.random.default_rng(0).uniform(0, 1, 26),
            }
        )
        out = tmp_path / "zonal.png"
        plot_zonal_mean(df, out)
        assert out.exists()

    def test_empty_df_no_error(self, tmp_path):
        out = tmp_path / "empty_zonal.png"
        plot_zonal_mean(pd.DataFrame(), out)
        assert not out.exists()


class TestPlotPftHeatmap:
    def test_creates_file(self, tmp_path):
        metrics = {
            "gleam": pd.DataFrame(
                [
                    {
                        "pft": "ENF",
                        "pft_name": "Evergreen Needleleaf",
                        "pearson_r": 0.8,
                        "rmse": 0.2,
                        "mae": 0.15,
                        "bias": 0.01,
                        "n_pixels": 100,
                    },
                    {
                        "pft": "DBF",
                        "pft_name": "Deciduous Broadleaf",
                        "pearson_r": 0.6,
                        "rmse": 0.3,
                        "mae": 0.2,
                        "bias": -0.02,
                        "n_pixels": 80,
                    },
                ]
            ),
            "era5land": pd.DataFrame(
                [
                    {
                        "pft": "ENF",
                        "pft_name": "Evergreen Needleleaf",
                        "pearson_r": 0.7,
                        "rmse": 0.25,
                        "mae": 0.18,
                        "bias": 0.05,
                        "n_pixels": 100,
                    },
                    {
                        "pft": "DBF",
                        "pft_name": "Deciduous Broadleaf",
                        "pearson_r": 0.5,
                        "rmse": 0.35,
                        "mae": 0.22,
                        "bias": -0.03,
                        "n_pixels": 80,
                    },
                ]
            ),
        }
        out = tmp_path / "heatmap.png"
        plot_pft_heatmap(metrics, "pearson_r", out)
        assert out.exists()

    def test_empty_metrics_no_error(self, tmp_path):
        out = tmp_path / "empty_heatmap.png"
        plot_pft_heatmap({}, "pearson_r", out)
        assert not out.exists()


# ── Map plots (cartopy optional) ─────────────────────────────────


class TestMapPlots:
    @pytest.fixture(autouse=True)
    def _skip_if_no_cartopy(self):
        pytest.importorskip("cartopy")

    def test_plot_global_map(self, tmp_path):
        from src.transpiration_comparison.visualization.maps import plot_global_map

        data = xr.DataArray(
            np.random.default_rng(0).random((10, 20)),
            dims=["lat", "lon"],
            coords={"lat": np.linspace(-45, 45, 10), "lon": np.linspace(-90, 90, 20)},
        )
        out = tmp_path / "global.png"
        plot_global_map(data, "Test", out)
        assert out.exists()

    def test_plot_correlation_map(self, tmp_path):
        from src.transpiration_comparison.visualization.maps import plot_correlation_map

        data = xr.DataArray(
            np.random.default_rng(1).uniform(-1, 1, (10, 20)),
            dims=["lat", "lon"],
            coords={"lat": np.linspace(-45, 45, 10), "lon": np.linspace(-90, 90, 20)},
        )
        out = tmp_path / "corr.png"
        plot_correlation_map(data, "Correlation", out)
        assert out.exists()

    def test_plot_multi_product_maps(self, tmp_path):
        from src.transpiration_comparison.visualization.maps import plot_multi_product_maps

        lat = np.linspace(-45, 45, 10)
        lon = np.linspace(-90, 90, 20)
        data_a = xr.DataArray(
            np.random.default_rng(0).random((10, 20)).astype(np.float32),
            dims=["lat", "lon"],
            coords={"lat": lat, "lon": lon},
        )
        data_b = xr.DataArray(
            np.random.default_rng(1).random((10, 20)).astype(np.float32),
            dims=["lat", "lon"],
            coords={"lat": lat, "lon": lon},
        )
        out = tmp_path / "multi.png"
        plot_multi_product_maps({"A": data_a, "B": data_b}, "Multi", out)
        assert out.exists()

    def test_plot_consensus_map(self, tmp_path):
        from src.transpiration_comparison.visualization.maps import plot_consensus_map

        lat = np.linspace(-45, 45, 10)
        lon = np.linspace(-90, 90, 20)
        ds = xr.Dataset(
            {
                "consensus_zone": (
                    ["lat", "lon"],
                    np.random.default_rng(0).choice([1, 2, 3], size=(10, 20)).astype(np.int8),
                )
            },
            coords={"lat": lat, "lon": lon},
        )
        out = tmp_path / "consensus.png"
        plot_consensus_map(ds, out)
        assert out.exists()


# ── Taylor diagram ────────────────────────────────────────────────


class TestTaylorDiagram:
    def test_manual_taylor_creates_file(self, tmp_path):
        from src.transpiration_comparison.visualization.taylor import _manual_taylor

        rng = np.random.default_rng(42)
        ref = xr.DataArray(rng.normal(0, 1, (100, 10, 10)), dims=["time", "lat", "lon"])
        products = {
            "gleam": xr.DataArray(ref.values + rng.normal(0, 0.3, ref.shape), dims=["time", "lat", "lon"]),
        }
        out = tmp_path / "taylor.png"
        _manual_taylor(ref, products, "Test Taylor", out)
        assert out.exists()

    def test_taylor_with_insufficient_data(self, tmp_path):
        from src.transpiration_comparison.visualization.taylor import taylor_diagram

        ref = xr.DataArray(np.full((5, 2, 2), np.nan), dims=["time", "lat", "lon"])
        products = {"a": xr.DataArray(np.full((5, 2, 2), np.nan), dims=["time", "lat", "lon"])}
        out = tmp_path / "taylor_empty.png"
        taylor_diagram(ref, products, "Empty", out)
        # Should not create file when no valid data
        assert not out.exists()
