"""Per-cell soil-moisture climatology: loading the cube, finding a site's cell, normalising."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from src.feature_contract.swvl_climatology import (
    MIN_VALID_SIGMA,
    CellStats,
    load_climatology,
    lookup_cell,
    normalise_swvl,
)

LATS = np.array([52.0, 51.9, 51.8])  # descending, like the v2 cube
LONS = np.array([10.0, 10.1, 10.2])


def _write_cube(path, sigma, mean, lats=LATS, lons=LONS):
    layers = np.arange(1, sigma.shape[0] + 1, dtype=np.int8)
    xr.Dataset(
        {"swvl_sigma": (("layer", "lat", "lon"), sigma), "swvl_mean": (("layer", "lat", "lon"), mean)},
        coords={"layer": layers, "lat": lats, "lon": lons},
        attrs={"eps_sigma": 1e-10, "version": "test"},
    ).to_netcdf(path)
    return path


def _default_arrays():
    sigma = np.full((2, 3, 3), 0.05, dtype=np.float32)
    mean = np.full((2, 3, 3), 0.30, dtype=np.float32)
    sigma[1] = 0.02
    mean[1] = 0.25
    sigma[0, 0, 0] = np.nan  # layer 1: an empty (ocean) cell at (52.0, 10.0)
    sigma[0, 2, 2] = 1e-10  # layer 1: a floor-clipped constant cell at (51.8, 10.2)
    mean[0, 1, 1] = 0.40  # layer 1: the centre cell is distinguishable by its mean
    return sigma, mean


@pytest.fixture
def cube(tmp_path):
    sigma, mean = _default_arrays()
    return load_climatology(_write_cube(tmp_path / "cube.nc", sigma, mean))


class TestLoad:
    def test_reads_the_stacked_schema(self, cube):
        assert cube.layers == (1, 2)
        assert cube.sigma[2].shape == (3, 3)
        np.testing.assert_allclose(cube.lats, LATS)

    def test_records_the_source_checksum(self, cube):
        assert len(cube.sha256) == 64

    def test_rejects_a_cube_without_sigma(self, tmp_path):
        path = tmp_path / "bad.nc"
        xr.Dataset({"x": (("lat",), LATS)}, coords={"lat": LATS}).to_netcdf(path)
        with pytest.raises(KeyError, match="swvl_sigma"):
            load_climatology(path)

    def test_rejects_a_0_360_longitude_axis(self, tmp_path):
        sigma, mean = _default_arrays()
        path = _write_cube(tmp_path / "c360.nc", sigma, mean, lons=np.array([190.0, 190.1, 190.2]))
        with pytest.raises(ValueError, match="longitude"):
            load_climatology(path)


class TestLookup:
    def test_exact_cell(self, cube):
        cell = lookup_cell(cube, lat=51.9, lon=10.1, layer=1)
        assert isinstance(cell, CellStats)
        assert cell.fallback is False
        assert cell.mean == pytest.approx(0.40)
        assert cell.sigma == pytest.approx(0.05)

    def test_nearest_cell_not_the_corner(self, cube):
        cell = lookup_cell(cube, lat=51.87, lon=10.13, layer=1)
        assert (cell.cell_lat, cell.cell_lon) == pytest.approx((51.9, 10.1))

    def test_empty_cell_falls_back_to_a_valid_neighbour(self, cube):
        cell = lookup_cell(cube, lat=52.0, lon=10.0, layer=1)
        assert cell.fallback is True
        assert cell.distance_km > 0
        assert np.isfinite(cell.sigma) and cell.sigma >= MIN_VALID_SIGMA

    def test_floor_clipped_sigma_is_not_a_valid_cell(self, cube):
        cell = lookup_cell(cube, lat=51.8, lon=10.2, layer=1)
        assert cell.fallback is True
        assert cell.sigma == pytest.approx(0.05)

    def test_layers_are_looked_up_independently(self, cube):
        cell = lookup_cell(cube, lat=52.0, lon=10.0, layer=2)
        assert cell.fallback is False
        assert cell.sigma == pytest.approx(0.02)

    def test_nothing_valid_within_the_radius_raises(self, cube):
        with pytest.raises(LookupError):
            lookup_cell(cube, lat=52.0, lon=10.0, layer=1, max_radius_cells=0)

    def test_a_site_outside_the_grid_raises(self, cube):
        with pytest.raises(LookupError, match="outside"):
            lookup_cell(cube, lat=-30.0, lon=150.0, layer=1)

    def test_unknown_layer_raises(self, cube):
        with pytest.raises(KeyError):
            lookup_cell(cube, lat=51.9, lon=10.1, layer=4)

    def test_fallback_takes_the_nearest_by_distance_not_the_first_ring(self, tmp_path):
        """At 70 deg N two longitude cells (7.6 km) are nearer than one latitude cell (11.1 km)."""
        lats = np.array([70.1, 70.0, 69.9])
        lons = np.array([10.0, 10.1, 10.2, 10.3, 10.4])
        sigma = np.full((1, 3, 5), np.nan, dtype=np.float32)
        mean = np.full((1, 3, 5), 0.3, dtype=np.float32)
        sigma[0, 0, 2] = 0.01  # (70.1, 10.2): ring 1, 11.1 km
        sigma[0, 1, 4] = 0.02  # (70.0, 10.4): ring 2, ~7.6 km
        clim = load_climatology(_write_cube(tmp_path / "arctic.nc", sigma, mean, lats=lats, lons=lons))
        cell = lookup_cell(clim, lat=70.0, lon=10.2, layer=1)
        assert (cell.cell_lat, cell.cell_lon) == pytest.approx((70.0, 10.4))
        assert cell.distance_km == pytest.approx(7.6, abs=0.2)


class TestNormalise:
    def test_bare_and_zscore(self):
        bare, z = normalise_swvl(np.array([0.30, 0.35]), sigma=0.05, mean=0.30)
        np.testing.assert_allclose(bare, [6.0, 7.0])
        np.testing.assert_allclose(z, [0.0, 1.0], atol=1e-12)

    def test_series_keep_their_index(self):
        bare, z = normalise_swvl(pd.Series([0.30, 0.40], index=[5, 9]), sigma=0.05, mean=0.30)
        assert list(bare.index) == [5, 9]
        assert list(z.index) == [5, 9]

    def test_values_are_not_clipped(self):
        bare, _ = normalise_swvl(np.array([0.9]), sigma=0.05, mean=0.3)
        assert bare[0] == pytest.approx(18.0)

    def test_nan_passes_through(self):
        bare, z = normalise_swvl(np.array([np.nan, 0.3]), sigma=0.05, mean=0.3)
        assert np.isnan(bare[0]) and np.isnan(z[0])

    @pytest.mark.parametrize("sigma", [0.0, -0.1, np.nan, 1e-10])
    def test_invalid_sigma_raises(self, sigma):
        with pytest.raises(ValueError, match="sigma"):
            normalise_swvl(np.array([0.3]), sigma=sigma, mean=0.3)
