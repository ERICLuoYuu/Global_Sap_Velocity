"""Tests for config module and ProductBase._standardize_coords."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from src.transpiration_comparison.config import (
    PRODUCTS,
    REGIONS,
    SpatialDomain,
    TimePeriod,
    make_common_grid,
)
from src.transpiration_comparison.products.base import ProductBase


class TestSpatialDomain:
    def test_defaults(self):
        d = SpatialDomain()
        assert d.lat_min == -60.0
        assert d.lat_max == 78.0
        assert d.lon_min == -180.0
        assert d.lon_max == 180.0
        assert d.resolution == 0.1

    def test_frozen(self):
        d = SpatialDomain()
        with pytest.raises(AttributeError):
            d.lat_min = 0.0


class TestTimePeriod:
    def test_defaults(self):
        p = TimePeriod()
        assert p.start == "2016-01-01"
        assert p.end == "2018-12-31"


class TestMakeCommonGrid:
    def test_default_grid_shape(self):
        lat, lon = make_common_grid()
        assert lat[0] == pytest.approx(-59.95, abs=0.01)
        assert lon[0] == pytest.approx(-179.95, abs=0.01)
        assert np.diff(lat).mean() == pytest.approx(0.1, abs=1e-6)
        assert np.diff(lon).mean() == pytest.approx(0.1, abs=1e-6)

    def test_custom_domain(self):
        d = SpatialDomain(lat_min=0, lat_max=10, lon_min=0, lon_max=10, resolution=1.0)
        lat, lon = make_common_grid(d)
        assert len(lat) == 10
        assert len(lon) == 10
        assert lat[0] == pytest.approx(0.5, abs=0.01)


class TestProductDefinitions:
    def test_all_products_present(self):
        expected = {"sap_velocity", "gleam", "era5land", "pmlv2", "gldas"}
        assert set(PRODUCTS.keys()) == expected

    def test_era5land_config(self):
        cfg = PRODUCTS["era5land"]
        assert cfg.units == "m/day"
        assert cfg.source_type == "cds"

    def test_regions_present(self):
        assert len(REGIONS) >= 6
        assert "amazon" in REGIONS
        assert "central_eu" in REGIONS


class TestStandardizeCoords:
    """Tests for ProductBase._standardize_coords including lon wrapping."""

    def _make_base(self) -> ProductBase:
        """Create a concrete ProductBase for testing."""

        # Use era5land config since it's the one needing lon wrapping
        class _Concrete(ProductBase):
            def download(self):
                pass

            def load(self):
                pass

        return _Concrete(config=PRODUCTS["era5land"])

    def test_rename_latitude_longitude(self):
        base = self._make_base()
        ds = xr.Dataset(
            {"temp": (["time", "latitude", "longitude"], np.ones((2, 3, 4)))},
            coords={
                "time": [0, 1],
                "latitude": [10.0, 20.0, 30.0],
                "longitude": [5.0, 10.0, 15.0, 20.0],
            },
        )
        result = base._standardize_coords(ds)
        assert "lat" in result.coords
        assert "lon" in result.coords
        assert "latitude" not in result.coords

    def test_rename_valid_time(self):
        base = self._make_base()
        ds = xr.Dataset(
            {"temp": (["valid_time", "lat", "lon"], np.ones((2, 3, 4)))},
            coords={
                "valid_time": [0, 1],
                "lat": [10.0, 20.0, 30.0],
                "lon": [5.0, 10.0, 15.0, 20.0],
            },
        )
        result = base._standardize_coords(ds)
        assert "time" in result.coords

    def test_ascending_lat(self):
        base = self._make_base()
        ds = xr.Dataset(
            {"temp": (["time", "lat", "lon"], np.arange(12).reshape(2, 3, 2).astype(float))},
            coords={
                "time": [0, 1],
                "lat": [30.0, 20.0, 10.0],  # descending
                "lon": [5.0, 10.0],
            },
        )
        result = base._standardize_coords(ds)
        assert result["lat"].values[0] < result["lat"].values[-1]

    def test_lon_wrapping_0_360_to_neg180_180(self):
        """ERA5-Land lon 0..360 should be wrapped to -180..180."""
        base = self._make_base()
        lons_360 = np.arange(0, 360, 90, dtype=float)  # [0, 90, 180, 270]
        ds = xr.Dataset(
            {"temp": (["time", "lat", "lon"], np.ones((1, 2, 4)))},
            coords={"time": [0], "lat": [10.0, 20.0], "lon": lons_360},
        )
        result = base._standardize_coords(ds)
        assert float(result["lon"].min()) >= -180.0
        assert float(result["lon"].max()) <= 180.0
        # 270 should become -90
        assert -90.0 in result["lon"].values

    def test_lon_sorted_after_wrapping(self):
        base = self._make_base()
        lons_360 = np.array([0.0, 90.0, 180.0, 270.0])
        ds = xr.Dataset(
            {"temp": (["time", "lat", "lon"], np.ones((1, 2, 4)))},
            coords={"time": [0], "lat": [10.0, 20.0], "lon": lons_360},
        )
        result = base._standardize_coords(ds)
        lon_vals = result["lon"].values
        assert np.all(np.diff(lon_vals) > 0), "Longitudes must be ascending after wrap"

    def test_no_wrapping_when_already_neg180_180(self):
        base = self._make_base()
        lons = np.array([-90.0, 0.0, 90.0])
        ds = xr.Dataset(
            {"temp": (["time", "lat", "lon"], np.ones((1, 2, 3)))},
            coords={"time": [0], "lat": [10.0, 20.0], "lon": lons},
        )
        result = base._standardize_coords(ds)
        np.testing.assert_array_equal(result["lon"].values, lons)

    def test_data_preserved_after_wrapping(self):
        """Ensure pixel values follow their original lon after the wrap+sort."""
        base = self._make_base()
        lons_360 = np.array([0.0, 90.0, 180.0, 270.0])
        # Unique values per lon column so we can track them
        data = np.array([[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]])
        ds = xr.Dataset(
            {"temp": (["time", "lat", "lon"], data)},
            coords={"time": [0], "lat": [10.0, 20.0], "lon": lons_360},
        )
        result = base._standardize_coords(ds)
        # Wrapping: 0->0, 90->90, 180->-180, 270->-90
        # Sorted: -180, -90, 0, 90  =>  original indices: 2, 3, 0, 1 => values: 3, 4, 1, 2
        expected_lons = np.array([-180.0, -90.0, 0.0, 90.0])
        np.testing.assert_array_equal(result["lon"].values, expected_lons)
        np.testing.assert_array_equal(result["temp"].values[0, 0, :], [3.0, 4.0, 1.0, 2.0])

    def test_metadata_added(self):
        base = self._make_base()
        ds = xr.Dataset(
            {"temp": (["time", "lat", "lon"], np.ones((1, 2, 3)))},
            coords={"time": [0], "lat": [10.0, 20.0], "lon": [5.0, 10.0, 15.0]},
        )
        result = base._standardize_coords(ds)
        assert result.attrs["product"] == PRODUCTS["era5land"].name
        assert result.attrs["units"] == PRODUCTS["era5land"].units
