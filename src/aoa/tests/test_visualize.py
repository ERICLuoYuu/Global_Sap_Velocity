"""Tests for visualize.py (M6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.aoa.visualize import parquet_to_geotiff

rasterio = pytest.importorskip("rasterio")


@pytest.fixture
def sample_di_df():
    """Small grid: 3 lat x 2 lon with known DI values."""
    lats = [50.2, 50.1, 50.0, 50.2, 50.1, 50.0]
    lons = [10.0, 10.0, 10.0, 10.1, 10.1, 10.1]
    return pd.DataFrame(
        {
            "latitude": lats,
            "longitude": lons,
            "median_DI": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "frac_inside_aoa": [1.0, 1.0, 1.0, 0.3, 0.3, 0.3],
        }
    )


@pytest.fixture
def per_ts_df():
    """Per-timestamp data with aoa_mask column."""
    return pd.DataFrame(
        {
            "latitude": [50.0, 50.1],
            "longitude": [10.0, 10.0],
            "DI": [0.2, 1.5],
            "aoa_mask": [True, False],
        }
    )


@pytest.mark.requires_rasterio
class TestGeoTIFFCRS:
    def test_crs_epsg4326(self, sample_di_df, tmp_path):
        out = tmp_path / "test.tif"
        parquet_to_geotiff(sample_di_df, out, threshold=0.5)
        with rasterio.open(out) as ds:
            assert ds.crs.to_epsg() == 4326


@pytest.mark.requires_rasterio
class TestGeoTIFFResolution:
    def test_pixel_size(self, sample_di_df, tmp_path):
        out = tmp_path / "test.tif"
        parquet_to_geotiff(sample_di_df, out, threshold=0.5, resolution=0.1)
        with rasterio.open(out) as ds:
            assert abs(ds.res[0] - 0.1) < 0.01
            assert abs(ds.res[1] - 0.1) < 0.01


@pytest.mark.requires_rasterio
class TestGeoTIFFBands:
    def test_two_bands(self, sample_di_df, tmp_path):
        out = tmp_path / "test.tif"
        parquet_to_geotiff(sample_di_df, out, threshold=0.5)
        with rasterio.open(out) as ds:
            assert ds.count == 2

    def test_dtype_float32(self, sample_di_df, tmp_path):
        out = tmp_path / "test.tif"
        parquet_to_geotiff(sample_di_df, out, threshold=0.5)
        with rasterio.open(out) as ds:
            assert ds.dtypes[0] == "float32"
            assert ds.dtypes[1] == "float32"


@pytest.mark.requires_rasterio
class TestAOAMaskValues:
    def test_mask_binary(self, sample_di_df, tmp_path):
        out = tmp_path / "test.tif"
        parquet_to_geotiff(sample_di_df, out, threshold=0.5)
        with rasterio.open(out) as ds:
            mask = ds.read(2)
            valid = mask[~np.isnan(mask)]
            assert set(valid.tolist()).issubset({0.0, 1.0})


@pytest.mark.requires_rasterio
class TestDINonNegative:
    def test_di_band_non_negative(self, sample_di_df, tmp_path):
        out = tmp_path / "test.tif"
        parquet_to_geotiff(sample_di_df, out, threshold=0.5)
        with rasterio.open(out) as ds:
            di = ds.read(1)
            valid = di[~np.isnan(di)]
            assert np.all(valid >= 0)


@pytest.mark.requires_rasterio
class TestPerTimestampAOAMask:
    def test_uses_aoa_mask_column(self, per_ts_df, tmp_path):
        out = tmp_path / "ts.tif"
        parquet_to_geotiff(per_ts_df, out, di_col="DI")
        with rasterio.open(out) as ds:
            mask = ds.read(2)
            valid = mask[~np.isnan(mask)]
            assert 1.0 in valid.tolist()
            assert 0.0 in valid.tolist()


@pytest.mark.requires_rasterio
class TestRoundtrip:
    def test_values_match_source(self, sample_di_df, tmp_path):
        out = tmp_path / "rt.tif"
        parquet_to_geotiff(sample_di_df, out, threshold=0.5)
        with rasterio.open(out) as ds:
            di = ds.read(1)
            valid = di[~np.isnan(di)]
            np.testing.assert_allclose(
                sorted(valid.tolist()),
                sorted(sample_di_df["median_DI"].tolist()),
                atol=1e-5,
            )
