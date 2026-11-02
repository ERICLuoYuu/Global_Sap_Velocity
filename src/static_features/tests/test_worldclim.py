from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from src.static_features.products.worldclim import extract
from src.static_features.results import NEIGHBOUR, NO_DATA, OK, ExtractionFailure, RunContext, validate

BIO_NODATA = -3.4e38


def _write(path, array, nodata=None, dtype="float32", west=0.0, north=9.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = array.shape
    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": 1, "dtype": dtype,
        "crs": "EPSG:4326", "transform": from_origin(west, north, 1.0, 1.0), "nodata": nodata,
    }  # fmt: skip
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)


def _rasters(tmp_path, bio=None, koppen=None, west=0.0):
    """bio: 9 x 9 float array or None for 10 everywhere, written times n for band n. koppen: 9 x 9 codes."""
    bio = np.full((9, 9), 10.0, dtype="float32") if bio is None else bio
    koppen = np.full((9, 9), 15, dtype="uint8") if koppen is None else koppen
    for n in range(1, 20):
        scaled = np.where(bio == BIO_NODATA, BIO_NODATA, np.where(bio == BIO_NODATA, 0.0, bio) * n)
        _write(tmp_path / f"spatial_features/bio_30s/wc2.1_30s_bio_{n}.tif", scaled, nodata=BIO_NODATA, west=west)
    _write(tmp_path / "koppen_geiger/koppen/Beck_KG_V1_present_0p0083.tif", koppen, dtype="uint8", west=west)


def _context(tmp_path, lat=4.5, lon=4.5, name="A"):
    sites = pd.DataFrame({"site_name": [name], "lat": [lat], "lon": [lon]})
    return RunContext(sites=sites, raw_root=tmp_path, output_root=tmp_path / "out", project="p")


def test_bio_reads_the_own_pixel_of_each_band(tmp_path):
    _rasters(tmp_path)
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "bio1"] == 10.0
    assert result.values.loc["A", "bio19"] == 190.0
    assert result.status.loc["A", "bio7"] == OK
    assert list(result.values.columns) == [f"bio{n}" for n in range(1, 20)] + ["koppen_class", "koppen_name"]
    validate(result, ["A"])


def test_bio_nodata_takes_the_nearest_valid_pixel(tmp_path):
    bio = np.full((9, 9), BIO_NODATA, dtype="float32")
    bio[4, 3] = 2.0  # one pixel west of the site
    bio[3, 3] = 9.0  # the corner, farther away
    _rasters(tmp_path, bio=bio)
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "bio1"] == 2.0
    assert result.status.loc["A", "bio1"] == NEIGHBOUR


def test_bio_reach_is_three_pixels(tmp_path):
    three_away = np.full((9, 9), BIO_NODATA, dtype="float32")
    three_away[4, 1] = 5.0
    _rasters(tmp_path, bio=three_away)
    assert extract(_context(tmp_path)).status.loc["A", "bio1"] == NEIGHBOUR

    four_away = np.full((9, 9), BIO_NODATA, dtype="float32")
    four_away[4, 0] = 5.0
    _rasters(tmp_path, bio=four_away)
    result = extract(_context(tmp_path))
    assert result.status.loc["A", "bio1"] == NO_DATA and np.isnan(result.values.loc["A", "bio1"])
    validate(result, ["A"])


def test_search_does_not_wrap_the_dateline(tmp_path):
    # Unchanged products keep today's lookup, which clips at the raster's edge.
    bio = np.full((9, 9), BIO_NODATA, dtype="float32")
    bio[4, 8] = 5.0
    _rasters(tmp_path, bio=bio, west=-180.0)
    result = extract(_context(tmp_path, lat=4.5, lon=-179.5))
    assert result.status.loc["A", "bio1"] == NO_DATA


def test_koppen_land_reads_its_class_and_name(tmp_path):
    _rasters(tmp_path)
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "koppen_class"] == 15
    assert result.values.loc["A", "koppen_name"] == "Cfb - Temperate, no dry season and warm summer"
    assert result.status.loc["A", "koppen_class"] == OK
    assert result.status.loc["A", "koppen_name"] == OK


def test_koppen_sea_takes_nearest_land(tmp_path):
    # The raster declares no nodata and encodes ocean as 0, so 0 has to be named as invalid.
    koppen = np.zeros((9, 9), dtype="uint8")
    koppen[4, 5] = 27
    _rasters(tmp_path, koppen=koppen)
    result = extract(_context(tmp_path))

    assert result.values.loc["A", "koppen_class"] == 27
    assert result.values.loc["A", "koppen_name"] == "Dfc - Cold, no dry season and cold summer"
    assert result.status.loc["A", "koppen_class"] == NEIGHBOUR


def test_koppen_with_no_land_in_reach_is_no_data(tmp_path):
    _rasters(tmp_path, koppen=np.zeros((9, 9), dtype="uint8"))
    result = extract(_context(tmp_path))

    assert np.isnan(result.values.loc["A", "koppen_class"])
    assert pd.isna(result.values.loc["A", "koppen_name"])
    assert result.status.loc["A", "koppen_name"] == NO_DATA
    validate(result, ["A"])


def test_koppen_code_outside_legend_stops(tmp_path):
    _rasters(tmp_path, koppen=np.full((9, 9), 31, dtype="uint8"))
    with pytest.raises(ExtractionFailure, match="A.*31"):
        extract(_context(tmp_path))
