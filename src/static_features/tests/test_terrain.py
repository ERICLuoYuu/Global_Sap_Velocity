from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import pytest
from src.static_features.earth_engine import cell_means, initialise, mean_and_count, reduce_cells, with_all_band
from src.static_features.grid import cell_bounds, cell_of
from src.static_features.products.terrain import aspect_vector, terrain_result
from src.static_features.results import NO_DATA, OK, validate

BANDS = ("elevation_m", "slope_deg", "aspect_sin", "aspect_cos")


def _cells(site, mean, count, all_count):
    row = {"all_count": float(all_count)}
    for band in BANDS:
        row[f"{band}_mean"] = mean
        row[f"{band}_count"] = float(count)
    return pd.DataFrame([row], index=pd.Index([site], name="site_name"))


def _land(site, share, all_count):
    row = {"land_mean": share, "land_count": float(all_count), "all_count": float(all_count)}
    return pd.DataFrame([row], index=pd.Index([site], name="site_name"))


def test_cell_with_no_land_is_no_data_with_zero_coverage():
    result = terrain_result(_cells("A", np.nan, 0, 13), _land("A", 0.0, 13))

    assert np.isnan(result.values.loc["A", "elevation_m"])
    assert result.status.loc["A", "elevation_m"] == NO_DATA
    assert result.coverage.loc["A", "elevation_m"] == 0.0
    assert result.values.loc["A", "land_share"] == 0.0
    assert result.status.loc["A", "land_share"] == OK
    validate(result, ["A"])


def test_terrain_coverage_is_land_pixels_over_all_pixels():
    result = terrain_result(_cells("A", 5.0, 10, 16), _land("A", 0.625, 4))

    assert result.coverage.loc["A", "slope_deg"] == 0.625
    assert result.values.loc["A", "aspect_cos"] == 5.0
    assert result.status.loc["A", "aspect_sin"] == OK
    assert list(result.values.columns) == ["elevation_m", "slope_deg", "aspect_sin", "aspect_cos", "land_share"]
    validate(result, ["A"])


def test_terrain_lines_up_sites_given_in_a_different_order():
    cells = pd.concat([_cells("A", 1.0, 16, 16), _cells("B", 2.0, 16, 16)])
    land = pd.concat([_land("B", 0.2, 4), _land("A", 0.1, 4)])
    result = terrain_result(cells, land)
    assert result.values.loc["A", "land_share"] == 0.1 and result.values.loc["B", "land_share"] == 0.2


def test_cell_means_scale_values_and_share_their_count():
    cells = pd.DataFrame(
        {"b_mean": [400.0, np.nan], "b_count": [3.0, 0.0], "all_count": [4.0, 4.0]},
        index=pd.Index(["A", "B"], name="site_name"),
    )
    values, status, coverage = cell_means(cells, {"sand": ("b", 0.1)})

    assert values.loc["A", "sand"] == pytest.approx(40.0)
    assert np.isnan(values.loc["B", "sand"])
    assert list(status["sand"]) == [OK, NO_DATA]
    assert list(coverage["sand"]) == [0.75, 0.0]


@pytest.mark.skipif(os.environ.get("STATIC_LIVE_TESTS") != "1", reason="set STATIC_LIVE_TESTS=1")
def test_live_aspect_seam():
    import ee

    initialise("era5download-447713")
    row, col = cell_of(-37.4222, 144.0944)
    west, _south, east, _north = cell_bounds(row, col)
    sites = pd.DataFrame([{"site_name": "AUS_WOM", "row": row, "col": col}])
    longitude = ee.Image.pixelLonLat().select("longitude")
    aspect = ee.Image.constant(355).where(longitude.gte((west + east) / 2), 5).toFloat()
    projection = ee.Projection("EPSG:4326").atScale(30)
    counts = ["aspect_sin_count", "aspect_cos_count", "all_count"]
    values = ["aspect_sin_mean", "aspect_cos_mean"]

    steep = reduce_cells(
        with_all_band(aspect_vector(aspect, ee.Image.constant(10))),
        mean_and_count(),
        projection,
        sites,
        counts,
        values,
        "live aspect seam",
    )
    flat = reduce_cells(
        with_all_band(aspect_vector(aspect, ee.Image.constant(0))),
        mean_and_count(),
        projection,
        sites,
        counts,
        values,
        "live aspect flat",
    )

    # A mean of degrees would give 180, due south; the vector gives north, with the sines cancelling.
    assert steep.loc["AUS_WOM", "aspect_sin_mean"] == pytest.approx(0.0, abs=0.01)
    assert steep.loc["AUS_WOM", "aspect_cos_mean"] == pytest.approx(math.cos(math.radians(5)), abs=0.001)
    assert flat.loc["AUS_WOM", "aspect_sin_mean"] == 0.0
    assert flat.loc["AUS_WOM", "aspect_cos_mean"] == 0.0
    assert flat.loc["AUS_WOM", "aspect_cos_count"] == flat.loc["AUS_WOM", "all_count"]  # flat pixels still count
