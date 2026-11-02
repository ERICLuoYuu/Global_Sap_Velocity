"""Live Earth Engine checks, skipped unless STATIC_LIVE_TESTS=1. Their numbers were measured on 2026-11-02."""

from __future__ import annotations

import os

import pandas as pd
import pytest
from src.static_features.earth_engine import (
    cell_collection,
    initialise,
    land_mask,
    mean_and_count,
    reduce_cells,
    with_all_band,
)
from src.static_features.grid import cell_of

pytestmark = pytest.mark.skipif(os.environ.get("STATIC_LIVE_TESTS") != "1", reason="set STATIC_LIVE_TESTS=1")

PROJECT = "era5download-447713"
COORDINATES = {
    "GBR_DEV_CON": (56.03333, -3.716667),
    "GBR_DEV_DRO": (56.03333, -3.716667),
    "CHE_DAV_SEE": (46.81668, 9.856198),
    "AUS_WOM": (-37.4222, 144.0944),
}


def _sites(*names):
    rows = []
    for name in names:
        lat, lon = COORDINATES[name]
        row, col = cell_of(lat, lon)
        rows.append({"site_name": name, "lat": lat, "lon": lon, "row": row, "col": col})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module", autouse=True)
def earth_engine_session():
    initialise(PROJECT)


def test_live_land_share_matches_measured():
    import ee

    projection = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").projection()
    frame = reduce_cells(
        with_all_band(land_mask()),
        mean_and_count(),
        projection,
        _sites("GBR_DEV_CON", "GBR_DEV_DRO", "CHE_DAV_SEE", "AUS_WOM"),
        ["land_count", "all_count"],
        ["land_mean"],
        "live land share",
    )

    assert frame.loc["GBR_DEV_CON", "land_mean"] == pytest.approx(0.37, abs=0.02)
    assert frame.loc["CHE_DAV_SEE", "land_mean"] == pytest.approx(0.30, abs=0.02)
    assert frame.loc["AUS_WOM", "land_mean"] > 0.99
    # Review Focus 4: two sites in one cell get a row each, with identical values.
    assert frame.loc["GBR_DEV_DRO", "land_mean"] == frame.loc["GBR_DEV_CON", "land_mean"]
    # Masked WorldCover pixels count as water, so the mask itself never drops a pixel.
    assert (frame["land_count"] == frame["all_count"]).all()


def test_live_count_counts_centres_not_touches():
    import ee

    image = ee.ImageCollection("MODIS/061/MOD44B").first().select("Percent_Tree_Cover")
    frame = reduce_cells(
        with_all_band(image),
        mean_and_count(),
        image.projection(),
        _sites("AUS_WOM"),
        ["all_count", "Percent_Tree_Cover_count"],
        ["Percent_Tree_Cover_mean"],
        "live count",
    )
    # About 13 pixel centres lie in the cell, against the 22 pixels that touch it.
    assert 11 <= frame.loc["AUS_WOM", "all_count"] <= 15


def test_live_mask_reads_its_own_grid():
    import ee

    modis = ee.ImageCollection("MODIS/061/MOD13Q1").first().select("EVI").projection()
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    sampled = with_all_band(land_mask()).sampleRegions(
        collection=cell_collection(_sites("GBR_DEV_CON", "CHE_DAV_SEE")),
        properties=["site_name"],
        projection=modis,
        geometries=True,
    )
    native = worldcover.unmask(0).reduceRegions(
        collection=sampled, reducer=ee.Reducer.first().unweighted(), crs=worldcover.projection()
    )
    rows = [feature["properties"] for feature in native.getInfo()["features"]]

    assert len(rows) >= 20
    assert {row["land"] for row in rows} == {0, 1}  # these two cells hold both land and water
    for row in rows:
        assert row["land"] == int(row["first"] not in (0, 80)), row
