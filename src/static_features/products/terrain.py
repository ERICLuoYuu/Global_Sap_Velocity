"""Terrain on the 1 km cell: ASTER GDEM elevation, slope and aspect at native scale, averaged over land pixels."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from src.static_features.earth_engine import (
    WORLDCOVER_ASSET,
    cell_means,
    initialise,
    land_mask,
    mean_and_count,
    reduce_cells,
    with_all_band,
)
from src.static_features.results import NO_DATA, OK, ProductResult, RunContext

DEM_ASSET = "projects/sat-io/open-datasets/ASTER/GDEM"
FLAT_SLOPE_DEG = 0.1
COLUMNS = {
    "elevation_m": ("elevation_m", 1.0),
    "slope_deg": ("slope_deg", 1.0),
    "aspect_sin": ("aspect_sin", 1.0),
    "aspect_cos": ("aspect_cos", 1.0),
}
PROVENANCE = {
    "source": DEM_ASSET,
    "version": "ASTER GDEM, Earth Engine community catalog copy",
    "checksum": None,
    "time_window": None,
    "method": (
        "Unweighted mean over land pixels centred in the 30 arc-second cell, in the DEM's own projection. "
        "Slope and aspect are computed per pixel at native scale; aspect is averaged as a vector, not renormalised, "
        f"with pixels flatter than {FLAT_SLOPE_DEG} degrees contributing (0, 0). land_share comes from ESA WorldCover "
        "v200 read on its own grid, where class 80 and masked pixels are water."
    ),
}


def aspect_vector(aspect_deg: Any, slope_deg: Any) -> Any:
    """Bands aspect_sin and aspect_cos. A flat pixel has no direction, so it contributes (0, 0)."""
    import ee

    radians = aspect_deg.multiply(math.pi / 180)
    steep = slope_deg.gte(FLAT_SLOPE_DEG)
    return ee.Image.cat(
        radians.sin().multiply(steep).rename("aspect_sin"),
        radians.cos().multiply(steep).rename("aspect_cos"),
    )


def terrain_result(cells: pd.DataFrame, land: pd.DataFrame) -> ProductResult:
    values, status, coverage = cell_means(cells, COLUMNS)
    land_share = land["land_mean"].reindex(values.index)
    return ProductResult(
        product="terrain",
        values=values.assign(land_share=land_share),
        status=status.assign(land_share=np.where(land_share.notna(), OK, NO_DATA)),
        coverage=coverage,
        by_year=None,
        extras={},
        provenance=PROVENANCE,
    )


def extract(context: RunContext) -> ProductResult:
    import ee

    initialise(context.project)
    dem = ee.Image(DEM_ASSET).select("b1")
    slope = ee.Terrain.slope(dem)
    land = land_mask()
    stack = ee.Image.cat(
        dem.rename("elevation_m"), slope.rename("slope_deg"), aspect_vector(ee.Terrain.aspect(dem), slope)
    ).updateMask(land)
    cells = reduce_cells(
        with_all_band(stack),
        mean_and_count(),
        dem.projection(),
        context.sites,
        [f"{band}_count" for band in COLUMNS] + ["all_count"],
        [f"{band}_mean" for band in COLUMNS],
        "terrain",
    )
    land_cells = reduce_cells(
        with_all_band(land),
        mean_and_count(),
        ee.ImageCollection(WORLDCOVER_ASSET).first().select("Map").projection(),
        context.sites,
        ["land_count", "all_count"],
        ["land_mean"],
        "terrain land share",
    )
    return terrain_result(cells, land_cells)
