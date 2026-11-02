"""Canopy height on the 1 km cell: Meta's 1.2 m map, over land, with canopy gaps counting as 0."""

from __future__ import annotations

import pandas as pd
from src.static_features.earth_engine import (
    cell_means,
    initialise,
    land_mask,
    mean_and_count,
    reduce_cells,
    with_all_band,
)
from src.static_features.results import ProductResult, RunContext

ASSET = "projects/sat-io/open-datasets/facebook/meta-canopy-height"
PROVENANCE = {
    "source": ASSET,
    "version": "Meta and WRI 1 m canopy height, Earth Engine community catalog copy, band cover_code",
    "checksum": None,
    "time_window": None,
    "method": (
        "Unweighted mean and 95th percentile (maxBuckets 1024, minBucketWidth 0.125) over land pixels centred in the "
        "30 arc-second cell, on a member image's own grid because a mosaic has no native projection. Gaps on land "
        "count as 0, so the mean behaves as cover x height."
    ),
}


def canopy_result(cells: pd.DataFrame) -> ProductResult:
    values, status, coverage = cell_means(cells, {"canopy_height_m": ("h", 1.0)})
    return ProductResult(
        product="canopy",
        values=values.assign(canopy_height_p95_m=cells["h_p95"]),
        status=status.assign(canopy_height_p95_m=status["canopy_height_m"]),
        coverage=coverage.assign(canopy_height_p95_m=coverage["canopy_height_m"]),
        by_year=None,
        extras={},
        provenance=PROVENANCE,
    )


def extract(context: RunContext) -> ProductResult:
    import ee

    initialise(context.project)
    collection = ee.ImageCollection(ASSET)
    height = collection.mosaic().select("cover_code").unmask(0).updateMask(land_mask()).rename("h")
    percentile = ee.Reducer.percentile([95], None, 1024, 0.125).unweighted()
    cells = reduce_cells(
        with_all_band(height),
        mean_and_count().combine(percentile, sharedInputs=True),
        collection.first().select("cover_code").projection(),
        context.sites,
        ["h_count", "all_count"],
        ["h_mean", "h_p95"],
        "canopy",
        batch_size=30,
    )
    return canopy_result(cells)
