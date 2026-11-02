"""Soil on the 1 km cell: the Earth Engine copy of SoilGrids, mean per property and depth."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.static_features.earth_engine import cell_means, initialise, mean_and_count, reduce_cells, with_all_band
from src.static_features.results import NO_DATA, OK, ProductResult, RunContext

ASSET = "projects/soilgrids-isric/{prop}_mean"
PROPERTIES = ("clay", "sand", "silt", "bdod", "cfvo", "soc", "phh2o", "nitrogen")
DEPTHS = ("0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm")
FACTOR = 0.1
FACTOR_BY_PROPERTY = {"bdod": 0.01, "nitrogen": 0.01}
PROVENANCE = {
    "source": "projects/soilgrids-isric/<property>_mean",
    "version": "SoilGrids250m 2.0, Earth Engine copy",
    "checksum": None,
    "time_window": None,
    "method": (
        "Unweighted mean over valid pixels centred in the 30 arc-second cell, each property in its own projection "
        "because bdod and soc have another origin. The old REST ring search is dropped: the cell covers that footprint. "
        "Conversion 0.1, except bdod and nitrogen 0.01."
    ),
}


def _depth_label(depth: str) -> str:
    return depth.replace("cm", "").replace("-", "_")


def texture_class(clay: float | None, sand: float | None, silt: float | None) -> str | None:
    """Simplified USDA class, as extract_soilgrid.calculate_derived_properties has it."""
    if pd.isna(clay) or pd.isna(sand) or pd.isna(silt):
        return None
    if clay >= 40:
        return "Clay"
    if sand >= 85:
        return "Sand"
    if silt >= 80:
        return "Silt"
    if clay >= 27 and sand <= 52:
        return "Clay Loam"
    if clay >= 20 and sand >= 45:
        return "Sandy Clay Loam"
    if silt >= 50:
        return "Silt Loam"
    if sand >= 70:
        return "Sandy Loam"
    return "Loam"


def soil_result(cells_by_property: dict[str, pd.DataFrame]) -> ProductResult:
    parts = []
    for prop in PROPERTIES:
        factor = FACTOR_BY_PROPERTY.get(prop, FACTOR)
        columns = {f"{prop}_{_depth_label(depth)}": (f"{prop}_{depth}_mean", factor) for depth in DEPTHS}
        parts.append(cell_means(cells_by_property[prop], columns))
    values = pd.concat([part[0] for part in parts], axis=1)
    status = pd.concat([part[1] for part in parts], axis=1)
    coverage = pd.concat([part[2] for part in parts], axis=1)

    classes = pd.Series(
        [
            texture_class(row.clay_0_5, row.sand_0_5, row.silt_0_5)
            for row in values[["clay_0_5", "sand_0_5", "silt_0_5"]].itertuples(index=False)
        ],
        index=values.index,
        dtype=object,
    )
    return ProductResult(
        product="soil",
        values=values.assign(texture_class_0_5=classes),
        status=status.assign(texture_class_0_5=np.where(classes.notna(), OK, NO_DATA)),
        coverage=coverage,
        by_year=None,
        extras={},
        provenance=PROVENANCE,
    )


def extract(context: RunContext) -> ProductResult:
    import ee

    initialise(context.project)
    cells_by_property = {}
    for prop in PROPERTIES:
        bands = [f"{prop}_{depth}_mean" for depth in DEPTHS]
        image = ee.Image(ASSET.format(prop=prop)).select(bands)
        cells_by_property[prop] = reduce_cells(
            with_all_band(image),
            mean_and_count(),
            image.select([bands[0]]).projection(),
            context.sites,
            [f"{band}_count" for band in bands] + ["all_count"],
            [f"{band}_mean" for band in bands],
            f"soil {prop}",
        )
    return soil_result(cells_by_property)
