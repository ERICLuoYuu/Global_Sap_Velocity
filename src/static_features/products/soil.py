"""Soil on the 1 km cell: the Earth Engine copy of SoilGrids, mean per property and depth."""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from src.static_features.earth_engine import cell_means, initialise, mean_and_count, reduce_cells, with_all_band
from src.static_features.neighbours import Borrowed, Node, apply_borrowed, borrowed_table, fill_gaps
from src.static_features.results import NEIGHBOUR, NO_DATA, OK, ProductResult, RunContext

ASSET = "projects/soilgrids-isric/{prop}_mean"
PROPERTIES = ("clay", "sand", "silt", "bdod", "cfvo", "soc", "phh2o", "nitrogen")
DEPTHS = ("0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm")
FACTOR = 0.1
FACTOR_BY_PROPERTY = {"bdod": 0.01, "nitrogen": 0.001}
TEXTURE = "texture_class_0_5"
TEXTURE_INPUTS = ["clay_0_5", "sand_0_5", "silt_0_5"]
CHUNK_SIZE = 60
PROVENANCE = {
    "source": "projects/soilgrids-isric/<property>_mean",
    "version": "SoilGrids250m 2.0, Earth Engine copy",
    "checksum": None,
    "time_window": None,
    "method": (
        "Unweighted mean over valid pixels centred in the 30 arc-second cell, each property in its own projection "
        "because bdod and soc have another origin. The old REST ring search is dropped: it could reach 750 m beyond the "
        "cell. A column the cell leaves empty takes the same mean from the nearest cell within 25 km holding it "
        "(neighbour, owner 2026-11-04). Conversion 0.1, except bdod 0.01 and nitrogen 0.001: this copy holds nitrogen "
        "ten times finer than the cg/kg of the REST service, so 0.01 would read ten times too high."
    ),
}


def _depth_label(depth: str) -> str:
    return depth.replace("cm", "").replace("-", "_")


def _columns_of(prop: str) -> dict[str, tuple[str, float]]:
    """Output column -> (reduced band, conversion factor), for one property."""
    factor = FACTOR_BY_PROPERTY.get(prop, FACTOR)
    return {f"{prop}_{_depth_label(depth)}": (f"{prop}_{depth}_mean", factor) for depth in DEPTHS}


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


def _texture(values: pd.DataFrame, status: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    """The class of each site, and its status: neighbour when any fraction it comes from was borrowed."""
    classes = pd.Series(
        [
            texture_class(row.clay_0_5, row.sand_0_5, row.silt_0_5)
            for row in values[TEXTURE_INPUTS].itertuples(index=False)
        ],
        index=values.index,
        dtype=object,
    )
    borrowed = (status[TEXTURE_INPUTS] == NEIGHBOUR).any(axis=1)
    return classes, np.where(classes.isna(), NO_DATA, np.where(borrowed, NEIGHBOUR, OK))


def soil_result(
    cells_by_property: Mapping[str, pd.DataFrame], borrowed: Mapping[str, Mapping[str, Borrowed]] | None = None
) -> ProductResult:
    """The cells' means, with any values borrowed from nearby cells in place of empty ones."""
    borrowed = {} if borrowed is None else borrowed
    parts = [cell_means(cells_by_property[prop], _columns_of(prop)) for prop in PROPERTIES]
    values = pd.concat([part[0] for part in parts], axis=1)
    status = pd.concat([part[1] for part in parts], axis=1)
    coverage = pd.concat([part[2] for part in parts], axis=1)
    values, status = apply_borrowed(values, status, borrowed)
    classes, texture_status = _texture(values, status)
    return ProductResult(
        product="soil",
        values=values.assign(**{TEXTURE: classes}),
        status=status.assign(**{TEXTURE: texture_status}),
        coverage=coverage,
        by_year=None,
        extras={"soil_neighbours": borrowed_table(borrowed)},
        provenance=PROVENANCE,
    )


def empty_columns(result: ProductResult) -> pd.DataFrame:
    """True where a site's own cell holds no value, over the measured columns; the texture class is derived."""
    columns = [column for column in result.status.columns if column != TEXTURE]
    return result.status[columns] == NO_DATA


def cell_frame(nodes: Sequence[Node]) -> pd.DataFrame:
    """Cells laid out as sites, so the reducers that read a site's cell read these."""
    return pd.DataFrame(
        {
            "site_name": [f"cell_{node.row}_{node.col}" for node in nodes],
            "row": [node.row for node in nodes],
            "col": [node.col for node in nodes],
        }
    )


def _reduce_property(image: Any, projection: Any, prop: str, sites: pd.DataFrame, label: str) -> pd.DataFrame:
    bands = [f"{prop}_{depth}_mean" for depth in DEPTHS]
    return reduce_cells(
        with_all_band(image),
        mean_and_count(),
        projection,
        sites,
        [f"{band}_count" for band in bands] + ["all_count"],
        [f"{band}_mean" for band in bands],
        label,
    )


def _neighbour_cells(
    images: Mapping[str, tuple[Any, Any]], nodes: Sequence[Node], columns: Sequence[str]
) -> pd.DataFrame:
    """The asked columns at each cell, one row per cell in order, reading only the properties asked for."""
    frame = cell_frame(nodes)
    pieces = []
    for prop in PROPERTIES:
        wanted = [column for column in _columns_of(prop) if column in columns]
        if not wanted:
            continue
        image, projection = images[prop]
        cells = _reduce_property(image, projection, prop, frame, f"soil {prop} neighbours")
        pieces.append(cell_means(cells, _columns_of(prop))[0][wanted])
    return pd.concat(pieces, axis=1).reset_index(drop=True)[list(columns)]


def extract(context: RunContext) -> ProductResult:
    import ee

    initialise(context.project)
    images, cells_by_property = {}, {}
    for prop in PROPERTIES:
        bands = [f"{prop}_{depth}_mean" for depth in DEPTHS]
        image = ee.Image(ASSET.format(prop=prop)).select(bands)
        projection = image.select([bands[0]]).projection()
        images[prop] = (image, projection)
        cells_by_property[prop] = _reduce_property(image, projection, prop, context.sites, f"soil {prop}")
    own = soil_result(cells_by_property)
    reader = functools.partial(_neighbour_cells, images)
    return soil_result(cells_by_property, fill_gaps(context.sites, empty_columns(own), reader, CHUNK_SIZE))
