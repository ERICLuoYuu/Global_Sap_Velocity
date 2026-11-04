"""Lesiv 2022 forest management, 2015: the share of each class among the 100 m pixels of a cell, and the dominant class."""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import rasterio
from src.static_features.fetch import fetch, sha256_of, zenodo_downloads
from src.static_features.grid import cell_bounds
from src.static_features.neighbours import Node, apply_borrowed, borrowed_table, fill_gaps
from src.static_features.rasters import centred_span
from src.static_features.results import NO_DATA, OK, ExtractionFailure, ProductResult, RunContext

RECORD_ID = 5879022
FILE_NAME = "FML_v3-2_with-colorbar.tif"
DIRECTORY = "lesiv_2022"
CLASSES = (11, 20, 31, 32, 40, 53)
NOT_FOREST = (0, 128)
DOMINANT = "fm_dominant"
CHUNK_SIZE = 64


def cell_shares(block: np.ndarray, nodata: float | None) -> tuple[dict[int, float], int | None]:
    """(share per class, dominant class). The denominator is every pixel; a tie goes to the lowest code."""
    if block.size == 0:
        raise ValueError("an empty block has no shares")
    allowed = list(CLASSES) + list(NOT_FOREST) + ([] if nodata is None else [nodata])
    unknown = np.unique(block[~np.isin(block, allowed)]).tolist()
    if unknown:
        raise ValueError(f"values {unknown} are not in the legend")
    shares = {code: float((block == code).sum() / block.size) for code in CLASSES}
    largest = max(shares.values())
    if largest == 0.0:
        return shares, None
    return shares, min(code for code in CLASSES if shares[code] == largest)


def _cell_block(src, row: int, col: int) -> np.ndarray | None:
    """The pixels centred in a cell, or None where no pixel of the map is."""
    west, south, east, north = cell_bounds(row, col)
    transform = src.transform
    row_start, row_stop = centred_span(south, north, transform.f, transform.e, src.height)
    col_start, col_stop = centred_span(west, east, transform.c, transform.a, src.width)
    if row_start == row_stop or col_start == col_stop:
        return None
    return src.read(1, window=((row_start, row_stop), (col_start, col_stop)))


def _read_cell(src, site) -> np.ndarray:
    block = _cell_block(src, site.row, site.col)
    if block is None:
        raise ExtractionFailure(f"lesiv: {site.site_name}: no pixel of the map is centred in the cell")
    return block


def _neighbour_dominants(src, nodes: Sequence[Node], columns: Sequence[str]) -> pd.DataFrame:
    """The dominant class of each cell, empty without forest or off the map. columns is always [DOMINANT]."""
    dominants = []
    for node in nodes:
        block = _cell_block(src, node.row, node.col)
        if block is None:
            dominants.append(math.nan)
            continue
        try:
            _, dominant = cell_shares(block, src.nodata)
        except ValueError as problem:
            raise ExtractionFailure(f"lesiv: cell ({node.row}, {node.col}): {problem}") from problem
        dominants.append(math.nan if dominant is None else dominant)
    return pd.DataFrame({DOMINANT: dominants})[list(columns)]


def extract(context: RunContext) -> ProductResult:
    path = context.raw_root / DIRECTORY / FILE_NAME
    columns = [f"fm_share_{code}" for code in CLASSES] + [DOMINANT]
    index = pd.Index(context.sites["site_name"], name="site_name")
    values, status, forest_share = [], [], []
    with rasterio.open(path) as src:
        for site in context.sites.itertuples(index=False):
            try:
                shares, dominant = cell_shares(_read_cell(src, site), src.nodata)
            except ValueError as problem:
                raise ExtractionFailure(f"lesiv: {site.site_name}: {problem}") from problem
            values.append([shares[code] for code in CLASSES] + [math.nan if dominant is None else dominant])
            status.append([OK] * len(CLASSES) + [NO_DATA if dominant is None else OK])
            forest_share.append(sum(shares.values()))
        status_frame = pd.DataFrame(status, index=index, columns=columns)
        empty = status_frame[[DOMINANT]] == NO_DATA
        borrowed = fill_gaps(context.sites, empty, functools.partial(_neighbour_dominants, src), CHUNK_SIZE)

    value_frame, status_frame = apply_borrowed(
        pd.DataFrame(values, index=index, columns=columns), status_frame, borrowed
    )
    return ProductResult(
        product="lesiv",
        values=value_frame,
        status=status_frame,
        coverage=pd.DataFrame({DOMINANT: forest_share}, index=index),
        by_year=None,
        extras={"lesiv_neighbours": borrowed_table(borrowed)},
        provenance={
            "source": f"https://zenodo.org/records/{RECORD_ID}, {FILE_NAME}",
            "version": "Lesiv et al. 2022, FML v3.2, 2015",
            "checksum": {FILE_NAME: sha256_of(path)},
            "time_window": "2015",
            "method": (
                "Share of every pixel centred in the cell, so a cell without forest is ok at 0; the dominant class "
                "takes the lowest code on a tie. A cell without forest borrows the dominant class of the nearest cell "
                "within 25 km that has forest (neighbour, owner 2026-11-04), its shares staying the cell's own. "
                "Coverage is the forest share of the cell."
            ),
        },
    )


def fetch_inputs(context: RunContext) -> None:
    directory = context.raw_root / DIRECTORY
    wanted = [item for item in zenodo_downloads(RECORD_ID, directory) if item.name == FILE_NAME]
    if not wanted:
        raise ExtractionFailure(f"lesiv: Zenodo record {RECORD_ID} no longer lists {FILE_NAME}")
    fetch(wanted[0], directory)
