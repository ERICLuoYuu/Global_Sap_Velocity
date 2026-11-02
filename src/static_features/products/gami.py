"""GAMI v2.1 forest age: within each ensemble member the mean of a cell's forest pixels, then the mean of the members."""

from __future__ import annotations

import logging
import math

import h5py
import numpy as np
import pandas as pd
from src.static_features.fetch import Download, fetch, sha256_of
from src.static_features.grid import cell_bounds
from src.static_features.rasters import centred_span, require_global_axes
from src.static_features.results import BY_YEAR_COLUMNS, NO_DATA, OK, ExtractionFailure, ProductResult, RunContext

LOGGER = logging.getLogger(__name__)

URL = "https://datapub.gfz.de/download/10.5880.GFZ.1.4.2023.006-VEnuo/GAMIv2-1_2010-2020_100m.nc"
FILE_NAME = "GAMIv2-1_2010-2020_100m.nc"
SIZE = 234035960874
DIRECTORY = "gami_v2_1"
COLUMN = "forest_age_gami"
DAYS_SINCE_2010 = {2010: 0, 2020: 3652}
MAP_YEAR = 2010
PIXELS_PER_DEGREE = 1125
# A chunk unpacks to 242 MB, so this holds 16 of them: enough for every site in the neighbourhood of a read.
CACHE_BYTES = 4 * 2**30
CACHE_SLOTS = 10007
CAUTION = (
    "The authors advise binning into age classes of at least two decades, because absolute ages at 100 m carry "
    "large uncertainty."
)


def cell_mean(block: np.ndarray, fill: int) -> tuple[float, float]:
    """(mean age of the forest pixels, share of the pixels that are forest) for one member's cell."""
    if block.size == 0:
        raise ValueError("an empty block has no forest share")
    forest = block != fill
    if not forest.any():
        return math.nan, 0.0
    return float(block[forest].mean()), float(forest.mean())


def read_order(
    sites: pd.DataFrame, chunk_rows: int, chunk_cols: int, pixels_per_degree: float = PIXELS_PER_DEGREE
) -> list[int]:
    """Row positions of the sites sorted by the chunk holding each, so one chunk is unpacked once per pass."""
    chunk_row = np.floor((90.0 - sites["lat"].to_numpy()) * pixels_per_degree / chunk_rows)
    chunk_col = np.floor((sites["lon"].to_numpy() + 180.0) * pixels_per_degree / chunk_cols)
    return np.lexsort((chunk_col, chunk_row)).tolist()


def _time_positions(days: np.ndarray) -> dict[int, int]:
    found = days.tolist()
    absent = [year for year, day in DAYS_SINCE_2010.items() if day not in found]
    if absent:
        raise ExtractionFailure(f"gami: time holds days {found} since 2010-01-01, without years {absent}")
    return {year: found.index(day) for year, day in DAYS_SINCE_2010.items()}


def _cell_spans(sites: pd.DataFrame, n_rows: int, n_cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    spans = []
    for site in sites.itertuples(index=False):
        west, south, east, north = cell_bounds(site.row, site.col)
        spans.append(
            (
                centred_span(south, north, 90.0, -180.0 / n_rows, n_rows),
                centred_span(west, east, -180.0, 360.0 / n_cols, n_cols),
            )
        )
    return spans


def _across_members(means: np.ndarray) -> float:
    """The mean of the members holding forest in the cell. No forest in any member is no value."""
    forest = means[~np.isnan(means)]
    return float(forest.mean()) if forest.size else math.nan


def _status_of(age: float) -> str:
    return NO_DATA if math.isnan(age) else OK


def _read_members(
    dataset: h5py.Dataset, fill: int, order: list[int], spans: list, time: int
) -> tuple[np.ndarray, np.ndarray]:
    """(mean age, forest share) with shape (site, member), one chunk at a time."""
    n_members = dataset.shape[0]
    means = np.full((len(spans), n_members), np.nan)
    shares = np.zeros((len(spans), n_members))
    for member in range(n_members):
        for position in order:
            (row_start, row_stop), (col_start, col_stop) = spans[position]
            block = dataset[member, row_start:row_stop, col_start:col_stop, time]
            means[position, member], shares[position, member] = cell_mean(block, fill)
        LOGGER.info("gami: time position %d, member %d of %d done", time, member + 1, n_members)
    return means, shares


def extract(context: RunContext) -> ProductResult:
    path = context.raw_root / DIRECTORY / FILE_NAME
    names = context.sites["site_name"].tolist()
    ages, shares = {}, {}
    with h5py.File(path, "r", rdcc_nbytes=CACHE_BYTES, rdcc_nslots=CACHE_SLOTS) as handle:
        positions = _time_positions(handle["time"][:])
        require_global_axes("gami", handle["latitude"][:], handle["longitude"][:])
        fill = int(np.asarray(handle.attrs["_FillValue"]).item())
        dataset = handle["forest_age"]
        _, n_rows, n_cols, _ = dataset.shape
        spans = _cell_spans(context.sites, n_rows, n_cols)
        order = read_order(context.sites, dataset.chunks[1], dataset.chunks[2], n_rows / 180.0)
        for year, time in positions.items():
            means, year_shares = _read_members(dataset, fill, order, spans, time)
            ages[year] = [_across_members(means[position]) for position in range(len(names))]
            shares[year] = year_shares

    rows = [
        (name, year, COLUMN, ages[year][position], _status_of(ages[year][position]))
        for year in positions
        for position, name in enumerate(names)
    ]
    index = pd.Index(names, name="site_name")
    return ProductResult(
        product="gami",
        values=pd.DataFrame({COLUMN: ages[MAP_YEAR]}, index=index),
        status=pd.DataFrame({COLUMN: [_status_of(age) for age in ages[MAP_YEAR]]}, index=index),
        coverage=pd.DataFrame({COLUMN: shares[MAP_YEAR].mean(axis=1)}, index=index),
        by_year=pd.DataFrame(rows, columns=list(BY_YEAR_COLUMNS)),
        extras={},
        provenance={
            "source": URL,
            "version": "GAMI v2.1, 100 m, 20 members",
            "checksum": {FILE_NAME: sha256_of(path)},
            "time_window": str(MAP_YEAR),
            "method": (
                "Per member, the mean of the forest pixels centred in the cell; then the mean of the members with a "
                "forest pixel. Coverage is the forest share of the cell, averaged over all members."
            ),
            "caution": CAUTION,
        },
    )


def fetch_inputs(context: RunContext) -> None:
    fetch(Download(URL, FILE_NAME, size=SIZE), context.raw_root / DIRECTORY)
