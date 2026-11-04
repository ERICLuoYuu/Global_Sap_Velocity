"""BGI forest age at the site's own node, and apart from it an age borrowed from the nearest node, for the merge's last resort.

The merge takes the mean of the four thresholds first, the ground st_age next, and the borrowed age last, so a value
borrowed from a pixel kilometres away never displaces an age measured at the stand.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import h5py
import numpy as np
import pandas as pd
from src.static_features.neighbours import (
    Borrowed,
    Node,
    apply_borrowed,
    borrowed_table,
    nearest_values,
    nodes_by_distance,
)
from src.static_features.results import NO_DATA, OK, ExtractionFailure, ProductResult, RunContext

THRESHOLDS = ("TC000", "TC010", "TC020", "TC030")
MAX_AGE = 500.0
NEIGHBOUR_COLUMN = "stand_age_neighbour"
CHUNK_SIZE = 256
PROVENANCE = {
    "source": "https://doi.org/10.17871/ForestAgeBGI.2021",
    "version": "BGIForestAge MPI-BGC 1.0.0 (Besnard et al. 2021)",
    "checksum": None,
    "time_window": "circa 2010",
    "method": (
        "Each threshold at the nearest node, valid when 0 < age <= 500, with no search. Where the node holds no "
        "threshold, stand_age_neighbour is the mean of the valid thresholds at the nearest node within 25 km holding "
        "any (owner, 2026-11-04). The merge reads it only after the ground st_age."
    ),
}


def age_value(raw: float) -> float:
    return float(raw) if 0.0 < raw <= MAX_AGE else math.nan  # NaN fails the comparison


def _read_node(dataset: h5py.Dataset, lat_index: int, lon_index: int, n_lats: int) -> float:
    if dataset.shape[0] == n_lats:
        return np.asarray(dataset[lat_index, lon_index]).item()
    return np.asarray(dataset[lon_index, lat_index]).item()


def _window_ages(datasets: Sequence[h5py.Dataset], n_lats: int, rows: slice, cols: slice) -> np.ndarray:
    """The mean of the valid thresholds at each node of a window, empty where none is valid."""
    stack = []
    for dataset in datasets:
        block = dataset[rows, cols] if dataset.shape[0] == n_lats else dataset[cols, rows].T
        block = np.asarray(block, dtype=float)
        stack.append(np.where((block > 0.0) & (block <= MAX_AGE), block, np.nan))
    ages = np.stack(stack)
    count = np.isfinite(ages).sum(axis=0)
    return np.where(count > 0, np.nansum(ages, axis=0) / np.maximum(count, 1), np.nan)


def _borrow(
    datasets: Sequence[h5py.Dataset], lats: np.ndarray, lons: np.ndarray, lat: float, lon: float
) -> Borrowed | None:
    nodes = nodes_by_distance(lat, lon, lats, lons)
    if not nodes:
        return None
    row_start, col_start = min(node.row for node in nodes), min(node.col for node in nodes)
    rows = slice(row_start, max(node.row for node in nodes) + 1)
    cols = slice(col_start, max(node.col for node in nodes) + 1)
    ages = _window_ages(datasets, len(lats), rows, cols)

    def evaluate(chunk: Sequence[Node]) -> pd.DataFrame:
        return pd.DataFrame({NEIGHBOUR_COLUMN: [ages[node.row - row_start, node.col - col_start] for node in chunk]})

    return nearest_values(nodes, evaluate, [NEIGHBOUR_COLUMN], CHUNK_SIZE).get(NEIGHBOUR_COLUMN)


def extract(context: RunContext) -> ProductResult:
    directory = context.raw_root / "stand_age"
    paths = sorted(directory.glob("*BGIForestAge*.nc"))
    if len(paths) != 1:
        raise ExtractionFailure(f"stand_age: expected one *BGIForestAge*.nc in {directory}, found {len(paths)}")

    sites = context.sites
    index = pd.Index(sites["site_name"], name="site_name")
    values = {}
    borrowed: dict[str, dict[str, Borrowed]] = {}
    with h5py.File(paths[0], "r") as handle:
        lats = np.asarray(handle["latitude"][:]).flatten()
        lons = np.asarray(handle["longitude"][:]).flatten()
        datasets = [handle[f"ForestAge_{threshold}"] for threshold in THRESHOLDS]
        for threshold in THRESHOLDS:
            dataset = handle[f"ForestAge_{threshold}"]
            column = []
            for site in sites.itertuples(index=False):
                lat_index = int(np.abs(lats - site.lat).argmin())
                lon_index = int(np.abs(lons - site.lon).argmin())
                column.append(age_value(_read_node(dataset, lat_index, lon_index, len(lats))))
            values[f"stand_age_{threshold}"] = column
        own = pd.DataFrame(values, index=index)
        for site in sites.itertuples(index=False):
            if own.loc[site.site_name].isna().all():
                hit = _borrow(datasets, lats, lons, site.lat, site.lon)
                if hit is not None:
                    borrowed[site.site_name] = {NEIGHBOUR_COLUMN: hit}

    frame = own.assign(**{NEIGHBOUR_COLUMN: math.nan})
    status = frame.notna().replace({True: OK, False: NO_DATA})
    frame, status = apply_borrowed(frame, status, borrowed)
    return ProductResult(
        product="stand_age",
        values=frame,
        status=status,
        coverage=pd.DataFrame(index=index),
        by_year=None,
        extras={"stand_age_neighbours": borrowed_table(borrowed)},
        provenance=PROVENANCE,
    )
