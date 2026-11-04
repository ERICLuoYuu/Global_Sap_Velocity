"""The one rule for borrowing a value: the nearest node holding one, within SEARCH_KM of the site, by ground distance.

Owner, 2026-11-04: the gaps left in the merged products are filled by neighbour search. 25 km is the farthest the
hold-out tests of 2026-11-03 reached (borrowed against known values, rho 0.83 to 0.94), so no value comes from farther
than has been measured. A node is a pixel of the product's own grid, or a 30 arc-second cell where the product is
read by cell; either way the distance is from the site to the node's centre.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from src.static_features.grid import CELLS_PER_DEGREE, N_COLS, N_ROWS
from src.static_features.results import NEIGHBOUR

KM_PER_DEGREE = 111.195
SEARCH_KM = 25.0
MIN_COS_LAT = 0.01  # keeps the longitude span finite next to a pole
TIE_DECIMALS = 6  # distances equal to the millimetre are a tie, broken by row, then column
BORROWED_COLUMNS = ["site_name", "column", "distance_km", "source_lat", "source_lon"]


@dataclass(frozen=True)
class Node:
    distance_km: float
    row: int
    col: int
    lat: float
    lon: float


@dataclass(frozen=True)
class Borrowed:
    value: Any
    node: Node


def _east_degrees(lons: np.ndarray, lon: float) -> np.ndarray:
    """Longitude differences from lon, taken the short way round the dateline."""
    delta = lons - lon
    return np.where(delta > 180.0, delta - 360.0, np.where(delta < -180.0, delta + 360.0, delta))


def nodes_by_distance(
    lat: float, lon: float, lats: np.ndarray, lons: np.ndarray, max_km: float = SEARCH_KM
) -> list[Node]:
    """Every node of a latitude-by-longitude grid within max_km of the site, nearest first, a tie by row then column.

    lats and lons are the centres along each axis. The distance is equirectangular at the site's latitude, within
    0.1% of the great circle over 25 km.
    """
    lats, lons = np.asarray(lats, dtype=float), np.asarray(lons, dtype=float)
    north_km = (lats - lat) * KM_PER_DEGREE
    east_km = _east_degrees(lons, lon) * KM_PER_DEGREE * max(math.cos(math.radians(lat)), MIN_COS_LAT)
    rows = np.flatnonzero(np.abs(north_km) <= max_km)
    cols = np.flatnonzero(np.abs(east_km) <= max_km)
    distance = np.hypot(north_km[rows][:, None], east_km[cols][None, :])
    inside_row, inside_col = np.nonzero(distance <= max_km)
    kept = distance[inside_row, inside_col]
    order = np.lexsort((cols[inside_col], rows[inside_row], np.round(kept, TIE_DECIMALS)))
    return [
        Node(
            distance_km=float(kept[position]),
            row=int(rows[inside_row[position]]),
            col=int(cols[inside_col[position]]),
            lat=float(lats[rows[inside_row[position]]]),
            lon=float(lons[cols[inside_col[position]]]),
        )
        for position in order
    ]


def cells_by_distance(lat: float, lon: float, max_km: float = SEARCH_KM) -> list[Node]:
    """The 30 arc-second cells within max_km of the site, nearest first; the site's own cell is normally the first."""
    step = 1.0 / CELLS_PER_DEGREE
    lats = 90.0 - (np.arange(N_ROWS) + 0.5) * step
    lons = -180.0 + (np.arange(N_COLS) + 0.5) * step
    return nodes_by_distance(lat, lon, lats, lons, max_km)


def nearest_values(
    nodes: Sequence[Node],
    evaluate: Callable[[Sequence[Node]], pd.DataFrame],
    columns: Sequence[str],
    chunk_size: int,
) -> dict[str, Borrowed]:
    """For each column, the value at the nearest node holding one. A column no node holds is left out.

    evaluate gets the nodes a chunk at a time, nearest first, and returns one row per node in that order, empty where
    the node holds nothing. Reading stops once every column has its value, so a site beside data costs one chunk.
    """
    found: dict[str, Borrowed] = {}
    for start in range(0, len(nodes), chunk_size):
        wanted = [column for column in columns if column not in found]
        if not wanted:
            break
        chunk = list(nodes[start : start + chunk_size])
        table = evaluate(chunk)
        if len(table) != len(chunk):
            raise ValueError(f"evaluate returned {len(table)} rows for {len(chunk)} nodes")
        for column in wanted:
            held = np.flatnonzero(table[column].notna().to_numpy())
            if held.size:
                found[column] = Borrowed(table[column].iloc[held[0]], chunk[held[0]])
    return found


def fill_gaps(
    sites: pd.DataFrame,
    empty: pd.DataFrame,
    evaluate: Callable[..., pd.DataFrame],
    chunk_size: int,
    max_km: float = SEARCH_KM,
) -> dict[str, dict[str, Borrowed]]:
    """Per site, the nearest cell values for the columns its own cell left empty.

    empty is indexed by site_name, True where the site's own cell holds no value. evaluate(nodes, columns=...) reads
    the asked columns at those cells, as nearest_values describes.
    """
    found = {}
    for site in sites.itertuples(index=False):
        columns = [column for column in empty.columns if bool(empty.at[site.site_name, column])]
        if not columns:
            continue
        reader = functools.partial(evaluate, columns=columns)
        hits = nearest_values(cells_by_distance(site.lat, site.lon, max_km), reader, columns, chunk_size)
        if hits:
            found[site.site_name] = hits
    return found


def apply_borrowed(
    values: pd.DataFrame, status: pd.DataFrame, borrowed: Mapping[str, Mapping[str, Borrowed]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copies of values and status with each borrowed value in place and its status neighbour."""
    new_values, new_status = values.copy(), status.copy()
    for site, hits in borrowed.items():
        for column, hit in hits.items():
            new_values.loc[site, column] = hit.value
            new_status.loc[site, column] = NEIGHBOUR
    return new_values, new_status


def borrowed_table(borrowed: Mapping[str, Mapping[str, Borrowed]]) -> pd.DataFrame:
    """One row per borrowed value: the site, the column, and how far away and where the value was read."""
    rows = [
        (site, column, hit.node.distance_km, hit.node.lat, hit.node.lon)
        for site, hits in sorted(borrowed.items())
        for column, hit in sorted(hits.items())
    ]
    return pd.DataFrame(rows, columns=BORROWED_COLUMNS)
