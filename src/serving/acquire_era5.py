"""Stage A: ERA5-Land daily base columns for every served cell, from Earth Engine, into the store.

Each UTC day is :func:`src.serving.era5_gee_daily.daily_image` (parity-checked against the merge's own
daily statistics to 2e-12), fetched in native-grid windows that hold at least one served cell and written
into ``era5_daily/<year>/<column>.npy``. A day counts as acquired only once every window landed and the
arrays were flushed; its marker then records what came back. A rerun skips marked days, so a killed job
resumes where it stopped. A day on which no cell came back valid (not yet in the collection) is left
unmarked and makes the run exit non-zero.

    python -m src.serving.acquire_era5 --store S --year 2015 [--start 2015-10-01] [--end 2015-12-31]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.feature_contract.definitions import SWC_LAYERS, swc_anomaly_col
from src.serving.base_columns import ERA5_COLUMNS
from src.serving.store import Store, load_grid, open_year_column, write_json, year_days

logger = logging.getLogger("acquire_era5")

GEE_PROJECT = "era5download-447713"
TILE_HEIGHT = 460
TILE_WIDTH = 600
DEFAULT_THREADS = 12
VALIDITY_COLUMN = "ta"
_ANOMALY_COLUMNS = frozenset(swc_anomaly_col(layer) for layer in SWC_LAYERS)
STORE_COLUMNS = tuple(column for column in ERA5_COLUMNS if column not in _ANOMALY_COLUMNS)


@dataclass(frozen=True, eq=False)
class Window:
    row0: int
    col0: int
    height: int
    width: int
    cells: np.ndarray


def cell_windows(rows, cols, grid, height: int = TILE_HEIGHT, width: int = TILE_WIDTH) -> list:
    """The ``height`` x ``width`` grid windows that hold a cell, each with the (sorted) indices of its cells."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if len(rows) == 0:
        raise ValueError("no cells to acquire")
    if rows.min() < 0 or rows.max() >= grid.n_rows or cols.min() < 0 or cols.max() >= grid.n_cols:
        raise ValueError("a cell lies outside the grid")
    n_tile_cols = -(-grid.n_cols // width)
    key = (rows // height) * n_tile_cols + cols // width
    windows = []
    for value in np.unique(key):
        row0, col0 = int(value // n_tile_cols) * height, int(value % n_tile_cols) * width
        windows.append(
            Window(
                row0=row0,
                col0=col0,
                height=min(height, grid.n_rows - row0),
                width=min(width, grid.n_cols - col0),
                cells=np.flatnonzero(key == value),
            )
        )
    return windows


def write_window(arrays: dict, day_index: int, window: Window, rows, cols, values: dict) -> int:
    """Copy one fetched window into the column arrays; returns how many of its cells came back valid."""
    missing = sorted(set(arrays) - set(values))
    if missing:
        raise KeyError(f"fetched window lacks columns {missing}")
    r = np.asarray(rows)[window.cells] - window.row0
    c = np.asarray(cols)[window.cells] - window.col0
    for column, array in arrays.items():
        tile = np.asarray(values[column])
        if tile.shape != (window.height, window.width):
            raise ValueError(f"{column}: window shape {tile.shape}, expected {(window.height, window.width)}")
        array[day_index, window.cells] = tile[r, c]
    return int(np.isfinite(np.asarray(values[VALIDITY_COLUMN])[r, c]).sum())


def _requested_days(year: int, start: Optional[str], end: Optional[str]) -> pd.DatetimeIndex:
    days = year_days(year)
    first = pd.Timestamp(start, tz="UTC") if start else days[0]
    last = pd.Timestamp(end, tz="UTC") if end else days[-1]
    chosen = days[(days >= first) & (days <= last)]
    if chosen.empty:
        raise ValueError(f"no day of {year} lies in [{start}, {end}]")
    return chosen


def acquire(
    store: Store,
    year: int,
    days: Optional[pd.DatetimeIndex] = None,
    threads: int = DEFAULT_THREADS,
    fetch: Optional[Callable] = None,
    columns=STORE_COLUMNS,
    window_shape: tuple = (TILE_HEIGHT, TILE_WIDTH),
) -> dict:
    """Fetch and store every unmarked day of ``days`` (default: the whole year); returns a summary.

    ``fetch(day, grid, row0, col0, height, width)`` returns column -> (height, width) array, NaN where
    masked; the default is :func:`src.serving.era5_gee_daily.fetch_tile`.
    """
    if fetch is None:
        from src.serving.era5_gee_daily import fetch_tile as fetch  # noqa: PLC0415
    grid = load_grid(store)
    cells = pd.read_parquet(store.cells, columns=["row", "col"])
    rows, cols = cells["row"].to_numpy(), cells["col"].to_numpy()
    all_days = year_days(year)
    position = {day: k for k, day in enumerate(all_days)}
    wanted = all_days if days is None else pd.DatetimeIndex(days)
    outside = [str(day) for day in wanted if day not in position]
    if outside:
        raise ValueError(f"days outside {year}: {outside[:3]}")
    todo = [day for day in wanted if not store.era5_done(year, day).exists()]
    windows = cell_windows(rows, cols, grid, *window_shape)
    arrays = {column: open_year_column(store.era5(year, column), len(all_days), len(cells)) for column in columns}
    per_batch = max(1, (2 * threads) // len(windows))
    summary = {"year": year, "requested": len(wanted), "already_done": len(wanted) - len(todo), "acquired": 0, "empty_days": []}
    logger.info("%d days to fetch, %d windows per day, %d cells, %d threads", len(todo), len(windows), len(cells), threads)
    pool = ThreadPoolExecutor(max_workers=threads)
    try:
        for start in range(0, len(todo), per_batch):
            batch = todo[start : start + per_batch]
            began = time.time()
            valid = dict.fromkeys(batch, 0)
            futures = {
                pool.submit(fetch, day, grid, w.row0, w.col0, w.height, w.width): (day, w) for day in batch for w in windows
            }
            for future in as_completed(futures):
                day, window = futures[future]
                valid[day] += write_window(arrays, position[day], window, rows, cols, future.result())
            for array in arrays.values():
                array.flush()
            seconds = (time.time() - began) / len(batch)
            for day in batch:
                if valid[day] == 0:
                    logger.error("%s: no cell came back valid; left unmarked", day.date())
                    summary["empty_days"].append(str(day.date()))
                    continue
                marker = {"day": str(day.date()), "cells": len(cells), "valid_cells": valid[day], "windows": len(windows), "seconds": round(seconds, 2)}
                write_json(store.era5_done(year, day), marker)
                summary["acquired"] += 1
            logger.info("%s..%s done (%.1f s per day)", batch[0].date(), batch[-1].date(), seconds)
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--start", default=None, help="first day (YYYY-MM-DD) inside --year")
    parser.add_argument("--end", default=None, help="last day (YYYY-MM-DD) inside --year")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--project", default=GEE_PROJECT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import ee  # noqa: PLC0415

    ee.Initialize(project=args.project)
    store = Store(Path(args.store))
    summary = acquire(store, args.year, _requested_days(args.year, args.start, args.end), threads=args.threads)
    print(json.dumps(summary, indent=2))
    return 1 if summary["empty_days"] else 0


if __name__ == "__main__":
    sys.exit(main())
