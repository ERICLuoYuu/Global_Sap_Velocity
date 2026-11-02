"""The on-disk layout of a serving store: what a global run reads, and where it writes its predictions.

::

    <root>/grid.json                           the served grid (a crop of ERA5-Land's native grid)
    <root>/cells.parquet                       one row per served cell: grid row/col, PFT, climatology, statics
    <root>/era5_daily/<year>/<column>.npy      float32 (days of the year, cells), NaN until fetched
    <root>/era5_daily/<year>/done/<day>.json   written once every window of that day is in place
    <root>/solar/<year>.npz                    ext_rad and day_length per (day, latitude, band[, elevation])
    <root>/lai/<year>.npy                      float32 (days of the year, cells)
    <root>/predictions/<run>/plan.json         features, base columns and spin-up the run needs
    <root>/predictions/<run>/<year>/block_<k>.npz
    <root>/predictions/<run>/sap_velocity_<run>_<year>.nc
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.serving.grid import Grid


def column_file_stem(column: str) -> str:
    """A base column as a file name: ``prcip/PET`` -> ``prcip_over_PET``."""
    return column.replace("/", "_over_")


def year_days(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D", tz="UTC")


@dataclass(frozen=True)
class Store:
    root: Path

    def _path(self, *parts) -> Path:
        return Path(self.root).joinpath(*(str(part) for part in parts))

    @property
    def grid(self) -> Path:
        return self._path("grid.json")

    @property
    def cells(self) -> Path:
        return self._path("cells.parquet")

    def era5(self, year: int, column: str) -> Path:
        return self._path("era5_daily", year, f"{column_file_stem(column)}.npy")

    def era5_done(self, year: int, day: pd.Timestamp) -> Path:
        return self._path("era5_daily", year, "done", f"{day.strftime('%Y-%m-%d')}.json")

    def solar(self, year: int) -> Path:
        return self._path("solar", f"{year}.npz")

    def lai(self, year: int) -> Path:
        return self._path("lai", f"{year}.npy")

    def plan(self, run_id: str) -> Path:
        return self._path("predictions", run_id, "plan.json")

    def block(self, run_id: str, year: int, block: int) -> Path:
        return self._path("predictions", run_id, year, f"block_{block:03d}.npz")

    def assembled(self, run_id: str, year: int) -> Path:
        return self._path("predictions", run_id, f"sap_velocity_{run_id}_{year}.nc")


def write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` atomically (a reader never sees half a file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(temporary, path)


def save_grid(store: Store, grid: Grid) -> None:
    write_json(store.grid, asdict(grid))


def load_grid(store: Store) -> Grid:
    return Grid(**json.loads(Path(store.grid).read_text()))


def open_year_column(path: Path, n_days: int, n_cells: int) -> np.memmap:
    """The (days, cells) float32 array at ``path``, created NaN-filled if absent, refused if the shape differs."""
    path = Path(path)
    if path.exists():
        array = np.lib.format.open_memmap(path, mode="r+")
        _check_shape(path, array, n_days, n_cells)
        return array
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n_days, n_cells))
    array[:] = np.nan
    return array


def read_year_column(path: Path, n_days: int, n_cells: int) -> np.memmap:
    """The (days, cells) float32 array at ``path``, read-only; ``FileNotFoundError`` if it was never built."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} has not been built")
    array = np.load(path, mmap_mode="r")
    _check_shape(path, array, n_days, n_cells)
    return array


def _check_shape(path: Path, array: np.ndarray, n_days: int, n_cells: int) -> None:
    if array.shape != (n_days, n_cells) or array.dtype != np.float32:
        raise ValueError(f"{path} holds {array.dtype} {array.shape}, expected float32 {(n_days, n_cells)}")
