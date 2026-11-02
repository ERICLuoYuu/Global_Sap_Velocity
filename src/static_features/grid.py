"""The one grid every site belongs to: WorldClim's 30 arc-seconds, origin (-180, 90)."""

from __future__ import annotations

import math

CELLS_PER_DEGREE = 120
N_ROWS = 180 * CELLS_PER_DEGREE
N_COLS = 360 * CELLS_PER_DEGREE


def cell_of(lat: float, lon: float) -> tuple[int, int]:
    """(row, col) of the cell holding a coordinate. Latitude -90 and longitude 180 clamp into the last cell."""
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):  # NaN fails both comparisons
        raise ValueError(f"coordinate outside the Earth: lat={lat}, lon={lon}")
    row = min(math.floor((90.0 - lat) * CELLS_PER_DEGREE), N_ROWS - 1)
    col = min(math.floor((lon + 180.0) * CELLS_PER_DEGREE), N_COLS - 1)
    return row, col


def cell_bounds(row: int, col: int) -> tuple[float, float, float, float]:
    """(west, south, east, north) of a cell, in degrees."""
    if not (0 <= row < N_ROWS and 0 <= col < N_COLS):
        raise ValueError(f"cell outside the grid: row={row}, col={col}")
    west = col / CELLS_PER_DEGREE - 180.0
    east = (col + 1) / CELLS_PER_DEGREE - 180.0
    north = 90.0 - row / CELLS_PER_DEGREE
    south = 90.0 - (row + 1) / CELLS_PER_DEGREE
    return west, south, east, north
