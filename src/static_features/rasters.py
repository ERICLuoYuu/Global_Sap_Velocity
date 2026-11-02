"""Local raster primitives: the native pixel holding a site, and the pixels centred in a cell."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from src.static_features.results import NEIGHBOUR, NO_DATA, OK, ExtractionFailure

# Pixel coordinates carry rounding of about 1e-10 at 200,000 pixels, and a centre can sit exactly on a cell
# edge (GAMI's 1125 pixels per degree does at every fourth edge), so near-integers count as integers.
EDGE_TOLERANCE_PX = 1e-9
# A float32 axis holds 89.995 to within 3e-6, so a hundredth of a 0.01 degree pixel tells rounding from a shift.
AXIS_TOLERANCE_PX = 0.01

ReadBlock = Callable[[int, int, int, int], np.ndarray]


@dataclass(frozen=True)
class PixelHit:
    value: float
    status: str
    row: int | None
    col: int | None


def _read_columns(
    read_block: ReadBlock, row_start: int, row_stop: int, col_start: int, col_stop: int, n_cols: int
) -> np.ndarray:
    """Read a window whose columns may run off either edge of the grid, joining the two pieces."""
    if col_start < 0:
        west = read_block(row_start, row_stop, col_start + n_cols, n_cols)
        east = read_block(row_start, row_stop, 0, col_stop)
        return np.concatenate((west, east), axis=1)
    if col_stop > n_cols:
        west = read_block(row_start, row_stop, col_start, n_cols)
        east = read_block(row_start, row_stop, 0, col_stop - n_cols)
        return np.concatenate((west, east), axis=1)
    return read_block(row_start, row_stop, col_start, col_stop)


def pixel_with_reach(
    read_block: ReadBlock,
    shape: tuple[int, int],
    row: int,
    col: int,
    reach: int,
    is_valid: Callable[[np.ndarray], np.ndarray],
    wrap_cols: bool = True,
) -> PixelHit:
    """The own pixel if valid, else the nearest valid pixel in the square window, else no data.

    The search order is extract_raster_value's: nearest by distance, and on a tie the first in
    row-then-column order. Rows clip at the poles; columns wrap the dateline when wrap_cols is set.
    """
    n_rows, n_cols = shape
    if not (0 <= row < n_rows and 0 <= col < n_cols):
        raise ValueError(f"pixel ({row}, {col}) is outside a grid of shape {shape}")
    row_start = max(row - reach, 0)
    row_stop = min(row + reach + 1, n_rows)
    if wrap_cols:
        col_start, col_stop = col - reach, col + reach + 1
    else:
        col_start, col_stop = max(col - reach, 0), min(col + reach + 1, n_cols)

    block = _read_columns(read_block, row_start, row_stop, col_start, col_stop, n_cols)
    valid = is_valid(block)
    best = None
    for block_row in range(block.shape[0]):
        for block_col in range(block.shape[1]):
            if not valid[block_row, block_col]:
                continue
            distance = (block_row + row_start - row) ** 2 + (block_col + col_start - col) ** 2
            if best is None or distance < best[0]:
                best = (distance, block_row, block_col)

    if best is None:
        return PixelHit(math.nan, NO_DATA, None, None)
    distance, block_row, block_col = best
    return PixelHit(
        float(block[block_row, block_col]),
        OK if distance == 0 else NEIGHBOUR,
        block_row + row_start,
        (block_col + col_start) % n_cols,
    )


def centred_span(low: float, high: float, origin: float, step: float, size: int) -> tuple[int, int]:
    """Pixel index range whose centres lie in low <= c < high. origin is the outer edge of pixel 0; step may be negative."""
    from_low = (low - origin) / step - 0.5
    from_high = (high - origin) / step - 0.5
    if step > 0:
        start = math.ceil(from_low - EDGE_TOLERANCE_PX)
        stop = math.ceil(from_high - EDGE_TOLERANCE_PX)
    else:
        start = math.floor(from_high + EDGE_TOLERANCE_PX) + 1
        stop = math.floor(from_low + EDGE_TOLERANCE_PX) + 1
    start = min(max(start, 0), size)
    stop = min(max(stop, start), size)
    return start, stop


def tiff_reader(src, band: int) -> ReadBlock:
    """A read_block over one band of an open rasterio dataset."""

    def read_block(row_start: int, row_stop: int, col_start: int, col_stop: int) -> np.ndarray:
        return src.read(band, window=((row_start, row_stop), (col_start, col_stop)))

    return read_block


def pixel_index(coordinate: float, origin: float, step: float, size: int) -> int:
    """Index of the pixel holding a coordinate. origin is the outer edge of pixel 0; the far edge clamps into the last pixel."""
    position = (coordinate - origin) / step  # NaN fails the comparison below
    if not 0 <= position <= size:
        raise ValueError(f"coordinate {coordinate} is outside a grid of {size} pixels from {origin} by {step}")
    return min(math.floor(position), size - 1)


def require_global_axes(product: str, latitude: np.ndarray, longitude: np.ndarray) -> None:
    """Stop unless the coordinates are the pixel centres of a global grid, north first and west first."""
    for name, axis, start, end in (("latitude", latitude, 90.0, -90.0), ("longitude", longitude, -180.0, 180.0)):
        step = (end - start) / axis.size
        first, last = start + step / 2, end - step / 2
        if max(abs(axis[0] - first), abs(axis[-1] - last)) > AXIS_TOLERANCE_PX * abs(step):
            raise ExtractionFailure(
                f"{product}: {name} runs from {axis[0]} to {axis[-1]}, expected {first} to {last}"
            )
