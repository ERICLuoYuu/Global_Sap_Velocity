"""The serving grid: ERA5-Land's native 0.1 degree grid, cropped to the forest/PFT mask.

Measured 2026-09-15 (``probes/serving/probe_grid_inventory.py``, job 46729288): ``ECMWF/ERA5_LAND/HOURLY``
has the transform ``[0.1, 0, -180.05, 0, -0.1, 90.05]``, pixel centres on multiples of 0.1 degree, and
``forest_pft_0p1deg.tif`` is that grid cropped to 78.05 N .. 59.95 S (1380 x 3600). Serving on it reads
ERA5-Land without resampling.

The soil-moisture climatology cube lies on another grid (step 0.100000457 degree, centres near x.x5), so a
cell takes its climatology the way a training site did: the nearest valid cube cell to its coordinates,
:func:`src.feature_contract.swvl_climatology.lookup_cell`, applied to the cell centre.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.feature_contract.definitions import SWC_LAYERS
from src.feature_contract.swvl_climatology import MIN_VALID_SIGMA, lookup_cell

IGBP_TO_PFT = {1: "ENF", 2: "EBF", 3: "DNF", 4: "DBF", 5: "MF", 8: "WSA", 9: "SAV", 11: "WET"}
#: The ``classes`` tag the mask raster must carry, so a re-coded raster is refused instead of misread.
MASK_CLASSES_TAG = "1=ENF 2=EBF 3=DNF 4=DBF 5=MF 8=WSA 9=SAV 11=WET"
COORD_DECIMALS = 6
ALIGN_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Grid:
    west: float
    north: float
    step: float
    n_rows: int
    n_cols: int
    crs: str = "EPSG:4326"

    def latitudes(self) -> np.ndarray:
        return np.round(self.north - (np.arange(self.n_rows) + 0.5) * self.step, COORD_DECIMALS)

    def longitudes(self) -> np.ndarray:
        return np.round(self.west + (np.arange(self.n_cols) + 0.5) * self.step, COORD_DECIMALS)

    def affine(self, row0: int = 0, col0: int = 0) -> dict:
        """The Earth Engine ``affineTransform`` of a window whose top-left pixel is (row0, col0)."""
        return {
            "scaleX": self.step,
            "shearX": 0.0,
            "translateX": self.west + col0 * self.step,
            "shearY": 0.0,
            "scaleY": -self.step,
            "translateY": self.north - row0 * self.step,
        }

    def tiles(self, height: int, width: int) -> list:
        """``(row0, col0, height, width)`` windows covering the grid, row by row."""
        return [
            (row0, col0, min(height, self.n_rows - row0), min(width, self.n_cols - col0))
            for row0 in range(0, self.n_rows, height)
            for col0 in range(0, self.n_cols, width)
        ]


ERA5_LAND_NATIVE = Grid(west=-180.05, north=90.05, step=0.1, n_rows=1801, n_cols=3600)


def _aligned(offset: float, step: float) -> bool:
    ratio = offset / step
    return abs(ratio - round(ratio)) < ALIGN_TOLERANCE * max(1.0, abs(ratio))


def grid_from_raster(path) -> Grid:
    """The grid of a north-up GeoTIFF, refused unless it is a crop of the ERA5-Land native grid."""
    import rasterio  # noqa: PLC0415

    with rasterio.open(path) as src:
        transform = src.transform
        grid = Grid(
            west=transform.c,
            north=transform.f,
            step=transform.a,
            n_rows=src.height,
            n_cols=src.width,
            crs=str(src.crs),
        )
        rotated = transform.b != 0 or transform.d != 0
        square = abs(transform.a + transform.e) < ALIGN_TOLERANCE
    native = ERA5_LAND_NATIVE
    if rotated or not square or grid.crs != native.crs or abs(grid.step - native.step) > ALIGN_TOLERANCE:
        raise ValueError(f"{path}: not a north-up {native.step} degree {native.crs} grid ({grid})")
    if not (_aligned(grid.west - native.west, native.step) and _aligned(native.north - grid.north, native.step)):
        raise ValueError(f"{path}: pixel edges are not on the ERA5-Land native grid ({grid})")
    return grid


def forest_cells(path) -> pd.DataFrame:
    """One row per forest/woody cell of the mask: row, col, centre latitude/longitude, IGBP code and PFT."""
    import rasterio  # noqa: PLC0415

    grid = grid_from_raster(path)
    with rasterio.open(path) as src:
        classes = src.tags().get("classes")
        codes = src.read(1)
    if classes != MASK_CLASSES_TAG:
        raise ValueError(f"{path}: classes tag {classes!r}, expected {MASK_CLASSES_TAG!r}")
    rows, cols = np.nonzero(np.isin(codes, list(IGBP_TO_PFT)))
    frame = pd.DataFrame(
        {
            "row": rows.astype(np.int32),
            "col": cols.astype(np.int32),
            "latitude": grid.latitudes()[rows],
            "longitude": grid.longitudes()[cols],
            "igbp": codes[rows, cols].astype(np.int16),
        }
    )
    frame["pft"] = frame["igbp"].map(IGBP_TO_PFT)
    return frame


def nearest_index(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """``argmin |axis - v|`` for every value, as ``lookup_cell`` takes it (ties to the first index)."""
    unique, inverse = np.unique(values, return_inverse=True)
    nearest = np.array([int(np.abs(axis - value).argmin()) for value in unique], dtype=np.int64)
    return nearest[inverse]


def climatology_for_cells(cells: pd.DataFrame, clim) -> pd.DataFrame:
    """Per cell and layer, the climatology mean/sigma a site at the cell centre would have read.

    Columns ``swvl_mean_L``, ``swvl_sigma_L`` and ``swvl_fallback_L``; NaN where no valid cube cell lies
    within ``lookup_cell``'s search radius. Index as ``cells``.
    """
    lat = cells["latitude"].to_numpy(dtype=float)
    lon = cells["longitude"].to_numpy(dtype=float)
    i0 = nearest_index(clim.lats, lat)
    j0 = nearest_index(clim.lons, lon)
    out = {}
    for layer in SWC_LAYERS:
        sigma = clim.sigma[layer][i0, j0].astype(float)
        mean = clim.mean[layer][i0, j0].astype(float)
        fallback = np.zeros(len(cells), dtype=bool)
        for k in np.flatnonzero(~(np.isfinite(sigma) & (sigma >= MIN_VALID_SIGMA))):
            try:
                cell = lookup_cell(clim, lat[k], lon[k], layer)
                sigma[k], mean[k], fallback[k] = cell.sigma, cell.mean, cell.fallback
            except LookupError:
                sigma[k], mean[k] = np.nan, np.nan
        out[f"swvl_mean_{layer}"] = mean
        out[f"swvl_sigma_{layer}"] = sigma
        out[f"swvl_fallback_{layer}"] = fallback
    return pd.DataFrame(out, index=cells.index)
