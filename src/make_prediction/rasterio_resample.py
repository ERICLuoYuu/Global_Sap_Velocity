"""Rasterio-based spatial resampling utilities.

Provides a single function to resample any GeoTIFF or in-memory raster
onto a target lat/lon grid using rasterio.warp.reproject.

Variable-type-aware:
  - Categorical (e.g., PFT land cover): Resampling.mode
  - Continuous  (e.g., elevation, LAI):  Resampling.nearest
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


# Fix PROJ database version conflict on HPC (must run before rasterio import).
# pyproj ships an older proj.db that confuses rasterio's PROJ. Point to
# rasterio's own proj_data directory which has a compatible version.
def _fix_proj_lib():
    if "PROJ_LIB" in os.environ:
        return
    # Find rasterio's proj_data without importing rasterio (avoiding PROJ init)
    import importlib.util

    spec = importlib.util.find_spec("rasterio")
    if spec and spec.origin:
        rio_dir = Path(spec.origin).parent / "proj_data"
        if rio_dir.is_dir():
            os.environ["PROJ_LIB"] = str(rio_dir)


_fix_proj_lib()

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject

logger = logging.getLogger(__name__)

# Lazy CRS initialization to avoid PROJ database version conflicts on HPC
_CRS_4326 = None


def _get_crs_4326():
    global _CRS_4326
    if _CRS_4326 is None:
        try:
            _CRS_4326 = CRS.from_epsg(4326)
        except Exception:
            # Fallback: define WGS84 via proj4 string (no PROJ DB needed)
            _CRS_4326 = CRS.from_proj4("+proj=longlat +datum=WGS84 +no_defs")
    return _CRS_4326


def resample_raster_to_grid(
    source: str | Path | dict,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    categorical: bool = False,
    nodata_value: float = np.nan,
) -> np.ndarray:
    """Resample a raster onto an arbitrary lat/lon grid.

    Parameters
    ----------
    source : str, Path, or dict
        - If str/Path: path to a GeoTIFF file.
        - If dict: must contain keys ``data`` (2-D ndarray), ``transform``
          (rasterio Affine), and optionally ``nodata``.
    target_lats : 1-D ndarray
        Target latitude values (sorted descending or ascending).
    target_lons : 1-D ndarray
        Target longitude values (sorted ascending).
    categorical : bool
        If True, uses Resampling.mode (majority vote).
        If False, uses Resampling.nearest.
    nodata_value : float
        Fill value for pixels without data.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(len(target_lats), len(target_lons))``.
    """
    resampling = Resampling.mode if categorical else Resampling.nearest

    # Read source data
    if isinstance(source, dict):
        src_data = np.asarray(source["data"], dtype=np.float32)
        src_transform = source["transform"]
        src_nodata = source.get("nodata")
        src_crs = source.get("crs", _get_crs_4326())
    else:
        source = Path(source)
        with rasterio.open(source) as ds:
            src_data = ds.read(1).astype(np.float32)
            src_transform = ds.transform
            src_nodata = ds.nodata
            src_crs = ds.crs or _get_crs_4326()

    # Mark source nodata as NaN for consistent handling
    if src_nodata is not None:
        src_data[src_data == src_nodata] = np.nan

    # Build destination grid
    dst_height = len(target_lats)
    dst_width = len(target_lons)

    # Determine grid ordering (lats may be ascending or descending)
    lat_min, lat_max = float(np.min(target_lats)), float(np.max(target_lats))
    lon_min, lon_max = float(np.min(target_lons)), float(np.max(target_lons))

    # Pixel size from the target grid
    if dst_height > 1:
        lat_res = abs(target_lats[1] - target_lats[0])
    else:
        lat_res = 0.1  # fallback
    if dst_width > 1:
        lon_res = abs(target_lons[1] - target_lons[0])
    else:
        lon_res = 0.1  # fallback

    # Expand bounds by half a pixel so pixel centres align with target coords
    dst_west = lon_min - lon_res / 2
    dst_east = lon_max + lon_res / 2
    dst_south = lat_min - lat_res / 2
    dst_north = lat_max + lat_res / 2

    dst_transform = from_bounds(dst_west, dst_south, dst_east, dst_north, dst_width, dst_height)

    # Allocate destination array
    dst_data = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

    # Reproject
    try:
        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=_get_crs_4326(),
            resampling=resampling,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )
    except Exception as exc:
        logger.error("rasterio reproject failed: %s", exc, exc_info=True)
        return np.full((dst_height, dst_width), nodata_value, dtype=np.float32)

    # If target_lats is ascending but rasterio wrote top-to-bottom, flip
    if dst_height > 1 and target_lats[0] < target_lats[-1]:
        # rasterio always writes north-to-south; if target is south-to-north, flip
        dst_data = np.flipud(dst_data)

    # Replace NaN with nodata_value if different
    if not np.isnan(nodata_value):
        dst_data[np.isnan(dst_data)] = nodata_value

    return dst_data
