"""HPC-ready prediction visualization pipeline with CLI and batch processing.

Extends the core rasterization logic from prediction_visualization.py to support:
- CLI via argparse (input dir, output dir, run_id, value column, etc.)
- Batch processing of multiple prediction CSV files
- Per-timestamp GeoTIFF and optional PNG map generation
- Temporal composite maps (mean, median, max, min, std, count)
- Dask-based out-of-core processing for multi-GB files
- Cartopy-based map rendering with coastlines

Designed for Palma II HPC zen4 partition (128+ cores, 480 GB RAM).

Usage:
    python -m src.make_prediction.prediction_visualization_hpc \
        --input-dir outputs/prediction \
        --output-dir outputs/maps \
        --run-id viz_2015_july \
        --value-column sap_velocity_cnn_lstm \
        --stats mean median max \
        --sw-threshold 15 \
        --resolution 0.1
"""
from __future__ import annotations

import argparse
import glob as globmod
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Canonical timestamp column candidates (tried in order)
_TIMESTAMP_CANDIDATES = [
    "timestamp", "timestamp.1", "date", "datetime", "time",
    "Date", "Timestamp", "TIMESTAMP",
]


def _resolve_timestamp_column(
    columns: "Iterable[str]",
    preferred: str = "timestamp",
) -> str:
    """Find the best timestamp column from available column names.

    Tries *preferred* first, then falls back through a list of common
    timestamp-like names, and finally searches for any column containing
    ``'time'`` or ``'date'`` (case-insensitive).

    Parameters
    ----------
    columns : iterable of str
        Available column names.
    preferred : str
        Column name to try first.

    Returns
    -------
    str
        Resolved column name.

    Raises
    ------
    KeyError
        If no suitable column is found.
    """
    cols_list = list(columns)
    cols_set = set(cols_list)
    if preferred in cols_set:
        return preferred
    for candidate in _TIMESTAMP_CANDIDATES:
        if candidate in cols_set:
            return candidate
    for col in cols_list:
        if "time" in col.lower() or "date" in col.lower():
            return col
    raise KeyError(
        f"No timestamp column found. Tried: {preferred!r} and "
        f"{_TIMESTAMP_CANDIDATES}. Available: {sorted(cols_set)}"
    )


def read_prediction_file(
    filepath: str,
    usecols: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
) -> "pd.DataFrame | pd.io.parsers.readers.TextFileReader":
    """Read a prediction file (CSV or Parquet) based on file extension.

    Parameters
    ----------
    filepath : str
        Path to the prediction file (.csv or .parquet).
    usecols : list of str, optional
        Columns to read.
    chunksize : int, optional
        If given and file is CSV, return an iterator of DataFrames.
        Ignored for Parquet files (read all at once).

    Returns
    -------
    pd.DataFrame or TextFileReader
        The loaded data.  For Parquet files, always returns a full
        DataFrame regardless of *chunksize*.
    """
    ext = Path(filepath).suffix.lower()
    if ext == ".parquet":
        df = pd.read_parquet(filepath, columns=usecols, engine="pyarrow")
        if chunksize is not None:
            # Simulate chunked reading for Parquet by yielding slices
            return _parquet_chunk_iter(df, chunksize)
        return df
    else:
        return pd.read_csv(
            filepath, usecols=usecols, chunksize=chunksize,
            low_memory=False,
        )


def _parquet_chunk_iter(df: pd.DataFrame, chunksize: int):
    """Yield DataFrame in chunks to match CSV chunked API.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame to chunk.
    chunksize : int
        Number of rows per chunk.

    Yields
    ------
    pd.DataFrame
        A slice of *df* with at most *chunksize* rows.
    """
    for start in range(0, len(df), chunksize):
        yield df.iloc[start : start + chunksize]


# ---------------------------------------------------------------------------
# Reusable helpers (ported from prediction_visualization.py for portability)
# ---------------------------------------------------------------------------

def get_valid_block_size(
    dimension: int,
    preferred_block_size: int = 256,
    block_multiple: int = 16,
) -> int:
    """Calculate a valid GeoTIFF block size for rasterio.

    Ensures the block size is a multiple of *block_multiple* and does not
    exceed *dimension*.

    Parameters
    ----------
    dimension : int
        The height or width of the raster.
    preferred_block_size : int
        The desired block size when dimension is large enough.
    block_multiple : int
        Required multiple for the block size.

    Returns
    -------
    int
        A valid block size.
    """
    if dimension <= 0:
        return block_multiple
    if dimension >= preferred_block_size:
        return (preferred_block_size // block_multiple) * block_multiple
    calculated = (dimension // block_multiple) * block_multiple
    return max(block_multiple, calculated)


# ---------------------------------------------------------------------------
# Timestamp discovery
# ---------------------------------------------------------------------------

def discover_timestamps(
    csv_path: str,
    timestamp_col: str = "timestamp",
    chunk_size: int = 500_000,
) -> List[str]:
    """Scan a prediction file and return a sorted list of unique timestamps.

    When *timestamp_col* is not found in the file header, the function
    falls back through common timestamp column names automatically.

    Parameters
    ----------
    csv_path : str
        Path to the prediction CSV or Parquet file.
    timestamp_col : str
        Preferred timestamp column name.
    chunk_size : int
        Number of rows per chunk for memory-efficient reading.

    Returns
    -------
    list of str
        Sorted unique timestamp strings found in the file.
    """
    logger.info("Discovering timestamps in %s …", os.path.basename(csv_path))
    ext = Path(csv_path).suffix.lower()

    # Resolve actual column name from file header
    if ext == ".parquet":
        import pyarrow.parquet as pq
        schema_cols = pq.read_schema(csv_path).names
    else:
        with open(csv_path, "r") as fh:
            schema_cols = fh.readline().strip().split(",")
    resolved_col = _resolve_timestamp_column(schema_cols, preferred=timestamp_col)
    if resolved_col != timestamp_col:
        logger.info("Timestamp column resolved: %r -> %r", timestamp_col, resolved_col)

    unique_ts: set = set()
    for chunk in read_prediction_file(csv_path, usecols=[resolved_col],
                                      chunksize=chunk_size):
        unique_ts.update(chunk[resolved_col].dropna().unique())
    sorted_ts = sorted(unique_ts)
    logger.info("Found %d unique timestamps.", len(sorted_ts))
    return sorted_ts


# ---------------------------------------------------------------------------
# Single-pass batch rasterization (avoids re-reading CSV per timestamp)
# ---------------------------------------------------------------------------

def rasterize_all_timestamps(
    csv_path: str,
    output_dir: str,
    value_column: str = "sap_velocity_cnn_lstm",
    timestamp_col: str = "timestamp",
    sw_in_threshold: float = 15.0,
    resolution: float = 0.1,
    lat_range: Optional[Tuple[float, float]] = None,
    lon_range: Optional[Tuple[float, float]] = None,
    chunk_size: int = 500_000,
    **kwargs: Any,
) -> List[Tuple[str, str]]:
    """Read a CSV once and produce one GeoTIFF per unique timestamp.

    This avoids the O(T × FileSize) cost of re-reading the full CSV for
    each timestamp.  Data is accumulated per-timestamp in a single chunked
    pass, then each timestamp is rasterized from its in-memory partition.

    Parameters
    ----------
    csv_path : str
        Path to the prediction file (CSV or Parquet).
    output_dir : str
        Directory for output GeoTIFFs.
    value_column : str
        Name of the predicted-value column.
    timestamp_col : str
        Name of the timestamp column.
    sw_in_threshold : float
        Minimum ``sw_in`` for daytime filtering (≤ 0 to disable).
    resolution : float
        Grid cell size in degrees.
    lat_range, lon_range : tuple of float, optional
        Clip extents.
    chunk_size : int
        Number of rows per pandas chunk.

    Returns
    -------
    list of (timestamp_str, tif_path) tuples
        Successfully created GeoTIFFs with their timestamp labels.
    """
    start = time.time()
    logger.info("Single-pass rasterization of %s …", os.path.basename(csv_path))

    # Auto-resolve timestamp column from file header
    ext = Path(csv_path).suffix.lower()
    if ext == ".parquet":
        import pyarrow.parquet as pq
        schema_cols = pq.read_schema(csv_path).names
    else:
        with open(csv_path, "r") as fh:
            schema_cols = fh.readline().strip().split(",")
    timestamp_col = _resolve_timestamp_column(schema_cols, preferred=timestamp_col)

    cols_needed = ["latitude", "longitude", timestamp_col, value_column]
    if sw_in_threshold > 0:
        cols_needed.append("sw_in")
    cols_needed = list(dict.fromkeys(cols_needed))

    # Accumulate per-timestamp aggregated data: {ts: DataFrame}
    ts_accum: Dict[str, List[pd.DataFrame]] = {}
    _hourly_mode = kwargs.get("hourly_mode", False)
    _hourly_agg = kwargs.get("hourly_agg", "mean")

    for chunk in tqdm(
        read_prediction_file(csv_path, usecols=cols_needed,
                             chunksize=chunk_size),
        desc="Reading prediction file",
    ):
        # Daytime filter
        if sw_in_threshold > 0 and "sw_in" in chunk.columns:
            chunk = chunk[chunk["sw_in"] > sw_in_threshold]

        # Drop rows with missing predictions
        chunk = chunk.dropna(subset=[value_column])

        if len(chunk) == 0:
            continue

        # For hourly data in daily mode: extract date from timestamp
        if _hourly_mode:
            try:
                chunk["_date"] = pd.to_datetime(
                    chunk[timestamp_col]
                ).dt.date.astype(str)
            except (ValueError, TypeError):
                chunk["_date"] = chunk[timestamp_col].astype(str).str[:10]
            group_col = "_date"
        else:
            group_col = timestamp_col

        # Partition by timestamp (or date) and accumulate
        for ts_val, group in chunk.groupby(group_col):
            ts_str = str(ts_val)
            agg = group.groupby(
                ["latitude", "longitude"], as_index=False,
            )[value_column].mean()
            ts_accum.setdefault(ts_str, []).append(agg)

    # Rasterize each timestamp
    os.makedirs(output_dir, exist_ok=True)
    results: List[Tuple[str, str]] = []

    for ts_str in sorted(ts_accum.keys()):
        ts_safe = _sanitize_filename(ts_str)
        tif_name = f"sap_velocity_{ts_safe}.tif"
        tif_path = os.path.join(output_dir, tif_name)

        # Skip if already exists (resume support)
        if os.path.exists(tif_path):
            logger.info("Already exists: %s – skipping.", tif_name)
            results.append((ts_str, tif_path))
            continue

        # Merge accumulated partials and re-aggregate
        combined = pd.concat(ts_accum[ts_str], ignore_index=True)
        agg_method = _hourly_agg if _hourly_mode else "mean"
        df = combined.groupby(
            ["latitude", "longitude"], as_index=False,
        )[value_column].agg(agg_method).rename(
            columns={value_column: value_column}
            if agg_method == "mean"
            else {}
        )
        # Ensure column name is consistent after agg
        if value_column not in df.columns:
            df = df.rename(columns={agg_method: value_column})

        if len(df) == 0:
            logger.warning("No valid data for %s – skipping.", ts_str)
            continue

        try:
            _write_geotiff(
                df, tif_path, value_column, resolution,
                lat_range, lon_range,
            )
            results.append((ts_str, tif_path))
            logger.info(
                "Rasterized %s → %s (%d pixels)",
                ts_str, tif_name, len(df),
            )
        except Exception as exc:
            logger.error("Failed to rasterize %s: %s", ts_str, exc)

    elapsed = time.time() - start
    logger.info(
        "Single-pass complete: %d timestamps, %.1fs.",
        len(results), elapsed,
    )
    return results


def _sanitize_filename(ts_str: str) -> str:
    """Sanitize a timestamp string for use as a filename component.

    Only allows alphanumeric characters, hyphens, underscores, and dots.

    Parameters
    ----------
    ts_str : str
        Raw timestamp string.

    Returns
    -------
    str
        Safe filename component.
    """
    safe = ts_str.replace(" ", "_").replace(":", "-").split("+")[0]
    safe = re.sub(r"[^A-Za-z0-9_\-.]", "", safe)
    return safe or "unknown"


# ---------------------------------------------------------------------------
# Per-timestamp rasterization (single-timestamp, for small files or testing)
# ---------------------------------------------------------------------------

def rasterize_timestamp(
    csv_path: str,
    timestamp_value: str,
    output_tif: str,
    value_column: str = "sap_velocity_cnn_lstm",
    timestamp_col: str = "timestamp.1",
    sw_in_threshold: float = 15.0,
    resolution: float = 0.1,
    lat_range: Optional[Tuple[float, float]] = None,
    lon_range: Optional[Tuple[float, float]] = None,
    dask_blocksize: str = "256MB",
) -> Optional[str]:
    """Read a CSV, filter to one timestamp, and produce a GeoTIFF.

    Parameters
    ----------
    csv_path : str
        Path to the prediction CSV file.
    timestamp_value : str
        The timestamp string to select.
    output_tif : str
        Path for the output GeoTIFF.
    value_column : str
        Name of the predicted-value column.
    timestamp_col : str
        Name of the timestamp column.
    sw_in_threshold : float
        Minimum ``sw_in`` for daytime filtering (set ≤ 0 to disable).
    resolution : float
        Grid cell size in degrees.
    lat_range : tuple of float, optional
        ``(min_lat, max_lat)`` to clip the output extent.
    lon_range : tuple of float, optional
        ``(min_lon, max_lon)`` to clip the output extent.
    dask_blocksize : str
        Dask CSV read blocksize (e.g. ``"256MB"``).

    Returns
    -------
    str or None
        Path to the created GeoTIFF, or ``None`` on failure.
    """
    try:
        import dask.dataframe as dd
    except ImportError:
        logger.warning("Dask not available – falling back to pandas chunked read.")
        return _rasterize_timestamp_pandas(
            csv_path, timestamp_value, output_tif,
            value_column, timestamp_col, sw_in_threshold,
            resolution, lat_range, lon_range,
        )

    start = time.time()
    cols_needed = ["latitude", "longitude", timestamp_col, value_column]
    if sw_in_threshold > 0:
        cols_needed.append("sw_in")
    cols_needed = list(dict.fromkeys(cols_needed))  # deduplicate, preserve order

    dtypes: Dict[str, Any] = {
        "latitude": float,
        "longitude": float,
        value_column: float,
    }
    if "sw_in" in cols_needed:
        dtypes["sw_in"] = float

    # Read with Dask
    ddf = dd.read_csv(
        csv_path,
        blocksize=dask_blocksize,
        usecols=cols_needed,
        dtype=dtypes,
        assume_missing=True,
    )

    # Filter to requested timestamp
    ddf = ddf[ddf[timestamp_col] == timestamp_value]

    # Daytime filter
    if sw_in_threshold > 0 and "sw_in" in ddf.columns:
        ddf = ddf[ddf["sw_in"] > sw_in_threshold]

    # Drop rows with missing predictions
    ddf = ddf.dropna(subset=[value_column])

    # Aggregate: mean per pixel
    grouped = (
        ddf.groupby(["latitude", "longitude"])[value_column]
        .mean()
        .compute(scheduler="threads")
    )
    df = grouped.reset_index()

    if len(df) == 0:
        logger.warning(
            "No valid data for timestamp %s – skipping.", timestamp_value,
        )
        return None

    # Rasterize to GeoTIFF
    _write_geotiff(df, output_tif, value_column, resolution, lat_range, lon_range)

    elapsed = time.time() - start
    logger.info(
        "Rasterized %s → %s (%d pixels, %.1fs)",
        timestamp_value, os.path.basename(output_tif), len(df), elapsed,
    )
    return output_tif


def _rasterize_timestamp_pandas(
    csv_path: str,
    timestamp_value: str,
    output_tif: str,
    value_column: str,
    timestamp_col: str,
    sw_in_threshold: float,
    resolution: float,
    lat_range: Optional[Tuple[float, float]],
    lon_range: Optional[Tuple[float, float]],
    chunk_size: int = 500_000,
) -> Optional[str]:
    """Pandas-chunked fallback for rasterize_timestamp when Dask is absent."""
    start = time.time()
    cols_needed = ["latitude", "longitude", timestamp_col, value_column]
    if sw_in_threshold > 0:
        cols_needed.append("sw_in")
    cols_needed = list(dict.fromkeys(cols_needed))

    accum: List[pd.DataFrame] = []
    for chunk in pd.read_csv(
        csv_path, usecols=cols_needed, chunksize=chunk_size, low_memory=False,
    ):
        chunk = chunk[chunk[timestamp_col] == timestamp_value]
        if sw_in_threshold > 0 and "sw_in" in chunk.columns:
            chunk = chunk[chunk["sw_in"] > sw_in_threshold]
        chunk = chunk.dropna(subset=[value_column])
        if len(chunk) > 0:
            accum.append(
                chunk[["latitude", "longitude", value_column]].copy()
            )

    if not accum:
        logger.warning(
            "No valid data for timestamp %s (pandas) – skipping.",
            timestamp_value,
        )
        return None

    combined = pd.concat(accum, ignore_index=True)
    df = combined.groupby(["latitude", "longitude"], as_index=False)[value_column].mean()

    _write_geotiff(df, output_tif, value_column, resolution, lat_range, lon_range)

    elapsed = time.time() - start
    logger.info(
        "Rasterized %s → %s (%d pixels, %.1fs) [pandas]",
        timestamp_value, os.path.basename(output_tif), len(df), elapsed,
    )
    return output_tif


def _write_geotiff(
    df: pd.DataFrame,
    output_tif: str,
    value_column: str,
    resolution: float,
    lat_range: Optional[Tuple[float, float]] = None,
    lon_range: Optional[Tuple[float, float]] = None,
) -> None:
    """Write a DataFrame of (latitude, longitude, value) to a GeoTIFF.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``latitude``, ``longitude``, and *value_column*.
    output_tif : str
        Output file path.
    value_column : str
        Column containing the values to rasterize.
    resolution : float
        Grid cell size in degrees.
    lat_range, lon_range : tuple of float, optional
        Clip extents.  If ``None``, derived from data.
    """
    lats = pd.to_numeric(df["latitude"], errors="coerce")
    lons = pd.to_numeric(df["longitude"], errors="coerce")
    vals = pd.to_numeric(df[value_column], errors="coerce")
    valid = lats.notna() & lons.notna() & vals.notna()
    lats, lons, vals = lats[valid], lons[valid], vals[valid]

    if len(lats) == 0:
        raise ValueError("No valid data to rasterize.")

    min_lat = lat_range[0] if lat_range else float(lats.min())
    max_lat = lat_range[1] if lat_range else float(lats.max())
    min_lon = lon_range[0] if lon_range else float(lons.min())
    max_lon = lon_range[1] if lon_range else float(lons.max())

    eps = 1e-9
    height = max(1, int(np.ceil((max_lat - min_lat + eps) / resolution)))
    width = max(1, int(np.ceil((max_lon - min_lon + eps) / resolution)))

    row_idx = np.floor((max_lat - lats.values + eps) / resolution).astype(int)
    col_idx = np.floor((lons.values - min_lon + eps) / resolution).astype(int)
    in_bounds = (row_idx >= 0) & (row_idx < height) & (col_idx >= 0) & (col_idx < width)

    # Warn if duplicate pixel coordinates exist (last-value-wins)
    bounded_rows = row_idx[in_bounds]
    bounded_cols = col_idx[in_bounds]
    linear_idx = bounded_rows * width + bounded_cols
    n_unique = len(np.unique(linear_idx))
    if n_unique < len(linear_idx):
        logger.warning(
            "Duplicate pixel coords detected (%d points → %d unique pixels). "
            "Pre-aggregate data to avoid silent overwrites.",
            len(linear_idx), n_unique,
        )

    grid = np.full((height, width), np.nan, dtype=np.float32)
    grid[bounded_rows, bounded_cols] = vals.values[in_bounds]

    transform = from_origin(min_lon, max_lat, resolution, resolution)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": rasterio.float32,
        "crs": "+proj=longlat +datum=WGS84 +no_defs",
        "transform": transform,
        "nodata": np.nan,
        "compress": "LZW",
        "tiled": True,
        "blockxsize": get_valid_block_size(width),
        "blockysize": get_valid_block_size(height),
    }

    os.makedirs(os.path.dirname(output_tif) or ".", exist_ok=True)
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(grid, 1)


# ---------------------------------------------------------------------------
# Map plotting (Cartopy)
# ---------------------------------------------------------------------------

def plot_map(
    tif_path: str,
    output_png: str,
    title: Optional[str] = None,
    cmap: str = "viridis",
    percentile_range: Tuple[float, float] = (2, 98),
    dpi: int = 200,
) -> Optional[str]:
    """Render a GeoTIFF as a PNG map with coastlines using Cartopy.

    Parameters
    ----------
    tif_path : str
        Input GeoTIFF path.
    output_png : str
        Output PNG path.
    title : str, optional
        Plot title.  Derived from filename if not given.
    cmap : str
        Matplotlib colormap name.
    percentile_range : tuple of float
        ``(min_pct, max_pct)`` for colour scaling.
    dpi : int
        PNG resolution.

    Returns
    -------
    str or None
        Path to PNG on success, ``None`` on failure.
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False
        logger.warning("Cartopy not available – falling back to plain matplotlib.")

    try:
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            bounds = src.bounds  # left, bottom, right, top

        masked = np.ma.masked_invalid(data)
        valid_count = masked.count()
        if valid_count == 0:
            logger.warning("No valid pixels in %s – skipping PNG.", tif_path)
            return None

        vmin, vmax = np.nanpercentile(
            data[~np.isnan(data)],
            list(percentile_range),
        )

        if title is None:
            title = Path(tif_path).stem.replace("_", " ").title()

        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

        if has_cartopy:
            fig, ax = plt.subplots(
                figsize=(14, 7),
                subplot_kw={"projection": ccrs.PlateCarree()},
            )
            img = ax.imshow(
                masked, cmap=cmap, origin="upper", extent=extent,
                transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax,
                interpolation="nearest",
            )
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="black")
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, color="gray")
            ax.set_global()
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
        else:
            fig, ax = plt.subplots(figsize=(14, 7))
            img = ax.imshow(
                masked, cmap=cmap, origin="upper", extent=extent,
                vmin=vmin, vmax=vmax, interpolation="nearest",
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        cbar = fig.colorbar(img, ax=ax, orientation="horizontal",
                            fraction=0.046, pad=0.08, shrink=0.8)
        cbar.set_label("Sap Velocity (cm³ cm⁻² h⁻¹)")
        ax.set_title(title, fontsize=12)

        os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
        fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved PNG: %s", output_png)
        return output_png

    except Exception as exc:
        logger.error("Failed to plot %s: %s", tif_path, exc)
        return None


# ---------------------------------------------------------------------------
# Composite generation
# ---------------------------------------------------------------------------

_STAT_FUNCS = {
    "mean": lambda stack: np.nanmean(stack, axis=0),
    "median": lambda stack: np.nanmedian(stack, axis=0),
    "max": lambda stack: np.nanmax(stack, axis=0),
    "min": lambda stack: np.nanmin(stack, axis=0),
    "std": lambda stack: np.nanstd(stack, axis=0),
    "count": lambda stack: np.sum(~np.isnan(stack), axis=0).astype(np.float32),
}


def compute_composites(
    tif_paths: Sequence[str],
    output_dir: str,
    stats: Sequence[str] = ("mean", "median", "max", "min", "std", "count"),
    render_png: bool = True,
    dpi: int = 200,
) -> Dict[str, str]:
    """Compute pixel-wise composites from per-timestamp GeoTIFFs.

    Uses streaming (incremental) statistics to avoid loading every GeoTIFF
    into memory simultaneously.  Only the ``median`` statistic requires a
    full in-memory stack; all other statistics (mean, std, min, max, count)
    are computed in a single streaming pass via Welford's online algorithm.

    Parameters
    ----------
    tif_paths : sequence of str
        Paths to per-timestamp GeoTIFFs (must share the same grid).
    output_dir : str
        Directory for composite outputs.
    stats : sequence of str
        Statistics to compute.  Valid: mean, median, max, min, std, count.
    render_png : bool
        Whether to also render PNG maps for each composite.
    dpi : int
        PNG resolution.

    Returns
    -------
    dict
        Mapping of stat name → output GeoTIFF path.
    """
    if not tif_paths:
        logger.warning("No GeoTIFFs to composite.")
        return {}

    logger.info("Computing composites from %d GeoTIFFs …", len(tif_paths))
    start = time.time()

    # Read reference metadata from first file
    with rasterio.open(tif_paths[0]) as ref:
        profile = ref.profile.copy()
        ref_height = ref.height
        ref_width = ref.width
        ref_transform = ref.transform

    # ----- Streaming pass: running mean, variance, min, max, count -----
    needs_streaming = set(stats) & {"mean", "std", "min", "max", "count"}
    count_arr = np.zeros((ref_height, ref_width), dtype=np.float64)
    mean_arr = np.zeros((ref_height, ref_width), dtype=np.float64)
    m2_arr = np.zeros((ref_height, ref_width), dtype=np.float64)
    min_arr = np.full((ref_height, ref_width), np.inf, dtype=np.float64)
    max_arr = np.full((ref_height, ref_width), -np.inf, dtype=np.float64)

    for path in tqdm(tif_paths, desc="Streaming composites"):
        try:
            with rasterio.open(path) as src:
                # Validate matching grid geometry
                if src.transform != ref_transform:
                    logger.warning(
                        "Transform mismatch in %s — skipping. "
                        "Expected %s, got %s.",
                        path, ref_transform, src.transform,
                    )
                    continue
                band = src.read(1).astype(np.float64)
                h = min(band.shape[0], ref_height)
                w = min(band.shape[1], ref_width)
                band_view = band[:h, :w]
        except Exception as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue

        valid = ~np.isnan(band_view)
        if not np.any(valid):
            continue

        # Welford's online algorithm for mean and M2 (variance accumulator)
        count_arr[:h, :w] = np.where(valid, count_arr[:h, :w] + 1, count_arr[:h, :w])
        delta = np.where(valid, band_view - mean_arr[:h, :w], 0.0)
        mean_arr[:h, :w] = np.where(
            valid,
            mean_arr[:h, :w] + delta / np.maximum(count_arr[:h, :w], 1),
            mean_arr[:h, :w],
        )
        delta2 = np.where(valid, band_view - mean_arr[:h, :w], 0.0)
        m2_arr[:h, :w] = np.where(valid, m2_arr[:h, :w] + delta * delta2, m2_arr[:h, :w])

        # Running min/max
        min_arr[:h, :w] = np.where(valid, np.minimum(min_arr[:h, :w], band_view), min_arr[:h, :w])
        max_arr[:h, :w] = np.where(valid, np.maximum(max_arr[:h, :w], band_view), max_arr[:h, :w])

    # Finalise derived arrays — mask pixels with zero observations
    no_data = count_arr == 0
    var_arr = np.where(count_arr > 1, m2_arr / (count_arr - 1), 0.0)

    streaming_results: Dict[str, np.ndarray] = {
        "mean": np.where(no_data, np.nan, mean_arr).astype(np.float32),
        "std": np.where(no_data, np.nan, np.sqrt(var_arr)).astype(np.float32),
        "min": np.where(no_data, np.nan, min_arr).astype(np.float32),
        "max": np.where(no_data, np.nan, max_arr).astype(np.float32),
        "count": count_arr.astype(np.float32),
    }

    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, str] = {}

    # Write streaming composites
    for stat in stats:
        if stat in streaming_results:
            composite = streaming_results[stat]
        elif stat == "median":
            # Median requires a full stack — load one band at a time into array
            logger.info("Computing median (requires full stack)…")
            stack = np.full(
                (len(tif_paths), ref_height, ref_width), np.nan, dtype=np.float32,
            )
            for i, path in enumerate(tif_paths):
                try:
                    with rasterio.open(path) as src:
                        band = src.read(1)
                        h = min(band.shape[0], ref_height)
                        w = min(band.shape[1], ref_width)
                        stack[i, :h, :w] = band[:h, :w]
                except Exception:
                    pass
            composite = np.nanmedian(stack, axis=0).astype(np.float32)
            del stack
        else:
            logger.warning("Unknown stat '%s' – skipping.", stat)
            continue

        out_tif = os.path.join(output_dir, f"sap_velocity_composite_{stat}.tif")
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(composite, 1)
        results[stat] = out_tif
        logger.info("Wrote %s", out_tif)

        if render_png:
            out_png = out_tif.replace(".tif", ".png")
            plot_map(
                out_tif, out_png,
                title=f"Sap Velocity – {stat.capitalize()} Composite",
                dpi=dpi,
            )

    elapsed = time.time() - start
    logger.info("Composite generation done in %.1fs.", elapsed)
    return results


# ---------------------------------------------------------------------------
# Batch orchestrator
# ---------------------------------------------------------------------------

def run_batch(
    input_dir: str,
    output_dir: str,
    run_id: str,
    value_column: str = "sap_velocity_cnn_lstm",
    value_columns: Optional[List[str]] = None,
    timestamp_col: str = "timestamp",
    time_scale: str = "daily",
    hourly_agg: str = "mean",
    hourly_maps: bool = False,
    file_glob: str = "*predictions*.*",
    sw_in_threshold: float = 15.0,
    resolution: float = 0.1,
    lat_range: Optional[Tuple[float, float]] = None,
    lon_range: Optional[Tuple[float, float]] = None,
    stats: Sequence[str] = ("mean", "median", "max", "min", "std", "count"),
    render_png: bool = True,
    dask_blocksize: str = "256MB",
    dpi: int = 200,
) -> None:
    """Orchestrate the full visualization pipeline.

    Uses single-pass CSV reading (``rasterize_all_timestamps``) to avoid
    re-reading multi-GB files once per timestamp.

    Parameters
    ----------
    input_dir : str
        Directory containing prediction CSV files.
    output_dir : str
        Base output directory.  A subdirectory named *run_id* is created.
    run_id : str
        Unique identifier for this run (namespaces outputs).
    value_column : str
        Column with predicted sap velocity values.
    timestamp_col : str
        Timestamp column name in the CSV.
    file_glob : str
        Glob pattern to select CSV files within *input_dir*.
    sw_in_threshold : float
        Minimum ``sw_in`` for daytime filtering (≤ 0 to disable).
    resolution : float
        Grid resolution in degrees.
    lat_range, lon_range : tuple of float, optional
        Clip extents.
    stats : sequence of str
        Composite statistics to compute.
    render_png : bool
        Whether to render PNG maps alongside GeoTIFFs.
    dask_blocksize : str
        Dask CSV blocksize (unused in single-pass mode; kept for API compat).
    dpi : int
        PNG output resolution.

    Raises
    ------
    FileNotFoundError
        If no CSV files match the glob pattern.
    ValueError
        If *run_id* resolves to a path outside *output_dir*.
    """
    # -- Validate run_id (path traversal protection) -----------------------
    run_dir = os.path.realpath(os.path.join(output_dir, run_id))
    base_dir = os.path.realpath(output_dir)
    if not run_dir.startswith(base_dir + os.sep) and run_dir != base_dir:
        raise ValueError(
            f"Invalid run_id '{run_id}': resolved path '{run_dir}' "
            f"escapes base output directory '{base_dir}'."
        )

    wall_start = time.time()
    os.makedirs(run_dir, exist_ok=True)
    logger.info("=== Prediction Visualization Pipeline ===")
    logger.info("Run ID   : %s", run_id)
    logger.info("Input    : %s/%s", input_dir, file_glob)
    logger.info("Output   : %s", run_dir)
    logger.info("Column   : %s", value_column)
    logger.info("Resolution: %.4f°", resolution)
    logger.info("SW_in thr: %.1f", sw_in_threshold)

    # Discover CSV files
    pattern = os.path.join(input_dir, file_glob)
    pred_files = sorted(globmod.glob(pattern))
    # Filter to only CSV and Parquet files
    pred_files = [f for f in pred_files if f.endswith((".csv", ".parquet"))]
    if not pred_files:
        raise FileNotFoundError(
            f"No prediction files (CSV/Parquet) matched pattern: {pattern}"
        )
    logger.info("Found %d prediction file(s).", len(pred_files))

    # Determine which columns to process
    columns_to_process = value_columns if value_columns else [value_column]
    hourly_mode = time_scale == "hourly" and not hourly_maps

    # Process each file → per-timestamp GeoTIFFs (single-pass per file)
    all_tifs: Dict[str, List[str]] = {col: [] for col in columns_to_process}
    for csv_path in pred_files:
        csv_basename = Path(csv_path).stem
        logger.info("--- Processing %s ---", csv_basename)

        for vcol in columns_to_process:
            # Create model-specific subdirectory if multi-model
            if len(columns_to_process) > 1:
                model_name = vcol.replace("sap_velocity_", "")
                model_dir = os.path.join(run_dir, model_name)
            else:
                model_dir = run_dir

            ts_results = rasterize_all_timestamps(
                csv_path=csv_path,
                output_dir=model_dir,
                value_column=vcol,
                timestamp_col=timestamp_col,
                sw_in_threshold=sw_in_threshold,
                resolution=resolution,
                lat_range=lat_range,
                lon_range=lon_range,
                hourly_mode=hourly_mode,
                hourly_agg=hourly_agg,
            )
            for ts_str, tif_path in ts_results:
                all_tifs[vcol].append(tif_path)
                if render_png:
                    png_path = tif_path.replace(".tif", ".png")
                    plot_map(
                        tif_path, png_path,
                        title=f"Sap Velocity ({vcol}) – {ts_str}",
                        dpi=dpi,
                    )

    # Compute composites per model
    for vcol in columns_to_process:
        tifs = all_tifs[vcol]
        if len(columns_to_process) > 1:
            model_name = vcol.replace("sap_velocity_", "")
            comp_dir = os.path.join(run_dir, model_name, "composites")
        else:
            comp_dir = os.path.join(run_dir, "composites")

        total_tifs = len(tifs)
        logger.info("Generated %d per-timestamp GeoTIFFs for %s.", total_tifs, vcol)

        if total_tifs >= 2 and stats:
            compute_composites(
                tif_paths=tifs,
                output_dir=comp_dir,
                stats=stats,
                render_png=render_png,
                dpi=dpi,
            )
        elif total_tifs == 1:
            logger.info("Only 1 timestamp for %s – skipping composites.", vcol)
        else:
            logger.warning("No GeoTIFFs for %s – skipping composites.", vcol)

    elapsed = time.time() - wall_start
    logger.info("=== Pipeline complete in %.1fs ===", elapsed)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="HPC-ready prediction visualization pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing prediction CSV files.",
    )
    parser.add_argument(
        "--output-dir", default="outputs/maps",
        help="Base output directory (a run_id subdirectory is created).",
    )
    parser.add_argument(
        "--run-id", required=True,
        help="Unique run identifier for output namespacing.",
    )
    parser.add_argument(
        "--value-column", default="sap_velocity_cnn_lstm",
        help="Name of the predicted-value column in the CSV.",
    )
    parser.add_argument(
        "--timestamp-col", default="timestamp",
        help="Preferred timestamp column name.  When the specified column "
             "is not found, the script auto-detects common alternatives "
             "(timestamp.1, date, datetime, etc.).",
    )
    parser.add_argument(
        "--csv-glob", default="*predictions*.csv",
        help="Glob pattern to select CSV files within --input-dir.",
    )
    parser.add_argument(
        "--sw-threshold", type=float, default=15.0,
        help="Minimum sw_in for daytime filtering (≤ 0 to disable).",
    )
    parser.add_argument(
        "--resolution", type=float, default=0.1,
        help="Grid resolution in degrees.",
    )
    parser.add_argument(
        "--lat-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="Latitude range (min max) to clip output extent.",
    )
    parser.add_argument(
        "--lon-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="Longitude range (min max) to clip output extent.",
    )
    parser.add_argument(
        "--stats", nargs="+",
        default=["mean", "median", "max", "min", "std", "count"],
        choices=["mean", "median", "max", "min", "std", "count"],
        help="Composite statistics to compute.",
    )
    parser.add_argument(
        "--no-png", action="store_true",
        help="Disable PNG map rendering (GeoTIFFs only).",
    )
    parser.add_argument(
        "--dask-blocksize", default="256MB",
        help="Dask CSV read blocksize.",
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="DPI for PNG output.",
    )
    parser.add_argument(
        "--input-format", type=str, default="auto",
        choices=["auto", "csv", "parquet"],
        help="Input file format. 'auto' detects by extension.",
    )
    parser.add_argument(
        "--time-scale", type=str, default="daily",
        choices=["daily", "hourly"],
        help="Temporal scale of prediction data.",
    )
    parser.add_argument(
        "--hourly-agg", type=str, default="mean",
        choices=["mean", "median", "max", "min", "sum"],
        help="Aggregation method for hourly→daily (only when --time-scale hourly).",
    )
    parser.add_argument(
        "--hourly-maps", action="store_true",
        help="Generate per-hour maps instead of daily aggregates (hourly mode only).",
    )
    parser.add_argument(
        "--value-columns", nargs="+", default=None,
        help="Multiple value columns for multi-model maps. Overrides --value-column.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for CLI execution.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments.  Uses ``sys.argv`` if ``None``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    lat_range = tuple(args.lat_range) if args.lat_range else None
    lon_range = tuple(args.lon_range) if args.lon_range else None

    run_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
        value_column=args.value_column,
        value_columns=args.value_columns,
        timestamp_col=args.timestamp_col,
        time_scale=args.time_scale,
        hourly_agg=args.hourly_agg,
        hourly_maps=args.hourly_maps,
        file_glob=args.csv_glob,
        sw_in_threshold=args.sw_threshold,
        resolution=args.resolution,
        lat_range=lat_range,
        lon_range=lon_range,
        stats=args.stats,
        render_png=not args.no_png,
        dask_blocksize=args.dask_blocksize,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
