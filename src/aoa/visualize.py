"""GeoTIFF/NetCDF map generation from AOA DI parquets."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LEVEL_CONFIG = {
    "per_timestamp": ("per_timestamp", "di_*_*.parquet", "per_timestamp", "DI"),
    "monthly": ("monthly", "di_monthly_*.parquet", "monthly", "median_DI"),
    "climatological": (
        "climatological",
        "di_clim_*.parquet",
        "climatological",
        "median_DI",
    ),
    "yearly": ("yearly", "di_yearly_*.parquet", "yearly", "median_DI"),
    "overall": (None, None, None, "median_DI"),
}


def parquet_to_geotiff(
    df: pd.DataFrame,
    output_path: Path,
    di_col: str = "median_DI",
    threshold: float | None = None,
) -> None:
    """Convert DI parquet to 2-band GeoTIFF (DI, AOA mask).

    Builds a full regular grid at the detected resolution so that pixel
    widths match the actual data spacing.  Ocean/missing cells are NaN.
    The previous implementation used a compact grid of unique coordinates
    which caused pixel stretching (0.107 vs 0.1 deg) when ocean columns
    were absent, leading to basemap-raster misalignment.
    """
    import rasterio
    from rasterio.transform import from_bounds

    lats_sorted = np.sort(df["latitude"].unique())[::-1]  # N->S
    lons_sorted = np.sort(df["longitude"].unique())

    # Detect actual grid resolution from median spacing
    res_lon = float(np.median(np.diff(np.sort(df["longitude"].unique()))))
    res_lat = float(np.median(np.diff(np.sort(df["latitude"].unique()))))

    # Build full regular grid anchored to the first data point
    full_lons = np.arange(lons_sorted[0], lons_sorted[-1] + res_lon / 2, res_lon)
    full_lats = np.arange(lats_sorted[0], lats_sorted[-1] - res_lat / 2, -res_lat)
    nrows, ncols = len(full_lats), len(full_lons)

    # Map data coordinates to grid positions via rounding
    col_indices = np.round(
        (df["longitude"].values - full_lons[0]) / res_lon
    ).astype(int)
    row_indices = np.round(
        (full_lats[0] - df["latitude"].values) / res_lat
    ).astype(int)
    valid = (
        (row_indices >= 0)
        & (row_indices < nrows)
        & (col_indices >= 0)
        & (col_indices < ncols)
    )

    di_grid = np.full((nrows, ncols), np.nan, dtype=np.float32)
    mask_grid = np.full((nrows, ncols), np.nan, dtype=np.float32)

    di_grid[row_indices[valid], col_indices[valid]] = (
        df[di_col].values[valid].astype(np.float32)
    )

    if "aoa_mask" in df.columns:
        mask_grid[row_indices[valid], col_indices[valid]] = (
            df["aoa_mask"].values[valid].astype(np.float32)
        )
    elif "frac_inside_aoa" in df.columns and threshold is not None:
        mask_grid[row_indices[valid], col_indices[valid]] = (
            (df["frac_inside_aoa"].values[valid] >= 0.5).astype(np.float32)
        )

    transform = from_bounds(
        float(full_lons[0]) - res_lon / 2,
        float(full_lats[-1]) - res_lat / 2,
        float(full_lons[-1]) + res_lon / 2,
        float(full_lats[0]) + res_lat / 2,
        ncols,
        nrows,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=2,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(di_grid, 1)
        dst.write(mask_grid, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AOA DI maps")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-scale", default="daily")
    parser.add_argument("--map-format", default="geotiff")
    parser.add_argument(
        "--levels",
        default="monthly,climatological,yearly,overall",
        help="Comma-separated levels to generate",
    )
    args = parser.parse_args()

    base = args.input_dir / args.time_scale
    maps_base = args.output_dir / args.time_scale / "maps"
    levels = args.levels.split(",")

    threshold = None
    meta_path = base / "aoa_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            threshold = json.load(f).get("threshold")

    for level in levels:
        if level == "overall":
            src = base / "di_overall.parquet"
            if src.exists():
                df = pd.read_parquet(src)
                maps_base.mkdir(parents=True, exist_ok=True)
                parquet_to_geotiff(
                    df,
                    maps_base / "di_overall.tif",
                    di_col="median_DI",
                    threshold=threshold,
                )
        else:
            cfg = LEVEL_CONFIG[level]
            src_dir = base / cfg[0]
            out_dir = maps_base / cfg[2]
            out_dir.mkdir(parents=True, exist_ok=True)
            for pq in sorted(src_dir.glob(cfg[1])):
                df = pd.read_parquet(pq)
                tif_name = pq.stem + ".tif"
                parquet_to_geotiff(df, out_dir / tif_name, di_col=cfg[3], threshold=threshold)


if __name__ == "__main__":
    main()
