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
    resolution: float = 0.1,
) -> None:
    """Convert DI parquet to 2-band GeoTIFF (DI, AOA mask)."""
    import rasterio
    from rasterio.transform import from_bounds

    lats = np.sort(df["latitude"].unique())[::-1]  # N->S
    lons = np.sort(df["longitude"].unique())
    nrows, ncols = len(lats), len(lons)

    lat_to_row = {lat: i for i, lat in enumerate(lats)}
    lon_to_col = {lon: j for j, lon in enumerate(lons)}

    di_grid = np.full((nrows, ncols), np.nan, dtype=np.float32)
    mask_grid = np.full((nrows, ncols), np.nan, dtype=np.float32)

    rows = df["latitude"].map(lat_to_row).values
    cols = df["longitude"].map(lon_to_col).values
    di_grid[rows, cols] = df[di_col].values.astype(np.float32)

    if "aoa_mask" in df.columns:
        mask_grid[rows, cols] = df["aoa_mask"].astype(np.float32)
    elif "frac_inside_aoa" in df.columns and threshold is not None:
        mask_grid[rows, cols] = (df["frac_inside_aoa"] >= 0.5).astype(np.float32)

    transform = from_bounds(
        float(lons.min()) - resolution / 2,
        float(lats.min()) - resolution / 2,
        float(lons.max()) + resolution / 2,
        float(lats.max()) + resolution / 2,
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
