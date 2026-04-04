"""Patch src/aoa/visualize.py to fix parquet_to_geotiff grid alignment."""

from pathlib import Path

viz_path = Path("src/aoa/visualize.py")
code = viz_path.read_text()

# Find and replace the function
old_sig = "def parquet_to_geotiff("
idx_start = code.find(old_sig)
if idx_start < 0:
    raise RuntimeError("parquet_to_geotiff not found")

# Find the end of the function (next def or end of file)
idx_after = code.find("\ndef ", idx_start + 1)
if idx_after < 0:
    idx_after = code.find("\n\ndef ", idx_start + 1)
if idx_after < 0:
    idx_after = len(code)

old_func = code[idx_start:idx_after]
print(f"Found function at chars {idx_start}:{idx_after} ({len(old_func)} chars)")

new_func = '''def parquet_to_geotiff(
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
    )'''

code = code[:idx_start] + new_func + code[idx_after:]
viz_path.write_text(code)
print("PATCHED successfully")
print("New grid: full regular grid anchored to first data point")
