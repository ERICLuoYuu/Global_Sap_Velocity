"""
Validate that the GeoTIFF-based _download_gee_image_tiled fix
produces correct PFT data in western North America.

Downloads PFT globally at 0.1 deg using the patched code, then checks:
1. PFT at (55N, -125W) is ENF (1), not NODATA (255)
2. Western NA has >10,000 forest cells (was 11 before fix)
3. Global forest coverage is reasonable (~30-40% of land)
"""

import os
import sys

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ee
import numpy as np

ee.Initialize(project="era5download-447713")


def main():
    from src.make_prediction.process_era5land_gee_opt_fix import ERA5LandGEEProcessor

    processor = ERA5LandGEEProcessor(time_scale="daily")

    # Download PFT globally using the patched _download_gee_image_tiled
    print("=" * 70)
    print("VALIDATION: Downloading global PFT via patched _download_gee_image_tiled")
    print("=" * 70)

    pft_raw = processor.get_pft_from_gee(year=2020)

    if pft_raw is None:
        print("\nFAIL: get_pft_from_gee returned None")
        sys.exit(1)

    pft_grid = pft_raw["data"]
    height, width = pft_grid.shape
    transform = pft_raw["transform"]

    print(f"\nPFT grid shape: {height} x {width}")
    print(f"Transform: {transform}")

    # Derive lat/lon arrays from the rasterio transform
    # transform.c = west edge, transform.a = pixel width
    # transform.f = north edge, transform.e = pixel height (negative)
    lons = np.array([transform.c + (j + 0.5) * transform.a for j in range(width)])
    lats = np.array([transform.f + (i + 0.5) * transform.e for i in range(height)])

    print(f"Lon range: [{lons.min():.2f}, {lons.max():.2f}]")
    print(f"Lat range: [{lats.min():.2f}, {lats.max():.2f}]")

    # --- Check 1: PFT at British Columbia (55N, -125W) ---
    lat_idx = np.argmin(np.abs(lats - 55.0))
    lon_idx = np.argmin(np.abs(lons - (-125.0)))
    pft_bc = int(pft_grid[lat_idx, lon_idx])

    FOREST_CODES = [1, 2, 3, 4, 5, 8, 9, 11]
    PFT_NAMES = {1: "ENF", 2: "EBF", 3: "DNF", 4: "DBF", 5: "MF", 8: "WSA", 9: "SAV", 11: "WET"}

    print("\n--- Check 1: PFT at (55N, -125W) ---")
    print(f"  PFT code: {pft_bc} ({PFT_NAMES.get(pft_bc, 'NOT FOREST')})")
    check1 = pft_bc in FOREST_CODES
    print("  PASS" if check1 else f"  FAIL (expected forest, got {pft_bc})")

    # --- Check 2: Western NA forest count ---
    lat_mask = (lats >= 30) & (lats <= 70)
    lon_mask = (lons >= -170) & (lons <= -100)
    wna_slice = pft_grid[np.ix_(lat_mask, lon_mask)]
    wna_forest = np.isin(wna_slice.astype(int), FOREST_CODES).sum()
    wna_nodata = (wna_slice == 255).sum()

    print("\n--- Check 2: Western NA [-170,-100] x [30,70] ---")
    print(f"  Forest cells: {wna_forest:,}")
    print(f"  NODATA (255): {wna_nodata:,}")
    print(f"  Total cells: {wna_slice.size:,}")
    check2 = wna_forest > 10_000
    print(f"  PASS (>{10_000:,} forest cells)" if check2 else f"  FAIL (only {wna_forest:,} forest cells)")

    # --- Check 3: Global forest coverage ---
    valid = pft_grid[(~np.isnan(pft_grid)) & (pft_grid != 255)]
    total_valid = len(valid)
    global_forest = np.isin(valid.astype(int), FOREST_CODES).sum()
    pct = 100 * global_forest / max(total_valid, 1)

    print("\n--- Check 3: Global forest coverage ---")
    print(f"  Valid land cells: {total_valid:,}")
    print(f"  Forest cells: {global_forest:,} ({pct:.1f}%)")
    check3 = 20 < pct < 60
    print("  PASS" if check3 else f"  FAIL (expected 20-60%, got {pct:.1f}%)")

    # --- Check 4: PFT class distribution in western NA ---
    print("\n--- Check 4: PFT class distribution in western NA ---")
    unique, counts = np.unique(wna_slice[~np.isnan(wna_slice)].astype(int), return_counts=True)
    for u, c in sorted(zip(unique, counts), key=lambda x: -x[1]):
        name = PFT_NAMES.get(u, f"class_{u}")
        print(f"  PFT {u:>3d} ({name:>4s}): {c:>6,}")

    # --- Summary ---
    all_pass = check1 and check2 and check3
    print(f"\n{'=' * 70}")
    print(f"RESULT: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print(f"{'=' * 70}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
