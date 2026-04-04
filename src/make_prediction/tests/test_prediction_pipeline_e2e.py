"""
End-to-end validation for the prediction pipeline output.

Tests spatial patterns, temporal consistency, physical plausibility,
feature back-transformation, PFT-specific behavior, and GeoTIFF integrity.

Designed for the 3-stage pipeline:
  Stage 1: process_era5land_gee  → ERA5 CSV
  Stage 2: predict_sap_velocity  → prediction CSV
  Stage 3: prediction_visualization → GeoTIFF + PNG

Usage:
    python test_prediction_pipeline_e2e.py \\
        --pred-file outputs/e2e_test/predictions/prediction_*.parquet \\
        --era5-file outputs/e2e_test/era5/2020_daily/prediction_*.parquet \\
        --tif-dir   outputs/e2e_test/maps/<run_id>/

Exit code 0 = all pass, 1 = failures found.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

from src.make_prediction.io_utils import read_df
import pandas as pd

PASS_COUNT = 0
FAIL_COUNT = 0
WARN_COUNT = 0


def check(name, condition, detail=""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  PASS: {name}")
    else:
        FAIL_COUNT += 1
        msg = f"  FAIL: {name}"
        if detail:
            msg += f"  [{detail}]"
        print(msg)


def warn(name, detail):
    global WARN_COUNT
    WARN_COUNT += 1
    print(f"  WARN: {name}  [{detail}]")


# ── Region definitions ────────────────────────────────────────────────
REGIONS = {
    "amazon": {"lat": (-10, 5), "lon": (-70, -45)},
    "congo": {"lat": (-5, 5), "lon": (15, 30)},
    "se_asia": {"lat": (-5, 15), "lon": (100, 130)},
    "boreal_canada": {"lat": (50, 65), "lon": (-130, -80)},
    "boreal_siberia": {"lat": (55, 68), "lon": (60, 140)},
    "scandinavia": {"lat": (58, 70), "lon": (5, 30)},
    "eastern_us": {"lat": (30, 45), "lon": (-90, -70)},
    "central_europe": {"lat": (45, 55), "lon": (5, 20)},
    "western_na": {"lat": (40, 65), "lon": (-160, -100)},
}

PFT_COLS = ["ENF", "EBF", "DNF", "DBF", "MF", "WSA", "SAV", "WET"]


def region_slice(df, region):
    """Extract rows within a lat/lon bounding box."""
    r = REGIONS[region]
    return df[
        (df["latitude"] >= r["lat"][0])
        & (df["latitude"] <= r["lat"][1])
        & (df["longitude"] >= r["lon"][0])
        & (df["longitude"] <= r["lon"][1])
    ]


# ======================================================================
# STAGE 2 TESTS — Prediction CSV
# ======================================================================


def test_prediction_physical_range(df):
    """Sap velocity must be in a physically plausible range."""
    print("\n" + "=" * 70)
    print("STAGE 2.1: PREDICTION PHYSICAL RANGE")
    print("=" * 70)

    pred_cols = [c for c in df.columns if "sap_velocity" in c]
    for col in pred_cols:
        vals = df[col].dropna()
        check(f"{col} all non-negative", (vals >= 0).all(), f"min={vals.min():.4f}")
        check(f"{col} max < 50 cm³/cm²/h", vals.max() < 50, f"max={vals.max():.4f}")
        check(f"{col} mean in [1, 15]", 1 <= vals.mean() <= 15, f"mean={vals.mean():.4f}")
        check(f"{col} no NaN", vals.isna().sum() == 0)


def test_feature_backtransform(df):
    """Features should be in physical units, not z-scores."""
    print("\n" + "=" * 70)
    print("STAGE 2.2: FEATURE BACK-TRANSFORMATION")
    print("=" * 70)

    checks = {
        "sw_in": (0, 500, "W/m²"),
        "ta": (-50, 60, "°C"),
        "ta_min": (-50, 60, "°C"),
        "ta_max": (-50, 60, "°C"),
        "vpd": (0, 10, "kPa"),
        "ppfd_in": (0, 2500, "µmol/m²/s"),
        "ext_rad": (0, 550, "W/m²"),
        "ws": (0, 50, "m/s"),
        "day_length": (0, 24, "hours"),
        "elevation": (-500, 9000, "m"),
        "canopy_height": (0, 100, "m"),
        "LAI": (0, 12, "m²/m²"),
    }
    for col, (lo, hi, unit) in checks.items():
        if col not in df.columns:
            warn(f"{col} present", "column missing")
            continue
        vals = df[col].dropna()
        actual_min, actual_max = vals.min(), vals.max()
        # Z-scores would be in ~[-3, 3]; physical values span wider
        in_range = actual_min >= lo and actual_max <= hi
        check(f"{col} in physical range [{lo}, {hi}] {unit}", in_range, f"actual=[{actual_min:.2f}, {actual_max:.2f}]")

    # Extra: sw_in should NOT look like z-scores
    if "sw_in" in df.columns:
        check("sw_in max > 50 (not z-scored)", df["sw_in"].max() > 50, f"max={df['sw_in'].max():.2f}")


def test_timestamp_integrity(df):
    """Timestamp column should contain real dates, no numeric index duplicate."""
    print("\n" + "=" * 70)
    print("STAGE 2.3: TIMESTAMP INTEGRITY")
    print("=" * 70)

    check("timestamp column exists", "timestamp" in df.columns)
    check("no timestamp.1 duplicate", "timestamp.1" not in df.columns)

    if "timestamp" in df.columns:
        sample = str(df["timestamp"].iloc[0])
        is_date = "20" in sample and "-" in sample  # e.g. "2020-07-15..."
        check("timestamp is a date string", is_date, f"sample={sample}")


def test_spatial_patterns(df):
    """Sap velocity should follow expected biogeographical gradients."""
    print("\n" + "=" * 70)
    print("STAGE 2.4: SPATIAL PATTERNS")
    print("=" * 70)

    pred_col = [c for c in df.columns if "sap_velocity" in c and "ensemble" not in c][0]

    # Tropics vs boreal — tropical forests should have higher sap velocity
    tropical = pd.concat([region_slice(df, r) for r in ["amazon", "congo", "se_asia"]])
    boreal = pd.concat([region_slice(df, r) for r in ["boreal_canada", "boreal_siberia"]])

    if len(tropical) > 100 and len(boreal) > 100:
        trop_mean = tropical[pred_col].mean()
        bor_mean = boreal[pred_col].mean()
        check(
            "tropical sap velocity > boreal", trop_mean > bor_mean, f"tropical={trop_mean:.2f}, boreal={bor_mean:.2f}"
        )
        check("tropical mean > 3 cm³/cm²/h", trop_mean > 3, f"tropical mean={trop_mean:.2f}")
    else:
        warn("tropical vs boreal", f"not enough data: tropical={len(tropical)}, boreal={len(boreal)}")

    # Latitude gradient — mean sap velocity should decrease with |latitude|
    if len(df) > 1000:
        df_copy = df.copy()
        df_copy["abs_lat"] = df_copy["latitude"].abs()
        lat_bins = pd.cut(df_copy["abs_lat"], bins=[0, 23, 45, 90], labels=["tropical", "temperate", "high_lat"])
        means = df_copy.groupby(lat_bins, observed=True)[pred_col].mean()
        if len(means) >= 2:
            check(
                "sap velocity decreases with latitude",
                means.iloc[0] > means.iloc[-1],
                f"tropical={means.iloc[0]:.2f}, high_lat={means.iloc[-1]:.2f}",
            )

    # Western NA presence (the PFT fix we validated)
    west_na = region_slice(df, "western_na")
    check("western NA has predictions (PFT fix)", len(west_na) > 500, f"got {len(west_na)} rows")
    if len(west_na) > 0:
        check("western NA sap velocity > 0", west_na[pred_col].mean() > 0, f"mean={west_na[pred_col].mean():.2f}")

    # All major forested regions should have data
    for region_name in REGIONS:
        r = region_slice(df, region_name)
        check(f"{region_name} has data (>50 cells)", len(r) > 50, f"got {len(r)}")


def test_pft_patterns(df):
    """Different PFTs should have distinct sap velocity distributions."""
    print("\n" + "=" * 70)
    print("STAGE 2.5: PFT-SPECIFIC PATTERNS")
    print("=" * 70)

    pred_col = [c for c in df.columns if "sap_velocity" in c and "ensemble" not in c][0]
    existing_pft = [c for c in PFT_COLS if c in df.columns]

    if not existing_pft:
        warn("PFT columns", "none found")
        return

    pft_means = {}
    for pft in existing_pft:
        subset = df[df[pft] == 1]
        if len(subset) > 100:
            pft_means[pft] = subset[pred_col].mean()

    print("  PFT mean sap velocity:")
    for pft, mean_val in sorted(pft_means.items(), key=lambda x: -x[1]):
        print(f"    {pft}: {mean_val:.2f} cm³/cm²/h (n={len(df[df[pft] == 1]):,})")

    # EBF (tropical broadleaf) should have higher sap velocity than ENF (boreal needleleaf)
    if "EBF" in pft_means and "ENF" in pft_means:
        check(
            "EBF sap velocity > ENF",
            pft_means["EBF"] > pft_means["ENF"],
            f"EBF={pft_means['EBF']:.2f}, ENF={pft_means['ENF']:.2f}",
        )

    # At least 4 PFTs should have distinct means (spread > 1 cm³/cm²/h)
    if len(pft_means) >= 4:
        spread = max(pft_means.values()) - min(pft_means.values())
        check("PFT mean spread > 1 cm³/cm²/h", spread > 1, f"spread={spread:.2f}")

    # SAV/WSA (savanna) should have lower sap velocity than EBF (dense forest)
    if "EBF" in pft_means and "SAV" in pft_means:
        check(
            "EBF > SAV sap velocity (dense forest > savanna)",
            pft_means["EBF"] > pft_means["SAV"],
            f"EBF={pft_means['EBF']:.2f}, SAV={pft_means['SAV']:.2f}",
        )


def test_feature_prediction_correlations(df):
    """Sap velocity should correlate with key environmental drivers."""
    print("\n" + "=" * 70)
    print("STAGE 2.6: FEATURE-PREDICTION CORRELATIONS")
    print("=" * 70)

    pred_col = [c for c in df.columns if "sap_velocity" in c and "ensemble" not in c][0]

    correlations = {}
    for feat in ["ta", "vpd", "sw_in", "ppfd_in", "day_length", "LAI"]:
        if feat in df.columns:
            valid = df[[feat, pred_col]].dropna()
            if len(valid) > 100:
                r = valid[feat].corr(valid[pred_col])
                correlations[feat] = r
                print(f"    {feat} vs {pred_col}: r={r:.4f}")

    # Temperature should positively correlate with sap velocity
    if "ta" in correlations:
        check("ta positively correlated with sap velocity", correlations["ta"] > 0, f"r={correlations['ta']:.4f}")

    # VPD should positively correlate (drives transpiration)
    if "vpd" in correlations:
        check("vpd positively correlated with sap velocity", correlations["vpd"] > 0, f"r={correlations['vpd']:.4f}")

    # Radiation should positively correlate
    if "sw_in" in correlations:
        check(
            "sw_in positively correlated with sap velocity", correlations["sw_in"] > 0, f"r={correlations['sw_in']:.4f}"
        )

    # LAI should positively correlate (more leaf area = more transpiration)
    if "LAI" in correlations:
        check("LAI positively correlated with sap velocity", correlations["LAI"] > 0, f"r={correlations['LAI']:.4f}")


def test_elevation_effect(df):
    """Higher elevation forests should generally have lower sap velocity."""
    print("\n" + "=" * 70)
    print("STAGE 2.7: ELEVATION EFFECT")
    print("=" * 70)

    pred_col = [c for c in df.columns if "sap_velocity" in c and "ensemble" not in c][0]

    if "elevation" not in df.columns:
        warn("elevation column", "missing")
        return

    # Compare low (<500m) vs high (>1500m) elevation forests
    low_elev = df[df["elevation"] < 500]
    high_elev = df[df["elevation"] > 1500]

    if len(low_elev) > 100 and len(high_elev) > 100:
        low_mean = low_elev[pred_col].mean()
        high_mean = high_elev[pred_col].mean()
        check(
            "low elevation sap velocity > high elevation",
            low_mean > high_mean,
            f"low(<500m)={low_mean:.2f}, high(>1500m)={high_mean:.2f}",
        )

    # Elevation-sap velocity correlation should be negative (within tropics to control for latitude)
    tropical = df[(df["latitude"].abs() < 23)]
    if len(tropical) > 500:
        r = tropical["elevation"].corr(tropical[pred_col])
        check("elevation vs sap velocity negative in tropics (lapse rate)", r < 0, f"r={r:.4f}")


def test_spatial_continuity(df):
    """Nearby grid cells should have similar sap velocity (spatial autocorrelation)."""
    print("\n" + "=" * 70)
    print("STAGE 2.8: SPATIAL CONTINUITY")
    print("=" * 70)

    pred_col = [c for c in df.columns if "sap_velocity" in c and "ensemble" not in c][0]

    # Sample a small region and check that variance is lower than global
    for region_name in ["amazon", "central_europe", "boreal_siberia"]:
        r = region_slice(df, region_name)
        if len(r) > 100:
            global_std = df[pred_col].std()
            regional_std = r[pred_col].std()
            check(
                f"{region_name} regional std < global std (spatial autocorrelation)",
                regional_std < global_std,
                f"regional={regional_std:.2f}, global={global_std:.2f}",
            )


def test_ensemble_consistency(df):
    """If ensemble column exists, it should be consistent with individual models."""
    print("\n" + "=" * 70)
    print("STAGE 2.9: ENSEMBLE CONSISTENCY")
    print("=" * 70)

    pred_cols = [
        c
        for c in df.columns
        if c.startswith("sap_velocity_")
        and "ensemble" not in c
        and not any(c.endswith(x) for x in ["_location", "_window_start", "_window_end", "_window_pos"])
    ]
    ens_col = "sap_velocity_ensemble"

    if ens_col not in df.columns:
        warn("ensemble column", "not found")
        return

    if len(pred_cols) == 1:
        # Single model — ensemble should equal the model
        diff = (df[ens_col] - df[pred_cols[0]]).abs().max()
        check("ensemble equals single model", diff < 1e-6, f"max_diff={diff:.8f}")
    elif len(pred_cols) > 1:
        # Multi-model — ensemble should be between min and max
        ens = df[ens_col]
        model_min = df[pred_cols].min(axis=1)
        model_max = df[pred_cols].max(axis=1)
        check("ensemble between model min/max", ((ens >= model_min - 1e-6) & (ens <= model_max + 1e-6)).all())


# ======================================================================
# STAGE 3 TESTS — GeoTIFF + PNG
# ======================================================================


def test_geotiff_structure(tif_path):
    """GeoTIFF should have correct CRS, resolution, and extent."""
    print("\n" + "=" * 70)
    print("STAGE 3.1: GEOTIFF STRUCTURE")
    print("=" * 70)

    import rasterio

    with rasterio.open(tif_path) as src:
        check("CRS is EPSG:4326", src.crs.to_epsg() == 4326, f"got {src.crs}")
        check("resolution is 0.1°", abs(src.res[0] - 0.1) < 0.01, f"got {src.res}")
        check("height > 100 pixels", src.height > 100, f"got {src.height}")
        check("width > 100 pixels", src.width > 100, f"got {src.width}")

        data = src.read(1)
        valid = data[~np.isnan(data)]

        check("has valid (non-NaN) pixels", len(valid) > 0, f"got {len(valid)} valid pixels")

        if len(valid) > 0:
            check("GeoTIFF values non-negative", valid.min() >= 0, f"min={valid.min():.4f}")
            check("GeoTIFF max < 50", valid.max() < 50, f"max={valid.max():.4f}")


def test_geotiff_vs_csv(tif_path, df):
    """GeoTIFF pixel values should match the prediction CSV statistics."""
    print("\n" + "=" * 70)
    print("STAGE 3.2: GEOTIFF vs PREDICTION CONSISTENCY")
    print("=" * 70)

    import rasterio

    pred_col = [c for c in df.columns if "sap_velocity" in c and "ensemble" not in c][0]

    with rasterio.open(tif_path) as src:
        data = src.read(1)
        valid = data[~np.isnan(data)]

    csv_vals = df[pred_col].dropna()

    if len(valid) > 0 and len(csv_vals) > 0:
        # Pixel count should be close to CSV row count (may differ slightly due to aggregation)
        ratio = len(valid) / len(csv_vals)
        check(
            "GeoTIFF pixel count ≈ CSV rows (within 5%)",
            0.95 < ratio < 1.05,
            f"tif={len(valid):,}, csv={len(csv_vals):,}, ratio={ratio:.3f}",
        )

        # Mean should be similar
        tif_mean = valid.mean()
        csv_mean = csv_vals.mean()
        mean_diff = abs(tif_mean - csv_mean) / csv_mean
        check(
            "GeoTIFF mean ≈ CSV mean (within 10%)",
            mean_diff < 0.10,
            f"tif_mean={tif_mean:.4f}, csv_mean={csv_mean:.4f}",
        )

        # Range should be similar
        check(
            "GeoTIFF min ≈ CSV min (within 0.5)",
            abs(valid.min() - csv_vals.min()) < 0.5,
            f"tif_min={valid.min():.4f}, csv_min={csv_vals.min():.4f}",
        )
        check(
            "GeoTIFF max ≈ CSV max (within 0.5)",
            abs(valid.max() - csv_vals.max()) < 0.5,
            f"tif_max={valid.max():.4f}, csv_max={csv_vals.max():.4f}",
        )


def test_geotiff_spatial_patterns(tif_path):
    """GeoTIFF should have spatial patterns consistent with biogeography."""
    print("\n" + "=" * 70)
    print("STAGE 3.3: GEOTIFF SPATIAL PATTERNS")
    print("=" * 70)

    import rasterio

    with rasterio.open(tif_path) as src:
        data = src.read(1)
        transform = src.transform

        # Convert pixel coords to lat/lon and check tropical vs boreal
        nrows, ncols = data.shape
        # Build lat/lon arrays from transform
        lons = np.array([transform.c + (j + 0.5) * transform.a for j in range(ncols)])
        lats = np.array([transform.f + (i + 0.5) * transform.e for i in range(nrows)])

        # Tropical band: |lat| < 23
        trop_rows = np.where(np.abs(lats) < 23)[0]
        boreal_rows = np.where(np.abs(lats) > 50)[0]

        if len(trop_rows) > 0 and len(boreal_rows) > 0:
            trop_data = data[trop_rows, :]
            bor_data = data[boreal_rows, :]

            trop_valid = trop_data[~np.isnan(trop_data)]
            bor_valid = bor_data[~np.isnan(bor_data)]

            if len(trop_valid) > 100 and len(bor_valid) > 100:
                check(
                    "GeoTIFF: tropical mean > boreal mean",
                    trop_valid.mean() > bor_valid.mean(),
                    f"tropical={trop_valid.mean():.2f}, boreal={bor_valid.mean():.2f}",
                )

        # Western hemisphere should have valid pixels (PFT fix)
        west_cols = np.where(lons < -100)[0]
        if len(west_cols) > 0:
            west_data = data[:, west_cols]
            west_valid = west_data[~np.isnan(west_data)]
            check(
                "GeoTIFF: western NA has valid pixels",
                len(west_valid) > 500,
                f"got {len(west_valid)} valid pixels west of -100°",
            )


def test_png_exists(tif_dir):
    """PNG files should exist alongside GeoTIFFs."""
    print("\n" + "=" * 70)
    print("STAGE 3.4: PNG OUTPUT")
    print("=" * 70)

    pngs = glob.glob(os.path.join(tif_dir, "*.png"))
    tifs = glob.glob(os.path.join(tif_dir, "*.tif"))

    check("at least 1 PNG map generated", len(pngs) > 0, f"found {len(pngs)} PNGs")
    check("at least 1 GeoTIFF generated", len(tifs) > 0, f"found {len(tifs)} TIFs")

    for png in pngs:
        size_kb = os.path.getsize(png) / 1024
        check(f"PNG {os.path.basename(png)} size > 50KB", size_kb > 50, f"got {size_kb:.1f} KB")


# ======================================================================
# CROSS-STAGE TESTS
# ======================================================================


def test_era5_vs_prediction_consistency(era5_df, pred_df):
    """Prediction output should have same spatial extent as ERA5 input."""
    print("\n" + "=" * 70)
    print("CROSS-STAGE: ERA5 vs PREDICTION CONSISTENCY")
    print("=" * 70)

    # Row count should match (prediction adds columns, doesn't drop rows)
    check(
        "row count matches ERA5 → prediction",
        len(pred_df) == len(era5_df),
        f"era5={len(era5_df):,}, pred={len(pred_df):,}",
    )

    # Longitude range should match
    era5_lon_range = (era5_df["longitude"].min(), era5_df["longitude"].max())
    pred_lon_range = (pred_df["longitude"].min(), pred_df["longitude"].max())
    check(
        "longitude range matches",
        abs(era5_lon_range[0] - pred_lon_range[0]) < 0.2 and abs(era5_lon_range[1] - pred_lon_range[1]) < 0.2,
        f"era5={era5_lon_range}, pred={pred_lon_range}",
    )

    # Prediction should add sap_velocity columns
    pred_only_cols = [c for c in pred_df.columns if c not in era5_df.columns]
    sap_cols = [c for c in pred_only_cols if "sap_velocity" in c]
    check("prediction adds sap_velocity columns", len(sap_cols) > 0, f"new cols: {pred_only_cols}")


# ======================================================================
# MAIN
# ======================================================================


def main():
    parser = argparse.ArgumentParser(description="E2E prediction pipeline validation")
    parser.add_argument("--pred-file", type=str, required=True, help="Path to prediction file (CSV or Parquet, glob pattern OK)")
    parser.add_argument("--era5-file", type=str, default=None, help="Path to ERA5 file for cross-stage checks (CSV or Parquet, optional)")
    parser.add_argument("--tif-dir", type=str, default=None, help="Directory with GeoTIFF/PNG output (optional)")
    args = parser.parse_args()

    # Resolve glob patterns
    pred_files = sorted(glob.glob(args.pred_file))
    if not pred_files:
        print(f"ERROR: No files match {args.pred_file}")
        sys.exit(1)

    print(f"Loading prediction file: {pred_files[0]}")
    pred_df = read_df(pred_files[0])
    print(f"  Shape: {pred_df.shape}")
    print(f"  Columns: {list(pred_df.columns)}")

    # ── Stage 2 tests ──
    test_prediction_physical_range(pred_df)
    test_feature_backtransform(pred_df)
    test_timestamp_integrity(pred_df)
    test_spatial_patterns(pred_df)
    test_pft_patterns(pred_df)
    test_feature_prediction_correlations(pred_df)
    test_elevation_effect(pred_df)
    test_spatial_continuity(pred_df)
    test_ensemble_consistency(pred_df)

    # ── Stage 3 tests ──
    if args.tif_dir:
        tifs = sorted(glob.glob(os.path.join(args.tif_dir, "*.tif")))
        if tifs:
            test_geotiff_structure(tifs[0])
            test_geotiff_vs_csv(tifs[0], pred_df)
            test_geotiff_spatial_patterns(tifs[0])
        else:
            warn("GeoTIFF files", f"none found in {args.tif_dir}")
        test_png_exists(args.tif_dir)

    # ── Cross-stage tests ──
    if args.era5_file:
        era5_files = sorted(glob.glob(args.era5_file))
        if era5_files:
            print(f"\nLoading ERA5 file: {era5_files[0]}")
            era5_df = read_df(era5_files[0])
            test_era5_vs_prediction_consistency(era5_df, pred_df)

    # ── Summary ──
    print(f"\n{'=' * 70}")
    total = PASS_COUNT + FAIL_COUNT
    print(
        f"RESULTS: {PASS_COUNT}/{total} passed ({100 * PASS_COUNT / max(total, 1):.1f}%), "
        f"{FAIL_COUNT} failed, {WARN_COUNT} warnings"
    )
    status = "ALL TESTS PASSED" if FAIL_COUNT == 0 else f"{FAIL_COUNT} TESTS FAILED"
    print(f"STATUS: {status}")
    print(f"{'=' * 70}")
    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    main()
