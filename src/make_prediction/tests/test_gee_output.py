"""
Comprehensive validation test suite for GEE ERA5-Land prediction output.

Handles large files (37GB+) via chunked pandas reading.
Tests: structural integrity, all processed variables present,
physical plausibility, forest mask effectiveness, placeholder/default detection,
internal consistency, derived variable formulas, PFT encoding,
spatial/temporal coverage, and cross-day consistency.

Usage:
    python test_gee_output.py [path_to_csv]

Exit code 0 = all pass, 1 = failures found.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Counters ──────────────────────────────────────────────────────────
PASS_COUNT = 0
FAIL_COUNT = 0
WARN_COUNT = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print("  PASS: %s" % name)
    else:
        FAIL_COUNT += 1
        msg = "  FAIL: %s" % name
        if detail:
            msg += "  [%s]" % detail
        print(msg)


def warn(name: str, detail: str) -> None:
    global WARN_COUNT
    WARN_COUNT += 1
    print("  WARN: %s  [%s]" % (name, detail))


# ── Configuration ─────────────────────────────────────────────────────
CHUNK_SIZE = 1_000_000

# Default path — override via CLI arg
DEFAULT_CSV = Path("/scratch/tmp/yluo2/gsv-wt/map-viz/outputs/predictions/2020_daily/prediction_2020_01_daily.csv")

# ALL processed variables that MUST be present in output
EXPECTED_COLUMNS = [
    # Coordinates
    "latitude",
    "longitude",
    # ERA5 dynamic (16)
    "sw_in",
    "ppfd_in",
    "ext_rad",
    "ws",
    "ws_min",
    "ws_max",
    "ta",
    "ta_min",
    "ta_max",
    "vpd",
    "vpd_min",
    "vpd_max",
    "day_length",
    "prcip/PET",
    "volumetric_soil_water_layer_1",
    "soil_temperature_level_1",
    # Static
    "elevation",
    "canopy_height",
    "LAI",
    # PFT one-hot (8)
    "ENF",
    "EBF",
    "DNF",
    "DBF",
    "MF",
    "WSA",
    "SAV",
    "WET",
    # Climate
    "mean_annual_temp",
    "mean_annual_precip",
    # Derived time features
    "solar_timestamp",
    "Day sin",
    "Year sin",
]

PFT_COLS = ["ENF", "EBF", "DNF", "DBF", "MF", "WSA", "SAV", "WET"]

# ERA5 dynamic columns — should all be NaN together (forest mask consistency)
ERA5_DYNAMIC = [
    "sw_in",
    "ppfd_in",
    "ws",
    "ws_min",
    "ws_max",
    "ta",
    "ta_min",
    "ta_max",
    "vpd",
    "vpd_min",
    "vpd_max",
    "prcip/PET",
    "volumetric_soil_water_layer_1",
    "soil_temperature_level_1",
]

# Physical range bounds: (min, max, unit)
RANGE_BOUNDS = {
    "ta": (-90, 60, "degC"),
    "ta_min": (-90, 60, "degC"),
    "ta_max": (-90, 60, "degC"),
    "vpd": (0, 8, "kPa"),
    "vpd_min": (0, 6, "kPa"),
    "vpd_max": (0, 12, "kPa"),
    "ws": (0, 50, "m/s"),
    "ws_min": (0, 50, "m/s"),
    "ws_max": (0, 60, "m/s"),
    "sw_in": (0, 500, "W/m2"),
    "ppfd_in": (0, 2500, "umol/m2/s"),
    "ext_rad": (0, 550, "W/m2"),
    "day_length": (0, 24, "hours"),
    "prcip/PET": (0, 1000, "ratio"),
    "volumetric_soil_water_layer_1": (-0.001, 1, "m3/m3"),
    "soil_temperature_level_1": (-50, 60, "degC"),
    "elevation": (-500, 9000, "m"),
    "canopy_height": (0, 100, "m"),
    "LAI": (0, 12, "m2/m2"),
    "mean_annual_temp": (-30, 35, "degC"),
    "mean_annual_precip": (0, 12000, "mm"),
    "Day sin": (-1, 1, ""),
    "Year sin": (-1, 1, ""),
}

# Placeholder defaults — flag if too many forest pixels have exactly this value
PLACEHOLDER_DEFAULTS = {
    "elevation": (0.0, "default_value=0.0 in add_static_var_to_merged"),
    "canopy_height": (0.0, "default_value=0.0 in add_static_var_to_merged"),
    "mean_annual_temp": (15.0, "default in point-based fallback"),
    "mean_annual_precip": (800.0, "default in point-based fallback"),
}

# Grid specs from pipeline
GRID_NLAT = 1381
GRID_NLON = 3601
GRID_CELLS_PER_DAY = GRID_NLAT * GRID_NLON  # 4,972,981
EXPECTED_DAYS = None  # auto-detected from data
EXPECTED_TOTAL_ROWS = None  # auto-detected

# Forest mask: 662,934 / 4,972,981 = 13.33% from job log
FOREST_MASK_PCT_MIN = 10.0
FOREST_MASK_PCT_MAX = 20.0

# Major forest regions for spatial spot-checks: (lat_range, lon_range)
FOREST_REGIONS = {
    "Amazon": ((-15, -2), (-70, -50)),
    "Congo Basin": ((-5, 5), (15, 30)),
    "Boreal Canada": ((50, 65), (-130, -60)),
    "Boreal Siberia": ((55, 70), (60, 140)),
    "SE Asia": ((-5, 15), (95, 130)),
    "Central Europe": ((45, 55), (5, 20)),
    "Eastern US": ((30, 45), (-90, -70)),
    "Scandinavia": ((58, 70), (5, 30)),
}


# =====================================================================
# MAIN
# =====================================================================
def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV

    print("=" * 72)
    print("GEE OUTPUT VALIDATION — COMPREHENSIVE TEST SUITE (CHUNKED)")
    print("=" * 72)
    print("File: %s" % csv_path)

    if not csv_path.exists():
        print("FATAL: File not found.")
        sys.exit(1)

    fsize_gb = csv_path.stat().st_size / 1e9
    print("Size: %.2f GB" % fsize_gb)

    # ─── 0. READ HEADER ONLY ─────────────────────────────────────────
    header_df = pd.read_csv(csv_path, nrows=0)
    raw_columns = list(header_df.columns)
    print("Raw columns (%d): %s" % (len(raw_columns), raw_columns))

    # Handle duplicate 'timestamp' column (index + actual timestamp)
    # pandas renames duplicates: timestamp, timestamp.1
    ts_col = "timestamp.1" if "timestamp.1" in raw_columns else "timestamp"

    # ================================================================
    # 1. STRUCTURAL INTEGRITY — ALL PROCESSED VARIABLES PRESENT
    # ================================================================
    print("\n" + "=" * 72)
    print("1. STRUCTURAL INTEGRITY — ALL PROCESSED VARIABLES PRESENT")
    print("=" * 72)

    actual_cols = set(raw_columns)
    if "timestamp.1" in actual_cols:
        actual_cols.add("timestamp")

    missing_cols = []
    for c in EXPECTED_COLUMNS:
        present = c in actual_cols
        check("Column present: %s" % c, present)
        if not present:
            missing_cols.append(c)

    if missing_cols:
        print("\n  ** MISSING COLUMNS: %s **" % missing_cols)
        print("  These variables should be produced by the processing pipeline.")
        print("  Check: climate data loaded? All GEE variables downloaded?")

    check(
        "No unexpected duplicate columns",
        len(raw_columns) == len(set(raw_columns))
        or (raw_columns.count("timestamp") == 2 and len(raw_columns) - 1 == len(set(raw_columns))),
        "duplicates beyond expected 'timestamp': %s"
        % [c for c in raw_columns if raw_columns.count(c) > 1 and c != "timestamp"],
    )

    # ================================================================
    # 2. CHUNKED FULL-FILE SCAN — accumulate statistics
    # ================================================================
    print("\n" + "=" * 72)
    print("2. CHUNKED FULL-FILE SCAN")
    print("=" * 72)
    print("  Chunk size: %d rows" % CHUNK_SIZE)

    t0 = time.time()

    # ── Accumulators ──────────────────────────────────────────────────
    total_rows = 0
    col_min: dict[str, float] = {}
    col_max: dict[str, float] = {}
    col_nan: dict[str, int] = {}
    col_count: dict[str, int] = {}

    day_row_counts: dict[str, int] = {}

    forest_rows = 0
    non_forest_rows = 0
    nan_inconsistent_rows = 0

    ta_order_violations = 0
    vpd_order_violations = 0
    ws_order_violations = 0

    pft_sum_zero = 0
    pft_sum_one = 0
    pft_sum_multi = 0
    pft_non_binary = 0
    pft_class_counts = {c: 0 for c in PFT_COLS}

    forest_no_pft = 0
    forest_lai_nan = 0

    placeholder_counts = {k: 0 for k in PLACEHOLDER_DEFAULTS}

    ppfd_max_err = 0.0
    ppfd_err_sum = 0.0
    ppfd_err_count = 0

    lat_min_global = np.inf
    lat_max_global = -np.inf
    lon_min_global = np.inf
    lon_max_global = -np.inf

    region_counts = {name: 0 for name in FOREST_REGIONS}

    south_daylen_sum = 0.0
    south_daylen_n = 0
    north_daylen_sum = 0.0
    north_daylen_n = 0
    year_sin_min = np.inf
    year_sin_max = -np.inf

    soil_temp_max = -np.inf

    # -- NEW: spatial/cross-variable pattern accumulators --
    sw_gt_ext_violations = 0

    tropical_ta_sum = 0.0; tropical_ta_n = 0
    boreal_ta_sum = 0.0; boreal_ta_n = 0
    tropical_ch_sum = 0.0; tropical_ch_n = 0
    boreal_ch_sum = 0.0; boreal_ch_n = 0
    tropical_lai_sum = 0.0; tropical_lai_n = 0
    boreal_lai_sum = 0.0; boreal_lai_n = 0
    equatorial_daylen_sum = 0.0; equatorial_daylen_n = 0

    CORR_SAMPLE_MAX = 50000
    corr_samples = {"ta": [], "soil_temp": [], "elev": [], "lat": [],
                    "mat": [], "map_": [], "lai": [], "sw_in": [], "ext_rad": []}

    col_sum: dict[str, float] = {}
    col_sum_sq: dict[str, float] = {}

    first_day_coords_seen: set[tuple] = set()
    first_day_dup_count = 0

    first_day_key: str | None = None
    first_day_statics: dict[tuple, dict] = {}
    cross_day_diffs = {v: 0 for v in ["elevation", "canopy_height", "mean_annual_temp"]}
    CROSS_DAY_SAMPLE = 500

    # ── Iterate chunks ────────────────────────────────────────────────
    chunk_num = 0
    reader = pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False)

    for chunk in reader:
        chunk_num += 1
        n = len(chunk)
        total_rows += n

        if chunk_num % 20 == 1:
            elapsed = time.time() - t0
            print("  Chunk %d: %d rows so far (%.0fs)" % (chunk_num, total_rows, elapsed))

        # ── Per-day row counts ────────────────────────────────────────
        if ts_col in chunk.columns:
            day_strings = chunk[ts_col].astype(str).str[:10]
            for day_str, cnt in day_strings.value_counts().items():
                day_row_counts[day_str] = day_row_counts.get(day_str, 0) + int(cnt)

        # ── Per-column numeric stats ──────────────────────────────────
        for col in chunk.columns:
            if col in ("timestamp", ts_col, "solar_timestamp"):
                continue
            if chunk[col].dtype in (
                np.float64,
                np.float32,
                np.int64,
                np.int32,
                np.int8,
                np.uint8,
            ):
                cmin = chunk[col].min()
                cmax = chunk[col].max()
                cnan = int(chunk[col].isna().sum())

                if not pd.isna(cmin):
                    col_min[col] = min(col_min.get(col, np.inf), cmin)
                if not pd.isna(cmax):
                    col_max[col] = max(col_max.get(col, -np.inf), cmax)
                col_nan[col] = col_nan.get(col, 0) + cnan
                col_count[col] = col_count.get(col, 0) + n

        # ── Lat/Lon range ─────────────────────────────────────────────
        if "latitude" in chunk.columns:
            lat_min_global = min(lat_min_global, chunk["latitude"].min())
            lat_max_global = max(lat_max_global, chunk["latitude"].max())
        if "longitude" in chunk.columns:
            lon_min_global = min(lon_min_global, chunk["longitude"].min())
            lon_max_global = max(lon_max_global, chunk["longitude"].max())

        # ── Forest mask effectiveness ─────────────────────────────────
        if "ta" not in chunk.columns:
            continue

        is_forest = chunk["ta"].notna()
        forest_n = int(is_forest.sum())
        forest_rows += forest_n
        non_forest_rows += n - forest_n

        # ── NaN consistency across ERA5 dynamic vars ──────────────────
        era5_present = [c for c in ERA5_DYNAMIC if c in chunk.columns]
        if len(era5_present) >= 2:
            era5_nan_matrix = chunk[era5_present].isna()
            all_nan = era5_nan_matrix.all(axis=1)
            none_nan = (~era5_nan_matrix).all(axis=1)
            inconsistent = ~(all_nan | none_nan)
            nan_inconsistent_rows += int(inconsistent.sum())

        # ── Forest-pixel-only tests ───────────────────────────────────
        forest_chunk = chunk[is_forest]
        if len(forest_chunk) == 0:
            continue

        # Cross-column ordering
        if all(c in forest_chunk.columns for c in ["ta", "ta_min", "ta_max"]):
            tol = 0.015
            ta_order_violations += int(
                (
                    (forest_chunk["ta"] < forest_chunk["ta_min"] - tol)
                    | (forest_chunk["ta"] > forest_chunk["ta_max"] + tol)
                ).sum()
            )

        if all(c in forest_chunk.columns for c in ["vpd_min", "vpd_max"]):
            vpd_order_violations += int((forest_chunk["vpd_min"] > forest_chunk["vpd_max"]).sum())

        if all(c in forest_chunk.columns for c in ["ws_min", "ws_max"]):
            ws_order_violations += int((forest_chunk["ws_min"] > forest_chunk["ws_max"]).sum())

        # sw_in <= ext_rad constraint
        if all(col in forest_chunk.columns for col in ["sw_in", "ext_rad"]):
            sw_gt_ext_violations += int((forest_chunk["sw_in"] > forest_chunk["ext_rad"] + 0.5).sum())

        # Latitude-binned spatial pattern accumulators
        if all(col in forest_chunk.columns for col in ["latitude", "ta"]):
            lat_abs = forest_chunk["latitude"].abs()
            trop = lat_abs < 23; bor = lat_abs > 55
            if trop.any():
                tropical_ta_sum += forest_chunk.loc[trop, "ta"].sum(); tropical_ta_n += int(trop.sum())
            if bor.any():
                boreal_ta_sum += forest_chunk.loc[bor, "ta"].sum(); boreal_ta_n += int(bor.sum())

        if all(col in forest_chunk.columns for col in ["latitude", "canopy_height"]):
            lat_abs = forest_chunk["latitude"].abs()
            trop = lat_abs < 23; bor = lat_abs > 55
            if trop.any():
                tropical_ch_sum += forest_chunk.loc[trop, "canopy_height"].sum(); tropical_ch_n += int(trop.sum())
            if bor.any():
                boreal_ch_sum += forest_chunk.loc[bor, "canopy_height"].sum(); boreal_ch_n += int(bor.sum())

        if all(col in forest_chunk.columns for col in ["latitude", "LAI"]):
            lat_abs = forest_chunk["latitude"].abs()
            trop = lat_abs < 23; bor = lat_abs > 55
            if trop.any():
                tropical_lai_sum += forest_chunk.loc[trop, "LAI"].sum(); tropical_lai_n += int(trop.sum())
            if bor.any():
                boreal_lai_sum += forest_chunk.loc[bor, "LAI"].sum(); boreal_lai_n += int(bor.sum())

        if all(col in forest_chunk.columns for col in ["latitude", "day_length"]):
            eq = forest_chunk["latitude"].abs() < 5
            if eq.any():
                equatorial_daylen_sum += forest_chunk.loc[eq, "day_length"].sum()
                equatorial_daylen_n += int(eq.sum())

        # Cross-variable correlation reservoir sample (vary seed per chunk)
        if len(corr_samples["ta"]) < CORR_SAMPLE_MAX:
            sample = forest_chunk.sample(n=min(500, len(forest_chunk)),
                                         random_state=42 + chunk_num)
            for var, col_name in [("ta", "ta"), ("soil_temp", "soil_temperature_level_1"),
                                   ("elev", "elevation"), ("lat", "latitude"),
                                   ("mat", "mean_annual_temp"), ("map_", "mean_annual_precip"),
                                   ("lai", "LAI"), ("sw_in", "sw_in"), ("ext_rad", "ext_rad")]:
                if col_name in sample.columns:
                    corr_samples[var].extend(sample[col_name].dropna().tolist())

        # Per-column running sums for std check
        for col in ["ta", "vpd", "sw_in", "elevation", "LAI", "day_length",
                     "canopy_height", "mean_annual_temp", "mean_annual_precip"]:
            if col in forest_chunk.columns:
                vals = forest_chunk[col].dropna()
                col_sum[col] = col_sum.get(col, 0) + float(vals.sum())
                col_sum_sq[col] = col_sum_sq.get(col, 0) + float((vals ** 2).sum())

        # Duplicate coordinate check (first day only)
        if ts_col in forest_chunk.columns and first_day_key is not None:
            day_str_fc = str(forest_chunk[ts_col].iloc[0])[:10]
            if day_str_fc == first_day_key:
                for _, row in forest_chunk[["latitude", "longitude"]].head(200).iterrows():
                    coord = (round(row["latitude"], 4), round(row["longitude"], 4))
                    if coord in first_day_coords_seen:
                        first_day_dup_count += 1
                    else:
                        first_day_coords_seen.add(coord)

        # PFT stats (on forest pixels)
        pft_present = [c for c in PFT_COLS if c in forest_chunk.columns]
        if pft_present:
            pft_vals = forest_chunk[pft_present]
            row_sums = pft_vals.sum(axis=1)
            pft_sum_zero += int((row_sums == 0).sum())
            pft_sum_one += int((row_sums == 1).sum())
            pft_sum_multi += int((row_sums > 1).sum())
            pft_non_binary += int((~pft_vals.isin([0, 1])).any(axis=1).sum())
            for c in pft_present:
                pft_class_counts[c] += int((forest_chunk[c] == 1).sum())
            forest_no_pft += int((row_sums == 0).sum())

        # Forest pixels with LAI NaN
        if "LAI" in forest_chunk.columns:
            forest_lai_nan += int(forest_chunk["LAI"].isna().sum())

        # Placeholder detection (on forest pixels)
        for var, (default_val, _reason) in PLACEHOLDER_DEFAULTS.items():
            if var in forest_chunk.columns:
                placeholder_counts[var] += int((forest_chunk[var] == default_val).sum())

        # PPFD formula: ppfd_in = clip(sw_in * 0.45 * 4.6, 0, 2500)
        if all(c in forest_chunk.columns for c in ["sw_in", "ppfd_in"]):
            sw = forest_chunk["sw_in"].values
            ppfd_actual = forest_chunk["ppfd_in"].values
            ppfd_expected = np.clip(sw * 0.45 * 4.6, 0, 2500)
            valid = ~(np.isnan(sw) | np.isnan(ppfd_actual))
            if valid.any():
                err = np.abs(ppfd_actual[valid] - ppfd_expected[valid])
                ppfd_max_err = max(ppfd_max_err, float(err.max()))
                ppfd_err_sum += float(err.sum())
                ppfd_err_count += int(valid.sum())

        # Day length: south vs north (January = southern summer)
        if all(c in forest_chunk.columns for c in ["day_length", "latitude"]):
            south = forest_chunk.loc[forest_chunk["latitude"] < -20, "day_length"]
            north = forest_chunk.loc[forest_chunk["latitude"] > 50, "day_length"]
            if len(south) > 0:
                south_daylen_sum += south.sum()
                south_daylen_n += len(south)
            if len(north) > 0:
                north_daylen_sum += north.sum()
                north_daylen_n += len(north)

        # Year sin range
        if "Year sin" in forest_chunk.columns:
            ys = forest_chunk["Year sin"]
            year_sin_min = min(year_sin_min, ys.min())
            year_sin_max = max(year_sin_max, ys.max())

        # Kelvin detector
        if "soil_temperature_level_1" in forest_chunk.columns:
            soil_temp_max = max(soil_temp_max, forest_chunk["soil_temperature_level_1"].max())

        # Spatial region counts
        for rname, ((lat_lo, lat_hi), (lon_lo, lon_hi)) in FOREST_REGIONS.items():
            mask = forest_chunk["latitude"].between(lat_lo, lat_hi) & forest_chunk["longitude"].between(lon_lo, lon_hi)
            region_counts[rname] += int(mask.sum())

        # Cross-day consistency: capture first day, compare later days
        if ts_col in forest_chunk.columns:
            day_str = str(forest_chunk[ts_col].iloc[0])[:10]
            static_vars_check = [
                v for v in ["elevation", "canopy_height", "mean_annual_temp"] if v in forest_chunk.columns
            ]
            if not static_vars_check:
                continue

            if first_day_key is None:
                first_day_key = day_str

            if day_str == first_day_key and len(first_day_statics) < CROSS_DAY_SAMPLE:
                sample = forest_chunk.head(min(50, len(forest_chunk)))
                for _, row in sample.iterrows():
                    key = (round(row["latitude"], 4), round(row["longitude"], 4))
                    if key not in first_day_statics and len(first_day_statics) < CROSS_DAY_SAMPLE:
                        first_day_statics[key] = {v: row[v] for v in static_vars_check}

            elif day_str != first_day_key and first_day_statics:
                for _, row in forest_chunk.iterrows():
                    key = (round(row["latitude"], 4), round(row["longitude"], 4))
                    if key in first_day_statics:
                        for v in static_vars_check:
                            ref_val = first_day_statics[key][v]
                            cur_val = row[v]
                            if pd.notna(ref_val) and pd.notna(cur_val):
                                if abs(ref_val - cur_val) > 0.001:
                                    cross_day_diffs[v] += 1

    elapsed = time.time() - t0
    print("\n  Scan complete: %d rows in %d chunks (%.0fs)" % (total_rows, chunk_num, elapsed))

    # ================================================================
    # 3. DIMENSIONAL INTEGRITY (forest mask aware)
    # ================================================================
    print("\n" + "=" * 72)
    print("3. DIMENSIONAL INTEGRITY (forest mask aware)")
    print("=" * 72)

    n_days = len(day_row_counts)
    is_filtered = total_rows < GRID_CELLS_PER_DAY * n_days * 0.5

    if is_filtered:
        print("  MODE: Forest-filtered output (NaN rows dropped)")
        expected_per_day_min = int(GRID_CELLS_PER_DAY * FOREST_MASK_PCT_MIN / 100)
        expected_per_day_max = int(GRID_CELLS_PER_DAY * FOREST_MASK_PCT_MAX / 100)
        expected_total_min = expected_per_day_min * n_days
        expected_total_max = expected_per_day_max * n_days
        check(
            "Total rows in forest-filtered range [%d, %d]" % (expected_total_min, expected_total_max),
            expected_total_min <= total_rows <= expected_total_max,
            "actual=%d" % total_rows,
        )
    else:
        print("  MODE: Full-grid output (NaN rows retained)")
        expected_full = GRID_CELLS_PER_DAY * n_days
        check(
            "Total rows = %d (%d days x %d cells)" % (expected_full, n_days, GRID_CELLS_PER_DAY),
            total_rows == expected_full,
            "actual=%d" % total_rows,
        )

    # Auto-detect expected days from the data
    if day_row_counts:
        sample_day = sorted(day_row_counts.keys())[0]
        year_month = sample_day[:7]  # e.g. "2018-07"
        year = int(year_month[:4])
        month = int(year_month[5:7])
        import calendar
        expected_days_in_month = calendar.monthrange(year, month)[1]
    else:
        expected_days_in_month = 31
        year, month = 2020, 1

    check("%d days present (expected for %04d-%02d)" % (expected_days_in_month, year, month),
          n_days == expected_days_in_month,
          "got %d days" % n_days)

    if day_row_counts:
        rpd_values = list(day_row_counts.values())
        check(
            "Consistent rows/day (all equal)",
            min(rpd_values) == max(rpd_values),
            "min=%d max=%d" % (min(rpd_values), max(rpd_values)),
        )
        for d in range(1, expected_days_in_month + 1):
            expected_day = "%04d-%02d-%02d" % (year, month, d)
            check(
                "Day %s present" % expected_day,
                expected_day in day_row_counts,
            )

    check("File size > 1 GB", fsize_gb > 1.0, "%.2f GB" % fsize_gb)

    # ================================================================
    # 4. FOREST MASK EFFECTIVENESS
    # ================================================================
    print("\n" + "=" * 72)
    print("4. FOREST MASK EFFECTIVENESS")
    print("=" * 72)

    total_pixels = forest_rows + non_forest_rows
    if total_pixels > 0:
        forest_pct = 100.0 * forest_rows / total_pixels
        print("  Forest pixels:     %d (%.1f%%)" % (forest_rows, forest_pct))
        print("  Non-forest pixels: %d (%.1f%%)" % (non_forest_rows, 100 - forest_pct))

        if is_filtered:
            check(
                "Filtered output: >= 95%% forest pixels",
                forest_pct >= 95.0,
                "%.1f%%" % forest_pct,
            )
        else:
            check(
                "Forest mask applied: %.0f-%.0f%% forest pixels" % (FOREST_MASK_PCT_MIN, FOREST_MASK_PCT_MAX),
                FOREST_MASK_PCT_MIN <= forest_pct <= FOREST_MASK_PCT_MAX,
                "%.1f%%" % forest_pct,
            )

        forest_per_day = forest_rows / max(n_days, 1)
        print("  Forest pixels/day: %d" % forest_per_day)

    check(
        "ERA5 NaN consistency (all-NaN or all-valid per row)",
        nan_inconsistent_rows == 0,
        "%d inconsistent rows" % nan_inconsistent_rows,
    )

    # ================================================================
    # 5. PHYSICAL RANGE VALIDATION (forest pixels only)
    # ================================================================
    print("\n" + "=" * 72)
    print("5. PHYSICAL RANGE VALIDATION (forest pixels)")
    print("=" * 72)

    for col, (lo, hi, unit) in RANGE_BOUNDS.items():
        if col not in col_min:
            if col in missing_cols:
                print("  SKIP: %s (column missing)" % col)
            else:
                check("%s has data" % col, False, "no numeric data found")
            continue
        vmin = col_min[col]
        vmax = col_max[col]
        in_range = vmin >= lo and vmax <= hi
        check(
            "%s in [%.1f, %.1f] %s" % (col, lo, hi, unit),
            in_range,
            "actual=[%.4f, %.4f]" % (vmin, vmax),
        )

    # NaN summary
    print("\n  NaN summary (key columns):")
    for col in ERA5_DYNAMIC + [
        "elevation",
        "canopy_height",
        "LAI",
        "mean_annual_temp",
        "mean_annual_precip",
    ]:
        if col in col_nan:
            nan_pct = 100.0 * col_nan[col] / col_count.get(col, 1)
            print("    %s: %.1f%% NaN (%d / %d)" % (col, nan_pct, col_nan[col], col_count.get(col, 0)))

    # Kelvin detector
    if soil_temp_max > -np.inf:
        check(
            "soil_temperature_level_1 is Celsius (max < 100)",
            soil_temp_max < 100,
            "max=%.2f — likely Kelvin if > 200" % soil_temp_max,
        )

    # ================================================================
    # 6. PLACEHOLDER / DEFAULT VALUE DETECTION
    # ================================================================
    print("\n" + "=" * 72)
    print("6. PLACEHOLDER / DEFAULT VALUE DETECTION (forest pixels)")
    print("=" * 72)

    for var, (default_val, reason) in PLACEHOLDER_DEFAULTS.items():
        count = placeholder_counts.get(var, 0)
        if var in missing_cols:
            check(
                "%s present for placeholder check" % var,
                False,
                "column missing entirely",
            )
        elif forest_rows > 0 and count > 0:
            pct = 100.0 * count / forest_rows
            if pct > 10.0:
                check(
                    "%s: not dominated by default (%.1f)" % (var, default_val),
                    False,
                    "%.1f%% of forest pixels = %.1f (%s)" % (pct, default_val, reason),
                )
            elif pct > 1.0:
                warn(
                    "%s: %.1f%% of forest pixels = exact default %.1f" % (var, pct, default_val),
                    reason,
                )
            else:
                check(
                    "%s: default value %.1f rare (%.2f%%)" % (var, default_val, pct),
                    True,
                )
        else:
            check("%s: no exact default values found" % var, True)

    # ================================================================
    # 7. INTERNAL CONSISTENCY
    # ================================================================
    print("\n" + "=" * 72)
    print("7. INTERNAL CONSISTENCY")
    print("=" * 72)

    check(
        "ta_min <= ta <= ta_max (tolerance 0.015)",
        ta_order_violations == 0,
        "%d violations" % ta_order_violations,
    )

    check(
        "vpd_min <= vpd_max",
        vpd_order_violations == 0,
        "%d violations" % vpd_order_violations,
    )

    if ws_order_violations > 0:
        ws_pct = 100.0 * ws_order_violations / max(forest_rows, 1)
        warn(
            "ws_min > ws_max (component-based approximation)",
            "%d rows (%.1f%%) — inherent limitation of sqrt(u^2+v^2)"
            " from independent min/max" % (ws_order_violations, ws_pct),
        )

    # ================================================================
    # 8. PFT ENCODING
    # ================================================================
    print("\n" + "=" * 72)
    print("8. PFT ENCODING (forest pixels)")
    print("=" * 72)

    if forest_rows > 0:
        check(
            "PFT columns are binary {0, 1}",
            pft_non_binary == 0,
            "%d non-binary rows" % pft_non_binary,
        )

        check(
            "No PFT sum > 1 (mutual exclusivity)",
            pft_sum_multi == 0,
            "%d rows with sum > 1" % pft_sum_multi,
        )

        if is_filtered:
            check(
                "All forest rows have PFT sum == 1",
                pft_sum_zero == 0,
                "%d rows with sum == 0 (%.1f%%)" % (pft_sum_zero, 100.0 * pft_sum_zero / forest_rows),
            )
        else:
            no_pft_pct = 100.0 * forest_no_pft / max(forest_rows, 1)
            if no_pft_pct > 50:
                check(
                    "Forest pixels with PFT class assigned (< 50%% missing)",
                    False,
                    "%.1f%% of forest pixels have no PFT class" % no_pft_pct,
                )
            elif no_pft_pct > 5:
                warn(
                    "Forest pixels without PFT class",
                    "%.1f%% (%d rows) — forest mask may include non-target PFT codes" % (no_pft_pct, forest_no_pft),
                )
            else:
                check(
                    "Forest pixels mostly have PFT assigned (%.1f%% missing)" % no_pft_pct,
                    True,
                )

        classes_present = sum(1 for c in PFT_COLS if pft_class_counts[c] > 0)
        check(
            "At least 4 PFT classes present",
            classes_present >= 4,
            "%d classes" % classes_present,
        )

        print("  PFT distribution (forest pixels):")
        for c in PFT_COLS:
            n = pft_class_counts[c]
            pct = 100.0 * n / max(forest_rows, 1)
            print("    %s: %d (%.1f%%)" % (c, n, pct))

    if forest_rows > 0:
        lai_nan_pct = 100.0 * forest_lai_nan / forest_rows
        check(
            "Forest LAI NaN < 5%%",
            lai_nan_pct < 5.0,
            "%.1f%% (%d rows)" % (lai_nan_pct, forest_lai_nan),
        )

    # ================================================================
    # 9. DERIVED VARIABLE FORMULAS
    # ================================================================
    print("\n" + "=" * 72)
    print("9. DERIVED VARIABLE FORMULAS")
    print("=" * 72)

    if ppfd_err_count > 0:
        ppfd_mae = ppfd_err_sum / ppfd_err_count
        check(
            "PPFD = sw_in * 0.45 * 4.6 (max_err < 0.01)",
            ppfd_max_err < 0.01,
            "max_err=%.6f" % ppfd_max_err,
        )
        check(
            "PPFD mean absolute error < 0.001",
            ppfd_mae < 0.001,
            "mae=%.6f" % ppfd_mae,
        )

    if south_daylen_n > 0 and north_daylen_n > 0:
        south_mean = south_daylen_sum / south_daylen_n
        north_mean = north_daylen_sum / north_daylen_n
        # In NH winter (Oct-Mar): south > north; in NH summer (Apr-Sep): north > south
        if month <= 3 or month >= 10:
            check("NH winter: south day_length > north",
                  south_mean > north_mean,
                  "south=%.2fh north=%.2fh" % (south_mean, north_mean))
        else:
            check("NH summer: north day_length > south",
                  north_mean > south_mean,
                  "north=%.2fh south=%.2fh" % (north_mean, south_mean))

    if year_sin_min < np.inf:
        # Year sin = sin(2*pi*doy/365.25); validate range is plausible for the month
        import math
        mid_doy = sum(calendar.monthrange(year, m)[1] for m in range(1, month)) + 15
        expected_ys = math.sin(2 * math.pi * mid_doy / 365.25)
        # Allow +/- 0.2 around expected mid-month value
        check("Year sin range plausible for month %d" % month,
              year_sin_min > expected_ys - 0.3 and year_sin_max < expected_ys + 0.3,
              "range=[%.4f, %.4f] expected~%.4f" % (year_sin_min, year_sin_max, expected_ys))

    # ================================================================
    # 10. SPATIAL COVERAGE
    # ================================================================
    print("\n" + "=" * 72)
    print("10. SPATIAL COVERAGE")
    print("=" * 72)

    check(
        "Latitude range covers [-60, 78]",
        lat_min_global < -55 and lat_max_global > 74,
        "[%.2f, %.2f]" % (lat_min_global, lat_max_global),
    )
    check(
        "Longitude range covers [-180, 180]",
        lon_min_global < -179 and lon_max_global > 179,
        "[%.2f, %.2f]" % (lon_min_global, lon_max_global),
    )

    for rname in FOREST_REGIONS:
        n = region_counts[rname]
        cells_per_day = n / max(n_days, 1)
        check(
            "%s has forest data (> 100 cells/day)" % rname,
            cells_per_day > 100,
            "%d total / %d per day" % (n, int(cells_per_day)),
        )

    # ================================================================
    # 11. CROSS-DAY CONSISTENCY
    # ================================================================
    print("\n" + "=" * 72)
    print("11. CROSS-DAY CONSISTENCY")
    print("=" * 72)

    for var, diff_count in cross_day_diffs.items():
        if var in missing_cols:
            print("  SKIP: %s (column missing)" % var)
            continue
        check(
            "%s identical across all days" % var,
            diff_count == 0,
            "%d mismatches in %d sampled coords" % (diff_count, len(first_day_statics)),
        )

    print(
        "  INFO: Dynamic variation via Year sin range: [%.4f, %.4f]"
        % (
            year_sin_min if year_sin_min < np.inf else 0,
            year_sin_max if year_sin_max > -np.inf else 0,
        )
    )

    # ================================================================
    # 12. SPATIAL PATTERNS (latitude-binned)
    # ================================================================
    print("\n" + "=" * 72)
    print("12. SPATIAL PATTERNS")
    print("=" * 72)

    if tropical_ta_n > 0 and boreal_ta_n > 0:
        trop_ta = tropical_ta_sum / tropical_ta_n
        bor_ta = boreal_ta_sum / boreal_ta_n
        check("Tropics warmer than boreal (ta)", trop_ta > bor_ta,
              "tropical=%.1f boreal=%.1f" % (trop_ta, bor_ta))

    if tropical_ch_n > 0 and boreal_ch_n > 0:
        trop_ch = tropical_ch_sum / tropical_ch_n
        bor_ch = boreal_ch_sum / boreal_ch_n
        check("Tropical canopy taller than boreal", trop_ch > bor_ch,
              "tropical=%.1f boreal=%.1f m" % (trop_ch, bor_ch))

    if tropical_lai_n > 0 and boreal_lai_n > 0:
        trop_lai = tropical_lai_sum / tropical_lai_n
        bor_lai = boreal_lai_sum / boreal_lai_n
        check("Tropical LAI higher than boreal", trop_lai > bor_lai,
              "tropical=%.2f boreal=%.2f" % (trop_lai, bor_lai))

    if equatorial_daylen_n > 0:
        eq_dl = equatorial_daylen_sum / equatorial_daylen_n
        check("Equatorial day_length near 12h (11-13)", 11.0 < eq_dl < 13.0,
              "mean=%.2fh" % eq_dl)

    # ================================================================
    # 13. CROSS-VARIABLE CORRELATIONS
    # ================================================================
    print("\n" + "=" * 72)
    print("13. CROSS-VARIABLE CORRELATIONS")
    print("=" * 72)

    def safe_corr(a, b):
        a = np.array(a[:CORR_SAMPLE_MAX], dtype=float)
        b = np.array(b[:CORR_SAMPLE_MAX], dtype=float)
        n = min(len(a), len(b))
        if n < 100:
            return None
        a, b = a[:n], b[:n]
        valid = np.isfinite(a) & np.isfinite(b)
        a, b = a[valid], b[valid]
        if len(a) < 100 or a.std() == 0 or b.std() == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    r = safe_corr(corr_samples["elev"], corr_samples["ta"])
    if r is not None:
        print("    elevation vs ta (global): r=%.4f (confounded by latitude)" % r)

    # Lapse rate within tropical band (|lat| < 23) to control for latitude
    trop_idx = [i for i in range(min(len(corr_samples["lat"]), len(corr_samples["elev"])))
                if abs(corr_samples["lat"][i]) < 23]
    if len(trop_idx) > 100:
        trop_elev = [corr_samples["elev"][i] for i in trop_idx]
        trop_ta = [corr_samples["ta"][i] for i in trop_idx]
        r_trop = safe_corr(trop_elev, trop_ta)
        if r_trop is not None:
            check("elevation vs ta (tropical, lapse rate)", r_trop < -0.3,
                  "r=%.4f (expect < -0.3)" % r_trop)
            print("    elevation vs ta (tropical |lat|<23): r=%.4f" % r_trop)

    r = safe_corr(corr_samples["lat"], corr_samples["mat"])
    if r is not None:
        check("latitude vs mean_annual_temp: negative (poles colder)", r < -0.3, "r=%.4f" % r)
        print("    latitude vs mean_annual_temp: r=%.4f" % r)

    r = safe_corr(corr_samples["map_"], corr_samples["lai"])
    if r is not None:
        check("mean_annual_precip vs LAI: positive (wetter=more leaf)", r > 0.1, "r=%.4f" % r)
        print("    mean_annual_precip vs LAI: r=%.4f" % r)

    r = safe_corr(corr_samples["sw_in"], corr_samples["ext_rad"])
    if r is not None:
        check("sw_in vs ext_rad: positive (r > 0.5)", r > 0.5, "r=%.4f" % r)
        print("    sw_in vs ext_rad: r=%.4f" % r)

    r = safe_corr(corr_samples["ta"], corr_samples["soil_temp"])
    if r is not None:
        check("ta vs soil_temp: positive (r > 0.5)", r > 0.5, "r=%.4f" % r)
        print("    ta vs soil_temp: r=%.4f" % r)

    # ================================================================
    # 14. RADIATION CONSTRAINT
    # ================================================================
    print("\n" + "=" * 72)
    print("14. RADIATION CONSTRAINT")
    print("=" * 72)

    check("sw_in <= ext_rad + 0.5 (atmosphere attenuates)",
          sw_gt_ext_violations == 0, "%d violations" % sw_gt_ext_violations)

    # ================================================================
    # 15. VARIABLE DIVERSITY (no constant/fill columns)
    # ================================================================
    print("\n" + "=" * 72)
    print("15. VARIABLE DIVERSITY (spatial std > 0)")
    print("=" * 72)

    for col in ["ta", "vpd", "sw_in", "elevation", "LAI", "day_length",
                "canopy_height", "mean_annual_temp", "mean_annual_precip"]:
        if col in col_sum and col in col_count and col_count[col] > 100:
            n_valid = col_count[col] - col_nan.get(col, 0)
            if n_valid > 100:
                mean = col_sum[col] / n_valid
                var = col_sum_sq[col] / n_valid - mean ** 2
                std = np.sqrt(max(var, 0))
                check("%s has spatial diversity (std > 0.1)" % col,
                      std > 0.1, "std=%.4f" % std)

    # ================================================================
    # 16. DUPLICATE COORDINATE CHECK
    # ================================================================
    print("\n" + "=" * 72)
    print("16. DUPLICATE COORDINATE CHECK (day 1)")
    print("=" * 72)

    check("No duplicate (lat,lon) on day 1",
          first_day_dup_count == 0, "%d duplicates" % first_day_dup_count)

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 72)
    total = PASS_COUNT + FAIL_COUNT
    pct = 100.0 * PASS_COUNT / total if total > 0 else 0
    print("RESULTS: %d/%d passed (%.1f%%), %d failed, %d warnings" % (PASS_COUNT, total, pct, FAIL_COUNT, WARN_COUNT))
    if FAIL_COUNT == 0:
        print("STATUS: ALL TESTS PASSED")
    else:
        print("STATUS: %d TESTS FAILED" % FAIL_COUNT)
    print("=" * 72)

    sys.exit(0 if FAIL_COUNT == 0 else 1)


if __name__ == "__main__":
    main()
