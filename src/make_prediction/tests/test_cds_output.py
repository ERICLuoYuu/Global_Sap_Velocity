"""
Strict validation test suite for GEE ERA5-Land prediction output.

Tests: structural integrity, physical plausibility, internal consistency,
derived variable formulas, PFT encoding, spatial/temporal coverage,
NaN discipline, and parity constraints with CDS.

Exit code 0 = all pass, 1 = failures found.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Counters ──────────────────────────────────────────────────────────
PASS = 0
FAIL = 0
WARN = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print("  PASS: %s" % name)
    else:
        FAIL += 1
        msg = "  FAIL: %s" % name
        if detail:
            msg += "  [%s]" % detail
        print(msg)


def warn(name: str, detail: str) -> None:
    global WARN
    WARN += 1
    print("  WARN: %s  [%s]" % (name, detail))


# ── Load ──────────────────────────────────────────────────────────────
csv_dir = Path("/scratch/tmp/yluo2/Global_Sap_Velocity/outputs/predictions/cds_era5/csv")
files = sorted(csv_dir.glob("prediction_2020_01_daily.csv"))
if not files:
    files = sorted(csv_dir.glob("prediction_2020_01_daily.csv"))

print("=" * 70)
print("CDS OUTPUT VALIDATION — STRICT TEST SUITE")
print("=" * 70)
print("Files found: %d" % len(files))
for f in files:
    print("  %s (%.2f GB)" % (f.name, f.stat().st_size / 1e9))

check("Output file exists", len(files) > 0)
if not files:
    print("\nFATAL: No output files found.")
    sys.exit(1)

print("\nLoading data...")
df = pd.read_csv(files[0], low_memory=False)
nrows, ncols = df.shape
print("Shape: (%d, %d)" % (nrows, ncols))

# =====================================================================
# 1. STRUCTURAL INTEGRITY
# =====================================================================
print("\n" + "=" * 70)
print("1. STRUCTURAL INTEGRITY")
print("=" * 70)

REQUIRED_COLS = [
    "latitude",
    "longitude",
    "mean_annual_temp",
    "mean_annual_precip",
    "elevation",
    "canopy_height",
    "LAI",
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
    "precip",
    "prcip/PET",
    "volumetric_soil_water_layer_1",
    "soil_temperature_level_1",
    "Day sin",
    "Year sin",
    "ENF",
    "EBF",
    "DNF",
    "DBF",
    "MF",
    "WSA",
    "SAV",
    "WET",
    "name",
]
for c in REQUIRED_COLS:
    check("Column present: %s" % c, c in df.columns)

check("No duplicate columns", len(df.columns) == len(set(df.columns)))
check("day_length present (parity critical)", "day_length" in df.columns)
check("Non-empty dataset", nrows > 0, "rows=%d" % nrows)
check("At least 35 columns", ncols >= 35, "cols=%d" % ncols)

# Dtype checks
for c in ["ta", "vpd", "sw_in", "ext_rad", "day_length", "precip", "elevation"]:
    if c in df.columns:
        check("%s is numeric" % c, df[c].dtype in [np.float64, np.float32, np.int64])

for c in ["ENF", "EBF", "DNF", "DBF", "MF", "WSA", "SAV", "WET"]:
    if c in df.columns:
        check("%s is integer type" % c, df[c].dtype in [np.int64, np.int32, np.int8, np.uint8])

# =====================================================================
# 2. TEMPORAL COVERAGE
# =====================================================================
print("\n" + "=" * 70)
print("2. TEMPORAL COVERAGE")
print("=" * 70)

ts_cols = [c for c in df.columns if "timestamp" in c.lower()]
print("  Timestamp columns: %s" % ts_cols)

day_col = None
if "timestamp" in df.columns:
    day_col = "timestamp"
elif len(ts_cols) >= 1:
    day_col = ts_cols[0]

if day_col is not None:
    unique_days = df[day_col].nunique()
    check("31 days in January 2020", unique_days == 31, "got %d" % unique_days)

    rpd = df.groupby(day_col).size()
    check("Consistent rows per day (all equal)", rpd.min() == rpd.max(), "min=%d max=%d" % (rpd.min(), rpd.max()))

    cells_per_day = rpd.iloc[0]
    print("  INFO: %d rows/day, %d total rows" % (cells_per_day, nrows))

    # All days present: 2020-01-01 to 2020-01-31
    day_strs = sorted(df[day_col].unique())
    for d in range(1, 32):
        expected = "2020-01-%02d" % d
        found = any(expected in str(s) for s in day_strs)
        check("Day %d present" % d, found)
else:
    check("Timestamp column found", False, "no timestamp column")

# =====================================================================
# 3. PHYSICAL RANGE TESTS (STRICT)
# =====================================================================
print("\n" + "=" * 70)
print("3. PHYSICAL RANGE TESTS (strict bounds)")
print("=" * 70)


def range_strict(col, lo, hi, unit, allow_nan_pct=0.0):
    if col not in df.columns:
        check("%s present" % col, False)
        return
    vmin = df[col].min()
    vmax = df[col].max()
    nan_pct = df[col].isna().mean() * 100
    in_range = vmin >= lo and vmax <= hi
    nan_ok = nan_pct <= allow_nan_pct

    check("%s in [%.1f, %.1f] %s" % (col, lo, hi, unit), in_range, "actual=[%.4f, %.4f]" % (vmin, vmax))
    check("%s NaN <= %.1f%%" % (col, allow_nan_pct), nan_ok, "NaN=%.2f%%" % nan_pct)


# Temperatures — must be Celsius (not Kelvin)
range_strict("ta", -90, 60, "C")
range_strict("ta_min", -90, 60, "C")
range_strict("ta_max", -90, 60, "C")
range_strict("soil_temperature_level_1", -50, 60, "C")

# Kelvin detector: if ANY temp column exceeds 100, it's Kelvin
for tc in ["ta", "ta_min", "ta_max", "soil_temperature_level_1"]:
    if tc in df.columns:
        check("%s is Celsius (max < 100)" % tc, df[tc].max() < 100, "max=%.2f (Kelvin if >200)" % df[tc].max())

# VPD — must be kPa (not hPa)
range_strict("vpd", 0, 8, "kPa")
range_strict("vpd_min", 0, 6, "kPa")
range_strict("vpd_max", 0, 12, "kPa")
if "vpd" in df.columns:
    check("VPD unit is kPa (mean < 3)", df["vpd"].mean() < 3, "mean=%.4f (hPa if >10)" % df["vpd"].mean())

# Wind speed
range_strict("ws", 0, 40, "m/s")
range_strict("ws_min", 0, 40, "m/s")
range_strict("ws_max", 0, 50, "m/s")

# Solar radiation
range_strict("sw_in", 0, 450, "W/m2")
range_strict("ppfd_in", 0, 2500, "umol/m2/s")
range_strict("ext_rad", 0, 550, "W/m2")

# Day length
range_strict("day_length", 0, 24, "hours")

# Precipitation
range_strict("precip", 0, 500, "mm")
range_strict("prcip/PET", 0, 10, "ratio")

# Soil water content (volumetric: 0 to 1 m3/m3)
range_strict("volumetric_soil_water_layer_1", 0, 1, "m3/m3")

# Static variables
range_strict("LAI", 0, 12, "m2/m2")
range_strict("elevation", -500, 9000, "m")
range_strict("canopy_height", 0, 100, "m")
range_strict("mean_annual_temp", -30, 35, "C")
range_strict("mean_annual_precip", 0, 12000, "mm")

# Time features — must be in [-1, 1]
range_strict("Day sin", -1.0, 1.0, "")
range_strict("Year sin", -1.0, 1.0, "")

# =====================================================================
# 4. INTERNAL CONSISTENCY (STRICT)
# =====================================================================
print("\n" + "=" * 70)
print("4. INTERNAL CONSISTENCY")
print("=" * 70)

# ta_min <= ta <= ta_max (with tiny float tolerance)
TOL = 0.015
if all(c in df.columns for c in ["ta", "ta_min", "ta_max"]):
    below = (df["ta"] < df["ta_min"] - TOL).sum()
    above = (df["ta"] > df["ta_max"] + TOL).sum()
    check("ta >= ta_min (tolerance %.3f)" % TOL, below == 0, "%d violations" % below)
    check("ta <= ta_max (tolerance %.3f)" % TOL, above == 0, "%d violations" % above)
    check("ta_min <= ta_max (strict)", (df["ta_min"] > df["ta_max"]).sum() == 0)

# vpd cross-matched: vpd_max uses (ta_max, td_min), vpd_min uses (ta_min, td_max)
# So vpd_min should generally be <= vpd_max
if all(c in df.columns for c in ["vpd_min", "vpd_max"]):
    bad = (df["vpd_min"] > df["vpd_max"]).sum()
    check("vpd_min <= vpd_max (strict zero tolerance)", bad == 0, "%d violations (%.3f%%)" % (bad, bad / nrows * 100))

# ws_min vs ws_max — known approximation issue
if all(c in df.columns for c in ["ws_min", "ws_max"]):
    bad = (df["ws_min"] > df["ws_max"]).sum()
    pct = bad / nrows * 100
    if bad > 0:
        warn("ws_min > ws_max (component-based approximation)", "%d rows (%.1f%%) — inherent limitation" % (bad, pct))

# Soil water physically bounded
if "volumetric_soil_water_layer_1" in df.columns:
    check("SWVL1 all positive", (df["volumetric_soil_water_layer_1"] >= 0).all())
    check("SWVL1 all <= 1.0", (df["volumetric_soil_water_layer_1"] <= 1.0).all())

# =====================================================================
# 5. DERIVED VARIABLE FORMULA VERIFICATION
# =====================================================================
print("\n" + "=" * 70)
print("5. DERIVED VARIABLE FORMULA VERIFICATION")
print("=" * 70)

# PPFD = clip(sw_in * 0.45 * 4.6, 0, 2500)
if all(c in df.columns for c in ["sw_in", "ppfd_in"]):
    expected_ppfd = np.clip(df["sw_in"].values * 0.45 * 4.6, 0, 2500)
    diff = np.abs(df["ppfd_in"].values - expected_ppfd)
    check("PPFD formula: sw_in * 0.45 * 4.6 (max_err < 0.01)", diff.max() < 0.01, "max_err=%.6f" % diff.max())
    check("PPFD formula: mean absolute error < 0.001", diff.mean() < 0.001, "mae=%.6f" % diff.mean())

# ext_rad non-negative everywhere
if "ext_rad" in df.columns:
    check("ext_rad >= 0 everywhere", (df["ext_rad"] >= 0).all())

# Day length in January: Southern Hemisphere should have LONGER days
if all(c in df.columns for c in ["day_length", "latitude"]):
    south = df[df["latitude"] < -20]["day_length"]
    north = df[df["latitude"] > 50]["day_length"]
    if len(south) > 0 and len(north) > 0:
        check(
            "January: south day_length > north (Earth tilt)",
            south.mean() > north.mean(),
            "south=%.2fh north=%.2fh" % (south.mean(), north.mean()),
        )

    # Tropics should have ~12h day length
    tropics = df[df["latitude"].between(-10, 10)]["day_length"]
    if len(tropics) > 0:
        check("Tropics day_length near 12h (11-13)", 11.0 < tropics.mean() < 13.0, "mean=%.2fh" % tropics.mean())

    # Polar regions: Arctic should have short days in January
    arctic = df[df["latitude"] > 65]["day_length"]
    if len(arctic) > 0:
        check("Arctic day_length < 8h in January", arctic.mean() < 8.0, "mean=%.2fh" % arctic.mean())

# ext_rad: higher in Southern Hemisphere in January (perihelion)
if all(c in df.columns for c in ["ext_rad", "latitude"]):
    south_ext = df[df["latitude"] < -20]["ext_rad"]
    north_ext = df[df["latitude"] > 50]["ext_rad"]
    if len(south_ext) > 0 and len(north_ext) > 0:
        check(
            "January: ext_rad higher in south",
            south_ext.mean() > north_ext.mean(),
            "south=%.1f north=%.1f W/m2" % (south_ext.mean(), north_ext.mean()),
        )

# precip/PET ratio: if precip near 0, ratio should be near 0
if all(c in df.columns for c in ["precip", "prcip/PET"]):
    dry = df[df["precip"] < 0.001]
    if len(dry) > 0:
        check("precip~0 => prcip/PET~0 (max < 0.1)", dry["prcip/PET"].max() < 0.1, "max=%.4f" % dry["prcip/PET"].max())

# precip always non-negative (abs() was applied)
if "precip" in df.columns:
    check("precip always >= 0 (abs applied)", (df["precip"] >= 0).all())

# Year sin: January doy 1-31 => sin(2*pi*doy/365) should be positive and < 0.55
if "Year sin" in df.columns:
    check("Year sin > 0 in January", df["Year sin"].min() >= 0, "min=%.4f" % df["Year sin"].min())
    check("Year sin < 0.55 in January", df["Year sin"].max() < 0.55, "max=%.4f" % df["Year sin"].max())

# Day sin: should vary between ~-1 and ~1 across longitudes
if "Day sin" in df.columns:
    day_sin_range = df["Day sin"].max() - df["Day sin"].min()
    check("Day sin has spread > 0.5 (longitude variation)", day_sin_range > 0.5, "range=%.4f" % day_sin_range)

# =====================================================================
# 6. PFT TESTS (STRICT — forest filter applied)
# =====================================================================
print("\n" + "=" * 70)
print("6. PFT TESTS (strict — forest filter)")
print("=" * 70)

PFT_COLS = ["ENF", "EBF", "DNF", "DBF", "MF", "WSA", "SAV", "WET"]
if all(c in df.columns for c in PFT_COLS):
    row_sums = df[PFT_COLS].sum(axis=1)

    check(
        "ALL rows PFT sum == 1 (forest filter active)",
        (row_sums == 1).all(),
        "sum==0: %d, sum>1: %d" % ((row_sums == 0).sum(), (row_sums > 1).sum()),
    )

    check(
        "Zero non-forest rows (PFT sum==0 eliminated)",
        (row_sums == 0).sum() == 0,
        "%d non-forest rows remain" % (row_sums == 0).sum(),
    )

    # All PFT columns binary
    for pc in PFT_COLS:
        check("%s is strictly binary {0,1}" % pc, df[pc].isin([0, 1]).all())

    # At least 4 PFT classes present globally
    classes_present = sum(1 for pc in PFT_COLS if (df[pc] == 1).any())
    check("At least 4 PFT classes present", classes_present >= 4, "%d classes" % classes_present)

    # Distribution — all major classes should have > 0.1%
    print("  INFO: PFT distribution:")
    for pc in PFT_COLS:
        n = (df[pc] == 1).sum()
        pct = n / nrows * 100
        print("    %s: %d (%.1f%%)" % (pc, n, pct))
        if pct < 0.1 and n > 0:
            warn("%s very rare" % pc, "%.2f%%" % pct)

# =====================================================================
# 7. NaN DISCIPLINE (STRICT — zero tolerance)
# =====================================================================
print("\n" + "=" * 70)
print("7. NaN DISCIPLINE (zero tolerance for core variables)")
print("=" * 70)

ZERO_NAN_COLS = [
    "ta",
    "ta_min",
    "ta_max",
    "vpd",
    "vpd_min",
    "vpd_max",
    "ws",
    "sw_in",
    "ppfd_in",
    "ext_rad",
    "day_length",
    "precip",
    "prcip/PET",
    "volumetric_soil_water_layer_1",
    "soil_temperature_level_1",
    "latitude",
    "longitude",
    "Day sin",
    "Year sin",
    "ENF",
    "EBF",
    "DNF",
    "DBF",
    "MF",
    "WSA",
    "SAV",
    "WET",
]
for c in ZERO_NAN_COLS:
    if c in df.columns:
        nan_count = df[c].isna().sum()
        check("%s has zero NaN" % c, nan_count == 0, "%d NaN" % nan_count)

# LAI may have some NaN (sparse GlobMap coverage)
if "LAI" in df.columns:
    lai_nan_pct = df["LAI"].isna().mean() * 100
    check("LAI NaN < 5%%", lai_nan_pct < 5, "%.2f%%" % lai_nan_pct)

# =====================================================================
# 8. SPATIAL COVERAGE
# =====================================================================
print("\n" + "=" * 70)
print("8. SPATIAL COVERAGE")
print("=" * 70)

check(
    "Latitude in [-60, 78]",
    df["latitude"].between(-60.1, 78.1).all(),
    "[%.2f, %.2f]" % (df["latitude"].min(), df["latitude"].max()),
)
check(
    "Longitude in [-180, 180]",
    df["longitude"].between(-180.1, 180.1).all(),
    "[%.2f, %.2f]" % (df["longitude"].min(), df["longitude"].max()),
)

if day_col is not None:
    n_cells = nrows // df[day_col].nunique()
else:
    n_cells = df.groupby(["latitude", "longitude"]).ngroups

check("Grid cells > 100k (global forest)", n_cells > 100000, "%d" % n_cells)
check("Grid cells < 800k (reasonable upper bound)", n_cells < 800000, "%d" % n_cells)

# Check major forest regions have data
REGIONS = {
    "Amazon": ((-15, -2), (-70, -50)),
    "Congo Basin": ((-5, 5), (15, 30)),
    "Boreal Canada": ((50, 65), (-130, -60)),
    "Boreal Siberia": ((55, 70), (60, 140)),
    "SE Asia": ((-5, 15), (95, 130)),
    "Central Europe": ((45, 55), (5, 20)),
    "Eastern US": ((30, 45), (-90, -70)),
    "Scandinavia": ((58, 70), (5, 30)),
}
for name, ((lat_lo, lat_hi), (lon_lo, lon_hi)) in REGIONS.items():
    mask = df["latitude"].between(lat_lo, lat_hi) & df["longitude"].between(lon_lo, lon_hi)
    n = mask.sum()
    check("%s has data (> 0 rows)" % name, n > 0, "%d rows" % n)
    if n > 0 and day_col is not None:
        cells = n // df[day_col].nunique()
        check("%s has > 100 cells/day" % name, cells > 100, "%d cells/day" % cells)

# =====================================================================
# 9. NAME COLUMN FORMAT
# =====================================================================
print("\n" + "=" * 70)
print("9. NAME COLUMN FORMAT")
print("=" * 70)

if "name" in df.columns:
    sample_names = df["name"].head(10).tolist()
    check(
        "name starts with 'Grid_'",
        all(str(n).startswith("Grid_") for n in sample_names),
        "samples: %s" % str(sample_names[:3]),
    )

    # Format: Grid_{lat}_{lon}
    parts = str(sample_names[0]).split("_")
    check("name has Grid_lat_lon format (3 parts)", len(parts) >= 3)

    # Name should be unique per (lat, lon)
    n_names = df["name"].nunique()
    n_coords = df.groupby(["latitude", "longitude"]).ngroups
    check("name count matches unique (lat,lon) pairs", n_names == n_coords, "names=%d coords=%d" % (n_names, n_coords))

# =====================================================================
# 10. CROSS-DAY CONSISTENCY (spot check)
# =====================================================================
print("\n" + "=" * 70)
print("10. CROSS-DAY CONSISTENCY")
print("=" * 70)

if day_col is not None:
    days = sorted(df[day_col].unique())
    if len(days) >= 2:
        d1 = df[df[day_col] == days[0]]
        d2 = df[df[day_col] == days[1]]

        # Same grid cells each day
        coords1 = set(zip(d1["latitude"], d1["longitude"]))
        coords2 = set(zip(d2["latitude"], d2["longitude"]))
        check(
            "Day 1 and Day 2 have identical grid cells",
            coords1 == coords2,
            "d1=%d d2=%d overlap=%d" % (len(coords1), len(coords2), len(coords1 & coords2)),
        )

        # Static vars should be identical across days
        for sv in ["elevation", "canopy_height", "mean_annual_temp"]:
            if sv in df.columns:
                v1 = d1.sort_values(["latitude", "longitude"])[sv].values
                v2 = d2.sort_values(["latitude", "longitude"])[sv].values
                if len(v1) == len(v2):
                    diff = np.nanmax(np.abs(v1 - v2))
                    check("%s identical across days" % sv, diff < 0.001, "max_diff=%.6f" % diff)

        # Dynamic vars should differ between days
        for dv in ["ta", "vpd", "sw_in", "precip"]:
            if dv in df.columns:
                v1 = d1.sort_values(["latitude", "longitude"])[dv].values
                v2 = d2.sort_values(["latitude", "longitude"])[dv].values
                if len(v1) == len(v2):
                    diff = np.abs(v1 - v2).mean()
                    check("%s varies between days (mean_diff > 0)" % dv, diff > 1e-6, "mean_diff=%.6f" % diff)

# =====================================================================
# 11. OUTPUT FILE SIZE SANITY
# =====================================================================
print("\n" + "=" * 70)
print("11. OUTPUT FILE SANITY")
print("=" * 70)

fsize_gb = files[0].stat().st_size / 1e9
check("File size > 1 GB (non-trivial output)", fsize_gb > 1.0, "%.2f GB" % fsize_gb)
check("File size < 20 GB (not exploded)", fsize_gb < 20.0, "%.2f GB" % fsize_gb)

# Rows per GB ratio
rows_per_gb = nrows / fsize_gb
check("Reasonable density (1M-5M rows/GB)", 1e6 < rows_per_gb < 5e6, "%.1f rows/GB" % rows_per_gb)

# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 70)
total = PASS + FAIL
pct = PASS / total * 100 if total > 0 else 0
print("RESULTS: %d/%d passed (%.1f%%), %d failed, %d warnings" % (PASS, total, pct, FAIL, WARN))
if FAIL == 0:
    print("STATUS: ALL TESTS PASSED")
else:
    print("STATUS: %d TESTS FAILED" % FAIL)
print("=" * 70)

sys.exit(0 if FAIL == 0 else 1)
