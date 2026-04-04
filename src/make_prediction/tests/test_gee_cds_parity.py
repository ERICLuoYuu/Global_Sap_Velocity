"""
Cross-comparison test suite: GEE vs CDS ERA5-Land prediction outputs.

Verifies structural parity, spatial overlap, static variable identity,
dynamic variable agreement, and distributional consistency between the
two independent pipelines.

Usage:
    python test_gee_cds_parity.py [gee_csv] [cds_csv]

Exit code 0 = all pass, 1 = failures found.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -- Counters --------------------------------------------------------------
PASS = 0
FAIL = 0
WARN = 0


def check(name, condition, detail=""):
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


def warn(name, detail):
    global WARN
    WARN += 1
    print("  WARN: %s  [%s]" % (name, detail))


# -- Defaults ---------------------------------------------------------------
DEFAULT_GEE = Path("/scratch/tmp/yluo2/gsv-wt/map-viz/outputs/predictions/2020_daily/prediction_2020_01_daily.csv")
DEFAULT_CDS = Path(
    "/scratch/tmp/yluo2/Global_Sap_Velocity/outputs/predictions/cds_era5/csv/prediction_2020_01_daily.csv"
)

gee_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_GEE
cds_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CDS

print("=" * 70)
print("GEE vs CDS CROSS-COMPARISON TEST SUITE")
print("=" * 70)
print("GEE: %s (%.2f GB)" % (gee_path, gee_path.stat().st_size / 1e9))
print("CDS: %s (%.2f GB)" % (cds_path, cds_path.stat().st_size / 1e9))

check("GEE file exists", gee_path.exists())
check("CDS file exists", cds_path.exists())
if not gee_path.exists() or not cds_path.exists():
    print("FATAL: Missing output file.")
    sys.exit(1)

# -- Load -------------------------------------------------------------------
print("\nLoading datasets...\n")

gee_full = pd.read_csv(gee_path, low_memory=False)
cds_full = pd.read_csv(cds_path, low_memory=False)

# Identify timestamp column
gee_ts_col = "timestamp.1" if "timestamp.1" in gee_full.columns else "timestamp"
cds_ts_col = "timestamp"

gee_days = sorted(gee_full[gee_ts_col].unique())
cds_days = sorted(cds_full[cds_ts_col].unique())

gee_day1 = gee_full[gee_full[gee_ts_col] == gee_days[0]].copy()
cds_day1 = cds_full[cds_full[cds_ts_col] == cds_days[0]].copy()

# =====================================================================
# 1. STRUCTURAL PARITY
# =====================================================================
print("=" * 70)
print("1. STRUCTURAL PARITY")
print("=" * 70)

gee_cols = set(gee_full.columns)
cds_cols = set(cds_full.columns)
shared_cols = sorted(gee_cols & cds_cols)
gee_only = sorted(gee_cols - cds_cols)
cds_only = sorted(cds_cols - gee_cols)

print("  GEE columns: %d" % len(gee_cols))
print("  CDS columns: %d" % len(cds_cols))
print("  Shared: %d" % len(shared_cols))
if gee_only:
    print("  GEE only: %s" % gee_only)
if cds_only:
    print("  CDS only: %s" % cds_only)

CORE_COLS = [
    "latitude",
    "longitude",
    "timestamp",
    "ta",
    "ta_min",
    "ta_max",
    "vpd",
    "vpd_min",
    "vpd_max",
    "sw_in",
    "ppfd_in",
    "ext_rad",
    "ws",
    "ws_min",
    "ws_max",
    "day_length",
    "prcip/PET",
    "volumetric_soil_water_layer_1",
    "soil_temperature_level_1",
    "elevation",
    "canopy_height",
    "LAI",
    "mean_annual_temp",
    "mean_annual_precip",
    "solar_timestamp",
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
for c in CORE_COLS:
    check("Core column in both: %s" % c, c in gee_cols and c in cds_cols)

check("Shared columns >= 30", len(shared_cols) >= 30, "%d" % len(shared_cols))

# =====================================================================
# 2. TEMPORAL PARITY
# =====================================================================
print("\n" + "=" * 70)
print("2. TEMPORAL PARITY")
print("=" * 70)

gee_n_days = len(gee_days)
cds_n_days = len(cds_days)
check("Same number of days", gee_n_days == cds_n_days, "GEE=%d CDS=%d" % (gee_n_days, cds_n_days))

gee_day_strs = set(str(d)[:10] for d in gee_days)
cds_day_strs = set(str(d)[:10] for d in cds_days)
day_overlap = gee_day_strs & cds_day_strs
check(
    "All days overlap",
    len(day_overlap) == max(gee_n_days, cds_n_days),
    "overlap=%d GEE=%d CDS=%d" % (len(day_overlap), gee_n_days, cds_n_days),
)

gee_rpd = len(gee_day1)
cds_rpd = len(cds_day1)
print("  GEE rows/day: %d" % gee_rpd)
print("  CDS rows/day: %d" % cds_rpd)
ratio = max(gee_rpd, cds_rpd) / max(min(gee_rpd, cds_rpd), 1)
check("Rows/day ratio < 1.2 (within 20%%)", ratio < 1.2, "ratio=%.3f" % ratio)

# =====================================================================
# 3. SPATIAL OVERLAP
# =====================================================================
print("\n" + "=" * 70)
print("3. SPATIAL OVERLAP (day 1)")
print("=" * 70)

gee_day1["lat_r"] = gee_day1["latitude"].round(1)
gee_day1["lon_r"] = gee_day1["longitude"].round(1)
cds_day1["lat_r"] = cds_day1["latitude"].round(1)
cds_day1["lon_r"] = cds_day1["longitude"].round(1)

gee_coords = set(zip(gee_day1["lat_r"], gee_day1["lon_r"]))
cds_coords = set(zip(cds_day1["lat_r"], cds_day1["lon_r"]))
overlap = gee_coords & cds_coords
gee_exclusive = gee_coords - cds_coords
cds_exclusive = cds_coords - gee_coords

print("  GEE cells: %d" % len(gee_coords))
print("  CDS cells: %d" % len(cds_coords))
print("  Overlap:   %d" % len(overlap))
print("  GEE only:  %d" % len(gee_exclusive))
print("  CDS only:  %d" % len(cds_exclusive))

jaccard = len(overlap) / max(len(gee_coords | cds_coords), 1)
overlap_pct_smaller = len(overlap) / max(min(len(gee_coords), len(cds_coords)), 1) * 100
check("Jaccard similarity > 0.80", jaccard > 0.80, "%.3f" % jaccard)
check(
    "Overlap >= 90%% of smaller set",
    overlap_pct_smaller >= 90,
    "%.1f%%" % overlap_pct_smaller,
)

for coord, label in [("latitude", "Lat"), ("longitude", "Lon")]:
    gmin, gmax = gee_day1[coord].min(), gee_day1[coord].max()
    cmin, cmax = cds_day1[coord].min(), cds_day1[coord].max()
    check(
        "%s range similar (diff < 5 deg)" % label,
        abs(gmin - cmin) < 5 and abs(gmax - cmax) < 5,
        "GEE=[%.1f,%.1f] CDS=[%.1f,%.1f]" % (gmin, gmax, cmin, cmax),
    )

# =====================================================================
# 4. STATIC VARIABLE COMPARISON (shared cells)
# =====================================================================
print("\n" + "=" * 70)
print("4. STATIC VARIABLE COMPARISON (shared cells, day 1)")
print("=" * 70)

merged = pd.merge(
    gee_day1,
    cds_day1,
    on=["lat_r", "lon_r"],
    suffixes=("_gee", "_cds"),
    how="inner",
)
print("  Merged cells: %d" % len(merged))

STATIC_VARS = [
    "elevation",
    "canopy_height",
    "LAI",
    "mean_annual_temp",
    "mean_annual_precip",
]
for var in STATIC_VARS:
    gcol = var + "_gee"
    ccol = var + "_cds"
    if gcol not in merged.columns or ccol not in merged.columns:
        check("%s present in both" % var, False, "column missing in merge")
        continue

    gv = merged[gcol].values.astype(float)
    cv = merged[ccol].values.astype(float)
    valid = np.isfinite(gv) & np.isfinite(cv)
    if valid.sum() == 0:
        check("%s has valid data" % var, False, "all NaN")
        continue

    gv, cv = gv[valid], cv[valid]
    diff = np.abs(gv - cv)
    mae = diff.mean()
    max_diff = diff.max()
    pct_exact = (diff < 0.001).sum() / len(diff) * 100

    if gv.std() > 0 and cv.std() > 0:
        corr = np.corrcoef(gv, cv)[0, 1]
    else:
        corr = float("nan")

    check("%s correlation > 0.90" % var, corr > 0.90 or np.isnan(corr), "r=%.4f" % corr)
    check("%s MAE < 50" % var, mae < 50, "MAE=%.4f max=%.4f" % (mae, max_diff))
    print("    %s: MAE=%.4f, max_diff=%.4f, exact=%.1f%%, r=%.4f" % (var, mae, max_diff, pct_exact, corr))

# =====================================================================
# 5. DYNAMIC VARIABLE COMPARISON (shared cells, day 1)
# =====================================================================
print("\n" + "=" * 70)
print("5. DYNAMIC VARIABLE COMPARISON (shared cells, day 1)")
print("=" * 70)

DYNAMIC_VARS = [
    ("ta", 2.0, "degC"),
    ("ta_min", 2.0, "degC"),
    ("ta_max", 2.0, "degC"),
    ("vpd", 0.5, "kPa"),
    ("vpd_min", 0.5, "kPa"),
    ("vpd_max", 0.5, "kPa"),
    ("sw_in", 30.0, "W/m2"),
    ("ppfd_in", 100.0, "umol/m2/s"),
    ("ext_rad", 10.0, "W/m2"),
    ("ws", 2.0, "m/s"),
    ("day_length", 0.5, "hours"),
    ("prcip/PET", 1.0, "ratio"),
    ("volumetric_soil_water_layer_1", 0.1, "m3/m3"),
    ("soil_temperature_level_1", 5.0, "degC"),
]

for var, tol, unit in DYNAMIC_VARS:
    gcol = var + "_gee"
    ccol = var + "_cds"
    if gcol not in merged.columns or ccol not in merged.columns:
        check("%s present in both" % var, False, "column missing")
        continue

    gv = merged[gcol].values.astype(float)
    cv = merged[ccol].values.astype(float)
    valid = np.isfinite(gv) & np.isfinite(cv)
    if valid.sum() == 0:
        check("%s has valid data" % var, False)
        continue

    gv, cv = gv[valid], cv[valid]
    diff = np.abs(gv - cv)
    mae = diff.mean()
    max_diff = diff.max()

    if gv.std() > 0 and cv.std() > 0:
        corr = np.corrcoef(gv, cv)[0, 1]
    else:
        corr = 1.0

    check("%s correlation > 0.95" % var, corr > 0.95, "r=%.4f" % corr)
    check("%s MAE < %.1f %s" % (var, tol, unit), mae < tol, "MAE=%.4f" % mae)
    print("    %s: MAE=%.4f %s, max=%.4f, r=%.4f" % (var, mae, unit, max_diff, corr))

# =====================================================================
# 6. DERIVED VARIABLE PARITY
# =====================================================================
print("\n" + "=" * 70)
print("6. DERIVED VARIABLE PARITY (shared cells, day 1)")
print("=" * 70)

for feat in ["Day sin", "Year sin"]:
    gcol = feat + "_gee"
    ccol = feat + "_cds"
    if gcol in merged.columns and ccol in merged.columns:
        gv = merged[gcol].values.astype(float)
        cv = merged[ccol].values.astype(float)
        valid = np.isfinite(gv) & np.isfinite(cv)
        gv, cv = gv[valid], cv[valid]
        mae = np.abs(gv - cv).mean()
        corr = np.corrcoef(gv, cv)[0, 1] if len(gv) > 1 else 1.0
        check("%s MAE < 0.05" % feat, mae < 0.05, "MAE=%.6f" % mae)
        check("%s correlation > 0.99" % feat, corr > 0.99, "r=%.6f" % corr)
        print("    %s: MAE=%.6f, r=%.6f" % (feat, mae, corr))

# PPFD formula check
if "ppfd_in_gee" in merged.columns and "ppfd_in_cds" in merged.columns:
    gp = merged["ppfd_in_gee"].values.astype(float)
    cp = merged["ppfd_in_cds"].values.astype(float)
    gs = merged["sw_in_gee"].values.astype(float)
    cs = merged["sw_in_cds"].values.astype(float)
    valid = np.isfinite(gp) & np.isfinite(cp)
    if valid.sum() > 0:
        gp_exp = np.clip(gs[valid] * 0.45 * 4.6, 0, 2500)
        cp_exp = np.clip(cs[valid] * 0.45 * 4.6, 0, 2500)
        check("GEE PPFD follows formula", np.abs(gp[valid] - gp_exp).max() < 0.01)
        check("CDS PPFD follows formula", np.abs(cp[valid] - cp_exp).max() < 0.01)
    else:
        check("PPFD has valid shared data", False, "no overlap")

# =====================================================================
# 7. PFT DISTRIBUTION COMPARISON
# =====================================================================
print("\n" + "=" * 70)
print("7. PFT DISTRIBUTION COMPARISON (day 1)")
print("=" * 70)

PFT_COLS = ["ENF", "EBF", "DNF", "DBF", "MF", "WSA", "SAV", "WET"]

print("  %-4s  %10s (%%)   %10s (%%)   diff" % ("PFT", "GEE", "CDS"))
print("  " + "-" * 55)
max_pft_diff = 0
for pc in PFT_COLS:
    if pc in gee_day1.columns and pc in cds_day1.columns:
        gn = (gee_day1[pc] == 1).sum()
        cn = (cds_day1[pc] == 1).sum()
        gpct = 100 * gn / max(len(gee_day1), 1)
        cpct = 100 * cn / max(len(cds_day1), 1)
        diff_pct = abs(gpct - cpct)
        max_pft_diff = max(max_pft_diff, diff_pct)
        print("  %-4s  %7d (%5.1f)   %7d (%5.1f)   %.1f pp" % (pc, gn, gpct, cn, cpct, diff_pct))

check("PFT distribution diff < 5 pp for all classes", max_pft_diff < 5, "max=%.1f pp" % max_pft_diff)

# PFT agreement on shared cells
if len(merged) > 0:
    pft_agree = 0
    pft_total = 0
    for pc in PFT_COLS:
        gc = pc + "_gee"
        cc = pc + "_cds"
        if gc in merged.columns and cc in merged.columns:
            pft_agree += (merged[gc] == merged[cc]).sum()
            pft_total += len(merged)
    if pft_total > 0:
        pft_acc = 100 * pft_agree / pft_total
        check("PFT agreement on shared cells > 95%%", pft_acc > 95, "%.1f%%" % pft_acc)

# =====================================================================
# 8. UNIT CONSISTENCY CHECK
# =====================================================================
print("\n" + "=" * 70)
print("8. UNIT CONSISTENCY CHECK")
print("=" * 70)

for df, label in [(gee_day1, "GEE"), (cds_day1, "CDS")]:
    if "ta" in df.columns:
        check("%s ta in Celsius (max < 100)" % label, df["ta"].max() < 100, "max=%.2f" % df["ta"].max())
    if "soil_temperature_level_1" in df.columns:
        check(
            "%s soil_temp in Celsius (max < 100)" % label,
            df["soil_temperature_level_1"].max() < 100,
            "max=%.2f" % df["soil_temperature_level_1"].max(),
        )
    if "vpd" in df.columns:
        check("%s vpd in kPa (mean < 3)" % label, df["vpd"].mean() < 3, "mean=%.4f" % df["vpd"].mean())

# =====================================================================
# 9. STATISTICAL DISTRIBUTION COMPARISON (day 1)
# =====================================================================
print("\n" + "=" * 70)
print("9. STATISTICAL DISTRIBUTION COMPARISON (day 1)")
print("=" * 70)

DIST_VARS = ["ta", "vpd", "sw_in", "elevation", "LAI", "day_length"]
for var in DIST_VARS:
    if var in gee_day1.columns and var in cds_day1.columns:
        gv = gee_day1[var].dropna()
        cv = cds_day1[var].dropna()
        if len(gv) == 0 or len(cv) == 0:
            continue
        gmean, gstd = gv.mean(), gv.std()
        cmean, cstd = cv.mean(), cv.std()
        mean_diff = abs(gmean - cmean)
        std_ratio = gstd / max(cstd, 1e-9)

        check(
            "%s mean diff < 20%%" % var,
            mean_diff < 0.2 * max(abs(gmean), abs(cmean), 1),
            "GEE=%.3f CDS=%.3f diff=%.3f" % (gmean, cmean, mean_diff),
        )
        check(
            "%s std ratio in [0.5, 2.0]" % var,
            0.5 < std_ratio < 2.0,
            "GEE_std=%.3f CDS_std=%.3f ratio=%.3f" % (gstd, cstd, std_ratio),
        )

# =====================================================================
# 10. CROSS-DAY STABILITY
# =====================================================================
print("\n" + "=" * 70)
print("10. CROSS-DAY STABILITY")
print("=" * 70)

gee_rpd_all = gee_full.groupby(gee_ts_col).size()
cds_rpd_all = cds_full.groupby(cds_ts_col).size()

check(
    "GEE rows/day consistent",
    gee_rpd_all.min() == gee_rpd_all.max(),
    "min=%d max=%d" % (gee_rpd_all.min(), gee_rpd_all.max()),
)
check(
    "CDS rows/day consistent",
    cds_rpd_all.min() == cds_rpd_all.max(),
    "min=%d max=%d" % (cds_rpd_all.min(), cds_rpd_all.max()),
)

gee_total = len(gee_full)
cds_total = len(cds_full)
total_ratio = max(gee_total, cds_total) / max(min(gee_total, cds_total), 1)
check("Total rows within 20%%", total_ratio < 1.2, "GEE=%d CDS=%d ratio=%.3f" % (gee_total, cds_total, total_ratio))

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
