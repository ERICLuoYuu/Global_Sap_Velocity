#!/bin/bash
#SBATCH --job-name=sv_rebuild
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=32G --time=02:00:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/sv_rebuild_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/sv_rebuild_%j.err

# Rebuild fullyear_fullday_tx under the SERVABLE feature contract.
#
# Only the normalised soil-moisture families change (raw / sigma_clim and the
# climatology z-score). Every other cell is copied as text. The verification
# below does not trust the module that wrote the tree:
#   * EVERY site, EVERY untouched cell: compared with the source as a string
#   * a sample of sites: raw re-derived from the NEW columns and the cell table,
#     with raw parsed exactly (float_precision="round_trip")
#
# History: the first run (job 46722291) parsed and re-wrote every column with pandas;
# the default float parser moved 4-54 of 223 untouched columns per site by 1-2 ulp,
# and this verification caught it.
#
#   GATES_ONLY=1 sbatch -p express job_sv_rebuild.sh    # gates in ~1 min, no writes

set -uo pipefail
hostname; date
cd /scratch/tmp/yluo2/gsv || exit 1
source .venv/bin/activate || exit 1
export PYTHONPATH=/scratch/tmp/yluo2/gsv
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

SRC=outputs/processed_data/sapwood_combined/merged/sweep_20260902_hohfix/fullyear_fullday_tx
DST=outputs/processed_data/sapwood_combined/merged/sweep_20260902_hohfix_sv/fullyear_fullday_tx
CUBE=outputs/derived/swvl_sigma_climatology/swvl_sigma_climatology_v2_1995_2019.nc
SHA="$(git rev-parse --short HEAD)+uncommitted($(git status --porcelain --untracked-files=no | wc -l))"

echo "=== GATE 0: inputs ==="
[ -f "$SRC/RUN_PROVENANCE.json" ] || { echo "GATE 0 FAIL: no $SRC/RUN_PROVENANCE.json"; exit 1; }
[ -f "$CUBE" ] || { echo "GATE 0 FAIL: no cube at $CUBE"; exit 1; }
[ ! -e "$DST" ] || { echo "GATE 0 FAIL: $DST already exists"; exit 1; }
[ ! -e "$DST.partial" ] || { echo "GATE 0 FAIL: $DST.partial left by a crashed run"; exit 1; }
python -m src.feature_contract.rebuild_daily --help > /dev/null || { echo "GATE 0 FAIL: module not importable"; exit 1; }
grep -q "def splice_csv" src/feature_contract/rebuild_daily.py || { echo "GATE 0 FAIL: rebuild module predates the text splice"; exit 1; }
echo "  GATE 0 PASS  src=$SRC  git=$SHA"
if [ "${GATES_ONLY:-0}" = "1" ]; then echo "GATES_ONLY=1: stopping before any write"; exit 0; fi

echo
echo "=== REBUILD ==="
date
python -u -m src.feature_contract.rebuild_daily --src "$SRC" --dst "$DST" --cube "$CUBE" --git-sha "$SHA"
RC=$?
echo "### rebuild exit: $RC"
date
[ $RC -eq 0 ] || exit $RC

echo
echo "=== VERIFY (independent of the module) ==="
python - "$SRC" "$DST" <<'PYEOF'
import csv
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
FAMILY_PREFIX = "volumetric_soil_water_layer_"


def fail(message):
    sys.exit("  VERIFY FAIL: " + message)


def rows_of(path):
    with open(path, newline="") as fh:
        return list(csv.reader(fh))


src_files = sorted(f for f in glob.glob(src + "/daily/*_daily.csv") if "all_biomes" not in f)
dst_files = sorted(glob.glob(dst + "/daily/*_daily.csv"))
names = [os.path.basename(f)[: -len("_daily.csv")] for f in src_files]
if [os.path.basename(f) for f in src_files] != [os.path.basename(f) for f in dst_files]:
    fail("site file sets differ")
print("  site files: %d = %d" % (len(src_files), len(dst_files)))

prov, src_prov = json.load(open(dst + "/RUN_PROVENANCE.json")), json.load(open(src + "/RUN_PROVENANCE.json"))
if prov["feature_contract"]["name"] != "servable":
    fail("provenance contract is %r" % prov["feature_contract"]["name"])
if prov["flags"] != src_prov["flags"] or prov["config_name"] != src_prov["config_name"]:
    fail("merge flags or config_name were not carried over")
print("  provenance: flags carried, contract=servable, fallback cells=%d"
      % prov["feature_contract"]["swvl"]["n_fallback_cells"])

cells = pd.read_csv(dst + "/feature_contract_cells.csv")
if len(cells) != 4 * len(src_files):
    fail("cell table has %d rows, expected %d" % (len(cells), 4 * len(src_files)))

# 1. Text identity of every untouched cell, every site.
n_cells = 0
for k, site in enumerate(names):
    a, b = rows_of(src_files[k]), rows_of(dst_files[k])
    if a[0] != b[0] or len(a) != len(b):
        fail("%s: header or row count changed" % site)
    keep = [j for j, col in enumerate(a[0]) if not (col.startswith(FAMILY_PREFIX) and "_raw" not in col)]
    for r in range(1, len(a)):
        for j in keep:
            if a[r][j] != b[r][j]:
                fail("%s row %d column %s: %r -> %r" % (site, r, a[0][j], a[r][j], b[r][j]))
    n_cells += len(keep) * (len(a) - 1)
print("  untouched cells identical as text: %d cells over %d sites" % (n_cells, len(names)))

# 2. Numeric reconstruction of raw from the rewritten columns, on a sample.
rng = np.random.RandomState(0)
check = set(rng.choice(len(src_files), size=min(25, len(src_files)), replace=False).tolist())
for must in ("ARG_MAZ", "DEU_HIN_OAK", "ESP_LAS"):
    if must in names:
        check.add(names.index(must))
worst = 0.0
for k in sorted(check):
    site = names[k]
    a = pd.read_csv(src_files[k], low_memory=False, float_precision="round_trip")
    b = pd.read_csv(dst_files[k], low_memory=False, float_precision="round_trip")
    for layer in (1, 2, 3, 4):
        c = cells[(cells["site"] == site) & (cells["layer"] == layer)].iloc[0]
        raw = a["volumetric_soil_water_layer_%d_raw" % layer]
        e_bare = np.nanmax(np.abs(b["volumetric_soil_water_layer_%d" % layer] * c["sigma"] - raw))
        e_z = np.nanmax(np.abs(b["volumetric_soil_water_layer_%d_zscore" % layer] * c["sigma"] + c["mean"] - raw))
        worst = max(worst, e_bare, e_z)
print("  reconstruction on %d sites: worst |raw error| %.3g m3/m3" % (len(check), worst))
if not worst < 1e-9:
    fail("reconstruction error %.3g" % worst)

# 3. Positive control: the short-record site must move a lot, the long-record site little.
for site in ("ARG_MAZ", "DEU_HIN_OAK"):
    if site in names:
        k = names.index(site)
        col = "volumetric_soil_water_layer_3"
        old = pd.read_csv(src_files[k], usecols=[col])[col].median()
        new = pd.read_csv(dst_files[k], usecols=[col])[col].median()
        print("  %-12s swvl3 bare median: per-site sigma %.2f -> climatology sigma %.2f" % (site, old, new))
print("VERIFY PASS")
PYEOF
VRC=$?
echo "### verify exit: $VRC"
[ $VRC -eq 0 ] || exit $VRC
du -sh "$DST"
echo "STATUS: DONE"
date
