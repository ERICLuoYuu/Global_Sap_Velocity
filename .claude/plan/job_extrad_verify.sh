#!/bin/bash
#SBATCH --job-name=extradverify
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/extradverify_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/extradverify_%j.err

# Verify the internal re-run before anything is promoted into the delivered tree.
#
# The claim under test is narrow and falsifiable: adding ext_rad to the corpus
# should change the ENV outputs of these sites and NOTHING ELSE. sap_analyzer
# derives its own day mask from pvlib and never reads ext_rad, so every sap
# output must come out byte-identical to what is already delivered. If a sap
# file moved, the change was not confined to what we think it was.
set -uo pipefail
NEW=/scratch/tmp/yluo2/qc_extrad
DELIVERED=/scratch/tmp/yluo2/gsv/outputs/processed_data/sapwood_combined
SITES="AUT_MMG AUT_ZOE CHE_DAV CHE_LAE CZE_BK1 CZE_LNZ CZE_RAJ CZE_STN DEU_HAI DEU_HAR DEU_HOH ESP_ABR ESP_LM1 ESP_LM2 ESP_LMA FIN_HYY FRA_BIL FRA_HES ITA_CP2 NOR_HUR POL_MEZ POL_TUC SWE_NOR SWE_SGR"

echo "### completion markers"
DONE=$(find "$NEW" -path "*/qc_flags/stage3_summary.json" | wc -l)
echo "  $DONE / 24"
[ "$DONE" -ne 24 ] && { echo "FATAL: incomplete run"; exit 1; }

echo
echo "### day/night-mask fallbacks across all 24 task logs (want 0)"
FB=$(cat /scratch/tmp/yluo2/gsv/logs/qcextrad_46236959_*.out 2>/dev/null | grep -c "Could not calculate day/night mask")
echo "  $FB"

echo
echo "### sap outputs: must be byte-identical to the delivered tree"
SAME=0; DIFF=0; MISS=0
for s in $SITES; do
  for rel in "sap/outliers_removed/${s}_sapf_data_outliers_removed.csv" \
             "sap/daily/${s}_sapf_data_daily.csv" \
             "sap/outliers_removed/completeness_reports/${s}_sapf_data_daily_completeness.csv"; do
    a=$(find "$NEW/$s" -path "*/$rel" -type f 2>/dev/null | head -1)
    b="$DELIVERED/$rel"
    if [ -z "$a" ] || [ ! -f "$b" ]; then MISS=$((MISS+1)); echo "  MISSING $s :: $rel"; continue; fi
    if cmp -s "$a" "$b"; then SAME=$((SAME+1)); else DIFF=$((DIFF+1)); echo "  SAP CHANGED: $s :: $rel"; fi
  done
done
echo "  identical=$SAME differing=$DIFF missing=$MISS"

echo
echo "### env outputs: expected to differ, and to carry ext_rad"
ECH=0; ESAME=0; EXT=0; NOEXT=0
for s in $SITES; do
  a=$(find "$NEW/$s" -name "${s}_env_data_outliers_removed.csv" -type f 2>/dev/null | head -1)
  b="$DELIVERED/env/outliers_removed/${s}_env_data_outliers_removed.csv"
  [ -z "$a" ] && continue
  if head -1 "$a" | tr "," "\n" | sed "s/\r$//" | grep -qx "ext_rad"; then EXT=$((EXT+1)); else NOEXT=$((NOEXT+1)); echo "  NO ext_rad: $s"; fi
  if [ -f "$b" ]; then
    if cmp -s "$a" "$b"; then ESAME=$((ESAME+1)); echo "  env UNCHANGED (suspicious): $s"; else ECH=$((ECH+1)); fi
  fi
done
echo "  env changed=$ECH unchanged=$ESAME | with ext_rad=$EXT without=$NOEXT"

echo
echo "### sanity: ext_rad survived QC intact at a sample site"
python3 - <<'PY' 2>/dev/null || echo "  (python check skipped)"
import csv, glob
f = glob.glob("/scratch/tmp/yluo2/qc_extrad/DEU_HAI/**/DEU_HAI_env_data_outliers_removed.csv", recursive=True)
if f:
    vals = []
    with open(f[0]) as fh:
        for row in csv.DictReader(fh):
            v = row.get("ext_rad", "")
            if v not in ("", "NA"):
                vals.append(float(v))
    if vals:
        nz = [v for v in vals if v == 0]
        print(f"  DEU_HAI ext_rad: n={len(vals)} min={min(vals):.1f} max={max(vals):.1f} zero_frac={len(nz)/len(vals):.3f}")
PY

if [ "$DIFF" -ne 0 ] || [ "$MISS" -ne 0 ] || [ "$NOEXT" -ne 0 ] || [ "$FB" -ne 0 ]; then
  echo "=== VERIFY FAILED ==="; exit 1
fi
echo "=== VERIFY OK ==="
