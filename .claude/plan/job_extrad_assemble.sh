#!/bin/bash
#SBATCH --job-name=extradasm
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/extradasm_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/extradasm_%j.err

# Promote the re-run env outputs for the 23 internal sites into the delivered
# tree. Sap outputs are NOT touched: they were verified byte-identical, so
# copying them would be churn that could only introduce risk.
#
# SWE_NOR is excluded on purpose. Its env file holds zero measured variables --
# the ICOS extraction failed -- and the ext_rad backfill briefly made it look
# processed. The corpus file was reverted; this keeps the delivered tree at 225
# env sites, matching the documented skip.
set -uo pipefail
NEW=/scratch/tmp/yluo2/qc_extrad
DST=/scratch/tmp/yluo2/gsv/outputs/processed_data/sapwood_combined
BAK=/scratch/tmp/yluo2/gsv/outputs/processed_data/sapwood_combined_env_pre_extrad
SITES="AUT_MMG AUT_ZOE CHE_DAV CHE_LAE CZE_BK1 CZE_LNZ CZE_RAJ CZE_STN DEU_HAI DEU_HAR DEU_HOH ESP_ABR ESP_LM1 ESP_LM2 ESP_LMA FIN_HYY FRA_BIL FRA_HES ITA_CP2 NOR_HUR POL_MEZ POL_TUC SWE_SGR"

NSITES=$(echo $SITES | wc -w)
echo "### promoting env outputs for $NSITES internal sites (SWE_NOR excluded by design)"
[ "$NSITES" -ne 23 ] && { echo "FATAL: expected 23 sites, got $NSITES"; exit 1; }

mkdir -p "$BAK/env/outliers_removed" "$BAK/env/daily"

COPIED=0; FAILED=0
for s in $SITES; do
  for pair in "env/outliers_removed/${s}_env_data_outliers_removed.csv" "env/daily/daily_${s}_env_data.csv"; do
    src=$(find "$NEW/$s" -path "*/$pair" -type f 2>/dev/null | head -1)
    dst="$DST/$pair"
    if [ -z "$src" ]; then echo "  MISSING SOURCE: $s :: $pair"; FAILED=$((FAILED+1)); continue; fi
    [ -f "$dst" ] && cp -p "$dst" "$BAK/$pair"
    cp -p "$src" "$dst" || { echo "  COPY FAILED: $pair"; FAILED=$((FAILED+1)); continue; }
    if [ "$(sha256sum "$src" | cut -d" " -f1)" != "$(sha256sum "$dst" | cut -d" " -f1)" ]; then
      echo "  SHA MISMATCH: $pair"; FAILED=$((FAILED+1)); continue
    fi
    COPIED=$((COPIED+1))
  done
done
echo "  copied+verified=$COPIED  failed=$FAILED  (expect 46)"

echo
echo "### delivered tree after promotion"
OR=$(find "$DST/env/outliers_removed" -maxdepth 1 -name "*_env_data_outliers_removed.csv" | wc -l)
DA=$(find "$DST/env/daily" -maxdepth 1 -name "daily_*_env_data.csv" | wc -l)
SA=$(find "$DST/sap/outliers_removed" -maxdepth 1 -name "*_sapf_data_outliers_removed.csv" | wc -l)
echo "  env/outliers_removed: $OR   env/daily: $DA   sap/outliers_removed: $SA"

echo
echo "### every delivered env output must now carry ext_rad"
W=0; WO=0
for f in "$DST/env/outliers_removed"/*_env_data_outliers_removed.csv; do
  if head -1 "$f" | tr "," "\n" | sed "s/\r$//" | grep -qx "ext_rad"; then W=$((W+1)); else WO=$((WO+1)); echo "  NO ext_rad: $(basename "$f")"; fi
done
echo "  outliers_removed with ext_rad: $W / $((W+WO))"

WD=0; WOD=0
for f in "$DST/env/daily"/daily_*_env_data.csv; do
  if head -1 "$f" | tr "," "\n" | sed "s/\r$//" | grep -q "ext_rad"; then WD=$((WD+1)); else WOD=$((WOD+1)); fi
done
echo "  daily with an ext_rad column:   $WD / $((WD+WOD))"

echo
echo "### SWE_NOR must NOT have gained an env output"
if [ -f "$DST/env/outliers_removed/SWE_NOR_env_data_outliers_removed.csv" ]; then
  echo "  FAIL: SWE_NOR env output present"; FAILED=$((FAILED+1))
else
  echo "  OK: absent, as documented"
fi

if [ "$FAILED" -ne 0 ] || [ "$COPIED" -ne 46 ] || [ "$WO" -ne 0 ]; then
  echo "=== ASSEMBLY FAILED ==="; exit 1
fi
echo "=== ENV PROMOTION OK ==="
