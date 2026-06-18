#!/bin/bash
#SBATCH --job-name=aoa_transp
#SBATCH --partition=requeue-zen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/aoa_transp_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/aoa_transp_%j.err

# Afternoon Depression of Transpiration — climate-driver decoupling (Liu et al. 2024 EC method).
# Full 185-site run against the growing-season-only, daytime-only, raw-SWC hourly dataset.

set -e

echo "=== Environment ==="
hostname
date
cd /scratch/tmp/yluo2/gsv
source .venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv:$PYTHONPATH
python --version

echo
echo "=== Run afternoon-depression decoupling ==="
echo "Input : outputs/processed_data/sapwood/merged/daytime_only/growing_season/hourly/"
echo "Drivers: in-situ VPD + in-situ Tair; SM = ERA5-Land raw swvl 0-100cm; LAI = GLOBMAP"
# Default uses in-situ met (vpd, ta) + ERA5-Land soil moisture. For an all-ERA5
# run (needs dewpoint_2m), pass --climate-source era5 instead.
python src/afternoon_depression/run_afternoon_depression.py \
    --scale sapwood \
    --climate-source site \
    --tair-min 5.0 \
    --min-daily-sf 0.0 \
    --min-am-pm-ratio 0.10 \
    --min-valid-days 120 \
    --n-bins 10 \
    --rf-models 100

echo
echo "=== Run afternoon-depression decoupling — canopy conductance (Gc) ==="
# Same pipeline on whole-tree canopy conductance (Flo et al. 2021 Eqn 2, derived from
# sap flow; needs the elevation column). Outputs land in afternoon_depression/gc/.
# CAVEAT: Gc∝1/VPD, so the VPD legs of the decoupling are confounded (Oren 1999) — the
# Gc REPORT.md leads with this; the SM|VPD leg is the interpretable one.
python src/afternoon_depression/run_afternoon_depression.py \
    --scale sapwood \
    --climate-source site \
    --response gc \
    --tair-min 5.0 \
    --min-daily-sf 0.0 \
    --min-am-pm-ratio 0.10 \
    --min-valid-days 120 \
    --n-bins 10 \
    --rf-models 100

echo
echo "=== Output summary ==="
# Cosmetic summary only — never let a missing-file `ls` abort the job under `set -e`
# (that would flag a completed analysis as FAILED). Guard each with `|| true`.
OUT=outputs/afternoon_depression
echo "[SF] Tables:"; ls "$OUT"/*.csv 2>/dev/null || true
echo "[SF] Figures:"; { ls "$OUT"/figures/*.png 2>/dev/null || true; } | wc -l
echo "[SF] Verdict:"; cat "$OUT/REPORT.md" 2>/dev/null || true
echo "[Gc] Tables:"; ls "$OUT"/gc/*.csv 2>/dev/null || true
echo "[Gc] Figures:"; { ls "$OUT"/gc/figures/*.png 2>/dev/null || true; } | wc -l
echo "[Gc] Verdict:"; cat "$OUT/gc/REPORT.md" 2>/dev/null || true

echo
echo "=== Done ==="
date
