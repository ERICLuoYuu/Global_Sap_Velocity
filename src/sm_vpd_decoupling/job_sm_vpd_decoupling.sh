#!/bin/bash
#SBATCH --job-name=smvpd_decouple
#SBATCH --partition=requeue-zen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/smvpd_decouple_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/smvpd_decouple_%j.err

# Site-level SM-VPD decoupling: E (sap velocity) + Gc (Flo 2021) x 5 SM depths.
# Parametrized: override TAIR_MIN / OUTDIR to run the sensitivity tier.
#   Primary (Tair>15): sbatch job_sm_vpd_decoupling.sh
#   Sensitivity (Tair>5): sbatch --export=ALL,TAIR_MIN=5.0,OUTDIR=outputs/sm_vpd_decoupling/tair5 job_sm_vpd_decoupling.sh
# Figures are rendered at every bin count (--n-bins 5 10), filenames tagged _nbins{N}.
set -e
hostname; date
cd /scratch/tmp/yluo2/gsv
source .venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv:$PYTHONPATH
python --version

TAIR_MIN="${TAIR_MIN:-15.0}"
OUTDIR="${OUTDIR:-outputs/sm_vpd_decoupling/tair15}"
echo "TAIR_MIN=$TAIR_MIN  OUTDIR=$OUTDIR"

# Idempotent run: clear stale figures so renamed/removed outputs don't linger.
mkdir -p "$OUTDIR/figures"
rm -f "$OUTDIR"/figures/*.png

python src/sm_vpd_decoupling/run_sm_vpd_decoupling.py \
    --climate-source site \
    --tair-min "$TAIR_MIN" \
    --n-bins 5 10 \
    --min-valid-days 120 240 360 \
    --out-dir "$OUTDIR"

echo "=== Output ==="
echo "Tables:"; ls "$OUTDIR"/*.csv 2>/dev/null || true
echo "Figures:"; { ls "$OUTDIR"/figures/*.png 2>/dev/null || true; } | wc -l
date
