#!/bin/bash
#SBATCH --job-name=aoa_sweep
#SBATCH --partition=requeue-zen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/aoa_sweep_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/aoa_sweep_%j.err

# Record-length robustness sweep (review problem 3): rerun the full afternoon-depression
# analysis keeping only DENSE sites (>=240 / >=360 valid site-days). --restrict-to-qualifying
# forces Fig 1, the RF cross-check, and the decoupling to share ONE dense-site set, so the
# RF sweep is "restricted to qualifying sites" exactly as requested.

set -e

echo "=== Environment ==="
hostname
date
cd /scratch/tmp/yluo2/gsv
source .venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv:$PYTHONPATH
python --version

COMMON="--scale sapwood --climate-source site --tair-min 5.0 --min-daily-sf 0.0 \
    --min-am-pm-ratio 0.10 --n-bins 10 --rf-models 100 --rf-sd-scope site --restrict-to-qualifying"

for THR in 240 360; do
    echo
    echo "=== threshold = ${THR} valid days ==="
    OUT="outputs/afternoon_depression/threshold_${THR}"
    python src/afternoon_depression/run_afternoon_depression.py \
        $COMMON --min-valid-days ${THR} --output-dir "${OUT}"
    echo "--- verdict (threshold ${THR}) ---"
    cat "${OUT}/REPORT.md" 2>/dev/null || true
    echo "--- RF (threshold ${THR}) ---"
    cat "${OUT}/rf_sensitivity.csv" 2>/dev/null || true
    echo "Figures:"; { ls "${OUT}/figures"/*.png 2>/dev/null || true; } | wc -l
done

echo
echo "=== Done ==="
date
