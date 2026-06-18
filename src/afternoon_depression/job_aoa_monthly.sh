#!/bin/bash
#SBATCH --job-name=aoa_monthly
#SBATCH --partition=requeue-zen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/aoa_monthly_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/aoa_monthly_%j.err
set -e
cd /scratch/tmp/yluo2/gsv
source .venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv:$PYTHONPATH
python --version
echo "=== pytest (py3.9 compat gate) ==="
python -m pytest src/afternoon_depression/tests/ -q
echo "=== run analysis (now incl. MONTHLY figures) ==="
python src/afternoon_depression/run_afternoon_depression.py \
    --scale sapwood --climate-source site --tair-min 5.0 --min-daily-sf 0.0 \
    --min-am-pm-ratio 0.10 --min-valid-days 120 --n-bins 10 --min-cell-sites 20 \
    --min-valid-months 3 --rf-models 100
echo "=== figures ==="
ls -la outputs/afternoon_depression/figures/*.png 2>/dev/null || true
echo "=== monthly table ==="
ls -la outputs/afternoon_depression/site_month_table.csv 2>/dev/null || true
echo "=== verdict ==="
cat outputs/afternoon_depression/REPORT.md 2>/dev/null || true
