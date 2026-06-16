#!/bin/bash
#SBATCH --job-name=smvpd_merge
#SBATCH --partition=requeue-zen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/smvpd_merge_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/smvpd_merge_%j.err

# Data production for the SM-VPD decoupling analysis:
# daytime-only + treatment-filter ON + growing-season OFF (Liu's Tair filter
# screens season downstream). Source = outliers_removed (measured, NOT gap-filled).
set -e
hostname; date
cd /scratch/tmp/yluo2/gsv
source .venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv:$PYTHONPATH
python --version

python notebooks/merge_gap_filled_hourly_orginal.py \
    --daytime-only \
    --apply-treatment-filter \
    --output-dir outputs/processed_data/sapwood/merged_decoupling

echo "=== Output ==="
ls outputs/processed_data/sapwood/merged_decoupling/daily/*.csv 2>/dev/null | wc -l || true
date
