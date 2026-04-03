#!/bin/bash
#SBATCH --partition=zen2-128C-496G
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=128
#SBATCH --mem=480G --time=12:00:00
#SBATCH --job-name=ffs_neg_rmse
#SBATCH --output=/scratch/tmp/yluo2/gsv/outputs/forward_selection/slurm_ffs_neg_rmse_%j.out

# Forward Feature Selection — neg RMSE
cd /scratch/tmp/yluo2/gsv
source /scratch/tmp/yluo2/gsv/.venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv

OUT=/scratch/tmp/yluo2/gsv/outputs/forward_selection
CACHE="$OUT/feature_cache.npz"

echo "=== FFS: neg_rmse ==="
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"

python -m src.forward_selection.run_selection \
    --scoring neg_rmse \
    --cache_path "$CACHE" \
    --output_dir "$OUT" \
    --n_jobs_sfs 32 \
    --n_jobs_xgb 4

echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
