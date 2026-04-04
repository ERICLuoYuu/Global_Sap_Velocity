#!/bin/bash
#SBATCH --partition=zen4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=192
#SBATCH --mem=750G --time=2-00:00:00
#SBATCH --job-name=ffs_mean_r2
#SBATCH --output=/scratch/tmp/yluo2/gsv/outputs/forward_selection/slurm_ffs_mean_r2_%j.out

# Forward Feature Selection — mean-fold R2
cd /scratch/tmp/yluo2/gsv
source /scratch/tmp/yluo2/gsv/.venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv

OUT=/scratch/tmp/yluo2/gsv/outputs/forward_selection
CACHE="$OUT/feature_cache.npz"

echo "=== FFS: mean_r2 ==="
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"

python -m src.forward_selection.run_selection \
    --scoring mean_r2 \
    --cache_path "$CACHE" \
    --output_dir "$OUT" \
    --n_jobs_sfs 32 \
    --n_jobs_xgb 4

echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
