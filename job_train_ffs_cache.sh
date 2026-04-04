#!/bin/bash
#SBATCH --partition=zen4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=192
#SBATCH --mem=700G --time=1:00:00
#SBATCH --job-name=train_cache
#SBATCH --output=/scratch/tmp/yluo2/gsv/outputs/forward_selection/slurm_train_cache_%j.out

# Train XGBoost directly from FFS cache (exact data match with FFS evaluation)
cd /scratch/tmp/yluo2/gsv
source /scratch/tmp/yluo2/gsv/.venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv

OUT=/scratch/tmp/yluo2/gsv/outputs/forward_selection
SCORING=${1:-neg_rmse}

echo "=== Training from FFS cache with ${SCORING} selected features ==="
date

python -m src.forward_selection.train_from_cache \
    --cache_path "$OUT/feature_cache.npz" \
    --selected_features "$OUT/selected_features_${SCORING}.json" \
    --run_id "ffs_${SCORING}_cache" \
    --output_dir "/scratch/tmp/yluo2/gsv/models/xgb/ffs_${SCORING}_cache" \
    --n_splits 10 \
    --random_seed 42 \
    --shap_sample_size 50000 \
    --n_jobs 48

echo ""
echo "=== Done ==="
date
