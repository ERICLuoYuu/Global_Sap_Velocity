#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=128G --time=01:00:00
#SBATCH --job-name=ffs_cache
#SBATCH --output=/scratch/tmp/yluo2/gsv/outputs/forward_selection/slurm_ffs_cache_%j.out

# Forward Feature Selection — Build feature cache
cd /scratch/tmp/yluo2/gsv
source /scratch/tmp/yluo2/gsv/.venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv

OUT=/scratch/tmp/yluo2/gsv/outputs/forward_selection
mkdir -p "$OUT"

echo "=== FFS: Building feature cache ==="
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"

python -m src.forward_selection.run_selection \
    --build_cache \
    --output_dir "$OUT" \
    --time_scale daily

echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Cache file:"
ls -lh "$OUT/feature_cache.npz" 2>/dev/null || echo "Cache not found!"
