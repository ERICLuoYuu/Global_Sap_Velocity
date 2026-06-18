#!/bin/bash
#SBATCH --job-name=fu_ann_sens
#SBATCH --partition=requeue-zen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/fu_ann_sens_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/fu_ann_sens_%j.err

# Fu et al. 2022 ANN SWC/VPD sensitivity: E (sap velocity) + Gc (Flo 2021) x 5 SM
# depths, at both 5x5 and 10x10 percentile bins. PyTorch CPU ensemble per site.
#   Default growing-season dataset: sbatch job_fu_ann_sensitivity.sh
#   Override data/out:  sbatch --export=ALL,DATADIR=...,OUTDIR=... job_fu_ann_sensitivity.sh
set -e
hostname; date
cd /scratch/tmp/yluo2/gsv
source .venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv:$PYTHONPATH
# Keep PyTorch / BLAS within the allocated cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
python --version
python -c "import torch; print('torch', torch.__version__); torch.set_num_threads(${OMP_NUM_THREADS})"

DATADIR="${DATADIR:-outputs/processed_data/sapwood/merged/daytime_only/growing_season/daily}"
OUTDIR="${OUTDIR:-outputs/fu_ann_sensitivity}"
echo "DATADIR=$DATADIR  OUTDIR=$OUTDIR"

# Idempotent run: clear stale figures so renamed/removed outputs don't linger.
mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/*.png

python src/fu_ann_sensitivity/run_fu_ann_sensitivity.py \
    --data-dir "$DATADIR" \
    --n-bins 5 10 \
    --n-repeats 5 \
    --min-valid-days 300 \
    --out-dir "$OUTDIR"

echo "=== Output ==="
echo "Tables:"; ls "$OUTDIR"/*.csv 2>/dev/null || true
echo "Figures:"; { ls "$OUTDIR"/*.png 2>/dev/null || true; } | wc -l
date
