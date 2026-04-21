#!/bin/bash
#SBATCH --partition=zen2-128C-496G
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --job-name=per_site_sm_shap
#SBATCH --output=logs/per_site_sm_shap_%j.out
#SBATCH --error=logs/per_site_sm_shap_%j.err

set -euo pipefail

cd /scratch/tmp/yluo2/gsv
mkdir -p logs

source .venv/bin/activate
export PYTHONUNBUFFERED=1
# keep XGBoost / sklearn / BLAS serial; joblib handles outer parallelism
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== raw variant ==="
python src/Analyzers/per_site_sm_shap.py --sm-variant raw    --n-jobs 128

echo "=== zscore variant ==="
python src/Analyzers/per_site_sm_shap.py --sm-variant zscore --n-jobs 128

echo "Done."
