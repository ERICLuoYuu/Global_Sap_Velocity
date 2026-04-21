#!/bin/bash
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --job-name=sm_shap_dryrun
#SBATCH --output=logs/sm_shap_dryrun_%j.out
#SBATCH --error=logs/sm_shap_dryrun_%j.err

set -euo pipefail
cd /scratch/tmp/yluo2/gsv
mkdir -p logs
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python src/Analyzers/per_site_sm_shap.py \
  --sm-variant raw --n-jobs 3 \
  --sites ARG_MAZ FIN_HYY AUS_KAR \
  --output-dir outputs/analysis/per_site_sm_shap_dryrun
