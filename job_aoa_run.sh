#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=sap_aoa_run
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/aoa_run_%j.log
#SBATCH --time=12:00:00
#SBATCH --mem=240G
#SBATCH --cpus-per-task=128
#SBATCH --partition=zen2-128C-496G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yluo2@uni-muenster.de

module load palma/2023a
module load GCC/12.3.0

source /scratch/tmp/yluo2/gsv/.venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd /scratch/tmp/yluo2/gsv-wt/aoa
mkdir -p logs

python -m src.aoa.run_aoa \
  --model-dir outputs/models/xgb/aoa_daily_zen2 \
  --shap-csv outputs/plots/hyperparameter_optimization/xgb/aoa_daily_zen2/shap_feature_importance.csv \
  --run-id aoa_daily_zen2 \
  --output-dir outputs/aoa \
  --chunk-size 50000
