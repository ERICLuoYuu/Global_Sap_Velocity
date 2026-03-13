#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=sap_aoa_train
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/aoa_train_%j.log
#SBATCH --time=24:00:00
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

python src/hyperparameter_optimization/test_hyperparameter_tuning_ML_spatial_stratified.py \
  --model xgb --RANDOM_SEED 42 --n_groups 10 \
  --SPLIT_TYPE spatial_stratified --TIME_SCALE daily \
  --IS_TRANSFORM True --TRANSFORM_METHOD log1p \
  --IS_STRATIFIED True --IS_CV True \
  --SHAP_SAMPLE_SIZE 50000 \
  --run_id aoa_daily_zen2
