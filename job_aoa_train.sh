#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=sap_aoa_train
#SBATCH --ntasks-per-node=1
#SBATCH --output=aoa_train_%j.log
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=36
#SBATCH --partition=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yluo2@uni-muenster.de

module load palma/2023a
module load GCC/12.3.0
module load OpenMPI/4.1.5
module load SciPy-bundle/2023.07
module load scikit-learn/1.3.1

pip install --user xgboost shap joblib

cd /scratch/tmp/yluo2/Global_Sap_Velocity

python src/hyperparameter_optimization/test_hyperparameter_tuning_ML_spatial_stratified.py \
  --model xgb --RANDOM_SEED 42 --n_groups 10 \
  --SPLIT_TYPE spatial_stratified --TIME_SCALE daily \
  --IS_TRANSFORM True --TRANSFORM_METHOD log1p \
  --IS_STRATIFIED True --IS_CV True \
  --SHAP_SAMPLE_SIZE 50000 \
  --run_id default_daily_nocoors_swcnor
