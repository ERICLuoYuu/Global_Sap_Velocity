#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=sap_aoa_run
#SBATCH --ntasks-per-node=1
#SBATCH --output=aoa_run_%j.log
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yluo2@uni-muenster.de

module load palma/2023a
module load GCC/12.3.0
module load OpenMPI/4.1.5
module load SciPy-bundle/2023.07
module load scikit-learn/1.3.1

cd /scratch/tmp/yluo2/Global_Sap_Velocity

python -m src.aoa.run_aoa \
  --model-dir outputs/models/xgb/default_daily_nocoors_swcnor \
  --shap-csv outputs/plots/hyperparameter_optimization/xgb/default_daily_nocoors_swcnor/shap_feature_importance.csv \
  --run-id default_daily_nocoors_swcnor \
  --output-dir outputs/aoa \
  --chunk-size 10000
