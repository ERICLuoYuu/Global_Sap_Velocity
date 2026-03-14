#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=predict-real
#SBATCH --ntasks-per-node=1
#SBATCH --output=/scratch/tmp/yluo2/gsv-wt/map-viz/logs/predict_real_%j.log
#SBATCH --error=/scratch/tmp/yluo2/gsv-wt/map-viz/logs/predict_real_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=64
#SBATCH --partition=zen4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yluo2@uni-muenster.de

sed -i 's/\r$//' "$0" 2>/dev/null || true

module load palma/2023a
module load GCC/12.3.0

source /scratch/tmp/yluo2/gsv/.venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=/scratch/tmp/yluo2/gsv-wt/map-viz

cd /scratch/tmp/yluo2/gsv-wt/map-viz
mkdir -p logs

echo "=== Real Data Prediction Test ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo ""

# Run prediction on all 2015 monthly files
python src/make_prediction/predict_sap_velocity_sequential.py \
    --input-dir /scratch/tmp/yluo2/Global_Sap_Velocity/data/dataset_for_prediction/2015_daily \
    --models-dir ./models \
    --model-type xgb \
    --run-id default_daily_without_coordinates \
    --time-scale daily \
    --output-format parquet \
    --compression gzip \
    --output ./outputs/prediction/daily/2015 \
    --validate \
    2>&1

EXIT_CODE=$?

echo ""
echo "=== Output files ==="
find ./outputs/prediction/daily/2015 -type f 2>/dev/null | head -20
echo ""
echo "=== Finished: $(date) ==="
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
