#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=fullyr-viz
#SBATCH --ntasks-per-node=1
#SBATCH --output=/scratch/tmp/yluo2/gsv-wt/map-viz/logs/fullyr_viz_%j.log
#SBATCH --error=/scratch/tmp/yluo2/gsv-wt/map-viz/logs/fullyr_viz_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=normal
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

echo "============================================"
echo "  Full Year 2015: Visualization Pipeline"
echo "============================================"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : $(hostname)"
echo "CPUs    : ${SLURM_CPUS_PER_TASK}"
echo "Memory  : 64G"
echo "Started : $(date)"
echo "============================================"

python -m src.make_prediction.prediction_visualization_hpc \
  --input-dir /scratch/tmp/yluo2/gsv/outputs/prediction \
  --output-dir outputs/maps \
  --run-id viz_2015_fullyr_test \
  --value-column sap_velocity_xgb \
  --csv-glob "prediction_2015_*_daily_predictions_xgb_default_daily_without_coordinates.csv" \
  --sw-threshold 0 \
  --resolution 0.1 \
  --stats mean median max min std count \
  --no-png \
  --time-scale daily

EXIT_CODE=$?

echo "============================================"
echo "  Pipeline exit code: ${EXIT_CODE}"
echo "  Finished: $(date)"
echo "============================================"

echo ""
echo "--- Output summary ---"
echo "GeoTIFFs:"
find outputs/maps/viz_2015_fullyr_test -name "*.tif" -not -path "*/composites/*" | wc -l
echo "Composites:"
ls outputs/maps/viz_2015_fullyr_test/composites/*.tif 2>/dev/null | wc -l
echo "Total size:"
du -sh outputs/maps/viz_2015_fullyr_test 2>/dev/null
