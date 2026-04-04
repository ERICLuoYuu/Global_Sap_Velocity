#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=smoke-viz
#SBATCH --ntasks-per-node=1
#SBATCH --output=/scratch/tmp/yluo2/gsv-wt/map-viz/logs/smoke_viz_%j.log
#SBATCH --error=/scratch/tmp/yluo2/gsv-wt/map-viz/logs/smoke_viz_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
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
echo "  Smoke Test: Visualization Pipeline"
echo "============================================"
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : $(hostname)"
echo "Started : $(date)"
echo "============================================"

# Test 1: Run viz on 1 month of daily XGB predictions (CSV input)
echo ""
echo "--- TEST 1: Daily CSV visualization (1 month) ---"
python -m src.make_prediction.prediction_visualization_hpc \
  --input-dir /scratch/tmp/yluo2/gsv/outputs/prediction \
  --output-dir outputs/maps \
  --run-id smoke_test_daily \
  --value-column sap_velocity_xgb \
  --csv-glob "prediction_2015_01_daily*xgb*.csv" \
  --sw-threshold 0 \
  --resolution 0.1 \
  --stats mean count \
  --no-png \
  --time-scale daily

T1_EXIT=$?
echo "TEST 1 exit code: ${T1_EXIT}"

# Test 2: Run pytest to verify all tests pass
echo ""
echo "--- TEST 2: pytest ---"
python -m pytest src/make_prediction/tests/test_prediction_visualization.py -v --tb=short
T2_EXIT=$?
echo "TEST 2 exit code: ${T2_EXIT}"

echo ""
echo "============================================"
echo "  Smoke Test Results"
echo "============================================"
echo "TEST 1 (viz pipeline): exit ${T1_EXIT}"
echo "TEST 2 (pytest):       exit ${T2_EXIT}"
echo "Finished: $(date)"
echo "============================================"

# Check outputs
echo ""
echo "--- Output files ---"
find outputs/maps/smoke_test_daily -type f 2>/dev/null | head -20
