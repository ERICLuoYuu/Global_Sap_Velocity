#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=map-viz
#SBATCH --ntasks-per-node=1
#SBATCH --output=/scratch/tmp/yluo2/gsv-wt/map-viz/logs/map_viz_%j.log
#SBATCH --error=/scratch/tmp/yluo2/gsv-wt/map-viz/logs/map_viz_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=480G
#SBATCH --cpus-per-task=128
#SBATCH --partition=zen4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yluo2@uni-muenster.de

# Fix Windows line endings (self-healing for scripts edited on Windows)
sed -i 's/\r$//' "$0" 2>/dev/null || true

module load palma/2023a
module load GCC/12.3.0

source /scratch/tmp/yluo2/gsv/.venv/bin/activate

# Avoid thread oversubscription — libraries should use 1 thread each;
# parallelism is handled by Python (pandas chunked processing).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=/scratch/tmp/yluo2/gsv-wt/map-viz

cd /scratch/tmp/yluo2/gsv-wt/map-viz
mkdir -p logs

# ---- Configuration ----
RUN_ID="${RUN_ID:-viz_default}"
INPUT_DIR="${INPUT_DIR:-outputs/prediction}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/maps}"
VALUE_COL="${VALUE_COL:-sap_velocity_cnn_lstm}"
CSV_GLOB="${CSV_GLOB:-*predictions_raw.csv}"
SW_THRESHOLD="${SW_THRESHOLD:-15}"
RESOLUTION="${RESOLUTION:-0.1}"
DASK_BLOCKSIZE="${DASK_BLOCKSIZE:-256MB}"

echo "============================================"
echo "  Prediction Visualization Pipeline"
echo "============================================"
echo "Job ID        : ${SLURM_JOB_ID}"
echo "Node          : $(hostname)"
echo "CPUs          : ${SLURM_CPUS_PER_TASK}"
echo "Memory        : ${SLURM_MEM_PER_NODE:-480G}"
echo "Run ID        : ${RUN_ID}"
echo "Input dir     : ${INPUT_DIR}"
echo "CSV glob      : ${CSV_GLOB}"
echo "Value column  : ${VALUE_COL}"
echo "SW threshold  : ${SW_THRESHOLD}"
echo "Resolution    : ${RESOLUTION}"
echo "============================================"

python -m src.make_prediction.prediction_visualization_hpc \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}" \
  --value-column "${VALUE_COL}" \
  --csv-glob "${CSV_GLOB}" \
  --sw-threshold "${SW_THRESHOLD}" \
  --resolution "${RESOLUTION}" \
  --dask-blocksize "${DASK_BLOCKSIZE}" \
  --stats mean median max min std count

echo "============================================"
echo "  Pipeline finished at $(date)"
echo "============================================"
