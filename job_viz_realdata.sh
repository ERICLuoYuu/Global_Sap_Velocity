#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=viz-2015
#SBATCH --ntasks-per-node=1
#SBATCH --output=/scratch/tmp/yluo2/gsv-wt/map-viz/logs/viz_real_%j.log
#SBATCH --error=/scratch/tmp/yluo2/gsv-wt/map-viz/logs/viz_real_%j.err
#SBATCH --time=06:00:00
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

echo "=== Visualization Pipeline Test ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo ""

python src/make_prediction/prediction_visualization_hpc.py \
    --input-dir ./outputs/prediction/daily/2015 \
    --output-dir ./outputs/maps \
    --run-id xgb_2015_daily \
    --value-column sap_velocity_xgb \
    --timestamp-col "timestamp.1" \
    --input-format parquet \
    --csv-glob "*predictions*.parquet" \
    --time-scale daily \
    --resolution 0.1 \
    --stats mean std count \
    --dpi 150 \
    --sw-threshold 0 \
    2>&1

EXIT_CODE=$?

echo ""
echo "=== Output files ==="
find ./outputs/maps/xgb_2015_daily -type f 2>/dev/null | wc -l
echo "files total"
echo ""
find ./outputs/maps/xgb_2015_daily -type f -name "*.tif" 2>/dev/null | wc -l
echo "GeoTIFF files"
echo ""
find ./outputs/maps/xgb_2015_daily -type f -name "*.png" 2>/dev/null | wc -l
echo "PNG files"
echo ""
echo "=== Finished: $(date) ==="
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
