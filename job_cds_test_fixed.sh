#!/bin/bash
#SBATCH --job-name=cds-pft-fix
#SBATCH --output=/scratch/tmp/yluo2/gsv-wt/map-viz/logs/cds_pft_fix_%j.log
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=zen5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yu.luo@uni-muenster.de

export PYTHONUNBUFFERED=1
source /scratch/tmp/yluo2/gsv/.venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv-wt/map-viz
cd /scratch/tmp/yluo2/gsv-wt/map-viz

SHAPEFILE="/scratch/tmp/yluo2/gsv/data/raw/grided/tree_cover_shapefile_dissolved/tree_cover_shapefile_dissolved.shp"
OUTPUT_DIR="/scratch/tmp/yluo2/gsv-wt/map-viz/outputs/predictions/cds_era5_fixed"

mkdir -p logs

echo "=========================================="
echo "CDS ERA5-Land — PFT + precip/PET fix test"
echo "Date: $(date)  Node: $(hostname)"
echo "Using worktree: $(pwd)"
echo "Shapefile: $SHAPEFILE"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

python3 -c "import cdsapi; c = cdsapi.Client(); print('CDS API OK')"
if [ $? -ne 0 ]; then
    echo "FATAL: CDS API failed"
    exit 1
fi

# GEE auth is handled inside the Python script using stored credentials

python3 src/make_prediction/process_era5land_cds.py \
    --year 2020 \
    --month 1 \
    --time-scale daily \
    --shapefile "$SHAPEFILE" \
    --output-dir "$OUTPUT_DIR" \
    --validate \
    --verbose

echo "=========================================="
echo "Job finished: $(date)"
echo "Exit code: $?"
echo "=========================================="
