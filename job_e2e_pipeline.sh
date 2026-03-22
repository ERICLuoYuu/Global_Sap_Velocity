#!/bin/bash
#SBATCH --job-name=e2e-pipeline
#SBATCH --partition=zen5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/tmp/yluo2/gsv-wt/map-viz/logs/e2e_pipeline_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv-wt/map-viz/logs/e2e_pipeline_%j.err

set -e

echo "========================================================================"
echo "END-TO-END PREDICTION PIPELINE TEST"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

cd /scratch/tmp/yluo2/gsv-wt/map-viz
source /scratch/tmp/yluo2/gsv/.venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv-wt/map-viz:/scratch/tmp/yluo2/gsv-wt/map-viz/src/make_prediction
export PYTHONUNBUFFERED=1
export GSV_ERA5LAND_TEMP_DIR=/scratch/tmp/yluo2/gsv-wt/map-viz/temp_era5

RUN_ID="default_daily_without_coordinates"
E2E_DIR="outputs/e2e_test"
ERA5_OUT="${E2E_DIR}/era5"
PRED_OUT="${E2E_DIR}/predictions"
VIZ_OUT="${E2E_DIR}/maps"

mkdir -p logs temp_era5 "${ERA5_OUT}/2020_daily" "${PRED_OUT}" "${VIZ_OUT}"

# ======================================================================
# STAGE 1: process_era5land_gee — one day (2020-07-15)
# ======================================================================
echo ""
echo "======== STAGE 1: ERA5-Land + Static Features (GEE) ========"
echo ""

python src/make_prediction/process_era5land_gee_opt_fix.py     --year 2020 --month 7 --day 15     --time-scale daily     --shapefile /scratch/tmp/yluo2/Global_Sap_Velocity/data/raw/grided/forest_pft_mask/forest_pft_mask.shp     --output "${ERA5_OUT}"

STAGE1_CSV="${ERA5_OUT}/2020_daily/prediction_2020_07_15.csv"
if [ ! -f "${STAGE1_CSV}" ]; then
    echo "FAIL: Stage 1 output not found: ${STAGE1_CSV}"
    exit 1
fi
echo ""
echo "Stage 1 output: $(wc -l < "${STAGE1_CSV}") lines"
echo "Stage 1 DONE"

# ======================================================================
# STAGE 2: predict_sap_velocity_sequential
# ======================================================================
echo ""
echo "======== STAGE 2: XGBoost Prediction ========"
echo ""

python src/make_prediction/predict_sap_velocity_sequential.py     --input-dir "${ERA5_OUT}/2020_daily"     --output "${PRED_OUT}"     --models-dir /scratch/tmp/yluo2/Global_Sap_Velocity/models     --model-type xgb     --run-id "${RUN_ID}"     --time-scale daily     --output-format csv     --validate

echo ""
echo "Stage 2 output files:"
ls -lh "${PRED_OUT}/" 2>/dev/null
echo "Stage 2 DONE"

# ======================================================================
# STAGE 3: prediction_visualization_hpc
# ======================================================================
echo ""
echo "======== STAGE 3: Visualization ========"
echo ""

# Find the prediction output (could be sap_velocity_xgb or similar column)
PRED_FILE=$(ls "${PRED_OUT}"/*prediction*2020*07*15* 2>/dev/null | head -1)
if [ -z "${PRED_FILE}" ]; then
    PRED_FILE=$(ls "${PRED_OUT}"/*.csv 2>/dev/null | head -1)
fi

if [ -z "${PRED_FILE}" ]; then
    echo "WARN: No prediction CSV found, listing dir:"
    ls -la "${PRED_OUT}/"
    echo "Skipping visualization"
else
    # Detect value column name
    VAL_COL=$(head -1 "${PRED_FILE}" | tr ',' '\n' | grep -i 'sap_velocity' | head -1)
    if [ -z "${VAL_COL}" ]; then
        VAL_COL=$(head -1 "${PRED_FILE}" | tr ',' '\n' | grep -i 'predicted' | head -1)
    fi
    echo "Prediction file: ${PRED_FILE}"
    echo "Value column: ${VAL_COL}"

    python src/make_prediction/prediction_visualization_hpc.py         --input-dir "${PRED_OUT}"         --output-dir "${VIZ_OUT}"         --run-id "${RUN_ID}"         --value-column "${VAL_COL}"         --csv-glob "*.csv"         --time-scale daily         --stats mean median max         --dpi 150

    echo ""
    echo "Stage 3 output files:"
    find "${VIZ_OUT}" -type f | head -20
    echo "Stage 3 DONE"
fi

# ======================================================================
# SUMMARY
# ======================================================================
echo ""
echo "========================================================================"
echo "E2E PIPELINE COMPLETE"
echo "========================================================================"
echo "ERA5 output:       ${STAGE1_CSV}"
echo "Prediction output: ${PRED_OUT}/"
echo "Visualization:     ${VIZ_OUT}/"

# Quick extent check on prediction output
python3 -c "
import pandas as pd, glob, os
files = sorted(glob.glob('${PRED_OUT}/*.csv'))
for f in files[:2]:
    df = pd.read_csv(f, nrows=500000)
    lon_col = [c for c in df.columns if 'lon' in c.lower()][0]
    lat_col = [c for c in df.columns if 'lat' in c.lower()][0]
    val_cols = [c for c in df.columns if 'sap_velocity' in c.lower() or 'predicted' in c.lower()]
    print(f'\n{os.path.basename(f)}:')
    print(f'  Rows: {len(df):,}')
    print(f'  Lon: [{df[lon_col].min():.2f}, {df[lon_col].max():.2f}]')
    print(f'  Lat: [{df[lat_col].min():.2f}, {df[lat_col].max():.2f}]')
    print(f'  West of -100: {(df[lon_col] < -100).sum():,}')
    for vc in val_cols:
        print(f'  {vc}: [{df[vc].min():.4f}, {df[vc].max():.4f}], mean={df[vc].mean():.4f}')
"

echo ""
echo "End: $(date)"
