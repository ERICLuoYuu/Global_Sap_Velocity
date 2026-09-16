#!/bin/bash
#SBATCH --job-name=m5_ml
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=128G --time=12:00:00
#SBATCH --array=1-2
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/m5_ml_%A_%a.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/m5_ml_%A_%a.err

# Tree half of the five-model comparison: 1=xgb, 2=rf.
#   dataset  fullyear_fullday_tx x hohfix
#   features the UNCAPPED FFS selection (50 columns), pinned by job_m5_pin.sh
#   CV       grid @ 0.05 deg -- the SAME grouping the selection was done under
#
# Grouping is not a free choice here.  The cache's `groups` array IS the FFS
# cross-validation design, so scoring a grid-selected set under the per-site
# grouping measures nothing except the co-location leakage (+0.145 pooled R2).
#
# Three trainer defaults are wrong for this corpus and every one fails QUIETLY:
#   --data_dir             defaults to paths.merged_daytime_only_dir, which does
#                          not exist for sapwood_combined; on a missing dir the
#                          loader logs CRITICAL and returns [] instead of raising
#   --IS_ONLY_DAY          must match the FFS cache (False)
#   --spatial_split_method defaults to per-site, the leaky grouping
# All three are passed explicitly below.
#
# ffs_to_training_args.py is deliberately NOT used: it hardcodes the April-era
# _prediction trainer, so the chain would silently train with stale code.

set -uo pipefail
hostname; date

cd /scratch/tmp/yluo2/gsv || exit 1
source .venv/bin/activate || exit 1

export SWEEP_NAME=sweep_20260902_hohfix
export GROUPING=grid
export FFS_TAG=_sm
source ./sweep5_config.sh || exit 1
sweep5_config 1 || exit 1

export ANALYSIS_SCALE=sapwood_combined
export PYTHONPATH=/scratch/tmp/yluo2/gsv:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

T="${SLURM_ARRAY_TASK_ID:?not running as an array task}"
case "$T" in
  1) MODEL=xgb; HYPER=src/hyperparameter_optimization/JSON/XGB_hyperparameters_fixed.json ;;
  2) MODEL=rf ; HYPER=src/hyperparameter_optimization/JSON/RF_hyperparameters_fixed.json  ;;
  *) echo "FATAL: task must be 1 (xgb) or 2 (rf), got '$T'"; exit 2 ;;
esac

M5_TAG="m5_${GROUP_TAG}${SWEEP_TAG}${FFS_TAG}"
FEATURES="$FFS_OUT/train_features_${M5_TAG}.json"
M5_RUN_ID="${RUN_ID}_${MODEL}"
TRAINER=src/hyperparameter_optimization/test_hyperparameter_tuning_ML_spatial_stratified.py

echo "=============================================================="
echo "$MODEL  x  uncapped FFS (50 cols)  x  $NAME"
echo "  task      : $T"
echo "  run_id    : $M5_RUN_ID"
echo "  data_dir  : $DATA_DIR   (trainer appends /daily)"
echo "  features  : $FEATURES"
echo "  hyperparam: $HYPER"
echo "  grouping  : grid @ 0.05 deg (matches the selection)"
echo "  contract  : $FEATURE_CONTRACT"
echo "  git sha   : $(git rev-parse --short HEAD)"
echo "=============================================================="

echo
echo "=== GATE 0: the tree was built under FEATURE_CONTRACT=$FEATURE_CONTRACT (see job_m5_pin.sh) ==="
sweep5_assert_contract "$DATA_DIR" "$FEATURE_CONTRACT" || exit 1

echo
echo "=== GATE A: dataset identity ==="
sweep5_assert_provenance "$DATA_DIR" "$NAME" "$EXP_GS" "$EXP_DAY" "$EXP_DAYSAP" "$EXP_TX" || exit 1

echo
echo "=== GATE B: the pinned feature file is the one job_m5_pin.sh wrote ==="
# Verifying the HASH, not just existence.  The whole point of the comparison is
# that all five models saw one feature set; only a shared hash can prove that,
# and it also catches a stage-2 re-run that rewrote the pin underneath us.
[ -f "$FEATURES" ]          || { echo "  GATE B FAIL: no $FEATURES -- run job_m5_pin.sh first"; exit 1; }
[ -f "${FEATURES}.sha256" ] || { echo "  GATE B FAIL: no ${FEATURES}.sha256"; exit 1; }
sha256sum -c "${FEATURES}.sha256" || { echo "  GATE B FAIL: pin hash mismatch"; exit 1; }
echo "  GATE B PASS: $(python -c "import json,sys;print(len(json.load(open(sys.argv[1]))['selected_features']))" "$FEATURES") columns, hash verified"

echo
echo "=== GATE C: run_id is fresh ==="
# The sweep-5 stage-3 XGB run already occupies the un-suffixed run_id
# ($RUN_ID).  This comparison writes to <run_id>_<model> so it cannot clobber
# it, and refuses to start if its own name is already taken.
for d in "models/$MODEL/$M5_RUN_ID" "outputs/models/$MODEL/$M5_RUN_ID" \
         "outputs/plots/hyperparameter_optimization/$MODEL/$M5_RUN_ID"; do
    if [ -d "$d" ]; then echo "  GATE C FAIL: $d already exists"; exit 1; fi
done
echo "  GATE C PASS: $M5_RUN_ID is fresh"

echo
echo "=== TRAINING ==="
date
python -u "$TRAINER" \
    --model "$MODEL" --RANDOM_SEED 42 --n_groups 10 \
    --SPLIT_TYPE spatial_stratified --TIME_SCALE daily \
    --IS_TRANSFORM True --TRANSFORM_METHOD log1p \
    --IS_STRATIFIED True --IS_CV True --IS_ONLY_DAY False \
    --SHAP_SAMPLE_SIZE 50000 \
    --selected_features "$FEATURES" \
    --hyperparameters "$HYPER" \
    --data_dir "$DATA_DIR" \
    --TARGET_COL sap_velocity \
    --r2_method both \
    --spatial_split_method "$TRAIN_METHOD" --grid_size 0.05 \
    --FEATURE_CONTRACT "$FEATURE_CONTRACT" \
    --run_id "$M5_RUN_ID"
TRAIN_EXIT=$?
echo "### training exit: $TRAIN_EXIT"
date
if [ $TRAIN_EXIT -ne 0 ]; then echo "STATUS: TRAINING FAILED"; exit $TRAIN_EXIT; fi

echo
echo "=== SITES DROPPED BY THE MISSING-COLUMN GUARD ==="
# The trainer logs via logging -> StreamHandler -> STDERR, so this must read the
# .err file; grepping .out silently reports zero.  A missing column drops the
# WHOLE SITE, so the feature list changes the site POPULATION, not just the
# columns.  The reference XGB run on this same 50-column set kept 156 of 212.
ERRLOG="logs/m5_ml_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
echo "  sites loaded : $(grep -c 'Loaded data for site' "$ERRLOG" 2>/dev/null)"
echo "  sites skipped: $(grep -c 'Missing columns' "$ERRLOG" 2>/dev/null)"
grep -oE "Missing columns: \[[^]]*\]" "$ERRLOG" 2>/dev/null | sort | uniq -c | sort -rn | head -10

echo
echo "=== OOF SUMMARY (original scale) ==="
OOF="outputs/plots/hyperparameter_optimization/$MODEL/$M5_RUN_ID/oof_predictions_$M5_RUN_ID.csv"
python - "$OOF" "$MODEL" <<'PYEOF'
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
try:
    df = pd.read_csv(sys.argv[1])
except FileNotFoundError:
    sys.exit('  OOF file not found: %s' % sys.argv[1])
y, p = df['observed'].to_numpy(), df['predicted'].to_numpy()
per = df.groupby('site_id')[['observed', 'predicted']].apply(
    lambda g: r2_score(g['observed'], g['predicted']) if len(g) > 2 else np.nan)
print('  model         : %s' % sys.argv[2])
print('  n_rows        : %d' % len(df))
print('  n_sites       : %d' % df['site_id'].nunique())
print('  Var(y)        : %.4f' % np.var(y))
print('  pooled R2     : %.4f' % r2_score(y, p))
print('  RMSE          : %.4f' % np.sqrt(mean_squared_error(y, p)))
print('  MAE           : %.4f' % mean_absolute_error(y, p))
print('  median site R2: %.4f  (n=%d)' % (np.nanmedian(per), per.notna().sum()))
PYEOF

echo
echo "STATUS: DONE  task=$T  model=$MODEL  run_id=$M5_RUN_ID"
date
