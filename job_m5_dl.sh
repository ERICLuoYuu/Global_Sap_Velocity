#!/bin/bash
#SBATCH --job-name=m5_dl
#SBATCH --partition=gpu3090
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=96G --time=1-00:00:00
#SBATCH --array=1-3
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/m5_dl_%A_%a.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/m5_dl_%A_%a.err

# Neural half of the five-model comparison: 1=ANN (non-windowing), 2=LSTM,
# 3=CNN-LSTM.  Same dataset, same pinned 50-column FFS set, same grid @ 0.05
# deg grouping and same log1p target as the xgb/rf half, so a tree-vs-net gap
# is the architecture rather than the objective or the CV design.
#
# ANN runs at INPUT_WIDTH=1 -- ONE timestep, no time window.  create_windowed_nn
# is Input(timesteps, features) -> Flatten -> Dense, so at width 1 it is exactly
# a plain MLP on a single day's drivers.  LSTM and CNN-LSTM keep INPUT_WIDTH=2,
# matching the trees, because a sequence model given one step is not a sequence
# model.  SHIFT stays 1 everywhere, so all five predict the SAME target instant
# and the ANN arm is a clean window ablation rather than a different task.
#
# Runs on .venv-gpu, NOT .venv.  .venv has tensorflow_cpu installed, so six
# earlier DL runs held an RTX 3090 at 0% for their entire duration and trained
# on a single CPU thread -- 99 ms/step instead of 14, a CNN-LSTM fold in 141
# min instead of 20.
#
# IS_CV is deliberately False: main()'s own docstring describes the design as
# "each fold serves as test set exactly once, with remaining folds split 8:1
# for train:validation", and DLOptimizer.fit takes that explicit X_val/y_val
# when is_cv is False.

set -uo pipefail
hostname; date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"

cd /scratch/tmp/yluo2/gsv || exit 1
source .venv-gpu/bin/activate || exit 1

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
  1) MODEL=ANN      ; INPUT_WIDTH=1 ; TAG=ann_iw1  ;;
  2) MODEL=LSTM     ; INPUT_WIDTH=2 ; TAG=lstm     ;;
  3) MODEL=CNN-LSTM ; INPUT_WIDTH=2 ; TAG=cnnlstm  ;;
  *) echo "FATAL: task must be 1 (ANN) 2 (LSTM) or 3 (CNN-LSTM), got '$T'"; exit 2 ;;
esac

M5_TAG="m5_${GROUP_TAG}${SWEEP_TAG}${FFS_TAG}"
FEATURES="$FFS_OUT/train_features_${M5_TAG}.json"
HYPER="src/hyperparameter_optimization/JSON/${MODEL}_hyperparameters_fixed.json"
M5_RUN_ID="${RUN_ID}_${TAG}"
TRAINER=src/hyperparameter_optimization/test_hyperparameter_tuning_DL_spatial_stratified.py

echo "=============================================================="
echo "$MODEL  x  uncapped FFS (50 cols)  x  $NAME   [GPU]"
echo "  task       : $T"
echo "  run_id     : $M5_RUN_ID"
echo "  INPUT_WIDTH: $INPUT_WIDTH  (LABEL_WIDTH=1, SHIFT=1)"
echo "  data_dir   : $DATA_DIR   (trainer appends /daily)"
echo "  features   : $FEATURES"
echo "  hyperparam : $HYPER"
echo "  venv       : .venv-gpu (tensorflow GPU build)"
echo "  contract   : $FEATURE_CONTRACT"
echo "  git sha    : $(git rev-parse --short HEAD)"
echo "=============================================================="

echo
echo "=== GATE 0: the tree was built under FEATURE_CONTRACT=$FEATURE_CONTRACT (see job_m5_pin.sh) ==="
# Ahead of the GPU guard: this check needs no GPU, and a refused tree should not cost a GPU start-up.
sweep5_assert_contract "$DATA_DIR" "$FEATURE_CONTRACT" || exit 1

echo
echo "=== GPU GUARD ==="
# nvidia-smi above proves the GPU is ALLOCATED, not that TensorFlow can USE it.
# That distinction hid tensorflow_cpu through six full runs.  Die in 30 s rather
# than burn a day at 0%.
python - <<'PYGUARD'
import sys
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print(f"  TF {tf.__version__}, built_with_cuda={tf.test.is_built_with_cuda()}, visible GPUs={len(gpus)}")
if not gpus:
    print("  GPU GUARD FAIL -- TensorFlow sees no GPU. Refusing to train on CPU.")
    sys.exit(90)
print("  GPU GUARD PASS")
PYGUARD
GUARD=$?
if [ $GUARD -ne 0 ]; then echo "STATUS: GPU GUARD FAILED"; exit $GUARD; fi

echo
echo "=== GATE A: dataset identity ==="
sweep5_assert_provenance "$DATA_DIR" "$NAME" "$EXP_GS" "$EXP_DAY" "$EXP_DAYSAP" "$EXP_TX" || exit 1

echo
echo "=== GATE B: the pinned feature file is the one job_m5_pin.sh wrote ==="
[ -f "$FEATURES" ]          || { echo "  GATE B FAIL: no $FEATURES -- run job_m5_pin.sh first"; exit 1; }
[ -f "${FEATURES}.sha256" ] || { echo "  GATE B FAIL: no ${FEATURES}.sha256"; exit 1; }
sha256sum -c "${FEATURES}.sha256" || { echo "  GATE B FAIL: pin hash mismatch"; exit 1; }
echo "  GATE B PASS: hash verified (same set the trees trained on)"

echo
echo "=== GATE C: run_id is fresh ==="
for d in "models/$MODEL/$M5_RUN_ID" "outputs/models/$MODEL/$M5_RUN_ID" \
         "outputs/plots/hyperparameter_optimization/$MODEL/$M5_RUN_ID"; do
    if [ -d "$d" ]; then echo "  GATE C FAIL: $d already exists"; exit 1; fi
done
echo "  GATE C PASS: $M5_RUN_ID is fresh"

echo -n "  site CSVs visible to the trainer: "
ls "$DATA_DIR/daily"/*daily.csv 2>/dev/null | grep -vc all_biomes_merged

echo
echo "=== TRAINING ==="
date
python -u "$TRAINER" \
    --model "$MODEL" --RANDOM_SEED 42 --n_groups 10 \
    --TIME_SCALE daily --BATCH_SIZE 64 \
    --INPUT_WIDTH "$INPUT_WIDTH" --LABEL_WIDTH 1 --SHIFT 1 \
    --IS_CV False --IS_STRATIFIED True --IS_SHUFFLE True --BALANCED False \
    --IS_TRANSFORM True --TRANSFORM_METHOD log1p \
    --selected_features "$FEATURES" \
    --hyperparameters "$HYPER" \
    --data_dir "$DATA_DIR" \
    --TARGET_COL sap_velocity \
    --spatial_split_method "$TRAIN_METHOD" --grid_size 0.05 \
    --FEATURE_CONTRACT "$FEATURE_CONTRACT" \
    --run_id "$M5_RUN_ID"
TRAIN_EXIT=$?
echo "### training exit: $TRAIN_EXIT"
date
if [ $TRAIN_EXIT -ne 0 ]; then echo "STATUS: TRAINING FAILED"; exit $TRAIN_EXIT; fi

echo
echo "=== GPU actually used? (memory + process list, NOT instantaneous util) ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null

echo
echo "=== SITES DROPPED BY THE MISSING-COLUMN GUARD ==="
ERRLOG="logs/m5_dl_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
echo "  sites loaded : $(grep -c 'Loaded data for site' "$ERRLOG" 2>/dev/null)"
echo "  sites skipped: $(grep -c 'Missing columns' "$ERRLOG" 2>/dev/null)"

echo
echo "=== artefacts ==="
ls -la "models/$MODEL/$M5_RUN_ID/" 2>/dev/null | head -20

echo
echo "=== OOF SUMMARY (original scale) ==="
# The DL out-of-fold writer is a recent addition; if this run predates it the
# summary is simply absent -- that is reported, not treated as a failure.
OOF="outputs/plots/hyperparameter_optimization/$MODEL/$M5_RUN_ID/oof_predictions_$M5_RUN_ID.csv"
if [ -f "$OOF" ]; then
python - "$OOF" "$MODEL" <<'PYEOF'
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
df = pd.read_csv(sys.argv[1])
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
else
    echo "  NOTE: no OOF file at $OOF -- compare via the trainer's own fold metrics"
fi

echo
echo "STATUS: DONE  task=$T  model=$MODEL  run_id=$M5_RUN_ID"
date
