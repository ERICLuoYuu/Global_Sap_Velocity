#!/bin/bash
#SBATCH --job-name=sw5_train
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=128G --time=08:00:00
#SBATCH --array=1-5
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/sw5_train_%A_%a.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/sw5_train_%A_%a.err

# Stage 3 of 3: Stage-5 XGBoost training on each config's own FFS selection.
#
# Evaluation grouping is GRID for all five, matching the grouping the selection
# was done under. Holding the CV design fixed is what makes the five numbers
# comparable to each other; the per-site arm is the leaky one (+0.145 pooled R2)
# and is deliberately not run here.
#
# Three trainer defaults are wrong for this corpus and every one fails QUIETLY:
#   --data_dir            defaults to paths.merged_daytime_only_dir, which does
#                         not exist for sapwood_combined; on a missing dir the
#                         loader logs CRITICAL and returns [] instead of raising
#   --IS_ONLY_DAY         must match the FFS cache (False). On a DAILY frame this
#                         is `sw_in > 10`, a brightness threshold on the daily
#                         mean -- NOT the daytime scoping, which already happened
#                         at the merge (and for config 1 deliberately did not)
#   --spatial_split_method  defaults to per-site, the leaky grouping
# All three are passed explicitly below.
#
# ffs_to_training_args.py is deliberately NOT used: it hardcodes the April-era
# _prediction trainer, so the chain would silently train with stale code.

set -uo pipefail
hostname; date

cd /scratch/tmp/yluo2/gsv || exit 1
source .venv/bin/activate || exit 1
source ./sweep5_config.sh || exit 1

export ANALYSIS_SCALE=sapwood_combined
export PYTHONPATH=/scratch/tmp/yluo2/gsv:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

T="${SLURM_ARRAY_TASK_ID:?not running as an array task}"
sweep5_config "$T" || exit 1

FFS_JSON="$FFS_OUT/ffs_mean_r2_results.json"
FEATURES="$FFS_OUT/train_features_${RUN_ID}.json"
HYPER=src/hyperparameter_optimization/JSON/XGB_hyperparameters_fixed.json
TRAINER=src/hyperparameter_optimization/test_hyperparameter_tuning_ML_spatial_stratified.py

echo "=============================================================="
echo "FFS-selected features x $NAME"
echo "  task      : $T"
echo "  run_id    : $RUN_ID"
echo "  data_dir  : $DATA_DIR  (trainer appends /daily)"
echo "  ffs json  : $FFS_JSON"
echo "  grouping  : grid @ 0.05 deg (matches the selection)"
echo "  git sha   : $(git rev-parse --short HEAD)"
echo "=============================================================="

echo
echo "=== GATE 1: dataset identity ==="
sweep5_assert_provenance "$DATA_DIR" "$NAME" "$EXP_GS" "$EXP_DAY" "$EXP_DAYSAP" "$EXP_TX" || exit 1

echo
echo "=== GATE 2: pin the FFS selection as this run's feature file ==="
# Pinned per run_id rather than read live: a later re-run of stage 2 must not be
# able to retro-change what this model was trained on.
python - "$FFS_JSON" "$FEATURES" <<'PYEOF' || exit 1
import json, sys
src, dst = sys.argv[1], sys.argv[2]
try:
    r = json.load(open(src))
except FileNotFoundError:
    sys.exit('  GATE 2 FAIL: %s not found -- did stage 2 finish?' % src)
feats = r.get('selected_features') or []
if not feats:
    sys.exit('  GATE 2 FAIL: no selected_features in %s' % src)
if 'sw_in' not in feats:
    sys.exit('  GATE 2 FAIL: sw_in not selected -- the trainer will raise')
if 'pft' in feats:
    sys.exit('  GATE 2 FAIL: raw `pft` selected -- trainer wants the 8 one-hots')

# Negative control: this must NOT be the 28-column driver-only set. If the path
# ever slipped back to configs/selected_features_default_no_fe.json the run
# would look healthy and quietly repeat the basic experiment.
basic = {
    'soil_clay', 'LAI', 'ws', 'ta', 'volumetric_soil_water_layer_1_zscore',
    'mean_annual_precip', 'mean_annual_temp', 'slope', 'sw_in', 'soil_sand',
    'vpd', 'soil_temperature_level_1', 'aspect_sin', 'canopy_height',
    'ext_rad', 'day_length', 'ppfd_in', 'precip', 'elevation', 'Year sin',
    'MF', 'DNF', 'ENF', 'EBF', 'WSA', 'WET', 'DBF', 'SAV',
}
if set(feats) == basic:
    sys.exit('  GATE 2 FAIL: selection is identical to the 28-feature BASIC set '
             '-- this is not an FFS run')

json.dump({'selected_features': feats}, open(dst, 'w'), indent=2)
engineered = [c for c in feats if 'roll' in c or 'lag' in c or c.endswith('_std')]
print('  GATE 2 PASS: %d columns pinned to %s' % (len(feats), dst))
print('  best_score (mean-fold R2 on LOG1P): %.4f' % r['best_score'])
print('  lag/rolling columns selected: %d %s' % (len(engineered), sorted(engineered)))
PYEOF

echo
echo "=== GATE 3: all_biomes_merged double-count guard ==="
AGG="$DATA_DIR/daily/all_biomes_merged_daily.csv"
if [ -f "$AGG" ]; then
    if ! grep -q 'all_biomes_merged' "$TRAINER"; then
        echo "  GATE 3 FAIL: aggregate present but trainer no longer filters it"; exit 1
    fi
    echo "  GATE 3 PASS: aggregate present AND trainer filters it"
else
    echo "  GATE 3 NOTE: no aggregate present"
fi
N_SITE=$(ls "$DATA_DIR/daily"/*daily.csv 2>/dev/null | grep -vc all_biomes_merged)
echo "  site CSVs visible to the trainer: $N_SITE"
if [ "$N_SITE" -lt 200 ]; then echo "  GATE 3 FAIL: only $N_SITE site files"; exit 1; fi

echo
echo "=== GATE 4: run_id is fresh ==="
for d in "models/xgb/$RUN_ID" "outputs/models/xgb/$RUN_ID" \
         "outputs/plots/hyperparameter_optimization/xgb/$RUN_ID"; do
    if [ -d "$d" ]; then echo "  GATE 4 FAIL: $d already exists"; exit 1; fi
done
echo "  GATE 4 PASS: $RUN_ID is fresh"

echo
echo "=== TRAINING ==="
date
python -u "$TRAINER" \
    --model xgb --RANDOM_SEED 42 --n_groups 10 \
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
    --run_id "$RUN_ID"
TRAIN_EXIT=$?
echo "### training exit: $TRAIN_EXIT"
date
if [ $TRAIN_EXIT -ne 0 ]; then echo "STATUS: TRAINING FAILED"; exit $TRAIN_EXIT; fi

echo
echo "=== SITES DROPPED BY THE MISSING-COLUMN GUARD ==="
# The trainer logs via logging -> StreamHandler -> STDERR, so this must read the
# .err file. Grepping .out silently reports zero. A missing column drops the
# WHOLE SITE, so the feature list changes the site population, not just the
# columns -- and an FFS set is much wider than the basic one, so expect more.
ERRLOG="logs/sw5_train_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
echo "  sites loaded : $(grep -c 'Loaded data for site' "$ERRLOG" 2>/dev/null)"
echo "  sites skipped: $(grep -c 'Missing columns' "$ERRLOG" 2>/dev/null)"
grep -oE "Missing columns: \[[^]]*\]" "$ERRLOG" 2>/dev/null | sort | uniq -c | sort -rn | head -10

echo
echo "=== OOF SUMMARY (original scale) ==="
OOF="outputs/plots/hyperparameter_optimization/xgb/$RUN_ID/oof_predictions_$RUN_ID.csv"
python - "$OOF" "$NAME" <<'PYEOF'
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
print('  config        : %s' % sys.argv[2])
print('  n_rows        : %d' % len(df))
print('  n_sites       : %d' % df['site_id'].nunique())
print('  Var(y)        : %.4f' % np.var(y))
print('  mean(y)       : %.4f' % y.mean())
print('  pooled R2     : %.4f' % r2_score(y, p))
print('  RMSE          : %.4f' % np.sqrt(mean_squared_error(y, p)))
print('  MAE           : %.4f' % mean_absolute_error(y, p))
print('  median site R2: %.4f  (n=%d)' % (np.nanmedian(per), per.notna().sum()))
PYEOF

echo
echo "STATUS: DONE  task=$T  config=$NAME  run_id=$RUN_ID"
date
