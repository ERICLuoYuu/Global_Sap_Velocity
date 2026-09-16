#!/bin/bash
#SBATCH --job-name=m5_pin
#SBATCH --partition=express
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2
#SBATCH --mem=16G --time=00:30:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/m5_pin_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/m5_pin_%j.err

# Stage 0 of a FIVE-MODEL comparison on ONE dataset:
#   fullyear_fullday_tx  x  grid005  x  hohfix  x  _sm   (uncapped FFS, 50 cols)
#
# Pins ONE feature file that all five models (xgb, rf, ann, lstm, cnn-lstm)
# read.  The point of the comparison is that the feature set is held FIXED, so
# it must be impossible for the five to drift apart in what they were trained
# on -- including by someone re-running stage 2 halfway through.  The pin is
# written once here, hashed, and every downstream job verifies the hash before
# it trains.  A gate that each job computes for itself cannot prove agreement
# BETWEEN jobs; a shared hash can.
#
# FEATURE CONTRACT (2026-09-16). FEATURE_CONTRACT comes from sweep5_config.sh (default servable) and
# reaches both trainers. GATE 1b here and GATE 0 in the ml/dl jobs assert the tree was built under it,
# and GATE 2 refuses a selection that either trainer would refuse. As encoded, this comparison cannot
# run under EITHER contract:
#   * hohfix is a SITE tree (gauge precipitation), so the servable default refuses it at GATE 1b;
#   * its 50-column selection names swc_layer3_norm and the bare volumetric_soil_water_layer_1 and _4,
#     all retired on 2026-09-14, so GATE 2 refuses it under site as well.
# Re-running it takes a fresh selection, and a GATE 2 identity (the 50-column count) to match it.

set -uo pipefail
hostname; date

cd /scratch/tmp/yluo2/gsv || exit 1
source .venv/bin/activate || exit 1

# Reuse the sweep's own dataset table rather than restating paths here.  Every
# path defect in this repo has come from a second copy of a table drifting from
# the first while the summary line stayed identical.
export SWEEP_NAME=sweep_20260902_hohfix
export GROUPING=grid
export FFS_TAG=_sm
source ./sweep5_config.sh || exit 1
sweep5_config 1 || exit 1          # task 1 == fullyear_fullday_tx

export ANALYSIS_SCALE=sapwood_combined
export PYTHONPATH=/scratch/tmp/yluo2/gsv:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

M5_TAG="m5_${GROUP_TAG}${SWEEP_TAG}${FFS_TAG}"
FFS_JSON="$FFS_OUT/ffs_mean_r2_results.json"
FEATURES="$FFS_OUT/train_features_${M5_TAG}.json"
ML_TRAINER=src/hyperparameter_optimization/test_hyperparameter_tuning_ML_spatial_stratified.py
DL_TRAINER=src/hyperparameter_optimization/test_hyperparameter_tuning_DL_spatial_stratified.py

echo "=============================================================="
echo "Five-model comparison -- feature pin"
echo "  config    : $NAME"
echo "  data_dir  : $DATA_DIR   (trainers append /daily)"
echo "  ffs run   : $FFS_OUT"
echo "  pin file  : $FEATURES"
echo "  contract  : $FEATURE_CONTRACT"
echo "  git sha   : $(git rev-parse --short HEAD)"
echo "=============================================================="

echo
echo "=== GATE 1: dataset identity ==="
sweep5_assert_provenance "$DATA_DIR" "$NAME" "$EXP_GS" "$EXP_DAY" "$EXP_DAYSAP" "$EXP_TX" || exit 1

echo
echo "=== GATE 1b: the tree was built under FEATURE_CONTRACT=$FEATURE_CONTRACT ==="
sweep5_assert_contract "$DATA_DIR" "$FEATURE_CONTRACT" || exit 1

echo
echo "=== GATE 2: pin the FFS selection ==="
python - "$FFS_JSON" "$FEATURES" "$FEATURE_CONTRACT" <<'PYEOF' || exit 1
import json, sys
from src.feature_contract.definitions import assert_no_retired_swc_features, assert_no_servable_duplicate_features
src, dst, contract = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    r = json.load(open(src))
except FileNotFoundError:
    sys.exit('  GATE 2 FAIL: %s not found' % src)
feats = r.get('selected_features') or []
if not feats:
    sys.exit('  GATE 2 FAIL: no selected_features in %s' % src)

# A list the trainers would refuse must not be pinned. Both refuse a retired soil-moisture name -- a
# column the merged CSVs still carry, so it never fails as missing -- and, under servable, the ERA5
# metres twin of precip. Checked here, before anything is written, so the refusal costs seconds.
try:
    assert_no_retired_swc_features(feats, src)
    assert_no_servable_duplicate_features(feats, src, contract)
except ValueError as exc:
    sys.exit('  GATE 2 FAIL: %s' % exc)

# This comparison is specifically the UNCAPPED (6,70) selection.  Its recorded
# size is 50 columns / 43 groups.  If the file on disk is not that, the path has
# slipped to a sibling run and every one of the five models would train on the
# wrong set while looking perfectly healthy.
if len(feats) != 50:
    sys.exit('  GATE 2 FAIL: expected the 50-column uncapped set, got %d' % len(feats))
if 'sw_in' not in feats:
    sys.exit('  GATE 2 FAIL: sw_in not selected -- the ML trainer will raise')
if 'pft' in feats:
    sys.exit('  GATE 2 FAIL: raw `pft` selected -- trainers want the 8 one-hots')
for c in ('MF', 'DNF', 'ENF', 'EBF', 'WSA', 'WET', 'DBF', 'SAV'):
    if c not in feats:
        sys.exit('  GATE 2 FAIL: PFT one-hot %r missing' % c)

# Negative control: the 28-column driver-only BASIC set.  If the path ever
# slipped back to configs/selected_features_default_no_fe.json the run would
# look healthy and quietly repeat the basic experiment.
basic = {
    'soil_clay', 'LAI', 'ws', 'ta', 'volumetric_soil_water_layer_1_zscore',
    'mean_annual_precip', 'mean_annual_temp', 'slope', 'sw_in', 'soil_sand',
    'vpd', 'soil_temperature_level_1', 'aspect_sin', 'canopy_height',
    'ext_rad', 'day_length', 'ppfd_in', 'precip', 'elevation', 'Year sin',
    'MF', 'DNF', 'ENF', 'EBF', 'WSA', 'WET', 'DBF', 'SAV',
}
if set(feats) == basic:
    sys.exit('  GATE 2 FAIL: selection is the 28-feature BASIC set, not an FFS run')

# Second negative control: the CAPPED k40 sibling is 32 columns.  Distinct size,
# so the length check above already separates them, but name it explicitly so a
# future edit to the length cannot silently admit it.
if len(feats) == 32:
    sys.exit('  GATE 2 FAIL: this is the k40 capped selection, not the uncapped one')

json.dump({'selected_features': feats}, open(dst, 'w'), indent=2)
eng = [c for c in feats if 'roll' in c or 'lag' in c or c.endswith('_std')]
print('  GATE 2 PASS: %d columns pinned to %s' % (len(feats), dst))
print('  best_score (mean-fold R2 on LOG1P): %.4f' % r['best_score'])
print('  lag/rolling columns: %d %s' % (len(eng), sorted(eng)))
PYEOF

sha256sum "$FEATURES" > "${FEATURES}.sha256" || exit 1
echo "  pin hash: $(cut -c1-16 < "${FEATURES}.sha256")..."

echo
echo "=== GATE 3: all_biomes_merged double-count guard (BOTH trainers) ==="
AGG="$DATA_DIR/daily/all_biomes_merged_daily.csv"
if [ -f "$AGG" ]; then
    for t in "$ML_TRAINER" "$DL_TRAINER"; do
        if ! grep -q 'all_biomes_merged' "$t"; then
            echo "  GATE 3 FAIL: aggregate present but $t does not filter it"; exit 1
        fi
    done
    echo "  GATE 3 PASS: aggregate present AND both trainers filter it"
else
    echo "  GATE 3 NOTE: no aggregate present"
fi
N_SITE=$(ls "$DATA_DIR/daily"/*daily.csv 2>/dev/null | grep -vc all_biomes_merged)
echo "  site CSVs visible to the trainers: $N_SITE"
if [ "$N_SITE" -lt 200 ]; then echo "  GATE 3 FAIL: only $N_SITE site files"; exit 1; fi

echo
echo "=== GATE 4: hyperparameter files for all five models ==="
for h in XGB RF ANN LSTM CNN-LSTM; do
    f="src/hyperparameter_optimization/JSON/${h}_hyperparameters_fixed.json"
    [ -f "$f" ] || { echo "  GATE 4 FAIL: missing $f"; exit 1; }
    echo "  ok: $f"
done

echo
echo "=== GATE 5: window/label alignment probe (empirical, not inferred) ==="
# The index math in WindowGenerator says the label sits `shift` steps PAST the
# last input step, while generate_windowed_feature_names labels the last input
# step '_t-0' as though it were contemporaneous with the label.  Both cannot be
# true.  Build one window from a synthetic ramp and read off which rows landed
# where, so the answer is measured rather than argued.
python - <<'PYEOF'
import numpy as np, pandas as pd
from src.hyperparameter_optimization.timeseries_processor import WindowGenerator
n = 8
df = pd.DataFrame({
    'drv':          np.arange(n, dtype='float32'),          # row index
    'sap_velocity': np.arange(100, 100 + n, dtype='float32'),  # 100 + row index
})
for iw in (2, 1):
    wg = WindowGenerator(input_width=iw, label_width=1, shift=1, data_df=df,
                         label_columns=['sap_velocity'], batch_size=1, shuffle=False,
                         exclude_targets_from_features=True,
                         exclude_labels_from_inputs=True)
    print('  INPUT_WIDTH=%d  total=%d  input_indices=%s  label_indices=%s'
          % (iw, wg.total_window_size, wg.input_indices, wg.label_indices))
    for x, y in wg.dataset.take(1):
        drv = x.numpy().ravel()
        lab = y.numpy().ravel()
        print('    first window: driver rows %s -> label sap %s (row %d)'
              % (drv.astype(int).tolist(), lab.astype(int).tolist(), int(lab[0]) - 100))
        gap = (int(lab[0]) - 100) - int(drv[-1])
        print('    => label is %d step(s) PAST the last input step' % gap)
PYEOF

echo
echo "STATUS: PIN DONE"
echo "  features : $FEATURES"
echo "  sha256   : ${FEATURES}.sha256"
date
