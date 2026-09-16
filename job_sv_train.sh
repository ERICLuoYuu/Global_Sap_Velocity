#!/bin/bash
#SBATCH --job-name=sv_train
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=128G --time=08:00:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/sv_train_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/sv_train_%j.err

# Retrain sweep5_fullyear_fullday_tx_ffs_grid005_hohfix_sm under the SERVABLE feature contract.
#
# LEGACY TREES ONLY: the tree below is one job_sv_rebuild.sh produced from a site tree. A tree from
# today's merge is already servable -- point the trainer straight at it and skip the rebuild.
#
# Held fixed against the base run: the 50 feature names (read from the base FINAL_config),
# the trainer and every flag, grid-0.05 CV, the merged rows.
# Changed: the tree (soil moisture renormalised by the per-cell climatology) and
# --FEATURE_CONTRACT servable (precip memory from ERA5-Land, swc_layer3_norm from the
# climatology z-score).
# The site population can still move: a site the base run skipped only for lacking gauge
# precip now has ERA5 precip. The comparison therefore reports ALL sites and COMMON sites.
#
#   GATES_ONLY=1 sbatch -p express job_sv_train.sh

set -uo pipefail
hostname; date
cd /scratch/tmp/yluo2/gsv || exit 1
source .venv/bin/activate || exit 1
source ./sweep5_config.sh || exit 1
export ANALYSIS_SCALE=sapwood_combined
export PYTHONPATH=/scratch/tmp/yluo2/gsv
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

sweep5_config 1 || exit 1
[ "$TRAIN_METHOD" = "grid" ] || { echo "FATAL: GROUPING must be grid, got '$GROUPING'"; exit 1; }

BASE_RUN=sweep5_fullyear_fullday_tx_ffs_grid005_hohfix_sm
RUN_ID=${BASE_RUN}_sv
LEGACY_DIR=outputs/processed_data/sapwood_combined/merged/sweep_20260902_hohfix/fullyear_fullday_tx
DATA_DIR=outputs/processed_data/sapwood_combined/merged/sweep_20260902_hohfix_sv/fullyear_fullday_tx
BASE_CONFIG=models/xgb/$BASE_RUN/FINAL_config_$BASE_RUN.json
FEATURES=outputs/forward_selection/sweep5_fullyear_fullday_tx_grid005_hohfix_sm/train_features_${RUN_ID}.json
HYPER=src/hyperparameter_optimization/JSON/XGB_hyperparameters_fixed.json
TRAINER=src/hyperparameter_optimization/test_hyperparameter_tuning_ML_spatial_stratified.py
PLOTS=outputs/plots/hyperparameter_optimization/xgb
SHA="$(git rev-parse --short HEAD)+uncommitted($(git status --porcelain --untracked-files=no | wc -l))"

echo "=============================================================="
echo "  base run : $BASE_RUN"
echo "  run_id   : $RUN_ID"
echo "  data_dir : $DATA_DIR  (trainer appends /daily)"
echo "  contract : servable"
echo "  git      : $SHA"
echo "=============================================================="

echo
echo "=== GATE 1: dataset identity and feature contract (two-sided) ==="
sweep5_assert_provenance "$DATA_DIR" "$NAME" "$EXP_GS" "$EXP_DAY" "$EXP_DAYSAP" "$EXP_TX" || exit 1
python - "$DATA_DIR" "$LEGACY_DIR" <<'PYEOF' || exit 1
import sys
from src.feature_contract.definitions import assert_dataset_contract
servable_tree, legacy_tree = sys.argv[1], sys.argv[2]
assert_dataset_contract(servable_tree, "servable")
print("  GATE 1 PASS: rebuilt tree accepted under servable")
for tree, contract in ((legacy_tree, "servable"), (servable_tree, "site")):
    try:
        assert_dataset_contract(tree, contract)
    except ValueError:
        print("  GATE 1 PASS: %s refused under %s (negative control)" % (tree, contract))
    else:
        sys.exit("  GATE 1 FAIL: %s was ACCEPTED under %s" % (tree, contract))
PYEOF

echo
echo "=== GATE 2: pin the base run's exact feature list ==="
python - "$BASE_CONFIG" "$FEATURES" <<'PYEOF' || exit 1
import json
import sys
cfg = json.load(open(sys.argv[1]))
feats = [f for f in cfg["selected_features"] if f != "sap_velocity"]
if len(feats) != 50:
    sys.exit("  GATE 2 FAIL: base run used %d features, expected 50" % len(feats))
if sorted(feats) != sorted(cfg["feature_names"]):
    sys.exit("  GATE 2 FAIL: selected_features and feature_names disagree in the base config")
json.dump({"selected_features": feats, "source": sys.argv[1]}, open(sys.argv[2], "w"), indent=2)
changed = {
    "volumetric_soil_water_layer_1", "volumetric_soil_water_layer_4", "swc_lag3d", "ta_x_swc",
    "vpd_x_swc", "swc_layer3_norm", "precip_sum_7d", "days_since_rain",
}
print("  GATE 2 PASS: %d features pinned to %s" % (len(feats), sys.argv[2]))
print("  features whose definition changes: %s" % sorted(changed & set(feats)))
PYEOF

echo
echo "=== GATE 3: trainer exposes --FEATURE_CONTRACT ==="
if ! python "$TRAINER" --help 2>/dev/null | grep -q -- "--FEATURE_CONTRACT"; then
    echo "  GATE 3 FAIL: $TRAINER has no --FEATURE_CONTRACT"; exit 1
fi
echo "  GATE 3 PASS"

echo
echo "=== GATE 4: run_id is fresh; site files visible ==="
for d in "models/xgb/$RUN_ID" "outputs/models/xgb/$RUN_ID" "$PLOTS/$RUN_ID"; do
    if [ -d "$d" ]; then echo "  GATE 4 FAIL: $d already exists"; exit 1; fi
done
N_SITE=$(ls "$DATA_DIR/daily"/*daily.csv 2>/dev/null | grep -vc all_biomes_merged)
[ "$N_SITE" -ge 200 ] || { echo "  GATE 4 FAIL: only $N_SITE site files"; exit 1; }
echo "  GATE 4 PASS: $RUN_ID is fresh; $N_SITE site files"

if [ "${GATES_ONLY:-0}" = "1" ]; then echo "GATES_ONLY=1: stopping before training"; exit 0; fi

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
    --FEATURE_CONTRACT servable \
    --run_id "$RUN_ID"
TRAIN_EXIT=$?
echo "### training exit: $TRAIN_EXIT"
date
[ $TRAIN_EXIT -eq 0 ] || { echo "STATUS: TRAINING FAILED"; exit $TRAIN_EXIT; }

echo
echo "=== SITES DROPPED BY THE MISSING-COLUMN GUARD (trainer logs to stderr) ==="
ERRLOG="logs/sv_train_${SLURM_JOB_ID}.err"
echo "  sites loaded : $(grep -c 'Loaded data for site' "$ERRLOG" 2>/dev/null)"
echo "  sites skipped: $(grep -c 'Missing columns' "$ERRLOG" 2>/dev/null)"
grep -oE "Missing columns: \[[^]]*\]" "$ERRLOG" 2>/dev/null | sort | uniq -c | sort -rn | head -10

echo
echo "=== CONFIG RECORDS THE CONTRACT ==="
python - "$RUN_ID" "$BASE_CONFIG" <<'PYEOF' || exit 1
import json
import sys
run = sys.argv[1]
cfg = json.load(open("models/xgb/%s/FINAL_config_%s.json" % (run, run)))
base = json.load(open(sys.argv[2]))
if cfg.get("feature_contract") != "servable":
    sys.exit("  FAIL: FINAL_config feature_contract=%r" % cfg.get("feature_contract"))
if not str(cfg["split_config"]["data_dir"]).endswith("sweep_20260902_hohfix_sv/fullyear_fullday_tx/daily"):
    sys.exit("  FAIL: data_dir=%s" % cfg["split_config"]["data_dir"])
if sorted(cfg["feature_names"]) != sorted(base["feature_names"]):
    sys.exit("  FAIL: feature_names differ from the base run")
print("  PASS: feature_contract=servable, data_dir rebuilt tree, same 50 feature names")
PYEOF

echo
echo "=== BASE vs SERVABLE (OOF, original scale) ==="
python - "$PLOTS" "$BASE_RUN" "$RUN_ID" <<'PYEOF'
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

plots, base_run, new_run = sys.argv[1], sys.argv[2], sys.argv[3]

def load(run):
    return pd.read_csv("%s/%s/oof_predictions_%s.csv" % (plots, run, run))

def site_nse(g):
    obs, pred = g["observed"].to_numpy(), g["predicted"].to_numpy()
    denom = np.sum((obs - obs.mean()) ** 2)
    return np.nan if len(g) < 3 or denom == 0 else 1.0 - np.sum((obs - pred) ** 2) / denom

def summary(df):
    y, p = df["observed"].to_numpy(), df["predicted"].to_numpy()
    oracle = df.groupby("site_id")["observed"].transform("mean").to_numpy()
    nse = df.groupby("site_id").apply(site_nse)
    return len(df), df["site_id"].nunique(), r2_score(y, p), r2_score(y, oracle), float(np.nanmedian(nse))

base, new = load(base_run), load(new_run)
common = set(base["site_id"]) & set(new["site_id"])
print("  %-14s %7s %6s %8s %8s %8s %8s" % ("", "rows", "sites", "pooled", "oracle", "margin", "medNSE"))
for label, frame in (
    ("base   all", base),
    ("sv     all", new),
    ("base   common", base[base["site_id"].isin(common)]),
    ("sv     common", new[new["site_id"].isin(common)]),
):
    n, s, pooled, oracle, med = summary(frame)
    print("  %-14s %7d %6d %8.4f %8.4f %8.4f %8.4f" % (label, n, s, pooled, oracle, pooled - oracle, med))
print("  sites only in base: %s" % sorted(set(base["site_id"]) - common))
print("  sites only in sv  : %s" % sorted(set(new["site_id"]) - common))
PYEOF

echo
echo "STATUS: DONE  run_id=$RUN_ID"
date
