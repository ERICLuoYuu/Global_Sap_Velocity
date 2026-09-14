#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=32G --time=04:00:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/swc_train_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/swc_train_%j.err

# One soil-moisture scale for EVERY soil-moisture feature, on top of the servable retrain.
#
#   SWC_SCALE=zscore sbatch --job-name=swc_zscore job_swc_scale_train.sh
#   SWC_SCALE=raw    sbatch --job-name=swc_raw    job_swc_scale_train.sh
#
# Held fixed against the servable run (..._sv, --SWC_SCALE ratio): the tree, the 50 feature
# names, every trainer flag, grid-0.05 CV, --FEATURE_CONTRACT servable. Changed ONLY:
#   zscore  volumetric_soil_water_layer_1/_4 (and swc_lag3d, ta_x_swc, vpd_x_swc built from
#           layer 1) = (raw - mu_clim) / sigma_clim; swc_layer3_norm unchanged
#   raw     all of those AND swc_layer3_norm = raw m3/m3
# The site set must equal the servable run's, so its StratifiedGroupKFold partition is the same
# and the two are comparable site for site. That is checked, not assumed.

set -uo pipefail
hostname; date
cd /scratch/tmp/yluo2/gsv || exit 1
source .venv/bin/activate || exit 1
source ./sweep5_config.sh || exit 1
export ANALYSIS_SCALE=sapwood_combined
export PYTHONPATH=/scratch/tmp/yluo2/gsv
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0

case "${SWC_SCALE:-}" in
  zscore|raw) ;;
  *) echo "FATAL: SWC_SCALE must be zscore or raw, got '${SWC_SCALE:-}'"; exit 2 ;;
esac

sweep5_config 1 || exit 1
[ "$TRAIN_METHOD" = "grid" ] || { echo "FATAL: GROUPING must be grid, got '$GROUPING'"; exit 1; }

BASE_RUN=sweep5_fullyear_fullday_tx_ffs_grid005_hohfix_sm
SV_RUN=${BASE_RUN}_sv
SV_ERRLOG=logs/sv_train_46722298.err
RUN_ID=${SV_RUN}_${SWC_SCALE}
DATA_DIR=outputs/processed_data/sapwood_combined/merged/sweep_20260902_hohfix_sv/fullyear_fullday_tx
BASE_CONFIG=models/xgb/$BASE_RUN/FINAL_config_$BASE_RUN.json
SV_CONFIG=models/xgb/$SV_RUN/FINAL_config_$SV_RUN.json
FEATURES=outputs/forward_selection/sweep5_fullyear_fullday_tx_grid005_hohfix_sm/train_features_${RUN_ID}.json
HYPER=src/hyperparameter_optimization/JSON/XGB_hyperparameters_fixed.json
TRAINER=src/hyperparameter_optimization/test_hyperparameter_tuning_ML_spatial_stratified.py
PLOTS=outputs/plots/hyperparameter_optimization/xgb
SHA="$(git rev-parse --short HEAD)+uncommitted($(git status --porcelain --untracked-files=no | wc -l))"

echo "=============================================================="
echo "  reference : $SV_RUN (swc scale ratio)"
echo "  run_id    : $RUN_ID"
echo "  swc scale : $SWC_SCALE"
echo "  data_dir  : $DATA_DIR"
echo "  git       : $SHA"
echo "=============================================================="

echo
echo "=== GATE 1: dataset identity and contract ==="
sweep5_assert_provenance "$DATA_DIR" "$NAME" "$EXP_GS" "$EXP_DAY" "$EXP_DAYSAP" "$EXP_TX" || exit 1
python -c "from src.feature_contract.definitions import assert_dataset_contract as a; a('$DATA_DIR', 'servable'); print('  GATE 1 PASS: tree accepted under servable')" || exit 1

echo
echo "=== GATE 2: pin the base run's exact feature list (same as the servable run) ==="
python - "$BASE_CONFIG" "$SV_CONFIG" "$FEATURES" <<'PYEOF' || exit 1
import json
import sys
base, sv = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
feats = [f for f in base["selected_features"] if f != "sap_velocity"]
if len(feats) != 50 or sorted(feats) != sorted(sv["feature_names"]):
    sys.exit("  GATE 2 FAIL: the base feature list (%d) is not the servable run's" % len(feats))
if sv.get("feature_contract") != "servable":
    sys.exit("  GATE 2 FAIL: reference run is not servable: %r" % sv.get("feature_contract"))
json.dump({"selected_features": feats, "source": sys.argv[1]}, open(sys.argv[3], "w"), indent=2)
print("  GATE 2 PASS: %d features pinned to %s" % (len(feats), sys.argv[3]))
PYEOF

echo
echo "=== GATE 3: trainer exposes --FEATURE_CONTRACT and --SWC_SCALE ==="
HELP=$(python "$TRAINER" --help 2>/dev/null)
for flag in --FEATURE_CONTRACT --SWC_SCALE; do
    echo "$HELP" | grep -q -- "$flag" || { echo "  GATE 3 FAIL: no $flag"; exit 1; }
done
echo "  GATE 3 PASS"

echo
echo "=== GATE 4: run_id is fresh; reference log present ==="
for d in "models/xgb/$RUN_ID" "outputs/models/xgb/$RUN_ID" "$PLOTS/$RUN_ID"; do
    if [ -d "$d" ]; then echo "  GATE 4 FAIL: $d already exists"; exit 1; fi
done
REF_SKIPPED=$(grep -c 'Missing columns' "$SV_ERRLOG" 2>/dev/null)
[ -n "$REF_SKIPPED" ] && [ "$REF_SKIPPED" -gt 0 ] || { echo "  GATE 4 FAIL: no skip count in $SV_ERRLOG"; exit 1; }
echo "  GATE 4 PASS: $RUN_ID is fresh; reference run skipped $REF_SKIPPED sites"

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
    --SWC_SCALE "$SWC_SCALE" \
    --run_id "$RUN_ID"
TRAIN_EXIT=$?
echo "### training exit: $TRAIN_EXIT"
date
[ $TRAIN_EXIT -eq 0 ] || { echo "STATUS: TRAINING FAILED"; exit $TRAIN_EXIT; }

echo
echo "=== SITE POPULATION MUST MATCH THE SERVABLE RUN ==="
ERRLOG="logs/swc_train_${SLURM_JOB_ID}.err"
SKIPPED=$(grep -c 'Missing columns' "$ERRLOG" 2>/dev/null)
echo "  sites loaded : $(grep -c 'Loaded data for site' "$ERRLOG" 2>/dev/null)"
echo "  sites skipped: $SKIPPED (reference $REF_SKIPPED)"
grep -iE "swc scale|has no .* column to take" "$ERRLOG" | head -3
if [ "$SKIPPED" != "$REF_SKIPPED" ]; then echo "  FAIL: skipped-site count differs"; exit 1; fi

echo
echo "=== CONFIG RECORDS THE SCALE ==="
python - "$RUN_ID" "$SV_CONFIG" "$SWC_SCALE" <<'PYEOF' || exit 1
import json
import sys
run, scale = sys.argv[1], sys.argv[3]
cfg = json.load(open("models/xgb/%s/FINAL_config_%s.json" % (run, run)))
sv = json.load(open(sys.argv[2]))
if cfg.get("feature_contract") != "servable" or cfg.get("swc_scale") != scale:
    sys.exit("  FAIL: feature_contract=%r swc_scale=%r" % (cfg.get("feature_contract"), cfg.get("swc_scale")))
if cfg["split_config"]["data_dir"] != sv["split_config"]["data_dir"]:
    sys.exit("  FAIL: data_dir differs from the servable run")
if sorted(cfg["feature_names"]) != sorted(sv["feature_names"]):
    sys.exit("  FAIL: feature_names differ from the servable run")
print("  PASS: servable, swc_scale=%s, same tree and 50 feature names as %s" % (scale, sys.argv[2]))
PYEOF

echo
echo "=== BASE vs SERVABLE ratio vs SERVABLE $SWC_SCALE (OOF, original scale) ==="
python - "$PLOTS" "$BASE_RUN" "$SV_RUN" "$RUN_ID" "$SWC_SCALE" <<'PYEOF'
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

plots, runs, scale = sys.argv[1], sys.argv[2:5], sys.argv[5]
labels = ["base (site)", "sv ratio", "sv " + scale]


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
    return len(df), df["site_id"].nunique(), r2_score(y, p), r2_score(y, oracle), float(np.nanmedian(nse)), nse


frames = [load(r) for r in runs]
common = set.intersection(*(set(f["site_id"]) for f in frames))
print("  %-18s %7s %6s %8s %8s %8s %8s" % ("", "rows", "sites", "pooled", "oracle", "margin", "medNSE"))
nse_by = {}
for scope in ("all", "common"):
    for label, frame in zip(labels, frames):
        f = frame if scope == "all" else frame[frame["site_id"].isin(common)]
        n, s, pooled, oracle, med, nse = summary(f)
        nse_by[(label, scope)] = nse
        print("  %-18s %7d %6d %8.4f %8.4f %8.4f %8.4f" % (label + " " + scope, n, s, pooled, oracle, pooled - oracle, med))
sv_sites, new_sites = set(frames[1]["site_id"]), set(frames[2]["site_id"])
print("  sites only in sv ratio: %s" % sorted(sv_sites - new_sites))
print("  sites only in sv %s: %s" % (scale, sorted(new_sites - sv_sites)))
d = (nse_by[(labels[2], "common")] - nse_by[(labels[1], "common")]).dropna()
print("  per-site NSE, sv %s minus sv ratio (%d sites): median %+.4f, better %d, worse %d"
      % (scale, len(d), d.median(), int((d > 0).sum()), int((d < 0).sum())))
PYEOF

echo
echo "STATUS: DONE  run_id=$RUN_ID"
date
