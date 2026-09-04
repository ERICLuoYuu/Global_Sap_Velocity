#!/bin/bash
#SBATCH --job-name=sw5_sel
#SBATCH --partition=zen4,zen5,zen4x
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=64
#SBATCH --mem=96G --time=2-00:00:00
#SBATCH --array=1-5
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/sw5_sel_%A_%a.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/sw5_sel_%A_%a.err

# Stage 2 of 3: forward feature selection, mean_r2, PFT-stratified, on the cache
# from stage 1. One array task per merge configuration. Whether that cache is
# grouped by 0.05 grid cell or by site is fixed by GROUPING and asserted, in
# both directions, by GATE 1.
#
# K_FEATURES is a knob; the default is the pair 6,70. A (min,max) PAIR walks the
# path and returns the k with the BEST CV score, so the size is chosen by the
# data. A bare INTEGER returns exactly that many features whatever the scores
# did -- the earlier k=40 arms were FORCED to 40, they were not optimal at 40.
# GATE 0 refuses a bare integer unless ALLOW_FORCED_K=1, because an integer is
# also what sbatch delivers when it eats the comma in an --export list:
#   --export=ALL,K_FEATURES=6,40   arrives as   K_FEATURES=6
#
# The two forms cost the SAME. Greedy forward selection (floating=False, never
# overridden here) must pass through every smaller k either way; the bound only
# decides where the walk stops and which k is returned. So the path to 40 is a
# strict PREFIX of the path to 70, and because selector.py persists
# step_features[k] and a per-step pooled R2 for EVERY k, a 6,70 run already
# contains every capped answer below it. A capped re-run buys latency, not
# information.
#
# 70 rather than the full ~141 is purely the wall. MEASURED, not modelled:
# 12 h 01 m for the full 6..70 path (46594721_1, 64 cores, zen4) and 6 h 13 m
# for 6..40, against a hard 48 h partition limit. The old ~(0.67 + 0.31k) s
# per-fit model integrates to ~21 h at k=70 and so runs ~1.75x hot -- do NOT
# size a wall off it. mlxtend emits results only at the end, so a timeout
# writes NOTHING; keep the margin.
#
# Those wall times are CONTENTION-DEPENDENT, not intrinsic. zen4 nodes are
# 192-core and Slurm packs three 64-core jobs onto one, so the arms share
# memory bandwidth with whatever else lands there. Observed spread on the SAME
# cache and config: 12 h 01 m for a 6..70 path alone-ish on r19n20, versus a
# STRICTLY SHORTER 6..40 path still running at 18 h on a three-way-packed
# r13n42. Treat 12 h as a floor, not an estimate.
#
# 70 is comfortably past the plateau and has been shown not to bind: on
# fullyear_fullday_tx the argmax was k=43 (interior), the curve was flat from
# k=25 (mean_r2 0.5491, identical to the argmax) and DECLINING to 0.5348 by
# k=70. If an arm ever returns exactly k=70 the bound DID bind, the result is
# not an optimum, and it must be re-run wider rather than reported.
#
# 16 SFS candidates x 4 XGBoost threads = 64, matching cpus-per-task. A 192-core
# request for a ~90 MB design matrix once sat in the queue for two days; size the
# request to the problem, not to the node. zen4/zen5/zen4x are the same family
# with a 2-day wall -- let Slurm take whichever frees first.

set -uo pipefail
hostname; date

cd /scratch/tmp/yluo2/gsv || exit 1
source .venv/bin/activate || exit 1
source ./sweep5_config.sh || exit 1

# Search bound, passed straight through as mlxtend k_features. A (min,max)
# PAIR walks the path and returns the best-scoring k; a bare INTEGER forces
# that size whatever the scores did. The two cost the same -- greedy forward
# selection must pass through every smaller k either way -- so the pair is
# strictly more informative and is what to prefer for a cap.
#   sbatch --export=ALL,K_FEATURES=6,40 ... job_sweep5_select.sh
K_FEATURES="${K_FEATURES:-6,70}"

# GATE 0: K_FEATURES is a pair, a keyword, or a DELIBERATE forced size.
# A bare integer forces that many features regardless of score, and is also
# exactly what sbatch produces when it eats the comma in an --export list:
#   --export=ALL,K_FEATURES=6,40   arrives as   K_FEATURES=6
# Nothing downstream can tell those two apart, so refuse the ambiguous form.
# Pass a pair by exporting it in the submitting shell instead:
#   export K_FEATURES=6,40; sbatch --export=ALL,GROUPING=grid ... 
case "$K_FEATURES" in
  *,*|best|parsimonious) ;;
  *)
    if [ "${ALLOW_FORCED_K:-0}" != 1 ]; then
      echo "  GATE 0 FAIL: K_FEATURES='$K_FEATURES' is a bare integer."
      echo "    That FORCES the size, and is the signature of a comma eaten by sbatch --export."
      echo "    Use: export K_FEATURES=6,40 ; sbatch --export=ALL,... "
      echo "    Set ALLOW_FORCED_K=1 if a forced size is genuinely intended."
      exit 1
    fi
    echo "  GATE 0: forced size $K_FEATURES accepted (ALLOW_FORCED_K=1)"
    ;;
esac
echo "  GATE 0 PASS: K_FEATURES=$K_FEATURES"

export ANALYSIS_SCALE=sapwood_combined
export PYTHONPATH=/scratch/tmp/yluo2/gsv:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

T="${SLURM_ARRAY_TASK_ID:?not running as an array task}"
sweep5_config "$T" || exit 1
CACHE="$CACHE_DIR/feature_cache.npz"

echo "=============================================================="
echo "FFS selection  task=$T  config=$NAME"
echo "  cache   : $CACHE"
echo "  output  : $FFS_OUT"
echo "  cores   : ${SLURM_CPUS_PER_TASK:-?}  partition: ${SLURM_JOB_PARTITION:-?}"
echo "  git sha : $(git rev-parse --short HEAD)"
echo "=============================================================="

echo
echo "=== GATE 1: the cache is the one stage 1 built, with the design we want ==="
# A stale or biome-stratified cache is the quiet failure here: a full day of
# compute reporting a clean result for the wrong design. Refuse to start.
python - "$CACHE" "$GROUPING" <<'PYEOF' || exit 1
import sys
import numpy as np
# allow_pickle: our own cache, written by stage 1 of this sweep on private scratch.
d = np.load(sys.argv[1], allow_pickle=True)
cats = sorted(str(c) for c in d['pft_categories'])
uniq = sorted(set(d['groups'].tolist()))
print('  X       : %s' % (d['X'].shape,))
print('  groups  : %d' % len(uniq))
print('  classes : %s' % cats)
if set(cats) != {'MF', 'DNF', 'ENF', 'EBF', 'WSA', 'WET', 'DBF', 'SAV'}:
    sys.exit('  GATE 1 FAIL: not PFT-stratified')
# Contiguous runs, not id sparsity, is the discriminator: BOTH methods densify
# ids to 0..n-1. Groups come from ONE lat/lon per SITE and rows are concatenated
# site by site, so under 'default' site->group is a bijection and every group is
# exactly one contiguous run (runs == n_groups); under 'grid' co-located sites
# share a group but are not adjacent in site order, so runs > n_groups.
# Asserting whichever relation the REQUESTED method implies makes this two-sided:
# it catches a site-grouped cache reaching a grid run AND a grid-grouped cache
# reaching a site run. The old one-sided form could only see the first.
want = sys.argv[2]
g = d['groups']
runs = 1 + int((g[1:] != g[:-1]).sum())
if want == 'grid' and runs <= len(uniq):
    sys.exit('  GATE 1 FAIL: runs == n_groups (%d) -- cache is site-grouped, '
             'GROUPING=grid was asked for' % len(uniq))
if want == 'site' and runs != len(uniq):
    sys.exit('  GATE 1 FAIL: %d runs over %d groups -- co-located sites were '
             'merged, so this cache is grid-grouped, not site' % (runs, len(uniq)))
print('  GATE 1 PASS: GROUPING=%s, %d groups over %d contiguous site runs'
      % (want, len(uniq), runs))
PYEOF

echo
echo "=== GATE 2b: every mandatory feature is actually IN the cache ==="
# build_feature_groups() looks each mandatory name up with `if feat in
# name_to_idx` and SILENTLY SKIPS the ones it cannot find. A mandatory feature
# missing from the cache therefore does not raise -- it just is not forced, and
# the run reports a clean selection for a design nobody chose. That is exactly
# how volumetric_soil_water_layer_1 could be promoted to mandatory and still be
# absent from every model. Check the names against the cache before spending a
# day on it.
python - "$CACHE" <<'PYEOF' || exit 1
import sys
import numpy as np
sys.path.insert(0, ".")
from src.forward_selection.feature_registry import MANDATORY_FEATURES, build_feature_groups

d = np.load(sys.argv[1], allow_pickle=True)
names = [str(n) for n in d["feature_names"]]
missing = [f for f in MANDATORY_FEATURES if f not in names]
print("  mandatory wanted (%d): %s" % (len(MANDATORY_FEATURES), list(MANDATORY_FEATURES)))
if missing:
    sys.exit("  GATE 2b FAIL: absent from the cache, would be silently unforced: %s" % missing)

mandatory_idx, candidate_groups = build_feature_groups(names)
if len(mandatory_idx) != len(MANDATORY_FEATURES):
    sys.exit("  GATE 2b FAIL: %d of %d mandatory names resolved to an index"
             % (len(mandatory_idx), len(MANDATORY_FEATURES)))

# The partition property, asserted against the REAL cache rather than the unit
# test's synthetic column list: a duplicated index double-counts a feature, an
# orphaned one is dropped from the design matrix. Both are silent.
seen = list(mandatory_idx) + [i for g in candidate_groups for i in g]
dupes = sorted({names[i] for i in seen if seen.count(i) > 1})
orphans = sorted({names[i] for i in set(range(len(names))) - set(seen)})
if dupes:
    sys.exit("  GATE 2b FAIL: duplicated across mandatory/candidates: %s" % dupes)
if orphans:
    sys.exit("  GATE 2b FAIL: in the cache but in no group, silently dropped: %s" % orphans)
print("  GATE 2b PASS: %d mandatory + %d candidate groups tile %d columns exactly"
      % (len(mandatory_idx), len(candidate_groups), len(names)))
PYEOF

echo
echo "=== GATE 2: results are fresh ==="
if [ -f "$FFS_OUT/ffs_mean_r2_results.json" ]; then
    echo "  GATE 2 FAIL: $FFS_OUT/ffs_mean_r2_results.json already exists"; exit 1
fi
echo "  GATE 2 PASS"

echo
echo "=== SELECTION ==="
date
python -u -m src.forward_selection.run_selection \
    --scoring mean_r2 \
    --cache_path "$CACHE" \
    --output_dir "$FFS_OUT" \
    --time_scale daily \
    --n_splits 10 \
    --k_features "$K_FEATURES" \
    --random_seed 42 \
    --n_jobs_sfs 16 \
    --n_jobs_xgb 4
SEL_EXIT=$?
echo "### selection exit: $SEL_EXIT"
date
if [ $SEL_EXIT -ne 0 ]; then echo "STATUS: SELECTION FAILED"; exit $SEL_EXIT; fi

echo
echo "=== GATE 3: the result is usable by the trainer ==="
python - "$FFS_OUT/ffs_mean_r2_results.json" <<'PYEOF' || exit 1
import json, sys
r = json.load(open(sys.argv[1]))
feats = r.get('selected_features') or []
if not feats:
    sys.exit('  GATE 3 FAIL: no selected_features in the results JSON')
# The trainer raises on a missing sw_in; fail here with a readable message.
if 'sw_in' not in feats:
    sys.exit('  GATE 3 FAIL: sw_in not selected -- the trainer will raise')
# The 8 PFT one-hots enter as a mandatory group; a raw `pft` column is rejected.
if 'pft' in feats:
    sys.exit('  GATE 3 FAIL: raw `pft` column selected -- trainer wants the 8 one-hots')
onehot = {'MF', 'DNF', 'ENF', 'EBF', 'WSA', 'WET', 'DBF', 'SAV'}
missing = onehot - set(feats)
if missing:
    sys.exit('  GATE 3 FAIL: PFT one-hots missing from the selection: %s' % sorted(missing))
print('  GATE 3 PASS: %d columns selected' % len(feats))
PYEOF

echo
echo "=== WHERE THE CURVE ACTUALLY PLATEAUS ==="
# best_score / step_scores are R2 on log1p(sap_velocity); step_pooled_r2 is R2
# on the ORIGINAL cm3 cm-2 h-1 scale. Adjacent keys, different targets -- two
# thirds of the apparent mean-vs-pooled gap is that scale change, not
# aggregation. Report both, labelled, and never compare them directly.
python - "$FFS_OUT/ffs_mean_r2_results.json" "$NAME" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
print('  config        : %s' % sys.argv[2])
scales = r.get('metric_scales')
print('  metric_scales : %s' % (scales if scales else 'ABSENT (pre-2026-08-25 selector)'))
print('  n_selected    : %d columns  (k_features counts the non-mandatory ones)'
      % len(r['selected_features']))
print('  best_score    : %.4f   <- mean-fold R2 on LOG1P, the flattering number'
      % r['best_score'])
pooled = {int(k): v for k, v in (r.get('step_pooled_r2') or {}).items()}
if pooled:
    best_k = max(pooled, key=pooled.get)
    print('  best pooled R2 (ORIGINAL scale): %.4f at k=%d' % (pooled[best_k], best_k))
    print('  pooled R2 by step:')
    for k in sorted(pooled):
        print('    k=%-3d %.4f%s' % (k, pooled[k], '  <== best' if k == best_k else ''))
    print('  NOTE: selected_features is the MEAN-FOLD best, not the pooled best.')
else:
    print('  (no step_pooled_r2 in results)')
print('  selected: %s' % sorted(r['selected_features']))
PYEOF

echo
echo "STATUS: DONE  task=$T  config=$NAME  results=$FFS_OUT/ffs_mean_r2_results.json"
date
