#!/bin/bash
#SBATCH --job-name=sw5_cache
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=128G --time=02:00:00
#SBATCH --array=1-5
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/sw5_cache_%A_%a.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/sw5_cache_%A_%a.err

# Stage 1 of 3: FFS feature cache for each of the five merge configurations
# built by job_merge_sweep5.sh (array 46585310, sweep_20260902).
#
# ONE cache per dataset, grid-grouped at 0.05 deg. Why only grid: the cache's
# `groups` array IS the FFS cross-validation design and stage 2 cannot override
# it. Per-site grouping is the leaky one (+0.145 pooled R2 measured on identical
# data) and selecting under it picks the features a neighbouring stand makes
# cheap -- lags and rolling means. So select once, honestly.
#
# 128G/2h rather than the 96G/1h the growing-season pair used: configs 1-3 are
# FULL YEAR, roughly 1.8x the rows of a GS tree (2.1 GB vs 1.2 GB of daily/).

set -uo pipefail
hostname; date

cd /scratch/tmp/yluo2/gsv || exit 1
source .venv/bin/activate || exit 1
source ./sweep5_config.sh || exit 1

export ANALYSIS_SCALE=sapwood_combined
export PYTHONPATH=/scratch/tmp/yluo2/gsv:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

T="${SLURM_ARRAY_TASK_ID:?not running as an array task}"
sweep5_config "$T" || exit 1
mkdir -p "$CACHE_DIR"

echo "=============================================================="
echo "FFS cache  task=$T  config=$NAME"
echo "  data_dir : $DATA_DIR   (run_selection appends /daily)"
echo "  output   : $CACHE_DIR"
echo "  git sha  : $(git rev-parse --short HEAD)"
echo "=============================================================="

echo
echo "=== GATE 1: dataset identity ==="
sweep5_assert_provenance "$DATA_DIR" "$NAME" "$EXP_GS" "$EXP_DAY" "$EXP_DAYSAP" "$EXP_TX" || exit 1

echo
echo "=== GATE 2: data_dir is the PARENT, and /daily is populated ==="
# Both loaders append the time scale themselves. Passing .../daily silently
# resolves to .../daily/daily; the failure surfaces only after the job queues.
if [ ! -d "$DATA_DIR/daily" ]; then
    echo "  GATE 2 FAIL: $DATA_DIR/daily does not exist -- is data_dir the parent?"; exit 1
fi
if [ -d "$DATA_DIR/daily/daily" ]; then
    echo "  GATE 2 FAIL: nested daily/daily present -- path convention has changed"; exit 1
fi
N_SITE=$(ls "$DATA_DIR/daily"/*daily.csv 2>/dev/null | grep -vc all_biomes_merged)
echo "  site CSVs (excluding the aggregate): $N_SITE"
if [ "$N_SITE" -lt 200 ]; then echo "  GATE 2 FAIL: only $N_SITE site files"; exit 1; fi
# The aggregate matches the *daily.csv glob and would double-count every site.
# data_loader.py filters it BY NAME; assert that guard still exists.
if [ -f "$DATA_DIR/daily/all_biomes_merged_daily.csv" ]; then
    if ! grep -q 'all_biomes_merged' src/forward_selection/data_loader.py; then
        echo "  GATE 2 FAIL: aggregate present but data_loader no longer filters it"; exit 1
    fi
    echo "  GATE 2 PASS: aggregate present AND data_loader filters it"
else
    echo "  GATE 2 PASS: no aggregate present"
fi

echo
echo "=== GATE 3: cache is fresh ==="
if [ -f "$CACHE_DIR/feature_cache.npz" ]; then
    echo "  GATE 3 FAIL: $CACHE_DIR/feature_cache.npz exists -- move it or pick a new name"; exit 1
fi
echo "  GATE 3 PASS"

echo
echo "=== BUILDING CACHE ==="
date
python -u -m src.forward_selection.run_selection \
    --build_cache \
    --data_dir "$DATA_DIR" \
    --output_dir "$CACHE_DIR" \
    --time_scale daily \
    --spatial_method "$CACHE_METHOD" \
    --grid_size 0.05
BUILD_EXIT=$?
echo "### cache build exit: $BUILD_EXIT"
date
if [ $BUILD_EXIT -ne 0 ]; then echo "STATUS: CACHE BUILD FAILED"; exit $BUILD_EXIT; fi

echo
echo "=== GATE 4: the cache describes the design we asked for ==="
python - "$CACHE_DIR/feature_cache.npz" "$NAME" "$GROUPING" <<'PYEOF' || exit 1
import sys
import numpy as np
# allow_pickle: our own cache, written seconds ago by the step above on private
# scratch. Required for the string arrays (feature_names, pft_categories).
d = np.load(sys.argv[1], allow_pickle=True)
X, groups = d['X'], d['groups']
cats = sorted(str(c) for c in d['pft_categories'])
uniq = sorted(set(groups.tolist()))
print('  config   : %s' % sys.argv[2])
print('  X        : %s' % (X.shape,))
print('  features : %d' % len(d['feature_names']))
print('  groups   : %d' % len(uniq))
print('  classes  : %s' % cats)

# Stratification. Extra labels mean the resolver fell back to biome, and the run
# would measure a different design than the log claims.
expected = {'MF', 'DNF', 'ENF', 'EBF', 'WSA', 'WET', 'DBF', 'SAV'}
extra, missing = set(cats) - expected, expected - set(cats)
if extra:
    sys.exit('  GATE 4 FAIL: non-PFT labels %s -- resolver fell back to biome' % sorted(extra))
if missing:
    sys.exit('  GATE 4 FAIL: PFT classes missing: %s' % sorted(missing))

# Two-sided control on --spatial_method. BOTH methods densify ids to 0..n-1, so
# the ids cannot discriminate. What does: groups come from ONE lat/lon per SITE
# and rows are concatenated site by site. Under 'default' site->group is a
# bijection so every group is one contiguous run (runs == n_groups); under 'grid'
# co-located sites share a group but are not adjacent in site order, so
# runs > n_groups. Assert whichever relation the REQUESTED method implies, so a
# mismatch is caught in BOTH directions and not only one.
want = sys.argv[3]  # argv[2] is $NAME here, not the grouping
runs = 1 + int((groups[1:] != groups[:-1]).sum())
if want == 'grid' and runs <= len(uniq):
    sys.exit('  GATE 4 FAIL: runs == n_groups (%d) -- every group is one contiguous '
             'site block, the signature of method=default, not grid' % len(uniq))
if want == 'site' and runs != len(uniq):
    sys.exit('  GATE 4 FAIL: %d runs over %d groups -- co-located sites were merged, '
             'the signature of method=grid, not default' % (runs, len(uniq)))
if np.isnan(X).any():
    sys.exit('  GATE 4 FAIL: NaN in X -- the row drop did not run')
print('  GATE 4 PASS: GROUPING=%s, 8/8 PFT classes, %d groups over %d contiguous '
      'site runs, no NaN' % (want, len(uniq), runs))
PYEOF

ls -lh "$CACHE_DIR/feature_cache.npz"
echo
echo "STATUS: DONE  task=$T  config=$NAME  cache=$CACHE_DIR/feature_cache.npz"
date
