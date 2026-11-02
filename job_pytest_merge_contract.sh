#!/bin/bash
#SBATCH --job-name=pt_mergectr
#SBATCH --partition=express
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2
#SBATCH --mem=16G --time=00:25:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/pytest_merge_contract_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/pytest_merge_contract_%j.err

# The servable feature contract as the Stage-4 merge applies it.
set -uo pipefail
ROOT="${GSV_ROOT:-/scratch/tmp/yluo2/gsv}"
cd "$ROOT" || exit 1
source /scratch/tmp/yluo2/gsv/.venv/bin/activate || exit 1
export PYTHONPATH="$ROOT"
export ANALYSIS_SCALE=sapwood_combined

NEW="src/feature_contract/tests notebooks/test_merge_feature_contract.py notebooks/test_merge_static_features.py src/clock_offsets/tests src/env_daily/tests src/Analyzers/tests notebooks/test_env_daily_merge.py notebooks/test_radiation_integration.py"
REG="notebooks/test_daily_own_hours.py notebooks/test_env_hourly_gaps.py notebooks/test_normalise_timestamps_streams.py notebooks/test_daytime_scope.py src/feature_contract/tests src/hyperparameter_optimization/tests src/forward_selection/tests src/serving/tests"

# Nothing is deselected. test_dl_parity was, while it failed: its two FFS selections name soil-moisture
# features retired on 2026-09-14, so the DL parity loader accepted no site at all. The DL trainer now
# refuses such a list outright, and the test strips the retired names before checking the rest.
DESELECT=""

# The regression suite runs by DEFAULT. It used to be opt-in behind WITH_REGRESSION=1, with REG_RC
# defaulting to 0, so a run that never executed a single regression test still printed "RESULT: PASS"
# -- a green verdict for work not done, which is the failure mode this whole change is about.
# WITH_REGRESSION=0 still skips it, and the verdict then says so rather than claiming a clean bill.
WITH_REGRESSION="${WITH_REGRESSION:-1}"

# Every path in NEW and REG must exist under $ROOT before anything runs. Without this a run against the
# wrong tree -- the default main clone instead of a worktree that still holds the branch -- collects
# nothing, and "no tests ran" is a pytest usage error whose wording nobody reads.
MISSING=""
for p in $NEW $REG; do
  [ -e "$p" ] || MISSING="$MISSING $p"
done
if [ -n "$MISSING" ]; then
  echo "RESULT: FAIL (ROOT=$ROOT has no$MISSING -- set GSV_ROOT to the tree that holds them)"
  exit 1
fi

echo "=== python: $(python -V 2>&1) numpy: $(python -c 'import numpy;print(numpy.__version__)')"
echo "=== merge md5: $(md5sum notebooks/merge_gap_filled_hourly_orginal.py | cut -d' ' -f1)"
echo "=== NEW ==="
python -m pytest -q $NEW
NEW_RC=$?
echo "=== NEW exit: $NEW_RC"

if [ "$WITH_REGRESSION" = "1" ]; then
  echo "=== REGRESSION (nothing deselected) ==="
  python -m pytest -q $REG $DESELECT
  REG_RC=$?
  echo "=== REGRESSION exit: $REG_RC"
  SCOPE="new + regression"
else
  REG_RC=0
  SCOPE="new only -- REGRESSION WAS NOT RUN"
fi

if [ "$NEW_RC" -eq 0 ] && [ "$REG_RC" -eq 0 ]; then
  echo "RESULT: PASS ($SCOPE)"
else
  # Exit non-zero so the verdict reaches SLURM. Printing FAIL and returning 0 left sacct saying
  # COMPLETED over a run whose own log said it had failed -- the same false green the WITH_REGRESSION
  # default was changed for.
  echo "RESULT: FAIL ($SCOPE; new=$NEW_RC regression=$REG_RC)"
  exit 1
fi
