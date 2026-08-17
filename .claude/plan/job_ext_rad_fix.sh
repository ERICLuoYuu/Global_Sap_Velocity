#!/bin/bash
#SBATCH --job-name=extradfix
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/extradfix_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/extradfix_%j.err

# Tests first, then a dry run. Nothing in the corpus is touched by this job --
# --apply is a separate submission, gated on this one passing.
set -uo pipefail
cd /scratch/tmp/yluo2/gsv || exit 1
source .venv/bin/activate || exit 1

echo "### python: $(python -V 2>&1)"

echo
echo "### import check on the cluster interpreter (3.9: a runtime 'type | type' dies here)"
python -c "from src.env_derive.ext_rad import extraterrestrial_radiation, fill_ext_rad; print('  import OK')" || exit 1

echo
echo "### unit tests + positive control against the shipped column"
python -m pytest src/env_derive/tests/ -v --tb=short -p no:cacheprovider
PYTEST_RC=$?
echo "### pytest exit: $PYTEST_RC"
if [ "$PYTEST_RC" -ne 0 ]; then
    echo "=== ABORT: tests failed, corpus untouched ==="
    exit 1
fi

echo
echo "### dry run over the corpus"
python .claude/plan/backfill_ext_rad.py --dry-run
DRY_RC=$?
echo "### dry-run exit: $DRY_RC"
exit $DRY_RC
