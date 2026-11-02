#!/bin/bash
#SBATCH --job-name=static_pytest
#SBATCH --partition=express
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=8G --time=00:30:00
#SBATCH --output=/scratch/tmp/yluo2/static_logs/%x_%j.out
#SBATCH --error=/scratch/tmp/yluo2/static_logs/%x_%j.err

# Tests for src/static_features, run from the worktree. A bare test_x.py means
# src/static_features/tests/test_x.py, and "." means that directory. Exit code is pytest's own.
set -uo pipefail
hostname; date
W=/scratch/tmp/yluo2/gsv-wt/static-features
cd "$W" || exit 1
source /scratch/tmp/yluo2/gsv/.venv/bin/activate || exit 1
export PYTHONPATH="$W"
export PYTHONHASHSEED=0
python --version

ARGS=()
for arg in "$@"; do
    case "$arg" in
        .) ARGS+=(src/static_features/tests) ;;
        test_*) ARGS+=("src/static_features/tests/$arg") ;;
        *) ARGS+=("$arg") ;;
    esac
done

python -m pytest -q -p no:cacheprovider -W ignore::FutureWarning "${ARGS[@]}"
RC=$?
echo "### pytest exit: $RC"
exit $RC
