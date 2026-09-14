#!/bin/bash
#SBATCH --job-name=pytest_sv
#SBATCH --partition=express
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=16G --time=00:30:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/pytest_sv_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/pytest_sv_%j.err

# Tests for the servable feature contract.
#   MODE=new  only the new test files (the RED run, before the implementation is uploaded)
#   MODE=all  the new files plus every existing suite the change touches (regression)
# The exit code is printed and returned as-is: never pipe pytest through tail.

set -uo pipefail
cd /scratch/tmp/yluo2/gsv || exit 1
source .venv/bin/activate || exit 1
export PYTHONPATH=/scratch/tmp/yluo2/gsv
export PYTHONHASHSEED=0

NEW="src/feature_contract/tests src/hyperparameter_optimization/tests/test_feature_contract_fe.py src/hyperparameter_optimization/tests/test_feature_contract_args.py src/hyperparameter_optimization/tests/test_feature_contract_swc_scale.py"
REGRESSION="src/hyperparameter_optimization/tests/test_feature_engineering.py src/hyperparameter_optimization/tests/test_training_utils.py src/forward_selection/tests/test_data_loader.py"

MODE="${MODE:-all}"
case "$MODE" in
  new) TARGETS="$NEW" ;;
  all) TARGETS="$NEW $REGRESSION" ;;
  *)   echo "FATAL: MODE must be new or all, got '$MODE'"; exit 2 ;;
esac

echo "MODE=$MODE  python=$(python --version 2>&1)  git=$(git rev-parse --short HEAD)"
python -m pytest $TARGETS -q -p no:cacheprovider 2>&1
RC=$?
echo "PYTEST_EXIT_CODE=$RC"
exit $RC
