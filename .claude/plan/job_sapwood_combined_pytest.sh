#!/bin/bash
#SBATCH --job-name=sapcomb_test
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/sapcomb_test_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/sapcomb_test_%j.err

# Validate src/sapwood_combined on the CANONICAL interpreter.
#
# The module was written and tested on local Windows Python 3.10. Palma runs
# 3.9.25, and the two differ in ways that do not show up locally -- zip(strict=)
# is the one that already bit this module. Nothing is written by this job; it
# only runs the suite and the dry run.

set -euo pipefail

REPO=/scratch/tmp/yluo2/gsv
cd "$REPO"
mkdir -p logs

source "$REPO/.venv/bin/activate"
export PYTHONPATH="$REPO"

echo "=== interpreter ==="
python --version
python -c "import sys; print('version_info:', sys.version_info)"
echo

echo "=== pandas / numpy ==="
python -c "import pandas, numpy; print('pandas', pandas.__version__, '| numpy', numpy.__version__)"
echo

echo "=== test suite ==="
python -m pytest src/sapwood_combined/tests/ -q -p no:cacheprovider \
    --cov=src.sapwood_combined --cov-report=term
echo

echo "=== import smoke (catches 3.9 syntax the tests might miss) ==="
python - <<'PY'
import importlib

for name in (
    "src.sapwood_combined.membership",
    "src.sapwood_combined.conversion",
    "src.sapwood_combined.assembly",
    "src.sapwood_combined.manifest",
    "src.sapwood_combined.build_sapwood_combined",
):
    importlib.import_module(name)
    print("imported", name)
PY
echo

echo "=== real-corpus dry run (writes nothing) ==="
python -m src.sapwood_combined.build_sapwood_combined --dry-run

echo
echo "=== DONE ==="
