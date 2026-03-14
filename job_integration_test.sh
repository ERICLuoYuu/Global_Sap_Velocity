#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=pipeline-tests
#SBATCH --ntasks-per-node=1
#SBATCH --output=/scratch/tmp/yluo2/gsv-wt/map-viz/logs/test_%j.log
#SBATCH --error=/scratch/tmp/yluo2/gsv-wt/map-viz/logs/test_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yluo2@uni-muenster.de

# Fix Windows line endings
sed -i 's/\r$//' "$0" 2>/dev/null || true

module load palma/2023a
module load GCC/12.3.0

source /scratch/tmp/yluo2/gsv/.venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH=/scratch/tmp/yluo2/gsv-wt/map-viz

cd /scratch/tmp/yluo2/gsv-wt/map-viz
mkdir -p logs

echo "=== Pipeline Test Suite ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Python: $(python --version)"
echo ""

python -m pytest src/make_prediction/tests/ \
    -v --tb=short \
    --junitxml=logs/test_results.xml \
    2>&1

EXIT_CODE=$?

echo ""
echo "=== Finished: $(date) ==="
echo "Exit code: ${EXIT_CODE}"

exit ${EXIT_CODE}
