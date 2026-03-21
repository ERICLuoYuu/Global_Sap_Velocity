#!/bin/bash
#SBATCH --job-name=gap-bench-clean
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=benchmark_clean_%j.out
#SBATCH --error=benchmark_clean_%j.err

module load palma/2023b
module load CUDA/12.4.0

VENV="/scratch/tmp/yluo2/gsv/.venv"
WORKTREE="/scratch/tmp/yluo2/gsv-wt/fix-gap-benchmark"
source "${VENV}/bin/activate"
cd "${WORKTREE}"
export PYTHONPATH="${WORKTREE}"

# Prevent thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "=== Clean Benchmark Run ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $(nproc)"
nvidia-smi --list-gpus 2>/dev/null || echo "No GPUs"
echo "Python: $(python --version)"
echo ""

python -m notebooks.gap_benchmark 2>&1

echo ""
echo "=== Done: $(date) ==="
