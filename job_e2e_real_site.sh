#!/bin/bash
#SBATCH --job-name=gf-e2e-real
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=e2e_real_%j.out
#SBATCH --error=e2e_real_%j.err

module load palma/2023b
module load CUDA/12.4.0

VENV="/scratch/tmp/yluo2/gsv/.venv"
WORKTREE="/scratch/tmp/yluo2/gsv-wt/fix-gap-benchmark"
source "${VENV}/bin/activate"
cd "${WORKTREE}"
export PYTHONPATH="${WORKTREE}"

echo "=== E2E real site test ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
nvidia-smi --list-gpus 2>/dev/null || echo "No GPUs"

python -m pytest src/gap_filling/tests/test_e2e_real_site.py -v -s 2>&1

echo "=== Done: $(date) ==="
