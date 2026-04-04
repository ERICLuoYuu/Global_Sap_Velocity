#!/bin/bash
#SBATCH --job-name=env-bench
#SBATCH --output=logs/env_bench_%j.log
#SBATCH --time=06:00:00
#SBATCH --mem=0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=19
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yu.luo@uni-muenster.de

module load palma/2023b
module load CUDA/12.4.0

source /scratch/tmp/yluo2/gsv/.venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv-wt/fix-gap-benchmark
cd /scratch/tmp/yluo2/gsv-wt/fix-gap-benchmark

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BENCHMARK_TARGET=env

echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "TARGET: ${BENCHMARK_TARGET}"
nvidia-smi --list-gpus 2>/dev/null || echo "No GPUs"

python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
"

python -u -c "
import sys; sys.path.insert(0, '.')
from notebooks.gap_benchmark import main
main()
"

echo ""
echo "Job finished at $(date)"
