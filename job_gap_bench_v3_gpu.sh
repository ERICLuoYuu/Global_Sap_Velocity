#!/bin/bash
#SBATCH --job-name=gap-v3-gpu
#SBATCH --output=/scratch/tmp/yluo2/gsv-wt/fix-gap-benchmark/gap_v3_gpu_%j.log
#SBATCH --time=06:00:00
#SBATCH --mem=0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yu.luo@uni-muenster.de

# Gap Benchmark v3 — All methods with GPU acceleration for DL
# Uses gpu4090 partition: 6× RTX 4090 (24 GB VRAM each), 32 cores, 352 GB RAM
# DL methods run in parallel across 6 GPUs; CPU methods use 32 cores.
#
# Fallback partitions (change --partition if gpu4090 is full):
#   gpua100  — 4× A100 (40 GB each), --gres=gpu:4
#   gpu3090  — 8× RTX 3090 (24 GB each), --gres=gpu:8

source /scratch/tmp/yluo2/gsv/.venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv-wt/fix-gap-benchmark
cd /scratch/tmp/yluo2/gsv-wt/fix-gap-benchmark

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "============================================================"
echo "  Gap Benchmark v3 — gpu4090 / 6h / 6 GPUs / PyTorch CUDA"
echo "============================================================"
echo "Job started on $(hostname) at $(date)"
echo "Python: $(which python)"
echo "CPUs: $(nproc)"
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} — {props.total_mem / 1e9:.1f} GB VRAM')
"
echo ""

python -u -c "
import sys; sys.path.insert(0, '.')
from notebooks.gap_benchmark import main
main()
"

echo ""
echo "Final GPU memory state:"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
echo ""
echo "Job finished at $(date)"
