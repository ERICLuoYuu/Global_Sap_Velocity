#!/bin/bash
#SBATCH --job-name=gap-v3-all
#SBATCH --output=/scratch/tmp/yluo2/gsv-wt/fix-gap-benchmark/gap_v3_%j.log
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --partition=zen3
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yu.luo@uni-muenster.de

# Gap Benchmark v3 — All methods on zen3 (CPU only, PyTorch)
# For GPU-accelerated DL, use job_gap_bench_v3_gpu.sh

source /scratch/tmp/yluo2/gsv/.venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv-wt/fix-gap-benchmark
cd /scratch/tmp/yluo2/gsv-wt/fix-gap-benchmark

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "============================================================"
echo "  Gap Benchmark v3 — zen3 / 24h / 96 cores / PyTorch CPU"
echo "============================================================"
echo "Job started on $(hostname) at $(date)"
echo "Python: $(which python)"
echo "CPUs: $(nproc)"

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
echo ""

python -u -c "
import sys; sys.path.insert(0, '.')
from notebooks.gap_benchmark import main
main()
"

echo ""
echo "Job finished at $(date)"
