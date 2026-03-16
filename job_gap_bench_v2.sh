#!/bin/bash
#SBATCH --job-name=gap-bench-v2
#SBATCH --output=/scratch/tmp/yluo2/gsv-wt/fix-gap-benchmark/gap_v2_%j.log
#SBATCH --time=06:00:00
#SBATCH --mem=0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --partition=zen4
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yu.luo@uni-muenster.de

# ── Environment ──────────────────────────────────────────────────────────────
source /scratch/tmp/yluo2/gsv/.venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv-wt/fix-gap-benchmark
cd /scratch/tmp/yluo2/gsv-wt/fix-gap-benchmark

# Global thread controls — overridden per method group in Python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# TF: limit default thread pools (fine-tuned per method group in code)
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=16

echo "============================================================"
echo "  Gap Benchmark v2 — Optimized with parallelization"
echo "============================================================"
echo "Job started on $(hostname) at $(date)"
echo "Python: $(which python)"
echo "Working dir: $(pwd)"
echo "CPUs: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo ""

python -u -c "
import sys
sys.path.insert(0, '.')
from notebooks.gap_benchmark import main
main()
"

echo ""
echo "Job finished at $(date)"
