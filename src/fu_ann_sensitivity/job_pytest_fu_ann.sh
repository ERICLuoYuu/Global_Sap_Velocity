#!/bin/bash
#SBATCH --job-name=fu_ann_pytest
#SBATCH --partition=requeue-zen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/fu_ann_pytest_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/fu_ann_pytest_%j.err
hostname; date
cd /scratch/tmp/yluo2/gsv
source .venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv:$PYTHONPATH
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
python --version
python -c "import torch; print('torch', torch.__version__)"
python -m pytest src/fu_ann_sensitivity/tests/ -v -W error::RuntimeWarning     --cov=src/fu_ann_sensitivity --cov-report=term-missing
echo "PYTEST_EXIT=$?"
date
