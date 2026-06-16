#!/bin/bash
#SBATCH --job-name=smvpd_pytest
#SBATCH --partition=requeue-zen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/smvpd_pytest_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/smvpd_pytest_%j.err
hostname; date
cd /scratch/tmp/yluo2/gsv
source .venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv:$PYTHONPATH
python --version
python -m pytest src/sm_vpd_decoupling/ -v
echo "PYTEST_EXIT=$?"
date
