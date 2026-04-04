#!/bin/bash
#SBATCH --job-name=transp_install
#SBATCH --partition=express
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/transp_install_%j.out
#SBATCH --error=logs/transp_install_%j.err
#SBATCH --mail-user=yu.luo@uni-muenster.de
#SBATCH --mail-type=END,FAIL

set -euo pipefail
mkdir -p logs

echo "=== Installing transpiration comparison dependencies ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

source /scratch/tmp/yluo2/gsv/.venv/bin/activate
which python
python --version

# Install required packages
pip install --no-cache-dir \
    xskillscore \
    SkillMetrics \
    earthaccess \
    paramiko

# xESMF requires ESMF (Fortran) -- try pip first, conda fallback
pip install xesmf 2>/dev/null || {
    echo "xesmf pip install failed (needs ESMF). Try: conda install -c conda-forge xesmf"
    echo "Falling back to xarray interpolation for regridding."
}

# xcdat for climatology
pip install xcdat 2>/dev/null || echo "xcdat install failed, using manual climatology"

echo "=== Installed packages ==="
pip list | grep -iE "xskillscore|skillmetrics|earthaccess|paramiko|xesmf|xcdat"

echo "=== Done ==="
