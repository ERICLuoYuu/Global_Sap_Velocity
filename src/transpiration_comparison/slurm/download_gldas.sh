#!/bin/bash
#SBATCH --job-name=gldas_dl
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/gldas_dl_%j.out
#SBATCH --error=logs/gldas_dl_%j.err
#SBATCH --mail-user=yu.luo@uni-muenster.de
#SBATCH --mail-type=END,FAIL

set -uo pipefail
mkdir -p logs

echo "=== Downloading GLDAS via earthaccess ==="
echo "Date: $(date)"

# Verify .netrc exists
if [ ! -f ~/.netrc ]; then
    echo "ERROR: ~/.netrc not found. Create it with:"
    echo "  machine urs.earthdata.nasa.gov login <user> password <pass>"
    exit 1
fi

source /scratch/tmp/yluo2/gsv/.venv/bin/activate
cd /scratch/tmp/yluo2/gsv-wt/map-viz

python -m src.transpiration_comparison.cli -v download --products gldas

echo "=== GLDAS download complete ==="
ls -lh /scratch/tmp/yluo2/gsv/data/transpiration_products/gldas/
