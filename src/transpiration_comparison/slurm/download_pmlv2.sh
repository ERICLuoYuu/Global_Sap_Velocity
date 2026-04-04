#!/bin/bash
#SBATCH --job-name=pmlv2_dl
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/pmlv2_dl_%j.out
#SBATCH --error=logs/pmlv2_dl_%j.err
#SBATCH --mail-user=yu.luo@uni-muenster.de
#SBATCH --mail-type=END,FAIL

set -uo pipefail
mkdir -p logs

echo "=== Downloading PMLv2 via tiled GEE export ==="
echo "Date: $(date)"

source /scratch/tmp/yluo2/gsv/.venv/bin/activate
cd /scratch/tmp/yluo2/gsv-wt/map-viz

python -m src.transpiration_comparison.cli -v download --products pmlv2

echo "=== PMLv2 download complete ==="
ls -lh /scratch/tmp/yluo2/gsv/data/transpiration_products/pmlv2/
