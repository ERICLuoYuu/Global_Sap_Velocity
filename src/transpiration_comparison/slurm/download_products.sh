#!/bin/bash
#SBATCH --job-name=transp_download
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/transp_download_%j.out
#SBATCH --error=logs/transp_download_%j.err
#SBATCH --mail-user=yu.luo@uni-muenster.de
#SBATCH --mail-type=END,FAIL

set -uo pipefail  # No -e: continue on individual product failure
mkdir -p logs

echo "=== Downloading transpiration products ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

source /scratch/tmp/yluo2/gsv/.venv/bin/activate
cd /scratch/tmp/yluo2/gsv-wt/map-viz

# ERA5-Land monthly means (fast: ~5 min total)
echo "--- ERA5-Land (monthly means) ---"
python -m src.transpiration_comparison.cli -v download --products era5land || echo "ERA5-Land download failed, continuing..."

# GLEAM SFTP (DO NOT COMMIT credentials to git)
echo "--- GLEAM ---"
export GLEAM_USER="gleamuser"
export GLEAM_PASS="GLEAM4#h-cel_924"
python -m src.transpiration_comparison.cli -v download --products gleam || echo "GLEAM download failed, continuing..."

# GLDAS via OPeNDAP
echo "--- GLDAS ---"
python -m src.transpiration_comparison.cli -v download --products gldas || echo "GLDAS download failed, continuing..."

# PMLv2 via GEE (monthly means from 8-day composites)
echo "--- PMLv2 ---"
python -m src.transpiration_comparison.cli -v download --products pmlv2 || echo "PMLv2 download failed, continuing..."

echo "=== Download complete ==="
echo "--- Downloaded files ---"
find /scratch/tmp/yluo2/gsv/data/transpiration_products/ -type f | sort
echo "--- File counts ---"
for d in era5 gldas gleam pmlv2; do
    echo "$d: $(find /scratch/tmp/yluo2/gsv/data/transpiration_products/$d -type f 2>/dev/null | wc -l) files"
done
