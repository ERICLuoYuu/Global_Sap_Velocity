#!/bin/bash
#SBATCH --job-name=pmlv2_8day
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/pmlv2_8day_%j.out
#SBATCH --error=logs/pmlv2_8day_%j.err
#SBATCH --mail-user=yu.luo@uni-muenster.de
#SBATCH --mail-type=END,FAIL

set -euo pipefail
mkdir -p logs

echo "=== PMLv2 8-day re-download + full pipeline ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

source /scratch/tmp/yluo2/gsv/.venv/bin/activate
cd /scratch/tmp/yluo2/gsv-wt/map-viz

# Step 1: Download PMLv2 as 8-day composites (requires GEE auth)
echo "--- Downloading PMLv2 8-day composites ---"
python -m src.transpiration_comparison.cli -v download --products pmlv2

# Step 2: Preprocess only PMLv2 (other products still cached)
echo "--- Preprocessing PMLv2 ---"
python -m src.transpiration_comparison.cli -v preprocess --products pmlv2

# Step 3-7: Full comparison pipeline
echo "--- Spatial comparison ---"
python -m src.transpiration_comparison.cli -v compare --phase spatial

echo "--- Temporal comparison ---"
python -m src.transpiration_comparison.cli -v compare --phase temporal

echo "--- Regional analysis ---"
python -m src.transpiration_comparison.cli -v compare --phase regional

echo "--- Agreement analysis ---"
python -m src.transpiration_comparison.cli -v compare --phase agreement

echo "--- Collocation ---"
python -m src.transpiration_comparison.cli -v compare --phase collocation

echo "=== Complete ==="
echo "Figures: outputs/figures/transpiration_comparison/"
echo "Reports: outputs/transpiration_comparison/"
