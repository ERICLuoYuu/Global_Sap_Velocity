#!/bin/bash
#SBATCH --job-name=transp_compare
#SBATCH --partition=zen2-128C-496G
#SBATCH --time=18:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/transp_compare_%j.out
#SBATCH --error=logs/transp_compare_%j.err
#SBATCH --mail-user=yu.luo@uni-muenster.de
#SBATCH --mail-type=END,FAIL

set -euo pipefail
mkdir -p logs

echo "=== Running transpiration comparison pipeline ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"

source /scratch/tmp/yluo2/gsv/.venv/bin/activate
cd /scratch/tmp/yluo2/gsv-wt/map-viz

# Step 1: Preprocess all available products
echo "--- Preprocessing ---"
python -m src.transpiration_comparison.cli -v preprocess --products all

# Step 2: Spatial comparison (Phase 3)
echo "--- Spatial comparison ---"
python -m src.transpiration_comparison.cli -v compare --phase spatial

# Step 3: Temporal comparison (Phase 4)
echo "--- Temporal comparison ---"
python -m src.transpiration_comparison.cli -v compare --phase temporal

# Step 4: Regional deep dives (Phase 5)
echo "--- Regional analysis ---"
python -m src.transpiration_comparison.cli -v compare --phase regional

# Step 5: Product agreement (Phase 6)
echo "--- Agreement analysis ---"
python -m src.transpiration_comparison.cli -v compare --phase agreement

# Step 6: Collocation analysis
echo "--- Collocation ---"
python -m src.transpiration_comparison.cli -v compare --phase collocation

echo "=== Comparison complete ==="
echo "Figures: outputs/figures/transpiration_comparison/"
echo "Reports: outputs/transpiration_comparison/"
