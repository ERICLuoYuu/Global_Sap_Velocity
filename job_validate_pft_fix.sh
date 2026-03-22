#!/bin/bash
#SBATCH --job-name=validate-pft-fix
#SBATCH --partition=zen5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/tmp/yluo2/gsv-wt/map-viz/logs/validate_pft_fix_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv-wt/map-viz/logs/validate_pft_fix_%j.err

echo "=== Validate PFT GeoTIFF fix ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

# Setup
cd /scratch/tmp/yluo2/gsv-wt/map-viz
source /scratch/tmp/yluo2/gsv/.venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv-wt/map-viz:/scratch/tmp/yluo2/gsv-wt/map-viz/src/make_prediction

mkdir -p logs

# Run validation
python validate_pft_fix.py
EXIT_CODE=$?

echo ""
echo "Exit code: $EXIT_CODE"
echo "End: $(date)"
exit $EXIT_CODE
