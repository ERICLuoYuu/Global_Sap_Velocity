#!/bin/bash
#SBATCH --partition=zen4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=192
#SBATCH --mem=700G --time=1-00:00:00
#SBATCH --job-name=train_ffs_sel
#SBATCH --output=/scratch/tmp/yluo2/gsv/outputs/forward_selection/slurm_train_ffs_selected_%j.out

# Training with EXACT FFS-selected features (--selected_features filter)
cd /scratch/tmp/yluo2/gsv
source /scratch/tmp/yluo2/gsv/.venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv

OUT=/scratch/tmp/yluo2/gsv/outputs/forward_selection
SCORING=${1:-neg_rmse}
RESULTS="$OUT/ffs_${SCORING}_results.json"

echo "=== Step 1: Generate selected features JSON + training command ==="
date
python -m src.forward_selection.ffs_to_training_args "$RESULTS" "ffs_${SCORING}_selected"

echo ""
echo "=== Step 2: Run training with --selected_features ==="
date

SF_JSON="$OUT/selected_features_${SCORING}.json"
if [ ! -f "$SF_JSON" ]; then
    echo "ERROR: $SF_JSON not found"
    exit 1
fi

TRAINING_CMD="$OUT/training_cmd_${SCORING}.sh"
if [ -f "$TRAINING_CMD" ]; then
    echo "Running: $TRAINING_CMD"
    cat "$TRAINING_CMD"
    echo ""
    bash "$TRAINING_CMD"
else
    echo "ERROR: $TRAINING_CMD not found"
    exit 1
fi

echo ""
echo "=== Done ==="
date
