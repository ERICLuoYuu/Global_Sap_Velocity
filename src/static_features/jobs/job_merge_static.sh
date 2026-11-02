#!/bin/bash
#SBATCH --job-name=merge_static
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=64G --time=06:00:00
#SBATCH --output=/scratch/tmp/yluo2/static_logs/%x_%j.out
#SBATCH --error=/scratch/tmp/yluo2/static_logs/%x_%j.err

# One Stage 4 merge from the worktree, configured as the fifth arm of job_merge_sweep5.sh (growing season,
# sap daytime only, treatment filter). OUTPUT_DIR is required and must be empty or absent, so no earlier tree
# is overwritten. The merge reads the 1 km features under static_1km/$ANALYSIS_SCALE and stops if any is absent.
set -uo pipefail
hostname; date
W=/scratch/tmp/yluo2/gsv-wt/static-features
cd "$W" || exit 1
source /scratch/tmp/yluo2/gsv/.venv/bin/activate || exit 1
export ANALYSIS_SCALE=sapwood_combined
export PYTHONPATH="$W"
export PYTHONUNBUFFERED=1
python --version

if [ -z "${OUTPUT_DIR:-}" ]; then
  echo "FATAL: set OUTPUT_DIR"; exit 2
fi
if [ -d "$OUTPUT_DIR" ] && [ -n "$(ls -A "$OUTPUT_DIR")" ]; then
  echo "FATAL: $OUTPUT_DIR is not empty, and a merge would write over it"; exit 2
fi

python -u notebooks/merge_gap_filled_hourly_orginal.py \
  --output-dir "$OUTPUT_DIR" \
  --daytime-sap-only \
  --growing-season-only \
  --apply-treatment-filter \
  --no-apply-flo2019-correction
RC=$?
echo "### merge exit: $RC"
exit $RC
