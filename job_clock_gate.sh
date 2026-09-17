#!/bin/bash
#SBATCH --job-name=clk_gate
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2
#SBATCH --mem=32G --time=03:00:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/clock_gate_%j.out
#
# The clock gate over a Stage-3 tree: every site measured from its own radiation against two independent
# estimators, each calibrated on the control population, and judged against src/clock_offsets/registry.py.
# Exit 1 on any ERROR -- a site whose clock is off and is not in the registry, or one whose `applied` entry
# did not land. Run it after every Stage-3 re-run that touches a registry site (plan WP-C.3).
#
#   sbatch job_clock_gate.sh                                  # the main clone, its own tree
#   GSV_ROOT=/path/to/worktree sbatch --export=ALL job_clock_gate.sh      # a branch's code, its own tree
#   GSV_ROOT=/path/to/worktree GSV_DATA=/scratch/tmp/yluo2/gsv sbatch --export=ALL job_clock_gate.sh
#
# A run over 225 sites takes about 13 minutes and reads nothing but CSV files.
set -uo pipefail
CODE="${GSV_ROOT:-/scratch/tmp/yluo2/gsv}"     # the clone whose src/ runs
DATA="${GSV_DATA:-$CODE}"                      # the clone whose Stage-3 tree is measured
VENV="${VENV:-/scratch/tmp/yluo2/gsv/.venv}"
SCALE="${ANALYSIS_SCALE:-sapwood_combined}"
ENV_DIR="${ENV_DIR:-$DATA/outputs/processed_data/$SCALE/env/outliers_removed}"
SAP_DIR="${SAP_DIR:-$DATA/outputs/processed_data/$SCALE/sap/outliers_removed}"
MD_DIR="${MD_DIR:-$DATA/data/raw/0.1.5/0.1.5/csv/$SCALE}"
OUT="${OUT:-$CODE/probes/radiation/clock_gate.json}"

cd "$CODE" || exit 1
source "$VENV/bin/activate" || exit 1
export PYTHONPATH="$CODE"
export ANALYSIS_SCALE="$SCALE"
mkdir -p "$(dirname "$OUT")"

echo "host=$(hostname) started=$(date -Is)"
echo "code=$CODE  data=$DATA  scale=$SCALE"
echo "git: $(git -C "$CODE" rev-parse --short HEAD 2>/dev/null) on $(git -C "$CODE" branch --show-current 2>/dev/null)"
python -u -m src.clock_offsets.gate --env-dir "$ENV_DIR" --sap-dir "$SAP_DIR" --md-dir "$MD_DIR" --out "$OUT"
RC=$?
echo "=== clock gate exit: $RC  (report: $OUT)"
date -Is
exit $RC
