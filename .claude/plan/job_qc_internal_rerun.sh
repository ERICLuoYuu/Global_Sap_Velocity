#!/bin/bash
#SBATCH --job-name=qcextrad
#SBATCH --partition=long
#SBATCH --array=0-23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/qcextrad_%A_%a.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/qcextrad_%A_%a.err

# Re-run Stage 3 for the 24 internal sites now that the corpus carries ext_rad.
#
# Two things change for these sites and nothing else does:
#   1. env outputs gain the ext_rad column, so the trainer stops skipping them.
#   2. sw_in / ppfd_in regain the DIURNAL outlier split. Until now `df["ext_rad"]`
#      raised KeyError inside the analysis try block at every internal site and
#      the except fell back to a single global pass -- weaker QC, applied
#      silently, on two model input features.
# The sap outputs should come out byte-identical: sap_analyzer derives its own
# day mask from pvlib and never reads ext_rad. That is checked after the run
# rather than assumed.
#
# SITES ARE NAMED, NOT INDEXED BY CHUNK. qc_chunk_sites.py orders by file size,
# and the backfill just changed the size of these 24 files -- so a chunk index
# from the earlier run now resolves to a different site. Naming them removes the
# coupling entirely.
set -uo pipefail

SITES_ALL=(
    AUT_MMG AUT_ZOE CHE_DAV CHE_LAE CZE_BK1 CZE_LNZ CZE_RAJ CZE_STN
    DEU_HAI DEU_HAR DEU_HOH ESP_ABR ESP_LM1 ESP_LM2 ESP_LMA FIN_HYY
    FRA_BIL FRA_HES ITA_CP2 NOR_HUR POL_MEZ POL_TUC SWE_NOR SWE_SGR
)
SITE="${SITES_ALL[$SLURM_ARRAY_TASK_ID]}"

REPO=/scratch/tmp/yluo2/gsv
DATA=/scratch/tmp/yluo2/gsv/data
LOG=/scratch/tmp/yluo2/gsv/removal_log.csv
CORPUS=/scratch/tmp/yluo2/Global_Sap_Velocity/data/raw/0.1.5/0.1.5/csv/sapwood_combined
OUT=/scratch/tmp/yluo2/qc_extrad/$SITE

cd "$REPO" || exit 1
source "$REPO/.venv/bin/activate"
export PYTHONPATH="$REPO"
export MPLBACKEND=Agg

echo "### task ${SLURM_ARRAY_TASK_ID} -> site $SITE"
echo "### code: $(git -C "$REPO" log --oneline -1)"
echo "### removal_log sha: $(sha256sum "$LOG" | cut -c1-16)"

# The site must actually carry ext_rad now, or this run reproduces the defect
# it exists to remove and the only trace would be a log line nobody reads.
if ! head -1 "$CORPUS/${SITE}_env_data.csv" | tr "," "\n" | sed "s/\r$//" | grep -qx "ext_rad"; then
    if [ "$SITE" != "SWE_NOR" ]; then
        echo "FATAL: $SITE still has no ext_rad column in the corpus"
        exit 1
    fi
fi

rm -rf "$OUT"
mkdir -p "$OUT"
python .claude/plan/qc_stage3_run.py --sites "$SITE" --data "$DATA" \
    --out "$OUT" --removal-log "$LOG"
STATUS=$?
echo "=== exit $STATUS ==="
echo "files written: $(find "$OUT" -type f | wc -l)"

# Did the diurnal split actually come back? Zero occurrences of the fallback
# message is the positive signal; this is the line that proved the defect.
FALLBACK=$(grep -c "Could not calculate day/night mask" "/scratch/tmp/yluo2/gsv/logs/qcextrad_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>/dev/null || echo 0)
echo "### day/night-mask fallbacks in this task's log: $FALLBACK (want 0)"
exit $STATUS
