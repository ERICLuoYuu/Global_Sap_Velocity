#!/bin/bash
#SBATCH --job-name=fu_ann_mvd
#SBATCH --partition=requeue-zen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/fu_ann_mvd_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/fu_ann_mvd_%j.err
# min-valid-days robustness sweep: confirm SM/VPD leg signs are stable to the
# >=N-day gate (not an artefact of Fu's strict 300-day cut on a ~28-site sample).
hostname; date
cd /scratch/tmp/yluo2/gsv
source .venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv:$PYTHONPATH
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
python -c "import torch; torch.set_num_threads(${OMP_NUM_THREADS})"
DATADIR="outputs/processed_data/sapwood/merged/daytime_only/growing_season/daily"
for MVD in 150 200 250 300; do
  OUTDIR="outputs/fu_ann_sensitivity_sweep/mvd${MVD}"
  echo "=== min_valid_days=${MVD} -> ${OUTDIR} ==="
  python src/fu_ann_sensitivity/run_fu_ann_sensitivity.py       --data-dir "$DATADIR" --n-bins 5 10 --n-repeats 5       --min-valid-days "$MVD" --no-figures       --out-dir "$OUTDIR"
done
echo '=== sweep performance tables ==='
for MVD in 150 200 250 300; do echo "-- mvd${MVD} --"; cat outputs/fu_ann_sensitivity_sweep/mvd${MVD}/performance.csv; done
date
