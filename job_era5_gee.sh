#!/bin/bash
#SBATCH --job-name=era5_gee_fix
#SBATCH --output=logs/era5_gee_%j.log
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yluo2@uni-muenster.de

# ── Environment ──────────────────────────────────────────
module load palma/2023a
module load GCC/12.3.0
module load OpenMPI/4.1.5
module load SciPy-bundle/2023.07

# Activate project venv
source /scratch/tmp/yluo2/gsv/.venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv-wt/map-viz

# ── Execution ────────────────────────────────────────────
cd /scratch/tmp/yluo2/gsv-wt/map-viz

echo "Starting ERA5-Land GEE processing (fixed script)"
echo "Date: $(date)"
echo "Node: $(hostname)"

python src/make_prediction/process_era5land_gee_opt_fix.py     --time_scale daily     --year 2020     --month 6     2>&1

echo "Finished: $(date)"
