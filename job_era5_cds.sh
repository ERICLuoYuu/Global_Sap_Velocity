#!/bin/bash
#SBATCH --job-name=era5_cds
#SBATCH --output=logs/era5_cds_%j.log
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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

# CDS credentials from ~/.cdsapirc (must exist on compute nodes)
# If CDS API unreachable, script will log error and exit

# ── Execution ────────────────────────────────────────────
cd /scratch/tmp/yluo2/gsv-wt/map-viz
mkdir -p logs

echo "Starting ERA5-Land CDS processing"
echo "Date: $(date)"
echo "Node: $(hostname)"

# Test CDS connectivity first
python -c "import cdsapi; c = cdsapi.Client(); print('CDS API connected')" 2>&1 || {
    echo 'ERROR: CDS API unreachable from compute node'
    exit 1
}

python src/make_prediction/process_era5land_cds.py     --time-scale daily     --year 2020     --month 6     --output-dir /scratch/tmp/yluo2/gsv/outputs/predictions/cds_era5     2>&1

echo "Finished: $(date)"
