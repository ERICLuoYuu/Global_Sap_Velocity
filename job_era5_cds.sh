#!/bin/bash
#SBATCH --job-name=era5_cds
#SBATCH --output=logs/era5_cds_%j.log
#SBATCH --time=48:00:00
#SBATCH --mem=768G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=zen5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yluo2@uni-muenster.de

# Venv has all dependencies; skip module loads that differ across partitions
source /scratch/tmp/yluo2/gsv/.venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv-wt/map-viz

cd /scratch/tmp/yluo2/gsv-wt/map-viz
mkdir -p logs

echo "ERA5-Land CDS processing"
echo "Date: $(date)  Node: $(hostname)  CPUs: $SLURM_CPUS_PER_TASK  Mem: ${SLURM_MEM_PER_NODE}MB"

python -c "import cdsapi; c = cdsapi.Client(); print('CDS API connected')" 2>&1 || {
    echo 'ERROR: CDS API unreachable from compute node'
    exit 1
}

python src/make_prediction/process_era5land_cds.py     --time-scale daily     --year 2020     --month 6     --output-dir /scratch/tmp/yluo2/gsv/outputs/predictions/cds_era5     --shapefile /scratch/tmp/yluo2/gsv-wt/map-viz/data/data/raw/grided/tree_cover_shapefile_dissolved/tree_cover_shapefile_dissolved.shp     2>&1

echo "Finished: $(date)"
