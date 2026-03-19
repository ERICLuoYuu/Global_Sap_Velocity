#!/bin/bash
#SBATCH --job-name=era5_gee_fix
#SBATCH --output=logs/era5_gee_%j.log
#SBATCH --time=48:00:00
#SBATCH --mem=768G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=zen5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yluo2@uni-muenster.de

source /scratch/tmp/yluo2/gsv/.venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv-wt/map-viz
cd /scratch/tmp/yluo2/gsv-wt/map-viz
mkdir -p logs

echo "ERA5-Land GEE processing"
echo "Date: $(date)  Node: $(hostname)  CPUs: $SLURM_CPUS_PER_TASK"

python src/make_prediction/process_era5land_gee_opt_fix.py     --time-scale daily     --year 2020     --month 6     --shapefile '/scratch/tmp/yluo2/gsv-wt/map-viz/data/data/raw/grided/tree_cover_shapefile_dissolved/tree_cover_shapefile_dissolved.shp'     2>&1

echo "Finished: $(date)"
