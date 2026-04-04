#!/bin/bash
#SBATCH --job-name=loritz_baseline
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

# Loritz et al. (2024) baseline models (CPU only)
# Gauged baseline target: KGE 0.64 +/- 0.05
# Ungauged baseline target: KGE -0.11 +/- 0.15

module purge
module load Python/3.11.5-GCCcore-13.2.0

source $HOME/envs/loritz/bin/activate

DATA_DIR="${WORK}/loritz_data"
OUTPUT_DIR="${WORK}/loritz_outputs"

mkdir -p logs

python -m src.paper_replicate.run_baseline \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --n-seeds 10 \
    --mode both

echo "Baseline complete"
