#!/bin/bash
#SBATCH --job-name=loritz_ungauged
#SBATCH --partition=gpu2080,gpua100,gpuv100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/ungauged_%j.out
#SBATCH --error=logs/ungauged_%j.err

# Loritz et al. (2024) ungauged-continental LSTM
# 10 Monte Carlo runs, 20 epochs each
# Target: KGE 0.52 +/- 0.16

module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1

source $HOME/envs/loritz/bin/activate

DATA_DIR="${WORK}/loritz_data"
OUTPUT_DIR="${WORK}/loritz_outputs"

mkdir -p logs

nvidia-smi

python -m src.paper_replicate.run_ungauged \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --n-seeds 10 \
    --seed-start 1

echo "Ungauged LSTM complete"
