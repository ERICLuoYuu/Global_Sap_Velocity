#!/bin/bash
#SBATCH --job-name=gapfill-e2e
#SBATCH --output=logs/gapfill_e2e_%j.log
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu3090
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yu.luo@uni-muenster.de

module load palma/2023b
module load CUDA/12.4.0

source /scratch/tmp/yluo2/gsv/.venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv-wt/fix-gap-benchmark
cd /scratch/tmp/yluo2/gsv-wt/fix-gap-benchmark

echo "=== Gap-filling E2E test ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
nvidia-smi --list-gpus 2>/dev/null || echo "No GPUs"

# 1. Run unit tests
echo ""
echo "=== Unit tests ==="
python -m pytest src/gap_filling/tests/ -v --tb=short 2>&1

# 2. Run integration test on a few sites
echo ""
echo "=== Integration test: SapFlowAnalyzer with gap filling ==="
python -c "
from src.Analyzers.sap_analyzer import SapFlowAnalyzer
from src.gap_filling.config import GapFillingConfig

cfg = GapFillingConfig(threshold=0.85)
analyzer = SapFlowAnalyzer(scale='sapwood', gap_filling_config=cfg)
analyzer.run_analysis_in_batches(batch_size=5, switch='load')
print('Gap-filling integration test PASSED')
" 2>&1

echo ""
echo "=== Done: $(date) ==="
