#!/bin/bash
# Setup script for Loritz et al. (2024) replication on Palma2
# Run this ONCE before submitting jobs

set -e

echo "=== Setting up Loritz replication environment ==="

# 1. Create virtual environment
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.1.1

python -m venv $HOME/envs/loritz
source $HOME/envs/loritz/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn hydroeval matplotlib seaborn

# 3. Download data.zip from Zenodo
DATA_DIR="${WORK}/loritz_data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ ! -f "data.zip" ]; then
    echo "Downloading data.zip from Zenodo..."
    wget -q "https://zenodo.org/records/10118262/files/data.zip"
fi

if [ ! -d "plant_md" ]; then
    echo "Extracting data.zip..."
    unzip -q data.zip
fi

# 4. Verify contents
echo ""
echo "=== Data verification ==="
echo "Sites (plant_md files):"
ls plant_md/ | wc -l
echo "env_data files:"
ls env_data/ | wc -l
echo "eo_data files:"
ls eo_data/ | wc -l
echo "sapf_data files:"
ls sapf_data/ | wc -l
echo "site_md files:"
ls site_md/ | wc -l

# 5. Verify GPU
echo ""
echo "=== GPU test ==="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"

# 6. Create output directory
mkdir -p "${WORK}/loritz_outputs"
mkdir -p logs

echo ""
echo "=== Setup complete ==="
echo "Data dir: $DATA_DIR"
echo "Output dir: ${WORK}/loritz_outputs"
echo ""
echo "Submit jobs with:"
echo "  sbatch src/paper_replicate/slurm/job_baseline.sh"
echo "  sbatch src/paper_replicate/slurm/job_gauged.sh"
echo "  sbatch src/paper_replicate/slurm/job_ungauged.sh"
