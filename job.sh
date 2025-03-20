cat > job.sh << 'EOF'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=sap_velocity_var_select
#SBATCH --ntasks-per-node=1
#SBATCH --output=output_%j.log
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=36
#SBATCH --partition=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yluo2@uni-muenster.de

# Load required modules
module load palma/2023a
module load GCC/12.3.0
module load OpenMPI/4.1.5
module load SciPy-bundle/2023.07
module load scikit-learn/1.3.1

# Create output directories with proper permissions
mkdir -p /scratch/tmp/yluo2/Global_Sap_Velocity/outputs/checkpoints
mkdir -p /scratch/tmp/yluo2/Global_Sap_Velocity/outputs/logs/var_select_logs
chmod -R 755 /scratch/tmp/yluo2/Global_Sap_Velocity/outputs

# Install CPU-based packages
pip install --user pandas numpy matplotlib seaborn joblib tensorflow

# Run your script
python /scratch/tmp/yluo2/Global_Sap_Velocity/src/variable_selectors/test_GHGA.py
EOF