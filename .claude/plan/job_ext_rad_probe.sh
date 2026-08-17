#!/bin/bash
#SBATCH --job-name=extradprobe
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/extradprobe_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/extradprobe_%j.err

# Decide what SAPFLUXNET's `ext_rad` column is, from the delivered data, and
# score candidate formulas against the 202 sites that ship it. Nothing is
# synthesised for the 24 internal sites until a candidate reproduces the 202.
set -uo pipefail
cd /scratch/tmp/yluo2/gsv || exit 1

if [ ! -f .venv/bin/activate ]; then
    echo "FATAL: no .venv at /scratch/tmp/yluo2/gsv/.venv -- refusing to run against a bare interpreter"
    exit 1
fi
source .venv/bin/activate

python -c "import pandas, numpy, pvlib; print('pandas', pandas.__version__, '| numpy', numpy.__version__, '| pvlib', pvlib.__version__)" || {
    echo "FATAL: required packages missing in the venv"
    exit 1
}

python .claude/plan/probe_ext_rad.py
RC=$?
echo "=== probe exit code: $RC ==="
exit $RC
