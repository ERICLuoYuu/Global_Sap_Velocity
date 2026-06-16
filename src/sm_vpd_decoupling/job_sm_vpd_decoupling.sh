#!/bin/bash
#SBATCH --job-name=smvpd_decouple
#SBATCH --partition=requeue-zen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/smvpd_decouple_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/smvpd_decouple_%j.err

# Site-level SM-VPD decoupling: E (sap velocity) + Gc (Flo 2021) x 5 SM depths.
# Primary run: site-measured climate, Tair>15C. (Add a Tair>5C sensitivity run
# by re-invoking with --tair-min 5.0 --out-dir .../tair5.)
set -e
hostname; date
cd /scratch/tmp/yluo2/gsv
source .venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv:$PYTHONPATH
python --version

python src/sm_vpd_decoupling/run_sm_vpd_decoupling.py \
    --climate-source site \
    --tair-min 15.0 \
    --n-bins 5 10 \
    --min-valid-days 120 240 360 \
    --out-dir outputs/sm_vpd_decoupling/tair15

echo "=== Output ==="
OUT=outputs/sm_vpd_decoupling/tair15
echo "Tables:"; ls "$OUT"/*.csv 2>/dev/null || true
echo "Figures:"; { ls "$OUT"/figures/*.png 2>/dev/null || true; } | wc -l
date
