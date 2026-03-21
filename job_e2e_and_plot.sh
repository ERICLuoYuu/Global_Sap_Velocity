#!/bin/bash
#SBATCH --job-name=gf-e2e-cap
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=e2e_cap_%j.out
#SBATCH --error=e2e_cap_%j.err

module load palma/2023b
module load CUDA/12.4.0

VENV="/scratch/tmp/yluo2/gsv/.venv"
WORKTREE="/scratch/tmp/yluo2/gsv-wt/fix-gap-benchmark"
source "${VENV}/bin/activate"
cd "${WORKTREE}"
export PYTHONPATH="${WORKTREE}"

echo "=== Step 1: E2E tests (with 72h cap) ==="
python -m pytest src/gap_filling/tests/test_e2e_real_site.py -v -s 2>&1

echo ""
echo "=== Step 2: Gap-fill + plot test sites ==="
python3 -c "
import pandas as pd
from pathlib import Path
from src.gap_filling.config import GapFillingConfig
from src.gap_filling.filler import GapFiller

DATA = Path('outputs/processed_data/sapwood')
SAP_DIR = DATA / 'sap' / 'outliers_removed'
ENV_DIR = DATA / 'env' / 'outliers_removed'
GF_DIR = DATA / 'sap' / 'gap_filled'
MASK_DIR = DATA / 'sap' / 'gap_filled_masks'
GF_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

config = GapFillingConfig(time_scale='hourly')
print(f'max_gap_hours = {config.max_gap_hours}')
filler = GapFiller(config, target='sap')

for site in ['AUS_WOM', 'DEU_HIN_OAK']:
    sap_files = list(SAP_DIR.glob(f'{site}_*_outliers_removed.csv'))
    env_files = list(ENV_DIR.glob(f'{site}_*_outliers_removed.csv'))
    if not sap_files or not env_files:
        continue
    print(f'\n=== {site} ===')
    sap_df = pd.read_csv(sap_files[0], parse_dates=['TIMESTAMP', 'solar_TIMESTAMP'])
    env_df = pd.read_csv(env_files[0], parse_dates=['TIMESTAMP', 'solar_TIMESTAMP'])
    meta_cols = [c for c in sap_df.columns if 'TIMESTAMP' in c.upper()]
    meta = sap_df[meta_cols].copy()
    sap_num = sap_df.set_index('TIMESTAMP').drop(columns=[c for c in meta_cols if c != 'TIMESTAMP'], errors='ignore')
    sap_num = sap_num.apply(pd.to_numeric, errors='coerce')
    env_num = env_df.set_index('TIMESTAMP').drop(columns=[c for c in meta_cols if c != 'TIMESTAMP'], errors='ignore')
    env_num = env_num.apply(pd.to_numeric, errors='coerce')
    pre_fill = sap_num.copy()
    filled = filler.fill_dataframe(sap_num, env_df=env_num)
    # Save gap-filled site CSV
    out = meta.copy()
    out.index = sap_df.index
    for col in filled.columns:
        out[col] = filled[col].values
    out_name = sap_files[0].name.replace('outliers_removed', 'gap_filled')
    out.to_csv(GF_DIR / out_name, index=False)
    # Save per-column masks
    solar_ts = sap_df['solar_TIMESTAMP']
    for col in filled.columns:
        was_nan = pre_fill[col].isna()
        now_filled = filled[col].notna()
        is_gf = was_nan & now_filled
        if not is_gf.any():
            continue
        mask_out = pd.DataFrame({'solar_TIMESTAMP': solar_ts.values, 'value': filled[col].values, 'is_gap_filled': is_gf.values}, index=sap_df.index)
        mask_out.to_csv(MASK_DIR / f'{col}_gap_filled.csv', index_label='timestamp')
    n_before = pre_fill.isna().sum().sum()
    n_after = filled.isna().sum().sum()
    print(f'  NaN: {n_before} -> {n_after} ({n_before - n_after} filled)')
"

echo ""
echo "=== Step 3: Generate plots ==="
python -m src.plot_gap_filled_sites 2>&1

echo ""
echo "=== Done: $(date) ==="
