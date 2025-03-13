import pandas as pd
import sys
from pathlib import Path
import numpy as np
import seaborn as sns
env_files = list(Path('./outputs/processed_data/env/gap_filled_size1/add_era5_data').glob("*_era5.csv"))
save_dir = Path('./outputs/processed_data/env/daily_gap_filled_size1_with_era5/')
if not save_dir.exists():
    save_dir.mkdir(parents=True, exist_ok=True)
for env_file in env_files:
  # aggregate the data to daily
  df = pd.read_csv(env_file, parse_dates=['TIMESTAMP'])
  df.set_index('TIMESTAMP', inplace=True)
  # remove hourly from colnames
  df.columns = [col.replace('_hourly', '') for col in df.columns]
  data_cols = [col for col in df.columns if col != 'solar_TIMESTAMP' and col and col != 'lat' and col != 'long']
  sum_cols = [col for col in df.columns if 'precip' in col or 'surface' in col or 'evaporation' in col ]
  mean_cols = [col for col in data_cols if col not in sum_cols and 'site_name' not in col]

  # Resample separately
  daily_sums = df[sum_cols].resample('D').sum() if sum_cols else None
  daily_means = df[mean_cols].resample('D').mean() if mean_cols else None

  # Combine results
  if daily_sums is not None and daily_means is not None:
      daily_df = pd.concat([daily_sums, daily_means], axis=1)
  elif daily_sums is not None:
      daily_df = daily_sums
  else:
      daily_df = daily_means
  
  # save the data
  site_name = env_file.stem.split('_')
  site_name = '_'.join(site_name[:-2])
  save_path = save_dir / f'{site_name}_daily_gap_filled_with_era5.csv'
  daily_df.to_csv(save_path, index=True)