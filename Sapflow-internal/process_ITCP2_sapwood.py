"""Process just IT-CP2_sapwood site to regenerate with correct TDSV columns."""

import pandas as pd
from pathlib import Path
import shutil

# Paths
source_dir = Path(r'e:\OneDrive - Universität Münster\Sap_velocity_project\global-sap-velocity\Sapflow-internal\Sapflow-internal\IT-CP2_sapwood')
output_dir = Path(r'e:\OneDrive - Universität Münster\Sap_velocity_project\global-sap-velocity\Sapflow-internal\Sapflow_SAPFLUXNET_format\sapwood')
output_dir.mkdir(parents=True, exist_ok=True)

# TDSV columns are the sap velocity columns
tdsv_cols = ['TDSV1 sap velocity in cm s-1', 'TDSV2', 'TDSV3', 'TDSV4', 'TDSV5', 
             'TDSV6', 'TDSV7', 'TDSV8', 'TDSV9', 'TDSV10']

# Environmental columns
env_cols = ['RH', 'Tair', 'Precipitations (mm)', 'Evapotransp. (mmol m-2s-1)', 
            'gpp (umol m-2s-1)', 'SoilUR10cm', 'SoilUR50cm', 'SoilUR100cm']

# Read and combine all yearly data files
data_files = sorted(source_dir.glob('SapFlow*.csv'))
print(f"Found {len(data_files)} data file(s)")

all_dfs = []
for f in data_files:
    df = pd.read_csv(f)
    print(f"  {f.name}: shape {df.shape}")
    print(f"    Columns: {df.columns.tolist()}")
    
    # Extract year from filename
    year = int(f.stem.replace('SapFlow', ''))
    df['_file_year'] = year
    all_dfs.append(df)

# Combine
combined_df = pd.concat(all_dfs, ignore_index=True)
print(f"\nCombined shape: {combined_df.shape}")

# Construct timestamp from Year + DOY + Hour
print("\nConstructing timestamps from DOY...")
base_date = pd.to_datetime(combined_df['_file_year'].astype(str) + '-01-01')
doy_offset = pd.to_timedelta(combined_df['DOY'].astype(float) - 1, unit='D')
hour_offset = pd.to_timedelta(combined_df['Hour'].astype(float), unit='h')
combined_df['timestamp'] = base_date + doy_offset + hour_offset

print(f"Timestamp range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")

# Get available TDSV columns
available_tdsv = [c for c in tdsv_cols if c in combined_df.columns]
print(f"\nAvailable TDSV columns ({len(available_tdsv)}): {available_tdsv}")

# Get available env columns
available_env = [c for c in env_cols if c in combined_df.columns]
print(f"Available env columns ({len(available_env)}): {available_env}")

# Create sapflow dataframe with TDSV columns (exclude _file_year)
sapflow_df = combined_df[['timestamp'] + available_tdsv].copy()
sapflow_df = sapflow_df.set_index('timestamp')

# Rename TDSV columns to tree IDs
tree_id_mapping = {
    'TDSV1 sap velocity in cm s-1': 'IT-CP2_sapwood-TDSV1-1',
    'TDSV2': 'IT-CP2_sapwood-TDSV2-1',
    'TDSV3': 'IT-CP2_sapwood-TDSV3-1',
    'TDSV4': 'IT-CP2_sapwood-TDSV4-1',
    'TDSV5': 'IT-CP2_sapwood-TDSV5-1',
    'TDSV6': 'IT-CP2_sapwood-TDSV6-1',
    'TDSV7': 'IT-CP2_sapwood-TDSV7-1',
    'TDSV8': 'IT-CP2_sapwood-TDSV8-1',
    'TDSV9': 'IT-CP2_sapwood-TDSV9-1',
    'TDSV10': 'IT-CP2_sapwood-TDSV10-1'
}
for old_name, new_name in tree_id_mapping.items():
    if old_name in sapflow_df.columns:
        sapflow_df.rename(columns={old_name: new_name}, inplace=True)

# Remove duplicates and sort
sapflow_df = sapflow_df.loc[~sapflow_df.index.duplicated(keep='first')]
sapflow_df = sapflow_df.sort_index()

print(f"\nSapflow data shape: {sapflow_df.shape}")
print(f"Sapflow columns: {sapflow_df.columns.tolist()}")
print(f"Sample data:\n{sapflow_df.head()}")

# Save sapflow file
sapflow_file = output_dir / 'IT-CP2_sapwood_sapflow_sapwood.csv'
sapflow_df.to_csv(sapflow_file)
print(f"\nSaved: {sapflow_file}")

# Create env data
if available_env:
    env_df = combined_df[['timestamp'] + available_env].copy()
    env_df = env_df.set_index('timestamp')
    env_df = env_df.loc[~env_df.index.duplicated(keep='first')]
    env_df = env_df.sort_index()
    env_file = output_dir / 'IT-CP2_sapwood_env_data.csv'
    env_df.to_csv(env_file)
    print(f"Saved: {env_file}")

# Create plant_md with correct units
plant_codes = [v for v in tree_id_mapping.values()]
tree_ids = [f'TDSV{i}' for i in range(1, 11)]
plant_md = pd.DataFrame({
    'pl_code': plant_codes[:len(available_tdsv)],
    'tree_id': tree_ids[:len(available_tdsv)],
    'pl_sap_units': 'cm s-1',
    'pl_sap_units_orig': 'cm s-1'
})
plant_md_file = output_dir / 'IT-CP2_sapwood_plant_md.csv'
plant_md.to_csv(plant_md_file, index=False)
print(f"Saved: {plant_md_file}")

# Create env_md
env_md = pd.DataFrame({'env_var': available_env})
env_md_file = output_dir / 'IT-CP2_sapwood_env_md.csv'
env_md.to_csv(env_md_file, index=False)
print(f"Saved: {env_md_file}")

# Create site_md
site_md = pd.DataFrame({'si_code': ['IT-CP2_sapwood'], 'latitude': [45.201], 'longitude': [7.567]})
site_md_file = output_dir / 'IT-CP2_sapwood_site_md.csv'
site_md.to_csv(site_md_file, index=False)
print(f"Saved: {site_md_file}")

# Create stand_md
stand_md = pd.DataFrame({'si_code': ['IT-CP2_sapwood']})
stand_md_file = output_dir / 'IT-CP2_sapwood_stand_md.csv'
stand_md.to_csv(stand_md_file, index=False)
print(f"Saved: {stand_md_file}")

print("\n" + "="*60)
print("Done! IT-CP2_sapwood files created in sapwood folder")
print(f"Units: cm s-1 (will be converted to cm h-1 by unit converter)")
print("="*60)
