"""Process IT-CP2_plant site with Sap columns (g h-1) to plant folder."""

import pandas as pd
from pathlib import Path

# Paths
source_dir = Path(r'e:\OneDrive - Universität Münster\Sap_velocity_project\global-sap-velocity\Sapflow-internal\Sapflow-internal\IT-CP2_plant')
output_dir = Path(r'e:\OneDrive - Universität Münster\Sap_velocity_project\global-sap-velocity\Sapflow-internal\Sapflow_SAPFLUXNET_format\plant')
output_dir.mkdir(parents=True, exist_ok=True)

# Sap columns are the plant-level (mass flow) columns
sap_cols = ['Sap1 (g h-1)', 'Sap2', 'Sap3', 'Sap4', 'Sap5', 'Sap6', 'Sap9', 'Sap10']

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

# Get available Sap columns
available_sap = [c for c in sap_cols if c in combined_df.columns]
print(f"\nAvailable Sap columns ({len(available_sap)}): {available_sap}")

# Get available env columns
available_env = [c for c in env_cols if c in combined_df.columns]
print(f"Available env columns ({len(available_env)}): {available_env}")

# Create sapflow dataframe with Sap columns
sapflow_df = combined_df[['timestamp'] + available_sap].copy()
sapflow_df = sapflow_df.set_index('timestamp')

# Rename Sap columns to tree IDs
tree_id_mapping = {
    'Sap1 (g h-1)': 'IT-CP2_plant-Sap1-1',
    'Sap2': 'IT-CP2_plant-Sap2-1',
    'Sap3': 'IT-CP2_plant-Sap3-1',
    'Sap4': 'IT-CP2_plant-Sap4-1',
    'Sap5': 'IT-CP2_plant-Sap5-1',
    'Sap6': 'IT-CP2_plant-Sap6-1',
    'Sap9': 'IT-CP2_plant-Sap9-1',
    'Sap10': 'IT-CP2_plant-Sap10-1'
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
sapflow_file = output_dir / 'IT-CP2_plant_sapflow_plant.csv'
sapflow_df.to_csv(sapflow_file)
print(f"\nSaved: {sapflow_file}")

# Create env data
if available_env:
    env_df = combined_df[['timestamp'] + available_env].copy()
    env_df = env_df.set_index('timestamp')
    env_df = env_df.loc[~env_df.index.duplicated(keep='first')]
    env_df = env_df.sort_index()
    env_file = output_dir / 'IT-CP2_plant_env_data.csv'
    env_df.to_csv(env_file)
    print(f"Saved: {env_file}")

# Create plant_md with correct units
plant_codes = [v for v in tree_id_mapping.values()]
tree_ids = ['Sap1', 'Sap2', 'Sap3', 'Sap4', 'Sap5', 'Sap6', 'Sap9', 'Sap10']
plant_md = pd.DataFrame({
    'pl_code': plant_codes[:len(available_sap)],
    'tree_id': tree_ids[:len(available_sap)],
    'pl_sap_units': 'g h-1',
    'pl_sap_units_orig': 'g h-1'
})
plant_md_file = output_dir / 'IT-CP2_plant_plant_md.csv'
plant_md.to_csv(plant_md_file, index=False)
print(f"Saved: {plant_md_file}")

# Create env_md
env_md = pd.DataFrame({'env_var': available_env})
env_md_file = output_dir / 'IT-CP2_plant_env_md.csv'
env_md.to_csv(env_md_file, index=False)
print(f"Saved: {env_md_file}")

# Create site_md
site_md = pd.DataFrame({'si_code': ['IT-CP2_plant'], 'latitude': [45.201], 'longitude': [7.567]})
site_md_file = output_dir / 'IT-CP2_plant_site_md.csv'
site_md.to_csv(site_md_file, index=False)
print(f"Saved: {site_md_file}")

# Create stand_md
stand_md = pd.DataFrame({'si_code': ['IT-CP2_plant']})
stand_md_file = output_dir / 'IT-CP2_plant_stand_md.csv'
stand_md.to_csv(stand_md_file, index=False)
print(f"Saved: {stand_md_file}")

print("\n" + "="*60)
print("Done! IT-CP2_plant files created in plant folder")
print(f"Units: g h-1 (will be converted to cm3 h-1 by unit converter)")
print("="*60)
