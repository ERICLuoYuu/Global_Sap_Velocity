import pandas as pd
import re

# Simulate tidy_metadata for SE-Nor
file_path = r'e:\OneDrive - Universität Münster\Sap_velocity_project\global-sap-velocity\Sapflow-internal\Sapflow-internal\SE-Nor\Metadata_Sapflow_SE-Nor.xlsx'
all_sheets = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')

site_sheet_name = next((s for s in all_sheets if 'site' in s.lower()), None)
sapflow_sheet_name = next((s for s in all_sheets if 'sapflow' in s.lower() or 'tree' in s.lower()), None)

site_df_wide = all_sheets[site_sheet_name].set_index(all_sheets[site_sheet_name].columns[0])
site_df_wide = site_df_wide.loc[~site_df_wide.index.duplicated(keep='first')]

plant_info = all_sheets[sapflow_sheet_name].set_index(all_sheets[sapflow_sheet_name].columns[0])
plant_info = plant_info.loc[~plant_info.index.duplicated(keep='first')].transpose().reset_index(drop=True)

# Merge site info
for col in site_df_wide.transpose().columns:
    plant_info[col] = site_df_wide.transpose()[col].iloc[0]

meta_df_tidy = plant_info

# Clean column names
def clean_col(c):
    return str(c).split('(')[0].strip() if isinstance(c, str) else c
meta_df_tidy.rename(columns=clean_col, inplace=True)

# Check first row
row = meta_df_tidy.iloc[0]
print('sensor_start:', row.get('sensor_start', 'NOT FOUND'))
print('sensor_stop:', row.get('sensor_stop', 'NOT FOUND'))  
print('data frequency:', row.get('data frequency', 'NOT FOUND'))

# Now try generate_timestamps_from_metadata
start_time = pd.to_datetime(row.get('sensor_start', None), errors='coerce')
end_time = pd.to_datetime(row.get('sensor_stop', None), errors='coerce')
freq_str = str(row.get('data frequency', '')).strip()

print()
print(f'start_time: {start_time}')
print(f'end_time: {end_time}')
print(f'freq_str: "{freq_str}"')

# Parse frequency
freq_value = re.search(r'(\d+)', freq_str)
if freq_value:
    freq_num = int(freq_value.group(1))
    print(f'freq_num: {freq_num}')
    if 'min' in freq_str.lower():
        freq = f'{freq_num}min'
    print(f'freq: {freq}')
    
    # Generate timestamps
    n_rows = 35038
    timestamps = pd.date_range(start=start_time, periods=n_rows, freq=freq)
    print(f'Generated {len(timestamps)} timestamps')
    print(f'First: {timestamps[0]}, Last: {timestamps[-1]}')
