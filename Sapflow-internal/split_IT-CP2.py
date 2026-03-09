"""
Split IT-CP2 data into plant-level and sapwood-level versions.

Plant-level: Sap1-Sap10 columns (g h-1) -> plant folder
Sapwood-level: TDSV1-TDSV10 columns (cm s-1 velocity) -> sapwood folder
"""

import pandas as pd
from pathlib import Path
import shutil

source_dir = Path(r'e:\OneDrive - Universität Münster\Sap_velocity_project\global-sap-velocity\Sapflow-internal\Sapflow-internal\IT-CP2')
output_base = Path(r'e:\OneDrive - Universität Münster\Sap_velocity_project\global-sap-velocity\Sapflow-internal\Sapflow-internal')

# Create output directories
plant_dir = output_base / 'IT-CP2_plant'
sapwood_dir = output_base / 'IT-CP2_sapwood'

plant_dir.mkdir(exist_ok=True)
sapwood_dir.mkdir(exist_ok=True)

# Common columns (timestamp and environmental)
common_cols = ['DOY', 'TOY', 'Hour', 'month', 'RH', 'Tair', 'Precipitations (mm)', 
               'Evapotransp. (mmol m-2s-1)', 'gpp (umol m-2s-1)', 
               'SoilUR10cm', 'SoilUR50cm', 'SoilUR100cm']

# Plant-level columns (Sap# - mass flow)
plant_cols = ['Sap1 (g h-1)', 'Sap2', 'Sap3', 'Sap4', 'Sap5', 'Sap6', 'Sap9', 'Sap10']

# Sapwood-level columns (TDSV# - sap velocity)
sapwood_cols = ['TDSV1 sap velocity in cm s-1', 'TDSV2', 'TDSV3', 'TDSV4', 'TDSV5', 
                'TDSV6', 'TDSV7', 'TDSV8', 'TDSV9', 'TDSV10']

# Excluded columns (intermediate calculations)
excluded_cols = ['K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8', 'K9', 'K10',
                 'TD1', 'TD2', 'TD3', 'TD4', 'TD5', 'TD6', 'TD7', 'TD8', 'TD9', 'TD10']

# Process each data file
data_files = list(source_dir.glob('SapFlow*.xls'))
print(f"Found {len(data_files)} data file(s)")

for data_file in data_files:
    print(f"\nProcessing: {data_file.name}")
    df = pd.read_excel(data_file)
    
    # Get available columns
    available_common = [c for c in common_cols if c in df.columns]
    available_plant = [c for c in plant_cols if c in df.columns]
    available_sapwood = [c for c in sapwood_cols if c in df.columns]
    
    # Create plant-level version
    plant_df = df[available_common + available_plant].copy()
    plant_output = plant_dir / data_file.name.replace('.xls', '.csv')
    plant_df.to_csv(plant_output, index=False)
    print(f"  Plant-level: {len(available_plant)} Sap columns -> {plant_output.name}")
    
    # Create sapwood-level version
    sapwood_df = df[available_common + available_sapwood].copy()
    sapwood_output = sapwood_dir / data_file.name.replace('.xls', '.csv')
    sapwood_df.to_csv(sapwood_output, index=False)
    print(f"  Sapwood-level: {len(available_sapwood)} TDSV columns -> {sapwood_output.name}")

# Copy metadata to both folders
metadata_file = source_dir / 'IT-Cp2 metadata.xlsx'
if metadata_file.exists():
    # Read metadata and create two versions with appropriate units
    meta_df = pd.read_excel(metadata_file)
    
    # Plant-level metadata (units: g h-1)
    plant_meta = meta_df.copy()
    # Always set/add units column
    plant_meta['units'] = 'g h-1'
    plant_meta.to_excel(plant_dir / 'IT-CP2_plant_metadata.xlsx', index=False)
    print(f"\nPlant metadata saved with units: g h-1")
    
    # Sapwood-level metadata (units: cm s-1)
    sapwood_meta = meta_df.copy()
    # Always set/add units column
    sapwood_meta['units'] = 'cm s-1'
    sapwood_meta.to_excel(sapwood_dir / 'IT-CP2_sapwood_metadata.xlsx', index=False)
    print(f"Sapwood metadata saved with units: cm s-1")

print("\n" + "="*60)
print("Done! Created:")
print(f"  Plant-level:   {plant_dir}")
print(f"  Sapwood-level: {sapwood_dir}")
print("="*60)
