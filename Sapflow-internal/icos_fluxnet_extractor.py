"""
Extract environmental data from ICOS Fluxnet Product for sites where
L2 Fluxnet (half-hourly) data doesn't cover the sapflow measurement period.

This script uses the Fluxnet Product data type which has historical data going
back further than the L2 products.

Sites to process:
- ES-LM1: Sapflow 2015-2018, Fluxnet Product 2014-2024
- ES-LM2: Sapflow 2015-2018, Fluxnet Product 2014-2024
- IT-CP2_sapwood: Sapflow 2014-2017, Fluxnet Product 2012-2024
- SE-Nor: Sapflow 2009-2010, no matching data available
- ES-LMa: Sapflow 2015-2018, no matching data available (starts 2020)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from icoscp.station import station
from icoscp.dobj import Dobj
import warnings
warnings.filterwarnings('ignore')

# Base directory - actual location in Sapflow-internal
BASE_DIR = Path(r"E:\OneDrive - Universität Münster\Sap_velocity_project\global-sap-velocity\Sapflow-internal\Sapflow_SAPFLUXNET_format_unitcon")
OUTPUT_DIR = BASE_DIR / "env_icos"
SAPWOOD_DIR = BASE_DIR / "sapwood"
PLANT_DIR = BASE_DIR / "plant"

# Sites to process with Fluxnet Product data
# Format: sapflow_name: (icos_station_id, product_label, data_type)
# data_type: 'sapwood' or 'plant' for finding the correct sapflow file
SITES_TO_PROCESS = {
    # Previously processed
    "ES-LM1": ("ES-LM1", "Fluxnet Product", "sapwood"),
    "ES-LM2": ("ES-LM2", "Fluxnet Product", "sapwood"),
    "IT-CP2_sapwood": ("IT-Cp2", "Fluxnet Product", "sapwood"),
    # Newly identified ICOS sites with overlapping data
    "AT_Mmg": ("AT-Mmg", "Fluxnet Product", "sapwood"),  # Sapflow 2023-2024, ICOS 2021-2024
    "CH-Lae_daily": ("CH-Lae", "Fluxnet Product", "sapwood"),  # Sapflow 2010-2023, ICOS 2004-2024
    "CH-Lae_halfhourly": ("CH-Lae", "Fluxnet Product", "sapwood"),  # Sapflow 2012-2022, ICOS 2004-2024
    "ES-Abr": ("ES-Abr", "Fluxnet Product", "sapwood"),  # Sapflow 2015-2018, ICOS 2015-2020
}

# Variable mapping from Fluxnet Product to our standard names
# Fluxnet Product uses different naming than L2 Fluxnet
VARIABLE_MAPPING = {
    'PA_F': 'pa',           # Pressure
    'P_F': 'precip',        # Precipitation
    'SW_IN_F': 'sw_in',     # Shortwave incoming radiation
    'TA_F': 'ta',           # Air temperature
    'VPD_F': 'vpd',         # Vapor pressure deficit
    'WS_F': 'ws',           # Wind speed
}

QC_MAPPING = {
    'PA_F_QC': 'pa_qc',
    'P_F_QC': 'precip_qc',
    'SW_IN_F_QC': 'sw_in_qc',
    'TA_F_QC': 'ta_qc',
    'VPD_F_QC': 'vpd_qc',
    'WS_F_QC': 'ws_qc',
}


def get_sapflow_time_range(sapflow_name: str, data_type: str) -> tuple:
    """Get the time range of sapflow data for a site."""
    if data_type == 'sapwood':
        sapf_file = SAPWOOD_DIR / f"{sapflow_name}_sapflow_sapwood.csv"
    else:
        sapf_file = PLANT_DIR / f"{sapflow_name}_sapflow_plant.csv"
    
    if not sapf_file.exists():
        print(f"  Warning: Sapflow file not found: {sapf_file}")
        return None, None
    
    df = pd.read_csv(sapf_file, nrows=1)
    
    # Find timestamp column (could be 'TIMESTAMP' or 'timestamp')
    ts_col = None
    for col in df.columns:
        if col.lower() == 'timestamp':
            ts_col = col
            break
    
    if ts_col is None:
        print(f"  Warning: No timestamp column in {sapf_file}")
        return None, None
    
    # Read full file for time range
    df = pd.read_csv(sapf_file, usecols=[ts_col])
    df[ts_col] = pd.to_datetime(df[ts_col])
    
    return df[ts_col].min(), df[ts_col].max()


def get_fluxnet_product_data(icos_id: str, product_label: str) -> pd.DataFrame:
    """Get Fluxnet Product data for a station."""
    st = station.get(icos_id)
    data = st.data()
    
    # Find the product
    product_rows = data[data['specLabel'] == product_label]
    if len(product_rows) == 0:
        print(f"  No {product_label} found for {icos_id}")
        return None
    
    row = product_rows.iloc[0]
    print(f"  Found {product_label}: {row['timeStart'][:10]} to {row['timeEnd'][:10]}")
    
    # Get the data
    dobj = Dobj(row['dobj'])
    df = dobj.data
    
    return df


def process_site(sapflow_name: str, icos_id: str, product_label: str, data_type: str) -> bool:
    """Process a single site, extracting Fluxnet Product data."""
    print(f"\nProcessing {sapflow_name} (ICOS: {icos_id})...")
    
    # Get sapflow time range
    sapf_start, sapf_end = get_sapflow_time_range(sapflow_name, data_type)
    if sapf_start is None:
        print(f"  Skipping: Could not get sapflow time range")
        return False
    
    print(f"  Sapflow period: {sapf_start.date()} to {sapf_end.date()}")
    
    # Get ICOS data
    df = get_fluxnet_product_data(icos_id, product_label)
    if df is None:
        return False
    
    print(f"  ICOS data shape: {df.shape}")
    
    # TIMESTAMP is already datetime in Fluxnet Product
    if 'TIMESTAMP' not in df.columns:
        print(f"  Error: No TIMESTAMP column found")
        return False
    
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    
    # Make timestamps timezone-naive for comparison
    if df['TIMESTAMP'].dt.tz is not None:
        df['TIMESTAMP'] = df['TIMESTAMP'].dt.tz_localize(None)
    
    sapf_start_naive = sapf_start.tz_localize(None) if sapf_start.tzinfo else sapf_start
    sapf_end_naive = sapf_end.tz_localize(None) if sapf_end.tzinfo else sapf_end
    
    # Filter to overlapping time range
    mask = (df['TIMESTAMP'] >= sapf_start_naive) & (df['TIMESTAMP'] <= sapf_end_naive)
    df_filtered = df[mask].copy()
    
    if len(df_filtered) == 0:
        print(f"  Error: No overlapping time range!")
        print(f"    ICOS data: {df['TIMESTAMP'].min().date()} to {df['TIMESTAMP'].max().date()}")
        print(f"    Sapflow: {sapf_start.date()} to {sapf_end.date()}")
        return False
    
    print(f"  Filtered to {len(df_filtered)} rows ({df_filtered['TIMESTAMP'].min().date()} to {df_filtered['TIMESTAMP'].max().date()})")
    
    # Select and rename columns
    output_cols = ['TIMESTAMP']
    
    for old_name, new_name in VARIABLE_MAPPING.items():
        if old_name in df_filtered.columns:
            output_cols.append(old_name)
    
    for old_name, new_name in QC_MAPPING.items():
        if old_name in df_filtered.columns:
            output_cols.append(old_name)
    
    df_output = df_filtered[output_cols].copy()
    
    # Rename columns
    rename_dict = {**VARIABLE_MAPPING, **QC_MAPPING}
    df_output = df_output.rename(columns=rename_dict)
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save
    output_file = OUTPUT_DIR / f"{sapflow_name}_env_icos.csv"
    df_output.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")
    print(f"  Variables: {list(df_output.columns)}")
    
    return True


def main():
    print("=" * 60)
    print("ICOS Fluxnet Product Extractor")
    print("For sites where L2 data doesn't cover sapflow period")
    print("=" * 60)
    
    results = {}
    
    for sapflow_name, (icos_id, product_label, data_type) in SITES_TO_PROCESS.items():
        success = process_site(sapflow_name, icos_id, product_label, data_type)
        results[sapflow_name] = success
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    success_count = sum(1 for v in results.values() if v)
    fail_count = len(results) - success_count
    
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    
    # Sites that cannot be processed
    print("\nNotes:")
    print("  - SE-Nor: Sapflow 2009-2010, but ICOS Fluxnet Product starts 2014")
    print("  - ES-LMa: Sapflow 2015-2018, but ICOS Fluxnet Product starts 2020")


if __name__ == "__main__":
    main()
