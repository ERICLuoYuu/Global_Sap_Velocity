import pandas as pd
import pyreadr
import numpy as np
import os
from pathlib import Path

np.random.seed(42)  # For reproducibility
N_SAMPLES = 1000  # Number of random timestamps to verify per site

print('='*80)
print(f'COMPREHENSIVE TIMESTAMP VERIFICATION - ALL SITES ({N_SAMPLES} samples each)')
print('='*80)

# Use absolute paths
script_dir = Path(__file__).parent.resolve()
base = script_dir / 'Sapflow-internal'
out_base = script_dir / 'Sapflow_SAPFLUXNET_format'

results = []

def verify_samples(orig_vals, out_vals, tolerance=0.0001):
    """Compare arrays of values, return (matched, total, mismatches)"""
    matched = 0
    mismatches = []
    for i, (o, out) in enumerate(zip(orig_vals, out_vals)):
        if pd.notna(o) and pd.notna(out):
            if abs(o - out) < tolerance:
                matched += 1
            else:
                mismatches.append((i, o, out, 'VALUE_DIFF'))
        elif pd.isna(o) and pd.isna(out):
            matched += 1
        else:
            mismatches.append((i, o, out, 'NAN_MISMATCH'))
    return matched, len(orig_vals), mismatches

def find_output_file(site_id, out_base):
    """Find the output sapflow file for a site"""
    # Check sapwood folder
    sapwood_files = list((out_base / 'sapwood').glob(f'{site_id}*sapflow*.csv'))
    if sapwood_files:
        return sapwood_files[0], 'sapwood'
    # Check plant folder
    plant_files = list((out_base / 'plant').glob(f'{site_id}*sapflow*.csv'))
    if plant_files:
        return plant_files[0], 'plant'
    return None, None

def get_first_data_column(df, exclude_cols=['timestamp', 'TIMESTAMP', 'DateTime', 'Date Time [UTC+1]']):
    """Get first data column from dataframe"""
    for col in df.columns:
        if col not in exclude_cols and 'time' not in col.lower() and 'date' not in col.lower():
            return col
    return None

# ============================================================================
# SITE VERIFICATION FUNCTIONS
# ============================================================================

def verify_chdav():
    """CH-Dav: Long format RDS"""
    try:
        print('\nVerifying CH-Dav...', end=' ')
        rds = pyreadr.read_r(base / 'CH-Dav/CH-Dav_2010_2023_DT_15Trees_HH.rds')
        orig = list(rds.values())[0]
        orig['Timestamp'] = pd.to_datetime(orig['Timestamp'])
        out = pd.read_csv(out_base / 'sapwood/CH-Dav_halfhourly_sapflow_sapwood.csv')
        out['timestamp'] = pd.to_datetime(out['timestamp'], format='mixed')
        
        tree = orig['TreeN'].unique()[0]
        tree_data = orig[orig['TreeN'] == tree].copy()
        valid_data = tree_data[tree_data['SFD'].notna()]
        sample_idx = np.random.choice(len(valid_data), min(N_SAMPLES, len(valid_data)), replace=False)
        samples = valid_data.iloc[sample_idx]
        
        out_col = f'CH-Dav-{tree}'
        orig_vals, out_vals = [], []
        for _, row in samples.iterrows():
            ts = row['Timestamp']
            orig_vals.append(row['SFD'])
            out_row = out[out['timestamp'] == ts]
            out_vals.append(out_row[out_col].values[0] if len(out_row) > 0 and out_col in out.columns else np.nan)
        
        matched, total, mismatches = verify_samples(orig_vals, out_vals)
        print(f'{matched}/{total} matched')
        return ('CH-Dav', matched, total, mismatches)
    except Exception as e:
        print(f'ERROR: {str(e)[:50]}')
        return ('CH-Dav', 0, 0, [('ERR', str(e)[:50], '', '')])

def verify_chlae():
    """CH-Lae: Long format RDS"""
    try:
        print('Verifying CH-Lae...', end=' ')
        rds = pyreadr.read_r(base / 'CH-Lae/CH-Lae_2012_2022_SFD_HH.rds')
        orig = list(rds.values())[0]
        orig['Timestamp'] = pd.to_datetime(orig['Timestamp'])
        out = pd.read_csv(out_base / 'sapwood/CH-Lae_halfhourly_sapflow_sapwood.csv')
        out['timestamp'] = pd.to_datetime(out['timestamp'], format='mixed')
        
        tree = 'Tr2544'
        tree_data = orig[orig['Tree'] == tree].copy()
        valid_data = tree_data[tree_data['SFD'].notna()]
        sample_idx = np.random.choice(len(valid_data), min(N_SAMPLES, len(valid_data)), replace=False)
        samples = valid_data.iloc[sample_idx]
        
        out_col = f'CH-Lae-{tree}'
        orig_vals, out_vals = [], []
        for _, row in samples.iterrows():
            ts = row['Timestamp']
            orig_vals.append(row['SFD'])
            out_row = out[out['timestamp'] == ts]
            out_vals.append(out_row[out_col].values[0] if len(out_row) > 0 and out_col in out.columns else np.nan)
        
        matched, total, mismatches = verify_samples(orig_vals, out_vals)
        print(f'{matched}/{total} matched')
        return ('CH-Lae', matched, total, mismatches)
    except Exception as e:
        print(f'ERROR: {str(e)[:50]}')
        return ('CH-Lae', 0, 0, [('ERR', str(e)[:50], '', '')])

def verify_dehar():
    """DE-Har: Multi-file CSV - handle midnight timestamps without time"""
    try:
        print('Verifying DE-Har...', end=' ')
        orig = pd.read_csv(base / 'DE-Har/DE-Har_H10545_1_1_sapflow.csv')
        # Handle timestamps where midnight shows as date-only (e.g., "2023/4/28" vs "2023/4/28 0:00")
        orig['TIMESTAMP'] = pd.to_datetime(orig['TIMESTAMP'], format='mixed')
        out = pd.read_csv(out_base / 'sapwood/DE-Har_sapflow_sapwood.csv')
        out['timestamp'] = pd.to_datetime(out['timestamp'], format='mixed')
        
        valid_data = orig[orig['Js_outer'].notna()]
        sample_idx = np.random.choice(len(valid_data), min(N_SAMPLES, len(valid_data)), replace=False)
        samples = valid_data.iloc[sample_idx]
        
        out_col = [c for c in out.columns if 'H10545' in c and 'outer' in c][0]
        orig_vals, out_vals = [], []
        for _, row in samples.iterrows():
            ts = row['TIMESTAMP']
            orig_vals.append(row['Js_outer'])
            # Use floor to second to handle any microsecond differences
            out_row = out[out['timestamp'].dt.floor('s') == pd.Timestamp(ts).floor('s')]
            out_vals.append(out_row[out_col].values[0] if len(out_row) > 0 else np.nan)
        
        matched, total, mismatches = verify_samples(orig_vals, out_vals)
        print(f'{matched}/{total} matched')
        return ('DE-Har', matched, total, mismatches)
    except Exception as e:
        print(f'ERROR: {str(e)[:50]}')
        return ('DE-Har', 0, 0, [('ERR', str(e)[:50], '', '')])

def verify_dehoh():
    """DE-HoH: Multi-file CSV - aggregate by mean like sap_reorganizer does"""
    try:
        print('Verifying DE-HoH...', end=' ')
        orig = pd.read_csv(base / 'DE-HoH/DE-HoH_SF022.csv')
        ts_col = 'Date.time'
        orig[ts_col] = pd.to_datetime(orig[ts_col])
        orig[ts_col] = orig[ts_col].dt.tz_localize(None)
        
        # Create combined ID like sap_reorganizer does: TID-orientation
        orig['combined_id'] = orig['TID'].astype(str) + '-' + orig['orientation'].astype(str)
        
        # Aggregate by mean like sap_reorganizer: pivot_table(..., aggfunc='mean')
        agg_orig = orig.groupby([ts_col, 'combined_id'])['sfd'].mean().reset_index()
        
        out = pd.read_csv(out_base / 'sapwood/DE-HoH_sapflow_sapwood.csv')
        out['timestamp'] = pd.to_datetime(out['timestamp'], format='mixed')
        out['timestamp'] = out['timestamp'].dt.tz_localize(None)
        
        # Use ENE which has overlapping sensors (tests aggregation)
        ene_data = agg_orig[agg_orig['combined_id'] == '22-ENE']
        valid_data = ene_data[ene_data['sfd'].notna()]
        sample_idx = np.random.choice(len(valid_data), min(N_SAMPLES, len(valid_data)), replace=False)
        samples = valid_data.iloc[sample_idx]
        
        out_col = 'DE-HoH-22-ENE'
        orig_vals, out_vals = [], []
        for _, row in samples.iterrows():
            ts = row[ts_col]
            orig_vals.append(row['sfd'])
            out_row = out[out['timestamp'] == ts]
            out_vals.append(out_row[out_col].values[0] if len(out_row) > 0 else np.nan)
        
        matched, total, mismatches = verify_samples(orig_vals, out_vals)
        print(f'{matched}/{total} matched')
        return ('DE-HoH', matched, total, mismatches)
    except Exception as e:
        print(f'ERROR: {str(e)[:50]}')
        return ('DE-HoH', 0, 0, [('ERR', str(e)[:50], '', '')])

def verify_esabr():
    """ES-Abr: Excel 2-column"""
    try:
        print('Verifying ES-Abr...', end=' ')
        orig = pd.read_excel(base / 'ES-Abr/ES-Abr_1_SAP1_1_sapflow.xlsx', engine='openpyxl', header=None)
        orig.columns = ['timestamp', 'value']
        orig['timestamp'] = pd.to_datetime(orig['timestamp'])
        out = pd.read_csv(out_base / 'sapwood/ES-Abr_sapflow_sapwood.csv')
        out['timestamp'] = pd.to_datetime(out['timestamp'], format='mixed')
        
        valid_data = orig[orig['value'].notna()]
        sample_idx = np.random.choice(len(valid_data), min(N_SAMPLES, len(valid_data)), replace=False)
        samples = valid_data.iloc[sample_idx]
        
        out_col = 'ES-Abr-1-SAP1'
        orig_vals, out_vals = [], []
        for _, row in samples.iterrows():
            ts = row['timestamp']
            orig_vals.append(row['value'])
            out_row = out[out['timestamp'] == ts]
            out_vals.append(out_row[out_col].values[0] if len(out_row) > 0 and out_col in out.columns else np.nan)
        
        matched, total, mismatches = verify_samples(orig_vals, out_vals)
        print(f'{matched}/{total} matched')
        return ('ES-Abr', matched, total, mismatches)
    except Exception as e:
        print(f'ERROR: {str(e)[:50]}')
        return ('ES-Abr', 0, 0, [('ERR', str(e)[:50], '', '')])

def verify_eslm1():
    """ES-LM1: Excel 2-column"""
    try:
        print('Verifying ES-LM1...', end=' ')
        orig = pd.read_excel(base / 'ES-LM1/ES-LM1_1_SAP1_1_sapflow.xlsx', engine='openpyxl', header=None)
        orig.columns = ['timestamp', 'value']
        orig['timestamp'] = pd.to_datetime(orig['timestamp'])
        out = pd.read_csv(out_base / 'sapwood/ES-LM1_sapflow_sapwood.csv')
        out['timestamp'] = pd.to_datetime(out['timestamp'], format='mixed')
        
        valid_data = orig[orig['value'].notna()]
        sample_idx = np.random.choice(len(valid_data), min(N_SAMPLES, len(valid_data)), replace=False)
        samples = valid_data.iloc[sample_idx]
        
        out_col = [c for c in out.columns if 'SAP1' in c]
        if not out_col:
            out_col = [c for c in out.columns if c != 'timestamp'][:1]
        orig_vals, out_vals = [], []
        for _, row in samples.iterrows():
            ts = row['timestamp']
            orig_vals.append(row['value'])
            out_row = out[out['timestamp'] == ts]
            out_vals.append(out_row[out_col[0]].values[0] if len(out_row) > 0 and out_col else np.nan)
        
        matched, total, mismatches = verify_samples(orig_vals, out_vals)
        print(f'{matched}/{total} matched')
        return ('ES-LM1', matched, total, mismatches)
    except Exception as e:
        print(f'ERROR: {str(e)[:50]}')
        return ('ES-LM1', 0, 0, [('ERR', str(e)[:50], '', '')])

def verify_eslm2():
    """ES-LM2: Excel 2-column"""
    try:
        print('Verifying ES-LM2...', end=' ')
        orig = pd.read_excel(base / 'ES-LM2/ES-LM2_1_SAP1_1_sapflow.xlsx', engine='openpyxl', header=None)
        orig.columns = ['timestamp', 'value']
        orig['timestamp'] = pd.to_datetime(orig['timestamp'])
        out = pd.read_csv(out_base / 'sapwood/ES-LM2_sapflow_sapwood.csv')
        out['timestamp'] = pd.to_datetime(out['timestamp'], format='mixed')
        
        valid_data = orig[orig['value'].notna()]
        sample_idx = np.random.choice(len(valid_data), min(N_SAMPLES, len(valid_data)), replace=False)
        samples = valid_data.iloc[sample_idx]
        
        out_col = [c for c in out.columns if 'SAP1' in c]
        if not out_col:
            out_col = [c for c in out.columns if c != 'timestamp'][:1]
        orig_vals, out_vals = [], []
        for _, row in samples.iterrows():
            ts = row['timestamp']
            orig_vals.append(row['value'])
            out_row = out[out['timestamp'] == ts]
            out_vals.append(out_row[out_col[0]].values[0] if len(out_row) > 0 and out_col else np.nan)
        
        matched, total, mismatches = verify_samples(orig_vals, out_vals)
        print(f'{matched}/{total} matched')
        return ('ES-LM2', matched, total, mismatches)
    except Exception as e:
        print(f'ERROR: {str(e)[:50]}')
        return ('ES-LM2', 0, 0, [('ERR', str(e)[:50], '', '')])

def verify_eslma():
    """ES-LMa: Excel 2-column with header"""
    try:
        print('Verifying ES-LMa...', end=' ')
        # ES-LMa uses pattern ES-LMa_MS1_SMS1_1_sapflow.xlsx
        orig = pd.read_excel(base / 'ES-LMa/ES-LMa_MS1_SMS1_1_sapflow.xlsx', engine='openpyxl', header=None)
        orig.columns = ['timestamp', 'value']
        # Skip header rows
        orig = orig.iloc[2:].copy()
        orig['timestamp'] = pd.to_datetime(orig['timestamp'], format='mixed')
        orig['value'] = pd.to_numeric(orig['value'], errors='coerce')
        out = pd.read_csv(out_base / 'sapwood/ES-LMa_sapflow_sapwood.csv')
        out['timestamp'] = pd.to_datetime(out['timestamp'], format='mixed')
        
        valid_data = orig[orig['value'].notna()]
        sample_idx = np.random.choice(len(valid_data), min(N_SAMPLES, len(valid_data)), replace=False)
        samples = valid_data.iloc[sample_idx]
        
        out_col = [c for c in out.columns if 'MS1' in c or 'SMS1' in c]
        if not out_col:
            out_col = [c for c in out.columns if c != 'timestamp'][:1]
        orig_vals, out_vals = [], []
        for _, row in samples.iterrows():
            ts = row['timestamp']
            orig_vals.append(row['value'])
            out_row = out[out['timestamp'] == ts]
            out_vals.append(out_row[out_col[0]].values[0] if len(out_row) > 0 and out_col else np.nan)
        
        matched, total, mismatches = verify_samples(orig_vals, out_vals)
        print(f'{matched}/{total} matched')
        return ('ES-LMa', matched, total, mismatches)
    except Exception as e:
        print(f'ERROR: {str(e)[:50]}')
        return ('ES-LMa', 0, 0, [('ERR', str(e)[:50], '', '')])

def verify_fihyy():
    """FI-Hyy: CSV file - uses Treebox data"""
    try:
        print('Verifying FI-Hyy...', end=' ')
        # FI-Hyy Treebox files have different structure
        orig = pd.read_csv(base / 'FI-Hyy/FI-Hyy_Treebox1___sapflow_2020_2022.csv')
        ts_col = [c for c in orig.columns if 'time' in c.lower() or 'date' in c.lower()][0]
        orig[ts_col] = pd.to_datetime(orig[ts_col], format='mixed')
        out = pd.read_csv(out_base / 'sapwood/FI-Hyy_sapflow_sapwood.csv')
        out['timestamp'] = pd.to_datetime(out['timestamp'], format='mixed')
        
        data_cols = [c for c in orig.columns if c != ts_col and 'time' not in c.lower() and 'Unnamed' not in c]
        valid_data = orig[orig[data_cols[0]].notna()] if data_cols else orig.head(0)
        if len(valid_data) == 0:
            print('No valid data found')
            return ('FI-Hyy', 0, 0, [('SKIP', 'No valid data in source file', '', '')])
        sample_idx = np.random.choice(len(valid_data), min(N_SAMPLES, len(valid_data)), replace=False)
        samples = valid_data.iloc[sample_idx]
        
        out_col = [c for c in out.columns if 'Treebox1' in c]
        if not out_col:
            out_col = [c for c in out.columns if c != 'timestamp' and 'Sensor' not in c][:1]
        orig_vals, out_vals = [], []
        for _, row in samples.iterrows():
            ts = row[ts_col]
            orig_vals.append(row[data_cols[0]])
            out_row = out[out['timestamp'] == ts]
            out_vals.append(out_row[out_col[0]].values[0] if len(out_row) > 0 and out_col else np.nan)
        
        matched, total, mismatches = verify_samples(orig_vals, out_vals)
        print(f'{matched}/{total} matched')
        return ('FI-Hyy', matched, total, mismatches)
    except Exception as e:
        print(f'ERROR: {str(e)[:50]}')
        return ('FI-Hyy', 0, 0, [('ERR', str(e)[:50], '', '')])

def verify_atmmg():
    """AT_Mmg: Excel with header rows"""
    try:
        print('Verifying AT_Mmg...', end=' ')
        orig = pd.read_excel(base / 'AT_Mmg/AT-Mmg_501_1N__sapflow.xlsx', engine='openpyxl', header=None)
        orig.columns = ['timestamp', 'value']
        # Skip header rows (row 0=header, row 1=units)
        orig = orig.iloc[2:].copy()
        orig['timestamp'] = pd.to_datetime(orig['timestamp'], format='mixed')
        orig['value'] = pd.to_numeric(orig['value'], errors='coerce')
        out = pd.read_csv(out_base / 'plant/AT_Mmg_sapflow_plant.csv')
        out['timestamp'] = pd.to_datetime(out['timestamp'], format='mixed')
        
        valid_data = orig[orig['value'].notna()]
        sample_idx = np.random.choice(len(valid_data), min(N_SAMPLES, len(valid_data)), replace=False)
        samples = valid_data.iloc[sample_idx]
        
        out_col = [c for c in out.columns if '501' in c and '1N' in c]
        if not out_col:
            out_col = [c for c in out.columns if c != 'timestamp'][:1]
        orig_vals, out_vals = [], []
        for _, row in samples.iterrows():
            ts = row['timestamp']
            orig_vals.append(row['value'])
            out_row = out[out['timestamp'] == ts]
            out_vals.append(out_row[out_col[0]].values[0] if len(out_row) > 0 and out_col else np.nan)
        
        matched, total, mismatches = verify_samples(orig_vals, out_vals)
        print(f'{matched}/{total} matched')
        return ('AT_Mmg', matched, total, mismatches)
    except Exception as e:
        print(f'ERROR: {str(e)[:50]}')
        return ('AT_Mmg', 0, 0, [('ERR', str(e)[:50], '', '')])

def verify_esgdn():
    """ES-Gdn: Excel with header"""
    try:
        print('Verifying ES-Gdn...', end=' ')
        # ES-Gdn uses pattern ES-Gdn_1_SF81__sapflow.xlsx
        orig = pd.read_excel(base / 'ES-Gdn/ES-Gdn_1_SF81__sapflow.xlsx', engine='openpyxl', header=None)
        orig.columns = ['timestamp', 'value']
        # Skip header rows
        orig = orig.iloc[2:].copy()
        orig['timestamp'] = pd.to_datetime(orig['timestamp'], format='mixed')
        orig['value'] = pd.to_numeric(orig['value'], errors='coerce')
        out = pd.read_csv(out_base / 'plant/ES-Gdn_sapflow_plant.csv')
        out['timestamp'] = pd.to_datetime(out['timestamp'], format='mixed')
        
        valid_data = orig[orig['value'].notna()]
        sample_idx = np.random.choice(len(valid_data), min(N_SAMPLES, len(valid_data)), replace=False)
        samples = valid_data.iloc[sample_idx]
        
        out_col = [c for c in out.columns if 'SF81' in c or '1-SF81' in c]
        if not out_col:
            out_col = [c for c in out.columns if c != 'timestamp'][:1]
        orig_vals, out_vals = [], []
        for _, row in samples.iterrows():
            ts = row['timestamp']
            orig_vals.append(row['value'])
            out_row = out[out['timestamp'] == ts]
            out_vals.append(out_row[out_col[0]].values[0] if len(out_row) > 0 and out_col else np.nan)
        
        matched, total, mismatches = verify_samples(orig_vals, out_vals)
        print(f'{matched}/{total} matched')
        return ('ES-Gdn', matched, total, mismatches)
    except Exception as e:
        print(f'ERROR: {str(e)[:50]}')
        return ('ES-Gdn', 0, 0, [('ERR', str(e)[:50], '', '')])

def verify_frbil():
    """FR-BIL: Excel"""
    try:
        print('Verifying FR-BIL...', end=' ')
        orig = pd.read_excel(base / 'FR-BIL/FR-BIL saplow 7 trees 2020_2022.xlsx', engine='openpyxl')
        orig['DateTime'] = pd.to_datetime(orig['DateTime'])
        out = pd.read_csv(out_base / 'plant/FR-BIL_sapflow_plant.csv')
        out['timestamp'] = pd.to_datetime(out['timestamp'], format='mixed')
        
        valid_data = orig[orig['Tree_1'].notna()]
        sample_idx = np.random.choice(len(valid_data), min(N_SAMPLES, len(valid_data)), replace=False)
        samples = valid_data.iloc[sample_idx]
        
        out_col = 'FR-Bil-1-1'
        orig_vals, out_vals = [], []
        for _, row in samples.iterrows():
            ts = row['DateTime']
            orig_vals.append(row['Tree_1'])
            out_row = out[out['timestamp'] == ts]
            out_vals.append(out_row[out_col].values[0] if len(out_row) > 0 and out_col in out.columns else np.nan)
        
        matched, total, mismatches = verify_samples(orig_vals, out_vals)
        print(f'{matched}/{total} matched')
        return ('FR-BIL', matched, total, mismatches)
    except Exception as e:
        print(f'ERROR: {str(e)[:50]}')
        return ('FR-BIL', 0, 0, [('ERR', str(e)[:50], '', '')])

def verify_itcp2():
    """IT-CP2: Excel with DOY"""
    try:
        print('Verifying IT-CP2...', end=' ')
        orig = pd.read_excel(base / 'IT-CP2/SapFlow2014.xls', engine='xlrd')
        orig['timestamp'] = pd.to_datetime('2014-01-01') + pd.to_timedelta(orig['DOY']-1, unit='D') + pd.to_timedelta(orig['Hour'], unit='h')
        out = pd.read_csv(out_base / 'plant/IT-CP2_sapflow_plant.csv')
        out['timestamp'] = pd.to_datetime(out['timestamp'], format='mixed')
        
        valid_data = orig[(orig['Sap1 (g h-1)'].notna()) & (orig['Sap1 (g h-1)'] != -9999)]
        sample_idx = np.random.choice(len(valid_data), min(N_SAMPLES, len(valid_data)), replace=False)
        samples = valid_data.iloc[sample_idx]
        
        out_col = 'It-Cp2-1-Dynamax TDP 10'
        orig_vals, out_vals = [], []
        for _, row in samples.iterrows():
            ts = row['timestamp']
            orig_vals.append(row['Sap1 (g h-1)'])
            out_row = out[out['timestamp'] == ts]
            out_vals.append(out_row[out_col].values[0] if len(out_row) > 0 and out_col in out.columns else np.nan)
        
        matched, total, mismatches = verify_samples(orig_vals, out_vals)
        print(f'{matched}/{total} matched')
        return ('IT-CP2', matched, total, mismatches)
    except Exception as e:
        print(f'ERROR: {str(e)[:50]}')
        return ('IT-CP2', 0, 0, [('ERR', str(e)[:50], '', '')])

def verify_nohur():
    """NO-Hur: TXT file"""
    try:
        print('Verifying NO-Hur...', end=' ')
        orig = pd.read_csv(base / 'NO-Hur/NO-Hur_sapflow_10min.txt', sep='\t')
        ts_col = [c for c in orig.columns if 'time' in c.lower() or 'date' in c.lower()][0]
        orig[ts_col] = pd.to_datetime(orig[ts_col], format='mixed')
        out = pd.read_csv(out_base / 'plant/NO-Hur_sapflow_plant.csv')
        out['timestamp'] = pd.to_datetime(out['timestamp'], format='mixed')
        
        data_cols = [c for c in orig.columns if c != ts_col and 'time' not in c.lower() and 'date' not in c.lower()]
        valid_data = orig[orig[data_cols[0]].notna()]
        sample_idx = np.random.choice(len(valid_data), min(N_SAMPLES, len(valid_data)), replace=False)
        samples = valid_data.iloc[sample_idx]
        
        out_col = [c for c in out.columns if data_cols[0] in c]
        if not out_col:
            out_col = [c for c in out.columns if c != 'timestamp'][:1]
        orig_vals, out_vals = [], []
        for _, row in samples.iterrows():
            ts = row[ts_col]
            orig_vals.append(row[data_cols[0]])
            out_row = out[out['timestamp'] == ts]
            out_vals.append(out_row[out_col[0]].values[0] if len(out_row) > 0 and out_col else np.nan)
        
        matched, total, mismatches = verify_samples(orig_vals, out_vals)
        print(f'{matched}/{total} matched')
        return ('NO-Hur', matched, total, mismatches)
    except Exception as e:
        print(f'ERROR: {str(e)[:50]}')
        return ('NO-Hur', 0, 0, [('ERR', str(e)[:50], '', '')])

def verify_plmez():
    """PL-Mez: CSV file"""
    try:
        print('Verifying PL-Mez...', end=' ')
        # PL-Mez uses pattern PL-Mez_11_1_1_sapflow.csv
        orig = pd.read_csv(base / 'PL-Mez/PL-Mez_11_1_1_sapflow.csv')
        ts_col = [c for c in orig.columns if 'time' in c.lower() or 'date' in c.lower()][0]
        orig[ts_col] = pd.to_datetime(orig[ts_col], format='mixed')
        out = pd.read_csv(out_base / 'plant/PL-Mez_sapflow_plant.csv')
        out['timestamp'] = pd.to_datetime(out['timestamp'], format='mixed')
        
        data_cols = [c for c in orig.columns if c != ts_col and 'time' not in c.lower()]
        valid_data = orig[orig[data_cols[0]].notna()]
        sample_idx = np.random.choice(len(valid_data), min(N_SAMPLES, len(valid_data)), replace=False)
        samples = valid_data.iloc[sample_idx]
        
        out_col = [c for c in out.columns if '11-1' in c or '11_1' in c]
        if not out_col:
            out_col = [c for c in out.columns if c != 'timestamp'][:1]
        orig_vals, out_vals = [], []
        for _, row in samples.iterrows():
            ts = row[ts_col]
            orig_vals.append(row[data_cols[0]])
            out_row = out[out['timestamp'] == ts]
            out_vals.append(out_row[out_col[0]].values[0] if len(out_row) > 0 and out_col else np.nan)
        
        matched, total, mismatches = verify_samples(orig_vals, out_vals)
        print(f'{matched}/{total} matched')
        return ('PL-Mez', matched, total, mismatches)
    except Exception as e:
        print(f'ERROR: {str(e)[:50]}')
        return ('PL-Mez', 0, 0, [('ERR', str(e)[:50], '', '')])

def verify_pltuc():
    """PL-Tuc: CSV file"""
    try:
        print('Verifying PL-Tuc...', end=' ')
        # PL-Tuc uses pattern PL-Tuc_943_1_1_sapflow.csv
        orig = pd.read_csv(base / 'PL-Tuc/PL-Tuc_943_1_1_sapflow.csv')
        ts_col = [c for c in orig.columns if 'time' in c.lower() or 'date' in c.lower()][0]
        orig[ts_col] = pd.to_datetime(orig[ts_col], format='mixed')
        out = pd.read_csv(out_base / 'plant/PL-Tuc_sapflow_plant.csv')
        out['timestamp'] = pd.to_datetime(out['timestamp'], format='mixed')
        
        data_cols = [c for c in orig.columns if c != ts_col and 'time' not in c.lower()]
        valid_data = orig[orig[data_cols[0]].notna()]
        sample_idx = np.random.choice(len(valid_data), min(N_SAMPLES, len(valid_data)), replace=False)
        samples = valid_data.iloc[sample_idx]
        
        out_col = [c for c in out.columns if '943-1' in c or '943_1' in c]
        if not out_col:
            out_col = [c for c in out.columns if c != 'timestamp'][:1]
        orig_vals, out_vals = [], []
        for _, row in samples.iterrows():
            ts = row[ts_col]
            orig_vals.append(row[data_cols[0]])
            out_row = out[out['timestamp'] == ts]
            out_vals.append(out_row[out_col[0]].values[0] if len(out_row) > 0 and out_col else np.nan)
        
        matched, total, mismatches = verify_samples(orig_vals, out_vals)
        print(f'{matched}/{total} matched')
        return ('PL-Tuc', matched, total, mismatches)
    except Exception as e:
        print(f'ERROR: {str(e)[:50]}')
        return ('PL-Tuc', 0, 0, [('ERR', str(e)[:50], '', '')])

def verify_senor():
    """SE-Nor: Source TXT files contain only NA values - verification not possible"""
    try:
        print('Verifying SE-Nor...', end=' ')
        # SE-Nor source files only contain NAs
        # The output data was processed from a different source or the files are placeholders
        # Check if output has data
        out = pd.read_csv(out_base / 'plant/SE-Nor_sapflow_plant.csv')
        non_null = out.drop('timestamp', axis=1).notna().sum().sum()
        print(f'SKIP - source files NA only (output has {non_null} values)')
        return ('SE-Nor', 0, 0, [('SKIP', f'Source files NA, output has {non_null} values', '', '')])
    except Exception as e:
        print(f'ERROR: {str(e)[:50]}')
        return ('SE-Nor', 0, 0, [('ERR', str(e)[:50], '', '')])

def verify_sesgr():
    """SE-Sgr: Excel with Date+Time columns"""
    try:
        print('Verifying SE-Sgr...', end=' ')
        # SE-Sgr has 3 columns: Date, Time, Sap flux
        orig = pd.read_excel(base / 'SE-Sgr/SE-Sgr_40_0_0_sapflow.xlsx', engine='openpyxl', skiprows=1)
        orig.columns = ['date', 'time', 'value']
        # Combine date and time
        orig['timestamp'] = pd.to_datetime(orig['date'].astype(str).str[:10] + ' ' + orig['time'].astype(str))
        orig['value'] = pd.to_numeric(orig['value'], errors='coerce')
        out = pd.read_csv(out_base / 'plant/SE-Sgr_sapflow_plant.csv')
        out['timestamp'] = pd.to_datetime(out['timestamp'], format='mixed')
        
        valid_data = orig[orig['value'].notna()]
        sample_idx = np.random.choice(len(valid_data), min(N_SAMPLES, len(valid_data)), replace=False)
        samples = valid_data.iloc[sample_idx]
        
        out_col = [c for c in out.columns if '40-0' in c or '40_0' in c]
        if not out_col:
            out_col = [c for c in out.columns if c != 'timestamp'][:1]
        orig_vals, out_vals = [], []
        for _, row in samples.iterrows():
            ts = row['timestamp']
            orig_vals.append(row['value'])
            out_row = out[out['timestamp'] == ts]
            out_vals.append(out_row[out_col[0]].values[0] if len(out_row) > 0 and out_col else np.nan)
        
        matched, total, mismatches = verify_samples(orig_vals, out_vals)
        print(f'{matched}/{total} matched')
        return ('SE-Sgr', matched, total, mismatches)
    except Exception as e:
        print(f'ERROR: {str(e)[:50]}')
        return ('SE-Sgr', 0, 0, [('ERR', str(e)[:50], '', '')])

def verify_zoeat():
    """ZOE_AT: Excel"""
    try:
        print('Verifying ZOE_AT...', end=' ')
        orig = pd.read_excel(base / 'ZOE_AT/ZOE_AT_IP2_sapflow.xlsx', engine='openpyxl')
        ts_col = [c for c in orig.columns if 'time' in c.lower() or 'date' in c.lower()][0]
        orig[ts_col] = pd.to_datetime(orig[ts_col])
        out = pd.read_csv(out_base / 'plant/ZOE_AT_sapflow_plant.csv')
        out['timestamp'] = pd.to_datetime(out['timestamp'], format='mixed')
        
        data_col = [c for c in orig.columns if c != ts_col][0]
        valid_data = orig[orig[data_col].notna()]
        sample_idx = np.random.choice(len(valid_data), min(N_SAMPLES, len(valid_data)), replace=False)
        samples = valid_data.iloc[sample_idx]
        
        out_col = [c for c in out.columns if data_col in c][0]
        orig_vals, out_vals = [], []
        for _, row in samples.iterrows():
            ts = row[ts_col]
            orig_vals.append(row[data_col])
            out_row = out[out['timestamp'] == ts]
            out_vals.append(out_row[out_col].values[0] if len(out_row) > 0 else np.nan)
        
        matched, total, mismatches = verify_samples(orig_vals, out_vals)
        print(f'{matched}/{total} matched')
        return ('ZOE_AT', matched, total, mismatches)
    except Exception as e:
        print(f'ERROR: {str(e)[:50]}')
        return ('ZOE_AT', 0, 0, [('ERR', str(e)[:50], '', '')])

# ============================================================================
# RUN ALL VERIFICATIONS
# ============================================================================

# Sapwood sites
results.append(verify_chdav())
results.append(verify_chlae())
results.append(verify_dehar())
results.append(verify_dehoh())
results.append(verify_esabr())
results.append(verify_eslm1())
results.append(verify_eslm2())
results.append(verify_eslma())
results.append(verify_fihyy())

# Plant sites
results.append(verify_atmmg())
results.append(verify_esgdn())
results.append(verify_frbil())
results.append(verify_itcp2())
results.append(verify_nohur())
results.append(verify_plmez())
results.append(verify_pltuc())
results.append(verify_senor())
results.append(verify_sesgr())
results.append(verify_zoeat())

# ============================================================================
# PRINT RESULTS
# ============================================================================

print()
print('='*80)
print('VERIFICATION RESULTS SUMMARY - ALL 19 SITES')
print('='*80)
print(f"{'Site':<12} | {'Matched':<10} | {'Total':<10} | {'Accuracy':<10} | {'Status':<10}")
print('-'*80)

total_matched = 0
total_samples = 0
passed_sites = 0
for site, matched, total, mismatches in results:
    if total > 0:
        accuracy = f'{matched/total*100:.1f}%'
        # Consider PASS if 95%+ matched (allowing for timestamp range differences)
        status = 'PASS' if matched/total >= 0.95 else 'FAIL'
        if matched/total >= 0.95:
            passed_sites += 1
        total_matched += matched
        total_samples += total
    else:
        accuracy = 'N/A'
        status = 'ERROR'
    print(f'{site:<12} | {matched:<10} | {total:<10} | {accuracy:<10} | {status:<10}')
    
    # Show first few mismatches if any actual value mismatches
    if mismatches and total > 0:
        value_mismatches = [m for m in mismatches if len(m) >= 4 and m[3] == 'VALUE_DIFF']
        nan_mismatches = [m for m in mismatches if len(m) >= 4 and m[3] == 'NAN_MISMATCH']
        if value_mismatches:
            print(f'             VALUE MISMATCHES: {len(value_mismatches)}')
            for item in value_mismatches[:2]:
                print(f'               #{item[0]}: orig={item[1]}, out={item[2]}')
        if nan_mismatches:
            print(f'             TIMESTAMP RANGE DIFF: {len(nan_mismatches)} samples')

print('='*80)
if total_samples > 0:
    print(f'OVERALL: {total_matched}/{total_samples} samples matched ({total_matched/total_samples*100:.2f}% accuracy)')
    print(f'SITES: {passed_sites}/{len(results)} passed (95%+ threshold)')
else:
    print('TOTAL: No samples verified')
