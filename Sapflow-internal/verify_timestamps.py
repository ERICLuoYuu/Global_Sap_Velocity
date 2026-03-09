import pandas as pd
import pyreadr
import numpy as np

np.random.seed(42)  # For reproducibility
N_SAMPLES = 1000  # Number of random timestamps to verify per site

print('='*80)
print(f'COMPREHENSIVE TIMESTAMP VERIFICATION ({N_SAMPLES} random samples per site)')
print('='*80)

results = []
base = 'Sapflow-internal'
out_base = 'Sapflow_SAPFLUXNET_format'

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
            # One is NaN, one is not
            mismatches.append((i, o, out, 'NAN_MISMATCH'))
    return matched, len(orig_vals), mismatches

# 1. CH-Dav (Long format RDS)
try:
    print('\nVerifying CH-Dav...', end=' ')
    rds = pyreadr.read_r(f'{base}/CH-Dav/CH-Dav_2010_2023_DT_15Trees_HH.rds')
    orig = list(rds.values())[0]
    orig['Timestamp'] = pd.to_datetime(orig['Timestamp'])
    out = pd.read_csv(f'{out_base}/sapwood/CH-Dav_halfhourly_sapflow_sapwood.csv')
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
    results.append(('CH-Dav', matched, total, mismatches))
    print(f'{matched}/{total} matched')
except Exception as e:
    results.append(('CH-Dav', 0, 0, [('ERR', str(e)[:50], '')]))
    print(f'ERROR: {str(e)[:50]}')

# 2. CH-Lae (Long format RDS)
try:
    print('Verifying CH-Lae...', end=' ')
    rds = pyreadr.read_r(f'{base}/CH-Lae/CH-Lae_2012_2022_SFD_HH.rds')
    orig = list(rds.values())[0]
    orig['Timestamp'] = pd.to_datetime(orig['Timestamp'])
    out = pd.read_csv(f'{out_base}/sapwood/CH-Lae_halfhourly_sapflow_sapwood.csv')
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
    results.append(('CH-Lae', matched, total, mismatches))
    print(f'{matched}/{total} matched')
except Exception as e:
    results.append(('CH-Lae', 0, 0, [('ERR', str(e)[:50], '')]))
    print(f'ERROR: {str(e)[:50]}')

# 3. DE-Har (Multi-file CSV)
try:
    print('Verifying DE-Har...', end=' ')
    orig = pd.read_csv(f'{base}/DE-Har/DE-Har_H10545_1_1_sapflow.csv')
    orig['TIMESTAMP'] = pd.to_datetime(orig['TIMESTAMP'], format='mixed')
    out = pd.read_csv(f'{out_base}/sapwood/DE-Har_sapflow_sapwood.csv')
    out['timestamp'] = pd.to_datetime(out['timestamp'], format='mixed')
    
    valid_data = orig[orig['Js_outer'].notna()]
    sample_idx = np.random.choice(len(valid_data), min(N_SAMPLES, len(valid_data)), replace=False)
    samples = valid_data.iloc[sample_idx]
    
    out_col = [c for c in out.columns if 'H10545' in c and 'outer' in c][0]
    orig_vals, out_vals = [], []
    for _, row in samples.iterrows():
        ts = row['TIMESTAMP']
        orig_vals.append(row['Js_outer'])
        out_row = out[out['timestamp'] == ts]
        out_vals.append(out_row[out_col].values[0] if len(out_row) > 0 else np.nan)
    
    matched, total, mismatches = verify_samples(orig_vals, out_vals)
    results.append(('DE-Har', matched, total, mismatches))
    print(f'{matched}/{total} matched')
except Exception as e:
    results.append(('DE-Har', 0, 0, [('ERR', str(e)[:50], '')]))
    print(f'ERROR: {str(e)[:50]}')

# 4. ZOE_AT (Excel)
try:
    print('Verifying ZOE_AT...', end=' ')
    orig = pd.read_excel(f'{base}/ZOE_AT/ZOE_AT_IP2_sapflow.xlsx', engine='openpyxl')
    ts_col = [c for c in orig.columns if 'time' in c.lower() or 'date' in c.lower()][0]
    orig[ts_col] = pd.to_datetime(orig[ts_col])
    out = pd.read_csv(f'{out_base}/plant/ZOE_AT_sapflow_plant.csv')
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
    results.append(('ZOE_AT', matched, total, mismatches))
    print(f'{matched}/{total} matched')
except Exception as e:
    results.append(('ZOE_AT', 0, 0, [('ERR', str(e)[:50], '')]))
    print(f'ERROR: {str(e)[:50]}')

# 5. IT-CP2 (Excel with DOY)
try:
    print('Verifying IT-CP2...', end=' ')
    orig = pd.read_excel(f'{base}/IT-CP2/SapFlow2014.xls', engine='xlrd')
    orig['timestamp'] = pd.to_datetime('2014-01-01') + pd.to_timedelta(orig['DOY']-1, unit='D') + pd.to_timedelta(orig['Hour'], unit='h')
    out = pd.read_csv(f'{out_base}/plant/IT-CP2_sapflow_plant.csv')
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
    results.append(('IT-CP2', matched, total, mismatches))
    print(f'{matched}/{total} matched')
except Exception as e:
    results.append(('IT-CP2', 0, 0, [('ERR', str(e)[:50], '')]))
    print(f'ERROR: {str(e)[:50]}')

# 6. FR-BIL (Excel)
try:
    print('Verifying FR-BIL...', end=' ')
    orig = pd.read_excel(f'{base}/FR-BIL/FR-BIL saplow 7 trees 2020_2022.xlsx', engine='openpyxl')
    orig['DateTime'] = pd.to_datetime(orig['DateTime'])
    out = pd.read_csv(f'{out_base}/plant/FR-BIL_sapflow_plant.csv')
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
    results.append(('FR-BIL', matched, total, mismatches))
    print(f'{matched}/{total} matched')
except Exception as e:
    results.append(('FR-BIL', 0, 0, [('ERR', str(e)[:50], '')]))
    print(f'ERROR: {str(e)[:50]}')

# 7. ES-Abr (Excel 2-column)
try:
    print('Verifying ES-Abr...', end=' ')
    orig = pd.read_excel(f'{base}/ES-Abr/ES-Abr_1_SAP1_1_sapflow.xlsx', engine='openpyxl', header=None)
    orig.columns = ['timestamp', 'value']
    orig['timestamp'] = pd.to_datetime(orig['timestamp'])
    out = pd.read_csv(f'{out_base}/sapwood/ES-Abr_sapflow_sapwood.csv')
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
    results.append(('ES-Abr', matched, total, mismatches))
    print(f'{matched}/{total} matched')
except Exception as e:
    results.append(('ES-Abr', 0, 0, [('ERR', str(e)[:50], '')]))
    print(f'ERROR: {str(e)[:50]}')

print()
print('='*80)
print('VERIFICATION RESULTS SUMMARY')
print('='*80)
print(f"{'Site':<12} | {'Matched':<10} | {'Total':<10} | {'Accuracy':<10} | {'Status':<10}")
print('-'*80)

total_matched = 0
total_samples = 0
for site, matched, total, mismatches in results:
    if total > 0:
        accuracy = f'{matched/total*100:.1f}%'
        status = 'PASS' if matched == total else 'FAIL'
        total_matched += matched
        total_samples += total
    else:
        accuracy = 'N/A'
        status = 'ERROR'
    print(f'{site:<12} | {matched:<10} | {total:<10} | {accuracy:<10} | {status:<10}')
    
    # Show first few mismatches if any
    if mismatches and total > 0:
        for item in mismatches[:5]:
            if len(item) == 4:
                idx, orig_v, out_v, reason = item
                print(f'             Mismatch #{idx}: orig={orig_v}, out={out_v}, reason={reason}')

print('='*80)
if total_samples > 0:
    print(f'TOTAL: {total_matched}/{total_samples} samples matched ({total_matched/total_samples*100:.2f}% accuracy)')
else:
    print('TOTAL: No samples verified')
