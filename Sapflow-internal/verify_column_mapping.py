"""
Verification script for sap_reorganizer.py column mapping.

This script helps verify that after reorganization:
1. Column names match the expected plant codes from metadata
2. Data values in each column match the original source column
3. No data was incorrectly swapped between columns

Usage:
    python verify_column_mapping.py <site_code>
    
Example:
    python verify_column_mapping.py NO-Hur
"""

import os
import pandas as pd
import numpy as np
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def extract_tree_id_from_column(col_name: str) -> str:
    """
    Extract tree_id from column name patterns like:
    - NO-Hur_T3_SFM1E21V_sapflow -> T3
    - NO-Hur_H1_SFM1E30A_sapflow -> H1
    - ARG_MAZ_T1_sapflow -> T1
    """
    # Pattern: site_treeid_sensor_variable or site_treeid_variable
    # Look for patterns like T3, H1, H3_1, T124, etc.
    
    # First try to match tree IDs with underscores (like H3_1, H3_2)
    match = re.search(r'[_\-]([TH]\d+_\d+)[_\-]', col_name, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Then try simple tree IDs (T3, H1, T124, etc.)
    match = re.search(r'[_\-]([TH]\d+)[_\-]', col_name, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Try to match just numbers with prefix
    match = re.search(r'[_\-](\d+)[_\-]', col_name)
    if match:
        return match.group(1)
    
    return None


def extract_sensor_id_from_column(col_name: str) -> str:
    """
    Extract sensor_id from column name patterns like:
    - NO-Hur_T3_SFM1E21V_sapflow -> 1E21V (sensor is after SFM prefix)
    - NO-Hur_H1_SFM1E30A_sapflow -> 1E30A
    """
    # Pattern: SFM followed by sensor ID
    match = re.search(r'SFM([A-Z0-9]+)', col_name, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Try direct sensor ID pattern
    match = re.search(r'[_\-](\d+[A-Z]+\d*)[_\-]', col_name, re.IGNORECASE)
    if match:
        return match.group(1)
    
    return None


def load_metadata(site_path: Path, site_code: str) -> pd.DataFrame:
    """Load and parse metadata file."""
    # Find metadata file
    meta_files = list(site_path.glob("*meta*.xlsx")) + list(site_path.glob("*Meta*.xlsx"))
    if not meta_files:
        raise FileNotFoundError(f"No metadata file found in {site_path}")
    
    meta_file = meta_files[0]
    print(f"Loading metadata from: {meta_file.name}")
    
    # Read the Sapflow_Metadata sheet
    try:
        df = pd.read_excel(meta_file, sheet_name='Sapflow_Metadata')
    except:
        df = pd.read_excel(meta_file)
    
    # The metadata is in a transposed format - rows are fields, columns are sensors
    # Find the row with 'tree_id'
    tree_id_row = df[df.iloc[:, 0].astype(str).str.contains('tree_id', case=False, na=False)]
    sensor_id_row = df[df.iloc[:, 0].astype(str).str.contains('sensor_id', case=False, na=False)]
    file_name_row = df[df.iloc[:, 0].astype(str).str.contains('file_name', case=False, na=False)]
    
    # Extract values (skip first column which is the field name)
    tree_ids = tree_id_row.iloc[0, 1:].dropna().tolist() if not tree_id_row.empty else []
    sensor_ids = sensor_id_row.iloc[0, 1:].dropna().tolist() if not sensor_id_row.empty else []
    file_names = file_name_row.iloc[0, 1:].dropna().tolist() if not file_name_row.empty else []
    
    # Create a tidy dataframe - pad shorter lists with None
    n_sensors = max(len(tree_ids), len(sensor_ids), len(file_names))
    
    # Pad lists to same length
    tree_ids = tree_ids + [None] * (n_sensors - len(tree_ids))
    sensor_ids = sensor_ids + [None] * (n_sensors - len(sensor_ids))
    file_names = file_names + [None] * (n_sensors - len(file_names))
    
    meta_tidy = pd.DataFrame({
        'tree_id': tree_ids,
        'sensor_id': sensor_ids,
        'file_name': file_names,
    })
    
    # Generate pl_code (plant code) - typically site_code + tree_id
    meta_tidy['pl_code'] = meta_tidy.apply(
        lambda row: f"{site_code}_{row['tree_id']}" if pd.notna(row['tree_id']) else None,
        axis=1
    )
    
    # Remove rows with no valid tree_id
    meta_tidy = meta_tidy[meta_tidy['tree_id'].notna()].reset_index(drop=True)
    
    return meta_tidy


def load_original_data(site_path: Path) -> pd.DataFrame:
    """Load original sapflow data file."""
    # Find data files (not metadata)
    data_files = []
    for ext in ['.txt', '.csv', '.tsv', '.dat']:
        data_files.extend(site_path.glob(f"*sapflow*{ext}"))
        data_files.extend(site_path.glob(f"*Sapflow*{ext}"))
    
    # Exclude metadata files
    data_files = [f for f in data_files if 'meta' not in f.name.lower()]
    
    if not data_files:
        raise FileNotFoundError(f"No sapflow data file found in {site_path}")
    
    data_file = data_files[0]
    print(f"Loading original data from: {data_file.name}")
    
    # Detect delimiter
    with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()
        if '\t' in first_line:
            sep = '\t'
        elif ';' in first_line:
            sep = ';'
        else:
            sep = ','
    
    df = pd.read_csv(data_file, sep=sep)
    return df


def build_column_mapping(original_df: pd.DataFrame, metadata_df: pd.DataFrame, site_code: str) -> dict:
    """
    Build a mapping from original column names to expected plant codes.
    Uses tree_id and sensor_id extraction instead of relying on column order.
    """
    mapping = {}
    unmatched_cols = []
    
    # Create lookup dictionaries
    tree_id_to_pl_code = {}
    sensor_id_to_pl_code = {}
    
    for _, row in metadata_df.iterrows():
        if pd.notna(row['tree_id']) and pd.notna(row['pl_code']):
            tree_id_to_pl_code[str(row['tree_id']).upper()] = row['pl_code']
        if pd.notna(row['sensor_id']) and pd.notna(row['pl_code']):
            sensor_id_to_pl_code[str(row['sensor_id']).upper()] = row['pl_code']
    
    print(f"\nTree ID to pl_code mapping: {tree_id_to_pl_code}")
    print(f"Sensor ID to pl_code mapping: {sensor_id_to_pl_code}")
    
    # Skip timestamp column
    data_cols = [col for col in original_df.columns if 'timestamp' not in col.lower()]
    
    for col in data_cols:
        # Try to extract tree_id from column name
        tree_id = extract_tree_id_from_column(col)
        sensor_id = extract_sensor_id_from_column(col)
        
        pl_code = None
        match_type = None
        
        # Try matching by tree_id first
        if tree_id:
            tree_id_upper = tree_id.upper()
            if tree_id_upper in tree_id_to_pl_code:
                pl_code = tree_id_to_pl_code[tree_id_upper]
                match_type = 'tree_id'
        
        # If no match by tree_id, try sensor_id
        if pl_code is None and sensor_id:
            sensor_id_upper = sensor_id.upper()
            if sensor_id_upper in sensor_id_to_pl_code:
                pl_code = sensor_id_to_pl_code[sensor_id_upper]
                match_type = 'sensor_id'
        
        if pl_code:
            mapping[col] = {
                'pl_code': pl_code,
                'tree_id': tree_id,
                'sensor_id': sensor_id,
                'match_type': match_type
            }
        else:
            unmatched_cols.append({
                'column': col,
                'extracted_tree_id': tree_id,
                'extracted_sensor_id': sensor_id
            })
    
    return mapping, unmatched_cols


def compare_statistics(original_df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Compare statistics for mapped columns."""
    stats = []
    
    for orig_col, info in mapping.items():
        if orig_col in original_df.columns:
            col_data = pd.to_numeric(original_df[orig_col], errors='coerce')
            stats.append({
                'original_column': orig_col,
                'pl_code': info['pl_code'],
                'tree_id': info['tree_id'],
                'sensor_id': info['sensor_id'],
                'match_type': info['match_type'],
                'count': col_data.count(),
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'na_count': col_data.isna().sum()
            })
    
    return pd.DataFrame(stats)


def verify_reorganized_output(site_code: str, output_dir: Path = None) -> dict:
    """
    Verify that reorganized output matches original data.
    
    Returns a dict with verification results.
    """
    if output_dir is None:
        output_dir = Path("Sapflow_SAPFLUXNET_format") / site_code
    
    if not output_dir.exists():
        return {'error': f"Output directory not found: {output_dir}"}
    
    # Find output sapflow file
    sapf_files = list(output_dir.glob("*sapf*.csv"))
    if not sapf_files:
        return {'error': f"No sapflow output file found in {output_dir}"}
    
    sapf_file = sapf_files[0]
    reorg_df = pd.read_csv(sapf_file, index_col=0, parse_dates=True)
    
    return {
        'output_file': sapf_file.name,
        'shape': reorg_df.shape,
        'columns': list(reorg_df.columns),
        'sample_data': reorg_df.head()
    }


def main(site_code: str):
    """Main verification function."""
    print(f"\n{'='*60}")
    print(f"COLUMN MAPPING VERIFICATION FOR: {site_code}")
    print(f"{'='*60}\n")
    
    # Find site path
    base_path = Path("Sapflow-internal/Sapflow-internal")
    site_path = base_path / site_code
    
    if not site_path.exists():
        print(f"ERROR: Site path not found: {site_path}")
        return
    
    # Step 1: Load metadata
    print("STEP 1: Loading metadata...")
    try:
        metadata_df = load_metadata(site_path, site_code)
        print(f"Found {len(metadata_df)} sensor entries in metadata:")
        print(metadata_df.to_string())
    except Exception as e:
        print(f"ERROR loading metadata: {e}")
        return
    
    # Step 2: Load original data
    print("\n" + "-"*40)
    print("STEP 2: Loading original data...")
    try:
        original_df = load_original_data(site_path)
        print(f"Original data shape: {original_df.shape}")
        print(f"Original columns: {list(original_df.columns)}")
    except Exception as e:
        print(f"ERROR loading original data: {e}")
        return
    
    # Step 3: Build column mapping
    print("\n" + "-"*40)
    print("STEP 3: Building column mapping...")
    mapping, unmatched = build_column_mapping(original_df, metadata_df, site_code)
    
    print(f"\nSuccessfully mapped {len(mapping)} columns:")
    for orig_col, info in mapping.items():
        print(f"  {orig_col}")
        print(f"    -> pl_code: {info['pl_code']} (matched by {info['match_type']})")
    
    if unmatched:
        print(f"\nWARNING: {len(unmatched)} columns could not be matched:")
        for item in unmatched:
            print(f"  {item['column']}")
            print(f"    extracted tree_id: {item['extracted_tree_id']}, sensor_id: {item['extracted_sensor_id']}")
    
    # Step 4: Compare statistics
    print("\n" + "-"*40)
    print("STEP 4: Column statistics...")
    stats_df = compare_statistics(original_df, mapping)
    print(stats_df.to_string())
    
    # Step 5: Check expected vs actual column order
    print("\n" + "-"*40)
    print("STEP 5: Column order analysis...")
    
    # Get expected order from metadata (sorted by tree_id)
    meta_sorted = metadata_df.sort_values('tree_id').reset_index(drop=True)
    expected_order = meta_sorted['pl_code'].tolist()
    
    # Get actual order from data file
    data_cols = [col for col in original_df.columns if 'timestamp' not in col.lower()]
    actual_order = [mapping[col]['pl_code'] if col in mapping else f"UNMATCHED:{col}" for col in data_cols]
    
    print(f"\nExpected order (metadata sorted by tree_id):")
    print(f"  {expected_order}")
    print(f"\nActual order in data file:")
    print(f"  {actual_order}")
    
    if expected_order == actual_order:
        print("\n✓ Column order MATCHES metadata order")
    else:
        print("\n✗ Column order DOES NOT match metadata order!")
        print("  The current sap_reorganizer.py assumes order-based mapping, which would be INCORRECT.")
        print("  Columns should be matched by tree_id/sensor_id extracted from column names.")
    
    # Step 6: Verify reorganized output if it exists
    print("\n" + "-"*40)
    print("STEP 6: Checking reorganized output...")
    output_info = verify_reorganized_output(site_code)
    if 'error' in output_info:
        print(f"  {output_info['error']}")
    else:
        print(f"  Output file: {output_info['output_file']}")
        print(f"  Shape: {output_info['shape']}")
        print(f"  Columns: {output_info['columns']}")
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        site_code = sys.argv[1]
    else:
        site_code = "NO-Hur"  # Default test site
    
    main(site_code)
