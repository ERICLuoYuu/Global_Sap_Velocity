import pandas as pd
from pathlib import Path

# Check one of the SE-Nor data files
data_file = "Sapflow-internal/SE-Nor/SE-Nor_Granier7_ST1_AF2_Psy_Jt_7_1_sapflow.txt"

print("=" * 80)
print("ENHANCED DIAGNOSTIC: SE-Nor File Analysis")
print("=" * 80)

# 1. Show raw bytes of first few lines
print("\n1. RAW BYTES (first 3 lines):")
print("-" * 80)
try:
    with open(data_file, 'rb') as f:
        for i in range(3):
            line_bytes = f.readline()
            print(f"Line {i}: {line_bytes}")
            print(f"  Decoded: {repr(line_bytes.decode('utf-8', errors='replace'))}")
            print(f"  Hex: {line_bytes.hex()}")
            print()
except Exception as e:
    print(f"Error: {e}")

# 2. Read first 100 lines with auto-detect
print("\n2. PANDAS AUTO-DETECT (first 100 rows):")
print("-" * 80)
try:
    df = pd.read_csv(data_file, sep=None, engine='python', nrows=100)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data types:\n{df.dtypes}")
    print(f"\nFirst 10 rows:\n{df.head(10)}")
    print(f"\nLast 10 rows:\n{df.tail(10)}")
    
    # Check unique values
    for col in df.columns:
        unique_vals = df[col].unique()
        print(f"\nColumn '{col}' unique values: {unique_vals[:10]}")
        print(f"  Total unique: {len(unique_vals)}")
        print(f"  Value counts:\n{df[col].value_counts().head()}")
except Exception as e:
    print(f"Error: {e}")

# 3. Try reading entire file
print("\n3. FULL FILE ANALYSIS:")
print("-" * 80)
try:
    df_full = pd.read_csv(data_file, sep=None, engine='python')
    print(f"Full file shape: {df_full.shape}")
    print(f"Columns: {df_full.columns.tolist()}")
    
    # Check if data changes after certain rows
    print(f"\nRows 0-5:")
    print(df_full.head())
    print(f"\nRows 1000-1005:")
    print(df_full.iloc[1000:1005])
    print(f"\nRows 10000-10005:")
    print(df_full.iloc[10000:10005])
    print(f"\nLast 5 rows:")
    print(df_full.tail())
    
    # Check for any numeric values
    for col in df_full.columns:
        numeric_col = pd.to_numeric(df_full[col], errors='coerce')
        n_numeric = numeric_col.notna().sum()
        if n_numeric > 0:
            print(f"\n✓ Column '{col}' has {n_numeric} numeric values!")
            print(f"  Sample numeric values: {numeric_col.dropna().head(10).tolist()}")
        else:
            print(f"\n✗ Column '{col}' has NO numeric values (all 'A' or NaN)")
            
except Exception as e:
    print(f"Error: {e}")

# 4. Check file encoding and size
print("\n4. FILE PROPERTIES:")
print("-" * 80)
try:
    import os
    file_size = os.path.getsize(data_file)
    print(f"File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
    
    # Count total lines
    with open(data_file, 'r') as f:
        line_count = sum(1 for _ in f)
    print(f"Total lines: {line_count:,}")
    
    # Check for BOM or special encoding markers
    with open(data_file, 'rb') as f:
        first_bytes = f.read(10)
        print(f"First 10 bytes (hex): {first_bytes.hex()}")
        if first_bytes.startswith(b'\xef\xbb\xbf'):
            print("  ⚠️  UTF-8 BOM detected")
        
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
print("ENHANCED DIAGNOSIS COMPLETE")
print("=" * 80)