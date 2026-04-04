import pandas as pd
from pathlib import Path as _Path

def _is_parquet(filepath):
    return _Path(filepath).suffix.lower() in {'.parquet', '.pq'}
import numpy as np
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import os
import sqlite3
import multiprocessing as mp # Keep for cpu_count, but Pool replaced
# Import ThreadPoolExecutor
import concurrent.futures
from tqdm import tqdm
import dask.dataframe as dd
# Optional: Import dask distributed for better diagnostics/control
# from dask.distributed import Client, LocalCluster
import psutil
import tempfile
import shutil
import time
from scipy.ndimage import zoom # Added for plot downsampling
import traceback # For detailed error printing

# =============================================================================
# Helper Function for Block Size Calculation (NEW)
# =============================================================================
def get_valid_block_size(dimension, preferred_block_size=256, block_multiple=16):
    """
    Calculates a valid block size for rasterio write operations.

    Ensures the block size is a multiple of `block_multiple` (default 16)
    and does not exceed the given dimension. Uses `preferred_block_size`
    if the dimension allows.

    Args:
        dimension (int): The height or width of the raster/tile.
        preferred_block_size (int): The desired block size if possible (e.g., 256).
        block_multiple (int): The required multiple for the block size (e.g., 16).

    Returns:
        int: A valid block size.
    """
    if dimension <= 0:
        # Return the minimum valid multiple if dimension is zero or negative
        return block_multiple
    # Use preferred block size if dimension is large enough
    if dimension >= preferred_block_size:
        # Ensure preferred size is also a multiple (usually is, but safe check)
        return (preferred_block_size // block_multiple) * block_multiple
    else:
        # Find the largest multiple of block_multiple that is <= dimension
        calculated_size = (dimension // block_multiple) * block_multiple
        # Ensure we return at least the minimum multiple (e.g., 16)
        return max(block_multiple, calculated_size)

# =============================================================================
# Data Inspection Function
# =============================================================================


def inspect_large_file(data_filepath, chunk_size=100000, sample_size=10):
    """
    Memory-efficient inspection of large data files (Parquet or CSV).
    """
    start_time = time.time()
    file_size_bytes = os.path.getsize(data_filepath)
    file_size_gb = file_size_bytes / (1024**3)
    print(f"File: {os.path.basename(data_filepath)}")
    print(f"Size: {file_size_gb:.2f} GB ({file_size_bytes:,} bytes)")
    print("\nLoading file schema and sample data...")
    try:
        if _is_parquet(data_filepath):
            first_chunk = pd.read_parquet(data_filepath, engine="pyarrow").head(5)
        else:
            first_chunk = pd.read_csv(data_filepath, nrows=5)
    except Exception as e:
        print(f"Error reading initial rows: {e}")
        return None
    print("\n=== COLUMNS ===")
    for i, col in enumerate(first_chunk.columns):
        print(f"{i+1}. {col} - {first_chunk[col].dtype}")

    total_rows = 0; lat_values = set(); lon_values = set()
    min_lat, max_lat, min_lon, max_lon = float('inf'), float('-inf'), float('inf'), float('-inf')
    min_sw_in, max_sw_in = float('inf'), float('-inf')
    min_sap_vel, max_sap_vel = float('inf'), float('-inf')
    sum_sap_vel, count_sap_vel, sum_squares_sap_vel = 0, 0, 0
    sap_vel_values = []; null_sap_vel_count = 0
    sap_vel_bins = {'Negative': 0, '0 to 0.5': 0, '0.5 to 1': 0, '1 to 2': 0, '2 to 5': 0, '5 to 10': 0, '10+': 0}
    beginning_rows = first_chunk.copy(); middle_rows = None; end_rows = None

    try:
        row_size_bytes = first_chunk.memory_usage(index=True, deep=True).sum() / len(first_chunk) if len(first_chunk) > 0 else 100
        total_chunks = int(np.ceil(file_size_bytes / (chunk_size * row_size_bytes))) if row_size_bytes > 0 else 1
        print(f"Estimated row size: {row_size_bytes:.2f} bytes, Estimated chunks: {total_chunks}")
    except Exception: total_chunks = 100

    print(f"\nAnalyzing data in chunks of {chunk_size:,} rows...")
    processed_chunks = 0
    try:
        # Use low_memory=False to potentially avoid dtype warnings, but uses more memory
        if _is_parquet(data_filepath):
            # Parquet: read entire file (much smaller on disk), then chunk in memory
            _full_df = pd.read_parquet(data_filepath, engine="pyarrow")
            chunk_iterator = [_full_df[i:i+chunk_size] for i in range(0, len(_full_df), chunk_size)]
        else:
            chunk_iterator = pd.read_csv(data_filepath, chunksize=chunk_size, low_memory=False)
        # Wrap with tqdm only if total_chunks is meaningful
        if total_chunks > 1:
             chunk_iterator = tqdm(chunk_iterator, total=total_chunks, desc="Inspecting Chunks")

        for chunk_num, chunk in enumerate(chunk_iterator):
            processed_chunks += 1
            # Update row count
            chunk_rows = len(chunk)
            total_rows += chunk_rows

            # Update min/max values for lat/lon
            if 'latitude' in chunk.columns and 'longitude' in chunk.columns:
                # Ensure numeric types before min/max
                lat_col = pd.to_numeric(chunk['latitude'], errors='coerce')
                lon_col = pd.to_numeric(chunk['longitude'], errors='coerce')

                # Sample lat/lon values to avoid memory issues with huge sets
                if chunk_rows > 10000:
                    sample_indices = np.random.choice(chunk_rows, min(1000, chunk_rows), replace=False)
                    sample_chunk = chunk.iloc[sample_indices]
                    lat_values.update(pd.to_numeric(sample_chunk['latitude'], errors='coerce').dropna().unique())
                    lon_values.update(pd.to_numeric(sample_chunk['longitude'], errors='coerce').dropna().unique())
                else:
                    lat_values.update(lat_col.dropna().unique())
                    lon_values.update(lon_col.dropna().unique())

                # Ignore NaNs introduced by coercion when finding min/max
                min_lat = min(min_lat, lat_col.min(skipna=True))
                max_lat = max(max_lat, lat_col.max(skipna=True))
                min_lon = min(min_lon, lon_col.min(skipna=True))
                max_lon = max(max_lon, lon_col.max(skipna=True))

            if 'sw_in' in chunk.columns:
                sw_col = pd.to_numeric(chunk['sw_in'], errors='coerce')
                min_sw_in = min(min_sw_in, sw_col.min(skipna=True))
                max_sw_in = max(max_sw_in, sw_col.max(skipna=True))

            # Process sap_velocity_cnn_lstm statistics
            if 'sap_velocity_cnn_lstm' in chunk.columns:
                sap_col = pd.to_numeric(chunk['sap_velocity_cnn_lstm'], errors='coerce')
                # Count null values (original NaNs + coercion errors)
                null_sap_vel_count += sap_col.isnull().sum()

                # Get non-null values for statistics
                valid_sap_vel = sap_col.dropna()

                if len(valid_sap_vel) > 0:
                    min_sap_vel = min(min_sap_vel, valid_sap_vel.min())
                    max_sap_vel = max(max_sap_vel, valid_sap_vel.max())
                    sum_sap_vel += valid_sap_vel.sum()
                    count_sap_vel += len(valid_sap_vel)
                    sum_squares_sap_vel += (valid_sap_vel ** 2).sum()

                    # Sample values for distribution analysis (keep a few hundred values)
                    if len(sap_vel_values) < 500:
                        sample_size_vel = min(100, len(valid_sap_vel))
                        sap_vel_values.extend(np.random.choice(valid_sap_vel.values, sample_size_vel, replace=False).tolist())

                    # Update distribution bins
                    sap_vel_bins['Negative'] += (valid_sap_vel < 0).sum()
                    sap_vel_bins['0 to 0.5'] += ((valid_sap_vel >= 0) & (valid_sap_vel < 0.5)).sum()
                    sap_vel_bins['0.5 to 1'] += ((valid_sap_vel >= 0.5) & (valid_sap_vel < 1)).sum()
                    sap_vel_bins['1 to 2'] += ((valid_sap_vel >= 1) & (valid_sap_vel < 2)).sum()
                    sap_vel_bins['2 to 5'] += ((valid_sap_vel >= 2) & (valid_sap_vel < 5)).sum()
                    sap_vel_bins['5 to 10'] += ((valid_sap_vel >= 5) & (valid_sap_vel < 10)).sum()
                    sap_vel_bins['10+'] += (valid_sap_vel >= 10).sum()

            # Sample from middle of file (approximately)
            if total_chunks > 1 and chunk_num == total_chunks // 2 and middle_rows is None:
                middle_rows = chunk.head(sample_size).copy()

            # Keep last chunk to sample end rows (handle potential single chunk)
            # Ensure end_rows is captured even if total_chunks estimation is off
            if processed_chunks >= total_chunks or chunk_num == (total_chunks if total_chunks > 0 else 1) -1 : # Heuristic for last chunk
                 end_rows = chunk.tail(sample_size).copy()

    except Exception as e:
        print(f"\nError during chunk processing: {e}\nProcessed {processed_chunks} chunks.")
        traceback.print_exc() # Print detailed traceback for debugging

    elapsed_time = time.time() - start_time
    # Calculate derived statistics for sap_velocity_cnn_lstm
    if count_sap_vel > 0:
        mean_sap_vel = sum_sap_vel / count_sap_vel
        variance_sap_vel_calc = (sum_squares_sap_vel / count_sap_vel) - (mean_sap_vel ** 2)
        variance_sap_vel = max(0, variance_sap_vel_calc) # Ensure variance is non-negative
        std_sap_vel = np.sqrt(variance_sap_vel)
    else: mean_sap_vel, std_sap_vel = np.nan, np.nan
    # Calculate quantiles from the sampled values (approximate)
    if sap_vel_values:
        try:
            percentiles = np.percentile(sap_vel_values, [25, 50, 75, 95])
            q1_sap_vel, median_sap_vel, q3_sap_vel, p95_sap_vel = percentiles
        except IndexError: q1_sap_vel = median_sap_vel = q3_sap_vel = p95_sap_vel = np.nan
    else: q1_sap_vel = median_sap_vel = q3_sap_vel = p95_sap_vel = np.nan

    # Print data overview
    print("\n=== DATA OVERVIEW ==="); print(f"Total rows processed: {total_rows:,}")
    if 'latitude' in first_chunk.columns and 'longitude' in first_chunk.columns:
        print(f"Unique latitudes (sampled): ~{len(lat_values):,}"); print(f"Unique longitudes (sampled): ~{len(lon_values):,}")
        if min_lat != float('inf'): print(f"Latitude range: [{min_lat}, {max_lat}]"); print(f"Longitude range: [{min_lon}, {max_lon}]")
        else: print("Lat/Lon range: Not determined")
    if 'sw_in' in first_chunk.columns: print(f"SW_in range: [{min_sw_in}, {max_sw_in}]" if min_sw_in != float('inf') else "SW_in range: Not determined")
    if 'sap_velocity_cnn_lstm' in first_chunk.columns:
        print("\n=== SAP VELOCITY STATISTICS ==="); print(f"Valid (non-null) measurements found: {count_sap_vel:,}")
        print(f"Missing/null values: {null_sap_vel_count:,}")
        if total_rows > 0: valid_perc = (count_sap_vel / total_rows) * 100 if count_sap_vel <= total_rows else 100; print(f"Valid data percentage: {valid_perc:.2f}%")
        else: print(f"Valid data percentage: N/A")
        if min_sap_vel != float('inf'):
            print(f"Value range (valid): [{min_sap_vel:.6f}, {max_sap_vel:.6f}]"); print(f"Mean (valid): {mean_sap_vel:.6f}"); print(f"Std dev (valid): {std_sap_vel:.6f}")
            print(f"Approx median (sampled): {median_sap_vel:.6f}"); print(f"Approx quartiles (sampled): Q1={q1_sap_vel:.6f}, Q3={q3_sap_vel:.6f}"); print(f"Approx 95th percentile: {p95_sap_vel:.6f}")
            print("\nValue distribution (valid):");
            for bin_name, count in sap_vel_bins.items(): percentage = (count / count_sap_vel) * 100 if count_sap_vel > 0 else 0; print(f"  {bin_name}: {count:,} ({percentage:.2f}%)")
        else: print("Sap velocity stats: Not calculated (no valid data)")

    print(f"\nInspection completed in {elapsed_time:.2f} seconds")
    print("\n=== SAMPLE DATA ==="); print("\nFirst few rows:"); print(beginning_rows)
    print(f"\nMiddle {len(middle_rows)} rows:" if middle_rows is not None else "\nMiddle rows sample: Not captured"); print(middle_rows if middle_rows is not None else "")
    print(f"\nLast {len(end_rows)} rows:" if end_rows is not None else "\nLast rows sample: Not captured"); print(end_rows if end_rows is not None else "")
    return {"total_rows": total_rows, "unique_lats": len(lat_values), "unique_lons": len(lon_values), "min_lat": min_lat if min_lat != float('inf') else None, "max_lat": max_lat if max_lat != float('-inf') else None, "min_lon": min_lon if min_lon != float('inf') else None, "max_lon": max_lon if max_lon != float('-inf') else None, "min_sw_in": min_sw_in if min_sw_in != float('inf') else None, "max_sw_in": max_sw_in if max_sw_in != float('-inf') else None, "min_sap_vel": min_sap_vel if min_sap_vel != float('inf') else None, "max_sap_vel": max_sap_vel if max_sap_vel != float('-inf') else None, "mean_sap_vel": mean_sap_vel, "median_sap_vel": median_sap_vel, "std_sap_vel": std_sap_vel, "valid_sap_vel_count": count_sap_vel, "null_sap_vel_count": null_sap_vel_count}


# =============================================================================
# Sorting Helper Functions
# =============================================================================
# Backward-compatible alias
inspect_large_csv = inspect_large_file


def _sort_using_sqlite(csv_filepath, chunk_size=100000):
    """
    Optimized SQLite sorting with progress reporting and efficient export.
    Includes fix for f-string syntax error.
    """
    print("Using optimized SQLite sorting strategy...")
    start_time = time.time(); temp_csv = None; temp_db = None
    try:
        # Setup temporary files
        temp_csv = os.path.splitext(csv_filepath)[0] + "_sorted_temp.csv"
        # Ensure temp directory exists if needed, or use default temp dir
        temp_dir = tempfile.gettempdir()
        # Create a unique DB name to avoid conflicts if run concurrently
        db_filename = os.path.basename(csv_filepath) + f"_{os.getpid()}.db"
        temp_db = os.path.join(temp_dir, db_filename)
        # Remove existing temp DB if it exists from a previous failed run
        if os.path.exists(temp_db):
            try: os.remove(temp_db)
            except OSError: print(f"Warning: Could not remove existing temp DB: {temp_db}")

        print(f"Temp DB: {temp_db}, Temp CSV: {temp_csv}")

        # Connect and configure SQLite
        conn = sqlite3.connect(temp_db)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA cache_size = -2000000;") # ~2GB cache

        # Get columns and check for 'name'
        if _is_parquet(csv_filepath):
            sample_df = pd.read_parquet(csv_filepath, engine="pyarrow").head(5)
        else:
            sample_df = pd.read_csv(csv_filepath, nrows=5)
        columns = sample_df.columns.tolist()
        if 'name' not in columns: raise ValueError("Sorting requires 'name' column.")

        # Create table
        create_table_sql = f"CREATE TABLE data ({', '.join([f'`{col}` TEXT' for col in columns])})"; conn.execute(create_table_sql)

        # Import data
        total_imported = 0
        if _is_parquet(csv_filepath):
            _full = pd.read_parquet(csv_filepath, engine="pyarrow")
            _chunks = [_full[i:i+chunk_size] for i in range(0, len(_full), chunk_size)]
        else:
            _chunks = pd.read_csv(csv_filepath, chunksize=chunk_size, low_memory=False)
        for chunk in tqdm(_chunks, desc="Importing to SQLite"):
            try: chunk.to_sql("data", conn, if_exists="append", index=False, method='multi', chunksize=10000); total_imported += len(chunk)
            except Exception as import_err: print(f"Error importing chunk: {import_err}. Skipping."); continue
        print(f"Finished importing {total_imported:,} rows.")

        # Create index
        print("Creating index on name column..."); conn.execute("CREATE INDEX idx_name ON data (name)"); print("Index created.")

        # Export sorted data
        print(f"Exporting sorted data to {temp_csv}...")
        cursor = conn.execute("SELECT COUNT(*) FROM data"); total_rows = cursor.fetchone()[0]; print(f"Total rows to export: {total_rows:,}")
        # Ensure temp CSV is removed if it exists
        if os.path.exists(temp_csv):
            try: os.remove(temp_csv)
            except OSError: print(f"Warning: Could not remove existing temp CSV: {temp_csv}")

        with open(temp_csv, 'w', newline='', encoding='utf-8') as f: f.write(','.join(columns) + '\n') # Write header first
        export_chunk_size = 250000; total_exported = 0
        with tqdm(total=total_rows, desc="Exporting from SQLite") as pbar:
            for offset in range(0, total_rows, export_chunk_size):
                cursor = conn.execute(f"SELECT * FROM data ORDER BY name LIMIT {export_chunk_size} OFFSET {offset}")
                with open(temp_csv, 'a', newline='', encoding='utf-8') as f: # Append to CSV
                    chunk_rows = 0
                    while True:
                        rows_batch = cursor.fetchmany(10000)
                        if not rows_batch: break
                        # *** CORRECTED SYNTAX FOR CSV QUOTING (No change from previous fix) ***
                        lines = []
                        for row in rows_batch:
                            processed_row = []
                            for val in row:
                                if val is None:
                                    processed_row.append('')
                                else:
                                    # Process the string first to handle quotes
                                    processed_val = str(val).replace('"', '""')
                                    # Then format it within quotes
                                    processed_row.append(f'"{processed_val}"') # This line is correct
                            lines.append(','.join(processed_row) + '\n')
                        # *********************************************************************
                        f.writelines(lines); chunk_rows += len(rows_batch); pbar.update(len(rows_batch))
                total_exported += chunk_rows
        elapsed = time.time() - start_time; print(f"SQLite sorting completed in {elapsed:.2f} seconds.")
        return temp_csv
    except Exception as e:
        print(f"Error during SQLite sorting: {e}")
        if temp_csv and os.path.exists(temp_csv):
            try: os.remove(temp_csv)
            except OSError: pass
        return csv_filepath # Return original on error
    finally:
        # Cleanup
        if 'conn' in locals() and conn: conn.close()
        if temp_db and os.path.exists(temp_db):
            try: os.remove(temp_db)
            except OSError as e: print(f"Warning: Could not remove temp DB {temp_db}: {e}")

def _sort_using_dask(csv_filepath):
    """
    Sort a large CSV file using Dask (distributed processing).
    """
    print("Sorting using Dask..."); start_time = time.time()
    temp_csv = os.path.splitext(csv_filepath)[0] + "_sorted_temp.csv"
    try:
        # Estimate blocksize
        file_size_mb = os.path.getsize(csv_filepath) / (1024**2); blocksize = '128MB' if file_size_mb < 10000 else '256MB'
        print(f"Using Dask blocksize: {blocksize}")
        # Read CSV
        if _is_parquet(csv_filepath):
            dask_df = dd.read_parquet(csv_filepath, engine="pyarrow")
        else:
            dask_df = dd.read_csv(csv_filepath, blocksize=blocksize, assume_missing=True)
        if 'name' not in dask_df.columns: raise ValueError("Sorting requires 'name' column.")
        # Set index and sort
        print("Setting index and sorting with Dask..."); sorted_dask_df = dask_df.set_index('name', shuffle='tasks').persist()
        # Write sorted file
        print(f"Writing sorted data to {temp_csv}..."); computation = sorted_dask_df.to_csv(temp_csv, single_file=True, index=False); computation.compute()
        elapsed = time.time() - start_time; print(f"Dask sorting completed in {elapsed:.2f} seconds.")
        return temp_csv
    except Exception as e:
        print(f"Error during Dask sorting: {e}")
        if os.path.exists(temp_csv):
            try: os.remove(temp_csv);
            except OSError: pass
        return csv_filepath # Return original on error
    finally: pass # Dask cleanup usually automatic

def _sort_in_memory(csv_filepath):
    """
    Sort a CSV file entirely in memory (for smaller files).
    """
    print("Attempting to sort in memory..."); start_time = time.time()
    temp_csv = os.path.splitext(csv_filepath)[0] + "_sorted_temp.csv"
    try:
        # Read, sort, write
        print("Reading entire file...")
        if _is_parquet(csv_filepath):
            df = pd.read_parquet(csv_filepath, engine="pyarrow")
        else:
            df = pd.read_csv(csv_filepath, low_memory=False)
        print(f"Read {len(df):,} rows.")
        if 'name' not in df.columns: raise ValueError("Sorting requires 'name' column.")
        print("Sorting DataFrame..."); df = df.sort_values('name')
        print(f"Writing sorted data to {temp_csv}..."); df.to_csv(temp_csv, index=False, encoding='utf-8')
        elapsed = time.time() - start_time; print(f"In-memory sorting completed in {elapsed:.2f} seconds.")
        return temp_csv
    except MemoryError: print("MemoryError: File too large for memory sort."); return csv_filepath
    except Exception as e:
        print(f"Error during in-memory sorting: {e}")
        if os.path.exists(temp_csv):
            try: os.remove(temp_csv);
            except OSError: pass
        return csv_filepath # Return original on error


# =============================================================================
# Tiling Helper Functions (MODIFIED TO USE ThreadPoolExecutor)
# =============================================================================
def _create_tiled_output(grouped_df, min_lat, max_lat, min_lon, max_lon,
                         lat_resolution, lon_resolution, output_tif, tile_size):
    """
    Create tiled output for very large grids using ThreadPoolExecutor.
    """
    print("Starting tiled output creation (using ThreadPoolExecutor)..."); start_time = time.time()
    epsilon = 1e-9 # Small value to handle floating point precision issues
    # Calculate total grid dimensions
    total_height = int(np.ceil((max_lat - min_lat + epsilon) / lat_resolution)); total_width = int(np.ceil((max_lon - min_lon + epsilon) / lon_resolution))
    print(f"Total grid dimensions for tiling: {total_height}x{total_width}")
    if total_height <= 0 or total_width <= 0: print("Error: Invalid grid dimensions."); return None, (min_lat, max_lat, min_lon, max_lon), (lat_resolution, lon_resolution)

    # Calculate number of tiles
    n_tiles_y = int(np.ceil(total_height / tile_size)); n_tiles_x = int(np.ceil(total_width / tile_size))
    print(f"Creating {n_tiles_y}×{n_tiles_x} tiles ({n_tiles_y * n_tiles_x} total)")

    # Setup output directory
    output_dir = os.path.splitext(output_tif)[0] + "_tiles"; os.makedirs(output_dir, exist_ok=True); print(f"Tile output directory: {output_dir}")

    # Prepare parameters for each tile process
    tile_params = []
    for tile_y in range(n_tiles_y):
        for tile_x in range(n_tiles_x):
            # Calculate pixel bounds for the tile
            y_start_px = tile_y * tile_size; x_start_px = tile_x * tile_size
            y_end_px = min(y_start_px + tile_size, total_height); x_end_px = min(x_start_px + tile_size, total_width)
            tile_h_px = y_end_px - y_start_px; tile_w_px = x_end_px - x_start_px
            if tile_h_px <= 0 or tile_w_px <= 0: continue # Skip empty dimension tiles

            # Calculate geographic bounds for filtering (slightly expanded)
            filter_max_lat = max_lat - y_start_px * lat_resolution + epsilon; filter_min_lat = max_lat - y_end_px * lat_resolution - epsilon
            filter_min_lon = min_lon + x_start_px * lon_resolution - epsilon; filter_max_lon = min_lon + x_end_px * lon_resolution + epsilon
            # Calculate tile origin (top-left corner) and transform
            tile_origin_lon = min_lon + x_start_px * lon_resolution; tile_origin_lat = max_lat - y_start_px * lat_resolution
            tile_transform = from_origin(tile_origin_lon, tile_origin_lat, lon_resolution, lat_resolution)
            tile_output_filename = os.path.join(output_dir, f"tile_{tile_y}_{tile_x}.tif")
            # Add parameters for the worker function
            tile_params.append((grouped_df, filter_min_lat, filter_max_lat, filter_min_lon, filter_max_lon, lat_resolution, lon_resolution, tile_output_filename, tile_h_px, tile_w_px, tile_origin_lat, tile_origin_lon, tile_transform))

    # Process tiles in parallel using ThreadPoolExecutor
    num_workers = max(1, mp.cpu_count() - 1) # Use physical cores minus one
    print(f"Processing tiles using {num_workers} threads...")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_param = {executor.submit(_process_tile, param): param for param in tile_params}
        # Process completed tasks using tqdm progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_param), total=len(tile_params), desc="Processing Tiles"):
            param = future_to_param[future]
            try:
                result = future.result() # Get result from future
                if result is not None:
                    results.append(result)
            except Exception as exc:
                # Get the original filename from params for logging
                failed_filename = param[7]
                print(f'\nTile {os.path.basename(failed_filename)} generated an exception: {exc}')
                # traceback.print_exc() # Optional: print full traceback for the failed future

    # Collect results and merge
    successful_tiles = [res[0] for res in results if res is not None and res[0] is not None]; print(f"Successfully created {len(successful_tiles)} tiles.")
    if not successful_tiles: print("No tiles created."); return None, (min_lat, max_lat, min_lon, max_lon), (lat_resolution, lon_resolution)

    merged_output_path = output_tif
    if len(successful_tiles) == 1:
         print(f"Single tile created. Copying to {merged_output_path}");
         try: shutil.copy(successful_tiles[0], merged_output_path);
         except Exception as copy_err: print(f"Error copying: {copy_err}"); merged_output_path = None
    elif len(successful_tiles) > 1:
        print(f"Merging {len(successful_tiles)} tiles into {merged_output_path}...")
        try:
            # Try merging with rasterio
            from rasterio.merge import merge; src_files_to_merge = [rasterio.open(tile_path) for tile_path in successful_tiles]
            mosaic, out_transform = merge(src_files_to_merge, nodata=np.nan, precision=7); out_meta = src_files_to_merge[0].meta.copy()
            for src in src_files_to_merge: src.close()
            # *** Ensure merged output also uses valid block sizes ***
            merged_height = mosaic.shape[1]
            merged_width = mosaic.shape[2]
            out_meta.update({"driver": "GTiff",
                             "height": merged_height,
                             "width": merged_width,
                             "transform": out_transform,
                             "crs": '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
                             "nodata": np.nan,
                             "compress": "LZW",
                             "tiled": True, # Keep tiled for potentially large merged files
                             "blockxsize": get_valid_block_size(merged_width), # Use helper
                             "blockysize": get_valid_block_size(merged_height) # Use helper
                             })
            with rasterio.open(merged_output_path, "w", **out_meta) as dest: dest.write(mosaic.astype(rasterio.float32))
            print(f"Merged GeoTIFF created: {merged_output_path}")
        except ImportError:
            # Fallback to GDAL VRT if rasterio merge fails
            print("Rasterio.merge unavailable. Creating VRT."); vrt_path = os.path.splitext(output_tif)[0] + ".vrt"
            try:
                import subprocess;
                if shutil.which('gdalbuildvrt'): subprocess.run(['gdalbuildvrt', '-overwrite', vrt_path] + successful_tiles, check=True, capture_output=True); print(f"VRT created: {vrt_path}"); merged_output_path = vrt_path
                else: print("ERROR: gdalbuildvrt not found."); merged_output_path = None
            except subprocess.CalledProcessError as vrt_err: print(f"Error running gdalbuildvrt: {vrt_err}\nStderr: {vrt_err.stderr.decode()}"); merged_output_path = None
            except Exception as vrt_err: print(f"Error creating VRT: {vrt_err}"); merged_output_path = None
        except Exception as merge_err: print(f"Error merging tiles: {merge_err}"); merged_output_path = None
    else: print("No successful tiles to merge."); merged_output_path = None

    elapsed = time.time() - start_time; print(f"Tiled processing complete in {elapsed:.2f} seconds.")
    # Return None for grid data as it's stored in files
    return None, (min_lat, max_lat, min_lon, max_lon), (lat_resolution, lon_resolution)

def _process_tile(params):
    """
    Helper function to process individual tiles for parallel processing.
    Filters data, creates grid, and saves tile GeoTIFF.
    Uses get_valid_block_size for profile.
    """
    (grouped_df, filter_min_lat, filter_max_lat, filter_min_lon, filter_max_lon,
     lat_resolution, lon_resolution, tile_output, height, width,
     tile_origin_lat, tile_origin_lon, tile_transform) = params
    try:
        # Filter data for the tile
        lat_f = pd.to_numeric(grouped_df['latitude'], errors='coerce'); lon_f = pd.to_numeric(grouped_df['longitude'], errors='coerce')
        tile_data_mask = (lat_f >= filter_min_lat) & (lat_f <= filter_max_lat) & (lon_f >= filter_min_lon) & (lon_f <= filter_max_lon)
        tile_data = grouped_df.loc[tile_data_mask].copy()
        if len(tile_data) == 0: return None # Skip empty tile

        # Prepare grid
        grid_data = np.full((height, width), np.nan, dtype=np.float32); epsilon = 1e-9
        # Ensure data types for calculation
        tile_lat = pd.to_numeric(tile_data['latitude'], errors='coerce'); tile_lon = pd.to_numeric(tile_data['longitude'], errors='coerce'); tile_vals = pd.to_numeric(tile_data['sap_velocity_cnn_lstm'], errors='coerce')
        valid_numeric_mask = tile_lat.notna() & tile_lon.notna() & tile_vals.notna()
        # Drop rows if coordinates/values became invalid
        if not valid_numeric_mask.all():
            tile_lat = tile_lat[valid_numeric_mask]; tile_lon = tile_lon[valid_numeric_mask]; tile_vals = tile_vals[valid_numeric_mask]
            if len(tile_lat) == 0: return None # Skip if no valid data left

        # Calculate indices relative to tile origin
        row_indices = np.floor((tile_origin_lat - tile_lat + epsilon) / lat_resolution).astype(int)
        col_indices = np.floor((tile_lon - tile_origin_lon + epsilon) / lon_resolution).astype(int)
        # Filter indices within tile bounds
        valid_mask = (row_indices >= 0) & (row_indices < height) & (col_indices >= 0) & (col_indices < width)
        valid_rows = row_indices[valid_mask]; valid_cols = col_indices[valid_mask]; valid_values = tile_vals.loc[valid_mask].values # Use .loc for safe indexing
        # Populate grid
        grid_data[valid_rows, valid_cols] = valid_values

        # Define raster profile and write tile
        # *** Use get_valid_block_size helper function ***
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': rasterio.float32,
            'crs': '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
            'transform': tile_transform,
            'nodata': np.nan,
            'compress': 'LZW',
            'tiled': True,
            'blockxsize': get_valid_block_size(width), # Apply fix here
            'blockysize': get_valid_block_size(height) # Apply fix here
        }
        # ************************************************

        with rasterio.open(tile_output, 'w', **profile) as dst: dst.write(grid_data, 1)
        return tile_output, (filter_min_lat, filter_max_lat, filter_min_lon, filter_max_lon) # Return path and extent
    except Exception as e:
        print(f"Error processing tile {os.path.basename(tile_output)}: {e}"); # traceback.print_exc() # Uncomment for full trace
        if os.path.exists(tile_output):
            try: os.remove(tile_output);
            except OSError: pass # Cleanup partial file
        return None # Indicate failure


# =============================================================================
# Plotting Function
# =============================================================================
def plot_sap_velocity_map(grid_data, extent, output_img='sap_velocity_map.png', max_size=5000,
                         shapefile_path='./data/raw/grided/tree_cover_shapefile_dissolved/tree_cover_shapefile_dissolved.shp', percentile_range=(10, 90), cmap='viridis'):
    """
    Visualize the sap velocity raster with proper geographical coordinates.
    Handles downsampling for large grids.
    
    Parameters:
    -----------
    grid_data : numpy.ndarray
        The grid data to plot
    extent : tuple
        (min_lat, max_lat, min_lon, max_lon) defining the geographic extent
    output_img : str
        Path to save the output image
    max_size : int
        Maximum size (in pixels) for either dimension before downsampling
    shapefile_path : str, optional
        Path to a shapefile for masking the data
    percentile_range : tuple
        (min_percentile, max_percentile) for color scaling, to control contrast
    cmap : str
        Name of the colormap to use
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import zoom
    
    if grid_data is None: 
        print("Grid data is None. Cannot plot.")
        return
    
    min_lat, max_lat, min_lon, max_lon = extent
    if not all(isinstance(x, (int, float)) for x in [min_lat, max_lat, min_lon, max_lon]): 
        print("Warning: Invalid extent for plotting.")
        return

    height, width = grid_data.shape
    plot_grid = grid_data.copy()
    
    # Downsample if grid is too large for plotting
    if height > max_size or width > max_size:
        downsample_factor = max(np.ceil(height / max_size), np.ceil(width / max_size))
        print(f"Downsampling visualization grid ({height}x{width}) by factor {downsample_factor:.1f}")
        zoom_factor = 1.0 / downsample_factor
        try:  # Try zoom first
            masked_grid = np.ma.masked_invalid(plot_grid)
            zoomed_data = zoom(masked_grid.data, zoom_factor, order=0, prefilter=False)
            zoomed_mask = zoom(masked_grid.mask, zoom_factor, order=0, prefilter=False)
            plot_grid = np.ma.masked_array(zoomed_data, mask=zoomed_mask).filled(np.nan)
            print(f"Downsampled shape: {plot_grid.shape}")
        except Exception as zoom_err:  # Fallback to slicing
            print(f"Warning: Zoom failed ({zoom_err}). Slicing.")
            step = max(1, int(downsample_factor))
            plot_grid = grid_data[::step, ::step]
    
    # Apply shapefile mask if provided
    if shapefile_path and os.path.exists(shapefile_path):
        try:
            print(f"Applying mask from shapefile: {shapefile_path}")
            # Import necessary libraries for shapefile processing
            import geopandas as gpd
            from rasterio import features
            from affine import Affine
            
            # Read the shapefile
            gdf = gpd.read_file(shapefile_path)
            
            # Create transform from pixel to geographic coordinates
            height, width = plot_grid.shape
            transform = Affine.translation(min_lon, max_lat) * Affine.scale(
                (max_lon - min_lon) / width, 
                (min_lat - max_lat) / height  # negative because of the orientation
            )
            
            # Create a mask from the shapefile
            mask = np.zeros(plot_grid.shape, dtype=np.uint8)
            shapes = [(geom, 1) for geom in gdf.geometry]
            features.rasterize(shapes, out=mask, transform=transform, fill=0, all_touched=True)
            
            # Apply the mask
            plot_grid = np.where(mask == 1, plot_grid, np.nan)
            print("Shapefile mask applied successfully")
        
        except ImportError as ie:
            print(f"Warning: Required library not found for shapefile masking: {ie}")
            print("Install geopandas and rasterio for shapefile support")
        except Exception as mask_err:
            print(f"Warning: Failed to apply shapefile mask: {mask_err}")
    
    # Mask invalid data for plotting
    masked_plot_data = np.ma.masked_invalid(plot_grid)
    valid_count_original = np.sum(~np.isnan(grid_data))
    valid_count_plot = np.sum(~np.isnan(plot_grid))

    # Handle case with no valid data to plot
    if valid_count_plot == 0:
        print("Warning: No valid data points found in the grid to plot.")
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_facecolor('gray')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Mean Sap Velocity Map\n(No valid data)')
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        plt.tight_layout()
        plt.savefig(output_img, dpi=150)
        plt.close()
        print(f"Empty visualization saved: {output_img}")
        return

    # Create plot - adjust the figure size to accommodate horizontal colorbar
    fig, ax = plt.subplots(figsize=(12, 11))  # Increase height slightly
    plot_extent = [min_lon, max_lon, min_lat, max_lat]
    
    # Determine color limits with user-defined percentiles for reduced contrast
    try:
        min_percentile, max_percentile = percentile_range
        vmin, vmax = np.nanpercentile(
            plot_grid[~np.isnan(plot_grid)], 
            [min_percentile, max_percentile]
        ) if valid_count_plot > 0 else (None, None)
        print(f"Color range: {vmin:.4f} to {vmax:.4f} (percentiles {min_percentile}-{max_percentile})")
    except IndexError:
        vmin, vmax = None, None  # Handle case of single value

    # Display image
    img = ax.imshow(
        masked_plot_data, 
        cmap=cmap, 
        origin='upper', 
        extent=plot_extent, 
        vmin=vmin, 
        vmax=vmax, 
        interpolation='nearest'
    )
    
    # Add labels, grid
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # Create horizontal colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.5)  # Position at bottom with padding
    cbar = plt.colorbar(img, cax=cax, orientation='horizontal')
    cbar.set_label('Mean Sap Velocity (kg/cm²/h)')
    
    # Add title
    ax.set_title(
        f'Mean Sap Velocity Map between 12:00-14:00 in July'
    )
    
    # Set aspect ratio
    try:
        mid_lat_rad = np.radians((min_lat + max_lat) / 2.0)
        cos_mid_lat = np.cos(mid_lat_rad)
        ax.set_aspect(1.0 / cos_mid_lat if abs(cos_mid_lat) > 1e-6 else 1.0)
    except ValueError:
        ax.set_aspect(1.0)
    
    # Save and close
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    plt.close(fig)
    print(f"Visualization saved: {output_img}")

# =============================================================================
# Main GeoTIFF Creation Function (MODIFIED TO CALC EXTENT/RES AFTER DASK AGG)
# =============================================================================
def create_sap_velocity_tif(data_filepath, output_tif='sap_velocity_map.tif',
                            chunk_size='128MB', # Dask blocksize
                            manual_resolution=None,
                            use_sparse=True,
                            create_tiles=False, tile_size=1000,
                            sw_in_threshold=15,
                            sorting_method='dask', # Default to dask
                            dask_scheduler='processes'): # Default to processes, suggest 'threads' for Windows issues
    """
    Process sap velocity data using Dask for parallel aggregation and create a GeoTIFF map.
    Calculates extent and resolution AFTER aggregation for robustness with sorted data.
    Uses get_valid_block_size for rasterio profile.
    """
    print(f"--- Starting GeoTIFF Creation Process (Dask Parallelized v2.1 - Block Size Fix) ---") # Updated version marker
    print(f"Input file: {data_filepath}")
    print(f"Output Target: {output_tif}")
    # ... [rest of initial parameter printing] ...
    print(f"SW_in Threshold (>): {sw_in_threshold}")
    print(f"Dask Blocksize: {chunk_size}")
    print(f"Tiling Enabled: {create_tiles}, Tile Size: {tile_size if create_tiles else 'N/A'}")
    print(f"Sorting Method: {sorting_method}")
    print(f"Sparse Array Usage (if not tiling): {use_sparse}")
    print(f"Dask Scheduler: {dask_scheduler if dask_scheduler else 'default'}")


    start_total_time = time.time()

    # --- File Checks and System Info ---
    if not os.path.exists(data_filepath): raise FileNotFoundError(f"Input file not found: {data_filepath}")
    file_size_gb = os.path.getsize(data_filepath) / (1024**3)
    print(f"Input file size: {file_size_gb:.2f} GB")
    try:
        mem_info = psutil.virtual_memory(); available_memory_gb = mem_info.available / (1024**3)
        total_memory_gb = mem_info.total / (1024**3)
        print(f"System Memory: {available_memory_gb:.2f} GB available / {total_memory_gb:.2f} GB total")
    except Exception as mem_err: print(f"Warning: Could not get system memory info: {mem_err}"); available_memory_gb = 4.0 # Assume low memory

    # --- Sorting Step (Optional) ---
    sorted_filepath = data_filepath
    temp_sorted_file_to_delete = None
    temp_dir_to_delete = None # For potential Dask temp dirs
    if _is_parquet(data_filepath):
        first_row = pd.read_parquet(data_filepath, engine="pyarrow").head(1)
    else:
        first_row = pd.read_csv(data_filepath, nrows=1)
    has_name_column = 'name' in first_row.columns
    if sorting_method and not has_name_column: print("WARNING: 'name' column not found. Cannot sort."); sorting_method = None

    if sorting_method:
        print(f"\n--- Sorting dataset by 'name' using '{sorting_method}' method ---")
        sort_start_time = time.time(); original_path = data_filepath
        # Call appropriate sorting function
        if sorting_method == 'dask': sorted_filepath = _sort_using_dask(data_filepath)
        elif sorting_method == 'sqlite':
             sqlite_chunk_size = 500000 if isinstance(chunk_size, str) and 'MB' in chunk_size else 100000
             sorted_filepath = _sort_using_sqlite(data_filepath, chunk_size=sqlite_chunk_size)
        elif sorting_method == 'memory':
             if file_size_gb < (available_memory_gb * 0.5): sorted_filepath = _sort_in_memory(data_filepath)
             else: print("File too large for memory sort. Falling back to Dask sort."); sorted_filepath = _sort_using_dask(data_filepath)
        else: print(f"Unknown sorting method '{sorting_method}'. Proceeding unsorted."); sorted_filepath = data_filepath
        # Handle result of sorting
        if sorted_filepath != original_path: temp_sorted_file_to_delete = sorted_filepath; print(f"Sorting completed in {time.time() - sort_start_time:.2f}s. Using: {sorted_filepath}")
        elif sorted_filepath == original_path and sorting_method in ['dask', 'sqlite', 'memory']: print(f"Sorting failed/aborted. Using original: {data_filepath}"); sorting_method = None
        else: print("Proceeding with original unsorted file.")

    # --- Quick Sample Check (Optional, for early warning only) ---
    print("\n--- Performing quick sample check ---")
    try:
        # Read a small sample quickly
        sample_dtypes = {'latitude': float, 'longitude': float, 'sw_in': float, 'sap_velocity_cnn_lstm': float}
        if _is_parquet(sorted_filepath):
            sample_df_check = pd.read_parquet(sorted_filepath, engine="pyarrow", columns=list(sample_dtypes.keys())).head(50000)
        else:
            sample_df_check = pd.read_csv(sorted_filepath, nrows=50000, usecols=list(sample_dtypes.keys()), dtype=sample_dtypes, low_memory=False)
        # Check for valid data in the sample
        sample_valid_check = sample_df_check.dropna(subset=['sap_velocity_cnn_lstm'])
        sample_daytime_valid_check = sample_valid_check[sample_valid_check['sw_in'] > sw_in_threshold]
        if len(sample_daytime_valid_check) == 0:
            print("WARNING: Initial sample (50k rows) has 0 valid daytime points. This might be normal for time-sorted data.")
        else:
            print(f"Quick sample check found {len(sample_daytime_valid_check)} valid points.")
        del sample_df_check, sample_valid_check, sample_daytime_valid_check # Cleanup
    except Exception as sample_err:
        print(f"Warning: Quick sample check failed: {sample_err}. Proceeding anyway.")


    # --- Process Full Dataset using Dask ---
    print(f"\n--- Processing full dataset ({sorted_filepath}) using Dask ---")
    grouped_df = None; agg_start_time = time.time()
    try:
        # Define dtypes for Dask read
        print(f"Reading CSV with Dask (blocksize={chunk_size})...")
        dtypes = {'latitude': float, 'longitude': float, 'sw_in': float, 'sap_velocity_cnn_lstm': float}
        if _is_parquet(sorted_filepath):
            all_cols = pd.read_parquet(sorted_filepath, engine="pyarrow").head(1).columns
            final_dtypes = {col: dtypes[col] for col in dtypes if col in all_cols}
            ddf = dd.read_parquet(sorted_filepath, engine="pyarrow", columns=list(final_dtypes.keys()))
        else:
            all_cols = pd.read_csv(sorted_filepath, nrows=1).columns
            final_dtypes = {col: dtypes[col] for col in dtypes if col in all_cols}
            ddf = dd.read_csv(sorted_filepath, blocksize=chunk_size, dtype=final_dtypes, usecols=list(final_dtypes.keys()), assume_missing=True)

        # Define Dask operations: filter, dropna, groupby, mean
        print("Filtering/Aggregating with Dask...")
        ddf_daytime = ddf[ddf['sw_in'] > sw_in_threshold]
        ddf_valid = ddf_daytime.dropna(subset=['sap_velocity_cnn_lstm'])
        # Group by lat/lon and calculate mean sap velocity
        grouped_dd = ddf_valid.groupby(['latitude', 'longitude'])['sap_velocity_cnn_lstm'].mean(split_out=16) # Adjust split_out

        # Execute Dask computation graph
        print("Computing aggregated results with Dask...")
        grouped_df = grouped_dd.compute(scheduler=dask_scheduler)
        grouped_df = grouped_df.reset_index() # Convert result to DataFrame
        unique_locations_found = len(grouped_df)
        print(f"Dask computation complete. Calculated means for {unique_locations_found:,} unique locations.")

        # Crucial Check: Ensure aggregation yielded results
        if unique_locations_found == 0:
             # Double-check if input was truly empty (this triggers computation again, potentially slow)
             try:
                 valid_count_check = len(ddf_valid)
                 if valid_count_check == 0:
                     raise ValueError("No valid daytime sap velocity data found in the entire dataset after Dask processing.")
                 else:
                     # This case should be rare if groupby worked correctly
                     raise ValueError(f"Dask resulted in empty grouped data, despite {valid_count_check} valid input records. Check grouping logic.")
             except Exception as check_err:
                  # If the check itself fails, assume no valid data
                  print(f"Could not verify input count: {check_err}")
                  raise ValueError("No valid daytime sap velocity data found in the entire dataset (or check failed).")

        # Cleanup Dask objects
        del ddf, ddf_daytime, ddf_valid, grouped_dd

    except Exception as dask_err:
        # Ensure temporary sorted file is removed on Dask error
        if temp_sorted_file_to_delete and os.path.exists(temp_sorted_file_to_delete):
            try: os.remove(temp_sorted_file_to_delete)
            except OSError: pass
        raise RuntimeError(f"Dask processing failed: {dask_err}") from dask_err
    print(f"Dask aggregation completed in {time.time() - agg_start_time:.2f} seconds.")

    # --- Clean up temporary sorted file ---
    if temp_sorted_file_to_delete and os.path.exists(temp_sorted_file_to_delete):
        print(f"Cleaning up temporary sorted file: {temp_sorted_file_to_delete}")
        try: os.remove(temp_sorted_file_to_delete)
        except Exception as e: print(f"Warning: Could not cleanup temp file {temp_sorted_file_to_delete}: {e}")
    # Dask might create temp dirs, but usually manages them; add specific cleanup if needed
    # if temp_dir_to_delete and os.path.exists(temp_dir_to_delete): ...

    # --- Calculate Extent and Resolution (NOW, after aggregation) ---
    print("\n--- Calculating Extent and Resolution from Aggregated Data ---")
    if grouped_df is None or len(grouped_df) == 0:
        raise ValueError("Aggregated data is empty, cannot proceed.")

    # Ensure coordinates are numeric before calculating extent/resolution
    grouped_df['latitude'] = pd.to_numeric(grouped_df['latitude'], errors='coerce')
    grouped_df['longitude'] = pd.to_numeric(grouped_df['longitude'], errors='coerce')
    grouped_df.dropna(subset=['latitude', 'longitude'], inplace=True) # Drop rows if coords became NaN
    if len(grouped_df) == 0:
        raise ValueError("Aggregated data has no valid coordinates after coercion.")

    # Calculate final extent
    min_lat = grouped_df['latitude'].min(); max_lat = grouped_df['latitude'].max()
    min_lon = grouped_df['longitude'].min(); max_lon = grouped_df['longitude'].max()
    print(f"Final data extent: Lat [{min_lat:.6f}, {max_lat:.6f}], Lon [{min_lon:.6f}, {max_lon:.6f}]")

    # Report value range
    min_mean_sap = grouped_df['sap_velocity_cnn_lstm'].min(); max_mean_sap = grouped_df['sap_velocity_cnn_lstm'].max()
    print(f"Mean Sap Velocity range: [{min_mean_sap:.6f}, {max_mean_sap:.6f}]")

    # Determine resolution (if not manual) from aggregated data
    if manual_resolution:
        lat_resolution, lon_resolution = manual_resolution
        print(f"Using manual resolution: Lat {lat_resolution:.8f}°, Lon {lon_resolution:.8f}°")
    else:
        print("Determining resolution from aggregated data...")
        unique_lats = np.sort(grouped_df['latitude'].unique())
        unique_lons = np.sort(grouped_df['longitude'].unique())
        # Use median difference for robustness
        lat_res_calc = np.median(np.diff(unique_lats)[np.diff(unique_lats) > 1e-9]) if len(unique_lats) > 1 else 0.001
        lon_res_calc = np.median(np.diff(unique_lons)[np.diff(unique_lons) > 1e-9]) if len(unique_lons) > 1 else 0.001
        # Fallbacks
        lat_resolution = lat_res_calc if not np.isnan(lat_res_calc) and lat_res_calc > 1e-9 else 0.001
        lon_resolution = lon_res_calc if not np.isnan(lon_res_calc) and lon_res_calc > 1e-9 else 0.001
        print(f"Determined resolution: Lat {lat_resolution:.8f}°, Lon {lon_resolution:.8f}°")

    # --- Prepare Grid and Raster Output ---
    print("\n--- Preparing grid for raster output ---")
    epsilon = 1e-9 # Small value for float precision
    # Calculate grid dimensions
    height = max(1, int(np.ceil((max_lat - min_lat + epsilon) / lat_resolution)))
    width = max(1, int(np.ceil((max_lon - min_lon + epsilon) / lon_resolution)))
    print(f"Calculated grid dimensions: {height:,} rows x {width:,} columns")
    grid_size_gb = (height * width * 4) / (1024**3) # Estimate size for float32
    print(f"Estimated grid size (dense): {grid_size_gb:.3f} GB")

    final_grid_data_for_plot = None # Will hold data for plotting if not tiling/too large

    # --- Tiling or Single File Output ---
    if create_tiles and (height > tile_size or width > tile_size):
        # Use the existing tiling function which uses multiprocessing
        print(f"\n--- Creating Tiled Output (Tile Size: {tile_size}x{tile_size}) ---")
        _, extent_tile, resolution_tile = _create_tiled_output(
            grouped_df, min_lat, max_lat, min_lon, max_lon,
            lat_resolution, lon_resolution, output_tif, tile_size
        )
        print("Tiling process finished.")
    else:
        # Create a single file
        print(f"\n--- Creating Single GeoTIFF Output: {output_tif} ---")
        # Calculate transform
        transform = from_origin(min_lon, max_lat, lon_resolution, lat_resolution)
        print(f"Raster transform: {transform}")
        # Decide if sparse intermediate is needed
        use_sparse_runtime = use_sparse and grid_size_gb > 1.0
        print(f"Using sparse array intermediate: {use_sparse_runtime}")

        grid_data = None # Initialize grid data

        # Ensure columns used for indexing are numeric
        lat_idx_col = pd.to_numeric(grouped_df['latitude'], errors='coerce')
        lon_idx_col = pd.to_numeric(grouped_df['longitude'], errors='coerce')
        val_idx_col = pd.to_numeric(grouped_df['sap_velocity_cnn_lstm'], errors='coerce')
        valid_idx_mask = lat_idx_col.notna() & lon_idx_col.notna() & val_idx_col.notna()

        # Warn and filter if any non-numeric values remain in aggregated data
        if not valid_idx_mask.all():
             print(f"Warning: Dropping {len(valid_idx_mask)-valid_idx_mask.sum()} rows before rasterizing due to non-numeric values.")
             lat_idx_col = lat_idx_col[valid_idx_mask]; lon_idx_col = lon_idx_col[valid_idx_mask]; val_idx_col = val_idx_col[valid_idx_mask]
             if len(lat_idx_col) == 0: raise ValueError("No valid numeric data left to rasterize.")

        # Calculate grid indices
        row_indices = np.floor((max_lat - lat_idx_col + epsilon) / lat_resolution).astype(int)
        col_indices = np.floor((lon_idx_col - min_lon + epsilon) / lon_resolution).astype(int)
        values = val_idx_col.values

        # Filter points/indices falling within grid bounds
        valid_mask = (row_indices >= 0) & (row_indices < height) & (col_indices >= 0) & (col_indices < width)
        valid_rows = row_indices[valid_mask]; valid_cols = col_indices[valid_mask]; valid_values = values[valid_mask]
        points_outside_grid = len(lat_idx_col) - len(valid_values)
        print(f"Points outside grid bounds: {points_outside_grid:,}")
        if len(valid_values) == 0: raise ValueError("No points fall within calculated grid bounds.")

        # Populate grid (sparse or dense)
        if use_sparse_runtime:
            print("Populating sparse matrix...")
            try:
                grid_data_sparse = csr_matrix((valid_values, (valid_rows, valid_cols)), shape=(height, width), dtype=np.float32)
                print(f"Sparse matrix created ({grid_data_sparse.nnz} non-zero). Converting to dense...")
                # Convert sparse to dense for writing
                grid_data = np.full((height, width), np.nan, dtype=np.float32)
                non_zero_rows, non_zero_cols = grid_data_sparse.nonzero()
                grid_data[non_zero_rows, non_zero_cols] = grid_data_sparse.data
                del grid_data_sparse; print("Conversion complete.")
            except MemoryError: print("MemoryError creating/converting sparse matrix."); raise
            except Exception as sparse_err: raise RuntimeError(f"Sparse matrix error: {sparse_err}") from sparse_err
        else:
            # Populate dense grid directly
            print("Populating dense array...")
            grid_data = np.full((height, width), np.nan, dtype=np.float32)
            grid_data[valid_rows, valid_cols] = valid_values
            print(f"Dense array populated ({len(valid_values):,} values).")

        # Write the GeoTIFF
        print(f"Writing GeoTIFF: {output_tif}")
        try:
            # Define output profile
            # *** Use get_valid_block_size helper function ***
            profile = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': 1,
                'dtype': rasterio.float32,
                'crs': '+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
                'transform': transform,
                'nodata': np.nan,
                'compress': 'LZW',
                'tiled': True,
                'blockxsize': get_valid_block_size(width), # Apply fix here
                'blockysize': get_valid_block_size(height) # Apply fix here
            }
            # ************************************************

            # Write data
            with rasterio.open(output_tif, 'w', **profile) as dst: dst.write(grid_data, 1)
            print(f"GeoTIFF created successfully: {output_tif}")
            # Keep data for plotting only if reasonably small
            if grid_size_gb < 2.0: final_grid_data_for_plot = grid_data
            else: print("Grid large, clearing from memory."); del grid_data; final_grid_data_for_plot = None
        except Exception as write_err:
            # Cleanup partial file on error
            if os.path.exists(output_tif):
                try: os.remove(output_tif);
                except OSError: pass
            # Re-raise with more context if possible
            if "height and width of TIFF dataset blocks must be multiples of 16" in str(write_err):
                 raise RuntimeError(f"Error writing GeoTIFF: {write_err}. Block sizes calculated: x={profile['blockxsize']}, y={profile['blockysize']} for raster dimensions w={width}, h={height}. Check get_valid_block_size logic.") from write_err
            else:
                 raise RuntimeError(f"Error writing GeoTIFF: {write_err}") from write_err


    total_elapsed = time.time() - start_total_time
    print(f"\n--- GeoTIFF Creation Process Finished ---")
    print(f"Total time: {total_elapsed:.2f} seconds")

    # Return data for plotting (or None) and metadata
    return final_grid_data_for_plot, (min_lat, max_lat, min_lon, max_lon), (lat_resolution, lon_resolution)


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    # --- Configuration ---
    # Use raw string (r"...") or forward slashes for paths, especially on Windows
    data_file = r'outputs/prediction/prediction_2015_07_01_02_03_predictions_raw.csv'  # Also accepts .parquet
    output_geotiff = r'sap_velocity_map_global_dask_v4_.tif' # Incremented version
    output_plot_png = r'sap_velocity_map_global_dask_v4.png' # Incremented version

    # Processing parameters
    DASK_BLOCKSIZE = '256MB' # Adjust based on RAM/cores (e.g., '128MB', '512MB')
    # DASK_SCHEDULER: Use 'threads' to avoid multiprocessing spawn issues on Windows
    DASK_SCHEDULER = 'threads' # Changed from 'processes'
    SW_IN_DAYTIME_THRESHOLD = 15 # Threshold for sw_in
    SORTING_METHOD = None # 'dask', 'sqlite', 'memory', or None
    ENABLE_TILING = False # Use True for very large global outputs
    TILE_PIXEL_SIZE = 1024 # Size of tiles if tiling is enabled
    USE_SPARSE_IF_NOT_TILING = True # Use sparse matrix for large single files
    MANUAL_RESOLUTION = None # e.g., (0.01, 0.01) to override auto-detection

    # --- Execution ---
    print("===== Starting Script (Dask Parallelized v2.1 - Block Size Fix) =====") # Updated version marker
    print(f"Input file: {data_file}")

    if not os.path.exists(data_file):
        print(f"ERROR: Input file not found at {data_file}")
    else:
        try:
            # 1. Inspect the CSV (optional)
            print("\n===== 1. Inspecting CSV =====")
            inspection_results = inspect_large_csv(csv_file, chunk_size=500000)
            if inspection_results is None: print("CSV inspection failed. Aborting."); exit()
            elif inspection_results.get("valid_sap_vel_count", 0) == 0: print("\nWARNING: No valid sap velocity values found in inspection."); # Potentially exit() here

            # 2. Create the GeoTIFF (or Tiles) using Dask aggregation
            print("\n===== 2. Creating GeoTIFF (using Dask) =====")
            grid_data_result, extent_result, resolution_result = create_sap_velocity_tif(
                data_filepath=data_file, output_tif=output_geotiff, chunk_size=DASK_BLOCKSIZE,
                manual_resolution=MANUAL_RESOLUTION, use_sparse=USE_SPARSE_IF_NOT_TILING,
                create_tiles=ENABLE_TILING, tile_size=TILE_PIXEL_SIZE,
                sw_in_threshold=SW_IN_DAYTIME_THRESHOLD, sorting_method=SORTING_METHOD,
                dask_scheduler=DASK_SCHEDULER
            )

            # 3. Plot the result (if applicable)
            print("\n===== 3. Plotting Result =====")
            if grid_data_result is not None and extent_result is not None:
                plot_sap_velocity_map(grid_data=grid_data_result, extent=extent_result, output_img=output_plot_png)
            elif ENABLE_TILING: print(f"Plotting skipped (tiling enabled). View output: '{output_geotiff}' or tiles.")
            else: print(f"Plotting skipped (grid not in memory). View output: '{output_geotiff}'.")

        # --- Error Handling ---
        except FileNotFoundError as fnf_err: print(f"\nERROR: File not found - {fnf_err}")
        except ValueError as val_err: print(f"\nERROR: Data issue - {val_err}")
        except MemoryError as mem_err: print(f"\nERROR: Out of memory - {mem_err}. Adjust DASK_BLOCKSIZE/tiling/RAM.")
        except ImportError as imp_err: print(f"\nERROR: Missing library - {imp_err}. Install required libraries (dask, pandas, etc.).")
        except RuntimeError as run_err: print(f"\nERROR: Runtime issue - {run_err}") # Catch errors raised within functions
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}"); print("Traceback:"); traceback.print_exc()
        finally:
            # Cleanup Dask client/cluster if started explicitly
            # Example:
            # if 'client' in locals() and client: client.close()
            # if 'cluster' in locals() and cluster: cluster.close()
            pass

    print("\n===== Script Finished =====")
