import time
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


from tqdm.auto import tqdm # Use auto version for better environment detection

def inspect_large_csv(
    csv_filepath,
    chunk_size=100000,
    sample_size=10,
    min_lat_filter=None,
    max_lat_filter=None,
    min_lon_filter=None,
    max_lon_filter=None
):
    """
    Memory-efficient inspection of large CSV files with optional regional filtering
    and detailed sap velocity analysis for the specified region.

    Args:
        csv_filepath (str): Path to the CSV file.
        chunk_size (int): Number of rows per chunk to process.
        sample_size (int): Number of rows to sample from the beginning, middle, and end.
        min_lat_filter (float, optional): Minimum latitude for filtering. Defaults to None.
        max_lat_filter (float, optional): Maximum latitude for filtering. Defaults to None.
        min_lon_filter (float, optional): Minimum longitude for filtering. Defaults to None.
        max_lon_filter (float, optional): Maximum longitude for filtering. Defaults to None.

    Returns:
        dict: A dictionary containing summary statistics about the file and the filtered region.
              Returns None if the file cannot be read.
    """
    start_time = time.time()
    region_filter_active = all(v is not None for v in [min_lat_filter, max_lat_filter, min_lon_filter, max_lon_filter])

    print(f"--- Starting Inspection for: {os.path.basename(csv_filepath)} ---")
    if region_filter_active:
        print(f"Applying Region Filter:")
        print(f"  Latitude Range: [{min_lat_filter}, {max_lat_filter}]")
        print(f"  Longitude Range: [{min_lon_filter}, {max_lon_filter}]")
    else:
        print("No region filter applied. Analyzing entire dataset.")

    # --- File Size and Initial Read ---
    try:
        file_size_bytes = os.path.getsize(csv_filepath)
        file_size_gb = file_size_bytes / (1024**3)
        print(f"\nFile Size: {file_size_gb:.2f} GB ({file_size_bytes:,} bytes)")
    except FileNotFoundError:
        print(f"Error: File not found at {csv_filepath}")
        return None
    except Exception as e:
        print(f"Error getting file size: {e}")
        return None

    print("\nLoading file schema and sample data...")
    try:
        # Read only a few rows initially to get columns and estimate row size
        first_chunk = pd.read_csv(csv_filepath, nrows=max(5, sample_size))
    except Exception as e:
        print(f"Error reading initial rows: {e}")
        return None

    print("\n=== FILE COLUMNS ===")
    if first_chunk.empty:
        print("File appears to be empty or could not be read.")
        return None
    for i, col in enumerate(first_chunk.columns):
        print(f"{i+1}. {col} - {first_chunk[col].dtype}")

    # --- Initialization ---
    # Overall file stats
    total_rows_processed = 0
    overall_min_lat, overall_max_lat = float('inf'), float('-inf')
    overall_min_lon, overall_max_lon = float('inf'), float('-inf')
    # Use sets for sampled unique values to avoid duplicates efficiently
    unique_lat_samples = set()
    unique_lon_samples = set()

    # Region-specific stats (initialized for the filtered region)
    filtered_rows_count = 0
    region_min_sw_in, region_max_sw_in = float('inf'), float('-inf')
    region_min_sap_vel, region_max_sap_vel = float('inf'), float('-inf')
    region_sum_sap_vel, region_count_sap_vel, region_sum_squares_sap_vel = 0, 0, 0
    region_sap_vel_values_sample = [] # For approximate median calculation within region
    region_null_sap_vel_count = 0
    region_sap_vel_bins = {'Negative': 0, '0 to 0.5': 0, '0.5 to 1': 0, '1 to 2': 0, '2 to 5': 0, '5 to 10': 0, '10+': 0}

    # Samples for display
    beginning_rows = first_chunk.head(sample_size).copy()
    middle_rows = None
    end_rows = None

    # --- Chunk Processing ---
    processed_chunks = 0
    estimated_total_chunks = 1 # Default if estimation fails

    try:
        # Estimate total chunks for progress bar
        row_size_bytes = first_chunk.memory_usage(index=True, deep=True).sum() / len(first_chunk) if len(first_chunk) > 0 else 100 # Avoid division by zero
        estimated_total_chunks = max(1, int(np.ceil(file_size_bytes / (chunk_size * row_size_bytes)))) if row_size_bytes > 0 else 1
        print(f"\nEstimated row size: {row_size_bytes:.2f} bytes")
        print(f"Estimated total chunks: {estimated_total_chunks:,}")
    except Exception as e:
        print(f"Could not estimate total chunks: {e}")
        estimated_total_chunks = None # Indicate estimation failed

    print(f"\nAnalyzing data in chunks of {chunk_size:,} rows...")

    try:
        chunk_iterator = pd.read_csv(csv_filepath, chunksize=chunk_size, low_memory=False)

        # Setup progress bar
        progress_bar = tqdm(
            chunk_iterator,
            total=estimated_total_chunks,
            desc="Inspecting Chunks",
            unit="chunk",
            disable=(estimated_total_chunks is None) # Disable if estimation failed
        )

        for chunk_num, chunk in enumerate(progress_bar):
            processed_chunks += 1
            current_chunk_rows = len(chunk)
            total_rows_processed += current_chunk_rows

            # --- Overall File Stats (Before Filtering) ---
            if 'latitude' in chunk.columns and 'longitude' in chunk.columns:
                lat_col_overall = pd.to_numeric(chunk['latitude'], errors='coerce')
                lon_col_overall = pd.to_numeric(chunk['longitude'], errors='coerce')

                # Update overall min/max, ignoring NaNs
                overall_min_lat = min(overall_min_lat, lat_col_overall.min(skipna=True))
                overall_max_lat = max(overall_max_lat, lat_col_overall.max(skipna=True))
                overall_min_lon = min(overall_min_lon, lon_col_overall.min(skipna=True))
                overall_max_lon = max(overall_max_lon, lon_col_overall.max(skipna=True))

                # Sample unique lat/lon values from this chunk (limit sample size per chunk)
                sample_limit = min(1000, current_chunk_rows) # Take at most 1000 unique values per chunk
                unique_lat_samples.update(lat_col_overall.dropna().unique()[:sample_limit])
                unique_lon_samples.update(lon_col_overall.dropna().unique()[:sample_limit])


            # --- Region Filtering ---
            filtered_chunk = chunk # Start with the original chunk
            if region_filter_active:
                if 'latitude' in chunk.columns and 'longitude' in chunk.columns:
                    # Use the already coerced columns if available, otherwise coerce again
                    lat_col = lat_col_overall if 'lat_col_overall' in locals() else pd.to_numeric(chunk['latitude'], errors='coerce')
                    lon_col = lon_col_overall if 'lon_col_overall' in locals() else pd.to_numeric(chunk['longitude'], errors='coerce')

                    # Apply the filter mask
                    filter_mask = (
                        (lat_col >= min_lat_filter) & (lat_col <= max_lat_filter) &
                        (lon_col >= min_lon_filter) & (lon_col <= max_lon_filter)
                    )
                    filtered_chunk = chunk[filter_mask] # Create the filtered view/copy
                else:
                    # If lat/lon columns needed for filtering are missing, skip region analysis for this chunk
                    filtered_chunk = pd.DataFrame(columns=chunk.columns) # Empty dataframe

            # Skip further processing for this chunk if filtering removed all rows
            if filtered_chunk.empty:
                continue

            # --- Region-Specific Stats (On Filtered Chunk) ---
            filtered_rows_count += len(filtered_chunk)

            # SW_in stats within region
            if 'sw_in' in filtered_chunk.columns:
                sw_col_region = pd.to_numeric(filtered_chunk['sw_in'], errors='coerce')
                region_min_sw_in = min(region_min_sw_in, sw_col_region.min(skipna=True))
                region_max_sw_in = max(region_max_sw_in, sw_col_region.max(skipna=True))

            # Sap velocity stats within region
            if 'sap_velocity_cnn_lstm' in filtered_chunk.columns:
                sap_col_region = pd.to_numeric(filtered_chunk['sap_velocity_cnn_lstm'], errors='coerce')
                region_null_sap_vel_count += sap_col_region.isnull().sum()
                valid_sap_vel_region = sap_col_region.dropna()

                if not valid_sap_vel_region.empty:
                    region_min_sap_vel = min(region_min_sap_vel, valid_sap_vel_region.min())
                    region_max_sap_vel = max(region_max_sap_vel, valid_sap_vel_region.max())
                    region_sum_sap_vel += valid_sap_vel_region.sum()
                    region_count_sap_vel += len(valid_sap_vel_region)
                    region_sum_squares_sap_vel += (valid_sap_vel_region ** 2).sum()

                    # Sample values for approximate median calculation (limit total samples)
                    if len(region_sap_vel_values_sample) < 5000: # Store up to 5000 samples for median
                        sample_size_vel = min(100, len(valid_sap_vel_region)) # Take up to 100 per chunk
                        region_sap_vel_values_sample.extend(
                            np.random.choice(valid_sap_vel_region.values, sample_size_vel, replace=False).tolist()
                        )

                    # Update distribution bins for the region
                    region_sap_vel_bins['Negative'] += (valid_sap_vel_region < 0).sum()
                    region_sap_vel_bins['0 to 0.5'] += ((valid_sap_vel_region >= 0) & (valid_sap_vel_region < 0.5)).sum()
                    region_sap_vel_bins['0.5 to 1'] += ((valid_sap_vel_region >= 0.5) & (valid_sap_vel_region < 1)).sum()
                    region_sap_vel_bins['1 to 2'] += ((valid_sap_vel_region >= 1) & (valid_sap_vel_region < 2)).sum()
                    region_sap_vel_bins['2 to 5'] += ((valid_sap_vel_region >= 2) & (valid_sap_vel_region < 5)).sum()
                    region_sap_vel_bins['5 to 10'] += ((valid_sap_vel_region >= 5) & (valid_sap_vel_region < 10)).sum()
                    region_sap_vel_bins['10+'] += (valid_sap_vel_region >= 10).sum()


            # --- Sample Middle/End Rows (from original chunk for context) ---
            if estimated_total_chunks and estimated_total_chunks > 1 and chunk_num == estimated_total_chunks // 2 and middle_rows is None:
                 middle_rows = chunk.head(sample_size).copy() # Sample from original chunk

            # Keep track of the last chunk processed
            end_rows = chunk.tail(sample_size).copy() # Sample from original chunk

        # Close the progress bar
        progress_bar.close()

    except Exception as e:
        print(f"\nError during chunk processing: {e}")
        print(f"Processed {processed_chunks} chunks ({total_rows_processed:,} rows) before error.")
        traceback.print_exc() # Print detailed traceback for debugging
        # Continue to report results gathered so far

    elapsed_time = time.time() - start_time

    # --- Final Calculations (Region Specific) ---
    if region_count_sap_vel > 0:
        region_mean_sap_vel = region_sum_sap_vel / region_count_sap_vel
        # Calculate variance carefully to avoid floating point issues
        variance_sap_vel_calc = (region_sum_squares_sap_vel / region_count_sap_vel) - (region_mean_sap_vel ** 2)
        region_variance_sap_vel = max(0, variance_sap_vel_calc) # Ensure variance is non-negative
        region_std_sap_vel = np.sqrt(region_variance_sap_vel)
    else:
        region_mean_sap_vel, region_std_sap_vel = np.nan, np.nan

    # Calculate approximate median from the sampled values for the region
    if region_sap_vel_values_sample:
        try:
            # Calculate 25th, 50th (median), 75th, 95th percentiles
            percentiles = np.percentile(region_sap_vel_values_sample, [25, 50, 75, 95])
            region_q1_sap_vel, region_median_sap_vel, region_q3_sap_vel, region_p95_sap_vel = percentiles
        except IndexError: # Handles cases with very few samples
             region_q1_sap_vel = region_median_sap_vel = region_q3_sap_vel = region_p95_sap_vel = np.nan
    else:
        region_q1_sap_vel = region_median_sap_vel = region_q3_sap_vel = region_p95_sap_vel = np.nan


    # --- Reporting ---
    print("\n" + "="*20 + " INSPECTION SUMMARY " + "="*20)
    print(f"Total rows processed from file: {total_rows_processed:,}")

    # Overall File Geo Extent
    print("\n=== OVERALL FILE GEOGRAPHIC EXTENT ===")
    if 'latitude' in first_chunk.columns and 'longitude' in first_chunk.columns:
        print(f"Unique latitudes sampled: ~{len(unique_lat_samples):,}")
        print(f"Unique longitudes sampled: ~{len(unique_lon_samples):,}")
        # Check if min/max were updated from initial infinity values
        overall_lat_range_valid = overall_min_lat != float('inf')
        overall_lon_range_valid = overall_min_lon != float('inf')
        print(f"Overall Latitude range: [{overall_min_lat}, {overall_max_lat}]" if overall_lat_range_valid else "Overall Latitude range: Not determined")
        print(f"Overall Longitude range: [{overall_min_lon}, {overall_max_lon}]" if overall_lon_range_valid else "Overall Longitude range: Not determined")
    else:
        print("Latitude/Longitude columns not found or not processed.")

    # Region Specific Results
    print("\n=== REGION SPECIFIC ANALYSIS ===")
    if region_filter_active:
        print(f"Region Filter Applied: Lat [{min_lat_filter}, {max_lat_filter}], Lon [{min_lon_filter}, {max_lon_filter}]")
        print(f"Rows found within specified region: {filtered_rows_count:,}")
        if total_rows_processed > 0:
             region_perc = (filtered_rows_count / total_rows_processed) * 100
             print(f"Percentage of total rows in region: {region_perc:.2f}%")
    else:
        print("Analysis covers the entire dataset (no region filter).")
        print(f"Total valid rows analyzed: {filtered_rows_count:,}") # Will equal total_rows_processed if no filter

    # SW_in stats for the region
    if 'sw_in' in first_chunk.columns:
        region_sw_in_range_valid = region_min_sw_in != float('inf')
        print(f"\nSW_in range (within region): [{region_min_sw_in}, {region_max_sw_in}]" if region_sw_in_range_valid else "SW_in range (within region): Not determined")

    # Sap velocity stats for the region
    if 'sap_velocity_cnn_lstm' in first_chunk.columns:
        print("\n--- SAP VELOCITY STATISTICS (within region) ---")
        print(f"Valid (non-null) measurements found: {region_count_sap_vel:,}")
        print(f"Missing/null values encountered: {region_null_sap_vel_count:,}")

        if filtered_rows_count > 0: # Use filtered_rows_count for percentage calculation
            valid_perc_region = (region_count_sap_vel / filtered_rows_count) * 100 if region_count_sap_vel <= filtered_rows_count else 100
            print(f"Valid data percentage (of region rows): {valid_perc_region:.2f}%")
        else:
            print(f"Valid data percentage (of region rows): N/A")

        region_sap_vel_range_valid = region_min_sap_vel != float('inf')
        if region_sap_vel_range_valid:
            print(f"\nValue range: [{region_min_sap_vel:.6f}, {region_max_sap_vel:.6f}]")
            print(f"Mean: {region_mean_sap_vel:.6f}")
            print(f"Median (approximate, from sample): {region_median_sap_vel:.6f}") # Clearly state it's approximate
            print(f"Min: {region_min_sap_vel:.6f}")
            print(f"Max: {region_max_sap_vel:.6f}")
            print(f"Std dev: {region_std_sap_vel:.6f}")
            print(f"Approx quartiles (sampled): Q1={region_q1_sap_vel:.6f}, Q3={region_q3_sap_vel:.6f}")
            print(f"Approx 95th percentile (sampled): {region_p95_sap_vel:.6f}")

            print("\nValue distribution (within region):")
            # Calculate percentages based on region_count_sap_vel
            total_bin_count = sum(region_sap_vel_bins.values()) # Should ideally equal region_count_sap_vel
            if total_bin_count > 0:
                 for bin_name, count in region_sap_vel_bins.items():
                     percentage = (count / total_bin_count) * 100
                     print(f"  {bin_name}: {count:,} ({percentage:.2f}%)")
            else:
                 print("  No data for distribution bins.")

        else:
            print("\nSap velocity stats: Not calculated (no valid data found in region).")
    else:
        print("\n'sap_velocity_cnn_lstm' column not found.")

    # --- Sample Data Display ---
    print("\n=== SAMPLE DATA ===")
    print("\nFirst few rows:")
    print(beginning_rows.to_string()) # Use to_string for better formatting

    print(f"\nMiddle {len(middle_rows) if middle_rows is not None else 0} rows (sampled near middle of file):")
    print(middle_rows.to_string() if middle_rows is not None else "Not captured (e.g., file too small or error)")

    print(f"\nLast {len(end_rows) if end_rows is not None else 0} rows (sampled from last processed chunk):")
    print(end_rows.to_string() if end_rows is not None else "Not captured (e.g., error during processing)")

    # --- Completion ---
    print(f"\nInspection completed in {elapsed_time:.2f} seconds.")
    print("="*58)

    # --- Return Dictionary ---
    results = {
        "file_path": csv_filepath,
        "file_size_gb": file_size_gb,
        "total_rows_processed": total_rows_processed,
        "overall_min_lat": overall_min_lat if overall_lat_range_valid else None,
        "overall_max_lat": overall_max_lat if overall_lat_range_valid else None,
        "overall_min_lon": overall_min_lon if overall_lon_range_valid else None,
        "overall_max_lon": overall_max_lon if overall_lon_range_valid else None,
        "unique_lats_sampled": len(unique_lat_samples),
        "unique_lons_sampled": len(unique_lon_samples),
        "region_filter_active": region_filter_active,
        "region_min_lat": min_lat_filter,
        "region_max_lat": max_lat_filter,
        "region_min_lon": min_lon_filter,
        "region_max_lon": max_lon_filter,
        "region_rows_count": filtered_rows_count,
        "region_sw_in_min": region_min_sw_in if region_sw_in_range_valid else None,
        "region_sw_in_max": region_max_sw_in if region_sw_in_range_valid else None,
        "region_sap_vel_min": region_min_sap_vel if region_sap_vel_range_valid else None,
        "region_sap_vel_max": region_max_sap_vel if region_sap_vel_range_valid else None,
        "region_sap_vel_mean": region_mean_sap_vel if region_sap_vel_range_valid else None,
        "region_sap_vel_median_approx": region_median_sap_vel if region_sap_vel_range_valid else None,
        "region_sap_vel_std_dev": region_std_sap_vel if region_sap_vel_range_valid else None,
        "region_sap_vel_valid_count": region_count_sap_vel,
        "region_sap_vel_null_count": region_null_sap_vel_count,
        "region_sap_vel_distribution": region_sap_vel_bins,
        "elapsed_time_seconds": elapsed_time,
    }
    return results

if __name__ == "__main__":
    # Replace with your actual file path
    file = './outputs/prediction/prediction_2015_07_01_02_03_predictions_raw.csv'

    region_stats = inspect_large_csv(
     file,
     chunk_size=500000, # Adjust chunk size based on memory
     min_lat_filter=60.0,
     max_lat_filter=65.0,
     min_lon_filter=-100.0,
     max_lon_filter=120.0
 )