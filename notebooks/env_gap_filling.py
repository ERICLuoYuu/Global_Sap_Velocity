import pandas as pd
from pathlib import Path
import concurrent.futures

def interpolate_small_gaps(df, col, max_gap):
    """
    Interpolates only gaps of size <= max_gap.
    Larger gaps remain as NaN.
    """
    series = df[col]
    # Create a copy of the series to avoid modifying the original
    interpolated = series.copy()
    
    # Find gaps (sequences of NaN values)
    is_null = series.isnull()
    gap_groups = (~is_null).cumsum()  # Assign a unique group ID to each gap
    
    # Calculate the size of each gap
    gap_sizes = is_null.groupby(gap_groups).transform('sum')
    
    # Interpolate only for gaps <= max_gap
    mask = (is_null) & (gap_sizes > 0) & (gap_sizes <= max_gap)
    interpolated[mask] = series.interpolate(method='linear', limit=max_gap, limit_direction='both')[mask]
    
    return interpolated

def process_file(file_path, max_gap=2):
    """Process a single file with proper error handling"""
    try:
        # Read the CSV file
        filter_df = pd.read_csv(file_path)
        
        # Extract the base filename without extension
        base_name = file_path.stem
        parts = base_name.split('_')
        
        # Check if the file is already standardized and remove that part from filename
        if parts[-1] == 'standardized':
            parts.pop()
        output_base_name = '_'.join(parts)
        
        # Identify columns to process (all except TIMESTAMP)
        process_columns = [col for col in filter_df.columns if col != 'TIMESTAMP']
        
        if not process_columns:
            print(f"No data columns found in {file_path}. Skipping file.")
            return False
        
        # Convert TIMESTAMP to datetime and create a copy for resampling
        filter_df['TIMESTAMP'] = pd.to_datetime(filter_df['TIMESTAMP'])
        df = filter_df[['TIMESTAMP'] + process_columns].copy()
        
        # Set TIMESTAMP as index and resample to hourly
        df = df.set_index('TIMESTAMP')
        df = df.resample(rule='h').mean()
        
        # Process each column with interpolation
        interpolated_series = {}
        for col in process_columns:
            interpolated_series[col] = interpolate_small_gaps(df, col, max_gap)
        
        # Create a new dataframe with the interpolated results
        result_df = pd.DataFrame(interpolated_series, index=df.index)
        
        # Convert back to dataframe with TIMESTAMP as column
        result_df = result_df.reset_index()
        
        # Create output directory if it doesn't exist
        output_dir = Path(f'./outputs/processed_data/env/gap_filled_size{max_gap - 1}')
        output_dir.mkdir(parents=True, exist_ok=True)
        resample_dir = Path(f'./outputs/processed_data/env/daily_gap_filled_size{max_gap - 1}')
        resample_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        output_path = output_dir / f'{output_base_name}_gap_filled.csv'
        result_df.to_csv(output_path, index=False)
        # resample to daily, for precipitation use mean
        df = result_df.copy()
        df = df.set_index('TIMESTAMP')
        # Get columns to resample (exclude solar_TIMESTAMP)
        data_cols = [col for col in df.columns if col != 'solar_TIMESTAMP' and col != 'lat' and col != 'long']
        
        # Get columns that need sum vs mean
        sum_cols = [col for col in data_cols if 'precip' in col.lower()]
        mean_cols = [col for col in data_cols if col not in sum_cols]

        # Resample separately
        daily_sums = df[sum_cols].resample('D').sum() if sum_cols else None
        daily_means = df[mean_cols].resample('D').mean() if mean_cols else None

        # Combine results
        if daily_sums is not None and daily_means is not None:
            daily_df = pd.concat([daily_sums, daily_means], axis=1)
        elif daily_sums is not None:
            daily_df = daily_sums
        else:
            daily_df = daily_means

        # Add appropriate suffixes
        daily_df.columns = [f"{col}_sum" if col in sum_cols else f"{col}_mean" 
                            for col in daily_df.columns]
        daily_df.to_csv(resample_dir / f'{output_base_name}_daily_gap_filled.csv', index=True)
        
        print(f"Successfully processed {file_path}")
        return True
        
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return False

def main():
    try:
        # Find all processed CSV files
        filtered_files = list(Path('./outputs/processed_data/env/standardized').rglob('*.csv'))
        
        if not filtered_files:
            print("No CSV files found in the specified directory.")
            return
        
        print(f"Found {len(filtered_files)} files to process.")
        
        # Process files using concurrent execution
        success_count = 0
        failure_count = 0
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
            # Submit each file for processing
            future_to_file = {executor.submit(process_file, file_path): file_path 
                             for file_path in filtered_files}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    if future.result():
                        success_count += 1
                    else:
                        failure_count += 1
                except Exception as e:
                    print(f"Exception while processing {file_path}: {str(e)}")
                    failure_count += 1
        
        print(f"Processing complete. Successful: {success_count}, Failed: {failure_count}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()