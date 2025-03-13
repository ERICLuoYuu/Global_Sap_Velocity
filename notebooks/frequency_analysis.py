import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def interpolate_small_gaps(series, max_gap=2):
    """
    Interpolates only gaps of size <= max_gap.
    Larger gaps remain as NaN.
    """
    # Create a copy of the series to avoid modifying the original
    interpolated = series.copy()
    
    # Find gaps (sequences of NaN values)
    is_null = series.isnull()
    gap_groups = (~is_null).cumsum()  # Assign a unique group ID to each gap
    
    # Calculate the size of each gap
    gap_sizes = is_null.groupby(gap_groups).transform('sum')
    
    # Interpolate only for gaps <= max_gap
    interpolated[(gap_sizes > 0) & (gap_sizes <= max_gap)] = series.interpolate(limit=max_gap, limit_direction='forward')
    
    return interpolated

def main():
    try:
        # Find all processed CSV files
        filtered_files = list(Path('./outputs/processed_data/sap/outlier_removed').rglob('*.csv'))
        if not filtered_files:
            print("No CSV files found in the specified directory.")
            return
            
        for filtered_file in filtered_files:  
            try:
                filter_df = pd.read_csv(filtered_file)
                process_columns = [col for col in filter_df.columns if col != 'TIMESTAMP']
                
                if not process_columns:
                    print(f"No data columns found in {filtered_file}. Skipping file.")
                    continue
                    
                for col in process_columns:
                    try:
                        # Extract data for this column
                        df = filter_df[['TIMESTAMP', col]].copy()
                        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
                        df = df.set_index('TIMESTAMP')
                        
                        # Resample data to hourly
                        df = df.resample(rule='h').mean()

                        # Interpolate gaps of size <= 2
                        df[col] = interpolate_small_gaps(df[col], max_gap=2)
                        
                        # Check if we have any valid data after processing
                        if df[col].isna().all() or len(df[col]) == 0:
                            print(f"No valid data for {col} in {filtered_file}. Skipping column.")
                            continue
                            
                        # Handle remaining NaNs by filling with the column mean
                        # This is necessary for FFT which cannot handle NaNs
                        col_mean = df[col].mean()
                        df[col] = df[col].fillna(col_mean)
                        
                        # Optional: Detrend the data (remove linear trend) for better FFT visualization
                        # Uncomment the following lines if you want to remove the linear trend
                        from scipy import signal
                        df[col] = signal.detrend(df[col].values)
                        
                        # Calculate FFT
                        fft_values = tf.signal.rfft(df[col].values)
                        fft_magnitudes = np.abs(fft_values)
                        
                        # Optional: Remove DC component (0 frequency) for better visualization
                        # Uncomment the line below if you want to zero out the DC component
                        fft_magnitudes = np.concatenate([[0], fft_magnitudes[1:]])
                        
                        # Calculate proper frequency bins
                        # For hourly data, the sampling frequency is 1/hour
                        n_samples = len(df[col])
                        sampling_freq = 1.0  # 1 sample per hour
                        freq_bins = np.fft.rfftfreq(n_samples, d=1/sampling_freq)
                        
                        # Convert frequency to more intuitive units
                        # freq_bins is in cycles/hour, convert to cycles/day and cycles/year
                        freq_per_day = freq_bins * 24  # 24 hours per day
                        freq_per_year = freq_bins * 24 * 365.2524  # hours per year
                        
                        # Create a new figure for each plot to avoid reuse
                        plt.figure(figsize=(12, 7))
                        
                        # Add a subplot for the original time series (optional)
                        # Uncomment this code block if you want to see the original time series above the FFT
                        """
                        plt.subplot(2, 1, 1)
                        df[col].plot()
                        plt.title(f'Time Series: {filtered_file.stem} - {col}')
                        plt.xlabel('Time')
                        plt.ylabel('Value')
                        plt.grid(True)
                        
                        plt.subplot(2, 1, 2)
                        """
                        
                        # Plot the FFT results
                        plt.step(freq_per_year, fft_magnitudes)
                        plt.xscale('log')
                        
                        # Exclude DC component (first value) when calculating y-axis limits
                        if len(fft_magnitudes) > 1:
                            # Skip the first value (DC component) for better scaling
                            max_amplitude = np.max(fft_magnitudes[1:])
                            # Set a reasonable limit based on the non-DC components
                            plt.ylim(0, max_amplitude * 1.5)  # Add 50% margin
                        else:
                            plt.ylim(0, np.max(fft_magnitudes) * 1.1)
                        
                        # Set dynamic x-axis limits and ticks
                        plt.xlim([0.1, max(freq_per_year)])
                        
                        # Add meaningful ticks - yearly, monthly, weekly, daily, hourly
                        major_cycles = [1, 12, 52, 365.2524, 365.2524*24]  # yearly, monthly, weekly, daily, hourly
                        major_labels = ['1/Year', '1/Month', '1/Week', '1/Day', '1/Hour']
                        
                        # Only include ticks that are within our frequency range
                        valid_ticks = [(freq, label) for freq, label in zip(major_cycles, major_labels) 
                                      if freq >= min(freq_per_year) and freq <= max(freq_per_year)]
                        
                        if valid_ticks:
                            plt.xticks([t[0] for t in valid_ticks], labels=[t[1] for t in valid_ticks])
                        
                        plt.xlabel('Frequency (cycles per year, log scale)')
                        plt.ylabel('Amplitude')
                        plt.title(f'FFT Analysis: {filtered_file.stem} - {col}')
                        plt.grid(True, which='both', linestyle='--', alpha=0.5)
                        
                        # Save the plot
                        output_dir = Path('./outputs/figures/fft')
                        output_dir.mkdir(parents=True, exist_ok=True)
                        output_path = output_dir / f'{filtered_file.stem}_{col}_fft.png'
                        plt.savefig(output_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        print(f"Successfully processed and saved FFT plot for {col} in {filtered_file}")
                        
                    except Exception as e:
                        print(f"Error processing column {col} in {filtered_file}: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"Error processing file {filtered_file}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()