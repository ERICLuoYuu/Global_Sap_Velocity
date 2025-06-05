# Import the pandas library, which provides data manipulation tools
import pandas as pd

# --- Configuration ---
# Specify the paths to your input CSV files
file1_path = './data/outputs/predictions/2015_07_daily_tropics/prediction_2015_07_01_tropics.csv'
file2_path = './data/outputs/predictions/2015_07_daily/prediction_2015_07_01.csv'

# Specify the path for the output combined CSV file
output_path = './data/outputs/predictions/combined.csv'

# --- Script Logic ---
try:
    # Read the first CSV file into a pandas DataFrame
    # A DataFrame is like a table for storing data
    df1 = pd.read_csv(file1_path)
    print(f"Successfully loaded {file1_path}. Shape: {df1.shape}") # Show rows, columns

    # Read the second CSV file into another DataFrame
    df2 = pd.read_csv(file2_path)
    print(f"Successfully loaded {file2_path}. Shape: {df2.shape}") # Show rows, columns

    # Concatenate (combine) the two DataFrames vertically
    # ignore_index=True resets the index for the combined DataFrame
    combined_df = pd.concat([df1, df2], ignore_index=True)
    print(f"DataFrames concatenated. Combined shape: {combined_df.shape}")

    # Save the combined DataFrame to a new CSV file
    # index=False prevents pandas from writing the DataFrame index as a column
    combined_df.to_csv(output_path, index=False)
    print(f"Successfully saved combined data to {output_path}")

except FileNotFoundError:
    print(f"Error: One or both input files not found. Please check paths:")
    print(f"- {file1_path}")
    print(f"- {file2_path}")
except Exception as e:
    print(f"An error occurred: {e}")

