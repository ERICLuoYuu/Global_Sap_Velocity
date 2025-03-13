import sys
from pathlib import Path
import pandas as pd
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from  src.tools import calculate_gap_size, plot_gap_size
"""
all_gaps = []
for filtered_sap_data_file in Path('./outputs/processed_data/sap/filtered').rglob("*filtered.csv"):
    
    df = pd.read_csv(filtered_sap_data_file)
    # resample to hourly
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    
    plant_columns = [col for col in df.columns if col != 'solar_TIMESTAMP' and col!= 'TIMESTAMP' and 'Unnamed' not in col]
    print(plant_columns)
    for plant_column in plant_columns:
        time_serie = pd.DataFrame(df.loc[:, ['TIMESTAMP', plant_column]])
        # time_serie = time_serie.resample(rule='H', on='TIMESTAMP').mean()
        print(time_serie.columns)
        print(time_serie.head())
        gaps = calculate_gap_size(time_serie)
        all_gaps.extend(gaps)
plot_gap_size(Path('outputs/figures/gaps'), all_gaps, frequency='H')  
"""
# calculate gap size for daily data
all_gaps = []
for daily_sap_data_file in Path('./outputs/processed_data/sap/daily').rglob("*daily.csv"):
    df = pd.read_csv(daily_sap_data_file)
    plant_columns = [col for col in df.columns if col != 'solar_TIMESTAMP' and col!= 'TIMESTAMP' and 'Unnamed' not in col]
    print(plant_columns)
    for plant_column in plant_columns:
        time_serie = pd.DataFrame(df.loc[:, ['TIMESTAMP', plant_column]])
        print(time_serie.columns)
        print(time_serie.head())
        gaps = calculate_gap_size(time_serie)
        all_gaps.extend(gaps)
plot_gap_size(Path('outputs/figures/gaps'), all_gaps, frequency='D')