import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Union, Tuple, List, Optional
# This script contains the tools used in the main script
# This function is used to integrate the sap flow data with the environmental data
# sap velocity data is stored in f'./data/processed/sap/daily/{location}_{plant_type}_daily.csv', evn data is stored in f'./data/processed/env/standardized/{location}_{plant_type}_standardized.csv'
# the output is stored in f'./data/processed/merged/{location}_{plant_type}_merged.csv'
# the function will gain location and plant type information from the file name
def merge_sap_env_data_plant(output_file):
    """Merge sap flow and environmental data for each plant type and location."""
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)
    plant_merged_data = {}
    
    for sap_data_file in Path('./data/processed/sap/daily').rglob("*.csv"):
        try:
            # Parse location and plant type
            parts = sap_data_file.stem.split("_")
            location = parts[0]
            plant_type = parts[1]
            
            # Load data files
            env_data_file = Path('./data/processed/env/standardized') / f"{location}_{plant_type}_standardized.csv"
            env_data = pd.read_csv(env_data_file, parse_dates=['TIMESTAMP'])
            # keep files having ['ta_mean', 'ws_mean', 'precip_sum', 'vpd_mean', sw_in_mean]
            if not all(col in env_data.columns for col in ['ta_mean', 'ws_mean', 'precip_sum', 'vpd_mean', 'sw_in_mean']):
                continue
            sap_data = pd.read_csv(sap_data_file, parse_dates=['TIMESTAMP'])
            # drop the columns that are not needed Unnamed: 0_mean
            sap_data.drop(columns=[col for col in sap_data.columns if 'Unnamed' in col], inplace=True)
            print(sap_data.columns)

            
            # Set index for env_data
            env_data.set_index('TIMESTAMP', inplace=True)
            
            # Process each sap flow column
            col_names = [col for col in sap_data.columns if col != 'solar_TIMESTAMP' and col != 'TIMESTAMP']
            for col_name in col_names:
                # Create proper DataFrame with TIMESTAMP
                plant_sap_data = pd.DataFrame({
                    'TIMESTAMP': sap_data['TIMESTAMP'],
                    col_name: sap_data[col_name]
                })
                plant_sap_data.set_index('TIMESTAMP', inplace=True)
                
                # Merge data
                df = pd.merge(
                    plant_sap_data, 
                    env_data,
                    left_index=True,
                    right_index=True
                )
                # change the column name
                df.rename(columns={df.columns[0]: 'sap_velocity'}, inplace=True)
                # save each df to the plant_merged_data dictionary
                df.to_csv(f'./data/processed/merged/plant/{col_name[:-5]}_merged.csv')
                if col_name not in plant_merged_data:
                    plant_merged_data[col_name] = df
                    
        except Exception as e:
            print(f"Error processing {sap_data_file}: {str(e)}")
            continue
    
    # Concatenate and save
    if plant_merged_data:
        merged_data = pd.concat(plant_merged_data.values())
        merged_data.to_csv(output_file)
        return merged_data
    else:
        raise ValueError("No data was successfully processed")
    
    

def merge_sap_env_data_site(output_file):
    """Merge sap flow and environmental data for each plant type and location."""
    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)
    plant_merged_data = {}
    count = 0
    for sap_data_file in Path('./outputs/processed_data/sap/daily').rglob("*.csv"):
        print(str(sap_data_file)[24:-10])
        try:
            # Parse location and plant type
            parts = sap_data_file.stem.split("_")
            location_type = '_'.join(parts[:-3])
            
            
            # Load data files
            env_data_file = Path('./outputs/processed_data/env/standardized') / f"{location_type}_env_data_standardized.csv"
            env_data = pd.read_csv(env_data_file, parse_dates=['TIMESTAMP'])
            # keep files having ['ta_mean', 'ws_mean', 'precip_sum', 'vpd_mean', sw_in_mean]
            if not all(col in env_data.columns for col in ['ta_mean', 'ws_mean', 'precip_sum', 'vpd_mean', 'sw_in_mean']):
                continue
            count += 1
            # count loaded files, the number
            print(f"loaded {location_type}")
            print(f"loaded {location_type}")
            print(f"count: {count}")


            sap_data = pd.read_csv(sap_data_file, parse_dates=['TIMESTAMP'])
            sap_data.drop(columns=[col for col in sap_data.columns if 'Unnamed' in col], inplace=True)
            
            # Set index for env_data
            env_data.set_index('TIMESTAMP', inplace=True)
            
            # Process each sap flow column
            col_names = [col for col in sap_data.columns if col != 'solar_TIMESTAMP' and col != 'TIMESTAMP']
            # create a new column for the site to calculate the average sap velocity for each site, there are several plants in each site
            sap_data['site'] = sap_data[col_names].mean(axis=1)
            

            plant_sap_data = pd.DataFrame({
                    'TIMESTAMP': sap_data['TIMESTAMP'],
                    'sap': sap_data['site']
                })
            plant_sap_data.set_index('TIMESTAMP', inplace=True)
                
            # Merge data
            df = pd.merge(
                    plant_sap_data, 
                    env_data,
                    left_index=True,
                    right_index=True
                )
                # change the column name
            df.rename(columns={df.columns[0]: 'sap_velocity'}, inplace=True)
                # save each df to the plant_merged_data dictionary
            df.to_csv(output_file/f'/{location_type}_merged.csv')

            plant_merged_data[location_type] = df
                    
        except Exception as e:
            print(f"Error processing {sap_data_file}: {str(e)}")
            continue
    
    # Concatenate and save
    if plant_merged_data:
        merged_data = pd.concat(plant_merged_data.values())
        merged_data.to_csv(output_file)
        return merged_data
    else:
        raise ValueError("No data was successfully processed")

def main():
    # Merge sap and env data
    output_file = Path('./outputs/processed_data/merged/site') / 'merged_data.csv'
    # merged_data = merge_sap_env_data_plant(output_file)
    merged_data = merge_sap_env_data_site(output_file)
    print(f"Data merged and saved to {output_file}")


if __name__ == "__main__":
    main()