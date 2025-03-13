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
    """Merge sap flow and environmental data for each plant type and location, grouping by biome."""
    if not output_file.exists():
        output_file.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store dataframes by biome
    biome_merged_data = {}
    
    # Dictionary to store individual site dataframes
    plant_merged_data = {}
    
    good_sites = []
    good_site_file = pd.read_csv('outputs/tables/env_era5_comparison/good_sites.csv')
    good_sites = good_site_file['site_name'].tolist()
    
    count = 0
    site_biome_mapping = {}  # Dictionary to keep track of which biome each site belongs to
    
    for sap_data_file in Path('./outputs/processed_data/sap/daily_gap_filled_size1').rglob("*.csv"):
        
        try:
            # Parse location and plant type
            parts = sap_data_file.stem.split("_")
            location_type = '_'.join(parts[:-5])
            # if location_type not in good_sites:
                #continue
            
            # Load data files
            env_data_file = Path('./outputs/processed_data/env/daily_gap_filled_size1_with_era5') / f"{location_type}_daily_gap_filled_with_era5.csv"
            
            # Load the site biome data
            biome_data_file = f'data/raw/0.1.5/0.1.5/csv/sapwood/{location_type}_site_md.csv'
            try:
                biome_info = pd.read_csv(biome_data_file)
                biome_type = biome_info['si_biome'][0]
                # Store the biome type for this site
                site_biome_mapping[location_type] = biome_type
            except Exception as be:
                print(f"Error reading biome data for {location_type}: {str(be)}")
                continue
                
            env_data = pd.read_csv(env_data_file, parse_dates=['TIMESTAMP'])
            
            
                
            count += 1
            print(f"loaded {location_type} (Biome: {biome_type})")
            print(f"count: {count}")

            sap_data = pd.read_csv(sap_data_file, parse_dates=['TIMESTAMP'])
            sap_data.drop(columns=[col for col in sap_data.columns if 'Unnamed' in col], inplace=True)
            
            # Set index for env_data
            env_data.set_index('TIMESTAMP', inplace=True)
            
            # Process each sap flow column
            col_names = [col for col in sap_data.columns if col != 'solar_TIMESTAMP' and col != 'TIMESTAMP']
            # Create a new column for the site to calculate the average sap velocity for each site
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
            
            # Change the column name
            df.rename(columns={df.columns[0]: 'sap_velocity'}, inplace=True)
            
            # Add biome information to the dataframe
            df['biome'] = biome_type
            df['site'] = location_type
            
            # Save individual site data
            df.to_csv(output_file/f'{location_type}_merged.csv')
            plant_merged_data[location_type] = df
            
            # Add to the biome dictionary
            if biome_type not in biome_merged_data:
                biome_merged_data[biome_type] = []
            
            biome_merged_data[biome_type].append(df)
                    
        except Exception as e:
            print(f"Error processing {sap_data_file}: {str(e)}")
            continue
    
    # Concatenate by biome and save
    if biome_merged_data:
        # Create a directory for biome-specific files
        biome_output_dir = output_file/'by_biome'
        biome_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each biome separately
        all_biome_dfs = []
        for biome, dfs in biome_merged_data.items():
            if dfs:
                # Concatenate all sites with the same biome
                biome_df = pd.concat(dfs)
                
                # Sanitize biome name for file path (remove special characters)
                safe_biome_name = ''.join(c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in biome)
                safe_biome_name = safe_biome_name.replace(' ', '_')
                
                # Save the biome-specific merged data
                biome_df.to_csv(biome_output_dir/f'{safe_biome_name}_merged_data.csv')
                all_biome_dfs.append(biome_df)
                print(f"Saved merged data for biome: {biome} with {len(dfs)} sites")
        
        # Also save the complete merged dataset as before
        if all_biome_dfs:
            all_data = pd.concat(all_biome_dfs)
            all_data.to_csv(output_file/'all_biomes_merged_data.csv')
            
            # Create a summary file with biome information
            biome_summary = pd.DataFrame({
                'site': list(site_biome_mapping.keys()),
                'biome': list(site_biome_mapping.values())
            })
            biome_summary.to_csv(output_file/'site_biome_mapping.csv', index=False)
            
            return all_data
        else:
            raise ValueError("No data was successfully processed after biome merging")
    else:
        raise ValueError("No data was successfully processed")

def main():
    # Merge sap and env data
    output_file = Path('./outputs/processed_data/merged/site/gap_filled_size1_with_era5') 
    # merged_data = merge_sap_env_data_plant(output_file)
    merged_data = merge_sap_env_data_site(output_file)
    print(f"Data merged and saved to {output_file}")


if __name__ == "__main__":
    main()