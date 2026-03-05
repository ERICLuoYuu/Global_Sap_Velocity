import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Union, Tuple, List, Optional
# Assuming path_config is in your local environment
from path_config import PathConfig, get_default_paths

paths = get_default_paths()

def calculate_soil_hydraulics(sand, clay, organic_matter, coarse_fragments_vol_percent):
    """
    Calculate soil hydraulic properties using Saxton & Rawls (2006).
    """
    # 1. Permanent Wilting Point (1500 kPa)
    theta_wp = (
        -0.024 * sand + 0.487 * clay + 0.006 * organic_matter + 
        0.005 * (sand * organic_matter) - 0.013 * (clay * organic_matter) + 
        0.068 * (sand * clay) + 0.031
    )
    
    # 2. Field Capacity (33 kPa)
    theta_fc = (
        -0.251 * sand + 0.195 * clay + 0.011 * organic_matter + 
        0.006 * (sand * organic_matter) - 0.027 * (clay * organic_matter) + 
        0.452 * (sand * clay) + 0.299
    ) + (1.283 * (theta_wp)**2 - 0.374 * (theta_wp) - 0.015)

    # 3. Saturation (Porosity)
    theta_sat = (
        0.278 * sand + 0.034 * clay + 0.022 * organic_matter + 
        -0.018 * (sand * organic_matter) - 0.027 * (clay * organic_matter) + 
        -0.584 * (sand * clay) + 0.078
    ) + (0.636 * theta_wp - 0.107)
    
    # ADJUST FOR COARSE FRAGMENTS (ROCKS)
    cf_frac = coarse_fragments_vol_percent / 100.0
    
    theta_wp_adj = theta_wp * (1 - cf_frac)
    theta_fc_adj = theta_fc * (1 - cf_frac)
    theta_sat_adj = theta_sat * (1 - cf_frac)
    
    return theta_wp_adj, theta_fc_adj, theta_sat_adj

def get_weighted_soil_props(soil_row, depths=['0_5', '5_15', '15_30', '30_60', '60_100']):
    """
    Calculate thickness-weighted average for soil properties down to 100cm.
    """
    thickness_map = {'0_5': 5, '5_15': 10, '15_30': 15, '30_60': 30, '60_100': 40}
    properties = ['sand', 'clay', 'soc', 'bdod', 'cfvo']
    weighted_sums = {p: 0.0 for p in properties}
    total_thickness = 0.0
    
    for depth in depths:
        thick = thickness_map.get(depth, 0)
        if f'sand_{depth}' not in soil_row.index: continue
        if pd.isna(soil_row.get(f'sand_{depth}')): continue
            
        total_thickness += thick
        for p in properties:
            val = soil_row.get(f'{p}_{depth}', 0)
            if pd.isna(val): val = 0
            weighted_sums[p] += val * thick
            
    if total_thickness == 0:
        return {p: np.nan for p in properties}
        
    return {p: weighted_sums[p] / total_thickness for p in properties}

def load_stand_age_map(base_path):
    """
    Load the global stand age CSV and create a dictionary mapping site_name -> mean_age.
    """
    age_file_path = base_path / "extracted_data" / "terrain_site_data" / "sapwood" / "stand_age_data.csv"
    
    if not age_file_path.exists():
        age_file_path = paths.base_data_dir / "raw" / "extracted_data" / "terrain_site_data" / "sapwood" / "stand_age_data.csv"

    if age_file_path.exists():
        print(f"Loading Stand Age data from: {age_file_path}")
        df = pd.read_csv(age_file_path)
        tc_cols = [c for c in df.columns if 'stand_age_TC' in c]
        if tc_cols:
            df['mean_age'] = df[tc_cols].mean(axis=1)
            return df.set_index('site_name')['mean_age'].to_dict()
    
    print(f"WARNING: Stand Age file not found at {age_file_path}")
    return {}

def merge_sap_env_data_site(output_base_dir):
    
    # --- SETUP DIRECTORIES ---
    if not output_base_dir.exists():
        output_base_dir.mkdir(parents=True, exist_ok=True)
        
    hourly_out_dir = output_base_dir / "hourly"
    hourly_out_dir.mkdir(exist_ok=True)
    
    daily_out_dir = output_base_dir / "daily"
    daily_out_dir.mkdir(exist_ok=True)
    
    # Collections for final merge
    biome_merged_hourly = {}
    biome_merged_daily = {}
    
    all_pft_types = set()
    site_biome_mapping = {}

    # --- LOAD STATIC DATASETS ---
    site_info = pd.read_csv(paths.env_extracted_data_path)
    
    soil_file_path = paths.terrain_attributes_data_path.parent / "soilgrids_data.csv"
    if soil_file_path.exists():
        soil_data = pd.read_csv(soil_file_path)
        print("Loaded SoilGrids data.")
    else:
        print(f"WARNING: SoilGrids data not found.")
        soil_data = pd.DataFrame() 

    stand_age_map = load_stand_age_map(paths.raw_csv_dir.parent)

    era5_data = pd.read_csv(paths.era5_discrete_data_path)
    # Avoid division by zero
    era5_data['prcip/PET'] = era5_data['total_precipitation_hourly'] / (era5_data['potential_evaporation_hourly'] + 1e-6)
    
    lai_data = pd.read_csv(paths.globmap_lai_data_path)
    pft_data = pd.read_csv(paths.pft_data_path)
    
    pft_mapping = {
        1: 'Evergreen Needleleaf Forests', 2: 'Evergreen Broadleaf Forests', 3: 'Deciduous Needleleaf Forests',
        4: 'Deciduous Broadleaf Forests', 5: 'Mixed Forests', 6: 'Closed Shrublands', 7: 'Open Shrublands',
        8: 'Woody Savannas', 9: 'Savannas', 10: 'Grasslands', 11: 'Permanent Wetlands', 12: 'Cropland',
        13: 'Urban and Built-up Lands', 14: 'Cropland/Natural Vegetation Mosaics', 15: 'Permanent Snow and Ice',
        16: 'Areas with less than 10% vegetation', 17: 'Water Bodies'
    }
    pft_data['pft'] = pft_data['landcover_type'].map(pft_mapping)

    for df_temp in [pft_data, lai_data, era5_data]:
        df_temp.rename(columns={'timestamp': 'TIMESTAMP', 'datetime': 'TIMESTAMP'}, inplace=True)
        df_temp['TIMESTAMP'] = pd.to_datetime(df_temp['TIMESTAMP'], utc=True)
        df_temp.sort_values('TIMESTAMP', inplace=True)

    count = 0

    # --- MAIN LOOP ---
    for sap_data_file in paths.sap_outliers_removed_dir.glob("*.csv"):
        try:
            parts = sap_data_file.stem.split("_")
            location_type = '_'.join(parts[:-4])
            
            # --- LOAD SITE METADATA ---
            env_data_file = paths.env_outliers_removed_dir / f"{location_type}_env_data_outliers_removed.csv"
            biome_data_file = paths.raw_csv_dir / f'{location_type}_site_md.csv'
            
            try:
                biome_info = pd.read_csv(biome_data_file)
                biome_type = biome_info['si_biome'][0]
                igbp_type = biome_info['si_igbp'][0]
                lat = biome_info['si_lat'][0]
                lon = biome_info['si_long'][0]
                
                curr_site_info = site_info[site_info['site_name'] == location_type]
                if curr_site_info.empty: continue
                    
                ele = curr_site_info['elevation_m'].values[0]
                slope = curr_site_info['slope_deg'].values[0]
                aspect_sin = curr_site_info['aspect_sin'].values[0]
                aspect_cos = curr_site_info['aspect_cos'].values[0]
                mean_annual_temp = curr_site_info['bio1'].values[0]
                mean_annual_precip = curr_site_info['bio12'].values[0]
                canopy_height = curr_site_info['canopy_height_m'].values[0]
                temp_seasonality = curr_site_info['bio4'].values[0]
                precip_seasonality = curr_site_info['bio15'].values[0]

                # --- 1. EXTRACT STAND AGE ---
                stand_age = np.nan
                if location_type in stand_age_map:
                    stand_age = stand_age_map[location_type]
                
                if pd.isna(stand_age):
                    stand_md_file = paths.raw_csv_dir / f'{location_type}_stand_md.csv'
                    if stand_md_file.exists():
                        try:
                            s_md = pd.read_csv(stand_md_file)
                            if 'st_age' in s_md.columns:
                                val = s_md['st_age'].iloc[0]
                                stand_age = pd.to_numeric(val, errors='coerce')
                        except Exception:
                            pass

                # --- 2. EXTRACT SOIL PROPS ---
                avg_props = {'sand': np.nan, 'clay': np.nan, 'soc': np.nan, 'bdod': np.nan, 'cfvo': np.nan}
                raw_soil_props = {}
                theta_wp, theta_fc, theta_sat = np.nan, np.nan, np.nan
                
                if not soil_data.empty and location_type in soil_data['site_name'].values:
                    soil_row = soil_data[soil_data['site_name'] == location_type].iloc[0]
                    try:
                        avg_props = get_weighted_soil_props(soil_row)
                        
                        raw_props = ['sand', 'clay', 'soc', 'bdod', 'cfvo']
                        raw_depths = ['0_5', '5_15', '15_30', '30_60', '60_100']
                        
                        for p in raw_props:
                            for d in raw_depths:
                                col_key = f"{p}_{d}"
                                val = soil_row.get(col_key, np.nan)
                                raw_soil_props[f"soil_{col_key}"] = val

                        if not pd.isna(avg_props['sand']) and not pd.isna(avg_props['clay']):
                            om_pct = (avg_props['soc'] / 10.0) * 1.72 
                            theta_wp, theta_fc, theta_sat = calculate_soil_hydraulics(
                                sand=avg_props['sand'], clay=avg_props['clay'], 
                                organic_matter=om_pct, coarse_fragments_vol_percent=avg_props['cfvo']
                            )
                    except Exception as soil_e:
                        print(f"Error calculating soil hydraulics for {location_type}: {soil_e}")

                site_biome_mapping[location_type] = biome_type
                all_pft_types.add(igbp_type)
                
            except Exception as be:
                continue

            # --- LOAD TIME SERIES ---
            env_data = pd.read_csv(env_data_file, parse_dates=['TIMESTAMP'])
            sap_data = pd.read_csv(sap_data_file, parse_dates=['TIMESTAMP'])
            sap_data.drop(columns=[col for col in sap_data.columns if 'Unnamed' in col], inplace=True)
            
            env_data.set_index('TIMESTAMP', inplace=True)
            
            col_names = [col for col in sap_data.columns if col not in ['solar_TIMESTAMP', 'TIMESTAMP', 'TIMESTAMP_LOCAL']]
            sap_data['site'] = sap_data[col_names].mean(axis=1)
            
            plant_sap_data = pd.DataFrame({
                'TIMESTAMP': sap_data['TIMESTAMP'],
                'sap_velocity': sap_data['site'],
                'TIMESTAMP_LOCAL': sap_data.get('TIMESTAMP_LOCAL')
            }).set_index('TIMESTAMP')

            df = pd.merge(plant_sap_data, env_data, left_index=True, right_index=True)

            # --- PROCESS HOURLY DATA ---
            # Resample env/sap data to strictly hourly
            mean_cols = [c for c in ['vpd', 'ta', 'ws', 'sw_in', 'rh', 'netrad', 'ppfd_in', 'ext_rad', 'sap_velocity'] if c in df.columns]
            sum_cols = [c for c in ['precip'] if c in df.columns]
            
            agg_dict = {
                **{col: 'sum' for col in sum_cols},
                **{col: 'mean' for col in mean_cols},
                **{col: 'first' for col in df.columns if 'solar_TIMESTAMP' in col}
            }
            
            df_hourly = df.resample('h').agg(agg_dict).reset_index()
            if 'solar_TIMESTAMP_y' in df_hourly.columns:
                df_hourly.rename(columns={'solar_TIMESTAMP_y': 'solar_TIMESTAMP'}, inplace=True)

            # --- ADD STATIC VARIABLES (Common function for DF population) ---
            def add_static_vars(target_df):
                target_df['biome'] = biome_type
                target_df['pft'] = igbp_type
                target_df['site_name'] = location_type
                target_df['latitude'] = lat
                target_df['longitude'] = lon
                target_df['elevation'] = ele
                target_df['slope'] = slope
                target_df['aspect_sin'] = aspect_sin
                target_df['aspect_cos'] = aspect_cos
                target_df['mean_annual_temp'] = mean_annual_temp
                target_df['mean_annual_precip'] = mean_annual_precip
                target_df['temp_seasonality'] = temp_seasonality
                target_df['precip_seasonality'] = precip_seasonality
                target_df['canopy_height'] = canopy_height
                target_df['stand_age'] = stand_age
                target_df['soil_sand'] = avg_props['sand']
                target_df['soil_clay'] = avg_props['clay']
                target_df['soil_bdod'] = avg_props['bdod']
                target_df['soil_soc'] = avg_props['soc']
                target_df['soil_cfvo'] = avg_props['cfvo']
                for k, v in raw_soil_props.items():
                    target_df[k] = v
                target_df['soil_theta_wp'] = theta_wp
                target_df['soil_theta_fc'] = theta_fc
                target_df['soil_theta_sat'] = theta_sat
                return target_df

            df_hourly = add_static_vars(df_hourly)

            # --- MERGE EXTERNAL DATA (LAI, ERA5) TO HOURLY ---
            df_hourly = pd.merge_asof(df_hourly, lai_data, on='TIMESTAMP', by='site_name', direction='nearest')
            df_hourly = pd.merge_asof(df_hourly, era5_data, on='TIMESTAMP', by='site_name', direction='nearest')

            # --- SAVE HOURLY ---
            df_hourly.to_csv(hourly_out_dir / f'{location_type}_hourly.csv', index=False)
            
            # --- PROCESS DAILY DATA (Aggregation from Hourly) ---
            # 1. Define aggregation rules
            # Sum precipitation columns
            daily_sum_cols = ['precip', 'total_precipitation_hourly']
            
            # Mean for continuous numeric variables (excluding sum_cols and TIMESTAMP)
            daily_agg_rules = {}
            
            for col in df_hourly.columns:
                if col == 'TIMESTAMP': continue
                
                if col in daily_sum_cols:
                    daily_agg_rules[col] = 'sum'
                elif pd.api.types.is_numeric_dtype(df_hourly[col]):
                    daily_agg_rules[col] = 'mean'
                else:
                    # For strings (biome, site_name) or dates (solar_timestamp), take the first valid
                    daily_agg_rules[col] = 'first'
            
            # 2. Resample
            df_daily = df_hourly.resample('D', on='TIMESTAMP').agg(daily_agg_rules).reset_index()
            
            # --- SAVE DAILY ---
            df_daily.to_csv(daily_out_dir / f'{location_type}_daily.csv', index=False)

            # --- COLLECT FOR BIOME MERGING ---
            if biome_type not in biome_merged_hourly:
                biome_merged_hourly[biome_type] = []
                biome_merged_daily[biome_type] = []
                
            biome_merged_hourly[biome_type].append(df_hourly)
            biome_merged_daily[biome_type].append(df_daily)
            
            count += 1
            if count % 10 == 0: print(f"Processed {count} sites...")

        except Exception as e:
            print(f"Error processing {sap_data_file}: {str(e)}")
            continue

    # --- FINAL MERGING (HOURLY & DAILY) ---
    def save_biome_collections(collection_dict, output_dir, file_suffix):
        all_dfs = []
        biome_dir = output_dir / 'by_biome'
        biome_dir.mkdir(exist_ok=True)
        
        for biome, dfs in collection_dict.items():
            if dfs:
                biome_df = pd.concat(dfs)
                safe_biome_name = ''.join(c if c.isalnum() or c in [' ', '_', '-'] else '_' for c in biome).replace(' ', '_')
                biome_df.to_csv(biome_dir / f'{safe_biome_name}_{file_suffix}.csv', index=False)
                all_dfs.append(biome_df)
        
        if all_dfs:
            all_data = pd.concat(all_dfs)
            all_data.to_csv(output_dir / f'all_biomes_{file_suffix}.csv', index=False)
            return all_data
        return None

    if biome_merged_hourly:
        print("Saving merged hourly data...")
        save_biome_collections(biome_merged_hourly, hourly_out_dir, "merged_hourly")
        
        print("Saving merged daily data...")
        daily_all = save_biome_collections(biome_merged_daily, daily_out_dir, "merged_daily")
        
        # Save mapping
        pd.DataFrame({
            'site': list(site_biome_mapping.keys()),
            'biome': list(site_biome_mapping.values())
        }).to_csv(output_base_dir / 'site_biome_mapping.csv', index=False)
        
        print(f"Processing complete. Data saved to {output_base_dir}")
        return daily_all # Returning daily all as an example, or both
    
    raise ValueError("No data was successfully processed")

def main():
    # Defines the base directory for output
    output_base_dir = paths.merged_data_root
    merge_sap_env_data_site(output_base_dir)

if __name__ == "__main__":
    main()