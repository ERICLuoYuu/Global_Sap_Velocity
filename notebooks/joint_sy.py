from pathlib import Path
import pandas as pd
import numpy as np
import sys

# Ensure path config is loaded
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from path_config import PathConfig

class JointAvailabilityAnalyzer:
    def __init__(self, paths: PathConfig = None, scale: str = 'sapwood'):
        self.paths = paths if paths is not None else PathConfig(scale=scale)
        self.sap_dir = self.paths.sap_data_dir
        self.env_dir = self.paths.env_data_dir

    def calculate_strict_overlap_site_years(self, 
                                              target_env_vars: list = None,
                                              output_filename: str = 'strict_overlap_site_years.csv') -> pd.DataFrame:
            """
            Calculates site-years where Sap Flow AND ALL specified environmental 
            variables are simultaneously available (Strict Intersection).
            """
            if target_env_vars is None:
                target_env_vars = ['sw_in', 'vpd', 'ta']

            print(f"Calculating strict intersection (Sapflow + { ' + '.join(target_env_vars) })")
            
            sap_files = list(self.sap_dir.glob("*_sapf_data.csv"))
            stats = []

            for i, sap_file in enumerate(sap_files):
                try:
                    # --- STEP 1: Smart File Matching (Same as before) ---
                    parts = sap_file.stem.split('_')
                    env_file = None
                    
                    # Match Country_Site_Subsite (e.g. USA_BNZ_BLA)
                    site_prefix_long = "_".join(parts[:3])
                    potential_env = list(self.env_dir.glob(f"{site_prefix_long}*_env_data.csv"))
                    
                    if potential_env:
                        env_file = potential_env[0]
                    else:
                        # Match Country_Site (e.g. USA_BNZ)
                        site_prefix_short = "_".join(parts[:2])
                        potential_env = list(self.env_dir.glob(f"{site_prefix_short}*_env_data.csv"))
                        if potential_env:
                            env_file = potential_env[0]

                    if not env_file:
                        continue

                    # --- STEP 2: Load Sap Flow Validity (WITH DEDUPLICATION) ---
                    sap_df = pd.read_csv(sap_file, parse_dates=['TIMESTAMP'])
                    sap_df = sap_df.set_index('TIMESTAMP')
                    
                    # FIX: Remove Duplicate Timestamps
                    if sap_df.index.duplicated().any():
                        sap_df = sap_df[~sap_df.index.duplicated(keep='first')]

                    sap_cols = [c for c in sap_df.columns if c not in ['solar_TIMESTAMP', 'lat', 'long', 'TIMESTAMP_LOCAL']]
                    if not sap_cols: 
                        continue
                    
                    sap_valid_mask = sap_df[sap_cols].notna().any(axis=1)
                    
                    if not sap_valid_mask.any():
                        continue

                    # --- STEP 3: Load Env Validity (WITH DEDUPLICATION) ---
                    env_header = pd.read_csv(env_file, nrows=0)
                    missing_vars = [v for v in target_env_vars if v not in env_header.columns]
                    
                    if missing_vars:
                        stats.append({
                            'sap_file': sap_file.name,
                            'env_file': env_file.name,
                            'overlap_years': 0.0,
                            'missing_vars': ", ".join(missing_vars)
                        })
                        continue

                    env_df = pd.read_csv(env_file, usecols=['TIMESTAMP'] + target_env_vars, parse_dates=['TIMESTAMP'])
                    env_df = env_df.set_index('TIMESTAMP')

                    # FIX: Remove Duplicate Timestamps
                    if env_df.index.duplicated().any():
                        env_df = env_df[~env_df.index.duplicated(keep='first')]

                    env_valid_mask = env_df[target_env_vars].notna().all(axis=1)

                    # --- STEP 4: Intersect and Calculate ---
                    # Now concat will work because indices are unique
                    aligned_df = pd.concat([sap_valid_mask, env_valid_mask], axis=1, join='inner')
                    aligned_df.columns = ['sap_valid', 'env_valid']

                    final_mask = aligned_df['sap_valid'] & aligned_df['env_valid']
                    valid_counts = final_mask.sum()

                    if valid_counts > 1:
                        timestep_sec = aligned_df.index.to_series().diff().median().total_seconds()
                        
                        if pd.isna(timestep_sec) or timestep_sec == 0:
                            timestep_sec = 1800 

                        seconds_per_year = 365.25 * 24 * 3600
                        years = (valid_counts * timestep_sec) / seconds_per_year
                    else:
                        years = 0.0

                    stats.append({
                        'sap_file': sap_file.name,
                        'env_file': env_file.name,
                        'overlap_years': years,
                        'missing_vars': 'None'
                    })

                except Exception as e:
                    # Add print to debug which file is still causing issues if any
                    print(f"Error processing {sap_file.name}: {e}")

            # --- STEP 5: Summary ---
            df_stats = pd.DataFrame(stats)
            
            if not df_stats.empty:
                df_stats = df_stats.sort_values('overlap_years', ascending=False)
                output_path = self.paths.sap_data_dir.parent / output_filename
                df_stats.to_csv(output_path, index=False)
                
                print("\n" + "="*60)
                print(f"STRICT OVERLAP SUMMARY (Sapflow + {target_env_vars})")
                print("="*60)
                print(f"Total Site-Years: {df_stats['overlap_years'].sum():.2f}")
                print(df_stats[['sap_file', 'overlap_years']].head())
                print(len(df_stats[df_stats['overlap_years'] > 0]), "sites processed.")
                return df_stats
            else:
                print("No matching data found.")
                return pd.DataFrame()

if __name__ == "__main__":
    analyzer = JointAvailabilityAnalyzer()
    
    # Define the variables that MUST exist for the data to be counted
    # Example: For Penman-Monteith you usually need Radiation, VPD (or RH/Ta), and Wind Speed.
    required_vars = ['sw_in', 'vpd', 'ws', 'ta', 'ppfd_in'] 
    
    df = analyzer.calculate_strict_overlap_site_years(target_env_vars=required_vars)