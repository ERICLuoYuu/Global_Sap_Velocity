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
def merge_sap_env_data(output_file):
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
            sap_data = pd.read_csv(sap_data_file, parse_dates=['TIMESTAMP'])
            
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
    

def create_spatial_groups(
    lat: Union[np.ndarray, List],
    lon: Union[np.ndarray, List],
    method: str = 'kmeans',
    n_clusters: int = 10,
    eps: float = 0.5,
    min_samples: int = 5,
    lat_grid_size: float = 5.0,
    lon_grid_size: float = 5.0,
    random_state: int = 42
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Group geographical data points using various spatial grouping methods.
    
    Parameters
    ----------
    lat : array-like
        Latitude values
    lon : array-like
        Longitude values
    method : str, default='kmeans'
        Clustering method to use: 'kmeans', 'dbscan', or 'grid'
    n_clusters : int, default=10
        Number of clusters for KMeans
    eps : float, default=0.5
        Maximum distance between samples for DBSCAN
    min_samples : int, default=5
        Minimum number of samples in a neighborhood for DBSCAN
    lat_grid_size : float, default=5.0
        Size of latitude intervals for grid method (in degrees)
    lon_grid_size : float, default=5.0
        Size of longitude intervals for grid method (in degrees)
    random_state : int, default=42
        Random state for reproducibility
        
    Returns
    -------
    groups : np.ndarray
        Array of group labels
    stats : pd.DataFrame
        DataFrame containing statistics for each group
    """
    # Convert inputs to numpy arrays
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    
    if method == 'grid':
        # Create grid cells
        lat_bins = np.arange(
            np.floor(lat.min()),
            np.ceil(lat.max()) + lat_grid_size,
            lat_grid_size
        )
        lon_bins = np.arange(
            np.floor(lon.min()),
            np.ceil(lon.max()) + lon_grid_size,
            lon_grid_size
        )
        
        # Assign points to grid cells
        lat_indices = np.digitize(lat, lat_bins) - 1
        lon_indices = np.digitize(lon, lon_bins) - 1
        
        # Create unique group numbers for each grid cell
        n_lon_bins = len(lon_bins)
        groups = lat_indices * n_lon_bins + lon_indices
        
    elif method in ['kmeans', 'dbscan']:
        # Scale coordinates
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(np.column_stack([lat, lon]))
        
        # Perform clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
        else:  # dbscan
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        
        groups = clusterer.fit_predict(coords_scaled)
    
    else:
        raise ValueError("Method must be one of: 'kmeans', 'dbscan', or 'grid'")
    
    # Calculate group statistics
    stats = []
    for group in np.unique(groups):
        mask = groups == group
        stats.append({
            'group': group,
            'size': np.sum(mask),
            'mean_lat': np.mean(lat[mask]),
            'mean_lon': np.mean(lon[mask]),
            'std_lat': np.std(lat[mask]),
            'std_lon': np.std(lon[mask]),
            'min_lat': np.min(lat[mask]),
            'max_lat': np.max(lat[mask]),
            'min_lon': np.min(lon[mask]),
            'max_lon': np.max(lon[mask])
        })
    
    return groups, pd.DataFrame(stats)

def main():
    # Merge sap and env data
    output_file = Path('./data/processed/merged') / 'merged_data.csv'
    merged_data = merge_sap_env_data(output_file)
    print(f"Data merged and saved to {output_file}")


if __name__ == "__main__":
    main()