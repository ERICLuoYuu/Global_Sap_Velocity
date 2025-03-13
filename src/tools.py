import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Union, Tuple, List, Optional
from pathlib import Path
from typing import Tuple
import copy
import matplotlib.pyplot as plt
from time import sleep
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
    for sap_data_file in Path('./data/processed/sap/daily').rglob("*.csv"):
        print(str(sap_data_file)[24:-10])
        try:
            # Parse location and plant type
            parts = sap_data_file.stem.split("_")
            location_type = '_'.join(parts[:-3])
            
            
            # Load data files
            env_data_file = Path('./data/processed/env/standardized') / f"{location_type}_env_data_standardized.csv"
            env_data = pd.read_csv(env_data_file, parse_dates=['TIMESTAMP'])
            # keep files having ['ta_mean', 'ws_mean', 'precip_sum', 'vpd_mean', sw_in_mean]
            if not all(col in env_data.columns for col in ['ta_mean', 'ws_mean', 'precip_sum', 'vpd_mean', 'sw_in_mean']):
                continue
            count += 1
            # count loaded files, the number
            print(f"loaded {str(sap_data_file)[24:-10]}")
            print(f"loaded {str(env_data_file)[24:-10]}")
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
            df.to_csv(f'./data/processed/merged/site/{str(sap_data_file)[24:-10]}_merged.csv')

            plant_merged_data[str(sap_data_file)[24:-10]] = df
                    
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



def create_timezone_mapping():
    """Create a mapping of timezone codes to their UTC offsets in hours and minutes."""
    timezone_map = {
        "1UTC-12:00, Y": (12, 0),
        "2UTC-11:00, X": (11, 0),
        "3UTC-10:00, W": (10, 0),
        "4UTC-09:30, V†": (9, 30),
        "5UTC-09:00, V": (9, 0),
        "6UTC-08:00, U": (8, 0),
        "7UTC-07:00, T": (7, 0),
        "8UTC-06:00, S": (6, 0),
        "9UTC-05:00, R": (5, 0),
        "11UTC-04:00, Q": (4, 0),
        "12UTC-03:30, P†": (3, 30),
        "13UTC-03:00, P": (3, 0),
        "14UTC-02:00, O": (2, 0),
        "15UTC-01:00, N": (1, 0),
        "16UTC±00:00, Z": (0, 0),
        "17UTC+01:00, A": (-1, 0),
        "18UTC+02:00, B": (-2, 0),
        "19UTC+03:00, C": (-3, 0),
        "20UTC+03:30, C†": (-3, -30),
        "21UTC+04:00, D": (-4, 0),
        "22UTC+04:30, D†": (-4, -30),
        "23UTC+05:00, E": (-5, 0),
        "24UTC+05:30, E†": (-5, -30),
        "25UTC+05:45, E*": (-5, -45),
        "26UTC+06:00, F": (-6, 0),
        "27UTC+06:30, F†": (-6, -30),
        "28UTC+07:00, G": (-7, 0),
        "29UTC+08:00, H": (-8, 0),
        "30UTC+08:30, H†": (-8, -30),
        "31UTC+08:45, H*": (-8, -45),
        "32UTC+09:00, I": (-9, 0),
        "33UTC+09:30, I†": (-9, -30),
        "34UTC+10:00, K": (-10, 0),
        "35UTC+10:30, K†": (-10, -30),
        "36UTC+11:00, L": (-11, 0),
        "37UTC+12:00, M": (-12, 0),
        "38UTC+12:45, M*": (-12, -45),
        "39UTC+13:00, M†": (-13, 0),
        "40UTC+14:00, M†": (-14, 0)
    }
    return timezone_map

def clean_timezone_string(timezone: str) -> str:
    """Clean timezone string by removing any special characters."""
    return timezone.replace('`', '').strip()

def adjust_time_to_utc(timestamp: pd.Timestamp, timezone: str, timezone_map: dict) -> pd.Timestamp:
    """Convert local time to UTC based on timezone offset."""
    # Clean the timezone string
    clean_tz = clean_timezone_string(timezone)
    if clean_tz not in timezone_map:
        raise ValueError(f"Unknown timezone: {timezone} (cleaned: {clean_tz})")

    hours, minutes = timezone_map[clean_tz]
    return timestamp + pd.Timedelta(hours=hours, minutes=minutes)

def extract_site_info(directory: Path) -> pd.DataFrame:
    """
    Extract site information from files and convert timestamps to UTC.

    Args:
        directory: Path object pointing to the directory containing the CSV files

    Returns:
        DataFrame containing site information with UTC-adjusted timestamps
    """
    # Create timezone mapping
    timezone_map = create_timezone_mapping()

    # Get all relevant files
    sapf_files = list(directory.glob("*_sapf_data.csv"))

   

    site_dict = {}

    for sapf_file in sapf_files:
        # Read the CSV files
        sapf_df = pd.read_csv(sapf_file)
        site_file = sapf_file.parent / f"{sapf_file.stem.replace('sapf_data', 'site_md')}.csv"
        site_df = pd.read_csv(site_file)
        tz_file = sapf_file.parent / f"{sapf_file.stem.replace('sapf_data', 'env_md')}.csv"
        tz_df = pd.read_csv(tz_file)
        if not (sapf_file and site_file and tz_file):
            raise FileNotFoundError("Missing required CSV files in directory")
        # Extract required information
        timezone = tz_df['env_time_zone'].iloc[0]
        lon = site_df['si_long'].iloc[0]
        lat = site_df['si_lat'].iloc[0]

        # Convert timestamps to datetime objects if they aren't already
        sapf_df['TIMESTAMP'] = pd.to_datetime(sapf_df['TIMESTAMP'])
        
        # Get min and max timestamps and convert to UTC
        start_date = adjust_time_to_utc(sapf_df['TIMESTAMP'].min(), timezone, timezone_map)
        end_date = adjust_time_to_utc(sapf_df['TIMESTAMP'].max(), timezone, timezone_map)

        # Store the information
        site_name = sapf_file.stem.split("_")
        site_name = "_".join(site_name[:-2])
        site_dict[site_name] = {
            'start_date': start_date,
            'end_date': end_date,
            'lat': lat,
            'lon': lon,
            'site_name': site_name
        }

    return pd.DataFrame.from_dict(site_dict, orient='index')
def calculate_gap_size(time_serie: Union[pd.Series, pd.DataFrame]) -> list:
    """
    Calculate the gap sizes in a time series dataset.
    
    Parameters:
    -----------
    time_serie : Union[pd.Series, pd.DataFrame]
        The time series data.
    frequency_unit : str, default 'h'
        The unit for frequency calculation.
        
    Returns:
    --------
    list
        A list of gap sizes in the time series.
    """
    # Check if time_serie is a DataFrame
    if isinstance(time_serie, pd.DataFrame):
        if 'TIMESTAMP' in time_serie.columns:
            # Ensure TIMESTAMP is a datetime object, not a method
            time_serie['TIMESTAMP'] = pd.to_datetime(time_serie['TIMESTAMP'])
        elif time_serie.index.name == 'TIMESTAMP':
            time_serie = time_serie.sort_index()
        else:
            raise ValueError("Time series must have a 'TIMESTAMP' column or index")
    else:
        raise ValueError("Input must be a pandas DataFrame")
    
    # Drop NaN values
    time_serie = time_serie.dropna()
    
    # Ensure timestamp column is properly processed for diff calculations
    # Calculate time differences and convert to seconds
    time_diffs = time_serie['TIMESTAMP'].diff()
    
    # Convert time differences to seconds - this is where the error was occurring
    time_diffs_seconds = time_diffs.dt.total_seconds()
    
    # Find the minimum non-zero time difference to use as the base unit
    # Use a small epsilon to avoid division by zero issues with floating point
    min_diff = time_diffs_seconds[time_diffs_seconds > 0].min()
    if pd.isna(min_diff):
        min_diff = 1.0  # Default if no valid differences
    
    print(f"Minimum time difference in seconds: {min_diff}")
    print(f"Time differences in seconds: {time_diffs_seconds}")
    
    # Calculate gaps as multiples of the minimum difference, subtract 1 to get actual gap
    time_serie.loc[:, 'gap'] = round(time_diffs_seconds / min_diff) - 1
    
    # Fill NA values (the first row will have NA since there's no previous timestamp)
    time_serie['gap'] = time_serie['gap'].fillna(0)
    
    # Filter rows where gap is at least 1
    gap_serie = time_serie.loc[time_serie['gap'] >= 1, 'gap']
    print(f"Gap sizes: {gap_serie}")
    # Convert gaps to integers and return as a list
    gap_sizes = list(gap_serie.astype(int))
    
    return gap_sizes

def plot_gap_size(save_dir: Path, gaps: Union[list, pd.Series, pd.DataFrame], frequency: str = 'h') -> None:
    """
    Plot the distribution of gap sizes in a histogram, showing only bins with data.
    
    Parameters:
    -----------
    save_dir : Path
        Directory where the plot should be saved
    gaps : Union[list, pd.Series, pd.DataFrame]
        The gap sizes to plot
    frequency : str, default 'h'
        The frequency unit (e.g., 'h' for hours, 'm' for minutes)
    
    Returns:
    --------
    None
    """
    # Input validation
    if not gaps:
        raise ValueError("No gap data provided")
    
    # Convert gaps to list if it's a Series or DataFrame
    if isinstance(gaps, (pd.Series, pd.DataFrame)):
        gaps = gaps.values.tolist()
    gaps = [gap for gap in gaps if gap > 0 and gap <10]
    # Get unique values and their counts
    unique_gaps = np.unique(gaps)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create histogram with bins centered on unique values
    ax.hist(gaps, 
            bins=len(unique_gaps),  # One bin per unique value
            range=(min(unique_gaps) - 0.5, max(unique_gaps) + 0.5),  # Center bins on integers
            color='skyblue', 
            edgecolor='black', 
            linewidth=1.2,
            alpha=0.7)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set labels and title
    ax.set_title(f"Gap Size Distribution (in {frequency})", pad=20)
    ax.set_xlabel(f"Gap Size ({frequency})")
    ax.set_ylabel("Frequency")
    
    # Set x-axis ticks to show only the actual gap values
    ax.set_xticks(unique_gaps)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot if directory is provided
    if save_dir:
        # Create directory if it doesn't exist
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save figure with high DPI
        fig.savefig(save_dir / f'gap_size_{frequency}.png', 
                   dpi=300, 
                   bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    # Close the figure to free memory
    plt.close(fig)

