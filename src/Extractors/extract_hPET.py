import numpy as np
import pandas as pd
from netCDF4 import Dataset
from datetime import datetime, timedelta

def extract_sites_from_csv(input_csv, t_resolution, datapath, output_csv):
    """
    Extract hPET values for multiple sites from a CSV file and save results to CSV.
    
    :param input_csv: Path to CSV file with columns: start_date, end_date, site_name, lat, lon
    :param t_resolution: 'daily' or 'hourly'
    :param datapath: Path where the global hPET netCDF files are stored
    :param output_csv: Path for output CSV file
    """
    
    # Read the sites CSV
    sites_df = pd.read_csv(input_csv)
    
    # Validate required columns
    required_cols = ['start_date', 'end_date', 'site_name', 'lat', 'lon']
    if not all(col in sites_df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    # Convert dates to datetime
    sites_df['start_date'] = pd.to_datetime(sites_df['start_date'])
    sites_df['end_date'] = pd.to_datetime(sites_df['end_date'])
    
    # List to store all results
    all_results = []
    
    # Process each site
    for idx, site in sites_df.iterrows():
        print(f"Processing site {idx+1}/{len(sites_df)}: {site['site_name']}")
        
        site_data = extract_site_timeseries(
            site['lat'], 
            site['lon'],
            site['start_date'],
            site['end_date'],
            site['site_name'],
            t_resolution,
            datapath
        )
        
        all_results.extend(site_data)
    
    # Create output DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save to CSV
    results_df.to_csv(output_csv, index=False)
    print(f"\nData extraction complete! Results saved to: {output_csv}")
    print(f"Total records: {len(results_df)}")
    
    return results_df


def extract_site_timeseries(lat, lon, start_date, end_date, site_name, t_resolution, datapath):
    """
    Extract hPET time series for a single site location.
    
    :param lat: Latitude of the site
    :param lon: Longitude of the site
    :param start_date: Start date (datetime object)
    :param end_date: End date (datetime object)
    :param site_name: Name of the site
    :param t_resolution: 'daily' or 'hourly'
    :param datapath: Path to netCDF files
    :return: List of dictionaries with extracted data
    """
    
    if t_resolution == 'daily':
        fname_suffix = '_daily_pet.nc'
    elif t_resolution == 'hourly':
        fname_suffix = '_hourly_pet.nc'
    else:
        raise ValueError("t_resolution must be 'daily' or 'hourly'")
    
    site_results = []
    
    # Get year range
    start_year = start_date.year
    end_year = end_date.year
    
    # Reference date for time axis (from original script)
    reference_date = datetime(1981, 1, 1)
    
    # Process each year
    for year in range(start_year, end_year + 1):
        try:
            # Open netCDF file for the year
            nc_file = Dataset(datapath + str(year) + fname_suffix, 'r')
            
            # Get lat/lon arrays
            lats = nc_file.variables['latitude'][:]
            lons = nc_file.variables['longitude'][:]
            
            # Find nearest grid point
            lat_idx, lon_idx = nearest_point(lat, lon, lats, lons)
            
            # Get time variable
            time_var = nc_file.variables['time'][:]
            
            # Convert time to dates to find the indices we need
            if t_resolution == 'daily':
                dates = [reference_date + timedelta(days=int(t)) for t in time_var]
            else:  # hourly
                dates = [reference_date + timedelta(hours=int(t)) for t in time_var]
            
            # Determine the time range for this year
            year_start = datetime(year, 1, 1)
            year_end = datetime(year, 12, 31, 23, 59, 59)
            
            # Adjust boundaries for the actual requested period
            if year == start_year:
                year_start = start_date
            if year == end_year:
                year_end = end_date
            
            # Find indices that fall within the requested date range
            time_indices = [i for i, d in enumerate(dates) if year_start <= d <= year_end]
            
            if not time_indices:
                nc_file.close()
                continue
            
            # Extract only the necessary time steps (more efficient)
            start_idx = min(time_indices)
            end_idx = max(time_indices) + 1
            
            # Extract PET data only for the required time range
            pet_values = nc_file.variables['pet'][start_idx:end_idx, lat_idx, lon_idx]
            selected_dates = dates[start_idx:end_idx]
            
            # Store results
            for date, pet_val in zip(selected_dates, pet_values):
                if start_date <= date <= end_date:
                    site_results.append({
                        'site_name': site_name,
                        'date': date.strftime('%Y-%m-%d %H:%M:%S') if t_resolution == 'hourly' else date.strftime('%Y-%m-%d'),
                        'lat': lat,
                        'lon': lon,
                        'grid_lat': float(lats[lat_idx]),
                        'grid_lon': float(lons[lon_idx]),
                        'hPET': float(pet_val) if not np.isnan(pet_val) else None
                    })
            
            nc_file.close()
            
        except FileNotFoundError:
            print(f"Warning: File not found for year {year} for site {site_name}")
            continue
        except Exception as e:
            print(f"Error processing year {year} for site {site_name}: {str(e)}")
            continue
    
    return site_results


def nearest_point(lat_var, lon_var, lats, lons):
    """
    Find the nearest grid location index for a specific lat-lon point.
    
    :param lat_var: Target latitude
    :param lon_var: Target longitude
    :param lats: Array of available latitudes
    :param lons: Array of available longitudes
    :return: lat_index, lon_index
    """
    # Handle longitude convention (0-360 vs -180-180)
    if any(lons > 180.0) and (lon_var < 0.0):
        lon_var = lon_var + 360.0
    
    lat = lats
    lon = lons
    
    # Handle 2D arrays
    if lat.ndim == 2:
        lat = lat[:, 0]
    if lon.ndim == 2:
        lon = lon[0, :]
    
    # Find nearest latitude index
    lat_diffs = np.abs(lat - lat_var)
    index_lat = np.argmin(lat_diffs)
    
    # Find nearest longitude index
    lon_diffs = np.abs(lon - lon_var)
    index_lon = np.argmin(lon_diffs)
    
    return index_lat, index_lon


def main(input_csv, output_csv, t_resolution='hourly', datapath=None):
    """
    Main function to run the hPET extraction.
    
    :param input_csv: Path to input CSV with site information
    :param output_csv: Path for output CSV with hPET values
    :param t_resolution: 'daily' or 'hourly' (default: 'hourly')
    :param datapath: Path to netCDF files (if None, uses default paths)
    """
    
    # Set default datapath if not provided
    if datapath is None:
        if t_resolution == 'daily':
            datapath = '/bp1store/geog-tropical/data/ERA-Land/driving_data/global_pet/daily_pet/'
        elif t_resolution == 'hourly':
            datapath = '/bp1store/geog-tropical/data/ERA-Land/driving_data/global_pet/hourly_pet/'
        else:
            raise ValueError("t_resolution must be 'daily' or 'hourly'")
    
    # Extract data
    results_df = extract_sites_from_csv(input_csv, t_resolution, datapath, output_csv)
    
    # Print summary statistics
    print("\n=== Summary ===")
    print(f"Number of sites: {results_df['site_name'].nunique()}")
    print(f"Date range: {results_df['date'].min()} to {results_df['date'].max()}")
    print(f"Total data points: {len(results_df)}")
    if results_df['hPET'].isna().sum() > 0:
        print(f"Missing values: {results_df['hPET'].isna().sum()}")
    
    return results_df


if __name__ == '__main__':
    # ===== CONFIGURATION =====
    # Update these paths according to your setup

    input_csv = 'data/raw/0.1.5/0.1.5/csv/plant/site_info.csv'  # Your input CSV file
    output_csv = 'data/raw/hPET_extracted_data.csv'  # Output file name
    t_resolution = 'hourly'  # 'daily' or 'hourly'
    
    # Optional: specify custom datapath (leave as None to use defaults)
    datapath = None
    # Or specify custom path:
    # datapath = '/your/custom/path/to/data/'
    
    # ===== RUN EXTRACTION =====
    results = main(input_csv, output_csv, t_resolution, datapath)
    
    print("\n✓ Extraction complete!")
##### ---------------------------######
