import os
import pandas as pd


from datetime import datetime
from tqdm import tqdm
import numpy as np
from pathlib import Path
import sys
import os
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(sys.path)
from path_config import PathConfig, get_default_paths
# check current sys.path
import rioxarray 
import xarray as xr
def parse_date_from_filename(filename):
    """
    Parses the date from a GLOBMAP LAI V3 filename.
    """
    try:
        base_name = os.path.basename(filename)
        parts = base_name.split('.')
        date_part = parts[1]
        if date_part.startswith('A') and len(date_part) == 8:
            year_doy_str = date_part[1:]
            year = int(year_doy_str[:4])
            doy = int(year_doy_str[4:])
            return datetime.strptime(f'{year}-{doy}', '%Y-%j')
        else:
            return None
    except (IndexError, ValueError):
        return None

def get_dimension_names(rds):
    """
    Get dimension names from a DataArray, handling both dict-like and tuple returns.
    """
    dims = rds.dims
    if hasattr(dims, 'keys'):
        # dims is a dict-like object (Mapping)
        return list(dims.keys())
    else:
        # dims is already a tuple or list
        return list(dims)

def has_valid_georeferencing(rds):
    """
    Check if a DataArray has valid georeferencing information.
    """
    try:
        # Check if CRS exists
        if rds.rio.crs is None:
            return False
        
        # Check if spatial dimensions exist and have proper coordinates
        if hasattr(rds, 'x') and hasattr(rds, 'y'):
            # Check if coordinates are not just sequential integers
            x_coords = rds.x.values
            y_coords = rds.y.values
            
            # Valid geographic coordinates should be in reasonable ranges
            if (x_coords.min() >= -180 and x_coords.max() <= 180 and
                y_coords.min() >= -90 and y_coords.max() <= 90):
                return True
        
        return False
    except:
        return False

def add_georeferencing(rds):
    """
    Manually adds georeferencing information to a raw GLOBMAP DataArray.
    This information is taken from the dataset's official metadata.
    Only applies if georeferencing is missing or invalid.
    """
    # Metadata from GLOBMAP V3 Description:
    # Spatial Coverage: 180ºW~180ºE, 63ºS~90ºN
    # Spatial Resolution: 0.0727273º
    # Projection: Geographic (which means EPSG:4326)
    
    resolution = 0.0727273
    
    # 1. Get the actual dimension names and rename them to 'y' and 'x'
    dim_names = get_dimension_names(rds)
    if len(dim_names) != 2:
        raise ValueError(f"Expected 2 dimensions, but found {len(dim_names)}: {dim_names}")
    
    # Rename: first dimension is latitude (y), second is longitude (x)
    rds = rds.rename({dim_names[0]: 'y', dim_names[1]: 'x'})

    # 2. Get the number of pixels in each dimension from the data's shape.
    num_lats, num_lons = rds.shape
    
    # Validate dimensions match expected values (approximately)
    expected_lons = int(360 / resolution)
    expected_lats = int(153 / resolution)  # From 90°N to 63°S = 153°
    
    if abs(num_lons - expected_lons) > 5 or abs(num_lats - expected_lats) > 5:
        print(f"Warning: Dimension mismatch. Expected ~{expected_lats}x{expected_lons}, got {num_lats}x{num_lons}")

    # 3. Create the coordinate arrays using pixel-center coordinates
    # Longitude (x) goes from West to East (-180 to +180)
    lon_start = -180 + resolution / 2
    lon_stop = 180 - resolution / 2
    lons = np.linspace(start=lon_start, stop=lon_stop, num=num_lons)
    
    # Latitude (y) goes from North to South (90 to -63)
    lat_start = 90 - resolution / 2
    lat_stop = -63 + resolution / 2
    lats = np.linspace(start=lat_start, stop=lat_stop, num=num_lats)

    # 4. Assign these arrays as the coordinates for the x and y dimensions.
    rds = rds.assign_coords(x=lons, y=lats)

    # 5. Set the Coordinate Reference System (CRS) to WGS 84 (EPSG:4326).
    rds = rds.rio.write_crs("EPSG:4326", inplace=True)

    return rds

def ensure_spatial_dimensions(rds, filepath):
    """
    Ensure the DataArray has proper spatial dimensions (x, y).
    Works for both GeoTIFF (which may use different names) and HDF files.
    """
    # Common dimension name variations
    lat_names = ['y', 'lat', 'latitude', 'Y', 'Lat', 'Latitude']
    lon_names = ['x', 'lon', 'longitude', 'X', 'Lon', 'Longitude']
    
    current_dims = get_dimension_names(rds)
    
    # Find which dimension is which
    lat_dim = None
    lon_dim = None
    
    for dim in current_dims:
        if dim in lat_names:
            lat_dim = dim
        elif dim in lon_names:
            lon_dim = dim
    
    # If we found standard names, rename to 'y' and 'x'
    if lat_dim and lon_dim:
        if lat_dim != 'y' or lon_dim != 'x':
            rds = rds.rename({lat_dim: 'y', lon_dim: 'x'})
    else:
        # Fallback: assume first dimension is y (lat), second is x (lon)
        if len(current_dims) >= 2:
            rds = rds.rename({current_dims[0]: 'y', current_dims[1]: 'x'})
    
    return rds


def extract_lai_from_files(sites_csv_path, data_dir_path):
    """
    Extracts LAI time-series for multiple sites from local GLOBMAP HDF or GeoTIFF files.
    """
    print("Step 1: Loading site information...")
    try:
        sites_df = pd.read_csv(sites_csv_path)
        sites_df['start_date'] = pd.to_datetime(sites_df['start_date'])
        sites_df['end_date'] = pd.to_datetime(sites_df['end_date'])
    except FileNotFoundError:
        print(f"Error: Site info file not found at '{sites_csv_path}'")
        return pd.DataFrame()
    print(f"Found {len(sites_df)} sites to process.")

    print("\nStep 2: Discovering and parsing data files...")
    valid_extensions = ('.hdf', '.tif', '.tiff')
    all_files = [f for f in os.listdir(data_dir_path) if f.lower().endswith(valid_extensions)]
    if not all_files:
        print(f"Error: No HDF or TIFF files found in '{data_dir_path}'")
        return pd.DataFrame()
    file_info = [{'datetime': parse_date_from_filename(f), 'filepath': os.path.join(data_dir_path, f)} 
                 for f in all_files if parse_date_from_filename(f)]
    files_df = pd.DataFrame(file_info).sort_values('datetime').reset_index(drop=True)
    if files_df.empty:
        print("Error: Could not parse dates from any filenames. Check file naming convention.")
        return pd.DataFrame()
    print(f"Found and parsed {len(files_df)} data files, ranging from {files_df['datetime'].min().date()} to {files_df['datetime'].max().date()}.")

    print("\nStep 3: Extracting LAI for each site...")
    all_site_data = []

    # Get this name from the inspect_hdf.py script if needed
    SUBDATASET_NAME = "long_term_lai_8km:LAI" 
    
    for _, site in tqdm(sites_df.iterrows(), total=len(sites_df), desc="Processing sites"):
        site_name = site['site_name']
        lat, lon = site['lat'], site['lon']
        start_date, end_date = site['start_date'], site['end_date']
        
        # Remove timezone info if present
        if hasattr(start_date, 'tz_localize'):
            start_date = start_date.tz_localize(None) if start_date.tz is not None else start_date
            end_date = end_date.tz_localize(None) if end_date.tz is not None else end_date

        site_files_df = files_df[(files_df['datetime'] >= (start_date - pd.Timedelta(days=16))) & (files_df['datetime'] <= (end_date + pd.Timedelta(days=16)))]

        site_lai_records = []
        for _, file_row in site_files_df.iterrows():
            filepath = file_row['filepath']
            is_geotiff = filepath.lower().endswith(('.tif', '.tiff'))
            
            try:
                if filepath.lower().endswith('.hdf'):
                    connection_string = f'HDF4_EOS:EOS_GRID:"{filepath}":{SUBDATASET_NAME}'
                    rds_raw = rioxarray.open_rasterio(connection_string, masked=True)
                else:  # GeoTIFF
                    rds_raw = rioxarray.open_rasterio(filepath, masked=True)

                # Handle potential multiple bands or unexpected dimensions
                if 'band' in get_dimension_names(rds_raw):
                    if len(rds_raw.band) == 1:
                        rds_squeezed = rds_raw.squeeze('band', drop=True)
                    else:
                        print(f"\nWarning: Multiple bands found in {filepath}. Using first band.")
                        rds_squeezed = rds_raw.isel(band=0).squeeze(drop=True)
                else:
                    rds_squeezed = rds_raw.squeeze(drop=True)

                # Ensure proper spatial dimension names
                rds_squeezed = ensure_spatial_dimensions(rds_squeezed, filepath)

                # For GeoTIFF files: check if they already have valid georeferencing
                # For HDF files: always add manual georeferencing
                if is_geotiff and has_valid_georeferencing(rds_squeezed):
                    # GeoTIFF already has proper georeferencing, use it as-is
                    rds_georeferenced = rds_squeezed
                else:
                    # Either HDF file or GeoTIFF without proper georeferencing
                    # Apply manual georeferencing
                    rds_georeferenced = add_georeferencing(rds_squeezed)

                # Extract pixel value at the site location
                pixel_value = rds_georeferenced.sel(x=lon, y=lat, method='nearest').item()
                
                if pd.notna(pixel_value):
                    lai_value = pixel_value * 0.01
                    site_lai_records.append({
                        'site_name': site_name,
                        'datetime': file_row['datetime'],
                        'LAI': lai_value
                    })
            except Exception as e:
                print(f"\nWarning: Could not process file '{filepath}' for site '{site_name}'. Error: {e}")
        
        if site_lai_records:
            all_site_data.extend(site_lai_records)

    if not all_site_data:
        print("\nWarning: No LAI data could be extracted for the given sites and date ranges.")
        return pd.DataFrame()

    print("\nStep 4: Finalizing results...")
    final_df = pd.DataFrame(all_site_data)
    return final_df

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # --- Configuration ---
    paths = get_default_paths()
    # Path to your CSV file with site information
    SITES_FILE = paths.site_info_path
    # Path to the folder where you unzipped all the HDF or TIFF files
    DATA_DIRECTORY = paths.globmap_lai_root

    # --- Pre-run checks ---
    if not os.path.exists(SITES_FILE):
        print(f"FATAL ERROR: The sites file '{SITES_FILE}' was not found.")
        exit()
    if not os.path.exists(DATA_DIRECTORY):
        print(f"FATAL ERROR: The data directory '{DATA_DIRECTORY}' was not found.")
        exit()

    # --- Run the extraction process ---
    lai_results_df = extract_lai_from_files(SITES_FILE, DATA_DIRECTORY)

    # --- Display and Save Results ---
    if not lai_results_df.empty:
        print("\nExtraction Complete! Here is a sample of the data:")
        print(lai_results_df.head())
        
        print(f"\nTotal LAI records extracted: {len(lai_results_df)}")
        
        # Optional: Save the final DataFrame to a new CSV file
        output_filename = paths.globmap_lai_data_path
        # check if output directory exists, if not create it
        output_dir = os.path.dirname(output_filename)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        lai_results_df.to_csv(output_filename, index=False)
        print(f"\nResults have been saved to '{output_filename}'")