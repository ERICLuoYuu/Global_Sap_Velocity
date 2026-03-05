"""
NetCDF File Explorer Utility

This script helps you understand the structure of your NetCDF files
so you can configure the extraction script correctly.
"""

import xarray as xr
import numpy as np
import sys
from pathlib import Path
# Add project root to path (adjust as needed)
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def explore_netcdf(file_path: str):
    """
    Explore and print detailed information about a NetCDF file.
    
    Parameters:
    -----------
    file_path : str
        Path to the NetCDF file
    """
    print("="*70)
    print(f"EXPLORING: {file_path}")
    print("="*70)
    
    try:
        ds = xr.open_dataset(file_path)
    except Exception as e:
        print(f"Error opening file: {e}")
        return
    
    # Basic info
    print("\n1. DIMENSIONS:")
    print("-"*40)
    for dim, size in ds.dims.items():
        print(f"   {dim}: {size}")
    
    print("\n2. COORDINATES:")
    print("-"*40)
    for coord in ds.coords:
        coord_data = ds.coords[coord]
        print(f"   {coord}:")
        print(f"      Shape: {coord_data.shape}")
        print(f"      Dtype: {coord_data.dtype}")
        if len(coord_data) > 0:
            print(f"      Range: {float(coord_data.min()):.4f} to {float(coord_data.max()):.4f}")
            if len(coord_data) > 1:
                resolution = np.abs(np.mean(np.diff(coord_data.values)))
                print(f"      Resolution: ~{resolution:.6f}")
    
    print("\n3. DATA VARIABLES:")
    print("-"*40)
    for var in ds.data_vars:
        var_data = ds[var]
        print(f"   {var}:")
        print(f"      Shape: {var_data.shape}")
        print(f"      Dtype: {var_data.dtype}")
        print(f"      Dimensions: {var_data.dims}")
        
        # Get valid data statistics
        try:
            values = var_data.values.flatten()
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 0:
                print(f"      Min: {np.min(valid_values):.2f}")
                print(f"      Max: {np.max(valid_values):.2f}")
                print(f"      Mean: {np.mean(valid_values):.2f}")
                print(f"      Valid pixels: {len(valid_values):,} / {len(values):,}")
        except:
            print(f"      (Could not compute statistics)")
        
        # Check attributes
        if var_data.attrs:
            print(f"      Attributes:")
            for attr, val in var_data.attrs.items():
                print(f"         {attr}: {val}")
    
    print("\n4. GLOBAL ATTRIBUTES:")
    print("-"*40)
    if ds.attrs:
        for attr, val in ds.attrs.items():
            val_str = str(val)[:80] + "..." if len(str(val)) > 80 else str(val)
            print(f"   {attr}: {val_str}")
    else:
        print("   (No global attributes)")
    
    print("\n5. SUGGESTED CONFIGURATION:")
    print("-"*40)
    
    # Detect coordinate names
    lat_candidates = ['lat', 'latitude', 'y', 'LAT', 'Latitude']
    lon_candidates = ['lon', 'longitude', 'x', 'LON', 'Longitude']
    
    lat_var = None
    lon_var = None
    
    for candidate in lat_candidates:
        if candidate in ds.coords or candidate in ds.dims:
            lat_var = candidate
            break
    
    for candidate in lon_candidates:
        if candidate in ds.coords or candidate in ds.dims:
            lon_var = candidate
            break
    
    # Detect age variable
    age_candidates = ['age', 'forest_age', 'stand_age', 'Age', 'FOREST_AGE', 
                      'mean', 'forest_age_mean', 'age_mean', 'b1']
    age_var = None
    
    for candidate in age_candidates:
        if candidate in ds.data_vars:
            age_var = candidate
            break
    
    if age_var is None and len(ds.data_vars) > 0:
        age_var = list(ds.data_vars)[0]
    
    print(f"   LAT_VAR = '{lat_var}'")
    print(f"   LON_VAR = '{lon_var}'")
    print(f"   AGE_VAR = '{age_var}'")
    
    print("\n" + "="*70)
    
    ds.close()


def quick_sample(file_path: str, lon: float, lat: float):
    """
    Quick test to extract a sample value at given coordinates.
    
    Parameters:
    -----------
    file_path : str
        Path to the NetCDF file
    lon : float
        Longitude
    lat : float
        Latitude
    """
    print(f"\nQuick sample extraction at ({lon}, {lat}):")
    print("-"*40)
    
    try:
        ds = xr.open_dataset(file_path)
        
        # Try common coordinate names
        lat_names = ['lat', 'latitude', 'y']
        lon_names = ['lon', 'longitude', 'x']
        
        lat_var = None
        lon_var = None
        
        for name in lat_names:
            if name in ds.coords:
                lat_var = name
                break
        
        for name in lon_names:
            if name in ds.coords:
                lon_var = name
                break
        
        if lat_var is None or lon_var is None:
            print("Could not identify coordinate variables")
            return
        
        # Extract from each data variable
        for var in ds.data_vars:
            try:
                value = ds[var].sel(
                    {lat_var: lat, lon_var: lon},
                    method='nearest'
                ).values
                
                if hasattr(value, 'item'):
                    value = value.item()
                
                print(f"   {var}: {value}")
            except Exception as e:
                print(f"   {var}: Error - {e}")
        
        ds.close()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python explore_netcdf.py <path_to_netcdf_file> [lon] [lat]")
        print("\nExamples:")
        print("  python explore_netcdf.py forest_age_2010.nc")
        print("  python explore_netcdf.py forest_age_2010.nc -122.5 45.5")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    explore_netcdf(file_path)
    
    # If coordinates provided, do a quick sample
    if len(sys.argv) >= 4:
        lon = float(sys.argv[2])
        lat = float(sys.argv[3])
        quick_sample(file_path, lon, lat)