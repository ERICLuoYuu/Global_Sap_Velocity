import os
import sys
import ee
import pandas as pd
import numpy as np
from pathlib import Path
# Initialize Earth Engine
try:
    ee.Initialize(project='era5download-447713')
    print("Google Earth Engine initialized successfully!")
except Exception as e:
    print("Error initializing Google Earth Engine:", e)
    raise
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(sys.path)
from path_config import PathConfig, get_default_paths
# Read your input CSV file
paths = get_default_paths()
input_csv = paths.site_info_path
df = pd.read_csv(input_csv)

# Load the Facebook/Meta canopy height dataset
canopy_height = ee.ImageCollection("projects/sat-io/open-datasets/facebook/meta-canopy-height").mosaic()

# Load elevation dataset - using SRTM
elevation = ee.Image("projects/sat-io/open-datasets/ASTER/GDEM")


# Calculate slope and aspect from elevation
slope = ee.Terrain.slope(elevation)
aspect = ee.Terrain.aspect(elevation)

# Function to standardize aspect values
def standardize_aspect(aspect_value, slope_value):
    """
    Standardize aspect values to handle the circular nature of aspect data.
    Account for flat terrain (slope = 0) where aspect is undefined.
    
    Parameters:
    -----------
    aspect_value : float or None
        Raw aspect value in degrees (0-360)
    slope_value : float or None
        Slope value in degrees
        
    Returns:
    --------
    dict
        Dictionary containing standardized_aspect, aspect_category, sin_aspect, cos_aspect
    """
    # For flat terrain (slope = 0 or very close to 0), aspect is undefined
    if slope_value is None or np.isnan(slope_value) or abs(slope_value) < 0.1:
        return {
            'aspect_standardized': None,
            'aspect_category': None,
            'aspect_sin': 0.0,
            'aspect_cos': 0.0
        }
    
    if aspect_value is None or np.isnan(aspect_value):
        return {
            'aspect_standardized': None,
            'aspect_category': None,
            'aspect_sin': None,
            'aspect_cos': None
        }
    
    # Ensure aspect is in 0-359.99 range (360 becomes 0)
    standardized_aspect = aspect_value % 360
    
    # Calculate the sine and cosine components for circular analysis
    sin_aspect = np.sin(np.radians(aspect_value))
    cos_aspect = np.cos(np.radians(aspect_value))
    
    # Convert to compass direction categories
    if standardized_aspect >= 337.5 or standardized_aspect < 22.5:
        category = "N"
    elif standardized_aspect >= 22.5 and standardized_aspect < 67.5:
        category = "NE"
    elif standardized_aspect >= 67.5 and standardized_aspect < 112.5:
        category = "E"
    elif standardized_aspect >= 112.5 and standardized_aspect < 157.5:
        category = "SE"
    elif standardized_aspect >= 157.5 and standardized_aspect < 202.5:
        category = "S"
    elif standardized_aspect >= 202.5 and standardized_aspect < 247.5:
        category = "SW"
    elif standardized_aspect >= 247.5 and standardized_aspect < 292.5:
        category = "W"
    elif standardized_aspect >= 292.5 and standardized_aspect < 337.5:
        category = "NW"
    else:
        category = None
    
    return {
        'aspect_standardized': standardized_aspect,
        'aspect_category': category,
        'aspect_sin': sin_aspect,
        'aspect_cos': cos_aspect
    }


def extract_terrain_attributes(lon, lat, buffer_size=250):
    """
    Extract terrain attributes (elevation, slope, aspect) at a given location
    
    Parameters:
    -----------
    lon : float
        Longitude in decimal degrees
    lat : float
        Latitude in decimal degrees
    buffer_size : float
        Buffer size in meters for spatial averaging (default: 250m)
    
    Returns:
    --------
    dict
        Dictionary containing all terrain attributes
    """
    try:
        # Create a point geometry with buffer
        point = ee.Geometry.Point([lon, lat]).buffer(buffer_size)

        # Extract elevation
        elevation_result = elevation.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=30,  # ASTER GDEM is 30m resolution
            maxPixels=1e9
        )
        elev = elevation_result.get('b1').getInfo()
        
        # Extract slope
        slope_result = slope.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=30,
            maxPixels=1e9
        )
        slope_value = slope_result.get('slope').getInfo()
        
        # Extract aspect
        aspect_result = aspect.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=30,
            maxPixels=1e9
        )
        aspect_value = aspect_result.get('aspect').getInfo()
        
        # Standardize aspect
        aspect_info = standardize_aspect(aspect_value, slope_value)
        
        return {
            'elevation_m': elev,
            'slope_deg': slope_value,
            'aspect_deg': aspect_value,
            'aspect_standardized': aspect_info['aspect_standardized'],
            'aspect_category': aspect_info['aspect_category'],
            'aspect_sin': aspect_info['aspect_sin'],
            'aspect_cos': aspect_info['aspect_cos']
        }
    
    except Exception as e:
        print(f"Error extracting terrain attributes for ({lon}, {lat}): {e}")
        return {
            'elevation_m': None,
            'slope_deg': None,
            'aspect_deg': None,
            'aspect_standardized': None,
            'aspect_category': None,
            'aspect_sin': None,
            'aspect_cos': None
        }


def extract_canopy_height(lon, lat, buffer_size=250):
    """
    Extract canopy height value at a given longitude and latitude
    
    Parameters:
    -----------
    lon : float
        Longitude in decimal degrees
    lat : float
        Latitude in decimal degrees
    buffer_size : float
        Buffer size in meters for spatial averaging (default: 250m)
    
    Returns:
    --------
    float or None
        Canopy height in meters
    """
    try:
        # Create a point geometry with buffer
        point = ee.Geometry.Point([lon, lat]).buffer(buffer_size)

        # Extract the canopy height value at this point
        result = canopy_height.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=1.2,
            maxPixels=1e9
        )
        
        # Get the value (the band name is 'cover_code')
        height = result.get('cover_code').getInfo()
        
        return height
    
    except Exception as e:
        print(f"Error extracting canopy height for ({lon}, {lat}): {e}")
        return None


# Extract all attributes for each site
print("Extracting canopy heights and terrain attributes...")
results = []

for idx, row in df.iterrows():
    site_name = row['site_name']
    lon = row['lon']
    lat = row['lat']
    
    print(f"Processing {idx+1}/{len(df)}: {site_name}")
    
    # Extract canopy height
    height = extract_canopy_height(lon, lat)
    
    # Extract terrain attributes
    terrain = extract_terrain_attributes(lon, lat)
    
    # Combine results
    site_result = {
        'canopy_height_m': height,
        **terrain  # Unpacks all terrain attributes
    }
    
    results.append(site_result)

# Convert results to dataframe and merge with original data
results_df = pd.DataFrame(results)
df_final = pd.concat([df, results_df], axis=1)

# Save to a new CSV file
output_csv = paths.terrain_attributes_data_path
if not os.path.exists(os.path.dirname(output_csv)):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df_final.to_csv(output_csv, index=False)

print(f"\nDone! Results saved to {output_csv}")
print(f"Successfully extracted data for {len(df_final)} sites")

# Display summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print("\nCanopy Height Statistics:")
print(df_final['canopy_height_m'].describe())

print("\nElevation Statistics:")
print(df_final['elevation_m'].describe())

print("\nSlope Statistics:")
print(df_final['slope_deg'].describe())

print("\nAspect Statistics:")
print(df_final['aspect_deg'].describe())

print("\nAspect Category Distribution:")
print(df_final['aspect_category'].value_counts(dropna=False))

print("\n" + "="*60)
print(f"Missing values:")
print(f"  Canopy Height: {df_final['canopy_height_m'].isna().sum()} sites")
print(f"  Elevation: {df_final['elevation_m'].isna().sum()} sites")
print(f"  Slope: {df_final['slope_deg'].isna().sum()} sites")
print(f"  Aspect: {df_final['aspect_deg'].isna().sum()} sites")
print("="*60)