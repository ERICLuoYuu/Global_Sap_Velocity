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

# Aspect is read from ONE pixel, not averaged over a buffer, and the logic for
# that lives in src/Extractors/terrain_aspect.py so this extractor and the data
# migration that backfilled the corpus cannot drift apart. See that module for
# why a mean of aspect degrees is meaningless and why the scale matters.
from src.Extractors.terrain_aspect import (  # noqa: E402
    assert_dem_scale,
    native_scale,
    terrain_attributes_at_point,
    terrain_bands,
)

# Slope and aspect as one image, so a single request describes a single pixel.
terrain = terrain_bands(elevation)

# The DEM's own scale, kept server-side. A rounded copy is a DIFFERENT grid, not
# the same one: at FRA_FON, asking for 30.922081 m instead of the DEM's
# 30.922080776 m returns slope 6.61 deg / aspect 192.16 deg where the exact value
# gives 3.96 / 200.66. Earth Engine builds terrain products in the projection you
# ask for, so the scale is not a pixel selector -- it is the grid.
dem_scale = native_scale(elevation)

# Alarm if the asset has been regridded under the documented constant.
print(f"ASTER GDEM nominal scale: {assert_dem_scale(elevation):.9f} m")

# Function to standardize aspect values
def extract_terrain_attributes(lon, lat, buffer_size=250):
    """
    Extract terrain attributes (elevation, slope, aspect) at a given location

    elevation and slope are 250 m buffered MEANS: neither is circular, so a mean
    is a defensible summary of the stand around the site.

    ASPECT is not. It is circular, so an arithmetic mean of degrees is meaningless
    across the 0/360 seam -- pixels reading 355, 0 and 5 average to 120, and over
    the 226-site corpus that artefact put 106 sites in the "S" sector and none in
    "N". Averaging is also the wrong operation on its own terms: the median vector
    strength over these 250 m windows is 0.277, so most windows have no coherent
    direction to average. Aspect therefore comes from the single DEM pixel
    containing the site, sampled at the DEM's native scale, and the flat-terrain
    guard is judged on THAT pixel's slope -- reported as slope_deg_centre so the
    decision is auditable. See src/Extractors/terrain_aspect.py.

    Note that slope_deg and slope_deg_centre are not directly comparable:
    slope_deg is the buffered mean at scale=30, kept unchanged for continuity with
    the existing corpus, and 30 m is FINER than the DEM's 30.922081 m so that
    request resamples. That is the same defect class the aspect fix removes, left
    in place here because moving elevation_m and slope_deg is a separate change.

    Parameters:
    -----------
    lon : float
        Longitude in decimal degrees
    lat : float
        Latitude in decimal degrees
    buffer_size : float
        Buffer size in meters for spatial averaging of elevation and slope
        (default: 250m). Aspect ignores it.

    Returns:
    --------
    dict
        Dictionary containing all terrain attributes
    """
    try:
        # Buffered region for the non-circular means
        region = ee.Geometry.Point([lon, lat]).buffer(buffer_size)

        # Extract elevation
        elevation_result = elevation.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=30,  # ASTER GDEM is 30m resolution
            maxPixels=1e9
        )
        elev = elevation_result.get('b1').getInfo()

        # Extract slope
        slope_result = slope.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=30,
            maxPixels=1e9
        )
        slope_value = slope_result.get('slope').getInfo()

        # Extract aspect from the site's own pixel, with its own slope for the
        # flat-terrain guard, and standardize -- one shared code path.
        aspect_info = terrain_attributes_at_point(terrain, 'site', lon, lat, dem_scale)

        return {
            'elevation_m': elev,
            'slope_deg': slope_value,
            'slope_deg_centre': aspect_info['slope_deg_centre'],
            'aspect_deg': aspect_info['aspect_deg'],
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
            'slope_deg_centre': None,
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