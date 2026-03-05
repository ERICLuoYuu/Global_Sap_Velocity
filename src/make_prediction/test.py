import ee
import geemap
import numpy as np

ee.Initialize(project='era5download-447713')

# Create a test region at different latitudes
def test_gee_dimensions(lat_center, scale=11132):
    """Test what dimensions GEE returns at different latitudes."""
    
    # 1 degree x 1 degree box
    region = ee.Geometry.Rectangle([
        0, lat_center - 0.5,  # lon_min, lat_min
        1, lat_center + 0.5   # lon_max, lat_max
    ], None, False)
    
    # Use a simple global dataset
    image = ee.Image('NASA/NASADEM_HGT/001').select('elevation')
    
    result = geemap.ee_to_numpy(image, region=region, scale=scale)
    if result.ndim > 2:
        result = result.squeeze()
    
    print(f"Latitude {lat_center}°: shape = {result.shape}")
    return result.shape

# Test at equator vs high latitude
test_gee_dimensions(0)    # Equator
test_gee_dimensions(30)   # Mid-latitude  
test_gee_dimensions(60)   # High latitude