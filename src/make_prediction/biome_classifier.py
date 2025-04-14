"""
Module for classifying biomes based on climate data
"""

import os
import sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.sample import sample_gen
from pathlib import Path
import logging
from scipy.spatial import cKDTree
import pickle

# Import from config
from config import (
    CLIMATE_DATA_DIR, MEAN_ANNUAL_TEMP_FILE, MEAN_ANNUAL_PRECIP_FILE,
    BIOME_CLASSIFICATION, ALL_POSSIBLE_BIOME_TYPES
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("biome_classifier.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class BiomeClassifier:
    """Class for classifying biomes based on climate data"""
    
    def __init__(self, temp_file=MEAN_ANNUAL_TEMP_FILE, precip_file=MEAN_ANNUAL_PRECIP_FILE, 
                 classification=BIOME_CLASSIFICATION, cache_file=None):
        """
        Initialize the biome classifier
        
        Parameters:
        -----------
        temp_file : Path
            Path to the mean annual temperature raster file
        precip_file : Path
            Path to the mean annual precipitation raster file
        classification : dict
            Dictionary mapping biome names to temperature and precipitation ranges
        cache_file : Path, optional
            Path to a cache file to speed up repeated lookups
        """
        self.temp_file = Path(temp_file)
        self.precip_file = Path(precip_file)
        self.classification = classification
        self.cache_file = cache_file
        
        # Check if files exist
        if not self.temp_file.exists():
            logger.error(f"Temperature file not found: {self.temp_file}")
            raise FileNotFoundError(f"Temperature file not found: {self.temp_file}")
            
        if not self.precip_file.exists():
            logger.error(f"Precipitation file not found: {self.precip_file}")
            raise FileNotFoundError(f"Precipitation file not found: {self.precip_file}")
            
        # Initialize cache
        self.climate_data_cache = {}
        self.kdtree = None
        self.climate_points = None
        
        # Load data and build spatial index
        self._load_data()
        
    def _load_data(self):
        """Load climate data and build spatial index"""
        logger.info("Loading climate data...")
        
        # Check if cache file exists
        if self.cache_file and Path(self.cache_file).exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.climate_data_cache = cache_data.get('cache', {})
                    self.kdtree = cache_data.get('kdtree')
                    self.climate_points = cache_data.get('points')
                    
                if self.kdtree is not None and self.climate_points is not None:
                    logger.info(f"Loaded climate data cache from {self.cache_file}")
                    return
            except Exception as e:
                logger.warning(f"Error loading cache file: {e}")
        
        # Load temperature data
        with rasterio.open(self.temp_file) as src_temp:
            temp_profile = src_temp.profile
            temp_data = src_temp.read(1)
            temp_transform = src_temp.transform
            temp_nodata = src_temp.nodata
            
            # Get coordinates for each pixel
            height, width = temp_data.shape
            rows, cols = np.mgrid[0:height, 0:width]
            
            # Transform pixel coordinates to geographic coordinates
            x, y = rasterio.transform.xy(temp_transform, rows.flatten(), cols.flatten())
            lons = np.array(x).reshape(height, width)
            lats = np.array(y).reshape(height, width)
        
        # Load precipitation data
        with rasterio.open(self.precip_file) as src_precip:
            precip_profile = src_precip.profile
            precip_data = src_precip.read(1)
            precip_transform = src_precip.transform
            precip_nodata = src_precip.nodata
            
            # Check if the two rasters have the same dimensions and transform
            if temp_profile['width'] != precip_profile['width'] or temp_profile['height'] != precip_profile['height']:
                logger.error("Temperature and precipitation rasters have different dimensions")
                raise ValueError("Temperature and precipitation rasters have different dimensions")
                
            if temp_transform != precip_transform:
                logger.error("Temperature and precipitation rasters have different transforms")
                raise ValueError("Temperature and precipitation rasters have different transforms")
        
        # Create points for KDTree (using only valid data points)
        valid_mask = ((temp_data != temp_nodata) & (precip_data != precip_nodata) & 
                      ~np.isnan(temp_data) & ~np.isnan(precip_data))
        
        valid_lons = lons[valid_mask]
        valid_lats = lats[valid_mask]
        valid_temps = temp_data[valid_mask]
        valid_precips = precip_data[valid_mask]
        
        # Create array of points for KDTree
        self.climate_points = np.column_stack([valid_lons, valid_lats, valid_temps, valid_precips])
        
        # Build KDTree for efficient spatial queries
        self.kdtree = cKDTree(self.climate_points[:, :2])  # Use only lon/lat for spatial indexing
        
        logger.info(f"Loaded climate data: {len(self.climate_points)} valid points")
        
        # Save to cache if cache file is specified
        if self.cache_file:
            try:
                cache_data = {
                    'cache': self.climate_data_cache,
                    'kdtree': self.kdtree,
                    'points': self.climate_points
                }
                
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                    
                logger.info(f"Saved climate data cache to {self.cache_file}")
            except Exception as e:
                logger.warning(f"Error saving cache file: {e}")
    
    def get_climate_at_location(self, lon, lat):
        """
        Get climate data (temperature and precipitation) at a specific location
        
        Parameters:
        -----------
        lon : float
            Longitude
        lat : float
            Latitude
            
        Returns:
        --------
        climate : dict
            Dictionary with 'temp' and 'precip' keys
        """
        # Check cache first
        cache_key = f"{lon:.5f}_{lat:.5f}"
        if cache_key in self.climate_data_cache:
            return self.climate_data_cache[cache_key]
        
        # Find nearest point in KDTree
        if self.kdtree is not None:
            distance, index = self.kdtree.query([lon, lat], k=1)
            if distance < 0.1:  # Within ~10km at the equator
                temp = self.climate_points[index, 2]
                precip = self.climate_points[index, 3]
                
                climate = {'temp': temp, 'precip': precip}
                
                # Cache the result
                self.climate_data_cache[cache_key] = climate
                
                return climate
            else:
                logger.warning(f"No climate data found within 0.1 degrees of ({lon}, {lat})")
        
        # Fallback: sample from rasters directly
        try:
            with rasterio.open(self.temp_file) as src_temp:
                temp_val = next(sample_gen(src_temp, [(lon, lat)], indexes=[1]))[0]
                
            with rasterio.open(self.precip_file) as src_precip:
                precip_val = next(sample_gen(src_precip, [(lon, lat)], indexes=[1]))[0]
                
            climate = {'temp': temp_val, 'precip': precip_val}
            
            # Cache the result
            self.climate_data_cache[cache_key] = climate
            
            return climate
        except Exception as e:
            logger.error(f"Error sampling climate data at ({lon}, {lat}): {e}")
            
            # Return a default climate if sampling fails
            return {'temp': 15.0, 'precip': 1000.0}  # Default to temperate forest
    
    def classify_biome(self, lon, lat):
        """
        Classify the biome at a specific location based on climate data
        
        Parameters:
        -----------
        lon : float
            Longitude
        lat : float
            Latitude
            
        Returns:
        --------
        biome : str
            Biome classification
        """
        # Get climate data at location
        climate = self.get_climate_at_location(lon, lat)
        
        if climate is None:
            logger.warning(f"No climate data available at ({lon}, {lat})")
            return "Unknown"
            
        temp = climate['temp']
        precip = climate['precip']
        
        # Classify biome based on temperature and precipitation
        for biome_name, thresholds in self.classification.items():
            temp_min, temp_max = thresholds['temp_range']
            precip_min, precip_max = thresholds['precip_range']
            
            if (temp_min <= temp <= temp_max) and (precip_min <= precip <= precip_max):
                return biome_name
        
        # If no biome matches the criteria, find the closest match
        best_biome = "Unknown"
        min_distance = float('inf')
        
        for biome_name, thresholds in self.classification.items():
            temp_min, temp_max = thresholds['temp_range']
            precip_min, precip_max = thresholds['precip_range']
            
            # Calculate distance to the biome's climate range
            temp_center = (temp_min + temp_max) / 2
            precip_center = (precip_min + precip_max) / 2
            
            # Normalize distances (temperature and precipitation have different scales)
            temp_range = max([abs(t_range[1] - t_range[0]) for t_range in 
                           [thresholds['temp_range'] for thresholds in self.classification.values()]])
            precip_range = max([abs(p_range[1] - p_range[0]) for p_range in 
                             [thresholds['precip_range'] for thresholds in self.classification.values()]])
            
            temp_distance = abs(temp - temp_center) / temp_range
            precip_distance = abs(precip - precip_center) / precip_range
            
            # Combined distance (Euclidean in normalized space)
            distance = np.sqrt(temp_distance**2 + precip_distance**2)
            
            if distance < min_distance:
                min_distance = distance
                best_biome = biome_name
        
        return best_biome


# Global instance for convenience
_classifier = None

def get_classifier(cache_file='biome_classifier_cache.pkl'):
    """Get or create a global BiomeClassifier instance"""
    global _classifier
    
    if _classifier is None:
        cache_path = Path(cache_file)
        _classifier = BiomeClassifier(cache_file=cache_path)
        
    return _classifier


def classify_biome(lon, lat):
    """
    Classify the biome at a specific location
    
    This is a convenience function that uses the global classifier instance.
    
    Parameters:
    -----------
    lon : float
        Longitude
    lat : float
        Latitude
        
    Returns:
    --------
    biome : str
        Biome classification
    """
    classifier = get_classifier()
    return classifier.classify_biome(lon, lat)


def get_climate_at_location(lon, lat):
    """
    Get climate data at a specific location
    
    This is a convenience function that uses the global classifier instance.
    
    Parameters:
    -----------
    lon : float
        Longitude
    lat : float
        Latitude
        
    Returns:
    --------
    climate : dict
        Dictionary with 'temp' and 'precip' keys
    """
    classifier = get_classifier()
    return classifier.get_climate_at_location(lon, lat)


if __name__ == "__main__":
    """Command-line interface for testing biome classification"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Classify biomes based on climate data')
    parser.add_argument('--lon', type=float, required=True, help='Longitude')
    parser.add_argument('--lat', type=float, required=True, help='Latitude')
    
    args = parser.parse_args()
    
    # Get climate data and classify biome
    climate = get_climate_at_location(args.lon, args.lat)
    biome = classify_biome(args.lon, args.lat)
    
    print(f"Location: ({args.lon}, {args.lat})")
    print(f"Climate: Temperature = {climate['temp']:.1f}°C, Precipitation = {climate['precip']:.1f}mm")
    print(f"Biome: {biome}")