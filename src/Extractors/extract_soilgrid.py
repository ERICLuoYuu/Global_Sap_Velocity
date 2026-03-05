"""
Extract SoilGrids Soil Properties at Site Locations
(With Nearest Neighbor Fallback)

This script extracts soil properties from SoilGrids 2.0 (ISRIC).
 IMPROVEMENT: If a site has no data (e.g., water/urban), it automatically 
 searches the surrounding area (up to 1km) for the nearest valid pixel.
"""

import os
import sys
import time
import math
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from path_config import PathConfig, get_default_paths
    USE_PATH_CONFIG = True
except ImportError:
    USE_PATH_CONFIG = False
    print("Note: path_config not found. Using manual paths.")


class SoilGridsExtractor:
    """
    Extract soil properties from SoilGrids 2.0 using the REST API.
    Includes spatial search for nearest valid data point.
    """
    
    API_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    
    # SoilGrids stores values as integers, these factors convert to standard units
    PROPERTIES = {
        'bdod': {'name': 'Bulk density', 'unit': 'g/cm³', 'conversion': 0.01},
        'cec': {'name': 'Cation exchange capacity', 'unit': 'mmol(c)/kg', 'conversion': 1.0},
        'cfvo': {'name': 'Coarse fragments', 'unit': '%', 'conversion': 0.1},
        'clay': {'name': 'Clay content', 'unit': '%', 'conversion': 0.1},
        'sand': {'name': 'Sand content', 'unit': '%', 'conversion': 0.1},
        'silt': {'name': 'Silt content', 'unit': '%', 'conversion': 0.1},
        'nitrogen': {'name': 'Total nitrogen', 'unit': 'g/kg', 'conversion': 0.01},
        'phh2o': {'name': 'pH (H2O)', 'unit': 'pH', 'conversion': 0.1},
        'soc': {'name': 'Soil organic carbon', 'unit': 'g/kg', 'conversion': 0.1},
        'ocd': {'name': 'Organic carbon density', 'unit': 'kg/m³', 'conversion': 0.1},
    }
    
    DEPTHS = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
    
    def __init__(self, 
                 properties: List[str] = None,
                 depths: List[str] = None,
                 timeout: int = 30,
                 max_retries: int = 3,
                 delay_between_requests: float = 0.5,
                 search_radius_km: float = 1.0):
        """
        Initialize the SoilGrids extractor.
        
        Parameters:
        -----------
        search_radius_km : float
            Radius to search for valid data if exact point is missing (default 1km)
        """
        self.properties = properties or ['clay', 'sand', 'silt', 'bdod', 'soc', 'phh2o']
        self.depths = depths or ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
        self.timeout = timeout
        self.max_retries = max_retries
        self.delay = delay_between_requests
        self.search_radius_km = search_radius_km
        
        # Validation checks
        invalid = set(self.properties) - set(self.PROPERTIES.keys())
        if invalid: raise ValueError(f"Invalid properties: {invalid}")
        
        print(f"SoilGrids Extractor initialized:")
        print(f"  Properties: {self.properties}")
        print(f"  Search Radius: {self.search_radius_km} km")
    
    def _calculate_destination_point(self, lat: float, lon: float, distance_m: float, bearing_deg: float) -> Tuple[float, float]:
        """
        Calculate new coordinates given a starting point, distance (m) and bearing (degrees).
        Using simple trigonometry suitable for short distances (<10km).
        """
        R = 6378137.0  # Earth Radius in meters
        
        # Convert to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        bearing_rad = math.radians(bearing_deg)
        
        # Calculate new lat/lon
        new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(distance_m / R) +
                                math.cos(lat_rad) * math.sin(distance_m / R) * math.cos(bearing_rad))
        
        new_lon_rad = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance_m / R) * math.cos(lat_rad),
                                           math.cos(distance_m / R) - math.sin(lat_rad) * math.sin(new_lat_rad))
        
        return math.degrees(new_lon_rad), math.degrees(new_lat_rad)

    def _make_request(self, lon: float, lat: float) -> Optional[Dict]:
        """Make API request to SoilGrids."""
        params = {
            'lon': lon, 
            'lat': lat, 
            'property': self.properties, 
            'depth': self.depths, 
            'value': 'mean'
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.API_URL, params=params, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    return None # Explicit no data
                elif response.status_code == 429:
                    time.sleep((attempt + 1) * 2)
            except requests.exceptions.RequestException:
                pass # Retry
            
            if attempt < self.max_retries - 1:
                time.sleep(self.delay)
        return None
    
    def _check_data_validity(self, data: Dict) -> bool:
        """Check if the API response actually contains non-null values."""
        if not data or 'properties' not in data:
            return False
            
        # Check if we have at least one non-null value in the returned data
        for prop_data in data['properties']['layers']:
            for depth_data in prop_data['depths']:
                if depth_data['values'].get('mean') is not None:
                    return True
        return False

    def _parse_response(self, data: Dict, meta: Dict) -> Dict[str, Optional[float]]:
        """Parse API response and flatten into a dictionary."""
        results = {}
        
        # Add metadata (where the data came from)
        results['data_source'] = meta.get('source', 'unknown')
        results['offset_distance_m'] = meta.get('distance', 0)
        results['snapped_lat'] = meta.get('lat', 0)
        results['snapped_lon'] = meta.get('lon', 0)
        
        # Pre-fill with None
        for prop in self.properties:
            for depth in self.depths:
                col = f"{prop}_{depth.replace('-', '_').replace('cm', '')}"
                results[col] = None

        if not data or 'properties' not in data:
            return results
        
        for prop_data in data['properties']['layers']:
            prop_name = prop_data['name']
            conversion = self.PROPERTIES.get(prop_name, {}).get('conversion', 1.0)
            for depth_data in prop_data['depths']:
                label = depth_data['label']
                val = depth_data['values'].get('mean')
                if val is not None:
                    col = f"{prop_name}_{label.replace('-', '_').replace('cm', '')}"
                    results[col] = val * conversion
        return results
    
    def extract_at_point(self, lon: float, lat: float) -> Dict[str, Optional[float]]:
        """
        Extract soil properties. 
        If no data at exact point, searches 1km radius in steps.
        """
        # 1. Try Exact Location
        response = self._make_request(lon, lat)
        
        if self._check_data_validity(response):
            return self._parse_response(response, {
                'source': 'exact', 'distance': 0, 'lat': lat, 'lon': lon
            })
            
        # 2. Nearest Neighbor Search
        # SoilGrids resolution is ~250m. We search in rings of 250m.
        max_dist_m = int(self.search_radius_km * 1000)
        step_size_m = 250 
        
        # Directions to check: N, NE, E, SE, S, SW, W, NW
        bearings = [0, 45, 90, 135, 180, 225, 270, 315]
        
        print(f"    > No data at ({lon:.4f}, {lat:.4f}). Searching {self.search_radius_km}km radius...")
        
        for dist in range(step_size_m, max_dist_m + 1, step_size_m):
            for bearing in bearings:
                # Calculate offset coordinates
                search_lon, search_lat = self._calculate_destination_point(lat, lon, dist, bearing)
                
                # Request
                response = self._make_request(search_lon, search_lat)
                
                if self._check_data_validity(response):
                    print(f"    > Found valid data at {dist}m (Bearing {bearing}°)")
                    return self._parse_response(response, {
                        'source': 'nearest_neighbor', 
                        'distance': dist, 
                        'lat': search_lat, 
                        'lon': search_lon
                    })
                
                # Tiny sleep to avoid hammering the API too hard
                time.sleep(0.1)
        
        # 3. Give up
        print(f"    > No data found within {self.search_radius_km}km.")
        return self._parse_response(None, {
            'source': 'no_data', 'distance': -1, 'lat': lat, 'lon': lon
        })
    
    def extract_for_sites(self, df: pd.DataFrame, lon_col='lon', lat_col='lat', site_col='site_name') -> pd.DataFrame:
        """Sequential extraction (safer for complex search logic)."""
        print(f"\nExtracting soil properties for {len(df)} sites...")
        results = []
        
        for idx, row in df.iterrows():
            site = row.get(site_col, f"Site_{idx}")
            lon, lat = row[lon_col], row[lat_col]
            
            if (idx + 1) % 5 == 0 or idx == 0:
                print(f"  Processing {idx + 1}/{len(df)}: {site}")
                
            result = self.extract_at_point(lon, lat)
            results.append(result)
            time.sleep(self.delay)
            
        results_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), results_df], axis=1)


def calculate_derived_properties(df: pd.DataFrame, depth: str = '0_5') -> pd.DataFrame:
    """Calculate texture classes if data exists."""
    clay_col, sand_col, silt_col = f'clay_{depth}', f'sand_{depth}', f'silt_{depth}'
    
    if all(col in df.columns for col in [clay_col, sand_col, silt_col]):
        def classify(row):
            c, s, si = row[clay_col], row[sand_col], row[silt_col]
            if pd.isna(c) or pd.isna(s) or pd.isna(si): return None
            # Simplified USDA
            if c >= 40: return 'Clay'
            elif s >= 85: return 'Sand'
            elif si >= 80: return 'Silt'
            elif c >= 27 and s <= 52: return 'Clay Loam'
            elif c >= 20 and s >= 45: return 'Sandy Clay Loam'
            elif si >= 50: return 'Silt Loam'
            elif s >= 70: return 'Sandy Loam'
            else: return 'Loam'
        
        df[f'texture_class_{depth}'] = df.apply(classify, axis=1)
    return df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    if USE_PATH_CONFIG:
        paths = get_default_paths()
        input_csv = paths.site_info_path
        output_csv = os.path.join(os.path.dirname(paths.terrain_attributes_data_path), "soilgrids_data.csv")
    else:
        # Manual paths
        input_csv = r".\data\raw\0.1.5\0.1.5\csv\sapwood\site_info.csv"
        output_csv = r".\data\processed\soilgrids_data.csv"
    
    PROPERTIES = [
        'clay', 'sand', 'silt',  # Texture -> Hydraulic Conductivity
        'bdod',                  # Bulk Density -> Porosity
        'cfvo',                  # Coarse Fragments -> Correcting available water volume
        'soc',                   # Organic Carbon -> Water retention
        'phh2o',                 # pH -> General root health
        'nitrogen'               # N -> Leaf Area proxy
    ]
    DEPTHS = ['0-5cm', '5-15cm', '15-30cm', '30-60cm', '60-100cm', '100-200cm']
    
    # Search Radius Configuration (1km)
    SEARCH_RADIUS_KM = 1.0 
    
    # --- RUN ---
    print(f"Reading input CSV: {input_csv}")
    if not os.path.exists(input_csv):
        print("Error: Input file not found.")
        sys.exit(1)
        
    df = pd.read_csv(input_csv)
    
    extractor = SoilGridsExtractor(
        properties=PROPERTIES, 
        depths=DEPTHS, 
        delay_between_requests=0.3,
        search_radius_km=SEARCH_RADIUS_KM
    )
    
    df_final = extractor.extract_for_sites(df, lon_col='lon', lat_col='lat', site_col='site_name')
    df_final = calculate_derived_properties(df_final, depth='0_5')
    
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    df_final.to_csv(output_csv, index=False)
    
    # --- SUMMARY ---
    print("\n" + "="*50)
    print("EXTRACTION SUMMARY")
    print("="*50)
    
    if 'data_source' in df_final.columns:
        counts = df_final['data_source'].value_counts()
        print("\nData Sources used:")
        for source, count in counts.items():
            print(f"  - {source}: {count} sites")
            
    print(f"\nResults saved to: {output_csv}")