"""
Check which SAPFLUXNET sites are covered by ICOS stations
"""
import os
import glob
import pandas as pd
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance in km between two points"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * 6371 * asin(sqrt(a))

# Get SAPFLUXNET sites
plant_path = r'E:\OneDrive - Universität Münster\Sap_velocity_project\global-sap-velocity\data\raw\0.1.5\0.1.5\csv\plant'
site_files = glob.glob(os.path.join(plant_path, '*_site_md.csv'))
print(f'Found {len(site_files)} SAPFLUXNET site metadata files')

# Extract coordinates
sapfluxnet_sites = []
for f in site_files:
    try:
        df = pd.read_csv(f)
        code = os.path.basename(f).replace('_site_md.csv', '')
        if 'si_lat' in df.columns and 'si_long' in df.columns:
            lat = float(df['si_lat'].iloc[0])
            lon = float(df['si_long'].iloc[0])
            name = str(df['si_name'].iloc[0]) if 'si_name' in df.columns else ''
            sapfluxnet_sites.append({'code': code, 'name': name, 'lat': lat, 'lon': lon})
    except Exception as e:
        print(f"Error reading {f}: {e}")

print(f'Extracted {len(sapfluxnet_sites)} SAPFLUXNET sites with coordinates')

# Get ICOS stations
try:
    from icoscp.station import station
    icos_stations = station.getIdList(project='all')
    print(f'\nFound {len(icos_stations)} ICOS stations')
    
    # Get coordinates for each ICOS station
    icos_coords = []
    for icos_id in icos_stations:
        try:
            s = station.get(icos_id)
            if s and hasattr(s, 'lat') and hasattr(s, 'lon'):
                icos_coords.append({
                    'id': icos_id,
                    'name': s.name if hasattr(s, 'name') else '',
                    'lat': float(s.lat),
                    'lon': float(s.lon)
                })
        except:
            pass
    
    print(f'Got coordinates for {len(icos_coords)} ICOS stations')
    
    # Find matches (within 10km)
    threshold_km = 10
    matches = []
    
    for sap in sapfluxnet_sites:
        for icos in icos_coords:
            try:
                dist = haversine(sap['lon'], sap['lat'], icos['lon'], icos['lat'])
                if dist < threshold_km:
                    matches.append({
                        'sapfluxnet_code': sap['code'],
                        'sapfluxnet_name': sap['name'],
                        'icos_id': icos['id'],
                        'icos_name': icos['name'],
                        'distance_km': round(dist, 2)
                    })
            except:
                pass
    
    print(f'\n=== SAPFLUXNET sites with ICOS stations within {threshold_km}km ===')
    if matches:
        for m in sorted(matches, key=lambda x: x['distance_km']):
            print(f"  {m['sapfluxnet_code']} -> {m['icos_id']} ({m['distance_km']} km)")
            print(f"    SAPFLUXNET: {m['sapfluxnet_name']}")
            print(f"    ICOS: {m['icos_name']}")
    else:
        print("  No matches found")
        
except ImportError:
    print("icoscp not installed")
except Exception as e:
    print(f"Error: {e}")
