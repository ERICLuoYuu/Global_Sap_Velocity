"""
Check which SAPFLUXNET sapwood-level sites are covered by ICOS stations.
"""
import os
import glob
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two points."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# Paths
sapwood_path = r'E:\OneDrive - Universität Münster\Sap_velocity_project\global-sap-velocity\data\raw\0.1.5\0.1.5\csv\sapwood'

# Get all site metadata files
site_md_files = glob.glob(os.path.join(sapwood_path, '*_site_md.csv'))
print(f'Found {len(site_md_files)} SAPFLUXNET sapwood-level sites')

# Extract coordinates from each
sites = []
for f in site_md_files:
    site_code = os.path.basename(f).replace('_site_md.csv', '')
    try:
        df = pd.read_csv(f)
        if 'si_lat' in df.columns and 'si_long' in df.columns:
            lat = df['si_lat'].iloc[0]
            lon = df['si_long'].iloc[0]
            if pd.notna(lat) and pd.notna(lon):
                sites.append({'site': site_code, 'lat': float(lat), 'lon': float(lon)})
    except Exception as e:
        pass

print(f'Extracted coordinates for {len(sites)} sites')

# Get ICOS stations
try:
    from icoscp.station import station
    icos_stations = station.getIdList()
    
    # Filter for ecosystem stations (flux towers)
    eco_stations = icos_stations[icos_stations['theme'] == 'ES']
    print(f'\nFound {len(eco_stations)} ICOS Ecosystem stations')
    
    # Get coordinates for each ICOS station
    icos_coords = []
    for idx, row in eco_stations.iterrows():
        try:
            st = station.get(row['id'])
            if hasattr(st, 'lat') and hasattr(st, 'lon'):
                icos_coords.append({
                    'id': row['id'],
                    'name': st.name if hasattr(st, 'name') else row['id'],
                    'lat': st.lat,
                    'lon': st.lon
                })
        except:
            pass
    
    print(f'Got coordinates for {len(icos_coords)} ICOS stations')
    
    # Find matches (within 5 km)
    matches = []
    for site in sites:
        for icos in icos_coords:
            dist = haversine(site['lat'], site['lon'], icos['lat'], icos['lon'])
            if dist < 5:  # Within 5 km
                matches.append({
                    'sapfluxnet_site': site['site'],
                    'sapfluxnet_lat': site['lat'],
                    'sapfluxnet_lon': site['lon'],
                    'icos_id': icos['id'],
                    'icos_name': icos['name'],
                    'icos_lat': icos['lat'],
                    'icos_lon': icos['lon'],
                    'distance_km': round(dist, 2)
                })
    
    print(f'\n=== MATCHES (within 5 km) ===')
    print(f'Found {len(matches)} SAPFLUXNET sites with nearby ICOS stations:\n')
    
    for m in sorted(matches, key=lambda x: x['distance_km']):
        print(f"{m['sapfluxnet_site']} <-> {m['icos_id']} ({m['icos_name']})")
        print(f"   Distance: {m['distance_km']} km")
        print(f"   SAPFLUXNET: ({m['sapfluxnet_lat']}, {m['sapfluxnet_lon']})")
        print(f"   ICOS: ({m['icos_lat']}, {m['icos_lon']})")
        print()
        
    # Save results
    if matches:
        df_matches = pd.DataFrame(matches)
        output_path = r'E:\OneDrive - Universität Münster\Sap_velocity_project\global-sap-velocity\Sapflow-internal\sapfluxnet_icos_matches.csv'
        df_matches.to_csv(output_path, index=False)
        print(f'Results saved to: {output_path}')
        
except ImportError:
    print('icoscp not installed')
except Exception as e:
    print(f'Error: {e}')
