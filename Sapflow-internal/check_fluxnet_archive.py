#!/usr/bin/env python3
"""Check FLUXNET Archive availability for sites with older sapflow data."""

import requests

sparql_url = 'https://meta.icos-cp.eu/sparql'

# Query for Fluxnet products - simpler query
query = """
PREFIX cpmeta: <http://meta.icos-cp.eu/ontologies/cpmeta/>
PREFIX prov: <http://www.w3.org/ns/prov#>
SELECT ?station ?spec ?timeStart ?timeEnd WHERE {
  ?dobj cpmeta:hasObjectSpec ?specUri .
  ?specUri rdfs:label ?spec .
  ?dobj cpmeta:hasStartTime ?timeStart .
  ?dobj cpmeta:hasEndTime ?timeEnd .
  ?dobj cpmeta:wasAcquiredBy/prov:wasAssociatedWith ?stationUri .
  ?stationUri cpmeta:hasStationId ?station .
  FILTER(?station = "ES-LMa")
  FILTER(CONTAINS(LCASE(?spec), "fluxnet"))
}
LIMIT 20
"""

print("Querying ICOS for Fluxnet products...")
response = requests.post(sparql_url, data={'query': query}, headers={'Accept': 'application/json'})

if response.ok:
    data = response.json()
    results = data['results']['bindings']
    print(f"Found {len(results)} Fluxnet products:\n")
    
    for r in results:
        station = r.get('station', {}).get('value', 'N/A')
        spec = r.get('spec', {}).get('value', 'N/A')
        earliest = r.get('earliest', {}).get('value', 'N/A')[:10]
        latest = r.get('latest', {}).get('value', 'N/A')[:10]
        print(f"{station}: {spec}")
        print(f"  Time range: {earliest} to {latest}\n")
else:
    print(f"Error: {response.status_code}")
    print(response.text[:500])
