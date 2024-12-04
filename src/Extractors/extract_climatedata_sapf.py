# The script is used to extract climate data for sap flow measured sites from ERA5-Land reanalysis data. The script is part of a larger project that aims to analyze the relationship between environmental variables and sap flow in trees. 
# The coordinates of the sites are stored in .csv files, the structure of the data is as follows: ./data/raw/0.1.5/0.1.5/csv/sapwood/f"{country_code}_{site_code}_{species_code}_site_md".csv, within the file, "si_lat" and "si_long" are the latitude and longitude of the site, respectively.
# The era5 data is stored in ./data/raw/grided/era5land_vars_1995_2018, the data is downloaded using the era5_downloader.py script. The script downloads hourly ERA5-Land data for all specified variables. The data is downloaded in .nc format and stored in directories based on the year and month of the data. the file structure follows the pattern: ./data/raw/grided/era5land_vars_1995_2018/{variable}/{year}/{month}/{variable}_{year}_{month:02d}_hourly.nc.
import pandas as pd
import xarray as xr
from pathlib import Path



class clmate_data_extractor:
    def __init__(self, sapflow_sites_dir, era5_data_dir):
        self.sapflow_sites_dir = sapflow_sites_dir
        self.era5_data_dir = era5_data_dir
    
    def extract_climate_data(self):
        # Load sapflow sites data
        sapflow_sites = self._load_sapflow_sites()
        
        # Load ERA5 data
        era5_data = self._load_era5_data()
        
        # Extract climate data for sapflow sites
        climate_data = self._extract_climate_data(sapflow_sites, era5_data)
        
        return climate_data
    
    def _load_sapflow_sites(self):
        sapflow_sites = {}
        for file in Path(self.sapflow_sites_dir).rglob("*site_md.csv"):
            site_data = pd.read_csv(file)
            country_code, site_code, species_code = file.stem.split("_")
            sapflow_sites[f"{country_code}_{site_code}_{species_code}"] = {
                "latitude": site_data["si_lat"].values[0],
                "longitude": site_data["si_long"].values[0]
            }
        
        for file in Path(self.sapflow_sites_dir).rglob("*sapf_data.csv"):
            site_sap_data = pd.read_csv(file)
            site_sap_data = pd.to_datetime(site_sap_data.dropna()["TIMESTAMP"])
            min_date = site_sap_data["TIMESTAMP"].min()
            max_date = site_sap_data["TIMESTAMP"].max()
            country_code, site_code, species_code = file.stem.split("_")
            sapflow_sites[f"{country_code}_{site_code}_{species_code}"].update({
                "time_range": {
                    "start": min_date,
                    "end": max_date
                                }
                                })
        return sapflow_sites
    
    def _load_era5_data(self):
        era5_data = {}
        for file in Path(self.era5_data_dir).rglob("*.nc"):
            variable, year, month = file.stem.split("_")
            data = xr.open_dataset(file)
            era5_data[f"{variable}_{year}_{month}"] = data
        return era5_data
    
    def _extract_climate_data(self, sapflow_sites, era5_data):
        climate_data = {}
        for site, site_info in sapflow_sites.items():
            site_lat = site_info["latitude"]
            site_lon = site_info["longitude"]
            start_date = site_info["time_range"]["start"]
            end_date = site_info["time_range"]["end"]
            
            # Initialize site level
            climate_data[site] = {}
            
            for data_key, data in era5_data.items():
                variable, year, month = data_key.split("_")
                
                # Skip data outside time range
                if year < start_date.year or year > end_date.year:
                    continue
                elif year == start_date.year and month < start_date.month:
                    continue
                elif year == end_date.year and month > end_date.month:
                    continue
                    
                # Initialize nested dictionaries if needed
                if variable not in climate_data[site]:
                    climate_data[site][variable] = {}
                if year not in climate_data[site][variable]:
                    climate_data[site][variable][year] = {}
                    
                # Extract and store data
                site_data = data.sel(latitude=site_lat, longitude=site_lon, method="nearest")
                climate_data[site][variable][year][month] = site_data.to_series()
            
            for site, site_data in climate_data.items():
                site_df = pd.DataFrame()
                for variable, var_data in site_data.items():
                    for year, year_data in var_data.items():
                        for month, month_data in year_data.items():
                            site_df[variable] = site_df[variable].concat(month_data)
                # Save site data to file
                site_file = Path(f"./data/processed/{site}_env_data.csv")
                site_df.to_csv(site_file)
        return climate_data

# Example usage
sapflow_sites_dir = "./data/raw/0.1.5/0.1.5/csv/sapwood"
era5_data_dir = "./data/raw/grided/era5land_vars_1995_2018"
extractor = clmate_data_extractor(sapflow_sites_dir, era5_data_dir)
climate_data = extractor.extract_climate_data()
print(climate_data)
