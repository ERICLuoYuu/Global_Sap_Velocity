
"""
Connects to CDS API and downloads data as xarray.
!! make sure to have .cdsapirc in your C:/<user>/<username> folder. It needs to contain the 
lines found at https://cds.climate.copernicus.eu/how-to-api after logging in!!

- Processes for each site and year individual requests
- saves .ncdata in a the <id>_ncdata subfolder
- concatenates the ncdata for all years to a csv into the <id>_csvdata subfolder

 LIST OF ERA5 meteo variables:
 
 	"10m_u_component_of_wind"
1	"10m_v_component_of_wind"
2	"2m_dewpoint_temperature"
3	"2m_temperature"
4	"evaporation_from_bare_soil"
5	"evaporation_from_open_water_surfaces_excluding_oceans"
6	"evaporation_from_the_top_of_canopy"
7	"evaporation_from_vegetation_transpiration"
8	"forecast_albedo"
9	"lake_bottom_temperature"
10	"lake_ice_depth"
11	"lake_ice_temperature"
12	"lake_mix_layer_depth"
13	"lake_mix_layer_temperature"
14	"lake_shape_factor"
15	"lake_total_layer_temperature"
16	"leaf_area_index_high_vegetation"
17	"leaf_area_index_low_vegetation"
18	"potential_evaporation"
19	"runoff"
20	"skin_reservoir_content"
21	"skin_temperature"
22	"snow_albedo"
23	"snow_cover"
24	"snow_density"
25	"snow_depth"
26	"snow_depth_water_equivalent"
27	"snow_evaporation"
28	"snowfall"
29	"snowmelt"
30	"soil_temperature_level_1"
31	"soil_temperature_level_2"
32	"soil_temperature_level_3"
33	"soil_temperature_level_4"
34	"sub_surface_runoff"
35	"surface_latent_heat_flux"
36	"surface_net_solar_radiation"
37	"surface_net_thermal_radiation"
38	"surface_pressure"
39	"surface_runoff"
40	"surface_sensible_heat_flux"
41	"surface_solar_radiation_downwards"
42	"surface_thermal_radiation_downwards"
43	"temperature_of_snow_layer"
44	"total_evaporation"
45	"total_precipitation"
46	"volumetric_soil_water_layer_1"
47	"volumetric_soil_water_layer_2"
48	"volumetric_soil_water_layer_3"
49	"volumetric_soil_water_layer_4"
"""


from dataclasses import dataclass
import os
import cdsapi
import pandas as pd
from pathlib import Path
import xarray as xr

# Initialize the CDS API client
c = cdsapi.Client()

# Quick helper class to define sites where to extract data
@dataclass
class Site:
    id: str
    lat: float
    lon: float
    years: list[str]
    variables : list[str]
    
    def __post_init__(self):
        """
        Just a check to fix if years are not given as strings
        """
        for year in self.years:
            if type(year) != str:
                year = str(year)
        self.ncdata_path = Path(f"./{self.id}_ncdata")
        self.csvdata_path = Path(f"./{self.id}_csvdata")

# Function to request and download data for a specific year
def download_ERA_data(id:str, lat:float, lon:float, year:int, variables:list[str])->None:
    print("------------")
    print(f"Processing {site.id}: Getting {variables} for year {year}...")
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': variables,
            'year': year,
            'month': [
                '01', '02', '03', '04', '05', '06',
                '07', '08', '09', '10', '11', '12',
            ],
            'day': [
                '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'
            ],
            'time': [
                '00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00',
                '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00',
                '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00'
            ],
            'format': 'netcdf',
            'area': [
                # extract with a buffer around the point
                lat + 0.25, lon - 0.25,
                lat - 0.25, lon + 0.25,
            ],
        },
        site.ncdata_path.joinpath(f"{id}_{year}_{str(variables)}_era5.nc")
    )

# Function to process the NetCDF file into concatenated csv
def make_csv_from_era_ncdata(sites:list[Site]) -> pd.DataFrame:
    for site in sites:
        files = [file for file in os.listdir() if site.id in file]
        dfs = []
        for file in files:
            ds = xr.open_dataset(file)
            df = ds.to_dataframe().reset_index()
            df = df[['time', site.variables]]
            df["ID"] = [site.id] * len(df)
        full_df = pd.concat(dfs)
        full_df.to_csv(site.csvdata_path)
    return full_df



# Define sites for which to extract data
sites = [
    Site(id="test1", lat = 46.826, lon = 6.1725, years = ['2020', '2021'], variables = ["2m_temperature"])
]

# Download data for each year
for site in sites:
    if not site.ncdata_path.exists():
        site.ncdata_path.mkdir(exist_ok=True, parents=True)
    if not site.csvdata_path.exists():
        site.csvdata_path.mkdir(exist_ok=True, parents=True)
    for year in site.years:
        download_ERA_data(site.id, site.lat, site.lon, year, site.variables)


# Combine data for all years
all_data = make_csv_from_era_ncdata(sites)

print('Data extraction and processing complete...')
