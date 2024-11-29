import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union
import logging
from src.derive_climate_data.climate_data_calculator import ClimateDataCalculator
class ERA5DerivedProcessor:
    """
    Process ERA5-Land data to calculate derived variables and store them in structured folders
    """
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.calculator = ClimateDataCalculator()
        
        # Define input variables needed for calculations
        self.required_variables = {
            'temperature': '2m_temperature',
            'dewpoint': '2m_dewpoint_temperature',
            'radiation': 'surface_solar_radiation_downwards'
        }
        
        # Define output variable paths and their metadata
        self.derived_variables = {
            'relative_humidity': {
                'path': 'relative_humidity',
                'units': '%',
                'long_name': 'Relative humidity',
                'description': 'Calculated from 2m temperature and dewpoint'
            },
            'vapor_pressure_deficit': {
                'path': 'vapor_pressure_deficit',
                'units': 'kPa',
                'long_name': 'Vapor pressure deficit',
                'description': 'Calculated from temperature and relative humidity'
            },
            'ppfd': {
                'path': 'photosynthetic_photon_flux_density',
                'units': 'umol/m2/s',
                'long_name': 'Photosynthetic photon flux density',
                'description': 'Calculated from downward shortwave radiation'
            }
        }
        
        # Create output directories
        self._create_directory_structure()
        
        # Set up logging
        logging.basicConfig(
            filename=self.output_dir / 'derived_variables_log.txt',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )

    def _create_directory_structure(self) -> None:
        """Create the directory structure for derived variables"""
        for var_info in self.derived_variables.values():
            var_dir = self.output_dir / var_info['path']
            var_dir.mkdir(parents=True, exist_ok=True)

    def process_month(self, year: int, month: int) -> None:
        """Calculate and save derived variables for a specific month"""
        try:
            # Load required input data
            temp_file = (self.input_dir / self.required_variables['temperature'] / 
                        str(year) / f"{self.required_variables['temperature']}_{year}_{month:02d}_hourly.nc")
            dewp_file = (self.input_dir / self.required_variables['dewpoint'] / 
                        str(year) / f"{self.required_variables['dewpoint']}_{year}_{month:02d}_hourly.nc")
            rad_file = (self.input_dir / self.required_variables['radiation'] / 
                       str(year) / f"{self.required_variables['radiation']}_{year}_{month:02d}_hourly.nc")

            # Open and process datasets
            with xr.open_dataset(temp_file) as temp_ds, \
                 xr.open_dataset(dewp_file) as dewp_ds, \
                 xr.open_dataset(rad_file) as rad_ds:
                
                # Convert temperatures from K to °C
                temp_c = temp_ds['t2m'] - 273.15
                dewp_c = dewp_ds['d2m'] - 273.15
                
                # Calculate and save each derived variable separately
                self._calculate_and_save_rh(temp_c, dewp_c, year, month)
                self._calculate_and_save_vpd(temp_c, dewp_c, year, month)
                self._calculate_and_save_ppfd(rad_ds['ssrd'], year, month)
                
                logging.info(f"Successfully processed all variables for {year}-{month:02d}")
                
        except Exception as e:
            logging.error(f"Error processing {year}-{month:02d}: {str(e)}")

    def _calculate_and_save_rh(self, temp: xr.DataArray, dewp: xr.DataArray, 
                             year: int, month: int) -> None:
        """Calculate and save relative humidity"""
        var_info = self.derived_variables['relative_humidity']
        rh = self.calculator.calculate_rh(temp, dewp)
        
        # Create dataset
        ds = xr.Dataset({
            'relative_humidity': (temp.dims, rh, {
                'units': var_info['units'],
                'long_name': var_info['long_name'],
                'description': var_info['description']
            })
        }, coords=temp.coords)
        
        # Save file
        year_dir = self.output_dir / var_info['path'] / str(year)
        year_dir.mkdir(exist_ok=True)
        output_file = year_dir / f"{var_info['path']}_{year}_{month:02d}_hourly.nc"
        ds.to_netcdf(output_file)

    def _calculate_and_save_vpd(self, temp: xr.DataArray, dewp: xr.DataArray, year: int, month: int) -> None:
        """Calculate and save vapor pressure deficit"""
        var_info = self.derived_variables['vapor_pressure_deficit']
        rh = self.calculator.calculate_rh(temp, dewp)
        vpd = self.calculator.calculate_vpd(temp, rh)
        
        ds = xr.Dataset({
            'vapor_pressure_deficit': (temp.dims, vpd, {
                'units': var_info['units'],
                'long_name': var_info['long_name'],
                'description': var_info['description']
            })
        }, coords=temp.coords)
        
        year_dir = self.output_dir / var_info['path'] / str(year)
        year_dir.mkdir(exist_ok=True)
        output_file = year_dir / f"{var_info['path']}_{year}_{month:02d}_hourly.nc"
        ds.to_netcdf(output_file)

    def _calculate_and_save_ppfd(self, rad: xr.DataArray, year: int, month: int) -> None:
        """Calculate and save PPFD"""
        var_info = self.derived_variables['ppfd']
        ppfd = self.calculator.calculate_ppfd_from_radiation(rad)
        
        ds = xr.Dataset({
            'ppfd': (rad.dims, ppfd, {
                'units': var_info['units'],
                'long_name': var_info['long_name'],
                'description': var_info['description']
            })
        }, coords=rad.coords)
        
        year_dir = self.output_dir / var_info['path'] / str(year)
        year_dir.mkdir(exist_ok=True)
        output_file = year_dir / f"{var_info['path']}_{year}_{month:02d}_hourly.nc"
        ds.to_netcdf(output_file)

    def process_period(self, start_year: int, end_year: int) -> None:
        """Process multiple years of data"""
        for year in range(start_year, end_year + 1):
            logging.info(f"Processing year {year}")
            for month in range(1, 13):
                self.process_month(year, month)


processor = ERA5DerivedProcessor(
    input_dir='./data/raw/grided/era5land_vars_1995_2018',
    output_dir='./data/raw/grided/era5land_vars_1995_2018'
)

processor.process_period(1995, 2018)
