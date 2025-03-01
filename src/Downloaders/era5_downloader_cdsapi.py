import cdsapi
import os
from datetime import datetime
import time
import logging
from pathlib import Path
import calendar
from datetime import datetime
import calendar

class ERA5LandDownloader:
    def __init__(self, output_dir='era5land_data'):
        self.c = cdsapi.Client()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            filename=self.output_dir / 'download_log.txt',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        
        # Define variables mapping
        self.variables_mapping = {
            'temperature': '2m_temperature',  # Air temperature
            'humidity': '2m_dewpoint_temperature',  # To calculate relative humidity
            'wind': [
                '10m_u_component_of_wind',  # For wind speed
                '10m_v_component_of_wind'
            ],
            'radiation': [
                'surface_solar_radiation_downwards',  # Incoming shortwave
                'surface_net_solar_radiation',  # For net radiation
                'surface_net_thermal_radiation'
            ],
            'precipitation': 'total_precipitation',
            'soil_moisture': [
                'volumetric_soil_water_layer_1',  # Shallow soil water (0-7cm)
                'volumetric_soil_water_layer_4'   # Deep soil water (100-289cm)
            ]
        }

    def download_data(self, start_year, end_year):
        """
        Download hourly ERA5-Land data for all specified variables.
        """
        # Flatten the variables list
        variables_to_download = []
        for var_group in self.variables_mapping.values():
            if isinstance(var_group, list):
                variables_to_download.extend(var_group)
            else:
                variables_to_download.append(var_group)

        for variable in variables_to_download:
            var_dir = self.output_dir / variable
            var_dir.mkdir(exist_ok=True)
            
            for year in range(start_year, end_year + 1):
                year_dir = var_dir / str(year)
                year_dir.mkdir(exist_ok=True)
                
                for month in range(1, 13):
                    month_dir = year_dir / str(month)
                    month_dir.mkdir(exist_ok=True)
                    output_file = year_dir / str(month) / f"{variable}_{year}_{month:02d}_hourly.nc"
                    
                    if output_file.exists():
                        logging.info(f"File already exists: {output_file}")
                        continue
                    
                    try:
                        self._download_month_hourly(
                            variable=variable,
                            year=year,
                            month=month,
                            output_file=output_file
                        )
                        logging.info(f"Successfully downloaded: {output_file}")
                        time.sleep(5)  # Prevent overwhelming the server
                        
                    except Exception as e:
                        logging.error(f"Error downloading {variable} for {year}-{month}: {str(e)}")
                        time.sleep(60)  # Wait longer if there's an error

    def _download_month_hourly(self, variable, year, month, output_file):
        """Download hourly data for a specific month."""
        dataset = 'reanalysis-era5-land'
        request_params = {
            'product_type': ['reanalysis'],
            'download_format': 'unarchived',
            'format': 'netcdf',
            'variable': variable,
            'year': str(year),
            'month': f"{month:02d}",
            'day': [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)],
            'time': [f"{h:02d}:00" for h in range(24)],
        }
        
        self.c.retrieve(
            dataset,
            request_params,
            str(output_file)
        )

def main():
    # Create downloader instance
    downloader = ERA5LandDownloader(output_dir='./data/raw/grided/era5land_vars')
    
    try:
        # Start the download
        downloader.download_data(
            start_year=1995,
            end_year=2025
        )
    except KeyboardInterrupt:
        logging.info("Download interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()