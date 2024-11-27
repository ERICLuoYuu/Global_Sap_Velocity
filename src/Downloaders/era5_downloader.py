import cdsapi
import os
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path

class ERA5LandHourlyDownloader:
    def __init__(self, output_dir='era5land_hourly_data'):
        self.c = cdsapi.Client()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            filename=self.output_dir / 'download_log.txt',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        
    def download_data(self, variables, start_year, end_year):
        """
        Download hourly ERA5-Land data for specified variables and time period.
        
        Args:
            variables (list): List of variables to download
            start_year (int): Starting year
            end_year (int): Ending year
        """
        hours = [f"{h:02d}:00" for h in range(24)]
        
        for variable in variables:
            var_dir = self.output_dir / variable
            var_dir.mkdir(exist_ok=True)
            
            for year in range(start_year, end_year + 1):
                year_dir = var_dir / str(year)
                year_dir.mkdir(exist_ok=True)
                
                for month in range(1, 13):
                    output_file = year_dir / f"{variable}_{year}_{month:02d}_hourly.nc"
                    
                    if output_file.exists():
                        logging.info(f"File already exists: {output_file}")
                        continue
                    
                    try:
                        self._download_month_hourly(
                            variable=variable,
                            year=year,
                            month=month,
                            hours=hours,
                            output_file=output_file
                        )
                        logging.info(f"Successfully downloaded: {output_file}")
                        
                        # Prevent overwhelming the server
                        time.sleep(5)
                        
                    except Exception as e:
                        logging.error(f"Error downloading {variable} for {year}-{month}: {str(e)}")
                        # Wait longer if there's an error
                        time.sleep(60)
                        
    def _download_month_hourly(self, variable, year, month, hours, output_file):
        """Download hourly data for a specific month."""
        request_params = {
            'format': 'netcdf',
            'variable': variable,
            'year': str(year),
            'month': f"{month:02d}",
            'day': [f"{d:02d}" for d in range(1, 32)],
            'time': hours,
            'area': [90, -180, -90, 180],  # Global coverage: North, West, South, East
        }
        
        self.c.retrieve(
            'reanalysis-era5-land',
            request_params,
            str(output_file)
        )