import ee
import datetime
import time
from typing import Dict, List, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Earth Engine
ee.Authenticate()
ee.Initialize()

class ERA5LandDownloader:
    def __init__(self):
        # Variable definitions
        self.variables = {
            'temperature': '2m_temperature',
            'humidity': '2m_dewpoint_temperature',
            'wind': [
                '10m_u_component_of_wind',
                '10m_v_component_of_wind'
            ],
            'radiation': [
                'surface_solar_radiation_downwards',
                'surface_net_solar_radiation',
                'surface_net_thermal_radiation'
            ],
            'precipitation': 'total_precipitation',
            'soil_moisture': [
                'volumetric_soil_water_layer_1',
                'volumetric_soil_water_layer_4'
            ]
        }
        
        # Define scale for different variable groups (in meters)
        self.scales = {
            'temperature': 11132,  # ERA5-Land native resolution
            'humidity': 11132,
            'wind': 11132,
            'radiation': 11132,
            'precipitation': 11132,
            'soil_moisture': 11132
        }

    def _get_all_bands(self) -> List[str]:
        """Flatten the variables dictionary into a list of band names."""
        bands = []
        for var_group in self.variables.values():
            if isinstance(var_group, list):
                bands.extend(var_group)
            else:
                bands.append(var_group)
        return bands

    def _create_export_task(
        self,
        image: ee.Image,
        description: str,
        folder: str,
        scale: int,
        region: ee.Geometry = None,
        file_format: str = 'GeoTIFF'
    ) -> ee.batch.Task:
        """Create an export task with error handling."""
        try:
            if region is None:
                region = ee.Geometry.Rectangle([-180, -90, 180, 90])
                
            task = ee.batch.Export.image.toDrive(
                image=image,
                description=description,
                folder=folder,
                scale=scale,
                region=region.getInfo()['coordinates'],
                maxPixels=1e13,  # Increased for global coverage
                fileFormat=file_format,
                formatOptions={
                    'cloudOptimized': True,
                    'compression': 'DEFLATE'
                }
            )
            return task
        except Exception as e:
            logging.error(f"Error creating export task {description}: {str(e)}")
            raise

    def download_by_variable_group(
        self,
        start_date: str,
        end_date: str,
        output_folder: str,
        variable_group: str,
        chunks_per_year: int = 12,  # Default to monthly chunks
        max_concurrent_tasks: int = 10
    ):
        """
        Download ERA5-Land data for a specific variable group in chunks.
        
        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        output_folder : str
            Name of the folder in Google Drive where files will be saved
        variable_group : str
            Name of the variable group to download
        chunks_per_year : int
            Number of time chunks per year (default: 12 for monthly)
        max_concurrent_tasks : int
            Maximum number of concurrent export tasks
        """
        # Get variables for the group
        vars_to_download = self.variables[variable_group]
        if not isinstance(vars_to_download, list):
            vars_to_download = [vars_to_download]

        # Get collection
        collection = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
        
        # Calculate time chunks
        start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        
        # Calculate chunk size in days
        days_per_chunk = 365 // chunks_per_year
        
        active_tasks = []
        
        current_date = start
        while current_date <= end:
            chunk_end = min(
                current_date + datetime.timedelta(days=days_per_chunk),
                end
            )
            
            # Wait if too many tasks are running
            while len(active_tasks) >= max_concurrent_tasks:
                active_tasks = [task for task in active_tasks 
                              if task.status()['state'] in ['READY', 'RUNNING']]
                time.sleep(60)  # Wait 1 minute before checking again
            
            try:
                # Filter collection for current chunk
                filtered = collection.filterDate(
                    current_date.strftime('%Y-%m-%d'),
                    chunk_end.strftime('%Y-%m-%d')
                ).select(vars_to_download)
                
                # Calculate means for the period
                means = filtered.mean()
                
                # Create description
                desc = f"ERA5Land_{variable_group}_{current_date.strftime('%Y%m')}"
                
                # Create and start export task
                task = self._create_export_task(
                    image=means,
                    description=desc,
                    folder=f"{output_folder}/{variable_group}",
                    scale=self.scales[variable_group]
                )
                
                task.start()
                active_tasks.append(task)
                
                logging.info(
                    f"Started export task for {variable_group} - "
                    f"{current_date.strftime('%Y-%m')}"
                )
                
            except Exception as e:
                logging.error(
                    f"Error processing chunk {current_date.strftime('%Y-%m')}: "
                    f"{str(e)}"
                )
                
            current_date = chunk_end + datetime.timedelta(days=1)

    def download_all_variables(
        self,
        start_date: str,
        end_date: str,
        output_folder: str
    ):
        """Download all variable groups."""
        for variable_group in self.variables.keys():
            logging.info(f"Starting download for {variable_group}")
            self.download_by_variable_group(
                start_date=start_date,
                end_date=end_date,
                output_folder=output_folder,
                variable_group=variable_group
            )
            
# Example usage
if __name__ == "__main__":
    # Initialize downloader
    downloader = ERA5LandDownloader()
    
    # Set date range (1995 to present)
    start_date = '1995-01-01'
    end_date = '2024-01-13'  # Current date
    
    # Start download
    downloader.download_all_variables(
        start_date=start_date,
        end_date=end_date,
        output_folder='ERA5_Land_Global'
    )