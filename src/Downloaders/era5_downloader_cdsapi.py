import cdsapi
import os
from datetime import datetime
import time
import logging
from pathlib import Path
import calendar
import multiprocessing as mp
from functools import partial
import random
import traceback

class ERA5LandDownloader:
    def __init__(self, output_dir='era5land_data', max_workers=4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        
        # Configure logging with lock to prevent conflicts in multiprocessing
        self.log_file = self.output_dir / 'download_log.txt'
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(processName)s - %(message)s'
        )
        
        # Define variables mapping
        self.variables_mapping = {
            
            
            'wind': [
                '10m_u_component_of_wind',  # For wind speed
                '10m_v_component_of_wind'
            ],
           
        }
        {
            'temperature': '2m_temperature',  # Air temperature
            'dewpoint_temperature': '2m_dewpoint_temperature',  # To calculate relative humidity
            'wind': [
                '10m_u_component_of_wind',  # For wind speed
                '10m_v_component_of_wind'
            ],
            'radiation': [
                'surface_solar_radiation_downwards',  # Incoming shortwave
            ],
        }

    def download_data(self, start_year, end_year):
        """
        Download hourly ERA5-Land data for all specified variables using multiprocessing.
        """
        # Flatten the variables list
        variables_to_download = []
        for var_group in self.variables_mapping.values():
            if isinstance(var_group, list):
                variables_to_download.extend(var_group)
            else:
                variables_to_download.append(var_group)

        # Create a list of all download tasks
        download_tasks = []
        
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
                    
                    # Add task to the list
                    download_tasks.append((variable, year, month, output_file))
        
        # Shuffle tasks to avoid hitting the same month/year/variable pattern consecutively
        random.shuffle(download_tasks)
        
        logging.info(f"Starting download of {len(download_tasks)} tasks with {self.max_workers} workers")
        
        # Create a pool of workers and map the download function to the tasks
        with mp.Pool(processes=self.max_workers) as pool:
            pool.map(self._download_task_wrapper, download_tasks)
            
        logging.info("All downloads completed")

    def _download_task_wrapper(self, task):
        """Wrapper function for download task to handle exceptions in worker processes"""
        variable, year, month, output_file = task
        process_name = mp.current_process().name
        
        try:
            logging.info(f"Process {process_name}: Starting download of {variable} for {year}-{month:02d}")
            
            # Create a new client for each process
            c = cdsapi.Client()
            
            dataset = 'reanalysis-era5-land'
            request_params = {
                'format': 'netcdf',
                'variable': variable,
                'year': str(year),
                'month': f"{month:02d}",
                'day': [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)],
                'time': [f"{h:02d}:00" for h in range(24)],
            }
            
            # Add jitter to prevent all processes from requesting simultaneously
            wait_time = random.uniform(1, 5)
            time.sleep(wait_time)
            
            # Download with retries
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    c.retrieve(
                        dataset,
                        request_params,
                        str(output_file)
                    )
                    logging.info(f"Process {process_name}: Successfully downloaded {variable} for {year}-{month:02d}")
                    # Success, break out of retry loop
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        retry_wait = (2 ** attempt) * 60 + random.uniform(1, 30)
                        logging.warning(f"Process {process_name}: Attempt {attempt+1} failed for {variable} {year}-{month:02d}: {str(e)}. Retrying in {retry_wait:.1f} seconds.")
                        time.sleep(retry_wait)
                    else:
                        # Last attempt failed
                        raise
                        
        except Exception as e:
            logging.error(f"Process {process_name}: Failed to download {variable} for {year}-{month:02d}: {str(e)}")
            logging.error(traceback.format_exc())
            
            # Create a marker file for failed downloads to retry later
            failed_marker = Path(str(output_file) + ".failed")
            with open(failed_marker, 'w') as f:
                f.write(f"Failed: {str(e)}")

def retry_failed_downloads(download_dir):
    """Utility function to retry previously failed downloads"""
    failed_files = list(Path(download_dir).glob("**/*.failed"))
    
    if not failed_files:
        print("No failed downloads to retry")
        return
        
    print(f"Found {len(failed_files)} failed downloads to retry")
    
    downloader = ERA5LandDownloader(output_dir=download_dir)
    
    for failed_file in failed_files:
        # Parse filename to get variable, year, month
        nc_file = failed_file.with_suffix('')
        parts = nc_file.name.split('_')
        
        if len(parts) >= 3:
            variable = parts[0]
            year = int(parts[1])
            month = int(parts[2].split('_')[0])
            
            print(f"Retrying download for {variable} {year}-{month}")
            
            try:
                # Create a new client
                c = cdsapi.Client()
                
                dataset = 'reanalysis-era5-land'
                request_params = {
                    'format': 'netcdf',
                    'variable': variable,
                    'year': str(year),
                    'month': f"{month:02d}",
                    'day': [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)],
                    'time': [f"{h:02d}:00" for h in range(24)],
                }
                
                c.retrieve(
                    dataset,
                    request_params,
                    str(nc_file)
                )
                
                # If successful, remove the failed marker
                failed_file.unlink()
                print(f"Successfully downloaded {variable} for {year}-{month}")
                
            except Exception as e:
                print(f"Retry failed for {variable} {year}-{month}: {str(e)}")
        
        time.sleep(10)  # Wait between retries

def main():
    # Set the number of parallel processes
    # A good rule of thumb is to use N-1 cores where N is the total number of cores
    # This prevents overloading the system
    num_cores = mp.cpu_count()
    max_workers = max(1, num_cores - 1)
    
    # Create downloader instance with multiprocessing
    downloader = ERA5LandDownloader(
        output_dir='./data/raw/grided/era5land_vars',
        max_workers=1
    )
    
    try:
        # Start the download
        print(f"Starting download with {max_workers} parallel workers")
        downloader.download_data(
            start_year=2017,
            end_year=2017
        )
        
        # Retry any failed downloads
        print("Checking for failed downloads to retry...")
        retry_failed_downloads('./data/raw/grided/era5land_vars')
        
    except KeyboardInterrupt:
        print("Download interrupted by user")
        logging.info("Download interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        logging.error(f"Unexpected error: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()