import cdsapi
import os
from datetime import datetime
import time
import logging
from pathlib import Path
import calendar
import pandas as pd
import xarray as xr
import zipfile
import shutil
import warnings

# Suppress specific warning about cfgrib
warnings.filterwarnings('ignore', message='Engine .?cfgrib.? loading failed')


class ERA5LandSiteExtractor:
    def cleanup_temp_files(self):
        """Clean up any temporary files."""
        patterns = ['*.nc', '*.zip']
        for pattern in patterns:
            for temp_file in self.output_dir.glob(pattern):
                try:
                    temp_file.unlink()
                except Exception as e:
                    logging.warning(f"Could not delete temporary file {temp_file}: {str(e)}")
        
        # Clean up temp directory if it exists
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def __init__(self, output_dir='era5land_site_data'):
        self.c = cdsapi.Client()
        self.output_dir = Path(output_dir).resolve()  # Get absolute path
        self.output_dir.mkdir(exist_ok=True)
        
        # Create temp directory for ZIP extraction
        self.temp_dir = self.output_dir / 'temp'
        self.temp_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            filename=self.output_dir / 'extraction_log.txt',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        
        # Define all variables to extract in a flat structure
        self.variables = [
            '10m_u_component_of_wind',  # u_compor
            '10m_v_component_of_wind',  # v_compor
            'surface_solar_radiation_downwards',  # surface_solar_radiation_downwards
            'surface_net_solar_radiation',  # surface_net_solar_radiation
            'surface_net_thermal_radiation',  # surface_net_thermal_radiation
            'total_precipitation',  # total_prec
            'volumetric_soil_water_layer_1',  # volumetric_soil_water_1
            'volumetric_soil_water_layer_4',  # volumetric_soil_water_4
            '2m_temperature',  # temperature
            '2m_dewpoint_temperature'  # dewpoint
        ]

    def parse_datetime(self, date_str):
        """Parse datetime string handling various formats."""
        try:
            # First try simple format
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            try:
                # Try format with time
                return datetime.strptime(date_str.split('+')[0], '%Y-%m-%d %H:%M:%S')
            except ValueError as e:
                logging.error(f"Could not parse date: {date_str}")
                raise e

    def extract_and_read_data(self, zip_file_path):
        """Extract ZIP file and read the NetCDF data."""
        try:
            # Clean temp directory
            for file in self.temp_dir.glob('*'):
                file.unlink()
            
            # Extract ZIP file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # Log ZIP contents
                contents = zip_ref.namelist()
                logging.info(f"ZIP file contents: {contents}")
                
                # Extract files
                zip_ref.extractall(self.temp_dir)
            
            # Find all files in temp directory
            all_files = list(self.temp_dir.glob('*'))
            logging.info(f"Files in temp directory: {[f.name for f in all_files]}")
            
            # Find NetCDF files
            nc_files = list(self.temp_dir.glob('*.nc'))
            if not nc_files:
                # Check for GRIB files as fallback
                grib_files = list(self.temp_dir.glob('*.grib'))
                if grib_files:
                    logging.error("Found GRIB files instead of NetCDF files. Need to modify request format.")
                    raise FileNotFoundError("Found GRIB files but expected NetCDF format")
                raise FileNotFoundError(f"No NetCDF files found in ZIP archive. ZIP contents: {contents}")
            
            # Read the first NC file found
            logging.info(f"Reading NetCDF file: {nc_files[0]}")
            ds = xr.open_dataset(nc_files[0], engine='netcdf4')
            return ds
            
        except Exception as e:
            logging.error(f"Error processing ZIP file {zip_file_path}: {str(e)}")
            raise e
        finally:
            # Clean up ZIP file
            if zip_file_path.exists():
                zip_file_path.unlink()

    def extract_site_data(self, site_info_path, output_path):
        """
        Extract ERA5-Land data for specific sites and save as a single CSV with all variables.
        """
        # Read site information
        sites_df = pd.read_csv(site_info_path)
        
        all_data = []
        
        # Process each site
        for idx, site in sites_df.iterrows():
            logging.info(f"Processing site: {site['site_name']}")
            
            try:
                # Convert dates with robust parsing
                start_date = self.parse_datetime(site['start_date'])
                end_date = self.parse_datetime(site['end_date'])
                
                current_date = start_date
                while current_date <= end_date:
                    year = current_date.year
                    month = current_date.month
                    
                    try:
                        # Download all variables for this month
                        request_params = {
                            'product_type': ['reanalysis'],
                            'format': 'netcdf.zip',  # Request ZIP format
                            'variable': self.variables,
                            'year': str(year),
                            'month': f"{month:02d}",
                            'day': [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)],
                            'time': [f"{h:02d}:00" for h in range(24)],
                            'area': [float(site['lat']), float(site['lon']), float(site['lat']), float(site['lon'])],
                        }
                        
                        zip_file = self.output_dir / f"temp_{site['site_name']}_{year}_{month:02d}.zip"
                        self.c.retrieve('reanalysis-era5-land', request_params, str(zip_file))
                        
                        try:
                            # Extract and read the data
                            ds = self.extract_and_read_data(zip_file)
                            
                            # Convert to dataframe
                            df = ds.to_dataframe().reset_index()
                            
                            # Convert time column to datetime if it's not already
                            if 'time' in df.columns:
                                df['time'] = pd.to_datetime(df['time'])
                            
                            # Rename columns to match desired format
                            column_mapping = {
                                '10m_u_component_of_wind': 'u_compor',
                                '10m_v_component_of_wind': 'v_compor',
                                'surface_solar_radiation_downwards': 'surface_solar_radiation_downwards',
                                'surface_net_solar_radiation': 'surface_net_solar_radiation',
                                'surface_net_thermal_radiation': 'surface_net_thermal_radiation',
                                'total_precipitation': 'total_prec',
                                'volumetric_soil_water_layer_1': 'volumetric_soil_water_1',
                                'volumetric_soil_water_layer_4': 'volumetric_soil_water_4',
                                '2m_temperature': 'temperature',
                                '2m_dewpoint_temperature': 'dewpoint',
                                'time': 'TIMESTAMP'
                            }
                            df = df.rename(columns=column_mapping)
                            
                            # Add site information
                            df['latitude'] = site['lat']
                            df['longitude'] = site['lon']
                            df['site_name'] = site['site_name']
                            
                            # Append to the list
                            all_data.append(df)
                            
                            logging.info(f"Successfully processed {site['site_name']} for {year}-{month}")
                            time.sleep(5)  # Prevent overwhelming the server
                        finally:
                            # Clean up all temporary files for this month
                            temp_pattern = f"temp_{site['site_name']}_{year}_{month:02d}.*"
                            for temp_file in self.output_dir.glob(temp_pattern):
                                try:
                                    temp_file.unlink()
                                except Exception as e:
                                    logging.warning(f"Could not delete temporary file {temp_file}: {str(e)}")
                        
                    except Exception as e:
                        logging.error(f"Error processing {site['site_name']} for {year}-{month}: {str(e)}")
                        time.sleep(60)  # Wait longer if there's an error
                    
                    # Move to next month
                    current_date = datetime(year, month % 12 + 1, 1) if month < 12 else datetime(year + 1, 1, 1)
                    
            except Exception as e:
                logging.error(f"Error processing site {site['site_name']}: {str(e)}")
                continue
        
        # Combine all data and save
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            
            # Sort by timestamp and site
            final_df = final_df.sort_values(['site_name', 'TIMESTAMP'])
            
            # Reorder columns to match desired format
            column_order = [
                'latitude', 'longitude', 'TIMESTAMP',
                'u_compor', 'v_compor',
                'surface_solar_radiation_downwards', 'surface_net_solar_radiation', 
                'surface_net_thermal_radiation', 'total_prec',
                'volumetric_soil_water_1', 'volumetric_soil_water_4',
                'temperature', 'dewpoint', 'site_name'
            ]
            final_df = final_df[column_order]
            
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            final_df.to_csv(output_path, index=False)
            logging.info(f"Data saved to {output_path}")
        else:
            logging.error("No data was processed successfully")

def main():
    # Use Path for consistent path handling
    base_path = Path('./data/raw').resolve()  # Get absolute path
    output_dir = base_path / 'era5land_site_data'
    
    # Create extractor instance
    extractor = ERA5LandSiteExtractor(output_dir=output_dir)
    
    try:
        # Start the extraction
        site_info_path = base_path / '0.1.5/0.1.5/csv/sapwood/site_info1.csv'
        output_path = output_dir / 'era5land_extracted_data.csv'
        
        extractor.extract_site_data(
            site_info_path=site_info_path,
            output_path=output_path
        )
    except KeyboardInterrupt:
        logging.info("Extraction interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
    finally:
        # Clean up all temporary files
        extractor.cleanup_temp_files()

if __name__ == "__main__":
    main()