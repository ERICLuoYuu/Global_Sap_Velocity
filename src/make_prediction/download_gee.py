"""
ERA5-Land data downloader from Google Earth Engine with multiprocessing capabilities.
Follows the same approach as the original ERA5LandGEEProcessor with optimized parallel downloading.
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import time
import warnings
import geopandas as gpd
import subprocess
import glob
import json
import calendar
import logging
import multiprocessing as mp
from pathlib import Path
from datetime import datetime, timedelta
import rasterio
from rasterio.windows import from_bounds

# GEE libraries
import ee
import geemap
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("gee_downloader")

# ERA5-Land variable names in Google Earth Engine
GEE_VARIABLE_MAPPING = {
    '2m_temperature': 'temperature_2m',
    '2m_dewpoint_temperature': 'dewpoint_temperature_2m',
    'total_precipitation': 'total_precipitation',
    'surface_solar_radiation_downwards': 'surface_solar_radiation_downwards',
    '10m_u_component_of_wind': 'u_component_of_wind_10m',
    '10m_v_component_of_wind': 'v_component_of_wind_10m',
    'surface_pressure': 'surface_pressure',
    'total_evaporation': 'total_evaporation',
    'volumetric_soil_water_layer_1': 'volumetric_soil_water_layer_1',
    'volumetric_soil_water_layer_2': 'volumetric_soil_water_layer_2',
    'volumetric_soil_water_layer_3': 'volumetric_soil_water_layer_3',
    'volumetric_soil_water_layer_4': 'volumetric_soil_water_layer_4',
}

# Default region if none specified
DEFAULT_REGION = {
    'lat_min': -57.0,
    'lat_max': 78.0,
    'lon_min': -180.0,
    'lon_max': 180.0
}

def initialize_gee(project=None):
    """Initialize the Google Earth Engine API."""
    try:
        # Try to authenticate and initialize
        try:
            if project:
                ee.Initialize(project=project)
            else:
                ee.Initialize(project='ee-yuluo-2')
        except:
            # If specific project fails, try default initialization
            ee.Initialize()
        
        logger.info("Google Earth Engine initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Google Earth Engine: {str(e)}")
        logger.error("You may need to authenticate first using 'earthengine authenticate'")
        return False

def read_shapefile(shapefile_path):
    """
    Read a shapefile and convert it to an ee.Geometry object.
    
    Parameters:
    -----------
    shapefile_path : str or Path
        Path to the shapefile
            
    Returns:
    --------
    ee.Geometry
        Earth Engine geometry object
    """
    try:
        # Read the shapefile using geopandas
        gdf = gpd.read_file(shapefile_path)
        
        # Check if CRS is WGS84, if not, reproject
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
            logger.info(f"Reprojected shapefile from {gdf.crs} to EPSG:4326")
        
        # If multiple features exist, use the union
        if len(gdf) > 1:
            logger.info(f"Shapefile contains {len(gdf)} features. Using the union of all features.")
            geometry = gdf.unary_union
        else:
            geometry = gdf.geometry.iloc[0]
            
        # Convert shapely geometry to GeoJSON
        geo_json = geometry.__geo_interface__
        
        # Create an ee.Geometry object
        ee_geometry = ee.Geometry(geo_json)
        
        logger.info(f"Successfully converted shapefile to ee.Geometry")
        return ee_geometry
    except Exception as e:
        logger.error(f"Error reading shapefile: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_geometry_from_region(region):
    """Create an Earth Engine geometry from a region dictionary."""
    try:
        region = region or DEFAULT_REGION
        
        ee_geometry = ee.Geometry.Polygon(
            [[[region['lon_min'], region['lat_min']], 
              [region['lon_max'], region['lat_min']], 
              [region['lon_max'], region['lat_max']], 
              [region['lon_min'], region['lat_max']], 
              [region['lon_min'], region['lat_min']]]],
            None, False  # Planar coordinates for global extent
        )
        
        logger.info(f"Created rectangular geometry: lon=({region['lon_min']:.4f}, {region['lon_max']:.4f}), lat=({region['lat_min']:.4f}, {region['lat_max']:.4f})")
        return ee_geometry
    except Exception as e:
        logger.error(f"Error creating geometry from region: {str(e)}")
        return None

def download_day_worker(args):
    """
    Worker function for processing a single day of ERA5-Land data.
    Downloads all requested variables for a specific day.
    
    Parameters:
    -----------
    args : tuple
        Tuple containing (year, month, day, variables, shapefile, region, scale, output_dir)
        
    Returns:
    --------
    dict
        Dictionary of results for this day's processing
    """
    year, month, day, variables, shapefile, region, scale, output_dir = args
    
    # Create process-specific output directory
    output_dir = Path(output_dir)
    day_dir = output_dir 
    day_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize GEE for this process
    if not initialize_gee():
        logger.error(f"Process {os.getpid()}: Failed to initialize GEE")
        return {"success": False, "message": "GEE initialization failed"}
    
    try:
        # Create geometry from shapefile or region
        if shapefile:
            geometry = read_shapefile(shapefile)
            if not geometry:
                # Fall back to region if shapefile fails
                logger.warning(f"Process {os.getpid()}: Shapefile read failed, falling back to region")
                geometry = create_geometry_from_region(region)
        else:
            geometry = create_geometry_from_region(region)
        
        if not geometry:
            logger.error(f"Process {os.getpid()}: Failed to create geometry")
            return {"success": False, "message": "Geometry creation failed"}
        
        # Set up date range
        start_date = ee.Date.fromYMD(year, month, day)
        
        # Determine end date (next day)
        if day == calendar.monthrange(year, month)[1]:  # Last day of month
            if month == 12:
                end_date = ee.Date.fromYMD(year + 1, 1, 1)
            else:
                end_date = ee.Date.fromYMD(year, month + 1, 1)
        else:
            end_date = ee.Date.fromYMD(year, month, day + 1)
        
        logger.info(f"Process {os.getpid()}: Processing {year}-{month:02d}-{day:02d} for {len(variables)} variables")
        
        results = {}
        
        # Process each variable
        for variable in variables:
            try:
                # Get GEE variable name
                gee_var = GEE_VARIABLE_MAPPING.get(variable, variable)
                
                # Access ERA5-Land collection
                collection = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
                
                # Filter by date range
                collection = collection.filterDate(start_date, end_date)
                
                # Select the variable
                collection = collection.select(gee_var)
                
                # Get the data as a list of images
                image_list = collection.toList(collection.size())
                collection_size = image_list.size().getInfo()
                
                if collection_size == 0:
                    logger.warning(f"Process {os.getpid()}: No data found for {variable} on {year}-{month:02d}-{day:02d}")
                    continue
                
                logger.info(f"Process {os.getpid()}: Found {collection_size} images for {variable}")
                
                # Download each image in the collection
                successful_downloads = 0
                
                for i in range(collection_size):
                    if i % 4 == 0:  # Log every 4th image to reduce verbosity
                        logger.info(f"Process {os.getpid()}: Processing image {i+1}/{collection_size} for {variable}")
                    
                    # Get the image
                    img = ee.Image(image_list.get(i))
                    
                    # Get the timestamp in the standard GEE format first
                    time_start_raw = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd-HH-mm-ss').getInfo()

                    # Modify the format: Replace the hyphen between date and time with an underscore
                    # Example: "2017-04-01-00-00-00" -> "2017-04-01_00-00-00"
                    # We assume the format is always YYYY-MM-DD-HH-MM-SS (19 characters)
                    time_start_formatted = time_start_raw
                    if len(time_start_raw) == 19 and time_start_raw[10] == '-':
                         time_start_formatted = time_start_raw[:10] + "_" + time_start_raw[11:]
                    else:
                        # Log a warning if the format is unexpected, but proceed with the raw name
                        logger.warning(f"Process {os.getpid()}: Unexpected timestamp format '{time_start_raw}'. Using raw format for filename.")

                    # Create output file name using the newly formatted timestamp
                    tif_file = day_dir / f"{variable}_{time_start_formatted}.tif"
                    
                    # Skip if file already exists
                    if tif_file.exists():
                        successful_downloads += 1
                        continue
                    
                    max_retries = 3
                    for retry in range(max_retries):
                        try:
                            # Export the image to a GeoTIFF
                            success = geemap.ee_export_image(
                                img.select(gee_var),
                                filename=str(tif_file),
                                scale=scale,
                                region=geometry,
                                file_per_band=False,
                                crs='EPSG:4326'
                            )
                            
                            # Check if file was created
                            if tif_file.exists():
                                successful_downloads += 1
                                break
                            else:
                                if retry < max_retries - 1:  # Not the last retry
                                    logger.warning(f"Process {os.getpid()}: Export reported success but file not found. Retrying ({retry+1}/{max_retries})...")
                                    time.sleep(2)  # Wait before retry
                                else:
                                    logger.error(f"Process {os.getpid()}: Failed to export image after {max_retries} retries")
                        except Exception as export_err:
                            if retry < max_retries - 1:  # Not the last retry
                                logger.warning(f"Process {os.getpid()}: Export error: {str(export_err)}. Retrying ({retry+1}/{max_retries})...")
                                time.sleep(2)  # Wait before retry
                            else:
                                logger.error(f"Process {os.getpid()}: Failed to export image after {max_retries} retries: {str(export_err)}")
                
                # Store results for this variable
                results[variable] = {
                    "success": successful_downloads > 0,
                    "total_images": collection_size,
                    "successful_downloads": successful_downloads,
                    "output_dir": str(day_dir)
                }
                
                logger.info(f"Process {os.getpid()}: Completed {variable}: {successful_downloads}/{collection_size} images downloaded")
                
            except Exception as var_err:
                logger.error(f"Process {os.getpid()}: Error processing {variable}: {str(var_err)}")
                results[variable] = {
                    "success": False,
                    "error": str(var_err)
                }
        
        # Create summary file
        summary_file = day_dir / "download_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "date": f"{year}-{month:02d}-{day:02d}",
                "variables": results,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        
        return {
            "success": any(v.get("success", False) for v in results.values()),
            "date": f"{year}-{month:02d}-{day:02d}",
            "results": results,
            "output_dir": str(day_dir)
        }
        
    except Exception as e:
        logger.error(f"Process {os.getpid()}: Error processing day {year}-{month:02d}-{day:02d}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "date": f"{year}-{month:02d}-{day:02d}",
            "error": str(e)
        }

def download_era5land_data_parallel(
    variables,
    year,
    month=None,
    months=None,
    days=None,
    region=None,
    shapefile=None,
    scale=11132,
    output_dir="./era5land_data",
    num_workers=None
):
    """
    Download ERA5-Land data for multiple days/months using parallel processing.
    
    Parameters:
    -----------
    variables : list
        List of variable names to download
    year : int
        Year to download data for
    month : int, optional
        Single month to download data for (if None and months is None, all months in year)
    months : list, optional
        List of specific months to download (overrides month parameter)
    days : list, optional
        List of days to download (if None, all days in each month)
    region : dict, optional
        Region to download data for (lat_min, lat_max, lon_min, lon_max)
    shapefile : str, optional
        Path to shapefile defining the region
    scale : float, optional
        Scale in meters for GEE export (default ERA5-Land resolution ~9km)
    output_dir : str, optional
        Directory to save downloaded data
    num_workers : int, optional
        Number of worker processes (defaults to CPU count - 1)
    
    Returns:
    --------
    dict
        Dictionary of results with dates as keys
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize Earth Engine (for main process)
    if not initialize_gee():
        logger.error("Failed to initialize Google Earth Engine. Exiting.")
        return {"success": False, "error": "GEE initialization failed"}
    
    # Set number of worker processes
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    else:
        num_workers = max(1, min(num_workers, mp.cpu_count()))
    
    logger.info(f"Using {num_workers} worker processes")
    
    # Generate tasks for each day to process
    tasks = []
    
    # Determine which months to process
    if months:
        months_to_process = months
    elif month is not None:
        months_to_process = [month]
    else:
        months_to_process = range(1, 13)  # All months
    
    for m in months_to_process:
        # Determine days to process for this month
        if days is not None:
            # Filter days to those that are valid for this month
            month_days = calendar.monthrange(year, m)[1]
            days_to_process = [d for d in days if 1 <= d <= month_days]
        else:
            # All days in the month
            days_to_process = range(1, calendar.monthrange(year, m)[1] + 1)
        
        # Add tasks for each day
        for d in days_to_process:
            tasks.append((year, m, d, variables, shapefile, region, scale, output_dir))
    
    logger.info(f"Prepared {len(tasks)} download tasks across {len(months_to_process)} months")
    
    # Set starting time to measure total duration
    start_time = time.time()
    
    # Process tasks with multiprocessing
    results = {}
    with mp.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(download_day_worker, tasks):
            if result.get("success", False):
                date = result.get("date")
                results[date] = result
                logger.info(f"Successfully processed {date}")
            else:
                date = result.get("date", "unknown date")
                logger.error(f"Failed to process {date}: {result.get('error', 'unknown error')}")
                results[date] = result
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Create summary of download results
    successful_days = sum(1 for r in results.values() if r.get("success", False))
    
    summary = {
        "total_tasks": len(tasks),
        "successful_days": successful_days,
        "failed_days": len(tasks) - successful_days,
        "elapsed_time": f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}",
        "elapsed_seconds": elapsed_time,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "results": results
    }
    
    # Save summary to file
    summary_file = output_dir / "download_summary.json"
    with open(summary_file, 'w') as f:
        # Convert the full results to a simpler format for the JSON file
        simplified_results = {}
        for date, result in results.items():
            if "results" in result:
                # Count successful variables
                var_success = sum(1 for v in result["results"].values() if v.get("success", False))
                simplified_results[date] = {
                    "success": result.get("success", False),
                    "variables_processed": len(result["results"]),
                    "variables_successful": var_success,
                    "output_dir": result.get("output_dir")
                }
            else:
                simplified_results[date] = {
                    "success": result.get("success", False),
                    "error": result.get("error", "unknown error")
                }
        
        simple_summary = {
            "total_tasks": summary["total_tasks"],
            "successful_days": summary["successful_days"],
            "failed_days": summary["failed_days"],
            "elapsed_time": summary["elapsed_time"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "timestamp": summary["timestamp"],
            "results": simplified_results
        }
        
        json.dump(simple_summary, f, indent=2)
    
    logger.info(f"Download completed in {summary['elapsed_time']} with {successful_days}/{len(tasks)} successful days")
    logger.info(f"Summary saved to {summary_file}")
    
    return summary

def main():
    """Main function to parse arguments and run the downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download ERA5-Land data from GEE with multiprocessing')
    
    # Time parameters
    parser.add_argument('--year', type=int, required=True, help='Year to download data for')
    parser.add_argument('--month', type=int, help='Single month to download data for (1-12)')
    parser.add_argument('--months', type=int, nargs='+', help='Multiple months to download (e.g., --months 1 3 6 12)')
    parser.add_argument('--month-range', type=str, help='Range of months to download (e.g., 6-8 for summer)')
    parser.add_argument('--days', type=int, nargs='+', help='Days to download (e.g., --days 1 15 30)')
    parser.add_argument('--day-range', type=str, help='Range of days to download (e.g., 1-15)')
    
    # Variables to download
    parser.add_argument('--variables', type=str, nargs='+', 
                      default=['2m_temperature', '2m_dewpoint_temperature', 
                               'surface_solar_radiation_downwards', '10m_u_component_of_wind', 
                               '10m_v_component_of_wind'],
                      help='Variables to download')
    
    # Region specification
    parser.add_argument('--region', type=str, help='Region as lon_min,lat_min,lon_max,lat_max')
    parser.add_argument('--shapefile', type=str, help='Path to shapefile for region definition')
    
    # Processing options
    parser.add_argument('--scale', type=float, default=11132, 
                        help='Scale in meters for GEE export (default ~9km for ERA5-Land)')
    parser.add_argument('--workers', type=int, help='Number of worker processes')
    parser.add_argument('--output-dir', type=str, default='./era5land_data', 
                        help='Directory to save downloaded data')
    
    args = parser.parse_args()
    
    # Process months argument - prioritize --months over --month-range over --month
    months = None
    if args.months:
        months = args.months
        for m in months:
            if m < 1 or m > 12:
                parser.error(f"Invalid month: {m}. Month must be between 1 and 12.")
    elif args.month_range:
        try:
            start_month, end_month = map(int, args.month_range.split('-'))
            if start_month < 1 or start_month > 12 or end_month < 1 or end_month > 12:
                parser.error("Month range must use values between 1 and 12")
            months = list(range(start_month, end_month + 1))
        except:
            parser.error("Invalid month range format. Use 'start-end' (e.g., '6-8')")
    elif args.month:
        if args.month < 1 or args.month > 12:
            parser.error("Month must be between 1 and 12")
        months = [args.month]
    
    # Process days argument
    days = None
    if args.days:
        days = args.days
    elif args.day_range:
        try:
            start_day, end_day = map(int, args.day_range.split('-'))
            days = list(range(start_day, end_day + 1))
        except:
            parser.error("Invalid day range format. Use 'start-end' (e.g., '1-15')")
    
    # Process region argument
    region = None
    if args.region:
        try:
            lon_min, lat_min, lon_max, lat_max = map(float, args.region.split(','))
            region = {
                'lon_min': lon_min,
                'lat_min': lat_min,
                'lon_max': lon_max,
                'lat_max': lat_max
            }
        except:
            parser.error("Invalid region format. Use 'lon_min,lat_min,lon_max,lat_max'")
    
    # Validate days for each month if specified
    if months and days:
        for m in months:
            month_days = calendar.monthrange(args.year, m)[1]
            invalid_days = [d for d in days if d < 1 or d > month_days]
            if invalid_days:
                parser.error(f"Invalid days for month {m}: {invalid_days}. Month has {month_days} days.")
    
    # Show download configuration
    month_str = "None (all months)"
    if months:
        if len(months) == 1:
            month_str = str(months[0])
        elif len(months) <= 4:
            month_str = ", ".join(map(str, months))
        else:
            month_str = f"{min(months)}-{max(months)} ({len(months)} months)"
    
    logger.info(f"Starting ERA5-Land download with the following parameters:")
    logger.info(f"Year: {args.year}")
    logger.info(f"Months: {month_str}")
    logger.info(f"Days: {days if days else 'All days in month(s)'}")
    logger.info(f"Variables: {args.variables}")
    logger.info(f"Region: {region if region else 'Default global'}")
    logger.info(f"Shapefile: {args.shapefile if args.shapefile else 'None'}")
    logger.info(f"Scale: {args.scale} meters")
    logger.info(f"Workers: {args.workers if args.workers else 'CPU count - 1'}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Run the download
    download_era5land_data_parallel(
        variables=args.variables,
        year=args.year,
        month=None,  # We're using the months list instead
        months=months,
        days=days,
        region=region,
        shapefile=args.shapefile,
        scale=args.scale,
        output_dir=args.output_dir,
        num_workers=args.workers
    )

if __name__ == "__main__":
    main()