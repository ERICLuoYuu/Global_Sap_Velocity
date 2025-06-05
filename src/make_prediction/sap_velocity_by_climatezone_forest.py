import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rasterio.warp import reproject, Resampling, calculate_default_transform
import os
from scipy import stats
import matplotlib.patches as mpatches
import ee
import geemap
from pathlib import Path

# Initialize Earth Engine - this requires authentication
# If running this fails, run 'earthengine authenticate' in your terminal first
try:
    ee.Initialize(project='era5download-447713')
    print("Google Earth Engine initialized successfully!")
except Exception as e:
    print("Error initializing Earth Engine. Please run 'earthengine authenticate' in your terminal.")
    print(f"Error details: {e}")
    
def download_igbp_landcover(output_path, year=2020, scale=10000):
    """
    Download IGBP land cover data from Google Earth Engine.
    
    Parameters:
    -----------
    output_path : str
        Path to save the downloaded raster
    year : int
        Year for the land cover data (default: 2020)
    scale : int
        Resolution in meters (default: 1000m, which is ~0.01 degrees)
    """
    print(f"Downloading IGBP land cover data for year {year}...")
    
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Get the MODIS land cover collection
        lc_collection = ee.ImageCollection("MODIS/061/MCD12Q1")
        
        # Filter by year
        lc_image = lc_collection.filter(ee.Filter.calendarRange(year, year, 'year')).first()
        
        # Select the IGBP classification (Type 1)
        igbp = lc_image.select('LC_Type1')
        
        # Define world extent (or limit to a specific region if needed)
        world_extent = ee.Geometry.Rectangle([-180, -90, 180, 90])
        
        # Create a GEE task to export the image
        print("Starting GEE export task...")
        
        # Option 1: Using geemap
        geemap.ee_export_image(
            igbp,
            filename=output_path,
            scale=scale,
            region=world_extent,
            file_per_band=False
        )
        
        # Verify the file was created
        if os.path.exists(output_path):
            print(f"IGBP land cover data downloaded to {output_path}")
            return output_path
        else:
            print(f"Warning: Expected output file not found at {output_path}")
            return None
            
    except Exception as e:
        print(f"Error downloading IGBP land cover data: {e}")
        return None

def get_igbp_forest_classes():
    """
    Returns dictionary of IGBP forest type classes and their descriptions.
    
    Returns:
    --------
    dict: Dictionary mapping class values to descriptions
    """
    # Create a dictionary for IGBP classification (focusing on forest types)
    igbp_classes = {
        1: "ENF - Evergreen Needleleaf",
        2: "EBF - Evergreen Broadleaf",
        3: "DNF - Deciduous Needleleaf",
        4: "DBF - Deciduous Broadleaf",
        5: "MF - Mixed Forest",
        8: "WSA - Woody Savannas",
        9: "SAV - Savannas"
    }
    return igbp_classes

def extract_major_climate_zones(koppen, koppen_nodata=None):
    """
    Properly extract major climate zones from Beck et al. (2018/2023) 
    Köppen-Geiger classification dataset.
    
    Parameters:
    -----------
    koppen : numpy array
        The Köppen-Geiger classification raster
    koppen_nodata : int or float, optional
        The nodata value for the koppen dataset
        
    Returns:
    --------
    koppen_major : numpy array
        Array with values representing major climate zones
    major_zones : dict
        Dictionary mapping zone codes to zone names
    """
    # Create a copy of the input data
    koppen_major = koppen.copy()
    
    # Define the ranges for each major climate zone based on Beck et al. legend
    zone_ranges = {
        1: [1, 2, 3],           # A - Tropical
        2: [4, 5, 6, 7],        # B - Arid
        3: list(range(8, 17)),  # C - Temperate (8-16)
        4: list(range(17, 29)), # D - Cold (17-28)
        5: [29, 30]             # E - Polar (29-30)
    }
    
    # Define zone names
    major_zones = {
        1: "A - Tropical",
        2: "B - Arid",
        3: "C - Temperate", 
        4: "D - Continental",
        5: "E - Polar"
    }
    
    # Create a mask for valid data if nodata value is provided
    if koppen_nodata is not None:
        valid_mask = (koppen != koppen_nodata)
    else:
        valid_mask = ~np.isnan(koppen)
    
    # Initialize with zeros or nodata
    if koppen_nodata is not None:
        koppen_major.fill(koppen_nodata)
    else:
        koppen_major.fill(np.nan)
    
    # Map each value to its major zone
    for major_zone, values in zone_ranges.items():
        for value in values:
            mask = (koppen == value) & valid_mask
            koppen_major[mask] = major_zone
    
    return koppen_major, major_zones

def resample_koppen_geiger(input_path, output_path, target_resolution_degrees):
    """
    Resample the Köppen-Geiger classification raster to a lower resolution.
    
    Parameters:
    -----------
    input_path : str
        Path to the original high-resolution Köppen-Geiger raster
    output_path : str
        Path to save the resampled raster
    target_resolution_degrees : float
        Target resolution in degrees (e.g., 0.05 for ~5km at the equator)
    """
    print(f"Resampling Köppen-Geiger classification to {target_resolution_degrees} degree resolution...")
    
    with rasterio.open(input_path) as src:
        # Get the data type and determine an appropriate nodata value
        dtype = src.dtypes[0]
        
        # Choose an appropriate nodata value based on the data type
        if 'uint8' in dtype:
            # For uint8, use 255 as nodata (assuming Köppen doesn't use this value)
            nodata_value = 255
        elif 'uint16' in dtype:
            nodata_value = 65535  # Max value for uint16
        elif 'int' in dtype:
            # For signed integers, we can use negative values
            nodata_value = -9999
        else:
            # For floating point, -9999 is usually fine
            nodata_value = -9999.0
            
        print(f"Using nodata value {nodata_value} for data type {dtype}")
        
        # Calculate new transform and dimensions
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs,
            src.crs,
            src.width,
            src.height,
            left=src.bounds.left,
            bottom=src.bounds.bottom,
            right=src.bounds.right,
            top=src.bounds.top,
            resolution=(target_resolution_degrees, target_resolution_degrees)
        )
        
        # Setup destination raster
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({
            'crs': src.crs,
            'transform': dst_transform,
            'width': dst_width,
            'height': dst_height,
            'nodata': nodata_value
        })
        
        # Create destination raster
        with rasterio.open(output_path, 'w', **dst_kwargs) as dst:
            # Read the source data
            data = src.read(1)
            
            # Initialize the destination array
            dst_data = np.empty((dst_height, dst_width), dtype=dtype)
            
            # Perform the reprojection
            reproject(
                source=data,
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=src.crs,
                resampling=Resampling.mode,  # Use mode for categorical data like climate zones
                src_nodata=src.nodata,
                dst_nodata=nodata_value
            )
            
            # Write the data to the output raster
            dst.write(dst_data, 1)
            
    print(f"Resampling complete. Resampled raster saved to {output_path}")
    return output_path

def analyze_sap_velocity_by_climate_and_forest(sap_velocity_path, koppen_path, igbp_path=None, year=2020, target_resolution=0.1):
    """
    Analyze sap velocity distributions across major climate zones and forest types.
    
    Parameters:
    -----------
    sap_velocity_path : str
        Path to the sap velocity raster (.tif)
    koppen_path : str
        Path to the original high-resolution Köppen-Geiger raster
    igbp_path : str, optional
        Path to the IGBP land cover classification. If None, it will be downloaded.
    year : int
        Year for IGBP data if it needs to be downloaded
    target_resolution : float
        Target resolution in degrees for resampling (default: 0.1° ≈ 10km at equator)
    """
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Safely attempt to load sap velocity data
        print("Loading sap velocity raster...")
        with rasterio.open(sap_velocity_path) as src:
            sap_velocity = src.read(1)
            sap_meta = src.meta
            sap_transform = src.transform
            sap_crs = src.crs
            sap_nodata = src.nodata
            
            # Validate sap velocity data
            if sap_velocity.size == 0:
                raise ValueError("Sap velocity raster is empty")
                
            print(f"  Data type: {src.dtypes[0]}")
            print(f"  No data value: {sap_nodata}")
            print(f"  Resolution: {src.res}")
            print(f"  Dimensions: {src.width} x {src.height}")
            print(f"  Sap velocity value range: {np.nanmin(sap_velocity)} to {np.nanmax(sap_velocity)}")
            
            # Additional validation - check for all NaN or infinite values
            if np.all(np.isnan(sap_velocity)) or np.all(np.isinf(sap_velocity)):
                raise ValueError("Sap velocity raster contains only NaN or infinite values")
    
        # Download or load IGBP data
        if igbp_path is None or not os.path.exists(igbp_path):
            igbp_path = os.path.join(output_dir, f"igbp_landcover_{year}.tif")
            downloaded_path = download_igbp_landcover(igbp_path, year)
            
            if downloaded_path is None or not os.path.exists(downloaded_path):
                raise FileNotFoundError(f"Failed to download IGBP land cover data. Please check GEE authentication or provide existing file.")
    
        # Get IGBP forest classes
        igbp_forest_classes = get_igbp_forest_classes()
    
        # Resample the Köppen-Geiger classification
        resampled_koppen_path = os.path.join(output_dir, "koppen_resampled.tif")
        resample_koppen_geiger(koppen_path, resampled_koppen_path, target_resolution)
        
        # Load resampled Köppen-Geiger climate classification
        print("Loading resampled Köppen-Geiger raster...")
        with rasterio.open(resampled_koppen_path) as src:
            koppen = src.read(1)
            koppen_meta = src.meta
            koppen_transform = src.transform
            koppen_crs = src.crs
            koppen_nodata = src.nodata
            
            print(f"  Data type: {src.dtypes[0]}")
            print(f"  No data value: {koppen_nodata}")
            print(f"  Resolution: {src.res}")
            print(f"  Dimensions: {src.width} x {src.height}")
            
            # Validate
            if koppen.size == 0:
                raise ValueError("Köppen-Geiger raster is empty")
        
        # Load IGBP land cover classification
        print("Loading IGBP land cover classification...")
        with rasterio.open(igbp_path) as src:
            igbp = src.read(1)
            igbp_meta = src.meta
            igbp_transform = src.transform
            igbp_crs = src.crs
            igbp_nodata = src.nodata if src.nodata is not None else 0  # MODIS often uses 0 for nodata
            
            print(f"  Data type: {src.dtypes[0]}")
            print(f"  No data value: {igbp_nodata}")
            print(f"  Resolution: {src.res}")
            print(f"  Dimensions: {src.width} x {src.height}")
            
            # Validate
            if igbp.size == 0:
                raise ValueError("IGBP raster is empty")
            
            # Check unique values to ensure we have forest classes
            unique_igbp = np.unique(igbp)
            print(f"  IGBP classes found: {unique_igbp}")
            
            # Verify we have some forest classes
            forest_classes_in_data = set(unique_igbp).intersection(set(igbp_forest_classes.keys()))
            if not forest_classes_in_data:
                print(f"WARNING: No forest classes found in IGBP data. Available classes: {unique_igbp}")
                print(f"Expected forest classes: {list(igbp_forest_classes.keys())}")
        
        # Extract major climate zones with error handling
        print("Extracting major climate zones...")
        try:
            koppen_major, major_zones = extract_major_climate_zones(koppen, koppen_nodata)
            
            # Verify we have climate zones extracted
            unique_major_zones = np.unique(koppen_major[~np.isnan(koppen_major) & (koppen_major != koppen_nodata) if koppen_nodata is not None else ~np.isnan(koppen_major)])
            if len(unique_major_zones) == 0:
                raise ValueError("No climate zones were extracted from the Köppen-Geiger data")
                
            print(f"Extracted climate zones: {unique_major_zones}")
        except Exception as e:
            print(f"Error extracting climate zones: {e}")
            raise
        
        # Align all grids to the same dimensions and resolution
        print("Aligning datasets to the same grid...")
        
        # Create copies to avoid modifying original arrays
        koppen_aligned = koppen_major.copy()
        igbp_aligned = igbp.copy()
        
        # Align Köppen data if needed
        if sap_crs != koppen_crs or sap_velocity.shape != koppen_major.shape:
            print("Aligning climate zones to match sap velocity grid...")
            dst_koppen = np.zeros_like(sap_velocity, dtype=koppen_major.dtype)
            reproject(
                source=koppen_major,
                destination=dst_koppen,
                src_transform=koppen_transform,
                src_crs=koppen_crs,
                dst_transform=sap_transform,
                dst_crs=sap_crs,
                resampling=Resampling.mode
            )
            koppen_aligned = dst_koppen
        
        # Align IGBP data if needed
        if sap_crs != igbp_crs or sap_velocity.shape != igbp.shape:
            print("Aligning IGBP forest types to match sap velocity grid...")
            dst_igbp = np.zeros_like(sap_velocity, dtype=igbp.dtype)
            reproject(
                source=igbp,
                destination=dst_igbp,
                src_transform=igbp_transform,
                src_crs=igbp_crs,
                dst_transform=sap_transform,
                dst_crs=sap_crs,
                resampling=Resampling.mode
            )
            igbp_aligned = dst_igbp
        
        # Check for successful alignment
        if koppen_aligned.shape != sap_velocity.shape or igbp_aligned.shape != sap_velocity.shape:
            raise ValueError(f"Dataset alignment failed. Shapes: sap={sap_velocity.shape}, koppen={koppen_aligned.shape}, igbp={igbp_aligned.shape}")
        
        # Create masks for valid data
        if sap_nodata is not None:
            sap_valid_mask = (sap_velocity != sap_nodata) & ~np.isnan(sap_velocity) & ~np.isinf(sap_velocity)
        else:
            sap_valid_mask = ~np.isnan(sap_velocity) & ~np.isinf(sap_velocity)
            
        if koppen_nodata is not None:
            koppen_valid_mask = (koppen_aligned != koppen_nodata) & ~np.isnan(koppen_aligned)
        else:
            koppen_valid_mask = ~np.isnan(koppen_aligned)
            
        igbp_valid_mask = (igbp_aligned != igbp_nodata)
        
        # Print mask statistics to diagnose potential issues
        print(f"Sap velocity valid pixels: {np.sum(sap_valid_mask)} out of {sap_valid_mask.size} ({np.sum(sap_valid_mask)/sap_valid_mask.size*100:.2f}%)")
        print(f"Köppen valid pixels: {np.sum(koppen_valid_mask)} out of {koppen_valid_mask.size} ({np.sum(koppen_valid_mask)/koppen_valid_mask.size*100:.2f}%)")
        print(f"IGBP valid pixels: {np.sum(igbp_valid_mask)} out of {igbp_valid_mask.size} ({np.sum(igbp_valid_mask)/igbp_valid_mask.size*100:.2f}%)")
        
        # Create forest mask
        forest_mask = np.zeros_like(igbp_valid_mask, dtype=bool)
        for forest_class in igbp_forest_classes.keys():
            forest_mask |= (igbp_aligned == forest_class)
            
        print(f"Forest pixels: {np.sum(forest_mask)} out of {forest_mask.size} ({np.sum(forest_mask)/forest_mask.size*100:.2f}%)")
        
        # Combine all masks
        valid_mask = sap_valid_mask & koppen_valid_mask & igbp_valid_mask & forest_mask
        print(f"Combined valid pixels: {np.sum(valid_mask)} out of {valid_mask.size} ({np.sum(valid_mask)/valid_mask.size*100:.2f}%)")
        
        if np.sum(valid_mask) == 0:
            raise ValueError("No valid pixels found after applying all masks!")
        
        # Extract data
        print("Extracting sap velocity values for each climate zone and forest type...")
        data = []
        
        # For each major climate zone and forest type, extract sap velocity values
        for zone_id, zone_name in major_zones.items():
            zone_mask = (koppen_aligned == zone_id) & valid_mask
            
            if not np.any(zone_mask):
                print(f"No data points found for {zone_name}")
                continue
                
            for forest_id, forest_name in igbp_forest_classes.items():
                # Create combined mask for this climate zone and forest type
                combined_mask = zone_mask & (igbp_aligned == forest_id)
                
                # Extract sap velocity values for this combination
                values = sap_velocity[combined_mask]
                
                if len(values) == 0:
                    print(f"No data points for {zone_name}, {forest_name}")
                    continue
                    
    
                
                # Sample if very large
                original_count = len(values)
                if len(values) > 500000:
                    np.random.seed(42)
                    indices = np.random.choice(len(values), 500000, replace=False)
                    values = values[indices]
                    print(f"  Sampled {len(values)} from {original_count} points")
                
                print(f"{zone_name}, {forest_name}: {len(values)} data points")
                
                # Skip if we don't have enough data points
                if len(values) < 10:
                    print(f"  Skipping {zone_name}, {forest_name} due to insufficient data points")
                    continue
                
                
                # Calculate statistics
                mean_value = np.mean(values)
                median_value = np.median(values)
                std_value = np.std(values)
                min_value = np.min(values)
                max_value = np.max(values)
                
                # Add to dataframe
                data.append({
                    'Climate Zone': zone_name,
                    'Forest Type': forest_name,
                    'Sap Velocity Mean': mean_value,
                    'Sap Velocity Median': median_value,
                    'Sap Velocity Std': std_value,
                    'Sap Velocity Min': min_value,
                    'Sap Velocity Max': max_value,
                    'Count': len(values)
                })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Check if we have data
        if df.empty:
            print("Warning: No data was extracted for any climate zone and forest type combination")
            return df
        
        # Print summary of data
        print("\nData summary:")
        print(f"  Number of climate zones with data: {df['Climate Zone'].nunique()}")
        print(f"  Number of forest types with data: {df['Forest Type'].nunique()}")
        print(f"  Total number of combinations: {len(df)}")
        print("\nSap velocity statistics across all combinations:")
        print(f"  Mean: {df['Sap Velocity Mean'].mean():.4f}")
        print(f"  Standard deviation: {df['Sap Velocity Std'].mean():.4f}")
        print(f"  Min: {df['Sap Velocity Mean'].min():.4f}")
        print(f"  Max: {df['Sap Velocity Mean'].max():.4f}")
        
        # Check for extreme variation in means that might cause plotting issues
        ratio = df['Sap Velocity Mean'].max() / df['Sap Velocity Mean'].min() if df['Sap Velocity Mean'].min() > 0 else float('inf')
        print(f"  Ratio of max/min means: {ratio:.2f}")
        if ratio > 1000:
            print("  WARNING: Extreme variation in means detected. Consider log-transforming data for visualization.")
        
        # Save to CSV
        csv_path = os.path.join(output_dir, "sap_velocity_by_climate_forest.csv")
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")
        
        # Create visualizations
        # create_stacked_bar_chart(df, output_dir)
        create_alternative_visualizations(df, output_dir)
        # log_transform_visualization(df, output_dir)
        create_absolute_value_charts(df, output_dir)
        plot_raw_sap_velocity_distributions(
        sap_velocity=sap_velocity,
        igbp_aligned=igbp_aligned,
        valid_mask=valid_mask,
        igbp_forest_classes=igbp_forest_classes,
        output_dir=output_dir)
        
        return df
        
    except Exception as e:
        print(f"ERROR in analysis: {e}")
        import traceback
        traceback.print_exc()
        # Return empty DataFrame on error
        return pd.DataFrame()


def plot_raw_sap_velocity_distributions(sap_velocity, igbp_aligned, valid_mask, igbp_forest_classes, output_dir, 
                                   max_sample_size=100000, filter_outliers=True):
    """
    Create visualizations showing the raw distribution of sap velocity values
    for each forest type, using already loaded and aligned data.
    
    Parameters:
    -----------
    sap_velocity : numpy.ndarray
        The sap velocity raster data
    igbp_aligned : numpy.ndarray
        The already aligned IGBP land cover classification
    valid_mask : numpy.ndarray
        Combined mask of valid data points
    igbp_forest_classes : dict
        Dictionary mapping forest class IDs to names
    output_dir : str
        Directory to save output figures
    max_sample_size : int, optional
        Maximum sample size per forest type for better performance (default: 100000)
    filter_outliers : bool, optional
        Whether to filter extreme outliers (default: True)
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "distributions"), exist_ok=True)
    
    print("\n=== Analyzing Raw Sap Velocity Distributions by Forest Type ===")
    
    try:
        # Collect raw data for each forest type
        forest_data = {}
        forest_stats = []
        
        print("\nExtracting raw sap velocity data by forest type:")
        for forest_id, forest_name in igbp_forest_classes.items():
            # Create mask for this forest type
            forest_type_mask = (igbp_aligned == forest_id) & valid_mask
            
            # Skip if no data
            if not np.any(forest_type_mask):
                print(f"No data points for {forest_name}")
                continue
            
            # Extract raw sap velocity values
            values = sap_velocity[forest_type_mask]
            
            # Skip if no values
            if len(values) == 0:
                print(f"No valid values for {forest_name}")
                continue
            
            original_count = len(values)
            
            # Filter out extreme outliers
            if filter_outliers and len(values) > 100:
                q1, q99 = np.percentile(values, [0.5, 99.5])
                iqr = q99 - q1
                lower_bound = max(0, q1 - 1.5 * iqr)  # Ensure no negative values
                upper_bound = q99 + 1.5 * iqr
                values = values[(values >= lower_bound) & (values <= upper_bound)]
                print(f"  {forest_name}: {original_count:,} pixels → {len(values):,} after outlier filtering")
            else:
                print(f"  {forest_name}: {original_count:,} pixels")
            
            # Sample if very large
            if len(values) > max_sample_size:
                np.random.seed(42)
                indices = np.random.choice(len(values), max_sample_size, replace=False)
                sampled_values = values[indices]
                print(f"    Sampled {max_sample_size:,} from {len(values):,} points")
                values = sampled_values
            
            # Store data
            forest_data[forest_name] = values
            
            # Calculate statistics
            stats_dict = {
                'Forest Type': forest_name,
                'Count': original_count,
                'Mean': np.mean(values),
                'Median': np.median(values),
                'Std Dev': np.std(values),
                'Min': np.min(values),
                'Max': np.max(values),
                '10th Percentile': np.percentile(values, 10),
                '25th Percentile': np.percentile(values, 25),
                '75th Percentile': np.percentile(values, 75),
                '90th Percentile': np.percentile(values, 90),
                'Skewness': stats.skew(values),
                'Kurtosis': stats.kurtosis(values)
            }
            forest_stats.append(stats_dict)
        
        # Create DataFrame with statistics
        stats_df = pd.DataFrame(forest_stats)
        
        # Save statistics to CSV
        stats_csv_path = os.path.join(output_dir, "sap_velocity_distribution_statistics.csv")
        stats_df.to_csv(stats_csv_path, index=False)
        print(f"\nStatistics saved to {stats_csv_path}")
        
        # Sort forest types by mean sap velocity
        forest_order = stats_df.sort_values('Mean', ascending=False)['Forest Type'].tolist()
        
        # --- CREATE VISUALIZATIONS ---
        
        # 1. Combined visualization with histograms and KDE
        print("\nCreating combined distribution visualization...")
        plt.figure(figsize=(15, 10))
        
        # Create color palette
        forest_colors =  {
        "ENF - Evergreen Needleleaf": "#2E8B57",  # Sea Green
        "EBF - Evergreen Broadleaf": "#228B22",   # Forest Green
        "DNF - Deciduous Needleleaf": "#6B8E23",  # Olive Drab
        "DBF - Deciduous Broadleaf": "#32CD32",   # Lime Green
        "MF - Mixed Forest": "#3CB371",           # Medium Sea Green
        "WSA - Woody Savannas": "#8FBC8F",        # Dark Sea Green
        "SAV - Savannas": "#9ACD32"               # Yellow Green
    }
        
        # Plot each distribution
        for i, forest_name in enumerate(forest_order):
            values = forest_data[forest_name]
            color = forest_colors[forest_name]
            
            # Add KDE plot
            sns.kdeplot(
                values, 
                label=f"{forest_name} (n={len(values):,})",
                color=color,
                alpha=0.7,
                linewidth=2
            )
        
        # Add labels and title
        plt.title('Sap Velocity Distribution by Forest Type', fontsize=16)
        plt.xlabel('Sap Velocity (kg / cm² • h)', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.xlim(0, stats_df['90th Percentile'].max() * 1.2)  # Limit x-axis to include 90th percentile
        
        # Add legend with better formatting
        plt.legend(title='Forest Type (IGBP)', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Add mean markers
        for forest_name in forest_order:
            mean_value = stats_df[stats_df['Forest Type'] == forest_name]['Mean'].values[0]
            plt.axvline(
                mean_value, 
                color=forest_colors[forest_name], 
                linestyle='--', 
                alpha=0.5
            )
        
        # Improve layout
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, "distributions", "sap_velocity_distribution_combined.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Combined distribution plot saved to {output_path}")
        plt.close()
        
        # 2. Individual distribution plots for each forest type
        print("Creating individual distribution plots...")
        for forest_name in forest_order:
            values = forest_data[forest_name]
            color = forest_colors[forest_name]
            
            # Create plot with both histogram and KDE
            plt.figure(figsize=(10, 6))
            
            # Plot histogram
            sns.histplot(
                values, 
                kde=True, 
                color=color,
                alpha=0.7,
                bins=50
            )
            
            # Get statistics for this forest type
            forest_stats = stats_df[stats_df['Forest Type'] == forest_name].iloc[0]
            
            # Add vertical lines for key statistics
            plt.axvline(forest_stats['Mean'], color='red', linestyle='-', label=f"Mean: {forest_stats['Mean']:.2f}")
            plt.axvline(forest_stats['Median'], color='blue', linestyle='--', label=f"Median: {forest_stats['Median']:.2f}")
            
            # Add shaded area for 25-75 percentile range
            plt.axvspan(
                forest_stats['25th Percentile'], 
                forest_stats['75th Percentile'], 
                alpha=0.2, 
                color='gray', 
                label='IQR (25th-75th)'
            )
            
            # Add labels and title
            plt.title(f'Sap Velocity Distribution: {forest_name}', fontsize=16)
            plt.xlabel('Sap Velocity (kg / cm² • h)', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            
            # Add legend
            plt.legend(loc='upper right')
            
            # Add text with statistics
            stats_text = (
                f"Mean = {forest_stats['Mean']:.2f}\n"
                f"Median = {forest_stats['Median']:.2f}\n"
                f"Std Dev = {forest_stats['Std Dev']:.2f}\n"
                f"Range = {forest_stats['Min']:.2f} - {forest_stats['Max']:.2f}\n"
                f"Skewness = {forest_stats['Skewness']:.2f}"
            )
            
            # Place text in upper left if distribution is right-skewed, otherwise upper right
            text_x = 0.02 if forest_stats['Skewness'] > 0 else 0.7
            plt.text(
                text_x, 0.95, 
                stats_text, 
                transform=plt.gca().transAxes, 
                fontsize=12,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            # Improve layout
            plt.tight_layout()
            
            # Save figure
            forest_name_safe = forest_name.replace(" - ", "_").replace(" ", "_")
            output_path = os.path.join(output_dir, "distributions", f"sap_velocity_distribution_{forest_name_safe}.png")
            plt.savefig(output_path, dpi=300)
            plt.close()
        
        print(f"Individual distribution plots saved to {os.path.join(output_dir, 'distributions')}")
        
        # 3. Create violin plot comparison
        print("Creating violin plot comparison...")
        
        # Prepare data for violin plot
        violin_data = []
        for forest_name in forest_order:
            values = forest_data[forest_name]
            
            # Sample if extremely large for better violin plotting
            if len(values) > 20000:
                np.random.seed(42)
                indices = np.random.choice(len(values), 20000, replace=False)
                values = values[indices]
            
            for val in values:
                violin_data.append({
                    'Forest Type': forest_name,
                    'Sap Velocity': val
                })
        
        violin_df = pd.DataFrame(violin_data)
        
        # Create violin plot
        plt.figure(figsize=(14, 8))
        
        # Set seaborn style
        sns.set_style("whitegrid")
        
        # Create violin plot with embedded box plot
        ax = sns.violinplot(
            x='Forest Type', 
            y='Sap Velocity',
            data=violin_df,
            order=forest_order,
            palette=forest_colors,
            inner='box',  # Show box plot inside violin
            cut=0,        # Don't extend beyond data range
            scale='width' # Scale violins to have same width
        )
        
        # Add labels and title
        plt.title('Distribution of Sap Velocity by Forest Type', fontsize=16)
        plt.xlabel('Forest Type (IGBP)', fontsize=14)
        plt.ylabel('Sap Velocity (kg / cm² • h)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # Add sample size annotations
        for i, forest_name in enumerate(forest_order):
            count = stats_df[stats_df['Forest Type'] == forest_name]['Count'].values[0]
            plt.text(
                i, 
                -0.2,
                f'n={count:,}',
                ha='center',
                fontsize=10,
                color='black',
                transform=ax.get_xaxis_transform()
            )
        
        # Ensure layout fits well
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, "sap_velocity_violin_plot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Violin plot saved to {output_path}")
        plt.close()
        
        # 4. Create boxplot showing distribution
        print("Creating boxplot comparison...")
        plt.figure(figsize=(8, 8))

        # Set seaborn style
        sns.set_style("whitegrid")
        sns.set_style("ticks")
        palette =  sns.color_palette("husl", len(forest_order))
        # Create boxplot
        ax = sns.boxplot(
            x='Forest Type', 
            y='Sap Velocity',
            data=violin_df,
            width=0.4,
            fill= False,
            order=forest_order,
            palette=forest_colors,
            showfliers=False,  # Hide outlier points for clarity
            boxprops=dict(alpha=0.5),  # Box fill opacity
            whiskerprops=dict(alpha=0.5),  # Whisker lines opacity
            capprops=dict(alpha=0.5),  # Cap lines opacity
            medianprops=dict(alpha=0.5) # Median line opacity
        )
        
        # Add labels and title
        plt.title('Distribution of Sap Velocity by Forest Type', fontsize=16)
        plt.xlabel('Forest Type (IGBP)', fontsize=14)
        plt.ylabel('Sap Velocity (kg / cm² • h)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        

        
        # Ensure layout fits well
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, "sap_velocity_boxplot_raw_data.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Raw data boxplot saved to {output_path}")
        plt.close()
        
        # 5. Create quantile comparison
        print("Creating quantile comparison plot...")
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Set up positions for all forest types
        positions = range(len(forest_order))
        
        # Create a color palette
        colors = sns.color_palette("husl", len(forest_order))
        forest_colors = dict(zip(forest_order, colors))
        
        # For each forest type, plot a horizontal line showing the range from min to max
        for i, forest_name in enumerate(forest_order):
            row = stats_df[stats_df['Forest Type'] == forest_name].iloc[0]
            
            # Plot min to max range
            plt.plot([row['Min'], row['Max']], [i, i], '-', 
                    color=forest_colors[forest_name], alpha=0.5, linewidth=1.5)
            
            # Plot 10th to 90th percentile range (thicker line)
            plt.plot([row['10th Percentile'], row['90th Percentile']], [i, i], '-', 
                    color=forest_colors[forest_name], alpha=0.7, linewidth=4)
            
            # Plot 25th to 75th percentile range (even thicker line)
            plt.plot([row['25th Percentile'], row['75th Percentile']], [i, i], '-', 
                    color=forest_colors[forest_name], alpha=0.9, linewidth=8)
            
            # Plot median as a circle
            plt.plot(row['Median'], i, 'o', 
                    color=forest_colors[forest_name], markersize=10)
            
            # Plot mean as a diamond
            plt.plot(row['Mean'], i, 'D', 
                    color='red', markersize=7)
        
        # Add labels and title
        plt.yticks(positions, forest_order)
        plt.xlabel('Sap Velocity (kg / cm² • h)', fontsize=14)
        plt.title('Sap Velocity Distribution Comparison Across Forest Types', fontsize=16)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gray', alpha=0.5, lw=1.5, label='Min-Max Range'),
            Line2D([0], [0], color='gray', alpha=0.7, lw=4, label='10th-90th Percentile'),
            Line2D([0], [0], color='gray', alpha=0.9, lw=8, label='25th-75th Percentile (IQR)'),
            Line2D([0], [0], marker='o', color='gray', label='Median', markersize=10, linestyle='None'),
            Line2D([0], [0], marker='D', color='red', label='Mean', markersize=7, linestyle='None')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Add grid lines
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Add sample sizes
        for i, forest_name in enumerate(forest_order):
            row = stats_df[stats_df['Forest Type'] == forest_name].iloc[0]
            plt.text(
                plt.xlim()[1] * 1.01, 
                i,
                f"n={row['Count']:,}",
                va='center',
                fontsize=10
            )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, "sap_velocity_quantile_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Quantile comparison plot saved to {output_path}")
        plt.close()
        
        print("\nDistribution analysis and visualization complete!")
        return stats_df
        
    except Exception as e:
        print(f"ERROR in distribution analysis: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def create_absolute_value_charts(df, output_dir):
    """
    Create charts showing the absolute mean values for each forest type within climate zones.
    The first chart is a grouped bar chart (already correct in your original code).
    The second chart is a modified 'stacked' bar chart that shows absolute values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the analyzed data
    output_dir : str
        Directory to save output figures
    """
    if df.empty:
        print("No data available for creating charts")
        return
    
    # Define a nice color palette for forest types
    forest_colors = {
        "ENF - Evergreen Needleleaf": "#2E8B57",  # Sea Green
        "EBF - Evergreen Broadleaf": "#228B22",   # Forest Green
        "DNF - Deciduous Needleleaf": "#6B8E23",  # Olive Drab
        "DBF - Deciduous Broadleaf": "#32CD32",   # Lime Green
        "MF - Mixed Forest": "#3CB371",           # Medium Sea Green
        "WSA - Woody Savannas": "#8FBC8F",        # Dark Sea Green
        "SAV - Savannas": "#9ACD32"               # Yellow Green
    }
    
    # Create a "pseudo-stacked" bar chart that shows absolute values
    try:
        sns.set(style="ticks")
        plt.figure(figsize=(8, 8))
        
        # Sort climate zones in a logical order
        climate_order = [
            "A - Tropical", 
            "B - Arid", 
            "C - Temperate", 
            "D - Continental", 
            "E - Polar"
        ]
        
        # Ensure we only use climate zones that are in our data
        available_climate_zones = df['Climate Zone'].unique()
        climate_zones = [zone for zone in climate_order if zone in available_climate_zones]
        
        # If there are zones in the data that aren't in our predefined order, add them at the end
        missing_zones = list(set(available_climate_zones) - set(climate_zones))
        climate_zones.extend(missing_zones)
        
        # Get all forest types
        forest_types = df['Forest Type'].unique()
        
        # Initialize the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Set up the x positions for bars
        x = np.arange(len(climate_zones))
        
        # Initialize a y-offset dictionary for each climate zone to track the starting point
        # of each forest type's contribution (all start at 0)
        y_offsets = {zone: 0 for zone in climate_zones}
        
        # Create a separate bar for each forest type in each climate zone
        # but give the appearance of a stacked bar chart
        for forest_type in forest_types:
            # Get data for this forest type
            heights = []
            for zone in climate_zones:
                subset = df[(df['Climate Zone'] == zone) & (df['Forest Type'] == forest_type)]
                value = subset['Sap Velocity Mean'].mean() if not subset.empty else 0
                heights.append(value)
            
            # Use the forest color
            color = forest_colors.get(forest_type, "#AFAFAF")
            
            # Plot the bars for this forest type
            ax.bar(x, heights, label=forest_type, color=color, edgecolor='white', linewidth=0.5, alpha=0.5, width=0.5)
        
        # Add labels and title
        ax.set_title('Average Sap Velocity by Forest Type within Climate Zones', fontsize=16)
        ax.set_xlabel('Köppen-Geiger Climate Zone', fontsize=14)
        ax.set_ylabel('Average Sap Velocity (kg / cm² • h)', fontsize=14)
        
        # Set the x-ticks
        ax.set_xticks(x)
        ax.set_xticklabels(climate_zones, rotation=45, ha='right')
        

        # Add a legend
        ax.legend(title='Forest Type (IGBP)', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(output_dir, "sap_velocity_absolute_values.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Absolute value chart saved to {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error creating absolute value chart: {e}")
        import traceback
        traceback.print_exc()
    
    # Create alternative visualization: side-by-side "stacked" bar chart
    try:
        # Create a visualization that places bars side by side for each climate zone
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Data preparation
        data_for_plot = {}
        for zone in climate_zones:
            zone_data = []
            for forest_type in forest_types:
                subset = df[(df['Climate Zone'] == zone) & (df['Forest Type'] == forest_type)]
                value = subset['Sap Velocity Mean'].mean() if not subset.empty else 0
                if value > 0:  # Only include non-zero values
                    zone_data.append((forest_type, value))
            
            # Sort by value (optional)
            zone_data.sort(key=lambda x: x[1], reverse=True)
            data_for_plot[zone] = zone_data
        
        # Set up the plot
        x_positions = []
        x_labels = []
        bar_colors = []
        bar_heights = []
        bar_labels = []
        
        # Position for each climate zone
        zone_width = 1.5  # Width allocated for each climate zone
        bar_width = 0.8   # Width of individual bars
        
        current_x = 0
        zone_positions = {}  # Store the center position of each climate zone
        
        for zone in climate_zones:
            zone_data = data_for_plot[zone]
            num_bars = len(zone_data)
            
            if num_bars == 0:
                current_x += zone_width
                continue
                
            # Calculate the start position for this zone's bars
            start_x = current_x
            
            # Store the center position of this climate zone
            zone_positions[zone] = start_x + (num_bars * bar_width) / 2
            
            # Create a bar for each forest type in this zone
            for i, (forest_type, value) in enumerate(zone_data):
                x_position = start_x + i * bar_width
                x_positions.append(x_position)
                bar_heights.append(value)
                bar_colors.append(forest_colors.get(forest_type, "#AFAFAF"))
                bar_labels.append(forest_type)
            
            # Update the current x position
            current_x = start_x + num_bars * bar_width + 0.5  # Add gap between zones
        
        # Create the bars
        bars = ax.bar(x_positions, bar_heights, width=bar_width, color=bar_colors, 
                     edgecolor='white', linewidth=0.5)
        
        # Create a legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=forest_colors.get(ft, "#AFAFAF"), 
                                edgecolor='white', label=ft) 
                          for ft in forest_types if ft in bar_labels]
        ax.legend(handles=legend_elements, title='Forest Type (IGBP)', 
                 bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add zone labels beneath the bars
        for zone, pos in zone_positions.items():
            ax.text(pos, -1, zone, ha='center', va='top', fontsize=12, rotation=45)
        
        # Don't show individual x-ticks
        ax.set_xticks([])
        
        # Add labels and title
        ax.set_title('Average Sap Velocity by Forest Type within Climate Zones', fontsize=16)
        ax.set_xlabel('Köppen-Geiger Climate Zone', fontsize=14)
        ax.set_ylabel('Average Sap Velocity (kg / cm² • h)', fontsize=14)
        
        # Add some padding at the bottom for the rotated labels
        plt.subplots_adjust(bottom=0.2)
        
        # Save the figure
        output_path = os.path.join(output_dir, "sap_velocity_side_by_side.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Side-by-side visualization saved to {output_path}")
        plt.close()
        
    except Exception as e:
        print(f"Error creating side-by-side visualization: {e}")
        import traceback
        traceback.print_exc()
        



def create_alternative_visualizations(df, output_dir):
    """
    Create alternative visualizations that might be more robust than stacked bar charts.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the analyzed data
    output_dir : str
        Directory to save output figures
    """
    if df.empty:
        print("No data available for creating alternative visualizations")
        return
    
    # 1. Create a grouped bar chart (instead of stacked)
    try:
        plt.figure(figsize=(14, 8))
        
        # Group by Climate Zone and Forest Type, then calculate mean
        pivot_df = df.pivot_table(
            index='Climate Zone',
            columns='Forest Type',
            values='Sap Velocity Mean',
            fill_value=0
        )
        
        # Set up the plot
        ax = pivot_df.plot(kind='bar', figsize=(14, 8))
        
        # Add labels and title
        plt.title('Average Sap Velocity by Climate Zone and Forest Type', fontsize=16)
        plt.xlabel('Köppen-Geiger Climate Zone', fontsize=14)
        plt.ylabel('Average Sap Velocity (kg / cm² • h)', fontsize=14)
        
        # Improve readability
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.legend(title='Forest Type (IGBP)', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Ensure layout fits well
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, "sap_velocity_grouped_bar.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Grouped bar chart saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"Error creating grouped bar chart: {e}")
    
    
    # 3. Create a facet grid of histograms for each climate zone
    try:
        # Check if we have enough unique climate zones for a meaningful facet grid
        if df['Climate Zone'].nunique() >= 2:
            # Set up a facet grid for histograms
            g = sns.FacetGrid(df, col="Climate Zone", col_wrap=2, height=4, aspect=1.5)
            
            # Map the histogram function to each facet
            g.map(sns.histplot, "Sap Velocity Mean", kde=True)
            
            # Add overall title
            g.fig.suptitle('Distribution of Sap Velocity in Different Climate Zones', 
                          fontsize=16, y=1.02)
            
            # Save figure
            output_path = os.path.join(output_dir, "sap_velocity_histograms_by_climate.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Histogram facet grid saved to {output_path}")
            plt.close(g.fig)
        else:
            print("Not enough climate zones for a facet grid")
    except Exception as e:
        print(f"Error creating facet grid: {e}")
    
    # 4. Create a scatter plot of sap velocity vs count (sample size)
    try:
        plt.figure(figsize=(10, 8))
        
        # Create a scatter plot with point size proportional to count
        scatter = plt.scatter(
            df['Count'], 
            df['Sap Velocity Mean'],
            c=df.index,  # Color by index to differentiate points
            s=df['Count'].apply(lambda x: min(x/100, 500)),  # Size by count, but cap the size
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5
        )
        
        # Add labels for each point
        for i, row in df.iterrows():
            plt.annotate(
                f"{row['Climate Zone'][:1]}-{row['Forest Type'][:3]}", 
                (row['Count'], row['Sap Velocity Mean']),
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=8
            )
        
        # Add axis labels and title
        plt.title('Sap Velocity vs. Sample Size', fontsize=16)
        plt.xlabel('Number of Data Points', fontsize=14)
        plt.ylabel('Average Sap Velocity (kg / cm² • h)', fontsize=14)
        
        # Use log scale for x-axis if data spans multiple orders of magnitude
        if df['Count'].max() / df['Count'].min() > 100:
            plt.xscale('log')
            plt.xlabel('Number of Data Points (log scale)', fontsize=14)
        
        # Add grid
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Ensure tight layout
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, "sap_velocity_vs_sample_size.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to {output_path}")
        plt.close()
    except Exception as e:
        print(f"Error creating scatter plot: {e}")
    
        


    print("Alternative visualizations created successfully")



if __name__ == "__main__":
    # File paths
    sap_velocity_path = "outputs/maps/sap_velocity_map_global_midday_v1.tif"
    koppen_path = "data/raw/grided/Beck_KG_V1/Beck_KG_V1_present_0p083.tif"
    
    # IGBP data path - YOU MUST SPECIFY THE CORRECT PATH TO YOUR IGBP DATA
    
    
    # Target resolution for resampling (in degrees)
    target_resolution = 0.1
    
    # Run the analysis
    analyze_sap_velocity_by_climate_and_forest(
        sap_velocity_path=sap_velocity_path,
        koppen_path=koppen_path,
        target_resolution=target_resolution
    )
