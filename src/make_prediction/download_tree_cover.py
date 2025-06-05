import ee
import os
import time
import argparse
import sys # Import sys for exit
import traceback # Keep traceback for detailed error reporting
import pandas as pd # Moved import to top for clarity

# --- Configuration ---
# Default Project ID can be overridden by command-line argument
DEFAULT_PROJECT_ID = 'era5download-447713'
DEFAULT_EXPORT_FOLDER = 'Earth_Engine_Exports'

# --- Earth Engine Initialization ---
def initialize_ee(project_id=DEFAULT_PROJECT_ID):
    """
    Initialize Earth Engine API using a specific project ID.
    May prompt for authentication if not already authenticated.
    """
    try:
        # Try to initialize directly. If credentials exist, this might work.
        # If it fails, or if authentication is explicitly needed, Authenticate() will be triggered.
        try:
             ee.Initialize(project=project_id, opt_url='https://earthengine-highvolume.googleapis.com')
             print(f"Earth Engine initialized successfully using project: {project_id}")
             return True
        except Exception as initial_init_error:
             print(f"Initial EE Initialize failed ({initial_init_error}), attempting Authenticate then Initialize.")
             ee.Authenticate()
             ee.Initialize(project=project_id, opt_url='https://earthengine-highvolume.googleapis.com')
             print(f"Earth Engine authenticated and initialized successfully using project: {project_id}")
             return True
    except Exception as e:
        print(f"ERROR - Failed to initialize Earth Engine: {e}", file=sys.stderr)
        print("Please ensure you have authenticated via 'earthengine authenticate'", file=sys.stderr)
        print(f"And that the project '{project_id}' is accessible and activated for Earth Engine.", file=sys.stderr)
        # exit(1) # Exit if initialization fails
        return False # Return False instead of exiting to allow potential fallback

# --- Helper Function to Get Median Image ---
# Extracted logic to avoid repetition and ensure it runs only once if needed
def get_tree_cover_median_image(project_id=DEFAULT_PROJECT_ID):
    """
    Computes the median tree cover image across GFCC TC v3 epochs (2000, 2005, 2010, 2015).

    Returns:
        ee.Image: The median tree cover image or None if data retrieval fails.
    """
    print("Attempting to compute median tree cover image across all epochs...")
    try:
        # Ensure EE is initialized (useful if this function is called independently)
        # initialize_ee(project_id) # Removed: Assume EE is initialized before calling functions needing it.

        dataset = ee.ImageCollection('NASA/MEASURES/GFCC/TC/v3')
        years = [2000, 2005, 2010, 2015]
        annual_images = []
        any_data_found = False

        for year in years:
            year_start = f"{year-2}-01-01"
            year_end = f"{year+2}-12-31"
            year_data = dataset.filter(ee.Filter.date(year_start, year_end))

            # Check size server-side first before getting info
            size = year_data.size()
            if size.getInfo() > 0: # Get info only if needed for the check
                tree_canopy = year_data.select('tree_canopy_cover').mean()
                annual_images.append(tree_canopy)
                # Avoid getInfo in loop for printing count, get it once if needed later or trust the process
                print(f"Found data for epoch {year}. Added mean image to collection.")
                any_data_found = True
            else:
                print(f"Warning: No data found for epoch {year}")

        if not any_data_found:
            print("ERROR: No data found for any epoch. Cannot compute median.", file=sys.stderr)
            return None

        all_years_collection = ee.ImageCollection.fromImages(annual_images)
        tree_canopy_median = all_years_collection.median()
        print("Successfully computed median tree cover image.")
        return tree_canopy_median

    except ee.EEException as e:
        print(f"ERROR during Earth Engine operation while computing median image: {e}", file=sys.stderr)
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"ERROR - Unexpected error computing median image: {e}", file=sys.stderr)
        traceback.print_exc()
        return None


# --- Threshold Determination (Optimized) ---
def determine_threshold_from_points(csv_path, tree_canopy_median, print_details=True, scale_meters=1000):
    """
    Determine tree cover threshold by sampling values at points from a CSV file.
    Uses server-side aggregation for efficiency.

    Args:
        csv_path (str): Path to CSV file with longitude and latitude columns.
        tree_canopy_median (ee.Image): Pre-computed median tree cover Image. *Required*.
        print_details (bool): Whether to print details about each sample point.
        scale_meters (int): Scale in meters for sampling (default: 1000).

    Returns:
        float: The minimum tree cover value across all points, or None on error.
    """
    if tree_canopy_median is None:
         print("ERROR: determine_threshold_from_points requires a pre-computed tree_canopy_median image.", file=sys.stderr)
         return None

    try:
        import pandas as pd
        # ee is imported globally now
    except ImportError:
        print("ERROR: Required package 'pandas' not found. Please install: pip install pandas", file=sys.stderr)
        return None

    # Read the CSV file with point locations
    try:
        print(f"Reading points from {csv_path}...")
        df = pd.read_csv(csv_path)
        lon_cols = [col for col in df.columns if col.lower() in ('lon', 'longitude', 'long', 'x')]
        lat_cols = [col for col in df.columns if col.lower() in ('lat', 'latitude', 'y')]

        if not lon_cols or not lat_cols:
            print(f"ERROR: CSV '{csv_path}' must contain columns for longitude and latitude.", file=sys.stderr)
            print(f"Found columns: {', '.join(df.columns)}", file=sys.stderr)
            return None

        lon_col = lon_cols[0]
        lat_col = lat_cols[0]
        print(f"Using columns: '{lon_col}' for longitude, '{lat_col}' for latitude")

        # Create EE FeatureCollection efficiently
        # Ensure coordinates are float numbers
        df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
        df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
        df = df.dropna(subset=[lon_col, lat_col]) # Remove rows where conversion failed
        if df.empty:
             print(f"ERROR: No valid coordinate pairs found in {csv_path} after cleaning.", file=sys.stderr)
             return None

        features = df.apply(lambda row: ee.Feature(ee.Geometry.Point(row[lon_col], row[lat_col])), axis=1).tolist()
        if not features:
             print(f"ERROR: Could not create any Earth Engine features from {csv_path}.", file=sys.stderr)
             return None
        fc = ee.FeatureCollection(features)

        # Sample the tree canopy cover at each point
        print(f"Sampling tree cover at {fc.size().getInfo()} points with scale {scale_meters}m...") # getInfo on size is acceptable here
        samples = tree_canopy_median.sampleRegions(
            collection=fc,
            scale=scale_meters,
            geometries=True # Keep geometries if print_details is True
        )

        # --- Efficient Minimum Calculation ---
        # Use aggregate_min for server-side calculation
        min_info = samples.aggregate_min('tree_canopy_cover').getInfo()

        if min_info is None:
             # This can happen if no points were sampled successfully (e.g., all points fall in nodata areas)
             print("Warning: Could not retrieve minimum value. No points sampled successfully or all sampled values were null.")
             # Decide on behavior: return None, or a default? Let's return None to indicate failure.
             # Check sample count to be sure
             sample_count = samples.size().getInfo()
             print(f"Number of features in samples collection: {sample_count}")
             if sample_count == 0:
                  print("Reason: Zero features were sampled (e.g., points outside image footprint or in masked areas).")
             else:
                  print("Reason: All sampled features had null values for 'tree_canopy_cover'.")
             return None # Indicate failure to determine threshold

        min_value = float(min_info) # Convert to float
        print(f"\nMinimum tree cover value across all points (server-side calculation): {min_value:.2f}%")

        # --- Print Details (if requested) ---
        # This part still needs to fetch data, but only if details are needed.
        # Fetch data ONCE if details are needed.
        if print_details:
            print("Fetching details for sampled points...")
            # Select properties AND geometry for printing
            # Using toList().getInfo() is generally safer than .getInfo() on the collection directly for large numbers of features
            max_points_for_details = 5000 # Limit to avoid huge transfers
            sample_list_info = samples.toList(max_points_for_details).getInfo() # List of dicts

            if not sample_list_info:
                 print("Could not retrieve sample details (maybe too many points or sampling issues).")
            else:
                 print(f"Details for first {len(sample_list_info)} sampled points:")
                 point_counter = 0
                 for feature_info in sample_list_info:
                     point_counter += 1
                     props = feature_info.get('properties', {})
                     geom = feature_info.get('geometry', {})
                     coords = geom.get('coordinates', [None, None])
                     tree_cover = props.get('tree_canopy_cover', None) # Get value, default to None

                     if tree_cover is not None:
                         print(f"  Point {point_counter}: lon={coords[0]:.5f}, lat={coords[1]:.5f}, tree_cover={tree_cover:.2f}%")
                     else:
                         print(f"  Point {point_counter}: lon={coords[0]:.5f}, lat={coords[1]:.5f}, tree_cover=NULL")

        # Adjust threshold if minimum is zero
        if min_value == 0:
            print("Minimum sampled value is 0%. Using 1% as minimum threshold to avoid issues.")
            return 1
        else:
            return min_value

    except ee.EEException as e:
        print(f"ERROR during Earth Engine operation in determine_threshold: {e}", file=sys.stderr)
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"ERROR - Unexpected error determining threshold from points: {e}", file=sys.stderr)
        traceback.print_exc()
        return None

# --- Global Tree Cover Download ---
def download_global_tree_cover(output_dir, export_folder, threshold=None, resolution_km=9, csv_path=None):
    """
    Calculates median tree cover and exports it along with a thresholded mask.
    """
    os.makedirs(output_dir, exist_ok=True) # Ensure local output dir exists (though not used directly here)

    # Get the median image
    # Assuming EE is initialized before calling this function
    tree_canopy_median = get_tree_cover_median_image()
    if tree_canopy_median is None:
        print("ERROR: Failed to compute the base median tree cover image. Cannot proceed.", file=sys.stderr)
        return None # Indicate failure

    # Determine threshold from points if CSV path is provided
    if csv_path:
        print("\n--- Determining Dynamic Threshold ---")
        # Pass the pre-calculated median image
        dynamic_threshold = determine_threshold_from_points(csv_path, tree_canopy_median)
        if dynamic_threshold is not None:
            threshold = dynamic_threshold
            print(f"Using dynamically determined threshold: {threshold:.2f}%")
        else:
            print(f"Warning: Failed to determine threshold from points. Using default threshold: {threshold}%")
        print("--- End Dynamic Threshold Determination ---\n")


    # Create a binary mask for areas with trees above the threshold
    # Ensure the median image is masked where it might be null before thresholding
    # (median() should handle this, but explicit masking is safer)
    # tree_canopy_median = tree_canopy_median.unmask(0) # Or handle mask appropriately
    tree_mask = tree_canopy_median.gte(threshold).unmask(0).rename('tree_existence') # Greater than or equal to threshold

    # Set the export scale in meters
    scale_meters = resolution_km * 1000

    # Define regions (consider making configurable if needed)
    regions = {
        'NorthAmerica': ee.Geometry.Rectangle([-180, 15, -50, 85]),
        'SouthAmerica': ee.Geometry.Rectangle([-90, -60, -30, 15]),
        'Europe':       ee.Geometry.Rectangle([-15, 35, 40, 75]),
        'Africa':       ee.Geometry.Rectangle([-20, -40, 55, 35]),
        'Asia':         ee.Geometry.Rectangle([40, 0, 150, 75]),
        'Oceania':      ee.Geometry.Rectangle([100, -50, 180, 0])
        # Add more regions if needed, e.g., Arctic, specific islands
    }

    print(f"\n--- Starting Exports (Scale: {resolution_km}km, Threshold: {threshold:.2f}%) ---")
    print(f"Exporting to Google Drive Folder: '{export_folder}'")
    export_tasks = {}
    for region_name, geometry in regions.items():
        try:
            # Export the binary tree mask for this region
            mask_task_name = f"Tree_Existence_{region_name}_{resolution_km}km_thresh{threshold:.1f}"
            mask_export_task = ee.batch.Export.image.toDrive(
                image=tree_mask.clip(geometry).byte(), # Clip and cast to byte for smaller file
                description=mask_task_name,
                folder=export_folder,
                scale=scale_meters,
                region=geometry,
                maxPixels=1e13,
                fileFormat='GeoTIFF'
                # Add fileDimensions or shardSize if exports are too large for Drive
            )

            # Also export the original median tree cover percentage
            canopy_task_name = f"Tree_Canopy_Median_{region_name}_{resolution_km}km"
            canopy_export_task = ee.batch.Export.image.toDrive(
                image=tree_canopy_median.clip(geometry).float(), # Clip and cast
                description=canopy_task_name,
                folder=export_folder,
                scale=scale_meters,
                region=geometry,
                maxPixels=1e13,
                fileFormat='GeoTIFF'
            )

            # Start the tasks
            mask_export_task.start()
            canopy_export_task.start()

            export_tasks[region_name] = {
                'binary_mask': mask_export_task,
                'canopy_median': canopy_export_task
            }

            print(f"Started exports for {region_name}: '{mask_task_name}' and '{canopy_task_name}'")

        except ee.EEException as e:
             print(f"ERROR submitting export task for region {region_name}: {e}", file=sys.stderr)
             # Continue to next region if one fails? Or stop? Currently continues.
        except Exception as e:
             print(f"ERROR - Unexpected error setting up export for {region_name}: {e}", file=sys.stderr)

    print("\nAll export tasks have been submitted.")
    print("Monitor the tasks in the Earth Engine Code Editor Task tab:")
    print("https://code.earthengine.google.com/tasks")
    print(f"\nFiles will appear in the '{export_folder}' folder in your Google Drive.")
    print("After downloading the GeoTIFF files, use the --convert step.")

    return export_tasks # Return dict of submitted tasks

# --- GeoTIFF to Shapefile Conversion ---
def convert_to_shapefile(input_dir, output_dir):
    """
    Convert downloaded GeoTIFF files to shapefiles.
    Primarily designed for the binary Tree_Existence GeoTIFFs (value 1 = tree).
    """
    try:
        import rasterio
        import geopandas as gpd
        from rasterio import features
        import numpy as np
    except ImportError:
        print("ERROR: Required packages not found. Please install: rasterio geopandas", file=sys.stderr)
        print("pip install rasterio geopandas", file=sys.stderr)
        return

    os.makedirs(output_dir, exist_ok=True)
    geotiff_files = [f for f in os.listdir(input_dir) if f.endswith('.tif') or f.endswith('.TIF')]

    if not geotiff_files:
        print(f"Warning: No GeoTIFF files found in {input_dir}")
        return

    print(f"\n--- Converting GeoTIFFs in '{input_dir}' to Shapefiles in '{output_dir}' ---")
    files_processed = 0
    files_skipped = 0
    for geotiff_file in geotiff_files:
        geotiff_path = os.path.join(input_dir, geotiff_file)
        # Create a more specific shapefile name perhaps
        shapefile_name = os.path.basename(geotiff_file).replace('.tif', '').replace('.TIF', '') + '.shp'
        shapefile_path = os.path.join(output_dir, shapefile_name)

        print(f"Processing {geotiff_file}...")
        try:
            with rasterio.open(geotiff_path) as src:
                image = src.read(1)
                transform = src.transform
                crs = src.crs

                # --- Masking Strategy ---
                # This mask assumes the input is the binary Tree_Existence file,
                # where pixel value 1 indicates tree presence.
                # If converting canopy % rasters, you might want a different mask
                # (e.g., based on nodata value or a threshold).
                mask = (image == 1) # Create mask where pixel value is exactly 1

                if not np.any(mask):
                    print(f"  Skipping: No valid data (pixels with value 1) found in {geotiff_file}.")
                    files_skipped += 1
                    continue

                # Extract shapes for pixels with value 1
                # Pass the image itself to get the value property (should be 1)
                shapes = features.shapes(image, mask=mask, transform=transform)

                # Convert shapes to GeoDataFrame features
                # Store the pixel value (should be 1) in properties
                geoms = [{'geometry': geom, 'properties': {'value': val}}
                         for geom, val in shapes] # val will be 1 here

                if not geoms:
                    print(f"  Skipping: No geometries extracted from {geotiff_file} (this shouldn't happen if mask has True values).")
                    files_skipped += 1
                    continue

                gdf = gpd.GeoDataFrame.from_features(geoms, crs=crs)

                # Ensure output directory exists before saving
                os.makedirs(os.path.dirname(shapefile_path), exist_ok=True)
                gdf.to_file(shapefile_path)
                print(f"  Shapefile saved to {shapefile_path}")
                files_processed += 1

        except Exception as e:
            print(f"  ERROR processing {geotiff_file}: {e}", file=sys.stderr)
            files_skipped += 1
            # Optional: print traceback for debugging
            # traceback.print_exc()

    print(f"\nConversion finished. Processed: {files_processed}, Skipped/Errors: {files_skipped}")

# --- Merge Shapefiles ---
def merge_shapefiles(shapefile_dir, output_file):
    """Merge multiple shapefiles into one global shapefile."""
    try:
        import geopandas as gpd
        import glob
        import pandas as pd
    except ImportError:
        print("ERROR: Required packages not found. Please install: geopandas pandas", file=sys.stderr)
        print("pip install geopandas pandas", file=sys.stderr)
        return None # Return None to indicate failure

    all_shapefiles = glob.glob(os.path.join(shapefile_dir, '*.shp'))

    if not all_shapefiles:
        print(f"Warning: No shapefiles (*.shp) found in {shapefile_dir}")
        return None

    print(f"\n--- Merging {len(all_shapefiles)} Shapefiles from '{shapefile_dir}' ---")
    gdfs = []
    valid_files_count = 0
    for shapefile in all_shapefiles:
        try:
            gdf = gpd.read_file(shapefile)
            if not gdf.empty:
                 gdfs.append(gdf)
                 print(f"  Read {shapefile} ({len(gdf)} features)")
                 valid_files_count += 1
            else:
                 print(f"  Skipping empty shapefile: {shapefile}")
        except Exception as e:
            print(f"  ERROR reading {shapefile}: {e}", file=sys.stderr)

    if not gdfs:
        print("ERROR: No valid, non-empty shapefiles could be read. Cannot merge.", file=sys.stderr)
        return None

    print(f"\nConcatenating {valid_files_count} GeoDataFrames...")
    # Note: Concatenation can be memory intensive for many large files
    global_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs) # Use CRS from first GDF

    print(f"Total features in merged file: {len(global_gdf)}")

    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        global_gdf.to_file(output_file)
        print(f"Global shapefile saved successfully: {output_file}")
        return global_gdf # Return the merged GeoDataFrame
    except Exception as e:
        print(f"ERROR saving merged shapefile to {output_file}: {e}", file=sys.stderr)
        traceback.print_exc()
        return None


# --- Plot Global Shapefile ---
def plot_global_shapefile(shapefile_path, output_image='global_tree_coverage.png',
                          figsize=(16, 10), dpi=300, show=True, basemap_provider=None):
    """
    Plot the global tree coverage shapefile with an optional basemap.
    """
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import contextily as ctx
        # numpy and shapely are implicitly used by geopandas/contextily
    except ImportError:
        print("ERROR: Required packages not found. Please install: geopandas matplotlib contextily", file=sys.stderr)
        print("pip install geopandas matplotlib contextily", file=sys.stderr)
        return

    print(f"\n--- Plotting Shapefile: {shapefile_path} ---")
    try:
        gdf = gpd.read_file(shapefile_path)
        if gdf.empty:
             print(f"Warning: Shapefile {shapefile_path} is empty. Cannot plot.")
             return
    except Exception as e:
        print(f"ERROR reading shapefile {shapefile_path}: {e}", file=sys.stderr)
        return

    print(f"Shapefile contains {len(gdf)} features")

    # Use a CRS suitable for global plotting if possible, or stick to Web Mercator for basemaps
    try:
         # Reproject to Web Mercator (EPSG:3857) for compatibility with contextily basemaps
         print("Reprojecting to EPSG:3857 for plotting...")
         gdf_proj = gdf.to_crs(epsg=3857)
         plot_crs_info = "EPSG:3857 (Web Mercator)"
    except Exception as e:
         print(f"Warning: Failed to reproject to EPSG:3857 ({e}). Plotting in original CRS: {gdf.crs}")
         gdf_proj = gdf # Plot in original CRS if reprojection fails
         plot_crs_info = gdf.crs

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.title('Global Tree Coverage', fontsize=16, fontweight='bold')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect('equal') # Try to maintain aspect ratio

    # Plot the tree coverage polygons
    print("Plotting geometries...")
    gdf_proj.plot(ax=ax, color='darkgreen', alpha=0.6, edgecolor='none') # No edgecolor for dense polygons

    # Add a basemap if requested and possible
    if basemap_provider:
        print(f"Adding basemap using provider: {basemap_provider}...")
        try:
            # Ensure plot limits are set reasonably before adding basemap
            minx, miny, maxx, maxy = gdf_proj.total_bounds
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            # Get the actual provider object from contextily
            provider = getattr(ctx.providers, basemap_provider.split('.')[0]).get(basemap_provider.split('.')[1], ctx.providers.OpenStreetMap.Mapnik) # Default fallback
            ctx.add_basemap(ax, source=provider, crs=gdf_proj.crs) # Use the projected CRS
            print("Basemap added.")
        except Exception as e:
            print(f"Warning: Could not add basemap using {basemap_provider}: {e}")
            print("Plotting without basemap.")

    # Add a simple legend
    tree_patch = mpatches.Patch(color='darkgreen', alpha=0.6, label='Tree Coverage (>= Threshold)')
    ax.legend(handles=[tree_patch], loc='lower left', fontsize='small') # Move legend maybe

    # Add credits
    plt.annotate(f'Data: NASA/MEASURES/GFCC/TC/v3 (Processed {time.strftime("%Y-%m-%d")})\nPlotted in {plot_crs_info}',
                 xy=(0.01, 0.01), xycoords='axes fraction', fontsize=7,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5, alpha=0.8))

    # Improve layout
    plt.tight_layout()

    # Save the figure
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_image), exist_ok=True)
        plt.savefig(output_image, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {output_image}")
    except Exception as e:
        print(f"ERROR saving plot to {output_image}: {e}", file=sys.stderr)

    # Show the plot if requested
    if show:
        plt.show()
    else:
        plt.close(fig) # Close the figure explicitly if not showing
import geopandas as gpd
import os

def dissolve_merged_shapefile(input_merged_shp, output_dissolved_shp):
    """
    Dissolves internal boundaries in the merged shapefile,
    typically merging polygons with value=1.

    Args:
        input_merged_shp (str): Path to the merged shapefile from merge_shapefiles.
        output_dissolved_shp (str): Path to save the dissolved shapefile.
    """
    print(f"\n--- Dissolving Boundaries in {input_merged_shp} ---")
    try:
        # Load the merged shapefile
        gdf = gpd.read_file(input_merged_shp)
        print(f"Loaded {len(gdf)} features.")

        if gdf.empty:
            print("Input shapefile is empty, nothing to dissolve.")
            return False

        # Check if 'value' column exists (created by convert_to_shapefile)
        if 'value' not in gdf.columns:
            print("ERROR: 'value' column not found. Cannot dissolve based on tree cover value.", file=sys.stderr)
            print("Ensure the input shapefile was generated by the convert_to_shapefile function.", file=sys.stderr)
            return False

        # Filter for polygons representing tree cover (assuming value=1)
        # Ensure the value column is numeric if necessary
        gdf['value'] = pd.to_numeric(gdf['value'], errors='coerce')
        tree_polygons = gdf[gdf['value'] == 1].copy() # Filter and copy to avoid SettingWithCopyWarning

        if tree_polygons.empty:
            print("No features with value=1 found to dissolve.")
            # Optionally save an empty file or just return
            # For consistency, let's save an empty file with the correct schema
            # Create empty gdf with same schema
            empty_dissolved = gpd.GeoDataFrame(columns=['value', 'geometry'], geometry='geometry', crs=gdf.crs)
            empty_dissolved.to_file(output_dissolved_shp)
            print(f"Saved empty dissolved shapefile (no value=1 features found): {output_dissolved_shp}")
            return True


        print(f"Found {len(tree_polygons)} features with value=1. Dissolving...")

        # Perform the dissolve operation based on the 'value' column
        # All polygons with value=1 will be merged.
        dissolved_gdf = tree_polygons.dissolve(by='value')

        # Reset the index so 'value' becomes a regular column again
        dissolved_gdf = dissolved_gdf.reset_index()

        print(f"Dissolved into {len(dissolved_gdf)} features.") # Should be 1 feature if all had value=1

        # Save the dissolved shapefile
        os.makedirs(os.path.dirname(output_dissolved_shp), exist_ok=True)
        dissolved_gdf.to_file(output_dissolved_shp)
        print(f"Dissolved shapefile saved successfully: {output_dissolved_shp}")
        return True

    except Exception as e:
        print(f"ERROR during dissolve process: {e}", file=sys.stderr)
        traceback.print_exc()
        return False
# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Download, process, and visualize global tree cover data from Google Earth Engine.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    parser.add_argument('--project_id', type=str, default=DEFAULT_PROJECT_ID,
                        help='Google Earth Engine Project ID to use for computations.')

    # Workflow Steps
    parser.add_argument('--download', action='store_true', help='STEP 1: Submit download tasks to Earth Engine.')
    parser.add_argument('--convert', action='store_true', help='STEP 2: Convert downloaded GeoTIFFs to Shapefiles.')
    parser.add_argument('--merge', action='store_true', help='STEP 3: Merge regional Shapefiles into one global file.')
    parser.add_argument('--dissolve', action='store_true', help='STEP 4: Dissolve merged shapefile boundaries.')
    parser.add_argument('--dissolved_shapefile', type=str, default='./data/raw/grided/tree_cover_shapefile_dissolved', help='Output path for the final dissolved global Shapefile.')
    parser.add_argument('--plot', action='store_true', help='STEP 4: Plot the final global Shapefile.')

    # Configuration for Download
    parser.add_argument('--export_folder', type=str, default=DEFAULT_EXPORT_FOLDER,
                        help='Google Drive folder name for Earth Engine exports.')
    parser.add_argument('--threshold', type=float, default=15,
                        help='Tree cover percentage threshold (%%) for binary mask.')
    parser.add_argument('--resolution', type=int, default=9,
                        help='Export resolution in kilometers.')
    parser.add_argument('--points_csv', type=str, default=None,
                        help='Optional CSV file with lon/lat points to dynamically determine threshold.')

    # Configuration for Convert/Merge
    parser.add_argument('--download_dir', type=str, default='geotiff_downloads',
                        help='Local directory where GeoTIFFs are downloaded from Drive (input for --convert).')
    parser.add_argument('--shapefile_dir', type=str, default='shapefiles_output',
                        help='Local directory to save regional Shapefiles (output for --convert, input for --merge).')
    parser.add_argument('--merged_shapefile', type=str, default='./data/raw/grided/tree_cover_shapefile', # Default to None, construct later
                        help='Output path for the final merged global Shapefile (output for --merge, input for --plot).')

    # Configuration for Plot
    parser.add_argument('--plot_output', type=str, default='global_tree_coverage.png',
                        help='Output image file name for the plot.')
    parser.add_argument('--hide_plot', action='store_true',
                        help='Do not display the plot interactively, just save it.')
    parser.add_argument('--basemap', type=str, default='Esri.WorldTerrain', # Example provider string
                        help='Contextily basemap provider (e.g., "OpenStreetMap.Mapnik", "Esri.WorldImagery", None). Set to None or empty to disable.')

    args = parser.parse_args()

    # --- Execute Workflow ---
    ee_initialized = False
    if args.download or args.points_csv: # Need EE for download or dynamic threshold calculation
        print("--- Initializing Earth Engine ---")
        ee_initialized = initialize_ee(args.project_id)
        if not ee_initialized:
             print("FATAL: Earth Engine initialization failed. Exiting.", file=sys.stderr)
             sys.exit(1) # Exit if EE needed but failed

    # STEP 1: Download
    if args.download:
        if not ee_initialized:
             print("FATAL: Earth Engine required for download but not initialized. Exiting.", file=sys.stderr)
             sys.exit(1)
        print("\n--- STEP 1: Submitting Download Tasks ---")
        # Pass output_dir (not really used by download func itself, but keeps pattern)
        # Pass export_folder, threshold, resolution, csv_path
        download_global_tree_cover(
            output_dir=args.download_dir, # Pass local dir for consistency, though not used by download
            export_folder=args.export_folder,
            threshold=args.threshold,
            resolution_km=args.resolution,
            csv_path=args.points_csv
        )
        # Note: Script exits here; user needs to wait for downloads and run again for next steps.
        print("\nDownload tasks submitted. Please wait for completion in Google Drive,")
        print(f"then download the GeoTIFF files to the '{args.download_dir}' directory")
        print("and run this script again with the --convert flag.")

    # STEP 2: Convert
    if args.convert:
        print("\n--- STEP 2: Converting GeoTIFFs to Shapefiles ---")
        convert_to_shapefile(args.download_dir, args.shapefile_dir)

    # STEP 3: Merge
    # Determine default merged shapefile name if not provided
    if args.merged_shapefile is None:
         # Construct default name based on resolution and threshold (might need threshold used in download)
         # For simplicity, just use resolution for now.
         merged_shapefile_path = os.path.join(args.shapefile_dir, f'Global_Tree_Existence_{args.resolution}km_merged.shp')
    else:
         merged_shapefile_path = args.merged_shapefile

    if args.merge:
        print("\n--- STEP 3: Merging Shapefiles ---")
        merge_shapefiles(args.shapefile_dir, merged_shapefile_path)
    if args.dissolve:
        print("\n--- STEP 4: Dissolving Boundaries ---")
        if not os.path.exists(merged_shapefile_path):
            print(f"ERROR: Merged shapefile not found for dissolving: {merged_shapefile_path}", file=sys.stderr)
        else:
            dissolve_merged_shapefile(merged_shapefile_path +'/tree_cover_shapefile.shp', args.dissolved_shapefile)
            # Subsequent steps like plotting should now use dissolved_shapefile_path
    # STEP 4: Plot
    if args.plot:
        print("\n--- STEP 5: Plotting Global Shapefile ---")
        if not os.path.exists(merged_shapefile_path):
            print(f"ERROR: Merged shapefile not found: {merged_shapefile_path}", file=sys.stderr)
            print("Please ensure you have run the --convert and --merge steps successfully,", file=sys.stderr)
            print(f"or provide the correct path using --merged_shapefile.", file=sys.stderr)
        else:
             basemap_provider = args.basemap if args.basemap and args.basemap.lower() != 'none' else None
             plot_global_shapefile(
                 merged_shapefile_path,
                 args.plot_output,
                 show=(not args.hide_plot),
                 basemap_provider=basemap_provider
             )

    # Check if any action was taken
    if not (args.download or args.convert or args.merge or args.plot):
        print("\nNo action specified. Use one or more of --download, --convert, --merge, --plot")
        parser.print_help()

    print("\nScript finished.")