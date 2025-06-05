import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import matplotlib.ticker as mticker
import rasterio
from scipy.ndimage import zoom
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os
from datetime import datetime  # Import datetime for proper date handling
def visualize_sap_velocity_tif(tif_path, output_img=None, title="Mean Sap Velocity Map between 12:00-14:00 in July", 
                              max_size=5000, contrast_reduction=0.8):
    """
    Visualize an existing sap velocity GeoTIFF file with enhanced cartographic elements.
    
    Args:
        tif_path: Path to the input GeoTIFF file
        output_img: Path for the output image file (if None, derived from input filename)
        title: Title for the map
        max_size: Maximum dimension for downsampling
        contrast_reduction: Gamma value for contrast control (0.5-1.5, lower = less contrast)
        
    Returns:
        None - saves the visualization to disk
    """
    if not os.path.exists(tif_path):
        print(f"Error: File not found at {tif_path}")
        return
        
    # Set default output filename if not provided
    if output_img is None:
        base_name = os.path.splitext(tif_path)[0]
        output_img = f"{base_name}_visualization.png"
    
    # Open the GeoTIFF file
    print(f"Opening {tif_path}...")
    try:
        with rasterio.open(tif_path) as src:
            # Read the data
            grid_data = src.read(1)  # Read first band
            
            # Get georeference information
            transform = src.transform
            min_lon = transform[0]  # xmin
            max_lat = transform[3]  # ymax
            
            # Calculate extent based on dimensions and transform
            pixel_width = transform[1]
            pixel_height = abs(transform[5])  # Usually negative
            
            height, width = grid_data.shape
            max_lon = min_lon + width * pixel_width
            min_lat = max_lat - height * pixel_height
            
            # Print information about the raster
            print(f"Raster dimensions: {width}x{height}")
            print(f"Spatial extent: Lon [{min_lon:.6f}, {max_lon:.6f}], " +
                  f"Lat [{min_lat:.6f}, {max_lat:.6f}]")
            
            # Get basic statistics
            valid_data = grid_data[~np.isnan(grid_data)]
            if len(valid_data) > 0:
                print(f"Data range: [{np.min(valid_data):.6f}, {np.max(valid_data):.6f}]")
                print(f"Mean: {np.mean(valid_data):.6f}")
                print(f"Valid data points: {len(valid_data):,} of {grid_data.size:,} " +
                      f"({100*len(valid_data)/grid_data.size:.2f}%)")
            else:
                print("Warning: No valid data found in raster")
    
    except Exception as e:
        print(f"Error reading GeoTIFF file: {e}")
        return
    
    # Set up the extent tuple in the format expected by our visualization
    extent = (min_lat, max_lat, min_lon, max_lon)
    
    # Downsample if grid is too large for plotting
    plot_grid = grid_data
    if height > max_size or width > max_size:
        downsample_factor = max(np.ceil(height / max_size), np.ceil(width / max_size))
        print(f"Downsampling visualization grid ({height}x{width}) by factor {downsample_factor:.1f}")
        zoom_factor = 1.0 / downsample_factor
        try:
            # Try zoom first
            masked_grid = np.ma.masked_invalid(plot_grid)
            zoomed_data = zoom(masked_grid.data, zoom_factor, order=0, prefilter=False)
            zoomed_mask = zoom(masked_grid.mask, zoom_factor, order=0, prefilter=False)
            plot_grid = np.ma.masked_array(zoomed_data, mask=zoomed_mask).filled(np.nan)
            print(f"Downsampled shape: {plot_grid.shape}")
        except Exception as zoom_err:
            # Fallback to slicing
            print(f"Warning: Zoom failed ({zoom_err}). Slicing.")
            step = max(1, int(downsample_factor))
            plot_grid = grid_data[::step, ::step]
    
    # Mask invalid data for plotting
    masked_plot_data = np.ma.masked_invalid(plot_grid)
    valid_count_original = np.sum(~np.isnan(grid_data))
    valid_count_plot = np.sum(~np.isnan(plot_grid))
    
    # Handle case with no valid data to plot
    if valid_count_plot == 0:
        print("Warning: No valid data points found in the grid to plot.")
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_facecolor('lightgray')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Mean Sap Velocity Map\n(No valid data)')
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        plt.tight_layout()
        plt.savefig(output_img, dpi=150)
        plt.close()
        print(f"Empty visualization saved: {output_img}")
        return
        
    # Create visualization
    print("Creating enhanced visualization...")
    
    # Setup figure with geographic projection
    plt.figure(figsize=(14, 10), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add geographic features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    
    # Set map extent slightly larger than data for context
    padding = 5  # degrees of padding around data
    ax.set_extent([min_lon-padding, max_lon+padding, min_lat-padding, max_lat+padding], 
                  crs=ccrs.PlateCarree())
    
    # Determine robust color limits using percentiles with reduced contrast
    valid_data = plot_grid[~np.isnan(plot_grid)]
    if len(valid_data) > 0:
        vmin = np.percentile(valid_data, 10)  # Using 10th percentile
        vmax = np.percentile(valid_data, 90)  # Using 90th percentile
    else:
        vmin, vmax = None, None
    
    # Create a colormap with less contrast between values
    colors = [(0.0, 'mediumblue'),
              (0.25, 'royalblue'),
              (0.5, 'cadetblue'),
              (0.75, 'mediumseagreen'),
              (1.0, 'darkseagreen')]
    
    cm = LinearSegmentedColormap.from_list('sap_velocity_subtle', colors, N=256)
    
    # Apply power normalization to further reduce contrast
    norm = PowerNorm(gamma=contrast_reduction)
    
    # Display the raster data with reduced contrast
    img = ax.imshow(
        masked_plot_data,
        origin='upper',
        extent=[min_lon, max_lon, min_lat, max_lat],
        transform=ccrs.PlateCarree(),
        cmap=cm,
        norm=norm,
        interpolation='nearest'
    )
    
    # Add gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.MaxNLocator(6)
    gl.ylocator = mticker.MaxNLocator(6)
    
    # Add colorbar at the bottom
    cbar = plt.colorbar(img, ax=ax, orientation='horizontal', shrink=0.7, 
                        pad=0.05, location='bottom')
    cbar.set_label('Sap Velocity (kg/cm²/h-1)', fontsize=12)
    
    # Adjust colorbar ticks and labels for better readability
    cbar.ax.xaxis.set_tick_params(labelsize=10)
    cbar.ax.xaxis.label.set_size(11)
    
    # Add title with additional context
    subtitle = f"(Original: {valid_count_original:,} locations, Plotted: ~{valid_count_plot:,})"
    plt.title(f"{title}\n{subtitle}", fontsize=14)
    
    # Add timestamp/attribution
    plt.figtext(0.02, 0.02, f"Generated: {datetime.now().strftime('%Y-%m-%d')}", fontsize=8, ha='left')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Enhanced visualization saved to: {output_img}")


# Example usage
if __name__ == "__main__":
    # Example call - replace with your actual GeoTIFF path
    input_tif = "sap_velocity_map_global_midday_v1.tif"
    output_png = "sap_velocity_visualization.png"
    
    # Call with different contrast settings
    visualize_sap_velocity_tif(
        tif_path=input_tif,
        output_img=output_png,
        title="Mean Sap Velocity Map (Mid-July)",
        contrast_reduction=0.8  # Lower value = less contrast
    )