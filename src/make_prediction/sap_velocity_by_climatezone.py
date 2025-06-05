import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rasterio.warp import reproject, Resampling, calculate_default_transform
import os
from scipy import stats

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

def analyze_sap_velocity_by_climate(sap_velocity_path, koppen_path, target_resolution=0.1):
    """
    Analyze sap velocity distributions across major climate zones using the 
    resampled Beck et al. (2018) Köppen-Geiger classification.
    
    Parameters:
    -----------
    sap_velocity_path : str
        Path to the sap velocity raster (.tif)
    koppen_path : str
        Path to the original high-resolution Köppen-Geiger raster
    target_resolution : float
        Target resolution in degrees for resampling (default: 0.1° ≈ 10km at equator)
    """
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Print information about the Köppen-Geiger file
    print("Inspecting Köppen-Geiger file...")
    with rasterio.open(koppen_path) as src:
        print(f"  Data type: {src.dtypes[0]}")
        print(f"  No data value: {src.nodata}")
        print(f"  Resolution: {src.res}")
        print(f"  Dimensions: {src.width} x {src.height}")
        unique_values = np.unique(src.read(1))[:]  # Look at unique values
        print(f"  Sample unique values: {unique_values}")
    
    # Resample the Köppen-Geiger classification first
    resampled_koppen_path = os.path.join(output_dir, "koppen_resampled.tif")
    resample_koppen_geiger(koppen_path, resampled_koppen_path, target_resolution)
    
    # Load the sap velocity raster
    print("Loading sap velocity raster...")
    with rasterio.open(sap_velocity_path) as src:
        sap_velocity = src.read(1)
        sap_meta = src.meta
        sap_transform = src.transform
        sap_crs = src.crs
        sap_nodata = src.nodata
        
        print(f"  Data type: {src.dtypes[0]}")
        print(f"  No data value: {sap_nodata}")
        print(f"  Resolution: {src.res}")
        print(f"  Dimensions: {src.width} x {src.height}")
        print(f"  Sap velocity value range: {np.nanmin(sap_velocity)} to {np.nanmax(sap_velocity)}")
    
    # Load the resampled Köppen-Geiger climate classification raster
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
    
    # Check if we need to match the resampled Köppen data to the sap velocity grid
    if sap_crs != koppen_crs or sap_velocity.shape != koppen.shape:
        print("Aligning resampled climate zones to match sap velocity raster...")
        
        # Create destination array
        dst_koppen = np.zeros_like(sap_velocity, dtype=koppen.dtype)
        
        # Perform reprojection to match sap velocity grid
        reproject(
            source=koppen,
            destination=dst_koppen,
            src_transform=koppen_transform,
            src_crs=koppen_crs,
            dst_transform=sap_transform,
            dst_crs=sap_crs,
            resampling=Resampling.mode  # Use mode for categorical data
        )
        
        koppen = dst_koppen
        print("Alignment complete.")
    
    # Extract major climate zones using the improved method
    print("Extracting major climate zones from the Beck et al. dataset...")
    koppen_major, major_zones = extract_major_climate_zones(koppen, koppen_nodata)
    
    # Display the unique major zones found in the data
    unique_major = np.unique(koppen_major)
    print(f"Major climate zones found: {[major_zones.get(zone, f'Unknown ({zone})') for zone in unique_major if zone != koppen_nodata and not np.isnan(zone)]}")
    
    # Extract sap velocity values for each climate zone
    data = []
    
    print("Extracting sap velocity values for each climate zone...")
    
    # Create masks for valid data
    if sap_nodata is not None:
        sap_valid_mask = (sap_velocity != sap_nodata)
    else:
        sap_valid_mask = ~np.isnan(sap_velocity)
        
    if koppen_nodata is not None:
        koppen_valid_mask = (koppen_major != koppen_nodata)
    else:
        koppen_valid_mask = ~np.isnan(koppen_major)
        
    valid_mask = sap_valid_mask & koppen_valid_mask
    
    # For each major zone, extract and sample the sap velocity values
    for zone_id, zone_name in major_zones.items():
        # Create a mask for the current climate zone
        zone_mask = (koppen_major == zone_id) & valid_mask
        
        # Extract sap velocity values for this zone
        zone_sap_values = sap_velocity[zone_mask]
        
        if len(zone_sap_values) == 0:
            print(f"No data points found for {zone_name}")
            continue
        
        # Sample the data if it's very large (to avoid memory issues)
        if len(zone_sap_values) > 500000000:
            print(f"Sampling {zone_name} (original size: {len(zone_sap_values)})")
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(len(zone_sap_values), 500000, replace=False)
            zone_sap_values = zone_sap_values[indices]
        
        print(f"{zone_name}: {len(zone_sap_values)} data points")
        
        # Add to dataframe
        zone_df = pd.DataFrame({
            'Climate Zone': zone_name,
            'Sap Velocity': zone_sap_values
        })
        data.append(zone_df)
    
    # Check if we have data
    if not data:
        print("No valid data points found. Please check the input files and masks.")
        return
    
    # Combine all data
    df = pd.concat(data, ignore_index=True)
    
    # Calculate summary statistics
    summary = df.groupby('Climate Zone')['Sap Velocity'].describe()
    print("\nSummary Statistics:")
    print(summary)
    
    # Save statistics to CSV
    summary.to_csv(f"{output_dir}/sap_velocity_climate_zone_stats.csv")
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Box plot with slimmer boxes and median lines
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid", palette="viridis", font_scale=1.2)
    # sns.set_style("ticks")
    sns.despine(offset=10, trim=True);
    # Remove the frame


    palette = sns.light_palette("seagreen", n_colors=len(df['Climate Zone'].unique()))[2::1]
    # Create box plot with slimmer boxes
    ax = sns.boxplot(
        x='Climate Zone', 
        y='Sap Velocity', 
        data=df,
        palette=palette,
        fill=False,
        linewidth=1.5,
        width=0.4,  # Make boxes slimmer (0.8 is default)
        showfliers=False  # Hide outliers for cleaner visualization
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Set opacity for boxes
    for patch in ax.artists:
        patch.set_alpha(0.2)  # Set box opacity to 20%

    # Add swarm plot with small sample for visualization
    sample_df = df.groupby('Climate Zone').apply(
        lambda x: x.sample(min(300, len(x)), random_state=42)
    ).reset_index(drop=True)

    sns.stripplot(
        x='Climate Zone', 
        y='Sap Velocity', 
        data=sample_df,
        size=5, 
        palette=palette, 
        alpha=0.1,
        jitter=True
    )

    # Calculate means for each climate zone
    means = df.groupby('Climate Zone')['Sap Velocity'].mean()

    # Add median dash lines and labels
    for i, zone in enumerate(sorted(df['Climate Zone'].unique())):
        color = palette[i % len(palette)]  # Cycle through colors
        median_val = means[zone]
        # Add dash line for mean
        plt.hlines(
            y=median_val,
            xmin=i - 0.2,  # Adjust based on box width
            xmax=i + 0.2,
            colors= color,
            linestyles='dashed',
            linewidth=1.5
        )
  
        

    # Add titles and labels
    plt.title('Sap Velocity Distribution by Major Climate Zone', fontsize=16)
    plt.xlabel('Köppen-Geiger Climate Zone', fontsize=14)
    plt.ylabel('Sap Velocity kg / (cm² • h)', fontsize=14)
    plt.xticks()
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{output_dir}/sap_velocity_by_climate_zone.png", dpi=300)
    
    # Also create a violin plot for a different visualization
    plt.figure(figsize=(12, 8))
    sns.violinplot(
        x='Climate Zone',
        y='Sap Velocity',
        data=df,
        inner='quartile',
        palette='viridis'
    )
    plt.title('Sap Velocity Distribution by Major Climate Zone', fontsize=16)
    plt.xlabel('Köppen-Geiger Climate Zone', fontsize=14)
    plt.ylabel('Sap Velocity kg / (cm² • h)', fontsize=14)
    plt.xticks()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sap_velocity_violin_plot.png", dpi=300)
    
    # Perform ANOVA to test for significant differences
    climate_zones = df['Climate Zone'].unique()
    if len(climate_zones) > 1:  # Need at least 2 groups for ANOVA
        try:
            samples = [df.loc[df['Climate Zone'] == zone, 'Sap Velocity'].values for zone in climate_zones]
            f_stat, p_value = stats.f_oneway(*samples)
            
            print(f"\nANOVA results: F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")
            if p_value < 0.05:
                print("There are significant differences in sap velocity between climate zones.")
                
                # Perform post-hoc Tukey HSD test to identify which specific groups differ
                try:
                    from statsmodels.stats.multicomp import pairwise_tukeyhsd
                    
                    # Prepare data for Tukey test
                    posthoc = pairwise_tukeyhsd(
                        df['Sap Velocity'],
                        df['Climate Zone'],
                        alpha=0.05
                    )
                    
                    print("\nTukey HSD Post-hoc Test Results:")
                    print(posthoc)
                    
                    # Save results to text file
                    with open(f"{output_dir}/tukey_hsd_results.txt", 'w') as f:
                        f.write(str(posthoc))
                except:
                    print("Could not perform Tukey HSD test. You may need to install statsmodels.")
            else:
                print("No significant differences in sap velocity between climate zones were detected.")
        except Exception as e:
            print(f"Error performing ANOVA: {e}")
    else:
        print("Need at least two climate zones with data to perform statistical comparison.")
    
    print(f"\nResults saved to {output_dir} directory")
    return df

if __name__ == "__main__":
    # File paths - MODIFY THESE to match your file locations
    sap_velocity_path = "outputs/maps/sap_velocity_map_global_midday_v1.tif"
    koppen_path = "data/raw/grided/Beck_KG_V1/Beck_KG_V1_present_0p083.tif"
    
    # Target resolution for resampling (in degrees)
    # 0.1° ≈ 10km at the equator
    # 0.5° ≈ 50km at the equator
    # Choose based on your computational resources and needs
    target_resolution = 0.1  # Modify this as needed
    
    # Run the analysis
    analyze_sap_velocity_by_climate(
        sap_velocity_path=sap_velocity_path,
        koppen_path=koppen_path,
        target_resolution=target_resolution
    )