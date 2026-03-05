from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy import stats
import concurrent.futures
from tqdm import tqdm
import warnings
import ee
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Suppress pandas future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Constants ---
MODIS_START = '2002-07-04'
MODIS_END = '2025-09-30'
OVERLAP_START = '2002-07-04'
OVERLAP_END = '2018-12-31'
AVHRR_START = '1981-06-24'
AVHRR_END = '2018-12-31'

# SR bins as defined in the paper
SR_BINS = [
    (0, 1.22),
    (1.22, 1.5),
    (1.5, 1.86),
    (1.86, 2.33),
    (2.33, 3.0),
    (3.0, 4.0),
    (4.0, 5.67),
    (5.67, 9.0),
    (9.0, 19.0),
    (19.0, float('inf'))
]

# --- Earth Engine Initialization ---
def initialize_ee(project_id):
    """Initializes the Earth Engine API."""
    try:
        ee.Authenticate()
        ee.Initialize(project=project_id)
        print("Google Earth Engine initialized successfully!")
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        raise

# --- Quality Control Functions ---
def mask_quality_modis(image):
    """Creates a mask to keep only the best quality pixels from a MODIS LAI image."""
    qc = image.select('FparLai_QC')
    quality_mask = qc.bitwiseAnd(1).eq(0)
    return image.updateMask(quality_mask)

def mask_quality_avhrr(image):
    """Creates a mask for AVHRR/GLASS LAI images."""
    # GLASS products typically have different QC schemes
    # Adjust based on the specific AVHRR product you're using
    lai = image.select('Lai')
    # Basic valid range check (0-100 for scaled LAI)
    mask = lai.gte(0).And(lai.lte(100))
    return image.updateMask(mask)
# --- Data Extraction Functions ---
def extract_modis_data(point, start_date, end_date):
    """Extract MEAN MODIS LAI data for a region around a point."""
    # Define a region, for example, a 5km box around the point
    region = point.buffer(2500).bounds() # 2.5km buffer = 5km box

    modis_collection = ee.ImageCollection('MODIS/061/MCD15A3H')
    filtered = modis_collection.filterDate(start_date, end_date).map(mask_quality_modis).select('Lai')

    def reduce_and_timestamp(image):
        # Calculate the mean over the region
        mean_lai = image.select('Lai').reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=500,
            maxPixels=1e9
        ).get('Lai')
        # Return the mean value with the image timestamp
        return ee.Feature(None, {'Lai': mean_lai, 'time': image.get('system:time_start')})

    # Map the reduction over the image collection
    reduced_data = filtered.map(reduce_and_timestamp).getInfo()

    # Process the results into a DataFrame
    if not reduced_data['features']:
        
        return pd.DataFrame()
    
    data_list = [f['properties'] for f in reduced_data['features']]
    df = pd.DataFrame(data_list)
    if df.dropna().empty:
        print(f"Warning: No valid data found for MODIS LAI extraction. site: {point.getInfo()}")
        return pd.DataFrame()
    df = df[df['Lai'].notna()] # Filter out nulls from empty reductions
    
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')
    df['Lai'] = pd.to_numeric(df['Lai'], errors='coerce') * 0.1
    df = df[['datetime', 'Lai']].dropna()
    
    return df
def extract_avhrr_sr_data(point, start_date, end_date):
    """
    Extract AVHRR Simple Ratio (SR) data for a specific point and date range.
    Following the paper's approach using GIMMS NDVI or computing SR from reflectances.
    """
    try:
        # Option 1: If using GIMMS NDVI (as in the paper)
        # Convert NDVI to SR using: SR = (1 + NDVI) / (1 - NDVI)
        # This would require access to GIMMS NDVI data
        
        # Option 2: Calculate SR from AVHRR reflectances
        # Using NOAA CDR AVHRR Surface Reflectance if available
        avhrr_collection = ee.ImageCollection('NOAA/CDR/AVHRR/SR/V5')
        
        filtered = avhrr_collection.filterDate(start_date, end_date)
        
        # Calculate Simple Ratio (NIR/Red)
        def calculate_sr(image):
            nir = image.select('SREFL_CH2')  # NIR band
            red = image.select('SREFL_CH1')  # Red band
            sr = nir.divide(red).rename('SR')
            return image.addBands(sr).select('SR')
        
        sr_collection = filtered.map(calculate_sr)
        raw_data = sr_collection.getRegion(point, scale=5000).getInfo()
        
        if len(raw_data) <= 1:
            print("Warning: No valid data found for AVHRR SR extraction.")
            return pd.DataFrame()
        
        df = pd.DataFrame(raw_data[1:], columns=raw_data[0])
        df['datetime'] = pd.to_datetime(df['time'], unit='ms')
        df['SR'] = pd.to_numeric(df['SR'], errors='coerce')
        df = df[['datetime', 'SR']].dropna()
        
        # Filter out invalid SR values
        df = df[(df['SR'] > 0) & (df['SR'] < 50)]  # Reasonable SR range
        
        return df
        
    except Exception as e:
        print(f"Error extracting AVHRR SR data: {e}")
        # Fallback: Convert from NDVI if SR calculation fails
        return extract_avhrr_from_ndvi(point, start_date, end_date)

def extract_avhrr_from_ndvi(point, start_date, end_date):
    """
    Alternative: Extract SR from AVHRR NDVI using the conversion formula from the paper.
    SR = (1 + NDVI) / (1 - NDVI)
    """
    try:
        # This would use GIMMS NDVI or similar NDVI product
        # Placeholder for GIMMS NDVI extraction
        avhrr_collection = ee.ImageCollection('NOAA/CDR/AVHRR/NDVI/V5')
        
        filtered = avhrr_collection.filterDate(start_date, end_date).select('NDVI')
        raw_data = filtered.getRegion(point, scale=5000).getInfo()
        
        if len(raw_data) <= 1:
            return pd.DataFrame()
        
        df = pd.DataFrame(raw_data[1:], columns=raw_data[0])
        df['datetime'] = pd.to_datetime(df['time'], unit='ms')
        df['NDVI'] = pd.to_numeric(df['NDVI'], errors='coerce') * 0.0001  # Scale factor
        
        # Convert NDVI to SR
        df['SR'] = (1 + df['NDVI']) / (1 - df['NDVI'])
        df = df[['datetime', 'SR']].dropna()
        
        # Filter out invalid SR values
        df = df[(df['SR'] > 0) & (df['SR'] < 50)]
        
        return df
        
    except Exception as e:
        print(f"Error converting NDVI to SR: {e}")
        return pd.DataFrame()

# --- Pixel-Based SR-LAI Relationship Building ---
class PixelSRLAIRelationship:
    """
    Implements the pixel-based SR-LAI relationship approach from the paper.
    """
    
    def __init__(self, sr_bins=SR_BINS):
        self.sr_bins = sr_bins
        self.bin_lai_values = {}
        self.linear_model = None
        self.validation_metrics = {}  # Store R², RMSE, etc.
        
    def build_relationship(self, modis_lai_df, avhrr_sr_df, site_info=None):
        """
        Build pixel-specific SR-LAI relationship using binned approach.
        """
        # Merge datasets on nearest dates
        modis_lai_df = modis_lai_df.set_index('datetime').sort_index()
        avhrr_sr_df = avhrr_sr_df.set_index('datetime').sort_index()
        
        # Resample to common frequency (4-day as in paper)
        modis_resampled = modis_lai_df.resample('4D').mean()
        avhrr_resampled = avhrr_sr_df.resample('4D').mean()
        
        # Join the datasets
        merged = pd.merge_asof(
            left=modis_resampled.reset_index(),
            right=avhrr_resampled.reset_index(),
            on='datetime',
            direction='nearest',
            suffixes=('_modis', '_avhrr'),
            tolerance=pd.Timedelta('2 days')
        )
        
        merged = merged.dropna()
        
        if len(merged) < 5:
            print(f"Warning: Insufficient overlap data ({len(merged)} points)")
            self._build_simple_linear_model(merged)
            return
        
        # Bin the SR values and calculate mean LAI for each bin
        for i, (sr_min, sr_max) in enumerate(self.sr_bins):
            bin_data = merged[(merged['SR'] >= sr_min) & (merged['SR'] < sr_max)]
            if len(bin_data) > 0:
                # Calculate mean SR and LAI for this bin
                mean_sr = (sr_min + min(sr_max, 19)) / 2  # Use bin center
                mean_lai = bin_data['Lai'].mean()
                self.bin_lai_values[i] = {
                    'sr_center': mean_sr,
                    'lai': mean_lai,
                    'n_samples': len(bin_data)
                }
        
        # For bins without data, interpolate or use linear model
        self._fill_missing_bins(merged)
        
        # Validate the relationship on the overlap data
        self.validate_relationship(merged)
        
    def validate_relationship(self, merged_data):
        """
        Validate the SR-LAI relationship by comparing predicted vs actual LAI.
        Calculate R², RMSE, and other metrics for the overlap period.
        """
        if len(merged_data) < 2:
            print("Insufficient data for validation")
            return
        
        # Predict LAI using the established relationship
        merged_data['LAI_predicted'] = merged_data['SR'].apply(self.predict_lai)
        
        # Calculate validation metrics
        actual = merged_data['Lai'].values
        predicted = merged_data['LAI_predicted'].values
        
        # Remove any NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) < 2:
            print("Insufficient valid data points for validation")
            return
        
        # Calculate R²
        r2 = r2_score(actual, predicted)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        # Calculate bias
        bias = np.mean(predicted - actual)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(actual, predicted)[0, 1]
        
        # Linear regression for slope and intercept
        slope, intercept, r_value, p_value, std_err = stats.linregress(actual, predicted)
        
        self.validation_metrics = {
            'r2': r2,
            'rmse': rmse,
            'bias': bias,
            'correlation': correlation,
            'slope': slope,
            'intercept': intercept,
            'n_samples': len(actual),
            'p_value': p_value,
            'actual_mean': np.mean(actual),
            'predicted_mean': np.mean(predicted),
            'actual_std': np.std(actual),
            'predicted_std': np.std(predicted)
        }
        
        print(f"\n=== Validation Results for Overlap Period ===")
        print(f"R²: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"Bias: {bias:.4f}")
        print(f"Correlation: {correlation:.4f}")
        print(f"Regression slope: {slope:.3f} (ideal = 1.0)")
        print(f"Regression intercept: {intercept:.3f} (ideal = 0.0)")
        print(f"Number of samples: {len(actual)}")
        print(f"P-value: {p_value:.6f}")
        print("=" * 45)
        
        # Store the validation data for plotting
        self.validation_data = merged_data.copy()
    
    def plot_validation(self, site_info=None, save_path=None):
        """
        Plot validation results comparing predicted vs actual LAI.
        """
        if not hasattr(self, 'validation_data') or self.validation_data.empty:
            print("No validation data available to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Scatter plot of predicted vs actual
        ax1 = axes[0, 0]
        actual = self.validation_data['Lai'].values
        predicted = self.validation_data['LAI_predicted'].values
        
        ax1.scatter(actual, predicted, alpha=0.6, s=50, color='blue', edgecolors='black', linewidth=0.5)
        
        # Add 1:1 line
        max_val = max(np.nanmax(actual), np.nanmax(predicted))
        min_val = min(np.nanmin(actual), np.nanmin(predicted))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')
        
        # Add regression line
        z = np.polyfit(actual[~np.isnan(actual) & ~np.isnan(predicted)], 
                      predicted[~np.isnan(actual) & ~np.isnan(predicted)], 1)
        p = np.poly1d(z)
        x_line = np.linspace(min_val, max_val, 100)
        ax1.plot(x_line, p(x_line), 'g-', linewidth=2, 
                label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
        
        ax1.set_xlabel('Actual MODIS LAI', fontsize=12)
        ax1.set_ylabel('Predicted LAI from AVHRR SR', fontsize=12)
        ax1.set_title(f'Validation: R²={self.validation_metrics["r2"]:.3f}, '
                     f'RMSE={self.validation_metrics["rmse"]:.3f}', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Time series comparison
        ax2 = axes[0, 1]
        dates = pd.to_datetime(self.validation_data['datetime'])
        ax2.plot(dates, actual, 'b-', label='Actual MODIS LAI', linewidth=2, alpha=0.7)
        ax2.plot(dates, predicted, 'r--', label='Predicted LAI', linewidth=2, alpha=0.7)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('LAI', fontsize=12)
        ax2.set_title('Time Series Comparison', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Residuals plot
        ax3 = axes[1, 0]
        residuals = predicted - actual
        ax3.scatter(actual, residuals, alpha=0.6, s=30)
        ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax3.set_xlabel('Actual MODIS LAI', fontsize=12)
        ax3.set_ylabel('Residuals (Predicted - Actual)', fontsize=12)
        ax3.set_title(f'Residuals Plot: Bias={self.validation_metrics["bias"]:.3f}', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        # Add confidence intervals
        std_residuals = np.std(residuals[~np.isnan(residuals)])
        ax3.axhline(y=2*std_residuals, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='±2σ')
        ax3.axhline(y=-2*std_residuals, color='orange', linestyle=':', linewidth=1, alpha=0.5)
        ax3.legend(fontsize=10)
        
        # 4. Histogram of residuals
        ax4 = axes[1, 1]
        ax4.hist(residuals[~np.isnan(residuals)], bins=20, edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Residuals', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Distribution of Residuals', fontsize=14)
        ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add normal distribution overlay
        mu, std = np.nanmean(residuals), np.nanstd(residuals)
        x = np.linspace(np.nanmin(residuals), np.nanmax(residuals), 100)
        ax4_twin = ax4.twinx()
        ax4_twin.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, label='Normal dist.')
        ax4_twin.set_ylabel('Probability Density', fontsize=12, color='r')
        ax4_twin.tick_params(axis='y', labelcolor='r')
        
        # Add site info if provided
        if site_info:
            fig.suptitle(f'Harmonization Validation - Lat: {site_info["lat"]:.2f}, Lon: {site_info["lon"]:.2f}', 
                        fontsize=16, fontweight='bold')
        else:
            fig.suptitle('Harmonization Validation Results', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Validation plot saved to {save_path}")
        
        plt.show()
    
    def _build_simple_linear_model(self, merged_data):
        """
        Build a simple linear regression model as fallback.
        """
        if len(merged_data) < 2:
            # Use default relationship
            self.linear_model = {'slope': 1.5, 'intercept': 0.5}
        else:
            slope, intercept, _, _, _ = stats.linregress(
                merged_data['SR'], merged_data['Lai']
            )
            self.linear_model = {'slope': slope, 'intercept': intercept}
            
            # Still validate even with linear model
            self.validate_relationship(merged_data)
    
    def _fill_missing_bins(self, merged_data):
        """
        Fill missing bins using linear interpolation or regression.
        """
        if len(merged_data) > 0 and not self.linear_model:
            self._build_simple_linear_model(merged_data)
        
        # Fill missing bins
        for i, (sr_min, sr_max) in enumerate(self.sr_bins):
            if i not in self.bin_lai_values:
                sr_center = (sr_min + min(sr_max, 19)) / 2
                
                # Try interpolation from neighboring bins
                interpolated = self._interpolate_from_neighbors(i, sr_center)
                
                if interpolated is not None:
                    self.bin_lai_values[i] = {
                        'sr_center': sr_center,
                        'lai': interpolated,
                        'n_samples': 0  # Indicates interpolated value
                    }
                elif self.linear_model:
                    # Use linear model as fallback
                    lai = self.linear_model['slope'] * sr_center + self.linear_model['intercept']
                    self.bin_lai_values[i] = {
                        'sr_center': sr_center,
                        'lai': max(0, lai),  # Ensure non-negative
                        'n_samples': 0
                    }
    
    def _interpolate_from_neighbors(self, bin_index, sr_center):
        """
        Interpolate LAI value from neighboring bins.
        """
        # Find nearest bins with data
        lower_bin = None
        upper_bin = None
        
        for i in range(bin_index - 1, -1, -1):
            if i in self.bin_lai_values and self.bin_lai_values[i]['n_samples'] > 0:
                lower_bin = i
                break
        
        for i in range(bin_index + 1, len(self.sr_bins)):
            if i in self.bin_lai_values and self.bin_lai_values[i]['n_samples'] > 0:
                upper_bin = i
                break
        
        if lower_bin is not None and upper_bin is not None:
            # Linear interpolation
            x1 = self.bin_lai_values[lower_bin]['sr_center']
            y1 = self.bin_lai_values[lower_bin]['lai']
            x2 = self.bin_lai_values[upper_bin]['sr_center']
            y2 = self.bin_lai_values[upper_bin]['lai']
            
            if x2 != x1:
                interpolated = y1 + (y2 - y1) * (sr_center - x1) / (x2 - x1)
                return max(0, interpolated)
        
        return None
    
    def predict_lai(self, sr_value):
        """
        Predict LAI for a given SR value using the established relationship.
        """
        if sr_value < self.sr_bins[0][0]:
            return 0  # Below minimum SR threshold
        
        # Find appropriate bin
        for i, (sr_min, sr_max) in enumerate(self.sr_bins):
            if sr_min <= sr_value < sr_max:
                if i in self.bin_lai_values:
                    if i == len(self.sr_bins) - 1:  # Last bin
                        return self.bin_lai_values[i]['lai']
                    
                    # Linear interpolation within bin
                    if i + 1 in self.bin_lai_values:
                        x1 = self.bin_lai_values[i]['sr_center']
                        y1 = self.bin_lai_values[i]['lai']
                        x2 = self.bin_lai_values[i + 1]['sr_center']
                        y2 = self.bin_lai_values[i + 1]['lai']
                        
                        if x2 != x1:
                            lai = y1 + (y2 - y1) * (sr_value - x1) / (x2 - x1)
                            return max(0, lai)
                    
                    return self.bin_lai_values[i]['lai']
        
        # SR exceeds maximum bin
        if len(self.bin_lai_values) > 0:
            last_bin = max(self.bin_lai_values.keys())
            return self.bin_lai_values[last_bin]['lai']
        
        return 0

def harmonize_avhrr_with_pixel_method(avhrr_sr_df, pixel_relationship):
    """
    Apply pixel-based SR-LAI relationship to AVHRR SR data.
    """
    df = avhrr_sr_df.copy()
    df['Lai'] = df['SR'].apply(pixel_relationship.predict_lai)
    df['Lai'] = df['Lai'].clip(lower=0)  # Ensure non-negative values
    return df[['datetime', 'Lai']]

# --- Enhanced Processing Function ---
def process_site_enhanced(site_info, method='high_freq', validate=True, plot_validation=True):
    """
    Process a site using the pixel-based SR-LAI relationship method from the paper.
    
    Parameters:
    -----------
    site_info : tuple
        Site information (index, site data)
    method : str
        Processing method ('high_freq' or 'low_freq')
    validate : bool
        Whether to perform validation on overlap period
    plot_validation : bool
        Whether to plot validation results
    """
    idx, site = site_info
    
    try:
        point = ee.Geometry.Point([site['lon'], site['lat']])
        site_start = site['start_date']
        site_end = site['end_date']
        
        # Extract MODIS LAI for the entire available period
        modis_lai_full = extract_modis_data(point, MODIS_START, MODIS_END)
        
        # Handle timezone consistency
        if hasattr(site_start, 'tzinfo') and site_start.tzinfo is not None:
            modis_start_dt = pd.to_datetime(MODIS_START, utc=True)
            overlap_start_dt = pd.to_datetime(OVERLAP_START, utc=True)
            overlap_end_dt = pd.to_datetime(OVERLAP_END, utc=True)
        else:
            modis_start_dt = pd.to_datetime(MODIS_START)
            overlap_start_dt = pd.to_datetime(OVERLAP_START)
            overlap_end_dt = pd.to_datetime(OVERLAP_END)
        
        use_avhrr = site_start < modis_start_dt
        use_modis = site_end >= modis_start_dt
        
        combined_df = pd.DataFrame()
        pixel_relationship = None
        
        if use_avhrr:
            print(f"\nSite {idx+1} ({site.get('site_name', 'Unknown')}): Building pixel-based SR-LAI relationship")
            print(f"Location: Lat {site['lat']:.3f}, Lon {site['lon']:.3f}")
            
            # Extract overlap period data
            overlap_modis = extract_modis_data(point, OVERLAP_START, OVERLAP_END)
            overlap_avhrr_sr = extract_avhrr_sr_data(point, OVERLAP_START, OVERLAP_END)
            
            if not overlap_modis.empty and not overlap_avhrr_sr.empty:
                # Build pixel-specific SR-LAI relationship
                pixel_relationship = PixelSRLAIRelationship()
                pixel_relationship.build_relationship(overlap_modis, overlap_avhrr_sr, site)
                
                # Validate on additional overlap data if requested
                if validate and hasattr(pixel_relationship, 'validation_metrics'):
                    print(f"\nValidation metrics saved for Site {idx+1}")
                    
                    # Plot validation if requested
                    if plot_validation:
                        save_path = f"validation_site_{idx+1}_{site.get('site_name', 'unknown')}.png"
                        pixel_relationship.plot_validation(site_info=site, save_path=save_path)
                
                # Extract historical AVHRR SR data
                avhrr_end = min(site_end, modis_start_dt)
                historical_avhrr_sr = extract_avhrr_sr_data(
                    point,
                    site_start.strftime('%Y-%m-%d'),
                    avhrr_end.strftime('%Y-%m-%d')
                )
                
                if not historical_avhrr_sr.empty:
                    # Apply pixel-based harmonization
                    harmonized_avhrr = harmonize_avhrr_with_pixel_method(
                        historical_avhrr_sr, pixel_relationship
                    )
                    combined_df = pd.concat([combined_df, harmonized_avhrr])
                    
                    # Visualize the SR-LAI relationship
                    # visualize_sr_lai_relationship(pixel_relationship, site, idx)
            
            # Add MODIS data for the modern period
            if use_modis:
                modis_start = max(site_start, modis_start_dt)
                modis_data = extract_modis_data(
                    point,
                    modis_start.strftime('%Y-%m-%d'),
                    site_end.strftime('%Y-%m-%d')
                )
                if not modis_data.empty:
                    combined_df = pd.concat([combined_df, modis_data])
        
        elif use_modis and not use_avhrr:
            print(f"Site {idx+1}: Using MODIS only")
            combined_df = extract_modis_data(
                point,
                max(site_start, modis_start_dt).strftime('%Y-%m-%d'),
                site_end.strftime('%Y-%m-%d')
            )
        
        if combined_df.empty:
            print(f"No data retrieved for site {idx+1}")
            return None
        
        # Sort and remove duplicates
        combined_df = combined_df.sort_values('datetime').drop_duplicates(subset=['datetime'])
        
        # Apply temporal processing
        if method == 'high_freq':
            result_df = apply_high_freq_processing(combined_df, modis_lai_full)
        else:
            result_df = apply_low_freq_processing(combined_df)
        
        if result_df is not None:
            result_df['site_id'] = idx + 1
            # Add validation metrics if available
            if pixel_relationship and hasattr(pixel_relationship, 'validation_metrics'):
                result_df['r2'] = pixel_relationship.validation_metrics['r2']
                result_df['rmse'] = pixel_relationship.validation_metrics['rmse']
        
        return result_df
        
    except Exception as e:
        print(f"Error processing site {idx+1}: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_sr_lai_relationship(pixel_relationship, site, idx):
    """
    Visualize the pixel-based SR-LAI relationship.
    """
    if not pixel_relationship.bin_lai_values:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bin centers and LAI values
    sr_centers = []
    lai_values = []
    sample_sizes = []
    
    for bin_data in pixel_relationship.bin_lai_values.values():
        sr_centers.append(bin_data['sr_center'])
        lai_values.append(bin_data['lai'])
        sample_sizes.append(bin_data['n_samples'])
    
    # Sort by SR
    sorted_indices = np.argsort(sr_centers)
    sr_centers = np.array(sr_centers)[sorted_indices]
    lai_values = np.array(lai_values)[sorted_indices]
    sample_sizes = np.array(sample_sizes)[sorted_indices]
    
    # Plot with different markers for interpolated vs actual data
    actual_mask = sample_sizes > 0
    if any(actual_mask):
        ax.scatter(sr_centers[actual_mask], lai_values[actual_mask], 
                  s=100, c='blue', label='Data-derived', alpha=0.7)
    if any(~actual_mask):
        ax.scatter(sr_centers[~actual_mask], lai_values[~actual_mask], 
                  s=50, c='red', marker='^', label='Interpolated', alpha=0.7)
    
    # Plot connecting line
    ax.plot(sr_centers, lai_values, 'g--', alpha=0.5, label='SR-LAI relationship')
    
    ax.set_xlabel('Simple Ratio (SR)')
    ax.set_ylabel('LAI')
    ax.set_title(f'Pixel-based SR-LAI Relationship - Site {idx+1}\n'
                f'Lat: {site["lat"]:.2f}, Lon: {site["lon"]:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def apply_high_freq_processing(df, climatology):
    """Apply high-frequency processing to the combined dataset."""
    df['doy_period'] = (pd.to_datetime(df['datetime']).dt.dayofyear // 4) * 4
    climatology['doy_period'] = (pd.to_datetime(climatology['datetime']).dt.dayofyear // 4) * 4
    climatology = climatology.groupby('doy_period')['Lai'].mean()
    
    # Resample to 4-day intervals
    df = df.set_index('datetime')[['Lai']].resample('4D').mean().reset_index()
    
    # Fill gaps with climatology
    df['doy_period'] = (df['datetime'].dt.dayofyear // 4) * 4
    fill_values = df['doy_period'].map(climatology)
    df['Lai_filled'] = df['Lai'].fillna(fill_values)
    
    if df['Lai_filled'].isnull().all():
        return None
    
    # Apply smoothing
    series_to_filter = df['Lai_filled'].interpolate(method='linear', limit_direction='both')
    window_length = min(31, len(df) - 1 if len(df) % 2 == 0 else len(df))
    
    if window_length > 3 and not series_to_filter.isnull().any():
        df['LAI'] = savgol_filter(series_to_filter, window_length=window_length, polyorder=2)
    else:
        df['LAI'] = series_to_filter
    
    return df[['datetime', 'LAI']]

def apply_low_freq_processing(df):
    """Apply low-frequency (monthly) processing to the combined dataset."""
    monthly_df = df.set_index('datetime').resample('MS')['Lai'].mean().to_frame()
    
    if monthly_df['Lai'].isnull().all():
        return None
    
    monthly_df['Lai_interp'] = monthly_df['Lai'].interpolate(method='linear', limit_direction='both')
    window_length = min(7, len(monthly_df) - 1 if len(monthly_df) % 2 == 0 else len(monthly_df))
    
    if window_length > 3:
        monthly_df['LAI'] = savgol_filter(monthly_df['Lai_interp'], 
                                         window_length=window_length, polyorder=2)
    else:
        monthly_df['LAI'] = monthly_df['Lai_interp']
    
    return monthly_df[['LAI']].reset_index()

# --- Main Function ---
def extract_lai_enhanced(sites_df, max_workers=8, force_timezone='naive', validate=True, plot_validation=True):
    """
    Main function using enhanced pixel-based SR-LAI harmonization.
    
    Parameters:
    -----------
    sites_df : DataFrame
        Sites information with columns: lat, lon, start_date, end_date
    max_workers : int
        Number of parallel workers
    force_timezone : str
        Timezone handling ('naive' or 'UTC')
    validate : bool
        Whether to perform validation on overlap period
    plot_validation : bool
        Whether to plot validation results for each site
    
    Returns:
    --------
    hourly_high_freq : DataFrame
        High frequency LAI results
    hourly_low_freq : DataFrame
        Low frequency LAI results
    validation_summary : DataFrame
        Summary of validation metrics for all sites
    """
    initialize_ee('era5download-447713')  # Replace with your project ID
    
    # Standardize timezone handling
    from datetime import timezone
    sites_df = sites_df.copy()
    for col in ['start_date', 'end_date']:
        if sites_df[col].dtype == 'object':
            sites_df[col] = pd.to_datetime(sites_df[col])
        if force_timezone == 'naive' and sites_df[col].dt.tz is not None:
            sites_df[col] = sites_df[col].dt.tz_localize(None)
    
    high_freq_results = []
    low_freq_results = []
    validation_metrics_list = []
    site_data_list = list(sites_df.iterrows())
    
    # Process with parallel execution
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # High-frequency processing
        futures_hf = {
            executor.submit(process_site_enhanced, site_data, 'high_freq', validate, plot_validation): site_data 
            for site_data in site_data_list
        }
        
        for future in tqdm(concurrent.futures.as_completed(futures_hf), 
                          total=len(site_data_list), 
                          desc="Processing High-Freq LAI (Enhanced)"):
            res = future.result()
            if res is not None:
                high_freq_results.append(res)
                # Extract validation metrics if available
                if 'r2' in res.columns and not res['r2'].isna().all():
                    site_data = futures_hf[future]
                    idx, site = site_data
                    validation_metrics_list.append({
                        'site_id': idx + 1,
                        'site_name': site.get('site_name', f'Site_{idx+1}'),
                        'lat': site['lat'],
                        'lon': site['lon'],
                        'r2': res['r2'].iloc[0] if len(res) > 0 else np.nan,
                        'rmse': res['rmse'].iloc[0] if len(res) > 0 else np.nan
                    })
        
        # Low-frequency processing
        futures_lf = {
            executor.submit(process_site_enhanced, site_data, 'low_freq', False, False): site_data[0] 
            for site_data in site_data_list
        }
        
        for future in tqdm(concurrent.futures.as_completed(futures_lf), 
                          total=len(site_data_list), 
                          desc="Processing Low-Freq LAI (Enhanced)"):
            res = future.result()
            if res is not None:
                low_freq_results.append(res)
    
    # Process results
    if high_freq_results:
        high_freq_df = pd.concat(high_freq_results, ignore_index=True)
        # Remove validation columns from the main results
        result_cols = [col for col in high_freq_df.columns if col not in ['r2', 'rmse']]
        high_freq_df = high_freq_df[result_cols]
        high_freq_df.set_index('datetime', inplace=True)
        hourly_high_freq = high_freq_df.groupby('site_id').apply(
            lambda x: x.resample('H').ffill()
        ).reset_index(level=0, drop=True).reset_index()
    else:
        hourly_high_freq = pd.DataFrame()
    
    if low_freq_results:
        low_freq_df = pd.concat(low_freq_results, ignore_index=True)
        # Remove validation columns if present
        result_cols = [col for col in low_freq_df.columns if col not in ['r2', 'rmse']]
        low_freq_df = low_freq_df[result_cols]
        low_freq_df.set_index('datetime', inplace=True)
        hourly_low_freq = low_freq_df.groupby('site_id').apply(
            lambda x: x.resample('H').ffill()
        ).reset_index(level=0, drop=True).reset_index()
    else:
        hourly_low_freq = pd.DataFrame()
    
    # Create validation summary
    validation_summary = pd.DataFrame(validation_metrics_list)
    
    if not validation_summary.empty:
        print("\n" + "="*60)
        print("VALIDATION SUMMARY - All Sites")
        print("="*60)
        print(f"\nNumber of sites with validation: {len(validation_summary)}")
        print(f"Mean R²: {validation_summary['r2'].mean():.4f}")
        print(f"Median R²: {validation_summary['r2'].median():.4f}")
        print(f"Min R²: {validation_summary['r2'].min():.4f}")
        print(f"Max R²: {validation_summary['r2'].max():.4f}")
        print(f"Mean RMSE: {validation_summary['rmse'].mean():.4f}")
        print(f"Median RMSE: {validation_summary['rmse'].median():.4f}")
        
        print("\nPer-site validation metrics:")
        print(validation_summary[['site_name', 'lat', 'lon', 'r2', 'rmse']].to_string(index=False))
        print("="*60)
        
        # Save validation summary
        validation_summary.to_csv(Path(__file__).parent / 'validation_summary.csv', index=False)
        print("\nValidation summary saved to 'validation_summary.csv'")
        
        # Create overall validation plot
        if plot_validation and len(validation_summary) > 1:
            plot_overall_validation(validation_summary)
    
    return hourly_high_freq, hourly_low_freq, validation_summary

def plot_overall_validation(validation_summary):
    """
    Create an overall validation summary plot for all sites.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. R² distribution
    ax1 = axes[0, 0]
    ax1.hist(validation_summary['r2'].dropna(), bins=15, edgecolor='black', alpha=0.7, color='blue')
    ax1.axvline(validation_summary['r2'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {validation_summary["r2"].mean():.3f}')
    ax1.axvline(validation_summary['r2'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {validation_summary["r2"].median():.3f}')
    ax1.set_xlabel('R²', fontsize=12)
    ax1.set_ylabel('Number of Sites', fontsize=12)
    ax1.set_title('Distribution of R² Values', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. RMSE distribution
    ax2 = axes[0, 1]
    ax2.hist(validation_summary['rmse'].dropna(), bins=15, edgecolor='black', alpha=0.7, color='green')
    ax2.axvline(validation_summary['rmse'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {validation_summary["rmse"].mean():.3f}')
    ax2.axvline(validation_summary['rmse'].median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {validation_summary["rmse"].median():.3f}')
    ax2.set_xlabel('RMSE', fontsize=12)
    ax2.set_ylabel('Number of Sites', fontsize=12)
    ax2.set_title('Distribution of RMSE Values', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Spatial distribution of R²
    ax3 = axes[1, 0]
    scatter = ax3.scatter(validation_summary['lon'], validation_summary['lat'], 
                         c=validation_summary['r2'], s=100, cmap='RdYlGn', 
                         edgecolor='black', linewidth=0.5, vmin=0, vmax=1)
    ax3.set_xlabel('Longitude', fontsize=12)
    ax3.set_ylabel('Latitude', fontsize=12)
    ax3.set_title('Spatial Distribution of R² Values', fontsize=14)
    plt.colorbar(scatter, ax=ax3, label='R²')
    ax3.grid(True, alpha=0.3)
    
    # 4. R² vs RMSE
    ax4 = axes[1, 1]
    ax4.scatter(validation_summary['r2'], validation_summary['rmse'], 
                s=60, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('R²', fontsize=12)
    ax4.set_ylabel('RMSE', fontsize=12)
    ax4.set_title('R² vs RMSE Relationship', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    # Add site labels for outliers
    for idx, row in validation_summary.iterrows():
        if row['r2'] < 0.7 or row['rmse'] > 1.0:
            ax4.annotate(row['site_name'], (row['r2'], row['rmse']), 
                        fontsize=8, alpha=0.7)
    
    fig.suptitle('Overall Harmonization Validation Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig('validation_summary_all_sites.png', dpi=300, bbox_inches='tight')
    print("Overall validation plot saved to 'validation_summary_all_sites.png'")
    plt.show()

# --- Usage Example ---
if __name__ == "__main__":
    # Create example dataset
    try:
        # Read CSV - let pandas parse dates first
        sites_df = pd.read_csv(
            'data/raw/0.1.5/0.1.5/csv/plant/site_info.csv',
            parse_dates=['start_date', 'end_date']
        )
        
        print(f"Loaded {len(sites_df)} sites from CSV")
        
        # Check timezone status
        if sites_df['start_date'].dt.tz is not None:
            print(f"Dates are timezone-aware: {sites_df['start_date'].iloc[0].tzinfo}")
        else:
            print("Dates are timezone-naive")
        
    except FileNotFoundError:
        print("Error: site_info.csv not found. Creating a dummy dataframe for demonstration.")
        sites_df = pd.DataFrame({
            'site_name': ['US-UMB', 'IT-Cpz', 'BR-Sa3'],
            'lat': [45.4916, 41.6559, -3.0180],
            'lon': [-84.7138, 12.1258, -54.9714],
            'start_date': ['1998-01-01', '2000-01-01', '2003-01-01'],  # Note dates before MODIS
            'end_date': ['2018-12-31', '2014-12-31', '2010-12-31']
        })
        # For the example, use timezone-naive dates
        sites_df['start_date'] = pd.to_datetime(sites_df['start_date'])
        sites_df['end_date'] = pd.to_datetime(sites_df['end_date'])
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        raise
    
    # Extract enhanced LAI data with validation
    lai_high_freq_hourly, lai_low_freq_hourly, validation_summary = extract_lai_enhanced(
        sites_df, 
        max_workers=10,
        force_timezone='naive',
        validate=True,        # Enable validation
        plot_validation=False  # Create validation plots
    )
    
    # Save results
    if not lai_high_freq_hourly.empty:
        print(f"\nHigh-frequency processing complete. Shape: {lai_high_freq_hourly.shape}")
        print(f"Date range: {lai_high_freq_hourly['datetime'].min()} to {lai_high_freq_hourly['datetime'].max()}")
        lai_high_freq_hourly.to_csv(Path(__file__).parent / 'enhanced_lai_high_freq_hourly.csv', index=False)
    
    if not lai_low_freq_hourly.empty:
        print(f"\nLow-frequency processing complete. Shape: {lai_low_freq_hourly.shape}")
        print(f"Date range: {lai_low_freq_hourly['datetime'].min()} to {lai_low_freq_hourly['datetime'].max()}")
        lai_low_freq_hourly.to_csv(Path(__file__).parent / 'enhanced_lai_low_freq_hourly.csv', index=False)
    
    # Display validation summary
    if not validation_summary.empty:
        print("\n" + "="*60)
        print("FINAL VALIDATION RESULTS")
        print("="*60)
        print(f"Sites with R² > 0.8: {(validation_summary['r2'] > 0.8).sum()}/{len(validation_summary)}")
        print(f"Sites with R² > 0.9: {(validation_summary['r2'] > 0.9).sum()}/{len(validation_summary)}")
        print(f"Sites with RMSE < 0.5: {(validation_summary['rmse'] < 0.5).sum()}/{len(validation_summary)}")
        print("="*60)