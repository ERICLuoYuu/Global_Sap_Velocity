"""
Module for spatial cross-validation of time series data using blockCV approach.
Implements systematic spatial blocking with buffer zones for proper spatial cross-validation.
Includes spatial autocorrelation analysis for data-driven block sizing.
"""
from pathlib import Path
import sys
import os

# Set environment variables for determinism BEFORE importing TensorFlow
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['PYTHONHASHSEED'] = '42'

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the randomization control module first
from src.utils.random_control import (
    set_seed, get_seed, deterministic,
)

# Import the TimeSeriesSegmenter and WindowGenerator classes
from src.hyperparameter_optimization.timeseries_processor1 import (
    TimeSeriesSegmenter, WindowGenerator
)

# Set the master seed at the very beginning
set_seed(42)

# Now import all other dependencies
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Union, Optional, Any
from collections import Counter
import matplotlib.patches as patches
from scipy import spatial, stats
import warnings


class BlockCV:
    """
    Implements spatial blocking cross-validation for ecological data.
    Based on blockCV package principles for spatial data partitioning.
    """
    
    @staticmethod
    @deterministic
    def calculate_spatial_autocorrelation(
        lat: Union[np.ndarray, List],
        lon: Union[np.ndarray, List],
        values: Union[np.ndarray, List],
        max_distance: Optional[float] = None,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Calculate spatial autocorrelation of values using variogram.
        
        Parameters
        ----------
        lat : array-like
            Latitude values
        lon : array-like
            Longitude values
        values : array-like
            Values to calculate autocorrelation for
        max_distance : float, optional
            Maximum distance to consider, in degrees
        n_bins : int, default=10
            Number of distance bins for variogram
            
        Returns
        -------
        dict
            Dictionary with autocorrelation analysis results
        """
        # Convert inputs to numpy arrays
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        values = np.asarray(values)
        
        # Filter out NaN values
        valid_mask = ~np.isnan(values)
        lat = lat[valid_mask]
        lon = lon[valid_mask]
        values = values[valid_mask]
        
        # Check if we have enough data
        if len(values) < 5:
            warnings.warn("Too few valid data points for spatial autocorrelation analysis")
            return {
                'practical_range': 1.0,  # Default value
                'moran_i': 0,
                'max_distance': 1.0
            }
        
        # Combine coordinates into points
        points = np.column_stack((lat, lon))
        
        # Calculate pairwise distances
        distances = spatial.distance.pdist(points)
        distances_square = spatial.distance.squareform(distances)
        
        # Set max_distance if not provided
        if max_distance is None:
            max_distance = np.percentile(distances, 90)  # Use 90th percentile
        
        # Create distance bins
        bins = np.linspace(0, max_distance, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Initialize arrays for variogram
        variogram = np.zeros(n_bins)
        counts = np.zeros(n_bins)
        
        # Calculate variogram
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                dist = distances_square[i, j]
                if dist <= max_distance:
                    # Find bin index
                    bin_idx = np.digitize(dist, bins) - 1
                    if bin_idx < n_bins:  # Ensure within range
                        # Calculate squared difference of values
                        diff_squared = (values[i] - values[j]) ** 2
                        variogram[bin_idx] += diff_squared
                        counts[bin_idx] += 1
        
        # Calculate mean variogram values, avoiding division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            variogram = np.where(counts > 0, variogram / (2 * counts), np.nan)
        
        # Calculate global variance for normalization
        global_variance = np.var(values)
        normalized_variogram = variogram / global_variance if global_variance > 0 else variogram
        
        # Estimate practical range (where autocorrelation reaches 95% of the sill)
        # Find the first bin where the variogram exceeds 95% of the plateau
        plateau = np.nanmean(normalized_variogram[int(0.8 * n_bins):]) if n_bins > 5 else np.nanmax(normalized_variogram)
        range_threshold = 0.95 * plateau
        
        range_bin = np.where(normalized_variogram >= range_threshold)[0]
        practical_range = bin_centers[range_bin[0]] if len(range_bin) > 0 else max_distance
        
        # Calculate Moran's I (global spatial autocorrelation)
        # Create spatial weights matrix (inverse distance)
        weights = 1.0 / (distances_square + np.eye(len(values)))  # Add eye to avoid division by zero
        np.fill_diagonal(weights, 0)  # Set diagonal to zero
        
        # Standardize values
        z = (values - np.mean(values)) / np.std(values) if np.std(values) > 0 else values - np.mean(values)
        
        # Calculate Moran's I
        zw = np.dot(weights, z)
        moran_i = np.sum(z * zw) / np.sum(z ** 2)
        
        # Return results
        return {
            'bin_centers': bin_centers,
            'variogram': variogram,
            'normalized_variogram': normalized_variogram,
            'counts': counts,
            'practical_range': practical_range,
            'moran_i': moran_i,
            'global_variance': global_variance,
            'max_distance': max_distance
        }
    
    @staticmethod
    @deterministic
    def select_block_size_from_autocorrelation(
        autocorrelation_results: Dict[str, Any],
        method: str = 'practical_range',
        multiplier: float = 1.5
    ) -> float:
        """
        Select appropriate block size based on spatial autocorrelation analysis.
        
        Parameters
        ----------
        autocorrelation_results : dict
            Results from calculate_spatial_autocorrelation function
        method : str, default='practical_range'
            Method to determine block size ('practical_range' or 'first_minimum')
        multiplier : float, default=1.5
            Multiplier for the selected block size
            
        Returns
        -------
        float
            Recommended block size in same units as input coordinates
        """
        if method == 'practical_range':
            # Use practical range from variogram
            block_size = autocorrelation_results['practical_range']
        elif method == 'first_minimum':
            # Find first local minimum in variogram
            variogram = autocorrelation_results['normalized_variogram']
            bin_centers = autocorrelation_results['bin_centers']
            
            # Smooth variogram to reduce noise
            window_size = min(3, len(variogram))
            if window_size < 3:
                # Not enough points for smoothing
                block_size = autocorrelation_results['practical_range']
            else:
                smoothed = np.convolve(variogram, np.ones(window_size)/window_size, mode='valid')
                
                # Find local minima
                minima_idx = []
                for i in range(1, len(smoothed) - 1):
                    if smoothed[i-1] > smoothed[i] and smoothed[i] < smoothed[i+1]:
                        minima_idx.append(i)
                
                if minima_idx:
                    # Use first minimum after initial increase
                    first_min_idx = minima_idx[0]
                    block_size = bin_centers[first_min_idx + window_size//2]
                else:
                    # Fallback to practical range
                    block_size = autocorrelation_results['practical_range']
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'practical_range' or 'first_minimum'.")
        
        # Apply multiplier and ensure positive value
        block_size = max(0.1, block_size * multiplier)
        
        print(f"Selected block size ({method}): {block_size:.4f} with multiplier {multiplier}")
        return block_size
    
    @staticmethod
    @deterministic
    def plot_spatial_autocorrelation(
        autocorrelation_results: Dict[str, Any],
        filename: Optional[str] = None,
        title: str = "Spatial Autocorrelation Analysis",
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot spatial autocorrelation analysis results.
        
        Parameters
        ----------
        autocorrelation_results : dict
            Results from calculate_spatial_autocorrelation function
        filename : str, optional
            If provided, save figure to this file
        title : str, default="Spatial Autocorrelation Analysis"
            Figure title
        figsize : tuple of int, default=(10, 6)
            Figure size
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with spatial autocorrelation plots
        """
        # Extract data
        bin_centers = autocorrelation_results['bin_centers']
        variogram = autocorrelation_results['normalized_variogram']
        counts = autocorrelation_results['counts']
        practical_range = autocorrelation_results['practical_range']
        moran_i = autocorrelation_results['moran_i']
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot variogram
        ax1.plot(bin_centers, variogram, 'o-', color='blue')
        ax1.axvline(x=practical_range, color='red', linestyle='--', 
                   label=f'Practical Range: {practical_range:.4f}')
        ax1.axhline(y=1.0, color='green', linestyle=':', alpha=0.7,
                   label='Global Variance')
        ax1.set_xlabel('Distance')
        ax1.set_ylabel('Normalized Variogram')
        ax1.set_title(f'Variogram (Moran\'s I: {moran_i:.4f})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot bin counts
        ax2.bar(bin_centers, counts, width=bin_centers[1]-bin_centers[0] if len(bin_centers) > 1 else 0.1, alpha=0.6, color='blue')
        ax2.set_xlabel('Distance')
        ax2.set_ylabel('Pair Counts')
        ax2.set_title('Number of Pairs per Distance Bin')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.suptitle(title, y=1.05)
        
        # Save figure if filename provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    @deterministic
    def create_spatial_blocks(
        lat: Union[np.ndarray, List],
        lon: Union[np.ndarray, List],
        data_counts: Union[np.ndarray, List] = None,
        block_size: Union[float, Tuple[float, float]] = 1.0,
        buffer_size: float = 0.0,
        random_state: int = 42
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create systematic spatial blocks with optional buffer zones.
        
        Parameters
        ----------
        lat : array-like
            Latitude values
        lon : array-like
            Longitude values
        data_counts : array-like, optional
            Count of data points for each site (for statistics)
        block_size : float or tuple of float, default=1.0
            Size of spatial blocks in degrees (single value for square, tuple for rectangle)
        buffer_size : float, default=0.0
            Size of buffer zone between blocks in degrees
        random_state : int, default=42
            Random state for reproducibility
            
        Returns
        -------
        groups : np.ndarray
            Array of group labels (-1 for buffer zones)
        block_info : dict
            Dictionary with block grid information
        """
        # Convert inputs to numpy arrays
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        
        # Use equal data counts if not provided
        if data_counts is None:
            data_counts = np.ones_like(lat)
        else:
            data_counts = np.asarray(data_counts)
        
        # Set random seed for reproducibility
        np.random.seed(random_state)
        
        # Handle block_size parameter
        if isinstance(block_size, (int, float)):
            lat_block_size = lon_block_size = float(block_size)
        else:
            lat_block_size, lon_block_size = block_size
        
        # Calculate the range of lat and lon
        lat_min, lat_max = np.min(lat), np.max(lat)
        lon_min, lon_max = np.min(lon), np.max(lon)
        
        # Add a small buffer to ensure all points are included
        lat_padding = 0.01 * (lat_max - lat_min)
        lon_padding = 0.01 * (lon_max - lon_min)
        
        # Calculate grid dimensions
        # Position blocks to cover the full range
        lat_range = lat_max - lat_min + 2 * lat_padding
        lon_range = lon_max - lon_min + 2 * lon_padding
        
        n_lat_blocks = max(1, int(np.ceil(lat_range / lat_block_size)))
        n_lon_blocks = max(1, int(np.ceil(lon_range / lon_block_size)))
        
        # Adjust block sizes to cover the full range
        lat_block_size = lat_range / n_lat_blocks
        lon_block_size = lon_range / n_lon_blocks
        
        # Calculate starting points
        lat_start = lat_min - lat_padding
        lon_start = lon_min - lon_padding
        
        # Create grid of lat and lon block edges
        lat_edges = [lat_start + i * lat_block_size for i in range(n_lat_blocks + 1)]
        lon_edges = [lon_start + i * lon_block_size for i in range(n_lon_blocks + 1)]
        
        # Initialize block assignments and create blocks info
        block_assignments = np.full(len(lat), -1)  # Default to -1 (buffer zone)
        
        # Store block info for visualization and reference
        block_info = {
            'lat_edges': lat_edges,
            'lon_edges': lon_edges,
            'lat_block_size': lat_block_size,
            'lon_block_size': lon_block_size,
            'buffer_size': buffer_size,
            'n_lat_blocks': n_lat_blocks,
            'n_lon_blocks': n_lon_blocks,
            'total_blocks': n_lat_blocks * n_lon_blocks,
            'blocks': []
        }
        
        # Assign points to blocks
        block_id = 0
        for i in range(n_lat_blocks):
            for j in range(n_lon_blocks):
                # Define block boundaries
                lat_min_block = lat_edges[i]
                lat_max_block = lat_edges[i + 1]
                lon_min_block = lon_edges[j]
                lon_max_block = lon_edges[j + 1]
                
                # Define buffer zone boundaries if buffer_size > 0
                if buffer_size > 0:
                    lat_min_core = lat_min_block + buffer_size
                    lat_max_core = lat_max_block - buffer_size
                    lon_min_core = lon_min_block + buffer_size
                    lon_max_core = lon_max_block - buffer_size
                    
                    # Skip if block is too small for buffer
                    if lat_min_core >= lat_max_core or lon_min_core >= lon_max_core:
                        continue
                    
                    # Assign points to core block (not in buffer zone)
                    in_block = ((lat >= lat_min_core) & (lat <= lat_max_core) & 
                                (lon >= lon_min_core) & (lon <= lon_max_core))
                else:
                    # No buffer, assign all points in block
                    in_block = ((lat >= lat_min_block) & (lat <= lat_max_block) & 
                                (lon >= lon_min_block) & (lon <= lon_max_block))
                
                # Store block information
                block_info['blocks'].append({
                    'id': block_id,
                    'lat_min': lat_min_block,
                    'lat_max': lat_max_block,
                    'lon_min': lon_min_block,
                    'lon_max': lon_max_block,
                    'buffer_size': buffer_size,
                    'n_points': np.sum(in_block)
                })
                
                # Skip empty blocks
                if np.sum(in_block) == 0:
                    block_id += 1
                    continue
                
                # Assign block ID to points in this block
                block_assignments[in_block] = block_id
                
                # Calculate statistics for this block
                block_stats = {
                    'block': block_id,
                    'size': np.sum(in_block),
                    'total_data_points': np.sum(data_counts[in_block]),
                    'mean_lat': np.mean(lat[in_block]),
                    'mean_lon': np.mean(lon[in_block]),
                    'min_lat': lat_min_block,
                    'max_lat': lat_max_block,
                    'min_lon': lon_min_block,
                    'max_lon': lon_max_block,
                    'buffer_size': buffer_size
                }
                
                # Add stats to block info
                block_info['blocks'][-1].update(block_stats)
                
                block_id += 1
        
        # Print summary of block distribution
        point_count_by_block = Counter(block_assignments)
        valid_blocks = [b for b in point_count_by_block.keys() if b >= 0]
        
        print("Spatial Block Statistics:")
        print(f"Total blocks: {len(valid_blocks)}")
        print(f"Points in buffer zones: {point_count_by_block.get(-1, 0)}")
        
        # Calculate data point statistics by block
        if valid_blocks:
            block_data_points = []
            for block in valid_blocks:
                mask = block_assignments == block
                block_data_points.append(np.sum(data_counts[mask]))
                
            print(f"Min data points in a block: {min(block_data_points)}")
            print(f"Max data points in a block: {max(block_data_points)}")
            print(f"Mean data points per block: {np.mean(block_data_points):.1f}")
        
        return block_assignments, block_info
    
    @staticmethod
    @deterministic
    def plot_spatial_blocks(
        lat: Union[np.ndarray, List],
        lon: Union[np.ndarray, List],
        block_assignments: np.ndarray,
        block_info: Dict[str, Any],
        filename: Optional[str] = None,
        title: str = "Spatial Blocks for Cross-Validation",
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot spatial blocks with buffer zones.
        
        Parameters
        ----------
        lat : array-like
            Latitude values
        lon : array-like
            Longitude values
        block_assignments : np.ndarray
            Block assignments for each point
        block_info : dict
            Dictionary with block grid information
        filename : str, optional
            If provided, save figure to this file
        title : str, default="Spatial Blocks for Cross-Validation"
            Figure title
        figsize : tuple of int, default=(10, 8)
            Figure size
            
        Returns
        -------
        matplotlib.figure.Figure
            Figure with spatial blocks
        """
        # Convert inputs to numpy arrays
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique blocks (excluding buffer zones)
        unique_blocks = np.unique(block_assignments)
        unique_blocks = unique_blocks[unique_blocks >= 0]
        
        # Create colormap for blocks
        cmap = plt.cm.get_cmap('tab10', len(unique_blocks))
        
        # Plot points by block assignment
        for i, block_id in enumerate(unique_blocks):
            # Get points in this block
            mask = block_assignments == block_id
            ax.scatter(
                lon[mask], lat[mask], 
                c=[cmap(i)], 
                s=50, 
                alpha=0.7, 
                label=f"Block {block_id}"
            )
        
        # Plot buffer zone points
        buffer_mask = block_assignments == -1
        if np.any(buffer_mask):
            ax.scatter(
                lon[buffer_mask], lat[buffer_mask], 
                c='gray', 
                s=30, 
                alpha=0.4, 
                marker='x', 
                label="Buffer Zone"
            )
        
        # Draw block boundaries
        for block in block_info['blocks']:
            rect = patches.Rectangle(
                (block['lon_min'], block['lat_min']),
                block['lon_max'] - block['lon_min'],
                block['lat_max'] - block['lat_min'],
                linewidth=1,
                edgecolor='black',
                facecolor='none',
                alpha=0.7
            )
            ax.add_patch(rect)
            
            # Draw buffer zone if applicable
            if block['buffer_size'] > 0:
                buffer_rect = patches.Rectangle(
                    (block['lon_min'] + block['buffer_size'], block['lat_min'] + block['buffer_size']),
                    (block['lon_max'] - block['lon_min']) - 2 * block['buffer_size'],
                    (block['lat_max'] - block['lat_min']) - 2 * block['buffer_size'],
                    linewidth=1,
                    edgecolor='red',
                    facecolor='none',
                    linestyle='--',
                    alpha=0.7
                )
                ax.add_patch(buffer_rect)
        
        # Add legend (but limit items if many blocks)
        if len(unique_blocks) > 10:
            # Show only first 5 and last 5 blocks
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 10:
                handles = handles[:5] + handles[-6:]
                labels = labels[:5] + ['...'] + labels[-5:]
            ax.legend(handles, labels, loc='best')
        else:
            ax.legend(loc='best')
        
        # Set labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure if filename provided
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    @deterministic
    def create_group_labels_for_windows(site_to_block: Dict[str, int], site_windows_count: Dict[str, int]) -> List[int]:
        """
        Create group labels for all windows based on site block assignments.
        
        Parameters:
        -----------
        site_to_block : dict
            Dictionary mapping site IDs to block numbers
        site_windows_count : dict
            Dictionary mapping site IDs to number of windows
            
        Returns:
        --------
        group_labels : list
            List containing group labels for each window
        """
        group_labels = []
        
        for site_id, window_count in site_windows_count.items():
            block = site_to_block.get(site_id, -1)  # Default to -1 (buffer) if not found
            # Assign the same block number to all windows from this site
            group_labels.extend([block] * window_count)
        
        return group_labels