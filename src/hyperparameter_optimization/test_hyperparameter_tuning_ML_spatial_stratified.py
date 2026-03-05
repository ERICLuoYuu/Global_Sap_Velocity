"""
Machine Learning model with a spatial cross-validation approach for site-based prediction.
Implements group-based spatial cross-validation with proper time windowing.
FIXED VERSION: Corrected SHAP analysis with hemisphere separation and proper index tracking.
"""
from pathlib import Path
import sys
import os
import argparse
import logging
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import joblib

# Set environment variables for determinism BEFORE importing TensorFlow/other libraries
os.environ['PYTHONHASHSEED'] = '42'

# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the randomization control module first
from src.utils.random_control import (
    set_seed, get_seed, deterministic
)

# Import the time series windowing modules
from src.hyperparameter_optimization.timeseries_processor1 import (
    TimeSeriesSegmenter, WindowGenerator
)
from src.hyperparameter_optimization.plot_fold_distribution import FoldDistributionAnalyzer

# Set the master seed at the very beginning
set_seed(42)

# Now import all other dependencies
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Union, List, Tuple, Dict, Optional
import json
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from path_config import PathConfig, get_default_paths

# Import StratifiedGroupKFold and GroupKFold for spatial stratified cross-validation
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
import tensorflow as tf

# Apply additional determinism settings
tf.random.set_seed(42)
np.random.seed(42)

# Import the hyperparameter optimizer
from src.hyperparameter_optimization.hyper_tuner import MLOptimizer
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.colors import Normalize
from matplotlib import cm


# =============================================================================
# TARGET TRANSFORMER CLASS
# =============================================================================

class TargetTransformer:
    """
    A class to handle various target variable transformations for regression.
    
    Supported methods:
    - 'log1p': log(1 + x) transformation, inverse: exp(x) - 1
    - 'sqrt': square root transformation, inverse: x^2
    - 'box-cox': Box-Cox transformation (requires positive values), inverse uses fitted lambda
    - 'yeo-johnson': Yeo-Johnson transformation (works with negative values), uses sklearn
    - 'none': no transformation
    
    Usage:
        transformer = TargetTransformer(method='log1p')
        y_transformed = transformer.fit_transform(y_train)
        y_test_transformed = transformer.transform(y_test)
        y_pred_original = transformer.inverse_transform(y_pred_transformed)
    """
    
    VALID_METHODS = ['log1p', 'sqrt', 'box-cox', 'yeo-johnson', 'none']
    
    def __init__(self, method: str = 'log1p'):
        """
        Initialize the transformer.
        
        Args:
            method: Transformation method. One of 'log1p', 'sqrt', 'box-cox', 'yeo-johnson', 'none'
        """
        if method not in self.VALID_METHODS:
            raise ValueError(f"Invalid method '{method}'. Must be one of {self.VALID_METHODS}")
        
        self.method = method
        self._fitted = False
        self._lambda = None  # For Box-Cox
        self._sklearn_transformer = None  # For Yeo-Johnson
        self._shift = 0  # Shift for handling zeros/negatives
        
    def fit(self, y: np.ndarray) -> 'TargetTransformer':
        """
        Fit the transformer to the data.
        
        Args:
            y: Target values to fit on
            
        Returns:
            self
        """
        y = np.asarray(y).flatten()
        
        if self.method == 'box-cox':
            # Box-Cox requires strictly positive values
            if np.any(y <= 0):
                self._shift = np.abs(y.min()) + 1e-6
                logging.info(f"Box-Cox: Shifting data by {self._shift:.4f} to ensure positive values")
            else:
                self._shift = 0
            
            y_shifted = y + self._shift
            _, self._lambda = boxcox(y_shifted)
            logging.info(f"Box-Cox: Fitted lambda = {self._lambda:.4f}")
            
        elif self.method == 'yeo-johnson':
            self._sklearn_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
            self._sklearn_transformer.fit(y.reshape(-1, 1))
            logging.info(f"Yeo-Johnson: Fitted lambda = {self._sklearn_transformer.lambdas_[0]:.4f}")
            
        elif self.method == 'sqrt':
            # Check for negative values
            if np.any(y < 0):
                self._shift = np.abs(y.min()) + 1e-6
                logging.info(f"Sqrt: Shifting data by {self._shift:.4f} to ensure non-negative values")
            else:
                self._shift = 0
        
        self._fitted = True
        return self
    
    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Transform the target values.
        
        Args:
            y: Target values to transform
            
        Returns:
            Transformed values
        """
        y = np.asarray(y).flatten()
        
        if self.method == 'none':
            return y
            
        if self.method == 'log1p':
            # Handle negative values by using signed log
            return np.log1p(np.maximum(y, 0))
            
        elif self.method == 'sqrt':
            y_shifted = y + self._shift
            return np.sqrt(np.maximum(y_shifted, 0))
            
        elif self.method == 'box-cox':
            if not self._fitted:
                raise RuntimeError("Transformer must be fitted before transform for Box-Cox")
            y_shifted = y + self._shift
            # Use the fitted lambda
            return boxcox(np.maximum(y_shifted, 1e-10), lmbda=self._lambda)
            
        elif self.method == 'yeo-johnson':
            if not self._fitted or self._sklearn_transformer is None:
                raise RuntimeError("Transformer must be fitted before transform for Yeo-Johnson")
            return self._sklearn_transformer.transform(y.reshape(-1, 1)).flatten()
            
        return y
    
    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            y: Target values to fit and transform
            
        Returns:
            Transformed values
        """
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        """
        Inverse transform the values back to original scale.
        
        Args:
            y_transformed: Transformed values
            
        Returns:
            Values in original scale
        """
        y_transformed = np.asarray(y_transformed).flatten()
        
        if self.method == 'none':
            return y_transformed
            
        elif self.method == 'log1p':
            return np.expm1(y_transformed)
            
        elif self.method == 'sqrt':
            y_squared = np.square(y_transformed)
            return y_squared - self._shift
            
        elif self.method == 'box-cox':
            if self._lambda is None:
                raise RuntimeError("Transformer must be fitted before inverse_transform for Box-Cox")
            y_original = inv_boxcox(y_transformed, self._lambda)
            return y_original - self._shift
            
        elif self.method == 'yeo-johnson':
            if self._sklearn_transformer is None:
                raise RuntimeError("Transformer must be fitted before inverse_transform for Yeo-Johnson")
            return self._sklearn_transformer.inverse_transform(y_transformed.reshape(-1, 1)).flatten()
            
        return y_transformed
    
    def get_params(self) -> Dict:
        """Get transformer parameters for saving."""
        return {
            'method': self.method,
            'fitted': self._fitted,
            'lambda': self._lambda,
            'shift': self._shift,
        }
    
    @classmethod
    def from_params(cls, params: Dict) -> 'TargetTransformer':
        """Reconstruct transformer from saved parameters."""
        transformer = cls(method=params['method'])
        transformer._fitted = params['fitted']
        transformer._lambda = params['lambda']
        transformer._shift = params['shift']
        return transformer
    
    def __repr__(self):
        return f"TargetTransformer(method='{self.method}', fitted={self._fitted})"

# =============================================================================
# FEATURE UNITS DICTIONARY
# =============================================================================
# =============================================================================
# PFT GROUPING AND ANALYSIS FUNCTIONS
# =============================================================================

# Define PFT categories
PFT_COLUMNS = ['MF', 'DNF', 'ENF', 'EBF', 'WSA', 'WET', 'DBF', 'SAV']

# PFT full names for better labels
PFT_FULL_NAMES = {
    'MF': 'Mixed Forest',
    'DNF': 'Deciduous Needleleaf Forest',
    'ENF': 'Evergreen Needleleaf Forest',
    'EBF': 'Evergreen Broadleaf Forest',
    'WSA': 'Woody Savanna',
    'WET': 'Wetland',
    'DBF': 'Deciduous Broadleaf Forest',
    'SAV': 'Savanna'
}

# PFT colors for consistent visualization
PFT_COLORS = {
    'MF': '#1f77b4',   # blue
    'DNF': '#ff7f0e',  # orange
    'ENF': '#2ca02c',  # green
    'EBF': '#d62728',  # red
    'WSA': '#9467bd',  # purple
    'WET': '#8c564b',  # brown
    'DBF': '#e377c2',  # pink
    'SAV': '#7f7f7f'   # gray
}

def group_pft_for_summary_plots(shap_values, X_df, feature_names, pft_cols):
    """
    Combines one-hot encoded PFT columns into a single 'PFT' feature for SHAP plotting.
    
    Returns:
        new_shap_values (np.array): SHAP matrix with PFT columns summed
        new_X_df (pd.DataFrame): Feature matrix with PFT columns converted to categorical codes
        new_feature_names (list): Updated list of feature names
    """
    # 1. Identify indices
    pft_indices = [i for i, f in enumerate(feature_names) if f in pft_cols]
    
    if not pft_indices:
        print("No PFT columns found to group.")
        return shap_values, X_df, feature_names

    print(f"Grouping {len(pft_indices)} PFT features into one 'PFT' feature...")

    # 2. Aggregate SHAP values (Sum across PFT columns)
    # shape: (n_samples, 1)
    pft_shap_sum = shap_values[:, pft_indices].sum(axis=1).reshape(-1, 1)
    
    # 3. Create Feature Value (Integer encoding of the active PFT)
    # Extract just the PFT columns from X_df
    X_pft = X_df[pft_cols]
    # Find which column is '1' (or max) -> returns series of strings ('ENF', 'MF', etc.)
    pft_labels = X_pft.idxmax(axis=1)
    # Convert to integer codes (0, 1, 2...) so SHAP plots can color-code them
    pft_codes = pft_labels.astype('category').cat.codes.values.reshape(-1, 1)
    
    # 4. Remove old columns and append new one
    # Create mask for non-PFT columns
    keep_mask = np.ones(shap_values.shape[1], dtype=bool)
    keep_mask[pft_indices] = False
    
    # Filter SHAP
    shap_non_pft = shap_values[:, keep_mask]
    new_shap_values = np.hstack([shap_non_pft, pft_shap_sum])
    
    # Filter X
    # Get non-PFT feature names
    non_pft_names = [f for i, f in enumerate(feature_names) if i not in pft_indices]
    X_non_pft = X_df[non_pft_names].values
    new_X_values = np.hstack([X_non_pft, pft_codes])
    
    # Update Feature Names
    new_feature_names = non_pft_names + ['PFT']
    
    # Create new DataFrame for X
    new_X_df = pd.DataFrame(new_X_values, columns=new_feature_names)
    
    return new_shap_values, new_X_df, new_feature_names
def aggregate_pft_shap_values(
    shap_values: np.ndarray,
    feature_names: List[str],
    pft_columns: List[str] = None,
    aggregation: str = 'sum'
) -> Tuple[np.ndarray, List[str]]:
    """
    Aggregate SHAP values for PFT one-hot encoded columns into a single 'PFT' feature.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array of shape (n_samples, n_features)
    feature_names : list
        List of feature names matching shap_values columns
    pft_columns : list, optional
        List of PFT column names to aggregate. Default uses PFT_COLUMNS.
    aggregation : str
        How to aggregate: 'sum', 'mean', or 'abs_sum'
        
    Returns:
    --------
    Tuple[np.ndarray, List[str]]
        (aggregated_shap_values, aggregated_feature_names)
    """
    if pft_columns is None:
        pft_columns = PFT_COLUMNS
    
    # Find which PFT columns exist in the feature names
    pft_indices = []
    existing_pft_cols = []
    for pft in pft_columns:
        if pft in feature_names:
            pft_indices.append(feature_names.index(pft))
            existing_pft_cols.append(pft)
    
    if not pft_indices:
        logging.warning("No PFT columns found in feature names. Returning original data.")
        return shap_values, feature_names
    
    logging.info(f"Aggregating {len(existing_pft_cols)} PFT columns: {existing_pft_cols}")
    
    # Get non-PFT feature indices and names
    non_pft_indices = [i for i, name in enumerate(feature_names) if name not in pft_columns]
    non_pft_names = [feature_names[i] for i in non_pft_indices]
    
    # Extract SHAP values
    shap_non_pft = shap_values[:, non_pft_indices]
    shap_pft = shap_values[:, pft_indices]
    
    # Aggregate PFT SHAP values
    if aggregation == 'sum':
        shap_pft_agg = shap_pft.sum(axis=1, keepdims=True)
    elif aggregation == 'mean':
        shap_pft_agg = shap_pft.mean(axis=1, keepdims=True)
    elif aggregation == 'abs_sum':
        shap_pft_agg = np.abs(shap_pft).sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    # Combine: non-PFT features + aggregated PFT
    shap_aggregated = np.hstack([shap_non_pft, shap_pft_agg])
    feature_names_aggregated = non_pft_names + ['PFT']
    
    logging.info(f"Aggregated SHAP shape: {shap_aggregated.shape}")
    logging.info(f"New feature count: {len(feature_names_aggregated)} (was {len(feature_names)})")
    
    return shap_aggregated, feature_names_aggregated


def get_sample_pft_labels(
    X_original: np.ndarray,
    feature_names: List[str],
    pft_columns: List[str] = None
) -> np.ndarray:
    """
    Get PFT label for each sample based on one-hot encoded columns.
    
    Parameters:
    -----------
    X_original : np.ndarray
        Original feature values
    feature_names : list
        List of feature names
    pft_columns : list, optional
        List of PFT column names
        
    Returns:
    --------
    np.ndarray
        Array of PFT labels (strings) for each sample
    """
    if pft_columns is None:
        pft_columns = PFT_COLUMNS
    
    # Find PFT column indices
    pft_indices = {}
    for pft in pft_columns:
        if pft in feature_names:
            pft_indices[pft] = feature_names.index(pft)
    
    if not pft_indices:
        logging.warning("No PFT columns found. Returning 'Unknown' for all samples.")
        return np.array(['Unknown'] * len(X_original))
    
    # For each sample, find which PFT has value 1 (or highest value)
    pft_labels = []
    for i in range(len(X_original)):
        max_val = -np.inf
        max_pft = 'Unknown'
        for pft, idx in pft_indices.items():
            if X_original[i, idx] > max_val:
                max_val = X_original[i, idx]
                max_pft = pft
        pft_labels.append(max_pft)
    
    return np.array(pft_labels)


def plot_shap_by_pft_boxplot(
    shap_values: np.ndarray,
    feature_names: List[str],
    pft_labels: np.ndarray,
    top_n: int = 10,
    output_dir: Optional[Path] = None
) -> plt.Figure:
    """
    Create boxplots showing SHAP value distributions for each feature, stratified by PFT.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array (should NOT include PFT columns, or use aggregated version)
    feature_names : list
        List of feature names
    pft_labels : np.ndarray
        PFT label for each sample
    top_n : int
        Number of top features to plot
    output_dir : Path, optional
        Directory to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    logging.info(f"Generating SHAP by PFT boxplots for top {top_n} features...")
    
    # Get top features by mean absolute SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_features = [feature_names[i] for i in top_indices]
    
    # Get unique PFTs (excluding 'Unknown' if present)
    unique_pfts = [p for p in np.unique(pft_labels) if p != 'Unknown']
    n_pfts = len(unique_pfts)
    
    if n_pfts == 0:
        logging.warning("No valid PFT labels found. Cannot create PFT-stratified plot.")
        return None
    
    logging.info(f"  Found {n_pfts} PFT types: {unique_pfts}")
    
    # Create subplots
    n_cols = 2
    n_rows = (top_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features):
        ax = axes[i]
        feat_idx = feature_names.index(feature)
        
        # Prepare data for boxplot
        data_by_pft = []
        labels = []
        colors = []
        
        for pft in unique_pfts:
            mask = pft_labels == pft
            if mask.sum() > 0:
                data_by_pft.append(shap_values[mask, feat_idx])
                labels.append(f"{pft}\n(n={mask.sum()})")
                colors.append(PFT_COLORS.get(pft, 'gray'))
        
        # Create boxplot
        bp = ax.boxplot(data_by_pft, labels=labels, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        
        # Add units to title
        feat_unit = get_feature_unit(feature)
        if feat_unit:
            title = f"{feature} ({feat_unit})"
        else:
            title = feature
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(get_shap_label(), fontsize=10)
        #ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(f"SHAP Value Distribution by Plant Functional Type\n"
                 f"(Top {top_n} Features)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_dir:
        save_path = output_dir / "shap_by_pft_boxplot.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"    Saved: {save_path}")
    
    plt.close()
    return fig


def plot_shap_by_pft_violin(
    shap_values: np.ndarray,
    feature_names: List[str],
    pft_labels: np.ndarray,
    top_n: int = 8,
    output_dir: Optional[Path] = None
) -> plt.Figure:
    """
    Create violin plots showing SHAP value distributions for each feature, stratified by PFT.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array
    feature_names : list
        List of feature names
    pft_labels : np.ndarray
        PFT label for each sample
    top_n : int
        Number of top features to plot
    output_dir : Path, optional
        Directory to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    logging.info(f"Generating SHAP by PFT violin plots for top {top_n} features...")
    
    # Get top features by mean absolute SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_features = [feature_names[i] for i in top_indices]
    
    # Get unique PFTs
    unique_pfts = [p for p in np.unique(pft_labels) if p != 'Unknown']
    n_pfts = len(unique_pfts)
    
    if n_pfts == 0:
        logging.warning("No valid PFT labels found.")
        return None
    
    # Create DataFrame for seaborn
    plot_data = []
    for i, feature in enumerate(top_features):
        feat_idx = feature_names.index(feature)
        for pft in unique_pfts:
            mask = pft_labels == pft
            for val in shap_values[mask, feat_idx]:
                plot_data.append({
                    'Feature': feature,
                    'PFT': pft,
                    'SHAP Value': val
                })
    
    df_plot = pd.DataFrame(plot_data)
    
    # Create subplots
    n_cols = 2
    n_rows = (top_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    palette = {pft: PFT_COLORS.get(pft, 'gray') for pft in unique_pfts}
    
    for i, feature in enumerate(top_features):
        ax = axes[i]
        
        feature_data = df_plot[df_plot['Feature'] == feature]
        
        sns.violinplot(
            data=feature_data,
            x='PFT',
            y='SHAP Value',
            palette=palette,
            ax=ax,
            inner='box',
            cut=0
        )
        
        ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        
        feat_unit = get_feature_unit(feature)
        if feat_unit:
            title = f"{feature} ({feat_unit})"
        else:
            title = feature
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(get_shap_label(), fontsize=10)
        ax.set_xlabel('')
         #ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(f"SHAP Value Distribution by Plant Functional Type\n"
                 f"(Violin Plots - Top {top_n} Features)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_dir:
        save_path = output_dir / "shap_by_pft_violin.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"    Saved: {save_path}")
    
    plt.close()
    return fig


def plot_feature_importance_by_pft(
    shap_values: np.ndarray,
    feature_names: List[str],
    pft_labels: np.ndarray,
    top_n: int = 10,
    output_dir: Optional[Path] = None
) -> plt.Figure:
    """
    Create a heatmap showing feature importance (mean |SHAP|) for each PFT.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array
    feature_names : list
        List of feature names
    pft_labels : np.ndarray
        PFT label for each sample
    top_n : int
        Number of top features to show
    output_dir : Path, optional
        Directory to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    logging.info(f"Generating feature importance heatmap by PFT...")
    
    # Get unique PFTs
    unique_pfts = [p for p in np.unique(pft_labels) if p != 'Unknown']
    n_pfts = len(unique_pfts)
    
    if n_pfts == 0:
        logging.warning("No valid PFT labels found.")
        return None
    
    # Calculate mean absolute SHAP for each feature and PFT
    importance_matrix = np.zeros((len(feature_names), n_pfts))
    
    for j, pft in enumerate(unique_pfts):
        mask = pft_labels == pft
        if mask.sum() > 0:
            importance_matrix[:, j] = np.abs(shap_values[mask]).mean(axis=0)
    
    # Get top features by overall importance
    overall_importance = importance_matrix.mean(axis=1)
    top_indices = np.argsort(overall_importance)[::-1][:top_n]
    
    # Subset to top features
    importance_subset = importance_matrix[top_indices]
    top_feature_names = [feature_names[i] for i in top_indices]
    
    # Add units to feature names
    top_feature_labels = []
    for feat in top_feature_names:
        unit = get_feature_unit(feat)
        if unit:
            top_feature_labels.append(f"{feat} ({unit})")
        else:
            top_feature_labels.append(feat)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use PFT full names for columns
    pft_display_names = [PFT_FULL_NAMES.get(p, p) for p in unique_pfts]
    
    im = ax.imshow(importance_subset, aspect='auto', cmap='YlOrRd')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f'Mean |SHAP Value| ({SHAP_UNITS})', fontsize=11)
    
    # Set ticks
    ax.set_xticks(range(n_pfts))
    ax.set_xticklabels(pft_display_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feature_labels, fontsize=10)
    
    # Add value annotations
    for i in range(top_n):
        for j in range(n_pfts):
            val = importance_subset[i, j]
            text_color = 'white' if val > importance_subset.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', 
                   fontsize=8, color=text_color, fontweight='bold')
    
    ax.set_xlabel('Plant Functional Type', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'Feature Importance by Plant Functional Type\n'
                 f'(Mean |SHAP Value| - Top {top_n} Features)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        save_path = output_dir / "shap_importance_heatmap_by_pft.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"    Saved: {save_path}")
    
    plt.close()
    return fig


def plot_top_features_per_pft(
    shap_values: np.ndarray,
    feature_names: List[str],
    pft_labels: np.ndarray,
    top_n: int = 5,
    output_dir: Optional[Path] = None
) -> plt.Figure:
    """
    Create bar plots showing top features for each PFT separately.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array
    feature_names : list
        List of feature names
    pft_labels : np.ndarray
        PFT label for each sample
    top_n : int
        Number of top features to show per PFT
    output_dir : Path, optional
        Directory to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    logging.info(f"Generating top {top_n} features plot for each PFT...")
    
    # Get unique PFTs
    unique_pfts = [p for p in np.unique(pft_labels) if p != 'Unknown']
    n_pfts = len(unique_pfts)
    
    if n_pfts == 0:
        logging.warning("No valid PFT labels found.")
        return None
    
    # Create subplots
    n_cols = min(3, n_pfts)
    n_rows = (n_pfts + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    
    if n_pfts == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, pft in enumerate(unique_pfts):
        ax = axes[i]
        mask = pft_labels == pft
        n_samples = mask.sum()
        
        if n_samples == 0:
            ax.text(0.5, 0.5, f"No data for {pft}", ha='center', va='center')
            continue
        
        # Calculate mean absolute SHAP for this PFT
        mean_abs_shap = np.abs(shap_values[mask]).mean(axis=0)
        
        # Get top features
        top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
        top_feat_names = [feature_names[idx] for idx in top_indices]
        top_values = mean_abs_shap[top_indices]
        
        # Add units to feature names
        top_feat_labels = []
        for feat in top_feat_names:
            unit = get_feature_unit(feat)
            if unit:
                top_feat_labels.append(f"{feat}\n({unit})")
            else:
                top_feat_labels.append(feat)
        
        # Create horizontal bar plot
        colors = [PFT_COLORS.get(pft, 'steelblue')] * top_n
        y_pos = np.arange(top_n)
        
        ax.barh(y_pos, top_values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_feat_labels, fontsize=9)
        ax.invert_yaxis()  # Top feature at top
        ax.set_xlabel(f'Mean |SHAP Value| ({SHAP_UNITS})', fontsize=10)
        
        pft_full_name = PFT_FULL_NAMES.get(pft, pft)
        ax.set_title(f"{pft} - {pft_full_name}\n(n={n_samples})", 
                    fontsize=11, fontweight='bold')
        #ax.grid(True, alpha=0.3, axis='x')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(f"Top {top_n} Most Important Features by Plant Functional Type", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_dir:
        save_path = output_dir / "shap_top_features_per_pft.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"    Saved: {save_path}")
    
    plt.close()
    return fig


def plot_pft_shap_summary(
    shap_values: np.ndarray,
    X_original: np.ndarray,
    feature_names: List[str],
    pft_labels: np.ndarray,
    top_n: int = 15,
    output_dir: Optional[Path] = None
) -> None:
    """
    Create SHAP summary (beeswarm) plots for each PFT separately.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array
    X_original : np.ndarray
        Original feature values for coloring
    feature_names : list
        List of feature names
    pft_labels : np.ndarray
        PFT label for each sample
    top_n : int
        Number of features to display
    output_dir : Path, optional
        Directory to save the plots
    """
    logging.info(f"Generating SHAP summary plots for each PFT...")
    
    # Get unique PFTs
    unique_pfts = [p for p in np.unique(pft_labels) if p != 'Unknown']
    
    for pft in unique_pfts:
        mask = pft_labels == pft
        n_samples = mask.sum()
        
        if n_samples < 10:
            logging.warning(f"  Skipping {pft} - only {n_samples} samples")
            continue
        
        logging.info(f"  Generating summary plot for {pft} (n={n_samples})...")
        
        # Subset data for this PFT
        shap_pft = shap_values[mask]
        X_pft = X_original[mask]
        
        # Create DataFrame for plotting
        df_X_pft = pd.DataFrame(X_pft, columns=feature_names)
        
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_pft, df_X_pft, show=False, max_display=top_n)
        
        pft_full_name = PFT_FULL_NAMES.get(pft, pft)
        plt.title(f"Feature Contributions - {pft} ({pft_full_name})\n"
                  f"n={n_samples} samples | SHAP units: {SHAP_UNITS}", 
                  fontsize=13, fontweight='bold')
        plt.xlabel(get_shap_label(), fontsize=11)
        plt.tight_layout()
        
        if output_dir:
            save_path = output_dir / f"shap_summary_{pft}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"    Saved: {save_path}")
        
        plt.close()


def plot_pft_contribution_comparison(
    shap_values: np.ndarray,
    feature_names: List[str],
    pft_labels: np.ndarray,
    output_dir: Optional[Path] = None
) -> plt.Figure:
    """
    Create a grouped bar chart comparing mean SHAP values across PFTs for top features.
    Shows both positive and negative contributions.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array
    feature_names : list
        List of feature names
    pft_labels : np.ndarray
        PFT label for each sample
    output_dir : Path, optional
        Directory to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    logging.info("Generating PFT contribution comparison plot...")
    
    # Get unique PFTs
    unique_pfts = [p for p in np.unique(pft_labels) if p != 'Unknown']
    n_pfts = len(unique_pfts)
    
    if n_pfts == 0:
        logging.warning("No valid PFT labels found.")
        return None
    
    # Get top 8 features by overall mean absolute SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:8]
    top_features = [feature_names[i] for i in top_indices]
    
    # Calculate mean SHAP (not absolute) for each feature and PFT
    mean_shap_by_pft = {}
    for pft in unique_pfts:
        mask = pft_labels == pft
        if mask.sum() > 0:
            mean_shap_by_pft[pft] = shap_values[mask].mean(axis=0)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(top_features))
    width = 0.8 / n_pfts
    
    for i, pft in enumerate(unique_pfts):
        values = [mean_shap_by_pft[pft][feature_names.index(f)] for f in top_features]
        offset = (i - n_pfts/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=pft, 
                     color=PFT_COLORS.get(pft, 'gray'), alpha=0.8, edgecolor='black')
    
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    
    # X-axis labels with units
    x_labels = []
    for feat in top_features:
        unit = get_feature_unit(feat)
        if unit:
            x_labels.append(f"{feat}\n({unit})")
        else:
            x_labels.append(feat)
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel(f'Mean SHAP Value ({SHAP_UNITS})', fontsize=12)
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_title('Mean Feature Contribution by Plant Functional Type\n'
                 '(Positive = increases prediction, Negative = decreases prediction)', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='PFT', loc='upper right', bbox_to_anchor=(1.15, 1))
    #ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        save_path = output_dir / "shap_pft_contribution_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"    Saved: {save_path}")
    
    plt.close()
    return fig


def plot_pft_radar_chart(
    shap_values: np.ndarray,
    feature_names: List[str],
    pft_labels: np.ndarray,
    top_n: int = 8,
    output_dir: Optional[Path] = None
) -> plt.Figure:
    """
    Create radar charts comparing feature importance across PFTs.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array
    feature_names : list
        List of feature names
    pft_labels : np.ndarray
        PFT label for each sample
    top_n : int
        Number of features to include in radar
    output_dir : Path, optional
        Directory to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    logging.info("Generating PFT radar chart...")
    
    # Get unique PFTs
    unique_pfts = [p for p in np.unique(pft_labels) if p != 'Unknown']
    n_pfts = len(unique_pfts)
    
    if n_pfts == 0:
        logging.warning("No valid PFT labels found.")
        return None
    
    # Get top features by overall importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
    top_features = [feature_names[i] for i in top_indices]
    
    # Calculate normalized importance for each PFT
    importance_by_pft = {}
    for pft in unique_pfts:
        mask = pft_labels == pft
        if mask.sum() > 0:
            imp = np.abs(shap_values[mask]).mean(axis=0)[top_indices]
            # Normalize to 0-1 for radar chart
            importance_by_pft[pft] = imp / imp.max() if imp.max() > 0 else imp
    
    # Create radar chart
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Compute angle for each feature
    angles = np.linspace(0, 2 * np.pi, top_n, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    for pft in unique_pfts:
        values = importance_by_pft[pft].tolist()
        values += values[:1]  # Close the polygon
        
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=f"{pft} ({PFT_FULL_NAMES.get(pft, pft)})",
               color=PFT_COLORS.get(pft, 'gray'))
        ax.fill(angles, values, alpha=0.15, color=PFT_COLORS.get(pft, 'gray'))
    
    # Set feature labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top_features, fontsize=10)
    
    ax.set_title('Feature Importance Profile by Plant Functional Type\n'
                 '(Normalized Mean |SHAP|)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    if output_dir:
        save_path = output_dir / "shap_pft_radar_chart.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"    Saved: {save_path}")
    
    plt.close()
    return fig


def generate_pft_shap_report(
    shap_values: np.ndarray,
    feature_names: List[str],
    pft_labels: np.ndarray,
    output_dir: Path
) -> pd.DataFrame:
    """
    Generate a comprehensive CSV report of SHAP statistics by PFT.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array
    feature_names : list
        List of feature names
    pft_labels : np.ndarray
        PFT label for each sample
    output_dir : Path
        Directory to save the report
        
    Returns:
    --------
    pd.DataFrame
        Report dataframe
    """
    logging.info("Generating PFT SHAP statistics report...")
    
    unique_pfts = [p for p in np.unique(pft_labels) if p != 'Unknown']
    
    report_data = []
    
    for pft in unique_pfts:
        mask = pft_labels == pft
        n_samples = mask.sum()
        
        if n_samples == 0:
            continue
        
        shap_pft = shap_values[mask]
        
        for i, feature in enumerate(feature_names):
            feat_shap = shap_pft[:, i]
            
            report_data.append({
                'PFT': pft,
                'PFT_Full_Name': PFT_FULL_NAMES.get(pft, pft),
                'N_Samples': n_samples,
                'Feature': feature,
                'Feature_Unit': get_feature_unit(feature),
                'Mean_SHAP': feat_shap.mean(),
                'Std_SHAP': feat_shap.std(),
                'Mean_Abs_SHAP': np.abs(feat_shap).mean(),
                'Median_SHAP': np.median(feat_shap),
                'Min_SHAP': feat_shap.min(),
                'Max_SHAP': feat_shap.max(),
                'Pct_Positive': (feat_shap > 0).mean() * 100,
                'Pct_Negative': (feat_shap < 0).mean() * 100,
            })
    
    report_df = pd.DataFrame(report_data)
    
    # Save report
    report_path = output_dir / "shap_statistics_by_pft.csv"
    report_df.to_csv(report_path, index=False)
    logging.info(f"    Saved: {report_path}")
    
    # Also create a summary pivot table
    pivot_df = report_df.pivot_table(
        index='Feature',
        columns='PFT',
        values='Mean_Abs_SHAP',
        aggfunc='mean'
    )
    pivot_path = output_dir / "shap_importance_pivot_by_pft.csv"
    pivot_df.to_csv(pivot_path)
    logging.info(f"    Saved: {pivot_path}")
    
    return report_df
FEATURE_UNITS = {
    # Target variable
    'sap_velocity': 'cm³ cm⁻² h⁻¹',
    
    # Meteorological variables
    'sw_in': 'W m⁻²',
    'ppfd_in': 'µmol m⁻² s⁻¹',
    'ext_rad': 'W m⁻²',
    'ta': '°C',
    'vpd': 'kPa',
    'ws': 'm s⁻¹',
    'rh': '%',
    'precip': 'mm',
    'precipitation': 'mm',
    
    # Soil variables
    'volumetric_soil_water_layer_1': 'm³ m⁻³',
    'soil_temperature_level_1': 'K',
    'swc': 'm³ m⁻³',
    
    # Vegetation/Site characteristics
    'canopy_height': 'm',
    'elevation': 'm',
    'LAI': 'm² m⁻²',
    'latitude': '°',
    'longitude': '°',
    'prcip/PET': 'dimensionless',
    
    # PFT categories (one-hot encoded, unitless)
    'MF': '',
    'DNF': '',
    'ENF': '',
    'EBF': '',
    'WSA': '',
    'WET': '',
    'DBF': '',
    'SAV': '',
    
    # Time features (cyclical, unitless)
    'Day sin': '',
    'Day cos': '',
    'Year sin': '',
    'Year cos': '',
    'Week sin': '',
    'Week cos': '',
    'Month sin': '',
    'Month cos': '',
    
   
}

# SHAP value units (same as target variable)
SHAP_UNITS = 'cm³ cm⁻² h⁻¹'


def get_feature_unit(feature_name: str) -> str:
    """
    Get the unit for a feature, handling windowed feature names.
    
    Parameters:
    -----------
    feature_name : str
        Feature name, potentially with time suffix (e.g., 'vpd_t-0', 'ta_t-2')
        
    Returns:
    --------
    str
        Unit string for the feature
    """
    # Remove time suffix if present (e.g., '_t-0', '_t-1', '_t-2')
    import re
    base_name = re.sub(r'_t-\d+$', '', feature_name)
    
    return FEATURE_UNITS.get(base_name, '')


def get_feature_label(feature_name: str, include_unit: bool = True) -> str:
    """
    Get a formatted label for a feature including its unit.
    
    Parameters:
    -----------
    feature_name : str
        Feature name
    include_unit : bool
        Whether to include the unit in parentheses
        
    Returns:
    --------
    str
        Formatted label like 'VPD (kPa)' or just 'VPD'
    """
    unit = get_feature_unit(feature_name)
    
    if include_unit and unit:
        return f"{feature_name} ({unit})"
    return feature_name


def get_shap_label(include_unit: bool = True) -> str:
    """
    Get the label for SHAP values with units.
    
    Parameters:
    -----------
    include_unit : bool
        Whether to include the unit
        
    Returns:
    --------
    str
        Label like 'SHAP Value (cm³ cm⁻² h⁻¹)'
    """
    if include_unit:
        return f"SHAP Value ({SHAP_UNITS})"
    return "SHAP Value"
# =============================================================================
# STATIC FEATURE AGGREGATION FUNCTIONS
# =============================================================================

def aggregate_static_feature_shap(
    shap_values: np.ndarray,
    windowed_feature_names: List[str],
    base_feature_names: List[str],
    input_width: int,
    static_features: List[str] = None,
    aggregation: str = 'sum'
) -> Tuple[np.ndarray, List[str]]:
    """
    Aggregate SHAP values for static features across time steps.
    
    For static features (that don't change within a window), it makes no sense
    to have separate SHAP values for each time step. This function combines them.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array of shape (n_samples, n_windowed_features)
    windowed_feature_names : list
        Feature names with time suffixes (e.g., 'temperature_t-2', 'elevation_t-0')
    base_feature_names : list
        Original feature names (without time suffixes)
    input_width : int
        Number of time steps in the window
    static_features : list, optional
        List of feature names that are static (don't change over time).
        If None, will use a default list of common static features.
    aggregation : str
        How to aggregate static features: 'sum' or 'mean'
        
    Returns:
    --------
    Tuple[np.ndarray, List[str]]
        (aggregated_shap_values, aggregated_feature_names)
    """
    if static_features is None:
        static_features = [
            'canopy_height', 'elevation', 'LAI', 'latitude', 'longitude',
            'MF', 'DNF', 'ENF', 'EBF', 'WSA', 'WET', 'DBF', 'SAV',
            'prcip/PET',
        ]
    
    n_samples = shap_values.shape[0]
    n_base_features = len(base_feature_names)
    
    # Identify which base features are static vs dynamic
    static_base_indices = [i for i, f in enumerate(base_feature_names) if f in static_features]
    dynamic_base_indices = [i for i, f in enumerate(base_feature_names) if f not in static_features]
    
    static_feature_list = [base_feature_names[i] for i in static_base_indices]
    dynamic_feature_list = [base_feature_names[i] for i in dynamic_base_indices]
    
    logging.info(f"Static features ({len(static_base_indices)}): {static_feature_list}")
    logging.info(f"Dynamic features ({len(dynamic_base_indices)}): {dynamic_feature_list}")
    
    # Build new feature list and aggregate SHAP values
    new_shap_columns = []
    new_feature_names = []
    
    # 1. Handle DYNAMIC features - keep separate by time step
    for t in range(input_width):
        time_offset = input_width - 1 - t
        suffix = f"_t-{time_offset}" if time_offset > 0 else "_t-0"
        
        for base_idx in dynamic_base_indices:
            windowed_idx = t * n_base_features + base_idx
            new_shap_columns.append(shap_values[:, windowed_idx])
            new_feature_names.append(f"{base_feature_names[base_idx]}{suffix}")
    
    # 2. Handle STATIC features - aggregate across time steps
    for base_idx in static_base_indices:
        feature_name = base_feature_names[base_idx]
        
        # Collect SHAP values for this feature across all time steps
        static_shap_across_time = []
        for t in range(input_width):
            windowed_idx = t * n_base_features + base_idx
            static_shap_across_time.append(shap_values[:, windowed_idx])
        
        # Stack and aggregate
        stacked = np.stack(static_shap_across_time, axis=1)  # (n_samples, input_width)
        
        if aggregation == 'sum':
            aggregated = stacked.sum(axis=1)
        elif aggregation == 'mean':
            aggregated = stacked.mean(axis=1)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        new_shap_columns.append(aggregated)
        new_feature_names.append(feature_name)  # No time suffix for static features
    
    # Combine into new array
    aggregated_shap = np.column_stack(new_shap_columns)
    
    logging.info(f"Aggregated SHAP shape: {aggregated_shap.shape}")
    logging.info(f"New feature count: {len(new_feature_names)} "
                f"(was {len(windowed_feature_names)})")
    
    return aggregated_shap, new_feature_names


def aggregate_static_feature_values(
    X_original: np.ndarray,
    base_feature_names: List[str],
    input_width: int,
    static_features: List[str] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Aggregate feature values for static features (for plotting axes).
    
    Takes the value from the most recent time step (t-0) for static features,
    since they should be the same across all time steps anyway.
    
    Parameters:
    -----------
    X_original : np.ndarray
        Original feature values (windowed), shape (n_samples, n_windowed_features)
    base_feature_names : list
        Original feature names (without time suffixes)
    input_width : int
        Number of time steps in the window
    static_features : list, optional
        List of feature names that are static
        
    Returns:
    --------
    Tuple[np.ndarray, List[str]]
        (aggregated_X, aggregated_feature_names)
    """
    if static_features is None:
        static_features = [
            'canopy_height', 'elevation', 'LAI', 'latitude', 'longitude',
            'MF', 'DNF', 'ENF', 'EBF', 'WSA', 'WET', 'DBF', 'SAV',
            'prcip/PET',
        ]
    
    n_base_features = len(base_feature_names)
    
    static_base_indices = [i for i, f in enumerate(base_feature_names) if f in static_features]
    dynamic_base_indices = [i for i, f in enumerate(base_feature_names) if f not in static_features]
    
    new_X_columns = []
    new_feature_names = []
    
    # 1. Dynamic features - keep separate by time step
    for t in range(input_width):
        time_offset = input_width - 1 - t
        suffix = f"_t-{time_offset}" if time_offset > 0 else "_t-0"
        
        for base_idx in dynamic_base_indices:
            windowed_idx = t * n_base_features + base_idx
            new_X_columns.append(X_original[:, windowed_idx])
            new_feature_names.append(f"{base_feature_names[base_idx]}{suffix}")
    
    # 2. Static features - use value from t-0 (most recent)
    last_time_step = input_width - 1
    for base_idx in static_base_indices:
        feature_name = base_feature_names[base_idx]
        windowed_idx = last_time_step * n_base_features + base_idx  # t-0 position
        new_X_columns.append(X_original[:, windowed_idx])
        new_feature_names.append(feature_name)
    
    aggregated_X = np.column_stack(new_X_columns)
    
    return aggregated_X, new_feature_names


def get_dynamic_features_only(
    shap_values: np.ndarray,
    feature_names: List[str],
    base_feature_names: List[str],
    input_width: int,
    static_features: List[str] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract only dynamic features from SHAP values (for temporal analysis).
    
    Parameters:
    -----------
    shap_values : np.ndarray
        Aggregated SHAP values
    feature_names : list
        Aggregated feature names
    base_feature_names : list
        Original base feature names
    input_width : int
        Number of time steps
    static_features : list, optional
        List of static feature names
        
    Returns:
    --------
    Tuple[np.ndarray, List[str]]
        (dynamic_shap_values, dynamic_feature_names)
    """
    if static_features is None:
        static_features = [
            'canopy_height', 'elevation', 'LAI', 'latitude', 'longitude',
            'MF', 'DNF', 'ENF', 'EBF', 'WSA', 'WET', 'DBF', 'SAV',
            'prcip/PET',
        ]
    
    # Find indices of dynamic features (those with time suffixes)
    dynamic_indices = []
    dynamic_names = []
    
    for i, name in enumerate(feature_names):
        # Check if this is a dynamic feature (has time suffix)
        is_static = name in static_features
        if not is_static:
            dynamic_indices.append(i)
            dynamic_names.append(name)
    
    dynamic_shap = shap_values[:, dynamic_indices]
    
    return dynamic_shap, dynamic_names
def generate_windowed_feature_names(base_feature_names: List[str], input_width: int) -> List[str]:
    """
    Generate feature names for windowed data.
    
    Parameters:
    -----------
    base_feature_names : list
        Original feature names (for a single time step)
    input_width : int
        Number of time steps in the window
        
    Returns:
    --------
    list
        Feature names with time step suffixes (e.g., 'temperature_t-2', 'temperature_t-1', 'temperature_t-0')
    """
    windowed_names = []
    for t in range(input_width):
        time_offset = input_width - 1 - t  # t-2, t-1, t-0 for input_width=3
        suffix = f"_t-{time_offset}" if time_offset > 0 else "_t-0"
        windowed_names.extend([f"{feat}{suffix}" for feat in base_feature_names])
    return windowed_names


def plot_seasonal_drivers_by_hemisphere(
    shap_values: np.ndarray,
    feature_names: List[str],
    timestamps: np.ndarray,
    latitudes: np.ndarray,
    top_n: int = 5,
    output_dir: Optional[Path] = None
) -> plt.Figure:
    """
    Seasonal Driver Analysis with hemisphere separation (Replicates Figure 7).
    
    Separates SHAP values into Positive (Pushing flow UP) and Negative (Pushing flow DOWN)
    contributions and stacks them on a Day-of-Year axis, with separate plots for
    Northern and Southern hemispheres.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array of shape (n_samples, n_features)
    feature_names : list
        List of feature names matching shap_values columns
    timestamps : np.ndarray
        Timestamps corresponding to each SHAP value row (MUST be same length as shap_values)
    latitudes : np.ndarray
        Latitude for each sample (MUST be same length as shap_values)
    top_n : int
        Number of top features to display individually
    output_dir : Path, optional
        Directory to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Validate inputs
    if len(shap_values) != len(timestamps):
        raise ValueError(f"Length mismatch: shap_values ({len(shap_values)}) != timestamps ({len(timestamps)})")
    if len(shap_values) != len(latitudes):
        raise ValueError(f"Length mismatch: shap_values ({len(shap_values)}) != latitudes ({len(latitudes)})")
    
    logging.info(f"Generating seasonal drivers plot with {len(shap_values)} samples...")
    
    # 1. Create DataFrame
    df = pd.DataFrame(shap_values, columns=feature_names)
    
    # Extract Day of Year (DOY)
    ts = pd.to_datetime(timestamps)
    if hasattr(ts, 'dayofyear'):
        df['DOY'] = ts.dayofyear
    else:
        df['DOY'] = ts.dt.dayofyear
    
    # Add hemisphere indicator
    df['is_southern'] = np.array(latitudes) < 0
    
    # 2. Identify Top Features globally (by total absolute impact)
    feature_cols = [c for c in df.columns if c not in ['DOY', 'is_southern']]
    total_impact = df[feature_cols].abs().sum().sort_values(ascending=False)
    top_features = total_impact.head(top_n).index.tolist()
    
    logging.info(f"Top {top_n} features by absolute SHAP impact: {top_features}")
    
    # 3. Define colors
    colors = sns.color_palette("husl", len(top_features) + 1)  # +1 for "Rest"
    
    # 4. Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    hemisphere_data = {
        'Northern Hemisphere': df[~df['is_southern']],
        'Southern Hemisphere': df[df['is_southern']]
    }
    
    for ax, (hemi_name, hemi_df) in zip(axes, hemisphere_data.items()):
        
        n_samples = len(hemi_df)
        logging.info(f"  {hemi_name}: {n_samples} samples")
        
        if n_samples == 0:
            ax.text(0.5, 0.5, f"No data for {hemi_name}", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(hemi_name, fontsize=13, fontweight='bold')
            continue
        
        # Calculate Mean SHAP per DOY for this hemisphere
        daily_avg = hemi_df.groupby('DOY')[feature_cols].mean()
        
        # Prepare positive/negative stacks
        pos_data = pd.DataFrame(index=daily_avg.index)
        neg_data = pd.DataFrame(index=daily_avg.index)
        
        plot_features = top_features.copy()
        
        for feat in top_features:
            if feat in daily_avg.columns:
                pos_data[feat] = daily_avg[feat].clip(lower=0)
                neg_data[feat] = daily_avg[feat].clip(upper=0)
            else:
                logging.warning(f"Feature '{feat}' not found in daily_avg columns")
        
        # "Rest" category (sum of all other features)
        other_cols = [c for c in feature_cols if c not in top_features]
        if other_cols:
            pos_data['Rest'] = daily_avg[other_cols].clip(lower=0).sum(axis=1)
            neg_data['Rest'] = daily_avg[other_cols].clip(upper=0).sum(axis=1)
            plot_features.append('Rest')
        
        color_map = dict(zip(plot_features, colors[:len(plot_features)]))
        
        # Net prediction line (sum of all SHAP values)
        net_prediction = daily_avg[feature_cols].sum(axis=1)
        
        # Plot stacked bars - Positive
        if not pos_data.empty:
            pos_data.plot(kind='bar', stacked=True, ax=ax, width=1.0,
                          color=[color_map.get(c, 'gray') for c in pos_data.columns],
                          legend=False, edgecolor='none')
        
        # Plot stacked bars - Negative
        if not neg_data.empty:
            neg_data.plot(kind='bar', stacked=True, ax=ax, width=1.0,
                          color=[color_map.get(c, 'gray') for c in neg_data.columns],
                          legend=False, edgecolor='none')
        
        # Plot net prediction line
        ax.plot(range(len(net_prediction)), net_prediction.values, 
                color='black', linewidth=2, label='Modeled Mean', zorder=10)
        
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_ylabel("SHAP Value (Contribution to Prediction)", fontsize=11)
        ax.set_title(f"{hemi_name} (n={n_samples:,})", fontsize=13, fontweight='bold')
        
        # Add season background shading
        if 'Northern' in hemi_name:
            # Northern Hemisphere seasons
            ax.axvspan(0, 59, alpha=0.08, color='blue')      # Winter (Jan-Feb)
            ax.axvspan(59, 151, alpha=0.08, color='green')   # Spring (Mar-May)
            ax.axvspan(151, 243, alpha=0.08, color='red')    # Summer (Jun-Aug)
            ax.axvspan(243, 334, alpha=0.08, color='orange') # Fall (Sep-Nov)
            ax.axvspan(334, 366, alpha=0.08, color='blue')   # Winter (Dec)
        else:
            # Southern Hemisphere seasons (flipped)
            ax.axvspan(0, 59, alpha=0.08, color='red')       # Summer (Jan-Feb)
            ax.axvspan(59, 151, alpha=0.08, color='orange')  # Fall (Mar-May)
            ax.axvspan(151, 243, alpha=0.08, color='blue')   # Winter (Jun-Aug)
            ax.axvspan(243, 334, alpha=0.08, color='green')  # Spring (Sep-Nov)
            ax.axvspan(334, 366, alpha=0.08, color='red')    # Summer (Dec)
        
         #ax.grid(True, alpha=0.3, axis='y')
    
    # X-axis formatting (on bottom plot only)
    ticks = np.arange(0, 366, 30)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels(ticks, rotation=0)
    axes[-1].set_xlabel("Day of Year", fontsize=12)
    
    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map.get(f, 'gray')) for f in plot_features]
    handles.append(plt.Line2D([0], [0], color='black', lw=2))
    labels = plot_features + ['Modeled Mean']
    
    # Add season legend
    season_handles = [
        plt.Rectangle((0, 0), 1, 1, alpha=0.3, color='blue'),
        plt.Rectangle((0, 0), 1, 1, alpha=0.3, color='green'),
        plt.Rectangle((0, 0), 1, 1, alpha=0.3, color='red'),
        plt.Rectangle((0, 0), 1, 1, alpha=0.3, color='orange'),
    ]
    season_labels = ['Winter', 'Spring', 'Summer', 'Fall']
    
    # Create two legends
    leg1 = fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.98), 
                      title="Drivers", fontsize=9)
    fig.legend(season_handles, season_labels, loc='upper right', bbox_to_anchor=(0.99, 0.60),
               title="Seasons (NH)", fontsize=9)
    fig.add_artist(leg1)  # Re-add first legend
    
    plt.suptitle("Seasonal Driver Analysis by Hemisphere", fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])
    
    if output_dir:
        save_path = output_dir / "fig7_seasonal_drivers_by_hemisphere.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved seasonal drivers plot to: {save_path}")
    
    plt.close()
    return fig


def plot_diurnal_drivers(
    shap_values: np.ndarray,
    feature_names: List[str],
    timestamps: np.ndarray,
    observed_values: np.ndarray = None,  # Observed values (original scale)
    base_value: float = 0.0,             # SHAP base value (transformed scale if IS_TRANSFORM)
    observed_mean: float = None,         # Mean of observed values (original scale) for centering
    top_n: int = 5,
    output_dir: Optional[Path] = None
) -> plt.Figure:
    """
    Diurnal Driver Analysis with Observed Data comparison.
    
    Both SHAP values and observed values are plotted as deviations from their means,
    allowing direct comparison even when they're on different scales.
    
    - SHAP values: sum of SHAP = deviation from model's expected value (base_value)
    - Observed values: plotted as deviation from observed mean
    """
    # 1. Create DataFrame
    df = pd.DataFrame(shap_values, columns=feature_names)
    
    # Extract Hour
    ts = pd.to_datetime(timestamps)
    if hasattr(ts, 'hour'):
        df['Hour'] = ts.hour
    else:
        df['Hour'] = ts.dt.hour

    # Add Observed data if provided
    if observed_values is not None:
        df['Observed'] = observed_values
        # Use provided mean or calculate from data
        if observed_mean is None:
            observed_mean = np.mean(observed_values)

    # 2. Identify Features (skip Hour and Observed)
    feature_cols = [c for c in df.columns if c not in ['Hour', 'Observed']]
    
    # If top_n equals total features, don't use "Rest"
    if top_n >= len(feature_cols):
        top_features = feature_cols # Use all
    else:
        total_impact = df[feature_cols].abs().sum().sort_values(ascending=False)
        top_features = total_impact.head(top_n).index.tolist()

    # 3. Colors
    colors = sns.color_palette("husl", len(top_features) + 1)
    
    # 4. Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate Hourly Means
    hourly_avg = df.groupby('Hour').mean()
    hourly_avg = hourly_avg.reindex(range(24), fill_value=0)
    
    # Prepare stacks
    pos_data = pd.DataFrame(index=hourly_avg.index)
    neg_data = pd.DataFrame(index=hourly_avg.index)
    
    plot_features = top_features.copy()
    
    # Fill feature data
    for feat in top_features:
        if feat in hourly_avg.columns:
            pos_data[feat] = hourly_avg[feat].clip(lower=0)
            neg_data[feat] = hourly_avg[feat].clip(upper=0)
            
    # Handle "Rest"
    other_cols = [c for c in feature_cols if c not in top_features]
    if other_cols:
        pos_data['Rest'] = hourly_avg[other_cols].clip(lower=0).sum(axis=1)
        neg_data['Rest'] = hourly_avg[other_cols].clip(upper=0).sum(axis=1)
        plot_features.append('Rest')

    color_map = dict(zip(plot_features, colors[:len(plot_features)]))

    # --- PLOT BARS ---
    bar_width = 0.8
    bottom_pos = np.zeros(24)
    bottom_neg = np.zeros(24)
    
    for feat in plot_features:
        if feat in pos_data.columns:
            ax.bar(pos_data.index, pos_data[feat], bottom=bottom_pos, 
                  width=bar_width, color=color_map.get(feat, 'gray'), 
                  label=feat, edgecolor='none')
            bottom_pos += pos_data[feat].values
            
            ax.bar(neg_data.index, neg_data[feat], bottom=bottom_neg,
                  width=bar_width, color=color_map.get(feat, 'gray'),
                  edgecolor='none')
            bottom_neg += neg_data[feat].values

    # --- PLOT LINES ---
    
    # 1. Modeled Net Effect (Sum of SHAP)
    # This represents (Predicted - BaseValue) on transformed scale
    net_prediction = hourly_avg[feature_cols].sum(axis=1)
    ax.plot(hourly_avg.index, net_prediction.values, 
            color='black', linewidth=3, linestyle='-',
            label='Modeled (Net SHAP)', zorder=10)

    # 2. Observed Mean (Relative to Observed Mean)
    # Both lines are now "deviation from mean" style, making them comparable
    if observed_values is not None:
        # Center observed around its own mean (not base_value which may be on different scale)
        obs_relative = hourly_avg['Observed'] - observed_mean
        ax.plot(hourly_avg.index, obs_relative.values, 
                color='red', linewidth=2.5, linestyle='--', marker='d', markersize=6,
                label='Observed (Centered)', zorder=11)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_ylabel(f"Deviation from Mean ({SHAP_UNITS})", fontsize=12)
    
    # Day/Night shading
    ax.axvspan(-0.5, 6, alpha=0.1, color='navy')
    ax.axvspan(18, 23.5, alpha=0.1, color='navy')
    ax.axvspan(6, 18, alpha=0.1, color='gold')
    
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right')
    ax.set_xlabel("Hour of Day (Local Solar Time)", fontsize=12)
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Ensure lines are at the top of legend
    ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.18, 1.0), 
              title="Drivers & Targets", fontsize=9)
    
    title_text = f"Diurnal Driver Analysis (n={len(df):,})\n(Grouped PFTs)"
    if observed_values is not None:
        title_text += f"\nBase Value (Model Average): {base_value:.2f}"
        
    plt.title(title_text, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.85, 1.0])
    
    if output_dir:
        save_path = output_dir / "fig_diurnal_drivers.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved diurnal drivers plot to: {save_path}")
    
    plt.close()
    return fig


def plot_diurnal_drivers_heatmap(
    shap_values: np.ndarray,
    feature_names: List[str],
    timestamps: np.ndarray,
    top_n: int = 10,
    output_dir: Optional[Path] = None
) -> plt.Figure:
    """
    Diurnal Driver Analysis as a heatmap.
    
    Shows mean SHAP values for top features across hours of the day.
    Rows = features, Columns = hours, Color = SHAP value.
    
    Uses local solar time, so no hemisphere separation needed.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array of shape (n_samples, n_features)
    feature_names : list
        List of feature names matching shap_values columns
    timestamps : np.ndarray
        Timestamps corresponding to each SHAP value row (local solar time)
    top_n : int
        Number of top features to display
    output_dir : Path, optional
        Directory to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Validate inputs
    if len(shap_values) != len(timestamps):
        raise ValueError(f"Length mismatch: shap_values ({len(shap_values)}) != timestamps ({len(timestamps)})")
    
    logging.info(f"Generating diurnal heatmap with {len(shap_values)} samples...")
    
    # 1. Create DataFrame
    df = pd.DataFrame(shap_values, columns=feature_names)
    
    # Extract Hour of Day from local solar time
    ts = pd.to_datetime(timestamps)
    if hasattr(ts, 'hour'):
        df['Hour'] = ts.hour
    else:
        df['Hour'] = ts.dt.hour
    
    # 2. Identify Top Features (by total absolute impact)
    feature_cols = [c for c in df.columns if c != 'Hour']
    total_impact = df[feature_cols].abs().sum().sort_values(ascending=False)
    top_features = total_impact.head(top_n).index.tolist()
    
    # 3. Calculate hourly means for top features
    hourly_avg = df.groupby('Hour')[top_features].mean()
    hourly_avg = hourly_avg.reindex(range(24), fill_value=0)
    
    # 4. Create heatmap
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Transpose so features are rows and hours are columns
    heatmap_data = hourly_avg[top_features].T
    
    # Use diverging colormap centered at 0
    vmax = max(abs(heatmap_data.values.min()), abs(heatmap_data.values.max()))
    
    im = ax.imshow(heatmap_data.values, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(f'Mean {get_shap_label()}', fontsize=12)
    
    # Set ticks
    ax.set_xticks(range(24))
    ax.set_xticklabels([f'{h:02d}' for h in range(24)])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    
    # Labels
    ax.set_xlabel('Hour of Day (Local Solar Time)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'Diurnal Pattern of Feature Contributions (n={len(df):,})\n'
                 f'(Mean SHAP Values in {SHAP_UNITS})', 
                 fontsize=14, fontweight='bold')
    
    # Add grid lines between hours
    ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(top_features), 1), minor=True)
    #ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
    
    # Add vertical lines for dawn/dusk (approximate)
    ax.axvline(x=5.5, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Dawn (~6:00)')
    ax.axvline(x=17.5, color='purple', linestyle='--', linewidth=2, alpha=0.7, label='Dusk (~18:00)')
    
    # Add text annotations for significant values
    threshold = vmax * 0.5  # Only annotate values > 50% of max
    for i in range(len(top_features)):
        for j in range(24):
            val = heatmap_data.values[i, j]
            if abs(val) > threshold:
                text_color = 'white' if abs(val) > vmax * 0.7 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       fontsize=7, color=text_color, fontweight='bold')
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    
    plt.tight_layout()
    
    if output_dir:
        save_path = output_dir / "fig_diurnal_drivers_heatmap.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved diurnal heatmap to: {save_path}")
    
    plt.close()
    return fig


def plot_diurnal_feature_lines(
    shap_values: np.ndarray,
    feature_names: List[str],
    timestamps: np.ndarray,
    top_n: int = 6,
    output_dir: Optional[Path] = None
) -> plt.Figure:
    """
    Diurnal Driver Analysis as line plots with confidence intervals.
    
    Shows mean SHAP values ± 95% CI for each feature across hours of the day.
    Separate subplots for each top feature.
    
    Uses local solar time, so no hemisphere separation needed.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array of shape (n_samples, n_features)
    feature_names : list
        List of feature names matching shap_values columns
    timestamps : np.ndarray
        Timestamps corresponding to each SHAP value row (local solar time)
    top_n : int
        Number of top features to display (max 9 recommended)
    output_dir : Path, optional
        Directory to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Validate inputs
    if len(shap_values) != len(timestamps):
        raise ValueError(f"Length mismatch: shap_values ({len(shap_values)}) != timestamps ({len(timestamps)})")
    
    logging.info(f"Generating diurnal line plots with {len(shap_values)} samples...")
    
    # 1. Create DataFrame
    df = pd.DataFrame(shap_values, columns=feature_names)
    
    # Extract Hour of Day from local solar time
    ts = pd.to_datetime(timestamps)
    if hasattr(ts, 'hour'):
        df['Hour'] = ts.hour
    else:
        df['Hour'] = ts.dt.hour
    
    # 2. Identify Top Features
    feature_cols = [c for c in df.columns if c != 'Hour']
    total_impact = df[feature_cols].abs().sum().sort_values(ascending=False)
    top_features = total_impact.head(top_n).index.tolist()
    
    # 3. Create subplots
    n_cols = 3
    n_rows = (top_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True)
    axes = axes.flatten()
    
    n_samples = len(df)
    
    for i, feature in enumerate(top_features):
        ax = axes[i]
        
        # Calculate mean and std per hour
        hourly_stats = df.groupby('Hour')[feature].agg(['mean', 'std', 'count'])
        hourly_stats = hourly_stats.reindex(range(24))
        
        # Calculate 95% CI
        hourly_stats['se'] = hourly_stats['std'] / np.sqrt(hourly_stats['count'])
        hourly_stats['ci95'] = 1.96 * hourly_stats['se']
        
        hours = hourly_stats.index
        means = hourly_stats['mean'].values
        ci = hourly_stats['ci95'].fillna(0).values
        
        # Plot line with confidence interval
        ax.plot(hours, means, color='steelblue', linewidth=2, 
               marker='o', markersize=4)
        ax.fill_between(hours, means - ci, means + ci, 
                       color='steelblue', alpha=0.2, label='95% CI')
        
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_title(feature, fontsize=11, fontweight='bold')
       # ax.grid(True, alpha=0.3)
        
        # Add day/night shading
        ax.axvspan(-0.5, 6, alpha=0.08, color='navy')
        ax.axvspan(18, 23.5, alpha=0.08, color='navy')
        ax.axvspan(6, 18, alpha=0.08, color='gold')
        
     
        if i % n_cols == 0:
            ax.set_ylabel(get_shap_label(), fontsize=10)
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Hour of Day', fontsize=10)
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # Set x-axis ticks on all visible plots
    for ax in axes[:top_n]:
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels([f'{h:02d}' for h in range(0, 24, 3)])
        ax.set_xlim(-0.5, 23.5)
    
 
    plt.suptitle(f'Diurnal Pattern of Feature Contributions (n={n_samples:,})\n'
                 f'(Mean SHAP ± 95% CI in {SHAP_UNITS})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_dir:
        save_path = output_dir / "fig_diurnal_drivers_lines.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved diurnal line plots to: {save_path}")
    
    plt.close()
    return fig



def plot_interaction_dependencies(
    shap_values: np.ndarray,
    X_original: np.ndarray,
    feature_names: List[str],
    interaction_pairs: Optional[List[Tuple[str, str]]] = None,
    output_dir: Optional[Path] = None,
    cmap: str = 'coolwarm'
) -> plt.Figure:
    """
    Dependence plots with interaction coloring (Replicates Figure 8).
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values array
    X_original : np.ndarray
        Original feature values (for human-readable axes)
    feature_names : list
        List of feature names
    interaction_pairs : list of tuples, optional
        Explicit pairs like [('T_air', 'Daylength'), ('VPD', 'SWC')]
        If None, automatically selects top features and finds best interaction
    output_dir : Path, optional
        Directory to save the plot
    cmap : str, optional
        Colormap for interaction coloring (default: 'coolwarm')
        Options: 'coolwarm', 'RdBu_r', 'viridis', 'plasma', 'Spectral_r', 'seismic'
        
    Returns:
    --------
    matplotlib.figure.Figure
    """
    logging.info("Generating interaction dependence plots...")
    
    # Validate inputs
    if len(shap_values) != len(X_original):
        raise ValueError(f"Length mismatch: shap_values ({len(shap_values)}) != X_original ({len(X_original)})")
    if shap_values.shape[1] != len(feature_names):
        raise ValueError(f"Feature count mismatch: shap_values has {shap_values.shape[1]} features, "
                        f"but {len(feature_names)} names provided")
    
    # Create DataFrames
    df_X = pd.DataFrame(X_original, columns=feature_names)
    df_shap = pd.DataFrame(shap_values, columns=feature_names)
    
    # Determine which features/pairs to plot
    if interaction_pairs is not None:
        # Use explicit pairs
        features_to_plot = [(pair[0], pair[1]) for pair in interaction_pairs]
        n_plots = len(features_to_plot)
    else:
        # Auto-select top 3 features by mean absolute SHAP
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:3]
        top_features = [feature_names[i] for i in top_indices]
        
        # Find best interaction partner for each
        features_to_plot = []
        for feature in top_features:
            # Find feature with highest correlation to SHAP value residual
            candidates = [f for f in feature_names if f != feature]
            if 'sw_in' in candidates:
                inter_feat = 'sw_in'
            elif 'vpd' in candidates:
                inter_feat = 'vpd'
            elif 'ta' in candidates:
                inter_feat = 'ta'
            else:
                inter_feat = candidates[0] if candidates else feature
            features_to_plot.append((feature, inter_feat))
        n_plots = len(features_to_plot)
    
    logging.info(f"Plotting {n_plots} interaction pairs: {features_to_plot}")
    
    # Create subplots
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    
    # Handle single plot case
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (feature, inter_feat) in enumerate(features_to_plot):
        ax = axes[i]
        
        # Check if features exist
        if feature not in df_X.columns:
            logging.warning(f"Feature '{feature}' not found in data. Skipping.")
            ax.text(0.5, 0.5, f"Feature '{feature}' not found", ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_frame_on(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            continue
        if inter_feat not in df_X.columns:
            logging.warning(f"Interaction feature '{inter_feat}' not found. Using feature values for color.")
            inter_feat = feature
        
        # Get data
        x_data = df_X[feature].values
        y_data = df_shap[feature].values
        c_data = df_X[inter_feat].values
        
        # Remove NaN/inf for plotting
        valid_mask = np.isfinite(x_data) & np.isfinite(y_data) & np.isfinite(c_data)
        x_plot = x_data[valid_mask]
        y_plot = y_data[valid_mask]
        c_plot = c_data[valid_mask]
        
        if len(x_plot) == 0:
            ax.text(0.5, 0.5, "No valid data", ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_frame_on(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            continue
        
        # Scatter plot with colormap
        scatter = ax.scatter(x_plot, y_plot, c=c_plot, cmap=cmap, 
                            s=10, alpha=0.6, edgecolor='none')
        
        # Add colorbar with units
        inter_unit = get_feature_unit(inter_feat)
        if inter_unit:
            cbar_label = f"{inter_feat} ({inter_unit})"
        else:
            cbar_label = inter_feat
        cbar = plt.colorbar(scatter, ax=ax, location='bottom', pad=0.15, aspect=30)
        cbar.set_label(cbar_label, fontsize=10)
        
        
        # Formatting with units
        feat_unit = get_feature_unit(feature)
        if feat_unit:
            xlabel = f"{feature} ({feat_unit})"
        else:
            xlabel = feature
        
        ax.axhline(0, color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel(get_shap_label() if i % n_cols == 0 else "", fontsize=11)
        ax.set_title(f"Driver: {feature}\nInteraction: {inter_feat}", fontsize=11)
       #ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle(f"Partial Dependence with Interactions\n(SHAP Values in {SHAP_UNITS})", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_dir:
        save_path = output_dir / "fig8_dependence_interaction.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved interaction dependence plot to: {save_path}")
    
    plt.close()
    return fig


def run_shap_analysis(
    model,
    X_scaled: np.ndarray,
    X_original: np.ndarray,
    y_all: np.ndarray,  # <--- ADD THIS
    feature_names: List[str],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    site_ids: np.ndarray,
    output_dir: Path,
    sample_size: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates comprehensive SHAP analysis plots.
    
    FIXED VERSION: Returns sampled indices for proper alignment with external data.
    
    Parameters:
    -----------
    model : trained model
        The trained model (XGBoost/RandomForest/etc)
    X_scaled : np.ndarray
        The data used for prediction (scaled/transformed)
    X_original : np.ndarray
        The original data (for human-readable x-axes)
    feature_names : list
        List of feature names
    latitudes, longitudes : np.ndarray
        Coordinates for spatial plotting
    site_ids : np.ndarray
        Site identifiers for each sample
    output_dir : Path
        Where to save plots
    sample_size : int
        Number of samples to calculate SHAP for (computation is expensive)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        (shap_values, sampled_indices) - SHAP values and the indices used for sampling
    """
    logging.info("="*60)
    logging.info("Starting SHAP Analysis")
    logging.info("="*60)
    logging.info(f"Input shapes: X_scaled={X_scaled.shape}, X_original={X_original.shape}")
    logging.info(f"Number of features: {len(feature_names)}")
    logging.info(f"Sample size for SHAP: {sample_size}")
    
    # Validate inputs
    if X_scaled.shape[1] != len(feature_names):
        raise ValueError(f"Feature count mismatch: X_scaled has {X_scaled.shape[1]} features, "
                        f"but {len(feature_names)} names provided")
    if X_scaled.shape[1] != X_original.shape[1]:
        raise ValueError(f"Shape mismatch: X_scaled has {X_scaled.shape[1]} features, "
                        f"but X_original has {X_original.shape[1]}")
    
    # 1. Sampling (SHAP is slow on large datasets)
    n_total = len(X_scaled)
    if n_total > sample_size:
        logging.info(f"Sampling {sample_size} from {n_total} total samples...")
        np.random.seed(42)  # For reproducibility
        sampled_indices = np.random.choice(n_total, sample_size, replace=False)
        sampled_indices = np.sort(sampled_indices)  # Sort for consistency
        
        X_bg = X_scaled[sampled_indices]
        X_org_bg = X_original[sampled_indices]
        y_bg = y_all[sampled_indices]
        lat_bg = latitudes[sampled_indices]
        lon_bg = longitudes[sampled_indices]
        site_bg = site_ids[sampled_indices]
    else:
        logging.info(f"Using all {n_total} samples (below sample_size threshold)")
        sampled_indices = np.arange(n_total)
        X_bg = X_scaled
        X_org_bg = X_original
        lat_bg = latitudes
        lon_bg = longitudes
        site_bg = site_ids
        y_bg = y_all

    logging.info(f"SHAP calculation on {len(X_bg)} samples")

    # 2. Calculate SHAP Values
    logging.info("Creating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    logging.info("Calculating SHAP values (this may take a while)...")
    shap_values = explainer.shap_values(X_bg)
    
    # Handle multi-output models (some sklearn versions return list)
    if isinstance(shap_values, list):
        logging.info("Model returned list of SHAP values, using first element")
        shap_values = shap_values[0]
    
    logging.info(f"SHAP values shape: {shap_values.shape}")
    
    # Handle expected_value (can be array for some models)
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value[0] if len(base_value) > 0 else base_value
    logging.info(f"Base value (expected_value): {base_value}")

    # Create DataFrames for easier handling
    df_shap = pd.DataFrame(shap_values, columns=feature_names)
    df_X = pd.DataFrame(X_org_bg, columns=feature_names)

    # =========================================================
    # PLOT 1: Global Importance (Beeswarm Summary)
    # =========================================================
    logging.info("Generating Global Importance Plot (Beeswarm)...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, df_X, show=False, max_display=20)
    plt.title("Feature Contribution to Prediction", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_beeswarm.png", dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"  Saved: shap_summary_beeswarm.png")

    # =========================================================
    # PLOT 2: Global Importance (Bar)
    # =========================================================
    logging.info("Generating Global Importance Plot (Bar)...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, df_X, plot_type="bar", show=False, max_display=20)
    plt.title("Average Magnitude of Feature Contribution", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "shap_global_importance_bar.png", dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"  Saved: shap_global_importance_bar.png")

    # =========================================================
    # PLOT 3: Partial Dependence (Top 9 Features)
    # =========================================================
    logging.info("Generating Partial Dependence Plots...")
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:9]
    top_features = [feature_names[i] for i in top_indices]
    
    logging.info(f"  Top 9 features: {top_features}")

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    for i, feature in enumerate(top_features):
        ax = axes[i]
        
        x_val = df_X[feature].values
        y_val = df_shap[feature].values
        
        # Filter valid values
        valid_mask = np.isfinite(x_val) & np.isfinite(y_val)
        x_valid = x_val[valid_mask]
        y_valid = y_val[valid_mask]
        
        if len(x_valid) == 0:
            ax.text(0.5, 0.5, "No valid data", ha='center', va='center')
            continue
        
        # Scatter plot
        ax.scatter(x_valid, y_valid, alpha=0.3, color='black', s=10)
        
        # Add smooth curve (LOWESS)
        try:
            sns.regplot(x=x_valid, y=y_valid, scatter=False, lowess=True, 
                        ax=ax, color='blue', line_kws={'linewidth': 2})
        except Exception as e:
            logging.warning(f"  LOWESS failed for {feature}: {e}")

        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title(f"Effect of {feature}", fontsize=12, fontweight='bold')
        ax.set_xlabel(f"{feature}", fontsize=10)
        ax.set_ylabel("SHAP Value", fontsize=10)
       #ax.grid(True, alpha=0.3)
        
    plt.suptitle("Partial Dependence Plots (Top 9 Features)", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_dir / "shap_partial_dependence.png", dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"  Saved: shap_partial_dependence.png")

    # =========================================================
    # PLOT 4: Spatial SHAP Maps (Top 4 Features)
    # =========================================================
    logging.info("Generating Spatial SHAP Maps...")
    
    top_4_features = top_features[:4]

    fig = plt.figure(figsize=(20, 12))
    
    for i, feature in enumerate(top_4_features):
        ax = fig.add_subplot(2, 2, i+1, projection=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        
        vals = df_shap[feature].values
        
        # Filter valid values
        valid_mask = np.isfinite(vals) & np.isfinite(lon_bg) & np.isfinite(lat_bg)
        
        if valid_mask.sum() == 0:
            ax.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax.transAxes)
            continue
        
        vals_valid = vals[valid_mask]
        lon_valid = lon_bg[valid_mask]
        lat_valid = lat_bg[valid_mask]
        
        # Diverging colormap centered at 0
        vmax = max(abs(vals_valid.min()), abs(vals_valid.max()))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        
        scatter = ax.scatter(lon_valid, lat_valid, c=vals_valid, s=30, 
                             cmap='RdBu_r', norm=norm, transform=ccrs.PlateCarree(),
                             edgecolor='k', linewidth=0.3, alpha=0.7)
        
        plt.colorbar(scatter, ax=ax, shrink=0.7, label=f'SHAP Value')
        ax.set_title(f"Spatial Contribution: {feature}", fontsize=13, fontweight='bold')
        
        # Set extent based on data
        ax.set_extent([lon_valid.min()-5, lon_valid.max()+5, 
                      lat_valid.min()-5, lat_valid.max()+5], 
                      crs=ccrs.PlateCarree())
        
    plt.suptitle("Spatial Distribution of Feature Contributions", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_dir / "shap_spatial_maps.png", dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"  Saved: shap_spatial_maps.png")

    # =========================================================
    # PLOT 5: Waterfall Plots (High/Low Predictions)
    # =========================================================
    logging.info("Generating Local Waterfall Plots...")
    
    pred_vals = model.predict(X_bg)
    high_idx = np.argmax(pred_vals)
    low_idx = np.argmin(pred_vals)
    
    for idx, name in [(high_idx, "High_Flow"), (low_idx, "Low_Flow")]:
        plt.figure(figsize=(10, 8))
        
        # Create Explanation object
        row_explainer = shap.Explanation(
            values=shap_values[idx],
            base_values=float(base_value),  # Ensure it's a scalar
            data=X_org_bg[idx],
            feature_names=feature_names
        )
        
        shap.plots.waterfall(row_explainer, show=False, max_display=12)
        plt.title(f"Why did the model predict {name}?\nSite: {site_bg[idx]}", 
                 fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_waterfall_{name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"  Saved: shap_waterfall_{name}.png")

    logging.info("="*60)
    logging.info("SHAP Analysis Complete!")
    logging.info(f"All plots saved to: {output_dir}")
    logging.info("="*60)
    
    # Return both SHAP values AND the sampled indices
    return shap_values, sampled_indices


# =============================================================================
# ORIGINAL HELPER FUNCTIONS (unchanged)
# =============================================================================

def add_sap_flow_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Add scientifically-grounded features for sap velocity prediction.
    """
    df = df.copy()
    new_features = []
    
    # Priority 1: Highest Impact Features
    if 'vpd' in df.columns and 'sw_in' in df.columns:
        df['vpd_x_sw_in'] = df['vpd'] * df['sw_in']
        new_features.append('vpd_x_sw_in')
    
    if 'vpd' in df.columns:
        df['vpd_squared'] = df['vpd'] ** 2
        new_features.append('vpd_squared')
    
    if 'vpd' in df.columns:
        df['vpd_lag_1h'] = df['vpd'].shift(1)
        new_features.append('vpd_lag_1h')
    
    if 'sw_in' in df.columns:
        df['is_daytime'] = (df['sw_in'] > 10).astype(np.float32)
        new_features.append('is_daytime')
    
    if 'latitude' in df.columns:
        df['southern_hemisphere'] = (df['latitude'] < 0).astype(np.float32)
        new_features.append('southern_hemisphere')
    
    # Priority 2: High Impact Features
    if 'vpd' in df.columns:
        df['vpd_mean_6h'] = df['vpd'].rolling(6, min_periods=1).mean()
        new_features.append('vpd_mean_6h')
    
    if 'vpd' in df.columns:
        df['vpd_high'] = (df['vpd'] > 2.5).astype(np.float32)
        new_features.append('vpd_high')
    
    if 'sw_in' in df.columns and 'ext_rad' in df.columns:
        df['clear_sky_index'] = (df['sw_in'] / (df['ext_rad'] + 1)).clip(0, 1)
        new_features.append('clear_sky_index')
    
    if 'canopy_height' in df.columns and 'vpd' in df.columns:
        df['height_x_vpd'] = df['canopy_height'] * df['vpd']
        new_features.append('height_x_vpd')
    
    if 'ta' in df.columns:
        df['ta_lag_1h'] = df['ta'].shift(1)
        new_features.append('ta_lag_1h')
    
    if 'ta' in df.columns:
        df['gdd'] = (df['ta'] - 5).clip(lower=0)
        new_features.append('gdd')
    
    # Priority 3: Additional Useful Features
    if 'vpd' in df.columns:
        df['vpd_log'] = np.log1p(df['vpd'])
        new_features.append('vpd_log')
    
    if 'sw_in' in df.columns and 'LAI' in df.columns:
        k = 0.5
        df['absorbed_radiation'] = df['sw_in'] * (1 - np.exp(-k * df['LAI']))
        new_features.append('absorbed_radiation')
    
    if 'ws' in df.columns and 'vpd' in df.columns:
        df['wind_x_vpd'] = df['ws'] * df['vpd']
        new_features.append('wind_x_vpd')
    
    if 'ta' in df.columns and 'vpd' in df.columns:
        df['ta_x_vpd'] = df['ta'] * df['vpd']
        new_features.append('ta_x_vpd')
    
    if 'vpd' in df.columns and 'prcip/PET' in df.columns:
        df['demand_x_supply'] = df['vpd'] * df['prcip/PET']
        new_features.append('demand_x_supply')
    
    if 'sw_in' in df.columns:
        df['sw_in_cumsum_day'] = df['sw_in'].rolling(24, min_periods=1).sum()
        new_features.append('sw_in_cumsum_day')
    
    if 'vpd' in df.columns:
        df['vpd_cumsum_6h'] = df['vpd'].rolling(6, min_periods=1).sum()
        new_features.append('vpd_cumsum_6h')
    
    if 'ta' in df.columns:
        df['ta_change_1h'] = df['ta'].diff(1)
        new_features.append('ta_change_1h')
    
    if 'latitude' in df.columns:
        df['tropical'] = (abs(df['latitude']) < 23.5).astype(np.float32)
        df['boreal'] = (abs(df['latitude']) > 55).astype(np.float32)
        new_features.extend(['tropical', 'boreal'])
    
    # Conditional Features
    soil_cols = [c for c in df.columns if any(x in c.lower() for x in 
                 ['swc', 'soil_moisture', 'sm_', 'vwc'])]
    if soil_cols:
        sm_col = soil_cols[0]
        sm_min, sm_max = df[sm_col].quantile([0.05, 0.95])
        df['soil_moisture_rel'] = ((df[sm_col] - sm_min) / (sm_max - sm_min + 0.01)).clip(0, 1)
        df['soil_dry'] = (df['soil_moisture_rel'] < 0.3).astype(np.float32)
        new_features.extend(['soil_moisture_rel', 'soil_dry'])
    
    if 'rh' in df.columns and 'ta' in df.columns:
        a, b = 17.27, 237.7
        alpha = (a * df['ta'] / (b + df['ta'])) + np.log(df['rh']/100 + 0.01)
        df['dew_point'] = b * alpha / (a - alpha)
        df['dew_point_depression'] = df['ta'] - df['dew_point']
        new_features.extend(['dew_point', 'dew_point_depression'])
    
    precip_col = None
    for col in ['precip', 'precipitation', 'rain', 'prcp']:
        if col in df.columns:
            precip_col = col
            break
    
    if precip_col:
        df['precip_sum_24h'] = df[precip_col].rolling(24, min_periods=1).sum()
        df['precip_sum_72h'] = df[precip_col].rolling(72, min_periods=1).sum()
        rain_events = df[precip_col] > 1
        df['hours_since_rain'] = (~rain_events).groupby((~rain_events).cumsum()).cumcount()
        new_features.extend(['precip_sum_24h', 'precip_sum_72h', 'hours_since_rain'])
    
    if 'dbh' in df.columns:
        df['dbh_log'] = np.log1p(df['dbh'])
        df['sapwood_area_est'] = 0.5 * df['dbh'] ** 1.8
        new_features.extend(['dbh_log', 'sapwood_area_est'])
    
    if verbose:
        print(f"Added {len(new_features)} new features: {new_features}")
    
    return df


def setup_logging(logger_name, log_dir=None):
    """Set up basic logging configuration"""
    logger = logging.getLogger(logger_name)

    if log_dir is None:
        log_dir = Path('./outputs/logs')
    
    log_dir = Path(log_dir)
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f'{logger_name}_optimizer.log'

    handlers = [
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    warnings.filterwarnings("ignore")
    
    return logging.getLogger()


@deterministic
def add_time_features(df, datetime_column='solar_TIMESTAMP'):
    """Create cyclical time features from a datetime column or index."""
    df = df.copy()
    
    if datetime_column is not None:
        if datetime_column not in df.columns:
            raise ValueError(f"Column '{datetime_column}' not found in DataFrame")
        
        if df[datetime_column].isna().any():
            n_nulls = df[datetime_column].isna().sum()
            logging.warning(f"Dropping {n_nulls} rows with NaN in {datetime_column}")
            df = df.dropna(subset=[datetime_column])
        
        date_time = pd.to_datetime(df[datetime_column])
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                date_time = pd.to_datetime(df.index)
            except:
                raise ValueError("DataFrame index cannot be converted to datetime")
        else:
            date_time = df.index

    # Use vectorized conversion to avoid pandas version issues with tz-aware datetimes
    # Convert to nanoseconds since epoch, then to seconds
    timestamp_s = date_time.astype('int64') / 1e9
    day = 24 * 60 * 60
    year = 365.2425 * day
    week = 7 * day
    month = 30.44 * day
    
    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    df['Week sin'] = np.sin(timestamp_s * (2 * np.pi / week))
    df['Week cos'] = np.cos(timestamp_s * (2 * np.pi / week))
    df['Month sin'] = np.sin(timestamp_s * (2 * np.pi / month))
    df['Month cos'] = np.cos(timestamp_s * (2 * np.pi / month))
    
    return df


@deterministic
def create_windows_from_segments(
    segments: List[pd.DataFrame],
    input_width: int,
    label_width: int,
    shift: int,
    label_columns: List[str] = None,
    exclude_targets_from_features: bool = True,
    exclude_labels_from_inputs: bool = True
) -> List[Tuple[tf.Tensor, tf.Tensor]]:
    """Create windows from multiple segments without splitting them."""
    all_windows = []
   
    for segment in segments:
        min_segment_length = input_width + shift
        if len(segment) < min_segment_length:
            continue
       
        window_gen = WindowGenerator(
            input_width=input_width,
            label_width=label_width,
            shift=shift,
            data_df=segment,
            label_columns=label_columns,
            batch_size=1,
            shuffle=False,
            exclude_targets_from_features=exclude_targets_from_features,
            exclude_labels_from_inputs=exclude_labels_from_inputs
        )
       
        segment_ds = window_gen.dataset
        for inputs, labels in segment_ds:
            all_windows.append((inputs, labels))
   
    return all_windows


@deterministic
def get_predictions(model, test_windows, X_test, y_test, is_windowing=True):
    """Get predictions from the model using windows data."""
    if is_windowing:
        X_test, y_test = convert_windows_to_numpy(test_windows)

    y_pred = model.predict(X_test)
    
    return y_pred.flatten(), y_test.flatten()


@deterministic
def convert_windows_to_numpy(windows: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a list of window tuples to numpy arrays."""
    if not windows:
        return np.array([]), np.array([])
    
    features = [w[0] for w in windows]
    labels = [w[1] for w in windows]
    
    X = np.array(features)
    y = np.array(labels)
    
    X_reshaped = X.reshape(X.shape[0], -1)
    y_flattened = y.flatten()

    return X_reshaped, y_flattened


@deterministic
def create_spatial_groups(
    lat: Union[np.ndarray, List],
    lon: Union[np.ndarray, List],
    data_counts: Union[np.ndarray, List] = None,
    n_groups: int = None,
    balanced: bool = True,
    method: str = 'grid',
    lat_grid_size: float = 0.05,
    lon_grid_size: float = 0.05,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Group geographical data points using various spatial grouping methods."""
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    
    if method == 'grid':
        lat_bins = np.arange(
            np.floor(lat.min()),
            np.ceil(lat.max()) + lat_grid_size,
            lat_grid_size
        )
        lon_bins = np.arange(
            np.floor(lon.min()),
            np.ceil(lon.max()) + lon_grid_size,
            lon_grid_size
        )
        
        lat_indices = np.digitize(lat, lat_bins) - 1
        lon_indices = np.digitize(lon, lon_bins) - 1
        
        n_lon_bins = len(lon_bins)
        grid_cell_ids = lat_indices * n_lon_bins + lon_indices
        
        if data_counts is None or n_groups is None or not balanced:
            unique_groups = np.unique(grid_cell_ids)
            group_map = {g: n for n, g in enumerate(unique_groups)}
            groups = np.array([group_map[g] for g in grid_cell_ids])
        else:
            unique_cells, cell_indices = np.unique(grid_cell_ids, return_inverse=True)
            cell_data_counts = np.bincount(cell_indices, weights=data_counts)

            sorted_cell_indices = np.argsort(cell_data_counts)[::-1]

            fold_data_counts = np.zeros(n_groups)
            cell_to_fold = np.zeros(len(unique_cells), dtype=int)
            for cell_idx in sorted_cell_indices:
                target_fold = np.argmin(fold_data_counts)
                cell_to_fold[cell_idx] = target_fold
                fold_data_counts[target_fold] += cell_data_counts[cell_idx]

            groups = cell_to_fold[cell_indices]
            
    elif method == 'default':
        if not len(lat) == len(lon):
            raise ValueError("Latitude and longitude arrays must have the same length.")
        if data_counts is None or n_groups is None or not balanced:
            groups = np.arange(len(lat))
        else:
            sorted_indices = np.argsort(data_counts)[::-1]
            groups = np.zeros(len(lat), dtype=int) - 1
            fold_data_counts = np.zeros(n_groups)
            
            for idx in sorted_indices:
                target_group = np.argmin(fold_data_counts)
                groups[idx] = target_group
                fold_data_counts[target_group] += data_counts[idx]
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods are 'grid' or 'default'.")

    # Calculate group statistics
    stats = []
    for group in np.unique(groups):
        if group == -1:
            continue
            
        mask = groups == group
        stats.append({
            'group': group,
            'size': np.sum(mask),
            'mean_lat': np.mean(lat[mask]),
            'mean_lon': np.mean(lon[mask]),
            'std_lat': np.std(lat[mask]),
            'std_lon': np.std(lon[mask]),
            'min_lat': np.min(lat[mask]),
            'max_lat': np.max(lat[mask]),
            'min_lon': np.min(lon[mask]),
            'max_lon': np.max(lon[mask])
        })
    
    stats_df = pd.DataFrame(stats)
    
    if len(stats_df) > 0:
        logging.info("Spatial Group Statistics:")
        logging.info(f"Total groups: {len(stats_df)}")
        logging.info(f"Group details:\n{stats_df[['group', 'size', 'mean_lat', 'mean_lon']].to_string(index=False)}")
    
    return groups, stats_df


def parse_args():
    """Create and return the argument parser for this script."""
    parser = argparse.ArgumentParser(description="Machine learning models with spatial cross-validation")
    parser.add_argument('--RANDOM_SEED', type=int, default=42,help='Random seed for reproducibility')
    parser.add_argument('--model', type=str, default='xgb', help='Model type to train')
    parser.add_argument('--run_id', type=str, default='default_daily_nocoors_swcnor', help='Run identifier for logging')
    parser.add_argument('--n_groups', type=int, default=10, help='Number of spatial groups for cross-validation')
    parser.add_argument('--INPUT_WIDTH', type=int, default=2, help='Input width for time series windows')
    parser.add_argument('--LABEL_WIDTH', type=int, default=1, help='Label width for time series windows')
    parser.add_argument('--SHIFT', type=int, default=1, help='Shift for time series windows')
    parser.add_argument('--TARGET_COL', type=str, default='sap_velocity', help='Target column name')
    parser.add_argument('--EXCLUDE_LABELS', type=bool, default=True, help='Exclude labels from input features')
    parser.add_argument('--EXCLUDE_TARGETS', type=bool, default=True, help='Exclude targets from input features')
    parser.add_argument('--IS_WINDOWING', type=bool, default=False, help='Enable time windowing for data processing')
    parser.add_argument('--spatial_split_method', type=str, default='default', help='Method for spatial splitting of data')
    parser.add_argument('--hyperparameters', type=str, help='Path to the JSON file of hyperparameters')
    parser.add_argument('--IS_SHUFFLE', type=bool, default=True, help='Whether to enable shuffling of data')
    parser.add_argument('--N_ITERATIONS', type=int, default=None, help='Number of iterations for random search')
    parser.add_argument('--IS_CV', type=bool, default=True, help='Whether to enable cross-validation for inner loop')
    parser.add_argument('--IS_STRATIFIED', type=bool, default=True, help='Whether to use stratified sampling')
    parser.add_argument('--BALANCED', type=bool, default=False, help='Whether to balance the spatial groups')
    parser.add_argument('--SPLIT_TYPE', type=str, default='spatial_stratified', help='Type of data splitting strategy')
    parser.add_argument('--IS_ONLY_DAY', type=bool, default=False, help='Whether to use only day data')
    parser.add_argument('--additional_features', nargs='*', default=[], help='List of features')
    parser.add_argument('--TIME_SCALE', type=str, default='daily', help='Time scale of the data: hourly or daily')
    parser.add_argument('--SHAP_SAMPLE_SIZE', type=int, default=50000, help='Sample size for SHAP analysis')
    parser.add_argument('--IS_TRANSFORM', type=bool, default=True, help='Whether to apply target transformation')
    parser.add_argument('--TRANSFORM_METHOD', type=str, default='log1p',
                        choices=['log1p', 'sqrt', 'box-cox', 'yeo-johnson', 'none'],
                        help='Target transformation method: log1p, sqrt, box-cox, yeo-johnson, or none')
    return parser.parse_args()


@deterministic
def main(run_id="default"):
    """
    Main function implementing the leave-one-group-out spatial CV for an ML model.
    """
    # Unpack arguments
    args = parse_args()
    RANDOM_SEED = args.RANDOM_SEED
    INPUT_WIDTH = args.INPUT_WIDTH
    LABEL_WIDTH = args.LABEL_WIDTH
    SHIFT = args.SHIFT
    TARGET_COL = args.TARGET_COL
    EXCLUDE_LABELS = args.EXCLUDE_LABELS
    EXCLUDE_TARGETS = args.EXCLUDE_TARGETS
    IS_WINDOWING = args.IS_WINDOWING
    MODEL_TYPE = args.model
    IS_SHUFFLE = args.IS_SHUFFLE
    IS_STRATIFIED = args.IS_STRATIFIED
    N_ITERATIONS = args.N_ITERATIONS
    IS_CV = args.IS_CV
    IS_ONLY_DAY = args.IS_ONLY_DAY
    n_groups = args.n_groups
    spatial_split_method = args.spatial_split_method
    SPLIT_TYPE = args.SPLIT_TYPE
    BALANCED = args.BALANCED
    additional_features = args.additional_features
    TIME_SCALE = args.TIME_SCALE
    SHAP_SAMPLE_SIZE = args.SHAP_SAMPLE_SIZE
    IS_TRANSFORM = args.IS_TRANSFORM
    TRANSFORM_METHOD = args.TRANSFORM_METHOD if IS_TRANSFORM else 'none'
    
    # Initialize target transformer
    target_transformer = TargetTransformer(method=TRANSFORM_METHOD)
    
    with open(args.hyperparameters, 'r') as f:
        hyperparameters = json.load(f)
    
    run_id = args.run_id
    scale = 'sapwood'
    paths = PathConfig(scale=scale)
    setup_logging(logger_name=MODEL_TYPE, log_dir=paths.optimization_logs_dir)
    
    logging.info(f"Starting {MODEL_TYPE} model training with {SPLIT_TYPE} CV, seed {RANDOM_SEED})")
    logging.info(f"IS_CV={IS_CV}, n_groups={n_groups}, input_width={INPUT_WIDTH}, "
                f"label_width={LABEL_WIDTH}, shift={SHIFT}, exclude_labels={EXCLUDE_LABELS}, "
                f"exclude_targets={EXCLUDE_TARGETS}, is_windowing={IS_WINDOWING}, "
                f"spatial_split_method={spatial_split_method}, is_shuffle={IS_SHUFFLE}, "
                f"is_stratified={IS_STRATIFIED}, is_only_day={IS_ONLY_DAY}, time_scale={TIME_SCALE}, "
                f"is_transform={IS_TRANSFORM}, transform_method={TRANSFORM_METHOD}")

    # --- File Paths ---
    import os
    plot_dir = paths.hyper_tuning_plots_dir / MODEL_TYPE / run_id
    os.makedirs(str(plot_dir), exist_ok=True)
    model_dir = paths.models_root / MODEL_TYPE / run_id
    os.makedirs(str(model_dir), exist_ok=True)
    #data_dir = paths.merged_data_root / TIME_SCALE
    data_dir = paths.merged_daytime_only_dir / TIME_SCALE

    # --- Data Loading and Processing ---
    data_list = sorted(list(data_dir.glob(f'*{TIME_SCALE}.csv')))
    print(data_list)
    data_list = [f for f in data_list if 'all_biomes_merged' not in f.name]
    
    if not data_list:
        logging.critical(f"ERROR: No CSV files found in {data_dir}. Exiting.")
        return [], []

    site_data_dict = {}
    site_info_dict = {}
    
    base_features = [
        TARGET_COL, 
        'sw_in', ''
        'ws', 
        'precip', 
        'ta', 'ta_max', 'ta_min',
        'vpd', 'vpd_max', 'vpd_min',
        'ext_rad', 'ppfd_in',
        'pft', 'canopy_height', 
        'elevation',   
        'LAI', 'prcip/PET', 
        'volumetric_soil_water_layer_1', 'soil_temperature_level_1',
        'day_length'
    
        
    ]
    
    used_cols = list(set(base_features + additional_features))
    logging.info(f"Using columns: {used_cols}")
    
    all_possible_pft_types = ['MF', 'DNF', 'ENF', 'EBF', 'WSA', 'WET', 'DBF', 'SAV']
    max_sap = 0
    max_site = None
    final_feature_names = None
    
    for data_file in data_list:
        try:
            df = pd.read_csv(data_file, parse_dates=['TIMESTAMP'])
            logging.info(f"Loaded data for site {data_file.name}, shape: {df.shape}")
       
            if IS_ONLY_DAY:
                if 'sw_in' not in df.columns:
                    logging.warning(f"  Warning: 'sw_in' column not found in {data_file.name}. Skipping.")
                    continue
                df = df[df['sw_in'] > 10]
                if df.empty:
                    logging.warning(f"  Warning: No data after filtering for day time in {data_file.name}. Skipping.")
                    continue 
           
            site_id = df['site_name'].iloc[0]
            if df[TARGET_COL].max() > max_sap:
                max_sap = df[TARGET_COL].max()
                max_site = data_file.name
            
            site_max = df[TARGET_COL].max()
            if site_max > 100:
                logging.info(f"!!! HIGH VELOCITY ALERT: Site {site_id} has max value: {site_max:.2f}")
            
            lat_col = next((col for col in df.columns if col.lower() in ['lat', 'latitude_x']), None)
            lon_col = next((col for col in df.columns if col.lower() in ['lon', 'longitude_x']), None)
            pft_col = next((col for col in df.columns if col.lower() in ['pft', 'plant_functional_type', 'biome']), None)
            
            logging.info(f"PFT column found: {pft_col}")
            
            if not (lat_col and lon_col and pft_col):
                logging.warning(f"  Warning: Lat/Lon/PFT not found for site {site_id}. Skipping.")
                continue

            latitude = df[lat_col].median()
            longitude = df[lon_col].median()
        
            if site_id.startswith('CZE'):
                logging.info(f"  Skipping site {site_id} as per rules.")
                continue
            
            df['latitude'] = latitude
            df['longitude'] = longitude
            
            pft_value = df[pft_col].mode()[0] 
            print(pft_value)
            
            df.set_index('solar_TIMESTAMP', inplace=True)
            df.sort_index(inplace=True)  # Ensure chronological order
            df = add_time_features(df, datetime_column=None)
            
            if TIME_SCALE == 'hourly':
                time_features = ['Day sin', 'Year sin']
            elif TIME_SCALE == 'daily':
                time_features = ['Year sin']
            
            feature_cols = used_cols + time_features
            logging.info(f"Successfully added time features for site {site_id}, time features: {time_features}")

            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                logging.warning(f"Warning: Missing columns: {missing_cols} in {data_file.name}: skipping.")
                continue

            df = df[feature_cols].copy()
            print(df.describe())
            
            if 'pft' in used_cols:
                if 'pft' in df.columns:
                    orig_cols = [col for col in feature_cols if col != 'pft']
                    pft_cat = pd.Categorical(df['pft'], categories=all_possible_pft_types)
                    pft_df = pd.get_dummies(pft_cat)
                    pft_df.index = df.index
                    df = df[orig_cols].join(pft_df)
                else:
                    logging.warning(f"Warning: Missing pft column in {data_file.name}")
                    continue
            
            final_feature_names = [col for col in df.columns.tolist() if col != TARGET_COL]
            logging.info(f"Captured the final feature order: {final_feature_names}")
            logging.info(f"Total number of features before windowing: {len(final_feature_names)}")
            
            print(f"\n=== DEBUG for {site_id} ===")
            print("DataFrame dtypes:")
            print(df.dtypes)
            print("\nNon-numeric columns:")
            non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
            print(non_numeric)
            
            if non_numeric:
                print("\nSample values from non-numeric columns:")
                for col in non_numeric:
                    print(f"{col}: {df[col].head()}")
                    print(f"  Unique values: {df[col].unique()[:10]}")
            
            df = df.astype(np.float32)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(inplace=True)
            
            site_info = {
                'latitude': latitude, 
                'longitude': longitude, 
                'pft': pft_value
            }

            if IS_WINDOWING:
                min_segment_length = INPUT_WIDTH + SHIFT
                if len(df) < min_segment_length:
                    logging.warning(f"  Warning: Not enough data after cleaning for site {site_id}. Skipping.")
                    continue
                segments = TimeSeriesSegmenter.segment_time_series(
                    df, gap_threshold=2, 
                    unit='hours' if TIME_SCALE == 'hourly' else 'days', 
                    min_segment_length=min_segment_length
                )
                data_count = sum(len(s) for s in segments)
                if segments:
                    site_data_dict[site_id] = segments
                    site_info['data_count'] = data_count
                    site_info_dict[site_id] = site_info
                    logging.info(f"  {site_id}: Successfully processed {len(segments)} segments with {data_count} total records")
                else:
                    logging.warning(f"  {site_id}: No valid segments after time series segmentation. Skipping.")
            else:
                site_data_dict[site_id] = df
                minimum_required_length = 0
                if len(df) < minimum_required_length:
                    logging.warning(f"  Warning: Not enough data after cleaning for site {site_id}. Skipping.")
                    site_data_dict.pop(site_id)
                    continue
                site_info['data_count'] = len(df)
                site_info_dict[site_id] = site_info

        except Exception as e:
            logging.error(f"Error processing {data_file.name}: {e}")

    logging.info(f"\nSite with max {TARGET_COL}: {max_site} with value {max_sap}")
    
    if not site_data_dict:
        logging.critical("ERROR: No valid site data could be processed.")
        return [], []
    
    # --- Prepare Data for CV Split ---
    site_ids = np.array(list(site_info_dict.keys()))
    latitudes = [site_info_dict[site]['latitude'] for site in site_ids]
    longitudes = [site_info_dict[site]['longitude'] for site in site_ids]
    data_counts = [site_info_dict[site]['data_count'] for site in site_ids]
    
    logging.info("\nCreating spatial groups...")
    
    if IS_STRATIFIED:
        spatial_groups, group_stats = create_spatial_groups(
            lat=latitudes,
            lon=longitudes,
            method=spatial_split_method,
        )
    else:
        spatial_groups, group_stats = create_spatial_groups(
            lat=latitudes,
            lon=longitudes,
            data_counts=data_counts,
            n_groups=n_groups,
            method=spatial_split_method,
            balanced=BALANCED,
        )
    
    logging.info(f"Spatial groups assigned: {np.unique(spatial_groups)}")
   
    site_to_group = {site_id: group for site_id, group in zip(site_ids, spatial_groups)}
    site_to_pft = {site_id: site_info_dict[site_id]['pft'] for site_id in site_ids}
    
    # Plot spatial grouping
    fig = plt.figure(figsize=(20, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)

    colors = plt.cm.tab20.colors[:n_groups]
    color_map = ListedColormap(colors)

    scatter = ax.scatter(longitudes, latitudes, c=spatial_groups, s=100, 
                        edgecolor='k', cmap=color_map, transform=ccrs.PlateCarree())

    cbar = plt.colorbar(scatter, ax=ax, label='Spatial Group', shrink=0.8)

    ax.set_extent([min(longitudes)-5, max(longitudes)+5, 
                min(latitudes)-5, max(latitudes)+5], crs=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Spatial Grouping of Sites')

    plt.savefig(plot_dir / f'spatial_groups_{run_id}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Process all data into unified, record-level numpy arrays
    logging.info("Preparing final RECORD-LEVEL data structures for all sites...")
    
    list_X, list_y, list_groups, list_pfts = [], [], [], []
    list_timestamps = []
    list_site_ids_str = []

    for site_id, raw_data in site_data_dict.items():
        if IS_WINDOWING:
            windows = create_windows_from_segments(
                segments=raw_data, input_width=INPUT_WIDTH, label_width=LABEL_WIDTH, shift=SHIFT,
                label_columns=[TARGET_COL], exclude_targets_from_features=EXCLUDE_TARGETS,
                exclude_labels_from_inputs=EXCLUDE_LABELS
            )
            if not windows:
                continue
            X_site, y_site = convert_windows_to_numpy(windows)
            
            site_timestamps = []
            min_len = INPUT_WIDTH + SHIFT 
            
            for segment in raw_data:
                if len(segment) < min_len:
                    continue
                start_index = min_len - 1
                # Use the DataFrame index (which contains TIMESTAMP) for timestamps
                valid_dates = segment.index.values[start_index:]
                site_timestamps.append(valid_dates)
            
            if not site_timestamps:
                continue
            
            timestamps_site_final = np.concatenate(site_timestamps)
            
            if len(timestamps_site_final) != len(y_site):
                logging.warning(f"Timestamp mismatch for {site_id}! y={len(y_site)}, t={len(timestamps_site_final)}")
                min_l = min(len(y_site), len(timestamps_site_final))
                timestamps_site_final = timestamps_site_final[:min_l]
        else:
            if raw_data.empty:
                continue
            # Extract timestamps from the index before modifying the DataFrame
            timestamps_site_final = raw_data.index.values
            y_site = raw_data.pop(TARGET_COL).values
            X_site = raw_data.values
        
        num_records = len(y_site)
        if num_records == 0:
            continue
            
        list_X.append(X_site)
        list_y.append(y_site)
        list_groups.append(np.full(num_records, site_to_group[site_id]))
        list_pfts.append(np.full(num_records, site_to_pft[site_id]))
        list_site_ids_str.append(np.full(num_records, site_id))
        list_timestamps.append(timestamps_site_final)
    
    del site_data_dict

    if not list_X:
        logging.critical("ERROR: No data records were generated after processing. Exiting.")
        return [], []
        
    X_all_records = np.vstack(list_X)
    y_all_records = np.concatenate(list_y)
    groups_all_records = np.concatenate(list_groups)
    pfts_all_records = np.concatenate(list_pfts)
    site_ids_all_records = np.concatenate(list_site_ids_str)
    timestamps_all = np.concatenate(list_timestamps)
    
    logging.info(f"Total records: {len(y_all_records)}")
    logging.info(f"Total timestamps: {len(timestamps_all)}")
    
    assert len(y_all_records) == len(timestamps_all), "Error: Timestamps do not align with Data!"
    logging.info(f"Total records processed and ready for CV: {len(y_all_records)}")
    
    pfts_encoded, pft_categories = pd.factorize(pfts_all_records)
    logging.info(f"Encoded PFTs into {len(pft_categories)} integer classes. All PFTs: {list(pft_categories)}")
    
    # --- Stratified Group K-Fold Cross-Validation ---
    logging.info("\nInitializing K-Fold Cross-Validation at RECORD LEVEL...")
    
    if IS_STRATIFIED:
        outer_cv = StratifiedGroupKFold(n_splits=n_groups, shuffle=True, random_state=RANDOM_SEED)
        y_all_stratified = pfts_encoded
        logging.info(f"Using StratifiedGroupKFold with {n_groups} splits, stratifying by PFTs.")
    else:
        outer_cv = GroupKFold(n_splits=n_groups)
        y_all_stratified = y_all_records
        logging.info(f"Using GroupKFold with {n_groups} splits, grouping by spatial groups.")

    # --- Prepare indices for selective scaling ---
    pft_cols = all_possible_pft_types 
    
    base_numeric_indices = [i for i, col in enumerate(final_feature_names) 
                           if col not in pft_cols]
    
    if IS_WINDOWING:
        full_numeric_indices = []
        n_features = len(final_feature_names)
        for step in range(INPUT_WIDTH):
            offset = step * n_features
            step_indices = [idx + offset for idx in base_numeric_indices]
            full_numeric_indices.extend(step_indices)
        numeric_indices = np.array(full_numeric_indices)
    else:
        numeric_indices = np.array(base_numeric_indices)

    logging.info(f"Selective scaling: Scaling {len(numeric_indices)} numeric features, "
                 f"preserving {X_all_records.shape[1] - len(numeric_indices)} categorical features.")
    
    # Diagnostic: Target Distribution by Fold
    logging.info("\n=== DIAGNOSTIC: Target Distribution by Fold ===")
    temp_split = outer_cv.split(X_all_records, y_all_stratified, groups_all_records)
    for fold_idx, (train_idx, test_idx) in enumerate(temp_split):
        y_train_temp = y_all_records[train_idx]
        y_test_temp = y_all_records[test_idx]
        logging.info(f"Fold {fold_idx+1}: "
                     f"Train max={y_train_temp.max():.1f}, Test max={y_test_temp.max():.1f}, "
                     f"Train 95th={np.percentile(y_train_temp, 95):.1f}, Test 95th={np.percentile(y_test_temp, 95):.1f}")
    
    # Reset the generator for actual training
    split_generator = outer_cv.split(X_all_records, y_all_stratified, groups_all_records)
    
    all_test_r2_scores, all_test_rmse_scores = [], []
    all_predictions, all_actuals = [], []
    all_test_pfts = []
    all_test_site_ids = []  # Track site IDs for each test prediction
    dist_analyzer = FoldDistributionAnalyzer(output_dir=str(plot_dir / 'distributions'))

    for fold_idx, (train_val_indices, test_indices) in enumerate(split_generator):
        logging.info(f"\n=== Fold {fold_idx + 1}/{n_groups} ===")
        
        test_groups = np.unique(groups_all_records[test_indices])
        test_site_ids = [site_ids[np.where(spatial_groups == g)[0][0]] for g in test_groups]
        
        logging.info(f"Fold {fold_idx+1} TEST sites: {test_site_ids}")
        for sid in test_site_ids:
            site_mask = groups_all_records[test_indices] == site_to_group[sid]
            site_y = y_all_records[test_indices][site_mask]
            logging.info(f"  {sid}: n={len(site_y)}, median={np.median(site_y):.1f}, "
                        f"max={site_y.max():.1f}, PFT={site_to_pft[sid]}")
        
        X_train_val = X_all_records[train_val_indices]
        y_train_val = y_all_records[train_val_indices]
        pfts_train_val = pfts_all_records[train_val_indices]
        groups_train_val = groups_all_records[train_val_indices]
        
        X_test = X_all_records[test_indices]
        y_test = y_all_records[test_indices]
        
        # Apply target transformation using the TargetTransformer
        if IS_TRANSFORM:
            # Create a new transformer for this fold (fit on training data only)
            fold_transformer = TargetTransformer(method=TRANSFORM_METHOD)
            y_train_val = fold_transformer.fit_transform(y_train_val)
            logging.info(f"  Applied {TRANSFORM_METHOD} transformation to training target")

        # Scaling X (numeric columns only)
        # Get the names of the numeric features being scaled
        # (Assuming you created 'numeric_indices' earlier based on 'final_feature_names')
        # If IS_WINDOWING is True,generate the windowed feature names first
        
        if IS_WINDOWING:
            # Generate the full list of windowed feature names matching the columns in X_all_records
            all_windowed_names = generate_windowed_feature_names(final_feature_names, INPUT_WIDTH)
            numeric_feature_names = [all_windowed_names[i] for i in numeric_indices]
        else:
            numeric_feature_names = [final_feature_names[i] for i in numeric_indices]

        # Scaling X (numeric columns only)
        scaler = StandardScaler()
        
        # 1. Create a DataFrame for the fit so the scaler learns the names
        X_train_numeric_df = pd.DataFrame(
            X_train_val[:, numeric_indices], 
            columns=numeric_feature_names
        )
        
        # 2. Fit on the DataFrame
        scaler.fit(X_train_numeric_df)
        
        # 3. Transform (You can pass numpy array here, but passing DF is safer)
        # Note: We assign back to the numpy array, which is fine
        X_train_val[:, numeric_indices] = scaler.transform(X_train_numeric_df)
        
        # Transform test set
        X_test_numeric_df = pd.DataFrame(
            X_test[:, numeric_indices], 
            columns=numeric_feature_names
        )
        X_test[:, numeric_indices] = scaler.transform(X_test_numeric_df)

        try:
            sw_in_base_index = final_feature_names.index('sw_in')
        except ValueError:
            raise ValueError("'sw_in' not found in the final feature list!")
        
        num_features = len(final_feature_names)
        SW_IN_INDEX = ((INPUT_WIDTH - 1) * num_features) + sw_in_base_index
        logging.info(f"SW_IN_INDEX for nighttime masking: {SW_IN_INDEX}")

        dist_analyzer.capture_fold_data(
            fold_idx=fold_idx,
            X_train=X_train_val,
            y_train=y_train_val,
            X_test=X_test,
            y_test=y_test
        )
        
        if IS_SHUFFLE:
            shuffled_indices = np.random.permutation(len(X_train_val))
            X_train_val, y_train_val = X_train_val[shuffled_indices], y_train_val[shuffled_indices]
            pfts_train_val, groups_train_val = pfts_train_val[shuffled_indices], groups_train_val[shuffled_indices]

        logging.info(f"  Train/Val records: {len(y_train_val)}, Test records: {len(y_test)}")
        
        # --- Hyperparameter Optimization and Training ---
        optimizer = MLOptimizer(
            param_grid=hyperparameters, scoring='neg_mean_squared_error', model_type=MODEL_TYPE,
            task='regression', random_state=RANDOM_SEED, n_splits=5
        )
        
        optimizer.fit(
            X=X_train_val, 
            y=y_train_val, 
            is_cv=IS_CV,
            y_stratify=pfts_train_val,
            groups=groups_train_val,
            is_refit=True,
            split_type=SPLIT_TYPE,
            n_iterations=N_ITERATIONS
        )
        best_model = optimizer.get_best_model()

        # --- Evaluation ---
        test_predictions = best_model.predict(X_test)
        if IS_TRANSFORM:
            # Use the fold transformer to inverse transform predictions
            test_predictions = fold_transformer.inverse_transform(test_predictions)
        
        y_test = np.asarray(y_test)
        test_predictions = np.asarray(test_predictions)
        pfts_test = pfts_all_records[test_indices]
        site_ids_test = site_ids_all_records[test_indices]  # Get site IDs for test set
        
        finite_mask = np.isfinite(y_test) & np.isfinite(test_predictions)
        n_total = y_test.shape[0]
        n_valid = finite_mask.sum()

        if n_valid == 0:
            logging.error(f"Fold {fold_idx + 1}: No finite prediction/actual pairs. Skipping fold.")
            continue

        if n_valid < n_total:
            logging.warning(f"Fold {fold_idx + 1}: Dropping {n_total - n_valid} rows with NaN/inf.")

        y_test_clean = y_test[finite_mask]
        preds_clean = test_predictions[finite_mask]
        pfts_test_clean = pfts_test[finite_mask]  
        site_ids_test_clean = site_ids_test[finite_mask]  # Track site IDs
        all_test_pfts.extend(pfts_test_clean)
        all_test_site_ids.extend(site_ids_test_clean)  # Collect site IDs
        
        fold_test_r2 = r2_score(y_test_clean, preds_clean)
        fold_test_rmse = np.sqrt(mean_squared_error(y_test_clean, preds_clean))
        
        all_test_r2_scores.append(fold_test_r2)
        all_test_rmse_scores.append(fold_test_rmse)
        all_predictions.extend(test_predictions)
        all_actuals.extend(y_test)

        logging.info(f"  Fold {fold_idx + 1} Results -> Test R²: {fold_test_r2:.4f}, Test RMSE: {fold_test_rmse:.4f}")

        best_model.save_model(str(model_dir / f'spatial_{MODEL_TYPE}_fold_{fold_idx + 1}_{run_id}.json'))
        
        # --- Generate Individual Fold Plots ---
        logging.info(f"  Generating plots for Fold {fold_idx + 1}...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].scatter(y_test_clean, preds_clean, alpha=0.5, s=20, color='steelblue', 
                       edgecolors='black', linewidth=0.5)
        axes[0].set_xlabel('Observed Sap Velocity (cm³ cm⁻² h⁻¹)', fontsize=13)
        axes[0].set_ylabel('Predicted Sap Velocity (cm³ cm⁻² h⁻¹)', fontsize=13)
        axes[0].set_title(f'Fold {fold_idx + 1}: Predictions vs Actuals\n'
                         f'$R^2 = {fold_test_r2:.3f}$, RMSE = ${fold_test_rmse:.3f}$', 
                         fontsize=13, fontweight='bold')

        min_val = min(y_test_clean.min(), preds_clean.min())
        max_val = max(y_test_clean.max(), preds_clean.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r-', alpha=0.8, 
                    linewidth=2, label='Perfect Prediction')
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].axis('equal')
        axes[0].axis('square')

        residuals = y_test_clean - preds_clean
        axes[1].scatter(preds_clean, residuals, alpha=0.5, s=20, color='coral', 
                       edgecolors='black', linewidth=0.5)
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
        axes[1].set_xlabel('Predicted Sap Velocity', fontsize=13)
        axes[1].set_ylabel('Residuals (Observed - Predicted)', fontsize=13)
        axes[1].set_title(f'Fold {fold_idx + 1}: Residual Plot\n'
                         f'MAE = ${mean_absolute_error(y_test_clean, preds_clean):.3f}$', 
                         fontsize=13, fontweight='bold')
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fold_plot_path = plot_dir / f'fold_{fold_idx + 1}_performance_{run_id}_{spatial_split_method}.png'
        plt.savefig(fold_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"  Fold {fold_idx + 1} plot saved to: {fold_plot_path}")

        # Distribution comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(y_test_clean, bins=50, alpha=0.6, label='Observed', color='blue', edgecolor='black')
        axes[0].hist(preds_clean, bins=50, alpha=0.6, label='Predicted', color='orange', edgecolor='black')
        axes[0].set_xlabel('Sap Velocity', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'Fold {fold_idx + 1}: Distribution Comparison', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title(f'Fold {fold_idx + 1}: Q-Q Plot of Residuals', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        dist_plot_path = plot_dir / f'fold_{fold_idx + 1}_distributions_{run_id}_{spatial_split_method}.png'
        plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"  Fold {fold_idx + 1} distribution plot saved to: {dist_plot_path}")

        fold_results_df = pd.DataFrame({
            'Observed': y_test_clean,
            'Predicted': preds_clean,
            'Residual': residuals,
            'Absolute_Error': np.abs(residuals)
        })
        fold_csv_path = plot_dir / f'fold_{fold_idx + 1}_predictions_{run_id}_{spatial_split_method}.csv'
        fold_results_df.to_csv(fold_csv_path, index=False)
        logging.info(f"  Fold {fold_idx + 1} predictions saved to: {fold_csv_path}")

    dist_analyzer.generate_all_plots()

    # --- Final Results ---
    if all_test_r2_scores:
        mean_r2 = np.mean(all_test_r2_scores)
        std_r2 = np.std(all_test_r2_scores)
        mean_rmse = np.mean(all_test_rmse_scores)
        std_rmse = np.std(all_test_rmse_scores)
        
        logging.info("\n=== OVERALL SPATIAL CROSS-VALIDATION RESULTS ===")
        logging.info(f"Test R² Score: {mean_r2:.4f} ± {std_r2:.4f}")
        logging.info(f"Test RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")

        plt.figure(figsize=(8, 8))
        plt.scatter(all_actuals, all_predictions, alpha=0.5, s=20)
        plt.xlabel('Observed Sap Velocity (cm³ cm⁻² h⁻¹)', fontsize=14)
        plt.ylabel('Predicted Sap Velocity', fontsize=14)
        plt.title(f'Spatial CV Results\n$R^2 = {mean_r2:.3f} \\pm {std_r2:.3f}$, '
                 f'RMSE = ${mean_rmse:.3f} \\pm {std_rmse:.3f}$', fontsize=12)
        
        min_val = min(min(all_actuals), min(all_predictions))
        max_val = max(max(all_actuals), max(all_predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r-', alpha=0.8, linewidth=2)
        
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.axis('square')
        plt.tight_layout()
        # Ensure plot directory exists (OneDrive sync can sometimes cause issues)
        import os
        save_path = plot_dir / f'spatial_cv_predictions_vs_actual_{run_id}_{spatial_split_method}.png'
        parent_dir = str(save_path.parent)
        logging.info(f"Creating directory: {parent_dir}")
        os.makedirs(parent_dir, exist_ok=True)
        # Verify directory was created
        if not os.path.exists(parent_dir):
            logging.error(f"Failed to create directory: {parent_dir}")
            raise FileNotFoundError(f"Could not create directory: {parent_dir}")
        logging.info(f"Saving plot to: {save_path}")
        plt.savefig(str(save_path), dpi=300)
        plt.close()
        
        logging.info(f"\nResults plot saved to: {save_path}")
        
        # PFT-stratified scatter plots
        logging.info("Generating PFT-stratified performance plots...")
        
        arr_actuals = np.array(all_actuals)
        arr_predictions = np.array(all_predictions)
        arr_pfts = np.array(all_test_pfts)
        
        unique_pfts = np.unique(arr_pfts)
        n_pfts = len(unique_pfts)
        
        if n_pfts > 0:
            n_cols = 3
            n_rows = (n_pfts + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
            axes = axes.flatten()
            
            g_min = min(arr_actuals.min(), arr_predictions.min())
            g_max = max(arr_actuals.max(), arr_predictions.max())
            
            for i, pft in enumerate(unique_pfts):
                ax = axes[i]
                mask = arr_pfts == pft
                pft_actuals = arr_actuals[mask]
                pft_preds = arr_predictions[mask]
                
                if len(pft_actuals) < 10:
                    ax.text(0.5, 0.5, f"{pft}\nInsufficient Data (n={len(pft_actuals)})", 
                            ha='center', va='center')
                    continue
                    
                pft_r2 = r2_score(pft_actuals, pft_preds)
                pft_rmse = np.sqrt(mean_squared_error(pft_actuals, pft_preds))
                
                ax.scatter(pft_actuals, pft_preds, alpha=0.4, s=15, color='teal', 
                          edgecolor='k', linewidth=0.3)
                ax.plot([g_min, g_max], [g_min, g_max], 'r--', alpha=0.7)
                ax.set_title(f"PFT: {pft} (n={len(pft_actuals)})\nR²={pft_r2:.3f}, RMSE={pft_rmse:.3f}", 
                            fontweight='bold')
                ax.set_xlabel('Observed')
                ax.set_ylabel('Predicted')
                ax.grid(True, alpha=0.3)
                ax.set_xlim(g_min, g_max)
                ax.set_ylim(g_min, g_max)
                ax.set_aspect('equal')

            for j in range(i + 1, len(axes)):
                axes[j].axis('off')
                
            plt.tight_layout()
            pft_plot_path = plot_dir / f'pft_performance_{run_id}.png'
            plt.savefig(pft_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"PFT performance plot saved to: {pft_plot_path}")
            
            pft_metrics = []
            for pft in unique_pfts:
                mask = arr_pfts == pft
                if np.sum(mask) > 0:
                    y_p = arr_actuals[mask]
                    y_hat_p = arr_predictions[mask]
                    pft_metrics.append({
                        'PFT': pft,
                        'N_Samples': len(y_p),
                        'R2': r2_score(y_p, y_hat_p),
                        'RMSE': np.sqrt(mean_squared_error(y_p, y_hat_p)),
                        'MAE': mean_absolute_error(y_p, y_hat_p)
                    })
            pd.DataFrame(pft_metrics).to_csv(plot_dir / f'pft_metrics_{run_id}.csv', index=False)

        # =========================================================================
        # SITE-LEVEL PERFORMANCE ANALYSIS
        # =========================================================================
        logging.info("\n" + "="*60)
        logging.info("=== SITE-LEVEL PERFORMANCE ANALYSIS ===")
        logging.info("="*60)
        
        arr_site_ids = np.array(all_test_site_ids)
        unique_sites = np.unique(arr_site_ids)
        
        site_metrics = []
        for site in unique_sites:
            mask = arr_site_ids == site
            if np.sum(mask) > 1:  # Need at least 2 samples for R2
                y_site = arr_actuals[mask]
                y_hat_site = arr_predictions[mask]
                
                # Calculate metrics
                site_r2 = r2_score(y_site, y_hat_site)
                site_rmse = np.sqrt(mean_squared_error(y_site, y_hat_site))
                site_mae = mean_absolute_error(y_site, y_hat_site)
                
                # Calculate bias (mean residual)
                residuals = y_site - y_hat_site
                site_bias = np.mean(residuals)
                
                # Get site PFT
                site_pft = arr_pfts[mask][0]
                
                site_metrics.append({
                    'site_name': site,
                    'PFT': site_pft,
                    'N_Samples': len(y_site),
                    'R2': site_r2,
                    'RMSE': site_rmse,
                    'MAE': site_mae,
                    'Bias': site_bias,
                    'Mean_Observed': np.mean(y_site),
                    'Mean_Predicted': np.mean(y_hat_site),
                    'Std_Observed': np.std(y_site),
                    'Std_Predicted': np.std(y_hat_site)
                })
        
        site_metrics_df = pd.DataFrame(site_metrics)
        site_metrics_df = site_metrics_df.sort_values('R2', ascending=True)
        site_metrics_df.to_csv(plot_dir / f'site_metrics_{run_id}.csv', index=False)
        logging.info(f"Site-level metrics saved to: {plot_dir / f'site_metrics_{run_id}.csv'}")
        
        # Log worst performing sites
        logging.info("\n=== WORST PERFORMING SITES (Bottom 10 by R²) ===")
        worst_sites = site_metrics_df.head(10)
        for _, row in worst_sites.iterrows():
            logging.info(f"  {row['site_name']}: R²={row['R2']:.3f}, RMSE={row['RMSE']:.3f}, "
                        f"Bias={row['Bias']:.3f}, N={row['N_Samples']}, PFT={row['PFT']}")
        
        # Log best performing sites  
        logging.info("\n=== BEST PERFORMING SITES (Top 10 by R²) ===")
        best_sites = site_metrics_df.tail(10).iloc[::-1]  # Reverse to show best first
        for _, row in best_sites.iterrows():
            logging.info(f"  {row['site_name']}: R²={row['R2']:.3f}, RMSE={row['RMSE']:.3f}, "
                        f"Bias={row['Bias']:.3f}, N={row['N_Samples']}, PFT={row['PFT']}")
        
        # Sites with negative R² (predictions worse than mean)
        negative_r2_sites = site_metrics_df[site_metrics_df['R2'] < 0]
        if len(negative_r2_sites) > 0:
            logging.info(f"\n=== SITES WITH NEGATIVE R² ({len(negative_r2_sites)} sites) ===")
            for _, row in negative_r2_sites.iterrows():
                logging.info(f"  {row['site_name']}: R²={row['R2']:.3f}, RMSE={row['RMSE']:.3f}, "
                            f"Bias={row['Bias']:.3f}, N={row['N_Samples']}, PFT={row['PFT']}")
        
        # Create site performance visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. R² distribution
        axes[0, 0].hist(site_metrics_df['R2'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='R²=0')
        axes[0, 0].axvline(x=site_metrics_df['R2'].median(), color='green', linestyle='-', 
                          linewidth=2, label=f'Median={site_metrics_df["R2"].median():.3f}')
        axes[0, 0].set_xlabel('R² Score', fontsize=12)
        axes[0, 0].set_ylabel('Number of Sites', fontsize=12)
        axes[0, 0].set_title('Distribution of Site-Level R² Scores', fontsize=13, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. R² vs Sample Size
        scatter = axes[0, 1].scatter(site_metrics_df['N_Samples'], site_metrics_df['R2'], 
                                     c=site_metrics_df['RMSE'], cmap='viridis', alpha=0.7, s=50)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
        axes[0, 1].set_xlabel('Number of Samples', fontsize=12)
        axes[0, 1].set_ylabel('R² Score', fontsize=12)
        axes[0, 1].set_title('R² vs Sample Size (colored by RMSE)', fontsize=13, fontweight='bold')
        plt.colorbar(scatter, ax=axes[0, 1], label='RMSE')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Bias distribution
        axes[1, 0].hist(site_metrics_df['Bias'], bins=30, edgecolor='black', alpha=0.7, color='coral')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Bias (Observed - Predicted)', fontsize=12)
        axes[1, 0].set_ylabel('Number of Sites', fontsize=12)
        axes[1, 0].set_title('Distribution of Site-Level Bias', fontsize=13, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. R² by PFT (boxplot)
        pft_order = site_metrics_df.groupby('PFT')['R2'].median().sort_values().index
        site_metrics_df['PFT_ordered'] = pd.Categorical(site_metrics_df['PFT'], categories=pft_order, ordered=True)
        site_metrics_df_sorted = site_metrics_df.sort_values('PFT_ordered')
        
        unique_pfts_for_box = site_metrics_df_sorted['PFT'].unique()
        box_data = [site_metrics_df_sorted[site_metrics_df_sorted['PFT'] == pft]['R2'].values 
                   for pft in unique_pfts_for_box]
        
        bp = axes[1, 1].boxplot(box_data, labels=unique_pfts_for_box, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=1.5)
        axes[1, 1].set_xlabel('Plant Functional Type', fontsize=12)
        axes[1, 1].set_ylabel('R² Score', fontsize=12)
        axes[1, 1].set_title('R² Distribution by PFT', fontsize=13, fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        site_plot_path = plot_dir / f'site_performance_{run_id}.png'
        plt.savefig(str(site_plot_path), dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Site performance plot saved to: {site_plot_path}")

    # =========================================================================
    # TRAINING FINAL MODEL ON ALL DATA
    # =========================================================================
    logging.info("\n" + "="*60)
    logging.info("=== TRAINING FINAL MODEL ON ALL DATA ===")
    logging.info("="*60)
    
    logging.info("Applying preprocessing to all data (Numeric Only)...")
    
    # Generate feature names for the columns corresponding to numeric_indices
    if IS_WINDOWING:
        all_windowed_names = generate_windowed_feature_names(final_feature_names, INPUT_WIDTH)
        numeric_feature_names = [all_windowed_names[i] for i in numeric_indices]
    else:
        numeric_feature_names = [final_feature_names[i] for i in numeric_indices]

    scaler_final = StandardScaler()
    X_all_scaled = X_all_records.copy()
    
    # Create DataFrame for fitting
    X_all_numeric_df = pd.DataFrame(
        X_all_scaled[:, numeric_indices], 
        columns=numeric_feature_names
    )
    
    # Fit on DataFrame -> Scaler stores feature names
    scaler_final.fit(X_all_numeric_df)
    
    # Transform
    X_all_scaled[:, numeric_indices] = scaler_final.transform(X_all_numeric_df)
    
    # Apply target transformation using the global transformer
    if IS_TRANSFORM:
        y_all_transformed = target_transformer.fit_transform(y_all_records)
        logging.info(f"Applied {TRANSFORM_METHOD} transformation to all target data")
    else:
        y_all_transformed = y_all_records
    
    logging.info(f"Data shapes: X={X_all_scaled.shape}, y={y_all_transformed.shape}")
    
    logging.info("Running hyperparameter optimization on all data...")
    
    final_optimizer = MLOptimizer(
        param_grid=hyperparameters,
        scoring='neg_mean_squared_error',
        model_type=MODEL_TYPE,
        task='regression',
        random_state=RANDOM_SEED,
        n_splits=5
    )
    
    final_optimizer.fit(
        X=X_all_scaled,
        y=y_all_transformed,
        is_cv=True,
        y_stratify=pfts_encoded,
        groups=groups_all_records,
        is_refit=True,
        split_type=SPLIT_TYPE,
        n_iterations=N_ITERATIONS
    )
    
    final_model = final_optimizer.get_best_model()
    best_params = final_optimizer.best_params_
    best_cv_score = final_optimizer.best_score_
    
    logging.info(f"Best hyperparameters: {best_params}")
    logging.info(f"Best CV score (neg_mse): {best_cv_score:.4f}")
    
    final_model_path = model_dir / f'FINAL_{MODEL_TYPE}_{run_id}.joblib'
    joblib.dump(final_model, final_model_path)
    logging.info(f"Final model saved to: {final_model_path}")
    
    final_scaler_path = model_dir / f'FINAL_scaler_{run_id}_feature.pkl'
    joblib.dump(scaler_final, final_scaler_path)
    logging.info(f"Scaler saved to: {final_scaler_path}")
    
    # Save target transformer if transformation was applied
    if IS_TRANSFORM:
        transformer_path = model_dir / f'FINAL_target_transformer_{run_id}.pkl'
        joblib.dump(target_transformer, transformer_path)
        logging.info(f"Target transformer saved to: {transformer_path}")
    
    # =========================================================================
    # GENERATE FEATURE NAMES FOR SHAP (HANDLES WINDOWING)
    # =========================================================================
    if IS_WINDOWING:
        shap_feature_names = generate_windowed_feature_names(final_feature_names, INPUT_WIDTH)
        logging.info(f"Generated {len(shap_feature_names)} windowed feature names for SHAP")
    else:
        shap_feature_names = final_feature_names
        logging.info(f"Using {len(shap_feature_names)} feature names for SHAP")
    
        # Save config/metadata
    final_config = {
        'model_type': MODEL_TYPE,
        'run_id': run_id,
        'best_params': best_params,
        'best_cv_score': float(best_cv_score),
         'cv_results': {
           'mean_r2': float(mean_r2),
   
          'std_r2': float(std_r2),
           'mean_rmse': float(mean_rmse),
           'std_rmse': float(std_rmse),
           'n_folds': n_groups,
       },
        'preprocessing': {
            'target_transform': TRANSFORM_METHOD if IS_TRANSFORM else None,
            'target_transform_params': target_transformer.get_params() if IS_TRANSFORM else None,
            'feature_scaling': 'StandardScaler',
        },
        'data_info': {
            'n_samples': len(y_all_records),
            'n_features': X_all_records.shape[1],
            'IS_WINDOWING': IS_WINDOWING,
            'input_width': INPUT_WIDTH if IS_WINDOWING else None,
            'label_width': LABEL_WIDTH if IS_WINDOWING else None,
            'shift': SHIFT if IS_WINDOWING else None,
        },
        'feature_names': final_feature_names,
        'shap_feature_names': shap_feature_names,
        'random_seed': RANDOM_SEED,
        'split_type': SPLIT_TYPE,
    }
    
    config_path = model_dir / f'FINAL_config_{run_id}.json'
    with open(config_path, 'w') as f:
        json.dump(final_config, f, indent=2, default=str)
    logging.info(f"Config saved to: {config_path}")
    
    # Sanity check
    logging.info("Running sanity check on training data...")
    y_pred_all_transformed = final_model.predict(X_all_scaled)
    if IS_TRANSFORM:
        y_pred_all = target_transformer.inverse_transform(y_pred_all_transformed)
    else:
        y_pred_all = y_pred_all_transformed
    
    sanity_r2 = r2_score(y_all_records, y_pred_all)
    sanity_rmse = np.sqrt(mean_squared_error(y_all_records, y_pred_all))
    sanity_mae = mean_absolute_error(y_all_records, y_pred_all)
    
    logging.info(f"Sanity check (on training data):")
    logging.info(f"  R²   = {sanity_r2:.4f}")
    logging.info(f"  RMSE = {sanity_rmse:.4f}")
    logging.info(f"  MAE  = {sanity_mae:.4f}")
    logging.info("(Note: These metrics are on training data - expect better than CV)")
    
    # Plot sanity check
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    axes[0].scatter(y_all_records, y_pred_all, alpha=0.3, s=10, color='steelblue')
    axes[0].set_xlabel('Observed Sap Velocity (cm³ cm⁻² h⁻¹)', fontsize=13)
    axes[0].set_ylabel('Predicted Sap Velocity (cm³ cm⁻² h⁻¹)', fontsize=13)
    axes[0].set_title(f'Final Model: Training Data Fit\n'
                     f'$R^2 = {sanity_r2:.3f}$, RMSE = ${sanity_rmse:.3f}$\n'
                     f'(CV estimate: $R^2 = {mean_r2:.3f} \\pm {std_r2:.3f}$)', 
                     fontsize=12, fontweight='bold')
    
    min_val = min(y_all_records.min(), y_pred_all.min())
    max_val = max(y_all_records.max(), y_pred_all.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label='1:1 Line')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal', 'box')
    
    residuals_all = y_all_records - y_pred_all
    axes[1].scatter(y_pred_all, residuals_all, alpha=0.3, s=10, color='coral')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted Sap Velocity', fontsize=13)
    axes[1].set_ylabel('Residuals (Observed - Predicted)', fontsize=13)
    axes[1].set_title(f'Final Model: Residual Plot\nMAE = ${sanity_mae:.3f}$', 
                     fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / f'FINAL_model_sanity_check_{run_id}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("\n" + "="*60)
    logging.info("=== FINAL MODEL TRAINING COMPLETE ===")
    logging.info("="*60)
    logging.info(f"Model:   {final_model_path}")
    logging.info(f"Scaler:  {final_scaler_path}")
    logging.info(f"Config:  {config_path}")
    logging.info(f"")
    logging.info(f"Expected generalization performance (from CV):")
    logging.info(f"  R²   = {mean_r2:.4f} ± {std_r2:.4f}")
    logging.info(f"  RMSE = {mean_rmse:.4f} ± {std_rmse:.4f}")
    
    # =========================================================================
    # RUN SHAP ANALYSIS (WITH STATIC FEATURE AGGREGATION)
    # =========================================================================
    logging.info("\n" + "="*60)
    logging.info("=== RUNNING SHAP ANALYSIS ===")
    logging.info("="*60)
    
    # Define static features for your dataset
    # These are features that don't change within a time window
    STATIC_FEATURES = [
        'canopy_height', 'elevation', 'prcip/PET',
        # PFT one-hot encoded columns
        'MF', 'DNF', 'ENF', 'EBF', 'WSA', 'WET', 'DBF', 'SAV',
    ]
    
    # Prepare data for SHAP
    X_for_shap_calculation = X_all_scaled 
    X_for_plotting_axes = X_all_records
    
    # Re-extract coordinates and site IDs for the full dataset
    lat_all = np.array([site_info_dict[s]['latitude'] for s in site_ids_all_records])
    lon_all = np.array([site_info_dict[s]['longitude'] for s in site_ids_all_records])
    
    try:
        # =================================================================
        # STEP 1: Calculate raw SHAP values
        # =================================================================
        logging.info("Step 1: Calculating raw SHAP values...")
        
        # Sample for SHAP (computation is expensive)
        n_total = len(X_for_shap_calculation)
        if n_total > SHAP_SAMPLE_SIZE:
            logging.info(f"Sampling {SHAP_SAMPLE_SIZE} from {n_total} total samples...")
            np.random.seed(42)
            sampled_indices = np.random.choice(n_total, SHAP_SAMPLE_SIZE, replace=False)
            sampled_indices = np.sort(sampled_indices)
        else:
            logging.info(f"Using all {n_total} samples (below sample_size threshold)")
            sampled_indices = np.arange(n_total)
        
        X_shap = X_for_shap_calculation[sampled_indices]
        X_original_sampled = X_for_plotting_axes[sampled_indices]
        lat_sampled = lat_all[sampled_indices]
        lon_sampled = lon_all[sampled_indices]
        site_ids_sampled = site_ids_all_records[sampled_indices]
        timestamps_sampled = timestamps_all[sampled_indices]
        y_sampled = y_all_records[sampled_indices]  # Observed values (original scale)
        
        logging.info(f"SHAP calculation on {len(X_shap)} samples")
        
        # Create SHAP explainer and calculate values
        logging.info("Creating SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(final_model)
        
        logging.info("Calculating SHAP values (this may take a while)...")
        shap_values_raw = explainer.shap_values(X_shap)
        
        # Handle multi-output models
        if isinstance(shap_values_raw, list):
            logging.info("Model returned list of SHAP values, using first element")
            shap_values_raw = shap_values_raw[0]
        
        logging.info(f"Raw SHAP values shape: {shap_values_raw.shape}")
        
        # Get base value
        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = base_value[0] if len(base_value) > 0 else base_value
        logging.info(f"Base value (expected_value): {base_value}")
        
        # =================================================================
        # STEP 2: Aggregate static features (if windowing)
        # =================================================================
        if IS_WINDOWING:
            logging.info("\nStep 2: Aggregating SHAP values for static features...")
            
            # Aggregate SHAP values
            shap_values_agg, feature_names_agg = aggregate_static_feature_shap(
                shap_values=shap_values_raw,
                windowed_feature_names=shap_feature_names,
                base_feature_names=final_feature_names,
                input_width=INPUT_WIDTH,
                static_features=STATIC_FEATURES,
                aggregation='sum'
            )
            
            # Aggregate X values for plotting
            X_original_agg, _ = aggregate_static_feature_values(
                X_original=X_original_sampled,
                base_feature_names=final_feature_names,
                input_width=INPUT_WIDTH,
                static_features=STATIC_FEATURES
            )
            
            # Use aggregated values for main plots
            shap_values = shap_values_agg
            shap_feature_names_final = feature_names_agg
            X_for_plots = X_original_agg
            
            # Also keep raw values for time-step specific analysis
            shap_values_windowed = shap_values_raw
            shap_feature_names_windowed = shap_feature_names
            X_for_plots_windowed = X_original_sampled
            
            logging.info(f"Aggregated SHAP values shape: {shap_values.shape}")
            logging.info(f"Aggregated feature count: {len(shap_feature_names_final)}")
        else:
            shap_values = shap_values_raw
            shap_feature_names_final = shap_feature_names
            X_for_plots = X_original_sampled
            
            shap_values_windowed = None
            shap_feature_names_windowed = None
            X_for_plots_windowed = None
        
        # Create DataFrames for easier handling
        df_shap = pd.DataFrame(shap_values, columns=shap_feature_names_final)
        df_X = pd.DataFrame(X_for_plots, columns=shap_feature_names_final)
        
        # =================================================================
        # STEP 3: Generate Global Importance Plots
        # =================================================================
        logging.info("\nStep 3 & 4: Generating Grouped PFT Plots...")

        # 1. DEFINE PFT COLUMNS
        pft_cols_to_group = ['MF', 'DNF', 'ENF', 'EBF', 'WSA', 'WET', 'DBF', 'SAV']
        
        # 2. PERFORM GROUPING
        # We perform this ONCE so all plots use the exact same data structure
        shap_values_grouped, df_X_grouped, feature_names_grouped = group_pft_for_summary_plots(
            shap_values=shap_values, 
            X_df=df_X, 
            feature_names=shap_feature_names_final,
            pft_cols=pft_cols_to_group
        )

        logging.info(f"Grouped Data Shape: {shap_values_grouped.shape}")
        
        # ---------------------------------------------------------
        # PLOT 3a: Beeswarm Summary (Grouped)
        # ---------------------------------------------------------
        logging.info("  Generating Beeswarm plot (PFT grouped)...")
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values_grouped, df_X_grouped, show=False, max_display=20)
        plt.xlabel(get_shap_label(), fontsize=12)
        plt.title(f"Feature Contribution (Land Cover Grouped as 'PFT')\n", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plot_dir / "shap_summary_beeswarm_grouped.png", dpi=300, bbox_inches='tight')
        plt.close()

        # ---------------------------------------------------------
        # PLOT 3b: Bar Plot (Grouped)
        # ---------------------------------------------------------
        logging.info("  Generating Bar plot (PFT grouped)...")
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values_grouped, df_X_grouped, plot_type="bar", show=False, max_display=20)
        plt.xlabel(f"Mean |SHAP Value| ({SHAP_UNITS})", fontsize=12)
        plt.title(f"Feature Importance (Land Cover Grouped as 'PFT')\n", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plot_dir / "shap_global_importance_bar_grouped.png", dpi=300, bbox_inches='tight')
        plt.close()

        # ---------------------------------------------------------
        # PLOT 4: Smart Partial Dependence Plots (Grouped)
        # ---------------------------------------------------------
        logging.info("  Generating Hybrid Partial Dependence Plots (Scatter for Numeric, Box for PFT)...")
        
        # Calculate importance using the GROUPED values
        mean_abs_shap = np.abs(shap_values_grouped).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1][:9] # Top 9
        top_features = [feature_names_grouped[i] for i in top_indices]
        
        logging.info(f"  Top 9 features (after grouping): {top_features}")
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()
        
        for i, feature in enumerate(top_features):
            ax = axes[i]
            
            # Get data for this feature
            x_val = df_X_grouped[feature].values
            y_val = shap_values_grouped[:, feature_names_grouped.index(feature)]
            
            # --- SPECIAL HANDLING FOR PFT (Categorical) ---
            if feature == 'PFT':
                # Convert numeric codes back to string labels for the plot
                # We do this by looking at which PFT column was active in the original df_X
                
                # 1. Reconstruct labels from original data for mapping
                temp_X_pft = df_X[pft_cols_to_group]
                # Series of strings ('ENF', 'MF', etc.)
                pft_labels_series = temp_X_pft.idxmax(axis=1) 
                
                # Prepare DataFrame for Seaborn
                plot_df = pd.DataFrame({
                    'PFT': pft_labels_series.values,
                    'SHAP': y_val
                })
                
                # Sort order by median SHAP value for cleaner look
                order = plot_df.groupby('PFT')['SHAP'].median().sort_values().index
                
                # Draw Boxplot
                sns.boxplot(data=plot_df, x='PFT', y='SHAP', ax=ax, palette='Set2', order=order)
                ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
                ax.set_title(f"Effect of {feature}", fontsize=12, fontweight='bold')
                ax.set_ylabel(get_shap_label(), fontsize=10)
                ax.set_xlabel("") # Labels are on ticks
                ax.tick_params(axis='x', rotation=45)
                #ax.grid(True, alpha=0.3, axis='y')
                
            # --- HANDLING FOR NUMERIC FEATURES ---
            else:
                # Filter valid values
                valid_mask = np.isfinite(x_val) & np.isfinite(y_val)
                x_valid = x_val[valid_mask]
                y_valid = y_val[valid_mask]
                
                if len(x_valid) == 0:
                    ax.text(0.5, 0.5, "No valid data", ha='center', va='center')
                    continue
                
                # Scatter plot
                ax.scatter(x_valid, y_valid, alpha=0.3, color='steelblue', s=10)
                
                # Add smooth curve (LOWESS)
                try:
                    sns.regplot(x=x_valid, y=y_valid, scatter=False, lowess=True, 
                                ax=ax, color='red', line_kws={'linewidth': 2})
                except Exception:
                    pass # Skip lowess if it fails (e.g. constant data)

                ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
                
                # Add units
                feat_unit = get_feature_unit(feature)
                xlabel = f"{feature} ({feat_unit})" if feat_unit else feature
                
                ax.set_title(f"Effect of {feature}", fontsize=12, fontweight='bold')
                ax.set_xlabel(xlabel, fontsize=10)
                ax.set_ylabel(get_shap_label(), fontsize=10)
                #ax.grid(True, alpha=0.3)
        
        plt.suptitle(f"Partial Dependence Plots (Top 9 Features)\n(PFT Grouped)", 
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(plot_dir / "shap_partial_dependence_grouped.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # =================================================================
        # STEP 5: Spatial SHAP Maps
        # =================================================================
        logging.info("\nStep 5: Generating Spatial SHAP Maps...")
        
        # Use top features calculated in Step 4 (which might include 'PFT')
        top_4_features = top_features[:4]
        
        fig = plt.figure(figsize=(20, 12))
        
        for i, feature in enumerate(top_4_features):
            ax = fig.add_subplot(2, 2, i+1, projection=ccrs.PlateCarree())
            
            # Add map features
            ax.add_feature(cfeature.BORDERS, linestyle=':')
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
            
            # --- FIX START: Handle 'PFT' vs Standard Features ---
            if feature == 'PFT':
                # If feature is PFT, get values from the grouped array
                # (feature_names_grouped and shap_values_grouped come from Step 3)
                pft_index = feature_names_grouped.index('PFT')
                vals = shap_values_grouped[:, pft_index]
            else:
                # Otherwise get from the original dataframe
                vals = df_shap[feature].values
            # --- FIX END ---
            
            # Filter valid values
            valid_mask = np.isfinite(vals) & np.isfinite(lon_sampled) & np.isfinite(lat_sampled)
            
            if valid_mask.sum() == 0:
                ax.text(0.5, 0.5, "No valid data", ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                continue
            
            vals_valid = vals[valid_mask]
            lon_valid = lon_sampled[valid_mask]
            lat_valid = lat_sampled[valid_mask]
            
            # Diverging colormap centered at 0
            vmax = max(abs(vals_valid.min()), abs(vals_valid.max()))
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            
            scatter = ax.scatter(lon_valid, lat_valid, c=vals_valid, s=30, 
                                 cmap='RdBu_r', norm=norm, transform=ccrs.PlateCarree(),
                                 edgecolor='k', linewidth=0.3, alpha=0.7)
            
            # Colorbar with units
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
            cbar.set_label(f'SHAP Value ({SHAP_UNITS})', fontsize=10)
            
            # Title with feature unit
            if feature == 'PFT':
                title = f"Spatial Contribution: Land Cover (PFT)"
            else:
                feat_unit = get_feature_unit(feature)
                if feat_unit:
                    title = f"Spatial Contribution: {feature}\n({feat_unit})"
                else:
                    title = f"Spatial Contribution: {feature}"
            
            ax.set_title(title, fontsize=13, fontweight='bold')
            
            # Set extent based on data
            ax.set_extent([lon_valid.min()-5, lon_valid.max()+5, 
                          lat_valid.min()-5, lat_valid.max()+5], 
                          crs=ccrs.PlateCarree())
        
        plt.suptitle(f"Spatial Distribution of Feature Contributions\n"
                     f"(SHAP Values in {SHAP_UNITS})", 
                     fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(plot_dir / "shap_spatial_maps.png", dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"    Saved: shap_spatial_maps.png")
        
        # =================================================================
        # STEP 6: Waterfall Plots (High/Low Predictions)
        # =================================================================
        logging.info("\nStep 6: Generating Local Waterfall Plots...")
        
        pred_vals = final_model.predict(X_shap)
        high_idx = np.argmax(pred_vals)
        low_idx = np.argmin(pred_vals)
        
        for idx, name in [(high_idx, "High_Flow"), (low_idx, "Low_Flow")]:
            plt.figure(figsize=(12, 9))
            
            # Create Explanation object
            row_explainer = shap.Explanation(
                values=shap_values[idx],
                base_values=float(base_value),
                data=X_for_plots[idx],
                feature_names=shap_feature_names_final
            )
            
            shap.plots.waterfall(row_explainer, show=False, max_display=12)
            
            # Get predicted value for title
            if IS_TRANSFORM:
                pred_original = target_transformer.inverse_transform(np.array([pred_vals[idx]]))[0]
            else:
                pred_original = pred_vals[idx]
            
            plt.title(f"Why did the model predict {name}?\n"
                      f"Site: {site_ids_sampled[idx]} | "
                      f"Predicted: {pred_original:.2f} {SHAP_UNITS}", 
                      fontsize=12, fontweight='bold')
            plt.xlabel(f"SHAP Value ({SHAP_UNITS})", fontsize=11)
            plt.tight_layout()
            plt.savefig(plot_dir / f"shap_waterfall_{name}.png", dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"    Saved: shap_waterfall_{name}.png")
        
        # =================================================================
        # STEP 7: Seasonal Drivers by Hemisphere (Figure 7)
        # =================================================================
        logging.info("\nStep 7: Generating Seasonal Driver Analysis (Figure 7)...")
        
        try:
            plot_seasonal_drivers_by_hemisphere(
                shap_values=shap_values,
                feature_names=shap_feature_names_final,
                timestamps=timestamps_sampled,
                latitudes=lat_sampled,
                top_n=5,
                output_dir=plot_dir
            )
            logging.info("    Generated Figure 7 (Seasonal Drivers by Hemisphere)")
        except Exception as e:
            logging.warning(f"    Could not generate Fig 7 (Seasonal): {e}")
            import traceback
            traceback.print_exc()
        
        # =================================================================
        # STEP 8: Diurnal Drivers (Hourly Data Only)
        # =================================================================
        if TIME_SCALE == 'hourly':
            logging.info("\nStep 8: Generating Diurnal Driver Analysis (Grouped PFT, All Variables)...")
            
            try:
                # 1. Aggregate PFTs into a single feature for the diurnal data
                # This removes 'MF', 'ENF', etc. and creates a single 'PFT' column
                shap_values_diurnal, feature_names_diurnal = aggregate_pft_shap_values(
                    shap_values=shap_values,
                    feature_names=shap_feature_names_final,
                    pft_columns=PFT_COLUMNS,
                    aggregation='sum'
                )
                
                # 2. Set top_n to the total number of features
                # This ensures the "Rest" category is never created and ALL variables are shown
                n_all_features = len(feature_names_diurnal)
                logging.info(f"    Plotting all {n_all_features} features (no 'Rest' category)")

                # Plot 8a: Stacked bar chart
                # Note: observed_values should be on original scale (not transformed)
                # base_value is the expected/mean prediction from SHAP explainer
                # observed_mean is the mean of observed values for centering
                plot_diurnal_drivers(
                    shap_values=shap_values_diurnal,
                    feature_names=feature_names_diurnal,
                    timestamps=timestamps_sampled,
                    observed_values=y_sampled,  # Original scale observed values
                    base_value=float(base_value),  # SHAP base value (model average prediction)
                    observed_mean=float(np.mean(y_sampled)),  # Mean of observed for centering
                    top_n=n_all_features, # Show ALL features
                    output_dir=plot_dir
                )
                logging.info("    Generated Diurnal Drivers (Stacked Bar with Observed)")
            
                # Plot 8b: Heatmap
                plot_diurnal_drivers_heatmap(
                    shap_values=shap_values_diurnal,
                    feature_names=feature_names_diurnal,
                    timestamps=timestamps_sampled,
                    top_n=n_all_features, # Show ALL features
                    output_dir=plot_dir
                )
                logging.info("    Generated Diurnal Drivers (Heatmap)")
            
                # Plot 8c: Line plots with CI
                # Note: We limit this to top 12 to prevent generating 20+ tiny subplots, 
                # but it will include PFT as a single line if it's important.
                # If you truly want ALL lines, change 12 to n_all_features.
                plot_diurnal_feature_lines(
                    shap_values=shap_values_diurnal,
                    feature_names=feature_names_diurnal,
                    timestamps=timestamps_sampled,
                    top_n=12, 
                    output_dir=plot_dir
                )
                logging.info("    Generated Diurnal Drivers (Line Plots)")

            except Exception as e:
                logging.warning(f"    Could not generate diurnal plots: {e}")
                import traceback
                traceback.print_exc()
        else:
            logging.info(f"\nStep 8: Skipping diurnal plots (TIME_SCALE={TIME_SCALE}, requires 'hourly')")
        
        # =================================================================
        # STEP 9: Interaction Dependencies (Figure 8)
        # =================================================================
        logging.info("\nStep 9: Generating Interaction Dependence Plots (Figure 8)...")
        
        try:
            # Define meaningful interaction pairs
            # Use feature names that exist in the aggregated feature list
            potential_pairs = [
                ('sw_in_t-0', 'vpd_t-0'),
                ('vpd_t-0', 'ta_t-0'),
                ('ta_t-0', 'volumetric_soil_water_layer_1_t-0'),
                ('sw_in_t-0', 'LAI'),  # Static feature
                ('vpd_t-0', 'canopy_height'),  # Static feature
            ]
            
            # For non-windowed data
            if not IS_WINDOWING:
                potential_pairs = [
                    ('vpd', 'volumetric_soil_water_layer_1'),  # Supply vs Demand
                    ('sw_in', 'vpd'),                           # Energy vs Demand
                    ('canopy_height', 'LAI'),                   # Structural interaction
                    ('vpd', 'PFT')                              # Strategy difference (Categorical interaction)
                ]
            
            # Filter to only include pairs where both features exist
            valid_pairs = []
            for f1, f2 in potential_pairs:
                if f1 in shap_feature_names_final and f2 in shap_feature_names_final:
                    valid_pairs.append((f1, f2))
                else:
                    missing = []
                    if f1 not in shap_feature_names_final:
                        missing.append(f1)
                    if f2 not in shap_feature_names_final:
                        missing.append(f2)
                    logging.debug(f"    Skipping pair ({f1}, {f2}) - missing: {missing}")
            
            if valid_pairs:
                logging.info(f"    Using interaction pairs: {valid_pairs}")
                plot_interaction_dependencies(
                    shap_values=shap_values,
                    X_original=X_for_plots,
                    feature_names=shap_feature_names_final,
                    interaction_pairs=valid_pairs[:3],  # Use top 3 valid pairs
                    output_dir=plot_dir
                )
                logging.info("    Generated Figure 8 (Interaction Dependencies)")
            else:
                logging.warning("    No valid interaction pairs found - skipping Figure 8")
                
        except Exception as e:
            logging.warning(f"    Could not generate Fig 8 (Interaction): {e}")
            import traceback
            traceback.print_exc()
        
        # =================================================================
        # STEP 10: Time-Step Comparison (for windowed data)
        # =================================================================
        if IS_WINDOWING and shap_values_windowed is not None:
            logging.info("\nStep 10: Generating Time-Step Importance Comparison...")
            
            try:
                # Calculate mean absolute SHAP by time step for each base feature
                n_base_features = len(final_feature_names)
                
                # Get dynamic features only
                dynamic_features = [f for f in final_feature_names if f not in STATIC_FEATURES]
                
                time_step_importance = {}
                for t in range(INPUT_WIDTH):
                    time_offset = INPUT_WIDTH - 1 - t
                    time_label = f"t-{time_offset}" if time_offset > 0 else "t-0"
                    time_step_importance[time_label] = {}
                    
                    for feat_idx, feat_name in enumerate(final_feature_names):
                        if feat_name in dynamic_features:
                            windowed_idx = t * n_base_features + feat_idx
                            mean_abs = np.abs(shap_values_windowed[:, windowed_idx]).mean()
                            time_step_importance[time_label][feat_name] = mean_abs
                
                # Create comparison plot
                fig, ax = plt.subplots(figsize=(14, 8))
                
                x = np.arange(len(dynamic_features))
                width = 0.8 / INPUT_WIDTH
                
                colors = plt.cm.viridis(np.linspace(0.2, 0.8, INPUT_WIDTH))
                
                for t_idx, (time_label, feat_importance) in enumerate(time_step_importance.items()):
                    values = [feat_importance.get(f, 0) for f in dynamic_features]
                    offset = (t_idx - INPUT_WIDTH/2 + 0.5) * width
                    bars = ax.bar(x + offset, values, width, label=time_label, color=colors[t_idx])
                
                ax.set_xlabel('Feature', fontsize=12)
                ax.set_ylabel(f'Mean |SHAP Value| ({SHAP_UNITS})', fontsize=12)
                ax.set_title(f'Feature Importance by Time Step\n'
                             f'(Dynamic Features Only)', 
                             fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(dynamic_features, rotation=45, ha='right')
                ax.legend(title='Time Step', loc='upper right')
                #ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                plt.savefig(plot_dir / "shap_time_step_comparison.png", dpi=300, bbox_inches='tight')
                plt.close()
                logging.info("    Saved: shap_time_step_comparison.png")
                
            except Exception as e:
                logging.warning(f"    Could not generate time-step comparison: {e}")
                import traceback
                traceback.print_exc()
        
        # =================================================================
        # STEP 11: Static vs Dynamic Feature Importance
        # =================================================================
        logging.info("\nStep 11: Generating Static vs Dynamic Feature Comparison...")
        
        try:
            # Calculate mean absolute SHAP for each feature
            mean_abs_importance = np.abs(shap_values).mean(axis=0)
            feature_importance_df = pd.DataFrame({
                'feature': shap_feature_names_final,
                'importance': mean_abs_importance
            })
            
            # Categorize as static or dynamic
            def categorize_feature(name):
                # Check if it's a static feature (no time suffix)
                if name in STATIC_FEATURES:
                    return 'Static'
                # Check for time suffix patterns
                if '_t-' in name or name.endswith('_t-0'):
                    return 'Dynamic'
                return 'Static'  # Default to static if no time suffix
            
            feature_importance_df['category'] = feature_importance_df['feature'].apply(categorize_feature)
            
            # Add units column
            feature_importance_df['unit'] = feature_importance_df['feature'].apply(get_feature_unit)
            
            # Sort by importance
            feature_importance_df = feature_importance_df.sort_values('importance', ascending=True)
            
            # Create plot
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Left: Bar plot colored by category
            colors_plot = feature_importance_df['category'].map({'Static': 'coral', 'Dynamic': 'steelblue'})
            axes[0].barh(range(len(feature_importance_df)), 
                        feature_importance_df['importance'], 
                        color=colors_plot)
            axes[0].set_yticks(range(len(feature_importance_df)))
            
            # Create labels with units
            labels_with_units = []
            for _, row in feature_importance_df.iterrows():
                if row['unit']:
                    labels_with_units.append(f"{row['feature']} ({row['unit']})")
                else:
                    labels_with_units.append(row['feature'])
            
            axes[0].set_yticklabels(labels_with_units, fontsize=8)
            axes[0].set_xlabel(f'Mean |SHAP Value| ({SHAP_UNITS})', fontsize=12)
            axes[0].set_title('Feature Importance by Category', fontsize=14, fontweight='bold')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='coral', label='Static'),
                             Patch(facecolor='steelblue', label='Dynamic')]
            axes[0].legend(handles=legend_elements, loc='lower right')
            #axes[0].grid(True, alpha=0.3, axis='x')
            
            # Right: Pie chart of total importance by category
            category_totals = feature_importance_df.groupby('category')['importance'].sum()
            axes[1].pie(category_totals, labels=category_totals.index, autopct='%1.1f%%',
                       colors=['steelblue', 'coral'], startangle=90,
                       explode=[0.02] * len(category_totals))
            axes[1].set_title(f'Total SHAP Importance\nby Feature Category', 
                             fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(plot_dir / "shap_static_vs_dynamic.png", dpi=300, bbox_inches='tight')
            plt.close()
            logging.info("    Saved: shap_static_vs_dynamic.png")
            
            # Save feature importance to CSV (with units)
            feature_importance_df.to_csv(plot_dir / "shap_feature_importance.csv", index=False)
            logging.info("    Saved: shap_feature_importance.csv")
            
        except Exception as e:
            logging.warning(f"    Could not generate static vs dynamic comparison: {e}")
            import traceback
            traceback.print_exc()
        # =================================================================
        # STEP 12: PFT-Stratified SHAP Analysis
        # =================================================================
        logging.info("\nStep 12: Generating PFT-Stratified SHAP Analysis...")
        
        try:
            # First, get PFT labels for each sample
            pft_labels_sampled = get_sample_pft_labels(
                X_original=X_original_sampled,
                feature_names=shap_feature_names if not IS_WINDOWING else shap_feature_names_windowed,
                pft_columns=PFT_COLUMNS
            )
            
            logging.info(f"  PFT distribution in SHAP samples:")
            for pft in np.unique(pft_labels_sampled):
                count = (pft_labels_sampled == pft).sum()
                logging.info(f"    {pft}: {count} samples ({100*count/len(pft_labels_sampled):.1f}%)")
            
            # Create a version of SHAP values and feature names with PFT aggregated
            # (for plots that don't need individual PFT columns)
            shap_values_pft_agg, feature_names_pft_agg = aggregate_pft_shap_values(
                shap_values=shap_values,
                feature_names=shap_feature_names_final,
                pft_columns=PFT_COLUMNS,
                aggregation='sum'
            )
            
            # Also get corresponding X values (for beeswarm plots)
            # For X values, we just need to remove PFT columns
            non_pft_mask = [name not in PFT_COLUMNS for name in shap_feature_names_final]
            X_for_plots_no_pft = X_for_plots[:, non_pft_mask]
            
            # Add a dummy PFT column for X (just use 0s, won't affect beeswarm coloring much)
            X_for_pft_plots = np.hstack([X_for_plots_no_pft, np.zeros((len(X_for_plots), 1))])
            
            # 12a: Feature importance heatmap by PFT
            logging.info("  Generating feature importance heatmap by PFT...")
            plot_feature_importance_by_pft(
                shap_values=shap_values_pft_agg,
                feature_names=feature_names_pft_agg,
                pft_labels=pft_labels_sampled,
                top_n=12,
                output_dir=plot_dir
            )
            
            # 12b: Top features per PFT
            logging.info("  Generating top features per PFT plot...")
            plot_top_features_per_pft(
                shap_values=shap_values_pft_agg,
                feature_names=feature_names_pft_agg,
                pft_labels=pft_labels_sampled,
                top_n=6,
                output_dir=plot_dir
            )
            
            # 12c: Boxplots by PFT
            logging.info("  Generating SHAP boxplots by PFT...")
            plot_shap_by_pft_boxplot(
                shap_values=shap_values_pft_agg,
                feature_names=feature_names_pft_agg,
                pft_labels=pft_labels_sampled,
                top_n=8,
                output_dir=plot_dir
            )
            
            # 12d: Violin plots by PFT
            logging.info("  Generating SHAP violin plots by PFT...")
            plot_shap_by_pft_violin(
                shap_values=shap_values_pft_agg,
                feature_names=feature_names_pft_agg,
                pft_labels=pft_labels_sampled,
                top_n=8,
                output_dir=plot_dir
            )
            
            # 12e: PFT contribution comparison
            logging.info("  Generating PFT contribution comparison...")
            plot_pft_contribution_comparison(
                shap_values=shap_values_pft_agg,
                feature_names=feature_names_pft_agg,
                pft_labels=pft_labels_sampled,
                output_dir=plot_dir
            )
            
            # 12f: Radar chart
            logging.info("  Generating PFT radar chart...")
            plot_pft_radar_chart(
                shap_values=shap_values_pft_agg,
                feature_names=feature_names_pft_agg,
                pft_labels=pft_labels_sampled,
                top_n=8,
                output_dir=plot_dir
            )
            
            # 12g: Individual SHAP summary plots per PFT
            logging.info("  Generating individual SHAP summary plots per PFT...")
            plot_pft_shap_summary(
                shap_values=shap_values_pft_agg,
                X_original=X_for_pft_plots,
                feature_names=feature_names_pft_agg,
                pft_labels=pft_labels_sampled,
                top_n=15,
                output_dir=plot_dir
            )
            
            # 12h: Generate comprehensive CSV report
            logging.info("  Generating PFT SHAP statistics report...")
            pft_report = generate_pft_shap_report(
                shap_values=shap_values_pft_agg,
                feature_names=feature_names_pft_agg,
                pft_labels=pft_labels_sampled,
                output_dir=plot_dir
            )
            
            logging.info("  PFT-Stratified SHAP Analysis complete!")
            
        except Exception as e:
            logging.warning(f"  Could not complete PFT-stratified analysis: {e}")
            import traceback
            traceback.print_exc()
        
        # =================================================================
        # STEP 13: Save SHAP values and metadata (was Step 12)
        # =================================================================
        logging.info("\nStep 13: Saving SHAP values...")
        
        # Save aggregated SHAP values
        shap_output_path = plot_dir / f'shap_values_{run_id}.npz'
        np.savez_compressed(
            shap_output_path,
            shap_values=shap_values,
            shap_values_pft_aggregated=shap_values_pft_agg,
            shap_values_raw=shap_values_raw if IS_WINDOWING else shap_values,
            sampled_indices=sampled_indices,
            feature_names=shap_feature_names_final,
            feature_names_pft_aggregated=feature_names_pft_agg,
            feature_names_windowed=shap_feature_names_windowed if IS_WINDOWING else shap_feature_names_final,
            pft_labels=pft_labels_sampled,
            timestamps=timestamps_sampled,
            latitudes=lat_sampled,
            longitudes=lon_sampled,
            base_value=base_value,
            static_features=STATIC_FEATURES,
            pft_columns=PFT_COLUMNS,
            shap_units=SHAP_UNITS
        )
        logging.info(f"    SHAP values saved to: {shap_output_path}")
        
        # Save feature units mapping
        feature_units_used = {f: get_feature_unit(f) for f in shap_feature_names_final}
        units_path = plot_dir / f'feature_units_{run_id}.json'
        with open(units_path, 'w') as f:
            json.dump({
                'shap_units': SHAP_UNITS,
                'feature_units': feature_units_used,
                'all_feature_units': FEATURE_UNITS,
                'pft_full_names': PFT_FULL_NAMES,
                'pft_colors': PFT_COLORS
            }, f, indent=2)
        logging.info(f"    Feature units saved to: {units_path}")
        
        logging.info("\n" + "="*60)
        logging.info("=== SHAP ANALYSIS COMPLETE ===")
        logging.info("="*60)
        logging.info(f"Total plots generated in: {plot_dir}")
        logging.info(f"SHAP value units: {SHAP_UNITS}")
        logging.info(f"PFT types analyzed: {PFT_COLUMNS}")
        # =================================================================
        # STEP 12: Save SHAP values and metadata
        # =================================================================
        logging.info("\nStep 12: Saving SHAP values...")
        
        # Save aggregated SHAP values
        shap_output_path = plot_dir / f'shap_values_{run_id}.npz'
        np.savez_compressed(
            shap_output_path,
            shap_values=shap_values,
            shap_values_raw=shap_values_raw if IS_WINDOWING else shap_values,
            sampled_indices=sampled_indices,
            feature_names=shap_feature_names_final,
            feature_names_windowed=shap_feature_names_windowed if IS_WINDOWING else shap_feature_names_final,
            timestamps=timestamps_sampled,
            latitudes=lat_sampled,
            longitudes=lon_sampled,
            base_value=base_value,
            static_features=STATIC_FEATURES,
            shap_units=SHAP_UNITS
        )
        logging.info(f"    SHAP values saved to: {shap_output_path}")
        
        # Save feature units mapping
        feature_units_used = {f: get_feature_unit(f) for f in shap_feature_names_final}
        units_path = plot_dir / f'feature_units_{run_id}.json'
        with open(units_path, 'w') as f:
            json.dump({
                'shap_units': SHAP_UNITS,
                'feature_units': feature_units_used,
                'all_feature_units': FEATURE_UNITS
            }, f, indent=2)
        logging.info(f"    Feature units saved to: {units_path}")
        
        logging.info("\n" + "="*60)
        logging.info("=== SHAP ANALYSIS COMPLETE ===")
        logging.info("="*60)
        logging.info(f"Total plots generated in: {plot_dir}")
        logging.info(f"SHAP value units: {SHAP_UNITS}")
        
    except Exception as e:
        logging.error(f"SHAP Analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    logging.info("\n" + "="*60)
    logging.info("=== ALL PROCESSING COMPLETE ===")
    logging.info("="*60)
    
    return all_test_r2_scores, all_test_rmse_scores
    
    


if __name__ == "__main__":
    main()