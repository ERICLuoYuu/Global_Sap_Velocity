"""
Fast Interactive Sap Flow Plotter with Solar Radiation using Plotly
Generates interactive HTML plots with two subplots per tree:
- Top: Sap flow data (with Outliers and Variability Flags)
- Bottom: Environmental data
Uses solar_TIMESTAMP as the x-axis for synchronization
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple
from datetime import datetime
import argparse


class FastSapFlowPlotterDual:
    """
    Efficient plotter that creates dual-subplot interactive Plotly visualizations
    combining sap flow data with environmental data
    """
    
    def __init__(self, data_dir: str, output_dir: str, env_dir: str, 
                 outlier_dir: str = None, variability_dir: str = None, 
                 flagged_dir: str = None, reversed_dir: str = None, env_var: str = 'sw_in'):
        """
        Initialize the plotter with directory paths
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.env_dir = Path(env_dir)
        self.outlier_dir = Path(outlier_dir) if outlier_dir else None
        self.variability_dir = Path(variability_dir) if variability_dir else None
        self.flagged_dir = Path(flagged_dir) if flagged_dir else None
        self.reversed_dir = Path(reversed_dir) if reversed_dir else None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.env_var = env_var
        
        # Decide unit based on env_var
        if self.env_var == 'sw_in':
            self.env_unit = 'W·m⁻²'
        elif self.env_var == 'ta':
            self.env_unit = '°C'
        elif self.env_var == 'vpd':
            self.env_unit = 'kPa'
    
    def _find_matching_env_file(self, location: str, plant_type: str) -> Optional[Path]:
        """Find the matching env file for given location and plant_type"""
        patterns = [
            f"{location}_{plant_type}_outliers_removed.csv",
            f"{location}_{plant_type}.csv",
            f"{location}_*_outliers_removed.csv",
        ]
        for pattern in patterns:
            matches = list(self.env_dir.glob(pattern))
            if matches:
                return matches[0]
        return None
    
    def _load_env_data(self, location: str, plant_type: str) -> Optional[pd.DataFrame]:
        """Load environmental data from env folder"""
        env_file = self._find_matching_env_file(location, plant_type)
        if env_file is None:
            return None
            
        try:
            # FIX 1: Load data first, then set index keeping the column
            df = pd.read_csv(env_file, parse_dates=['solar_TIMESTAMP'])
            if 'solar_TIMESTAMP' in df.columns:
                df.set_index('solar_TIMESTAMP', drop=False, inplace=True)
            
            if self.env_var not in df.columns:
                return None
            
            # Ensure env var is numeric
            df[self.env_var] = pd.to_numeric(df[self.env_var], errors='coerce')
            
            return df
        except Exception as e:
            print(f"    ERROR loading env file: {str(e)}")
            return None
    
    def _load_auxiliary_data(self, column: str):
        """Load outlier, variability, and flagged data for a specific column"""
        outliers_df = None
        variability_df = None
        flagged_df = None
        reversed_df = None
        
        # 1. Load Standard Outliers
        if self.outlier_dir:
            outlier_path = self.outlier_dir / f"{column}_outliers.csv"
            if outlier_path.exists():
                try:
                    outliers_df = pd.read_csv(outlier_path, parse_dates=['solar_TIMESTAMP'])
                except Exception:
                    pass

        # 2. Load Variability Issues (NEW)
        if self.variability_dir:
            var_path = self.variability_dir / f"{column}_variability_mask.csv"
            if var_path.exists():
                try:
                    variability_df = pd.read_csv(var_path, parse_dates=['solar_TIMESTAMP'])
                except Exception:
                    pass
        
        # 3. Load Flagged Data
        if self.flagged_dir:
            flagged_path = self.flagged_dir / f"{column}_flagged.csv"
            if flagged_path.exists():
                try:
                    flagged_df = pd.read_csv(flagged_path, parse_dates=['solar_TIMESTAMP'])
                except Exception:
                    pass
        
        # 4. Load Reversed Data
        if self.reversed_dir:
            reversed_path = self.reversed_dir / f"{column}_reversed.csv"
            if reversed_path.exists():
                try:
                    reversed_df = pd.read_csv(reversed_path, parse_dates=['solar_TIMESTAMP'])
                except Exception:
                    pass
        return outliers_df, variability_df, flagged_df, reversed_df
    
    def create_dual_subplot(self, sap_df: pd.DataFrame, env_df: Optional[pd.DataFrame], 
                           column: str, location: str, plant_type: str):
        """
        Create an interactive Plotly plot with two subplots
        """
        if 'solar_TIMESTAMP' not in sap_df.columns:
            print(f"    WARNING: solar_TIMESTAMP not found in sap data for {column}")
            return None
        
        # Ensure sap_df index is datetime for reliable merging
        if not pd.api.types.is_datetime64_any_dtype(sap_df.index):
            sap_df.index = pd.to_datetime(sap_df.index)

        # Load auxiliary data
        outliers_df, variability_df, flagged_df, reversed_df = self._load_auxiliary_data(column)
        
        # FIX 2: Force numeric conversion to avoid 'str' format errors
        # This turns any non-numeric strings into NaN
        sap_df[column] = pd.to_numeric(sap_df[column], errors='coerce')
        
        # Get valid sap flow data (main line)
        valid_sap = sap_df[[column, 'solar_TIMESTAMP']].dropna()
        
        if valid_sap.empty and (outliers_df is None) and (variability_df is None):
            return None
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'Sap Flow - Tree {column}', self.env_var),
            vertical_spacing=0.12,
            row_heights=[0.5, 0.5],
            shared_xaxes=True
        )
        
        # ===== Subplot 1: Sap Flow Data =====
        
        # 1. Main Trace (Clean Data)
        fig.add_trace(
            go.Scatter(
                x=valid_sap['solar_TIMESTAMP'],
                y=valid_sap[column],
                mode='lines+markers',
                name='Sap Flow',
                line=dict(color='blue', width=1),
                marker=dict(size=3, color='blue'),
                hovertemplate='<b>Time:</b> %{x}<br><b>Value:</b> %{y:.3f}<extra></extra>',
                legendgroup='sap'
            ),
            row=1, col=1
        )
        
        # 2. Flagged Data
        if flagged_df is not None and not flagged_df.empty:
            if 'solar_TIMESTAMP' in flagged_df.columns:
                flagged_merged = pd.merge(
                    flagged_df[[column, 'solar_TIMESTAMP']],
                    sap_df[['solar_TIMESTAMP']],
                    left_on='solar_TIMESTAMP',
                    right_index=True,
                    how='inner'
                ).dropna()
                
                if not flagged_merged.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=flagged_merged['solar_TIMESTAMP'],
                            y=flagged_merged[column],
                            mode='markers',
                            name='Flagged',
                            marker=dict(size=8, color='red', symbol='x'),
                            hovertemplate='<b>Time:</b> %{x}<br><b>Value:</b> %{y:.3f}<br><b>Status:</b> Flagged<extra></extra>',
                            legendgroup='sap'
                        ),
                        row=1, col=1
                    )
        
        # 3. Standard Outliers (Orange)
        if outliers_df is not None and not outliers_df.empty:
            # Filter for rows where is_outlier is True
            outlier_mask = outliers_df['is_outlier'] == True
            outlier_points = outliers_df[outlier_mask]
            
            if not outlier_points.empty:
                # Robust merge using DatetimeIndex
                outlier_merged = pd.merge(
                    outlier_points[['solar_TIMESTAMP', 'value']],
                    sap_df[['solar_TIMESTAMP']],
                    left_on='solar_TIMESTAMP',
                    right_index=True,
                    how='inner'
                )
                
                if not outlier_merged.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=outlier_merged['solar_TIMESTAMP'],
                            y=outlier_merged['value'],
                            mode='markers',
                            name='Outliers',
                            marker=dict(size=8, color='orange', symbol='circle'),
                            hovertemplate='<b>Time:</b> %{x}<br><b>Value:</b> %{y:.3f}<br><b>Type:</b> Statistical Outlier<extra></extra>',
                            legendgroup='sap'
                        ),
                        row=1, col=1
                    )

        # 4. Variability Issues (Black Square)
        if variability_df is not None and not variability_df.empty:
            # Filter for rows where is_variability_issue is True
            if 'is_variability_issue' in variability_df.columns:
                var_mask = variability_df['is_variability_issue'] == True
                var_points = variability_df[var_mask]
                
                if not var_points.empty:
                    var_merged = pd.merge(
                        var_points[['solar_TIMESTAMP', 'value']],
                        sap_df[['solar_TIMESTAMP']],
                        left_on='solar_TIMESTAMP',
                        right_index=True,
                        how='inner'
                    )
                    
                    if not var_merged.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=var_merged['solar_TIMESTAMP'],
                                y=var_merged['value'],
                                mode='markers',
                                name='Variability Filter',
                                marker=dict(size=6, color='black', symbol='square'),
                                hovertemplate='<b>Time:</b> %{x}<br><b>Value:</b> %{y:.3f}<br><b>Type:</b> Variability Issue<extra></extra>',
                                legendgroup='sap'
                            ),
                            row=1, col=1
                        )
        
        # 5. Reversed Data (Purple Diamond)
        if reversed_df is not None and not reversed_df.empty:
            # Filter for rows where is_reversed is True
            if 'is_reversed' in reversed_df.columns:
                rev_mask = reversed_df['is_reversed'] == True
                rev_points = reversed_df[rev_mask]
                
                if not rev_points.empty:
                    rev_merged = pd.merge(
                        rev_points[['solar_TIMESTAMP', 'value']],
                        sap_df[['solar_TIMESTAMP']],
                        left_on='solar_TIMESTAMP',
                        right_index=True,
                        how='inner'
                    )
                    
                    if not rev_merged.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=rev_merged['solar_TIMESTAMP'],
                                y=rev_merged['value'],
                                mode='markers',
                                name='Reversed Data',
                                marker=dict(size=6, color='purple', symbol='diamond'),
                                hovertemplate='<b>Time:</b> %{x}<br><b>Value:</b> %{y:.3f}<br><b>Type:</b> Reversed Data<extra></extra>',
                                legendgroup='sap'
                            ),
                            row=1, col=1
                        )
        
        # ===== Subplot 2: Environmental =====
        if env_df is not None and self.env_var in env_df.columns and 'solar_TIMESTAMP' in env_df.columns:
            valid_env = env_df[[self.env_var, 'solar_TIMESTAMP']].dropna()
            
            if not valid_env.empty:
                fig.add_trace(
                    go.Scatter(
                        x=valid_env['solar_TIMESTAMP'],
                        y=valid_env[self.env_var],
                        mode='lines+markers',
                        name=self.env_var,
                        line=dict(color='green', width=1),
                        marker=dict(size=3, color='green'),
                        hovertemplate=f'<b>Time:</b> %{{x}}<br><b>{self.env_var}:</b> %{{y:.3f}}<extra></extra>',
                        legendgroup='env'
                    ),
                    row=2, col=1
                )
        
        # Stats
        valid_percent = (len(valid_sap) / len(sap_df)) * 100 if len(sap_df) > 0 else 0
        outlier_count = outliers_df['is_outlier'].sum() if outliers_df is not None else 0
        var_count = variability_df['is_variability_issue'].sum() if variability_df is not None else 0
        
        date_start = sap_df.index.min().strftime('%Y-%m-%d')
        date_end = sap_df.index.max().strftime('%Y-%m-%d')
        
        # Calculate min/max safely now that data is numeric
        sap_min = valid_sap[column].min()
        sap_max = valid_sap[column].max()
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'Sap Flow & {self.env_var.upper()} - {location} {plant_type}<br>' +
                        f'Tree {column} | Period: {date_start} to {date_end}<br>' +
                        f'<sub>Valid: {valid_percent:.1f}% | Outliers: {outlier_count} | Var. Removed: {var_count} | ' +
                        f'Sap Range: {sap_min:.3f} to {sap_max:.3f}</sub>',
                'x': 0.5,
                'xanchor': 'center'
            },
            hovermode='closest',
            template='plotly_white',
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Solar Timestamp", row=2, col=1)
        fig.update_yaxes(title_text="Sap Flow (cm³·cm⁻²·h⁻¹)", row=1, col=1)
        fig.update_yaxes(title_text=f"{self.env_var} ({self.env_unit})", row=2, col=1)
        
        fig.update_xaxes(
            rangeslider_visible=True,
            row=2, col=1
        )
        
        return fig
    
    def process_file(self, file_path: Path):
        """Process a single outlier_removed file"""
        print(f"\nProcessing: {file_path.name}")
        parts = file_path.stem.split('_')
        
        if 'outliers_removed' in file_path.stem:
            outliers_idx = parts.index('outliers')
            location = '_'.join(parts[:2])
            plant_type = '_'.join(parts[2:outliers_idx])
        else:
            location = '_'.join(parts[:2])
            plant_type = '_'.join(parts[2:])
        
        # FIX 1: Load the DataFrame then set the index with drop=False
        sap_df = pd.read_csv(file_path, parse_dates=['solar_TIMESTAMP'])
        sap_df.set_index('solar_TIMESTAMP', drop=False, inplace=True)
        
        env_df = self._load_env_data(location, plant_type)
        
        # Filter columns to skip non-sap data
        # Added 'TIMESTAMP' to exclusion to prevent warning
        plot_columns = [
            col for col in sap_df.columns 
            if col not in ['solar_TIMESTAMP', 'TIMESTAMP', 'TIMESTAMP_LOCAL', 'lat', 'long', 'Unnamed: 0']
            and not col.startswith('Unnamed')
        ]
        
        file_output_dir = self.output_dir / f"{location}_{plant_type}"
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        for column in plot_columns:
            if sap_df[column].isna().all():
                continue
            
            fig = self.create_dual_subplot(sap_df, env_df, column, location, plant_type)
            
            if fig is not None:
                output_path = file_output_dir / f"tree_{column}_dual.html"
                fig.write_html(output_path, config={'displayModeBar': True})
                print(f"    Created plot: {output_path.name}")
                del fig

    def process_all_files(self, file_pattern: str = "*_outliers_removed.csv"):
        files = list(self.data_dir.glob(file_pattern))
        print(f"Found {len(files)} files to process")
        for file_path in files:
            try:
                self.process_file(file_path)
            except Exception as e:
                print(f"ERROR processing {file_path.name}: {str(e)}")
                import traceback
                traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--env-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--outlier-dir', default=None)
    parser.add_argument('--variability-dir', default=None)
    parser.add_argument('--flagged-dir', default=None)
    args = parser.parse_args()
    
    plotter = FastSapFlowPlotterDual(
        data_dir=args.data_dir,
        env_dir=args.env_dir,
        output_dir=args.output_dir,
        outlier_dir=args.outlier_dir,
        variability_dir=args.variability_dir,
        flagged_dir=args.flagged_dir
    )
    plotter.process_all_files()

if __name__ == "__main__":
    # --- UPDATE YOUR PATHS HERE ---
    DATA_DIR = "./outputs/processed_data/sapwood/sap/outliers_removed"
    ENV_DIR = "./outputs/processed_data/sapwood/env/outliers_removed"
    OUTPUT_DIR = "./outputs/figures/sapwood/sap/cleaned_interactive_dual_vpd"
    
    # Path to standard outliers
    OUTLIER_DIR = "./outputs/processed_data/sapwood/sap/outliers"
    
    # Path to variability masks
    VARIABILITY_DIR = "./outputs/processed_data/sapwood/sap/variability_masks"
    
    FLAGGED_DIR = "./outputs/processed_data/sapwood/sap/filtered"
    REVERSED_DIR = "./outputs/processed_data/sapwood/sap/reversed"
    
    print("Fast Interactive Sap Flow Plotter (Dual Subplots)")
    
    plotter = FastSapFlowPlotterDual(
        data_dir=DATA_DIR,
        env_dir=ENV_DIR,
        output_dir=OUTPUT_DIR,
        outlier_dir=OUTLIER_DIR,
        variability_dir=VARIABILITY_DIR,
        flagged_dir=FLAGGED_DIR,
        reversed_dir=REVERSED_DIR,
        env_var='vpd'
    )
    
    plotter.process_all_files()