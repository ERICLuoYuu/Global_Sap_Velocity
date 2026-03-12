"""
Interactive Sapwood Data Plotter using Plotly
Generates interactive HTML plots for sapwood-level sap flow data.
Similar to sapf_plotter.py but adapted for the SAPFLUXNET format sapwood data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List
import argparse


class SapwoodPlotter:
    """
    Interactive plotter for sapwood-level sap flow data
    Creates dual-subplot visualizations with sap flow and optional environmental data
    """
    
    def __init__(self, data_dir: str, output_dir: str, env_dir: str = None,
                 env_var: str = None):
        """
        Initialize the plotter with directory paths
        
        Parameters
        ----------
        data_dir : str
            Directory containing sapflow_sapwood.csv files
        output_dir : str
            Directory to save output HTML plots
        env_dir : str, optional
            Directory containing env_data.csv files (if available)
        env_var : str, optional
            Single environmental variable to plot alongside sap flow
            (e.g. 'ta', 'vpd', 'sw_in', 'precip', 'ws', 'pa').
            If None, no env subplot is added.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.env_dir = Path(env_dir) if env_dir else self.data_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.env_var = env_var
        
        # Default unit label
        self.sap_unit = 'cm³·cm⁻²·h⁻¹'
    
    # --- env variable display helpers ---
    _ENV_LABELS = {
        'ta':        'Air Temperature (°C)',
        'ta_qc':     'TA QC Flag',
        'vpd':       'VPD (hPa)',
        'vpd_qc':    'VPD QC Flag',
        'sw_in':     'SW Radiation (W/m²)',
        'sw_in_qc':  'SW_IN QC Flag',
        'pa':        'Pressure (kPa)',
        'pa_qc':     'PA QC Flag',
        'ws':        'Wind Speed (m/s)',
        'ws_qc':     'WS QC Flag',
        'precip':    'Precipitation (mm)',
        'precip_qc': 'Precip QC Flag',
        'ppfd':      'PPFD (µmol/m²/s)',
        'rh':        'Relative Humidity (%)',
        'ts':        'Soil Temperature (°C)',
        'swc':       'Soil Water Content (%)',
    }

    def _env_label(self, col_name: str) -> str:
        """Return a human-readable axis label for an env variable."""
        return self._ENV_LABELS.get(col_name, col_name)
    
    def _load_env_data(self, site_prefix: str) -> Optional[pd.DataFrame]:
        """Load environmental data for a given site"""
        env_patterns = [
            f"{site_prefix}_env_data.csv",
            f"{site_prefix}*_env_data.csv",
        ]
        
        for pattern in env_patterns:
            matches = list(self.env_dir.glob(pattern))
            if matches:
                try:
                    df = pd.read_csv(matches[0])
                    
                    # Normalise timestamp column to lowercase 'timestamp'
                    if 'TIMESTAMP' in df.columns and 'timestamp' not in df.columns:
                        df = df.rename(columns={'TIMESTAMP': 'timestamp'})
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
                    df.set_index('timestamp', drop=False, inplace=True)
                    
                    # Check if there are any actual env columns (not just timestamp)
                    env_cols = [c for c in df.columns if c not in ['timestamp', 'Unnamed: 0']]
                    if env_cols:
                        return df
                except Exception as e:
                    print(f"    Warning: Could not load env data: {e}")
        return None
    
    def _get_color_palette(self, n_colors: int) -> List[str]:
        """Generate a color palette for n trees"""
        # Use a set of distinct colors
        base_colors = [
            '#1f77b4',  # blue
            '#ff7f0e',  # orange
            '#2ca02c',  # green
            '#d62728',  # red
            '#9467bd',  # purple
            '#8c564b',  # brown
            '#e377c2',  # pink
            '#7f7f7f',  # gray
            '#bcbd22',  # olive
            '#17becf',  # cyan
            '#aec7e8',  # light blue
            '#ffbb78',  # light orange
            '#98df8a',  # light green
            '#ff9896',  # light red
            '#c5b0d5',  # light purple
        ]
        
        if n_colors <= len(base_colors):
            return base_colors[:n_colors]
        
        # If we need more colors, cycle through
        return [base_colors[i % len(base_colors)] for i in range(n_colors)]
    
    def create_site_overview_plot(self, sap_df: pd.DataFrame, env_df: Optional[pd.DataFrame],
                                   site_name: str, tree_columns: List[str]) -> go.Figure:
        """
        Create an overview plot showing all trees for a site
        
        Parameters
        ----------
        sap_df : pd.DataFrame
            Sapflow data with timestamp index
        env_df : pd.DataFrame, optional
            Environmental data
        site_name : str
            Name of the site
        tree_columns : list
            List of column names (tree identifiers)
        
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        env_cols = []
        if env_df is not None and self.env_var:
            available = [c for c in env_df.columns if c not in ['timestamp', 'TIMESTAMP', 'Unnamed: 0']]
            if self.env_var in available:
                env_cols = [self.env_var]
            else:
                print(f"    Warning: env variable '{self.env_var}' not found. Available: {available}")
        has_env = len(env_cols) > 0
        n_rows = 2 if has_env else 1
        
        # Build subplot titles and row heights
        subplot_titles = [f'Sapwood Sap Flow - {site_name} (All Trees)']
        if has_env:
            row_heights = [0.65, 0.35]
            subplot_titles.append(self._env_label(self.env_var))
        else:
            row_heights = [1.0]
        
        fig = make_subplots(
            rows=n_rows, cols=1,
            subplot_titles=tuple(subplot_titles),
            vertical_spacing=0.08,
            row_heights=row_heights,
            shared_xaxes=True,
        )
        
        colors = self._get_color_palette(len(tree_columns))
        
        # Add traces for each tree
        for i, col in enumerate(tree_columns):
            if col not in sap_df.columns:
                continue
            
            # Ensure numeric
            sap_df[col] = pd.to_numeric(sap_df[col], errors='coerce')
            valid_data = sap_df[['timestamp', col]].dropna()
            
            if valid_data.empty:
                continue
            
            # Simplify the name for legend
            display_name = col.replace(f"{site_name}-", "")
            
            fig.add_trace(
                go.Scatter(
                    x=valid_data['timestamp'],
                    y=valid_data[col],
                    mode='lines',
                    name=display_name,
                    line=dict(color=colors[i], width=1),
                    hovertemplate=f'<b>{display_name}</b><br>' +
                                  '<b>Time:</b> %{x}<br>' +
                                  '<b>Value:</b> %{y:.4f} ' + self.sap_unit + '<extra></extra>',
                    legendgroup='sap'
                ),
                row=1, col=1
            )
        
        # Add the single environmental variable subplot
        if has_env:
            env_col = env_cols[0]
            valid_env = env_df[['timestamp', env_col]].dropna() if 'timestamp' in env_df.columns else env_df[['TIMESTAMP', env_col]].dropna().rename(columns={'TIMESTAMP': 'timestamp'})
            if not valid_env.empty:
                fig.add_trace(
                    go.Scatter(
                        x=valid_env['timestamp'],
                        y=valid_env[env_col],
                        mode='lines',
                        name=self._env_label(env_col),
                        line=dict(color='#2ca02c', width=1),
                        hovertemplate=f'<b>{self._env_label(env_col)}</b><br>' +
                                      '<b>Time:</b> %{x}<br>' +
                                      '<b>Value:</b> %{y:.3f}<extra></extra>',
                        legendgroup='env',
                        showlegend=False,
                    ),
                    row=2, col=1
                )
        
        # Calculate stats
        date_start = pd.to_datetime(sap_df['timestamp'].min())
        date_end = pd.to_datetime(sap_df['timestamp'].max())
        n_trees = len(tree_columns)
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'Sapwood Sap Flow Overview - {site_name}<br>' +
                        f'<sub>{n_trees} trees | {date_start.strftime("%Y-%m-%d")} to {date_end.strftime("%Y-%m-%d")}</sub>',
                'x': 0.5,
                'xanchor': 'center'
            },
            hovermode='x unified',
            template='plotly_white',
            height=800 if has_env else 600,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=10)
            )
        )
        
        # Set datetime axis type for all x-axes
        for r in range(1, n_rows + 1):
            fig.update_xaxes(type='date', row=r, col=1)
        fig.update_xaxes(title_text="Timestamp", row=n_rows, col=1)
        
        fig.update_yaxes(title_text=f"Sap Flow ({self.sap_unit})", row=1, col=1)
        
        if has_env:
            fig.update_yaxes(
                title_text=self._env_label(env_cols[0]),
                row=2, col=1,
            )
        
        # Add range slider on the bottom-most subplot
        fig.update_xaxes(
            rangeslider_visible=True,
            row=n_rows, col=1
        )
        
        return fig
    
    def create_single_tree_plot(self, sap_df: pd.DataFrame, env_df: Optional[pd.DataFrame],
                                 site_name: str, tree_col: str) -> Optional[go.Figure]:
        """
        Create a detailed plot for a single tree
        
        Parameters
        ----------
        sap_df : pd.DataFrame
            Sapflow data with timestamp index
        env_df : pd.DataFrame, optional
            Environmental data
        site_name : str
            Name of the site
        tree_col : str
            Column name for the tree
        
        Returns
        -------
        go.Figure or None
            Plotly figure object, or None if no valid data
        """
        if tree_col not in sap_df.columns:
            return None
        
        # Ensure numeric
        sap_df[tree_col] = pd.to_numeric(sap_df[tree_col], errors='coerce')
        valid_data = sap_df[['timestamp', tree_col]].dropna()
        
        if valid_data.empty:
            return None
        
        env_cols = []
        if env_df is not None and self.env_var:
            available = [c for c in env_df.columns if c not in ['timestamp', 'TIMESTAMP', 'Unnamed: 0']]
            if self.env_var in available:
                env_cols = [self.env_var]
        has_env = len(env_cols) > 0
        n_rows = 2 if has_env else 1
        
        subplot_titles = [f'Sap Flow - {tree_col}']
        if has_env:
            row_heights = [0.65, 0.35]
            subplot_titles.append(self._env_label(self.env_var))
        else:
            row_heights = [1.0]
        
        fig = make_subplots(
            rows=n_rows, cols=1,
            subplot_titles=tuple(subplot_titles),
            vertical_spacing=0.08,
            row_heights=row_heights,
            shared_xaxes=True,
        )
        
        # Main sap flow trace
        fig.add_trace(
            go.Scatter(
                x=valid_data['timestamp'],
                y=valid_data[tree_col],
                mode='lines+markers',
                name='Sap Flow',
                line=dict(color='#1f77b4', width=1),
                marker=dict(size=3, color='#1f77b4'),
                hovertemplate='<b>Time:</b> %{x}<br>' +
                              '<b>Value:</b> %{y:.4f} ' + self.sap_unit + '<extra></extra>',
            ),
            row=1, col=1
        )
        
        # Add the single environmental variable subplot
        if has_env:
            env_col = env_cols[0]
            valid_env = env_df[['timestamp', env_col]].dropna() if 'timestamp' in env_df.columns else env_df[['TIMESTAMP', env_col]].dropna().rename(columns={'TIMESTAMP': 'timestamp'})
            if not valid_env.empty:
                fig.add_trace(
                    go.Scatter(
                        x=valid_env['timestamp'],
                        y=valid_env[env_col],
                        mode='lines',
                        name=self._env_label(env_col),
                        line=dict(color='#2ca02c', width=1),
                        hovertemplate=f'<b>{self._env_label(env_col)}</b><br>' +
                                      '<b>Time:</b> %{x}<br>' +
                                      '<b>Value:</b> %{y:.3f}<extra></extra>',
                    ),
                    row=2, col=1
                )
        
        # Stats
        date_start = pd.to_datetime(valid_data['timestamp'].min())
        date_end = pd.to_datetime(valid_data['timestamp'].max())
        sap_min = valid_data[tree_col].min()
        sap_max = valid_data[tree_col].max()
        sap_mean = valid_data[tree_col].mean()
        valid_pct = (len(valid_data) / len(sap_df)) * 100
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'Sapwood Sap Flow - {site_name}<br>' +
                        f'<b>{tree_col}</b><br>' +
                        f'<sub>{date_start.strftime("%Y-%m-%d")} to {date_end.strftime("%Y-%m-%d")} | Valid: {valid_pct:.1f}% | ' +
                        f'Min: {sap_min:.4f} | Max: {sap_max:.4f} | Mean: {sap_mean:.4f}</sub>',
                'x': 0.5,
                'xanchor': 'center'
            },
            hovermode='closest',
            template='plotly_white',
            height=700 if has_env else 500,
            showlegend=True
        )
        
        # Set datetime axis type for all x-axes
        for r in range(1, n_rows + 1):
            fig.update_xaxes(type='date', row=r, col=1)
        fig.update_xaxes(title_text="Timestamp", row=n_rows, col=1)
        
        fig.update_yaxes(title_text=f"Sap Flow ({self.sap_unit})", row=1, col=1)
        
        if has_env:
            fig.update_yaxes(
                title_text=self._env_label(env_cols[0]),
                row=2, col=1,
            )
        
        # Add range slider on the bottom-most subplot
        fig.update_xaxes(
            rangeslider_visible=True,
            row=n_rows, col=1
        )
        
        return fig
    
    def process_file(self, file_path: Path, create_individual: bool = True):
        """
        Process a single sapflow_sapwood.csv file
        
        Parameters
        ----------
        file_path : Path
            Path to the sapflow data file
        create_individual : bool
            Whether to create individual tree plots (default True)
        """
        print(f"\nProcessing: {file_path.name}")
        
        # Extract site name from filename (e.g., "CH-Dav_halfhourly" from "CH-Dav_halfhourly_sapflow_sapwood.csv")
        stem = file_path.stem.replace('_sapflow_sapwood', '')
        site_name = stem
        
        # Load sapflow data
        try:
            sap_df = pd.read_csv(file_path, parse_dates=['timestamp'])
            sap_df.set_index('timestamp', drop=False, inplace=True)
        except Exception as e:
            print(f"    ERROR loading file: {e}")
            return
        
        # Load environmental data
        env_df = self._load_env_data(stem)
        
        # Get tree columns (exclude timestamp and unnamed columns)
        tree_columns = [
            col for col in sap_df.columns 
            if col not in ['timestamp', 'Unnamed: 0']
            and not col.startswith('Unnamed')
        ]
        
        if not tree_columns:
            print(f"    No tree columns found in {file_path.name}")
            return
        
        print(f"    Found {len(tree_columns)} trees")
        
        # Create output directory for this site
        site_output_dir = self.output_dir / site_name
        site_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Create overview plot (all trees)
        fig_overview = self.create_site_overview_plot(sap_df, env_df, site_name, tree_columns)
        overview_path = site_output_dir / f"{site_name}_overview.html"
        fig_overview.write_html(overview_path, config={'displayModeBar': True})
        print(f"    Created overview: {overview_path.name}")
        del fig_overview
        
        # 2. Create individual tree plots
        if create_individual:
            for tree_col in tree_columns:
                fig_tree = self.create_single_tree_plot(sap_df, env_df, site_name, tree_col)
                if fig_tree is not None:
                    # Sanitize column name for filename
                    safe_name = tree_col.replace('/', '_').replace('\\', '_').replace(':', '_')
                    tree_path = site_output_dir / f"{safe_name}.html"
                    fig_tree.write_html(tree_path, config={'displayModeBar': True})
                    del fig_tree
            print(f"    Created {len(tree_columns)} individual tree plots")
    
    def process_all_files(self, file_pattern: str = "*_sapflow_sapwood.csv", create_individual: bool = True):
        """
        Process all sapflow files matching the pattern
        
        Parameters
        ----------
        file_pattern : str
            Glob pattern for finding files
        create_individual : bool
            Whether to create individual tree plots
        """
        files = list(self.data_dir.glob(file_pattern))
        print(f"Found {len(files)} sapwood files to process")
        
        for file_path in files:
            try:
                self.process_file(file_path, create_individual=create_individual)
            except Exception as e:
                print(f"ERROR processing {file_path.name}: {str(e)}")
                import traceback
                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Interactive Sapwood Data Plotter')
    parser.add_argument('--data-dir', required=True, help='Directory containing sapflow_sapwood.csv files')
    parser.add_argument('--output-dir', required=True, help='Directory to save output HTML plots')
    parser.add_argument('--env-dir', default=None, help='Directory containing env_data.csv files')
    parser.add_argument('--env-var', default='SW_IN',
                        help='Single env variable to plot (e.g. ta, vpd, sw_in, precip, ws, pa)')
    parser.add_argument('--no-individual', action='store_true', help='Skip individual tree plots')
    args = parser.parse_args()
    
    plotter = SapwoodPlotter(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        env_dir=args.env_dir,
        env_var=args.env_var,
    )
    plotter.process_all_files(create_individual=not args.no_individual)


if __name__ == "__main__":
    # --- Default paths for direct execution ---
    DATA_DIR = "./Sapflow_SAPFLUXNET_format_unitcon/sapwood"
    OUTPUT_DIR = "./Sapflow_SAPFLUXNET_format_unitcon/plots/sapwood"
    
    print("=" * 60)
    print("Interactive Sapwood Data Plotter")
    print("=" * 60)
    
    plotter = SapwoodPlotter(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        env_dir=DATA_DIR,  # env_data.csv files are in the same directory
        env_var='sw_in',   # shortwave incoming radiation
    )
    
    # Process all sapwood files, creating both overview and individual plots
    plotter.process_all_files(create_individual=True)
    
    print("\n" + "=" * 60)
    print("Plotting complete!")
    print(f"Output saved to: {OUTPUT_DIR}")
    print("=" * 60)
