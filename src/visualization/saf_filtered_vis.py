#!/usr/bin/env python3
"""
Sap Flow Data Visualization Tool
Creates interactive Plotly visualizations from sapf_data_filtered.csv files
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
from datetime import datetime
import sys

# ============================================================================
# CONFIGURATION - EDIT THIS SECTION
# ============================================================================

# Path to your data directory
DATA_DIR = "./outputs/processed_data/sapwood/sap/filtered/TEST"

# Output directory for HTML plots
OUTPUT_DIR = "./sapflow_plots"

# Plot settings
PLOT_HEIGHT = 600  # Height of individual plots in pixels
COMPARISON_HEIGHT_PER_SITE = 300  # Height per site in comparison plot
PLOT_TEMPLATE = "plotly_white"  # Options: plotly, plotly_white, plotly_dark, ggplot2, seaborn

# Color palette (optional - leave empty to use default)
COLOR_PALETTE = []  # e.g., ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# ============================================================================
# MAIN SCRIPT - NO NEED TO EDIT BELOW
# ============================================================================

def find_data_files(data_dir):
    """Find all sapf_data_filtered.csv files in the directory"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Error: Directory '{data_dir}' does not exist!")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Please update DATA_DIR in the script or create the directory.")
        return []
    
    files = list(data_path.glob("*sapf_data_filtered.csv"))
    return sorted(files)


def load_and_prepare_data(file_path):
    """Load CSV file and prepare data for plotting"""
    try:
        df = pd.read_csv(file_path)
        
        # Convert solar_TIMESTAMP to datetime
        if 'solar_TIMESTAMP' in df.columns:
            df['solar_TIMESTAMP'] = pd.to_datetime(df['solar_TIMESTAMP'])
        else:
            print(f"   ⚠️  Warning: 'solar_TIMESTAMP' column not found in {file_path.name}")
            return None
        
        # Get sapflow columns (exclude timestamp columns)
        timestamp_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'TIMESTAMP_LOCAL']
        sapflow_cols = [col for col in df.columns if col not in timestamp_cols]
        
        return df, sapflow_cols
    
    except Exception as e:
        print(f"   ❌ Error loading {file_path.name}: {str(e)}")
        return None


def create_individual_plot(file_path, output_dir, color_palette=None):
    """Create an interactive plot for a single site"""
    
    result = load_and_prepare_data(file_path)
    if result is None:
        return None
    
    df, sapflow_cols = result
    site_name = file_path.stem.replace('_sapf_data_filtered', '')
    
    print(f"\n📊 Creating plot for: {site_name}")
    print(f"   - Time range: {df['solar_TIMESTAMP'].min()} to {df['solar_TIMESTAMP'].max()}")
    print(f"   - Number of sensors: {len(sapflow_cols)}")
    print(f"   - Data points: {len(df)}")
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each plant/sensor
    for idx, col in enumerate(sapflow_cols):
        # Determine color
        if color_palette and len(color_palette) > 0:
            color = color_palette[idx % len(color_palette)]
            line_dict = dict(color=color)
        else:
            line_dict = None
        
        fig.add_trace(go.Scatter(
            x=df['solar_TIMESTAMP'],
            y=df[col],
            mode='lines',
            name=col,
            line=line_dict,
            hovertemplate=(
                '<b>%{fullData.name}</b><br>' +
                'Time: %{x|%Y-%m-%d %H:%M}<br>' +
                'Sap Flow: %{y:.3f}<br>' +
                '<extra></extra>'
            )
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'Sap Flow Data - {site_name}',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title='Solar Timestamp',
        yaxis_title='Sap Flow',
        hovermode='x unified',
        template=PLOT_TEMPLATE,
        height=PLOT_HEIGHT,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.05),
            type='date'
        ),
        margin=dict(r=150)  # More space for legend
    )
    
    # Add range selector buttons
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1 Day", step="day", stepmode="backward"),
                dict(count=3, label="3 Days", step="day", stepmode="backward"),
                dict(count=7, label="1 Week", step="day", stepmode="backward"),
                dict(count=1, label="1 Month", step="month", stepmode="backward"),
                dict(count=3, label="3 Months", step="month", stepmode="backward"),
                dict(step="all", label="All Data")
            ]),
            bgcolor="rgba(150, 150, 150, 0.1)",
            activecolor="rgba(100, 100, 200, 0.3)"
        )
    )
    
    # Save HTML file
    output_file = output_dir / f"{site_name}_sapflow.html"
    fig.write_html(
        output_file,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
        }
    )
    
    print(f"   ✅ Saved: {output_file}")
    
    return fig, df, sapflow_cols


def create_comparison_plot(files_data, output_dir, max_sites=None):
    """Create a comparison plot with subplots for multiple sites"""
    
    if max_sites:
        files_data = files_data[:max_sites]
    
    n_sites = len(files_data)
    
    print(f"\n📊 Creating comparison plot with {n_sites} sites...")
    
    # Create subplots
    subplot_titles = [data['site_name'] for data in files_data]
    
    fig = make_subplots(
        rows=n_sites,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08 / n_sites if n_sites > 1 else 0.1,
        shared_xaxes=True,
        x_title='Solar Timestamp'
    )
    
    # Add traces for each site
    for idx, data in enumerate(files_data, 1):
        df = data['df']
        sapflow_cols = data['sapflow_cols']
        site_name = data['site_name']
        
        for col in sapflow_cols:
            fig.add_trace(
                go.Scatter(
                    x=df['solar_TIMESTAMP'],
                    y=df[col],
                    mode='lines',
                    name=f"{site_name} - {col}",
                    legendgroup=site_name,
                    hovertemplate=(
                        f'<b>{site_name} - {col}</b><br>' +
                        'Time: %{x|%Y-%m-%d %H:%M}<br>' +
                        'Sap Flow: %{y:.3f}<br>' +
                        '<extra></extra>'
                    )
                ),
                row=idx,
                col=1
            )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Sap Flow Comparison - All Sites',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=COMPARISON_HEIGHT_PER_SITE * n_sites,
        hovermode='x unified',
        template=PLOT_TEMPLATE,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        margin=dict(r=200)
    )
    
    # Update y-axes labels
    for i in range(1, n_sites + 1):
        fig.update_yaxes(title_text="Sap Flow", row=i, col=1)
    
    # Save HTML file
    output_file = output_dir / "sapflow_comparison_all_sites.html"
    fig.write_html(
        output_file,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
        }
    )
    
    print(f"   ✅ Saved: {output_file}")
    
    return fig


def generate_summary_statistics(files_data, output_dir):
    """Generate and save summary statistics"""
    
    print(f"\n📈 Generating summary statistics...")
    
    summary_data = []
    
    for data in files_data:
        df = data['df']
        sapflow_cols = data['sapflow_cols']
        site_name = data['site_name']
        
        for col in sapflow_cols:
            values = df[col].dropna()
            
            if len(values) > 0:
                summary_data.append({
                    'Site': site_name,
                    'Sensor': col,
                    'Count': len(values),
                    'Mean': values.mean(),
                    'Std': values.std(),
                    'Min': values.min(),
                    'Q25': values.quantile(0.25),
                    'Median': values.median(),
                    'Q75': values.quantile(0.75),
                    'Max': values.max(),
                    'Start_Date': df['solar_TIMESTAMP'].min(),
                    'End_Date': df['solar_TIMESTAMP'].max()
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    output_file = output_dir / "sapflow_summary_statistics.csv"
    summary_df.to_csv(output_file, index=False)
    
    print(f"   ✅ Saved: {output_file}")
    print(f"\n📊 Summary Statistics Preview:")
    print(summary_df.head(10).to_string(index=False))
    
    return summary_df


def main():
    """Main function to run the visualization pipeline"""
    
    print("=" * 70)
    print("  SAP FLOW DATA VISUALIZATION TOOL")
    print("=" * 70)
    print(f"\n🔍 Looking for data files in: {DATA_DIR}")
    
    # Find data files
    files = find_data_files(DATA_DIR)
    
    if len(files) == 0:
        print("\n❌ No *sapf_data_filtered.csv files found!")
        print("\nTroubleshooting:")
        print("1. Check that DATA_DIR is correct in the script")
        print("2. Ensure files end with 'sapf_data_filtered.csv'")
        print("3. Verify you're running the script from the correct directory")
        return
    
    print(f"✅ Found {len(files)} file(s):")
    for f in files:
        print(f"   - {f.name}")
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir.absolute()}")
    
    # Process files and create individual plots
    print("\n" + "=" * 70)
    print("  CREATING INDIVIDUAL PLOTS")
    print("=" * 70)
    
    files_data = []
    
    for file_path in files:
        result = create_individual_plot(file_path, output_dir, COLOR_PALETTE)
        
        if result:
            fig, df, sapflow_cols = result
            site_name = file_path.stem.replace('_sapf_data_filtered', '')
            files_data.append({
                'site_name': site_name,
                'df': df,
                'sapflow_cols': sapflow_cols,
                'file_path': file_path
            })
    
    # Create comparison plot if multiple sites
    if len(files_data) > 1:
        print("\n" + "=" * 70)
        print("  CREATING COMPARISON PLOT")
        print("=" * 70)
        create_comparison_plot(files_data, output_dir)
    
    # Generate summary statistics
    print("\n" + "=" * 70)
    print("  GENERATING SUMMARY STATISTICS")
    print("=" * 70)
    generate_summary_statistics(files_data, output_dir)
    
    # Final summary
    print("\n" + "=" * 70)
    print("  ✅ VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Created {len(files_data)} individual plots")
    if len(files_data) > 1:
        print(f"📊 Created 1 comparison plot")
    print(f"📈 Generated summary statistics")
    print(f"\n📁 All files saved to: {output_dir.absolute()}")
    print("\n💡 Tip: Open the HTML files in your web browser to explore the interactive plots!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)