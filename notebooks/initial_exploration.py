import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import warnings
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser" #This will force Plotly to open plots in your default web browser.
warnings.filterwarnings('ignore')

class GermanSapFlowAnalyzer:
    """
    Simplified analyzer for German (DEU) SapFlow data with individual plant plots
    """
    def __init__(self, base_dir: str = "./data/raw/0.1.5/0.1.5/csv/sapwood"):
        self.data_dir = Path(base_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Directory not found: {self.data_dir}")
        
        self.sapflow_data = {}
        self._load_sapflow_data()
    
    def _load_sapflow_data(self):
        """Load all German sapflow data files"""
        sapf_files = list(self.data_dir.glob("DEU_*_sapf_data.csv"))
        print(f"Found {len(sapf_files)} German sapflow files")
        
        for file in sapf_files:
            parts = file.stem.split('_')
            location = parts[1]  # e.g., HIN, STE
            plant_type = parts[2]  # e.g., 2P3, 4P5
            
            try:
                df = pd.read_csv(file, parse_dates=['TIMESTAMP'])
                df = self._process_sapflow_data(df)
                
                if location not in self.sapflow_data:
                    self.sapflow_data[location] = {}
                self.sapflow_data[location][plant_type] = df
                
            except Exception as e:
                print(f"Error loading {file.name}: {str(e)}")
    
    def _process_sapflow_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process sapflow data"""
        df = df.set_index('TIMESTAMP')
        df = df.sort_index()
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='first')]
        return df
    
    def plot_individual_plants(self, location: str, plant_type: str, 
                             figsize=(12, 6), save_dir: str = None):
        """
        Create individual plots for each plant/tree
        
        Parameters:
        -----------
        location: str
            Location code (e.g., 'HIN', 'STE')
        plant_type: str
            Plant type code (e.g., '2P3', '4P5')
        figsize: tuple
            Size of each individual plot
        save_dir: str, optional
            Directory to save plots
        """
        if location not in self.sapflow_data or plant_type not in self.sapflow_data[location]:
            raise ValueError(f"No data found for {location}_{plant_type}")
        
        data = self.sapflow_data[location][plant_type].copy()
        plot_columns = [col for col in data.columns if col != 'solar_TIMESTAMP']
        
        if save_dir:
            save_path = Path(save_dir) / f"DEU_{location}_{plant_type}"
            save_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving plots to {save_path}")
        
        for column in plot_columns:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot single tree data
            ax.plot(data.index, data[column], alpha=0.8)
            
            # Customize plot
            ax.set_title(f'Sap Flow Time Series - {location} {plant_type}\nTree {column}', pad=20)
            ax.set_xlabel('Date')
            ax.set_ylabel('Sap Flow')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            # plt.tight_layout()
            
            if save_dir:
                filename = f"tree_{column}.png"
                fig.savefig(save_path / filename)
                plt.close(fig)
                print(f"Saved {filename}")
            else:
                plt.show()
    
    def plot_all_plants(self, figsize=(12, 6), save_dir: str = None):
        """Plot individual trees for all sites"""
        for location in self.sapflow_data:
            for plant_type in self.sapflow_data[location]:
                try:
                    print(f"\nProcessing DEU_{location}_{plant_type}")
                    self.plot_individual_plants(
                        location=location,
                        plant_type=plant_type,
                        figsize=figsize,
                        save_dir=save_dir
                    )
                except Exception as e:
                    print(f"Error plotting {location}_{plant_type}: {str(e)}")
    
    def get_summary(self, location: str, plant_type: str) -> Dict:
        """Get summary statistics for a specific site"""
        if location not in self.sapflow_data or plant_type not in self.sapflow_data[location]:
            raise ValueError(f"No data found for {location}_{plant_type}")
        
        data = self.sapflow_data[location][plant_type]
        plot_columns = [col for col in data.columns if col != 'solar_TIMESTAMP']
        
        summary = {
            'location': location,
            'plant_type': plant_type,
            'time_range': {
                'start': data.index.min().strftime('%Y-%m-%d'),
                'end': data.index.max().strftime('%Y-%m-%d'),
                'duration_days': (data.index.max() - data.index.min()).days
            },
            'trees': len(plot_columns),
            'measurements': len(data),
            'missing_data': (data[plot_columns].isna().sum() / len(data) * 100).mean()
        }
        
        return summary

    def plot_plant(self, location, plant_type, figsize=(12, 6), save_dir=None):
        data = self.sapflow_data[location][plant_type].copy()
        plot_columns = [col for col in data.columns if col != 'solar_TIMESTAMP']
        
        if save_dir:
            save_path = Path(save_dir) / f"DEU_{location}_{plant_type}"
            save_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving plots to {save_path}")
        
        for column in plot_columns:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot with line and markers
            ax.plot(data.index, data[column], 
                    marker='o',          # Add circular markers
                    linestyle='-',       # Solid line
                    markersize=3,        # Small markers
                    alpha=0.8)           # Slight transparency
            
            # Customize plot
            ax.set_title(f'Sap Flow Time Series - {location} {plant_type}\nTree {column}', pad=20)
            ax.set_xlabel('Date')
            ax.set_ylabel('Sap Flow')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            if save_dir:
                filename = f"tree_{column}.png"
                fig.savefig(save_path / filename, dpi = 1000, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved {filename}")
            else:
                plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = GermanSapFlowAnalyzer()
    
    # Print available sites and their basic info
    print("\nSite summaries:")
    print("-" * 50)
    for location in analyzer.sapflow_data:
        for plant_type in analyzer.sapflow_data[location]:
            summary = analyzer.get_summary(location, plant_type)
            print(f"\nDEU_{location}_{plant_type}")
            print(f"Period: {summary['time_range']['start']} to {summary['time_range']['end']}")
            print(f"Trees: {summary['trees']}")
            print(f"Missing data: {summary['missing_data']:.1f}%")
    
    # Plot individual plants for all sites
    print("\nGenerating individual plant plots...")
    analyzer.plot_all_plants(
        figsize=(100, 10),
        save_dir='german_sapflow_plots'
    )