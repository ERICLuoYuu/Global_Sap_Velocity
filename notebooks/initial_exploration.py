import sys
from pathlib import Path
# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.Analyzers import sap_analyzer


   

# Example usage
if __name__ == "__main__":
 
    # Initialize analyzer
    analyzer = sap_analyzer.GermanSapFlowAnalyzer()
    
    
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
        figsize=(20, 10),
        save_dir='german_sapflow_plots'
    )
