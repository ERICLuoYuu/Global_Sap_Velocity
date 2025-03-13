import sys
from pathlib import Path
# Add parent directory to Python path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import time
from src.Analyzers import sap_analyzer_parallel
from src.Analyzers import env_analyzer_parallel


   

# Example usage
if __name__ == "__main__":
    # Start the timer
    start_time = time.time()
    # Initialize analyzer
    # analyzer = sap_analyzer_parallel.SapFlowAnalyzer(max_workers=8)
    env_analyzer_parallel = env_analyzer_parallel.EnvironmentalAnalyzer(max_workers=8)
    
    """
    # Print available sites and their basic info
    print("\nSite summaries:")
    print("-" * 50)
    
    for location in analyzer.sapflow_data:
        for plant_type in analyzer.sapflow_data[location]:
            summary = analyzer.get_summary(location, plant_type)
            print(f"\n{location}_{plant_type}")
            print(f"Period: {summary['time_range']['start']} to {summary['time_range']['end']}")
            print(f"Trees: {summary['trees']}")
            print(f"Missing data: {summary['missing_data']:.1f}%")
    
    
    print("\nGenerating individual plant plots...")
    # analyzer.plot_histogram_parallel(save_dir='./outputs/figures/sap/histograms', max_workers=4)
    
    summary = analyzer.plot_all_plants(
        figsize=(15, 8),
        save_dir='./outputs/figures/sap/cleaned',
        skip_empty=True,
        plot_limit=20,  # limit plots per location
        progress_update=True
    )
    
    # Print summary
    print(summary)
    
    end_time = time.time()
    
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")  
    """
    
    # plot environmental data
    # Plot all environmental variables with customization
    # env_analyzer_parallel.plot_histogram_parallel(save_dir='./outputs/figures/env/histograms')
    '''
    summary = env_analyzer_parallel.plot_all_parallel(
        figsize=(15, 8),
        save_dir='./outputs/figures/env/cleaned',
        skip_empty=True,
        plot_limit=10,  # limit plots per location
        progress_update=True
    )
    
    print(summary)
    '''
    '''
    # Collect all DataFrames
    all_dfs = []
    for location in env_analyzer_parallel.env_data:
        for plant_type in env_analyzer_parallel.env_data[location]:
            df = env_analyzer_parallel.env_data[location][plant_type]
            all_dfs.append(df)
            print(f"Added DataFrame for {location}_{plant_type}")
    
    # Find common columns
    common_cols = env_analyzer_parallel.get_common_columns(all_dfs)
    '''