"""
Centralized Path Configuration Module
======================================

This module provides path configuration for all scripts in the project.
Import and use PathConfig across all your analysis scripts.

Usage:
    from path_config import PathConfig, get_default_paths
    
    # Use default configuration
    paths = get_default_paths()
    
    # Or customize
    paths = PathConfig(scale='site', base_data_dir='/my/data')
"""

from pathlib import Path
from typing import Optional, Dict
import os
import json


class PathConfig:
    """
    Centralized path configuration for all project scripts.
    
    This class provides consistent path management across:
    - Environmental data analysis
    - Sap flow data analysis
    - Data preprocessing
    - Visualization scripts
    - Model training/inference (ML & DL)
    - Hyperparameter optimization
    
    Attributes:
        scale (str): Analysis scale ('plant', 'site', etc.)
        base_data_dir (Path): Root directory for all input data
        base_output_dir (Path): Root directory for all outputs
    """
    
    def __init__(
        self, 
        scale: str = 'sapwood',
        base_data_dir: str = "./data",
        base_output_dir: str = "./outputs",
        data_version: str = "0.1.5"
    ):
        """
        Initialize path configuration.
        
        Args:
            scale: Analysis scale ('plant', 'site', 'inverter', etc.)
            base_data_dir: Root directory for input data
            base_output_dir: Root directory for outputs
            data_version: Version of the dataset
        """
        self.scale = scale
        self.data_version = data_version
        self.base_data_dir = Path(base_data_dir).resolve()
        self.base_output_dir = Path(base_output_dir).resolve()
        
        # =====================================================================
        # INPUT DATA PATHS
        # =====================================================================
        
        # Raw data directory structure
        self.raw_data_root = self.base_data_dir / "raw" / self.data_version / self.data_version
        self.raw_csv_dir = self.raw_data_root / "csv" / self.scale
        self.raw_parquet_dir = self.raw_data_root / "parquet" / self.scale

        # =====================================================================
        # RAW GRIDED DATA PATHS
        # =====================================================================
        self.raw_grided_root = self.base_data_dir / "raw" / "grided"
        self.globmap_lai_root = self.raw_grided_root / "globmap_lai"
        self.worldclim_data_dir = self.raw_grided_root / "worldclim"
        self.koppen_data_dir = self.raw_grided_root / "koppen_geiger"
        # =====================================================================
        # SITE INFO DATA PATHS
        # =====================================================================
        self.site_info_path = self.raw_csv_dir / "site_info.csv"
        
        # Specific data type directories (all point to raw_csv_dir)
        self.env_data_dir = self.raw_csv_dir  # Environmental data
        self.sap_data_dir = self.raw_csv_dir  # Sap flow data
        self.metadata_dir = self.raw_csv_dir  # Metadata files
        
        # =====================================================================
        # Remote sensing/extracted data paths
        # =====================================================================
        self.extracted_data_dir = self.base_data_dir / "raw" / "extracted_data"
        # Era5 extracted data
        self.era5_discrete_data_path = self.extracted_data_dir / "era5land_site_data" / self.scale /'era5_extracted_data.csv'
        # LAI extracted data
        self.globmap_lai_data_path = self.extracted_data_dir / "globmap_lai_site_data" / self.scale / 'extracted_globmap_lai_hourly.csv'
        self.pft_data_path = self.extracted_data_dir / "landcover_data" / self.scale / 'landcover_output.csv'

        # LAI extracted data
        self.lai_extracted_data_path = self.extracted_data_dir / "globmap_lai_site_data" / self.scale / 'extracted_globmap_lai.csv'
        # ENV extracted data
        self.env_extracted_data_dir = self.extracted_data_dir / "env_site_data" / self.scale 
        self.env_extracted_data_path = self.extracted_data_dir / "env_site_data" / self.scale / 'site_info_with_env_data.csv'
    
        self.terrain_attributes_data_path = self.extracted_data_dir / "terrain_site_data" / self.scale / 'site_info_with_terrain_data.csv'


        # =====================================================================
        # PROCESSED DATA PATHS
        # =====================================================================
        
        self.processed_root = self.base_output_dir / "processed_data"
        
        # Environmental data processing
        self.env_processed_root = self.processed_root / self.scale / "env"
        self.env_outliers_removed_dir = self.env_processed_root / "outliers_removed"
        self.env_outliers_dir = self.env_processed_root / "outliers"
        self.env_timezone_adjusted_dir = self.env_processed_root / "timezone_adjusted"
        self.env_standardized_dir = self.env_processed_root / "standardized"
        self.env_cleaned_dir = self.env_processed_root / "cleaned"
        self.env_imputed_dir = self.env_processed_root / "imputed"
        self.env_filtered_dir = self.env_processed_root / "filtered"
        self.env_flagged_dir = self.env_processed_root / "flagged"
        self.env_day_masks_dir = self.env_processed_root / "day_masks"
        self.env_daily_resampled_dir = self.env_processed_root / "daily"
        
        # Sap flow data processing
        self.sap_processed_root = self.processed_root / self.scale / "sap"
        self.sap_outliers_removed_dir = self.sap_processed_root / "outliers_removed"
        self.sap_variability_filtered_dir = self.sap_processed_root / "variability_filtered"
        self.sap_variability_masks_dir = self.sap_processed_root / "variability_masks"
        self.sap_reversed_dir = self.sap_processed_root / "reversed"
        self.sap_baseline_drift_corrected_dir = self.sap_processed_root / "baseline_drift_corrected"
        self.sap_outliers_dir = self.sap_processed_root / "outliers"
        self.sap_duplicates_dir = self.sap_processed_root / "duplicates"
        self.sap_timezone_adjusted_dir = self.sap_processed_root / "timezone_adjusted"
        self.sap_standardized_dir = self.sap_processed_root / "standardized"
        self.sap_filtered_dir = self.sap_processed_root / "filtered"
        self.sap_day_masks_dir = self.sap_processed_root / "day_masks"
        self.sap_daily_resampled_dir = self.sap_processed_root / "daily"
        
        # Merged/combined data (site level, hourly)
        self.merged_data_root = self.processed_root / self.scale / "merged"
        # self.merged_site_hourly_after_outlier_removal_dir = self.merged_data_root / "hourly_after_outlier_removal"
        self.merged_daily_dir = self.merged_data_root / "daily"

        
       
    
        
        # Train/test splits
        self.train_test_dir = self.processed_root / "train_test"
        self.train_dir = self.train_test_dir / "train"
        self.test_dir = self.train_test_dir / "test"
        self.validation_dir = self.train_test_dir / "validation"
        
        # =====================================================================
        # FIGURE/VISUALIZATION PATHS
        # =====================================================================
        
        self.figures_root = self.base_output_dir / "figures"
        
        # Environmental figures
        self.env_figures_root = self.figures_root / "env"
        self.env_raw_figures_dir = self.env_figures_root / "raw"
        self.env_cleaned_figures_dir = self.env_figures_root / "cleaned"
        self.env_analysis_figures_dir = self.env_figures_root / "analysis"
        self.env_correlation_figures_dir = self.env_figures_root / "correlation"
        
        
        # Sap flow figures
        self.sap_figures_root = self.figures_root / "sap"
        self.sap_cleaned_figures_dir = self.sap_figures_root / "cleaned"
 
        
        # Combined/comparison figures
        self.combined_figures_dir = self.figures_root / "combined"
        self.comparison_figures_dir = self.figures_root / "comparison"
        
        # EDA figures
        self.eda_figures_dir = self.figures_root / "eda"
        
        # Model training plots (directly under plots/)
        self.plots_root = self.base_output_dir / "plots"
        self.hyper_tuning_plots_dir = self.plots_root / "hyperparameter_optimization"
        
        # =====================================================================
        # MODEL PATHS
        # =====================================================================
        
        self.models_root = Path("./models")  # Root level models directory
        self.output_models_root = self.base_output_dir / "models"  # Output models directory
        
        self.trained_models_dir = self.output_models_root / "trained"
        self.model_checkpoints_dir = self.output_models_root / "checkpoints"
        self.model_configs_dir = self.output_models_root / "configs"
        
        # Hyperparameter tuning
        self.hyper_tuning_root = self.base_output_dir / "hyper_tuning"
        self.hyper_tuning_tmp_dir = self.hyper_tuning_root / "tmp"

        # =====================================================================
        # SCALERS PATHS
        # =====================================================================

        self.scalers_root = self.base_output_dir / "scalers"
        self.feature_scaler_path = self.scalers_root / "feature_scaler.joblib"
        self.label_scaler_path = self.scalers_root / "label_scaler.joblib"

        # =====================================================================
        # REPORT/RESULTS PATHS
        # =====================================================================
        
        self.reports_root = self.base_output_dir / "reports"
        self.analysis_reports_dir = self.reports_root / "analysis"
        self.model_reports_dir = self.reports_root / "models"
        self.data_quality_reports_dir = self.reports_root / "data_quality"
        
        # Results/metrics
        self.results_root = self.base_output_dir / "results"
        self.metrics_dir = self.results_root / "metrics"
        self.predictions_dir = self.results_root / "predictions"
        self.evaluation_dir = self.results_root / "evaluation"
        self.cv_results_dir = self.results_root / "cross_validation"
        
        # =====================================================================
        # LOG PATHS
        # =====================================================================
        
        self.logs_root = self.base_output_dir / "logs"
        self.processing_logs_dir = self.logs_root / "processing"
        self.training_logs_dir = self.logs_root / "training"
        self.error_logs_dir = self.logs_root / "errors"
        self.optimization_logs_dir = self.logs_root / "optimization"
        
        # =====================================================================
        # CACHE PATHS
        # =====================================================================
        
        self.cache_root = self.base_output_dir / "cache"
        self.data_cache_dir = self.cache_root / "data"
        self.computation_cache_dir = self.cache_root / "computation"
    
    def create_all_directories(self):
        """Create all output directories if they don't exist."""
        directories = [
            # Processed data
            self.env_outliers_removed_dir,
            self.env_outliers_dir,
            self.env_timezone_adjusted_dir,
            self.env_standardized_dir,
            self.env_cleaned_dir,
            self.env_imputed_dir,
            self.sap_outliers_removed_dir,
            self.sap_outliers_dir,
            self.sap_timezone_adjusted_dir,
            self.sap_standardized_dir,
            self.sap_cleaned_dir,
            self.merged_site_hourly_dir,
            self.merged_plant_hourly_dir,
            self.merged_daily_dir,
            self.feature_engineered_dir,
            self.train_dir,
            self.test_dir,
            self.validation_dir,
            
            # Figures
            self.env_raw_figures_dir,
            self.env_cleaned_figures_dir,
            self.env_analysis_figures_dir,
            self.env_correlation_figures_dir,
            self.sap_raw_figures_dir,
            self.sap_cleaned_figures_dir,
            self.sap_analysis_figures_dir,
            self.combined_figures_dir,
            self.comparison_figures_dir,
            self.eda_figures_dir,
            
            # Plots (model training plots go directly under plots/)
            self.plots_root,
            
            # Models
            self.models_root,
            self.trained_models_dir,
            self.model_checkpoints_dir,
            self.model_configs_dir,
            self.hyper_tuning_tmp_dir,
            
            # Reports and results
            self.analysis_reports_dir,
            self.model_reports_dir,
            self.data_quality_reports_dir,
            self.metrics_dir,
            self.predictions_dir,
            self.evaluation_dir,
            self.cv_results_dir,
            
            # Logs
            self.processing_logs_dir,
            self.training_logs_dir,
            self.error_logs_dir,
            self.optimization_logs_dir,
            
            # Cache
            self.data_cache_dir,
            self.computation_cache_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Created {len(directories)} directories")
    
    def validate_input_paths(self):
        """Validate that required input directories exist."""
        required_paths = [
            (self.base_data_dir, "Base data directory"),
        ]
        
        optional_paths = [
            (self.raw_data_root, "Raw data root"),
            (self.raw_csv_dir, "Raw CSV directory"),
        ]
        
        missing_required = []
        missing_optional = []
        
        for path, description in required_paths:
            if not path.exists():
                missing_required.append(f"{description}: {path}")
        
        for path, description in optional_paths:
            if not path.exists():
                missing_optional.append(f"{description}: {path}")
        
        if missing_required:
            error_msg = "Missing required directories:\n  - " + "\n  - ".join(missing_required)
            raise ValueError(error_msg)
        
        if missing_optional:
            print("⚠ Warning: Some optional directories don't exist:")
            for missing in missing_optional:
                print(f"  - {missing}")
        
        print(f"✓ Input paths validated")
        print(f"  - Base data directory: {self.base_data_dir}")
    
    def get_model_save_dir(self, model_type: str, task: str = 'regression') -> Path:
        """
        Get model save directory with standardized naming.
        
        Args:
            model_type: Type of model (e.g., 'lstm', 'random_forest')
            task: Task type (e.g., 'regression', 'classification')
            
        Returns:
            Path to model save directory
        """
        model_dir = self.output_models_root / f"{model_type}_{task}"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    def get_plot_dir(self, model_type: str) -> Path:
        """
        Get plot directory for model training.
        
        Args:
            model_type: Type of model
            
        Returns:
            Path to plot directory (./outputs/plots/{model_type}/)
        """
        plot_dir = self.plots_root / model_type
        plot_dir.mkdir(parents=True, exist_ok=True)
        return plot_dir
    
    def get_log_file(self, logger_name: str) -> Path:
        """
        Get log file path for a specific logger.
        
        Args:
            logger_name: Name of the logger
            
        Returns:
            Path to log file
        """
        self.logs_root.mkdir(parents=True, exist_ok=True)
        return self.logs_root / f"{logger_name}_optimizer.log"
    
    def get_env_file_pattern(self, file_type: str = "data") -> str:
        """
        Get file pattern for environmental data files.
        
        Args:
            file_type: Type of file ('data', 'flags', 'md')
            
        Returns:
            Glob pattern string
        """
        return f"*_env_{file_type}.csv"
    
    def get_sap_file_pattern(self, file_type: str = "data") -> str:
        """
        Get file pattern for sap flow data files.
        
        Args:
            file_type: Type of file ('data', 'flags', 'md')
            
        Returns:
            Glob pattern string
        """
        return f"*_sapf_{file_type}.csv"
    
    def to_dict(self) -> Dict:
        """Export configuration as dictionary."""
        return {
            'scale': self.scale,
            'data_version': self.data_version,
            'base_data_dir': str(self.base_data_dir),
            'base_output_dir': str(self.base_output_dir),
        }
    
    def save_config(self, filepath: Optional[Path] = None):
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to save config. If None, saves to outputs/config.json
        """
        if filepath is None:
            filepath = self.base_output_dir / "config.json"
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        print(f"✓ Configuration saved to {filepath}")
    
    @classmethod
    def from_config_file(cls, filepath: Path) -> 'PathConfig':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to config file
            
        Returns:
            PathConfig instance
        """
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        return cls(**config)
    
    def __repr__(self):
        return (f"PathConfig(scale='{self.scale}', "
                f"base_data_dir='{self.base_data_dir}', "
                f"base_output_dir='{self.base_output_dir}')")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_default_paths(scale: str = 'sapwood') -> PathConfig:
    """
    Get default path configuration.
    
    Args:
        scale: Analysis scale
        
    Returns:
        PathConfig instance with default settings
    """
    return PathConfig(scale=scale)


def get_paths_from_env() -> PathConfig:
    """
    Get path configuration from environment variables.
    
    Environment variables:
        ANALYSIS_SCALE: Analysis scale (default: 'plant')
        DATA_DIR: Base data directory (default: './data')
        OUTPUT_DIR: Base output directory (default: './outputs')
        DATA_VERSION: Data version (default: '0.1.5')
    
    Returns:
        PathConfig instance configured from environment
    """
    return PathConfig(
        scale=os.getenv('ANALYSIS_SCALE', 'plant'),
        base_data_dir=os.getenv('DATA_DIR', './data'),
        base_output_dir=os.getenv('OUTPUT_DIR', './outputs'),
        data_version=os.getenv('DATA_VERSION', '0.1.5')
    )


def setup_project_paths(scale: str = 'plant', create_dirs: bool = True) -> PathConfig:
    """
    Setup project with path configuration.
    
    Args:
        scale: Analysis scale
        create_dirs: Whether to create all directories
        
    Returns:
        Configured PathConfig instance
    """
    paths = PathConfig(scale=scale)
    
    try:
        paths.validate_input_paths()
    except ValueError as e:
        print(f"Warning: {e}")
        print("Some input directories are missing. Please ensure data is in the correct location.")
    
    if create_dirs:
        paths.create_all_directories()
    
    return paths


# =============================================================================
# ENVIRONMENT PRESETS
# =============================================================================

def get_development_paths() -> PathConfig:
    """Get paths configured for development environment."""
    return PathConfig(
        scale='plant',
        base_data_dir='./data',
        base_output_dir='./dev_outputs'
    )


def get_production_paths() -> PathConfig:
    """Get paths configured for production environment."""
    return PathConfig(
        scale='plant',
        base_data_dir='/data/production',
        base_output_dir='/outputs/production'
    )


def get_testing_paths() -> PathConfig:
    """Get paths configured for testing environment."""
    return PathConfig(
        scale='plant',
        base_data_dir='./tests/data',
        base_output_dir='./tests/outputs'
    )


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("Path Configuration Module")
    print("=" * 80)
    
    # Create default configuration
    paths = get_default_paths()
    print(f"\n{paths}")
    
    # Show some key paths
    print(f"\nKey Paths:")
    print(f"  Raw data: {paths.raw_csv_dir}")
    print(f"  Env processed: {paths.env_processed_root}")
    print(f"  Sap processed: {paths.sap_processed_root}")
    print(f"  Merged data: {paths.merged_site_hourly_dir}")
    print(f"  Figures: {paths.figures_root}")
    print(f"  Plots: {paths.plots_root}")
    print(f"  Models: {paths.models_root}")
    print(f"  Logs: {paths.logs_root}")
    
    # Create directories
    print("\nCreating directories...")
    paths.create_all_directories()
    
    print("\n✓ Path configuration ready!")