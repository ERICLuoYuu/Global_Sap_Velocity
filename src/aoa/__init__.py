"""
Area of Applicability (AOA) module following Meyer & Pebesma (2021).

Quantifies how similar new prediction locations are to the training data
in feature space, using the Dissimilarity Index (DI) and a threshold
derived from cross-validation.
"""

from src.aoa.aoa import compute_aoa
from src.aoa.model_bridge import (
    load_aoa_arrays,
    load_model_config,
    load_shap_weights,
    reconstruct_fold_indices,
    save_aoa_arrays,
)
from src.aoa.plotting import plot_aoa_map, plot_di_histogram
from src.aoa.run_aoa import run_aoa

__all__ = [
    "compute_aoa",
    "load_aoa_arrays",
    "load_model_config",
    "load_shap_weights",
    "reconstruct_fold_indices",
    "save_aoa_arrays",
    "plot_aoa_map",
    "plot_di_histogram",
    "run_aoa",
]
