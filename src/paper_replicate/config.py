"""Configuration for Loritz et al. (2024) replication.

All hyperparameters, feature lists, and paths match their published code
(Zenodo DOI 10.5281/zenodo.10118262) exactly.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np


def cyclical_encode(hour: float, max_hour: float = 24.0) -> float:
    """Encode hour using cyclical (circular) encoding.

    Matches their Cell 4 exactly: -cos(2π * hour/24).
    """
    hour_normalized = hour / max_hour
    return np.cos(2 * np.pi * hour_normalized) * -1


# --- Feature definitions (Cell 5) ---

USECOLS_ENV = ["TIMESTAMP", "ta", "precip", "rh", "sw_in", "ws", "vpd"]
USECOLS_EO = [
    "date",
    "leaf_area_index_high_vegetation",
    "leaf_area_index_low_vegetation",
]
USECOLS_SI = ["si_elev", "si_mat", "si_map", "si_igbp"]
USECOLS_PL = ["pl_code", "pl_dbh", "pl_species"]

FEATURES_ONEHOT = ["si_igbp", "pl_genus"]
FEATURES_WITHOUT_ONEHOT = [
    "ta",
    "precip",
    "rh",
    "sw_in",
    "ws",
    "vpd",
    "si_elev",
    "si_mat",
    "si_map",
    "pl_dbh",
    "leaf_area_index_high_vegetation",
    "leaf_area_index_low_vegetation",
]

PLANT_LIST = [
    "pl_genus_Quercus",
    "pl_genus_Pseudotsuga",
    "pl_genus_Pinus",
    "pl_genus_Picea",
    "pl_genus_Fagus",
    "pl_genus_Larix",
]

IGBP_TYPES = ["DBF", "DNF", "EBF", "ENF", "MF", "SAV"]

GENERA = ["Fagus", "Larix", "Picea", "Pinus", "Pseudotsuga", "Quercus"]

MONTH_LIST = [4, 5, 6, 7, 8, 9]  # April-September


@dataclass
class NetworkParams:
    """Network hyperparameters matching their Cell 10 exactly."""

    batch_size: int = 64
    # input_size is determined at runtime from len(features)
    no_of_layers: int = 1
    sequence_length: int = 24
    drop_last: bool = True
    hidden_size: int = 256
    no_of_epochs: int = 20
    drop_out: float = 0.4
    learning_rate: float = 0.0005
    weight_decay: float = 0.001
    set_forget_gate: float = 3.0
    grad_clip: bool = True
    max_norm: float = 1.0
    noise_std: float = 0.05


@dataclass
class DataSplitParams:
    """Data split parameters."""

    # Gauged: 50% train, 10% val, 40% test
    train_frac: float = 0.5
    val_frac: float = 0.6  # cumulative: train ends at 0.5, val ends at 0.6

    # Seeds
    torch_seed: int = 1
    cuda_seed: int = 1


@dataclass
class ReplicationConfig:
    """Top-level configuration for the replication."""

    network: NetworkParams = field(default_factory=NetworkParams)
    split: DataSplitParams = field(default_factory=DataSplitParams)

    # Paths — set at runtime based on HPC or local
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs/paper_replicate")

    # Number of Monte Carlo runs
    n_mc_runs: int = 10

    @property
    def model_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def results_dir(self) -> Path:
        return self.output_dir / "results"
