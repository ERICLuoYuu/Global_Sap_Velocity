"""Configuration dataclasses for forward feature selection."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ScoringMode(str, Enum):
    """Scoring metric used during SFS selection (mean-fold).

    True pooled R2/RMSE are always computed post-hoc after selection
    completes — see selector._compute_pooled_metrics().
    """

    MEAN_R2 = "mean_r2"
    NEG_RMSE = "neg_rmse"


@dataclass(frozen=True)
class HyperparamConfig:
    """Fixed XGBoost hyperparameters for forward selection.

    These are the best hyperparameters from previous optimization runs.
    Fixed during selection to avoid prohibitive re-tuning at each step.
    """

    n_estimators: int = 1000
    learning_rate: float = 0.01
    max_depth: int = 10
    min_child_weight: int = 7
    subsample: float = 0.67
    colsample_bytree: float = 0.7
    gamma: float = 0.3
    reg_alpha: float = 1.0
    reg_lambda: float = 10.0

    def to_xgb_params(self, n_jobs: int = 4, random_state: int = 42) -> dict:
        """Convert to XGBRegressor keyword arguments."""
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbosity": 0,
        }


@dataclass(frozen=True)
class SelectionConfig:
    """Configuration for a forward feature selection run."""

    scoring: ScoringMode
    n_splits: int = 10
    random_seed: int = 42
    n_jobs_sfs: int = 32
    n_jobs_xgb: int = 4
    k_features: str = "best"
    forward: bool = True
    floating: bool = False
    is_only_day: bool = False
    time_scale: str = "daily"
    cache_dir: Path = field(default_factory=lambda: Path("outputs/forward_selection"))
    output_dir: Path = field(default_factory=lambda: Path("outputs/forward_selection"))
