"""Scoring functions for forward feature selection.

SFS selection uses mean-fold metrics via sklearn builtins:
  - 'r2' for mean-fold R2
  - 'neg_root_mean_squared_error' for mean-fold RMSE

True pooled R2/RMSE (concatenate all OOF predictions, then score) are
computed post-hoc by selector._compute_pooled_metrics() using
cross_val_predict.  They cannot be done inside cross_val_score because
make_scorer is called per-fold.
"""

from __future__ import annotations

from src.forward_selection.config import ScoringMode


def get_scorer(mode: ScoringMode | str) -> str:
    """Return a sklearn scoring string for mlxtend SFS.

    Parameters
    ----------
    mode : ScoringMode or str
        'mean_r2' or 'neg_rmse'.

    Returns
    -------
    str
        Built-in sklearn scorer name.
    """
    mode = ScoringMode(mode) if isinstance(mode, str) else mode

    if mode is ScoringMode.MEAN_R2:
        return "r2"
    if mode is ScoringMode.NEG_RMSE:
        return "neg_root_mean_squared_error"

    raise ValueError(f"Unknown scoring mode: {mode}")
