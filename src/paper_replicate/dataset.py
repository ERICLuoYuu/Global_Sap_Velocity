"""Sequence dataset for Loritz et al. (2024) replication.

Faithful port of their Cell 11 SequenceDataset.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import FEATURES_WITHOUT_ONEHOT


class SequenceDataset(Dataset):
    """PyTorch dataset producing 24-hour windowed sequences.

    Ports their Cell 11 exactly. Key behaviors:
    - NaN filled with 0
    - Non-one-hot features standardized with global mean/std
    - Target optionally standardized with non-zero sapf mean/std
    - Sequences shorter than seq_len are padded from earlier data
    """

    def __init__(
        self,
        df_pl: pd.DataFrame,
        target_var: str,
        features_var: List[str],
        sequence_length: int,
        standardize: bool,
        all_mean: pd.Series,
        all_std: pd.Series,
        sapf_mean: float = 0.0,
        sapf_std: float = 1.0,
    ):
        self.target = target_var
        self.features = features_var
        self.sequence_length = sequence_length

        # Clean NaN -> 0
        df_pl = df_pl.fillna(0)

        # Convert to float64
        df_pl[self.features] = df_pl[self.features].astype(np.float64)
        df_pl[self.target] = df_pl[self.target].astype(np.float64)

        # Standardize non-one-hot features with global stats
        cols_to_standardize = [
            c for c in FEATURES_WITHOUT_ONEHOT if c in df_pl.columns
        ]
        df_pl[cols_to_standardize] = (
            df_pl[cols_to_standardize] - all_mean[cols_to_standardize]
        ) / all_std[cols_to_standardize]

        # Standardize target if specified
        if standardize:
            df_pl[target_var] = (df_pl[target_var] - sapf_mean) / sapf_std

        # Build tensors
        self.X = torch.tensor(df_pl[self.features].values)
        self.y = torch.tensor(df_pl[self.target].values)

        # Offset by sequence_length - 1 (their code: values[(seq_len-1):])
        self.X_out = torch.tensor(
            df_pl[self.features].values[(self.sequence_length - 1) :]
        )
        self.y_out = torch.tensor(
            df_pl[self.target].values[(self.sequence_length - 1) :]
        )

    def __len__(self) -> int:
        return self.X_out.shape[0]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X_out[i_start : (i + 1), :]
        else:
            # Pad sequences shorter than sequence_length
            padding = self.X[i : (self.sequence_length - 1)]
            x = self.X_out[0 : (i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y_out[i]
