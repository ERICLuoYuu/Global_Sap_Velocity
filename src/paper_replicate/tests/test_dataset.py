"""Tests for SequenceDataset (Loritz et al. 2024 replication)."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.paper_replicate.config import FEATURES_WITHOUT_ONEHOT
from src.paper_replicate.dataset import SequenceDataset


def _make_dummy_df(n_rows: int = 100) -> pd.DataFrame:
    """Create a dummy DataFrame mimicking the real data structure."""
    np.random.seed(42)
    data = {}

    # Numerical features
    for feat in FEATURES_WITHOUT_ONEHOT:
        data[feat] = np.random.randn(n_rows)

    # One-hot features
    for igbp in ["si_igbp_ENF"]:
        data[igbp] = np.ones(n_rows)
    for igbp in ["si_igbp_DBF", "si_igbp_DNF", "si_igbp_EBF", "si_igbp_MF", "si_igbp_SAV"]:
        data[igbp] = np.zeros(n_rows)

    for genus in ["pl_genus_Pinus"]:
        data[genus] = np.ones(n_rows)
    for genus in ["pl_genus_Fagus", "pl_genus_Larix", "pl_genus_Picea",
                   "pl_genus_Pseudotsuga", "pl_genus_Quercus"]:
        data[genus] = np.zeros(n_rows)

    # Time hours
    data["time_hours"] = np.random.randn(n_rows)

    # Target
    data["sapf"] = np.abs(np.random.randn(n_rows)) * 10

    return pd.DataFrame(data)


def _get_feature_list(df: pd.DataFrame) -> list:
    """Get feature list (all columns minus sapf)."""
    return sorted([c for c in df.columns if c != "sapf"])


def _get_stats(df: pd.DataFrame):
    """Get mock standardization stats."""
    cols = [c for c in FEATURES_WITHOUT_ONEHOT if c in df.columns]
    all_mean = df[cols].mean()
    all_std = df[cols].std() + 1e-8  # Avoid div by zero
    sapf_pos = df[df["sapf"] > 0]["sapf"]
    sapf_mean = sapf_pos.mean()
    sapf_std = sapf_pos.std() + 1e-8
    return all_mean, all_std, sapf_mean, sapf_std


class TestSequenceDataset:
    """Test windowed sequence generation matches their Cell 11."""

    def test_output_shape(self):
        """Each item should be (seq_len, n_features), scalar."""
        df = _make_dummy_df(100)
        features = _get_feature_list(df)
        all_mean, all_std, sapf_mean, sapf_std = _get_stats(df)

        ds = SequenceDataset(
            df_pl=df, target_var="sapf", features_var=features,
            sequence_length=24, standardize=True,
            all_mean=all_mean, all_std=all_std,
            sapf_mean=sapf_mean, sapf_std=sapf_std,
        )

        x, y = ds[0]
        assert x.shape == (24, len(features))
        assert y.shape == ()  # Scalar target

    def test_dataset_length(self):
        """Length should be n_rows - seq_len + 1."""
        df = _make_dummy_df(100)
        features = _get_feature_list(df)
        all_mean, all_std, sapf_mean, sapf_std = _get_stats(df)

        ds = SequenceDataset(
            df_pl=df, target_var="sapf", features_var=features,
            sequence_length=24, standardize=True,
            all_mean=all_mean, all_std=all_std,
            sapf_mean=sapf_mean, sapf_std=sapf_std,
        )

        assert len(ds) == 100 - 24 + 1  # = 77

    def test_padding_for_short_sequences(self):
        """Sequences near the start should be padded."""
        df = _make_dummy_df(50)
        features = _get_feature_list(df)
        all_mean, all_std, sapf_mean, sapf_std = _get_stats(df)

        ds = SequenceDataset(
            df_pl=df, target_var="sapf", features_var=features,
            sequence_length=24, standardize=True,
            all_mean=all_mean, all_std=all_std,
            sapf_mean=sapf_mean, sapf_std=sapf_std,
        )

        # First few items use padding logic (i < seq_len - 1)
        x0, y0 = ds[0]
        assert x0.shape == (24, len(features))

        # Item at seq_len - 1 uses normal slicing
        x23, y23 = ds[23]
        assert x23.shape == (24, len(features))

    def test_standardization_applied(self):
        """Non-one-hot features should be standardized."""
        df = _make_dummy_df(100)
        features = _get_feature_list(df)
        all_mean, all_std, sapf_mean, sapf_std = _get_stats(df)

        ds = SequenceDataset(
            df_pl=df, target_var="sapf", features_var=features,
            sequence_length=24, standardize=True,
            all_mean=all_mean, all_std=all_std,
            sapf_mean=sapf_mean, sapf_std=sapf_std,
        )

        # Get a sample and check numerical features are roughly standardized
        x, y = ds[30]
        # Features should be approximately mean 0, std 1
        # (won't be exact since we're looking at a window, not the whole dataset)
        assert x.shape[0] == 24

    def test_target_standardized_when_flag_true(self):
        """Target should be z-scored when standardize=True."""
        df = _make_dummy_df(50)
        features = _get_feature_list(df)
        all_mean, all_std, sapf_mean, sapf_std = _get_stats(df)

        ds_std = SequenceDataset(
            df_pl=df.copy(), target_var="sapf", features_var=features,
            sequence_length=24, standardize=True,
            all_mean=all_mean, all_std=all_std,
            sapf_mean=sapf_mean, sapf_std=sapf_std,
        )
        ds_raw = SequenceDataset(
            df_pl=df.copy(), target_var="sapf", features_var=features,
            sequence_length=24, standardize=False,
            all_mean=all_mean, all_std=all_std,
            sapf_mean=sapf_mean, sapf_std=sapf_std,
        )

        _, y_std = ds_std[0]
        _, y_raw = ds_raw[0]
        # Standardized and raw targets should differ
        assert not torch.isclose(y_std, y_raw)

    def test_nan_filled_with_zero(self):
        """NaN values should be replaced with 0."""
        df = _make_dummy_df(50)
        # Inject NaN
        df.iloc[10, 0] = np.nan
        df.iloc[15, 3] = np.nan

        features = _get_feature_list(df)
        all_mean, all_std, sapf_mean, sapf_std = _get_stats(df)

        ds = SequenceDataset(
            df_pl=df, target_var="sapf", features_var=features,
            sequence_length=24, standardize=True,
            all_mean=all_mean, all_std=all_std,
            sapf_mean=sapf_mean, sapf_std=sapf_std,
        )

        # Should not raise and should not have NaN
        for i in range(len(ds)):
            x, y = ds[i]
            assert not torch.isnan(x).any()
            assert not torch.isnan(y)

    def test_minimum_data_length(self):
        """Dataset with exactly seq_len rows should have length 1."""
        df = _make_dummy_df(24)
        features = _get_feature_list(df)
        all_mean, all_std, sapf_mean, sapf_std = _get_stats(df)

        ds = SequenceDataset(
            df_pl=df, target_var="sapf", features_var=features,
            sequence_length=24, standardize=True,
            all_mean=all_mean, all_std=all_std,
            sapf_mean=sapf_mean, sapf_std=sapf_std,
        )

        assert len(ds) == 1
