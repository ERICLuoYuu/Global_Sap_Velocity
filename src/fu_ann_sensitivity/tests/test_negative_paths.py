"""Phase 5 Round 2 - negative paths & error handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fu_ann_sensitivity.aggregate import aggregate_at_nbins, fit_site_sensitivities
from src.fu_ann_sensitivity.loader import load_table
from src.fu_ann_sensitivity.plotting import plot_pft_panels
from src.fu_ann_sensitivity.sensitivity import site_row_sensitivities

PREDICTORS_Z = ["tair_z", "vpd_z", "sm_z", "ppfd_z"]


def _zscored_multisite_table():
    rng = np.random.default_rng(1)
    frames = []
    for site in ("a", "b", "c"):
        n = 200
        frames.append(
            pd.DataFrame(
                {
                    "site_name": [site] * n,
                    "pft": ["ENF"] * n,
                    "tair_z": rng.normal(size=n),
                    "vpd_z": rng.normal(size=n),
                    "sm_z": rng.normal(size=n),
                    "ppfd_z": rng.normal(size=n),
                    "E_z": rng.normal(size=n),  # pure noise -> low r everywhere
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_aggregate_empty_per_site_raises():
    with pytest.raises(ValueError):
        aggregate_at_nbins([], n_bins=5)


def test_fit_drops_all_sites_when_threshold_unreachable():
    # r can never reach 1.1 -> every trained site is dropped.
    out = fit_site_sensitivities(
        _zscored_multisite_table(),
        response="E_z",
        sm_col="sm_z",
        vpd_col="vpd_z",
        predictors=PREDICTORS_Z,
        min_valid_days=20,
        n_repeats=2,
        r_threshold=1.1,
    )
    assert out.per_site == []
    assert out.n_dropped_r == 3
    assert out.n_trained == 3


def test_load_table_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_table(str(tmp_path / "does_not_exist"))


def test_site_row_sensitivities_unknown_sm_col_raises():
    df = pd.DataFrame({"tair": [1.0, 2.0], "vpd": [1.0, 2.0], "sm": [1.0, 2.0], "ppfd": [1.0, 2.0]})
    # bad sm_col is rejected (predictors.index) before any model is used.
    with pytest.raises(ValueError):
        site_row_sensitivities([], df, ["tair", "vpd", "sm", "ppfd"], sm_col="NOPE")


def test_plot_pft_panels_empty_dict_writes_png(tmp_path):
    out = tmp_path / "pft_empty.png"
    plot_pft_panels({}, "E", out)
    assert out.exists() and out.stat().st_size > 0
