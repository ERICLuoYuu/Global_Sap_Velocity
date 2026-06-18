"""Tests for the Fu loader wrapper (Task 3): per-site z-scoring + cropland/wetland
exclusion on top of the sibling sm_vpd_decoupling.loader.load_table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.fu_ann_sensitivity.loader import (
    PREDICTORS,
    exclude_site_types,
    zscore_per_site,
)


def _toy_table():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "site_name": ["a"] * 50 + ["b"] * 50,
            "tair": np.r_[rng.normal(20, 3, 50), rng.normal(10, 2, 50)],
            "vpd": rng.normal(1.5, 0.4, 100),
            "sm": rng.normal(0.3, 0.05, 100),
            "ppfd": rng.normal(800, 100, 100),
            "E": rng.normal(5, 1, 100),
        }
    )


def test_zscore_per_site_unit_scale():
    z = zscore_per_site(_toy_table(), cols=PREDICTORS + ["E"], response="E")
    a = z[z.site_name == "a"]
    assert abs(a["tair_z"].mean()) < 1e-6
    assert abs(a["tair_z"].std(ddof=0) - 1.0) < 1e-6
    # z-score is per-site: site b (cooler) must also be centred independently.
    b = z[z.site_name == "b"]
    assert abs(b["tair_z"].mean()) < 1e-6
    assert "E_z" in z.columns


def test_zscore_zero_variance_is_nan():
    df = pd.DataFrame({"site_name": ["a"] * 5, "x": [3.0] * 5})
    z = zscore_per_site(df, cols=["x"], response="x")
    assert z["x_z"].isna().all()


def test_exclude_site_types_drops_wetland_and_cropland():
    df = pd.DataFrame(
        {
            "site_name": ["forest1", "wet1", "crop1", "forest2"],
            "pft": ["ENF", "WET", "CRO", "DBF"],
            "E": [1.0, 2.0, 3.0, 4.0],
        }
    )
    kept = exclude_site_types(df)
    assert set(kept["site_name"]) == {"forest1", "forest2"}
    # Opt-out keeps everything.
    assert len(exclude_site_types(df, exclude_pft=())) == 4


def test_exclude_site_types_noop_without_pft_column():
    df = pd.DataFrame({"site_name": ["a", "b"], "E": [1.0, 2.0]})
    assert len(exclude_site_types(df)) == 2
