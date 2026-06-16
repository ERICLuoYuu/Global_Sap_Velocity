# src/sm_vpd_decoupling/tests/test_aggregate.py
from __future__ import annotations

import numpy as np
import pandas as pd
from src.sm_vpd_decoupling.aggregate import decouple_all_sites, dominance_summary


def _table(n_per_site=400, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for _, site in enumerate(["A", "B", "C"]):
        sm = rng.uniform(0.1, 0.35, n_per_site)
        vpd = rng.uniform(0.5, 2.5, n_per_site)
        resp = (sm - 0.1) / 0.25  # SM-dominant, normalized-ish
        frames.append(
            pd.DataFrame(
                {
                    "site_name": site,
                    "vpd": vpd,
                    "swvl1": sm,
                    "E_norm": resp,
                    "biome": "temperate",
                    "pft": "ENF",
                    "aridity": 0.7,
                    "canopy_height": 18.0,
                    "lat": 46.0,
                    "lon": 8.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_decouple_all_sites_columns_and_filter():
    table = _table()
    out = decouple_all_sites(table, sm_col="swvl1", response="E_norm", n_bins=5, min_valid_days=120)
    assert set(
        ["site_name", "n_days", "sm_given_vpd", "vpd_given_sm", "sensitivity", "biome", "pft", "aridity"]
    ).issubset(out.columns)
    assert len(out) == 3  # all sites have >= 120 days
    # SM-dominant synthetic -> |sm_given_vpd| > |vpd_given_sm| for every site.
    assert (out["sm_given_vpd"].abs() > out["vpd_given_sm"].abs()).all()


def test_decouple_all_sites_min_valid_days():
    table = _table(n_per_site=50)
    out = decouple_all_sites(table, sm_col="swvl1", response="E_norm", n_bins=5, min_valid_days=120)
    assert len(out) == 0  # no site reaches 120 days


def test_dominance_summary():
    effects = pd.DataFrame(
        {
            "sm_given_vpd": [-0.4, -0.3, -0.1],
            "vpd_given_sm": [0.1, 0.2, 0.5],
        }
    )
    pct, n = dominance_summary(effects)
    # SM wins in 2 of 3 valid sites.
    assert n == 3
    assert abs(pct - (2 / 3 * 100)) < 1e-9
