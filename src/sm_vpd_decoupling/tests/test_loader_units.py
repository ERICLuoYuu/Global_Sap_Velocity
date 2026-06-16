# src/sm_vpd_decoupling/tests/test_loader_units.py
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from src.sm_vpd_decoupling.loader import (
    NORM_QUANTILE,
    normalize_per_site,
    resolve_ppfd,
    root_zone_sm,
)


def test_root_zone_sm_weights():
    out = root_zone_sm(pd.Series([0.2]), pd.Series([0.3]), pd.Series([0.4]))
    assert math.isclose(out.iloc[0], 0.07 * 0.2 + 0.21 * 0.3 + 0.72 * 0.4, rel_tol=1e-12)


def test_resolve_ppfd_prefers_measured():
    df = pd.DataFrame(
        {
            "ppfd_in": [600.0, 700.0],
            "sw_in": [100.0, 100.0],
            "surface_solar_radiation_downwards_hourly": [50.0, 50.0],
        }
    )
    ppfd, source = resolve_ppfd(df)
    assert source == "ppfd_in"
    assert list(ppfd) == [600.0, 700.0]


def test_resolve_ppfd_falls_back_to_swin():
    df = pd.DataFrame(
        {
            "ppfd_in": [np.nan, np.nan],
            "sw_in": [300.0, 400.0],
            "surface_solar_radiation_downwards_hourly": [50.0, 50.0],
        }
    )
    ppfd, source = resolve_ppfd(df)
    assert source == "sw_in"
    assert math.isclose(ppfd.iloc[0], 300.0 * 2.04, rel_tol=1e-9)


def test_resolve_ppfd_falls_back_to_era5():
    df = pd.DataFrame(
        {
            "ppfd_in": [np.nan, np.nan],
            "sw_in": [np.nan, np.nan],
            "surface_solar_radiation_downwards_hourly": [250.0, 260.0],
        }
    )
    ppfd, source = resolve_ppfd(df)
    assert source == "era5_ssrd"
    assert math.isclose(ppfd.iloc[0], 250.0 * 2.04, rel_tol=1e-9)


def test_normalize_per_site_anchor():
    s = pd.Series(np.arange(1.0, 101.0))  # 1..100
    out = normalize_per_site(s)
    anchor = s[s >= s.quantile(NORM_QUANTILE)].mean()
    assert math.isclose(out.iloc[-1], 100.0 / anchor, rel_tol=1e-9)


def test_normalize_per_site_constant_series_is_nan_safe():
    s = pd.Series([5.0] * 10)
    out = normalize_per_site(s)
    # anchor == 5, so normalized == 1 everywhere (no div-by-zero / no NaN).
    assert (out == 1.0).all()
