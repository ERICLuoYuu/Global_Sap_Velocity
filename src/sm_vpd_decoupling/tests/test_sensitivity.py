# src/sm_vpd_decoupling/tests/test_sensitivity.py
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from src.sm_vpd_decoupling.sensitivity import sm_sensitivity


def test_sensitivity_linear_response():
    # response = 2.0 * SM ; SM and VPD independent.
    rng = np.random.default_rng(0)
    sm = rng.uniform(0.0, 1.0, 3000)
    vpd = rng.uniform(0.5, 3.0, 3000)
    df = pd.DataFrame({"sm": sm, "vpd": vpd, "resp": 2.0 * sm})
    # d(resp)/d(SM) = 2.0 ; per 0.1 m3/m3 -> 0.2.
    s = sm_sensitivity(df, response="resp", n_bins=5, vpd_col="vpd", sm_col="sm")
    assert math.isclose(s, 0.2, abs_tol=0.05)


def test_sensitivity_no_sm_effect_is_zero():
    rng = np.random.default_rng(1)
    sm = rng.uniform(0.0, 1.0, 3000)
    vpd = rng.uniform(0.5, 3.0, 3000)
    df = pd.DataFrame({"sm": sm, "vpd": vpd, "resp": (vpd - 0.5) / 2.5})
    s = sm_sensitivity(df, response="resp", n_bins=5, vpd_col="vpd", sm_col="sm")
    assert abs(s) < 0.03


def test_sensitivity_insufficient_returns_nan():
    df = pd.DataFrame({"sm": np.ones(100), "vpd": np.ones(100), "resp": np.ones(100)})
    s = sm_sensitivity(df, response="resp", n_bins=5, vpd_col="vpd", sm_col="sm")
    assert np.isnan(s)
