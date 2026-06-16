# src/sm_vpd_decoupling/tests/test_conductance.py
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from src.sm_vpd_decoupling.conductance import (
    ETA,
    T0_K,
    canopy_conductance,
    sfd_to_kg_m2_s,
)


def test_sfd_unit_conversion():
    # 1 cm3 cm-2 h-1 of water = 1e-3 * 1e4 / 3600 kg m-2 s-1
    assert sfd_to_kg_m2_s(1.0) == np.float64(1e-3 * 1e4 / 3600)
    assert sfd_to_kg_m2_s(0.0) == 0.0


def test_canopy_conductance_matches_flo_eqn2():
    # Independently recompute Flo 2021 Eqn 2 for T=20C, VPD=1 kPa, h=0,
    # sap_velocity=1 cm3 cm-2 h-1.
    t, vpd, h, sv = 20.0, 1.0, 0.0, 1.0
    sfd = sfd_to_kg_m2_s(sv)
    expected = (115.8 + 0.4236 * t) * (sfd / vpd) * (ETA * T0_K / (T0_K + t)) * math.exp(0.00012 * h)
    got = canopy_conductance(sv, t, vpd, h)
    assert math.isclose(got, expected, rel_tol=1e-9)


def test_canopy_conductance_inverse_vpd_property():
    # Gc proportional to 1/VPD: doubling VPD halves Gc (documents the confound).
    g1 = canopy_conductance(1.0, 20.0, 1.0, 0.0)
    g2 = canopy_conductance(1.0, 20.0, 2.0, 0.0)
    assert math.isclose(g2, g1 / 2.0, rel_tol=1e-9)


def test_canopy_conductance_vectorized():
    sv = pd.Series([1.0, 2.0, np.nan])
    t = pd.Series([20.0, 20.0, 20.0])
    vpd = pd.Series([1.0, 1.0, 1.0])
    h = pd.Series([0.0, 0.0, 0.0])
    out = canopy_conductance(sv, t, vpd, h)
    assert isinstance(out, pd.Series)
    assert math.isclose(out.iloc[1], 2 * out.iloc[0], rel_tol=1e-9)
    assert np.isnan(out.iloc[2])


def test_canopy_conductance_zero_vpd_is_nan():
    # VPD=0 would divide by zero; must return NaN, not inf.
    assert np.isnan(canopy_conductance(1.0, 20.0, 0.0, 0.0))
