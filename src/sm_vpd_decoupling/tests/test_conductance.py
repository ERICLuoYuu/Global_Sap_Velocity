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
    # Hand-computed reference (NOT echoing the implementation): T=20C, VPD=1 kPa,
    # h=0, sap_velocity=1 cm3 cm-2 h-1.
    #   sfd   = 1e-3*1e4/3600           = 2.777778e-3 kg m-2 s-1
    #   K_G   = 115.8 + 0.4236*20       = 124.272 kPa m3 kg-1
    #   n_air = 44.6 * 273/(273+20)     = 41.5556 mol m-3   (sea level, h=0)
    #   Gc    = 124.272 * 2.777778e-3 * 41.5556 = 14.345 mol m-2 s-1
    got = canopy_conductance(1.0, 20.0, 1.0, 0.0)
    assert math.isclose(got, 14.345, rel_tol=1e-3)


def test_canopy_conductance_decreases_with_elevation():
    # Physical check (independent of the formula): lower barometric pressure at
    # altitude -> lower molar air density -> LOWER molar conductance. A 2000 m
    # site must give a smaller Gc than an otherwise identical sea-level site.
    g_sea = canopy_conductance(1.0, 20.0, 1.0, 0.0)
    g_alt = canopy_conductance(1.0, 20.0, 1.0, 2000.0)
    assert g_alt < g_sea
    # magnitude: exp(-0.00012*2000) = exp(-0.24) ~= 0.7866
    assert math.isclose(g_alt / g_sea, math.exp(-0.24), rel_tol=1e-6)


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
