"""Unit tests for decoupling — binned percentile driver isolation (Liu SI Text S2).

Key correctness test: when ΔSF depends ONLY on VPD (Tair orthogonal), the decoupling
must report a large ΔSF(VPD|Tair) and ~0 ΔSF(Tair|VPD). Plus the SM sign convention
(Eq. 2: low SM − high SM, because LOW soil moisture is the stressor).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.afternoon_depression.decoupling import (
    assign_deciles,
    decouple_effect,
    decouple_site,
)


# ── Per-site decile assignment ──────────────────────────────────────────────
@pytest.mark.unit
def test_assign_deciles_monotonic_to_ten_bins() -> None:
    vals = pd.Series(np.arange(100, dtype=float))
    bins = assign_deciles(vals, n_bins=10)
    assert bins.nunique() == 10
    assert bins.min() == 0 and bins.max() == 9
    # monotonic input → bin index is non-decreasing
    assert (bins.to_numpy() == np.sort(bins.to_numpy())).all()


@pytest.mark.unit
def test_assign_deciles_low_variance_does_not_crash() -> None:
    bins = assign_deciles(pd.Series([0.3] * 50), n_bins=10)
    # degenerate distribution collapses to a single (or few) bin, no exception
    assert bins.notna().any()


# ── decouple_effect: orthogonal-driver correctness (THE key test) ───────────
def _orthogonal_grid(driver_only: str) -> pd.DataFrame:
    """10x10 fully-populated grid of (vpd_bin, tair_bin); ΔSF depends only on `driver_only`."""
    rows = []
    for vb in range(10):
        for tb in range(10):
            delta = float(vb) if driver_only == "vpd" else float(tb)
            # several rows per cell so means are well defined
            for _ in range(3):
                rows.append({"vpd_bin": vb, "tair_bin": tb, "delta_sf": delta})
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_decouple_effect_isolates_vpd_when_only_vpd_matters() -> None:
    df = _orthogonal_grid("vpd")
    vpd_given_tair = decouple_effect(df, driver_bin="vpd_bin", cond_bin="tair_bin", response="delta_sf")
    tair_given_vpd = decouple_effect(df, driver_bin="tair_bin", cond_bin="vpd_bin", response="delta_sf")
    # within each Tair bin: high VPD (9) − low VPD (0) = 9
    assert vpd_given_tair == pytest.approx(9.0)
    # within each VPD bin ΔSF is constant → Tair effect ≈ 0
    assert tair_given_vpd == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_decouple_effect_isolates_tair_when_only_tair_matters() -> None:
    df = _orthogonal_grid("tair")
    vpd_given_tair = decouple_effect(df, driver_bin="vpd_bin", cond_bin="tair_bin", response="delta_sf")
    tair_given_vpd = decouple_effect(df, driver_bin="tair_bin", cond_bin="vpd_bin", response="delta_sf")
    assert tair_given_vpd == pytest.approx(9.0)
    assert vpd_given_tair == pytest.approx(0.0, abs=1e-9)


# ── SM sign convention: low SM − high SM (Eq. 2) ────────────────────────────
@pytest.mark.unit
def test_decouple_effect_sm_sign_low_minus_high_positive_when_dry_stresses() -> None:
    # ΔSF higher (more depression) when SM is LOW: delta = (9 - sm_bin)
    rows = []
    for vb in range(10):
        for sb in range(10):
            rows.append({"vpd_bin": vb, "sm_bin": sb, "delta_sf": float(9 - sb)})
    df = pd.DataFrame(rows)
    sm_given_vpd = decouple_effect(
        df, driver_bin="sm_bin", cond_bin="vpd_bin", response="delta_sf", low_minus_high=True
    )
    # low SM (0): delta 9 ; high SM (9): delta 0 → low − high = 9 (positive stress)
    assert sm_given_vpd == pytest.approx(9.0)


# ── Sparse / single-driver-bin conditioning bins are skipped, not errored ───
@pytest.mark.unit
def test_decouple_effect_skips_cond_bins_with_one_driver_bin() -> None:
    # tair_bin 0 has two vpd bins; tair_bin 1 has only one → contributes nothing
    df = pd.DataFrame(
        [
            {"vpd_bin": 0, "tair_bin": 0, "delta_sf": 2.0},
            {"vpd_bin": 4, "tair_bin": 0, "delta_sf": 8.0},
            {"vpd_bin": 2, "tair_bin": 1, "delta_sf": 100.0},
        ]
    )
    eff = decouple_effect(df, driver_bin="vpd_bin", cond_bin="tair_bin", response="delta_sf")
    # only tair_bin 0 populated with ≥2 vpd bins: 8 − 2 = 6
    assert eff == pytest.approx(6.0)


# ── Per-site wrapper returns all four decoupled effects ─────────────────────
@pytest.mark.unit
def test_decouple_site_returns_four_named_effects() -> None:
    rng = np.random.RandomState(0)
    n = 600
    site = pd.DataFrame(
        {
            "vpd": rng.uniform(0.2, 4.0, n),
            "tair": rng.uniform(6.0, 35.0, n),
            "sm": rng.uniform(0.1, 0.4, n),
        }
    )
    # ΔSF driven mainly by VPD
    site["delta_sf"] = 5.0 * site["vpd"] + rng.normal(0, 0.1, n)
    out = decouple_site(site, response="delta_sf", n_bins=10)
    assert set(out) == {"vpd_given_tair", "tair_given_vpd", "vpd_given_sm", "sm_given_vpd"}
    # VPD-conditioned effects should dominate the Tair-conditioned one
    assert out["vpd_given_tair"] > abs(out["tair_given_vpd"])
