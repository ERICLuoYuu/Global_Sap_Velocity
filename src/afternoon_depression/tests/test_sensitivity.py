"""Tests for the RF ±1 SD perturbation sensitivity (Liu SI Text S3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.afternoon_depression.sensitivity import rf_sensitivity


@pytest.mark.unit
def test_rf_sensitivity_flags_vpd_when_it_drives_delta() -> None:
    rng = np.random.RandomState(0)
    n = 400
    df = pd.DataFrame(
        {
            "vpd": rng.uniform(0.2, 4.0, n),
            "tair": rng.uniform(8.0, 32.0, n),
            "sm": rng.uniform(0.1, 0.4, n),
            "rad": rng.uniform(50, 900, n),
            "lai": rng.uniform(1, 6, n),
        }
    )
    df["delta_sf"] = 12.0 * df["vpd"] + rng.normal(0, 0.5, n)  # depends only on VPD
    sens = rf_sensitivity(df, n_models=8, n_estimators=40)

    assert set(sens) == {"vpd", "tair", "sm", "rad", "lai"}
    # +1 SD VPD raises predicted ΔSF the most; non-drivers stay near zero
    assert sens["vpd"] > 0
    assert sens["vpd"] == max(sens.values())
    assert abs(sens["tair"]) < sens["vpd"]


@pytest.mark.unit
def test_rf_sensitivity_per_site_sd_demotes_between_site_only_driver() -> None:
    # Two sites. SM is NEARLY constant within each site (tiny within-site SD) but differs
    # a lot BETWEEN them — a mostly between-site signal. VPD varies WITHIN each site and
    # drives ΔSF. Global SD sees SM's big between-site range; per-site SD sees a tiny SM
    # nudge (but nonzero, so no global fallback), demoting SM relative to VPD.
    rng = np.random.RandomState(3)
    frames = []
    for site, sm_val in [("A", 0.15), ("B", 0.40)]:
        n = 250
        vpd = rng.uniform(0.3, 3.5, n)
        sm = sm_val + rng.normal(0, 0.01, n)  # within-site SD ≈ 0.01 ≪ between-site gap 0.25
        frames.append(
            pd.DataFrame(
                {
                    "site_name": site,
                    "vpd": vpd,
                    "tair": rng.uniform(8, 30, n),
                    "sm": sm,
                    "rad": rng.uniform(50, 900, n),
                    "lai": rng.uniform(1, 6, n),
                    "delta_sf": 8.0 * vpd + 200.0 * sm + rng.normal(0, 0.5, n),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    glob = rf_sensitivity(df, n_models=8, n_estimators=40, sd_scope="global")
    site = rf_sensitivity(df, n_models=8, n_estimators=40, sd_scope="site")
    # within-site SM SD ≈ 0 → per-site nudge for SM collapses toward zero
    assert abs(site["sm"]) < abs(glob["sm"])
    # VPD (varies within site) stays a strong, positive sensitivity under per-site SD
    assert site["vpd"] > abs(site["sm"])


@pytest.mark.unit
def test_rf_sensitivity_returns_nan_when_too_few_samples() -> None:
    df = pd.DataFrame({"vpd": [1.0, 2.0], "tair": [10.0, 20.0], "sm": [0.2, 0.3], "delta_sf": [1.0, 2.0]})
    sens = rf_sensitivity(df, predictors=("vpd", "tair", "sm"), min_samples=50)
    assert all(np.isnan(v) for v in sens.values())
