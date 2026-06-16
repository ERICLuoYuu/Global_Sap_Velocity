# Site-level SM–VPD Decoupling (sap velocity & canopy conductance) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port Liu et al. 2020's SM/VPD percentile-binning decoupling to site-level sap-flow data, disentangling whether low soil moisture or high VPD dominates the dryness response of two responses — sap velocity (E, water-use) and Flo 2021 canopy conductance (Gc, the SIF-analog) — across 5 soil-moisture depth variants.

**Architecture:** New standalone module `src/sm_vpd_decoupling/`. A fresh merge run produces a daytime, treatment-filtered, all-season daily dataset; `loader.py` standardises it, resolves a PPFD source, applies Liu's day filter, computes root-zone SM and Gc, and per-site-normalizes both responses; `decoupling.py` runs the nested-binning estimator (Liu Eqs. 1–2); `run_*.py` sweeps responses × SM variants × bin counts × min-valid-days and writes CSVs, a depth-profile dissociation table, an attrition table, and figures.

**Tech Stack:** Python 3.9 (Palma) / 3.10 (local), pandas, numpy, matplotlib (Agg), pytest. Imports use `from src.sm_vpd_decoupling.…`. Every runtime module starts with `from __future__ import annotations` (3.9 compat).

**Spec:** `docs/superpowers/specs/2026-06-16-sm-vpd-decoupling-sapflow-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/sm_vpd_decoupling/__init__.py` | Package docstring |
| `src/sm_vpd_decoupling/conductance.py` | SFD unit conversion + Flo 2021 Gc (Eqn 2) |
| `src/sm_vpd_decoupling/decoupling.py` | Percentile binning + ΔResp(SM\|VPD)/ΔResp(VPD\|SM) estimator |
| `src/sm_vpd_decoupling/sensitivity.py` | δResp/δSM per 0.1 m³/m³ within VPD bins |
| `src/sm_vpd_decoupling/loader.py` | Resolve daily dir, PPFD source, root-zone SM, Liu day-filter, per-site normalization, build table |
| `src/sm_vpd_decoupling/aggregate.py` | Per-site effects table, depth-profile dissociation table, attrition table |
| `src/sm_vpd_decoupling/plotting.py` | Figures (a1) example sites, (a2) cross-site aggregate, (b) depth dominance, (c) gradient violins |
| `src/sm_vpd_decoupling/run_sm_vpd_decoupling.py` | CLI orchestration |
| `src/sm_vpd_decoupling/job_merge_decoupling.sh` | SLURM: data-production merge run |
| `src/sm_vpd_decoupling/job_sm_vpd_decoupling.sh` | SLURM: analysis run |
| `src/sm_vpd_decoupling/tests/test_*.py` | pytest suites |

**Constants (defined once in the modules that own them):**
- `conductance.py`: `SFD_CM3CM2H_TO_KGM2S = 1e-3 * 1e4 / 3600` (≈2.7778e-3); `ETA = 44.6`; `T0_K = 273.0`; `SW_TO_PPFD = 2.04`
- `decoupling.py`: `MIN_BIN_COUNT = 3`; `MIN_COND_BINS = 2`
- `loader.py`: `ROOT_ZONE_WEIGHTS = (0.07, 0.21, 0.72)`; `VPD_MIN_KPA = 0.5`; `PPFD_MIN = 500.0`; `NORM_QUANTILE = 0.90`
- `run`: `SM_VARIANTS = ("swvl1","swvl2","swvl3","swvl4","root_zone_sm")`; `RESPONSES = ("E_norm","Gc_norm")`; `N_BINS = (5, 10)`; `MIN_VALID_DAYS = (120, 240, 360)`; `T_MIN_PRIMARY = 15.0`; `T_MIN_SENSITIVITY = 5.0`

---

### Task 0: Module scaffold + data-production SLURM script

**Files:**
- Create: `src/sm_vpd_decoupling/__init__.py`
- Create: `src/sm_vpd_decoupling/tests/__init__.py`
- Create: `src/sm_vpd_decoupling/job_merge_decoupling.sh`

- [ ] **Step 1: Create the package `__init__.py`**

```python
"""Site-level SM-VPD decoupling for sap velocity and canopy conductance.

Ports Liu et al. (2020, Nat. Commun. 11:4892) SM/VPD percentile-binning
decoupling to site-level sap flow. Two responses: sap velocity (E, water-use)
and Flo et al. (2021, New Phytol. 231:617-630, Eqn 2) canopy conductance (Gc,
SIF-analog). See docs/superpowers/specs/2026-06-16-sm-vpd-decoupling-sapflow-design.md.
"""

from __future__ import annotations
```

- [ ] **Step 2: Create empty test package marker**

Create `src/sm_vpd_decoupling/tests/__init__.py` with an empty file (no content).

- [ ] **Step 3: Write the data-production SLURM script**

Create `src/sm_vpd_decoupling/job_merge_decoupling.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=smvpd_merge
#SBATCH --partition=requeue-zen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/smvpd_merge_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/smvpd_merge_%j.err

# Data production for the SM-VPD decoupling analysis:
# daytime-only + treatment-filter ON + growing-season OFF (Liu's Tair filter
# screens season downstream). Source = outliers_removed (measured, NOT gap-filled).
set -e
hostname; date
cd /scratch/tmp/yluo2/gsv
source .venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv:$PYTHONPATH
python --version

python notebooks/merge_gap_filled_hourly_orginal.py \
    --daytime-only \
    --apply-treatment-filter \
    --output-dir outputs/processed_data/sapwood/merged_decoupling

echo "=== Output ==="
ls outputs/processed_data/sapwood/merged_decoupling/daily/*.csv 2>/dev/null | wc -l || true
date
```

- [ ] **Step 4: Commit**

```bash
git add src/sm_vpd_decoupling/__init__.py src/sm_vpd_decoupling/tests/__init__.py src/sm_vpd_decoupling/job_merge_decoupling.sh
git commit -m "feat(sm-vpd): scaffold module + data-production SLURM script"
```

---

### Task 1: Canopy conductance (`conductance.py`)

**Files:**
- Create: `src/sm_vpd_decoupling/conductance.py`
- Test: `src/sm_vpd_decoupling/tests/test_conductance.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest src/sm_vpd_decoupling/tests/test_conductance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.sm_vpd_decoupling.conductance'`

- [ ] **Step 3: Write the implementation**

```python
# src/sm_vpd_decoupling/conductance.py
"""Whole-tree canopy conductance from sap flux density (Flo et al. 2021, Eqn 2).

Flo, V., Martinez-Vilalta, J., et al. (2021). "Climate and functional traits
jointly mediate tree water-use strategies." New Phytologist 231(2): 617-630,
doi:10.1111/nph.17404, Eqn 2 (after Phillips & Oren 1998).

    G_Asw = (115.8 + 0.4236*T) * (SFD/VPD) * (eta*T0/(T0+T)) * exp(0.00012*h)

with SFD in kg m-2_Asw s-1, T in degC, VPD in kPa, h = altitude (m).

NOTE: Gc is proportional to 1/VPD. Binning Gc BY VPD therefore induces a
spurious negative Gc-VPD relationship (Oren et al. 1999). The SM leg of the
decoupling (computed within VPD bins) is unaffected; the VPD leg is biased.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 1 cm3 cm-2 h-1 of water -> kg m-2 s-1 (rho_water = 1 g cm-3).
SFD_CM3CM2H_TO_KGM2S = 1e-3 * 1e4 / 3600.0
ETA = 44.6  # mol m-3, molar air density at STP
T0_K = 273.0  # K
SW_TO_PPFD = 2.04  # umol J-1, shortwave -> PAR (used by loader)

Numeric = "float | pd.Series"


def sfd_to_kg_m2_s(sfd_cm3_cm2_h):
    """Convert sap flux density from cm3 cm-2 h-1 to kg m-2 s-1."""
    return sfd_cm3_cm2_h * SFD_CM3CM2H_TO_KGM2S


def canopy_conductance(sap_velocity, tair_c, vpd_kpa, altitude_m):
    """Flo 2021 Eqn 2 whole-tree canopy conductance per sapwood area (mol m-2 s-1).

    Accepts scalars or pandas Series (broadcast). VPD <= 0 -> NaN (avoids div0).
    """
    sfd = sfd_to_kg_m2_s(sap_velocity)
    vpd = vpd_kpa
    if isinstance(vpd, pd.Series):
        vpd = vpd.where(vpd > 0)
    elif vpd <= 0:
        return float("nan")
    gc = (
        (115.8 + 0.4236 * tair_c)
        * (sfd / vpd)
        * (ETA * T0_K / (T0_K + tair_c))
        * np.exp(0.00012 * altitude_m)
    )
    return gc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest src/sm_vpd_decoupling/tests/test_conductance.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sm_vpd_decoupling/conductance.py src/sm_vpd_decoupling/tests/test_conductance.py
git commit -m "feat(sm-vpd): canopy conductance (Flo 2021 Eqn 2) with 1/VPD confound documented"
```

---

### Task 2: Decoupling estimator (`decoupling.py`)

**Files:**
- Create: `src/sm_vpd_decoupling/decoupling.py`
- Test: `src/sm_vpd_decoupling/tests/test_decoupling.py`

- [ ] **Step 1: Write the failing tests**

```python
# src/sm_vpd_decoupling/tests/test_decoupling.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.sm_vpd_decoupling.decoupling import (
    assign_percentile_bins,
    decouple_effect,
    decouple_site,
)


def _synthetic(driver: str, n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """SM and VPD fully decoupled (independent uniforms) so both axes populate.
    `driver` selects which one the response depends on (response = that driver)."""
    rng = np.random.default_rng(seed)
    sm = rng.uniform(0.0, 1.0, n)
    vpd = rng.uniform(0.5, 3.0, n)
    resp = sm.copy() if driver == "sm" else (vpd - 0.5) / 2.5
    return pd.DataFrame({"sm": sm, "vpd": vpd, "resp": resp})


def test_assign_percentile_bins_basic():
    s = pd.Series(np.arange(100.0))
    codes = assign_percentile_bins(s, 5)
    assert codes.notna().all()
    assert sorted(codes.unique()) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_assign_percentile_bins_degenerate_all_same():
    s = pd.Series([1.0] * 50)
    codes = assign_percentile_bins(s, 5)
    # Single-value series collapses to one bin (no gradient), no exception.
    assert set(codes.dropna().unique()).issubset({0.0})


def test_sm_dominant_case():
    df = _synthetic("sm")
    eff = decouple_site(df, response="resp", n_bins=5, vpd_col="vpd", sm_col="sm")
    # Low SM is the stressor -> sm_given_vpd strongly negative.
    assert eff["sm_given_vpd"] < -0.5
    # VPD has no effect on response -> ~0.
    assert abs(eff["vpd_given_sm"]) < 0.15
    assert abs(eff["sm_given_vpd"]) > abs(eff["vpd_given_sm"])


def test_vpd_dominant_case():
    df = _synthetic("vpd")
    eff = decouple_site(df, response="resp", n_bins=5, vpd_col="vpd", sm_col="sm")
    assert eff["vpd_given_sm"] > 0.5
    assert abs(eff["sm_given_vpd"]) < 0.15
    assert abs(eff["vpd_given_sm"]) > abs(eff["sm_given_vpd"])


def test_min_cond_bins_returns_nan():
    # Only one VPD value -> for vpd_given_sm there are <2 driver bins everywhere,
    # and for sm_given_vpd only one conditioning (VPD) bin -> < MIN_COND_BINS.
    df = pd.DataFrame({"sm": np.linspace(0, 1, 100), "vpd": np.ones(100), "resp": np.linspace(0, 1, 100)})
    eff = decouple_site(df, response="resp", n_bins=5, vpd_col="vpd", sm_col="sm")
    assert np.isnan(eff["sm_given_vpd"])
    assert np.isnan(eff["vpd_given_sm"])


def test_min_bin_count_drops_sparse_cells():
    df = _synthetic("sm", n=40)  # ~1-2 points per 5x5 cell
    eff = decouple_site(df, response="resp", n_bins=5, vpd_col="vpd", sm_col="sm", min_bin_count=3)
    # Too sparse: most cells < 3 points -> NaN (not a misleading number).
    assert np.isnan(eff["sm_given_vpd"]) or np.isnan(eff["vpd_given_sm"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest src/sm_vpd_decoupling/tests/test_decoupling.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/sm_vpd_decoupling/decoupling.py
"""Nested percentile-binning decoupling of SM and VPD (Liu et al. 2020, Eqs. 1-2).

For each site, bin SM and VPD into per-site percentiles, then measure each
driver's effect on the response WITHIN bins of the other (where residual SM-VPD
correlation is ~0):

  dResp(SM|VPD) = mean over populated VPD bins of [resp(lowest SM bin)
                  - resp(highest SM bin)]      (low SM is the stressor; Eq. 2)
  dResp(VPD|SM) = mean over populated SM bins of [resp(highest VPD bin)
                  - resp(lowest VPD bin)]      (Eq. 1)

A driver cell counts only with >= MIN_BIN_COUNT points; a site effect needs
>= MIN_COND_BINS populated conditioning bins, else NaN (reported insufficient).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_BIN_COUNT = 3
MIN_COND_BINS = 2


def assign_percentile_bins(values: pd.Series, n_bins: int) -> pd.Series:
    """Per-series percentile bin index (0..n_bins-1) via ``qcut`` (duplicates dropped).

    Low-variance series collapse to fewer bins; a single-value series -> bin 0.
    """
    v = pd.to_numeric(values, errors="coerce")
    try:
        codes = pd.qcut(v, q=n_bins, labels=False, duplicates="drop")
    except (ValueError, IndexError):
        codes = pd.Series(np.nan, index=v.index)
    codes = pd.Series(np.asarray(codes, dtype="float64"), index=v.index)
    if codes.isna().all():
        codes = pd.Series(np.where(v.notna(), 0.0, np.nan), index=v.index)
    return codes


def decouple_effect(
    df: pd.DataFrame,
    driver_bin: str,
    cond_bin: str,
    response: str,
    low_minus_high: bool = False,
    min_bin_count: int = MIN_BIN_COUNT,
    min_cond_bins: int = MIN_COND_BINS,
) -> float:
    """Mean over populated conditioning bins of (high - low) driver-bin response.

    ``low_minus_high=True`` flips the sign (low - high), used for SM where low SM
    is the stressor (Liu Eq. 2). Cells with < ``min_bin_count`` points are dropped;
    < ``min_cond_bins`` populated conditioning bins -> NaN.
    """
    grouped = df.groupby([cond_bin, driver_bin])[response]
    cell_mean = grouped.mean()
    cell_n = grouped.size()
    cell_mean = cell_mean[cell_n >= min_bin_count]
    effects: list[float] = []
    for _cond_val, sub in cell_mean.groupby(level=0):
        sub = sub.droplevel(0)
        if sub.index.nunique() < 2:
            continue
        high = float(sub.loc[sub.index.max()])
        low = float(sub.loc[sub.index.min()])
        effects.append(low - high if low_minus_high else high - low)
    if len(effects) < min_cond_bins:
        return float("nan")
    return float(np.mean(effects))


def decouple_site(
    site_df: pd.DataFrame,
    response: str,
    n_bins: int,
    vpd_col: str = "vpd",
    sm_col: str = "sm",
    min_bin_count: int = MIN_BIN_COUNT,
) -> dict[str, float]:
    """Both decoupled effects for one site's site-day records (VPD-vs-SM pair)."""
    df = site_df.copy()
    df["_vpd_bin"] = assign_percentile_bins(df[vpd_col], n_bins)
    df["_sm_bin"] = assign_percentile_bins(df[sm_col], n_bins)
    df = df.dropna(subset=["_vpd_bin", "_sm_bin", response])
    return {
        "sm_given_vpd": decouple_effect(
            df, "_sm_bin", "_vpd_bin", response, low_minus_high=True, min_bin_count=min_bin_count
        ),
        "vpd_given_sm": decouple_effect(
            df, "_vpd_bin", "_sm_bin", response, low_minus_high=False, min_bin_count=min_bin_count
        ),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest src/sm_vpd_decoupling/tests/test_decoupling.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sm_vpd_decoupling/decoupling.py src/sm_vpd_decoupling/tests/test_decoupling.py
git commit -m "feat(sm-vpd): nested-binning SM/VPD decoupling estimator (Liu Eqs. 1-2)"
```

---

### Task 3: SM sensitivity (`sensitivity.py`)

**Files:**
- Create: `src/sm_vpd_decoupling/sensitivity.py`
- Test: `src/sm_vpd_decoupling/tests/test_sensitivity.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest src/sm_vpd_decoupling/tests/test_sensitivity.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/sm_vpd_decoupling/sensitivity.py
"""Standardized sensitivity of the response to SM (Liu et al. 2020).

delta(Resp)/delta(SM) per 0.1 m3/m3, computed WITHIN VPD bins (so the SM-VPD
coupling is broken) and averaged over populated VPD bins. Removes the SM-range
effect so the sensitivity is comparable across sites.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.sm_vpd_decoupling.decoupling import (
    MIN_BIN_COUNT,
    MIN_COND_BINS,
    assign_percentile_bins,
)

SM_STEP = 0.1  # m3/m3


def sm_sensitivity(
    site_df: pd.DataFrame,
    response: str,
    n_bins: int,
    vpd_col: str = "vpd",
    sm_col: str = "sm",
    min_bin_count: int = MIN_BIN_COUNT,
    min_cond_bins: int = MIN_COND_BINS,
) -> float:
    """Mean over VPD bins of [d(resp)/d(SM)] * 0.1, using the highest/lowest
    populated SM bins within each VPD bin (Liu approach i, slope form)."""
    df = site_df.copy()
    df["_vpd_bin"] = assign_percentile_bins(df[vpd_col], n_bins)
    df["_sm_bin"] = assign_percentile_bins(df[sm_col], n_bins)
    df = df.dropna(subset=["_vpd_bin", "_sm_bin", response, sm_col])

    grouped = df.groupby(["_vpd_bin", "_sm_bin"])
    resp_mean = grouped[response].mean()
    sm_mean = grouped[sm_col].mean()
    cell_n = grouped.size()
    keep = cell_n >= min_bin_count
    resp_mean = resp_mean[keep]
    sm_mean = sm_mean[keep]

    slopes: list[float] = []
    for _vpd_val, sub in resp_mean.groupby(level=0):
        sub = sub.droplevel(0)
        if sub.index.nunique() < 2:
            continue
        hi_bin, lo_bin = sub.index.max(), sub.index.min()
        sm_hi = float(sm_mean.loc[(_vpd_val, hi_bin)])
        sm_lo = float(sm_mean.loc[(_vpd_val, lo_bin)])
        if sm_hi == sm_lo:
            continue
        slope = (float(sub.loc[hi_bin]) - float(sub.loc[lo_bin])) / (sm_hi - sm_lo)
        slopes.append(slope * SM_STEP)
    if len(slopes) < min_cond_bins:
        return float("nan")
    return float(np.mean(slopes))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest src/sm_vpd_decoupling/tests/test_sensitivity.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sm_vpd_decoupling/sensitivity.py src/sm_vpd_decoupling/tests/test_sensitivity.py
git commit -m "feat(sm-vpd): standardized SM sensitivity per 0.1 m3/m3 within VPD bins"
```

---

### Task 4: Loader — PPFD resolution, root-zone SM, normalization (`loader.py` part 1)

**Files:**
- Create: `src/sm_vpd_decoupling/loader.py`
- Test: `src/sm_vpd_decoupling/tests/test_loader_units.py`

- [ ] **Step 1: Write the failing tests**

```python
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
    df = pd.DataFrame({
        "ppfd_in": [600.0, 700.0],
        "sw_in": [100.0, 100.0],
        "surface_solar_radiation_downwards_hourly": [50.0, 50.0],
    })
    ppfd, source = resolve_ppfd(df)
    assert source == "ppfd_in"
    assert list(ppfd) == [600.0, 700.0]


def test_resolve_ppfd_falls_back_to_swin():
    df = pd.DataFrame({
        "ppfd_in": [np.nan, np.nan],
        "sw_in": [300.0, 400.0],
        "surface_solar_radiation_downwards_hourly": [50.0, 50.0],
    })
    ppfd, source = resolve_ppfd(df)
    assert source == "sw_in"
    assert math.isclose(ppfd.iloc[0], 300.0 * 2.04, rel_tol=1e-9)


def test_resolve_ppfd_falls_back_to_era5():
    df = pd.DataFrame({
        "ppfd_in": [np.nan, np.nan],
        "sw_in": [np.nan, np.nan],
        "surface_solar_radiation_downwards_hourly": [250.0, 260.0],
    })
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest src/sm_vpd_decoupling/tests/test_loader_units.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation (part 1 of loader)**

```python
# src/sm_vpd_decoupling/loader.py
"""Load and prepare the daily merged dataset for SM-VPD decoupling.

Reads the daytime, treatment-filtered, all-season daily CSVs produced by the
data-production merge run, resolves a PPFD source, computes root-zone SM and Gc,
applies Liu's day filter, and per-site-normalizes both responses (E and Gc).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.sm_vpd_decoupling.conductance import SW_TO_PPFD, canopy_conductance

logger = logging.getLogger(__name__)

ClimateSource = Literal["site", "era5"]

ROOT_ZONE_WEIGHTS = (0.07, 0.21, 0.72)  # ERA5-Land layer thickness fractions to 100 cm
VPD_MIN_KPA = 0.5
PPFD_MIN = 500.0
NORM_QUANTILE = 0.90


def root_zone_sm(swvl1: pd.Series, swvl2: pd.Series, swvl3: pd.Series) -> pd.Series:
    """Fixed 0-100 cm thickness-weighted root-zone SM (Liu's 0-1 m analog)."""
    w1, w2, w3 = ROOT_ZONE_WEIGHTS
    return w1 * swvl1 + w2 * swvl2 + w3 * swvl3


def resolve_ppfd(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """PPFD source resolution: ppfd_in -> sw_in*2.04 -> ERA5 ssrd*2.04.

    Returns (ppfd_series, source_label). Picks the first source that has any
    non-null value; conversions assume daily-mean W m-2 -> umol m-2 s-1 PAR.
    """
    if "ppfd_in" in df and pd.to_numeric(df["ppfd_in"], errors="coerce").notna().any():
        return pd.to_numeric(df["ppfd_in"], errors="coerce"), "ppfd_in"
    if "sw_in" in df and pd.to_numeric(df["sw_in"], errors="coerce").notna().any():
        return pd.to_numeric(df["sw_in"], errors="coerce") * SW_TO_PPFD, "sw_in"
    col = "surface_solar_radiation_downwards_hourly"
    if col in df and pd.to_numeric(df[col], errors="coerce").notna().any():
        return pd.to_numeric(df[col], errors="coerce") * SW_TO_PPFD, "era5_ssrd"
    return pd.Series(np.nan, index=df.index), "none"


def normalize_per_site(series: pd.Series) -> pd.Series:
    """Divide by the mean of values at/above the 90th percentile (Liu).

    Uses >= the quantile so the anchor is never empty; a constant series
    normalizes to 1.0 everywhere (no div-by-zero).
    """
    s = pd.to_numeric(series, errors="coerce")
    anchor = s[s >= s.quantile(NORM_QUANTILE)].mean()
    if not np.isfinite(anchor) or anchor == 0:
        return pd.Series(np.nan, index=s.index)
    return s / anchor
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest src/sm_vpd_decoupling/tests/test_loader_units.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sm_vpd_decoupling/loader.py src/sm_vpd_decoupling/tests/test_loader_units.py
git commit -m "feat(sm-vpd): loader PPFD resolution, root-zone SM, per-site normalization"
```

---

### Task 5: Loader — day filter + table builder (`loader.py` part 2)

**Files:**
- Modify: `src/sm_vpd_decoupling/loader.py` (append functions)
- Test: `src/sm_vpd_decoupling/tests/test_loader_table.py`

- [ ] **Step 1: Write the failing tests**

```python
# src/sm_vpd_decoupling/tests/test_loader_table.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.sm_vpd_decoupling.loader import apply_day_filter, standardise_site_frame


def _raw_site_frame() -> pd.DataFrame:
    n = 10
    return pd.DataFrame({
        "site_name": ["S1"] * n,
        "TIMESTAMP": pd.date_range("2010-06-01", periods=n, freq="D"),
        "sap_velocity": np.linspace(1.0, 10.0, n),
        "ta": np.linspace(10.0, 25.0, n),
        "vpd": np.linspace(0.2, 2.0, n),
        "ppfd_in": np.linspace(300.0, 900.0, n),
        "sw_in": np.linspace(100.0, 400.0, n),
        "surface_solar_radiation_downwards_hourly": np.linspace(100.0, 400.0, n),
        "volumetric_soil_water_layer_1": np.linspace(0.1, 0.3, n),
        "volumetric_soil_water_layer_2": np.linspace(0.1, 0.3, n),
        "volumetric_soil_water_layer_3": np.linspace(0.1, 0.3, n),
        "volumetric_soil_water_layer_4": np.linspace(0.1, 0.3, n),
        "temperature_2m": np.linspace(10.0, 25.0, n) + 273.15,
        "dewpoint_2m": np.linspace(5.0, 15.0, n) + 273.15,
        "elevation": [100.0] * n,
        "pft": ["ENF"] * n,
        "biome": ["temperate"] * n,
        "prcip/PET": [0.6] * n,
        "canopy_height": [20.0] * n,
        "latitude_x": [45.0] * n,
        "longitude_x": [7.0] * n,
    })


def test_standardise_site_frame_schema():
    out = standardise_site_frame(_raw_site_frame(), climate_source="site")
    for col in ["site_name", "date", "E", "Gc", "vpd", "tair", "ppfd",
                "swvl1", "swvl2", "swvl3", "swvl4", "root_zone_sm",
                "pft", "biome", "aridity", "canopy_height", "elevation"]:
        assert col in out.columns, col
    # negative sap velocity dropped -> here none negative, all finite E.
    assert out["E"].notna().all()


def test_standardise_drops_negative_sap_velocity():
    raw = _raw_site_frame()
    raw.loc[0, "sap_velocity"] = -5.0
    out = standardise_site_frame(raw, climate_source="site")
    assert np.isnan(out.loc[out.index[0], "E"])


def test_apply_day_filter_thresholds():
    out = standardise_site_frame(_raw_site_frame(), climate_source="site")
    filt15 = apply_day_filter(out, tair_min=15.0)
    filt5 = apply_day_filter(out, tair_min=5.0)
    # All retained rows satisfy the three thresholds.
    assert (filt15["tair"] > 15.0).all()
    assert (filt15["vpd"] > 0.5).all()
    assert (filt15["ppfd"] > 500.0).all()
    # Relaxing Tair keeps at least as many rows.
    assert len(filt5) >= len(filt15)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest src/sm_vpd_decoupling/tests/test_loader_table.py -v`
Expected: FAIL with `ImportError: cannot import name 'apply_day_filter'`

- [ ] **Step 3: Append the implementation to `loader.py`**

```python
# --- append to src/sm_vpd_decoupling/loader.py ---

_CARRY_COLS = {
    "pft": "pft",
    "biome": "biome",
    "prcip/PET": "aridity",
    "canopy_height": "canopy_height",
    "elevation": "elevation",
    "latitude_x": "lat",
    "longitude_x": "lon",
}


def _kelvin_to_celsius(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s - 273.15 if s.median(skipna=True) > 100 else s


def _vpd_from_era5(tair_c: pd.Series, dewpoint_c: pd.Series) -> pd.Series:
    """VPD (kPa) from temperature and dewpoint via Tetens (es - ea)."""
    es = 0.6108 * np.exp(17.27 * tair_c / (tair_c + 237.3))
    ea = 0.6108 * np.exp(17.27 * dewpoint_c / (dewpoint_c + 237.3))
    return (es - ea).clip(lower=0.0)


def standardise_site_frame(df: pd.DataFrame, climate_source: ClimateSource) -> pd.DataFrame:
    """Map one site's raw daily CSV to the analysis schema (E, Gc, drivers, SM, carry)."""
    out = pd.DataFrame(index=df.index)
    out["site_name"] = df["site_name"] if "site_name" in df.columns else "unknown"
    out["date"] = pd.to_datetime(df["TIMESTAMP"]).dt.date

    e = pd.to_numeric(df["sap_velocity"], errors="coerce")
    e[e < 0] = np.nan  # reverse/invalid flow
    out["E"] = e

    if climate_source == "era5":
        tair = _kelvin_to_celsius(df["temperature_2m"])
        out["tair"] = tair
        out["vpd"] = _vpd_from_era5(tair, _kelvin_to_celsius(df["dewpoint_2m"]))
    else:
        out["tair"] = pd.to_numeric(df["ta"], errors="coerce")
        out["vpd"] = pd.to_numeric(df["vpd"], errors="coerce")

    out["ppfd"], out["ppfd_source"] = resolve_ppfd(df)

    out["swvl1"] = pd.to_numeric(df["volumetric_soil_water_layer_1"], errors="coerce")
    out["swvl2"] = pd.to_numeric(df["volumetric_soil_water_layer_2"], errors="coerce")
    out["swvl3"] = pd.to_numeric(df["volumetric_soil_water_layer_3"], errors="coerce")
    out["swvl4"] = pd.to_numeric(df["volumetric_soil_water_layer_4"], errors="coerce")
    out["root_zone_sm"] = root_zone_sm(out["swvl1"], out["swvl2"], out["swvl3"])

    elevation = pd.to_numeric(df.get("elevation", pd.Series(np.nan, index=df.index)), errors="coerce")
    out["Gc"] = canopy_conductance(out["E"], out["tair"], out["vpd"], elevation.fillna(0.0))

    for raw_col, std_col in _CARRY_COLS.items():
        if raw_col in df.columns:
            out[std_col] = df[raw_col]
    return out


def apply_day_filter(table: pd.DataFrame, tair_min: float) -> pd.DataFrame:
    """Liu day filter (AND): Tair > tair_min, VPD > 0.5 kPa, PPFD > 500 umol m-2 s-1."""
    before = len(table)
    mask = (
        (table["tair"] > tair_min)
        & (table["vpd"] > VPD_MIN_KPA)
        & (table["ppfd"] > PPFD_MIN)
    )
    out = table[mask].copy()
    logger.info("Day filter (Tair>%.0f,VPD>%.1f,PPFD>%.0f): %d -> %d rows",
                tair_min, VPD_MIN_KPA, PPFD_MIN, before, len(out))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest src/sm_vpd_decoupling/tests/test_loader_table.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sm_vpd_decoupling/loader.py src/sm_vpd_decoupling/tests/test_loader_table.py
git commit -m "feat(sm-vpd): loader day filter + site-frame standardisation"
```

---

### Task 6: Loader — directory resolution + full table build (`loader.py` part 3)

**Files:**
- Modify: `src/sm_vpd_decoupling/loader.py` (append)
- Test: `src/sm_vpd_decoupling/tests/test_loader_load.py`

- [ ] **Step 1: Write the failing test (uses a tmp dir of CSVs)**

```python
# src/sm_vpd_decoupling/tests/test_loader_load.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sm_vpd_decoupling.loader import load_table, resolve_daily_dir


def _write_site_csv(path, site, n=400, seed=0):
    rng = np.random.default_rng(seed)
    sm = rng.uniform(0.1, 0.35, n)
    vpd = rng.uniform(0.3, 2.5, n)
    df = pd.DataFrame({
        "site_name": [site] * n,
        "TIMESTAMP": pd.date_range("2008-01-01", periods=n, freq="D"),
        "sap_velocity": 5.0 * sm + rng.normal(0, 0.05, n),
        "ta": rng.uniform(12.0, 28.0, n),
        "vpd": vpd,
        "ppfd_in": rng.uniform(400.0, 1200.0, n),
        "sw_in": rng.uniform(100.0, 500.0, n),
        "surface_solar_radiation_downwards_hourly": rng.uniform(100.0, 500.0, n),
        "volumetric_soil_water_layer_1": sm,
        "volumetric_soil_water_layer_2": sm,
        "volumetric_soil_water_layer_3": sm,
        "volumetric_soil_water_layer_4": sm,
        "temperature_2m": rng.uniform(12.0, 28.0, n) + 273.15,
        "dewpoint_2m": rng.uniform(5.0, 15.0, n) + 273.15,
        "elevation": [200.0] * n,
        "pft": ["ENF"] * n,
        "biome": ["temperate"] * n,
        "prcip/PET": [0.7] * n,
        "canopy_height": [18.0] * n,
        "latitude_x": [46.0] * n,
        "longitude_x": [8.0] * n,
    })
    df.to_csv(path, index=False)


def test_resolve_daily_dir_explicit(tmp_path):
    d = tmp_path / "daily"
    d.mkdir()
    _write_site_csv(d / "S1_daily.csv", "S1")
    assert resolve_daily_dir(str(d)) == d


def test_load_table_builds_normalized_responses(tmp_path):
    d = tmp_path / "daily"
    d.mkdir()
    _write_site_csv(d / "S1_daily.csv", "S1", seed=1)
    _write_site_csv(d / "S2_daily.csv", "S2", seed=2)
    table = load_table(str(d), climate_source="site", tair_min=15.0)
    assert {"E_norm", "Gc_norm", "swvl1", "root_zone_sm"}.issubset(table.columns)
    assert set(table["site_name"].unique()) == {"S1", "S2"}
    # Normalized responses are positive and finite where E is present.
    assert table["E_norm"].dropna().gt(0).all()


def test_load_table_raises_on_empty_dir(tmp_path):
    d = tmp_path / "daily"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        load_table(str(d), climate_source="site", tair_min=15.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest src/sm_vpd_decoupling/tests/test_loader_load.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_table'`

- [ ] **Step 3: Append the implementation to `loader.py`**

```python
# --- append to src/sm_vpd_decoupling/loader.py ---

_DEFAULT_CANDIDATES = (
    "merged_decoupling/daily",
    "merged_decoupling/daytime_only/daily",
    "merged/daytime_only/daily",
)


def resolve_daily_dir(data_dir: str | None, processed_root: Path | None = None) -> Path:
    """Return the daily CSV directory. An explicit ``data_dir`` wins; otherwise
    search known candidates under ``processed_root/sapwood``."""
    if data_dir is not None:
        d = Path(data_dir)
        if not (d.exists() and any(d.glob("*.csv"))):
            raise FileNotFoundError(f"--data-dir has no *.csv files: {d}")
        return d
    root = (processed_root or Path("outputs/processed_data")) / "sapwood"
    for cand in _DEFAULT_CANDIDATES:
        d = root / cand
        if d.exists() and any(d.glob("*.csv")):
            logger.info("Resolved daily dir: %s", d)
            return d
    raise FileNotFoundError(f"No daily dir found under {root}")


def load_table(
    data_dir: str | None,
    climate_source: ClimateSource = "site",
    tair_min: float = 15.0,
    processed_root: Path | None = None,
) -> pd.DataFrame:
    """Full pipeline: per-site daily CSVs -> filtered, normalized site-day table.

    Normalization (E_norm, Gc_norm) is computed PER SITE on the day-filtered rows.
    """
    daily_dir = resolve_daily_dir(data_dir, processed_root)
    files = [f for f in sorted(daily_dir.glob("*.csv")) if "all_biomes" not in f.name]
    if not files:
        raise FileNotFoundError(f"No per-site daily CSVs in {daily_dir}")

    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            raw = pd.read_csv(f)
            if raw.empty or "sap_velocity" not in raw.columns:
                continue
            std = standardise_site_frame(raw, climate_source)
            std = apply_day_filter(std, tair_min=tair_min)
            if std.empty:
                continue
            std["E_norm"] = normalize_per_site(std["E"])
            std["Gc_norm"] = normalize_per_site(std["Gc"])
            frames.append(std)
        except (ValueError, KeyError) as exc:
            logger.warning("Skipping %s: %s", f.name, exc)
    if not frames:
        raise ValueError("No site files survived loading/filtering.")
    table = pd.concat(frames, ignore_index=True)
    logger.info("Loaded table: %d rows, %d sites", len(table), table["site_name"].nunique())
    return table
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest src/sm_vpd_decoupling/tests/test_loader_load.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sm_vpd_decoupling/loader.py src/sm_vpd_decoupling/tests/test_loader_load.py
git commit -m "feat(sm-vpd): loader daily-dir resolution + full normalized table build"
```

---

### Task 7: Aggregation — per-site effects, depth-profile, attrition (`aggregate.py`)

**Files:**
- Create: `src/sm_vpd_decoupling/aggregate.py`
- Test: `src/sm_vpd_decoupling/tests/test_aggregate.py`

- [ ] **Step 1: Write the failing tests**

```python
# src/sm_vpd_decoupling/tests/test_aggregate.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.sm_vpd_decoupling.aggregate import decouple_all_sites, dominance_summary


def _table(n_per_site=400, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for si, site in enumerate(["A", "B", "C"]):
        sm = rng.uniform(0.1, 0.35, n_per_site)
        vpd = rng.uniform(0.5, 2.5, n_per_site)
        resp = (sm - 0.1) / 0.25  # SM-dominant, normalized-ish
        frames.append(pd.DataFrame({
            "site_name": site, "vpd": vpd, "swvl1": sm, "E_norm": resp,
            "biome": "temperate", "pft": "ENF", "aridity": 0.7,
            "canopy_height": 18.0, "lat": 46.0, "lon": 8.0,
        }))
    return pd.concat(frames, ignore_index=True)


def test_decouple_all_sites_columns_and_filter():
    table = _table()
    out = decouple_all_sites(table, sm_col="swvl1", response="E_norm",
                             n_bins=5, min_valid_days=120)
    assert set(["site_name", "n_days", "sm_given_vpd", "vpd_given_sm",
                "sensitivity", "biome", "pft", "aridity"]).issubset(out.columns)
    assert len(out) == 3  # all sites have >= 120 days
    # SM-dominant synthetic -> |sm_given_vpd| > |vpd_given_sm| for every site.
    assert (out["sm_given_vpd"].abs() > out["vpd_given_sm"].abs()).all()


def test_decouple_all_sites_min_valid_days():
    table = _table(n_per_site=50)
    out = decouple_all_sites(table, sm_col="swvl1", response="E_norm",
                             n_bins=5, min_valid_days=120)
    assert len(out) == 0  # no site reaches 120 days


def test_dominance_summary():
    effects = pd.DataFrame({
        "sm_given_vpd": [-0.4, -0.3, -0.1],
        "vpd_given_sm": [0.1, 0.2, 0.5],
    })
    pct, n = dominance_summary(effects)
    # SM wins in 2 of 3 valid sites.
    assert n == 3
    assert abs(pct - (2 / 3 * 100)) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest src/sm_vpd_decoupling/tests/test_aggregate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/sm_vpd_decoupling/aggregate.py
"""Aggregate per-site decoupled effects into site tables and dominance summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.sm_vpd_decoupling.decoupling import MIN_BIN_COUNT, decouple_site
from src.sm_vpd_decoupling.sensitivity import sm_sensitivity

_CARRY = ("biome", "pft", "aridity", "canopy_height", "lat", "lon")


def decouple_all_sites(
    table: pd.DataFrame,
    sm_col: str,
    response: str,
    n_bins: int,
    min_valid_days: int,
    vpd_col: str = "vpd",
    site_col: str = "site_name",
    min_bin_count: int = MIN_BIN_COUNT,
) -> pd.DataFrame:
    """Per-site effects + sensitivity for sites with >= ``min_valid_days`` valid rows."""
    rows: list[dict] = []
    for site, g in table.groupby(site_col, sort=True):
        valid = g.dropna(subset=[vpd_col, sm_col, response])
        if len(valid) < min_valid_days:
            continue
        eff = decouple_site(valid, response, n_bins, vpd_col, sm_col, min_bin_count)
        sens = sm_sensitivity(valid, response, n_bins, vpd_col, sm_col, min_bin_count)
        rec: dict = {site_col: site, "n_days": int(len(valid)), **eff, "sensitivity": sens}
        for c in _CARRY:
            if c in g.columns:
                vals = g[c].dropna()
                rec[c] = vals.iloc[0] if len(vals) else np.nan
        rows.append(rec)
    return pd.DataFrame.from_records(rows)


def dominance_summary(effects: pd.DataFrame) -> tuple[float, int]:
    """(% of valid sites where |sm_given_vpd| > |vpd_given_sm|, n valid sites).

    A site is valid only if both effects are finite.
    """
    valid = effects.dropna(subset=["sm_given_vpd", "vpd_given_sm"])
    n = len(valid)
    if n == 0:
        return float("nan"), 0
    sm_wins = (valid["sm_given_vpd"].abs() > valid["vpd_given_sm"].abs()).sum()
    return float(sm_wins) / n * 100.0, n
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest src/sm_vpd_decoupling/tests/test_aggregate.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sm_vpd_decoupling/aggregate.py src/sm_vpd_decoupling/tests/test_aggregate.py
git commit -m "feat(sm-vpd): per-site effects table + dominance summary"
```

---

### Task 8: Plotting (`plotting.py`)

**Files:**
- Create: `src/sm_vpd_decoupling/plotting.py`
- Test: `src/sm_vpd_decoupling/tests/test_plotting.py`

- [ ] **Step 1: Write the failing tests (smoke tests — files are written)**

```python
# src/sm_vpd_decoupling/tests/test_plotting.py
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from src.sm_vpd_decoupling.plotting import (
    plot_cross_site_aggregate,
    plot_depth_dominance,
    plot_example_sites,
    plot_gradient_violins,
    select_example_sites,
)


def _table(seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for site in ["A", "B"]:
        sm = rng.uniform(0.1, 0.35, 400)
        vpd = rng.uniform(0.5, 2.5, 400)
        frames.append(pd.DataFrame({
            "site_name": site, "vpd": vpd, "swvl1": sm,
            "E_norm": (sm - 0.1) / 0.25,
        }))
    return pd.concat(frames, ignore_index=True)


def test_plot_cross_site_aggregate(tmp_path):
    out = tmp_path / "agg.png"
    plot_cross_site_aggregate(_table(), response="E_norm", sm_col="swvl1",
                              n_bins=5, out_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_select_and_plot_example_sites(tmp_path):
    table = _table()
    # add an aridity column so selection spans a climate gradient
    table["aridity"] = np.where(table["site_name"] == "A", 0.3, 1.2)
    sites = select_example_sites(table, n=2)
    assert 1 <= len(sites) <= 2
    out = tmp_path / "examples.png"
    plot_example_sites(table, sites=sites, response="E_norm", sm_col="swvl1",
                       n_bins=5, out_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_depth_dominance(tmp_path):
    depth = pd.DataFrame({
        "sm_variant": ["swvl1", "swvl2", "swvl3", "swvl4", "root_zone_sm"],
        "pct_sm_dominant": [60.0, 65.0, 70.0, 72.0, 68.0],
        "n_sites": [40, 40, 40, 38, 40],
    })
    out = tmp_path / "depth.png"
    plot_depth_dominance(depth, response="E_norm", out_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_gradient_violins(tmp_path):
    effects = pd.DataFrame({
        "sm_given_vpd": np.random.default_rng(0).normal(-0.3, 0.1, 30),
        "aridity": np.random.default_rng(1).uniform(0.1, 1.5, 30),
        "pft": (["ENF"] * 15) + (["DBF"] * 15),
        "biome": (["temperate"] * 30),
        "canopy_height": np.random.default_rng(2).uniform(5, 35, 30),
    })
    out = tmp_path / "grad.png"
    plot_gradient_violins(effects, group_col="pft", out_path=str(out))
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest src/sm_vpd_decoupling/tests/test_plotting.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/sm_vpd_decoupling/plotting.py
"""Figures for the SM-VPD decoupling analysis (matplotlib, Agg-safe)."""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.sm_vpd_decoupling.decoupling import MIN_BIN_COUNT, assign_percentile_bins

logger = logging.getLogger(__name__)


def _binned_curve(df, axis_col, response, n_bins):
    """Mean response per percentile bin of ``axis_col`` (one site)."""
    bins = assign_percentile_bins(df[axis_col], n_bins)
    g = pd.DataFrame({"bin": bins, "resp": df[response]}).dropna()
    return g.groupby("bin")["resp"].mean()


def select_example_sites(table, n=4, sm_col="root_zone_sm"):
    """Pick up to ``n`` most-data sites spread across the aridity gradient.

    Bins sites by aridity (if present) and picks the most-data site per bin so
    the (a1) panel shows how the decoupling shape varies across climates.
    """
    counts = table.groupby("site_name").size().rename("n_days")
    meta = table.groupby("site_name")["aridity"].first() if "aridity" in table.columns else None
    if meta is None or meta.notna().sum() == 0:
        return counts.sort_values(ascending=False).head(n).index.tolist()
    df = pd.concat([counts, meta], axis=1).dropna(subset=["aridity"])
    df["arid_bin"] = pd.qcut(df["aridity"], q=min(n, df["aridity"].nunique()), duplicates="drop")
    picks = df.sort_values("n_days", ascending=False).groupby("arid_bin", observed=True).head(1)
    return picks.sort_values("aridity").index.tolist()


def plot_example_sites(table, sites, response, sm_col, n_bins, out_path):
    """(a1) Per-site small-multiples: Resp-vs-VPD binned by SM, and Resp-vs-SM
    binned by VPD, for the chosen example sites."""
    sites = list(sites) or [table["site_name"].iloc[0]]
    fig, axes = plt.subplots(len(sites), 2, figsize=(9, 3.2 * len(sites)), squeeze=False)
    for r, site in enumerate(sites):
        g = table[table["site_name"] == site].dropna(subset=[response, sm_col, "vpd"])
        vpd_by_sm = _binned_curve(g, sm_col, response, n_bins)
        sm_by_vpd = _binned_curve(g, "vpd", response, n_bins)
        axes[r][0].plot(vpd_by_sm.index, vpd_by_sm.values, "o-")
        axes[r][0].set_ylabel(f"{site}\n{response}")
        axes[r][0].set_xlabel("SM percentile bin")
        axes[r][1].plot(sm_by_vpd.index, sm_by_vpd.values, "s-", color="tab:orange")
        axes[r][1].set_xlabel("VPD percentile bin")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cross_site_aggregate(table, response, sm_col, n_bins, out_path):
    """(a2) Mean & median across sites of response per SM/VPD percentile bin."""
    sm_curves, vpd_curves = [], []
    for _site, g in table.groupby("site_name"):
        gg = g.dropna(subset=[response, sm_col, "vpd"])
        if len(gg) < MIN_BIN_COUNT * n_bins:
            continue
        sm_curves.append(_binned_curve(gg, sm_col, response, n_bins))
        vpd_curves.append(_binned_curve(gg, "vpd", response, n_bins))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, curves, label in ((axes[0], sm_curves, "SM percentile bin"),
                              (axes[1], vpd_curves, "VPD percentile bin")):
        if curves:
            mat = pd.concat(curves, axis=1)
            ax.plot(mat.index, mat.mean(axis=1), "o-", label="mean")
            ax.plot(mat.index, mat.median(axis=1), "s--", label="median")
        ax.set_xlabel(label)
        ax.set_ylabel(f"{response} (normalized)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_depth_dominance(depth_df, response, out_path):
    """(b) % of sites where SM dominates, across SM depth variants."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(depth_df["sm_variant"], depth_df["pct_sm_dominant"])
    for x, (pct, n) in enumerate(zip(depth_df["pct_sm_dominant"], depth_df["n_sites"])):
        ax.text(x, pct + 1, f"n={n}", ha="center", fontsize=8)
    ax.axhline(50, color="grey", ls=":")
    ax.set_ylabel("% sites SM-dominant")
    ax.set_title(f"SM-vs-VPD dominance by depth ({response})")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gradient_violins(effects, group_col, out_path):
    """(c) Violin of sm_given_vpd grouped by a categorical/binned column."""
    df = effects.dropna(subset=["sm_given_vpd", group_col]).copy()
    if group_col in ("aridity", "canopy_height"):
        df[group_col] = pd.cut(df[group_col], bins=5).astype(str)
    groups = sorted(df[group_col].unique())
    data = [df.loc[df[group_col] == gname, "sm_given_vpd"].values for gname in groups]
    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.2), 4))
    if any(len(d) > 0 for d in data):
        ax.violinplot([d for d in data if len(d) > 0], showmedians=True)
        ax.set_xticks(range(1, len([d for d in data if len(d) > 0]) + 1))
        ax.set_xticklabels([g for g, d in zip(groups, data) if len(d) > 0], rotation=30, ha="right")
    ax.axhline(0, color="grey", ls=":")
    ax.set_ylabel("ΔResp(SM|VPD)")
    ax.set_title(f"SM limitation by {group_col}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest src/sm_vpd_decoupling/tests/test_plotting.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/sm_vpd_decoupling/plotting.py src/sm_vpd_decoupling/tests/test_plotting.py
git commit -m "feat(sm-vpd): cross-site aggregate, depth dominance, gradient violin plots"
```

---

### Task 9: CLI orchestration (`run_sm_vpd_decoupling.py`) + analysis SLURM script

**Files:**
- Create: `src/sm_vpd_decoupling/run_sm_vpd_decoupling.py`
- Create: `src/sm_vpd_decoupling/job_sm_vpd_decoupling.sh`
- Test: `src/sm_vpd_decoupling/tests/test_run_integration.py`

- [ ] **Step 1: Write the failing integration test**

```python
# src/sm_vpd_decoupling/tests/test_run_integration.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.sm_vpd_decoupling.run_sm_vpd_decoupling import run_analysis


def _write_site_csv(path, site, seed):
    rng = np.random.default_rng(seed)
    n = 500
    sm = rng.uniform(0.1, 0.35, n)
    vpd = rng.uniform(0.4, 2.5, n)
    df = pd.DataFrame({
        "site_name": [site] * n,
        "TIMESTAMP": pd.date_range("2007-01-01", periods=n, freq="D"),
        "sap_velocity": 6.0 * sm + rng.normal(0, 0.05, n),
        "ta": rng.uniform(12.0, 28.0, n),
        "vpd": vpd,
        "ppfd_in": rng.uniform(400.0, 1300.0, n),
        "sw_in": rng.uniform(100.0, 500.0, n),
        "surface_solar_radiation_downwards_hourly": rng.uniform(100.0, 500.0, n),
        "volumetric_soil_water_layer_1": sm,
        "volumetric_soil_water_layer_2": sm * 0.9,
        "volumetric_soil_water_layer_3": sm * 0.8,
        "volumetric_soil_water_layer_4": sm * 0.7,
        "temperature_2m": rng.uniform(12.0, 28.0, n) + 273.15,
        "dewpoint_2m": rng.uniform(5.0, 15.0, n) + 273.15,
        "elevation": [150.0] * n,
        "pft": ["ENF"] * n,
        "biome": ["temperate"] * n,
        "prcip/PET": [0.7] * n,
        "canopy_height": [20.0] * n,
        "latitude_x": [46.0] * n,
        "longitude_x": [8.0] * n,
    })
    df.to_csv(path, index=False)


def test_run_analysis_end_to_end(tmp_path):
    daily = tmp_path / "daily"
    daily.mkdir()
    for i, s in enumerate(["A", "B", "C"]):
        _write_site_csv(daily / f"{s}_daily.csv", s, seed=i)
    out_dir = tmp_path / "out"
    run_analysis(data_dir=str(daily), out_dir=str(out_dir),
                 climate_source="site", tair_min=15.0,
                 n_bins_list=[5], min_valid_days_list=[120], make_figures=False)
    # Depth-profile dissociation table exists and covers both responses + 5 SM variants.
    depth = pd.read_csv(out_dir / "depth_profile.csv")
    assert set(depth["response"].unique()) == {"E_norm", "Gc_norm"}
    assert set(depth["sm_variant"].unique()) == {
        "swvl1", "swvl2", "swvl3", "swvl4", "root_zone_sm"}
    # Per-site CSV written for at least one combination.
    assert (out_dir / "per_site_E_norm_swvl1_nbins5_mvd120.csv").exists()
    # Attrition table exists.
    assert (out_dir / "attrition.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/sm_vpd_decoupling/tests/test_run_integration.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/sm_vpd_decoupling/run_sm_vpd_decoupling.py
"""CLI: orchestrate SM-VPD decoupling over responses x SM variants x bin counts
x min-valid-days. Writes per-site CSVs, a depth-profile dissociation table, an
attrition table, and figures.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.sm_vpd_decoupling.aggregate import decouple_all_sites, dominance_summary
from src.sm_vpd_decoupling.loader import (
    PPFD_MIN,
    VPD_MIN_KPA,
    load_table,
)
from src.sm_vpd_decoupling.plotting import (
    plot_cross_site_aggregate,
    plot_depth_dominance,
    plot_example_sites,
    plot_gradient_violins,
    select_example_sites,
)

logger = logging.getLogger(__name__)

SM_VARIANTS = ("swvl1", "swvl2", "swvl3", "swvl4", "root_zone_sm")
RESPONSES = ("E_norm", "Gc_norm")


def _attrition(table: pd.DataFrame, min_valid_days_list: list[int]) -> pd.DataFrame:
    """Sites & rows surviving each downstream threshold (post day-filter table)."""
    rows = [{"stage": "day_filtered", "n_sites": table["site_name"].nunique(),
             "n_rows": len(table)}]
    per_site_days = table.groupby("site_name").size()
    for mvd in min_valid_days_list:
        rows.append({"stage": f"min_valid_days>={mvd}",
                     "n_sites": int((per_site_days >= mvd).sum()),
                     "n_rows": int(per_site_days[per_site_days >= mvd].sum())})
    return pd.DataFrame(rows)


def run_analysis(
    data_dir: str | None,
    out_dir: str,
    climate_source: str = "site",
    tair_min: float = 15.0,
    n_bins_list: list[int] | None = None,
    min_valid_days_list: list[int] | None = None,
    make_figures: bool = True,
) -> None:
    n_bins_list = n_bins_list or [5, 10]
    min_valid_days_list = min_valid_days_list or [120, 240, 360]
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    table = load_table(data_dir, climate_source=climate_source, tair_min=tair_min)
    _attrition(table, min_valid_days_list).to_csv(out / "attrition.csv", index=False)

    depth_rows: list[dict] = []
    for response in RESPONSES:
        for sm_col in SM_VARIANTS:
            for n_bins in n_bins_list:
                for mvd in min_valid_days_list:
                    eff = decouple_all_sites(table, sm_col=sm_col, response=response,
                                             n_bins=n_bins, min_valid_days=mvd)
                    tag = f"{response}_{sm_col}_nbins{n_bins}_mvd{mvd}"
                    eff.to_csv(out / f"per_site_{tag}.csv", index=False)
                    pct, n = dominance_summary(eff)
                    depth_rows.append({
                        "response": response, "sm_variant": sm_col, "n_bins": n_bins,
                        "min_valid_days": mvd, "pct_sm_dominant": pct, "n_sites": n,
                        "mean_sm_given_vpd": eff["sm_given_vpd"].mean(),
                        "mean_vpd_given_sm": eff["vpd_given_sm"].mean(),
                    })
    depth = pd.DataFrame(depth_rows)
    depth.to_csv(out / "depth_profile.csv", index=False)

    if make_figures:
        _make_figures(table, depth, out, n_bins_list, min_valid_days_list)
    logger.info("Analysis complete -> %s", out)


def _make_figures(table, depth, out, n_bins_list, min_valid_days_list):
    nb, mvd = n_bins_list[0], min_valid_days_list[0]
    example_sites = select_example_sites(table, n=4)
    for response in RESPONSES:
        for sm_col in SM_VARIANTS:
            plot_cross_site_aggregate(
                table, response=response, sm_col=sm_col, n_bins=nb,
                out_path=str(out / "figures" / f"agg_{response}_{sm_col}.png"))
        # (a1) example-site small-multiples on the root-zone SM axis
        plot_example_sites(
            table, sites=example_sites, response=response, sm_col="root_zone_sm",
            n_bins=nb, out_path=str(out / "figures" / f"examples_{response}.png"))
        sub = depth[(depth["response"] == response) & (depth["n_bins"] == nb)
                    & (depth["min_valid_days"] == mvd)]
        if not sub.empty:
            plot_depth_dominance(sub, response=response,
                                 out_path=str(out / "figures" / f"depth_{response}.png"))
        eff = decouple_all_sites(table, sm_col="root_zone_sm", response=response,
                                 n_bins=nb, min_valid_days=mvd)
        for grp in ("pft", "biome", "aridity", "canopy_height"):
            if grp in eff.columns:
                plot_gradient_violins(
                    eff, group_col=grp,
                    out_path=str(out / "figures" / f"grad_{response}_{grp}.png"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Site-level SM-VPD decoupling.")
    p.add_argument("--data-dir", default=None, help="Daily CSV dir (auto-resolved if omitted)")
    p.add_argument("--out-dir", default="outputs/sm_vpd_decoupling")
    p.add_argument("--climate-source", choices=["site", "era5"], default="site")
    p.add_argument("--tair-min", type=float, default=15.0)
    p.add_argument("--n-bins", type=int, nargs="+", default=[5, 10])
    p.add_argument("--min-valid-days", type=int, nargs="+", default=[120, 240, 360])
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args()
    run_analysis(
        data_dir=args.data_dir, out_dir=args.out_dir, climate_source=args.climate_source,
        tair_min=args.tair_min, n_bins_list=args.n_bins,
        min_valid_days_list=args.min_valid_days, make_figures=not args.no_figures,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/sm_vpd_decoupling/tests/test_run_integration.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Write the analysis SLURM script**

Create `src/sm_vpd_decoupling/job_sm_vpd_decoupling.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=smvpd_decouple
#SBATCH --partition=requeue-zen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/smvpd_decouple_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/smvpd_decouple_%j.err

# Site-level SM-VPD decoupling: E (sap velocity) + Gc (Flo 2021) x 5 SM depths.
# Primary run: site-measured climate, Tair>15C. (Add a Tair>5C sensitivity run
# by re-invoking with --tair-min 5.0 --out-dir .../tair5.)
set -e
hostname; date
cd /scratch/tmp/yluo2/gsv
source .venv/bin/activate
export PYTHONPATH=/scratch/tmp/yluo2/gsv:$PYTHONPATH
python --version

python src/sm_vpd_decoupling/run_sm_vpd_decoupling.py \
    --climate-source site \
    --tair-min 15.0 \
    --n-bins 5 10 \
    --min-valid-days 120 240 360 \
    --out-dir outputs/sm_vpd_decoupling/tair15

echo "=== Output ==="
OUT=outputs/sm_vpd_decoupling/tair15
echo "Tables:"; ls "$OUT"/*.csv 2>/dev/null || true
echo "Figures:"; { ls "$OUT"/figures/*.png 2>/dev/null || true; } | wc -l
date
```

- [ ] **Step 6: Commit**

```bash
git add src/sm_vpd_decoupling/run_sm_vpd_decoupling.py src/sm_vpd_decoupling/job_sm_vpd_decoupling.sh src/sm_vpd_decoupling/tests/test_run_integration.py
git commit -m "feat(sm-vpd): CLI orchestration + analysis SLURM script"
```

---

### Task 10: Full suite + real-data verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole module test suite**

Run: `pytest src/sm_vpd_decoupling/ -v`
Expected: PASS (all tests across tasks 1–9)

- [ ] **Step 2: Check coverage**

Run: `pytest src/sm_vpd_decoupling/ --cov=src/sm_vpd_decoupling --cov-report=term-missing`
Expected: ≥80% coverage; note any uncovered lines.

- [ ] **Step 3: Produce the real dataset on Palma (sbatch — never on the login node)**

Run: `sbatch src/sm_vpd_decoupling/job_merge_decoupling.sh`
After completion, verify: `ls outputs/processed_data/sapwood/merged_decoupling/daily/*.csv | wc -l` (expect ~140+ site files).

- [ ] **Step 4: Run the analysis on Palma (sbatch)**

Run: `sbatch src/sm_vpd_decoupling/job_sm_vpd_decoupling.sh`
After completion, inspect `outputs/sm_vpd_decoupling/tair15/depth_profile.csv` (the E-vs-Gc dissociation across depths) and `attrition.csv` (confirm surviving site counts match the spec's expectation, ~16 at 360 days as a lower bound — daytime run should exceed it).

- [ ] **Step 5: Commit any verification notes / generated REPORT (if added)**

```bash
git add -A
git commit -m "chore(sm-vpd): real-data verification run notes"
```

---

## Self-Review

**Spec coverage:**
- Two responses (E, Gc) — Tasks 1, 5, 6, 9 ✓
- 5 SM variants incl. root-zone weights — Tasks 4, 9 ✓
- Liu day-filter (Tair 15/5, VPD>0.5, PPFD>500, AND) — Task 5 ✓
- PPFD fallback ppfd_in→sw_in→ERA5 — Task 4 ✓
- Per-site normalization (90th-pct anchor) — Task 4 ✓
- Nested-binning estimator (Eqs. 1–2), MIN_BIN_COUNT=3, ≥2 cond bins — Task 2 ✓
- Sensitivity δResp/δSM per 0.1 — Task 3 ✓
- Bin counts {5,10}, min-valid-days {120,240,360} sweep — Task 9 ✓
- Depth-profile dissociation table, attrition table, per-site CSVs — Tasks 7, 9 ✓
- Figures (a1) example sites, (a2) cross-site, (b) depth, (c) gradients incl. canopy-height — Task 8 ✓
- Treatment filter / daytime / no-growing-season / no gap-fill — Task 0 (merge flags) ✓
- Climate source site/era5 switch — Tasks 5, 9 ✓
- Gc 1/VPD confound documented — Task 1 ✓
- **Deferred (spec out-of-scope):** G′ aerodynamic conductance, Tair>5 run is a re-invocation (documented in job script), per-site Fan-2017 root depth. ✓

**Placeholder scan:** No TBD/TODO; all code blocks complete.

**Type consistency:** `decouple_site` returns `{"sm_given_vpd","vpd_given_sm"}` used identically in `aggregate.py`, `run`, and tests. `normalize_per_site`, `resolve_ppfd` (returns tuple), `canopy_conductance` signatures consistent across loader/tests. `decouple_all_sites` output columns match `dominance_summary` inputs and plotting expectations.
