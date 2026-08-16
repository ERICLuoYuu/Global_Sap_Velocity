"""Geometric plausibility of a measured sapwood area against its own DBH.

Sapwood is an annulus inside the stem cross-section, so a tree's measured
``pl_sapw_area`` and ``pl_dbh`` constrain each other::

    fraction = ASW / (pi * (DBH / 2) ** 2)

When that fraction lands far outside the range real stems occupy, at least one
of the two numbers carries the wrong unit. In log-log space such a tree is not
merely noisy: a DBH recorded in mm scales the fraction by 1/100, which is a
4.6-unit displacement in ``ln(ASW)``, so a handful of them can visibly bend a
global allometry fitted on ~2000 trees. This module refuses those pairs rather
than letting them vote.

Calibration
-----------
The bounds are read off the SAPFLUXNET v0.1.5 archive itself -- 1982 measured
trees across both scales, deduplicated -- not chosen a priori::

    min 1.03%   p0.1 1.87%   p2.5 10.68%   p50 60.98%   p99 115.29%   max 509.48%

:data:`MIN_SAPWOOD_FRACTION` therefore sits at the archive floor. It rejects
nothing the archive contains, and fires only on values an order of magnitude
below anything a provider has ever reported -- the signature of a unit error,
not of an unusual tree.

The upper side is deliberately **not** a rejection. ``ASW > basal area`` is
geometrically impossible, yet 3.9% of archive trees report it, and eight sites
(ESP_CA, IDN_PON_ST, GBR_ABE_PL, the ESP_YUN plots, ...) set ASW to the whole
cross-section for every tree -- a convention Poyatos et al. (2021) fitted on.
Rejecting it here would hold new data to a stricter standard than the published
relationship this package reproduces, so it is counted and reported instead.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

#: Below this fraction of basal area a measured ASW/DBH pair is refused.
#: Equal to the SAPFLUXNET v0.1.5 archive minimum (1.03%), see module docstring.
MIN_SAPWOOD_FRACTION = 0.01

#: Above this fraction the pair is geometrically impossible but *tolerated*,
#: because the archive itself is full of it. Counted, never dropped.
OVERSIZED_SAPWOOD_FRACTION = 1.0


def basal_area(dbh: pd.Series) -> pd.Series:
    """Stem cross-sectional area (cm²) from DBH (cm). Non-positive DBH -> NaN."""
    positive = dbh.where(dbh > 0)
    return math.pi * (positive / 2.0) ** 2


def sapwood_fraction(asw: pd.Series, dbh: pd.Series) -> pd.Series:
    """Measured sapwood area as a fraction of basal area.

    NaN wherever the fraction cannot be formed (missing ASW, missing or
    non-positive DBH), so callers can distinguish "implausible" from "unknown".
    """
    return asw.where(asw > 0) / basal_area(dbh)


def implausible_mask(asw: pd.Series, dbh: pd.Series) -> pd.Series:
    """True where the ASW/DBH pair is too small to be a real stem.

    Only ``False`` for pairs that are actually checkable -- a tree missing
    either value is *not* implausible, it is simply unmeasured, and must stay
    eligible for imputation.
    """
    fraction = sapwood_fraction(asw, dbh)
    return fraction.notna() & (fraction < MIN_SAPWOOD_FRACTION)


def oversized_mask(asw: pd.Series, dbh: pd.Series) -> pd.Series:
    """True where ASW exceeds basal area. Reported, not rejected."""
    fraction = sapwood_fraction(asw, dbh)
    return fraction.notna() & (fraction > OVERSIZED_SAPWOOD_FRACTION)


def is_implausible(asw: float, dbh: float) -> bool:
    """Scalar form of :func:`implausible_mask`, for per-row estimation."""
    if asw is None or dbh is None:
        return False
    try:
        asw_f, dbh_f = float(asw), float(dbh)
    except (TypeError, ValueError):
        return False
    if not (np.isfinite(asw_f) and np.isfinite(dbh_f)) or asw_f <= 0 or dbh_f <= 0:
        return False
    return asw_f / (math.pi * (dbh_f / 2.0) ** 2) < MIN_SAPWOOD_FRACTION


def describe(asw: pd.Series, dbh: pd.Series) -> dict:
    """Summarise a corpus against the gate, for the run log and JSON report."""
    fraction = sapwood_fraction(asw, dbh)
    checkable = fraction.notna()
    return {
        "checkable_trees": int(checkable.sum()),
        "implausible": int(implausible_mask(asw, dbh).sum()),
        "oversized": int(oversized_mask(asw, dbh).sum()),
        "min_fraction": float(fraction.min()) if checkable.any() else None,
        "median_fraction": float(fraction.median()) if checkable.any() else None,
        "max_fraction": float(fraction.max()) if checkable.any() else None,
    }
