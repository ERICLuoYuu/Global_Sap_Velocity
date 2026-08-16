"""Tell a *measured* sapwood area from one the provider computed.

Many SAPFLUXNET contributors did not measure ASW on every tree. They measured a
subset (or none) and filled the rest from an allometry of their own -- typically
``ASW = a * DBH^b`` fitted at their site, sometimes simply ``ASW = basal area``.
Those values are predictions, and fitting *our* DBH->ASW allometry to them is
circular: we would be regressing on the output of other regressions.

The bias is not the obvious one. A derived tree lies *exactly* on its own site's
line, so pooling many of them does not add noise -- it adds structure that has
already absorbed the species and functional-type signal into each site's private
coefficients. Measured on the real corpus, that dilution is severe: the
gymnosperm offset collapses from +0.221 to +0.067 (a x1.25 group ratio reported
as x1.07) and the functional type's marginal R2 drops from +0.008 to +0.001. The
"functional type is worth almost nothing" reading of this dataset is an artifact
of derived trees, not a property of trees.

Detection
---------
Two signatures, both computed per *site x species* group with at least
:data:`MIN_GROUP_SIZE` trees spanning that many distinct DBH values:

``allometry_derived``
    ``ln(ASW) ~ ln(DBH)`` within the group has R2 >= :data:`DERIVED_R2`. Real
    within-species sapwood scatter runs R2 ~ 0.7-0.95; 0.9999 is a formula.
    Corroborating evidence where provider workbooks are available: AT-Zoe ships
    an EMPTY ``tree_sapwood_thickness_cm`` beside a ``tree_sapwood_area`` of
    ``1632.1082144631541`` -- 13 decimals is float output, not a ruler.

``assumed_basal_area``
    The group's median ASW is the whole stem cross-section (fraction ~ 1.0 with
    a fitted exponent ~ 2.0). The provider had no sapwood depth and used the
    entire section. Distinguished from the above because the exponent is
    pinned by geometry rather than fitted.

Groups too small to assess are **kept**. Absence of evidence is not evidence of
derivation, and discarding every site with three trees would cost far more than
it protects.

Provider-declared estimates
---------------------------
Statistics cannot catch an estimate that does not track DBH. ES-Gdn assigns just
two distinct ASW values (10.05, 10.68) across four olives spanning 107-150 cm,
with a flat 1 cm assumed sapwood depth -- clearly not measured, but no
regression will say so. :data:`ESTIMATED_SITES` records such cases from the
provider documentation, keyed by site code and matched case/separator-insensitively.

Detecting is not excluding
--------------------------
The two statistical signatures are an *inference* from a threshold; only
:data:`ESTIMATED_SITES` rests on a document. Those are different grades of
evidence, so the caller chooses how far to act via :func:`excluded_verdicts`:
``POLICY_DECLARED`` (the default) bars only the documented sites and merely
*reports* the rest, ``POLICY_ALL`` bars everything detected, ``POLICY_NONE``
bars nothing and reproduces the published Poyatos pool.

Reporting without excluding is the conservative choice, and it is deliberate:
"36% of the archive looks derived" is a finding a reader can weigh, whereas
silently deleting 778 trees on a chosen R2 cut is a claim they must simply
accept. The cost is a real one -- see :func:`excluded_verdicts` and the
functional-type dilution described above, which ``POLICY_DECLARED`` leaves in
place.

What this module never does
---------------------------
It does not delete or overwrite anybody's value. A provider estimate is still
the best number available for that tree, and it is what their published sap flux
density was computed against -- replacing it would silently desynchronise the two.
Barred trees are dropped from the FIT and flagged in the output; their
``pl_sapw_area`` is preserved untouched.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Verdicts. Only the first is a positive finding of measurement.
MEASURED = "measured"
ALLOMETRY_DERIVED = "allometry_derived"
ASSUMED_BASAL_AREA = "assumed_basal_area"
PROVIDER_DECLARED = "provider_declared_estimate"
NOT_ASSESSED = "not_assessed"

#: Verdicts that are a positive finding of estimation, by any route.
ESTIMATED_VERDICTS = frozenset({ALLOMETRY_DERIVED, ASSUMED_BASAL_AREA, PROVIDER_DECLARED})

#: Which of those findings actually bar a tree from the fit. Detection and
#: exclusion are deliberately separate decisions: the signatures below are an
#: *inference* from a threshold, while :data:`ESTIMATED_SITES` rests on the
#: provider's own workbook. The default acts only on the documented cases and
#: reports the rest, so the artifact stays defensible without a reviewer having
#: to accept a chosen R2 cut.
POLICY_DECLARED = "declared"
POLICY_ALL = "all"
POLICY_NONE = "none"
EXCLUSION_POLICIES = (POLICY_DECLARED, POLICY_ALL, POLICY_NONE)


def excluded_verdicts(policy: str) -> frozenset[str]:
    """Verdicts that ``policy`` bars from the fit. Raises on an unknown policy."""
    if policy == POLICY_NONE:
        return frozenset()
    if policy == POLICY_DECLARED:
        return frozenset({PROVIDER_DECLARED})
    if policy == POLICY_ALL:
        return ESTIMATED_VERDICTS
    raise ValueError(f"unknown exclusion policy {policy!r}; expected one of {EXCLUSION_POLICIES}")


#: Minimum trees (and distinct DBH values) before a group can be judged.
MIN_GROUP_SIZE = 4

#: Within-group R2 at or above which ln(ASW)~ln(DBH) is a formula, not a measurement.
#: Real within-species scatter does not reach this; see the module docstring.
DERIVED_R2 = 0.9999

#: Tolerance for recognising ASW == basal area (fraction ~ 1, exponent ~ 2).
BASAL_AREA_TOLERANCE = 0.02

#: Sites whose provider documentation states the ASW is estimated, where the
#: statistical signatures cannot see it. Value is the evidence, for the report.
ESTIMATED_SITES = {
    "ESP_GDN": (
        "two distinct ASW values (10.05, 10.68) across four olives spanning "
        "DBH 107-150 cm, with a flat assumed 1 cm sapwood depth; ASW does not "
        "track DBH at all, and describes the instrumented branch rather than "
        "the stem the DBH refers to"
    ),
    "AUT_ZOE": (
        "ZOE_AT_Metadata.xlsx leaves tree_sapwood_thickness_cm entirely EMPTY "
        "while reporting tree_sapwood_area to 13 decimals "
        "(1632.1082144631541); nothing was measured to derive an area from. "
        "Registered explicitly because the statistical test only reaches its "
        "4 Fagus trees -- the 3 Acer form a group below MIN_GROUP_SIZE and "
        "would otherwise slip through from the same derived source"
    ),
}

_ESTIMATED_INDEX = {re.sub(r"[^A-Z0-9]+", "_", k.upper()).strip("_"): k for k in ESTIMATED_SITES}


def declared_estimate(site: str) -> str | None:
    """Provider-declared evidence for ``site``, or ``None``."""
    key = re.sub(r"[^A-Z0-9]+", "_", str(site).upper()).strip("_")
    declared = _ESTIMATED_INDEX.get(key)
    return ESTIMATED_SITES[declared] if declared else None


@dataclass(frozen=True)
class GroupVerdict:
    """How one site x species group's sapwood area came to exist."""

    site: str
    species: str
    verdict: str
    n: int
    r2: float
    slope: float
    median_fraction: float

    @property
    def is_estimated(self) -> bool:
        return self.verdict in ESTIMATED_VERDICTS


def _judge(dbh: np.ndarray, asw: np.ndarray) -> tuple[str, float, float, float]:
    """Return (verdict, r2, slope, median_fraction) for one group."""
    fraction = float(np.median(asw / (math.pi * (dbh / 2.0) ** 2)))
    if len(dbh) < MIN_GROUP_SIZE or len(np.unique(dbh)) < MIN_GROUP_SIZE:
        return NOT_ASSESSED, float("nan"), float("nan"), fraction

    x, y = np.log(dbh), np.log(asw)
    slope, intercept = np.polyfit(x, y, 1)
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst <= 0:
        return NOT_ASSESSED, float("nan"), float(slope), fraction
    r2 = 1.0 - float(np.sum((y - (intercept + slope * x)) ** 2)) / sst

    # Order matters: ASW == basal area also produces R2 == 1, but its exponent is
    # fixed by geometry rather than fitted, and it is a different provider act.
    if abs(fraction - 1.0) < BASAL_AREA_TOLERANCE and abs(slope - 2.0) < BASAL_AREA_TOLERANCE:
        return ASSUMED_BASAL_AREA, r2, float(slope), fraction
    if r2 >= DERIVED_R2:
        return ALLOMETRY_DERIVED, r2, float(slope), fraction
    return MEASURED, r2, float(slope), fraction


def assess(
    df: pd.DataFrame,
    site_col: str,
    species_col: str,
    dbh_col: str,
    asw_col: str,
) -> list[GroupVerdict]:
    """Judge every site x species group that has a usable ASW/DBH pair."""
    usable = df[df[asw_col].notna() & df[dbh_col].notna() & (df[asw_col] > 0) & (df[dbh_col] > 0)]
    verdicts: list[GroupVerdict] = []
    for (site, species), group in usable.groupby([site_col, species_col]):
        declared = declared_estimate(site)
        dbh = group[dbh_col].to_numpy(dtype=float)
        asw = group[asw_col].to_numpy(dtype=float)
        verdict, r2, slope, fraction = _judge(dbh, asw)
        if declared is not None:
            # A provider statement outranks any inference from the numbers.
            verdict = PROVIDER_DECLARED
        verdicts.append(
            GroupVerdict(
                site=str(site),
                species=str(species),
                verdict=verdict,
                n=len(group),
                r2=r2,
                slope=slope,
                median_fraction=fraction,
            )
        )
    return verdicts


def estimated_mask(
    df: pd.DataFrame,
    site_col: str,
    species_col: str,
    dbh_col: str,
    asw_col: str,
    policy: str = POLICY_DECLARED,
) -> pd.Series:
    """True for rows whose ASW ``policy`` bars from the fit.

    Rows that cannot be judged (missing site column, group too small, no ASW)
    come back ``False``: this flags positive findings only.
    """
    barred = excluded_verdicts(policy)  # validates the policy even when unused
    if not barred:
        return pd.Series(False, index=df.index)
    if site_col not in df.columns or species_col not in df.columns:
        # Without both keys the groups cannot be formed. Fall back to the
        # declared registry alone if a site column exists at all.
        if site_col in df.columns:
            return df[site_col].map(lambda s: declared_estimate(s) is not None).fillna(False).astype(bool)
        return pd.Series(False, index=df.index)

    flagged = {(v.site, v.species) for v in assess(df, site_col, species_col, dbh_col, asw_col) if v.verdict in barred}
    if not flagged:
        return pd.Series(False, index=df.index)
    # No zip(strict=): Palma runs Python 3.9. Both are columns of the same
    # frame, so equal length is structural rather than something to assert.
    keys = pd.Series(
        list(zip(df[site_col].astype(str), df[species_col].astype(str))),  # noqa: B905
        index=df.index,
    )
    return keys.map(lambda k: k in flagged).astype(bool)


def summarise(verdicts: list[GroupVerdict]) -> dict:
    """Counts by verdict plus the flagged sites, for the run log and JSON report."""
    by_verdict: dict[str, dict[str, int]] = {}
    for v in verdicts:
        entry = by_verdict.setdefault(v.verdict, {"groups": 0, "trees": 0})
        entry["groups"] += 1
        entry["trees"] += v.n
    sites: dict[str, list[str]] = {}
    for v in verdicts:
        if v.is_estimated:
            sites.setdefault(v.verdict, [])
            if v.site not in sites[v.verdict]:
                sites[v.verdict].append(v.site)
    return {
        "thresholds": {
            "min_group_size": MIN_GROUP_SIZE,
            "derived_r2": DERIVED_R2,
            "basal_area_tolerance": BASAL_AREA_TOLERANCE,
        },
        "by_verdict": by_verdict,
        "estimated_sites": {k: sorted(v) for k, v in sites.items()},
    }
