"""Flo 2019 calibration correction for thermal-dissipation sap flow methods.

Flo et al. 2019 (Agricultural and Forest Meteorology 271, 362–374, Table 1)
report that Dissipation-family methods — Thermal Dissipation (TD, a.k.a. HD
or Granier) and Transient Thermal Dissipation (TTD, a.k.a. CHD) — systematically
underestimate sap flux density. The paper recommends using the reported
accuracy deviations as first-order corrections when combining sap flow data
from different studies.

We invert the Ln-Ratio bias so the corrected value ≈ the true SFD:

    multiplier = exp(-Ln-Ratio)

    HD  (TD):  Ln-Ratio = -0.519 → multiplier = exp(0.519) ≈ 1.6804
    CHD (TTD): Ln-Ratio = -0.493 → multiplier = exp(0.493) ≈ 1.6373

The correction is applied only to plants whose ``pl_sens_calib`` flag is
NOT truthy — i.e. plants that do not already carry a study-specific
calibration. All other sensor methods (HR, HPTM, CHP, HFD, SHB, TSHB) pass
through unchanged, consistent with Flo 2019 Table 1 reporting their Ln-Ratio
not significantly different from zero.

This module is pure-functional and immutable: it never mutates caller data.
"""

from __future__ import annotations

import logging
import math

import pandas as pd

logger = logging.getLogger(__name__)


# Flo et al. 2019, Table 1. Multiplier = exp(-Ln-Ratio) to invert bias.
FLO2019_HD_MULTIPLIER: float = math.exp(0.519)  # ≈ 1.6804, from Ln-Ratio -0.519
FLO2019_CHD_MULTIPLIER: float = math.exp(0.493)  # ≈ 1.6373, from Ln-Ratio -0.493

FLO2019_MULTIPLIERS: dict[str, float] = {
    "HD": FLO2019_HD_MULTIPLIER,
    "CHD": FLO2019_CHD_MULTIPLIER,
}

# Case-insensitive tokens for pl_sens_calib that indicate a calibrated sensor.
_CALIB_TRUE_TOKENS: frozenset[str] = frozenset({"TRUE", "T", "1", "YES", "Y"})

# Timestamp-like columns to always preserve.
_TIMESTAMP_COLUMNS: frozenset[str] = frozenset({"TIMESTAMP", "TIMESTAMP_LOCAL", "TIMESTAMP_solar", "solar_TIMESTAMP"})


def _is_calibrated(val: object) -> bool:
    """Return True iff ``val`` represents an explicitly-calibrated sensor flag.

    Handles bool, string, int, float(NaN), and None. Matching is case-insensitive.
    """
    if val is None:
        return False
    if isinstance(val, float) and math.isnan(val):
        return False
    if isinstance(val, bool):
        return val
    try:
        return str(val).strip().upper() in _CALIB_TRUE_TOKENS
    except Exception:  # noqa: BLE001
        return False


def apply_flo2019_correction(
    sapf_wide: pd.DataFrame,
    plant_md: pd.DataFrame | None,
    *,
    multipliers: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Multiply SFD of non-calibrated HD/CHD plant columns by per-method multipliers.

    Parameters
    ----------
    sapf_wide
        Wide-format sap flow DataFrame with one column per plant plus one or
        more TIMESTAMP columns. Returned copy leaves non-plant columns
        untouched.
    plant_md
        Plant-level metadata with columns ``pl_code``, ``pl_sens_meth``, and
        ``pl_sens_calib``. When ``None`` or empty, the function is a no-op —
        this lets internal ICOS sites (which lack plant_md) pass through.
    multipliers
        Optional override for the per-method multiplier dict. Defaults to
        ``FLO2019_MULTIPLIERS``. Keys are sensor method codes (e.g. 'HD',
        'CHD'); values are scalar multipliers.

    Returns
    -------
    corrected
        New DataFrame with the same shape and columns as ``sapf_wide``.
        Original input is not mutated.
    audit
        List of per-plant records ``{'pl_code': str, 'method': str,
        'multiplier': float}`` describing every correction applied. Empty
        when the function was a no-op.
    """
    out = sapf_wide.copy(deep=True)

    if plant_md is None or len(plant_md) == 0:
        return out, []

    active_multipliers = dict(multipliers) if multipliers is not None else dict(FLO2019_MULTIPLIERS)

    plant_columns = [c for c in out.columns if c not in _TIMESTAMP_COLUMNS]
    md_by_code = plant_md.set_index("pl_code", drop=False)
    if md_by_code.index.duplicated().any():
        dupes = md_by_code.index[md_by_code.index.duplicated()].unique().tolist()
        logger.warning(
            "Duplicate pl_code entries in plant_md (%s); keeping the first row for each.",
            dupes,
        )
        md_by_code = md_by_code[~md_by_code.index.duplicated(keep="first")]

    audit: list[dict] = []

    for col in plant_columns:
        if col not in md_by_code.index:
            logger.warning(
                "Plant column %r present in sap_data but missing from plant_md; passing through unchanged.",
                col,
            )
            continue

        row = md_by_code.loc[col]
        method = row.get("pl_sens_meth")
        calib = row.get("pl_sens_calib")

        if not isinstance(method, str):
            continue

        method_key = method.strip().upper()
        if method_key not in active_multipliers:
            continue

        if _is_calibrated(calib):
            continue

        multiplier = active_multipliers[method_key]
        out[col] = out[col] * multiplier
        audit.append({"pl_code": col, "method": method_key, "multiplier": multiplier})

    return out, audit
