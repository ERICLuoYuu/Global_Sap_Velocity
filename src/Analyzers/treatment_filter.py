"""Treatment-based plant filter for the global sap-velocity training set.

Exclusion criterion: **predictor-target decoupling**. Drop only plants whose
experimental treatment changes local water / energy forcing in a way that
ERA5-Land / WorldClim / SoilGrids cannot see at 0.1 degree resolution. Keep
thinning, fertilization, plantation composition, post-fire recovery, age
classes, and other stand-state variations — those are legitimate states the
global model must learn to handle, and they are (at least partially)
encoded in our existing predictors (LAI, canopy height, PFT, stand age).

Label classification uses a precedence chain:

    CONTROL tokens  (keep) > SAMPLING_ONLY tokens (unknown — fall through)
                          > DROP tokens      (drop)
                          > default          (keep)

Plant-level label is trusted first; stand-level label is used only when
the plant label is unknown. A hard site-override map handles the known
edge-case sites (EucFACE ambient rings, SEN_SOU baseline phases) where
stand-level labels are semantically misleading.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Literal

import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Token lists
# --------------------------------------------------------------------------

# Control-like labels. ALWAYS classified as 'keep', even if they contain a
# drop-token substring (e.g. 'Pre Irrigation' is a baseline phase with no
# water added). Precedence: CONTROL is checked first.
CONTROL_TOKENS: tuple[str, ...] = (
    "ambient",
    "control",
    "reference",
    "non thin",
    "non-thin",
    "non defol",
    "non-defol",
    "pre thin",
    "pre-thin",
    "before thin",
    "pre irrigat",  # baseline phase bracketing an irrigation experiment
    "post irrigat",
    "premortality",  # baseline before tree death
)

# Decoupling labels. Plants with these in pl_treatment or st_treatment
# have water or energy forcing that gridded predictors cannot see.
DROP_TOKENS: tuple[str, ...] = (
    "irrig",  # any active irrigation
    "drought",
    "throughfall exclu",
    "trench",  # root trenching
    "elevated atmospheric co2",
    "elevated co2",
    "shade",
    "radiation level",
)

# Sampling artefacts — not a "treatment" in the water/energy sense.
# Falls through to stand-level decision.
SAMPLING_ONLY: tuple[str, ...] = (
    "increment cores",
    "destructive sampling",
    "distructive sampling",  # typo seen in real SAPFLUXNET data
)

# Hard site overrides where label semantics are misleading.
# - AUS_RIC_EUC_ELE: EucFACE facility but sampled trees are all in ambient
#   rings R2/R3/R6 (pl_treatment='Ambient CO2'); stand label says 'Elevated
#   atmospheric CO2' because the facility itself is a CO2 experiment.
# - ESP_SEN_SOU_PRE/POS: 'Pre Irrigation' / 'Post Irrigation' are baseline
#   phases bracketing an experiment, with no water added during the
#   SAPFLUXNET contribution.
SITE_OVERRIDES: dict[str, str] = {
    "AUS_RIC_EUC_ELE": "keep_all",
    "ESP_SEN_SOU_PRE": "keep_all",
    "ESP_SEN_SOU_POS": "keep_all",
}

_TIMESTAMP_COLUMNS: frozenset[str] = frozenset({"TIMESTAMP", "TIMESTAMP_LOCAL", "TIMESTAMP_solar", "solar_TIMESTAMP"})


def _validate_token_lists() -> None:
    """Catch maintenance regressions where a CONTROL_TOKEN becomes a substring of
    a DROP_TOKEN. Such an overlap would silently break the precedence rule
    because ``in`` is a substring test, not a word match. The reverse (a
    DROP_TOKEN substring of a CONTROL_TOKEN, e.g. 'irrig' inside 'pre irrigat')
    is intentional and safe because CONTROL is checked first.
    Runs at import time so it fires before any caller can be misled.
    """
    for ct in CONTROL_TOKENS:
        for dt in DROP_TOKENS:
            if ct in dt:
                raise AssertionError(
                    f"CONTROL_TOKEN {ct!r} is a substring of DROP_TOKEN {dt!r} — this would silently break precedence."
                )


_validate_token_lists()


Decision = Literal["keep", "drop", "unknown"]


# --------------------------------------------------------------------------
# Label normalization + classification
# --------------------------------------------------------------------------


def _normalize(raw: Any) -> str:
    """Lowercase + strip a label; returns empty string for NA-likes."""
    if raw is None:
        return ""
    if isinstance(raw, float) and math.isnan(raw):
        return ""
    text = str(raw).strip().lower()
    if text in ("", "na", "nan", "none"):
        return ""
    return text


def classify_label(raw: Any) -> Decision:
    """Classify a single ``pl_treatment`` or ``st_treatment`` string.

    Precedence: CONTROL > SAMPLING_ONLY > DROP > default-keep.

    Returns
    -------
    'keep'
        Label is a control/reference/baseline (e.g. 'Control', 'Ambient',
        'Pre Irrigation') OR is not matched by any token list (default).
    'drop'
        Label contains a decoupling token (e.g. 'Irrigation', 'drought_1',
        'Root trenching', 'Elevated atmospheric CO2').
    'unknown'
        Label is empty / NA / None / nan, or is a sampling-only token
        ('Increment cores'). Caller should fall back to the next level
        of labelling (plant → stand).
    """
    text = _normalize(raw)
    if not text:
        return "unknown"

    if any(tok in text for tok in CONTROL_TOKENS):
        return "keep"

    if any(tok in text for tok in SAMPLING_ONLY):
        return "unknown"

    if any(tok in text for tok in DROP_TOKENS):
        return "drop"

    return "keep"


def classify_plant(pl_treatment: Any, st_treatment: Any) -> Literal["keep", "drop"]:
    """Plant-level decision with stand fallback.

    Logic
    -----
    1. Classify the plant-level label. If 'keep' or 'drop', return it.
    2. Otherwise (plant label is 'unknown' — empty or sampling-only),
       classify the stand-level label. If that says 'drop', drop the plant;
       otherwise keep it.

    A stand 'drop' never overrides a plant 'keep' — this protects the
    EucFACE ambient-ring case where the facility's stand label is
    'Elevated atmospheric CO2' but the sampled trees are actually in
    ambient rings.
    """
    plant_decision = classify_label(pl_treatment)
    if plant_decision in ("keep", "drop"):
        return plant_decision

    stand_decision = classify_label(st_treatment)
    if stand_decision == "drop":
        return "drop"
    return "keep"


# --------------------------------------------------------------------------
# DataFrame-level filter
# --------------------------------------------------------------------------


def _stand_treatment_for(stand_md: pd.DataFrame | None, si_code: Any) -> Any:
    """Return st_treatment for the given si_code, or None if not found."""
    if stand_md is None or len(stand_md) == 0 or si_code is None:
        return None
    if isinstance(si_code, float) and math.isnan(si_code):
        return None
    mask = stand_md["si_code"] == si_code
    matches = stand_md.loc[mask, "st_treatment"]
    if len(matches) == 0:
        return None
    return matches.iloc[0]


def filter_by_treatment(
    sapf_wide: pd.DataFrame,
    plant_md: pd.DataFrame | None,
    stand_md: pd.DataFrame | None,
    *,
    site_code: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Drop plant columns whose treatment classifies as 'drop'.

    Parameters
    ----------
    sapf_wide
        Wide-format sap flow DataFrame with one column per plant plus
        timestamp columns.
    plant_md
        Plant-level metadata with ``pl_code``, ``pl_treatment``, ``si_code``.
        When ``None`` or empty, the function is a no-op — every plant is
        kept. This lets internal ICOS sites (which lack plant_md) pass
        through.
    stand_md
        Stand-level metadata with ``si_code``, ``st_treatment``. When
        ``None`` or empty, stand fallback is skipped.
    site_code
        Optional site code used to look up ``SITE_OVERRIDES``. When the
        site has a ``'keep_all'`` override, every plant is retained
        regardless of label. Pass ``None`` to disable the override check.

    Returns
    -------
    filtered
        New DataFrame with dropped plant columns removed. Non-plant
        (timestamp) columns are always preserved.
    report
        Dict with keys:
            - ``'kept'``: list[str] of surviving plant codes
            - ``'dropped'``: list[str] of removed plant codes
            - ``'reason_by_plant'``: dict mapping each dropped plant code
              to a human-readable reason (e.g. 'pl_treatment=Irrigation')
    """
    out = sapf_wide.copy(deep=True)
    report: dict = {"kept": [], "dropped": [], "reason_by_plant": {}}

    if plant_md is None or len(plant_md) == 0:
        # ICOS pass-through: keep every plant column.
        report["kept"] = [c for c in out.columns if c not in _TIMESTAMP_COLUMNS]
        return out, report

    # Whole-site override short-circuit
    if site_code is not None and SITE_OVERRIDES.get(site_code) == "keep_all":
        logger.info(
            "Site %r matches SITE_OVERRIDES=keep_all; retaining all plants.",
            site_code,
        )
        report["kept"] = [c for c in out.columns if c not in _TIMESTAMP_COLUMNS]
        return out, report

    md_by_code = plant_md.set_index("pl_code", drop=False)
    if md_by_code.index.duplicated().any():
        dupes = md_by_code.index[md_by_code.index.duplicated()].unique().tolist()
        logger.warning(
            "Duplicate pl_code entries in plant_md (%s); keeping the first row for each.",
            dupes,
        )
        md_by_code = md_by_code[~md_by_code.index.duplicated(keep="first")]
    plant_columns = [c for c in out.columns if c not in _TIMESTAMP_COLUMNS]

    cols_to_drop: list[str] = []

    for col in plant_columns:
        if col not in md_by_code.index:
            logger.warning(
                "Plant column %r present in sap_data but missing from plant_md; defaulting to keep.",
                col,
            )
            report["kept"].append(col)
            continue

        row = md_by_code.loc[col]
        pl_treatment = row.get("pl_treatment")
        si = row.get("si_code")
        st_treatment = _stand_treatment_for(stand_md, si)

        decision = classify_plant(pl_treatment, st_treatment)
        if decision == "drop":
            cols_to_drop.append(col)
            # Record the first-matching level's label as the drop reason.
            if classify_label(pl_treatment) == "drop":
                reason = f"pl_treatment={pl_treatment}"
            else:
                reason = f"st_treatment={st_treatment}"
            report["reason_by_plant"][col] = reason
            report["dropped"].append(col)
        else:
            report["kept"].append(col)

    if cols_to_drop:
        out = out.drop(columns=cols_to_drop)

    return out, report
