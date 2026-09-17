"""Write-time invariants of the daily env statistics, and the radiation gate on a finished tree.

:func:`check_support` re-checks the rule of :mod:`src.env_daily.support` on the statistics about to be
written. A violation is a programming error, not a messy site, so it raises -- and the merge re-raises it
past its per-site catch-all, which would otherwise print one line and drop the site.

:func:`kt_exceedance_days` counts site-days on which daily surface shortwave exceeds daily top-of-atmosphere
radiation, which a correct daily mean cannot do. On the 2026-09-16 tree there were 2,141 such days; under the
new rule the probe (Palma 46757405) left 221, at ten sites whose cause is a different defect (FIN_HYY_SME's
constant ext_rad offset, sensor calibration at the others). :func:`radiation_gate` holds that line: a total
above the baseline, any such day at a site outside :data:`KT_RESIDUAL_SITES`, or a residual site above its
own recorded count, fails.

The probe counted under ``served > 5``, and the baseline below is that count, so the gate applies the same
floor -- see :data:`KT_MIN_EXT_RAD`. It belongs to the gate and not to the data: nothing here changes what the
merge writes, so it is deliberately absent from :func:`env_daily_rule`, which records how a tree was built.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from src.env_daily.support import (
    BODY_SAMPLE_MINUTES,
    DARK_SAMPLE_MINUTES,
    ENV_CONTINUOUS_COLS,
    ENV_RADIATION_COLS,
    EXPECTED_HOURS_COL,
    HOURS_PER_DAY,
    MIN_ENV_HOURS,
    MIN_TOA_COVERAGE,
    RULE_VERSION,
)

#: kt > 1 site-days left by the new rule on the 2026-09-16 servable tree (probe_support_rules, 46757405).
KT_RESIDUAL_SITES: Mapping[str, int] = MappingProxyType(
    {
        "FIN_HYY_SME": 155,
        "USA_BNZ_BLA": 18,
        "KOR_TAE_TC1_LOW": 11,
        "KOR_TAE_TC2_MED": 11,
        "KOR_TAE_TC3_EXT": 11,
        "RUS_FYO": 6,
        "CAN_TUR_P39_POS": 3,
        "CAN_TUR_P39_PRE": 3,
        "SWE_NOR_ST3": 2,
        "SWE_SVA_MIX_NON": 1,
    }
)
KT_BASELINE_TOTAL = 221

#: Daily-mean TOA below which the clearness ratio is not a measurement. Either side of polar night both terms
#: fall under a pyranometer's resolution and the quotient stops meaning anything: at RUS_CHE_Y4 (68.74N) on
#: 2014-11-28 the tree holds sw_in 0.8632 W m-2 over ext_rad 0.0009 W m-2, a kt of 923.
#: :data:`KT_RESIDUAL_SITES` was recorded under exactly this mask -- probe_support_rules counted ``served > 5``
#: (probes/radiation/probe_support_rules.py:238) -- so a gate without it does not measure its own baseline. On
#: the 2026-09-17 tree that cost 20 phantom site-days, all of them polar twilight at the corpus's two
#: highest-latitude sites, while no other site moved by a single day (Palma 46781424).
KT_MIN_EXT_RAD = 5.0

#: Violations quoted in an error message; the count is always complete.
MAX_EXAMPLES = 5


class SupportInvariantError(RuntimeError):
    """A daily env value was about to be written without the support its rule requires."""


def check_support(daily: pd.DataFrame) -> None:
    """Raise :class:`SupportInvariantError` if any written value lacks the support its rule requires."""
    expected = _expected_hours(daily)
    required = (expected * MIN_ENV_HOURS + HOURS_PER_DAY - 1) // HOURS_PER_DAY
    violations: List[str] = []
    for col in ENV_CONTINUOUS_COLS:
        violations += _hour_violations(daily, col, [col, f"{col}_min", f"{col}_max", f"{col}_sum"], required)
    for col in ENV_RADIATION_COLS:
        violations += _coverage_violations(daily, col, [col, f"{col}_sum"])
        violations += _hour_violations(daily, col, [f"{col}_min", f"{col}_max"], required)
    if violations:
        shown = "\n  ".join(violations[:MAX_EXAMPLES])
        raise SupportInvariantError(
            f"{len(violations)} daily env value(s) written without the support the rule requires:\n  {shown}"
        )


def kt_exceedance_days(frame: pd.DataFrame, day_col: str = "TIMESTAMP") -> int:
    """UTC days on which daily ``sw_in`` exceeds a daily ``ext_rad`` worth dividing by; a day counts once.

    Days whose daily ``ext_rad`` is at or below :data:`KT_MIN_EXT_RAD` are not scored, which is the mask
    :data:`KT_RESIDUAL_SITES` was recorded under. A plant-level day still counts once.
    """
    if "sw_in" not in frame.columns or "ext_rad" not in frame.columns:
        return 0
    sw_in = pd.to_numeric(frame["sw_in"], errors="coerce")
    ext_rad = pd.to_numeric(frame["ext_rad"], errors="coerce")
    over = ((ext_rad > KT_MIN_EXT_RAD) & (sw_in > ext_rad)).to_numpy()
    if not over.any():
        return 0
    stamps = frame.loc[over, day_col] if day_col in frame.columns else frame.index[over]
    return int(pd.DatetimeIndex(pd.to_datetime(stamps, utc=True)).floor("D").nunique())


def radiation_gate(
    counts: Mapping[str, int],
    baseline_total: int = KT_BASELINE_TOTAL,
    residual_sites: Optional[Mapping[str, int]] = None,
) -> List[str]:
    """Why a tree's per-site kt > 1 counts fail the gate; empty when they pass."""
    residual = KT_RESIDUAL_SITES if residual_sites is None else residual_sites
    errors: List[str] = []
    total = sum(int(n) for n in counts.values())
    if total > baseline_total:
        errors.append(f"{total} site-days with daily sw_in above ext_rad, above the baseline of {baseline_total}")
    for site in sorted(counts):
        days = int(counts[site])
        if days <= 0:
            continue
        if site not in residual:
            errors.append(f"{site}: {days} site-days with daily sw_in above ext_rad at a new site")
        elif days > int(residual[site]):
            # The total alone would let one site pay for another: a documented residual site can climb
            # while a second one falls, and the corpus looks unchanged. Each answers for its own count.
            errors.append(
                f"{site}: {days} site-days with daily sw_in above ext_rad, above its own "
                f"baseline of {int(residual[site])}"
            )
    return errors


def env_daily_rule() -> Dict[str, object]:
    """The rule a tree was built under, for its provenance block."""
    return {
        "version": RULE_VERSION,
        "min_hours": MIN_ENV_HOURS,
        "min_toa_coverage": MIN_TOA_COVERAGE,
        "ext_rad": "geometry",
        "continuous_columns": list(ENV_CONTINUOUS_COLS),
        "radiation_columns": list(ENV_RADIATION_COLS),
        "dark_window_minutes": [DARK_SAMPLE_MINUTES[0], DARK_SAMPLE_MINUTES[-1]],
        "toa_sample_minutes": list(BODY_SAMPLE_MINUTES),
        "not_covered": ["netrad"],
    }


def _expected_hours(daily: pd.DataFrame) -> pd.Series:
    if EXPECTED_HOURS_COL in daily.columns:
        return pd.to_numeric(daily[EXPECTED_HOURS_COL], errors="coerce").fillna(0).astype(int)
    return pd.Series(HOURS_PER_DAY, index=daily.index)


def _written(daily: pd.DataFrame, stats: List[str]) -> pd.Series:
    present = [col for col in stats if col in daily.columns]
    if not present:
        return pd.Series(False, index=daily.index)
    return daily[present].notna().any(axis=1)


def _hour_violations(daily: pd.DataFrame, col: str, stats: List[str], required: pd.Series) -> List[str]:
    written = _written(daily, stats)
    if not written.any():
        return []
    hours_col = f"{col}_hours"
    if hours_col not in daily.columns:
        return [f"{col}: values written with no {hours_col} column to show their support"]
    # Positional throughout: the merge's plant-level frame repeats every day, once per tree.
    hours = pd.to_numeric(daily[hours_col], errors="coerce").to_numpy(dtype=float)
    needed = required.to_numpy()
    bad = written.to_numpy() & ~(hours >= needed)
    labels = _row_labels(daily)
    return [
        f"{col} on {_day(labels[i])}: {hours[i]:g} hours, the rule needs {needed[i]}" for i in np.flatnonzero(bad)
    ]


def _coverage_violations(daily: pd.DataFrame, col: str, stats: List[str]) -> List[str]:
    written = _written(daily, stats)
    if not written.any():
        return []
    coverage_col = f"{col}_toa_coverage"
    if coverage_col not in daily.columns:
        return [f"{col}: values written with no {coverage_col} column to show their support"]
    coverage = pd.to_numeric(daily[coverage_col], errors="coerce").to_numpy(dtype=float)
    bad = written.to_numpy() & ~(coverage >= MIN_TOA_COVERAGE)
    labels = _row_labels(daily)
    return [
        f"{col} on {_day(labels[i])}: TOA coverage {coverage[i]:.3f}, the rule needs {MIN_TOA_COVERAGE}"
        for i in np.flatnonzero(bad)
    ]


def _row_labels(daily: pd.DataFrame) -> pd.Index:
    """What names a row in a message: the merge frame's TIMESTAMP column, else the index."""
    return pd.Index(daily["TIMESTAMP"]) if "TIMESTAMP" in daily.columns else daily.index


def _day(value) -> str:
    return str(value.date()) if isinstance(value, pd.Timestamp) else str(value)
