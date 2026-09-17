"""Reviewed clock offsets of the corpus: which site, which component, how far, and since when.

A handful of SAPFLUXNET sites ship a ``TIMESTAMP`` that claims to be UTC and is not. The offsets are in the
archive itself -- the raw corpus file and the Stage-3 output carry the same stamps -- and everything
downstream reads the label as UTC: the Stage-3 day/night QC strata, sap incomplete-day removal, the merge's
hourly join, and every daily statistic. Measured 2026-09-14/16 against solar geometry (clear-day
correlation and daylight edges, which cloud cannot fake together) and ERA5-Land.

**Sign convention: a positive offset means the label is LATE** -- the reading was taken before its stamp --
so the true time is ``label - offset``. The 2026-09-14 probes (``probe_sap_clock*.py``) printed the
opposite sign; negate their numbers before comparing them with this table.

**A reviewed table, not a run-time correction.** :mod:`src.clock_offsets.detect` measures clocks and the
gate fails on what it finds, but only this table moves a timestamp. An estimator fed by a dead pyranometer
(USA_PAR_FER) or by night readings (FRA_PUE) reports a clock error that is not there, and an automatic
correction would silently rotate a healthy site's diurnal cycle.

**Python, not CSV.** The repository ignores ``*.csv``: ``removal_log.csv`` sat untracked, and absent on
Palma, for months because of it. A module is tracked, reviewed in diffs, and validated by the loaders
before they apply a single entry.

``status``
    ``applied``       the Stage-3 loaders move this site's component by ``offset_hours`` inside the period;
    ``acknowledged``  suspected or measured and deliberately NOT applied, recorded so the gate reports it as
                      a known finding rather than a new one. ``offset_hours`` may be 0.0 while unresolved.

``start`` / ``end`` are UTC LABEL times -- the uncorrected stamps -- half-open ``[start, end)``; ``None`` is
open-ended.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

COMPONENTS = ("env", "sap")
STATUSES = ("applied", "acknowledged")

#: No civil UTC offset exceeds 14 h; an entry beyond that is a typo or a date error, not a clock.
MAX_OFFSET_HOURS = 14.0

#: Clocks are set in whole or quarter hours (UTC+5:45 is the finest real zone), so a finer offset is a
#: measurement pasted in unrounded.
OFFSET_QUANTUM_HOURS = 0.25

#: The corpus file each component is read from -- an entry must name a site that has one, by exact name.
CORPUS_SUFFIXES = {"env": "_env_data.csv", "sap": "_sapf_data.csv"}

_SITE_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class ClockRegistryError(ValueError):
    """The registry cannot be applied as written; the message lists every problem found."""


@dataclass(frozen=True)
class ClockOffset:
    """One reviewed clock offset; see the module docstring for the sign convention and ``status``."""

    site: str
    component: str
    offset_hours: float
    status: str
    start: Optional[str] = None
    end: Optional[str] = None
    evidence: str = ""


#: The reviewed offsets. Measured by the T5 probes (Palma 46760679, 46760722, 46760762, 46760767) over the
#: whole corpus, each estimator calibrated on the median of the control sites -- the sites whose own radiation
#: lag is under 0.75 h -- so that ERA5's accumulation stamp, interval-end labels and the physiological delay of
#: sap behind radiation cancel. An empty registry makes both loaders a no-op.
#:
#: `applied` values are whole hours (ISR's two periods, +5 at USA_HUY_LIN_NON and USA_ORN_ST2_AMB,
#: -6 at SWE_SVA_MIX_NON) except FIN_PET's env, which three independent estimators put at 3.5 h.
#: `acknowledged` entries are measured and NOT corrected: the 1-1.5 h group, whose component or period the
#: evidence does not settle, and FIN_PET's sap. They exist so a later pass starts from the measurement.
REGISTRY: Tuple[ClockOffset, ...] = (
    # ISR_YAT_YAT: env only -- every sap tree is on true time (sap vs TOA +0.08 h, vs ERA5-Land +0.03 h).
    # Three clocks in one record: right until the logger stopped on 2009-09-18, -5 h when it came back, and
    # -6 h after a reset during the night of 2009-11-23.
    ClockOffset(
        site="ISR_YAT_YAT",
        component="env",
        offset_hours=-5.0,
        status="applied",
        start="2009-09-30T00:00:00Z",
        end="2009-11-23T12:00:00Z",
        evidence=(
            "per day: TOA correlation -5.0..-5.2, ERA5-Land ssrd -5.0..-5.2, daylight edges -5.0..-5.25. "
            "No env reading 2009-09-19..09-29; before that gap the clock is right (daily ta maximum at label "
            "11.0-12.5 h, against 6.0-7.5 h from 2009-10-01), so the period starts inside the gap."
        ),
    ),
    ClockOffset(
        site="ISR_YAT_YAT",
        component="env",
        offset_hours=-6.0,
        status="applied",
        start="2009-11-23T12:00:00Z",
        evidence=(
            "2010-2015 whole years: TOA correlation -5.91..-6.03, edges -6.0, ERA5-Land -5.9..-6.2. The step "
            "is in the night of 2009-11-23: that day's daylight still fits the -5 h clock (sw_in / TOA(label+5h) "
            "flat at 0.84-0.86 all afternoon, while label+6h gives an impossible 1.0-1.5), and the next sunrise "
            "already appears at label 21:30. The label sequence itself is regular throughout."
        ),
    ),
    # USA_HUY_LIN_NON: env only; the site ships no sap file.
    ClockOffset(
        site="USA_HUY_LIN_NON",
        component="env",
        offset_hours=5.0,
        status="applied",
        evidence=(
            "TOA correlation +4.97, daylight edges +5.00, clear-day correlation +4.96, ERA5-Land +4.79, "
            "morning crossing +4.85 (calibrated); 2017, the only full year, reads the same."
        ),
    ),
    # USA_ORN_ST1_AMB and ST2_AMB stand at one location with one set of sensors, so the twin difference is
    # free of every site convention: it is exactly 5.00 h on all five env estimators. ST1 itself is on time.
    ClockOffset(
        site="USA_ORN_ST2_AMB",
        component="env",
        offset_hours=5.0,
        status="applied",
        evidence=(
            "ST2 minus its twin ST1: +5.00 h on TOA correlation, clear-day correlation, both daylight-edge "
            "estimators and ERA5-Land, and +5.00 on the morning crossing. 2007 and 2008 agree (+4.93, +4.96)."
        ),
    ),
    ClockOffset(
        site="USA_ORN_ST2_AMB",
        component="sap",
        offset_hours=5.0,
        status="applied",
        evidence=(
            "ST2 minus ST1: sap vs TOA +4.54, vs ERA5-Land +4.66, morning crossing +5.12. Sap against the "
            "site's OWN sw_in is -0.22..-0.30 h at both twins, so sap and env share the one shifted clock."
        ),
    ),
    # FIN_PET: 2008-06-18..09-11 at 69.5 N. Daylight edges are unreliable there (polar day until late July,
    # shallow sunrises after it), and the two estimators that do not need an edge agree on 3.5 h.
    ClockOffset(
        site="FIN_PET",
        component="env",
        offset_hours=3.5,
        status="applied",
        evidence=(
            "TOA correlation +3.44, ERA5-Land ssrd +3.46, morning crossing +3.44 (global zero) / +3.47 "
            "(boreal); daylight edges read +2.75 and are not trusted at 69.5 N. CAVEAT: this site's vpd peaks "
            "0.2 h after its sw_in, against 1.9-2.3 h at every control site, so its thermal columns are "
            "probably on a clock about 2 h earlier than its radiation columns. The registry corrects a "
            "component, not a column: ta/rh/vpd are moved by the radiation offset with the rest."
        ),
    ),
    # SWE_SVA_MIX_NON: sap only -- its own sw_in is on time (every env estimator within 0.11 h).
    ClockOffset(
        site="SWE_SVA_MIX_NON",
        component="sap",
        offset_hours=-6.0,
        status="applied",
        evidence=(
            "Pair lags against the site's own env, calibrated on the 16 boreal controls: sw_in -5.93, "
            "vpd -5.95, ta -6.10, TOA -5.80, morning crossing -6.38. The boreal zero point is the like-for-like "
            "one for a 64 N site and is not one network's convention (Finnish, Russian and Swedish controls "
            "all lie at 1.6-3.1 h); under it the radiation and thermal references agree to 0.02 h, where the "
            "global zero leaves them 0.46 h apart and gives -4.8."
        ),
    ),
    # Measured, NOT corrected (user decision 2026-09-16: big offsets first).
    ClockOffset(
        site="FIN_PET",
        component="sap",
        offset_hours=1.5,
        status="acknowledged",
        evidence=(
            "Radiation references put it at +1.4..+2.5 (pair against sw_in, boreal to global zero), thermal "
            "references at +3.4..+4.2 (vpd, ta). The two cannot both be right: this site's own vpd-to-sw_in "
            "phase is 0.2 h against 1.9-2.3 h everywhere else, so one of its loggers is on a different clock. "
            "Unresolved, so not corrected."
        ),
    ),
    ClockOffset(
        site="RUS_CHE_Y4",
        component="env",
        offset_hours=1.0,
        status="acknowledged",
        evidence=(
            "TOA correlation +1.18, edges +1.00, ERA5-Land +1.25, morning crossing +1.07; sap ambiguous (+0.1 boreal, "
            "+1.3 global)."
        ),
    ),
    ClockOffset(
        site="USA_SYL_HL2",
        component="env",
        offset_hours=-1.5,
        status="acknowledged",
        evidence=(
            "TOA correlation -1.41, edges -1.50, clear-day -1.43, ERA5-Land -1.49, morning crossing -1.47; sap reads "
            "about +0.35."
        ),
    ),
    ClockOffset(
        site="ESP_LAS",
        component="env",
        offset_hours=-1.0,
        status="acknowledged",
        evidence=(
            "TOA correlation -1.07, edges -1.25, ERA5-Land -1.13, morning crossing -1.28; sap -0.77 (global zero)."
        ),
    ),
    ClockOffset(
        site="ITA_TOR",
        component="env",
        offset_hours=-1.0,
        status="acknowledged",
        evidence="TOA correlation -1.10, edges -1.00, ERA5-Land -1.05, morning crossing -1.07; 2015 and 2016 agree.",
    ),
    ClockOffset(
        site="ITA_TOR",
        component="sap",
        offset_hours=-2.0,
        status="acknowledged",
        evidence=(
            "Sap vs TOA -1.96, vs ERA5-Land -1.97, pair against its own sw_in -0.83: one hour earlier than its env."
        ),
    ),
    ClockOffset(
        site="ESP_ALT_ARM",
        component="env",
        offset_hours=1.0,
        status="acknowledged",
        evidence=(
            "TOA correlation +0.97, edges +0.75, ERA5-Land +1.03 for 2009-2013; 2014 reads +0.26, so the "
            "record changes clock part way and needs a period before it can be applied."
        ),
    ),
    ClockOffset(
        site="ESP_ALT_HUE",
        component="env",
        offset_hours=1.25,
        status="acknowledged",
        evidence=(
            "TOA correlation +1.14, edges +0.75, ERA5-Land +1.21, morning crossing +2.17; its sap reads "
            "about 0, unlike its neighbour ESP_ALT_ARM."
        ),
    ),
    ClockOffset(
        site="ESP_LAH_COM",
        component="env",
        offset_hours=-1.0,
        status="acknowledged",
        evidence="TOA correlation -1.11, edges -1.25, ERA5-Land -1.11, morning crossing -1.29; 2012 and 2013 agree.",
    ),
    ClockOffset(
        site="ESP_LAH_COM",
        component="sap",
        offset_hours=-1.25,
        status="acknowledged",
        evidence=(
            "Sap vs TOA -1.23, vs ERA5-Land -1.16; the pair against its own sw_in is -0.04, so sap and env share the "
            "offset."
        ),
    ),
    ClockOffset(
        site="GBR_GUI_ST1",
        component="env",
        offset_hours=0.75,
        status="acknowledged",
        evidence=(
            "TOA correlation +0.65, edges +0.75, ERA5-Land +0.62, morning crossing +1.10. ST1, ST2 and ST3 "
            "share one env file."
        ),
    ),
    ClockOffset(
        site="GBR_GUI_ST2",
        component="env",
        offset_hours=0.75,
        status="acknowledged",
        evidence="The same env file as GBR_GUI_ST1.",
    ),
    ClockOffset(
        site="GBR_GUI_ST3",
        component="env",
        offset_hours=0.75,
        status="acknowledged",
        evidence="The same env file as GBR_GUI_ST1.",
    ),
    ClockOffset(
        site="GBR_DEV_CON",
        component="env",
        offset_hours=1.5,
        status="acknowledged",
        evidence=(
            "TOA correlation +1.46, edges +1.50, ERA5-Land +1.52, morning crossing +1.56; a summer-only record, and "
            "GBR_DEV_DRO shares the file."
        ),
    ),
    ClockOffset(
        site="GBR_DEV_DRO",
        component="env",
        offset_hours=1.5,
        status="acknowledged",
        evidence="The same env file as GBR_DEV_CON.",
    ),
    ClockOffset(
        site="USA_HIL_HF1_POS",
        component="env",
        offset_hours=-1.5,
        status="acknowledged",
        evidence=(
            "TOA correlation -1.49, ERA5-Land -1.41, morning crossing -1.12; no day passes the edge estimator's gates. "
            "USA_HIL_HF2 shares the file."
        ),
    ),
    ClockOffset(
        site="USA_HIL_HF2",
        component="env",
        offset_hours=-1.5,
        status="acknowledged",
        evidence="The same env file as USA_HIL_HF1_POS.",
    ),
    ClockOffset(
        site="CHN_YUN_YUN",
        component="env",
        offset_hours=1.25,
        status="acknowledged",
        evidence=(
            "TOA correlation +1.33, clear-day +1.36, ERA5-Land +1.22, morning crossing +0.77; sw_in units are also in "
            "question at this site."
        ),
    ),
)


def offsets_for(site: str, component: str, registry: Iterable[ClockOffset] = REGISTRY) -> Tuple[ClockOffset, ...]:
    """The ``applied`` entries for exactly ``site`` and ``component``, in registry order.

    ``site`` is the FULL site code (:func:`src.clock_offsets.apply.site_code`). The Stage-3 loaders' own
    ``location`` key is the first two tokens of the file name -- ``USA_ORN`` for both ORN twins -- and must
    never be passed here.
    """
    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS}, got {component!r}")
    return tuple(
        entry for entry in registry if entry.site == site and entry.component == component and entry.status == "applied"
    )


def validate_registry(registry: Iterable[ClockOffset] = REGISTRY, corpus_dir=None) -> List[str]:
    """Every reason the registry cannot be applied; empty when it can.

    ``corpus_dir`` enables the check that each entry's site has its corpus file. ``None`` skips only that
    check, for callers with no corpus on disk.
    """
    errors: List[str] = []
    bounded = []
    for entry in registry:
        if not isinstance(entry, ClockOffset):
            errors.append(f"registry entries must be ClockOffset, got {type(entry).__name__}")
            continue
        entry_errors, bounds = _entry_errors(entry, corpus_dir)
        errors.extend(entry_errors)
        bounded.append((entry, bounds))
    errors.extend(_overlap_errors(bounded))
    return errors


def require_valid_registry(registry: Iterable[ClockOffset] = REGISTRY, corpus_dir=None) -> None:
    """Raise :class:`ClockRegistryError` naming every problem, or return silently."""
    errors = validate_registry(registry, corpus_dir)
    if errors:
        raise ClockRegistryError("the clock-offset registry cannot be applied:\n  " + "\n  ".join(errors))


def registry_sha256(registry: Iterable[ClockOffset] = REGISTRY) -> str:
    """A content hash of the registry for provenance; independent of entry order."""
    rows = sorted(json.dumps(asdict(entry), sort_keys=True) for entry in registry)
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _entry_errors(entry: ClockOffset, corpus_dir):
    where = f"{entry.site!r}/{entry.component!r}"
    errors: List[str] = []
    site_ok = isinstance(entry.site, str) and _SITE_CODE.match(entry.site) is not None
    if not site_ok:
        errors.append(f"{where}: site must be a corpus site code, got {entry.site!r}")
    component_ok = entry.component in COMPONENTS
    if not component_ok:
        errors.append(f"{where}: component must be one of {COMPONENTS}")
    if entry.status not in STATUSES:
        errors.append(f"{where}: status must be one of {STATUSES}, got {entry.status!r}")
    errors.extend(_offset_errors(entry, where))
    if entry.status == "applied" and not str(entry.evidence).strip():
        errors.append(f"{where}: an applied offset must name its evidence")
    start = _bound(entry.start, "start", where, errors)
    end = _bound(entry.end, "end", where, errors)
    if start is not None and end is not None and not start < end:
        errors.append(f"{where}: start {entry.start} must be before end {entry.end}")
    if corpus_dir is not None and site_ok and component_ok:
        name = entry.site + CORPUS_SUFFIXES[entry.component]
        if not (Path(corpus_dir) / name).is_file():
            errors.append(f"{where}: no corpus file {name} in {corpus_dir}")
    return errors, (start, end)


def _offset_errors(entry: ClockOffset, where: str) -> List[str]:
    try:
        hours = float(entry.offset_hours)
    except (TypeError, ValueError):
        return [f"{where}: offset_hours must be a number, got {entry.offset_hours!r}"]
    if not math.isfinite(hours):
        return [f"{where}: offset_hours must be finite, got {hours}"]
    if abs(hours) > MAX_OFFSET_HOURS:
        return [f"{where}: offset {hours} h exceeds the {MAX_OFFSET_HOURS} h no UTC offset goes beyond"]
    steps = hours / OFFSET_QUANTUM_HOURS
    if abs(steps - round(steps)) > 1e-9:
        return [f"{where}: offset {hours} h is not a multiple of {OFFSET_QUANTUM_HOURS} h"]
    if hours == 0.0 and entry.status == "applied":
        return [f"{where}: an applied offset of zero moves nothing; mark the entry acknowledged instead"]
    return []


def _bound(value, label: str, where: str, errors: List[str]):
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        stamp = pd.NaT
    if pd.isna(stamp):
        errors.append(f"{where}: cannot parse {label} {value!r} as a UTC time")
        return None
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _overlap_errors(bounded) -> List[str]:
    groups = {}
    for entry, (start, end) in bounded:
        groups.setdefault((entry.site, entry.component), []).append((entry, start, end))
    errors: List[str] = []
    for (site, component), members in groups.items():
        for i, (first, first_start, first_end) in enumerate(members):
            for second, second_start, second_end in members[i + 1 :]:
                if _overlaps(first_start, first_end, second_start, second_end):
                    errors.append(
                        f"{site!r}/{component!r}: periods overlap -- "
                        f"[{first.start}, {first.end}) and [{second.start}, {second.end})"
                    )
    return errors


def _overlaps(start_a, end_a, start_b, end_b) -> bool:
    """Whether two half-open periods share an instant; ``None`` is unbounded."""
    a_ends_first = end_a is not None and start_b is not None and end_a <= start_b
    b_ends_first = end_b is not None and start_a is not None and end_b <= start_a
    return not (a_ends_first or b_ends_first)
