"""Measure every site's clock in a Stage-3 tree and judge it against the registry.

    python -m src.clock_offsets.gate --env-dir ENV --sap-dir SAP --md-dir MD [--out report.json]

The directories are Stage-3 ``outliers_removed`` output and the corpus metadata. Each site is measured twice
over -- the clear-day correlation and the daylight-edge shift of :mod:`src.clock_offsets.detect` -- and every
estimate is calibrated on the median of the CONTROL sites, those measuring under 0.75 h, so that interval-end
labels and other shared conventions cancel.

ERROR   a site with no ``applied`` env entry whose two estimators agree within 0.5 h and EACH reach 1.0 h:
        an unregistered clock, or one the registry names under a different key.
ERROR   a site whose ``applied`` env entry left it still measuring 0.5 h or more on either of two readings:
        a correction skipped, one applied twice, or one applied with the wrong sign. The message says which
        it looks like, unless the entry has periods, whose corrections are several and sum to nothing real.
WARN    a corrected site only one estimator can read, a single-estimator or disagreeing reading elsewhere, an
        ``acknowledged`` entry, and a sap-versus-own-radiation lag outside the control band.

Two estimators must agree before anything is an ERROR, because one alone is fooled: a dead pyranometer reads
as a clock, a polar day has no edges, and thick afternoon cloud moves an all-day correlation. A site the gate
cannot measure produces no finding at all -- its reason is in the report.

Exit 1 if any ERROR, else 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.clock_offsets.detect import MIN_CLEAR_DAYS, MIN_EDGE_DAYS, Estimate, clear_day_lag, edge_shift, pair_lag
from src.clock_offsets.registry import REGISTRY, registry_sha256

ENV_SUFFIX = "_env_data_outliers_removed.csv"
SAP_SUFFIX = "_sapf_data_outliers_removed.csv"
MD_SUFFIX = "_site_md.csv"
TIME_COLUMNS = ("TIMESTAMP", "solar_TIMESTAMP", "TIMESTAMP_LOCAL")

ERROR = "ERROR"
WARN = "WARN"

#: A site measuring less than this is a control: its conventions are the population's zero point.
CONTROL_BAND_HOURS = 0.75

#: An unregistered site needs both estimators this far out before the gate calls it an error.
UNREGISTERED_ERROR_HOURS = 1.0

#: Two estimators this close are one finding; further apart, neither is trusted alone.
AGREEMENT_HOURS = 0.5

#: A corrected site should measure nothing; this much left is a correction that did not land.
APPLIED_ERROR_HOURS = 0.5

#: How far outside the control band a sap pair lag goes before it is worth saying.
PAIR_TOLERANCE_HOURS = 0.5

#: Fewer control sites than this and the zero points are noise: every calibrated offset in the report moves
#: with them, so the run says so. The whole corpus gives around 106.
MIN_CONTROLS_FOR_TRUST = 20


@dataclass(frozen=True)
class Finding:
    level: str
    site: str
    component: str
    message: str


@dataclass(frozen=True)
class SiteClock:
    """What one site's own data say about its clock, before calibration."""

    site: str
    latitude: Optional[float]
    longitude: Optional[float]
    clear_day: Estimate
    edge: Estimate
    pair: Estimate

    def as_dict(self, zeros: Dict[str, float]) -> dict:
        return {
            "site": self.site,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "clear_day": _estimate_dict(self.clear_day, zeros.get("clear_day", 0.0)),
            "edge": _estimate_dict(self.edge, zeros.get("edge", 0.0)),
            "pair": _estimate_dict(self.pair, zeros.get("pair", 0.0)),
        }


def measure_site(
    site: str,
    env_dir,
    sap_dir,
    md_dir,
    min_clear_days: int = MIN_CLEAR_DAYS,
    min_edge_days: int = MIN_EDGE_DAYS,
) -> SiteClock:
    """Both env estimators and the sap pair lag for one site; every failure comes back as a reason."""
    coordinates = site_coordinates(Path(md_dir) / f"{site}{MD_SUFFIX}")
    if coordinates is None:
        missing = Estimate(reason="no usable si_lat / si_long")
        return SiteClock(site, None, None, missing, missing, missing)
    latitude, longitude = coordinates
    sw_in = read_sw_in(Path(env_dir) / f"{site}{ENV_SUFFIX}")
    if sw_in is None:
        missing = Estimate(reason="no sw_in record")
        return SiteClock(site, latitude, longitude, missing, missing, missing)
    edge = edge_shift(sw_in, latitude, longitude, min_days=min_edge_days)
    # The edge shift needs no day frame, so it is the rough estimate the clear-day frame is centred on.
    clear_day = clear_day_lag(
        sw_in, latitude, longitude, reference_hours=edge.hours or 0.0, min_days=min_clear_days
    )
    sap = read_sap(Path(sap_dir) / f"{site}{SAP_SUFFIX}")
    pair = pair_lag(sap, sw_in) if sap is not None else Estimate(reason="no sap record")
    return SiteClock(site, latitude, longitude, clear_day, edge, pair)


def zero_points(clocks: Sequence[SiteClock]) -> Dict[str, float]:
    """The control population's medians: what a site on time measures, conventions and physiology included."""
    controls = [clock for clock in clocks if clock.clear_day and abs(clock.clear_day.hours) < CONTROL_BAND_HOURS]
    pairs = [clock.pair.hours for clock in controls if clock.pair]
    return {
        "clear_day": _median([clock.clear_day.hours for clock in controls if clock.clear_day]),
        "edge": _median([clock.edge.hours for clock in controls if clock.edge]),
        "pair": _median(pairs),
        "pair_p10": _quantile(pairs, 0.1),
        "pair_p90": _quantile(pairs, 0.9),
        "n_controls": len(controls),
        "n_pair_controls": len(pairs),
    }


def findings(clocks: Sequence[SiteClock], zeros: Dict[str, float], registry=REGISTRY) -> List[Finding]:
    """Everything the tree has to answer for, worst first."""
    found: List[Finding] = []
    for clock in clocks:
        found += _env_findings(clock, zeros, registry)
        found += _sap_findings(clock, zeros, registry)
    return sorted(found, key=lambda finding: (finding.level != ERROR, finding.site, finding.component))


def _env_findings(clock: SiteClock, zeros: Dict[str, float], registry) -> List[Finding]:
    clear = _calibrated(clock.clear_day, zeros.get("clear_day", 0.0))
    edge = _calibrated(clock.edge, zeros.get("edge", 0.0))
    applied = _entries(registry, clock.site, "env", "applied")
    acknowledged = _entries(registry, clock.site, "env", "acknowledged")
    measured = [value for value in (clear, edge) if value is not None]
    if not measured:
        return []
    reading = f"clear-day {_hours(clear)}, edges {_hours(edge)}"
    both = len(measured) == 2
    agree = both and abs(clear - edge) <= AGREEMENT_HOURS
    loudest = max(measured, key=abs)

    if applied:
        # A corrected site should measure nothing. When the two estimators agree, the residual IS their
        # mean, and that is the number judged; when they contradict each other there is no single
        # residual, so the loudest is judged and the message says they do not agree. Judging one number
        # and printing another is not a stricter rule, it is an unreadable one: it failed USA_ORN_ST2_AMB
        # for a loudest of 0.50 h while reporting a mean of 0.34 h, a verdict its own number does not
        # support. A disagreement still fails whichever way the two point, because that is not what a
        # landed correction looks like either, and at seven corrected sites a false alarm costs a look,
        # not a rebuild.
        value = float(np.mean(measured)) if (both and agree) else loudest
        if abs(value) < APPLIED_ERROR_HOURS:
            return []
        if both:
            return [
                Finding(
                    ERROR,
                    clock.site,
                    "env",
                    f"{_correction(applied)}, and it still measures {value:+.2f} h ({reading})"
                    f"{_diagnosis(applied, value)}" + ("" if agree else "; the two do not agree"),
                )
            ]
        # Only one estimator can read it -- the polar, short or cloudy sites. Too little to fail on, far
        # too much to swallow: a correction that never landed would otherwise leave the gate silent.
        return [
            Finding(
                WARN,
                clock.site,
                "env",
                f"{_correction(applied)}, and the one estimator that can read it still measures "
                f"{value:+.2f} h ({reading}); one witness is not two, so this is a warning, not a "
                f"failure: {_why(clock, both)}",
            )
        ]

    # Not corrected. Two witnesses that agree, BOTH of them over the threshold, is the only failure: a
    # pair that only averages its way across the line has one estimator saying the site is nearly fine.
    if agree and min(abs(clear), abs(edge)) >= UNREGISTERED_ERROR_HOURS:
        value = float(np.mean(measured))
        if acknowledged:
            return [Finding(WARN, clock.site, "env", f"measures {value:+.2f} h ({reading}); {_ack(acknowledged)}")]
        return [Finding(ERROR, clock.site, "env", f"measures {value:+.2f} h ({reading}) and is not in the registry")]
    if abs(loudest) < UNREGISTERED_ERROR_HOURS:
        return []
    lead = f"measures {loudest:+.2f} h" if both else f"one estimator measures {loudest:+.2f} h"
    if acknowledged:
        return [Finding(WARN, clock.site, "env", f"{lead} ({reading}); {_ack(acknowledged)}: {_why(clock, both)}")]
    return [
        Finding(
            WARN,
            clock.site,
            "env",
            f"{lead} ({reading}), which is not enough to fail on: {_why(clock, both)}",
        )
    ]


def _why(clock: SiteClock, both: bool) -> str:
    """Why this site got one reading instead of two, in the words the estimator used."""
    if both:
        return "the two estimators do not agree"
    return clock.clear_day.reason or clock.edge.reason or "only one estimator could measure"


def _ack(acknowledged) -> str:
    offset = float(acknowledged[0].offset_hours)
    return f"acknowledged in the registry at {offset:+.2f} h and deliberately not corrected"


def _correction(applied) -> str:
    """What the registry says was done to this component -- one offset, or a period entry's several."""
    if len(applied) == 1:
        return f"the registry applied {float(applied[0].offset_hours):+.2f} h"
    offsets = ", ".join(f"{float(entry.offset_hours):+.2f} h" for entry in applied)
    return f"the registry applied {len(applied)} period corrections ({offsets})"


def _diagnosis(applied, value: float) -> str:
    """Never applied, or applied twice -- a reading only a single-entry site can support.

    Over periods there is no single correction to compare against: their sum is a number that was applied to
    nothing, and printing it teaches the reader to distrust the whole line.
    """
    if len(applied) != 1:
        return ""
    correction = float(applied[0].offset_hours)
    if abs(value - correction) <= AGREEMENT_HOURS:
        return ", which reads as if it was never applied"
    if abs(value + correction) <= AGREEMENT_HOURS:
        return ", which reads as if it was applied twice, or with the wrong sign"
    return ""


def _sap_findings(clock: SiteClock, zeros: Dict[str, float], registry) -> List[Finding]:
    pair = _calibrated(clock.pair, zeros.get("pair", 0.0))
    if pair is None or not zeros.get("n_pair_controls"):
        return []
    low = zeros["pair_p10"] - zeros["pair"] - PAIR_TOLERANCE_HOURS
    high = zeros["pair_p90"] - zeros["pair"] + PAIR_TOLERANCE_HOURS
    if low <= pair <= high:
        return []
    applied = _entries(registry, clock.site, "sap", "applied")
    state = "after its correction" if applied else "and has no applied entry"
    return [
        Finding(
            WARN,
            clock.site,
            "sap",
            f"sap runs {pair:+.2f} h against its own radiation {state}; the control band is "
            f"{low:+.2f}..{high:+.2f} h",
        )
    ]


def read_sw_in(path) -> Optional[pd.Series]:
    path = Path(path)
    if not path.exists():
        return None
    header = pd.read_csv(path, nrows=0).columns
    if "sw_in" not in header or "TIMESTAMP" not in header:
        return None
    frame = pd.read_csv(path, usecols=["TIMESTAMP", "sw_in"])
    series = _stamped(pd.to_numeric(frame["sw_in"], errors="coerce").to_numpy(), frame["TIMESTAMP"])
    return series if series is not None and series.notna().any() else None


def read_sap(path) -> Optional[pd.Series]:
    """The site's mean sap, every tree scaled by its own p95 so one big tree cannot carry the phase."""
    path = Path(path)
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    trees = [col for col in frame.columns if col not in TIME_COLUMNS and not str(col).startswith("Unnamed")]
    if not trees or "TIMESTAMP" not in frame.columns:
        return None
    values = frame[trees].apply(pd.to_numeric, errors="coerce")
    scale = values.quantile(0.95)
    usable = [col for col in trees if np.isfinite(scale[col]) and scale[col] > 0]
    if not usable:
        return None
    mean = (values[usable] / scale[usable]).mean(axis=1)
    series = _stamped(mean.to_numpy(), frame["TIMESTAMP"])
    return series if series is not None and series.notna().any() else None


def site_coordinates(path) -> Optional[Tuple[float, float]]:
    path = Path(path)
    if not path.exists():
        return None
    meta = pd.read_csv(path)
    if "si_lat" not in meta.columns or "si_long" not in meta.columns or meta.empty:
        return None
    latitude = pd.to_numeric(meta["si_lat"], errors="coerce").iloc[0]
    longitude = pd.to_numeric(meta["si_long"], errors="coerce").iloc[0]
    if not (np.isfinite(latitude) and np.isfinite(longitude)):
        return None
    return float(latitude), float(longitude)


def _stamped(values, stamps) -> Optional[pd.Series]:
    index = pd.DatetimeIndex(pd.to_datetime(stamps, utc=True, format="mixed", errors="coerce"))
    series = pd.Series(values, index=index)
    series = series[series.index.notna()]
    return series if len(series) else None


def _entries(registry, site: str, component: str, status: str):
    return [
        entry
        for entry in registry
        if entry.site == site and entry.component == component and entry.status == status
    ]


def _calibrated(estimate: Estimate, zero: float) -> Optional[float]:
    return None if not estimate else round(float(estimate.hours) - float(zero), 3)


def _estimate_dict(estimate: Estimate, zero: float) -> dict:
    return {
        "hours": estimate.hours,
        "calibrated_hours": _calibrated(estimate, zero),
        "sample": estimate.sample,
        "quality": estimate.quality,
        "reason": estimate.reason,
    }


def _hours(value: Optional[float]) -> str:
    return "none" if value is None else f"{value:+.2f} h"


def _median(values) -> float:
    return float(np.median(values)) if len(values) else 0.0


def _quantile(values, fraction: float) -> float:
    return float(np.quantile(values, fraction)) if len(values) else 0.0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.clock_offsets.gate", description=__doc__.splitlines()[0])
    parser.add_argument("--env-dir", required=True, help="Stage-3 env outliers_removed directory")
    parser.add_argument("--sap-dir", required=True, help="Stage-3 sap outliers_removed directory")
    parser.add_argument("--md-dir", required=True, help="corpus directory holding {site}_site_md.csv")
    parser.add_argument("--out", help="write the JSON report here")
    parser.add_argument("--sites", nargs="*", help="measure only these sites (default: every env file)")
    parser.add_argument("--min-clear-days", type=int, default=MIN_CLEAR_DAYS)
    parser.add_argument("--min-edge-days", type=int, default=MIN_EDGE_DAYS)
    args = parser.parse_args(argv)

    env_dir = Path(args.env_dir)
    sites = args.sites or sorted(path.name[: -len(ENV_SUFFIX)] for path in env_dir.glob("*" + ENV_SUFFIX))
    if not sites:
        # The most dangerous thing a gate can do is agree with an empty directory: a mistyped --env-dir
        # and a tree that never finished both arrive here as "nothing to report".
        print(f"  CLOCK GATE FAIL: no site files matching *{ENV_SUFFIX} under {env_dir}")
        return 1
    clocks = [
        measure_site(site, env_dir, args.sap_dir, args.md_dir, args.min_clear_days, args.min_edge_days)
        for site in sites
    ]
    zeros = zero_points(clocks)
    if zeros["n_controls"] < MIN_CONTROLS_FOR_TRUST:
        print(
            f"  CLOCK GATE CAUTION: {zeros['n_controls']} control sites. Every offset below is calibrated "
            f"on their median, so on this few it is noise. Measure the whole tree, not a subset of it."
        )
    found = findings(clocks, zeros)
    for finding in found:
        print(f"  CLOCK GATE {finding.level}: {finding.site} {finding.component}: {finding.message}")
    summary = {
        "sites": len(clocks),
        "measured": sum(1 for clock in clocks if clock.clear_day or clock.edge),
        "errors": sum(1 for finding in found if finding.level == ERROR),
        "warnings": sum(1 for finding in found if finding.level == WARN),
    }
    if args.out:
        report = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "registry_sha256": registry_sha256(),
            "zero_points": zeros,
            "summary": summary,
            "findings": [asdict(finding) for finding in found],
            "sites": [clock.as_dict(zeros) for clock in clocks],
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
    print(
        f"  CLOCK GATE: {summary['measured']}/{summary['sites']} sites measured, {summary['errors']} error(s), "
        f"{summary['warnings']} warning(s); zero points clear-day {zeros['clear_day']:+.2f} h, edges "
        f"{zeros['edge']:+.2f} h over {zeros['n_controls']} control sites"
    )
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
