"""The combined corpus's account of itself.

Written **last**, on purpose: a directory without a manifest is an interrupted
run, and a consumer can tell the two apart without trusting a file count.

The one field worth explaining is ``asw_model``. Every converted value here is a
quotient whose denominator came from the allometric imputer, and the division
discards the denominator -- a converted ``sapf_data`` holds ``1.83`` with no way
to recover whether that was ``flow / 411.9`` or ``flow / 714.9``. Recording which
model produced the areas is the only chance to preserve that, and it has to
happen here because nothing downstream can reconstruct it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path

from src.sapwood_combined.assembly import SiteOutcome
from src.sapwood_combined.conversion import SAPWOOD_UNITS
from src.sapwood_combined.membership import INTERNAL, OVERLAP_NOTES, ROUTE_CONVERTED, ROUTE_NATIVE

MANIFEST_NAME = "MANIFEST.json"


#: Kinds that only one of the two corpora ships.
_FLAG_KINDS = ("sapf_flags", "env_flags")


def _flags_available(written: Iterable[SiteOutcome]) -> dict[str, object]:
    """Which sites carry QC flag files, and which legitimately do not.

    Stated explicitly because the absence is otherwise indistinguishable from
    assembly loss. The internal corpus never generated flags; QC degrades
    gracefully without them (``flags = None``), which is exactly what makes the
    gap invisible unless the product names it.
    """
    with_flags, without = [], []
    for o in written:
        (with_flags if all(k in o.kinds for k in _FLAG_KINDS) else without).append(o.site)
    return {
        "kinds": list(_FLAG_KINDS),
        "sites_with_flags": len(with_flags),
        "sites_without_flags": len(without),
        "without_flags": sorted(without),
        "note": (
            "Sites listed here ship no sapf_flags/env_flags because their source corpus "
            "never generated them, not because assembly dropped them. QC treats a missing "
            "flags file as 'no flags' rather than an error, so flag-based filtering simply "
            "does not run for these sites."
        ),
    }


def _corpora_block(included: Iterable[str] | None, written: Iterable[SiteOutcome]) -> dict[str, object]:
    """Name the corpora this run drew on, and how many sites each contributed.

    ``internal_included`` is stated explicitly rather than inferred from the
    site count: a build run with ``--skip-internal`` is a legitimate product,
    but it must not be mistaken later for a full one that lost 24 sites.
    """
    per_origin: dict[str, int] = {}
    for o in written:
        per_origin[o.origin] = per_origin.get(o.origin, 0) + 1
    names = sorted(included) if included is not None else sorted(per_origin)
    return {
        "included": names,
        "internal_included": INTERNAL in names,
        "sites_per_origin": per_origin,
    }


def build_manifest(
    outcomes: Iterable[SiteOutcome],
    asw_model: Mapping[str, object],
    duplicate_sites: Mapping[str, str] | None = None,
    stem_overlaps: Mapping[str, object] | None = None,
    corpora_included: Iterable[str] | None = None,
) -> dict[str, object]:
    """Assemble the manifest document from what the run actually did."""
    outcomes = list(outcomes)
    written = [o for o in outcomes if o.was_written]
    skipped = [o for o in outcomes if not o.was_written]

    sites: dict[str, object] = {}
    for o in written:
        sites[o.site] = {
            "route": o.route,
            "origin": o.origin,
            "n_trees_in": o.n_trees_in,
            "n_trees_out": o.n_trees_out,
            "dropped": [[code, reason] for code, reason in o.dropped],
            "asw_source_flags": dict(o.source_flags),
            "kinds": list(o.kinds),
            "files": o.files_written,
        }

    return {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "units": SAPWOOD_UNITS,
        "summary": {
            "sites_written": len(written),
            "sites_skipped": len(skipped),
            "native": sum(1 for o in written if o.route == ROUTE_NATIVE),
            "converted": sum(1 for o in written if o.route == ROUTE_CONVERTED),
            "trees_written": sum(o.n_trees_out for o in written),
            "trees_dropped": sum(len(o.dropped) for o in written),
            "files_written": sum(o.files_written for o in written),
        },
        "asw_model": dict(asw_model),
        # Which source corpora this run drew on. A 202-site archive-only build
        # and a 226-site full build differ only in their site list, which is
        # not something a consumer should have to count to discover.
        "corpora": _corpora_block(corpora_included, written),
        "flags_available": _flags_available(written),
        "sites": sites,
        "skipped": {o.site: o.skipped_reason for o in skipped},
        # Advisory, not applied. Every site the corpora hold is written; these
        # are the pairs that describe shared stems, recorded so a consumer can
        # filter and so the evidence behind the finding is not lost.
        "stem_overlaps": dict(stem_overlaps) if stem_overlaps is not None else dict(OVERLAP_NOTES),
        "duplicate_stems": dict(duplicate_sites) if duplicate_sites else {},
    }


def write_manifest(out_dir: Path, manifest: Mapping[str, object]) -> Path:
    """Serialise the manifest into ``out_dir``. Returns the path written."""
    path = Path(out_dir) / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
    return path
