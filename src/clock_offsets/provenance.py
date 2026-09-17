"""What clock correction a Stage-3 output actually received, recorded beside it and checked by the merge.

The registry alone cannot answer that. Stage 3 applies the offsets and is almost never re-run -- the
production tree's 226 sites were written on 2026-09-14 and the clock work re-ran ten of them -- while the
Stage-4 merge imports the registry afresh on every sweep and writes its hash into the tree's contract
block. Edit the registry and re-run only the merge, which is the ordinary workflow and an hour against
Stage 3's morning, and the block asserts corrections the inputs never received, with nothing anywhere to
contradict it. A hash of the registry records what the MERGE read; it says nothing about what Stage 3 did.

So the claim is kept next to the evidence. :func:`write_stamp` drops a small JSON beside each Stage-3
output the registry actually moved, naming the offsets that moved it; :func:`stamp_errors` is the merge's
per-site check that the stamp beside its input says what the registry the merge holds says. It fails both
ways round: a site the registry corrects whose input carries no stamp was never re-run, and a stamped
input the registry no longer names was moved by a rule since withdrawn.

A site the registry names nowhere gets no stamp and needs none. Its Stage-3 output is byte-identical to
what it would be if the registry did not exist -- which is precisely what the negative control measures --
so there is nothing for a stamp to add there and 216 empty files for it to cost.

Per file, never per directory. ``outliers_removed/`` holds sites from more than one Stage-3 vintage, and a
single manifest for the directory can neither describe that nor survive the batched runs that fill it.

The site is read from the file name by :func:`src.clock_offsets.apply.site_code`, the same exact-suffix
parser that picks the entries at the load boundary, so a stamp and the correction it records cannot end up
keyed differently -- not even for the ORN twins, which a prefix match would confuse.

The digest covers what decides where a row lands: site, component, offset, status and period. Not
``evidence``: prose is corrected far more often than a clock is, and a tree should not need rebuilding
because a sentence was rewritten.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from src.clock_offsets.apply import site_code
from src.clock_offsets.registry import COMPONENTS, REGISTRY, offsets_for, registry_sha256

#: The stamp beside ``<name>.csv`` is ``<name>.clock.json``.
STAMP_SUFFIX = ".clock.json"

#: The stamp format. A reader that meets a version it does not know refuses it rather than guessing.
STAMP_VERSION = 1

#: The fields that decide where a row lands. ``evidence`` is deliberately absent; see the module docstring.
ACTING_FIELDS = ("site", "component", "offset_hours", "status", "start", "end")

#: Mismatched sites quoted in a gate message; the count itself is always complete.
MAX_REPORTED_SITES = 5


def stamp_path(data_path) -> Path:
    """The stamp that belongs beside the Stage-3 output ``data_path``."""
    return Path(data_path).with_suffix(STAMP_SUFFIX)


def offsets_digest(entries: Iterable) -> str:
    """A content hash of the acting fields of ``entries``, independent of their order."""
    rows = sorted(json.dumps(_acting(entry), sort_keys=True) for entry in entries)
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def describe(entries: Iterable) -> str:
    """What was applied, in the words the loader prints: every entry listed, never summed."""
    rows = [_acting(entry) for entry in entries]
    if not rows:
        return "no offset"
    return "; ".join("{0:+g} h{1}".format(float(row["offset_hours"]), _period(row)) for row in rows)


def stamp_content(site: str, component: str, entries: Sequence, registry=REGISTRY) -> Dict[str, object]:
    """What :func:`write_stamp` writes: the offsets applied, their digest, and the registry they came from.

    ``registry_sha256`` is for a human tracing the run; it is never compared, because the registry
    legitimately changes for OTHER sites between the Stage-3 run that wrote this file and the merge that
    reads it. Only ``offsets_sha256``, which covers this site and component alone, is judged.
    """
    return {
        "version": STAMP_VERSION,
        "site": site,
        "component": component,
        "offsets": [_acting(entry) for entry in entries],
        "offsets_sha256": offsets_digest(entries),
        "registry_sha256": registry_sha256(registry),
    }


def write_stamp(data_path, component: str, registry=REGISTRY) -> Optional[Path]:
    """Record beside ``data_path`` which offsets moved it; return the stamp, or None when none did.

    The Stage-3 analyzers call this straight after writing the file, with the registry they corrected it
    with. A site with no applied entry gets no stamp, and a stamp left by an earlier registry that no
    longer names the site is removed -- so the pair on disk always describes the run that wrote it, and a
    rollback cannot leave a claim standing over data that no longer supports it.
    """
    site = site_code(data_path)
    entries = offsets_for(site, component, registry)
    path = stamp_path(data_path)
    if not entries:
        if path.is_file():
            path.unlink()
        return None
    content = stamp_content(site, component, entries, registry)
    path.write_text(json.dumps(content, indent=2, sort_keys=True))
    return path


def read_stamp(data_path) -> Optional[Dict[str, object]]:
    """The stamp beside ``data_path``, or None when there is none; ValueError when there is a damaged one."""
    path = stamp_path(data_path)
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError("{0} does not hold a clock stamp object".format(path.name))
    version = loaded.get("version")
    if version != STAMP_VERSION:
        raise ValueError(
            "{0} is a version {1!r} clock stamp and this reader knows version {2}".format(
                path.name, version, STAMP_VERSION
            )
        )
    return loaded


def stamp_errors(data_path, component: str, registry=REGISTRY) -> List[str]:
    """Every reason the Stage-3 input ``data_path`` lacks the provenance ``registry`` implies; empty when it has it.

    The merge's per-site check. It never raises: the merge's per-site ``except Exception: continue`` would
    turn a raise into a silently dropped site, which is a worse outcome than the mismatch it reports.
    """
    site = site_code(data_path)
    where = "{0}/{1}".format(site, component)
    expected = offsets_for(site, component, registry)
    try:
        stamp = read_stamp(data_path)
    except (OSError, ValueError) as exc:
        return ["{0}: its clock stamp cannot be read: {1}".format(where, exc)]
    recorded = list(stamp.get("offsets") or []) if stamp is not None else []
    if not expected and not recorded:
        return []
    if stamp is None:
        return [
            "{0}: the registry applies {1} and its Stage-3 input carries no clock stamp -- Stage 3 was "
            "not re-run after the registry named this site".format(where, describe(expected))
        ]
    if stamp.get("site") != site or stamp.get("component") != component:
        return [
            "{0}: its clock stamp is for {1}/{2} -- the stamp and the file it sits beside are not the "
            "same reading".format(where, stamp.get("site"), stamp.get("component"))
        ]
    if not expected:
        return [
            "{0}: its Stage-3 input was moved by {1}, which the registry no longer applies to this site -- "
            "re-run Stage 3 to put it back".format(where, describe(recorded))
        ]
    if stamp.get("offsets_sha256") != offsets_digest(expected):
        return [
            "{0}: its Stage-3 input was moved by {1}, and the registry now applies {2}".format(
                where, describe(recorded), describe(expected)
            )
        ]
    return []


def tree_provenance(errors_by_site: Mapping[str, Sequence[str]], registry=REGISTRY) -> Dict[str, object]:
    """The merge's clock-provenance record for the feature-contract block.

    ``errors_by_site`` maps every site WRITTEN to the tree to what :func:`stamp_errors` said about its two
    Stage-3 inputs, an empty list when they matched. Sites the merge skipped are absent, so a tree never
    answers for a site it does not contain.
    """
    mismatches = {site: list(reasons) for site, reasons in sorted(errors_by_site.items()) if reasons}
    corrected = [
        site
        for site in sorted(errors_by_site)
        if any(offsets_for(site, component, registry) for component in COMPONENTS)
    ]
    return {
        "definition": "per site, the clock stamp beside its Stage-3 input against the offsets the registry applies",
        "registry_sha256": registry_sha256(registry),
        "n_sites_checked": len(errors_by_site),
        "sites_corrected": corrected,
        "mismatches": mismatches,
    }


def provenance_errors(provenance) -> List[str]:
    """Why a tree's clock-provenance record fails the gate; empty when it passes.

    Absence is a failure, not a pass. A tree built before this check exists cannot show that its Stage-3
    inputs ever received the corrections its block claims, which is the whole question.
    """
    if not isinstance(provenance, dict):
        return [
            "no clock_provenance: the tree never checked its Stage-3 inputs against the clock registry, so "
            "its clock_registry_sha256 records only what the merge imported; rebuild it with the Stage-4 merge"
        ]
    checked = provenance.get("n_sites_checked")
    if isinstance(checked, bool) or not isinstance(checked, int) or checked <= 0:
        return [
            "clock_provenance.n_sites_checked is {0!r}: not one site's Stage-3 input was checked".format(checked)
        ]
    mismatches = provenance.get("mismatches")
    if not isinstance(mismatches, dict):
        return ["clock_provenance carries no mismatches map: the check did not run to completion"]
    errors: List[str] = []
    for site in sorted(mismatches)[:MAX_REPORTED_SITES]:
        for reason in list(mismatches[site]) or ["no reason recorded"]:
            errors.append("clock provenance: {0}".format(reason))
    beyond = len(mismatches) - MAX_REPORTED_SITES
    if beyond > 0:
        errors.append("clock provenance: and {0} further site(s) whose Stage-3 input does not match".format(beyond))
    return errors


def _acting(entry) -> Dict[str, object]:
    """The acting fields of a registry entry, or of a stamp's record of one."""
    row = entry if isinstance(entry, dict) else asdict(entry)
    return {field: row.get(field) for field in ACTING_FIELDS}


def _period(row: Mapping) -> str:
    start, end = row.get("start"), row.get("end")
    if start is None and end is None:
        return ""
    if end is None:
        return " from {0}".format(start)
    if start is None:
        return " until {0}".format(end)
    return " in [{0}, {1})".format(start, end)
