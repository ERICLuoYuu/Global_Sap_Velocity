"""Placing one site's files into the combined directory.

A native site is copied byte for byte. A converted site has exactly three files
rewritten -- ``sapf_data`` (divided), ``plant_md`` (restated at sapwood scale)
and ``sapf_flags`` (filtered to the surviving trees) -- and the rest copied
verbatim.

**How many files a site has depends on where it came from.** The archive ships
nine kinds; the internal corpus never generated ``sapf_flags`` or ``env_flags``
and ships seven. Measured 2026-08-15: 359/359 external files carry both flag
kinds, 0/30 internal files carry either. So the kind set is resolved per site
from what is on disk, and reported in the outcome, rather than assumed globally.

Unchanged files are never re-serialised. A pandas round-trip would silently
re-render timestamps and drift float precision on 3 GB of data that has no
reason to change, and would make the product impossible to diff against its
sources.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.sapwood_combined.conversion import (
    CONVERSION_COL,
    NATIVE,
    SAPWOOD_UNITS,
    UNITS_COL,
    align_flags,
    check_flags_columns,
    convert_metadata,
    convert_sapflow,
    data_columns,
    source_flag_counts,
)
from src.sapwood_combined.membership import ROUTE_CONVERTED, ROUTE_NATIVE, SiteRoute

#: Every file kind a site may ship, in the order they are written.
#:
#: The last two are archive-only. Omitting them would not raise anywhere: QC
#: resolves the flags file by name beside the data file and falls back to
#: ``flags = None`` when it is absent, so a combined corpus without them would
#: silently run with flag-based filtering -- step 1 of the six-step QC -- turned
#: off for every archive site.
FILE_KINDS = (
    "sapf_data",
    "plant_md",
    "env_data",
    "env_md",
    "site_md",
    "stand_md",
    "species_md",
    "sapf_flags",
    "env_flags",
)

#: Kinds a converted site rewrites rather than copies.
_REWRITTEN = ("sapf_data", "plant_md", "sapf_flags")

#: Precision for rewritten numeric data. The archive carries ~6 significant
#: digits; writing full float64 repr would inflate the files severalfold for
#: digits that are division noise.
FLOAT_FORMAT = "%.6g"

NO_CONVERTIBLE_TREE = "no tree has a usable sapwood area"


@dataclass(frozen=True)
class SiteOutcome:
    """What happened to one site. Feeds the manifest."""

    site: str
    origin: str
    route: str
    n_trees_in: int
    n_trees_out: int
    dropped: tuple[tuple[str, str], ...] = ()
    source_flags: dict[str, int] = field(default_factory=dict)
    files_written: int = 0
    #: Which of :data:`FILE_KINDS` this site actually wrote. Recorded because the
    #: count differs by corpus (9 archive, 7 internal), and without it a consumer
    #: cannot tell a legitimately flagless internal site from one that lost files.
    kinds: tuple[str, ...] = ()
    skipped_reason: str | None = None

    @property
    def was_written(self) -> bool:
        return self.skipped_reason is None


def place_file(src: Path, dst: Path, link: bool = False) -> None:
    """Put ``src`` at ``dst`` unchanged.

    With ``link``, try a hard link first: the combined corpus repeats ~3 GB of
    files that are identical to their source, and on one filesystem a link makes
    that free. Falls back to a copy whenever the link cannot be made -- a
    different device, a filesystem without link support, or an existing target.
    """
    if dst.exists():
        dst.unlink()
    if link:
        try:
            os.link(str(src), str(dst))
            return
        except OSError:
            pass
    shutil.copy2(str(src), str(dst))


def prune_unclaimed(out_dir: Path, outcomes: Iterable[SiteOutcome]) -> tuple[str, ...]:
    """Delete corpus files an earlier build left that this one does not write.

    ``--force`` rebuilds *over* a directory rather than replacing it. Files the
    new run writes are overwritten; files it does not write are simply left, so
    the directory accumulates the union of every build ever run there while
    MANIFEST.json describes only the last. Two ways that happens in practice:
    a full build followed by a ``--skip-internal`` one, and a site that has since
    lost its last usable sapwood area and is now skipped. In both the leftovers
    are not merely stale, they are *disowned* -- a consumer globbing the
    directory reads sites the manifest, the provenance block and
    ``corpora.internal_included`` all deny are present.

    Deletion is scoped by name: only ``{something}_{kind}.csv`` for a kind in
    :data:`FILE_KINDS` can be removed, so a note, a README or a derived table
    kept beside the product is left alone rather than assumed to be ours. Under
    ``--link`` the entries removed are hard links, so unlinking them frees the
    name without touching the source file they share an inode with.

    Returns the names removed, so the caller can report them. A rebuild that
    prunes nothing is the normal case and says nothing.
    """
    out_dir = Path(out_dir)
    claimed = {f"{o.site}_{kind}.csv" for o in outcomes if o.was_written for kind in o.kinds}
    removed = []
    for path in sorted(out_dir.glob("*.csv")):
        if path.name in claimed:
            continue
        if not any(path.name.endswith(f"_{kind}.csv") for kind in FILE_KINDS):
            continue
        path.unlink()
        removed.append(path.name)
    return tuple(removed)


def _read_csv(path: Path) -> pd.DataFrame:
    # dtype=str on timestamps is handled inside conversion; here we only need the
    # frame to survive the round trip, so let pandas infer and keep NA tokens.
    return pd.read_csv(path, low_memory=False)


def assemble_site(route: SiteRoute, out_dir: Path, link: bool = False) -> SiteOutcome:
    """Write one site into ``out_dir`` and report what happened.

    A site whose every tree lacks a usable area is skipped outright and writes
    nothing. Emitting a timestamps-only sapf_data would produce a file that
    reads downstream as a real site that simply never flowed.
    """
    out_dir = Path(out_dir)
    if route.route == ROUTE_NATIVE:
        return _assemble_native(route, out_dir, link)
    return _assemble_converted(route, out_dir, link)


def _present_kinds(route: SiteRoute) -> tuple[str, ...]:
    return tuple(k for k in FILE_KINDS if (route.source_dir / f"{route.site}_{k}.csv").exists())


def _check_already_at_sapwood_scale(route: SiteRoute, md: pd.DataFrame) -> None:
    """Refuse a 'native' site whose own metadata does not claim sapwood scale.

    Being in the sapwood flavour directory is a claim about the series, and this
    is where it gets tested. Without it the route simply *assigns* the sapwood
    unit, so a plant-scale file misfiled under ``csv/sapwood/`` would be
    relabelled rather than refused, and would sit in the product roughly 100x too
    large under a header asserting otherwise -- the exact failure
    :mod:`src.sapwood_combined.conversion` exists to prevent, arriving by the one
    route that does no arithmetic.

    It is also what makes the claim checkable. Reading ``pl_sap_units`` back out
    of the output cannot fail while the code writes it unconditionally, so both
    the integration test and the build's acceptance criterion 2 were asserting a
    constant. Measured 2026-08-16 across ``sapwood_asw_filled/``: 181/181 native
    sites already declare ``cm3 cm-2 h-1``, so this refuses nothing we hold.
    """
    if UNITS_COL not in md.columns:
        raise ValueError(
            f"{route.site}: {route.asw_md_path.name} has no {UNITS_COL!r} column, so nothing states "
            "the series is at sapwood scale. Assigning the unit anyway would make the product's "
            "central claim unverifiable."
        )
    declared = sorted(set(md[UNITS_COL].dropna().astype(str)) - {SAPWOOD_UNITS})
    if declared:
        raise ValueError(
            f"{route.site} is routed as native because it appears in the sapwood corpus, but its "
            f"metadata declares {declared} rather than {SAPWOOD_UNITS!r}. Refusing to relabel it: "
            "if the series really is at plant scale it must be divided by a sapwood area, not "
            "renamed."
        )


def _assemble_native(route: SiteRoute, out_dir: Path, link: bool) -> SiteOutcome:
    """Copy verbatim, but take plant_md from the filled-ASW directory.

    The series is already at sapwood scale, so nothing about it changes. The
    metadata is taken from ``sapwood_asw_filled/`` rather than the raw archive so
    that every site in the product carries the same columns -- a consumer reading
    ``pl_sapw_area_filled`` should not have to know which route a site took.
    """
    kinds = _present_kinds(route)

    md = _read_csv(route.asw_md_path)
    _check_already_at_sapwood_scale(route, md)

    # The converted route validates flags against data and raises. Copying
    # verbatim is not a reason to skip the check -- it is the reason it was
    # missing, and it left 181 of 226 sites outside the one guard that protects
    # the pairing sap_analyzer relies on.
    if "sapf_flags" in kinds and "sapf_data" in kinds:
        check_flags_columns(
            pd.read_csv(route.source_dir / f"{route.site}_sapf_flags.csv", nrows=0).columns,
            pd.read_csv(route.source_dir / f"{route.site}_sapf_data.csv", nrows=0).columns,
        )

    for kind in kinds:
        if kind == "plant_md":
            continue
        place_file(
            route.source_dir / f"{route.site}_{kind}.csv",
            out_dir / f"{route.site}_{kind}.csv",
            link,
        )

    md[UNITS_COL] = SAPWOOD_UNITS
    md[CONVERSION_COL] = NATIVE
    md.to_csv(out_dir / f"{route.site}_plant_md.csv", index=False, float_format=FLOAT_FORMAT)

    sapf = route.source_dir / f"{route.site}_sapf_data.csv"
    n_trees = len(data_columns(pd.read_csv(sapf, nrows=0).columns)) if sapf.exists() else 0

    return SiteOutcome(
        site=route.site,
        origin=route.origin,
        route=ROUTE_NATIVE,
        n_trees_in=n_trees,
        n_trees_out=n_trees,
        source_flags=source_flag_counts(md),
        files_written=len(kinds),
        kinds=kinds,
    )


def _assemble_converted(route: SiteRoute, out_dir: Path, link: bool) -> SiteOutcome:
    """Divide sapf_data, restate plant_md, filter sapf_flags, copy the rest."""
    sapf_path = route.source_dir / f"{route.site}_sapf_data.csv"
    md = _read_csv(route.asw_md_path)
    sapflow = _read_csv(sapf_path)
    n_in = len(data_columns(sapflow.columns))

    result = convert_sapflow(sapflow, md)
    asw_flags = source_flag_counts(md)
    kinds = _present_kinds(route)

    if not result.converted:
        return SiteOutcome(
            site=route.site,
            origin=route.origin,
            route=ROUTE_CONVERTED,
            n_trees_in=n_in,
            n_trees_out=0,
            dropped=result.dropped,
            source_flags=asw_flags,
            files_written=0,
            kinds=(),
            skipped_reason=NO_CONVERTIBLE_TREE,
        )

    for kind in kinds:
        if kind in _REWRITTEN:
            continue
        place_file(
            route.source_dir / f"{route.site}_{kind}.csv",
            out_dir / f"{route.site}_{kind}.csv",
            link,
        )

    result.frame.to_csv(out_dir / f"{route.site}_sapf_data.csv", index=False, float_format=FLOAT_FORMAT)

    convert_metadata(md, result.converted, result.dropped).to_csv(
        out_dir / f"{route.site}_plant_md.csv", index=False, float_format=FLOAT_FORMAT
    )

    if "sapf_flags" in kinds:
        src = route.source_dir / f"{route.site}_sapf_flags.csv"
        dst = out_dir / f"{route.site}_sapf_flags.csv"
        # Validate from the header, before choosing copy or filter, so the cheap
        # path is checked as thoroughly as the rewriting one.
        check_flags_columns(pd.read_csv(src, nrows=0).columns, sapflow.columns)
        if result.dropped:
            # dtype=str + keep_default_na=False so an empty flag cell stays the
            # empty string it is on disk instead of becoming NaN. Flag values are
            # labels, and a NaN round-trip erases the difference between "no flag
            # here" and "this value is missing".
            source = pd.read_csv(src, dtype=str, keep_default_na=False, low_memory=False)
            align_flags(source, result.converted).to_csv(dst, index=False)
        else:
            # No tree was dropped, so the column set is unchanged and there is
            # nothing to filter. Copy, rather than putting 2.9 GB of flags
            # through a pandas round-trip that cannot change them.
            place_file(src, dst, link)

    return SiteOutcome(
        site=route.site,
        origin=route.origin,
        route=ROUTE_CONVERTED,
        n_trees_in=n_in,
        n_trees_out=len(result.converted),
        dropped=result.dropped,
        source_flags=asw_flags,
        files_written=len(kinds),
        kinds=kinds,
    )
