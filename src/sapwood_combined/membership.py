"""Which site takes which route into the combined corpus.

Two decisions live here:

1. **Precedence.** A site holding a measured sapwood series takes it; only a
   site that exists *solely* at plant scale is converted. A measurement always
   beats an estimate, and the estimate is only as good as the imputed area.
2. **Provenance.** Each route carries the ``*_asw_filled`` metadata for its own
   scale, because a converted site needs the plant-scale ASW while a native one
   needs the sapwood-scale copy of the same stems.

**Nothing is excluded here.** Every site the corpora hold enters the product.
Several pairs of sites do describe the same stems, and one stand entering twice
gets double weight in training -- but that is a modelling decision, not an
assembly one, and it is reversible only in one direction. So the overlaps are
*recorded* (:data:`OVERLAP_NOTES`) rather than acted on: the manifest carries
the evidence, and a consumer can filter on it. The only site that fails to
appear does so for a data reason, not a policy one -- ESP_GDN has no tree with a
usable sapwood area, so there is nothing to divide (see
:mod:`src.sapwood_combined.assembly`).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

EXTERNAL = "external"
INTERNAL = "internal"

ROUTE_NATIVE = "native"
ROUTE_CONVERTED = "converted"

_PLANT_MD_SUFFIX = "_plant_md.csv"
PLANT_MD_GLOB = "*_plant_md.csv"


def canonical_site(code: str) -> str:
    """Fold a site code so case or separator drift cannot miss a registry key.

    Mirrors :func:`src.sapwood_imputation.sources.canonical_site`. Site-keyed
    registries in this repository have silently stopped matching twice before --
    once on a hyphen-to-underscore migration, once on an ISO2-to-ISO3 rename --
    and in both cases the guard they protected simply stopped firing.
    """
    return re.sub(r"[^A-Z0-9]+", "_", str(code).upper()).strip("_")


#: Sites that describe stems another site also describes, with the evidence.
#:
#: **Advisory only.** Nothing here changes what is written; it is copied into the
#: manifest so a consumer can decide whether to filter, and so the analysis is
#: not lost. Established 2026-08-15 by a sweep of all 227 sites: co-location
#: within 3 km as the screen, then tree-level DBH matching within 0.15 cm and
#: ``pl_name`` overlap, then a correlation test on the shared window.
#:
#: That correlation test discriminates as follows: two series from the *same
#: stem* pair one-to-one at r ~ 0.98, whereas two stems in one stand reach only
#: r ~ 0.85-0.95 with several columns collapsing onto a single partner. It is
#: sharper in open stands than under closed canopy, where between-tree
#: correlation is already high.
#:
#: Pairs that share stems but NOT time are deliberately absent: the archive's own
#: pre/post and treatment splits (CAN_TUR_P39, SEN_SOU, USA_HIL_HF1,
#: USA_WIL_WC1/WC2) and the cross-corpus pairs FIN_HYY_SME/FIN_HYY and
#: CZE_RAJ_RAJ/CZE_RAJ. Those are repeated measures by design, not duplication.
OVERLAP_NOTES: dict[str, dict[str, str]] = {
    "SWE_NOR": {
        "overlaps": "SWE_NOR_ST1_AF1 + SWE_NOR_ST1_AF2",
        "relation": "same stems, merged plots",
        "evidence": "internal 26 trees = AF1(12) + AF2(14); pl_name identical, DBH and ASW to "
        "6 dp, series pair 1:1 at r=0.979-0.990. The archive pair additionally keeps the "
        "Af1/Af2 fertilisation treatment and st_age that this merged copy blanked",
    },
    "SWE_SKO_MIN": {
        "overlaps": "SWE_SGR",
        "relation": "same stems, contained in time",
        "evidence": "11/12 pair 1:1 at r=0.978-0.988; this archive copy's real data "
        "(2015-05..2015-10) lies entirely inside SWE_SGR's window (2015-05..2017-10), holding "
        "0 rows SWE_SGR lacks against 9322 the other way",
    },
    "CZE_BIK": {
        "overlaps": "CZE_BK1",
        "relation": "archive subset of internal",
        "evidence": "all 7 archive trees present in CZE_BK1's 41 by pl_name and DBH; archive "
        "covers 2016 only against 2015-2023; r=0.929-0.949 degenerate (5/7 partners)",
    },
    "CZE_STI": {
        "overlaps": "CZE_STN",
        "relation": "archive subset of internal",
        "evidence": "all 8 archive trees present in CZE_STN's 40 by pl_name (6/8 by DBH); "
        "archive covers 2016 only against 2016-2023; r=0.883-0.950 degenerate (3/8 partners)",
    },
    "CZE_LAN": {
        "overlaps": "CZE_LNZ",
        "relation": "archive subset of internal",
        "evidence": "13/17 archive trees find a distinct partner in CZE_LNZ's 69; archive "
        "covers 2016 only against 2016-2025; r=0.921-0.964",
    },
    "CHE_DAV_SEE": {
        "overlaps": "CHE_DAV",
        "relation": "archive subset of internal",
        "evidence": "all 5 archive trees present in CHE_DAV's 15 by pl_name (1/5 by DBH); "
        "archive covers 2010 only against 2010-2024; r=0.879-0.952 degenerate (3/5 partners)",
    },
    "FRA_HES_HE1_NON": {
        "overlaps": "FRA_HES",
        "relation": "archive subset of internal",
        "evidence": "its 10 pl_names are 10 of FRA_HES's 22 (HE1 + HE2 + 2 trees partition it "
        "exactly). UNVERIFIED by signal: the two files share no timestamp at all despite years "
        "of nominal overlap, so their sampling grids differ",
    },
    "FRA_HES_HE2_NON": {
        "overlaps": "FRA_HES",
        "relation": "archive subset of internal",
        "evidence": "its 10 pl_names are 10 of FRA_HES's 22; same unverified timestamp-grid "
        "mismatch as FRA_HES_HE1_NON",
    },
    "ESP_LMA": {
        "overlaps": "ESP_MAJ_MAI",
        "relation": "UNRESOLVED -- same inventory, probably not the same instrumented stems",
        "evidence": "5/6 DBH match within 0.09 cm, but 0 pl_name overlap, equal tree counts "
        "(not a subset), and the correlation is degenerate and LOWER against its supposed twin "
        "(0.867) than against a different plot, ESP_MAJ_NOR_LM1 (0.892) -- impossible if the "
        "stems were shared. Majadas del Tietar is a multi-plot fertilisation experiment "
        "(archive: Main + North LM1; internal: LMA + LM1 + LM2) and the archive reports one "
        "site-level coordinate for all its plots, so apparent plot distances mean nothing. "
        "Plot correspondence needs the provider. Species differ in name only: Q. ilex vs "
        "Q. rotundifolia is a segregate pair",
    },
}

_OVERLAP_INDEX = {canonical_site(k): k for k in OVERLAP_NOTES}


def overlap_note(site: str) -> dict[str, str] | None:
    """Advisory record of stems ``site`` shares with another, or ``None``."""
    declared = _OVERLAP_INDEX.get(canonical_site(site))
    return OVERLAP_NOTES[declared] if declared else None


@dataclass(frozen=True)
class Corpus:
    """One source of sites, in its two flavours."""

    origin: str
    plant_dir: Path
    sapwood_dir: Path


@dataclass(frozen=True)
class SiteRoute:
    """How one site reaches the combined corpus."""

    site: str
    origin: str
    route: str
    #: Directory holding this site's seven source files.
    source_dir: Path
    #: The ``*_asw_filled`` plant_md carrying this site's sapwood areas.
    asw_md_path: Path

    @property
    def is_converted(self) -> bool:
        return self.route == ROUTE_CONVERTED


def _sites_in(directory: Path) -> list[str]:
    if not Path(directory).is_dir():
        return []
    return sorted(p.name[: -len(_PLANT_MD_SUFFIX)] for p in Path(directory).glob(PLANT_MD_GLOB))


def resolve_routes(
    corpora: Iterable[Corpus],
    plant_asw_dir: Path,
    sapwood_asw_dir: Path,
) -> list[SiteRoute]:
    """Route every site of every corpus, excluding those barred.

    Raises on a filename collision between corpora -- they share one flat output
    directory, so a shared site code would silently overwrite one with the other
    -- and on a routed site whose filled-ASW metadata is missing, because
    skipping it quietly would drop a site from the product with no trace.
    """
    plant_asw_dir, sapwood_asw_dir = Path(plant_asw_dir), Path(sapwood_asw_dir)
    routes: list[SiteRoute] = []
    seen: dict[str, str] = {}

    for corpus in corpora:
        plant = _sites_in(corpus.plant_dir)
        sapwood = _sites_in(corpus.sapwood_dir)
        # Folded, and folded for the ROUTE decision too -- not just the collision
        # check below. Testing membership with the raw string while colliding on
        # the folded one means a site spelled differently in the two flavour
        # directories fails the native test and is converted instead: a measured
        # series silently replaced by a quotient of an imputed area, with the
        # output still looking complete.
        native = {canonical_site(s): s for s in sapwood}
        derived = {canonical_site(s): s for s in plant}

        # Iterate folded keys, so a site spelled two ways across the flavour
        # directories is ONE site rather than two candidates that then collide
        # with each other.
        for key in sorted(set(native) | set(derived)):
            if key in seen:
                raise ValueError(
                    f"Site {key!r} appears in both the {seen[key]} and {corpus.origin} corpora; they share one "
                    "output directory, so one would overwrite the other."
                )
            seen[key] = corpus.origin

            if key in native:
                # The sapwood flavour's own spelling: that is the directory the
                # files are read from, so its filenames are the ones on disk.
                site = native[key]
                route, source_dir, asw_dir = ROUTE_NATIVE, corpus.sapwood_dir, sapwood_asw_dir
            else:
                site = derived[key]
                route, source_dir, asw_dir = ROUTE_CONVERTED, corpus.plant_dir, plant_asw_dir

            asw_md_path = asw_dir / f"{site}{_PLANT_MD_SUFFIX}"
            if not asw_md_path.exists():
                raise FileNotFoundError(
                    "No filled sapwood-area metadata for {!r} at {}. Run "
                    "src.sapwood_imputation.impute_sapwood_area for the {} scale first.".format(
                        site, asw_md_path, "sapwood" if route == ROUTE_NATIVE else "plant"
                    )
                )
            routes.append(
                SiteRoute(
                    site=site,
                    origin=corpus.origin,
                    route=route,
                    source_dir=Path(source_dir),
                    asw_md_path=asw_md_path,
                )
            )
    return routes


def summarise_routes(routes: Iterable[SiteRoute]) -> dict[str, int]:
    """Counts per route and origin, for the manifest and the console."""
    out: dict[str, int] = {ROUTE_NATIVE: 0, ROUTE_CONVERTED: 0}
    for r in routes:
        out[r.route] = out.get(r.route, 0) + 1
        out[f"{r.origin}_{r.route}"] = out.get(f"{r.origin}_{r.route}", 0) + 1
    return out
