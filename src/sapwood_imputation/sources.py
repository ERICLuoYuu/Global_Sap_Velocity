"""Where plant metadata comes from, and which trees may vote on the allometry.

Two corpora feed the imputer:

**external**
    the SAPFLUXNET v0.1.5 archive under :attr:`PathConfig.metadata_dir`
    (194 plant / 165 sapwood sites, 2094 trees with measured ASW).

**internal**
    the 25 European sites processed by ``Sapflow-internal/``, written in the
    same SAPFLUXNET schema under :attr:`PathConfig.internal_metadata_dir`
    (14 plant / 16 sapwood sites).

Every discovered file is *processed* -- it gets ``pl_sapw_area_filled`` written
out -- but not every tree may *vote* on the global fit. Eleven internal sites
stand in the same forest as a SAPFLUXNET site, and three of them re-measure the
very same stems. Pooling those hands one tree two votes and quietly
over-weights a single stand.

Duplicate evidence (verified tree-by-tree, 2026-08-13)
------------------------------------------------------
==========  ==============================  ========================================
internal    external holder                 evidence
==========  ==============================  ========================================
SWE_NOR     SWE_NOR_ST1_AF1 + ST1_AF2       26/26 ``pl_name`` match, DBH and ASW
                                            identical to 6 decimals (481.977661 ...)
SWE_SGR     SWE_SKO_MIN                     0.00 km apart, same species and stand
                                            age 55; 12 vs 11 stems, DBH pairs 1:1
                                            (28.7/29, 47.6/48, 55.0/55 ...) with the
                                            archive copy rounded to whole cm
ESP_LMA     ESP_MAJ_MAI                     0.17 km apart; 5/6 DBH match to 0.1 cm
                                            (47.2/47.29, 34.6/34.56, 49.4/49.44,
                                            38.9/38.97, 44.1/44.11); species differ
                                            only as the *Quercus ilex* /
                                            *Q. rotundifolia* segregate pair
==========  ==============================  ========================================

The remaining eight same-forest pairs (CHE_DAV, CZE_BK1, CZE_LNZ, CZE_RAJ,
CZE_STN, ESP_LM1, ESP_LM2, FIN_HYY, FRA_HES) hold **no** measured ASW on the
internal side, so they cannot contaminate the fit and are left to vote-by-
absence. They are recorded in :data:`SAME_FOREST_NO_MEASURED_ASW` because they
are candidates for a future join -- FRA_HES's 20 stems and CHE_DAV's 5 already
have a *measured* archive value that would beat any allometric estimate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

EXTERNAL = "external"
INTERNAL = "internal"

PLANT_MD_GLOB = "*_plant_md.csv"
_PLANT_MD_SUFFIX = "_plant_md.csv"


def canonical_site(code: str) -> str:
    """Fold a site code so case and separator drift cannot miss a registry key.

    ``Sapflow-internal.column_naming.SiteMap`` solves the same problem, but that
    package lives under a hyphenated directory and is not importable from here;
    this is the same fold, kept local so the module stays self-contained.
    """
    return re.sub(r"[^A-Z0-9]+", "_", str(code).upper()).strip("_")


#: internal si_code -> the external site(s) already holding those stems.
#: Their measured trees are excluded from the fit pool (see module docstring).
DUPLICATE_SITES = {
    "SWE_NOR": "SWE_NOR_ST1_AF1 + SWE_NOR_ST1_AF2",
    "SWE_SGR": "SWE_SKO_MIN",
    "ESP_LMA": "ESP_MAJ_MAI",
}

#: Same forest as an archive site but no measured ASW internally, so harmless to
#: the fit. Listed as documentation of a known -- and better -- alternative to
#: imputing them: joining the archive's measured value on ``pl_name``.
SAME_FOREST_NO_MEASURED_ASW = {
    "CHE_DAV": "CHE_DAV_SEE",
    "CZE_BK1": "CZE_BIK",
    "CZE_LNZ": "CZE_LAN",
    "CZE_RAJ": "CZE_RAJ_RAJ",
    "CZE_STN": "CZE_STI",
    "ESP_LM1": "ESP_MAJ_MAI",
    "ESP_LM2": "ESP_MAJ_MAI",
    "FIN_HYY": "FIN_HYY_SME",
    "FRA_HES": "FRA_HES_HE1_NON + FRA_HES_HE2_NON",
}

_DUPLICATE_INDEX = {canonical_site(k): k for k in DUPLICATE_SITES}


def duplicate_holder(site: str) -> str | None:
    """External site(s) already holding ``site``'s stems, or ``None``."""
    declared = _DUPLICATE_INDEX.get(canonical_site(site))
    return DUPLICATE_SITES[declared] if declared else None


@dataclass(frozen=True)
class MetadataFile:
    """One ``*_plant_md.csv``, tagged with where it came from."""

    path: Path
    site: str
    origin: str
    #: False when this site's measured trees already exist in the other corpus.
    may_vote: bool
    #: Set when ``may_vote`` is False, naming the external holder.
    duplicate_of: str | None = None
    #: Which dataset flavour this copy was read from ('plant' / 'sapwood').
    #: A site can exist at both; the two copies describe the SAME stems, so only
    #: one may vote -- see :func:`discover_across_scales`.
    scale: str = ""

    @property
    def name(self) -> str:
        return self.path.name


def _site_of(path: Path) -> str:
    return path.name[: -len(_PLANT_MD_SUFFIX)]


def _collect(directory: Path, origin: str, scale: str = "") -> list[MetadataFile]:
    files: list[MetadataFile] = []
    for path in sorted(directory.glob(PLANT_MD_GLOB)):
        site = _site_of(path)
        holder = duplicate_holder(site) if origin == INTERNAL else None
        files.append(
            MetadataFile(
                path=path,
                site=site,
                origin=origin,
                may_vote=holder is None,
                duplicate_of=holder,
                scale=scale,
            )
        )
    return files


def discover(
    metadata_dir: Path,
    internal_dir: Path | None = None,
    scale: str = "",
) -> list[MetadataFile]:
    """List every plant_md file to process, external first.

    ``internal_dir`` is optional and silently skipped when absent -- the
    internal corpus is a working tree, not a shipped dataset, so a checkout
    without it must still run. Passing a directory that exists but holds no
    plant metadata *is* an error: that means the path is wrong.

    Raises on a filename collision between the corpora. Both are written into
    one output directory keyed by filename, so a collision would silently
    overwrite one site with another.
    """
    files = _collect(Path(metadata_dir), EXTERNAL, scale)
    if internal_dir is not None:
        internal_path = Path(internal_dir)
        if internal_path.is_dir():
            internal = _collect(internal_path, INTERNAL, scale)
            if not internal:
                raise FileNotFoundError(f"No {PLANT_MD_GLOB} files in internal metadata dir {internal_path}")
            clashes = sorted({f.name for f in files} & {f.name for f in internal})
            if clashes:
                raise ValueError(
                    "Internal and external metadata share filenames, which would overwrite "
                    f"one another in a single output directory: {clashes}"
                )
            files.extend(internal)
    return files


def discover_across_scales(
    metadata_dirs: dict[str, Path],
    internal_dirs: dict[str, Path] | None = None,
) -> list[MetadataFile]:
    """Every plant_md across all given scales, each copy tagged with its scale.

    ``plant`` and ``sapwood`` are two normalisations of the same *flux*; the
    stem geometry in ``plant_md`` is the same measurement either way, and for
    157 of 157 shared archive sites the two copies are byte-identical. So a
    site present at both scales appears here TWICE, describing one set of
    stems. Letting both vote would give those trees a double weight -- exactly
    the error :data:`DUPLICATE_SITES` guards against between corpora.

    This function does not resolve that; it only reports what exists. The
    caller picks one copy per site, because "which copy is better" depends on
    which columns it needs -- ITA_CP2's plant copy carries DBH but no ASW at
    all, while its sapwood copy has both for the same eight stems.

    Missing scale directories are skipped, so a partial checkout still runs.
    """
    files: list[MetadataFile] = []
    for scale, directory in metadata_dirs.items():
        if not Path(directory).is_dir():
            continue
        internal = (internal_dirs or {}).get(scale)
        files.extend(discover(Path(directory), internal, scale=scale))
    return files


def summarise(files: list[MetadataFile]) -> dict:
    """Counts per origin plus the excluded duplicates, for the JSON report."""
    by_origin: dict[str, int] = {}
    for f in files:
        by_origin[f.origin] = by_origin.get(f.origin, 0) + 1
    return {
        "files_by_origin": by_origin,
        "duplicate_sites_excluded_from_fit": {f.site: f.duplicate_of for f in files if not f.may_vote},
    }
