"""Resolve one IGBP land-cover class per site.

Two sources disagree about how many values a site has:

* the provider's ``si_igbp`` in ``<site>_site_md.csv`` is **one** short IGBP
  code per site, and it is what the merge writes into the ``pft`` column;
* the extracted MODIS MCD12Q1 ``LC_Type1`` is **one value per site-year**,
  because the product is annual.

This module collapses the MODIS side to one value per site by taking the mode
**across years** -- each year counted once -- and then fills in for the sites
whose ``si_igbp`` the provider never supplied.

Why the mode across years rather than across rows
-------------------------------------------------
MCD12Q1 reclassifies a pixel from one year to the next far more readily than
the vegetation actually changes.  Measured over the 226-site ``sapwood_combined``
corpus on 2026-08-17: 209 sites hold one class for their whole record, and 17
flip.  Davos (``CHE_DAV``) alternates between Evergreen Needleleaf and Woody
Savanna over 15 years -- it is a subalpine spruce stand, so ENF is right and the
mode recovers it while any single year is a coin toss.

The mode must weight **years**, not rows.  The extracted table is expanded to
hourly, so a site's first year can hold 8,760 rows against a later year's 700
and a row-wise mode would let one long year outvote several short ones.
``site_landcover_modes`` therefore reduces to one value per site-year first.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

#: MODIS MCD12Q1 ``LC_Type1`` (IGBP scheme) to the short IGBP abbreviation.
#: This is the vocabulary ``si_igbp`` already uses -- measured over the corpus,
#: the provider values are exactly ENF/DBF/EBF/MF/SAV/DNF/WSA/WET -- so a
#: fallback expressed in long names ("Evergreen Needleleaf Forests") would sit
#: in the same column as a different vocabulary and one-hot into its own
#: category.
IGBP_BY_LC_TYPE1 = {
    1: "ENF",  # Evergreen Needleleaf Forests
    2: "EBF",  # Evergreen Broadleaf Forests
    3: "DNF",  # Deciduous Needleleaf Forests
    4: "DBF",  # Deciduous Broadleaf Forests
    5: "MF",  # Mixed Forests
    6: "CSH",  # Closed Shrublands
    7: "OSH",  # Open Shrublands
    8: "WSA",  # Woody Savannas
    9: "SAV",  # Savannas
    10: "GRA",  # Grasslands
    11: "WET",  # Permanent Wetlands
    12: "CRO",  # Croplands
    13: "URB",  # Urban and Built-up Lands
    14: "CVM",  # Cropland / Natural Vegetation Mosaics
    15: "SNO",  # Permanent Snow and Ice
    16: "BSV",  # Barren or Sparsely Vegetated
    17: "WAT",  # Water Bodies
}

#: Anything in this set means "the provider did not tell us".
_ABSENT = {"", "nan", "na", "n/a", "none", "null", "-9999", "unknown"}

SITE_COLUMN = "site_name"
CLASS_COLUMN = "landcover_type"

#: Column holding the short IGBP name, written by ``extract_pft`` alongside the
#: raw integer.  The integer is KEPT rather than replaced: it is the only
#: recoverable record of what Earth Engine returned, re-extraction is one
#: ``getInfo`` round trip per site-year, and a consumer that still maps the
#: integer (``notebooks/merge_gap_filled_hourly.py`` does) would turn every row
#: into NaN *without raising* if the column changed dtype.
CODE_COLUMN = "pft"

#: The only values ``CODE_COLUMN`` may hold.
IGBP_CODES = frozenset(IGBP_BY_LC_TYPE1.values())


class LandcoverVocabularyError(ValueError):
    """An ``LC_Type1`` value outside the published 1..17 IGBP scheme."""


def igbp_code(landcover_value) -> str:
    """One raw ``LC_Type1`` value as the short IGBP code that ``si_igbp`` uses.

    Lives here, not in the extractor, so the repo holds exactly ONE copy of the
    mapping: a second private copy in a second live script is how the VPD unit
    error survived being fixed.  Raises rather than returning ``None`` -- a
    blank name written into the product would be indistinguishable from a site
    MODIS could not classify, which is the failure this column exists to remove.
    """
    if landcover_value is None:
        raise LandcoverVocabularyError(
            "LC_Type1 is None -- the point is outside the image or over masked water"
        )
    # NB: LandcoverVocabularyError subclasses ValueError, so it must never be
    # raised inside a `try` that catches ValueError -- it would be swallowed.
    try:
        missing = bool(pd.isna(landcover_value))
    except (TypeError, ValueError):
        missing = False
    if missing:
        raise LandcoverVocabularyError("LC_Type1 is NaN -- no class was returned")
    try:
        code = int(landcover_value)
    except (TypeError, ValueError):
        code = None
    if code is None:
        raise LandcoverVocabularyError(
            "LC_Type1 %r is not an integer land-cover code" % (landcover_value,)
        )
    if code not in IGBP_BY_LC_TYPE1:
        raise LandcoverVocabularyError(
            "LC_Type1 %d is outside the published 1..17 IGBP scheme" % code
        )
    return IGBP_BY_LC_TYPE1[code]


def _class_series(series: pd.Series, numeric: bool) -> pd.Series:
    """The class column as something ``dropna`` can filter.

    Numeric for the raw ``LC_Type1``; stripped text for the short code, with the
    absent-value vocabulary blanked so a missing name is dropped rather than
    becoming the literal class ``"nan"``.
    """
    if numeric:
        return pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip()
    return text.mask(text.str.lower().isin(_ABSENT))


def _check_code_matches_raw(frame: pd.DataFrame) -> None:
    """Both columns present means they must agree, row for row.

    The redundancy is the point: it turns a hand-edited file, a half-finished
    re-extraction, or a future change of vocabulary into a loud failure instead
    of a silently different PFT.
    """
    if CLASS_COLUMN not in frame.columns:
        return
    raw = pd.to_numeric(frame[CLASS_COLUMN], errors="coerce")
    code = frame[CODE_COLUMN].astype(str).str.strip()
    both = raw.notna() & ~code.str.lower().isin(_ABSENT)
    if not both.any():
        return
    expected = raw[both].astype(int).map(IGBP_BY_LC_TYPE1)
    disagree = code[both] != expected
    if disagree.any():
        sample = [
            (int(raw[both][i]), expected[i], code[both][i])
            for i in code[both][disagree].index[:5]
        ]
        raise LandcoverVocabularyError(
            "%d row(s) where %r disagrees with %r; (LC_Type1, expected, found): %r"
            % (int(disagree.sum()), CODE_COLUMN, CLASS_COLUMN, sample)
        )


def is_absent(value) -> bool:
    """True when a metadata field carries no usable value.

    Checked as text after stripping, so ``NaN``, ``""``, ``"NA"`` and ``"na"``
    are all one case.  ``si_igbp`` arrives as ``float('nan')`` at the 23 sites
    that lack it, but SAPFLUXNET writes literal ``"NA"`` elsewhere.
    """
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in _ABSENT


def _year_series(frame: pd.DataFrame) -> pd.Series:
    """Calendar year per row, from whichever time column the product carries.

    ``extract_pft`` used to expand every site-year to hourly rows
    (``timestamp``); it now emits one row per site-year (``year``).  Accepting
    both means the reduction works against an already-extracted product and
    against a re-extracted one without a flag day.
    """
    if "year" in frame.columns:
        return pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    if "timestamp" in frame.columns:
        return pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).dt.year.astype("Int64")
    raise KeyError(f"landcover frame needs a 'year' or 'timestamp' column; got {list(frame.columns)}")


def site_year_classes(frame: pd.DataFrame, column: str = CLASS_COLUMN, numeric: bool = True) -> dict:
    """``{site_name: {year: LC_Type1}}``, one class per site-year.

    A site-year holding more than one distinct class would mean the extraction
    changed its mind mid-year, which the annual product cannot do; the modal
    value is taken rather than raising, and the caller can see it in the
    per-site year map.
    """
    year = _year_series(frame)
    grouped = (
        pd.DataFrame(
            {
                SITE_COLUMN: frame[SITE_COLUMN],
                "year": year,
                "class": _class_series(frame[column], numeric),
            }
        )
        .dropna(subset=[SITE_COLUMN, "year", "class"])
        .groupby([SITE_COLUMN, "year", "class"])
        .size()
    )

    per_site: dict = {}
    for (site, yr, cls), rows in grouped.items():
        by_year = per_site.setdefault(site, {})
        best = by_year.get(int(yr))
        if best is None or rows > best[1]:
            by_year[int(yr)] = (int(cls) if numeric else str(cls), int(rows))
    return {site: {y: v[0] for y, v in years.items()} for site, years in per_site.items()}


def modal_class(per_year: dict):
    """The class holding the most years; ties go to the more recent year.

    Returns ``None`` for an empty mapping.  The tie-break is deterministic
    rather than dictionary-order so that re-running the pipeline on the same
    inputs cannot quietly change a site's PFT.
    """
    if not per_year:
        return None
    counts = Counter(per_year.values())
    most = max(counts.values())
    tied = [cls for cls, n in counts.items() if n == most]
    if len(tied) == 1:
        return tied[0]
    return max(tied, key=lambda cls: max(y for y, c in per_year.items() if c == cls))


def site_landcover_modes(frame: pd.DataFrame) -> dict:
    """``{site_name: short IGBP code}`` -- one value per site.

    Raises on an ``LC_Type1`` outside 1..17 instead of dropping it: an
    unmapped code would otherwise become ``NaN`` and be indistinguishable from
    a site that has no MODIS value at all.
    """
    if CODE_COLUMN in frame.columns:
        _check_code_matches_raw(frame)
        by_site = site_year_classes(frame, column=CODE_COLUMN, numeric=False)
        stored = {}
        rejected = {}
        for site, per_year in by_site.items():
            code = modal_class(per_year)
            if code is None:
                continue
            if code in IGBP_CODES:
                stored[site] = code
            else:
                rejected[site] = code
        if rejected:
            raise LandcoverVocabularyError(
                "%r value(s) outside the short IGBP vocabulary: %r -- a long name "
                "here would one-hot into its own category alongside the codes in "
                "si_igbp" % (CODE_COLUMN, rejected)
            )
        return stored

    per_site_year = site_year_classes(frame)
    resolved = {}
    unknown = {}
    for site, per_year in per_site_year.items():
        code = modal_class(per_year)
        if code is None:
            continue
        if code not in IGBP_BY_LC_TYPE1:
            unknown[site] = code
            continue
        resolved[site] = IGBP_BY_LC_TYPE1[code]
    if unknown:
        raise LandcoverVocabularyError(f"LC_Type1 value(s) outside the published 1..17 IGBP scheme: {unknown}")
    return resolved


def load_site_landcover_modes(path) -> dict:
    """Read the extracted land-cover product and reduce it to one value per site.

    Only the three columns the reduction needs are read.  The hourly-expanded
    form of this product is 303 MB and 5.4 M rows for what is ~615 distinct
    site-year values, and the merge used to load all of it (and then never join
    it).
    """
    header = pd.read_csv(path, nrows=0).columns.tolist()
    time_column = "year" if "year" in header else "timestamp"
    if time_column not in header:
        raise KeyError(f"{path} has neither 'year' nor 'timestamp'; columns are {header}")
    if SITE_COLUMN not in header:
        raise KeyError(f"{path} has no {SITE_COLUMN!r} column; columns are {header}")
    if CODE_COLUMN not in header and CLASS_COLUMN not in header:
        raise KeyError(
            f"{path} has neither {CODE_COLUMN!r} nor {CLASS_COLUMN!r}; columns are {header}"
        )

    usecols = [SITE_COLUMN, time_column]
    if CLASS_COLUMN in header:
        usecols.append(CLASS_COLUMN)
    if CODE_COLUMN in header:
        usecols.append(CODE_COLUMN)

    frame = pd.read_csv(path, usecols=usecols)
    return site_landcover_modes(frame)


def resolve_pft(provider_value, fallback, derived=None):
    """One site's ``pft``, and where it came from.

    Returns ``(value, source)`` with source in ``{"provider", "derived",
    "modis", "unresolved"}``.  ``derived`` comes from
    ``src.site_info.derive_igbp``, which reads the stand and species metadata
    the site itself ships.  It ranks above MODIS because it is site-specific
    evidence, while MODIS is a 463 m pixel that put a holm-oak dehesa in
    Croplands and a maritime-pine plantation in Woody Savannas.

    Never raises: the merge calls this inside a ``try`` whose
    ``except`` skips the whole site, so a raise here would silently drop a site
    from the training set.
    """
    if not is_absent(provider_value):
        return str(provider_value).strip(), "provider"
    if not is_absent(derived):
        return str(derived).strip(), "derived"
    if fallback is not None and not is_absent(fallback):
        return str(fallback).strip(), "modis"
    return None, "unresolved"


def summarise(sources: dict) -> str:
    """One log line per source, for printing once at the end of a merge run."""
    counts = Counter(sources.values())
    parts = [f"{counts.get(k, 0)} {k}" for k in ("provider", "derived", "modis", "unresolved")]
    derived = sorted(s for s, src in sources.items() if src == "derived")
    borrowed = sorted(s for s, src in sources.items() if src == "modis")
    unresolved = sorted(s for s, src in sources.items() if src == "unresolved")
    lines = [f"pft resolution: {', '.join(parts)} (of {len(sources)} sites)"]
    if derived:
        lines.append(f"  metadata-derived: {derived}")
    if borrowed:
        lines.append(f"  MODIS-derived: {borrowed}")
    if unresolved:
        lines.append(f"  UNRESOLVED (pft will be null): {unresolved}")
    return "\n".join(lines)
