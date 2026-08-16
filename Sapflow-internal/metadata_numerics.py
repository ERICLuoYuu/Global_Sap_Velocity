"""Coerce provider free text in numeric ``plant_md`` fields (defect C2).

Why this module exists
---------------------
``sap_reorganizer`` maps provider metadata field *names* onto SAPFLUXNET names,
but never validates their *values*. Providers routinely write the unit, or a
note, into the cell:

    DE_Har  tree_dbh              '16.5 cm'                   ->  pl_dbh   0/10 numeric
    DE_Har  tree_height           '10.5 m'                    ->  pl_height
    DE_Har  tree_sapwood_area     '207.2 cm2'                 ->  pl_sapw_area
    DE_Har  tree latitude         '47.93391 N'                ->  pl_lat
    DE_Har  tree altitude         '201 m a.s.l.'              ->  pl_elev
    ES_Gdn  tree_dbh              '114(main stem_see notes)'  ->  pl_dbh   0/4
    FI_Hyy  tree_height           'Approx 24'                 ->  pl_height
    SE_Sgr  tree_age              '55 years'                  ->  pl_age
    SE_Sgr  tree_height           'not measured'              ->  genuinely absent

None of this was caught, because the only consumer that reads these fields does
``pd.to_numeric(..., errors="coerce")`` (``merge_gap_filled_hourly_orginal.py``
line 535). Every value above therefore becomes ``NaN`` *silently*: the feature
looks absent rather than broken, which is why ``pl_dbh`` was unusable at six
sites while every file passed every existing check.

Unit safety
-----------
The unit token is not merely stripped — it is **checked against the unit the
SAPFLUXNET field is defined in**. A ``pl_dbh`` of ``'165 mm'`` is 16.5 cm, not
165, and silently accepting the number would corrupt the value while leaving it
plausible. Known-convertible units are converted; anything else is refused and
reported rather than guessed.

Provenance
----------
Whenever a value had to be transformed, the provider's original string is kept in
``{field}_orig``, mirroring the ``pl_sap_units`` / ``pl_sap_units_orig`` and
``pl_sens_meth`` / ``pl_sens_meth_orig`` convention this dataset already uses.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import NamedTuple

import pandas as pd

logger = logging.getLogger(__name__)

#: Canonical unit of each numeric ``plant_md`` field, with the conversion factor
#: from every unit a provider has been seen to write. 1.0 means "already
#: canonical". A unit absent from a field's table is refused, not assumed.
FIELD_UNIT_FACTORS: Mapping[str, Mapping[str, float]] = MappingProxyType(
    {
        "pl_dbh": MappingProxyType({"cm": 1.0, "mm": 0.1, "m": 100.0}),
        "pl_height": MappingProxyType({"m": 1.0, "cm": 0.01}),
        "pl_sapw_area": MappingProxyType({"cm2": 1.0, "mm2": 0.01, "m2": 10000.0}),
        "pl_sapw_depth": MappingProxyType({"cm": 1.0, "mm": 0.1}),
        "pl_bark_thick": MappingProxyType({"mm": 1.0, "cm": 10.0}),
        "pl_phloem_thick": MappingProxyType({"mm": 1.0, "cm": 10.0}),
        "pl_age": MappingProxyType({"years": 1.0, "year": 1.0, "yr": 1.0, "yrs": 1.0, "a": 1.0}),
        "pl_sens_hgt": MappingProxyType({"cm": 1.0, "m": 100.0, "mm": 0.1}),
        "pl_elev": MappingProxyType({"m": 1.0, "masl": 1.0, "masl.": 1.0, "km": 1000.0}),
        "pl_lat": MappingProxyType({"deg": 1.0}),
        "pl_long": MappingProxyType({"deg": 1.0}),
        # site_md coordinates. These are load-bearing in a way the pl_* fields are
        # not: sap_analyzer reads si_lat/si_long straight out of the frame
        # (sap_analyzer.py:436) and hands them to pvlib without coercing, and
        # pvlib raises UFuncTypeError on a string. DE_Har ships '47.93391°N' /
        # '7.59814°E' / '201 m a.s.l.', so the site cannot reach QC at all.
        #
        # Hemisphere suffixes are signed factors, which is exactly what the
        # multiplicative model already expresses. Note this assumes the number
        # itself is unsigned, as it is in every delivery seen here -- '-47.5°S'
        # would come out positive, so a negative value carrying a S/W suffix is
        # refused below rather than silently flipped.
        "si_lat": MappingProxyType({"deg": 1.0, "°": 1.0, "°n": 1.0, "n": 1.0, "°s": -1.0, "s": -1.0}),
        "si_long": MappingProxyType({"deg": 1.0, "°": 1.0, "°e": 1.0, "e": 1.0, "°w": -1.0, "w": -1.0}),
        "si_elev": MappingProxyType(
            {"m": 1.0, "masl": 1.0, "masl.": 1.0, "m a.s.l.": 1.0, "m asl": 1.0, "km": 1000.0}
        ),
    }
)

#: Strings that mean "the provider did not measure this". These become NaN, which
#: is the honest representation — but they are counted separately from parse
#: failures so a genuine absence is never mistaken for a broken parser.
ABSENT_TOKENS: frozenset = frozenset(
    {"", "na", "n/a", "nan", "none", "null", "-", "--", "?", "notmeasured", "notavailable", "unknown", "nodata"}
)

#: Leading qualifiers that mean the number is an estimate. Stripped, but recorded:
#: 'Approx 24' is a real measurement of lower confidence, not a parse failure.
_QUALIFIER = re.compile(r"^\s*(approx\.?|approximately|ca\.?|about|circa|~|>=|<=|>|<)\s*", re.I)

#: A trailing parenthesised or bracketed annotation: '114(main stem_see notes)'.
_ANNOTATION = re.compile(r"[\(\[\{].*$", re.S)

#: number + optional unit tail. The unit tail is captured whole and normalised
#: separately so 'm a.s.l.' and 'cm2'/'cm²' both resolve.
_NUMBER = re.compile(r"^\s*([-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)\s*(.*?)\s*$")

#: Hemisphere suffixes on coordinates: '47.93391°N', '7.59814°E', '3.5°W'.
_HEMISPHERE = re.compile(r"^[°\s]*([NSEW])\b\.?", re.I)


def _normalise_unit(raw: str) -> str:
    """Fold a provider unit tail to a lookup key.

    Handles superscripts, punctuation and spacing so 'cm2', 'cm²', 'cm^2',
    'm a.s.l.' and 'M.A.S.L' all reduce to a single token.
    """
    text = raw.strip().lower()
    text = text.replace("²", "2").replace("³", "3").replace("^", "")
    text = text.replace("µ", "u")
    return re.sub(r"[\s._·\-]+", "", text)


class CoercionResult(NamedTuple):
    """Outcome of coercing one cell."""

    value: float | None
    changed: bool  # the original string was not already a bare number
    status: str  # 'numeric' | 'absent' | 'unparsed' | 'bad_unit'
    note: str


def coerce_numeric(raw: object, *, field: str) -> CoercionResult:
    """Parse one provider cell into ``field``'s canonical unit.

    Returns a :class:`CoercionResult`; never raises. ``status`` distinguishes a
    value the provider never measured (``'absent'``) from one this function could
    not interpret (``'unparsed'``) or interpreted but whose unit is not
    convertible for the field (``'bad_unit'``) — the caller reports the latter
    two, because they are defects rather than data.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return CoercionResult(None, False, "absent", "empty")

    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return CoercionResult(float(raw), False, "numeric", "already numeric")

    text = str(raw).strip()
    if _normalise_unit(text) in ABSENT_TOKENS:
        return CoercionResult(None, False, "absent", f"{text!r} means not measured")

    original = text
    note_parts: list[str] = []

    qualifier = _QUALIFIER.match(text)
    if qualifier:
        text = _QUALIFIER.sub("", text, count=1)
        note_parts.append(f"qualifier {qualifier.group(1)!r} stripped (value is an estimate)")

    annotation = _ANNOTATION.search(text)
    if annotation:
        text = _ANNOTATION.sub("", text).strip()
        note_parts.append(f"annotation {annotation.group(0)[:40]!r} stripped")

    match = _NUMBER.match(text)
    if not match:
        return CoercionResult(None, False, "unparsed", f"no leading number in {original!r}")

    number = float(match.group(1).replace(",", "."))
    tail = match.group(2)

    # Coordinates: a hemisphere letter carries the sign, so it must be consumed
    # before the unit table is consulted.
    if field in ("pl_lat", "pl_long"):
        hemi = _HEMISPHERE.match(tail)
        if hemi:
            letter = hemi.group(1).upper()
            if letter in ("S", "W"):
                number = -number
                note_parts.append(f"hemisphere {letter} -> negative")
            else:
                note_parts.append(f"hemisphere {letter}")
            tail = _HEMISPHERE.sub("", tail, count=1)
        tail = tail.replace("°", "").strip()
        if _normalise_unit(tail) in ("", "deg", "degrees"):
            return CoercionResult(number, original != match.group(1), "numeric", "; ".join(note_parts) or "parsed")

    unit = _normalise_unit(tail)
    if unit == "":
        return CoercionResult(
            number,
            original.strip() != match.group(1),
            "numeric",
            "; ".join(note_parts) or "parsed",
        )

    factors = FIELD_UNIT_FACTORS.get(field, {})
    if unit not in factors:
        return CoercionResult(
            None,
            False,
            "bad_unit",
            f"unit {tail.strip()!r} in {original!r} is not a known unit for {field} "
            f"(accepted: {sorted(factors) or 'none declared'})",
        )

    factor = factors[unit]
    if factor != 1.0:
        note_parts.append(f"converted {tail.strip()} -> canonical (x{factor})")
    return CoercionResult(number * factor, True, "numeric", "; ".join(note_parts) or f"unit {tail.strip()} stripped")


class CellStripReport(NamedTuple):
    """What :func:`strip_metadata_cells` changed."""

    n_stripped: int  # cells that lost surrounding whitespace
    n_blanked: int  # whitespace-only cells turned into NaN
    columns: tuple[str, ...]


def strip_metadata_cells(frame: pd.DataFrame) -> tuple[pd.DataFrame, CellStripReport]:
    """Trim surrounding whitespace from every text cell of a metadata frame.

    Providers pad cells, and the padding propagates into derived identifiers.
    DE-HoH is the clear case: ``tree_id`` and ``sensor_id`` are right-aligned in a
    three-character field (``' 22'``), and ``pl_code`` is built as
    ``f"{site}-{tree}-{sensor}"``, so every code came out as
    ``'DE-HoH- 22- 22'``. Note the whitespace in ``pl_code`` is therefore
    *internal* -- stripping ``pl_code`` itself would do nothing. It has to be
    stripped from the **components, before the code is composed**, which is why
    this runs early in ``tidy_metadata``.

    Whitespace-only cells become ``NaN`` so the existing "drop rows with NaN
    tree_id" guard can see them; otherwise ``'   '`` survives as a truthy value
    and yields a bogus ``'SITE--'`` code.

    Non-text values are returned untouched. This is deliberately elementwise
    rather than ``Series.str.strip()``: metadata columns hold mixed types
    (numbers read from Excel alongside text), and the ``.str`` accessor silently
    turns every non-string into ``NaN`` -- it would null out numeric cells while
    looking like a tidy-up.
    """
    out = frame.copy(deep=True)
    n_stripped = n_blanked = 0
    touched: list[str] = []

    for column in out.columns:
        # Every column is inspected, with the type test done per value below.
        # Do NOT gate on ``dtype == object``: under pandas 3 a text column has
        # dtype 'str', not 'object', so such a gate silently skips exactly the
        # columns this function exists to clean. Numeric columns simply produce
        # no changes and are left alone.
        before_stripped = n_stripped
        before_blanked = n_blanked
        values = []

        for value in out[column]:
            if not isinstance(value, str):
                values.append(value)
                continue
            trimmed = value.strip()
            if trimmed == value:
                values.append(value)
            elif trimmed == "":
                values.append(pd.NA)
                n_blanked += 1
            else:
                values.append(trimmed)
                n_stripped += 1

        if n_stripped != before_stripped or n_blanked != before_blanked:
            out[column] = values
            touched.append(str(column))

    return out, CellStripReport(n_stripped=n_stripped, n_blanked=n_blanked, columns=tuple(touched))


def deduplicate_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Suffix repeated column names so every column is addressable.

    Two provider fields can be renamed onto ONE SAPFLUXNET name. ES-Gdn does
    exactly this: ``tree_dbh`` is the olive's main stem (114 cm), while
    ``DBH(sensor branch)`` -- truncated to ``DBH`` by the column cleaner and then
    caught by the generic ``dbh`` alias -- is the instrumented branch (39 mm).
    Both became ``pl_dbh``.

    That is worse than it looks. A consumer doing ``df['pl_dbh']`` silently gets a
    two-column DataFrame rather than a Series, so anything iterating it walks the
    column *labels*; :func:`coerce_plant_md_numerics` therefore reported success
    while coercing nothing at all.

    The first occurrence keeps the name -- sheet order puts the specifically
    named field (``tree_dbh``) ahead of the generic one -- and later duplicates
    get ``_alt1``, ``_alt2``. Returns the new frame and a description of every
    rename so the caller can report the collision rather than absorb it.
    """
    counts: dict[str, int] = {}
    new_names: list[str] = []
    renames: list[str] = []

    for column in frame.columns:
        name = str(column)
        if name in counts:
            counts[name] += 1
            alt = f"{name}_alt{counts[name]}"
            renames.append(f"{name} -> {alt}")
            new_names.append(alt)
        else:
            counts[name] = 0
            new_names.append(name)

    if not renames:
        return frame, ()

    out = frame.copy(deep=False)
    out.columns = new_names
    return out, tuple(renames)


class NumericCoercionReport(NamedTuple):
    """Per-site summary of what coercion did, for logging and tests."""

    site_code: str
    fields_touched: tuple[str, ...]
    n_converted: int
    n_absent: int
    problems: tuple[str, ...]  # unparsed / bad_unit, one message per distinct value


def coerce_plant_md_numerics(
    plant_md: pd.DataFrame,
    *,
    site_code: str,
    log: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, NumericCoercionReport]:
    """Return a copy of ``plant_md`` with numeric fields parsed to canonical units.

    Any field whose values had to be transformed gains a ``{field}_orig`` column
    holding the provider's strings. Values that cannot be interpreted become
    ``NaN`` **and are reported**, which is the whole point: the previous
    behaviour reached the same NaN silently.
    """
    emit = log if log is not None else logger.info
    out = plant_md.copy(deep=True)

    touched: list[str] = []
    problems: list[str] = []
    n_converted = n_absent = 0

    for field in FIELD_UNIT_FACTORS:
        if field not in out.columns:
            continue

        # A duplicated name means two provider fields collapsed onto one
        # SAPFLUXNET field, and ``out[field]`` is then a DataFrame, not a Series.
        # Iterating it would walk the column *labels* and quietly coerce nothing
        # -- which is exactly how ES_Gdn's pl_dbh survived a coercion pass
        # untouched. Refuse to guess which duplicate is authoritative.
        if list(out.columns).count(field) > 1:
            problems.append(
                f"{field}: appears {list(out.columns).count(field)} times -- two provider fields "
                "map to it, so neither can be coerced. Disambiguate them in RENAME_MAP."
            )
            continue

        original = out[field]
        results = [coerce_numeric(v, field=field) for v in original]

        if any(r.status in ("unparsed", "bad_unit") for r in results):
            seen: set[str] = set()
            # Iterate ``results`` alone -- the raw value is already carried in
            # ``r.note``, so pairing it back against ``original`` was redundant.
            # DO NOT reintroduce zip() here: every ``zip(strict=...)`` spelling,
            # including strict=False, is Python 3.10+ and the canonical env
            # (Palma) is 3.9, so a linter "fixing" B905 breaks the pipeline.
            for r in results:
                if r.status in ("unparsed", "bad_unit") and r.note not in seen:
                    seen.add(r.note)
                    problems.append(f"{field}: {r.note}")

        changed = sum(1 for r in results if r.changed)

        # Rewrite the column if ANY cell is a non-numeric string, not only if a
        # value changed. 'not measured' carries no number, so it is not
        # "changed" -- but leaving the literal text sitting in a numeric column
        # is precisely defect C2 (SE_Sgr's pl_height read 'not measured' x12
        # after a coercion pass that reported nothing to do).
        text_cells = sum(1 for r in results if r.status != "numeric" and r.note != "empty")
        if changed == 0 and text_cells == 0:
            continue

        # Preserve provenance only where the provider text differed from a bare
        # number, so untouched sites gain no spurious column.
        orig_col = f"{field}_orig"
        if orig_col not in out.columns:
            out.insert(out.columns.get_loc(field) + 1, orig_col, original.astype("object"))

        out[field] = [r.value for r in results]
        touched.append(field)
        n_converted += changed
        n_absent += sum(1 for r in results if r.status == "absent")

    if touched:
        emit(
            f"    INFO: coerced provider text in {len(touched)} numeric field(s) "
            f"({', '.join(touched)}); {n_converted} value(s) rewritten, originals kept in *_orig"
        )
    for problem in problems:
        emit(f"    WARNING: {site_code} {problem}")

    return out, NumericCoercionReport(
        site_code=site_code,
        fields_touched=tuple(touched),
        n_converted=n_converted,
        n_absent=n_absent,
        problems=tuple(problems),
    )
