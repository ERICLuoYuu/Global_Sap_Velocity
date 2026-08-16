"""cm3 h-1 -> cm3 cm-2 h-1: sap flow divided by the stem area carrying it.

    sap flux density = sap flow / sapwood area
    cm3 cm-2 h-1     = cm3 h-1  / cm2

Pure functions: frames in, frames out, nothing touched on disk and no input
mutated.

The single rule this module enforces is that **a column we cannot divide is
dropped, never passed through**. An undivided column is not a smaller error than
a missing one, it is a much larger one: it leaves plant-level magnitudes under a
sapwood-level header, roughly 100x too big, with nothing downstream able to tell.
``Sapflow-internal/plant_to_sapwood_converter.py`` shipped exactly that once,
when a skip-guard testing ``'sapflow' in name`` stopped matching after the files
were renamed to ``_sapf_data.csv``; it copied the plant file over the sapwood
output it had just written, and SE_Nor's median read 122 instead of 0.30.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.sapwood_imputation.columns import FILLED_COL, SOURCE_COL

#: What ``pl_sap_units`` must read once a frame is at sapwood scale.
SAPWOOD_UNITS = "cm3 cm-2 h-1"
PLANT_UNITS = "cm3 h-1"

UNITS_COL = "pl_sap_units"
CODE_COL = "pl_code"

#: Column this package adds to plant_md, recording how each tree got here.
CONVERSION_COL = "pl_sapwood_scale_origin"
NATIVE = "native"
CONVERTED = "converted"
DROPPED_NO_AREA = "dropped: no usable sapwood area"

#: Substring identifying the timestamp columns, which are identity rather than
#: data: they are carried through as strings and never parsed, because a
#: round-trip through pandas would re-render the offset and lose 'Z'.
_TIMESTAMP_TOKEN = "TIMESTAMP"


def timestamp_columns(columns: Iterable[str]) -> tuple[str, ...]:
    """The columns of a sapf_data frame that are timestamps, in order."""
    return tuple(c for c in columns if _TIMESTAMP_TOKEN in str(c).upper())


def data_columns(columns: Iterable[str]) -> tuple[str, ...]:
    """The tree columns of a sapf_data frame, in order."""
    return tuple(c for c in columns if _TIMESTAMP_TOKEN not in str(c).upper())


def _usable(area: object) -> bool:
    """True when ``area`` can divide: finite and strictly positive."""
    try:
        value = float(area)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value)) and value > 0.0


def areas_from_metadata(metadata: pd.DataFrame) -> dict[str, float]:
    """``pl_code -> sapwood area`` for every tree with a usable one.

    Trees whose area is missing, zero, negative or infinite are simply absent
    from the mapping; :func:`convert_sapflow` turns that absence into a drop.
    """
    if CODE_COL not in metadata.columns:
        raise ValueError(f"plant metadata has no {CODE_COL!r} column")

    codes = metadata[CODE_COL].astype(str)
    duplicated = sorted(set(codes[codes.duplicated()]))
    if duplicated:
        raise ValueError(
            f"plant metadata holds duplicate {CODE_COL} values, so no single sapwood area "
            f"can be attributed to those trees: {duplicated}"
        )
    if FILLED_COL not in metadata.columns:
        raise ValueError(
            f"plant metadata has no {FILLED_COL!r} column; it must come from a "
            "*_asw_filled/ directory, not the raw archive"
        )

    # itertuples rather than zip(): this runs on Palma's Python 3.9, where
    # zip(strict=) is a TypeError. Iterating one frame also makes the pairing
    # structural, so the length equality strict= would assert cannot be violated
    # in the first place.
    areas = {}
    for code, area in metadata[[CODE_COL, FILLED_COL]].itertuples(index=False, name=None):
        if _usable(area):
            areas[str(code)] = float(area)
    return areas


@dataclass(frozen=True)
class ConversionResult:
    """A converted sapf_data frame and the fate of every tree in it."""

    frame: pd.DataFrame
    converted: tuple[str, ...]
    dropped: tuple[tuple[str, str], ...]


def convert_sapflow(sapflow: pd.DataFrame, metadata: pd.DataFrame) -> ConversionResult:
    """Divide each tree's series by its own sapwood area.

    Every data column must have a metadata row: a column with no row is a join
    failure, not a missing measurement, and raises. A row with no column is
    normal -- metadata routinely describes trees the flux file does not carry.
    """
    areas = areas_from_metadata(metadata)
    known = set(metadata[CODE_COL].astype(str))

    stamps = timestamp_columns(sapflow.columns)
    trees = data_columns(sapflow.columns)

    unknown = [c for c in trees if c not in known]
    if unknown:
        raise ValueError(
            f"sapf_data carries {len(unknown)} column(s) with no row in plant_md, so their sapwood area is unknown: {unknown}"
        )

    out = {c: sapflow[c] for c in stamps}
    converted, dropped = [], []
    for code in trees:
        area = areas.get(code)
        if area is None:
            dropped.append((code, DROPPED_NO_AREA))
            continue
        out[code] = pd.to_numeric(sapflow[code], errors="coerce") / area
        converted.append(code)

    return ConversionResult(
        frame=pd.DataFrame(out, index=sapflow.index),
        converted=tuple(converted),
        dropped=tuple(dropped),
    )


def check_flags_columns(flag_columns: Iterable[str], data_columns_: Iterable[str]) -> None:
    """Refuse a ``sapf_flags`` whose trees are not exactly ``sapf_data``'s.

    Measured on the corpus as delivered: the two files match exactly, order
    included, for all 359 external files. So any difference is a change in the
    source rather than a case to absorb -- and absorbing it would be the same
    class of error this module exists to prevent, moved from the data into the
    QC record. A flags column with no data column would survive into the product
    describing a tree that is not there; a data column with no flags column would
    be QC'd as if unflagged.

    Takes column lists rather than frames so the caller can validate from a
    header alone, without reading gigabytes it may only need to copy.
    """
    flags_trees = set(data_columns(flag_columns))
    data_trees = set(data_columns(data_columns_))
    if flags_trees == data_trees:
        return
    raise ValueError(
        "sapf_flags and sapf_data disagree about which trees this site has: "
        f"{sorted(data_trees - flags_trees)} have data but no flags, "
        f"{sorted(flags_trees - data_trees)} have flags but no data"
    )


def align_flags(flags: pd.DataFrame, converted: Sequence[str]) -> pd.DataFrame:
    """Cut ``sapf_flags`` down to the trees that survived the conversion.

    Flags are the QC record -- ``NA_ADDED``, ``OUT_WARN``, ``NA_PRESENT`` -- and
    are **categorical**: they are filtered, never divided, never coerced to
    numeric. What must hold is that ``sapf_flags`` and ``sapf_data`` agree on
    which trees the site has, because :mod:`src.Analyzers.sap_analyzer` pairs
    them positionally by column name and a tree present in one but not the other
    has no defined meaning.

    Raises when a converted tree has no flags column. On the corpus as delivered
    the two files match exactly -- order included -- for all 359 external files,
    so a mismatch is a change in the source rather than a case to absorb, and
    quietly intersecting would hide it.
    """
    stamps = timestamp_columns(flags.columns)
    available = set(data_columns(flags.columns))

    missing = [c for c in converted if c not in available]
    if missing:
        raise ValueError(
            f"sapf_flags is missing {len(missing)} column(s) that sapf_data carries, so the two "
            f"files disagree about which trees this site has: {missing}"
        )

    keep = list(stamps) + [c for c in data_columns(flags.columns) if c in set(converted)]
    return flags.loc[:, keep].copy()


def convert_metadata(
    metadata: pd.DataFrame,
    converted: Sequence[str],
    dropped: Sequence[tuple[str, str]],
    origin: str = CONVERTED,
) -> pd.DataFrame:
    """Restate metadata at sapwood scale, recording each tree's fate.

    ``pl_sap_units`` becomes the sapwood unit because the series now is one.
    ``pl_sap_units_orig`` is deliberately untouched: it records what the provider
    delivered, which this package has no business rewriting.
    """
    out = metadata.copy()
    out[UNITS_COL] = SAPWOOD_UNITS

    fate: Mapping[str, str] = dict(dropped)
    codes = out[CODE_COL].astype(str)
    kept = set(converted)
    out[CONVERSION_COL] = [origin if code in kept else fate.get(code, DROPPED_NO_AREA) for code in codes]
    return out


def source_flag_counts(metadata: pd.DataFrame) -> dict[str, int]:
    """How each tree's sapwood area was arrived at, for the manifest."""
    if SOURCE_COL not in metadata.columns:
        return {}
    counts = metadata[SOURCE_COL].astype(str).value_counts()
    return {str(k): int(v) for k, v in counts.items()}
