"""Telling variables, QC columns and timestamps apart.

Two defects live here.

**D-3.** Neither analyzer excludes `*_qc` columns from the env variable set, so
`ta_qc` and friends are z-scored, outlier-filtered and daily-resampled as if
they were weather.  Stage 4 then reads `{site}_env_data_outliers_removed.csv`
and joins it wholesale (`pd.merge(..., left_index=True, right_index=True)`,
dropping only `solar_TIMESTAMP`), so they would arrive in `merged_data.csv` as
feature-shaped columns.  They have not, only because D-1 kills every ICOS site
before Stage 3 emits anything -- the two defects mask each other.

**D-5.** Flag columns were matched with ``[f for f in flags.columns if col in
f][0]``: substring, first hit.  Correct only while no column name is a prefix of
another that sorts earlier.  `SWE_NOR_sapf_data.csv` lists `..._Jt_10`, `_Jt_11`
and `_Jt_12` *before* `_Jt_1`, so that expression resolves `_Jt_1` to a
different tree.  Nothing goes wrong today only because that site ships no flag
file.  :func:`resolve_flag_column` matches by equality, which removes the
question instead of re-measuring it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Union

import pandas as pd

QC_SUFFIX = "_qc"

#: Columns that are neither variables nor QC: time axes and coordinates.
NON_DATA_COLUMNS = frozenset(
    {
        "TIMESTAMP",
        "solar_TIMESTAMP",
        "TIMESTAMP_solar",
        "TIMESTAMP_LOCAL",
        "lat",
        "long",
    }
)

# `Union`, not `|`: this is a module-level assignment, so it is evaluated at
# import time and `from __future__ import annotations` does not reach it. Palma
# runs Python 3.9, where `pd.DataFrame | pd.Index` raises TypeError.
ColumnSource = Union[pd.DataFrame, pd.Index, Sequence[str]]


def _names(columns: ColumnSource) -> list[str]:
    """Accept a frame, an Index or a plain sequence of names."""
    if isinstance(columns, pd.DataFrame):
        return list(columns.columns)
    return list(columns)


def split_qc_columns(columns: ColumnSource) -> tuple[list[str], list[str]]:
    """Split into (variables, qc columns), dropping timestamps and coordinates.

    Order is preserved, because the variable order decides the column order of
    every downstream output and criterion 4 compares those byte for byte.
    """
    variables = []
    qc_columns = []
    for name in _names(columns):
        if name in NON_DATA_COLUMNS:
            continue
        if name.endswith(QC_SUFFIX):
            qc_columns.append(name)
        else:
            variables.append(name)
    return variables, qc_columns


def qc_column_for(variable: str) -> str:
    """The name of the QC column belonging to ``variable``."""
    return f"{variable}{QC_SUFFIX}"


def variable_for(qc_column: str) -> str:
    """The variable a QC column belongs to."""
    if not qc_column.endswith(QC_SUFFIX):
        raise ValueError(f"{qc_column!r} is not a QC column (expected a {QC_SUFFIX} suffix)")
    return qc_column[: -len(QC_SUFFIX)]


def resolve_flag_column(variable: str, flag_columns: Iterable[str]) -> str | None:
    """The flag column for ``variable``, matched by equality.

    Returns None when the flag frame has no column of that name -- an absent
    flag is not an error, it just means nothing to mask.
    """
    names = _names(flag_columns) if isinstance(flag_columns, (pd.DataFrame, pd.Index)) else list(flag_columns)
    return variable if variable in names else None
