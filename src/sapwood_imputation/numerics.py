"""Unit-aware numeric coercion for SAPFLUXNET plant metadata columns.

``pl_dbh`` and ``pl_sapw_area`` reach us as provider free text: ``'16.5 cm'``,
``'165 mm'``, ``'207.2 cm2'``, ``'114(main stem_see notes)'``, ``'n/a'``.

Two things go wrong if such a cell is used raw:

* comparing it numerically (``'16.5 cm' > 0``) raises ``TypeError`` and aborts
  the run;
* coercing it with a plain ``pd.to_numeric(errors='coerce')`` turns a real
  measurement into ``NaN``, silently deleting the tree.

``Sapflow-internal/metadata_numerics.py`` already solves this, including
conversion into each field's canonical SAPFLUXNET unit (``pl_dbh`` -> cm,
``pl_sapw_area`` -> cm2). That directory name contains a hyphen, so it is not
importable as a package; this module loads it by path and wraps it at the
Series level.

If the source module cannot be loaded, coercion degrades to
``pd.to_numeric(errors='coerce')`` and warns loudly -- values with units are
then lost rather than recovered, but the run never crashes and the loss is
counted rather than silent.
"""

from __future__ import annotations

import importlib.util
import warnings
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import pandas as pd

#: This file lives at <repo_root>/src/sapwood_imputation/numerics.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
_METADATA_NUMERICS = _REPO_ROOT / "Sapflow-internal" / "metadata_numerics.py"

#: Private module name for the by-path import; avoids colliding with any real
#: top-level module called ``metadata_numerics``.
_LOADED_AS = "_sapflow_internal_metadata_numerics"

#: Statuses that mean the cell is a *defect* rather than an honest absence.
#: ``'absent'`` is data (the provider never measured it); these two are bugs in
#: the delivery and must be reported, not swallowed.
DEFECT_STATUSES = ("unparsed", "bad_unit")

#: Status recorded when the unit-aware coercer was unavailable.
FALLBACK_STATUS = "fallback_to_numeric"


@lru_cache(maxsize=1)
def load_metadata_numerics() -> ModuleType | None:
    """Load ``Sapflow-internal/metadata_numerics.py`` by path.

    Returns ``None`` (with a warning) if it cannot be loaded, so that callers
    can fall back instead of failing hard. Cached: the file is read once.
    """
    if not _METADATA_NUMERICS.is_file():
        warnings.warn(
            f"metadata_numerics.py not found at {_METADATA_NUMERICS}; "
            "falling back to pd.to_numeric -- values carrying units "
            "(e.g. '16.5 cm') will be dropped instead of converted.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    spec = importlib.util.spec_from_file_location(_LOADED_AS, _METADATA_NUMERICS)
    if spec is None or spec.loader is None:
        warnings.warn(
            f"Could not build an import spec for {_METADATA_NUMERICS}; falling back to pd.to_numeric.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - optional dependency, must not abort the run
        warnings.warn(
            f"Failed to execute {_METADATA_NUMERICS} ({exc!r}); falling back to pd.to_numeric.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    if not hasattr(module, "coerce_numeric"):
        warnings.warn(
            f"{_METADATA_NUMERICS} defines no coerce_numeric(); falling back to pd.to_numeric.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return module


def coerce_series(values: pd.Series, field: str) -> tuple[pd.Series, Counter]:
    """Coerce one metadata column to ``field``'s canonical unit as ``float64``.

    Parameters
    ----------
    values
        Raw column straight off ``pd.read_csv`` (may hold strings).
    field
        SAPFLUXNET field name, e.g. ``'pl_dbh'`` or ``'pl_sapw_area'``. Selects
        the unit table used for conversion.

    Returns
    -------
    (pd.Series, Counter)
        Float series aligned to ``values.index`` (``NaN`` where unusable), and
        a count of coercion statuses so callers can report defects.
    """
    module = load_metadata_numerics()
    if module is None:
        coerced = pd.to_numeric(values, errors="coerce").astype(float)
        return coerced, Counter({FALLBACK_STATUS: int(len(values))})

    numbers: list[float] = []
    counts: Counter = Counter()
    for raw in values:
        result = module.coerce_numeric(raw, field=field)
        counts[result.status] += 1
        numbers.append(float("nan") if result.value is None else float(result.value))
    return pd.Series(numbers, index=values.index, dtype=float), counts


def defect_count(counts: Counter) -> int:
    """Number of cells that were malformed rather than merely absent."""
    return sum(counts.get(status, 0) for status in DEFECT_STATUSES)
