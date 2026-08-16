"""Which flag mechanism a site has, decided from what is on disk.

The corpus carries three, measured over all 226 sites of `csv/sapwood_combined`:

===========================  ====  ==============================
population                      n  mechanism
===========================  ====  ==============================
SAPFLUXNET external           202  ``*_flags.csv`` companion file
internal, ICOS-derived env     19  in-file ``*_qc`` columns
internal, non-ICOS             4   none
internal, empty env            1   none (``SWE_NOR``, skipped)
===========================  ====  ==============================

Detection reads the directory and the CSV header, deliberately *not* whatever
the analyzer decided to do.  Acceptance criterion 1 compares the run report
against an independent census, and a census derived from the code under test
would only prove the code agrees with itself.
"""

from __future__ import annotations

import enum
from pathlib import Path

import pandas as pd

from src.qc_flags.columns import split_qc_columns


class Mechanism(enum.Enum):
    """How a site's quality information is expressed."""

    SAPFLUXNET_FLAGS = "sapfluxnet_flags"
    QC_COLUMNS = "qc_columns"
    NONE = "none"


def detect_mechanism(site: str, kind: str, corpus_dir: str | Path) -> Mechanism:
    """Classify one site.

    Args:
        site: site code, e.g. ``CHE_DAV``.
        kind: ``"sapf"`` or ``"env"``.
        corpus_dir: directory holding the flat corpus.

    Raises:
        FileNotFoundError: if the site's data file is absent. A site that was
            asked for but is not there is a caller bug, not a site without a
            mechanism, and conflating the two is how D-1 stayed invisible.
    """
    corpus_dir = Path(corpus_dir)
    data_path = corpus_dir / f"{site}_{kind}_data.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"no {kind} data file for site {site}: {data_path}")

    if (corpus_dir / f"{site}_{kind}_flags.csv").exists():
        # The archive's per-cell tokens are richer than a per-row integer, so
        # they take precedence. No site carries both today; the ordering is
        # defined here rather than discovered later.
        return Mechanism.SAPFLUXNET_FLAGS

    _, qc_columns = split_qc_columns(pd.read_csv(data_path, nrows=0).columns)
    return Mechanism.QC_COLUMNS if qc_columns else Mechanism.NONE


#: `SWE_NOR`'s env file is a fully-built scaffold -- 16,973 rows on a timestamp
#: grid identical to its sapf_data, and zero variables -- because its ICOS
#: extraction failed silently. Recorded verbatim so the run report distinguishes
#: it from a site that legitimately has no environmental data.
NO_VARIABLES_REASON = (
    "no environmental variables (ICOS extraction produced a variable-less scaffold -- see deviations 2026-08-16)"
)


def skip_reason(frame: pd.DataFrame) -> str | None:
    """Why this frame cannot be processed, or None if it can.

    A frame carrying QC columns but no variables has nothing to clean, so QC
    columns alone do not count.
    """
    variables, _ = split_qc_columns(frame)
    if not variables:
        return NO_VARIABLES_REASON
    return None
