"""Column names and source flags shared across the package.

Split out so the reporting layer can name a column without importing the
orchestrator that prints through it -- these constants are vocabulary, not
behaviour, and both sides need them.

:mod:`src.sapwood_imputation.impute_sapwood_area` re-exports every name here, so
``from ...impute_sapwood_area import ASW_COL`` keeps working.
"""

from __future__ import annotations

# --- SAPFLUXNET plant metadata, as delivered -------------------------------
SPECIES_COL = "pl_species"
DBH_COL = "pl_dbh"
ASW_COL = "pl_sapw_area"

# --- columns this package adds ---------------------------------------------
FT_COL = "pl_functional_type"
FILLED_COL = "pl_sapw_area_filled"
SOURCE_COL = "pl_sapw_area_source"

# --- values of SOURCE_COL ---------------------------------------------------
SOURCE_MEASURED = "measured"
SOURCE_IMPUTED = "imputed"
SOURCE_UNRESOLVED = "unresolved"

#: A measured ASW that its own DBH contradicts. One of the two carries the wrong
#: unit and nothing here can say which, so the tree is neither passed through nor
#: imputed -- imputing would trust the very DBH that failed the check.
SOURCE_IMPLAUSIBLE = "implausible"

#: A provider-computed ASW: kept in FILLED_COL (it is still the best number for
#: that tree, and what their published flux density used) but barred from the fit
#: and labelled, so a consumer can tell it from a measurement.
SOURCE_PROVIDER_ESTIMATE = "provider_estimate"

SOURCE_FLAGS = (
    SOURCE_MEASURED,
    SOURCE_PROVIDER_ESTIMATE,
    SOURCE_IMPUTED,
    SOURCE_IMPLAUSIBLE,
    SOURCE_UNRESOLVED,
)

# --- internal bookkeeping attached to the pooled fit frame ------------------
#: Which corpus a pooled row came from, so the log can report each separately.
ORIGIN_COL = "_origin"
#: Site code of a pooled row: the key provenance groups on, and the unit
#: leave-one-site-out holds out.
SITE_COL = "_site"
