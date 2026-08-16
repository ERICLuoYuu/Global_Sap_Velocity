"""Stage-3 quality control over a corpus with three different flag mechanisms.

The 226-site combined corpus does not express quality one way.  202 SAPFLUXNET
sites ship companion `*_flags.csv` files carrying `;`-joined tokens; 19 internal
ICOS-derived sites carry in-file `*_qc` integer columns; 4 carry neither; and
one (`SWE_NOR`) has no environmental variables at all.

Both analyzers previously handled only the first mechanism, inline and
duplicated, and died on the rest -- `env_analyzer` printed `flags.dtypes`
*above* the `if flags is not None` guard, so every flagless site raised
`AttributeError` into a bare `except` and vanished as one line of stdout.  Three
mechanisms across two analyzers is six code paths written inline and one written
here.  The derived-variable propagation rule in particular must have exactly one
home; it is the kind of rule that rots the moment it is copied.

Deciding what to mask and applying the mask are deliberately separate, so the
decision can be counted and cross-checked before anything is destroyed.
"""

from src.qc_flags.columns import qc_column_for, resolve_flag_column, split_qc_columns, variable_for
from src.qc_flags.config import DERIVED_FROM, QcFlagConfig
from src.qc_flags.masking import apply_mask, build_mask, count_masked, propagate_to_derived
from src.qc_flags.mechanisms import Mechanism, detect_mechanism, skip_reason
from src.qc_flags.report import RunReport, SiteFlagOutcome
from src.qc_flags.vocabulary import (
    ACTIONABLE_TOKENS,
    KNOWN_TOKENS,
    NEVER_ACTIONABLE,
    NO_OP_TOKENS,
    actionable_mask,
    is_actionable,
)

__all__ = [
    "ACTIONABLE_TOKENS",
    "DERIVED_FROM",
    "KNOWN_TOKENS",
    "NEVER_ACTIONABLE",
    "NO_OP_TOKENS",
    "Mechanism",
    "QcFlagConfig",
    "RunReport",
    "SiteFlagOutcome",
    "actionable_mask",
    "apply_mask",
    "build_mask",
    "count_masked",
    "detect_mechanism",
    "is_actionable",
    "propagate_to_derived",
    "qc_column_for",
    "resolve_flag_column",
    "skip_reason",
    "split_qc_columns",
    "variable_for",
]
