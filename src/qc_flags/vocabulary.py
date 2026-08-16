"""The SAPFLUXNET flag vocabulary, and which tokens act on a retained value.

A token census over 12 sites (both `sapf_flags` and `env_flags`) found seven
tokens, `;`-joined into composites.  Only two of them sit on a value that is
still present:

- ``RANGE_WARN`` -- the value is physically implausible.  Nothing else in this
  pipeline performs a range check, so the token supplies information no other
  step does.  It is the one token we act on.
- ``OUT_WARN`` -- the value is a statistical outlier under SAPFLUXNET's own
  detector.  This pipeline already runs its own (rolling z-score, 3 sigma by day
  and 5 sigma by night over a 30-day window).  Acting on ``OUT_WARN`` would
  stack the archive's definition of "outlier" on top of the one this project
  chose and tuned, with the archive's silently winning.

``OUT_WARN`` is therefore never applied -- not "not yet", and not under any
configuration (decision D2, 2026-08-16).  It is named in ``NEVER_ACTIONABLE``
rather than merely left out, so the ruling is visible at the point where
someone would otherwise add it back.

The remaining five tokens describe cells that are already NA or already
removed, so acting on them would be a no-op.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

SEPARATOR = ";"

#: Tokens that mask a value which is still present.
ACTIONABLE_TOKENS = frozenset({"RANGE_WARN"})

#: Tokens ruled out permanently. See the module docstring for why.
NEVER_ACTIONABLE = frozenset({"OUT_WARN"})

#: Tokens describing cells that are already NA or already removed.
NO_OP_TOKENS = frozenset(
    {
        "NA_PRESENT",
        "NA_ADDED",
        "CALCULATED",
        "MANUAL_REMOVED",
        "RANGE_REMOVED",
        "OUT_REMOVED",
    }
)

#: Every token seen in the corpus. A token outside this set is unrecognised and
#: is never acted on -- silence is the safe direction for something unknown.
KNOWN_TOKENS = ACTIONABLE_TOKENS | NEVER_ACTIONABLE | NO_OP_TOKENS


def split_tokens(value: Any) -> tuple[str, ...]:
    """Split a flag cell into its component tokens.

    Returns an empty tuple for anything missing, so callers never have to guard
    NaN separately.
    """
    if value is None:
        return ()
    try:
        if pd.isna(value):
            return ()
    except (TypeError, ValueError):
        # Array-likes land here; they are not valid flag cells.
        return ()
    text = str(value).strip()
    if not text:
        return ()
    return tuple(part.strip() for part in text.split(SEPARATOR) if part.strip())


def is_actionable(value: Any, tokens: frozenset[str] | None = None) -> bool:
    """Does this flag cell mask its value?

    Membership, never containment: ``RANGE_REMOVED`` must not read as
    ``RANGE_WARN`` just because one is a prefix-sibling of the other.
    """
    if tokens is None:
        tokens = ACTIONABLE_TOKENS
    return bool(set(split_tokens(value)) & tokens)


def _pattern(tokens: frozenset[str]) -> str:
    """A regex matching any whole token in ``tokens``.

    Anchored on the separator at both ends so a token can only match as a whole
    element of the composite, which is the vectorised form of the membership
    test in :func:`is_actionable`.
    """
    alternation = "|".join(re.escape(token) for token in sorted(tokens))
    return rf"(?:^|;)\s*(?:{alternation})\s*(?:;|$)"


def actionable_mask(series: pd.Series, tokens: frozenset[str] | None = None) -> pd.Series:
    """Vectorised :func:`is_actionable` over a flag column.

    Flag frames run to millions of rows, so this is a regex over the string
    series rather than a per-cell split.
    """
    if tokens is None:
        tokens = ACTIONABLE_TOKENS
    if not tokens:
        return pd.Series(False, index=series.index)
    # astype(str) renders NaN as "nan", which matches no token.
    return series.astype(str).str.contains(_pattern(tokens), regex=True, na=False)
