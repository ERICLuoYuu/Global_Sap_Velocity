"""Configuration for QC masking.

The threshold is the one genuinely uncertain number here.  The FLUXNET wording
for QC levels 0/1/2 could not be confirmed from documentation (fluxnet.org
returns 403, the Pastorello 2020 paper sits behind an auth redirect, and the
ONEFlux README does not define the levels).  What the data shows, over
1,184,579 flagged rows:

- level 0 is 72-77% of rows, with full-resolution signal;
- level 1 is 0.1-1.3%, value present but 10-20x smoother than level 0 at the
  same site -- the signature of an interpolated fill, not a measurement;
- level 2 is 1.4-7.4%, and at 7 of 13 sites the variable is entirely NaN there.

Monotone in quality, 0 best.  That is all the algorithm needs, so the default
follows the evidence and the exact wording stays recorded as unconfirmed rather
than guessed.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType

from src.qc_flags.vocabulary import ACTIONABLE_TOKENS

#: Variables **this pipeline** computes rather than measures, and their sources.
#:
#: Applies to the QC_COLUMNS mechanism only. At the 202 SAPFLUXNET sites both
#: variables arrive from the archive as independent measurements with their own
#: flag columns -- 19 of 176 carry `ppfd_in` on rows where `sw_in` is absent,
#: which is proof they were not computed from it. See masking.build_mask.
#:
#: `icos_data_extractor` synthesises both: `fill_ppfd_in(sw_in)` and
#: `fill_rh(ta, vpd)`. Verified at all 13 checked ICOS sites that `ppfd_in` is
#: non-null on exactly the rows where `sw_in` is, and `rh` on exactly the rows
#: where `vpd` is. So masking a source while keeping its derivative would retain
#: a number computed from data we just rejected. No site has an independent PAR
#: or RH measurement, so propagation cannot destroy a real observation.
DERIVED_FROM: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "ppfd_in": ("sw_in",),
        "rh": ("ta", "vpd"),
    }
)


@dataclasses.dataclass(frozen=True)
class QcFlagConfig:
    """How to turn flags into a mask.

    Attributes:
        qc_threshold: mask in-file QC values at or above this level. Default 1
            keeps only level 0 (decision D1).
        actionable_tokens: SAPFLUXNET tokens that mask a retained value.
        derived_from: derived variable -> the variables it is computed from.
    """

    qc_threshold: int = 1
    actionable_tokens: frozenset = ACTIONABLE_TOKENS
    derived_from: Mapping[str, tuple[str, ...]] = DERIVED_FROM

    def __post_init__(self) -> None:
        if self.qc_threshold < 1:
            # qc >= 0 masks every row including the level the data calls good,
            # which is almost certainly a typo for "keep only level 0". Refuse
            # it rather than silently blanking six variables at 19 sites.
            raise ValueError(
                f"qc_threshold must be >= 1 (got {self.qc_threshold}); a threshold of 0 would mask every value including level 0"
            )
