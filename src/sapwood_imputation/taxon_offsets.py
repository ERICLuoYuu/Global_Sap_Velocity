"""Refine the functional-type offset with taxon-specific corrections.

Functional type is too coarse. Measured on the real pool, *Larix* carries 3.96x
less sapwood than the gymnosperm offset gives it and *Abies* 1.69x more -- a
6.7x spread of systematic bias inside a single group, and it does not average
out: every *Larix* in the delivered file is wrong the same way.

Simply adding genus as a predictor is the wrong fix, and measurably so. Most
genera in this corpus sit at few sites, so a raw genus mean is partly that
site's mean wearing a genus label. Leave-one-site-out shows what that costs:
hard genus indicators improve the corpus overall (R2 0.755 -> 0.805) while
making *Populus* **83% worse**, because that genus here holds a desert riparian
(*P. euphratica*) and a boreal aspen (*P. tremuloides*), with one site supplying
104 of its 135 trees. A user drawing a Populus does not get to know that.

Three devices make the correction generalise instead:

**site-balanced means**
    A taxon's offset is the mean of its *site* means, not of its trees, so one
    large stand cannot define a global taxon.

**clustered standard error**
    Trees cluster in sites, so the evidence for a taxon is its number of SITES,
    not of trees. The offset's variance is the between-site variance of those
    means divided by the number of sites.

**gate, then shrink**
    A taxon whose offset cannot be told from site noise (``|z| < Z_MIN``) keeps
    no offset at all; one that survives is still shrunk by
    ``tau2 / (tau2 + se2)``, so a marginal taxon moves a little and a
    well-evidenced one moves nearly all the way.

Applied finest-level-first (species, then genus), falling back a rung whenever
the finer level has not earned an offset, and finally to the functional type
itself -- which is exactly today's model. **Coverage therefore never drops**: on
the real corpus 50% of imputable trees receive a taxon correction and the other
50% are predicted byte-identically to the unrefined fit.

Leave-one-site-out, 162 sites, versus functional type alone:

===========================  ======  ============  ==============
model                            R2  median error  worst genus
===========================  ======  ============  ==============
functional type only          0.755         34.9%  --
genus, hard indicators        0.805         28.2%  -83% (Populus)
species -> genus, gated       0.813         26.5%  -4.7%
===========================  ======  ============  ==============
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.sapwood_imputation.functional_type import extract_genus

#: |z| below which a taxon's offset cannot be separated from site-to-site noise
#: and is not applied at all. Tuned by leave-one-site-out: below 2.5 the
#: site-driven genera (Populus, Eucalyptus, Acacia) start being given offsets
#: that make them worse, and Populus reaches -83%.
Z_MIN = 2.5

#: Prior variance of true taxon effects, in log units. Sets how hard a surviving
#: offset is still shrunk. 0.02 was the value that maximised out-of-site skill
#: while leaving no genus more than ~5% worse than the unrefined model.
TAU2 = 0.02

#: Finest to coarsest. A taxon must appear at >= 2 sites at a level to be
#: judged there at all -- with one site there is no between-site variance to
#: measure, and the "taxon effect" would be that site's effect exactly.
LEVELS = ("species", "genus")
MIN_SITES = 2

#: Prior degrees of freedom for the moderated between-site variance.
#:
#: With two or three sites the sample variance is wildly unstable, and when two
#: site means land close together by luck it collapses toward zero and ``z``
#: explodes -- manufacturing certainty out of almost no evidence. Unmoderated,
#: this corpus produced *Goupia* at z=244 on **2 trees at 2 sites**, and
#: *Juniperus* at z=113. Each taxon's variance is therefore pooled toward the
#: corpus-wide typical between-site variance, weighted ``d0`` observations
#: against its own ``n_sites - 1``. A taxon seen at many sites is barely
#: affected; one seen at two is judged mostly on the prior.
VARIANCE_PRIOR_DF = 2.0

#: Fallback prior variance when the corpus is too small to estimate one, in
#: squared log units. Roughly the between-site spread seen in this archive.
DEFAULT_PRIOR_VARIANCE = 0.09


@dataclass(frozen=True)
class TaxonOffset:
    """One earned correction, with the evidence that earned it."""

    level: str
    key: str
    offset: float  # shrunken, log scale
    raw_offset: float  # before shrinkage
    z: float
    n_trees: int
    n_sites: int


@dataclass(frozen=True)
class TaxonOffsets:
    """Every taxon correction for one fit, applied finest level first."""

    entries: tuple[TaxonOffset, ...]
    z_min: float
    tau2: float
    levels: tuple[str, ...]

    def _table(self, level: str) -> dict:
        return {e.key: e for e in self.entries if e.level == level}

    def lookup(self, species: str | None) -> tuple[float, str]:
        """Return (offset, level_used) for a species string.

        ``('', 'functional_type')`` means no taxon offset was earned and the
        prediction is exactly the unrefined model's.
        """
        keys = {"species": normalise_species(species), "genus": normalise_genus(species)}
        for level in self.levels:
            key = keys.get(level)
            if not key:
                continue
            entry = self._table(level).get(key)
            if entry is not None:
                return entry.offset, level
        return 0.0, "functional_type"

    def to_report(self) -> dict:
        """Serialise for the JSON artifact, coarsest evidence first."""
        return {
            "z_min": self.z_min,
            "tau2": self.tau2,
            "levels": list(self.levels),
            "n_offsets": len(self.entries),
            "by_level": {level: sum(1 for e in self.entries if e.level == level) for level in self.levels},
            "offsets": [
                {
                    "level": e.level,
                    "taxon": e.key,
                    "offset_log": e.offset,
                    "raw_offset_log": e.raw_offset,
                    # >1 = the unrefined model gave this taxon too much sapwood
                    "corrects_over_prediction_by": float(math.exp(-e.offset)),
                    "z": e.z,
                    "n_trees": e.n_trees,
                    "n_sites": e.n_sites,
                }
                for e in sorted(self.entries, key=lambda e: e.offset)
            ],
        }


def normalise_species(species: str | None) -> str:
    """Lowercased binomial, or '' when unusable. Case/whitespace insensitive."""
    if not isinstance(species, str):
        return ""
    return " ".join(species.strip().lower().split())


def normalise_genus(species: str | None) -> str:
    """Capitalised genus, or '' -- family-level names resolve to nothing."""
    return extract_genus(species) or ""


def fit_taxon_offsets(
    frame: pd.DataFrame,
    fit,
    group_col: str,
    dbh_col: str,
    asw_col: str,
    site_col: str,
    species_col: str,
    z_min: float = Z_MIN,
    tau2: float = TAU2,
    levels: tuple[str, ...] = LEVELS,
    min_sites: int = MIN_SITES,
) -> TaxonOffsets:
    """Earn a taxon correction wherever the evidence survives site clustering.

    Operates on the residuals of ``fit``, so it is a strictly additive refinement:
    the slope, the functional-type offsets and the correction factor are all
    untouched, and a taxon that earns nothing is predicted exactly as before.
    """
    if not z_min >= 0 or not tau2 > 0:
        raise ValueError(f"z_min must be >= 0 and tau2 > 0; got {z_min!r}, {tau2!r}")

    usable = frame[(frame[dbh_col] > 0) & (frame[asw_col] > 0)].copy()
    labels = usable[group_col].astype(str)
    offsets = dict(fit.offsets)
    known = labels.isin(offsets)
    usable = usable[known]
    if usable.empty:
        return TaxonOffsets((), z_min, tau2, tuple(levels))

    x = np.log(usable[dbh_col].to_numpy(dtype=float))
    y = np.log(usable[asw_col].to_numpy(dtype=float))
    residual = y - (fit.intercept + fit.slope * x + usable[group_col].astype(str).map(offsets).to_numpy())
    usable = usable.assign(
        _residual=residual,
        _site=usable[site_col].astype(str),
        species=usable[species_col].map(normalise_species),
        genus=usable[species_col].map(normalise_genus),
    )

    entries: list[TaxonOffset] = []
    for level in levels:
        groups = {
            str(key): sub.groupby("_site")["_residual"].mean()
            for key, sub in usable.groupby(level)
            if key and len(sub.groupby("_site")) >= min_sites
        }
        # Typical between-site variance at this level, used to moderate each
        # taxon's own unstable estimate. Median, not mean: a single taxon with a
        # freak variance must not set the prior for everyone else.
        spreads = [float(v.var(ddof=1)) for v in groups.values() if len(v) > 1]
        prior = float(np.median(spreads)) if spreads else DEFAULT_PRIOR_VARIANCE
        if not np.isfinite(prior) or prior <= 0:
            prior = DEFAULT_PRIOR_VARIANCE

        for key, site_means in groups.items():
            sub = usable[usable[level].astype(str) == key]
            n_sites = len(site_means)
            own_df = n_sites - 1
            own = float(site_means.var(ddof=1)) if own_df > 0 else 0.0
            # Moderated variance: the taxon's own estimate pooled toward the
            # corpus prior, so few-site taxa cannot fake a tiny standard error.
            moderated = (VARIANCE_PRIOR_DF * prior + own_df * own) / (VARIANCE_PRIOR_DF + own_df)
            se2 = moderated / n_sites
            raw = float(site_means.mean())
            if se2 <= 0:
                continue
            z = abs(raw) / math.sqrt(se2)
            if z < z_min:
                continue
            entries.append(
                TaxonOffset(
                    level=level,
                    key=str(key),
                    offset=raw * tau2 / (tau2 + se2),
                    raw_offset=raw,
                    z=z,
                    n_trees=int(len(sub)),
                    n_sites=int(len(site_means)),
                )
            )
    return TaxonOffsets(tuple(entries), z_min, tau2, tuple(levels))
