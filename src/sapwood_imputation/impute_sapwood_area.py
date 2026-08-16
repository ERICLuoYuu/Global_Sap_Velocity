"""Impute missing sapwood area (ASW) for SAPFLUXNET trees via allometry.

Fits one global log-log relationship taking DBH *and functional type* as
predictors,

    ln(ASW) = intercept + slope * ln(DBH) + offset[functional_type]

on every tree that HAS a measured ``pl_sapw_area``, then predicts ASW (cm²,
Baskerville-corrected) for trees where it is missing but DBH and functional
type are known. This is the form described by Poyatos et al. (2021), the
SAPFLUXNET data paper, whose own ASW estimates this reproduces: functional type
shifts the intercept, it does not get its own slope. See
:mod:`src.sapwood_imputation.allometry` for the alternative interaction form.

Two corpora feed this: the SAPFLUXNET archive and, with ``--include-internal``,
the European sites processed by ``Sapflow-internal/``. See
:mod:`src.sapwood_imputation.sources` for which internal trees are allowed to
vote on the fit (three sites re-measure stems the archive already holds) and
:mod:`src.sapwood_imputation.geometry` for the unit-error gate applied to both.

Output is non-destructive. ``pl_sapw_area`` is never changed; three columns
are added:

* ``pl_functional_type`` -- angiosperm / gymnosperm / unknown
* ``pl_sapw_area_filled`` -- measured value, else the imputed estimate
* ``pl_sapw_area_source``  -- ``measured`` / ``imputed`` / ``implausible`` /
  ``unresolved``

CLI
---
    python -m src.sapwood_imputation.impute_sapwood_area \
        --scale plant --base-data-dir ./data --include-internal

Run with ``--dry-run`` to print the fit + a summary without writing files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.sapwood_imputation import reporting, validation
from src.sapwood_imputation.allometry import (
    BACKTRANSFORM_BASKERVILLE,
    BACKTRANSFORMS,
    AdditiveFit,
    fit_loglog_with_group_offset,
    predict_asw_additive,
)
from src.sapwood_imputation.columns import (
    ASW_COL,
    DBH_COL,
    FILLED_COL,
    FT_COL,
    ORIGIN_COL,
    SITE_COL,
    SOURCE_COL,
    SOURCE_FLAGS,
    SOURCE_IMPLAUSIBLE,
    SOURCE_IMPUTED,
    SOURCE_MEASURED,
    SOURCE_PROVIDER_ESTIMATE,
    SOURCE_UNRESOLVED,
    SPECIES_COL,
)
from src.sapwood_imputation.functional_type import (
    ANGIOSPERM,
    GYMNOSPERM,
    classify,
)
from src.sapwood_imputation.geometry import implausible_mask, is_implausible
from src.sapwood_imputation.numerics import coerce_series, defect_count

# Aliased: both modules export a summarise(), and importing either bare would
# silently shadow the other -- sources' one names the duplicate-stem sites.
from src.sapwood_imputation.provenance import (
    EXCLUSION_POLICIES,
    POLICY_DECLARED,
    estimated_mask,
    excluded_verdicts,
)
from src.sapwood_imputation.provenance import assess as assess_provenance
from src.sapwood_imputation.provenance import summarise as summarise_provenance
from src.sapwood_imputation.sources import (
    MetadataFile,
    discover,
    discover_across_scales,
    summarise,
)
from src.sapwood_imputation.taxon_offsets import TaxonOffsets, fit_taxon_offsets

#: Dataset flavours pooled into the single global fit. Both hold the same
#: ``plant_md`` geometry; only the flux normalisation differs.
FIT_SCALES = ("plant", "sapwood")

# Only these functional types participate in the allometric fit.
FITTABLE_TYPES = (ANGIOSPERM, GYMNOSPERM)

# NOTE: the column names and source flags live in columns.py and the
# BACKTRANSFORM_* names in allometry.py, both imported above. Do NOT redeclare
# any of them here: a module-level copy shadows the import and silently changes
# which trees reach the fit or which correction factor is applied.
# NOTE: POLICY_* live in provenance.py and are imported above. Do NOT redefine
# them here -- a local copy shadows the import and silently changes which trees
# reach the fit.


# ---------------------------------------------------------------------------
# Dataframe-level logic (pure, unit-tested)
# ---------------------------------------------------------------------------
def add_functional_type(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a :data:`FT_COL` column derived from species."""
    out = df.copy()
    if SPECIES_COL in out.columns:
        species = out[SPECIES_COL]
    else:
        species = pd.Series([None] * len(out), index=out.index)
    out[FT_COL] = species.map(classify)
    return out


def coerce_inputs(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict[str, Counter]]:
    """Return (measured_asw, dbh) as canonical-unit floats, plus status counts.

    Provider cells arrive as free text (``'16.5 cm'``, ``'207.2 cm2'``, ``'n/a'``).
    Coercing here -- once, at the module's data boundary -- means every downstream
    comparison operates on floats, and values carrying a unit are *converted*
    rather than dropped. Missing columns yield all-``NaN`` series so callers can
    process a partial file instead of rejecting it.
    """
    nan_series = pd.Series(np.nan, index=df.index, dtype=float)
    counts: dict[str, Counter] = {}

    coerced: dict[str, pd.Series] = {}
    for col in (ASW_COL, DBH_COL):
        if col in df.columns:
            coerced[col], counts[col] = coerce_series(df[col], field=col)
        else:
            coerced[col], counts[col] = nan_series, Counter({"missing_column": int(len(df))})

    return coerced[ASW_COL], coerced[DBH_COL], counts


def coerced_and_typed(df: pd.DataFrame) -> pd.DataFrame:
    """Add functional type and replace ASW/DBH with canonical-unit floats."""
    typed = add_functional_type(df)
    # Coerce before anything compares these values, so unit-carrying cells
    # ('16.5 cm') contribute to the regression instead of being silently
    # discarded by the log-log valid mask.
    measured, dbh, _ = coerce_inputs(typed)
    return typed.assign(**{ASW_COL: measured, DBH_COL: dbh})


def fit_frame_from_dataframe(df: pd.DataFrame, policy: str = POLICY_DECLARED) -> pd.DataFrame:
    """Coerce, type, and gate ``df`` down to the trees allowed into the fit.

    Three exclusions:

    1. a functional type the model cannot carry (family-level names would
       otherwise form a bogus third group);
    2. an ASW/DBH pair that contradicts itself geometrically -- see
       :mod:`src.sapwood_imputation.geometry`;
    3. an ASW the provider *computed* rather than measured, which would make
       this regression a regression on other regressions -- see
       :mod:`src.sapwood_imputation.provenance`.

    The third is governed by ``policy`` and defaults to the documented sites
    only. It needs ``_site`` and ``pl_species`` to form its groups, so it is a
    no-op on a bare frame; ``run()`` supplies both via :func:`_pool_measured`.
    """
    typed = coerced_and_typed(df)
    keep = typed[FT_COL].isin(FITTABLE_TYPES) & ~implausible_mask(typed[ASW_COL], typed[DBH_COL])
    keep &= ~estimated_mask(typed, SITE_COL, SPECIES_COL, DBH_COL, ASW_COL, policy=policy)
    return typed[keep]


def fit_from_dataframe(df: pd.DataFrame, min_n: int = 2, policy: str = POLICY_DECLARED) -> AdditiveFit:
    """Fit the global DBH + functional-type allometry from measured trees."""
    frame = fit_frame_from_dataframe(df, policy=policy)
    return fit_loglog_with_group_offset(frame, FT_COL, DBH_COL, ASW_COL, min_n=min_n)


def _row_estimate(
    measured: float,
    dbh: float,
    ftype: str,
    fit: AdditiveFit,
    provider_estimate: bool = False,
    taxon_offset: float = 0.0,
) -> tuple[float, str]:
    """Return (filled_value, source) for a single tree.

    ``measured`` and ``dbh`` must already be floats (see :func:`coerce_inputs`).
    ``provider_estimate`` labels a value the provider computed; it changes the
    label only -- the value is passed through, because it is still the best
    number for that tree and is what their published flux density used.

    ``taxon_offset`` is an additive log-scale correction for this tree's species
    or genus (see :mod:`src.sapwood_imputation.taxon_offsets`). It applies ONLY
    to an imputed value: a measured ASW is never adjusted toward a model.
    """
    if pd.notna(measured) and measured > 0:
        # Refuse a measurement its own DBH contradicts, and do NOT fall through
        # to imputation: the allometry would read that same suspect DBH.
        # Scalar predicate, NOT implausible_mask -- that one takes Series.
        if is_implausible(measured, dbh):
            return float("nan"), SOURCE_IMPLAUSIBLE
        return float(measured), SOURCE_PROVIDER_ESTIMATE if provider_estimate else SOURCE_MEASURED
    # Need to impute: requires a fitted functional type and a positive DBH.
    # 'unknown' is deliberately NOT imputed at the reference level -- an
    # unresolvable species must stay visible, not inherit angiosperm by default.
    if ftype not in fit.groups or pd.isna(dbh) or dbh <= 0:
        return float("nan"), SOURCE_UNRESOLVED
    try:
        estimate = predict_asw_additive(float(dbh), ftype, fit)
    except (KeyError, ValueError):
        return float("nan"), SOURCE_UNRESOLVED
    # exp() because the offset lives on the log scale, where the fit does; a
    # zero offset (the default, and every unrefined taxon) leaves it untouched.
    return estimate * math.exp(taxon_offset), SOURCE_IMPUTED


def impute_dataframe_with_report(
    df: pd.DataFrame,
    fit: AdditiveFit | None = None,
    estimated_species: frozenset[str] = frozenset(),
    taxon: TaxonOffsets | None = None,
) -> tuple[pd.DataFrame, dict[str, Counter]]:
    """:func:`impute_dataframe`, also returning per-column coercion status counts.

    ``estimated_species`` names the species *at this site* whose ASW the provider
    computed. It is decided globally (the verdict needs the whole site x species
    group) and passed down, because one file alone cannot tell a tight
    measurement series from a formula.
    """
    out = add_functional_type(df)
    if fit is None:
        fit = fit_from_dataframe(df)

    measured, dbh, coercion = coerce_inputs(out)
    species = out[SPECIES_COL].astype(str) if SPECIES_COL in out.columns else pd.Series("", index=out.index)

    filled: list[float] = []
    source: list[str] = []
    # No zip(strict=): Palma runs Python 3.9, where that keyword does not exist.
    # All four iterables are columns of `out`, so equal length is guaranteed.
    for m, d, ft, sp in zip(measured, dbh, out[FT_COL], species):  # noqa: B905
        offset = taxon.lookup(sp)[0] if taxon is not None else 0.0
        value, src = _row_estimate(m, d, ft, fit, provider_estimate=sp in estimated_species, taxon_offset=offset)
        filled.append(value)
        source.append(src)

    out[FILLED_COL] = filled
    out[SOURCE_COL] = source
    return out, coercion


def impute_dataframe(
    df: pd.DataFrame,
    fit: AdditiveFit | None = None,
    estimated_species: frozenset[str] = frozenset(),
    taxon: TaxonOffsets | None = None,
) -> pd.DataFrame:
    """Return a copy of ``df`` with functional type + filled ASW + source columns.

    ``fit`` lets callers pass a *global* fit (computed across all sites). If
    omitted, the fit is computed from ``df`` alone. Tolerates a frame missing
    any of ``pl_species`` / ``pl_dbh`` / ``pl_sapw_area``: the affected rows
    simply come back as ``unresolved``.
    """
    return impute_dataframe_with_report(df, fit, estimated_species, taxon)[0]


# ---------------------------------------------------------------------------
# File-level orchestration
# ---------------------------------------------------------------------------
def _fittable_count(path: Path) -> int:
    """Trees in ``path`` usable by the regression: positive DBH *and* ASW.

    Uses the same unit-aware coercion as the fit itself. A plain
    ``pd.to_numeric`` would score ``'16.5 cm'`` as unusable, so a copy whose
    provider wrote units would lose the cross-scale tie-break to a poorer one --
    silently, and in the one place where the choice decides which trees vote.
    """
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        # Ranked last rather than crashing the run, but never silently: an
        # unreadable copy losing a tie-break must be visible in the log.
        warnings.warn(
            f"Could not read {path} while ranking scale copies ({exc!r}); counted as 0 fittable trees.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0
    if not {DBH_COL, ASW_COL}.issubset(df.columns):
        return 0
    dbh, _ = coerce_series(df[DBH_COL], field=DBH_COL)
    asw, _ = coerce_series(df[ASW_COL], field=ASW_COL)
    return int(((dbh > 0) & (asw > 0)).sum())


def one_copy_per_site(files: list[MetadataFile]) -> tuple[list[MetadataFile], list[dict]]:
    """Keep one copy of each site, preferring the one with the most fit-usable trees.

    A site present at both scales describes ONE set of stems, so only one copy
    may vote. Picking the richer copy rather than a fixed scale matters: ITA_CP2
    has 8 stems at both scales, but its plant copy carries no ``pl_sapw_area``
    at all, so preferring 'plant' blindly would silently discard eight measured
    trees. Ties break on scale name, so the choice is deterministic.

    Returns the kept files and a record of what was dropped, for the report.
    """
    by_site: dict[str, list[MetadataFile]] = {}
    for f in files:
        by_site.setdefault(f.site, []).append(f)

    kept: list[MetadataFile] = []
    dropped: list[dict] = []
    for site in sorted(by_site):
        copies = by_site[site]
        if len(copies) == 1:
            kept.append(copies[0])
            continue
        ranked = sorted(copies, key=lambda f: (-_fittable_count(f.path), f.scale))
        kept.append(ranked[0])
        for loser in ranked[1:]:
            dropped.append(
                {
                    "site": loser.site,
                    "dropped_scale": loser.scale,
                    "kept_scale": ranked[0].scale,
                    "fittable_dropped": _fittable_count(loser.path),
                    "fittable_kept": _fittable_count(ranked[0].path),
                }
            )
    return kept, dropped


def _pool_measured(files: list[MetadataFile]) -> pd.DataFrame:
    """Concatenate (species, DBH, ASW) across every site allowed to vote.

    Sites flagged ``may_vote=False`` are skipped: their stems are already in the
    pool under an archive site code, and counting them twice would over-weight a
    single stand. Origin and site are carried through so the run log can report
    what each corpus actually contributed.
    """
    frames: list[pd.DataFrame] = []
    unusable: list[str] = []
    for f in files:
        if not f.may_vote:
            continue
        df = pd.read_csv(f.path)
        if not {SPECIES_COL, DBH_COL, ASW_COL}.issubset(df.columns):
            # A site dropping out of the fit pool must never be invisible: it
            # changes the relationship every other tree is measured against.
            unusable.append(
                f"{f.site} ({f.scale or '?'}): missing {sorted({SPECIES_COL, DBH_COL, ASW_COL} - set(df.columns))}"
            )
            continue
        frame = df[[SPECIES_COL, DBH_COL, ASW_COL]].copy()
        frame[ORIGIN_COL] = f.origin
        frame[SITE_COL] = f.site
        frames.append(frame)
    if unusable:
        print(f"  [fit pool] {len(unusable)} site(s) excluded -- required columns absent:")
        for entry in unusable:
            print(f"             {entry}")
    if not frames:
        raise FileNotFoundError(
            f"No plant_md files with the required columns ({SPECIES_COL}, {DBH_COL}, {ASW_COL}) found."
        )
    return pd.concat(frames, ignore_index=True)


def _fit_pool_candidates(
    scale: str,
    base_data_dir: str,
    base_output_dir: str,
    include_internal: bool,
    internal_dir: str | None,
    resolved_internal: Path | None,
) -> list[MetadataFile]:
    """Every site at every scale that exists, before per-site deduplication.

    MUST enumerate all of :data:`FIT_SCALES` -- roughly 37 archive sites and 20
    internal ones exist at only ONE scale, and reading a single directory
    silently discards them.

    An explicit ``internal_dir`` REPLACES the internal corpus rather than adding
    to it, so it applies to the written scale alone. Falling back to
    ``PathConfig.internal_metadata_dir`` for the other scales would quietly mix
    the repository's real internal sites into a run that named its own: that
    path is anchored on the REPO, not on ``base_data_dir``, so it exists even
    under a throwaway data root.
    """
    from path_config import PathConfig

    overridden = internal_dir is not None
    metadata_dirs: dict[str, Path] = {}
    internal_dirs: dict[str, Path] = {}
    for other in FIT_SCALES:
        p = PathConfig(scale=other, base_data_dir=base_data_dir, base_output_dir=base_output_dir)
        metadata_dirs[other] = p.metadata_dir
        if not include_internal:
            continue
        if not overridden:
            internal_dirs[other] = p.internal_metadata_dir
        elif other == scale and resolved_internal is not None:
            internal_dirs[other] = resolved_internal
    return discover_across_scales(metadata_dirs, internal_dirs or None)


def run(
    scale: str = "plant",
    base_data_dir: str = "./data",
    base_output_dir: str = "./outputs",
    output_subdir: str | None = None,
    min_n: int = 2,
    dry_run: bool = False,
    include_internal: bool = False,
    internal_dir: str | None = None,
    exclusion_policy: str = POLICY_DECLARED,
    back_transform: str = BACKTRANSFORM_BASKERVILLE,
    taxon_refinement: bool = True,
) -> AdditiveFit:
    """Fit globally and write per-site imputed plant_md files.

    ``include_internal`` adds the ``Sapflow-internal/`` sites to both the fit
    pool and the output. It is opt-in so the archive-only fit stays exactly
    reproducible for comparison against the published relationship.
    ``internal_dir`` overrides where those sites are read from; by default it
    follows :attr:`PathConfig.internal_metadata_dir`, which is anchored on the
    repository rather than on ``base_data_dir``.
    ``exclusion_policy`` chooses how far to act on the provenance findings:
    ``'declared'`` (default) bars only the sites documented in
    :data:`provenance.ESTIMATED_SITES` and reports the statistically detected
    rest; ``'all'`` bars those too; ``'none'`` bars nothing.

    ``taxon_refinement`` adds the species/genus offsets of
    :mod:`src.sapwood_imputation.taxon_offsets` on top of the functional-type
    offset. It defaults to **on**. It used to default to off, so that a bare run
    reproduced the published Poyatos-form fit; that default was reversed once the
    imputed area started feeding ``sap velocity = flow / area``, because there the
    error is not a metadata blemish but a proportional error in the training
    target -- and an unrefined run is indistinguishable from a refined one after
    the fact. Identical row count, identical coverage, identical source flags, and
    a byte-identical fit report (the refinement is applied at prediction time and
    never moves the fitted line); only 280 of 587 imputed values quietly differ,
    by a median of 27%%. Pass ``taxon_refinement=False`` for the baseline.

    Returns the fitted relationship (also useful for testing).
    """
    from path_config import PathConfig

    paths = PathConfig(scale=scale, base_data_dir=base_data_dir, base_output_dir=base_output_dir)
    metadata_dir = paths.metadata_dir
    resolved_internal = None
    if include_internal:
        resolved_internal = Path(internal_dir) if internal_dir else paths.internal_metadata_dir
    files = discover(metadata_dir, resolved_internal, scale=scale)
    if not files:
        raise FileNotFoundError(f"No *_plant_md.csv files in {metadata_dir}")

    origins = summarise(files)
    print(f"[ASW imputation] scale={scale}  files={len(files)}  dir={metadata_dir}")
    if include_internal:
        print(f"  + internal corpus: {resolved_internal}")
        print(f"  files by origin: {origins['files_by_origin']}")
        for site, holder in origins["duplicate_sites_excluded_from_fit"].items():
            print(f"  [duplicate] {site} stems already in the archive as {holder} -> excluded from the fit")

    # --- Pass 1: global fit -------------------------------------------------
    # The FIT pools every scale, because DBH -> ASW is a property of stems, not
    # of how the flux was normalised; ~37 archive sites exist only at one scale,
    # and fitting per scale threw those away and produced two near-but-not-quite
    # equal relationships. The OUTPUT stays scale-keyed: consumers glob
    # {scale}_asw_filled/ and a site must appear in the scale it belongs to.
    fit_files, cross_scale_dropped = one_copy_per_site(
        _fit_pool_candidates(scale, base_data_dir, base_output_dir, include_internal, internal_dir, resolved_internal)
    )
    reporting.print_fit_pool_scales(fit_files, cross_scale_dropped)

    # fit_from_dataframe restricts to fittable functional types (angiosperm /
    # gymnosperm) so family-level "unknown" names never form a bogus group, and
    # drops ASW/DBH pairs that contradict each other geometrically.
    pooled = _pool_measured(fit_files)

    # Provenance is judged ONCE on the pooled frame: a site x species group is
    # the smallest unit that can reveal a formula, and no single file contains
    # enough of one to decide. The verdicts are then handed to the per-file pass.
    pooled_typed = coerced_and_typed(pooled)
    verdicts = assess_provenance(pooled_typed, SITE_COL, SPECIES_COL, DBH_COL, ASW_COL)
    provenance_report = summarise_provenance(verdicts)
    # The per-row label follows the POLICY, not the raw finding: a tree we chose
    # not to act on must not be branded 'provider_estimate' in the delivered
    # file. Asserting it there is itself acting on the inference. The full
    # verdict set survives in the JSON report as a diagnostic.
    barred = excluded_verdicts(exclusion_policy)
    estimated_by_site: dict[str, set[str]] = {}
    for v in verdicts:
        if v.verdict in barred:
            estimated_by_site.setdefault(v.site, set()).add(v.species)

    voting = fit_frame_from_dataframe(pooled, policy=exclusion_policy)
    fit = fit_loglog_with_group_offset(voting, FT_COL, DBH_COL, ASW_COL, min_n=min_n, back_transform=back_transform)

    gate_report = reporting.geometry_report(pooled, pooled_typed[ASW_COL], pooled_typed[DBH_COL])
    reporting.print_pool_provenance(pooled, voting, gate_report, pooled_typed[ASW_COL], pooled_typed[DBH_COL])
    reporting.print_provenance(provenance_report, exclusion_policy)

    print("\n[Global allometric fit]  ln(ASW) = intercept + slope * ln(DBH) + offset[functional type]")
    print(
        f"  slope b={fit.slope:5.3f}   R2={fit.r2:5.3f} "
        f"(ln(DBH) alone: {fit.r2_without_group:5.3f})   "
        f"CF={fit.correction_factor:6.4f}   n={fit.n}"
    )
    sizes = dict(fit.n_by_group)
    spread = dict(fit.residual_sd_by_group)
    for ft, offset in fit.offsets:
        marker = "  (reference)" if ft == fit.reference else ""
        # a = exp(intercept + offset) is the group's multiplicative coefficient,
        # i.e. predicted ASW at DBH = 1 cm before the bias correction.
        print(
            f"  {ft:11s}: a={np.exp(fit.intercept + offset):8.4f}  offset={offset:+6.3f}  "
            f"resid_sd={spread[ft]:5.3f}  n={sizes[ft]}{marker}"
        )
    # One pooled sigma is only honest if the groups scatter alike; print both so
    # a violated assumption is visible in the log rather than buried in the fit.
    if len(spread) > 1:
        lo, hi = min(spread.values()), max(spread.values())
        if lo > 0 and hi / lo > 1.5:
            print(
                f"  [warning] residual SD differs {hi / lo:.2f}x between groups; "
                "the single pooled correction factor over/under-corrects one of them."
            )
    reporting.print_backtransform(fit)
    # Diagnostics only -- validate() never touches a coefficient. Computed on the
    # frame the fit actually used, so the in-sample and out-of-site numbers
    # describe exactly one model. Must NOT be stubbed to {}: an empty report
    # silently turns the artifact back into an in-sample R2 and nothing else.
    validation_report = validation.validate(voting, fit, FT_COL, DBH_COL, ASW_COL, SITE_COL, SPECIES_COL)
    reporting.print_validation(validation_report)

    # Opt-in refinement of the functional-type offset. Strictly additive: the
    # slope, the FT offsets and the correction factor are untouched, and a taxon
    # that earns no offset is predicted exactly as it is without this.
    taxon = None
    if taxon_refinement:
        taxon = fit_taxon_offsets(voting, fit, FT_COL, DBH_COL, ASW_COL, SITE_COL, SPECIES_COL)
        reporting.print_taxon_offsets(taxon)

    # --- Pass 2: apply per file --------------------------------------------
    out_dir = (
        Path(base_data_dir).resolve()
        / "raw"
        / paths.data_version
        / paths.data_version
        / "csv"
        / (output_subdir or f"{scale}_asw_filled")
    )
    # Every input file now produces an output file, so an output dir equal to the
    # input dir would overwrite the source metadata in place. Refuse rather than
    # destroy: there is no other copy of these provider files.
    if out_dir.resolve() == Path(metadata_dir).resolve():
        raise ValueError(
            f"--output-subdir would write into the input directory ({out_dir}), "
            "overwriting the source plant metadata. Choose a different subdir."
        )

    counts = {src: 0 for src in SOURCE_FLAGS}
    counts_by_origin: dict[str, Counter] = {}
    # Files missing pl_species/pl_dbh are processed anyway (their rows come back
    # 'unresolved') so the output directory always mirrors the input site set.
    # A consumer globbing out_dir must never silently see fewer sites than exist.
    degraded: list[dict] = []
    coercion_totals: dict[str, Counter] = {ASW_COL: Counter(), DBH_COL: Counter()}

    for f in files:
        df = pd.read_csv(f.path)
        missing = sorted({SPECIES_COL, DBH_COL} - set(df.columns))
        if missing:
            print(f"  DEGRADED {f.name}: missing {', '.join(missing)} -> rows cannot be imputed")
            degraded.append({"file": f.name, "origin": f.origin, "missing_columns": missing})

        filled, coercion = impute_dataframe_with_report(df, fit, frozenset(estimated_by_site.get(f.site, set())), taxon)
        for col, col_counts in coercion.items():
            coercion_totals[col].update(col_counts)
        tally = counts_by_origin.setdefault(f.origin, Counter())
        for src in SOURCE_FLAGS:
            n = int((filled[SOURCE_COL] == src).sum())
            counts[src] += n
            tally[src] += n
        if not dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            filled.to_csv(out_dir / f.name, index=False)

    print(
        f"\n[Summary] measured={counts[SOURCE_MEASURED]}  "
        f"provider_estimate={counts[SOURCE_PROVIDER_ESTIMATE]}  "
        f"imputed={counts[SOURCE_IMPUTED]}  "
        f"implausible={counts[SOURCE_IMPLAUSIBLE]}  "
        f"unresolved={counts[SOURCE_UNRESOLVED]}"
    )
    if len(counts_by_origin) > 1:
        for origin in sorted(counts_by_origin):
            tally = counts_by_origin[origin]
            print(
                f"  {origin:9s}: measured={tally[SOURCE_MEASURED]:5d}  "
                f"provider_estimate={tally[SOURCE_PROVIDER_ESTIMATE]:5d}  "
                f"imputed={tally[SOURCE_IMPUTED]:4d}  "
                f"implausible={tally[SOURCE_IMPLAUSIBLE]:3d}  "
                f"unresolved={tally[SOURCE_UNRESOLVED]:4d}"
            )
    if degraded:
        print(f"[Degraded] {len(degraded)} file(s) lacked {SPECIES_COL}/{DBH_COL}; written as unresolved.")
    for col, col_counts in coercion_totals.items():
        defects = defect_count(col_counts)
        if defects:
            print(f"[Coercion] {col}: {defects} malformed cell(s) -- {dict(col_counts)}")

    if not dry_run:
        report_path = out_dir / "asw_allometry_fit_report.json"
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "scale": scale,
                    "include_internal": include_internal,
                    "fit": reporting.fit_report(fit),
                    "counts": counts,
                    "counts_by_origin": {o: dict(c) for o, c in counts_by_origin.items()},
                    "sources": origins,
                    "geometry_gate": gate_report,
                    "asw_provenance": provenance_report,
                    "validation": validation_report,
                    "taxon_offsets": taxon.to_report() if taxon is not None else None,
                    "fit_pool": {
                        "scales_pooled": sorted({f.scale for f in fit_files if f.scale}),
                        # Distinguished because they differ: duplicate-stem sites
                        # are read and reported but barred from voting, so
                        # 'sites' alone would overstate the fit's evidence base.
                        "sites": len(fit_files),
                        "sites_voting": sum(1 for f in fit_files if f.may_vote),
                        "cross_scale_duplicates_dropped": cross_scale_dropped,
                    },
                    "estimated_exclusion_policy": exclusion_policy,
                    "estimated_verdicts_excluded": sorted(barred),
                    "n_input_files": len(files),
                    "degraded_files": degraded,
                    "coercion": {col: dict(c) for col, c in coercion_totals.items()},
                },
                fh,
                indent=2,
            )
        print(f"[Written] imputed metadata -> {out_dir}")
        print(f"[Written] fit report      -> {report_path}")
    else:
        print("[dry-run] no files written.")

    return fit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Impute missing sapwood area via DBH/functional-type allometry.")
    parser.add_argument("--scale", default="plant", help="Dataset flavor (plant|sapwood). Default: plant")
    parser.add_argument("--base-data-dir", default="./data", help="Root data dir (default: ./data)")
    parser.add_argument("--base-output-dir", default="./outputs", help="Root output dir (default: ./outputs)")
    parser.add_argument(
        "--output-subdir", default=None, help="csv/ subdir name for output (default: <scale>_asw_filled)"
    )
    parser.add_argument("--min-n", type=int, default=2, help="Min trees per functional type to fit (default: 2)")
    parser.add_argument("--dry-run", action="store_true", help="Fit and summarise without writing files")
    parser.add_argument(
        "--include-internal",
        action="store_true",
        help="Also fit on and impute the Sapflow-internal/ European sites (default: archive only)",
    )
    parser.add_argument(
        "--estimated-exclusion",
        choices=EXCLUSION_POLICIES,
        default=POLICY_DECLARED,
        help=(
            "How far to act on the ASW provenance findings. "
            "'declared' (default) drops only the sites documented as estimated "
            "(ESP_GDN, AUT_ZOE) and reports the statistically detected rest; "
            "'all' drops those too; 'none' drops nothing."
        ),
    )
    parser.add_argument(
        "--back-transform",
        choices=BACKTRANSFORMS,
        default=BACKTRANSFORM_BASKERVILLE,
        help=(
            "Correction applied when exponentiating the log-scale fit. "
            "'baskerville' (default) is exp(sigma^2/2), exact only for Gaussian "
            "log residuals -- which this corpus does not have; 'smearing' is the "
            "distribution-free Duan (1983) estimator; 'none' gives a "
            "median-unbiased prediction. All are reported whichever is applied."
        ),
    )
    # One dest, two spellings. --taxon-refinement is redundant now that the
    # refinement is the default, but job scripts and the documented commands pass
    # it explicitly, and silently rejecting them would be a worse failure than the
    # redundancy.
    parser.add_argument(
        "--taxon-refinement",
        dest="taxon_refinement",
        action="store_true",
        default=True,
        help=(
            "Add species/genus corrections on top of the functional-type offset, "
            "kept only where they survive site clustering. Leave-one-site-out: "
            "R2 0.755 -> 0.813, median error 34.9%% -> 26.5%%. This is the default; "
            "the flag is accepted for compatibility with existing commands."
        ),
    )
    parser.add_argument(
        "--no-taxon-refinement",
        dest="taxon_refinement",
        action="store_false",
        help=(
            "Fit the plain functional-type model instead, reproducing the published "
            "Poyatos-form baseline. Use only for comparison: the imputed area feeds "
            "a division (sap velocity = flow / area), so the coarser model's error "
            "lands undiluted in downstream values."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    run(
        scale=args.scale,
        base_data_dir=args.base_data_dir,
        base_output_dir=args.base_output_dir,
        output_subdir=args.output_subdir,
        min_n=args.min_n,
        dry_run=args.dry_run,
        include_internal=args.include_internal,
        exclusion_policy=args.estimated_exclusion,
        back_transform=args.back_transform,
        taxon_refinement=args.taxon_refinement,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
