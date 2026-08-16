"""Everything the imputer *says* about a fit, in print and in JSON.

Split from the orchestrator so that module stays about deciding which trees
vote and what gets written, not about formatting. Nothing here computes a
coefficient: every function takes a finished fit or a finished diagnostic and
renders it.

The run log is treated as an artifact in its own right. A reader should be able
to tell from it alone which trees entered the fit, which correction factor was
applied and whether it was entitled to be, and what the relationship is worth on
a site it has never seen -- without opening the code or the JSON.
"""

from __future__ import annotations

import math
from collections import Counter

import pandas as pd

from src.sapwood_imputation.allometry import AdditiveFit
from src.sapwood_imputation.columns import ASW_COL, DBH_COL, ORIGIN_COL, SITE_COL
from src.sapwood_imputation.geometry import describe, implausible_mask
from src.sapwood_imputation.provenance import excluded_verdicts
from src.sapwood_imputation.sources import MetadataFile


def fit_report(fit: AdditiveFit) -> dict:
    """Serialise the fit for the JSON report.

    Written explicitly rather than via ``asdict`` so the tuple-of-pairs fields
    come out as readable objects and the model form is stated in the artifact
    itself -- a reader should not have to open the code to know whether the
    functional type carried its own slope.
    """
    return {
        "form": "ln(ASW) = intercept + slope * ln(DBH) + offset[functional_type]",
        "reference": fit.reference,
        "intercept": fit.intercept,
        "slope": fit.slope,
        "offsets": dict(fit.offsets),
        "correction_factor": fit.correction_factor,
        # Which back-transform was applied, what the alternative was, and the
        # residual shape that decides between them -- so a reader can tell
        # whether exp(sigma^2/2) was entitled to be used here.
        "back_transform": {
            "applied": fit.back_transform,
            "baskerville_factor": fit.baskerville_factor,
            "smearing_factor": fit.smearing_factor,
            "residual_skew": fit.residual_skew,
            "residual_excess_kurtosis": fit.residual_excess_kurtosis,
            "residuals_look_normal": fit.residuals_look_normal,
        },
        "dbh_range_cm": [fit.dbh_min, fit.dbh_max],
        "mse": fit.mse,
        "residual_sd": math.sqrt(fit.mse),
        "r2": fit.r2,
        "r2_without_functional_type": fit.r2_without_group,
        "n": fit.n,
        "n_by_group": dict(fit.n_by_group),
        "residual_sd_by_group": dict(fit.residual_sd_by_group),
    }


def geometry_report(pooled: pd.DataFrame, measured: pd.Series, dbh: pd.Series) -> dict:
    """Gate statistics for the whole pool and for each corpus separately.

    Reported per origin on purpose: the point of the split is to show that the
    internal corpus is held to the archive's own standard, not a stricter one.

    ``measured`` and ``dbh`` are the *already coerced* columns. They are passed
    in rather than derived here so this module never has to reach back into the
    orchestrator, and so the run coerces once instead of per report section.
    """
    report = {"all": describe(measured, dbh)}
    if ORIGIN_COL in pooled.columns:
        for origin, index in pooled.groupby(ORIGIN_COL).groups.items():
            report[str(origin)] = describe(measured.loc[index], dbh.loc[index])
    return report


def print_provenance(report: dict, policy: str) -> None:
    """Show how each tree's sapwood area came to exist, and what that costs."""
    by_verdict = report.get("by_verdict", {})
    if not by_verdict:
        return
    total = sum(c["trees"] for c in by_verdict.values())
    print("\n[ASW provenance]  is the value a measurement, or the provider's own formula?")
    for verdict in sorted(by_verdict, key=lambda v: -by_verdict[v]["trees"]):
        c = by_verdict[verdict]
        share = 100.0 * c["trees"] / total if total else 0.0
        print(f"  {verdict:26s} {c['trees']:5d} tree(s) in {c['groups']:4d} group(s)  ({share:5.1f}%)")

    sites = report.get("estimated_sites", {})
    if not sites:
        return
    # Say per verdict whether the finding was ACTED ON or merely recorded, so a
    # reader of the log never has to infer the policy from the tree counts.
    barred = excluded_verdicts(policy)
    for verdict, names in sorted(sites.items()):
        fate = "excluded" if verdict in barred else "reported only"
        print(f"  [{verdict}] {len(names)} site(s), {fate}: {', '.join(names)}")

    n_barred = sum(by_verdict[v]["trees"] for v in sites if v in barred and v in by_verdict)
    n_kept = sum(by_verdict[v]["trees"] for v in sites if v not in barred and v in by_verdict)
    print(f"  -> policy '{policy}': {n_barred} tree(s) excluded from the fit, {n_kept} detected but kept.")
    if n_kept:
        print("     (--estimated-exclusion all would drop those too; see the module docstring for the cost)")


def print_fit_pool_scales(fit_files: list[MetadataFile], cross_scale_dropped: list[dict]) -> None:
    """Show which scales fed the single fit, and what deduplication removed."""
    by_scale = Counter(f.scale or "?" for f in fit_files)
    print(f"\n[Fit pool] one relationship fitted across all scales: {len(fit_files)} site(s)")
    print(f"  contributing copies by scale: {dict(sorted(by_scale.items()))}")
    if not cross_scale_dropped:
        return
    print(f"  {len(cross_scale_dropped)} site(s) exist at >1 scale; kept the copy with more fit-usable trees:")
    # Only the cases where the choice actually mattered are worth the lines.
    informative = [d for d in cross_scale_dropped if d["fittable_dropped"] != d["fittable_kept"]]
    for d in informative:
        print(
            f"    {d['site']}: kept {d['kept_scale']} ({d['fittable_kept']} trees) "
            f"over {d['dropped_scale']} ({d['fittable_dropped']})"
        )
    if len(informative) < len(cross_scale_dropped):
        print(f"    (+{len(cross_scale_dropped) - len(informative)} identical at both scales; either copy would do)")


def print_pool_provenance(
    pooled: pd.DataFrame,
    voting: pd.DataFrame,
    geometry: dict,
    measured: pd.Series,
    dbh: pd.Series,
) -> None:
    """Show what each corpus contributed, and what the gate refused."""
    if ORIGIN_COL not in pooled.columns:
        return
    print("\n[Fit pool]")
    # Count only the trees that actually enter the regression: eligible AND
    # carrying both a positive ASW and a positive DBH. Reporting the eligible
    # rows instead would print per-origin numbers that do not sum to fit.n.
    contributing = voting[(voting[ASW_COL] > 0) & (voting[DBH_COL] > 0)]
    for origin in sorted(pooled[ORIGIN_COL].unique()):
        n_voting = int((contributing[ORIGIN_COL] == origin).sum()) if ORIGIN_COL in contributing.columns else 0
        n_offered = int((pooled[ORIGIN_COL] == origin).sum())
        stats = geometry.get(str(origin), {})
        print(
            f"  {str(origin):9s}: {n_voting:5d} of {n_offered:5d} trees contribute to the fit   "
            f"(gate refused {stats.get('implausible', 0)}, "
            f"ASW>basal-area {stats.get('oversized', 0)})"
        )
    refused = geometry["all"].get("implausible", 0)
    if refused:
        rejected = pooled[implausible_mask(measured, dbh)]
        by_site = rejected.groupby(SITE_COL).size().sort_values(ascending=False)
        print(f"  [gate] {refused} tree(s) refused -- ASW contradicts its own DBH; by site:")
        for site, n in by_site.items():
            print(f"           {site}: {n}")


def print_backtransform(fit: AdditiveFit) -> None:
    """State which correction factor was applied, and whether it was entitled to be.

    ``exp(sigma^2/2)`` is exact for Gaussian log residuals and merely plausible
    otherwise. On the real pool the residuals are strongly left-skewed, so the
    two estimators disagree; printing both keeps that from being a silent choice.
    """
    print(
        f"  back-transform: {fit.back_transform}  CF={fit.correction_factor:6.4f}  "
        f"(baskerville {fit.baskerville_factor:6.4f} | smearing {fit.smearing_factor:6.4f})"
    )
    if fit.residuals_look_normal:
        return
    divergence = 100.0 * (fit.baskerville_factor / fit.smearing_factor - 1.0) if fit.smearing_factor else 0.0
    print(
        f"  [warning] log residuals are not Gaussian (skew {fit.residual_skew:+.2f}, "
        f"excess kurtosis {fit.residual_excess_kurtosis:+.2f}), so exp(sigma^2/2) has no closed-form"
    )
    print(
        f"            warrant here; it sits {divergence:+.1f}% from the distribution-free smearing "
        "estimate. Use --back-transform smearing to apply the latter."
    )


def print_validation(report: dict) -> None:
    """Print what the fit is worth out of sample, not just how well it fits."""
    loso = report.get("leave_one_site_out", {})
    un = loso.get("unweighted")
    if un:
        print(f"\n[Validation] leave-one-site-out over {loso['scored_sites']} site(s)")
        print(
            f"  out-of-site R2={un['r2']:5.3f}  residual SD={un['residual_sd_log']:5.3f}  "
            f"median |relative error|={100 * un['median_relative_error']:4.1f}%"
        )
        print(
            f"  per-site (each site counts once): median |relative error|="
            f"{100 * un['site_weighted_median_relative_error']:4.1f}%  "
            f"median bias={un['site_weighted_median_bias_log']:+.3f} log-units"
        )
        balanced = loso.get("site_balanced")
        beats = loso.get("site_balanced_beats_unweighted_at")
        if balanced and beats:
            verdict = "worse" if balanced["r2"] < un["r2"] else "better"
            print(
                f"  site-balanced weighting would predict a held-out site {verdict} "
                f"(R2={balanced['r2']:5.3f}, wins at {beats['sites']}/{beats['of']} sites)"
                " -- the unweighted fit is kept."
            )

    size = report.get("residual_spread_by_size", {})
    if size.get("heteroscedastic"):
        rows = size.get("by_bin", [])
        if rows:
            first, last = rows[0], rows[-1]
            print(
                f"  [warning] residual SD rises {size['sd_ratio_largest_over_smallest']:.2f}x across tree size "
                f"({first['residual_sd']:.2f} at {first['dbh_min']:.0f}-{first['dbh_max']:.0f} cm -> "
                f"{last['residual_sd']:.2f} at {last['dbh_min']:.0f}-{last['dbh_max']:.0f} cm, "
                f"z={size['sd_trend_z']:.1f});"
            )
            print("            one pooled correction factor cannot be right for both ends.")

    genus = report.get("bias_by_genus", {})
    worst_over, worst_under = genus.get("worst_over_predicted"), genus.get("worst_under_predicted")
    if worst_over and worst_under:
        print(
            f"  [warning] functional type is too coarse: across {genus['genera']} genera "
            f"(n>={genus['min_n']}) the systematic bias spans {genus['bias_spread']:.1f}x"
        )
        print(
            f"            worst over-predicted {worst_over['genus']} "
            f"({worst_over['over_prediction_factor']:.2f}x too much sapwood, {worst_over['functional_type']}); "
            f"worst under-predicted {worst_under['genus']} "
            f"({1 / worst_under['over_prediction_factor']:.2f}x too little, {worst_under['functional_type']})"
        )

    leverage = report.get("site_leverage", {})
    if leverage.get("sites"):
        print(
            f"  leverage: {leverage['sites']} sites, median {leverage['median_trees_per_site']:.0f} trees each; "
            f"the largest 5 hold {100 * leverage['top_5_share']:.0f}% of the evidence"
        )


def print_taxon_offsets(taxon) -> None:
    """Show which taxa earned a correction, and how much of it survived shrinkage.

    Both the raw and the shrunken offset are printed: the gap between them is
    the price of site clustering, and a reader should see that the model was not
    simply handed each taxon's sample mean.
    """
    report = taxon.to_report()
    if not report["n_offsets"]:
        print(f"\n[Taxon refinement] no taxon cleared |z| >= {report['z_min']}; predictions are the unrefined fit")
        return
    print(
        f"\n[Taxon refinement] {report['n_offsets']} taxon offset(s) earned "
        f"(|z| >= {report['z_min']}, shrunk by tau2={report['tau2']}) {report['by_level']}"
    )
    print("  a correction >1 means the functional-type model gave that taxon too much sapwood")
    shown = [e for e in report["offsets"] if e["level"] == "genus"]
    for e in sorted(shown, key=lambda e: -abs(e["offset_log"]))[:8]:
        print(
            f"    {e['taxon']:14s} x{e['corrects_over_prediction_by']:5.2f}  "
            f"offset={e['offset_log']:+6.3f} (raw {e['raw_offset_log']:+6.3f})  "
            f"z={e['z']:5.1f}  {e['n_trees']:4d} trees at {e['n_sites']:2d} sites"
        )
    n_species = report["by_level"].get("species", 0)
    if n_species:
        print(f"    (+{n_species} species-level offsets, applied before the genus ones)")
