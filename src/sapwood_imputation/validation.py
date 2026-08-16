"""Does the allometry actually predict a site nobody measured?

The fit reports an in-sample R2. That is not the number a consumer of
``pl_sapw_area_filled`` needs: roughly a third of the delivered trees have no
measured ASW at all, and they are not scattered at random -- **missingness is
site-wise**. A site either measured sapwood area or it did not. So the honest
error estimate is leave-one-SITE-out, not leave-one-tree-out, and a random
k-fold would flatter the model by letting it learn each held-out site from its
own neighbours.

Everything here is diagnostic. Nothing in this module changes a fitted
coefficient or a delivered value; it exists so the JSON report states what the
fit is worth rather than only what it is.

Four questions, in decreasing order of how much they move the answer:

:func:`bias_by_genus`
    Does one offset per functional type suffice? On the real pool it does not:
    *Larix* sits 1.38 log-units below its own prediction and *Abies* 0.52 above,
    yet both are gymnosperms -- a 6.7x spread in systematic bias inside a single
    group. Reported so a consumer can see which trees the model serves badly.

:func:`leave_one_site_out`
    What error should a new site expect? Reported against the in-sample figure,
    so the optimism is explicit.

:func:`residual_spread_by_size`
    Is one pooled sigma honest? The fit already warns when the two functional
    types scatter differently; the violation is actually across DBH, where the
    residual SD climbs steadily with tree size.

:func:`site_leverage`
    How much of a "global" relationship rests on a few well-instrumented
    stands? The unweighted fit lets a site with 155 trees outvote seventeen
    sites with 9. That is a defensible choice -- :func:`leave_one_site_out`
    shows site-balanced weighting predicts a held-out site *worse* -- but it is
    a choice, and it should be visible.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.sapwood_imputation.functional_type import extract_genus

#: Minimum trees before a genus's mean residual is worth reporting.
MIN_GENUS_N = 30

#: Minimum trees in the training set before a held-out site is scored.
MIN_TRAIN_N = 20

#: Quantile bins used to look for heteroscedasticity across tree size.
SIZE_BINS = 10

#: z at which the smallest-vs-largest size bin counts as genuinely different
#: scatter. A bare ratio threshold will not do: the ratio of two sample SDs is
#: noisy, and max/min over ten bins reaches 1.6 on perfectly homoscedastic data
#: when each bin holds ~20 trees. The comparison is therefore pre-specified
#: (first bin vs last, the direction the question is about) and tested against
#: its own standard error, so the flag does not fire on small corpora.
#: NOTE: there is deliberately NO bare ratio threshold here. An earlier
#: SPREAD_RATIO_WARN fired on homoscedastic data; the z above replaced it.
SPREAD_Z_WARN = 2.5


def _design(x: np.ndarray, labels: np.ndarray, groups: list[str], reference: str) -> np.ndarray:
    """Intercept + ln(DBH) + one indicator per non-reference group."""
    dummies = [g for g in groups if g != reference]
    return np.column_stack([np.ones(len(x)), x, *((labels == g).astype(float) for g in dummies)])


def _ols(design: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        return coefficients
    root = np.sqrt(weights)
    coefficients, *_ = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)
    return coefficients


def _arrays(
    frame: pd.DataFrame, group_col: str, dbh_col: str, asw_col: str
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """(usable rows, ln DBH, ln ASW, group label) for a log-log fit.

    The filtered frame is returned alongside the arrays deliberately. Every
    caller needs both -- the arrays to compute residuals, the frame to attach
    them back to a site or a species -- and deriving the two separately invites
    the filters to drift apart, which would silently pair each residual with the
    wrong tree's label and corrupt the headline validation numbers without ever
    raising.
    """
    usable = frame[(frame[dbh_col] > 0) & (frame[asw_col] > 0)]
    x = np.log(usable[dbh_col].to_numpy(dtype=float))
    y = np.log(usable[asw_col].to_numpy(dtype=float))
    return usable, x, y, usable[group_col].astype(str).to_numpy()


def leave_one_site_out(
    frame: pd.DataFrame,
    group_col: str,
    dbh_col: str,
    asw_col: str,
    site_col: str,
    min_train: int = MIN_TRAIN_N,
) -> dict:
    """Refit without each site in turn and score the trees it held out.

    Reported two ways, because they answer different questions and disagree:

    * **tree-weighted** -- every tree counts once. This is what the fit
      optimises, and what to quote for the corpus as a whole.
    * **site-weighted** -- every site counts once. This is what a global map
      needs, where a stand with 155 trees is one place on Earth, not 155.

    ``site_balanced`` repeats the whole exercise training on weights of
    ``1/n_site``, to test whether the unweighted fit's dependence on a few
    large stands actually costs it predictive skill.
    """
    usable, x, y, labels = _arrays(frame, group_col, dbh_col, asw_col)
    sites = usable[site_col].astype(str).to_numpy()
    groups = sorted(set(labels))
    if len(x) < min_train + 1 or len(groups) == 0:
        return {"scored_sites": 0, "note": "too few trees to cross-validate"}
    reference = groups[0]
    counts = pd.Series(sites).value_counts()

    per_site: dict[str, list[dict]] = {"unweighted": [], "site_balanced": []}
    predictions = {k: np.full(len(y), np.nan) for k in per_site}
    for site in sorted(set(sites)):
        test = sites == site
        train = ~test
        if int(train.sum()) < min_train or len(set(labels[train])) < len(groups):
            continue
        design_train = _design(x[train], labels[train], groups, reference)
        design_test = _design(x[test], labels[test], groups, reference)
        weights = np.array([1.0 / counts[s] for s in sites[train]])
        for name, w in (("unweighted", None), ("site_balanced", weights)):
            residual = y[test] - design_test @ _ols(design_train, y[train], w)
            predictions[name][test] = y[test] - residual
            per_site[name].append(
                {
                    "site": site,
                    "n": int(test.sum()),
                    "rmse_log": float(np.sqrt(np.mean(residual**2))),
                    "bias_log": float(np.mean(residual)),
                    "median_relative_error": float(np.median(np.abs(np.exp(residual) - 1.0))),
                }
            )

    out: dict = {"scored_sites": len(per_site["unweighted"]), "trees": int(len(y)), "min_train_n": min_train}
    for name, rows in per_site.items():
        if not rows:
            continue
        covered = np.isfinite(predictions[name])
        residual = y[covered] - predictions[name][covered]
        # SST over the SCORED trees, not the whole pool. A site whose training
        # set lost a functional type is skipped, and scoring its residuals
        # against the full corpus's variance would credit the model with
        # variance it was never asked to explain.
        sst = float(np.sum((y[covered] - y[covered].mean()) ** 2))
        n = np.array([r["n"] for r in rows], dtype=float)
        out[name] = {
            "trees_scored": int(covered.sum()),
            "r2": 1.0 - float(np.sum(residual**2)) / sst if sst > 0 else 1.0,
            "residual_sd_log": float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0,
            "bias_log": float(np.mean(residual)),
            "median_relative_error": float(np.median(np.abs(np.exp(residual) - 1.0))),
            "site_weighted_median_rmse_log": float(np.median([r["rmse_log"] for r in rows])),
            "site_weighted_median_relative_error": float(np.median([r["median_relative_error"] for r in rows])),
            "site_weighted_median_bias_log": float(np.median([r["bias_log"] for r in rows])),
            "trees_per_scored_site_median": float(np.median(n)),
        }
    if "unweighted" in out and "site_balanced" in out:
        rmse = {k: {r["site"]: r["rmse_log"] for r in v} for k, v in per_site.items()}
        shared = set(rmse["unweighted"]) & set(rmse["site_balanced"])
        better = sum(1 for s in shared if rmse["site_balanced"][s] < rmse["unweighted"][s])
        out["site_balanced_beats_unweighted_at"] = {"sites": better, "of": len(shared)}
    return out


def residual_spread_by_size(
    frame: pd.DataFrame,
    group_col: str,
    dbh_col: str,
    asw_col: str,
    fit,
    bins: int = SIZE_BINS,
) -> dict:
    """Residual SD and mean per DBH quantile bin.

    A single pooled ``sigma`` -- and therefore a single back-transform factor --
    assumes the scatter is the same for a 5 cm sapling and a 150 cm veteran. The
    mean column matters as much as the SD: a trend in it is lack of fit, i.e. the
    log-log line is curved and the straight one is systematically wrong at both
    ends of the range.
    """
    _, x, y, labels = _arrays(frame, group_col, dbh_col, asw_col)
    if len(x) < bins * 2:
        return {"bins": 0, "note": "too few trees to bin"}
    offsets = dict(fit.offsets)
    known = np.array([g in offsets for g in labels])
    x, y, labels = x[known], y[known], labels[known]
    residual = y - (fit.intercept + fit.slope * x + np.array([offsets[g] for g in labels]))

    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    index = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = index == b
        if m.sum() < 2:
            continue
        rows.append(
            {
                "dbh_min": float(np.exp(x[m].min())),
                "dbh_max": float(np.exp(x[m].max())),
                "n": int(m.sum()),
                "residual_sd": float(np.std(residual[m], ddof=1)),
                "residual_mean": float(np.mean(residual[m])),
            }
        )
    spreads = [r["residual_sd"] for r in rows if r["residual_sd"] > 0]
    ratio = max(spreads) / min(spreads) if spreads else 1.0
    means = [r["residual_mean"] for r in rows]

    # Pre-specified test: smallest trees vs largest, the direction the question
    # is about. log(s) has SE ~ 1/sqrt(2(n-1)), so the ratio of two bins' SDs
    # gets a z without needing a chi-square quantile (and so without scipy).
    z = 0.0
    trend_ratio = 1.0
    if len(rows) >= 2 and rows[0]["residual_sd"] > 0 and rows[-1]["residual_sd"] > 0:
        first, last = rows[0], rows[-1]
        trend_ratio = last["residual_sd"] / first["residual_sd"]
        se = math.sqrt(1.0 / (2 * max(first["n"] - 1, 1)) + 1.0 / (2 * max(last["n"] - 1, 1)))
        z = math.log(trend_ratio) / se if se > 0 else 0.0
    return {
        "bins": len(rows),
        "by_bin": rows,
        "sd_ratio_across_size": float(ratio),
        "sd_ratio_largest_over_smallest": float(trend_ratio),
        "sd_trend_z": float(z),
        "heteroscedastic": bool(abs(z) > SPREAD_Z_WARN),
        "mean_residual_range": float(max(means) - min(means)) if means else 0.0,
    }


def bias_by_genus(
    frame: pd.DataFrame,
    group_col: str,
    dbh_col: str,
    asw_col: str,
    species_col: str,
    fit,
    min_n: int = MIN_GENUS_N,
) -> dict:
    """Mean residual per genus -- how much the two-group model misses.

    A genus whose mean residual is ``r`` has its ASW predicted ``exp(-r)`` times
    too large. This is the model's largest systematic error on this corpus and
    the one a downstream user is most likely to be bitten by, because it does not
    average out: every *Quercus* in the delivered file is wrong the same way.

    **A genus that is the only one in its functional type always comes back
    unbiased**, however wrong the model is for it: the group's own indicator
    absorbs the offset, so the residuals within that group sum to zero by
    construction. The numbers here measure disagreement *between* genera sharing
    a type, not the absolute accuracy of any of them.
    """
    usable, x, y, labels = _arrays(frame, group_col, dbh_col, asw_col)
    usable = usable.copy()
    offsets = dict(fit.offsets)
    known = np.array([g in offsets for g in labels])
    if not known.any():
        return {"genera": 0}
    residual = np.full(len(y), np.nan)
    residual[known] = y[known] - (fit.intercept + fit.slope * x[known] + np.array([offsets[g] for g in labels[known]]))
    usable = usable.assign(
        _residual=residual,
        _genus=usable[species_col].map(extract_genus) if species_col in usable.columns else None,
    )
    if usable["_genus"].isna().all():
        return {"genera": 0}

    rows = []
    for genus, sub in usable.dropna(subset=["_genus", "_residual"]).groupby("_genus"):
        if len(sub) < min_n:
            continue
        mean = float(sub["_residual"].mean())
        rows.append(
            {
                "genus": str(genus),
                "functional_type": str(sub[group_col].mode().iat[0]),
                "n": int(len(sub)),
                "mean_residual_log": mean,
                # >1 means the model predicts more sapwood than the trees have.
                "over_prediction_factor": float(np.exp(-mean)),
            }
        )
    rows.sort(key=lambda r: r["mean_residual_log"])
    factors = [r["over_prediction_factor"] for r in rows]
    return {
        "genera": len(rows),
        "min_n": min_n,
        "by_genus": rows,
        "worst_over_predicted": rows[0] if rows else None,
        "worst_under_predicted": rows[-1] if rows else None,
        "bias_spread": float(max(factors) / min(factors)) if factors else 1.0,
    }


def site_leverage(frame: pd.DataFrame, dbh_col: str, asw_col: str, site_col: str, top: int = 10) -> dict:
    """How concentrated the fit's evidence is across sites."""
    usable = frame[(frame[dbh_col] > 0) & (frame[asw_col] > 0)]
    counts = usable[site_col].astype(str).value_counts()
    total = int(counts.sum())
    if total == 0:
        return {"sites": 0}
    return {
        "sites": int(len(counts)),
        "trees": total,
        "median_trees_per_site": float(counts.median()),
        "top_5_share": float(counts.head(5).sum() / total),
        "top_10_share": float(counts.head(10).sum() / total),
        "largest_sites": [{"site": s, "trees": int(n)} for s, n in counts.head(top).items()],
    }


def validate(
    frame: pd.DataFrame,
    fit,
    group_col: str,
    dbh_col: str,
    asw_col: str,
    site_col: str,
    species_col: str,
) -> dict:
    """Every diagnostic above, for the JSON report."""
    return {
        "leave_one_site_out": leave_one_site_out(frame, group_col, dbh_col, asw_col, site_col),
        "residual_spread_by_size": residual_spread_by_size(frame, group_col, dbh_col, asw_col, fit),
        "bias_by_genus": bias_by_genus(frame, group_col, dbh_col, asw_col, species_col, fit),
        "site_leverage": site_leverage(frame, dbh_col, asw_col, site_col),
    }
