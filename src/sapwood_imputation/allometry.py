"""Log-log allometric fits for sapwood area, in two model forms.

**Additive** (:func:`fit_loglog_with_group_offset`) -- the form used by the
imputer, and the one described by Poyatos et al. (2021) as taking "DBH and
functional type as predictors":

    ln(ASW) = intercept + slope * ln(DBH) + offset[group]

One slope shared by every group; the group only shifts the intercept, so the
predicted ratio between two groups is the constant ``exp(offset_a - offset_b)``
at every DBH. One pooled residual variance, hence one correction factor.

**Interaction** (:func:`fit_loglog_by_group`) -- each group gets its own slope,
intercept and residual variance, i.e. a fully independent regression per group.
Retained for model comparison and diagnostics; it is a strictly larger model,
so its R² is always >= the additive form's.

Back-transformation
-------------------
Exponentiating a log-scale fit predicts the *median*, not the mean, so a
correction factor is applied. Two are available, and they do not agree on this
corpus:

``baskerville`` (default)
    ``exp(sigma^2 / 2)`` -- Baskerville (1972) / Sprugel (1983). Exact **if the
    log residuals are Gaussian**.

``smearing``
    ``mean(exp(residual))`` -- Duan (1983). Non-parametric: consistent whatever
    the residual distribution, at the cost of being an estimate rather than a
    closed form.

``none``
    1.0, i.e. a median-unbiased prediction. Appropriate when the value is used
    as a per-tree divisor rather than summed.

The default is kept at ``baskerville`` for continuity with the published
relationship this package reproduces, **not** because the assumption holds here:
on the real pool the residuals are strongly left-skewed and heavy-tailed
(skew -1.54, excess kurtosis +4.24), and Baskerville's 1.1700 lies outside the
bootstrap 95% CI of the smearing estimate (1.111-1.152). :attr:`AdditiveFit`
therefore carries both factors and the skew/kurtosis that decide between them,
so the choice is visible in the report instead of buried in a formula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Back-transform estimators; see the module docstring for when each is right.
BACKTRANSFORM_BASKERVILLE = "baskerville"
BACKTRANSFORM_SMEARING = "smearing"
BACKTRANSFORM_NONE = "none"
BACKTRANSFORMS = (BACKTRANSFORM_BASKERVILLE, BACKTRANSFORM_SMEARING, BACKTRANSFORM_NONE)

#: |skew| above this, or excess kurtosis above this, means the Gaussian residual
#: assumption behind ``exp(sigma^2/2)`` is not tenable and the smearing estimate
#: should be preferred. Deliberately loose: it is a flag, not a hypothesis test.
NONNORMAL_SKEW = 0.5
NONNORMAL_EXCESS_KURTOSIS = 1.0


def _moments(residuals: np.ndarray) -> tuple[float, float]:
    """Population skew and excess kurtosis of ``residuals``.

    Computed with numpy rather than ``scipy.stats`` so the package keeps its
    dependency surface at numpy/pandas; the values match ``scipy.stats.skew``
    and ``scipy.stats.kurtosis`` at their (biased, Fisher) defaults.
    """
    n = len(residuals)
    if n < 3:
        return 0.0, 0.0
    centred = residuals - residuals.mean()
    sd = float(np.sqrt(np.mean(centred**2)))
    if sd <= 0:
        return 0.0, 0.0
    return float(np.mean(centred**3) / sd**3), float(np.mean(centred**4) / sd**4 - 3.0)


def _correction(residuals: np.ndarray, mse: float, method: str) -> float:
    """Back-transform factor for ``method``. Raises on an unknown method."""
    if method == BACKTRANSFORM_NONE:
        return 1.0
    if method == BACKTRANSFORM_BASKERVILLE:
        return math.exp(mse / 2.0)
    if method == BACKTRANSFORM_SMEARING:
        return float(np.mean(np.exp(residuals))) if len(residuals) else 1.0
    raise ValueError(f"unknown back_transform {method!r}; expected one of {BACKTRANSFORMS}")


@dataclass(frozen=True)
class GroupFit:
    """Immutable result of one functional-type log-log regression."""

    group: str
    intercept: float  # ln(a)
    slope: float  # exponent b
    correction_factor: float  # exp(sigma^2 / 2), >= 1
    r2: float
    mse: float  # residual variance (sigma^2) on the log scale
    n: int  # number of trees used in the fit


def _valid_mask(dbh: pd.Series, asw: pd.Series) -> pd.Series:
    """Rows usable for a log-log fit: finite, strictly positive DBH and ASW."""
    return (
        dbh.notna()
        & asw.notna()
        & (pd.to_numeric(dbh, errors="coerce") > 0)
        & (pd.to_numeric(asw, errors="coerce") > 0)
    )


def fit_loglog_by_group(
    df: pd.DataFrame,
    group_col: str,
    dbh_col: str,
    asw_col: str,
    min_n: int = 2,
) -> dict[str, GroupFit]:
    """Fit ``ln(ASW) ~ ln(DBH)`` separately within each group.

    Groups with fewer than ``min_n`` valid trees are skipped (not enough data
    to estimate a slope). Rows with non-positive or missing DBH/ASW are dropped.

    Returns
    -------
    dict[str, GroupFit]
        One entry per group that could be fit.
    """
    if min_n < 2:
        raise ValueError("min_n must be >= 2 to estimate a slope")

    fits: dict[str, GroupFit] = {}
    for group, sub in df.groupby(group_col):
        mask = _valid_mask(sub[dbh_col], sub[asw_col])
        sub = sub[mask]
        n = len(sub)
        if n < min_n:
            continue

        x = np.log(sub[dbh_col].to_numpy(dtype=float))
        y = np.log(sub[asw_col].to_numpy(dtype=float))

        slope, intercept = np.polyfit(x, y, 1)
        y_hat = intercept + slope * x
        residuals = y - y_hat
        sse = float(np.sum(residuals**2))

        # Residual variance: use the unbiased (n-2) denominator when possible.
        dof = n - 2
        mse = sse / dof if dof > 0 else 0.0

        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - sse / sst if sst > 0 else 1.0
        cf = math.exp(mse / 2.0)

        fits[str(group)] = GroupFit(
            group=str(group),
            intercept=float(intercept),
            slope=float(slope),
            correction_factor=cf,
            r2=r2,
            mse=mse,
            n=n,
        )
    return fits


def predict_asw(dbh: float, group: str, fits: dict[str, GroupFit]) -> float:
    """Predict bias-corrected sapwood area (cm²) for one tree.

    Raises
    ------
    KeyError
        If ``group`` has no fitted relationship.
    ValueError
        If ``dbh`` is not strictly positive.
    """
    if dbh is None or not np.isfinite(dbh) or dbh <= 0:
        raise ValueError(f"DBH must be a positive finite number, got {dbh!r}")
    fit = fits[group]  # KeyError if group not fitted
    ln_pred = fit.intercept + fit.slope * math.log(dbh)
    return math.exp(ln_pred) * fit.correction_factor


# ---------------------------------------------------------------------------
# Additive form: group as a predictor (one model, one slope, one sigma)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AdditiveFit:
    """Immutable result of ``ln(ASW) ~ ln(DBH) + group``.

    The group enters as a set of indicator variables, so it shifts the
    intercept but not the slope. ``offsets`` is stored as a tuple of pairs
    rather than a dict so the whole object is genuinely immutable and its
    ordering is deterministic in the JSON report.
    """

    reference: str  # group whose offset is pinned to 0
    intercept: float  # ln(a) at the reference group
    slope: float  # exponent b, shared by every group
    offsets: tuple[tuple[str, float], ...]  # group -> additive shift on the log scale
    correction_factor: float  # exp(sigma^2 / 2), one pooled value
    r2: float  # of the full model
    r2_without_group: float  # of the nested ln(DBH)-only model
    mse: float  # pooled residual variance on the log scale
    n: int  # total trees used
    n_by_group: tuple[tuple[str, int], ...]
    residual_sd_by_group: tuple[tuple[str, float], ...]  # homoscedasticity check

    # --- back-transform provenance -----------------------------------------
    # Both factors are always computed. `correction_factor` above holds the one
    # actually applied; these say what the alternative was and whether the
    # assumption behind the parametric choice survives contact with the data.
    back_transform: str = BACKTRANSFORM_BASKERVILLE
    baskerville_factor: float = 1.0  # exp(sigma^2/2), Gaussian-residual closed form
    smearing_factor: float = 1.0  # mean(exp(residual)), Duan (1983), distribution-free
    residual_skew: float = 0.0
    residual_excess_kurtosis: float = 0.0

    #: DBH range the fit actually saw, in cm. Predicting outside it is
    #: extrapolation of a power law and is flagged rather than silently allowed.
    dbh_min: float = 0.0
    dbh_max: float = float("inf")

    @property
    def residuals_look_normal(self) -> bool:
        """Whether ``exp(sigma^2/2)`` has a distributional leg to stand on."""
        return (
            abs(self.residual_skew) <= NONNORMAL_SKEW
            and abs(self.residual_excess_kurtosis) <= NONNORMAL_EXCESS_KURTOSIS
        )

    @property
    def groups(self) -> tuple[str, ...]:
        """Groups that were fitted, in report order."""
        return tuple(group for group, _ in self.offsets)

    def offset(self, group: str) -> float:
        """Additive log-scale shift for ``group``; ``KeyError`` if not fitted."""
        for name, value in self.offsets:
            if name == group:
                return value
        raise KeyError(group)


def _pick_reference(groups: list[str], counts: dict[str, int]) -> str:
    """Largest group wins (best-determined intercept); ties broken alphabetically.

    The choice is cosmetic: predictions are identical for any reference.
    """
    return sorted(groups, key=lambda g: (-counts[g], g))[0]


def _ols(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coefficients


def fit_loglog_with_group_offset(
    df: pd.DataFrame,
    group_col: str,
    dbh_col: str,
    asw_col: str,
    min_n: int = 2,
    reference: str | None = None,
    back_transform: str = BACKTRANSFORM_BASKERVILLE,
) -> AdditiveFit:
    """Fit ``ln(ASW) ~ ln(DBH) + group`` as a single model.

    Groups with fewer than ``min_n`` valid trees are excluded entirely rather
    than given an indicator estimated from almost no data; they are then absent
    from :attr:`AdditiveFit.groups` and cannot be predicted.

    ``back_transform`` selects which correction factor :func:`predict_asw_additive`
    applies; all of them are computed and reported regardless. See the module
    docstring -- on this corpus the parametric default and the distribution-free
    alternative differ by ~3%, in a direction the residual skew predicts.

    Raises
    ------
    ValueError
        If no group clears ``min_n``, if the design matrix is rank deficient
        (e.g. every tree shares one DBH, so no slope exists), or if
        ``back_transform`` is not one of :data:`BACKTRANSFORMS`.
    """
    if back_transform not in BACKTRANSFORMS:
        raise ValueError(f"unknown back_transform {back_transform!r}; expected one of {BACKTRANSFORMS}")
    if min_n < 1:
        raise ValueError("min_n must be >= 1")

    data = df[_valid_mask(df[dbh_col], df[asw_col])]
    sizes = data[group_col].astype(str).value_counts().to_dict()
    groups = sorted(name for name, size in sizes.items() if size >= min_n)
    if not groups:
        raise ValueError(
            f"No group in {group_col!r} has >= {min_n} trees with positive "
            f"{dbh_col} and {asw_col}; cannot fit the allometry."
        )

    data = data[data[group_col].astype(str).isin(groups)]
    x = np.log(data[dbh_col].to_numpy(dtype=float))
    y = np.log(data[asw_col].to_numpy(dtype=float))
    labels = data[group_col].astype(str).to_numpy()
    n = int(len(data))

    if reference is not None and reference not in groups:
        raise ValueError(f"reference {reference!r} is not among the fitted groups {groups}")
    ref = reference or _pick_reference(groups, sizes)

    # Indicator columns for every group except the reference (dummy coding).
    dummies = [name for name in groups if name != ref]
    design = np.column_stack([np.ones(n), x, *((labels == name).astype(float) for name in dummies)])
    n_params = design.shape[1]
    if np.linalg.matrix_rank(design) < n_params:
        raise ValueError(
            "Rank-deficient design for the additive allometry: ln(DBH) has no "
            "variation, or a group indicator is collinear with it. Check that "
            f"{dbh_col} varies within the fitted trees."
        )

    coefficients = _ols(design, y)
    residuals = y - design @ coefficients
    sse = float(np.sum(residuals**2))
    dof = n - n_params
    mse = sse / dof if dof > 0 else 0.0
    sst = float(np.sum((y - y.mean()) ** 2))

    # Nested ln(DBH)-only model: makes the marginal value of `group` a reported
    # number instead of something the reader has to take on trust.
    plain = np.column_stack([np.ones(n), x])
    sse_plain = float(np.sum((y - plain @ _ols(plain, y)) ** 2))

    offsets = {ref: 0.0}
    for name, value in zip(dummies, coefficients[2:]):  # noqa: B905 - Python 3.9 has no strict=
        offsets[name] = float(value)

    skew, excess_kurtosis = _moments(residuals)
    return AdditiveFit(
        reference=ref,
        intercept=float(coefficients[0]),
        slope=float(coefficients[1]),
        offsets=tuple((name, offsets[name]) for name in groups),
        correction_factor=_correction(residuals, mse, back_transform),
        r2=1.0 - sse / sst if sst > 0 else 1.0,
        r2_without_group=1.0 - sse_plain / sst if sst > 0 else 1.0,
        mse=mse,
        n=n,
        n_by_group=tuple((name, int(sizes[name])) for name in groups),
        residual_sd_by_group=tuple(
            (
                name,
                float(np.std(residuals[labels == name], ddof=1)) if int((labels == name).sum()) > 1 else 0.0,
            )
            for name in groups
        ),
        back_transform=back_transform,
        baskerville_factor=math.exp(mse / 2.0),
        smearing_factor=_correction(residuals, mse, BACKTRANSFORM_SMEARING),
        residual_skew=skew,
        residual_excess_kurtosis=excess_kurtosis,
        dbh_min=float(np.exp(x.min())),
        dbh_max=float(np.exp(x.max())),
    )


def predict_asw_additive(dbh: float, group: str, fit: AdditiveFit) -> float:
    """Predict bias-corrected sapwood area (cm²) from the additive model.

    Raises
    ------
    KeyError
        If ``group`` was not fitted.
    ValueError
        If ``dbh`` is not strictly positive.
    """
    if dbh is None or not np.isfinite(dbh) or dbh <= 0:
        raise ValueError(f"DBH must be a positive finite number, got {dbh!r}")
    ln_pred = fit.intercept + fit.slope * math.log(dbh) + fit.offset(group)
    return math.exp(ln_pred) * fit.correction_factor
