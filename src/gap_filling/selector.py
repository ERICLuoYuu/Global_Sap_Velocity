"""Lookup table query and hierarchical method selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.gap_filling.config import GapFillingConfig


@dataclass
class SelectionResult:
    """Result of hierarchical method selection for one column."""

    non_env_method: str | None
    max_non_env_gap_h: int
    env_method: str | None
    max_env_gap_h: int
    eligible_groups: list[str]
    lookup_metrics: pd.DataFrame


class MethodSelector:
    """Queries benchmark results to select the best gap-filling method.

    Loads benchmark_results_full.csv once, pre-groups sufficient statistics.
    Per-column queries filter by seg_length, compute pooled metrics, apply
    data volume guards, and select methods maximizing gap size coverage.
    """

    def __init__(self, config: GapFillingConfig, target: str = "sap"):
        csv_path = config.lookup_csv_sap if target == "sap" else config.lookup_csv_env
        self._config = config
        self._raw = pd.read_csv(csv_path)
        # Filter to requested time_scale
        if "time_scale" in self._raw.columns:
            self._raw = self._raw[self._raw["time_scale"] == config.time_scale]
        # Pre-group: sum sufficient stats by (method, gap_size, seg_length)
        group_cols = ["method", "gap_size", "seg_length"]
        if "group" in self._raw.columns:
            group_cols = ["method", "group", "gap_size", "seg_length"]
        self._grouped = (
            self._raw.groupby(group_cols)
            .agg(
                n_points=("n_points", "sum"),
                ss_res=("ss_res", "sum"),
                sum_abs_err=("sum_abs_err", "sum"),
                sum_true=("sum_true", "sum"),
                sum_true_sq=("sum_true_sq", "sum"),
            )
            .reset_index()
        )
        # Infer group from method name if not in CSV
        if "group" not in self._grouped.columns:
            self._grouped["group"] = self._grouped["method"].str.split("_").str[0]
        self._cache: dict[tuple, SelectionResult] = {}

    def _compute_pooled_metrics(self, column_length: int) -> pd.DataFrame:
        """Compute pooled R2/RMSE/MAE for all (method, gap_size) combos
        using only rows where seg_length <= column_length + buffer."""
        buf = self._config.buffer_hours
        filtered = self._grouped[self._grouped["seg_length"] <= column_length + buf]
        if filtered.empty:
            return pd.DataFrame()

        agg = (
            filtered.groupby(["method", "group", "gap_size"])
            .agg(
                N=("n_points", "sum"),
                total_ss_res=("ss_res", "sum"),
                total_abs_err=("sum_abs_err", "sum"),
                total_sum_true=("sum_true", "sum"),
                total_sum_true_sq=("sum_true_sq", "sum"),
            )
            .reset_index()
        )
        # Pooled metrics
        agg["pooled_mean"] = agg["total_sum_true"] / agg["N"]
        agg["pooled_ss_tot"] = agg["total_sum_true_sq"] - agg["N"] * agg["pooled_mean"] ** 2
        # Numerical stability guard
        agg["pooled_ss_tot"] = agg["pooled_ss_tot"].clip(lower=0.0)

        mask_valid = agg["pooled_ss_tot"] > 1e-12
        agg["r2_pooled"] = np.where(
            mask_valid,
            1.0 - agg["total_ss_res"] / agg["pooled_ss_tot"],
            np.where(agg["total_ss_res"] < 1e-12, 1.0, 0.0),
        )
        agg["rmse_pooled"] = np.sqrt(agg["total_ss_res"] / agg["N"])
        agg["mae_pooled"] = agg["total_abs_err"] / agg["N"]
        return agg

    def select(self, column_length: int, env_available: bool) -> SelectionResult:
        """Select best non-env and env methods for a column."""
        cache_key = (round(column_length / 100) * 100, env_available)
        if cache_key in self._cache:
            return self._cache[cache_key]

        eligible = self._config.eligible_groups(column_length)
        metrics = self._compute_pooled_metrics(column_length)

        if metrics.empty:
            result = SelectionResult(None, 0, None, 0, eligible, metrics)
            self._cache[cache_key] = result
            return result

        metric_col = self._config.metric
        thr = self._config.threshold

        # Non-env selection: groups A, B, C, D (exclude Ce, De)
        non_env_groups = [g for g in eligible if g in ("A", "B", "C", "D")]
        non_env_df = metrics[(metrics["group"].isin(non_env_groups)) & (metrics[metric_col] >= thr)]
        non_env_method = None
        max_non_env_gap = 0
        if not non_env_df.empty:
            # Per method: find max gap_size meeting threshold
            best_per_method = non_env_df.groupby("method")["gap_size"].max()
            # Pick method with largest max gap_size
            non_env_method = best_per_method.idxmax()
            max_non_env_gap = int(best_per_method.max())

        # Env selection (only for gaps larger than non-env can handle)
        env_method = None
        max_env_gap = max_non_env_gap
        if env_available and any(g in eligible for g in ("Ce", "De")):
            env_groups = [g for g in eligible if g in ("Ce", "De")]
            env_df = metrics[
                (metrics["group"].isin(env_groups))
                & (metrics[metric_col] >= thr)
                & (metrics["gap_size"] > max_non_env_gap)
            ]
            if not env_df.empty:
                best_env = env_df.groupby("method")["gap_size"].max()
                env_method = best_env.idxmax()
                max_env_gap = int(best_env.max())

        # Enforce hard cap on maximum gap size
        cap = self._config.max_gap_hours
        if cap > 0:
            max_non_env_gap = min(max_non_env_gap, cap)
            max_env_gap = min(max_env_gap, cap)

        result = SelectionResult(
            non_env_method=non_env_method,
            max_non_env_gap_h=max_non_env_gap,
            env_method=env_method,
            max_env_gap_h=max_env_gap,
            eligible_groups=eligible,
            lookup_metrics=metrics,
        )
        self._cache[cache_key] = result
        return result
