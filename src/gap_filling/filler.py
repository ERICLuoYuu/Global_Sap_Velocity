"""Orchestrator: detect gaps -> select method -> fill column."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.gap_filling.config import GapFillingConfig
from src.gap_filling.detector import GapDetector
from src.gap_filling.methods import METHODS
from src.gap_filling.selector import MethodSelector

logger = logging.getLogger(__name__)


class GapFiller:
    """Hierarchical gap filler for time series columns.

    Usage:
        filler = GapFiller(config, target="sap")
        filled = filler.fill_column(series, env_df=env_data)
        filled_df = filler.fill_dataframe(df, env_df=env_data)
    """

    def __init__(self, config: GapFillingConfig, target: str = "sap"):
        self.config = config
        self.target = target
        self.selector = MethodSelector(config, target=target)

    def fill_column(
        self,
        series: pd.Series,
        env_df: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Fill gaps in a single column. Returns new Series (original unchanged)."""
        gaps = GapDetector.detect(series)
        if not gaps:
            return series.copy()

        column_length = int(series.notna().sum())
        result = self.selector.select(
            column_length=column_length,
            env_available=env_df is not None,
        )

        filled = series.copy()

        # Step 1: Non-env fill
        if result.non_env_method and result.max_non_env_gap_h > 0:
            small_gaps = [g for g in gaps if g.size_hours <= result.max_non_env_gap_h]
            if small_gaps:
                filled = self._apply_method(
                    filled,
                    result.non_env_method,
                    env_df=None,
                    max_gap_h=result.max_non_env_gap_h,
                    original_series=series,
                )

        # Step 2: Env fill for larger gaps
        if result.env_method and result.max_env_gap_h > result.max_non_env_gap_h and env_df is not None:
            remaining_gaps = GapDetector.detect(filled)
            medium_gaps = [g for g in remaining_gaps if g.size_hours <= result.max_env_gap_h]
            if medium_gaps:
                filled = self._apply_method(
                    filled,
                    result.env_method,
                    env_df=env_df,
                    max_gap_h=result.max_env_gap_h,
                    original_series=series,
                )

        return filled

    def _apply_method(
        self,
        series: pd.Series,
        method_name: str,
        env_df: pd.DataFrame | None,
        max_gap_h: int,
        original_series: pd.Series,
    ) -> pd.Series:
        """Apply a single gap-filling method to qualifying gaps."""
        info = METHODS.get(method_name)
        if info is None:
            logger.warning("Method %s not found in registry", method_name)
            return series

        is_env = info.get("env", False)
        filled = series.copy()

        if "fill" in info:
            # Group A/B: stateless fill function
            filled = info["fill"](filled)
        elif "fit" in info and "predict" in info:
            # Group C/D/Ce/De: fit then predict
            fit_fn = info["fit"]
            predict_fn = info["predict"]
            model_name = info.get("model_name")

            fit_kw = {"env_df": env_df} if is_env else {}
            pred_kw = {"env_df": env_df} if is_env else {}
            if model_name:
                fit_kw["model_name"] = model_name
                pred_kw["model_name"] = model_name

            model = fit_fn(filled, **fit_kw)
            if model is not None:
                filled = predict_fn(filled, model, **pred_kw)

        # Mask back gaps that exceed max_gap_h (method may have filled them too)
        original_gaps = GapDetector.detect(original_series)
        for gap in original_gaps:
            if gap.size_hours > max_gap_h:
                mask = (filled.index >= gap.start) & (filled.index <= gap.end)
                filled[mask] = np.nan

        return filled.clip(lower=0)

    def fill_dataframe(
        self,
        df: pd.DataFrame,
        env_df: pd.DataFrame | None = None,
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """Fill gaps in all columns of a DataFrame.

        For n_jobs > 1, uses joblib for column-level parallelism.
        """
        if n_jobs <= 1:
            result = pd.DataFrame(index=df.index)
            for col in df.columns:
                result[col] = self.fill_column(df[col], env_df=env_df)
            return result

        from joblib import Parallel, delayed

        filled_cols = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self.fill_column)(df[col], env_df=env_df) for col in df.columns
        )
        return pd.DataFrame(
            {col: filled for col, filled in zip(df.columns, filled_cols)},
            index=df.index,
        )
