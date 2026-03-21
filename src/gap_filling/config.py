"""Configuration for the hierarchical gap-filling system."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_DEFAULT_SAP_LOOKUP = "outputs/statistics/gap_experiment/benchmark_results_full.csv"
_DEFAULT_ENV_LOOKUP = "outputs/statistics/gap_experiment/benchmark_results_full_env.csv"

ALL_GROUPS = ["A", "B", "C", "Ce", "D", "De"]


@dataclass(frozen=True)
class GapFillingConfig:
    """All tunables for the hierarchical gap-filling system."""

    metric: str = "r2_pooled"
    threshold: float = 0.9
    buffer_hours: int = 500
    min_data_ml: int = 500
    min_data_dl: int = 2000
    prefer_non_env_below_h: int = 6
    max_gap_hours: int = 72
    lookup_csv_sap: Path | str = _DEFAULT_SAP_LOOKUP
    lookup_csv_env: Path | str = _DEFAULT_ENV_LOOKUP
    time_scale: str = "hourly"

    def eligible_groups(self, column_length: int) -> list[str]:
        """Return eligible method groups given the column's valid data length."""
        if column_length < self.min_data_ml:
            return ["A", "B"]
        if column_length < self.min_data_dl:
            return ["A", "B", "C", "Ce"]
        return list(ALL_GROUPS)
