"""AOA configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AOAConfig:
    """Immutable configuration for AOA computation.

    Args:
        model_type: Model type identifier (e.g. "xgb").
        run_id: Model run identifier.
        time_scale: "daily" or "hourly".
        aoa_reference_path: Path to the AOA reference NPZ file.
        input_dir: ERA5 preprocessed parent directory.
        model_config_path: Path to FINAL_config JSON.
        output_dir: Base output directory for AOA results.
        iqr_multiplier: IQR multiplier for threshold (default 1.5).
        batch_size: Rows per KD-tree batch query (default 500_000).
        n_jobs: Parallelism (-1 = all cores).
        save_per_timestamp: Save per-timestamp parquets. None = auto
            (True for daily, False for hourly).
        map_format: "geotiff" or "netcdf".
        years: Years to process (empty = all discovered).
        months: Months to process (default 1-12).
    """

    model_type: str
    run_id: str
    time_scale: str
    aoa_reference_path: Path
    input_dir: Path
    model_config_path: Path
    output_dir: Path
    iqr_multiplier: float = 1.5
    batch_size: int = 500_000
    n_jobs: int = -1
    save_per_timestamp: bool | None = None
    map_format: str = "geotiff"
    years: tuple[int, ...] = ()
    months: tuple[int, ...] = tuple(range(1, 13))

    def __post_init__(self) -> None:
        if self.time_scale not in ("daily", "hourly"):
            raise ValueError(f"time_scale must be 'daily' or 'hourly', got '{self.time_scale}'")
        if self.iqr_multiplier <= 0:
            raise ValueError(f"iqr_multiplier must be > 0, got {self.iqr_multiplier}")
        if self.map_format not in ("geotiff", "netcdf"):
            raise ValueError(f"map_format must be 'geotiff' or 'netcdf', got '{self.map_format}'")
        if self.save_per_timestamp is None:
            object.__setattr__(self, "save_per_timestamp", self.time_scale == "daily")
