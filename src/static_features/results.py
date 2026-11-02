"""The contract every product meets: status codes, the result, its validation, and atomic parts."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from src.static_features.sites import load_sites

OK = "ok"
NEIGHBOUR = "neighbour"
NO_DATA = "no_data"
STATUSES = frozenset({OK, NEIGHBOUR, NO_DATA})

BY_YEAR_COLUMNS = ("site_name", "year", "column", "value", "status")
PART_FILES = frozenset({"values", "status", "coverage", "by_year", "provenance"})


class ExtractionFailure(RuntimeError):  # noqa: N818
    """A request or a product broke. Never a status: it names the site and product, and stops the run."""


@dataclass(frozen=True, eq=False)
class RunContext:
    sites: pd.DataFrame
    raw_root: Path
    output_root: Path
    project: str

    @classmethod
    def for_scale(cls, scale: str, project: str) -> RunContext:
        import path_config

        paths = path_config.get_default_paths(scale)
        return cls(
            sites=load_sites(paths.site_info_path, paths.metadata_dir),
            raw_root=paths.base_data_dir / "raw" / "grided",
            output_root=paths.extracted_data_dir / "static_1km" / scale,
            project=project,
        )

    @property
    def parts_root(self) -> Path:
        return self.output_root / "parts"


@dataclass(frozen=True, eq=False)
class ProductResult:
    """values and status are indexed by site_name with the same columns; coverage holds a subset of them."""

    product: str
    values: pd.DataFrame
    status: pd.DataFrame
    coverage: pd.DataFrame
    by_year: pd.DataFrame | None = None
    extras: dict[str, pd.DataFrame] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


def _check_index(product: str, name: str, frame: pd.DataFrame, expected: set[str]) -> None:
    repeated = sorted(frame.index[frame.index.duplicated()].unique())
    if repeated:
        raise ExtractionFailure(f"{product}: {name} repeats sites {repeated}")
    missing = sorted(expected - set(frame.index))
    if missing:
        raise ExtractionFailure(f"{product}: {name} is missing sites {missing}")
    unexpected = sorted(set(frame.index) - expected)
    if unexpected:
        raise ExtractionFailure(f"{product}: {name} has unexpected sites {unexpected}")


def _check_statuses(product: str, where: str, statuses: pd.Series) -> None:
    unknown = sorted(set(statuses.astype(str)) - STATUSES)
    if unknown:
        raise ExtractionFailure(f"{product}: {where}: unknown status {unknown}")


def _check_column(product: str, column: str, values: pd.Series, status: pd.Series) -> None:
    _check_statuses(product, f"column {column}", status)
    present = values.notna()
    has_value_status = status != NO_DATA
    empty = sorted(values.index[has_value_status & ~present])
    if empty:
        raise ExtractionFailure(f"{product}: column {column}: NaN with a value status at {empty}")
    filled = sorted(values.index[~has_value_status & present])
    if filled:
        raise ExtractionFailure(f"{product}: column {column}: value with status no_data at {filled}")
    if pd.api.types.is_numeric_dtype(values):
        infinite = sorted(values.index[np.isinf(values.astype(float))])
        if infinite:
            raise ExtractionFailure(f"{product}: column {column}: infinite value at {infinite}")


def _check_coverage(result: ProductResult) -> None:
    unknown = sorted(set(result.coverage.columns) - set(result.values.columns))
    if unknown:
        raise ExtractionFailure(f"{result.product}: coverage of unknown columns {unknown}")
    for column in result.coverage.columns:
        shares = pd.to_numeric(result.coverage[column], errors="coerce")
        outside = sorted(shares.index[~shares.between(0.0, 1.0)])  # NaN is outside
        if outside:
            raise ExtractionFailure(f"{result.product}: coverage of {column} outside [0, 1] at {outside}")


def _check_by_year(result: ProductResult, expected: set[str]) -> None:
    table = result.by_year
    if table is None:
        return
    absent = [column for column in BY_YEAR_COLUMNS if column not in table.columns]
    if absent:
        raise ExtractionFailure(f"{result.product}: by_year is missing columns {absent}")
    unexpected = sorted(set(table["site_name"]) - expected)
    if unexpected:
        raise ExtractionFailure(f"{result.product}: by_year has unexpected sites {unexpected}")
    _check_statuses(result.product, "by_year", table["status"])
    present = table["value"].notna()
    has_value_status = table["status"] != NO_DATA
    if (present != has_value_status).any():
        first = table[present != has_value_status].iloc[0]
        raise ExtractionFailure(
            f"{result.product}: by_year value and status disagree: "
            f"{first['site_name']} {first['year']} {first['column']}"
        )


def validate(result: ProductResult, site_names: Iterable[str]) -> None:
    """Stop rather than save a result whose numbers and reason codes disagree."""
    expected = set(site_names)
    for name, frame in (("values", result.values), ("status", result.status), ("coverage", result.coverage)):
        _check_index(result.product, name, frame, expected)
    different = sorted(set(result.values.columns) ^ set(result.status.columns))
    if different:
        raise ExtractionFailure(f"{result.product}: values and status differ in columns {different}")
    status = result.status.loc[result.values.index]
    for column in result.values.columns:
        _check_column(result.product, column, result.values[column], status[column])
    _check_coverage(result)
    _check_by_year(result, expected)
    reserved = sorted(PART_FILES & set(result.extras))
    if reserved:
        raise ExtractionFailure(f"{result.product}: extras named like a part file: {reserved}")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {value!r}")


def _write_files(result: ProductResult, site_hash: str, directory: Path) -> None:
    result.values.to_csv(directory / "values.csv", index_label="site_name")
    result.status.to_csv(directory / "status.csv", index_label="site_name")
    result.coverage.to_csv(directory / "coverage.csv", index_label="site_name")
    if result.by_year is not None:
        result.by_year.to_csv(directory / "by_year.csv", index=False)
    for name, table in result.extras.items():
        table.to_csv(directory / f"{name}.csv", index=False)
    with open(directory / "provenance.json", "w", encoding="utf-8") as handle:
        json.dump(result.provenance, handle, indent=2, sort_keys=True, default=_json_default)
    (directory / "site_hash.txt").write_text(site_hash, encoding="utf-8")


def _swap_into_place(temp: Path, final: Path) -> None:
    if not final.exists():
        temp.rename(final)
        return
    old = final.with_name(f".old_{final.name}_{os.getpid()}")
    final.rename(old)
    try:
        temp.rename(final)
    except OSError:
        old.rename(final)
        raise
    shutil.rmtree(old)


def write_part(result: ProductResult, parts_root: Path, site_hash: str) -> Path:
    """Save a part atomically: a job that dies leaves no part, and any earlier good part intact."""
    parts_root = Path(parts_root)
    parts_root.mkdir(parents=True, exist_ok=True)
    stale_name = re.compile(rf"\.tmp_{re.escape(result.product)}_\d+")
    for stale in parts_root.iterdir():
        if stale_name.fullmatch(stale.name):
            shutil.rmtree(stale)  # left by a job that was killed; one job per product at a time
    temp = parts_root / f".tmp_{result.product}_{os.getpid()}"
    final = parts_root / result.product
    temp.mkdir()
    try:
        _write_files(result, site_hash, temp)
        _swap_into_place(temp, final)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return final


def read_table(path: Path, **kwargs: Any) -> pd.DataFrame:
    # Only a blank is missing, so a literal "NA" stays a value; round_trip keeps floats exact.
    return pd.read_csv(
        path, float_precision="round_trip", keep_default_na=False, na_values=[""], low_memory=False, **kwargs
    )


def read_part(parts_root: Path, product: str) -> tuple[ProductResult, str]:
    directory = Path(parts_root) / product
    if not (directory / "site_hash.txt").exists():
        raise ExtractionFailure(f"{product}: no part at {directory}")
    by_year_path = directory / "by_year.csv"
    extras = {path.stem: read_table(path) for path in sorted(directory.glob("*.csv")) if path.stem not in PART_FILES}
    result = ProductResult(
        product=product,
        values=read_table(directory / "values.csv", index_col="site_name"),
        status=read_table(directory / "status.csv", index_col="site_name"),
        coverage=read_table(directory / "coverage.csv", index_col="site_name"),
        by_year=read_table(by_year_path) if by_year_path.exists() else None,
        extras=extras,
        provenance=json.loads((directory / "provenance.json").read_text(encoding="utf-8")),
    )
    return result, (directory / "site_hash.txt").read_text(encoding="utf-8")
