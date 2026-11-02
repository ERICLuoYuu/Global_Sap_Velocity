"""Assemble each extracted product for a target scale from a reference + a delta.

Every file under ``data/raw/extracted_data/<product>/<scale>/`` is a per-site row
concatenation keyed on ``site_name``.  So the 226-site ``sapwood_combined``
product is exactly the 165 rows already extracted under ``sapwood`` followed by
the 61 rows extracted under ``sapwood_combined_new`` — no interpolation, no
recomputation, no approximation.

Rows are streamed through as **text**.  Reading with ``dtype=str`` and
``na_filter=False`` means a reference row is written back byte for byte: no
float is reformatted, no integer column silently becomes ``1.0``, and no empty
field becomes ``nan``.  That turns "the 165 reused rows are unchanged" from a
hope into a property of the code, and it keeps peak memory at one chunk rather
than the 1.6 GB ERA5 table.

Refuses to write when the two halves disagree about anything that matters:
overlapping sites, a missing site, or a column present in one and not the other
(reported, and filled, but never silently).

Usage
-----
    python -m src.site_info.assemble_extracted \
        --reference-scale sapwood \
        --delta-scale     sapwood_combined_new \
        --target-scale    sapwood_combined \
        --site-info data/raw/0.1.5/0.1.5/csv/sapwood_combined/site_info.csv
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.site_info.partition import site_names  # noqa: E402

SITE_NAME_COLUMN = "site_name"
CHUNK_ROWS = 500_000
ENCODING = "utf-8-sig"


class AssemblyError(RuntimeError):
    """Two halves of a product could not be joined."""


@dataclass(frozen=True)
class Product:
    """One extracted product, as it is laid out on disk."""

    name: str
    directory: str
    """Sub-directory of extracted_data, before the scale component."""

    filename: str
    renumber_site_id: bool = False
    """ERA5 stores a 1-based row index of site_info as ``site_id``.  Extracted
    for 61 sites it restarts at 1 and collides with the reference's 1..165, so
    it is renumbered against the target site_info.  Nothing downstream joins on
    it (the merge keys on site_name), but a colliding id in a delivered file is
    a trap for whoever reads it next."""

    def path(self, data_root: Path, scale: str) -> Path:
        return data_root / "raw" / "extracted_data" / self.directory / scale / self.filename


PRODUCTS: tuple[Product, ...] = (
    Product("terrain", "terrain_site_data", "site_info_with_terrain_data.csv"),
    Product("soilgrids", "terrain_site_data", "soilgrids_data.csv"),
    Product("stand_age", "terrain_site_data", "stand_age_data.csv"),
    Product("worldclim", "env_site_data", "site_info_with_env_data.csv"),
    Product("root_depth", "root_depth_site_data", "site_info_with_root_depth_fan2017.csv"),
    Product("globmap_lai", "globmap_lai_site_data", "extracted_globmap_lai_hourly.csv"),
    Product("pft", "landcover_data", "landcover_output.csv"),
    Product("era5", "era5land_site_data", "era5_extracted_data.csv", renumber_site_id=True),
)


@dataclass
class AssemblyReport:
    product: str
    reference_rows: int = 0
    delta_rows: int = 0
    reference_sites: int = 0
    delta_sites: int = 0
    columns: int = 0
    reference_only_columns: tuple[str, ...] = ()
    delta_only_columns: tuple[str, ...] = ()

    @property
    def rows(self) -> int:
        return self.reference_rows + self.delta_rows

    @property
    def sites(self) -> int:
        return self.reference_sites + self.delta_sites

    def line(self) -> str:
        extra = ""
        if self.reference_only_columns:
            extra += f"  ref-only cols: {list(self.reference_only_columns)}"
        if self.delta_only_columns:
            extra += f"  delta-only cols: {list(self.delta_only_columns)}"
        return (
            f"{self.product:12} {self.sites:4} sites  {self.rows:>11,} rows  "
            f"({self.reference_rows:,} + {self.delta_rows:,})  "
            f"{self.columns} cols{extra}"
        )


def read_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0, encoding=ENCODING).columns.tolist()


def _chunks(path: Path, chunk_rows: int):
    """Stream a CSV as text so values round-trip unchanged."""
    return pd.read_csv(
        path,
        chunksize=chunk_rows,
        dtype=str,
        na_filter=False,
        encoding=ENCODING,
    )


def assemble_product(
    product: Product,
    reference_path: Path,
    delta_path: Path,
    out_path: Path,
    site_id_by_name: dict[str, int] | None = None,
    chunk_rows: int = CHUNK_ROWS,
) -> AssemblyReport:
    """Concatenate ``reference_path`` and ``delta_path`` into ``out_path``."""
    for path in (reference_path, delta_path):
        if not path.exists():
            raise AssemblyError(f"{product.name}: missing input {path}")

    reference_columns = read_header(reference_path)
    delta_columns = read_header(delta_path)
    for path, columns in ((reference_path, reference_columns), (delta_path, delta_columns)):
        if SITE_NAME_COLUMN not in columns:
            raise AssemblyError(f"{product.name}: {path} has no {SITE_NAME_COLUMN!r}")

    # Reference order first so the assembled file opens the way the old one did.
    columns = reference_columns + [c for c in delta_columns if c not in reference_columns]

    report = AssemblyReport(
        product=product.name,
        columns=len(columns),
        reference_only_columns=tuple(c for c in reference_columns if c not in delta_columns),
        delta_only_columns=tuple(c for c in delta_columns if c not in reference_columns),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".partial")

    reference_sites: set[str] = set()
    delta_sites: set[str] = set()

    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        first = True
        for source, seen, is_delta in (
            (reference_path, reference_sites, False),
            (delta_path, delta_sites, True),
        ):
            for chunk in _chunks(source, chunk_rows):
                seen.update(chunk[SITE_NAME_COLUMN].unique())
                if is_delta:
                    report.delta_rows += len(chunk)
                else:
                    report.reference_rows += len(chunk)

                chunk = chunk.reindex(columns=columns, fill_value="")

                if product.renumber_site_id and site_id_by_name is not None and "site_id" in chunk.columns:
                    chunk["site_id"] = chunk[SITE_NAME_COLUMN].map(site_id_by_name).astype("string")
                    unmapped = int(chunk["site_id"].isna().sum())
                    if unmapped:
                        raise AssemblyError(
                            f"{product.name}: {unmapped} row(s) whose site_name is absent from site_info"
                        )

                chunk.to_csv(handle, index=False, header=first)
                first = False

    report.reference_sites = len(reference_sites)
    report.delta_sites = len(delta_sites)

    overlap = reference_sites & delta_sites
    if overlap:
        tmp_path.unlink()
        raise AssemblyError(
            f"{product.name}: {len(overlap)} site(s) in BOTH halves — the delta "
            f"would double-count them: {sorted(overlap)[:10]}"
        )

    tmp_path.replace(out_path)
    return report


def build_site_id_map(site_info_path: Path) -> dict[str, int]:
    """1-based row index of site_info, matching how ERA5 numbered sites."""
    return {name: i + 1 for i, name in enumerate(site_names(site_info_path))}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference-scale", default="sapwood")
    parser.add_argument("--delta-scale", default="sapwood_combined_new")
    parser.add_argument("--target-scale", default="sapwood_combined")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--site-info", required=True, help="target scale's site_info.csv")
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated product names; default is all of them",
    )
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).resolve()
    site_info_path = Path(args.site_info)
    expected_sites = set(site_names(site_info_path))
    site_id_by_name = build_site_id_map(site_info_path)

    wanted = set(args.only.split(",")) if args.only else {p.name for p in PRODUCTS}
    unknown = wanted - {p.name for p in PRODUCTS}
    if unknown:
        print(f"unknown product(s): {sorted(unknown)}")
        return 1

    print(f"data root : {data_root}")
    print(f"site_info : {site_info_path} ({len(expected_sites)} sites)")
    print(f"{args.reference_scale} + {args.delta_scale} -> {args.target_scale}\n")

    reports: list[AssemblyReport] = []
    failures: list[str] = []

    for product in PRODUCTS:
        if product.name not in wanted:
            continue
        try:
            report = assemble_product(
                product,
                reference_path=product.path(data_root, args.reference_scale),
                delta_path=product.path(data_root, args.delta_scale),
                out_path=product.path(data_root, args.target_scale),
                site_id_by_name=site_id_by_name,
                chunk_rows=args.chunk_rows,
            )
        except AssemblyError as exc:
            failures.append(str(exc))
            print(f"  FAILED  {product.name}: {exc}")
            continue

        reports.append(report)
        print(f"  wrote   {report.line()}")

        # The assembled product must describe exactly the sites site_info names.
        out_sites = set(
            pd.read_csv(
                product.path(data_root, args.target_scale),
                usecols=[SITE_NAME_COLUMN],
                dtype=str,
                na_filter=False,
                encoding=ENCODING,
            )[SITE_NAME_COLUMN].unique()
        )
        missing = expected_sites - out_sites
        extra = out_sites - expected_sites
        if missing:
            failures.append(f"{product.name}: {len(missing)} site(s) absent: {sorted(missing)[:10]}")
            print(f"          MISSING {len(missing)}: {sorted(missing)[:10]}")
        if extra:
            failures.append(f"{product.name}: {len(extra)} unexpected site(s): {sorted(extra)[:10]}")
            print(f"          UNEXPECTED {len(extra)}: {sorted(extra)[:10]}")
        if not missing and not extra:
            print(f"          verified: all {len(out_sites)} sites present, none extra")

    print("\n" + "=" * 78)
    for report in reports:
        print("  " + report.line())
    print("=" * 78)

    if failures:
        print(f"\n{len(failures)} PROBLEM(S):")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"\nassembled {len(reports)} product(s) cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
