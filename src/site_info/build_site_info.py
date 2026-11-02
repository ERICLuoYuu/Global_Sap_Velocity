"""Build ``site_info.csv`` — the per-site index every gridded extractor runs from.

Every extractor in ``src/Extractors`` reads exactly one input describing *where*
and *when* to sample: ``PathConfig.site_info_path``, a five-column table of
``start_date, end_date, lat, lon, site_name``.  This module produces that table
for any scale directory under ``data/raw/<version>/<version>/csv/<scale>/``.

It replaces the hardcoded ``notebooks/extract_siteinfo.py`` (which could only
ever target ``sapwood``) and ``src.tools.extract_site_info`` (which read whole
sap-flow files into memory to take a min and a max, and whose "convert to UTC"
docstring described work it never did).

Two properties of the combined corpus drive the design:

* **Timestamp formats are not uniform across the corpus.**  Externally sourced
  sites write ISO-8601 with an explicit zone (``2009-11-19T03:00:00Z``) while
  internally processed sites write a naive stamp (``2015-04-22 22:00:00``) that
  is UTC by contract but carries no marker.  Parsing with ``utc=True`` localises
  the naive ones and converts the aware ones, so the output column is a single
  ``datetime64[ns, UTC]`` rather than a mixed-dtype object column that would
  serialise two different string formats under one header.

* **Site codes are prefixes of one another** (``CHE_DAV`` ⊂ ``CHE_DAV_SEE``,
  ``SWE_NOR`` ⊂ ``SWE_NOR_ST1_AF1``).  Sites are therefore discovered by
  stripping an exact filename suffix, never by a prefix glob.

Anything unexpected raises.  A site silently missing from ``site_info.csv`` is a
site silently missing from every downstream gridded feature, and that is not
visible until model training is already underway.

Usage
-----
    # every site in a scale
    python -m src.site_info.build_site_info --scale sapwood_combined

    # only a named subset, written somewhere else (delta extraction runs)
    python -m src.site_info.build_site_info \
        --scale sapwood_combined \
        --only new_sites.txt \
        --out data/raw/0.1.5/0.1.5/csv/sapwood_combined_new/site_info.csv
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_config import PathConfig  # noqa: E402

# The column order the extractors depend on.  extract_canopyheight.py does
# `pd.concat([df, results_df], axis=1)`, so this table's shape is load-bearing.
SITE_INFO_COLUMNS = ["start_date", "end_date", "lat", "lon", "site_name"]

SAPF_SUFFIX = "_sapf_data.csv"
SITE_MD_SUFFIX = "_site_md.csv"
TIMESTAMP_COLUMN = "TIMESTAMP"
LAT_COLUMN = "si_lat"
LON_COLUMN = "si_long"

# utf-8-sig so a byte-order mark left by a provider's spreadsheet export does not
# end up glued to the first header name.
ENCODING = "utf-8-sig"


class SiteInfoError(RuntimeError):
    """A site could not be described. Never downgraded to a skip."""


def discover_sites(csv_dir: Path) -> list[str]:
    """Return every site code in ``csv_dir``, sorted.

    A site is defined by the presence of ``<site>_sapf_data.csv``: sap flow is
    the measurement the whole pipeline exists to model, so a directory entry
    without one is not a site.
    """
    if not csv_dir.is_dir():
        raise SiteInfoError(f"scale directory does not exist: {csv_dir}")

    sites = sorted(path.name[: -len(SAPF_SUFFIX)] for path in csv_dir.iterdir() if path.name.endswith(SAPF_SUFFIX))
    if not sites:
        raise SiteInfoError(f"no *{SAPF_SUFFIX} files found in {csv_dir}")
    return sites


def read_coordinates(csv_dir: Path, site: str) -> tuple[float, float]:
    """Return ``(lat, lon)`` for ``site`` from its ``site_md`` file."""
    path = csv_dir / f"{site}{SITE_MD_SUFFIX}"
    if not path.exists():
        raise SiteInfoError(f"{site}: no site_md file at {path}")

    frame = pd.read_csv(path, encoding=ENCODING)
    if frame.empty:
        raise SiteInfoError(f"{site}: site_md has a header but no rows")

    for column in (LAT_COLUMN, LON_COLUMN):
        if column not in frame.columns:
            raise SiteInfoError(f"{site}: site_md has no {column!r} column (columns: {list(frame.columns)})")

    try:
        lat = float(frame[LAT_COLUMN].iloc[0])
        lon = float(frame[LON_COLUMN].iloc[0])
    except (TypeError, ValueError) as exc:
        raise SiteInfoError(
            f"{site}: coordinates are not numeric — "
            f"{LAT_COLUMN}={frame[LAT_COLUMN].iloc[0]!r}, "
            f"{LON_COLUMN}={frame[LON_COLUMN].iloc[0]!r}"
        ) from exc

    return lat, lon


def read_time_span(csv_dir: Path, site: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the ``(first, last)`` sap-flow timestamp for ``site``, in UTC.

    Only the timestamp column is read: the sap-flow tables run to gigabytes
    across the corpus and every other column is irrelevant to a min and a max.
    """
    path = csv_dir / f"{site}{SAPF_SUFFIX}"
    if not path.exists():
        raise SiteInfoError(f"{site}: no sapf_data file at {path}")

    try:
        raw = pd.read_csv(path, usecols=[TIMESTAMP_COLUMN], encoding=ENCODING)
    except ValueError as exc:
        # pandas raises ValueError when a name in usecols is absent.
        header = pd.read_csv(path, nrows=0, encoding=ENCODING).columns.tolist()
        raise SiteInfoError(f"{site}: sapf_data has no {TIMESTAMP_COLUMN!r} column (columns: {header[:6]})") from exc

    if raw.empty:
        raise SiteInfoError(f"{site}: sapf_data has a header but no rows")

    # utc=True is what reconciles the corpus's two timestamp dialects: naive
    # stamps are localised to UTC (their documented meaning) and aware stamps
    # are converted to it.  Without it this column comes back as dtype object
    # and the two dialects survive all the way into the written CSV.
    stamps = pd.to_datetime(raw[TIMESTAMP_COLUMN], utc=True, errors="coerce")

    unparsed = int(stamps.isna().sum())
    if unparsed:
        sample = raw.loc[stamps.isna(), TIMESTAMP_COLUMN].head(3).tolist()
        raise SiteInfoError(f"{site}: {unparsed} of {len(stamps)} timestamps did not parse (e.g. {sample})")

    return stamps.min(), stamps.max()


def build_site_info(csv_dir: Path, only: Iterable[str] | None = None) -> pd.DataFrame:
    """Build the site_info table for ``csv_dir``.

    Args:
        csv_dir: a scale directory, e.g. ``.../csv/sapwood_combined``.
        only: restrict to these site codes.  Every one of them must exist in
            ``csv_dir`` — a requested site that is absent is an error, not an
            empty row, because the caller asked for it by name.
    """
    available = discover_sites(csv_dir)

    if only is None:
        sites = available
    else:
        requested = list(dict.fromkeys(only))  # de-duplicate, keep order
        missing = sorted(set(requested) - set(available))
        if missing:
            raise SiteInfoError(f"{len(missing)} requested site(s) absent from {csv_dir}: {missing}")
        sites = sorted(requested)

    records = []
    for site in sites:
        lat, lon = read_coordinates(csv_dir, site)
        start, end = read_time_span(csv_dir, site)
        records.append(
            {
                "start_date": start,
                "end_date": end,
                "lat": lat,
                "lon": lon,
                "site_name": site,
            }
        )

    frame = pd.DataFrame.from_records(records, columns=SITE_INFO_COLUMNS)
    # from_records on an empty list yields object columns; the guards in
    # discover_sites make that unreachable, but be explicit about the dtypes
    # the validator and the extractors expect.
    frame["start_date"] = pd.to_datetime(frame["start_date"], utc=True)
    frame["end_date"] = pd.to_datetime(frame["end_date"], utc=True)
    return frame


def validate(frame: pd.DataFrame) -> list[str]:
    """Return a list of problems with ``frame``; empty means it is usable.

    These are the failures that produce *plausible but wrong* gridded features
    rather than a crash: a site sampled at the wrong pixel, or over the wrong
    years, looks exactly like a site sampled correctly.
    """
    problems: list[str] = []

    missing_columns = [c for c in SITE_INFO_COLUMNS if c not in frame.columns]
    if missing_columns:
        return [f"missing columns: {missing_columns}"]

    duplicated = frame.loc[frame["site_name"].duplicated(keep=False), "site_name"]
    if not duplicated.empty:
        problems.append(f"duplicate site_name: {sorted(duplicated.unique())}")

    for column, low, high in (("lat", -90.0, 90.0), ("lon", -180.0, 180.0)):
        values = pd.to_numeric(frame[column], errors="coerce")
        bad = frame.loc[values.isna(), "site_name"].tolist()
        if bad:
            problems.append(f"non-numeric {column}: {bad}")
        out_of_range = frame.loc[values.notna() & ((values < low) | (values > high)), "site_name"].tolist()
        if out_of_range:
            problems.append(f"{column} outside [{low}, {high}]: {out_of_range}")

    # (0, 0) is in the Gulf of Guinea.  It is the value a failed coordinate
    # lookup produces, and no forest sits there.
    null_island = frame.loc[
        np.isclose(pd.to_numeric(frame["lat"], errors="coerce").fillna(1), 0)
        & np.isclose(pd.to_numeric(frame["lon"], errors="coerce").fillna(1), 0),
        "site_name",
    ].tolist()
    if null_island:
        problems.append(f"coordinates at (0, 0): {null_island}")

    for column in ("start_date", "end_date"):
        unparsed = frame.loc[pd.isna(frame[column]), "site_name"].tolist()
        if unparsed:
            problems.append(f"missing {column}: {unparsed}")

    both_present = frame["start_date"].notna() & frame["end_date"].notna()
    reversed_span = frame.loc[both_present & (frame["start_date"] > frame["end_date"]), "site_name"].tolist()
    if reversed_span:
        problems.append(f"start_date after end_date: {reversed_span}")

    return problems


def write_site_info(frame: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return out_path


def _read_site_list(path: Path) -> list[str]:
    """Read one site code per line; blank lines and ``#`` comments ignored."""
    lines = path.read_text(encoding=ENCODING).splitlines()
    return [stripped for stripped in (line.strip() for line in lines) if stripped and not stripped.startswith("#")]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build site_info.csv for a scale directory.",
    )
    parser.add_argument(
        "--scale",
        default=None,
        help="Scale directory under csv/ (default: $ANALYSIS_SCALE, else sapwood).",
    )
    parser.add_argument(
        "--csv-dir",
        default=None,
        help="Override the scale directory outright (bypasses PathConfig).",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="File of site codes, one per line; restrict output to these.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path (default: <csv-dir>/site_info.csv).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate, but write nothing.",
    )
    args = parser.parse_args(argv)

    paths = PathConfig(scale=args.scale)
    csv_dir = Path(args.csv_dir) if args.csv_dir else paths.raw_csv_dir
    out_path = Path(args.out) if args.out else csv_dir / "site_info.csv"
    only = _read_site_list(Path(args.only)) if args.only else None

    print(f"scale     : {paths.scale}")
    print(f"csv dir   : {csv_dir}")
    print(f"output    : {out_path}")
    if only is not None:
        print(f"restricted: {len(only)} site(s) from {args.only}")

    frame = build_site_info(csv_dir, only=only)
    print(f"built     : {len(frame)} site(s)")

    problems = validate(frame)
    if problems:
        print("\nVALIDATION FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("validation: OK")

    print(f"\nspan      : {frame['start_date'].min()} .. {frame['end_date'].max()}")
    print(f"lat range : {frame['lat'].min():.4f} .. {frame['lat'].max():.4f}")
    print(f"lon range : {frame['lon'].min():.4f} .. {frame['lon'].max():.4f}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    write_site_info(frame, out_path)
    print(f"\nwrote {len(frame)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
