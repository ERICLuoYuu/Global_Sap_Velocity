"""Add the derived ``ext_rad`` column to corpus env files that lack it.

202 of the 226 sites in ``csv/sapwood_combined`` ship ``ext_rad``; the 24
internal ones do not, and a missing feature column makes both trainers skip the
whole site. This backfills the column for exactly those 24, using the formula
validated against the 202 that already carry it.

**The column is appended textually, not written by pandas.** Round-tripping a
frame through ``read_csv``/``to_csv`` reformats every other column -- integers
become ``0.0``, float repr changes, ``NA`` sentinels move -- and that would be a
silent, corpus-wide data change dressed up as adding one column. This project
has already been bitten once by exactly that (an unconditional assignment
upcasting int64 to float64 and writing ``0.0`` where ``0`` had been). Streaming
the original bytes and appending one field per line makes the diff provably
confined to the new column.

Verification does not stop at "the column is present". For the internal sites
there is no reference ``ext_rad`` to compare against, so correctness is
established indirectly against measurements the site *does* carry:

* ``sw_in <= ext_rad`` -- the atmosphere attenuates; the surface cannot receive
  more shortwave than arrives at the top of it. A wrong scale factor or a
  hemisphere error breaks this immediately.
* peak-hour alignment -- the daily maximum of measured ``sw_in`` must land at
  the same hour as the daily maximum of computed ``ext_rad``. This is what
  catches a clock error, which every magnitude check would sail straight past.

Usage
-----
    python backfill_ext_rad.py --dry-run      # report, touch nothing
    python backfill_ext_rad.py --apply        # back up, write, verify
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Anchored on the working directory, not on __file__: this script lives in
# .claude/plan/ on the cluster, so parents[1] would be `.claude`. Run it from the
# repository root.
sys.path.insert(0, str(Path.cwd()))

from src.env_derive.ext_rad import extraterrestrial_radiation  # noqa: E402

DEFAULT_CORPUS = Path("/scratch/tmp/yluo2/Global_Sap_Velocity/data/raw/0.1.5/0.1.5/csv/sapwood_combined")


def header_fields(path: Path) -> list:
    with path.open() as handle:
        return handle.readline().rstrip("\n").rstrip("\r").split(",")


def site_coordinates(corpus: Path, site: str):
    md_path = corpus / f"{site}_site_md.csv"
    if not md_path.exists():
        return None
    md = pd.read_csv(md_path)
    if md.empty:
        return None
    row = md.iloc[0]

    def num(*names):
        for name in names:
            if name in md.columns:
                value = pd.to_numeric(pd.Series([row[name]]), errors="coerce").iloc[0]
                if pd.notna(value):
                    return float(value)
        return None

    lat = num("si_lat", "latitude", "lat")
    lon = num("si_long", "si_lon", "longitude", "lon")
    elev = num("si_elev", "elevation", "elev")
    if lat is None or lon is None:
        return None
    return lat, lon, 0.0 if elev is None else elev


def format_value(value: float) -> str:
    """Render one ext_rad cell.

    Empty string for a missing value: that is what the internal files already
    use for absent data (``0,,2``), and inventing ``NA`` here would introduce a
    second missing-value dialect into files that have one.
    """
    if not np.isfinite(value):
        return ""
    return f"{value:.6g}"


def append_column(source: Path, destination: Path, values: np.ndarray, column: str = "ext_rad") -> int:
    """Stream *source* to *destination*, appending one field per data row."""
    written = 0
    with source.open("r", newline="") as src, destination.open("w", newline="") as dst:
        header = src.readline()
        if not header:
            raise ValueError(f"{source} is empty")
        newline = "\r\n" if header.endswith("\r\n") else "\n"
        dst.write(header.rstrip("\r\n") + "," + column + newline)
        for i, line in enumerate(src):
            stripped = line.rstrip("\r\n")
            if not stripped:
                continue
            if i >= len(values):
                raise ValueError(f"{source}: more data rows than computed values at row {i}")
            dst.write(stripped + "," + format_value(values[i]) + newline)
            written += 1
    if written != len(values):
        raise ValueError(f"{source}: wrote {written} rows but computed {len(values)} values")
    return written


def verify(path: Path) -> list:
    """Physical checks on the written file. Returns a list of problem strings."""
    problems = []
    usecols = ["ext_rad"]
    available = header_fields(path)
    for optional in ("sw_in", "TIMESTAMP"):
        if optional in available:
            usecols.append(optional)
    frame = pd.read_csv(path, usecols=usecols)
    ext = pd.to_numeric(frame["ext_rad"], errors="coerce")

    if ext.isna().all():
        problems.append("ext_rad is entirely NaN")
        return problems
    if (ext.dropna() < 0).any():
        problems.append(f"{int((ext < 0).sum())} negative ext_rad values")

    zero_fraction = float((ext.dropna() == 0).mean())
    if not 0.10 <= zero_fraction <= 0.75:
        problems.append(f"implausible night fraction {zero_fraction:.3f}")

    if "sw_in" in frame.columns and "TIMESTAMP" in frame.columns:
        sw = pd.to_numeric(frame["sw_in"], errors="coerce")
        both = sw.notna() & ext.notna()
        if both.sum() > 100:
            # 20 W/m2 of slack absorbs sensor offset and the instantaneous-vs-
            # interval mismatch without tolerating a real scale error.
            violations = int((sw[both] > ext[both] + 20.0).sum())
            rate = violations / int(both.sum())
            if rate > 0.02:
                problems.append(f"sw_in exceeds ext_rad on {rate:.1%} of rows ({violations})")

            stamps = pd.to_datetime(frame.loc[both, "TIMESTAMP"], errors="coerce", utc=True)
            daily = pd.DataFrame(
                {"day": stamps.dt.date, "hour": stamps.dt.hour, "sw": sw[both].to_numpy(), "ext": ext[both].to_numpy()}
            )
            # Only days with real daylight can speak to phase.
            daily = daily[daily.groupby("day")["ext"].transform("max") > 100]
            if len(daily) > 100:
                peak_sw = daily.loc[daily.groupby("day")["sw"].idxmax()].set_index("day")["hour"]
                peak_ext = daily.loc[daily.groupby("day")["ext"].idxmax()].set_index("day")["hour"]
                offset = (peak_sw - peak_ext).dropna()
                if len(offset) > 20:
                    median_offset = float(offset.median())
                    if abs(median_offset) > 1.5:
                        problems.append(f"sw_in peaks {median_offset:+.1f} h from ext_rad -- clock error")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if args.dry_run == args.apply:
        print("FATAL: pass exactly one of --dry-run / --apply")
        return 2
    if not args.corpus.is_dir():
        print(f"FATAL: corpus not found at {args.corpus}")
        return 2

    env_files = sorted(args.corpus.glob("*_env_data.csv"))
    todo, have = [], []
    for path in env_files:
        (have if "ext_rad" in header_fields(path) else todo).append(path)

    print(f"corpus: {args.corpus}")
    print(f"  env files with ext_rad   : {len(have)}")
    print(f"  env files needing ext_rad: {len(todo)}")
    if not todo:
        print("nothing to do")
        return 0

    backup_dir = args.backup_dir or (args.corpus.parent / "sapwood_combined_ext_rad_backup")
    if args.apply:
        backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"  backups -> {backup_dir}")

    failures, done, skipped = 0, 0, 0
    for path in todo:
        site = path.name.replace("_env_data.csv", "")
        coords = site_coordinates(args.corpus, site)
        if coords is None:
            print(f"  {site:<12} SKIP: no usable coordinates in site_md")
            skipped += 1
            continue
        lat, lon, elev = coords

        stamps_frame = pd.read_csv(path, usecols=["TIMESTAMP"])
        if stamps_frame.empty:
            print(f"  {site:<12} SKIP: no data rows")
            skipped += 1
            continue
        values = extraterrestrial_radiation(stamps_frame["TIMESTAMP"], lat, lon, elev).to_numpy(dtype=float)
        finite = np.isfinite(values)

        if args.dry_run:
            print(
                f"  {site:<12} lat={lat:7.3f} lon={lon:8.3f} rows={len(values):>7} "
                f"computed={int(finite.sum()):>7} max={np.nanmax(values) if finite.any() else float('nan'):7.1f} "
                f"night={float((values[finite] == 0).mean()) if finite.any() else float('nan'):.3f}"
            )
            continue

        shutil.copy2(path, backup_dir / path.name)
        staged = path.parent / (path.name + ".ext_rad_tmp")
        try:
            append_column(path, staged, values)
        except Exception as exc:
            staged.unlink(missing_ok=True)
            print(f"  {site:<12} FAIL during write: {exc}")
            failures += 1
            continue

        problems = verify(staged)
        if problems:
            staged.unlink(missing_ok=True)
            print(f"  {site:<12} FAIL verification: {'; '.join(problems)}")
            failures += 1
            continue

        staged.replace(path)
        done += 1
        print(f"  {site:<12} OK   rows={len(values):>7} lat={lat:7.3f}")

    print(f"\nwritten={done} skipped={skipped} failed={failures}")
    if failures:
        print("=== BACKFILL INCOMPLETE ===")
        return 1
    print("=== BACKFILL OK ===" if args.apply else "=== DRY RUN OK ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
