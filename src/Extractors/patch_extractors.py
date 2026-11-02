"""Apply the fixes the extractors need before they can run on the cluster.

Each patch is an exact string replacement that must match exactly once.  A
patch that matches zero times, or more than once, aborts the whole run without
writing anything -- so this script is safe to re-run and cannot half-apply.

Fixes, and why each one matters for this extraction:

1. extract_stand_age.py pinned the forest-age NetCDF to a hardcoded Windows
   path (``r".\\data\\raw\\grided\\stand_age\\<one release>.nc"``).  It cannot
   resolve on Linux, so the extractor exits 1 before reading a single site.

2. extract_pft.py wrapped the entire extraction *and* the output write in one
   ``except Exception: print(...)``.  A GEE quota error at site 40 of 61 left
   no output file and still exited 0 -- a job that reports success and produces
   nothing.  Now it re-raises.

3. extract_pft.py and extract_climatedata_gee.py call ``ee.Authenticate()``
   unconditionally.  In a batch job with no TTY that can block on an
   interactive consent flow until the wall clock kills it.  Stored credentials
   are tried first, and authentication is attempted only if they fail.

4. extract_globmap_lai.py calls bare ``exit()`` when its inputs are missing,
   which exits 0.  A missing sites file therefore looked like a clean run.
"""

from __future__ import annotations

import sys
from pathlib import Path

EXTRACTORS = Path(__file__).resolve().parent

PATCHES: list[tuple[str, str, str]] = []


def patch(filename: str, old: str, new: str) -> None:
    PATCHES.append((filename, old, new))


# --------------------------------------------------------------------------
# 1. stand age: resolve the NetCDF from the configured data root
# --------------------------------------------------------------------------
patch(
    "extract_stand_age.py",
    """    # Path to your MPI-BGC forest age file
    NC_PATH = r".\\data\\raw\\grided\\stand_age\\2026113115224222_BGIForestAgeMPIBGC1.0.0.nc"
""",
    """    # Path to the MPI-BGC forest age NetCDF.  Resolved from the configured
    # data root and matched by glob: the hardcoded Windows path this replaced
    # could not resolve on Linux, and pinned one release's filename.
    _stand_age_dir = (
        Path(paths.raw_grided_root) / "stand_age"
        if USE_PATH_CONFIG
        else Path("./data/raw/grided/stand_age")
    )
    _nc_candidates = sorted(_stand_age_dir.glob("*BGIForestAge*.nc"))
    if not _nc_candidates:
        print(f"\\nERROR: no *BGIForestAge*.nc found in {_stand_age_dir}")
        sys.exit(1)
    if len(_nc_candidates) > 1:
        print(f"NOTE: {len(_nc_candidates)} forest-age files found; "
              f"using {_nc_candidates[0].name}")
    NC_PATH = str(_nc_candidates[0])
""",
)

# --------------------------------------------------------------------------
# 2. PFT: stop swallowing the failure of the whole extraction
# --------------------------------------------------------------------------
patch(
    "extract_pft.py",
    """    except Exception as e:
        print(f"An error occurred: {e}")
""",
    """    except Exception as e:
        # Re-raise.  This handler wrapped the extraction *and* the write, so
        # any failure mid-run left no output file behind and still exited 0 --
        # a job that reports success and produces nothing.
        print(f"Landcover extraction FAILED: {e}")
        raise
""",
)

# --------------------------------------------------------------------------
# 3. Earth Engine: never block on an interactive consent flow in a batch job
# --------------------------------------------------------------------------
patch(
    "extract_pft.py",
    """        ee.Authenticate()
        ee.Initialize(project='era5download-447713')
""",
    """        # Stored credentials first: ee.Authenticate() can block on an
        # interactive consent flow, which never returns inside a batch job.
        try:
            ee.Initialize(project='era5download-447713')
        except Exception:
            ee.Authenticate()
            ee.Initialize(project='era5download-447713')
""",
)

patch(
    "extract_climatedata_gee.py",
    """def initialize_ee(project_id):
    ee.Authenticate()
    ee.Initialize(project=project_id)
    print("Authentication successful!")
""",
    """def initialize_ee(project_id):
    # Stored credentials first: ee.Authenticate() can block on an interactive
    # consent flow, which never returns inside a batch job.
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)
    print("Authentication successful!")
""",
)

# --------------------------------------------------------------------------
# 4. GlobMap LAI: a missing input must not exit 0
# --------------------------------------------------------------------------
patch(
    "extract_globmap_lai.py",
    """    if not os.path.exists(SITES_FILE):
        print(f"FATAL ERROR: The sites file '{SITES_FILE}' was not found.")
        exit()
    if not os.path.exists(DATA_DIRECTORY):
        print(f"FATAL ERROR: The data directory '{DATA_DIRECTORY}' was not found.")
        exit()
""",
    """    if not os.path.exists(SITES_FILE):
        print(f"FATAL ERROR: The sites file '{SITES_FILE}' was not found.")
        raise SystemExit(1)   # bare exit() returns 0: a missing input looked clean
    if not os.path.exists(DATA_DIRECTORY):
        print(f"FATAL ERROR: The data directory '{DATA_DIRECTORY}' was not found.")
        raise SystemExit(1)
""",
)


def main() -> int:
    # Verify every patch before writing any of them.
    planned: dict[str, str] = {}
    failures: list[str] = []

    for filename, old, new in PATCHES:
        path = EXTRACTORS / filename
        if not path.exists():
            failures.append(f"{filename}: file not found")
            continue

        text = planned.get(filename, path.read_text(encoding="utf-8"))
        occurrences = text.count(old)

        if occurrences == 1:
            planned[filename] = text.replace(old, new)
        elif occurrences == 0 and text.count(new) >= 1:
            print(f"  already applied : {filename}: {old.strip().splitlines()[0][:60]}")
            planned.setdefault(filename, text)
        else:
            failures.append(
                f"{filename}: expected 1 match, found {occurrences} for {old.strip().splitlines()[0][:70]!r}"
            )

    if failures:
        print("REFUSING TO PATCH -- nothing written:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    for filename, text in planned.items():
        path = EXTRACTORS / filename
        original = path.read_text(encoding="utf-8")
        if text == original:
            print(f"  unchanged       : {filename}")
            continue
        backup = path.with_suffix(path.suffix + ".prepatch")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        path.write_text(text, encoding="utf-8")
        print(f"  patched         : {filename}  (backup: {backup.name})")

    # Every patched file must still parse.
    import ast

    for filename in planned:
        path = EXTRACTORS / filename
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            print(f"SYNTAX ERROR after patching {filename}: {exc}")
            return 1
        print(f"  parses          : {filename}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
