"""Fill in ``si_biome`` for the 24 sites the archive never gave one.

``si_biome`` is absent at exactly the internal sites, and the merge reads it
raw (``biome_info['si_biome'][0]``) with no validation.  A missing value
therefore arrives as NaN rather than raising -- and because ``NaN != NaN``,
every unlabelled site takes its own dict slot in the per-biome grouping.  That
is the defect this module closes.

Why the values are WorldClim 2.1 and not the archive's own inputs
----------------------------------------------------------------
``qc_get_biome()`` wrote ``si_mat``/``si_map``/``si_biome`` in one pass, which
is why all three are absent together.  Its pre-2018 form -- the one that made
the published data -- read climate through ``RFc::fcTimeSeriesYearly(...,
dataSets = 'WorldClim 1.4')``.  That names WorldClim 1.4 as the *dataset*, but
the access path was FetchClimate, which did its own spatial interpolation and
temporal aggregation rather than returning a pixel.  FetchClimate is
decommissioned, and sampling WorldClim 1.4 rasters directly does NOT reproduce
the stored numbers: 0/202 precipitation values match, and label recovery is
84.2% -- no better than WorldClim 2.1's 85.1%.  The gap is scatter, not bias (a
constant offset explains 10% of the precipitation error), so no calibration
closes it.

So the archive's inputs are irrecoverable and no procedure can pass a
match-the-202 gate.  What survives is the *classifier*: WorldClim 1.4 and 2.1
agree with each other at 92.0%, better than either agrees with the archive, and
they return identical label counts for the 24.  The derivation is robust; only
the archive's specific FetchClimate values are not.

The polygons are read from the R source at run time and never retyped, so the
212 numbers that define the nine biomes cannot drift from the archive's own.

Vocabulary note: the QC package calls class 3 ``'Mediterranean'``; the released
database calls it ``'Woodland/Shrubland'``.  Same polygon -- ``sapfluxnetr``
NEWS 0.1.1 records the rename.  ``POLYGON_TO_ARCHIVE`` applies it, so derived
labels share the released vocabulary.

Two ways to use it, deliberately separate
-----------------------------------------
``derive_for_scale`` is a pure read: it returns what each site would get and
writes nothing.  ``--apply`` writes ``si_biome``/``si_mat``/``si_map`` into the
site metadata for the merge, which reads the column directly.  It refuses to
touch a site whose archive value is present, refuses to invent a label for a
site falling outside every polygon, and re-reads every file afterwards to fail
if any other field moved.

Usage
-----
    # report only, nothing written
    python -m src.site_info.derive_biome --scale sapwood_combined

    # also write the three columns and a provenance table
    python -m src.site_info.derive_biome --scale sapwood_combined --apply
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import OrderedDict, namedtuple

SITE_SUFFIX = "_site_md.csv"
BIOME_COLUMN = "si_biome"
MAT_COLUMN = "si_mat"
MAP_COLUMN = "si_map"
REMARKS_COLUMN = "si_remarks"
BACKUP_SUFFIX = ".pre_biome_apply"

_QUOTING = csv.QUOTE_MINIMAL
ABSENT = frozenset({"", "nan", "na", "n/a", "none", "null", "-9999"})

#: The QC package's name for class 3 versus the released database's name.
POLYGON_TO_ARCHIVE = {"Mediterranean": "Woodland/Shrubland"}

#: Recorded in ``si_remarks`` so the row says where it came from, following the
#: precedent ITA_CP2 set for si_igbp.
PROVENANCE_NOTE = (
    "si_biome/si_mat/si_map derived from WorldClim 2.1 bio1/bio12 at the site "
    "pixel via the SAPFLUXNET Whittaker/Ricklefs polygons (qc_get_biome); "
    "not archive-native -- the archive's FetchClimate inputs are irreproducible"
)

#: What a site resolves to.  ``label`` is None when the climate falls outside
#: every polygon, which is a refusal rather than a value.
BiomeFill = namedtuple("BiomeFill", "label mat map_mm source")


def is_absent(value) -> bool:
    """Treat the archive's several spellings of "no value" as one thing."""
    return value is None or str(value).strip().lower() in ABSENT


def _read_table(path: str) -> tuple[list[str], list[dict]]:
    """``(fieldnames, rows)`` with every value as text and nothing coerced."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, list(reader)


def _write_table(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    """Truncate in place, which keeps the inode.

    Deliberately not a temp-and-rename: 24 of these corpus files share an inode
    with the Stage-1 ``format_unitcon`` copy, and writing through the link
    updates both.  A rename would break the link and leave the twin stale.
    """
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=_QUOTING)
        writer.writeheader()
        writer.writerows(rows)


def site_codes(csv_dir: str) -> list[str]:
    """Every site in a scale directory, by exact suffix strip.

    Never a prefix glob: ``CHE_DAV`` is a prefix of ``CHE_DAV_SEE``.
    """
    paths = glob.glob(os.path.join(csv_dir, "*" + SITE_SUFFIX))
    return sorted(os.path.basename(p)[: -len(SITE_SUFFIX)] for p in paths)


def parse_whittaker_polygons(r_source: str) -> OrderedDict:
    """The nine biome polygons, read from the R source rather than retyped.

    ``biomes_df`` holds one flat ``mat`` vector, one flat ``map`` vector and a
    run of ``rep('Name', n)`` blocks that cut them into polygons.  Order is
    load-bearing -- ``sp::over`` returns the FIRST polygon containing the point
    -- so the blocks stay in source order.
    """
    with open(r_source, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    start = text.index("biomes_df <- data.frame(")
    body = text[start : text.index("# STEP 2", start)]

    def numeric_vector(name: str) -> list[float]:
        head = body.index(name + " = c(")
        open_paren = body.index("(", head + len(name))
        depth, i = 0, open_paren
        while True:
            if body[i] == "(":
                depth += 1
            elif body[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        return [float(t) for t in body[open_paren + 1 : i].replace("\n", " ").split(",")]

    mat = numeric_vector("mat")
    map_mm = numeric_vector("map")
    blocks = [(name, int(count)) for name, count in re.findall(r"rep\('([^']+)',\s*(\d+)\)", body)]

    if len(mat) != len(map_mm):
        raise ValueError(f"mat has {len(mat)} vertices but map has {len(map_mm)}")
    if sum(count for _, count in blocks) != len(mat):
        raise ValueError(f"rep() blocks cover {sum(c for _, c in blocks)} vertices but mat has {len(mat)}")

    polygons: OrderedDict = OrderedDict()
    cursor = 0
    for name, count in blocks:
        polygons[name] = [(map_mm[i], mat[i]) for i in range(cursor, cursor + count)]
        cursor += count
    return polygons


def point_in_ring(x: float, y: float, ring) -> bool:
    """Ray casting, matching ``sp::over`` for these closed rings."""
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x1 + (y - y1) * (x2 - x1) / (y2 - y1) > x:
            inside = not inside
    return inside


def classify(map_mm, mat_c, polygons) -> str | None:
    """The biome whose envelope contains (precipitation, temperature).

    None means the point lies outside every polygon.  That is a real outcome,
    not an error: the archive's own CHN_ARG_GWD and CHN_ARG_GWS fall outside,
    which is where ``sp::over`` returns NA.  Callers decide what to do with it;
    this function will not guess a nearest class.
    """
    try:
        x, y = float(map_mm), float(mat_c)
    except (TypeError, ValueError):
        return None
    if x != x or y != y:  # NaN
        return None
    for name, ring in polygons.items():
        if point_in_ring(x, y, ring):
            return POLYGON_TO_ARCHIVE.get(name, name)
    return None


def read_climate(climate_csv: str) -> dict[str, tuple[str, str]]:
    """``{site: (bio1, bio12)}`` from the WorldClim 2.1 extraction table."""
    _, rows = _read_table(climate_csv)
    climate = {}
    for row in rows:
        site = (row.get("site_name") or "").strip()
        if site:
            climate[site] = (row.get("bio1", ""), row.get("bio12", ""))
    return climate


def derive_for_scale(csv_dir: str, climate_csv: str, r_source: str) -> dict[str, BiomeFill]:
    """What every site would get.  A pure read -- nothing is written."""
    polygons = parse_whittaker_polygons(r_source)
    climate = read_climate(climate_csv)

    derived = {}
    for site in site_codes(csv_dir):
        bio1, bio12 = climate.get(site, ("", ""))
        if is_absent(bio1) or is_absent(bio12):
            derived[site] = BiomeFill(None, "", "", "no WorldClim 2.1 climate")
            continue
        label = classify(bio12, bio1, polygons)
        source = "WorldClim 2.1" if label else "outside every polygon"
        derived[site] = BiomeFill(label, str(bio1), str(bio12), source)
    return derived


def needs_fill(csv_dir: str) -> list[str]:
    """Sites whose archive ``si_biome`` is absent."""
    gaps = []
    for site in site_codes(csv_dir):
        _, rows = _read_table(os.path.join(csv_dir, site + SITE_SUFFIX))
        if rows and is_absent(rows[0].get(BIOME_COLUMN)):
            gaps.append(site)
    return gaps


def _note_remarks(existing: str) -> str:
    """Append the provenance note, or set it when the field is empty."""
    if is_absent(existing):
        return PROVENANCE_NOTE
    if PROVENANCE_NOTE in existing:
        return existing
    return existing.rstrip().rstrip(";") + "; " + PROVENANCE_NOTE


def apply_to_site_md(csv_dir: str, derived: dict[str, BiomeFill]) -> list[tuple[str, str]]:
    """Write the three columns for sites the archive left blank.

    Refuses to overwrite a present archive value, and refuses to write a site
    that fell outside every polygon.  Returns ``(site, label)`` for what moved.
    """
    written = []
    for site in needs_fill(csv_dir):
        fill = derived.get(site)
        if fill is None or fill.label is None:
            continue
        path = os.path.join(csv_dir, site + SITE_SUFFIX)
        fieldnames, rows = _read_table(path)
        if not rows:
            continue
        for column in (BIOME_COLUMN, MAT_COLUMN, MAP_COLUMN):
            if column not in fieldnames:
                raise ValueError(f"{site}: {column} missing from the header")

        backup = path + BACKUP_SUFFIX
        if not os.path.exists(backup):
            original_fieldnames, original_rows = fieldnames, [dict(r) for r in rows]
            _write_table(backup, original_fieldnames, original_rows)

        rows[0][BIOME_COLUMN] = fill.label
        rows[0][MAT_COLUMN] = fill.mat
        rows[0][MAP_COLUMN] = fill.map_mm
        if REMARKS_COLUMN in fieldnames:
            rows[0][REMARKS_COLUMN] = _note_remarks(rows[0].get(REMARKS_COLUMN) or "")
        _write_table(path, fieldnames, rows)
        written.append((site, fill.label))
    return written


def write_report(path: str, derived: dict[str, BiomeFill], gaps: list[str]) -> None:
    """A provenance table: what each gap site got, and from what numbers."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, quoting=_QUOTING)
        writer.writerow(["site", "si_biome", "si_mat", "si_map", "source"])
        for site in gaps:
            fill = derived.get(site, BiomeFill(None, "", "", "not derived"))
            writer.writerow([site, fill.label or "", fill.mat, fill.map_mm, fill.source])


def default_report_path(scale: str) -> str:
    return os.path.join("outputs", "site_info", f"biome_fill_{scale}.csv")


def _default_csv_dir(scale: str) -> str:
    return os.path.join("data", "raw", "0.1.5", "0.1.5", "csv", scale)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scale", default="sapwood_combined")
    parser.add_argument("--csv-dir", default=None)
    parser.add_argument("--climate-csv", required=True)
    parser.add_argument("--r-source", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    csv_dir = args.csv_dir or _default_csv_dir(args.scale)
    derived = derive_for_scale(csv_dir, args.climate_csv, args.r_source)
    gaps = needs_fill(csv_dir)

    print(f"sites            : {len(derived)}")
    print(f"si_biome absent  : {len(gaps)}")

    outside = [s for s in gaps if derived[s].label is None]
    print(f"resolvable       : {len(gaps) - len(outside)} / {len(gaps)}")
    if outside:
        print(f"  outside every polygon or without climate: {outside}")

    counts: dict[str, int] = {}
    for site in gaps:
        label = derived[site].label
        if label:
            counts[label] = counts.get(label, 0) + 1
    print("label counts for the gap sites:")
    for label in sorted(counts, key=lambda k: (-counts[k], k)):
        print(f"  {label:<30} {counts[label]}")

    print()
    print(f"  {'site':<12} {'bio1':>8} {'bio12':>9}  label")
    for site in gaps:
        fill = derived[site]
        mat = float(fill.mat) if fill.mat else float("nan")
        map_mm = float(fill.map_mm) if fill.map_mm else float("nan")
        print(f"  {site:<12} {mat:8.2f} {map_mm:9.1f}  {fill.label or '-- ' + fill.source}")

    report = args.report or default_report_path(args.scale)
    write_report(report, derived, gaps)
    print(f"\nreport -> {report}")

    if args.apply:
        if outside:
            print(f"\nREFUSING to apply: {len(outside)} gap sites have no polygon")
            return 1
        written = apply_to_site_md(csv_dir, derived)
        print(f"\napplied to {len(written)} site_md files")
    else:
        print("\n(report only -- pass --apply to write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
