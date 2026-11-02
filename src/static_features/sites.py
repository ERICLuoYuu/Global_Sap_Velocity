"""The site list: one validated row per site, with its cell and its IGBP class."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from src.static_features.grid import cell_of

REQUIRED_COLUMNS = ("site_name", "lat", "lon", "start_date", "end_date")
OUTPUT_COLUMNS = ["site_name", "lat", "lon", "start_date", "end_date", "si_igbp", "row", "col"]


_DEVILLA = {
    "published_lat": 56.03333,
    "published_lon": -3.716667,
    "lat": 56.0792,
    "lon": -3.6625,
    "reason": "Irvine et al. 1998 print 56 deg 2 min N, 3 deg 43 min W, which is water in the Firth of Forth, for a closed "
    "15 m stand at 75 m. Inferred, not surveyed: the Devilla Forest cell closest to 75 m, 6.1 km from the published point.",
}

#: Published points that miss their stand, and where the stand is. Owner's decision, 2026-11-02. The delivered
#: metadata stays as it is, and only static extraction reads these: the merge still places daily climate by si_lat.
COORDINATE_CORRECTIONS = {
    "CHE_DAV_SEE": {
        "published_lat": 46.81668,
        "published_lon": 9.856198,
        "lat": 46.81533,
        "lon": 9.85591,
        "reason": "The published point lies 1.5 m inside the cell holding Lake Davos, 70% water. Same forest as CHE_DAV, "
        "about 150 m south, whose coordinates it takes.",
    },
    "GBR_DEV_CON": _DEVILLA,
    "GBR_DEV_DRO": _DEVILLA,
}


def _apply_corrections(frame: pd.DataFrame) -> pd.DataFrame:
    """A copy with corrected points. A site whose list no longer holds the published point stops the run."""
    lats, lons, stale = frame["lat"].tolist(), frame["lon"].tolist(), []
    for position, name in enumerate(frame["site_name"]):
        correction = COORDINATE_CORRECTIONS.get(name)
        if correction is None:
            continue
        if (lats[position], lons[position]) != (correction["published_lat"], correction["published_lon"]):
            stale.append(f"{name}: list holds {lats[position]}, {lons[position]}")
        lats[position], lons[position] = correction["lat"], correction["lon"]
    if stale:
        raise ValueError(f"coordinate corrections no longer match the published point: {stale}")
    return frame.assign(lat=lats, lon=lons)


def _read_igbp(site_md_dir: Path, name: str) -> object:
    path = Path(site_md_dir) / f"{name}_site_md.csv"
    if not path.exists():
        raise ValueError(f"{name}: no metadata file {path}")
    metadata = pd.read_csv(path, nrows=1)
    if "si_igbp" not in metadata.columns or metadata.empty:
        raise ValueError(f"{name}: {path} holds no si_igbp")
    return metadata["si_igbp"].iloc[0]


def load_sites(site_info_csv: Path, site_md_dir: Path) -> pd.DataFrame:
    """Read the site list and attach each site's cell and si_igbp, naming every site that cannot be placed.

    Two sites may share a cell. One name appearing twice stops here, because later joins on the name
    would silently multiply rows. A blank si_igbp stays missing: only the agreement report reads it.
    """
    frame = pd.read_csv(
        site_info_csv,
        float_precision="round_trip",
        dtype={"site_name": str, "start_date": str, "end_date": str},
    )
    absent = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if absent:
        raise ValueError(f"{site_info_csv}: missing columns {absent}")
    if frame.empty:
        raise ValueError(f"{site_info_csv}: no sites")
    repeated = sorted(frame.loc[frame["site_name"].duplicated(), "site_name"].unique())
    if repeated:
        raise ValueError(f"{site_info_csv}: site names appear twice: {repeated}")

    frame = frame.assign(
        lat=pd.to_numeric(frame["lat"], errors="coerce"),
        lon=pd.to_numeric(frame["lon"], errors="coerce"),
    )
    frame = _apply_corrections(frame)
    rows, cols, igbp, problems = [], [], [], []
    for site in frame.itertuples(index=False):
        try:
            row, col = cell_of(site.lat, site.lon)
            igbp.append(_read_igbp(site_md_dir, site.site_name))
        except ValueError as problem:
            problems.append(f"{site.site_name}: {problem}")
            continue
        rows.append(row)
        cols.append(col)
    if problems:
        raise ValueError(f"{len(problems)} sites cannot be used:\n" + "\n".join(problems))

    return frame.assign(si_igbp=igbp, row=rows, col=cols)[OUTPUT_COLUMNS]


def site_list_hash(sites: pd.DataFrame) -> str:
    """sha256 of the sorted name,lat,lon lines, so a part can tell which site list it was built from."""
    lines = sorted(
        f"{site.site_name},{float(site.lat)!r},{float(site.lon)!r}" for site in sites.itertuples(index=False)
    )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
