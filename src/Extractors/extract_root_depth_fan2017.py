"""Fan et al. 2017 rooting-depth + hydrologic-regime extractor.

Streams six Fan-2017 hydrologic-regime variables from the Universidade de
Santiago de Compostela THREDDS server at 30 arcsec native resolution,
sampling one pixel per SAPFLUXNET site via OPeNDAP. No full-file downloads
are performed: slicing an :class:`xarray.Dataset` opened from an OPeNDAP
URL issues only the HTTP byte-range needed for the selected indices.

Feature columns produced (all per-site static features):

========================  ===========  ====================  =======================================
Column                    Source var   Units                 Meaning
========================  ===========  ====================  =======================================
``root_depth``            ETDEPTH      m (positive)          Depth supplying ET (effective root depth)
``infiltration_depth``    INFDEPTH     m (positive)          Max infiltration depth (vadose thickness)
``deep_drainage``         DD           mm/day                Deep-drainage flux, annual mean
``drainage_frequency``    FRDD         fraction (0-1)        Fraction of months with deep drainage
``regolith_flush_rate``   FLUSHRATE    mm/day                Regolith flushing rate, 15-year mean
``gw_residence_time``     RTIME        s                     Groundwater residence time
========================  ===========  ====================  =======================================

Sign convention — important
---------------------------
Fan 2017 stores ``ETDEPTH`` and ``INFDEPTH`` as **negative z-values below
surface** (geomorphology convention). A site with 2 m effective root depth
decodes to ``-2.0 m`` from the raw Int16 + scale/offset. This extractor
flips the sign on read via ``VariableSpec.sign = -1`` so the output CSV
stores positive magnitudes, which is the ML-friendly convention and
produces sensible SHAP attributions (bigger = deeper = more water access).
Flux / frequency / time variables are kept as-is.

Plus five provenance columns named ``root_depth_*`` that record the source
continent file, the sampled pixel centre, the mask-validity flag, and the
fallback-radius used by the nearest-valid-pixel search.

Reference
---------
Fan Y, Miguez-Macho G, Jobbágy EG, Jackson RB, Otero-Casal C (2017).
    Hydrologic regulation of plant rooting depth.
    *PNAS* 114(40):10572-10577. doi:10.1073/pnas.1712381114

Data source
-----------
``http://thredds-gfnl.usc.es/thredds/dodsC/INFILTRATION_DATA/{CONTINENT}_{FILE_KIND}.nc``

Creator: Gonzalo Miguez-Macho, Universidade de Santiago de Compostela.
The server uses plain HTTP (the HTTPS variant presents a self-signed cert).
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

THREDDS_DODS_BASE: str = "http://thredds-gfnl.usc.es/thredds/dodsC/INFILTRATION_DATA"

Continent = Literal["AFRICA", "AUSTRALIA", "EURASIA", "NAMERICA", "SAMERICA"]
FileKind = Literal["ETDEPTH", "INFDEPTH_DD_FRDD", "FLUSHRATE_RTIME"]

_FILE_SUFFIX: dict[FileKind, str] = {
    "ETDEPTH": "ETDEPTH.nc",
    "INFDEPTH_DD_FRDD": "INFDEPTH_DD_FRDD.nc",
    "FLUSHRATE_RTIME": "FLUSHRATE_RTIME.nc",
}


@dataclass(frozen=True)
class VariableSpec:
    """Static description of one Fan 2017 source variable.

    ``physical_min`` / ``physical_max`` bound the *post-sign* value — i.e.
    the value stored in the output CSV after ``sign`` has been applied.
    ``sign = -1`` flips below-surface z-values (ETDEPTH, INFDEPTH) to the
    positive-magnitude convention used throughout the sap-velocity pipeline.
    """

    feature_name: str
    file_kind: FileKind
    source_var: str
    units: str
    physical_min: float
    physical_max: float
    sign: int = 1


VARIABLE_SCHEMA: tuple[VariableSpec, ...] = (
    # ETDEPTH / INFDEPTH: source stores negative z-values below surface;
    # sign=-1 flips to positive magnitude. Max decoded magnitude after flip
    # is ~500 m (raw Int16 range limit), so physical_max=500 is the hard cap.
    VariableSpec("root_depth", "ETDEPTH", "ETDEPTH", "m", 0.0, 500.0, sign=-1),
    VariableSpec("infiltration_depth", "INFDEPTH_DD_FRDD", "INFDEPTH", "m", 0.0, 500.0, sign=-1),
    # Fluxes / frequencies / times: source stores positive values directly.
    VariableSpec("deep_drainage", "INFDEPTH_DD_FRDD", "DD", "mm/day", -1.0, 50.0),
    VariableSpec("drainage_frequency", "INFDEPTH_DD_FRDD", "FRDD", "fraction", 0.0, 1.0),
    VariableSpec("regolith_flush_rate", "FLUSHRATE_RTIME", "FLUSHRATE", "mm/day", -1.0, 50.0),
    VariableSpec("gw_residence_time", "FLUSHRATE_RTIME", "RTIME", "s", 0.0, 1.0e13),
)

FEATURE_COLUMNS: tuple[str, ...] = tuple(v.feature_name for v in VARIABLE_SCHEMA)

PROVENANCE_COLUMNS: tuple[str, ...] = (
    "root_depth_mask_valid",
    "root_depth_fallback_px",
    "root_depth_continent",
    "root_depth_lat_pixel",
    "root_depth_lon_pixel",
)


@dataclass(frozen=True)
class ContinentBBox:
    """Bounding box in geographic coordinates used for continent routing."""

    name: Continent
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def contains(self, lat: float, lon: float) -> bool:
        return self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max


# Ordered: first match wins. The EURASIA bbox is verified against the live
# file (lat 0-83, lon -14-180). Other continents use conservative envelopes;
# the nearest-valid-pixel search disambiguates overlap regions via the mask.
CONTINENT_BBOXES: tuple[ContinentBBox, ...] = (
    ContinentBBox("EURASIA", lat_min=0.0, lat_max=83.0, lon_min=-14.0, lon_max=180.0),
    ContinentBBox("NAMERICA", lat_min=5.0, lat_max=85.0, lon_min=-180.0, lon_max=-50.0),
    ContinentBBox("SAMERICA", lat_min=-60.0, lat_max=15.0, lon_min=-85.0, lon_max=-30.0),
    ContinentBBox("AFRICA", lat_min=-40.0, lat_max=40.0, lon_min=-20.0, lon_max=55.0),
    ContinentBBox("AUSTRALIA", lat_min=-50.0, lat_max=0.0, lon_min=100.0, lon_max=180.0),
)


@dataclass(frozen=True)
class RootDepthResult:
    """Per-site extraction result, including provenance for debugging."""

    site_name: str
    lat_site: float
    lon_site: float
    lat_pixel: float
    lon_pixel: float
    continent: Continent | None
    mask_valid: bool
    fallback_radius_px: int
    values: dict[str, float | None]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def pick_continent(lat: float, lon: float) -> Continent | None:
    """Return the continent file whose bounding box contains ``(lat, lon)``.

    Overlapping bounding boxes resolve to the earliest entry in
    ``CONTINENT_BBOXES``. Returns ``None`` for mid-ocean / out-of-domain sites.
    """
    for bbox in CONTINENT_BBOXES:
        if bbox.contains(lat, lon):
            return bbox.name
    return None


def opendap_url(continent: Continent, file_kind: FileKind) -> str:
    """Construct the OPeNDAP dataset URL for a (continent, file) pair."""
    return f"{THREDDS_DODS_BASE}/{continent}_{_FILE_SUFFIX[file_kind]}"


def open_continent_datasets(continent: Continent) -> dict[FileKind, xr.Dataset]:
    """Lazily open the three Fan 2017 files for a continent via OPeNDAP.

    xarray defers the network read until ``.values`` is accessed on a slice,
    so this call costs one TCP handshake + metadata read per file.
    """
    return {
        kind: xr.open_dataset(
            opendap_url(continent, kind),
            engine="netcdf4",
            decode_cf=True,
        )
        for kind in _FILE_SUFFIX
    }


def _nearest_valid_pixel_indices(
    mask: np.ndarray,
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    lat: float,
    lon: float,
    max_radius_px: int,
) -> tuple[int, int, bool, int]:
    """Find the nearest pixel with ``mask == 1`` to ``(lat, lon)``.

    Returns a tuple ``(lat_idx, lon_idx, nearest_was_valid, fallback_used)``:

    * If the nearest pixel is valid, ``fallback_used == 0``.
    * If the nearest pixel is masked but a valid neighbour is found within
      ``max_radius_px`` Chebyshev-distance pixels, ``fallback_used`` is the
      smallest radius at which a valid pixel was found.
    * If no valid pixel exists within ``max_radius_px``, the nearest-by-
      distance indices are returned with ``fallback_used == -1``.

    The fallback search is Chebyshev (square window) for cheap slicing, then
    ranks candidates inside the window by squared Euclidean pixel distance.
    """
    lat_idx0 = int(np.abs(lat_arr - lat).argmin())
    lon_idx0 = int(np.abs(lon_arr - lon).argmin())

    if int(mask[lat_idx0, lon_idx0]) == 1:
        return lat_idx0, lon_idx0, True, 0

    n_lat, n_lon = mask.shape
    for radius in range(1, max_radius_px + 1):
        lat_lo = max(0, lat_idx0 - radius)
        lat_hi = min(n_lat, lat_idx0 + radius + 1)
        lon_lo = max(0, lon_idx0 - radius)
        lon_hi = min(n_lon, lon_idx0 + radius + 1)
        window = mask[lat_lo:lat_hi, lon_lo:lon_hi]
        if not (window == 1).any():
            continue
        ys, xs = np.where(window == 1)
        dy = ys - (lat_idx0 - lat_lo)
        dx = xs - (lon_idx0 - lon_lo)
        closest = int(np.argmin(dy * dy + dx * dx))
        return (
            int(lat_lo + ys[closest]),
            int(lon_lo + xs[closest]),
            False,
            radius,
        )

    return lat_idx0, lon_idx0, False, -1


# ---------------------------------------------------------------------------
# Per-site fetch
# ---------------------------------------------------------------------------


OpenerFn = Callable[[Continent], dict[FileKind, xr.Dataset]]


def fetch_all_variables_at_point(
    lat: float,
    lon: float,
    site_name: str = "unknown",
    *,
    dataset_cache: dict[Continent, dict[FileKind, xr.Dataset]] | None = None,
    max_radius_px: int = 3,
    opener: OpenerFn = open_continent_datasets,
) -> RootDepthResult:
    """Fetch all six Fan 2017 variables at one site location.

    The pixel index is chosen once via a mask-aware nearest-valid-pixel
    search, then reused across all six ``.isel()`` calls. The
    ``dataset_cache`` is shared across calls so the OPeNDAP handshake is paid
    once per continent per batch.

    For sites that fall outside every continent bounding box (e.g. mid-ocean)
    the result carries ``continent = None``, ``fallback_radius_px = -1`` and
    all ``values`` set to ``None``.
    """
    continent = pick_continent(lat, lon)
    if continent is None:
        logger.warning("Site %s at (%s, %s) outside all continent bboxes", site_name, lat, lon)
        return RootDepthResult(
            site_name=site_name,
            lat_site=lat,
            lon_site=lon,
            lat_pixel=float("nan"),
            lon_pixel=float("nan"),
            continent=None,
            mask_valid=False,
            fallback_radius_px=-1,
            values={fn: None for fn in FEATURE_COLUMNS},
        )

    if dataset_cache is None:
        dataset_cache = {}
    if continent not in dataset_cache:
        logger.info("Opening Fan 2017 datasets for %s", continent)
        dataset_cache[continent] = opener(continent)
    datasets = dataset_cache[continent]

    # All three files share the same lat/lon grid and mask. Use ETDEPTH as
    # the reference for the nearest-valid-pixel search. Normalise the mask to
    # int8 so `(mask == 1)` comparisons are robust against float-typed masks
    # that some OPeNDAP servers or CF-decode paths deliver.
    reference = datasets["ETDEPTH"]
    mask_raw = np.asarray(reference["mask"].values)
    mask_arr = (mask_raw != 0).astype(np.int8)
    lat_arr = reference["lat"].values
    lon_arr = reference["lon"].values

    lat_idx, lon_idx, nearest_valid, fallback_used = _nearest_valid_pixel_indices(
        mask_arr, lat_arr, lon_arr, lat, lon, max_radius_px
    )

    values: dict[str, float | None] = {}
    if fallback_used == -1:
        for feature_name in FEATURE_COLUMNS:
            values[feature_name] = None
    else:
        for spec in VARIABLE_SCHEMA:
            data_array = datasets[spec.file_kind][spec.source_var]
            raw = data_array.isel(time=0, lat=lat_idx, lon=lon_idx).values
            if np.isfinite(raw):
                # spec.sign flips ETDEPTH/INFDEPTH from below-surface z-values
                # to positive magnitude; pass-through (sign=1) for fluxes/time.
                val: float | None = float(raw) * spec.sign
            else:
                val = None
            if val is not None and not (spec.physical_min <= val <= spec.physical_max):
                logger.warning(
                    "Site %s: %s = %s outside expected range [%s, %s]",
                    site_name,
                    spec.feature_name,
                    val,
                    spec.physical_min,
                    spec.physical_max,
                )
            values[spec.feature_name] = val

    # When no valid pixel was found within max_radius_px, return NaN for the
    # pixel coordinates so downstream consumers cannot confuse the masked
    # nearest pixel with a real sample location (consistent with the
    # continent-is-None path above).
    if fallback_used == -1:
        lat_pixel_out = float("nan")
        lon_pixel_out = float("nan")
    else:
        lat_pixel_out = float(lat_arr[lat_idx])
        lon_pixel_out = float(lon_arr[lon_idx])

    return RootDepthResult(
        site_name=site_name,
        lat_site=lat,
        lon_site=lon,
        lat_pixel=lat_pixel_out,
        lon_pixel=lon_pixel_out,
        continent=continent,
        mask_valid=nearest_valid,
        fallback_radius_px=fallback_used,
        values=values,
    )


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------


def _result_to_row(result: RootDepthResult, site_col: str) -> dict:
    row: dict = {site_col: result.site_name}
    row.update(result.values)
    row["root_depth_mask_valid"] = result.mask_valid
    row["root_depth_fallback_px"] = result.fallback_radius_px
    row["root_depth_continent"] = result.continent
    row["root_depth_lat_pixel"] = result.lat_pixel
    row["root_depth_lon_pixel"] = result.lon_pixel
    return row


def extract_root_depth_for_sites(
    input_csv: Path,
    output_csv: Path,
    *,
    lon_col: str = "lon",
    lat_col: str = "lat",
    site_col: str = "site_name",
    max_radius_px: int = 3,
    resume: bool = True,
    flush_every: int = 10,
    opener: OpenerFn = open_continent_datasets,
) -> pd.DataFrame:
    """Extract all six Fan 2017 variables for every site in ``input_csv``.

    Writes an output CSV with one row per site, containing the six
    feature columns plus provenance columns. The writer is idempotent: when
    ``resume=True`` and ``output_csv`` already exists, sites listed in that
    file are skipped, and results are appended atomically (temp-file rename)
    every ``flush_every`` sites so a mid-run network failure loses at most
    ``flush_every - 1`` sites of work.
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df_in = pd.read_csv(input_csv)
    required = {lon_col, lat_col, site_col}
    missing = required - set(df_in.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")

    done: set[str] = set()
    if resume and output_csv.exists():
        existing = pd.read_csv(output_csv)
        done = set(existing[site_col].astype(str))
        logger.info("Resuming: %d sites already present in %s", len(done), output_csv)

    cache: dict[Continent, dict[FileKind, xr.Dataset]] = {}
    pending_rows: list[dict] = []

    def _flush() -> None:
        if not pending_rows:
            return
        new_df = pd.DataFrame(pending_rows)
        if output_csv.exists():
            prev = pd.read_csv(output_csv)
            combined = pd.concat([prev, new_df], ignore_index=True)
        else:
            combined = new_df
        tmp = output_csv.with_suffix(output_csv.suffix + ".tmp")
        combined.to_csv(tmp, index=False)
        tmp.replace(output_csv)
        pending_rows.clear()

    try:
        for _, row in df_in.iterrows():
            site_name = str(row[site_col])
            if site_name in done:
                continue
            result = fetch_all_variables_at_point(
                lat=float(row[lat_col]),
                lon=float(row[lon_col]),
                site_name=site_name,
                dataset_cache=cache,
                max_radius_px=max_radius_px,
                opener=opener,
            )
            pending_rows.append(_result_to_row(result, site_col))
            # Guard against duplicate site names in the same input CSV — without
            # this, a repeated row would get processed twice and written twice.
            done.add(site_name)
            if len(pending_rows) >= flush_every:
                _flush()
    finally:
        _flush()
        for datasets in cache.values():
            for ds in datasets.values():
                try:
                    ds.close()
                except Exception:  # noqa: BLE001 — best-effort close
                    logger.debug("Failed to close dataset", exc_info=True)

    if not output_csv.exists():
        # Can happen when resume=True, every site was already marked done, and
        # the existing output_csv was deleted between the resume-read and now.
        return pd.DataFrame()
    return pd.read_csv(output_csv)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main() -> int:
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from path_config import get_default_paths  # noqa: PLC0415 — late import

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    paths = get_default_paths()
    input_csv = Path(paths.site_info_path)
    output_csv = Path(paths.root_depth_fan2017_data_path)

    logger.info("Reading sites from %s", input_csv)
    logger.info("Writing Fan 2017 features to %s", output_csv)

    result_df = extract_root_depth_for_sites(input_csv, output_csv)

    logger.info("Extracted %d rows.", len(result_df))
    for feature_name in FEATURE_COLUMNS:
        series = result_df[feature_name]
        n_missing = int(series.isna().sum())
        logger.info(
            "  %-22s  non-null=%d  mean=%.4g  missing=%d",
            feature_name,
            int(series.notna().sum()),
            float(series.dropna().mean()) if series.notna().any() else float("nan"),
            n_missing,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
