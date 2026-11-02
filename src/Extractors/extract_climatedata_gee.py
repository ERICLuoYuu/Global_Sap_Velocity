"""Extract hourly ERA5-Land at each site over that site's own record window.

Reads ``site_info.csv`` (one row per site, with a closed ``[start_date,
end_date]`` window) and writes one row per site-hour to
``paths.era5_discrete_data_path``.

Three properties this module is responsible for, each of which was previously
violated silently:

* **The window is closed and hour-resolved.**  ``filterDate`` is half-open and
  the bounds used to be formatted ``%Y-%m-%d``, which floored them to midnight.
  The head therefore gained the hours before ``start_date`` and the tail lost
  every hour of ``end_date``'s own day.
* **A band that does not exist is a startup failure, not a NaN column.**
  ``feature['properties'].get(var)`` returns ``None`` for a misspelt band, so a
  wrong name produced a fully-null column that looked like missing data.
* **A dropped chunk fails the run.**  Chunk errors used to be printed and
  ``continue``d, so a site could lose whole months and the job still exited 0
  and wrote a CSV that looked complete.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import os
import random
import time
from pathlib import Path

import ee
import pandas as pd
from tqdm import tqdm

from path_config import (
    PathConfig,  # noqa: F401  (kept: callers import it from here)
    get_default_paths,
)

COLLECTION = "ECMWF/ERA5_LAND/HOURLY"
PROJECT_ID = "era5download-447713"

#: ERA5-Land HOURLY publishes one image per hour, so the closed window
#: [start, end] is the half-open window [start, end + STEP).
STEP = pd.Timedelta(hours=1)

#: A monthly chunk is at most 744 hourly images, i.e. 744 features out of
#: ``getInfo()``.  Earth Engine aborts a collection ``getInfo`` once it
#: accumulates ~5000 elements, so the chunk MUST stay sub-quarterly.
MAX_FEATURES_PER_CHUNK = 5000

#: Earth Engine's concurrency ceiling is an ACCOUNT property, not a code one: a
#: project in Restricted Mode admits far fewer parallel requests than the 32
#: this module used to send.  On 2026-08-18 that cost 160 of 226 sites, every
#: one of them to "Too Many Requests" and none of them to a data problem.
#: Default low and let the run raise it once the account state is known.
MAX_WORKERS = int(os.environ.get("ERA5_MAX_WORKERS", "4"))

#: A quota refusal is not a fault.  The request was well-formed and the answer
#: is "later", so it gets its own and much larger budget, while a genuine error
#: (a misspelt band, a malformed geometry) still fails fast on MAX_ATTEMPTS.
#: Sharing one 4-attempt budget meant a transient 429 burned the same allowance
#: as a permanent defect.
MAX_ATTEMPTS = int(os.environ.get("ERA5_MAX_ATTEMPTS", "4"))
MAX_QUOTA_ATTEMPTS = int(os.environ.get("ERA5_MAX_QUOTA_ATTEMPTS", "12"))

BACKOFF_SECONDS = 5.0

#: Cap the exponential so a long run cannot park a worker for an hour.
BACKOFF_CAP_SECONDS = 300.0

#: Substrings that mark a refusal as "come back later" rather than "this is
#: broken".  Matched case-insensitively against str(exc).  Deliberately does NOT
#: include "user memory limit exceeded", which is a real computational failure
#: that retrying cannot fix.
QUOTA_MARKERS = (
    "too many requests",
    "concurrency limit",
    "restricted mode",
    "quota",
    "rate limit",
    "429",
)

#: Opt-in directory for per-site resumability.  Unset (the default) reproduces
#: the previous all-or-nothing behaviour exactly.  A throttled 226-site run
#: takes hours, and without this a single refusal near the end discards every
#: site that already succeeded -- which is what happened on 2026-08-18.
CACHE_DIR = os.environ.get("ERA5_CACHE_DIR") or None

#: Bump this whenever a change in THIS FILE alters which rows a cached site
#: would contain -- chunk tiling, row shaping, what the sampling scale means.
#: The digest below covers the inputs, which is not enough on its own: after a
#: logic change the inputs are byte-identical, so every entry would still HIT
#: while being wrong.  Found on 2026-08-19, when the closed-window fix left one
#: surplus hour in 139 cached sites and nothing in the signature could see it.
#:
#:   (none)  first throttle/cache release (2026-08-19).  That digest carried NO
#:           version field at all, so forcing this constant to "1" does NOT
#:           reproduce those keys -- a migration has to hash the old payload
#:           shape itself.  See .claude/plan/rekey_cache.py.
#:   "2"     get_monthly_chunks stops at the last image INSIDE the window
CACHE_BEHAVIOUR_VERSION = "2"

#: ERA5-Land's 69 bands do not all share one projection, so the scale probe has
#: to name a single band.  Any band with the native land grid does; this one is
#: also what the coverage report checks.
SCALE_PROBE_BAND = "temperature_2m"

#: ERA5-Land is published on a regular 0.1 degree grid, which Earth Engine
#: reports as a nominal scale of 11132 m.  Sampling happens AT that scale, not
#: near it.  The extractor used to hardcode 10000, which is FINER than native,
#: and asking Earth Engine for a finer grid makes it reproject: the reprojected
#: pixel containing the point then draws its value from whichever source pixel is
#: nearest the *reprojected* pixel's centre, not necessarily the source pixel the
#: point falls in.  Measured over the 226-site corpus on 2026-08-17:
#:
#:   * USA_SMI_SER and ITA_CP2 landed on masked water, so every variable came
#:     back null for the site's whole record -- 52,560 site-hours that read as
#:     missing data rather than as a bad request;
#:   * 76 of the other 224 sites returned a neighbouring cell's value, differing
#:     by up to 3.29 K.
#:
#: At 11132 m the request grid aligns with the source and the pixel read is the
#: one containing the point: 226/226 sites return a value, 0 null, verified
#: through this module's own ``reduceRegions`` call shape.
ERA5_NATIVE_SCALE_M = 11132.0

#: How far Earth Engine's reported nominal scale may drift from the constant
#: before the run refuses to continue.  ERA5-Land's grid is fixed, so a real
#: difference means the product was regridded and the constant is stale.
SCALE_TOLERANCE_M = 1.0

VARIABLES = [
    "leaf_area_index_high_vegetation",
    "leaf_area_index_low_vegetation",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "temperature_2m",
    "dewpoint_temperature_2m",
    "surface_latent_heat_flux_hourly",
    "surface_net_solar_radiation_hourly",
    "surface_net_thermal_radiation_hourly",
    "surface_sensible_heat_flux_hourly",
    "surface_solar_radiation_downwards_hourly",
    "surface_thermal_radiation_downwards_hourly",
    "potential_evaporation_hourly",
    "evaporation_from_the_top_of_canopy_hourly",
    "total_evaporation_hourly",
    "total_precipitation_hourly",
    "evaporation_from_vegetation_transpiration_hourly",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    # Accumulated since 00 UTC, NOT an hourly increment.  The merge uses
    # `total_precipitation_hourly`; this one is kept only because dropping a
    # column would make the re-extracted delta disagree with the already
    # extracted reference half.  Do not use it as an hourly rate.
    "total_precipitation",
    "surface_pressure",
]


class ExtractionError(RuntimeError):
    """A site or chunk could not be extracted after every retry."""


def initialize_ee(project_id: str = PROJECT_ID) -> None:
    # Stored credentials first: ee.Authenticate() can block on an interactive
    # consent flow, which never returns inside a batch job.
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)
    print("Authentication successful!")


def as_naive_utc(value) -> pd.Timestamp:
    """Interpret a site_info stamp as UTC and drop the tzinfo.

    ``site_info`` is written with ``utc=True``, so a bare string is UTC.  Doing
    the conversion explicitly means an offset-carrying string can never be
    handed to Earth Engine as if it were UTC wall time.
    """
    ts = pd.to_datetime(value, utc=True)
    return ts.tz_localize(None) if ts.tzinfo is None else ts.tz_convert("UTC").tz_localize(None)


def collection_time_bounds(collection) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last image instant, asked of Earth Engine.

    Server-side ``reduceColumns`` rather than ``aggregate_array``: ERA5-Land
    HOURLY holds hundreds of thousands of images and pulling every start time
    to the client would be absurd.  Asked rather than hardcoded because the
    collection's end advances with a few days' latency, so any literal written
    here starts rotting immediately.
    """
    stats = collection.reduceColumns(ee.Reducer.minMax(), ["system:time_start"]).getInfo()
    if stats.get("min") is None or stats.get("max") is None:
        raise ExtractionError(f"{COLLECTION} reported no images at all")
    return (
        pd.to_datetime(stats["min"], unit="ms"),
        pd.to_datetime(stats["max"], unit="ms"),
    )


def validate_bands(collection, variables) -> None:
    """Fail at startup on a band name the collection does not have.

    ``feature['properties'].get(var)`` yields ``None`` for a name that was
    never returned, so a misspelt band is indistinguishable from a pixel with
    no data -- it becomes a column of NaN across every site and every hour and
    nothing in the run complains.  `dewpoint_2m` (the real band is
    `dewpoint_temperature_2m`) is exactly that mistake.
    """
    # Probe both ends of the record, not just the first image: the ``*_hourly``
    # increment bands are derived from a preceding step, so asking only the
    # earliest image risks calling a perfectly good band unknown.  A band that
    # exists at one end but not the other is reported, because sites early in
    # the record would silently get NaN for it.
    first_bands = set(ee.Image(collection.first()).bandNames().getInfo())
    last_bands = set(ee.Image(collection.sort("system:time_start", False).first()).bandNames().getInfo())
    available = first_bands | last_bands
    partial = sorted(available - (first_bands & last_bands))
    if partial:
        print(f"WARNING: band(s) not present throughout the record: {partial}")

    unknown = [v for v in variables if v not in available]
    if unknown:
        hints = {}
        for name in unknown:
            stem = name.replace("_hourly", "")
            near = sorted(b for b in available if stem in b or b in name)
            hints[name] = near[:4]
        raise ExtractionError(
            f"{len(unknown)} requested band(s) are not in {COLLECTION}: {unknown}\n"
            f"  closest available names: {hints}\n"
            f"  collection publishes {len(available)} bands"
        )
    print(f"all {len(variables)} requested bands exist in {COLLECTION}")


def get_monthly_chunks(start_date, end_date, step: pd.Timedelta = STEP):
    """Tile the CLOSED window ``[start_date, end_date]`` with half-open chunks.

    Returns ``[(chunk_start, chunk_end), ...]`` where each chunk is meant as
    ``[chunk_start, chunk_end)`` -- matching ``filterDate``, whose end instant
    is EXCLUDED.  Consecutive chunks share a bound, so they tile with neither
    overlap nor gap, and the last chunk ends one ``step`` past the last IMAGE
    INSTANT inside the window, so that the final image is requested and the one
    after it is not.

    Boundaries are normalised to midnight so that chunks after the first are
    month-aligned regardless of what time of day the record starts.
    """
    start = as_naive_utc(start_date)
    end = as_naive_utc(end_date)
    if end < start:
        raise ValueError(f"window ends before it starts: {start} .. {end}")

    # The last image inside the CLOSED window sits at ``end`` floored to the
    # collection's cadence -- not at ``end`` itself.  Stopping one step past
    # ``end`` instead admits the following image whenever ``end`` carries
    # minutes: measured on 2026-08-19 as exactly one surplus row at each of the
    # 139 sites whose end_date is not hour-aligned, and none at the other 87.
    stop = end.floor(step) + step
    chunks = []
    chunk_start = start
    while chunk_start < stop:
        if chunk_start.month == 12:
            boundary = chunk_start.replace(year=chunk_start.year + 1, month=1, day=1)
        else:
            boundary = chunk_start.replace(month=chunk_start.month + 1, day=1)
        boundary = boundary.normalize()
        chunk_end = min(boundary, stop)
        chunks.append((chunk_start, chunk_end))
        chunk_start = chunk_end
    return chunks


def _iso(ts: pd.Timestamp) -> str:
    """Full-precision bound for ``filterDate``; a date-only string floors to 00:00."""
    return ts.strftime("%Y-%m-%dT%H:%M:%S")




def assert_native_scale(collection) -> float:
    """Confirm ``ERA5_NATIVE_SCALE_M`` is still this collection's own pixel size.

    The constant is what gets sent; this only checks that it has not gone stale.
    Hardcoding the sampling scale is deliberate -- it is the one field in the
    request that silently changes WHICH PIXEL is read, so it belongs in the
    source where it can be reviewed and diffed, not in a number materialising
    out of a network call.  But a hardcoded scale that stops matching the product
    is exactly how ``scale=10000`` survived, so the run pays one round trip to
    check it and refuses if the two disagree.

    ``.select`` comes first: ERA5-Land's 69 bands do not all share one projection
    and ``Image.projection()`` refuses a multi-projection image.
    """
    reported = ee.Image(collection.first()).select(SCALE_PROBE_BAND).projection().nominalScale().getInfo()
    if reported is None:
        raise ExtractionError(f"{COLLECTION} would not report a nominal scale")
    reported = float(reported)
    if abs(reported - ERA5_NATIVE_SCALE_M) > SCALE_TOLERANCE_M:
        raise ExtractionError(
            f"{COLLECTION} reports a nominal scale of {reported:.3f} m, but this extractor "
            f"samples at ERA5_NATIVE_SCALE_M={ERA5_NATIVE_SCALE_M:.3f} m. The product's grid "
            "appears to have changed. Sampling off-grid does not fail, it silently reads a "
            "neighbouring pixel, so correct the constant rather than widening SCALE_TOLERANCE_M."
        )
    return reported


def _fetch_chunk(era5_dataset, point_fc, chunk_start, chunk_end, scale):
    """One chunk of hourly features, retried, then raised.

    Earth Engine returns transient 429/500s under concurrency, so a single
    failure must be retried -- but a persistent one must end the run.  The old
    code printed and ``continue``d, which turned "Earth Engine refused us" into
    "this site has no data for March".
    """
    filtered = era5_dataset.filterDate(_iso(chunk_start), _iso(chunk_end))

    def extract_values(image):
        timestamp = ee.Date(image.get("system:time_start"))
        values = image.reduceRegions(
            collection=point_fc,
            reducer=ee.Reducer.first(),
            scale=scale,
        )
        return values.map(lambda f: f.set("timestamp", timestamp))

    values = filtered.map(extract_values).flatten()

    last_error = None
    fault_attempts = 0
    quota_attempts = 0
    while True:
        try:
            return values.getInfo()
        except Exception as exc:  # noqa: BLE001 - re-raised below
            last_error = exc
            if _is_quota_error(exc):
                quota_attempts += 1
                if quota_attempts >= MAX_QUOTA_ATTEMPTS:
                    kind, spent = "quota refusal", quota_attempts
                    break
                _sleep_backoff(quota_attempts, quota=True)
            else:
                fault_attempts += 1
                if fault_attempts >= MAX_ATTEMPTS:
                    kind, spent = "error", fault_attempts
                    break
                _sleep_backoff(fault_attempts, quota=False)
    raise ExtractionError(
        f"chunk {_iso(chunk_start)} .. {_iso(chunk_end)} gave up after {spent} "
        f"{kind}(s) (quota={quota_attempts}, faults={fault_attempts}); "
        f"last error: {last_error}"
    ) from last_error


def _is_quota_error(exc):
    """True when Earth Engine refused for capacity, not for correctness."""
    text = str(exc).lower()
    return any(marker in text for marker in QUOTA_MARKERS)


def _sleep_backoff(attempt, quota):
    """Exponential for quota, linear for faults -- and jittered either way.

    The jitter is the load-bearing part.  ``time.sleep(BACKOFF_SECONDS * attempt)``
    put every worker on an identical schedule, so all of them woke together and
    re-triggered the very limit they were waiting out; the 2026-08-18 run
    degraded from 3 s/site to 70 s/site instead of recovering.  Sleeping a
    random 50-100% of the interval decorrelates them.
    """
    if quota:
        base = BACKOFF_SECONDS * (2 ** (attempt - 1))
    else:
        base = BACKOFF_SECONDS * attempt
    base = min(base, BACKOFF_CAP_SECONDS)
    time.sleep(random.uniform(base * 0.5, base))


def _cache_path(site, scale, variables):
    """Cache file for one site, with its inputs hashed INTO the name.

    The digest covers everything that changes the answer -- coordinates, window,
    sampling scale, band list, and ``CACHE_BEHAVIOUR_VERSION`` -- so altering any
    of them yields a different path and therefore a miss.  A stale cache cannot
    be silently reused; it can only be orphaned.  That property is the whole
    reason this is safe to add to a pipeline whose recent history is silent-reuse
    bugs, and the version token is what extends it to changes in this module:
    without one, a fix to the tiling leaves every input identical and every
    stale entry still hits.
    """
    payload = "|".join([
        CACHE_BEHAVIOUR_VERSION,
        str(site["site_name"]),
        "%.6f" % float(site["lat"]),
        "%.6f" % float(site["lon"]),
        str(site["start_date"]),
        str(site["end_date"]),
        "%.3f" % float(scale),
        ",".join(sorted(variables)),
    ])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return Path(CACHE_DIR) / ("%s.%s.csv" % (site["site_name"], digest))


def _cache_load(site, scale, variables, idx):
    """A previously completed site, or None.

    ``process_site`` either returns a site's whole window or raises, so anything
    on disk is complete by construction.  ``site_id`` is re-stamped from the
    current frame rather than trusted, since it is a row position and the site
    order is not part of the signature.
    """
    if not CACHE_DIR:
        return None
    path = _cache_path(site, scale, variables)
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["datetime"])
    missing = [var for var in variables if var not in frame.columns]
    if missing:
        raise ExtractionError(
            "cache %s is missing %d requested band(s): %s -- delete it and re-run"
            % (path, len(missing), ", ".join(missing))
        )
    frame["site_id"] = idx + 1
    return frame.to_dict("records")


def _cache_store(site, scale, variables, rows):
    """Persist one completed site, atomically."""
    if not CACHE_DIR or not rows:
        return
    path = _cache_path(site, scale, variables)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".csv.partial")
    pd.DataFrame(rows).to_csv(partial, index=False)
    os.replace(partial, path)


def process_site(site_data, variables, era5_dataset, scale):
    """Every hour of one site's window.  Raises rather than returning a short series."""
    idx, site = site_data

    cached = _cache_load(site, scale, variables, idx)
    if cached is not None:
        return cached

    site_results = []

    chunks = get_monthly_chunks(site["start_date"], site["end_date"])
    point = ee.Geometry.Point([site["lon"], site["lat"]])
    point_fc = ee.FeatureCollection([ee.Feature(point, {"site_id": str(idx)})])

    for chunk_start, chunk_end in chunks:
        data = _fetch_chunk(era5_dataset, point_fc, chunk_start, chunk_end, scale)
        features = (data or {}).get("features") or []
        if len(features) >= MAX_FEATURES_PER_CHUNK:
            raise ExtractionError(
                f"{site['site_name']}: chunk {_iso(chunk_start)} returned "
                f"{len(features)} features, at or past Earth Engine's "
                f"{MAX_FEATURES_PER_CHUNK}-element getInfo ceiling -- results "
                f"may be truncated.  Use finer chunks."
            )

        for feature in features:
            props = feature["properties"]
            stamp = (props.get("timestamp") or {}).get("value")
            if stamp is None:
                # Defaulting to 0 here silently stamped the row 1970-01-01.
                raise ExtractionError(
                    f"{site['site_name']}: a feature in chunk {_iso(chunk_start)} carries no timestamp property"
                )

            row = {
                "site_id": idx + 1,
                "site_name": site["site_name"],
                "latitude": site["lat"],
                "longitude": site["lon"],
                "datetime": pd.to_datetime(stamp, unit="ms"),
            }
            for var in variables:
                row[var] = props.get(var)
            site_results.append(row)

    _cache_store(site, scale, variables, site_results)
    return site_results


def extract_era5_data_parallel(sites_df, variables=None, max_workers=MAX_WORKERS):
    """Extract every site in parallel.  Any site that fails, fails the run."""
    variables = list(variables or VARIABLES)
    initialize_ee()

    era5_dataset = ee.ImageCollection(COLLECTION)
    validate_bands(era5_dataset, variables)
    available_from, available_to = collection_time_bounds(era5_dataset)
    print(f"{COLLECTION} publishes {available_from} .. {available_to}")

    # The scale is a constant; this one round trip only checks it is not stale.
    reported_scale = assert_native_scale(era5_dataset)
    scale = ERA5_NATIVE_SCALE_M
    print(
        f"sampling at {scale:.0f} m, ERA5-Land's native grid "
        f"(Earth Engine reports {reported_scale:.3f} m; was a hardcoded 10000)"
    )
    print(
        f"concurrency: {max_workers} worker(s); per chunk up to {MAX_ATTEMPTS} "
        f"fault retries and {MAX_QUOTA_ATTEMPTS} quota retries, jittered"
    )
    print(f"per-site cache: {CACHE_DIR}" if CACHE_DIR else "per-site cache: disabled")

    all_results = []
    failures = {}

    # ThreadPoolExecutor, not ProcessPoolExecutor: the Earth Engine client
    # holds credentials in process-global state.
    site_data_list = list(sites_df.iterrows())
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_site, site_data, variables, era5_dataset, scale): site_data
            for site_data in site_data_list
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Processing sites",
        ):
            idx, site = futures[future]
            try:
                all_results.extend(future.result())
            except Exception as exc:  # noqa: BLE001 - collected and raised below
                failures[site["site_name"]] = str(exc)

    if failures:
        lines = "\n".join(f"  {name}: {msg}" for name, msg in sorted(failures.items()))
        raise ExtractionError(f"{len(failures)} of {len(site_data_list)} site(s) failed:\n{lines}")

    if not all_results:
        raise ExtractionError("no data was retrieved for any site")

    return pd.DataFrame(all_results).sort_values(["site_id", "datetime"]), (
        available_from,
        available_to,
    )


def coverage_report(result_df, sites_df, available_from, available_to):
    """Per-site hourly coverage against the window the collection can actually serve.

    A site whose record runs past the collection's end is short for a reason
    that is not a defect, so the wanted window is the INTERSECTION of the site
    window and the collection's own span.  Anything missing inside that
    intersection is a real hole.
    """
    stamps = {
        name: set(pd.DatetimeIndex(group).round("h").astype("int64").tolist())
        for name, group in result_df.groupby("site_name")["datetime"]
    }

    rows = []
    for _, site in sites_df.iterrows():
        name = site["site_name"]
        start = max(as_naive_utc(site["start_date"]), available_from)
        end = min(as_naive_utc(site["end_date"]), available_to)
        if end < start:
            rows.append((name, 0, 0, 0, 0, "window outside collection"))
            continue
        want = set(pd.date_range(start.ceil("h"), end.floor("h"), freq="h").astype("int64").tolist())
        have = stamps.get(name, set())
        rows.append((name, len(have), len(want), len(want - have), len(have - want), ""))

    report = pd.DataFrame(rows, columns=["site_name", "have", "want", "missing", "extra", "note"])
    short = report[report["missing"] > 0]
    print("\n" + "=" * 70)
    print("COVERAGE")
    print("=" * 70)
    print(f"sites               : {len(report)}")
    print(f"hours wanted        : {int(report['want'].sum()):,}")
    print(f"hours extracted     : {int(report['have'].sum()):,}")
    print(f"hours MISSING       : {int(report['missing'].sum()):,}")
    print(f"hours outside window: {int(report['extra'].sum()):,}")
    print(f"sites short         : {len(short)}")
    if len(short):
        print(short.to_string(index=False))

    nulls = result_df[["temperature_2m"]].isna().sum().iloc[0] if "temperature_2m" in result_df else 0
    print(f"null temperature_2m : {int(nulls):,}")
    return report


def main():
    paths = get_default_paths()
    sites_df = pd.read_csv(paths.site_info_path)
    print(f"site_info: {paths.site_info_path} ({len(sites_df)} sites)")

    data, (available_from, available_to) = extract_era5_data_parallel(sites_df, VARIABLES)
    report = coverage_report(data, sites_df, available_from, available_to)

    out_path = Path(paths.era5_discrete_data_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a sibling first and promote only once coverage is verified, so a
    # short extraction never occupies the path the merge reads.  Same pattern as
    # src/site_info/assemble_extracted.py.
    partial = out_path.with_suffix(out_path.suffix + ".partial")
    data.to_csv(partial, index=False)

    if int(report["missing"].sum()) > 0:
        raise SystemExit(
            f"{int(report['missing'].sum()):,} hour(s) missing across "
            f"{int((report['missing'] > 0).sum())} site(s); left the incomplete "
            f"extraction at {partial} rather than promoting it to {out_path}"
        )

    os.replace(partial, out_path)
    print(f"\nwrote {len(data):,} rows to {out_path}")


if __name__ == "__main__":
    main()
