"""MOD13Q1 EVI summaries over 2001-2020: mean and p90 - p10 of each year's clear composites, averaged across years,
and the annual harmonic of the whole series, which says how seasonal the stand is and when it peaks."""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from src.static_features.earth_engine import (
    BACKOFF_S,
    ShortReply,
    cell_collection,
    feature_rows,
    initialise,
    land_mask,
    mean_and_count,
    reduce_cells,
    with_all_band,
    with_retries,
)
from src.static_features.results import BY_YEAR_COLUMNS, NO_DATA, OK, ProductResult, RunContext

ASSET = "MODIS/061/MOD13Q1"
FIRST_YEAR = 2001
LAST_YEAR = 2020
BATCH_SIZE = 60
SCALE = 0.0001
COUNTED_CLASSES = (0, 1)  # good and marginal; 2 is snow or ice and 3 is cloud
SNOW_CLASS = 2
DORMANT_PERCENTILE = 10
MIN_COMPOSITES_PER_YEAR = 6
COMPOSITE_COLUMNS = ["site_name", "date", "qa_class", "mean_evi", "pixel_count", "land_pixels", "all_pixels"]
SUMMARIES = {"evi_mean": "year_mean", "evi_amplitude": "year_amplitude"}
HARMONIC_COLUMNS = ("evi_season_amplitude", "evi_season_r2", "evi_peak_cos_w", "evi_peak_sin_w")
DAYS_PER_YEAR = 365.25
COMPOSITE_CENTRE_DAYS = 8  # a 16-day composite is dated by its first day
PROVENANCE = {
    "source": ASSET,
    "version": "MOD13Q1.061",
    "checksum": None,
    "time_window": f"{FIRST_YEAR}-{LAST_YEAR}",
    "method": (
        "Unweighted land mean per composite and SummaryQA class on the product's own grid. A composite is valid with "
        "at least half its land pixels in classes 0 and 1, and its value is their pixel-weighted mean. A year counts "
        f"with at least {MIN_COMPOSITES_PER_YEAR} valid composites. Mean and p90 - p10 are taken within each year, "
        "then averaged across years, so boreal values describe the snow-free season. The harmonic is fitted by least "
        "squares to the same composites, c + a cos wt + b sin wt + a2 cos 2wt + b2 sin 2wt with w = 2 pi / 365.25 and t "
        "the composite's middle day of year: evi_season_amplitude is hypot(a, b), evi_season_r2 the share of variance "
        "both harmonics explain, and evi_peak_cos_w, evi_peak_sin_w are (a, b), the amplitude times the cosine and "
        "sine of the annual peak, so they fade to (0, 0) with the season. A composite of a counted year whose land is "
        "at least half snow enters the harmonic only, at the site's dormant level, the 10th percentile of its counted "
        "composites: a stand clear only in summer is otherwise extrapolated across winter (review, 2026-11-04). "
        "They replace the MCD12Q2 dates in the merge "
        "(owner, 2026-11-04): MCD12Q2 dates no cycle below an EVI2 amplitude of 0.1, so its dates were missing "
        "exactly where a stand is least seasonal."
    ),
}


def _class_rows(site: str, date: str, groups: list[dict]) -> list[tuple]:
    if not groups:  # kept, so the table records that this composite was asked for and held nothing
        return [(site, date, math.nan, math.nan, 0)]
    rows = []
    for group in groups:
        if group.get("mean") is None:
            raise ShortReply(f"{site} {date}: class {group.get('qa')} holds no mean")
        rows.append((site, date, int(group["qa"]), float(group["mean"]) * SCALE, int(group["count"])))
    return rows


def composites_from_reply(
    reply: dict, site_names: Sequence[str], dates: Sequence[str], cell_pixels: pd.DataFrame
) -> pd.DataFrame:
    """One row per site, composite and class. Every site must come back for every date, exactly once."""
    rows, seen = [], set()
    for properties in feature_rows(reply):
        key = (str(properties.get("site_name")), str(properties.get("date")))
        if key in seen:
            raise ShortReply(f"reply repeats {key}")
        seen.add(key)
        groups = properties.get("groups")
        if groups is None:
            raise ShortReply(f"{key}: reply holds no groups")
        rows.extend(_class_rows(key[0], key[1], groups))
    asked = {(name, date) for name in site_names for date in dates}
    missing, unasked = sorted(asked - seen), sorted(seen - asked)
    if missing or unasked:
        raise ShortReply(f"reply is missing {missing[:6]} and holds unasked {unasked[:6]}")
    table = pd.DataFrame(rows, columns=COMPOSITE_COLUMNS[:5])
    return table.assign(
        land_pixels=table["site_name"].map(cell_pixels["land_pixels"]).astype(int),
        all_pixels=table["site_name"].map(cell_pixels["all_pixels"]).astype(int),
    )[COMPOSITE_COLUMNS]


def _valid_composites(composites: pd.DataFrame, first_year: int, last_year: int) -> pd.DataFrame:
    """One row per composite that counts: the pixel-weighted mean of its clear classes, and its year."""
    counted = composites[composites["qa_class"].isin(COUNTED_CLASSES)]
    table = (
        counted.assign(evi_x_pixels=counted["mean_evi"] * counted["pixel_count"])
        .groupby(["site_name", "date", "land_pixels", "all_pixels"], as_index=False)[["evi_x_pixels", "pixel_count"]]
        .sum()
    )
    table = table.assign(
        mean_evi=table["evi_x_pixels"] / table["pixel_count"],
        year=pd.to_datetime(table["date"]).dt.year,
    )
    has_half = (table["pixel_count"] > 0) & (table["pixel_count"] * 2 >= table["land_pixels"])
    return table[has_half & table["year"].between(first_year, last_year)]


def _amplitude(series: pd.Series) -> float:
    return float(np.percentile(series, 90) - np.percentile(series, 10))


def _used_years(valid: pd.DataFrame) -> pd.DataFrame:
    """One row per site and year with enough composites, with that year's mean and amplitude."""
    rows = []
    for (site, year), group in valid.groupby(["site_name", "year"]):
        if len(group) >= MIN_COMPOSITES_PER_YEAR:
            rows.append((site, year, float(group["mean_evi"].mean()), _amplitude(group["mean_evi"])))
    return pd.DataFrame(rows, columns=["site_name", "year", "year_mean", "year_amplitude"])


def seasonal_fit(day: np.ndarray, evi: np.ndarray) -> dict[str, float]:
    """The annual and semi-annual harmonics of one series by least squares; see PROVENANCE.

    A flat series has R2 0. Its sums of squares are rounding noise, so it is told by max == min, not by them.
    """
    angle = 2 * np.pi * np.asarray(day, dtype=float) / DAYS_PER_YEAR
    evi = np.asarray(evi, dtype=float)
    design = np.column_stack(
        [np.ones_like(angle), np.cos(angle), np.sin(angle), np.cos(2 * angle), np.sin(2 * angle)]
    )
    coefficients, *_ = np.linalg.lstsq(design, evi, rcond=None)
    total = float(np.sum((evi - evi.mean()) ** 2))
    residual = float(np.sum((evi - design @ coefficients) ** 2))
    return {
        "evi_season_amplitude": float(np.hypot(coefficients[1], coefficients[2])),
        "evi_season_r2": 0.0 if np.ptp(evi) == 0.0 else min(max(1.0 - residual / total, 0.0), 1.0),
        "evi_peak_cos_w": float(coefficients[1]),
        "evi_peak_sin_w": float(coefficients[2]),
    }


def _centre_days(dates: pd.Series) -> np.ndarray:
    return pd.to_datetime(dates).dt.dayofyear.to_numpy(dtype=float) + COMPOSITE_CENTRE_DAYS


def _snow_composites(composites: pd.DataFrame, valid: pd.DataFrame, used: pd.DataFrame) -> pd.DataFrame:
    """Site and date of each composite of a counted year whose land is at least half snow and that is not valid."""
    snow = composites[composites["qa_class"] == SNOW_CLASS]
    snow = snow[snow["pixel_count"] * 2 >= snow["land_pixels"]]
    snow = snow.assign(year=pd.to_datetime(snow["date"]).dt.year).merge(used[["site_name", "year"]], on=["site_name", "year"])
    clear = valid[["site_name", "date"]].drop_duplicates()
    sides = snow[["site_name", "date"]].merge(clear, on=["site_name", "date"], how="left", indicator=True)
    return sides.loc[sides["_merge"] == "left_only", ["site_name", "date"]].drop_duplicates()


def _harmonics(in_used: pd.DataFrame, snow: pd.DataFrame) -> pd.DataFrame:
    """One row per site with a counted year: its counted composites, and its snow composites at the dormant level."""
    winter = {site: _centre_days(group["date"]) for site, group in snow.groupby("site_name")}
    fits = {}
    for site, group in in_used.groupby("site_name"):
        evi = group["mean_evi"].to_numpy(dtype=float)
        snow_days = winter.get(site, np.empty(0))
        dormant = np.full(snow_days.size, np.percentile(evi, DORMANT_PERCENTILE))
        fits[site] = seasonal_fit(
            np.concatenate([_centre_days(group["date"]), snow_days]), np.concatenate([evi, dormant])
        )
    return pd.DataFrame(list(fits.values()), index=list(fits), columns=list(HARMONIC_COLUMNS))


def _by_year(index: pd.Index, used: pd.DataFrame, first_year: int, last_year: int) -> pd.DataFrame:
    every_year = pd.MultiIndex.from_product([index, range(first_year, last_year + 1)], names=["site_name", "year"])
    per_year = used.set_index(["site_name", "year"]).reindex(every_year)
    tables = []
    for column, source in SUMMARIES.items():
        table = per_year[source].rename("value").reset_index()
        tables.append(table.assign(column=column, status=np.where(table["value"].notna(), OK, NO_DATA)))
    return pd.concat(tables, ignore_index=True)[list(BY_YEAR_COLUMNS)]


def summarise(composites: pd.DataFrame, first_year: int = FIRST_YEAR, last_year: int = LAST_YEAR) -> ProductResult:
    index = pd.Index(composites["site_name"].drop_duplicates(), name="site_name")
    valid = _valid_composites(composites, first_year, last_year)
    used = _used_years(valid)
    by_site = used.groupby("site_name")
    years_used = by_site.size().reindex(index, fill_value=0)
    has_year = years_used > 0

    in_used = valid.merge(used[["site_name", "year"]], on=["site_name", "year"])
    share = (in_used["pixel_count"] / in_used["all_pixels"]).groupby(in_used["site_name"]).mean()
    coverage = share.reindex(index, fill_value=0.0)
    harmonics = _harmonics(in_used, _snow_composites(composites, valid, used)).reindex(index)
    fitted = {column: harmonics[column] for column in HARMONIC_COLUMNS}
    return ProductResult(
        product="evi",
        values=pd.DataFrame(
            {
                "evi_mean": by_site["year_mean"].mean().reindex(index),
                "evi_amplitude": by_site["year_amplitude"].mean().reindex(index),
                "evi_years_used": years_used,
                **fitted,
            }
        ),
        status=pd.DataFrame(
            {
                "evi_mean": np.where(has_year, OK, NO_DATA),
                "evi_amplitude": np.where(has_year, OK, NO_DATA),
                "evi_years_used": OK,
                **{column: np.where(has_year, OK, NO_DATA) for column in HARMONIC_COLUMNS},
            },
            index=index,
        ),
        coverage=pd.DataFrame(
            {"evi_mean": coverage, "evi_amplitude": coverage, **dict.fromkeys(HARMONIC_COLUMNS, coverage)}
        ),
        by_year=_by_year(index, used, first_year, last_year),
        extras={"evi_composites": composites},
        provenance=PROVENANCE,
    )


def _request_composites(
    images: Any, land: Any, projection: Any, part: pd.DataFrame, dates: list[str], cell_pixels: pd.DataFrame
) -> pd.DataFrame:
    # The request and the check of its reply are one attempt, so a short reply is retried like a dropped one.
    reducer = mean_and_count().group(groupField=1, groupName="qa")
    collection = cell_collection(part)

    def per_image(image: Any) -> Any:
        pair = image.select(["EVI", "SummaryQA"]).updateMask(land)
        return pair.reduceRegions(collection=collection, reducer=reducer, crs=projection).map(
            lambda feature: feature.set("date", image.date().format("YYYY-MM-dd"))
        )

    reply = images.map(per_image).flatten().getInfo()
    return composites_from_reply(reply, list(part["site_name"]), dates, cell_pixels)


def _composite_dates(images: Any, label: str) -> list[str]:
    starts = with_retries(images.aggregate_array("system:time_start").getInfo, f"{label} dates", backoff_s=BACKOFF_S)
    return pd.to_datetime(starts, unit="ms").strftime("%Y-%m-%d").tolist()


def extract(context: RunContext) -> ProductResult:
    import ee

    initialise(context.project)
    collection = ee.ImageCollection(ASSET)
    projection = collection.first().select("EVI").projection()
    land = land_mask()
    pixels = reduce_cells(
        with_all_band(land.selfMask()),
        mean_and_count(),
        projection,
        context.sites,
        ["land_count", "all_count"],
        ["land_mean"],
        "evi land pixels",
    )
    cell_pixels = pixels.rename(columns={"land_count": "land_pixels", "all_count": "all_pixels"})
    frames = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        images = collection.filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        dates = _composite_dates(images, f"evi {year}")
        for start in range(0, len(context.sites), BATCH_SIZE):
            part = context.sites.iloc[start : start + BATCH_SIZE]
            request = functools.partial(_request_composites, images, land, projection, part, dates, cell_pixels)
            frames.append(with_retries(request, f"evi {year} sites {list(part['site_name'])}", backoff_s=BACKOFF_S))
    return summarise(pd.concat(frames, ignore_index=True))
