import os
import sys
from pathlib import Path

import ee
import pandas as pd

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.site_info.pft_resolution import igbp_code  # noqa: E402

LANDCOVER_COLLECTION = "MODIS/061/MCD12Q1"


def published_landcover_years():
    """Every calendar year for which MCD12Q1 publishes an image.

    Asked of Earth Engine rather than hardcoded: MODIS land cover is released
    with a multi-year lag, so any literal written here starts rotting the day
    it is committed.  A site window that runs past the last published year --
    the combined corpus reaches 2025-12-11 -- filtered an empty collection,
    and ``ee.Image(empty.first())`` is null, so ``.select('LC_Type1')`` failed
    with "Parameter 'input' is required and may not be null".

    One round trip for the whole run: the per-year availability test is then a
    local set lookup rather than another call to Earth Engine.
    """
    starts = ee.ImageCollection(LANDCOVER_COLLECTION).aggregate_array("system:time_start").getInfo()
    years = sorted({int(pd.to_datetime(ms, unit="ms").year) for ms in starts})
    if not years:
        raise RuntimeError(f"{LANDCOVER_COLLECTION} returned no images at all")
    return years


def landcover_row(site_name, year, download_year, lon, lat, landcover_value):
    """One site-year of land cover, carrying BOTH the raw code and its name.

    The short IGBP name is written HERE, at the generator, so the extracted
    product is self-describing and every consumer reads one vocabulary -- the
    one ``si_igbp`` already uses.  Before this, the only thing that turned
    ``LC_Type1`` into words was a long-name map inside the merge, in a frame
    that was never joined to anything; a long name substituted into the same
    column as ``si_igbp`` would have one-hot encoded into its own category.

    The raw integer is KEPT alongside the name, not replaced.  It is the only
    recoverable record of what Earth Engine returned, re-extraction costs one
    ``getInfo`` round trip per site-year, and the readers that still map the
    integer would coerce a text column to NaN *without raising*.

    Split out of the extraction loop so the naming can be tested without a GEE
    round trip -- the mapping was previously unreachable from any test.
    """
    return pd.DataFrame(
        {
            "site_name": [site_name],
            "year": [year],
            "landcover_source_year": [download_year],
            "lon": [lon],
            "lat": [lat],
            "landcover_type": [landcover_value],
            "pft": [igbp_code(landcover_value)],
        }
    )


def get_landcover_for_sites(csv_path, output_path):
    """
    Extracts landcover data for sites from a CSV file using Google Earth Engine.

    Args:
        csv_path (str): The file path for the input CSV file.
        output_path (str): The file path for the output CSV file.
    """
    try:
        # Initialize the Earth Engine library.
        # Stored credentials first: ee.Authenticate() can block on an
        # interactive consent flow, which never returns inside a batch job.
        try:
            ee.Initialize(project="era5download-447713")
        except Exception:
            ee.Authenticate()
            ee.Initialize(project="era5download-447713")
        print("Google Earth Engine initialized successfully!")
        print("Successfully initialized Google Earth Engine.")

        # A set, not a range: a gap in the middle of the record must not be
        # silently treated as available.
        available_years = set(published_landcover_years())
        first_year, last_year = min(available_years), max(available_years)
        print(f"{LANDCOVER_COLLECTION} publishes {first_year}..{last_year} ({len(available_years)} years)")

        # Read the site information from the CSV file.
        sites_df = pd.read_csv(csv_path)
        print(f"Reading site information from {csv_path}.")

        all_sites_data = []
        borrowed = {}

        # Iterate over each site in the DataFrame.
        for _, row in sites_df.iterrows():
            site_name = row["site_name"]

            # Keep datetime objects for calculations
            start_date_dt = pd.to_datetime(row["start_date"])
            end_date_dt = pd.to_datetime(row["end_date"])

            # Create formatted strings for Earth Engine
            start_date = start_date_dt.strftime("%Y-%m-%d")
            end_date = end_date_dt.strftime("%Y-%m-%d")

            lon = row["lon"]
            lat = row["lat"]
            site_years_data = []

            print(f"Processing site: {site_name}")

            # Define the point of interest.
            point = ee.Geometry.Point(lon, lat)

            # Use datetime objects for year calculations
            if end_date_dt.year > start_date_dt.year:
                download_years = range(start_date_dt.year, end_date_dt.year + 1)
            else:
                download_years = [start_date_dt.year]

            for year in download_years:
                # Borrow the nearest published year at BOTH ends.  The low-end
                # clamp was already here; without its high-end twin a site whose
                # record runs past the collection killed the whole run.  Land
                # cover enters the model as a static per-site attribute, so
                # borrowing the nearest year is the same treatment either way.
                download_year = min(max(year, first_year), last_year)
                if download_year != year:
                    borrowed.setdefault(site_name, []).append((year, download_year))
                # Select the MODIS landcover dataset.
                # This product is updated yearly.
                landcover_collection = ee.ImageCollection(LANDCOVER_COLLECTION).filterDate(
                    f"{download_year}-01-01", f"{download_year}-01-02"
                )

                # Checked locally against the year list fetched once at startup:
                # an empty collection here yields a null image, and the failure
                # then surfaces as an opaque "Parameter 'input' ... may not be
                # null" from .select() several lines later.
                if download_year not in available_years:
                    raise RuntimeError(
                        f"{LANDCOVER_COLLECTION} has no image for {download_year} "
                        f"(site {site_name}, requested year {year}); published "
                        f"years are {min(available_years)}..{max(available_years)}"
                    )

                # Get the landcover image for the period. We take the first available image.
                landcover_image = ee.Image(landcover_collection.first())

                # Extract the landcover value at the specified point.
                # We select the 'LC_Type1' band which is the primary land cover classification.
                landcover_band = landcover_image.select("LC_Type1")
                landcover_value = (
                    landcover_band.reduceRegion(
                        reducer=ee.Reducer.first(),
                        geometry=point,
                        scale=500,  # The resolution of the MODIS data is 500 meters.
                    )
                    .get("LC_Type1")
                    .getInfo()
                )

                # A point outside the image, or over water, reduces to None.
                # Left unchecked that became a NaN column for the whole
                # site-year, indistinguishable from a class the mapping does
                # not know.
                if landcover_value is None:
                    raise RuntimeError(
                        f"{LANDCOVER_COLLECTION} returned no LC_Type1 value for "
                        f"{site_name} ({lat}, {lon}) in {download_year} -- the "
                        f"point is outside the image or over masked water"
                    )

                # ONE ROW PER SITE-YEAR.  MCD12Q1 is an annual product, so an
                # hourly expansion stored ~8,760 identical copies of each value:
                # 5.4 M rows and 303 MB for ~615 distinct site-year values.  It
                # was also wrong at both ends -- `date_range(end="YYYY-12-31",
                # freq="h")` stops at 00:00, dropping the last 23 hours of every
                # December, and the first year's start was floored to midnight,
                # adding hours before the record began.  Measured over the
                # 226-site corpus: 13,279 hours net, 192 sites short and 33 over.
                #
                # Consumers reduce this to one class per site anyway; see
                # src/site_info/pft_resolution.py.
                site_years_data.append(
                    landcover_row(site_name, year, download_year, lon, lat, landcover_value)
                )

            all_sites_data.append(pd.concat(site_years_data, ignore_index=True))

        # Concatenate the data for all sites.
        final_df = pd.concat(all_sites_data, ignore_index=True)

        # Say out loud which site-years did not get their own land cover, so
        # this is a line in the log rather than a silent substitution.
        if borrowed:
            print(f"\nborrowed land cover from the nearest published year at {len(borrowed)} site(s):")
            for site in sorted(borrowed):
                pairs = borrowed[site]
                wanted = sorted({w for w, _ in pairs})
                used = sorted({u for _, u in pairs})
                print(f"  {site:22} {len(pairs):3} year(s) {wanted[0]}..{wanted[-1]} -> {used}")
        else:
            print("\nevery site-year had its own published land cover")

        # Save the results to a new CSV file.
        final_df.to_csv(output_path, index=False)
        print(f"Successfully exported landcover data to {output_path}.")

    except Exception as e:
        # Re-raise.  This handler wrapped the extraction *and* the write, so
        # any failure mid-run left no output file behind and still exited 0 --
        # a job that reports success and produces nothing.
        print(f"Landcover extraction FAILED: {e}")
        raise


if __name__ == "__main__":
    from path_config import get_default_paths

    paths = get_default_paths()
    # Define the path to your input CSV file.
    # Make sure to replace 'your_sites.csv' with the actual file name.
    input_csv_file = paths.site_info_path

    # Define the path for the output CSV file.
    output_csv_file = paths.pft_data_path
    if not os.path.exists(os.path.dirname(output_csv_file)):
        os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)

    get_landcover_for_sites(input_csv_file, output_csv_file)
