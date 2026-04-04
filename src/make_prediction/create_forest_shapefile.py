"""
Create a strict forest-only shapefile from MODIS PFT at 0.1° resolution.

Only includes grid cells where the dominant MODIS IGBP land cover class
is one of the 8 forest/woody PFT types used in the XGBoost training:
  ENF (1), EBF (2), DNF (3), DBF (4), MF (5), WSA (8), SAV (9), WET (11)

Usage
-----
    python create_forest_shapefile.py --year 2020 --output forest_pft_mask.shp
    python create_forest_shapefile.py --year 2020 --output forest_pft_mask.shp --resolution 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
from shapely.geometry import box

# IGBP PFT codes that match training data
FOREST_PFT_CODES = [1, 2, 3, 4, 5, 8, 9, 11]
FOREST_PFT_NAMES = {
    1: "ENF",
    2: "EBF",
    3: "DNF",
    4: "DBF",
    5: "MF",
    8: "WSA",
    9: "SAV",
    11: "WET",
}

# Spatial domain (matches ERA5-Land processing)
LAT_MIN, LAT_MAX = -60.0, 78.0
LON_MIN, LON_MAX = -180.0, 180.0


def get_pft_from_gee(year: int, resolution: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Download MODIS PFT data from GEE at the target resolution.

    Returns (pft_grid, lats, lons) where pft_grid is (nlat, nlon) int array.
    """
    ee.Initialize(project="era5download-447713")

    collection = "MODIS/061/MCD12Q1"
    img = (
        ee.ImageCollection(collection)
        .filter(ee.Filter.calendarRange(year, year, "year"))
        .first()
        .select("LC_Type1")  # IGBP classification
    )

    region = ee.Geometry.Rectangle([LON_MIN, LAT_MIN, LON_MAX, LAT_MAX])

    # Reduce to target resolution using mode (majority class)
    pft_img = img.reduceResolution(
        reducer=ee.Reducer.mode(),
        bestEffort=True,
        maxPixels=65536,
    ).reproject(crs="EPSG:4326", scale=resolution * 111320)  # deg → m at equator

    # Download as numpy array via getInfo (chunked for large regions)
    lats = np.arange(LAT_MAX, LAT_MIN, -resolution)
    lons = np.arange(LON_MIN, LON_MAX, resolution)

    print(f"Grid: {len(lats)} lat x {len(lons)} lon = {len(lats) * len(lons):,} cells")
    print("Downloading PFT from GEE (this may take several minutes)...")

    # Use ee.Image.sampleRectangle for smaller regions, or tile for global
    # For global at 0.1°, we tile by latitude bands
    band_height = 10  # degrees per band
    pft_grid = np.full((len(lats), len(lons)), fill_value=255, dtype=np.uint8)

    for lat_start in np.arange(LAT_MIN, LAT_MAX, band_height):
        lat_end = min(lat_start + band_height, LAT_MAX)
        band_region = ee.Geometry.Rectangle([LON_MIN, lat_start, LON_MAX, lat_end])

        lat_mask = (lats <= lat_end) & (lats > lat_start)
        if not lat_mask.any():
            continue

        try:
            result = pft_img.sample(
                region=band_region,
                scale=resolution * 111320,
                geometries=True,
            ).getInfo()

            for feat in result["features"]:
                coords = feat["geometry"]["coordinates"]
                lon, lat = coords[0], coords[1]
                pft_val = feat["properties"].get("LC_Type1", 255)

                lat_idx = int(round((LAT_MAX - lat) / resolution))
                lon_idx = int(round((lon - LON_MIN) / resolution))

                if 0 <= lat_idx < len(lats) and 0 <= lon_idx < len(lons):
                    pft_grid[lat_idx, lon_idx] = pft_val

            print(f"  Band [{lat_start:.0f}, {lat_end:.0f}]: OK")
        except Exception as e:
            print(f"  Band [{lat_start:.0f}, {lat_end:.0f}]: ERROR - {e}")

    return pft_grid, lats, lons


def get_pft_from_local_gee(year: int, resolution: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alternative: download PFT via GEE getDownloadURL as GeoTIFF tiles."""
    import tempfile

    import rasterio
    import requests

    ee.Initialize(project="era5download-447713")

    collection = "MODIS/061/MCD12Q1"
    img = ee.ImageCollection(collection).filter(ee.Filter.calendarRange(year, year, "year")).first().select("LC_Type1")

    lats = np.arange(LAT_MAX, LAT_MIN - resolution / 2, -resolution)
    lons = np.arange(LON_MIN, LON_MAX + resolution / 2, resolution)
    pft_grid = np.full((len(lats), len(lons)), fill_value=255, dtype=np.uint8)

    print(f"Grid: {len(lats)} lat x {len(lons)} lon")
    print("Downloading PFT tiles from GEE...")

    tile_size = 30  # degrees
    for lat_start in np.arange(LAT_MIN, LAT_MAX, tile_size):
        lat_end = min(lat_start + tile_size, LAT_MAX)
        for lon_start in np.arange(LON_MIN, LON_MAX, tile_size):
            lon_end = min(lon_start + tile_size, LON_MAX)

            region = ee.Geometry.Rectangle([lon_start, lat_start, lon_end, lat_end])
            try:
                url = img.getDownloadURL(
                    {
                        "region": region,
                        "scale": resolution * 111320,
                        "format": "GEO_TIFF",
                        "crs": "EPSG:4326",
                    }
                )
                resp = requests.get(url, timeout=300)
                resp.raise_for_status()

                with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                    tmp.write(resp.content)
                    tmp_path = tmp.name

                with rasterio.open(tmp_path) as src:
                    data = src.read(1)
                    transform = src.transform
                    for r in range(data.shape[0]):
                        for c in range(data.shape[1]):
                            y = transform.f + r * transform.e
                            x = transform.c + c * transform.a
                            li = int(round((LAT_MAX - y) / resolution))
                            lj = int(round((x - LON_MIN) / resolution))
                            if 0 <= li < len(lats) and 0 <= lj < len(lons):
                                pft_grid[li, lj] = data[r, c]

                Path(tmp_path).unlink(missing_ok=True)
                print(f"  Tile [{lat_start:.0f},{lon_start:.0f}]-[{lat_end:.0f},{lon_end:.0f}]: OK")
            except Exception as e:
                print(f"  Tile [{lat_start:.0f},{lon_start:.0f}]-[{lat_end:.0f},{lon_end:.0f}]: ERROR - {e}")

    return pft_grid, lats, lons


def create_forest_shapefile(
    pft_grid: np.ndarray, lats: np.ndarray, lons: np.ndarray, resolution: float, output_path: Path
) -> gpd.GeoDataFrame:
    """Create a dissolved shapefile from forest PFT grid cells."""
    print(f"\nFiltering to forest PFT codes: {FOREST_PFT_CODES}")
    forest_mask = np.isin(pft_grid, FOREST_PFT_CODES)
    n_forest = forest_mask.sum()
    n_total = (pft_grid != 255).sum()  # exclude NoData
    print(f"Forest cells: {n_forest:,} / {n_total:,} ({n_forest / max(n_total, 1) * 100:.1f}%)")

    # Create grid cell polygons for forest cells
    print("Creating grid cell polygons...")
    geometries = []
    pft_values = []
    pft_labels = []

    for i in range(len(lats)):
        for j in range(len(lons)):
            if forest_mask[i, j]:
                lat = lats[i]
                lon = lons[j]
                half = resolution / 2
                geom = box(lon - half, lat - half, lon + half, lat + half)
                geometries.append(geom)
                pft_values.append(int(pft_grid[i, j]))
                pft_labels.append(FOREST_PFT_NAMES.get(int(pft_grid[i, j]), "UNK"))

    print(f"Created {len(geometries):,} polygons")

    gdf = gpd.GeoDataFrame(
        {"pft_code": pft_values, "pft_name": pft_labels},
        geometry=geometries,
        crs="EPSG:4326",
    )

    # Dissolve into single multipolygon for efficient masking
    print("Dissolving into single geometry...")
    gdf_dissolved = gdf.dissolve()
    gdf_dissolved = gdf_dissolved.reset_index(drop=True)
    gdf_dissolved["name"] = "forest_pft_mask"

    # Snap bounds to safe range: clip to [-179, 179] to avoid GEE
    # antimeridian wrapping caused by float64 imprecision near +/-180
    SAFE_LON_MIN, SAFE_LON_MAX = -179.0, 179.0
    SAFE_LAT_MIN, SAFE_LAT_MAX = -90.0, 90.0
    from shapely.ops import clip_by_rect

    print("Clipping dissolved geometry to safe bounds [-179, 179] lon...")
    gdf_dissolved["geometry"] = gdf_dissolved["geometry"].apply(
        lambda geom: clip_by_rect(geom, SAFE_LON_MIN, SAFE_LAT_MIN, SAFE_LON_MAX, SAFE_LAT_MAX)
    )
    print(f"  Bounds after clip: {gdf_dissolved.total_bounds}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_dissolved.to_file(output_path)
    print(f"Saved to {output_path}")
    print(f"  Bounds: {gdf_dissolved.total_bounds}")

    # Also save undissolved version with PFT labels for reference
    ref_path = output_path.parent / (output_path.stem + "_by_pft" + output_path.suffix)
    gdf["geometry"] = gdf["geometry"].apply(
        lambda geom: clip_by_rect(geom, SAFE_LON_MIN, SAFE_LAT_MIN, SAFE_LON_MAX, SAFE_LAT_MAX)
    )
    gdf.to_file(ref_path)
    print(f"Saved PFT-labeled version to {ref_path}")

    return gdf_dissolved


def main():
    parser = argparse.ArgumentParser(description="Create forest-only shapefile from MODIS PFT via GEE")
    parser.add_argument("--year", type=int, default=2020, help="MODIS year (default: 2020)")
    parser.add_argument("--resolution", type=float, default=0.1, help="Grid resolution in degrees (default: 0.1)")
    parser.add_argument("--output", type=str, required=True, help="Output shapefile path")
    parser.add_argument(
        "--method",
        choices=["sample", "download"],
        default="download",
        help="GEE download method: 'sample' (slower, more robust) or 'download' (faster)",
    )
    args = parser.parse_args()

    if args.method == "sample":
        pft_grid, lats, lons = get_pft_from_gee(args.year, args.resolution)
    else:
        pft_grid, lats, lons = get_pft_from_local_gee(args.year, args.resolution)

    create_forest_shapefile(pft_grid, lats, lons, args.resolution, Path(args.output))


if __name__ == "__main__":
    main()
