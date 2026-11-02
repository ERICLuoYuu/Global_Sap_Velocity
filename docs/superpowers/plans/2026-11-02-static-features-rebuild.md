# Static Features Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `src/static_features/`, which extracts every static feature for the 226 `sapwood_combined` sites on one 1 km rule, adds ten new datasets, gives every value a reason code, and reaches the merge only through an opt-in switch.

**Architecture:** Each product is one module with `extract(context) -> ProductResult`, run as its own job and saved atomically as a part. `assemble` joins the parts into the output files and refuses stale ones. The arithmetic lives in pure functions tested with hand-computed answers; thin readers around them (Earth Engine, curl, rasterio, h5py, netCDF4) are checked by live tests and by acceptance numbers measured while opening the data.

**Tech Stack:** Python 3.9 on Palma, pandas, numpy, earthengine-api, rasterio, pyproj, h5py, netCDF4, xarray (through the existing Fan reader), pytest, SLURM, curl.

**Spec:** `docs/superpowers/specs/2026-10-30-static-features-rebuild-design.md`

## Global Constraints

- Palma is primary. The code lives in worktree `/scratch/tmp/yluo2/gsv-wt/static-features` (`$W`), branch `feat/static-features`. Commit and push only from Palma, never from the local machine.
- No Python on the login node. Everything, pytest included, runs through `sbatch`, on `/scratch/tmp/yluo2/gsv/.venv` (Python 3.9.25).
- Every module: `from __future__ import annotations`; no `zip` at all (the local linter hook adds `strict=True`, which 3.9 rejects), no `match`, no bare `assert` outside tests, `import ee` only inside functions, no work at import.
- Run modules with `python -m` from `$W` with `PYTHONPATH=$W`. A script run by path from the main clone silently imports the main clone's `src`.
- Nothing already on disk is overwritten. Output goes under `data/raw/extracted_data/static_1km/<scale>/`, raw data in new directories under `data/raw/grided/`.
- The merge's default behaviour does not change.
- Status strings are exactly `ok`, `neighbour`, `no_data`. A request failure is never a status: it raises `ExtractionFailure` naming site and product, and stops the run.
- Grid: `col = floor((lon + 180) * 120)`, `row = floor((90 - lat) * 120)`. A native pixel belongs to a cell when its centre lies in `low <= c < high`. Every Earth Engine reducer is `.unweighted()`.
- Earth Engine project `era5download-447713`, overridable by `--project`, never `ee-yuluo-2`. One Earth Engine product job at a time (Restricted Mode).
- Downloads use `curl` only, each dataset in its own new directory, and are verified by md5 or size before use. Nothing executes from inside a data directory.
- SSH only through `/c/Windows/System32/OpenSSH/ssh.exe` and `scp.exe`, with `-o "MACs=hmac-sha2-256,hmac-sha2-512" -i "/c/Users/Ab Leo/.ssh/id_ecdsa_palma" -o IdentitiesOnly=yes`. Poll at 5 minutes or longer. A failed SSH call is not a failed job.
- Run `/verify` before each code commit. Commit messages are conventional (`feat(static): ...`) with no attribution lines, per the owner's global rule.

**Shorthand used below.** Files are written locally, copied to `$W` with `scp.exe`, read back for linter edits, and run there.

- `T <args>`: `sbatch --wait $W/src/static_features/jobs/job_pytest.sh <args>`, then read `### pytest exit: N` from the log. A bare `test_x.py` means `src/static_features/tests/test_x.py`, and `T .` means that directory. Live tests add `STATIC_LIVE_TESTS=1` and `--partition=normal`, where Earth Engine access is verified.
- `M <module> [args]`: `sbatch $W/src/static_features/jobs/job_module.sh <module> --scale sapwood_combined [args]`, then read `### module exit: N`. Every module's `main` takes `--scale`. Resources are set on the `sbatch` line, and any queue wait goes through a background monitor.
- `P <product> [flags]`: `M src.static_features.run --product <product> [flags]`.

## Review Focus

1. **Earth Engine answers but leaves something out** (a site missing from the reply, a mean that is null although pixels were counted). Expected: retried, then stops naming the sites. Never NaN. Tests in Task 4.
2. **A job dies mid-product** (walltime, out of memory, or a request failing at site 200 of 226). Expected: no part appears, an earlier good part stays intact, and a rerun starts clean. Tests in Task 2.
3. **The site list changes after some parts were made.** Expected: `assemble` refuses and names the stale parts, rather than joining old values to new sites. Test in Task 12.
4. **Two sites share a cell, or one name appears twice** (`GBR_DEV_CON` and `GBR_DEV_DRO` share a cell). Expected: each site gets its own row with identical values, while a duplicated name stops at load. Tests in Tasks 2 and 4.
5. **The merge's switch points at an incomplete root** (a file absent, a site absent from a file). Expected: the merge stops naming the file or site, rather than writing NaN columns. Tests in Task 14.

---

## File map

| Path under `src/static_features/` | Responsibility |
|---|---|
| `grid.py` | Cell of a coordinate, bounds of a cell |
| `sites.py` | Load and validate the site list, site-list hash |
| `results.py` | Status codes, `ExtractionFailure`, `RunContext`, `ProductResult`, `validate`, atomic parts |
| `run.py` | CLI, product registry, runs one product |
| `rasters.py` | Native pixel with reach, pixel span of a cell, window readers |
| `earth_engine.py` | Retries, cell geometry, reducers, sampling, reply validation, land mask |
| `fetch.py` | `curl` download with verification, Zenodo manifest, checksums |
| `products/*.py` | One module per product, each exposing `extract(context)` and, if it downloads, `fetch_inputs(context)` |
| `assemble.py` | Parts to output files, provenance |
| `checks/{reproduce,compare,ranges,agreement,merge_parity}.py` | The spec's checks, each a `main` writing to `<output_root>/checks/` |
| `tests/`, `jobs/` | Tests; `job_pytest.sh`, `job_module.sh`, `job_merge_static.sh` |

Raw data directories under `data/raw/grided/`: `han_2025`, `yang_2024`, `lusk_2026`, `liang_2022`, `lesiv_2022`, `esa_cci_biomass_v7`, `gami_v2_1`.

---

### Task 1: Snapshot, worktree, skeleton, and guards

**Files:**
- Commit in main clone: 6 modified extractors, untracked files in `src/Extractors`, `src/serving/`, `src/site_info/`
- Create: `src/static_features/{__init__,products/__init__,checks/__init__,tests/__init__}.py`, `jobs/job_pytest.sh`, `jobs/job_module.sh`
- Test: `tests/test_module_hygiene.py`

**Interfaces:**
- Produces: `job_pytest.sh <pytest args>` (express, 8 G, 30 min, prints `### pytest exit: N`), `job_module.sh <module> [args]` (normal, 8 G, 2 h by default, runs `python -u -m <module> [args]` from `$W`, prints `### module exit: N`). Both log to `/scratch/tmp/yluo2/static_logs/%x_%j.{out,err}` and start from the preamble of `/scratch/tmp/yluo2/gsv/job_ann_pytest.sh` (untracked, so absent from the worktree) with `$W` paths.

- [ ] **Step 1: List the snapshot.** On Palma: `git status --porcelain src/Extractors src/serving src/site_info`. Expected: the six modified extractors, untracked code, and backups matching `*.pre*`, which stay out.

- [ ] **Step 2: Commit and push the snapshot** with explicit paths, excluding those backups, on `feat/sapwood-combined`: `chore: snapshot Palma's extractor, serving and site_info code before the static rebuild`, then `git push origin feat/sapwood-combined`. Expected: a rerun of Step 1 lists only backups.

- [ ] **Step 3: Create the worktree.**

```bash
git worktree add -b feat/static-features /scratch/tmp/yluo2/gsv-wt/static-features feat/sapwood-combined
cd /scratch/tmp/yluo2/gsv-wt/static-features && rm -rf data outputs
G=/scratch/tmp/yluo2/gsv
ln -s $G/data data; ln -s $G/outputs outputs; ln -sf $G/path_config.py path_config.py
ln -sfn $G/src/hyperparameter_optimization/JSON src/hyperparameter_optimization/JSON; ln -sfn $G/configs configs
mkdir -p /scratch/tmp/yluo2/static_logs
```

- [ ] **Step 4: Write the failing hygiene test**, modelled on `src/sapwood_combined/tests/test_module_hygiene.py`, walking every `.py` under `src.static_features` with `ast.parse(source, feature_version=(3, 9))`:

```python
def test_every_module_uses_future_annotations(): ...
def test_no_zip_anywhere(): ...              # catches the linter's strict=True
def test_no_bare_assert_outside_tests(): ...
def test_ee_is_never_imported_at_module_level(): ...
def test_modules_do_no_work_at_import(): ... # top level holds only imports, defs, classes, constants, __main__ guard
```

- [ ] **Step 5: Run it.** The job scripts must exist first; submit `T test_module_hygiene.py`. Expected: FAIL with `ModuleNotFoundError: src.static_features`.

- [ ] **Step 6: Create the package skeleton.** Empty `__init__.py` files.

- [ ] **Step 7: Run it.** `T test_module_hygiene.py`. Expected: `### pytest exit: 0`.

- [ ] **Step 8: Commit** the skeleton, jobs, test, and copies of this plan and the spec under `docs/superpowers/`: `feat(static): package skeleton, jobs and import guards`.

---

### Task 2: Grid, sites, results, and the product runner

**Files:**
- Create: `grid.py`, `sites.py`, `results.py`, `run.py`
- Test: `tests/test_grid.py`, `test_sites.py`, `test_results.py`, `test_run.py`

**Interfaces:**
- Produces:
  - `grid.cell_of(lat: float, lon: float) -> Tuple[int, int]` returning `(row, col)`; `grid.cell_bounds(row: int, col: int) -> Tuple[float, float, float, float]` returning `(west, south, east, north)`.
  - `sites.load_sites(site_info_csv: Path, site_md_dir: Path) -> pd.DataFrame` with columns `site_name, lat, lon, start_date, end_date, si_igbp, row, col`; `sites.site_list_hash(sites) -> str`, sha256 of sorted `site_name,lat,lon`.
  - `results.OK, NEIGHBOUR, NO_DATA`; `class ExtractionFailure(RuntimeError)`.
  - `results.RunContext(sites, raw_root: Path, output_root: Path, project: str)`, frozen, with `RunContext.for_scale(scale, project) -> RunContext` using `path_config.get_default_paths(scale)`: `site_info_path`, `metadata_dir`, `base_data_dir / "raw" / "grided"`, `extracted_data_dir / "static_1km" / scale`.
  - `results.ProductResult(product: str, values: DataFrame, status: DataFrame, coverage: DataFrame, by_year: Optional[DataFrame], extras: Dict[str, DataFrame], provenance: Dict[str, Any])`, frozen. `values` and `status` are indexed by `site_name` with the same columns; `coverage` holds a subset of those columns as floats in `[0, 1]`; `by_year` has columns `site_name, year, column, value, status`.
  - `results.validate(result, site_names) -> None`; `results.write_part(result, parts_root, site_hash) -> Path`; `results.read_part(parts_root, product) -> Tuple[ProductResult, str]`.
  - `run.PRODUCTS: Dict[str, str]`, name to module path, for all 16 products, imported only on use; `run.run_product(name, context, registry=PRODUCTS) -> Path`; `run.main(argv=None) -> int` with `--product`, `--scale`, `--project`, `--fetch-only`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_cell_of_hand_computed():
    assert cell_of(90.0, -180.0) == (0, 0)
    assert cell_of(51.5, 10.25) == (4620, 22830)
    assert cell_of(-90.0, 180.0) == (21599, 43199)     # the far edges clamp into the last cell

def test_cell_bounds_inverts_cell_of():
    assert cell_bounds(4620, 22830) == pytest.approx((10.25, 51.5 - 1 / 120, 10.25 + 1 / 120, 51.5))

def test_cell_of_refuses_outside_earth_and_nan(): ...     # lat 91, lon 181, NaN -> ValueError

# Review Focus 4
def test_load_sites_refuses_duplicate_names(tmp_path): ...  # ValueError naming the site
def test_load_sites_adds_cell_and_igbp(tmp_path): ...
def test_load_sites_refuses_missing_metadata(tmp_path): ...

def test_validate_refuses_nan_with_ok_status(): ...
def test_validate_refuses_value_with_no_data_status(): ...
def test_validate_refuses_unknown_status(): ...
def test_validate_refuses_missing_extra_or_repeated_sites(): ...
def test_validate_refuses_coverage_outside_unit_interval(): ...

def test_part_round_trips_floats_exactly(tmp_path):
    # to_csv with read_csv's default parser moves floats by one ulp; read with float_precision="round_trip"
    ...                                                   # 0.1 + 0.2 survives exactly

# Review Focus 2
def test_write_part_failure_keeps_previous_part(tmp_path, monkeypatch): ...
def test_failed_product_writes_no_part(tmp_path): ...     # fake extract raises ExtractionFailure at site 200
def test_run_product_refuses_invalid_result(tmp_path): ...
def test_main_returns_nonzero_on_failure(tmp_path): ...
```

- [ ] **Step 2: Run them.** `T test_grid.py test_sites.py test_results.py test_run.py`. Expected: FAIL, modules missing.

- [ ] **Step 3: Implement.** `write_part` builds `parts_root/.tmp_<product>_<pid>/` (`values.csv`, `status.csv`, `coverage.csv`, `by_year.csv`, `<extra>.csv`, `provenance.json`, `site_hash.txt`), then swaps it for `parts_root/<product>/` by rename, and removes the temp directory on failure. `run_product` calls `validate` before `write_part`. `--fetch-only` calls the module's `fetch_inputs(context)` and writes no part. `extract` and `fetch_inputs` take a `RunContext`.

- [ ] **Step 4: Run them.** Same command. Expected: `### pytest exit: 0`.

- [ ] **Step 5: Commit.** `feat(static): cell grid, site list, result contract and product runner`.

---

### Task 3: Local raster primitives

**Files:**
- Create: `rasters.py`
- Test: `tests/test_rasters.py`

**Interfaces:**
- Consumes: `results.OK, NEIGHBOUR, NO_DATA`.
- Produces:
  - `rasters.PixelHit(value: float, status: str, row: Optional[int], col: Optional[int])`, frozen.
  - `rasters.pixel_with_reach(read_block: Callable[[int, int, int, int], np.ndarray], shape: Tuple[int, int], row: int, col: int, reach: int, is_valid: Callable[[np.ndarray], np.ndarray], wrap_cols: bool = True) -> PixelHit`, where `read_block(row_start, row_stop, col_start, col_stop)` returns a 2-D array.
  - `rasters.centred_span(low: float, high: float, origin: float, step: float, size: int) -> Tuple[int, int]`, pixel index range whose centres lie in `low <= c < high`; `origin` is the outer edge of pixel 0, `step` may be negative.
  - `rasters.tiff_reader(src, band: int) -> Callable` returning a `read_block` for an open rasterio dataset.

- [ ] **Step 1: Write the failing tests** on small arrays where `-1` is invalid.

```python
def test_valid_own_pixel_is_ok_even_with_valid_neighbours(): ...       # status OK, the own value
def test_nearest_beats_the_corner(): ...      # own invalid; (-1,-1) and (0,-1) valid -> (0,-1), the CHN_YUN_YUN defect
def test_tie_keeps_first_in_row_then_col_order(): ...                  # (-1,0) and (1,0) valid -> (-1,0), as extract_raster_value does
def test_reach_limits_the_search(): ...       # only valid pixel 2 away, reach 1 -> NO_DATA, NaN, row None
def test_wraps_the_dateline(): ...            # col 0, valid at last col: wrap True -> NEIGHBOUR, wrap False -> NO_DATA
def test_clips_at_the_poles(): ...            # row 0, reach 1, no error, no wrap in rows

def test_centred_span_hand_computed():
    assert centred_span(0.3, 1.0, 0.0, 0.25, 8) == (1, 4)       # centres .375 .625 .875
    assert centred_span(0.3, 1.0, 1.0, -0.25, 8) == (0, 3)      # descending axis, same centres
    assert centred_span(0.125, 0.375, 0.0, 0.25, 8) == (0, 1)   # low edge in, high edge out
    assert centred_span(0.13, 0.14, 0.0, 0.25, 8)[0] == centred_span(0.13, 0.14, 0.0, 0.25, 8)[1]  # none
```

- [ ] **Step 2: Run them.** `T test_rasters.py`. Expected: FAIL, module missing.

- [ ] **Step 3: Implement.** Rank candidates by squared pixel distance, keeping the first in row-then-column order on a tie; that is the same order as `extract_raster_value`'s `math.hypot`. When the window crosses the column edge and `wrap_cols` is set, read two blocks and join them.

- [ ] **Step 4: Run them.** Expected: `### pytest exit: 0`.

- [ ] **Step 5: Commit.** `feat(static): native pixel lookup and cell pixel span`.

---

### Task 4: Earth Engine core

**Files:**
- Create: `earth_engine.py`
- Test: `tests/test_earth_engine.py` (pure), `tests/test_earth_engine_live.py` (skipped unless `STATIC_LIVE_TESTS=1`)

**Interfaces:**
- Consumes: `results.ExtractionFailure`, `grid.cell_bounds`.
- Produces:
  - `with_retries(call, label: str, attempts: int = 6, backoff_s: float = 15.0, transient=None, sleep=time.sleep)`: `transient` defaults to `(ee.EEException, ShortReply)`; waits `backoff_s * attempt * jitter`; raises `ExtractionFailure` containing `label` once attempts run out.
  - `class ShortReply(Exception)`.
  - `cells_from_reply(reply: dict, site_names, counts: Sequence[str], values: Sequence[str]) -> pd.DataFrame`, indexed by `site_name`; `counts` must be present per site, `values` may be null or absent only when their paired count is 0 (pairing `<band>_mean` or `<band>_p95` with `<band>_count`).
  - `pixels_from_reply(reply: dict, site_names, columns) -> pd.DataFrame`, one row per pixel.
  - `check_bands(requested, available, label) -> None`.
  - `initialise(project: str) -> None`, `cell_collection(sites)`, `mean_and_count()`, `with_all_band(image)` (adds `constant(1).rename("all")`, so names stay `<band>_mean` and coverage is `<band>_count / all_count`).
  - `reduce_cells(image, reducer, projection, sites, counts, values, label, batch_size=60) -> pd.DataFrame`.
  - `sample_cells(image, projection, sites, columns, label, batch_size=60) -> pd.DataFrame`.
  - `land_mask()`: band `land`, 1 land, 0 water, read on WorldCover's own grid.

Verified call shapes: `ee.Geometry.Rectangle([w, s, e, n], "EPSG:4326", False)`; `image.reduceRegions(collection, reducer, crs=<Projection>)`; `image.sampleRegions(collection, properties, projection, geometries=False)`; `ee.Reducer.mean().unweighted().combine(ee.Reducer.count().unweighted(), sharedInputs=True)`. Land mask: `ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")`, then `.neq(80).unmask(0).rename("land").reproject(<its projection>)`.

- [ ] **Step 1: Write the failing pure tests.**

```python
def test_with_retries_recovers(): ...                      # two failures then success; two sleeps
def test_with_retries_stops_after_six_with_label(): ...    # 6 calls; ExtractionFailure containing the label
def test_with_retries_lets_programming_errors_through(): ...   # TypeError on the first call, not retried
def test_check_bands_refuses_unknown_band(): ...

# Review Focus 1
def test_reply_missing_a_site_is_short(): ...              # ShortReply naming the site
def test_reply_with_an_unasked_site_is_short(): ...
def test_reply_without_count_is_short(): ...
def test_reply_null_mean_with_pixels_is_short(): ...       # elevation_m_count=13, elevation_m_mean=None
def test_reply_null_mean_with_no_pixels_is_nan(): ...      # elevation_m_count=0 -> NaN, count 0.0
def test_pixels_reply_refuses_site_with_no_pixels(): ...

# Review Focus 4
def test_sites_sharing_a_cell_each_get_a_row(): ...
```

- [ ] **Step 2: Run them.** `T test_earth_engine.py`. Expected: FAIL, module missing.

- [ ] **Step 3: Implement.** Every request's `getInfo` and its reply check run inside one `with_retries` call, so a short reply is retried like a transient error. Check band names once per image with `check_bands` before the first request.

- [ ] **Step 4: Run them.** Expected: `### pytest exit: 0`.

- [ ] **Step 5: Write the live tests**, with sites given as literal coordinates read from `site_info.csv`:

```python
def test_live_land_share_matches_measured(): ...   # GBR_DEV_CON 0.37 +- 0.02, CHE_DAV_SEE 0.30 +- 0.02, AUS_WOM > 0.99
def test_live_count_counts_centres_not_touches(): ...   # MOD44B at AUS_WOM: about 13 centres, not the 22 touched
def test_live_mask_reads_its_own_grid(): ...       # land at five points through MODIS's grid equals Map read natively
```

- [ ] **Step 6: Run them.** `STATIC_LIVE_TESTS=1 T test_earth_engine_live.py`. Expected: `### pytest exit: 0`. If a measured number moves, stop and report it rather than widening the tolerance.

- [ ] **Step 7: Commit.** `feat(static): Earth Engine retries, reply checks, reducers and land mask`.

---

### Task 5: Terrain, canopy, soil

**Files:**
- Create: `products/terrain.py`, `products/canopy.py`, `products/soil.py`
- Test: `tests/test_terrain.py`, `test_canopy.py`, `test_soil.py`

**Interfaces:**
- Consumes: Task 4's `reduce_cells`, `mean_and_count`, `with_all_band`, `land_mask`, `check_bands`.
- Produces: `extract(context) -> ProductResult` in each, plus pure builders the tests call:
  - `terrain.terrain_result(cells: DataFrame, land: DataFrame) -> ProductResult`, columns `elevation_m, slope_deg, aspect_sin, aspect_cos, land_share`; `terrain.aspect_vector(aspect_deg, slope_deg)`, an `ee.Image` of bands `aspect_sin, aspect_cos`.
  - `canopy.canopy_result(cells) -> ProductResult`, columns `canopy_height_m, canopy_height_p95_m`.
  - `soil.soil_result(cells_by_property: Dict[str, DataFrame]) -> ProductResult`, columns `<prop>_<depth>` and `texture_class_0_5`; `soil.texture_class(clay, sand, silt) -> Optional[str]`.

Pinned values:
- Terrain: `projects/sat-io/open-datasets/ASTER/GDEM`, band `b1`, requested in its own projection, with `ee.Terrain.slope` and `ee.Terrain.aspect` masked by land. `FLAT_SLOPE_DEG = 0.1`, flat pixels contribute `(0, 0)`. `land_share` comes from `land_mask()` on its own grid.
- Canopy: `ee.ImageCollection("projects/sat-io/open-datasets/facebook/meta-canopy-height")`, `.mosaic().select("cover_code")`, `crs=col.first().select("cover_code").projection()`, then `.unmask(0).updateMask(land)`, so gaps on land count as 0. p95 reducer: `ee.Reducer.percentile([95], None, 1024, 0.125)`. `batch_size=30` (about 1 s per 3 cells).
- Soil: `projects/soilgrids-isric/{prop}_mean`, bands `{prop}_{depth}_mean`; properties `clay, sand, silt, bdod, cfvo, soc, phh2o, nitrogen`; depths `0-5cm, 5-15cm, 15-30cm, 30-60cm, 60-100cm, 100-200cm` become `0_5, 5_15, 15_30, 30_60, 60_100, 100_200`. Each property is requested in its own projection (`bdod` and `soc` have another origin). Conversion 0.1, except `bdod` and `nitrogen` 0.01.
- `texture_class` is ported as written from `extract_soilgrid.calculate_derived_properties`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_cell_with_no_land_is_no_data_with_zero_coverage():
    ...  # all_count 13, elevation_m_count 0, mean None -> NaN, NO_DATA, coverage 0.0, land_share 0.0 ok

def test_terrain_coverage_is_land_pixels_over_all_pixels(): ...       # 10 of 16 -> 0.625
def test_canopy_zero_is_a_value_not_missing(): ...                    # mean 0.0 with pixels -> OK, 0.0
def test_soil_applies_each_conversion(): ...                          # bdod 140 -> 1.4, sand 400 -> 40.0
def test_texture_class_matches_old_rules():
    assert texture_class(45, 20, 35) == "Clay"
    assert texture_class(5, 90, 5) == "Sand"
    assert texture_class(30, 40, 30) == "Clay Loam"
    assert texture_class(None, 20, 35) is None
```

- [ ] **Step 2: Run them.** `T test_terrain.py test_canopy.py test_soil.py`. Expected: FAIL, modules missing.

- [ ] **Step 3: Implement** the pure builders and each `extract`.

- [ ] **Step 4: Run them.** Expected: `### pytest exit: 0`.

- [ ] **Step 5: Add a live aspect test and run it.** `test_live_aspect_seam`: on a synthetic image split between 355 and 5 degrees with slope 10, the cell means give `aspect_sin` about 0 and `aspect_cos` about 0.9962; with slope 0 both are 0. `STATIC_LIVE_TESTS=1 T test_terrain.py`. Expected: `### pytest exit: 0`.

- [ ] **Step 6: Run the products for real, one at a time.** `P terrain`, `P canopy`, `P soil` (8 G, 2 h each). Expected: 226 rows each. Terrain: elevation against the current file r about 0.998 with median change about 8 m, `slope_deg` r about 0.90, `aspect_sin` r about 0.38, `CHE_PFY_CON` near 26 degrees. Soil: `sand_0_5` against REST r about 0.996.

- [ ] **Step 7: Commit.** `feat(static): terrain, canopy and soil at the 1 km cell`.

---

### Task 6: WorldClim, Köppen, BGI, Fan, and the reproduce check

**Files:**
- Create: `products/worldclim.py`, `products/stand_age.py`, `products/fan.py`, `checks/reproduce.py`
- Test: `tests/test_worldclim.py`, `test_stand_age.py`, `test_fan.py`, `test_reproduce.py`

**Interfaces:**
- Consumes: `rasters.pixel_with_reach(wrap_cols=False)`, `rasters.tiff_reader`.
- Produces: `extract(context)` in each; `stand_age.age_value(raw) -> float`; `fan.status_from_fallback(radius: int) -> str`; `reproduce.classify(old: Series, new: Series, rtol: float = 1e-6) -> Series` of `match, differs, empty_to_value, value_to_empty`; `reproduce.main(argv=None) -> int`.

Pinned values:
- WorldClim: `spatial_features/bio_30s/wc2.1_30s_bio_{n}.tif`, reach 3. Beck: `koppen_geiger/koppen/Beck_KG_V1_present_0p0083.tif`, reach 3, sentinel 0 (ocean), which the raster does not declare. Port `BIO_SCALE_FACTORS`, `KOPPEN_CLASSES` from `spatial_features_extractor.py`. A code outside the legend raises `ExtractionFailure`.
- BGI: `stand_age/*BGIForestAge*.nc` by h5py; datasets `latitude`, `longitude`, `ForestAge_TC000/010/020/030`; nearest node by `np.abs(arr - value).argmin()`, no search; valid when `0 < v <= 500`. Columns `stand_age_TC000/010/020/030`.
- Fan: `src.Extractors.extract_root_depth_fan2017.fetch_all_variables_at_point` with one shared `dataset_cache`. Fallback 0 is `ok`, above 0 `neighbour`, -1 `no_data`. Any exception from the reader becomes `ExtractionFailure` naming the site. Columns as in the current file's header.
- Current files: `env_site_data/<scale>/site_info_with_env_data.csv`, `terrain_site_data/<scale>/stand_age_data.csv`, `root_depth_site_data/<scale>/site_info_with_root_depth_fan2017.csv`, under `extracted_data`. `reproduce` only reads them.

- [ ] **Step 1: Write the failing tests.**

```python
def test_classify_hand_computed():
    old = pd.Series([1.0, np.nan, 2.0, np.nan, 5.0])
    new = pd.Series([1.0 + 1e-8, np.nan, np.nan, 3.0, 6.0])
    assert list(classify(old, new)) == ["match", "match", "value_to_empty", "empty_to_value", "differs"]

def test_classify_compares_strings_exactly(): ...
def test_reproduce_exits_nonzero_and_names_site_on_empty_to_value(tmp_path): ...

def test_koppen_sea_takes_nearest_land(tmp_path): ...          # small GeoTIFF: own 0, neighbour 31 -> 31, NEIGHBOUR
def test_koppen_code_outside_legend_stops(tmp_path): ...
def test_age_value_boundaries():
    assert age_value(500.0) == 500.0
    assert np.isnan(age_value(0.0)) and np.isnan(age_value(500.5)) and np.isnan(age_value(np.nan))

def test_fan_fallback_to_status():
    assert (status_from_fallback(0), status_from_fallback(2), status_from_fallback(-1)) == ("ok", "neighbour", "no_data")
```

- [ ] **Step 2: Run them.** `T test_worldclim.py test_stand_age.py test_fan.py test_reproduce.py`. Expected: FAIL, modules missing.

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run them.** Expected: `### pytest exit: 0`.

- [ ] **Step 5: Run the products and the check.** `P worldclim`, `P stand_age`, `P fan`, then `M src.static_features.checks.reproduce`. Expected: `match` at 226 of 226 sites for every column. BGI's August audit had 15/33/36/37 empty in TC000/010/020/030. If any site is `empty_to_value`, stop and take the list to the owner, because it changes the merge's `st_age` fallback.

- [ ] **Step 6: Commit.** `feat(static): unchanged products and the reproduce check`.

---

### Task 7: Downloads

**Files:**
- Create: `fetch.py`
- Test: `tests/test_fetch.py`

**Interfaces:**
- Consumes: `results.ExtractionFailure`.
- Produces:
  - `fetch.Download(url: str, name: str, md5: Optional[str] = None, size: Optional[int] = None)`, frozen.
  - `fetch.fetch(item: Download, directory: Path, runner=subprocess.run) -> Path`.
  - `fetch.zenodo_downloads(record_id: int, directory: Path, runner=subprocess.run) -> List[Download]`, saving `record.json` beside the files; URLs are `https://zenodo.org/api/records/<id>/files/<name>/content`, md5s come from the record.
  - `fetch.sha256_of(path: Path) -> str`, cached in a `<name>.sha256` sidecar.

Command: `curl -fL -C - --retry 5 --retry-delay 30 -o <name>.partial <url>`. Python's urllib and GDAL's HTTP reader get 403 from Zenodo; curl gets 200 and 206 from compute nodes.

- [ ] **Step 1: Write the failing tests**, with a fake runner that writes bytes to the `.partial`.

```python
def test_good_download_is_moved_into_place(tmp_path): ...
def test_wrong_md5_stops_and_leaves_no_final_file(tmp_path): ...    # .partial removed too, so a rerun starts fresh
def test_wrong_size_stops(tmp_path): ...
def test_interrupted_curl_keeps_partial_for_resume(tmp_path): ...   # runner returns 18; ExtractionFailure, .partial kept
def test_verified_file_is_not_downloaded_again(tmp_path): ...       # runner not called
def test_download_with_no_checksum_or_size_is_refused(): ...
```

- [ ] **Step 2: Run them.** `T test_fetch.py`. Expected: FAIL, module missing.

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run them.** Expected: `### pytest exit: 0`.

- [ ] **Step 5: Commit.** `feat(static): verified curl downloads`.

---

### Task 8: Han, Yang, Lusk, Liang

**Files:**
- Create: `products/han.py`, `products/yang.py`, `products/lusk.py`, `products/liang.py`
- Test: `tests/test_han.py`, `test_yang.py`, `test_lusk.py`, `test_liang.py`

**Interfaces:**
- Consumes: `rasters.pixel_with_reach`, `fetch.fetch`, `fetch.zenodo_downloads`, `fetch.sha256_of`, `results.ProductResult`.
- Produces: `extract(context)` and `fetch_inputs(context)` in each, plus `han.load_grids(directory) -> HanGrids` (depth, uncertainty, and type index as 2-D arrays, trees only); `yang.load_ensemble(directory) -> np.ndarray` (row-first, NaN where any model is invalid); `yang.layers_at(directory, row, col) -> np.ndarray` (the 32 layers at one pixel); `lusk.read_trait(path, sites) -> DataFrame`.

Pinned values:
- **Han.** `han_2025/History/Global_RD_{EST,UI}_{type}_2015.nc`, variable `depth`, 3600 x 7200, missing -32768, latitude 89.975 going south, longitude from -179.975. Tree types `BETTr, BETTe, BDTTr, BDTTe, BDTBo, NETTe, NETBo, NDT`; others `BESTe, BDSTe, BDSBo, C3G, C4G`. Reach 1, trees only, longitude wraps. `fetch_inputs` copies `/scratch/tmp/yluo2/han_probe/data/History/` and records the `History.rar` md5 `a3ddf1a20334ada08b5525cecd82cb3c`, because Palma has no unrar.
- **Yang.** `zenodo_downloads(10692059, ...)`, `global-wood-density-main.zip`, md5 `79168d8122ecfd684b51d24a962470ce`. `fetch_inputs` unpacks with `unzip` only `global-wood-density-main/Global_wood_density_predictions/{rf,lightgbm,xgboost,lightgbm_rf}_all.mat`. They are HDF5 (MATLAB 7.3), read with h5py. Datasets `wd_mean` (9000, 4500) and `wd` (8, 9000, 4500), float64. MATLAB stores column-major, so **h5py's axis order is (lon, lat)**: index `[col, row]` with `col = floor((lon + 180) / 0.04)` and `row = floor((90 - lat) / 0.04)`. Valid when finite and above 0 in all four models; 0 is the files' mask. `wood_density` is the mean of the four `wd_mean`, and `wood_density_sd` is `np.std(layers, ddof=1)` over the 32 layers at the chosen pixel. Reach 1, longitude wraps.
- **Lusk.** `zenodo_downloads(14646322, ...)`, 31 files `X{id}_mean_Shrub_Tree_Grass_1km.tif`. Three bands: mean, coefficient of variation, area of applicability. Read raw int16, test against -32768, then apply `src.scales[b] * raw + src.offsets[b]`. Project sites with `pyproj.Transformer.from_crs(4326, 6933, always_xy=True)`, because rasterio reports `LOCAL_CS`, and raise `ExtractionFailure` unless the file's `crs` tag contains `6933`. Reach 3 on the mean band; the other two bands are read at the pixel it chose. Columns `lusk_<id>`, `lusk_<id>_cov`, `lusk_<id>_aoa`, except `root_depth_lusk*` (6) and `wood_density_lusk*` (4). IDs: 282, 237, 47, 3113, 13, 146, 78, 55, 163, 144, 50, 14, 15, 46, 3120, 145, 3106, 6, 95, 27, 26, 138, 3117, 1080, 614, 4, 169, 281, 21, 289, 297. The authors star 15 as experimental: 282, 237, 47, 163, 144, 3120, 3106, 95, 27, 26, 138, 614, 21, 289, 297 (1080 has r 0.46 and no star; record it as published). `provenance.json` takes each trait's name, unit, and r from its tags, and the licence tag. Raise `ExtractionFailure` unless tag `trait_unit` is `m` for 6 and `g cm-3` for 4.
- **Liang.** `https://ndownloader.figshare.com/files/35133814`, md5 `bf745222a79918df7e52e4b785ccd993`, `S_mean_raster.tif`, nodata -3.4e38, valid when finite and above 0. Reach 1, longitude wraps. Column `tree_richness_per_ha`.

- [ ] **Step 1: Write the failing tests**, building tiny NetCDF, HDF5, and GeoTIFF files in `tmp_path`.

```python
def test_han_tree_pixel_is_ok_with_uncertainty_and_type(tmp_path): ...
def test_han_grass_pixel_is_skipped_for_a_neighbouring_tree(tmp_path): ...     # NEIGHBOUR, type of the tree
def test_han_no_tree_within_one_pixel_is_no_data(tmp_path): ...
def test_han_two_types_on_one_pixel_stops(tmp_path): ...
def test_han_wraps_the_dateline(tmp_path): ...

def test_yang_axes_are_lon_then_lat(tmp_path): ...               # value written at [col, row] is what the site reads
def test_yang_zero_is_no_data_not_a_density(tmp_path): ...
def test_yang_ensemble_is_the_mean_of_four_models(tmp_path): ... # 0.4 0.5 0.6 0.7 -> 0.55
def test_yang_pixel_counts_only_with_all_four_models(tmp_path): ...
def test_yang_sd_spans_all_32_layers(tmp_path): ...
def test_yang_beyond_one_pixel_is_no_data(tmp_path): ...

def test_lusk_applies_scale_and_offset(tmp_path):
    ...  # raw 100, scale 0.01132, offset 370.9 -> root_depth_lusk 372.032
def test_lusk_other_bands_read_at_the_chosen_pixel(tmp_path): ...              # mean from a neighbour; cov and aoa from there too
def test_lusk_nodata_is_tested_before_scaling(tmp_path): ...                   # -32768 with an offset is still NO_DATA
def test_lusk_refuses_file_whose_crs_tag_is_not_6933(tmp_path): ...

def test_liang_declared_nodata_takes_the_neighbour(tmp_path): ...
def test_liang_beyond_one_pixel_is_no_data(tmp_path): ...
```

- [ ] **Step 2: Run them.** `T test_han.py test_yang.py test_lusk.py test_liang.py`. Expected: FAIL, modules missing.

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run them.** Expected: `### pytest exit: 0`.

- [ ] **Step 5: Fetch and run for real.** `P han --fetch-only`, `P yang --fetch-only`, `P lusk --fetch-only` (4 h), `P liang --fetch-only`, then `P han`, `P yang`, `P lusk`, `P liang`. Expected: Han 144 `ok`, 59 `neighbour`, 23 `no_data`; Yang 180, 17, 29, with own-pixel values from 0.438 to 0.656; Liang 185, 11, 30, values within 1.79 to 201; Lusk 93 columns and 31 traits in provenance. A difference is reported, not patched over.

- [ ] **Step 6: Commit.** `feat(static): Han, Yang, Lusk and Liang`.

---

### Task 9: CCI, GAMI, Lesiv

**Files:**
- Create: `products/cci.py`, `products/gami.py`, `products/lesiv.py`
- Test: `tests/test_cci.py`, `test_gami.py`, `test_lesiv.py`

**Interfaces:**
- Consumes: `rasters.pixel_with_reach`, `rasters.centred_span`, `fetch.fetch`, `grid.cell_bounds`.
- Produces: `extract(context)` and `fetch_inputs(context)` in each, plus pure functions:
  - `cci.window_value(hits: Sequence[PixelHit]) -> Tuple[float, str]`: mean of years with a value; `ok` only if every one is the own pixel; `neighbour` if any is not; `no_data` if none.
  - `gami.cell_mean(block: np.ndarray, fill: int) -> Tuple[float, float]`: `(mean age, forest share)` for one member's cell; `gami.read_order(sites, chunk_rows, chunk_cols) -> List[int]`, row positions sorted by chunk.
  - `lesiv.cell_shares(block: np.ndarray, nodata) -> Tuple[Dict[int, float], Optional[int]]`: `(share per class, dominant class)`.

Pinned values:
- **CCI.** `https://dap.ceda.ac.uk/neodc/esacci/biomass/data/agb/maps/v7.0/netcdf/ESACCI-BIOMASS-L4-AGB-MERGED-1000m-fv7.0.nc`, size 2,983,287,549. Variables `agb`, `agb_sd`, shape (18, 18000, 36000), int16, fill -32768, Mg/ha. Latitude 89.995 step -0.01, longitude from -179.995. Time in days since 1990-01-01; take years from the file and require 2010, 2011, 2012 (7305, 7670, 8035). 0 is a valid value. Reach 3. `agb_mg_ha` and `agb_sd` are the means over 2010-2012 of the per-year cell values; all 18 years go to `by_year`.
- **GAMI.** `https://datapub.gfz.de/download/10.5880.GFZ.1.4.2023.006-VEnuo/GAMIv2-1_2010-2020_100m.nc`, size 234,035,960,874. HDF5 dataset `forest_age`, dims (member 20, lat 202500, lon 405000, time 2), int16, gzip chunks (1, 7789, 15577, 1), root attribute `_FillValue` -9999. Latitude 89.99955 step -1/1125. Time 0 and 3652 days since 2010-01-01. Read with `h5py.File(path, rdcc_nbytes=4 * 2**30, rdcc_nslots=10007)`. Loop time, then member, then sites by `read_order`, so each chunk is decompressed once per member and year. Per-member cell mean, then the mean of members with a forest pixel. `forest_age_gami` is 2010; 2010 and 2020 go to `by_year`; `forest_age_gami_coverage` is the mean forest share over members. Provenance records the authors' advice to bin into classes of at least two decades.
- **Lesiv.** `FML_v3-2_with-colorbar.tif` from Zenodo 5879022. Classes 11, 20, 31, 32, 40, 53. Window values must lie in the classes plus declared nodata, 0, and 128, or the run stops. Denominator is every pixel centred in the cell. Dominant tie goes to the lowest code; no forest pixel means `fm_dominant` is `no_data` while shares stay `ok` at 0.

- [ ] **Step 1: Write the failing tests.**

```python
def test_cci_window_mean_and_status():
    ...  # hits [10 ok, 20 ok, nan no_data] -> (15.0, "ok"); [10 ok, 20 neighbour] -> (15.0, "neighbour")
def test_cci_zero_is_a_value(tmp_path): ...
def test_cci_year_outside_window_goes_to_by_year_only(tmp_path): ...
def test_cci_refuses_file_without_the_window_years(tmp_path): ...

def test_gami_cell_mean_hand_computed():
    block = np.array([[10, -9999], [30, -9999]])
    assert cell_mean(block, -9999) == (20.0, 0.5)
def test_gami_no_forest_is_no_data_with_zero_coverage(tmp_path): ...
def test_gami_uses_2010_and_keeps_2020_in_by_year(tmp_path): ...
def test_gami_read_order_groups_sites_in_one_chunk(): ...

def test_lesiv_shares_hand_computed():
    block = np.array([11, 11, 11, 11, 31, 31, 31, 128, 128, 128])
    shares, dominant = cell_shares(block, None)
    assert shares[11] == 0.4 and shares[31] == 0.3 and shares[53] == 0.0 and dominant == 11
def test_lesiv_tie_takes_lowest_code(): ...
def test_lesiv_no_forest_has_zero_shares_and_no_dominant(): ...
def test_lesiv_unknown_value_stops(): ...
```

- [ ] **Step 2: Run them.** `T test_cci.py test_gami.py test_lesiv.py`. Expected: FAIL, modules missing.

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run them.** Expected: `### pytest exit: 0`.

- [ ] **Step 5: Fetch and run for real.** `P cci --fetch-only`, `P lesiv --fetch-only`, `P cci`, `P lesiv`, then `P gami --fetch-only` with `--time=24:00:00`, then `P gami` with `--mem=32G --time=24:00:00`. Check free disk first: at least 300 GiB. Expected: `fm_share_*` sums at most 1 at every site, `forest_age_gami` within 0 to 300, `agb_mg_ha` within 0 to 1500, and the count of `ok`, `neighbour`, and `no_data` per column reported, since these three have no acceptance numbers yet.

- [ ] **Step 6: Commit.** `feat(static): CCI biomass, GAMI forest age and Lesiv management`.

---

### Task 10: Tree cover and EVI

**Files:**
- Create: `products/tree_cover.py`, `products/evi.py`
- Test: `tests/test_tree_cover.py`, `test_evi.py`

**Interfaces:**
- Consumes: Task 4's `reduce_cells`, `mean_and_count`, `with_all_band`, `land_mask`, `with_retries`, `ShortReply`.
- Produces: `extract(context)` in each; `tree_cover.window_result(yearly: DataFrame) -> ProductResult`; `evi.composites_from_reply(reply, site_names, dates, cell_pixels: DataFrame) -> DataFrame` with columns `site_name, date, qa_class, mean_evi, pixel_count, land_pixels, all_pixels`, where `cell_pixels` holds the last two per site, from one reduction of the land band on EVI's grid; `evi.summarise(composites, first_year=2001, last_year=2020) -> ProductResult` with columns `evi_mean, evi_amplitude, evi_years_used`.

Pinned values:
- Tree cover: `MODIS/061/MOD44B`, `Percent_Tree_Cover`, years 2000 to 2025, in its own projection. Water and fill are already masked in the asset. `tree_cover_pct` is the mean over 2010-2012 of per-year cell means.
- EVI: `MODIS/061/MOD13Q1`, bands `EVI`, `SummaryQA`, land mask applied, 460 composites over 2001-2020. Group form `mean_and_count().group(groupField=1, groupName="qa")`, returning `{'groups': [{'count', 'mean', 'qa'}]}`, then `col.map(per_image).flatten().getInfo()`, one year and at most 60 sites per request. Scale 0.0001. Classes 0 and 1 count; 2 snow and 3 cloud do not. A composite is valid with at least half its land pixels in classes 0+1. A year counts with at least 6 valid composites. Amplitude is `np.percentile(..., 90) - np.percentile(..., 10)` with numpy's linear default. Coverage is valid pixels over all pixels, averaged over valid composites in used years.

- [ ] **Step 1: Write the failing tests.**

```python
def test_tree_cover_zero_counts(): ...
def test_tree_cover_window_mean_and_by_year(): ...       # 2010-2012 mean; all years in by_year
def test_composite_mean_weights_classes_by_pixels(): ... # class 0 mean .5 of 6, class 1 mean .3 of 2 -> .45
def test_snow_and_cloud_are_excluded(): ...
def test_composite_needs_half_its_land_pixels():
    ...  # land 16: valid 7 -> invalid; valid 8 -> valid
def test_year_needs_six_composites(): ...                # 5 -> not used; 6 -> used
def test_amplitude_is_p90_minus_p10(): ...               # composites 0..10 -> 8.0
def test_years_used_and_no_qualifying_year_is_no_data(): ...
def test_summaries_average_years_not_composites(): ...
def test_composites_reply_missing_a_date_is_short(): ...
```

- [ ] **Step 2: Run them.** `T test_tree_cover.py test_evi.py`. Expected: FAIL, modules missing.

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run them.** Expected: `### pytest exit: 0`.

- [ ] **Step 5: Run for real, one at a time.** `P tree_cover` (2 h), `P evi` (16 G, 8 h). Expected: 226 rows; `tree_cover_pct` within 0 to 100; `evi_mean` within -0.2 to 1; `evi_composites` holds 460 dates. Report sites with `evi_years_used` below 20, by name.

- [ ] **Step 6: Commit.** `feat(static): tree cover and EVI summaries`.

---

### Task 11: Phenology

**Files:**
- Create: `products/phenology.py`
- Test: `tests/test_phenology.py`

**Interfaces:**
- Consumes: Task 4's `sample_cells`, `land_mask`, `check_bands`.
- Produces: `extract(context)`; `phenology.main_cycle(pixels) -> DataFrame`; `phenology.summarise(pixels, first_year=2001, last_year=2020) -> ProductResult`, whose `extras["phenology_pixels"]` is the sampled table.

Pinned values:
- `MODIS/061/MCD12Q2`, product years 2001 to 2025, projection from `Peak_1` (SR-ORG:6974, 463.3127 m). 23 bands: `NumCycles`; `{Greenup, MidGreenup, Maturity, Peak, Senescence, MidGreendown, Dormancy}_{1,2}`; `{EVI_Minimum, EVI_Amplitude, EVI_Area, QA_Overall}_{1,2}`. All `.unmask(-9999)`, plus the `land` band, sampled per year.
- Masked `NumCycles` means no cycle, never 0 in the data. Dates are days since 1970-01-01. Main cycle is the larger `EVI_Amplitude`; on a tie, `_1`.
- Angle `2 * pi * doy / 365.25`, not renormalised. Averages: over land pixels, then over years. `phen_season_days = Dormancy - Greenup`, `phen_midseason_days = MidGreendown - MidGreenup`, per pixel-year. Scale: `EVI_Minimum`, `EVI_Amplitude` 0.0001; `EVI_Area` 0.1.
- `phen_cycle_share`: land pixel-years with a cycle over land pixel-years. `phen_num_cycles`: mean of `NumCycles` with no cycle as 0. `phen_years_used`: years with at least one land pixel with a cycle. All three are `ok` at any site with land. No land means every column is `no_data`. No cycle means dates, lengths, and EVI2 columns are `no_data`, and the three above are `ok`, with `phen_cycle_share` and `phen_years_used` at 0. Per-year cell values go to `by_year`, 2001 to 2025. `provenance.json` records the guide's warning on greenup and dormancy at high latitude with low amplitude.

- [ ] **Step 1: Write the failing tests** on hand-built pixel-year tables.

```python
def test_peak_dates_across_new_year_average_to_january():
    ...  # peaks at days 14610 (2010-01-01) and 14974 (2010-12-31): phen_peak_cos > 0.99, abs(phen_peak_sin) < 0.02
def test_main_cycle_is_the_larger_not_the_first(): ...       # amplitude_1 1000, amplitude_2 3000 -> dates of _2
def test_main_cycle_tie_takes_the_first(): ...
def test_season_days_per_pixel_before_averaging(): ...
def test_season_days_skip_a_pixel_missing_one_date(): ...
def test_no_cycle_counts_as_zero_for_share_and_num_cycles(): ...   # 4 land pixel-years, 1 cycle -> .25 and .25
def test_water_pixels_are_ignored_everywhere(): ...
def test_no_land_is_no_data_for_every_column(): ...
def test_land_without_cycle_keeps_share_zero_and_dates_no_data(): ...
def test_evi2_scaling(): ...                                 # 3000 -> .3; area 300 -> 30.0
def test_years_used_counts_years_with_a_cycle(): ...
def test_years_outside_window_reach_only_by_year(): ...      # 2021 not in the summary
```

- [ ] **Step 2: Run them.** `T test_phenology.py`. Expected: FAIL, module missing.

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run them.** Expected: `### pytest exit: 0`.

- [ ] **Step 5: Run for real.** `P phenology` (8 G, 4 h). Expected, against the Hansen-based probe, reported by name where different: 204 sites with dates; 21 with no cycle in 2001-2020; `CHE_DAV_SEE` `no_data`; land pixel-years with a cycle about 68.5%; share by IGBP about EBF .31, WSA .41, SAV .60, WET .63, ENF .82, DBF .85, MF .86, DNF 1.0; `ZAF_RAD` peak near day 358.

- [ ] **Step 6: Commit.** `feat(static): MCD12Q2 phenology summaries`.

---

### Task 12: Assemble and provenance

**Files:**
- Create: `assemble.py`
- Test: `tests/test_assemble.py`

**Interfaces:**
- Consumes: `results.read_part`, `sites.site_list_hash`, `run.PRODUCTS`.
- Produces: `assemble.assemble(context, products=tuple(PRODUCTS), git_commit=None) -> Path`; `assemble.aspect_report(sin, cos) -> Tuple[Optional[float], Optional[str]]`; `assemble.main(argv=None) -> int`. Output files as the spec's Outputs section.

Pinned headers, exact:
- `site_info_with_env_data.csv`: `start_date,end_date,lat,lon,site_name,canopy_height_m,elevation_m,slope_deg,aspect_deg,aspect_standardized,aspect_category,aspect_sin,aspect_cos,bio1..bio19,koppen_class,koppen_name,slope_deg_centre`. `aspect_deg` and `aspect_standardized` are the vector's bearing, `aspect_category` its sector, all empty on `(0, 0)`. `slope_deg_centre` stays empty, because 1 km has no centre-pixel slope; provenance says so.
- `soilgrids_data.csv`: `start_date,end_date,lat,lon,site_name,data_source,offset_distance_m,snapped_lat,snapped_lon`, then `clay,sand,silt,bdod,cfvo,soc,phh2o,nitrogen` by depth, then `texture_class_0_5`. `data_source` is `earth_engine_1km`, offset 0, snapped equal to the site.
- `stand_age_data.csv`: `start_date,end_date,lat,lon,site_name,stand_age_TC000,stand_age_TC010,stand_age_TC020,stand_age_TC030`.
- `site_info_with_root_depth_fan2017.csv`: `site_name,root_depth,infiltration_depth,deep_drainage,drainage_frequency,regolith_flush_rate,gw_residence_time,root_depth_mask_valid,root_depth_fallback_px,root_depth_continent,root_depth_lat_pixel,root_depth_lon_pixel`.
- `new_static_features.csv`: `site_name` plus all value columns of Han, Yang, Lusk, Liang, Lesiv, CCI, GAMI, tree cover, EVI, phenology, plus `canopy_height_p95_m` and `land_share`. `static_quality.csv`: `site_name` plus `<col>_status` for every value column and `<col>_coverage` where measured.
- `static_features_by_year.csv` concatenates every part's `by_year`. Each part's `extras` become `<name>.csv` at the root (`evi_composites.csv`, `phenology_pixels.csv`).

- [ ] **Step 1: Write the failing tests** on parts built by a small fixture helper.

```python
def test_env_file_has_the_exact_header(tmp_path): ...
def test_soil_file_has_the_exact_header(tmp_path): ...
def test_stand_age_file_holds_only_four_stand_age_columns(tmp_path): ...  # the merge averages any "stand_age_TC" column
def test_fan_file_has_the_exact_header(tmp_path): ...
def test_new_static_features_has_no_status_columns(tmp_path): ...
def test_quality_file_has_a_status_for_every_value_column(tmp_path): ...

# Review Focus 3
def test_assemble_refuses_part_from_another_site_list(tmp_path): ...      # message names the part
def test_assemble_refuses_a_missing_product(tmp_path): ...
def test_assemble_refuses_a_column_from_two_products(tmp_path): ...
def test_assemble_failure_leaves_previous_output_intact(tmp_path): ...

def test_aspect_report_hand_computed():
    assert aspect_report(1.0, 0.0) == (90.0, "E")
    assert aspect_report(0.0, 0.0) == (None, None)
def test_provenance_records_source_checksum_window_and_commit(tmp_path): ...
```

- [ ] **Step 2: Run them.** `T test_assemble.py`. Expected: FAIL, module missing.

- [ ] **Step 3: Implement.** Port `compass_sector` from `terrain_aspect.py`. Build into a temp directory under `output_root`, then move files into place.

- [ ] **Step 4: Run them.** Expected: `### pytest exit: 0`.

- [ ] **Step 5: Assemble for real.** `M src.static_features.assemble`. Expected: all files present with 226 rows; `head -1` of the four merge files matches the current files on Palma, compared with `diff`.

- [ ] **Step 6: Commit.** `feat(static): assemble parts into output files and provenance`.

---

### Task 13: Comparison, range, and agreement checks

**Files:**
- Create: `checks/compare.py`, `checks/ranges.py`, `checks/agreement.py`
- Test: `tests/test_compare.py`, `test_ranges.py`, `test_agreement.py`

**Interfaces:**
- Consumes: the output root of Task 12, `sites.load_sites`.
- Produces, each with `main(argv=None) -> int` writing CSVs to `<output_root>/checks/`:
  - `compare.column_report(old: Series, new: Series, top: int = 10) -> Dict[str, Any]`, with `r`, `n`, `largest_movers`.
  - `ranges.RANGES: Dict[str, Tuple[float, float]]`; `ranges.out_of_range(values) -> DataFrame`; `ranges.coverage_report(quality) -> DataFrame`, per product and reason; `ranges.low_land_sites(values, threshold=0.5) -> List[str]`.
  - `agreement.pair_stats(a: Series, b: Series, seed: int = 42, draws: int = 2000) -> Dict[str, float]`, with Pearson and Spearman r, bias `mean(a - b)`, RMSE, `n` after dropping duplicate `(a, b)` pairs, and 2.5/97.5 bootstrap intervals; `agreement.report(frame, pairs, igbp) -> DataFrame`, overall and by `si_igbp`.

Pinned values:
- `compare` runs elevation, slope, aspect, canopy, and soil against the current files, plus `canopy_height_m` against `metasq1km_h` in `probes/canopy_height_ground/canopy_stats_sites.csv`.
- `RANGES`: `elevation_m` -450..9000; `slope_deg` 0..90; `aspect_sin`, `aspect_cos` -1..1; `canopy_height_m` 0..60; `canopy_height_p95_m` 0..100; `land_share`, `fm_share_*`, `phen_cycle_share` 0..1; soil `clay`, `sand`, `silt`, `cfvo` 0..100, `bdod` 0..2.2, `soc` 0..1000, `phh2o` 3..11, `nitrogen` 0..100; `root_depth_han` 0..100; `wood_density` 0.1..1.5; `tree_richness_per_ha` 1..250; `agb_mg_ha` 0..1500; `forest_age_gami` 0..300; `agb_sd` 0..1500; `tree_cover_pct` 0..100; `evi_mean`, `evi_amplitude`, `phen_evi2_min` -0.2..1; `phen_evi2_amplitude` 0..1; `phen_*_sin`, `phen_*_cos` -1..1; `phen_season_days`, `phen_midseason_days` 0..730; `phen_num_cycles` 0..2; `phen_years_used`, `evi_years_used` 0..20; `root_depth_lusk` 0..100; `wood_density_lusk` 0.1..1.5.
- Pairs: `root_depth_lusk`–`root_depth_han`, `root_depth_lusk`–`root_depth`, `root_depth_han`–`root_depth`, `wood_density_lusk`–`wood_density`. Before comparing, the Lusk unit tags from provenance must read `m` and `g cm-3`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_duplicate_pairs_collapse():
    ...  # three sites with the same (a, b) plus two others -> n == 3
def test_bias_and_rmse_hand_computed():
    stats = pair_stats(pd.Series([1.0, 2.0, 3.0]), pd.Series([2.0, 2.0, 2.0]))
    assert stats["bias"] == 0.0 and stats["rmse"] == pytest.approx(0.8165, abs=1e-4)
def test_constant_series_gives_nan_r_not_a_crash(): ...
def test_bootstrap_is_repeatable_with_seed_42(): ...
def test_agreement_refuses_an_unexpected_unit(): ...
def test_report_has_a_row_per_igbp_and_overall(): ...
def test_column_report_names_largest_movers(): ...
def test_out_of_range_names_site_and_column(): ...
def test_nan_is_not_out_of_range(): ...
def test_low_land_sites_lists_below_half(): ...
def test_shares_summing_above_one_are_reported(): ...
```

- [ ] **Step 2: Run them.** `T test_compare.py test_ranges.py test_agreement.py`. Expected: FAIL, modules missing.

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run them.** Expected: `### pytest exit: 0`.

- [ ] **Step 5: Run on the real output.** `M src.static_features.checks.compare`, `M src.static_features.checks.ranges`, `M src.static_features.checks.agreement`. Expected: `ranges` exits 0, or lists each exception to report to the owner. `low_land_sites` includes `GBR_DEV_CON`, `GBR_DEV_DRO`, `CHE_DAV_SEE`. Canopy against `metasq1km_h` is high but not identical, because of the site-centred footprint. `agreement` is a report, not a gate.

- [ ] **Step 6: Commit.** `feat(static): comparison, range and agreement reports`.

---

### Task 14: The merge switch

**Files:**
- Modify: `notebooks/merge_gap_filled_hourly_orginal.py`: `merge_sap_env_data_site` (713), static loads (830-854), `load_stand_age_map`, per-site lookups (~997-1012), `add_static_vars` (~1220), `static_prefixes`/`static_exact` (~1382), MANIFEST `flags` (~1681), argparse (~1713)
- Modify: `job_pytest_merge_contract.sh`: add the new test to its `NEW` list
- Create: `checks/merge_parity.py`, `jobs/job_merge_static.sh` (modelled on `/scratch/tmp/yluo2/gsv/job_merge_sweep5.sh`, which is untracked, with `$W` paths and `STATIC_ROOT` as an optional variable)
- Test: `notebooks/test_merge_static_switch.py` (style of `notebooks/test_merge_feature_contract.py`), `tests/test_merge_parity.py`

**Interfaces:**
- Consumes: the output files of Task 12.
- Produces:
  - `static_input_paths(static_root: Optional[Path]) -> Dict[str, Path]` with keys `env`, `soil`, `stand_age`, `fan`, plus `new` when on. Off returns today's four paths exactly. On raises `FileNotFoundError` naming every absent file.
  - `load_stand_age_map(base_path, age_file_path=None)`: unchanged when `age_file_path` is `None`.
  - `load_new_static(path: Path, expected_sites) -> Tuple[pd.DataFrame, List[str]]`: frame indexed by `site_name` and the value columns.
  - `merge_sap_env_data_site(..., static_root=None)`; CLI `--static-root`. Value columns reach `add_static_vars` and `static_exact` only when on. MANIFEST `flags` records `static_root`.
  - `merge_parity.compare(old: DataFrame, new: DataFrame, static_columns) -> Dict[str, Any]`, with `same_keys`, `dynamic_columns_differing`, `added_columns`, `removed_columns`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_off_returns_todays_paths(): ...                  # exactly the current four
def test_off_stand_age_loader_is_unchanged(tmp_path): ...

# Review Focus 5
def test_on_missing_file_raises_naming_it(tmp_path): ...
def test_new_static_missing_a_site_raises_naming_it(tmp_path): ...

def test_new_columns_are_static_only_when_on(): ...       # present in static_exact on, absent off
def test_stand_age_map_reads_the_given_file(tmp_path): ...

def test_parity_passes_when_only_static_columns_change(): ...
def test_parity_reports_a_changed_dynamic_column(): ...
def test_parity_reports_a_lost_row(): ...
```

- [ ] **Step 2: Run them.** `GSV_ROOT=$W WITH_REGRESSION=0 sbatch --wait job_pytest_merge_contract.sh`, and `T test_merge_parity.py`. Expected: `RESULT: FAIL` and a failing pytest exit, names missing.

- [ ] **Step 3: Write the patch.** Author a unified diff and apply it with `git apply --check`, then `git apply`, in `$W`, because the merge file is large and the local linter hook must not touch it. Print `git diff notebooks/merge_gap_filled_hourly_orginal.py` and read every hunk: each must be inert when `static_root` is `None`.

- [ ] **Step 4: Run them, with the regression suite.** `GSV_ROOT=$W sbatch --wait job_pytest_merge_contract.sh`, and `T test_merge_parity.py`. Expected: `RESULT: PASS (new + regression)` and `### pytest exit: 0`.

- [ ] **Step 5: Run the merge twice from this branch**, with the arm 5 flags `--daytime-sap-only --growing-season-only --apply-treatment-filter`, through `job_merge_static.sh` (64 G, 6 h each), to two new `--output-dir` directories: once without the switch, once with `--static-root data/raw/extracted_data/static_1km/sapwood_combined`. Same code on both sides, because the newest arm 5 output on disk predates the merge's last commit. Expected: both exit 0, and the second `MANIFEST.json` records the static root.

- [ ] **Step 6: Run the parity check** on the two outputs, `M src.static_features.checks.merge_parity --off <dir> --on <dir>`. Expected: same site-day keys, no column removed, the new columns added. Columns computed from soil properties in the merge may differ and are named in the report. Any other dynamic column that differs goes to the owner with its name.

- [ ] **Step 7: Commit.** `feat(merge): opt-in --static-root reads the 1 km features`.

---

### Task 15: Final run, review, and records

**Files:**
- Modify: `CLAUDE.md` (module table, Stage 2 note), memory `static_features_rebuild.md`, `MEMORY.md`

- [ ] **Step 1: Run the whole test directory.** `T .` and `GSV_ROOT=$W sbatch --wait job_pytest_merge_contract.sh`. Expected: `### pytest exit: 0` and `RESULT: PASS (new + regression)`. If `pytest-cov` is in the venv, add `--cov=src/static_features` and report the number against the 80% target. Reader lines that run only live and in the real runs may hold it below that; say so if they do.

- [ ] **Step 2: Reassemble and rerun every check** from the current commit. Expected: `reproduce` 226 of 226, `ranges` exit 0 or each exception listed, and provenance's `code_commit` equal to `HEAD`.

- [ ] **Step 3: Whole-branch review** by a fresh reviewer on the most capable model (`superpowers:requesting-code-review`). Fix critical and high findings, then rerun Step 1.

- [ ] **Step 4: Update the records.** `CLAUDE.md`: a module row, the Stage 2 note, and the `--static-root` switch. Memory: `static_features_rebuild.md` and its index line.

- [ ] **Step 5: Commit and push from Palma.** `docs: record the static feature module`, then `git push -u origin feat/static-features`. No merge into another branch and no old script deleted without the owner's agreement.

- [ ] **Step 6: Write the work log** with the `work-log-reporter` skill, per `CLAUDE.md`.
