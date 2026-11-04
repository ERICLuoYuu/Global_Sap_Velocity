# Static feature extraction rebuild: design

Date: 2026-10-30, revised 2026-11-02. Status: approved by the owner on 2026-11-01. The owner then added MCD12Q2, and
that dataset's rules plus the changes from opening the data on 2026-11-02 are reviewed with the plan. Runs on Palma
(`/scratch/tmp/yluo2/gsv`).

Items marked **(my call)** were decided by the author of this document, not by the project owner, and can be
overridden. Everything else was decided by the owner in conversation.

## Why

The static extractors grew by patching: scripts that run on import, one script holding unrelated products, 14
`.prepatch`-style backup files in `src/Extractors`, and "no data" sharing a return channel with "the request failed"
(`stage2_extraction_audit`). Footprints are also inconsistent: the same site is described by a 250 m disc, one 30 m
pixel, or a 1 km pixel depending on the feature.

The owner wants one clean module, ten new datasets, and one consistent spatial scale of 1 km.

## Goals

1. A new package, `src/static_features/` **(my call on the name)**, that extracts every static feature for the
   226-site `sapwood_combined` corpus.
2. Every product finer than 1 km is aggregated to a 1 km cell before extraction, including existing features.
3. Ten new datasets added (below).
4. Every value carries a reason code, and a failed request stops the run rather than becoming NaN.
5. Current trained models stay reproducible: nothing already on disk is overwritten.

## Non-goals (this round)

- Serving the new features on the 0.1 degree grid. The code must be reusable for it, but `src/serving` is not changed.
- Retraining, feature selection, or changing the FFS feature registry.
- Dynamic inputs: ERA5-Land, GlobMap LAI, hPET, or a 16-day EVI series.
- `extract_pft.py` stays as it is. The merge still loads its output as the last-choice PFT source (`resolve_pft`:
  provider, then metadata-derived, then MODIS), and `preflight_merge.py` and `assemble_extracted.py` both list it, so the
  file must keep existing. No corpus site reaches that last choice: all 22 merge logs on Palma that report the sources
  show `0 modis`, most recently `226 provider`. Its known `scale=500` defect (the native grid is 463.31 m) therefore
  stays too, and would matter only if a future site had neither a provider value nor usable metadata.
- Deleting the old scripts. They stay until the new module has passed its checks and the owner agrees.

## Anti-goals

- A merge that writes NaN columns because a static file or site is absent. It stops and names what is missing.
- A NaN that could mean either "no forest here" or "Earth Engine timed out".
- Unpacking large archives onto scratch. File quota is 466,556 of 1,000,000.

## The 1 km rule

**Grid.** The WorldClim 30 arc-second grid, origin (-180, 90). A site belongs to the cell
`col = floor((lon + 180) * 120)`, `row = floor((90 - lat) * 120)`. Sites are not centred in their cell, so a plot can
sit up to about 650 m from the cell centre. This is accepted because it matches what aggregating a raster and then
sampling it would give, and a fixed grid is what lets sites and grid cells share one definition.

**Finer than 1 km:** aggregate native pixels inside the cell. **About 1 km or coarser:** read the native pixel holding
the site. Nothing is upsampled, and products close to 1 km on another grid are not resampled onto the 30 arc-second grid.

**Which pixels count (my call).** A native pixel belongs to a cell when its centre lies inside the cell, and each such
pixel counts once. Earth Engine's default mean also takes in pixels that only touch the cell, weighted by overlap, and a
`count` combined with it then counts every touched pixel (22 at AUS_WOM against about 13 centres), so every Earth Engine
reducer here is `.unweighted()`. Local rasters use the same rule. Measured at the 226 sites: 1 to 4 pixel centres per
cell at MCD12Q2's 463 m (only 1 at four high-latitude sites), 5 to 16 at 232 m, and no cell is empty.

**Global rasters are not built (my call).** Only the cells containing sites are computed. The values are the same and
the storage far smaller. Meta canopy height alone would need about 700,000 pixels per cell if done globally.

**Zero versus missing** decides what each number means, so every aggregated product states it (see tables). Each
aggregated product also writes `<column>_coverage`, the share of the cell with valid pixels.

**Land only (my call, mask changed 2026-11-02).** Elevation, slope, aspect, canopy height, EVI, and phenology are
averaged over land pixels, using ESA WorldCover v200 (10 m) as the one land mask: a pixel is land unless its class is 80
(permanent water), and a masked WorldCover pixel counts as water. Hansen `datamask == 1` was the first choice, but it
reads too little water in the Forth estuary, the case this rule exists for: 29% in the GBR_DEV cell, against 63% in
WorldCover and 57% in JRC surface water. The mask is read on its own grid and then sampled, because reading it through
a request's grid moved the answer (measured on Hansen: 140 of 12,780 pixel-years at 463 m). Testing for "is land" means
it does not matter how each product encodes sea. Water never counts as zero, so a coastal cell keeps a true land
elevation and canopy height, and a cell with no land is `no_data`. Tree cover uses its own water class and soil its own
mask. WorldCover spans 60 S to 84 N, and site latitudes run from -51.6 to 69.5.

**Order.** Space first, then time. Per-year cell values go in a long table, so the time window can be changed later
without extracting again.

## Existing features

| Column(s) | Product | Rule | Notes |
|---|---|---|---|
| `elevation_m` | ASTER GDEM, Earth Engine | Cell mean | Same DEM as now |
| `slope_deg` | `ee.Terrain.slope` at native scale | Cell mean of pixel slopes | Not the slope of a mean DEM |
| `aspect_sin`, `aspect_cos` | `ee.Terrain.aspect` at native scale | Cell mean of pixel sin and cos, not renormalised **(my call)** | Flat pixels contribute (0, 0). Magnitude below 1 expresses incoherent direction |
| `canopy_height_m` | Meta 1.2 m | Cell mean over land, canopy gaps on land count as 0 | Behaves as cover x height |
| `canopy_height_p95_m` (new) | Meta 1.2 m | Cell 95th percentile **(my call)** | Added because tree cover now overlaps the mean |
| `sand_0_5` ... `cfvo_60_100` | SoilGrids, Earth Engine copy | Cell mean per property and depth | Replaces REST, which cannot aggregate. Hydraulics are still computed in the merge from these |
| `bio1..19`, `koppen_*` | WorldClim, Beck | Native pixel, existing search | About 1 km, must reproduce current values |
| `stand_age_TC*` | BGI forest age | Nearest pixel, no neighbour search (as today) | About 1 km. Kept alongside GAMI **(my call)** |
| `root_depth` and 5 others | Fan 2017, 30 arc-sec | Native pixel, existing nearest-valid search | Kept alongside Han **(my call)** |
| `land_share` (new) | ESA WorldCover v200 | Share of the cell that is land **(my call)** | Lets the coverage report name cells that are mostly water |

Soil coverage: the old 1 km ring search is dropped because the cell mean already covers that footprint.

**Stand age keeps today's lookup (my call).** Palma's `extract_stand_age.py` reads the nearest pixel with no search. A
search would change more than this column: the merge falls back to the ground `st_age` only when the mean of the TC
columns is empty, so a neighbour value could displace field data, and a partial fill would shift the mean elsewhere.

Known consequence: the owner's earlier test found no skill gain for canopy height at 1 km (r 0.75 vs 0.74, bias
-9.5 vs -8.8 m). The owner chose 1 km for consistency, knowing this.

Measured for terrain on 2026-11-02, also a consequence of that choice: 1 km elevation stays close to today's (r 0.998,
median change 8 m), slope less so (r 0.90), and aspect changes most (`aspect_sin` r 0.38). The cell-mean aspect vector has
a median length of only 0.20, because most 1 km cells hold no single direction. A valley-floor stand also takes on its
valley's slope: `CHE_PFY_CON` and `CHE_PFY_IRR` go from 5 to 26 degrees.

## New features

| # | Dataset | Access | Native | Rule | Columns |
|---|---|---|---|---|---|
| 1 | Han 2025 rooting depth, Zenodo 15361409 | `History.rar` (24 MB, 26 NetCDF, 2015 only) | 0.05 deg | Native pixel, tree types only | `root_depth_han`, `root_depth_han_uncertainty`, `root_depth_han_type` |
| 2 | Yang 2024 wood density, Zenodo 10692059 | `global-wood-density-main.zip` (1.1 GB), four `.mat` prediction files, no GeoTIFF | 0.04 deg | Native pixel, mean of the four models | `wood_density`, `wood_density_sd` |
| 3 | Lusk 2026 traits, Zenodo 14646322 | All 31 GeoTIFFs, 20.3 GB, EPSG:6933, scaled int16 | 1 km | Native pixel | See below |
| 4 | Liang 2022 richness, figshare 17232491 | `S_mean_raster.tif` (51 MB) | 0.025 deg | Native pixel | `tree_richness_per_ha` |
| 5 | Lesiv 2022 forest management, Zenodo 5879022 | `FML_v3-2_with-colorbar.tif` (1.6 GB) | 100 m | Share of each class, plus dominant class | `fm_share_{11,20,31,32,40,53}`, `fm_dominant` |
| 6 | MOD44B.061 tree cover, Earth Engine | `Percent_Tree_Cover` | 232 m | Cell mean; 0% counts, water and fill are masked in the asset | `tree_cover_pct` |
| 7 | ESA CCI Biomass v7.0, CEDA | `ESACCI-BIOMASS-L4-AGB-MERGED-1000m-fv7.0.nc` (3 GB), variables `agb` and `agb_sd`, 18 years | 0.01 deg | Native pixel | `agb_mg_ha`, `agb_sd` |
| 8 | GAMI v2.1, GFZ | `GAMIv2-1_2010-2020_100m.nc` (218 GiB), variable `forest_age`, 20 members | 1/1125 deg | Cell mean of forest pixels; non-forest excluded; mean over the 20 members **(my call)** | `forest_age_gami`, `forest_age_gami_coverage` |
| 9 | MOD13Q1.061 EVI, Earth Engine | `EVI`, split by `SummaryQA` class | 232 m | Land cell mean per 16-day image and class | `evi_mean`, `evi_amplitude`, `evi_years_used` |
| 10 | MCD12Q2.061 phenology, Earth Engine (owner's request, 2026-11-02) | Every pixel-year of the cell, read one by one | 463 m | See "Phenology" below | `phen_*` |

**Fixed time window (owner's decision).** One fixed value per site from a common period. 138 of 226 records end before
2015, and record midpoints cluster around 2010-2014.

- Tree cover and biomass: mean of 2010-2012.
- Forest age: GAMI's 2010 map.
- Phenology: 2001-2020, the same as EVI **(my call)**, because both are seasonal summaries.
- All available years are still extracted into the long table.

**EVI (owner's decision).** Summaries, not a series, over 2001-2020. For each year, take the mean and p90 minus p10 of
the cell's composites, then average across years. p90-p10 is used so a single bad composite cannot set the amplitude
**(my call)**.

**EVI quality filter (my call).** MOD13Q1 classes are 0 good, 1 marginal, 2 snow/ice, 3 cloudy.

- The summaries use classes 0 and 1. Keeping only class 0 could leave few composites in cloudy tropical years.
- Snow and cloud are excluded, so **boreal values describe the snow-free season** while tropical values describe the
  whole year. EVI over snow reads the snow, not the canopy.
- A composite counts for a cell when at least half its land pixels are valid. A year counts with at least 6 valid
  composites, about three months. `evi_years_used` reports how many of the 20 years counted, and no qualifying year
  means `no_data`.
- The extraction stores mean EVI and pixel count per composite and per class in `evi_composites.csv`. Any other filter
  can then be recomputed from counts without going back to Earth Engine.

**Phenology (owner asked for the dataset; these rules are my call).** MCD12Q2 gives yearly dates and greenness measures
per vegetation cycle. Each cell holds 1 to 4 of its pixels, so every pixel-year is read and stored in
`phenology_pixels.csv`, and the summaries are computed from that file, so any rule below can change without going back
to Earth Engine. Measured at the 226 sites over 2001-2020:

- **A missing cycle is a real answer, not missing data.** Earth Engine masks a pixel-year where no cycle passed the
  product's test (amplitude at least 0.1, and at least 35% of the three-year range), and `NumCycles` is never 0. 31% of land
  pixel-years have no cycle: EBF 69%, WSA 59%, but DBF 15% and DNF 0%. 21 sites never show one, including `AUS_CAN`, `AUS_MAR`
  (most), `CHN_ARG`, `CHN_YIN`, and `USA_PJS`. So `phen_cycle_share`, the share of land pixel-years with a cycle, is itself
  a feature, defined at every site with land. Dates at a site with no cycle are `no_data`.
- **Dates are circular, as aspect is.** Each date becomes the sine and cosine of its day of year, averaged over the cell's
  pixels and then over years, not renormalised. An arithmetic mean of day of year would misplace the peak by more than 30
  days at 22 of 204 sites, by up to 171 days: `ZAF_RAD` would peak in July instead of late December. A cycle belongs to
  the year of its peak, so 9.4% of greenups fall in the year before.
- **Lengths are linear.** `phen_season_days` (dormancy minus greenup) and `phen_midseason_days` (mid-greendown minus
  mid-greenup) are taken per pixel-year before any averaging.
- **The main cycle is the larger one.** `_1` only means the earlier of two: it came first in all 84 two-cycle pixel-years,
  and was the larger in 41%. Each pixel-year therefore uses whichever cycle has the larger amplitude. `phen_num_cycles` is the
  mean of `NumCycles` with no cycle as 0.
- **All seven dates are kept, with no quality filter.** `QA_Overall` is best in 93.7% of pixel-years, good in 6.1%, fair in
  0.1%, and poor never, so a filter would change almost nothing. The guide warns that greenup and dormancy can be
  anomalous at high latitudes with low amplitude, and recommends the mid-season dates there; `provenance.json` records this.
- **A year counts when at least one land pixel has a cycle**, and `phen_years_used` reports how many of the 20 did: 204
  sites have at least one year, 196 at least six. No rule on a minimum is imposed, and the count is there for the model.

Columns: `phen_{greenup,midgreenup,maturity,peak,senescence,midgreendown,dormancy}_{sin,cos}`, `phen_season_days`,
`phen_midseason_days`, `phen_evi2_min`, `phen_evi2_amplitude`, `phen_evi2_area`, `phen_num_cycles`, `phen_cycle_share`,
`phen_years_used`.

**Fixed by the product:** Lesiv is 2015, Han is 2015 (its other years are future scenarios), Liang and Yang are static.

**Lusk traits (owner's decision).** All 31 traits are extracted, and the model decides later which are useful. Columns
are `lusk_<TRY id>` with a readable name in `provenance.json` **(my call)**, for example `lusk_3117` for SLA, except
`root_depth_lusk` (6) and `wood_density_lusk` (4), which the agreement test names. Each trait also writes its coefficient
of variation and area-of-applicability bands, so 93 columns in all; those 62 band columns go to `lusk_quality.csv` and stay
out of the merge **(owner's decision, 2026-11-02)**. The authors' cross-validated r runs from 0.22 (wood
ray density) to 0.63 (SLA, leaf N per area), and 15 of the 31 are flagged experimental, below their r = 0.5 threshold.
`provenance.json` records each trait's r and flag so feature selection can see them. The maps cover all vegetation, not
trees only. The bands are stored as int16 with a per-band scale and offset (rooting depth: 0.01132 and 370.9), so a read
that ignores them returns nonsense. Each file's tags carry its own trait name, unit, and r, and also say "research
purposes only, no redistribution"; the extracted site values stay inside this project.

**Han (owner's decision: no matching on `si_igbp`).** Han has 13 files, one per plant type, 8 of them trees, but they do
not overlap: measured on the 2015 files, no pixel on Earth holds more than one type, so Han is one depth per pixel, and
there is nothing to average or match. `root_depth_han_type` records which type it is.

At the 226 sites, 144 pixels hold a tree type, 34 a shrub or grass type, and 48 nothing. **Tree types only, with the
1-pixel neighbour rule (my call):** this gives 203 sites (144 `ok`, 59 `neighbour`) and leaves 23 `no_data`. A grass or
shrub depth is not used for a tree measurement, even though accepting them would reach 215 sites. Reaching 3 pixels would
add only 8 sites while reaching about 17 km away or more. Han's type also need not match the stand: Hyytiala reads
`NETTe` (temperate), not `NETBo`. The 23 are not random: most are dry or open stands (`ESP_SAN`, the `ESP_LM` dehesa,
`SEN_SOU`, `CHN_ARG`, `ISR_YAT`, `UZB_YAN`), where rooting depth may matter most.

**Yang (opened 2026-11-02).** The record's description and its prediction script say 0.01 degrees, but the deposited maps
are 9000 x 4500, so 0.04 degrees, about 4.4 km. It is therefore read by native pixel with the 1-pixel rule for coarser
products. There are four model files (RF, LightGBM, XGBoost, LightGBM-RF), each holding 8 cross-validation layers and
their mean. `wood_density` is the mean of the four models, as the record's "ensemble of four" describes, and
`wood_density_sd` is the standard deviation across all 32 layers **(both my call)**. The files carry no coordinates, so
the orientation was tested: 180 of 226 sites hold a value with north up and longitude from the dateline, at most 12 with
any other reading, and a map of the mask then shows the continents in place (central Amazon 98% valid, Sahara 0%). A 0 is
the files' mask, not a density: 9.8% of finite pixels in every model, and no value lies between 0
and 0.27. A few pixels read impossibly high (up to 2.15 g/cm3), but none at the sites, where the mean runs from 0.44 to 0.66.
The map covers forest only, giving 197 sites (180 `ok`, 17 `neighbour`) and 29 `no_data`. Reaching 3 pixels would add 6
while reaching about 13 km. The gaps are again mostly dry and open stands (`AUS_CAN`, `AUS_CAR_THI`, `CHN_ARG`, `CHN_YIN`,
`ESP_SAN`, `ESP_LAS`, `SEN_SOU`, `UZB_YAN_DIS`).

**Liang coverage (opened 2026-11-02).** The map holds a value at the own pixel of 185 sites. With the 1-pixel rule for
coarser products that becomes 196 (185 `ok`, 11 `neighbour`) and 30 `no_data`; reaching 3 pixels would add only 10. Its
gaps overlap Han's dry and open stands (`ESP_SAN`, the `ESP_LM` dehesa, `SEN_SOU`, `CHN_ARG`, `ISR_YAT`).

**Forest management shares.** The denominator is all pixels in the cell, so the shares sum to the forest share.

## Reason codes

Each value has a `<column>_status` of `ok`, `neighbour`, or `no_data`. A request failure is never a status: it retries
as transient, then stops the run naming the site and product.

`neighbour` means the nearest valid pixel was used: within 3 pixels for products near 1 km, Lusk and CCI (the module's
existing rule), within 1 pixel for coarser products, Han, Yang, and Liang, so the reach stays short **(my call)**.
Aggregated products have no neighbour search: no valid pixels means `no_data`. These reaches apply to new products only. Unchanged products keep
today's lookup exactly, so they can reproduce.

## Earth Engine traps to respect

From `ee_scale_is_the_grid`, `canopy_height_ground_agreement_result`, `gee_project_id`, and `aspect_centre_pixel`:

- Take the scale server-side from the image, never a rounded literal. A rounded scale is a different grid.
- A `.mosaic()` loses the native projection, so take it from a member image.
- Terrain derivatives are computed in the requested projection, so request the DEM's native one.
- Percentile reducers need fixed buckets (`maxBuckets=1024, minBucketWidth=0.125`) to be repeatable.
- Earth Engine project is `era5download-447713`, overridable by argument. Restricted Mode can return under load.
- Reducers name their outputs after the bands only on a multi-band image. A one-band image returns `mean`, and a null can
  mean a masked pixel or a typo'd band, so requests always carry at least two bands and band names are checked first.
- SoilGrids bands do not share one grid (`bdod` and `soc` have a different origin), so each property takes its own.

## Outputs

Separate root under `data/raw/extracted_data/`, for example `static_1km/<scale>/` **(my call)**:

- Files the merge already reads, same names and columns: `site_info_with_env_data.csv`, `soilgrids_data.csv`,
  `stand_age_data.csv`, `site_info_with_root_depth_fan2017.csv`.
- `new_static_features.csv`: one row per site, the new columns.
- `static_quality.csv`: one row per site, every `<column>_status` and `<column>_coverage`, for old and new columns alike
  **(my call)**. They live apart because the merge's `load_stand_age_map` averages any column whose name contains
  `stand_age_TC`, so a status column beside the values would break it.
- `static_features_by_year.csv`: the long table (site, year, column, value, status).
- `evi_composites.csv`: site, composite date, QA class, mean EVI, pixel count.
- `lusk_quality.csv`: each Lusk trait's coefficient of variation and area of applicability, one row per site.
- `phenology_pixels.csv`: site, year, land flag, and every MCD12Q2 band per pixel.
- `provenance.json`: source URL, version, checksum, time window, code commit.

The merge always reads this root and carries the new columns as static. There is no switch **(owner's decision,
2026-11-02, replacing an opt-in switch that defaulted to off)**. A missing file or site stops the merge instead of
becoming NaN columns, so a scale must be built with this package before it can be merged.

## Data on disk

About 245 GB new, under `data/raw/grided/<product>/`, against about 710 GiB free on scratch (4408 of 5120 GiB used).
GAMI is 218 GB of that. Compute jobs do the downloads and check repository checksums.

Van Tiel species maps (13.5 GB, 849,276 files) are **not** downloaded. Liang replaced them.

## Checks before the module counts as done

1. **Unchanged products reproduce:** WorldClim, Koppen, BGI, and Fan match the current CSVs at every site, where a
   match is the same value or empty in both, except where `checks/reproduce.py` names the site, the outcome and the
   reason (the three corrected coordinates, CHN_YUN_YUN's WorldClim neighbour, and since 2026-11-04 the two French
   Guiana stands, for Fan only). BGI TC000/010/020/030 were empty at 15/33/36/37 sites in the August
   audit, and today's code cannot say why, because masked pixels and swallowed exceptions both return `None`. A site
   that was empty and now has a value is a swallowed failure, not a pass: it stops the check, is listed by name, and
   goes to the owner, because it changes the merge's `st_age` fallback.
2. **Changed products are compared and reported:** old vs new per column, r and largest movers. Canopy height should
   agree with the existing 1 km probe in `probes/canopy_height_ground/`, allowing for its site-centred footprint.
3. **Unit tests with hand-computed answers** using synthetic rasters: cell indexing at the dateline and poles, zero
   versus missing, aspect across the 0/360 seam, phenology dates across the New Year seam, shares summing correctly,
   reason codes, failure stops the run.
4. **Range and unit checks per product**, plus a coverage report by product and reason. Known counts from opening the data
   are acceptance numbers: Han 144 `ok`, 59 `neighbour`, 23 `no_data`; Yang 180, 17, 29; Liang 185, 11, 30; phenology dates
   204 sites with a value before the land mask change.
5. Everything runs on Palma's Python 3.9 through `sbatch`, including pytest. No Python on the login node.
6. **Agreement between sources (owner's request):** `root_depth_lusk` vs `root_depth_han` and `wood_density_lusk` vs
   `wood_density`. Fan's `root_depth` joins as a third rooting source **(my call)**. Each pair gets Pearson and Spearman
   r, bias, and RMSE, with bootstrap intervals, after collapsing sites that share a pixel so duplicates do not inflate
   the fit, overall and by `si_igbp` **(my call)**. Units are checked before comparing. This is a report, not a gate: both
   sources are kept whatever it shows. Some disagreement is expected, because Lusk averages all vegetation while Yang and
   Han describe trees, most of all at SAV and WSA sites.

## Owner decisions, 2026-11-04

Taken after the module was built and reviewed, and implemented in `bbc3ff7`, `0d8cd75`, `4fe6a19` and `10e1220`:

- **Phenology: seasonality, not MCD12Q2 dates.** MCD12Q2 keeps a cycle only when green-up and green-down each exceed
  0.1 EVI2 and 35%, so its dates were empty at the 20 least seasonal stands, and the trainer drops every row with an
  empty feature. The merge now takes, of MCD12Q2, only `phen_cycle_share` and `phen_num_cycles`, and gains the annual
  harmonic of the EVI composites: `evi_season_amplitude`, `evi_season_r2`, and `evi_peak_cos_w`, `evi_peak_sin_w`,
  the amplitude times the cosine and sine of the annual peak, which fade to (0, 0) with the season. Composites
  under snow (class 2 on at least half the land pixels, in a counted year) enter the fit at the site's 10th-percentile
  EVI, so a winter with no valid composite reads as dormant, not as an extrapolated peak (`10e1220`, my call). The
  phenology part is still built in full.
- **GAMI, Han, Liang and Yang stay out of the merge**, too many sites lack them. Their parts, their status in
  `static_quality.csv`, and the agreement report (Lusk against Han and Yang) remain.
- **The other gaps are filled by neighbour search**: the nearest node holding a value within 25 km
  (`src/static_features/neighbours.py`), the farthest the hold-out tests measured. Each fill has status `neighbour` and
  is listed with its distance in `<product>_neighbours.csv`. Lesiv borrows only `fm_dominant`, its shares staying the
  cell's own; soil borrows column by column; BGI's borrowed age goes in its own column, `stand_age_neighbour`, which the
  merge reads after the ground `st_age`, so a pixel kilometres away never displaces an age measured at the stand
  (my call). This reverses the earlier "BGI with no search".
- **Fan reads the continent file that holds the site.** The reader tries overlapping boxes in order and reads the first
  whose file has a valid pixel within reach; the French Guiana stands now read SAMERICA, not the empty NAMERICA file.

## Known risks

- **Han archive.** A real RAR file, with no unrar tool found on Palma. Windows `tar` reads it, and a `.tgz` copy with
  all 26 files unpacked is in `/scratch/tmp/yluo2/han_probe/data/`. Opened 2026-11-01: variable `depth`, 3600 x 7200, so
  the README's "0.05 radian" means 0.05 degrees.
- **GAMI.** The authors recommend binning into classes of at least two decades, because absolute ages at 100 m carry
  biases. The 1 km mean over 20 members smooths this but does not remove it, and `provenance.json` records the warning.
  Its 218 GiB must be read with a large chunk cache, because each compressed chunk holds about 120 million pixels.
- **Zenodo refuses Python.** `urllib`, `requests`, and GDAL's HTTP reader all get 403, while `curl` works from compute nodes
  (checked 2026-11-02), so every download goes through `curl`.
- **Overlap between features.** Lusk's model uses climate and canopy height, GAMI uses CCI biomass v5, and MCD12Q2's
  greenness measures partly repeat EVI, so several new features partly repeat existing ones. Feature selection can sort
  this out; the extraction does not try to.
- **Landscape versus stand.** Plots sit in denser forest than their surroundings, so 1 km values often read low.
- **Gaps carry information.** Phenology, Han, Yang, and Liang are all empty most often at dry or evergreen stands, so a
  model can learn from the gap itself rather than the value. Since 2026-11-04 none of them reaches the merge with
  gaps; the borrowed values that fill other gaps are landscape values from up to 25 km away, marked `neighbour`.
- **Wrong coordinates get plausible values.** `GBR_DEV_CON` and `GBR_DEV_DRO` plotted in the Forth estuary, and land-only
  means gave them values from the shore. They are now extracted at corrected coordinates (`COORDINATE_CORRECTIONS`,
  owner 2026-11-02), as is `CHE_DAV_SEE`; the coverage report still lists every site whose land share is below 50%.
- **The fixed grid can split a stand from its surroundings.** `CHE_DAV_SEE` and `CHE_DAV`, the same forest 150 m apart,
  fell in different cells, one 70% Lake Davos. `CHE_DAV_SEE` now takes `CHE_DAV`'s coordinates, so the two share every
  static value, and they can land in different folds.
- **Later serving cost.** Canopy height per cell is roughly 16x the pixels of the current 250 m disc. Fine for 226
  sites, but must be costed before running 565k grid cells.

## Before building

The current module exists only in Palma's working tree: `terrain_aspect.py`, `src/site_info/`, and `src/serving/` are
untracked, and six extractors have uncommitted edits. A snapshot commit should come first, because scratch can be
purged and a worktree would not contain untracked code. Committing needs the owner's agreement.

Proposed **(my call on the branches)**: commit that code without the backup files on `feat/sapwood-combined` and push it
from Palma, then build the module on a new branch `feat/static-features` in its own worktree.

## Left to the builder

Internal layout of the package, the product registry's shape, job script structure, download and retry mechanics, and the
way the merge reads the root.
