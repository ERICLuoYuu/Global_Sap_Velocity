# Technical Documentation: Global Sap Velocity Prediction Project

**Version:** 1.0
**Last updated:** 2026-03
**Author:** [FILL IN: your name]
**Supervisors:** [FILL IN: Mana, and others]
**GEE Project:** `ee-yuluo-2`

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Data Sources](#2-data-sources)
3. [Dataset Summary Statistics](#3-dataset-summary-statistics)
4. [Data Processing and Quality Control](#4-data-processing-and-quality-control)
5. [Non-Standard Data Sources (Sapflow-internal)](#5-non-standard-data-sources-sapflow-internal)
6. [Data Integration and Merging](#6-data-integration-and-merging)
7. [Feature Engineering](#7-feature-engineering)
8. [Model Architecture and Training](#8-model-architecture-and-training)
9. [Cross-Validation Strategies](#9-cross-validation-strategies)
10. [SHAP Interpretability Analysis](#10-shap-interpretability-analysis)
11. [Results and Evaluation](#11-results-and-evaluation)
12. [Global Prediction Pipeline](#12-global-prediction-pipeline)
13. [Post-Prediction Analysis](#13-post-prediction-analysis)
14. [Project Structure and Setup](#14-project-structure-and-setup)
15. [Limitations and Future Work](#15-limitations-and-future-work)
16. [References](#16-references)

---

## 1. Introduction

### 1.1 Scientific Motivation

Sap velocity — the rate at which water moves through the xylem of trees — is a direct indicator of whole-tree transpiration and a key variable linking forest carbon assimilation, water-cycle feedback, and ecosystem response to climate variability. While point-level sap flow instruments have generated rich in-situ datasets at individual forest sites, scaling these observations to continental or global extents remains a fundamental challenge in eco-hydrology and climate science.

Existing global land-surface models treat plant water use through parameterized functions of vapour-pressure deficit (VPD) and soil moisture, but they rarely capture the structural, taxonomic, and microclimatic diversity that shapes real-world sap velocity across forest types and biomes. Machine learning offers an empirical bridge: trained on the largest available compilation of sap flow observations and a suite of meteorological and structural covariates extracted from satellite and reanalysis products, a well-validated model can produce spatially continuous, daily or hourly sap velocity estimates anywhere on Earth where trees exist.

This project develops, validates, and applies such a model. The trained model generates the first globally coherent sap velocity dataset at **0.1° × 0.1°** resolution (matching ERA5-Land native resolution, ~11 km at equator), enabling downstream analyses of forest water use, transpiration trends, and biome-level responses to drought and warming.

### 1.2 Research Objectives

1. Compile and harmonize the largest possible database of in-situ sap velocity observations from SAPFLUXNET and internal Sapflow-internal datasets, applying rigorous quality control.
2. Extract a consistent set of environmental, structural, and phenological covariates at each observation site using Google Earth Engine (GEE) and ancillary datasets.
3. Train and compare multiple machine learning models (XGBoost, Random Forest, ANN, CNN-LSTM, Transformer) under spatially independent cross-validation to select the best-generalising architecture.
4. Quantify feature importance through SHAP analysis, identifying which drivers most strongly control sap velocity across biomes, seasons, and time-of-day.
5. Apply the trained model with global ERA5-Land forcing to produce gridded sap velocity predictions across all forested land areas (60°S–78°N).
6. Analyse predicted global patterns by climate zone and plant functional type, and evaluate physical plausibility against independent literature estimates.

### 1.3 Methodological Summary

In-situ sap velocity observations from [FILL IN: N] sites across [FILL IN: N] biomes are quality-controlled and merged with co-located hourly ERA5-Land meteorological forcing, GlobMap LAI time series, MODIS plant functional type (PFT), canopy height, terrain attributes, WorldClim bioclimatic variables, SoilGrids properties, and forest stand age. The merged dataset is used to train tree-ensemble and deep-learning regression models under stratified spatial cross-validation. The best-performing model is then applied globally using GEE-extracted ERA5-Land fields and satellite-derived structural features to produce gridded sap velocity maps. SHAP TreeExplainer analysis identifies the dominant drivers of model predictions at global, biome, and diurnal timescales.

---

## 2. Data Sources

### 2.1 SAPFLUXNET Database

| Property | Value |
|---|---|
| **What it contains** | Standardized sap flow velocity and environmental measurements from peer-reviewed forest sites worldwide |
| **Variables** | Sap flow velocity (cm h⁻¹ or equivalent), air temperature, vapour-pressure deficit, shortwave radiation, wind speed, precipitation, soil moisture (site-dependent) |
| **Spatial coverage** | Global; [FILL IN: N] sites in [FILL IN: N] countries |
| **Temporal coverage** | [FILL IN: year range, e.g., 1995–2022], site-dependent |
| **Temporal resolution** | Sub-hourly (30-min to 2-h depending on site), resampled to 1-h |
| **Access method** | Direct download from SAPFLUXNET repository; stored locally at `data/raw/0.1.5/…/csv/sapwood/` |
| **Version** | 0.1.5 |
| **Data format** | Per-site CSV files: `*_sapf_data.csv`, `*_env_data.csv`, `*_site_md.csv` |
| **Known limitations** | Heavy geographic bias toward temperate Europe and North America; sparse coverage in tropical regions, boreal Asia, and southern hemisphere; variable measurement methods across sites (heat-pulse, thermal dissipation, heat-ratio) affect absolute magnitudes |
| **Citation** | Poyatos, R. et al. (2021). Global transpiration data from sap flow measurements: the SAPFLUXNET database. *Earth System Science Data*, 13(6), 2607–2649. https://doi.org/10.5194/essd-13-2607-2021 |

The SAPFLUXNET database is already structured in a consistent format and requires no reorganization. Each site provides a sap flow file (one column per tree-species combination), an environmental file (co-located meteorological measurements), and a site metadata file containing geographic coordinates, stand characteristics, and measurement methodology.

### 2.2 Sapflow-internal Dataset

| Property | Value |
|---|---|
| **What it contains** | Proprietary sap flow measurements from internal research sites, primarily in Europe |
| **Variables** | Sap velocity, ICOS ecosystem flux tower environmental data |
| **Format** | Non-standard; heterogeneous across sites |
| **Access method** | Local files in `Sapflow-internal/` directory |
| **ICOS integration** | Sites co-located with ICOS flux towers; environmental data extracted from `icos_env_extractor.py` and `icos_fluxnet_extractor.py` |
| **Known limitations** | Heterogeneous raw formats require bespoke parsing; see Section 5 for site-specific details |

These sites are not part of the public SAPFLUXNET release and therefore require a dedicated ETL pipeline (see Section 5).

### 2.3 ERA5-Land Reanalysis — Site-Level Extraction

| Property | Value |
|---|---|
| **What it contains** | Hourly land-surface meteorological reanalysis fields co-located at each observation site |
| **Variables extracted** | `temperature_2m`, `dewpoint_temperature_2m`, `total_precipitation`, `surface_solar_radiation_downwards`, `10m_u_component_of_wind`, `10m_v_component_of_wind`, `volumetric_soil_water_layer_1`, `soil_temperature_level_1`, `potential_evaporation` |
| **Spatial resolution** | 0.1° × 0.1° (~9 km) |
| **Temporal resolution** | Hourly |
| **Access method** | Google Earth Engine `ImageCollection("ECMWF/ERA5_LAND/HOURLY")`; GEE project `ee-yuluo-2` |
| **Script** | `src/Extractors/extract_climatedata_gee.py` |
| **Output** | `data/raw/extracted_data/era5land_site_data/sapwood/era5_extracted_data.csv` |
| **Known limitations** | ERA5 VPD derived from 2-m dewpoint temperature can differ from site-level VPD computed from measured T and RH; documented in `notebooks/VPD_mismatch_analysis.py` |
| **Citation** | Muñoz-Sabater et al. (2021) |

### 2.4 GlobMap LAI — Site-Level Extraction

| Property | Value |
|---|---|
| **What it contains** | Leaf Area Index (LAI) time series at each observation site |
| **Product** | GlobMap LAI v3 — global 8-day composite, 500 m resolution |
| **Access method** | **Local pre-downloaded GeoTIFF files** at `data/raw/grided/globmap_lai/GlobMapLAIV3.A*.Global.LAI.tif`; must be downloaded by user before running extraction |
| **Script** | `src/Extractors/extract_globmap_lai.py` |
| **Output** | `data/raw/extracted_data/globmap_lai_site_data/sapwood/extracted_globmap_lai_hourly.csv` |
| **Temporal interpolation** | 8-day composites interpolated to hourly resolution during merge |
| **Known limitations** | 500 m footprint may not match sap flow measurement footprint for heterogeneous canopies; gap-filled values during cloud cover periods |
| **Citation** | Yuan, H. et al. (2011). Reprocessing the MODIS Leaf Area Index products for land surface and climate modelling. *Remote Sensing of Environment*, 115(5), 1171–1187. https://doi.org/10.1016/j.rse.2011.01.001 |

> **Note:** A second LAI extraction script (`src/Extractors/extract_lai.py`) retrieves a MODIS+AVHRR fusion product via GEE and writes to `src/Extractors/`. This is a separate analysis tool and its output is **not** consumed by the merge pipeline.

### 2.5 MODIS Plant Functional Type (PFT)

| Property | Value |
|---|---|
| **What it contains** | Annual land-cover classification per site mapped to IGBP PFT scheme |
| **Product** | MODIS MCD12Q1 v061, `LC_Type1` band |
| **Spatial resolution** | 500 m |
| **Temporal resolution** | Annual |
| **Access method** | Google Earth Engine `ImageCollection("MODIS/061/MCD12Q1")` |
| **Script** | `src/Extractors/extract_pft.py` |
| **Output** | `data/raw/extracted_data/landcover_data/sapwood/landcover_output.csv` |
| **One-hot encoding** | PFT classes one-hot encoded for model input: `MF`, `DNF`, `ENF`, `EBF`, `WSA`, `WET`, `DBF`, `SAV` |

### 2.6 Canopy Height and Terrain Attributes

| Property | Value |
|---|---|
| **Canopy height product** | Meta/Facebook global canopy height at 1.2 m scale (`projects/sat-io/open-datasets/facebook/meta-canopy-height`) |
| **Terrain product** | ASTER Global DEM at 30 m (`projects/sat-io/open-datasets/ASTER/GDEM`) |
| **Access method** | Google Earth Engine |
| **Script** | `src/Extractors/extract_canopyheight.py` |
| **Output** | `data/raw/extracted_data/terrain_site_data/sapwood/site_info_with_terrain_data.csv` |
| **Columns extracted** | `canopy_height_m`, `elevation_m`, `slope_deg`, `aspect_deg`, `aspect_sin`, `aspect_cos` |
| **Sequential dependency** | This output is the **input** to `spatial_features_extractor.py`; must be run first |

### 2.7 WorldClim Bioclimatic Variables and Köppen-Geiger Classification

| Property | Value |
|---|---|
| **What it contains** | Long-term mean climate normals (19 bioclimatic variables, e.g., BIO1 = mean annual temperature, BIO12 = annual precipitation) and climate zone classification |
| **Access method** | **Auto-downloaded at runtime** by `spatial_features_extractor.py` from worldclim.org and Figshare; cached at `data/raw/grided/worldclim/` and `data/raw/grided/koppen_geiger/` |
| **Script** | `src/Extractors/spatial_features_extractor.py` |
| **Input** | `terrain_attributes_data_path` (output of `extract_canopyheight.py`) — not `site_info.csv` directly |
| **Output** | `data/raw/extracted_data/env_site_data/sapwood/site_info_with_env_data.csv` |
| **Variables used in model** | `mean_annual_temp`, `mean_annual_precip`, Köppen zone |

### 2.8 SoilGrids Soil Properties

| Property | Value |
|---|---|
| **What it contains** | Soil physical and chemical properties at each site |
| **Access method** | **REST API**: `https://rest.isric.org/soilgrids/v2.0/properties/query`; no local pre-download needed |
| **Script** | `src/Extractors/extract_soilgrid.py` |
| **Output** | `data/raw/extracted_data/terrain_site_data/sapwood/soilgrids_data.csv` |
| **Known limitations** | REST API rate limits may slow batch extraction; 250 m resolution may not reflect local soil heterogeneity |

### 2.9 Forest Stand Age

| Property | Value |
|---|---|
| **What it contains** | Forest age (years) at each site |
| **Product** | BGI Global Forest Age dataset v1.0 |
| **Access method** | **Local pre-downloaded NetCDF** at `data/raw/grided/stand_age/2026113115224222_BGIForestAgeMPIBGC1.0.0.nc`; must be downloaded by user |
| **Script** | `src/Extractors/extract_stand_age.py` |
| **Output** | `data/raw/extracted_data/terrain_site_data/sapwood/stand_age_data.csv` |
| **Citation** | Poulter, B. et al. (2019). The global forest age dataset and its uncertainties (GFA_v1). *Earth System Science Data*, 11, 1793–1808. https://doi.org/10.5194/essd-11-1793-2019 *(Confirm version; file name suggests BGI/MPI-BGC 1.0.0 product)* |

### 2.10 ERA5-Land — Global Gridded (Prediction Phase)

| Property | Value |
|---|---|
| **What it contains** | Hourly ERA5-Land fields at every grid cell globally, used only during global prediction — not during training data extraction |
| **Access method** | Google Earth Engine `ERA5LandGEEProcessor`; per-cell extraction fetches the same variables listed in Section 2.3 |
| **Temp cache** | `D:/Temp/era5land_extracted/` — intermediate extracted data; can be deleted after prediction |
| **Script** | `src/make_prediction/process_era5land_gee_opt_fix.py` |
| **Spatial domain** | 60°S–78°N, 180°W–180°E |

---

## 3. Dataset Summary Statistics

> All statistics in this section should be traced to the outputs of specific analysis scripts and regenerated when the dataset changes.

### 3.1 Overall Dataset

| Metric | Value |
|---|---|
| Total sites | [FILL IN] |
| Total site-years | [FILL IN] |
| Total hourly observations (after QC) | [FILL IN] |
| Total daily observations (after QC) | [FILL IN] |
| Temporal range | [FILL IN: e.g., 1995–2023] |
| Number of biomes | 9 |
| Number of countries | [FILL IN] |

*Generated by: [FILL IN: script name and run date]*

### 3.2 Target Variable Distribution

The target variable is sap velocity (cm h⁻¹ at sapwood scale).

| Statistic | Value |
|---|---|
| Mean | [FILL IN] cm h⁻¹ |
| Median | [FILL IN] cm h⁻¹ |
| Standard deviation | [FILL IN] cm h⁻¹ |
| 5th–95th percentile range | [FILL IN] cm h⁻¹ |
| Maximum observed | [FILL IN] cm h⁻¹ |
| Fraction of zero or near-zero values | [FILL IN]% |

**Distribution shape:** Strongly right-skewed with a large mass near zero corresponding to nighttime and dormant-season observations. `log1p` transformation significantly reduces skewness — see Section 7.4.

### 3.3 Biome Distribution

| Biome | Sites | % of total sites | Notes |
|---|---|---|---|
| Boreal forest | [FILL IN] | [FILL IN]% | |
| Subtropical desert | [FILL IN] | [FILL IN]% | |
| Temperate forest | [FILL IN] | [FILL IN]% | Largest class — potential class imbalance |
| Temperate grassland/desert | [FILL IN] | [FILL IN]% | |
| Temperate rain forest | [FILL IN] | [FILL IN]% | |
| Tropical forest/savanna | [FILL IN] | [FILL IN]% | Underrepresented |
| Tropical rain forest | [FILL IN] | [FILL IN]% | Underrepresented |
| Tundra | [FILL IN] | [FILL IN]% | Very sparse |
| Woodland/Shrubland | [FILL IN] | [FILL IN]% | |

The uneven biome distribution motivates the use of **stratified** spatial cross-validation (Section 9.2) and should be acknowledged as a limitation in per-biome prediction uncertainty (Section 15).

### 3.4 Missing Data and Completeness

| Variable | % complete (all sites) | Notes |
|---|---|---|
| Sap velocity | [FILL IN]% | After QC; missingness varies strongly by season |
| ERA5 T, VPD, SW | ~100% | Reanalysis has no gaps |
| GlobMap LAI | [FILL IN]% | 8-day composites; cloud gaps filled |
| PFT | ~100% | Annual; replicated across time |
| Canopy height | [FILL IN]% | Sparse coverage in some tropical areas |
| SoilGrids | [FILL IN]% | A few sites in data-sparse regions |
| Stand age | [FILL IN]% | |

---

## 4. Data Processing and Quality Control

### 4.1 Overview

Quality control is applied separately to sap flow and environmental data before merging. The pipeline consists of:

1. **Flag-based filtering** — remove data flagged as erroneous by site operators
2. **Manual QC** — apply expert removal decisions from `removal_log.csv`
3. **Automated outlier detection** — rolling-window z-score approach, separately for daytime and nighttime
4. **Variability filtering** — detect periods with abnormally low or high coefficient of variation (frozen sensors, sensor drift, baseline issues)
5. **Incomplete-day removal** — discard days with less than 50% valid daytime data

Scripts: `src/Analyzers/sap_analyzer.py` (`SapFlowAnalyzer`), `src/Analyzers/env_analyzer.py` (`EnvironmentalAnalyzer`), `src/Analyzers/mannual_removal_processor.py` (`RemovalLogProcessor`).

### 4.2 Flag-Based Filtering

SAPFLUXNET provides a `solar_TIMESTAMP` column and associated quality flags. Values associated with warning or error flags are set to NaN before any further processing. The exact flag codes and their handling are:

- **`REMOVE`** — value is unphysical or known erroneous; set to NaN
- **`WARN`** — value is suspicious; flagged records are written to `{col}_flagged.csv` for inspection and then set to NaN
- Other flag values: retained

*Implementation: `SapFlowAnalyzer._filter_flags()` called inside `_load_sapflow_data()`.*

### 4.3 Manual Quality Control

A structured removal log (`removal_log.csv`) captures expert decisions made after visual inspection of raw time series. The `RemovalLogProcessor` class applies these decisions during data loading, before any automated QC step.

The log supports three types of removal actions:

| Action | Description |
|---|---|
| `skip_site` | Entire site is excluded from the analysis |
| `remove_column` | A specific sensor column (tree individual) is excluded |
| `remove_period` | A date-range period within a column is set to NaN |

**Important:** Manual QC decisions are applied *before* automated outlier detection — they are not redundant with it. They capture failure modes that automated methods cannot detect: sensor relocation, known vandalism, unexplained step changes in baseline, or duplicate data from parallel sensors.

*To generate this summary, run the following after QC has been completed:*

```python
from src.Analyzers.sap_analyzer import SapFlowAnalyzer
analyzer = SapFlowAnalyzer(scale='sapwood')
# Stats are populated in analyzer.removal_stats after run_analysis_in_batches()
print("Sites skipped:", analyzer.removal_stats['sites_skipped'])
print("Columns removed:", analyzer.removal_stats['columns_removed'])
print("Total points removed:", analyzer.removal_stats['total_points_removed'])
```

*Fill in from the `outputs/processed_data/sapwood/sap/outliers/removal_report.csv` generated by `SapFlowAnalyzer.save_removal_report()`.*

### 4.4 Automated Outlier Detection

**Method:** Rolling-window z-score with separate treatment for daytime and nighttime observations. The daytime/nighttime partition is computed using `pvlib.solarposition.get_solarposition()` at each site's coordinates; any timestamp where solar elevation > 0° is classified as daytime.

**Rolling window:** Approximately 30 days of data at the native measurement frequency. The window adapts to the actual data interval of each site:

```python
pre_timewindow = self._get_timewindow(30 * 24 * 60 * 60, df.index)
DAY_SAPFLOW_WINDOW_POINTS  = pre_timewindow   # ~30-day window
NIGHT_SAPFLOW_WINDOW_POINTS = pre_timewindow  # same for night
```

A 30-day window is chosen rather than a global threshold because sap velocity shows strong seasonal dynamics; a global threshold would conflate summer high-flow with winter near-zero values, producing false positives.

**Thresholds:**

| Period | Z-score threshold (n_std) | Rationale |
|---|---|---|
| Daytime | 3 | Standard outlier criterion; daytime sap flow is well-structured |
| Nighttime | 5 | Nighttime values are near zero and noisier; a stricter threshold would remove legitimate low-flow events |

**Method flag:** The rolling-window method (`method='C'`) computes:

```
outlier if |x - rolling_mean| > n_std × rolling_std
```

### 4.5 Variability Filtering

A secondary filter removes periods of anomalously low or high variability — indicative of sensor freezing, baseline drift, or saturated outputs — which may pass outlier detection because they deviate from the rolling mean gradually.

**Window:** 2 days at the native measurement frequency.

**Thresholds applied separately to daytime and nighttime:**

| Period | Metric | Low threshold | High threshold | Rationale |
|---|---|---|---|---|
| Daytime | CV (std/mean) | 0.08 | 3.5 | CV < 0.08: suspiciously flat signal; CV > 3.5: erratic noise |
| Daytime | Rolling STD | 0.4 cm h⁻¹ | 55.0 cm h⁻¹ | Absolute bounds on variability |
| Nighttime | CV | — | 3.5 | Only upper bound for night (near-zero means small CV is expected) |
| Nighttime | Rolling STD | — | 10.0 cm h⁻¹ | |

*Implementation: `SapFlowAnalyzer._detect_variability_issues()`.*

### 4.6 Incomplete-Day Removal

After outlier and variability filtering, any calendar day for which fewer than 50% of expected daytime observations remain valid is removed entirely. This prevents days where the diurnal cycle is poorly sampled from biasing daily aggregations.

*Implementation: `SapFlowAnalyzer._remove_incomplete_days(completeness_threshold=0.5)`.*

### 4.7 Environmental Data QC

Environmental variables from site sensors are processed by `env_analyzer.py` (`EnvironmentalAnalyzer`). The key steps are:

1. Timezone adjustment to local solar time using the timezone mapping in `src/tools.py`
2. Outlier removal (same rolling z-score approach as sap flow, with variable-specific thresholds)
3. Precipitation zero-inflation handling: precipitation is inherently zero-inflated (many hours with zero rain). It is **not** subject to rolling z-score outlier removal, as zero is a physically valid value. In the merge step, precipitation is aggregated using `sum` (not mean) to obtain daily totals. The `prcip/PET` ratio clips the precipitation/PET ratio to [0, 10] to handle extreme outliers (e.g., very small PET denominators)
4. Standardisation of variable names to a common schema

**ERA5 data bypass:** ERA5-Land site-extracted data does **not** pass through `env_analyzer.py`. It is joined directly to the merge dataset because it is already quality-assured reanalysis data.

### 4.8 Output Files

| File | Contents |
|---|---|
| `outputs/processed_data/sapwood/sap/outliers_removed/*_sapf_data_outliers_removed.csv` | Per-site sap flow after all QC steps |
| `outputs/processed_data/sapwood/env/outliers_removed/*_env_data_outliers_removed.csv` | Per-site environmental data after QC |
| `outputs/processed_data/sapwood/sap/outliers/*` | Flagged-but-not-removed values (audit trail) |
| `outputs/processed_data/sapwood/sap/variability_filtered/*_filtering_stats.csv` | Per-site variability filter statistics |

### 4.9 Gap Filling (Separate Utility)

Scripts `notebooks/sap_gap_filling.py` and `notebooks/env_gap_filling.py` implement linear interpolation with a maximum gap of one time step. **These scripts are not part of the main merge pipeline.** They read from `sap/filtered/` and `env/filtered/` directories (not `outliers_removed/`) and produce outputs in `gap_filled_size1_after_filter/`. They were used for specific supplementary analyses but the merge script reads directly from the `outliers_removed/` directories.

---

## 5. Non-Standard Data Sources (Sapflow-internal)

### 5.1 Overview

The Sapflow-internal dataset contains sap flow measurements from internal research sites that are not part of the public SAPFLUXNET release. These sites differ from SAPFLUXNET in:

- Raw file format (non-standardised; each site may use different column naming, timestamp formats, or units)
- Environmental data source: co-location with ICOS flux tower network rather than on-site sensors

A dedicated ETL pipeline converts these sites to SAPFLUXNET-compatible format before they join the main processing pipeline.

### 5.2 ETL Pipeline

The Sapflow-internal pipeline consists of four scripts executed in sequence:

| Step | Script | Action |
|---|---|---|
| 1 | `sap_reorganizer.py` | Parse heterogeneous raw files; standardise column names, timestamp format, and directory structure |
| 2 | `sap_unit_converter.py` | Convert sap velocity units to cm h⁻¹ (sapwood scale) |
| 3 | `plant_to_sapwood_converter.py` | Scale plant-level measurements to sapwood area |
| 4 | `icos_env_extractor.py` / `icos_fluxnet_extractor.py` | Extract co-located ICOS environmental data for these sites only |

**Output:** `Sapflow_SAPFLUXNET_format_unitcon/sapwood/*_sapf_data.csv` and `env_icos/*_env_data.csv` — files that are structurally identical to SAPFLUXNET outputs and can be processed by the same downstream pipeline.

### 5.3 Site-Specific Handling

The following sites are part of the Sapflow-internal collection (i.e., not in the public SAPFLUXNET release) and require the ETL pipeline in `Sapflow-internal/`:

| Site | Country | Special handling |
|---|---|---|
| AT_Mmg | Austria | Not in ICOS; env data sourced separately |
| CH-Dav | Switzerland | Matched to ICOS station CH-Dav |
| CH-Lae | Switzerland | Not in ICOS ecosystem list; env data sourced separately |
| DE-Har | Germany | Measurement interval changed mid-record; time-conditional unit conversion required |
| DE-HoH | Germany | Matched to ICOS station DE-HoH |
| ES-Abr | Spain | Not in ICOS; env data sourced separately |
| ES-Gdn | Spain | Not in ICOS (local site); env data sourced separately |
| ES-LM1 | Spain | Matched to ICOS station ES-LMa (Las Majadas del Tiétar) |
| ES-LM2 | Spain | Matched to ICOS station ES-LMa |
| ES-LMa | Spain | Matched to ICOS station ES-LMa |
| FI-Hyy | Finland | Matched to ICOS station FI-Hyy |
| FR-BIL | France | Matched to ICOS station FR-Bil |
| IT-CP2 | Italy | Timestamp reconstruction required — original files lack proper datetime index; `process_ITCP2_plant.py` and `process_ITCP2_sapwood.py` handle this |
| NO-Hur | Norway | Norwegian site; ICOS status not confirmed |
| PL-Mez | Poland | Polish site; env data sourced separately |
| PL-Tuc | Poland | Polish site; env data sourced separately |
| SE-Nor | Sweden | Matched to ICOS station SE-Nor |
| SE-Sgr | Sweden | Not in ICOS; env data sourced separately |
| ZOE_AT | Austria | Additional Austrian site; env data sourced separately |

### 5.4 Unit Conversion Table

Target unit: **cm³ cm⁻² h⁻¹** (sap flux density, sapwood-normalised). Unit conversion factors are defined in `Sapflow-internal/sap_unit_converter.py`:

| Source unit | Conversion factor | Notes |
|---|---|---|
| cm h⁻¹ | × 1.0 | Already equivalent (cm = cm³ cm⁻²) |
| cm s⁻¹ | × 3600 | s → h |
| cm³ cm⁻² h⁻¹ | × 1.0 | Identity |
| cm³ cm⁻² 30 min⁻¹ | × 2.0 | 30 min → h |
| cm³ cm⁻² 20 min⁻¹ | × 3.0 | 20 min → h |
| cm³ cm⁻² 10 min⁻¹ | × 6.0 | 10 min → h |
| cm³ cm⁻² min⁻¹ | × 60 | min → h |
| cm³ cm⁻² s⁻¹ | × 3600 | s → h |
| mm³ mm⁻² s⁻¹ | × 0.1 × 3600 | mm → cm, s → h |
| g h⁻¹ cm⁻² | × 1.0 | g/h/cm² = cm³/h/cm² (water density 1 g cm⁻³) |
| kg h⁻¹ cm⁻² | × 1000 | kg → g (then as above) |

**Plant-level scaling:** `plant_to_sapwood_converter.py` divides plant-level total flow (cm³ h⁻¹) by sapwood area (cm²) to obtain sapwood-normalised flux density (cm³ cm⁻² h⁻¹). Sapwood area is read from site metadata (`st_sapw_area` or `pl_sapw_area` columns).

### 5.5 ICOS Environmental Data Integration

ICOS data is matched to Sapflow-internal sites by a static site-name mapping defined in `Sapflow-internal/icos_env_extractor.py`. The following ICOS FLUXNET variables are extracted and renamed to match the SAPFLUXNET environmental schema:

| ICOS FLUXNET variable | Mapped name | Units | Description |
|---|---|---|---|
| `TA_F` | `ta` | °C | Air temperature (gap-filled) |
| `VPD_F` | `vpd` | hPa | Vapour-pressure deficit (gap-filled) |
| `SW_IN_F` | `sw_in` | W m⁻² | Shortwave incoming radiation (gap-filled) |
| `PPFD_IN` | `ppfd` | µmol m⁻² s⁻¹ | Photosynthetic photon flux density |
| `PA_F` | `pa` | kPa | Atmospheric pressure |
| `RH` | `rh` | % | Relative humidity |
| `WS_F` | `ws` | m s⁻¹ | Wind speed |
| `P_F` | `precip` | mm | Precipitation |
| `TS_F_MDS_1` | `ts` | °C | Soil temperature (MDS gap-filled, layer 1) |
| `SWC_F_MDS_1` | `swc` | % | Soil water content (MDS gap-filled, layer 1) |

Quality-flag columns (`TA_F_QC`, `VPD_F_QC`, `SW_IN_F_QC`) are also extracted alongside the data columns for downstream filtering.

---

## 6. Data Integration and Merging

### 6.1 Active Merge Script

The operative merge script is **`notebooks/merge_gap_filled_hourly_orginal.py`** (not `merge_gap_filled_hourly.py`, which is a variant with GSI-based growing season logic). The active script includes a daytime-only filter option and joins soil hydraulic properties.

### 6.2 Merge Logic

The merge function `merge_sap_env_data_site()` processes each site individually and concatenates results. For each site:

1. **Sap flow data** read from `paths.sap_outliers_removed_dir`
2. **Environmental data** read from `paths.env_outliers_removed_dir`
3. Both are resampled to 1-hour frequency (mean aggregation for sap flow; mean for environmental)
4. Inner join on site ID and timestamp

Static features are then joined by site ID (left join, one row per site):

| Dataset | Join key | Path variable |
|---|---|---|
| ERA5-Land site | site + timestamp (hourly) | `paths.era5_discrete_data_path` |
| GlobMap LAI | site + 8-day period → interpolated to hourly | `paths.globmap_lai_data_path` |
| MODIS PFT | site (annual) | `paths.pft_data_path` |
| WorldClim + terrain | site | `paths.env_extracted_data_path` |
| SoilGrids | site | `soilgrids_data.csv` |
| Stand age | site | `stand_age_data.csv` |

### 6.3 Biome Label Assignment

Each site is assigned a biome label based on its Köppen-Geiger zone and WorldClim climatology. The nine biome types used are:

```
'Boreal forest', 'Subtropical desert', 'Temperate forest',
'Temperate grassland desert', 'Temperate rain forest',
'Tropical forest savanna', 'Tropical rain forest',
'Tundra', 'Woodland/Shrubland'
```

A site-to-biome mapping is saved to `outputs/processed_data/sapwood/merged/site_biome_mapping.csv`.

### 6.4 Output Dataset Schema

The merged dataset is written to:
- `outputs/processed_data/sapwood/merged/merged_data.csv` — all sites, hourly
- `outputs/processed_data/sapwood/merged/hourly/` — per-site CSVs at hourly resolution
- `outputs/processed_data/sapwood/merged/daily/` — per-site CSVs aggregated to daily

The merged CSV files contain the following column groups. For daily files, dynamic variables are expanded to four aggregation columns: `{var}` (mean), `{var}_min`, `{var}_max`, `{var}_sum`. Static variables appear once.

**Identifiers and timestamps**

| Column | Units | Source |
|---|---|---|
| `TIMESTAMP` | UTC datetime | Merge |
| `solar_TIMESTAMP` | Local solar time | Merge |
| `site_name` | — | SAPFLUXNET metadata |
| `biome` | — | Köppen-Geiger biome label |
| `pft` | — | MODIS MCD12Q1 PFT category string |

**Sap flow**

| Column | Units | Source |
|---|---|---|
| `sap_velocity` | cm³ cm⁻² h⁻¹ | SAPFLUXNET (site average of plant sensors) |

**Meteorological (dynamic)**

| Column | Units | Source |
|---|---|---|
| `ta` | °C | ERA5-Land / SAPFLUXNET env |
| `vpd` | kPa | ERA5-Land / SAPFLUXNET env |
| `sw_in` | W m⁻² | ERA5-Land / SAPFLUXNET env |
| `ppfd_in` | µmol m⁻² s⁻¹ | ERA5-Land (derived) |
| `ext_rad` | W m⁻² | Astronomical (pvlib) |
| `ws` | m s⁻¹ | ERA5-Land |
| `rh` | % | ERA5-Land / SAPFLUXNET env |
| `netrad` | W m⁻² | SAPFLUXNET env (site-level, where available) |
| `precip` | mm | ERA5-Land |
| `day_length` | h | Astronomical (pvlib) |

**ERA5-Land reanalysis (dynamic)**

| Column | Units | Source |
|---|---|---|
| `volumetric_soil_water_layer_1` | m³ m⁻³ (z-score normalised per site) | ERA5-Land |
| `soil_temperature_level_1` | K | ERA5-Land |
| `total_precipitation_hourly` | m | ERA5-Land |
| `potential_evaporation_hourly` | m | ERA5-Land |
| `prcip/PET` | dimensionless (clipped 0–10) | ERA5-Land derived |

**Remote sensing (dynamic)**

| Column | Units | Source |
|---|---|---|
| `LAI` | m² m⁻² | GlobMap LAI (8-day, nearest merge) |

**Site metadata (static)**

| Column | Units | Source |
|---|---|---|
| `latitude` | ° | SAPFLUXNET site metadata |
| `longitude` | ° | SAPFLUXNET site metadata |
| `elevation` | m | ASTER GDEM (GEE) |
| `slope` | ° | ASTER GDEM derived |
| `aspect_sin`, `aspect_cos` | — | ASTER GDEM derived (cyclical encoding) |
| `canopy_height` | m | Meta/Facebook canopy height (GEE) |
| `mean_annual_temp` | °C | WorldClim BIO1 |
| `mean_annual_precip` | mm yr⁻¹ | WorldClim BIO12 |
| `temp_seasonality` | — | WorldClim BIO4 |
| `precip_seasonality` | — | WorldClim BIO15 |
| `stand_age` | years | BGI Forest Age dataset |

**Soil properties (static, depth-weighted 0–100 cm)**

| Column | Units | Source |
|---|---|---|
| `soil_sand` | g kg⁻¹ | SoilGrids 2.0 |
| `soil_clay` | g kg⁻¹ | SoilGrids 2.0 |
| `soil_soc` | dg kg⁻¹ | SoilGrids 2.0 |
| `soil_bdod` | cg cm⁻³ | SoilGrids 2.0 |
| `soil_cfvo` | cm³ dm⁻³ | SoilGrids 2.0 |
| `soil_theta_wp` | m³ m⁻³ | Saxton & Rawls (2006) derived |
| `soil_theta_fc` | m³ m⁻³ | Saxton & Rawls (2006) derived |
| `soil_theta_sat` | m³ m⁻³ | Saxton & Rawls (2006) derived |
| `soil_{prop}_{depth}` | (varies) | Raw per-depth SoilGrids columns (sand, clay, soc, bdod, cfvo × 5 depth bands: 0_5, 5_15, 15_30, 30_60, 60_100) |

---

## 7. Feature Engineering

### 7.1 Dynamic (Time-Varying) Features

These features vary at hourly or daily timescales and are derived from the ERA5-Land site extraction:

| Model feature name | Source variable | Derivation |
|---|---|---|
| `ta` | `temperature_2m` | Direct (K → °C) |
| `vpd` | `temperature_2m` + `dewpoint_temperature_2m` | Magnus formula: VPD = e_sat(T) − e_act(T_dew) |
| `sw_in` | `surface_solar_radiation_downwards` | Direct (J m⁻² → W m⁻²) |
| `ppfd_in` | `sw_in` | Photosynthetically active radiation: PPFD ≈ 0.46 × SW_in × 4.57 (µmol m⁻² s⁻¹) |
| `ext_rad` | Astronomical calculation | Extraterrestrial radiation from solar geometry |
| `ws` | `10m_u_component_of_wind` + `10m_v_component_of_wind` | Wind speed: ws = √(u² + v²) |
| `volumetric_soil_water_layer_1` | `volumetric_soil_water_layer_1` | Direct |
| `soil_temperature_level_1` | `soil_temperature_level_1` | Direct |
| `LAI` | GlobMap LAI | 8-day composites linearly interpolated to hourly |
| `prcip/PET` | `total_precipitation` / `potential_evaporation` | Ratio; aridity index |

### 7.2 Static (Site-Level) Features

These features are constant through time for a given site:

| Model feature name | Source | Notes |
|---|---|---|
| `canopy_height` | Meta/Facebook canopy height (GEE) | m |
| `elevation` | ASTER GDEM (GEE) | m |
| `mean_annual_temp` | WorldClim BIO1 | °C |
| `mean_annual_precip` | WorldClim BIO12 | mm yr⁻¹ |
| `prcip/PET` | Computed from ERA5 monthly means | Long-term aridity |
| PFT one-hot: `MF`, `DNF`, `ENF`, `EBF`, `WSA`, `WET`, `DBF`, `SAV` | MODIS MCD12Q1 LC_Type1 | 8-class one-hot encoding |

### 7.3 Cyclical Time Encoding

When `TIME_FEATURES = True`, the following cyclical features are generated:

| Feature | Formula | Rationale |
|---|---|---|
| `sin_doy`, `cos_doy` | sin/cos(2π × day_of_year / 365) | Seasonal cycle |
| `sin_hod`, `cos_hod` | sin/cos(2π × hour_of_day / 24) | Diurnal cycle |

Cyclical encoding preserves the continuity across year-end (day 365 → day 1) and midnight (hour 23 → hour 0) that would be broken by treating day-of-year or hour-of-day as linear integers.

### 7.4 Target Variable Transformation

**Transformation used: `log1p`** (default, `IS_TRANSFORM = True`, `TRANSFORM_METHOD = 'log1p'`).

```
y_train_transformed = log(1 + max(y, 0))
```

**Rationale:** Sap velocity is strongly right-skewed, with a heavy tail driven by high-flow events and a mass near zero from nighttime and dormant-season observations. The `log1p` transform compresses the upper tail, reduces heteroscedasticity, and improves model fitting for gradient boosting. The `max(y, 0)` clip handles the small fraction of negative values that result from sensor noise.

**Inverse transform at prediction time:**
```
y_pred_original = exp(y_pred_transformed) − 1   # i.e., expm1(y_pred_transformed)
```

This is applied via `target_transformer.inverse_transform()` (class `TargetTransformer` in the training script). SHAP base values and SHAP contributions are therefore on the log-scale; SHAP waterfall plots that display original-scale values call `inverse_transform()` explicitly.

*Alternative transforms supported:* `sqrt`, `box-cox`, `yeo-johnson`, `none` — selectable via `--TRANSFORM_METHOD` flag.

### 7.5 Temporal Windowing

When `IS_WINDOWING = True`, a sliding window of `INPUT_WIDTH` past time steps is used as model input. Default parameters:

| Parameter | Default | Description |
|---|---|---|
| `INPUT_WIDTH` | 2 | Number of past time steps included |
| `LABEL_WIDTH` | 1 | Number of target steps to predict |
| `SHIFT` | 1 | Steps between last input and first target |

Windowed feature names follow the pattern `{feature_name}_t-{lag}` where `t-0` is the current step and `t-1` is one step back. Static features (canopy height, PFT, etc.) are not windowed; they appear once in the feature vector without a time suffix.

*Implementation: `src/hyperparameter_optimization/timeseries_processor1.py` (`TimeSeriesSegmenter`, `WindowGenerator`).*

### 7.6 Complete Feature List

The following features are used by the final XGBoost model (run `default_daily_nocoors_swcnor`, daily timescale, no coordinates, soil-water normalised). Feature order is determined by the first site processed:

**Dynamic features (mean across daytime hours, daily aggregation):**

| Feature | Unit | Aggregates available |
|---|---|---|
| `sw_in` | W m⁻² | mean, `sw_in_min`, `sw_in_max`, `sw_in_sum` |
| `ws` | m s⁻¹ | mean, `ws_min`, `ws_max`, `ws_sum` |
| `ta` | °C | mean, `ta_min`, `ta_max`, `ta_sum` |
| `ta_max` | °C | daily maximum |
| `ta_min` | °C | daily minimum |
| `vpd` | kPa | mean, `vpd_min`, `vpd_max`, `vpd_sum` |
| `vpd_max` | kPa | daily maximum |
| `vpd_min` | kPa | daily minimum |
| `ext_rad` | W m⁻² | mean |
| `ppfd_in` | µmol m⁻² s⁻¹ | mean |
| `precip` | mm | sum |
| `volumetric_soil_water_layer_1` | m³ m⁻³ (z-normalised) | mean |
| `soil_temperature_level_1` | K | mean |
| `LAI` | m² m⁻² | mean (8-day nearest) |
| `prcip/PET` | dimensionless | daily ratio |
| `day_length` | h | value |
| `Year sin` | — | sin(2π × doy / 365) |

**Static features:**

| Feature | Unit |
|---|---|
| `canopy_height` | m |
| `elevation` | m |
| `MF` | binary (one-hot PFT) |
| `DNF` | binary |
| `ENF` | binary |
| `EBF` | binary |
| `WSA` | binary |
| `WET` | binary |
| `DBF` | binary |
| `SAV` | binary |

*Note: latitude and longitude are excluded from this run (`default_daily_nocoors`). The exact feature order is stored in `outputs/plots/hyperparameter_optimization/xgb/{run_id}/feature_units_{run_id}.json`.*

---

## 8. Model Architecture and Training

### 8.1 Models Evaluated

| Model | Script | CV strategy | Input format |
|---|---|---|---|
| XGBoost | `test_hyperparameter_tuning_ML_spatial_stratified.py` | Spatial stratified | 2D flattened |
| Random Forest | `test_hyperparameter_tuning_rf_spatial.py` | Spatial | 2D flattened |
| SVM | `test_hyperparameter_tuning_svm.py` | Spatial | 2D flattened |
| ANN | `test_hyperparameter_tuning_ann_spatial.py` | Spatial GroupKFold | 2D flattened |
| ANN (temporal) | `test_hyperparameter_tuning_ann_temporal_seg.py` | Leave-segment-out | 2D flattened |
| ANN (block CV) | `test_hyperparameter_tuning_ann_blockcv.py` | Spatial block CV | 2D flattened |
| CNN-LSTM | `test_hyperparameter_tuning_DL_spatial_stratified.py` | Spatial stratified | 3D windowed |
| LSTM | `test_hyperparameter_tuning_DL_spatial_stratified.py` | Spatial stratified | 3D windowed |
| Transformer | `test_hyperparameter_tuning_DL_spatial_stratified.py` | Spatial stratified | 3D windowed |
| AutoML | `test_hyperparameter_tuning_autoML_spatial_stratified.py` | Spatial stratified | 2D |

### 8.2 XGBoost (Primary Model)

XGBoost gradient boosting (`xgboost.XGBRegressor`) was selected as the primary model for global prediction based on CV performance comparison across XGBoost, Random Forest, SVM, ANN, and CNN-LSTM (see Section 11.1).

**Hyperparameter search space** (grid search; from `outputs/logs/xgb_optimizer.log`):

```json
{
  "n_estimators": [500, 1000],
  "learning_rate": [0.05, 0.1, 0.2],
  "max_depth": [3, 5],
  "min_child_weight": [1, 3],
  "subsample": [0.7, 0.8, 0.9],
  "colsample_bytree": [0.7, 0.8],
  "gamma": [0, 0.1, 0.2],
  "reg_alpha": [0.005, 0.01],
  "reg_lambda": [1.5, 2]
}
```

**Best parameters found** (repeated across all 10 CV folds):

```json
{
  "n_estimators": 1000,
  "learning_rate": 0.05,
  "max_depth": 10,
  "min_child_weight": 5,
  "subsample": 0.67,
  "colsample_bytree": 0.8,
  "gamma": 0.2
}
```

**Default arguments from training script:**
- Random seed: `RANDOM_SEED = 42`
- Cross-validation folds: `n_groups = 10`
- Scoring: `neg_mean_squared_error`
- Stratified sampling: `IS_STRATIFIED = True`
- SHAP sample size: `SHAP_SAMPLE_SIZE = 50000`

**Model output files:**
- Per-fold models: `outputs/models/xgb/{run_id}/spatial_xgb_fold_{i}_{run_id}.json`
- Final model: `outputs/models/xgb/{run_id}/FINAL_xgb_{run_id}.joblib`
- Feature scaler: `outputs/models/xgb/{run_id}/FINAL_scaler_{run_id}_feature.pkl`
- Config: `outputs/models/xgb/{run_id}/model_config_{run_id}.json`

### 8.3 ANN Architecture

ANN architecture (from `outputs/logs/ANN_optimizer.log`). The network takes a flattened 2D feature vector as input.

**Hyperparameter search grid:**

| Parameter | Values searched |
|---|---|
| `n_layers` | 2, 3 |
| `units` | 32, 64 |
| `dropout_rate` | 0.1 |
| `optimizer` | Adam |
| `learning_rate` | 0.001 |
| `batch_size` | 32 |
| `epochs` | 100 |
| `patience` (early stopping) | 10 |

**Best parameters:** `n_layers = 2`, `units = 64`, `dropout_rate = 0.1`, Adam (lr = 0.001), batch size = 32, early stopping patience = 10. Activation function is ReLU (Keras default for dense layers). No batch normalisation layer in the searched grid.

### 8.4 Deep Learning Architectures (CNN-LSTM, LSTM, Transformer)

All deep learning models use a 3D windowed input format `(batch, INPUT_WIDTH, n_features)` with `INPUT_WIDTH = 2` (current + 1 lagged time step). Trained via `test_hyperparameter_tuning_DL_spatial_stratified.py`.

**CNN-LSTM** (from `outputs/logs/CNN-LSTM_optimizer.log`):

| Parameter | Value |
|---|---|
| `cnn_layers` | 2 |
| `lstm_layers` | 2 |
| `cnn_filters` | 12 |
| `lstm_units` | 8 |
| `dropout_rate` | 0.3 |
| Optimizer | Adam (default learning rate) |
| `batch_size` | 32 |
| `epochs` | 100 |
| `patience` (early stopping) | 20 |

*Architecture:* Two Conv1D layers (12 filters each) followed by two LSTM layers (8 units each), with Dropout(0.3) after each layer pair, and a Dense(1) output head.

**LSTM and Transformer:** Architecture details are defined in `src/hyperparameter_optimization/hyper_tuner.py`. Refer to the `ANN_optimizer.log` and the model config JSON in `outputs/models/{MODEL_TYPE}/{run_id}/model_config_{run_id}.json` for the exact architecture used in the final run.

### 8.5 Reproducibility Infrastructure

All stochastic elements are controlled with a global seed `RANDOM_SEED = 42` applied to:
- Python `random.seed()`
- `numpy.random.seed()`
- `torch.manual_seed()` / `tensorflow.random.set_seed()` (where applicable)
- XGBoost/sklearn `random_state` parameter

The following additional determinism flags are set in `src/hyperparameter_optimization/hyper_tuner.py` (commit `23a9b311 finish determinism control`):

```python
os.environ['TF_DETERMINISTIC_OPS'] = '1'      # Force deterministic TF ops
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'    # Deterministic cuDNN algorithms
os.environ['PYTHONHASHSEED'] = '42'           # Disable hash randomisation

tf.config.experimental.enable_op_determinism()   # TF 2.6+
tf.config.threading.set_inter_op_parallelism_threads(n_physical_cores)
tf.config.threading.set_intra_op_parallelism_threads(n_physical_cores)
```

In `test_hyperparameter_tuning_ML_spatial_stratified.py`:
```python
tf.random.set_seed(42)
np.random.seed(42)
set_seed(42)   # from src/utils/random_control.py — wraps random, numpy, tf, sklearn seeds
```

*Note:* This project does not use PyTorch; the `torch.backends.cudnn.deterministic` flag is not applicable. Determinism was verified by running the same configuration twice and confirming identical CV fold scores.

---

## 9. Cross-Validation Strategies

Three CV strategies are implemented, each addressing a different source of statistical dependence in the training data.

### 9.1 Spatial Stratified GroupKFold (Primary)

**Addresses:** Spatial autocorrelation — observations from the same or nearby sites are not independent.

**Construction:** Sites are assigned to `n_groups = 10` spatial groups using `create_spatial_groups()` in the training script (`method = 'grid'` or `'default'`, controlled via `--spatial_split_method`). The **`'grid'`** method bins sites into 0.05° × 0.05° geographic grid cells, then uses a greedy balancing algorithm to merge grid cells into `n_groups` folds such that data counts are approximately equal across folds. The **`'default'`** method assigns one group per site (unbalanced). `sklearn.model_selection.StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)` then uses the biome label as the stratification variable, ensuring each fold contains examples from all biome types.

**Train/test split:** 9-fold training, 1-fold test; outer loop only.

**Window inheritance:** When windowing is enabled, all windows derived from a given site inherit that site's group label, preventing data leakage across the spatial boundary.

**Script:** `test_hyperparameter_tuning_ML_spatial_stratified.py`, `test_hyperparameter_tuning_DL_spatial_stratified.py`

### 9.2 Temporal Leave-Segment-Out

**Addresses:** Temporal autocorrelation — consecutive observations within a site are serially correlated.

**Construction:** Long continuous time series within each site are segmented by `TimeSeriesSegmenter` in `timeseries_processor1.py`. Segments shorter than `INPUT_WIDTH + SHIFT` steps are excluded. Each segment is a "group" for leave-one-out CV.

**Rationale:** A temporally left-out segment tests whether the model generalises to periods it has not seen — relevant for predicting future climate scenarios.

**Script:** `test_hyperparameter_tuning_ann_temporal_seg.py`

### 9.3 Spatial Block Cross-Validation

**Addresses:** Spatial autocorrelation with buffer zones to prevent test-set contamination from nearby training sites.

**Construction:** `BlockCV` (`src/hyperparameter_optimization/blockcv.py`) builds spatial blocks sized according to the empirical autocorrelation range from a variogram analysis of sap velocity residuals.

*Note: `blockcv.py` was not found at `src/hyperparameter_optimization/blockcv.py` at time of documentation. Autocorrelation range, block size, and buffer zone parameters should be documented here once the block CV run has been executed and the variogram analysis completed. Reference: Roberts et al. (2017) for the methodology.*

**Script:** `test_hyperparameter_tuning_ann_blockcv.py`

> **Note:** `blockcv.py` was temporarily deleted from version control. Ensure it is present at `src/hyperparameter_optimization/blockcv.py` before running block CV training. Restore if missing: `git checkout HEAD -- src/hyperparameter_optimization/blockcv.py`.

---

## 10. SHAP Interpretability Analysis

### 10.1 Explainer and Sampling

SHAP values are computed using `shap.TreeExplainer(final_model)`, which provides exact Shapley values for tree-ensemble models (XGBoost, Random Forest) without approximation error. This is preferred over `KernelExplainer` because it is both exact and computationally tractable.

Since computing SHAP values for the entire dataset is expensive, a random sample of up to **50,000 observations** (configurable via `--SHAP_SAMPLE_SIZE`) is drawn with `numpy.random.seed(42)` before explanation. If the dataset is smaller than 50,000, all observations are used.

### 10.2 Static Feature Aggregation (Windowed Mode)

When `IS_WINDOWING = True`, the feature vector contains `INPUT_WIDTH` copies of each dynamic feature (one per time step, e.g., `vpd_t-0`, `vpd_t-1`). The function `aggregate_static_feature_shap()` sums SHAP values for repeated static features (which do not change across time steps) and averages or sums SHAP values for the same dynamic feature across time steps (`aggregate_shap_values_by_timestep()`), producing a reduced feature set that is interpretable.

### 10.3 PFT Aggregation

PFT is one-hot encoded into 8 binary columns (`MF`, `DNF`, `ENF`, `EBF`, `WSA`, `WET`, `DBF`, `SAV`). For summary plots, the SHAP values of all PFT columns are summed into a single `PFT` column via `group_pft_for_summary_plots()` and `aggregate_pft_shap_values()`. This allows the collective contribution of land-cover type to be compared against continuous environmental drivers.

### 10.4 Output Files

All SHAP outputs are written to `outputs/plots/hyperparameter_optimization/{MODEL_TYPE}/{run_id}/`:

**Numeric outputs:**
- `shap_values_{run_id}.npz` — NumPy archive containing `shap_values`, `shap_values_pft_aggregated`, `shap_values_raw`
- `shap_feature_importance.csv` — mean |SHAP| per feature, with feature category (static/dynamic) and physical unit
- `shap_statistics_by_pft.csv` — per-PFT SHAP statistics (mean, std, min, max) for each feature
- `shap_importance_pivot_by_pft.csv` — pivoted importance matrix (features × PFT)

**Global importance plots:**
- `shap_summary_beeswarm.png` — beeswarm plot (raw PFT columns)
- `shap_summary_beeswarm_grouped.png` — beeswarm plot (PFT grouped into single column)
- `shap_global_importance_bar.png` — mean |SHAP| bar chart (raw)
- `shap_global_importance_bar_grouped.png` — mean |SHAP| bar chart (PFT grouped)
- `shap_partial_dependence.png` — partial dependence plots for top 9 features
- `shap_partial_dependence_grouped.png` — same with PFT grouped

**Spatial and local plots:**
- `shap_spatial_maps.png` — geographic scatter of SHAP values for top 4 features
- `shap_waterfall_High_Flow.png`, `shap_waterfall_Low_Flow.png` — local explanation for highest and lowest predicted observations

**Temporal plots:**
- `shap_static_vs_dynamic.png` — pie chart of total SHAP importance by feature category + bar chart
- `shap_time_step_comparison.png` — importance by lag (windowed models only)
- `fig7_seasonal_drivers_by_hemisphere.png` — monthly mean SHAP values by hemisphere for top 5 features
- `fig_diurnal_drivers.png` — stacked-bar diurnal profile of feature contributions
- `fig_diurnal_drivers_heatmap.png` — heatmap of SHAP by hour × feature
- `fig_diurnal_drivers_lines.png` — line plots with CI for top 12 features × hour of day
- `fig8_dependence_interaction.png` — SHAP interaction plots for selected feature pairs

**PFT-stratified plots:**
- `shap_importance_heatmap_by_pft.png` — top-12 feature importance heatmap across PFTs
- `shap_top_features_per_pft.png` — top 6 features for each PFT
- `shap_by_pft_boxplot.png`, `shap_by_pft_violin.png` — SHAP distributions by PFT for top 8 features
- `shap_pft_contribution_comparison.png`, `shap_pft_radar_chart.png` — cross-PFT comparison
- `shap_summary_{pft}.png` — individual beeswarm plot for each PFT (e.g., `shap_summary_ENF.png`)

### 10.5 Key Findings

[FILL IN: Summarise the main SHAP results. For example:
- Which feature had the highest global mean |SHAP|? (e.g., VPD? sw_in?)
- How does the importance ranking change across biomes?
- What does the diurnal analysis reveal about the time-of-day patterns?
- Is stand age or canopy height important for distinguishing between PFTs?
- Do windowed models show that lagged VPD contributes meaningfully beyond t-0?]

---

## 11. Results and Evaluation

### 11.1 Model Comparison

All models were evaluated under 10-fold spatial stratified cross-validation. Performance metrics are computed on held-out spatial folds.

| Model | CV R² | CV RMSE (cm h⁻¹) | CV MAE (cm h⁻¹) | Notes |
|---|---|---|---|---|
| XGBoost | [FILL IN] | [FILL IN] | [FILL IN] | Primary model |
| Random Forest | [FILL IN] | [FILL IN] | [FILL IN] | |
| SVM | [FILL IN] | [FILL IN] | [FILL IN] | |
| ANN (spatial) | [FILL IN] | [FILL IN] | [FILL IN] | |
| ANN (temporal) | [FILL IN] | [FILL IN] | [FILL IN] | |
| CNN-LSTM | [FILL IN] | [FILL IN] | [FILL IN] | |

**Selected model:** [FILL IN: e.g., XGBoost, selected based on highest mean CV R² and lowest variance across folds.]

### 11.2 Final Model Performance

The final model is trained on the full dataset after hyperparameter selection and evaluated on a spatially independent held-out test set.

| Metric | All sites | Temperate forest | Tropical RF | Boreal | Tundra |
|---|---|---|---|---|---|
| R² | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| RMSE (cm h⁻¹) | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| MAE (cm h⁻¹) | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |

[FILL IN: Discuss where and why the model under/over-performs. For example: "The model performs poorest for tropical rain forest sites (R² = X), consistent with these biomes being underrepresented in training (only N sites, Table 3.3). Conversely, temperate forest performance is strong (R² = X), reflecting the abundance of training data from European and North American sites."]

### 11.3 Key Figures

[FILL IN: Reference and interpret each result figure. For each figure, provide:
- File path
- What the x and y axes represent
- What the main pattern is
- What this implies for model behaviour]

---

## 12. Global Prediction Pipeline

### 12.1 Overview

The global prediction pipeline applies the trained model to produce gridded sap velocity estimates across all forested land between 60°S and 78°N. It is implemented in two scripts:

1. `src/make_prediction/process_era5land_gee_opt_fix.py` — ERA5-Land retrieval and feature derivation per grid cell
2. `src/make_prediction/predict_sap_velocity_sequantial.py` — model application and output serialization

### 12.2 ERA5-Land Retrieval (GEE-based)

`ERA5LandGEEProcessor` fetches ERA5-Land ImageCollection fields from Google Earth Engine for each grid cell in the prediction domain. The processing steps mirror the training-time feature derivation to ensure consistency:

1. Extract the same 9 ERA5-Land variables listed in Section 2.3
2. Derive VPD from temperature and dewpoint (same Magnus formula as training)
3. Compute PPFD from shortwave radiation and PET from potential evaporation
4. Compute wind speed from U and V components
5. Fetch LAI, PFT, and canopy height from their respective GEE collections (same as extraction scripts)
6. Add WorldClim bioclimatic variables and biome classification

Intermediate results are cached at `D:/Temp/era5land_extracted/` to allow resuming interrupted runs without repeating GEE API calls.

### 12.3 Feature Consistency with Training

To prevent train–predict feature mismatch (a common source of subtle bugs):
- The same feature scaler (`FINAL_scaler_{run_id}_feature.pkl`) is applied at prediction time
- The feature order in the prediction dataframe is enforced to match the model config JSON (`final_feature_names` key)
- Column name assertions: the prediction script checks that all required features are present in the ERA5-Land grid data before calling `model.predict()`; missing columns raise a `ValueError` with a list of missing features
- Predicted values are clipped to `[0, max_training_value]` to prevent physically implausible extrapolations beyond the training range
- Negative predictions (artefacts from the inverse log-transform on very negative model outputs) are clipped to 0

### 12.4 Spatial Domain and Chunking

| Parameter | Value |
|---|---|
| Latitude range | −60° to +78° |
| Longitude range | −180° to +180° |
| Maximum grid cells per batch | 1,000 |
| Chunk size (ERA5 time) | 32 time steps |
| Chunk size (spatial) | 1 × 1 cell |

The 1,000-cell batch limit (`MAX_GRID_CELLS`) prevents memory overflow when accumulating prediction results.

### 12.5 Output Files

| File | Description |
|---|---|
| `data/predictions/combined_predictions_improved.csv` | Full combined predictions across all dates |
| `data/predictions/prediction_YYYY_MM_DD_predictions_improved.csv` | Per-date prediction files |
| `outputs/maps/global_sap_velocity_*.tif` | GeoTIFF raster outputs |
| `outputs/maps/global_sap_velocity_*.png` | Global map visualizations |

---

## 13. Post-Prediction Analysis

### 13.1 Climate Zone × Forest Type Stratification

Script: `src/make_prediction/sap_velocity_by_climatezone_forest.py`

Predicted sap velocities are stratified by:
- Köppen-Geiger climate zone (resampled from the 1-km classification to match the prediction grid resolution using nearest-neighbour resampling to preserve categorical values)
- MODIS PFT (resampled to match prediction grid)

**Resampling method for categorical rasters:** Nearest-neighbour resampling (`rasterio.Resampling.nearest`) is used when reprojecting Köppen-Geiger climate zones and MODIS PFT from their native resolution (1 km and 500 m respectively) to the 0.1° prediction grid. Nearest-neighbour preserves the integer class codes, avoiding the introduction of non-existent intermediate class values that would result from bilinear or cubic interpolation. The dominant class within each 0.1° cell is then assigned via majority-vote aggregation.

### 13.2 Physical Plausibility Checks

The following physical plausibility checks are applied to the global prediction output:

1. **Range check:** Predicted values are clipped to `[0, max_training_value]`. Any predictions exceeding the maximum observed sap velocity in training (the script logs a warning for sites with max > 100 cm h⁻¹) are flagged.
2. **Spatial pattern sanity:** Expected gradients are verified visually — arid regions (Sahara, Arabian Peninsula, Australian interior) should show near-zero predictions corresponding to sparse or absent forest cover; tropical humid zones (Amazon, Congo, Southeast Asia) should show elevated values.
3. **Zero-forest masking:** Cells with Tree Existence probability < 15% (from the Meta canopy height model) are masked to zero in the final prediction maps.
4. **Seasonal coherence:** Northern hemisphere predictions should peak in summer (JJA) and approach zero in winter (DJF). Southern hemisphere predictions should show the inverse pattern.

*Additional quantitative validation (e.g., Pearson correlation of predicted spatial means vs. VPD/LAI gradients) is deferred to the results analysis phase (Section 11).*

---

## 14. Project Structure and Setup

### 14.1 Directory Structure

```
global-sap-velocity/
├── data/
│   ├── raw/
│   │   ├── 0.1.5/…/csv/sapwood/          # SAPFLUXNET database
│   │   ├── extracted_data/                # Per-site extracted features
│   │   │   ├── era5land_site_data/sapwood/
│   │   │   ├── globmap_lai_site_data/sapwood/
│   │   │   ├── landcover_data/sapwood/
│   │   │   ├── terrain_site_data/sapwood/
│   │   │   └── env_site_data/sapwood/
│   │   └── grided/
│   │       ├── globmap_lai/               # Pre-download required
│   │       ├── stand_age/                 # Pre-download required
│   │       ├── worldclim/                 # Auto-downloaded
│   │       └── koppen_geiger/             # Auto-downloaded
│   └── predictions/                       # Global prediction outputs
├── outputs/
│   ├── processed_data/sapwood/
│   │   ├── sap/outliers_removed/          # Post-QC sap flow (per site)
│   │   ├── env/outliers_removed/          # Post-QC env data (per site)
│   │   └── merged/                        # Merged training dataset
│   ├── models/                            # Trained model files
│   ├── scalers/                           # Feature and label scalers
│   └── plots/hyperparameter_optimization/ # Training diagnostics + SHAP
├── src/
│   ├── Analyzers/                         # QC scripts
│   ├── Extractors/                        # Feature extraction scripts
│   ├── hyperparameter_optimization/       # Training and evaluation scripts
│   └── make_prediction/                   # Global prediction scripts
├── notebooks/                             # Merge, exploration, analysis scripts
├── Sapflow-internal/                      # Sapflow-internal ETL pipeline
├── Sapflow_SAPFLUXNET_format_unitcon/     # Sapflow-internal formatted output
├── docs/                                  # Documentation
└── path_config.py                         # Centralised path configuration
```

### 14.2 Path Configuration

All file paths are managed centrally through `path_config.py` (`PathConfig` class). The default output root is `./outputs/`. To change the output location, set the environment variable `OUTPUT_DIR` or pass `base_output_dir` to `PathConfig()`.

The `.venv/path_config.py` is a copy/symlink used by certain scripts. Always edit the root-level `path_config.py` as the canonical source.

### 14.3 Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt  # requirements.txt is present in the project root

# GEE authentication
earthengine authenticate
# Set project
earthengine set_project ee-yuluo-2
```

**Key package versions:**

| Package | Version | Notes |
|---|---|---|
| Python | 3.10.11 | CPython, Windows |
| xgboost | 2.1.3 | Primary model |
| scikit-learn | 1.5.2 | CV, preprocessing |
| shap | 0.49.1 | SHAP interpretability |
| numpy | 1.26.4 | |
| pandas | 2.2.3 | |
| tensorflow | 2.18.0 | For ANN / CNN-LSTM / Transformer |
| keras | 3.6.0 | High-level TF API |
| keras-tuner | 1.4.7 | Hyperparameter tuning (Keras models) |
| earthengine-api | 1.5.1 | GEE data extraction |
| pvlib | 0.13.1 | Solar position, day length |
| rasterio | 1.4.3 | GeoTIFF I/O |
| geopandas | 1.0.1 | Spatial operations |
| scipy | 1.13.1 | Box-Cox, statistics |
| matplotlib | 3.9.2 | Visualisation |
| seaborn | 0.13.2 | Statistical plots |
| optuna | 4.1.0 | AutoML hyperparameter search |
| joblib | 1.4.2 | Model serialisation |
| GDAL | 3.6.2 | Raster I/O (system library) |

### 14.4 Pre-Download Requirements

Before running the extraction pipeline, the following datasets must be manually downloaded:

| Dataset | Local path | Source |
|---|---|---|
| GlobMap LAI tiles | `data/raw/grided/globmap_lai/GlobMapLAIV3.A*.Global.LAI.tif` | http://globalchange.bnu.edu.cn/research/laiv3 (GlobMap LAI v3) |
| BGI Stand Age NetCDF | `data/raw/grided/stand_age/2026113115224222_BGIForestAgeMPIBGC1.0.0.nc` | Contact BGI/MPI-BGC data portal; see Poulter et al. (2019) |

WorldClim and Köppen-Geiger are downloaded automatically by `spatial_features_extractor.py` on first run.

### 14.5 Running the Pipeline

```bash
# Step 1: Reorganise Sapflow-internal (if applicable)
python Sapflow-internal/sap_reorganizer.py
python Sapflow-internal/sap_unit_converter.py
python Sapflow-internal/plant_to_sapwood_converter.py
python Sapflow-internal/icos_env_extractor.py

# Step 2: Extract site-level features
python src/Extractors/extract_siteinfo.py
python src/Extractors/extract_climatedata_gee.py
python src/Extractors/extract_pft.py
python src/Extractors/extract_canopyheight.py    # Must run before spatial_features_extractor
python src/Extractors/spatial_features_extractor.py
python src/Extractors/extract_globmap_lai.py
python src/Extractors/extract_soilgrid.py
python src/Extractors/extract_stand_age.py

# Step 3: Quality control
python -c "
from src.Analyzers.sap_analyzer import SapFlowAnalyzer
analyzer = SapFlowAnalyzer(scale='sapwood')
# Optional: apply manual removal log
# analyzer.set_removal_log('path/to/removal_log.csv')
analyzer.run_analysis_in_batches(batch_size=10, switch='both')
"
python -c "
from src.Analyzers.env_analyzer import EnvironmentalAnalyzer
ea = EnvironmentalAnalyzer()
ea.run_analysis_in_batches(batch_size=10)
"

# Step 4: Merge
python notebooks/merge_gap_filled_hourly_orginal.py

# Step 5: Train model
python src/hyperparameter_optimization/test_hyperparameter_tuning_ML_spatial_stratified.py \
  --model xgb \
  --RANDOM_SEED 42 \
  --n_groups 10 \
  --SPLIT_TYPE spatial_stratified \
  --TIME_SCALE daily \
  --IS_TRANSFORM True \
  --TRANSFORM_METHOD log1p \
  --IS_STRATIFIED True \
  --IS_CV True \
  --IS_WINDOWING False \
  --SHAP_SAMPLE_SIZE 50000 \
  --run_id default_daily_nocoors_swcnor \
  --spatial_split_method default

# Step 6: Global prediction
python src/make_prediction/process_era5land_gee_opt_fix.py
python src/make_prediction/predict_sap_velocity_sequantial.py

# Step 7: Visualise
python src/make_prediction/prediction_visualization.py
python src/make_prediction/sap_velocity_by_climatezone_forest.py
```

---

## 15. Limitations and Future Work

### 15.1 Data Limitations

1. **Geographic bias:** Training sites are heavily concentrated in temperate Europe and North America (Section 3.3). Predictions in underrepresented biomes (tropical rain forest, tundra, boreal Asia) carry higher uncertainty and may extrapolate beyond the training distribution.

2. **Measurement method heterogeneity:** SAPFLUXNET aggregates sites using different sap flow measurement technologies (thermal dissipation, heat-pulse velocity, heat-ratio method). These methods have different sensitivities and systematic biases, particularly in low-flow conditions. The harmonisation assumes that sapwood-scale conversion eliminates method-specific offsets — an assumption that merits further validation.

3. **ERA5 VPD mismatch:** ERA5 VPD (derived from grid-cell mean dewpoint temperature) can differ systematically from site-level VPD measured at canopy or 2-m height. The magnitude and sign of this bias varies by site and season; see `notebooks/VPD_mismatch_analysis.py` for a quantification.

4. **Temporal coverage gaps:** Many sites have multi-year gaps due to sensor maintenance or data availability issues. The QC pipeline removes these, but their absence may affect model training for specific seasons or climate conditions.

5. **GlobMap LAI:** The 500-m resolution and 8-day temporal resolution of GlobMap LAI may not capture the within-site LAI variability relevant to sap flow at individual tree level. Cloud contamination during the growing season (especially in tropical regions) introduces interpolation artefacts.

### 15.2 Methodological Limitations

1. **Static spatial features:** Canopy height, stand age, and WorldClim bioclimatic variables are treated as time-invariant over the study period. In reality, these change at multi-year timescales (particularly post-disturbance). This limits the model's ability to capture ecosystem trajectories.

2. **Temporal extrapolation:** The model is trained on historical observations. Its ability to generalise to future climate states with unprecedented combinations of VPD, temperature, and drought severity has not been validated.

3. **Biome imbalance:** Despite stratified CV, the small number of tropical and tundra sites means the model may have limited skill in these biomes. Per-biome confidence intervals should be interpreted cautiously.

4. **Global prediction feature consistency:** The global prediction pipeline derives features from GEE ERA5-Land ImageCollections, which may differ slightly from the site-extraction pipeline in temporal aggregation or interpolation methods. These discrepancies should be quantified.

### 15.3 Future Work

- Incorporate additional sites from ongoing SAPFLUXNET v2 updates and newly curated Sapflow-internal records
- Extend temporal coverage of global predictions beyond the ERA5-Land data window currently processed
- Incorporate plant hydraulic traits (e.g., wood density, sapwood-to-leaf area ratio) as features to improve biome-level performance
- Implement seasonal cross-validation to explicitly test temporal generalisation to future climate conditions
- Develop a multi-model ensemble combining XGBoost, ANN, and CNN-LSTM outputs for uncertainty quantification
- Improve tropical coverage by targeting new data collection efforts at underrepresented biomes (tropical rain forest, boreal Asia)
- Transition from ERA5-Land point extraction to full gridded ERA5-Land downloads for seamless global prediction

---

## 16. References

### Datasets

- **SAPFLUXNET:** Poyatos, R. et al. (2021). Global transpiration data from sap flow measurements: the SAPFLUXNET database. *Earth System Science Data*, 13(6), 2607–2649. https://doi.org/10.5194/essd-13-2607-2021
- **ERA5-Land:** Muñoz-Sabater, J. et al. (2021). ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. *Earth System Science Data*, 13, 4349–4383. https://doi.org/10.5194/essd-13-4349-2021
- **MODIS MCD12Q1:** Friedl, M., Sulla-Menashe, D. (2019). MCD12Q1 MODIS/Terra+Aqua Land Cover Type Yearly L3 Global 500m SIN Grid V006. NASA EOSDIS Land Processes DAAC.
- **GlobMap LAI:** Yuan, H. et al. (2011). Reprocessing the MODIS Leaf Area Index products for land surface and climate modelling. *Remote Sensing of Environment*, 115(5), 1171–1187. https://doi.org/10.1016/j.rse.2011.01.001
- **BGI Forest Age:** Poulter, B. et al. (2019). The global forest age dataset and its uncertainties (GFA_v1). *Earth System Science Data*, 11, 1793–1808. https://doi.org/10.5194/essd-11-1793-2019
- **SoilGrids:** Poggio, L. et al. (2021). SoilGrids 2.0: producing soil information for the globe with quantified spatial uncertainty. *SOIL*, 7, 217–240.
- **WorldClim:** Fick, S.E., Hijmans, R.J. (2017). WorldClim 2: new 1-km spatial resolution climate surfaces for global land areas. *International Journal of Climatology*, 37, 4302–4315.
- **Meta Canopy Height:** Tolan, J. et al. (2024). Very high resolution canopy height maps from RGB imagery using self-supervised vision transformer and convolutional decoder trained on aerial lidar. *Remote Sensing of Environment*, 300, 113888. https://doi.org/10.1016/j.rse.2023.113888
- **ASTER GDEM:** NASA/METI/AIST/Japan Spacesystems, and U.S./Japan ASTER Science Team (2019). ASTER Global Digital Elevation Model V003. NASA EOSDIS Land Processes DAAC. https://doi.org/10.5067/ASTER/ASTGTM.003

### Methods

- **SHAP:** Lundberg, S.M., Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
- **XGBoost:** Chen, T., Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
- **Spatial block CV:** Roberts, D.R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. https://doi.org/10.1111/ecog.02881
- **pvlib:** Holmgren, W.F., Hansen, C.W., Mikofski, M.A. (2018). pvlib python: a python package for modeling solar energy systems. *Journal of Open Source Software*, 3(29), 884.
- **Köppen-Geiger:** Beck, H.E. et al. (2018). Present and future Köppen-Geiger climate classification maps at 1-km resolution. *Scientific Data*, 5, 180214.

---

## Appendix A: Complete Removal Log

[FILL IN: Table of all manual QC decisions from `removal_log.csv`, including site code, action type, affected column or period, failure mode, and date of decision.]

## Appendix B: Feature Importance Tables

[FILL IN: Complete SHAP feature importance tables from `shap_feature_importance.csv` and `shap_importance_pivot_by_pft.csv`.]

## Appendix C: Hyperparameter Search Results

**XGBoost hyperparameter grid** (from `outputs/logs/xgb_optimizer.log`):

| n_estimators | learning_rate | max_depth | min_child_weight | subsample | colsample_bytree | gamma | reg_alpha | reg_lambda | Mean CV Score |
|---|---|---|---|---|---|---|---|---|---|
| 500 | 0.05 | 3 | 1 | 0.7 | 0.7 | 0.0 | 0.005 | 1.5 | *[run to get]* |
| 500 | 0.05 | 3 | 1 | 0.7 | 0.7 | 0.0 | 0.005 | 2.0 | *[run to get]* |
| … | … | … | … | … | … | … | … | … | … |
| **1000** | **0.05** | **10** | **5** | **0.67** | **0.8** | **0.2** | — | — | **Best (repeated across all 10 folds)** |

*Full grid × CV score table: rerun with `--verbose 2` flag and capture from `outputs/logs/xgb_optimizer.log`.*

**ANN hyperparameter grid** (from `outputs/logs/ANN_optimizer.log`, 4 combinations):

| n_layers | units | dropout_rate | lr | Best CV score |
|---|---|---|---|---|
| 2 | 32 | 0.1 | 0.001 | *[run to get]* |
| 2 | **64** | 0.1 | 0.001 | **0.0015 (best)** |
| 3 | 32 | 0.1 | 0.001 | *[run to get]* |
| 3 | 64 | 0.1 | 0.001 | *[run to get]* |

**CNN-LSTM** (single configuration tested): cnn_layers=2, lstm_layers=2, cnn_filters=12, lstm_units=8, dropout=0.3. Best CV score: 0.3494.
