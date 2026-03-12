# Stage 1: Sapflow-internal ETL Codemap

**Last Updated:** 2026-03-12
**Module:** `Sapflow-internal/`
**Entry Points:** `sap_reorganizer.py`, `sap_unit_converter.py`, `plant_to_sapwood_converter.py`, `icos_data_extractor.py`

---

## Purpose

Standardize 20 internal European sapflow sites into SAPFLUXNET format, convert measurement units, and extract environmental data from co-located ICOS flux towers.

**Output:** Data ready for merging with SAPFLUXNET (165 sites) in Stage 4

---

## Processing Pipeline

```
┌─────────────────────────────────────────┐
│  RAW SAPFLOW-INTERNAL FILES             │
│  (heterogeneous, non-standard)          │
│  Sapflow-internal/raw_data/             │
└────────────┬────────────────────────────┘
             │
    ┌────────▼────────────────┐
    │  sap_reorganizer.py     │
    │  - Parse raw formats    │
    │  - Extract TIMESTAMP    │
    │  - Rename columns       │
    │  → SAPFLUXNET format    │
    └────────┬─────────────────┘
             │
    ┌────────▼──────────────────────────┐
    │  sap_unit_converter.py            │
    │  - Identify unit type from meta   │
    │  - Convert to cm³ cm⁻² h⁻¹        │
    │  - Preserve NaN, QC flags         │
    └────────┬───────────────────────────┘
             │
    ┌────────▼────────────────────────┐
    │  plant_to_sapwood_converter.py  │
    │  - Aggregate plant → sapwood    │
    │  - Apply sapwood area scaling   │
    │  - Merge sensor weights         │
    └────────┬─────────────────────────┘
             │
    ┌────────▼──────────────────────┐
    │  icos_data_extractor.py       │
    │  (unified, replaces 2 scripts)│
    │  - Strategy pattern: L2 → FP  │
    │  - Fetch from ICOS CP API     │
    │  - Standardize column names   │
    └────────┬───────────────────────┘
             │
    ┌────────▼──────────────────────────────┐
    │  OUTPUT: SAPFLUXNET-format            │
    │  Sapflow_SAPFLUXNET_format_unitcon/   │
    │  ├── sapwood/                         │
    │  │   └── {site}_sapf_data.csv         │
    │  └── env_icos/                        │
    │      └── {site}_env_data.csv          │
    └───────────────────────────────────────┘
```

---

## Key Modules

### 1. `sap_reorganizer.py`

**Purpose:** Parse raw sapflow files and convert to SAPFLUXNET CSV structure

**Key Functions:**

| Function | Description |
|----------|-------------|
| `reorganize_sapflow_data(input_dir, output_dir, scale)` | Main orchestrator |
| `parse_site_raw_format(site_code)` | Detect and parse site-specific raw format |
| `build_sapfluxnet_csv(data_dict, site_code)` | Write standardized `{site}_sapf_data.csv` |

**Inputs:**
- Raw data in various formats (site-dependent)
  - Some: Excel files with heterogeneous sheet layouts
  - Some: Text files with fixed or variable columns
  - Some: HDF5 or NetCDF binary formats

**Outputs:**
- `Sapflow_SAPFLUXNET_format_unitcon/sapwood/{site}_sapf_data.csv`
- Columns: `TIMESTAMP`, `TSPD_1`, `TSPD_2`, ..., `TSPD_N_QC`

**Usage:**
```bash
python Sapflow-internal/sap_reorganizer.py
```

---

### 2. `sap_unit_converter.py`

**Purpose:** Convert sapflow velocity units to standardized cm³ cm⁻² h⁻¹

**Key Functions:**

| Function | Description |
|----------|-------------|
| `identify_unit_type(site_code)` | Query metadata to determine unit of raw data |
| `convert_to_target_units(values, from_unit, to_unit, sapwood_area)` | Apply unit conversion formula |

**Supported Unit Conversions:**

| From | To | Formula | Sapwood Area Dependency |
|------|----|---------|-----------------------|
| cm h⁻¹ (length) | cm³ cm⁻² h⁻¹ | v × A_tree / A_sapwood | Yes (requires tree dimensions) |
| mm h⁻¹ | cm³ cm⁻² h⁻¹ | v / 10 × A_tree / A_sapwood | Yes |
| g h⁻¹ (mass) | cm³ cm⁻² h⁻¹ | v / ρ_water / A_sapwood | Yes (density = 1 g/cm³) |
| cm³ h⁻¹ (volume) | cm³ cm⁻² h⁻¹ | v / A_sapwood | Yes |
| Already cm³ cm⁻² h⁻¹ | — | Identity | No |

**Metadata Source:**
- `Sapflow-internal/metadata/unit_mapping.csv` (defines unit type per site)
- Tree dimensions from corresponding `*_plant_md.csv`

**Inputs:**
- `Sapflow_SAPFLUXNET_format_unitcon/sapwood/{site}_sapf_data.csv` (from sap_reorganizer)

**Outputs:**
- Same location, overwritten with unit-converted values

**Usage:**
```bash
python Sapflow-internal/sap_unit_converter.py
```

---

### 3. `plant_to_sapwood_converter.py`

**Purpose:** Aggregate plant-level sensors to sapwood-area-weighted sapwood velocity

**Key Functions:**

| Function | Description |
|----------|-------------|
| `aggregate_plant_to_sapwood(sapf_data, plant_metadata)` | Main aggregation logic |
| `load_plant_metadata(site_code)` | Read `{site}_plant_md.csv` with sapwood area per plant |
| `compute_weighted_mean(sensor_readings, sapwood_areas)` | Weighted average by sapwood area |

**Aggregation Logic:**

1. **Input:** Multi-sensor data (e.g., 5 plants with 2 sensors each = 10 columns)
2. **Grouping:** Group readings by plant ID from metadata
3. **Weighting:** Weight each plant's mean by its sapwood area
4. **Output:** Single `TSPD` column representing sapwood-area-weighted velocity

**Formula:**
```
V_sapwood = Σ(V_plant_i × A_sapwood_i) / Σ(A_sapwood_i)
```

**Handling Missing Data:**
- If ≥50% of sensors within a plant are missing, mark that plant's contribution as NaN
- If ≥50% of plants are valid, compute aggregate; else mark timestamp as NaN

**Inputs:**
- `Sapflow_SAPFLUXNET_format_unitcon/sapwood/{site}_sapf_data.csv` (unit-converted)
- `Sapflow-internal/raw_data/{site}_plant_md.csv` (plant IDs, sapwood areas)

**Outputs:**
- `Sapflow_SAPFLUXNET_format_unitcon/sapwood/{site}_sapf_data.csv` (single TSPD column + QC flag)

**Usage:**
```bash
python Sapflow-internal/plant_to_sapwood_converter.py
```

---

### 4. `icos_data_extractor.py` — Unified ICOS Extractor (NEW/REFACTORED)

**Purpose:** Extract environmental data from ICOS CO2-permitting sites; replaces two separate scripts

**Replaces:**
- `icos_env_extractor.py` (L2-tier products)
- `icos_fluxnet_extractor.py` (Fluxnet Product fallback)

**Architecture:** Strategy Pattern

```python
├── ICOSSiteConfig(dataclass)
│   ├── sapflow_name: str
│   └── icos_id: str
│
├── EnvironmentDataStrategy(ABC)
│   ├── L2ProductStrategy
│   │   └── Tries: ETC L2 Fluxnet, Meteo
│   └── FluxnetProductStrategy
│       └── Fallback: "Fluxnet Product"
│
└── ICOSExtractor
    ├── load_site_registry()
    ├── fetch_and_validate()
    └── write_standardized_csv()
```

**Key Components:**

| Component | Responsibility |
|-----------|-----------------|
| `ICOSSiteConfig` | Frozen dataclass linking sapflow_name ↔ icos_id |
| `SITE_REGISTRY` | Hardcoded list of 17 known sapflow-ICOS site pairs |
| `VARIABLE_MAPPING` | Maps ICOS column names to standardized vars (ta, vpd, ws, ppfd, etc.) |
| `QC_MAPPING` | Maps ICOS QC flags to standardized format |
| `EnvironmentDataStrategy` | Abstract base for L2 vs Fluxnet Product strategies |
| `ICOSExtractor` | Orchestrates fetch, validate, standardize |

**Site Registry (17 sites):**

```python
SITE_REGISTRY = [
    # From icos_fluxnet_extractor (authoritative)
    ICOSSiteConfig("ES-LM1", "ES-LM1"),
    ICOSSiteConfig("ES-LM2", "ES-LM2"),
    ICOSSiteConfig("IT-CP2_sapwood", "IT-Cp2"),
    ICOSSiteConfig("AT_Mmg", "AT-Mmg"),
    ICOSSiteConfig("CH-Lae_daily", "CH-Lae"),
    ICOSSiteConfig("CH-Lae_halfhourly", "CH-Lae"),
    ICOSSiteConfig("ES-Abr", "ES-Abr"),
    # From icos_env_extractor
    ICOSSiteConfig("CH-Dav_daily", "CH-Dav"),
    ICOSSiteConfig("CH-Dav_halfhourly", "CH-Dav"),
    ICOSSiteConfig("DE-Har", "DE-Har"),
    ICOSSiteConfig("DE-HoH", "DE-HoH"),
    ICOSSiteConfig("ES-LMa", "ES-LMa"),
    ICOSSiteConfig("FI-Hyy", "FI-Hyy"),
    ICOSSiteConfig("FR-BIL", "FR-Bil"),
    ICOSSiteConfig("SE-Nor", "SE-Nor"),
]
```

**Variable Mapping (Superset):**

| ICOS Column | Target Name | Tier | Description |
|-------------|-------------|------|-------------|
| TA_F | ta | L2/FP | Air temperature (°C) |
| VPD_F | vpd | L2/FP | Vapor pressure deficit (kPa) |
| SW_IN_F | sw_in | L2/FP | Shortwave radiation (W m⁻²) |
| PA_F | pa | L2/FP | Atmospheric pressure (kPa) |
| WS_F | ws | L2/FP | Wind speed (m s⁻¹) |
| P_F | precip | L2/FP | Precipitation (mm) |
| PPFD_IN | ppfd | L2 only | Photosynthetic photon flux density (µmol m⁻² s⁻¹) |
| RH | rh | L2 only | Relative humidity (%) |
| TS_F_MDS_1 | ts | L2 only | Soil temperature (°C) |
| SWC_F_MDS_1 | swc | L2 only | Soil water content (m³ m⁻³) |

**Extraction Strategy:**

1. **Try L2-tier products first:**
   - Endpoint: ICOS Data Portal → L2 `ETC_L2_Fluxnet` or `ETC_L2_Meteo`
   - Variables: Core (TA_F, VPD_F, etc.) + extended (PPFD_IN, RH, TS, SWC)
   - Temporal check: Require ≥30 overlapping days with sapflow obs

2. **If L2 fails or insufficient overlap:**
   - Fall back to Fluxnet Product endpoint
   - Variables: Core only (no PPFD_IN, RH, TS, SWC)

3. **Standardize output:**
   - Column names: lowercase + `_qc` suffix for flags
   - Datetime: ISO 8601 format
   - Reindex to hourly grid (forward-fill if sub-hourly)

**Inputs:**
- ICOS CP API (https://data.icos-cp.eu/)
- Site registry hardcoded in script

**Outputs:**
- `Sapflow_SAPFLUXNET_format_unitcon/env_icos/{site}_env_data.csv`
- Columns: `TIMESTAMP`, `ta`, `vpd`, `sw_in`, `pa`, `ws`, `precip`, [optional: `ppfd`, `rh`, `ts`, `swc`], and corresponding `_qc` flags

**Requirements:**
- `icoscp` package: `pip install icoscp`

**Usage:**
```bash
# Extract for all registered sites
python Sapflow-internal/icos_data_extractor.py

# With verbose logging
python Sapflow-internal/icos_data_extractor.py --verbose

# Single site (useful for debugging)
python Sapflow-internal/icos_data_extractor.py --site ES-LM1
```

**Error Handling:**
- Logs all ICOS fetch errors and fallback attempts
- Continues on site failure; summary report at end
- Writes empty CSV if site has no valid data (placeholder for manual intervention)

---

## Execution Order (Stage 1 within Full Pipeline)

```bash
# 1. Reorganize raw sapflow files to SAPFLUXNET structure
python Sapflow-internal/sap_reorganizer.py

# 2. Convert sapflow units to cm³ cm⁻² h⁻¹
python Sapflow-internal/sap_unit_converter.py

# 3. Aggregate plant-level to sapwood-area-weighted sapwood
python Sapflow-internal/plant_to_sapwood_converter.py

# 4. Extract and standardize environmental data from ICOS
python Sapflow-internal/icos_data_extractor.py
```

**Time to completion:** ~5–15 minutes (depending on ICOS API responsiveness)

**Output directories ready:**
```
Sapflow_SAPFLUXNET_format_unitcon/
├── sapwood/
│   ├── ES-LM1_sapf_data.csv
│   ├── ES-LM2_sapf_data.csv
│   └── ... (20 sites)
└── env_icos/
    ├── ES-LM1_env_data.csv
    ├── ES-LM2_env_data.csv
    └── ... (17 sites with ICOS match)
```

---

## Related Codemaps

- **[extractors-stage2.md](extractors-stage2.md)** — Consumes Sapflow-internal data alongside SAPFLUXNET for feature extraction
- **[merging-stage4.md](merging-stage4.md)** — Merges Sapflow-internal with SAPFLUXNET before feature integration
- **[INDEX.md](INDEX.md)** — Overview of all stages

---

## Important Notes

1. **Heterogeneous Input:** Raw Sapflow-internal formats vary widely; each site may require bespoke parsing
2. **Unit Metadata Critical:** sap_unit_converter relies on accurate `unit_mapping.csv`; verify before running
3. **Plant Metadata Required:** plant_to_sapwood_converter needs sapwood area per plant; missing data handled gracefully
4. **ICOS Dependency:** icos_data_extractor requires internet and ICOS CP API availability; sites without ICOS match get empty env files (filled later with alternative sources if needed)
5. **Unified Extractor:** icos_data_extractor replaces 2 scripts and adds strategy pattern for robust fallback
6. **No Overwrite Protection:** All scripts write to output directories without confirmation; ensure backups if rerunning

---

## Testing

```bash
# Unit tests in Sapflow-internal/tests/
pytest Sapflow-internal/tests/test_icos_data_extractor.py -v
pytest Sapflow-internal/tests/ -v --tb=short
```
