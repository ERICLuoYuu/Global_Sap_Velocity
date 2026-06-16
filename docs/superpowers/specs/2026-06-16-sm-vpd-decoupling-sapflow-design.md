# Site-level SM–VPD decoupling for sap velocity and canopy conductance

**Date:** 2026-06-16
**Status:** Design under review (revised after data-grounded review)
**Author:** (brainstormed with Claude)

## 1. Motivation

Liu et al. 2020 (*Nature Communications* 11:4892, "Soil moisture dominates
dryness stress on ecosystem production globally") showed that **soil moisture
(SM), not vapour pressure deficit (VPD), is the dominant driver of dryness
stress** on ecosystem production over ~71% of vegetated land. Their key
methodological move is to **decouple** the strongly collinear SM and VPD by
percentile-binning one variable and measuring the effect of the other *within*
each bin, where the residual SM–VPD correlation is ≈0.

Liu worked at the **pixel** level using satellite SIF as a GPP proxy. This
project ports the same decoupling technique to the **site** level, using
**ground-measured sap flow** from tree-dominated sites. It is a **new,
standalone analysis** that borrows the estimator math from
`src/afternoon_depression/decoupling.py` (Liu Eqs. 1–2) but shares no code path
with the afternoon-depression study (which measures diurnal depression ΔSF on an
AM/PM-split structure).

### Why this is worth doing even though sap velocity ≠ SIF
1. **Different question — water, not carbon.** Liu settled what limits ecosystem
   *carbon* uptake. Sap flow asks what controls *transpiration* (tree water
   use), a distinct function with its own live SM-vs-VPD debate
   (Novick 2016, Sulman 2016 — Liu's refs 11–12). The equivalent decoupling has
   not been done for the water flux with direct transpiration measurements.
2. **Direct measurement vs. proxy.** Sap flow is a direct physiological flux with
   site-measured VPD — an independent, ground-based test of Liu's SM dominance,
   free of the SIF→GPP proxy, reanalysis-SM, and 0.5° spatial-mismatch
   uncertainties.
3. **Trees specifically.** Liu's signal was weakest and most uncertain for
   high-tree-fraction pixels (Fig. 5b). This dataset is all trees — exactly that
   regime.
4. **The depth-mismatch question Liu flagged.** Liu's own closing limitation:
   shallow SM may *underestimate* SM effects for deep-rooted plants. The
   multi-depth SM design (below) directly tests this; `rooted_experiment_finding`
   already shows the verdict flips with depth.

## 2. Two response variables (the core design)

Because transpiration E ≈ gₛ·VPD, **VPD is a direct physical driver of sap
velocity**, unlike its indirect (stomatal) effect on SIF. To handle this we
analyze **two** response variables that bracket the question:

- **E — sap velocity** (water-use question). VPD enters directly; answers what
  controls how much water trees lose.
- **Gc — whole-tree canopy conductance per sapwood area** (physiological /
  SIF-analog question). Removes the direct VPD demand term, leaving the
  stomatal-type response — directly comparable to Liu's SIF.

### Canopy conductance calculation (Phillips & Oren 1998)
Daytime SFD is converted from `cm³ cm⁻²_Asw h⁻¹` to `kg m⁻²_Asw s⁻¹`
(× 2.778e-3, i.e. ÷360; ρ_water = 1 g cm⁻³) and then to daily whole-tree canopy
conductance normalized per unit sapwood area:

```
G_Asw = (115.8 + 0.4236·T) · (SFD / VPD) · (η·T0 / (T0 + T)) · exp(0.00012·h)
```

where SFD = daytime sap flux density (kg m⁻²_Asw s⁻¹), T = daytime temperature
(°C), VPD = daytime VPD (kPa), η = 44.6 mol m⁻³ (molar air density at STP),
T0 = 273 K, h = site altitude (m, from `elevation`; SRTM fallback for the rare
missing case, as in Flo). Gc is in molar units per sapwood area. Valid under the
well-coupled canopy–atmosphere assumption (g_aero ≫ g_stomatal).

> **Source (confirmed):** Flo, V., Martínez-Vilalta, J., et al. (2021).
> "Climate and functional traits jointly mediate tree water-use strategies."
> *New Phytologist* 231(2): 617–630, doi:10.1111/nph.17404, **Eqn 2** (after
> Phillips & Oren 1998). Constants, units, SRTM fallback, and the bulk-vs-
> stomatal (G′_Asw) distinction all verified against the paper.

**Methodological novelty vs. the source papers.** Flo (2021) derived Gc
sensitivity with an additive log-log linear mixed model
(`GAsw ~ −logVPD + logSWC`), which partials out SM–VPD collinearity
*statistically* — the very approach Liu (2020) argues is confounded by that
collinearity. This project instead applies **Liu's nested-binning decoupling**
(collinearity-robust by construction) to **Flo's rigorous Gc** response (and to
raw E). Combining the two is the contribution. Note Flo independently used
**5 SWC bins**, supporting our quintile-primary choice.

**Optional SI robustness (G′_Asw):** on the wind-available subset
(`ws` present), remove the aerodynamic-conductance contribution to obtain
whole-tree stomatal conductance, mirroring the source paper's SI. Not in the v1
headline; flagged as a later robustness analysis.

### ⚠️ The Gc–VPD arithmetic confound (must be respected)
Because Gc ∝ SFD/VPD, binning Gc *by VPD* induces a **spurious** negative
Gc–VPD relationship even with zero stomatal response (Oren et al. 1999; Liu
ref 10). Therefore:
- **ΔGc(SM|VPD)** is computed *within* VPD bins (VPD≈constant) → **free of the
  artifact**, and is the **clean, Liu-comparable SM signal**.
- **ΔGc(VPD|SM)** is biased toward VPD-dominance by the 1/VPD term → report it,
  but interpret only qualitatively / as an upper bound on VPD's role.
- The true VPD effect is **bracketed**: E inflates VPD via the demand term; Gc
  deflates it via the 1/VPD term. Reporting both bounds the answer honestly.

### Scientific design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Response variables | **E (sap velocity) AND Gc (canopy conductance)**, each normalized per site by the mean of values above its 90th percentile | E = water-use; Gc = physiological SIF-analog. Per-site normalization → cross-site comparability (Liu) |
| SM axis | **5 variants**: `swvl1`, `swvl2`, `swvl3`, `swvl4` (per-depth) + **root-zone** (fixed 0–100 cm: 0.07·swvl1 + 0.21·swvl2 + 0.72·swvl3) | Tests the depth-mismatch finding; root-zone is Liu's 0–1 m analog (ERA5-Land layer thicknesses to 100 cm) |
| Day filter | Liu's three thresholds on daytime daily means: **Tair > T_min, VPD > 0.5 kPa, PPFD > 500 µmol m⁻² s⁻¹** (all three, AND). **T_min = 15 °C primary; 5 °C as a sensitivity run** | Screens to dryness-relevant days. 15 °C is Liu-faithful but warm-biased (excludes boreal); the 5 °C frost-free run includes cold-conifer sites |
| Rain filter | **None** (Liu-faithful) | PPFD>500 + VPD>0.5 already removes wet-canopy low-demand days; an explicit precip filter would truncate the wettest-SM end the SM effect needs |
| Negative days | Negative daily sap velocity → drop | Reverse/invalid flow (belt-and-suspenders on upstream reverse-flow QC). No min-activity floor: a low SV on a high-demand day can be *genuine* severe SM limitation, so a floor would erase the strongest SM signal |
| Gap-filling | **Not used** — source is `outliers_removed` (measured, QC'd) | Avoids circularity: env-driven gap-filling would make binning SV against SM/VPD partly self-referential. Upstream QC already did reverse-flow detection |
| PPFD source | `ppfd_in` → else `sw_in`×2.04 → else ERA5 `surface_solar_radiation_downwards_hourly`×2.04; record source per site | 25/165 sites lack `ppfd_in`+`sw_in`; ERA5 path recovers them. 2.04 µmol J⁻¹ shortwave→PAR |
| Treatment filter | **Required** — merge run with `--apply-treatment-filter` | Drops irrigation/drought/trenching/CO₂/shade plants that inject an artificial SM signal (causal hygiene) |
| Growing-season screen | **Off** at merge time | Liu's Tair>15 °C threshold does the seasonal screening; avoids double-filtering |
| Climate source | Parameterized like the afternoon-depression loader: site-measured VPD+Tair primary, ERA5 selectable | Site-measured is the true microclimate; ERA5 kept for robustness |
| Bin count | **Both 5 (quintiles) and 10 (deciles)** reported | Site data are sparse (median ~104 valid days); quintiles keep ~7 pts/nested cell, deciles are the Liu-exact check on data-rich sites |
| Min points per bin | **≥3** (fixed `MIN_BIN_COUNT = 3`) | Numerical-stability floor; Liu's >10 empties site-level bins |
| Min valid days per site | **Sweep {120, 240, 360}** | Site-inclusion threshold; report verdict at all three for robustness |
| Bins type | `qcut` percentiles, `duplicates="drop"` | Matches Liu; quantile binning makes VPD vs ln(VPD) irrelevant |
| Aridity classes | Liu/UNESCO thresholds (0.05, 0.2, 0.5, 0.75, 1.2 of prcip/PET) | Matches Liu Fig. 5a bins |

### Inherited / structural caveats (documented, not fixed)
- **Severe attrition.** The three thresholds each sit near the data median
  (VPD≈0.52, PPFD≈467, Tair≈15.0), so ≈0.5³≈12% of days survive. On the 24-h
  merged set only 63 / 29 / 16 sites reach 120 / 240 / 360 valid days; the
  daytime-only run (higher daytime VPD/PPFD/Tair) will retain more — treat those
  as a conservative lower bound. Ship an **attrition table** (sites & days
  surviving each stage).
- **VPD>0.5 truncation** removes the lower half of the VPD range → attenuates
  ΔSV/ΔGc(VPD|SM). Inherited from Liu.
- **Deep SM (swvl4, 100–289 cm)** has low temporal variance; its bins may track
  seasonal drift, not dryness events → interpret deepest-layer decoupling
  cautiously.
- **Report signs**, not just magnitudes: ΔE(VPD|SM) may be positive (demand
  drives transpiration), unlike Liu's negative ΔSIF(VPD|SM).

### Known gaps (accepted for v1)
- **No tree-cover column** → Liu Fig. 5b tree-cover gradient replaced by
  biome / PFT / aridity / **canopy-height-class** groupings (canopy height is a
  partial proxy for stand stature/fraction).
- **No per-site root depth** → root-zone uses fixed 0–100 cm, matching Liu (who
  used fixed 0–1 m and did not weight by root depth). Fan-2017 not used.

## 3. Data flow

```
merge_gap_filled_hourly_orginal.py  --daytime-only --apply-treatment-filter
        (NO --growing-season-only)  --output-dir <decoupling_run>
                 │
                 ▼
   <decoupling_run>/daily/*_daily.csv   (daytime, treatment-filtered, all-season)
                 │   loader.py: standardise + PPFD source resolution + Liu day-filter
                 │   conductance.py: SFD→Gc (Phillips & Oren); per-site normalize E and Gc
                 ▼
   per-site-day table: site_name, date, E_norm, Gc_norm, vpd, tair, ppfd,
                       swvl1..4, root_zone_sm, biome, pft, aridity,
                       canopy_height, elevation, lat, lon
                 │   decoupling.py  (per response × per SM variant × per bin count)
                 ▼
   per-site effects: dResp_sm_given_vpd, dResp_vpd_given_sm, sensitivity
                 │   run_*.py aggregation + plotting.py
                 ▼
   CSVs + depth-profile dissociation table + attrition table + figures
```

Confirmed columns in the daily merged schema: `sap_velocity`, `vpd`, `ta`,
`temperature_2m`, `dewpoint_2m`, `ppfd_in`, `sw_in`,
`surface_solar_radiation_downwards_hourly`, `volumetric_soil_water_layer_1..4`,
`pft`, `biome`, `prcip/PET`, `canopy_height`, `elevation`, `ws`,
`latitude_x`, `longitude_x`.

## 4. Estimator (`decoupling.py`)

Per site, per response (E or Gc), per SM variant, per bin count (5, 10), on the
per-site-normalized response `resp_norm`:

- Percentile-bin the SM axis and the VPD axis independently (`qcut`,
  `duplicates="drop"`).
- **ΔResp(SM|VPD)** = mean over populated VPD bins of
  `resp_norm(lowest populated SM bin) − resp_norm(highest populated SM bin)`
  (low SM = stressor → sign flipped, Liu Eq. 2).
- **ΔResp(VPD|SM)** = mean over populated SM bins of
  `resp_norm(highest populated VPD bin) − resp_norm(lowest populated VPD bin)`
  (Liu Eq. 1).
- A conditioning bin contributes only if it has ≥2 distinct driver bins, each
  with ≥ `MIN_BIN_COUNT` points. Require ≥2 populated conditioning bins for a
  site's effect to be valid (else NaN, site reported as insufficient).
- **Sensitivity** δResp/δSM per **0.1 m³/m³** within VPD bins (Liu's
  standardized sensitivity; removes the SM-range effect).

Math ported from `afternoon_depression/decoupling.py`, reduced to the VPD-vs-SM
pair (no Tair leg).

## 5. Outputs (`run_sm_vpd_decoupling.py` + `plotting.py`)

- **Per-site CSV** per (response × SM variant × bin count):
  `site_name, n_days, dResp_sm_given_vpd, dResp_vpd_given_sm, sensitivity,
  biome, pft, aridity, canopy_height, lat, lon`.
- **Headline depth-profile dissociation table**: % of sites where
  `|ΔResp(SM|VPD)| > |ΔResp(VPD|SM)|`, for **E vs Gc**, per SM variant, at each
  `min_valid_days` ∈ {120, 240, 360} and bin count ∈ {5, 10}, with surviving
  site counts. The E-vs-Gc contrast is the scientific headline.
- **Attrition table**: sites & days surviving each filter stage (treatment →
  Tair>15 → VPD>0.5 → PPFD>500 → min_valid_days).
- **Figures**:
  - **(a1)** Multiple example sites spanning a climate gradient (most-data site
    per aridity class / contrasting biomes): Resp-vs-VPD binned by SM, and
    Resp-vs-SM binned by VPD (Liu Fig. 3 c/d analog), small-multiples.
  - **(a2)** Cross-site aggregate: per-site normalized Resp per SM-percentile
    and per VPD-percentile bin, then mean & median across sites per bin (each
    site = one unit, unweighted). Valid via shared percentile x-axis + per-site
    normalization.
  - **(b)** Per-depth SM-vs-VPD dominance across swvl1→4 + root-zone, for E and
    Gc side by side.
  - **(c)** Violin gradients of ΔResp(SM|VPD) grouped by biome, PFT, aridity,
    and canopy-height class (Liu Fig. 5 analog).

## 6. Module layout (`src/sm_vpd_decoupling/`)

| File | Responsibility | Approx. size |
|---|---|---|
| `loader.py` | Resolve daily dir, standardise, PPFD source resolution, Liu day-filter, root-zone weighting, per-site normalization | <350 |
| `conductance.py` | SFD unit conversion + Gc (Phillips & Oren), optional G′ with wind | <150 |
| `decoupling.py` | Percentile binning + ΔResp(SM\|VPD) / ΔResp(VPD\|SM) estimator | <150 |
| `sensitivity.py` | δResp/δSM per 0.1 m³/m³ | <100 |
| `plotting.py` | Figures (a1), (a2), (b), (c) | <350 |
| `run_sm_vpd_decoupling.py` | CLI: orchestrate responses × variants × bin counts × thresholds; write CSVs + tables + figures | <300 |
| `job_sm_vpd_decoupling.sh` | Palma SLURM submission | <50 |
| `tests/` | pytest suite (see §7) | — |

Each file focused and <400 lines, per project coding style.

## 7. Testing (TDD)

- **Estimator correctness** on synthetic data: SM-dominant case (Resp driven
  only by SM → `|ΔResp(SM|VPD)| ≫ |ΔResp(VPD|SM)|`, correct signs); VPD-dominant
  mirror.
- **Conductance**: Gc equation reproduces hand-computed values for known
  (SFD, T, VPD, h); SFD unit conversion (cm³ cm⁻² h⁻¹ → kg m⁻² s⁻¹) exact;
  Gc ∝ 1/VPD verified (documents the confound).
- **PPFD source resolution**: ppfd_in→sw_in→ERA5 fallback picks the right source
  and conversion; site lacking all radiation is reported, not silently dropped.
- **Binning edge cases**: sparse site, degenerate (low-variance) series →
  collapses via `qcut` duplicates-drop, bins below `MIN_BIN_COUNT`, <2 populated
  conditioning bins → NaN and reported as insufficient (this subsumes an
  explicit range guard).
- **Normalization**: per-site 90th-percentile anchor; dominance verdict
  invariant to normalization at a single site.
- **Liu day-filter**: row counts match threshold logic; T_min switch (15 vs 5)
  changes retained days as expected; AND-combination of the three thresholds;
  negative daily SV dropped.
- **No rain filter**: precip>0 days with PPFD>500 & VPD>0.5 are retained.
- **Loader**: column standardisation, root-zone weighting arithmetic, ERA5 vs
  site climate-source switch, attrition counts.
- Target ≥80% coverage.

## 8. Out of scope (v1)
- Tree-cover gradient (no data column; canopy height substitutes).
- Per-site root-depth weighting (fixed 0–100 cm, matching Liu).
- Tair leg of the decoupling (VPD-vs-SM only).
- G′_Asw aerodynamic-corrected conductance as a headline result (kept as an
  optional SI robustness on the wind-available subset).
- Converting ΔE to absolute carbon/water-flux changes.
