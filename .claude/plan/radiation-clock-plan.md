# Plan: radiation and clock defects — sw_in support, ext_rad, site clock offsets

Branch `feat/sapwood-combined` on Palma (`/scratch/tmp/yluo2/gsv`, HEAD `66252b4`). All work in a
worktree on a new branch `feat/radiation-clock` (`/scratch/tmp/yluo2/gsv-wt/radfix`) so the chain in flight
(select `46747901`, train `46747902`) and the serving jobs never import half-changed code.
Written 2026-09-16 at the end of dev-workflow Phase 2. Decisions still open are in §8.

**Sign convention used everywhere below: positive = the label is LATE (the reading was taken before its
stamp), true time = label − offset.** The 09-14 probes (`probe_sap_clock*.py`) used the opposite sign.

---

## 0. Evidence this plan rests on

| # | Finding | Source |
|---|---|---|
| E1 | Trained daily `sw_in` is inflated because the archive ships `sw_in = NA` at night and the merge averages only the hours present. Median bias vs a clearness-scaled reference: USA_HIL_HF1_PRE +96.7 %, USA_HIL_HF1_POS / HF2 +83.8 %, USA_UMB_CON +71.4 %, USA_TNO +60.0 %, USA_INM +31.1 %, THA_KHU +20.0 %, FIN_HYY_SME +4.3 %. 7 of 184 sites exceed 5 %. | 46752793, 46757405 |
| E2 | kt = sw_in / ext_rad > 1 in the in-flight servable tree: **2,141** site-days. Geometry ext_rad alone: 2,177 (ext_rad is not the cause). Clearness-scaled sw_in + served ext_rad: **221** (FIN_HYY_SME 155, USA_BNZ_BLA 18, KOR_TAE_TC1/2/3 11 each, RUS_FYO 6, CAN_TUR_P39_* 3, SWE_NOR_ST3 2, SWE_SVA_MIX_NON 1). | 46757405 |
| E3 | The merge has **no completeness rule** for env daily statistics. `resample("D").agg(["mean","min","max","sum"])` runs over the hours in which sap AND env both had rows (`merge_gap_filled_hourly_orginal.py` hourly inner join ~l.1144, daily agg ~l.1340–1390). ERA5 and the gauge were moved to own-hours complete-day statistics on 09-14; env sensor columns were not. Consequences beyond E1: partial-day `ext_rad` p95 error 414 / 226 / 207 / 202 W m-2 at COL_MAC_SAF_RAD / ARG_MAZ / ARG_TRE / POL_TUC; POL_TUC trained `ta` p95 2.4 K and `rh` 8.8 % off its own complete-day mean. | code; 46757405 |
| E4 | Cost of candidate rules, rows with every core env feature valid (ta, vpd, ws, rh, sw_in) vs today's 169,224: H24 everywhere 94.4 %; H20 everywhere 98.6 %; **KT80 radiation + H20 others 101.1 %**; NF24 radiation + H24 others 95.3 %. Per-site worst under KT80+H20: USA_BNZ_BLA −28 %, FIN_HYY_SME −23 %, USA_SYL_HL2 −22 %, ESP_VAL_SOR −19 %, ITA_TOR / SEN_SOU_PRE / USA_UMB_CON −18 %. | 46757405 |
| E5 | Knockout on complete days (truth known), median over 182 sites. sw_in, k daylight hours removed at random: k=2 nightfill+mean 4.2 % (p95 16 %) vs clearness-scaled 1.3 % (p95 7.2 %); k=4 afternoon: nightfill+mean 26 %, scaled refuses (coverage < 0.8). ta mean, k=4: random 0.15 K (p95 0.58), block 0.38 K, afternoon 0.67 K (p95 1.27); ta_max afternoon k=4 p95 1.84 K. vpd mean k=4 random 0.011 kPa, afternoon 0.051 kPa. | 46757568 |
| E6 | Shipped `ext_rad` is displaced 2–11 h at 15 sites (+ BRA_CAX_CON, DEU_STE_2P3 by mask); pure displacement cancels in a full UTC day (median daily error 1–3.6 W m-2). The served value is `src.serving.era5_daily.ext_rad_daily` — geometry averaged over 48 instants of the UTC day. Corpus median site p95 of tree-vs-served ext_rad: 0.87 W m-2; the large errors are partial days (E3) and FIN_HYY_SME's constant ~21 W m-2. | 46728968, 46729009, 46752793, 46757405 |
| E7 | Clock offsets are in the **archive**, not our code: raw `TIMESTAMP` equals Stage-3 `outliers_removed` (FIN_PET `2008-06-15T22:00:00Z` in both; USA_HUY_LIN_NON `2017-01-01T05:00:00Z`). Stage 3 does no tz conversion (`env_analyzer.py` ~l.196–201 is a bare string literal). Both QC day/night masks come from `TIMESTAMP` + pvlib (`env_analyzer.py` ~l.502, `sap_analyzer.py` ~l.728); sap incomplete-day removal groups by `solar_TIMESTAMP` date (`_remove_incomplete_days`); `solar_TIMESTAMP` is shifted together with `TIMESTAMP` at the offset sites, so it cannot self-check. | explore 1–8 |
| E8 | Per-component offsets (table in §3 A6). ISR_YAT_YAT and USA_HUY_LIN_NON: env only (sap on time). USA_ORN_ST2_AMB and FIN_PET: env + sap. SWE_SVA_MIX_NON: sap only. The thick-cloud objection is answered: clear-day correlation and daylight edges agree at the big four. | 46729171, 46756840, 46756897 |
| E9 | Corpus data files are **hardlinked** to the archive copies (`links=2` on native and converted env/sapf files). In-place rewriting would edit the archive too. | stat |
| E10 | `removal_log.csv` has rows at ITA_TOR (12) and USA_HIL_HF1_POS (11). Periods are matched on `TIMESTAMP`, i.e. in LABEL time. The env analyzer applies no removal log. | grep |
| E11 | FFS `data_loader` and the ML trainer `dropna` whole rows, so a NaN feature costs a training row. | grep |
| E12 | Stage-3 loaders key `location` on the first two filename tokens: USA_ORN_ST1_AMB and USA_ORN_ST2_AMB both become `USA_ORN`. | code |
| E13 | Precedent: internal local-clock sites were fixed ONCE at ingest (`Sapflow-internal/stored_clock.py`, `timestamp_conformance.conform_for_site`), downstream `reconform` has no offset parameter, and a value gate (`reorg_review.check_stored_clock_is_utc`) proves each file. A missed registry key there would have been silent because the lookup defaults to 0.0. | local repo |

---

## 1. Problem space (Aspect 1)

### What changes
- **WP-A — clock (P3).** A reviewed registry of per-site, per-component clock offsets applied once at the
  Stage-3 load boundary; a validator; a detection gate; Stage 3 re-run for registry sites.
- **WP-B — daily env support and ext_rad (P1, P2).** Env sensor daily statistics come from env's OWN hours
  under an explicit support rule, with radiation clearness-scaled against computed geometry. The daily
  `ext_rad` becomes the served geometry value. Invariants are enforced and recorded in the tree's provenance.
- **WP-C — gates and rebuild.** Test gate, clock gate, rebuild of the servable tree, before/after comparison
  against the tree now in flight.

### Explicitly out of scope
- Rewriting any corpus or archive file (E9). The shipped `ext_rad` column stays on disk untouched.
- `fill_ext_rad` policy in the Stage-1 generators (internal sites). After WP-B nothing trained reads the
  shipped column, so the policy error becomes inert for the model. Documented, not changed.
- The residual 221 kt>1 site-days (FIN_HYY_SME, USA_BNZ_BLA, KOR_TAE_*): a different defect. WP-B records
  them per site and the gate fails on regressions, but the cause is investigated separately.
- Sap TARGET support (daily sap mean over partial days) — the same class, separate change.
- Per-tree anti-phase series at ISR_YAT_YAT (e.g. tree Pha_Jt_2) — per-tree QC, separate.
- CHN_YUN_YUN `sw_in` units; USA_PAR_FER dead radiation sensor; FRA_PUE night readings.
- The hourly merged tree (`hourly/`), hourly models, DST-style seasonal offsets (none confirmed).
- Training on ERA5-Land radiation instead of site sensors (§2 P1-D) — a modelling decision, not this fix.
- The 57 duplicate daily rows at 22 sites (`daily_frame_carries_time_of_day`).

---

## 2. Approaches considered (Aspect 2)

### P3 — clock offsets
| | Approach | Verdict |
|---|---|---|
| A | **Reviewed registry applied at the Stage-3 load boundary** (env + sap loaders, flags with them) | **Chosen.** Raw data stays byte-identical to the archive; QC runs on true time; one choke point feeds the merge; mirrors the E13 precedent. |
| B | Rewrite the corpus files once (like `repair_stored_clock_to_utc.py`) | Rejected: files are hardlinked to the archive (E9); float round-trip and text-splicing risk; the corpus assembler would silently revert it on rebuild. |
| C | Shift in the merge (Stage 4) | Rejected: QC day/night strata, incomplete-day grouping and removal-log periods stay on the wrong clock (E7). |
| D | Exclude affected sites | Fallback only: loses ISR_YAT_YAT (1,239 tree rows, a dry-end site) and others whose offsets are unambiguous. |
| E | Detect and correct automatically at run time | Rejected: an estimator fed by a dead sensor (USA_PAR_FER) or night readings (FRA_PUE) would silently shift a healthy clock. Detection belongs in a gate; correction in a reviewed registry. |

### P1 — daily env support
| | Approach | Verdict |
|---|---|---|
| A | **Own-hours statistics; radiation clearness-scaled with ≥ 80 % TOA-energy coverage; continuous columns ≥ 20 of 24 hours** | **Recommended.** 101.1 % of rows (E4); sw_in error 1–3 % median (E5); ta mean 0.15 K random gaps. |
| B | Own-hours, strict: radiation nightfilled then 24/24; continuous 24/24 | 95.3 % of rows; exact values; mirrors `complete_day_statistics`. The cost is 4.7 % of rows. |
| C | Keep sap∩env hours, add nightfill only | Rejected: fixes E1's 8 sites but not partial days (E3) or daytime-gap bias (E5: 4–26 %). |
| D | Train on ERA5-Land `sw_in` (and ta/vpd) instead of site sensors, as for precip | Out of scope: removes the train/serve skew entirely but is a modelling change for the user. |

### P2 — ext_rad
| | Approach | Verdict |
|---|---|---|
| A | **Daily ext_rad from geometry via the served function `ext_rad_daily`** | **Recommended.** Identical to serving by construction; no partial days; the shipped column is simply not read. Reverses the 09-14 "leave ext_rad as it is" for the TRAINED feature (data untouched) → §8 Q1. |
| B | Keep shipped ext_rad, apply the new completeness rule | Displaced columns stay at 17 sites; partial days become NaN, costing rows for a quantity that is exactly computable. |
| C | Overwrite the corpus ext_rad | Rejected: E9 and the 09-14 data decision. |

---

## 3. Design

### WP-A — clock offsets

**A1 `src/clock_offsets/registry.py` (new).**
- `@dataclass(frozen=True) ClockOffset(site, component, offset_hours, status, start=None, end=None, evidence="")`
  - `component` ∈ {`env`, `sap`}; `status` ∈ {`applied`, `acknowledged`}
  - `start` / `end` are optional UTC LABEL times, half-open.
- `REGISTRY: tuple[ClockOffset, ...]`, with the sign convention and the 09-14 sign trap stated in the module docstring.
- `offsets_for(site, component)` returns only `applied` entries.
- `validate_registry(registry, corpus_dir)` returns a list of errors, and the loaders raise when it is non-empty:
  - component and status must be in their vocabularies;
  - |offset| ≤ 14 h, and the offset is a multiple of 0.25 h;
  - `(site, component)` periods must not overlap, and `start < end`;
  - the site's `{site}_env_data.csv` or `{site}_sapf_data.csv` must exist under the corpus by EXACT filename. A glob would over-match: `CHE_DAV` ⊂ `CHE_DAV_SEE`.
- **Why Python rather than CSV:** the repo's blanket `*.csv` ignore kept `removal_log.csv` untracked and
  absent on Palma for months. A module is tracked, typed and reviewed in diffs.

**A2 `src/clock_offsets/apply.py` (new).**
- `site_code(path)` returns the full site code by stripping an exact suffix (`_env_data.csv`, `_sapf_data.csv`, `_env_flags.csv`, `_sapf_flags.csv`), never by token count (E12).
- `correct_clock(frame, entries, time_cols=("TIMESTAMP", "solar_TIMESTAMP"))` returns a NEW frame.
  - Every row whose label falls in an entry's period has every listed time column (index or column) moved by `−offset`.
  - Time zone and name are preserved, and NaT stays NaT.
  - The input is never mutated.

**A3 Wiring — the only two call sites.**
- `EnvironmentalAnalyzer._load_environmental_data`: right after reading, `df` and `flags` both pass through
  `correct_clock(..., offsets_for(site, "env"))` before anything else. This load has no removal log.
- `SapFlowAnalyzer._load_sapflow_data`, in this order:
  1. read data and flags;
  2. apply the removal log (its periods are LABEL time, E10);
  3. `correct_clock` on data and flags with `offsets_for(site, "sap")`;
  4. compute `TIMESTAMP_LOCAL` from the corrected `TIMESTAMP` (the line moves below the removal log);
  5. process.
- No downstream function takes an offset (merge, FE, serving). The merge reads corrected `outliers_removed`.
  A test asserts the merge module never imports `src.clock_offsets.apply`.
- Both loaders call `validate_registry` once per run against the corpus directory and raise on any error.

**A4 `src/clock_offsets/detect.py` (new).** Estimators ported from `probes/radiation/probe_cloud_vs_clock.py`:
- `clear_day_lag(sw_in, toa)`: cross-correlation on the clearest quarter of days, parabolic refinement.
- `edge_shift(sw_in, toa)`: daylight-window centre on a 24 h circle; threshold 1 % of p99.
- `pair_lag(sap, sw_in)`: sap against sw_in, compared with the corpus control band.
- `reference_lag(sw_in, era5_ssrd)`: against ERA5-Land hourly `surface_solar_radiation_downwards_hourly`,
  zero-point calibrated on the on-time population so ERA5's accumulation convention cancels.

Each returns a value plus a `reason` when it cannot measure (too short, dead sensor kt < 0.05, night readings, polar day).

**A5 Gate `python -m src.clock_offsets.gate` (new).** Runs over Stage-3 `outliers_removed` and writes a JSON report. Exit 1 on any ERROR.
- **ERROR** when a site outside the registry has clear-day lag and edge shift that agree within 0.5 h and both reach |x| ≥ 1.0 h.
- **ERROR** when an `applied` env entry still measures |x| ≥ 0.5 h after correction. This catches a missed application, a double application, and a sign error.
- **WARN** for single-method findings, `acknowledged` entries, and sap pair-lags outside the control band.

**A6 Registry content — MEASURED 2026-09-16 (T5 probes 46760679, 46760722, 46760762, 46760767).** The table
below is the plan's original target; `src/clock_offsets/registry.py` now holds what the probes measured, and
the user reviews it before WP-C. What changed against this table:

- **ISR_YAT_YAT has three clocks, not one.** Right until the logger stopped on 2009-09-18; −5 h from the
  restart (no reading 09-19..09-29); −6 h after a reset in the night of 2009-11-23 (that day's daylight still
  fits −5 h; the next sunrise is already at −6 h). Two `applied` period entries.
- **FIN_PET env = +3.5 h** (TOA correlation +3.44, ERA5-Land +3.46, morning crossing +3.44/+3.47). Its
  daylight edges (+2.75) are not trustworthy at 69.5° N. Its **sap is `acknowledged`, not applied**: radiation
  references say +1.4..+2.5 h, vpd/ta references +3.4..+4.2 h, and the site's own vpd peaks 0.2 h after its
  sw_in against 1.9–2.3 h everywhere else, so one of its loggers is on a different clock (deviations D11, D12).
- **SWE_SVA_MIX_NON sap = −6.0 h**, calibrated on the 16 boreal controls, where the radiation and thermal
  references agree to 0.02 h (−5.93 / −5.95 / −6.10). The global zero gives −4.8 and leaves them 0.46 h apart.
  The boreal zero is not one network's convention: Finnish, Russian and Swedish controls all lie at 1.6–3.1 h.
- **USA_HUY_LIN_NON +5.0 h env** and **USA_ORN_ST2_AMB +5.0 h env and sap** are confirmed; the ORN twin
  difference is exactly 5.00 h on all five env estimators, which no site convention can fake.
- The **1–1.5 h group is `acknowledged`** with its measured values (RUS_CHE_Y4, USA_SYL_HL2, ESP_LAS, ITA_TOR
  env and sap, ESP_ALT_ARM, ESP_ALT_HUE, ESP_LAH_COM env and sap, GBR_GUI_ST1/2/3, GBR_DEV_CON/DRO,
  USA_HIL_HF1_POS/HF2, CHN_YUN_YUN). AUS_BRI_BRI, CHN_HOR_AFF and ESP_RIN carry no usable radiation at all and
  get no entry.

| site | component | offset h | tier | evidence / open point |
|---|---|---|---|---|
| ISR_YAT_YAT | env | −6.0 | 1 | −5.91 / −5.91 / −5.55 / −6.00; 20+ of 24 trees sap on time. **2009 measured −5.25** (09-14): T5 checks per year → may need a period entry |
| USA_HUY_LIN_NON | env | +5.0 | 1 | +4.97 / +5.00 / +4.98 / +5.00; both trees sap on time |
| USA_ORN_ST2_AMB | env + sap | +5.0 | 1 | +4.95 / +5.08 / +5.31 / +4.50; twin ST1 difference 5.00 by both methods; all 10 trees shifted |
| FIN_PET | env + sap | +3.0 or +3.5 | 1 | +3.60 / +3.38 / +3.62 / +2.75; EEST-stored-as-UTC predicts 3.0 → T5 pins |
| SWE_SVA_MIX_NON | sap | ≈ −5 (±1) | 1 | sw_in on time; 20 trees sap 4.5–6 h early vs corpus → T5 pins with pair lag |
| RUS_CHE_Y4 | ? | +1.25 | 2 | both methods, summer = winter; component unknown |
| USA_SYL_HL2 | ? | −1.25 | 2 | both methods |
| ESP_LAS | ? | −1.0 | 2 | both methods (also ext_rad-displaced) |
| ITA_TOR | ? | ≈ −0.75 | 2 | both methods; 12 removal-log rows |
| ESP_ALT_ARM / ESP_ALT_HUE | ? | +1.0 | 2 | edges +1.00; clear-day correlation +1.3 / +1.5, cloudy +0.75 / +0.8 |
| ESP_LAH_COM | ? | −1.0 | 2 | both methods (new 09-16) |
| GBR_GUI_ST1 / ST2 / ST3 | ? | +1.0 | 2 | both methods (new 09-16) |
| GBR_DEV_CON/DRO, ESP_RIN, CHN_HOR_AFF, USA_HIL_*, CHN_YUN_YUN, AUS_BRI_BRI | — | — | 3 | `acknowledged`: summer-only records, correlation-only, units, or too short |

### WP-B — daily env statistics from env's own hours

**B1 `env_daily_statistics(env_hourly, lat, lon, keep_hours=None)` in the merge (new).** Input is the env
frame from `aggregate_env_to_hourly` BEFORE the sap join. Output is one row per UTC day with the bare name
(mean), `_min`, `_max`, `_sum` and `{col}_hours`; radiation also gets `{col}_toa_coverage`.
- **Continuous columns** (`ta`, `vpd`, `ws`, `rh`, `netrad`): valid when present hours ≥ `MIN_ENV_HOURS`
  (20). mean / min / max are taken over present hours; `_sum` = mean × expected hours. Otherwise all four are NaN.
- **Radiation** (`sw_in`, `ppfd_in`):
  - A missing hour is a known zero when computed TOA is 0 across [h − 30 min, h + 59 min]. The margin covers end-of-interval labels.
  - mean = Σ present ÷ Σ TOA(present) × mean TOA(day).
  - Valid when Σ TOA(present) ÷ Σ TOA(day) ≥ `MIN_TOA_COVERAGE` (0.8). A polar-night day is 0.0.
  - `_min` / `_max` come from the filled hours and follow the continuous rule.
  - Measured night values, including small negative sensor offsets, are kept: only NaN is filled.
- **Under `--daytime-only`** (`keep_hours`), expected hours are the kept hours and the thresholds scale with them, as `complete_day_statistics` already does.
- Applied through `overwrite_daily_columns(df_daily, ...)` after the ERA5 and gauge overwrites, so a
  plant-level tree broadcasts like ERA5.

**B2 ext_rad.** `ext_rad` = `src.serving.era5_daily.ext_rad_daily(days, lat, lon)`; `_min` / `_max` over the same
48 instants; `_sum` = 24 × mean. The shipped hourly column is not read for the daily tree. The hourly tree is unchanged (out of scope).

**B3 Invariants, enforced at write time.**
- Support: wherever a column value is finite, `{col}_hours` meets the rule. A violation raises, because it is a programming error.
- kt: per-site count of days with `sw_in / ext_rad > 1.0`, written to the provenance block. The gate job fails if
  the corpus total exceeds the 221 baseline, or if a site outside the documented residual list gains any such day.

**B4 Provenance.** The servable contract block gains:
- `env_daily_rule: {"version": 1, "min_hours": 20, "min_toa_coverage": 0.8, "ext_rad": "geometry"}`
- `clock_registry_sha256`

`assert_dataset_contract` stays informational. New sweep job scripts assert `version ≥ 1`, so an old tree cannot be mistaken for a fixed one.

### WP-C — gates, rebuild, comparison
1. Merge `feat/radiation-clock` into `feat/sapwood-combined` in the main clone only after 46747902 has finished
   (or is confirmed to import no changed module), and when no pending serving job imports a changed module.
2. Back up the `outliers_removed` files of registry sites, then re-run Stage 3 for them with explicit file lists.
   `SapFlowAnalyzer.run_analysis_in_batches` hardcodes one site, so call `_load_sapflow_data(files_to_load=...)`; follow the `job_qc_hohfix.sh` pattern.
3. Clock gate over all sites.
4. Merge a full servable tree under a new sweep tag, re-run `probe_support_rules.py` and `probe_knockout.py`
   on it, then cache → FFS → train.
5. Compare with `sweep_20260916_svmerge`: kt>1, per-site feature bias, rows, pooled R2, median NSE, and residuals at registry sites.

---

## 4. Failure modes (Aspect 3)

| # | What could go wrong | Mitigation / test |
|---|---|---|
| F1 | Offset applied twice (loader + another path) | Single call site per loader; merge never imports `apply` (test); gate ERROR if an `applied` site still measures ≥ 0.5 h |
| F2 | Sign error | Synthetic series with a known +5 h label delay: detector → +5, after `correct_clock` → 0 (test) |
| F3 | Registry key misses (spelling, prefix twin) | Exact-suffix site codes (E12); validator fails on entries with no corpus file; ST1 unchanged when ST2 corrected (test) |
| F4 | Wrong component (env vs sap) | Per-component entries; pair-lag check in gate; ISR / HUY env-only, SWE_SVA sap-only fixtures |
| F5 | Removal-log periods written in label time get shifted | Removal log BEFORE `correct_clock` in the sap loader; test that a period row removes the same readings with and without a registry entry |
| F6 | Flags drift from data | Flags pass through the same `correct_clock`; mask alignment test |
| F7 | `TIMESTAMP_LOCAL` / incomplete-day grouping on the old clock | `TIMESTAMP_LOCAL` recomputed after correction; `solar_TIMESTAMP` shifted with `TIMESTAMP`; tests |
| F8 | Offset changes within a record (ISR 2009) | Optional periods; validator refuses overlap; gate measures per year |
| F9 | Env-only shift misaligns env and sap rows | The merge aggregates each side to the hour before joining (already so); quarter-hour shifts land in hour bins correctly (test) |
| F10 | Gate false positives (FRA_PUE night readings, USA_PAR_FER dead sensor, polar day, short records) | Two methods must agree for ERROR; `reason` codes; fixtures for each |
| F11 | Nightfill runs on a wrong clock | Merge reads corrected Stage-3 output; registry sites are re-run in Stage 3 before the merge (WP-C order) |
| F12 | Dawn/dusk hour wrongly treated as dark | Dark only when TOA is 0 across [h − 30, h + 59] min; sunrise-at-:50 test |
| F13 | Clearness scaling biased when gaps coincide with cloud | Coverage ≥ 0.8 bounds it; measured 1–3 % median (E5); knockout re-run on the rebuilt tree |
| F14 | Polar night / polar day | 0.0 when every hour is dark; scaling only when TOA > 0; tests at 69.5° N |
| F15 | Row loss hurts small sites | Measured +1.1 % net; per-site losses are days that were biased (E4); reported per site in provenance |
| F16 | New columns (`{col}_hours`, `_toa_coverage`) enter FFS as candidates | Registry-driven loader; test that the FFS candidate set is unchanged |
| F17 | `ext_rad_daily` import drags serving dependencies into the merge | Only `src.env_derive` + `src.feature_contract` (already merge dependencies); import test |
| F18 | In-flight chain or serving jobs import changed code | Worktree + branch; merge to main clone after the jobs (WP-C.1); changed files limited to the analyzers, merge, and new `src/clock_offsets` |
| F19 | Linter hook rewrites meaning; Python 3.9 on Palma | Install patches via Bash, verify with grep + `py_compile` after every write; no `zip(strict=)`, no `match`, `from __future__ import annotations` |
| F20 | Stage-3 re-run touches shared outputs used elsewhere | Only registry sites; files backed up first; nothing in flight reads `outliers_removed` |
| F21 | An old tree is trained as if fixed | `env_daily_rule.version` recorded; sweep scripts assert it |

---

## 5. Dependency chain (Aspect 4)

```
T0 worktree + branch
├── T1 registry + validator            (tests first)          ┐
├── T2 detect estimators               (tests first)          │ parallel
├── T5 evidence probe: ERA5-referenced lag, per-year, pair lag │ (data only)
└── T6 env_daily_statistics            (tests first)          ┘
T3 loader wiring                        ← T1
T7 ext_rad geometry                     ← decision Q1
T8 invariants + provenance              ← T6, T7
T4 registry values                      ← T5 + user review
T9 gate job: new tests + job_pytest_merge_contract.sh + analyzer tests + clock gate on fixtures
── merge to main clone after 46747902 (WP-C.1)
C2 Stage-3 re-run (registry sites) → C3 clock gate → C4 merge tree → probes → cache → FFS → train → C5 compare
```
No new external dependency: pvlib, pandas, numpy, and the ERA5-Land site extract already on disk.

---

## 6. Test strategy (Aspect 5) — written before implementation

**`src/clock_offsets/tests/`**
- `test_registry.py`
  - A valid registry passes.
  - The validator refuses: an unknown component or status; |offset| > 14; an offset that is not a multiple of 0.25; overlapping periods; `start ≥ end`; a site with no corpus file.
  - A prefix twin does not match.
  - `offsets_for` returns `applied` entries only.
- `test_apply.py`
  - `TIMESTAMP` and `solar_TIMESTAMP` move by −offset, whether the timestamp is the index or a column.
  - A period entry moves only rows inside its period.
  - tz and name are preserved; NaT stays NaT; the input is not mutated; an empty entry list returns an equal frame.
  - `site_code` handles both ORN twins and `CHE_DAV` / `CHE_DAV_SEE`.
- `test_detect.py`
  - Synthetic regimes (baseline, random cloud, thick noon cloud, afternoon storms, morning fog) at offsets 0 and 3: clear-day lag and edge shift each within ±0.25 h.
  - All-day correlation under afternoon storms is documented as biased; the regression guard shows why the gate uses clear days.
  - Night readings, dead sensor, and short record each return a `reason`, not a number.
- `test_gate.py`
  - ERROR for an unregistered +5 h site.
  - ERROR for an `applied` site whose correction was skipped, and for one corrected twice.
  - WARN for `acknowledged` entries and for single-method findings.
  - Exit codes.

**`src/Analyzers/tests/test_clock_offset_loading.py`** (tiny corpus fixture with ORN-like twins)
- The env loader corrects ST2 and leaves ST1 untouched.
- The sap loader applies the removal log in label time before the shift.
- Flags stay aligned with data.
- `TIMESTAMP_LOCAL` follows the corrected clock.
- The QC day mask peaks at the corrected noon.
- The validator error propagates.
- Sites not in the registry come out byte-identical.

**`notebooks/test_env_daily_support.py`**
- A complete day equals today's mean / min / max.
- Night-NaN `sw_in` with full daylight equals the exact 24 h mean, not the daylight mean.
- Two missing daylight hours with uniform kt: the value is within 1 % of truth. Coverage < 0.8 gives NaN.
- Dark-margin boundary: sunrise at :50 is not dark.
- A polar-night day is 0.0; a polar day is scaled only.
- A continuous column with 19 hours is NaN; with 20 it is the mean of 20.
- Sap gaps do not change the env value (the POL_TUC case).
- Under `keep_hours`, expected hours equal kept hours.
- Plant-level rows broadcast.
- Negative measured night values are kept.
- Input frames are not mutated.

**`notebooks/test_ext_rad_daily_geometry.py`**
- The merge's daily `ext_rad` equals `ext_rad_daily` on every row, partial days included (max |Δ| ≤ 1e-9).
- A displaced shipped column has no effect.
- `_min` / `_max` / `_sum` definitions hold.

**`notebooks/test_radiation_invariants.py`**
- The support invariant raises on a forged violation.
- kt>1 counts are written to provenance.
- The gate fails when the total exceeds the baseline or a new site appears.
- The provenance block carries `env_daily_rule` and `clock_registry_sha256`.
- The FFS candidate set is unchanged by the new columns.

**R3 integration**
- End-to-end merge on three fixture sites: night-NaN, clock-corrected, and clean. Assert on daily CSV values.
- Verification job on real data: Stage 3 for two registry sites, a merge into a scratch directory, then both probes and the gate. The acceptance numbers come from this job.

---

## 7. Acceptance criteria (Aspect 6)

- **AC1** `validate_registry` raises for each of: unknown component, |offset| > 14 h, overlapping periods, an entry whose site has no corpus file (4 tests red → green).
- **AC2** After the Stage-3 re-run, the clock gate measures |offset| ≤ 0.5 h by clear-day lag AND edge shift at every `applied` env entry, for every year with ≥ 30 usable days.
- **AC3** For every `applied` sap entry, the sap-vs-sw_in pair lag lies inside the corpus control band (09-16 sign, p10..p90 of the 158 controls).
- **AC4** For ≥ 5 control sites not in the registry, the Stage-3 `outliers_removed` `TIMESTAMP` column is byte-identical (md5) before and after.
- **AC5** Removal-log rows at ITA_TOR and USA_HIL_HF1_POS null the same raw readings (count and label instants) as before the change.
- **AC6** Rebuilt servable tree: kt>1 site-days ≤ 221 (from 2,141), and 0 at USA_HIL_HF1_PRE / HF1_POS / HF2, USA_UMB_CON, USA_TNO, USA_INM, THA_KHU.
- **AC7** Rebuilt tree: trained `sw_in` median bias vs the clearness-scaled reference within ±1 % at every site (probe_support_rules re-run).
- **AC8** Rebuilt tree: daily `ext_rad` equals `ext_rad_daily` on 100 % of rows (max |Δ| ≤ 1e-9 W m-2).
- **AC9** Rebuilt tree: every finite env daily value satisfies its rule. Checked by scanning `{col}_hours` and `{col}_toa_coverage` over the whole tree: 0 violations.
- **AC10** Rebuilt tree: rows with all core env features valid ≥ 98 % of today's 169,224, clock-corrected sites included.
- **AC11** All new tests pass. `job_pytest_merge_contract.sh` passes with nothing deselected; `src/Analyzers/tests` pass; 0 INTERNALERROR. New code ≥ 80 % line coverage.
- **AC12** The tree's provenance block carries `env_daily_rule.version == 1` and `clock_registry_sha256`.
- **AC13** No file imported by 46747902 or a pending serving job is modified in the main clone before those jobs start or finish (commit time vs `sacct` start times).

---

## 8. Decisions (user, 2026-09-16)

1. **ext_rad (P2): geometry.** Train on `ext_rad_daily`, the served definition. This reverses the 09-14 "leave ext_rad
   as it is" for the trained feature only; corpus data stays untouched.
2. **Support rule (P1): A, "but ignore netrad".** Radiation is clearness-scaled with ≥ 0.8 TOA coverage; `ta`, `vpd`,
   `rh` and `ws` need ≥ 20 h. `netrad` is left out of WP-B and keeps today's statistics, because no feature,
   registry entry, contract or serving path reads it (deviation D5).
3. **Clock scope (P3): big offsets first.** Tier 1 is applied after T5 pins the values (FIN_PET 3 vs 3.5, ISR 2009).
   The 1–1.25 h group gets the same check; unresolved sites are recorded as `acknowledged`, not corrected.
4. **In-flight chain: let it finish.** 46747901/46747902 become the "before" arm.

## 9. Rollback
- An empty `REGISTRY` makes both loaders a no-op (AC4 proves byte identity).
- WP-B lands as one commit, reverted with `git revert`. The pre-fix tree `sweep_20260916_svmerge` stays on disk as the baseline.
- Backed-up `outliers_removed` files of registry sites are restored by copy.

## 10. Evidence log (filled as phases close)

**Phase 3 (TDD) — GREEN 2026-09-17.** `src/clock_offsets/tests` 123 passed (46761404); the merge-contract gate
584 new + 994 regression passed with nothing deselected (46760892); `/verify` PASS (46761137): py_compile on
3.9.25, no duplicate definitions, no `zip(strict=)` or `match`, no secrets, every authored line ≤ 120 chars,
and **98 % line coverage of the two new packages** (1,759 statements, 42 missed; lowest module 89 %).

**Phase 4 — the clock gate over the real, still-uncorrected tree (46761146, 13 min for 225 sites).**
- 143 sites measured, **106 controls**, zero points clear-day +0.13 h, edges +0.25 h.
- **3 ERRORs, every one of them a registry `applied` site** — ISR_YAT_YAT −6.14 h, USA_HUY_LIN_NON +4.79 h,
  USA_ORN_ST2_AMB +4.66 h — and **no unregistered site reached ERROR**: zero false positives on 225 real sites.
- The twin guard holds on real data: USA_ORN_**ST1** measures −0.17 / −0.50 / −0.30 h and stays silent while
  its twin errors at +4.66 h.
- The two-witness rule earns its keep at FRA_PUE (clear-day +0.13 h, edges +2.75 h — the known night-reading
  site): a WARN, never an ERROR.
- The run found three defects in the gate itself, fixed under deviation D18 (tests first, RED 4 → GREEN 123).
- 82 of 225 sites cannot be measured at all: 26 carry no `sw_in`, the rest have too few clear days (< 40) or no
  readable daylight window (< 30 days). They produce no finding and their reason is in the JSON report.

**AC5 is satisfied vacuously, and that is worth writing down.** `removal_log.csv` has **no rows at any site
with an `applied` entry** (ISR_YAT_YAT, USA_HUY_LIN_NON, USA_ORN_ST2_AMB, FIN_PET, SWE_SVA_MIX_NON), so no
correction can move a reviewed removal period today. The order the loaders use — removal log first, then the
clock — is therefore inert for now, and is the right default for the day an entry lands on a logged site.

**Phase 4 — the same gate after D18 (46761408).** ISR_YAT_YAT now reads *"the registry applied 2 period
corrections (−5.00 h, −6.00 h)"* instead of their phantom sum, and FIN_PET, silent before, now carries a WARN:
*"the registry applied +3.50 h, and the one estimator that can read it still measures +2.75 h (clear-day none,
edges +2.75 h); one witness is not two … 22 clear days, fewer than 40."* 3 errors, 34 warnings, same 143/225.

**Predicted residual at FIN_PET after the Stage-3 re-run, so it is not read as a failure.** Its edges are the
one estimator that can measure it and they say +2.75 h, while the three estimators the entry rests on (TOA
correlation +3.44, ERA5-Land +3.46, morning crossing +3.44/+3.47) say +3.5. After the correction the edge will
therefore read about **−0.75 h**, which is above the 0.5 h threshold and will keep FIN_PET on the WARN list,
alone and by one witness. That is the documented edge bias at 69.5° N (plan §3 A6, deviation D11), not a
correction that missed. **AC2 cannot be evaluated at FIN_PET**: it asks for clear-day AND edge agreement, and
the clear-day estimator has 22 of the 40 days it needs there.

**Phase 5 close — the last three jobs before the commit (2026-09-17).** `/verify` PASS at 284 tests and 99 %
coverage (46765364); the radfix suites 416 passed (46765363); the repo gate 614 new + 994 regression with
nothing deselected (46765642). The +1 over the previous counts is `test_a_thin_control_population_is_said_out_loud`
(deviation D21).

**One defect found by that last run, in the gate script itself (deviation D22).** Submitted bare, the repo gate
ran against the default `$ROOT` — the main clone, which does not carry this branch — collected nothing, and
pytest returned 4. The script printed `RESULT: FAIL` and then exited 0, so SLURM recorded `COMPLETED`: a green
job over a run that tested none of the new code. `job_pytest_merge_contract.sh` now refuses up front a `$ROOT`
missing any path it is about to test, and exits 1 on its own FAIL verdict. Proved both ways: 46765641 (bare)
fails in under a second naming the four missing paths, 46765642 (`GSV_ROOT` set) passes.

**What this commit does not do.** It does not rebuild the data. Stage 3 still holds uncorrected timestamps, so
the clock gate exits 1 over the tree on disk today (46762193: 3 errors, all of them registry sites). That run
is the "before" arm; WP-C is the rebuild.
