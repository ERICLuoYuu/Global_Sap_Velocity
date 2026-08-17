# Stage-3 QC over a corpus with three different flag mechanisms

**Status:** DRAFT — awaiting review. No code written yet.
**Date:** 2026-08-16
**Input corpus:** `data/raw/0.1.5/0.1.5/csv/sapwood_combined/` (226 sites, delivered 2026-08-16)

---

## 1. What the corpus actually looks like

Measured, not assumed. Sources: `MANIFEST.json` on Palma, headers of all 226
`*_env_data.csv`, and a token census over a 12-site sample of the external
`sapf_flags` / `env_flags`.

| population | n | `sapf_flags` | `env_flags` | in-file `*_qc` cols |
|---|---:|:--:|:--:|:--:|
| SAPFLUXNET external | 202 | yes | yes | — |
| internal, ICOS-derived env | 19 | no | no | yes (6) |
| internal, non-ICOS env | 4 | no | no | no |
| internal, **empty** env | 1 | no | no | no |

- The flag-file sets are **identical** for sap and env: a site has both files or
  neither (`comm -3` over the two listings is empty). So "no flags" is one
  population of 24, not two overlapping ones.
- The 24 are exactly the internal sites (25 internal minus `ESP_GDN`, skipped
  at build time for having no usable sapwood area).
- The 4 with variables but no QC: `ESP_LMA`, `POL_MEZ`, `POL_TUC`, `SWE_SGR`.
- The 1 empty: **`SWE_NOR`** — its `env_data.csv` header is
  `TIMESTAMP,solar_TIMESTAMP` and nothing else.
- The 19 ICOS sites all carry the same six columns:
  `ta_qc, vpd_qc, sw_in_qc, pa_qc, precip_qc, ws_qc`, written by
  `Sapflow-internal/icos_data_extractor.py::QC_MAPPING`.

### The SAPFLUXNET flag vocabulary is richer than the code that reads it

Token census (12 sites, both kinds). Tokens are `;`-joined composites:

| token | sapf count | env count | meaning |
|---|---:|---:|---|
| `NA_PRESENT` | 8,640,207 | 330,440 | already NA in source |
| `CALCULATED` | — | 909,897 | value derived, not measured |
| `NA_ADDED` | 523,367 | 625,434 | NA inserted by SAPFLUXNET |
| `OUT_WARN` | 308,117 | 199,194 | outlier warning, **value retained** |
| `MANUAL_REMOVED` | 52,308 | 816 | already removed |
| `RANGE_WARN` | 7,610 | 8,428 | range warning, **value retained** |
| `RANGE_REMOVED` / `OUT_REMOVED` | small | small | already removed |

Both analyzers match **`RANGE_WARN` only**, and that is **correct and
deliberate** (§6 D2). Of the seven tokens only two sit on a value that is still
present — `RANGE_WARN` and `OUT_WARN` — and the two mean different things:

- `RANGE_WARN` marks a **physically implausible** value. Nothing else in the
  pipeline performs a range check, so this token contributes information no
  other step supplies.
- `OUT_WARN` marks a **statistical outlier** under SAPFLUXNET's own detector.
  This pipeline already runs its own outlier detection — rolling z-score, 3σ by
  day and 5σ by night over a 30-day window, in both analyzers. Acting on
  `OUT_WARN` would apply two different outlier definitions in sequence to the
  same series, with the archive's definition silently taking precedence over
  the one this project chose and tuned.

`OUT_WARN` is therefore **never applied**, at any site, under any
configuration. It is not a gap to close later. The remaining five tokens
(`NA_PRESENT`, `NA_ADDED`, `CALCULATED`, `*_REMOVED`) describe cells that are
already NA or already removed, so acting on them would be a no-op.

### What 0 / 1 / 2 mean in the ICOS QC columns

You said you don't know; I could not confirm it from documentation either.
Three attempts failed: fluxnet.org returned 403, the Pastorello 2020 paper on
nature.com is behind an auth redirect, and the ONEFlux README does not define
the levels. **So this plan does not rest on the official wording.** What the
data itself shows, over all 1,184,579 flagged rows:

- **`0`** — 72–77% of rows. Full-resolution signal: `sd(ta)` 6.5–9.4 K,
  median `|Δta|` between consecutive steps 0.18–0.50 K.
- **`1`** — 0.1–1.3%. Value **present** but median `|Δta|` is
  **0.013–0.15 K, i.e. 10–20x smoother** than level 0 at the same site. That is
  the signature of an interpolated/look-up fill, not of a measurement.
- **`2`** — 1.4–7.4%. At 7 of 13 sites the variable is **entirely NaN** where
  the flag is 2 (`sd = nan`); at `CHE_DAV` and `FRA_HES` it is present and
  mildly smoothed (`|Δta|` 0.21 vs 0.30 measured).
- `pa_qc`, `precip_qc`, `ws_qc` **never take the value 1** — only 0 and 2.
- No level 3 anywhere. No missing QC values anywhere (`qc_nan = 0`).

Monotone in quality, 0 best. That is all the engineering needs; the exact
FLUXNET wording is recorded as unconfirmed rather than guessed.

### What each threshold costs

Loss counted only where the value is **present** — masking an already-NaN cell
is a no-op, so this is the real price. Pooled over the ICOS sites:

| variable | present | lose at `qc>=1` | lose at `qc>=2` |
|---|---:|---:|---:|
| `ta` | 1,170,330 | 3.38% | 2.29% |
| `vpd` | 1,162,033 | 4.08% | 2.42% |
| `sw_in` | 1,169,278 | 3.33% | 2.99% |
| `precip` | 1,138,433 | 2.38% | 2.38% |
| `ws` | 1,126,956 | 2.81% | 2.81% |
| `pa` | 1,165,634 | 7.68% | 7.68% |

Worst single case: **`FRA_HES` `pa`, 37.4%** at either threshold. `pa` is not
in the model's dynamic feature list, so that number is loud but harmless.
Worst case on a feature that is used: `CHE_DAV` `vpd`, 8.90% / 7.29%.

### `ppfd_in` and `rh` are purely derived — verified

At all 13 checked ICOS sites, `ppfd_in` is non-null on **exactly** the rows
where `sw_in` is non-null, and `rh` on exactly the rows where `vpd` is
(`without_source = 0` at every site). Neither carries a `_qc` column, because
`icos_data_extractor` synthesises them via `fill_ppfd_in(sw_in)` and
`fill_rh(ta, vpd)`. So masking `sw_in` while keeping `ppfd_in` would retain a
number computed from data we just rejected. Propagation is mandatory, not
optional, and unambiguous: no site has an independent PAR or RH measurement to
protect.

---

## 2. Defects found while surveying (all pre-existing)

| # | severity | defect |
|---|---|---|
| **D-1** | **critical** | `env_analyzer.py:194` calls `print(flags.dtypes)` **unconditionally**, above the `if flags is not None` guard at line 225. Every flagless site raises `AttributeError`, which is swallowed by the bare `except Exception` at line 173 and printed as one line of "Error loading …". **Env QC has never run for any of the 24 internal sites** — it fails before the guard that was supposed to protect it. |
| **D-2** | high | In `sap_analyzer.py`, the negative-value clamp (`df.loc[df[col] < 0, col] = np.nan`, line 565) and the `filtered/` export (line 573) live **inside** `if flags is not None`. A flagless site keeps its negative sap flow and never produces a `filtered/` file. The cleaning that has nothing to do with flags is gated on flags. |
| **D-3** | high | `*_qc` columns are not excluded from the env variable set. `numeric_cols` (line 207) and `data_cols` (line 230) both include `ta_qc` etc., so the flags are z-scored, outlier-filtered and daily-resampled **as if they were environmental variables**. |
| **D-4** | medium | `env_analyzer.py:143` writes `flags['solar_TIMESTAMP']` unconditionally — `KeyError` on any flag file lacking that column (sap_analyzer guards this at line 444; env does not). |
| **D-5** | **medium** *(regraded from "low, latent" — see below)* | Flag columns are matched by substring, then the first hit is taken: `[fcol for fcol in flags.columns if col in fcol][0]`. Correct only if no column name is a substring of another that precedes it. **The hazardous ordering already exists in the delivered corpus.** `SWE_NOR_sapf_data.csv` lists `SWE_NOR_Psy_Jt_10`, `_Jt_11`, `_Jt_12` **before** `_Jt_1`, so `col = SWE_NOR_Psy_Jt_1` resolves to `SWE_NOR_Psy_Jt_10` — a different tree. The only thing preventing a silent mis-application is that `SWE_NOR` ships no flag file. |
| **D-6** | low | `env_analyzer.py:252` writes a `*_flagged.csv` even when `warn_mask` is empty (the write sits outside the `if warn_mask.any()` that guards only the print). |

D-1 is the one that matters: it means the current answer to "does QC handle
flagless sites?" is not "it skips the step" but "it silently drops the site".

### Correction to my own D-5 evidence (2026-08-16)

This row first read *"low, latent — 63 of 165 external sap sites have 128
ambiguous columns, but in 0 cases does `[0]` pick the wrong one; one reorder
away from being a silent bug."* Two things were wrong with that.

1. **The census covered the wrong population.** It measured the 165 external
   *sapwood* files only. `SWE_NOR` is internal and was never in it — and it is
   precisely where the bad ordering turns up. The whole reason this piece of
   work exists is that the internal sites differ from the external ones, so
   generalising from the external corpus was the one inference not available.
2. **"One reorder away" was too comfortable.** No reorder is required. The
   ordering that breaks it is already on disk; only the absence of a flag file
   hides it. The accurate phrasing is *one flag file away*.

What survives: **no site currently mis-applies flags.** What does not: the
claim that this holds across all 202 flagged sites. The census never touched
the ~36 converted sites sourced from the `plant/` directory. So "0 wrong picks
across the 202" is not a result I hold, and §7 Phase 3 step 0 measures it
before any code is written.

A defect that is inert only because of how an upstream tool happened to order
its headers should be graded on that luck ending, not on the current count of
zero. Hence medium, and hence the fix is exact equality (`fcol == col`), which
removes the question rather than re-measuring it.

---

## 2b. The removal log — measured against the combined corpus

Palma job 46187326, `removal_log.csv` sha256 `8f353c92…` (identical to the local
copy). The loader's `_validate_entries` checks the log against **itself** —
missing fields, `start > end`, unknown type. It never asks whether a row
resolves to a real file, a real column, or an overlapping time range, and
`get_entries_for_site` matches `location` and `plant_type` by exact equality
with no report on a miss. So a row that matches nothing simply does not happen.

**Inventory.** 335 raw rows → 38 carry no location (placeholders) → 405 entries
after semicolon expansion → **371 with a location**, across 40 distinct
`(location, plant_type)` targets.

### A. The log is not under version control, and is absent from Palma

`git ls-files | grep -i removal` returns only `mannual_removal_processor.py`.
`/scratch/tmp/yluo2/gsv/removal_log.csv` did not exist; the validation job
failed on its absence before I uploaded a copy by hand. **Every Stage-3 run on
Palma has applied no manual removals at all** — not because the keys missed,
but because the file was never there. Roughly 300 reviewed decisions live in one
untracked file on one Windows workstation. This is the highest-severity finding
and is independent of naming, keys, and everything else below.

### B. 18 entries name a column that does not exist — three different meanings

| site | n | example | status |
|---|---:|---|---|
| `DEU_HAR` | 10 | `DE_HAR_H10545_1_inner` | **superseded by design.** `sensor_grouping.SITE_SENSOR_LAYOUTS["DE_HAR"]` is `COMBINE_PREFER`, `prefer_order=("outer","inner")`, and its `combine_reason` cites *"removal_log rows 250-259"* verbatim. The decision was promoted into the Stage-1 generator, so the columns are gone because it was honoured. Data correct; log stale. |
| `ITA_CP2` | 1 | `IT_CP2_sapwood_TDSV7_1` | superseded — the phantom column was deleted at source. |
| `DEU_HOH` | 6 | `SF185_20_NW_1` | **LOST.** DE_HoH is `COMBINE_MEAN` over azimuthal replicates, with **no exclusion mechanism**. The six flagged sensors are averaged into their trees' series. |
| `CHE_DAV` | 1 | `CH_DAV_Sap_150N_Avg` | **LOST** — same mechanism. |

So **7 reviewed removals are being silently overridden by the sensor collapse**,
and 11 are stale but harmless. The distinction matters: the DE_Har rows look
identical to the DE_HoH rows in the log, and only reading `sensor_grouping.py`
separates "already done properly" from "quietly discarded".

### C. Structural defects

- 1 **reversed period** — entry 47, `ESP_TIL_MIX_Qil_Js_29`, 2013-02-21 → 2011-10-04.
- 3 rows typed `column_entire` that **also carry dates** — `ESP_GUA` entries
  16–18. The dates are silently ignored and the entire column goes, while the
  note describes a six-week window ("wrong seasonal pattern", 2012-04-09 to
  2012-05-18). Over-removal of three trees.
- 6 `CZE_LIZ_LES` rows with a **blank `removal_type`** — parsed, never actioned.
- 1 genuine duplicate row (`USA_SIL_OAK_1PR_Qve_Js_9`, listed twice).
- 2 columns carrying both a `period` and a `column_entire` row (redundant, harmless).

### D. What is clean

- **0** period entries fall outside their file's time range (154 overlap correctly).
- **0** `(location, plant_type)` key collisions across all 226 files. The
  2-token split is arbitrary but *injective*: `location + "_" + plant_type`
  reconstructs the filename stem exactly, so the key scheme itself is sound.
- 39 of 40 targets resolve to a real file (the 40th is the placeholder artefact).

**Total: 33 of 371 entries inert — but only 7 are lost QC decisions.**

### Correction to my own first pass

The first run reported "31 exact duplicate rows". That was my script keying on
`(site, column, type)` and counting the semicolon expansion — one row carrying
four periods becomes four entries — as four duplicates. Re-keyed on the period
as well, the real number is **1**. The defect was in the measurement, not the log.

## 2c. The log was reformatted — DONE 2026-08-16

Your decision, superseding my earlier "do not reformat". Done on Palma, jobs
46188003 / 46188083, by `reformat_removal_log.py`.

`removal_log.csv`: sha256 `8f353c92…` → **`3c13a380…`**, 335 rows → **366**.
Pre-reformat copy kept as `removal_log.backup_20260816_prereformat.csv`.

| change | count |
|---|---|
| placeholder rows dropped (no location) | 38 |
| rows added by expanding semicolon periods to one row each | +70 |
| exact duplicate dropped (`USA_SIL_OAK_1PR_Qve_Js_9`) | 1 |
| `removal_type` filled in (`CZE_LIZ` ×6) | 6 |
| retyped `period` → `column_entire` (entry 47) | 1 |
| dates cleared on `column_entire` rows (`ESP_GUA` ×3, entry 47) | 4 |
| dates normalised to ISO-8601 | all |

**Not changed:** column names and order, `issue_category` / `notes` / `reviewer`
text, and **`entry_id` is never renumbered** — `sensor_grouping.py`'s
`combine_reason` cites *"removal_log rows 250-259"* by number, so renumbering
would silently break that reference. Verified: ids 250-259 all present, and 0
`(location, plant_type, column)` targets gained or lost.

### The reformat exposed a live loader defect

Removing the semicolons **broke 10 rows**, and the first run proved it:
`period` entries fell 155 → 144. Cause:

`load_removal_log` calls `pd.to_datetime` on the **whole column at once**.
pandas 2.x infers a single format from the first element, so a column mixing
`2011-09-21` with `2016-05-04 12:41:00` coerces the second kind to `NaT`. Those
ten are the `CZE_STN` rows 260-269 — the post-outage duplicate-delivery block
with the most detailed forensic notes in the file.

In the original file they survived **only by accident**: the column also
contained semicolons, so `has_semicolons` was true and the loader skipped the
bulk parse, going value-by-value instead. The semicolons were load-bearing.

Two fixes, both required:

1. **Done.** The reformat emits `start_time`/`end_time` uniformly as
   `%Y-%m-%d %H:%M:%S`. Re-run: 154 of 154 period entries parse and overlap
   their data. (`decision_date` stays date-only — uniform already, parsed in its
   own call, and more readable.)
2. **Phase 3.** Harden `load_removal_log` to parse per-value rather than
   per-column, so a future hand-edit that reintroduces a bare date cannot
   silently NaT the rest. Fix (1) makes the file safe today; only (2) keeps it
   safe. Acceptance criterion 11c.

`★` The general shape: a format that is only correct because of an unrelated
quirk elsewhere in the same field. Tidying the quirk breaks it, and the breakage
is silent — 10 removals would simply have stopped happening.

### Remaining after the reformat

18 entries still cannot act, all for the same reason: the named column no longer
exists after the B1 sensor collapse. 11 honoured-by-generator, **7 lost** —
fixed by the Stage-1 work in §3b, not by editing the log.

## 2d. Sensor-level rows brought onto the corpus's level — DONE 2026-08-16

Decision D8. The corpus has no sensor level, so all 18 sensor-level rows were
wrong *as records*. They were not one case — three groups, opposite treatment
(measured, §3a). `removal_log.csv` sha `3c13a380…` → **`61d9196e…`**, 366 → 352
rows.

**A. Converted, 6 rows → 4 (DEU_HOH).** Each sensor mapped to the tree it was
averaged into. Justified by contamination, not convenience:

| tree | delivered column | probes | flagged | share of rows |
|---|---|---:|---:|---:|
| 082 | `DEU_HOH_Fsy_Js_11` | 2 | 1 | **70.8%** (period 2023 only) |
| 185 | `DEU_HOH_Qpe_Js_18` | 5 | 2 | **63.5%** |
| 194 | `DEU_HOH_Qpe_Js_23` | 4 | 1 | **40.2%** |
| 301 | `DEU_HOH_Qpe_Js_28` | 4 | 2 | **79.4%** |

What survives aggregation is majority-bad in three of four, so dropping the tree
is the honest reading rather than an over-removal. Cost: ~~4 of DEU_HOH's 28
trees, 14% of the site~~ — **corrected 2026-08-17 from the validation run: 3
whole trees plus one tree's 2023 season, so the site ends with 25 trees carrying
data, not 24.** Tree 082's row is a `period` row and keeps its bounds (as the
next paragraph says), so it retains 14,988 of its 19,587 records. The two claims
were inconsistent with each other from the day they were written; only measuring
the output caught it. Each converted row's `notes` records the original sensor
names, the probe count and the share, so the provenance survives the conversion.
The 082 row keeps its `period` bounds — conversion changed the *level*, not the
window.

**B. Deleted, 11 rows (DEU_HAR ×10, ITA_CP2 ×1).** These name sensors whose
decision the generator **already carried out**: `DE_HAR` is `COMBINE_PREFER`
with `prefer_order=("outer","inner")`, so the inner needles are excluded.
Converting them like group A would have deleted **all 10 DEU_HAR trees** — the
whole site — for work already done. `ITA_CP2`'s was a phantom column, never a
tree, so it had no tree to convert to.

Deleting them broke a live reference: `sensor_grouping.py`'s `combine_reason`
cited *"removal_log rows 250-259"* in prose. That citation now dangles, so the
same change **rewrote it to be self-contained** — the ten sensor names, the
date, the reviewer, and why the rows were removed rather than converted. A
provenance pointer into a file that no longer holds the target is worse than no
pointer.

**C. Held, 1 row (CHE_DAV).** Moved to `removal_log.pending_review.csv` with a
blank `removal_type` so it cannot act. Retargeted to `CHE_DAV_Pab_Js_4` and
annotated with why a sensor-level decision is unreachable here: the formatted
file carries 15 tree columns and no sensor level, so the `150N` channel cannot
be excluded at the level we process. Awaits your tree-level review.

**Result: 0 of 352 entries inert** — from 33 of 371 at the start of this
exercise. Every row now names a file that exists, a column that exists, and a
period that overlaps its data.

### Tree identity across the two edits — audited 2026-08-16

"0 inert" says every row now points at something real. It does not say the rows
point at the *same trees* they pointed at before. Those are different claims,
and the second is the one that decides whether a reviewed decision survived.
Audited by diffing the target `(location, plant_type, column)` — not the row,
since the reformat deliberately expanded semicolon periods one-row-per-period.

| | before | after | |
|---|---:|---:|---|
| rows | 335 | 353 | 352 log + 1 pending |
| distinct targets | 295 | 281 | 276 unchanged |
| sites | 32 | 30 | `DEU_HAR` + one blank |

**19 targets lost, 5 gained, and every one of the 24 is accounted for:**

- **38 rows → 1 blank target.** Every field but `entry_id` was empty. Not a
  tree; no decision was carried. (Verified field-by-field, not assumed from the
  empty `column`.)
- **6 `DEU_HOH` sensor targets → 4 tree targets.** The mapping is confirmed
  from `plant_md.pl_name`, independently of how I derived it in §2d:
  `SF082→DEU_HOH_Fsy_Js_11`, `SF185(_NW_1,_NW_2)→Qpe_Js_18`,
  `SF194→Qpe_Js_23`, `SF301(_1,_2)→Qpe_Js_28`. Two sensors collapse onto one
  tree twice, which is why 6 became 4. All four columns exist in
  `DEU_HOH_sapf_data.csv`.
- **1 `CHE_DAV` sensor target → 1 pending target.** `Sap_150N_Avg` → `pl_name`
  150 → `CHE_DAV_Pab_Js_4`, present in the file.
- **11 deleted targets** (`DEU_HAR` ×10, `ITA_CP2` ×1) name columns that are
  genuinely absent. The `DEU_HAR` ten map to `pl_name` H10545/H10546/H10547/
  H10559/H10560/K10012/K10025/K10084/K28/K605 — which is **exactly the site's
  full tree list, all 10, with none left over**. That is the arithmetic behind
  "converting them would have removed the whole site", now measured rather than
  claimed.

**No tree lost a decision, and no tree gained one it was not already named in.**

Seven surviving targets changed their `removal_type`, all by your instruction:
`ESP_TIL_MIX_Qil_Js_29` period → `column_entire`, and six `CZE_LIZ_LES_Pab_Jt_*`
blank → `column_entire`. Same trees, stronger removal.

### Superseded rows are *computed*, not annotated

Criterion 11 originally said the 11 superseded rows would be "marked as such in
the log". Annotating them would be an edit, and a stale one the moment the
generator changes. The validator classifies them instead, from the same source
of truth the generator uses:

> the named column is absent from the file, **but** the site's
> `SITE_SENSOR_LAYOUTS` resolver maps it onto a `pl_code` that *is* present
> → the sensor was collapsed into that tree. Then `combine == COMBINE_PREFER`
> and the column is not the preferred depth → **honoured**; `combine ==
> COMBINE_MEAN` and the sensor is not in `exclude` → **LOST**.

That is exactly the `DEU_HAR`-versus-`DEU_HOH` distinction, derived rather than
asserted, and it stays correct when the layouts change. (Implementation note:
the log spells these with underscores — `DE_HAR_H10545_1_inner` — while the raw
sensor grammar uses hyphens, `{pl_code}-inner`, so the resolver needs the same
separator normalisation `column_naming` already provides.)

### What the log needs from you, not from me

The §2b C defects are mostly questions of intent, and I will not guess at them.
They are reported by the validator and left for you:

| row | question |
|---|---|
| entry 47, `ESP_TIL_MIX_Qil_Js_29` | dates are 2013-02-21 → 2011-10-04. Transposed, or a different range meant? |
| entries 16–18, `ESP_GUA_VAL_Qpy_Jt_13/14/15` | typed `column_entire` but carry 2012-04-09 → 2012-05-18, and the note describes a seasonal window. Whole column, or that window? |
| 6 × `CZE_LIZ_LES_Pab_Jt_1..6` | `removal_type` is blank, so nothing happens. What was intended? |
| `USA_SIL_OAK_1PR_Qve_Js_9` | listed twice, identically. Safe to drop one — the only one I would change unprompted, and only with your nod. |

## 3. Scope

**In:**
1. Separate flag application from the cleaning that does not depend on flags
   (D-2), and stop flagless sites from dying (D-1).
2. A third mechanism: read in-file `*_qc` columns, mask on a configurable
   threshold, propagate to derived variables, and keep the QC columns out of
   the variable set (D-3).
3. Sites with neither mechanism: process normally, skip both flag steps, and
   **say so in a report** rather than in a print statement.
4. `SWE_NOR` (no env variables at all): skipped with an explicit reason.
5. Fix D-4; harden D-5 to exact matching; fix D-6.
6. **A removal-log validator that runs as a gate**, promoting the throwaway
   script behind §2b into `src/qc_flags/` (or beside it): every row must
   resolve to a file, a column, and — for periods — an overlapping range, or
   the run reports it. Silent no-ops stop being possible. The seven **lost**
   removals in §2b B are handled by the Stage-1 work in §3b, not silently.
   **The log file itself is not reformatted — see §2c.**
7. Tests. **Neither analyzer has a single test today** (`src/Analyzers/tests/`
   covers `calibration_corrector`, `per_site_sm_shap`, `scatter_drydown`,
   `sensor_method_vocabulary`, `treatment_filter` — nothing else).

**Out:**
- Any change to the outlier / variability / reversal / incomplete-day steps.
- Stage-4 merge changes.
- Extending the SAPFLUXNET token vocabulary — decision D2 below.
- Re-running the full Stage 3 for production. This plan delivers the code and a
  validation run, not a new canonical dataset.

---

## 3a. The aggregation mismatch — why sensor rows cannot work at Stage 3

The pipeline aggregates **twice**, and the removal log sits between them:

```
raw sensors ──(Stage 1: sensor_grouping, mean/prefer)──▶ tree series
            ──(Stage 3: removal log + QC, per tree column)──▶ cleaned trees
            ──(Stage 4: merge)──▶ site-level sap_velocity
```

A **tree-level** row works: dropping a tree column at Stage 3 correctly removes
it from the Stage-4 site mean. That layer is sound.

A **sensor-level** row cannot work at Stage 3 by construction. The sensor is no
longer a column, so the row matches nothing — and there is no repair available
at that layer, because the two things Stage 3 *could* do are both wrong:

- match the sensor's **tree** and drop it → removes the good sensors too
  (over-removal: tree 185 would lose 3 sound probes to exclude 2 bad ones)
- do nothing → the rejected sensor stays inside the tree mean
  (under-removal: what happens today, silently)

The reviewer inspected **sensor-level** plots for these sites. The pipeline
consumes **tree-level** series. The removal log mixes rows from both levels with
nothing to distinguish them — which is why `DEU_HAR`'s honoured rows and
`DEU_HOH`'s lost rows look identical in the file.

### Measured: what is actually recoverable

Both sites ship **long-format** raw, so `SF185_20_NW_1` is a name the
reorganizer *constructs*, not a column it reads. Whether exclusion is
expressible therefore depends on whether the raw rows still carry sensor
identity — and the two sites differ:

**`DE-HoH` — 6 rows, RECOVERABLE.** The raw carries an explicit `sensor_id`
column whose values match the flagged names exactly. Tree 185, for example:

| `sensor_id` | rows | status |
|---|---:|---|
| `SF185_10_ENE` | 21,947 | keep |
| `SF185_10_W` | 17,376 | keep |
| `SF185_20_NNE_1` | 44,066 | keep |
| `SF185_20_NW_1` | 77,530 | **flagged** |
| `SF185_20_NW_2` | 67,542 | **flagged** |

So the tree does **not** empty — 3 of 5 sensors survive. But the two rejected
probes carry **63.5% of that tree's rows** (145,072 of 228,461), so the
contamination is not marginal. One consequence to handle: the survivors are
2 × 10 mm + 1 × 20 mm, so excluding shifts the tree's radial-depth balance, and
`sensor_grouping`'s own docstring already warns that an unweighted mean over
mixed depths "is then an assumption rather than a definition".

**`CH-Dav` — 1 row, same case as DE-HoH.** Held for a tree-level review rather
than converted, by your decision, so tree 150 can be looked at before it is
dropped.

> **Correction, 2026-08-16.** This paragraph previously claimed CH-Dav was "NOT
> RECOVERABLE by regeneration" because "the provider aggregated the azimuths
> before delivering", and flagged `SITE_SENSOR_LAYOUTS["CH_DAV"]`'s "15 trees,
> 36 columns" as a stale description. **Both were wrong, and the flagged item is
> withdrawn.** I inspected `CH-Dav_2010_2023_SFD_15Trees_Daily.csv` and
> concluded from it; the reorganizer actually reads
> `CH-Dav_2010_2023_DT_15Trees_HH.rds` via `pyreadr` — 2,827,152 rows whose
> `TreeN` column holds exactly **36 distinct values**, including
> `Sap_150N_Avg`. The daily CSV is the coarser non-canonical product (the
> delivered file has 195,091 half-hourly rows, not 4,191 daily ones). My file
> glob was `*.csv`/`*.xlsx`, so the two `.rds` files were invisible to it. The
> layout description is accurate.

The correction does not change what happens, because **the level we process is
the formatted file**, not the raw. `CHE_DAV_sapf_data.csv` carries 15 tree
columns and no sensor level, so a sensor-level decision is inexpressible there
regardless of what the raw retains — exactly as for DE-HoH. What the raw
contains would only matter if Stage 1 were being changed, and D8 removed that
from scope.

Method note worth keeping: "is this recoverable?" was the wrong question, and
answering it sent me into the raw. The question that decides the log's
correctness is "does this row name a column in the file QC opens?" — which the
validator already answers, for every row, without any of this.

## 3b. Stage-1 work pulled in by D7

Fixing the 7 lost removals is not a QC change. Stage 1 collapses sensors to one
series per tree *before* Stage 3 ever opens the file, so a sensor-level removal
is unexpressible by the time the removal log is read. The fix has to happen
where the collapse happens.

### The seven, with the tree each maps to (from `plant_md.pl_name`)

| site | sensor | tree | delivered column | kind | fixable |
|---|---|---|---|---|---|
| `DEU_HOH` | `SF185_20_NW_1`, `_NW_2` | 185 | `DEU_HOH_Qpe_Js_18` | whole sensor | yes |
| `DEU_HOH` | `SF301_20_NNE_1`, `_NNE_2` | 301 | `DEU_HOH_Qpe_Js_28` | whole sensor | yes |
| `DEU_HOH` | `SF194_20_N_2` | 194 | `DEU_HOH_Qpe_Js_23` | whole sensor | yes |
| `DEU_HOH` | `SF082_20_N_1` | 082 | `DEU_HOH_Fsy_Js_11` | **windowed** 2023-01-01→2024-01-01 | yes |
| `CHE_DAV` | `CH_DAV_Sap_150N_Avg` | 150 | `CHE_DAV_Pab_Js_4` | whole sensor | held for review |

`SF082_20_N_1` is a `period` row, so `exclude` cannot be a flat name list: it
needs optional time bounds, applied before the mean rather than after.

### Sub-question 1 — RESOLVED by measurement, not deferred

Trees 185, 194, 301 and 082 all retain surviving sensors after exclusion (tree
185: 3 of 5). **No tree empties**, so the all-NaN-column contingency I proposed
is not needed. What does need handling is the radial-depth shift at trees whose
survivors mix 10 mm and 20 mm probes — report it, do not silently re-weight.

### Sub-question 2 — where the exclusion list lives

Still: **transcribed into `SITE_SENSOR_LAYOUTS`**, citing the log rows, as
`DE_HAR`'s `combine_reason` already does. It matches the module's stated design
("Sites declare themselves in `SITE_SENSOR_LAYOUTS`… a fail-open default would
silently average a site nobody has inspected"), and transcription drift is
caught by scope item 6 — the validator reports any log row naming a column that
does not exist, on every run.

**Correction to my earlier argument.** I also justified this by saying the
alternative "makes regeneration depend on a file that until today was not even
in the repository". D6 tracks `removal_log.csv` in git, so that objection no
longer holds and should not be counted. The case now rests on the design-fit
argument alone, which I still think is the stronger one — but it is one
argument, not two.

### The log needs to distinguish the two levels

Whatever holds the exclusions, the log itself currently gives no way to tell a
sensor-level row from a tree-level one (§3a). Proposal, deferred to you rather
than done unilaterally since it changes the file's schema: add a `level` column
(`tree` default, `sensor` explicit). It would make the `DEU_HAR`-versus-
`DEU_HOH` confusion structurally impossible instead of merely detected.

### Regeneration hazards

- Re-running a Stage-1 generator **silently reverts five patchers**. They must
  be re-run, in order, per `regeneration_patcher_checklist`.
- Run Stage 1 under `.venv`, never bare `python`.
- `csv/sapwood_combined` was built with `--link`, so its files are hardlinks to
  the sources. It must be **rebuilt**, never edited in place.
- Only `DEU_HOH` and `CHE_DAV` may change. Every other site's files must come
  out byte-identical — the same gate as criterion 4, applied to the rebuild.

## 4. Approach

### Chosen: extract `src/qc_flags/`, then call it from both analyzers

A small pure package that turns *(data frame, optional flag frame, config)*
into a boolean mask plus a provenance record. Both analyzers call it.

```
src/qc_flags/
    config.py       QcFlagConfig (frozen): threshold, token set, propagation map
    vocabulary.py   SAPFLUXNET tokens; which imply "act on a retained value"
    mechanisms.py   detect(site, dir, columns) -> SAPFLUXNET | QC_COLUMNS | NONE
    columns.py      split_qc_columns(frame) -> (variables, qc)   # fixes D-3
    masking.py      build_mask(...) -> DataFrame[bool]           # one algorithm
    report.py       SiteFlagOutcome + run summary (JSON + printed table)
    tests/
```

Why not patch in place: the two analyzers are 1,381 and 1,936 lines with zero
tests, and the flag logic is already duplicated between them. Three mechanisms
x two analyzers is six code paths if written inline, one if written once. The
derived-variable propagation rule in particular must have exactly one home —
it is the kind of rule that rots the moment it is copied.

Why not the shortcut (synthesise `*_flags.csv` files for the internal sites
from their `*_qc` columns, changing no analyzer code): it fabricates
archive-shaped files asserting a token vocabulary the data never had; it leaves
D-1 in place for the 5 sites with no QC at all; and it writes into a corpus
that is a generated artefact and must be rebuildable, not edited.

### Hard constraint

**The 202 external sites must come out byte-identical.** Proven by running
Stage 3 on a sample before and after and comparing outputs file by file, the
same discipline that validated `sapwood_combined`. A refactor of untested code
is only safe if "nothing moved" is a measurement.

### Input wiring

`PathConfig` already parametrises on `scale`, so `PathConfig(scale='sapwood_combined')`
resolves `raw_csv_dir` to the combined corpus and — usefully — sends outputs to
`outputs/processed_data/sapwood_combined/`, isolated from the existing sapwood
outputs so nothing is clobbered. To confirm in Phase 3 that
`internal_metadata_dir` (which would resolve to a non-existent path) is unread
on the QC path. No default changes.

---

## 5. Failure modes and how the plan answers them

| risk | answer |
|---|---|
| A site fails silently, as in D-1 | Every site ends in a recorded state — `mechanism`, per-variable masked counts, or a skip reason. Acceptance is `226 in, 226 accounted for`. The report is written from an **independent** re-derivation of the mechanism (does `{site}_env_flags.csv` exist? does the header carry `*_qc`?), not from what the code decided to do — the tautology trap from the last review. |
| QC semantics wrong | Threshold is config, not a literal. Cost is bounded and measured (§1). Evidence recorded in the deviations log. |
| Over-masking derived variables | Verified that no site has an independent `ppfd_in` or `rh` measurement, so propagation cannot destroy a real observation. |
| `SWE_NOR` degenerate input | Explicit skip with reason; must not emit an empty file the merge would join to nothing. |
| Regression on the 202 | Byte-comparison gate before anything else is accepted. |
| Palma is Python 3.9 | No `zip(strict=)`, no PEP 604 outside `from __future__ import annotations`, no `match`. Write loops the linter hook has nothing to "fix" — it has injected `zip(strict=)` into this repo repeatedly. |
| Memory: 226 sites, `CHE_DAV` env is 241k rows | Keep the existing batch mechanism; do not load the corpus at once. |

---

## 6. Decisions — SETTLED 2026-08-16

All three answered as recommended. Restated here as requirements, not options:

- **D1 → mask `qc >= 1`.** Default threshold is 1; only level 0 survives.
  Configurable, but this is the default the acceptance criteria assume.
- **D2 → `OUT_WARN` is never applied.** Not "not yet" — never, at any site,
  under any configuration. `RANGE_WARN` remains the only actionable token.
  Your call, 2026-08-16, and it is the stronger form of what I had proposed:
  I had it as a deferred gap, you ruled it out permanently. The technical
  reading that supports it (mine, not stated by you): `OUT_WARN` is a
  statistical-outlier flag and this pipeline runs its own outlier detection
  (rolling z-score, 3σ day / 5σ night, 30-day window), so applying it would
  stack the archive's outlier definition on top of the project's own, whereas
  `RANGE_WARN` is a physical-plausibility check that nothing else here
  duplicates.
  Encoded as `ACTIONABLE_TOKENS = frozenset({"RANGE_WARN"})` in
  `vocabulary.py`, with `OUT_WARN` listed in a named `NEVER_ACTIONABLE` set
  carrying this rationale, plus a test that fails if it ever moves (§8.11).
  Side benefit: criterion 4 ("0 of 202 files differ") stays usable as the
  safety gate, because no external site's result changes.
- **D3 → `FRA_HES` `pa` at 37.4% accepted.** Not in the dynamic feature list;
  the loss does not reach training. To be restated in the run report so it is
  visible rather than buried.
- **D4 → `SWE_NOR` skipped here, extraction failure raised separately.** Its
  `env_data.csv` is a *fully-built* scaffold — 16,973 rows on a timestamp grid
  identical to its `sapf_data.csv`, and zero variables. `SE-Nor` is registered
  in `icos_data_extractor.py:74`, so this is a silent extraction failure, not a
  site that has no environmental data. 26 trees are stranded until it is fixed.
  Skipped in QC with reason `no environmental variables (ICOS extraction
  produced a variable-less scaffold — see deviations 2026-08-16)`; the
  re-extraction is its own task, not a dependency of this one.
- **D6 → track `removal_log.csv` in git.** So Palma has it and every change has
  history. The seven untracked `removal_log.backup_*.csv` files are superseded
  by that history; I will leave them on disk and not add them.
- **D7 → SUPERSEDED by D8.** I proposed adding an exclusion capability to
  `SensorLayout` and regenerating; you then ruled that the log's sensor-level
  rows are simply wrong as records, because the corpus has no sensor level.
  That is the better framing: D7 fixed the *data* to match the log, D8 fixes the
  *log* to match the data — and only D8 also handles the `CH_Dav` row, which no
  regeneration could reach (§3a). **No Stage-1 change, no regeneration, no
  corpus rebuild.** §3b is retained below only as the record of what was
  considered.
- **D8 → bring the log onto the level the corpus has. DONE 2026-08-16**, see §2d.

- **D5 → validation run only.** The Palma run over all 226 proves §8 and
  nothing more. Output is *not* blessed as the canonical Stage-3 product and
  the Stage-4 merge is *not* repointed; both wait on your inspection of the QC
  figures. Two reasons beyond caution: this QC output has had no visual review,
  and `removal_log.csv` was tuned against the old corpus, so its `location`
  keys need re-checking against the combined site codes before anything
  downstream trusts the result.

Original wording and rationale below.

**D1 — QC threshold default.** Recommend **mask `qc >= 1`** (keep only level 0)
for the six flagged variables, configurable. Rationale: level 1 values are
demonstrably 10–20x smoother than measured data at the same site, i.e. model
output, and training a sap-flow model on modelled drivers is the coupling we
otherwise work to avoid. Cost 2.4–7.7% pooled. The alternative, `>= 2`, saves
about one percentage point on `ta`/`vpd`/`sw_in` and is identical for the
other three.

**D2 — SAPFLUXNET token vocabulary.** *(Superseded — I proposed deferring
`OUT_WARN` to separate work; the ruling is that it is never applied. See the
settled entry above. The ~308k sap and ~199k env cells it would have masked
across the 202 external sites stay untouched, permanently.)*

**D3 — `pa` at `FRA_HES` (37.4% masked).** Recommend accepting it: `pa` is not
in the model's dynamic feature list. Flagging it because the number is large
enough that you should see it before I bury it in a log.

---

## 7. Phases

| phase | content | gate |
|---|---|---|
| **3 step 0** | **Measure before coding.** Over all 226 sites in the combined corpus, for both `sapf_data` and `env_data`: count columns whose name is a substring of another, and count cases where the first substring match is not the exact column. Replaces the external-only census the D-5 correction invalidated. Also settles the three §7-adjacent verifications: do flag columns match data columns exactly on all 202; is `internal_metadata_dir` read on the QC path; do `*_qc` columns currently reach the merge. | numbers recorded in the deviations log |
| **3 step 0b** | ~~Stage-1 fix (D7)~~ — **dropped**, superseded by D8 (§2d), which is done. Remaining here: track `removal_log.csv` **and** `removal_log.pending_review.csv` in git (D6). | criterion 14 |
| 3 | RED tests for D-1..D-4 + the three mechanisms; then `src/qc_flags/`; then wire both analyzers | `/verify` |
| 4 | `/code-review`, `/security-scan` | `/verify` |
| 5 R1 | edge cases: empty env, all-NaN variable, QC column without its variable, mixed int/float QC dtype | `/verify` |
| 5 R2 | negative paths: malformed QC values, flag file whose columns do not match data, missing `solar_TIMESTAMP` | `/verify` |
| 5 R3 | integration on real sites; byte-comparison of the 202; full-corpus report | `/verify` + `/test-coverage` 80% |
| — | Palma sbatch validation over all 226 | acceptance §8 |

---

## 8. Acceptance criteria

Each is a measurement, and each is checked against something the code under
test did not itself produce.

1. **Mechanism census.** The run report classifies exactly **202** sites as
   `sapfluxnet_flags`, **19** as `qc_columns`, **4** as `none`, and skips
   **1** (`SWE_NOR`) with reason `no environmental variables`. Cross-checked
   against a directory listing and CSV headers computed independently.
2. **No silent losses.** `sites_in == sites_reported == 226`; every site's
   outcome carries either masked counts or a skip reason. Zero sites reach the
   bare `except`.
3. **Env QC completes for all 24 flagless sites.** Today the number is 0.

   *Measured corpus-wide, 2026-08-17 (job 46228827).* Daily env outputs across
   the two full-corpus arms: **before 202, after 225**. The 202 present in both
   are exactly the 202 external sites, byte-identical; the **23** that appear
   only in the new run are every internal site except `SWE_NOR`, which is the
   deliberate skip (no env variables at all). So the criterion holds at **23 of
   24, with the 24th skipped for a stated reason** — env QC had genuinely never
   completed for a single internal site. The earlier "8 -> 11" figure was the
   same fact on the 12-site sample.

   These files were found by an audit of what *neither* site list scored, not by
   the criterion-4 comparison: they are named `daily_{SITE}_env_data.csv`, and
   the comparer matched site codes as a basename **prefix**, so the `daily_`
   prefix hid all 202 of them. Fixed to a substring match.
4. **No regression.** For the 202 external sites, every Stage-3 output file is
   byte-identical to the pre-change run. Target: 0 files differ.

   *Scope correction, 2026-08-17.* The first pass scored this on **8 of the 202**
   — `job_qc_validate.sh` says so in a comment ("chosen small so the comparison
   completes"), but the narrowing never left that file: the deviations row, the
   commit message and the run report all recorded "411 of 411 external output
   files byte-identical" without saying the 411 files came from 8 sites. 411 is
   a file count, not a site count. Re-scored over all 202 by
   `job_qc_compare225.sh`, which derives the site list from the corpus (`ls
   *_env_flags.csv`) rather than a pasted constant, so it cannot drift from the
   census. Note what the sample does *not* threaten: the masking **decision** is
   already measured corpus-wide by criteria 6, 7 and 15b, and D-5 measured
   `columns_exact_absent = 0` across all 202. The residual risk the sample
   leaves is in the surrounding plumbing — dtypes, export gating, row order —
   which is exactly what the byte gate exists to catch and exactly what the
   `CHN_YIN_ST3_DRO` int64/float64 finding turned out to be.

   *Re-scored, 2026-08-17.* Both arms re-run over the **whole corpus**, one site
   per array task (`job_qc_full226.sh before|after`, 226 tasks each), compared by
   `job_qc_compare225.sh` (job 46228406, 10 min on 1 cpu). Result: **14,476 of
   14,476 external output files byte-identical, 0 differing, across 202 of 202
   external sites** — 35.2x the files and 25x the sites of the first pass. 1,601
   files are no longer written, **all verified header-only** (the D-6 fix), and
   **0** files appear at an external site that did not exist before. The one
   defect the 8-site sample had caught (`CHN_YIN_ST3_DRO` int64/float64) recurs
   nowhere else in the corpus.

   `CZE_LNZ` was still running at compare time (69 trees, ~9 h) and is excluded,
   which costs this criterion nothing: it ships **no `*_env_flags.csv` and six
   in-file `*_qc` columns**, so it is one of the 24 internal sites and lies
   outside this criterion's population entirely. 225 of 226 chunks still covers
   **all 202** externals — verified by checking that every code in `ls
   *_env_flags.csv` is represented by at least one output file in both arms.
   Criterion 4 is complete at 202/202, not 201/202.

   *The scored set had a hole, found the same day and closed.* This comparison
   first reported 14,274 identical files rather than the 14,476 above: `is_scored` matched a site code as a
   basename **prefix**, and the env exporter writes `daily_{SITE}_env_data.csv`,
   so one real Stage-3 output per external site — 202 in all — was scored by
   **neither** population and went unmeasured for the whole exercise. They are
   byte-identical, so no verdict ever moved, but the scored set was not what it
   claimed to be. Found by auditing what neither site list matched (job
   46228812): **439** such files, **417 identical, 22 differing**, and all 22 are
   `stage3_summary.json` — the harness's own record, at 22 internal chunks. The
   arithmetic now closes with nothing unexplained: whole-tree differing **384 =
   362 internal outputs + 22 internal summaries + 0 external**. The matcher now
   tests substrings, which over-matches (`FRA_HES` also catches `FRA_HES_HE1`
   files) — the safe direction for a gate, because a file scored under two
   populations reports its difference in both, while a file scored under none
   reports it nowhere.
5. **QC columns are not variables.** No `*_qc` column appears in any env QC
   output — outliers-removed, standardized, or daily.
6. **Propagation — at the 19 ICOS sites only.** `ppfd_in` is masked on exactly
   the rows where `sw_in` is; `rh` on exactly the rows where `ta` or `vpd` is.
   Checked by set equality, not by count. **At the 202 external sites
   propagation must NOT happen**, and the criterion is the opposite: masking
   `sw_in` leaves `ppfd_in` untouched unless `ppfd_in` carries its own token.

   *Corrected 2026-08-16, having originally been stated unconditionally.* The
   §1 evidence ("`without_source = 0` at every site") was gathered at **ICOS
   sites**, and I generalised it to the whole corpus without checking. Measured
   over all 226 (job 46191671): 19 of 176 external sites carry `ppfd_in` on rows
   where `sw_in` is **absent** — 154,155 rows, and at `USA_DUK_HAR` that is
   36,047 of 70,128 present values — with 35 of 200 doing the same for `rh`. A
   value present where its source is missing was not computed from that source,
   so at those sites both are independent measurements with their own flag
   columns. The 19 ICOS sites have **zero** such rows, exactly as
   `icos_data_extractor` implies. Propagating everywhere would have destroyed
   real observations at 19 sites and broken criterion 4 without either being
   attributable to the flag logic.
7. **Measured loss matches the census.** Per-variable masked fractions equal
   the §1 table within 0.01 percentage points.
8. **Negative sap flow is removed for flagless sites too** — the count of
   negative values surviving Stage 3 is 0 for all 226, not just the 202.
9. **Tests.** New module at 80%+ coverage; the full suite green on **Palma
   3.9.25** via sbatch, never the login node.
10. **Flag columns are resolved by equality, not containment.** No call site
    uses `in` to pick a flag column. Proven two ways: an AST/source check that
    the substring idiom is gone, and a behavioural test on a fixture whose
    columns are ordered `Jt_10, Jt_11, Jt_12, Jt_1` — the real `SWE_NOR`
    ordering — asserting that `Jt_1` gets `Jt_1`'s flags. The fixture is built
    from the ordering that actually exists on disk, so it cannot drift into
    testing a case the corpus never contains.
11. **Every removal-log row resolves, or is classified.** The validator sorts
    every non-acting row into `honoured-by-generator`, `lost`, or
    `needs-your-decision`, computing the first two from `SITE_SENSOR_LAYOUTS`
    rather than from an annotation (§2c). After §3b, the `lost` bucket is empty.
    The validator runs as part of the QC entry point, not as a script someone
    remembers to invoke.
11b. **The reformatted log is structurally valid**: 0 rows with an unknown or
    blank `removal_type`, 0 reversed periods, 0 duplicates, 0 `column_entire`
    rows carrying dates, 0 targets that resolve to no file, and every
    `entry_id` unique. **Met** (job 46188083). Nothing further rewrites the file.
11c. **A bare date cannot silently disable other rows.** After the loader is
    hardened, a log mixing `2020-05-01` with `2020-05-01 12:00:00` parses every
    row. Tested with a fixture holding both spellings in one column, asserting
    both entries survive — the case that cost 10 `CZE_STN` rows in §2c.
12. **The manual removals actually apply on Palma.** The count of points
    removed by the removal log in the validation run is > 0 and matches the
    validator's prediction per site. Today that count is necessarily 0, because
    the file is not on Palma at all.
13. **No row in the log is inert.** The validator reports 0 entries that cannot
    act, for any cause. **Met** (job 46188462, 0 of 352). Any future row that
    names a sensor rather than a tree fails this immediately, which is the
    property that was missing when 33 rows sat unnoticed.
13b. **The 4 converted tree removals actually fire. MET — with two corrections
    to this criterion's own wording, 2026-08-17 (job 46192753).** Measured
    against the output, not the removal report, as required.

    - `DEU_HOH_Qpe_Js_18`, `_Qpe_Js_23`, `_Qpe_Js_28`: **0 non-null of 82,778**,
      from 23,930 / 16,312 / 15,960 in the raw. ~~absent from the Stage-3
      output~~ — they are **present as headers and entirely NaN** in
      `outliers_removed`, because `apply_removals` blanks a `column_entire`
      rather than dropping it, and are absent from the `daily/` file one stage
      later. The removal fires; "absent" named the wrong artifact.
    - `DEU_HOH_Fsy_Js_11`: **0 non-null in 2023, 14,988 outside it.** Exact.
    - ~~DEU_HOH ends with 24 usable trees~~ → **25**. 28 columns, 3 fully
      blanked. The fourth conversion is a `period` row and keeps its bounds, so
      that tree retains 14,988 of its 19,587 records (76.5%) and is still
      usable. This also corrects §2d, which claims both that the 082 row "keeps
      its `period` bounds" and that the cost is "4 of DEU_HOH's 28 trees" —
      those cannot both hold. **The cost is 3 trees plus one tree's 2023
      season**, not 4 trees.
13c. **DEU_HAR keeps all 10 trees.** The deletion of its 10 log rows must change
    nothing: `DEU_HAR`'s Stage-3 output is byte-identical to a run before this
    change. This is the check that would have caught converting group B by
    mistake.
14. **`removal_log.csv` is in git**, present on Palma, and its sha256 matches
    the local copy at the time of the validation run.
15. **`OUT_WARN` never masks anything.** Two independent checks, because a
    constant is easy to edit and a behavioural test on a fixture proves only
    that one fixture behaved. (a) `ACTIONABLE_TOKENS == {"RANGE_WARN"}` exactly,
    and `OUT_WARN` is in `NEVER_ACTIONABLE`; (b) over the real corpus, the count
    of cells masked by the flag mechanism equals the count of cells whose token
    contains `RANGE_WARN`, on every one of the 202 sites — so a cell carrying
    `OUT_WARN` alone is measurably still present after Stage 3.

---

## 9. Open questions carried, not resolved here

- Whether Stage 4 should read `outputs/processed_data/sapwood_combined/` — out
  of scope, but this plan is what makes it possible.
- The FLUXNET wording for QC levels 0/1/2 remains unconfirmed by documentation.
  If you have the ICOS ETC L2 product PDF locally it would close the last gap;
  it changes nothing structural, only the comment above the default.
