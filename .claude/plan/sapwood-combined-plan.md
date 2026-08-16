# Plan — `csv/sapwood_combined`: one corpus at sapwood scale

**Goal.** Assemble every site we hold into a single directory expressed in sap flux
density (cm³ cm⁻² h⁻¹): copy the sites that already carry that scale, and convert
the plant-only sites by dividing their cm³ h⁻¹ series by an imputed sapwood area.

**Status. DELIVERED** (2026-08-16, Palma job 46162603). All acceptance criteria met.

`/scratch/tmp/yluo2/Global_Sap_Velocity/data/raw/0.1.5/0.1.5/csv/sapwood_combined/`
— **226 sites, 1,986 files + MANIFEST.json, 6.5 GB on disk**, 3,182 trees, 0 trees
dropped. 181 native + 45 converted; ESP_GDN skipped for want of a usable sapwood
area. ASW model `taxon_refinement=True, offsets=41, n=2139`.

Verified: 120/120 tests on Palma's Python 3.9.25 and 278 locally at 99% coverage;
per-site kinds match disk 226/226; every `plant_md` reads `cm3 cm-2 h-1`; flags
align with data on all 202 sites that have them; no converted median outside
`[1e-4, 1e4]`.

**One open item.** Two converted sites, `CZE_BK1` (median 0.0073) and `CZE_RAJ`
(0.0066), sit below the `[0.01, 100]` review band — flagged, not failed. Their
imputed sapwood areas are plausible, so the anomaly is upstream in the
plant-scale flux. Cause not established; see the deviations log.

---

## Aspect 1 — Problem space

### Membership (verified against the corpus, 2026-08-15)

| corpus | native sapwood | plant-only → convert |
| --- | --- | --- |
| SAPFLUXNET `data/raw/0.1.5/0.1.5/csv/` | 165 | 37 |
| internal `Sapflow-internal/Sapflow_SAPFLUXNET_format_unitcon/` | 16 | 9 |
| **total** | **181** | **46** |

227 unique sites. **Zero filename collisions** between the two corpora, so one flat
output directory is safe.

### File kinds (corrected 2026-08-15)

SAPFLUXNET ships **nine** files per site, not seven:

| kind | external | internal |
| --- | --- | --- |
| `sapf_data`, `plant_md`, `env_data`, `env_md`, `site_md`, `stand_md`, `species_md` | ✅ | ✅ |
| **`sapf_flags`**, **`env_flags`** | ✅ (194 plant / 165 sapwood) | ❌ never generated |

An earlier draft of this plan said seven, because the inventory script hardcoded
the seven kinds it searched for and therefore found exactly seven. Two
consequences follow, and neither is cosmetic:

1. **Flag filtering is step 1 of the 6-step QC.** `sap_analyzer.py:440` resolves
   the flags file by name *beside* the data file and falls back to ``flags =
   None`` when absent (line 447). Shipping a combined corpus without flags would
   therefore disable flag-based QC for all 181 external sites **silently** --
   `NA_ADDED` / `OUT_WARN` / `NA_PRESENT` records simply never applied.
2. **`sapf_flags` is column-per-tree, mirroring `sapf_data`.** A converted site
   that drops a tree for lacking a usable area must drop the same column from its
   flags file, or the pair goes out of alignment. Flags are categorical, never
   divided -- only column-filtered.

The internal corpus has no flags files at all. That asymmetry is legitimate (QC
degrades gracefully) but must be **stated in the manifest**, so it reads as a
known property rather than as data lost in assembly.

### Membership policy: nothing is excluded (decided 2026-08-15)

**All 227 sites enter. 226 are written**; the one absence, `ESP_GDN`, is a *data*
outcome rather than a policy one — none of its four trees has a usable sapwood
area, so there is nothing to divide and the assembler's normal skip path handles
it. That skip is self-correcting: if the provider resolves the area, a re-run
includes the site with no code change.

Earlier drafts of this plan excluded four sites. That was reversed: whether one
stand should count once or twice is a **modelling** decision, not an assembly
one, and it is far easier to filter later than to recover a dropped site. So the
overlap findings are **recorded, not applied** — `membership.OVERLAP_NOTES` feeds
the manifest's `stem_overlaps` block, carrying the evidence for each pair.

#### The overlap sweep (all 227 sites, 2026-08-15)

Screen: co-location within 3 km. Test: tree-level DBH matching within 0.15 cm and
`pl_name` overlap, then correlation on the shared window.

**Diagnostic.** Two series from the *same stem* pair one-to-one at r≈0.98; two
stems in one stand reach only r≈0.85–0.95 with several columns collapsing onto a
single partner. Sharper in open stands than closed canopy, where between-tree
correlation is already high. Metadata alone misclassifies both `SWE_SGR` (looks
distinct — different tree ids) and `ESP_LMA` (looks identical — DBH to 0.1 cm).

**Shares stems but not time — repeated measures by design, not duplication.**
Archive pre/post and treatment splits (`CAN_TUR_P39`, `SEN_SOU`, `USA_HIL_HF1`,
`USA_WIL_WC1`/`WC2`) and the cross-corpus pairs `FIN_HYY_SME`/`FIN_HYY`
(2013–16 vs 2019–23) and `CZE_RAJ_RAJ`/`CZE_RAJ` (2012 vs 2017–21). Not recorded.

**Shares stems and time — recorded in `stem_overlaps`.**

| site | overlaps | relation | key evidence |
| --- | --- | --- | --- |
| `SWE_NOR` (int) | `ST1_AF1` + `ST1_AF2` | same stems, merged plots | 26 = 12 ∪ 14, `pl_name` identical, DBH/ASW to 6 dp, 1:1 at r=0.979–0.990. Archive keeps the Af1/Af2 treatment this copy blanked |
| `SWE_SKO_MIN` (ext) | `SWE_SGR` | same stems, contained in time | 11/12 1:1 at r=0.978–0.988; archive data lies entirely inside 2015-05→2017-10, holding 0 unique rows vs 9,322 |
| `CZE_BIK` (ext) | `CZE_BK1` | archive ⊂ internal | 7/7 by `pl_name` and DBH; 2016 only vs 2015–2023 |
| `CZE_STI` (ext) | `CZE_STN` | archive ⊂ internal | 8/8 by `pl_name`; 2016 only vs 2016–2023 |
| `CZE_LAN` (ext) | `CZE_LNZ` | archive ⊂ internal | 13/17 distinct partners; 2016 only vs 2016–2025 |
| `CHE_DAV_SEE` (ext) | `CHE_DAV` | archive ⊂ internal | 5/5 by `pl_name`; 2010 only vs 2010–2024 |
| `FRA_HES_HE1_NON`, `HE2_NON` (ext) | `FRA_HES` | archive ⊂ internal | 10+10 of 22 `pl_name`, exact partition. **Unverified by signal** — the files share no timestamp at all, so their sampling grids differ |
| `ESP_LMA` (int) | `ESP_MAJ_MAI` | **UNRESOLVED** | 5/6 DBH within 0.09 cm, but 0 `pl_name` overlap, equal counts (not a subset), and correlation is *lower* against its supposed twin (0.867) than against a different plot `ESP_MAJ_NOR_LM1` (0.892) — impossible if stems were shared |

Two notes on `ESP_LMA`. Majadas del Tiétar is a multi-plot fertilisation
experiment (archive: Main + North LM1; internal: LMA + LM1 + LM2) and **the
archive reports one site-level coordinate for all its plots**, so apparent
plot-to-plot distances there mean nothing. The species difference is nominal:
*Q. ilex* vs *Q. rotundifolia* is a segregate pair. Plot correspondence needs the
provider.

### The unit contract (verified, not assumed)

`pl_sap_units` is uniform within each flavour:

- all 194 external + 14 internal **plant** sites: `cm3 h-1`
- all 165 external + 16 internal **sapwood** sites: `cm3 cm-2 h-1`

So the conversion is a plain per-tree division by sapwood area with **no per-site
unit dispatch**. `pl_sap_units_orig` varies wildly (`kg h-1`, `g m-2 s-1`,
`dm3 dm-2 h-1`, …) but that is the *provider's* original unit, already normalised
upstream; it must be preserved untouched, never acted on.

### Out of scope

- Re-running the ASW imputation. `plant_asw_filled/` and `sapwood_asw_filled/` are
  already delivered from the refined taxon-offset model (2026-08-14) and are inputs here.
- Any QC, gap-filling, or resampling (Stage 3+).
- Pointing the merge pipeline at the new directory.
- The known open provider questions (ESP_GDN branch-DBH units; FRA_HES / CHE_DAV
  measured-archive join). This plan *reports* them, it does not resolve them.

---

## Aspect 2 — Approaches considered

**A. New module `src/sapwood_combined/` — CHOSEN.**
Reads the four source directories plus the two `*_asw_filled/` directories and
writes one output tree. Reuses `src.sapwood_imputation.columns` and `.sources`
rather than duplicating the column names and duplicate-site registry.
*Why:* assembly is a distinct responsibility from estimation; it is testable on
synthetic fixtures with no bulk data; and it cannot regress either existing pipeline.

**B. Extend `Sapflow-internal/plant_to_sapwood_converter.py` — rejected.**
That script is a Stage-1 *generator* for the internal corpus: hard-coded absolute
paths, a hard-coded four-site list, and imports (`metadata_schema`,
`timestamp_conformance`) from a hyphenated directory that is not importable from
`src/`. Teaching it about the external archive would make one script own two
pipelines, and per `regeneration_patcher_checklist` any Stage-1 re-run would then
silently rewrite the combined product too.

**C. Add `--emit-combined` to `impute_sapwood_area.py` — rejected.**
Conflates producing a number with assembling a dataset, and would make the
imputer's test suite depend on multi-GB `sapf_data` files.

### Size (measured 2026-08-15, dry run over the real corpus)

**7.12 GB, 1,993 files across 227 candidate sites** — less ESP_GDN's 7 → the
**1,986 files / 226 sites** the acceptance criteria state. An earlier draft said
3.96 GB because it costed seven kinds; flags are not a rounding error:

| source | seven kinds | flags |
| --- | --- | --- |
| external plant | 3.13 GB | 2.93 GB |
| external sapwood | 2.31 GB | 2.18 GB |
| internal (both flavours) | 0.71 GB | — |

Flags are ~45% of the archive by volume. Verified no site code is a prefix of
another within any source directory, so the `{site}_*.csv` estimate is exact
rather than over-counted.

### Sub-decision: write vs. copy

Only two things are genuinely new: the 46 converted `sapf_data` files (0.73 GB) and
the `plant_md` files (0.001 GB). A converted site's `sapf_flags` is rewritten
**only when a tree was actually dropped**; otherwise the column set is unchanged
and it is copied, keeping 2.9 GB of flags out of a pandas round-trip that could
not alter them. Unchanged files are
copied with `shutil.copy2` (byte-identical, mtime preserved); `--link` opts into
hardlinks where the filesystem allows, falling back to copy on `OSError`
(cross-device). **Never** re-serialise a file that does not change — a pandas
round-trip would silently reformat timestamps and drift float precision.

---

## Aspect 3 — Failure modes

| # | Failure | Mitigation |
| --- | --- | --- |
| 1 | **A tree column left undivided** → cm³ h⁻¹ under a cm³ cm⁻² h⁻¹ header. This is not hypothetical: `plant_to_sapwood_converter` did exactly this when its skip-guard stopped matching after a rename, and SE_Nor's median read 122 instead of 0.30. | Build the output frame from an explicit converted-column list. Any data column without a positive finite ASW is **dropped**, never passed through. Post-write assertion: every retained data column differs from its input. |
| 2 | **Site with zero convertible trees** (ESP_GDN: all 4 trees `implausible`) written as a timestamps-only file that later reads as an empty site. | Skip the site entirely; name it in `MANIFEST.json` under `skipped` with the reason. |
| 3 | **Filename collision across corpora** → one site silently overwrites another. | Assert disjoint at membership time and raise. (None today; the guard is for the day an internal site is renamed onto an archive code.) |
| 4 | **Name drift between the flavour directories** — a site spelled differently in `plant/` and `sapwood/` fails the `site in native` test, so its *measured* series is silently replaced by one derived from an imputed area. The route decision currently uses an exact string match while the collision check three lines above folds the name, and that inconsistency is the whole exposure. Live pattern in this repo: site-keyed registries have missed silently on a case change before, and `WindowsPath.__eq__` is case-insensitive so `exists()` cannot even see a case rename. | Route on `canonical_site(site)`, the same key the collision check uses. Assert that folding never merges two distinct sites within one corpus. Verified 2026-08-15: 157 external and 5 internal shared sites, identical raw and folded — the defect is latent, not active. |
| 4b | **Flags dropped or left misaligned** — see *File kinds*: shipping without `sapf_flags` disables step 1 of QC silently, and keeping a flags column whose data column was dropped misaligns the pair. | `FILE_KINDS` carries all nine. Converted sites filter `sapf_flags` to the surviving tree columns. A test asserts every written site has exactly the kinds its source had. |
| 5 | **`pl_code` join drift** — a `sapf_data` column absent from `plant_md`. 0/599 today. | Raise, do not warn. (The internal converter already raises here; keep that contract.) |
| 6 | **Missing `*_asw_filled` file** → site silently skipped. | Raise, naming the expected path. |
| 7 | **Stale ASW model** — regenerating `plant_asw_filled/` without `--taxon-refinement` silently changes 280 trees and leaves an identical fit report. | Read `asw_allometry_fit_report.json` and record in our manifest whether `taxon_offsets` is present, so the combined output states which ASW model produced it. |
| 8 | **Interrupted run** leaves a half dataset that looks complete. | `MANIFEST.json` is written **last**; its absence marks the directory incomplete. `--force` required to overwrite a non-empty output dir. |
| 9 | **Absurd ASW** (0, negative, NaN, inf) → division blow-up or silent inf. | Require finite and `> 0`; otherwise drop the column (case 1). |
| 10 | **Non-numeric cells** (`NA` strings) coerced wrongly. | `pd.to_numeric(errors="coerce")` on data columns only; timestamp columns read and written as **strings**, never parsed. |
| 11 | **Float bloat** — division turns clean decimals into 17-digit floats and inflates the file. | Write with `float_format="%.6g"`, matching archive precision. |
| 12 | **Duplicate stems** — one stand entering twice under two codes gets double weight in training. | Deliberately **not** resolved here: recorded in `membership.OVERLAP_NOTES` (canonical-site keyed) and surfaced in the manifest's `stem_overlaps`, so the filtering decision stays with the modeller. Dropping a site at build time is the harder error to undo. |
| 13 | Disk: E: has 0.2 GB free against a **7.12 GB** product. | Build on Palma; `--dry-run` reports the byte estimate before writing. Measured 2026-08-15 — see *Size* below. |

---

## Aspect 4 — Dependency chain

1. `membership.py` — pure, no dependencies beyond `sources.py`. First.
2. `conversion.py` — pure (DataFrame in, DataFrame out). Independent of 1; can be built in parallel.
3. `assembly.py` — needs 1 and 2.
4. `manifest.py` — needs the records emitted by 3.
5. CLI — needs all.

External dependencies: none (no network, no GEE). Inputs `plant_asw_filled/` and
`sapwood_asw_filled/` already exist.

**Python 3.9 (Palma).** No `zip(strict=)`, no PEP-604 unions at runtime, no `match`.
`from __future__ import annotations` in every module so `X | None` annotations stay legal.

---

## Aspect 5 — Test strategy

All tests run on synthetic fixtures under `tmp_path`. No test reads the real corpus
except one explicitly-marked dry-run smoke test.

**Round 1 — core (Phase 3, written before implementation)**
- native sapwood wins over plant for a site present at both scales
- plant-only site is routed to `converted`; route sets are disjoint
- conversion arithmetic: `out[c] == in[c] / asw[c]`, per-column area
- `pl_sap_units` rewritten to `cm3 cm-2 h-1`; `pl_sap_units_orig` untouched
- `TIMESTAMP` and `solar_TIMESTAMP` survive **verbatim** as strings
- **a tree with no usable ASW is dropped from `sapf_data`** (the anti-SE_Nor test)
- **and dropped from `sapf_data`'s companion `sapf_flags` too**, so the two agree
- flags values are carried through unchanged (categorical, never divided)
- a site whose source has no flags files produces none, without error
- a written site has exactly the kinds its source had (the anti-7-kinds test)
- routing survives a case/separator difference between the two flavour directories
- a site with zero convertible trees is skipped, not written
- copied files are byte-identical to source
- `MANIFEST.json` is written last and lists every site with its route and kinds

**Round 2 — edge (Phase 5 R1)**
empty `sapf_data`; all-NA column; ASW `0` / negative / NaN / inf; single-tree site;
`plant_md` row with no matching column; float round-trip precision; a site whose
`sapf_data` has only timestamp columns.

**Round 3 — negative (Phase 5 R2)**
missing `*_asw_filled` file → raise; `sapf_data` column absent from `plant_md` → raise;
cross-corpus filename collision → raise; non-empty output dir without `--force` → raise;
missing source directory → raise; unreadable `float_format` inputs.

**Round 4 — integration (Phase 5 R3)**
end-to-end over a miniature two-corpus fixture (native + converted + skipped);
idempotence (two runs produce identical bytes apart from the manifest timestamp);
plus a real-corpus **dry-run** smoke test asserting 227 candidates / 46 conversions /
ESP_GDN skipped, with no writes.

---

## Aspect 6 — Acceptance criteria

1. `sapwood_combined/` holds **226 sites** (227 candidates − ESP_GDN) plus
   `MANIFEST.json`: **181 native + 45 converted**, and because the two corpora
   ship different kind sets, **202 external × 9 + 24 internal × 7 = 1986 files**.

   | | native | converted | files |
   | --- | --- | --- | --- |
   | external | 165 | 37 | 202 × 9 = 1818 |
   | internal | 16 | 8 | 24 × 7 = 168 |
   | **total** | **181** | **45** | **1986** |

1b. Every written site has **exactly the kinds its source directory had** —
   no kind silently dropped, none invented.
2. Every `*_plant_md.csv` in the output has `pl_sap_units == "cm3 cm-2 h-1"` on every row.
3. For each of the 181 native sites, `{site}_sapf_data.csv` is **byte-identical** to its
   source in `csv/sapwood/` or `unitcon/sapwood/`.
3b. `ESP_GDN` is absent and named in `MANIFEST.json` under `skipped` with its
   reason. No site is absent for any other cause. The nine `stem_overlaps`
   entries are all present in the output and each carries `overlaps`,
   `relation` and `evidence` in the manifest.

4. For each of the 45 converted sites and every retained data column `c`:
   `out[c] == in[c] / asw[c]` within 1e-9 relative, and `asw[c]` is finite and `> 0`.
5. No output `sapf_data` contains a column whose ASW was missing or `implausible`.
   ESP_GDN is absent from the output and named in `MANIFEST.json` under `skipped`.

5b. For every converted external site, `sapf_flags` carries **exactly the same
   tree columns** as its `sapf_data`, and its flag *values* are unchanged from
   the source. For native sites `sapf_flags` and `env_flags` are byte-identical
   to source.
6. Each converted site's median value lies in `[0.01, 100]` cm³ cm⁻² h⁻¹. (Observed
   native band is 0.48–12.0 at p5–p95; the 100× envelope catches a unit error without
   failing a legitimately extreme site.)
7. `MANIFEST.json` records per site: `route`, `origin`, `n_trees_in`, `n_trees_out`,
   `dropped` (with reasons), **`kinds` (which of the nine were written)**, and for
   converted sites the ASW source-flag counts; plus top-level blocks naming the ASW
   model (`taxon_refinement: true/false`), the `stem_overlaps` findings, and
   **`flags_available`, stating that the 24 internal sites ship no `sapf_flags` /
   `env_flags` so consumers do not read the gap as assembly loss**.
8. Re-running over unchanged inputs produces identical bytes except the manifest timestamp.
9. `pytest src/sapwood_combined/tests/ -q` passes with ≥80% statement coverage of the module.

---

## Module layout

```
src/sapwood_combined/
    __init__.py
    membership.py   # which site takes which route  (pure)
    conversion.py   # cm3/h -> cm3/cm2/h            (pure, DataFrame in/out)
    assembly.py     # copy / write one site         (I/O)
    manifest.py     # MANIFEST.json                 (pure + one write)
    build_sapwood_combined.py   # CLI: python -m src.sapwood_combined.build_sapwood_combined
    tests/
```

CLI mirrors `impute_sapwood_area.py`'s conventions:
`--base-data-dir`, `--base-output-dir`, `--output-subdir` (default `sapwood_combined`),
`--dry-run`, `--force`, `--link`, `--skip-internal`, `--internal-root`.

The internal corpus is located by `_internal_root()`: explicit `--internal-root`,
else `paths.internal_metadata_dir` if the installed `path_config` has it, else a
repository-relative default. Reading that attribute unconditionally broke the
first Palma run, because the canonical environment's `path_config.py` is older
than this machine's.

## Deployment

E: cannot hold the product (0.2 GB free vs 7.12 GB) and is a syncing OneDrive tree.
Build and test the module locally on synthetic fixtures, then SCP to Palma and run
the real generation there under `sbatch` — never on the login node, never with bare
`python` (use the Palma venv, Python 3.9.25).
