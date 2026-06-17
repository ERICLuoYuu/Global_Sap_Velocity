# SM-VPD Decoupling — Deviations Log

Tracks every divergence from `docs/superpowers/plans/2026-06-16-sm-vpd-decoupling-sapflow.md`
during subagent-driven execution (dev-workflow Phase 3+).

Rules:
- ANY change to scope, approach, file list, or acceptance criteria gets logged.
- Trivial (variable rename, minor refactor) = auto-approved.
- Significant (new module, changed signature, dropped feature) = STOP, ask user.

---

## Deviation 1: Task 2 test count label
- **Phase**: 3 (implement)
- **Planned**: Plan prose says Task 2 yields "PASS (7 tests)".
- **Actual**: `test_decoupling.py` contains 6 test functions (all pass); total estimator-core suite = 9 (6 + 3 sensitivity).
- **Reason**: Off-by-one in the plan's prose label; the listed test code only ever defined 6 functions. No test was dropped.
- **Impact on acceptance criteria**: None (estimator fully covered).
- **Approved**: auto (trivial, documentation-only).

## Deviation 2: loader.py written whole vs incremental append (ruff hook)
- **Phase**: 3 (implement, Tasks 4-6)
- **Planned**: Append loader.py in 3 separate stages (parts 1/2/3), each its own RED→GREEN.
- **Actual**: A ruff PostToolUse hook strips imports not yet referenced, deleting `Path`/`canopy_conductance` on each partial save. loader.py was therefore written complete in one Write during Task 5; Task 6's `load_table`/`resolve_daily_dir` were already present, so Task 6 had no RED phase.
- **Reason**: Format-on-write hook incompatible with import-before-use incremental builds.
- **Impact on acceptance criteria**: None. Final loader.py is identical in content; all 12 loader tests pass; 3 commits still landed per stage.
- **Approved**: auto (trivial, process-only).

## Deviation 3: negative-sap-velocity drop uses .mask()
- **Phase**: 3 (implement, Task 5)
- **Planned**: `e[e < 0] = np.nan`.
- **Actual**: `e = e.mask(e < 0)`.
- **Reason**: pandas 3.0 raises ChainedAssignmentError on the in-place form (plan anticipated this). `.mask()` is behaviourally identical AND portable to Palma's older pandas.
- **Impact on acceptance criteria**: None (test_standardise_drops_negative_sap_velocity passes).
- **Approved**: auto (anticipated by plan).

## Deviation 4: load_table exception clause broadened
- **Phase**: 3 (implement, Task 6) — found in Phase 4 review.
- **Planned**: `except (ValueError, KeyError)` around per-file load.
- **Actual**: `except (ValueError, KeyError, AttributeError, UnicodeDecodeError, OSError)` + logs `type(exc).__name__`.
- **Reason**: More robust batch loading of heterogeneous per-site CSVs (encoding/IO issues skip the file with a warning rather than aborting the whole run).
- **Impact on acceptance criteria**: None. Not silent — each skip is logged at WARNING with the exception type.
- **Approved**: auto (trivial robustness; logged, not swallowed).

---

## Phase 4 Review Summary (2026-06-16)
Manual code review (reviewer subagents did not surface findings; review performed directly on on-disk diff `77dc322..HEAD`).
- **CRITICAL/HIGH:** none.
- **MEDIUM:** none.
- **LOW (by design, documented in spec):** source-level (not per-row) PPFD fallback; Gc∝1/VPD VPD-leg confound.
- **Security:** no secrets/eval/exec/shell-injection; SLURM scripts write only to scratch; `pd.read_csv` default engine.
- **Py3.9/Palma:** portable (future-annotations).
- Tests: 34 passed, 91% coverage (≥80% gate met).
**Verdict: Phase 4 PASS.**

---

## Deviation 5: Python 3.9 compat — drop zip(strict=True)  [REAL BUG, Phase 5 R1]
- **Phase**: 5 round 1 (edge-case expansion surfaced it).
- **Planned/Committed**: plotting.py had `zip(..., strict=True)` in 2 places (auto-injected by a ruff B905 PostToolUse hook on write, committed in df3db34).
- **Actual**: removed `strict=True` → plain `zip(...)`.
- **Reason**: `zip(strict=)` is Python 3.10+. Palma runs Python 3.9 (memory hpc_python_39) → committed code would raise TypeError there. Local 3.14 tests never caught it.
- **Impact**: CRITICAL for Palma run; no behavior change locally. Fixed in commit a442175. Also added named T_MIN constants (plan-aligned) + broadened loader except in same commit.
- **Approved**: auto (correctness fix; required before Palma).
- **WATCH**: the ruff B905 hook may re-inject `strict=True` on any future edit to plotting.py — re-verify `grep -rn "strict=" src/sm_vpd_decoupling/` returns nothing before deploying to Palma.

## Phase 5 Round 1 (edge cases) — DONE
- +17 tests in test_edge_cases.py (commit 802720f). Full suite 51 passed.
- 1 real bug found+fixed: the zip(strict=) Palma incompatibility (above).

## Phase 5 Round 2 (negative paths) — DONE
- +13 tests in test_error_paths.py (commit added test file only — no source bugs found).
- Full suite 64 passed; coverage 95% (run_*.py 62% = _make_figures/main CLI, exercised in real run).
- strict= re-injection check: clean.
**Phase 5 (Standard, 2 rounds) COMPLETE.**

---

## Phase 10 — Palma real-data run (COMPLETE)
- Deployed module to `/scratch/tmp/yluo2/gsv` (tarball SCP; `.sh` CRLF→LF fixed).
- Palma Python 3.9.25 pytest (job 42942842): **64 passed**, exit 0 — confirms 3.9 compat.
- Merge job 42943001 (`--daytime-only --apply-treatment-filter`): COMPLETED, 156 daily CSVs.
- Analysis job 42943017 (afterok dep): COMPLETED in 53s, 62 CSVs + 22 figures.
- Canonical commit on Palma: **53a0b4a** (Palma is the primary repo; not pushed to origin — left to user).

### Real-data results (tair15, n_bins=5, min_valid_days=120)
Attrition: 146 day-filtered sites (40,267 site-days) → 87 (≥120d) → 54 (≥240d) → 35 (≥360d).

| response | SM variant | % SM-dominant | mean ΔSM\|VPD | mean ΔVPD\|SM |
|---|---|---|---|---|
| E (sap velocity) | swvl1→4, rootzone | 52–62% | −0.09 to −0.15 | **+0.09 to +0.11** |
| Gc (Flo 2021) | swvl1→4, rootzone | 8–14% | −0.07 to −0.13 | **−0.34 to −0.39** |

Headline E-vs-Gc dissociation confirmed: the **SM leg is nearly identical** across both
responses (clean, ~−0.09 to −0.15; deepens with soil depth — swvl3/rootzone strongest),
while the **VPD leg flips**: positive for E (transpiration demand) vs strongly negative for
Gc (stomatal closure + the documented Gc∝1/VPD confound). So "does VPD dominate?" depends
entirely on the response chosen — exactly the design's scientific point.

## Deviation 6: plotting rework — 2-D figures, both legs, leg comparison  [user-requested]
- **Phase**: post-delivery (user reviewed figures).
- **Planned**: original plotting showed 1-D marginal curves (response vs single axis), only the SM leg in gradient plots, and no leg-comparison figure.
- **Issue (user-caught):** 1-D marginals do NOT represent the 2-D nested decoupling; the VPD leg (vpd_given_sm) was computed but never plotted; no figure compared the two legs. The estimator itself was correct (2-D, both legs in CSVs) — only the visuals were wrong. Smoke tests (file-written) never checked plot content, so it slipped past review.
- **Actual:** reworked plotting.py:
  - (a2) `plot_cross_site_aggregate` → cross-site mean 2-D heatmap (SM-bin × VPD-bin) with marginal effect bars (SM|VPD per VPD row, VPD|SM per SM col).
  - (a1) `plot_example_sites` → within-bin line plots: response vs SM-bin one line per VPD-bin (= SM|VPD) and response vs VPD-bin one line per SM-bin (= VPD|SM).
  - (c) `plot_gradient_violins` → `plot_gradient_groups`: BOTH legs (paired boxes) per group.
  - NEW `plot_leg_comparison_box` (signed paired boxplots across sites per SM variant) + `plot_leg_comparison_scatter` (|SM|VPD| vs |VPD|SM| with y=x dominance line).
  - All labels use REAL variable names (E_norm/Gc_norm, swvl*/root_zone_sm, vpd) per user; no generic "ΔResp".
  - Rewrote the lingering `zip()` in `plot_depth_dominance` so the ruff B905 hook can't re-inject `strict=` (Palma 3.9 safety).
- **Tests:** test_plotting expanded 4→8 (incl. both-legs-required contract). Full suite 68 passed; local end-to-end smoke = 26 figures, all families present.
- **Impact on acceptance criteria:** figures criterion now genuinely satisfied (2-D + both legs + comparison).
- **Approved:** user-requested.

## Deviation 7: post-review hardening — plot-semantics tests + attrition funnel  [user-requested]
- **Phase**: post-delivery (thorough review, user-requested fixes).
- **Findings (from review of deployed code vs spec):**
  - **A (Medium, testing gap):** plotting smoke tests only asserted file-size>0; the pure
    2-D helpers (`_grid_cell_means`, `_sm_effect_per_vpd`, `_vpd_effect_per_sm`, `_hi_lo`)
    that encode the decoupling semantics had NO assertions — the same blind spot that let
    the original 1-D plots pass review.
  - **B (Low, unlogged spec gap):** `attrition.csv` collapsed the Liu day filter into one
    `day_filtered` row; spec §5 asks for the funnel (input -> Tair -> +VPD -> +PPFD).
  - **F (trivial):** dead `Numeric = "float | pd.Series"` alias in conductance.py.
- **Actual:**
  - A: +5 helper unit tests in test_plotting.py asserting grid shape/orientation and the
    low-minus-high (SM) / high-minus-low (VPD) leg signs on hand-built grids.
  - B: `loader.load_table` gained an optional `attrition_sink`; `_staged_day_counts`
    records per-site cumulative survival; `run._attrition` emits the 4-stage funnel +
    min_valid_days. Real-data funnel: 153->151->151->146 sites, terminal day_filtered
    (146 sites, 40,267 rows) reconciles exactly with the prior collapsed value.
  - F: alias removed.
- **Tests:** full suite 69 -> 74 (local 3.14 + Palma 3.9.25 both green). strict= clean.
- **Impact on acceptance criteria:** figures-semantics now regression-guarded; attrition
  table now matches spec §5. No estimator/result change.
- **Approved:** user-requested.

