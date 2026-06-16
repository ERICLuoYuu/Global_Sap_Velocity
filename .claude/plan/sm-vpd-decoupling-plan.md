# SM-VPD Decoupling — Live Tracking (dev-workflow Phase 3+)

**Canonical plan (full code, source of truth):**
`docs/superpowers/plans/2026-06-16-sm-vpd-decoupling-sapflow.md`
**Spec:** `docs/superpowers/specs/2026-06-16-sm-vpd-decoupling-sapflow-design.md`
**Deviations:** `.claude/plan/sm-vpd-decoupling-deviations.md`

**Goal:** Port Liu et al. 2020 SM/VPD percentile-binning decoupling to site-level
sap flow for two responses — sap velocity (E) and Flo 2021 canopy conductance (Gc)
— across 5 SM depth variants.

## File list (deviation-detector scope)
- `src/sm_vpd_decoupling/__init__.py`
- `src/sm_vpd_decoupling/conductance.py`
- `src/sm_vpd_decoupling/decoupling.py`
- `src/sm_vpd_decoupling/sensitivity.py`
- `src/sm_vpd_decoupling/loader.py`
- `src/sm_vpd_decoupling/aggregate.py`
- `src/sm_vpd_decoupling/plotting.py`
- `src/sm_vpd_decoupling/run_sm_vpd_decoupling.py`
- `src/sm_vpd_decoupling/job_merge_decoupling.sh`
- `src/sm_vpd_decoupling/job_sm_vpd_decoupling.sh`
- `src/sm_vpd_decoupling/tests/*.py`

## Task checklist (TDD: test → fail → impl → pass → commit)
- [ ] Task 0 — scaffold + data-production SLURM script
- [ ] Task 1 — conductance.py (Flo 2021 Eqn 2, 1/VPD confound) [5 tests]
- [ ] Task 2 — decoupling.py (nested-binning estimator, Liu Eqs. 1-2) [7 tests]
- [ ] Task 3 — sensitivity.py (δResp/δSM per 0.1) [3 tests]
- [ ] Task 4 — loader.py: PPFD resolution, root-zone SM, normalization [6 tests]
- [ ] Task 5 — loader.py: day filter + site-frame standardise [3 tests]
- [ ] Task 6 — loader.py: dir resolution + full table build [3 tests]
- [ ] Task 7 — aggregate.py: per-site effects + dominance [3 tests]
- [ ] Task 8 — plotting.py: a1/a2/b/c figures [4 tests]
- [ ] Task 9 — run_*.py CLI + analysis SLURM script [1 integration test]
- [ ] Task 10 — full suite + coverage + real-data sbatch on Palma

## Acceptance criteria
- [ ] All module tests pass locally (`pytest src/sm_vpd_decoupling/ -v`)
- [ ] Coverage ≥ 80% (`--cov=src/sm_vpd_decoupling`)
- [ ] Two responses (E_norm, Gc_norm) covered end-to-end
- [ ] 5 SM variants incl. root-zone weights (0.07,0.21,0.72)
- [ ] Liu day-filter (Tair>15/5, VPD>0.5, PPFD>500, AND) + PPFD fallback chain
- [ ] Per-site normalization (90th-pct anchor)
- [ ] Nested-binning estimator: MIN_BIN_COUNT=3, ≥2 conditioning bins
- [ ] Bin counts {5,10} × min-valid-days {120,240,360} sweep
- [ ] depth_profile.csv (E-vs-Gc dissociation), attrition.csv, per-site CSVs
- [ ] Figures a1/a2/b/c incl. canopy-height grouping
- [ ] Gc 1/VPD confound documented in module docstring
- [ ] Real-data run via sbatch on Palma (NOT login node), commit from Palma

## Out of scope (per spec)
- G′ aerodynamic-corrected conductance
- Per-site Fan-2017 root depth (fixed 0-100cm aggregate used instead)
- Tair>5 run is a re-invocation of the same CLI (documented in job script)
