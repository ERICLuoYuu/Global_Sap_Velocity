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

