
### Deviation 16: Narrowed broad except in process_files
- **Phase:** Final Review (Phase 4 re-review)
- **Planned:** No change expected
- **Actual:** Narrowed `except Exception` to `except (OSError, ValueError, KeyError, pd.errors.EmptyDataError)`
- **Reason:** The broad except previously masked a `NameError` (Deviation 14). Programming errors like `TypeError`, `AttributeError`, `NameError` should propagate immediately, not be silently logged as data issues.
- **Impact on acceptance criteria:** None — defensive hardening
- **Approved:** Auto (bug prevention)

### Deviation 17: Code review round 2 — 5 correctness fixes
- **Phase:** Final Review (Phase 4 re-review, parallel agents)
- **Planned:** No additional changes expected
- **Actual:** 5 fixes from thorough code review:
  1. CRITICAL: `allow_pickle=False` missing in `backfill.py:backfill_from_saved_arrays`
  2. HIGH: Spatial group ID collision in `_create_spatial_groups` when sites at integer-degree boundaries
  3. MEDIUM: `assert` replaced with `raise ValueError` for `-O` flag safety
  4. MEDIUM: File stem parsing now uses regex with validation instead of blind split
  5. MEDIUM: SHAP CSV reindex now validates all features exist before silent NaN injection
- **Reason:** Found by 3 parallel review agents (plan verifier, code reviewer, security reviewer)
- **Impact on acceptance criteria:** None — defensive hardening, no behavioral change
- **Approved:** Auto (bug prevention)
