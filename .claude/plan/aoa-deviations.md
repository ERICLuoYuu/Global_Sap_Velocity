
### Deviation 16: Narrowed broad except in process_files
- **Phase:** Final Review (Phase 4 re-review)
- **Planned:** No change expected
- **Actual:** Narrowed `except Exception` to `except (OSError, ValueError, KeyError, pd.errors.EmptyDataError)`
- **Reason:** The broad except previously masked a `NameError` (Deviation 14). Programming errors like `TypeError`, `AttributeError`, `NameError` should propagate immediately, not be silently logged as data issues.
- **Impact on acceptance criteria:** None — defensive hardening
- **Approved:** Auto (bug prevention)
