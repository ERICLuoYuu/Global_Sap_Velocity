# Implementation Plan: Plant-Level Plot Encoding + Output Folder Routing

## Task Type
- [x] Backend (data loading / path routing)
- [x] Frontend (plot coloring / label)

---

## Discovery Summary

### Problem 1 — Plant-level color encoding in tree-size plots

**Root cause:** `self.tree_df` IS already per-plant (one row per individual).
The "aggregation" is the *color encoding* — `hue_col='biome'` in the overall plot
groups all plants from different sites under 9 biome colors, hiding site identity.

**Affected call site:** `plot_tree_size_vs_sapwood_area()`, L591–599.

**NOT affected:** `load_tree_metadata()` — no change needed there.
**NOT affected:** faceted-by-biome and faceted-by-PFT plots — those groupings are correct.

### Problem 2 — Output folder separation

**Root cause:** `self.output_dir` defaults to `.../relationship_exploration/` regardless
of `self.use_raw`. Both raw and outlier-removed runs write to the same folder,
overwriting each other.

**`self.use_raw`** is already available at the point where `output_dir` is set (L127 < L132).

**All `savefig()` calls already use `self.output_dir / save_name`** — no scattered paths.
Fix is localized to a single 2-line change in `__init__`.

---

## Implementation Steps

### Step 1 — Fix output folder routing (Problem 2 first — foundational)

**File:** `src/Analyzers/explore_relationship_observations.py`
**Lines:** 129–133 (`__init__`)

Current:
```python
if output_dir is not None:
    self.output_dir = Path(output_dir)
else:
    self.output_dir = self.paths.figures_root / 'relationship_exploration'
self.output_dir.mkdir(parents=True, exist_ok=True)
```

Replace with:
```python
if output_dir is not None:
    self.output_dir = Path(output_dir)
else:
    data_subfolder = 'raw' if self.use_raw else 'outlier_removed'
    self.output_dir = (
        self.paths.figures_root / 'relationship_exploration' / data_subfolder
    )
self.output_dir.mkdir(parents=True, exist_ok=True)
```

**Expected deliverable:** running with `use_raw=True` creates
`.../relationship_exploration/raw/`; running with `use_raw=False` creates
`.../relationship_exploration/outlier_removed/`. Filenames unchanged.

---

### Step 2 — Add site color map helper for plant-level plots (Problem 1 prerequisite)

**File:** `src/Analyzers/explore_relationship_observations.py`
**Location:** module level, after `PFT_FULL_NAMES` block (~L97)

Add a helper function:
```python
def _make_site_color_map(site_codes: list[str]) -> dict[str, str]:
    """Auto-generate a color per site from matplotlib's tab20/tab20b cycle."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    palette = (
        list(mcolors.TABLEAU_COLORS.values())           # 10 colors
        + [cm.tab20(i) for i in range(20)]              # 20 more
        + [cm.tab20b(i) for i in range(20)]             # 20 more
    )
    return {code: palette[i % len(palette)] for i, code in enumerate(sorted(site_codes))}
```

Alternatively (simpler, no module-level state):
- Just pass `color_map=None` and let matplotlib cycle. The legend will have too many
  entries for 165 sites, so **suppress the legend** for the overall plot — annotate
  with "N plants, M sites" instead.

**Decision: use the simpler approach** — no color_map, no legend for the overall
plant plot. The per-biome and per-PFT faceted plots already provide group-level
color coding. The overall plot's purpose is to show the trend across all plants.

---

### Step 3 — Change hue encoding in overall tree-size plot (Problem 1 main fix)

**File:** `src/Analyzers/explore_relationship_observations.py`
**Lines:** L591–599 inside `plot_tree_size_vs_sapwood_area()`

Current call:
```python
self._make_overall_plot(
    valid, x_col, 'pl_sapw_area',
    x_label=x_label,
    y_label=y_label,
    title=f'{x_label} vs {y_label}',
    save_name=f'{tag}_vs_sapwood_area_overall.png',
    hue_col='biome',
    color_map=BIOME_COLORS,
)
```

Replace with:
```python
n_sites = valid['site_code'].nunique()
self._make_overall_plot(
    valid, x_col, 'pl_sapw_area',
    x_label=x_label,
    y_label=y_label,
    title=f'{x_label} vs Sapwood Area — {len(valid):,} plants, {n_sites} sites',
    save_name=f'{tag}_vs_sapwood_area_overall.png',
    hue_col='site_code',
    color_map=None,       # auto-cycle; legend suppressed (too many sites)
    suppress_legend=True,
)
```

Also update `y_label` at the call site to `'Sapwood Area (cm²) [per plant]'`
to make the unit level explicit.

---

### Step 4 — Add `suppress_legend` parameter to `_make_overall_plot`

**File:** `src/Analyzers/explore_relationship_observations.py`
**Lines:** `_make_overall_plot` signature and body (L480–561)

Add `suppress_legend: bool = False` parameter.

In the body, change:
```python
ax.legend(fontsize=7, loc='upper left', framealpha=0.9, ncol=2, markerscale=2)
```
to:
```python
if not suppress_legend:
    ax.legend(fontsize=7, loc='upper left', framealpha=0.9, ncol=2, markerscale=2)
```

When `hue_col='site_code'` and there are 165 sites, the legend is unreadable —
suppress it and let the title carry the "N plants, M sites" information.

---

### Step 5 — Update faceted tree-size plots to color by site within each group

**File:** `src/Analyzers/explore_relationship_observations.py`
**Location:** `plot_tree_size_vs_sapwood_area()`, the biome and PFT faceted calls (L601–625)

Within each faceted panel (one biome or one PFT), color points by `site_code`
so within-group site variation is visible. This requires passing site colors
to `_make_faceted_plot`.

Two options:
- **Option A**: Add `hue_col` param to `_make_faceted_plot` (scatter within panel colored by hue_col, NOT group_col). Medium complexity.
- **Option B**: Keep current faceted coloring (single color per panel = biome/PFT color). Only change the OVERALL plot. Lower complexity.

**Decision: Option B** — keep faceted plots unchanged. The overall plot is where
site identity matters most. Faceted panels already distinguish groups.

So Steps 5 is a no-op: only the overall plot changes.

---

## Key Files

| File | Lines | Operation | Description |
|------|-------|-----------|-------------|
| `src/Analyzers/explore_relationship_observations.py` | L129–133 | Modify | Auto-route output_dir to raw/ or outlier_removed/ subfolder |
| `src/Analyzers/explore_relationship_observations.py` | L480–490 | Modify | Add `suppress_legend` param to `_make_overall_plot` |
| `src/Analyzers/explore_relationship_observations.py` | ~L507–509 | Modify | Conditional legend in `_make_overall_plot` body |
| `src/Analyzers/explore_relationship_observations.py` | L591–599 | Modify | Change `hue_col='biome'` → `hue_col='site_code'`, add `suppress_legend=True`, update title with plant/site counts |
| `src/Analyzers/explore_relationship_observations.py` | L589 | Modify | Update `y_label` to `'Sapwood Area (cm²) [per plant]'` |

**Total edits: 5 targeted changes in 1 file.**

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| 165+ sites → auto-color cycle wraps, sites share colors | Acceptable: overall plot shows trend not individual site ID; title says "M sites" |
| User-supplied `--output-dir` bypasses auto-routing | By design: user override takes full control |
| Faceted plots by biome/PFT still use biome/PFT colors | Correct behavior, no change needed |
| `_make_overall_plot` now has `suppress_legend` — existing callers unaffected | Default is `False`, backwards-compatible |

---

## SESSION_ID
- CODEX_SESSION: N/A (no external model used)
- GEMINI_SESSION: N/A (no external model used)
