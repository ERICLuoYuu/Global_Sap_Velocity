#!/bin/bash
#SBATCH --job-name=sapcomb_diag
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/sapcomb_diag_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/sapcomb_diag_%j.err

# Two questions left open by the build's acceptance check:
#
#  1. Criterion 1b failed for 5 sites. Hypothesis: the CHECK is wrong, not the
#     product -- it globs "{site}_*.csv" in the flat output directory, where
#     CHE_DAV also matches CHE_DAV_SEE. Re-test with exact filenames.
#  2. Criterion 6 failed for CZE_BK1 (median 0.0073) and CZE_RAJ (0.0066),
#     ~66x below the native p5 of 0.48. Is that a unit/area error or a real
#     low-flow stand? Compare against CZ sites that passed.

set -euo pipefail
REPO=/scratch/tmp/yluo2/gsv
cd "$REPO"
source "$REPO/.venv/bin/activate"
export PYTHONPATH="$REPO"

python - <<'PY'
import json
from pathlib import Path

import numpy as np
import pandas as pd

out = Path("data/raw/0.1.5/0.1.5/csv/sapwood_combined")
manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
KINDS = ("sapf_data", "plant_md", "env_data", "env_md", "site_md",
         "stand_md", "species_md", "sapf_flags", "env_flags")

print("=" * 72)
print("Q1  Criterion 1b re-tested with EXACT filenames")
print("=" * 72)
suspects = ["CHE_DAV", "CZE_RAJ", "FIN_HYY", "FRA_HES", "SWE_NOR"]
for site in suspects:
    declared = sorted(manifest["sites"][site]["kinds"])
    exact = sorted(k for k in KINDS if (out / f"{site}_{k}.csv").exists())
    globbed = sorted({p.name[len(site) + 1: -4] for p in out.glob(f"{site}_*.csv")})
    print(f"\n{site}")
    print(f"  manifest : {declared}")
    print(f"  exact    : {exact}   -> {'MATCH' if declared == exact else 'MISMATCH'}")
    print(f"  globbed  : {len(globbed)} entries (over-matches prefix siblings)")
    sibs = sorted({p.name.split('_sapf_data')[0] for p in out.glob(f"{site}_*_sapf_data.csv")})
    if sibs:
        print(f"  siblings the glob swept in: {sibs}")

bad = []
for site, entry in manifest["sites"].items():
    exact = sorted(k for k in KINDS if (out / f"{site}_{k}.csv").exists())
    if sorted(entry["kinds"]) != exact:
        bad.append(site)
print(f"\nAll {len(manifest['sites'])} sites, exact-filename test: {len(bad)} mismatch(es) {bad}")

print()
print("=" * 72)
print("Q2  Why are CZE_BK1 and CZE_RAJ ~66x below the native p5?")
print("=" * 72)

src = {"CZE_BK1": "plant", "CZE_RAJ": "plant", "CZE_STN": "plant",
       "CZE_LNZ": "plant", "CZE_RAJ_RAJ": "plant", "CZE_BIK": "plant"}
internal = {"CZE_BK1", "CZE_RAJ", "CZE_STN", "CZE_LNZ"}

rows = []
for site in src:
    md_path = Path("data/raw/0.1.5/0.1.5/csv/plant_asw_filled") / f"{site}_plant_md.csv"
    if not md_path.exists():
        print(f"  {site}: no filled metadata")
        continue
    md = pd.read_csv(md_path)

    raw_dir = (Path("Sapflow-internal/Sapflow_SAPFLUXNET_format_unitcon/plant")
               if site in internal else Path("data/raw/0.1.5/0.1.5/csv/plant"))
    raw = pd.read_csv(raw_dir / f"{site}_sapf_data.csv", low_memory=False)
    trees = [c for c in raw.columns if "TIMESTAMP" not in c.upper()]
    raw_med = raw[trees].apply(pd.to_numeric, errors="coerce").stack().median()

    conv = pd.read_csv(out / f"{site}_sapf_data.csv", low_memory=False)
    ctrees = [c for c in conv.columns if "TIMESTAMP" not in c.upper()]
    conv_med = conv[ctrees].stack().median()

    asw = pd.to_numeric(md["pl_sapw_area_filled"], errors="coerce")
    dbh = pd.to_numeric(md.get("pl_dbh"), errors="coerce")
    basal = np.pi * (dbh / 2.0) ** 2 if dbh is not None else pd.Series(dtype=float)
    srcflags = md["pl_sapw_area_source"].value_counts().to_dict() if "pl_sapw_area_source" in md else {}

    rows.append({
        "site": site,
        "origin": "internal" if site in internal else "archive",
        "n_trees": len(trees),
        "raw_median_cm3_h": round(float(raw_med), 2) if pd.notna(raw_med) else None,
        "asw_median_cm2": round(float(asw.median()), 1),
        "asw_min": round(float(asw.min()), 1),
        "asw_max": round(float(asw.max()), 1),
        "dbh_median_cm": round(float(dbh.median()), 1) if dbh is not None and dbh.notna().any() else None,
        "basal_median_cm2": round(float(basal.median()), 1) if len(basal) and basal.notna().any() else None,
        "asw_over_basal": round(float((asw / basal).median()), 3) if len(basal) and basal.notna().any() else None,
        "converted_median": round(float(conv_med), 4),
        "asw_source": srcflags,
    })

df = pd.DataFrame(rows)
pd.set_option("display.width", 200, "display.max_columns", 30)
print(df.to_string(index=False))

print("\nReading: if converted_median is low because asw is too BIG, asw_over_basal")
print("will be implausible (>1 means sapwood exceeds the whole stem cross-section).")
print("If asw looks sane, the raw flux itself is small and the site is genuinely low-flow.")

print()
print("Native reference band, for scale:")
natives = [s for s, e in manifest["sites"].items() if e["route"] == "native"][:40]
meds = []
for s in natives:
    f = pd.read_csv(out / f"{s}_sapf_data.csv", low_memory=False)
    t = [c for c in f.columns if "TIMESTAMP" not in c.upper()]
    if t:
        m = f[t].stack().median()
        if pd.notna(m):
            meds.append(float(m))
print(f"  {len(meds)} native sites sampled: p5={np.percentile(meds,5):.3f} "
      f"median={np.median(meds):.3f} p95={np.percentile(meds,95):.3f} min={min(meds):.4f}")
low = sorted(m for m in meds if m < 0.01)
print(f"  native sites ALSO below 0.01: {len(low)} {[round(x,5) for x in low[:10]]}")
PY

echo
echo "=== DONE ==="
