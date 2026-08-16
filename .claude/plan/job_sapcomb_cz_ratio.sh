#!/bin/bash
#SBATCH --job-name=cz_ratio
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/cz_ratio_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/cz_ratio_%j.err

# Decisive test for the CZE_BK1 / CZE_RAJ magnitude gap.
#
# The overlap sweep matched these internal sites to archive twins using
# CORRELATION, which is scale-invariant -- r=0.93 is equally consistent with
# identical series and with series differing by a constant factor. So a unit
# error between the two copies would have passed that sweep unnoticed.
#
# Here: pair the shared stems by pl_name, restrict to overlapping timestamps,
# and take the ratio. A tight ratio clustered on one number is a unit error.
# A scattered ratio means the series are genuinely different measurements.

set -euo pipefail
REPO=/scratch/tmp/yluo2/gsv
cd "$REPO"
source "$REPO/.venv/bin/activate"
export PYTHONPATH="$REPO"

python - <<'PY'
import numpy as np
import pandas as pd
from pathlib import Path

ARCHIVE = Path("data/raw/0.1.5/0.1.5/csv/plant")
INTERNAL = Path("Sapflow-internal/Sapflow_SAPFLUXNET_format_unitcon/plant")
PAIRS = [("CZE_BIK", "CZE_BK1"), ("CZE_RAJ_RAJ", "CZE_RAJ")]


def load(base, site):
    data = pd.read_csv(base / f"{site}_sapf_data.csv", low_memory=False)
    md = pd.read_csv(base / f"{site}_plant_md.csv", low_memory=False)
    names = dict(zip(md["pl_code"].astype(str), md["pl_name"].astype(str))) if "pl_name" in md else {}
    return data, names, md


for arch_site, int_site in PAIRS:
    print("=" * 72)
    print(f"{arch_site} (archive)  vs  {int_site} (internal)")
    print("=" * 72)

    a, a_names, a_md = load(ARCHIVE, arch_site)
    b, b_names, b_md = load(INTERNAL, int_site)

    print(f"  units  archive={set(a_md.get('pl_sap_units', ['?']))} "
          f"internal={set(b_md.get('pl_sap_units', ['?']))}")
    print(f"  orig   archive={set(a_md.get('pl_sap_units_orig', ['?']))} "
          f"internal={set(b_md.get('pl_sap_units_orig', ['?']))}")

    # pair stems by pl_name
    b_by_name = {v: k for k, v in b_names.items()}
    pairs = [(ac, b_by_name[nm]) for ac, nm in a_names.items() if nm in b_by_name]
    print(f"  stems paired by pl_name: {len(pairs)}")
    if not pairs:
        print("  (no shared names -- cannot compare)")
        continue

    a = a.set_index("TIMESTAMP")
    b = b.set_index("TIMESTAMP")
    shared_t = a.index.intersection(b.index)
    print(f"  shared timestamps: {len(shared_t)}")
    if len(shared_t) == 0:
        print("  (no shared timestamps -- sampling grids differ)")
        continue

    ratios = []
    for ac, bc in pairs:
        if ac not in a.columns or bc not in b.columns:
            continue
        x = pd.to_numeric(a.loc[shared_t, ac], errors="coerce")
        y = pd.to_numeric(b.loc[shared_t, bc], errors="coerce")
        m = x.notna() & y.notna() & (y.abs() > 1e-9) & (x.abs() > 1e-9)
        if m.sum() < 50:
            continue
        r = float((x[m] / y[m]).median())
        corr = float(np.corrcoef(x[m], y[m])[0, 1])
        ratios.append(r)
        print(f"    {ac:<28s} / {bc:<28s} n={int(m.sum()):>6d}  ratio={r:>9.3f}  r={corr:.4f}")

    if ratios:
        arr = np.array(ratios)
        print(f"\n  ratio across {len(arr)} stems: median={np.median(arr):.3f} "
              f"mean={arr.mean():.3f} sd={arr.std():.3f} "
              f"min={arr.min():.3f} max={arr.max():.3f}")
        cv = arr.std() / abs(arr.mean()) if arr.mean() else float("nan")
        print(f"  coefficient of variation = {cv:.3f}")
        print("  -> a tight CV (<0.2) around one value indicates a CONSTANT FACTOR,")
        print("     i.e. a unit/scaling error rather than different measurements.")
    print()
PY

echo "=== DONE ==="
