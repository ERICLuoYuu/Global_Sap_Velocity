#!/bin/bash
#SBATCH --job-name=sapcomb_build
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/sapcomb_build_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/sapcomb_build_%j.err

# Build csv/sapwood_combined: every site we hold, at sapwood scale.
#
# Expected product (see .claude/plan/sapwood-combined-plan.md):
#   226 sites written, 181 native + 45 converted, 1986 files, ~7.1 GB
#   ESP_GDN skipped -- no tree with a usable sapwood area
#
# --link hardlinks the ~6.4 GB that is byte-identical to its source, falling
# back to a copy per-file when the filesystem refuses. NOTE: a hardlinked file
# edited in place would also change the source. The combined corpus is a
# generated artefact and must be rebuilt rather than edited.

set -euo pipefail

REPO=/scratch/tmp/yluo2/gsv
cd "$REPO"
mkdir -p logs

source "$REPO/.venv/bin/activate"
export PYTHONPATH="$REPO"

OUT="$REPO/data/raw/0.1.5/0.1.5/csv/sapwood_combined"

echo "=== interpreter ==="
python --version
echo

echo "=== free space before ==="
df -h /scratch | tail -1
echo

echo "=== dry run first: routes and size, nothing written ==="
python -m src.sapwood_combined.build_sapwood_combined --dry-run
echo

echo "=== build ==="
python -m src.sapwood_combined.build_sapwood_combined --force --link
echo

echo "=== verification against the acceptance criteria ==="
python - <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

out = Path("data/raw/0.1.5/0.1.5/csv/sapwood_combined")
manifest = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
failures = []


def check(label, got, want):
    ok = got == want
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}: {got!r}" + ("" if ok else f" (expected {want!r})"))
    if not ok:
        failures.append(label)


s = manifest["summary"]
print("Criterion 1 -- site and file counts")
check("sites written", s["sites_written"], 226)
check("native", s["native"], 181)
check("converted", s["converted"], 45)
check("files written", s["files_written"], 1986)

print("Criterion 1b -- per-site kinds match what is on disk")
# Exact filenames, NOT glob("{site}_*.csv"). The output directory is flat and
# holds both corpora, so a site whose code is a prefix of another's -- CHE_DAV
# and CHE_DAV_SEE, SWE_NOR and its eight SWE_NOR_ST* siblings -- would sweep in
# its neighbours' files and report a false mismatch. (This check did exactly
# that on the first build, flagging 5 correct sites.)
KINDS = (
    "sapf_data", "plant_md", "env_data", "env_md", "site_md",
    "stand_md", "species_md", "sapf_flags", "env_flags",
)
mismatched = []
for site, entry in manifest["sites"].items():
    on_disk = sorted(k for k in KINDS if (out / f"{site}_{k}.csv").exists())
    if sorted(entry["kinds"]) != on_disk:
        mismatched.append(site)
check("sites whose kinds disagree with disk", len(mismatched), 0)
if mismatched:
    print("        first few:", mismatched[:5])

print("Criterion 2 -- every plant_md declares sapwood scale")
bad_units = []
for md_path in out.glob("*_plant_md.csv"):
    units = set(pd.read_csv(md_path, usecols=["pl_sap_units"])["pl_sap_units"])
    if units != {"cm3 cm-2 h-1"}:
        bad_units.append((md_path.name, units))
check("plant_md files not at sapwood scale", len(bad_units), 0)
if bad_units:
    print("        first few:", bad_units[:5])

print("Criterion 3b -- ESP_GDN skipped, nothing else missing")
check("ESP_GDN in skipped", "ESP_GDN" in manifest["skipped"], True)
check("number skipped", len(manifest["skipped"]), 1)

print("Criterion 5b -- flags align with data for every site that has them")
misaligned = []
for site in manifest["sites"]:
    f = out / f"{site}_sapf_flags.csv"
    if not f.exists():
        continue
    d = out / f"{site}_sapf_data.csv"
    dc = [c for c in pd.read_csv(d, nrows=0).columns if "TIMESTAMP" not in c.upper()]
    fc = [c for c in pd.read_csv(f, nrows=0).columns if "TIMESTAMP" not in c.upper()]
    if dc != fc:
        misaligned.append(site)
check("sites whose flags and data disagree", len(misaligned), 0)
if misaligned:
    print("        first few:", misaligned[:5])

print("Criterion 6 -- converted magnitudes inside the band the native sites define")
# NOT the all-hours median. Sap flow is zero all night, and the fraction of a
# record that is night varies by site, latitude and how much winter was logged.
# Once a site passes ~40% exact zeros the 50th percentile falls INTO the zero
# block, so the all-hours median becomes a step function of the zero fraction
# rather than a measure of flux. That is not hypothetical: it is what the first
# version of this check did, and it flagged CZE_BK1 (43.7% zeros) and CZE_RAJ
# (40.6%) while passing CZE_BIK (31.2%) -- sites whose daytime magnitudes are
# ordinary. See the deviations log, 2026-08-16.
#
# So screen on two statistics night cannot move, and calibrate the band against
# the 181 NATIVE sites, which are measured rather than derived and therefore
# define what this quantity actually looks like.
import numpy as np


def magnitudes(site):
    frame = pd.read_csv(out / f"{site}_sapf_data.csv", low_memory=False)
    trees = [c for c in frame.columns if "TIMESTAMP" not in c.upper()]
    if not trees:
        return None
    vals = frame[trees].apply(pd.to_numeric, errors="coerce").stack().dropna()
    if vals.empty:
        return None
    v = vals.to_numpy(dtype=float)
    pos = v[v > 0]
    if not pos.size:
        return None
    return float(np.median(pos)), float(np.percentile(v, 90))


native_stats, converted_stats = {}, {}
for site, entry in manifest["sites"].items():
    m = magnitudes(site)
    if m is None:
        continue
    (native_stats if entry["route"] == "native" else converted_stats)[site] = m

ref = np.array([m[0] for m in native_stats.values()])
lo, hi = float(np.percentile(ref, 1)), float(np.percentile(ref, 99))
print(f"  native reference (n={len(native_stats)}): median(>0) 1st pct={lo:.4f}, 99th pct={hi:.4f}")

impossible, outliers = [], []
for site, (nz, p90) in converted_stats.items():
    if not (1e-4 <= nz <= 1e4):
        impossible.append((site, round(nz, 6)))
    elif not (lo <= nz <= hi):
        outliers.append((site, round(nz, 4), round(p90, 3)))
check("converted sites outside the impossible band [1e-4, 1e4]", len(impossible), 0)
if impossible:
    print("        offenders:", impossible)
if outliers:
    print(f"  [WARN] {len(outliers)} converted site(s) outside the native 1-99 pct band:")
    for site, nz, p90 in outliers:
        print(f"           {site}: median(>0)={nz} p90={p90}")
    print("         Not a build failure -- a low-flow stand is allowed to exist.")
else:
    print("  [OK ] every converted site's median(>0) lies inside the native 1-99 pct band")

print("Manifest provenance")
print("  asw_model:", manifest["asw_model"].get("taxon_refinement"), "offsets:", manifest["asw_model"].get("taxon_offsets"))
print("  stem_overlaps recorded:", len(manifest["stem_overlaps"]))
print("  sites without flags:", manifest["flags_available"]["sites_without_flags"])
print("  route counts:", Counter(e["route"] for e in manifest["sites"].values()))

if failures:
    print(f"\nFAILED {len(failures)} criteria: {failures}")
    sys.exit(1)
print("\nAll acceptance criteria met.")
PY

echo
echo "=== on-disk size ==="
du -sh "$OUT"
echo "file count: $(ls "$OUT" | wc -l)"
echo
echo "=== DONE ==="
