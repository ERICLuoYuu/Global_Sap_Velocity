#!/bin/bash
#SBATCH --job-name=extradapply
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/tmp/yluo2/gsv/logs/extradapply_%j.out
#SBATCH --error=/scratch/tmp/yluo2/gsv/logs/extradapply_%j.err

# Write ext_rad into the 24 internal env files, then prove the change is
# confined to the new column.
set -uo pipefail
cd /scratch/tmp/yluo2/gsv || exit 1
source .venv/bin/activate || exit 1

CORPUS=/scratch/tmp/yluo2/Global_Sap_Velocity/data/raw/0.1.5/0.1.5/csv/sapwood_combined
BACKUP=/scratch/tmp/yluo2/Global_Sap_Velocity/data/raw/0.1.5/0.1.5/csv/sapwood_combined_ext_rad_backup

# A record of the 202 external files as they stand now. They are never opened
# for writing, but "never written" is a claim about the code; this makes it a
# measurement.
echo "### fingerprinting the 202 external env files before the change"
python - <<'PY'
import hashlib, json
from pathlib import Path
corpus = Path("/scratch/tmp/yluo2/Global_Sap_Velocity/data/raw/0.1.5/0.1.5/csv/sapwood_combined")
digests = {}
for path in sorted(corpus.glob("*_env_data.csv")):
    with path.open() as fh:
        # rstrip is load-bearing: the header's LAST field carries the trailing
        # newline, so `"ext_rad" in line.split(",")` is False for every file
        # whose ext_rad column sits last -- which is all of them. Without it
        # this loop fingerprints nothing and the comparison below then reports
        # "0 / 0 unchanged", which reads exactly like a pass. It shipped that
        # way once; a zero denominator is not a green check.
        if "ext_rad" not in fh.readline().rstrip("\r\n").split(","):
            continue
    digests[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
if not digests:
    raise SystemExit("FATAL: fingerprinted 0 external files -- the check would be vacuous")
Path("/scratch/tmp/yluo2/gsv/logs/external_env_before.json").write_text(json.dumps(digests))
print(f"  fingerprinted {len(digests)} external files")
PY

echo
echo "### applying"
python .claude/plan/backfill_ext_rad.py --apply --backup-dir "$BACKUP"
RC=$?
echo "### backfill exit: $RC"
[ "$RC" -ne 0 ] && { echo "=== APPLY FAILED ==="; exit 1; }

echo
echo "### post-checks"
python - <<'PY'
import hashlib, json, sys
from pathlib import Path

corpus = Path("/scratch/tmp/yluo2/Global_Sap_Velocity/data/raw/0.1.5/0.1.5/csv/sapwood_combined")
backup = Path("/scratch/tmp/yluo2/Global_Sap_Velocity/data/raw/0.1.5/0.1.5/csv/sapwood_combined_ext_rad_backup")
bad = 0

# 1. every site now carries the column
missing = []
for path in sorted(corpus.glob("*_env_data.csv")):
    with path.open() as fh:
        if "ext_rad" not in fh.readline().rstrip("\n").split(","):
            missing.append(path.name)
total = len(list(corpus.glob("*_env_data.csv")))
print(f"  env files carrying ext_rad: {total - len(missing)} / {total}")
if missing:
    print(f"  STILL MISSING: {missing}")
    bad += 1

# 2. the 202 external files are byte-for-byte what they were
before = json.loads(Path("/scratch/tmp/yluo2/gsv/logs/external_env_before.json").read_text())
changed = [n for n, d in before.items() if hashlib.sha256((corpus / n).read_bytes()).hexdigest() != d]
print(f"  external files unchanged: {len(before) - len(changed)} / {len(before)}")
if changed:
    print(f"  CHANGED: {changed[:10]}")
    bad += 1

# 3. the rewritten files differ from their backup ONLY by the appended field.
#    Stripping the last comma-separated field must reproduce the original bytes
#    exactly -- this is what proves no other column was silently reformatted.
checked = 0
for original in sorted(backup.glob("*_env_data.csv")):
    current = corpus / original.name
    with original.open() as a, current.open() as b:
        for lineno, (old, new) in enumerate(zip(a, b), start=1):
            old_s, new_s = old.rstrip("\r\n"), new.rstrip("\r\n")
            if new_s.rsplit(",", 1)[0] != old_s:
                print(f"  {original.name}:{lineno} differs beyond the new column")
                print(f"    was: {old_s[:90]}")
                print(f"    now: {new_s[:90]}")
                bad += 1
                break
        else:
            extra_old = sum(1 for _ in a)
            extra_new = sum(1 for _ in b)
            if extra_old or extra_new:
                print(f"  {original.name}: row count changed ({extra_old} vs {extra_new} trailing)")
                bad += 1
                continue
            checked += 1
print(f"  backfilled files whose other columns are byte-identical: {checked} / {len(list(backup.glob('*_env_data.csv')))}")

sys.exit(1 if bad else 0)
PY
POST=$?
echo "### post-check exit: $POST"
[ "$POST" -ne 0 ] && { echo "=== POST-CHECKS FAILED ==="; exit 1; }
echo "=== EXT_RAD BACKFILL COMPLETE ==="
