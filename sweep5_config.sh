# Shared dataset table for the five-config FFS + training sweep.
# Sourced by job_sweep5_cache.sh, job_sweep5_select.sh, job_sweep5_train.sh.
#
# ONE definition, three consumers. The prior guard failures in this repo came
# from a site code drifting in one copy of a table and not the others, where the
# shape and the summary line stayed identical so the gate still PASSed. Keeping
# the table in a single file makes that class of drift impossible.
#
# EXP_* are the flags each dataset's RUN_PROVENANCE.json must carry. They are
# asserted EXACTLY (all four, every stage), so a task cannot silently operate on
# a sibling dataset -- the failure mode that produces a clean-looking result for
# the wrong design.

# Which merged tree this sweep reads. Overridable so a DEU_HOH-corrected rebuild
# can run beside the original instead of clobbering it:
#   sbatch --export=ALL,SWEEP_NAME=sweep_20260902_hohfix job_sweep5_select.sh
SWEEP_NAME="${SWEEP_NAME:-sweep_20260902}"
SWEEP_ROOT=outputs/processed_data/sapwood_combined/merged/$SWEEP_NAME

# Artifact names carry the dataset tag. Every guard failure in this repo has come
# from an output path that did not encode the config it was built from -- two
# runs land on one name, the shape and the summary line match, and the gate still
# PASSes. A corrected run must be unmistakable from its filename alone.
# A second tag for the SELECTION DESIGN, independent of the dataset tag above.
# The merged tree can stay fixed while the selection changes (mandatory set,
# k_features), and those runs must not land on one name either:
#   sbatch --export=ALL,SWEEP_NAME=sweep_20260902_hohfix,FFS_TAG=_sm job_sweep5_select.sh
FFS_TAG="${FFS_TAG:-}"

# Spatial grouping, for the cache AND the trainer together. They MUST match: the
# cache's `groups` array IS the FFS cross-validation design, so an FFS run
# grouped one way and scored by a trainer grouped the other measures nothing.
#
#   grid  co-located sites share a fold. The honest one, and what prediction at
#         an unseen cell actually faces.
#   site  each site its own fold. Leaky by co-location (+0.145 pooled R2 on
#         identical data), so it is an UPPER bound, not the headline. Run it to
#         PRICE the leakage on this corpus, not to quote it.
#
#   sbatch --export=ALL,GROUPING=site ... job_sweep5_cache.sh
GROUPING="${GROUPING:-grid}"
case "$GROUPING" in
  grid) GROUP_TAG=grid005 ; CACHE_METHOD=grid    ; TRAIN_METHOD=grid    ;;
  site) GROUP_TAG=site    ; CACHE_METHOD=default ; TRAIN_METHOD=default ;;
  *)    echo "FATAL: GROUPING must be 'grid' or 'site', got '$GROUPING'" >&2
        return 1 2>/dev/null || exit 1 ;;
esac


case "$SWEEP_NAME" in
  *_hohfix) SWEEP_TAG=_hohfix ;;
  *)        SWEEP_TAG="" ;;
esac

sweep5_config () {
    local t="$1"
    case "$t" in
      1) NAME=fullyear_fullday_tx; SUBDIR=fullyear_fullday_tx
         EXP_GS=false; EXP_DAY=false; EXP_DAYSAP=false; EXP_TX=true ;;
      2) NAME=fullyear_daytime;    SUBDIR=fullyear_daytime
         EXP_GS=false; EXP_DAY=true;  EXP_DAYSAP=false; EXP_TX=false ;;
      3) NAME=fullyear_daytime_tx; SUBDIR=fullyear_daytime_tx
         EXP_GS=false; EXP_DAY=true;  EXP_DAYSAP=false; EXP_TX=true ;;
      4) NAME=gs_daytime_tx;       SUBDIR=gs_daytime_tx/growing_season
         EXP_GS=true;  EXP_DAY=true;  EXP_DAYSAP=false; EXP_TX=true ;;
      5) NAME=gs_daysap_tx;        SUBDIR=gs_daysap_tx/growing_season
         EXP_GS=true;  EXP_DAY=false; EXP_DAYSAP=true;  EXP_TX=true ;;
      *) echo "FATAL: bad task id '$t' (expected 1-5)"; return 1 ;;
    esac
    DATA_DIR="$SWEEP_ROOT/$SUBDIR"
    # CACHE_DIR never carries FFS_TAG. The cache holds the feature MATRIX, which
    # depends only on the merged tree and the loader's column list -- not on the
    # mandatory/candidate split or k_features, both of which are read at
    # selection time. So a re-selection with a different design reuses the cache
    # and writes its outputs elsewhere.
    CACHE_DIR="outputs/forward_selection/sweep5_${NAME}_${GROUP_TAG}${SWEEP_TAG}"
    FFS_OUT="outputs/forward_selection/sweep5_${NAME}_${GROUP_TAG}${SWEEP_TAG}${FFS_TAG}"
    RUN_ID="sweep5_${NAME}_ffs_${GROUP_TAG}${SWEEP_TAG}${FFS_TAG}"
    return 0
}

# Assert the dataset on disk is the one this task was asked for.
# Every flag is checked, including the ones expected to be False -- a gate that
# only checks the True flags cannot tell dataset 3 from dataset 4.
sweep5_assert_provenance () {
    python - "$1" "$2" "$3" "$4" "$5" "$6" <<'PYEOF'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
name = sys.argv[2]
want = {
    "growing_season_only":     sys.argv[3] == "true",
    "daytime_only":            sys.argv[4] == "true",
    "daytime_sap_only":        sys.argv[5] == "true",
    "apply_treatment_filter":  sys.argv[6] == "true",
}
prov = d / "RUN_PROVENANCE.json"
if not prov.exists():
    sys.exit("  GATE FAIL: no RUN_PROVENANCE.json under %s" % d)
got = json.load(open(prov))
flags = got["flags"]
if got.get("config_name") != name:
    sys.exit("  GATE FAIL: provenance says config_name=%r, task wants %r"
             % (got.get("config_name"), name))
bad = {k: (flags.get(k), v) for k, v in want.items() if flags.get(k) is not v}
if bad:
    for k, (g, w) in sorted(bad.items()):
        print("  GATE FAIL: %s=%r, want %r" % (k, g, w))
    sys.exit(1)
if flags.get("apply_flo2019_correction") is not False:
    sys.exit("  GATE FAIL: flo2019 calibration is ON; the sweep ran with it off")
print("  GATE PASS: %s  %s" % (name, {k: flags[k] for k in sorted(want)}))
PYEOF
}
