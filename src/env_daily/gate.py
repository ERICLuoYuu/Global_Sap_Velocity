"""Refuse a daily tree built before the env daily rule, interrupted, or whose radiation regressed.

    python -m src.env_daily.gate CONFIG_DIR

``CONFIG_DIR`` is the merged config directory, the parent of ``daily/`` -- the directory
``assert_dataset_contract`` reads, with its precedence: the ``feature_contract`` block of ``RUN_PROVENANCE.json``
when it carries one, else the merge's sidecar. The sweep's cache and train jobs run it beside their contract gate
(plan WP-B4, failure mode F21), so a tree built under the old env statistics cannot be trained as if it were
fixed. Exit 0 with a PASS line; exit 1 naming every reason.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from src.clock_offsets.provenance import provenance_errors
from src.env_daily.invariants import KT_BASELINE_TOTAL, radiation_gate
from src.env_daily.support import RULE_VERSION
from src.feature_contract.definitions import PROVENANCE_FILE, PROVENANCE_KEY, SIDECAR_FILE


def read_block(config_dir) -> Optional[dict]:
    """The tree's feature-contract block, or None when neither file carries one."""
    directory = Path(config_dir)
    block = None
    provenance = directory / PROVENANCE_FILE
    if provenance.exists():
        with open(provenance) as handle:
            loaded = json.load(handle)
        block = loaded.get(PROVENANCE_KEY) if isinstance(loaded, dict) else None
    sidecar = directory / SIDECAR_FILE
    if not isinstance(block, dict) and sidecar.exists():
        with open(sidecar) as handle:
            block = json.load(handle)
    return block if isinstance(block, dict) else None


def tree_errors(block: Optional[dict]) -> List[str]:
    """Every reason the tree ``block`` describes fails the gate; empty when it passes."""
    if block is None:
        return [f"no feature-contract block: neither {PROVENANCE_FILE} nor {SIDECAR_FILE} carries one"]
    errors: List[str] = []
    rule = block.get("env_daily_rule")
    version = rule.get("version") if isinstance(rule, dict) else None
    if isinstance(version, bool) or not isinstance(version, int) or version < RULE_VERSION:
        errors.append(
            f"env_daily_rule version is {version!r}: the tree was built before the env daily rule "
            f"(version {RULE_VERSION}); rebuild it with the Stage-4 merge"
        )
    if not block.get("clock_registry_sha256"):
        errors.append("no clock_registry_sha256: the registry this tree's merge imported is unrecorded")
    # The hash above names what the MERGE read. Only this compares it with what the Stage-3 inputs were
    # actually moved by -- the merge runs every sweep and Stage 3 hardly ever, so the two drift apart in
    # the ordinary way of working, and registry_sha256(()) is a perfectly valid hash of nothing.
    errors += provenance_errors(block.get("clock_provenance"))
    if block.get("complete") is not True:
        errors.append("the block is not complete: the merge was interrupted, so its sites and kt counts are partial")
    kt = block.get("kt_exceedance")
    by_site = kt.get("by_site") if isinstance(kt, dict) else None
    if not isinstance(by_site, dict):
        errors.append("no kt_exceedance.by_site: the tree's daily sw_in was never checked against ext_rad")
    else:
        errors += radiation_gate(by_site)
    return errors


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.env_daily.gate", description=__doc__.splitlines()[0])
    parser.add_argument("config_dir", help="the merged config directory, the parent of daily/")
    args = parser.parse_args(argv)
    try:
        block = read_block(args.config_dir)
    except (OSError, ValueError) as exc:
        print(f"  ENV DAILY GATE FAIL: cannot read the provenance under {args.config_dir}: {exc}")
        return 1
    errors = tree_errors(block)
    for error in errors:
        print(f"  ENV DAILY GATE FAIL: {error}")
    if errors:
        return 1
    over = sum(int(days) for days in block["kt_exceedance"]["by_site"].values())
    clock = block["clock_provenance"]
    print(
        f"  ENV DAILY GATE PASS: env daily rule version {block['env_daily_rule']['version']}, clock registry "
        f"{str(block['clock_registry_sha256'])[:12]}, {len(clock['sites_corrected'])} of "
        f"{clock['n_sites_checked']} sites clock-corrected at Stage 3 and matching, {over} site-days with "
        f"daily sw_in above ext_rad (baseline {KT_BASELINE_TOTAL})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
