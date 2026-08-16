"""Build ``csv/sapwood_combined``: every site we hold, at sapwood scale.

    python -m src.sapwood_combined.build_sapwood_combined --dry-run
    python -m src.sapwood_combined.build_sapwood_combined --force

Reads four source directories (``csv/{plant,sapwood}`` and the internal corpus's
two flavours) plus the two ``*_asw_filled/`` directories the imputer writes, and
produces one flat directory of seven files per site plus ``MANIFEST.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from src.sapwood_combined.assembly import FILE_KINDS, assemble_site, prune_unclaimed
from src.sapwood_combined.manifest import build_manifest, write_manifest
from src.sapwood_combined.membership import (
    EXTERNAL,
    INTERNAL,
    Corpus,
    resolve_routes,
    summarise_routes,
)

DEFAULT_SUBDIR = "sapwood_combined"
FIT_REPORT = "asw_allometry_fit_report.json"


#: Where the internal corpus lives relative to the repository root, when no
#: other answer is available.
_INTERNAL_SUBPATH = ("Sapflow-internal", "Sapflow_SAPFLUXNET_format_unitcon")


def _internal_root(paths, override: Path | None = None) -> Path:
    """Locate the internal corpus without depending on PathConfig's vintage.

    ``internal_metadata_dir`` is a recent addition to ``path_config.py`` and is
    absent from older checkouts -- reading it unconditionally made this CLI die
    with an ``AttributeError`` on the very machine the product is built on. The
    attribute is a convenience, not the source of truth: the internal corpus
    sits at a fixed path inside the repository, so it can always be derived.
    """
    if override is not None:
        return Path(override)
    declared = getattr(paths, "internal_metadata_dir", None)
    if declared is not None:
        return Path(declared).parent
    return Path(__file__).resolve().parents[2].joinpath(*_INTERNAL_SUBPATH)


def _corpora(paths, internal_root: Path | None = None, skip_internal: bool = False) -> list[Corpus]:
    csv_root = Path(paths.raw_csv_dir).parent
    corpora = [Corpus(origin=EXTERNAL, plant_dir=csv_root / "plant", sapwood_dir=csv_root / "sapwood")]
    if not skip_internal:
        root = _internal_root(paths, internal_root)
        corpora.append(Corpus(origin=INTERNAL, plant_dir=root / "plant", sapwood_dir=root / "sapwood"))
    return corpora


def _asw_model(plant_asw_dir: Path) -> dict:
    """Name the allometric model behind the areas we are about to divide by.

    The fit statistics are identical with or without the taxon refinement -- it
    is applied at prediction time and never moves the fitted line -- so the
    presence of the ``taxon_offsets`` block is the only thing that distinguishes
    them. Recorded because the division destroys the evidence.
    """
    report = Path(plant_asw_dir) / FIT_REPORT
    if not report.exists():
        return {"available": False, "reason": f"no {FIT_REPORT} beside the filled metadata"}
    try:
        doc = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"available": False, "reason": f"unreadable fit report: {exc}"}

    taxon = doc.get("taxon_offsets")
    fit = doc.get("fit", {}) or {}
    return {
        "available": True,
        "taxon_refinement": taxon is not None,
        "taxon_offsets": len(taxon.get("offsets", [])) if isinstance(taxon, dict) else 0,
        "n": fit.get("n"),
        "slope": fit.get("slope"),
        "intercept": fit.get("intercept"),
        "r2": fit.get("r2"),
    }


def run(
    base_data_dir: str = "./data",
    base_output_dir: str = "./outputs",
    output_subdir: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    link: bool = False,
    skip_internal: bool = False,
    internal_root: Path | None = None,
) -> int:
    """Assemble the combined corpus. Returns the number of sites written."""
    from path_config import PathConfig

    paths = PathConfig(scale="sapwood", base_data_dir=base_data_dir, base_output_dir=base_output_dir)
    csv_root = Path(paths.raw_csv_dir).parent
    plant_asw = csv_root / "plant_asw_filled"
    sapwood_asw = csv_root / "sapwood_asw_filled"
    out_dir = csv_root / (output_subdir or DEFAULT_SUBDIR)

    corpora = _corpora(paths, internal_root=internal_root, skip_internal=skip_internal)
    if skip_internal:
        print("[Corpora] internal corpus EXCLUDED (--skip-internal); archive sites only")
    routes = resolve_routes(corpora, plant_asw_dir=plant_asw, sapwood_asw_dir=sapwood_asw)
    counts = summarise_routes(routes)
    print("[Routes] {} site(s): {} native, {} converted".format(len(routes), counts["native"], counts["converted"]))
    for origin in (EXTERNAL, INTERNAL):
        print(
            "  {:<9s} native={:<4d} converted={}".format(
                origin,
                counts.get(f"{origin}_native", 0),
                counts.get(f"{origin}_converted", 0),
            )
        )

    model = _asw_model(plant_asw)
    if model.get("available"):
        print(
            "[ASW model] taxon_refinement={}  offsets={}  n={}".format(
                model.get("taxon_refinement"), model.get("taxon_offsets"), model.get("n")
            )
        )
    else:
        print("[ASW model] UNKNOWN -- {}".format(model.get("reason")))

    if dry_run:
        # Exact filenames, not glob(f"{r.site}_*.csv"). A site whose code is a
        # prefix of another's would sweep in its neighbour's files and inflate
        # the estimate -- the same over-match that made the build's criterion 1b
        # flag five correct sites. No such pair exists inside a single source
        # directory today (measured: 0 across all four), which is exactly why
        # this would go unnoticed until one does.
        candidates = (r.source_dir / f"{r.site}_{k}.csv" for r in routes for k in FILE_KINDS)
        bytes_in = sum(f.stat().st_size for f in candidates if f.exists())
        print(f"[Dry run] would write {len(routes)} site(s), about {bytes_in / 1e9:.2f} GB, to {out_dir}")
        print("[Dry run] nothing written")
        return 0

    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise SystemExit(
            f"{out_dir} exists and is not empty. Pass --force to rebuild it, or choose another --output-subdir."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    outcomes = []
    for i, route in enumerate(routes, start=1):
        outcome = assemble_site(route, out_dir, link=link)
        outcomes.append(outcome)
        tag = "SKIP" if not outcome.was_written else route.route[:4].upper()
        print(
            "  [{:>3d}/{}] {:<5s} {:<20s} trees {}->{}{}".format(
                i,
                len(routes),
                tag,
                route.site,
                outcome.n_trees_in,
                outcome.n_trees_out,
                f"  ({outcome.skipped_reason})" if outcome.skipped_reason else "",
            )
        )

    # Before the manifest, so the manifest is still the last thing written and
    # its absence still marks an interrupted run. A --force rebuild writes over
    # the previous product but never removes what it does not write, so without
    # this the directory keeps every site any earlier build put there while the
    # manifest describes only this one.
    pruned = prune_unclaimed(out_dir, outcomes)
    if pruned:
        print(f"\n[Pruned] {len(pruned)} file(s) left by an earlier build that this run does not write")
        for name in pruned[:10]:
            print(f"          {name}")
        if len(pruned) > 10:
            print(f"          ... and {len(pruned) - 10} more")

    manifest = build_manifest(outcomes, asw_model=model, corpora_included=[c.origin for c in corpora])
    path = write_manifest(out_dir, manifest)

    s = manifest["summary"]
    print(
        "\n[Summary] written={}  skipped={}  native={}  converted={}".format(
            s["sites_written"], s["sites_skipped"], s["native"], s["converted"]
        )
    )
    print("          trees={}  dropped={}  files={}".format(s["trees_written"], s["trees_dropped"], s["files_written"]))
    print(f"[Written] {out_dir}")
    print(f"[Written] {path}")
    return int(s["sites_written"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-data-dir", default="./data")
    parser.add_argument("--base-output-dir", default="./outputs")
    parser.add_argument(
        "--output-subdir",
        default=None,
        help=f"Directory name under csv/ to write (default: {DEFAULT_SUBDIR}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve routes and report the size estimate without writing anything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild into a non-empty output directory.",
    )
    parser.add_argument(
        "--skip-internal",
        action="store_true",
        help=(
            "Build from the SAPFLUXNET archive alone, omitting the internal European "
            "sites. Recorded in the manifest so a partial build cannot be mistaken "
            "for a full one."
        ),
    )
    parser.add_argument(
        "--internal-root",
        default=None,
        type=Path,
        help=(
            "Directory holding the internal corpus's plant/ and sapwood/ flavours. "
            "Defaults to Sapflow-internal/Sapflow_SAPFLUXNET_format_unitcon/ inside "
            "the repository."
        ),
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help=(
            "Hard-link unchanged files instead of copying, where the filesystem allows "
            "it. About 3 GB of the product is identical to its source."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run(
        base_data_dir=args.base_data_dir,
        base_output_dir=args.base_output_dir,
        output_subdir=args.output_subdir,
        dry_run=args.dry_run,
        force=args.force,
        link=args.link,
        skip_internal=args.skip_internal,
        internal_root=args.internal_root,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
