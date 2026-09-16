"""Name the run that produced a preprocessed cache, and refuse a foreign one.

``preprocess_products`` skips any ``{key}_preprocessed.nc`` that already exists
and ``_load_preprocessed`` globs whatever is in the directory, so a cache built
from one prediction is silently reused for another and the figures carry the
wrong model's name. The sidecar written here is the same device the Stage-4
merge uses: a record beside the data that outranks the filename.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PROVENANCE_NAME = "run_provenance.json"

# The fields that make two caches the same cache. `prediction_files` and
# `created` are recorded but not compared: re-running a year rewrites the file
# without changing what the cache means.
IDENTITY_FIELDS = ("sap_source", "run_id", "period_start", "period_end")


@dataclass(frozen=True)
class RunProvenance:
    """Which prediction, over which period, produced a preprocessed cache."""

    sap_source: str  # 'parquet' (legacy stage2) or 'serving' (store_v1 cube)
    run_id: str
    period_start: str
    period_end: str
    prediction_files: tuple = field(default_factory=tuple)

    def identity(self) -> dict:
        return {name: getattr(self, name) for name in IDENTITY_FIELDS}


def write_provenance(directory: Path, provenance: RunProvenance) -> Path:
    """Write the sidecar into ``directory``. Returns its path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PROVENANCE_NAME

    payload = asdict(provenance)
    payload["prediction_files"] = [str(p) for p in provenance.prediction_files]
    payload["created"] = datetime.now().isoformat(timespec="seconds")

    path.write_text(json.dumps(payload, indent=2))
    logger.info(f"Wrote provenance: {path}")
    return path


def read_provenance(directory: Path) -> RunProvenance | None:
    """Return the sidecar in ``directory``, or None if there is none."""
    path = Path(directory) / PROVENANCE_NAME
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return RunProvenance(
        sap_source=payload["sap_source"],
        run_id=payload["run_id"],
        period_start=payload["period_start"],
        period_end=payload["period_end"],
        prediction_files=tuple(payload.get("prediction_files", ())),
    )


def assert_provenance_matches(directory: Path, provenance: RunProvenance) -> None:
    """Raise if ``directory`` holds a cache built by a different run.

    An empty directory claims nothing and is accepted -- that is a first run.
    """
    stored = read_provenance(directory)
    if stored is None:
        return

    differing = {
        name: (stored.identity()[name], provenance.identity()[name])
        for name in IDENTITY_FIELDS
        if stored.identity()[name] != provenance.identity()[name]
    }
    if not differing:
        return

    detail = "; ".join(f"{name}: cache has {was!r}, run asks for {now!r}" for name, (was, now) in differing.items())
    raise ValueError(
        f"{directory} holds a preprocessed cache from a different run -- {detail}. "
        "Use a fresh --out-tag, or delete the cache; do not re-use it."
    )
