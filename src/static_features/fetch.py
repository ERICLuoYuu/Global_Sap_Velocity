"""Verified downloads through curl, because Zenodo gives 403 to urllib, requests and GDAL's HTTP reader."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess  # nosec B404
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from src.static_features.results import ExtractionFailure

CHUNK_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class Download:
    url: str
    name: str
    md5: str | None = None
    size: int | None = None

    def __post_init__(self) -> None:
        if self.md5 is None and self.size is None:
            raise ValueError(f"{self.name}: a download needs an md5 or a size to verify it against")


def _hash_of(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _problem_with(path: Path, item: Download) -> str | None:
    """What is wrong with a file, or None if it verifies. Size first, because hashing 218 GiB takes a while."""
    if item.size is not None and path.stat().st_size != item.size:
        return f"size {path.stat().st_size}, expected {item.size}"
    if item.md5 is not None:
        found = _hash_of(path, "md5")
        if found != item.md5:
            return f"md5 {found}, expected {item.md5}"
    return None


def fetch(item: Download, directory: Path, runner: Callable = subprocess.run) -> Path:
    """Download into directory and verify. A failed check removes the partial; an interrupted curl keeps it to resume."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / item.name
    partial = directory / f"{item.name}.partial"

    if final.exists():
        problem = _problem_with(final, item)
        if problem is not None:
            raise ExtractionFailure(f"{item.name}: the file already on disk has {problem}; move it aside and rerun")
        return final

    complete = item.size is not None and partial.exists() and partial.stat().st_size == item.size
    if not complete:
        command = ["curl", "-fL", "-C", "-", "--retry", "5", "--retry-delay", "30", "-o", str(partial), item.url]
        completed = runner(command, check=False)
        if completed.returncode != 0:
            raise ExtractionFailure(f"{item.name}: curl exited {completed.returncode} for {item.url}; rerun to resume")

    problem = _problem_with(partial, item)
    if problem is not None:
        partial.unlink()
        raise ExtractionFailure(f"{item.name}: the download has {problem}; removed it so a rerun starts fresh")
    os.replace(partial, final)
    return final


def zenodo_downloads(record_id: int, directory: Path, runner: Callable = subprocess.run) -> list[Download]:
    """Every file of a Zenodo record, with md5s from the record, which is saved beside the files as record.json."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / ".record.json.partial"
    url = f"https://zenodo.org/api/records/{record_id}"
    command = ["curl", "-fsSL", "--retry", "5", "--retry-delay", "30", "-o", str(temporary), url]
    completed = runner(command, check=False)
    if completed.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise ExtractionFailure(f"Zenodo record {record_id}: curl exited {completed.returncode}")
    record = json.loads(temporary.read_text(encoding="utf-8"))
    os.replace(temporary, directory / "record.json")

    items = []
    for entry in record["files"]:
        name = entry["key"]
        algorithm, _, checksum = entry["checksum"].partition(":")
        if algorithm != "md5":
            raise ExtractionFailure(f"Zenodo record {record_id}: {name} has checksum {entry['checksum']}, not an md5")
        items.append(
            Download(
                url=f"https://zenodo.org/api/records/{record_id}/files/{quote(name)}/content",
                name=name,
                md5=checksum,
                size=entry["size"],
            )
        )
    return items


def sha256_of(path: Path) -> str:
    """The file's sha256, cached in a <name>.sha256 sidecar for as long as the file is not newer than it."""
    path = Path(path)
    sidecar = path.with_name(f"{path.name}.sha256")
    if sidecar.exists() and sidecar.stat().st_mtime >= path.stat().st_mtime:
        return sidecar.read_text(encoding="utf-8").strip()
    digest = _hash_of(path, "sha256")
    sidecar.write_text(digest, encoding="utf-8")
    return digest
