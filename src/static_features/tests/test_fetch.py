from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from src.static_features.fetch import Download, fetch, sha256_of, zenodo_downloads
from src.static_features.results import ExtractionFailure

BYTES = b"abc"
MD5_OF_BYTES = "900150983cd24fb0d6963f7d28e17f72"
SHA256_OF_BYTES = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
URL = "https://example.org/data.bin"


class FakeCurl:
    """Writes given bytes to curl's -o path, appending as curl -C - does, and returns a given exit code."""

    def __init__(self, data=BYTES, returncode=0):
        self.data = data
        self.returncode = returncode
        self.commands = []

    def __call__(self, command, check=False):
        self.commands.append(command)
        target = Path(command[command.index("-o") + 1])
        with open(target, "ab") as handle:
            handle.write(self.data)
        return SimpleNamespace(returncode=self.returncode)


def test_good_download_is_moved_into_place(tmp_path):
    curl = FakeCurl()
    path = fetch(Download(URL, "data.bin", md5=MD5_OF_BYTES, size=3), tmp_path, runner=curl)

    assert path == tmp_path / "data.bin"
    assert path.read_bytes() == BYTES
    assert not (tmp_path / "data.bin.partial").exists()


def test_curl_is_asked_to_fail_on_http_errors_follow_redirects_and_resume(tmp_path):
    curl = FakeCurl()
    fetch(Download(URL, "data.bin", md5=MD5_OF_BYTES), tmp_path, runner=curl)

    command = curl.commands[0]
    assert command[0] == "curl"
    assert "-fL" in command
    assert command[command.index("-C") + 1] == "-"
    assert command[command.index("-o") + 1] == str(tmp_path / "data.bin.partial")
    assert command[-1] == URL


def test_directory_is_created(tmp_path):
    path = fetch(Download(URL, "data.bin", size=3), tmp_path / "new" / "dir", runner=FakeCurl())
    assert path.exists()


def test_wrong_md5_stops_and_leaves_no_final_file(tmp_path):
    with pytest.raises(ExtractionFailure, match="data.bin.*md5"):
        fetch(Download(URL, "data.bin", md5="0" * 32), tmp_path, runner=FakeCurl())
    assert not (tmp_path / "data.bin").exists()
    assert not (tmp_path / "data.bin.partial").exists()  # so a rerun starts fresh


def test_wrong_size_stops(tmp_path):
    with pytest.raises(ExtractionFailure, match="data.bin.*size"):
        fetch(Download(URL, "data.bin", size=999), tmp_path, runner=FakeCurl())
    assert not (tmp_path / "data.bin").exists()


def test_interrupted_curl_keeps_partial_for_resume(tmp_path):
    with pytest.raises(ExtractionFailure, match="data.bin.*18"):
        fetch(Download(URL, "data.bin", md5=MD5_OF_BYTES), tmp_path, runner=FakeCurl(returncode=18))
    assert (tmp_path / "data.bin.partial").exists()
    assert not (tmp_path / "data.bin").exists()


def test_rerun_resumes_from_the_partial(tmp_path):
    (tmp_path / "data.bin.partial").write_bytes(b"a")
    path = fetch(Download(URL, "data.bin", md5=MD5_OF_BYTES), tmp_path, runner=FakeCurl(data=b"bc"))
    assert path.read_bytes() == BYTES


def test_complete_partial_is_verified_without_curl(tmp_path):
    # A job killed after curl finished: asking for bytes past the end would make curl -f fail on HTTP 416.
    (tmp_path / "data.bin.partial").write_bytes(BYTES)
    curl = FakeCurl()
    path = fetch(Download(URL, "data.bin", md5=MD5_OF_BYTES, size=3), tmp_path, runner=curl)
    assert path.read_bytes() == BYTES and curl.commands == []


def test_verified_file_is_not_downloaded_again(tmp_path):
    (tmp_path / "data.bin").write_bytes(BYTES)
    curl = FakeCurl()
    path = fetch(Download(URL, "data.bin", md5=MD5_OF_BYTES, size=3), tmp_path, runner=curl)
    assert path == tmp_path / "data.bin" and curl.commands == []


def test_existing_file_that_fails_verification_stops_and_is_kept(tmp_path):
    (tmp_path / "data.bin").write_bytes(b"something else")
    curl = FakeCurl()
    with pytest.raises(ExtractionFailure, match="data.bin"):
        fetch(Download(URL, "data.bin", md5=MD5_OF_BYTES), tmp_path, runner=curl)
    assert curl.commands == []
    assert (tmp_path / "data.bin").read_bytes() == b"something else"  # nothing on disk is overwritten


def test_download_with_no_checksum_or_size_is_refused():
    with pytest.raises(ValueError, match="data.bin"):
        Download(URL, "data.bin")


RECORD = {
    "id": 10692059,
    "files": [
        {"key": "a.zip", "size": 3, "checksum": f"md5:{MD5_OF_BYTES}"},
        {"key": "b file.tif", "size": 7, "checksum": "md5:" + "1" * 32},
    ],
}


class FakeRecordCurl:
    def __init__(self, record=RECORD, returncode=0):
        self.record = record
        self.returncode = returncode
        self.commands = []

    def __call__(self, command, check=False):
        self.commands.append(command)
        Path(command[command.index("-o") + 1]).write_text(json.dumps(self.record), encoding="utf-8")
        return SimpleNamespace(returncode=self.returncode)


def test_zenodo_downloads_read_names_sizes_and_md5s_from_the_record(tmp_path):
    curl = FakeRecordCurl()
    items = zenodo_downloads(10692059, tmp_path, runner=curl)

    assert curl.commands[0][-1] == "https://zenodo.org/api/records/10692059"
    assert items[0] == Download(
        "https://zenodo.org/api/records/10692059/files/a.zip/content", "a.zip", md5=MD5_OF_BYTES, size=3
    )
    assert items[1].name == "b file.tif"
    assert items[1].url == "https://zenodo.org/api/records/10692059/files/b%20file.tif/content"
    assert json.loads((tmp_path / "record.json").read_text(encoding="utf-8"))["id"] == 10692059


def test_zenodo_failure_stops_naming_the_record_and_keeps_an_earlier_record(tmp_path):
    (tmp_path / "record.json").write_text("earlier", encoding="utf-8")
    with pytest.raises(ExtractionFailure, match="10692059"):
        zenodo_downloads(10692059, tmp_path, runner=FakeRecordCurl(returncode=22))
    assert (tmp_path / "record.json").read_text(encoding="utf-8") == "earlier"


def test_zenodo_checksum_that_is_not_md5_stops(tmp_path):
    record = {"id": 1, "files": [{"key": "a.zip", "size": 3, "checksum": "sha1:abc"}]}
    with pytest.raises(ExtractionFailure, match="a.zip"):
        zenodo_downloads(1, tmp_path, runner=FakeRecordCurl(record=record))


def test_sha256_is_computed_and_cached_in_a_sidecar(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(BYTES)

    assert sha256_of(path) == SHA256_OF_BYTES
    sidecar = tmp_path / "data.bin.sha256"
    assert sidecar.read_text(encoding="utf-8").strip() == SHA256_OF_BYTES

    sidecar.write_text("from the sidecar", encoding="utf-8")
    assert sha256_of(path) == "from the sidecar"


def test_sha256_is_recomputed_when_the_file_is_newer_than_its_sidecar(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(BYTES)
    sidecar = tmp_path / "data.bin.sha256"
    sidecar.write_text("stale", encoding="utf-8")
    old = path.stat().st_mtime - 100
    os.utime(sidecar, (old, old))

    assert sha256_of(path) == SHA256_OF_BYTES
