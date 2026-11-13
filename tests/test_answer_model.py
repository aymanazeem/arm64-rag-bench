import hashlib
import importlib
import ssl
import subprocess
import urllib.error
from pathlib import Path

import pytest


@pytest.fixture
def downloader(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("download_answer_model")


@pytest.fixture
def asset():
    raw = b"Original license text.\n"
    return {
        "url": "https://example.test/pinned/LICENSE",
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }, raw


def test_normal_download_uses_existing_helper(downloader, monkeypatch, asset, tmp_path):
    record, _ = asset
    destination = tmp_path / "LICENSE"
    calls = []
    monkeypatch.setattr(
        downloader, "download", lambda *args: calls.append(args) or destination
    )
    assert downloader.download_asset(record, destination) == destination
    assert calls == [(record, destination)]


def test_certificate_failure_uses_verified_https_curl(
    downloader, monkeypatch, asset, tmp_path
):
    record, raw = asset
    destination = tmp_path / "nested" / "LICENSE"

    def certificate_failure(*args):
        raise urllib.error.URLError(ssl.SSLCertVerificationError("Untrusted chain"))

    def curl(command, check):
        assert check is True
        assert command[:2] == ["curl", "--disable"]
        assert "--insecure" not in command and "-k" not in command
        assert command[command.index("--proto") + 1] == "=https"
        assert command[command.index("--proto-redir") + 1] == "=https"
        assert command[command.index("--connect-timeout") + 1] == "30"
        assert command[command.index("--max-time") + 1] == "1200"
        assert command[-1] == record["url"]
        partial = Path(command[command.index("--output") + 1])
        assert partial.exists() and not destination.exists()
        partial.write_bytes(raw)

    monkeypatch.setattr(downloader, "download", certificate_failure)
    monkeypatch.setattr(downloader.subprocess, "run", curl)
    assert downloader.download_asset(record, destination) == destination
    assert destination.read_bytes() == raw
    assert not destination.with_suffix(".part").exists()


def test_other_network_failures_are_not_retried(
    downloader, monkeypatch, asset, tmp_path
):
    def network_failure(*args):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(downloader, "download", network_failure)
    monkeypatch.setattr(
        downloader, "download_with_curl", lambda *args: pytest.fail("Unexpected retry")
    )
    with pytest.raises(urllib.error.URLError, match="Connection refused"):
        downloader.download_asset(asset[0], tmp_path / "LICENSE")


@pytest.mark.parametrize("failure", ["checksum", "download"])
def test_failed_fallback_never_publishes_partial_file(
    downloader, monkeypatch, asset, tmp_path, failure
):
    destination = tmp_path / "LICENSE"

    def curl(command, check):
        Path(command[command.index("--output") + 1]).write_bytes(asset[1][::-1])
        if failure == "download":
            raise subprocess.CalledProcessError(22, command)

    monkeypatch.setattr(downloader.subprocess, "run", curl)
    expected = ValueError if failure == "checksum" else subprocess.CalledProcessError
    with pytest.raises(expected):
        downloader.download_with_curl(asset[0], destination)
    assert not destination.exists()
    assert not destination.with_suffix(".part").exists()


def test_existing_assets_are_verified_without_network(
    downloader, monkeypatch, asset, tmp_path
):
    record, raw = asset
    destination = tmp_path / "LICENSE"
    destination.write_bytes(raw)
    monkeypatch.setattr(
        downloader.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected network")
    )
    assert downloader.download_with_curl(record, destination) == destination
    destination.write_bytes(b"Changed")
    with pytest.raises(ValueError, match="Checksum or size"):
        downloader.download_with_curl(record, destination)


@pytest.mark.parametrize("matching", [True, False])
def test_curl_preserves_a_concurrently_created_destination(
    downloader, monkeypatch, asset, tmp_path, matching
):
    record, raw = asset
    destination = tmp_path / "LICENSE"
    competing = raw if matching else b"Preserve this file"
    created = {}

    def curl(command, check):
        Path(command[command.index("--output") + 1]).write_bytes(raw)
        destination.write_bytes(competing)
        created["inode"] = destination.stat().st_ino

    monkeypatch.setattr(downloader.subprocess, "run", curl)
    if matching:
        assert downloader.download_with_curl(record, destination) == destination
    else:
        with pytest.raises(ValueError, match="Checksum or size"):
            downloader.download_with_curl(record, destination)
    assert destination.read_bytes() == competing
    assert destination.stat().st_ino == created["inode"]
    assert not destination.with_suffix(".part").exists()
