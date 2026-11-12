import hashlib
import json
from pathlib import Path

import pytest

from arm64_rag.corpus import load_manifest, load_paths


def manifest_for(root, text=b"# Guide\r\nCaf\xc3\xa9 and flags.\r\n"):
    (root / "guide.md").write_bytes(text)
    canonical = text.decode().replace("\r\n", "\n").replace("\r", "\n")
    value = {
        "schema_version": 1,
        "canonical_format": "utf8-lf-v1",
        "licenses": {"fixture": {"provenance": "Original test text."}},
        "sources": [
            {
                "id": "guide",
                "revision": "one",
                "local_path": "guide.md",
                "license_id": "fixture",
                "bytes": len(text),
                "sha256": hashlib.sha256(text).hexdigest(),
                "canonical_chars": len(canonical),
                "canonical_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            }
        ],
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(value))
    return path, value


def test_manifest_preserves_canonical_text_and_source_identity(tmp_path):
    path, value = manifest_for(tmp_path)
    (document,) = load_manifest(path, tmp_path)
    assert document.text == "# Guide\nCafé and flags.\n"
    assert document.raw_sha256 == value["sources"][0]["sha256"]
    assert document.canonical_sha256 != document.raw_sha256
    assert document.source_id == "guide"
    assert document.revision == "one"


def test_changed_source_rejected_before_ingestion(tmp_path):
    path, _ = manifest_for(tmp_path)
    (tmp_path / "guide.md").write_text("Different content")
    with pytest.raises(ValueError, match="Source bytes"):
        load_manifest(path, tmp_path)


def test_license_is_preserved_and_verified(tmp_path):
    path, value = manifest_for(tmp_path)
    license_raw = b"Test license\r\nPreserve this notice.\r\n"
    (tmp_path / "LICENSE").write_bytes(license_raw)
    value["licenses"]["fixture"] = {
        "local_path": "LICENSE",
        "sha256": hashlib.sha256(license_raw).hexdigest(),
    }
    path.write_text(json.dumps(value))
    assert load_manifest(path, tmp_path)[0].license_text.encode() == license_raw
    (tmp_path / "LICENSE").write_text("Changed")
    with pytest.raises(ValueError, match="License bytes"):
        load_manifest(path, tmp_path)


def test_duplicate_manifest_ids_and_unsupported_format_fail(tmp_path):
    path, value = manifest_for(tmp_path)
    value["sources"] *= 2
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Duplicate source"):
        load_manifest(path, tmp_path)
    value["canonical_format"] = "other"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Unsupported document manifest"):
        load_manifest(path, tmp_path)


def test_local_paths_are_stable_deduplicated_and_versioned(tmp_path):
    (tmp_path / "a.txt").write_text("First version")
    (tmp_path / "b.md").write_text("# Another source")
    (tmp_path / "ignored.bin").write_bytes(b"\xff")
    docs = load_paths([tmp_path, tmp_path / "a.txt"], tmp_path)
    assert [d.source_id for d in docs] == ["a.txt", "b.md"]
    assert docs[0].format == "text"
    assert docs[1].format == "markdown"
    assert docs == load_paths([tmp_path], tmp_path)
    (tmp_path / "a.txt").write_text("Second version")
    changed = load_paths([tmp_path], tmp_path)
    assert changed[0].revision != docs[0].revision
    assert changed[0].source_id == docs[0].source_id


def test_invalid_encoding_unsupported_files_and_empty_input_fail(tmp_path):
    with pytest.raises(ValueError, match="No text"):
        load_paths([tmp_path], tmp_path)
    (tmp_path / "bad.txt").write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        load_paths([tmp_path / "bad.txt"], tmp_path)
    (tmp_path / "bad.pdf").write_bytes(b"pdf")
    with pytest.raises(ValueError, match="Only text"):
        load_paths([tmp_path / "bad.pdf"], tmp_path)


def test_manifest_paths_cannot_leave_root(tmp_path):
    path, value = manifest_for(tmp_path)
    value["sources"][0]["local_path"] = "../outside.md"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="relative"):
        load_manifest(path, tmp_path)
    with pytest.raises(ValueError, match="inside --root"):
        load_paths([Path(__file__)], tmp_path)


def test_symlink_outside_root_is_rejected(tmp_path):
    (tmp_path / "external.md").symlink_to(Path(__file__).resolve())
    with pytest.raises(ValueError, match="leaves the document root"):
        load_paths([tmp_path], tmp_path)


def test_symlink_ids_match_direct_and_directory_imports(tmp_path):
    (tmp_path / "target.md").write_text("Original document")
    alias = tmp_path / "alias.md"
    alias.symlink_to(tmp_path / "target.md")
    scanned = {doc.source_id: doc for doc in load_paths([tmp_path], tmp_path)}
    (direct,) = load_paths([alias], tmp_path)
    assert direct == scanned["alias.md"]
    assert set(scanned) == {"alias.md", "target.md"}


def test_symlink_document_root_preserves_relative_ids(tmp_path):
    root = tmp_path / "documents"
    root.mkdir()
    (root / "guide.md").write_text("Guide")
    alias = tmp_path / "collection"
    alias.symlink_to(root, target_is_directory=True)
    assert load_paths([alias / "guide.md"], alias) == load_paths([root], root)


@pytest.mark.parametrize("field", ["url", "license_path"])
def test_malformed_manifest_metadata_is_rejected(tmp_path, field):
    path, value = manifest_for(tmp_path)
    if field == "url":
        value["sources"][0]["url"] = ["not a URL"]
    else:
        value["licenses"]["fixture"]["local_path"] = 7
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="must be text"):
        load_manifest(path, tmp_path)
