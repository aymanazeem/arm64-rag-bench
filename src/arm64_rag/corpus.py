import hashlib
import json
import os
from pathlib import Path

from arm64_rag.documents import Document, canonicalize

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _source_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Source paths must be relative to the document root.")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("A source path leaves the document root.")
    if resolved.suffix.lower() not in TEXT_SUFFIXES:
        raise ValueError("Only text and Markdown documents are supported.")
    return resolved


def load_manifest(path: Path, root: Path) -> list[Document]:
    root = root.resolve()
    manifest = json.loads(path.read_bytes())
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("canonical_format") != "utf8-lf-v1"
        or not isinstance(manifest.get("sources"), list)
        or not isinstance(manifest.get("licenses"), dict)
    ):
        raise ValueError("Unsupported document manifest.")
    documents = []
    seen = set()
    for source in manifest["sources"]:
        required = {
            "id",
            "local_path",
            "revision",
            "license_id",
            "sha256",
            "bytes",
            "canonical_sha256",
            "canonical_chars",
        }
        if not isinstance(source, dict) or not required <= source.keys():
            raise ValueError("A manifest source is incomplete.")
        if any(
            not isinstance(source[key], str) or not source[key]
            for key in required - {"bytes", "canonical_chars"}
        ):
            raise ValueError("Source IDs, paths, revisions and hashes must be text.")
        source_id = source["id"]
        if not isinstance(source.get("url", ""), str):
            raise ValueError(f"Source URL must be text: {source_id}")
        if source_id in seen:
            raise ValueError(f"Duplicate source ID: {source_id}")
        seen.add(source_id)
        source_path = _source_path(root, source["local_path"])
        raw = source_path.read_bytes()
        text = canonicalize(raw)
        if len(raw) != source["bytes"] or digest(raw) != source["sha256"]:
            raise ValueError(f"Source bytes do not match the manifest: {source_id}")
        if (
            len(text) != source["canonical_chars"]
            or digest(text.encode()) != source["canonical_sha256"]
        ):
            raise ValueError(f"Canonical text does not match: {source_id}")
        license_id = source["license_id"]
        license_record = manifest["licenses"].get(license_id)
        if not isinstance(license_record, dict):
            raise ValueError(f"Missing license record: {source_id}")
        license_text = ""
        if "local_path" in license_record:
            if not isinstance(license_record["local_path"], str):
                raise ValueError("License paths must be text.")
            license_path = Path(license_record["local_path"])
            if license_path.is_absolute() or ".." in license_path.parts:
                raise ValueError("License paths must be relative to the document root.")
            resolved = (root / license_path).resolve()
            if not resolved.is_relative_to(root):
                raise ValueError("A license path leaves the document root.")
            license_raw = resolved.read_bytes()
            if digest(license_raw) != license_record.get("sha256"):
                raise ValueError(f"License bytes do not match: {license_id}")
            license_text = license_raw.decode("utf-8")
        elif not isinstance(license_record.get("provenance"), str):
            raise ValueError(f"Missing license text or provenance: {license_id}")
        documents.append(
            Document(
                source_id=source_id,
                revision=source["revision"],
                text=text,
                raw_sha256=source["sha256"],
                canonical_sha256=source["canonical_sha256"],
                title=source_id,
                url=source.get("url", ""),
                license_id=license_id,
                license_text=license_text,
                format="text" if source_path.suffix.lower() == ".txt" else "markdown",
            )
        )
    return documents


def load_paths(paths: list[Path], root: Path) -> list[Document]:
    root = Path(os.path.abspath(root))
    resolved_root = root.resolve()
    selected: dict[str, Path] = {}
    for entry in paths:
        entry = Path(os.path.abspath(entry))
        if not entry.is_relative_to(root) or not entry.resolve().is_relative_to(
            resolved_root
        ):
            raise ValueError("Input documents must be inside --root.")
        if not entry.exists():
            raise ValueError(f"Input does not exist: {entry.name}")
        candidates = (
            sorted(p for p in entry.rglob("*") if p.suffix.lower() in TEXT_SUFFIXES)
            if entry.is_dir()
            else [entry]
        )
        for candidate in candidates:
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(root).as_posix()
            selected[relative] = _source_path(resolved_root, relative)
    if not selected:
        raise ValueError("No text or Markdown documents found.")
    documents = []
    for source_id, path in sorted(selected.items()):
        raw = path.read_bytes()
        text = canonicalize(raw)
        documents.append(
            Document(
                source_id=source_id,
                revision=digest(raw),
                text=text,
                raw_sha256=digest(raw),
                canonical_sha256=digest(text.encode()),
                title=Path(source_id).stem,
                url="",
                license_id="user-provided",
                license_text="",
                format="text" if path.suffix.lower() == ".txt" else "markdown",
            )
        )
    return documents
