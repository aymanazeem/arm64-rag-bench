import hashlib
import json
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from markdown_it import MarkdownIt

CHUNK_CONFIG = {
    "version": "structure-lexical-v1",
    "canonical_format": "utf8-lf-v1",
    "parser": "markdown-it-py",
    "parser_version": "4.0.0",
    "preset": "commonmark",
    "enabled_rules": ["table"],
    "max_lexical_tokens": 256,
    "token_pattern": r"\w+|[^\w\s]",
    "overlap": 0,
}


@dataclass(frozen=True)
class Document:
    source_id: str
    revision: str
    text: str
    raw_sha256: str
    canonical_sha256: str
    title: str
    url: str
    license_id: str
    license_text: str
    format: str = "markdown"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_id: str
    ordinal: int
    start: int
    end: int
    line_start: int
    line_end: int
    text: str
    headings: tuple[str, ...]


def canonicalize(raw: bytes) -> str:
    return raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def _blocks(text: str, format: str):
    if format == "text":
        start = 0
        for boundary in re.finditer(r"\n[ \t]*\n", text):
            yield start, boundary.end(), (), False
            start = boundary.end()
        if start < len(text):
            yield start, len(text), (), False
        return
    if format != "markdown":
        raise ValueError("Only text and Markdown documents are supported.")

    offsets = [0, *(match.end() for match in re.finditer("\n", text))]
    if offsets[-1] != len(text):
        offsets.append(len(text))
    tokens = MarkdownIt("commonmark").enable("table").parse(text)
    headings = []
    cursor = 0
    for index, token in enumerate(tokens):
        if token.level != 0 or token.map is None:
            continue
        start, end = (offsets[line] for line in token.map)
        if start > cursor:
            yield cursor, start, tuple(title for _, title in headings), False
        section = token.type == "heading_open"
        if section:
            level = int(token.tag[1:])
            headings = [(depth, title) for depth, title in headings if depth < level]
            headings.append((level, tokens[index + 1].content))
        yield start, end, tuple(title for _, title in headings), section
        cursor = end
    if cursor < len(text):
        yield cursor, len(text), tuple(title for _, title in headings), False


def _bounded_spans(text: str, start: int, end: int):
    pattern = re.compile(CHUNK_CONFIG["token_pattern"])
    positions = [match.start() for match in pattern.finditer(text, start, end)]
    limit = CHUNK_CONFIG["max_lexical_tokens"]
    first = 0
    while len(positions) - first > limit:
        cut = positions[first + limit]
        line_cut = text.rfind("\n", start, cut) + 1
        if line_cut > start:
            cut = line_cut
        next_first = bisect_left(positions, cut)
        yield start, cut, next_first - first
        start, first = cut, next_first
    if start < end:
        yield start, end, len(positions) - first


def chunk_document(document: Document) -> list[Chunk]:
    text = document.text
    if hashlib.sha256(text.encode()).hexdigest() != document.canonical_sha256:
        raise ValueError("Document text does not match its canonical hash.")
    spans = []
    pending = None
    limit = CHUNK_CONFIG["max_lexical_tokens"]
    for start, end, headings, section in _blocks(text, document.format):
        for part_start, part_end, count in _bounded_spans(text, start, end):
            if pending is not None and not text[pending[0] : pending[1]].strip():
                pending = (pending[0], pending[1], headings, pending[3])
                section = False
            if pending is not None and (
                section or pending[2] != headings or pending[3] + count > limit
            ):
                spans.append(pending)
                pending = None
            if pending is None:
                pending = (part_start, part_end, headings, count)
            else:
                pending = (pending[0], part_end, headings, pending[3] + count)
            section = False
    if pending is not None:
        spans.append(pending)

    line_starts = [0, *(match.end() for match in re.finditer("\n", text))]
    identity = {
        "source_id": document.source_id,
        "revision": document.revision,
        "raw_sha256": document.raw_sha256,
        "canonical_sha256": document.canonical_sha256,
        "format": document.format,
        "chunk_config": CHUNK_CONFIG,
    }
    chunks = []
    for start, end, headings, _ in spans:
        if not text[start:end].strip():
            continue
        chunk_id = hashlib.sha256(
            json.dumps(
                {**identity, "start": start, "end": end}, sort_keys=True
            ).encode()
        ).hexdigest()
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                source_id=document.source_id,
                ordinal=len(chunks),
                start=start,
                end=end,
                line_start=bisect_right(line_starts, start),
                line_end=bisect_right(line_starts, end - 1),
                text=text[start:end],
                headings=headings,
            )
        )
    return chunks
