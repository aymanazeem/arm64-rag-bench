import hashlib
import re
from dataclasses import replace

import pytest

from arm64_rag.documents import CHUNK_CONFIG, Document, canonicalize, chunk_document


def document(raw: bytes, **changes) -> Document:
    text = canonicalize(raw)
    return Document(
        source_id="guide",
        revision="1",
        text=text,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        canonical_sha256=hashlib.sha256(text.encode()).hexdigest(),
        title="Guide",
        url="",
        license_id="test",
        license_text="Original test fixture.",
        **changes,
    )


def assert_spans(doc, chunks):
    covered = set()
    previous_end = 0
    for ordinal, chunk in enumerate(chunks):
        assert chunk.ordinal == ordinal
        assert previous_end <= chunk.start < chunk.end <= len(doc.text)
        assert chunk.text == doc.text[chunk.start : chunk.end]
        assert chunk.line_start == doc.text.count("\n", 0, chunk.start) + 1
        assert chunk.line_end == doc.text.count("\n", 0, chunk.end - 1) + 1
        assert len(re.findall(CHUNK_CONFIG["token_pattern"], chunk.text)) <= 256
        covered.update(range(chunk.start, chunk.end))
        previous_end = chunk.end
    assert all(
        index in covered or char.isspace() for index, char in enumerate(doc.text)
    )
    if doc.text.strip():
        assert "".join(chunk.text for chunk in chunks) == doc.text


def test_canonicalization_preserves_unicode_and_only_changes_newlines():
    raw = "\ufeff# Café\r\nα\rβ\n\u2028γ\t  ".encode()
    assert canonicalize(raw) == "\ufeff# Café\nα\nβ\n\u2028γ\t  "
    with pytest.raises(UnicodeDecodeError):
        canonicalize(b"bad\xff")


def test_markdown_structure_and_source_lines():
    doc = document(
        (
            "# Root\r\n\r\nFirst α paragraph.\r\n\r\n"
            "```python\r\n# not a heading\r\nx = 1\r\n```\r\n\r\n"
            "## Child\r\n\r\n| Name | Value |\r\n| --- | --- |\r\n"
            "| batch | 128 |\r\n\r\nChild paragraph.\r\n\r\n"
            "Sibling\r\n-------\r\n\r\nFinal paragraph.\r\n"
        ).encode()
    )
    chunks = chunk_document(doc)
    assert [chunk.headings for chunk in chunks] == [
        ("Root",),
        ("Root", "Child"),
        ("Root", "Sibling"),
    ]
    assert "```python\n# not a heading\nx = 1\n```" in chunks[0].text
    assert "| Name | Value |\n| --- | --- |\n| batch | 128 |" in chunks[1].text
    assert_spans(doc, chunks)


def test_unmapped_markdown_and_nested_blocks_are_preserved():
    doc = document(
        b"[reference]: https://example.test\n\n# Guide\n\n"
        b"> quoted text\n> ## quoted heading\n\n"
        b"- first\n  - nested\n- second\n\n"
        b"    # indented code\n\n[end]: /path\n"
    )
    chunks = chunk_document(doc)
    assert_spans(doc, chunks)
    assert all(chunk.headings in {(), ("Guide",)} for chunk in chunks)
    assert "[reference]: https://example.test" in "".join(c.text for c in chunks)
    assert "[end]: /path" in "".join(c.text for c in chunks)


def test_large_blocks_split_at_lines_and_long_lines_at_lexical_boundaries():
    code = "```text\n" + "value " * 300 + "\n" + "small row\n" * 100 + "```\n"
    doc = document(code.encode())
    chunks = chunk_document(doc)
    assert len(chunks) >= 2
    assert_spans(doc, chunks)
    assert "".join(chunk.text for chunk in chunks) == doc.text
    for chunk in chunks[:-1]:
        assert chunk.text.endswith(("\n", " "))


def test_plain_text_does_not_infer_markdown_headings():
    doc = document(
        "# This is plain text\n\nUnicode Café.\u2028Still one line.".encode(),
        format="text",
    )
    chunks = chunk_document(doc)
    assert_spans(doc, chunks)
    assert len(chunks) == 1
    assert chunks[0].headings == ()
    assert chunks[0].line_end == 3


def test_chunk_ids_are_deterministic_and_include_source_version(monkeypatch):
    doc = document(b"# Guide\n\nKeep source locations.\n")
    original = chunk_document(doc)
    assert original == chunk_document(doc)
    assert (
        original[0].chunk_id
        != chunk_document(replace(doc, source_id="copy"))[0].chunk_id
    )
    assert (
        original[0].chunk_id != chunk_document(replace(doc, revision="2"))[0].chunk_id
    )
    crlf = document(b"# Guide\r\n\r\nKeep source locations.\r\n")
    assert doc.canonical_sha256 == crlf.canonical_sha256
    assert original[0].chunk_id != chunk_document(crlf)[0].chunk_id
    monkeypatch.setitem(CHUNK_CONFIG, "version", "test-version")
    assert original[0].chunk_id != chunk_document(doc)[0].chunk_id


@pytest.mark.parametrize("raw", [b"", b" \n\t\n"])
def test_blank_documents_have_no_chunks(raw):
    assert chunk_document(document(raw)) == []


def test_invalid_format_or_text_hash_is_rejected():
    doc = document(b"Text.")
    with pytest.raises(ValueError, match="Only text and Markdown"):
        chunk_document(replace(doc, format="pdf"))
    with pytest.raises(ValueError, match="canonical hash"):
        chunk_document(replace(doc, text="Changed."))


def test_leading_blank_lines_and_trailing_whitespace_remain_in_source_spans():
    doc = document(b"\n \n# Heading\n\nText.\n\n \t")
    chunks = chunk_document(doc)
    assert_spans(doc, chunks)
    assert len(chunks) == 1
    assert chunks[0].headings == ("Heading",)
