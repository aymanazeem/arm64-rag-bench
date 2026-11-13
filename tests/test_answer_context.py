import copy
import hashlib

import pytest

from arm64_rag.context import expand_sources
from arm64_rag.documents import Document
from arm64_rag.index import search_index, update_index


def document(text, source_id="build-note"):
    digest = hashlib.sha256(text.encode()).hexdigest()
    return Document(
        source_id=source_id,
        revision="test-revision",
        text=text,
        raw_sha256=digest,
        canonical_sha256=digest,
        title="Build notes",
        url="https://example.test/build",
        license_id="fixture",
        license_text="Original test text.",
    )


def retrieve(path, query, label="S1"):
    result = search_index(path, query, limit=1)
    return {**result["results"][0], "label": label}, result


@pytest.fixture
def indexed(tmp_path):
    text = (
        "# Build\n\nCafé requirements apply to all platforms.\n\n"
        "## Native\n\nnativeanchor uses the CPU option.\n\n"
        "## Android\n\nandroidanchor uses a mobile ABI.\n"
    )
    path = tmp_path / "index.sqlite"
    update_index(path, [document(text)])
    return path, text


def test_expansion_restores_parent_conditions_without_adding_sibling_platform(indexed):
    path, text = indexed
    anchor, snapshot = retrieve(path, "nativeanchor")
    before = copy.deepcopy(anchor)
    expanded = expand_sources(path, [anchor], snapshot)[0]
    assert expanded["text"] == text[: text.index("## Android")]
    assert expanded["start"] == 0
    assert expanded["end"] == text.index("## Android")
    assert expanded["line_start"] == 1
    assert expanded["line_end"] == text[: expanded["end"]].count("\n")
    assert expanded["headings"] == ["Build"]
    assert expanded["anchor_chunk_id"] == anchor["chunk_id"]
    assert expanded["context_chunk_ids"][-1] == anchor["chunk_id"]
    assert len(expanded["context_chunk_ids"]) == 2
    assert "chunk_id" not in expanded
    assert (expanded["label"], expanded["rank"], expanded["score"]) == (
        anchor["label"],
        anchor["rank"],
        anchor["score"],
    )
    assert anchor == before
    assert expand_sources(path, [anchor], snapshot) == [expanded]


def test_equal_heading_neighbours_are_not_added(tmp_path):
    paragraphs = [
        f"{name} " + "ordinaryword " * 170 + "\n\n"
        for name in ("beforeanchor", "middleanchor", "afteranchor", "distantanchor")
    ]
    text = "# One section\n\n" + "".join(paragraphs)
    path = tmp_path / "index.sqlite"
    update_index(path, [document(text)])
    anchor, snapshot = retrieve(path, "middleanchor")
    distant, _ = retrieve(path, "distantanchor", "S2")
    expanded = expand_sources(path, [anchor, distant], snapshot)
    assert expanded == [anchor, distant]


def test_unheaded_preamble_is_not_a_named_parent(tmp_path):
    path = tmp_path / "index.sqlite"
    text = "A general unheaded introduction.\n\n# Guide\n\nnamedanchor is the fact.\n"
    update_index(path, [document(text)])
    anchor, snapshot = retrieve(path, "namedanchor")
    assert expand_sources(path, [anchor], snapshot) == [anchor]


def test_sibling_and_child_sections_are_not_added(indexed):
    path, _ = indexed
    parent, snapshot = retrieve(path, "requirements")
    sibling, _ = retrieve(path, "androidanchor", "S2")
    assert expand_sources(path, [parent, sibling], snapshot) == [parent, sibling]


def test_plain_text_neighbours_never_cross_documents(tmp_path):
    path = tmp_path / "index.sqlite"
    update_index(
        path, [document("firstanchor", "one"), document("secondanchor", "two")]
    )
    anchor, snapshot = retrieve(path, "firstanchor")
    assert expand_sources(path, [anchor], snapshot) == [anchor]


@pytest.mark.parametrize("key", ["generation", "fingerprint"])
def test_expansion_rejects_a_different_retrieval_snapshot(indexed, key):
    path, _ = indexed
    anchor, snapshot = retrieve(path, "nativeanchor")
    snapshot[key] = -1 if key == "generation" else "different-fingerprint"
    with pytest.raises(ValueError, match="index changed"):
        expand_sources(path, [anchor], snapshot)


def test_expansion_rejects_an_index_updated_after_retrieval(indexed):
    path, text = indexed
    anchor, snapshot = retrieve(path, "nativeanchor")
    update_index(path, [document(text + "\nA later update.\n")])
    with pytest.raises(ValueError, match="index changed"):
        expand_sources(path, [anchor], snapshot)


@pytest.mark.parametrize("key", ["chunk_id", "text", "revision", "start", "headings"])
def test_expansion_rejects_an_anchor_that_does_not_match_storage(indexed, key):
    path, _ = indexed
    anchor, snapshot = retrieve(path, "nativeanchor")
    anchor[key] = {"start": -1, "headings": ["Other section"]}.get(key, "changed")
    with pytest.raises(ValueError, match="indexed"):
        expand_sources(path, [anchor], snapshot)
