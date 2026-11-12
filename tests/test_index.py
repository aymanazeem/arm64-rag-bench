import hashlib
import sqlite3
from dataclasses import replace

import pytest

from arm64_rag import index
from arm64_rag.documents import Document
from arm64_rag.index import index_inventory, search_index, update_index


def document(source_id, text, **metadata):
    digest = hashlib.sha256(text.encode()).hexdigest()
    return Document(
        source_id=source_id,
        revision=metadata.get("revision", "1"),
        text=text,
        raw_sha256=digest,
        canonical_sha256=digest,
        title=metadata.get("title", source_id),
        url=metadata.get("url", ""),
        license_id="fixture",
        license_text=metadata.get("license_text", "Original test text."),
        format="markdown",
    )


def ids(result):
    return [item["source_id"] for item in result["results"]]


def test_ingest_search_and_inventory_preserve_sources(tmp_path):
    path = tmp_path / "nested" / "index.sqlite"
    text = "# Batching\n\nCafé records a batch size of 128 tokens.\n"
    doc = document("batching", text, url="https://example.test/batching")
    result = update_index(path, [doc])
    assert result["updated"] == result["documents"] == 1
    assert result["generation"] == 1
    inventory = index_inventory(path)
    assert inventory["fingerprint"] == result["fingerprint"]
    assert "text" not in inventory["documents"][0]
    assert index_inventory(path, include_text=True)["documents"][0]["text"] == text
    matches = search_index(path, "cafe batch size")
    assert ids(matches) == ["batching"]
    match = matches["results"][0]
    assert match["rank"] == 1
    assert match["score"] < 0
    assert match["text"] == text[match["start"] : match["end"]]
    assert match["url"] == doc.url
    with sqlite3.connect(path) as connection:
        assert (
            connection.execute("SELECT license_text FROM documents").fetchone()[0]
            == doc.license_text
        )


def test_unchanged_updates_preserve_generation_and_chunk_ids(tmp_path):
    path = tmp_path / "index.sqlite"
    docs = [document("b", "alpha"), document("a", "alpha")]
    first = update_index(path, docs)
    before = search_index(path, "alpha")
    second = update_index(path, list(reversed(docs)))
    assert second["unchanged"] == 2
    assert second["updated"] == second["deleted"] == 0
    assert second["generation"] == first["generation"]
    assert search_index(path, "alpha") == before
    assert ids(before) == ["a", "b"]
    other_path = tmp_path / "other.sqlite"
    assert (
        update_index(other_path, list(reversed(docs)))["fingerprint"]
        == first["fingerprint"]
    )


def test_partial_update_keeps_others_and_replaces_old_search_entries(tmp_path):
    path = tmp_path / "index.sqlite"
    update_index(path, [document("a", "oldterm"), document("b", "retained")])
    old_chunk = search_index(path, "oldterm")["results"][0]["chunk_id"]
    changed = update_index(path, [document("a", "newterm", revision="2")])
    assert changed["updated"] == 1
    assert changed["deleted"] == 0
    assert changed["documents"] == 2
    assert changed["generation"] == 2
    assert not search_index(path, "oldterm")["results"]
    assert ids(search_index(path, "retained")) == ["b"]
    new_match = search_index(path, "newterm")["results"][0]
    assert new_match["chunk_id"] != old_chunk
    assert new_match["revision"] == "2"


def test_full_sync_deletes_omitted_sources_and_empty_sync_clears(tmp_path):
    path = tmp_path / "index.sqlite"
    doc = document("a", "alpha")
    update_index(path, [doc, document("b", "beta")])
    assert update_index(path, [], sync=False)["generation"] == 1
    result = update_index(path, [doc], sync=True)
    assert result["deleted"] == result["unchanged"] == 1
    assert result["documents"] == 1
    assert not search_index(path, "beta")["results"]
    empty = update_index(path, [], sync=True)
    assert empty["generation"] == 3
    assert empty["documents"] == empty["chunks"] == 0
    assert not search_index(path, "alpha beta")["results"]
    assert update_index(path, [], sync=True)["generation"] == 3


def test_failed_sql_write_rolls_back_sources_chunks_fts_and_generation(tmp_path):
    path = tmp_path / "index.sqlite"
    update_index(path, [document("a", "original"), document("b", "retained")])
    before = index_inventory(path, include_text=True)
    before_search = search_index(path, "original retained")
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TRIGGER reject_broken BEFORE INSERT ON chunks
            WHEN NEW.text LIKE '%broken%'
            BEGIN SELECT RAISE(ABORT, 'simulated write failure'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="simulated write failure"):
        update_index(
            path, [document("a", "changed"), document("c", "broken")], sync=True
        )
    assert index_inventory(path, include_text=True) == before
    assert search_index(path, "original retained") == before_search
    assert not search_index(path, "changed broken")["results"]


def test_preparation_failure_leaves_existing_index_unchanged(tmp_path, monkeypatch):
    path = tmp_path / "index.sqlite"
    update_index(path, [document("a", "original")])
    before = index_inventory(path)

    def fail(_document):
        raise ValueError("Cannot parse document")

    monkeypatch.setattr(index, "chunk_document", fail)
    with pytest.raises(ValueError, match="Cannot parse"):
        update_index(path, [document("b", "new")], sync=True)
    assert index_inventory(path) == before


def test_failed_initial_write_can_be_retried(tmp_path, monkeypatch):
    path = tmp_path / "index.sqlite"
    original = index._insert_source

    def fail_after_insert(*args):
        original(*args)
        raise RuntimeError("Interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(index, "_insert_source", fail_after_insert)
        with pytest.raises(RuntimeError, match="Interrupted"):
            update_index(path, [document("a", "alpha")])
    with sqlite3.connect(path) as connection:
        assert not connection.execute("SELECT name FROM sqlite_master").fetchall()
    assert update_index(path, [document("a", "alpha")])["generation"] == 1


def test_duplicate_ids_and_invalid_hashes_do_not_change_index(tmp_path):
    path = tmp_path / "index.sqlite"
    doc = document("a", "alpha")
    update_index(path, [doc])
    before = index_inventory(path)
    for documents in (
        [doc, doc],
        [replace(doc, canonical_sha256="0" * 64)],
        [replace(doc, raw_sha256="invalid")],
        [replace(doc, source_id="")],
    ):
        with pytest.raises(ValueError):
            update_index(path, documents)
        assert index_inventory(path) == before


def test_metadata_and_revision_changes_are_real_updates(tmp_path):
    path = tmp_path / "index.sqlite"
    doc = document("a", "alpha")
    first = update_index(path, [doc])
    edited = replace(doc, revision="2", title="New title", license_text="New notice")
    second = update_index(path, [edited])
    assert second["generation"] == 2
    assert second["fingerprint"] != first["fingerprint"]
    result = search_index(path, "alpha")["results"][0]
    assert result["revision"] == "2"
    assert result["title"] == "New title"


@pytest.mark.parametrize(
    "query, expected",
    [
        ("GGML_CPU_KLEIDIAI", "kleidiai"),
        ("Q4_0", "quantized"),
        ("--n-gpu-layers", "flag"),
        ("foo/bar.h", "path"),
        ("v1.32.0", "version"),
    ],
)
def test_identifiers_remain_specific(tmp_path, query, expected):
    path = tmp_path / "index.sqlite"
    update_index(
        path,
        [
            document("kleidiai", "Enable GGML_CPU_KLEIDIAI."),
            document("quantized", "The format is Q4_0."),
            document("flag", "Set --n-gpu-layers to zero."),
            document("path", "The header is foo/bar.h."),
            document("version", "Release v1.32.0."),
            document(
                "distractor", "GGML CPU Q4 n gpu and layers foo elsewhere bar h v1 0"
            ),
        ],
    )
    assert ids(search_index(path, query)) == [expected]


@pytest.mark.parametrize(
    "query",
    [
        '"alpha"',
        "alpha OR beta",
        "alpha NOT beta",
        "alpha NEAR(beta)",
        "body:alpha",
        '" OR 1=1 -- alpha',
        "alpha*",
        "alpha ) (",
    ],
)
def test_query_syntax_is_literal_and_cannot_escape_fts(tmp_path, query):
    path = tmp_path / "index.sqlite"
    update_index(path, [document("a", "alpha"), document("b", "beta")])
    assert "a" in ids(search_index(path, query))
    assert index_inventory(path)["generation"] == 1


@pytest.mark.parametrize("query", ["", " \n\t", '"()*+-/:'])
def test_empty_or_punctuation_query_returns_no_results(tmp_path, query):
    path = tmp_path / "index.sqlite"
    update_index(path, [document("a", "alpha")])
    assert not search_index(path, query)["results"]


def test_scope_is_explicit_and_cannot_include_unknown_sources(tmp_path):
    path = tmp_path / "index.sqlite"
    update_index(path, [document("a", "alpha"), document("b", "alpha")])
    assert ids(search_index(path, "alpha", source_ids=["b"])) == ["b"]
    assert not search_index(path, "alpha", source_ids=[])["results"]
    with pytest.raises(ValueError, match="Unknown source IDs"):
        search_index(path, "", source_ids=["missing"])


def test_input_limits_fail_without_silent_truncation(tmp_path):
    path = tmp_path / "index.sqlite"
    update_index(path, [document("a", "alpha")])
    for query in (
        "a" * (index.MAX_QUERY_CHARS + 1),
        "a " * (index.MAX_QUERY_TERMS + 1),
    ):
        with pytest.raises(ValueError, match="Keep queries within"):
            search_index(path, query)
    for limit in (0, -1, 101, True, 2.5):
        with pytest.raises(ValueError, match="Result limit"):
            search_index(path, "alpha", limit=limit)


def test_search_and_inventory_never_create_missing_database(tmp_path):
    path = tmp_path / "missing.sqlite"
    for operation in (
        lambda: search_index(path, "alpha"),
        lambda: index_inventory(path),
    ):
        with pytest.raises(FileNotFoundError, match="Index not found"):
            operation()
        assert not path.exists()


@pytest.mark.parametrize("change", ["config", "version"])
def test_incompatible_index_is_rejected_for_reads_and_writes(tmp_path, change):
    path = tmp_path / "index.sqlite"
    doc = document("a", "alpha")
    update_index(path, [doc])
    with sqlite3.connect(path) as connection:
        if change == "config":
            connection.execute("UPDATE metadata SET value = '{}' WHERE key = 'config'")
        else:
            connection.execute("PRAGMA user_version = 99")
    for operation in (
        lambda: search_index(path, "alpha"),
        lambda: update_index(path, [doc]),
        lambda: index_inventory(path),
    ):
        with pytest.raises(ValueError, match="Index format changed"):
            operation()


def test_unrelated_database_is_not_modified(tmp_path):
    path = tmp_path / "unrelated.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
        connection.execute("INSERT INTO unrelated VALUES ('keep')")
    with pytest.raises(ValueError, match="not a document index"):
        update_index(path, [document("a", "alpha")])
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT value FROM unrelated").fetchone()[0] == "keep"
        assert (
            connection.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] == 1
        )


def test_query_case_handling_matches_sqlite_unicode_tokenization(tmp_path):
    path = tmp_path / "index.sqlite"
    update_index(path, [document("street", "Straße software reference.")])
    assert ids(search_index(path, "Straße")) == ["street"]
    assert ids(search_index(path, "SOFTWARE")) == ["street"]
