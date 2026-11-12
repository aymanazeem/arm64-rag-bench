import hashlib
import json
import sqlite3
from dataclasses import replace

import numpy as np
import pytest

from arm64_rag.dense import embed_index, embedding_info, search_vectors
from arm64_rag.documents import Document
from arm64_rag.index import index_inventory, search_index, update_index

CONFIG = {
    "dimension": 3,
    "model": "synthetic",
    "revision": "1",
    "window_policy": "test",
}


def document(source_id, text):
    digest = hashlib.sha256(text.encode()).hexdigest()
    return Document(source_id, "1", text, digest, digest, source_id, "", "test", "")


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "index.sqlite"
    documents = [
        document("a", "alpha exact identifier"),
        document("b", "beta description"),
        document("c", "gamma other description"),
    ]
    update_index(path, documents)
    return path, documents


def encode_documents(texts):
    return [
        np.array([[2, 0, 0]], dtype=np.float32)
        if "alpha" in text
        else np.array([[0, 3, 0], [0, 0, 4]], dtype=np.float32)
        if "beta" in text
        else np.array([[1, 1, 0]], dtype=np.float32)
        for text in texts
    ]


def query_encoder(texts):
    return np.tile(np.array([[0, 0, 3]], dtype=np.float32), (len(texts), 1))


def search(corpus, **kwargs):
    return search_vectors(corpus[0], "description", query_encoder, CONFIG, **kwargs)


def test_vectors_add_without_changing_existing_lexical_index(corpus):
    path, _ = corpus
    inventory = index_inventory(path, include_text=True)
    lexical = search_index(path, "exact identifier")
    state = embed_index(path, encode_documents, CONFIG)
    assert state["chunks"] == 3
    assert state["windows"] == 4
    assert state["source_fingerprint"] == inventory["fingerprint"]
    assert embedding_info(path, CONFIG) == state
    assert index_inventory(path, include_text=True) == inventory
    assert search_index(path, "exact identifier") == lexical
    assert embed_index(path, encode_documents, CONFIG) == state


def test_exact_normalized_cosine_and_best_window(corpus):
    embed_index(corpus[0], encode_documents, CONFIG)
    results = search(corpus)["results"]
    assert [row["source_id"] for row in results] == ["b", "a", "c"]
    assert [row["score"] for row in results] == pytest.approx([1, 0, 0])
    assert results[0]["best_window"] == 1
    assert results[0]["window_count"] == 2
    assert results[0]["text"] == corpus[1][1].text
    assert results[0]["rank"] == 1
    response = search_vectors(
        corpus[0], "alpha", lambda _: [[5, 0, 0]], CONFIG, limit=3
    )
    assert [row["source_id"] for row in response["results"]] == ["a", "c", "b"]
    assert response["results"][1]["score"] == pytest.approx(1 / 2**0.5)


def test_scope_and_stable_ties(corpus):
    embed_index(corpus[0], encode_documents, CONFIG)
    assert [
        row["source_id"] for row in search(corpus, source_ids=["c", "a"])["results"]
    ] == ["a", "c"]
    assert search(corpus, source_ids=[])["results"] == []
    with pytest.raises(ValueError, match="Unknown source"):
        search(corpus, source_ids=["unknown"])


def test_hybrid_uses_fixed_reciprocal_ranks_in_one_scope(corpus):
    path, _ = corpus
    embed_index(path, encode_documents, CONFIG)
    result = search_vectors(path, "alpha", query_encoder, CONFIG, method="hybrid")
    assert result["method"] == "hybrid-rrf"
    assert result["rrf_constant"] == 60
    assert result["candidate_depth"] == 20
    assert [item["source_id"] for item in result["results"]] == ["a", "b", "c"]
    assert result["results"][0]["score"] == pytest.approx(1 / 61 + 1 / 62)
    scoped = search_vectors(
        path, "alpha", query_encoder, CONFIG, method="hybrid", source_ids=["b"]
    )
    assert [item["source_id"] for item in scoped["results"]] == ["b"]
    assert scoped["results"][0]["score"] == pytest.approx(1 / 61)


@pytest.mark.parametrize("operation", ["update", "delete", "metadata"])
def test_source_changes_invalidate_all_vectors_in_same_transaction(corpus, operation):
    path, documents = corpus
    embed_index(path, encode_documents, CONFIG)
    if operation == "update":
        update_index(path, [document("a", "new alpha")])
    elif operation == "delete":
        update_index(path, documents[:1], sync=True)
    else:
        update_index(path, [replace(documents[0], title="New title")])
    with pytest.raises(ValueError, match="missing or stale"):
        search(corpus)
    assert search_index(path, "alpha")["results"]
    with sqlite3.connect(path) as connection:
        assert (
            connection.execute("SELECT count(*) FROM chunk_vectors").fetchone()[0] == 0
        )
        assert (
            connection.execute("SELECT count(*) FROM embedding_state").fetchone()[0]
            == 0
        )


def test_noop_import_preserves_vectors(corpus):
    state = embed_index(corpus[0], encode_documents, CONFIG)
    update_index(corpus[0], corpus[1], sync=True)
    assert embedding_info(corpus[0]) == state


def test_failed_source_update_retains_source_and_vectors(corpus):
    path, _ = corpus
    state = embed_index(path, encode_documents, CONFIG)
    before = search(corpus)
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TRIGGER reject_new BEFORE INSERT ON chunks
            WHEN NEW.text LIKE '%broken%'
            BEGIN SELECT RAISE(ABORT, 'test failure'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="test failure"):
        update_index(path, [document("a", "broken")], sync=True)
    assert embedding_info(path) == state
    after = search(corpus)
    assert after["results"] == before["results"]


def test_failed_embedding_publication_preserves_previous_vectors(corpus):
    path, _ = corpus
    state = embed_index(path, encode_documents, CONFIG)
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TRIGGER reject_vector
            BEFORE INSERT ON chunk_vectors
            BEGIN SELECT RAISE(ABORT, 'vector failure'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="vector failure"):
        embed_index(path, encode_documents, {**CONFIG, "revision": "2"})
    assert embedding_info(path) == state
    assert search(corpus)["results"][0]["source_id"] == "b"


def test_source_change_during_embedding_cannot_publish_stale_vectors(corpus):
    path, _ = corpus

    def changing_encoder(texts):
        update_index(path, [document("a", "new alpha")])
        return encode_documents(texts)

    with pytest.raises(ValueError, match="index changed during embedding"):
        embed_index(path, changing_encoder, CONFIG)
    with pytest.raises(ValueError, match="missing"):
        embedding_info(path)
    assert search_index(path, "new")["results"]


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [[0, 0, 0]],
        [[float("nan"), 1, 1]],
        [[float("inf"), 1, 1]],
        [[1, 2]],
        [1, 2, 3],
    ],
)
def test_bad_vectors_never_replace_valid_embeddings(corpus, bad):
    path, _ = corpus
    state = embed_index(path, encode_documents, CONFIG)
    with pytest.raises(ValueError, match="shape|finite|nonzero"):
        embed_index(path, lambda texts: [bad] * len(texts), CONFIG)
    assert embedding_info(path) == state


def test_wrong_batch_size_and_query_shape_are_rejected(corpus):
    path, _ = corpus
    with pytest.raises(ValueError, match="count"):
        embed_index(path, lambda texts: [], CONFIG)
    embed_index(path, encode_documents, CONFIG)
    with pytest.raises(ValueError, match="exactly one"):
        search_vectors(path, "alpha", lambda _: [[1, 0, 0], [0, 1, 0]], CONFIG)


def test_config_mismatch_and_invalid_dimensions_are_rejected(corpus):
    path, _ = corpus
    embed_index(path, encode_documents, CONFIG)
    with pytest.raises(ValueError, match="configuration differs"):
        search_vectors(path, "alpha", query_encoder, {**CONFIG, "revision": "new"})
    for dimension in (0, True, -1, 1.5):
        with pytest.raises(ValueError, match="dimension"):
            embed_index(path, encode_documents, {**CONFIG, "dimension": dimension})


@pytest.mark.parametrize("damage", ["missing", "wrong_size", "nan", "content", "stale"])
def test_stored_incomplete_or_corrupt_vectors_are_rejected(corpus, damage):
    path, _ = corpus
    embed_index(path, encode_documents, CONFIG)
    with sqlite3.connect(path) as connection:
        if damage == "missing":
            connection.execute("DELETE FROM chunk_vectors WHERE rowid=1")
        elif damage == "stale":
            state = json.loads(
                connection.execute("SELECT value FROM embedding_state").fetchone()[0]
            )
            state["source_generation"] += 1
            connection.execute(
                "UPDATE embedding_state SET value=?", (json.dumps(state),)
            )
        else:
            value = {
                "wrong_size": b"bad",
                "nan": np.array([np.nan, 0, 1], dtype="<f4").tobytes(),
                "content": np.array([0, 1, 0], dtype="<f4").tobytes(),
            }[damage]
            connection.execute(
                "UPDATE chunk_vectors SET vector=? WHERE rowid=1", (value,)
            )
    with pytest.raises(
        ValueError, match="incomplete|out of order|dimensions|finite|hash|stale"
    ):
        search(corpus)


def test_empty_index_and_empty_queries_need_no_encoder(tmp_path, corpus):
    path = tmp_path / "empty.sqlite"
    update_index(path, [])

    def forbidden(_):
        raise AssertionError("Encoder should not run")

    assert embed_index(path, forbidden, CONFIG)["windows"] == 0
    assert search_vectors(path, "", forbidden, CONFIG)["results"] == []
    assert search_vectors(path, "alpha", query_encoder, CONFIG)["results"] == []
    embed_index(corpus[0], encode_documents, CONFIG)
    assert search_vectors(corpus[0], '"()+-*/', forbidden, CONFIG)["results"] == []


def test_missing_database_and_legacy_index_fail_without_mutation(tmp_path, corpus):
    path = tmp_path / "missing.sqlite"
    with pytest.raises(FileNotFoundError):
        embed_index(path, encode_documents, CONFIG)
    assert not path.exists()
    before = index_inventory(corpus[0])
    with pytest.raises(ValueError, match="missing"):
        search(corpus)
    assert index_inventory(corpus[0]) == before


def test_result_limits_methods_and_query_bounds_are_explicit(corpus):
    embed_index(corpus[0], encode_documents, CONFIG)
    for limit in (0, 21, True, 1.5):
        with pytest.raises(ValueError, match="limit"):
            search(corpus, limit=limit)
    with pytest.raises(ValueError, match="method"):
        search(corpus, method="unknown")
    with pytest.raises(ValueError, match="4096"):
        search_vectors(corpus[0], "a" * 4097, query_encoder, CONFIG)
