import hashlib
import json
import time
from pathlib import Path

from arm64_rag.index import (
    _connection,
    _hash,
    _json,
    _lexical_results,
    _query_expression,
    _state,
    _validated_scope,
)

RRF_CONSTANT = 60
CANDIDATE_DEPTH = 20


def _numpy():
    try:
        import numpy
    except ImportError as error:
        raise RuntimeError("Install the retrieval extra to use embeddings.") from error
    return numpy


def _embedding_config(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError("Embedding configuration must be an object.")
    dimension = config.get("dimension")
    if type(dimension) is not int or not 1 <= dimension <= 65536:
        raise ValueError("Embedding dimension must be a positive integer.")
    return json.loads(
        json.dumps(
            {
                "version": 1,
                "encoder": config,
                "dtype": "float32-le",
                "normalization": "l2",
                "input_text": "heading-path-newline-body-v1",
                "chunk_score": "maximum-window-cosine",
            },
            sort_keys=True,
            allow_nan=False,
        )
    )


def _normalized(values, dimension: int):
    np = _numpy()
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != dimension or not len(values):
        raise ValueError("Embedding vectors have an invalid shape.")
    if not np.isfinite(values).all():
        raise ValueError("Embedding vectors must be finite.")
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    if not (norms > 0).all():
        raise ValueError("Embedding vectors must be nonzero.")
    return np.ascontiguousarray(values / norms[:, None], dtype="<f4")


def _chunks(connection) -> list[dict]:
    records = []
    for row in connection.execute(
        "SELECT c.chunk_id, c.source_id, d.revision, d.title, d.url, d.license_id, "
        "c.start, c.end, c.line_start, c.line_end, c.text, c.headings "
        "FROM chunks c JOIN documents d ON d.source_id = c.source_id "
        "ORDER BY c.source_id, c.start, c.chunk_id"
    ):
        record = dict(row)
        record["headings"] = json.loads(record["headings"])
        records.append(record)
    return records


def _vector_fingerprint(state: dict, rows) -> str:
    digest = hashlib.sha256(
        _json(
            {
                "source_fingerprint": state["source_fingerprint"],
                "config_hash": state["config_hash"],
            }
        ).encode()
    )
    for chunk_id, window, vector in rows:
        digest.update(_json([chunk_id, window, len(vector)]).encode())
        digest.update(vector)
    return digest.hexdigest()


def _ensure_tables(connection) -> None:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS embedding_state ("
        "singleton INTEGER PRIMARY KEY CHECK(singleton = 1), value TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS chunk_vectors ("
        "chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE, "
        "window INTEGER NOT NULL CHECK(window >= 0), vector BLOB NOT NULL, "
        "PRIMARY KEY (chunk_id, window))"
    )


def embed_index(path: Path, encode_documents, config: dict) -> dict:
    config = _embedding_config(config)
    dimension = config["encoder"]["dimension"]
    with _connection(path) as connection:
        snapshot = _state(connection)
        chunks = _chunks(connection)
    texts = ["\n".join((*chunk["headings"], chunk["text"])) for chunk in chunks]
    encoded = encode_documents(texts) if texts else []
    if len(encoded) != len(chunks):
        raise ValueError("Embedding count does not match the chunks.")
    rows = []
    for chunk, windows in zip(chunks, encoded, strict=True):
        for ordinal, vector in enumerate(_normalized(windows, dimension)):
            rows.append((chunk["chunk_id"], ordinal, vector.tobytes()))
    rows.sort(key=lambda row: (row[0], row[1]))
    state = {
        "version": 1,
        "source_generation": snapshot["generation"],
        "source_fingerprint": snapshot["fingerprint"],
        "config": config,
        "config_hash": _hash(_json(config)),
        "chunks": len(chunks),
        "windows": len(rows),
    }
    state["fingerprint"] = _vector_fingerprint(state, rows)
    with _connection(path, writable=True) as connection:
        current = _state(connection)
        if (current["generation"], current["fingerprint"]) != (
            snapshot["generation"],
            snapshot["fingerprint"],
        ):
            raise ValueError("The index changed during embedding; rerun the import.")
        _ensure_tables(connection)
        connection.execute("DELETE FROM chunk_vectors")
        connection.executemany("INSERT INTO chunk_vectors VALUES (?, ?, ?)", rows)
        connection.execute(
            "INSERT OR REPLACE INTO embedding_state VALUES (1, ?)", (_json(state),)
        )
    return state


def _read_embeddings(connection, snapshot: dict, config: dict | None = None):
    np = _numpy()
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if not {"embedding_state", "chunk_vectors"}.issubset(tables):
        raise ValueError("Embeddings are missing; run the embedding import first.")
    record = connection.execute(
        "SELECT value FROM embedding_state WHERE singleton = 1"
    ).fetchone()
    if record is None:
        raise ValueError("Embeddings are missing or stale; run the embedding import.")
    try:
        state = json.loads(record[0])
        if state["version"] != 1:
            raise ValueError("Unsupported embedding index version.")
        if (state["source_generation"], state["source_fingerprint"]) != (
            snapshot["generation"],
            snapshot["fingerprint"],
        ):
            raise ValueError("Embeddings are stale; run the embedding import.")
        if state["config_hash"] != _hash(_json(state["config"])):
            raise ValueError("Embedding configuration hash differs.")
        if config is not None and state["config"] != config:
            raise ValueError("Embedding configuration differs; rebuild embeddings.")
        dimension = state["config"]["encoder"]["dimension"]
        rows = list(
            connection.execute(
                "SELECT chunk_id, window, vector FROM chunk_vectors "
                "ORDER BY chunk_id, window"
            )
        )
        chunk_ids = {
            row[0] for row in connection.execute("SELECT chunk_id FROM chunks")
        }
        grouped = {}
        vectors = []
        for chunk_id, window, blob in rows:
            if chunk_id not in chunk_ids or window != len(grouped.get(chunk_id, [])):
                raise ValueError("Embedding windows are incomplete or out of order.")
            if not isinstance(blob, bytes) or len(blob) != dimension * 4:
                raise ValueError("Stored embedding dimensions differ.")
            grouped.setdefault(chunk_id, []).append(len(vectors))
            vectors.append(np.frombuffer(blob, dtype="<f4"))
        if (
            set(grouped) != chunk_ids
            or len(chunk_ids) != state["chunks"]
            or len(rows) != state["windows"]
        ):
            raise ValueError("Embedding coverage is incomplete; rebuild embeddings.")
        matrix = np.stack(vectors) if vectors else np.empty((0, dimension), dtype="<f4")
        if not np.isfinite(matrix).all() or not np.allclose(
            np.linalg.norm(matrix.astype(np.float64), axis=1), 1, atol=1e-6, rtol=0
        ):
            raise ValueError("Stored embeddings must be finite unit vectors.")
        if _vector_fingerprint(state, rows) != state["fingerprint"]:
            raise ValueError("Embedding content hash differs.")
        return state, matrix, grouped
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Embedding metadata is invalid; rebuild embeddings."
        ) from error


def embedding_info(path: Path, config: dict | None = None) -> dict:
    expected = _embedding_config(config) if config is not None else None
    with _connection(path) as connection:
        snapshot = _state(connection)
        state, _, _ = _read_embeddings(connection, snapshot, expected)
        return state


def _tie_key(result: dict):
    return result["source_id"], result["start"], result["chunk_id"]


def search_vectors(
    path: Path,
    query: str,
    encode_queries,
    config: dict,
    *,
    method: str = "dense",
    limit: int = 5,
    source_ids: list[str] | None = None,
) -> dict:
    if type(limit) is not int or not 1 <= limit <= CANDIDATE_DEPTH:
        raise ValueError(f"Result limit must be from 1 to {CANDIDATE_DEPTH}.")
    if method not in {"dense", "hybrid"}:
        raise ValueError("Embedding search method must be dense or hybrid.")
    expression = _query_expression(query)
    config = _embedding_config(config)
    started = time.perf_counter()
    query_vector = None
    if expression:
        query_vectors = _normalized(
            encode_queries([query]), config["encoder"]["dimension"]
        )
        if len(query_vectors) != 1:
            raise ValueError("Query embedding must contain exactly one vector.")
        query_vector = query_vectors[0]
    embedding_seconds = time.perf_counter() - started
    with _connection(path) as connection:
        snapshot = _state(connection)
        scope = _validated_scope(connection, source_ids)
        state, matrix, grouped = _read_embeddings(connection, snapshot, config)
        output = {
            "generation": snapshot["generation"],
            "fingerprint": snapshot["fingerprint"],
            "embedding_fingerprint": state["fingerprint"],
            "embedding_config_hash": state["config_hash"],
            "method": "dense-cosine" if method == "dense" else "hybrid-rrf",
            "candidate_depth": CANDIDATE_DEPTH,
            "rrf_constant": RRF_CONSTANT if method == "hybrid" else None,
            "query_embedding_seconds": embedding_seconds,
            "results": [],
        }
        if query_vector is None or scope == []:
            return output
        window_scores = matrix @ query_vector
        dense = []
        for chunk in _chunks(connection):
            if scope is not None and chunk["source_id"] not in scope:
                continue
            indices = grouped[chunk["chunk_id"]]
            scores = window_scores[indices]
            best = int(scores.argmax())
            dense.append(
                {
                    **chunk,
                    "score": float(scores[best]),
                    "best_window": best,
                    "window_count": len(indices),
                }
            )
        dense.sort(key=lambda item: (-item["score"], *_tie_key(item)))
        dense = dense[:CANDIDATE_DEPTH]
        if method == "hybrid":
            lexical = _lexical_results(connection, expression, CANDIDATE_DEPTH, scope)
            fused = {}
            for results in (dense, lexical):
                for rank, item in enumerate(results, 1):
                    key = item["chunk_id"]
                    fused.setdefault(key, {**item, "score": 0.0})
                    fused[key]["score"] += 1 / (RRF_CONSTANT + rank)
            dense = sorted(
                fused.values(), key=lambda item: (-item["score"], *_tie_key(item))
            )
        for rank, item in enumerate(dense[:limit], 1):
            output["results"].append({**item, "rank": rank})
        return output
