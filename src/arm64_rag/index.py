import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from arm64_rag.documents import CHUNK_CONFIG, Document, chunk_document

SCHEMA_VERSION = 1
TOKENIZER = "unicode61 tokenchars '_'"
MAX_QUERY_CHARS = 4096
MAX_QUERY_TERMS = 64
SOURCE_COLUMNS = (
    "source_id",
    "revision",
    "title",
    "url",
    "license_id",
    "raw_sha256",
    "canonical_sha256",
    "format",
    "license_text",
    "text",
)


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _config() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "canonical_format": "utf8-lf-v1",
        "chunking": CHUNK_CONFIG,
        "tokenizer": TOKENIZER,
        "search_text": "heading-path-newline-body-v1",
        "query": "quoted-or-identifier-phrases-v1",
    }


@contextmanager
def _connection(path: Path, *, writable: bool = False):
    path = Path(path)
    if writable:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif not path.is_file():
        raise FileNotFoundError("Index not found; ingest documents first.")
    uri = path.resolve().as_uri() + ("?mode=rwc" if writable else "?mode=ro")
    connection = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=5)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE" if writable else "BEGIN")
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def _fingerprint(connection) -> str:
    sources = [
        list(row)
        for row in connection.execute(
            "SELECT source_id, digest FROM documents ORDER BY source_id"
        )
    ]
    return _hash(_json({"config": _config(), "sources": sources}))


def _state(connection, *, create: bool = False) -> dict:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    if not tables and create:
        statements = [
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            "CREATE TABLE documents ("
            + ", ".join(f"{name} TEXT NOT NULL" for name in SOURCE_COLUMNS)
            + ", digest TEXT NOT NULL, PRIMARY KEY (source_id))",
            """CREATE TABLE chunks (
                rowid INTEGER PRIMARY KEY,
                chunk_id TEXT NOT NULL UNIQUE,
                source_id TEXT NOT NULL REFERENCES documents ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                start INTEGER NOT NULL, end INTEGER NOT NULL,
                line_start INTEGER NOT NULL, line_end INTEGER NOT NULL,
                text TEXT NOT NULL, headings TEXT NOT NULL,
                UNIQUE (source_id, ordinal)
            )""",
            f'CREATE VIRTUAL TABLE chunk_fts USING fts5(body, tokenize="{TOKENIZER}")',
        ]
        for statement in statements:
            connection.execute(statement)
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [
                ("config", _json(_config())),
                ("generation", "0"),
                ("fingerprint", _fingerprint(connection)),
            ],
        )
    elif not {"metadata", "documents", "chunks", "chunk_fts"}.issubset(tables):
        raise ValueError("This file is not a document index; use a new index path.")
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    if version != SCHEMA_VERSION or metadata.get("config") != _json(_config()):
        raise ValueError("Index format changed; build a new index.")
    try:
        return {
            "generation": int(metadata["generation"]),
            "fingerprint": metadata["fingerprint"],
            "config": json.loads(metadata["config"]),
        }
    except (KeyError, ValueError) as error:
        raise ValueError("Index metadata is invalid; build a new index.") from error


def _remove_source(connection, source_id: str) -> None:
    connection.execute(
        "DELETE FROM chunk_fts WHERE rowid IN "
        "(SELECT rowid FROM chunks WHERE source_id = ?)",
        (source_id,),
    )
    connection.execute("DELETE FROM documents WHERE source_id = ?", (source_id,))


def _insert_source(connection, document: Document, chunks, digest: str) -> None:
    values = [getattr(document, name) for name in SOURCE_COLUMNS] + [digest]
    placeholders = ", ".join("?" for _ in values)
    connection.execute(f"INSERT INTO documents VALUES ({placeholders})", values)
    for chunk in chunks:
        cursor = connection.execute(
            "INSERT INTO chunks (chunk_id, source_id, ordinal, start, end, "
            "line_start, line_end, text, headings) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                chunk.chunk_id,
                chunk.source_id,
                chunk.ordinal,
                chunk.start,
                chunk.end,
                chunk.line_start,
                chunk.line_end,
                chunk.text,
                _json(chunk.headings),
            ),
        )
        connection.execute(
            "INSERT INTO chunk_fts (rowid, body) VALUES (?, ?)",
            (cursor.lastrowid, "\n".join((*chunk.headings, chunk.text))),
        )


def update_index(path: Path, documents: list[Document], *, sync: bool = False) -> dict:
    prepared = {}
    for document in documents:
        if not document.source_id or document.source_id in prepared:
            raise ValueError("Source IDs must be nonempty and unique.")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", document.raw_sha256)
            or document.canonical_sha256 != _hash(document.text)
            or "\r" in document.text
        ):
            raise ValueError(f"Invalid source text or hash: {document.source_id}")
        prepared[document.source_id] = (
            document,
            chunk_document(document),
            _hash(_json(asdict(document))),
        )
    with _connection(path, writable=True) as connection:
        state = _state(connection, create=True)
        existing = dict(connection.execute("SELECT source_id, digest FROM documents"))
        removed = set(existing) - set(prepared) if sync else set()
        changed = {
            source_id
            for source_id, (_, _, digest) in prepared.items()
            if existing.get(source_id) != digest
        }
        for source_id in sorted(removed | changed):
            _remove_source(connection, source_id)
        for source_id in sorted(changed):
            _insert_source(connection, *prepared[source_id])
        if removed or changed:
            embedding_tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name IN ('embedding_state', 'chunk_vectors')"
                )
            }
            if embedding_tables:
                if len(embedding_tables) != 2:
                    raise ValueError(
                        "Embedding tables are incomplete; rebuild the index."
                    )
                connection.execute("DELETE FROM chunk_vectors")
                connection.execute("DELETE FROM embedding_state")
            state["generation"] += 1
            state["fingerprint"] = _fingerprint(connection)
            connection.executemany(
                "UPDATE metadata SET value = ? WHERE key = ?",
                [
                    (str(state["generation"]), "generation"),
                    (state["fingerprint"], "fingerprint"),
                ],
            )
        return {
            "generation": state["generation"],
            "fingerprint": state["fingerprint"],
            "updated": len(changed),
            "unchanged": len(prepared) - len(changed),
            "deleted": len(removed),
            "documents": connection.execute(
                "SELECT count(*) FROM documents"
            ).fetchone()[0],
            "chunks": connection.execute("SELECT count(*) FROM chunks").fetchone()[0],
        }


def index_inventory(path: Path, *, include_text: bool = False) -> dict:
    with _connection(path) as connection:
        state = _state(connection)
        documents = []
        for row in connection.execute("SELECT * FROM documents ORDER BY source_id"):
            document = dict(row)
            document.pop("digest")
            document["license_sha256"] = _hash(document.pop("license_text"))
            if not include_text:
                document.pop("text")
            documents.append(document)
        return {**state, "documents": documents}


def _query_expression(query: str) -> str:
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"Keep queries within {MAX_QUERY_CHARS} characters.")
    terms = re.findall(r"\w+(?:[-./]\w+)*", query, flags=re.UNICODE)
    if len(terms) > MAX_QUERY_TERMS:
        raise ValueError(f"Keep queries within {MAX_QUERY_TERMS} terms.")
    return " OR ".join(f'"{term}"' for term in dict.fromkeys(terms))


def search_index(
    path: Path,
    query: str,
    *,
    limit: int = 5,
    source_ids: list[str] | None = None,
) -> dict:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("Result limit must be between 1 and 100.")
    expression = _query_expression(query)
    with _connection(path) as connection:
        state = _state(connection)
        scope = _validated_scope(connection, source_ids)
        return {
            "generation": state["generation"],
            "fingerprint": state["fingerprint"],
            "method": "fts5-bm25",
            "results": _lexical_results(connection, expression, limit, scope),
        }


def _validated_scope(connection, source_ids):
    scope = sorted(set(source_ids)) if source_ids is not None else None
    if scope is not None:
        known = {
            row[0] for row in connection.execute("SELECT source_id FROM documents")
        }
        unknown = set(scope) - known
        if unknown:
            raise ValueError("Unknown source IDs: " + ", ".join(sorted(unknown)))
    return scope


def _lexical_results(connection, expression: str, limit: int, scope) -> list[dict]:
    if not expression or scope == []:
        return []
    parameters = [expression]
    scope_sql = ""
    if scope is not None:
        scope_sql = " AND c.source_id IN (" + ",".join("?" for _ in scope) + ")"
        parameters.extend(scope)
    parameters.append(limit)
    rows = connection.execute(
        "SELECT c.chunk_id, c.source_id, d.revision, d.title, d.url, d.license_id, "
        "c.start, c.end, c.line_start, c.line_end, c.text, c.headings, "
        "bm25(chunk_fts) AS score FROM chunk_fts "
        "JOIN chunks c ON c.rowid = chunk_fts.rowid "
        "JOIN documents d ON d.source_id = c.source_id "
        "WHERE chunk_fts MATCH ?"
        + scope_sql
        + " ORDER BY score, c.source_id, c.start, c.chunk_id LIMIT ?",
        parameters,
    )
    output = []
    for rank, row in enumerate(rows, start=1):
        result = dict(row)
        result["headings"] = json.loads(result["headings"])
        result["rank"] = rank
        output.append(result)
    return output
