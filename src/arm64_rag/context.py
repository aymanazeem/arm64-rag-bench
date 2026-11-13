import hashlib
import json
from pathlib import Path

from arm64_rag.index import _connection, _state


def expand_sources(index_path: Path, sources: list[dict], snapshot: dict) -> list[dict]:
    expanded = []
    with _connection(index_path) as connection:
        state = _state(connection)
        if any(state[key] != snapshot[key] for key in ("generation", "fingerprint")):
            raise ValueError("The index changed after retrieval; repeat the question.")
        for source in sources:
            anchor = connection.execute(
                "SELECT c.*, d.revision, d.url, d.text AS document_text "
                "FROM chunks c JOIN documents d ON d.source_id = c.source_id "
                "WHERE c.chunk_id = ?",
                (source["chunk_id"],),
            ).fetchone()
            if anchor is None or any(
                source[key] != anchor[key]
                for key in (
                    "source_id",
                    "revision",
                    "url",
                    "start",
                    "end",
                    "line_start",
                    "line_end",
                    "text",
                )
            ):
                raise ValueError("An answer source does not match the indexed chunk.")
            headings = json.loads(anchor["headings"])
            if source["headings"] != headings:
                raise ValueError("An answer source has different indexed headings.")
            previous = connection.execute(
                "SELECT * FROM chunks WHERE source_id = ? AND ordinal = ?",
                (anchor["source_id"], anchor["ordinal"] - 1),
            ).fetchone()
            parent = json.loads(previous["headings"]) if previous is not None else []
            if not parent or not (
                len(parent) < len(headings) and headings[: len(parent)] == parent
            ):
                expanded.append(source)
                continue
            window = [previous, anchor]
            first, last = window[0], window[-1]
            text = anchor["document_text"][first["start"] : last["end"]]
            identity = {
                "source_id": source["source_id"],
                "revision": source["revision"],
                "start": first["start"],
                "end": last["end"],
                "text": text,
            }
            passage_id = hashlib.sha256(
                json.dumps(
                    identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            passage = {
                **source,
                **identity,
                "line_start": first["line_start"],
                "line_end": last["line_end"],
                "headings": json.loads(first["headings"]),
                "anchor_chunk_id": source["chunk_id"],
                "context_chunk_ids": [row["chunk_id"] for row in window],
                "passage_id": passage_id,
            }
            passage.pop("chunk_id")
            expanded.append(passage)
    return expanded
