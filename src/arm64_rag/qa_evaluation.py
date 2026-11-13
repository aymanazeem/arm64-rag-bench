import hashlib
import json
import sqlite3
from pathlib import Path

from arm64_rag.answers import SYSTEM_PROMPT, answer_question
from arm64_rag.context import expand_sources
from arm64_rag.evaluation import _covered, _validate_contract, _validate_results
from arm64_rag.index import index_inventory


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _validate_lines(results: list[dict], documents: dict) -> None:
    for item in results:
        text = documents[item["source_id"]]["text"]
        for key, expected in (
            ("line_start", text.count("\n", 0, item["start"]) + 1),
            ("line_end", text.count("\n", 0, item["end"] - 1) + 1),
        ):
            if type(item.get(key)) is not int or item[key] != expected:
                raise ValueError("Source line numbers differ from canonical text.")


def evaluate_answers(
    index_path: Path,
    dataset_path: Path,
    manifest_path: Path,
    server,
    protocol: dict,
    *,
    on_case=None,
    on_case_start=None,
) -> dict:
    dataset_raw, manifest_raw = dataset_path.read_bytes(), manifest_path.read_bytes()
    dataset, manifest = json.loads(dataset_raw), json.loads(manifest_raw)
    if _sha(manifest_raw) != protocol["manifest"]["sha256"]:
        raise ValueError("The source manifest differs from the answer protocol.")
    if _sha(dataset_raw) not in {item["sha256"] for item in protocol["datasets"]}:
        raise ValueError("The questions differ from the answer protocol.")
    if dataset.get("manifest_sha256") != _sha(manifest_raw):
        raise ValueError("The dataset's source pin differs.")
    inventory = index_inventory(index_path, include_text=True)
    documents = _validate_contract(dataset, manifest, inventory)
    snapshot = (inventory["generation"], inventory["fingerprint"])
    scopes = {scope["id"]: scope["source_ids"] for scope in manifest["corpora"]}
    records = []
    for case in dataset["cases"]:
        if on_case_start is not None:
            on_case_start(case)
        try:
            record = answer_question(
                index_path,
                case["question"],
                server,
                protocol,
                source_ids=scopes[case["corpus_id"]],
            )
        except (ValueError, OSError, RuntimeError, sqlite3.Error) as error:
            record = {
                "question": case["question"],
                "runtime": server.info(),
                "raw_completion": getattr(error, "response", None),
                "answer": None,
                "failure": str(error),
            }
        snapshot_changed = (
            "retrieval" in record
            and (record["retrieval"]["generation"], record["retrieval"]["fingerprint"])
            != snapshot
        )
        results, anchors, sources = [], [], []
        if "retrieval" in record:
            try:
                if snapshot_changed:
                    raise ValueError("The index changed during answer evaluation.")
                candidates = record["retrieval"]["results"]
                _validate_results(candidates, documents, scopes[case["corpus_id"]])
                _validate_lines(candidates, documents)
                results = candidates
                selected = record["packed"]["anchors"]
                if not isinstance(selected, list):
                    raise ValueError("Packed anchors must be a list.")
                available = {item["chunk_id"]: item for item in candidates}
                seen = set()
                for number, source in enumerate(selected, 1):
                    if not isinstance(source, dict):
                        raise ValueError("Invalid packed source.")
                    key = source.get("chunk_id")
                    if not isinstance(key, str) or key not in available or key in seen:
                        raise ValueError("Packed source is missing or repeated.")
                    if source != {**available[key], "label": f"S{number}"}:
                        raise ValueError(
                            "Packed source differs from its retrieved passage."
                        )
                    seen.add(key)
                _validate_lines(selected, documents)
                anchors = selected
                packed = record["packed"]["sources"]
                if not isinstance(packed, list) or len(packed) != len(anchors):
                    raise ValueError("Packed sources must retain every anchor.")
                expanded = expand_sources(index_path, anchors, record["retrieval"])
                normalized = []
                for number, (source, anchor, context) in enumerate(
                    zip(packed, anchors, expanded, strict=True), 1
                ):
                    if source not in (anchor, context):
                        raise ValueError(
                            "Packed source differs from its canonical context."
                        )
                    if (
                        not source["start"]
                        <= anchor["start"]
                        < anchor["end"]
                        <= source["end"]
                    ):
                        raise ValueError("Expanded context omits its original anchor.")
                    normalized.append(
                        {**source, "chunk_id": anchor["chunk_id"], "rank": number}
                    )
                _validate_results(normalized, documents, scopes[case["corpus_id"]])
                _validate_lines(packed, documents)
                sources = packed
            except (
                KeyError,
                TypeError,
                ValueError,
                OSError,
                RuntimeError,
                sqlite3.Error,
            ) as error:
                record["answer"] = None
                previous = record["failure"]
                record["source_validation_failure"] = str(error)
                record["failure"] = f"Source validation failed: {error}"
                if previous:
                    record["failure"] += f" Earlier failure: {previous}"
        record["id"] = case["id"]
        record["expected_answerability"] = case["answerability"]
        record["evidence"] = []
        for key in case["evidence_ids"]:
            span = dataset["evidence"][key]
            record["evidence"].append(
                {
                    "id": key,
                    "retrieved": _covered(span, results),
                    "included_before_expansion": _covered(span, anchors),
                    "included": _covered(span, sources),
                }
            )
        records.append(record)
        if on_case is not None:
            on_case(record)
        if snapshot_changed:
            raise ValueError("The index changed during answer evaluation.")
    final = index_inventory(index_path)
    if (final["generation"], final["fingerprint"]) != snapshot:
        raise ValueError("The index changed during answer evaluation.")
    with_evidence = [record for record in records if record["evidence"]]
    return {
        "schema_version": 1,
        "kind": "answer-evaluation",
        "dataset_id": dataset["id"],
        "split": dataset["split"],
        "dataset_sha256": _sha(dataset_raw),
        "manifest_sha256": _sha(manifest_raw),
        "protocol_sha256": _sha(json.dumps(protocol, sort_keys=True).encode()),
        "protocol_hash_format": "Python json.dumps with sort_keys=True; UTF-8.",
        "system_prompt_sha256": _sha(SYSTEM_PROMPT.encode()),
        "runtime": server.info(),
        "index": {"generation": snapshot[0], "fingerprint": snapshot[1]},
        "summary": {
            "cases_attempted": len(records),
            "cases_failed": sum(record["failure"] is not None for record in records),
            "structurally_valid": sum(
                record["answer"] is not None for record in records
            ),
            "cases_with_evidence": len(with_evidence),
            "complete_evidence_in_prompt": sum(
                all(unit["included"] for unit in record["evidence"])
                for record in with_evidence
            ),
            "statuses": {
                status: sum(
                    record["answer"] is not None
                    and record["answer"]["status"] == status
                    for record in records
                )
                for status in (
                    "answered",
                    "insufficient_evidence",
                    "conflicting_evidence",
                )
            },
        },
        "review_status": "Facts and citation support require independent review.",
        "cases": records,
    }
