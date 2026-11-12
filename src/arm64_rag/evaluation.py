import hashlib
import json
import math
import sqlite3
import time
from pathlib import Path

from arm64_rag.index import index_inventory, search_index


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _records_by_id(records: list[dict], label: str) -> dict[str, dict]:
    _require(isinstance(records, list), f"Invalid {label} list.")
    result = {}
    for record in records:
        _require(isinstance(record, dict), f"Invalid {label} record.")
        key = record.get("id")
        _require(isinstance(key, str) and bool(key), f"Invalid {label} ID.")
        _require(key not in result, f"Duplicate {label} ID: {key}.")
        result[key] = record
    return result


def _validate_contract(dataset: dict, manifest: dict, inventory: dict) -> dict:
    _require(dataset.get("schema_version") == 1, "Unsupported dataset schema.")
    _require(dataset.get("split") == "development", "Expected a development dataset.")
    _require(manifest.get("schema_version") == 1, "Unsupported manifest schema.")
    _require(
        manifest.get("canonical_format") == "utf8-lf-v1",
        "Unsupported canonical text format.",
    )
    sources = _records_by_id(manifest.get("sources"), "source")
    corpora = _records_by_id(manifest.get("corpora"), "corpus")
    documents = {item["source_id"]: item for item in inventory["documents"]}
    for source_id, source in sources.items():
        _require(source_id in documents, f"Source is not indexed: {source_id}.")
        document = documents[source_id]
        for indexed, expected in (
            ("revision", "revision"),
            ("raw_sha256", "sha256"),
            ("canonical_sha256", "canonical_sha256"),
            ("license_id", "license_id"),
        ):
            _require(
                indexed in document
                and expected in source
                and document[indexed] == source[expected],
                f"Indexed source differs from the manifest: {source_id}.",
            )
        text = document["text"]
        _require(
            len(text) == source.get("canonical_chars")
            and _sha256(text.encode()) == source["canonical_sha256"],
            f"Indexed text differs from the manifest: {source_id}.",
        )
    _require(
        documents.keys() == sources.keys(),
        "Evaluation needs exactly the manifest sources in a dedicated corpus index.",
    )
    for corpus_id, corpus in corpora.items():
        scope = corpus.get("source_ids")
        _require(
            isinstance(scope, list)
            and bool(scope)
            and all(isinstance(key, str) and key in sources for key in scope)
            and len(scope) == len(set(scope)),
            f"Invalid source scope: {corpus_id}.",
        )
    evidence = dataset.get("evidence")
    _require(isinstance(evidence, dict), "Invalid evidence map.")
    for evidence_id, span in evidence.items():
        _require(isinstance(span, dict), f"Invalid evidence: {evidence_id}.")
        source_id = span.get("source_id")
        _require(
            isinstance(source_id, str) and source_id in sources,
            f"Unknown evidence source: {evidence_id}.",
        )
        text = documents[source_id]["text"]
        start, end = span.get("start"), span.get("end")
        _require(
            type(start) is int and type(end) is int and 0 <= start < end <= len(text),
            f"Invalid evidence offsets: {evidence_id}.",
        )
        _require(
            _sha256(text[start:end].encode()) == span.get("sha256"),
            f"Evidence hash differs: {evidence_id}.",
        )
        lines = [text.count("\n", 0, start) + 1, text.count("\n", 0, end - 1) + 1]
        _require(lines == span.get("lines"), f"Evidence lines differ: {evidence_id}.")
    cases = _records_by_id(dataset.get("cases"), "case")
    _require(bool(cases), "The dataset has no cases.")
    for case_id, case in cases.items():
        _require(
            isinstance(case.get("corpus_id"), str) and case["corpus_id"] in corpora,
            f"Unknown scope: {case_id}.",
        )
        question = case.get("question")
        _require(
            isinstance(question, str) and bool(question.strip()),
            f"Missing question: {case_id}.",
        )
        answerability = case.get("answerability")
        _require(
            isinstance(answerability, str)
            and answerability in {"answerable", "unanswerable", "conflicting"},
            f"Invalid answerability: {case_id}.",
        )
        keys = case.get("evidence_ids")
        _require(
            isinstance(keys, list)
            and all(isinstance(key, str) and key in evidence for key in keys)
            and len(keys) == len(set(keys)),
            f"Invalid evidence references: {case_id}.",
        )
        scope = corpora[case["corpus_id"]]["source_ids"]
        _require(
            all(evidence[key]["source_id"] in scope for key in keys),
            f"Evidence is outside the case scope: {case_id}.",
        )
        _require(
            (not keys and bool(case.get("absence_rationale")))
            if answerability == "unanswerable"
            else bool(keys),
            f"Evidence does not match answerability: {case_id}.",
        )
        if answerability == "conflicting":
            _require(len(keys) >= 2, f"A conflict needs both records: {case_id}.")
    return documents


def _validate_results(results: list[dict], documents: dict, scope: list[str]) -> None:
    _require(isinstance(results, list), "Search results must be a list.")
    seen = set()
    for rank, item in enumerate(results, 1):
        _require(isinstance(item, dict), "Invalid search result.")
        source_id = item.get("source_id")
        _require(source_id in scope, "Search returned a source outside the scope.")
        document = documents[source_id]
        text = document["text"]
        start, end = item.get("start"), item.get("end")
        _require(
            type(start) is int and type(end) is int and 0 <= start < end <= len(text),
            "Search returned invalid source offsets.",
        )
        _require(
            item.get("text") == text[start:end]
            and item.get("revision") == document["revision"],
            "Search returned different source text or revision.",
        )
        score = item.get("score")
        _require(
            type(score) in (int, float) and math.isfinite(score),
            "Search returned an invalid score.",
        )
        chunk_id = item.get("chunk_id")
        _require(
            isinstance(chunk_id, str) and chunk_id not in seen,
            "Search returned a missing or duplicate chunk ID.",
        )
        _require(item.get("rank") == rank, "Search returned invalid ranks.")
        seen.add(chunk_id)


def _covered(span: dict, results: list[dict]) -> bool:
    position = span["start"]
    intervals = sorted(
        (item["start"], item["end"])
        for item in results
        if item["source_id"] == span["source_id"]
    )
    for start, end in intervals:
        if start > position:
            break
        position = max(position, end)
        if position >= span["end"]:
            return True
    return False


def evaluate_retrieval(
    index_path: Path,
    dataset_path: Path,
    manifest_path: Path,
    *,
    limit: int = 5,
) -> dict:
    _require(type(limit) is int and 1 <= limit <= 100, "Limit must be from 1 to 100.")
    dataset_raw = dataset_path.read_bytes()
    manifest_raw = manifest_path.read_bytes()
    dataset, manifest = json.loads(dataset_raw), json.loads(manifest_raw)
    _require(isinstance(dataset, dict), "Invalid dataset.")
    _require(isinstance(manifest, dict), "Invalid manifest.")
    _require(
        dataset.get("manifest_sha256") == _sha256(manifest_raw),
        "The manifest differs from the dataset pin.",
    )
    inventory = index_inventory(index_path, include_text=True)
    documents = _validate_contract(dataset, manifest, inventory)
    scopes = {item["id"]: item["source_ids"] for item in manifest["corpora"]}
    snapshot = (inventory["generation"], inventory["fingerprint"])
    cases = []
    for case in dataset["cases"]:
        started = time.perf_counter()
        failure, results, response = None, [], None
        try:
            response = search_index(
                index_path,
                case["question"],
                limit=limit,
                source_ids=scopes[case["corpus_id"]],
            )
            _require(isinstance(response, dict), "Invalid search response.")
            _require(response.get("method") == "fts5-bm25", "Unexpected search method.")
        except (OSError, sqlite3.Error, ValueError) as exc:
            failure, response = str(exc), None
        if response is not None:
            _require(
                (response.get("generation"), response.get("fingerprint")) == snapshot,
                "The index changed during evaluation; rerun on one snapshot.",
            )
            try:
                results = response["results"]
                _require(len(results) <= limit, "Search exceeded the result limit.")
                _validate_results(results, documents, scopes[case["corpus_id"]])
            except (KeyError, TypeError, ValueError) as exc:
                failure, results = str(exc), []
        units = []
        for key in case["evidence_ids"]:
            span = dataset["evidence"][key]
            first_rank = next(
                (
                    rank
                    for rank in range(1, len(results) + 1)
                    if _covered(span, results[:rank])
                ),
                None,
            )
            units.append(
                {
                    "id": key,
                    "covered": first_rank is not None,
                    "first_covered_rank": first_rank,
                }
            )
        covered = [unit["id"] for unit in units if unit["covered"]]
        ranks = [unit["first_covered_rank"] for unit in units if unit["covered"]]
        cases.append(
            {
                "id": case["id"],
                "question": case["question"],
                "corpus_id": case["corpus_id"],
                "answerability": case["answerability"],
                "results": results,
                "evidence": units,
                "covered_evidence_ids": covered,
                "missing_evidence_ids": [
                    unit["id"] for unit in units if not unit["covered"]
                ],
                "any_hit": bool(covered) if units else None,
                "all_covered": len(covered) == len(units) if units else None,
                "first_evidence_rank": min(ranks) if ranks else None,
                "all_evidence_rank": (
                    max(ranks) if units and len(covered) == len(units) else None
                ),
                "failure": failure,
                "seconds": time.perf_counter() - started,
            }
        )
    final = index_inventory(index_path)
    _require(
        (final["generation"], final["fingerprint"]) == snapshot,
        "The index changed during evaluation; rerun on one snapshot.",
    )
    supported = [case for case in cases if case["evidence"]]
    total_units = sum(len(case["evidence"]) for case in supported)
    covered_units = sum(len(case["covered_evidence_ids"]) for case in supported)
    hits = sum(case["any_hit"] for case in supported)
    complete = sum(case["all_covered"] for case in supported)
    return {
        "schema_version": 1,
        "kind": "retrieval-development",
        "dataset_id": dataset.get("id"),
        "split": dataset["split"],
        "provenance": dataset.get("provenance"),
        "dataset_sha256": _sha256(dataset_raw),
        "manifest_sha256": _sha256(manifest_raw),
        "index": {
            "generation": inventory["generation"],
            "fingerprint": inventory["fingerprint"],
            "config": inventory["config"],
            "sqlite_version": sqlite3.sqlite_version,
        },
        "method": "fts5-bm25",
        "limit": limit,
        "scoring": "Retrieved spans must jointly cover each complete evidence span.",
        "timing": "Per-case wall time includes search and coverage scoring.",
        "limits": "Retrieval only; no answer, citation or abstention score.",
        "summary": {
            "cases_total": len(cases),
            "cases_with_evidence": len(supported),
            "cases_without_evidence": len(cases) - len(supported),
            "cases_failed": sum(case["failure"] is not None for case in cases),
            "cases_any_hit": hits,
            "cases_all_covered": complete,
            "evidence_units_total": total_units,
            "evidence_units_covered": covered_units,
            "hit_rate": hits / len(supported) if supported else None,
            "all_covered_rate": complete / len(supported) if supported else None,
            "evidence_coverage_rate": covered_units / total_units
            if total_units
            else None,
        },
        "cases": cases,
    }
