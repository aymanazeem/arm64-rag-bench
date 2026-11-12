import hashlib
import json
import sqlite3
from dataclasses import replace

import numpy as np
import pytest

from arm64_rag import evaluation
from arm64_rag.dense import embed_index, embedding_info
from arm64_rag.documents import Document
from arm64_rag.index import index_inventory, update_index


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def corpus(tmp_path):
    documents = []
    sources, evidence = [], {}
    for source_id, text in (("a", "alpha café four\n"), ("b", "alpha café eight\n")):
        raw = text.encode()
        sha = digest(raw)
        documents.append(
            Document(
                source_id=source_id,
                revision="1",
                text=text,
                raw_sha256=sha,
                canonical_sha256=sha,
                title=source_id,
                url="",
                license_id="fixture",
                license_text="Original fixture.",
                format="text",
            )
        )
        sources.append(
            {
                "id": source_id,
                "revision": "1",
                "sha256": sha,
                "canonical_sha256": sha,
                "canonical_chars": len(text),
                "license_id": "fixture",
                "bytes": len(raw),
            }
        )
        evidence[source_id] = {
            "source_id": source_id,
            "start": 0,
            "end": len(text),
            "lines": [1, 1],
            "sha256": sha,
        }
    manifest = {
        "schema_version": 1,
        "canonical_format": "utf8-lf-v1",
        "sources": sources,
        "corpora": [{"id": "both", "source_ids": ["a", "b"]}],
    }
    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, manifest)
    dataset = {
        "schema_version": 1,
        "id": "test-development",
        "split": "development",
        "manifest_sha256": digest(manifest_path.read_bytes()),
        "provenance": {"human_validated": False},
        "evidence": evidence,
        "cases": [
            {
                "id": "one",
                "corpus_id": "both",
                "question": "four",
                "answerability": "answerable",
                "evidence_ids": ["a"],
            },
            {
                "id": "conflict",
                "corpus_id": "both",
                "question": "alpha",
                "answerability": "conflicting",
                "evidence_ids": ["a", "b"],
            },
            {
                "id": "missing",
                "corpus_id": "both",
                "question": "memory",
                "answerability": "unanswerable",
                "evidence_ids": [],
                "absence_rationale": "Neither source records memory.",
            },
        ],
    }
    dataset_path = tmp_path / "dataset.json"
    write_json(dataset_path, dataset)
    index_path = tmp_path / "index.sqlite3"
    update_index(index_path, documents)
    return index_path, dataset_path, manifest_path, dataset, manifest, documents


def evaluate(corpus, **kwargs):
    return evaluation.evaluate_retrieval(*corpus[:3], **kwargs)


def result(document, start, end, rank):
    return {
        "source_id": document.source_id,
        "revision": document.revision,
        "chunk_id": f"{document.source_id}-{start}-{end}",
        "start": start,
        "end": end,
        "text": document.text[start:end],
        "headings": [],
        "rank": rank,
        "score": -1.0,
    }


def stub_search(monkeypatch, corpus, results):
    inventory = index_inventory(corpus[0])
    response = {
        "generation": inventory["generation"],
        "fingerprint": inventory["fingerprint"],
        "method": "fts5-bm25",
        "results": results,
    }
    monkeypatch.setattr(evaluation, "search_index", lambda *args, **kwargs: response)


def test_real_retrieval_reports_counts_and_no_answer_quality(corpus):
    report = evaluate(corpus)
    assert report["summary"] == {
        "cases_total": 3,
        "cases_with_evidence": 2,
        "cases_without_evidence": 1,
        "cases_failed": 0,
        "cases_any_hit": 2,
        "cases_all_covered": 2,
        "evidence_units_total": 3,
        "evidence_units_covered": 3,
        "hit_rate": 1.0,
        "all_covered_rate": 1.0,
        "evidence_coverage_rate": 1.0,
    }
    assert report["cases"][0]["first_evidence_rank"] == 1
    assert report["cases"][0]["all_evidence_rank"] == 1
    assert report["cases"][1]["covered_evidence_ids"] == ["a", "b"]
    assert report["cases"][1]["first_evidence_rank"] == 1
    assert report["cases"][1]["all_evidence_rank"] == 2
    assert report["cases"][2]["any_hit"] is None
    assert report["cases"][2]["all_covered"] is None
    assert report["cases"][2]["first_evidence_rank"] is None
    assert report["cases"][2]["all_evidence_rank"] is None
    assert report["dataset_sha256"] == digest(corpus[1].read_bytes())
    assert report["manifest_sha256"] == digest(corpus[2].read_bytes())
    assert report["index"]["sqlite_version"] == sqlite3.sqlite_version
    assert all(case["seconds"] >= 0 for case in report["cases"])


def test_adjacent_spans_cover_one_unit_only_at_combined_rank(corpus, monkeypatch):
    document = corpus[-1][0]
    results = [result(document, 6, len(document.text), 1), result(document, 0, 6, 2)]
    stub_search(monkeypatch, corpus, results)
    report = evaluate(corpus)
    assert report["cases"][0]["first_evidence_rank"] == 2
    assert report["cases"][0]["all_evidence_rank"] == 2
    assert report["cases"][0]["evidence"][0]["first_covered_rank"] == 2
    assert report["cases"][1]["all_covered"] is False
    assert report["cases"][1]["all_evidence_rank"] is None


@pytest.mark.parametrize("different_source", [False, True])
def test_overlap_or_other_sources_cannot_fill_missing_evidence(
    corpus, monkeypatch, different_source
):
    document, other = corpus[-1]
    results = [result(document, 0, 6, 1)]
    if different_source:
        results.append(result(other, 6, len(other.text), 2))
    else:
        results.append(result(document, 7, len(document.text), 2))
    stub_search(monkeypatch, corpus, results)
    first = evaluate(corpus)["cases"][0]
    assert first["any_hit"] is False
    assert first["missing_evidence_ids"] == ["a"]
    assert first["first_evidence_rank"] is None
    assert first["all_evidence_rank"] is None


def test_first_and_all_evidence_ranks_differ_for_multiple_units(corpus, monkeypatch):
    first, second = corpus[-1]
    results = [
        result(first, 0, 6, 1),
        result(second, 0, len(second.text), 2),
        result(first, 6, len(first.text), 3),
    ]
    stub_search(monkeypatch, corpus, results)
    conflict = evaluate(corpus)["cases"][1]
    assert conflict["first_evidence_rank"] == 2
    assert conflict["all_evidence_rank"] == 3
    assert conflict["all_covered"] is True


def test_failures_remain_in_the_denominator(corpus, monkeypatch):
    original = evaluation.search_index

    def failing_search(path, query, **kwargs):
        if query == "four":
            raise sqlite3.OperationalError("Database is busy.")
        return original(path, query, **kwargs)

    monkeypatch.setattr(evaluation, "search_index", failing_search)
    report = evaluate(corpus)
    assert report["summary"]["cases_failed"] == 1
    assert report["summary"]["cases_with_evidence"] == 2
    assert report["summary"]["evidence_units_total"] == 3
    assert report["summary"]["hit_rate"] == 0.5
    assert report["cases"][0]["failure"] == "Database is busy."
    assert report["cases"][0]["missing_evidence_ids"] == ["a"]


def test_changed_manifest_is_rejected(corpus):
    corpus[2].write_text(corpus[2].read_text() + "\n")
    with pytest.raises(ValueError, match="manifest differs"):
        evaluate(corpus)


def test_extra_indexed_sources_are_rejected(corpus):
    update_index(corpus[0], [replace(corpus[-1][0], source_id="extra")])
    with pytest.raises(ValueError, match="exactly the manifest sources"):
        evaluate(corpus)


@pytest.mark.parametrize("change", ["revision", "content", "missing"])
def test_stale_or_missing_index_source_is_rejected(corpus, change):
    document = corpus[-1][0]
    if change == "missing":
        update_index(corpus[0], corpus[-1][1:], sync=True)
    elif change == "revision":
        update_index(corpus[0], [replace(document, revision="2")])
    else:
        text = document.text + "Changed.\n"
        sha = digest(text.encode())
        update_index(
            corpus[0],
            [
                replace(
                    document,
                    text=text,
                    raw_sha256=sha,
                    canonical_sha256=sha,
                )
            ],
        )
    with pytest.raises(ValueError, match="not indexed|differs from the manifest"):
        evaluate(corpus)


@pytest.mark.parametrize(
    "field,value",
    [
        ("start", -1),
        ("end", 999),
        ("sha256", "0" * 64),
        ("lines", [2, 2]),
    ],
)
def test_bad_evidence_is_rejected_before_search(corpus, field, value):
    dataset = corpus[3]
    dataset["evidence"]["a"][field] = value
    write_json(corpus[1], dataset)
    with pytest.raises(ValueError, match="evidence|Evidence"):
        evaluate(corpus)


def test_case_cannot_use_evidence_outside_its_scope(corpus):
    manifest = corpus[4]
    manifest["corpora"][0]["source_ids"] = ["b"]
    write_json(corpus[2], manifest)
    dataset = corpus[3]
    dataset["manifest_sha256"] = digest(corpus[2].read_bytes())
    write_json(corpus[1], dataset)
    with pytest.raises(ValueError, match="outside the case scope"):
        evaluate(corpus)


def test_index_change_during_evaluation_is_rejected(corpus, monkeypatch):
    original = evaluation.search_index

    def changing_search(*args, **kwargs):
        response = original(*args, **kwargs)
        response["generation"] += 1
        return response

    monkeypatch.setattr(evaluation, "search_index", changing_search)
    with pytest.raises(ValueError, match="index changed"):
        evaluate(corpus)


def test_invalid_result_is_failure_not_evidence(corpus, monkeypatch):
    document = corpus[-1][0]
    forged = result(document, 0, len(document.text), 1)
    forged["text"] = "Different evidence."
    stub_search(monkeypatch, corpus, [forged])
    report = evaluate(corpus)
    assert report["summary"]["cases_failed"] == 3
    assert report["summary"]["evidence_units_covered"] == 0
    assert report["summary"]["cases_with_evidence"] == 2


def test_wrong_split_and_duplicate_cases_are_rejected(corpus):
    corpus[3]["split"] = "heldout"
    write_json(corpus[1], corpus[3])
    with pytest.raises(ValueError, match="development dataset"):
        evaluate(corpus)
    corpus[3]["split"] = "development"
    corpus[3]["cases"].append(corpus[3]["cases"][0])
    write_json(corpus[1], corpus[3])
    with pytest.raises(ValueError, match="Duplicate case ID"):
        evaluate(corpus)


def test_no_gold_evidence_has_no_retrieval_quality_denominator(corpus):
    corpus[3]["cases"] = [corpus[3]["cases"][-1]]
    write_json(corpus[1], corpus[3])
    summary = evaluate(corpus)["summary"]
    assert summary["cases_total"] == 1
    assert summary["cases_with_evidence"] == 0
    assert summary["evidence_units_total"] == 0
    assert summary["hit_rate"] is None
    assert summary["all_covered_rate"] is None
    assert summary["evidence_coverage_rate"] is None


def test_missing_response_is_a_reported_failure(corpus, monkeypatch):
    monkeypatch.setattr(evaluation, "search_index", lambda *args, **kwargs: None)
    report = evaluate(corpus)
    assert report["summary"]["cases_failed"] == 3
    assert report["summary"]["cases_any_hit"] == 0


def test_index_change_after_last_search_is_rejected(corpus, monkeypatch):
    original = evaluation.search_index

    def changing_search(path, query, **kwargs):
        response = original(path, query, **kwargs)
        if query == "memory":
            update_index(path, [replace(corpus[-1][0], revision="2")])
        return response

    monkeypatch.setattr(evaluation, "search_index", changing_search)
    with pytest.raises(ValueError, match="index changed"):
        evaluate(corpus)


class SyntheticEncoder:
    config = {"dimension": 2, "model": "test", "revision": "1"}

    def encode_documents(self, texts):
        return [np.array([[1, 0] if "four" in text else [0, 1]]) for text in texts]

    def encode_queries(self, texts):
        return np.array([[1, 0] if "four" in text else [1, 1] for text in texts])

    def token_count(self, text):
        return len(text)


@pytest.mark.parametrize("method", ["dense", "hybrid"])
def test_dense_and_hybrid_evaluate_real_stored_vectors(corpus, method):
    encoder = SyntheticEncoder()
    state = embed_index(corpus[0], encoder.encode_documents, encoder.config)
    report = evaluate(corpus, method=method, encoder=encoder)
    assert report["method"] == {"dense": "dense-cosine", "hybrid": "hybrid-rrf"}[method]
    assert report["embedding"]["fingerprint"] == state["fingerprint"]
    assert report["embedding"]["config_hash"] == state["config_hash"]
    assert report["summary"]["cases_all_covered"] == 2
    assert report["summary"]["evidence_units_covered"] == 3
    assert report["summary"]["cases_failed"] == 0
    assert report["cases"][0]["results"][0]["source_id"] == "a"
    assert all(case["query_embedding_seconds"] >= 0 for case in report["cases"])
    assert report["cases"][2]["any_hit"] is None
    if method == "hybrid":
        assert report["fusion"] == {"constant": 60, "candidate_depth_per_method": 20}


@pytest.mark.parametrize("when", ["during", "after_last"])
def test_embedding_republication_is_detected_without_source_changes(
    corpus, monkeypatch, when
):
    from arm64_rag import dense

    encoder = SyntheticEncoder()
    state = embed_index(corpus[0], encoder.encode_documents, encoder.config)
    source_snapshot = index_inventory(corpus[0])
    original = dense.search_vectors

    def republishing_search(path, query, *args, **kwargs):
        response = original(path, query, *args, **kwargs)
        if query == ("four" if when == "during" else "memory"):
            embed_index(
                path,
                lambda texts: [
                    np.array([[0, 1] if "four" in text else [1, 0]]) for text in texts
                ],
                encoder.config,
            )
        return response

    monkeypatch.setattr(dense, "search_vectors", republishing_search)
    with pytest.raises(ValueError, match="Embeddings changed"):
        evaluate(corpus, method="dense", encoder=encoder)
    assert index_inventory(corpus[0]) == source_snapshot
    assert embedding_info(corpus[0])["fingerprint"] != state["fingerprint"]


def test_budget_skips_large_passages_without_truncating_or_losing_rank(
    corpus, monkeypatch
):
    first, second = corpus[-1]
    candidates = [
        result(second, 0, len(second.text), 1),
        result(first, 0, len(first.text), 2),
    ]
    stub_search(monkeypatch, corpus, candidates)
    report = evaluate(
        corpus, encoder=SyntheticEncoder(), token_budget=len(first.text), limit=2
    )
    case = report["cases"][0]
    assert case["results"][0]["source_id"] == "a"
    assert case["results"][0]["text"] == first.text
    assert case["results"][0]["retrieval_rank"] == 2
    assert case["results"][0]["rank"] == 1
    assert case["used_tokens"] == len(first.text)
    assert case["omitted"] == [
        {"chunk_id": candidates[0]["chunk_id"], "reason": "token_budget"}
    ]
    assert case["candidates"] == candidates
    conflict = report["cases"][1]
    assert conflict["candidate_covered_evidence_ids"] == ["a", "b"]
    assert conflict["covered_evidence_ids"] == ["a"]
    assert conflict["missing_evidence_ids"] == ["b"]
    assert report["summary"]["cases_with_evidence"] == 2
    assert report["summary"]["evidence_units_total"] == 3
    assert report["summary"]["evidence_units_covered"] == 2


def test_common_budget_counts_headings_and_separators(corpus, monkeypatch):
    first, second = corpus[-1]
    candidates = [
        {**result(first, 0, len(first.text), 1), "headings": ["Heading"]},
        {**result(second, 0, len(second.text), 2), "headings": ["Other"]},
    ]
    stub_search(monkeypatch, corpus, candidates)
    full_text = f"Heading\n{first.text}\n\nOther\n{second.text}"
    report = evaluate(
        corpus, encoder=SyntheticEncoder(), token_budget=len(full_text), limit=2
    )
    assert report["cases"][1]["used_tokens"] == len(full_text)
    assert report["cases"][1]["all_covered"] is True
    smaller = evaluate(
        corpus, encoder=SyntheticEncoder(), token_budget=len(full_text) - 1, limit=2
    )
    assert smaller["cases"][1]["missing_evidence_ids"] == ["b"]
    assert smaller["cases"][1]["omitted"][0]["reason"] == "token_budget"


def test_passage_limit_is_distinct_from_budget(corpus, monkeypatch):
    first, second = corpus[-1]
    candidates = [
        result(first, 0, len(first.text), 1),
        result(second, 0, len(second.text), 2),
    ]
    stub_search(monkeypatch, corpus, candidates)
    report = evaluate(corpus, encoder=SyntheticEncoder(), token_budget=100, limit=1)
    assert report["cases"][1]["omitted"] == [
        {"chunk_id": candidates[1]["chunk_id"], "reason": "passage_limit"}
    ]
    assert report["budget"]["candidate_depth"] == 20
    assert report["cases"][1]["all_evidence_rank"] is None


def test_tokenizer_failure_stays_in_denominators(corpus):
    encoder = SyntheticEncoder()

    def fail_on_evidence(text):
        if "four" in text:
            raise RuntimeError("Token counting failed.")
        return len(text)

    encoder.token_count = fail_on_evidence
    report = evaluate(corpus, encoder=encoder, token_budget=100)
    assert report["summary"]["cases_failed"] == 2
    assert report["summary"]["cases_with_evidence"] == 2
    assert report["summary"]["evidence_units_total"] == 3
    assert report["summary"]["evidence_units_covered"] == 0
    assert report["cases"][0]["results"] == []
    assert report["cases"][0]["failure"] == "Token counting failed."


def test_dense_encoder_failure_stays_in_denominators(corpus):
    encoder = SyntheticEncoder()
    embed_index(corpus[0], encoder.encode_documents, encoder.config)

    def unavailable(_):
        raise RuntimeError("Embedding runtime failed.")

    encoder.encode_queries = unavailable
    report = evaluate(corpus, method="dense", encoder=encoder)
    assert report["summary"]["cases_failed"] == 3
    assert report["summary"]["cases_with_evidence"] == 2
    assert report["summary"]["evidence_units_total"] == 3
    assert report["summary"]["evidence_units_covered"] == 0


def test_missing_result_headings_are_failure_not_budget_crash(corpus, monkeypatch):
    first = corpus[-1][0]
    malformed = result(first, 0, len(first.text), 1)
    malformed.pop("headings")
    stub_search(monkeypatch, corpus, [malformed])
    report = evaluate(corpus, encoder=SyntheticEncoder(), token_budget=100)
    assert report["summary"]["cases_failed"] == 3
    assert report["summary"]["evidence_units_covered"] == 0


def test_validation_split_keeps_its_identity(corpus):
    corpus[3]["split"] = "validation"
    write_json(corpus[1], corpus[3])
    report = evaluate(corpus)
    assert report["kind"] == "retrieval-validation"
    assert report["split"] == "validation"


@pytest.mark.parametrize("budget", [0, -1, True])
def test_invalid_common_budgets_are_rejected(corpus, budget):
    with pytest.raises(ValueError, match="budget"):
        evaluate(corpus, encoder=SyntheticEncoder(), token_budget=budget)
