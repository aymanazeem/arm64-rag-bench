import copy
import hashlib
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from arm64_rag import answers, qa_evaluation
from arm64_rag.documents import Document
from arm64_rag.index import update_index
from arm64_rag.inference import InferenceError

ROOT = Path(__file__).resolve().parents[1]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class FakeRuntime:
    def __init__(self, *, malformed=False, false_claim=False):
        self.malformed = malformed
        self.false_claim = false_claim
        self.calls = []
        self.sources = []
        self.render_calls = []
        self.settings = {}

    def render(self, messages, *, json_schema=None):
        self.render_calls.append(copy.deepcopy((messages, json_schema)))
        self.sources = json.loads(messages[-1]["content"])["sources"]
        return json.dumps(messages)

    def tokenize(self, prompt):
        return list(range(len(prompt)))

    def complete(self, token_ids, max_tokens, *, json_schema, messages=None):
        self.calls.append(copy.deepcopy((token_ids, max_tokens, json_schema, messages)))
        if messages is not None:
            self.sources = json.loads(messages[-1]["content"])["sources"]
        if self.malformed:
            return {"content": "unfinished JSON"}
        source = self.sources[0]
        return {
            "content": json.dumps(
                {
                    "status": "answered",
                    "claims": [
                        {
                            "text": (
                                "There are 999 workers."
                                if self.false_claim
                                else source["text"].strip()
                            ),
                            "citations": [{"label": source["label"]}],
                        }
                    ],
                }
            )
        }

    def info(self):
        return {
            "model_sha256": "test-model",
            "context_size": self.settings["context_size"],
            "reasoning_budget_tokens": self.settings.get("reasoning_budget_tokens"),
        }


@pytest.fixture
def corpus(tmp_path):
    documents, sources, evidence = [], [], {}
    for source_id, text in (
        ("a", "The alpha worker count is four.\n"),
        ("b", "The beta worker count is eight.\n"),
    ):
        sha = digest(text.encode())
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
                license_text="Original test text.",
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
        "id": "answer-fixture",
        "split": "development",
        "manifest_sha256": digest(manifest_path.read_bytes()),
        "evidence": evidence,
        "cases": [
            {
                "id": "one",
                "corpus_id": "both",
                "question": "alpha worker count",
                "answerability": "answerable",
                "evidence_ids": ["a"],
            },
            {
                "id": "conflict",
                "corpus_id": "both",
                "question": "worker count",
                "answerability": "conflicting",
                "evidence_ids": ["a", "b"],
            },
            {
                "id": "missing",
                "corpus_id": "both",
                "question": "peak memory",
                "answerability": "unanswerable",
                "evidence_ids": [],
                "absence_rationale": "Neither source records memory.",
            },
        ],
    }
    dataset_path = tmp_path / "dataset.json"
    write_json(dataset_path, dataset)
    protocol = json.loads((ROOT / "eval/answers-protocol.json").read_bytes())
    protocol["manifest"]["sha256"] = digest(manifest_path.read_bytes())
    protocol["datasets"] = [{"sha256": digest(dataset_path.read_bytes())}]
    index = tmp_path / "index.sqlite"
    update_index(index, documents)
    return {
        "paths": (index, dataset_path, manifest_path),
        "dataset": dataset,
        "documents": documents,
        "protocol": protocol,
    }


def evaluate(corpus, runtime=None, **kwargs):
    runtime = runtime or FakeRuntime()
    runtime.settings = corpus["protocol"]["settings"]
    return qa_evaluation.evaluate_answers(
        *corpus["paths"], runtime, corpus["protocol"], **kwargs
    )


def test_summary_keeps_failures_in_attempted_denominator(corpus):
    runtime = FakeRuntime(malformed=True)
    report = evaluate(corpus, runtime)
    assert report["summary"] == {
        "cases_attempted": 3,
        "cases_failed": 2,
        "structurally_valid": 1,
        "cases_with_evidence": 2,
        "complete_evidence_in_prompt": 2,
        "statuses": {
            "answered": 0,
            "insufficient_evidence": 1,
            "conflicting_evidence": 0,
        },
    }
    assert len(runtime.calls) == 2
    assert [case["id"] for case in report["cases"]] == ["one", "conflict", "missing"]
    assert report["cases"][0]["raw_completion"]["content"] == "unfinished JSON"
    assert report["cases"][0]["failure"]
    assert report["dataset_sha256"] == digest(corpus["paths"][1].read_bytes())
    assert report["manifest_sha256"] == digest(corpus["paths"][2].read_bytes())
    assert report["runtime"] == runtime.info()


def test_evidence_coverage_uses_included_context_not_candidate_list(corpus):
    corpus["protocol"]["settings"]["max_passages"] = 1
    report = evaluate(corpus)
    conflict = report["cases"][1]
    assert all(item["retrieved"] for item in conflict["evidence"])
    assert sum(item["included"] for item in conflict["evidence"]) == 1
    assert len(conflict["retrieval"]["results"]) == 2
    assert len(conflict["packed"]["sources"]) == 1
    assert report["summary"]["complete_evidence_in_prompt"] == 1


def test_valid_citation_does_not_mark_false_claim_as_factually_correct(corpus):
    report = evaluate(corpus, FakeRuntime(false_claim=True))
    case = report["cases"][0]
    assert case["answer"]["claims"][0]["text"] == "There are 999 workers."
    assert case["failure"] is None
    assert "unverified" in case["validation"]
    assert "independent review" in report["review_status"]
    assert report["summary"]["structurally_valid"] == 3
    assert not {"accuracy", "factual_accuracy", "citation_accuracy", "passed"} & set(
        report["summary"]
    )


@pytest.mark.parametrize("changed", ["dataset", "manifest", "dataset_manifest_pin"])
def test_frozen_input_pins_checked_before_generation(corpus, changed):
    runtime = FakeRuntime()
    if changed == "dataset_manifest_pin":
        corpus["dataset"]["manifest_sha256"] = "0" * 64
        write_json(corpus["paths"][1], corpus["dataset"])
        corpus["protocol"]["datasets"][0]["sha256"] = digest(
            corpus["paths"][1].read_bytes()
        )
    else:
        path = corpus["paths"][1 if changed == "dataset" else 2]
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="differs|differ"):
        evaluate(corpus, runtime)
    assert runtime.calls == []


@pytest.mark.parametrize(
    "changes",
    [
        {"revision": "2"},
        {"raw_sha256": "0" * 64},
        {"canonical_sha256": "0" * 64},
        {"license_id": "different"},
        {"text": "Changed text with an unchanged claimed hash.\n"},
    ],
)
def test_indexed_source_identity_must_match_frozen_manifest(corpus, changes):
    runtime = FakeRuntime()
    if "text" in changes or "canonical_sha256" in changes:
        with sqlite3.connect(corpus["paths"][0]) as connection:
            column, value = next(iter(changes.items()))
            connection.execute(
                f"UPDATE documents SET {column} = ? WHERE source_id = 'a'", (value,)
            )
    else:
        update_index(corpus["paths"][0], [replace(corpus["documents"][0], **changes)])
    with pytest.raises(ValueError, match="differs from the manifest"):
        evaluate(corpus, runtime)
    assert runtime.calls == []


def test_extra_indexed_source_is_not_silently_used(corpus):
    update_index(
        corpus["paths"][0], [replace(corpus["documents"][0], source_id="extra")]
    )
    with pytest.raises(ValueError, match="exactly the manifest sources"):
        evaluate(corpus)


def test_callback_receives_enriched_failures_before_next_case(corpus, monkeypatch):
    original = qa_evaluation.answer_question
    retained = []
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert len(retained) == calls - 1
        if calls == 1:
            raise RuntimeError("Startup failed.")
        return original(*args, **kwargs)

    monkeypatch.setattr(qa_evaluation, "answer_question", fail_once)
    report = evaluate(
        corpus, on_case=lambda record: retained.append(copy.deepcopy(record))
    )
    assert retained == report["cases"]
    assert retained[0]["id"] == "one"
    assert retained[0]["failure"] == "Startup failed."
    assert retained[0]["evidence"] == [
        {
            "id": "a",
            "retrieved": False,
            "included_before_expansion": False,
            "included": False,
        }
    ]
    assert report["summary"]["cases_attempted"] == 3
    assert report["summary"]["cases_failed"] == 1


def test_template_failure_retains_raw_response_and_runtime(corpus, monkeypatch):
    runtime = FakeRuntime()
    render = runtime.render
    retained = []
    raw_response = {
        "http_status": 502,
        "raw_body": "Invalid response from local server.",
        "body_encoding": "utf-8",
        "body_truncated": False,
    }

    def fail_once(messages, *, json_schema=None):
        monkeypatch.setattr(runtime, "render", render)
        raise InferenceError("Invalid JSON response.", response=raw_response)

    monkeypatch.setattr(runtime, "render", fail_once)
    report = evaluate(
        corpus, runtime, on_case=lambda record: retained.append(copy.deepcopy(record))
    )
    assert retained == report["cases"]
    assert retained[0]["failure"] == "Invalid JSON response."
    assert retained[0]["raw_completion"] == raw_response
    assert retained[0]["runtime"] == runtime.info()
    assert retained[0]["answer"] is None
    assert report["summary"]["cases_attempted"] == 3
    assert report["summary"]["cases_failed"] == 1


def test_callback_failure_stops_run_without_losing_prior_record(corpus):
    runtime = FakeRuntime()
    retained = []

    def interrupt(record):
        retained.append(copy.deepcopy(record))
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        evaluate(corpus, runtime, on_case=interrupt)
    assert [case["id"] for case in retained] == ["one"]
    assert len(runtime.calls) == 1


@pytest.mark.parametrize("after_case", ["one", "missing"])
def test_index_change_during_or_after_cases_invalidates_run(corpus, after_case):
    retained = []

    def mutate_index(record):
        retained.append(record["id"])
        if record["id"] == after_case:
            update_index(
                corpus["paths"][0],
                [replace(corpus["documents"][0], revision="2")],
            )

    with pytest.raises(ValueError, match="index changed during answer evaluation"):
        evaluate(corpus, on_case=mutate_index)
    assert retained == (
        ["one", "conflict"] if after_case == "one" else ["one", "conflict", "missing"]
    )


@pytest.mark.parametrize(
    ("column", "value"),
    [("text", "The alpha worker count is nine.\n"), ("line_start", 9), ("line_end", 9)],
)
def test_corrupt_chunk_fails_case_without_evidence_credit(corpus, column, value):
    with sqlite3.connect(corpus["paths"][0]) as connection:
        connection.execute(
            f"UPDATE chunks SET {column} = ? WHERE source_id = 'a'", (value,)
        )
    report = evaluate(corpus)
    case = report["cases"][0]
    assert case["failure"].startswith("Source validation failed:")
    assert case["answer"] is None
    assert case["raw_completion"]["content"]
    assert case["retrieval"]["results"][0][column] == value
    assert case["packed"]["prompt"]
    assert case["evidence"] == [
        {
            "id": "a",
            "retrieved": False,
            "included_before_expansion": False,
            "included": False,
        }
    ]
    assert report["summary"]["cases_attempted"] == 3
    assert report["summary"]["cases_failed"] == 2
    assert report["summary"]["complete_evidence_in_prompt"] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"text": "Invented source text."},
        {"chunk_id": "not-retrieved"},
        {"revision": "2"},
        {"label": "S2"},
        {"label": None},
        {"line_start": 9},
        {"start": 0.0},
    ],
)
def test_packed_source_changes_fail_without_discarding_raw_trace(
    corpus, monkeypatch, changes
):
    original = qa_evaluation.answer_question

    def corrupt_packed(*args, **kwargs):
        record = original(*args, **kwargs)
        if record["question"] == "alpha worker count":
            record["packed"]["sources"][0] = {
                **record["packed"]["sources"][0],
                **changes,
            }
        return record

    monkeypatch.setattr(qa_evaluation, "answer_question", corrupt_packed)
    report = evaluate(corpus)
    case = report["cases"][0]
    assert case["answer"] is None
    assert case["source_validation_failure"]
    assert case["raw_completion"]["content"]
    assert case["packed"]["sources"][0].items() >= changes.items()
    assert case["evidence"] == [
        {
            "id": "a",
            "retrieved": True,
            "included_before_expansion": True,
            "included": False,
        }
    ]
    assert report["summary"]["cases_failed"] == 1
    assert report["summary"]["complete_evidence_in_prompt"] == 1


@pytest.mark.parametrize("same_label", [True, False])
def test_repeated_packed_passage_fails_even_with_different_labels(
    corpus, monkeypatch, same_label
):
    original = qa_evaluation.answer_question

    def repeat_packed(*args, **kwargs):
        record = original(*args, **kwargs)
        if record["question"] == "alpha worker count":
            sources = record["packed"]["sources"]
            sources.append({**sources[0], "label": "S1" if same_label else "S3"})
        return record

    monkeypatch.setattr(qa_evaluation, "answer_question", repeat_packed)
    case = evaluate(corpus)["cases"][0]
    assert "retain every anchor" in case["failure"]
    assert case["answer"] is None
    assert not any(item["included"] for item in case["evidence"])


def test_start_callback_precedes_generation_and_completed_callback(corpus, monkeypatch):
    original = qa_evaluation.answer_question
    events = []

    def observe_generation(*args, **kwargs):
        events.append("generate")
        return original(*args, **kwargs)

    monkeypatch.setattr(qa_evaluation, "answer_question", observe_generation)
    evaluate(
        corpus,
        on_case_start=lambda case: events.append(f"start:{case['id']}"),
        on_case=lambda record: events.append(f"done:{record['id']}"),
    )
    assert events == [
        event
        for case_id in ("one", "conflict", "missing")
        for event in (f"start:{case_id}", "generate", f"done:{case_id}")
    ]


@pytest.fixture
def expandable(corpus):
    text = (
        "# Configuration\nThe alpha limit is four.\n\n"
        "## Procedure\nThe setup anchor is described here.\n"
    )
    sha = digest(text.encode())
    document = replace(
        corpus["documents"][0],
        text=text,
        raw_sha256=sha,
        canonical_sha256=sha,
        format="markdown",
    )
    update_index(corpus["paths"][0], [document])
    manifest_path = corpus["paths"][2]
    manifest = json.loads(manifest_path.read_bytes())
    manifest["sources"][0].update(
        sha256=sha, canonical_sha256=sha, canonical_chars=len(text)
    )
    write_json(manifest_path, manifest)
    dataset = corpus["dataset"]
    dataset["manifest_sha256"] = digest(manifest_path.read_bytes())
    start = text.index("The alpha")
    end = text.index("\n", start) + 1
    dataset["evidence"] = {
        "a": {
            "source_id": "a",
            "start": start,
            "end": end,
            "lines": [2, 2],
            "sha256": digest(text[start:end].encode()),
        }
    }
    dataset["cases"] = [{**dataset["cases"][0], "question": "setup anchor"}]
    write_json(corpus["paths"][1], dataset)
    corpus["protocol"]["manifest"]["sha256"] = dataset["manifest_sha256"]
    corpus["protocol"]["datasets"][0]["sha256"] = digest(
        corpus["paths"][1].read_bytes()
    )
    return corpus


def test_parent_introduction_adds_canonical_evidence_without_losing_anchor(expandable):
    runtime = FakeRuntime()
    report = evaluate(expandable, runtime)
    case = report["cases"][0]
    assert case["failure"] is None
    assert case["evidence"] == [
        {
            "id": "a",
            "retrieved": False,
            "included_before_expansion": False,
            "included": True,
        }
    ]
    anchor = case["packed"]["anchors"][0]
    source = case["packed"]["sources"][0]
    assert anchor["headings"] == ["Configuration", "Procedure"]
    assert source["headings"] == ["Configuration"]
    assert len(source["context_chunk_ids"]) == 2
    assert source["anchor_chunk_id"] == anchor["chunk_id"]
    assert source["start"] < anchor["start"] < anchor["end"] == source["end"]
    assert anchor["text"] in source["text"]
    assert "chunk_id" not in source
    assert report["summary"]["complete_evidence_in_prompt"] == 1
    token_ids, _, schema, messages = runtime.calls[0]
    assert messages == json.loads(case["packed"]["prompt"])
    assert (messages, schema) in runtime.render_calls
    assert schema == expandable["protocol"]["model_output_schema"]
    assert token_ids == runtime.tokenize(case["packed"]["prompt"])
    supplied = json.loads(messages[-1]["content"])["sources"]
    assert supplied[0]["text"] == source["text"]
    assert "The alpha limit is four." in supplied[0]["text"]


def test_expansion_can_be_omitted_while_original_anchor_stays_valid(expandable):
    case = evaluate(expandable)["cases"][0]
    runtime = FakeRuntime()
    prompt = runtime.render(
        answers._messages(
            case["question"],
            case["packed"]["anchors"],
            expandable["protocol"]["model_output_schema"],
        ),
        json_schema=expandable["protocol"]["model_output_schema"],
    )
    settings = expandable["protocol"]["settings"]
    settings["context_size"] = (
        len(runtime.tokenize(prompt)) + settings["max_output_tokens"] + 1
    )
    record = evaluate(expandable, runtime)["cases"][0]
    assert record["failure"] is None
    assert record["packed"]["anchors"] == record["packed"]["sources"]
    assert record["packed"]["context_omitted"][0]["reason"] == "token_budget"
    assert record["evidence"] == [
        {
            "id": "a",
            "retrieved": False,
            "included_before_expansion": False,
            "included": False,
        }
    ]


@pytest.mark.parametrize(
    "changes",
    [
        {"passage_id": "0" * 64},
        {"anchor_chunk_id": "different"},
        {"context_chunk_ids": []},
        {"text": "Invented context."},
        {"line_start": 8},
        {"label": "S2"},
        {"headings": ["Different"]},
    ],
)
def test_expanded_passage_must_match_recomputed_canonical_context(
    expandable, monkeypatch, changes
):
    original = qa_evaluation.answer_question

    def corrupt_context(*args, **kwargs):
        record = original(*args, **kwargs)
        record["packed"]["sources"][0].update(changes)
        return record

    monkeypatch.setattr(qa_evaluation, "answer_question", corrupt_context)
    case = evaluate(expandable)["cases"][0]
    assert case["answer"] is None
    assert "canonical context" in case["failure"]
    assert case["raw_completion"]["content"]
    assert not case["evidence"][0]["included"]


def test_expansion_validation_rejects_an_index_change(expandable, monkeypatch):
    original = qa_evaluation.expand_sources
    retained = []

    def mutate_before_expansion(index_path, sources, snapshot):
        update_index(index_path, [replace(expandable["documents"][1], revision="2")])
        return original(index_path, sources, snapshot)

    monkeypatch.setattr(qa_evaluation, "expand_sources", mutate_before_expansion)
    with pytest.raises(ValueError, match="index changed during answer evaluation"):
        evaluate(expandable, on_case=retained.append)
    assert len(retained) == 1
    assert retained[0]["answer"] is None
    assert "index changed" in retained[0]["failure"]
    assert not retained[0]["evidence"][0]["included"]


def test_index_change_retains_invalid_generated_case_before_aborting(corpus):
    runtime = FakeRuntime()
    retained = []

    def retain_and_change_index(record):
        retained.append(copy.deepcopy(record))
        if record["id"] == "one":
            update_index(
                corpus["paths"][0],
                [replace(corpus["documents"][0], revision="2")],
            )

    with pytest.raises(ValueError, match="index changed during answer evaluation"):
        evaluate(corpus, runtime, on_case=retain_and_change_index)
    assert len(runtime.calls) == 2
    assert [record["id"] for record in retained] == ["one", "conflict"]
    invalid = retained[1]
    assert invalid["raw_completion"]["content"]
    assert invalid["answer"] is None
    assert "index changed" in invalid["source_validation_failure"]
    assert invalid["failure"].startswith("Source validation failed:")
    assert all(
        not unit[key]
        for unit in invalid["evidence"]
        for key in ("retrieved", "included_before_expansion", "included")
    )
