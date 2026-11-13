import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "eval/answers-protocol.json").read_bytes())


def test_answer_case_pins_retain_the_original_case_requirements():
    manifest = PROTOCOL["manifest"]
    manifest_raw = (ROOT / manifest["path"]).read_bytes()
    assert hashlib.sha256(manifest_raw).hexdigest() == manifest["sha256"]
    seen_case_ids = set()
    for entry in PROTOCOL["datasets"]:
        raw = (ROOT / entry["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        dataset = json.loads(raw)
        case_ids = {case["id"] for case in dataset["cases"]}
        assert len(case_ids) == len(dataset["cases"])
        assert seen_case_ids.isdisjoint(case_ids)
        seen_case_ids.update(case_ids)
        assert dataset["manifest_sha256"] == manifest["sha256"]
        assert all(case["required_facts"] for case in dataset["cases"])
        assert all(case["unacceptable_claims"] for case in dataset["cases"])
        assert not dataset["provenance"]["human_validated"]
    assert [entry["role"] for entry in PROTOCOL["datasets"]] == [
        "development",
        "development_regression",
        "validation",
    ]


def test_answer_fixtures_cover_actual_conflict_absence_and_injection_sources():
    dataset = json.loads((ROOT / PROTOCOL["datasets"][0]["path"]).read_bytes())
    cases = {case["id"]: case for case in dataset["cases"]}
    fixtures = {item["id"]: item for item in PROTOCOL["contract_fixtures"]}
    conflict = cases[fixtures["conflicting_sources"]["case_id"]]
    assert conflict["answerability"] == "conflicting"
    sources = {
        dataset["evidence"][key]["source_id"] for key in conflict["evidence_ids"]
    }
    assert len(sources) >= 2
    missing = cases[fixtures["missing_fact"]["case_id"]]
    assert missing["answerability"] == "unanswerable"
    assert missing["evidence_ids"] == []
    assert missing["absence_rationale"]
    injection = cases[fixtures["document_instruction"]["case_id"]]
    assert "instruction_in_document" in injection["tags"]
    manifest = json.loads((ROOT / PROTOCOL["manifest"]["path"]).read_bytes())
    paths = {
        source["id"]: ROOT / source["local_path"] for source in manifest["sources"]
    }
    span = dataset["evidence"][injection["evidence_ids"][0]]
    text = paths[span["source_id"]].read_text()
    assert "BANANA" in text[span["start"] : span["end"]]
    assert "128" in text[span["start"] : span["end"]]


def test_answer_schema_keeps_claims_and_citations_separate_from_source_metadata():
    schema = PROTOCOL["model_output_schema"]
    claim = schema["properties"]["claims"]["items"]
    citation = claim["properties"]["citations"]["items"]
    assert schema["additionalProperties"] is False
    assert claim["additionalProperties"] is False
    assert citation["additionalProperties"] is False
    assert set(schema["required"]) == {"status", "claims"}
    assert set(claim["required"]) == {"text", "citations"}
    assert set(citation["required"]) == {"label"}
    assert set(citation["properties"]) == {"label"}
    assert claim["properties"]["citations"]["minItems"] == 1
