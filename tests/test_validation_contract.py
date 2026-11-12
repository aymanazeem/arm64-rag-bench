import hashlib
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data/manifests/corpus.json"
VALIDATION = json.loads((ROOT / "eval/retrieval-validation.json").read_bytes())
DEVELOPMENT = json.loads((ROOT / "eval/development.json").read_bytes())
MANIFEST = json.loads(MANIFEST_PATH.read_bytes())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def test_validation_is_pinned_and_separate_from_development():
    assert VALIDATION["schema_version"] == 1
    assert VALIDATION["split"] == "validation"
    assert ROOT / VALIDATION["manifest_path"] == MANIFEST_PATH
    assert VALIDATION["manifest_sha256"] == digest(MANIFEST_PATH.read_bytes())
    assert VALIDATION["provenance"]["human_validated"] is False
    assert VALIDATION["provenance"]["review"] == "independent_agent_source_checked"
    cases = VALIDATION["cases"]
    assert cases
    assert len({case["id"] for case in cases}) == len(cases)
    assert len({case["family"] for case in cases}) == len(cases)
    assert {case["family"] for case in cases}.isdisjoint(
        case["family"] for case in DEVELOPMENT["cases"]
    )
    assert {case["question"].strip().casefold() for case in cases}.isdisjoint(
        case["question"].strip().casefold() for case in DEVELOPMENT["cases"]
    )
    corpora = {corpus["id"]: corpus["source_ids"] for corpus in MANIFEST["corpora"]}
    used = set()
    for case in cases:
        assert case["origin"] == "source_derived"
        assert case["answerability"] == "answerable"
        assert case["question"].strip()
        assert case["required_facts"] and case["unacceptable_claims"]
        assert all(fact.strip() for fact in case["required_facts"])
        assert all(claim.strip() for claim in case["unacceptable_claims"])
        assert case["evidence_ids"]
        assert len(set(case["evidence_ids"])) == len(case["evidence_ids"])
        for key in case["evidence_ids"]:
            assert (
                VALIDATION["evidence"][key]["source_id"] in corpora[case["corpus_id"]]
            )
            used.add(key)
    assert used == VALIDATION["evidence"].keys()


def test_validation_spans_do_not_overlap_development_labels():
    sources = {source["id"]: source for source in MANIFEST["sources"]}
    for span in VALIDATION["evidence"].values():
        assert (
            0
            <= span["start"]
            < span["end"]
            <= sources[span["source_id"]]["canonical_chars"]
        )
        assert 1 <= span["lines"][0] <= span["lines"][1]
        assert len(bytes.fromhex(span["sha256"])) == 32
        for other in DEVELOPMENT["evidence"].values():
            if span["source_id"] == other["source_id"]:
                assert max(span["start"], other["start"]) >= min(
                    span["end"], other["end"]
                )


@pytest.fixture(scope="module")
def canonical_sources():
    required = {span["source_id"] for span in VALIDATION["evidence"].values()}
    texts = {}
    for source in MANIFEST["sources"]:
        if source["id"] not in required:
            continue
        raw = (ROOT / source["local_path"]).read_bytes()
        assert len(raw) == source["bytes"]
        assert digest(raw) == source["sha256"]
        text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        assert len(text) == source["canonical_chars"]
        assert digest(text.encode()) == source["canonical_sha256"]
        texts[source["id"]] = text
    return texts


@pytest.mark.skipif(
    os.environ.get("ARM64_VERIFY_UPSTREAM") != "1", reason="Optional upstream check"
)
@pytest.mark.parametrize("case", VALIDATION["cases"], ids=lambda case: case["id"])
def test_validation_evidence_matches_pinned_sources(case, canonical_sources):
    for key in case["evidence_ids"]:
        span = VALIDATION["evidence"][key]
        text = canonical_sources[span["source_id"]]
        assert digest(text[span["start"] : span["end"]].encode()) == span["sha256"]
        assert span["lines"] == [
            text.count("\n", 0, span["start"]) + 1,
            text.count("\n", 0, span["end"] - 1) + 1,
        ]
