import copy
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("answer_baseline")
    root, directory = tmp_path / "project", tmp_path / "run"
    public = root / "docs/results/grounded-answers"
    public.mkdir(parents=True)
    directory.mkdir()
    protocol = json.dumps({"model_output_schema": {"type": "object"}}).encode()
    runtime = {
        "binary_sha256": "cpu",
        "sibling_libraries_sha256": {"cpu": "library"},
        "startup_seconds": 3,
        "model_sha256": "model",
        "template_sha256": "template",
        "temperature": 0.6,
    }
    run = {
        "state": "completed",
        "planned": ["one"],
        "completed": ["one"],
        "datasets": {"development": {"sha256": "dataset"}},
        "system_prompt_sha256": "prompt",
        "model_sha256": "model",
        "implementation_sha256": {
            "src/arm64_rag/answers.py": "answers",
            "scripts/check_answers.py": "old-runner",
        },
        "runtime": runtime,
    }
    case = {
        "id": "one",
        "question": "Which setting?",
        "failure": None,
        "answer": {"status": "answered", "claims": ["known answer"]},
        "retrieval": {"generation": 1, "results": ["canonical source"]},
        "packed": {
            "token_ids": [2, 3],
            "max_output_tokens": 12,
            "prompt": "frozen rendered prompt",
            "sources": ["canonical source"],
        },
        "evidence": [{"included": True}],
    }
    run_raw, case_raw = json.dumps(run).encode(), json.dumps(case).encode()
    (directory / "run.json").write_bytes(run_raw)
    (directory / "one.json").write_bytes(case_raw)
    report = {
        "role": "development",
        "run_file_sha256": module.digest(run_raw),
        "dataset_sha256": "dataset",
        "protocol_file_sha256": module.digest(protocol),
        "system_prompt_sha256": "prompt",
        "runtime": runtime,
        "cases": [{"id": "one", "case_file_sha256": module.digest(case_raw)}],
    }
    (public / "development.json").write_text(json.dumps(report))
    current = copy.deepcopy(run)
    current["state"] = "starting"
    current["completed"] = []
    current["implementation_sha256"]["scripts/check_answers.py"] = "new-runner"

    def load():
        return module.BaselineComparison(
            root, directory, "development", current, protocol
        )

    class Server:
        def __init__(self):
            self.calls = []
            self.prompt = case["packed"]["prompt"]
            self.runtime = {
                **runtime,
                "binary_sha256": "kai",
                "sibling_libraries_sha256": {"kai": "other-library"},
                "startup_seconds": 8,
            }

        def info(self):
            return self.runtime

        def render(self, messages, *, json_schema):
            self.calls.append("render")
            return self.prompt

        def tokenize(self, prompt):
            return [2, 3]

        def complete(self, *args, **kwargs):
            self.calls.append("complete")
            return {"content": "new answer", "tokens_predicted": 4}

    return load, current, case, Server(), directory


def test_backend_change_keeps_runtime_settings_and_exact_generation_inputs(baseline):
    load, _, case, server, _ = baseline
    comparison = load().bind(server)
    comparison.start(case)
    result = comparison.complete(
        [2, 3], 12, json_schema={"type": "object"}, messages=[]
    )
    assert result["content"] == "new answer"
    assert server.calls == ["render", "complete"]
    assert comparison.tokenize("prompt") == [2, 3]
    record = copy.deepcopy(case)
    record["answer"]["claims"] = ["different answer"]
    comparison.retain(record)
    assert record["failure"] is None
    assert record["baseline_comparison"]["inputs_match"]
    assert not record["baseline_comparison"]["answer_identical"]


@pytest.mark.parametrize("mutation", ["tokens", "budget", "schema", "prompt"])
def test_changed_inputs_are_rejected_before_generation(baseline, mutation):
    load, _, case, server, _ = baseline
    comparison = load().bind(server)
    comparison.start(case)
    tokens, budget, schema = [2, 3], 12, {"type": "object"}
    if mutation == "tokens":
        tokens = [2, 4]
    elif mutation == "budget":
        budget = 13
    elif mutation == "schema":
        schema = {"type": "string"}
    else:
        server.prompt = "changed"
    with pytest.raises(ValueError, match="differ"):
        comparison.complete(tokens, budget, json_schema=schema, messages=[])
    assert "complete" not in server.calls


@pytest.mark.parametrize("field", ["model_sha256", "template_sha256", "temperature"])
def test_changed_runtime_settings_are_rejected(baseline, field):
    load, _, _, server, _ = baseline
    server.runtime[field] = "different"
    with pytest.raises(ValueError, match="settings differ"):
        load().bind(server)


@pytest.mark.parametrize("file", ["run.json", "one.json"])
def test_published_hashes_authenticate_raw_baseline_records(baseline, file):
    load, _, _, _, directory = baseline
    with (directory / file).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="differs|changed"):
        load()


def test_application_change_requires_a_new_quality_baseline(baseline):
    load, current, _, _, _ = baseline
    current["implementation_sha256"]["src/arm64_rag/answers.py"] = "changed"
    with pytest.raises(ValueError, match="Application code changed"):
        load()


@pytest.mark.parametrize("field", ["retrieval", "packed", "evidence"])
def test_input_mismatch_keeps_failure_and_raw_output(baseline, field):
    load, _, case, _, _ = baseline
    record = copy.deepcopy(case)
    record[field] = {"changed": True}
    record["failure"] = "Earlier failure"
    record["raw_completion"] = {"content": "retained output"}
    load().retain(record)
    assert record["answer"] is None
    assert "Earlier failure" in record["failure"]
    assert record["raw_completion"] == {"content": "retained output"}
    assert record["baseline_comparison"]["mismatched_fields"] == [field]


@pytest.fixture
def local_baseline(baseline):
    _, current, case, server, directory = baseline
    module = importlib.import_module("answer_baseline")
    protocol = json.dumps({"model_output_schema": {"type": "object"}}).encode()
    prompt = b"synthetic prompt"
    run = json.loads((directory / "run.json").read_bytes())
    run.update(
        backend="cpu",
        attempted=run["planned"],
        cases_failed=0,
        active_case=None,
        failure=None,
        protocol_sha256=module.digest(protocol),
        system_prompt_sha256=module.digest(prompt),
    )
    current["system_prompt_sha256"] = run["system_prompt_sha256"]
    (directory / "run.json").write_text(json.dumps(run))
    (directory / "answers-protocol.json").write_bytes(protocol)
    (directory / "system-prompt.txt").write_bytes(prompt)
    path = directory.parent / "pins/local.json"
    digest = module.freeze_local(directory, "development", path)

    def load():
        return module.BaselineComparison.from_local(
            path, digest, "development", current, protocol
        )

    return module, load, directory, path, current, case, server


def test_fresh_cpu_manifest_enforces_inputs_without_quality_approval(local_baseline):
    _, load, _, _, _, case, server = local_baseline
    comparison = load().bind(server)
    assert comparison.metadata["semantic_review"] == "not_assessed"
    assert "report_sha256" not in comparison.metadata
    assert comparison.metadata["local_manifest_sha256"]
    comparison.start(case)
    comparison.complete([2, 3], 12, json_schema={"type": "object"}, messages=[])
    record = copy.deepcopy(case)
    record["answer"] = {"status": "answered", "claims": ["changed"]}
    comparison.retain(record)
    assert record["baseline_comparison"]["inputs_match"]
    assert not record["baseline_comparison"]["answer_identical"]


def test_local_mode_does_not_relax_published_mode(local_baseline, baseline):
    _, load, *_ = local_baseline
    assert load().metadata["baseline_kind"] == "local_input_equality"
    published_load, *_ = baseline
    with pytest.raises(ValueError, match="published evidence"):
        published_load()


@pytest.mark.parametrize(
    "file",
    ["run.json", "one.json", "answers-protocol.json", "system-prompt.txt", "manifest"],
)
def test_local_baseline_rejects_changed_pins_or_raw_inputs(local_baseline, file):
    _, load, directory, manifest, *_ = local_baseline
    path = manifest if file == "manifest" else directory / file
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed|differ"):
        load()


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend", "kleidiai"),
        ("state", "interrupted"),
        ("completed", []),
        ("attempted", []),
        ("cases_failed", 1),
        ("failure", "failed"),
        ("planned", ["../escaped"]),
        ("planned", ["one", "one"]),
    ],
)
def test_local_freeze_requires_complete_cpu_cases(local_baseline, field, value):
    module, _, directory, manifest, *_ = local_baseline
    run = json.loads((directory / "run.json").read_bytes())
    run[field] = value
    if field == "planned":
        run["attempted"] = run["completed"] = value
    (directory / "run.json").write_text(json.dumps(run))
    destination = manifest.parent / "rejected.json"
    with pytest.raises(ValueError):
        module.freeze_local(directory, "development", destination)
    assert not destination.exists()


def test_local_freeze_is_create_only_and_still_rejects_code_changes(local_baseline):
    module, load, directory, manifest, current, *_ = local_baseline
    with pytest.raises(FileExistsError):
        module.freeze_local(directory, "development", manifest)
    current["implementation_sha256"]["src/arm64_rag/answers.py"] = "changed"
    with pytest.raises(ValueError, match="Application code changed"):
        load()


def test_local_baseline_stops_changed_tokens_before_generation(local_baseline):
    _, load, _, _, _, case, server = local_baseline
    comparison = load().bind(server)
    comparison.start(case)
    with pytest.raises(ValueError, match="Token IDs"):
        comparison.complete([2, 4], 12, json_schema={"type": "object"}, messages=[])
    assert not server.calls
