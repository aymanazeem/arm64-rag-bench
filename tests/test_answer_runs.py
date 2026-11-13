import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from arm64_rag.inference import InferenceError


@pytest.fixture
def runner(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("check_answers")
    root = tmp_path / "project"
    (root / "eval").mkdir(parents=True)
    (root / "configs").mkdir()
    protocol = {
        "settings": {"context_size": 128},
        "manifest": {"path": "data/manifests/corpus.json"},
    }
    (root / "eval/answers-protocol.json").write_text(json.dumps(protocol))
    cases = [{"id": "dev-001"}, {"id": "dev-002"}]
    (root / "eval/development.json").write_text(json.dumps({"cases": cases}))
    model_hash = "a" * 64
    (root / "configs/answer-model.json").write_text(
        json.dumps(
            {
                "directory": "models/candidate",
                "model": {"filename": "model.gguf", "sha256": model_hash},
            }
        )
    )
    output = tmp_path / "run"
    state = SimpleNamespace(startup_error=None, entries=0, exits=0)

    class FakeServer:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            state.entries += 1
            if state.startup_error:
                raise state.startup_error
            return self

        def __exit__(self, *args):
            state.exits += 1

        def info(self):
            return {"model_sha256": model_hash}

    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "LocalServer", FakeServer)
    monkeypatch.setattr(
        module, "evaluate_answers", lambda *a, **k: pytest.fail("Unexpected evaluation")
    )
    monkeypatch.setattr(sys, "argv", ["check_answers.py", "--output", str(output)])
    return SimpleNamespace(
        module=module, root=root, output=output, cases=cases, state=state
    )


def read_json(path):
    return json.loads(path.read_text())


def test_failed_server_startup_preserves_failure_and_configuration(runner):
    runner.state.startup_error = InferenceError("Model load failed")
    with pytest.raises(InferenceError, match="Model load failed"):
        runner.module.main()
    run = read_json(runner.output / "run.json")
    assert run["state"] == "failed"
    assert run["planned"] == [case["id"] for case in runner.cases]
    assert run["attempted"] == run["completed"] == []
    assert run["active_case"] is None
    assert run["failure"] == "InferenceError: Model load failed"
    assert (runner.output / "answers-protocol.json").read_bytes() == (
        runner.root / "eval/answers-protocol.json"
    ).read_bytes()
    assert (runner.output / "system-prompt.txt").read_text() == (
        runner.module.SYSTEM_PROMPT
    )
    assert not (runner.output / "development.json").exists()


def test_interruption_retains_completed_and_active_attempts(runner, monkeypatch):
    completed = {
        "id": "dev-001",
        "answer": {"status": "insufficient_evidence", "claims": []},
        "failure": None,
    }

    def evaluate(*args, on_case_start, on_case):
        on_case_start(runner.cases[0])
        on_case(completed)
        on_case_start(runner.cases[1])
        raise KeyboardInterrupt

    monkeypatch.setattr(runner.module, "evaluate_answers", evaluate)
    with pytest.raises(KeyboardInterrupt):
        runner.module.main()
    run = read_json(runner.output / "run.json")
    assert run["state"] == "interrupted"
    assert run["attempted"] == ["dev-001", "dev-002"]
    assert run["completed"] == ["dev-001"]
    assert run["active_case"] == "dev-002"
    assert run["failure"].startswith("KeyboardInterrupt:")
    assert read_json(runner.output / "dev-001.json") == completed
    assert not (runner.output / "dev-002.json").exists()
    assert not (runner.output / "development.json").exists()
    assert runner.state.exits == 1


def test_completed_case_failure_is_counted_and_exits_nonzero(runner, monkeypatch):
    records = [
        {
            "id": "dev-001",
            "answer": {"status": "insufficient_evidence", "claims": []},
            "failure": None,
        },
        {
            "id": "dev-002",
            "answer": None,
            "failure": "Invalid model output.",
            "raw_completion": {"content": "Malformed \ud800 response with café text."},
        },
    ]
    report = {
        "summary": {"cases_attempted": 2, "cases_failed": 1},
        "cases": records,
    }

    def evaluate(*args, on_case_start, on_case):
        for case, record in zip(runner.cases, records):
            on_case_start(case)
            on_case(record)
        return report

    monkeypatch.setattr(runner.module, "evaluate_answers", evaluate)
    with pytest.raises(SystemExit) as error:
        runner.module.main()
    assert error.value.code == 1
    run = read_json(runner.output / "run.json")
    assert run["state"] == "completed"
    assert run["attempted"] == run["completed"] == ["dev-001", "dev-002"]
    assert run["active_case"] is None
    assert run["cases_failed"] == 1
    assert read_json(runner.output / "development.json") == report
    assert read_json(runner.output / "dev-002.json") == records[1]
    assert not list(runner.output.glob("*.tmp"))
    assert runner.state.exits == 1


def test_existing_run_is_rejected_without_changing_files(runner):
    runner.output.mkdir()
    previous = b'{"state":"interrupted","attempted":["dev-001"]}\n'
    (runner.output / "run.json").write_bytes(previous)
    with pytest.raises(SystemExit) as error:
        runner.module.main()
    assert error.value.code == 2
    assert (runner.output / "run.json").read_bytes() == previous
    assert [path.name for path in runner.output.iterdir()] == ["run.json"]
    assert runner.state.entries == 0


@pytest.mark.parametrize(
    ("split", "selected"),
    [
        ("development", ["development"]),
        ("regression", ["regression"]),
        ("validation", ["validation"]),
        ("all", ["development", "regression", "validation"]),
    ],
)
def test_split_routes_exact_datasets_and_records_planned_cases(
    runner, monkeypatch, split, selected
):
    datasets = {
        "development": ("development.json", runner.cases),
        "regression": ("retrieval-validation.json", [{"id": "spent-001"}]),
        "validation": ("answer-validation.json", [{"id": "fresh-001"}]),
    }
    for filename, cases in datasets.values():
        (runner.root / "eval" / filename).write_text(json.dumps({"cases": cases}))
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_answers.py", "--split", split, "--output", str(runner.output)],
    )
    calls = []

    def evaluate(db, dataset, manifest, server, protocol, on_case_start, on_case):
        calls.append(dataset.name)
        records = []
        for case in read_json(dataset)["cases"]:
            on_case_start(case)
            record = {
                "id": case["id"],
                "answer": {"status": "insufficient_evidence", "claims": []},
                "failure": None,
            }
            records.append(record)
            on_case(record)
        return {"summary": {"cases_failed": 0}, "cases": records}

    monkeypatch.setattr(runner.module, "evaluate_answers", evaluate)
    runner.module.main()
    run = read_json(runner.output / "run.json")
    assert calls == [datasets[key][0] for key in selected]
    assert (
        run["planned"]
        == run["attempted"]
        == run["completed"]
        == [case["id"] for key in selected for case in datasets[key][1]]
    )
    assert run["datasets"] == {
        key: {
            "path": f"eval/{datasets[key][0]}",
            "sha256": hashlib.sha256(
                (runner.root / "eval" / datasets[key][0]).read_bytes()
            ).hexdigest(),
        }
        for key in selected
    }
    assert run["state"] == "completed"
    assert run["active_case"] is None
    for key in datasets:
        assert (runner.output / f"{key}.json").exists() == (key in selected)
    assert runner.state.entries == runner.state.exits == 1


def test_missing_fresh_validation_never_uses_spent_questions(runner, monkeypatch):
    (runner.root / "eval/retrieval-validation.json").write_text(
        json.dumps({"cases": [{"id": "spent-001"}]})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_answers.py", "--split", "validation", "--output", str(runner.output)],
    )
    with pytest.raises(FileNotFoundError, match="answer-validation.json"):
        runner.module.main()
    assert runner.state.entries == 0
    assert not (runner.output / "validation.json").exists()


def test_all_rejects_duplicate_ids_before_starting_server(runner, monkeypatch):
    for name in ("retrieval-validation.json", "answer-validation.json"):
        (runner.root / "eval" / name).write_text(
            json.dumps({"cases": [{"id": "duplicate-001"}]})
        )
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_answers.py", "--split", "all", "--output", str(runner.output)],
    )
    with pytest.raises(SystemExit) as error:
        runner.module.main()
    assert error.value.code == 2
    assert runner.state.entries == 0
    assert not list(runner.output.glob("*.json"))
