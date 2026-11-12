import hashlib
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/compare_retrieval.py"
    spec = spec_from_file_location("compare_retrieval", script)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    dataset = tmp_path / "questions.json"
    manifest = tmp_path / "manifest.json"
    dataset.write_text('{"id": "synthetic-questions"}\n')
    manifest.write_text('{"id": "synthetic-corpus"}\n')
    policy = {
        "schema_version": 1,
        "manifest": manifest.name,
        "manifest_sha256": digest(manifest.read_bytes()),
        "datasets": [{"path": dataset.name, "sha256": digest(dataset.read_bytes())}],
        "methods": ["lexical", "dense", "hybrid"],
        "passage_limit": 5,
        "candidate_depth": 20,
        "token_budget": 1024,
        "rrf_constant": 60,
    }
    config_path = tmp_path / "configs/retrieval-comparison.json"
    config_path.write_text(json.dumps(policy))
    output = tmp_path / "results/comparison.json"
    output.parent.mkdir()
    output.write_text('{"previous_success": true}')
    monkeypatch.setattr("sys.argv", ["compare_retrieval.py", "--output", str(output)])
    snapshot = {
        "source_fingerprint": "source-v1",
        "source_generation": 1,
        "fingerprint": "embedding-v1",
    }
    monkeypatch.setattr(module, "BGEEncoder", lambda *args: SimpleNamespace(config={}))
    monkeypatch.setattr(module, "embedding_info", lambda *args: dict(snapshot))
    calls = []

    def evaluate(db, dataset_path, manifest_path, *, method, **kwargs):
        calls.append(method)
        covered = {
            "lexical": [True, False, None],
            "dense": [False, True, None],
            "hybrid": [True, True, None],
        }[method]
        result = {
            "dataset_id": "synthetic-questions",
            "dataset_sha256": policy["datasets"][0]["sha256"],
            "manifest_sha256": policy["manifest_sha256"],
            "index": {"fingerprint": "source-v1", "generation": 1},
            "method": method,
            "summary": {"cases_failed": 0},
            "cases": [
                {"id": key, "all_covered": value}
                for key, value in zip(("a", "b", "unsupported"), covered, strict=True)
            ],
        }
        if method != "lexical":
            result["embedding"] = dict(snapshot)
        return result

    monkeypatch.setattr(module, "evaluate_retrieval", evaluate)
    return SimpleNamespace(
        module=module,
        output=output,
        dataset=dataset,
        config_path=config_path,
        config_raw=config_path.read_bytes(),
        snapshot=snapshot,
        evaluate=evaluate,
        calls=calls,
    )


def test_failed_preflight_removes_old_success_before_loading_model(
    comparison, monkeypatch, capsys
):
    comparison.dataset.write_text("Changed questions")

    def unexpected_model(*args):
        pytest.fail("Preflight failure must not load the model.")

    monkeypatch.setattr(comparison.module, "BGEEncoder", unexpected_model)
    with pytest.raises(SystemExit) as error:
        comparison.module.main()
    assert error.value.code == 2
    assert "frozen pins" in capsys.readouterr().err
    assert not comparison.output.exists()
    assert comparison.calls == []


@pytest.mark.parametrize(
    "change",
    ["source_generation", "source_fingerprint", "embedding", "dataset", "manifest"],
)
def test_changed_inputs_between_methods_reject_report(
    comparison, monkeypatch, capsys, change
):
    def changed(*args, **kwargs):
        result = comparison.evaluate(*args, **kwargs)
        if kwargs["method"] == "dense":
            if change == "source_generation":
                result["index"]["generation"] += 1
            elif change == "source_fingerprint":
                result["index"]["fingerprint"] = "changed-source"
            elif change == "embedding":
                result["embedding"]["fingerprint"] = "changed-vectors"
            elif change == "dataset":
                result["dataset_sha256"] = "changed-questions"
            else:
                result["manifest_sha256"] = "changed-manifest"
        return result

    monkeypatch.setattr(comparison.module, "evaluate_retrieval", changed)
    with pytest.raises(SystemExit) as error:
        comparison.module.main()
    assert error.value.code == 2
    assert "changed between runs" in capsys.readouterr().err
    assert comparison.calls == ["lexical", "dense"]
    assert not comparison.output.exists()


def test_changed_embedding_after_final_run_rejects_report(
    comparison, monkeypatch, capsys
):
    reads = 0

    def embedding_info(*args):
        nonlocal reads
        reads += 1
        result = dict(comparison.snapshot)
        if reads > 1:
            result["fingerprint"] = "new-vectors"
        return result

    monkeypatch.setattr(comparison.module, "embedding_info", embedding_info)
    with pytest.raises(SystemExit) as error:
        comparison.module.main()
    assert error.value.code == 2
    assert "Embeddings changed" in capsys.readouterr().err
    assert comparison.calls == ["lexical", "dense", "hybrid"]
    assert not comparison.output.exists()


def test_report_is_published_atomically_with_the_parsed_protocol(
    comparison, monkeypatch, capsys
):
    def model(*args):
        comparison.config_path.write_text('{"changed_after_parsing": true}')
        return SimpleNamespace(config={})

    monkeypatch.setattr(comparison.module, "BGEEncoder", model)
    original_replace = Path.replace
    publications = []

    def publish(source, destination):
        assert destination == comparison.output
        assert source.parent == destination.parent
        assert not destination.exists()
        report = json.loads(source.read_bytes())
        assert len(report["runs"]) == 3
        assert report["protocol_sha256"] == digest(comparison.config_raw)
        assert report["protocol"] == json.loads(comparison.config_raw)
        publications.append(report)
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", publish)
    comparison.module.main()
    capsys.readouterr()
    assert len(publications) == 1
    report = json.loads(comparison.output.read_bytes())
    assert report["comparisons"] == [
        {
            "dataset": "synthetic-questions",
            "method": "dense",
            "fully_covered_wins": ["b"],
            "fully_covered_losses": ["a"],
        },
        {
            "dataset": "synthetic-questions",
            "method": "hybrid",
            "fully_covered_wins": ["b"],
            "fully_covered_losses": [],
        },
    ]
    assert list(comparison.output.parent.iterdir()) == [comparison.output]


def test_failed_publication_leaves_no_partial_report(comparison, monkeypatch):
    def fail_replace(*args):
        raise OSError("Cannot publish report")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="Cannot publish report"):
        comparison.module.main()
    assert not comparison.output.exists()
    assert list(comparison.output.parent.iterdir()) == []


def test_case_failure_is_reported_and_returns_nonzero(comparison, monkeypatch, capsys):
    def failed_case(*args, **kwargs):
        result = comparison.evaluate(*args, **kwargs)
        if kwargs["method"] == "dense":
            result["summary"]["cases_failed"] = 1
            result["cases"][0]["failure"] = "Synthetic query failure"
        return result

    monkeypatch.setattr(comparison.module, "evaluate_retrieval", failed_case)
    with pytest.raises(SystemExit) as error:
        comparison.module.main()
    assert error.value.code == 1
    report = json.loads(comparison.output.read_bytes())
    assert report["runs"][1]["summary"]["cases_failed"] == 1
    assert report["runs"][1]["cases"][0]["failure"] == "Synthetic query failure"
    assert json.loads(capsys.readouterr().out)["summaries"][1]["cases_failed"] == 1
