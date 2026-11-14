import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from arm64_rag.answers import render_answer, validate_answer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "application_benchmark", ROOT / "scripts/benchmark_answers.py"
)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
sys.path.pop(0)


def save(path, value):
    path.write_text(json.dumps(value) + "\n")


@pytest.fixture(autouse=True)
def no_native_work(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "These tests must not start native processes or host probes."
        )

    monkeypatch.setattr(app.common.host, "collect", forbidden)
    monkeypatch.setattr(app.common.subprocess, "Popen", forbidden)


@pytest.fixture
def case_record():
    source = {
        "label": "S1",
        "source_id": "test-note",
        "revision": "1",
        "text": "The batch size is 128 tokens.",
        "start": 0,
        "end": 29,
        "line_start": 1,
        "line_end": 1,
        "url": "",
        "chunk_id": "chunk",
        "headings": [],
    }
    content = json.dumps(
        {
            "status": "answered",
            "claims": [
                {
                    "text": "The batch size is 128 tokens.",
                    "citations": [{"label": "S1"}],
                }
            ],
        }
    )
    record = {
        "question": "What batch size?",
        "retrieval": {"generation": 1, "fingerprint": "index", "results": [source]},
        "packed": {
            "sources": [source],
            "prompt": "prompt",
            "token_ids": [1, 2],
            "max_output_tokens": 2048,
        },
        "raw_completion": {
            "content": content,
            "tokens_predicted": 5,
            "timings": {"cache_n": 0, "prompt_ms": 3, "predicted_ms": 4},
        },
        "answer": validate_answer(content, [source]),
        "failure": None,
        "runtime": {"model_sha256": "m", "binary_sha256": "b"},
    }
    case = {
        "id": "dev-014",
        "question": record["question"],
        "source_ids": ["test-note"],
        "required_facts": ["The batch size is 128 tokens."],
        "unacceptable_claims": [],
        "published_prompt": {
            "prompt_sha256": app.digest(b"prompt"),
            "prompt_tokens": 2,
            "max_output_tokens": 2048,
        },
    }
    return case, record


def test_schedule_has_three_repeats_and_no_fixed_empty_control():
    cli, fixed = app.schedule("cli"), app.schedule("fixed")
    assert len(cli) == 12 and len(fixed) == 9
    assert len({row["id"] for row in cli + fixed}) == 21
    for stage in (cli, fixed):
        for case in app.SELECTED:
            assert [row["block"] for row in stage if row["case_id"] == case] == [
                1,
                2,
                3,
            ]
    assert sum(row["case_id"] == app.EMPTY_ID for row in cli) == 3
    assert all(row["case_id"] != app.EMPTY_ID for row in fixed)
    assert app.schedule("cli") == cli


def test_portable_input_identity_ignores_only_local_generation(case_record):
    _, record = case_record
    other = copy.deepcopy(record)
    other["retrieval"]["generation"] = 7
    assert app.input_identity(other) == app.input_identity(record)
    other["packed"]["token_ids"].reverse()
    assert app.input_identity(other) != app.input_identity(record)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["packed"].update(prompt="changed"),
        lambda r: r["packed"].update(token_ids=[1]),
        lambda r: r["raw_completion"]["timings"].update(cache_n=1),
        lambda r: r["raw_completion"]["timings"].pop("cache_n"),
        lambda r: r["answer"]["claims"][0].update(text="A changed answer."),
        lambda r: r.update(failure="interrupted"),
    ],
)
def test_changed_inputs_cache_and_answer_are_rejected(case_record, mutation):
    case, record = case_record
    mutation(record)
    with pytest.raises(ValueError):
        app.check_record(record, case, {"settings": {"max_output_tokens": 2048}})


def test_cached_slot_size_is_not_reuse(case_record):
    _, record = case_record
    record["raw_completion"]["tokens_cached"] = 2048
    assert app.native_metrics(record)["cache_reused_tokens"] == 0
    record["raw_completion"]["timings"].pop("cache_n")
    assert app.native_metrics(record)["cache_reused_tokens"] is None


def test_empty_control_requires_no_generation(case_record):
    case, record = case_record
    case.update(id=app.EMPTY_ID, published_prompt=None)
    record["retrieval"]["results"] = []
    record["packed"]["sources"] = []
    record.update(
        answer={"status": "insufficient_evidence", "claims": []}, raw_completion=None
    )
    app.check_record(record, case, {"settings": {"max_output_tokens": 2048}})
    record["raw_completion"] = {"content": ""}
    with pytest.raises(ValueError, match="empty control"):
        app.check_record(record, case, {"settings": {"max_output_tokens": 2048}})


def observation(stage="cli"):
    result = {
        "events": {
            "server_initialization_start_ns": 20,
            "native_spawn_start_ns": 30,
            "native_spawn_return_ns": 40,
            "server_ready_ns": 50,
            "answer_start_ns": 60,
            "complete_start_ns": 70,
            "complete_end_ns": 80,
            "answer_end_ns": 90,
            "application_return_ns": 100,
            "stdout_flushed_ns": 110,
            "observer_export_start_ns": 120,
        },
        "exit_code": 0,
        "failure": None,
        "native_launches": 1,
        "native_reaped": True,
        "worker_peak_rss_bytes": 4000,
        "native_peak_rss_bytes": 5000,
    }
    if stage == "fixed":
        result["events"].pop("answer_start_ns")
        result["events"].pop("answer_end_ns")
        result["events"].update(validation_start_ns=82, validation_end_ns=88)
    return result


def metric_row():
    return {
        "id": "cli-1-1",
        "case_id": "dev-014",
        "stage": "cli",
        "started_ns": 10,
        "first_stdout_ns": 111,
        "exit_observed_ns": 150,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda o: o.update(native_launches=2),
        lambda o: o.update(native_reaped=False),
        lambda o: o.update(native_peak_rss_bytes=None),
        lambda o: o["events"].update(stdout_flushed_ns=500),
        lambda o: o["events"].update(server_ready_ns=5),
    ],
)
def test_observation_rejects_ambiguous_memory_and_impossible_clocks(
    case_record, mutation
):
    _, record = case_record
    observed = observation()
    mutation(observed)
    with pytest.raises(ValueError):
        app.check_observation(
            observed,
            metric_row(),
            record,
            {"runtime": {"model": {"sha256": "m"}, "binary": {"sha256": "b"}}},
            {"export_end_ns": 140},
        )


def test_buffered_boundary_excludes_sidecar_and_does_not_invent_ttft(
    tmp_path, case_record
):
    _, record = case_record
    save(tmp_path / "answer.json", record)
    save(tmp_path / "observation.json", observation())
    save(tmp_path / "observer-end.json", {"export_end_ns": 140})
    row = {**metric_row(), "files": app.record_files(tmp_path)}
    metrics, _, _, _ = app.metrics(row)
    assert metrics["application_return_seconds"] == 90 / 1e9
    assert metrics["pipe_observed_answer_seconds"] == 101 / 1e9
    assert metrics["worker_exit_observed_seconds"] == 140 / 1e9
    assert metrics["observer_export_seconds"] == 20 / 1e9
    assert metrics["model_time_to_first_token_seconds"] is None
    assert metrics["native_peak_rss_bytes"] == 5000


def test_fixed_validation_metric_stays_within_its_request_and_return(
    tmp_path, case_record
):
    _, record = case_record
    save(tmp_path / "answer.json", record)
    save(tmp_path / "observation.json", observation("fixed"))
    save(tmp_path / "observer-end.json", {"export_end_ns": 140})
    row = {**metric_row(), "stage": "fixed", "files": app.record_files(tmp_path)}
    metrics, _, _, _ = app.metrics(row)
    assert metrics["answer_validation_seconds"] == 6 / 1e9
    assert metrics["fixed_workflow_return_seconds"] == 90 / 1e9
    assert metrics["application_return_seconds"] is None
    assert metrics["ready_server_rag_seconds"] is None


def test_private_artifact_names_cannot_enter_public_map(tmp_path):
    (tmp_path / "private-customer-name.txt").write_text("secret")
    with pytest.raises(ValueError, match="unexpected file"):
        app.record_files(tmp_path)


def test_replay_selects_first_planned_trace_not_fastest_or_best(
    tmp_path, monkeypatch, case_record
):
    _, record = case_record
    rows = []
    for index, scheduled in enumerate(app.schedule("cli")):
        path = tmp_path / f"answer-{index}.json"
        save(path, record)
        rows.append({**scheduled, "files": {"answer.json": app.common.asset(path)}})
    run = {"state": "completed", "timing_valid": True, "rows": rows}
    monkeypatch.setattr(app, "load_run", lambda *args, **kwargs: run)
    selected = app.replay_inputs({}, tmp_path / "run.json")
    for case_id in app.SELECTED:
        first = next(row for row in rows if row["case_id"] == case_id)
        assert selected[case_id] == first["files"]["answer.json"]
    run["state"] = "stopped"
    with pytest.raises(ValueError, match="complete valid CLI"):
        app.replay_inputs({}, tmp_path / "run.json")


class FakeGuard:
    def __init__(self, *, fail=None, elapsed=100):
        self.error, self.elapsed = fail, elapsed
        self.first = True

    def continuous_ns(self):
        return self.elapsed

    def check(self, label=None):
        if not self.first and self.error:
            raise self.error
        value = 10 if self.first else self.elapsed
        self.first = False
        return {
            "continuous_before_ns": value,
            "continuous_after_ns": value,
            "awake_ns": value,
        }

    def fail(self, reason, sample=None):
        raise app.common.timing.GuardError(reason, sample)


def test_guard_interrupt_prevents_release_and_cleans_worker(
    tmp_path, monkeypatch, case_record
):
    case, _ = case_record
    calls = []

    class Owner:
        first_stdout_ns = None

        def __init__(self, *args, **kwargs):
            kwargs["stdout_path"].write_bytes(b"")
            kwargs["stderr_path"].write_bytes(b"")

        def release(self):
            calls.append("release")

        def poll_stdout(self):
            return []

        def finish(self, *, terminate):
            calls.append(("finish", terminate))
            return {"returncode": -15, "cleanup_verified": True}

    plan = {"workloads": {case["id"]: case}, "timing_policy": {"poll_ns": 100}}
    row = {"id": "cli-1-1", "case_id": case["id"], "stage": "cli", "state": "planned"}
    assert not app.execute_row(
        plan,
        row,
        tmp_path,
        FakeGuard(fail=RuntimeError("sleep")),
        lambda: None,
        owner_factory=Owner,
    )
    assert calls == [("finish", True)]
    assert row["state"] == "finished" and row["outcome"] == "failed"
    assert row["ownership"]["cleanup_verified"]
    assert "stdout.txt" in row["files"]


def test_late_exit_is_timeout_even_with_completed_stdout(tmp_path, case_record):
    case, _ = case_record

    class LateGuard(FakeGuard):
        def check(self, label=None):
            if label and label.endswith(":release"):
                return {
                    "continuous_before_ns": 20,
                    "continuous_after_ns": 20,
                    "awake_ns": 20,
                }
            return super().check(label)

    class Owner:
        first_stdout_ns = 100

        def __init__(self, *args, **kwargs):
            kwargs["stdout_path"].write_bytes(b"answer\n")
            kwargs["stderr_path"].write_bytes(b"")

        def release(self):
            pass

        def poll_stdout(self):
            return []

        def exited(self):
            return True

        def finish(self, *, terminate):
            return {"returncode": 0, "cleanup_verified": True}

    plan = {"workloads": {case["id"]: case}, "timing_policy": {"poll_ns": 100}}
    row = {"id": "cli-1-1", "case_id": case["id"], "stage": "cli"}
    app.execute_row(
        plan,
        row,
        tmp_path,
        LateGuard(elapsed=601 * 10**9),
        lambda: None,
        owner_factory=Owner,
    )
    assert row["outcome"] == "timeout"
    assert row["ownership"]["returncode"] == 0
    assert (tmp_path / "stdout.txt").read_text() == "answer\n"


def test_worker_calls_original_cli_and_only_changes_native_group(
    tmp_path, monkeypatch, capsys, case_record
):
    from arm64_rag import answers, cli, inference

    case, record = case_record
    kwargs_seen, argv_seen = [], []
    stamps = iter(range(1, 1000))
    monkeypatch.setattr(app, "continuous_ns", lambda: next(stamps))
    monkeypatch.setattr(
        app.resource, "getrusage", lambda who: SimpleNamespace(ru_maxrss=4096)
    )

    class Process:
        returncode = None

    process = Process()

    def launch(argv, **kwargs):
        argv_seen.append(argv)
        kwargs_seen.append(kwargs)
        return process

    monkeypatch.setattr(inference.subprocess, "Popen", launch)

    class Server:
        def __enter__(self):
            inference.subprocess.Popen(
                ["server", "--ctx-size", "8192"],
                env={"LLAMA_API_KEY": "test"},
                stdin=0,
                stdout=1,
                stderr=2,
                start_new_session=True,
            )
            return self

        def __exit__(self, *args):
            process.returncode = 0

        def complete(self, *args, **kwargs):
            return record["raw_completion"]

    monkeypatch.setattr(inference, "LocalServer", Server)
    monkeypatch.setattr(answers, "answer_question", lambda *args, **kwargs: record)

    def cli_main():
        with inference.LocalServer() as server:
            answer = answers.answer_question(None, None, server, None)
            server.complete([1, 2], 2048)
        save(tmp_path / "answer.json", answer)
        print(render_answer(answer))

    monkeypatch.setattr(cli, "main", cli_main)
    plan = {
        "workloads": {case["id"]: case},
        "index": {"path": "index.sqlite"},
        "runtime": {"binary": {"path": "server"}},
    }
    assert (
        app.worker(
            {
                "plan": plan,
                "row": {"case_id": case["id"]},
                "stage": "cli",
                "directory": str(tmp_path),
            }
        )
        == 0
    )
    assert capsys.readouterr().out == render_answer(record) + "\n"
    assert argv_seen == [["server", "--ctx-size", "8192"]]
    assert kwargs_seen == [
        {
            "env": {"LLAMA_API_KEY": "test"},
            "stdin": 0,
            "stdout": 1,
            "stderr": 2,
            "start_new_session": False,
        }
    ]
    observed = app.common.read_json(tmp_path / "observation.json")
    assert observed["native_reaped"] and observed["native_launches"] == 1
    assert (
        observed["events"]["application_return_ns"]
        < observed["events"]["observer_export_start_ns"]
    )


def test_archive_failure_keeps_original_capture_failure(
    tmp_path, monkeypatch, case_record
):
    case, _ = case_record

    class Owner:
        first_stdout_ns = None

        def __init__(self, *args, **kwargs):
            kwargs["stdout_path"].write_bytes(b"partial")
            kwargs["stderr_path"].write_bytes(b"")

        def release(self):
            raise RuntimeError("original capture failure")

        def finish(self, *, terminate):
            return {"cleanup_verified": True, "returncode": -15}

    def broken_archive(directory):
        raise OSError("secondary storage failure")

    monkeypatch.setattr(app, "record_files", broken_archive)
    plan = {"workloads": {case["id"]: case}}
    row = {"id": "cli-1-1", "case_id": case["id"], "stage": "cli"}
    saved = []
    app.execute_row(
        plan,
        row,
        tmp_path,
        FakeGuard(),
        lambda: saved.append(copy.deepcopy(row)),
        owner_factory=Owner,
    )
    assert row["failure"] == app.safe_failure(RuntimeError("original capture failure"))
    assert row["evidence_failures"] == [
        app.safe_failure(OSError("secondary storage failure"))
    ]
    assert saved[-1]["failure"] == row["failure"]
    assert (tmp_path / "stdout.txt").read_bytes() == b"partial"


def test_stage_stops_after_failure_and_verifies_before_guard_close(
    tmp_path, monkeypatch
):
    model = tmp_path / "model"
    model.write_bytes(b"model")
    plan = {
        "runtime": {"model": {"path": str(model)}},
        "sources": {},
        "timing_policy": {"poll_ns": 100},
        "host_policy": {"identity": {}},
    }
    plan_path = tmp_path / "plan.json"
    save(plan_path, plan)
    guards, checks, launches = [], [], []

    class Guard:
        def __init__(self, *args, **kwargs):
            self.closed = False
            self.evidence = {"first_failure": None}
            guards.append(self)

        def start(self):
            pass

        def idle(self, *args):
            pass

        def check(self, label=None):
            assert not self.closed
            return {}

        def close(self):
            self.closed = True
            return True

    def verify(value, *, verify_model=True):
        checks.append(verify_model)
        if not verify_model:
            assert guards and not guards[0].closed
        return value

    def execute(plan, row, directory, guard, persist, **kwargs):
        launches.append(row["id"])
        row.update(state="finished", outcome="failed", failure={"kind": "fixture"})
        persist()
        return False

    monkeypatch.setattr(app, "check_plan", verify)
    monkeypatch.setattr(
        app, "collection_identity", lambda *args: {"fixture": "unchanged"}
    )
    monkeypatch.setattr(app.common.timing, "TimingGuard", Guard)
    monkeypatch.setattr(app.common.host, "collect", lambda: {"fixture": True})
    monkeypatch.setattr(app.common.host, "admit", lambda *args: [])
    monkeypatch.setattr(app.common.host, "interval_violations", lambda *args: [])
    monkeypatch.setattr(app, "execute_row", execute)
    run = app.run_stage(plan_path, "cli", tmp_path / "run", owner_factory=object)
    assert len(launches) == 1
    assert checks == [True, False]
    assert guards[0].closed and run["state"] == "stopped"
    assert sum(row["state"] == "planned" for row in run["rows"]) == 11


def unstarted_run(tmp_path, plan):
    plan.setdefault("sources", {})
    path = tmp_path / "cli"
    path.mkdir()
    save(path / "plan.json", plan)
    claim_path = tmp_path / "claim.json"
    save(
        claim_path,
        {
            "plan_sha256": app.digest((path / "plan.json").read_bytes()),
            "stage": "cli",
            "purpose": "measurement",
            "output": str(path),
        },
    )
    run = {
        "kind": app.KIND,
        "stage": "cli",
        "state": "planned",
        "purpose": "measurement",
        "plan_sha256": app.digest((path / "plan.json").read_bytes()),
        "rows": [{**row, "stage": "cli"} for row in app.schedule("cli")],
        "timing_guard": None,
        "timing_valid": False,
        "integrity_verified": False,
        "failure": None,
        "launch_claim": app.common.asset(claim_path),
        "source_snapshot": {"complete": False, "files": {}},
    }
    save(path / "run.json", run)
    return path / "run.json", run


def test_report_retains_unobserved_stage_and_omits_private_fields(
    tmp_path, monkeypatch
):
    source = tmp_path / "private-person-project.py"
    source.write_text("private source bytes")
    plan = {
        "runtime": {"model": {"sha256": "m"}, "binary": {"sha256": "b"}},
        "sources": {"src/arm64_rag/answers.py": app.common.asset(source)},
        "private": {"pid": 123, "environment": {"SECRET": "keep local"}},
    }
    plan_path = tmp_path / "plan.json"
    save(plan_path, plan)
    path, run = unstarted_run(tmp_path, plan)
    run["private_notes"] = "/Users/private/person"
    save(path, run)
    monkeypatch.setattr(app, "check_plan", lambda value, **kwargs: value)
    report = app.make_report(plan_path, path)
    assert report["planned"] == 21
    assert report["unattempted"] == 12 and report["unobserved"] == 9
    assert report["successes"] == report["failed"] == report["unresolved"] == 0
    assert report["protocol_complete"] is False
    assert report["quality_review"] == "pending"
    assert sum(len(value["rows"]) for value in report["stages"].values()) == 21
    raw = json.dumps(report)
    for secret in ("/Users", "keep local", "private_notes", "environment"):
        assert secret not in raw


def test_unattempted_row_cannot_smuggle_private_host_findings(tmp_path):
    plan = {}
    path, run = unstarted_run(tmp_path, plan)
    run["rows"][0]["host_violations"] = ["/Users/private/person"]
    save(path, run)
    with pytest.raises(ValueError, match="host findings"):
        app.load_run(plan, path, "cli")


def test_initial_timing_failure_forbids_later_attempt(tmp_path, monkeypatch):
    plan = {"timing_policy": {}, "host_policy": {"identity": {}}}
    path, run = unstarted_run(tmp_path, plan)
    run["timing_guard"] = {
        "checkpoints": [],
        "first_failure": {"reason": "assertion_lost", "phase": "session:start"},
    }
    run["state"] = "stopped"
    run["source_snapshot"]["complete"] = True
    run["rows"][0]["state"] = "attempted"
    save(path, run)
    monkeypatch.setattr(app.common.timing, "validate_evidence", lambda *args: False)
    with pytest.raises(ValueError, match="after a timing stop"):
        app.load_run(plan, path, "cli")


def test_raw_stdout_failure_is_not_a_successful_timing_sample(
    tmp_path, monkeypatch, case_record
):
    case, record = case_record
    observed = observation()

    class Owner:
        first_stdout_ns = 111

        def __init__(self, *args, **kwargs):
            kwargs["stdout_path"].write_text("wrong answer\n")
            kwargs["stderr_path"].write_text("")
            save(tmp_path / "answer.json", record)
            save(tmp_path / "observation.json", observed)
            save(tmp_path / "observer-end.json", {"export_end_ns": 140})

        def release(self):
            pass

        def poll_stdout(self):
            return []

        def exited(self):
            return True

        def finish(self, *, terminate):
            return {
                "returncode": 0,
                "cleanup_verified": True,
                "first_failure": None,
                "cleanup_failures": [],
            }

    plan = {
        "workloads": {case["id"]: case},
        "timing_policy": {"poll_ns": 100},
        "runtime": {"model": {"sha256": "m"}, "binary": {"sha256": "b"}},
    }
    row = {"id": "cli-1-1", "case_id": case["id"], "stage": "cli"}
    assert not app.execute_row(
        plan, row, tmp_path, FakeGuard(elapsed=150), lambda: None, owner_factory=Owner
    )
    assert row["outcome"] == "failed"
    assert row["failure"] == app.safe_failure(
        ValueError("Captured output differs from the original answer renderer.")
    )


def test_compatibility_uses_only_median_case_and_is_not_measured_input(tmp_path):
    for stage in app.STAGES:
        rows = app.schedule(stage, compatibility=True)
        assert len(rows) == 1 and rows[0]["case_id"] == "val-015"
    plan = {}
    path, run = unstarted_run(tmp_path, plan)
    run["purpose"] = "excluded_compatibility"
    save(path, run)
    with pytest.raises(ValueError, match="Compatibility observations"):
        app.load_run(plan, path, "cli")


def test_failed_compatibility_cli_leaves_fixed_replay_unattempted(
    tmp_path, monkeypatch
):
    plan = tmp_path / "plan.json"
    save(plan, {})
    called = []

    def stopped(plan, stage, output, **kwargs):
        called.append((stage, kwargs))
        output.mkdir()
        result = {"state": "stopped"}
        save(output / "run.json", result)
        return result

    monkeypatch.setattr(app, "run_stage", stopped)
    result = app.run_compatibility(plan, tmp_path / "check")
    assert len(called) == 1 and called[0][0] == "cli"
    assert called[0][1]["compatibility"] is True
    assert result["planned"] == 2 and result["state"] == "stopped"
    assert result["stages"][1]["state"] == "planned"
    assert result["included_in_measurements"] is False


def test_guard_failure_after_deadline_retains_first_cause_as_timeout(
    tmp_path, case_record
):
    case, _ = case_record
    endpoint = {
        "continuous_before_ns": 601 * 10**9,
        "continuous_after_ns": 601 * 10**9,
        "awake_ns": 100,
    }

    class Owner:
        first_stdout_ns = None

        def __init__(self, *args, **kwargs):
            kwargs["stdout_path"].write_bytes(b"")
            kwargs["stderr_path"].write_bytes(b"")

        def release(self):
            pass

        def poll_stdout(self):
            return []

        def finish(self, *, terminate):
            return {"returncode": -15, "cleanup_verified": True}

    error = app.common.timing.GuardError("clock_discontinuity", endpoint)
    row = {"id": "cli-1-1", "case_id": case["id"], "stage": "cli"}
    app.execute_row(
        {"workloads": {case["id"]: case}},
        row,
        tmp_path,
        FakeGuard(fail=error),
        lambda: None,
        owner_factory=Owner,
    )
    assert row["outcome"] == "timeout" and row["deadline_exceeded"]
    assert row["failure"] == app.safe_failure(error)


def test_real_cli_and_fixed_replay_send_identical_inputs(tmp_path, monkeypatch, capsys):
    from arm64_rag import inference
    from arm64_rag.corpus import load_paths
    from arm64_rag.index import update_index

    note = tmp_path / "note.txt"
    note.write_text("The batch size is 128 tokens.\n")
    db = tmp_path / "index.sqlite"
    update_index(db, load_paths([note], tmp_path))
    config = app.common.read_json(ROOT / "configs/answer-model.json")
    protocol = app.common.read_json(ROOT / "eval/answers-protocol.json")
    calls, processes = [], []
    stamps = iter(range(1, 10000))
    monkeypatch.setattr(app, "continuous_ns", lambda: next(stamps))
    monkeypatch.setattr(
        app.resource, "getrusage", lambda who: SimpleNamespace(ru_maxrss=4096)
    )
    monkeypatch.setattr(sys, "argv", ["pytest"])

    def launch(*args, **kwargs):
        process = SimpleNamespace(returncode=None)
        processes.append(process)
        return process

    monkeypatch.setattr(inference.subprocess, "Popen", launch)

    class Runtime:
        def __init__(self, *args, **kwargs):
            self.settings = {
                "context_size": kwargs["context_size"],
                "reasoning_budget_tokens": kwargs["reasoning_budget_tokens"],
            }

        def __enter__(self):
            self.process = inference.subprocess.Popen(
                ["server"], start_new_session=True
            )
            return self

        def __exit__(self, *args):
            self.process.returncode = 0

        def info(self):
            return {
                "model_sha256": config["model"]["sha256"],
                "binary_sha256": "b",
                **self.settings,
            }

        def _render(self, body, *, deadline=None):
            return "prompt"

        def render(self, messages, *, json_schema=None):
            return self._render(messages)

        def tokenize(self, prompt):
            return [1, 2]

        def complete(self, token_ids, max_tokens, *, json_schema=None, messages=None):
            self._render(messages)
            calls.append(
                copy.deepcopy(
                    {
                        "tokens": token_ids,
                        "max_tokens": max_tokens,
                        "messages": messages,
                        "schema": json_schema,
                    }
                )
            )
            return {
                "content": json.dumps(
                    {
                        "status": "answered",
                        "claims": [
                            {
                                "text": "The batch size is 128 tokens.",
                                "citations": [{"label": "S1"}],
                            }
                        ],
                    }
                ),
                "tokens_predicted": 5,
                "timings": {"cache_n": 0},
            }

    monkeypatch.setattr(inference, "LocalServer", Runtime)
    case = {
        "id": "dev-014",
        "question": "What is the batch size?",
        "source_ids": ["note.txt"],
        "published_prompt": {
            "prompt_sha256": app.digest(b"prompt"),
            "prompt_tokens": 2,
            "max_output_tokens": 2048,
        },
    }
    plan = {
        "workloads": {case["id"]: case},
        "index": {"path": str(db)},
        "runtime": {"binary": {"path": "server"}, "model": {"path": "model"}},
    }
    cli_dir, fixed_dir = tmp_path / "cli", tmp_path / "fixed"
    cli_dir.mkdir()
    fixed_dir.mkdir()
    assert (
        app.worker(
            {
                "plan": plan,
                "row": {"case_id": case["id"]},
                "stage": "cli",
                "directory": str(cli_dir),
            }
        )
        == 0
    )
    cli_output = capsys.readouterr().out
    assert "128 tokens. [S1]" in cli_output
    baseline = app.common.asset(cli_dir / "answer.json")
    assert (
        app.worker(
            {
                "plan": plan,
                "row": {"case_id": case["id"]},
                "stage": "fixed",
                "replay": baseline,
                "directory": str(fixed_dir),
            }
        )
        == 0
    )
    assert capsys.readouterr().out == cli_output
    assert len(processes) == 2 and all(p.returncode == 0 for p in processes)
    assert len(calls) == 2 and calls[0] == calls[1]
    assert calls[0]["schema"] == protocol["model_output_schema"]
    assert calls[0]["max_tokens"] == protocol["settings"]["max_output_tokens"]
    cli_record = app.common.read_json(cli_dir / "answer.json")
    fixed_record = app.common.read_json(fixed_dir / "answer.json")
    assert app.input_identity(cli_record) == app.input_identity(fixed_record)


@pytest.fixture
def synthetic_install(tmp_path, monkeypatch, case_record):
    from arm64_rag.corpus import load_paths
    from arm64_rag.index import index_inventory, update_index

    project = tmp_path / "project"
    config_dir = project / "configs"
    config_dir.mkdir(parents=True)
    protocol_path = project / "eval/answers-protocol.json"
    protocol_path.parent.mkdir()
    protocol_path.write_bytes((ROOT / "eval/answers-protocol.json").read_bytes())
    runtime = app.common.read_json(ROOT / "configs/runtime.json")
    save(config_dir / "runtime.json", runtime)
    binary = project / "build/cpu/bin/llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"synthetic server; never executable")
    library = binary.with_name("libggml-cpu.dylib")
    library.write_bytes(b"synthetic library")
    model = project / "models/test/fixture.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"synthetic model; never loaded")
    save(
        config_dir / "answer-model.json",
        {
            "directory": "models/test",
            "model": {
                "filename": model.name,
                "size": model.stat().st_size,
                "sha256": app.digest(model.read_bytes()),
            },
        },
    )
    build_dir = project / "artifacts/feasibility/cpu"
    build_dir.mkdir(parents=True)
    flags = {**runtime["cmake"], "GGML_CPU_KLEIDIAI": "OFF"}
    cache = build_dir / "CMakeCache.txt"
    cache.write_text(
        "".join(f"{key}:STRING={value}\n" for key, value in flags.items())
        + "GGML_AVAILABLE_BACKENDS:STRING=ggml-cpu\n"
    )
    commands = build_dir / "compile_commands.json"
    commands.write_text("[]\n")
    save(
        build_dir / "build.json",
        {
            "backend": "cpu",
            "source": runtime["llama_cpp"],
            "flags": flags,
            "kleidiai": None,
            "backends": ["ggml-cpu"],
            "targets": {"llama-server": app.digest(binary.read_bytes())},
            "libraries": {library.name: app.digest(library.read_bytes())},
            "cache_sha256": app.digest(cache.read_bytes()),
            "compile_commands_sha256": app.digest(commands.read_bytes()),
        },
    )
    source = project / "note.txt"
    source.write_text("The batch size is 128 tokens.\n")
    db = project / "data/private/index.sqlite"
    update_index(db, load_paths([source], project))
    inventory = index_inventory(db)
    case, _ = case_record
    cases = {key: {**case, "id": key} for key in (*app.SELECTED, app.EMPTY_ID)}
    host = {
        "identity": {
            "system": "Darwin",
            "release": "synthetic",
            "machine": "arm64",
            "hardware": {
                "model": "Synthetic",
                "physicalcpu": 8,
                "logicalcpu": 8,
                "memsize": 32 * 1024**3,
            },
        },
        "power_source": app.common.host.available("AC Power"),
        "power_mode": app.common.host.available({"powermode": 0}),
        "memory_pressure": app.common.host.available({"code": 1, "label": "normal"}),
        "thermal": app.common.host.unavailable("cpu_limits_not_reported"),
        "pages": app.common.host.available(
            {
                "page_size_bytes": 16384,
                "Pageins": 10,
                "Pageouts": 2,
                "Swapins": 3,
                "Swapouts": 4,
            }
        ),
    }
    monkeypatch.setattr(app, "ROOT", project)
    monkeypatch.setattr(
        app,
        "source_paths",
        lambda: [
            config_dir / "runtime.json",
            config_dir / "answer-model.json",
            protocol_path,
        ],
    )
    monkeypatch.setattr(
        app,
        "public_workloads",
        lambda path: (
            copy.deepcopy(cases),
            {key: inventory[key] for key in ("generation", "fingerprint")},
        ),
    )
    monkeypatch.setattr(app.common.host, "collect", lambda: copy.deepcopy(host))
    return project, db, model, binary, cache


def test_prepare_and_check_real_build_record_contract(synthetic_install):
    project, db, _, _, _ = synthetic_install
    destination = project / "study/plan.json"
    plan = app.freeze(db, destination)
    assert app.check_plan(plan) == plan
    assert app.common.read_json(Path(plan["runtime"]["build"]["path"]))["backends"] == [
        "ggml-cpu"
    ]
    assert plan["host_policy"]["thermal"]["status"] == "unavailable"
    assert set(plan["runtime"]) == {
        "binary",
        "model",
        "libraries",
        "build",
        "cache",
        "compile_commands",
    }


@pytest.mark.parametrize("change", ["native_name", "metal", "model", "library"])
def test_frozen_runtime_changes_are_rejected(synthetic_install, change):
    project, db, model, binary, _ = synthetic_install
    plan = app.freeze(db, project / "study/plan.json")
    if change == "model":
        model.write_bytes(b"changed model")
    elif change == "library":
        binary.with_name("libggml-cpu.dylib").write_bytes(b"changed library")
    else:
        path = project / "artifacts/feasibility/cpu/build.json"
        value = app.common.read_json(path)
        if change == "native_name":
            value["backends"] = ["CPU"]
        else:
            value["flags"]["GGML_METAL"] = "ON"
        save(path, value)
    with pytest.raises(ValueError):
        app.check_plan(plan)


def test_collection_identity_detects_index_wal_and_source_changes(synthetic_install):
    project, db, _, _, _ = synthetic_install
    path = project / "study/plan.json"
    plan = app.freeze(db, path)
    original = app.collection_identity(plan, path)
    wal = Path(str(db) + "-wal")
    wal.write_bytes(b"")
    assert app.collection_identity(plan, path) != original
    wal.unlink()
    (project / "configs/runtime.json").write_text("{}")
    assert app.collection_identity(plan, path) != original


def test_shared_lock_refuses_before_any_worker(tmp_path, monkeypatch):
    lock = tmp_path / "artifacts/benchmarks/benchmark.lock"
    lock.parent.mkdir(parents=True)
    monkeypatch.setattr(app, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_answers.py",
            "run",
            "--plan",
            "unused.json",
            "--stage",
            "cli",
            "--output",
            str(tmp_path / "output"),
        ],
    )
    monkeypatch.setattr(
        app, "run_stage", lambda *args, **kwargs: pytest.fail("Worker launched")
    )
    with app.common.exclusive_lock(lock):
        with pytest.raises(RuntimeError, match="Another benchmark"):
            app.main()


def test_report_rejects_ledger_mutation_after_validation(tmp_path, monkeypatch):
    plan = {
        "sources": {},
        "runtime": {"model": {"sha256": "m"}, "binary": {"sha256": "b"}},
    }
    plan_path = tmp_path / "plan.json"
    save(plan_path, plan)
    path, _ = unstarted_run(tmp_path, plan)
    monkeypatch.setattr(app, "check_plan", lambda value, **kwargs: value)
    original = app.load_run

    def mutate_after(*args, **kwargs):
        result = original(*args, **kwargs)
        changed = app.common.read_json(path)
        changed["extra"] = "changed after validation"
        save(path, changed)
        return result

    monkeypatch.setattr(app, "load_run", mutate_after)
    with pytest.raises(ValueError, match="Evidence changed"):
        app.make_report(plan_path, path)


def test_report_stability_check_follows_final_source_validation(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_bytes(b"original")
    plan = {
        "sources": {"src/arm64_rag/answers.py": app.common.asset(source)},
        "runtime": {"model": {"sha256": "m"}, "binary": {"sha256": "b"}},
    }
    plan_path = tmp_path / "plan.json"
    save(plan_path, plan)
    path, _ = unstarted_run(tmp_path, plan)
    calls = []

    def verify(value, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            source.write_bytes(b"changed during final check")
        return value

    monkeypatch.setattr(app, "check_plan", verify)
    with pytest.raises(ValueError, match="Evidence changed"):
        app.make_report(plan_path, path)


def test_missing_failure_sidecar_keeps_memory_unavailable():
    row = {
        "started_ns": 10,
        "finished_ns": 100,
        "ownership": {
            "wait4_peak_rss_bytes": 999999,
            "returncode": -15,
            "cleanup_verified": True,
            "worker_reaped": True,
        },
        "files": {},
    }
    result = app.failed_observation(row)
    assert result["attempt_and_cleanup_seconds"] == 90 / 1e9
    assert result["native_peak_rss_bytes"] is result["worker_peak_rss_bytes"] is None
    assert result["reported_generated_tokens"] is None
    assert result["cleanup_verified"] and not result["included_in_latency_summary"]


class StageGuard:
    def __init__(self, *args, **kwargs):
        self.evidence = {"first_failure": None}
        self.tick = 0

    def start(self):
        pass

    def idle(self, *args):
        pass

    def check(self, label=None):
        self.tick += 1
        return {
            "continuous_before_ns": self.tick,
            "continuous_after_ns": self.tick,
            "awake_ns": self.tick,
        }

    def close(self):
        return True


def test_source_change_stops_before_second_worker(
    synthetic_install, monkeypatch, case_record
):
    project, db, _, _, _ = synthetic_install
    _, record = case_record
    path = project / "study/plan.json"
    app.freeze(db, path)
    launched = []

    def execute(plan, row, directory, guard, persist, **kwargs):
        launched.append(row["id"])
        save(directory / "answer.json", record)
        row.update(state="finished", outcome="success", failure=None)
        source = project / "configs/runtime.json"
        source.write_text(source.read_text() + " ")
        return True

    monkeypatch.setattr(app.common.timing, "TimingGuard", StageGuard)
    monkeypatch.setattr(app, "execute_row", execute)
    run = app.run_stage(path, "cli", project / "study/cli", owner_factory=object)
    assert len(launched) == 1
    assert run["state"] == "stopped" and not run["integrity_verified"]
    assert run["rows"][0]["inputs_unchanged"] is False
    assert run["rows"][0]["outcome"] == "failed"
    assert all(row["state"] == "planned" for row in run["rows"][1:])


def test_changed_repeat_inputs_stop_during_cli_collection(
    synthetic_install, monkeypatch, case_record
):
    project, db, _, _, _ = synthetic_install
    _, baseline = case_record
    path = project / "study/plan.json"
    app.freeze(db, path)
    seen, launched = set(), []

    def execute(plan, row, directory, guard, persist, **kwargs):
        launched.append(row["id"])
        record = copy.deepcopy(baseline)
        if row["case_id"] in seen:
            record["packed"]["token_ids"] = [2, 1]
        seen.add(row["case_id"])
        save(directory / "answer.json", record)
        row.update(state="finished", outcome="success", failure=None)
        return True

    monkeypatch.setattr(app.common.timing, "TimingGuard", StageGuard)
    monkeypatch.setattr(app, "execute_row", execute)
    run = app.run_stage(path, "cli", project / "study/cli", owner_factory=object)
    assert len(launched) == 5
    assert run["state"] == "stopped" and run["integrity_verified"]
    assert run["rows"][4]["outcome"] == "failed"
    assert run["rows"][4]["input_failure"] == app.safe_failure(
        ValueError("Repeated CLI inputs changed during collection.")
    )
    assert all(row["state"] == "planned" for row in run["rows"][5:])


def test_overdue_constructor_never_releases_application(tmp_path, case_record):
    case, _ = case_record
    released = []

    class Owner:
        first_stdout_ns = None

        def __init__(self, *args, **kwargs):
            kwargs["stdout_path"].write_bytes(b"")
            kwargs["stderr_path"].write_bytes(b"")

        def release(self):
            released.append(True)

        def finish(self, *, terminate):
            assert terminate
            return {"cleanup_verified": True, "returncode": -15}

    row = {"id": "cli-1-1", "case_id": case["id"], "stage": "cli"}
    app.execute_row(
        {"workloads": {case["id"]: case}},
        row,
        tmp_path,
        FakeGuard(elapsed=601 * 10**9),
        lambda: None,
        owner_factory=Owner,
    )
    assert released == []
    assert row["outcome"] == "timeout" and row["deadline_exceeded"]


def test_source_snapshot_keeps_exact_bytes_and_rejects_changes(tmp_path):
    original = tmp_path / "source.py"
    original.write_bytes(b"# Exact source bytes.\r\nvalue = 1\n")
    plan = {"sources": {"scripts/example.py": app.common.asset(original)}}
    output = tmp_path / "stage"
    output.mkdir()
    snapshot = {"complete": False, "files": {}}
    app.snapshot_sources(plan, output, snapshot, lambda: None)
    archived = output / "sources/scripts/example.py"
    assert snapshot["complete"] and archived.read_bytes() == original.read_bytes()
    app.verify_source_snapshot(plan, output, snapshot, required=True)
    with pytest.raises(FileExistsError):
        app.snapshot_sources(
            plan, output, {"complete": False, "files": {}}, lambda: None
        )
    archived.write_bytes(b"modified")
    with pytest.raises(ValueError):
        app.verify_source_snapshot(plan, output, snapshot, required=True)


def test_source_snapshot_failure_preserves_completed_copies(tmp_path, monkeypatch):
    first, second = tmp_path / "one.py", tmp_path / "two.py"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    plan = {
        "sources": {
            "one.py": app.common.asset(first),
            "two.py": app.common.asset(second),
        }
    }
    output = tmp_path / "stage"
    output.mkdir()
    snapshot = {"complete": False, "files": {}}
    original = app.common.save_bytes

    def fail_second(path, raw, **kwargs):
        if path.name == "two.py":
            raise OSError("storage unavailable")
        return original(path, raw, **kwargs)

    monkeypatch.setattr(app.common, "save_bytes", fail_second)
    with pytest.raises(OSError):
        app.snapshot_sources(plan, output, snapshot, lambda: None)
    assert not snapshot["complete"] and set(snapshot["files"]) == {"one.py"}
    assert (output / "sources/one.py").read_bytes() == b"one"
    with pytest.raises(ValueError, match="incomplete"):
        app.verify_source_snapshot(plan, output, snapshot, required=True)


def partial_success(tmp_path, monkeypatch, case_record, *, stage="cli"):
    case, record = copy.deepcopy(case_record)
    planned = app.schedule(stage)
    assert planned[0]["case_id"] != app.EMPTY_ID
    plan = {
        "sources": {},
        "runtime": {"model": {"sha256": "m"}, "binary": {"sha256": "b"}},
        "host_policy": {"identity": {}},
        "timing_policy": app.common.timing.POLICY,
        "workloads": {
            key: {**case, "id": key} for key in (*app.SELECTED, app.EMPTY_ID)
        },
    }
    plan_path = tmp_path / "plan.json"
    save(plan_path, plan)
    directory = tmp_path / stage
    directory.mkdir()
    save(directory / "plan.json", plan)
    claim_path = tmp_path / "claim.json"
    save(
        claim_path,
        {
            "plan_sha256": app.digest((directory / "plan.json").read_bytes()),
            "stage": stage,
            "purpose": "measurement",
            "output": str(directory),
        },
    )
    rows = [{**row, "stage": stage} for row in planned]
    first = rows[0]
    row_dir = directory / first["id"]
    row_dir.mkdir()
    save(row_dir / "answer.json", record)
    save(row_dir / "observation.json", observation(stage))
    save(row_dir / "observer-end.json", {"export_end_ns": 140})
    (row_dir / "stdout.txt").write_text(render_answer(record) + "\n")
    first.update(
        state="finished",
        outcome="success",
        failure=None,
        started_ns=10,
        first_stdout_ns=111,
        exit_observed_ns=150,
        inputs_unchanged=True,
        host_before={"synthetic": True},
        host_after={"synthetic": True},
        host_violations=[],
        files=app.record_files(row_dir),
        ownership={
            "cleanup_verified": True,
            "returncode": 0,
            "first_stdout_ns": 111,
            "first_failure": None,
            "cleanup_failures": [],
        },
    )
    if stage == "fixed":
        replay = tmp_path / "selected-cli.json"
        save(replay, record)
        first["replay"] = app.common.asset(replay)
    run = {
        "kind": app.KIND,
        "stage": stage,
        "purpose": "measurement",
        "state": "stopped",
        "plan_sha256": app.digest((directory / "plan.json").read_bytes()),
        "launch_claim": app.common.asset(claim_path),
        "source_snapshot": {"complete": True, "files": {}},
        "timing_valid": True,
        "integrity_verified": True,
        "rows": rows,
        "timing_guard": {
            "first_failure": None,
            "checkpoints": [
                {
                    "label": first["id"] + ":start",
                    "sample": {"continuous_before_ns": 10},
                },
                {
                    "label": first["id"] + ":exit",
                    "sample": {"continuous_after_ns": 150},
                },
            ],
        },
    }
    save(directory / "run.json", run)
    monkeypatch.setattr(app, "check_plan", lambda value, **kwargs: value)
    monkeypatch.setattr(app.common.timing, "validate_evidence", lambda *args: True)
    monkeypatch.setattr(
        app.public, "public_timing", lambda value, passed: {"valid": passed}
    )
    monkeypatch.setattr(app.common.host, "admit", lambda *args: [])
    monkeypatch.setattr(app.common.host, "interval_violations", lambda *args: [])
    return plan_path, directory / "run.json", run


@pytest.mark.parametrize(
    "change",
    [
        "export_after_exit",
        "export_before_start",
        "validation_before_complete",
        "validation_after_return",
        "validation_backwards",
        "validation_missing",
    ],
)
def test_saved_observer_intervals_reject_impossible_rehashed_timestamps(
    tmp_path, monkeypatch, case_record, change
):
    stage = "cli" if change.startswith("export_") else "fixed"
    plan_path, path, run = partial_success(
        tmp_path, monkeypatch, case_record, stage=stage
    )
    first = run["rows"][0]
    filename = "observer-end.json" if stage == "cli" else "observation.json"
    raw_path = Path(first["files"][filename]["path"])
    raw = app.common.read_json(raw_path)
    if change == "export_after_exit":
        raw["export_end_ns"] = first["exit_observed_ns"] + 1
    elif change == "export_before_start":
        raw["export_end_ns"] = 119
    elif change == "validation_before_complete":
        raw["events"]["validation_start_ns"] = 79
    elif change == "validation_after_return":
        raw["events"]["validation_end_ns"] = 101
    elif change == "validation_backwards":
        raw["events"]["validation_end_ns"] = 81
    else:
        raw["events"].pop("validation_end_ns")
    save(raw_path, raw)
    first["files"][filename] = app.common.asset(raw_path)
    save(path, run)
    match = "Observer export" if stage == "cli" else "Fixed answer validation"
    with pytest.raises(ValueError, match=match):
        app.load_run(app.common.read_json(plan_path), path, stage)
    with pytest.raises(ValueError, match=match):
        app.metrics(first)


def test_failed_post_host_probe_excludes_completed_answer_from_summary(
    tmp_path, monkeypatch, case_record
):
    plan, path, run = partial_success(tmp_path, monkeypatch, case_record)
    first = run["rows"][0]
    first.pop("host_after")
    first["host_observation_failure"] = app.safe_failure(OSError("probe failed"))
    save(path, run)
    report = app.make_report(plan, path)
    result = report["stages"]["cli"]
    assert report["successes"] == 1
    assert result["rows"][0]["metrics"]["application_return_seconds"] is not None
    assert result["rows"][0]["timing_valid"] is False
    assert result["rows"][0]["host_observation_failure_sha256"]
    assert result["descriptive_summaries"] == {}
    assert not report["protocol_complete"]
    first.pop("host_observation_failure")
    save(path, run)
    with pytest.raises(ValueError, match="retained failure"):
        app.make_report(plan, path)


@pytest.mark.parametrize(
    "change", ["same_length_tokens", "source_headings", "packed_metadata"]
)
def test_offline_fixed_replay_rejects_full_input_drift(
    tmp_path, monkeypatch, case_record, change
):
    plan_path, path, run = partial_success(
        tmp_path, monkeypatch, case_record, stage="fixed"
    )
    first = run["rows"][0]
    answer_path = Path(first["files"]["answer.json"]["path"])
    record = app.common.read_json(answer_path)
    if change == "same_length_tokens":
        record["packed"]["token_ids"].reverse()
    elif change == "source_headings":
        record["packed"]["sources"][0]["headings"] = ["Changed heading"]
    else:
        record["packed"]["omitted"] = [{"reason": "different packing metadata"}]
    save(answer_path, record)
    first["files"]["answer.json"] = app.common.asset(answer_path)
    save(path, run)
    with pytest.raises(ValueError, match="Saved fixed-context inputs changed"):
        app.load_run(app.common.read_json(plan_path), path, "fixed")
