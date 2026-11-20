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
            "raw_content": content,
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


def test_schedule_pairs_every_question_once_per_backend_in_balanced_order():
    rows = app.schedule()
    assert len(rows) == 88 and len({row["id"] for row in rows}) == 88
    first = {}
    for left, right in zip(rows[::2], rows[1::2], strict=True):
        assert left["pair"] == right["pair"] and left["case_id"] == right["case_id"]
        assert {left["backend"], right["backend"]} == set(app.BACKENDS)
        first.setdefault(left["case_id"], []).append(left["backend"])
    assert set(first) == {*app.CASES, app.EMPTY_ID}
    assert all(len(value) == 1 for case, value in first.items() if case in app.CASES)
    assert len(first[app.EMPTY_ID]) == 4
    questions = [value[0] for case, value in first.items() if case in app.CASES]
    assert questions.count("cpu") == questions.count("kleidiai") == 20
    assert first[app.EMPTY_ID].count("cpu") == 2
    for block in range(1, 5):
        orders = [
            left["backend"]
            for left in rows[::2]
            if left["block"] == block and left["case_id"] != app.EMPTY_ID
        ]
        assert orders.count("cpu") == orders.count("kleidiai") == 5
    assert app.schedule() == rows


def published_builds():
    return {
        backend: (
            None,
            json.loads(
                (
                    ROOT / f"docs/results/runtime-feasibility/{backend}-build.json"
                ).read_text()
            ),
        )
        for backend in app.BACKENDS
    }


def test_reviewed_records_cover_both_backends_with_shared_inputs(monkeypatch):
    monkeypatch.setattr(app, "builds", published_builds)
    records = app.reviewed_records()
    assert sorted(records) == sorted(app.CASES)
    tokens = {
        backend: sum(
            item["reviewed_answers"][backend]["generated_tokens"]
            for item in records.values()
        )
        for backend in app.BACKENDS
    }
    assert tokens == {"cpu": 39522, "kleidiai": 45009}
    assert records["val-006"]["published_prompt"]["prompt_tokens"] == 2278


def test_reviewed_records_reject_answers_from_another_server(monkeypatch):
    def swapped():
        records = published_builds()
        servers = {
            backend: build["targets"]["llama-server"]
            for backend, (_, build) in records.items()
        }
        records["cpu"][1]["targets"]["llama-server"] = servers["kleidiai"]
        return records

    monkeypatch.setattr(app, "builds", swapped)
    with pytest.raises(ValueError, match="another server"):
        app.reviewed_records()


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


def observation():
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
    return result


def runtime_plan():
    return {
        "runtime": {
            "model": {"sha256": "m"},
            "backends": {
                "cpu": {"binary": {"sha256": "b"}},
                "kleidiai": {"binary": {"sha256": "k"}},
            },
        }
    }


def metric_row():
    return {
        "id": "b1-p01-cpu",
        "pair": "b1-p01",
        "case_id": "dev-014",
        "backend": "cpu",
        "started_ns": 10,
        "first_stdout_ns": 111,
        "exit_observed_ns": 150,
    }


def test_observation_rejects_the_other_backends_server(case_record):
    _, record = case_record
    row = {**metric_row(), "backend": "kleidiai"}
    with pytest.raises(ValueError, match="differs from the frozen runtime"):
        app.check_observation(
            observation(), row, record, runtime_plan(), {"export_end_ns": 140}
        )
    record["runtime"]["binary_sha256"] = "k"
    app.check_observation(
        observation(), row, record, runtime_plan(), {"export_end_ns": 140}
    )


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
            runtime_plan(),
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
    metrics, _, _, answer = app.metrics(row)
    assert answer == {
        "final_content_sha256": app.digest(
            record["raw_completion"]["content"].encode()
        ),
        "raw_content_sha256": app.digest(
            record["raw_completion"]["raw_content"].encode()
        ),
        "generated_tokens": 5,
    }
    assert metrics["application_return_seconds"] == 90 / 1e9
    assert metrics["pipe_observed_answer_seconds"] == 101 / 1e9
    assert metrics["worker_exit_observed_seconds"] == 140 / 1e9
    assert metrics["observer_export_seconds"] == 20 / 1e9
    assert metrics["model_time_to_first_token_seconds"] is None
    assert metrics["native_peak_rss_bytes"] == 5000


def test_private_artifact_names_cannot_enter_public_map(tmp_path):
    (tmp_path / "private-customer-name.txt").write_text("secret")
    with pytest.raises(ValueError, match="unexpected file"):
        app.record_files(tmp_path)


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
    row = {
        "id": "b1-p01-cpu",
        "case_id": case["id"],
        "backend": "cpu",
        "state": "planned",
    }
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
    row = {"id": "b1-p01-cpu", "case_id": case["id"], "backend": "cpu"}
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
        argv_seen.append(list(sys.argv))
        with inference.LocalServer() as server:
            answer = answers.answer_question(None, None, server, None)
            server.complete([1, 2], 2048)
        save(tmp_path / "answer.json", answer)
        print(render_answer(answer))

    monkeypatch.setattr(cli, "main", cli_main)
    plan = {
        "workloads": {case["id"]: case},
        "index": {"path": "index.sqlite"},
        "runtime": {
            "backends": {
                "cpu": {"binary": {"path": "cpu-server"}},
                "kleidiai": {"binary": {"path": "kleidiai-server"}},
            }
        },
    }
    assert (
        app.worker(
            {
                "plan": plan,
                "row": {"case_id": case["id"], "backend": "kleidiai"},
                "directory": str(tmp_path),
            }
        )
        == 0
    )
    assert capsys.readouterr().out == render_answer(record) + "\n"
    command, native = argv_seen
    assert command[:3] == ["arm64-rag", "ask", case["question"]]
    assert command[command.index("--server-binary") + 1] == "kleidiai-server"
    assert command.count("--server-binary") == 1
    assert native == ["server", "--ctx-size", "8192"]
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
    row = {"id": "b1-p01-cpu", "case_id": case["id"], "backend": "cpu"}
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


def test_collection_stops_after_failure_and_verifies_before_guard_close(
    tmp_path, monkeypatch
):
    model = tmp_path / "model"
    model.write_bytes(b"model")
    plan = {
        "runtime": {"model": {"path": str(model)}, "backends": {}},
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
    run = app.run_collection(plan_path, tmp_path / "run", owner_factory=object)
    assert len(launches) == 1
    assert checks == [True, False]
    assert guards[0].closed and run["state"] == "stopped"
    assert sum(row["state"] == "planned" for row in run["rows"]) == 87


def unstarted_run(tmp_path, plan):
    plan.setdefault("sources", {})
    path = tmp_path / "run"
    path.mkdir()
    save(path / "plan.json", plan)
    claim_path = tmp_path / "claim.json"
    save(
        claim_path,
        {
            "plan_sha256": app.digest((path / "plan.json").read_bytes()),
            "purpose": "measurement",
            "output": str(path),
        },
    )
    run = {
        "kind": app.KIND,
        "state": "planned",
        "purpose": "measurement",
        "plan_sha256": app.digest((path / "plan.json").read_bytes()),
        "rows": app.schedule(),
        "timing_guard": None,
        "timing_valid": False,
        "integrity_verified": False,
        "failure": None,
        "launch_claim": app.common.asset(claim_path),
        "source_snapshot": {"complete": False, "files": {}},
    }
    save(path / "run.json", run)
    return path / "run.json", run


def test_report_retains_unattempted_rows_and_omits_private_fields(
    tmp_path, monkeypatch
):
    source = tmp_path / "private-person-project.py"
    source.write_text("private source bytes")
    plan = {
        **runtime_plan(),
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
    assert report["planned"] == report["unattempted"] == 88
    assert report["successes"] == report["failed"] == report["unresolved"] == 0
    assert report["protocol_complete"] is False and report["comparison"] is None
    assert len(report["collection"]["rows"]) == 88
    raw = json.dumps(report)
    for secret in ("/Users", "keep local", "private_notes", "environment"):
        assert secret not in raw
    unstarted = app.make_report(plan_path)
    assert unstarted["unobserved"] == 88 and unstarted["comparison"] is None


def test_unattempted_row_cannot_smuggle_private_host_findings(tmp_path):
    plan = {}
    path, run = unstarted_run(tmp_path, plan)
    run["rows"][0]["host_violations"] = ["/Users/private/person"]
    save(path, run)
    with pytest.raises(ValueError, match="host findings"):
        app.load_run(plan, path)


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
        app.load_run(plan, path)


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
        **runtime_plan(),
    }
    row = {"id": "b1-p01-cpu", "case_id": case["id"], "backend": "cpu"}
    assert not app.execute_row(
        plan, row, tmp_path, FakeGuard(elapsed=150), lambda: None, owner_factory=Owner
    )
    assert row["outcome"] == "failed"
    assert row["failure"] == app.safe_failure(
        ValueError("Captured output differs from the original answer renderer.")
    )


def test_compatibility_pairs_only_the_median_case_and_is_not_measured_input(
    tmp_path,
):
    rows = app.schedule(compatibility=True)
    assert [(row["case_id"], row["backend"]) for row in rows] == [
        ("val-015", "cpu"),
        ("val-015", "kleidiai"),
    ]
    plan = {}
    path, run = unstarted_run(tmp_path, plan)
    run["purpose"] = "excluded_compatibility"
    save(path, run)
    with pytest.raises(ValueError, match="Compatibility observations"):
        app.load_run(plan, path)


def test_check_command_runs_only_the_excluded_pair(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(app, "ROOT", tmp_path)
    monkeypatch.setattr(
        app,
        "run_collection",
        lambda *args, **kwargs: calls.append(kwargs) or {"state": "completed"},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_answers.py", "check", "--plan", "plan.json", "--output", "out"],
    )
    app.main()
    assert calls == [{"compatibility": True}]


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
    row = {"id": "b1-p01-cpu", "case_id": case["id"], "backend": "cpu"}
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


def test_real_cli_sends_identical_inputs_to_both_servers(tmp_path, monkeypatch, capsys):
    from arm64_rag import inference
    from arm64_rag.corpus import load_paths
    from arm64_rag.index import update_index

    note = tmp_path / "note.txt"
    note.write_text("The batch size is 128 tokens.\n")
    db = tmp_path / "index.sqlite"
    update_index(db, load_paths([note], tmp_path))
    config = app.common.read_json(ROOT / "configs/answer-model.json")
    protocol = app.common.read_json(ROOT / "eval/answers-protocol.json")
    calls, processes, binaries = [], [], []
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
        def __init__(self, binary, *args, **kwargs):
            binaries.append(str(binary))
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
        "runtime": {
            "backends": {
                backend: {"binary": {"path": f"{backend}-server"}}
                for backend in app.BACKENDS
            }
        },
    }
    outputs, records = [], []
    for backend in app.BACKENDS:
        directory = tmp_path / backend
        directory.mkdir()
        assert (
            app.worker(
                {
                    "plan": plan,
                    "row": {"case_id": case["id"], "backend": backend},
                    "directory": str(directory),
                }
            )
            == 0
        )
        outputs.append(capsys.readouterr().out)
        records.append(app.common.read_json(directory / "answer.json"))
    assert "128 tokens. [S1]" in outputs[0] and outputs[0] == outputs[1]
    assert binaries == ["cpu-server", "kleidiai-server"]
    assert len(processes) == 2 and all(p.returncode == 0 for p in processes)
    assert len(calls) == 2 and calls[0] == calls[1]
    assert calls[0]["schema"] == protocol["model_output_schema"]
    assert calls[0]["max_tokens"] == protocol["settings"]["max_output_tokens"]
    assert app.input_identity(records[0]) == app.input_identity(records[1])


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
    for backend in app.BACKENDS:
        binary = project / f"build/{backend}/bin/llama-server"
        binary.parent.mkdir(parents=True)
        binary.write_bytes(f"synthetic {backend} server; never executable".encode())
        library = binary.with_name("libggml-cpu.dylib")
        library.write_bytes(f"synthetic {backend} library".encode())
        build_dir = project / "artifacts/feasibility" / backend
        build_dir.mkdir(parents=True)
        flags = {
            **runtime["cmake"],
            "GGML_CPU_KLEIDIAI": "ON" if backend == "kleidiai" else "OFF",
        }
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
                "backend": backend,
                "source": runtime["llama_cpp"],
                "flags": flags,
                "kleidiai": runtime["kleidiai"] if backend == "kleidiai" else None,
                "backends": ["ggml-cpu"],
                "targets": {"llama-server": app.digest(binary.read_bytes())},
                "libraries": {library.name: app.digest(library.read_bytes())},
                "cache_sha256": app.digest(cache.read_bytes()),
                "compile_commands_sha256": app.digest(commands.read_bytes()),
                "compiler_flags": {"c": "-O3"},
                "compilers": {"c": "clang"},
                "tools": {"cmake": "4"},
                "system": {"machine": "arm64"},
            },
        )
    source = project / "note.txt"
    source.write_text("The batch size is 128 tokens.\n")
    db = project / "data/private/index.sqlite"
    update_index(db, load_paths([source], project))
    inventory = index_inventory(db)
    case, _ = case_record
    cases = {key: {**case, "id": key} for key in (*app.CASES, app.EMPTY_ID)}
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
    return project, db, model


def test_prepare_and_check_real_build_record_contract(synthetic_install):
    project, db, _ = synthetic_install
    destination = project / "study/plan.json"
    plan = app.freeze(db, destination)
    assert app.check_plan(plan) == plan
    assert plan["host_policy"]["thermal"]["status"] == "unavailable"
    assert set(plan["runtime"]) == {"model", "backends"}
    for backend, assets in plan["runtime"]["backends"].items():
        assert set(assets) == {
            "binary",
            "libraries",
            "build",
            "cache",
            "compile_commands",
        }
        assert (
            Path(assets["binary"]["path"])
            == project / f"build/{backend}/bin/llama-server"
        )
    cpu, kleidiai = (plan["runtime"]["backends"][name] for name in app.BACKENDS)
    assert cpu["binary"]["sha256"] != kleidiai["binary"]["sha256"]


@pytest.mark.parametrize(
    "change",
    [
        "native_name",
        "metal",
        "model",
        "library",
        "kleidiai_off",
        "cpu_kleidiai_pin",
        "paired_compiler",
        "swapped_server",
    ],
)
def test_frozen_runtime_changes_are_rejected(synthetic_install, change):
    project, db, model = synthetic_install
    plan = app.freeze(db, project / "study/plan.json")
    records = {
        backend: project / "artifacts/feasibility" / backend / "build.json"
        for backend in app.BACKENDS
    }
    if change == "model":
        model.write_bytes(b"changed model")
    elif change == "library":
        (project / "build/kleidiai/bin/libggml-cpu.dylib").write_bytes(b"changed")
    elif change == "swapped_server":
        cpu = project / "build/cpu/bin/llama-server"
        kleidiai = project / "build/kleidiai/bin/llama-server"
        raw = cpu.read_bytes()
        cpu.write_bytes(kleidiai.read_bytes())
        kleidiai.write_bytes(raw)
    else:
        backend = "cpu" if change in {"native_name", "cpu_kleidiai_pin"} else "kleidiai"
        value = app.common.read_json(records[backend])
        if change == "native_name":
            value["backends"] = ["CPU"]
        elif change == "metal":
            value["flags"]["GGML_METAL"] = "ON"
        elif change == "kleidiai_off":
            value["flags"]["GGML_CPU_KLEIDIAI"] = "OFF"
        elif change == "cpu_kleidiai_pin":
            value["kleidiai"] = app.common.read_json(records["kleidiai"])["kleidiai"]
        else:
            value["compilers"] = {"c": "another clang"}
        save(records[backend], value)
    with pytest.raises(ValueError):
        app.check_plan(plan)


def test_collection_identity_detects_index_wal_and_source_changes(synthetic_install):
    project, db, _ = synthetic_install
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
            "--output",
            str(tmp_path / "output"),
        ],
    )
    monkeypatch.setattr(
        app, "run_collection", lambda *args, **kwargs: pytest.fail("Worker launched")
    )
    with app.common.exclusive_lock(lock):
        with pytest.raises(RuntimeError, match="Another benchmark"):
            app.main()


def test_report_rejects_ledger_mutation_after_validation(tmp_path, monkeypatch):
    plan = {"sources": {}, **runtime_plan()}
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
        **runtime_plan(),
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
    project, db, _ = synthetic_install
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
    run = app.run_collection(path, project / "study/run", owner_factory=object)
    assert len(launched) == 1
    assert run["state"] == "stopped" and not run["integrity_verified"]
    assert run["rows"][0]["inputs_unchanged"] is False
    assert run["rows"][0]["outcome"] == "failed"
    assert all(row["state"] == "planned" for row in run["rows"][1:])


def test_backend_input_difference_stops_collection(
    synthetic_install, monkeypatch, case_record
):
    project, db, _ = synthetic_install
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
    run = app.run_collection(path, project / "study/run", owner_factory=object)
    assert len(launched) == 2
    assert run["state"] == "stopped" and run["integrity_verified"]
    assert run["rows"][1]["outcome"] == "failed"
    assert run["rows"][1]["input_failure"] == app.safe_failure(
        ValueError("Inputs for the same question differ between observations.")
    )
    assert all(row["state"] == "planned" for row in run["rows"][2:])


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

    row = {"id": "b1-p01-cpu", "case_id": case["id"], "backend": "cpu"}
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


def successful_row(directory, row, record, *, started=10):
    row_dir = directory / row["id"]
    row_dir.mkdir()
    save(row_dir / "answer.json", record)
    save(row_dir / "observation.json", observation())
    save(row_dir / "observer-end.json", {"export_end_ns": 140})
    (row_dir / "stdout.txt").write_text(render_answer(record) + "\n")
    row.update(
        state="finished",
        outcome="success",
        failure=None,
        started_ns=started,
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


def partial_success(tmp_path, monkeypatch, case_record, *, completed=1):
    case, record = copy.deepcopy(case_record)
    record["runtime"]["binary_sha256"] = "b"
    raw = record["raw_completion"]
    case["reviewed_answers"] = {
        "cpu": {
            "final_content_sha256": app.digest(raw["content"].encode()),
            "raw_content_sha256": app.digest(raw["raw_content"].encode()),
            "generated_tokens": raw["tokens_predicted"],
        },
        "kleidiai": {
            "final_content_sha256": app.digest(b"another reviewed answer"),
            "raw_content_sha256": app.digest(b"another reviewed answer"),
            "generated_tokens": 9,
        },
    }
    planned = app.schedule()
    plan = {
        "sources": {},
        **runtime_plan(),
        "host_policy": {"identity": {}},
        "timing_policy": app.common.timing.POLICY,
        "workloads": {key: {**case, "id": key} for key in (*app.CASES, app.EMPTY_ID)},
    }
    plan["runtime"]["backends"]["kleidiai"]["binary"]["sha256"] = "b"
    plan_path = tmp_path / "plan.json"
    save(plan_path, plan)
    directory = tmp_path / "run"
    directory.mkdir()
    save(directory / "plan.json", plan)
    claim_path = tmp_path / "claim.json"
    save(
        claim_path,
        {
            "plan_sha256": app.digest((directory / "plan.json").read_bytes()),
            "purpose": "measurement",
            "output": str(directory),
        },
    )
    rows = planned
    checkpoints = []
    for row in rows[:completed]:
        assert row["case_id"] != app.EMPTY_ID
        successful_row(directory, row, record)
        checkpoints.extend(
            [
                {"label": row["id"] + ":start", "sample": {"continuous_before_ns": 10}},
                {"label": row["id"] + ":exit", "sample": {"continuous_after_ns": 150}},
            ]
        )
    run = {
        "kind": app.KIND,
        "purpose": "measurement",
        "state": "stopped",
        "plan_sha256": app.digest((directory / "plan.json").read_bytes()),
        "launch_claim": app.common.asset(claim_path),
        "source_snapshot": {"complete": True, "files": {}},
        "timing_valid": True,
        "integrity_verified": True,
        "rows": rows,
        "timing_guard": {"first_failure": None, "checkpoints": checkpoints},
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


@pytest.mark.parametrize("change", ["export_after_exit", "export_before_start"])
def test_saved_observer_intervals_reject_impossible_rehashed_timestamps(
    tmp_path, monkeypatch, case_record, change
):
    plan_path, path, run = partial_success(tmp_path, monkeypatch, case_record)
    first = run["rows"][0]
    raw_path = Path(first["files"]["observer-end.json"]["path"])
    raw = app.common.read_json(raw_path)
    if change == "export_after_exit":
        raw["export_end_ns"] = first["exit_observed_ns"] + 1
    else:
        raw["export_end_ns"] = 119
    save(raw_path, raw)
    first["files"]["observer-end.json"] = app.common.asset(raw_path)
    save(path, run)
    with pytest.raises(ValueError, match="Observer export"):
        app.load_run(app.common.read_json(plan_path), path)
    with pytest.raises(ValueError, match="Observer export"):
        app.metrics(first)


def test_failed_post_host_probe_excludes_completed_answer_from_comparison(
    tmp_path, monkeypatch, case_record
):
    plan, path, run = partial_success(tmp_path, monkeypatch, case_record)
    first = run["rows"][0]
    first.pop("host_after")
    first["host_observation_failure"] = app.safe_failure(OSError("probe failed"))
    save(path, run)
    report = app.make_report(plan, path)
    result = report["collection"]
    assert report["successes"] == 1
    assert result["rows"][0]["metrics"]["application_return_seconds"] is not None
    assert result["rows"][0]["timing_valid"] is False
    assert run["rows"][0]["backend"] == "kleidiai"
    assert result["rows"][0]["matches_reviewed_answer"] is False
    assert report["quality_review"] == "pending"
    assert result["rows"][0]["host_observation_failure_sha256"]
    assert report["comparison"] is None
    assert not report["protocol_complete"]
    first.pop("host_observation_failure")
    save(path, run)
    with pytest.raises(ValueError, match="retained failure"):
        app.make_report(plan, path)


@pytest.mark.parametrize(
    "change", ["same_length_tokens", "source_headings", "packed_metadata"]
)
def test_saved_backend_pair_rejects_full_input_drift(
    tmp_path, monkeypatch, case_record, change
):
    plan_path, path, run = partial_success(
        tmp_path, monkeypatch, case_record, completed=2
    )
    app.load_run(app.common.read_json(plan_path), path)
    second = run["rows"][1]
    answer_path = Path(second["files"]["answer.json"]["path"])
    record = app.common.read_json(answer_path)
    if change == "same_length_tokens":
        record["packed"]["token_ids"].reverse()
    elif change == "source_headings":
        record["packed"]["sources"][0]["headings"] = ["Changed heading"]
    else:
        record["packed"]["omitted"] = [{"reason": "different packing metadata"}]
    save(answer_path, record)
    second["files"]["answer.json"] = app.common.asset(answer_path)
    save(path, run)
    with pytest.raises(ValueError, match="Saved inputs for one question differ"):
        app.load_run(app.common.read_json(plan_path), path)


def compared_row(case_id, backend, *, seconds, tokens, start, prompt_s, rss=10):
    return {
        "case_id": case_id,
        "backend": backend,
        "metrics": {
            "pipe_observed_answer_seconds": seconds,
            "server_initialization_seconds": start,
            "native_spawn_to_ready_seconds": start - 1,
            "complete_request_seconds": seconds - start - 1,
            "native_peak_rss_bytes": rss,
        },
        "native_diagnostics": {
            "prompt_n": 100,
            "prompt_ms": prompt_s * 1000,
            "predicted_ms": (seconds - start - 1 - prompt_s) * 1000,
            "predicted_n": tokens,
            "generated_tokens": tokens,
        },
        "matches_reviewed_answer": case_id != "dev-001" or backend == "cpu",
    }


def test_comparison_totals_ratios_and_similar_length_subset(monkeypatch):
    monkeypatch.setattr(app, "CASES", ("dev-001", "dev-002"))
    rows = [
        compared_row(
            "dev-001", "kleidiai", seconds=30, tokens=200, start=10, prompt_s=4
        ),
        compared_row("dev-001", "cpu", seconds=20, tokens=100, start=5, prompt_s=6),
        compared_row("dev-002", "cpu", seconds=40, tokens=100, start=5, prompt_s=6),
        compared_row(
            "dev-002", "kleidiai", seconds=50, tokens=104, start=10, prompt_s=4
        ),
        compared_row(app.EMPTY_ID, "cpu", seconds=6, tokens=1, start=5, prompt_s=0),
        compared_row(
            app.EMPTY_ID, "kleidiai", seconds=11, tokens=1, start=10, prompt_s=0
        ),
    ]
    result = app.compare(rows)
    total = result["all_questions"]
    assert total["questions"] == 2
    assert total["answer_seconds_total"] == {"cpu": 60, "kleidiai": 80}
    assert total["kleidiai_over_cpu_total"] == pytest.approx(80 / 60)
    assert total["kleidiai_over_cpu_per_question"]["median"] == pytest.approx(
        (30 / 20 + 50 / 40) / 2
    )
    assert total["kleidiai_slower_questions"] == 2
    assert total["generated_tokens_total"] == {"cpu": 200, "kleidiai": 304}
    assert total["prompt_tokens_per_second"]["kleidiai"] == pytest.approx(200 / 8)
    assert total["generation_ms_per_token"]["cpu"] == pytest.approx(
        1000 * (8 + 28) / 200
    )
    similar = result["similar_length"]
    assert similar["questions"] == 1 and similar["kleidiai_over_cpu_total"] == 1.25
    assert result["by_first_backend"]["kleidiai"]["questions"] == 1
    assert result["by_first_backend"]["cpu"]["answer_seconds_total"]["cpu"] == 40
    assert (
        result["empty_retrieval"]["kleidiai"]["pipe_observed_answer_seconds"]["median"]
        == 11
    )
    assert result["reviewed_answer_matches"] == {"cpu": 2, "kleidiai": 1}
    with pytest.raises(ValueError, match="every question"):
        app.compare(rows[1:])


class PipelineGuard:
    def __init__(self, *args, **kwargs):
        self.now = 10**12
        self.evidence = {"first_failure": None, "checkpoints": []}

    def continuous_ns(self):
        self.now += 1
        return self.now

    def check(self, label=None):
        self.now += 10**6
        sample = {
            "continuous_before_ns": self.now,
            "awake_ns": self.now,
            "continuous_after_ns": self.now,
        }
        if label:
            self.evidence["checkpoints"].append({"label": label, "sample": sample})
        return sample

    def start(self):
        return self.check("session:start")

    def idle(self, seconds, label):
        self.check(label + ":start")
        return self.check(label + ":end")

    def fail(self, reason, sample=None):
        raise app.common.timing.GuardError(reason, sample)

    def close(self):
        self.check("session:end")
        self.check("session:closed")
        return True


def test_full_pipeline_produces_the_planned_comparison(
    tmp_path, monkeypatch, case_record
):
    case, base = case_record
    binaries = {backend: app.digest(backend.encode()) for backend in app.BACKENDS}
    for backend in app.BACKENDS:
        (tmp_path / backend).write_bytes(backend.encode())

    def record_for(case_id, backend):
        record = copy.deepcopy(base)
        record["question"] = case_id
        record["runtime"] = {"model_sha256": "m", "binary_sha256": binaries[backend]}
        if case_id == app.EMPTY_ID:
            record["retrieval"]["results"] = []
            record["packed"]["sources"] = []
            record.update(
                raw_completion=None,
                answer={"status": "insufficient_evidence", "claims": []},
            )
            return record
        tokens = 100 if backend == "cpu" else 103
        record["raw_completion"].update(
            raw_content=record["raw_completion"]["content"] + backend,
            tokens_predicted=tokens,
            timings={
                "cache_n": 0,
                "prompt_n": 2,
                "prompt_ms": 1000.0 if backend == "cpu" else 700.0,
                "predicted_n": tokens,
                "predicted_ms": 8000.0 if backend == "cpu" else 12000.0,
            },
        )
        return record

    def reviewed(case_id, backend):
        raw = record_for(case_id, backend)["raw_completion"]
        return {
            "final_content_sha256": app.digest(raw["content"].encode()),
            "raw_content_sha256": app.digest(raw["raw_content"].encode()),
            "generated_tokens": raw["tokens_predicted"],
        }

    workloads = {
        case_id: {
            **case,
            "id": case_id,
            "question": case_id,
            "reviewed_answers": {b: reviewed(case_id, b) for b in app.BACKENDS},
        }
        for case_id in app.CASES
    }
    workloads[app.EMPTY_ID] = {
        **case,
        "id": app.EMPTY_ID,
        "question": app.EMPTY_ID,
        "published_prompt": None,
        "reviewed_answers": None,
    }
    workloads["val-015"]["reviewed_answers"]["kleidiai"]["generated_tokens"] = 1
    plan = {
        "kind": app.KIND,
        "workloads": workloads,
        "sources": {},
        "runtime": {
            "model": {"sha256": "m", "path": str(tmp_path / "model")},
            "backends": {
                b: {"binary": app.common.asset(tmp_path / b)} for b in app.BACKENDS
            },
        },
        "host_policy": {"identity": {}},
        "timing_policy": app.common.timing.POLICY,
    }
    plan_path = tmp_path / "plan.json"
    save(plan_path, plan)

    class Owner:
        def __init__(self, argv, *, cwd, stdout_path, stderr_path, clock_ns):
            directory = stdout_path.parent
            row = app.common.read_json(directory / "input.json")["row"]
            record = record_for(row["case_id"], row["backend"])
            start = clock_ns()
            names = [
                "server_initialization_start_ns",
                "native_spawn_start_ns",
                "native_spawn_return_ns",
                "server_ready_ns",
                "answer_start_ns",
            ]
            if row["case_id"] != app.EMPTY_ID:
                names += ["complete_start_ns", "complete_end_ns"]
            names += [
                "answer_end_ns",
                "application_return_ns",
                "stdout_flushed_ns",
                "observer_export_start_ns",
            ]
            delay = 2 if row["backend"] == "kleidiai" else 1
            events = {
                name: start + delay * 10 * number
                for number, name in enumerate(names, 1)
            }
            save(directory / "answer.json", record)
            save(
                directory / "observation.json",
                {
                    "events": events,
                    "native_launches": 1,
                    "native_reaped": True,
                    "failure": None,
                    "exit_code": 0,
                    "worker_peak_rss_bytes": 1000,
                    "native_peak_rss_bytes": 2000 * delay,
                },
            )
            save(directory / "observer-end.json", {"export_end_ns": start + 500})
            stdout_path.write_text(render_answer(record) + "\n")
            stderr_path.write_text("")
            self.first_stdout_ns = start + delay * 100

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
                "first_stdout_ns": self.first_stdout_ns,
                "first_failure": None,
                "cleanup_failures": [],
            }

    monkeypatch.setattr(app, "check_plan", lambda value, **kwargs: value)
    monkeypatch.setattr(app, "collection_identity", lambda *args: {"fixture": True})
    monkeypatch.setattr(app.common.timing, "TimingGuard", PipelineGuard)
    monkeypatch.setattr(app.common.timing, "validate_evidence", lambda *args: True)
    monkeypatch.setattr(
        app.public, "public_timing", lambda value, passed: {"valid": passed}
    )
    monkeypatch.setattr(app.common.host, "collect", lambda: {"fixture": True})
    monkeypatch.setattr(app.common.host, "admit", lambda *args: [])
    monkeypatch.setattr(app.common.host, "interval_violations", lambda *args: [])
    run = app.run_collection(plan_path, tmp_path / "run", owner_factory=Owner)
    assert run["state"] == "completed", run.get("failure")
    report = app.make_report(plan_path, tmp_path / "run/run.json")
    assert report["successes"] == 88 and report["protocol_complete"]
    comparison = report["comparison"]
    assert comparison["all_questions"]["questions"] == 40
    assert comparison["all_questions"]["kleidiai_over_cpu_total"] > 1
    assert comparison["all_questions"]["prompt_tokens_per_second"] == {
        "cpu": 2.0,
        "kleidiai": pytest.approx(2 / 0.7),
    }
    assert comparison["similar_length"]["questions"] == 40
    assert {k: v["questions"] for k, v in comparison["by_first_backend"].items()} == {
        "cpu": 20,
        "kleidiai": 20,
    }
    assert comparison["reviewed_answer_matches"] == {"cpu": 40, "kleidiai": 39}
    assert comparison["native_peak_rss_bytes"]["kleidiai"]["median"] == 4000
    assert report["quality_review"] == "pending"
