import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pilot = load("desktop_setup_test", ROOT / "scripts/benchmark_setup.py")
host_tests = load("desktop_host_fixtures", ROOT / "tests/test_benchmark_host.py")
timing_tests = load("desktop_timing_fixtures", ROOT / "tests/test_benchmark_timing.py")
benchmark_tests = load("setup_native_fixtures", ROOT / "tests/test_benchmark.py")
REAL_GUARD = pilot.timing.TimingGuard


@pytest.fixture
def fake(tmp_path, monkeypatch):
    _, _, root = benchmark_tests.frozen.__wrapped__(tmp_path)
    output = root / "output"
    supplied = pilot.bench.read_json(root / "spec.json")
    commands = root / "compile_commands.json"
    commands.write_text("[]\n")
    for build in supplied["backends"].values():
        path = Path(build["build_record"])
        record = pilot.bench.read_json(path)
        record["compile_commands_sha256"] = pilot.sha(commands)
        pilot.bench.save_json(path, record)
    method = root / "method.json"
    pilot.bench.save_json(
        method,
        {
            "study": "desktop",
            "study_id": "synthetic-desktop-setup",
            "setup": {
                "runs": pilot.PLANNED_RUNS,
                "fresh_observations_required": 4,
                "timeout_seconds": 1200,
                "output_limit_bytes": 16 << 20,
                "swap_policy": "record_counter_increases",
                "timings_in_final_comparison": False,
                "measured_deadline_rule": pilot.DEADLINE_RULE,
            },
            "measurement": {
                "idle_before_session_seconds": 120,
                "idle_between_pairs_seconds": 120,
            },
        },
    )
    supplied.update(study="desktop", method=str(method))
    spec_path, plan_path = root / "setup-spec.json", root / "setup-plan.json"
    pilot.bench.save_json(spec_path, supplied)
    plan_sha256 = pilot.freeze(spec_path, plan_path)
    plan = pilot.bench.read_json(plan_path)
    model = Path(plan["model"]["path"])
    monkeypatch.setattr(pilot.platform, "system", lambda: "Darwin")
    snapshot = host_tests.snapshot()
    calls = []
    clock = timing_tests.Clock()
    guards, assertions = [], []

    def guard_factory(policy, identity, *, previous_checkpoint):
        assertion = timing_tests.Child()

        def query(argv, **kwargs):
            raw = (
                timing_tests.BOOT
                if argv[0] == "/usr/sbin/sysctl"
                else (b"pid 321(caffeinate): PreventUserIdleSystemSleep\n")
            )
            return SimpleNamespace(returncode=0, stdout=raw, stderr=b"")

        guard = REAL_GUARD(
            policy,
            identity,
            previous_checkpoint=previous_checkpoint,
            clock=clock,
            platform="darwin",
            popen=lambda *a, **kw: assertion,
            run=query,
            sleep=clock.sleep,
            getpid=lambda: 765,
        )
        guards.append(guard)
        assertions.append(assertion)
        return guard

    monkeypatch.setattr(pilot.timing, "TimingGuard", guard_factory)
    monkeypatch.setattr(pilot.host, "collect", lambda: copy.deepcopy(snapshot))
    monkeypatch.setattr(
        pilot,
        "MeasuredChild",
        lambda *args, **kwargs: SimpleNamespace(
            usage=SimpleNamespace(
                ru_maxrss=123456,
                ru_utime=1.0,
                ru_stime=0.2,
                ru_minflt=3,
                ru_majflt=0,
                ru_nswap=0,
            )
        ),
    )

    def capture(argv, env, directory, deadline, limit, started, *, popen, guard, label):
        calls.append(argv)
        assert deadline == 1200 and limit == 16 << 20
        assert env == {"PATH": pilot.os.defpath, "LANG": "C", "LC_ALL": "C"}
        start = guard.check(label + ":start")
        popen()
        started(123)
        clock.awake += 10_000_000_000
        end = guard.check(label + ":end")
        native = copy.deepcopy(benchmark_tests.SAMPLE[0])

        def option(name):
            return argv[argv.index(name) + 1]

        native.update(
            model_filename=option("-m"),
            n_prompt=int(option("-p")),
            n_gen=int(option("-n")),
            n_depth=int(option("-d")),
            samples_ns=[1_000_000_000] * 4,
            avg_ns=1_000_000_000,
        )
        native["samples_ts"] = [float(native["n_prompt"] + native["n_gen"])] * 4
        pilot.bench.save_json(directory / "stdout.json", [native])
        (directory / "stderr.log").write_text("raw log")
        return {
            "outcome": "captured",
            "returncode": 0,
            "output_incomplete": False,
            "cleanup_clock": "not_needed",
            "wall_seconds": 10.0,
            "timing": pilot.bench.row_timing(start, end, timeout_seconds=1200),
        }, None

    monkeypatch.setattr(pilot, "capture", capture)
    return SimpleNamespace(
        root=root,
        output=output,
        model=model,
        method=method,
        spec_path=spec_path,
        plan=plan,
        plan_path=plan_path,
        plan_sha256=plan_sha256,
        snapshot=snapshot,
        calls=calls,
        clock=clock,
        guards=guards,
        assertions=assertions,
        run=lambda backend="cpu", workload="pp2048": pilot.run(
            plan_path,
            plan_sha256,
            backend,
            workload,
            output=output,
            lock_path=root / "test.lock",
        ),
    )


def ledger(fake):
    return pilot.bench.read_json(fake.output / "setup-ledger.json")


def rows(plan):
    values = []
    for workload, shape in pilot.WORKLOADS.items():
        for backend, seconds in [("cpu", 200.1), ("kleidiai", 340.2)]:
            guard, clock, _, _ = timing_tests.make_guard()
            guard.evidence["host_identity_sha256"] = pilot.timing.host_identity_hash(
                host_tests.snapshot()["identity"]
            )
            guard.start()
            key = f"{backend}-{workload}"
            start = guard.check(key + ":start")
            clock.awake += round(seconds * 1e9)
            end = guard.check(key + ":end")
            guard.close()
            values.append(
                dict(
                    state="completed",
                    returncode=0,
                    output_incomplete=False,
                    cleanup_clock="not_needed",
                    backend=backend,
                    workload=workload,
                    shape=list(shape),
                    eligible_for_deadline_calibration=True,
                    host_violations=[],
                    wall_seconds=seconds,
                    study="desktop",
                    study_id=plan["study_id"],
                    method_sha256=plan["method"]["sha256"],
                    timing_valid=True,
                    timing_policy=copy.deepcopy(pilot.timing.POLICY),
                    timing_helper_sha256=plan["timing_helper_sha256"],
                    timing_guard=guard.evidence,
                    host_before=host_tests.snapshot(),
                    native={"samples_ns": [1] * 4},
                    timing=pilot.bench.row_timing(start, end, timeout_seconds=1200),
                )
            )
    return values


def test_freeze_uses_public_helpers_and_create_only_inputs(fake):
    assert set(fake.plan["controller_sources"]) == {
        "scripts/benchmark_setup.py",
        "scripts/benchmark.py",
        "scripts/runtime.py",
        "scripts/benchmark_host.py",
        "scripts/benchmark_timing.py",
    }
    assert fake.plan["method"]["sha256"] == pilot.sha(fake.method)
    with pytest.raises(FileExistsError):
        pilot.freeze(fake.spec_path, fake.plan_path)


def test_four_runs_preserve_commands_resources_archive_and_method(fake):
    for key in pilot.PLANNED_RUNS:
        backend, workload = key.split("-", 1)
        row = fake.run(backend, workload)
        assert row["state"] == "completed" and row["eligible_for_deadline_calibration"]
        assert (
            row["study"] == "desktop"
            and row["method_sha256"] == fake.plan["method"]["sha256"]
        )
        assert row["resources"]["peak_resident_bytes"] == 123456
        assert row["host_findings"] == row["host_violations"] == []
        for flag, value in zip(("-p", "-n", "-d", "-r"), pilot.WORKLOADS[workload]):
            assert row["argv"][row["argv"].index(flag) + 1] == str(value)
        assert (
            row["artifacts"]["config/method.json"]["sha256"]
            == fake.plan["method"]["sha256"]
        )
        for name in (
            "benchmark_setup.py",
            "benchmark.py",
            "benchmark_host.py",
            "benchmark_timing.py",
            "runtime.py",
        ):
            assert f"config/{name}" in row["artifacts"]
    result = ledger(fake)
    assert (
        result["study"] == "desktop"
        and result["method_sha256"] == fake.plan["method"]["sha256"]
    )
    assert not result["included_in_final_timing_results"]
    assert result["planned_runs"] == pilot.PLANNED_RUNS
    assert result["host_policy"]["swap_activity_policy"] == "record_counter_increases"
    assert pilot.suggested_deadline(fake.plan, list(result["runs"].values())) == 600


def export(fake, destination="candidate.json"):
    return pilot.report(
        fake.plan_path,
        fake.plan_sha256,
        fake.output,
        pilot.sha(fake.output / "setup-ledger.json"),
        fake.root / destination,
    )


def test_report_calibrates_all_four_but_requires_real_review(fake):
    for key in pilot.PLANNED_RUNS:
        fake.run(*key.split("-", 1))
    candidate = export(fake)
    assert candidate["status"] == "pending_review"
    assert candidate["timeout_seconds"] == 600
    assert (
        candidate["deadline_rule"]
        == pilot.bench.read_json(fake.method)["setup"]["measured_deadline_rule"]
    )
    assert candidate["review_required"]
    assert candidate["model_sha256"] == fake.plan["model"]["sha256"]
    assert len(candidate["planned_outcomes"]) == 4
    assert not candidate["included_in_final_timing_results"]
    with pytest.raises(FileExistsError):
        export(fake)


def test_partial_setup_has_no_deadline_or_approval(fake):
    fake.run()
    candidate = export(fake)
    assert candidate["status"] == "incomplete" and candidate["timeout_seconds"] is None
    assert [row["state"] for row in candidate["planned_outcomes"]] == [
        "completed",
        "unattempted",
        "unattempted",
        "unattempted",
    ]


@pytest.mark.parametrize("part", ["ledger", "record", "raw", "method", "plan"])
def test_report_rejects_changed_evidence(fake, part):
    for key in pilot.PLANNED_RUNS:
        fake.run(*key.split("-", 1))
    ledger_sha = pilot.sha(fake.output / "setup-ledger.json")
    path = {
        "ledger": fake.output / "setup-ledger.json",
        "record": fake.output / "cpu-pp2048/run.json",
        "raw": fake.output / "cpu-pp2048/stdout.json",
        "method": fake.method,
        "plan": fake.plan_path,
    }[part]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed|mismatch"):
        pilot.report(
            fake.plan_path,
            fake.plan_sha256,
            fake.output,
            ledger_sha,
            fake.root / "rejected.json",
        )
    assert not (fake.root / "rejected.json").exists()


@pytest.mark.parametrize(
    "part", ["raw", "record", "archive", "method", "plan", "ledger", "source", "extra"]
)
def test_report_rejects_inputs_changed_after_their_first_validation(
    fake, monkeypatch, part
):
    for key in pilot.PLANNED_RUNS:
        fake.run(*key.split("-", 1))
    if part == "source":
        source_root = fake.root / "source-checkout"
        sources = []
        for source in pilot.SOURCE_PATHS:
            copied = source_root / source.relative_to(pilot.ROOT)
            copied.parent.mkdir(parents=True, exist_ok=True)
            copied.write_bytes(source.read_bytes())
            sources.append(copied)
        monkeypatch.setattr(pilot, "ROOT", source_root)
        monkeypatch.setattr(pilot, "SOURCE_PATHS", tuple(sources))
    first = fake.output / "cpu-pp2048"
    target = {
        "raw": first / "stdout.json",
        "record": first / "run.json",
        "archive": first / "config/benchmark.py",
        "method": fake.method,
        "plan": fake.plan_path,
        "ledger": fake.output / "setup-ledger.json",
        "source": pilot.ROOT / "scripts/benchmark_host.py",
        "extra": first / "unexpected.txt",
    }[part]
    validate = pilot.verify_eligible_row

    def change_earlier_input(plan, ledger, row, previous, directory):
        validate(plan, ledger, row, previous, directory)
        if directory.name == "kleidiai-tg128-d2048":
            if part == "extra":
                target.write_text("changed inventory")
            else:
                target.write_bytes(target.read_bytes() + b"\n")

    monkeypatch.setattr(pilot, "verify_eligible_row", change_earlier_input)
    with pytest.raises(ValueError, match="changed during export"):
        export(fake, "rejected.json")
    assert not (fake.root / "rejected.json").exists()


def test_report_keeps_the_observation_inventory_immutable(fake):
    fake.run()
    with pytest.raises(ValueError, match="outside setup observation directories"):
        export(fake, "output/cpu-pp2048/candidate.json")
    assert not (fake.output / "cpu-pp2048/candidate.json").exists()


@pytest.mark.parametrize("change", ["settings", "method", "order", "build"])
def test_freeze_rejects_unsupported_or_unverified_inputs(fake, change):
    supplied = pilot.bench.read_json(fake.spec_path)
    if change in {"method", "order"}:
        method = pilot.bench.read_json(fake.method)
        if change == "method":
            method["setup"]["timeout_seconds"] = 2400
        else:
            method["setup"]["runs"] = list(reversed(pilot.PLANNED_RUNS))
        pilot.bench.save_json(fake.method, method)
    elif change == "build":
        path = Path(supplied["backends"]["cpu"]["build_record"])
        record = pilot.bench.read_json(path)
        record["compile_commands_sha256"] = "0" * 64
        pilot.bench.save_json(path, record)
    else:
        supplied["expected_model"]["model_size"] = True
        pilot.bench.save_json(fake.spec_path, supplied)
    with pytest.raises(ValueError):
        pilot.freeze(fake.spec_path, fake.root / "rejected.json")
    assert not (fake.root / "rejected.json").exists()


@pytest.mark.parametrize("phase", ["freeze", "run"])
@pytest.mark.parametrize("rule", [None, "ceil(longest continuous seconds)"])
def test_calibration_rule_must_be_declared_and_supported_before_launch(
    fake, phase, rule
):
    method = pilot.bench.read_json(fake.method)
    if rule is None:
        del method["setup"]["measured_deadline_rule"]
    else:
        method["setup"]["measured_deadline_rule"] = rule
    pilot.bench.save_json(fake.method, method)
    with pytest.raises(ValueError, match="supported four-run setup"):
        if phase == "freeze":
            pilot.freeze(fake.spec_path, fake.root / "rejected.json")
        else:
            plan = copy.deepcopy(fake.plan)
            plan["method"] = pilot.bench.asset(fake.method)
            pilot.bench.save_json(fake.plan_path, plan)
            pilot.run(
                fake.plan_path,
                pilot.sha(fake.plan_path),
                "cpu",
                "pp2048",
                output=fake.output,
                lock_path=fake.root / "test.lock",
            )
    assert not fake.calls
    assert not fake.output.exists()
    assert not (fake.root / "rejected.json").exists()


def test_failed_setup_report_preserves_attempt_and_no_calibration(fake):
    fake.snapshot["memory_pressure"]["value"] = {"code": 2, "label": "warning"}
    fake.run()
    candidate = export(fake)
    assert candidate["status"] == "incomplete"
    assert not candidate["planned_outcomes"][0]["attempted"]
    assert candidate["planned_outcomes"][0]["state"] == "failed"


def test_frozen_json_digest_authenticates_the_bytes_that_are_parsed(tmp_path):
    path = tmp_path / "pin.json"
    path.write_text('{"a": 1, "a": 2}')
    with pytest.raises(ValueError, match="Duplicate"):
        pilot.pinned_json(path, pilot.sha(path))
    path.write_text(json.dumps({"a": True}))
    assert pilot.pinned_json(path, pilot.sha(path)) == {"a": True}


@pytest.mark.parametrize("counter", ["Swapins", "Swapouts"])
def test_swap_alone_retained_without_exclusion(fake, monkeypatch, counter):
    after = copy.deepcopy(fake.snapshot)
    after["pages"]["value"][counter] += 999
    reads = iter([fake.snapshot, fake.snapshot, after])
    monkeypatch.setattr(pilot.host, "collect", lambda: next(reads))
    row = fake.run()
    assert row["host_findings"] == ["swap_activity"]
    assert row["host_violations"] == [] and row["eligible_for_deadline_calibration"]
    assert row["host_deltas"]["pages"]["value"][counter] == 999


@pytest.mark.parametrize("change", ["pressure", "reset", "missing"])
def test_other_departure_retains_completed_observation_and_stops_setup(
    fake, monkeypatch, change
):
    after = copy.deepcopy(fake.snapshot)
    if change == "pressure":
        after["memory_pressure"]["value"] = {"code": 2, "label": "warning"}
    elif change == "reset":
        after["pages"]["value"]["Swapins"] = 0
    else:
        after["pages"] = pilot.host.unavailable("missing")
    reads = iter([fake.snapshot, fake.snapshot, after])
    monkeypatch.setattr(pilot.host, "collect", lambda: next(reads))
    row = fake.run()
    assert row["state"] == "completed" and not row["eligible_for_deadline_calibration"]
    assert row["host_findings"] and row["host_violations"]
    with pytest.raises(ValueError, match="Setup stopped"):
        fake.run("kleidiai")
    assert len(fake.calls) == 1


@pytest.mark.parametrize("failure", ["native", "timeout", "model_changed"])
def test_failure_retains_raw_output_and_prevents_partial_continuation(
    fake, monkeypatch, failure
):
    if failure == "native":

        def invalid(*args):
            raise ValueError("invalid native JSON")

        monkeypatch.setattr(pilot.bench, "validate_samples", invalid)
    else:
        capture = pilot.capture

        def changed(*args, **kwargs):
            result, interruption = capture(*args, **kwargs)
            if failure == "timeout":
                result["outcome"] = "timeout"
            else:
                fake.model.write_text("changed")
            return result, interruption

        monkeypatch.setattr(pilot, "capture", changed)
    row = fake.run()
    assert row["state"] == "failed" and not row["eligible_for_deadline_calibration"]
    assert row["artifacts"]["stdout.json"]["bytes"] > 10
    assert row["artifacts"]["stderr.log"]["bytes"] == 7
    assert row["resources"]["peak_resident_bytes"] == 123456
    with pytest.raises(ValueError, match="Setup stopped"):
        fake.run("kleidiai")
    assert len(fake.calls) == 1


def test_create_only_and_frozen_order(fake):
    with pytest.raises(ValueError, match="order"):
        fake.run("kleidiai")
    assert not fake.calls
    fake.run()
    with pytest.raises(ValueError, match="Preserve"):
        fake.run()
    with pytest.raises(ValueError, match="order"):
        fake.run("cpu", "tg128-d2048")
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    "field,value",
    [("study", "controlled"), ("study_id", "other"), ("method_sha256", "0" * 64)],
)
def test_existing_strict_or_other_method_ledger_cannot_be_reused(fake, field, value):
    fake.run()
    result = ledger(fake)
    result[field] = value
    pilot.bench.save_json(fake.output / "setup-ledger.json", result)
    with pytest.raises(ValueError, match="not bound"):
        fake.run("kleidiai")
    assert len(fake.calls) == 1


def test_method_change_rejected_before_launch(fake):
    fake.method.write_bytes(fake.method.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed|mismatch"):
        fake.run()
    assert not fake.calls


def test_changed_runtime_build_rejected_and_retained(fake):
    Path(fake.plan["backends"]["kleidiai"]["build_record"]["path"]).write_text("{}")
    row = fake.run()
    assert row["state"] == "failed" and row["outcome"] == "preflight_error"
    assert "mismatch" in row["failure"] and not fake.calls


def test_loaded_helper_change_rejected_before_launch(fake, monkeypatch):
    changed = dict(pilot.LOADED_SOURCES)
    changed["scripts/benchmark_host.py"] = "0" * 64
    monkeypatch.setattr(pilot, "LOADED_SOURCES", changed)
    with pytest.raises(ValueError, match="helpers"):
        fake.run()
    assert not fake.calls


@pytest.mark.parametrize(
    "change",
    [
        {"study": "controlled"},
        {"study_id": "other"},
        {"method_sha256": "0" * 64},
        {"eligible_for_deadline_calibration": False},
        {"backend": "cpu"},
        {"shape": [0, 128, 512, 4]},
        {"state": "failed"},
    ],
)
def test_deadline_needs_all_four_eligible_method_bound_desktop_rows(fake, change):
    values = rows(fake.plan)
    assert pilot.suggested_deadline(fake.plan, values) == 720
    values[1].update(change)
    with pytest.raises(ValueError, match="All four"):
        pilot.suggested_deadline(fake.plan, values)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 0, True, None])
def test_each_wall_duration_must_be_finite_positive_numeric(fake, value):
    values = rows(fake.plan)
    values[1]["wall_seconds"] = value
    with pytest.raises(ValueError):
        pilot.suggested_deadline(fake.plan, values)


def test_no_partial_calibration(fake):
    with pytest.raises(ValueError, match="All four"):
        pilot.suggested_deadline(fake.plan, rows(fake.plan)[:3])


@pytest.mark.parametrize("change", ["reset", "missing", "size"])
def test_gap_counter_problem_stops_before_second_child(fake, monkeypatch, change):
    fake.run()
    current = copy.deepcopy(fake.snapshot)
    if change == "reset":
        current["pages"]["value"]["Swapins"] = 0
    elif change == "size":
        current["pages"]["value"]["page_size_bytes"] *= 2
    else:
        result = ledger(fake)
        del result["runs"]["cpu-pp2048"]["host_after"]
        pilot.bench.save_json(fake.output / "setup-ledger.json", result)
    monkeypatch.setattr(pilot.host, "collect", lambda: copy.deepcopy(current))
    row = fake.run("kleidiai")
    assert row["state"] == "failed" and row["outcome"] == "preflight_error"
    if change == "missing":
        assert "ledger differs" in row["failure"]
    else:
        assert row["host_gap"]["violations"] == ["swap_counters_unverified"]
        assert row["preflight_host_findings"] == ["swap_counters_unverified"]
        assert row["host_gap"]["deltas"]["pages"]["status"] == "unavailable"
    assert len(fake.calls) == 1
    with pytest.raises(ValueError, match="Setup stopped"):
        fake.run("cpu", "tg128-d2048")


def test_positive_swap_in_gap_remains_visible_and_eligible(fake, monkeypatch):
    fake.run()
    current = copy.deepcopy(fake.snapshot)
    current["pages"]["value"]["Swapins"] += 13
    monkeypatch.setattr(pilot.host, "collect", lambda: copy.deepcopy(current))
    row = fake.run("kleidiai")
    assert row["state"] == "completed" and row["eligible_for_deadline_calibration"]
    assert row["host_gap"]["deltas"]["pages"]["value"]["Swapins"] == 13
    assert row["host_gap"]["findings"] == ["swap_activity"]
    assert row["host_findings"] == ["swap_activity"]
    assert row["host_violations"] == [] and len(fake.calls) == 2


def test_first_host_rejection_is_retained_without_child(fake):
    fake.snapshot["memory_pressure"]["value"] = {"code": 2, "label": "warning"}
    row = fake.run()
    assert row["state"] == "failed" and not fake.calls
    assert row["host_before"]["memory_pressure"]["value"]["code"] == 2
    assert row["preflight_host_findings"] == ["host_baseline_unadmitted"]


def test_frozen_power_policy_rejects_later_change_without_child(fake):
    fake.run()
    fake.snapshot["power_source"]["value"] = "Battery Power"
    row = fake.run("kleidiai")
    assert row["state"] == "failed" and len(fake.calls) == 1
    assert "power_source_changed" in row["preflight_host_violations"]


def test_deadline_cannot_override_effective_host_failure_with_eligibility_flag(fake):
    values = rows(fake.plan)
    values[1]["host_violations"] = ["swap_counters_unverified"]
    with pytest.raises(ValueError, match="All four"):
        pilot.suggested_deadline(fake.plan, values)


@pytest.mark.parametrize("contents", [b"{}", b"not JSON"])
def test_changed_or_malformed_build_manifest_rejected_before_capture(fake, contents):
    Path(fake.plan["backends"]["cpu"]["build_record"]["path"]).write_bytes(contents)
    row = fake.run()
    assert row["state"] == "failed" and row["outcome"] == "preflight_error"
    assert "mismatch" in row["failure"]
    assert not fake.calls


@pytest.mark.parametrize("part", ["record", "stdout", "archive", "extra", "checkpoint"])
def test_previous_attempt_must_match_immutable_records_and_assets(fake, part):
    fake.run()
    directory = fake.output / "cpu-pp2048"
    if part == "record":
        (directory / "run.json").write_text("{}")
    elif part == "stdout":
        (directory / "stdout.json").write_text("modified raw output")
    elif part == "archive":
        (directory / "config/benchmark_timing.py").write_text("changed")
    elif part == "extra":
        (directory / "unexpected.txt").write_text("extra")
    else:
        record = ledger(fake)
        record["runs"]["cpu-pp2048"]["timing_guard"]["exit_checkpoint"]["sample"][
            "awake_ns"
        ] += 10
        pilot.bench.save_json(fake.output / "setup-ledger.json", record)
    row = fake.run("kleidiai")
    assert row["state"] == "failed" and not row["eligible_for_deadline_calibration"]
    assert len(fake.calls) == 1
    assert all(
        guard.process is None or guard.process.returncode is not None
        for guard in fake.guards
    )


def test_suspend_between_setup_processes_rejects_before_second_child(fake):
    fake.run()
    fake.clock.suspended += 60_000_000_000
    row = fake.run("kleidiai")
    assert row["state"] == "failed" and len(fake.calls) == 1
    assert row["timing_guard"]["first_failure"]["reason"] == "clock_discontinuity"
    assert row["timing_guard"]["exit_checkpoint"] is None


def test_fresh_guard_does_not_reset_previous_checkpoint(fake):
    first = fake.run()
    fake.clock.awake += 900_000_000_000
    second = fake.run("kleidiai")
    assert second["eligible_for_deadline_calibration"]
    assert (
        second["timing_guard"]["previous_checkpoint"]
        == first["timing_guard"]["exit_checkpoint"]
    )
    assert second["previous_record_sha256"] == first["record_sha256"]
    assert second["wall_seconds"] == 10


@pytest.mark.parametrize(
    "phase",
    ["fingerprint_before", "host_after", "fingerprint_after", "archive", "close"],
)
def test_sleep_after_start_or_success_revokes_eligibility(fake, monkeypatch, phase):
    if phase.startswith("fingerprint"):
        calls = 0

        def fingerprint(*args):
            nonlocal calls
            calls += 1
            if calls == (1 if phase.endswith("before") else 2):
                fake.clock.suspended += 1_000_000_000
            return {"stable": "pin"}

        monkeypatch.setattr(pilot, "fingerprint", fingerprint)
    elif phase == "host_after":
        calls = 0

        def collect():
            nonlocal calls
            calls += 1
            if calls == 3:
                fake.clock.suspended += 1_000_000_000
            return copy.deepcopy(fake.snapshot)

        monkeypatch.setattr(pilot.host, "collect", collect)
    elif phase == "archive":
        original = pilot.sha

        def digest(path):
            if path.name == "stdout.json":
                fake.clock.suspended += 1_000_000_000
            return original(path)

        monkeypatch.setattr(pilot, "sha", digest)
    else:
        original = REAL_GUARD.close

        def close(guard):
            fake.clock.suspended += 1_000_000_000
            return original(guard)

        monkeypatch.setattr(REAL_GUARD, "close", close)
    row = fake.run()
    assert not row["eligible_for_deadline_calibration"] and row["state"] == "failed"
    assert row["timing_guard"]["first_failure"]["reason"] == "clock_discontinuity"
    assert fake.assertions[0].returncode is not None
    assert len(fake.calls) == (0 if phase == "fingerprint_before" else 1)


def test_native_sample_sum_cannot_exceed_continuous_child_time(fake, monkeypatch):
    monkeypatch.setattr(
        pilot.bench,
        "validate_samples",
        lambda *a: {"samples_ns": [3_000_000_000] * 4},
    )
    row = fake.run()
    assert row["state"] == "failed" and not row["eligible_for_deadline_calibration"]
    assert (
        row["timing_guard"]["first_failure"]["reason"] == "native_duration_inconsistent"
    )


def test_interrupt_in_evidence_finalization_still_releases_assertion_and_persists(
    fake, monkeypatch
):
    original = pilot.sha

    def digest(path):
        if path.name == "stdout.json":
            raise KeyboardInterrupt
        return original(path)

    monkeypatch.setattr(pilot, "sha", digest)
    with pytest.raises(KeyboardInterrupt):
        fake.run()
    row = ledger(fake)["runs"]["cpu-pp2048"]
    assert row["state"] == "failed" and not row["eligible_for_deadline_calibration"]
    assert fake.assertions[0].returncode is not None
    assert (fake.output / "cpu-pp2048/run.json").is_file()


@pytest.mark.parametrize(
    "change", ["guard", "measurement", "helper", "policy", "native"]
)
def test_deadline_revalidates_guard_and_measurement_not_only_eligibility(fake, change):
    values = rows(fake.plan)
    row = values[0]
    if change == "guard":
        row["timing_guard"]["assertion"]["released"] = False
    elif change == "measurement":
        row["timing"]["end"]["continuous_after_ns"] += 1
    elif change == "helper":
        row["timing_helper_sha256"] = "0" * 64
    elif change == "policy":
        row["timing_policy"]["allowance_ns"] *= 2
    else:
        row["native"]["samples_ns"] = [1_000_000_000_000] * 4
    with pytest.raises(ValueError, match="All four"):
        pilot.suggested_deadline(fake.plan, values)


REAL_CAPTURE = load(
    "direct_guarded_capture", ROOT / "scripts/benchmark_setup.py"
).capture


@pytest.fixture
def capture_fixture(fake, monkeypatch, tmp_path):
    module = REAL_CAPTURE.__globals__
    clock = fake.clock
    guard = pilot.timing.TimingGuard(
        pilot.timing.POLICY,
        pilot.timing.host_identity_hash(fake.snapshot["identity"]),
        previous_checkpoint=None,
    )
    guard.start()
    module["timing"] = pilot.timing
    original_selector = pilot.selectors.DefaultSelector
    state = SimpleNamespace(
        callback=None,
        selections=0,
        output=b"raw output",
        error=b"raw error",
        code=0,
        finish=clock.awake + 200_000_000,
        resist=False,
        ignore_kill=False,
        children=[],
    )

    class Selector:
        def __init__(self):
            self.inner = original_selector()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.inner.close()

        def register(self, *args):
            return self.inner.register(*args)

        def unregister(self, *args):
            return self.inner.unregister(*args)

        def get_map(self):
            return self.inner.get_map()

        def select(self, timeout):
            state.selections += 1
            clock.awake += 100_000_000
            if state.callback:
                state.callback(state.selections)
            return self.inner.select(0)

    monkeypatch.setattr(pilot.selectors, "DefaultSelector", Selector)

    def popen(argv, **kwargs):
        child = timing_tests.Child(
            clock=clock,
            finish_ns=state.finish,
            resist=state.resist,
            ignore_kill=state.ignore_kill,
        )
        poll = child.poll

        def observed():
            result = poll()
            if result == 0:
                child.returncode = state.code
            return child.returncode

        child.poll = observed
        for name, content in [("stdout", state.output), ("stderr", state.error)]:
            read, write = pilot.os.pipe()
            pilot.os.write(write, content)
            pilot.os.close(write)
            setattr(child, name, pilot.os.fdopen(read, "rb"))
        state.children.append(child)
        return child

    def run(limit=1024, deadline=1200):
        directory = tmp_path / "capture"
        directory.mkdir()
        try:
            return REAL_CAPTURE(
                ["never launched"],
                {},
                directory,
                deadline,
                limit,
                lambda pid: None,
                guard=guard,
                label="cpu-pp2048",
                popen=popen,
            )
        finally:
            guard.close()

    state.run, state.guard, state.clock = run, guard, clock
    state.directory = tmp_path / "capture"
    yield state
    guard.close()


def test_capture_normal_output_and_wait4_compatible_child(capture_fixture):
    state = capture_fixture
    result, interrupted = state.run()
    assert interrupted is None and result["outcome"] == "captured"
    assert result["bytes_observed"] == result["bytes_retained"] == 19
    assert result["timing"]["valid"] and result["wall_seconds"] == 0.2
    assert not result["output_incomplete"] and result["cleanup_clock"] == "not_needed"
    assert state.children[0].returncode == 0
    assert pilot.timing.validate_evidence(
        state.guard.evidence,
        pilot.timing.POLICY,
        state.guard.evidence["host_identity_sha256"],
    )


def test_capture_caps_retained_output_and_stops_owned_child(capture_fixture):
    state = capture_fixture
    state.finish = None
    result, _ = state.run(limit=4)
    assert result["outcome"] == "output_limit" and result["bytes_retained"] == 4
    assert result["bytes_observed"] > 4 and result["output_incomplete"]
    assert state.children[0].terminated == 1
    assert sum(p.stat().st_size for p in state.directory.iterdir()) == 4


@pytest.mark.parametrize("problem", ["sleep", "deadline", "assertion", "interrupt"])
def test_capture_checks_after_selector_wakes_and_retains_prior_output(
    capture_fixture, problem
):
    state = capture_fixture
    state.finish = None

    def disturb(count):
        if count != 2:
            return
        if problem == "sleep":
            state.clock.suspended += 60_000_000_000
        elif problem == "deadline":
            state.clock.awake += 1_201_000_000_000
            state.children[0].returncode = 0
        elif problem == "assertion":
            state.guard.process.returncode = 1
        else:
            raise KeyboardInterrupt

    state.callback = disturb
    result, interrupted = state.run()
    assert (
        result["outcome"]
        == {
            "sleep": "timing_error",
            "deadline": "timeout",
            "assertion": "timing_error",
            "interrupt": "interrupted",
        }[problem]
    )
    assert result["bytes_retained"] == 19
    assert state.children[0].returncode is not None
    assert bool(interrupted) == (problem == "interrupt")
    assert (state.directory / "stdout.json").read_bytes() == b"raw output"


def test_capture_first_failure_survives_cleanup_failure(capture_fixture):
    state = capture_fixture
    state.finish = None
    state.resist = state.ignore_kill = True
    result, _ = state.run(limit=4)
    assert result["outcome"] == "output_limit" and "OutputLimit" in result["failure"]
    assert result["cleanup_error"] == "reap_failed"
    assert state.guard.evidence["cleanup_errors"][0] == {
        "target": "native",
        "reason": "reap_failed",
    }
    assert not pilot.timing.validate_evidence(
        state.guard.evidence,
        pilot.timing.POLICY,
        state.guard.evidence["host_identity_sha256"],
    )


def test_capture_nonzero_native_exit_is_preserved(capture_fixture):
    state = capture_fixture
    state.code = 7
    result, _ = state.run()
    assert result["outcome"] == "exit_error" and result["returncode"] == 7


def test_measured_child_reaps_specific_pid_and_retains_usage(monkeypatch):
    process = SimpleNamespace(pid=47, stdout=None, stderr=None, returncode=None)
    usage = object()
    calls = []
    monkeypatch.setattr(pilot.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(
        pilot.os,
        "wait4",
        lambda pid, flags: calls.append((pid, flags)) or (47, 0, usage),
    )
    child = pilot.MeasuredChild(["fake"])
    assert child.poll() == child.poll() == 0
    assert calls == [(47, pilot.os.WNOHANG)] and child.usage is usage


def test_reordered_ledger_is_rejected_before_launch(fake):
    fake.run()
    fake.run("kleidiai")
    record = ledger(fake)
    record["runs"] = dict(reversed(list(record["runs"].items())))
    pilot.bench.save_json(fake.output / "setup-ledger.json", record)
    with pytest.raises(ValueError, match="Setup stopped"):
        fake.run("cpu", "tg128-d2048")
    assert len(fake.calls) == 2


def test_host_identity_change_during_verification_prevents_launch(fake, monkeypatch):
    reads = 0

    def collect():
        nonlocal reads
        reads += 1
        value = copy.deepcopy(fake.snapshot)
        if reads > 1:
            value["identity"]["release"] = "different"
        return value

    monkeypatch.setattr(pilot.host, "collect", collect)
    row = fake.run()
    assert not fake.calls and row["state"] == "failed"
    assert "Host identity changed" in row["failure"]


def test_closed_guard_does_not_admit_broken_measurement_record(fake, monkeypatch):
    original = pilot.capture

    def capture(*args, **kwargs):
        result, interrupted = original(*args, **kwargs)
        result["timing"]["continuous_seconds"] += 1
        return result, interrupted

    monkeypatch.setattr(pilot, "capture", capture)
    row = fake.run()
    assert row["timing_valid"] and not row["eligible_for_deadline_calibration"]


def test_capture_rechecks_clock_when_exit_zero_becomes_visible(capture_fixture):
    state = capture_fixture
    state.finish = None

    def disturb(count):
        if count == 2:
            child = state.children[0]

            def late_exit():
                state.clock.awake += 1_201_000_000_000
                child.returncode = 0
                return 0

            child.poll = late_exit

    state.callback = disturb
    result, _ = state.run()
    assert result["outcome"] == "timeout" and result["returncode"] == 0
    assert result["timing"]["deadline_exceeded"] and not result["timing"]["valid"]


@pytest.mark.parametrize("secondary", ["fsync", "pipe_close"])
def test_capture_output_finalization_cannot_replace_timeout(
    capture_fixture, monkeypatch, secondary
):
    state = capture_fixture
    state.finish = None

    def disturb(count):
        if count != 2:
            return
        state.clock.awake += 1_201_000_000_000
        child = state.children[0]
        child.returncode = 0
        if secondary == "pipe_close":
            pipe = child.stdout

            class BrokenClose:
                @property
                def closed(self):
                    return pipe.closed

                def close(self):
                    pipe.close()
                    raise OSError("fixture pipe close")

            child.stdout = BrokenClose()

    state.callback = disturb
    if secondary == "fsync":

        def broken(*args):
            raise OSError("fixture fsync")

        monkeypatch.setattr(pilot.os, "fsync", broken)
    result, interrupted = state.run()
    assert interrupted is None and result["outcome"] == "timeout"
    assert result["failure"] == "GuardError: deadline_exceeded"
    assert result["bytes_retained"] == 19 and result["returncode"] == 0
    assert result["timing"]["reason"] == "deadline_exceeded"
    assert result["record_errors"][0]["operation"] == (
        "output_flush" if secondary == "fsync" else "pipe_close"
    )
    assert state.guard.evidence["first_failure"]["reason"] == "deadline_exceeded"


def test_capture_success_is_revoked_when_output_fsync_fails(
    capture_fixture, monkeypatch
):
    def broken(*args):
        raise OSError("fixture fsync")

    monkeypatch.setattr(pilot.os, "fsync", broken)
    result, _ = capture_fixture.run()
    assert result["outcome"] == "record_error" and result["output_incomplete"]
    assert result["returncode"] == 0 and result["bytes_retained"] == 19
