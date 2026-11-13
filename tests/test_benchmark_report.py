import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = load("controlled_benchmark_report", ROOT / "scripts/benchmark_report.py")
fixtures = load("benchmark_report_fixtures", ROOT / "tests/test_benchmark.py")
bench = report.bench
frozen = fixtures.frozen


@pytest.fixture(autouse=True)
def never_execute(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("An offline export must not execute or probe the host.")

    monkeypatch.setattr(bench.subprocess, "Popen", forbidden)
    monkeypatch.setattr(bench.host, "collect", forbidden)


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def finish(frozen, sessions, number, mutate):
    plan = bench.read_json(frozen[0])
    item = sessions[number - 1]
    directory = item["directory"]
    run = bench.read_json(directory / "run.json")
    mutate(run, directory)
    save(directory / "run.json", run)
    item["run_sha256"] = bench.runtime.digest(directory / "run.json")
    save(directory / "summary.json", bench.summarize(plan, run))
    return run


@pytest.fixture
def sessions(frozen):
    return make_sessions(frozen)


@pytest.fixture
def desktop(frozen):
    return fixtures.desktop_plan(frozen)


@pytest.fixture
def desktop_sessions(desktop):
    return make_sessions(desktop)


def make_sessions(frozen):
    path, digest, root = frozen
    plan = bench.read_json(path)
    result = []
    for number in range(1, 4):
        directory = root / f"session-{number}"
        directory.mkdir()
        (directory / "plan.json").write_bytes(path.read_bytes())
        run = {
            "schema_version": 1,
            "session": number,
            "plan_sha256": digest,
            "state": "finished",
            "integrity_verified": True,
            "failure": None,
            "host_violations": [],
            "rows": [],
        }
        for scheduled in plan["schedule"]:
            if scheduled["session"] != number:
                continue
            row = {
                **scheduled,
                "state": "finished",
                "outcome": "success",
                "wall_seconds": 60.0,
                "returncode": 0,
            }
            row["argv"] = bench.command(plan, row)
            row["host_before"] = fixtures.stable_host()
            row["host_after"] = fixtures.stable_host()
            row["host_deltas"] = bench.host.deltas(
                row["host_before"], row["host_after"]
            )
            native = copy.deepcopy(fixtures.SAMPLE[0])
            p, n, d = bench.WORKLOADS[row["workload"]]
            # Deliberately unbalanced nested samples: only the process median matters.
            factor = 2 if row["backend"] == "cpu" else 1
            native.update(
                model_filename=plan["model"]["path"],
                n_prompt=p,
                n_gen=n,
                n_depth=d,
                cpu_info="PRIVATE MACHINE",
                gpu_info="PRIVATE GPU",
            )
            native["samples_ns"] = [
                9000000000,
                1000000000 * factor,
                3000000000 * factor,
                8000000000 * factor,
            ]
            native["avg_ns"] = sum(native["samples_ns"]) // 4
            native["samples_ts"] = [
                1e9 * (p + n) / value for value in native["samples_ns"]
            ]
            row["observed"] = native
            destination = directory / row["id"]
            destination.mkdir()
            save(destination / "stdout.json", [native])
            (destination / "stderr.log").write_text(
                "/Users/private-user/project secret machine stderr\n"
            )
            row["artifacts"] = {
                name: bench.asset(destination / name)
                for name in ("stdout.json", "stderr.log")
            }
            run["rows"].append(row)
        save(directory / "run.json", run)
        save(directory / "summary.json", bench.summarize(plan, run))
        result.append(
            {
                "session": number,
                "directory": directory,
                "run_sha256": bench.runtime.digest(directory / "run.json"),
            }
        )
    return result


def add_desktop_swap(run, directory):
    for index, row in enumerate(run["rows"]):
        row["host_before"]["pages"]["value"]["Swapins"] += index
        row["host_after"]["pages"]["value"]["Swapins"] += index + 1
        row["host_deltas"] = bench.host.deltas(row["host_before"], row["host_after"])


def test_desktop_report_keeps_swap_and_uses_means(desktop, desktop_sessions):
    for number in range(1, 4):
        finish(desktop, desktop_sessions, number, add_desktop_swap)
    value = export(desktop, desktop_sessions)
    assert value["kind"] == "desktop-fixed-token-report"
    assert value["status"] == "complete" and value["protocol_complete"]
    assert value["accounting"]["included_pairs"] == 24
    plan = bench.read_json(desktop[0])
    assert value["provenance"]["method"] == report.public_asset(plan["method"])
    for session in value["sessions"]:
        assert session["host_violations"] == []
        for pair in session["pairs"]:
            assert pair["host_findings"] == ["swap_activity"]
            assert pair["host_protocol_passed"]
            assert pair["mean_ns"] == {"cpu": 8e9, "kleidiai": 4e9}
    for workload in value["workloads"].values():
        assert workload["cpu_over_kleidiai"]["count"] == 6
        assert workload["sessions"][0]["process_mean_ns"]["cpu"]["median"] == 8e9
        assert "process_median_ns" not in workload["sessions"][0]
    report.privacy_check(value)


def test_desktop_final_host_stop_keeps_pair_but_not_completion(
    desktop, desktop_sessions
):
    def violate(run, directory):
        run["state"] = "host_blocked"
        row = run["rows"][-1]
        row["host_after"]["memory_pressure"]["value"] = {"code": 2, "label": "warning"}
        row["host_deltas"] = bench.host.deltas(row["host_before"], row["host_after"])
        run["host_violations"] = report.boundary_violations(
            bench.read_json(desktop[0]), run["rows"]
        )

    finish(desktop, desktop_sessions, 3, violate)
    value = export(desktop, desktop_sessions)
    assert value["accounting_complete"]
    assert value["accounting"]["included_pairs"] == 24
    assert value["status"] == "incomplete" and not value["protocol_complete"]
    pair = value["sessions"][-1]["pairs"][-1]
    assert pair["included"] and not pair["host_protocol_passed"]
    assert pair["host_findings"]


def test_desktop_summary_cannot_hide_swap(desktop, desktop_sessions):
    finish(desktop, desktop_sessions, 1, add_desktop_swap)
    path = desktop_sessions[0]["directory"] / "summary.json"
    summary = bench.read_json(path)
    summary["pairs"][0]["host_findings"] = []
    save(path, summary)
    with pytest.raises(ValueError, match="summary differs"):
        export(desktop, desktop_sessions)


def test_desktop_method_change_rejected(desktop, desktop_sessions):
    plan = bench.read_json(desktop[0])
    Path(plan["method"]["path"]).write_text("changed")
    with pytest.raises(ValueError):
        export(desktop, desktop_sessions)


def test_desktop_counter_reset_between_sides_stops_remaining_work(
    desktop, desktop_sessions
):
    def reset(run, directory):
        truncate(run, directory, 1)
        run.update(state="host_blocked", integrity_verified=True, failure=None)
        first, second = run["rows"][:2]
        first["host_after"]["pages"]["value"]["Swapins"] += 10
        first["host_deltas"] = bench.host.deltas(
            first["host_before"], first["host_after"]
        )
        second["host_before"] = fixtures.stable_host()
        second["host_before"]["pages"]["value"]["Swapins"] += 5
        run["host_violations"] = report.boundary_violations(
            bench.read_json(desktop[0]), run["rows"]
        )

    run = finish(desktop, desktop_sessions, 1, reset)
    assert run["host_violations"][0]["boundary"] == "before"
    assert run["host_violations"][0]["reasons"] == ["swap_counters_unverified"]
    value = export(desktop, desktop_sessions[:1])
    assert not value["protocol_complete"]
    assert value["accounting"]["known_attempted"] == 1
    assert value["accounting"]["known_unattempted"] == 15
    assert value["accounting"]["included_pairs"] == 0


def test_desktop_counter_reset_during_idle_stops_next_pair(desktop, desktop_sessions):
    def reset(run, directory):
        truncate(run, directory, 2)
        run.update(state="host_blocked", integrity_verified=True, failure=None)
        for index, row in enumerate(run["rows"][:2]):
            row["host_before"]["pages"]["value"]["Swapins"] += index * 10
            row["host_after"]["pages"]["value"]["Swapins"] += (index + 1) * 10
            row["host_deltas"] = bench.host.deltas(
                row["host_before"], row["host_after"]
            )
        run["rows"][2]["host_before"] = fixtures.stable_host()
        run["host_violations"] = report.boundary_violations(
            bench.read_json(desktop[0]), run["rows"]
        )

    run = finish(desktop, desktop_sessions, 1, reset)
    assert run["host_violations"][0]["row"] == run["rows"][2]["id"]
    assert run["host_violations"][0]["reasons"] == ["swap_counters_unverified"]
    value = export(desktop, desktop_sessions[:1])
    assert value["accounting"]["included_pairs"] == 1
    assert not value["protocol_complete"]


@pytest.mark.parametrize("boundary", ["host_before", "host_after"])
def test_desktop_missing_boundary_cannot_claim_completion(
    desktop, desktop_sessions, boundary
):
    def remove(run, directory):
        del run["rows"][0][boundary]

    finish(desktop, desktop_sessions, 1, remove)
    with pytest.raises(ValueError, match="lacks host boundaries"):
        export(desktop, desktop_sessions)


def export(frozen, sessions):
    return report.create_report(frozen[0], frozen[1], sessions)


def truncate(run, directory, completed, *, unresolved=False):
    run.update(
        state="interrupted",
        integrity_verified=False,
        failure="private /Users/name/error",
    )
    for index, row in enumerate(run["rows"]):
        if index < completed:
            continue
        scheduled = {
            key: row[key]
            for key in ("id", "session", "round", "pair", "backend", "workload")
        }
        if index == completed and unresolved:
            scheduled.update(
                state="attempted", argv=row["argv"], host_before=row["host_before"]
            )
        else:
            scheduled["state"] = "planned"
            for path in (directory / row["id"]).iterdir():
                path.unlink()
        run["rows"][index] = scheduled


def test_complete_pair_first_report(frozen, sessions):
    value = export(frozen, sessions)
    assert value["status"] == "complete" and value["accounting_complete"]
    assert value["accounting"]["known_attempted"] == 48
    assert value["accounting"]["included_pairs"] == 24
    assert value["quality_comparison"] == {
        "quality_preserved": True,
        "backend_adoption": False,
    }
    for workload in value["workloads"].values():
        assert workload["cpu_over_kleidiai"] == {
            "count": 6,
            "median": 2.0,
            "minimum": 2.0,
            "maximum": 2.0,
        }
        assert all(
            item["cpu_over_kleidiai"]["count"] == 2 for item in workload["sessions"]
        )
    raw = json.dumps(value)
    assert "PRIVATE" not in raw and str(frozen[2]) not in raw
    assert "private-user" not in raw
    assert value["sessions"][0]["rows"][0]["samples_ns"][0] == 9000000000
    assert (
        value["sessions"][0]["rows"][0]["host_before"]["thermal"]["status"]
        == "unavailable"
    )


def test_missing_sessions_are_unknown_not_unattempted(frozen, sessions):
    value = export(frozen, sessions[:1])
    assert value["status"] == "incomplete" and not value["accounting_complete"]
    assert value["accounting"]["unobserved_invocations"] == 32
    assert value["accounting"]["known_unattempted"] == 0
    assert len([row for item in value["sessions"] for row in item["rows"]]) == 48
    assert value["workloads"]["pp512"]["cpu_over_kleidiai"]["count"] == 2


def test_ratio_distribution_keeps_six_pairs_and_session_ranges(frozen, sessions):
    def vary(run, directory):
        for row in run["rows"]:
            if row["backend"] != "cpu":
                continue
            factor = 2 ** (2 * (row["session"] - 1) + row["round"] - 1)
            native = row["observed"]
            native["samples_ns"] = [value * factor for value in native["samples_ns"]]
            native["avg_ns"] = sum(native["samples_ns"]) // 4
            native["samples_ts"] = [
                1e9 * (native["n_prompt"] + native["n_gen"]) / value
                for value in native["samples_ns"]
            ]
            path = directory / row["id"] / "stdout.json"
            save(path, [native])
            row["artifacts"]["stdout.json"] = bench.asset(path)

    for number in range(1, 4):
        finish(frozen, sessions, number, vary)
    value = export(frozen, sessions)
    for workload in value["workloads"].values():
        assert workload["cpu_over_kleidiai"] == {
            "count": 6,
            "minimum": 2.0,
            "maximum": 64.0,
            "median": 12.0,
        }
        assert [
            item["cpu_over_kleidiai"]["median"] for item in workload["sessions"]
        ] == [3.0, 12.0, 48.0]


def test_unresolved_attempt_keeps_unpinned_raw_files_without_success(frozen, sessions):
    finish(
        frozen,
        sessions,
        1,
        lambda run, directory: truncate(run, directory, 2, unresolved=True),
    )
    value = export(frozen, sessions)
    assert value["accounting"]["unresolved_attempts"] == 1
    assert value["accounting"]["known_unattempted"] == 13
    row = value["sessions"][0]["rows"][2]
    assert row["state"] == "attempted" and len(row["raw_files"]) == 2
    assert "samples_ns" not in row and "outcome" not in row
    assert (
        value["accounting"]["included_pairs"] == 16
    )  # Failed final integrity excludes session 1.


def test_failed_output_is_retained_without_changing_denominators(frozen, sessions):
    def fail(run, directory):
        row = run["rows"][0]
        row.update(outcome="invalid_output", failure="/Users/private-user/error")
        raw = directory / row["id"] / "stdout.json"
        raw.write_bytes(b"!" * (bench.MAX_JSON_BYTES + 1))
        row["artifacts"]["stdout.json"] = bench.asset(raw)

    finish(frozen, sessions, 1, fail)
    value = export(frozen, sessions)
    assert value["accounting_complete"] and value["status"] == "incomplete"
    assert value["accounting"]["outcomes"]["invalid_output"] == 1
    assert value["accounting"]["included_pairs"] == 23
    row = value["sessions"][0]["rows"][0]
    assert row["raw_files"]["stdout.json"]["size"] > bench.MAX_JSON_BYTES
    assert "samples_ns" not in row and "failure_sha256" in row


def test_final_host_violation_excludes_only_last_pair(frozen, sessions):
    def violate(run, directory):
        run["state"] = "host_blocked"
        row = run["rows"][-1]
        row["host_after"]["pages"]["value"]["Swapouts"] += 1
        row["host_deltas"] = bench.host.deltas(row["host_before"], row["host_after"])
        plan = bench.read_json(frozen[0])
        run["host_violations"] = report.boundary_violations(plan, run["rows"])

    finish(frozen, sessions, 1, violate)
    value = export(frozen, sessions)
    assert value["accounting"]["outcomes"]["success"] == 48
    assert value["accounting"]["included_pairs"] == 23
    assert value["status"] == "incomplete"
    assert (
        value["sessions"][0]["pairs"][-1]["host_deltas"]["pages"]["value"]["Swapouts"]
        == 1
    )


def test_host_stop_before_first_launch_preserves_zero_attempts(frozen, sessions):
    def stop(run, directory):
        truncate(run, directory, 0)
        run.update(state="host_blocked", integrity_verified=True, failure=None)
        row = run["rows"][0]
        row["host_before"] = fixtures.stable_host()
        row["host_before"]["memory_pressure"]["value"] = {"code": 2, "label": "warning"}
        run["host_violations"] = report.boundary_violations(
            bench.read_json(frozen[0]), run["rows"]
        )

    finish(frozen, sessions, 1, stop)
    value = export(frozen, sessions)
    assert value["accounting"]["known_unattempted"] == 16
    assert value["accounting"]["known_attempted"] == 32
    assert value["accounting"]["included_pairs"] == 16
    assert value["accounting_complete"] and value["status"] == "incomplete"


@pytest.mark.parametrize(
    "mutation",
    [
        "run_hash",
        "raw_hash",
        "archive_plan",
        "summary",
        "model",
        "harness",
        "helper",
        "build",
        "artifact_path",
        "symlink",
        "schedule",
        "identity",
        "outcome",
        "missing_raw",
        "host_violations",
    ],
)
def test_corrupted_evidence_rejected(frozen, sessions, mutation):
    path, digest, root = frozen
    directory = sessions[0]["directory"]
    run = bench.read_json(directory / "run.json")
    plan = bench.read_json(path)
    row = run["rows"][0]
    if mutation == "run_hash":
        sessions[0]["run_sha256"] = "0" * 64
    elif mutation == "raw_hash":
        Path(row["artifacts"]["stderr.log"]["path"]).write_text("changed")
    elif mutation == "archive_plan":
        (directory / "plan.json").write_text("{}")
    elif mutation == "summary":
        summary = bench.read_json(directory / "summary.json")
        summary["pairs"][0]["cpu_over_kleidiai"] = 999
        save(directory / "summary.json", summary)
    elif mutation == "model":
        Path(plan["model"]["path"]).write_bytes(b"changed")
    elif mutation in {"harness", "helper"}:
        plan["harness_sha256" if mutation == "harness" else "host_helper_sha256"] = (
            "0" * 64
        )
        save(path, plan)
        digest = bench.runtime.digest(path)
    elif mutation == "build":
        Path(plan["backends"]["cpu"]["build_record"]["path"]).write_text("{}")
    elif mutation == "artifact_path":
        alias = root / "alias.log"
        alias.write_bytes(Path(row["artifacts"]["stderr.log"]["path"]).read_bytes())
        row["artifacts"]["stderr.log"] = bench.asset(alias)
    elif mutation == "symlink":
        original = Path(row["artifacts"]["stderr.log"]["path"])
        alias = root / "alias.log"
        original.rename(alias)
        original.symlink_to(alias)
    elif mutation == "schedule":
        run["rows"][0], run["rows"][1] = run["rows"][1], run["rows"][0]
    elif mutation == "identity":
        run["session"] = 2
    elif mutation == "outcome":
        row["outcome"] = "invented"
    elif mutation == "missing_raw":
        Path(row["artifacts"]["stderr.log"]["path"]).unlink()
    else:
        run["host_violations"] = [{"reasons": ["invented"]}]
    if mutation in {
        "artifact_path",
        "schedule",
        "identity",
        "outcome",
        "host_violations",
    }:
        save(directory / "run.json", run)
        sessions[0]["run_sha256"] = bench.runtime.digest(directory / "run.json")
    with pytest.raises((ValueError, FileNotFoundError)):
        report.create_report(path, digest, sessions)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_id",
        "duplicate_directory",
        "invalid_session",
        "missing_summary",
        "out_of_order",
    ],
)
def test_partial_and_inconsistent_ledgers(frozen, sessions, mutation):
    if mutation == "missing_summary":
        (sessions[0]["directory"] / "summary.json").unlink()
        value = export(frozen, sessions)
        assert value["status"] == "incomplete" and not value["accounting_complete"]
        return
    if mutation == "duplicate_id":
        sessions[1]["session"] = 1
    elif mutation == "duplicate_directory":
        sessions[1]["directory"] = sessions[0]["directory"]
    elif mutation == "invalid_session":
        sessions[0]["session"] = True
    else:
        directory = sessions[0]["directory"]
        run = bench.read_json(directory / "run.json")
        run["rows"][0]["state"] = "attempted"
        save(directory / "run.json", run)
        sessions[0]["run_sha256"] = bench.runtime.digest(directory / "run.json")
    with pytest.raises(ValueError):
        export(frozen, sessions)


def test_negative_quality_is_not_suppressed(frozen, sessions):
    path, _, root = frozen
    plan = bench.read_json(path)
    approval = Path(plan["prerequisites"]["quality_baseline"]["path"])
    quality = bench.read_json(approval)
    quality["quality_preserved"] = False
    save(approval, quality)
    plan["quality_comparison"]["quality_preserved"] = False
    plan["prerequisites"]["quality_baseline"] = bench.asset(approval)
    save(path, plan)
    digest = bench.runtime.digest(path)
    for item in sessions:
        (item["directory"] / "plan.json").write_bytes(path.read_bytes())
        run = bench.read_json(item["directory"] / "run.json")
        run["plan_sha256"] = digest
        save(item["directory"] / "run.json", run)
        save(item["directory"] / "summary.json", bench.summarize(plan, run))
        item["run_sha256"] = bench.runtime.digest(item["directory"] / "run.json")
    value = report.create_report(path, digest, sessions)
    assert value["status"] == "complete"
    assert value["quality_comparison"] == {
        "quality_preserved": False,
        "backend_adoption": False,
    }


def test_output_is_create_only(frozen, sessions, tmp_path, monkeypatch):
    output = tmp_path / "public.json"
    arguments = [
        "report",
        "--plan",
        str(frozen[0]),
        "--plan-sha256",
        frozen[1],
        "--output",
        str(output),
    ]
    for item in sessions:
        arguments += [
            "--session",
            str(item["session"]),
            str(item["directory"]),
            item["run_sha256"],
        ]
    monkeypatch.setattr("sys.argv", arguments)
    report.main()
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        report.main()
    assert output.read_bytes() == original


def test_evidence_mutation_during_export_rejected(tmp_path):
    path = tmp_path / "record.json"
    path.write_text("{}")
    evidence = report.Evidence()
    evidence.raw(path)
    path.write_text("[]")
    with pytest.raises(ValueError):
        evidence.check()


def test_helper_change_during_export_is_rejected(frozen, sessions, monkeypatch):
    original = report.signature
    count = 0

    def changed(path):
        nonlocal count
        value = original(path)
        if path == bench.HOST_HELPER:
            count += 1
            if count >= 3:
                return (*value[:-1], value[-1] + 1)
        return value

    monkeypatch.setattr(report, "signature", changed)
    with pytest.raises(ValueError, match="changed during export"):
        export(frozen, sessions)


def test_host_reason_codes_remain_distinct_without_error_text():
    snapshot = fixtures.stable_host()
    snapshot["memory_pressure"] = {
        "status": "unavailable",
        "reason": "permission_denied",
    }
    snapshot["swap"] = {"status": "unavailable", "reason": "/Users/private/error"}
    value = report.public_host(snapshot)
    assert value["thermal"]["reason"] == "cpu_limits_not_reported"
    assert value["memory_pressure"]["reason"] == "permission_denied"
    assert "reason" not in value["swap"] and "reason_sha256" in value["swap"]
    report.privacy_check(value)


@pytest.mark.parametrize(
    "mutation", ["planned_finished", "host_unverified", "after_host_stop"]
)
def test_impossible_session_state_rejected(frozen, sessions, mutation):
    def alter(run, directory):
        if mutation == "planned_finished":
            run.update(state="planned", integrity_verified=False)
            return
        run["state"] = "host_blocked"
        row = run["rows"][-1 if mutation == "host_unverified" else 0]
        row["host_after"]["pages"]["value"]["Swapouts"] += 1
        row["host_deltas"] = bench.host.deltas(row["host_before"], row["host_after"])
        run["host_violations"] = report.boundary_violations(
            bench.read_json(frozen[0]), run["rows"]
        )
        if mutation == "host_unverified":
            run["integrity_verified"] = False

    finish(frozen, sessions, 1, alter)
    with pytest.raises(ValueError):
        export(frozen, sessions)
