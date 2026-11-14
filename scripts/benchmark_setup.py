"""Freeze and record desktop setup; prepare calibration for explicit review."""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import selectors
import signal
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "setup_benchmark", ROOT / "scripts/benchmark.py"
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
host, timing = bench.host, bench.timing
sha, require = bench.runtime.digest, bench.require
SOURCE_PATHS = (
    Path(__file__),
    ROOT / "scripts/benchmark.py",
    bench.HELPER,
    bench.HOST_HELPER,
    bench.TIMING_HELPER,
)
WORKLOADS = {"pp2048": (2048, 0, 0, 4), "tg128-d2048": (0, 128, 2048, 4)}
DEADLINE_SECONDS = 1200
OUTPUT_LIMIT_BYTES = 16 << 20
DEADLINE_RULE = "max(600, ceil(2 * longest eligible continuous seconds / 60) * 60)"
PLANNED_RUNS = [
    f"{backend}-{workload}" for workload in WORKLOADS for backend in ("cpu", "kleidiai")
]


class OutputLimit(RuntimeError):
    pass


def controller_sources():
    return {str(path.relative_to(ROOT)): sha(path) for path in SOURCE_PATHS}


LOADED_SOURCES = controller_sources()


def runtime_assets(plan):
    yield plan["runtime_config"]
    yield plan["method"]
    for build in plan["backends"].values():
        for key in ("binary", "cache", "build_record", "compile_commands"):
            yield build[key]
        yield from build["libraries"].values()


def fingerprint(plan):
    for item in runtime_assets(plan):
        bench.verify_asset(item)
    bench.verify_builds(plan)
    for build in plan["backends"].values():
        record = bench.read_json(Path(build["build_record"]["path"]))
        require(
            record["compile_commands_sha256"] == build["compile_commands"]["sha256"],
            "Compile commands differ from their build record.",
        )
    require(
        controller_sources() == plan["controller_sources"] == LOADED_SOURCES,
        "Setup execution helpers changed.",
    )
    return {item["path"]: item["sha256"] for item in runtime_assets(plan)}


def file_stat(path):
    status = path.stat()
    return {
        "path": str(path),
        "size": status.st_size,
        "device": status.st_dev,
        "inode": status.st_ino,
        "mtime_ns": status.st_mtime_ns,
        "ctime_ns": status.st_ctime_ns,
    }


def model_identity(plan, ledger):
    model = Path(plan["model"]["path"])
    signature = file_stat(model)
    cached = ledger.get("verified_model")
    if cached is None or cached["stat"] != signature:
        bench.verify_asset(plan["model"])
        require(file_stat(model) == signature, "Model changed during verification.")
        cached = {"stat": signature, "sha256": plan["model"]["sha256"]}
        ledger["verified_model"] = cached
    require(cached["sha256"] == plan["model"]["sha256"], "Model pin changed.")
    return model, cached


def check_method(method, plan):
    require(
        method.get("study") == "desktop" and method.get("study_id") == plan["study_id"],
        "Setup method identifies a different study.",
    )
    expected = {
        "runs": PLANNED_RUNS,
        "fresh_observations_required": 4,
        "timeout_seconds": DEADLINE_SECONDS,
        "output_limit_bytes": OUTPUT_LIMIT_BYTES,
        "swap_policy": "record_counter_increases",
        "timings_in_final_comparison": False,
        "measured_deadline_rule": DEADLINE_RULE,
    }
    require(
        all(
            method["setup"].get(key) == value
            and type(method["setup"].get(key)) is type(value)
            for key, value in expected.items()
        ),
        "Method differs from the supported four-run setup.",
    )
    for key in ("idle_before_session_seconds", "idle_between_pairs_seconds"):
        require(
            method["measurement"][key] == plan[key], "Method rest interval differs."
        )
    if "identity" in method:
        require(
            method["identity"]
            == {
                "model_sha256": plan["model"]["sha256"],
                "llama_cpp_revision": bench.REVISION,
                "timing_builds": {
                    name: build["build_record"]["sha256"]
                    for name, build in plan["backends"].items()
                },
            },
            "Method model or builds differ from the frozen inputs.",
        )
    for key in ("timing_policy", "timing_helper_sha256"):
        if key in method:
            require(method[key] == plan[key], "Method timing guard differs.")


def check_plan(plan):
    require(
        plan.get("schema_version") == 1
        and type(plan["schema_version"]) is int
        and plan.get("kind") == "desktop-benchmark-setup"
        and plan.get("study") == "desktop",
        "Unknown setup plan.",
    )
    require(
        isinstance(plan.get("study_id"), str) and bool(plan["study_id"].strip()),
        "A study ID is required.",
    )
    require(
        plan["planned_runs"] == PLANNED_RUNS
        and set(plan["backends"]) == set(bench.BACKENDS),
        "Setup order or backends differ.",
    )
    require(
        plan["timeout_seconds"] == DEADLINE_SECONDS
        and type(plan["timeout_seconds"]) is int
        and plan["output_limit_bytes"] == OUTPUT_LIMIT_BYTES
        and type(plan["output_limit_bytes"]) is int,
        "Setup limits differ.",
    )
    require(
        plan["controller_sources"] == controller_sources() == LOADED_SOURCES
        and plan["timing_helper_sha256"]
        == plan["controller_sources"]["scripts/benchmark_timing.py"]
        and json.dumps(plan["timing_policy"], sort_keys=True)
        == json.dumps(timing.POLICY, sort_keys=True),
        "Frozen setup helpers or timing policy changed.",
    )
    require(
        json.dumps(plan["settings"], sort_keys=True)
        == json.dumps(bench.SETTINGS, sort_keys=True)
        and plan["flags"] == bench.FLAGS,
        "Setup native settings differ.",
    )
    expected = plan["expected_model"]
    require(
        set(expected) == {"model_type", "model_size", "model_n_params"}
        and isinstance(expected["model_type"], str)
        and bool(expected["model_type"])
        and all(
            type(expected[k]) is int and expected[k] > 0
            for k in ("model_size", "model_n_params")
        ),
        "Expected model identity is invalid.",
    )
    for key in ("idle_before_session_seconds", "idle_between_pairs_seconds"):
        require(
            type(plan[key]) in (int, float)
            and math.isfinite(plan[key])
            and plan[key] >= 0,
            "Rest intervals must be finite nonnegative numbers.",
        )
    bench.verify_asset(plan["method"])
    check_method(
        pinned_json(Path(plan["method"]["path"]), plan["method"]["sha256"]), plan
    )


def freeze(spec_path, destination):
    supplied = bench.read_json(spec_path)
    require(
        supplied.get("study") == "desktop",
        "Public setup currently supports desktop studies.",
    )
    method = bench.asset(supplied["method"])
    method_record = bench.read_json(Path(method["path"]))
    plan = {
        "schema_version": 1,
        "kind": "desktop-benchmark-setup",
        "study": "desktop",
        "study_id": method_record["study_id"],
        "method": method,
        "specification_sha256": sha(spec_path),
        "planned_runs": PLANNED_RUNS,
        "timeout_seconds": DEADLINE_SECONDS,
        "output_limit_bytes": OUTPUT_LIMIT_BYTES,
        "controller_sources": controller_sources(),
        "settings": bench.SETTINGS,
        "flags": bench.FLAGS,
        "timing_policy": copy.deepcopy(timing.POLICY),
        "timing_helper_sha256": sha(bench.TIMING_HELPER),
        "runtime_config": bench.asset(supplied["runtime_config"]),
        "model": {
            **supplied["model"],
            "path": str(Path(supplied["model"]["path"]).resolve(strict=True)),
        },
        "expected_model": supplied["expected_model"],
        "backends": {},
        **{
            key: method_record["measurement"][key]
            for key in ("idle_before_session_seconds", "idle_between_pairs_seconds")
        },
    }
    for name in bench.BACKENDS:
        build = supplied["backends"][name]
        binary = Path(build["binary"]).resolve(strict=True)
        plan["backends"][name] = {
            **{
                key: bench.asset(build[key])
                for key in ("binary", "cache", "build_record")
            },
            "compile_commands": bench.asset(
                binary.parent.parent / "compile_commands.json"
            ),
            "libraries": {
                key: bench.asset(path) for key, path in bench.libraries(binary).items()
            },
        }
    check_plan(plan)
    fingerprint(plan)
    bench.verify_asset(plan["model"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    bench.save_json(destination, plan, create=True)
    return sha(destination)


def pinned_json(path, expected_sha256):
    with path.open("rb") as stream:
        raw = stream.read(bench.MAX_JSON_BYTES + 1)
    require(len(raw) <= bench.MAX_JSON_BYTES, "JSON record exceeds its limit.")
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "Frozen JSON changed.")
    return json.loads(
        raw,
        object_pairs_hook=bench.unique_object,
        parse_constant=bench.reject_constant,
        parse_float=bench.finite_float,
    )


def load_plan(path, expected_sha256):
    plan = pinned_json(path, expected_sha256)
    check_plan(plan)
    return plan


class MeasuredChild:
    """Keep wait4 as the sole reaper; Popen.poll/wait would discard its rusage."""

    def __init__(self, *args, **kwargs):
        self.process = subprocess.Popen(*args, **kwargs)
        self.pid = self.process.pid
        self.stdout, self.stderr = self.process.stdout, self.process.stderr
        self.usage = None

    @property
    def returncode(self):
        return self.process.returncode

    def poll(self):
        if self.returncode is None:
            pid, status, usage = os.wait4(self.pid, os.WNOHANG)
            if pid:
                if pid != self.pid:
                    raise RuntimeError("wait4 returned a different child")
                self.process.returncode = os.waitstatus_to_exitcode(status)
                self.usage = usage
        return self.returncode

    def terminate(self):
        os.kill(self.pid, signal.SIGTERM)

    def kill(self):
        os.kill(self.pid, signal.SIGKILL)


def usage_record(usage, *, system=None):
    if (system or platform.system()) != "Darwin":
        raise ValueError("This pilot's ru_maxrss unit contract requires Darwin")
    if usage is None or type(usage.ru_maxrss) is not int or usage.ru_maxrss <= 0:
        raise ValueError("Missing positive child peak resident memory")
    return {
        "method": "wait4(specific_owned_pid, WNOHANG)",
        "peak_resident_bytes": usage.ru_maxrss,
        "user_seconds": usage.ru_utime,
        "system_seconds": usage.ru_stime,
        "minor_faults": usage.ru_minflt,
        "major_faults": usage.ru_majflt,
        "reported_swaps": usage.ru_nswap,
        "limit": (
            "OS child resource report, including descendants; not host memory, "
            "physical footprint or proof of active paging."
        ),
    }


def study_identity(plan):
    return {
        "study": "desktop",
        "study_id": plan["study_id"],
        "method_sha256": plan["method"]["sha256"],
        "timing_helper_sha256": plan["timing_helper_sha256"],
        "timing_policy": plan["timing_policy"],
    }


def check_next_run(plan, ledger, key):
    if (
        ledger.get("schema_version") != 1
        or ledger.get("purpose") != "setup_only"
        or ledger.get("included_in_final_timing_results") is not False
        or any(ledger.get(k) != value for k, value in study_identity(plan).items())
        or ledger.get("planned_runs") != PLANNED_RUNS
    ):
        raise ValueError("Setup ledger is not bound to this desktop method")
    rows = ledger["runs"]
    if key in rows:
        raise ValueError("Preserve this setup attempt; retries are forbidden")
    if list(rows) != PLANNED_RUNS[: len(rows)] or any(
        row.get("state") != "completed"
        or row.get("eligible_for_deadline_calibration") is not True
        or row.get("host_violations") != []
        or any(row.get(k) != value for k, value in study_identity(plan).items())
        or row.get("timing_valid") is not True
        for row in rows.values()
    ):
        raise ValueError("Setup stopped; no retry, replacement or partial calibration")
    if len(rows) >= len(PLANNED_RUNS) or key != PLANNED_RUNS[len(rows)]:
        raise ValueError("Follow the frozen setup order")


def valid_row_timing(plan, row):
    try:
        identity = timing.host_identity_hash(row["host_before"]["identity"])
        measured = row["timing"]
        key = f"{row['backend']}-{row['workload']}"
        return (
            row.get("returncode") == 0
            and row.get("output_incomplete") is False
            and row.get("cleanup_clock") in {"continuous", "not_needed"}
            and row["timing_helper_sha256"] == plan["timing_helper_sha256"]
            and row["timing_policy"] == timing.POLICY
            and timing.validate_evidence(row["timing_guard"], timing.POLICY, identity)
            and measured
            == bench.row_timing(
                measured["start"], measured["end"], timeout_seconds=DEADLINE_SECONDS
            )
            and measured["valid"]
            and 0 < measured["continuous_seconds"] < DEADLINE_SECONDS
            and row["wall_seconds"] == measured["continuous_seconds"]
            and bench.native_duration_matches(row["native"], measured, timing.POLICY)
            and all(
                {"label": key + ":" + boundary, "sample": measured[boundary]}
                in row["timing_guard"]["checkpoints"]
                for boundary in ("start", "end")
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def verify_previous_rows(plan, output, ledger, rows, *, require_eligible=True):
    require(list(rows) == PLANNED_RUNS[: len(rows)], "Setup record order differs.")
    previous = None
    for key, row in rows.items():
        require(
            key == f"{row['backend']}-{row['workload']}", "Setup row identity differs."
        )
        directory = output / key
        record = directory / "run.json"
        if sha(record) != row["record_sha256"]:
            raise ValueError("Previous setup record changed")
        if pinned_json(record, row["record_sha256"]) != {
            k: v for k, v in row.items() if k != "record_sha256"
        }:
            raise ValueError("Previous setup ledger differs from its record")
        if (
            require_eligible or row.get("eligible_for_deadline_calibration")
        ) and not valid_row_timing(plan, row):
            raise ValueError("Previous setup timing is invalid")
        if row["previous_record_sha256"] != (
            previous["record_sha256"] if previous else None
        ):
            raise ValueError("Previous setup record chain differs")
        if "timing_guard" in row and row["timing_guard"]["previous_checkpoint"] != (
            previous["timing_guard"]["exit_checkpoint"] if previous else None
        ):
            raise ValueError("Previous setup checkpoint chain differs")
        files = {
            str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()
        }
        if files != set(row["artifacts"]) | {"run.json"}:
            raise ValueError("Previous setup artifacts differ")
        for name, asset in row["artifacts"].items():
            path = directory / name
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("Previous setup artifact path escaped its directory")
            if path.stat().st_size != asset["bytes"] or sha(path) != asset["sha256"]:
                raise ValueError("Previous setup artifact changed")
        for name, expected in ledger.get("controller_sources", {}).items():
            archived = "config/" + Path(name).name
            if (
                archived in row["artifacts"]
                and row["artifacts"][archived]["sha256"] != expected
            ):
                raise ValueError("Previous setup execution sources differ")
        if require_eligible or row.get("eligible_for_deadline_calibration"):
            require(
                row["artifacts"]["config/method.json"]["sha256"]
                == plan["method"]["sha256"]
                and row["artifacts"]["config/setup-plan.json"]["sha256"]
                == ledger["plan_sha256"],
                "Archived setup inputs differ.",
            )
            for name, expected in plan["controller_sources"].items():
                require(
                    row["artifacts"]["config/" + Path(name).name]["sha256"] == expected,
                    "Archived setup helper differs.",
                )
            verify_eligible_row(plan, ledger, row, previous, directory)
        previous = row


def verify_eligible_row(plan, ledger, row, previous, directory):
    require(
        row["state"] == "completed"
        and row["outcome"] == "setup_observed"
        and row["timing_valid"] is True
        and row["eligible_for_deadline_calibration"] is True
        and all(row.get(k) == value for k, value in study_identity(plan).items())
        and row["shape"] == list(WORKLOADS[row["workload"]]),
        "Setup observation is not eligible.",
    )
    require(
        row["argv"] == bench.command(plan, row)
        and row["environment"] == {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"},
        "Setup command or environment differs.",
    )
    expected = {item["path"]: item["sha256"] for item in runtime_assets(plan)}
    require(
        row["fingerprint_before"] == row["fingerprint_after"] == expected,
        "Setup runtime fingerprints differ.",
    )
    require(
        row["model"]["sha256"] == plan["model"]["sha256"]
        and row["model"]["stat"]["path"] == plan["model"]["path"]
        and row["model"]["stat"]["size"] == plan["model"]["size"],
        "Setup model differs.",
    )
    native = bench.validate_samples(
        bench.read_json(directory / "stdout.json"), plan, row
    )
    require(native == row["native"], "Setup native record differs from raw output.")
    resources = row["resources"]
    require(
        resources["method"] == "wait4(specific_owned_pid, WNOHANG)"
        and type(resources["peak_resident_bytes"]) is int
        and resources["peak_resident_bytes"] > 0,
        "Setup peak memory is missing or invalid.",
    )
    for key in ("user_seconds", "system_seconds"):
        require(
            type(resources[key]) in (int, float)
            and math.isfinite(resources[key])
            and resources[key] >= 0,
            "Setup CPU resource value is invalid.",
        )
    for key in ("minor_faults", "major_faults", "reported_swaps"):
        require(
            type(resources[key]) is int and resources[key] >= 0,
            "Setup resource count is invalid.",
        )
    policy = ledger["host_policy"]
    host.check_policy(policy)
    before, after = row["host_before"], row["host_after"]
    if previous is None:
        require(
            policy
            == host.policy_from(
                before, swap_activity_policy="record_counter_increases"
            ),
            "Setup host baseline differs.",
        )
    before_failures = host.admit(before, policy)
    if previous is not None:
        before_failures += host.interval_violations(
            previous["host_after"], before, policy
        )
    after_failures = host.admit(after, policy) + host.interval_violations(
        before, after, policy
    )
    require(
        before_failures == row["preflight_host_violations"] == []
        and after_failures == row["host_violations"] == []
        and row["host_deltas"] == host.deltas(before, after),
        "Setup host conditions do not support calibration.",
    )


def capture(
    argv,
    env,
    directory,
    deadline,
    limit,
    on_start,
    *,
    guard,
    label,
    popen=MeasuredChild,
):
    process, observed, retained = None, 0, 0
    start, end = None, None
    result = {"outcome": None, "output_incomplete": False}
    interruption = None
    stdout = (directory / "stdout.json").open("xb")
    try:
        stderr = (directory / "stderr.log").open("xb")
    except BaseException:
        stdout.close()
        raise

    def record_error(operation, error):
        nonlocal interruption
        description = f"{type(error).__name__}: {error}"
        result.setdefault("record_errors", []).append(
            {"operation": operation, "error": description}
        )
        result.setdefault("failure", description)
        result["output_incomplete"] = True
        if result["outcome"] in (None, "captured"):
            result["outcome"] = "record_error"
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            interruption = error

    try:
        start = guard.check(label + ":start")
        process = popen(
            argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        on_start(process.pid)
        with selectors.DefaultSelector() as selector:
            for pipe, output in (
                (process.stdout, stdout),
                (process.stderr, stderr),
            ):
                os.set_blocking(pipe.fileno(), False)
                selector.register(pipe, selectors.EVENT_READ, output)
            while selector.get_map():
                sample = guard.check()
                remaining = (
                    start["continuous_before_ns"]
                    + int(deadline * 1e9)
                    - sample["continuous_after_ns"]
                )
                if remaining <= 0:
                    guard.fail("deadline_exceeded", sample, start)
                events = selector.select(min(timing.POLICY["poll_ns"], remaining) / 1e9)
                sample = guard.check()
                if sample["continuous_after_ns"] >= start["continuous_before_ns"] + int(
                    deadline * 1e9
                ):
                    guard.fail("deadline_exceeded", sample, start)
                for key, _ in events:
                    data = os.read(key.fd, 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    observed += len(data)
                    accepted = data[: max(0, limit - retained)]
                    key.data.write(accepted)
                    retained += len(accepted)
                    if len(accepted) != len(data):
                        raise OutputLimit(
                            "Combined stdout/stderr exceeded its frozen limit"
                        )
            code, end = guard.wait(process, start, deadline, label + ":end")
        result.update(
            returncode=code, outcome="captured" if code == 0 else "exit_error"
        )
        if code:
            result["failure"] = f"Child exited with status {code}"
    except BaseException as error:
        if isinstance(error, timing.GuardError):
            end = error.sample
        result.update(
            outcome="timeout"
            if isinstance(error, timing.GuardError)
            and error.reason == "deadline_exceeded"
            else "timing_error"
            if isinstance(error, timing.GuardError)
            else "output_limit"
            if isinstance(error, OutputLimit)
            else "interrupted"
            if isinstance(error, (KeyboardInterrupt, SystemExit))
            else "launch_error"
            if process is None
            else "capture_error",
            failure=f"{type(error).__name__}: {error}",
            output_incomplete=True,
        )
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            interruption = error
    finally:
        try:
            mode = timing.stop_process(
                process,
                clock_ns=guard.continuous_ns,
                sleep=guard.sleep,
                fallback_ns=guard.clock.monotonic_ns,
                seconds=timing.POLICY["cleanup_seconds"],
            )
            result["cleanup_clock"] = mode
            guard.record_cleanup("native", mode)
        except (timing.CleanupError, OSError) as error:
            result["cleanup_error"] = str(error)
            guard.record_cleanup("native", None, error)
            if result["outcome"] in (None, "captured"):
                result.update(
                    outcome="cleanup_error",
                    failure=str(error),
                    output_incomplete=True,
                )
        except BaseException as error:
            guard.record_cleanup("native", None, error)
            result["cleanup_error"] = f"{type(error).__name__}: {error}"
            result.setdefault("failure", result["cleanup_error"])
            if result["outcome"] in (None, "captured"):
                result.update(outcome="cleanup_error", output_incomplete=True)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                interruption = error
        finally:
            if process is not None:
                result["returncode"] = process.returncode
                for pipe in (process.stdout, process.stderr):
                    if pipe is not None and not pipe.closed:
                        try:
                            pipe.close()
                        except BaseException as error:
                            record_error("pipe_close", error)
            for output in (stdout, stderr):
                try:
                    output.flush()
                    os.fsync(output.fileno())
                except BaseException as error:
                    record_error("output_flush", error)
                finally:
                    try:
                        output.close()
                    except BaseException as error:
                        record_error("output_close", error)
            reason = guard.evidence["first_failure"]
            result["timing"] = (
                bench.row_timing(
                    start,
                    end,
                    reason["reason"] if reason else None,
                    timeout_seconds=deadline,
                )
                if start is not None
                else {"valid": False, "start": None, "end": None}
            )
            result.update(
                bytes_observed=observed,
                bytes_retained=retained,
                wall_seconds=result["timing"].get("continuous_seconds"),
            )
    return result, interruption


def run(plan_path, plan_sha256, backend, workload, *, output, lock_path=None):
    if backend not in {"cpu", "kleidiai"} or workload not in WORKLOADS:
        raise ValueError("Choose a declared backend and workload")
    shape = WORKLOADS[workload]
    key = f"{backend}-{workload}"
    if platform.system() != "Darwin":
        raise ValueError("Setup pilot supports only the reviewed Darwin RSS units")
    plan = load_plan(plan_path, plan_sha256)
    output.mkdir(parents=True, exist_ok=True)
    lock = lock_path or ROOT / "artifacts/benchmarks/benchmark.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with bench.exclusive_lock(lock):
        ledger_path = output / "setup-ledger.json"
        ledger = (
            bench.read_json(ledger_path)
            if ledger_path.exists()
            else {
                "schema_version": 1,
                "purpose": "setup_only",
                **study_identity(plan),
                "plan_sha256": plan_sha256,
                "included_in_final_timing_results": False,
                "planned_runs": PLANNED_RUNS,
                "runs": {},
            }
        )
        require(
            ledger["plan_sha256"] == plan_sha256,
            "Setup ledger belongs to a different plan.",
        )
        check_next_run(plan, ledger, key)
        directory = output / key
        if directory.exists():
            raise ValueError("Preserve this setup directory; retries are forbidden")
        row = {
            "state": "planned",
            **study_identity(plan),
            "eligible_for_deadline_calibration": False,
            "planned_at": bench.timestamp(),
            "backend": backend,
            "workload": workload,
            "shape": list(shape),
            "deadline_seconds": DEADLINE_SECONDS,
            "output_limit_bytes": OUTPUT_LIMIT_BYTES,
            "timing_valid": False,
            "previous_record_sha256": (
                next(reversed(ledger["runs"].values()))["record_sha256"]
                if ledger["runs"]
                else None
            ),
        }
        ledger["runs"][key] = row
        bench.save_json(ledger_path, ledger)
        interruption = None
        children = []
        guard = None
        try:
            directory.mkdir()
            initial_host = host.collect()
            row["host_initial"] = initial_host
            initial_identity = initial_host["identity"]
            previous_rows = {k: v for k, v in ledger["runs"].items() if k != key}
            previous = next(reversed(previous_rows.values()), None)
            guard = timing.TimingGuard(
                timing.POLICY,
                timing.host_identity_hash(initial_identity),
                previous_checkpoint=previous["timing_guard"]["exit_checkpoint"]
                if previous
                else None,
            )
            row["timing_guard"] = guard.evidence
            guard.start()
            guard.check("setup:verify-prior")
            verify_previous_rows(plan, output, ledger, previous_rows)
            row["previous_record_sha256"] = (
                previous["record_sha256"] if previous else None
            )
            before_sources = controller_sources()
            frozen = ledger.setdefault("controller_sources", before_sources)
            if frozen != before_sources or before_sources != LOADED_SOURCES:
                raise ValueError("Setup execution helpers changed between backends")
            config_dir = directory / "config"
            config_dir.mkdir()
            for path, expected in before_sources.items():
                archived = config_dir / Path(path).name
                archived.write_bytes((ROOT / path).read_bytes())
                if sha(archived) != expected:
                    raise ValueError("Execution helper changed while archiving")
            for name, path, expected in (
                ("method.json", Path(plan["method"]["path"]), plan["method"]["sha256"]),
                ("setup-plan.json", plan_path, plan_sha256),
            ):
                archived = config_dir / name
                archived.write_bytes(path.read_bytes())
                require(
                    sha(archived) == expected, "Setup input changed while archiving."
                )
            model, model_record = model_identity(plan, ledger)
            before = fingerprint(plan)
            row.update(
                model=model_record,
                fingerprint_before=before,
                argv=bench.command(plan, row),
                environment={"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"},
            )
            guard.check("setup:before-host")
            row["host_before"] = host.collect()
            policy = ledger.get("host_policy")
            if policy is None:
                try:
                    policy = host.policy_from(
                        row["host_before"],
                        swap_activity_policy="record_counter_increases",
                    )
                except ValueError:
                    row["host_violations"] = ["host_baseline_unadmitted"]
                    row["host_findings"] = ["host_baseline_unadmitted"]
                    row["preflight_host_violations"] = ["host_baseline_unadmitted"]
                    row["preflight_host_findings"] = ["host_baseline_unadmitted"]
                    raise
                ledger["host_policy"] = policy
            if policy.get("swap_activity_policy") != "record_counter_increases":
                raise ValueError("Setup host policy is not the frozen desktop policy")
            violations = host.admit(row["host_before"], policy)
            findings = list(violations)
            position = PLANNED_RUNS.index(key)
            if position:
                previous_key = PLANNED_RUNS[position - 1]
                previous = ledger["runs"][previous_key].get("host_after", {})
                gap_violations = host.interval_violations(
                    previous, row["host_before"], policy
                )
                gap_findings = host.interval_violations(previous, row["host_before"])
                row["host_gap"] = {
                    "previous_run": previous_key,
                    "deltas": host.deltas(previous, row["host_before"]),
                    "violations": gap_violations,
                    "findings": gap_findings,
                }
                violations += gap_violations
                findings += gap_findings
            row["preflight_host_violations"] = list(violations)
            row["preflight_host_findings"] = list(findings)
            row["host_violations"] = list(violations)
            row["host_findings"] = list(findings)
            if row["host_before"]["identity"] != initial_identity:
                raise ValueError("Host identity changed during setup verification")
            guard.check("setup:after-host")
            if violations:
                raise ValueError(f"Host preflight failed: {', '.join(violations)}")
            row.update(state="attempted", attempted_at=bench.timestamp())
            bench.save_json(ledger_path, ledger)

            def started(pid):
                row["pid"] = pid
                bench.save_json(ledger_path, ledger)

            def spawn(*args, **kwargs):
                child = MeasuredChild(*args, **kwargs)
                children.append(child)
                return child

            captured, interruption = capture(
                row["argv"],
                row["environment"],
                directory,
                DEADLINE_SECONDS,
                OUTPUT_LIMIT_BYTES,
                started,
                popen=spawn,
                guard=guard,
                label=key,
            )
            row.update(captured)
            if children and children[0].usage is not None:
                row["resources"] = usage_record(children[0].usage)
            guard.check("setup:before-final-host")
            row["host_after"] = host.collect()
            row["host_deltas"] = host.deltas(row["host_before"], row["host_after"])
            row["host_violations"] = host.admit(
                row["host_after"], policy
            ) + host.interval_violations(row["host_before"], row["host_after"], policy)
            row["host_findings"] = list(
                dict.fromkeys(
                    findings
                    + host.admit(row["host_after"], policy)
                    + host.interval_violations(row["host_before"], row["host_after"])
                )
            )
            guard.check("setup:after-final-host")
            after = fingerprint(plan)
            row["fingerprint_after"] = after
            if (
                before != after
                or before_sources != controller_sources()
                or file_stat(model) != model_record["stat"]
            ):
                raise ValueError("Runtime, model or execution helpers changed")
            if row["outcome"] != "captured":
                row["state"] = "failed"
            else:
                row["native"] = bench.validate_samples(
                    bench.read_json(directory / "stdout.json"), plan, row
                )
                if not bench.native_duration_matches(
                    row["native"], row["timing"], timing.POLICY
                ):
                    guard.fail("native_duration_inconsistent")
                if "resources" not in row:
                    raise ValueError("No wait4 resource report")
                row.update(
                    state="completed",
                    outcome="setup_observed",
                    eligible_for_deadline_calibration=False,
                )
        except BaseException as error:
            row["state"] = "failed"
            row.setdefault("failure", f"{type(error).__name__}: {error}")
            if isinstance(error, timing.GuardError) and row.get("outcome") in (
                None,
                "captured",
                "setup_observed",
            ):
                row["outcome"] = (
                    "timeout" if error.reason == "deadline_exceeded" else "timing_error"
                )
            if row.get("outcome") in (None, "captured"):
                row["outcome"] = (
                    "preflight_error"
                    if "attempted_at" not in row
                    else "validation_error"
                )
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                interruption = error
        finally:
            try:
                row["artifacts"] = (
                    {
                        str(path.relative_to(directory)): {
                            "sha256": sha(path),
                            "bytes": path.stat().st_size,
                        }
                        for path in directory.rglob("*")
                        if path.is_file()
                    }
                    if directory.exists()
                    else {}
                )
                if guard is not None and guard.evidence["state"] == "active":
                    guard.check("setup:verified")
            except BaseException as error:
                row.update(
                    state="failed", evidence_error=f"{type(error).__name__}: {error}"
                )
                row.setdefault("failure", row["evidence_error"])
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    interruption = error
            finally:
                if guard is not None:
                    guard.close()
                    try:
                        row["timing_valid"] = timing.validate_evidence(
                            guard.evidence,
                            timing.POLICY,
                            timing.host_identity_hash(initial_identity),
                        )
                    except ValueError as error:
                        row["timing_evidence_error"] = str(error)
                    if not row["timing_valid"]:
                        row["state"] = "failed"
                        row.setdefault(
                            "failure", "Timing guard did not close successfully"
                        )
                        if row.get("outcome") in (None, "captured", "setup_observed"):
                            row["outcome"] = "timing_error"
                row["eligible_for_deadline_calibration"] = (
                    row["state"] == "completed"
                    and row["timing_valid"]
                    and row.get("host_violations") == []
                    and valid_row_timing(plan, row)
                )
                row["finished_at"] = bench.timestamp()
                if directory.exists():
                    record = directory / "run.json"
                    if record.exists():
                        row.update(
                            state="failed", eligible_for_deadline_calibration=False
                        )
                        row.setdefault("failure", "Setup record already exists")
                    else:
                        bench.save_json(record, row)
                        row["record_sha256"] = sha(record)
                bench.save_json(ledger_path, ledger)
        if interruption is not None:
            raise interruption
        return row


def suggested_deadline(plan, rows):
    expected = {
        (backend, workload) for backend in ("cpu", "kleidiai") for workload in WORKLOADS
    }
    if (
        len(rows) != len(expected)
        or {(row.get("backend"), row.get("workload")) for row in rows} != expected
        or any(
            row.get("state") != "completed"
            or row.get("eligible_for_deadline_calibration") is not True
            or row.get("host_violations") != []
            or any(row.get(k) != value for k, value in study_identity(plan).items())
            or row.get("timing_valid") is not True
            or not valid_row_timing(plan, row)
            or row.get("shape") != list(WORKLOADS[row["workload"]])
            for row in rows
        )
    ):
        raise ValueError(
            "All four eligible desktop setup observations "
            "with the frozen method are required"
        )
    seconds = [row.get("wall_seconds") for row in rows]
    if any(
        type(value) not in (float, int) or not math.isfinite(value) or value <= 0
        for value in seconds
    ):
        raise ValueError("Invalid setup wall duration")
    longest = max(seconds)
    return max(600, math.ceil(2 * longest / 60) * 60)


def report(plan_path, plan_sha256, output, ledger_sha256, destination):
    ledger_path = output / "setup-ledger.json"
    consumed = {
        path: file_stat(path) for path in (plan_path, ledger_path, *SOURCE_PATHS)
    }
    plan = pinned_json(plan_path, plan_sha256)
    method_path = Path(plan["method"]["path"])
    consumed[method_path] = file_stat(method_path)
    check_plan(plan)
    ledger = pinned_json(ledger_path, ledger_sha256)
    require(
        ledger["plan_sha256"] == plan_sha256
        and ledger.get("schema_version") == 1
        and ledger.get("purpose") == "setup_only"
        and ledger.get("included_in_final_timing_results") is False
        and ledger["planned_runs"] == PLANNED_RUNS
        and all(ledger.get(k) == value for k, value in study_identity(plan).items()),
        "Setup ledger belongs to a different plan.",
    )
    rows = ledger["runs"]
    require(list(rows) == PLANNED_RUNS[: len(rows)], "Setup record order differs.")
    inventories = {}
    for key, row in rows.items():
        directory = output / key
        require(
            not destination.resolve().is_relative_to(directory.resolve()),
            "Save the candidate outside setup observation directories.",
        )
        inventories[directory] = set(row["artifacts"]) | {"run.json"}
        for name in inventories[directory]:
            path = directory / name
            require(
                path.resolve().is_relative_to(directory.resolve()),
                "Setup artifact path escaped its directory.",
            )
            consumed[path] = file_stat(path)
    verify_previous_rows(plan, output, ledger, rows, require_eligible=False)
    complete = len(rows) == 4 and all(
        row.get("eligible_for_deadline_calibration") is True for row in rows.values()
    )
    deadline = suggested_deadline(plan, list(rows.values())) if complete else None
    result = {
        "schema_version": 1,
        "purpose": "setup_pilot",
        "status": "pending_review" if complete else "incomplete",
        **study_identity(plan),
        "model_sha256": plan["model"]["sha256"],
        "timing_builds": {
            name: build["build_record"]["sha256"]
            for name, build in plan["backends"].items()
        },
        "timeout_seconds": deadline,
        **{
            key: plan[key]
            for key in ("idle_before_session_seconds", "idle_between_pairs_seconds")
        },
        "included_in_final_timing_results": False,
        "deadline_rule": DEADLINE_RULE,
        "review_required": (
            "Review the method, raw observations, host conditions and calibration. "
            "Record that review before marking approved_for_desktop_measurement."
        ),
        "evidence": {
            "setup_plan_sha256": plan_sha256,
            "setup_ledger_sha256": ledger_sha256,
            "row_sha256": {key: row["record_sha256"] for key, row in rows.items()},
        },
        "planned_outcomes": [],
    }
    for key in PLANNED_RUNS:
        row = rows.get(key)
        result["planned_outcomes"].append(
            {
                "id": key,
                "state": row["state"] if row else "unattempted",
                "attempted": row is not None and "attempted_at" in row,
                "outcome": row.get("outcome") if row else None,
                "eligible_for_deadline_calibration": row.get(
                    "eligible_for_deadline_calibration"
                )
                if row
                else False,
                "timing_valid": valid_row_timing(plan, row) if row else False,
                "continuous_seconds": row.get("wall_seconds") if row else None,
                "peak_resident_bytes": row.get("resources", {}).get(
                    "peak_resident_bytes"
                )
                if row
                else None,
                "host_findings": row.get("host_findings", []) if row else [],
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(
        all(file_stat(path) == before for path, before in consumed.items()),
        "Setup evidence changed during export.",
    )
    require(
        all(
            {
                str(path.relative_to(directory))
                for path in directory.rglob("*")
                if path.is_file()
            }
            == names
            for directory, names in inventories.items()
        ),
        "Setup artifact inventory changed during export.",
    )
    bench.save_json(destination, result, create=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser(
        "freeze", help="Pin setup inputs without starting a model."
    )
    prepare.add_argument("--spec", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser(
        "run", help="Run the next declared setup observation."
    )
    execute.add_argument("--backend", choices=bench.BACKENDS, required=True)
    execute.add_argument("--workload", choices=WORKLOADS, required=True)
    export = commands.add_parser(
        "report", help="Verify records and prepare explicit review."
    )
    export.add_argument("--runs", type=Path, required=True)
    export.add_argument("--ledger-sha256", required=True)
    for command in (execute, export):
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--plan-sha256", required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        print(freeze(args.spec, args.output))
    elif args.action == "run":
        row = run(
            args.plan, args.plan_sha256, args.backend, args.workload, output=args.output
        )
        print(f"Desktop setup: {row['state']} ({row.get('outcome')})")
        if row["state"] != "completed" or not row["eligible_for_deadline_calibration"]:
            parser.exit(1, "Setup stopped; preserve the records.\n")
    else:
        result = report(
            args.plan, args.plan_sha256, args.runs, args.ledger_sha256, args.output
        )
        print(f"Setup: {result['status']}")
        if result["status"] != "pending_review":
            parser.exit(1, "Incomplete setup cannot calibrate a measured study.\n")


if __name__ == "__main__":
    main()
