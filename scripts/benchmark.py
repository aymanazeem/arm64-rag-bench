"""Run a frozen fixed-token benchmark with durable evidence."""

import argparse
import contextlib
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/runtime.py"
spec = importlib.util.spec_from_file_location("benchmark_runtime", HELPER)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
HOST_HELPER = Path(__file__).with_name("benchmark_host.py")
spec = importlib.util.spec_from_file_location("benchmark_host", HOST_HELPER)
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)

REVISION = "8ca5a8008b57aa39f2601546c0d07912045080a5"
BACKENDS = ("cpu", "kleidiai")
STUDIES = {
    "controlled": "controlled-fixed-token-benchmark",
    "desktop": "desktop-fixed-token-benchmark",
}
WORKLOADS = {
    "pp512": (512, 0, 0),
    "pp2048": (2048, 0, 0),
    "tg128-d512": (0, 128, 512),
    "tg128-d2048": (0, 128, 2048),
}
SETTINGS = {
    "n_batch": 512,
    "n_ubatch": 512,
    "n_threads": 4,
    "cpu_mask": "0x0",
    "cpu_strict": False,
    "poll": 50,
    "type_k": "f16",
    "type_v": "f16",
    "n_gpu_layers": 0,
    "n_cpu_moe": 0,
    "split_mode": "layer",
    "main_gpu": 0,
    "no_kv_offload": False,
    "flash_attn": 0,
    "devices": "auto",
    "tensor_split": "0.00",
    "tensor_buft_overrides": "none",
    "load_mode": "auto",
    "lazy_mode": "auto",
    "embeddings": False,
    "no_op_offload": 0,
    "no_host": False,
    "repack": True,
    "fit_target": 0,
    "fit_min_ctx": 0,
}
FLAGS = [
    "--offline",
    "-r",
    "4",
    "-t",
    "4",
    "-b",
    "512",
    "-ub",
    "512",
    "-ctk",
    "f16",
    "-ctv",
    "f16",
    "-fa",
    "off",
    "-ngl",
    "0",
    "-lm",
    "auto",
    "-lzm",
    "auto",
    "--repack",
    "1",
    "-o",
    "json",
]
MAX_JSON_BYTES = 2 * 1024 * 1024
CLEANUP_SECONDS = 2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def finite_float(value):
    number = float(value)
    require(math.isfinite(number), "Non-finite JSON number.")
    return number


def read_json(path):
    require(path.stat().st_size <= MAX_JSON_BYTES, "JSON record exceeds its limit.")
    return json.loads(
        path.read_bytes(),
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
        parse_float=finite_float,
    )


def save_bytes(path, raw, *, create=False):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if create:
            path.hardlink_to(temporary)
        else:
            temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path, value, *, create=False):
    raw = (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2) + "\n"
    ).encode()
    save_bytes(path, raw, create=create)


def asset(path):
    path = Path(path).resolve(strict=True)
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": runtime.digest(path),
    }


def verify_asset(item):
    require(type(item["size"]) is int and item["size"] >= 0, "Invalid asset size.")
    require(
        isinstance(item["sha256"], str)
        and len(item["sha256"]) == 64
        and all(char in "0123456789abcdef" for char in item["sha256"]),
        "Invalid asset digest.",
    )
    runtime.verify(Path(item["path"]), item)


def libraries(binary):
    return {
        p.name: p
        for p in binary.parent.iterdir()
        if p.is_file() and (p.suffix == ".dylib" or ".so" in p.name)
    }


def schedule(seed):
    require(type(seed) is int, "The schedule seed must be an integer.")
    rng, rows = random.Random(seed), []
    for session in range(1, 4):
        orders = [BACKENDS, tuple(reversed(BACKENDS))]
        rng.shuffle(orders)
        for number, order in enumerate(orders, 1):
            names = list(WORKLOADS)
            rng.shuffle(names)
            for workload in names:
                pair = f"s{session}-r{number}-{workload}"
                for backend in order:
                    rows.append(
                        {
                            "id": f"{pair}-{backend}",
                            "session": session,
                            "round": number,
                            "pair": pair,
                            "backend": backend,
                            "workload": workload,
                        }
                    )
    return rows


def all_assets(plan):
    yield plan["model"]
    yield plan["runtime_config"]
    yield plan["host_baseline"]
    if "method" in plan:
        yield plan["method"]
    yield from plan["prerequisites"].values()
    for build in plan["backends"].values():
        for key in ("binary", "cache", "build_record"):
            yield build[key]
        yield from build["libraries"].values()


def verify_runtime(plan):
    desktop = plan["kind"] == STUDIES["desktop"]
    for item in all_assets(plan):
        verify_asset(item)
    require(
        host.policy_from(
            read_json(Path(plan["host_baseline"]["path"])),
            swap_activity_policy="record_counter_increases"
            if desktop
            else "exclude_any_counter_increase",
        )
        == plan["host_policy"],
        "Host policy differs from the pinned baseline.",
    )
    for purpose, item in plan["prerequisites"].items():
        approval = read_json(Path(item["path"]))
        require(
            approval.get("status")
            == (
                "approved_for_desktop_measurement"
                if desktop and purpose == "setup_pilot"
                else "approved_for_measurement"
            )
            and approval.get("purpose") == purpose
            and approval.get("model_sha256") == plan["model"]["sha256"]
            and approval.get("timing_builds")
            == {
                name: build["build_record"]["sha256"]
                for name, build in plan["backends"].items()
            },
            "Prerequisite evidence is unapproved or belongs to different inputs.",
        )
        if purpose == "setup_pilot":
            if desktop:
                require(
                    approval.get("study") == "desktop"
                    and approval.get("method_sha256") == plan["method"]["sha256"]
                    and all(
                        type(approval.get(key)) in (int, float)
                        and approval[key] == plan[key]
                        for key in (
                            "timeout_seconds",
                            "idle_before_session_seconds",
                            "idle_between_pairs_seconds",
                        )
                    ),
                    "Desktop setup approval differs from the method or timing policy.",
                )
            else:
                require(
                    approval.get("study", "controlled") == "controlled",
                    "Controlled setup cannot use a desktop approval.",
                )
    quality = read_json(Path(plan["prerequisites"]["quality_baseline"]["path"]))
    require(
        all(
            type(quality.get(key)) is bool
            for key in ("quality_preserved", "backend_adoption")
        )
        and (quality["quality_preserved"] or not quality["backend_adoption"])
        and plan["quality_comparison"]
        == {key: quality[key] for key in ("quality_preserved", "backend_adoption")},
        "Quality comparison flags differ from the reviewed evidence.",
    )
    config = read_json(Path(plan["runtime_config"]["path"]))
    require(config["llama_cpp"]["revision"] == REVISION, "Unexpected native revision.")
    for key, value in {
        "CMAKE_BUILD_TYPE": "Release",
        "GGML_CPU": "ON",
        "GGML_CPU_REPACK": "ON",
        "GGML_METAL": "OFF",
        "GGML_BLAS": "OFF",
        "GGML_ACCELERATE": "OFF",
        "GGML_OPENMP": "OFF",
    }.items():
        require(
            config["cmake"].get(key) == value, f"Controlled build option differs: {key}"
        )
    records = {}
    for name in BACKENDS:
        build = plan["backends"][name]
        record = read_json(Path(build["build_record"]["path"]))
        flags = {
            **config["cmake"],
            "GGML_CPU_KLEIDIAI": "ON" if name == "kleidiai" else "OFF",
        }
        require(
            record["backend"] == name and record["flags"] == flags,
            "Build flags differ.",
        )
        require(record["source"] == config["llama_cpp"], "Build source pin differs.")
        require(
            record["kleidiai"] == (config["kleidiai"] if name == "kleidiai" else None),
            "KleidiAI pin differs.",
        )
        require(record["backends"] == ["ggml-cpu"], "A non-CPU backend is enabled.")
        require(
            record["targets"]["llama-bench"] == build["binary"]["sha256"],
            "Executable differs from its normal build.",
        )
        require(
            record["cache_sha256"] == build["cache"]["sha256"],
            "Build cache pin differs.",
        )
        require(
            record["libraries"]
            == {key: value["sha256"] for key, value in build["libraries"].items()},
            "Library pins differ.",
        )
        require(
            set(libraries(Path(build["binary"]["path"]))) == set(build["libraries"]),
            "Library inventory changed.",
        )
        runtime.checked_cache(Path(build["cache"]["path"]), flags)
        records[name] = record
    for key in ("compiler_flags", "compilers", "tools", "system"):
        require(
            records["cpu"][key] == records["kleidiai"][key], f"Paired {key} differ."
        )


def check_plan(plan):
    require(
        type(plan["schema_version"]) is int
        and plan["schema_version"] == 1
        and plan["kind"] in STUDIES.values(),
        "Unknown plan format.",
    )
    desktop = plan["kind"] == STUDIES["desktop"]
    require(
        plan.get("study", "desktop" if desktop else "controlled")
        == ("desktop" if desktop else "controlled")
        and ("method" in plan) == desktop
        and plan["host_policy"]["swap_activity_policy"]
        == ("record_counter_increases" if desktop else "exclude_any_counter_increase"),
        "Study kind, method and host policy differ.",
    )
    require(set(plan["backends"]) == set(BACKENDS), "Both CPU builds are required.")
    require(
        json.dumps(plan["schedule"], sort_keys=True)
        == json.dumps(schedule(plan["seed"]), sort_keys=True),
        "Frozen balanced schedule differs.",
    )
    require(
        json.dumps(plan["settings"], sort_keys=True)
        == json.dumps(SETTINGS, sort_keys=True)
        and plan["flags"] == FLAGS,
        "Frozen workload settings differ.",
    )
    require(
        type(plan["repetitions"]) is int
        and plan["repetitions"] == 4
        and plan["excluded_sample_indices"] == [0]
        and type(plan["excluded_sample_indices"][0]) is int,
        "Warmup rule differs.",
    )
    require(
        plan["harness_sha256"] == runtime.digest(Path(__file__)),
        "Harness changed after freeze.",
    )
    require(
        plan["runtime_helper_sha256"] == runtime.digest(HELPER),
        "Runtime helper changed after freeze.",
    )
    require(
        plan["host_helper_sha256"] == runtime.digest(HOST_HELPER),
        "Host helper changed after freeze.",
    )
    host.check_policy(plan["host_policy"])
    require(
        set(plan["prerequisites"])
        == {"quality_baseline", "dispatch", "depth_state", "setup_pilot"},
        "The prerequisite evidence pins are incomplete.",
    )
    for key in (
        "timeout_seconds",
        "idle_before_session_seconds",
        "idle_between_pairs_seconds",
    ):
        value = plan[key]
        require(
            type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"Invalid {key}.",
        )
    require(plan["timeout_seconds"] > 0, "A positive process deadline is required.")
    expected = plan["expected_model"]
    require(
        set(expected) == {"model_type", "model_size", "model_n_params"},
        "Model output identity is incomplete.",
    )
    require(
        isinstance(expected["model_type"], str) and bool(expected["model_type"]),
        "Model type is required.",
    )
    for key in ("model_size", "model_n_params"):
        require(
            type(expected[key]) is int and expected[key] > 0,
            "Model tensor size/parameter count is invalid.",
        )


def freeze(spec_path, destination):
    spec = read_json(spec_path)
    study = spec.get("study", "controlled")
    require(type(study) is str and study in STUDIES, "Unknown benchmark study.")
    require(
        ("method" in spec) == (study == "desktop")
        and spec.get("kind", STUDIES[study]) == STUDIES[study],
        "Study kind and method differ.",
    )
    quality = read_json(Path(spec["prerequisites"]["quality_baseline"]))
    plan = {
        "schema_version": 1,
        "kind": STUDIES[study],
        "seed": spec["seed"],
        "schedule": schedule(spec["seed"]),
        "settings": SETTINGS,
        "flags": FLAGS,
        "repetitions": 4,
        "excluded_sample_indices": [0],
        "harness_sha256": runtime.digest(Path(__file__)),
        "runtime_helper_sha256": runtime.digest(HELPER),
        "host_helper_sha256": runtime.digest(HOST_HELPER),
        "host_baseline": asset(spec["host_baseline"]),
        "host_policy": host.policy_from(
            read_json(Path(spec["host_baseline"])),
            swap_activity_policy="record_counter_increases"
            if study == "desktop"
            else "exclude_any_counter_increase",
        ),
        "prerequisites": {
            key: asset(path) for key, path in spec["prerequisites"].items()
        },
        "quality_comparison": {
            key: quality[key] for key in ("quality_preserved", "backend_adoption")
        },
        "runtime_config": asset(spec["runtime_config"]),
        "model": {
            **spec["model"],
            "path": str(Path(spec["model"]["path"]).resolve(strict=True)),
        },
        "expected_model": spec["expected_model"],
        "timeout_seconds": spec["timeout_seconds"],
        "idle_before_session_seconds": spec["idle_before_session_seconds"],
        "idle_between_pairs_seconds": spec["idle_between_pairs_seconds"],
        "backends": {},
    }
    if study == "desktop":
        plan["method"] = asset(spec["method"])
    for name in BACKENDS:
        build = spec["backends"][name]
        binary = Path(build["binary"]).resolve(strict=True)
        plan["backends"][name] = {
            **{key: asset(build[key]) for key in ("binary", "cache", "build_record")},
            "libraries": {key: asset(path) for key, path in libraries(binary).items()},
        }
    check_plan(plan)
    verify_runtime(plan)
    save_json(destination, plan, create=True)
    return runtime.digest(destination)


def command(plan, row):
    prompt, generated, depth = WORKLOADS[row["workload"]]
    return [
        plan["backends"][row["backend"]]["binary"]["path"],
        "-m",
        plan["model"]["path"],
        "-p",
        str(prompt),
        "-n",
        str(generated),
        "-d",
        str(depth),
        *FLAGS,
    ]


def validate_samples(value, plan, row):
    require(
        isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict),
        "Expected exactly one benchmark row.",
    )
    value = value[0]
    p, n, d = WORKLOADS[row["workload"]]
    expected = {
        **SETTINGS,
        **plan["expected_model"],
        "build_commit": REVISION[:7],
        "model_filename": plan["model"]["path"],
        "backends": "CPU",
        "n_prompt": p,
        "n_gen": n,
        "n_depth": d,
    }
    for key, item in expected.items():
        require(
            type(value.get(key)) is type(item) and value[key] == item,
            f"Observed setting differs: {key}",
        )
    for key in ("build_number", "avg_ns", "stddev_ns"):
        require(
            type(value.get(key)) is int and value[key] >= 0,
            f"Invalid native field: {key}",
        )
    for key in ("cpu_info", "gpu_info", "test_time"):
        require(isinstance(value.get(key), str), f"Missing native field: {key}")
    for key in ("samples_ns", "samples_ts"):
        require(
            isinstance(value.get(key), list) and len(value[key]) == 4,
            "Expected four raw samples.",
        )
    samples = value["samples_ns"]
    require(
        all(type(x) is int and 0 < x < 2**64 for x in samples),
        "Invalid nanosecond samples.",
    )
    for duration, rate in zip(samples, value["samples_ts"]):
        require(
            type(rate) in (int, float) and math.isfinite(rate) and rate > 0,
            "Invalid throughput sample.",
        )
        require(
            math.isclose(rate, 1e9 * (p + n) / duration, rel_tol=6e-6, abs_tol=1e-6),
            "Throughput differs from raw timing.",
        )
    require(
        value["avg_ns"] == sum(samples) // 4, "Native mean differs from raw timing."
    )
    for key in ("avg_ts", "stddev_ts"):
        require(
            type(value.get(key)) in (int, float)
            and math.isfinite(value[key])
            and value[key] >= 0,
            f"Invalid native field: {key}",
        )
    return value


@contextlib.contextmanager
def exclusive_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another benchmark session owns the lock.") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def fingerprint(plan):
    return {
        item["path"]: (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        for item in all_assets(plan)
        for stat in (Path(item["path"]).stat(),)
    }


def timestamp():
    return datetime.datetime.now(datetime.UTC).isoformat()


def stop_child(process):
    if process is not None and process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            process.wait(timeout=CLEANUP_SECONDS)
            return
        try:
            process.wait(timeout=CLEANUP_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=CLEANUP_SECONDS)


def execute(plan, row, output, persist, *, popen=subprocess.Popen):
    directory = output / row["id"]
    directory.mkdir()
    row.update(state="attempted", started_at=timestamp(), argv=command(plan, row))
    persist()
    process, started, interrupted = None, time.monotonic(), None
    row["outcome"] = None
    try:
        with (
            (directory / "stdout.json").open("xb") as stdout,
            (directory / "stderr.log").open("xb") as stderr,
        ):
            process = popen(
                row["argv"],
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                env={"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"},
            )
            row["pid"] = process.pid
            persist()
            code = process.wait(timeout=plan["timeout_seconds"])
        row["returncode"] = code
        if code:
            row.update(
                outcome="exit_error", failure=f"Child exited with status {code}."
            )
        else:
            try:
                row["observed"] = validate_samples(
                    read_json(directory / "stdout.json"), plan, row
                )
                row["outcome"] = "success"
            except (
                ValueError,
                TypeError,
                KeyError,
                UnicodeError,
                RecursionError,
            ) as error:
                row.update(
                    outcome="invalid_output", failure=f"{type(error).__name__}: {error}"
                )
    except subprocess.TimeoutExpired:
        row.update(outcome="timeout", failure="The child exceeded its deadline.")
    except KeyboardInterrupt as error:
        row.update(outcome="interrupted", failure="The session was interrupted.")
        interrupted = error
    except OSError as error:
        row.update(
            outcome="launch_error" if process is None else "record_error",
            failure=f"{type(error).__name__}: {error}",
        )
    finally:
        try:
            stop_child(process)
        except (OSError, subprocess.TimeoutExpired) as error:
            row.update(
                outcome="cleanup_error", failure=f"{type(error).__name__}: {error}"
            )
            interrupted = error
        row.update(finished_at=timestamp(), wall_seconds=time.monotonic() - started)
        if process is not None:
            row["returncode"] = process.returncode
        try:
            row["artifacts"] = {}
            for name in ("stdout.json", "stderr.log"):
                path = directory / name
                if path.exists():
                    with path.open("rb") as stream:
                        os.fsync(stream.fileno())
                    row["artifacts"][name] = asset(path)
        except OSError as error:
            row.update(
                outcome="record_error", failure=f"{type(error).__name__}: {error}"
            )
            interrupted = error
        row["state"] = "finished"
        persist()
    if interrupted is not None:
        raise interrupted


def summarize(plan, run):
    desktop = plan["kind"] == STUDIES["desktop"]
    idle_policy = {
        **plan["host_policy"],
        "swap_activity_policy": "record_counter_increases",
    }
    planned = [row for row in plan["schedule"] if row["session"] == run["session"]]
    require(len(run["rows"]) == len(planned), "Ledger lost schedule rows.")
    for expected, row in zip(planned, run["rows"]):
        require(
            all(row[key] == value for key, value in expected.items()),
            "Ledger order differs.",
        )
        require(
            row["state"] in {"planned", "attempted", "finished"},
            "Unknown attempt state.",
        )
        if row["state"] != "planned":
            require(row["argv"] == command(plan, row), "Recorded command differs.")
        for item in row.get("artifacts", {}).values():
            verify_asset(item)
        if row.get("outcome") == "success":
            native = read_json(Path(row["artifacts"]["stdout.json"]["path"]))
            require(
                validate_samples(native, plan, row) == row["observed"],
                "Recorded samples differ from raw output.",
            )
    pairs = []
    previous_after = None
    for pair_id in dict.fromkeys(row["pair"] for row in planned):
        group = [row for row in run["rows"] if row["pair"] == pair_id]
        gap_findings = (
            host.interval_violations(
                previous_after, group[0]["host_before"], idle_policy
            )
            if previous_after is not None and "host_before" in group[0]
            else []
        )
        previous_after = group[-1].get("host_after")
        complete = run["integrity_verified"] and all(
            row["state"] == "finished" and row.get("outcome") == "success"
            for row in group
        )
        observations = [
            row[boundary]
            for row in group
            for boundary in ("host_before", "host_after")
            if boundary in row
        ]
        intervals = list(zip(observations, observations[1:]))
        if observations:
            intervals.append((observations[0], observations[-1]))
        host_valid = (
            not gap_findings
            and len(observations) == 2 * len(group)
            and all(
                not host.admit(snapshot, plan["host_policy"])
                for snapshot in observations
            )
        )
        host_valid = host_valid and all(
            not host.interval_violations(before, after, plan["host_policy"])
            for before, after in intervals
        )
        complete = complete and (desktop or host_valid)
        pair = {row["backend"]: row for row in group}
        item = {
            "pair": pair_id,
            "workload": group[0]["workload"],
            "included": bool(complete),
        }
        if desktop:
            findings = [
                reason
                for snapshot in observations
                for reason in host.admit(snapshot, plan["host_policy"])
            ]
            findings += gap_findings + [
                reason
                for before, after in intervals
                for reason in host.interval_violations(before, after)
            ]
            item.update(
                host_findings=sorted(set(findings)),
                host_protocol_passed=bool(host_valid),
            )
        if "host_before" in group[0] and "host_after" in group[-1]:
            item["host_deltas"] = host.deltas(
                group[0]["host_before"], group[-1]["host_after"]
            )
        if complete:
            samples = {
                name: row["observed"]["samples_ns"][1:] for name, row in pair.items()
            }
            durations = {
                name: (
                    statistics.mean(values) if desktop else statistics.median(values)
                )
                for name, values in samples.items()
            }
            item.update(
                {
                    "retained_samples_ns": samples,
                    "mean_ns" if desktop else "median_ns": durations,
                    "cpu_over_kleidiai": durations["cpu"] / durations["kleidiai"],
                }
            )
        else:
            item["exclusion"] = (
                "Incomplete/failed pair or unverified runtime integrity; "
                "no replacement or retry."
                if desktop
                else "Incomplete/failed pair, failed host admission "
                "or unverified runtime integrity; no replacement or retry."
            )
        pairs.append(item)
    rows = run["rows"]
    return {
        "planned": len(rows),
        "attempted": sum(row["state"] != "planned" for row in rows),
        "finished": sum(row["state"] == "finished" for row in rows),
        "unfinished": sum(row["state"] == "attempted" for row in rows),
        "successful": sum(
            row["state"] == "finished" and row.get("outcome") == "success"
            for row in rows
        ),
        "failed": sum(
            row["state"] == "finished" and row.get("outcome") != "success"
            for row in rows
        ),
        "outcomes": {
            key: sum(
                row["state"] == "finished" and row.get("outcome") == key for row in rows
            )
            for key in (
                "success",
                "invalid_output",
                "exit_error",
                "timeout",
                "interrupted",
                "launch_error",
                "cleanup_error",
                "record_error",
            )
        },
        "host_violations": run.get("host_violations", []),
        "quality_comparison": plan["quality_comparison"],
        "pairs": pairs,
        "warmup": "Sample index 0 retained but excluded; upstream warmup also enabled.",
        "limit": (
            "Nested samples within process pairs. Native deviation statistics "
            "are not trusted. Host checks cover invocation boundaries only. "
            "No dispatch, answer-quality or publication claim."
        ),
    }


def run_session(
    plan_path,
    expected_sha256,
    session,
    output,
    *,
    lock_path=None,
    popen=subprocess.Popen,
    sleep=time.sleep,
    collect_host=None,
):
    require(plan_path.stat().st_size <= MAX_JSON_BYTES, "Plan exceeds its size limit.")
    raw_plan = plan_path.read_bytes()
    require(
        hashlib.sha256(raw_plan).hexdigest() == expected_sha256,
        "Plan bytes changed after freeze.",
    )
    plan = json.loads(
        raw_plan,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
        parse_float=finite_float,
    )
    check_plan(plan)
    collect_host = collect_host or host.collect
    require(type(session) is int and 1 <= session <= 3, "Choose session 1, 2 or 3.")
    lock_path = lock_path or ROOT / "artifacts/benchmarks/benchmark.lock"
    with exclusive_lock(lock_path):
        output.mkdir()
        save_bytes(output / "plan.json", raw_plan, create=True)
        run = {
            "schema_version": 1,
            "plan_sha256": expected_sha256,
            "session": session,
            "state": "planned",
            "integrity_verified": False,
            "failure": None,
            "host_violations": [],
            "rows": [
                {**row, "state": "planned"}
                for row in plan["schedule"]
                if row["session"] == session
            ],
        }

        def persist():
            save_json(output / "run.json", run)

        persist()
        try:
            verify_runtime(plan)
            snapshot = fingerprint(plan)
            sleep(plan["idle_before_session_seconds"])
            run["state"] = "running"
            persist()
            last_pair = None
            pair_start = None
            previous = None
            idle_policy = {
                **plan["host_policy"],
                "swap_activity_policy": "record_counter_increases",
            }
            for row in run["rows"]:
                if last_pair is not None and row["pair"] != last_pair:
                    sleep(plan["idle_between_pairs_seconds"])
                require(
                    fingerprint(plan) == snapshot,
                    "Pinned runtime changed during the session.",
                )
                row["host_before"] = collect_host()
                new_pair = row["pair"] != last_pair
                if new_pair:
                    pair_start = row["host_before"]
                if previous is None:
                    previous = pair_start
                violations = host.admit(row["host_before"], plan["host_policy"])
                violations += host.interval_violations(
                    pair_start, row["host_before"], plan["host_policy"]
                )
                violations += host.interval_violations(
                    previous,
                    row["host_before"],
                    idle_policy if new_pair else plan["host_policy"],
                )
                violations = list(dict.fromkeys(violations))
                if violations:
                    run["host_violations"].append(
                        {
                            "pair": row["pair"],
                            "row": row["id"],
                            "boundary": "before",
                            "reasons": violations,
                        }
                    )
                    persist()
                    break
                persist()
                try:
                    execute(plan, row, output, persist, popen=popen)
                finally:
                    row["host_after"] = collect_host()
                    row["host_deltas"] = host.deltas(
                        row["host_before"], row["host_after"]
                    )
                    violations = host.admit(row["host_after"], plan["host_policy"])
                    violations += host.interval_violations(
                        pair_start, row["host_after"], plan["host_policy"]
                    )
                    violations += host.interval_violations(
                        row["host_before"], row["host_after"], plan["host_policy"]
                    )
                    violations = list(dict.fromkeys(violations))
                    previous = row["host_after"]
                    if violations:
                        run["host_violations"].append(
                            {
                                "pair": row["pair"],
                                "row": row["id"],
                                "boundary": "after",
                                "reasons": violations,
                            }
                        )
                    persist()
                if violations:
                    break
                last_pair = row["pair"]
            verify_runtime(plan)
            run.update(
                state="host_blocked" if run["host_violations"] else "finished",
                integrity_verified=True,
            )
        except BaseException as error:
            run.update(
                state="interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                failure=f"{type(error).__name__}: {error}",
            )
            raise
        finally:
            persist()
            save_json(output / "summary.json", summarize(plan, run), create=True)
        return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("freeze")
    prepare.add_argument("--spec", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--plan-sha256", required=True)
    run.add_argument("--session", type=int, choices=(1, 2, 3), required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        print(freeze(args.spec, args.output))
    else:
        result = run_session(args.plan, args.plan_sha256, args.session, args.output)
        if result["state"] != "finished" or any(
            row.get("outcome") != "success" for row in result["rows"]
        ):
            parser.exit(1, "Session checks failed; inspect the retained records.\n")


if __name__ == "__main__":
    main()
