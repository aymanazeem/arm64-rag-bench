import copy
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location(
    "controlled_fixed_bench", DIRECTORY / "benchmark.py"
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
SAMPLE = [
    {
        "build_commit": "8ca5a80",
        "build_number": 1,
        "cpu_info": "Synthetic CPU",
        "gpu_info": "",
        "backends": "CPU",
        "model_filename": "synthetic.gguf",
        "model_type": "synthetic Q8_0",
        "model_size": 128,
        "model_n_params": 64,
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
        "n_prompt": 512,
        "n_gen": 0,
        "n_depth": 0,
        "test_time": "2000-01-01T00:00:00Z",
        "avg_ns": 2500000000,
        "stddev_ns": 0,
        "avg_ts": 266.666667,
        "stddev_ts": 0.0,
        "samples_ns": [1000000000, 2000000000, 3000000000, 4000000000],
        "samples_ts": [512.0, 256.0, 170.667, 128.0],
    }
]


def stable_host():
    available = bench.host.available
    return {
        "schema_version": 1,
        "identity": {
            "system": "Darwin",
            "release": "synthetic",
            "machine": "arm64",
            "hardware": {
                "model": "Synthetic",
                "physicalcpu": 10,
                "logicalcpu": 10,
                "memsize": 64 * 1024**3,
            },
        },
        "power_source": available("AC Power"),
        "power_mode": available({"powermode": 0}),
        "memory_pressure": available({"code": 1, "label": "normal"}),
        "thermal": bench.host.unavailable("cpu_limits_not_reported"),
        "load_average": available([7.0, 6.0, 5.0]),
        "swap": available({"total_bytes": 12e9, "used_bytes": 11e9, "free_bytes": 1e9}),
        "pages": available(
            {
                "page_size_bytes": 16384,
                "Pageins": 100,
                "Pageouts": 20,
                "Swapins": 5,
                "Swapouts": 10,
            }
        ),
    }


@pytest.fixture(autouse=True)
def synthetic_host(monkeypatch):
    monkeypatch.setattr(bench.host, "collect", stable_host)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def frozen(tmp_path):
    model = tmp_path / "weights sample.gguf"
    model.write_bytes(b"synthetic weights")
    config = {
        "llama_cpp": {
            "revision": bench.REVISION,
            "repository": "https://example.invalid/pinned",
        },
        "kleidiai": {"version": "1.32.0", "sha256": "a" * 64},
        "cmake": {
            "CMAKE_BUILD_TYPE": "Release",
            "GGML_CPU": "ON",
            "GGML_CPU_REPACK": "ON",
            "GGML_METAL": "OFF",
            "GGML_BLAS": "OFF",
            "GGML_ACCELERATE": "OFF",
            "GGML_OPENMP": "OFF",
        },
    }
    config_path = tmp_path / "runtime.json"
    write(config_path, config)
    builds = {}
    for backend in bench.BACKENDS:
        directory = tmp_path / backend
        directory.mkdir()
        binary = directory / "llama-bench"
        binary.write_text(
            f"#!{sys.executable}\n"
            "import json,sys\n"
            f"value = {SAMPLE!r}[0]\n"
            "def option(name): return sys.argv[sys.argv.index(name)+1]\n"
            "value.update(model_filename=option('-m'),n_prompt=int(option('-p')),n_gen=int(option('-n')),n_depth=int(option('-d')))\n"
            f"factor = {2 if backend == 'cpu' else 1}\n"
            "value['samples_ns'] = [v*factor for v in value['samples_ns']]\n"
            "value['avg_ns'] = sum(value['samples_ns'])//4\n"
            "tokens = value['n_prompt']+value['n_gen']\n"
            "value['samples_ts'] = [float(format(1e9*tokens/v,'.6g')) "
            "for v in value['samples_ns']]\n"
            "print(json.dumps([value]))\n"
        )
        binary.chmod(0o755)
        library = directory / "libsynthetic.dylib"
        library.write_bytes(backend.encode())
        flags = {
            **config["cmake"],
            "GGML_CPU_KLEIDIAI": "ON" if backend == "kleidiai" else "OFF",
        }
        cache = directory / "CMakeCache.txt"
        cache.write_text(
            "\n".join(
                f"{key}:STRING={value}"
                for key, value in {
                    **flags,
                    "GGML_AVAILABLE_BACKENDS": "ggml-cpu",
                }.items()
            )
        )
        record = directory / "build.json"
        write(
            record,
            {
                "backend": backend,
                "flags": flags,
                "source": config["llama_cpp"],
                "kleidiai": config["kleidiai"] if backend == "kleidiai" else None,
                "targets": {"llama-bench": bench.runtime.digest(binary)},
                "cache_sha256": bench.runtime.digest(cache),
                "backends": ["ggml-cpu"],
                "libraries": {library.name: bench.runtime.digest(library)},
                "compiler_flags": {"CMAKE_C_FLAGS": ""},
                "compilers": {"C": "Synthetic compiler"},
                "tools": {"cmake": "synthetic"},
                "system": {"os": "Synthetic OS"},
            },
        )
        builds[backend] = {
            "binary": str(binary),
            "cache": str(cache),
            "build_record": str(record),
        }
    spec_path = tmp_path / "spec.json"
    host_baseline = tmp_path / "host.json"
    write(host_baseline, stable_host())
    prerequisites = {}
    for name in ("quality_baseline", "dispatch", "depth_state", "setup_pilot"):
        path = tmp_path / (name + ".json")
        write(
            path,
            {
                "synthetic_evidence": name,
                "purpose": name,
                "status": "approved_for_measurement",
                "quality_preserved": True,
                "backend_adoption": False,
                "model_sha256": bench.runtime.digest(model),
                "timing_builds": {
                    backend: bench.runtime.digest(Path(build["build_record"]))
                    for backend, build in builds.items()
                },
            },
        )
        prerequisites[name] = str(path)
    write(
        spec_path,
        {
            "runtime_config": str(config_path),
            "model": bench.asset(model),
            "expected_model": {
                key: SAMPLE[0][key]
                for key in ("model_type", "model_size", "model_n_params")
            },
            "backends": builds,
            "seed": 42,
            "timeout_seconds": 30,
            "idle_before_session_seconds": 0,
            "idle_between_pairs_seconds": 0,
            "host_baseline": str(host_baseline),
            "prerequisites": prerequisites,
        },
    )
    plan_path = tmp_path / "plan.json"
    digest = bench.freeze(spec_path, plan_path)
    return plan_path, digest, tmp_path


def run(frozen, **kwargs):
    path, digest, root = frozen
    return bench.run_session(
        path, digest, 1, root / "result", lock_path=root / "exclusive.lock", **kwargs
    )


def refreeze(frozen, mutate):
    path, _, root = frozen
    value = bench.read_json(path)
    mutate(value)
    write(path, value)
    return path, bench.runtime.digest(path), root


def test_schedule_balances_each_session_and_matches_workload_order():
    rows = bench.schedule(42)
    assert len(rows) == 48 and len({row["id"] for row in rows}) == 48
    assert rows == bench.schedule(42) and rows != bench.schedule(43)
    for session in range(1, 4):
        groups = [
            [row for row in rows if row["session"] == session and row["round"] == n]
            for n in (1, 2)
        ]
        assert {group[0]["backend"] for group in groups} == set(bench.BACKENDS)
        for group in groups:
            assert len(group) == 8
            for index in range(0, 8, 2):
                a, b = group[index : index + 2]
                assert a["pair"] == b["pair"] and a["workload"] == b["workload"]
                assert {a["backend"], b["backend"]} == set(bench.BACKENDS)
                assert a["backend"] == group[0]["backend"]
            assert set(row["workload"] for row in group) == set(bench.WORKLOADS)


def test_fake_executables_produce_durable_records_and_paired_summaries(frozen):
    path, _, root = frozen
    calls = []

    def launcher(argv, **kwargs):
        ledger = bench.read_json(root / "result/run.json")
        active = [row for row in ledger["rows"] if row["state"] == "attempted"]
        assert len(active) == 1 and active[0]["argv"] == argv
        assert set(kwargs["env"]) == {"PATH", "LANG", "LC_ALL"}
        assert "--no-warmup" not in argv
        assert argv[argv.index("-m") + 1].endswith("weights sample.gguf")
        calls.append(argv)
        return subprocess.Popen(argv, **kwargs)

    result = run(frozen, popen=launcher)
    assert len(calls) == 16
    assert result["state"] == "finished" and result["integrity_verified"]
    summary = bench.read_json(root / "result/summary.json")
    assert (
        summary["planned"],
        summary["attempted"],
        summary["finished"],
        summary["successful"],
    ) == (16, 16, 16, 16)
    assert all(pair["cpu_over_kleidiai"] == 2 for pair in summary["pairs"])
    assert all(
        len(pair["retained_samples_ns"]["cpu"]) == 3 for pair in summary["pairs"]
    )
    assert len(result["rows"][0]["observed"]["samples_ns"]) == 4
    assert bench.runtime.digest(root / "result/plan.json") == bench.runtime.digest(path)


def test_host_rejection_before_launch_retains_planned_denominator(frozen):
    observed = stable_host()
    observed["memory_pressure"] = bench.host.unavailable("probe_failed")

    def must_not_launch(*args, **kwargs):
        pytest.fail("Host rejection must happen before starting the child.")

    result = run(
        frozen, collect_host=lambda: copy.deepcopy(observed), popen=must_not_launch
    )
    summary = bench.read_json(frozen[2] / "result/summary.json")
    assert result["state"] == "host_blocked" and result["integrity_verified"]
    assert summary["planned"] == 16 and summary["attempted"] == 0
    assert len(summary["host_violations"]) == 1
    assert not any(pair["included"] for pair in summary["pairs"])


def test_host_change_after_attempt_keeps_samples_and_stops_new_work(frozen):
    observations = 0

    def observe():
        nonlocal observations
        observations += 1
        value = stable_host()
        if observations == 2:
            value["power_source"] = bench.host.available("Battery Power")
        return value

    result = run(frozen, collect_host=observe)
    summary = bench.read_json(frozen[2] / "result/summary.json")
    assert summary["attempted"] == summary["finished"] == summary["successful"] == 1
    assert result["rows"][0]["outcome"] == "success"
    assert len(result["rows"][0]["observed"]["samples_ns"]) == 4
    assert sum(row["state"] == "planned" for row in result["rows"]) == 15
    assert not any(pair["included"] for pair in summary["pairs"])


def test_later_pair_host_rejection_preserves_earlier_valid_pair(frozen):
    observations = 0

    def observe():
        nonlocal observations
        observations += 1
        value = stable_host()
        if observations == 5:
            value["power_mode"] = bench.host.available({"powermode": 1})
        return value

    result = run(frozen, collect_host=observe)
    summary = bench.read_json(frozen[2] / "result/summary.json")
    assert summary["attempted"] == summary["finished"] == summary["successful"] == 2
    assert sum(pair["included"] for pair in summary["pairs"]) == 1
    assert result["state"] == "host_blocked"


def test_swap_activity_excludes_complete_pair_without_erasing_successes(frozen):
    observations = 0

    def observe():
        nonlocal observations
        observations += 1
        value = stable_host()
        if observations == 4:
            value["pages"]["value"]["Swapouts"] += 1
        return value

    result = run(frozen, collect_host=observe)
    summary = bench.read_json(frozen[2] / "result/summary.json")
    assert summary["successful"] == summary["attempted"] == 2
    assert result["host_violations"][0]["reasons"] == ["swap_activity"]
    assert not any(pair["included"] for pair in summary["pairs"])


def test_swap_between_pair_sides_blocks_second_launch(frozen):
    observations = 0

    def observe():
        nonlocal observations
        observations += 1
        value = stable_host()
        if observations == 3:
            value["pages"]["value"]["Swapins"] += 1
        return value

    result = run(frozen, collect_host=observe)
    summary = bench.read_json(frozen[2] / "result/summary.json")
    assert summary["successful"] == summary["attempted"] == 1
    assert result["host_violations"][0]["boundary"] == "before"
    assert result["host_violations"][0]["reasons"] == ["swap_activity"]
    assert not any(pair["included"] for pair in summary["pairs"])


def test_final_host_violation_keeps_native_successes_but_cli_fails(frozen, monkeypatch):
    observations = 0

    def observe():
        nonlocal observations
        observations += 1
        value = stable_host()
        if observations == 32:
            value["memory_pressure"] = bench.host.available(
                {"code": 2, "label": "warning"}
            )
        return value

    result = run(frozen, collect_host=observe)
    summary = bench.read_json(frozen[2] / "result/summary.json")
    assert summary["attempted"] == summary["finished"] == summary["successful"] == 16
    assert result["state"] == "host_blocked"
    assert sum(pair["included"] for pair in summary["pairs"]) == 7
    monkeypatch.setattr(bench, "run_session", lambda *args, **kwargs: result)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "run",
            "--plan",
            str(frozen[0]),
            "--plan-sha256",
            frozen[1],
            "--session",
            "1",
            "--output",
            str(frozen[2] / "unused"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 1


@pytest.mark.parametrize(
    "change", ["pending", "wrong_model", "wrong_build", "wrong_purpose"]
)
def test_unapproved_or_stale_prerequisite_blocks_plan_before_launch(frozen, change):
    root = frozen[2]
    path = root / "dispatch.json"
    value = bench.read_json(path)
    if change == "pending":
        value["status"] = "pending"
    elif change == "wrong_model":
        value["model_sha256"] = "0" * 64
    elif change == "wrong_purpose":
        value["purpose"] = "setup_pilot"
    else:
        value["timing_builds"]["cpu"] = "0" * 64
    write(path, value)
    with pytest.raises(ValueError, match="Prerequisite"):
        bench.freeze(root / "spec.json", root / "rejected-plan.json")
    assert not (root / "rejected-plan.json").exists()


def test_reviewed_quality_regression_can_be_measured_without_backend_adoption(frozen):
    root = frozen[2]
    path = root / "quality_baseline.json"
    value = bench.read_json(path)
    value.update(quality_preserved=False, backend_adoption=False)
    write(path, value)
    plan = root / "negative-quality-plan.json"
    digest = bench.freeze(root / "spec.json", plan)
    run((plan, digest, root))
    summary = bench.read_json(root / "result/summary.json")
    assert summary["successful"] == 16
    assert summary["quality_comparison"] == {
        "quality_preserved": False,
        "backend_adoption": False,
    }


@pytest.mark.parametrize(
    "change",
    [
        "plan_bytes",
        "schedule",
        "settings",
        "warmup",
        "helper",
        "model",
        "binary",
        "library",
        "cache",
        "extra_library",
    ],
)
def test_changed_pins_prevent_any_child_launch(frozen, change):
    path, digest, root = frozen
    plan = bench.read_json(path)
    if change == "plan_bytes":
        path.write_bytes(path.read_bytes() + b" ")
    elif change in {"schedule", "settings", "warmup", "helper"}:

        def mutate(value):
            if change == "schedule":
                value["schedule"][0], value["schedule"][1] = (
                    value["schedule"][1],
                    value["schedule"][0],
                )
            elif change == "settings":
                value["settings"]["n_threads"] = 8
            elif change == "warmup":
                value["excluded_sample_indices"] = []
            else:
                value["runtime_helper_sha256"] = "0" * 64

        frozen = refreeze(frozen, mutate)
    else:
        target = {
            "model": plan["model"]["path"],
            "binary": plan["backends"]["cpu"]["binary"]["path"],
            "library": plan["backends"]["cpu"]["libraries"]["libsynthetic.dylib"][
                "path"
            ],
            "cache": plan["backends"]["cpu"]["cache"]["path"],
            "extra_library": str(root / "cpu/new.dylib"),
        }[change]
        with Path(target).open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError):
        run(frozen, popen=lambda *a, **k: pytest.fail("A child must not launch."))


def test_new_output_directory_and_exclusive_lock_preserve_previous_data(frozen):
    _, _, root = frozen
    output = root / "result"
    output.mkdir()
    (output / "keep").write_text("previous result")
    with pytest.raises(FileExistsError):
        run(frozen)
    assert (output / "keep").read_text() == "previous result"
    with bench.exclusive_lock(root / "exclusive.lock"):
        with pytest.raises(RuntimeError, match="owns the lock"):
            bench.run_session(
                frozen[0], frozen[1], 1, root / "new", lock_path=root / "exclusive.lock"
            )
    assert not (root / "new").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_row",
        "short",
        "bool_ns",
        "nan_rate",
        "wrong_rate",
        "bool_flag",
        "wrong_depth",
        "wrong_model",
        "wrong_backend",
        "mean",
        "duplicate",
    ],
)
def test_upstream_output_validation_rejects_inconsistent_samples(frozen, mutation):
    plan = bench.read_json(frozen[0])
    row = {"workload": "pp512"}
    value = copy.deepcopy(SAMPLE)
    value[0]["model_filename"] = plan["model"]["path"]
    if mutation == "extra_row":
        value.append(value[0])
    elif mutation == "short":
        value[0]["samples_ns"].pop()
    elif mutation == "bool_ns":
        value[0]["samples_ns"][0] = True
    elif mutation == "nan_rate":
        value[0]["samples_ts"][0] = float("nan")
    elif mutation == "wrong_rate":
        value[0]["samples_ts"][0] *= 2
    elif mutation == "bool_flag":
        value[0]["flash_attn"] = False
    elif mutation == "wrong_depth":
        value[0]["n_depth"] = 1
    elif mutation == "wrong_model":
        value[0]["model_size"] = plan["model"]["size"]
    elif mutation == "wrong_backend":
        value[0]["backends"] = "CPU,Metal"
    elif mutation == "mean":
        value[0]["avg_ns"] += 1
    else:
        path = frozen[2] / "duplicate.json"
        path.write_text('{"n_prompt":512,"n_prompt":0}')
        with pytest.raises(ValueError, match="Duplicate"):
            bench.read_json(path)
        return
    with pytest.raises(ValueError):
        bench.validate_samples(value, plan, row)


class FakeChild:
    pid = 987654

    def __init__(self, *, error=None, kill_required=False):
        self.returncode = None
        self.error = error
        self.kill_required = kill_required
        self.terminated = self.killed = False

    def wait(self, timeout):
        if self.returncode is not None:
            return self.returncode
        if self.terminated and not self.kill_required:
            self.returncode = -15
            return self.returncode
        if self.error is not None:
            raise self.error
        self.returncode = 0
        return 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.returncode = -9


def test_timeout_cleanup_retains_every_attempt_and_excludes_failed_pairs(frozen):
    children = []

    def launcher(argv, **kwargs):
        kwargs["stdout"].write(b"partial output")
        child = FakeChild(
            error=subprocess.TimeoutExpired(argv, 0.1), kill_required=True
        )
        children.append(child)
        return child

    run(frozen, popen=launcher)
    summary = bench.read_json(frozen[2] / "result/summary.json")
    assert len(children) == 16 and all(
        child.terminated and child.killed for child in children
    )
    assert summary["attempted"] == summary["finished"] == summary["failed"] == 16
    assert summary["outcomes"]["timeout"] == 16
    assert not any(pair["included"] for pair in summary["pairs"])
    assert len(list((frozen[2] / "result").glob("*/stdout.json"))) == 16


def test_keyboard_interrupt_is_retained_and_reraised_without_extra_work(frozen):
    child = FakeChild(error=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        run(frozen, popen=lambda *args, **kwargs: child)
    result = bench.read_json(frozen[2] / "result/run.json")
    assert result["state"] == "interrupted"
    assert result["rows"][0]["outcome"] == "interrupted" and child.terminated
    assert all(row["state"] == "planned" for row in result["rows"][1:])
    with bench.exclusive_lock(frozen[2] / "exclusive.lock"):
        pass


@pytest.mark.parametrize(
    "mode",
    [
        "exit",
        "malformed",
        "oversized",
        "launch",
        "nonfinite_extra",
        "overflow_extra",
        "nested",
    ],
)
def test_failure_records_survive_and_do_not_become_valid_timings(frozen, mode):
    def launcher(argv, **kwargs):
        if mode == "launch":
            raise OSError("synthetic launch failure")
        raw = b"x" * (bench.MAX_JSON_BYTES + 1) if mode == "oversized" else b"not-json"
        if mode == "nonfinite_extra":
            value = copy.deepcopy(SAMPLE)
            value[0]["extra"] = float("nan")
            raw = json.dumps(value).encode()
        elif mode == "overflow_extra":
            value = copy.deepcopy(SAMPLE)
            value[0]["extra"] = "OVERFLOW"
            raw = json.dumps(value).replace('"OVERFLOW"', "1e999").encode()
        elif mode == "nested":
            raw = b"[" * 2000 + b"0" + b"]" * 2000
        kwargs["stdout"].write(raw)
        child = FakeChild()
        child.returncode = 4 if mode == "exit" else 0
        return child

    result = run(frozen, popen=launcher)
    expected = {
        "exit": "exit_error",
        "malformed": "invalid_output",
        "oversized": "invalid_output",
        "launch": "launch_error",
        "nonfinite_extra": "invalid_output",
        "overflow_extra": "invalid_output",
        "nested": "invalid_output",
    }[mode]
    assert all(row["outcome"] == expected for row in result["rows"])
    summary = bench.read_json(frozen[2] / "result/summary.json")
    assert summary["failed"] == 16 and summary["successful"] == 0


def test_runtime_mutation_stops_new_work_and_preserves_partial_ledger(frozen):
    plan = bench.read_json(frozen[0])

    def launcher(argv, **kwargs):
        value = copy.deepcopy(SAMPLE)
        p, n, d = bench.WORKLOADS[plan["schedule"][0]["workload"]]
        value[0].update(
            model_filename=plan["model"]["path"], n_prompt=p, n_gen=n, n_depth=d
        )
        value[0]["samples_ts"] = [1e9 * (p + n) / x for x in value[0]["samples_ns"]]
        kwargs["stdout"].write(json.dumps(value).encode())
        Path(plan["model"]["path"]).write_bytes(b"changed")
        return FakeChild()

    with pytest.raises(ValueError, match="changed during"):
        run(frozen, popen=launcher)
    summary = bench.read_json(frozen[2] / "result/summary.json")
    assert summary["attempted"] == summary["finished"] == 1
    assert not any(pair["included"] for pair in summary["pairs"])


def test_corrupt_raw_evidence_cannot_be_summarized(frozen):
    result = run(frozen)
    first = Path(result["rows"][0]["artifacts"]["stdout.json"]["path"])
    first.write_bytes(b"[]")
    with pytest.raises(ValueError, match="Checksum"):
        bench.summarize(bench.read_json(frozen[0]), result)


def test_freeze_is_create_only(frozen):
    before = frozen[0].read_bytes()
    with pytest.raises(FileExistsError):
        bench.freeze(frozen[2] / "spec.json", frozen[0])
    assert frozen[0].read_bytes() == before


def test_exact_plan_bytes_are_archived_even_with_different_json_whitespace(frozen):
    path, _, root = frozen
    path.write_bytes(json.dumps(bench.read_json(path), separators=(",", ":")).encode())
    run((path, bench.runtime.digest(path), root))
    assert (root / "result/plan.json").read_bytes() == path.read_bytes()


@pytest.mark.parametrize("field", ["settings", "schedule", "excluded_sample_indices"])
def test_boolean_integer_substitution_is_rejected_in_frozen_plan(frozen, field):
    def mutate(plan):
        if field == "settings":
            plan[field]["flash_attn"] = False
        elif field == "schedule":
            plan[field][0]["session"] = True
        else:
            plan[field] = [False]

    with pytest.raises(ValueError):
        run(
            refreeze(frozen, mutate),
            popen=lambda *a, **k: pytest.fail("Must not launch."),
        )


def test_real_fake_process_deadline_kills_only_owned_child(frozen, monkeypatch):
    path, _, root = frozen
    plan = bench.read_json(path)
    row = {**plan["schedule"][0], "state": "planned"}
    sleeper = root / "ignore-termination"
    sleeper.write_text(
        f"#!{sys.executable}\n"
        "import signal,time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('started', flush=True)\n"
        "time.sleep(60)\n"
    )
    sleeper.chmod(0o755)
    plan["backends"][row["backend"]]["binary"]["path"] = str(sleeper)
    plan["timeout_seconds"] = 0.1
    monkeypatch.setattr(bench, "CLEANUP_SECONDS", 0.05)
    output = root / "deadline"
    output.mkdir()
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])

    def launcher(argv, **kwargs):
        child = subprocess.Popen(argv, **kwargs)
        deadline = time.monotonic() + 5
        marker = output / row["id"] / "stdout.json"
        while marker.stat().st_size == 0 and child.poll() is None:
            if time.monotonic() >= deadline:
                bench.stop_child(child)
                pytest.fail("The synthetic child did not install its signal handler.")
            time.sleep(0.01)
        return child

    try:
        start = time.monotonic()
        bench.execute(
            plan,
            row,
            output,
            lambda: bench.save_json(output / "record.json", row),
            popen=launcher,
        )
        assert time.monotonic() - start < 7
        assert row["outcome"] == "timeout" and row["returncode"] == -9
        assert (output / row["id"] / "stdout.json").read_bytes() == b"started\n"
        assert other.poll() is None
        with pytest.raises(ProcessLookupError):
            os.kill(row["pid"], 0)
    finally:
        other.terminate()
        other.wait(timeout=2)


def test_persistence_failure_after_launch_cleans_up_and_keeps_failure(frozen):
    path, _, root = frozen
    plan = bench.read_json(path)
    row = {**plan["schedule"][0], "state": "planned"}
    output = root / "persistence"
    output.mkdir()
    child = FakeChild()
    calls = 0

    def persist():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("Synthetic ledger failure after spawn")
        bench.save_json(output / "record.json", row)

    bench.execute(plan, row, output, persist, popen=lambda *a, **k: child)
    retained = bench.read_json(output / "record.json")
    assert child.terminated and child.returncode == -15
    assert retained["state"] == "finished" and retained["outcome"] == "record_error"
    assert "ledger failure" in retained["failure"]


def test_unresolved_attempt_remains_in_denominator(frozen):
    plan = bench.read_json(frozen[0])
    rows = [
        {**row, "state": "planned"} for row in plan["schedule"] if row["session"] == 1
    ]
    rows[0].update(
        state="attempted", argv=bench.command(plan, rows[0]), outcome="launch_error"
    )
    summary = bench.summarize(
        plan, {"session": 1, "rows": rows, "integrity_verified": False}
    )
    assert (
        summary["attempted"] == 1 and summary["finished"] == summary["successful"] == 0
    )
    assert summary["unfinished"] == 1
    assert summary["failed"] == 0
    assert all(count == 0 for count in summary["outcomes"].values())
    assert not any(pair["included"] for pair in summary["pairs"])


@pytest.mark.parametrize(
    "constant", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"]
)
def test_nonfinite_json_constants_are_rejected_even_in_extra_fields(tmp_path, constant):
    path = tmp_path / "invalid.json"
    path.write_text('{"extra":' + constant + "}")
    with pytest.raises(ValueError, match="Non-finite"):
        bench.read_json(path)
