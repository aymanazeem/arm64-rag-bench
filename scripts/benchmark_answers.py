"""Measure the buffered CPU application and replay its exact request inputs."""

import argparse
import hashlib
import json
import math
import os
import random
import resource
import signal
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import benchmark as common
import benchmark_report as public

ROOT = Path(__file__).resolve().parents[1]
KIND = "buffered-answer-performance-v1"
SELECTED = ("dev-014", "val-015", "val-006")
COMPATIBILITY_CASE = "val-015"
EMPTY_ID = "empty-retrieval"
EMPTY_QUERY = "applicationbenchmarkemptyretrieval9fa83b2c"
DATASETS = ("development.json", "retrieval-validation.json", "answer-validation.json")
STAGES = ("cli", "fixed")
SETTINGS = {
    "blocks": 3,
    "seed": 42,
    "idle_seconds": 120,
    "worker_timeout_seconds": 600,
    "startup_timeout_seconds": 120,
    "request_timeout_seconds": 300,
    "output_mode": "plain_stdout_pipe_with_explicit_post_return_flush",
    "backend": "cpu",
    "retrieval_method": "lexical",
    "fixed_input_selection": "first_planned_cli_observation_per_case",
}
LIMITS = [
    "These are known regression questions, not fresh accuracy evaluation.",
    "The application buffers its answer. Model time to first token is unobserved.",
    "Fresh processes do not imply cold file caches. Model verification reads the file.",
    "The stages are descriptive observations, not paired speed comparisons.",
    "Boundary host observations do not establish constant conditions between checks.",
    "Missing thermal telemetry is unavailable, not evidence of normal temperature.",
    "Resource peaks are separate process lifetimes, not additive memory usage.",
    "Structural validation does not establish factual completeness or support.",
    "The benchmark makes the native server inherit the worker process group.",
    "Command durations start at the parent attempt checkpoint, before ledger I/O.",
    "They include worker creation, the ownership handshake and worker imports.",
]
RAW_FILES = {
    "input.json",
    "stdout.txt",
    "stderr.txt",
    "answer.json",
    "answer.server.log",
    "answer.json.server.log",
    "observation.json",
    "observer-end.json",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def continuous_ns():
    require(sys.platform == "darwin", "Application measurements require macOS.")
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def input_identity(record):
    from arm64_rag.answers import _messages

    packed = record["packed"]
    schema = common.read_json(ROOT / "eval/answers-protocol.json")[
        "model_output_schema"
    ]
    messages = _messages(record["question"], packed["sources"], schema)
    retrieval = dict(record["retrieval"])
    retrieval.pop("generation", None)
    return {
        "question_sha256": digest(record["question"].encode()),
        "retrieval_sha256": digest(encoded(retrieval)),
        "packed_sha256": digest(encoded(packed)),
        "messages_sha256": digest(encoded(messages)),
        "schema_sha256": digest(encoded(schema)),
        "sources_sha256": digest(encoded(packed["sources"])),
        "prompt_sha256": digest(packed["prompt"].encode()),
        "token_ids_sha256": digest(encoded(packed["token_ids"])),
        "prompt_tokens": len(packed["token_ids"]),
        "max_output_tokens": packed["max_output_tokens"],
    }


def schedule(stage, *, compatibility=False):
    require(stage in STAGES, "Unknown measurement stage.")
    if compatibility:
        return [
            {
                "id": f"check-{stage}-{COMPATIBILITY_CASE}",
                "block": 1,
                "case_id": COMPATIBILITY_CASE,
                "state": "planned",
            }
        ]
    rng, rows = random.Random(SETTINGS["seed"]), []
    for block in range(1, SETTINGS["blocks"] + 1):
        cases = list(SELECTED) + ([EMPTY_ID] if stage == "cli" else [])
        rng.shuffle(cases)
        rows.extend(
            {
                "id": f"{stage}-{block}-{number}",
                "block": block,
                "case_id": case,
                "state": "planned",
            }
            for number, case in enumerate(cases, 1)
        )
    return rows


def public_workloads(index_path):
    import arm64_rag.index
    from arm64_rag.evaluation import _validate_contract
    from arm64_rag.index import index_inventory, search_index

    require(
        Path(arm64_rag.index.__file__).resolve() == ROOT / "src/arm64_rag/index.py",
        "Run with this checkout's locked Python environment.",
    )

    protocol = common.read_json(ROOT / "eval/answers-protocol.json")
    manifest_path = ROOT / protocol["manifest"]["path"]
    require(
        digest(manifest_path.read_bytes()) == protocol["manifest"]["sha256"],
        "Corpus manifest differs from the answer protocol.",
    )
    manifest = common.read_json(manifest_path)
    inventory = index_inventory(index_path, include_text=True)
    scopes = {item["id"]: item["source_ids"] for item in manifest["corpora"]}
    cases = {}
    allowed = {item["sha256"] for item in protocol["datasets"]}
    for name in DATASETS:
        path = ROOT / "eval" / name
        require(digest(path.read_bytes()) in allowed, "Question dataset changed.")
        dataset = common.read_json(path)
        _validate_contract(dataset, manifest, inventory)
        for case in dataset["cases"]:
            if case["id"] in SELECTED:
                cases[case["id"]] = {
                    "id": case["id"],
                    "question": case["question"],
                    "source_ids": scopes[case["corpus_id"]],
                    "dataset": "eval/" + name,
                    "required_facts": case["required_facts"],
                    "unacceptable_claims": case["unacceptable_claims"],
                }
    require(set(cases) == set(SELECTED), "Public workloads are missing.")
    for role in ("development", "regression"):
        path = ROOT / "docs/results/grounded-answers" / f"{role}.json"
        for case in common.read_json(path)["cases"]:
            if case["id"] in cases:
                cases[case["id"]]["published_prompt"] = case["prompt"]
    for case in cases.values():
        require("published_prompt" in case, "Published prompt identity is missing.")
    require(
        not search_index(index_path, EMPTY_QUERY, limit=20)["results"],
        "The empty-retrieval control unexpectedly matches a source.",
    )
    cases[EMPTY_ID] = {
        "id": EMPTY_ID,
        "question": EMPTY_QUERY,
        "source_ids": [],
        "required_facts": [],
        "unacceptable_claims": [],
        "published_prompt": None,
    }
    return cases, {key: inventory[key] for key in ("generation", "fingerprint")}


def source_paths():
    return [
        *sorted((ROOT / "src/arm64_rag").glob("*.py")),
        *(
            ROOT / "scripts" / name
            for name in (
                "benchmark_answers.py",
                "answer_benchmark_owner.py",
                "benchmark.py",
                "benchmark_host.py",
                "benchmark_timing.py",
                "benchmark_report.py",
                "runtime.py",
            )
        ),
        ROOT / "uv.lock",
        ROOT / "pyproject.toml",
        ROOT / "configs/answer-model.json",
        ROOT / "configs/runtime.json",
        ROOT / "eval/answers-protocol.json",
        ROOT / "data/manifests/corpus.json",
        *(ROOT / "eval" / name for name in DATASETS),
        ROOT / "docs/results/grounded-answers/development.json",
        ROOT / "docs/results/grounded-answers/regression.json",
    ]


def cpu_build():
    build_path = ROOT / "artifacts/feasibility/cpu/build.json"
    build = common.read_json(build_path)
    runtime = common.read_json(ROOT / "configs/runtime.json")
    flags = {**runtime["cmake"], "GGML_CPU_KLEIDIAI": "OFF"}
    require(
        build["backend"] == "cpu"
        and build["source"] == runtime["llama_cpp"]
        and build["flags"] == flags
        and build["kleidiai"] is None
        and build["backends"] == ["ggml-cpu"],
        "Use the pinned CPU-only build.",
    )
    require(
        all(
            flags.get(key) == value
            for key, value in {
                "GGML_CPU": "ON",
                "GGML_CPU_REPACK": "ON",
                "GGML_METAL": "OFF",
                "GGML_BLAS": "OFF",
                "GGML_ACCELERATE": "OFF",
                "GGML_OPENMP": "OFF",
            }.items()
        ),
        "CPU build acceleration settings changed.",
    )
    common.runtime.checked_cache(build_path.with_name("CMakeCache.txt"), flags)
    return build_path, build


def runtime_assets():
    config = common.read_json(ROOT / "configs/answer-model.json")
    build_path, build = cpu_build()
    binary = ROOT / "build/cpu/bin/llama-server"
    model = ROOT / config["directory"] / config["model"]["filename"]
    assets = {"binary": common.asset(binary), "model": common.asset(model)}
    require(
        assets["binary"]["sha256"] == build["targets"]["llama-server"]
        and assets["model"]["sha256"] == config["model"]["sha256"]
        and assets["model"]["size"] == config["model"]["size"],
        "CPU server or model differs from its pin.",
    )
    libraries = common.libraries(binary)
    require(set(libraries) == set(build["libraries"]), "Library inventory changed.")
    assets["libraries"] = {name: common.asset(path) for name, path in libraries.items()}
    require(
        all(
            item["sha256"] == build["libraries"][name]
            for name, item in assets["libraries"].items()
        ),
        "CPU library differs from its build record.",
    )
    assets["build"] = common.asset(build_path)
    for name, filename in (
        ("cache", "CMakeCache.txt"),
        ("compile_commands", "compile_commands.json"),
    ):
        assets[name] = common.asset(build_path.with_name(filename))
        require(
            assets[name]["sha256"] == build[name + "_sha256"],
            "Build metadata pin differs.",
        )
    return assets


def freeze(index_path, destination):
    cases, index = public_workloads(index_path)
    runtime = runtime_assets()
    snapshot = common.host.collect()
    plan = {
        "kind": KIND,
        "settings": SETTINGS,
        "workloads": cases,
        "schedules": {stage: schedule(stage) for stage in STAGES},
        "index": {"path": str(index_path.resolve()), **index},
        "sources": {
            str(path.relative_to(ROOT)): common.asset(path) for path in source_paths()
        },
        "runtime": runtime,
        "host_policy": common.host.policy_from(
            snapshot, swap_activity_policy="record_counter_increases"
        ),
        "timing_policy": common.timing.POLICY,
        "prepared_host": snapshot,
        "limits": LIMITS,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    common.save_json(destination, plan, create=True)
    return plan


def check_plan(plan, *, verify_model=True):
    from arm64_rag.index import index_inventory

    require(plan["kind"] == KIND and plan["settings"] == SETTINGS, "Plan changed.")
    require(
        plan["schedules"] == {stage: schedule(stage) for stage in STAGES},
        "Measurement schedule changed.",
    )
    common.timing.check_policy(plan["timing_policy"])
    common.host.check_policy(plan["host_policy"])
    require(
        plan["host_policy"]
        == common.host.policy_from(
            plan["prepared_host"], swap_activity_policy="record_counter_increases"
        ),
        "Host policy differs from its frozen observation.",
    )
    require(
        plan["host_policy"]["swap_activity_policy"] == "record_counter_increases",
        "This study records swap activity without selecting observations by it.",
    )
    require(
        set(plan["sources"]) == {str(p.relative_to(ROOT)) for p in source_paths()},
        "Application source inventory changed.",
    )
    for name, item in plan["sources"].items():
        require(
            Path(item["path"]).resolve() == (ROOT / name).resolve(), "Source moved."
        )
        common.verify_asset(item)
    assets = plan["runtime"]
    config = common.read_json(ROOT / "configs/answer-model.json")
    build_path, build = cpu_build()
    require(
        Path(assets["binary"]["path"]).resolve()
        == (ROOT / "build/cpu/bin/llama-server").resolve()
        and Path(assets["model"]["path"]).resolve()
        == (ROOT / config["directory"] / config["model"]["filename"]).resolve()
        and assets["binary"]["sha256"] == build["targets"]["llama-server"]
        and assets["model"]["sha256"] == config["model"]["sha256"]
        and assets["model"]["size"] == config["model"]["size"]
        and Path(assets["build"]["path"]).resolve() == build_path.resolve(),
        "Frozen runtime is not the pinned CPU build and answer model.",
    )
    require(
        set(common.libraries(Path(assets["binary"]["path"])))
        == set(assets["libraries"])
        == set(build["libraries"]),
        "CPU library inventory changed.",
    )
    require(
        all(
            item["sha256"] == build["libraries"][name]
            for name, item in assets["libraries"].items()
        ),
        "CPU library pin differs.",
    )
    for name, filename in (
        ("cache", "CMakeCache.txt"),
        ("compile_commands", "compile_commands.json"),
    ):
        require(
            assets[name]["sha256"] == build[name + "_sha256"]
            and Path(assets[name]["path"]).resolve()
            == build_path.with_name(filename).resolve(),
            "Frozen build metadata differs.",
        )
    for item in (
        assets["binary"],
        assets["build"],
        assets["cache"],
        assets["compile_commands"],
        *assets["libraries"].values(),
    ):
        common.verify_asset(item)
    if verify_model:
        common.verify_asset(assets["model"])
    inventory = index_inventory(Path(plan["index"]["path"]))
    require(
        all(
            inventory[key] == plan["index"][key]
            for key in ("generation", "fingerprint")
        ),
        "The local index changed after freezing.",
    )
    cases, _ = public_workloads(Path(plan["index"]["path"]))
    require(cases == plan["workloads"], "Frozen workload definitions changed.")
    return plan


def file_stamp(path):
    info = Path(path).stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def collection_identity(plan, plan_path, additional=()):
    from arm64_rag.index import index_inventory

    runtime = plan["runtime"]
    files = [
        plan_path,
        *(Path(item["path"]) for item in plan["sources"].values()),
        *(
            Path(runtime[name]["path"])
            for name in ("binary", "model", "build", "cache", "compile_commands")
        ),
        *(Path(item["path"]) for item in runtime["libraries"].values()),
        *common.libraries(Path(runtime["binary"]["path"])).values(),
        *additional,
    ]
    database = Path(plan["index"]["path"])
    stamps = {str(path): file_stamp(path) for path in files}
    stamps[str(database)] = file_stamp(database)
    wal = Path(str(database) + "-wal")
    stamps[str(wal)] = file_stamp(wal) if wal.exists() else None
    inventory = index_inventory(database)
    return {
        "files": stamps,
        "index": {key: inventory[key] for key in ("generation", "fingerprint")},
    }


def native_metrics(record):
    raw = record.get("raw_completion")
    if raw is None:
        return {"generated_tokens": 0, "cache_reused_tokens": 0}
    chat = raw.get("chat_response", {})
    timings = raw.get("timings", chat.get("timings", {}))
    cached = timings.get("cache_n")
    usage = chat.get("usage", {})
    if cached is None:
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
    return {
        "generated_tokens": raw.get("tokens_predicted"),
        "cache_reused_tokens": cached,
        "prompt_ms": timings.get("prompt_ms"),
        "predicted_ms": timings.get("predicted_ms"),
        "predicted_n": timings.get("predicted_n"),
        "predicted_per_second": timings.get("predicted_per_second"),
    }


def check_record(record, case, protocol):
    from arm64_rag.answers import validate_answer

    require(
        record.get("failure") is None and record.get("answer") is not None,
        "The application did not produce a structurally valid answer.",
    )
    require(record["question"] == case["question"], "Question changed.")
    if case["id"] == EMPTY_ID:
        require(
            not record["retrieval"]["results"]
            and not record["packed"]["sources"]
            and record["raw_completion"] is None
            and record["answer"] == {"status": "insufficient_evidence", "claims": []},
            "The empty control generated an answer or retrieved passages.",
        )
    else:
        expected = case["published_prompt"]
        identity = input_identity(record)
        require(
            identity["prompt_sha256"] == expected["prompt_sha256"]
            and identity["prompt_tokens"] == expected["prompt_tokens"]
            and identity["max_output_tokens"] == expected["max_output_tokens"],
            "Packed input differs from the public CPU workload.",
        )
        require(
            validate_answer(
                record["raw_completion"]["content"], record["packed"]["sources"]
            )
            == record["answer"],
            "Saved answer does not match the validated completion.",
        )
        require(
            native_metrics(record)["cache_reused_tokens"] == 0,
            "Unexpected or unverified prompt-cache reuse.",
        )
    require(
        record["packed"]["max_output_tokens"]
        == protocol["settings"]["max_output_tokens"],
        "Output budget changed.",
    )


def worker(payload):
    from arm64_rag import answers, cli, inference

    require(
        Path(answers.__file__).resolve() == ROOT / "src/arm64_rag/answers.py",
        "The worker imported another checkout's application.",
    )

    plan, row = payload["plan"], payload["row"]
    case = plan["workloads"][row["case_id"]]
    protocol = common.read_json(ROOT / "eval/answers-protocol.json")
    directory = Path(payload["directory"])
    observation = {"events": {}, "native_launches": 0, "failure": None}
    events = observation["events"]
    original_server, original_answer = inference.LocalServer, answers.answer_question
    original_subprocess = inference.subprocess
    completing = False
    native_process = None

    def popen(*args, **kwargs):
        nonlocal native_process
        require(
            kwargs.get("start_new_session") is True, "Native launch contract changed."
        )
        kwargs["start_new_session"] = False
        events["native_spawn_start_ns"] = continuous_ns()
        process = original_subprocess.Popen(*args, **kwargs)
        events["native_spawn_return_ns"] = continuous_ns()
        observation["native_launches"] += 1
        native_process = process
        return process

    class ObservedServer(original_server):
        def __enter__(self):
            events["server_initialization_start_ns"] = continuous_ns()
            result = super().__enter__()
            events["server_ready_ns"] = continuous_ns()
            return result

        def _render(self, body, *, deadline=None):
            value = super()._render(body, deadline=deadline)
            if completing and case["published_prompt"] is not None:
                require(
                    digest(value.encode()) == case["published_prompt"]["prompt_sha256"],
                    "Rendered request differs from the public workload.",
                )
            return value

        def complete(self, token_ids, max_tokens, *, json_schema=None, messages=None):
            nonlocal completing
            require(
                not completing and "complete_start_ns" not in events,
                "Only one generation is allowed per fresh worker.",
            )
            require(case["id"] != EMPTY_ID, "Empty control must not generate.")
            require(
                len(token_ids) == case["published_prompt"]["prompt_tokens"],
                "Input token count differs from the public workload.",
            )
            completing = True
            events["complete_start_ns"] = continuous_ns()
            try:
                return super().complete(
                    token_ids, max_tokens, json_schema=json_schema, messages=messages
                )
            finally:
                events["complete_end_ns"] = continuous_ns()
                completing = False

    def observed_answer(*args, **kwargs):
        events["answer_start_ns"] = continuous_ns()
        try:
            return original_answer(*args, **kwargs)
        finally:
            events["answer_end_ns"] = continuous_ns()

    inference.subprocess = SimpleNamespace(
        **{
            name: getattr(original_subprocess, name)
            for name in dir(original_subprocess)
            if not name.startswith("__")
        }
    )
    inference.subprocess.Popen = popen
    inference.LocalServer, answers.answer_question = ObservedServer, observed_answer
    exit_code = 0
    try:
        if payload["stage"] == "cli":
            sys.argv = [
                "arm64-rag",
                "ask",
                case["question"],
                "--db",
                plan["index"]["path"],
                "--method",
                "lexical",
                "--server-binary",
                plan["runtime"]["binary"]["path"],
                "--answer-model-config",
                str(ROOT / "configs/answer-model.json"),
                "--protocol",
                str(ROOT / "eval/answers-protocol.json"),
                "--trace",
                str(directory / "answer.json"),
            ]
            for source in case["source_ids"]:
                sys.argv.extend(["--source", source])
            cli.main()
        else:
            baseline = common.read_json(Path(payload["replay"]["path"]))
            common.verify_asset(payload["replay"])
            check_record(baseline, case, protocol)
            packed = baseline["packed"]
            with ObservedServer(
                Path(plan["runtime"]["binary"]["path"]),
                Path(plan["runtime"]["model"]["path"]),
                context_size=protocol["settings"]["context_size"],
                startup_timeout=120,
                request_timeout=300,
                reasoning_budget_tokens=protocol["settings"]["reasoning_budget_tokens"],
                log_path=directory / "answer.json.server.log",
            ) as server:
                record = {
                    key: baseline[key] for key in ("question", "retrieval", "packed")
                }
                record.update(
                    runtime=server.info(),
                    raw_completion=None,
                    answer=None,
                    failure=None,
                )
                try:
                    response = server.complete(
                        packed["token_ids"],
                        packed["max_output_tokens"],
                        json_schema=protocol["model_output_schema"],
                        messages=answers._messages(
                            case["question"],
                            packed["sources"],
                            protocol["model_output_schema"],
                        ),
                    )
                    record["raw_completion"] = response
                    events["validation_start_ns"] = continuous_ns()
                    record["answer"] = answers.validate_answer(
                        response["content"], packed["sources"]
                    )
                    events["validation_end_ns"] = continuous_ns()
                except (
                    OSError,
                    ValueError,
                    RuntimeError,
                    KeyError,
                    TypeError,
                ) as error:
                    record["failure"] = str(error)
                    if getattr(error, "response", None) is not None:
                        record["raw_completion"] = error.response
                common.save_json(directory / "answer.json", record, create=True)
                print(answers.render_answer(record))
                exit_code = int(record["failure"] is not None)
    except SystemExit as error:
        exit_code = (
            error.code if type(error.code) is int else int(error.code is not None)
        )
    except BaseException as error:
        observation["failure"] = f"{type(error).__name__}: {error}"
        exit_code = 1
    finally:
        events["application_return_ns"] = continuous_ns()
        sys.stdout.flush()
        events["stdout_flushed_ns"] = continuous_ns()
        observation["worker_peak_rss_bytes"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        observation["native_peak_rss_bytes"] = (
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            if observation["native_launches"] == 1
            and native_process is not None
            and native_process.returncode is not None
            else None
        )
        observation["native_reaped"] = (
            native_process is not None and native_process.returncode is not None
        )
        observation["exit_code"] = exit_code
        events["observer_export_start_ns"] = continuous_ns()
        common.save_json(directory / "observation.json", observation, create=True)
        common.save_json(
            directory / "observer-end.json",
            {"export_end_ns": continuous_ns()},
            create=True,
        )
        inference.LocalServer, answers.answer_question = (
            original_server,
            original_answer,
        )
        inference.subprocess = original_subprocess
    return exit_code


def record_files(directory):
    require(
        all(path.name in RAW_FILES for path in directory.iterdir()),
        "An unexpected file appeared in the observation directory.",
    )
    return {
        path.name: common.asset(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def snapshot_sources(plan, output, snapshot, persist):
    for name, original in plan["sources"].items():
        destination = output / "sources" / name
        require(
            destination.resolve().is_relative_to((output / "sources").resolve()),
            "Source snapshot path is outside its directory.",
        )
        raw = Path(original["path"]).read_bytes()
        require(
            len(raw) == original["size"] and digest(raw) == original["sha256"],
            "Source changed before it could be archived.",
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        common.save_bytes(destination, raw, create=True)
        before = file_stamp(destination)
        item = {
            "path": str(destination.resolve()),
            "size": len(raw),
            "sha256": digest(raw),
        }
        common.verify_asset(item)
        require(
            file_stamp(destination) == before,
            "Source snapshot changed while verifying.",
        )
        snapshot["files"][name] = item
        persist()
    snapshot["complete"] = True
    persist()


def verify_source_snapshot(plan, output, snapshot, *, evidence=None, required=False):
    require(
        set(snapshot["files"]) <= set(plan["sources"]), "Unexpected archived source."
    )
    if required or snapshot["complete"]:
        require(
            snapshot["complete"] and set(snapshot["files"]) == set(plan["sources"]),
            "Source snapshot is incomplete.",
        )
    for name, item in snapshot["files"].items():
        require(
            Path(item["path"]).resolve() == (output / "sources" / name).resolve()
            and public.public_asset(item) == public.public_asset(plan["sources"][name]),
            "Source snapshot differs from the frozen inventory.",
        )
        if evidence is None:
            common.verify_asset(item)
        else:
            pin = evidence.file(item["path"], expected=item["sha256"])
            require(pin["size"] == item["size"], "Archived source size changed.")


def safe_failure(error):
    return {"type": type(error).__name__, "message_sha256": digest(str(error).encode())}


def replay_inputs(plan, cli_path, *, compatibility=False, evidence=None):
    evidence = evidence or public.Evidence()
    run = load_run(
        plan, cli_path, "cli", compatibility=compatibility, evidence=evidence
    )
    require(
        run["state"] == "completed" and run["timing_valid"],
        "Fixed-context replay requires a complete valid CLI stage.",
    )
    selected, identities = {}, {}
    for row in run["rows"]:
        if row["case_id"] == EMPTY_ID:
            continue
        path = Path(row["files"]["answer.json"]["path"])
        record, _ = evidence.document(
            path, expected=row["files"]["answer.json"]["sha256"]
        )
        identity = input_identity(record)
        case_id = row["case_id"]
        require(
            case_id not in identities or identities[case_id] == identity,
            "CLI repeats have different packed inputs.",
        )
        identities[case_id] = identity
        selected.setdefault(case_id, row["files"]["answer.json"])
    require(
        set(selected) == ({COMPATIBILITY_CASE} if compatibility else set(SELECTED)),
        "Replay inputs are incomplete.",
    )
    evidence.check()
    return selected


def check_observer_intervals(events, row, exported):
    intervals = {
        "Observer export": [
            events.get("observer_export_start_ns"),
            exported.get("export_end_ns"),
            row.get("exit_observed_ns"),
        ]
    }
    if row["stage"] == "fixed":
        intervals["Fixed answer validation"] = [
            events.get("complete_end_ns"),
            events.get("validation_start_ns"),
            events.get("validation_end_ns"),
            events.get("application_return_ns"),
        ]
    for label, stamps in intervals.items():
        require(
            all(type(value) is int and value >= 0 for value in stamps)
            and stamps == sorted(stamps),
            f"{label} boundaries are inconsistent.",
        )


def check_observation(observation, row, record, plan, exported):
    events = observation["events"]
    require(
        observation["exit_code"] == 0 and observation["failure"] is None,
        "Worker reported an application failure.",
    )
    require(
        observation["native_launches"] == 1 and observation["native_reaped"],
        "Native memory cannot be attributed to one reaped server.",
    )
    for key in ("worker_peak_rss_bytes", "native_peak_rss_bytes"):
        require(
            type(observation[key]) is int and observation[key] > 0,
            "Process peak memory is unavailable.",
        )
    ordered = [
        "server_initialization_start_ns",
        "native_spawn_start_ns",
        "native_spawn_return_ns",
        "server_ready_ns",
    ]
    if row["case_id"] != EMPTY_ID:
        ordered.extend(["complete_start_ns", "complete_end_ns"])
    ordered.extend(
        ["application_return_ns", "stdout_flushed_ns", "observer_export_start_ns"]
    )
    stamps = [
        row["started_ns"],
        *(events.get(key) for key in ordered),
        row["exit_observed_ns"],
    ]
    require(
        all(type(value) is int and value >= 0 for value in stamps)
        and stamps == sorted(stamps),
        "Worker metric boundaries are inconsistent.",
    )
    check_observer_intervals(events, row, exported)
    require(
        row["started_ns"] <= row["first_stdout_ns"] <= row["exit_observed_ns"],
        "The answer was not observed before worker exit.",
    )
    if row["stage"] == "cli":
        require(
            events["server_ready_ns"]
            <= events["answer_start_ns"]
            <= events["answer_end_ns"]
            <= events["application_return_ns"],
            "Ready-server answer boundaries changed.",
        )
    if row["case_id"] == EMPTY_ID:
        require("complete_start_ns" not in events, "The empty control generated text.")
    runtime = record["runtime"]
    require(
        runtime["model_sha256"] == plan["runtime"]["model"]["sha256"]
        and runtime["binary_sha256"] == plan["runtime"]["binary"]["sha256"],
        "Observed model or server differs from the frozen CPU runtime.",
    )
    if "libraries" in plan["runtime"]:
        libraries = {
            Path(item["path"]).resolve().name: item["sha256"]
            for item in plan["runtime"]["libraries"].values()
        }
        require(
            runtime["sibling_libraries_sha256"] == libraries,
            "Observed CPU libraries differ from the frozen build.",
        )
        protocol = common.read_json(ROOT / "eval/answers-protocol.json")
        require(
            runtime["context_size"] == protocol["settings"]["context_size"]
            and runtime["reasoning_budget_tokens"]
            == protocol["settings"]["reasoning_budget_tokens"]
            and runtime["threads"] == 4
            and runtime["gpu_layers"] == 0,
            "Observed inference settings changed.",
        )


def execute_row(plan, row, directory, guard, persist, *, owner_factory):
    payload = {
        "plan": plan,
        "row": row,
        "stage": row["stage"],
        "directory": str(directory),
    }
    if "replay" in row:
        payload["replay"] = row["replay"]
    common.save_json(directory / "input.json", payload, create=True)
    owner, stdout, sample = None, directory / "stdout.txt", None
    row.update(
        state="attempted",
        started_ns=guard.check(row["id"] + ":start")["continuous_before_ns"],
        first_stdout_ns=None,
        exit_observed_ns=None,
        outcome=None,
        ownership=None,
        failure=None,
    )
    persist()
    try:
        owner = owner_factory(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                str(directory / "input.json"),
            ],
            cwd=ROOT,
            stdout_path=stdout,
            stderr_path=directory / "stderr.txt",
            clock_ns=guard.continuous_ns,
        )
        sample = guard.check(row["id"] + ":release")
        if sample["continuous_after_ns"] - row["started_ns"] >= int(600 * 1e9):
            guard.fail("deadline_exceeded", sample)
        owner.release()
        while True:
            owner.poll_stdout()
            row["first_stdout_ns"] = owner.first_stdout_ns
            sample = guard.check()
            if sample["continuous_after_ns"] - row["started_ns"] >= int(600 * 1e9):
                guard.fail("deadline_exceeded", sample)
            if owner.exited():
                owner.poll_stdout()
                row["first_stdout_ns"] = owner.first_stdout_ns
                sample = guard.check(row["id"] + ":exit")
                row["exit_observed_ns"] = sample["continuous_after_ns"]
                if row["exit_observed_ns"] - row["started_ns"] >= int(600 * 1e9):
                    guard.fail("deadline_exceeded", sample)
                break
            time.sleep(plan["timing_policy"]["poll_ns"] / 1e9)
        row["ownership"] = owner.finish(terminate=False)
        require(
            row["ownership"]["cleanup_verified"],
            "Worker group cleanup was not verified.",
        )
        require(
            not row["ownership"].get("first_failure")
            and not row["ownership"].get("cleanup_failures"),
            "Worker ownership or cleanup reported a failure.",
        )
        require(row["ownership"]["returncode"] == 0, "Application worker failed.")
        observation = common.read_json(directory / "observation.json")
        exported = common.read_json(directory / "observer-end.json")
        record = common.read_json(directory / "answer.json")
        protocol = common.read_json(ROOT / "eval/answers-protocol.json")
        check_record(record, plan["workloads"][row["case_id"]], protocol)
        check_observation(observation, row, record, plan, exported)
        from arm64_rag.answers import render_answer

        require(
            stdout.read_text() == render_answer(record) + "\n",
            "Captured output differs from the original answer renderer.",
        )
        if "replay" in row:
            expected = common.read_json(Path(row["replay"]["path"]))
            require(
                input_identity(record) == input_identity(expected),
                "Replay inputs changed.",
            )
        row["outcome"] = "success"
    except BaseException as error:
        if owner is None and getattr(error, "record", None) is not None:
            row["ownership"] = error.record
        row["failure"] = safe_failure(error)
        endpoint = getattr(error, "sample", None) or sample
        row["deadline_exceeded"] = (
            isinstance(endpoint, dict)
            and type(endpoint.get("continuous_after_ns")) is int
            and endpoint["continuous_after_ns"] - row["started_ns"] >= int(600 * 1e9)
        )
        row["outcome"] = "timeout" if row["deadline_exceeded"] else "failed"
    finally:
        if owner is not None and row["ownership"] is None:
            try:
                row["ownership"] = owner.finish(terminate=True)
            except BaseException as error:
                row["cleanup_failure"] = safe_failure(error)
        try:
            row["finished_ns"] = guard.continuous_ns()
        except BaseException as error:
            row["finished_ns"] = None
            row["finish_clock_failure"] = safe_failure(error)
            if row["outcome"] in {None, "success"}:
                row["outcome"] = "failed"
            row["failure"] = row["failure"] or safe_failure(error)
        row["state"] = "finished"
        try:
            row["files"] = record_files(directory)
        except BaseException as error:
            row.setdefault("evidence_failures", []).append(safe_failure(error))
            row["files"] = {}
            if row["outcome"] in {None, "success"}:
                row["outcome"] = "failed"
            if row["failure"] is None:
                row["failure"] = safe_failure(error)
        try:
            persist()
        except BaseException as error:
            row.setdefault("evidence_failures", []).append(safe_failure(error))
            if row["outcome"] in {None, "success"}:
                row["outcome"] = "failed"
            if row["failure"] is None:
                row["failure"] = safe_failure(error)
            persist()
    return row["outcome"] == "success"


def run_stage(
    plan_path, stage, output, *, cli_path=None, owner_factory=None, compatibility=False
):
    if owner_factory is None:
        from answer_benchmark_owner import OwnedWorker

        owner_factory = OwnedWorker
    plan = common.read_json(plan_path)
    plan_raw = plan_path.read_bytes()
    require(json.loads(plan_raw) == plan, "Plan changed while loading.")
    identity = collection_identity(plan, plan_path)
    check_plan(plan)
    require(
        collection_identity(plan, plan_path) == identity,
        "Inputs changed during authentication.",
    )
    require(stage in STAGES, "Unknown stage.")
    require(
        (stage == "fixed") == (cli_path is not None), "Fixed replay needs its CLI run."
    )
    replay_evidence = public.Evidence()
    replay = (
        replay_inputs(
            plan, cli_path, compatibility=compatibility, evidence=replay_evidence
        )
        if stage == "fixed"
        else {}
    )
    replay_stamps = {
        str(path): list(stamp) for path, stamp in replay_evidence.seen.items()
    }
    require(
        not output.exists(), "Use a new directory to preserve earlier observations."
    )
    claim_path = plan_path.with_name(
        plan_path.name + (".check-" if compatibility else ".run-") + stage + ".json"
    )
    common.save_json(
        claim_path,
        {
            "plan_sha256": digest(plan_raw),
            "stage": stage,
            "purpose": "excluded_compatibility" if compatibility else "measurement",
            "output": str(output.resolve()),
        },
        create=True,
    )
    output.mkdir(parents=True, exist_ok=False)
    common.save_bytes(output / "plan.json", plan_raw, create=True)
    run = {
        "kind": KIND,
        "stage": stage,
        "state": "planned",
        "purpose": "excluded_compatibility" if compatibility else "measurement",
        "launch_claim": common.asset(claim_path),
        "plan_sha256": digest(plan_raw),
        "rows": schedule(stage, compatibility=compatibility),
        "cli_run": common.asset(cli_path) if cli_path else None,
        "timing_guard": None,
        "timing_valid": False,
        "failure": None,
        "integrity_verified": False,
        "input_identity": identity,
        "replay_stamps": replay_stamps,
        "source_snapshot": {"complete": False, "files": {}},
    }
    for row in run["rows"]:
        row["stage"] = stage
        if stage == "fixed":
            row["replay"] = replay[row["case_id"]]

    def persist():
        common.save_json(output / "run.json", run)

    persist()
    guard = common.timing.TimingGuard(
        plan["timing_policy"],
        common.timing.host_identity_hash(plan["host_policy"]["identity"]),
    )
    previous_host, seen_inputs = None, {}

    def verify_inputs():
        require(
            collection_identity(plan, plan_path) == identity,
            "Frozen source, runtime or index changed during collection.",
        )
        require(
            all(file_stamp(path) == stamp for path, stamp in replay_stamps.items()),
            "A fixed-context input changed during collection.",
        )

    try:
        guard.start()
        snapshot_sources(plan, output, run["source_snapshot"], persist)
        run["state"] = "running"
        for row in run["rows"]:
            guard.idle(SETTINGS["idle_seconds"], row["id"] + ":rest")
            before = common.host.collect()
            row["host_before"] = before
            violations = common.host.admit(before, plan["host_policy"])
            if previous_host is not None:
                violations += common.host.interval_violations(
                    previous_host, before, plan["host_policy"]
                )
            row["host_violations"] = list(dict.fromkeys(violations))
            persist()
            guard.check(row["id"] + ":host_before")
            require(not violations, "Host conditions changed before launch.")
            verify_inputs()
            verify_source_snapshot(plan, output, run["source_snapshot"], required=True)
            directory = output / row["id"]
            directory.mkdir()
            success = execute_row(
                plan, row, directory, guard, persist, owner_factory=owner_factory
            )
            try:
                verify_inputs()
                if success:
                    actual = input_identity(common.read_json(directory / "answer.json"))
                    key = row["case_id"]
                    require(
                        key not in seen_inputs or seen_inputs[key] == actual,
                        "Repeated CLI inputs changed during collection.",
                    )
                    seen_inputs[key] = actual
                row["inputs_unchanged"] = True
            except BaseException as error:
                row["inputs_unchanged"] = False
                row["input_failure"] = safe_failure(error)
                if row["outcome"] == "success":
                    row["outcome"] = "failed"
                row["failure"] = row.get("failure") or safe_failure(error)
                success = False
            persist()
            try:
                after = common.host.collect()
                row["host_after"] = after
                row["host_violations"] += common.host.admit(after, plan["host_policy"])
                row["host_violations"] += common.host.interval_violations(
                    before, after, plan["host_policy"]
                )
                previous_host = after
                row["host_violations"] = list(dict.fromkeys(row["host_violations"]))
                persist()
                guard.check(row["id"] + ":host_after")
            except BaseException as error:
                row["host_observation_failure"] = safe_failure(error)
                persist()
                raise
            require(
                success,
                "An application observation failed; later rows remain unattempted.",
            )
            require(
                not row["host_violations"],
                "Host conditions changed after the observation.",
            )
        run["state"] = "completed"
    except BaseException as error:
        run["state"] = "stopped"
        run["failure"] = safe_failure(error)
    finally:
        try:
            check_plan(plan, verify_model=False)
            require(
                plan_path.read_bytes() == plan_raw,
                "Frozen plan changed during the stage.",
            )
            verify_inputs()
            verify_source_snapshot(plan, output, run["source_snapshot"])
            run["integrity_verified"] = True
        except BaseException as error:
            run["integrity_failure"] = safe_failure(error)
        finally:
            try:
                if guard.evidence["first_failure"] is None:
                    guard.check("stage:verified")
            except BaseException as error:
                run["final_timing_failure"] = safe_failure(error)
            finally:
                run["timing_valid"] = guard.close()
                run["timing_guard"] = guard.evidence
        if not run["timing_valid"] or not run["integrity_verified"]:
            run["state"] = "stopped"
        persist()
    return run


def load_run(plan, path, stage, *, compatibility=False, evidence=None):
    evidence = evidence or public.Evidence()
    run, _ = evidence.document(path)
    archived_plan = path.parent / "plan.json"
    actual_plan, _ = evidence.document(archived_plan, expected=run["plan_sha256"])
    require(actual_plan == plan, "Run plan differs.")
    require(
        run["kind"] == KIND and run["stage"] == stage, "Run belongs to another stage."
    )
    require(
        run.get("purpose")
        == ("excluded_compatibility" if compatibility else "measurement"),
        "Compatibility observations cannot be used as measured-stage inputs.",
    )
    claim, _ = evidence.document(
        run["launch_claim"]["path"], expected=run["launch_claim"]["sha256"]
    )
    require(
        claim["plan_sha256"] == run["plan_sha256"]
        and claim["stage"] == stage
        and claim["purpose"] == run["purpose"]
        and Path(claim["output"]).resolve() == path.parent.resolve(),
        "Run differs from its create-only launch claim.",
    )
    expected = schedule(stage, compatibility=compatibility)
    require(len(run["rows"]) == len(expected), "Planned observations were removed.")
    verify_source_snapshot(
        plan,
        path.parent,
        run["source_snapshot"],
        evidence=evidence,
        required=any(row["state"] != "planned" for row in run["rows"]),
    )
    timing = run.get("timing_guard")
    passed = False
    if timing is not None:
        passed = common.timing.validate_evidence(
            timing,
            plan["timing_policy"],
            common.timing.host_identity_hash(plan["host_policy"]["identity"]),
        )
    checkpoints = {
        item["label"]: item["sample"] for item in (timing or {}).get("checkpoints", [])
    }
    phases = {
        "session:start",
        "session:end",
        "session:closed",
        "session:cleanup",
        "stage:verified",
    }
    phases.update(
        row["id"] + suffix
        for row in expected
        for suffix in (
            ":rest:start",
            ":rest:end",
            ":host_before",
            ":start",
            ":release",
            ":exit",
            ":host_after",
        )
    )
    require(set(checkpoints) <= phases, "Unexpected timing checkpoint label.")
    failure = (timing or {}).get("first_failure")
    if failure is not None:
        phase = failure["phase"]
        require(phase in phases, "Unexpected timing failure phase.")
        bound = -1 if phase == "session:start" else len(expected) - 1
        for index, planned in enumerate(expected):
            if phase.startswith(planned["id"] + ":"):
                bound = (
                    index - 1
                    if ":rest:" in phase or phase.endswith(":host_before")
                    else index
                )
                break
        require(
            all(row["state"] == "planned" for row in run["rows"][bound + 1 :]),
            "An observation launched after a timing stop.",
        )
    require(
        run["timing_valid"] == passed, "Timing status contradicts the guard evidence."
    )
    stopped, previous_host = False, None
    protocol, _ = evidence.document(ROOT / "eval/answers-protocol.json")
    for row, planned in zip(run["rows"], expected, strict=True):
        require(
            all(row[key] == planned[key] for key in ("id", "block", "case_id"))
            and row["stage"] == stage,
            "Row identity or order changed.",
        )
        require(
            row["state"] in {"planned", "attempted", "finished"}, "Unknown row state."
        )
        if stopped:
            require(row["state"] == "planned", "A new observation followed a stop.")
        if row["state"] == "planned":
            require(
                row.get("outcome") is None and row.get("ownership") is None,
                "Unattempted row contains an execution result.",
            )
            findings = []
            if row.get("host_before") is not None:
                findings = common.host.admit(row["host_before"], plan["host_policy"])
                if previous_host is not None:
                    findings += common.host.interval_violations(
                        previous_host, row["host_before"], plan["host_policy"]
                    )
            require(
                set(row.get("host_violations", [])) == set(findings),
                "Unattempted host findings differ.",
            )
            stopped = True
            continue
        require(
            row.get("outcome") in {None, "success", "failed", "timeout"},
            "Unknown observation outcome.",
        )
        require(
            row.get("host_before") is not None, "Launched row has no host boundary."
        )
        before, after = row["host_before"], row.get("host_after")
        violations = common.host.admit(before, plan["host_policy"])
        if previous_host is not None:
            violations += common.host.interval_violations(
                previous_host, before, plan["host_policy"]
            )
        require(not violations, "An observation launched after a known host departure.")
        if after is not None:
            violations += common.host.admit(after, plan["host_policy"])
            violations += common.host.interval_violations(
                before, after, plan["host_policy"]
            )
            previous_host = after
        require(set(row["host_violations"]) == set(violations), "Host findings differ.")
        files = row.get("files", {})
        require(set(files) <= RAW_FILES, "Unknown raw artifact name.")
        for name, item in files.items():
            require(
                Path(item["path"]).resolve()
                == (path.parent / row["id"] / name).resolve()
                and Path(name).name == name,
                "Raw record is outside its observation.",
            )
            pin = evidence.file(item["path"], expected=item["sha256"])
            require(pin["size"] == item["size"], "Raw artifact size differs.")
        if row["state"] == "finished" and row["outcome"] == "success":
            require(
                row.get("inputs_unchanged") is True,
                "Successful observation lacks input verification.",
            )
            require(
                after is not None
                or (run["state"] == "stopped" and row.get("host_observation_failure")),
                "Successful row lacks its final host boundary and retained failure.",
            )
            require(
                row["ownership"]["cleanup_verified"]
                and row["ownership"]["returncode"] == 0
                and not row["ownership"].get("first_failure")
                and not row["ownership"].get("cleanup_failures"),
                "Successful row lacks owned cleanup.",
            )
            require(
                row["ownership"]["first_stdout_ns"] == row["first_stdout_ns"],
                "Answer receipt differs from the owned capture evidence.",
            )
            record, _ = evidence.document(
                files["answer.json"]["path"], expected=files["answer.json"]["sha256"]
            )
            observation, _ = evidence.document(
                files["observation.json"]["path"],
                expected=files["observation.json"]["sha256"],
            )
            exported, _ = evidence.document(
                files["observer-end.json"]["path"],
                expected=files["observer-end.json"]["sha256"],
            )
            check_record(record, plan["workloads"][row["case_id"]], protocol)
            check_observation(observation, row, record, plan, exported)
            require(
                row["started_ns"]
                == checkpoints[row["id"] + ":start"]["continuous_before_ns"]
                and row["exit_observed_ns"]
                == checkpoints[row["id"] + ":exit"]["continuous_after_ns"],
                "Row times differ from the owned guard checkpoints.",
            )
            require(
                row["exit_observed_ns"] - row["started_ns"]
                < SETTINGS["worker_timeout_seconds"] * 1e9,
                "Successful worker exceeded the whole-process deadline.",
            )
            from arm64_rag.answers import render_answer

            stdout, _ = evidence.raw(
                files["stdout.txt"]["path"], expected=files["stdout.txt"]["sha256"]
            )
            require(
                stdout.decode() == render_answer(record) + "\n",
                "Captured output differs from the original renderer.",
            )
            if stage == "fixed":
                replay, _ = evidence.document(
                    row["replay"]["path"], expected=row["replay"]["sha256"]
                )
                require(
                    input_identity(record) == input_identity(replay),
                    "Saved fixed-context inputs changed.",
                )
        else:
            stopped = True
        if violations or after is None:
            stopped = True
    require(
        run["state"] in {"planned", "running", "stopped", "completed"},
        "Unknown run state.",
    )
    if run["state"] == "completed":
        require(
            not stopped and passed and run["integrity_verified"],
            "Incomplete evidence cannot pass.",
        )
    evidence.check()
    return run


def seconds(start, end):
    if start is None or end is None:
        return None
    require(
        type(start) is int and type(end) is int and 0 <= start <= end,
        "Invalid duration boundaries.",
    )
    return (end - start) / 1e9


def metrics(row):
    observation = common.read_json(Path(row["files"]["observation.json"]["path"]))
    record = common.read_json(Path(row["files"]["answer.json"]["path"]))
    event = observation["events"]
    exported = common.read_json(Path(row["files"]["observer-end.json"]["path"]))
    check_observer_intervals(event, row, exported)
    result = {
        "pipe_observed_answer_seconds": seconds(
            row["started_ns"], row["first_stdout_ns"]
        ),
        "application_return_seconds": seconds(
            row["started_ns"], event["application_return_ns"]
        )
        if row["stage"] == "cli"
        else None,
        "fixed_workflow_return_seconds": seconds(
            row["started_ns"], event["application_return_ns"]
        )
        if row["stage"] == "fixed"
        else None,
        "worker_exit_observed_seconds": seconds(
            row["started_ns"], row["exit_observed_ns"]
        ),
        "post_return_flush_seconds": seconds(
            event["application_return_ns"], event["stdout_flushed_ns"]
        ),
        "post_return_observer_and_exit_seconds": seconds(
            event["stdout_flushed_ns"], row["exit_observed_ns"]
        ),
        "observer_export_seconds": seconds(
            event["observer_export_start_ns"], exported["export_end_ns"]
        ),
        "server_initialization_seconds": seconds(
            event["server_initialization_start_ns"], event["server_ready_ns"]
        ),
        "native_spawn_to_ready_seconds": seconds(
            event["native_spawn_start_ns"], event["server_ready_ns"]
        ),
        "ready_server_rag_seconds": seconds(
            event.get("answer_start_ns"), event.get("answer_end_ns")
        ),
        "complete_request_seconds": seconds(
            event.get("complete_start_ns"), event.get("complete_end_ns")
        ),
        "answer_validation_seconds": seconds(
            event.get("validation_start_ns"), event.get("validation_end_ns")
        ),
        "worker_peak_rss_bytes": observation["worker_peak_rss_bytes"],
        "native_peak_rss_bytes": observation["native_peak_rss_bytes"],
        "model_time_to_first_token_seconds": None,
    }
    native = native_metrics(record)
    require(
        all(
            value is None
            or (type(value) in (int, float) and math.isfinite(value) and value >= 0)
            for value in native.values()
        ),
        "Invalid native diagnostic number.",
    )
    return result, native, input_identity(record), digest(encoded(record["answer"]))


def failed_observation(row):
    def elapsed(end):
        start = row.get("started_ns")
        return (
            (end - start) / 1e9
            if type(start) is int and type(end) is int and 0 <= start <= end
            else None
        )

    def read(name):
        item = row.get("files", {}).get(name)
        try:
            value = common.read_json(Path(item["path"])) if item else None
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, UnicodeError):
            return {}

    observation, record = read("observation.json"), read("answer.json")
    owner = row.get("ownership") or {}
    raw = record.get("raw_completion") or {}
    answer = record.get("answer") or {}
    status = answer.get("status") if isinstance(answer, dict) else None
    tokens = raw.get("tokens_predicted") if isinstance(raw, dict) else None
    memory = {}
    for key in ("worker_peak_rss_bytes", "native_peak_rss_bytes"):
        value = observation.get(key)
        available = type(value) is int and value > 0
        if key == "native_peak_rss_bytes":
            available = (
                available
                and observation.get("native_launches") == 1
                and observation.get("native_reaped") is True
            )
        memory[key] = value if available else None
    return {
        "included_in_latency_summary": False,
        "attempt_and_cleanup_seconds": elapsed(row.get("finished_ns")),
        "worker_exit_observed_seconds": elapsed(row.get("exit_observed_ns")),
        "first_stdout_seconds": elapsed(owner.get("first_stdout_ns")),
        "cleanup_verified": owner.get("cleanup_verified") is True,
        "worker_reaped": owner.get("worker_reaped") is True,
        "returncode": owner.get("returncode")
        if type(owner.get("returncode")) is int
        else None,
        "ownership_sha256": digest(encoded(owner)) if owner else None,
        "reported_generated_tokens": tokens
        if type(tokens) is int and tokens >= 0
        else None,
        "reported_answer_status": status
        if isinstance(status, str)
        and status in {"answered", "insufficient_evidence", "conflicting_evidence"}
        else None,
        **memory,
    }


def make_report(plan_path, cli_path=None, fixed_path=None):
    evidence = public.Evidence()
    plan, _ = evidence.document(plan_path)
    for item in plan["sources"].values():
        evidence.file(item["path"], expected=item["sha256"])
    for name in ("binary", "build", "cache", "compile_commands"):
        item = plan["runtime"].get(name)
        if item is not None and "path" in item:
            evidence.file(item["path"], expected=item["sha256"])
    for item in plan["runtime"].get("libraries", {}).values():
        evidence.file(item["path"], expected=item["sha256"])
    if plan.get("index"):
        database = Path(plan["index"]["path"]).resolve()
        evidence.seen[database] = public.signature(database)
    check_plan(plan, verify_model=False)
    report = {
        "kind": KIND,
        "plan_sha256": digest(plan_path.read_bytes()),
        "settings": SETTINGS,
        "limits": LIMITS,
        "model_sha256": plan["runtime"]["model"]["sha256"],
        "binary_sha256": plan["runtime"]["binary"]["sha256"],
        "build": public.public_asset(plan["runtime"]["build"])
        if "build" in plan["runtime"]
        else None,
        "libraries": {
            name: public.public_asset(item)
            for name, item in plan["runtime"].get("libraries", {}).items()
        },
        "index_fingerprint": plan.get("index", {}).get("fingerprint"),
        "sources": {
            name: public.public_asset(item) for name, item in plan["sources"].items()
        },
        "stages": {},
        "planned": 21,
        "successes": 0,
        "failed": 0,
        "unresolved": 0,
        "unattempted": 0,
        "unobserved": 0,
        "protocol_complete": False,
        "quality_review": "pending",
        "study_interpretation_review": "pending",
        "backend_adoption": False,
        "portable_inputs": dict.fromkeys((*SELECTED, EMPTY_ID)),
    }
    paths = {"cli": cli_path, "fixed": fixed_path}
    require(
        fixed_path is None or cli_path is not None,
        "Replay report requires its CLI evidence.",
    )
    for stage in STAGES:
        path = paths[stage]
        if path is None:
            report["unobserved"] += len(schedule(stage))
            report["stages"][stage] = {
                "state": "unobserved",
                "planned": len(schedule(stage)),
                "rows": [{**row, "state": "unobserved"} for row in schedule(stage)],
            }
            continue
        run = load_run(plan, path, stage, evidence=evidence)
        if stage == "fixed":
            require(
                run["cli_run"]["sha256"] == digest(cli_path.read_bytes()),
                "Replay CLI run differs.",
            )
            expected = replay_inputs(plan, cli_path, evidence=evidence)
            for row in run["rows"]:
                require(
                    row["replay"] == expected[row["case_id"]],
                    "Replay selection differs.",
                )
        rows, grouped = [], {}
        for row in run["rows"]:
            item = {key: row[key] for key in ("id", "block", "case_id", "state")}
            item["outcome"] = row.get("outcome")
            item["host_before"] = public.public_host(row.get("host_before") or {})
            item["host_after"] = public.public_host(row.get("host_after") or {})
            item["host_violations"] = row.get("host_violations", [])
            item["raw_files"] = {
                name: public.public_asset(pin)
                for name, pin in row.get("files", {}).items()
            }
            item["failure_sha256"] = (
                digest(encoded(row["failure"])) if row.get("failure") else None
            )
            item["host_observation_failure_sha256"] = (
                digest(encoded(row["host_observation_failure"]))
                if row.get("host_observation_failure")
                else None
            )
            if row["state"] == "planned":
                report["unattempted"] += 1
            elif row["state"] == "attempted":
                report["unresolved"] += 1
                item["available_failure_evidence"] = failed_observation(row)
            elif row["outcome"] != "success":
                report["failed"] += 1
                item["available_failure_evidence"] = failed_observation(row)
            else:
                report["successes"] += 1
                values, native, inputs, answer_hash = metrics(row)
                item.update(
                    metrics=values,
                    native_diagnostics=native,
                    inputs=inputs,
                    answer_sha256=answer_hash,
                    timing_valid=run["timing_valid"]
                    and run["integrity_verified"]
                    and not row["host_violations"]
                    and row.get("host_after") is not None
                    and not row.get("host_observation_failure"),
                )
                if stage == "cli" and report["portable_inputs"][row["case_id"]] is None:
                    report["portable_inputs"][row["case_id"]] = inputs
                if item["timing_valid"]:
                    grouped.setdefault(row["case_id"], []).append(values)
            rows.append(item)
        summary = {}
        for case_id, values in grouped.items():
            summary[case_id] = {
                key: {
                    "observations": len(numbers),
                    "median": statistics.median(numbers),
                    "minimum": min(numbers),
                    "maximum": max(numbers),
                }
                for key in values[0]
                if (
                    numbers := [
                        value[key] for value in values if value[key] is not None
                    ]
                )
            }
        report["stages"][stage] = {
            "state": run["state"],
            "run_sha256": digest(path.read_bytes()),
            "launch_claim": public.public_asset(run["launch_claim"]),
            "source_snapshot": {
                "complete": run["source_snapshot"]["complete"],
                "files": {
                    name: public.public_asset(item)
                    for name, item in run["source_snapshot"]["files"].items()
                },
            },
            "timing_valid": run["timing_valid"],
            "integrity_verified": run["integrity_verified"],
            "timing": public.public_timing(run["timing_guard"], run["timing_valid"])
            if run["timing_guard"] is not None
            else None,
            "rows": rows,
            "descriptive_summaries": summary,
        }
    report["protocol_complete"] = all(
        report["stages"][stage]["state"] == "completed" for stage in STAGES
    )
    require(
        sum(
            report[key]
            for key in (
                "successes",
                "failed",
                "unresolved",
                "unattempted",
                "unobserved",
            )
        )
        == 21,
        "Observation accounting changed.",
    )
    check_plan(plan, verify_model=False)
    evidence.check()
    return report


def run_compatibility(plan_path, output):
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "kind": KIND,
        "purpose": "excluded_compatibility",
        "plan_sha256": digest(plan_path.read_bytes()),
        "state": "planned",
        "case_id": COMPATIBILITY_CASE,
        "planned": 2,
        "stages": [{"stage": stage, "state": "planned"} for stage in STAGES],
        "included_in_measurements": False,
    }
    common.save_json(output / "check.json", record)
    for entry in record["stages"]:
        entry["state"] = "attempted"
        common.save_json(output / "check.json", record)
        try:
            stage = entry["stage"]
            result = run_stage(
                plan_path,
                stage,
                output / stage,
                compatibility=True,
                cli_path=output / "cli/run.json" if stage == "fixed" else None,
            )
            entry.update(
                state=result["state"], run=common.asset(output / stage / "run.json")
            )
            if result["state"] != "completed":
                record["state"] = "stopped"
                break
        except BaseException as error:
            entry.update(state="failed", failure=safe_failure(error))
            record["state"] = "stopped"
            break
        finally:
            common.save_json(output / "check.json", record)
    else:
        record["state"] = "completed"
    common.save_json(output / "check.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(
        description="Measure the existing buffered CPU answer path."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Freeze public inputs and local CPU/host pins."
    )
    prepare.add_argument("--db", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    check = commands.add_parser(
        "check", help="Run two excluded compatibility observations."
    )
    check.add_argument("--plan", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="Run one declared stage without retries.")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--stage", choices=STAGES, required=True)
    run.add_argument("--cli-run", type=Path)
    run.add_argument("--output", type=Path, required=True)
    report = commands.add_parser(
        "report", help="Export allowlisted descriptive records."
    )
    report.add_argument("--plan", type=Path, required=True)
    report.add_argument("--cli-run", type=Path)
    report.add_argument("--fixed-run", type=Path)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        freeze(args.db, args.output)
    elif args.command in {"run", "check"}:
        lock = ROOT / "artifacts/benchmarks/benchmark.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)

        def interrupted(signum, frame):
            raise KeyboardInterrupt("Application measurement interrupted.")

        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            with common.exclusive_lock(lock):
                result = (
                    run_compatibility(args.plan, args.output)
                    if args.command == "check"
                    else run_stage(
                        args.plan, args.stage, args.output, cli_path=args.cli_run
                    )
                )
        finally:
            signal.signal(signal.SIGTERM, previous)
        if result["state"] != "completed":
            parser.exit(
                1, "The stage stopped. Retained records explain the incomplete run.\n"
            )
    else:
        value = make_report(args.plan, args.cli_run, args.fixed_run)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        common.save_json(args.output, value, create=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        gate = int(os.environ.pop("ARM64_BENCHMARK_GATE_FD"))
        try:
            require(os.read(gate, 1) == b"G", "Worker was not released by its owner.")
        finally:
            os.close(gate)
        raise SystemExit(worker(common.read_json(Path(sys.argv[2]))))
    main()
