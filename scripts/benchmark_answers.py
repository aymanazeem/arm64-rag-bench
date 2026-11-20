"""Compare the buffered answer command on the CPU and KleidiAI servers."""

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
KIND = "buffered-answer-comparison-v1"
BACKENDS = ("cpu", "kleidiai")
DATASETS = {
    "development.json": "development",
    "retrieval-validation.json": "regression",
    "answer-validation.json": "validation",
}
REVIEWED = {
    "cpu": "docs/results/grounded-answers/{}.json",
    "kleidiai": "docs/results/controlled-cpu/quality-{}.json",
}
CASES = (
    *(f"dev-{number:03}" for number in range(1, 15)),
    *(f"val-{number:03}" for number in range(1, 19)),
    *(f"fresh-{number:03}" for number in range(1, 9)),
)
COMPATIBILITY_CASE = "val-015"
EMPTY_ID = "empty-retrieval"
EMPTY_QUERY = "applicationbenchmarkemptyretrieval9fa83b2c"
SIMILAR_LENGTH = 0.05
SETTINGS = {
    "blocks": 4,
    "seed": 42,
    "idle_seconds": 120,
    "worker_timeout_seconds": 600,
    "startup_timeout_seconds": 120,
    "request_timeout_seconds": 300,
    "output_mode": "plain_stdout_pipe_with_explicit_post_return_flush",
    "backends": list(BACKENDS),
    "pairing": "adjacent_backends_balanced_order",
    "empty_controls_per_block": 1,
    "retrieval_method": "lexical",
    "similar_length_tolerance": SIMILAR_LENGTH,
}
LIMITS = [
    "These are known questions, not fresh accuracy evaluation.",
    "The application buffers its answer. Model time to first token is unobserved.",
    "Fresh processes do not imply cold file caches. Model verification reads the file.",
    "Each backend answers each question once with the application's fixed seed.",
    "Answers differ between backends, so answer length is part of the result.",
    "Both servers use the application's four threads; other settings are untested.",
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


def schedule(*, compatibility=False):
    """Pair both backends on each case, with each order used equally often."""
    if compatibility:
        return [
            {
                "id": f"check-{backend}",
                "block": 1,
                "pair": "check",
                "case_id": COMPATIBILITY_CASE,
                "backend": backend,
                "state": "planned",
            }
            for backend in BACKENDS
        ]
    rng, blocks = random.Random(SETTINGS["seed"]), SETTINGS["blocks"]
    cases = list(CASES)
    rng.shuffle(cases)
    size = len(cases) // blocks
    empty_orders = [BACKENDS, BACKENDS[::-1]] * (blocks // 2)
    rng.shuffle(empty_orders)
    rows = []
    for block in range(1, blocks + 1):
        orders = [BACKENDS, BACKENDS[::-1]] * (size // 2)
        rng.shuffle(orders)
        planned = [
            *zip(cases[(block - 1) * size : block * size], orders),
            (EMPTY_ID, empty_orders[block - 1]),
        ]
        rng.shuffle(planned)
        for number, (case, order) in enumerate(planned, 1):
            pair = f"b{block}-p{number:02}"
            rows.extend(
                {
                    "id": f"{pair}-{backend}",
                    "block": block,
                    "pair": pair,
                    "case_id": case,
                    "backend": backend,
                    "state": "planned",
                }
                for backend in order
            )
    return rows


def reviewed_answer(case):
    generation = case["generation"]
    require(
        case["attempt_state"] == "completed"
        and generation["tokens_predicted"] > 0
        and isinstance(case["final_content_hash"]["sha256"], str)
        and isinstance(case["raw_completion_content_hash"]["sha256"], str),
        "A reviewed answer record is incomplete.",
    )
    return {
        "final_content_sha256": case["final_content_hash"]["sha256"],
        "raw_content_sha256": case["raw_completion_content_hash"]["sha256"],
        "generated_tokens": generation["tokens_predicted"],
    }


def reviewed_records():
    """Return each question's shared input and both reviewed answer identities."""
    model = common.read_json(ROOT / "configs/answer-model.json")["model"]["sha256"]
    servers = {
        backend: build["targets"]["llama-server"]
        for backend, (_, build) in builds().items()
    }
    cases = {}
    for role in DATASETS.values():
        for backend, pattern in REVIEWED.items():
            reviewed = common.read_json(ROOT / pattern.format(role))
            runtime = reviewed["runtime"]
            require(
                runtime["binary_sha256"] == servers[backend]
                and runtime["model_sha256"] == model
                and runtime["threads"] == 4,
                "A reviewed answer came from another server, model or thread count.",
            )
            for case in reviewed["cases"]:
                require(case["id"] in CASES, "A reviewed answer has no question.")
                item = cases.setdefault(
                    case["id"],
                    {"published_prompt": case["prompt"], "reviewed_answers": {}},
                )
                require(
                    item["published_prompt"] == case["prompt"],
                    "Reviewed backends used different inputs.",
                )
                answers = item["reviewed_answers"]
                require(backend not in answers, "A reviewed answer is repeated.")
                answers[backend] = reviewed_answer(case)
    require(
        sorted(cases) == sorted(CASES)
        and all(
            set(item["reviewed_answers"]) == set(BACKENDS) for item in cases.values()
        ),
        "A reviewed answer record is missing.",
    )
    return cases


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
            require(case["id"] not in cases, "A question ID is repeated.")
            cases[case["id"]] = {
                "id": case["id"],
                "question": case["question"],
                "source_ids": scopes[case["corpus_id"]],
                "dataset": "eval/" + name,
                "required_facts": case["required_facts"],
                "unacceptable_claims": case["unacceptable_claims"],
            }
    require(sorted(cases) == sorted(CASES), "The fixed question set changed.")
    for case_id, reviewed in reviewed_records().items():
        cases[case_id].update(reviewed)
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
        "reviewed_answers": None,
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
        *(
            ROOT / pattern.format(role)
            for pattern in REVIEWED.values()
            for role in DATASETS.values()
        ),
    ]


def builds():
    """Return the pinned server builds, which may differ only in KleidiAI."""
    runtime = common.read_json(ROOT / "configs/runtime.json")
    records = {}
    for backend in BACKENDS:
        build_path = ROOT / "artifacts/feasibility" / backend / "build.json"
        build = common.read_json(build_path)
        flags = {
            **runtime["cmake"],
            "GGML_CPU_KLEIDIAI": "ON" if backend == "kleidiai" else "OFF",
        }
        require(
            build["backend"] == backend
            and build["source"] == runtime["llama_cpp"]
            and build["flags"] == flags
            and build["kleidiai"]
            == (runtime["kleidiai"] if backend == "kleidiai" else None)
            and build["backends"] == ["ggml-cpu"],
            "Use the pinned CPU-only builds.",
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
            "Build acceleration settings changed.",
        )
        common.runtime.checked_cache(build_path.with_name("CMakeCache.txt"), flags)
        records[backend] = build_path, build
    for key in ("compiler_flags", "compilers", "tools", "system"):
        require(
            records["cpu"][1][key] == records["kleidiai"][1][key],
            f"Paired build {key} differ.",
        )
    return records


def server_path(backend):
    return ROOT / "build" / backend / "bin/llama-server"


def runtime_assets():
    config = common.read_json(ROOT / "configs/answer-model.json")
    model = ROOT / config["directory"] / config["model"]["filename"]
    result = {"model": common.asset(model), "backends": {}}
    require(
        result["model"]["sha256"] == config["model"]["sha256"]
        and result["model"]["size"] == config["model"]["size"],
        "Model differs from its pin.",
    )
    for backend, (build_path, build) in builds().items():
        binary = server_path(backend)
        assets = {"binary": common.asset(binary)}
        require(
            assets["binary"]["sha256"] == build["targets"]["llama-server"],
            "Server differs from its build record.",
        )
        libraries = common.libraries(binary)
        require(set(libraries) == set(build["libraries"]), "Library inventory changed.")
        assets["libraries"] = {
            name: common.asset(path) for name, path in libraries.items()
        }
        require(
            all(
                item["sha256"] == build["libraries"][name]
                for name, item in assets["libraries"].items()
            ),
            "Library differs from its build record.",
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
        result["backends"][backend] = assets
    return result


def freeze(index_path, destination):
    cases, index = public_workloads(index_path)
    runtime = runtime_assets()
    snapshot = common.host.collect()
    plan = {
        "kind": KIND,
        "settings": SETTINGS,
        "workloads": cases,
        "schedule": schedule(),
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
    require(plan["schedule"] == schedule(), "Measurement schedule changed.")
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
    config = common.read_json(ROOT / "configs/answer-model.json")
    model = plan["runtime"]["model"]
    require(
        Path(model["path"]).resolve()
        == (ROOT / config["directory"] / config["model"]["filename"]).resolve()
        and model["sha256"] == config["model"]["sha256"]
        and model["size"] == config["model"]["size"],
        "Frozen model is not the pinned answer model.",
    )
    records = builds()
    require(
        set(plan["runtime"]["backends"]) == set(BACKENDS), "Frozen backends changed."
    )
    for backend, (build_path, build) in records.items():
        assets = plan["runtime"]["backends"][backend]
        require(
            Path(assets["binary"]["path"]).resolve() == server_path(backend).resolve()
            and assets["binary"]["sha256"] == build["targets"]["llama-server"]
            and Path(assets["build"]["path"]).resolve() == build_path.resolve(),
            "Frozen server is not its pinned build.",
        )
        require(
            set(common.libraries(Path(assets["binary"]["path"])))
            == set(assets["libraries"])
            == set(build["libraries"]),
            "Library inventory changed.",
        )
        require(
            all(
                item["sha256"] == build["libraries"][name]
                for name, item in assets["libraries"].items()
            ),
            "Library pin differs.",
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
        common.verify_asset(model)
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

    files = [
        plan_path,
        *(Path(item["path"]) for item in plan["sources"].values()),
        Path(plan["runtime"]["model"]["path"]),
        *additional,
    ]
    for backend in plan["runtime"]["backends"].values():
        files.extend(
            Path(backend[name]["path"])
            for name in ("binary", "build", "cache", "compile_commands")
        )
        files.extend(Path(item["path"]) for item in backend["libraries"].values())
        files.extend(common.libraries(Path(backend["binary"]["path"])).values())
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
        "prompt_n": timings.get("prompt_n"),
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
            "Packed input differs from the public workload.",
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
    binary = plan["runtime"]["backends"][row["backend"]]["binary"]["path"]
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
        sys.argv = [
            "arm64-rag",
            "ask",
            case["question"],
            "--db",
            plan["index"]["path"],
            "--method",
            "lexical",
            "--server-binary",
            binary,
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


def check_observer_intervals(events, row, exported):
    stamps = [
        events.get("observer_export_start_ns"),
        exported.get("export_end_ns"),
        row.get("exit_observed_ns"),
    ]
    require(
        all(type(value) is int and value >= 0 for value in stamps)
        and stamps == sorted(stamps),
        "Observer export boundaries are inconsistent.",
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
    backend = plan["runtime"]["backends"][row["backend"]]
    require(
        runtime["model_sha256"] == plan["runtime"]["model"]["sha256"]
        and runtime["binary_sha256"] == backend["binary"]["sha256"],
        "Observed model or server differs from the frozen runtime.",
    )
    if "libraries" in backend:
        libraries = {
            Path(item["path"]).resolve().name: item["sha256"]
            for item in backend["libraries"].values()
        }
        require(
            runtime["sibling_libraries_sha256"] == libraries,
            "Observed libraries differ from the frozen build.",
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
    payload = {"plan": plan, "row": row, "directory": str(directory)}
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


def run_collection(plan_path, output, *, owner_factory=None, compatibility=False):
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
    require(
        not output.exists(), "Use a new directory to preserve earlier observations."
    )
    claim_path = plan_path.with_name(
        plan_path.name + (".check" if compatibility else ".run") + ".json"
    )
    common.save_json(
        claim_path,
        {
            "plan_sha256": digest(plan_raw),
            "purpose": "excluded_compatibility" if compatibility else "measurement",
            "output": str(output.resolve()),
        },
        create=True,
    )
    output.mkdir(parents=True, exist_ok=False)
    common.save_bytes(output / "plan.json", plan_raw, create=True)
    run = {
        "kind": KIND,
        "state": "planned",
        "purpose": "excluded_compatibility" if compatibility else "measurement",
        "launch_claim": common.asset(claim_path),
        "plan_sha256": digest(plan_raw),
        "rows": schedule(compatibility=compatibility),
        "timing_guard": None,
        "timing_valid": False,
        "failure": None,
        "integrity_verified": False,
        "input_identity": identity,
        "source_snapshot": {"complete": False, "files": {}},
    }

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
                        "Inputs for the same question differ between observations.",
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
                "Frozen plan changed during collection.",
            )
            verify_inputs()
            verify_source_snapshot(plan, output, run["source_snapshot"])
            run["integrity_verified"] = True
        except BaseException as error:
            run["integrity_failure"] = safe_failure(error)
        finally:
            try:
                if guard.evidence["first_failure"] is None:
                    guard.check("collection:verified")
            except BaseException as error:
                run["final_timing_failure"] = safe_failure(error)
            finally:
                run["timing_valid"] = guard.close()
                run["timing_guard"] = guard.evidence
        if not run["timing_valid"] or not run["integrity_verified"]:
            run["state"] = "stopped"
        persist()
    return run


def load_run(plan, path, *, compatibility=False, evidence=None):
    evidence = evidence or public.Evidence()
    run, _ = evidence.document(path)
    archived_plan = path.parent / "plan.json"
    actual_plan, _ = evidence.document(archived_plan, expected=run["plan_sha256"])
    require(actual_plan == plan, "Run plan differs.")
    require(run["kind"] == KIND, "Run belongs to another study.")
    require(
        run.get("purpose")
        == ("excluded_compatibility" if compatibility else "measurement"),
        "Compatibility observations cannot be used as measurements.",
    )
    claim, _ = evidence.document(
        run["launch_claim"]["path"], expected=run["launch_claim"]["sha256"]
    )
    require(
        claim["plan_sha256"] == run["plan_sha256"]
        and claim["purpose"] == run["purpose"]
        and Path(claim["output"]).resolve() == path.parent.resolve(),
        "Run differs from its create-only launch claim.",
    )
    expected = schedule(compatibility=compatibility)
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
        "collection:verified",
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
    stopped, previous_host, inputs = False, None, {}
    protocol, _ = evidence.document(ROOT / "eval/answers-protocol.json")
    for row, planned in zip(run["rows"], expected, strict=True):
        require(
            all(
                row[key] == planned[key]
                for key in ("id", "block", "pair", "case_id", "backend")
            ),
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
            identity = input_identity(record)
            require(
                inputs.setdefault(row["case_id"], identity) == identity,
                "Saved inputs for one question differ between observations.",
            )
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
        ),
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
    raw = record["raw_completion"]
    answer = (
        None
        if raw is None
        else {
            "final_content_sha256": digest(raw["content"].encode()),
            "raw_content_sha256": digest(raw["raw_content"].encode()),
            "generated_tokens": raw["tokens_predicted"],
        }
    )
    return result, native, input_identity(record), answer


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


def distribution(values):
    return {
        "observations": len(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def compare(rows):
    """Apply the analysis fixed before collection to one complete valid run."""
    sides, first = {}, {}
    for item in rows:
        sides.setdefault(item["case_id"], {})[item["backend"]] = item
        first.setdefault(item["case_id"], item["backend"])
    require(
        set(sides) == {*CASES, EMPTY_ID}
        and all(set(value) == set(BACKENDS) for value in sides.values()),
        "Comparison needs every question on both backends.",
    )
    pairs = []
    for case_id in CASES:
        item = {"case_id": case_id, "first_backend": first[case_id]}
        for key, source, scale in (
            ("answer_seconds", "pipe_observed_answer_seconds", 1),
            ("server_initialization_seconds", "server_initialization_seconds", 1),
            ("native_spawn_to_ready_seconds", "native_spawn_to_ready_seconds", 1),
            ("complete_request_seconds", "complete_request_seconds", 1),
            ("prompt_seconds", "prompt_ms", 1e-3),
            ("generation_seconds", "predicted_ms", 1e-3),
            ("prompt_tokens", "prompt_n", 1),
            ("timed_generated_tokens", "predicted_n", 1),
            ("generated_tokens", "generated_tokens", 1),
            ("native_peak_rss_bytes", "native_peak_rss_bytes", 1),
        ):
            item[key] = {}
            for backend in BACKENDS:
                observed = sides[case_id][backend]
                value = {**observed["native_diagnostics"], **observed["metrics"]}[
                    source
                ]
                require(
                    type(value) in (int, float) and value > 0,
                    "A comparison value is unavailable.",
                )
                item[key][backend] = value * scale
        item["matches_reviewed_answer"] = {
            backend: sides[case_id][backend]["matches_reviewed_answer"]
            for backend in BACKENDS
        }
        item["kleidiai_over_cpu"] = (
            item["answer_seconds"]["kleidiai"] / item["answer_seconds"]["cpu"]
        )
        pairs.append(item)

    def total(selected, key, backend):
        return sum(pair[key][backend] for pair in selected)

    def summary(selected):
        if not selected:
            return {"questions": 0}
        totals = {
            backend: total(selected, "answer_seconds", backend) for backend in BACKENDS
        }
        return {
            "questions": len(selected),
            "answer_seconds_total": totals,
            "kleidiai_over_cpu_total": totals["kleidiai"] / totals["cpu"],
            "kleidiai_over_cpu_per_question": distribution(
                [pair["kleidiai_over_cpu"] for pair in selected]
            ),
            "kleidiai_slower_questions": sum(
                pair["kleidiai_over_cpu"] > 1 for pair in selected
            ),
            "component_seconds_total": {
                key: {backend: total(selected, key, backend) for backend in BACKENDS}
                for key in (
                    "server_initialization_seconds",
                    "native_spawn_to_ready_seconds",
                    "complete_request_seconds",
                    "prompt_seconds",
                    "generation_seconds",
                )
            },
            "generated_tokens_total": {
                backend: total(selected, "generated_tokens", backend)
                for backend in BACKENDS
            },
            "prompt_tokens_per_second": {
                backend: total(selected, "prompt_tokens", backend)
                / total(selected, "prompt_seconds", backend)
                for backend in BACKENDS
            },
            "generation_ms_per_token": {
                backend: 1000
                * total(selected, "generation_seconds", backend)
                / total(selected, "timed_generated_tokens", backend)
                for backend in BACKENDS
            },
        }

    similar = [
        pair
        for pair in pairs
        if abs(
            pair["generated_tokens"]["kleidiai"] / pair["generated_tokens"]["cpu"] - 1
        )
        <= SIMILAR_LENGTH
    ]
    empty = {
        backend: {
            key: distribution(
                [
                    item["metrics"][key]
                    for item in rows
                    if item["case_id"] == EMPTY_ID and item["backend"] == backend
                ]
            )
            for key in (
                "pipe_observed_answer_seconds",
                "server_initialization_seconds",
            )
        }
        for backend in BACKENDS
    }
    return {
        "definition": (
            "Answer seconds are the pipe-observed buffered answer times. Ratios divide "
            "KleidiAI by CPU. Token rates pool native prompt and generation timings."
        ),
        "pairs": pairs,
        "all_questions": summary(pairs),
        "similar_length": {
            "tolerance": SIMILAR_LENGTH,
            **summary(similar),
        },
        "by_first_backend": {
            backend: summary(
                [pair for pair in pairs if pair["first_backend"] == backend]
            )
            for backend in BACKENDS
        },
        "empty_retrieval": empty,
        "native_peak_rss_bytes": {
            backend: distribution(
                [pair["native_peak_rss_bytes"][backend] for pair in pairs]
            )
            for backend in BACKENDS
        },
        "reviewed_answer_matches": {
            backend: sum(pair["matches_reviewed_answer"][backend] for pair in pairs)
            for backend in BACKENDS
        },
    }


def make_report(plan_path, run_path=None):
    evidence = public.Evidence()
    plan, _ = evidence.document(plan_path)
    for item in plan["sources"].values():
        evidence.file(item["path"], expected=item["sha256"])
    for backend in plan["runtime"].get("backends", {}).values():
        for name in ("binary", "build", "cache", "compile_commands"):
            item = backend.get(name)
            if item is not None and "path" in item:
                evidence.file(item["path"], expected=item["sha256"])
        for item in backend.get("libraries", {}).values():
            evidence.file(item["path"], expected=item["sha256"])
    if plan.get("index"):
        database = Path(plan["index"]["path"]).resolve()
        evidence.seen[database] = public.signature(database)
    check_plan(plan, verify_model=False)
    planned = schedule()
    report = {
        "kind": KIND,
        "plan_sha256": digest(plan_path.read_bytes()),
        "settings": SETTINGS,
        "limits": LIMITS,
        "model_sha256": plan["runtime"]["model"]["sha256"],
        "backends": {
            name: {
                "binary_sha256": backend["binary"]["sha256"],
                "build": public.public_asset(backend["build"])
                if "build" in backend
                else None,
                "libraries": {
                    library: public.public_asset(item)
                    for library, item in backend.get("libraries", {}).items()
                },
            }
            for name, backend in plan["runtime"].get("backends", {}).items()
        },
        "index_fingerprint": plan.get("index", {}).get("fingerprint"),
        "sources": {
            name: public.public_asset(item) for name, item in plan["sources"].items()
        },
        "planned": len(planned),
        "successes": 0,
        "failed": 0,
        "unresolved": 0,
        "unattempted": 0,
        "unobserved": 0,
        "protocol_complete": False,
        "quality_review": "pending",
        "study_interpretation_review": "pending",
        "backend_adoption": False,
        "portable_inputs": dict.fromkeys((*CASES, EMPTY_ID)),
        "collection": None,
        "comparison": None,
    }
    if run_path is None:
        report["unobserved"] = len(planned)
        report["collection"] = {
            "state": "unobserved",
            "rows": [{**row, "state": "unobserved"} for row in planned],
        }
        check_plan(plan, verify_model=False)
        evidence.check()
        return report
    run = load_run(plan, run_path, evidence=evidence)
    rows, valid = [], []
    for row in run["rows"]:
        item = {
            key: row[key]
            for key in ("id", "block", "pair", "case_id", "backend", "state")
        }
        item["outcome"] = row.get("outcome")
        item["host_before"] = public.public_host(row.get("host_before") or {})
        item["host_after"] = public.public_host(row.get("host_after") or {})
        item["host_violations"] = row.get("host_violations", [])
        item["raw_files"] = {
            name: public.public_asset(pin) for name, pin in row.get("files", {}).items()
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
            values, native, inputs, answer = metrics(row)
            reviewed = plan["workloads"][row["case_id"]]["reviewed_answers"]
            item.update(
                metrics=values,
                native_diagnostics=native,
                inputs=inputs,
                answer=answer,
                matches_reviewed_answer=None
                if answer is None
                else answer == reviewed[row["backend"]],
                timing_valid=run["timing_valid"]
                and run["integrity_verified"]
                and not row["host_violations"]
                and row.get("host_after") is not None
                and not row.get("host_observation_failure"),
            )
            if report["portable_inputs"][row["case_id"]] is None:
                report["portable_inputs"][row["case_id"]] = inputs
            if item["timing_valid"]:
                valid.append(item)
        rows.append(item)
    report["collection"] = {
        "state": run["state"],
        "run_sha256": digest(run_path.read_bytes()),
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
    }
    report["protocol_complete"] = run["state"] == "completed"
    if report["protocol_complete"] and len(valid) == len(planned):
        report["comparison"] = compare(valid)
        if report["comparison"]["reviewed_answer_matches"] == dict.fromkeys(
            BACKENDS, len(CASES)
        ):
            report["quality_review"] = "every_answer_matches_its_reviewed_record"
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
        == len(planned),
        "Observation accounting changed.",
    )
    check_plan(plan, verify_model=False)
    evidence.check()
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Compare the buffered answer command on CPU and KleidiAI."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Freeze public inputs and local server and host pins."
    )
    prepare.add_argument("--db", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    for name, text in (
        ("check", "Run one excluded compatibility pair."),
        ("run", "Run the paired comparison without retries."),
    ):
        command = commands.add_parser(name, help=text)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    report = commands.add_parser(
        "report", help="Export allowlisted records and the fixed comparison."
    )
    report.add_argument("--plan", type=Path, required=True)
    report.add_argument("--run", type=Path)
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
                result = run_collection(
                    args.plan, args.output, compatibility=args.command == "check"
                )
        finally:
            signal.signal(signal.SIGTERM, previous)
        if result["state"] != "completed":
            parser.exit(
                1, "Collection stopped. Retained records explain the incomplete run.\n"
            )
    else:
        value = make_report(args.plan, args.run)
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
