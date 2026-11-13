# Controlled CPU benchmarks

Compare standard CPU inference with KleidiAI on the same machine, model and
settings. Build both runtimes using the [runtime guide](runtime.md). The answer
application continues to use the standard CPU build.

The pinned KleidiAI integration [requantizes Q8 weights by row](https://github.com/ggml-org/llama.cpp/blob/8ca5a8008b57aa39f2601546c0d07912045080a5/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp#L1530-L1600).
Matching model files therefore does not guarantee identical numerical results.
This compares the complete CPU backends, including that conversion.

## Before measuring

Collect and review four kinds of evidence on the target machine:

- **Answer quality:** run all 40 fixed questions on KleidiAI and compare each
  answer with the accepted CPU baseline. Keep missing facts and failed cases.
- **Dispatch:** observe actual Q8 kernel entries and the libraries loaded by
  the normal benchmark binaries. A build flag alone is insufficient.
- **State and inputs:** check the full synthetic token streams, attention and
  recurrent snapshots, and a same-backend continuation after restoration.
- **Setup:** check both larger prompt and generation workloads with normal
  binaries. Record per-process peak RSS in bytes, host conditions and duration
  limits. Neither workload is assumed to be universally slowest.

Diagnostic and setup timings do not enter the performance results. Keep their
source changes, commands, raw output and review records under ignored
`artifacts/benchmarks/` directories.
The [diagnostic guide](results/controlled-cpu/diagnostics.md) explains how to
rebuild the published instrumentation and its validation limits.

Each prerequisite record identifies its purpose, model hash and both normal
build-record hashes. `quality_preserved` and `backend_adoption` are separate
from measurement readiness: a rejected backend can still be measured. Mark
evidence approved only after its checks and review are complete.

The record keys are `status: "approved_for_measurement"`, `purpose` (one of
`quality_baseline`, `dispatch`, `depth_state`, `setup_pilot`), `model_sha256`,
and `timing_builds` mapping `cpu` and `kleidiai` to their build-record hashes.
The quality record also contains the two Boolean quality/adoption fields.
Include the supporting evidence and review hashes with each record.

## Workloads and controls

| Workload | Prompt tokens | Generated tokens | Saved prefix tokens |
| --- | ---: | ---: | ---: |
| `pp512` | 512 | 0 | 0 |
| `pp2048` | 2048 | 0 | 0 |
| `tg128-d512` | 0 | 128 | 512 |
| `tg128-d2048` | 0 | 128 | 2048 |

Each process uses four CPU threads, batches of 512, F16 caches, no GPU layers
and no flash attention. Threads use the normal macOS scheduler without core
affinity. The native timer measures model computation;
startup, prefix creation and snapshot restoration are outside it. Context
allocation follows each synthetic workload. The application uses an
8,192-token context. These synthetic-token tests exclude retrieval, template
rendering, tokenization and sampling; they do not measure complete Q&A requests.
Integrity hashing warms the model file cache, so these are not cold-load tests.

Use three sessions with two adjacent CPU/KleidiAI pairs per workload per
session. Freeze a balanced, shuffled order before starting. Each process runs
upstream warmup and four repetitions. Keep every raw sample; exclude sample
index 0 by the frozen rule.

Require AC power, the same power mode, normal observed memory pressure and no
swap activity across a pair. Existing stable swap allocation is allowed.
Unavailable thermal information stays unavailable. Boundary checks do not
prove stable conditions throughout a run. Keep other builds and model jobs
stopped during measurements.

## Freeze, run and report

Start with [the specification example](../configs/benchmark.example.json).
Copy it to `artifacts/benchmarks/spec.json`; fill its `null` timeout and idle
interval from the reviewed setup checks. Keep the chosen values fixed.
`expected_model` comes from native JSON; its `model_size` is tensor bytes,
which differs from the GGUF file's `model.size`.

After setup, record the Mac's host state in a new file. Review this observation
before freezing the plan; collecting it alone does not establish valid conditions.

```sh
uv run --locked --offline --group build --extra retrieval python - <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from benchmark_host import collect

directory = Path("artifacts/benchmarks")
directory.mkdir(parents=True, exist_ok=True)
with (directory / "host-baseline.json").open("x") as stream:
    json.dump(collect(), stream, indent=2)
PY
```

Paths resolve from the current directory. Freezing validates the model,
runtime configuration, builds, host policy and reviewed prerequisites, then
prints the plan's SHA-256 hash:

```sh
uv run --locked --offline --group build --extra retrieval \
  python scripts/benchmark.py freeze --spec artifacts/benchmarks/spec.json \
  --output artifacts/benchmarks/plan.json
```

Run sessions 1, 2 and 3 sequentially, each in a new output directory:

```sh
uv run --locked --offline --group build --extra retrieval \
  python scripts/benchmark.py run --plan artifacts/benchmarks/plan.json \
  --plan-sha256 PLAN_SHA --session 1 --output artifacts/benchmarks/session-1
```

The runner preserves failures and unfinished attempts. It stops admitting work
when host checks fail. It does not retry or replace excluded pairs.

After review, export the supplied sessions using their recorded ledger hashes:

```sh
shasum -a 256 artifacts/benchmarks/session-{1,2,3}/run.json
```

Use those three SHA-256 values as `RUN_1_SHA`, `RUN_2_SHA` and `RUN_3_SHA`:

```sh
uv run --locked --offline --group build --extra retrieval \
  python scripts/benchmark_report.py \
  --plan artifacts/benchmarks/plan.json --plan-sha256 PLAN_SHA \
  --session 1 artifacts/benchmarks/session-1 RUN_1_SHA \
  --session 2 artifacts/benchmarks/session-2 RUN_2_SHA \
  --session 3 artifacts/benchmarks/session-3 RUN_3_SHA \
  --output artifacts/benchmarks/report.json
```

The report verifies raw files and retains all scheduled work. Missing session
evidence remains unknown. Public fields exclude local paths and raw errors.
Keep the referenced local evidence available for verification.

Each included pair contributes one ratio: CPU median duration divided by
KleidiAI median duration. A ratio above 1 favors KleidiAI. Report six planned
pairs per workload and each session's median and range. Repetitions inside a
process are nested observations, not 18 independent experiments. These small
local samples do not establish tail latency or performance on other machines.
