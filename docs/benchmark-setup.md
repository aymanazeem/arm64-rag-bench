# Reproduce desktop setup

These commands run on Apple Silicon macOS after the [runtime builds](runtime.md)
and [answer-model download](answers.md). Use a fresh checkout and new ignored
output paths. The public runner needs no private helper or debugger permission.
Its setup observations are excluded from final performance results.

The [native workflow check](results/public-setup/README.md) completed all four
setup observations using cached assets; its candidate remains pending review.

Copy [desktop-method.example.json](../configs/desktop-method.example.json) to
`artifacts/benchmarks/method.json`. Give the study a new ID, record the target
machine, background services and prior observations, then review the method.
Keep its schedule, setup limits and analysis fixed. Copy
[benchmark.example.json](../configs/benchmark.example.json) to
`artifacts/benchmarks/spec.json`, add `"study": "desktop"` and
`"method": "artifacts/benchmarks/method.json"`, and check all input paths.
The method supplies the rest intervals. Setup does not read the specification's
unfilled measurement deadline or prerequisite approvals.
The runner requires the example's declared calibration formula; a missing or
different formula is rejected before setup starts.

Freeze the setup inputs before any setup run:

```sh
uv run --locked --offline --group build python scripts/benchmark_setup.py freeze \
  --spec artifacts/benchmarks/spec.json --output artifacts/benchmarks/setup-plan.json
```

Save the printed SHA-256 as `SETUP_PLAN_SHA`. The plan pins the model bytes, both
normal builds, compile commands, libraries, runtime configuration, method and
public helpers. Editing those inputs requires a new prospective study; preserve
earlier attempts. Freezing hashes the model and warms the file cache.

Use AC power and stop other model, build, test and benchmark work. Record any
service changes and restore the prior state afterward. Run these commands in
order, stopping if any command fails:

```sh
uv run --locked --offline --group build python scripts/benchmark_setup.py run \
  --plan artifacts/benchmarks/setup-plan.json --plan-sha256 SETUP_PLAN_SHA \
  --backend cpu --workload pp2048 --output artifacts/benchmarks/setup-runs
uv run --locked --offline --group build python scripts/benchmark_setup.py run \
  --plan artifacts/benchmarks/setup-plan.json --plan-sha256 SETUP_PLAN_SHA \
  --backend kleidiai --workload pp2048 --output artifacts/benchmarks/setup-runs
uv run --locked --offline --group build python scripts/benchmark_setup.py run \
  --plan artifacts/benchmarks/setup-plan.json --plan-sha256 SETUP_PLAN_SHA \
  --backend cpu --workload tg128-d2048 --output artifacts/benchmarks/setup-runs
uv run --locked --offline --group build python scripts/benchmark_setup.py run \
  --plan artifacts/benchmarks/setup-plan.json --plan-sha256 SETUP_PLAN_SHA \
  --backend kleidiai --workload tg128-d2048 --output artifacts/benchmarks/setup-runs
shasum -a 256 artifacts/benchmarks/setup-runs/setup-ledger.json
```

The runner records commands, raw output, host readings, child resource usage and
continuous-clock evidence. It checks the previous record and clock checkpoint
before the next launch. Each process has a 1,200-second deadline and a 16-MiB
combined output limit. Swap increases remain visible; other host or timing
failures stop continuation. Never retry or replace a failed observation.

Use the ledger's SHA-256 as `SETUP_LEDGER_SHA`:

```sh
uv run --locked --offline --group build python scripts/benchmark_setup.py report \
  --plan artifacts/benchmarks/setup-plan.json --plan-sha256 SETUP_PLAN_SHA \
  --runs artifacts/benchmarks/setup-runs --ledger-sha256 SETUP_LEDGER_SHA \
  --output artifacts/benchmarks/setup-candidate.json
```

All four valid observations are required for calibration. The candidate uses
twice the longest continuous process duration, rounded up to a minute, with a
600-second minimum. Partial or failed setup remains `incomplete`, with no
deadline. A complete candidate remains `pending_review`.

Review the frozen method, raw output, resource reports, host findings, timing
evidence and calibration. Record that review separately with its evidence
hashes. Only then copy the candidate to the specification's setup approval path,
retain its evidence fields, add the review hash and set its status to
`approved_for_desktop_measurement`. Copy its deadline and two rest intervals into
the measurement specification. The runner cannot perform this scientific review
or establish answer quality from elapsed time.

Collect and review the other [benchmark prerequisites](benchmarks.md#before-measuring)
for the actual model and builds. Fresh CPU/KleidiAI answers can use a
[local input baseline](answers.md#compare-a-fresh-cpu-baseline); diagnostic
instrumentation has a separate [manual guide](results/controlled-cpu/diagnostics.md).
Then collect a fresh host baseline and follow
[freeze, run and report](benchmarks.md#freeze-run-and-report).

Published summaries and hashes describe historical evidence. Original raw answer
traces, native logs, local review notes and old private controllers are not
distributed. A fresh checkout can collect a new study using this public setup
runner; it cannot recreate missing historical records or rerun the old private
parsers from hashes alone. Local setup candidates contain evidence identifiers,
not a general public-report sanitization guarantee; review files before publishing.
