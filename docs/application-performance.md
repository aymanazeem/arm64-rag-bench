# Compare the application on both backends

This harness times the unchanged `arm64-rag ask` command with the pinned CPU and KleidiAI servers. Only `--server-binary` differs. The model, index, prompt, settings, four threads, seed and token budgets are the same. The [results](results/application-performance/README.md) report what it measured.

## Design

- **Questions:** all 40 fixed questions, plus a question with no matching passages. Each backend's sampled answer repeats exactly for a question, but answers differ between backends, so their lengths differ too. Using every question shows the length effect as it occurs instead of depending on a few cases.
- **Pairs:** each question runs once on each backend, back to back. Within each block of ten questions, CPU goes first five times and KleidiAI five times. The no-match control runs once per block, twice with each backend first. Seed 42 fixes the order. 88 observations in total.
- **Rest:** 120 guarded idle seconds before every observation, including between the two halves of a pair.
- **Fresh command:** every observation is a new process that verifies the model, starts a server, retrieves passages, generates and shuts down. Fresh processes do not mean cold file caches: model verification reads the file.
- **No retries:** the first failure stops collection. Later rows stay unattempted and every attempted row is kept.

The analysis is fixed in `compare()` before collection: total time over the 40 questions, per-question KleidiAI/CPU ratios, start-up, prompt and generation components, pooled token rates, a subset where answer lengths are within 5%, results by which backend ran first, the no-match control and peak memory.

## Prepare

Use the [runtime guide](runtime.md) to build both pinned servers and download the [answer model](answers.md). Build the corpus index:

~~~sh
uv run --locked --group build arm64-rag index \
  --manifest data/manifests/corpus.json \
  --db data/private/development.sqlite --sync
~~~

Preparation checks the index against the public manifest and questions. It freezes both builds, the model, the application sources, the reviewed answer records, the host policy and the schedule. It does not start the model.

~~~sh
uv run --locked --group build python scripts/benchmark_answers.py prepare \
  --db data/private/development.sqlite \
  --output artifacts/benchmarks/application-comparison/plan.json
~~~

The two builds must come from the same llama.cpp commit, compiler and flags, except `GGML_CPU_KLEIDIAI`. Do not change the frozen files until collection ends.

## Collect

Use AC power, keep the lid open and close other applications. The harness holds an idle-sleep assertion and stops if a clock jump shows the Mac slept. Apple's [idle-sleep assertion](https://github.com/apple-oss-distributions/IOKitUser/blob/main/pwr_mgt.subproj/IOPMLib.h#L253-L270) has no effect during Dark Wake, so also hold a system-sleep assertion in a second terminal, for example `caffeinate -s -t 32400`, and release it afterwards. See Apple's [caffeinate manual](https://github.com/apple-oss-distributions/PowerManagement/blob/main/caffeinate/caffeinate.8).

Run one excluded compatibility pair (the median-prompt-length question `val-015` on both servers), then the comparison:

~~~sh
uv run --locked --group build python scripts/benchmark_answers.py check \
  --plan artifacts/benchmarks/application-comparison/plan.json \
  --output artifacts/benchmarks/application-comparison/check

uv run --locked --group build python scripts/benchmark_answers.py run \
  --plan artifacts/benchmarks/application-comparison/plan.json \
  --output artifacts/benchmarks/application-comparison/run
~~~

A plan can be collected once; a create-only launch record beside it prevents reuse. Start-up keeps its 120-second deadline, requests their 300-second deadline, and each observation has a 600-second limit.

## Metrics

| Metric | Boundary |
| --- | --- |
| Answer time | Attempt checkpoint to the first answer bytes received from the command's output pipe. |
| Server start-up | Server entry to validated readiness, including model and binary hashing. |
| Complete request | The existing request call, including its input checks. |
| Prompt and generation | The server's own timings for reading the prompt and generating tokens. |
| Peak memory | Peak resident memory of the reaped server process. |

The command buffers its answer, so time to first token is not observed. The attempt checkpoint comes before ledger writing, worker creation and the ownership handshake, so answer time includes those steps. Only the benchmark worker changes the native launch, to inherit its process group; the command, arguments and requests stay unchanged.

Each answer is compared with the reviewed quality record for the same question and backend. If its text and token count match exactly, the existing [quality review](results/controlled-cpu/quality.md) describes it.

## Report

~~~sh
uv run --locked --group build python scripts/benchmark_answers.py report \
  --plan artifacts/benchmarks/application-comparison/plan.json \
  --run artifacts/benchmarks/application-comparison/run/run.json \
  --output artifacts/benchmarks/application-comparison/report.json
~~~

The report checks raw file hashes, inputs, cleanup, host readings and clock evidence. It computes the comparison only when every observation succeeded with valid timing. Raw traces contain source passages and model reasoning; they stay local. The public report keeps metric values, counts, labels and hashes, not raw text, local paths, process IDs or environment values.

## Runtime evidence

After collection, start each measured server once with the application's server arguments plus `-v`, and keep its weight-buffer and repack lines. Sample one start and one short request with macOS `/usr/bin/sample` to see where loading spends its time and which matrix kernels run. Unnamed addresses can be resolved with `nm -n` on the build's `libggml-cpu` library. These diagnostic starts are not measurements.
