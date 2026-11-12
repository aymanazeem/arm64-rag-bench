# Runtime checks

This setup builds a controlled CPU pair on Apple Silicon. The application and full RAG benchmark are not implemented yet.

## Install and test

Use macOS on Apple Silicon with Command Line Tools and [uv](https://docs.astral.sh/uv/getting-started/installation/). uv installs the pinned Python version and project tools; it does not replace system Python.

```sh
uv sync --locked --group build
uv run --locked --group build arm64-rag doctor
uv run --locked --offline --group build ruff check .
uv run --locked --offline --group build ruff format --check .
uv run --locked --offline --group build pytest
```

Tests use small local fixtures. They do not download models. The upstream-document check is optional; enable it after downloading the pinned sources:

```sh
ARM64_VERIFY_UPSTREAM=1 uv run --locked --offline --group build pytest
```

## Build the CPU pair

```sh
uv run --locked --group build python scripts/runtime.py build cpu
uv run --locked --group build python scripts/runtime.py build kleidiai
uv run --locked --group build python scripts/runtime.py download-model
```

[runtime.json](../configs/runtime.json) pins llama.cpp, KleidiAI and the official Qwen 1.5B Q4_0 model. Downloads are checked by size and SHA-256. Sources, models and build output stay outside Git.

Both Release builds disable Metal, BLAS, Accelerate and OpenMP. Normal CPU repacking stays enabled. Only the KleidiAI setting differs. The script starts with fresh CMake settings, checks the effective configuration and rejects inherited compiler flags. It preserves changed source directories rather than overwriting them.

Each successful build writes logs, its CMake cache, compile commands and a hash manifest under `artifacts/feasibility/<backend>/`. A failed attempt removes the old success manifest. These controlled builds are not a claim about the fastest possible Mac configuration.

## Check generation

Run this for each backend, replacing `cpu` with `kleidiai` for the second check:

```sh
build/cpu/bin/llama-completion \
  -m models/qwen2.5-1.5b-instruct-q4_0.gguf \
  --single-turn --jinja \
  -p 'Answer with only the city name: What is the capital of France?' \
  -n 24 -c 512 -t 4 -tb 4 -ngl 0 --temp 0 --seed 42 \
  --no-display-prompt --simple-io --no-warmup
```

The expected answer is `Paris`; the program may also print its end marker. This is a smoke check, not an answer-quality benchmark. The separate [evaluation contract](../eval/README.md) defines source-grounded cases for later application tests.

## Prove kernel execution without debugger access

```sh
uv run --locked --group build python scripts/trace_dispatch.py --jobs 4
```

The script copies verified KleidiAI source into `.local/kleidiai-trace`, adds a log statement to two named Q4_0 kernels and builds a separate `llama-bench`. It checks prompt processing and generation independently, with warm-up disabled. It fails if the expected kernel names and workload dimensions are absent.

Evidence is saved under `artifacts/feasibility/dispatch/`: the patch, source hashes, binary/library hashes, commands and raw logs. Normal sources and CPU builds are untouched. Debugger permission is not required.

**Never use diagnostic-build timings for performance claims.** The log statements add overhead. Successful probes establish execution of the named kernels in those workloads, not complete model coverage or a speedup.

## Check the benchmark output

After all builds finish, run each normal backend separately:

```sh
build/cpu/bin/llama-bench \
  -m models/qwen2.5-1.5b-instruct-q4_0.gguf \
  -p 128 -n 32 -r 3 -t 4 -b 128 -ub 128 \
  -ngl 0 -fa off -ctk f16 -ctv f16 -o json
```

Replace `cpu` with `kleidiai` for the other build. This small run checks that the tool produces valid per-repeat data. It is not a controlled final comparison: final runs need balanced order, independent sessions, representative context lengths and quality checks.

The pinned integration can requantize some Q6_K/Q8_0 weights during loading. The smoke model contains a Q6_K output matrix, despite its Q4_0 filename. Equal model files therefore do not imply identical arithmetic. Report the integration's effect and test answer quality separately. [Pinned packing source](https://github.com/ggml-org/llama.cpp/blob/8ca5a8008b57aa39f2601546c0d07912045080a5/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp#L1530).
