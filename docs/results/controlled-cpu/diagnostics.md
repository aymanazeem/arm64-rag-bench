# Reproduce the Qwen3.5 diagnostics

The published patch reproduces instrumented native work on Apple Silicon macOS. It does not provide the local controller or parser. Its logging, synchronization and optional replay invalidate every diagnostic timing. Keep normal `build/cpu` and `build/kleidiai` binaries unchanged for the separate [benchmark method](../../benchmarks.md).

Run from the repository root, after the runtime tools and verified answer model are installed. Command Line Tools are needed to compile; debugger or Developer Tools privacy permission is not needed. The selected file is `models/qwen35-9b/Qwen3.5-9B-Q8_0.gguf`, 9,527,502,048 bytes, SHA-256 `809626574d0cb43d4becfa56169980da2bb448f2299270f7be443cb89d0a6ae4`.

Create separate Git checkouts so CMake records the upstream identity rather than this project's parent commit:

```sh
set -eu
diag_root="$PWD/.local/qwen35-reproduction"
mkdir -p .local
mkdir "$diag_root"
mkdir "$diag_root/vendor"
git clone --no-checkout https://github.com/ggml-org/llama.cpp "$diag_root/vendor/llama.cpp"
git -C "$diag_root/vendor/llama.cpp" checkout --detach 8ca5a8008b57aa39f2601546c0d07912045080a5
git clone --no-checkout https://github.com/ARM-software/kleidiai "$diag_root/vendor/kleidiai-v1.32.0"
git -C "$diag_root/vendor/kleidiai-v1.32.0" checkout --detach c80157210987eedaab2509638ef5e15950c53a8c
shasum -a 256 docs/results/controlled-cpu/diagnostic.patch
patch --dry-run -p1 -d "$diag_root" < docs/results/controlled-cpu/diagnostic.patch
patch -p1 -d "$diag_root" < docs/results/controlled-cpu/diagnostic.patch
```

Require patch SHA-256 `90cb617ce5931add8b37cf7356460c0863b6c05f38634ebdd62e207b109bbd7f`. Check all six original and modified file hashes against `docs/results/controlled-cpu/diagnostic-source.json`. Preserve the MIT and Apache-2.0 notices listed in `docs/results/controlled-cpu/NOTICE.md`, including upstream notices in modified files.

Build both variants with the recorded flags. This shell function only avoids repeating the common CMake arguments:

```sh
configure_trace() {
  trace_backend="$1"
  shift
  uv run --locked --offline --group build --extra retrieval cmake \
    -S "$diag_root/vendor/llama.cpp" -B "$diag_root/build/$trace_backend" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DGGML_CPU=ON -DGGML_CPU_REPACK=ON \
    -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_ACCELERATE=OFF -DGGML_OPENMP=OFF \
    -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON "$@"
  uv run --locked --offline --group build --extra retrieval cmake \
    --build "$diag_root/build/$trace_backend" --target llama-bench -j 4
}
configure_trace cpu -DGGML_CPU_KLEIDIAI=OFF
configure_trace kleidiai -DGGML_CPU_KLEIDIAI=ON \
  "-DFETCHCONTENT_SOURCE_DIR_KLEIDIAI=$diag_root/vendor/kleidiai-v1.32.0"
```

Use a clean compiler environment as in the runtime guide. Retain CMake cache, compile commands, compiler version and executable/library hashes. `GGML_NATIVE=ON` was the observed upstream default. A different compiler or host may produce different binaries and dispatch; matching source hashes does not promise byte-identical builds.

Run one backend at a time, starting with the small prompt probe. Substitute `kleidiai` for `cpu` and create a new output directory for each invocation:

```sh
diag_backend=cpu
diag_output="$diag_root/smoke-$diag_backend-pp32"
mkdir "$diag_output"
env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
  "$diag_root/build/$diag_backend/bin/llama-bench" \
  -m "$PWD/models/qwen35-9b/Qwen3.5-9B-Q8_0.gguf" \
  -p 32 -n 0 -d 0 -r 1 --offline \
  -t 4 -b 512 -ub 512 -ctk f16 -ctv f16 -fa off -ngl 0 \
  -lm auto -lzm auto --repack 1 -o json \
  > "$diag_output/stdout.json" 2> "$diag_output/stderr.log"
```

Keep upstream warmup enabled. Use these exact workload arguments with the same shared flags:

| Workload | Arguments |
| --- | --- |
| Small depth probe | `-p 0 -n 2 -d 32 -r 2` |
| Prompt 512 | `-p 512 -n 0 -d 0 -r 4` |
| Prompt 2048 | `-p 2048 -n 0 -d 0 -r 4` |
| Generation after depth 512 | `-p 0 -n 128 -d 512 -r 4` |
| Generation after depth 2048 | `-p 0 -n 128 -d 2048 -r 4` |

For a separate state-continuation invocation, use either full depth workload and add `Q35_STATE_PROBE_DIR="$diag_output"` to the `env -i` assignments. The directory must be new. It receives `state-reference.f32` and `state-replay.f32`; the probe checks finite, bitwise-equal full-vocabulary logits and positions after one fixed-token continuation. It restores only that backend's own snapshot and consumes no additional random draws. This is not a cross-backend state comparison or proof of every future continuation.

The recorded controller imposed 600-second/64-MiB bounds for smoke and 1,200-second/128-MiB bounds for full input/state runs, plus the two replay files. These manual commands do not implement those bounds, failure ledgers or automatic validation. Use bounded process supervision for equivalent orchestration, keep failures, and never overwrite an earlier attempt.

`stdout.json` is the upstream benchmark array; its timing values are excluded. `stderr.log` includes `Q35 ` JSON lines for actual inputs, sampled kernel entries, selected CPU fallbacks, loaded Mach-O images and hybrid state events. Kernel tracing covers the first batch of each phase in repetition zero, not every operation. The patch instruments two M1 DotProd Q8 entry points; another dispatched kernel needs separate instrumentation. Loaded images describe the diagnostic process only.

The public source manifest lets readers verify and apply the exact instrumentation. The raw native records allow an independent parser to check phase/order, kernel-call pairing, token streams and state sizes. Input digests use token IDs encoded as unsigned 32-bit little-endian bytes. Original local parser/controller hashes bind the retained validation records; hashes alone neither publish those helpers nor let readers rerun their automated checks.

Normal loaded libraries require a separate untimed observation of each unmodified binary: use the same clean environment and shared flags, `-p 1 -n 0 -d 0 -r 1`, and add only `DYLD_PRINT_LIBRARIES=1`. Retain and review the actual loader output; the patch does not authenticate normal binaries. Timed runs must have no `DYLD_*` or `Q35_*` controls. Review the observed loader format before treating it as evidence.
