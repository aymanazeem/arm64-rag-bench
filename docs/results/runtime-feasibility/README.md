# Runtime feasibility: M1 Max

Date: 2026-11-12. Hardware: Apple M1 Max, 32 GB memory, macOS 26.7.1.

Both CPU builds run the official Qwen 1.5B Q4_0 model. A separate diagnostic build confirmed the two expected Q4_0 DotProd kernel paths. This establishes a useful local test platform; it does not establish a speedup or RAG answer quality.

| Check | Result | Evidence |
| --- | --- | --- |
| Controlled CPU builds | Both succeeded; effective settings checked | [CPU](cpu-build.json), [KleidiAI](kleidiai-build.json) |
| Basic generation | Two cases passed on each backend | [Commands and answers](smoke.json) |
| Prompt-processing kernel | 193 calls with `m=32` | [Dispatch record](dispatch.json), [sample](prefill-sample.txt) |
| Generation kernel | 1,568 calls with `m=1` | [Dispatch record](dispatch.json), [sample](decode-sample.txt) |
| Benchmark output | Six valid samples per backend | [Validation](bench-check.json), [CPU samples](cpu-bench.json), [KleidiAI samples](kleidiai-bench.json) |
| Local tests | 44 passed, including upstream-source validation | See commands in the [runtime guide](../../runtime.md) |

The [hardware inventory](hardware.json) excludes personal identifiers. Build records include source, compiler, settings and binary/library hashes. Assets are pinned in [runtime.json](../../../configs/runtime.json).

## Dispatch evidence

Debugger attachment was unavailable. The diagnostic script instead patched a separate copy of the verified KleidiAI source to log entry into the two named kernels. The [patch](kernels.patch) adds only logging; the normal source and builds are unchanged.

Each probe used one thread, no GPU layers and no warm-up. Prompt processing used 32 prompt tokens and no generated tokens. Generation used eight generated tokens. The parser required the exact kernel name and positive workload dimensions. Both processes exited successfully. Valid calls in these wrappers proceed to the named assembly kernels.

The dispatch record includes source-tree, patch, executable and library hashes. The text files contain clearly labelled excerpts; full stderr logs remain under `artifacts/feasibility/dispatch/` locally. Reproduce them with [the diagnostic command](../../runtime.md#prove-kernel-execution-without-debugger-access).

Diagnostic timings are invalid for performance comparisons. These probes cover two Q4_0 paths, not every operation or tensor in the model.

## Limits

- The generation checks are two simple cases, not the application's quality evaluation.
- Benchmark samples validate the output format. They use one session, fixed backend order and small workloads; power and thermal conditions were not controlled. No performance conclusion is drawn.
- The pinned integration can requantize Q6_K/Q8_0 weights. This model has a Q6_K output matrix. Equal model files do not guarantee identical internal arithmetic.
- Normal binary hashes were rechecked after the diagnostic build. The published normal-build samples contain no tracing overhead.
- Local CI-style tests do not measure this Mac's performance. The upstream-source test is skipped in ordinary CI, where downloaded sources are absent.
