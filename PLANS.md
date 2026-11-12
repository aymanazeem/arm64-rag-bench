# Arm64 RAG implementation plan

Updated: 2026-11-12. Owner: Azeem Ayman. Public repository: `aymanazeem/arm64-rag-bench`.

## Outcome and boundaries

Build a useful, fully local document-Q&A application with source citations, then measure what KleidiAI changes on the actual machine. Deliver reproducible code, versioned evaluation cases, raw experimental evidence and a concise report. The initial target is the observed **M1 Max, 32 GB unified memory, 8 performance plus 2 efficiency cores**, not the generic 8/16 GB M1 assumption in the reference brief.

The owner made the repository public. Work uses feature branches, reviewed PRs, minimal code and comments, and concise English. Preserve behavior and answer quality. Paid cloud hardware requires approval. No application, native build or performance result is complete yet.

## Critical assessment

1. **The central experiment is plausible but unproven.** Current llama.cpp documents Apple arm64 support and includes Apple Q4_0 DotProd kernel selections. A successful build or a `CPU_KLEIDIAI` buffer log is insufficient evidence of execution. Prove dispatch before spending time on a polished interface. Sources: [build support](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#arm-kleidiai), [kernel selection](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/kleidiai/kernels.cpp).
2. **Default macOS builds can spoil the comparison.** Metal and Apple BLAS can introduce different execution paths. Disable both for the controlled CPU pair, keep ordinary CPU optimization/repacking enabled in both, and compare full build configurations. An Apple BLAS CPU build may be added later as a practical baseline. Source: [GGML options](https://github.com/ggml-org/llama.cpp/blob/master/ggml/CMakeLists.txt).
3. **Quantization is part of the experiment.** Kernel coverage depends on the pinned runtime, tensor types, dimensions and CPU capabilities. Start with Q4_0 for a dispatch probe; choose the application's eventual model and quantization using quality evidence. A change of model or quantization is a separate experiment. Source: [dispatch implementation](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp).
4. **Fast generation does not establish fast RAG.** Embedding, retrieval, prompt assembly, cache behavior and startup all affect user latency. Separate inference-only, frozen-context RAG and full-pipeline experiments. `llama-bench` excludes tokenization and sampling. Source: [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md).
5. **Quality needs an explicit contract.** Preserve deterministic ingestion, ranking tie breaks, prompt formats and citation behavior exactly where appropriate. For generated answers, use frozen factual/evidence requirements and inspect regressions; cross-backend floating-point behavior can change wording. Specify cache policy instead of relying on server defaults. Source: [llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
6. **A tutorial is not the architecture.** The supplied Arm learning path targets Linux/cloud Arm hardware. Use it as a reference and validate local assumptions. Do not import its server dependencies or hardware requirements into the Mac project. Source: [Arm learning path](https://learn.arm.com/learning-paths/servers-and-cloud-computing/rag/).

These source URLs describe evolving upstream code. Milestone 1 must replace implementation-critical references with commit permalinks and record a retrieval date before producing results.

## Initial architecture decisions

| Area | Starting decision | Reason and reconsideration trigger |
| --- | --- | --- |
| Application | Small Python package and CLI | Fast iteration, inspectable data flow; add a UI only after Q&A and evaluation work |
| Environment | uv-managed supported Python, committed `uv.lock` | Reproducible environment without changing system Python; verify native dependencies before choosing the exact Python minor |
| Inference | Pinned external llama.cpp builds, local `llama-server` | Keeps runtime flags and execution visible; avoids Python binding rebuilds obscuring backend differences |
| Embeddings | Small local model, fixed independently of generation | Prefer a compatible llama.cpp embedding endpoint if quality and pooling support suffice; otherwise document the smallest suitable local dependency |
| Storage | SQLite metadata and versioned local index | Atomic updates, simple inspection and no service to operate |
| Retrieval | Exact normalized-vector search with NumPy; FTS5 lexical baseline | Small corpora do not need ANN infrastructure; compare lexical retrieval on technical identifiers before adding hybrid search |
| Orchestration | Explicit ingest → retrieve → prompt → generate functions | No agent framework, vector database, remote inference API or distributed queue needed initially |
| Interface | CLI first; server bound to loopback | Makes behavior easy to test; expose no network service beyond the machine by default |
| Testing | pytest for behavior; Ruff for lint/format once code exists | Keep offline unit checks quick; native integration and full evaluation are separate jobs |

The proposed layout grows only as functionality is implemented: `src/arm64_rag/`, `tests/`, `configs/`, `scripts/`, `data/manifests/`, `eval/`, and `docs/`. Do not create empty module hierarchies. Default local indexes to `.local/indexes/`, models to `models/`, native sources to `vendor/`, native builds to `build/` and large run artifacts to `artifacts/`; these are ignored. Commit small curated evidence only after checking provenance and privacy.

Further primary references: [SQLite FTS5/BM25](https://sqlite.org/fts5.html), [uv lock checks](https://docs.astral.sh/uv/concepts/projects/sync/). Use `uv sync --locked` and `uv run --locked` in automation so stale locks fail visibly.

## Milestones and acceptance gates

### 0 — Private repository foundation

Branch: `chore/project-bootstrap`.

- Add working agreements, a minimal project Codex configuration, ignore rules, PR template, editor settings and this plan.
- Configure personal identity locally; enable branch/identity/attribution hooks. Preserve the global work identity.
- Create the private personal repository after verifying authentication. Publish an empty base, then introduce all files through a reviewed setup PR.
- Check available branch protection; record limits rather than implying local hooks enforce GitHub policy.

Acceptance: private owner/remote verified; setup PR reviewed and merged; no credentials, downloaded weights or work identity in project commits. Local hook positive/negative checks and diff checks pass. Add application CI in milestone 1 when executable checks exist.

### 1 — Reproducible environment and feasibility probe

Branch: `feat/runtime-feasibility`.

1. Select a supported Python minor; create `pyproject.toml`, lock dependencies, add lint/test commands and fast offline CI. Pin Actions by immutable revision when adding workflows.
2. Record CPU feature flags, OS/compiler/tool versions and sanitized hardware inventory. Provision local CMake/Ninja if required. Pin llama.cpp and record the exact bundled KleidiAI revision, source hashes, flags and build logs.
3. Build two Release CPU variants from identical source: `GGML_METAL=OFF`, `GGML_BLAS=OFF`, with `GGML_CPU_KLEIDIAI=OFF` versus `ON`. Save full CMake caches and compiler commands. Do not disable normal CPU repacking only in the baseline.
4. Choose one modest dense 1–3B instruct model for the smoke test. Check its primary model card, license, architecture support and GGUF provenance. Pin upstream revisions and SHA-256; record conversion and quantization if performed locally. Use Q4_0 as the first dispatch probe.
5. Run fixed prefill and decode workloads. Prove a named KleidiAI kernel executes in each relevant operation using a debugger backtrace or a minimal diagnostic counter patch. Record unsupported/fallback operations. Allocation messages alone fail this gate.
6. Keep instrumented evidence runs separate from uninstrumented timed binaries. Confirm runtime stability, finite output and memory pressure. Run a tiny CPU-pair benchmark for harness validation, without claiming final speedups.

Acceptance: a clean checkout can reproduce the pinned build procedure; both variants run the same model; dispatch evidence or a precise failure explanation is saved. If KleidiAI cannot execute, identify why and continue useful local RAG work; a supplemental Arm host is a documented proposal requiring cost approval, not an automatic purchase. Do not promise acceleration before this gate.

### 2 — Deterministic corpus and retrieval

Branch: `feat/local-retrieval`.

1. Select a small, versioned, redistributable technical corpus. Keep source URL/revision, license, document hash and extraction version. Include synthetic fixtures; exclude work documents.
2. Support UTF-8 text and Markdown first. Add text-based PDF only if needed for the selected corpus; defer OCR and arbitrary web crawling.
3. Implement stable document/chunk IDs, heading/source spans, bounded chunking and deterministic overlap. Record tokenizer/chunker versions and parameters; never silently truncate.
4. Store metadata and index fingerprints atomically; support idempotent re-indexing, source updates/deletions and interrupted-write recovery.
5. Freeze embedding revision, pooling, normalization, query/document prefixes, vector dimension and truncation behavior. Reject incompatible query/index configurations.
6. Implement exact vector retrieval and a small FTS5 lexical comparator. Establish approximately 20 development queries with expected source spans; select settings on that set alone. Add hybrid retrieval or reranking only with measured benefit.

Acceptance: deterministic fixtures, source mapping, duplicate/update/delete behavior, index compatibility and failure recovery pass. Report development Recall@k and reciprocal rank separately for lexical and dense retrieval, including actual index size and retrieval latency.

### 3 — Grounded local question answering

Branch: `feat/grounded-qa`.

1. Manage a local inference process with bounded startup/readiness, explicit shutdown, timeouts and readable errors. Keep embedding configuration fixed across generation-backend tests.
2. Version prompt construction: ordered chunk IDs, exact context text, chat template, context budget, reserved generation tokens and truncation decisions.
3. Return answers with resolvable citations. Preserve supporting spans so citation validity is checkable, and abstain when evidence is insufficient. Treat document instructions as untrusted text.
4. Choose the main model using the development set and observed memory/latency. The 32 GB machine permits investigating a 7–8B candidate if the smaller model's quality is insufficient; memory fit must still be measured at the chosen context length.
5. Expose practical CLI commands for indexing, querying and inspecting sources; add example commands only after verifying them.

Acceptance: the CLI answers a documented end-to-end example without a hosted model service. Tests cover citations, context limits, empty indexes, malformed documents, unavailable runtime, unsupported settings and instruction-like text inside retrieved documents. No placeholder output is presented as an inference result.

### 4 — Frozen evaluation and quality baseline

Branch: `feat/quality-evaluation`.

1. Build a held-out set of at least 50 reviewed cases, separate from tuning examples. Include answerable, unanswerable, conflicting-source and multi-document questions plus an instruction-in-document case.
2. Record expected evidence spans and factual requirements. Report retrieval Recall@k/MRR, answer correctness, citation support/coverage and abstention separately.
3. Use exact checks for deterministic components and a documented rubric for answers. Automated/model grading is supplementary; audit sampled outputs and all new critical failures against sources.
4. Freeze corpus/model/configuration hashes and retain outputs. Repeat generation cases where instability matters. Set regression tolerances from baseline variability before performance tuning, not after seeing experimental results.

Acceptance: no new critical factual/citation failures; all known correct behavior retained; unexplained quality deterioration blocks a performance merge. Report uncertainty and failures alongside aggregate scores. Baseline changes require explicit justification and a version bump, not edited expected answers to make tests pass.

### 5 — Controlled benchmark matrix

Branch: `feat/benchmark-suite`.

- Compare standard CPU and KleidiAI CPU first; then add Metal as a separately labeled execution path. A BLAS CPU variant is optional and distinct from the controlled pair.
- Fix model/tokenizer bytes, prompt bytes, retrieval outputs, context length, generated-token count, decoding/seed, batch and microbatch sizes, KV precision, flash-attention settings, thread counts and cache policy. Record unsupported configuration differences explicitly.
- Explore CPU threads at 1/4/8/10 in a pilot; freeze the final settings before measurement. Eight threads do not establish performance-core affinity. Separate matched-setting comparisons from each backend's tuned best setting.
- Test prefill, decode and combined workloads with `llama-bench`; separately measure startup/load, query embedding, retrieval, time to first token, generation and total CLI/request latency using a monotonic clock.
- Explicitly disable prompt-cache reuse for fresh-request tests; label cache reuse as a separate workload. Separate filesystem/page cache, model warm-up and prompt/KV reuse. Do not use privileged cache purging as an unrecorded condition.
- Use a pilot to choose repeat counts; start with at least 10 paired observations per condition across at least 3 independent sessions for the final small matrix. Alternate/randomize backend order and retain session identity, warm-up, power mode, AC/battery state and thermal/memory-pressure observations.
- Save every observation and failure, including excluded samples with reasons. Report medians, variation and paired uncertainty intervals; treat sessions as clusters when estimating uncertainty. Avoid presenting unstable tail percentiles from a tiny sample as reliable.
- Record process memory and system memory pressure/swap; document measurement units and the limits of comparing CPU and unified-memory GPU accounting. Report tokens/s separately from user latency.

Acceptance: repeatable commands, complete manifests/raw samples, actual dispatch evidence, paired comparisons with uncertainty and the frozen quality report. No kernel-specific causal claim without evidence. Negative results remain in the report. CI validates benchmark parsing/aggregation with small fixtures; final performance runs happen on the recorded target Mac, not an unrelated hosted CI runner.

### 6 — Reproduction and technical report

Branch: `docs/benchmark-report`.

- Reproduce setup from a clean checkout using documented commands and pinned artifacts. Confirm the app can run offline once its licensed assets are installed.
- Explain hardware, runtime/model/data versions, dispatch proof, methodology, quality, timing, memory and limitations. Link results to exact configuration and source commits.
- Separate observed conclusions from hypotheses; limit generalization to measured hardware/workloads. Explain discrepancies and any failed runs.
- Add a minimal UI only if it demonstrably helps use the completed Q&A application and can reuse the existing interfaces; keep it outside benchmark timing.

Acceptance: another developer can run the example and reproduce the experiments. Retained evidence supports each conclusion; published data has clear provenance and licensing.

## Execution, review and status

For each milestone: branch → implement the smallest complete increment → run relevant checks → independent review → fix findings → commit and push → PR → confirm checks → merge. Split a milestone when it becomes too large to review coherently. Authentication failures block remote operations, not unrelated local work. Use concise messages and only the owner's identity.

| Milestone | Status | Evidence / next action |
| --- | --- | --- |
| 0: foundation | Complete: [PR #1](https://github.com/aymanazeem/arm64-rag-bench/pull/1) | Setup checks passed. The owner made the repository public; branch protection now requires PRs, including for administrators. |
| 1: feasibility | Not started | Pin toolchain/runtime/model, build controlled CPU pair and establish dispatch evidence |
| 2–6 | Not started | Begin in dependency order after the preceding acceptance gates |

No model downloaded, application implemented, inference build completed, speedup measured or answer-quality baseline established at this stage.
