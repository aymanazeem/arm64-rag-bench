# Arm64 RAG Benchmark

A local document question-answering application and an engineering investigation into CPU inference on Apple Silicon: standard Arm64 and KleidiAI-enabled Arm64.

**Status:** local indexing, keyword and embedding search, and document answers with citations are available. The CPU application completed 21 reviewed latency and memory observations on known questions. A valid citation location does not guarantee a correct answer.

## Results

The latest completed CPU/KleidiAI timing study is from **14 November 2026**.
Both backends used Qwen3.5-9B Q8_0 on an **Apple M1 Max with 32 GB memory and four threads**.
KleidiAI processed prompts faster but generated tokens slower.

| Workload | KleidiAI computation time compared with CPU |
| --- | ---: |
| Process 512 prompt tokens | **33.4% less** |
| Process 2,048 prompt tokens | **31.7% less** |
| Generate 128 tokens after a 512-token prefix | **45.0% more** |
| Generate 128 tokens after a 2,048-token prefix | **45.0% more** |

Each percentage uses the median of six paired CPU/KleidiAI duration ratios.
**All 48 runs and 24 pairs completed; every pair recorded swap activity.**
Docker was stopped, other desktop activity continued, and thermal readings were
unavailable. These are descriptive observations from one Mac.
[Full results and method](docs/results/desktop-cpu/README.md).

These timings cover native computation, excluding model loading and the full Q&A
workflow. Separate [application measurements](docs/results/application-performance/README.md)
use CPU only; no KleidiAI application speedup has been measured.

**CPU remains the application default.** KleidiAI passed runtime checks but did
not pass the project's specific [no-regression answer-quality gate](docs/results/controlled-cpu/quality.md).

Both readiness checks for the attempted rerun failed before any model timing began.

## Guides

- [Development and GitHub setup](docs/setup.md)
- [Index and search documents](docs/search.md)
- [Embedding search and comparison](docs/embeddings.md)
- [Ask questions with source citations](docs/answers.md)
- [Build and test the runtime](docs/runtime.md)
- [Runtime evidence and limits](docs/results/runtime-feasibility/README.md)
- [CPU and KleidiAI quality and runtime checks](docs/results/controlled-cpu/README.md)
- [Guarded desktop CPU results](docs/results/desktop-cpu/README.md)
- [Buffered application results](docs/results/application-performance/README.md)
- [Project report](docs/report.md)
- [CPU benchmark methods](docs/benchmarks.md)
- [Evaluation cases and scoring](eval/README.md)
- [Original reference brief](docs/project-brief.md)

## Design

A small Python application handles ingestion, retrieval, citations and evaluation. Pinned llama.cpp binaries handle local inference. CPU comparisons use the same model and settings, with evidence that the intended kernels actually ran. Metal remains an optional, unmeasured extension.

The application uses a command-line workflow and a small licensed technical corpus. Add complexity only when measurements or answer quality justify it. Model weights and private documents stay outside version control.

## Working here

After cloning, enable the repository hooks:

```sh
git config --local core.hooksPath .githooks
git config --local user.name 'Azeem Ayman'
git config --local user.email '110621518+aymanazeem@users.noreply.github.com'
git config --local user.useConfigOnly true
```

Work on feature branches and merge reviewed PRs. To install the pinned tools and inspect the machine:

```sh
uv sync --locked --group build
uv run --locked --group build arm64-rag doctor
```

The runtime guide includes build, generation and diagnostic checks. Plans and agent settings remain local and ignored.
