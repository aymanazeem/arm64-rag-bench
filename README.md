# Arm64 RAG Benchmark

A local document question-answering application and an engineering investigation into CPU inference on Apple Silicon: standard Arm64 and KleidiAI-enabled Arm64.

**Status:** local indexing, keyword and embedding search, and document answers with citations are available. CPU is the application default. A valid citation location does not guarantee a correct answer.

## Results

Both backends ran Qwen3.5-9B Q8_0 on an **Apple M1 Max with 32 GB memory and
four threads**. The comparison completed **48 runs and 24 paired comparisons in
one uninterrupted invocation**, with **no observed swap activity** or detected
timing interruption.

| Workload | KleidiAI computation time compared with CPU |
| --- | ---: |
| Process 512 prompt tokens | **33.5% less** |
| Process 2,048 prompt tokens | **31.7% less** |
| Generate 128 tokens after a 512-token prefix | **43.0% more** |
| Generate 128 tokens after a 2,048-token prefix | **42.1% more** |

Percentages use the median of six paired CPU/KleidiAI mean-duration ratios. All
six pairs for each workload favor the same backend. CPU processed about 59–62
prompt tokens per second and generated about 12; KleidiAI processed about 86–92
and generated about 8.4–8.6. All observations are retained.
[Full results and method](docs/results/desktop-cpu/README.md).

**Why:** on this M1 Max, KleidiAI's kernels add up each weight row in one
unbroken running total per token; the CPU kernels restart theirs every 32
weights. In kernel tests, with 16 prompt tokens' totals running at once,
KleidiAI kept the multiply units about 98% busy, against 60% for CPU, whose
restarts cost extra instructions. With one token, KleidiAI's kernel has a single
total, so each multiply instruction waits about 3 cycles for the previous one;
at four threads it processed 73 billion weights per second, against 101 billion
for CPU. [Kernel timings](docs/results/kernel-analysis/README.md).

### Answer quality

Both backends answered all **40 fixed questions** without execution or format
failures. All 80 cases remain in the reviewed results.

| Backend | Valid format | Supported claims | Required facts | Complete cases |
| --- | ---: | ---: | ---: | ---: |
| CPU | 40/40 | **78/78** | 66/83 | **30/40** |
| KleidiAI | 40/40 | 80/81 | 67/83 | **30/40** |

Complete cases pass every applicable review criterion. Supported claims agree
with both the supplied text and their attached citations. Each backend had 30
complete cases, 28 of them the same. KleidiAI supplied three facts CPU missed, missed
two facts CPU supplied, and made one claim with an unsupported scheduling clause.

The project's frozen quality rule rejects a backend that loses any fact CPU
supplied or adds an unsupported claim, whatever else it gains, so KleidiAI does
not pass. The rule is deliberately strict and one-sided: applied the other way
round, it would also reject CPU, which misses three facts KleidiAI supplied. Each
question was answered once with seeded sampling at temperature 0.6. Small
numerical differences between backends can change which tokens are sampled, so
these differences do not show that either backend is more accurate.
[Quality results](docs/results/controlled-cpu/quality.md).

The pinned KleidiAI integration replaces Q8_0's one scale per 32 weights with
one scale per weight row, and it quantizes activations per row too. KleidiAI
therefore computes with coarser numbers than the standard CPU backend, even from
the same model file. These records do not show whether this caused any answer
difference. The questions are known and were reviewed by project agents, so the
scores do not establish general accuracy or human validation.
[Interpretation and limits](docs/report.md).

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
- [Prompt and generation kernel timings](docs/results/kernel-analysis/README.md)
- [Project report](docs/report.md)
- [CPU benchmark methods](docs/benchmarks.md)
- [Evaluation cases and scoring](eval/README.md)
- [Original reference brief](docs/project-brief.md)

## Design

A small Python application handles ingestion, retrieval, citations and evaluation. Pinned llama.cpp binaries handle local inference. CPU comparisons use the same model and settings, with evidence that the intended kernels actually ran. Metal is an optional, unmeasured extension.

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
