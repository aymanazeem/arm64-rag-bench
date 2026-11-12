# Arm64 RAG Benchmark

A local document question-answering application and an engineering investigation into CPU inference on Apple Silicon: standard Arm64, KleidiAI-enabled Arm64, and a separate Metal comparison.

The project starts on an **Apple M1 Max with 32 GB unified memory**. The aim is useful answers with citations and credible measurements of speed, memory and answer quality. A zero or negative speedup is a valid finding.

**Status:** repository foundation and researched implementation plan, delivered through [setup PR #1](https://github.com/aymanazeem/arm64-rag-bench/pull/1). The application, inference builds and benchmarks are not implemented yet. The [GitHub repository](https://github.com/aymanazeem/arm64-rag-bench) is private. Next: the reproducible runtime and KleidiAI feasibility probe.

- [Implementation plan and acceptance gates](PLANS.md)
- [Development and GitHub setup](docs/setup.md)
- [Working agreements](AGENTS.md)
- [Original reference brief](docs/project-brief.md)

## Planned approach

A small Python application handles ingestion, retrieval, citations and evaluation. Pinned llama.cpp binaries handle local inference. CPU comparisons use the same model and settings, with evidence that the intended kernels actually ran. Metal is reported as a separate backend.

Start with a command-line workflow and a small licensed technical corpus. Add complexity only when measurements or answer quality justify it. Model weights and private documents stay outside version control.

## Working here

After cloning, enable the repository hooks:

```sh
git config --local core.hooksPath .githooks
git config --local user.name 'Azeem Ayman'
git config --local user.email '110621518+aymanazeem@users.noreply.github.com'
git config --local user.useConfigOnly true
```

Work on feature branches and merge reviewed PRs. The setup guide explains the one-time empty repository bootstrap. Runtime installation commands will be added with the first tested implementation; there are no application commands to run yet.
