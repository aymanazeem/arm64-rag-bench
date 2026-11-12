# Arm64 RAG Inference Benchmark — Project Brief

## Idea

Build a **local retrieval-augmented generation (RAG) application** on an **Apple M1 MacBook Pro**, then investigate how efficiently it runs on Arm64 hardware. The main technical focus is **KleidiAI**, Arm's optimised CPU microkernel library, and its integration with **llama.cpp**.

This should be an engineering investigation, not just a chatbot demo: build something useful, establish credible baselines, measure performance, test answer quality, and explain the results.

## Key questions

- Can a practical, fully local RAG system run reliably on an M1 MacBook Pro?
- What, if any, performance difference does **KleidiAI-enabled CPU inference** make against a comparable standard Arm64 CPU configuration?
- How does **Apple Metal GPU inference** compare, where useful? Treat Metal as a separate execution path, **not** as KleidiAI acceleration.
- What are the trade-offs among latency, throughput, memory use, answer quality and implementation complexity?
- Can the results be reproduced and the **actual kernel/backend execution** verified?

## Desired outcome

A public, reproducible software project with a functioning local document-Q&A system; a meaningful benchmark and evaluation suite; measured comparisons; and a concise technical report explaining findings and limitations. Prefer clear experimental evidence over impressive-looking but unsupported speedup claims. Negative results are valid.

The implementation should make its **own informed decisions** about model size, quantisation, RAG architecture, datasets, evaluation methods, testing, code organisation and presentation. Start by checking current documentation and hardware feasibility, especially the machine's available unified memory (commonly 8 or 16 GB on M1 MacBook Pro models). If M1 limitations prevent a meaningful KleidiAI comparison, document why and propose a defensible alternative or supplemental Arm64 test environment.

**Important:** Enabling KleidiAI at build time does **not** establish that a given operation used its kernels. Verify runtime dispatch where possible; distinguish CPU microkernel effects from GPU effects and from end-to-end RAG latency. Control for changes in models, settings, caching and measurement conditions.

## Starting resources

1. **[KleidiAI repository](https://gitlab.arm.com/kleidi/kleidiai)** — primary source, microkernels and supported targets. [GitHub mirror](https://github.com/ARM-software/kleidiai).
2. **[llama.cpp repository](https://github.com/ggml-org/llama.cpp)** — local inference runtime.
3. **[llama.cpp build documentation — Arm KleidiAI](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#arm-kleidiai)** — Apple arm64 support, configuration, limitations and dispatch verification.
4. **[llama-bench documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md)** — reference inference benchmarking tool and metrics.
5. **[Arm Learning Path: RAG with llama.cpp and KleidiAI](https://learn.arm.com/learning-paths/servers-and-cloud-computing/rag/)** — useful end-to-end reference; designed for Arm cloud/server hardware, so adapt rather than copying its setup to an M1 laptop.
6. **[Apple Metal documentation](https://developer.apple.com/metal/)** — Apple GPU compute, relevant if testing GPU inference.
7. **[Apple M1 MacBook Pro specifications](https://support.apple.com/en-gb/111893)** — hardware reference; verify exact configuration locally.

Research any additional documentation needed and make the technical design decisions independently. The goal is a well-engineered, carefully evaluated project with defensible conclusions—not adherence to a predetermined implementation plan.
