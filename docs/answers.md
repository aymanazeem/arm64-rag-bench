# Answer questions locally

The `ask` command retrieves document passages and answers with source references. It uses a local llama.cpp server and pinned Qwen3.5-9B Q8_0 with bounded reasoning. No hosted model account is needed. The [initial quality checks](results/grounded-answers/README.md) found supported claims but some incomplete answers.

## Setup

Follow the [runtime guide](runtime.md) to install the tools and build the CPU server. Download the answer model, then index your text or Markdown files:

```sh
uv run --locked --group build python scripts/download_answer_model.py
uv run --locked --group build arm64-rag index eval/fixtures
uv run --locked --group build arm64-rag ask 'What batch size does the instruction-handling fixture use?'
```

The model download is about 9.53 GB. This is [Unsloth's third-party GGUF quantization](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/tree/3885219b6810b007914f3a7950a8d1b469d598a5) of [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a), under Apache-2.0. The publisher does not state the exact conversion input revision or recipe. [answer-model.json](../configs/answer-model.json) pins the model bytes, embedded template, license and source notices. Downloads verify TLS, size and SHA-256. If Python cannot verify the certificate chain, the downloader tries system curl with TLS verification still enabled.

`ask` starts its own server on an authenticated loopback port and closes it afterward. These checks use an M1 Max with 32 GB of unified memory. The CPU preset uses four threads, an 8,192-token context and at most five passages. It reserves 2,048 generated tokens for reasoning and the final answer together, with a 1,024-token reasoning budget. Only the final answer is displayed. Startup has a two-minute deadline; each request has a five-minute deadline. Model loading adds time to each command; this is not a persistent chat service.

Sampling uses temperature 0.6, top-p 0.95, top-k 20, min-p 0 and no presence penalty, matching [Qwen's precise-coding preset](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/README.md). Seed 42 is fixed locally. Our sampler applies temperature last, unlike [vLLM's order](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/sample/sampler.py). This is a local preset with shorter context and reasoning limits. There is no controlled latency result yet.

## Sources and output

```sh
uv run --locked --group build arm64-rag ask 'How many CPU threads did demo-17 use?' --source eval/fixtures/run-note-a.md
uv run --locked --group build arm64-rag ask 'How many CPU threads did demo-17 use?' --json
```

Use `--db PATH` for another collection. Repeated `--source ID` options restrict the source set. Optional `--method dense` and `--method hybrid` use the [embedding index](embeddings.md); add `--extra retrieval` to the `uv run` options for either method. Keyword search remains the default. The answer-quality checks use keyword search only.

Plain output shows each claim followed by source labels and line references. JSON also includes source revisions, offsets, URLs and the original cited passages. The model supplies labels; the application copies passage text and locations. Unknown labels, duplicate labels within a claim, malformed JSON and incomplete generation are rejected. A valid citation location does not prove that the passage supports the claim.

The application counts the complete rendered prompt with the model's own tokenizer and checks it against the chat endpoint's input count and returned prompt. It reserves output space, skips passages that do not fit and never silently truncates them. After selection, spare space can add the parent introduction immediately before a passage. It does not add neighbouring text from the same section. Selected passages are never removed to make room. Expanded passages keep exact source spans and separate references to their original chunks.

The design draws on context expansion used by [Haystack](https://docs.haystack.deepset.ai/docs/sentencewindowretriever) and [LlamaIndex](https://developers.llamaindex.ai/python/examples/node_postprocessor/metadatareplacementdemo/). Broader windows introduced distracting option tables during development, so this rule is narrower. Expansion applies only to answers; search rankings and stored chunks remain unchanged. It cannot recover distant conditions or every fact split across chunks.

Questions are limited to 4,096 characters. Empty retrieval returns a fixed statement that the sources do not establish an answer. Other missing evidence, disagreements and instruction-like document text are handled by the answer prompt and checked separately during review.

Answer runs save traces under `artifacts/answers/`. Early setup errors may exit before a trace is created; later failures retain the available evidence. Use `--trace PATH` to choose a new location. Existing traces are preserved. Traces contain questions, document passages and model reasoning, so keep private collections and their traces outside Git. Invalid model output stays in the trace and is not printed as an answer. The reasoning cap can force that stage to end; the final answer must still finish normally and pass validation.

## Check answer quality

Build the pinned corpus index as described in the [search guide](search.md), then run:

```sh
uv run --locked --group build python scripts/check_answers.py --split development --output artifacts/answers/development-01
uv run --locked --group build python scripts/check_answers.py --split regression --output artifacts/answers/regression-01
uv run --locked --group build python scripts/check_answers.py --split validation --output artifacts/answers/validation-01
```

Each run requires a new output directory. It saves the prompt and protocol, model/runtime hashes, attempted cases, individual outputs, failures and completion state. Interrupted attempts remain visible. These runs use the controlled CPU build by default; they are not speed comparisons.

Add `--backend kleidiai` to check the second build. The optional `--baseline PATH` flag expects the original local CPU run whose hashes appear in the published baseline. It requires one split and checks unchanged sources, prompt text, token IDs and settings. Those original raw records stay local. Changed answers still need factual and citation review; verification time is not latency evidence. All 40 questions now serve as fixed backend regressions. The `ask` command continues to use CPU. See the separate [benchmark method](benchmarks.md) for performance checks.

[answers-protocol.json](../eval/answers-protocol.json) defines three question roles:

| Split | Questions | Use |
| --- | ---: | --- |
| `development` | 14 | Model and prompt development. |
| `regression` | 18 | Earlier retrieval-validation questions used to diagnose answer failures; now development regressions. |
| `validation` | 8 | Fresh answer questions, frozen before final runs; they share the pinned source documents. |

Use `--split all` to run all three in order. Keep validation separate from tuning. Facts, citation support, abstention, conflicts and document instructions require independent source review. The questions and reviews are agent-authored, not human validation. Structural checks alone are not an accuracy score. In the frozen fresh check, all 15 claims had supporting citations and six of eight answers were complete. The [results](results/grounded-answers/README.md) retain missing facts, retrieval gaps and earlier failed candidates.

The answer model differs from the Q4_0 runtime probe. Its separate [KleidiAI checks](results/controlled-cpu/README.md) confirmed the intended kernel dispatch but failed the answer-quality comparison. CPU remains the application backend. A [guarded desktop study](results/desktop-cpu/README.md) now reports native computation times. Full answer latency and memory measurements remain separate.
