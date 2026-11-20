# Local document Q&A and Arm64 inference

The project works as a local document-answering CLI and a focused backend study on an Apple M1 Max with 32 GB memory. KleidiAI processes the tested prompts faster and generates tokens slower. Both backends answered all 40 fixed questions and each produced 30 complete answers, 28 of them on the same questions. **CPU is the application default.** KleidiAI does not pass the project's strict no-regression quality rule, and the whole application took 46% longer with it: generation takes most of the answer time, and every answer starts a server, which takes twice as long on KleidiAI.

## Design and use

Python handles parsing, retrieval, prompt preparation and validation. SQLite stores canonical text, exact source locations, keyword search and optional embeddings. Transactional imports preserve the previous usable index after failure. Pinned llama.cpp binaries run in a separate authenticated local server, started and closed for each answer. [Search](search.md), [embeddings](embeddings.md), [answer setup](answers.md).

```mermaid
flowchart LR
    D[Documents] --> S[(SQLite)]
    Q[Question] --> R[Retrieve and pack]
    S --> R
    R --> L[Local llama.cpp]
    L --> V[Validate answer and citations]
```

After installing the runtime and model:

```sh
uv run --locked --group build arm64-rag index eval/fixtures
uv run --locked --group build arm64-rag ask 'What batch size does the instruction-handling fixture use?'
```

The preset uses Qwen3.5-9B Q8_0, four threads, an 8,192-token context and at most five passages. Its 2,048-token output allowance includes up to 1,024 reasoning tokens; only the final answer appears. Model bytes and template are pinned, although the third-party publisher does not disclose the exact quantization recipe.

Packing counts the complete template and reserves output space. Spare capacity may add the immediately preceding parent introduction. The model returns claims and source labels; the application attaches canonical passages. Invalid labels, malformed output and unfinished generation are rejected. Valid reference locations alone do not prove factual support. [Answer contract](answers.md#sources-and-output).

## Retrieval decision

The corpus contains seven pinned documents and 383 chunks. With five passages and a shared 1,024-token text budget, complete evidence coverage was:

| Questions | Keyword | Dense | Hybrid |
| --- | ---: | ---: | ---: |
| Development | 8/13 | 6/13 | 9/13 |
| Retrieval validation | 14/18 | 13/18 | 16/18 |

Hybrid improved aggregate coverage but lost `val-006`, which keyword search covered, so **keyword retrieval is the default** under the no-regression rule. All six runs completed. These are retrieval scores, not answer-quality or controlled latency measurements. An unbudgeted keyword check reached 10/13 under a different protocol. The 18 retrieval-validation questions also form the answer `regression` split; the answer `validation` split is a separate set of eight questions. [Detailed comparison](results/embedding-comparison/README.md), [keyword baseline](results/lexical-baseline/README.md).

## Answer quality and backend selection

“Complete” means passing every applicable review criterion. Supported claims must agree with both the supplied sources and their attached citations.

| Backend | Valid / attempted | Supported claims | Passed facts | Complete cases |
| --- | ---: | ---: | ---: | ---: |
| CPU | 40/40 | 78/78 | 66/83 | 30/40 |
| KleidiAI | 40/40 | 80/81 | 67/83 | 30/40 |

All 80 planned cases completed without execution or format failures, with no observed swap activity or detected timing interruption. Both backends used the same questions, retrieved passages, prompts, settings and 300-second request deadline. Inputs matched in all 40 pairs. Quality collection times are excluded from performance comparisons.

CPU passed all applicable criteria in 11/14 development, 13/18 regression and 6/8 validation cases. Its 17 missing fact units include evidence lost during passage selection and omitted details. Supported statements alone do not make an answer complete. [CPU results](results/grounded-answers/README.md).

Each backend had 30 complete cases, 28 of them the same. KleidiAI supplied three facts CPU missed (`dev-010`, `val-004`, `fresh-005`). It missed two facts CPU supplied: a warning about changing archive hashes (`val-009`) and the lack of automatic SME ABI detection (`val-011`). One KleidiAI scheduling clause in `dev-010` was unsupported by its supplied text and attached citation.

The project's frozen quality rule rejects a backend that loses any fact CPU supplied or adds an unsupported claim, whatever else it gains, so KleidiAI does not pass. The rule is deliberately strict and one-sided: applied the other way round, it would also reject CPU. Each question was answered once with seeded sampling at temperature 0.6, and different arithmetic can change which tokens are sampled. Single answers therefore cannot separate a backend effect from sampling variation, and these results do not show that either backend is more accurate. This finding is limited to these questions and this rubric. [Paired comparison](results/controlled-cpu/quality.md).

Reviews for 40 CPU and 38 KleidiAI answers were transferred from authenticated, identical outputs from the same backend. Two KleidiAI answers received separate source review. All 40 questions are known regression cases. Questions and reviews are agent-authored, with material ambiguities separately checked by a project contributor. They are not human or external validation and do not establish accuracy on unseen documents. The evidence records retain individual claims, citations, verdicts and review provenance.

## What the native evidence proves

Eighteen reviewed diagnostic probes passed. Eight instrumented backend pairs matched inputs. In every traced phase, KleidiAI handled all 249 Q8_0 matrix weights in the model, with positive-shape Neon/DotProd kernel entries. The model's only other Q8_0 tensor is the token-embedding table, which is read by lookup rather than matrix multiplication. Selected recurrent operations remained on CPU. These samples establish actual dispatch. They trace only the first batch of each phase, so they do not measure how much total work was accelerated. [Dispatch evidence](results/controlled-cpu/README.md).

At prefixes 512 and 2,048, each backend restored its own state and reproduced 248,320 finite continuation logits byte-for-byte. CPU and KleidiAI vectors differed. Normal runs rely on source/build/library bindings; diagnostic timings are excluded. [Reproduction and limits](results/controlled-cpu/diagnostics.md).

The pinned KleidiAI integration converts each Q8_0 weight row to 8-bit values with a single scale, set by the row's largest value. Q8_0 stores one scale per 32 weights, and the standard CPU backend keeps those scales. Blocks of smaller values therefore lose precision, so KleidiAI computes with a coarser version of the same model. It also quantizes activations with one scale per row, where the standard path uses one scale per 32 values. These differences could contribute to answer differences, but these records do not establish the cause of any omission or unsupported claim. The comparison covers complete backend integrations, not kernels alone. [Pinned conversion code](https://github.com/ggml-org/llama.cpp/blob/8ca5a8008b57aa39f2601546c0d07912045080a5/ggml/src/ggml-cpu/kleidiai/kleidiai.cpp#L1530-L1600).

The weight conversion runs once while loading, outside the native computation timing, so its one-time cost cannot explain the slower generation; the converted format is used inside the timed work. Start-up is about 16 seconds longer per server, mostly in this conversion; see the [application results](#application-results). [Kernel timings](results/kernel-analysis/README.md) trace the slower generation mainly to the single-token kernel itself.

## Desktop computation results

The study **completed 48/48 runs and 24/24 pairs in one uninterrupted
invocation**, with no observed swap activity or detected timing interruption within
frozen tolerances. Both builds used the same model, four threads and F16 caches;
Metal, BLAS, Accelerate and OpenMP were disabled.

Median CPU/KleidiAI duration ratios were **1.503 and 1.464** for 512- and
2,048-token prompts, and **0.699 and 0.704** for the two generation workloads.
KleidiAI used **33.5% and 31.7% less prompt time**, but **43.0% and 42.1% more
generation time**. CPU processed about 59–62 prompt tokens per second and
generated about 12; KleidiAI processed about 86–92 and generated about 8.4–8.6.
Each workload has six pairs; each process mean uses three designated
repetitions. All 192 raw samples remain recorded, with 48 initial samples
excluded and 144 used. [Results and method](results/desktop-cpu/README.md).

Prompt processing evaluates many token positions together; generation adds one token at a time, and each phase uses its own kernel in each backend. Separate [kernel timings](results/kernel-analysis/README.md) account for most of both differences. KleidiAI's prompt kernel kept the multiply units about 98% busy, against 60% for the CPU kernel, which restarts and rescales its totals every 32 weights. KleidiAI's single-token kernel on this M1 Max adds every product for a group of columns into one running total, so each multiply instruction waits for the previous one. At four threads it processed 73 billion weights per second, against 101 billion for the CPU kernel; in a diagnostic thread sweep, full-model generation tied at eight threads. All six pairs per workload favor the same backend. Including initial samples or using medians instead of means does not reverse that direction.

No competing Docker, VM or project job was observed. macOS, managed security and the editor remained active, and thermal data was unavailable.

Recorded load was high before and at the start of measurement: the 15-minute load average was 65.7 before setup and 22.7 when the first block began, and stayed below 8 in the second and third blocks. Block median ratios differed between blocks by at most 0.042. CPU generation after the 512-token prefix varied most, with block medians of 10.98, 10.42 and 10.25 seconds.

KleidiAI ran first in round 1 and CPU first in round 2 of each block. Pair ratios were similar in either order, but because order is tied to round, order and round effects cannot be separated. The three consecutive blocks are not independent replications. These limits affect precision and attribution; the results do not establish general Arm64 performance, population confidence or tail latency.

In two setup runs per backend, KleidiAI's peak memory was 17.3–17.4 GiB, against 15.2–16.7 GiB for CPU. The measured runs did not record memory, so this is not a controlled memory comparison.

Native timings exclude loading, retrieval, tokenization, sampling and prefix/state preparation. Prompt and generation percentages cannot be averaged into one speedup: the balance depends on input and output lengths, reasoning and startup.

## Application results

The unchanged `ask` command answered all 40 questions once on each backend, with only the server binary changed. Each question ran on both backends back to back, CPU first in 20 pairs and KleidiAI first in 20, after 120 idle seconds each. **All 88 observations succeeded** in one session with valid timing. Every question's inputs matched the published quality records, and all 80 answers matched the reviewed answers for their question and backend, text and token count. [Results](results/application-performance/README.md), [method](application-performance.md).

| Measure | CPU | KleidiAI |
| --- | ---: | ---: |
| Total answer time, 40 questions | 85.9 min | 125.1 min |
| Median answer time (range), s | 142.1 (71.7–175.3) | 196.3 (94.8–213.4) |
| Median server start-up, s | 14.6 | 30.5 |
| Prompt processing, tokens/s | 58.1 | 82.3 |
| Generation, ms/token | 83.7 | 119.7 |
| Generated tokens, total | 39,522 | 45,009 |

**KleidiAI took 46% longer in total** and was slower on 38 of 40 questions; the median per-question increase was 37%. Of its 2,352 extra seconds, generation added 2,076 and start-up 634, while faster prompts saved 363; retrieval and harness steps added 5. Generation took 64% of CPU's answer time and 72% of KleidiAI's.

The prompt and generation rates are close to the desktop study's. With the same inputs and seed, the backends' different arithmetic leads sampling to different tokens, so answer lengths differ by question. KleidiAI wrote 14% more tokens; CPU wrote at least twice as many on the only two questions where KleidiAI was faster. On the 22 questions whose answers were within 5% in length, KleidiAI took 33% longer and was slower on every one. Requests alone, without start-up, took 25% longer there.

Model loading and context setup took 25.1–25.7 seconds on KleidiAI and 9.6–10.0 on CPU in every start. In a sampled start, KleidiAI's single-threaded weight conversion took 2.7 times as many samples as CPU's, most often in the scalar loops that give each row one scale. A question with no matching passages, run four times per backend, took 15.1 seconds on CPU and 30.9 on KleidiAI, almost all of it start-up.

Median model-server peak memory was 16.81 GiB on CPU and 17.20 GiB on KleidiAI. The longest request took 182 seconds on KleidiAI, within the 300-second limit. Prompt and generation rates differed by at most about 1.5% depending on which backend ran first. All 176 host checks found AC power and normal memory pressure. Before each observation, the median one-minute load average was 5.5 on CPU and 5.2 on KleidiAI; other applications stayed open. No memory was swapped out, and no sleep or clock jump was detected. Thermal readings were unavailable.

The command prints its whole answer at once, so **time to the first token was not observed**. Answer times include small harness steps before the command starts. Fresh processes do not mean cold file caches. Each question was answered once per backend with one seed, and other seeds could give other answer lengths. The questions are known and the answers repeat reviewed records, so they add no new accuracy evidence. Load logs and samples from diagnostic starts of the measured binaries show each build's weight layout and kernels. [Runtime evidence](results/application-performance/runtime.txt).

## Reproduction and remaining work

With the optional upstream, embedding and real-model checks enabled, the code published with these results passed **958 tests and 42 subtests**, with Ruff checks and formatting for 86 files. The application comparison froze 35 source files, both builds, the model and the reviewed answer records; the published sources match those hashes. These checks used the existing environment and cached assets; they do not establish a fresh installation or clean native build.

Four setup-only observations through the unchanged public API completed with no observed swap. The exported 600-second candidate has status `pending_review`; a separate review approved that calibration for the desktop study. These setup observations are excluded from performance summaries and make no separate command-line execution claim. [Validation](results/public-setup/README.md).

Public guides support new data collection and a [new local CPU baseline](answers.md#compare-a-fresh-cpu-baseline). Published hashes cannot reconstruct undistributed raw traces. The desktop study used collector/reporter commit `15dc33dc736c5aaa92fa8afd3e8f61d07dd65241`. Answers with several different seeds per question on each backend would show whether answer differences exceed sampling variation. A long-running server, other thread counts and chips with I8MM or SME would extend the application comparison. Operator profiling, human answer review and an optional Metal comparison would extend these findings.
