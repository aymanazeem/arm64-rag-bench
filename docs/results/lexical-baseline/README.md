# Keyword retrieval baseline

Measured on 2026-11-12 using the seven pinned documents and 383 chunks. No retrieval settings were tuned against these cases.

| Check | Result |
| --- | --- |
| Questions run | 14 |
| Questions with required evidence | 13 |
| All required evidence in the first five passages | 10 of 13 |
| Required evidence units covered | 13 of 16 |
| Execution failures | 0 |
| Question without labelled evidence | 1; no correctness score |

The [per-case record](retrieval.json) includes source locations, scores, missing evidence, configuration, implementation hashes and dataset hashes. It omits passage text. The complete raw report remains locally at `artifacts/retrieval/lexical-baseline.json`; its hash is recorded.

Three questions missed required evidence: the native KleidiAI build option (`dev-001`), benchmark workload options (`dev-004`) and KleidiAI scheduling (`dev-010`). Each required span crosses two chunks. The first five results do not jointly cover those spans. These misses remain recorded for the later retrieval comparison.

An unchanged import preserved the index. Rebuilding from the same sources in reverse order produced identical results, scores, spans and index fingerprint. All 125 local tests passed, including source integrity checks; 42 subtests also passed. Independent review verified coverage against the stored source characters.

Follow the [index and evaluation commands](../../search.md#run-the-development-evaluation) to reproduce the run. Keep the limit at five. The report's configuration identifies the exact parser, chunker, tokenizer and query policy.

These are small, agent-authored development cases, not held-out or human-validated results. Whole-span coverage is deliberately strict. The run does not measure generated-answer accuracy, citation support, abstention, inference speed or general reliability. Per-case times include retrieval validation and scoring; they are not a controlled latency benchmark.
