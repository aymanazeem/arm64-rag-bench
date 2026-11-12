# Embedding retrieval comparison

Measured on 2026-11-12 on the M1 Max. Keyword search remains the default. Hybrid search improves aggregate coverage, but loses one validation case that keyword search covers. Dense search alone performs worse on both sets.

Each method uses the same 383 chunks, up to five selected passages and a 1,024-token BGE text budget. Questions and settings were frozen before the comparison. No settings were tuned to these results.

| Set | Keyword | Dense | Hybrid |
| --- | --- | --- | --- |
| Development: all evidence covered | 8/13 | 6/13 | 9/13 |
| Validation: all evidence covered | 14/18 | 13/18 | 16/18 |
| Development: evidence units covered | 11/16 | 9/16 | 12/16 |
| Validation: evidence units covered | 17/22 | 17/22 | 20/22 |

All six runs completed without execution failures. The development set also has one question without labelled evidence; it receives no retrieval correctness score.

Against keyword search, hybrid gains one development case and loses none. On validation it gains `val-008`, `val-016` and `val-017`, but loses `val-006`, about starting a benchmark with an existing KV cache. The [per-case record](retrieval.json) preserves all wins, losses and selected source spans.

The earlier [keyword baseline](../lexical-baseline/README.md) had no text budget and covered 10/13 development questions. Its original results, scores and spans still match exactly. This comparison adds a shared budget; it does not replace that baseline.

## Compatibility and verification

- The [reference check](compatibility.json) passed 18 vector comparisons against Transformers. The largest absolute difference was `2.91e-7`.
- All 383 chunk token sequences matched the reference tokenizer. The longest input used 444 tokens, so every chunk fits in one model window.
- [Index creation](index-build.json) stored 383 vectors atomically. Its observed wall time includes process startup and model loading; it is not a controlled benchmark.
- [CLI checks](cli-smoke.json) passed for all three methods and validation evaluation.
- All 216 local tests and 42 subtests passed, including the downloaded-source and model checks. Independent reviews covered storage, model compatibility, labels, evaluation and report publication.

See the [reproduction guide](../../embeddings.md). The reports contain asset, source, vector and implementation identities. Full candidate passages remain locally in `artifacts/retrieval/embedding-comparison.json`; the published record includes its hash. The setup, comparison and reference checks make no model downloads after their assets and dependencies are installed.

## Limits

These are agent-authored questions with source review, not human validation. The validation questions share documents with development. The sample is small and supports no broad reliability or significance claim.

Coverage requires complete labelled spans after passage selection. The BGE text budget excludes a future generator's instructions and template. Per-case times include search, token counting and scoring; they are not controlled latency measurements.

No generated answers, citation support or abstention were evaluated here. Hybrid retrieval remains an optional candidate for the next Q&A checks. Its validation regression must stay visible when choosing the application's retrieval method.
