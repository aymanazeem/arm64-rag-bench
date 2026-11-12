# Embedding retrieval

Keyword search remains the default. Optional dense search finds passages using a small local model. Hybrid search combines keyword and dense rankings. The [first comparison](results/embedding-comparison/README.md) records gains and losses.

## Install and index

Use the pinned source checkouts from the [runtime setup](runtime.md), then run:

```sh
uv sync --locked --group build --extra retrieval
uv run --locked --group build --extra retrieval python scripts/download_embeddings.py
uv run --locked --group build --extra retrieval arm64-rag index --manifest data/manifests/corpus.json --db data/private/development.sqlite --sync
uv run --locked --group build --extra retrieval arm64-rag embed --db data/private/development.sqlite
```

The download verifies fixed model, tokenizer and configuration hashes. The pinned model card declares MIT licensing and stays beside the model. Upstream supplies no separate license file. Model files remain ignored by Git.

Search and indexing use local files after setup. They make no model downloads. PyTorch and Transformers are used only by the separate reference check.

```sh
uv run --locked --group build --extra retrieval arm64-rag search 'Which build option enables KleidiAI?' --db data/private/development.sqlite --method hybrid
```

Choose `--method lexical`, `dense` or `hybrid`. Dense and hybrid return at most 20 passages. Lower BM25 scores rank first; higher cosine and fusion scores rank first. None is a confidence probability.

After source changes, run `index` and then `embed`. Changed sources invalidate saved vectors in the same transaction as the document update. Keyword search remains available. Unchanged imports preserve vectors. Embeddings publish only if the source snapshot still matches; a failed import preserves the previous usable state. Changed embedding settings require re-embedding.

## Model contract

The fixed encoder is BGE-small-en-v1.5 through its official FP32 ONNX export. It uses CPU execution, 384 dimensions, the first token's representation and L2 normalization. Queries receive the model's retrieval instruction; documents receive no instruction. The text contains the heading path followed by the exact chunk body.

Every input is counted before inference. Large chunks use non-overlapping windows of 510 content tokens plus two special tokens. Search scores a chunk by its highest window similarity. This can favor longer chunks; the current corpus needs only one window per chunk. Oversized queries fail clearly. Text is never silently truncated.

SQLite stores all vectors and their configuration with the source snapshot. Search loads a contiguous matrix, calculates exact similarities and uses stable source locations to break ties. Hybrid search uses reciprocal rank fusion with constant 60 and 20 candidates from each method. There is no vector service or approximate index.

These choices follow the [official model contract](https://huggingface.co/BAAI/bge-small-en-v1.5), [ONNX Runtime API](https://onnxruntime.ai/docs/api/python/api_summary.html) and [rank-fusion method](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf).

## Verify and compare

The optional reference check downloads an additional pinned safetensors file and uses an isolated environment with pinned direct dependencies:

```sh
uv run --locked --group build --extra retrieval python scripts/download_embeddings.py --reference
uv run --script scripts/check_embeddings.py
```

It compares token IDs and vectors with Transformers, including Unicode, identifiers, batching and input boundaries. It records the actual reference package versions. This checks compatibility, not retrieval quality.

Run the fixed comparison after embedding the complete corpus:

```sh
uv run --locked --group build --extra retrieval python scripts/compare_retrieval.py
```

The [protocol](../configs/retrieval-comparison.json) pins both question sets. Each method receives the same chunks, at most five selected passages and a 1,024-token text budget counted by BGE. Twenty candidates are considered; passages that do not fit are skipped without truncation. The budget covers headings and bodies, not a future generator's chat template.

The report separates candidates from selected evidence, lists omissions and records per-case wins and losses. Source, vector and dataset identities must stay fixed across the complete run. A new run clears its previous output; the completed report is published atomically. Operational failures stay in the denominator and return a nonzero exit status.

The 18 validation questions were source-reviewed and frozen before this comparison. They share documents with development questions and have not been validated by humans. They are not a general reliability benchmark.

Run offline tests with installed assets:

```sh
ARM64_VERIFY_UPSTREAM=1 ARM64_VERIFY_EMBEDDINGS=1 uv run --locked --offline --group build --extra retrieval pytest
```

CI runs the offline synthetic checks and skips checks requiring downloaded source or model files.
