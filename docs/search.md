# Index and search documents

The `search` command finds local text and Markdown. Default keyword search runs without a model; optional dense and hybrid search use BGE embeddings. Results contain exact passages, source IDs, revisions and line numbers. Use [ask](answers.md) to generate answers from those passages.

Install the locked environment, then index a folder or selected files:

```sh
uv sync --locked --group build
uv run --locked --group build arm64-rag index eval/fixtures
uv run --locked --group build arm64-rag search 'demo-17 CPU threads'
```

Commands return JSON. The default database is `data/private/index.sqlite`, which is ignored by Git. Use `--db PATH` to keep separate collections. SQLite files and their sidecars are also ignored. Search never creates a missing database.

Local source IDs are paths relative to `--root`, which defaults to the current directory. Use the same root on later imports so IDs stay stable. Symlinked files keep their alias paths as IDs; their targets must stay inside the root. Only `.txt`, `.md` and `.markdown` files are supported; directory scans select those extensions. Files must use UTF-8.

## Updates and removal

An ordinary import adds or replaces the named sources. It retains all other sources. An unchanged import preserves the index generation and chunk IDs.

`--sync` treats the input as the complete collection and deletes every omitted source from that database:

```sh
uv run --locked --group build arm64-rag index eval/fixtures --sync
```

Read and parse errors abort the import. Source records, chunks and search entries change in one transaction. A failed transaction leaves the previous index usable. Search reads one committed snapshot. Incompatible index formats require a new database; they are not silently rebuilt.

## Search behavior

```sh
uv run --locked --group build arm64-rag search 'GGML_CPU_KLEIDIAI' --limit 5
uv run --locked --group build arm64-rag search 'threads' --source eval/fixtures/run-note-a.md
uv run --locked --group build arm64-rag search -- '--n-gpu-layers'
```

Use `--` before a query that starts with a dash.

Search uses SQLite FTS5 with BM25 ranking. Lower scores rank higher; scores are not confidence values. Ties use stable source locations. Query terms are quoted before matching, so typed operators cannot alter the query language. Underscores stay inside identifiers; punctuation-connected terms such as flags are matched as phrases. A blank or punctuation-only query returns no passages.

Markdown parsing preserves headings, paragraphs, code and tables when they fit. Large blocks split at line or lexical boundaries. The initial limit is 256 lexical units: Unicode words or individual punctuation marks. It is a retrieval baseline, not an embedding or generation token limit. Heading metadata is separate from the exact passage text. No text is rendered as HTML or executed.

Canonical text changes CRLF and CR newlines to LF and preserves everything else. Offsets count Unicode characters, with an included start and excluded end. Line numbers start at one. Source hashes, parser settings and chunk boundaries determine versioned chunk IDs.

The implementation follows the [FTS5 query and ranking contract](https://sqlite.org/fts5.html) and [Markdown parser source maps](https://markdown-it-py.readthedocs.io/en/latest/using.html).

## Run the development evaluation

The pinned corpus uses source checkouts from the [runtime setup](runtime.md). Indexing verifies every source and license hash and stores license text with the source. It makes no network requests.

```sh
uv run --locked --group build arm64-rag index --manifest data/manifests/corpus.json --db data/private/development.sqlite --sync
uv run --locked --group build arm64-rag evaluate --db data/private/development.sqlite --limit 5
```

The runner verifies the dataset, source versions and evidence spans before scoring. Use a dedicated database containing exactly the manifest's sources: BM25 statistics depend on the whole collection. Each question searches only its declared corpus. A required evidence unit counts as found only when retrieved passages cover its complete source span; adjacent passages can cover it together.

Reports keep per-case results, missing evidence, failures and aggregate denominators. Cases without labelled evidence have no retrieval correctness score. Returned text on those cases does not prove that an answer is supported. These development cases are agent-authored and have not been validated by humans.

Operational failures produce a nonzero exit status. Retrieval misses remain in the report and are not command errors. `first_evidence_rank` is the rank at which the first complete evidence unit is available; `all_evidence_rank` is set only when all required units are covered.

This measures retrieval only. It does not measure answer accuracy, citations, abstention or inference speed. The [embedding comparison](embeddings.md) preserves this baseline and uses separately frozen evaluation cases.

See the [first keyword baseline](results/lexical-baseline/README.md) for measured coverage and known misses.

[Embedding retrieval](embeddings.md) adds optional dense and hybrid methods. Keyword search and the original evaluation defaults remain unchanged.
