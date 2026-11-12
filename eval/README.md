# Evaluation contract

This is a small development set, not a held-out benchmark or a quality result. Cases are agent-authored and independently checked by another agent against the named sources. They have not been validated by humans.

`development.json` defines questions, evidence and scoring requirements. `data/manifests/corpus.json` pins four upstream documents and three synthetic fixtures. Each case names its complete corpus scope; absence and conflict labels apply only to that scope. Upstream text stays outside Git. The manifest records immutable URLs, byte hashes and license references. Preserve upstream license notices when materializing the corpus.

## Source and span format

- Decode UTF-8 strictly. Normalize CRLF and CR to LF; make no other text changes. This format is `utf8-lf-v1`.
- Keep the raw file hash and canonical text hash. A content change creates a new source revision; it must not silently reuse old labels.
- Evidence uses a source ID and zero-based Unicode character offsets: `start` is included and `end` is excluded. It also records one-based inclusive line numbers and a hash of the exact canonical span.
- Labels refer to document spans, never retrieval chunks. A new chunking strategy must still locate the same evidence.
- `source_ids` is the full document set for a corpus scope. A partial source update must not delete unmentioned documents; only an explicit full-manifest sync may do that.
- The dataset pins the manifest's raw SHA-256. Update the dataset version and review affected cases when source bytes, normalization or labels change.

## Case format and scoring

Each case has an ID, question family, tags, corpus ID, question, answerability, evidence spans, required facts and unacceptable claims. `answerability` is `answerable`, `unanswerable` or `conflicting`. The `origin` field separates source-derived questions from synthetic ones.

`required_facts` and `unacceptable_claims` are review criteria, not substring-match rules. Accept accurate paraphrases. For each run, retain the rendered prompt, retrieved and packed spans, answer, source/configuration hashes, timings, failures and scoring reasons.

| Check | What counts |
| --- | --- |
| Retrieval | Required evidence units found; report each unit on multi-source cases |
| Context | Required evidence retained after token-budget packing |
| Facts | Required facts satisfied without unsupported or contradicted claims |
| Citations | IDs resolve, cited spans support claims, and material claims have support |
| Abstention | Unsupported questions are declined; answerable questions remain useful |
| Conflicts | Both conflicting records are cited without inventing precedence |
| Reliability | Timeouts, malformed output and failures stay in the denominator |

Passing a citation-ID check does not establish factual support. A model grader can assist review; it must not be the sole authority or be described as a human reviewer. Report counts and per-case failures, not one combined RAG score.

## Splits and regression rules

These cases are available for tuning and regression checks. Before tuning retrieval, prompts or models for a comparison, create and freeze a separate evaluation set, corpus, labels and scoring policy. Keep related question families together when splitting. Record author and reviewer provenance.

If a final case later guides a fix, record that use and version or replace it for future claims. Fix incorrect labels transparently and rerun baseline and candidate. Do not edit labels only to make a candidate pass.

After each major feature, run relevant tests and existing regressions. Use a real native/end-to-end check when that layer changes. Resolve important new factual or citation failures before merging. A small passing set supports only “no regression observed on these cases.”

## Validate the data

The default check needs only Python and reads no downloaded source:

```sh
python3 -m unittest discover -s tests -p test_eval_contract.py
```

When the pinned checkouts exist under `vendor/`, also verify upstream bytes and every source-backed evidence span:

```sh
ARM64_VERIFY_UPSTREAM=1 python3 -m unittest discover -s tests -p test_eval_contract.py
```

The optional check makes no network requests and fails if a required source is missing or changed. These checks validate data integrity, not model answer quality.
