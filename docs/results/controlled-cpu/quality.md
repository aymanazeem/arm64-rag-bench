# CPU and KleidiAI answer quality

The KleidiAI build failed the frozen CPU quality comparison on 2026-11-13.
The application continues to use CPU. All 40 attempted cases are retained in
the [comparison record](quality-comparison.json).

| Measure | CPU baseline | KleidiAI |
| --- | ---: | ---: |
| Outputs with valid format | 40/40 | 38/40 |
| Claims supported by supplied text and attached citations | 78/78 | 74/75 |
| Required facts passed | 66/83 | 63/83 |
| Required facts unassessed after execution failure | 0 | 4 |
| Cases passing every applicable criterion | 30/40 | 28/40 |

KleidiAI has 16 failed fact units and four unassessed units. Failed executions
remain in the case denominator; their unavailable answers are not graded as
factual errors. A valid answer format alone does not establish correctness.

The [development](quality-development.json),
[regression](quality-regression.json) and
[former validation](quality-validation.json) records retain each claim,
citation location, verdict and evidence hash. The
[accepted CPU baseline](../grounded-answers/README.md) has known omissions.

## What changed

Three facts improved: caller-controlled thread dispatch (`dev-010`), Apple's
unknown SVE vector length (`val-004`), and field renaming when moving a nested
response field to the root (`fresh-005`).

Two previously passed details were lost: the warning that generated archive
hashes can change (`val-009`), and the lack of automatic SME ABI support
detection (`val-011`). The latter still gives the correct build flag. Its
review records the possible implication of manual configuration while retaining
the missing detection caveat.

`dev-010` also adds a scheduling contrast to a claim about runtime kernel
selection. Its attached passage supports kernel selection but does not establish
the full scheduling claim. The review retains a plausible broader reading.
The recovered facts do not cancel this citation failure or the two omissions.

## Execution failures

`fresh-007` retained matching inputs but no completion. Its roughly 301-second
record and server cancellation are consistent with the 300-second request
deadline. The adapter then closed its server. `fresh-008` failed before retrieval
and prompt packing because the server was no longer running.

This is one request failure and one dependent preparation failure. The records
do not establish a backend-specific cause. Missing inputs for the final case
remain unavailable; they are not declared equal. Neither case was retried, and
the deadline was unchanged.

## Scope

Model bytes, prompts, settings, sources and rubric stayed fixed. The 39 cases
that reached input checks matched the CPU records. All frozen source and runtime
hashes remained unchanged during the KleidiAI run. The
[benchmark method](../../benchmarks.md) explains
why the two backends can still produce different numerical results.

Five exact final answers reused authenticated CPU verdicts. The other 33 valid
outputs received new source review. Reviews are by project-contributing agents,
with material ambiguities separately checked; they are not human or external
validation. These questions were already used for CPU evaluation, so this is a
fixed backend regression test, with one attempt per case and no claim of general
accuracy. Guarded quality-run timings are excluded from performance comparisons.
