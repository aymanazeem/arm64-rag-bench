# CPU and KleidiAI answer quality

Both backends completed all 40 known questions, with 30 complete answers each,
28 of them on the same questions. KleidiAI gained three facts, lost two and added one
claim without full citation support, so it does not pass the project's strict
no-regression rule. These results do not show that either backend is more
accurate. CPU is the application backend. The
[comparison record](quality-comparison.json) retains every case.

| Measure | CPU | KleidiAI |
| --- | ---: | ---: |
| Outputs with valid format | 40/40 | 40/40 |
| Claims supported by supplied text and attached citations | 78/78 | 80/81 |
| Required facts passed | 66/83 | 67/83 |
| Required facts unassessed | 0 | 0 |
| Cases passing every applicable criterion | 30/40 | 30/40 |

CPU has 17 failed fact units; KleidiAI has 16. These counts include missing
facts even when an answer correctly declines to fill gaps in its supplied text.
A valid answer format alone does not establish correctness. Equal complete-case
counts do not mean that the same answers passed.

The KleidiAI [development](quality-development.json),
[regression](quality-regression.json) and
[validation](quality-validation.json) records retain each claim, citation
location, verdict and evidence hash. The [CPU results](../grounded-answers/README.md)
retain their own omissions.

## Differences between backends

KleidiAI includes three facts missing from CPU: caller-controlled thread dispatch
(`dev-010`), Apple's unknown SVE vector length (`val-004`), and field renaming
when moving a nested response field to the root (`fresh-005`).

It omits two details supplied by CPU: the warning that generated archive hashes
can change (`val-009`), and the lack of automatic SME ABI support detection
(`val-011`). The latter still gives the correct build flag. Its review records
the possible implication of manual configuration while retaining the missing
detection caveat.

`dev-010` also adds a scheduling contrast to a claim about runtime kernel
selection. Its attached passage supports kernel selection but does not establish
the full scheduling claim. The review retains a plausible broader reading.

## Reading the result

The project's rule is that a backend change must preserve answer quality: the
three fact gains do not cancel this citation failure or the two omissions. The
rule is deliberately strict and one-sided. Applied the other way round, it would
also reject CPU, which misses three facts KleidiAI supplied.

Each question was answered once with seeded sampling at temperature 0.6.
Different arithmetic can change which tokens are sampled, and KleidiAI computes
with coarser weights and activations than CPU. Single answers therefore cannot separate a backend
effect from sampling variation. These results do not establish that either
backend is generally more or less accurate.

## Collection and review

Both backends used one attempt per question and completed all 80 cases without
execution or format failures. Model bytes, prompts, settings, sources and rubric
stayed fixed. All 40 input pairs matched. Collection guards passed, with no
observed swap; system and editor activity remained. This was not CPU isolation.
The [benchmark method](../../benchmarks.md) explains the KleidiAI weight
conversion and why the two backends produce different numerical results.

Reviews reuse authenticated source verdicts for 40 CPU and 38 KleidiAI answers
whose inputs, final text and validated answers match their same-backend review
records exactly. Two KleidiAI answers received new source review. Each reused
verdict comes from a review record for the same backend, not from this
collection's CPU answers, and none is a human review.

Questions and reviews are by project-contributing agents, with material
ambiguities separately checked. They are not human or external validation.
All 40 questions are known, so this is a fixed backend regression check with no
claim of general accuracy or unseen-question performance. Quality collection
durations are excluded from performance comparisons.
