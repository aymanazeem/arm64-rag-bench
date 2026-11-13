# Grounded answer checks

Checked on 2026-11-13 on an M1 Max with 32 GB memory. The answer command uses keyword retrieval and pinned Qwen3.5-9B Q8_0 on the CPU. These small, source-reviewed checks establish neither general accuracy nor inference speed.

## Results

| Set | Executed | Supported claims | Required facts | Complete cases |
| --- | ---: | ---: | ---: | ---: |
| [Development](development.json) | 14/14 | 28/28 | 23/27 | 11/14 |
| [Regression](regression.json) | 18/18 | 35/35 | 30/39 | 13/18 |
| [Fresh validation](validation.json) | 8/8 | 15/15 | 13/17 | 6/8 |

All 40 cases completed without execution or format failures. Supported claims require both supplied-source support and support from the attached citations. Complete cases pass every applicable review criterion. Execution and valid citation locations alone do not establish answer quality.

Development cases for missing evidence, conflicting records and embedded document instructions pass. Two regression questions receive faithful refusals because their required passages fall outside the five-passage limit.

Fresh validation has two incomplete cases. `fresh-001` declines a combined Mac/Linux question because the Mac passage was excluded at rank 6. `fresh-005` gives supported field-selection behavior but omits the required renaming detail. These failures remain in the counts; no settings changed in response.

## Candidate selection

Across the 32 known questions, the earlier 4B thinking candidate had 54/56 supported claims, 50/66 required facts and 18/32 complete cases. The selected 9B candidate has 63/63 supported claims, 53/66 facts and 24/32 complete cases. It fixes both unsupported claims and restores six fact units, but loses three details:

- `dev-010`: the caller can dispatch work across threads.
- `val-004`: the selector treats Apple's vector length as unknown.
- `val-013`: a missing router model parameter returns HTTP 400.

These remain failed facts. The central answers remain supported, but completeness is not unchanged. Retrieval and supplied passages match across all 32 pairs.

All earlier candidates, interruptions and failed attempts remain in [the pilot record](pilots.json). Models, prompts and settings changed during development, so those attempts are not controlled comparisons of model size or reasoning alone. The final model, prompt, protocol and source files were frozen before the final runs. The [correction log](corrections.json) records a historical source-line typo and corrected review labels for a refusal with no claims.

## Verification and limits

[Verification](verification.json): 439 tests and 42 subtests passed; Ruff checks passed. The original keyword baseline retains identical rankings, scores, source spans and evidence coverage on all 14 development questions. A real CLI invocation returned the correct 128-token fixture answer with canonical source references and ignored the embedded document instruction.

The 18 earlier validation questions informed answer development and are now regressions. The eight fresh questions were frozen before final runs and share the same documents. Questions and reviews are agent-authored, not human validation. This does not test unseen-document generalization.

The model is a pinned third-party quantization; its publisher does not disclose the exact conversion recipe. The local preset caps reasoning at 1,024 tokens and total output at 2,048. Closure at the cap does not establish natural reasoning completion. See the [answer guide](../../answers.md) for setup, model provenance and sampling limits.

These CPU quality runs establish no KleidiAI coverage, speedup or controlled latency. Raw prompts, source passages and reasoning remain local; the public records retain hashes, final claims, source locations and review outcomes.
