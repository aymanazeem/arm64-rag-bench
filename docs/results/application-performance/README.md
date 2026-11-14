# Buffered application performance

The CPU application completed **21/21 observations** on an M1 Max with 32 GB memory: 12 CLI commands and nine fixed-context replays. Timing, input identity, cleanup and answer review passed. Service restoration and the separate source-export workflow check also passed.

Each entry below is the median (minimum–maximum) across three observations. Times are seconds. These are descriptive repeats of three known questions, not a fresh accuracy evaluation or a backend comparison.

## CLI

| Case | Buffered first output | Server initialization | Complete request |
| --- | ---: | ---: | ---: |
| `dev-014`, 618 input tokens | 116.79 (115.92–117.01) | 16.71 (16.53–16.77) | 99.56 (98.74–100.02) |
| `val-015`, 1,807 input tokens | 149.25 (148.39–149.99) | 16.01 (15.54–16.19) | 132.84 (132.39–133.20) |
| `val-006`, 2,278 input tokens | 157.67 (156.85–159.17) | 16.71 (16.44–16.80) | 140.28 (139.75–141.97) |
| Empty retrieval | 15.96 (15.53–16.11) | 15.52 (15.06–15.67) | No generation |

Command time begins at the parent attempt checkpoint, before ledger writing, worker creation and the ownership handshake. Output uses a pipe with an explicit post-return flush. Buffered first output is **not model time to first token**, which remains unobserved. Initialization includes integrity checks and startup; the complete request includes input-validation calls.

The generated cases produced 1,110, 1,239 and 1,212 tokens respectively, including reasoning, in every repetition. The empty control starts a server but makes no generation request.

## Fixed context

Each replay used the exact packed input from its question's first planned CLI observation, with a fresh server and unchanged settings.

| Case | Fixed workflow return | Server initialization | Complete request |
| --- | ---: | ---: | ---: |
| `dev-014` | 115.56 (114.93–115.67) | 15.62 (15.54–15.64) | 99.63 (98.97–99.66) |
| `val-015` | 151.49 (149.87–151.62) | 16.46 (16.06–16.76) | 134.32 (132.89–134.62) |
| `val-006` | 159.20 (158.19–159.32) | 16.70 (15.77–16.80) | 142.10 (141.11–143.04) |

Workflow return uses the parent attempt checkpoint through fixed-worker application return, before observer export. It differs from the CLI table's first-byte receipt boundary. Request durations are measured directly. The stages ran separately; do not subtract their medians to estimate retrieval cost or infer a speedup. Fresh processes do not imply cold file caches.

## Memory and quality

| Stage / case | Worker peak RSS, MiB | Native-child peak RSS, GiB |
| --- | ---: | ---: |
| CLI `dev-014` | 35.70 (35.66–37.31) | 15.60 (15.48–15.65) |
| CLI `val-015` | 38.64 (38.42–38.64) | 15.70 (15.57–15.80) |
| CLI `val-006` | 35.64 (35.58–35.77) | 15.39 (15.16–15.56) |
| CLI empty retrieval | 36.69 (36.66–36.94) | 15.72 (15.65–16.11) |
| Fixed `dev-014` | 35.55 (35.53–35.86) | 15.77 (15.71–15.83) |
| Fixed `val-015` | 36.06 (35.91–36.08) | 15.22 (15.13–15.61) |
| Fixed `val-006` | 36.19 (35.80–36.19) | 15.59 (15.43–15.65) |

These are separate process-lifetime peaks, not additive memory use. MiB and GiB use powers of 1,024; the [report](report.json) retains bytes and all observations.

Across 18 generated answers, **42/42 claims had supplied-source and attached-citation support, and 24/24 required fact instances passed**. Three empty controls declined faithfully. These repetitions do not extend the [40-case baseline](../grounded-answers/README.md). Contributor AI agents performed the reviews; this is not human or external validation. [Quality evidence](quality.json).

## Conditions and retained failures

Swap activity occurred in seven CLI and five fixed observations; all 12 remained included. Thermal readings were unavailable. Host readings cover boundaries, not continuous conditions. The benchmark changes the native child's process group for owned cleanup while preserving requests and settings.

V1's excluded compatibility attempt stopped after maintenance sleep without an answer; its fixed replay was unattempted. V2 retained the clock rules and added a temporary AC-only system-sleep assertion. Both excluded v2 checks passed. Owned assertions were confirmed and released after each stage. No measured observation was retried or replaced. Docker restoration verified the same 11 running container identities, five healthy services and recorded configuration/mount metadata; the initially exited setup container stayed untouched. This does not establish identical RAM or volume contents. [Execution and exclusions](execution.json).

The unmodified report records mechanical completion and deliberately leaves semantic-review fields pending. The separate quality and execution records provide that review; they do not rewrite raw measurements. See the [method and reproduction commands](../../application-performance.md) and [project report](../../report.md).

## Reproduction

A clean export of proposed tracked tree `3f2d7f7b824ab36016ef2fbc6a178467746d8e6a` used a new Python 3.12.15 environment with locked, offline cached dependencies. All **933 tests passed**; 23 optional checks were skipped. Lint, formatting and six command-help checks passed. A new corpus index reproduced seven documents, 383 chunks and the expected fingerprint.

One familiar fixture answer correctly reported its 128-token batch size, with two supporting citations and the embedded instruction ignored. Cleanup passed. Setup freeze and application preparation succeeded; an unstarted report retained all 21 rows as unobserved. Native assets were cached: this was not a fresh download/build, a repeat of the performance study or a new 40-case quality run. An exact merged-checkout check is a later release step. [Verification scope](execution.json).
