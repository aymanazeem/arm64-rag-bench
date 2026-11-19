# Buffered application performance

The CPU application completed **21/21 observations** on an M1 Max with 32 GB memory: 12 CLI commands and nine fixed-context replays. Timing, input identity, cleanup and answer review passed. No swap activity or timing interruption was observed.

Each entry is the median (minimum–maximum) across three eligible observations. Times are seconds. These are known questions, not new accuracy evidence or a backend comparison. KleidiAI was not measured here; the [project report](../../report.md#kleidiai-estimate-for-this-application) gives an untested estimate for it.

## CLI

| Case | Buffered first output | Server initialization | Complete request |
| --- | ---: | ---: | ---: |
| `dev-014`, 618 input tokens | 113.74 (112.95–116.51) | 14.28 (14.18–14.46) | 98.98 (98.29–101.59) |
| `val-015`, 1,807 input tokens | 147.78 (147.05–150.17) | 14.20 (14.17–14.26) | 132.99 (132.36–135.45) |
| `val-006`, 2,278 input tokens | 155.64 (154.80–156.38) | 14.16 (14.15–14.17) | 141.01 (139.88–141.73) |
| Empty retrieval | 14.81 (14.62–15.74) | 14.30 (14.20–14.35) | No generation |

Command time starts before ledger writing, worker creation and the ownership handshake. Buffered first output is **not model time to first token**, which remains unobserved. Initialization includes integrity checks and startup; about 9.8 seconds of it runs from server launch to readiness. Complete request includes input validation.

The generated cases produced 1,110, 1,239, 1,212 tokens respectively, including reasoning, in every repetition and stage. Median generation time was about 89–102 seconds of each complete request. Empty retrieval starts a server but makes no generation request.

## Fixed context

Each replay uses its question's first planned CLI input, with a fresh server and unchanged settings.

| Case | Fixed workflow return | Server initialization | Complete request |
| --- | ---: | ---: | ---: |
| `dev-014` | 113.49 (113.46–114.96) | 14.20 (14.13–14.24) | 98.86 (98.84–100.21) |
| `val-015` | 147.74 (146.75–150.42) | 14.22 (14.13–14.35) | 133.07 (132.18–134.43) |
| `val-006` | 154.46 (154.24–154.53) | 14.18 (14.18–14.24) | 139.64 (139.59–139.91) |

Fixed workflow return ends at application return, before observer export. It differs from the CLI first-byte receipt boundary. Do not subtract stage medians to estimate retrieval cost or infer a speedup. Fresh processes do not imply cold caches.

## Memory and quality

| Stage / case | Worker peak RSS, MiB | Native-child peak RSS, GiB |
| --- | ---: | ---: |
| CLI `dev-014` | 37.61 (37.50–37.77) | 16.806 (16.806–16.806) |
| CLI `val-015` | 39.08 (38.61–39.17) | 16.808 (16.806–16.808) |
| CLI `val-006` | 38.47 (38.33–38.83) | 16.806 (16.806–16.808) |
| CLI empty retrieval | 36.73 (36.73–36.98) | 16.806 (16.806–16.808) |
| Fixed `dev-014` | 36.12 (35.97–36.19) | 16.808 (16.806–16.808) |
| Fixed `val-015` | 36.33 (36.31–37.03) | 16.808 (16.808–16.808) |
| Fixed `val-006` | 36.61 (36.58–36.69) | 16.808 (16.807–16.808) |

These are separate process-lifetime peaks and must not be added. MiB and GiB use powers of 1,024; the [report](report.json) retains bytes and all observations.

Across 18 generated answers, **42/42 claims had supplied-source and attached-citation support, and 24/24 required fact instances passed**. The three empty controls declined faithfully. These repeated known answers do not extend the [40-case quality check](../grounded-answers/README.md). Reviews were reused only where inputs and final answers matched exactly. Empty controls were checked against the observed no-generation path. Reviews are by project-contributing agents, not human or external validation. [Quality evidence](quality.json).

## Conditions and scope

All measured observations are retained. Across 55 recorded host boundaries, swap counters did not increase. macOS, managed security and the controlling editor remained active; thermal readings were unavailable. These boundary readings do not establish CPU isolation or constant temperatures.

One bounded outer guard and AC-only sleep assertion covered readiness and all three stages. The two excluded compatibility checks passed. Collection-owned workers, native children and assertions were released and verified. No services or persistent power settings changed. [Execution](execution.json), [compatibility](compatibility.json).

The unmodified report retains pending semantic-review labels; companions supply that review. Its CPU source hashes identify frozen collection references, which may differ from current quality presentation hashes. [Method](../../application-performance.md), [project report](../../report.md).

## Reproduction

The recorded base working tree passed **956 tests and 42 subtests**, plus Ruff checks and formatting for 85 files. All 96 retrieval records matched their frozen rankings, scores, spans and coverage; 18 embedding checks and 29 asset-pin checks passed. The existing environment and cached assets were used. These checks do not establish a fresh installation, clean native build or exact merged checkout. [Verification scope](verification.json).
