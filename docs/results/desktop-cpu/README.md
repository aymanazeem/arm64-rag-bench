# Uninterrupted CPU comparison

All **48 runs and 24 CPU/KleidiAI pairs** completed in one
controller invocation. All three blocks passed input-integrity and timing checks;
no timing interruption was detected within the frozen tolerances. No observation
was retried or replaced. No swap activity was observed across the measured period.

KleidiAI processed prompts faster but generated tokens slower. CPU is the
application default: KleidiAI does not pass the separate
[answer-quality rule](../controlled-cpu/quality.md), and the whole application
took 46% longer with it in the
[application comparison](../application-performance/README.md).

| Workload | Median CPU/KleidiAI ratio | Pair range | Pairs |
| --- | ---: | ---: | ---: |
| Process 512 prompt tokens | 1.503 | 1.473–1.512 | 6 |
| Process 2,048 prompt tokens | 1.464 | 1.443–1.467 | 6 |
| Generate 128 tokens after a 512-token prefix | 0.699 | 0.670–0.746 | 6 |
| Generate 128 tokens after a 2,048-token prefix | 0.704 | 0.702–0.710 | 6 |

Each ratio divides the CPU mean duration by the KleidiAI mean duration. Above one
favors KleidiAI. The corresponding KleidiAI time changes are **33.5% and 31.7%
less** for prompts, and **43.0% and 42.1% more** for generation. These are native
computation times, excluding loading, retrieval and full document answers.

## Computation time

| Workload | CPU seconds | KleidiAI seconds | CPU tokens/s | KleidiAI tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Process 512 prompt tokens | 8.277 | 5.545 | 61.9 | 92.3 |
| Process 2,048 prompt tokens | 34.943 | 23.861 | 58.6 | 85.8 |
| Generate 128 tokens after a 512-token prefix | 10.485 | 14.971 | 12.2 | 8.6 |
| Generate 128 tokens after a 2,048-token prefix | 10.784 | 15.303 | 11.9 | 8.4 |

These times are medians of six process means per backend, and the token rates
are derived from them. They are distinct from the median paired ratios above.
All six pairs per workload are included.

## Method and limits

One M1 Max with 32 GB memory ran the pinned Qwen3.5-9B Q8_0 model with four threads,
F16 caches and zero GPU layers. Metal, BLAS, Accelerate and OpenMP were disabled.
Three consecutive logical blocks each ran two rounds of four paired workloads,
with fixed 120-second rests. In each block, KleidiAI ran first in every round-1
pair and CPU ran first in every round-2 pair. Pair ratios were similar in either
order, but because order is tied to round, order and round effects cannot be
separated. The blocks are not independent replications.

Each process retained four raw samples. The frozen calculation excludes sample
index 0 and averages the other three. All **192 raw samples**, including the
**48 excluded initial samples**, are retained. Four separate setup observations
set the 600-second deadline and are excluded from the comparison.

The Mac was restarted before preparation. Docker/VM and other intentional jobs
were absent; no applications or services needed stopping. macOS, managed security
and the controlling editor remained active. Small status-file reads occurred
during collection. This was not CPU isolation or zero background activity.
Thermal readings were unavailable; host observations otherwise cover boundaries.
The desktop policy retains swap activity if it occurs. These results do not
establish a general Arm speedup.

Recorded load was high before and at the start of measurement. The 15-minute
load average was 65.7 before setup and 22.7 when the first block began, and
stayed below 8 in the second and third blocks. Block median ratios differed
between blocks by at most 0.042. CPU
generation after the 512-token prefix varied most, with block medians of 10.98,
10.42 and 10.25 seconds.

In the four setup observations, KleidiAI's peak memory was 17.3–17.4 GiB,
against 15.2–16.7 GiB for CPU. Measured runs did not record memory, so this is
not a controlled memory comparison. Subtracting the timed samples and
llama-bench's untimed warmup or prefix run from each process's wall time leaves
about 26 seconds per KleidiAI process, against about 10 for CPU. This suggests
KleidiAI takes about 16 seconds longer to load, consistent with its weight
conversion.

One outer sleep guard covered readiness, all three measured blocks and automatic
handoffs. Per-block guards, suspend-aware deadlines and exact checkpoint transfer
remained active. All owned assertions were released. Detection has finite
tolerances and cannot prove that every possible interruption was detected.

This study does not establish population confidence intervals or tail latency.

## Evidence and reproduction

- [Method summary](method-summary.json): settings, controls and analysis rules.
- [Setup evidence](setup.json): four excluded observations and deadline calibration.
- [Full report](report.json): every measured sample, pair, host finding and outcome.
- [Execution note](execution.json): controller provenance, continuity and cleanup.

The method summary is an edited description. The report and setup record retain
the hash of the original frozen method, which remains local; that hash does not
identify the summary.

The public collector and reporter used commit
`15dc33dc736c5aaa92fa8afd3e8f61d07dd65241`. See the
[benchmark guide](../../benchmarks.md) and [setup workflow](../../benchmark-setup.md).
The local supervisor called the unchanged public APIs. Its source, private raw
records and local reviews are not distributed; their hashes cannot reconstruct
missing files. Local automated review is not external validation.
