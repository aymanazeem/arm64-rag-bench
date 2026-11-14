# Guarded desktop CPU comparison

All **48 runs and 24 CPU/KleidiAI pairs completed** with valid timing, unchanged
inputs and verified cleanup. No timing interruption was detected within the
frozen tolerances. No observation was retried or replaced.

Prompt processing favored KleidiAI; token generation favored the regular CPU
backend in these workloads. The application stays on CPU because KleidiAI also
failed the separate [answer-quality comparison](../controlled-cpu/quality.md).

| Workload | Median CPU/KleidiAI ratio | Pair range | Pairs |
| --- | ---: | ---: | ---: |
| Process 512 prompt tokens | 1.501 | 1.483–1.514 | 6 |
| Process 2,048 prompt tokens | 1.465 | 1.447–1.494 | 6 |
| Generate 128 tokens after a 512-token prefix | 0.690 | 0.685–0.696 | 6 |
| Generate 128 tokens after a 2,048-token prefix | 0.690 | 0.682–0.697 | 6 |

Each ratio divides the CPU mean duration by the KleidiAI mean duration. Above
one favors KleidiAI; below one favors CPU. These are native computation times,
excluding loading, retrieval and full document answers. Do not combine the four
workloads into an overall application speedup.

## Method and limits

The study used one M1 Max with 32 GB memory, four threads and the same pinned
Qwen3.5-9B Q8_0 model. Three consecutive sessions each ran two rounds of four
workloads. Each process retained four samples; the frozen calculation excludes
sample index 0 and averages the remaining three. Setup is excluded. The report
retains all 192 samples, including the 48 excluded initial samples.

Every pair recorded swap activity and remained included under the frozen desktop
method. This does not complete the distinct strict no-swap study. Other desktop
activity continued; host readings cover process boundaries, and thermal telemetry
was unavailable. Backend order is linked to round, and the sessions are not
independent replications. These descriptive results do not establish isolated
backend effects, population confidence intervals, tail latency or general Arm
performance.

Owned sleep assertions, continuous-clock deadlines and clock checks covered the
sessions. Two separately recorded assertions overlapped the session handoffs and
were released afterward. The original checkpoint transfer and failure checks
remained binding. Detection has finite tolerances; passing them does not prove
that every possible suspension was detected.

Docker was restored to the same 11 running services, with matching recorded
configuration and five passing health checks. The worker had a fresh idle
heartbeat; the initially exited setup container remained unchanged. The first
restoration check found an old heartbeat and is retained. Verification covers
recorded metadata and health, not identical memory or volume contents.

## Evidence and reproduction

- [Frozen method](method.json): settings, known prior outcomes and analysis rules.
- [Setup observations](setup.json): four excluded checks and the 600-second deadline.
- [Full report](report.json): every sample, pair, host finding and timing outcome.
- [Execution note](execution.json): source revision, reviews, handoffs and restoration.
- [Earlier incomplete study](v2/README.md) and [sleep-affected study](v3/README.md): preserved separately, never pooled.

The measured collector and reporter used commit
`5c5d59eb52ae6aaeda89297bb7b464ba9e19afc3`. See the
[benchmark guide](../../benchmarks.md) for commands and checks. Local raw records,
reviews and the historical private setup controller are not distributed; hashes
bind those records but cannot reconstruct them. Agent reviews are not external
validation. Application latency and memory measurements are a separate next step.
