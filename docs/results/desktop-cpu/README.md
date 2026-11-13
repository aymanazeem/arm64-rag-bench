# Desktop CPU study: sleep affected the timings

All **48 runs and 24 CPU/KleidiAI pairs completed**, but the Mac slept during
three runs. These results do **not** establish an uninterrupted backend speed
comparison. Every observation remains in the report; none was removed or rerun.

The largest mismatch illustrates the problem: one KleidiAI generation sample
lasted 1,032.60 seconds, while the parent recorded only 98.20 seconds for the
whole process. The actual start-to-finish interval was 1,115.40 seconds.
The native timer counted system sleep; Python's deadline clock paused during it.
Power logs confirm sleep during this run, its paired CPU run, and a later CPU
prompt run. The [clock review](clock-review.json) records the evidence and limits.

The frozen report says `protocol_complete: true` because every scheduled run
finished and its declared boundary checks passed. Those checks did not detect
sleep. This flag does not establish valid uninterrupted timing. The recorded
600-second deadline also did not bound real elapsed time across sleep.

## What the record contains

The study used an M1 Max with 32 GB memory, four threads and the same pinned
Qwen3.5-9B Q8_0 model for both backends. Three consecutive sessions each ran two
rounds of four workloads, with balanced backend order.

Each process recorded four native samples. The frozen calculation excludes
sample index 0 and divides the CPU mean of the remaining three samples by the
KleidiAI mean. All 24 ratios, including those affected by sleep, remain in the
[report](report.json). No sleep duration was subtracted and no replacement
estimate was calculated.

Swap activity was retained under the desktop method. Power and memory checks
were taken at process boundaries; thermal limits were unavailable. Other
desktop activity continued. These consecutive sessions are not independent
replications, and the timers exclude loading, retrieval and the Q&A application.

Docker was temporarily stopped. After measurement, the same 11 updated services
were restored with matching recorded configuration and five passing health
checks. The worker was idle and the previously exited setup container was
unchanged. Verification covers recorded metadata and health, not identical
process memory or data-volume contents.

## Prior work and next step

The [earlier incomplete study](v2/README.md) is preserved separately. Its outcomes
were known when this repeat was planned and are not pooled with these results.
The strict study also remains incomplete.

KleidiAI still fails the [answer-quality comparison](../controlled-cpu/quality.md).
The application stays on CPU. Future measurements need sleep prevention,
a deadline that counts sleep, and checks that stop the study after an interruption.

## Evidence

- [Frozen method](method.json): settings, prior information and analysis rules.
- [Setup observations](setup.json): four checks and the derived deadline.
- [Frozen report](report.json): every sample, pair and planned outcome.
- [Execution note](execution.json): session accounting and Docker restoration.
- [Clock review](clock-review.json): sleep events and clock differences.

Setup timings are excluded from the comparison. Local evidence is bound by
hashes; automated reviews are not external validation. See the
[benchmark guide](../../benchmarks.md) for commands and measurement limits.
