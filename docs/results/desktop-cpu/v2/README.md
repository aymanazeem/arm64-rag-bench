# Incomplete desktop CPU study

The study stopped after **9 of 48 planned runs** when memory pressure rose to
warning. All nine produced valid output; four pairs finished, one per workload.
Sessions 2 and 3 were not started. There were no retries or timeouts.
This is insufficient for a stable CPU/KleidiAI performance comparison.

The frozen method required Docker to be stopped. Replacement containers were
created during the ninth run, before the restoration attempt. The declared
condition therefore did not hold for the full study. The owner later confirmed
the Docker update; its effect on timing is not quantified. The newer running stack was
preserved; the exact original container state was not restored.

## Observed pairs

| Workload | Completed pairs / planned | CPU / KleidiAI compute duration |
| --- | ---: | ---: |
| Prompt: 512 tokens | 1 / 6 | 1.60 |
| Prompt: 2,048 tokens | 1 / 6 | 1.48 |
| Generate 128 after 512 cached tokens | 1 / 6 | 0.77 |
| Generate 128 after 2,048 cached tokens | 1 / 6 | 0.69 |

Each ratio divides the two process means, using three retained durations per
process. Above 1 favors KleidiAI for that observed pair. These are native compute
timings, excluding loading, prefix creation, state restoration and the Q&A
application. All four completed pairs ran KleidiAI first; the planned order
balance was not completed. Do not treat these values as general speedups.

The complete pairs ended before the first recorded replacement-container
creation. Docker was not monitored continuously. Swap activity remains visible
in the evidence and does not exclude valid pairs in this desktop mode.
The separate strict study remains incomplete, with its earlier setup excluded.

KleidiAI still fails the [answer-quality comparison](../../controlled-cpu/quality.md).
The application continues to use standard CPU inference.

## Evidence

- [Frozen method](method.json): settings, prior work, host rules and analysis.
- [Setup observations](setup.json): four eligible checks and the 600-second limit;
  setup timings are excluded from the comparison.
- [Verified report](report.json): all raw samples, pair ratios and 48 planned rows.
- [Execution note](execution.json): the stop and later sessions not being started.

The report has seven known unattempted rows in session 1 and 32 unobserved rows
because sessions 2 and 3 have no ledgers. The execution note separately records
that those sessions were not started. Local evidence is bound by hashes;
automated reviews are not external validation. See the
[benchmark guide](../../../benchmarks.md) for the runner and reporting commands.
