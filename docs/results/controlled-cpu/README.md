# CPU and KleidiAI checks

The Qwen3.5-9B Q8_0 checks confirm that KleidiAI's intended kernels ran, but
the backend failed the frozen answer-quality comparison. The application stays
on CPU. Controlled performance results are pending host and setup checks.
A separate [desktop study](../desktop-cpu/README.md) completed all 48 runs, but
system sleep affected its timings. It does not complete the controlled comparison.

These observations use the M1 Max with 32 GB memory, four CPU threads and the
pinned builds in the [benchmark method](../../benchmarks.md).

## Answer quality

CPU passed 30 of 40 complete cases; KleidiAI passed 28. KleidiAI added one
citation failure, lost two previously supplied details, recovered three others,
and had one request failure followed by a dependent preparation failure.
The [quality report](quality.md) retains every case and the limits of this
comparison. Diagnostic success does not override these quality failures.

## Runtime evidence

All 18 final diagnostic probes passed review. One earlier probe remains
excluded because its controller changed after launch. All eight instrumented
CPU/KleidiAI pairs had identical input tokens, positions, phases and repetitions.
The four planned workloads included these observed inputs:

| Workload | Decode batches per process | Input tokens per process |
| --- | ---: | ---: |
| Prompt: 512 tokens | 5 | 2,560 |
| Prompt: 2,048 tokens | 20 | 10,240 |
| Generate 128 after 512 cached tokens | 514 | 1,025 |
| Generate 128 after 2,048 cached tokens | 517 | 2,561 |

Totals include warmup, any cached prefix and all four repetitions. Dispatch
was sampled in the first batch of each phase in repetition zero.
Each sampled KleidiAI phase used 249 distinct Q8 weights and entered the
Neon/DotProd kernels 3,600 times. These include column slices, so they are not
full-operation counts or a percentage of accelerated work. Both backends also
showed 24 selected `SSM_CONV` and 24 `GATED_DELTA_NET` CPU observations per phase.

At prefix lengths 512 and 2,048, each backend restored its own state and
reproduced all 248,320 finite logits byte-for-byte for one fixed continuation.
CPU and KleidiAI logits differed from each other. This checks one-token replay
within each backend; it does not prove equal arithmetic or all future outputs.

Normal and instrumented builds matched all 35 compiler-flag fields and
controlled options. Their complete compile commands matched after replacing
source and build roots. Separate normal probes authenticated the executable
and six project libraries per backend. Later uninstrumented processes still
rely on those source and build bindings; they are not directly traced.

## Evidence and reproduction

The [compressed diagnostic record](diagnostics.json.gz) preserves all reviewed
fields, including the excluded attempt. It contains no performance results.
Raw traces and token arrays remain local; the public record retains hashes,
counts and reviewed observations. Internal agent reviews are not external or
human validation.

The [diagnostic guide](diagnostics.md) explains the pinned source patch,
commands and validation limits. All diagnostic timings are excluded from
performance comparisons. Decompress the record with:

```sh
gzip -dc docs/results/controlled-cpu/diagnostics.json.gz > /tmp/arm64-diagnostics.json
shasum -a 256 /tmp/arm64-diagnostics.json
```

The uncompressed SHA-256 is
`5e00781aaf2af710c3cfcfe08b5708c2653d1e014143c39c1146f6f13194b652`.
