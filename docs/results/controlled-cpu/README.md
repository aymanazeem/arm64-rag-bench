# CPU and KleidiAI checks

The Qwen3.5-9B Q8_0 checks confirm that KleidiAI's intended kernels ran. KleidiAI
does not pass the project's strict no-regression answer-quality rule, and CPU is
the application backend. The [desktop study](../desktop-cpu/README.md) completed 48 valid runs
in one uninterrupted invocation, with no observed swap activity. Prompt
processing favored KleidiAI; generation favored CPU.

These observations use the M1 Max with 32 GB memory, four CPU threads and the
pinned builds in the [benchmark method](../../benchmarks.md).

## Answer quality

Both backends completed all 40 known questions, with 30 complete answers each,
28 of them on the same questions. CPU had 78/78 supported claims and 66/83
required facts; KleidiAI had 80/81 supported claims and 67/83 facts. KleidiAI
gained three facts, but it also lost two and added one citation failure, which
the no-regression rule does not allow. The [quality report](quality.md) retains each case and review provenance,
and explains why these differences do not show that either backend is more
accurate. Successful dispatch does not override the quality rule.

## Runtime evidence

All 18 included diagnostic probes passed review. All eight instrumented
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
Each sampled KleidiAI phase used all 249 Q8_0 matrix weights in the model and
entered the Neon/DotProd kernels 3,600 times. The model's only other Q8_0 tensor
is the token-embedding table, which is read by lookup rather than matrix
multiplication. The entry counts include column slices, so they are not
full-operation counts or a percentage of accelerated work. Both backends also
showed 24 selected `SSM_CONV` and 24 `GATED_DELTA_NET` CPU observations per phase.

At prefix lengths 512 and 2,048, each backend restored its own state and
reproduced all 248,320 finite logits byte-for-byte for one fixed continuation.
CPU and KleidiAI logits differed from each other. This checks one-token replay
within each backend; it does not prove equal arithmetic or all future outputs.

Normal and instrumented builds matched all 35 compiler-flag fields and
controlled options. Their complete compile commands matched after replacing
source and build roots. Separate normal probes authenticated the executable
and six project libraries per backend. Uninstrumented timing processes rely on
those source and build bindings; they are not directly traced.

## Evidence and reproduction

The [compressed diagnostic record](diagnostics.json.gz) preserves all diagnostic
observations and exclusions. It contains no performance results.
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
