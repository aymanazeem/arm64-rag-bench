# Why KleidiAI prompts are faster and generation slower

In the [desktop study](../desktop-cpu/README.md), KleidiAI used about a third less
time on prompts and about 42% more on generation. Most of that time is spent
multiplying the model's weights by token data, and each build uses its own
kernels for this. These timings show why those kernels behave differently.

They are diagnostic runs on the study's M1 Max, outside the controlled study.
Harness figures are the best of three to five trials; llama-bench figures are
means of three. [Conditions](conditions.txt) record power and load for the
harness run.

## What ran

- Both builds convert the same Q8_0 matrices (8,041 MiB in the model file, all
  except the token-embedding table) into their own layout: `CPU_REPACK` for CPU
  and `CPU_KLEIDIAI` for KleidiAI ([load log](buffers.txt)). The embedding table
  is read by lookup.
- On this chip (DotProd, no I8MM or SME), KleidiAI uses
  `kai_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod` for prompts
  and `kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod` for one
  token at a time. The study's dispatch traces
  ([records](../controlled-cpu/diagnostics.json.gz),
  [method](../controlled-cpu/diagnostics.md)) record both.
- CPU uses `ggml_gemm_q8_0_4x4_q8_0` for prompts and `ggml_gemv_q8_0_4x4_q8_0`
  for one token. In the study's CPU library
  ([hashes](../runtime-feasibility/cpu-build.json)), the prompt kernel has the
  same instruction sequence as the copy timed here.
- Both drivers split each large matrix into 16 column chunks at four threads.
  The prompt harness copies that split and the model's layer shapes.

## Prompts: KleidiAI keeps the multiply units busy

Each core can start four `SDOT` instructions per cycle; one `SDOT` does 16
8-bit multiply-adds. With four threads running, the measured peak was about 196
billion multiply-adds per second per core.

| Prompt kernel, 512 tokens, 4 threads | Billion multiply-adds/s | Share of peak | Weight time per pass |
| --- | ---: | ---: | ---: |
| KleidiAI | 765 | 98% | 4.63 s |
| CPU | 472 | 60% | 7.50 s |
| CPU without per-block scaling (test only) | 603 | 77% | 5.87 s |

KleidiAI's kernel works on 16 tokens and 4 weight columns at once. It keeps 16
independent sets of running totals, one per token, each covering the 4 columns.
Each total runs across the whole weight row, and one scale is applied at the
end. Its inner loop has no vector instructions other than `SDOT`.

The CPU kernel works on 4 tokens and 4 columns. Q8_0 has one scale per 32
weights, so it restarts its four sets of totals every 32 weights, then converts
and scales them. That adds 18 vector instructions to every 32 `SDOT`s. On one
core with data in cache, against a peak of about 202 billion per second,
instruction counts predict 64% and it measured 61%. Removing only the scaling
predicts 80%, and it measured 78%. KleidiAI measured 96%.

The kernels alone make KleidiAI 1.62 times as fast. The full model gains less
(1.50 and 1.46 times in the study), mostly because the rest of its work runs the
same code in both builds. On this chip, KleidiAI has no kernel for the F16
attention products.

## Generation: KleidiAI waits on one set of totals

| One token, weights from memory, billion weights/s | 1 thread | 4 threads | 8 threads |
| --- | ---: | ---: | ---: |
| KleidiAI | 19.3 | 72.8 | 110.2 |
| CPU | 53.6 | 101.2 | 108.3 |
| KleidiAI layout, 4 sets of totals (test only) | 58.4 | 108.6 | 114.0 |

Each KleidiAI weight takes about one byte. CPU weights take 34 bytes per 32,
including their scales, so the CPU's raw reads are 6% higher than shown
(107.5 GB/s at four threads).

KleidiAI's single-token kernel adds every product for four columns into one
register across the whole row. Each `SDOT` must wait about 3.0 cycles for the
previous one. A single chain like that allows about 17 billion weights per
second per core; the kernel measured about 19, probably because neighbouring
column groups overlap slightly. With data already in cache it is no faster (19.4), so
the limit is the wait, not memory.

The CPU kernel restarts its totals every 32 weights, so the core can work on
several blocks at once. It gains little from four to eight threads, but
KleidiAI needs eight to catch up. Splitting KleidiAI's loop into four sets of
totals, with the same data and bit-identical results, made it three to four
times as fast on one core.

The full model follows the same pattern (llama-bench, 32 generated tokens, no
prior context):

| Threads | CPU ms/token | KleidiAI ms/token | CPU/KleidiAI |
| ---: | ---: | ---: | ---: |
| 1 | 160.0 | 437.1 | 0.37 |
| 2 | 93.3 | 224.0 | 0.42 |
| 4 | 83.7 | 118.4 | 0.71 |
| 8 | 84.9 | 84.1 | 1.01 |

CPU stops improving at four threads, consistent with a memory limit. KleidiAI
keeps improving and ties at eight. At four threads, a two-token batch took
133.2 ms, only 14 ms more than one token (119.4 ms). In llama.cpp's source, any
batch of more than one token uses KleidiAI's prompt kernel.

## Limits

- Normal desktop background activity: the window server, accessibility services
  and the editor were active. No thermal data was recorded.
- The harnesses time kernels outside llama.cpp, with random values. They do not
  include activation packing, llama.cpp's thread pool or the cache state
  between layers.
- The kernel timings explain most, not all, of the full-model generation times.
  At four threads, the harness rates account for about 78 of CPU's 84 ms per
  token and 109 of KleidiAI's 118 ms. They cover about 31 of the 35 ms
  difference.
- Chips with I8MM but no SME select a different single-token kernel, with two
  sets of totals. Timed on this M1, it read 36 billion weights per second on
  one core from memory, against 19 for the kernel this chip uses. Those chips
  were not tested.

## Files and rerun

| File | Contents |
| --- | --- |
| `sdot_latency.c`, `sdot_latency.txt` | Wait between dependent `SDOT`s |
| `sdot_peak.c`, `sdot_peak.txt` | Peak `SDOT` rate and clock at 1, 2 and 4 threads |
| `gemm_bench.c`, `gemm_bench.txt` | Prompt kernels on the model's layer shapes |
| `gemv_bench.c`, `gemv_bench.txt` | Single-token kernels in GB/s, bit-identity check |
| `gemv_1x8_bench.c`, `gemv_1x8_bench.txt` | Single-token kernel used with I8MM |
| `conditions.txt` | Power source and load averages around the harness run |
| `buffers.txt` | Each build's weight layout from the load log |
| `llama-bench-cpu.json`, `llama-bench-kleidiai.json` | Full-model thread sweep |
| `run.sh` | Builds and reruns the harnesses |

The harnesses need the pinned KleidiAI checkout:

```sh
git clone --no-checkout https://github.com/ARM-software/kleidiai vendor/kleidiai-v1.32.0
git -C vendor/kleidiai-v1.32.0 checkout --detach c80157210987eedaab2509638ef5e15950c53a8c
sh docs/results/kernel-analysis/run.sh
```

The ggml kernel copies keep llama.cpp's MIT notice in `LICENSE-llama.cpp.txt`.
The thread sweep ran each study build's `llama-bench` on the pinned model with
`env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C` and
`-p 1,2,4,16 -n 32 -d 0 -r 3 -t 1,2,4,8 --offline -b 512 -ub 512 -ctk f16
-ctv f16 -fa off -ngl 0 -lm auto -lzm auto --repack 1 -o json`.
