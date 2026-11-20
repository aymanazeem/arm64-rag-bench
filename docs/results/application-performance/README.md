# CPU and KleidiAI application comparison

The unchanged `arm64-rag ask` command answered all 40 fixed questions once on each
backend, on an Apple M1 Max with 32 GB memory and four threads. Only the server
binary differed. **All 88 observations succeeded** in one session, with valid
timing and identical inputs for each question. Every answer matched the reviewed
answer for its question and backend, text and token count.

**KleidiAI took 46% longer in total: 125.1 minutes against 85.9 for CPU.** It was
slower on 38 of 40 questions, and the median per-question increase was 37%.

| Per question, median (range) | CPU | KleidiAI |
| --- | ---: | ---: |
| Answer time, s | 142.1 (71.7–175.3) | 196.3 (94.8–213.4) |
| Server start-up, s | 14.6 (14.4–15.7) | 30.5 (29.9–32.2) |
| Complete request, s | 126.5 (56.5–159.8) | 165.1 (63.9–181.9) |
| Generated tokens | 1,158 (326–1,266) | 1,170 (369–1,345) |

Answer time runs until the command's buffered answer arrives. Start-up includes
about 4.5 seconds, on both backends, spent mainly hashing the model and binaries
before the server launches. Complete request covers prompt processing and
generation. Prompts had 618–2,278 tokens and were the same on both backends.

## Where the time goes

| Total over 40 questions, s | CPU | KleidiAI | Difference |
| --- | ---: | ---: | ---: |
| Answer time | 5,155 | 7,508 | **+2,352** |
| Server start-up | 588 | 1,221 | +634 |
| Prompt processing | 1,233 | 871 | −363 |
| Generation | 3,310 | 5,386 | +2,076 |
| Retrieval, checks and harness steps | 24 | 30 | +5 |

Differences use unrounded totals.

- **Prompts:** KleidiAI processed 82.3 prompt tokens per second against 58.1, using
  29% less prompt time.
- **Generation:** KleidiAI took 119.7 ms per token against 83.7 ms, 43% more. Per
  question, its time per token was 1.18–1.49 times CPU's (median 1.44).
- **Answer length:** KleidiAI generated 45,009 tokens, 14% more than CPU's 39,522.
  With the same inputs and seed, the backends' different arithmetic leads sampling
  to different tokens, so lengths differ by question. CPU wrote at least twice as
  many tokens on `val-006` and `val-017`, the only questions where KleidiAI was
  faster. KleidiAI wrote at least twice as many on five questions.
- **Start-up:** `ask` starts a server for each question. Model loading and context
  setup took 25.1–25.7 seconds on KleidiAI and 9.6–10.0 on CPU across all 44
  starts per backend. While loading, each build converts the 8,041 MiB of Q8_0
  matrix weights into its own layout on one thread. In a sampled start, KleidiAI's
  conversion took 2.7 times as many samples as CPU's: 73% in llama.cpp's KleidiAI
  integration code, most often in two scalar loops that give each weight row a
  single scale, and 26% in KleidiAI's packing step.

The prompt and generation rates are close to the
[desktop study](../desktop-cpu/README.md), and the
[kernel timings](../kernel-analysis/README.md) account for most of the
differences.

## Similar-length answers

On the 22 questions whose two answers were within 5% in length (25,793 and 25,796
tokens in total), **KleidiAI took 33% longer** (median 33%, range 13–47%) and was
slower on all 22. Requests alone, without start-up, took 25% longer. Faster prompts
do not make up for slower generation.

## Checks and conditions

- **Pairs and order:** each question ran on both backends back to back, CPU first
  in 20 pairs and KleidiAI first in 20. Seed 42 fixed the order. Prompt and
  generation rates differed by at most about 1.5% depending on which backend ran
  first.
- **Inputs:** every question's passages and prompt matched the published quality
  records, and each question's inputs were identical on both backends.
- **No matching passages:** a question with no matching passages ran four times
  per backend and started a server without generating. It took 15.1 seconds on
  CPU and 30.9 on KleidiAI (medians), so start-up costs about 16 seconds more per
  command.
- **Memory:** median peak resident memory of the model server was 16.81 GiB on CPU
  and 17.20 GiB on KleidiAI over the question runs.
- **Deadlines:** the longest request took 182 seconds on KleidiAI and 160 on CPU,
  against a 300-second limit.
- **Host:** 120 idle seconds preceded every observation. All 176 host checks found
  AC power and normal memory pressure. Before each observation, the median
  one-minute load average was 5.5 on CPU and 5.2 on KleidiAI; other applications
  stayed open. No memory was swapped out, and swap-ins totalled 37 MiB from the
  first host check to the last. No sleep or clock jump was detected. Thermal
  readings were unavailable.
- **Excluded check:** one compatibility pair (`val-015` on each server) ran before
  collection. Both answers matched their reviewed records. It is not part of the
  results.

## Answers

All 80 generated answers matched the reviewed answers exactly, so the
[quality comparison](../controlled-cpu/quality.md) describes them: 30 complete
answers on each backend, and KleidiAI does not pass the strict no-regression rule.
The eight no-match commands declined without generating. The questions are known
and the answers repeat reviewed records, so they add no new accuracy evidence.

## Runtime evidence

The measured CPU build loads weights into `CPU_REPACK` and repacks 249 tensors as
`q8_0_4x4`. The KleidiAI build loads them into `CPU_KLEIDIAI`, with DotProd
kernels and SME disabled. In a sampled request, CPU ran `ggml_gemm_q8_0_4x4_q8_0`
and `ggml_gemv_q8_0_4x4_q8_0`. KleidiAI ran
`kai_run_matmul_clamp_f32_qai8dxp4x4_qsi8cxp4x4_16x4_neon_dotprod` and
`kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4x4_1x4_neon_dotprod`, its
prompt and single-token kernels. These diagnostic starts of the measured binaries
ran after collection with the application's server arguments plus `-v`.
[Load logs and samples](runtime.txt).

## Limits

- One machine, four threads and one model. Chips with I8MM or SME use other
  KleidiAI kernels. In the kernel study's thread sweep, generation tied at eight
  threads; the application was not tested there.
- Each question was answered once per backend with seed 42. Other seeds could give
  other answer lengths.
- The command prints its whole answer at once, so time to the first token was not
  observed.
- Fresh processes do not mean cold file caches: model verification reads the file.
- KleidiAI's extra start-up cost comes mostly from this pinned integration's Q8_0
  conversion. Other weight types and a long-running server were not tested.

## Files

| File | Contents |
| --- | --- |
| [report.json](report.json) | Plan and source hashes, all 88 observations, host readings and the comparison |
| [compatibility.json](compatibility.json) | The excluded compatibility pair |
| [runtime.txt](runtime.txt) | Load-log lines, start-up timings and sampled kernels |

Raw traces contain source passages and model reasoning and stay local. The
[method](../../application-performance.md) describes the design and how to rerun
it. The generated report leaves its interpretation label pending; this page gives
the interpretation.
