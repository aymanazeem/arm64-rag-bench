# Measure application latency

This harness measures the existing buffered CPU answer command and its fixed-context request path. It does not change the prompt, model, retrieval settings or application output. The [initial results](results/application-performance/README.md) retain 21 reviewed observations and the limits of this known-question study. A separate [tracked-source workflow check](results/application-performance/README.md#reproduction) used cached native assets.

The two stages answer different questions:

- **CLI:** How long does a fresh command take to return an answer, including model checks, server startup, retrieval, generation and shutdown?
- **Fixed context:** How long does the existing complete request take after a fresh server is ready, using exactly the inputs collected by the CLI stage?

Each request uses a fresh server. Neither stage establishes a cold filesystem cache: model verification reads the model file, and the operating system may retain pages. The stages run separately. Do not subtract their summary values to estimate retrieval cost or claim a speedup.

## Prepare public inputs

Use the [runtime guide](runtime.md) to build the pinned CPU server and download the [answer model](answers.md). Build the corpus index from the checked source revisions:

~~~sh
uv run --locked --group build arm64-rag index \
  --manifest data/manifests/corpus.json \
  --db data/private/development.sqlite --sync
~~~

Preparation verifies that index against the public manifest and evaluation labels. It freezes the current CPU build, model, application sources, host policy, timing helper and complete schedule:

~~~sh
uv run --locked --group build python scripts/benchmark_answers.py prepare \
  --db data/private/development.sqlite \
  --output artifacts/benchmarks/application-01/plan.json
~~~

Review the saved plan before running. Use AC power and normal memory pressure. The baseline power mode, host identity and available thermal fields must stay unchanged. Unavailable thermal data remains a limitation. Swap counters are recorded; a positive increase alone does not select or exclude observations. Missing or reset counters stop later launches. Declare the background-service condition locally before collection and restore any service that you temporarily stop.

On macOS, Apple's [idle-sleep assertion](https://github.com/apple-oss-distributions/IOKitUser/blob/main/pwr_mgt.subproj/IOPMLib.h#L253-L270)
has no effect during Dark Wake. For this operating condition, also hold a temporary AC-only
system-sleep assertion in a second terminal. Run `caffeinate -s -t 1800` before
the compatibility sequence, `caffeinate -s -t 9000` before the CLI stage, and
`caffeinate -s -t 7200` before the fixed stage. Use one assertion at a time.
Confirm its own PID and `PreventSystemSleep` assertion with `pmset -g assertions`
before and after the stage, then press Ctrl-C in that terminal and verify release.
These timeouts are safety caps, not promised stage durations. If the assertion
expires or disappears early, retain an operational failure; do not silently
renew it or replace the run. Keep the lid open. This changes no persistent power
setting and does not replace the runner's clock, power or cleanup checks. See
Apple's [caffeinate manual](https://github.com/apple-oss-distributions/PowerManagement/blob/main/caffeinate/caffeinate.8).

Preparation hashes the model and other assets. It does not launch the model. The model is verified again before each stage and by the original application on every fresh command. Each stage keeps a verified local copy of the exact source, configuration and dataset bytes under its sources directory. No model or binary is copied. Do not alter the frozen source, index, model or runtime files until both stages finish.

The three public questions are dev-014, val-015 and val-006. They represent the short, lower-median and long packed inputs in the published CPU run: 618, 1,807 and 2,278 tokens. Selection used input length, not answer quality or latency. These questions are known regression cases. The CLI stage also includes a synthetic query verified to have no lexical matches.

## Collect both stages

After fake tests and owned-process checks pass, perform at most one compatibility check for this frozen plan:

~~~sh
uv run --locked --group build python scripts/benchmark_answers.py check \
  --plan artifacts/benchmarks/application-01/plan.json \
  --output artifacts/benchmarks/application-01/compatibility
~~~

This runs the already-selected median case, val-015, once through the CLI and then replays that exact input once if the CLI passes. Both observations are marked excluded. They cannot supply the measured replay stage or enter its summaries. A failed first observation leaves the second unattempted. Do not tune the prompt or replace the case using this check.

~~~sh
uv run --locked --group build python scripts/benchmark_answers.py run \
  --plan artifacts/benchmarks/application-01/plan.json --stage cli \
  --output artifacts/benchmarks/application-01/cli

uv run --locked --group build python scripts/benchmark_answers.py run \
  --plan artifacts/benchmarks/application-01/plan.json --stage fixed \
  --cli-run artifacts/benchmarks/application-01/cli/run.json \
  --output artifacts/benchmarks/application-01/fixed
~~~

The CLI stage has three blocks, each with three generated answers and one empty-retrieval control: 12 fresh commands and nine generations. The fixed stage has nine fresh servers and nine generations. Seed 42 fixes the within-block order. There is a 120-second guarded rest before each observation. These are descriptive repeats, not a statistical power calculation.

The fixed stage takes the **first planned CLI trace for each generated question**. Every CLI repeat must have the same question, retrieved sources, packed passages, prompt and token IDs. A local index generation counter is retained but excluded from portable input identity. No private owner trace is needed. The original complete API still verifies the reconstructed prompt and token IDs before generation.

Startup keeps its 120-second application deadline; requests keep their 300-second deadline and the existing 1,024-token reasoning and 2,048-token total budgets. An independent 600-second continuous-clock deadline runs from the parent attempt checkpoint through worker exit. The harness never retries or replaces a failed observation. A create-only launch record beside the plan prevents reusing a stage. A stopped stage retains all planned rows, partial output and later unattempted rows. Fixed replay requires the complete, valid CLI stage.

The supervisor owns the sleep assertion, host probes and worker group. A start handshake registers ownership before application execution. Only the benchmark worker changes the native launch to inherit its process group; the executable, arguments, environment, standard streams and requests stay unchanged. SIGINT and SIGTERM lead to retained failure and bounded cleanup. The harness does not guarantee cleanup after supervisor SIGKILL, power loss or kernel failure.

## Read the metrics

| Metric | Boundary |
| --- | --- |
| Pipe-observed answer | Parent attempt checkpoint to first nonempty stdout bytes received by the parent, before disk capture. |
| Application return | Parent attempt checkpoint to the original CLI return, before observer export. |
| Worker exit observed | Parent attempt checkpoint to the parent's non-reaping exit observation; includes observer work and polling delay. |
| Verified server initialization | Original server entry to validated readiness, including integrity checks. |
| Native spawn to ready | Native launch to validated readiness; not pure model-loading time. |
| Ready-server RAG | Original answer function entry to its validated result. |
| Complete request | Existing nonstreaming complete call, including its input-validation requests. |
| Peak RSS | Separate worker and reaped native-child lifetime peaks, in Darwin bytes. |

The attempt checkpoint precedes the durable ledger write, worker creation and ownership handshake. Command-level durations include those steps and worker imports.

The plain-text command is buffered. The worker timestamps application return, flushes stdout immediately, then records memory and writes observer files. That explicit flush is a benchmark condition, not a claim about unchanged redirected buffering. Flush and observer durations remain separate; fixed replay labels its workflow return separately from CLI return. **Model time to first token is unobserved.** Native prompt timing is not a substitute for it.

The native-child memory value is attributed only when one native server was launched and reaped. Missing values stay unavailable. Do not sum process peaks or treat RSS as unique system memory. System-wide swap activity does not prove that this child paged.

Native timing diagnostics and generated-token counts accompany latency. Prompt-cache reuse must be explicitly zero. A final slot token count is not cache reuse. Native decode throughput has its own token-count convention and is not the buffered application's throughput.

## Export and review

~~~sh
uv run --locked --group build python scripts/benchmark_answers.py report \
  --plan artifacts/benchmarks/application-01/plan.json \
  --cli-run artifacts/benchmarks/application-01/cli/run.json \
  --fixed-run artifacts/benchmarks/application-01/fixed/run.json \
  --output artifacts/benchmarks/application-01/report.json
~~~

The report retains all 21 planned observations. Omit an unstarted stage argument to report it as unobserved. It checks raw file hashes, source and input identity, owned cleanup, host boundaries and clock evidence. Its completion flag covers those implemented checks; background-condition and scientific interpretation review remain separate. Summaries show per-question medians and ranges within each stage; they never pool the stages or produce a backend speed ratio. Invalid timing remains visible without a valid timing summary.

Raw traces contain source passages and model reasoning. Keep them local. The public report uses an allowlist of metric values, counts, fixed labels and hashes; it excludes raw text, local paths, process IDs, environment values and private errors. Its portable input manifest includes the first collected packed, source, token, message and schema identities. Failed observations retain available time-to-failure, cleanup status and nullable memory or token observations separately from completed-answer latency.

Structural checks do not grade answers. Review every changed final answer against the frozen required facts and actual cited passages before making a quality claim. The plan carries those labels; each report row binds its raw trace and validated-answer hash. An owner may use authenticated prior raw records for exact-output review reuse, but that is optional and does not replace the public input workflow. Incomplete facts, refusals and support failures must remain visible. The report deliberately leaves semantic quality pending until that separate review is complete.

Before the first native use, run the focused fake tests and the small owned-process lifecycle checks on the target Mac. Those checks must cover early worker death, launch cancellation, output capture, a surviving child, group cleanup and identity reuse. They are compatibility checks, not latency samples. Do not make a reproducibility or performance claim before the real public workflow and its retained evidence have been reviewed.
