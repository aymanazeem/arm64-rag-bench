# Setup workflow validation

The public setup API completed **four observations** on the M1 Max using cached
Qwen3.5-9B Q8_0 assets. Both CPU backends completed prompt and generation setup,
with four native repetitions each. A local supervisor called the unchanged API
and kept one outer sleep guard active throughout collection.

All observations passed the native, timing, host and cleanup checks. **No swap
activity was observed** across readiness and setup. The calibration produced a
**600-second deadline**, which was separately reviewed and approved for the
[desktop CPU study](../desktop-cpu/README.md). The exported candidate remains
unchanged; approval is a separate record.

The [validation record](validation.json) contains the counts, source hashes,
observation hashes and guard checks. [Detailed setup evidence](../desktop-cpu/setup.json)
retains all native samples and resource observations. These are the same setup
observations used by the desktop study; they are excluded from its performance
comparison.

This validates the setup API and calibration workflow. It does not establish
fresh downloads, clean builds or cold caches. macOS, managed security and the
editor remained active; thermal readings were unavailable. Boundary and clock
checks have finite limits. Raw local records and the supervisor are not distributed.
