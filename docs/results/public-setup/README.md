# Public setup workflow validation

The [public setup commands](../../benchmark-setup.md) completed four new native observations using cached Qwen3.5-9B Q8_0 assets on an M1 Max. Both CPU backends completed prompt processing and generation setup, with four native repetitions per observation.

The [validation record](validation.json) retains source and evidence hashes, all four outcomes and the deadline calculation. All observations passed the frozen native, timing, host and cleanup checks. The linked clock checkpoints and additional idle-sleep assertion were verified after closure. Swap activity was recorded in three observations and retained under the desktop policy.

The candidate deadline is **600 seconds**. Its status remains **`pending_review`**; this record does not approve it for measurement.

This validates the freeze → setup → candidate-report workflow. It does not test a fresh download, clean build or cold file cache. Setup durations are excluded from performance comparisons. No new 48-run study, answer-quality result or backend adoption is claimed.

Raw machine-specific records remain local. The independent audit rechecked source and raw-record hashes, the recorded runtime fingerprints and cached model stat continuity. It did not rehash the model or rerun native work.
