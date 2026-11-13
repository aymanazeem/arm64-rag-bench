"""Session-owned sleep prevention and suspend-aware benchmark deadlines."""

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time

POLICY = {
    "version": 1,
    "platform": "darwin",
    "continuous_clock": "CLOCK_MONOTONIC_RAW",
    "awake_clock": "CLOCK_UPTIME_RAW",
    "sample_attempts": 3,
    "sample_budget_ns": 100_000_000,
    "max_bracket_ns": 1_000_000,
    "allowance_ns": 1_000_000,
    "native_sample_allowance_ns": 1_000_000,
    "poll_ns": 100_000_000,
    "assertion_flags": ["-i", "-w"],
    "assertion_start_seconds": 3,
    "probe_seconds": 0.5,
    "cleanup_seconds": 5,
}
REASONS = {
    "unsupported_clock",
    "clock_unavailable",
    "clock_went_backwards",
    "clock_sampling_uncertain",
    "clock_discontinuity",
    "clock_context_unavailable",
    "previous_checkpoint_mismatch",
    "assertion_start_failed",
    "assertion_lost",
    "deadline_exceeded",
    "assertion_cleanup_failed",
    "native_cleanup_failed",
    "native_duration_inconsistent",
    "guard_interrupted",
    "guard_check_failed",
}
SAMPLE_KEYS = {"continuous_before_ns", "awake_ns", "continuous_after_ns"}
STATS_KEYS = {
    "sample_attempts",
    "accepted_samples",
    "resamples",
    "max_bracket_ns",
    "max_gap_lower_ns",
}
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
LABEL = re.compile(r"[A-Za-z0-9_.:-]{1,100}\Z")


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def host_identity_hash(identity):
    return hashlib.sha256(encoded(identity).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_policy(policy):
    require(encoded(policy) == encoded(POLICY), "Timing policy differs.")


def check_sample(sample):
    require(
        isinstance(sample, dict) and set(sample) == SAMPLE_KEYS, "Invalid clock sample."
    )
    require(
        all(type(value) is int and value >= 0 for value in sample.values()),
        "Invalid clock count.",
    )
    require(
        sample["continuous_after_ns"] >= sample["continuous_before_ns"],
        "Reversed clock sample.",
    )


def interval_reason(before, after, policy=POLICY):
    check_sample(before)
    check_sample(after)
    if (
        after["continuous_before_ns"] < before["continuous_after_ns"]
        or after["awake_ns"] < before["awake_ns"]
    ):
        return "clock_went_backwards"
    lower = (
        after["continuous_before_ns"]
        - after["awake_ns"]
        - (before["continuous_after_ns"] - before["awake_ns"])
    )
    upper = (
        after["continuous_after_ns"]
        - after["awake_ns"]
        - (before["continuous_before_ns"] - before["awake_ns"])
    )
    if lower > policy["allowance_ns"] or upper < -policy["allowance_ns"]:
        return "clock_discontinuity"
    return None


class GuardError(RuntimeError):
    def __init__(self, reason, sample=None):
        super().__init__(reason)
        self.reason = reason
        self.sample = sample


class CleanupError(RuntimeError):
    pass


def stop_process(
    process,
    *,
    clock_ns=None,
    sleep=time.sleep,
    fallback_ns=time.monotonic_ns,
    seconds=5,
):
    """Retry one interruption of owned cleanup, then report it after reaping."""
    mode, interrupted = "continuous", False

    def call(function, *args):
        nonlocal interrupted
        try:
            return function(*args)
        except KeyboardInterrupt:
            interrupted = True
            return function(*args)

    if process is None:
        return "not_needed"
    if call(process.poll) is not None:
        if interrupted:
            raise CleanupError("cleanup_interrupted")
        return "not_needed"
    try:
        if clock_ns is None:
            raise OSError
        call(clock_ns)
    except (OSError, ValueError, AttributeError):
        clock_ns, mode = fallback_ns, "monotonic_fallback"

    def wait():
        nonlocal clock_ns, mode
        for fallback in (False, True):
            if fallback:
                clock_ns, mode = fallback_ns, "monotonic_fallback"
            try:
                previous = call(clock_ns)
                deadline = previous + int(seconds * 1e9)
                waited = False
                while call(process.poll) is None:
                    current = call(clock_ns)
                    if current < previous or (waited and current == previous):
                        raise ValueError("cleanup_clock_stalled")
                    previous = current
                    remaining = deadline - current
                    if remaining <= 0:
                        return False
                    call(sleep, min(0.1, remaining / 1e9))
                    waited = True
                return True
            except (OSError, ValueError, AttributeError):
                continue
        return False

    for action, reason in (
        (process.terminate, "terminate_failed"),
        (process.kill, "kill_failed"),
    ):
        try:
            call(action)
        except ProcessLookupError:
            pass
        except OSError as error:
            raise CleanupError(reason) from error
        if wait():
            if interrupted:
                raise CleanupError("cleanup_interrupted")
            return mode
    raise CleanupError("reap_failed")


class TimingGuard:
    def __init__(
        self,
        policy,
        host_identity_sha256,
        *,
        previous_checkpoint=None,
        clock=time,
        platform=sys.platform,
        popen=subprocess.Popen,
        run=subprocess.run,
        sleep=time.sleep,
        getpid=os.getpid,
    ):
        check_policy(policy)
        require(
            isinstance(host_identity_sha256, str)
            and DIGEST.fullmatch(host_identity_sha256),
            "Invalid host identity hash.",
        )
        self.policy = json.loads(encoded(policy))
        self.clock, self.platform, self.popen, self.run = clock, platform, popen, run
        self.sleep, self.getpid = sleep, getpid
        self.process = None
        self.last_sample = None
        self._last_good = None
        self._phase = "session:start"
        self._closed = False
        self.evidence = {
            "schema_version": 1,
            "policy": self.policy,
            "state": "planned",
            "host_identity_sha256": host_identity_sha256,
            "context": None,
            "anchor": None,
            "previous_checkpoint": json.loads(encoded(previous_checkpoint)),
            "checkpoints": [],
            "statistics": dict.fromkeys(STATS_KEYS, 0),
            "assertion": {
                "pid": None,
                "acquired": False,
                "confirmed": False,
                "released": False,
                "observation_sha256": None,
                "cleanup_clock": None,
            },
            "first_failure": None,
            "cleanup_errors": [],
            "exit_checkpoint": None,
        }

    def _fail(self, reason, sample=None, reference=None):
        if self.evidence["first_failure"] is None:
            self.evidence["first_failure"] = {
                "reason": reason,
                "phase": self._phase,
                "sample": sample,
                "reference": reference,
            }
        self.evidence["state"] = "failed"
        raise GuardError(self.evidence["first_failure"]["reason"], sample)

    def fail(self, reason, sample=None, reference=None):
        require(reason in REASONS, "Unknown timing failure.")
        self._fail(reason, sample, reference)

    def continuous_ns(self):
        return self.clock.clock_gettime_ns(
            getattr(self.clock, POLICY["continuous_clock"])
        )

    def _sample(self):
        stats, beginning, previous = self.evidence["statistics"], None, self._last_good
        for attempt in range(POLICY["sample_attempts"]):
            try:
                c0 = self.continuous_ns()
                awake = self.clock.clock_gettime_ns(
                    getattr(self.clock, POLICY["awake_clock"])
                )
                c1 = self.continuous_ns()
            except (OSError, ValueError, AttributeError):
                self._fail("clock_unavailable")
            sample = {
                "continuous_before_ns": c0,
                "awake_ns": awake,
                "continuous_after_ns": c1,
            }
            if any(type(value) is not int or value < 0 for value in sample.values()):
                self._fail("clock_unavailable")
            if c1 < c0:
                self._fail("clock_went_backwards")
            if previous is not None and (
                c0 < previous["continuous_after_ns"] or awake < previous["awake_ns"]
            ):
                self._fail("clock_went_backwards", sample, previous)
            previous = sample
            beginning = c0 if beginning is None else beginning
            self.last_sample = sample
            stats["sample_attempts"] += 1
            stats["resamples"] += int(attempt > 0)
            stats["max_bracket_ns"] = max(stats["max_bracket_ns"], c1 - c0)
            if c1 - beginning > POLICY["sample_budget_ns"]:
                self._fail("clock_sampling_uncertain", sample)
            if c1 - c0 <= POLICY["max_bracket_ns"]:
                stats["accepted_samples"] += 1
                return sample
        self._fail("clock_sampling_uncertain", sample)

    def _observe(self, label=None):
        if label is not None:
            require(
                isinstance(label, str) and LABEL.fullmatch(label),
                "Invalid timing phase.",
            )
            self._phase = label
        sample = self._sample()
        for reference in (self.evidence["anchor"], self._last_good):
            if reference is not None:
                lower = (
                    sample["continuous_before_ns"]
                    - sample["awake_ns"]
                    - (reference["continuous_after_ns"] - reference["awake_ns"])
                )
                stats = self.evidence["statistics"]
                stats["max_gap_lower_ns"] = max(stats["max_gap_lower_ns"], lower)
                reason = interval_reason(reference, sample)
                if reason:
                    self._fail(reason, sample, reference)
        self._last_good = sample
        if label is not None:
            self.evidence["checkpoints"].append({"label": label, "sample": sample})
        return sample

    def _query(self, arguments, timeout):
        result = self.run(
            arguments,
            capture_output=True,
            timeout=timeout,
            check=False,
            env={"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"},
        )
        if (
            result.returncode != 0
            or len(result.stdout) > 1 << 20
            or len(result.stderr) > 1 << 20
        ):
            raise OSError("probe_failed")
        return result.stdout

    def start(self):
        require(
            self.evidence["state"] == "planned" and not self._closed,
            "Timing guard already started.",
        )
        if self.platform != "darwin" or not all(
            hasattr(self.clock, name)
            for name in (
                "clock_gettime_ns",
                POLICY["continuous_clock"],
                POLICY["awake_clock"],
            )
        ):
            self._fail("unsupported_clock")
        self.evidence["anchor"] = self._observe()
        try:
            raw = self._query(
                ["/usr/sbin/sysctl", "-n", "kern.bootsessionuuid"],
                POLICY["probe_seconds"],
            )
            boot = raw.decode("ascii").strip().lower()
            if not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", boot):
                raise ValueError
        except (OSError, ValueError, UnicodeError, subprocess.TimeoutExpired):
            self._fail("clock_context_unavailable")
        context = {
            "platform": "darwin",
            "host_identity_sha256": self.evidence["host_identity_sha256"],
            "boot_session_sha256": hashlib.sha256(boot.encode()).hexdigest(),
            "continuous_clock": POLICY["continuous_clock"],
            "awake_clock": POLICY["awake_clock"],
        }
        self.evidence["context"] = context
        previous = self.evidence["previous_checkpoint"]
        if previous is not None:
            try:
                require(
                    set(previous) == {"context", "sample"}
                    and encoded(previous["context"]) == encoded(context),
                    "Previous context differs.",
                )
                check_sample(previous["sample"])
                reason = interval_reason(previous["sample"], self.evidence["anchor"])
            except (ValueError, TypeError, KeyError):
                self._fail("previous_checkpoint_mismatch")
            if reason:
                self._fail(reason, self.evidence["anchor"], previous["sample"])
        try:
            self.process = self.popen(
                ["/usr/bin/caffeinate", "-i", "-w", str(self.getpid())],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env={"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"},
            )
        except OSError:
            self._fail("assertion_start_failed")
        self.evidence["assertion"].update(pid=self.process.pid, acquired=True)
        deadline = self.evidence["anchor"]["continuous_before_ns"] + int(
            POLICY["assertion_start_seconds"] * 1e9
        )
        while True:
            sample = self._observe()
            remaining = deadline - sample["continuous_after_ns"]
            if remaining <= 0 or self.process.poll() is not None:
                self._fail("assertion_start_failed", sample)
            try:
                raw = self._query(
                    ["/usr/bin/pmset", "-g", "assertions"],
                    min(POLICY["probe_seconds"], remaining / 1e9),
                )
                found = re.search(
                    rf"^\s*pid\s+{self.process.pid}\(caffeinate\):.*\bPreventUserIdleSystemSleep\b",
                    raw.decode("utf-8"),
                    re.M,
                )
            except (OSError, UnicodeError, subprocess.TimeoutExpired):
                found = None
            sample = self._observe()
            if (
                sample["continuous_after_ns"] >= deadline
                or self.process.poll() is not None
            ):
                self._fail("assertion_start_failed", sample)
            if found:
                self.evidence["assertion"].update(
                    confirmed=True, observation_sha256=hashlib.sha256(raw).hexdigest()
                )
                self.evidence["state"] = "active"
                return self.check("session:start")
            self.sleep(
                min(
                    POLICY["poll_ns"] / 1e9,
                    (deadline - sample["continuous_after_ns"]) / 1e9,
                )
            )

    def check(self, label=None):
        if self.evidence["first_failure"] is not None:
            raise GuardError(self.evidence["first_failure"]["reason"], self.last_sample)
        require(self.evidence["state"] == "active", "Timing guard is not active.")
        sample = self._observe(label)
        if self.process.poll() is not None:
            self._fail("assertion_lost", sample)
        return sample

    def idle(self, seconds, label):
        require(
            type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0,
            "Invalid idle duration.",
        )
        start = self.check(label + ":start")
        deadline = start["continuous_after_ns"] + int(seconds * 1e9)
        while True:
            sample = self.check()
            remaining = deadline - sample["continuous_after_ns"]
            if remaining <= 0:
                return self.check(label + ":end")
            self.sleep(min(POLICY["poll_ns"], remaining) / 1e9)

    def wait(self, process, start_sample, timeout_seconds, label):
        check_sample(start_sample)
        require(
            type(timeout_seconds) in (int, float)
            and math.isfinite(timeout_seconds)
            and timeout_seconds > 0,
            "Invalid process deadline.",
        )
        deadline = start_sample["continuous_before_ns"] + int(timeout_seconds * 1e9)
        while True:
            sample = self.check()
            if sample["continuous_after_ns"] >= deadline:
                self._fail("deadline_exceeded", sample, start_sample)
            code = process.poll()
            if code is not None:
                sample = self.check(label)
                if sample["continuous_after_ns"] >= deadline:
                    self._fail("deadline_exceeded", sample, start_sample)
                return code, sample
            self.sleep(
                min(POLICY["poll_ns"], deadline - sample["continuous_after_ns"]) / 1e9
            )

    def close(self):
        if self._closed:
            return (
                self.evidence["first_failure"] is None
                and not self.evidence["cleanup_errors"]
            )
        try:
            if self.evidence["state"] == "active":
                self.check("session:end")
        except GuardError:
            pass
        except BaseException as error:
            self._record_interruption(error)
        finally:
            try:
                mode = stop_process(
                    self.process,
                    clock_ns=self.continuous_ns,
                    sleep=self.sleep,
                    fallback_ns=getattr(self.clock, "monotonic_ns", time.monotonic_ns),
                    seconds=POLICY["cleanup_seconds"],
                )
                self.evidence["assertion"]["cleanup_clock"] = mode
                self.record_cleanup("assertion", mode)
            except (CleanupError, OSError) as error:
                self.record_cleanup("assertion", None, error)
            except BaseException as error:
                self._record_interruption(error)
            finally:
                self._closed = True
                self.evidence["assertion"]["released"] = (
                    self.process is not None and self.process.returncode is not None
                )
        if (
            self.evidence["first_failure"] is None
            and self.evidence["state"] == "active"
        ):
            try:
                sample = self._observe("session:closed")
                self.evidence["exit_checkpoint"] = {
                    "context": self.evidence["context"],
                    "sample": sample,
                }
                self.evidence["state"] = "closed"
            except GuardError:
                pass
            except BaseException as error:
                self._record_interruption(error)
        if self.evidence["first_failure"] is not None:
            self.evidence["state"] = "failed"
        return (
            self.evidence["first_failure"] is None
            and not self.evidence["cleanup_errors"]
        )

    def _record_interruption(self, error):
        if self.evidence["first_failure"] is None:
            self.evidence["first_failure"] = {
                "reason": "guard_interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "guard_check_failed",
                "phase": "session:cleanup",
                "sample": None,
                "reference": None,
            }
        self.evidence["state"] = "failed"

    def record_cleanup(self, target, mode, error=None):
        require(target in {"native", "assertion"}, "Invalid cleanup target.")
        if error is not None or mode == "monotonic_fallback":
            reason = (
                str(error)
                if isinstance(error, CleanupError)
                else "cleanup_failed"
                if error
                else "clock_unavailable"
            )
            self.evidence["cleanup_errors"].append({"target": target, "reason": reason})
            if self.evidence["first_failure"] is None:
                self.evidence["first_failure"] = {
                    "reason": "clock_unavailable"
                    if mode == "monotonic_fallback"
                    else target + "_cleanup_failed",
                    "phase": "session:cleanup",
                    "sample": None,
                    "reference": None,
                }
            self.evidence["state"] = "failed"


def _validate_evidence(record, policy, host_identity_sha256):
    check_policy(policy)
    require(
        type(record["schema_version"]) is int
        and record["schema_version"] == 1
        and encoded(record["policy"]) == encoded(policy),
        "Timing evidence policy differs.",
    )
    require(
        record["host_identity_sha256"] == host_identity_sha256,
        "Timing host identity differs.",
    )
    require(
        record["state"] in {"planned", "active", "closed", "failed"},
        "Invalid timing state.",
    )
    stats = record["statistics"]
    require(
        set(stats) == STATS_KEYS
        and all(type(x) is int and x >= 0 for x in stats.values()),
        "Invalid timing statistics.",
    )
    require(
        stats["accepted_samples"] <= stats["sample_attempts"]
        and stats["resamples"] <= stats["sample_attempts"] - stats["accepted_samples"],
        "Inconsistent timing counts.",
    )
    assertion = record["assertion"]
    require(
        all(
            type(assertion[key]) is bool
            for key in ("acquired", "confirmed", "released")
        ),
        "Invalid assertion state.",
    )
    require(
        not assertion["confirmed"] or assertion["acquired"],
        "Unowned assertion confirmation.",
    )
    require(
        not assertion["released"] or assertion["acquired"], "Unowned assertion release."
    )
    require(
        assertion["cleanup_clock"]
        in {None, "continuous", "monotonic_fallback", "not_needed"},
        "Invalid cleanup clock.",
    )
    if assertion["acquired"]:
        require(
            type(assertion["pid"]) is int and assertion["pid"] > 0,
            "Invalid owned assertion PID.",
        )
    if assertion["confirmed"]:
        require(
            isinstance(assertion["observation_sha256"], str)
            and DIGEST.fullmatch(assertion["observation_sha256"]),
            "Missing assertion observation.",
        )
    context = record["context"]
    if context is not None:
        require(
            set(context)
            == {
                "platform",
                "host_identity_sha256",
                "boot_session_sha256",
                "continuous_clock",
                "awake_clock",
            },
            "Invalid timing context.",
        )
        require(
            context["platform"] == "darwin"
            and context["host_identity_sha256"] == host_identity_sha256
            and context["continuous_clock"] == policy["continuous_clock"]
            and context["awake_clock"] == policy["awake_clock"]
            and isinstance(context["boot_session_sha256"], str)
            and DIGEST.fullmatch(context["boot_session_sha256"]),
            "Clock context differs.",
        )
    anchor, previous = record["anchor"], record["previous_checkpoint"]
    if anchor is not None:
        check_sample(anchor)
        require(
            anchor["continuous_after_ns"] - anchor["continuous_before_ns"]
            <= policy["max_bracket_ns"],
            "Uncertain clock anchor.",
        )
    if previous is not None and record["state"] == "closed":
        require(
            set(previous) == {"context", "sample"}
            and encoded(previous["context"]) == encoded(context),
            "Previous context differs.",
        )
        require(
            interval_reason(previous["sample"], anchor, policy) is None,
            "Previous clock checkpoint differs.",
        )
    last = None
    for checkpoint in record["checkpoints"]:
        require(
            set(checkpoint) == {"label", "sample"}
            and isinstance(checkpoint["label"], str)
            and LABEL.fullmatch(checkpoint["label"]),
            "Invalid timing checkpoint.",
        )
        sample = checkpoint["sample"]
        check_sample(sample)
        require(
            sample["continuous_after_ns"] - sample["continuous_before_ns"]
            <= policy["max_bracket_ns"],
            "Uncertain checkpoint accepted.",
        )
        if record["state"] == "closed":
            require(
                anchor is not None and interval_reason(anchor, sample, policy) is None,
                "Clock checkpoint changed.",
            )
            if last is not None:
                require(
                    interval_reason(last, sample, policy) is None,
                    "Clock checkpoints reversed.",
                )
        last = sample
    failure = record["first_failure"]
    if failure is not None:
        require(
            set(failure) == {"reason", "phase", "sample", "reference"}
            and failure["reason"] in REASONS
            and isinstance(failure["phase"], str)
            and LABEL.fullmatch(failure["phase"]),
            "Invalid timing failure.",
        )
        for key in ("sample", "reference"):
            if failure[key] is not None:
                check_sample(failure[key])
    for error in record["cleanup_errors"]:
        require(
            set(error) == {"target", "reason"}
            and error["target"] in {"assertion", "native"}
            and error["reason"]
            in {
                "terminate_failed",
                "kill_failed",
                "reap_failed",
                "cleanup_failed",
                "cleanup_interrupted",
                "clock_unavailable",
            },
            "Invalid timing cleanup evidence.",
        )
    passed = record["state"] == "closed"
    if passed:
        require(
            context is not None
            and anchor is not None
            and last is not None
            and failure is None
            and not record["cleanup_errors"]
            and all(assertion[key] for key in ("acquired", "confirmed", "released")),
            "Incomplete timing pass.",
        )
        require(
            record["checkpoints"][0]["label"] == "session:start"
            and record["checkpoints"][-1]["label"] == "session:closed",
            "Missing session clock boundaries.",
        )
        require(
            encoded(record["exit_checkpoint"])
            == encoded({"context": context, "sample": last}),
            "Exit checkpoint differs.",
        )
        widest = max(
            s["continuous_after_ns"] - s["continuous_before_ns"]
            for s in [anchor, *[c["sample"] for c in record["checkpoints"]]]
        )
        require(
            stats["accepted_samples"] > len(record["checkpoints"])
            and stats["max_gap_lower_ns"] <= policy["allowance_ns"]
            and widest <= stats["max_bracket_ns"] <= policy["sample_budget_ns"]
            and assertion["cleanup_clock"] in {"continuous", "not_needed"},
            "Timing pass contradicts statistics.",
        )
    else:
        require(
            record["exit_checkpoint"] is None,
            "Failed timing guard claims a continuation checkpoint.",
        )
    return passed


def validate_evidence(record, policy, host_identity_sha256):
    try:
        return _validate_evidence(record, policy, host_identity_sha256)
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError("Malformed timing evidence.") from error
