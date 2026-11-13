"""Fake clocks and owned children; these tests never change power settings."""

import copy
import importlib.util
import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "benchmark_timing_test", Path(__file__).parents[1] / "scripts/benchmark_timing.py"
)
timing = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(timing)
HOST = timing.host_identity_hash({"system": "Darwin", "hardware": "fixture"})
BOOT = b"12345678-1234-1234-1234-123456789abc\n"


class Clock:
    CLOCK_MONOTONIC_RAW = 4
    CLOCK_UPTIME_RAW = 8

    def __init__(self):
        self.awake = 1_000_000_000
        self.suspended = 0
        self.readings = deque()
        self.sleeps = []
        self.on_sleep = None
        self.broken = False

    def clock_gettime_ns(self, clock_id):
        if self.broken:
            raise OSError("clock unavailable")
        if self.readings:
            return self.readings.popleft()
        return self.awake + (self.suspended if clock_id == 4 else 0)

    def monotonic_ns(self):
        return self.awake

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.awake += round(seconds * 1e9)
        if self.on_sleep:
            callback, self.on_sleep = self.on_sleep, None
            callback()


class Child:
    pid = 321

    def __init__(self, *, clock=None, finish_ns=None, resist=False, ignore_kill=False):
        self.returncode = None
        self.clock, self.finish_ns = clock, finish_ns
        self.resist, self.ignore_kill = resist, ignore_kill
        self.terminated = self.killed = 0
        self.on_poll = None

    def poll(self):
        if self.on_poll:
            callback, self.on_poll = self.on_poll, None
            callback()
        if self.finish_ns is not None and self.clock.awake >= self.finish_ns:
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.terminated += 1
        if not self.resist:
            self.returncode = -15

    def kill(self):
        self.killed += 1
        if not self.ignore_kill:
            self.returncode = -9


def make_guard(
    *, previous=None, platform="darwin", boot=BOOT, confirmed=True, assertion=None
):
    clock = Clock()
    assertion = assertion or Child()
    calls = []

    def popen(argv, **kwargs):
        calls.append(("popen", argv, kwargs))
        return assertion

    def run(argv, **kwargs):
        calls.append(("run", argv, kwargs))
        output = (
            boot
            if argv[0] == "/usr/sbin/sysctl"
            else (
                b"pid 321(caffeinate): [0x1] 00:00:00 "
                b'PreventUserIdleSystemSleep named: "fixture"\n'
                if confirmed
                else b"pid 3210(caffeinate): PreventUserIdleSystemSleep\n"
            )
        )
        return SimpleNamespace(returncode=0, stdout=output, stderr=b"")

    guard = timing.TimingGuard(
        timing.POLICY,
        HOST,
        previous_checkpoint=previous,
        clock=clock,
        platform=platform,
        popen=popen,
        run=run,
        sleep=clock.sleep,
        getpid=lambda: 765,
    )
    return guard, clock, assertion, calls


def completed_record():
    guard, clock, _, _ = make_guard()
    guard.start()
    start = guard.check("s1-r1-pp512-cpu:start")
    child = Child(clock=clock, finish_ns=clock.awake + 200_000_000)
    code, _ = guard.wait(child, start, 1, "s1-r1-pp512-cpu:end")
    assert code == 0
    guard.close()
    assert timing.validate_evidence(guard.evidence, timing.POLICY, HOST)
    return guard.evidence


def test_normal_owned_assertion_and_compact_wait():
    guard, clock, assertion, calls = make_guard()
    guard.start()
    guard.idle(2, "initial-idle")
    start = guard.check("row:start")
    child = Child(clock=clock, finish_ns=clock.awake + 300_000_000)
    code, end = guard.wait(child, start, 1, "row:end")
    assert code == 0 and end["continuous_after_ns"] > start["continuous_before_ns"]
    guard.close()
    guard.close()
    assert assertion.terminated == 1 and assertion.killed == 0
    assert all(delay <= 0.1 for delay in clock.sleeps)
    assert len(guard.evidence["checkpoints"]) == 7
    assert guard.evidence["statistics"]["accepted_samples"] > 20
    assert timing.validate_evidence(guard.evidence, timing.POLICY, HOST)
    argv = next(row[1] for row in calls if row[0] == "popen")
    assert argv == ["/usr/bin/caffeinate", "-i", "-w", "765"]
    assert "fixture" not in str(guard.evidence)


@pytest.mark.parametrize("phase", ["initial-idle", "between-pairs", "child"])
def test_suspend_stops_idle_and_child_without_resetting_anchor(phase):
    guard, clock, assertion, _ = make_guard()
    guard.start()
    anchor = copy.deepcopy(guard.evidence["anchor"])
    clock.on_sleep = lambda: setattr(clock, "suspended", 900_000_000_000)
    with pytest.raises(timing.GuardError, match="clock_discontinuity"):
        if phase == "child":
            guard.wait(Child(), guard.check("row:start"), 600, "row:end")
        else:
            guard.idle(120, phase)
    assert guard.evidence["anchor"] == anchor
    guard.close()
    assert assertion.terminated == 1 and guard.evidence["exit_checkpoint"] is None
    assert not timing.validate_evidence(guard.evidence, timing.POLICY, HOST)


def test_zero_exit_observed_after_deadline_remains_failed():
    guard, clock, _, _ = make_guard()
    guard.start()
    start = guard.check("row:start")
    child = Child(clock=clock, finish_ns=clock.awake)
    child.on_poll = lambda: setattr(clock, "awake", clock.awake + 2_000_000_000)
    with pytest.raises(timing.GuardError, match="deadline_exceeded") as error:
        guard.wait(child, start, 1, "row:end")
    assert child.returncode == 0 and error.value.sample is not None
    guard.close()
    assert guard.evidence["first_failure"]["reason"] == "deadline_exceeded"


def test_resampling_tolerates_preemption_without_resetting_anchor():
    guard, clock, _, _ = make_guard()
    guard.start()
    base = clock.awake
    anchor = copy.deepcopy(guard.evidence["anchor"])
    clock.readings.extend(
        [
            base,
            base,
            base + 2_000_000,
            base + 2_000_000,
            base + 2_000_000,
            base + 2_000_000,
        ]
    )
    clock.awake += 2_000_000
    guard.check("after-preemption")
    guard.close()
    assert guard.evidence["anchor"] == anchor
    assert guard.evidence["statistics"]["resamples"] == 1
    assert guard.evidence["statistics"]["max_bracket_ns"] == 2_000_000
    assert timing.validate_evidence(guard.evidence, timing.POLICY, HOST)


def test_twenty_ms_preemption_can_recover_within_fixed_sample_budget():
    guard, clock, _, _ = make_guard()
    guard.start()
    base = clock.awake
    clock.readings.extend(
        [
            base,
            base,
            base + 20_000_000,
            base + 20_000_000,
            base + 20_000_000,
            base + 20_000_000,
        ]
    )
    clock.awake += 20_000_000
    guard.check("recovered-read")
    guard.close()
    assert guard.evidence["statistics"]["resamples"] == 1
    assert timing.validate_evidence(guard.evidence, timing.POLICY, HOST)


@pytest.mark.parametrize(
    "mode", ["three_attempts", "budget", "backwards_continuous", "backwards_awake"]
)
def test_resampling_limits_and_rejected_backwards_reads(mode):
    guard, clock, _, _ = make_guard()
    guard.start()
    base = clock.awake
    if mode == "three_attempts":
        clock.readings.extend(
            [
                base,
                base,
                base + 2_000_000,
                base + 2_000_000,
                base + 2_000_000,
                base + 4_000_000,
                base + 4_000_000,
                base + 4_000_000,
                base + 6_000_000,
            ]
        )
    elif mode == "budget":
        clock.readings.extend([base, base, base + 101_000_000])
    else:
        clock.readings.extend([base, base, base + 2_000_000])
        clock.readings.extend(
            [base + 1, base + 1, base + 1]
            if mode == "backwards_continuous"
            else [base + 2_000_000, base - 1, base + 2_000_000]
        )
        clock.readings.extend([base + 3_000_000] * 3)
    reason = (
        "clock_went_backwards"
        if mode.startswith("backwards")
        else "clock_sampling_uncertain"
    )
    with pytest.raises(timing.GuardError, match=reason):
        guard.check("read")
    if mode.startswith("backwards"):
        assert len(clock.readings) == 3
    clock.readings.clear()
    guard.close()


@pytest.mark.parametrize("platform,boot", [("linux", BOOT), ("darwin", b"not-a-uuid")])
def test_unsupported_clock_or_context_launches_no_assertion(platform, boot):
    guard, _, _, calls = make_guard(platform=platform, boot=boot)
    with pytest.raises(timing.GuardError):
        guard.start()
    guard.close()
    assert not any(row[0] == "popen" for row in calls)
    assert not timing.validate_evidence(guard.evidence, timing.POLICY, HOST)


def test_assertion_confirmation_requires_owned_pid_and_stops_at_deadline():
    guard, clock, assertion, _ = make_guard(confirmed=False)
    with pytest.raises(timing.GuardError, match="assertion_start_failed"):
        guard.start()
    guard.close()
    assert 3_000_000_000 <= clock.awake - 1_000_000_000 < 3_100_000_000
    assert assertion.terminated == 1


def test_assertion_liveness_failure_preserves_original_reason():
    guard, _, assertion, _ = make_guard()
    guard.start()
    assertion.returncode = 7
    with pytest.raises(timing.GuardError, match="assertion_lost"):
        guard.check("row:start")
    guard.record_cleanup("native", None, timing.CleanupError("reap_failed"))
    guard.close()
    assert guard.evidence["first_failure"]["reason"] == "assertion_lost"
    assert guard.evidence["cleanup_errors"] == [
        {"target": "native", "reason": "reap_failed"}
    ]


def test_matching_prior_session_checkpoint_and_boot_identity():
    previous = completed_record()["exit_checkpoint"]
    guard, clock, _, _ = make_guard(previous=previous)
    clock.awake = previous["sample"]["awake_ns"] + 1_000_000
    guard.start()
    guard.close()
    assert timing.validate_evidence(guard.evidence, timing.POLICY, HOST)
    other, _, _, calls = make_guard(
        previous=previous, boot=b"87654321-1234-1234-1234-123456789abc\n"
    )
    with pytest.raises(timing.GuardError, match="previous_checkpoint_mismatch"):
        other.start()
    other.close()
    assert not any(row[0] == "popen" for row in calls)


def test_between_session_suspend_blocks_before_assertion():
    previous = completed_record()["exit_checkpoint"]
    guard, clock, _, calls = make_guard(previous=previous)
    clock.awake = previous["sample"]["awake_ns"] + 1_000_000
    clock.suspended = 1_000_000_000
    with pytest.raises(timing.GuardError, match="clock_discontinuity"):
        guard.start()
    guard.close()
    assert not any(row[0] == "popen" for row in calls)


def test_cleanup_kills_term_resistant_owned_child_and_reaps():
    clock, child = Clock(), Child(resist=True)
    mode = timing.stop_process(
        child, clock_ns=clock.monotonic_ns, sleep=clock.sleep, seconds=0.2
    )
    assert mode == "continuous" and child.terminated == child.killed == 1
    assert child.poll() == -9


def test_cleanup_interrupt_does_not_abandon_owned_child():
    clock, child = Clock(), Child(resist=True)

    def interrupt():
        raise KeyboardInterrupt

    clock.on_sleep = interrupt
    with pytest.raises(timing.CleanupError, match="cleanup_interrupted"):
        timing.stop_process(
            child, clock_ns=clock.monotonic_ns, sleep=clock.sleep, seconds=0.2
        )
    assert child.returncode == -9 and child.killed == 1


@pytest.mark.parametrize(
    "point",
    ["poll_start", "poll_loop", "clock_start", "clock_loop", "terminate", "kill"],
)
def test_single_interrupt_at_each_cleanup_operation_still_reaps(point):
    clock, child = Clock(), Child(resist=True)

    def once(function, number):
        calls = 0

        def wrapped(*args):
            nonlocal calls
            calls += 1
            if calls == number:
                raise KeyboardInterrupt
            return function(*args)

        return wrapped

    clock_fn = clock.monotonic_ns
    if point.startswith("poll"):
        child.poll = once(child.poll, 1 if point.endswith("start") else 2)
    elif point.startswith("clock"):
        clock_fn = once(clock_fn, 1 if point.endswith("start") else 3)
    else:
        setattr(child, point, once(getattr(child, point), 1))
    with pytest.raises(timing.CleanupError, match="cleanup_interrupted"):
        timing.stop_process(child, clock_ns=clock_fn, sleep=clock.sleep, seconds=0.2)
    assert child.returncode == -9


@pytest.mark.parametrize("failure", [KeyboardInterrupt, RuntimeError])
def test_interrupted_final_check_always_releases_assertion(failure):
    guard, _, assertion, _ = make_guard()
    guard.start()

    def interrupted(label=None):
        raise failure

    guard.check = interrupted
    assert guard.close() is False
    assert assertion.terminated == 1 and assertion.returncode is not None
    assert guard.evidence["assertion"]["released"] is True
    assert guard.evidence["first_failure"]["reason"] == (
        "guard_interrupted" if failure is KeyboardInterrupt else "guard_check_failed"
    )
    assert not timing.validate_evidence(guard.evidence, timing.POLICY, HOST)
    assert guard.close() is False and assertion.terminated == 1


def test_cleanup_interrupt_keeps_the_original_clock_failure():
    assertion = Child(resist=True)
    guard, clock, _, _ = make_guard(assertion=assertion)
    guard.start()
    clock.suspended = 2_000_000
    with pytest.raises(timing.GuardError, match="clock_discontinuity"):
        guard.check("native")

    def interrupt():
        raise KeyboardInterrupt

    clock.on_sleep = interrupt
    assert guard.close() is False
    assert assertion.returncode == -9
    assert guard.evidence["first_failure"]["reason"] == "clock_discontinuity"
    assert guard.evidence["cleanup_errors"] == [
        {"target": "assertion", "reason": "cleanup_interrupted"}
    ]


def test_failed_clock_cleanup_fallback_never_becomes_timing_pass():
    guard, clock, assertion, _ = make_guard()
    guard.start()
    clock.broken = True
    with pytest.raises(timing.GuardError, match="clock_unavailable"):
        guard.check()
    guard.close()
    assert assertion.returncode is not None
    assert guard.evidence["assertion"]["cleanup_clock"] == "monotonic_fallback"
    assert not timing.validate_evidence(guard.evidence, timing.POLICY, HOST)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(schema_version=True),
        lambda r: r["anchor"].update(
            continuous_after_ns=r["anchor"]["continuous_before_ns"] + 2_000_000
        ),
        lambda r: r["anchor"].pop("awake_ns"),
        lambda r: r.update(checkpoints=[]),
        lambda r: r["statistics"].update(accepted_samples=0),
        lambda r: r["statistics"].update(max_gap_lower_ns=2_000_000),
        lambda r: r["statistics"].update(
            max_bracket_ns=timing.POLICY["sample_budget_ns"] + 1
        ),
        lambda r: r["assertion"].update(confirmed=False),
        lambda r: r["assertion"].update(cleanup_clock=None),
        lambda r: r["assertion"].update(cleanup_clock="monotonic_fallback"),
        lambda r: r["exit_checkpoint"]["context"].update(boot_session_sha256="0" * 64),
    ],
)
def test_completed_evidence_rejects_malformed_or_unproved_pass(mutation):
    record = json.loads(json.dumps(completed_record()))
    mutation(record)
    with pytest.raises(ValueError):
        timing.validate_evidence(record, timing.POLICY, HOST)
