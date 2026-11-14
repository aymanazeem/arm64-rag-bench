"""Fake ownership checks and a separately enabled macOS process-group smoke."""

import importlib.util
import json
import os
import signal
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "answer_owner_test", Path(__file__).parents[1] / "scripts/answer_benchmark_owner.py"
)
owner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(owner)


class Clock:
    def __init__(self):
        self.now = 5_000_000_000
        self.fallback = 1_000_000
        self.mode = "normal"
        self.sleeps = []
        self.on_sleep = None

    def read(self):
        if self.mode == "unavailable":
            raise AttributeError("Missing clock")
        if self.mode == "reverse":
            self.now -= 1
        return self.now

    def monotonic_ns(self):
        return self.fallback

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.fallback += round(seconds * 1e9)
        if self.mode == "normal":
            self.now += round(seconds * 1e9)
        if self.on_sleep:
            callback, self.on_sleep = self.on_sleep, None
            callback()


class Pipe:
    def __init__(self, fd):
        self.fd = fd
        self.closed = False

    def fileno(self):
        return self.fd

    def close(self):
        self.closed = True


class Child:
    pid = 321

    def __init__(self):
        self.stdout, self.stderr = Pipe(200), Pipe(201)
        self.returncode = None

    def forbidden(self, *args, **kwargs):
        raise AssertionError("Popen must not observe or reap this worker")

    poll = wait = send_signal = terminate = kill = forbidden


class System:
    def __init__(self, clock):
        self.clock = clock
        self.calls = []
        self.child = Child()
        self.group_alive = True
        self.worker_exited = False
        self.term_exits = self.kill_exits = True
        self.reaped = False
        self.reap_polls = 0
        self.reads = {200: deque(), 201: deque()}
        self.on_write = None
        self.on_popen = None
        self.denied = False
        self.reap_pending = 0
        self.nonblocking = set()
        self.blocking_reads = []

    def __getattr__(self, key):
        return getattr(os, key)

    def pipe(self):
        return 100, 101

    def close(self, fd):
        self.calls.append(("close", fd))

    def getpgid(self, pid):
        assert pid == self.child.pid
        return pid

    def getpgrp(self):
        return 999

    def set_blocking(self, fd, mode):
        self.calls.append(("nonblocking", fd, mode))
        assert mode is False
        self.nonblocking.add(fd)

    def popen(self, argv, **kwargs):
        self.calls.append(("spawn", argv, kwargs))
        if self.on_popen:
            self.on_popen()
        return self.child

    def write(self, fd, data):
        self.calls.append(("release", fd, data))
        if self.on_write:
            self.on_write()
        return len(data)

    def read(self, fd, size):
        if fd not in self.nonblocking:
            self.blocking_reads.append(fd)
            raise AssertionError("A worker pipe must be nonblocking before every read")
        self.calls.append(("read", fd))
        if self.reads[fd]:
            return self.reads[fd].popleft()
        if not self.group_alive:
            return b""
        raise BlockingIOError()

    def fsync(self, fd):
        self.calls.append(("fsync", fd))

    def killpg(self, pgid, number):
        assert pgid == self.child.pid
        assert not self.reaped, "Never signal a reaped/reusable group ID"
        self.calls.append(("group_signal", number))
        if self.denied:
            raise PermissionError("Group permission denied")
        if not self.group_alive:
            raise ProcessLookupError()
        if number == signal.SIGTERM and self.term_exits:
            self.group_alive = False
            self.worker_exited = True
        if number == signal.SIGKILL and self.kill_exits:
            self.group_alive = False
            self.worker_exited = True

    def listpids(self, kind, pgid, buffer, size):
        assert kind == 2 and pgid == self.child.pid and size == 8
        assert not self.reaped
        self.calls.append(("membership", pgid))
        buffer[0] = self.child.pid
        if self.group_alive and self.worker_exited:
            buffer[1] = self.child.pid + 1
            return 8
        return 4

    def kill(self, pid, number):
        assert pid == self.child.pid and not self.reaped
        self.calls.append(("direct_signal", number))
        self.group_alive = False
        self.worker_exited = True

    def wait4(self, pid, options):
        assert not self.group_alive, "Group cleanup must precede the only reaper"
        assert pid == self.child.pid and options == os.WNOHANG
        assert not self.reaped
        self.calls.append(("wait4", pid))
        self.reap_polls += 1
        if self.reap_pending:
            self.reap_pending -= 1
            return 0, 0, None
        self.reaped = True
        usage = SimpleNamespace(ru_maxrss=123456, ru_utime=1.25, ru_stime=0.5)
        return pid, 0, usage


class Queue:
    def __init__(self, system):
        self.system = system
        self.registered = False
        self.registration_events = []
        self.next_events = deque()
        self.closed = False
        self.on_register = None

    def control(self, changes, maximum, timeout):
        assert maximum == 1 and timeout == 0
        if changes:
            self.system.calls.append(("register", changes[0]))
            self.registered = True
            if self.on_register:
                self.on_register()
            return self.registration_events
        assert self.registered
        if self.next_events:
            return self.next_events.popleft()
        if self.system.worker_exited:
            return [event()]
        return []

    def close(self):
        self.closed = True


def event(**changes):
    return SimpleNamespace(
        **({"ident": 321, "filter": -5, "flags": 0, "fflags": 8} | changes)
    )


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    clock = Clock()
    system = System(clock)
    queue = Queue(system)
    handlers = {
        signal.SIGINT: object(),
        signal.SIGTERM: object(),
        signal.SIGCHLD: signal.SIG_DFL,
    }
    original_handlers = dict(handlers)

    def signal_handler(number, handler):
        old = handlers[number]
        handlers[number] = handler
        return old

    fake_select = SimpleNamespace(
        kqueue=lambda: queue,
        kevent=lambda ident, **kw: SimpleNamespace(ident=ident, **kw),
        KQ_FILTER_PROC=-5,
        KQ_NOTE_EXIT=8,
        KQ_EV_ADD=1,
        KQ_EV_ONESHOT=16,
        KQ_EV_ERROR=16384,
    )
    monkeypatch.setattr(owner, "os", system)
    monkeypatch.setattr(owner, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(owner, "select", fake_select)
    monkeypatch.setattr(owner, "time", clock)
    monkeypatch.setattr(
        owner,
        "signal",
        SimpleNamespace(
            SIGINT=signal.SIGINT,
            SIGTERM=signal.SIGTERM,
            SIGKILL=signal.SIGKILL,
            SIGCHLD=signal.SIGCHLD,
            SIG_DFL=signal.SIG_DFL,
            Signals=signal.Signals,
            getsignal=lambda number: handlers[number],
            signal=signal_handler,
        ),
    )
    main = object()
    monkeypatch.setattr(
        owner,
        "threading",
        SimpleNamespace(
            active_count=lambda: 1,
            current_thread=lambda: main,
            main_thread=lambda: main,
        ),
    )
    monkeypatch.setattr(
        owner,
        "subprocess",
        SimpleNamespace(
            Popen=system.popen,
            DEVNULL=-3,
            PIPE=-1,
        ),
    )
    monkeypatch.setattr(owner, "_HELD", {})
    monkeypatch.setattr(owner, "_load_group_reader", lambda: system.listpids)

    def create(**kwargs):
        return owner.OwnedWorker(
            ["fixture-python", "worker.py"],
            cwd=tmp_path,
            stdout_path=tmp_path / "stdout",
            stderr_path=tmp_path / "stderr",
            clock_ns=clock.read,
            cleanup_seconds=kwargs.pop("cleanup_seconds", 0.03),
            **kwargs,
        )

    yield SimpleNamespace(
        create=create,
        system=system,
        clock=clock,
        queue=queue,
        handlers=handlers,
        original_handlers=original_handlers,
        tmp=tmp_path,
    )
    # Unresolved fake children own no real resources.
    for held in owner._HELD.values():
        held._close_capture()
    owner._HELD.clear()


def test_registration_precedes_release_and_only_wait4_reaps(fixture):
    worker = fixture.create(env={"EXAMPLE": "value"})
    system = fixture.system
    spawned = next(row for row in system.calls if row[0] == "spawn")
    assert spawned[2]["start_new_session"] is True
    assert spawned[2]["pass_fds"] == (100,)
    assert spawned[2]["env"] == {"EXAMPLE": "value", owner.GATE_ENV: "100"}
    assert not any(row[0] in ("release", "wait4") for row in system.calls)
    worker.release()
    assert [row[0] for row in system.calls].index("register") < [
        row[0] for row in system.calls
    ].index("release")
    assert not worker.exited()
    system.worker_exited = True
    system.group_alive = False
    assert worker.exited() and not system.reaped
    result = worker.finish()
    assert result["returncode"] == 0
    assert result["cleanup_verified"] and result["worker_reaped"]
    assert result["wait4_peak_rss_bytes"] == 123456
    assert result["wait4_user_seconds"] == 1.25
    assert result["first_failure"] is None
    assert system.reap_polls == 1 and not owner._HELD
    calls = list(system.calls)
    assert worker.finish() == result and system.calls == calls
    assert fixture.handlers == fixture.original_handlers


def test_exited_leader_stays_reserved_until_surviving_child_is_killed(fixture):
    worker = fixture.create()
    worker.release()
    fixture.system.worker_exited = True
    fixture.system.term_exits = False
    assert worker.exited()
    result = worker.finish()
    assert result["cleanup_verified"]
    assert result["first_failure"]["reason"] == "group_present_after_worker_exit"
    actions = fixture.system.calls
    kill_index = actions.index(("group_signal", signal.SIGKILL))
    reap_index = actions.index(("wait4", worker.pid))
    assert kill_index < reap_index
    assert result["cleanup_signals"] == ["SIGTERM", "SIGKILL"]


@pytest.mark.parametrize("kind", ["denied", "resistant"])
def test_unproved_group_is_retained_and_blocks_next_worker(fixture, kind):
    worker = fixture.create()
    worker.release()
    if kind == "denied":
        fixture.system.denied = True
    else:
        fixture.system.term_exits = fixture.system.kill_exits = False
    result = worker.finish(terminate=True)
    assert not result["cleanup_verified"] and not result["worker_reaped"]
    assert fixture.system.reap_polls == 0 and owner._HELD[worker.pid] is worker
    with pytest.raises(owner.OwnerError, match="previous_worker_unresolved"):
        fixture.create()


@pytest.mark.parametrize(
    "bad_event,reason",
    [
        (event(flags=16384), "process_event_error"),
        (event(ident=999), "unexpected_process_event"),
        (event(filter=999), "unexpected_process_event"),
        (event(fflags=0), "unexpected_process_event"),
        (event(), "worker_exited_before_release"),
    ],
)
def test_registration_failure_never_opens_gate(fixture, bad_event, reason):
    fixture.queue.registration_events = [bad_event]
    with pytest.raises(owner.OwnerError, match=reason) as caught:
        fixture.create()
    assert caught.value.record["cleanup_verified"]
    assert not any(row[0] == "release" for row in fixture.system.calls)
    assert fixture.system.reaped and fixture.handlers == fixture.original_handlers


def test_registration_oserror_cleans_started_child(fixture):
    def fail():
        raise OSError("No event observer")

    fixture.queue.on_register = fail
    with pytest.raises(owner.OwnerError, match="worker_start_failed") as caught:
        fixture.create()
    assert caught.value.record["cleanup_verified"]
    assert not caught.value.record["released"]


def test_group_discovery_failure_uses_only_unreleased_child_pid(fixture, monkeypatch):
    def fail(pid):
        raise ProcessLookupError()

    monkeypatch.setattr(fixture.system, "getpgid", fail)
    with pytest.raises(owner.OwnerError) as caught:
        fixture.create()
    assert caught.value.record["cleanup_verified"]
    assert ("direct_signal", signal.SIGKILL) in fixture.system.calls
    assert not any(row[0] == "group_signal" for row in fixture.system.calls)


@pytest.mark.parametrize("phase", ["read_gate", "group_discovery"])
def test_early_startup_cleanup_configures_pipes_before_read(
    fixture, monkeypatch, phase
):
    fixture.system.reap_pending = 2
    if phase == "read_gate":
        original = fixture.system.close
        failed = []

        def close(fd):
            if fd == 100 and not failed:
                failed.append(True)
                raise OSError("Initial gate close failed")
            original(fd)

        monkeypatch.setattr(fixture.system, "close", close)
    else:

        def getpgid(pid):
            raise OSError("Group lookup failed")

        monkeypatch.setattr(fixture.system, "getpgid", getpgid)
    with pytest.raises(owner.OwnerError, match="worker_start_failed") as caught:
        fixture.create()
    assert caught.value.record["cleanup_verified"]
    assert not fixture.system.blocking_reads
    assert fixture.system.nonblocking == {200, 201}
    assert fixture.system.reap_polls == 3


@pytest.mark.parametrize("failed_fd", [200, 201])
def test_nonblocking_setup_failure_never_reads_blocking_pipe(
    fixture, monkeypatch, failed_fd
):
    original = fixture.system.set_blocking

    def set_blocking(fd, mode):
        if fd == failed_fd:
            raise OSError("Cannot configure pipe")
        original(fd, mode)

    monkeypatch.setattr(fixture.system, "set_blocking", set_blocking)
    with pytest.raises(owner.OwnerError, match="pipe_configuration_failed") as caught:
        fixture.create()
    assert caught.value.record["worker_reaped"]
    assert not caught.value.record["cleanup_verified"]
    assert not fixture.system.blocking_reads
    assert ("read", failed_fd) not in fixture.system.calls
    assert fixture.system.child.stdout.closed and fixture.system.child.stderr.closed
    assert caught.value.record["first_failure"]["reason"] == "pipe_configuration_failed"


@pytest.mark.parametrize("phase", ["registration", "release", "cleanup"])
def test_deferred_signals_cleanup_restore_handlers_and_remain_visible(fixture, phase):
    def cancel():
        fixture.handlers[signal.SIGTERM](signal.SIGTERM, None)

    if phase == "registration":
        fixture.queue.on_register = cancel
        with pytest.raises(
            owner.OwnerError, match="cancelled_before_release"
        ) as caught:
            fixture.create()
        result = caught.value.record
    else:
        worker = fixture.create()
        if phase == "release":
            fixture.system.on_write = cancel
            with pytest.raises(
                owner.OwnerError, match="cancelled_during_release"
            ) as caught:
                worker.release()
            result = caught.value.record
        else:
            worker.release()
            fixture.system.term_exits = False
            fixture.clock.on_sleep = cancel
            result = worker.finish(terminate=True)
    assert result["cleanup_verified"] and result["deferred_signal"] == signal.SIGTERM
    assert result["first_failure"] is not None
    assert fixture.handlers == fixture.original_handlers


def test_release_failure_cleans_and_preserves_reason(fixture):
    worker = fixture.create()

    def fail():
        raise BrokenPipeError()

    fixture.system.on_write = fail
    with pytest.raises(owner.OwnerError, match="worker_release_failed") as caught:
        worker.release()
    assert caught.value.record["cleanup_verified"]


@pytest.mark.parametrize("mode", ["stall", "reverse", "unavailable"])
def test_bad_cleanup_clock_uses_bounded_fallback_and_invalidates(fixture, mode):
    worker = fixture.create()
    worker.release()
    fixture.system.term_exits = False
    fixture.clock.mode = mode
    result = worker.finish(terminate=True)
    assert result["cleanup_verified"] and result["worker_reaped"]
    assert result["cleanup_clock"] == "monotonic_fallback"
    assert result["first_failure"] is not None
    assert len(fixture.clock.sleeps) < 10


def test_one_cleanup_interrupt_does_not_skip_group_cleanup(fixture, monkeypatch):
    worker = fixture.create()
    worker.release()
    original = fixture.system.killpg
    once = [True]

    def interrupted(pgid, number):
        if once:
            once.pop()
            raise KeyboardInterrupt()
        return original(pgid, number)

    monkeypatch.setattr(fixture.system, "killpg", interrupted)
    result = worker.finish(terminate=True)
    assert result["cleanup_verified"]
    assert result["first_failure"]["reason"] == "cleanup_interrupted"


def test_stdout_receipt_precedes_write_and_stderr_is_captured(fixture):
    worker = fixture.create()
    worker.release()
    first = fixture.clock.now
    output = worker._outputs["stdout"]

    class DelayedWrite:
        def __getattr__(self, key):
            return getattr(output, key)

        def write(self, data):
            fixture.clock.now += 1_000_000_000
            return output.write(data)

    worker._outputs["stdout"] = DelayedWrite()
    fixture.system.reads[200].append(b"answer\n")
    fixture.system.reads[201].append(b"notice\n")
    receipts = worker.poll_stdout()
    assert receipts == [
        {"stream": "stdout", "received_ns": first, "bytes": 7},
        {"stream": "stderr", "received_ns": fixture.clock.now, "bytes": 7},
    ]
    assert worker.first_stdout_ns == first
    fixture.system.group_alive = False
    fixture.system.worker_exited = True
    assert worker.exited()
    result = worker.finish()
    assert result["bytes_retained"] == {"stdout": 7, "stderr": 7}
    assert (fixture.tmp / "stdout").read_bytes() == b"answer\n"
    assert (fixture.tmp / "stderr").read_bytes() == b"notice\n"


def test_shared_capture_cap_keeps_prefix_and_cleans(fixture):
    worker = fixture.create(output_limit=5)
    worker.release()
    fixture.system.reads[200].append(b"1234")
    fixture.system.reads[201].append(b"abcd")
    with pytest.raises(owner.OwnerError, match="capture_limit_exceeded"):
        worker.poll_stdout()
    result = worker.finish(terminate=True)
    assert result["cleanup_verified"] and result["output_incomplete"]
    assert result["bytes_observed"] == {"stdout": 4, "stderr": 4}
    assert (fixture.tmp / "stdout").read_bytes() == b"1234"
    assert (fixture.tmp / "stderr").read_bytes() == b"a"
    assert result["first_failure"]["reason"] == "capture_limit_exceeded"


@pytest.mark.parametrize("name,fd", [("stdout", 200), ("stderr", 201)])
def test_receipt_clock_recovery_never_replaces_missing_first_timestamp(
    fixture, name, fd
):
    worker = fixture.create()
    worker.release()
    fixture.clock.mode = "unavailable"
    fixture.system.reads[fd].append(b"first")
    with pytest.raises(owner.OwnerError, match="receipt_clock_failed"):
        worker.poll_stdout()
    fixture.clock.mode = "normal"
    fixture.system.reads[fd].append(b"later")
    result = worker.finish(terminate=True)
    assert result["first_" + name + "_ns"] is None and result["cleanup_verified"]
    assert worker.first_stdout_ns is None
    assert result["bytes_observed"][name] == 10
    assert (fixture.tmp / name).read_bytes() == b"firstlater"


def test_fsync_failure_preserves_first_capture_failure(fixture, monkeypatch):
    worker = fixture.create(output_limit=1)
    worker.release()
    fixture.system.reads[200].append(b"too large")
    with pytest.raises(owner.OwnerError, match="capture_limit_exceeded"):
        worker.poll_stdout()

    def fail(fd):
        raise OSError("Disk failure")

    monkeypatch.setattr(fixture.system, "fsync", fail)
    result = worker.finish(terminate=True)
    assert result["cleanup_verified"] and result["output_incomplete"]
    assert result["first_failure"]["reason"] == "capture_limit_exceeded"
    assert any(
        row["reason"] == "capture_flush_failed" for row in result["cleanup_failures"]
    )


@pytest.mark.parametrize("bound", [True, 0, -1, float("nan"), float("inf"), 61, "5"])
def test_invalid_cleanup_bound_fails_before_spawn(fixture, bound):
    with pytest.raises(owner.OwnerError, match="invalid_bounds"):
        fixture.create(cleanup_seconds=bound)
    assert not fixture.system.calls


def test_unsupported_platform_fails_before_accessing_kqueue(fixture, monkeypatch):
    monkeypatch.setattr(owner, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(owner, "select", SimpleNamespace())
    with pytest.raises(owner.OwnerError, match="unsupported_platform"):
        fixture.create()
    assert not fixture.system.calls


def test_existing_output_is_not_overwritten(fixture):
    (fixture.tmp / "stdout").write_bytes(b"previous")
    with pytest.raises(owner.OwnerError, match="worker_start_failed") as caught:
        fixture.create()
    assert caught.value.record["cleanup_verified"]
    assert (fixture.tmp / "stdout").read_bytes() == b"previous"
    assert not fixture.system.calls


def test_other_reaper_is_rejected_before_spawn(fixture):
    fixture.handlers[signal.SIGCHLD] = object()
    with pytest.raises(owner.OwnerError, match="exclusive_reaping_required"):
        fixture.create()
    assert not fixture.system.calls


def test_delayed_wait4_is_bounded_without_signalling_reusable_pid(fixture):
    worker = fixture.create()
    worker.release()
    fixture.system.reap_pending = 2
    fixture.system.group_alive = False
    fixture.system.worker_exited = True
    assert worker.exited()
    result = worker.finish()
    assert result["cleanup_verified"] and fixture.system.reap_polls == 3
    assert not any(
        row == ("group_signal", signal.SIGTERM) for row in fixture.system.calls
    )


def test_child_start_error_preserves_empty_capture_and_closes_gate(fixture):
    def fail():
        raise FileNotFoundError()

    fixture.system.on_popen = fail
    with pytest.raises(owner.OwnerError, match="worker_start_failed") as caught:
        fixture.create()
    assert caught.value.record["cleanup_verified"]
    assert caught.value.record["pid"] is None
    assert (fixture.tmp / "stdout").read_bytes() == b""
    assert (fixture.tmp / "stderr").read_bytes() == b""
    assert ("close", 101) in fixture.system.calls
    assert ("close", 100) in fixture.system.calls


def test_secondary_gate_close_failure_does_not_skip_child_cleanup(fixture, monkeypatch):
    worker = fixture.create()
    original = fixture.system.close

    def fail(descriptor):
        if descriptor == 101:
            raise OSError("Gate close failed")
        original(descriptor)

    monkeypatch.setattr(fixture.system, "close", fail)
    result = worker.finish(terminate=True)
    assert not result["cleanup_verified"] and result["worker_reaped"]
    assert result["first_failure"]["reason"] == "gate_close_failed"
    assert worker.pid in owner._HELD
    before = [
        row for row in fixture.system.calls if row[0] in ("group_signal", "wait4")
    ]
    monkeypatch.setattr(fixture.system, "close", original)
    assert worker.finish(terminate=True)["cleanup_verified"]
    assert before == [
        row for row in fixture.system.calls if row[0] in ("group_signal", "wait4")
    ]


def test_failed_eof_pipe_close_is_retained_and_retried(fixture, monkeypatch):
    worker = fixture.create()
    worker.release()
    fixture.system.group_alive = False
    fixture.system.worker_exited = True
    pipe = fixture.system.child.stdout
    original = pipe.close
    attempts = []

    def fail_once():
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError("Pipe still open")
        original()

    monkeypatch.setattr(pipe, "close", fail_once)
    with pytest.raises(owner.OwnerError, match="pipe_close_failed"):
        worker.poll_stdout()
    assert worker._pipes["stdout"] is pipe and not pipe.closed
    assert worker.exited()
    result = worker.finish()
    assert result["cleanup_verified"] and pipe.closed
    assert len(attempts) == 2 and fixture.system.reap_polls == 1
    assert result["first_failure"]["reason"] == "pipe_close_failed"


@pytest.mark.parametrize("target", ["capture", "queue"])
def test_failed_close_blocks_next_worker_and_retry_never_resignals(fixture, target):
    worker = fixture.create()
    worker.release()
    failure = [True]
    if target == "capture":
        stream = worker._outputs["stdout"]

        class FailingClose:
            def __getattr__(self, key):
                return getattr(stream, key)

            def close(self):
                if failure[0]:
                    raise OSError("Capture still open")
                stream.close()

        worker._outputs["stdout"] = FailingClose()
    else:
        original_close = fixture.queue.close

        def close_queue():
            if failure[0]:
                raise OSError("Observer still open")
            original_close()

        fixture.queue.close = close_queue
    fixture.system.worker_exited = True
    fixture.system.group_alive = False
    assert worker.exited()
    result = worker.finish()
    assert result["worker_reaped"] and not result["cleanup_verified"]
    assert worker.pid in owner._HELD
    with pytest.raises(owner.OwnerError, match="previous_worker_unresolved"):
        fixture.create()
    actions = [
        row for row in fixture.system.calls if row[0] in ("group_signal", "wait4")
    ]
    failure[0] = False
    recovered = worker.finish(terminate=True)
    assert recovered["cleanup_verified"] and recovered["first_failure"]
    assert worker.pid not in owner._HELD
    assert actions == [
        row for row in fixture.system.calls if row[0] in ("group_signal", "wait4")
    ]


def test_open_pipe_after_group_exit_remains_unverified_after_close(
    fixture, monkeypatch
):
    worker = fixture.create()
    worker.release()
    fixture.system.worker_exited = True
    fixture.system.group_alive = False
    assert worker.exited()
    original_read = fixture.system.read

    def still_open(fd, size):
        if fd == 200:
            raise BlockingIOError()
        return original_read(fd, size)

    monkeypatch.setattr(fixture.system, "read", still_open)
    result = worker.finish()
    assert result["worker_reaped"] and not result["cleanup_verified"]
    assert result["first_failure"]["reason"] == "pipes_open_after_group_exit"
    assert fixture.system.child.stdout.closed
    assert worker.pid in owner._HELD
    actions = [
        row for row in fixture.system.calls if row[0] in ("group_signal", "wait4")
    ]
    assert not worker.finish(terminate=True)["cleanup_verified"]
    assert actions == [
        row for row in fixture.system.calls if row[0] in ("group_signal", "wait4")
    ]
    with pytest.raises(owner.OwnerError, match="previous_worker_unresolved"):
        fixture.create()


def test_deferred_signal_during_capture_closure_remains_visible(fixture, monkeypatch):
    worker = fixture.create()
    worker.release()
    fixture.system.worker_exited = True
    fixture.system.group_alive = False
    assert worker.exited()
    original_fsync = fixture.system.fsync

    def cancel(fd):
        fixture.handlers[signal.SIGTERM](signal.SIGTERM, None)
        original_fsync(fd)

    monkeypatch.setattr(fixture.system, "fsync", cancel)
    result = worker.finish()
    assert result["cleanup_verified"]
    assert result["first_failure"]["reason"] == "termination_deferred"
    assert result["deferred_signal"] == signal.SIGTERM
    assert fixture.handlers == fixture.original_handlers


def test_external_reap_loses_ownership_without_later_signalling(fixture, monkeypatch):
    worker = fixture.create()
    worker.release()
    fixture.system.group_alive = False
    fixture.system.worker_exited = True
    assert worker.exited()

    def stolen(pid, flags):
        raise ChildProcessError()

    monkeypatch.setattr(fixture.system, "wait4", stolen)
    result = worker.finish()
    assert not result["cleanup_verified"]
    assert result["first_failure"]["reason"] == "worker_reaping_ownership_lost"
    calls = list(fixture.system.calls)
    assert worker.finish(terminate=True) == result
    assert fixture.system.calls == calls
    assert worker.pid in owner._HELD


def test_membership_api_is_loaded_before_any_child(fixture, monkeypatch):
    def unavailable():
        raise OSError("No process-group API")

    monkeypatch.setattr(owner, "_load_group_reader", unavailable)
    with pytest.raises(owner.OwnerError, match="worker_start_failed") as caught:
        fixture.create()
    assert caught.value.record["cleanup_verified"]
    assert not fixture.system.calls


def test_membership_reader_has_explicit_portable_abi(monkeypatch):
    reader = SimpleNamespace()
    calls = []

    def library(path, **kwargs):
        calls.append((path, kwargs))
        return SimpleNamespace(proc_listpids=reader)

    monkeypatch.setattr(owner.ctypes, "CDLL", library)
    assert owner._load_group_reader() is reader
    assert calls == [("/usr/lib/libproc.dylib", {"use_errno": True})]
    assert reader.argtypes == (
        owner.ctypes.c_uint32,
        owner.ctypes.c_uint32,
        owner.ctypes.c_void_p,
        owner.ctypes.c_int,
    )
    assert reader.restype == owner.ctypes.c_int


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "denied",
        "partial",
        "oversize",
        "zero_pid",
        "duplicate",
        "missing_leader",
    ],
)
def test_bad_membership_never_reaps_or_signals(fixture, kind):
    worker = fixture.create()
    worker.release()
    fixture.system.worker_exited = True

    def read(kind_, pgid, buffer, size):
        buffer[0] = worker.pid
        if kind == "empty":
            return 0
        if kind == "denied":
            owner.ctypes.set_errno(owner.errno.EPERM)
            return 0
        if kind == "partial":
            return 3
        if kind == "oversize":
            return 12
        if kind == "zero_pid":
            buffer[0] = 0
        if kind == "duplicate":
            buffer[1] = worker.pid
            return 8
        if kind == "missing_leader":
            buffer[0] = worker.pid + 1
        return 4

    worker._list_group_pids = read
    result = worker.finish()
    assert not result["cleanup_verified"] and not result["worker_reaped"]
    assert not any(row[0] in {"wait4", "group_signal"} for row in fixture.system.calls)
    assert worker.pid in owner._HELD


def test_full_membership_buffer_never_proves_cleanup(fixture):
    worker = fixture.create()
    worker.release()
    fixture.system.worker_exited = True
    fixture.system.term_exits = fixture.system.kill_exits = False
    result = worker.finish()
    assert not result["cleanup_verified"] and result["reap_proof"] is None
    assert fixture.system.reap_polls == 0
    assert result["cleanup_signals"] == ["SIGTERM", "SIGKILL"]


def test_singleton_without_exit_event_is_not_reaped(fixture):
    worker = fixture.create()
    worker.release()
    fixture.system.term_exits = fixture.system.kill_exits = False
    result = worker.finish(terminate=True)
    assert not result["cleanup_verified"] and not result["exit_observed"]
    assert result["reap_proof"] is None and fixture.system.reap_polls == 0


def test_terminated_child_exit_event_is_refreshed_before_reap(fixture):
    worker = fixture.create()
    worker.release()
    assert not worker.exited()
    result = worker.finish(terminate=True)
    assert result["cleanup_verified"] and result["exit_observed"]
    assert result["reap_proof"] == "note_exit_and_only_unreaped_leader"
    assert result["cleanup_signals"] == ["SIGTERM"]
    assert fixture.system.reap_polls == 1


def test_zombie_only_group_needs_no_signal_probe(fixture):
    worker = fixture.create()
    worker.release()
    fixture.system.worker_exited = True
    fixture.system.group_alive = False
    fixture.system.denied = True
    assert worker.exited()
    result = worker.finish()
    assert result["cleanup_verified"] and result["first_failure"] is None
    assert result["reap_proof"] == "note_exit_and_only_unreaped_leader"
    assert not any(row[0] == "group_signal" for row in fixture.system.calls)


def test_signal_denial_requires_later_positive_membership_proof(fixture, monkeypatch):
    worker = fixture.create()
    worker.release()

    def exit_during_signal(pgid, number):
        fixture.system.calls.append(("group_signal", number))
        fixture.system.worker_exited = True
        fixture.system.group_alive = False
        raise PermissionError(owner.errno.EPERM, "Now zombie-only")

    monkeypatch.setattr(fixture.system, "killpg", exit_during_signal)
    result = worker.finish(terminate=True)
    assert result["cleanup_verified"] and result["first_failure"] is None
    assert result["cleanup_signal_errors"] == [
        {
            "signal": "SIGTERM",
            "error_type": "PermissionError",
            "resolved_by_membership": True,
        }
    ]
    assert result["reap_proof"] == "note_exit_and_only_unreaped_leader"


def test_failed_registration_uses_unreleased_gate_proof(fixture):
    def fail():
        raise OSError("No event registration")

    fixture.queue.on_register = fail
    with pytest.raises(owner.OwnerError) as caught:
        fixture.create()
    result = caught.value.record
    assert result["cleanup_verified"] and not result["exit_observed"]
    assert result["reap_proof"] == "unreleased_startup_gate"
    assert not any(
        row[0] in {"group_signal", "membership"} for row in fixture.system.calls
    )


@pytest.mark.skipif(
    sys.platform != "darwin" or os.environ.get("ARM64_RUN_OWNER_SMOKE") != "1",
    reason="Opt-in macOS process-group smoke; run only when measurements are idle.",
)
@pytest.mark.parametrize("mode", ["clean", "orphan", "cancel"])
def test_real_owned_group_smoke(tmp_path, mode):
    program = r"""
import os, subprocess, sys, time
gate = int(os.environ.pop('ARM64_BENCHMARK_GATE_FD'))
allowed = os.read(gate, 1)
os.close(gate)
if allowed != b'G':
    raise SystemExit(3)
mode = sys.argv[1]
if mode == 'orphan':
    subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                     start_new_session=False)
os.write(1, b'ready\n')
if mode == 'cancel':
    time.sleep(30)
os._exit(0)
"""

    def clock():
        return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)

    worker = owner.OwnedWorker(
        [sys.executable, "-c", program, mode],
        cwd=tmp_path,
        stdout_path=tmp_path / "stdout",
        stderr_path=tmp_path / "stderr",
        clock_ns=clock,
        cleanup_seconds=0.5,
    )
    result = None
    try:
        worker.release()
        deadline = clock() + 3_000_000_000
        while clock() < deadline:
            worker.poll_stdout()
            if mode == "cancel" and worker.first_stdout_ns is not None:
                break
            if mode != "cancel" and worker.exited():
                break
            time.sleep(0.01)
        else:
            pytest.fail("Owned test worker did not reach its expected boundary.")
        result = worker.finish(terminate=mode == "cancel")
        assert result["cleanup_verified"] and result["worker_reaped"], result
        assert result["wait4_peak_rss_bytes"] > 0
        assert (tmp_path / "stdout").read_bytes() == b"ready\n"
        if mode == "orphan":
            assert (
                result["first_failure"]["reason"] == "group_present_after_worker_exit"
            )
            assert result["cleanup_signals"]
        else:
            assert result["first_failure"] is None, result
        assert not owner._HELD
    finally:
        if result is None:
            result = worker.finish(terminate=True)
        with (tmp_path / "ownership.json").open("x") as output:
            json.dump({"mode": mode, "ownership": result}, output, indent=2)
            output.write("\n")
