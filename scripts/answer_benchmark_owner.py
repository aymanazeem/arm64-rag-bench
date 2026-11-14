"""Own one gated benchmark worker and its contained native server on macOS."""

import ctypes
import errno
import math
import os
import select
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager

GATE_ENV = "ARM64_BENCHMARK_GATE_FD"
_HELD = {}


def _load_group_reader():
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    reader = library.proc_listpids
    reader.argtypes = (ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int)
    reader.restype = ctypes.c_int
    return reader


class OwnerError(RuntimeError):
    def __init__(self, reason, record):
        super().__init__(reason)
        self.record = record


class OwnedWorker:
    """Keep the leader unreaped until no other group member remains."""

    def __init__(
        self,
        argv,
        *,
        cwd,
        stdout_path,
        stderr_path,
        clock_ns,
        env=None,
        output_limit=16 << 20,
        cleanup_seconds=5,
    ):
        self.clock_ns = clock_ns
        self.output_limit = output_limit
        self.cleanup_seconds = cleanup_seconds
        self.process = self.queue = None
        self.pid = self.pgid = None
        self.first_stdout_ns = None
        self._gate = self._read_gate = None
        self._outputs = {}
        self._pipes = {}
        self._required_pipes = set()
        self._nonblocking_pipes = set()
        self._pipe_eof = set()
        self._drain_timed_out = False
        self._exit_seen = self._released = self._reaped = False
        self._reap_lost = False
        self._group_verified = False
        self._last_clock = None
        self._pending_signal = None
        self._deferring = False
        self._record = {
            "pid": None,
            "pgid": None,
            "released": False,
            "exit_observed": False,
            "returncode": None,
            "worker_reaped": False,
            "cleanup_verified": False,
            "wait4_peak_rss_bytes": None,
            "wait4_user_seconds": None,
            "wait4_system_seconds": None,
            "first_failure": None,
            "cleanup_failures": [],
            "cleanup_signals": [],
            "cleanup_signal_errors": [],
            "group_membership_checks": 0,
            "reap_proof": None,
            "cleanup_clock": "continuous",
            "bytes_observed": {"stdout": 0, "stderr": 0},
            "bytes_retained": {"stdout": 0, "stderr": 0},
            "first_stdout_ns": None,
            "first_stderr_ns": None,
            "output_incomplete": False,
        }
        if sys.platform != "darwin" or any(
            not hasattr(select, name)
            for name in (
                "kqueue",
                "kevent",
                "KQ_FILTER_PROC",
                "KQ_NOTE_EXIT",
                "KQ_EV_ADD",
                "KQ_EV_ONESHOT",
                "KQ_EV_ERROR",
            )
        ):
            self._raise("startup", "unsupported_platform")
        if _HELD:
            self._raise("startup", "previous_worker_unresolved")
        if (
            threading.current_thread() is not threading.main_thread()
            or threading.active_count() != 1
            or signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL
        ):
            self._raise("startup", "exclusive_reaping_required")
        if (
            type(output_limit) is not int
            or output_limit <= 0
            or type(cleanup_seconds) not in (int, float)
            or not math.isfinite(cleanup_seconds)
            or not 0 < cleanup_seconds <= 60
        ):
            self._raise("startup", "invalid_bounds")
        with self._defer_signals():
            try:
                self._list_group_pids = _load_group_reader()
                self._outputs["stdout"] = stdout_path.open("xb")
                self._outputs["stderr"] = stderr_path.open("xb")
                self._read_gate, self._gate = os.pipe()
                environment = dict(os.environ if env is None else env)
                environment[GATE_ENV] = str(self._read_gate)
                self.process = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    pass_fds=(self._read_gate,),
                    start_new_session=True,
                )
                self.pid = self.process.pid
                self._record["pid"] = self.pid
                _HELD[self.pid] = self
                self._pipes = {
                    "stdout": self.process.stdout,
                    "stderr": self.process.stderr,
                }
                self._required_pipes = set(self._pipes)
                os.close(self._read_gate)
                self._read_gate = None
                self.pgid = os.getpgid(self.pid)
                self._record["pgid"] = self.pgid
                if self.pgid != self.pid or self.pid == os.getpgrp():
                    self._raise("startup", "worker_group_mismatch")
                self._group_verified = True
                for name, pipe in self._pipes.items():
                    self._prepare_pipe(name, pipe)
                self.queue = select.kqueue()
                event = select.kevent(
                    self.pid,
                    filter=select.KQ_FILTER_PROC,
                    flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                    fflags=select.KQ_NOTE_EXIT,
                )
                events = self.queue.control([event], 1, 0)
                self._events(events)
                if self._exit_seen:
                    self._raise("startup", "worker_exited_before_release")
                if self._pending_signal is not None:
                    self._raise("startup", "cancelled_before_release")
            except BaseException as error:
                self._failure("startup", "worker_start_failed", error)
                self.finish(terminate=True)
                raise OwnerError(
                    self._record["first_failure"]["reason"], self.record()
                ) from error

    def record(self):
        import copy

        return copy.deepcopy(self._record)

    def _failure(self, phase, reason, error=None, *, cleanup=False):
        item = {"phase": phase, "reason": reason}
        if error is not None:
            item["error_type"] = type(error).__name__
        if self._record["first_failure"] is None:
            self._record["first_failure"] = item
        if cleanup:
            self._record["cleanup_failures"].append(item)

    def _raise(self, phase, reason, error=None):
        self._failure(phase, reason, error)
        raise OwnerError(reason, self.record())

    @contextmanager
    def _defer_signals(self):
        if self._deferring:
            yield
            return
        saved = {}

        def defer(number, frame):
            if self._pending_signal is None:
                self._pending_signal = number

        try:
            for number in (signal.SIGINT, signal.SIGTERM):
                saved[number] = signal.signal(number, defer)
            self._deferring = True
            yield
        finally:
            self._deferring = False
            for number, handler in saved.items():
                signal.signal(number, handler)

    def _stamp(self):
        value = self.clock_ns()
        if (
            type(value) is not int
            or value < 0
            or (self._last_clock is not None and value < self._last_clock)
        ):
            self._raise("clock", "invalid_continuous_clock")
        self._last_clock = value
        return value

    def _events(self, events):
        for event in events:
            if event.flags & select.KQ_EV_ERROR:
                self._raise("exit_observer", "process_event_error")
            if (
                event.ident != self.pid
                or event.filter != select.KQ_FILTER_PROC
                or event.fflags != select.KQ_NOTE_EXIT
            ):
                self._raise("exit_observer", "unexpected_process_event")
            self._exit_seen = True
            self._record["exit_observed"] = True

    def exited(self):
        if self._reaped:
            return True
        if self.queue is None:
            self._raise("exit_observer", "missing_process_observer")
        try:
            self._events(self.queue.control(None, 1, 0))
        except OSError as error:
            self._raise("exit_observer", "process_observer_failed", error)
        return self._exit_seen

    def release(self):
        with self._defer_signals():
            try:
                if self._released or self._gate is None or self._reaped:
                    self._raise("release", "invalid_release")
                if self.exited() or self._pending_signal is not None:
                    self._raise("release", "worker_unavailable_before_release")
                if os.write(self._gate, b"G") != 1:
                    self._raise("release", "gate_write_failed")
                self._released = True
                self._record["released"] = True
                self._close_gate()
                if self._pending_signal is not None:
                    self._raise("release", "cancelled_during_release")
            except BaseException as error:
                self._failure("release", "worker_release_failed", error)
                self.finish(terminate=True)
                raise OwnerError(
                    self._record["first_failure"]["reason"], self.record()
                ) from error

    def _prepare_pipe(self, name, pipe):
        if name not in self._nonblocking_pipes:
            try:
                os.set_blocking(pipe.fileno(), False)
            except OSError as error:
                self._record["output_incomplete"] = True
                self._raise("capture", "pipe_configuration_failed", error)
            self._nonblocking_pipes.add(name)

    def poll_stdout(self):
        receipts = []
        for name, pipe in list(self._pipes.items()):
            if name not in self._pipe_eof:
                self._prepare_pipe(name, pipe)
            try:
                data = b"" if name in self._pipe_eof else os.read(pipe.fileno(), 65536)
            except BlockingIOError:
                continue
            except OSError as error:
                self._record["output_incomplete"] = True
                self._raise("capture", "pipe_read_failed", error)
            if not data:
                self._pipe_eof.add(name)
                try:
                    pipe.close()
                except OSError as error:
                    self._raise("capture", "pipe_close_failed", error)
                del self._pipes[name]
                continue
            stamp = None
            clock_error = None
            try:
                stamp = self._stamp()
            except BaseException as error:
                clock_error = error
                self._failure("capture", "receipt_clock_failed", error)
            first_read = self._record["bytes_observed"][name] == 0
            self._record["bytes_observed"][name] += len(data)
            if first_read:
                self._record["first_" + name + "_ns"] = stamp
                self.first_stdout_ns = self._record["first_stdout_ns"]
            remaining = self.output_limit - sum(self._record["bytes_retained"].values())
            kept = data[: max(0, remaining)]
            try:
                if self._outputs[name].write(kept) != len(kept):
                    raise OSError("Short capture write")
            except OSError as error:
                self._record["output_incomplete"] = True
                self._raise("capture", "capture_write_failed", error)
            self._record["bytes_retained"][name] += len(kept)
            receipts.append({"stream": name, "received_ns": stamp, "bytes": len(data)})
            if len(kept) != len(data):
                self._record["output_incomplete"] = True
                self._raise("capture", "capture_limit_exceeded")
            if clock_error is not None:
                raise OwnerError(
                    self._record["first_failure"]["reason"], self.record()
                ) from clock_error
        return receipts

    def _close_gate(self):
        failure = None
        for field in ("_gate", "_read_gate"):
            descriptor = getattr(self, field)
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError as error:
                    failure = failure or error
                else:
                    setattr(self, field, None)
        if failure is not None:
            raise failure

    def _group_alive(self):
        if not self._exit_seen:
            self.exited()
        # XNU killpg skips zombies; EPERM cannot prove the owned group is empty.
        members = (ctypes.c_int * 2)()
        ctypes.set_errno(0)
        count = self._list_group_pids(2, self.pgid, members, ctypes.sizeof(members))
        error = ctypes.get_errno()
        if error or count <= 0:
            raise OSError(error or errno.EIO, "Process-group query failed")
        if count > ctypes.sizeof(members) or count % ctypes.sizeof(ctypes.c_int):
            self._raise("cleanup", "invalid_group_membership")
        values = list(members[: count // ctypes.sizeof(ctypes.c_int)])
        if any(pid <= 0 for pid in values) or len(set(values)) != len(values):
            self._raise("cleanup", "invalid_group_membership")
        self._record["group_membership_checks"] += 1
        if len(values) == 2:
            return True  # Full buffers cannot prove singleton membership.
        if values != [self.pid]:
            self._raise("cleanup", "worker_missing_from_group")
        if self._exit_seen:
            self._record["reap_proof"] = "note_exit_and_only_unreaped_leader"
            return False
        return True

    def _cleanup_call(self, function):
        try:
            return function()
        except (InterruptedError, KeyboardInterrupt) as error:
            self._failure("cleanup", "cleanup_interrupted", error, cleanup=True)
            return function()

    def _drain_cleanup(self):
        try:
            self.poll_stdout()
        except (OwnerError, OSError) as error:
            self._failure(
                "cleanup", "capture_failed_during_cleanup", error, cleanup=True
            )
        return not self._pipes

    def _wait_cleanup(self, ready, *, drain=True):
        fallback = self._record["cleanup_clock"] == "monotonic_fallback"
        previous = deadline = None
        while True:
            if self._cleanup_call(ready):
                return True
            try:
                now = time.monotonic_ns() if fallback else self._stamp()
                if (
                    type(now) is not int
                    or now < 0
                    or (previous is not None and now <= previous)
                ):
                    raise ValueError("Cleanup clock did not advance")
            except (AttributeError, OSError, ValueError, OwnerError) as error:
                if fallback:
                    self._raise("cleanup", "cleanup_fallback_failed", error)
                fallback = True
                previous = deadline = None
                self._record["cleanup_clock"] = "monotonic_fallback"
                self._failure(
                    "cleanup", "cleanup_clock_unavailable", error, cleanup=True
                )
                continue
            if deadline is None:
                deadline = now + int(self.cleanup_seconds * 1e9)
            if now >= deadline:
                return False
            if drain:
                self._cleanup_call(self._drain_cleanup)
            previous = now
            self._cleanup_call(lambda: time.sleep(0.01))

    def _reap(self):
        try:
            pid, status, usage = os.wait4(self.pid, os.WNOHANG)
        except ChildProcessError as error:
            self._reap_lost = True
            self._raise("cleanup", "worker_reaping_ownership_lost", error)
        if not pid:
            return False
        if pid != self.pid:
            self._raise("cleanup", "wrong_child_reaped")
        self.process.returncode = os.waitstatus_to_exitcode(status)
        self._reaped = True
        self._record.update(
            worker_reaped=True,
            returncode=self.process.returncode,
            wait4_peak_rss_bytes=usage.ru_maxrss,
            wait4_user_seconds=usage.ru_utime,
            wait4_system_seconds=usage.ru_stime,
        )
        return True

    def _cleanup_process(self, terminate):
        if not self._released:
            # A closed startup gate prevents application descendants.
            try:
                self._cleanup_call(lambda: os.kill(self.pid, signal.SIGKILL))
            except ProcessLookupError:
                pass
            group_gone = True
            self._record["reap_proof"] = "unreleased_startup_gate"
        else:
            if not self._group_verified:
                self._raise("cleanup", "unverified_released_group")
            if not terminate and not self._exit_seen:
                self._failure("cleanup", "finish_before_exit")
            group_gone = not self._cleanup_call(self._group_alive)
            if not terminate and self._exit_seen and not group_gone:
                self._failure("cleanup", "group_present_after_worker_exit")
            for number in (signal.SIGTERM, signal.SIGKILL):
                if group_gone:
                    break
                denied = None
                try:
                    self._cleanup_call(lambda: os.killpg(self.pgid, number))
                    self._record["cleanup_signals"].append(signal.Signals(number).name)
                except ProcessLookupError:
                    pass
                except PermissionError as error:
                    denied = error
                    self._record["cleanup_signal_errors"].append(
                        {
                            "signal": signal.Signals(number).name,
                            "error_type": "PermissionError",
                            "resolved_by_membership": False,
                        }
                    )
                try:
                    group_gone = self._wait_cleanup(lambda: not self._group_alive())
                except (OwnerError, OSError) as error:
                    self._failure("cleanup", "group_wait_failed", error, cleanup=True)
                    group_gone = not self._cleanup_call(self._group_alive)
                if denied is not None:
                    if not group_gone:
                        raise denied
                    self._record["cleanup_signal_errors"][-1][
                        "resolved_by_membership"
                    ] = True
        if not group_gone:
            self._raise("cleanup", "group_cleanup_unproved")
        if not self._wait_cleanup(self._reap):
            self._raise("cleanup", "worker_reap_timeout")
        if not self._wait_cleanup(self._drain_cleanup, drain=False):
            self._drain_timed_out = True
            self._raise("cleanup", "pipes_open_after_group_exit")

    def finish(self, terminate=False):
        if self._record["cleanup_verified"]:
            return self.record()
        closed = False
        try:
            with self._defer_signals():
                try:
                    try:
                        self._cleanup_call(self._close_gate)
                    except OSError as error:
                        self._failure(
                            "cleanup", "gate_close_failed", error, cleanup=True
                        )
                    if (
                        self.process is not None
                        and not self._reaped
                        and not self._reap_lost
                    ):
                        self._cleanup_process(terminate)
                    elif self._reaped and self._pipes:
                        if not self._wait_cleanup(self._drain_cleanup, drain=False):
                            self._drain_timed_out = True
                            self._raise("cleanup", "pipes_open_after_group_exit")
                finally:
                    if self.process is None or self._reaped or self._reap_lost:
                        closed = self._close_capture()
        except BaseException as error:
            self._failure("cleanup", "cleanup_failed", error, cleanup=True)
        if self._pending_signal is not None:
            self._failure("cleanup", "termination_deferred")
            self._record["deferred_signal"] = self._pending_signal
        self._record["cleanup_verified"] = bool(
            (self.process is None or self._reaped)
            and closed
            and self._gate is None
            and self._read_gate is None
            and self._required_pipes <= self._pipe_eof
            and not self._drain_timed_out
        )
        if self._record["cleanup_verified"]:
            if _HELD.get(self.pid) is self:
                _HELD.pop(self.pid)
        else:
            _HELD[self.pid] = self
        return self.record()

    def _close_capture(self):
        closed = True
        for name, stream in list(self._pipes.items()):
            try:
                self._cleanup_call(stream.close)
            except BaseException as error:
                closed = False
                self._failure("cleanup", "pipe_close_failed", error, cleanup=True)
            else:
                del self._pipes[name]
        for stream in self._outputs.values():
            if stream.closed:
                continue
            try:
                self._cleanup_call(stream.flush)
                self._cleanup_call(lambda: os.fsync(stream.fileno()))
            except BaseException as error:
                self._record["output_incomplete"] = True
                self._failure("cleanup", "capture_flush_failed", error, cleanup=True)
            finally:
                try:
                    self._cleanup_call(stream.close)
                except BaseException as error:
                    closed = False
                    self._failure(
                        "cleanup", "capture_close_failed", error, cleanup=True
                    )
        if self.queue is not None:
            try:
                self._cleanup_call(self.queue.close)
            except BaseException as error:
                closed = False
                self._failure("cleanup", "observer_close_failed", error, cleanup=True)
            else:
                self.queue = None
        return (
            closed
            and not self._pipes
            and all(stream.closed for stream in self._outputs.values())
            and self.queue is None
        )
