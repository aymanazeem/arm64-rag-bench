import copy
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "benchmark_host_test",
    Path(__file__).resolve().parents[1] / "scripts/benchmark_host.py",
)
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


def snapshot():
    return {
        "identity": {
            "system": "Darwin",
            "release": "synthetic",
            "machine": "arm64",
            "hardware": {
                "model": "Synthetic",
                "physicalcpu": 10,
                "logicalcpu": 10,
                "memsize": 64 * 1024**3,
            },
        },
        "power_source": host.available("AC Power"),
        "power_mode": host.available({"powermode": 0}),
        "memory_pressure": host.available({"code": 1, "label": "normal"}),
        "thermal": host.unavailable("cpu_limits_not_reported"),
        "swap": host.available(
            {"total_bytes": 12e9, "used_bytes": 11e9, "free_bytes": 1e9}
        ),
        "pages": host.available(
            {
                "page_size_bytes": 16384,
                "Pageins": 10,
                "Pageouts": 2,
                "Swapins": 3,
                "Swapouts": 4,
            }
        ),
    }


@pytest.mark.parametrize("code,label", [(1, "normal"), (2, "warning"), (4, "critical")])
def test_pressure_uses_userspace_flags(code, label):
    assert host.parse("memory_pressure", str(code)) == host.available(
        {"code": code, "label": label}
    )
    value = snapshot()
    value["memory_pressure"] = host.parse("memory_pressure", str(code))
    assert bool(host.admit(value, host.policy_from(snapshot()))) == (code != 1)


def test_unknown_or_unavailable_pressure_cannot_admit_work():
    value = snapshot()
    value["memory_pressure"] = host.parse("memory_pressure", "0")
    assert value["memory_pressure"]["raw_code"] == 0
    with pytest.raises(ValueError, match="admission"):
        host.policy_from(value)


def test_power_mode_parser_ignores_paths_and_other_power_profiles():
    value = host.parse(
        "power_mode",
        "Battery Power:\n powermode 1\nAC Power:\n powermode 0\n"
        " hibernatefile /private/secret\n",
    )
    assert value == host.available({"powermode": 0})
    assert host.parse(
        "power_source", "Now drawing from 'AC Power'\nPrivate battery serial\n"
    ) == host.available("AC Power")


def test_thermal_failure_text_is_unavailable_even_with_zero_exit():
    def run(argv, **kwargs):
        assert kwargs["timeout"] == 3 and kwargs["env"]["LANG"] == "C"
        return SimpleNamespace(
            returncode=0,
            stdout=b"Error: No CPU power status with error code 0xe00002bc",
            stderr=b"",
        )

    actual = host.probe("thermal", run=run)
    assert actual["status"] == "unavailable" and "value" not in actual
    assert actual["returncode"] == 0 and actual["stdout_sha256"]


@pytest.mark.parametrize("result", ["denied", "timeout", "oversized"])
def test_probe_failures_are_sanitized_and_never_normal(result):
    def run(argv, **kwargs):
        if result == "timeout":
            raise subprocess.TimeoutExpired(argv, 3, output=b"/Users/private")
        return SimpleNamespace(
            returncode=1 if result == "denied" else 0,
            stdout=b"1" if result == "denied" else b"x" * (host.MAX_PROBE_BYTES + 1),
            stderr=b"/Users/private denied",
        )

    actual = host.probe("memory_pressure", run=run)
    assert actual["status"] == "unavailable" and "value" not in actual
    assert "/Users/" not in repr(actual)


def test_permission_failure_is_distinct_from_missing_host_metric():
    result = SimpleNamespace(
        returncode=1, stdout=b"", stderr=b"sysctl: Operation not permitted"
    )
    actual = host.probe("memory_pressure", run=lambda *args, **kwargs: result)
    assert actual["reason"] == "permission_denied"
    assert actual["status"] == "unavailable" and "value" not in actual


def test_known_cpu_limits_are_frozen_and_throttled_baseline_rejected():
    value = snapshot()
    value["thermal"] = host.parse(
        "thermal",
        "CPU_Speed_Limit = 100\nCPU_Scheduler_Limit = 100\nCPU_Available_CPUs = 10\n",
    )
    policy = host.policy_from(value)
    changed = copy.deepcopy(value)
    changed["thermal"]["value"]["CPU_Speed_Limit"] = 50
    assert host.admit(changed, policy) == ["thermal_observation_changed"]
    with pytest.raises(ValueError, match="CPU limit"):
        host.policy_from(changed)
    changed["thermal"] = host.unavailable("probe_failed")
    assert host.admit(changed, policy) == ["thermal_observation_changed"]


@pytest.mark.parametrize(
    "field,warning", [("Thermal Warning Level", 5), ("Performance Warning Level", 100)]
)
def test_reported_warning_cannot_be_hidden_by_missing_cpu_limits(field, warning):
    value = snapshot()
    value["thermal"] = host.parse("thermal", f"2026-11-13 {field} = 0\n")
    policy = host.policy_from(value)
    assert "CPU_Speed_Limit" in value["thermal"]["missing_fields"]
    value["thermal"] = host.parse("thermal", f"{field} = {warning}\n")
    assert host.admit(value, policy) == ["thermal_observation_changed"]
    with pytest.raises(ValueError, match="thermal"):
        host.policy_from(value)


def test_swap_allocation_and_load_do_not_become_arbitrary_thresholds():
    value = snapshot()
    value["load_average"] = host.available([50.0, 20.0, 10.0])
    assert not host.admit(value, host.policy_from(value))
    parsed = host.parse(
        "swap", "total = 12288.00M used = 11256.50M free = 1031.50M (encrypted)"
    )
    assert parsed["value"]["used_bytes"] == 11256.5 * 1024**2
    assert (
        host.parse("swap", "total = 1M used = 1M used = 2M free = 0M")["status"]
        == "unavailable"
    )


def test_active_swap_invalidates_pair_but_ordinary_pageins_are_recorded():
    before, after = snapshot(), snapshot()
    after["pages"]["value"]["Pageins"] += 10
    assert not host.interval_violations(before, after)
    assert host.deltas(before, after)["pages"]["value"]["Pageins"] == 10
    after["pages"]["value"]["Swapins"] += 1
    assert host.interval_violations(before, after) == ["swap_activity"]


@pytest.mark.parametrize("change", ["reset", "missing", "page_size"])
def test_missing_or_reset_counters_cannot_establish_swap_free_interval(change):
    before, after = snapshot(), snapshot()
    if change == "reset":
        after["pages"]["value"]["Swapouts"] = 0
    elif change == "missing":
        after["pages"] = host.unavailable("probe_failed")
    else:
        after["pages"]["value"]["page_size_bytes"] = 4096
    assert host.interval_violations(before, after) == ["swap_counters_unverified"]


def test_desktop_policy_records_swap_without_hiding_raw_activity():
    before, after = snapshot(), snapshot()
    policy = host.policy_from(before, swap_activity_policy="record_counter_increases")
    after["pages"]["value"]["Swapins"] += 7
    assert not host.interval_violations(before, after, policy)
    assert host.interval_violations(before, after) == ["swap_activity"]
    assert host.deltas(before, after)["pages"]["value"]["Swapins"] == 7
    with pytest.raises(ValueError, match="policy"):
        host.policy_from(before, swap_activity_policy="ignore_everything")


@pytest.mark.parametrize("change", ["reset", "missing", "page_size"])
def test_desktop_policy_still_rejects_unverified_counters(change):
    before, after = snapshot(), snapshot()
    policy = host.policy_from(before, swap_activity_policy="record_counter_increases")
    if change == "reset":
        after["pages"]["value"]["Swapouts"] = 0
    elif change == "missing":
        after["pages"] = host.unavailable("probe_failed")
    else:
        after["pages"]["value"]["page_size_bytes"] = 4096
    assert host.interval_violations(before, after, policy) == [
        "swap_counters_unverified"
    ]
