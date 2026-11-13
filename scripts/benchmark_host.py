"""Read-only macOS observations and frozen benchmark admission rules."""

import hashlib
import json
import math
import os
import platform
import re
import subprocess
from datetime import UTC, datetime

PROBE_SECONDS = 3
MAX_PROBE_BYTES = 16384
PROBES = {
    "power_source": ["/usr/bin/pmset", "-g", "batt"],
    "power_mode": ["/usr/bin/pmset", "-g", "custom"],
    "memory_pressure": [
        "/usr/sbin/sysctl",
        "-n",
        "kern.memorystatus_vm_pressure_level",
    ],
    "thermal": ["/usr/bin/pmset", "-g", "therm"],
    "swap": ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
    "pages": ["/usr/bin/vm_stat"],
    "hardware": [
        "/usr/sbin/sysctl",
        "hw.model",
        "hw.physicalcpu",
        "hw.logicalcpu",
        "hw.memsize",
    ],
}
PRESSURE = {1: "normal", 2: "warning", 4: "critical"}
MODE_KEYS = {"powermode", "lowpowermode", "highpowermode"}
THERMAL_KEYS = {
    "CPU_Speed_Limit",
    "CPU_Scheduler_Limit",
    "CPU_Available_CPUs",
    "Thermal Warning Level",
    "Performance Warning Level",
}


def unavailable(reason, **metadata):
    return {"status": "unavailable", "reason": reason, **metadata}


def available(value, **metadata):
    return {"status": "available", "value": value, **metadata}


def parse(name, text):
    if name == "power_source":
        match = re.search(
            r"^Now drawing from '(AC Power|Battery Power|UPS Power)'$", text, re.M
        )
        if match:
            return available(match[1])
    elif name == "power_mode":
        section, values = None, {}
        for line in text.splitlines():
            if line and not line[0].isspace():
                section = line.strip()
            match = re.fullmatch(r"\s*(\w+)\s+(\d+)\s*", line)
            if section == "AC Power:" and match and match[1] in MODE_KEYS:
                if match[1] in values:
                    return unavailable("duplicate_field")
                values[match[1]] = int(match[2])
        if values:
            return available(values)
    elif name == "memory_pressure":
        if re.fullmatch(r"\s*\d+\s*", text):
            code = int(text)
            if code in PRESSURE:
                return available({"code": code, "label": PRESSURE[code]})
            return unavailable("unknown_pressure_code", raw_code=code)
    elif name == "thermal":
        values = {}
        for key, number in re.findall(
            r"(CPU_\w+|Thermal Warning Level|Performance Warning Level)\s*=\s*(\d+)",
            text,
        ):
            if key in THERMAL_KEYS:
                if key in values:
                    return unavailable("duplicate_field")
                values[key] = int(number)
        if values:
            for key, value in values.items():
                if (
                    key in {"CPU_Speed_Limit", "CPU_Scheduler_Limit"}
                    and not 0 <= value <= 100
                ):
                    return unavailable("invalid_limit")
            return available(
                values, missing_fields=sorted(THERMAL_KEYS - values.keys())
            )
        return unavailable("cpu_limits_not_reported")
    elif name == "swap":
        values = {}
        units = {"K": 1024, "M": 1024**2, "G": 1024**3}
        for key, number, unit in re.findall(
            r"\b(total|used|free)\s*=\s*(\d+(?:\.\d+)?)([KMG])\b", text
        ):
            if key + "_bytes" in values:
                return unavailable("duplicate_field")
            value = float(number) * units[unit]
            if not math.isfinite(value):
                return unavailable("invalid_number")
            values[key + "_bytes"] = value
        if len(values) == 3:
            return available(values, precision="Rounded sysctl display values.")
    elif name == "pages":
        page_size = re.search(r"page size of (\d+) bytes", text)
        values = {}
        for key, value in re.findall(
            r"^(Pageins|Pageouts|Swapins|Swapouts):\s*(\d+)\.?\s*$", text, re.M
        ):
            if key in values:
                return unavailable("duplicate_field")
            values[key] = int(value)
        if page_size and len(values) == 4 and int(page_size[1]) > 0:
            return available({"page_size_bytes": int(page_size[1]), **values})
    elif name == "hardware":
        values = {}
        for key, value in re.findall(
            r"^hw\.(model|physicalcpu|logicalcpu|memsize):\s*([A-Za-z0-9,._-]+)\s*$",
            text,
            re.M,
        ):
            if key in values:
                return unavailable("duplicate_field")
            if key != "model":
                if not value.isdecimal() or int(value) <= 0:
                    return unavailable("invalid_hardware_value")
                value = int(value)
            values[key] = value
        if len(values) == 4:
            return available(values)
    return unavailable("unrecognized_output")


def probe(name, *, run=subprocess.run):
    try:
        result = run(
            PROBES[name],
            capture_output=True,
            timeout=PROBE_SECONDS,
            env={"PATH": os.defpath, "LANG": "C", "LC_ALL": "C"},
        )
    except subprocess.TimeoutExpired:
        return unavailable("probe_timeout")
    except OSError:
        return unavailable("probe_unavailable")
    metadata = {
        "returncode": result.returncode,
        "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
    }
    if result.returncode:
        reason = (
            "permission_denied"
            if b"Operation not permitted" in result.stderr
            or b"Permission denied" in result.stderr
            else "probe_failed"
        )
        return unavailable(reason, **metadata)
    if len(result.stdout) > MAX_PROBE_BYTES or len(result.stderr) > MAX_PROBE_BYTES:
        return unavailable("probe_output_limit", **metadata)
    try:
        return {**parse(name, result.stdout.decode("utf-8")), **metadata}
    except (UnicodeError, ValueError, OverflowError):
        return unavailable("malformed_output", **metadata)


def collect(*, run=subprocess.run, getloadavg=os.getloadavg):
    observations = {name: probe(name, run=run) for name in PROBES}
    try:
        values = list(getloadavg())
        if len(values) != 3 or not all(math.isfinite(x) and x >= 0 for x in values):
            raise ValueError("Invalid load averages.")
        observations["load_average"] = available(values)
    except (OSError, ValueError):
        observations["load_average"] = unavailable("load_unavailable")
    return {
        "schema_version": 1,
        "observed_at": datetime.now(UTC).isoformat(),
        "identity": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "hardware": observations["hardware"].get("value"),
        },
        **observations,
    }


def policy_from(snapshot):
    policy = {
        "schema_version": 1,
        "identity": snapshot["identity"],
        "power_source": "AC Power",
        "power_mode": snapshot["power_mode"].get("value"),
        "memory_pressure_code": 1,
        "thermal": {
            key: snapshot["thermal"][key]
            for key in ("status", "value")
            if key in snapshot["thermal"]
        },
        "probe_timeout_seconds": PROBE_SECONDS,
        "observation_boundary": "before_and_after_each_invocation",
        "swap_activity_policy": "exclude_any_counter_increase",
        "load_policy": "observe_only",
        "limit": (
            "Boundary observations do not prove constant conditions between checks. "
            "Unavailable CPU-limit telemetry is not a normal thermal reading."
        ),
    }
    check_policy(policy)
    if admit(snapshot, policy):
        raise ValueError("Host snapshot does not meet admission conditions.")
    return policy


def check_policy(policy):
    if not (
        policy["schema_version"] == 1
        and policy["identity"]["system"] == "Darwin"
        and isinstance(policy["identity"].get("hardware"), dict)
        and set(policy["identity"]["hardware"])
        == {"model", "physicalcpu", "logicalcpu", "memsize"}
        and policy["power_source"] == "AC Power"
        and policy["power_mode"]
        and set(policy["power_mode"]) <= MODE_KEYS
        and all(type(x) is int and x >= 0 for x in policy["power_mode"].values())
        and type(policy["memory_pressure_code"]) is int
        and policy["memory_pressure_code"] == 1
        and policy["probe_timeout_seconds"] == PROBE_SECONDS
        and policy["observation_boundary"] == "before_and_after_each_invocation"
        and policy["swap_activity_policy"] == "exclude_any_counter_increase"
        and policy["load_policy"] == "observe_only"
        and policy["thermal"]["status"] in {"available", "unavailable"}
    ):
        raise ValueError("Invalid frozen host policy.")
    thermal = policy["thermal"]
    if thermal["status"] == "available":
        values = thermal.get("value", {})
        if (
            not values
            or not set(values) <= THERMAL_KEYS
            or any(
                type(value) is not int
                or (
                    value <= 0
                    if key == "CPU_Available_CPUs"
                    else value != 0
                    if key.endswith("Warning Level")
                    else value != 100
                )
                for key, value in values.items()
            )
        ):
            raise ValueError(
                "The baseline reports a CPU limit or invalid thermal data."
            )
    elif "value" in thermal:
        raise ValueError("Unavailable thermal telemetry cannot have a value.")


def admit(snapshot, policy):
    violations = []
    if snapshot.get("identity") != policy["identity"]:
        violations.append("host_identity_changed")
    for name, expected in (
        ("power_source", policy["power_source"]),
        ("power_mode", policy["power_mode"]),
        ("memory_pressure", {"code": 1, "label": "normal"}),
    ):
        value = snapshot.get(name, {})
        if value.get("status") != "available":
            violations.append(name + "_unavailable")
        elif value.get("value") != expected:
            violations.append(name + "_changed")
    current = snapshot.get("thermal", {})
    actual = {key: current[key] for key in ("status", "value") if key in current}
    if actual != policy["thermal"]:
        violations.append("thermal_observation_changed")
    if snapshot.get("pages", {}).get("status") != "available":
        violations.append("swap_counters_unavailable")
    return violations


def deltas(before, after):
    result = {}
    for name, fields in (
        ("swap", ("used_bytes",)),
        ("pages", ("Pageins", "Pageouts", "Swapins", "Swapouts")),
    ):
        start, end = before.get(name, {}), after.get(name, {})
        if start.get("status") == end.get("status") == "available":
            if (
                name == "pages"
                and start["value"]["page_size_bytes"] != end["value"]["page_size_bytes"]
            ):
                result[name] = unavailable("page_size_changed")
            else:
                changes = {
                    key: end["value"][key] - start["value"][key] for key in fields
                }
                if name == "pages" and any(x < 0 for x in changes.values()):
                    result[name] = unavailable("counter_reset_or_wrap")
                else:
                    result[name] = available(changes)
        else:
            result[name] = unavailable("missing_boundary_observation")
    return result


def interval_violations(before, after):
    pages = deltas(before, after)["pages"]
    if pages["status"] != "available":
        return ["swap_counters_unverified"]
    if any(pages["value"][key] > 0 for key in ("Swapins", "Swapouts")):
        return ["swap_activity"]
    return []


if __name__ == "__main__":
    print(json.dumps(collect(), allow_nan=False, indent=2))
