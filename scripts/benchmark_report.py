"""Verify frozen benchmark records and export pair-level public evidence."""

import argparse
import hashlib
import importlib.util
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "report_benchmark", ROOT / "scripts/benchmark.py"
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
require = bench.require
OUTCOMES = {
    "success",
    "invalid_output",
    "exit_error",
    "timeout",
    "interrupted",
    "launch_error",
    "cleanup_error",
    "record_error",
}
RUN_STATES = {"planned", "running", "finished", "host_blocked", "failed", "interrupted"}
TERMINAL = {"finished", "host_blocked", "failed", "interrupted"}
HOST_FIELDS = {
    "power_mode": bench.host.MODE_KEYS,
    "thermal": bench.host.THERMAL_KEYS,
    "swap": {"total_bytes", "used_bytes", "free_bytes"},
    "pages": {"page_size_bytes", "Pageins", "Pageouts", "Swapins", "Swapouts"},
}
HOST_REASONS = {
    "cpu_limits_not_reported",
    "duplicate_field",
    "unknown_pressure_code",
    "invalid_limit",
    "invalid_number",
    "invalid_hardware_value",
    "unrecognized_output",
    "probe_timeout",
    "probe_unavailable",
    "permission_denied",
    "probe_failed",
    "probe_output_limit",
    "malformed_output",
    "load_unavailable",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def signature(path):
    value = path.stat()
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


class Evidence:
    """Detect changed inputs across a single offline export."""

    def __init__(self):
        self.seen = {}

    def raw(self, path, *, expected=None, bounded=True):
        path = Path(path).resolve(strict=True)
        before = signature(path)
        require(
            not bounded or before[2] <= bench.MAX_JSON_BYTES,
            "Evidence exceeds its size limit.",
        )
        raw = path.read_bytes()
        require(signature(path) == before, "Evidence changed while reading.")
        require(
            path not in self.seen or self.seen[path] == before,
            "Evidence changed during export.",
        )
        self.seen[path] = before
        digest = sha(raw)
        require(expected is None or digest == expected, "Evidence hash differs.")
        return raw, {"sha256": digest, "size": len(raw)}

    def document(self, path, *, expected=None):
        raw, pin = self.raw(path, expected=expected)
        return json.loads(
            raw,
            object_pairs_hook=bench.unique_object,
            parse_constant=bench.reject_constant,
            parse_float=bench.finite_float,
        ), pin

    def file(self, path, *, expected=None):
        path = Path(path).resolve(strict=True)
        before = signature(path)
        digest = bench.runtime.digest(path)
        require(signature(path) == before, "Evidence changed while hashing.")
        require(
            path not in self.seen or self.seen[path] == before,
            "Evidence changed during export.",
        )
        self.seen[path] = before
        require(expected is None or digest == expected, "Evidence hash differs.")
        return {"sha256": digest, "size": before[2]}

    def check(self):
        require(
            all(signature(path) == value for path, value in self.seen.items()),
            "Evidence changed during export.",
        )


def public_asset(item):
    return {key: item[key] for key in ("size", "sha256")}


def numeric(value):
    require(
        type(value) in (int, float) and math.isfinite(value),
        "Invalid public numeric observation.",
    )
    return value


def public_host(snapshot):
    result = {}
    for name in (*HOST_FIELDS, "power_source", "memory_pressure", "load_average"):
        item = snapshot.get(name, {"status": "unavailable"})
        require(
            item.get("status") in {"available", "unavailable"},
            "Unknown host observation status.",
        )
        clean = {"status": item["status"]}
        if item["status"] == "available":
            value = item["value"]
            if name in HOST_FIELDS:
                require(
                    isinstance(value, dict) and set(value) <= HOST_FIELDS[name],
                    "Unknown host observation field.",
                )
                clean["value"] = {key: numeric(number) for key, number in value.items()}
                if name == "thermal":
                    clean["missing_fields"] = sorted(
                        bench.host.THERMAL_KEYS - value.keys()
                    )
            elif name == "power_source":
                require(
                    value in {"AC Power", "Battery Power", "UPS Power"},
                    "Unknown power source.",
                )
                clean["value"] = value
            elif name == "memory_pressure":
                require(
                    type(value["code"]) is int
                    and bench.host.PRESSURE.get(value["code"]) == value["label"],
                    "Unknown pressure observation.",
                )
                clean["value"] = {"code": value["code"], "label": value["label"]}
            else:
                require(
                    isinstance(value, list) and len(value) == 3,
                    "Invalid load observation.",
                )
                clean["value"] = [numeric(number) for number in value]
        elif "reason" in item:
            if item["reason"] in HOST_REASONS:
                clean["reason"] = item["reason"]
            else:
                clean["reason_sha256"] = sha(str(item["reason"]).encode())
        if "raw_code" in item:
            clean["raw_code"] = numeric(item["raw_code"])
        result[name] = clean
    return result


def boundary_violations(plan, rows):
    violations, pair_start, previous, pair = [], None, None, None
    for row in rows:
        if "host_before" not in row:
            continue
        new_pair = row["pair"] != pair
        if new_pair:
            pair, pair_start = row["pair"], row["host_before"]
        for boundary in ("before", "after"):
            snapshot = row.get("host_" + boundary)
            if snapshot is None:
                continue
            reasons = bench.host.admit(snapshot, plan["host_policy"])
            reasons += bench.host.interval_violations(
                pair_start, snapshot, plan["host_policy"]
            )
            if previous is not None:
                interval_policy = plan["host_policy"]
                if new_pair and boundary == "before":
                    interval_policy = {
                        **interval_policy,
                        "swap_activity_policy": "record_counter_increases",
                    }
                reasons += bench.host.interval_violations(
                    previous, snapshot, interval_policy
                )
            reasons = list(dict.fromkeys(reasons))
            previous = snapshot
            if reasons:
                violations.append(
                    {
                        "pair": row["pair"],
                        "row": row["id"],
                        "boundary": boundary,
                        "reasons": reasons,
                    }
                )
        if "host_after" in row:
            require(
                row.get("host_deltas")
                == bench.host.deltas(row["host_before"], row["host_after"]),
                "Recorded host deltas differ.",
            )
    return violations


def checked_rows(evidence, plan, run, directory):
    expected = [row for row in plan["schedule"] if row["session"] == run["session"]]
    require(
        isinstance(run["rows"], list) and len(run["rows"]) == len(expected),
        "Ledger lost schedule rows.",
    )
    public, pending = [], False
    for item, row in zip(expected, run["rows"], strict=True):
        require(
            all(
                type(row.get(key)) is type(value) and row[key] == value
                for key, value in item.items()
            ),
            "Ledger schedule differs.",
        )
        state = row.get("state")
        require(state in {"planned", "attempted", "finished"}, "Unknown attempt state.")
        require(not pending or state == "planned", "Ledger attempts are out of order.")
        pending = state != "finished"
        clean = {**item, "state": state}
        if state == "planned":
            require(
                not any(
                    key in row
                    for key in (
                        "argv",
                        "outcome",
                        "pid",
                        "artifacts",
                        "observed",
                        "host_after",
                    )
                ),
                "Planned row contains an attempt.",
            )
        else:
            require(
                row["argv"] == bench.command(plan, row), "Recorded command differs."
            )
        if state == "finished":
            require(
                not run["integrity_verified"]
                or all(key in row for key in ("host_before", "host_after")),
                "Verified finished row lacks host boundaries.",
            )
            require(row.get("outcome") in OUTCOMES, "Unknown finished outcome.")
            clean["outcome"] = row["outcome"]
            require(numeric(row["wall_seconds"]) >= 0, "Negative process duration.")
            clean["wall_seconds"] = row["wall_seconds"]
            require(
                row.get("returncode") is None or type(row["returncode"]) is int,
                "Invalid process return code.",
            )
            clean["returncode"] = row.get("returncode")
        if "failure" in row:
            clean["failure_sha256"] = sha(str(row["failure"]).encode())
        artifacts = row.get("artifacts", {})
        require(
            set(artifacts) <= {"stdout.json", "stderr.log"}, "Unexpected raw artifact."
        )
        clean["raw_files"] = {}
        for name in ("stdout.json", "stderr.log"):
            path = directory / row["id"] / name
            if name in artifacts:
                require(
                    Path(artifacts[name]["path"]) == path and path.resolve() == path,
                    "Raw artifact escaped its scheduled location.",
                )
                pin = evidence.file(path, expected=artifacts[name]["sha256"])
                require(
                    pin == public_asset(artifacts[name]), "Raw artifact size differs."
                )
            elif path.exists():
                require(
                    state == "attempted",
                    "Unrecorded raw output outside an unresolved attempt.",
                )
                require(
                    path.resolve() == path,
                    "Raw artifact escaped its scheduled location.",
                )
                pin = evidence.file(path)
            else:
                continue
            clean["raw_files"][name] = pin
        if state == "finished" and row["outcome"] == "success":
            require(
                set(artifacts) == {"stdout.json", "stderr.log"}
                and row["returncode"] == 0,
                "Successful process lacks complete raw evidence.",
            )
            native, _ = evidence.document(directory / row["id"] / "stdout.json")
            observed = bench.validate_samples(native, plan, row)
            require(
                observed == row["observed"], "Recorded samples differ from raw output."
            )
            clean["samples_ns"] = observed["samples_ns"]
            clean["excluded_sample_indices"] = [0]
        for boundary in ("host_before", "host_after"):
            if boundary in row:
                clean[boundary] = public_host(row[boundary])
        public.append(clean)
    return public


def session_record(evidence, plan, plan_raw, plan_sha256, item):
    session = item["session"]
    require(type(session) is int and 1 <= session <= 3, "Unknown session ID.")
    directory = Path(item["directory"]).resolve(strict=True)
    raw, archive = evidence.raw(directory / "plan.json")
    require(raw == plan_raw, "Archived session plan differs.")
    run, pin = evidence.document(directory / "run.json", expected=item["run_sha256"])
    require(
        type(run.get("schema_version")) is int
        and run["schema_version"] == 1
        and type(run.get("session")) is int
        and run["session"] == session
        and run.get("plan_sha256") == plan_sha256,
        "Ledger identity differs.",
    )
    require(
        run.get("state") in RUN_STATES and type(run.get("integrity_verified")) is bool,
        "Invalid session state.",
    )
    require(
        not run["integrity_verified"] or run["state"] in {"finished", "host_blocked"},
        "Nonterminal session claims verified integrity.",
    )
    rows = checked_rows(evidence, plan, run, directory)
    violations = boundary_violations(plan, run["rows"])
    require(
        run.get("host_violations") == violations, "Recorded host violations differ."
    )
    if run["state"] == "planned":
        require(
            all(row["state"] == "planned" and "host_before" not in row for row in rows),
            "Planned session contains execution evidence.",
        )
    if violations:
        stopped = next(
            index for index, row in enumerate(rows) if row["id"] == violations[0]["row"]
        )
        boundary = violations[0]["boundary"]
        require(
            rows[stopped]["state"]
            == ("planned" if boundary == "before" else "finished")
            and all(
                row["state"] == "planned" and "host_before" not in row
                for row in rows[stopped + 1 :]
            ),
            "Session continued after a host stop.",
        )
    if run["state"] == "finished":
        require(
            all(row["state"] == "finished" for row in rows)
            and not violations
            and run["integrity_verified"],
            "Finished session is incomplete.",
        )
    if run["state"] == "host_blocked":
        require(
            bool(violations) and run["integrity_verified"],
            "Host-blocked session lacks verified stop evidence.",
        )
    summary = bench.summarize(plan, run)
    summary_pin = None
    if (directory / "summary.json").exists():
        recorded, summary_pin = evidence.document(directory / "summary.json")
        require(
            recorded == summary, "Stored session summary differs from raw evidence."
        )
    result = {
        "session": session,
        "state": run["state"],
        "integrity_verified": run["integrity_verified"],
        "evidence": {"plan": archive, "run": pin, "summary": summary_pin},
        "rows": rows,
        "pairs": summary["pairs"],
        "host_violations": violations,
    }
    if run.get("failure"):
        result["failure_sha256"] = sha(str(run["failure"]).encode())
    return result


def distribution(values):
    return (
        {
            "count": len(values),
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
        }
        if values
        else {"count": 0, "median": None, "minimum": None, "maximum": None}
    )


def privacy_check(value):
    forbidden = {
        "path",
        "argv",
        "pid",
        "environment",
        "env",
        "identity",
        "hardware",
        "cpu_info",
        "gpu_info",
        "model_filename",
        "observed",
        "reasoning",
        "prompt",
        "failure",
    }
    if isinstance(value, dict):
        require(not forbidden.intersection(value), "Private field in public report.")
        for key, item in value.items():
            privacy_check(key)
            privacy_check(item)
    elif isinstance(value, list):
        for item in value:
            privacy_check(item)
    elif isinstance(value, str):
        require(
            not re.search(
                r"/Users/|/home/|/private/|/tmp/|Bearer\s|gh[pousr]_[A-Za-z0-9]|sk-[A-Za-z0-9]",
                value,
            ),
            "Private string in public report.",
        )


def create_report(plan_path, expected_plan_sha256, sessions):
    evidence = Evidence()
    plan_raw, plan_pin = evidence.raw(plan_path, expected=expected_plan_sha256)
    plan, _ = evidence.document(plan_path)
    bench.check_plan(plan)
    desktop = plan["kind"] == bench.STUDIES["desktop"]
    for key, path in (
        ("harness_sha256", Path(bench.__file__)),
        ("runtime_helper_sha256", bench.HELPER),
        ("host_helper_sha256", bench.HOST_HELPER),
    ):
        evidence.file(path, expected=plan[key])
    reporter_pin = evidence.file(Path(__file__))
    assets_before = bench.fingerprint(plan)
    bench.verify_runtime(plan)
    require(1 <= len(sessions) <= 3, "Supply one to three session records.")
    require(
        len({row["session"] for row in sessions}) == len(sessions),
        "Duplicate session ID.",
    )
    require(
        len({Path(row["directory"]).resolve() for row in sessions}) == len(sessions),
        "Duplicate session directory.",
    )
    observed = {
        item["session"]: session_record(
            evidence, plan, plan_raw, expected_plan_sha256, item
        )
        for item in sessions
    }
    records = []
    for number in range(1, 4):
        if number in observed:
            records.append(observed[number])
        else:
            schedule = [row for row in plan["schedule"] if row["session"] == number]
            records.append(
                {
                    "session": number,
                    "state": "unobserved",
                    "integrity_verified": False,
                    "evidence": None,
                    "rows": [{**row, "state": "unobserved"} for row in schedule],
                    "pairs": [
                        {
                            "pair": pair,
                            "workload": next(
                                row["workload"]
                                for row in schedule
                                if row["pair"] == pair
                            ),
                            "included": False,
                            "exclusion": (
                                "Session evidence was not supplied; "
                                "attempt status is unknown."
                            ),
                        }
                        for pair in dict.fromkeys(row["pair"] for row in schedule)
                    ],
                    "host_violations": [],
                }
            )
    rows = [row for record in records for row in record["rows"]]
    pairs = [pair for record in records for pair in record["pairs"]]
    counts = {
        state: sum(row["state"] == state for row in rows)
        for state in ("planned", "attempted", "finished", "unobserved")
    }
    accounting = {
        "planned_invocations": 48,
        "observed_sessions": len(observed),
        "planned_sessions": 3,
        "known_attempted": counts["attempted"] + counts["finished"],
        "known_unattempted": counts["planned"],
        "unresolved_attempts": counts["attempted"],
        "unobserved_invocations": counts["unobserved"],
        "finished": counts["finished"],
        "outcomes": {
            outcome: sum(row.get("outcome") == outcome for row in rows)
            for outcome in sorted(OUTCOMES)
        },
        "planned_pairs": 24,
        "included_pairs": sum(pair["included"] for pair in pairs),
        "excluded_or_unobserved_pairs": sum(not pair["included"] for pair in pairs),
    }
    all_terminal = (
        len(observed) == 3
        and all(record["state"] in TERMINAL for record in records)
        and not counts["attempted"]
        and all(record["evidence"]["summary"] is not None for record in records)
    )
    complete = all_terminal and accounting["included_pairs"] == 24
    if desktop:
        complete = complete and all(record["state"] == "finished" for record in records)
    duration_key = "mean_ns" if desktop else "median_ns"
    workload_summaries = {}
    for name in bench.WORKLOADS:
        values = [
            pair["cpu_over_kleidiai"]
            for pair in pairs
            if pair["included"] and pair["workload"] == name
        ]
        by_session = []
        for record in records:
            selected = [
                pair
                for pair in record["pairs"]
                if pair["included"] and pair["workload"] == name
            ]
            by_session.append(
                {
                    "session": record["session"],
                    "planned_pairs": 2,
                    "cpu_over_kleidiai": distribution(
                        [pair["cpu_over_kleidiai"] for pair in selected]
                    ),
                    "process_" + duration_key: {
                        backend: distribution(
                            [pair[duration_key][backend] for pair in selected]
                        )
                        for backend in bench.BACKENDS
                    },
                }
            )
        workload_summaries[name] = {
            "planned_pairs": 6,
            "cpu_over_kleidiai": distribution(values),
            "sessions": by_session,
        }
    result = {
        "schema_version": 1,
        "kind": "desktop-fixed-token-report"
        if desktop
        else "controlled-fixed-token-report",
        "status": "complete" if complete else "incomplete",
        "accounting_complete": all_terminal,
        "provenance": {
            "plan": plan_pin,
            "reporter_sha256": reporter_pin["sha256"],
            **{
                key: plan[key]
                for key in (
                    "harness_sha256",
                    "runtime_helper_sha256",
                    "host_helper_sha256",
                )
            },
            "model": public_asset(plan["model"]),
            "runtime_config": public_asset(plan["runtime_config"]),
            "host_baseline": public_asset(plan["host_baseline"]),
            "prerequisites": {
                key: public_asset(item) for key, item in plan["prerequisites"].items()
            },
            "backends": {
                name: {
                    **{
                        key: public_asset(build[key])
                        for key in ("binary", "cache", "build_record")
                    },
                    "libraries": {
                        key: public_asset(item)
                        for key, item in build["libraries"].items()
                    },
                }
                for name, build in plan["backends"].items()
            },
        },
        "quality_comparison": plan["quality_comparison"],
        "protocol": {
            "native_revision": bench.REVISION,
            "seed": plan["seed"],
            "settings": plan["settings"],
            "flags": plan["flags"],
            "workloads": {
                name: {
                    "prompt_tokens": value[0],
                    "generated_tokens": value[1],
                    "prefilled_tokens": value[2],
                }
                for name, value in bench.WORKLOADS.items()
            },
            **{
                key: plan[key]
                for key in (
                    "repetitions",
                    "excluded_sample_indices",
                    "timeout_seconds",
                    "idle_before_session_seconds",
                    "idle_between_pairs_seconds",
                )
            },
        },
        "host_policy": {
            key: value
            for key, value in plan["host_policy"].items()
            if key != "identity"
        },
        "accounting": accounting,
        "workloads": workload_summaries,
        "sessions": records,
        "limitations": [
            "One ratio per adjacent CPU/KleidiAI process pair; six planned pairs "
            "per workload nested in three sessions, not eighteen independent samples.",
            "Each process retains four samples; index 0 is excluded and the upstream "
            "warmup remains enabled.",
            "Ratios divide the CPU median duration by the KleidiAI median duration. "
            "Values above one favor KleidiAI only for this frozen workload.",
            "Incomplete evidence retains every planned row and cannot establish a "
            "complete comparison. Unobserved session attempts are unknown.",
            "Host observations cover boundaries only. Unavailable thermal telemetry "
            "is not evidence of normal temperature; stable historical swap use is "
            "allowed, active swap-counter increases exclude a pair.",
            "Hashes bind supplied local evidence; they are not external attestation. "
            "No hardware-wide, dispatch or answer-quality claim follows from timing.",
        ],
    }
    if desktop:
        result["protocol_complete"] = complete
        result["provenance"]["method"] = public_asset(plan["method"])
        result["protocol"]["process_duration"] = "arithmetic_mean_of_retained_samples"
        result["limitations"][2] = (
            "Ratios divide the CPU mean duration by the KleidiAI mean duration, "
            "using the three retained samples. Values above one favor KleidiAI "
            "for this workload and observed desktop work period."
        )
        result["limitations"][4] = (
            "Valid completed pairs remain included with host findings. Swap-counter "
            "increases are recorded without filtering or timing adjustment. Other "
            "host departures stop new work and make protocol completion false. "
            "Boundary readings do not prove stable conditions or normal temperature."
        )
        result["limitations"].append(
            "These completion-conditioned descriptive ratios do not isolate backend "
            "effects from background activity. Consecutive session blocks are not "
            "independent replications. No significance test, population confidence "
            "interval or tail-latency estimate is supported."
        )
    privacy_check(result)
    evidence.check()
    require(
        bench.fingerprint(plan) == assets_before,
        "Pinned runtime changed during export.",
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument(
        "--session",
        nargs=3,
        action="append",
        required=True,
        metavar=("ID", "DIRECTORY", "RUN_SHA256"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = create_report(
        args.plan,
        args.plan_sha256,
        [
            {"session": int(number), "directory": Path(directory), "run_sha256": digest}
            for number, directory, digest in args.session
        ],
    )
    bench.save_json(args.output, value, create=True)


if __name__ == "__main__":
    main()
