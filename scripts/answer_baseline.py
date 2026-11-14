"""Require unchanged inputs when comparing answer backends."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

ROLES = ("development", "regression", "validation")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


class BaselineComparison:
    def __init__(self, root, directory, role, run, protocol_raw):
        path = root / "docs/results/grounded-answers" / f"{role}.json"
        report_raw = path.read_bytes()
        report = json.loads(report_raw)
        self._load(directory, role, run, protocol_raw, report, report_raw)

    @classmethod
    def from_local(cls, path, expected_sha256, role, run, protocol_raw):
        raw = path.read_bytes()
        require(digest(raw) == expected_sha256, "Local baseline manifest changed.")
        manifest = json.loads(raw)
        require(
            manifest.get("schema_version") == 1
            and manifest.get("kind") == "local-answer-input-baseline"
            and manifest.get("semantic_review") == "not_assessed",
            "Unknown local baseline manifest.",
        )
        directory = (path.parent / manifest["run_directory"]).resolve(strict=True)
        require(
            manifest == local_manifest(directory, role, path.parent),
            "Local baseline inputs changed since freeze.",
        )
        comparison = cls.__new__(cls)
        comparison._load(directory, role, run, protocol_raw, manifest, raw)
        comparison.metadata.pop("report_sha256")
        comparison.metadata.update(
            local_manifest_sha256=expected_sha256,
            baseline_kind="local_input_equality",
            semantic_review="not_assessed",
        )
        return comparison

    def _load(self, directory, role, run, protocol_raw, report, report_raw):
        baseline_raw = (directory / "run.json").read_bytes()
        baseline = json.loads(baseline_raw)
        require(
            digest(baseline_raw) == report["run_file_sha256"],
            "Baseline run differs from the published evidence.",
        )
        require(
            report["role"] == role
            and report["dataset_sha256"] == run["datasets"][role]["sha256"]
            and report["protocol_file_sha256"] == digest(protocol_raw)
            and report["system_prompt_sha256"] == run["system_prompt_sha256"]
            and report["runtime"]["model_sha256"] == run["model_sha256"],
            "Baseline questions, protocol, prompt or model differ.",
        )
        old_code = baseline["implementation_sha256"]
        new_code = run["implementation_sha256"]
        require(set(old_code) == set(new_code), "Application file inventory differs.")
        require(
            all(
                old_code[name] == new_code[name]
                for name in old_code
                if name != "scripts/check_answers.py"
            ),
            "Application code changed since the answer baseline.",
        )
        require(
            baseline["state"] == "completed"
            and baseline["planned"] == baseline["completed"] == run["planned"],
            "Baseline case order or completion differs.",
        )
        self.cases, case_hashes = {}, {}
        for case in report["cases"]:
            raw = (directory / f"{case['id']}.json").read_bytes()
            require(
                digest(raw) == case["case_file_sha256"],
                f"Baseline case changed: {case['id']}",
            )
            record = json.loads(raw)
            require(
                record["id"] == case["id"] and record["failure"] is None,
                "Baseline contains an invalid case record.",
            )
            self.cases[case["id"]] = record
            case_hashes[case["id"]] = digest(raw)
        require(list(self.cases) == run["planned"], "Published baseline cases differ.")
        self.schema = json.loads(protocol_raw)["model_output_schema"]
        self.runtime = baseline["runtime"]
        self.current, self.server = None, None
        self.metadata = {
            "report_sha256": digest(report_raw),
            "run_sha256": digest(baseline_raw),
            "case_sha256": case_hashes,
            "runner_change": {
                "baseline": old_code["scripts/check_answers.py"],
                "current": new_code["scripts/check_answers.py"],
            },
            "scope": "Input equality only; changed answers require source review.",
        }

    def bind(self, server):
        allowed = {"binary_sha256", "sibling_libraries_sha256", "startup_seconds"}
        observed = server.info()
        require(
            {k: v for k, v in observed.items() if k not in allowed}
            == {k: v for k, v in self.runtime.items() if k not in allowed},
            "Backend model, template or generation settings differ.",
        )
        self.server = server
        return self

    def __getattr__(self, name):
        return getattr(self.server, name)

    def start(self, case):
        self.current = self.cases[case["id"]]
        require(
            case["question"] == self.current["question"],
            "Question differs from the baseline.",
        )

    def complete(self, token_ids, max_tokens, *, json_schema=None, messages=None):
        require(self.current is not None, "No baseline case selected.")
        packed = self.current["packed"]
        require(
            token_ids == packed["token_ids"]
            and max_tokens == packed["max_output_tokens"]
            and json.dumps(json_schema) == json.dumps(self.schema),
            "Token IDs, output budget or schema differ from the baseline.",
        )
        prompt = self.server.render(messages, json_schema=json_schema)
        require(prompt == packed["prompt"], "Rendered prompt differs from baseline.")
        return self.server.complete(
            token_ids, max_tokens, json_schema=json_schema, messages=messages
        )

    def retain(self, record):
        expected = self.cases[record["id"]]
        mismatches = [
            key
            for key in ("question", "retrieval", "packed", "evidence")
            if record.get(key) != expected.get(key)
        ]
        record["baseline_comparison"] = {
            "case_sha256": self.metadata["case_sha256"][record["id"]],
            "inputs_match": not mismatches,
            "mismatched_fields": mismatches,
            "answer_identical": record["answer"] == expected["answer"],
        }
        if mismatches:
            earlier = record["failure"]
            record["answer"] = None
            record["failure"] = "Inputs differ from the answer baseline."
            if earlier:
                record["failure"] += f" Earlier failure: {earlier}"


def local_manifest(directory, role, parent):
    require(role in ROLES, "Choose one answer split.")
    directory = directory.resolve(strict=True)
    run_raw = (directory / "run.json").read_bytes()
    run = json.loads(run_raw)
    planned = run["planned"]
    require(
        run.get("backend") == "cpu"
        and run["state"] == "completed"
        and run.get("failure") is None
        and run.get("cases_failed") == 0
        and run.get("active_case") is None
        and isinstance(planned, list)
        and bool(planned)
        and run["attempted"] == run["completed"] == planned
        and set(run["datasets"]) == {role},
        "Freeze a complete, successful CPU run for one split.",
    )
    require(
        all(
            isinstance(case, str) and re.fullmatch(r"[A-Za-z0-9_-]+", case)
            for case in planned
        )
        and len(set(planned)) == len(planned),
        "Baseline case IDs are invalid or repeated.",
    )
    protocol_raw = (directory / "answers-protocol.json").read_bytes()
    prompt_raw = (directory / "system-prompt.txt").read_bytes()
    require(
        digest(protocol_raw) == run["protocol_sha256"]
        and digest(prompt_raw) == run["system_prompt_sha256"]
        and run["runtime"]["model_sha256"] == run["model_sha256"],
        "Baseline protocol, prompt or model records differ.",
    )
    cases = []
    for case_id in planned:
        raw = (directory / f"{case_id}.json").read_bytes()
        case = json.loads(raw)
        require(
            case["id"] == case_id
            and case["failure"] is None
            and isinstance(case["answer"], dict),
            "Baseline contains a failed case.",
        )
        cases.append({"id": case_id, "case_file_sha256": digest(raw)})
    return {
        "schema_version": 1,
        "kind": "local-answer-input-baseline",
        "semantic_review": "not_assessed",
        "role": role,
        "run_directory": os.path.relpath(directory, parent.resolve()),
        "run_file_sha256": digest(run_raw),
        "dataset_sha256": run["datasets"][role]["sha256"],
        "protocol_file_sha256": digest(protocol_raw),
        "system_prompt_sha256": run["system_prompt_sha256"],
        "runtime": run["runtime"],
        "cases": cases,
        "scope": (
            "Input equality only. Review CPU and candidate answers "
            "against their sources."
        ),
    }


def freeze_local(directory, role, destination):
    manifest = local_manifest(directory, role, destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write(
            json.dumps(manifest, ensure_ascii=True, allow_nan=False, indent=2) + "\n"
        )
    return digest(destination.read_bytes())


def main():
    parser = argparse.ArgumentParser(description="Freeze a local CPU answer baseline.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", choices=ROLES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(freeze_local(args.run, args.split, args.output))


if __name__ == "__main__":
    main()
