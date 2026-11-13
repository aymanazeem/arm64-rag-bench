"""Record answer runs for independent factual and citation review."""

import argparse
import hashlib
import json
from pathlib import Path

from arm64_rag.answers import SYSTEM_PROMPT
from arm64_rag.inference import LocalServer
from arm64_rag.qa_evaluation import evaluate_answers

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Run local answer checks.")
    parser.add_argument(
        "--split",
        choices=("development", "regression", "validation", "all"),
        default="development",
        help="Choose development, spent regression, fresh validation, or all three.",
    )
    parser.add_argument(
        "--db", type=Path, default=ROOT / "data/private/development.sqlite"
    )
    parser.add_argument("--backend", choices=("cpu", "kleidiai"), default="cpu")
    parser.add_argument(
        "--baseline", type=Path, help="Compare with a published CPU run directory."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Use a new output directory for each run.",
    )
    args = parser.parse_args()
    if args.baseline is not None and args.split == "all":
        parser.error("Choose one split when comparing with a baseline run.")
    if args.output.exists():
        parser.error("Use a new output directory to preserve earlier attempts.")
    args.output.mkdir(parents=True)
    protocol_raw = (ROOT / "eval/answers-protocol.json").read_bytes()
    protocol = json.loads(protocol_raw)
    config = json.loads((ROOT / "configs/answer-model.json").read_bytes())
    asset = config["model"]
    model = ROOT / config["directory"] / asset["filename"]
    datasets = {
        "development": ROOT / "eval/development.json",
        "regression": ROOT / "eval/retrieval-validation.json",
        "validation": ROOT / "eval/answer-validation.json",
    }
    selected = list(datasets) if args.split == "all" else [args.split]
    inputs, planned = {}, []
    for split in selected:
        raw = datasets[split].read_bytes()
        inputs[split] = {
            "path": str(datasets[split].relative_to(ROOT)),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        planned.extend(case["id"] for case in json.loads(raw)["cases"])
    if len(set(planned)) != len(planned):
        parser.error("Selected datasets contain repeated case IDs.")
    failed = 0
    (args.output / "answers-protocol.json").write_bytes(protocol_raw)
    (args.output / "system-prompt.txt").write_text(SYSTEM_PROMPT)
    run = {
        "state": "starting",
        "backend": args.backend,
        "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "protocol_hash_format": "Raw file bytes.",
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "model_sha256": asset["sha256"],
        "implementation_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                *sorted((ROOT / "src/arm64_rag").glob("*.py")),
                ROOT / "scripts/check_answers.py",
                ROOT / "uv.lock",
            ]
            if path.is_file()
        },
        "datasets": inputs,
        "planned": planned,
        "attempted": [],
        "completed": [],
        "active_case": None,
        "failure": None,
    }
    write_json(args.output / "run.json", run)
    comparison = None

    def start_case(case):
        run["state"] = "running"
        run["active_case"] = case["id"]
        run["attempted"].append(case["id"])
        write_json(args.output / "run.json", run)
        if comparison is not None:
            comparison.start(case)

    def retain_case(record):
        if comparison is not None:
            comparison.retain(record)
        write_json(args.output / f"{record['id']}.json", record)
        run["completed"].append(record["id"])
        run["active_case"] = None
        write_json(args.output / "run.json", run)
        status = record["answer"]["status"] if record["answer"] else "failed"
        print(f"{record['id']}: {status}", flush=True)

    try:
        if args.baseline is not None:
            from answer_baseline import BaselineComparison

            comparison = BaselineComparison(
                ROOT, args.baseline, args.split, run, protocol_raw
            )
            run["baseline_comparison"] = {
                **comparison.metadata,
                "helper_sha256": hashlib.sha256(
                    (ROOT / "scripts/answer_baseline.py").read_bytes()
                ).hexdigest(),
            }
            write_json(args.output / "run.json", run)
        with LocalServer(
            ROOT / f"build/{args.backend}/bin/llama-server",
            model,
            context_size=protocol["settings"]["context_size"],
            request_timeout=300,
            reasoning_budget_tokens=protocol["settings"].get("reasoning_budget_tokens"),
            log_path=args.output / "server.log",
        ) as server:
            run["runtime"] = server.info()
            write_json(args.output / "run.json", run)
            if server.info()["model_sha256"] != asset["sha256"]:
                raise ValueError("The answer model differs from its pin.")
            evaluator = comparison.bind(server) if comparison is not None else server
            for split in selected:
                report = evaluate_answers(
                    args.db,
                    datasets[split],
                    ROOT / protocol["manifest"]["path"],
                    evaluator,
                    protocol,
                    on_case_start=start_case,
                    on_case=retain_case,
                )
                write_json(args.output / f"{split}.json", report)
                print(json.dumps({"split": split, **report["summary"]}), flush=True)
                failed += report["summary"]["cases_failed"]
    except BaseException as error:
        run["state"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        run["failure"] = f"{type(error).__name__}: {error}"
        write_json(args.output / "run.json", run)
        raise
    run["state"] = "completed"
    run["cases_failed"] = failed
    write_json(args.output / "run.json", run)
    if failed:
        parser.exit(1, "Some answers failed validation; inspect the saved records.\n")


if __name__ == "__main__":
    main()
