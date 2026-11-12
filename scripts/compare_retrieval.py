"""Run the frozen retrieval comparison without changing its settings."""

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

from arm64_rag.dense import embedding_info
from arm64_rag.embeddings import BGEEncoder
from arm64_rag.evaluation import evaluate_retrieval

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(
        description="Compare keyword, dense and hybrid retrieval."
    )
    parser.add_argument(
        "--db", type=Path, default=ROOT / "data/private/development.sqlite"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/retrieval/embedding-comparison.json",
    )
    args = parser.parse_args()
    args.output.unlink(missing_ok=True)
    config_path = ROOT / "configs/retrieval-comparison.json"
    config_raw = config_path.read_bytes()
    config = json.loads(config_raw)
    pinned = [
        *config["datasets"],
        {"path": config["manifest"], "sha256": config["manifest_sha256"]},
    ]
    for source in pinned:
        if (
            hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest()
            != source["sha256"]
        ):
            parser.error("Comparison inputs differ from the frozen pins.")
    if config["candidate_depth"] != 20 or config["rrf_constant"] != 60:
        parser.error("Unsupported comparison policy.")
    started = time.perf_counter()
    encoder = BGEEncoder(
        ROOT / "models/bge-small-en-v1.5", ROOT / "configs/embeddings.json"
    )
    initial = embedding_info(args.db, encoder.config)
    report = {
        "schema_version": 1,
        "protocol": config,
        "protocol_sha256": hashlib.sha256(config_raw).hexdigest(),
        "encoder_load_seconds": time.perf_counter() - started,
        "embedding": initial,
        "runs": [],
        "comparisons": [],
    }
    for dataset in config["datasets"]:
        runs = {}
        for method in config["methods"]:
            result = evaluate_retrieval(
                args.db,
                ROOT / dataset["path"],
                ROOT / config["manifest"],
                limit=config["passage_limit"],
                method=method,
                encoder=encoder,
                token_budget=config["token_budget"],
            )
            if (
                result["index"]["fingerprint"] != initial["source_fingerprint"]
                or result["index"]["generation"] != initial["source_generation"]
                or (
                    "embedding" in result
                    and result["embedding"]["fingerprint"] != initial["fingerprint"]
                )
                or result["dataset_sha256"] != dataset["sha256"]
                or result["manifest_sha256"] != config["manifest_sha256"]
            ):
                parser.error("Comparison inputs changed between runs.")
            runs[method] = result
            report["runs"].append(result)
        baseline = {case["id"]: case for case in runs["lexical"]["cases"]}
        for method in ("dense", "hybrid"):
            wins, losses = [], []
            for case in runs[method]["cases"]:
                before = baseline[case["id"]]["all_covered"]
                after = case["all_covered"]
                if before is False and after is True:
                    wins.append(case["id"])
                if before is True and after is False:
                    losses.append(case["id"])
            report["comparisons"].append(
                {
                    "dataset": runs[method]["dataset_id"],
                    "method": method,
                    "fully_covered_wins": wins,
                    "fully_covered_losses": losses,
                }
            )
    if embedding_info(args.db, encoder.config)["fingerprint"] != initial["fingerprint"]:
        parser.error("Embeddings changed during the comparison.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "summaries": [
                    {
                        "dataset": run["dataset_id"],
                        "method": run["method"],
                        **run["summary"],
                    }
                    for run in report["runs"]
                ],
                "comparisons": report["comparisons"],
            },
            indent=2,
        )
    )
    if any(run["summary"]["cases_failed"] for run in report["runs"]):
        parser.exit(1)


if __name__ == "__main__":
    main()
