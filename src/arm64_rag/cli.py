import argparse
import json
import sqlite3
import uuid
from pathlib import Path

from arm64_rag.corpus import load_manifest, load_paths
from arm64_rag.doctor import collect_inventory
from arm64_rag.evaluation import evaluate_retrieval
from arm64_rag.index import search_index, update_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Local RAG and Arm64 tools.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "doctor", help="Show hardware and tool versions without personal identifiers."
    )
    indexing = commands.add_parser("index", help="Index local text and Markdown.")
    indexing.add_argument("paths", type=Path, nargs="*")
    indexing.add_argument(
        "--manifest", type=Path, help="Read a pinned source manifest."
    )
    indexing.add_argument("--root", type=Path, default=Path.cwd())
    indexing.add_argument(
        "--sync",
        action="store_true",
        help="Also delete sources absent from this input.",
    )
    search = commands.add_parser("search", help="Find passages with source locations.")
    search.add_argument("query")
    search.add_argument(
        "--source", action="append", help="Limit search to a source ID."
    )
    evaluate = commands.add_parser("evaluate", help="Measure development retrieval.")
    evaluate.add_argument("--dataset", type=Path, default=Path("eval/development.json"))
    evaluate.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/corpus.json")
    )
    evaluate.add_argument("--token-budget", type=int, help="Budget in BGE text tokens.")
    ask = commands.add_parser(
        "ask", help="Answer a question with local source citations."
    )
    ask.add_argument("question")
    ask.add_argument("--source", action="append", help="Limit answers to a source ID.")
    ask.add_argument(
        "--server-binary", type=Path, default=Path("build/cpu/bin/llama-server")
    )
    ask.add_argument(
        "--answer-model-config", type=Path, default=Path("configs/answer-model.json")
    )
    ask.add_argument(
        "--protocol", type=Path, default=Path("eval/answers-protocol.json")
    )
    ask.add_argument(
        "--trace", type=Path, help="Save the full local answer record here."
    )
    ask.add_argument("--json", action="store_true", help="Print the answer as JSON.")
    embedding = commands.add_parser(
        "embed", help="Add embeddings to the document index."
    )
    for command in (search, evaluate, ask):
        command.add_argument(
            "--method", choices=("lexical", "dense", "hybrid"), default="lexical"
        )
    for command in (embedding, search, evaluate, ask):
        command.add_argument(
            "--model-dir", type=Path, default=Path("models/bge-small-en-v1.5")
        )
        command.add_argument(
            "--embedding-config", type=Path, default=Path("configs/embeddings.json")
        )
    for command in (indexing, embedding, search, evaluate, ask):
        command.add_argument(
            "--db", type=Path, default=Path("data/private/index.sqlite")
        )
    for command in (search, evaluate):
        command.add_argument("--limit", type=int, default=5, help="Maximum passages.")
    args = parser.parse_args()
    try:
        encoder = None
        if args.command == "embed" or (
            args.command in {"search", "evaluate", "ask"}
            and (
                args.method != "lexical"
                or getattr(args, "token_budget", None) is not None
            )
        ):
            from arm64_rag.embeddings import BGEEncoder

            encoder = BGEEncoder(args.model_dir, args.embedding_config)
        if args.command == "doctor":
            result = collect_inventory()
        elif args.command == "index":
            if bool(args.paths) == bool(args.manifest):
                parser.error("Provide document paths or --manifest.")
            documents = (
                load_manifest(args.manifest, args.root)
                if args.manifest
                else load_paths(args.paths, args.root)
            )
            result = update_index(args.db, documents, sync=args.sync)
        elif args.command == "embed":
            from arm64_rag.dense import embed_index

            result = embed_index(args.db, encoder.encode_documents, encoder.config)
        elif args.command == "search":
            if args.method == "lexical":
                result = search_index(
                    args.db, args.query, limit=args.limit, source_ids=args.source
                )
            else:
                from arm64_rag.dense import search_vectors

                result = search_vectors(
                    args.db,
                    args.query,
                    encoder.encode_queries,
                    encoder.config,
                    method=args.method,
                    limit=args.limit,
                    source_ids=args.source,
                )
        elif args.command == "ask":
            from arm64_rag.answers import answer_question, render_answer
            from arm64_rag.inference import LocalServer

            trace = args.trace or Path("artifacts/answers") / f"{uuid.uuid4().hex}.json"
            if trace.exists() or trace.with_suffix(".server.log").exists():
                raise ValueError("Use a new trace path to preserve earlier answers.")
            runtime, interruption = None, None
            try:
                try:
                    protocol = json.loads(args.protocol.read_bytes())
                    model_config = json.loads(args.answer_model_config.read_bytes())
                    settings = protocol["settings"]
                    model = (
                        Path(model_config["directory"])
                        / model_config["model"]["filename"]
                    )
                    model_hash = model_config["model"]["sha256"]
                    if (
                        not isinstance(settings, dict)
                        or not isinstance(protocol["model_output_schema"], dict)
                        or not isinstance(model_hash, str)
                        or not model_hash
                        or any(
                            type(settings.get(key)) is not int or settings[key] < 1
                            for key in (
                                "context_size",
                                "max_output_tokens",
                                "max_passages",
                            )
                        )
                    ):
                        raise ValueError
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        "Invalid answer protocol or model configuration."
                    ) from error
                with LocalServer(
                    args.server_binary,
                    model,
                    context_size=protocol["settings"]["context_size"],
                    request_timeout=300,
                    reasoning_budget_tokens=protocol["settings"].get(
                        "reasoning_budget_tokens"
                    ),
                    log_path=trace.with_suffix(".server.log"),
                ) as server:
                    runtime = server.info()
                    if runtime["model_sha256"] != model_hash:
                        raise ValueError(
                            "Answer model differs from its configured hash."
                        )
                    result = answer_question(
                        args.db,
                        args.question,
                        server,
                        protocol,
                        method=args.method,
                        encoder=encoder,
                        source_ids=args.source,
                    )
            except (
                OSError,
                ValueError,
                RuntimeError,
                sqlite3.Error,
                KeyboardInterrupt,
            ) as error:
                if isinstance(error, KeyboardInterrupt):
                    interruption = error
                result = {
                    "schema_version": 1,
                    "question": args.question,
                    "runtime": runtime,
                    "answer": None,
                    "failure": "Interrupted by user."
                    if interruption is not None
                    else str(error),
                    "raw_completion": getattr(error, "response", None),
                }
            trace.parent.mkdir(parents=True, exist_ok=True)
            with trace.open("x") as output:
                output.write(json.dumps(result, ensure_ascii=True, indent=2) + "\n")
            if interruption is not None:
                raise interruption
            if args.json:
                print(
                    json.dumps(
                        {"answer": result["answer"], "failure": result["failure"]},
                        ensure_ascii=True,
                        indent=2,
                    )
                )
            else:
                print(render_answer(result))
            if result["failure"]:
                parser.exit(1, f"Details saved to {trace}.\n")
            return
        else:
            result = evaluate_retrieval(
                args.db,
                args.dataset,
                args.manifest,
                limit=args.limit,
                method=args.method,
                encoder=encoder,
                token_budget=args.token_budget,
            )
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    if args.command == "evaluate" and result["summary"]["cases_failed"]:
        parser.exit(1)


if __name__ == "__main__":
    main()
