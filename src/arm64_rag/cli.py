import argparse
import json
import sqlite3
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
    for command in (indexing, search, evaluate):
        command.add_argument(
            "--db", type=Path, default=Path("data/private/index.sqlite")
        )
    for command in (search, evaluate):
        command.add_argument("--limit", type=int, default=5, help="Maximum passages.")
    args = parser.parse_args()
    try:
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
        elif args.command == "search":
            result = search_index(
                args.db, args.query, limit=args.limit, source_ids=args.source
            )
        else:
            result = evaluate_retrieval(
                args.db, args.dataset, args.manifest, limit=args.limit
            )
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    if args.command == "evaluate" and result["summary"]["cases_failed"]:
        parser.exit(1)


if __name__ == "__main__":
    main()
