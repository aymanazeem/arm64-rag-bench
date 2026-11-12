import argparse
import json

from arm64_rag.doctor import collect_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description="Local RAG and Arm64 tools.")
    parser.add_subparsers(dest="command", required=True).add_parser(
        "doctor", help="Show hardware and tool versions without personal identifiers."
    )
    parser.parse_args()
    print(json.dumps(collect_inventory(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
