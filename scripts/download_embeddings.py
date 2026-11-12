"""Fetch checksum-pinned embedding assets; never fetch during search."""

import argparse
import json

from runtime import ROOT, download


def main():
    parser = argparse.ArgumentParser(description="Download the pinned BGE model.")
    parser.add_argument(
        "--reference", action="store_true", help="Also fetch reference weights."
    )
    args = parser.parse_args()
    manifest = json.loads((ROOT / "configs/embeddings.json").read_bytes())
    destination = ROOT / "models/bge-small-en-v1.5"
    for asset in manifest["files"]:
        if asset["role"] == "reference" and not args.reference:
            continue
        url = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{asset['path']}"
        download({**asset, "url": url}, destination / asset["path"])
        print(f"Verified {asset['path']}")


if __name__ == "__main__":
    main()
