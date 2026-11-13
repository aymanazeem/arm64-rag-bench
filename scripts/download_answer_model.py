"""Fetch the pinned answer model and preserve its upstream notices."""

import argparse
import json
import ssl
import subprocess
import urllib.error

from runtime import ROOT, download, publish_verified, verify


def download_with_curl(asset, destination):
    if destination.exists():
        verify(destination, asset)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.touch(exist_ok=False)
    try:
        subprocess.run(
            [
                "curl",
                "--disable",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--connect-timeout",
                "30",
                "--max-time",
                "1200",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--output",
                str(partial),
                asset["url"],
            ],
            check=True,
        )
        verify(partial, asset)
        return publish_verified(partial, destination, asset)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def download_asset(asset, destination):
    try:
        return download(asset, destination)
    except urllib.error.URLError as error:
        if not isinstance(error.reason, ssl.SSLCertVerificationError):
            raise
        return download_with_curl(asset, destination)


def main():
    parser = argparse.ArgumentParser(description="Download the pinned answer model.")
    parser.parse_args()
    config = json.loads((ROOT / "configs/answer-model.json").read_bytes())
    destination = ROOT / config["directory"]
    try:
        for asset in config["files"]:
            download_asset(asset, destination / asset["filename"])
            print(f"Verified {asset['filename']}", flush=True)
        download_asset(config["model"], destination / config["model"]["filename"])
        print(f"Verified {config['model']['filename']}", flush=True)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
