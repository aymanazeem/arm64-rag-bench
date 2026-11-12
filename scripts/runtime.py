"""Build the controlled CPU pair and fetch its checksum-pinned smoke model."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("llama-bench", "llama-server", "llama-completion")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path, asset):
    if path.stat().st_size != asset["size"] or digest(path) != asset["sha256"]:
        raise ValueError(f"Checksum or size mismatch: {path.name}")


def download(asset, destination):
    if destination.exists():
        verify(destination, asset)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with partial.open("xb") as output:
        try:
            with urllib.request.urlopen(asset["url"], timeout=60) as response:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            output.flush()
            verify(partial, asset)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
    partial.replace(destination)
    return destination


def capture(*args):
    return subprocess.check_output(args, text=True, cwd=ROOT).strip()


def source_checkout(config):
    source = ROOT / "vendor/llama.cpp"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="llama-fetch-", dir=source.parent
        ) as tmp:
            checkout = Path(tmp) / "source"
            subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
            for arguments in (
                ["remote", "add", "origin", config["repository"]],
                ["fetch", "--depth=1", "origin", config["revision"]],
                ["checkout", "--detach", "FETCH_HEAD"],
            ):
                subprocess.run(["git", "-C", str(checkout), *arguments], check=True)
            check_source(checkout, config["revision"])
            checkout.rename(source)
    check_source(source, config["revision"])
    return source


def check_source(source, expected):
    if not (source / ".git").exists():
        raise ValueError("Existing llama.cpp source is not a Git checkout.")
    revision = capture("git", "-C", str(source), "rev-parse", "HEAD")
    dirty = capture("git", "-C", str(source), "status", "--porcelain")
    if revision != expected or dirty:
        raise ValueError("llama.cpp must be clean and at the configured revision.")


def tree_hashes(directory):
    result = {}
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("KleidiAI source must not contain symlinks.")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = digest(path)
    return result


def kleidiai_source(asset):
    archive = download(asset, ROOT / ".local/downloads" / asset["filename"])
    destination = ROOT / "vendor" / f"kleidiai-{asset['version']}"
    with tempfile.TemporaryDirectory(dir=ROOT / ".local") as temporary:
        with tarfile.open(archive) as bundle:
            if any(not (member.isfile() or member.isdir()) for member in bundle):
                raise ValueError("Unexpected links or devices in the KleidiAI archive.")
            bundle.extractall(temporary, filter="data")
        entries = list(Path(temporary).iterdir())
        source = (
            entries[0] if len(entries) == 1 and entries[0].is_dir() else Path(temporary)
        )
        if not (source / "CMakeLists.txt").is_file():
            raise ValueError("KleidiAI archive has no build definition.")
        if destination.exists():
            if tree_hashes(source) != tree_hashes(destination):
                raise ValueError(
                    "Existing KleidiAI source differs from the verified archive."
                )
        else:
            shutil.copytree(source, destination)
    return destination


def logged(command, path):
    print(f"Running {Path(command[0]).name}; log: {path.relative_to(ROOT)}", flush=True)
    with path.open("w") as output:
        subprocess.run(
            command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=True
        )


def checked_cache(path, flags):
    cache = {}
    for line in path.read_text().splitlines():
        if not line.startswith(("#", "//")) and ":" in line and "=" in line:
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    for key, expected in flags.items():
        if cache.get(key) != expected:
            raise ValueError(f"Configured value differs from the requested {key}.")
    if cache.get("GGML_AVAILABLE_BACKENDS") != "ggml-cpu":
        raise ValueError("The controlled build must use only the CPU backend.")
    return cache


def build(config, backend, jobs):
    artifacts = ROOT / "artifacts/feasibility" / backend
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts / "build.json"
    manifest_path.unlink(missing_ok=True)
    if jobs < 1:
        raise ValueError("Jobs must be at least 1.")
    for name in ("CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "CMAKE_TOOLCHAIN_FILE"):
        if os.environ.get(name):
            raise ValueError(f"Unset {name} for the controlled build.")
    for tool in ("git", "cmake", "ninja", "clang"):
        if not shutil.which(tool):
            raise ValueError(f"Missing {tool}. Run through uv with the build group.")
    source = source_checkout(config["llama_cpp"])
    flags = dict(
        config["cmake"], GGML_CPU_KLEIDIAI="ON" if backend == "kleidiai" else "OFF"
    )
    if backend == "kleidiai":
        flags["FETCHCONTENT_SOURCE_DIR_KLEIDIAI"] = str(
            kleidiai_source(config["kleidiai"])
        )
    build_dir = ROOT / "build" / backend
    configure = [
        "cmake",
        "--fresh",
        "-S",
        str(source),
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
    ]
    configure.extend(f"-D{key}={value}" for key, value in flags.items())
    logged(configure, artifacts / "configure.log")
    for name in ("CMakeCache.txt", "compile_commands.json"):
        shutil.copy2(build_dir / name, artifacts / name)
    cache = checked_cache(build_dir / "CMakeCache.txt", flags)
    logged(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--parallel",
            str(jobs),
            "--target",
            *TARGETS,
        ],
        artifacts / "build.log",
    )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "backend": backend,
        "source": config["llama_cpp"],
        "kleidiai": config["kleidiai"] if backend == "kleidiai" else None,
        "flags": {
            key: cache[key] for key in flags if not key.startswith("FETCHCONTENT_")
        },
        "compiler_flags": {
            key: value
            for key, value in cache.items()
            if key.startswith("CMAKE_")
            and "_FLAGS" in key
            and not key.endswith("-ADVANCED")
        },
        "backends": cache["GGML_AVAILABLE_BACKENDS"].split(";"),
        "cache_sha256": digest(artifacts / "CMakeCache.txt"),
        "compile_commands_sha256": digest(artifacts / "compile_commands.json"),
        "system": {
            "os": platform.system(),
            "release": platform.release(),
            "arch": platform.machine(),
        },
        "tools": {
            tool: capture(tool, "--version").splitlines()[0]
            for tool in ("cmake", "ninja")
        },
        "compilers": {
            language: capture(
                cache[f"CMAKE_{language}_COMPILER"], "--version"
            ).splitlines()[0]
            for language in ("C", "CXX")
        },
        "targets": {name: digest(build_dir / "bin" / name) for name in TARGETS},
        "libraries": {
            path.name: digest(path)
            for path in (build_dir / "bin").iterdir()
            if path.is_file() and (".so" in path.name or path.suffix == ".dylib")
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built {backend}. Manifest: {manifest_path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="Build one CPU backend.")
    build_parser.add_argument("backend", choices=("cpu", "kleidiai"))
    build_parser.add_argument("--jobs", type=int, default=4)
    commands.add_parser("download-model", help="Fetch and verify the smoke model.")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/runtime.json").read_text())
    try:
        if args.command == "build":
            build(config, args.backend, args.jobs)
        else:
            path = download(
                config["model"], ROOT / "models" / config["model"]["filename"]
            )
            print(f"Verified model: {path.relative_to(ROOT)}")
    except (
        OSError,
        ValueError,
        tarfile.TarError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
