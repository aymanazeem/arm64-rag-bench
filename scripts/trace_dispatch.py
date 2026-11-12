"""Prove Q4_0 kernel execution in a separate build; its timings are invalid."""

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import runtime

KERNELS = {
    "prefill": "kai_run_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod",
    "decode": "kai_run_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod",
}
PARAMETERS = (
    "size_t m, size_t n, size_t k, size_t bl, const void* restrict lhs_packed, "
    "const void* restrict rhs_packed, float* restrict dst, size_t dst_stride_row, "
    "size_t dst_stride_col, float scalar_min, float scalar_max"
)


def instrument(text, name):
    matches = list(re.finditer(rf"\bvoid {re.escape(name)}\(([^{{}};]+)\)\s*\{{", text))
    if len(matches) != 1:
        raise ValueError(f"Expected one kernel definition: {name}")
    match = matches[0]
    parameters = re.sub(r"//[^\n]*", "", match.group(1))
    if re.sub(r"\s", "", parameters) != re.sub(r"\s", "", PARAMETERS):
        raise ValueError(f"Unexpected kernel signature: {name}")
    marker = (
        '\n    fprintf(stderr, "KAI_TRACE %s m=%zu n=%zu k=%zu\\n", __func__, m, n, k);'
    )
    return "#include <stdio.h>\n" + text[: match.end()] + marker + text[match.end() :]


def tree_digest(source):
    content = json.dumps(runtime.tree_hashes(source), sort_keys=True).encode()
    return hashlib.sha256(content).hexdigest()


def prepare_source(source, destination, patch_path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    changes, files = [], {}
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staged = Path(temporary) / "source"
        shutil.copytree(source, staged)
        for name in KERNELS.values():
            paths = list(staged.rglob(name.replace("kai_run_", "kai_") + ".c"))
            if len(paths) != 1:
                raise ValueError(f"Expected one kernel source: {name}")
            path = paths[0]
            relative = path.relative_to(staged).as_posix()
            original = path.read_text()
            modified = instrument(original, name)
            path.write_text(modified)
            changes.extend(
                difflib.unified_diff(
                    original.splitlines(True),
                    modified.splitlines(True),
                    fromfile=f"a/{relative}",
                    tofile=f"b/{relative}",
                )
            )
            files[relative] = {
                "original_sha256": runtime.digest(source / relative),
                "traced_sha256": runtime.digest(path),
            }
        traced_hash = tree_digest(staged)
        if destination.exists():
            if tree_digest(destination) != traced_hash:
                raise ValueError(
                    "Existing trace source differs from the expected patch."
                )
        else:
            staged.rename(destination)
    patch_path.write_text("".join(changes))
    return {
        "original_tree_sha256": tree_digest(source),
        "traced_tree_sha256": traced_hash,
        "patch_sha256": runtime.digest(patch_path),
        "files": files,
    }


def parse_dispatch(stderr, phase):
    name = KERNELS[phase]
    pattern = rf"^KAI_TRACE {re.escape(name)} m=(\d+) n=(\d+) k=(\d+)$"
    shapes = [
        tuple(map(int, match)) for match in re.findall(pattern, stderr, re.MULTILINE)
    ]
    expected_m = 32 if phase == "prefill" else 1
    if not shapes or any(m != expected_m or n < 1 or k < 1 for m, n, k in shapes):
        raise ValueError(f"No valid {phase} kernel evidence for m={expected_m}.")
    return {"kernel": name, "calls": len(shapes), "shapes_m_n_k": sorted(set(shapes))}


def trace(config, jobs, timeout):
    root = runtime.ROOT
    artifacts = root / "artifacts/feasibility/dispatch"
    artifacts.mkdir(parents=True, exist_ok=True)
    summary_path = artifacts / "summary.json"
    summary_path.unlink(missing_ok=True)
    if jobs < 1 or timeout <= 0:
        raise ValueError("Jobs and timeout must be positive.")
    for name in ("CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "CMAKE_TOOLCHAIN_FILE"):
        if os.environ.get(name):
            raise ValueError(f"Unset {name} for the diagnostic build.")
    source = runtime.source_checkout(config["llama_cpp"])
    clean_kai = runtime.kleidiai_source(config["kleidiai"])
    model = root / "models" / config["model"]["filename"]
    runtime.verify(model, config["model"])
    traced = root / ".local/kleidiai-trace"
    patch = prepare_source(clean_kai, traced, artifacts / "kernels.patch")
    flags = dict(
        config["cmake"],
        GGML_CPU_KLEIDIAI="ON",
        FETCHCONTENT_SOURCE_DIR_KLEIDIAI=str(traced),
    )
    build = root / "build/kleidiai-trace"
    configure = ["cmake", "--fresh", "-S", str(source), "-B", str(build), "-G", "Ninja"]
    configure.extend(f"-D{key}={value}" for key, value in flags.items())
    runtime.logged(configure, artifacts / "configure.log")
    cache = runtime.checked_cache(build / "CMakeCache.txt", flags)
    for name in ("CMakeCache.txt", "compile_commands.json"):
        shutil.copy2(build / name, artifacts / name)
    runtime.logged(
        [
            "cmake",
            "--build",
            str(build),
            "--parallel",
            str(jobs),
            "--target",
            "llama-bench",
        ],
        artifacts / "build.log",
    )
    executable = build / "bin/llama-bench"
    results = {}
    for phase, lengths in (("prefill", (32, 0)), ("decode", (0, 8))):
        command = [
            str(executable.relative_to(root)),
            "-m",
            str(model.relative_to(root)),
            "--no-warmup",
            "-p",
            str(lengths[0]),
            "-n",
            str(lengths[1]),
            "-r",
            "1",
            "-ngl",
            "0",
            "-t",
            "1",
            "-b",
            "32",
            "-ub",
            "32",
        ]
        with (
            (artifacts / f"{phase}.stdout").open("w") as out,
            (artifacts / f"{phase}.stderr").open("w") as err,
        ):
            subprocess.run(
                command, cwd=root, stdout=out, stderr=err, check=True, timeout=timeout
            )
        results[phase] = {
            "command": command,
            **parse_dispatch((artifacts / f"{phase}.stderr").read_text(), phase),
        }
    summary = {
        "timings_valid": False,
        "purpose": "Kernel dispatch evidence only; stderr tracing changes timing.",
        "source": config["llama_cpp"],
        "kleidiai": config["kleidiai"],
        "model": config["model"],
        "patch": patch,
        "flags": {
            key: cache[key] for key in flags if not key.startswith("FETCHCONTENT_")
        },
        "executable_sha256": runtime.digest(executable),
        "libraries": {
            p.name: runtime.digest(p)
            for p in (build / "bin").iterdir()
            if p.is_file()
            and not p.is_symlink()
            and (".so" in p.name or p.suffix == ".dylib")
        },
        "probes": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print("Prefill and decode kernels executed. Trace timings must not be used.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    try:
        config = json.loads((runtime.ROOT / "configs/runtime.json").read_text())
        trace(config, args.jobs, args.timeout)
    except (OSError, ValueError, tarfile.TarError, subprocess.SubprocessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
