import hashlib
import io
import json
import subprocess
import tarfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/runtime.py"
    spec = spec_from_file_location("runtime", script)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return module


def asset_for(content):
    return {
        "url": "https://example.invalid/model",
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def test_verified_cached_model_needs_no_download(runtime, monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"verified model")

    def unexpected_download(*args, **kwargs):
        pytest.fail("A verified cached model should not use the network.")

    monkeypatch.setattr(runtime.urllib.request, "urlopen", unexpected_download)
    assert runtime.download(asset_for(model.read_bytes()), model) == model


def test_wrong_cached_hash_is_rejected_without_changes(runtime, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="Checksum or size mismatch"):
        runtime.download(asset_for(b"right"), model)
    assert model.read_bytes() == b"wrong"


def test_failed_download_removes_partial_file(runtime, monkeypatch, tmp_path):
    model = tmp_path / "models/model.gguf"
    monkeypatch.setattr(
        runtime.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(b"wrong")
    )
    with pytest.raises(ValueError, match="Checksum or size mismatch"):
        runtime.download(asset_for(b"right"), model)
    assert not model.exists()
    assert not model.with_suffix(".gguf.part").exists()


def test_valid_download_is_published_after_verification(runtime, monkeypatch, tmp_path):
    model = tmp_path / "models/model.gguf"
    partial = model.with_suffix(".gguf.part")

    def response(*args, **kwargs):
        assert not model.exists()
        assert partial.exists()
        return io.BytesIO(b"verified model")

    monkeypatch.setattr(runtime.urllib.request, "urlopen", response)
    assert runtime.download(asset_for(b"verified model"), model) == model
    assert model.read_bytes() == b"verified model"
    assert not partial.exists()


def source_archive(runtime, name="kleidiai/CMakeLists.txt", symlink=False):
    archive = runtime.ROOT / ".local/downloads/kleidiai.tar.gz"
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w:gz") as bundle:
        entry = tarfile.TarInfo(name)
        if symlink:
            entry.type = tarfile.SYMTYPE
            entry.linkname = "../../outside"
            bundle.addfile(entry)
        else:
            content = b"project(kleidiai)\n"
            entry.size = len(content)
            bundle.addfile(entry, io.BytesIO(content))
    return {
        **asset_for(archive.read_bytes()),
        "filename": archive.name,
        "version": "test",
    }


@pytest.mark.parametrize("kind", ["traversal", "symlink"])
def test_unsafe_source_archive_is_rejected(runtime, kind):
    asset = source_archive(
        runtime,
        name="../../outside" if kind == "traversal" else "kleidiai/link",
        symlink=kind == "symlink",
    )
    with pytest.raises((ValueError, tarfile.FilterError)):
        runtime.kleidiai_source(asset)
    assert not (runtime.ROOT / "outside").exists()
    assert not (runtime.ROOT / "vendor/kleidiai-test").exists()


def test_changed_extracted_source_is_rejected(runtime):
    asset = source_archive(runtime)
    source = runtime.kleidiai_source(asset)
    assert runtime.kleidiai_source(asset) == source
    (source / "CMakeLists.txt").write_text("changed\n")
    with pytest.raises(ValueError, match="differs from the verified archive"):
        runtime.kleidiai_source(asset)
    assert (source / "CMakeLists.txt").read_text() == "changed\n"


def test_wrong_or_dirty_git_checkout_is_rejected(runtime):
    source = runtime.ROOT / "vendor/llama.cpp"
    source.mkdir(parents=True)

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(source), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    git("init", "-b", "fixture")
    git("config", "user.name", "Test Fixture")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "commit.gpgsign", "false")
    git("config", "core.hooksPath", "/dev/null")
    (source / "source.cpp").write_text("original\n")
    git("add", "source.cpp")
    git("commit", "-m", "Fixture")
    config = {"revision": git("rev-parse", "HEAD")}
    assert runtime.source_checkout(config) == source
    with pytest.raises(ValueError, match="clean and at the configured revision"):
        runtime.source_checkout({"revision": "0" * 40})
    (source / "source.cpp").write_text("modified\n")
    with pytest.raises(ValueError, match="clean and at the configured revision"):
        runtime.source_checkout(config)


def test_failed_initial_fetch_can_retry_without_partial_checkout(runtime, tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    for args in (
        ["init", "-b", "fixture"],
        ["config", "user.name", "Test Fixture"],
        ["config", "user.email", "fixture@example.invalid"],
        ["config", "commit.gpgsign", "false"],
        ["config", "core.hooksPath", "/dev/null"],
        ["commit", "--allow-empty", "-m", "Fixture"],
    ):
        subprocess.run(
            ["git", "-C", str(remote), *args], check=True, capture_output=True
        )
    revision = subprocess.check_output(
        ["git", "-C", str(remote), "rev-parse", "HEAD"], text=True
    ).strip()
    config = {"repository": str(tmp_path / "missing"), "revision": revision}
    with pytest.raises(subprocess.CalledProcessError):
        runtime.source_checkout(config)
    assert not (tmp_path / "vendor/llama.cpp").exists()
    assert list((tmp_path / "vendor").iterdir()) == []
    config["repository"] = str(remote)
    source = runtime.source_checkout(config)
    assert runtime.capture("git", "-C", str(source), "rev-parse", "HEAD") == revision
    assert list((tmp_path / "vendor").iterdir()) == [source]


@pytest.mark.parametrize(
    "name", ["CFLAGS", "CXXFLAGS", "LDFLAGS", "CMAKE_TOOLCHAIN_FILE"]
)
def test_inherited_build_options_invalidate_old_manifest(runtime, monkeypatch, name):
    manifest = runtime.ROOT / "artifacts/feasibility/cpu/build.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"old": true}')
    monkeypatch.setenv(name, "unexpected option")
    with pytest.raises(ValueError, match=f"Unset {name}"):
        runtime.build({}, "cpu", 1)
    assert not manifest.exists()


def test_early_build_failure_invalidates_old_manifest(runtime):
    manifest = runtime.ROOT / "artifacts/feasibility/cpu/build.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"old": true}')
    with pytest.raises(ValueError, match="Jobs must be at least 1"):
        runtime.build({}, "cpu", 0)
    assert not manifest.exists()


@pytest.mark.parametrize(
    "cache_text, expected",
    [
        ("GGML_CPU:BOOL=OFF\nGGML_AVAILABLE_BACKENDS:INTERNAL=ggml-cpu\n", "GGML_CPU"),
        (
            "GGML_CPU:BOOL=ON\nGGML_AVAILABLE_BACKENDS:INTERNAL=ggml-cpu;ggml-metal\n",
            "only the CPU backend",
        ),
    ],
)
def test_changed_flags_or_extra_backends_are_rejected(runtime, cache_text, expected):
    cache = runtime.ROOT / "CMakeCache.txt"
    cache.write_text(cache_text)
    with pytest.raises(ValueError, match=expected):
        runtime.checked_cache(cache, {"GGML_CPU": "ON"})


def test_rebuild_removes_stale_cmake_settings(runtime, monkeypatch):
    source = runtime.ROOT / "source"
    source.mkdir()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.28)\n"
        "project(Fixture LANGUAGES C CXX)\n"
        'option(GGML_CPU_KLEIDIAI "Fixture option" OFF)\n'
        'option(GGML_RPC "Fixture option" OFF)\n'
        'set(GGML_AVAILABLE_BACKENDS ggml-cpu CACHE INTERNAL "Backends" FORCE)\n'
        "if(GGML_RPC)\n"
        '  set(GGML_AVAILABLE_BACKENDS "ggml-cpu;ggml-rpc" CACHE INTERNAL "" FORCE)\n'
        "endif()\n"
        'set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")\n'
        "foreach(name llama-bench llama-server llama-completion)\n"
        "  add_executable(${name} main.c)\n"
        "endforeach()\n"
    )
    (source / "main.c").write_text("int main(void) { return 0; }\n")
    build = runtime.ROOT / "build/cpu"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DCMAKE_C_FLAGS=-DSTALE_FLAG",
            "-DGGML_RPC=ON",
        ],
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(runtime, "source_checkout", lambda _: source)
    config = {
        "llama_cpp": {"revision": "fixture"},
        "cmake": {"CMAKE_BUILD_TYPE": "Release", "CMAKE_EXPORT_COMPILE_COMMANDS": "ON"},
    }
    runtime.build(config, "cpu", 1)
    artifacts = runtime.ROOT / "artifacts/feasibility/cpu"
    manifest = json.loads((artifacts / "build.json").read_text())
    assert manifest["compiler_flags"]["CMAKE_C_FLAGS"] == ""
    assert manifest["backends"] == ["ggml-cpu"]
    assert manifest["cache_sha256"] == runtime.digest(artifacts / "CMakeCache.txt")
    assert manifest["compile_commands_sha256"] == runtime.digest(
        artifacts / "compile_commands.json"
    )
    assert "STALE_FLAG" not in (artifacts / "compile_commands.json").read_text()
    for target in runtime.TARGETS:
        subprocess.run([str(build / "bin" / target)], check=True)
