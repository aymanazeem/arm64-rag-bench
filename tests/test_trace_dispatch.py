import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def trace(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "trace_dispatch", scripts / "trace_dispatch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def marker(trace, phase="prefill", m=32, n=1536, k=1536):
    return f"KAI_TRACE {trace.KERNELS[phase]} m={m} n={n} k={k}"


def test_real_marker_lines_record_calls_and_shapes(trace):
    text = "loader message\n" + marker(trace) + "\n" + marker(trace) + "\n"
    evidence = trace.parse_dispatch(text, "prefill")
    assert evidence["calls"] == 2
    assert evidence["shapes_m_n_k"] == [(32, 1536, 1536)]
    assert trace.parse_dispatch(marker(trace, "decode", m=1), "decode")["calls"] == 1


@pytest.mark.parametrize(
    "kind", ["missing", "echo", "symbol", "suffix", "wrong_kernel"]
)
def test_non_execution_text_is_not_dispatch_evidence(trace, kind):
    text = {
        "missing": "CPU_KLEIDIAI buffer allocated",
        "echo": f"$ printf '{marker(trace)}'",
        "symbol": f"00001234 T _{trace.KERNELS['prefill']}",
        "suffix": marker(trace) + " (example)",
        "wrong_kernel": marker(trace, "decode", m=1),
    }[kind]
    with pytest.raises(ValueError, match="No valid prefill"):
        trace.parse_dispatch(text, "prefill")


@pytest.mark.parametrize(
    "m,n,k", [(1, 1536, 1536), (0, 1536, 1536), (32, 0, 1536), (32, 1536, 0)]
)
def test_wrong_workload_dimensions_are_rejected(trace, m, n, k):
    with pytest.raises(ValueError, match="No valid prefill"):
        trace.parse_dispatch(marker(trace, m=m, n=n, k=k), "prefill")


def test_unexpected_or_duplicate_kernel_signature_is_rejected(trace):
    name = trace.KERNELS["prefill"]
    definition = f"void {name}({trace.PARAMETERS}) {{\n}}\n"
    with pytest.raises(ValueError, match="Unexpected kernel signature"):
        trace.instrument(definition.replace("size_t m", "int m"), name)
    with pytest.raises(ValueError, match="Expected one kernel definition"):
        trace.instrument(definition + definition, name)


def test_patch_is_isolated_and_repeatable(trace, tmp_path):
    clean = tmp_path / "clean"
    clean.mkdir()
    for name in trace.KERNELS.values():
        path = clean / (name.replace("kai_run_", "kai_") + ".c")
        path.write_text(f"void {name}({trace.PARAMETERS}) {{\n    return;\n}}\n")
    (clean / "other.c").write_text("unchanged\n")
    original = trace.runtime.tree_hashes(clean)
    destination, patch = tmp_path / "traced", tmp_path / "kernels.patch"

    evidence = trace.prepare_source(clean, destination, patch)

    assert trace.runtime.tree_hashes(clean) == original
    assert (destination / "other.c").read_text() == "unchanged\n"
    assert len(evidence["files"]) == 2
    assert evidence["original_tree_sha256"] != evidence["traced_tree_sha256"]
    assert "fprintf(stderr" in patch.read_text()
    assert str(tmp_path) not in patch.read_text()
    assert trace.prepare_source(clean, destination, patch) == evidence

    (destination / "other.c").write_text("unexpected edit\n")
    with pytest.raises(ValueError, match="trace source differs"):
        trace.prepare_source(clean, destination, patch)
    assert trace.runtime.tree_hashes(clean) == original
    assert (destination / "other.c").read_text() == "unexpected edit\n"
