import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from arm64_rag.embeddings import QUERY_PREFIX, BGEEncoder

np = pytest.importorskip("numpy")
tokenizers = pytest.importorskip("tokenizers")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def assets(tmp_path, monkeypatch):
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordPiece(
            {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "alpha": 4, "beta": 5},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx/model.onnx").write_bytes(b"Synthetic model bytes")
    records = []
    for name in ("tokenizer.json", "onnx/model.onnx"):
        raw = (tmp_path / name).read_bytes()
        records.append(
            {
                "path": name,
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "role": "runtime",
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repository": "BAAI/bge-small-en-v1.5",
                "revision": "test-revision",
                "files": records,
            }
        )
    )
    calls = []
    disabled = []

    class SessionOptions:
        def add_session_config_entry(self, *args):
            pass

    class Session:
        def __init__(self, path, sess_options, providers):
            assert providers == ["CPUExecutionProvider"]
            assert disabled

        def run(self, output_names, inputs):
            calls.append({key: value.copy() for key, value in inputs.items()})
            ids = inputs["input_ids"]
            hidden = np.zeros((*ids.shape, 384), dtype=np.float32)
            hidden[:, :, 2] = 9
            hidden[:, 0, 2] = 0
            hidden[:, 0, 0] = (ids * inputs["attention_mask"]).sum(axis=1) + 1
            hidden[:, 0, 1] = 2
            return [hidden]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            disable_telemetry_events=lambda: disabled.append(True),
            SessionOptions=SessionOptions,
            InferenceSession=Session,
            ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL=0),
        ),
    )
    return tmp_path, manifest, calls


def test_document_vectors_use_cls_normalization_and_masked_padding(assets):
    directory, manifest, calls = assets
    encoder = BGEEncoder(directory, manifest)
    vectors = encoder.encode_documents(["alpha", "alpha beta"])
    assert [value.shape for value in vectors] == [(1, 384), (1, 384)]
    assert np.allclose(np.linalg.norm(np.concatenate(vectors), axis=1), 1)
    assert not np.concatenate(vectors)[:, 2:].any()
    batch = calls[0]
    assert batch["input_ids"].tolist() == [[2, 4, 3, 0], [2, 4, 5, 3]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0], [1, 1, 1, 1]]
    assert not batch["token_type_ids"].any()
    assert batch["input_ids"].dtype == np.int64


def test_query_prefix_is_present_only_for_queries(assets):
    directory, manifest, calls = assets
    encoder = BGEEncoder(directory, manifest)
    encoder.encode_queries(["alpha"])
    expected = encoder.tokenizer.encode(QUERY_PREFIX + "alpha").ids
    assert calls[-1]["input_ids"][0].tolist() == expected
    encoder.encode_documents(["alpha"])
    assert calls[-1]["input_ids"][0].tolist() == [2, 4, 3]


def test_oversized_document_keeps_every_token_in_order(assets):
    directory, manifest, calls = assets
    encoder = BGEEncoder(directory, manifest)
    text = "alpha " * 511
    vectors = encoder.encode_documents([text])[0]
    assert vectors.shape == (2, 384)
    batch = calls[0]
    lengths = batch["attention_mask"].sum(axis=1).tolist()
    assert lengths == [512, 3]
    reconstructed = []
    for row, length in enumerate(lengths):
        reconstructed.extend(batch["input_ids"][row, 1 : length - 1].tolist())
    assert reconstructed == encoder.token_ids(text)


def test_exact_model_limit_succeeds_and_oversized_query_fails(assets):
    directory, manifest, calls = assets
    encoder = BGEEncoder(directory, manifest)
    assert encoder.encode_documents(["alpha " * 510])[0].shape == (1, 384)
    before = len(calls)
    with pytest.raises(ValueError, match="Query exceeds"):
        encoder.encode_queries(["alpha " * 511])
    assert len(calls) == before


def test_batching_preserves_document_order_and_empty_inputs(assets):
    directory, manifest, calls = assets
    encoder = BGEEncoder(directory, manifest)
    texts = ["alpha"] * 8 + ["beta", ""]
    result = encoder.encode_documents(texts)
    assert len(result) == 10
    assert len(calls) == 2
    assert calls[1]["input_ids"].tolist() == [[2, 5, 3], [2, 3, 0]]
    assert encoder.encode_documents([]) == []
    assert encoder.encode_queries([]).shape == (0, 384)


def test_changed_model_bytes_are_rejected(assets):
    directory, manifest, _ = assets
    (directory / "onnx/model.onnx").write_bytes(b"Changed model")
    with pytest.raises(ValueError, match="differs from its pin"):
        BGEEncoder(directory, manifest)


def test_asset_paths_cannot_leave_model_directory(assets):
    directory, manifest, _ = assets
    value = json.loads(manifest.read_text())
    value["files"][0]["path"] = "../tokenizer.json"
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="inside the model folder"):
        BGEEncoder(directory, manifest)


@pytest.mark.parametrize(
    "damage",
    ["list", "revision", "files", "asset", "path", "size", "hash", "role", "duplicate"],
)
def test_malformed_manifest_returns_a_plain_validation_error(assets, damage):
    directory, manifest, _ = assets
    value = json.loads(manifest.read_text())
    if damage == "list":
        value = []
    elif damage == "revision":
        del value["revision"]
    elif damage == "files":
        value["files"] = {}
    elif damage == "asset":
        value["files"][0] = "bad asset"
    elif damage == "duplicate":
        value["files"].append(value["files"][0])
    else:
        field, invalid = {
            "path": ("path", 7),
            "size": ("size", True),
            "hash": ("sha256", "wrong"),
            "role": ("role", []),
        }[damage]
        value["files"][0][field] = invalid
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="manifest|asset"):
        BGEEncoder(directory, manifest)


@pytest.mark.parametrize("invalid", ["shape", "nan", "zero"])
def test_invalid_model_output_fails(assets, invalid):
    directory, manifest, _ = assets
    encoder = BGEEncoder(directory, manifest)

    def invalid_output(names, inputs):
        shape = (*inputs["input_ids"].shape, 383 if invalid == "shape" else 384)
        return [np.full(shape, np.nan if invalid == "nan" else 0, dtype=np.float32)]

    encoder.session.run = invalid_output
    with pytest.raises(ValueError, match="output shape|invalid vectors"):
        encoder.encode_documents(["alpha"])


@pytest.mark.skipif(
    os.environ.get("ARM64_VERIFY_EMBEDDINGS") != "1", reason="Optional embedding assets"
)
def test_pinned_model_runs_locally_without_truncation():
    encoder = BGEEncoder(
        ROOT / "models/bge-small-en-v1.5", ROOT / "configs/embeddings.json"
    )
    vector = encoder.encode_queries(["Which CPU build flag enables KleidiAI?"])
    assert vector.shape == (1, 384)
    assert np.isfinite(vector).all()
    assert np.allclose(np.linalg.norm(vector, axis=1), 1, atol=1e-6)
    windows = encoder.encode_documents(["alpha " * 511])[0]
    assert windows.shape == (2, 384)
    assert np.allclose(np.linalg.norm(windows, axis=1), 1, atol=1e-6)
