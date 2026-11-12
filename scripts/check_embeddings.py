# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "markdown-it-py==4.0.0",
#   "numpy==2.5.4",
#   "onnxruntime==1.31.1",
#   "tokenizers==0.22.2",
#   "torch==2.15.0",
#   "transformers==4.57.6",
# ]
# ///
import argparse
import hashlib
import importlib.metadata
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MAX_ABSOLUTE_DIFFERENCE = 1e-4
MIN_COSINE = 0.99999
MAX_BATCH_DIFFERENCE = 1e-5


def verify_reference_assets(directory, manifest_path):
    manifest = json.loads(manifest_path.read_bytes())
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        raise ValueError("Invalid embedding manifest.")
    if not all(isinstance(item, dict) for item in manifest["files"]):
        raise ValueError("Invalid embedding asset record.")
    reference = [item for item in manifest["files"] if item.get("role") == "reference"]
    if not any(item.get("path") == "model.safetensors" for item in reference):
        raise ValueError("The manifest must pin the reference safetensors file.")
    for item in reference:
        if (
            not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
            or type(item.get("size")) is not int
        ):
            raise ValueError("Invalid reference asset record.")
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Reference assets must stay inside the model folder.")
        path = directory / relative
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"] or path.stat().st_size != item["size"]:
            raise ValueError(f"Reference asset differs from its pin: {relative}")


def run_check(directory, manifest_path, database):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    from arm64_rag.embeddings import QUERY_PREFIX, WINDOW_TOKENS, BGEEncoder

    verify_reference_assets(directory, manifest_path)
    encoder = BGEEncoder(directory, manifest_path)
    reference_tokenizer = AutoTokenizer.from_pretrained(
        directory, local_files_only=True
    )
    torch.set_num_threads(4)
    model = AutoModel.from_pretrained(
        directory,
        local_files_only=True,
        use_safetensors=True,
        attn_implementation="eager",
    ).eval()

    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.execute("BEGIN")
        metadata = dict(db.execute("SELECT key, value FROM metadata"))
        rows = db.execute(
            "SELECT chunk_id, headings, text FROM chunks ORDER BY source_id, start"
        ).fetchall()
    if not rows:
        raise ValueError("Index the pinned corpus before checking embeddings.")
    passages = ["\n".join((*json.loads(headings), text)) for _, headings, text in rows]
    lengths = []
    for text in passages:
        actual = encoder.token_ids(text)
        expected = reference_tokenizer(text, add_special_tokens=False)["input_ids"]
        if actual != expected:
            raise ValueError("Application and reference token IDs differ.")
        lengths.append(len(actual) + 2)

    samples = [passages[i] for i in range(0, len(passages), 40) if lengths[i] <= 512]
    samples.extend(
        [
            "Enable GGML_CPU_KLEIDIAI with Q4_0 and --n-gpu-layers 0.",
            "Café Straße αβ 中文 👋",
            "# Header\n```sh\ncmake -B build\n```",
            "",
            "alpha " * 510,
        ]
    )

    def reference(ids):
        values = torch.tensor([[101, *ids, 102]], dtype=torch.int64)
        with torch.inference_mode():
            hidden = model(
                input_ids=values,
                attention_mask=torch.ones_like(values),
                token_type_ids=torch.zeros_like(values),
            )[0][:, 0, :]
            return torch.nn.functional.normalize(hidden, p=2, dim=1).numpy()[0]

    if (
        reference_tokenizer.cls_token_id != 101
        or reference_tokenizer.sep_token_id != 102
    ):
        raise ValueError("Unexpected reference special tokens.")

    def compare(vector, ids, label):
        expected = reference(ids)
        difference = float(np.max(np.abs(vector - expected)))
        cosine = float(np.dot(vector, expected))
        if (
            vector.shape != (384,)
            or not np.isfinite(vector).all()
            or abs(float(np.linalg.norm(vector)) - 1.0) > 1e-6
            or difference > MAX_ABSOLUTE_DIFFERENCE
            or cosine < MIN_COSINE
        ):
            raise ValueError(f"Embedding reference comparison failed: {label}")
        return {
            "sample": label,
            "tokens_including_specials": len(ids) + 2,
            "max_absolute_difference": difference,
            "cosine": cosine,
        }

    metrics = []
    separate = []
    for number, text in enumerate(samples):
        ids = reference_tokenizer(text, add_special_tokens=False)["input_ids"]
        vector = encoder.encode_documents([text])[0][0]
        separate.append(vector)
        metrics.append(compare(vector, ids, f"document-{number}"))
        metrics[-1]["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    batched = np.concatenate(encoder.encode_documents(samples))
    batch_difference = float(np.max(np.abs(batched - np.stack(separate))))
    if batch_difference > MAX_BATCH_DIFFERENCE:
        raise ValueError("Batch and individual embeddings differ.")
    repeated = np.concatenate(encoder.encode_documents(samples))
    repeat_difference = float(np.max(np.abs(repeated - batched)))
    if repeat_difference > 1e-6:
        raise ValueError("Repeated embeddings differ.")

    query = "Which port does the local server use?"
    query_ids = reference_tokenizer(QUERY_PREFIX + query, add_special_tokens=False)[
        "input_ids"
    ]
    metrics.append(compare(encoder.encode_queries([query])[0], query_ids, "query"))
    long_text = "alpha " * 511
    long_ids = reference_tokenizer(long_text, add_special_tokens=False)["input_ids"]
    windows = encoder.encode_documents([long_text])[0]
    if len(windows) != 2:
        raise ValueError("The oversized document was not split into two windows.")
    for number, vector in enumerate(windows):
        ids = long_ids[number * WINDOW_TOKENS : (number + 1) * WINDOW_TOKENS]
        metrics.append(compare(vector, ids, f"oversized-window-{number}"))
    try:
        encoder.encode_queries([long_text])
    except ValueError:
        pass
    else:
        raise ValueError("An oversized query was silently accepted.")
    return {
        "schema_version": 1,
        "kind": "embedding-compatibility",
        "encoder": encoder.config,
        "reference_packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "safetensors")
        },
        "index": {
            "generation": int(metadata["generation"]),
            "fingerprint": metadata["fingerprint"],
        },
        "chunk_tokenization": {
            "chunks": len(rows),
            "all_token_ids_equal": True,
            "min_tokens": min(lengths),
            "median_tokens": float(np.median(lengths)),
            "max_tokens": max(lengths),
            "over_512": sum(length > 512 for length in lengths),
        },
        "thresholds": {
            "max_absolute_difference": MAX_ABSOLUTE_DIFFERENCE,
            "min_cosine": MIN_COSINE,
            "max_batch_difference": MAX_BATCH_DIFFERENCE,
            "max_repeat_difference": 1e-6,
        },
        "samples": metrics,
        "batch_max_absolute_difference": batch_difference,
        "repeat_max_absolute_difference": repeat_difference,
        "oversized_document_windows": len(windows),
        "oversized_query_rejected": True,
        "limits": "Compatibility only; no retrieval or answer quality claim.",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check embeddings against Transformers."
    )
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT / "models/bge-small-en-v1.5"
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "configs/embeddings.json"
    )
    parser.add_argument(
        "--db", type=Path, default=ROOT / "data/private/development.sqlite"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/embeddings/compatibility.json"
    )
    args = parser.parse_args()
    try:
        report = run_check(args.model_dir, args.manifest, args.db)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Compatibility passed: {len(report['samples'])} vector checks.")


if __name__ == "__main__":
    main()
