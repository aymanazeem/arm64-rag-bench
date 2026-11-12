import hashlib
import importlib.metadata
import json
from pathlib import Path

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DIMENSION = 384
MAX_TOKENS = 512
WINDOW_TOKENS = MAX_TOKENS - 2


class BGEEncoder:
    def __init__(self, directory: Path, manifest_path: Path):
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as error:
            raise RuntimeError(
                "Install embeddings with uv sync --extra retrieval."
            ) from error

        manifest = json.loads(manifest_path.read_bytes())
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != 1
            or manifest.get("repository") != "BAAI/bge-small-en-v1.5"
            or not isinstance(manifest.get("revision"), str)
            or not manifest["revision"]
            or not isinstance(manifest.get("files"), list)
        ):
            raise ValueError("Unsupported embedding manifest.")
        assets = {}
        seen = set()
        for asset in manifest["files"]:
            if (
                not isinstance(asset, dict)
                or not isinstance(asset.get("path"), str)
                or not asset["path"]
                or type(asset.get("size")) is not int
                or asset["size"] < 1
                or not isinstance(asset.get("sha256"), str)
                or len(asset["sha256"]) != 64
                or set(asset["sha256"]) - set("0123456789abcdef")
                or not isinstance(asset.get("role"), str)
                or asset["role"] not in {"runtime", "reference"}
            ):
                raise ValueError("Invalid embedding asset record.")
            if asset["path"] in seen:
                raise ValueError("Duplicate embedding asset path.")
            seen.add(asset["path"])
            if asset.get("role") == "reference":
                continue
            relative = Path(asset["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    "Embedding asset paths must stay inside the model folder."
                )
            path = directory / relative
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != asset["sha256"] or path.stat().st_size != asset["size"]:
                raise ValueError(f"Embedding asset differs from its pin: {relative}")
            assets[relative.as_posix()] = digest
        if not {"onnx/model.onnx", "tokenizer.json"} <= assets.keys():
            raise ValueError("The embedding manifest is missing required assets.")
        self.np = np
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.special = [
            self.tokenizer.token_to_id(token) for token in ("[CLS]", "[SEP]", "[PAD]")
        ]
        if any(token is None for token in self.special):
            raise ValueError("The embedding tokenizer is missing special tokens.")
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(directory / "onnx/model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.config = {
            "version": "bge-onnx-windows-v1",
            "model": manifest["repository"],
            "revision": manifest["revision"],
            "assets": assets,
            "dimension": DIMENSION,
            "dtype": "float32",
            "pooling": "CLS",
            "normalization": "L2",
            "query_prefix": QUERY_PREFIX,
            "document_prefix": "",
            "document_text": "heading-path-newline-body-v1",
            "max_tokens": MAX_TOKENS,
            "window_tokens": WINDOW_TOKENS,
            "window_overlap": 0,
            "chunk_score": "maximum-window-cosine",
            "provider": "CPUExecutionProvider",
            "intra_op_threads": 4,
            "inter_op_threads": 1,
            "batch_size": 8,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "onnxruntime", "tokenizers")
            },
        }

    def token_ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def token_count(self, text: str) -> int:
        return len(self.token_ids(text))

    def _encode(self, windows: list[list[int]]):
        np = self.np
        cls, sep, pad = self.special
        batches = []
        for start in range(0, len(windows), self.config["batch_size"]):
            batch = windows[start : start + self.config["batch_size"]]
            if any(len(tokens) > WINDOW_TOKENS for tokens in batch):
                raise ValueError("Embedding input exceeds the model token limit.")
            width = max(len(tokens) for tokens in batch) + 2
            ids = np.full((len(batch), width), pad, dtype=np.int64)
            mask = np.zeros_like(ids)
            for row, tokens in enumerate(batch):
                size = len(tokens) + 2
                ids[row, :size] = [cls, *tokens, sep]
                mask[row, :size] = 1
            hidden = self.session.run(
                None,
                {
                    "input_ids": ids,
                    "attention_mask": mask,
                    "token_type_ids": np.zeros_like(ids),
                },
            )[0]
            if hidden.shape != (len(batch), width, DIMENSION):
                raise ValueError("Unexpected embedding output shape.")
            vectors = np.asarray(hidden[:, 0, :], dtype=np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if (
                not np.isfinite(vectors).all()
                or not np.isfinite(norms).all()
                or (norms == 0).any()
            ):
                raise ValueError("The embedding model returned invalid vectors.")
            batches.append(vectors / norms)
        return (
            np.concatenate(batches)
            if batches
            else np.empty((0, DIMENSION), dtype=np.float32)
        )

    def encode_documents(self, texts: list[str]):
        windows, counts = [], []
        for text in texts:
            ids = self.token_ids(text)
            pieces = [
                ids[start : start + WINDOW_TOKENS]
                for start in range(0, len(ids), WINDOW_TOKENS)
            ] or [[]]
            windows.extend(pieces)
            counts.append(len(pieces))
        vectors = self._encode(windows)
        output, start = [], 0
        for count in counts:
            output.append(vectors[start : start + count])
            start += count
        return output

    def encode_queries(self, texts: list[str]):
        windows = [self.token_ids(QUERY_PREFIX + text) for text in texts]
        if any(len(ids) > WINDOW_TOKENS for ids in windows):
            raise ValueError("Query exceeds 512 model tokens including its prefix.")
        return self._encode(windows)
