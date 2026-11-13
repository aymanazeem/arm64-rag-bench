import base64
import hashlib
import http.client
import json
import math
import os
import secrets
import socket
import subprocess
import threading
import time
from pathlib import Path

MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class InferenceError(RuntimeError):
    def __init__(self, message, *, response=None):
        super().__init__(message)
        self.response = response


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _positive_timeout(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive number.")


def _invalid_body(message, status, raw):
    retained = raw[:MAX_RESPONSE_BYTES]
    try:
        content, encoding = retained.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        content, encoding = base64.b64encode(retained).decode("ascii"), "base64"
    return InferenceError(
        message,
        response={
            "http_status": status,
            "raw_body": content,
            "body_encoding": encoding,
            "body_bytes_retained": len(retained),
            "body_bytes_read": len(raw),
            "body_byte_limit": MAX_RESPONSE_BYTES,
            "body_truncated": len(raw) > MAX_RESPONSE_BYTES,
        },
    )


class LocalServer:
    def __init__(
        self,
        binary: Path,
        model: Path,
        *,
        context_size=8192,
        threads=4,
        startup_timeout=120,
        request_timeout=120,
        reasoning_budget_tokens=None,
        log_path=Path("artifacts/qa/server.log"),
    ):
        if type(context_size) is not int or context_size < 32:
            raise ValueError("Context size must be at least 32 tokens.")
        if type(threads) is not int or threads < 1:
            raise ValueError("Threads must be a positive integer.")
        if reasoning_budget_tokens is not None and (
            type(reasoning_budget_tokens) is not int
            or not 0 <= reasoning_budget_tokens < 2**31
        ):
            raise ValueError("Reasoning budget must be a nonnegative integer.")
        _positive_timeout(startup_timeout, "Startup timeout")
        _positive_timeout(request_timeout, "Request timeout")
        self.binary, self.model = Path(binary).resolve(), Path(model).resolve()
        self.context_size, self.threads = context_size, threads
        self.startup_timeout, self.request_timeout = startup_timeout, request_timeout
        self.reasoning_budget_tokens = reasoning_budget_tokens
        self.log_path = Path(log_path)
        self.process = None
        self._log = None
        self._info = None
        self._key = secrets.token_urlsafe(32)
        self._port = None

    def __enter__(self):
        if self.process is not None:
            raise InferenceError("This server instance has already been started.")
        if not self.binary.is_file() or not self.model.is_file():
            raise FileNotFoundError(
                "Build llama-server and download the configured model first."
            )
        libraries = {
            path.resolve()
            for pattern in ("*.dylib", "*.so", "*.so.*")
            for path in self.binary.parent.glob(pattern)
            if path.is_file()
        }
        identity = {
            "binary_sha256": _sha256(self.binary),
            "model_sha256": _sha256(self.model),
            "model_file": self.model.name,
            "sibling_libraries_sha256": {
                path.name: _sha256(path) for path in sorted(libraries)
            },
            "library_inventory": "Sibling library hashes; not a loaded-library trace.",
        }
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
            reservation.bind(("127.0.0.1", 0))
            self._port = reservation.getsockname()[1]
        command = [
            str(self.binary),
            "--model",
            str(self.model),
            "--host",
            "127.0.0.1",
            "--port",
            str(self._port),
            "--alias",
            "arm64-rag-local",
            "--ctx-size",
            str(self.context_size),
            "--parallel",
            "1",
            "--threads",
            str(self.threads),
            "--threads-batch",
            str(self.threads),
            "--n-gpu-layers",
            "0",
            "--batch-size",
            "512",
            "--ubatch-size",
            "512",
            "--cache-type-k",
            "f16",
            "--cache-type-v",
            "f16",
            "--flash-attn",
            "off",
            "--jinja",
            "--no-context-shift",
            "--no-webui",
            "--no-warmup",
        ]
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("LLAMA_", "GGML_", "DYLD_", "LD_"))
        }
        environment["LLAMA_API_KEY"] = self._key
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("wb")
        started = time.monotonic()
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            deadline = started + self.startup_timeout
            while True:
                if self.process.poll() is not None:
                    raise InferenceError(
                        "llama-server exited during startup; inspect its local log."
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise InferenceError(
                        "llama-server did not become ready before the startup deadline."
                    )
                try:
                    status, props = self._request(
                        "GET", "/props", timeout=min(remaining, 0.5)
                    )
                except (OSError, http.client.HTTPException):
                    time.sleep(min(0.05, max(0, deadline - time.monotonic())))
                    continue
                if status == 503:
                    time.sleep(min(0.05, max(0, deadline - time.monotonic())))
                    continue
                if status != 200:
                    raise InferenceError(
                        "The local server failed its authenticated readiness check."
                    )
                settings = props.get("default_generation_settings")
                context = settings.get("n_ctx") if isinstance(settings, dict) else None
                template = props.get("chat_template")
                if (
                    type(context) is not int
                    or context != self.context_size
                    or type(props.get("total_slots")) is not int
                    or props["total_slots"] != 1
                    or props.get("model_alias") != "arm64-rag-local"
                    or props.get("model_path") != str(self.model)
                    or not isinstance(template, str)
                    or not template
                ):
                    raise InferenceError(
                        "The running model, template or context differs."
                    )
                self._info = {
                    **identity,
                    "context_size": context,
                    "threads": self.threads,
                    "parallel": 1,
                    "gpu_layers": 0,
                    "cache_type_k": "f16",
                    "cache_type_v": "f16",
                    "flash_attention": False,
                    "context_shift": False,
                    "cache_prompt": False,
                    "enable_thinking": False,
                    "temperature": 0.0,
                    "seed": 42,
                    "batch_size": 512,
                    "microbatch_size": 512,
                    "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                    "build_info": props.get("build_info"),
                    "model_alias": props["model_alias"],
                    "startup_seconds": time.monotonic() - started,
                    "startup_timeout_seconds": self.startup_timeout,
                    "request_timeout_seconds": self.request_timeout,
                }
                if self.reasoning_budget_tokens is not None:
                    self._info.update(
                        completion_endpoint="/v1/chat/completions",
                        enable_thinking=True,
                        reasoning_format="deepseek",
                        reasoning_budget_tokens=self.reasoning_budget_tokens,
                        reasoning_budget_message="",
                        temperature=0.6,
                        top_p=0.95,
                        top_k=20,
                        min_p=0.0,
                        samplers=["top_k", "top_p", "min_p", "temperature"],
                        output_budget_scope="All generated tokens, including reasoning",
                    )
                return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, *_):
        self.close()

    def close(self):
        try:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=2)
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None

    def _request(self, method, path, body=None, *, timeout=None):
        timeout = self.request_timeout if timeout is None else timeout
        started = time.monotonic()
        connection = http.client.HTTPConnection(
            "127.0.0.1", self._port, timeout=timeout
        )
        expired = threading.Event()
        timer = None
        try:
            connection.connect()
            transport = connection.sock

            def expire():
                expired.set()
                try:
                    transport.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

            remaining = max(0.001, timeout - (time.monotonic() - started))
            timer = threading.Timer(remaining, expire)
            timer.daemon = True
            timer.start()
            headers = {
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
            }
            payload = (
                None if body is None else json.dumps(body, ensure_ascii=False).encode()
            )
            connection.request(method, path, body=payload, headers=headers)
            response = connection.getresponse()
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if expired.is_set():
                raise TimeoutError("Local inference request exceeded its deadline.")
            if len(raw) > MAX_RESPONSE_BYTES:
                raise _invalid_body(
                    "Local inference response exceeded its size limit.",
                    response.status,
                    raw,
                )
            try:
                result = json.loads(raw)
            except (ValueError, UnicodeError) as error:
                raise _invalid_body(
                    "The local server returned invalid JSON.", response.status, raw
                ) from error
            if not isinstance(result, dict):
                raise _invalid_body(
                    "The local server returned an invalid response object.",
                    response.status,
                    raw,
                )
            return response.status, result
        except (OSError, http.client.HTTPException) as error:
            if expired.is_set():
                raise TimeoutError(
                    "Local inference request exceeded its deadline."
                ) from error
            raise
        finally:
            if timer is not None:
                timer.cancel()
            connection.close()

    def _call(self, path, body, *, deadline=None):
        if (
            self.process is None
            or self.process.poll() is not None
            or self._info is None
        ):
            raise InferenceError("The local server is not running.")
        try:
            timeout = None if deadline is None else deadline - time.monotonic()
            if timeout is not None and timeout <= 0:
                raise TimeoutError("Local inference request exceeded its deadline.")
            status, response = self._request("POST", path, body, timeout=timeout)
            if status != 200:
                raise InferenceError(
                    f"Local inference request failed with HTTP {status}.",
                    response=response,
                )
            return response
        except (OSError, http.client.HTTPException, InferenceError) as error:
            self.close()
            if isinstance(error, InferenceError):
                raise
            raise InferenceError(
                "The local inference request failed or timed out."
            ) from error

    def _template_body(self, messages, json_schema):
        if (
            not isinstance(messages, list)
            or not messages
            or any(
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
                for message in messages
            )
        ):
            raise ValueError("Messages must contain text and a valid chat role.")
        if json_schema is not None and not isinstance(json_schema, dict):
            raise ValueError("The output schema must be a JSON object.")
        body = {
            "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": {
                "enable_thinking": self.reasoning_budget_tokens is not None
            },
        }
        if self.reasoning_budget_tokens is not None:
            if json_schema is None or messages[-1]["role"] != "user":
                raise ValueError(
                    "Reasoning requires a schema and a final user message."
                )
            body.update(
                reasoning_format="deepseek",
                reasoning_budget_tokens=self.reasoning_budget_tokens,
                reasoning_budget_message="",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "strict": True,
                        "schema": json_schema,
                    },
                },
            )
        return body

    def render(self, messages: list[dict], *, json_schema: dict | None = None) -> str:
        return self._render(self._template_body(messages, json_schema))

    def _render(self, body, *, deadline=None):
        response = self._call("/apply-template", body, deadline=deadline)
        prompt = response.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise InferenceError(
                "The chat template returned an empty or invalid prompt.",
                response=response,
            )
        return prompt

    def tokenize(self, prompt: str) -> list[int]:
        return self._tokenize(prompt)

    def _tokenize(self, prompt: str, *, deadline=None):
        if not isinstance(prompt, str):
            raise ValueError("The prompt must be text.")
        response = self._call(
            "/tokenize",
            {
                "content": prompt,
                "add_special": self.reasoning_budget_tokens is not None,
                "parse_special": True,
                "with_pieces": False,
            },
            deadline=deadline,
        )
        tokens = response.get("tokens")
        if not isinstance(tokens, list) or any(
            type(token) is not int or token < 0 for token in tokens
        ):
            raise InferenceError(
                "The server returned invalid token IDs.", response=response
            )
        return tokens

    def complete(
        self,
        token_ids: list[int],
        max_tokens: int,
        *,
        json_schema: dict | None = None,
        messages: list[dict] | None = None,
    ) -> dict:
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(type(token) is not int or token < 0 for token in token_ids)
        ):
            raise ValueError("Completion requires a nonempty list of token IDs.")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("Output token limit must be a positive integer.")
        if len(token_ids) + max_tokens >= self.context_size:
            raise ValueError("Prompt and reserved output exceed the context budget.")
        if json_schema is not None and not isinstance(json_schema, dict):
            raise ValueError("The output schema must be a JSON object.")
        if self.reasoning_budget_tokens is not None:
            return self._complete_chat(token_ids, max_tokens, json_schema, messages)
        body = {
            "prompt": token_ids,
            "n_predict": max_tokens,
            "stream": False,
            "cache_prompt": False,
            "temperature": 0.0,
            "seed": 42,
            "samplers": ["temperature"],
            "n_keep": -1,
            "return_tokens": True,
            "id_slot": 0,
        }
        if json_schema is not None:
            body["json_schema"] = json_schema
        response = self._call("/completion", body)
        if (
            not isinstance(response.get("content"), str)
            or response.get("stop") is not True
            or response.get("truncated") is not False
            or not isinstance(response.get("stop_type"), str)
            or response["stop_type"] not in {"eos", "word"}
            or type(response.get("tokens_evaluated")) is not int
            or response.get("tokens_evaluated") != len(token_ids)
            or type(response.get("tokens_predicted")) is not int
            or not 0 <= response["tokens_predicted"] <= max_tokens
            or not isinstance(response.get("tokens"), list)
            or len(response["tokens"]) != response["tokens_predicted"]
            or any(type(token) is not int or token < 0 for token in response["tokens"])
        ):
            raise InferenceError(
                "Generation was incomplete, truncated or exceeded its token budget.",
                response=response,
            )
        return response

    def _complete_chat(self, token_ids, max_tokens, json_schema, messages):
        if self.reasoning_budget_tokens >= max_tokens:
            raise ValueError("The total output limit must exceed the reasoning budget.")
        body = self._template_body(messages, json_schema)
        body.update(
            model="arm64-rag-local",
            n_predict=max_tokens,
            stream=False,
            n=1,
            cache_prompt=False,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            seed=42,
            samplers=["top_k", "top_p", "min_p", "temperature"],
            n_keep=-1,
            return_tokens=True,
            verbose=True,
            id_slot=0,
        )
        deadline = time.monotonic() + self.request_timeout
        prompt = self._render(body, deadline=deadline)
        if self._tokenize(prompt, deadline=deadline) != token_ids:
            raise InferenceError("The chat prompt differs from the packed token IDs.")
        count = self._call("/v1/chat/completions/input_tokens", body, deadline=deadline)
        if type(count.get("input_tokens")) is not int or count["input_tokens"] != len(
            token_ids
        ):
            raise InferenceError("The chat prompt token counts differ.", response=count)
        response = self._call("/v1/chat/completions", body, deadline=deadline)
        choices, native, usage = (
            response.get("choices"),
            response.get("__verbose"),
            response.get("usage"),
        )
        choice = choices[0] if isinstance(choices, list) and len(choices) == 1 else None
        message = choice.get("message") if isinstance(choice, dict) else None
        if (
            not isinstance(message, dict)
            or message.get("role") != "assistant"
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
            or not isinstance(message.get("reasoning_content", ""), str)
            or message.get("tool_calls")
            or choice.get("finish_reason") != "stop"
            or type(choice.get("index")) is not int
            or choice["index"] != 0
            or response.get("object") != "chat.completion"
            or response.get("model") != "arm64-rag-local"
            or not isinstance(native, dict)
            or not isinstance(native.get("content"), str)
            or native.get("prompt") != prompt
            or native.get("stop") is not True
            or native.get("truncated") is not False
            or native.get("stop_type") not in ("eos", "word")
            or type(native.get("tokens_evaluated")) is not int
            or native["tokens_evaluated"] != len(token_ids)
            or type(native.get("tokens_predicted")) is not int
            or not 0 <= native["tokens_predicted"] <= max_tokens
            or not isinstance(native.get("tokens"), list)
            or len(native["tokens"]) != native["tokens_predicted"]
            or any(type(token) is not int or token < 0 for token in native["tokens"])
            or not isinstance(usage, dict)
            or any(
                type(usage.get(key)) is not int or usage[key] != expected
                for key, expected in (
                    ("prompt_tokens", len(token_ids)),
                    ("completion_tokens", native["tokens_predicted"]),
                    ("total_tokens", len(token_ids) + native["tokens_predicted"]),
                )
            )
        ):
            raise InferenceError(
                "Chat generation was incomplete or returned inconsistent metadata.",
                response=response,
            )
        return {
            **native,
            "content": message["content"],
            "raw_content": native["content"],
            "reasoning_content": message.get("reasoning_content", ""),
            "chat_response": response,
        }

    def info(self) -> dict:
        if self._info is None:
            raise InferenceError(
                "The local server has no verified runtime information."
            )
        return json.loads(json.dumps(self._info))
