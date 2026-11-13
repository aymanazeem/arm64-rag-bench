import base64
import json
import sys
import textwrap
import time

import pytest

from arm64_rag import inference
from arm64_rag.inference import InferenceError, LocalServer

FAKE_SERVER = r"""
import argparse
import json
import os
import signal
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--host')
parser.add_argument('--port', type=int)
parser.add_argument('--alias')
parser.add_argument('--ctx-size', type=int)
parser.add_argument('--parallel', type=int)
args, _ = parser.parse_known_args()
settings = json.loads(Path(args.model).read_text())
mode = settings.get('mode')
if mode == 'exit':
    raise SystemExit(7)
if mode == 'hang':
    time.sleep(30)
if mode == 'ignore-terminate':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, value, status=200):
        raw = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def authorized(self):
        expected = 'Bearer ' + os.environ['LLAMA_API_KEY']
        if mode == 'wrong-auth' or self.headers.get('Authorization') != expected:
            self.send_json({'error': 'Unauthorized'}, 401)
            return False
        return True

    def do_GET(self):
        if not self.authorized():
            return
        if mode == 'loading':
            self.send_json({'error': 'Loading'}, 503)
            return
        context = 999 if mode == 'wrong-context' else args.ctx_size
        self.send_json({
            'default_generation_settings': {'n_ctx': context},
            'total_slots': args.parallel, 'model_alias': args.alias,
            'model_path': args.model, 'chat_template': 'test template',
            'build_info': 'test-build',
        })

    def do_POST(self):
        if not self.authorized():
            return
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if 'capture' in settings:
            with Path(settings['capture']).open('a') as stream:
                stream.write(json.dumps({'path': self.path, 'body': body}) + '\n')
        if settings.get('delay'):
            time.sleep(settings['delay'])
        if mode == 'bad-json':
            self.send_response(200)
            self.send_header('Content-Length', '5')
            self.end_headers()
            self.wfile.write(b'wrong')
            return
        if mode == 'nonobject':
            self.send_json(['unexpected', 'array'])
            return
        if mode == 'oversize':
            self.send_json({'content': 'x' * 4096})
            return
        if mode == 'bad-bytes':
            raw = b'\xff\xfe\x80broken'
            self.send_response(502)
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if mode == 'drip':
            raw = b'{"prompt":"' + b'a' * 100 + b'"}'
            self.send_response(200)
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            for byte in raw:
                self.wfile.write(bytes([byte]))
                self.wfile.flush()
                time.sleep(0.03)
            return
        if self.path == '/apply-template':
            self.send_json({'prompt': '<chat>\nQuestion\n<assistant>'})
        elif self.path == '/tokenize':
            self.send_json({'tokens': [101, 42, 102]})
        elif self.path == '/v1/chat/completions/input_tokens':
            self.send_json({'input_tokens': settings.get('count', 3)})
        elif self.path == '/v1/chat/completions':
            if mode == 'chat-error':
                self.send_json({'error': {'message': 'Output parser failed'}}, 500)
                return
            native = {
                'content': '<think>Check evidence.</think>{"answer":"Paris"}',
                'prompt': '<chat>\nQuestion\n<assistant>',
                'stop': True, 'stop_type': 'eos', 'truncated': False,
                'tokens_predicted': 6, 'tokens_evaluated': 3,
                'tokens': [201, 202, 203, 204, 205, 206],
                'timings': {'predicted_ms': 1},
            }
            native.update(settings.get('completion', {}))
            result = {
                'object': 'chat.completion', 'model': args.alias,
                'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {
                    'role': 'assistant', 'content': '{"answer":"Paris"}',
                    'reasoning_content': 'Check evidence.',
                }}],
                'usage': {
                    'prompt_tokens': 3, 'completion_tokens': 6, 'total_tokens': 9,
                },
                '__verbose': native,
            }
            result.update(settings.get('chat', {}))
            self.send_json(result)
        elif self.path == '/completion':
            result = {
                'content': '{"answer":"Paris"}', 'stop': True,
                'stop_type': 'eos', 'truncated': False, 'tokens_predicted': 4,
                'tokens': [201, 202, 203, 204],
                'tokens_evaluated': len(body['prompt']), 'timings': {'predicted_ms': 1},
            }
            result.update(settings.get('completion', {}))
            self.send_json(result)

HTTPServer((args.host, args.port), Handler).serve_forever()
"""


@pytest.fixture
def server_files(tmp_path):
    binary = tmp_path / "fake-server"
    binary.write_text(f"#!{sys.executable}\n" + textwrap.dedent(FAKE_SERVER))
    binary.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_text("{}")
    return binary, model, tmp_path / "server.log"


def server(
    server_files, *, mode=None, completion=None, chat=None, count=3, delay=0, **kwargs
):
    binary, model, log = server_files
    model.write_text(
        json.dumps(
            {
                "mode": mode,
                "completion": completion or {},
                "chat": chat or {},
                "count": count,
                "delay": delay,
            }
        )
    )
    return LocalServer(binary, model, log_path=log, startup_timeout=5, **kwargs)


def test_template_token_ids_completion_and_sanitized_provenance(
    server_files, monkeypatch
):
    binary, model, log = server_files
    capture = model.with_suffix(".jsonl")
    model.write_text(json.dumps({"capture": str(capture)}))
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("LLAMA_ARG_N_GPU_LAYERS", "999")
    client = LocalServer(binary, model, log_path=log, startup_timeout=5)
    with client:
        prompt = client.render([{"role": "user", "content": "Question"}])
        tokens = client.tokenize(prompt)
        result = client.complete(tokens, 32, json_schema={"type": "object"})
        assert result["content"] == '{"answer":"Paris"}'
        info = client.info()
        assert info["context_size"] == 8192
        assert info["gpu_layers"] == 0
        assert len(info["binary_sha256"]) == len(info["model_sha256"]) == 64
        assert str(model.parent) not in json.dumps(info)
        assert client._key not in json.dumps(info)
    assert client.process.poll() is not None
    assert client.info() == info
    records = [json.loads(line) for line in capture.read_text().splitlines()]
    assert records[0]["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert records[1]["body"] == {
        "content": prompt,
        "add_special": False,
        "parse_special": True,
        "with_pieces": False,
    }
    body = records[2]["body"]
    assert body["prompt"] == tokens
    assert body["cache_prompt"] is False
    assert body["temperature"] == 0
    assert body["seed"] == 42
    assert body["json_schema"] == {"type": "object"}


@pytest.mark.parametrize("mode", ["exit", "wrong-auth", "wrong-context"])
def test_startup_failure_closes_owned_process(server_files, mode):
    client = server(server_files, mode=mode)
    with pytest.raises(InferenceError):
        with client:
            pytest.fail("Invalid server became ready")
    assert client.process.poll() is not None
    assert client._log is None


@pytest.mark.parametrize("mode", ["hang", "loading"])
def test_startup_deadline_is_bounded(server_files, mode):
    client = server(server_files, mode=mode)
    client.startup_timeout = 0.2
    started = time.monotonic()
    with pytest.raises(InferenceError, match="startup deadline"):
        client.__enter__()
    assert time.monotonic() - started < 2
    assert client.process.poll() is not None


def test_request_deadline_stops_dripping_response_and_owned_process(server_files):
    client = server(server_files, mode="drip", request_timeout=0.15)
    with client:
        started = time.monotonic()
        with pytest.raises(InferenceError, match="timed out"):
            client.render([{"role": "user", "content": "Question"}])
        assert time.monotonic() - started < 2
        assert client.process.poll() is not None


def test_malformed_json_aborts_process(server_files):
    client = server(server_files, mode="bad-json")
    with client:
        with pytest.raises(InferenceError, match="invalid JSON") as error:
            client.tokenize("question")
        assert client.process.poll() is not None
        assert error.value.response["http_status"] == 200
        assert error.value.response["raw_body"] == "wrong"
        assert error.value.response["body_encoding"] == "utf-8"
        assert error.value.response["body_bytes_retained"] == 5
        assert error.value.response["body_truncated"] is False


@pytest.mark.parametrize(
    "mode,status,raw,encoding",
    [
        ("nonobject", 200, b'["unexpected", "array"]', "utf-8"),
        ("bad-bytes", 502, b"\xff\xfe\x80broken", "base64"),
    ],
)
def test_invalid_http_body_preserves_exact_bytes_and_closes_child(
    server_files, mode, status, raw, encoding
):
    with server(server_files, mode=mode) as client:
        with pytest.raises(InferenceError) as error:
            client.tokenize("question")
        result = error.value.response
        retained = (
            result["raw_body"].encode("utf-8")
            if encoding == "utf-8"
            else base64.b64decode(result["raw_body"])
        )
        assert retained == raw
        assert result["http_status"] == status
        assert result["body_encoding"] == encoding
        assert result["body_bytes_retained"] == result["body_bytes_read"] == len(raw)
        assert result["body_truncated"] is False
        assert client.process.poll() is not None


def test_oversize_http_body_retains_only_bounded_prefix(server_files, monkeypatch):
    monkeypatch.setattr(inference, "MAX_RESPONSE_BYTES", 1024)
    with server(server_files, mode="oversize") as client:
        with pytest.raises(InferenceError, match="size limit") as error:
            client.tokenize("question")
        result = error.value.response
        original = json.dumps({"content": "x" * 4096}).encode()
        assert result["raw_body"].encode() == original[:1024]
        assert result["http_status"] == 200
        assert result["body_bytes_retained"] == result["body_byte_limit"] == 1024
        assert result["body_bytes_read"] == 1025
        assert result["body_truncated"] is True
        assert client.process.poll() is not None


@pytest.mark.parametrize(
    "invalid",
    [
        {"stop_type": "limit"},
        {"truncated": True},
        {"stop": False},
        {"tokens_evaluated": 9},
        {"tokens_predicted": 100},
        {"tokens_predicted": True},
        {"tokens": None},
        {"tokens": [201, 202, 203]},
        {"tokens": [201, 202, 203, True]},
        {"tokens": [201, 202, 203, 204.0]},
        {"tokens": [201, 202, 203, -1]},
        {"tokens": [201, 202, 203, "204"]},
        {"content": None},
    ],
)
def test_incomplete_generation_is_rejected_with_raw_response(server_files, invalid):
    with server(server_files, completion=invalid) as client:
        with pytest.raises(InferenceError, match="incomplete") as error:
            client.complete([101, 42, 102], 32)
        assert all(error.value.response[key] == value for key, value in invalid.items())


@pytest.mark.parametrize("count", [True, 1.0])
def test_noninteger_prompt_token_count_is_rejected(server_files, count):
    with server(server_files, completion={"tokens_evaluated": count}) as client:
        with pytest.raises(InferenceError, match="incomplete"):
            client.complete([101], 32)


def test_bad_inputs_and_context_overflow_fail_before_request(server_files):
    with server(server_files, context_size=64) as client:
        for tokens, count in (([], 4), ([True], 4), ([1], 0), ([1] * 32, 32)):
            with pytest.raises(ValueError):
                client.complete(tokens, count)
        with pytest.raises(ValueError, match="schema"):
            client.complete([1], 4, json_schema=[])
        with pytest.raises(ValueError, match="Messages"):
            client.render([{"role": "tool", "content": "Untrusted"}])
        assert client.process.poll() is None


def test_context_exception_and_repeated_close_clean_up(server_files):
    client = server(server_files)
    with pytest.raises(RuntimeError, match="caller failed"):
        with client:
            raise RuntimeError("caller failed")
    client.close()
    assert client.process.poll() is not None
    with pytest.raises(InferenceError, match="not running"):
        client.tokenize("after close")


def test_shutdown_kills_server_that_ignores_termination(server_files):
    client = server(server_files, mode="ignore-terminate")
    with client:
        process = client.process
    assert process.returncode < 0


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan")])
def test_invalid_deadlines_fail_early(server_files, value):
    binary, model, log = server_files
    with pytest.raises(ValueError, match="timeout"):
        LocalServer(binary, model, log_path=log, request_timeout=value)


def test_missing_assets_and_info_fail_cleanly(tmp_path):
    client = LocalServer(tmp_path / "missing", tmp_path / "missing.gguf")
    with pytest.raises(FileNotFoundError, match="Build llama-server"):
        client.__enter__()
    with pytest.raises(InferenceError, match="verified"):
        client.info()


MESSAGES = [{"role": "user", "content": "Question"}]
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}}


def test_reasoning_chat_keeps_raw_output_and_checks_the_actual_prompt(server_files):
    binary, model, log = server_files
    capture = model.with_suffix(".jsonl")
    model.write_text(json.dumps({"capture": str(capture)}))
    with LocalServer(
        binary, model, log_path=log, startup_timeout=5, reasoning_budget_tokens=16
    ) as client:
        prompt = client.render(MESSAGES, json_schema=SCHEMA)
        tokens = client.tokenize(prompt)
        result = client.complete(tokens, 32, messages=MESSAGES, json_schema=SCHEMA)
        assert result["content"] == '{"answer":"Paris"}'
        assert result["reasoning_content"] == "Check evidence."
        assert result["raw_content"] == result["chat_response"]["__verbose"]["content"]
        assert result["raw_content"].startswith("<think>")
        assert result["tokens_predicted"] == 6
        assert result["tokens_evaluated"] == len(tokens)
        info = client.info()
        assert info["enable_thinking"] is True
        assert info["reasoning_budget_tokens"] == 16
        assert info["reasoning_format"] == "deepseek"
        assert info["completion_endpoint"] == "/v1/chat/completions"
        assert info["seed"] == 42
        assert (info["temperature"], info["top_p"], info["top_k"], info["min_p"]) == (
            0.6,
            0.95,
            20,
            0.0,
        )
    records = [json.loads(line) for line in capture.read_text().splitlines()]
    assert [record["path"] for record in records] == [
        "/apply-template",
        "/tokenize",
        "/apply-template",
        "/tokenize",
        "/v1/chat/completions/input_tokens",
        "/v1/chat/completions",
    ]
    for record in (records[1], records[3]):
        assert record["body"]["add_special"] is True
        assert record["body"]["parse_special"] is True
    body = records[-1]["body"]
    assert records[2]["body"] == records[4]["body"] == body
    for key, value in records[0]["body"].items():
        assert body[key] == value
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert body["reasoning_budget_tokens"] == 16
    assert body["reasoning_budget_message"] == ""
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["n_predict"] == 32
    assert body["stream"] is False
    assert body["verbose"] is body["return_tokens"] is True
    assert body["cache_prompt"] is False
    assert body["samplers"] == info["samplers"]


def test_reasoning_rejects_changed_tokens_even_when_the_count_matches(server_files):
    with server(server_files, reasoning_budget_tokens=16) as client:
        with pytest.raises(InferenceError, match="packed token IDs"):
            client.complete([101, 99, 102], 32, messages=MESSAGES, json_schema=SCHEMA)


@pytest.mark.parametrize("count", [2, True, 3.0, "3"])
def test_reasoning_rejects_different_chat_token_count(server_files, count):
    with server(server_files, reasoning_budget_tokens=16, count=count) as client:
        with pytest.raises(InferenceError, match="token counts") as error:
            client.complete([101, 42, 102], 32, messages=MESSAGES, json_schema=SCHEMA)
        assert error.value.response == {"input_tokens": count}


@pytest.mark.parametrize(
    "native",
    [
        {"stop_type": "limit"},
        {"truncated": True},
        {"stop": False},
        {"content": None},
        {"prompt": "changed prompt"},
        {"prompt": None},
        {"tokens_evaluated": 4},
        {"tokens_evaluated": 3.0},
        {"tokens_predicted": 33},
        {"tokens_predicted": True},
        {"tokens": [1]},
        {"tokens": [1, 2, 3, 4, 5, True]},
    ],
)
def test_reasoning_rejects_native_failures_without_losing_raw_output(
    server_files, native
):
    with server(server_files, reasoning_budget_tokens=16, completion=native) as client:
        with pytest.raises(InferenceError, match="incomplete") as error:
            client.complete([101, 42, 102], 32, messages=MESSAGES, json_schema=SCHEMA)
        response = error.value.response
        assert (
            response["choices"][0]["message"]["reasoning_content"] == "Check evidence."
        )
        assert all(response["__verbose"][key] == value for key, value in native.items())


@pytest.mark.parametrize(
    "chat",
    [
        {"__verbose": None},
        {"choices": []},
        {"choices": [None]},
        {"choices": [{"message": {"content": "{}"}}]},
        {"usage": {"prompt_tokens": 3, "completion_tokens": 6, "total_tokens": 10}},
        {"usage": {"prompt_tokens": 3, "completion_tokens": 6.0, "total_tokens": 9}},
        {"model": "wrong-model"},
    ],
)
def test_reasoning_rejects_inconsistent_chat_metadata(server_files, chat):
    with server(server_files, reasoning_budget_tokens=16, chat=chat) as client:
        with pytest.raises(InferenceError, match="inconsistent") as error:
            client.complete([101, 42, 102], 32, messages=MESSAGES, json_schema=SCHEMA)
        assert all(error.value.response[key] == value for key, value in chat.items())


@pytest.mark.parametrize(
    "finish,extra",
    [
        ("length", {}),
        ("tool_calls", {}),
        ("stop", {"reasoning_content": []}),
        ("stop", {"content": None}),
        ("stop", {"content": ""}),
        ("stop", {"content": " \t\n"}),
        ("stop", {"tool_calls": [{"name": "unexpected"}]}),
    ],
)
def test_reasoning_checks_finish_and_message_even_with_complete_json(
    server_files, finish, extra
):
    message = {"role": "assistant", "content": '{"answer":"Paris"}', **extra}
    chat = {"choices": [{"index": 0, "finish_reason": finish, "message": message}]}
    with server(server_files, reasoning_budget_tokens=16, chat=chat) as client:
        with pytest.raises(InferenceError, match="incomplete") as error:
            client.complete([101, 42, 102], 32, messages=MESSAGES, json_schema=SCHEMA)
        assert error.value.response["choices"] == chat["choices"]


def test_reasoning_preflight_and_generation_share_one_deadline(server_files):
    client = server(
        server_files, reasoning_budget_tokens=16, delay=0.08, request_timeout=0.2
    )
    with client:
        started = time.monotonic()
        with pytest.raises(InferenceError, match="timed out"):
            client.complete([101, 42, 102], 32, messages=MESSAGES, json_schema=SCHEMA)
        assert time.monotonic() - started < 0.6
        assert client.process.poll() is not None


def test_reasoning_server_parser_error_is_retained_and_stops_child(server_files):
    with server(server_files, reasoning_budget_tokens=16, mode="chat-error") as client:
        with pytest.raises(InferenceError, match="HTTP 500") as error:
            client.complete([101, 42, 102], 32, messages=MESSAGES, json_schema=SCHEMA)
        assert error.value.response == {"error": {"message": "Output parser failed"}}
        assert client.process.poll() is not None


@pytest.mark.parametrize("value", [-1, True, 1.5, "1", 2**31])
def test_invalid_reasoning_budget_fails_before_start(server_files, value):
    binary, model, _ = server_files
    with pytest.raises(ValueError, match="Reasoning budget"):
        LocalServer(binary, model, reasoning_budget_tokens=value)


def test_reasoning_requires_schema_messages_and_space_for_final_answer(server_files):
    with server(server_files, reasoning_budget_tokens=16, context_size=64) as client:
        with pytest.raises(ValueError, match="schema"):
            client.render(MESSAGES)
        with pytest.raises(ValueError, match="Messages"):
            client.complete([101, 42, 102], 32, json_schema=SCHEMA)
        with pytest.raises(ValueError, match="total output limit"):
            client.complete([101, 42, 102], 16, messages=MESSAGES, json_schema=SCHEMA)
        with pytest.raises(ValueError, match="context budget"):
            client.complete([1] * 32, 32, messages=MESSAGES, json_schema=SCHEMA)


def test_zero_reasoning_budget_still_uses_the_chat_contract(server_files):
    with server(server_files, reasoning_budget_tokens=0) as client:
        result = client.complete(
            [101, 42, 102], 32, messages=MESSAGES, json_schema=SCHEMA
        )
        assert result["content"] == '{"answer":"Paris"}'
        assert client.info()["enable_thinking"] is True
