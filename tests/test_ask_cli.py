import json
from pathlib import Path

import pytest

from arm64_rag.cli import main
from arm64_rag.corpus import load_paths
from arm64_rag.index import update_index
from arm64_rag.inference import InferenceError

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "eval/answers-protocol.json").read_bytes())
RUNTIME_INFO = {
    "model_sha256": "pin",
    "context_size": PROTOCOL["settings"]["context_size"],
    "reasoning_budget_tokens": PROTOCOL["settings"].get("reasoning_budget_tokens"),
}


@pytest.fixture
def asking(tmp_path, monkeypatch):
    source = tmp_path / "note.txt"
    source.write_text("The batch size is 128 tokens.\n")
    db = tmp_path / "index.sqlite"
    update_index(db, load_paths([source], tmp_path))
    config = tmp_path / "model.json"
    config.write_text(
        json.dumps(
            {
                "directory": "models",
                "model": {"filename": "fixture.gguf", "sha256": "pin"},
            }
        )
    )

    class Runtime:
        content = json.dumps(
            {
                "status": "answered",
                "claims": [
                    {
                        "text": "The batch size is 128 tokens.",
                        "citations": [{"label": "S1"}],
                    }
                ],
            }
        )
        closed = False
        calls = 0

        def __init__(self, *args, **kwargs):
            self.settings = {
                "context_size": kwargs.get("context_size", 8192),
                "reasoning_budget_tokens": kwargs.get("reasoning_budget_tokens"),
            }

        def __enter__(self):
            return self

        def __exit__(self, *args):
            Runtime.closed = True

        def info(self):
            return {"model_sha256": "pin", **self.settings}

        def render(self, messages, *, json_schema=None):
            return json.dumps(messages)

        def tokenize(self, prompt):
            return list(range(len(prompt)))

        def complete(self, token_ids, max_tokens, *, json_schema, messages=None):
            Runtime.calls += 1
            return {"content": Runtime.content}

    monkeypatch.setattr("arm64_rag.inference.LocalServer", Runtime)
    trace = tmp_path / "answer.json"

    def invoke(*extra, question="What is the batch size?"):
        monkeypatch.setattr(
            "sys.argv",
            [
                "arm64-rag",
                "ask",
                question,
                "--db",
                str(db),
                "--answer-model-config",
                str(config),
                "--protocol",
                str(ROOT / "eval/answers-protocol.json"),
                "--trace",
                str(trace),
                *extra,
            ],
        )
        main()

    return invoke, Runtime, trace


def test_answer_cli_renders_citation_and_saves_source_trace(asking, capsys):
    invoke, runtime, trace = asking
    invoke()
    output = capsys.readouterr()
    assert "128 tokens. [S1]" in output.out
    assert "note.txt:1-1" in output.out
    assert output.err == ""
    record = json.loads(trace.read_bytes())
    assert record["answer"]["claims"][0]["citations"][0]["quote"] == (
        "The batch size is 128 tokens.\n"
    )
    assert record["packed"]["prompt_tokens"] > 0
    assert runtime.closed


def test_answer_cli_returns_json_without_exposing_prompt(asking, capsys):
    invoke, _, _ = asking
    invoke("--json")
    result = json.loads(capsys.readouterr().out)
    assert set(result) == {"answer", "failure"}
    assert result["failure"] is None


def test_json_output_escapes_terminal_controls_without_changing_decoded_answer(
    asking, capsys
):
    invoke, runtime, trace = asking
    text = "Café batch size: 128 tokens.\x9b2J\x1b[2J"
    value = json.loads(runtime.content)
    value["claims"][0]["text"] = text
    runtime.content = json.dumps(value)
    invoke("--json")
    output = capsys.readouterr()
    assert "\x9b" not in output.out and "\x1b" not in output.out
    assert "\\u009b" in output.out and "\\u001b" in output.out
    answer = json.loads(output.out)["answer"]
    assert answer["claims"][0]["text"] == text
    assert json.loads(trace.read_bytes())["answer"] == answer


def test_bad_model_output_is_saved_but_not_presented_as_an_answer(asking, capsys):
    invoke, runtime, trace = asking
    runtime.content = "invalid PRIVATE RAW OUTPUT"
    with pytest.raises(SystemExit) as error:
        invoke()
    assert error.value.code == 1
    output = capsys.readouterr()
    assert "PRIVATE RAW OUTPUT" not in output.out + output.err
    record = json.loads(trace.read_bytes())
    assert record["answer"] is None
    assert record["failure"]
    assert record["raw_completion"]["content"] == runtime.content
    assert runtime.closed


@pytest.mark.parametrize("location", ["claim", "raw_response"])
def test_invalid_unicode_output_is_rejected_and_preserved_in_nonempty_trace(
    asking, capsys, location
):
    invoke, runtime, trace = asking
    if location == "claim":
        value = json.loads(runtime.content)
        value["claims"][0]["text"] = "Invalid \ud800 claim."
        runtime.content = json.dumps(value)
    else:
        runtime.content = "Malformed raw content: \ud800"
    with pytest.raises(SystemExit) as error:
        invoke()
    assert error.value.code == 1
    assert trace.stat().st_size > 0
    record = json.loads(trace.read_bytes())
    assert record["answer"] is None
    assert record["failure"]
    assert record["raw_completion"]["content"] == runtime.content
    output = capsys.readouterr()
    assert "Invalid" not in output.out and "Malformed raw" not in output.out
    assert "Details saved" in output.err
    assert runtime.closed


def test_empty_search_abstains_without_generation(asking, capsys):
    invoke, runtime, _ = asking
    invoke(question="unfindablezebrafish")
    assert "do not establish an answer" in capsys.readouterr().out
    assert runtime.calls == 0
    assert runtime.closed


def test_existing_trace_is_preserved(asking, capsys):
    invoke, runtime, trace = asking
    trace.write_text("prior evidence")
    with pytest.raises(SystemExit) as error:
        invoke()
    assert error.value.code == 1
    assert "new trace path" in capsys.readouterr().err
    assert trace.read_text() == "prior evidence"
    assert runtime.calls == 0


@pytest.mark.parametrize(
    ("option", "content"),
    [
        ("--protocol", "{"),
        ("--protocol", "null"),
        ("--protocol", '{"settings": {}}'),
        ("--protocol", json.dumps({**PROTOCOL, "settings": []})),
        ("--protocol", json.dumps({**PROTOCOL, "model_output_schema": []})),
        *[
            (
                "--protocol",
                json.dumps(
                    {**PROTOCOL, "settings": {**PROTOCOL["settings"], key: value}}
                ),
            )
            for key, value in (
                ("context_size", True),
                ("max_output_tokens", "2048"),
                ("max_passages", None),
            )
        ],
        ("--answer-model-config", "{"),
        ("--answer-model-config", "[]"),
        ("--answer-model-config", '{"directory": "models", "model": {}}'),
        (
            "--answer-model-config",
            '{"directory": "models", "model": {"filename": [], "sha256": "pin"}}',
        ),
    ],
)
def test_invalid_configuration_saves_trace_before_starting_server(
    asking, tmp_path, monkeypatch, capsys, option, content
):
    invoke, runtime, trace = asking
    config = tmp_path / "invalid.json"
    config.write_text(content)

    def unexpected_start(*args, **kwargs):
        pytest.fail("Invalid configuration must fail before starting a server.")

    monkeypatch.setattr(runtime, "__init__", unexpected_start)
    with pytest.raises(SystemExit) as error:
        invoke(option, str(config), "--json")
    assert error.value.code == 1
    record = json.loads(trace.read_bytes())
    assert record["failure"] == "Invalid answer protocol or model configuration."
    assert record["answer"] is record["runtime"] is record["raw_completion"] is None
    output = capsys.readouterr()
    assert json.loads(output.out)["failure"] == record["failure"]
    assert "Details saved" in output.err


def test_missing_configuration_saves_failure_trace(asking, tmp_path, capsys):
    invoke, runtime, trace = asking
    with pytest.raises(SystemExit) as error:
        invoke("--protocol", str(tmp_path / "missing.json"))
    assert error.value.code == 1
    record = json.loads(trace.read_bytes())
    assert "missing.json" in record["failure"]
    assert record["answer"] is record["runtime"] is record["raw_completion"] is None
    assert runtime.calls == 0
    assert "Details saved" in capsys.readouterr().err


def test_display_escapes_model_and_source_controls_without_changing_trace(
    asking, tmp_path, capsys
):
    invoke, runtime, trace = asking
    text = "The batch size is 128 tokens.\x1b[2J\x9b31m\r\t"
    value = json.loads(runtime.content)
    value["claims"][0]["text"] = text
    runtime.content = json.dumps(value)
    source = tmp_path / "note\x1b[2J.txt"
    source.write_text("The batch size is 128 tokens.\n")
    update_index(tmp_path / "index.sqlite", load_paths([source], tmp_path), sync=True)
    invoke()
    output = capsys.readouterr()
    assert not any(
        character in output.out for character in ("\x1b", "\x9b", "\r", "\t")
    )
    assert "\\u001b[2J\\u009b31m\\u000d\\u0009" in output.out
    assert "note\\u001b[2J.txt:1-1" in output.out
    record = json.loads(trace.read_bytes())
    assert record["answer"]["claims"][0]["text"] == text
    assert record["answer"]["claims"][0]["citations"][0]["source_id"] == source.name
    assert record["raw_completion"]["content"] == runtime.content


def test_display_escapes_revision_controls_and_preserves_normal_unicode():
    from arm64_rag.answers import render_answer

    record = {
        "failure": None,
        "answer": {
            "status": "answered",
            "claims": [
                {
                    "text": "Café uses four workers.",
                    "citations": [
                        {
                            "label": "S1",
                            "source_id": "café",
                            "revision": "1\x7f\n",
                            "line_start": 1,
                            "line_end": 2,
                        }
                    ],
                }
            ],
        },
    }
    original = json.dumps(record)
    display = render_answer(record)
    assert "Café uses four workers." in display
    assert "café:1-2 (revision 1\\u007f\\u000a)" in display
    assert json.dumps(record) == original


def test_display_preserves_multiline_claims(asking, capsys):
    invoke, runtime, trace = asking
    value = json.loads(runtime.content)
    value["claims"][0]["text"] = (
        "- Batch size: 128 tokens.\n- Source: the batching note."
    )
    runtime.content = json.dumps(value)
    invoke()
    output = capsys.readouterr().out
    assert "128 tokens.\n- Source: the batching note. [S1]" in output
    assert "\\u000a" not in output
    record = json.loads(trace.read_bytes())
    assert record["answer"]["claims"][0]["text"] == value["claims"][0]["text"]


@pytest.mark.parametrize("stage", ["render", "tokenize", "__enter__"])
def test_runtime_failure_retains_minimal_trace_and_hides_raw_response(
    asking, monkeypatch, capsys, stage
):
    invoke, runtime, trace = asking
    raw = {"content": "PRIVATE RAW RESPONSE"}

    def fail(*args, **kwargs):
        raise InferenceError("Template request timed out.", response=raw)

    monkeypatch.setattr(runtime, stage, fail)
    with pytest.raises(SystemExit) as error:
        invoke()
    assert error.value.code == 1
    output = capsys.readouterr()
    assert "PRIVATE RAW RESPONSE" not in output.out + output.err
    assert "Details saved" in output.err
    record = json.loads(trace.read_bytes())
    assert record["question"] == "What is the batch size?"
    assert record["answer"] is None
    assert record["failure"] == "Template request timed out."
    assert record["raw_completion"] == raw
    assert record["runtime"] == (None if stage == "__enter__" else RUNTIME_INFO)
    assert runtime.closed is (stage != "__enter__")


def test_keyboard_interrupt_is_saved_then_propagated(asking, monkeypatch, capsys):
    invoke, runtime, trace = asking

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(runtime, "tokenize", interrupt)
    with pytest.raises(KeyboardInterrupt):
        invoke()
    record = json.loads(trace.read_bytes())
    assert record["question"] == "What is the batch size?"
    assert record["failure"] == "Interrupted by user."
    assert record["runtime"] == RUNTIME_INFO
    assert record["answer"] is None
    assert runtime.closed
    output = capsys.readouterr()
    assert output.out == output.err == ""
