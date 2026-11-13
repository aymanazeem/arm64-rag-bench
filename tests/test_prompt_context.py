import hashlib
import json
from types import SimpleNamespace

from arm64_rag.answers import add_context, pack_prompt


def test_context_uses_spare_budget_without_displacing_original_passages():
    schema = {
        "type": "object",
        "required": ["answer"],
        "properties": {"answer": {"type": "string"}},
        "additionalProperties": False,
    }
    render_calls = []

    def render(messages, *, json_schema=None):
        render_calls.append((messages, json_schema))
        return json.dumps(messages)

    server = SimpleNamespace(
        render=render,
        tokenize=lambda text: list(range(len(text))),
    )
    sources = [
        {
            "chunk_id": f"chunk-{number}",
            "source_id": f"source-{number}",
            "revision": "v1",
            "line_start": 1,
            "line_end": 1,
            "headings": ["Guide"],
            "text": f"Original fact {number}.",
        }
        for number in range(2)
    ]
    packed = pack_prompt(
        "Which facts?",
        sources,
        server,
        max_passages=2,
        context_size=10000,
        max_output_tokens=100,
        json_schema=schema,
    )
    anchors = list(packed["sources"])
    expanded = []
    for anchor in anchors:
        item = {
            **anchor,
            "anchor_chunk_id": anchor["chunk_id"],
            "text": "Parent condition. " + anchor["text"],
        }
        del item["chunk_id"]
        expanded.append(item)
    packed["context_size"] = packed["prompt_tokens"] + 100 + 1
    original_prompt = packed["prompt"]
    add_context(packed, "Which facts?", expanded, server, json_schema=schema)
    assert packed["sources"] == anchors
    assert packed["anchors"] == anchors
    assert packed["prompt"] == original_prompt
    assert len(packed["context_omitted"]) == 2

    packed["context_size"] = 10000
    add_context(packed, "Which facts?", expanded, server, json_schema=schema)
    assert packed["sources"] == expanded
    assert packed["anchors"] == anchors
    assert packed["prompt"] != original_prompt
    assert packed["token_ids"] == server.tokenize(packed["prompt"])
    assert packed["context_omitted"] == []
    for messages, generation_schema in render_calls:
        assert (
            json.loads(messages[-1]["content"])["output_schema"]
            == generation_schema
            == schema
        )
    for anchor, source in zip(anchors, packed["sources"], strict=True):
        assert anchor["text"] in source["text"]


def test_skipped_large_window_leaves_room_for_a_later_window_at_the_budget_limit():
    server = SimpleNamespace(
        render=lambda messages, *, json_schema=None: json.dumps(messages),
        tokenize=lambda text: list(range(len(text))),
    )
    candidates = [
        {
            "chunk_id": f"chunk-{number}",
            "source_id": f"source-{number}",
            "revision": "v1",
            "line_start": 1,
            "line_end": 1,
            "headings": ["Guide"],
            "text": f"Original fact {number}.",
        }
        for number in range(2)
    ]
    packed = pack_prompt(
        "Which facts?",
        candidates,
        server,
        max_passages=2,
        context_size=10000,
        max_output_tokens=100,
    )
    anchors = list(packed["sources"])
    expanded = []
    for anchor, extra in zip(
        anchors, ["Long context. " * 100, "Condition. "], strict=True
    ):
        passage = {
            **anchor,
            "anchor_chunk_id": anchor["chunk_id"],
            "text": extra + anchor["text"],
        }
        del passage["chunk_id"]
        expanded.append(passage)
    packed["context_size"] = packed["prompt_tokens"] + len("Condition. ") + 100 + 1
    add_context(packed, "Which facts?", expanded, server)
    assert packed["anchors"] == anchors
    assert packed["sources"] == [anchors[0], expanded[1]]
    assert packed["context_omitted"] == [
        {"anchor_chunk_id": anchors[0]["chunk_id"], "reason": "token_budget"}
    ]
    assert (
        packed["prompt_tokens"] + packed["max_output_tokens"] + 1
        == packed["context_size"]
    )
    assert packed["token_ids"] == server.tokenize(packed["prompt"])
    assert (
        packed["prompt_sha256"] == hashlib.sha256(packed["prompt"].encode()).hexdigest()
    )
    messages = json.loads(packed["prompt"])
    supplied = json.loads(messages[-1]["content"])["sources"]
    assert [source["text"] for source in supplied] == [
        anchors[0]["text"],
        expanded[1]["text"],
    ]
