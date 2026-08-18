"""Ollama never read the `reasoning` dial, and always discarded the model's
own separate reasoning field -- G2 in the S17 investigation ledger.

Ollama's real /api/chat accepts `think` as false/true or a graded "low"/
"medium"/"high" string -- confirmed live: an invalid value is rejected with
`must be "high", "medium", "low", "max", true, or false`. Measured live
against qwen3:8b: with nothing sent, the model burned an entire 300-token
budget thinking and returned `content: ""` after 142 seconds. Asking
explicitly for `reasoning: "off"` had NO effect -- `reasoning_applied: false`
either way, byte-for-byte identical to sending nothing -- proof the field
was never read at all, only ever forwarded as an unused parameter.

Ollama's response also separates reasoning into its own `message.thinking`
field, never merged into `content`. The gateway read only `content` and threw
`thinking` away, even though it was already generated and billed for.
"""
from __future__ import annotations

import pytest

from glc import providers as P
from glc.llm_schemas import ChatResponse


class _R:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload

    def json(self):
        return self._payload


class _Client:
    def __init__(self, response_json):
        self.seen: list[dict] = []
        self._response_json = response_json

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json=None):
        self.seen.append(dict(json))
        return _R(self._response_json)


def _ollama_reply(*, content="", thinking=None):
    message = {"role": "assistant", "content": content}
    if thinking is not None:
        message["thinking"] = thinking
    return {"message": message, "prompt_eval_count": 10, "eval_count": 5}


# ── the dial reaches Ollama's own field ─────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning,expected_think", [
    ("off", False),
    ("low", "low"),
    ("medium", "medium"),
    ("high", "high"),
])
async def test_reasoning_is_translated_into_ollamas_own_think_field(monkeypatch, reasoning, expected_think):
    client = _Client(_ollama_reply(content="391"))
    monkeypatch.setattr(P.httpx, "AsyncClient", lambda **kw: client)
    provider = P.OllamaProvider("qwen3:8b")

    out = await provider.chat([{"role": "user", "content": "hi"}], reasoning=reasoning)

    assert client.seen[0]["think"] == expected_think
    assert out["reasoning_applied"] is True


@pytest.mark.asyncio
async def test_unset_reasoning_sends_no_think_field(monkeypatch):
    """Deliberately unchanged: S17Code always sends "off" explicitly
    (gateway.py:38), so this path is never exercised by that caller. The fix
    does not attempt to make unset auto-suppress -- that is G3, a separate,
    deliberately deferred finding."""
    client = _Client(_ollama_reply(content=""))
    monkeypatch.setattr(P.httpx, "AsyncClient", lambda **kw: client)
    provider = P.OllamaProvider("qwen3:8b")

    out = await provider.chat([{"role": "user", "content": "hi"}])

    assert "think" not in client.seen[0]
    assert out["reasoning_applied"] is False


# ── Ollama's own reasoning field is captured, not discarded ────────────────


@pytest.mark.asyncio
async def test_thinking_field_is_captured_not_discarded(monkeypatch):
    client = _Client(_ollama_reply(
        content="391", thinking="17*24=408, sqrt(289)=17, 408-17=391"))
    monkeypatch.setattr(P.httpx, "AsyncClient", lambda **kw: client)
    provider = P.OllamaProvider("qwen3:8b")

    out = await provider.chat([{"role": "user", "content": "hi"}], reasoning="high")

    assert out["reasoning_text"] == "17*24=408, sqrt(289)=17, 408-17=391"


@pytest.mark.asyncio
async def test_reasoning_text_is_none_when_ollama_sends_no_thinking_field(monkeypatch):
    client = _Client(_ollama_reply(content="391"))
    monkeypatch.setattr(P.httpx, "AsyncClient", lambda **kw: client)
    provider = P.OllamaProvider("qwen3:8b")

    out = await provider.chat([{"role": "user", "content": "hi"}], reasoning="off")

    assert out["reasoning_text"] is None


# ── the new field reaches the schema a caller actually sees ────────────────


def test_chat_response_accepts_reasoning_text():
    with_reasoning = ChatResponse(provider="ollama", model="qwen3:8b", text="391",
                                  reasoning_text="because 408-17=391")
    without_reasoning = ChatResponse(provider="ollama", model="qwen3:8b", text="391")

    assert with_reasoning.reasoning_text == "because 408-17=391"
    assert without_reasoning.reasoning_text is None
