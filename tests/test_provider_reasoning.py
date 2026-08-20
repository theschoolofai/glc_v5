"""The reasoning dial and the timeout cooldown.

Both of these were found by calling every provider and reading what came back,
and both produced the same symptom from opposite directions: a call that was
billed in full and returned nothing useful.

* `reasoning: "off"` emitted no field at all, so a model that thinks by default
  kept thinking. zai-glm-4.7 spent all 512 tokens of a `max_tokens: 512` budget
  in its reasoning channel and returned `content: ""` — measured, 3/3.
* An `httpx.ReadTimeout` stringifies to the EMPTY string, so the cooldown matcher
  recognised nothing and handed back a ZERO-second bench. The NVIDIA endpoint
  that hangs for 180 s was therefore a candidate again on the very next request.
"""

from __future__ import annotations

import httpx
import pytest

from glc import providers as P
from glc.routes import chat as C
from glc.routing import policy as PO


@pytest.fixture(autouse=True)
def _fresh():
    PO.reset_policy()
    yield
    PO.reset_policy()


class _Compat(P.OpenAICompatProvider):
    name = "test"

    def __init__(self):
        super().__init__("k", "openai/gpt-oss-120b", "https://example.invalid/v1")


# ── the reasoning dial ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model",
    ["openai/gpt-oss-120b", "zai-glm-4.7", "nvidia/nemotron-3-super-120b-a12b:free"],
)
def test_off_is_stated_for_a_model_that_thinks_by_default(model):
    body: dict = {}
    assert _Compat()._apply_reasoning(body, "off", model) is True
    assert body["reasoning_effort"] == P.REASONING_NONE
    assert body["chat_template_kwargs"] == {"thinking": False}


def test_off_stays_silent_for_a_model_with_no_thinking_channel():
    """gpt-4.1 has nothing to switch off; sending the field would only risk a
    400 from a server that has never heard of it."""
    body: dict = {}
    assert _Compat()._apply_reasoning(body, "off", "openai/gpt-4.1") is False
    assert body == {}


def test_an_explicit_effort_reaches_the_open_weight_thinking_families():
    """These were accepted by the gateway and then silently dropped, because the
    hint list predated the models."""
    for model in ("zai-glm-4.7", "nvidia/nemotron-3-super-120b-a12b:free", "deepseek-ai/deepseek-v4-pro"):
        body: dict = {}
        assert _Compat()._apply_reasoning(body, "high", model) is True, model
        assert body["reasoning_effort"] == "high"


def test_no_dial_means_no_field():
    body: dict = {}
    assert _Compat()._apply_reasoning(body, None, "openai/gpt-oss-120b") is False
    assert body == {}


@pytest.mark.asyncio
async def test_a_rejected_dialect_is_healed_one_complaint_at_a_time(monkeypatch):
    """Groq's real sequence: it rejects `chat_template_kwargs` as unsupported and
    only THEN tells you `reasoning_effort` must be low|medium|high. One retry is
    not enough, and giving up loses the whole call."""
    seen: list[dict] = []

    class _R:
        def __init__(self, status, text):
            self.status_code, self.text = status, text

        def json(self):
            return {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            }

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, json=None):
            seen.append(dict(json))
            if "chat_template_kwargs" in json:
                return _R(400, '{"error":{"message":"property \'chat_template_kwargs\' is unsupported"}}')
            if json.get("reasoning_effort") == P.REASONING_NONE:
                return _R(
                    400,
                    '{"error":{"message":"`reasoning_effort` must be one of `low`, `medium`, or `high`"}}',
                )
            return _R(200, "")

    monkeypatch.setattr(P.httpx, "AsyncClient", lambda **kw: _Client())
    out = await _Compat().chat([{"role": "user", "content": "hi"}], reasoning="off", max_tokens=64)
    assert out["text"] == "ok"
    assert len(seen) == 3
    # It landed on the least effort the server admits rather than abandoning the
    # dial: "off" means "think as little as you are allowed to".
    assert seen[-1]["reasoning_effort"] == "low"
    assert "chat_template_kwargs" not in seen[-1]
    assert out["reasoning_applied"] is True


@pytest.mark.asyncio
async def test_healing_gives_up_rather_than_spinning(monkeypatch):
    class _R:
        status_code, text = 400, '{"error":{"message":"`reasoning_effort` is unsupported"}}'

    class _Client:
        def __init__(self):
            self.calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, json=None):
            self.calls += 1
            assert self.calls < 10, "the healing loop must be bounded"
            return _R()

    monkeypatch.setattr(P.httpx, "AsyncClient", lambda **kw: _Client())
    with pytest.raises(P.ProviderError):
        await _Compat().chat([{"role": "user", "content": "hi"}], reasoning="off", max_tokens=64)


# ── the provider's own reasoning field ──────────────────────────────────────
#
# Groq and Cerebras always return reasoning in its own `message.reasoning`
# field -- confirmed live, even when `reasoning` is completely unset. Those
# tokens are generated and billed (`usage.completion_tokens_details.
# reasoning_tokens`, measured live: 49 at unset, 21 at "low", 189 at "high",
# same prompt) and were read nowhere in this file. G8 in the S17 ledger.


@pytest.mark.asyncio
async def test_the_providers_own_reasoning_field_is_captured_not_discarded(monkeypatch):
    class _R:
        status_code = 200

        def json(self):
            return {
                "choices": [{
                    "message": {"content": "391", "reasoning": "17*24=408, sqrt(289)=17, 408-17=391"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 12, "completion_tokens": 31},
            }

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, json=None):
            return _R()

    monkeypatch.setattr(P.httpx, "AsyncClient", lambda **kw: _Client())
    out = await _Compat().chat([{"role": "user", "content": "hi"}], max_tokens=64)

    assert out["reasoning_text"] == "17*24=408, sqrt(289)=17, 408-17=391"


@pytest.mark.asyncio
async def test_reasoning_text_is_none_when_the_provider_sends_none(monkeypatch):
    class _R:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": "391"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            }

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, json=None):
            return _R()

    monkeypatch.setattr(P.httpx, "AsyncClient", lambda **kw: _Client())
    out = await _Compat().chat([{"role": "user", "content": "hi"}], max_tokens=64)

    assert out["reasoning_text"] is None


def test_chat_response_accepts_reasoning_text():
    from glc.llm_schemas import ChatResponse

    with_reasoning = ChatResponse(provider="groq", model="openai/gpt-oss-120b", text="391",
                                  reasoning_text="because 408-17=391")
    without_reasoning = ChatResponse(provider="groq", model="openai/gpt-oss-120b", text="391")

    assert with_reasoning.reasoning_text == "because 408-17=391"
    assert without_reasoning.reasoning_text is None


# ── Gemini's own reasoning channel ───────────────────────────────────────────
#
# Gemini marks reasoning content with a sibling `thought: true` on the response
# part, not a separate response field like Groq/Cerebras's `message.reasoning`.
# Two compounding gaps: the request never set `includeThoughts`, so thought
# parts never came back at all; and if they had, the old single-line
# `text = "".join(p.get("text","") for p in parts if "text" in p)` would have
# merged them straight into the visible answer, since it only ever checked
# `"text" in p`, never the `thought` flag.


@pytest.mark.asyncio
async def test_geminis_own_reasoning_field_is_captured_not_discarded(monkeypatch):
    class _R:
        status_code = 200

        def json(self):
            return {
                "candidates": [{
                    "content": {"parts": [
                        {"text": "17*24=408, sqrt(289)=17, 408-17=391", "thought": True},
                        {"text": "391"},
                    ]},
                    "finishReason": "STOP",
                }],
                "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 4},
            }

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            return _R()

    monkeypatch.setattr(P.httpx, "AsyncClient", lambda **kw: _Client())
    out = await P.GeminiProvider("k", "gemini-2.5-flash", None).chat(
        [{"role": "user", "content": "hi"}], reasoning="high", max_tokens=64
    )

    assert out["reasoning_text"] == "17*24=408, sqrt(289)=17, 408-17=391"


@pytest.mark.asyncio
async def test_gemini_text_excludes_thought_content_when_both_are_present(monkeypatch):
    """The regression test: thought parts must never land in the visible answer."""
    class _R:
        status_code = 200

        def json(self):
            return {
                "candidates": [{
                    "content": {"parts": [
                        {"text": "scratch work: 6*7=42", "thought": True},
                        {"text": "42"},
                    ]},
                    "finishReason": "STOP",
                }],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 1},
            }

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            return _R()

    monkeypatch.setattr(P.httpx, "AsyncClient", lambda **kw: _Client())
    out = await P.GeminiProvider("k", "gemini-2.5-flash", None).chat(
        [{"role": "user", "content": "hi"}], reasoning="high", max_tokens=64
    )

    assert out["text"] == "42"
    assert "scratch work" not in out["text"]


@pytest.mark.asyncio
async def test_gemini_reasoning_text_is_none_when_no_thought_parts_return(monkeypatch):
    class _R:
        status_code = 200

        def json(self):
            return {
                "candidates": [{
                    "content": {"parts": [{"text": "391"}]},
                    "finishReason": "STOP",
                }],
                "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 4},
            }

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            return _R()

    monkeypatch.setattr(P.httpx, "AsyncClient", lambda **kw: _Client())
    out = await P.GeminiProvider("k", "gemini-2.5-flash", None).chat(
        [{"role": "user", "content": "hi"}], max_tokens=64
    )

    assert out["reasoning_text"] is None
    assert out["text"] == "391"


# ── the timeout cooldown ────────────────────────────────────────────────────


def test_a_bare_read_timeout_is_recognised_as_a_timeout():
    """The bug: str(httpx.ReadTimeout("")) is "", so nothing matched and the
    provider was benched for zero seconds after burning the full client timeout."""
    err = httpx.ReadTimeout("")
    assert str(err) == ""
    secs, reason = C._backoff_for(err)
    assert reason == "timeout"
    assert secs >= 180, "a bench shorter than the timeout that produced it is no bench at all"


def test_the_cooldown_comes_from_routing_yaml(tmp_path, monkeypatch):
    p = tmp_path / "routing.yaml"
    p.write_text("version: 1\ntiers:\n  T: {order: [groq]}\nladder: [T]\nbackoff:\n  timeout: 42\n")
    monkeypatch.setenv("GLC_ROUTING_YAML", str(p))
    PO.reload_policy()
    assert C._backoff_for(httpx.ReadTimeout(""))[0] == 42


def test_the_shipped_config_benches_a_hanging_provider_for_longer_than_it_hangs():
    pol = PO.reload_policy()
    assert pol.backoff["timeout"] >= 180
    assert C._backoff_for(httpx.ConnectTimeout(""))[0] == pol.backoff["timeout"]


def test_the_other_failure_classes_keep_their_v3_numbers():
    class _E(Exception):
        status = 429

    assert C._backoff_for(_E("RPM quota exceeded"))[0] == 60
    assert C._backoff_for(_E("queue_exceeded"))[0] == 15
    assert C._backoff_for(_E("slow down"))[0] == 30
