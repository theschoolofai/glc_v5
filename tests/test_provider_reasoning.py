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


def test_a_daily_quota_is_benched_for_the_day_not_for_a_minute():
    """The bug: a key exhausted until midnight was benched for 60 seconds.

    Two independent reasons the RPD branch was unreachable for Gemini, whose
    429 body is reproduced here verbatim from a live gateway:

    * every Google quota message contains the word "quota", and that check runs
      first, so the per-minute branch always won;
    * Google writes the window as ``PerDay``, one word, and the matcher looked
      for ``"per day"`` with a space.

    A key that cannot serve another request until tomorrow therefore came back
    as a candidate every 60 seconds, all day.
    """

    class _E(Exception):
        status = 429

    gemini_daily = _E(
        "gemini HTTP 429: You exceeded your current quota, please check your plan and "
        "billing details. For more information on this error, head to: "
        "https://ai.google.dev/gemini-api/docs/rate-limits.\n"
        "* Quota exceeded for metric: generativelanguage.googleapis.com/"
        "generate_content_free_tier_requests, limit: 50, quota_dimensions: PerDay"
    )
    seconds, reason = C._backoff_for(gemini_daily)
    assert reason == "RPD quota burned"
    assert seconds == 3600, "a key that is done until tomorrow must not return in a minute"

    # A genuine per-minute burst still gets the short bench.
    minute, reason = C._backoff_for(_E("Quota exceeded ... limit_value: 15 PerMinute"))
    assert reason == "RPM quota burned"
    assert minute == 60
