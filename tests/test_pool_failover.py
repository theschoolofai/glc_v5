"""A named *pool* is not a hard pin: a 429 on one Gemini key advances to the next.

`gemini` is a logical provider. `Router.expand` turns it into gemini_1..N, one
entry per independently-metered API key, and S17Code's gateway client says so in
as many words before it sends `provider: "gemini"`. The failover ring treated any
named provider as an explicit override and re-raised on the first error, so ten
configured keys delivered the throughput of one: `gemini_1 failed: gemini HTTP
429: You exceeded your current quota`, with nine keys idle.

Naming a *member* — `gemini_1` — is a different request and still binds hard, and
an error that no key can survive still stops on the first one rather than burning
the pool on a call that is already doomed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from glc import db
from glc import providers as P
from glc.llm_schemas import ChatRequest
from glc.routes.chat import chat
from glc.routing import LIMITS, Router


@pytest.fixture(autouse=True)
def _restore_limits():
    """`build_providers` adds one LIMITS row per key and these tests do the same
    by hand; put the shared table back so the next test sees the shipped one."""
    before = dict(LIMITS)
    yield
    LIMITS.clear()
    LIMITS.update(before)


def _quota_burned() -> P.ProviderError:
    """The live failure, verbatim: a retryable 429 against one key's quota."""
    return P.ProviderError("gemini HTTP 429: You exceeded your current quota", status=429, retryable=True)


class _Key:
    """One pool member. Records that it was reached, then answers or fails."""

    capabilities = {"tools": True, "reasoning": True, "structured": True, "vision": True}

    def __init__(self, name: str, reached: list[str], error: Exception | None = None):
        self.name, self.reached, self.error = name, reached, error
        self.model = "gemini-3.1-flash-lite"

    async def chat(self, messages, **kwargs):
        self.reached.append(self.name)
        if self.error is not None:
            raise self.error
        return {
            "text": "ok",
            "model": self.model,
            "tool_calls": [],
            "stop_reason": "end_turn",
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "tool_call_dialect": "native",
            "reasoning_applied": False,
        }


def _gateway(error_by_key: dict[str, Exception] | None = None, members: int = 3):
    """A request object whose worker pool is `members` Gemini keys.

    Returns `(request, reached)`; `reached` is the ordered list of keys the ring
    actually contacted, which is the only thing these tests assert about.
    """
    db.init()
    errors = error_by_key or {}
    reached: list[str] = []
    pool = {
        f"gemini_{i}": _Key(f"gemini_{i}", reached, errors.get(f"gemini_{i}")) for i in range(1, members + 1)
    }
    for name in pool:
        LIMITS[name] = dict(LIMITS["gemini"])
    state = SimpleNamespace(
        router=Router(pool, ["gemini"]),
        router_pool=SimpleNamespace(providers={}, state={}, candidates=lambda: []),
        budget=None,
        telemetry=None,
        meter=None,
        semantic_cache=None,
    )
    return SimpleNamespace(app=SimpleNamespace(state=state)), reached


def _ask(request, provider: str):
    return chat(ChatRequest(prompt="hi", provider=provider, max_tokens=16), request)


# ── the headline: the pool is a pool ────────────────────────────────────────


async def test_a_burned_key_advances_to_the_next_key_in_the_pool():
    request, reached = _gateway({"gemini_1": _quota_burned()})
    out = await _ask(request, "gemini")
    assert reached == ["gemini_1", "gemini_2"], "the 429 must not have ended the call"
    assert out["provider"] == "gemini_2"
    assert out["text"] == "ok"


async def test_the_ring_walks_the_whole_pool_before_giving_up():
    request, reached = _gateway({"gemini_1": _quota_burned(), "gemini_2": _quota_burned()})
    out = await _ask(request, "gemini")
    assert reached == ["gemini_1", "gemini_2", "gemini_3"]
    assert out["provider"] == "gemini_3"


# ── the other half: naming a key still means that key ───────────────────────


async def test_naming_one_key_binds_to_that_key_and_does_not_fail_over():
    request, reached = _gateway({"gemini_1": _quota_burned()})
    with pytest.raises(HTTPException) as raised:
        await _ask(request, "gemini_1")
    assert raised.value.status_code == 502
    assert "gemini_1 failed" in raised.value.detail
    assert reached == ["gemini_1"], "a request for one key must not spend a second one"


async def test_a_doomed_call_stops_on_the_first_key():
    """A 400 is the request's fault, not the key's — every key would reject it."""
    bad_request = P.ProviderError("gemini HTTP 400: invalid argument", status=400, retryable=False)
    request, reached = _gateway(dict.fromkeys(("gemini_1", "gemini_2", "gemini_3"), bad_request))
    with pytest.raises(HTTPException) as raised:
        await _ask(request, "gemini")
    assert raised.value.status_code == 502
    assert reached == ["gemini_1"], "a non-retryable error must not burn the pool"
