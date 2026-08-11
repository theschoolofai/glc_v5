"""Candidate D: BudgetController.admit() reads spend and writes nothing: the
actual spend is only written by Meter.record(), after the (slow) provider
call returns. Two concurrent requests from the same principal can both pass
admit() while the first is still in that gap.

`test_a_third_request_is_correctly_refused_once_two_concurrent_holds_are_committed`
is the headline proof: it dispatches two concurrent calls through the real
`/v1/chat` handler (`glc.routes.chat.chat`) against a stubbed slow provider,
then — while the second call is still in flight, before it has recorded
anything — fires the two probe admissions from the design doc's worked
example (limit $2.00/day; committed should read $1.60, so a $0.50 request
must be refused and a $0.30 request must be admitted). On unfixed `admit()`
the in-flight call is invisible to the ledger, so both probes are wrongly
decided against a committed total of only $0.60.

`test_a_released_holds_dollar_figure_survives_but_stops_counting` pins the
same arithmetic without the concurrency machinery, as a direct unit check on
`BudgetController` + `Meter`.
"""

from __future__ import annotations

import asyncio
import types

import pytest
from fastapi import HTTPException

from glc import db
from glc.economics import budget as B
from glc.economics import meter as M
from glc.economics import pricing as P
from glc.llm_schemas import ChatRequest
from glc.routes.chat import chat as chat_route
from glc.routing.core import Router


@pytest.fixture(autouse=True)
def _fresh():
    db.init()
    B.init_store()
    P.reload_pricing()
    M.reset_meter()
    B.reset_controller()
    yield
    B.reset_controller()


def _ctl(tmp_path, monkeypatch, yaml_text: str) -> B.BudgetController:
    p = tmp_path / "budgets.yaml"
    p.write_text(yaml_text)
    monkeypatch.setenv("GLC_BUDGETS_YAML", str(p))
    return B.reload_controller()


def _fake_result(output_tokens: int, input_tokens: int = 10) -> dict:
    return {
        "text": "ok",
        "model": "fake-model",
        "tool_calls": [],
        "stop_reason": "end_turn",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "tool_call_dialect": "none",
        "reasoning_applied": False,
    }


class _TaggedProvider:
    """A stub worker provider whose behaviour depends on which call tagged
    it. T1 resolves quickly. T2 blocks until the test releases it, so its
    admission-to-metering gap is real and controllable rather than a timing
    guess. Anything else (the probe requests) resolves immediately."""

    model = "fake-model"
    capabilities: dict = {}

    def __init__(self, t2_started: asyncio.Event, t2_may_finish: asyncio.Event):
        self._t2_started = t2_started
        self._t2_may_finish = t2_may_finish

    async def chat(self, messages, **kw):
        tag = messages[0]["content"]
        if tag == "T1":
            await asyncio.sleep(0.01)
            return _fake_result(output_tokens=600)
        if tag == "T2":
            self._t2_started.set()
            await self._t2_may_finish.wait()
            return _fake_result(output_tokens=1000)
        return _fake_result(output_tokens=300)


def _fake_request(ctl: B.BudgetController, meter: M.Meter, provider) -> types.SimpleNamespace:
    """The pieces of `request.app.state` that `glc.routes.chat.chat` reads."""
    rtr = Router({"ollama": provider}, order=["ollama"])
    state = types.SimpleNamespace(
        router=rtr,
        router_pool=None,
        budget=ctl,
        meter=meter,
        semantic_cache=None,
    )
    return types.SimpleNamespace(app=types.SimpleNamespace(state=state))


async def test_a_third_request_is_correctly_refused_once_two_concurrent_holds_are_committed(
    tmp_path, monkeypatch
):
    """The worked example from the design doc (section 3.6), reached through
    real concurrent dispatch instead of being constructed by hand.

    T1 projects $0.70 and settles for $0.60 (worst-case padding resolved
    down). T2 projects $1.00 and is held open — its provider call is still
    in flight — when the two probes fire. Committed must read
    0.60 + 1.00 = 1.60, so a $0.50 probe must be refused and a $0.30 probe
    must be admitted. Unfixed `admit()` never wrote anything for either call
    until their provider call returned, so a probe landing in T2's gap saw
    only T1's settled $0.60 and wrongly decided both probes.
    """
    ctl = _ctl(
        tmp_path,
        monkeypatch,
        "version: 1\npolicies:\n  - principal: 'session:*'\n    limit_usd: 2.00\n    period: day\n",
    )
    meter = M.get_meter()
    # Decouple the dollar figures from the real pricing table (which has no
    # entry for the "ollama"/"fake-model" stub) so the projected and actual
    # costs land exactly on the doc's worked-example numbers: $ = tokens/1000.
    monkeypatch.setattr(
        ctl, "project", lambda provider, model, est_input_tokens, max_output_tokens, batch=False: (
            max_output_tokens / 1000.0
        )
    )
    monkeypatch.setattr(
        meter, "price", lambda provider, model, usage, batch=False: P.CostBreakdown(
            total_usd=usage.output_tokens / 1000.0, price_source="test"
        )
    )

    t2_started = asyncio.Event()
    t2_may_finish = asyncio.Event()
    t1_done = asyncio.Event()
    provider = _TaggedProvider(t2_started, t2_may_finish)

    async def dispatch(tag: str, max_tokens: int):
        req = ChatRequest(prompt=tag, provider="ollama", max_tokens=max_tokens, session="racer")
        return await chat_route(req, _fake_request(ctl, meter, provider))

    async def do_t1():
        await dispatch("T1", max_tokens=700)  # projects $0.70, settles at $0.60
        t1_done.set()

    async def do_t2():
        await dispatch("T2", max_tokens=1000)  # projects $1.00, held open until released below

    async def do_probes():
        await t1_done.wait()
        await t2_started.wait()  # T2 has been admitted and is now in its slow gap

        refused = False
        try:
            await dispatch("probe-big", max_tokens=500)  # $0.50: 1.60 + 0.50 > 2.00
        except HTTPException as e:
            refused = e.status_code == 402

        admitted = False
        try:
            await dispatch("probe-small", max_tokens=300)  # $0.30: 1.60 + 0.30 <= 2.00
            admitted = True
        except HTTPException:
            admitted = False

        t2_may_finish.set()
        return refused, admitted

    _, _, (refused, admitted) = await asyncio.gather(do_t1(), do_t2(), do_probes())

    assert refused is True, "a $0.50 request must be refused once $1.60 is committed"
    assert admitted is True, "a $0.30 request must still be admitted with $0.40 remaining"


def test_a_released_holds_dollar_figure_survives_but_stops_counting(tmp_path, monkeypatch):
    """Same arithmetic as the concurrency test, without the machinery: a
    direct unit check that a released hold does not double-count and an
    open hold counts as committed."""
    ctl = _ctl(
        tmp_path,
        monkeypatch,
        "version: 1\npolicies:\n  - principal: 'session:x'\n    limit_usd: 2.00\n    period: day\n",
    )
    meter = M.get_meter()
    p = M.Principal(session="x")

    hold = ctl.admit(p, 0.70)
    assert hold.allowed is True
    meter.record(
        provider="gemini_1",
        model="gemini-3.1-flash-lite",  # $0.25/Mtok in
        principal=p,
        usage=M.Usage(input_tokens=int(0.60 / 0.25 * 1_000_000)),
        status="ok",
        reservation_id=hold.reservation_id,
    )

    still_open = ctl.admit(p, 1.00)
    assert still_open.allowed is True

    status = ctl.status_for("session", "x")
    assert status.spent_usd == pytest.approx(1.60)
    assert status.remaining_usd == pytest.approx(0.40)

    assert ctl.admit(p, 0.50).allowed is False
    assert ctl.admit(p, 0.30).allowed is True
