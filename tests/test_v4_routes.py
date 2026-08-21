"""The v4 HTTP surface, and the promise that nothing v3 exposed has moved.

`app_client` boots the real app, so these also prove the boot sequence loads
every config file and survives a missing collector / missing embedder.
"""

from __future__ import annotations

import pytest

from glc import db
from glc.economics import budget as B
from glc.economics import meter as M
from glc.economics import pricing as P


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def _fresh():
    P.reload_pricing()
    M.reset_meter()
    B.reset_controller()
    yield
    B.reset_controller()


# ── every v3 route still answers ────────────────────────────────────────────

V3_ROUTES = [
    "/v1/chat",
    "/v1/chat/batch",
    "/v1/vision",
    "/v1/embed",
    "/v1/cost/by_agent",
    "/v1/providers",
    "/v1/capabilities",
    "/v1/status",
    "/v1/routers",
    "/v1/calls",
    "/v1/embedders",
    "/v1/transcribe",
    "/v1/speak",
    "/v1/control/kill",
    "/v1/control/pair",
    "/v1/control/pair/confirm",
    "/v1/control/presence",
]

V4_ROUTES = [
    "/v1/budget",
    "/v1/budget/{principal}",
    "/v1/cost/by_principal",
    "/v1/cache/stats",
    "/v1/pricing",
    "/v1/telemetry",
]


def test_all_v3_routes_are_still_registered(app_client):
    paths = set(app_client.get("/openapi.json").json()["paths"])
    missing = [p for p in V3_ROUTES if p not in paths]
    assert not missing, f"v4 dropped v3 routes: {missing}"


def test_all_v4_routes_are_registered(app_client):
    paths = set(app_client.get("/openapi.json").json()["paths"])
    missing = [p for p in V4_ROUTES if p not in paths]
    assert not missing, f"missing v4 routes: {missing}"


def test_v3_response_shapes_are_unchanged(app_client):
    for k in ("order", "providers", "shortcuts", "limits", "models"):
        assert k in app_client.get("/v1/providers").json()
    for k in ("order", "live", "today", "limits"):
        assert k in app_client.get("/v1/status").json()
    body = app_client.get("/v1/routers").json()
    for k in ("order", "providers", "models", "live", "today", "limits", "tier_to_order"):
        assert k in body
    # v4 adds a key without removing one
    assert "routing_policy" in body
    assert set(body["tier_to_order"]) >= {"TINY", "LARGE"}
    assert isinstance(app_client.get("/v1/cost/by_agent").json(), dict)
    assert isinstance(app_client.get("/v1/calls").json(), list)


def test_a_v3_chat_body_is_still_accepted():
    """A glc_v3 client sends no tenant/user/cache fields at all.

    Asserted against the model rather than through HTTP: this is a schema
    claim, and routing it through /v1/chat would make a live provider call to
    prove something Pydantic already knows.
    """
    from glc.llm_schemas import ChatRequest

    v3_body = {
        "messages": [{"role": "user", "content": "hi"}],
        "system": "be brief",
        "provider": "g",
        "model": "gemini-3.1-flash-lite",
        "max_tokens": 64,
        "temperature": 0.5,
        "stream": False,
        "tools": [{"name": "f", "description": "d", "input_schema": {}}],
        "tool_choice": "auto",
        "cache_system": True,
        "reasoning": "low",
        "response_format": {"type": "json_schema", "schema": {"type": "object"}},
        "auto_route": "decision",
        "agent": "a",
        "session": "s",
    }
    req = ChatRequest(**v3_body)
    assert req.agent == "a" and req.session == "s"
    # v4 fields default to inert values
    assert (req.tenant, req.project, req.user) == (None, None, None)
    assert req.semantic_cache is None
    assert req.batch is False
    assert req.cost_quality_tradeoff is None
    assert req.escalate is None


def test_the_v3_auto_route_values_still_validate():
    from glc.llm_schemas import ChatRequest

    for role in ("perception", "memory", "decision"):
        assert ChatRequest(prompt="hi", auto_route=role).auto_route == role


def test_a_role_added_only_in_yaml_is_accepted_by_the_schema():
    """The Literal was widened so adding a role is a routing.yaml edit."""
    from glc.llm_schemas import ChatRequest

    assert ChatRequest(prompt="hi", auto_route="triage").auto_route == "triage"


def test_a_v3_client_ignores_the_new_response_fields():
    from glc.llm_schemas import ChatResponse

    r = ChatResponse(provider="gemini_1", model="m", text="hi").model_dump()
    for k in ("cost", "budget", "cache", "trace"):
        assert r[k] is None
    # every v3 key is still present
    for k in (
        "provider",
        "model",
        "text",
        "tool_calls",
        "stop_reason",
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "latency_ms",
        "tool_call_dialect",
        "reasoning_applied",
        "parsed",
        "attempted",
        "router_decision",
        "retries",
    ):
        assert k in r


# ── budgets ─────────────────────────────────────────────────────────────────


def test_budget_defaults_to_ungoverned(app_client):
    body = app_client.get("/v1/budget/agent:nobody").json()
    assert body["governed"] is False
    assert body["limit_usd"] is None
    assert "unlimited" in body["message"]


def test_post_then_get_a_budget(app_client, install_token):
    r = app_client.post(
        "/v1/budget",
        headers=_auth(install_token),
        json={"principal": "session:run-9", "limit_usd": 0.25, "period": "lifetime"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["policy"]["limit_usd"] == 0.25

    body = app_client.get("/v1/budget/session:run-9").json()
    assert body["governed"] is True
    assert body["limit_usd"] == 0.25
    assert body["remaining_usd"] == 0.25
    assert body["policy"]["source"] == "runtime"


def test_budget_reflects_real_ledger_spend(app_client, install_token):
    app_client.post(
        "/v1/budget",
        headers=_auth(install_token),
        json={"principal": "user:u9", "limit_usd": 1.0, "period": "lifetime"},
    )
    M.get_meter().record(
        provider="gemini_1",
        model="gemini-3.1-flash-lite",
        principal=M.Principal(user="u9"),
        usage=M.Usage(input_tokens=1_000_000),  # $0.25
    )
    body = app_client.get("/v1/budget/user:u9").json()
    assert body["spent_usd"] == pytest.approx(0.25)
    assert body["remaining_usd"] == pytest.approx(0.75)
    assert body["fraction_used"] == pytest.approx(0.25)


def test_delete_a_runtime_budget(app_client, install_token):
    app_client.post(
        "/v1/budget",
        headers=_auth(install_token),
        json={"principal": "agent:tmp", "limit_usd": 0.1, "period": "day"},
    )
    assert app_client.get("/v1/budget/agent:tmp").json()["governed"] is True
    assert app_client.request(
        "DELETE", "/v1/budget/agent:tmp", headers=_auth(install_token)
    ).json()["removed"] is True
    assert app_client.get("/v1/budget/agent:tmp").json()["governed"] is False


def test_a_bad_principal_is_a_400(app_client, install_token):
    assert app_client.get("/v1/budget/favourite_colour:blue").status_code == 400
    assert (
        app_client.post(
            "/v1/budget", headers=_auth(install_token), json={"principal": "nope:x", "limit_usd": 1}
        ).status_code
        == 400
    )


def test_a_negative_limit_is_rejected_by_the_schema(app_client, install_token):
    assert (
        app_client.post(
            "/v1/budget", headers=_auth(install_token), json={"principal": "agent:a", "limit_usd": -1}
        ).status_code
        == 422
    )


def test_list_budgets_describes_the_loaded_config(app_client):
    body = app_client.get("/v1/budget").json()
    assert "policies" in body and "projection" in body and "defaults" in body
    assert body["enabled"] is True


# ── the headline: a refused call ────────────────────────────────────────────


def test_a_tiny_budget_refuses_the_call_with_402(app_client, install_token):
    """The kill-switch, end to end through HTTP.

    A $0 ceiling on the session means the projection cannot fit, so the gateway
    answers 402 and no provider is contacted.
    """
    app_client.post(
        "/v1/budget",
        headers=_auth(install_token),
        json={"principal": "session:broke", "limit_usd": 0.0, "period": "lifetime"},
    )
    before = len(db.recent(limit=1000))
    r = app_client.post(
        "/v1/chat",
        json={
            "prompt": "this must never reach a provider",
            "provider": "gemini",
            "model": "gemini-3.1-flash-lite",
            "max_tokens": 256,
            "session": "broke",
        },
    )
    if r.status_code == 400:
        pytest.skip("no gemini provider configured in this environment")
    assert r.status_code == 402, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "budget_exceeded"
    assert detail["principal"] == "session:broke"
    assert detail["limit_usd"] == 0.0
    assert detail["projected_usd"] > 0
    assert detail["shortfall_usd"] > 0
    # nothing was billed: no worker row was appended
    after = [r for r in db.recent(limit=1000) if r["call_role"] == "worker"]
    assert len(after) == 0
    assert before == len(db.recent(limit=1000))


def test_a_generous_budget_admits_the_same_request(app_client, install_token):
    """Same request, bigger ceiling: admission passes.

    Checked at the controller rather than over HTTP so the assertion is about
    the gate and not about whether a provider happened to be reachable.
    """
    from glc.economics import budget as _b

    app_client.post(
        "/v1/budget",
        headers=_auth(install_token),
        json={"principal": "session:rich", "limit_usd": 1000.0, "period": "lifetime"},
    )
    ctl = _b.get_controller()
    projected = ctl.project("gemini_1", "gemini-3.1-flash-lite", 5000, 256)
    assert projected > 0
    assert ctl.admit(M.Principal(session="rich"), projected).allowed is True

    app_client.post(
        "/v1/budget",
        headers=_auth(install_token),
        json={"principal": "session:rich", "limit_usd": 0.0, "period": "lifetime"},
    )
    _b.reload_controller()
    assert _b.get_controller().admit(M.Principal(session="rich"), projected).allowed is False


# ── cost attribution ────────────────────────────────────────────────────────


def test_cost_by_principal_covers_all_five_dimensions(app_client):
    M.get_meter().record(
        provider="groq",
        model="openai/gpt-oss-120b",
        principal=M.Principal(tenant="acme", project="p1", user="u1", agent="a1", session="s1"),
        usage=M.Usage(input_tokens=1_000_000, output_tokens=1_000_000),
    )
    body = app_client.get("/v1/cost/by_principal?period=lifetime").json()
    assert body["dimensions"] == ["tenant", "project", "user", "agent", "session"]
    for dim in body["dimensions"]:
        assert body["totals_usd"][dim] == pytest.approx(0.90)
    assert body["by_dimension"]["tenant"][0]["principal"] == "tenant:acme"


def test_cost_by_principal_single_dimension(app_client):
    M.get_meter().record(
        provider="groq",
        model="openai/gpt-oss-120b",
        principal=M.Principal(user="solo"),
        usage=M.Usage(input_tokens=1_000_000),
    )
    body = app_client.get("/v1/cost/by_principal?dimension=user&period=lifetime").json()
    assert body["dimension"] == "user"
    assert body["rows"][0]["principal"] == "user:solo"
    assert body["total_usd"] == pytest.approx(0.15)


def test_cost_by_principal_rejects_an_unknown_dimension(app_client):
    assert app_client.get("/v1/cost/by_principal?dimension=vibes").status_code == 400


def test_by_principal_is_a_superset_of_by_agent(app_client):
    M.get_meter().record(
        provider="groq",
        model="openai/gpt-oss-120b",
        principal=M.Principal(agent="shared"),
        usage=M.Usage(input_tokens=1_000_000, output_tokens=500_000),
    )
    legacy = app_client.get("/v1/cost/by_agent").json()
    modern = app_client.get("/v1/cost/by_principal?dimension=agent").json()
    assert legacy["shared"][0]["in_tok"] == 1_000_000
    row = modern["rows"][0]
    assert row["in_tok"] == 1_000_000
    # v3 priced per provider ($0.15/$0.75 for groq) and still does; v4 adds
    # cache columns and savings that by_agent never had.
    assert legacy["shared"][0]["dollars"] == pytest.approx(row["dollars"])
    assert {"cache_hits", "tokens_saved", "dollars_saved"} <= set(row)


# ── cache + introspection ───────────────────────────────────────────────────


def test_cache_stats_shape(app_client):
    body = app_client.get("/v1/cache/stats").json()
    assert body["enabled"] is True
    for k in ("lookups", "hits", "misses", "hit_rate", "tokens_saved", "usd_saved", "entries"):
        assert k in body
    assert body["config"]["threshold"] >= 0.92
    # prompt caching is reported separately, because it is a different mechanism
    assert "gemini_prompt_cache" in body


def test_cache_purge(app_client, install_token):
    assert app_client.post("/v1/cache/purge", headers=_auth(install_token)).json()["ok"] is True


def test_pricing_endpoint_resolves_one_model(app_client):
    body = app_client.get("/v1/pricing?provider=gemini_1&model=gemini-3.1-flash-lite").json()
    assert body["input_usd_per_mtok"] == 0.25
    assert body["price_source"] == "model"
    assert body["priced"] is True


def test_pricing_endpoint_lists_the_table_and_live_providers(app_client):
    body = app_client.get("/v1/pricing").json()
    assert body["model_count"] > 10
    assert "configured_providers" in body


def test_telemetry_endpoint_reports_content_capture_off(app_client):
    body = app_client.get("/v1/telemetry").json()
    assert body["capture_content"] is False
    assert "exporters" in body


def test_boot_reported_no_config_errors(app_client):
    """Every shipped YAML must parse — if one does not, boot records it."""
    import glc.main as m

    assert m.app.state.config_errors == {}, m.app.state.config_errors
