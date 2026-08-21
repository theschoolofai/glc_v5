"""The observability surface: the dashboard route, the trace ring, refusals.

Three things are worth pinning down, and none of them needs a collector running:

1. The page is served, and it is *generic* — no model, provider, principal or task
   name is baked into the HTML. The v4 dashboard's whole claim is that it renders
   whatever the API returns, and a hardcoded model name would quietly break that.
2. The trace ring holds real spans with their real ids, so a row on the page can
   deep-link into a backend, and it degrades to an honest "tracing off" instead of
   an empty table when nothing is configured.
3. A 402 is recorded. Refusals are the one thing the gateway does not otherwise
   remember, so the recorder is the only witness a reviewer has.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from glc.routes import observability as obs
from glc.telemetry import otel as _otel

# ── the page ────────────────────────────────────────────────────────────────


def test_dashboard_is_served_and_self_contained(app_client):
    r = app_client.get("/dashboard")
    assert r.status_code == 200
    html = r.text
    # It reads the live API rather than shipping a snapshot.
    for endpoint in (
        "/v1/status",
        "/v1/providers",
        "/v1/capabilities",
        "/v1/calls",
        "/v1/routers",
        "/v1/cost/by_principal",
        "/v1/budget",
        "/v1/cache/stats",
        "/v1/traces/recent",
        "/v1/refusals",
    ):
        assert endpoint in html, endpoint
    # One file, no build step: the only external reference may be the webfont.
    external = [
        line
        for line in html.splitlines()
        if ("http://" in line or "https://" in line) and "fonts.googleapis.com" not in line
    ]
    assert not external, external
    assert "<script src" not in html and 'stylesheet" href="/' not in html


def test_dashboard_hardcodes_no_domain_values(app_client):
    """A panel must not know any provider, model, principal or task by name."""
    html = app_client.get("/dashboard").text.lower()
    for forbidden in (
        "gemini",
        "cerebras",
        "openrouter",
        "nvidia",
        "groq",
        "ollama",
        "github",
        "flash-lite",
        "gpt-",
        "llama",
        "claude-",
        "inkers",
        "acme",
    ):
        assert forbidden not in html, f"{forbidden!r} is baked into the dashboard"


def test_v3_dashboard_still_at_root(app_client):
    """Backward compatibility: the inherited page keeps its route."""
    assert app_client.get("/").status_code == 200
    assert app_client.get("/dashboard").text != app_client.get("/").text


# ── the trace ring ──────────────────────────────────────────────────────────


def test_recent_traces_is_honest_when_tracing_is_off(app_client):
    body = app_client.get("/v1/traces/recent").json()
    assert body["active"] is False
    assert body["traces"] == []
    assert body["trace_ui"] is None
    assert body["capture_content"] is False


def test_recent_traces_reports_real_spans(monkeypatch, app_client):
    """A span emitted in-process shows up with its id, cost and usage."""
    monkeypatch.setenv("GLC_OTEL_IN_MEMORY", "1")
    monkeypatch.setenv("GLC_TRACE_UI", "http://trace-ui.example:16686")
    _otel.reset_telemetry()
    t = _otel.init_telemetry(force=True, in_memory=True)
    assert t.active and t.recent is not None
    app_client.app.state.telemetry = t

    with t.span(_otel.SPAN_CHAT) as span:
        span.set_request(provider="someprovider_2", model="some-model-x", max_tokens=10)
        span.set_usage(input_tokens=11, output_tokens=7, response_model="some-model-x")
        span.set_cost({"total_usd": 0.25, "price_source": "model"})
        span.set(_otel.GLC_ROUTING_ROLE, "somerole")
        span.set(_otel.GLC_ROUTING_TIER, "SOMETIER")
    t.flush()

    body = app_client.get("/v1/traces/recent").json()
    assert body["active"] is True
    assert body["trace_ui"] == "http://trace-ui.example:16686"
    assert len(body["traces"]) == 1
    row = body["traces"][0]
    assert row["model"] == "some-model-x"
    assert row["provider"] == "someprovider"  # vendor, not the key slot
    assert row["provider_instance"] == "someprovider_2"
    assert (row["input_tokens"], row["output_tokens"]) == (11, 7)
    assert row["cost_usd"] == pytest.approx(0.25)
    assert row["role"] == "somerole" and row["tier"] == "SOMETIER"
    assert row["url"] == f"http://trace-ui.example:16686/trace/{row['trace_id']}"

    detail = app_client.get(f"/v1/traces/{row['trace_id']}").json()
    assert detail["trace_id"] == row["trace_id"]
    assert detail["spans"] and detail["spans"][0]["attributes"][_otel.GLC_COST_USD] == 0.25
    # Content capture is off, so no prompt or completion text is anywhere.
    dumped = json.dumps(detail)
    assert _otel.GEN_AI_INPUT_MESSAGES not in dumped
    assert _otel.GEN_AI_OUTPUT_MESSAGES not in dumped


def test_unknown_trace_is_a_404_not_an_empty_waterfall(app_client):
    r = app_client.get("/v1/traces/00000000000000000000000000000000")
    assert r.status_code == 404
    assert "ring" in r.json()["detail"]


def test_trace_ring_is_bounded(monkeypatch):
    """The ring must not be an unbounded leak in a long-lived gateway."""
    ring = _otel._RecentTraceExporter(limit=3)

    class _Ctx:
        def __init__(self, t):
            self.trace_id, self.span_id = t, 1

    class _Span:
        def __init__(self, t):
            self.name, self.context, self.parent = "chat", _Ctx(t), None
            self.start_time, self.end_time, self.attributes, self.status = 1, 2, {}, None

    ring.export([_Span(i) for i in range(1, 11)])
    assert len(ring.traces) == 3
    assert len(ring.summaries()) == 3


# ── telemetry configuration ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "endpoint,protocol,expected_path,expected_label",
    [
        ("http://host:4318", None, "http://host:4318/v1/traces", "http"),
        ("http://host:4318/", None, "http://host:4318/v1/traces", "http"),
        # Already spelled out: appended once, not twice.
        ("http://host:4318/v1/traces", None, "http://host:4318/v1/traces", "http"),
        ("host:4317", None, "host:4317", "grpc"),
        ("http://host:4317", "grpc", "http://host:4317", "grpc"),
    ],
)
def test_otlp_endpoint_and_protocol(monkeypatch, endpoint, protocol, expected_path, expected_label):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", endpoint)
    if protocol:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_PROTOCOL", protocol)
    else:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_PROTOCOL", raising=False)
    _otel.reset_telemetry()
    t = _otel.Telemetry()
    t.endpoint = endpoint
    assert t.protocol_label == expected_label
    assert t.traces_endpoint == expected_path


def test_trace_ui_is_derived_from_the_collector_host(monkeypatch):
    monkeypatch.delenv("GLC_TRACE_UI", raising=False)
    monkeypatch.delenv("GLC_JAEGER_UI", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.internal:4318")
    assert _otel.trace_ui_base() == "http://collector.internal:16686"
    monkeypatch.setenv("GLC_TRACE_UI", "https://traces.example.com/")
    assert _otel.trace_ui_base() == "https://traces.example.com"
    assert _otel.trace_url("abc") == "https://traces.example.com/trace/abc"


def test_no_trace_ui_means_no_link(monkeypatch):
    for var in (
        "GLC_TRACE_UI",
        "GLC_JAEGER_UI",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    ):
        monkeypatch.delenv(var, raising=False)
    assert _otel.trace_ui_base() is None
    assert _otel.trace_url("abc") is None


# ── refusals ────────────────────────────────────────────────────────────────


def test_refusals_route_starts_empty(app_client):
    obs.REFUSALS.clear()
    body = app_client.get("/v1/refusals").json()
    assert body == {"watching": [402], "total": 0, "kept": 0, "refusals": []}


def test_the_middleware_records_a_402_with_its_envelope():
    """The gateway's 402 body is structured; the recorder must keep it intact."""
    log = obs.RefusalLog(limit=5)
    app = FastAPI()

    @app.post("/v1/chat")
    async def chat():
        raise HTTPException(
            402,
            {
                "code": "budget_exceeded",
                "principal": "session:x",
                "limit_usd": 0.001,
                "spent_usd": 0.001,
                "projected_usd": 0.02,
            },
        )

    @app.get("/ok")
    async def ok():
        return {"fine": True}

    app.add_middleware(obs.RefusalRecorder, log=log)
    with TestClient(app) as c:
        assert c.get("/ok").status_code == 200
        assert c.post("/v1/chat").status_code == 402
        assert c.post("/v1/chat").status_code == 402

    assert log.total == 2, "a 200 must not be recorded, and both 402s must be"
    newest = log.events[0]
    assert newest["status"] == 402 and newest["path"] == "/v1/chat" and newest["method"] == "POST"
    assert newest["detail"]["principal"] == "session:x"
    assert newest["detail"]["projected_usd"] == 0.02
    described = log.describe(limit=1)
    assert described["total"] == 2 and len(described["refusals"]) == 1


def test_refusal_log_is_bounded_but_keeps_counting():
    log = obs.RefusalLog(limit=2)
    for i in range(5):
        log.record(status=402, method="POST", path="/v1/chat", body=f'{{"i":{i}}}'.encode())
    assert log.total == 5 and len(log.events) == 2
    assert log.events[0]["detail"]["i"] == 4


def test_a_non_json_refusal_is_still_recorded():
    log = obs.RefusalLog()
    log.record(status=402, method="POST", path="/v1/chat", body=b"not json at all")
    assert log.events[0]["detail"] == "not json at all"


# ── the agent-run proxy ─────────────────────────────────────────────────────


def test_agent_trace_says_so_when_no_runtime_is_configured(monkeypatch, app_client):
    monkeypatch.delenv("GLC_S15_BASE_URL", raising=False)
    r = app_client.get("/v1/observability/agent_trace/run-1")
    assert r.status_code == 503
    assert "GLC_S15_BASE_URL" in r.json()["detail"]


def test_agent_trace_reports_an_unreachable_runtime(monkeypatch, app_client):
    # The target is operator configuration; a caller-supplied base_url would
    # make this route an SSRF with response reflection.
    monkeypatch.setenv("GLC_S15_BASE_URL", "http://127.0.0.1:1")  # nothing listens on port 1
    r = app_client.get("/v1/observability/agent_trace/run-1")
    assert r.status_code == 502
    assert "unreachable" in r.json()["detail"]
