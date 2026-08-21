"""The agent-trace proxy must not accept a caller-supplied target URL."""
from __future__ import annotations

from fastapi.testclient import TestClient

import glc.main as m


def test_query_string_base_url_is_ignored(monkeypatch):
    """?base_url= used to make the gateway fetch and reflect any URL (SSRF)."""
    monkeypatch.delenv("GLC_S15_BASE_URL", raising=False)
    with TestClient(m.app) as client:
        resp = client.get(
            "/v1/observability/agent_trace/run1",
            params={"base_url": "http://169.254.169.254/latest/meta-data"},
        )
    assert resp.status_code == 503
    assert "no agent runtime configured" in resp.json()["detail"]


def test_env_configured_target_is_used(monkeypatch):
    calls: list[str] = []

    class _Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"spans": []}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def aclose(self):
            pass

        async def get(self, url):
            calls.append(url)
            return _Resp()

    import httpx

    monkeypatch.setenv("GLC_S15_BASE_URL", "http://127.0.0.1:8113")
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    with TestClient(m.app) as client:
        resp = client.get("/v1/observability/agent_trace/run1")
    assert resp.status_code == 200
    assert calls == ["http://127.0.0.1:8113/v1/agent/runs/run1/trace"]