"""The observability surface: one page, and the three reads it needs.

`/v1/status` and the economics routes already answer every question this session
asks — what is rate-limited, what was routed where, what it cost, what the budget
has left, what the cache saved. What was missing is somewhere to *look*, and a way
to get from a number back to the request that produced it.

    GET  /dashboard                          the page
    GET  /v1/traces/recent                   the traces this process emitted
    GET  /v1/traces/{trace_id}               one trace's spans, for a waterfall
    GET  /v1/refusals                        the 402s, which nothing else records
    GET  /v1/observability/agent_trace/{id}  an S15Code run's trace, proxied

A refused call is the one event in this gateway that leaves no trace of itself.
It writes no ledger row, because nothing was billed; it opens no span, because the
span starts after admission passes. That is correct behaviour and a blind spot: the
most important thing a budget controller does is the thing you cannot see
afterwards. So refusals are observed at the HTTP boundary instead, by an ASGI
middleware that keeps a bounded ring of 402 responses. Watching the status code
means the economics code needs to know nothing about this page, and any future
route that refuses on cost is recorded for free.

The trace list is served from the gateway's own bounded ring rather than queried
back out of Jaeger. Two reasons: the dashboard must still work when the collector
is down (a panel that says "no collector configured" is honest; a panel that
cannot render at all is not), and the gateway knows things the exporter drops —
which principal, which tier, whether the budget admitted the call. The trace *ids*
are the same ids Jaeger holds, so every row deep-links into the real waterfall.

The agent-trace proxy exists because a browser cannot reach the S15Code process
from a page served here without CORS, and widening CORS to satisfy a dashboard is
a poor trade. The gateway holds the credential-free HTTP hop instead. Set
`GLC_S15_BASE_URL`; leave it unset and the route says so.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from glc.telemetry import otel as _otel

router = APIRouter()

#: Status codes worth remembering at the boundary. 402 is the budget refusal.
WATCHED_STATUSES = (402,)
REFUSAL_RING = 100

#: Same directory `glc.main` mounts at /static — resolved here so this module
#: never has to import the app it is mounted into.
STATIC = Path(__file__).resolve().parents[2] / "static"
DASHBOARD = STATIC / "observability.html"


def _telemetry(request: Request) -> _otel.Telemetry:
    return getattr(request.app.state, "telemetry", None) or _otel.get_telemetry()


# ── refusals, observed at the boundary ──────────────────────────────────────


class RefusalLog:
    """A bounded, newest-first log of refused responses."""

    def __init__(self, limit: int = REFUSAL_RING):
        self.events: deque[dict] = deque(maxlen=limit)
        self.total = 0

    def record(self, *, status: int, method: str | None, path: str | None, body: bytes) -> dict:
        try:
            payload = json.loads(body.decode("utf-8"))
            detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
        except Exception:  # a non-JSON refusal is still a refusal
            detail = body.decode("utf-8", "replace")[:500] or None
        row = {"ts": time.time(), "status": status, "method": method, "path": path, "detail": detail}
        self.total += 1
        self.events.appendleft(row)
        return row

    def clear(self) -> None:
        self.events.clear()
        self.total = 0

    def describe(self, limit: int = 25) -> dict:
        return {
            "watching": list(WATCHED_STATUSES),
            "total": self.total,
            "kept": len(self.events),
            "refusals": list(self.events)[: max(1, int(limit))],
        }


#: One log per process. The middleware writes it; `/v1/refusals` reads it.
REFUSALS = RefusalLog()


class RefusalRecorder:
    """Pure ASGI middleware that remembers refused responses.

    Deliberately not a `BaseHTTPMiddleware`: this must never buffer or delay a
    streaming `/v1/chat` response. It watches the status line, and only for a
    watched status does it also collect the body — which is where the budget
    envelope's numbers live.
    """

    def __init__(self, app, statuses=WATCHED_STATUSES, log: RefusalLog | None = None):
        self.app = app
        self.statuses = tuple(statuses)
        self.log = log or REFUSALS

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        state: dict = {"watched": False, "status": None, "chunks": []}

        async def _send(message):
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
                state["watched"] = message["status"] in self.statuses
            elif state["watched"] and message["type"] == "http.response.body":
                body = message.get("body") or b""
                if body and sum(len(c) for c in state["chunks"]) < 16384:
                    state["chunks"].append(body)
                if not message.get("more_body"):
                    self.log.record(
                        status=state["status"],
                        method=scope.get("method"),
                        path=scope.get("path"),
                        body=b"".join(state["chunks"]),
                    )
            await send(message)

        await self.app(scope, receive, _send)


@router.get("/dashboard", response_class=FileResponse)
async def dashboard() -> FileResponse:
    """The v4 observability dashboard. No build step; it is one file."""
    return FileResponse(DASHBOARD, media_type="text/html")


@router.get("/v1/traces/recent")
async def recent_traces(request: Request, limit: int = 25):
    """The last N traces, newest first, each with a deep link into the trace UI.

    `trace_ui` is null when no UI is configured or derivable, which the dashboard
    renders as "no trace UI" instead of a dead link.
    """
    t = _telemetry(request)
    described = t.describe()
    return {
        "trace_ui": described["trace_ui"],
        "service_name": described["service_name"],
        "endpoint": described["endpoint"],
        "traces_endpoint": described["traces_endpoint"],
        "protocol": described["protocol"],
        "exporters": described["exporters"],
        "capture_content": described["capture_content"],
        "active": described["active"],
        "traces": t.recent_traces(limit=max(1, min(int(limit), 200))),
    }


@router.get("/v1/refusals")
async def refusals(limit: int = 25):
    """Every 402 this process has served, newest first, with the envelope intact.

    `total` keeps counting past the ring, so "how many refusals today" stays
    answerable after the detail has aged out.
    """
    return REFUSALS.describe(limit=limit)


@router.get("/v1/traces/{trace_id}")
async def trace_detail(trace_id: str, request: Request):
    """Every span of one trace: name, parent, start, duration, attributes."""
    detail = _telemetry(request).trace_detail(trace_id)
    if detail is None:
        raise HTTPException(
            404,
            "trace not in this process's ring (it may still be in the trace backend)",
        )
    return detail


@router.get("/v1/observability/agent_trace/{run_id}")
async def agent_trace(run_id: str, request: Request):
    """Proxy one S15Code run's OTel span tree, so the page can draw a waterfall."""
    # The target comes from operator configuration only. A caller-supplied
    # base_url would make this an unauthenticated SSRF with response
    # reflection: the gateway would fetch any URL and return its body.
    target = (os.getenv("GLC_S15_BASE_URL") or "").rstrip("/")
    if not target:
        raise HTTPException(
            503,
            "no agent runtime configured: set GLC_S15_BASE_URL to enable agent-run traces",
        )
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{target}/v1/agent/runs/{run_id}/trace")
    except Exception as e:
        raise HTTPException(502, f"agent runtime unreachable at {target}: {type(e).__name__}") from e
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, f"agent runtime said: {resp.text[:300]}")
    return {"source": target, **resp.json()}
