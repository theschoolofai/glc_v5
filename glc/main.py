"""FastAPI app for glc_v5. Port 8111 by default. V9 routes are mounted
as-is (S9 Browser / S10 Computer-Use clients work unchanged); the
S11 surfaces (transcribe, speak, channels WS, control) sit alongside, and
v4 adds the economics surface (budgets, principal cost, cache stats).

v4 boot additions, all fail-soft: a broken pricing.yaml, an unreachable OTLP
collector or a missing embedder must never stop the gateway from serving
`/v1/chat`. Each is caught, reported on the console, and the corresponding
feature stays off.
"""

from __future__ import annotations

import os
import signal
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent
# Source checkout keeps the pages at the repo root; an installed wheel keeps
# them beside the package. Try both, so "the API works but every page is
# missing" cannot happen silently because of how somebody installed it.
STATIC = next((c for c in (ROOT.parent / "static", ROOT / "_static") if c.is_dir()),
              ROOT.parent / "static")
load_dotenv(ROOT.parent / ".env")  # repo .env, if present

from glc import db  # noqa: E402
from glc import embedders as E  # noqa: E402
from glc import providers as P  # noqa: E402
from glc.audit import init_store as init_audit  # noqa: E402
from glc.cache import GeminiCache  # noqa: E402
from glc.cache.semantic import build_semantic_cache  # noqa: E402
from glc.channels.agent_bridge import S16AgentBridge  # noqa: E402
from glc.channels.setup import apply_saved_environment  # noqa: E402
from glc.config import get_or_create_install_token  # noqa: E402
from glc.economics import budget as budget_mod  # noqa: E402
from glc.economics import meter as meter_mod  # noqa: E402
from glc.economics import pricing as pricing_mod  # noqa: E402
from glc.policy import reload_engine  # noqa: E402
from glc.routes import channels as channels_route  # noqa: E402
from glc.routes import chat as chat_route  # noqa: E402
from glc.routes import control as control_route  # noqa: E402
from glc.routes import economics as economics_route  # noqa: E402
from glc.routes import observability as observability_route  # noqa: E402
from glc.routes import speak as speak_route  # noqa: E402
from glc.routes import transcribe as transcribe_route  # noqa: E402
from glc.routing import Router, RouterPool  # noqa: E402
from glc.routing import policy as routing_policy  # noqa: E402
from glc.telemetry import otel as otel_mod  # noqa: E402

PORT = int(os.getenv("GLC_PORT", "8111"))


def _install_sighup_reload() -> None:
    """Hot-reload policy.yaml on SIGHUP. Windows lacks SIGHUP so this is
    a no-op there."""
    if not hasattr(signal, "SIGHUP"):
        return

    def _handler(signum, frame):  # noqa: ARG001
        try:
            reload_engine()
            print("[glc] policy.yaml reloaded via SIGHUP")
        except Exception as e:
            print(f"[glc] SIGHUP reload failed: {e!r}")

    try:
        signal.signal(signal.SIGHUP, _handler)
    except ValueError:
        # signal() only works on the main thread; tests using TestClient
        # spawn lifespan from a worker thread. Silent skip is correct here.
        pass


def _boot_economics(app: FastAPI) -> None:
    """Load the v4 config-driven layers. Each failure is isolated: a bad
    pricing.yaml costs you dollar reports, not the gateway."""
    app.state.config_errors = {}

    # Reload rather than load: the singletons may be holding a table read
    # against a previous GLC_CONFIG_DIR (tests, or a SIGHUP).
    try:
        app.state.pricing = pricing_mod.reload_pricing()
        meter_mod.reset_meter()
        app.state.meter = meter_mod.get_meter()
    except Exception as e:
        app.state.config_errors["pricing.yaml"] = repr(e)
        print(f"[glc] pricing.yaml unusable — dollars will report as 0: {e!r}")
        app.state.meter = meter_mod.get_meter()

    try:
        budget_mod.init_store()
        app.state.budget = budget_mod.reload_controller()
        armed = [p.principal for p in app.state.budget.policies()]
        if armed:
            print(f"[glc] budget controller armed for {armed}")
    except Exception as e:
        # A malformed budgets.yaml must NOT silently mean "no budget". The
        # gateway boots, but the error is recorded and surfaced on
        # /v1/budget so nobody mistakes a broken ceiling for a generous one.
        app.state.config_errors["budgets.yaml"] = repr(e)
        app.state.budget = None
        print(f"[glc] budgets.yaml unusable — budget enforcement is OFF: {e!r}")

    try:
        app.state.routing_policy = routing_policy.reload_policy()
    except Exception as e:
        app.state.config_errors["routing.yaml"] = repr(e)
        app.state.routing_policy = None
        print(f"[glc] routing.yaml unusable — falling back to v3 TIER_TO_ORDER: {e!r}")

    try:
        otel_mod.reset_telemetry()
        app.state.telemetry = otel_mod.init_telemetry()
        print(f"[glc] telemetry: {app.state.telemetry.describe()}")
    except Exception as e:
        app.state.config_errors["telemetry"] = repr(e)
        app.state.telemetry = otel_mod.Telemetry()
        print(f"[glc] telemetry init failed — spans are a no-op: {e!r}")

    try:
        app.state.semantic_cache = build_semantic_cache(app.state.embedders)
    except Exception as e:
        app.state.config_errors["cache.yaml"] = repr(e)
        app.state.semantic_cache = None
        print(f"[glc] semantic cache unavailable: {e!r}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Channel settings are local to this gateway installation.  Existing
    # adapter environment variables remain the runtime contract, so load saved
    # values before adapters are ever instantiated (without serving them back).
    apply_saved_environment()
    db.init()
    init_audit()
    get_or_create_install_token()
    _install_sighup_reload()
    app.state.cache = GeminiCache(ttl_seconds=300)
    app.state.providers = P.build_providers(app.state.cache)
    app.state.router = Router(app.state.providers, chat_route.ORDER)
    app.state.router_providers = P.build_router_providers()
    app.state.router_pool = RouterPool(app.state.router_providers, chat_route.ROUTER_ORDER)
    app.state.embedders, app.state.embed_order = E.build_embedders()
    _boot_economics(app)
    app.state.started_at = time.time()
    agent_bridge = S16AgentBridge()
    app.state.agent_bridge = agent_bridge
    yield
    await agent_bridge.close()
    t = getattr(app.state, "telemetry", None)
    if t is not None:
        t.flush()
        t.shutdown()


app = FastAPI(
    title="GLC v5 — Connected Gateway for Models, Channels and S16 Agents",
    lifespan=lifespan,
)

app.include_router(chat_route.router)
app.include_router(transcribe_route.router)
app.include_router(speak_route.router)
app.include_router(control_route.router)
app.include_router(channels_route.router)
app.include_router(economics_route.router)
app.include_router(observability_route.router)
# A budget refusal writes no ledger row and opens no span. Observed here instead,
# at the boundary, so `/v1/refusals` can show what the controller actually stopped.
app.add_middleware(observability_route.RefusalRecorder)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    return FileResponse(STATIC / "dashboard.html")


@app.get("/help", response_class=FileResponse)
async def help_page() -> FileResponse:
    return FileResponse(STATIC / "help.html")


@app.get("/channels", response_class=FileResponse)
async def channels_page() -> FileResponse:
    return FileResponse(STATIC / "channels.html")


@app.get("/healthz")
async def healthz():
    return {"ok": True, "port": PORT}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("glc.main:app", host="0.0.0.0", port=PORT, reload=False)
