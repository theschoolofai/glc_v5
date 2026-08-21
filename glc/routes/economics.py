"""v4 economics surface: budgets, principal cost rollups, cache statistics.

Four new endpoints, none of which changes an existing one:

    GET  /v1/budget/{principal}   where does this principal stand
    POST /v1/budget               arm or move a ceiling at runtime
    GET  /v1/cost/by_principal    the five-dimension rollup (superset of by_agent)
    GET  /v1/cache/stats          hit rate, tokens and dollars saved

Plus two read-only introspection routes, because "the config is the API" only
holds if you can see the loaded config:

    GET  /v1/pricing              the resolved per-model price table
    GET  /v1/telemetry            tracer state, exporters, content-capture flag

`{principal}` is `"<dimension>:<value>"` — for example `agent:summarizer` or
`session:run-42`. A colon inside a path segment is legal, so these read
naturally in a browser.
"""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Request

from glc import db
from glc.config import get_or_create_install_token
from glc.economics import budget as _budget
from glc.economics import meter as _meter
from glc.economics import pricing as _pricing
from glc.llm_schemas import BudgetSetRequest
from glc.telemetry import otel as _otel

router = APIRouter()


def _require_token(authorization: str | None) -> None:
    """The economics surface arms and disarms the budget ceiling and purges
    the cache — operator actions. Same guard as /v1/control/*."""
    expected = get_or_create_install_token()
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token (Authorization: Bearer <install_token>)")
    presented = authorization.removeprefix("Bearer ").strip()
    if presented != expected:
        raise HTTPException(403, "install token mismatch")


def _controller(request: Request) -> _budget.BudgetController:
    return getattr(request.app.state, "budget", None) or _budget.get_controller()


def _meter_for(request: Request) -> _meter.Meter:
    return getattr(request.app.state, "meter", None) or _meter.get_meter()


# ── budgets ─────────────────────────────────────────────────────────────────


@router.get("/v1/budget")
async def list_budgets(request: Request):
    """Every loaded policy, from budgets.yaml and from runtime overrides."""
    return _controller(request).describe()


@router.get("/v1/budget/{principal:path}")
async def get_budget(principal: str, request: Request):
    """Limit, spend, and remaining allowance for one principal."""
    ctl = _controller(request)
    try:
        dim, value = _budget._parse_principal(principal)
    except _budget.BudgetConfigError as e:
        raise HTTPException(400, str(e)) from e
    status = ctl.status_for(dim, value)
    if status is None:
        return {
            "principal": f"{dim}:{value}",
            "dimension": dim,
            "value": value,
            "governed": False,
            "limit_usd": None,
            "spent_usd": round(db.spend_usd(dim, value, since=db._period_start("day")), 8),
            "spent_lifetime_usd": round(db.spend_usd(dim, value, since=0.0), 8),
            "message": (
                "No budget policy matches this principal, so it is unlimited. "
                "Add one to budgets.yaml or POST /v1/budget."
            ),
        }
    out = status.as_dict()
    out["governed"] = True
    out["policy"] = ctl.policy_for(dim, value).as_dict()
    out["spent_lifetime_usd"] = round(db.spend_usd(dim, value, since=0.0), 8)
    return out


@router.post("/v1/budget")
async def set_budget(
    req: BudgetSetRequest,
    request: Request,
    authorization: str | None = Header(default=None),
):
    """Arm, move, or (with limit_usd 0) immediately close a ceiling.

    The write goes to a runtime table that shadows budgets.yaml, so an operator
    can clamp a runaway session in one call without editing a file or
    redeploying — which is the whole point of a kill-switch.
    """
    _require_token(authorization)
    ctl = _controller(request)
    try:
        pol = ctl.set_limit(
            req.principal,
            req.limit_usd,
            period=req.period,
            include_errors=req.include_errors,
        )
    except _budget.BudgetConfigError as e:
        raise HTTPException(400, str(e)) from e
    status = ctl.status_for(pol.dimension, pol.pattern)
    return {
        "ok": True,
        "policy": pol.as_dict(),
        "status": status.as_dict() if status else None,
    }


@router.delete("/v1/budget/{principal:path}")
async def delete_budget(
    principal: str,
    request: Request,
    authorization: str | None = Header(default=None),
):
    """Remove a runtime override. Any budgets.yaml entry underneath re-applies."""
    _require_token(authorization)
    ctl = _controller(request)
    try:
        removed = ctl.clear_limit(principal)
    except _budget.BudgetConfigError as e:
        raise HTTPException(400, str(e)) from e
    return {"ok": True, "removed": removed, "principal": principal}


# ── cost attribution ────────────────────────────────────────────────────────


@router.get("/v1/cost/by_principal")
async def cost_by_principal(
    request: Request,
    dimension: str | None = None,
    value: str | None = None,
    since: float | None = None,
    period: str = "day",
    group_by_provider: bool = True,
):
    """Cost rollup across the five attribution dimensions.

    A superset of `/v1/cost/by_agent`: same counts, plus real per-model dollars,
    cache savings, and the tenant/project/user levels that v3 had no column for.
    Omit `dimension` to get all five at once.
    """
    m = _meter_for(request)
    start = since if since is not None else db._period_start(period)
    if dimension:
        if dimension not in _meter.DIMENSIONS:
            raise HTTPException(
                400,
                f"unknown dimension {dimension!r}; expected one of {list(_meter.DIMENSIONS)}",
            )
        rows = m.rollup(dimension, since=start, value=value, group_by_provider=group_by_provider)
        return {
            "period": period,
            "since": start,
            "dimension": dimension,
            "rows": rows,
            "total_usd": round(sum(r["dollars"] for r in rows), 8),
        }
    by_dim = m.rollup_all(since=start)
    return {
        "period": period,
        "since": start,
        "dimensions": list(_meter.DIMENSIONS),
        "by_dimension": by_dim,
        "totals_usd": {d: round(sum(r["dollars"] for r in rows), 8) for d, rows in by_dim.items()},
    }


# ── cache ───────────────────────────────────────────────────────────────────


@router.get("/v1/cache/stats")
async def cache_stats(request: Request):
    """Semantic-cache hit rate plus tokens and dollars not spent.

    `usd_saved` is the sum of what the *original* calls actually cost, so it is
    a measured number and not a modelled one. `best_similarity_missed` is the
    honest counterweight: how close the closest rejected match got, which is
    what you look at before lowering the threshold.
    """
    sc = getattr(request.app.state, "semantic_cache", None)
    if sc is None:
        return {
            "enabled": False,
            "message": "semantic cache not initialised (see semantic.enabled in cache.yaml)",
        }
    stats = sc.stats()
    stats["enabled"] = sc.config.enabled
    stats["gemini_prompt_cache"] = {
        "note": (
            "Prompt caching is a separate mechanism: it discounts input tokens on a "
            "call that still runs. Its accounting is in the ledger's "
            "cache_read_tokens / cache_create_tokens columns and in /v1/status."
        ),
        "today": {
            p: {
                "cache_read_tokens": row.get("cache_reads") or 0,
                "cache_create_tokens": row.get("cache_creates") or 0,
            }
            for p, row in db.aggregate(call_role="worker").items()
        },
    }
    return stats


@router.post("/v1/cache/purge")
async def cache_purge(
    request: Request,
    expired_only: bool = True,
    authorization: str | None = Header(default=None),
):
    _require_token(authorization)
    sc = getattr(request.app.state, "semantic_cache", None)
    if sc is None:
        raise HTTPException(503, "semantic cache not initialised")
    return {"ok": True, "removed": sc.purge(expired_only=expired_only)}


# ── introspection ───────────────────────────────────────────────────────────


@router.get("/v1/pricing")
async def pricing(request: Request, provider: str | None = None, model: str | None = None):
    """The loaded price table, or the resolved rate for one (provider, model).

    `price_source` on a single lookup tells you *how* the rate was found, so a
    $0.00 report can always be traced to either a genuine free tier or a
    missing pricing.yaml entry.
    """
    table = _pricing.load_pricing()
    if provider or model:
        p = table.price_for(provider or "", model)
        return {
            "provider": provider,
            "model": model,
            "input_usd_per_mtok": p.input_usd_per_mtok,
            "output_usd_per_mtok": p.output_usd_per_mtok,
            "cache_read_multiplier": p.cache_read_multiplier,
            "cache_write_multiplier": p.cache_write_multiplier,
            "batch_multiplier": p.batch_multiplier,
            "quality": p.quality,
            "price_source": p.source,
            "priced": p.priced,
        }
    rtr = getattr(request.app.state, "router", None)
    live = {}
    if rtr is not None:
        for name, prov in rtr.providers.items():
            p = table.price_for(name, prov.model)
            live[name] = {
                "model": prov.model,
                "input_usd_per_mtok": p.input_usd_per_mtok,
                "output_usd_per_mtok": p.output_usd_per_mtok,
                "quality": p.quality,
                "price_source": p.source,
            }
    return {**table.describe(), "configured_providers": live}


@router.get("/v1/telemetry")
async def telemetry(request: Request):
    """Tracer state. `capture_content` false is the safe default."""
    t = getattr(request.app.state, "telemetry", None) or _otel.get_telemetry()
    return t.describe()
