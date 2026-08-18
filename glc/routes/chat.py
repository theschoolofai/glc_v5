"""V9-compatibility routes: /v1/chat, /v1/chat/batch, /v1/providers,
/v1/capabilities, /v1/status, /v1/routers, /v1/calls.

This module is a near-verbatim port of llm_gatewayV9/main.py's chat
pipeline. Bare imports were rewritten as package-relative imports; the
V9 fixes (json_object hint injection, Gemini cooldown handling, day
rollover, default URL inheritance) are preserved verbatim. Do not
regress them — tests in test_v9_compat.py assert behaviour shape.

V4 additions inside `/v1/chat`, in the order they run:

1. **semantic cache** — on a hit, return the stored response and never
   contact a provider,
2. **routing policy** — apply the role's tier floor/ceiling to whatever the
   classifier said, then order candidates by cost/quality,
3. **budget admission** — project the worst-case cost and refuse with a 402
   envelope before the provider is contacted,
4. **OTel span** — one `chat` span per provider call, carrying
   `gen_ai.usage.*` and the computed cost,
5. **meter** — one priced ledger row per call, attributed to all five
   principal dimensions,
6. **cascade** — on a low-confidence answer, escalate one tier and retry.

Every one of those is inert unless configured, so a glc_v4 with the shipped
config files behaves exactly like glc_v3.
"""

from __future__ import annotations

import asyncio as _asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from jsonschema import Draft202012Validator, ValidationError

from glc import db
from glc import providers as P
from glc.economics import budget as _budget
from glc.economics import meter as _meter
from glc.llm_schemas import (
    BatchChatRequest,
    ChatRequest,
    ChatResponse,
    EmbedRequest,
    EmbedResponse,
    ResponseFormat,
    RouterDecision,
    ToolCall,
    VisionRequest,
)
from glc.routing import DEFAULT_ROUTER_ORDER, LIMITS, SHORTCUTS
from glc.routing import policy as _policy
from glc.telemetry import otel as _otel

DEFAULT_ORDER = ["ollama", "gemini", "nvidia", "groq", "cerebras", "openrouter", "github"]
ORDER = [x.strip() for x in os.getenv("LLM_ORDER", ",".join(DEFAULT_ORDER)).split(",") if x.strip()]
ROUTER_ORDER = [
    x.strip() for x in os.getenv("ROUTER_ORDER", ",".join(DEFAULT_ROUTER_ORDER)).split(",") if x.strip()
]

_AGENT_ROUTING_PATH = Path(__file__).parent.parent / "agent_routing.yaml"
AGENT_ROUTING: dict[str, str] = {}
if _AGENT_ROUTING_PATH.exists():
    try:
        AGENT_ROUTING = yaml.safe_load(_AGENT_ROUTING_PATH.read_text()) or {}
    except Exception as e:  # pragma: no cover
        print(f"[glc] failed to parse agent_routing.yaml: {e!r}")

# V3's hardcoded tier ring. Kept as the module-level name because callers and
# `/v1/routers` read it, but it is now the *fallback* — `routing.yaml` is
# authoritative and `_tier_order()` prefers it. The two agree by construction:
# routing.yaml's TINY and LARGE rings are these lists verbatim.
TIER_TO_ORDER = {
    "TINY": ["github", "openrouter", "groq", "nvidia", "cerebras", "gemini", "ollama"],
    "LARGE": ["gemini", "groq", "nvidia", "cerebras", "github", "openrouter", "ollama"],
}


def _routing_policy() -> _policy.RoutingPolicy | None:
    """The loaded routing policy, or None if routing.yaml is unusable.

    A broken routing.yaml must not take the gateway down — it falls back to
    v3's TIER_TO_ORDER and says so on the console.
    """
    try:
        return _policy.load_policy()
    except Exception as e:  # pragma: no cover - config error path
        print(f"[glc] routing.yaml unusable, falling back to v3 TIER_TO_ORDER: {e!r}")
        return None


ROUTER_SAMPLE_HEAD = 400
ROUTER_SAMPLE_TAIL = 400
ROUTER_PROMPT = (
    "You are a routing classifier. Given a token_count and a content sample, "
    "output exactly one of: TINY, LARGE, or HUGE.\n\n"
    "Rules:\n"
    "- TINY: token_count below 1000 with simple factual content.\n"
    "- LARGE: token_count between 1000 and 8000, OR token_count below 1000 "
    "but content is dense (code, base64, multilingual, technical).\n"
    "- HUGE: token_count above 8000.\n\n"
    "Output the single word and nothing else."
)

router = APIRouter()


# ─────────────────────────── helpers (verbatim port) ──────────────────────────


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.4)


def _build_sample(text: str) -> str:
    if len(text) <= ROUTER_SAMPLE_HEAD + ROUTER_SAMPLE_TAIL + 10:
        return text
    return text[:ROUTER_SAMPLE_HEAD] + "\n...\n" + text[-ROUTER_SAMPLE_TAIL:]


def _tier_from_count(tokens: int) -> str:
    if tokens > 8000:
        return "HUGE"
    if tokens >= 1000:
        return "LARGE"
    return "TINY"


def _parse_tier(text: str) -> str | None:
    up = (text or "").upper()
    for tier in ("HUGE", "LARGE", "TINY"):
        if tier in up:
            return tier
    return None


async def _classify_tier(req, role, router_pool, prompt_text):
    estimated = _estimate_tokens(prompt_text)
    if estimated > 8000:
        return RouterDecision(
            role=role,
            tier="HUGE",
            estimated_tokens=estimated,
            router_provider="(skipped)",
            router_model="(skipped)",
            router_latency_ms=0,
            fallback_used=True,
        )
    sample = _build_sample(prompt_text)
    envelope = f"token_count: {estimated}\nsample:\n{sample}"
    call_role = f"router_{role}"
    last_provider = last_model = ""
    last_latency = 0
    for name in router_pool.candidates():
        ok, _ = router_pool.state[name].can_use(LIMITS[name], 400)
        if not ok:
            continue
        provider = router_pool.providers[name]
        t0 = time.time()
        router_pool.state[name].record(0)
        last_provider, last_model = name, provider.model
        try:
            result = await provider.chat(
                messages=[{"role": "user", "content": envelope}],
                system_blocks=ROUTER_PROMPT,
                max_tokens=8,
                temperature=0,
                model=None,
                tools=None,
                tool_choice=None,
                reasoning="off",
                response_format=None,
                cache_system=False,
            )
            latency = int((time.time() - t0) * 1000)
            last_latency = latency
            tokens = (result.get("input_tokens") or 0) + (result.get("output_tokens") or 0)
            router_pool.state[name].tokens_today += tokens
            router_pool.state[name].tokens_minute.append((time.time(), tokens))
            tier = _parse_tier(result.get("text", ""))
            if tier == "HUGE" and estimated <= 8000:
                tier = "LARGE"
            if tier is None:
                db.log_call(
                    provider=name,
                    model=result.get("model", provider.model),
                    input_tokens=result.get("input_tokens", 0),
                    output_tokens=result.get("output_tokens", 0),
                    latency_ms=latency,
                    status="error",
                    error=f"unparseable tier reply: {result.get('text', '')[:100]}",
                    prompt_chars=len(envelope),
                    call_role=call_role,
                    router_decision="unparseable",
                )
                continue
            db.log_call(
                provider=name,
                model=result.get("model", provider.model),
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
                latency_ms=latency,
                status="ok",
                prompt_chars=len(envelope),
                response_chars=len(result.get("text", "")),
                call_role=call_role,
                router_decision=tier,
            )
            return RouterDecision(
                role=role,
                tier=tier,
                estimated_tokens=estimated,
                router_provider=name,
                router_model=result.get("model", provider.model),
                router_latency_ms=latency,
                fallback_used=False,
            )
        except Exception as e:
            latency = int((time.time() - t0) * 1000)
            last_latency = latency
            db.log_call(
                provider=name,
                model=provider.model,
                status="error",
                error=str(e)[:500],
                latency_ms=latency,
                call_role=call_role,
                router_decision="error",
            )
            continue
    return RouterDecision(
        role=role,
        tier=_tier_from_count(estimated),
        estimated_tokens=estimated,
        router_provider=last_provider or "(unavailable)",
        router_model=last_model or "(unavailable)",
        router_latency_ms=last_latency,
        fallback_used=True,
    )


def _normalize_messages(req: ChatRequest):
    if req.messages:
        return list(req.messages)
    return [{"role": "user", "content": req.prompt or ""}]


def _system_blocks(req: ChatRequest):
    if req.system is None:
        return None
    if isinstance(req.system, str):
        if req.cache_system:
            return [{"text": req.system, "cache": True}]
        return req.system
    return [b.model_dump() if hasattr(b, "model_dump") else b for b in req.system]


def _est_tokens(messages, system_blocks, max_tokens):
    chars = 0
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, list):
            chars += len(P._extract_text_blocks(c))
            chars += 1200 * sum(
                1 for b in c if isinstance(b, dict) and b.get("type") in ("image_url", "image", "input_image")
            )
        else:
            chars += len(str(c))
    if isinstance(system_blocks, str):
        chars += len(system_blocks)
    elif isinstance(system_blocks, list):
        for b in system_blocks:
            chars += len(b.get("text", "") if isinstance(b, dict) else "")
    return chars // 4 + max_tokens


#: Default cooldown per failure class, in seconds — v3's numbers verbatim except
#: `timeout`. routing.yaml's `backoff:` block overrides any of them by name, so
#: benching a flaky provider for longer is a config edit and not a code change.
#:
#: `timeout` was 10s, and the measured consequence was ugly: NVIDIA's configured
#: model answers nothing and the HTTP client waits its full 180s, so a 10s bench
#: means the very next request pays 180s again. A dead endpoint has to stay
#: benched for much longer than it takes to notice it is dead.
BACKOFF_DEFAULTS = {
    "queue": 15,
    "rpm": 60,
    "rpd": 3600,
    "rate_limited": 30,
    "upstream_5xx": 20,
    "timeout": 600,
    "auth": 600,
}


def _backoff_seconds(kind: str) -> float:
    """Cooldown for a failure class, from routing.yaml when it says so."""
    default = BACKOFF_DEFAULTS.get(kind, 0)
    pol = _routing_policy()
    cfg = getattr(pol, "backoff", None) or {}
    try:
        return float(cfg.get(kind, default))
    except (TypeError, ValueError):
        return float(default)


def _backoff_for(err: Exception, has_model_override: bool = False):
    # The class name is part of the haystack on purpose. An httpx ReadTimeout
    # stringifies to the EMPTY STRING, so matching str(err) alone matched nothing
    # and returned a zero cooldown — the provider that had just burned 180s came
    # straight back as a candidate on the next request.
    msg = f"{type(err).__name__} {err}".lower()
    status = getattr(err, "status", None)
    if status == 429:
        if "queue" in msg:
            return _backoff_seconds("queue"), "server queue full"
        if "quota" in msg or "rpm" in msg or "per minute" in msg:
            return _backoff_seconds("rpm"), "RPM quota burned"
        if "rpd" in msg or "per day" in msg or "daily" in msg:
            return _backoff_seconds("rpd"), "RPD quota burned"
        return _backoff_seconds("rate_limited"), "rate limited"
    if status and 500 <= status < 600:
        return _backoff_seconds("upstream_5xx"), f"upstream {status}"
    if status == 408 or "timeout" in msg or "timedout" in msg:
        return _backoff_seconds("timeout"), "timeout"
    if status in (401, 403):
        if has_model_override:
            return 0, ""
        return _backoff_seconds("auth"), "auth error"
    if status == 404 and has_model_override:
        return 0, ""
    return 0, ""


def _attempts_str(attempts):
    return "; ".join(f"{a['provider']}:{a['reason']}" for a in attempts)


def _required_caps(req: ChatRequest):
    caps = []
    if req.tools:
        caps.append("tools")
    if req.reasoning and req.reasoning != "off":
        caps.append("reasoning")
    if req.response_format:
        caps.append("structured")
    if req.messages:
        for m in req.messages:
            if P._content_has_image(m.get("content")):
                caps.append("vision")
                break
    return caps


async def _resolve_image_urls(messages):
    import base64

    import httpx as _httpx

    async def _fetch_to_data_url(url: str) -> str:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; GLCv1/0.1; +image-resolver)",
            "Accept": "image/*,*/*;q=0.8",
        }
        async with _httpx.AsyncClient(timeout=30, follow_redirects=True, headers=headers) as c:
            try:
                r = await c.get(url)
                r.raise_for_status()
            except _httpx.HTTPError as e:
                raise HTTPException(400, f"failed to fetch image url {url!r}: {e}")
            mt = (r.headers.get("content-type") or "image/png").split(";")[0].strip()
            b64 = base64.b64encode(r.content).decode()
            return f"data:{mt};base64,{b64}"

    out = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            out.append(m)
            continue
        new_blocks = []
        changed = False
        for b in content:
            if isinstance(b, dict) and b.get("type") == "image_url":
                iu = b.get("image_url")
                url = iu.get("url") if isinstance(iu, dict) else iu
                if isinstance(url, str) and url.startswith(("http://", "https://")):
                    data_url = await _fetch_to_data_url(url)
                    new_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
                    changed = True
                    continue
            new_blocks.append(b)
        if changed:
            new_m = dict(m)
            new_m["content"] = new_blocks
            out.append(new_m)
        else:
            out.append(m)
    return out


def _validate_structured(text: str, schema: dict):
    try:
        obj = json.loads(text)
    except Exception as e:
        raise ValueError(f"output is not JSON: {e}")
    Draft202012Validator(schema).validate(obj)
    return obj


# ─────────────────────── v4 helpers (economics / policy) ──────────────────────


def _tier_candidates(pol, tier: str, rtr, est_tokens: int, tradeoff: float | None = None):
    """Ordered provider instances for a tier, plus per-candidate rejections.

    Prefers `routing.yaml`. Falls back to v3's TIER_TO_ORDER when the policy is
    unavailable or does not declare the tier.

    Note on a v3 bug this fixes: v3 filtered the tier ring with
    `p in rtr.providers`, but Gemini is registered as `gemini_1..N` (one entry
    per key), so the literal name `gemini` never matched and Gemini was silently
    dropped from every auto-routed call — including the LARGE tier where it is
    listed first. The policy expands base names to live instances, so it is
    reachable again, which is also what makes the HUGE tier servable (Gemini's
    1M-token window is the only one wide enough).
    """
    if pol is not None and tier in pol.tiers:
        return pol.order_for(tier, rtr.providers, limits=LIMITS, est_tokens=est_tokens, tradeoff=tradeoff)
    ring = TIER_TO_ORDER.get(tier) or []
    return rtr.expand(ring), []


def _cache_request_fields(req: ChatRequest, system_blocks) -> dict:
    """The fields that must match exactly before two prompts are comparable."""
    return {
        "model": req.model,
        "provider": req.provider,
        "system": system_blocks if isinstance(system_blocks, str) else json.dumps(system_blocks, default=str),
        "tools": json.dumps([t.model_dump() for t in req.tools], default=str) if req.tools else None,
        "response_format": (req.response_format.model_dump() if req.response_format else None),
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "auto_route": req.auto_route,
    }


def _trace_ids(span) -> dict | None:
    tid = span.attributes.get("trace_id")
    if tid:
        return {"trace_id": tid, "span_id": span.attributes.get("span_id")}
    try:
        from opentelemetry import trace as _t

        ctx = _t.get_current_span().get_span_context()
        if ctx and ctx.trace_id:
            return {"trace_id": f"{ctx.trace_id:032x}", "span_id": f"{ctx.span_id:016x}"}
    except Exception:  # pragma: no cover
        pass
    return None


def _budget_refusal(admission) -> HTTPException:
    """A 402 with numbers in it.

    402 Payment Required is the honest status: the request was well-formed and
    permitted, and the only thing wrong with it is that it costs more than the
    principal has left. Nothing was sent to a provider, so nothing was billed.
    """
    return HTTPException(402, admission.envelope())


# ─────────────────────────── routes ───────────────────────────


@router.post("/v1/chat")
async def chat(req: ChatRequest, request: Request):
    state = request.app.state
    rtr = state.router
    router_pool = state.router_pool
    pol = _routing_policy()
    ctl = getattr(state, "budget", None) or _budget.get_controller()
    telemetry = getattr(state, "telemetry", None) or _otel.get_telemetry()
    meter = getattr(state, "meter", None) or _meter.get_meter()
    scache = getattr(state, "semantic_cache", None)
    principal = _meter.Principal.from_request(req)

    messages = _normalize_messages(req)
    if any(P._content_has_image(m.get("content")) for m in messages):
        messages = await _resolve_image_urls(messages)
    system_blocks = _system_blocks(req)
    prompt_text = "".join(
        (
            P._extract_text_blocks(m.get("content", ""))
            if isinstance(m.get("content"), list)
            else str(m.get("content", ""))
        )
        for m in messages
    )
    est = _est_tokens(messages, system_blocks, req.max_tokens)
    est_input = max(0, est - req.max_tokens)
    explicit_override = bool(req.provider)
    required_caps = _required_caps(req)

    if req.agent and not req.provider:
        pinned = AGENT_ROUTING.get(req.agent)
        if pinned and pinned in rtr.providers:
            req.provider = pinned
            explicit_override = True

    # ── 1. semantic cache: a hit means no provider is contacted at all ───────
    cache_info: dict | None = None
    cache_fields = _cache_request_fields(req, system_blocks)
    if scache is not None:
        lookup = await scache.lookup(
            prompt_text,
            cache_fields,
            principal=principal,
            opt_in=req.semantic_cache,
            temperature=req.temperature,
            has_tools=bool(req.tools),
            stream=bool(req.stream),
        )
        cache_info = lookup.as_dict()
        if lookup.hit and lookup.response:
            t0 = time.time()
            with telemetry.span(_otel.SPAN_CHAT) as span:
                span.set_request(
                    provider=lookup.provider or "(cache)",
                    model=lookup.model,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                )
                # Zero usage is the point: the tokens the provider would have
                # charged for are recorded as *saved*, not as spent.
                span.set_usage(input_tokens=0, output_tokens=0, response_model=lookup.model)
                span.set_principal(principal)
                span.set_many(
                    {
                        _otel.GLC_CACHE_HIT: True,
                        _otel.GLC_CACHE_KIND: "semantic",
                        _otel.GLC_CACHE_SIMILARITY: round(lookup.similarity, 6),
                        _otel.GLC_CACHE_TOKENS_SAVED: lookup.tokens_saved,
                        _otel.GLC_CACHE_USD_SAVED: lookup.usd_saved,
                        _otel.GLC_COST_USD: 0.0,
                        _otel.GLC_ROUTING_ROLE: req.auto_route,
                    }
                )
                latency = int((time.time() - t0) * 1000)
                meter.record(
                    provider=lookup.provider or "(cache)",
                    model=lookup.model or "",
                    principal=principal,
                    usage=_meter.Usage(),
                    latency_ms=latency,
                    status="ok",
                    prompt_chars=len(prompt_text),
                    response_chars=len(str(lookup.response.get("text") or "")),
                    override=req.provider,
                    call_role="worker",
                    role=req.auto_route,
                    cache_hit=True,
                    cache_kind="semantic",
                    tokens_saved=lookup.tokens_saved,
                    usd_saved=lookup.usd_saved,
                )
                out = dict(lookup.response)
                out["latency_ms"] = latency
                out["cache"] = cache_info
                out["cost"] = {"total_usd": 0.0, "price_source": "semantic-cache-hit"}
                out["trace"] = _trace_ids(span)
                return out

    # ── 2. tier classification, then the role policy on top of it ───────────
    retries = 0
    router_decision: RouterDecision | None = None
    tier: str | None = None
    tier_path: list[str] = []
    policy_rejections: list[dict] = []

    if req.auto_route and not req.provider:
        router_decision = await _classify_tier(req, req.auto_route, router_pool, prompt_text)
        classified = router_decision.tier
        if pol is not None:
            decision = pol.tier_for(req.auto_route, classified)
            tier = decision.tier
            router_decision.role = decision.role
            router_decision.tier = decision.tier
            router_decision.classified_tier = classified
            router_decision.policy_clamped = decision.clamped
            router_decision.policy_reason = decision.reason
        else:
            tier = classified
            router_decision.classified_tier = classified
        tier_path = [tier]
        candidates, policy_rejections = _tier_candidates(
            pol, tier, rtr, est_input, tradeoff=req.cost_quality_tradeoff
        )
        if not candidates:
            # v3 raised a blanket 503 for HUGE pointing at an unimplemented
            # summariser. v4 only refuses when no configured provider can hold
            # the context, and says which limit was the problem.
            raise HTTPException(
                503,
                {
                    "error": "no provider can serve this tier",
                    "tier": tier,
                    "estimated_input_tokens": est_input,
                    "rejected": policy_rejections,
                    "hint": (
                        "Every candidate for this tier was filtered out — usually because the "
                        "prompt exceeds each provider's max_ctx. Widen tiers."
                        f"{tier}.order in routing.yaml, chunk the input, or pin a "
                        "long-context provider explicitly."
                    ),
                    "router_decision": router_decision.model_dump(),
                },
            )
    else:
        candidates = rtr.candidates(req.provider) if req.provider else rtr.candidates()

    if req.provider and not candidates:
        raise HTTPException(
            400,
            f"unknown provider '{req.provider}'. Try one of: {list(rtr.providers)} or shortcuts {list(SHORTCUTS)}",
        )

    all_attempts: list[dict] = list(policy_rejections)
    last_err = None
    escalations = 0
    budget_info: dict | None = None
    last_refusal = None

    if explicit_override and len(candidates) == 1:
        deadline = time.time() + 30
        while time.time() < deadline:
            name, _ = rtr.pick(est, candidates, required_caps=required_caps)
            if name is not None:
                break
            cd = rtr.state[candidates[0]].snapshot(LIMITS[candidates[0]])["cooldown_remaining"]
            if cd <= 0 or cd > 30:
                break
            await _asyncio.sleep(min(cd + 0.05, 5))

    # ── 3. the candidate ring, wrapped in the cascade loop ──────────────────
    # The inner `for` is v3's failover ring, unchanged. The outer `while` is the
    # cascade: an escalation sets `escalated` and re-enters the ring with the
    # next tier's candidates. Any other exit leaves the loop for good, so a
    # cascade cannot spin.
    while True:
        escalated = False
        for _ in range(len(candidates) + 1):
            name, atts = rtr.pick(est, candidates, required_caps=required_caps)
            all_attempts.extend(atts)
            if name is None:
                break
            provider = rtr.providers[name]
            model_for_call = req.model or provider.model

            # ── budget admission. Runs BEFORE the provider is contacted, on a
            # worst-case projection, for the model that is actually about to be
            # called. A refusal here costs nothing because nothing was sent.
            admission = ctl.admit(
                principal,
                ctl.project(name, model_for_call, est_input, req.max_tokens, batch=bool(req.batch)),
            )
            if not admission.allowed:
                last_refusal = admission
                budget_info = admission.as_dict()
                all_attempts.append({"provider": name, "reason": f"refused:budget {admission.reason}"})
                # Try a cheaper candidate rather than failing the whole request:
                # the next provider in the ring may be affordable. If none are,
                # the 402 below fires.
                candidates = [c for c in candidates if c != name]
                continue
            budget_info = admission.as_dict()

            t0 = time.time()
            rtr.state[name].record(0)
            span_cm = telemetry.span(_otel.SPAN_CHAT)
            span = span_cm.__enter__()
            span.set_request(
                provider=name,
                model=model_for_call,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )
            span.set_principal(principal)
            span.set_admission(admission)
            span.set_many(
                {
                    _otel.GLC_ROUTING_ROLE: req.auto_route,
                    _otel.GLC_ROUTING_TIER: tier,
                    _otel.GLC_ROUTING_TIER_CLASSIFIED: (
                        router_decision.classified_tier if router_decision else None
                    ),
                    _otel.GLC_ROUTING_ESCALATIONS: escalations,
                    _otel.GLC_CACHE_HIT: False,
                }
            )
            span.set_content(messages=messages)
            try:
                if req.stream:

                    async def gen():
                        try:
                            agg = []
                            async for chunk in provider.stream(
                                messages,
                                max_tokens=req.max_tokens,
                                temperature=req.temperature,
                                model=req.model,
                                tools=req.tools,
                                tool_choice=req.tool_choice,
                                reasoning=req.reasoning,
                                response_format=req.response_format,
                                system_blocks=system_blocks,
                                cache_system=bool(req.cache_system),
                            ):
                                agg.append(chunk)
                                if chunk.startswith("[[TOOL_CALL_DELTA]]"):
                                    yield f"data: {json.dumps({'provider': name, 'tool_call_delta': chunk[len('[[TOOL_CALL_DELTA]] ') :]})}\n\n"
                                else:
                                    yield f"data: {json.dumps({'provider': name, 'delta': chunk})}\n\n"
                            text = "".join(agg)
                            latency = int((time.time() - t0) * 1000)
                            meter.record(
                                provider=name,
                                model=req.model or provider.model,
                                principal=principal,
                                usage=_meter.Usage(),
                                latency_ms=latency,
                                status="ok",
                                prompt_chars=len(prompt_text),
                                response_chars=len(text),
                                override=req.provider,
                                attempted=_attempts_str(all_attempts),
                                retries=retries,
                                role=req.auto_route,
                                tier=tier,
                            )
                            yield f"data: {json.dumps({'done': True, 'provider': name})}\n\n"
                        except Exception as e:
                            meter.record(
                                provider=name,
                                model=req.model or provider.model,
                                principal=principal,
                                usage=_meter.Usage(),
                                status="error",
                                error=str(e)[:500],
                                latency_ms=int((time.time() - t0) * 1000),
                                prompt_chars=len(prompt_text),
                                override=req.provider,
                                attempted=_attempts_str(all_attempts),
                                retries=retries,
                                role=req.auto_route,
                                tier=tier,
                            )
                            yield f"data: {json.dumps({'error': str(e)[:300]})}\n\n"

                    span_cm.__exit__(None, None, None)
                    return StreamingResponse(gen(), media_type="text/event-stream")

                try:
                    result = await provider.chat(
                        messages,
                        max_tokens=req.max_tokens,
                        temperature=req.temperature,
                        model=req.model,
                        tools=req.tools,
                        tool_choice=req.tool_choice,
                        reasoning=req.reasoning,
                        response_format=req.response_format,
                        system_blocks=system_blocks,
                        cache_system=bool(req.cache_system),
                    )
                except (P.ProviderError, Exception) as transient:
                    status = getattr(transient, "status", None)
                    msg = str(transient).lower()
                    retryable = (
                        (status is not None and 500 <= status < 600) or status == 408 or "timeout" in msg
                    )
                    if not retryable:
                        raise
                    await _asyncio.sleep(min(2.0, 0.5 * (2**retries)))
                    retries += 1
                    result = await provider.chat(
                        messages,
                        max_tokens=req.max_tokens,
                        temperature=req.temperature,
                        model=req.model,
                        tools=req.tools,
                        tool_choice=req.tool_choice,
                        reasoning=req.reasoning,
                        response_format=req.response_format,
                        system_blocks=system_blocks,
                        cache_system=bool(req.cache_system),
                    )
                latency = int((time.time() - t0) * 1000)

                parsed = None
                schema_failed = False
                if req.response_format and req.response_format.schema_ and not result["tool_calls"]:
                    try:
                        parsed = _validate_structured(result["text"], req.response_format.schema_)
                    except (ValueError, ValidationError) as ve:
                        fix_msgs = list(messages) + [
                            {"role": "assistant", "content": result["text"]},
                            {
                                "role": "user",
                                "content": f"Your previous reply did not match the required JSON schema: {ve}. Reply ONLY with valid JSON conforming to the schema.",
                            },
                        ]
                        result = await provider.chat(
                            fix_msgs,
                            max_tokens=req.max_tokens,
                            temperature=0,
                            model=req.model,
                            response_format=req.response_format,
                            system_blocks=system_blocks,
                            cache_system=bool(req.cache_system),
                        )
                        try:
                            parsed = _validate_structured(result["text"], req.response_format.schema_)
                        except (ValueError, ValidationError) as ve2:
                            # v3 raised 503 here unconditionally. v4 gives the
                            # cascade a chance to escalate first — a schema
                            # failure is the cleanest low-confidence signal
                            # there is — and only 503s if it cannot.
                            schema_failed = True
                            if pol is None or not _may_escalate(
                                pol, req, tier, escalations, schema_failed=True
                            ):
                                raise HTTPException(503, f"structured output failed validation: {ve2}")

                tokens = (result["input_tokens"] or 0) + (result["output_tokens"] or 0)
                rtr.state[name].tokens_today += tokens
                rtr.state[name].tokens_minute.append((time.time(), tokens))
                if router_decision is not None:
                    router_decision.chosen_worker_provider = name
                    router_decision.chosen_worker_model = result["model"]

                usage = _meter.Usage.from_provider_result(result)
                record = meter.record(
                    provider=name,
                    model=result["model"],
                    principal=principal,
                    usage=usage,
                    batch=bool(req.batch),
                    latency_ms=latency,
                    status="ok",
                    prompt_chars=len(prompt_text),
                    response_chars=len(result["text"]),
                    override=req.provider,
                    attempted=_attempts_str(all_attempts),
                    tool_calls=len(result["tool_calls"]),
                    reasoning_applied=result["reasoning_applied"],
                    tool_dialect=result["tool_call_dialect"],
                    call_role="worker",
                    router_decision=router_decision.tier if router_decision else None,
                    retries=retries,
                    role=req.auto_route,
                    tier=tier,
                    escalations=escalations,
                )
                span.set_usage(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read_tokens=usage.cache_read_tokens,
                    cache_write_tokens=usage.cache_write_tokens,
                    response_model=result["model"],
                    finish_reason=result["stop_reason"],
                )
                span.set_cost(record.cost_breakdown)
                span.set(_otel.GLC_LEDGER_ROW_ID, record.row_id)
                span.set_content(completion=result["text"])

                # ── cascade: was this answer good enough to keep? ───────────
                confidence = None
                if pol is not None:
                    assessed = pol.assess({**result, "schema_failed": schema_failed})
                    confidence = assessed.score
                    span.set(_otel.GLC_ROUTING_CONFIDENCE, round(assessed.score, 4))
                    do_escalate, why = pol.should_escalate(req.auto_route, tier or "", assessed, escalations)
                    if do_escalate and req.escalate is not False and tier:
                        nxt = pol.next_tier(tier)
                        span.set("glc.routing.escalation_reason", why)
                        span_cm.__exit__(None, None, None)
                        all_attempts.append({"provider": name, "reason": f"escalated: {why}"})
                        escalations += 1
                        tier = nxt
                        tier_path.append(tier)
                        candidates, extra = _tier_candidates(
                            pol, tier, rtr, est_input, tradeoff=req.cost_quality_tradeoff
                        )
                        all_attempts.extend(extra)
                        if router_decision is not None:
                            router_decision.tier = tier
                            router_decision.escalations = escalations
                            router_decision.tier_path = list(tier_path)
                        escalated = True
                        break  # leave the inner ring, re-enter with the new tier

                if router_decision is not None:
                    router_decision.escalations = escalations
                    router_decision.tier_path = list(tier_path)
                    router_decision.confidence = confidence

                response = ChatResponse(
                    provider=name,
                    model=result["model"],
                    text=result["text"],
                    tool_calls=[ToolCall(**tc) for tc in result["tool_calls"]],
                    stop_reason=result["stop_reason"],
                    input_tokens=result["input_tokens"],
                    output_tokens=result["output_tokens"],
                    cache_creation_input_tokens=result["cache_creation_input_tokens"],
                    cache_read_input_tokens=result["cache_read_input_tokens"],
                    latency_ms=latency,
                    tool_call_dialect=result["tool_call_dialect"],
                    reasoning_applied=result["reasoning_applied"],
                    reasoning_text=result.get("reasoning_text"),
                    parsed=parsed,
                    attempted=all_attempts,
                    router_decision=router_decision,
                    retries=retries,
                    cost=record.cost_breakdown,
                    budget=budget_info,
                    cache=cache_info,
                    trace=_trace_ids(span),
                ).model_dump()
                span_cm.__exit__(None, None, None)

                if scache is not None and cache_info and not cache_info.get("skipped_reason"):
                    await scache.store(
                        prompt_text,
                        cache_fields,
                        response,
                        provider=name,
                        model=result["model"],
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        usd=record.usd,
                        principal=principal,
                    )
                return response
            except P.ProviderError as e:
                last_err = str(e)
                span.set_error(str(e)[:300])
                span_cm.__exit__(type(e), e, e.__traceback__)
                secs, reason = _backoff_for(e, has_model_override=bool(req.model))
                if secs > 0:
                    rtr.state[name].mark_unavailable(secs, reason)
                meter.record(
                    provider=name,
                    model=req.model or provider.model,
                    principal=principal,
                    usage=_meter.Usage(),
                    status="error",
                    error=str(e)[:500],
                    latency_ms=int((time.time() - t0) * 1000),
                    prompt_chars=len(prompt_text),
                    override=req.provider,
                    attempted=_attempts_str(all_attempts),
                    retries=retries,
                    role=req.auto_route,
                    tier=tier,
                )
                tag = f"failed: {str(e)[:100]}"
                if secs > 0:
                    tag += f" → backoff {secs:.0f}s ({reason})"
                all_attempts.append({"provider": name, "reason": tag})
                if explicit_override or not getattr(e, "retryable", True):
                    raise HTTPException(502, f"{name} failed: {e}")
                candidates = [c for c in candidates if c != name]
                continue
            except HTTPException:
                span_cm.__exit__(None, None, None)
                raise
            except Exception as e:
                last_err = str(e)
                span.set_error(str(e)[:300])
                span_cm.__exit__(type(e), e, e.__traceback__)
                secs, reason = _backoff_for(e, has_model_override=bool(req.model))
                if secs > 0:
                    rtr.state[name].mark_unavailable(secs, reason)
                meter.record(
                    provider=name,
                    model=req.model or provider.model,
                    principal=principal,
                    usage=_meter.Usage(),
                    status="error",
                    error=str(e)[:500],
                    latency_ms=int((time.time() - t0) * 1000),
                    prompt_chars=len(prompt_text),
                    override=req.provider,
                    attempted=_attempts_str(all_attempts),
                    retries=retries,
                    role=req.auto_route,
                    tier=tier,
                )
                all_attempts.append({"provider": name, "reason": f"exception: {str(e)[:120]}"})
                if explicit_override:
                    raise HTTPException(502, f"{name} failed: {e}")
                candidates = [c for c in candidates if c != name]
                continue
        if not (escalated and candidates):
            break

    # Every candidate was refused on cost, and none was even attempted. That is
    # a budget answer, not an availability answer, so it gets the 402 envelope.
    if last_refusal is not None and last_err is None:
        raise _budget_refusal(last_refusal)

    raise HTTPException(503, f"all providers unavailable. attempts: {all_attempts}. last_error: {last_err}")


def _may_escalate(pol, req: ChatRequest, tier: str | None, escalations: int, schema_failed: bool) -> bool:
    """Cheap pre-check used inside the schema-repair path."""
    if req.escalate is False or not tier:
        return False
    assessed = _policy.ConfidenceAssessment(0.1, [_policy.TRIGGER_SCHEMA] if schema_failed else [])
    ok, _ = pol.should_escalate(req.auto_route, tier, assessed, escalations)
    return ok


@router.post("/v1/chat/batch")
async def chat_batch(req: BatchChatRequest, request: Request):
    sem = _asyncio.Semaphore(max(1, req.max_concurrency))

    async def _one(call: ChatRequest):
        async with sem:
            try:
                return await chat(call, request)
            except HTTPException as he:
                return {"error": str(he.detail), "status_code": he.status_code}
            except Exception as e:
                return {"error": str(e)[:400], "status_code": 500}

    results = await _asyncio.gather(*[_one(c) for c in req.calls])
    return {"results": results}


@router.post("/v1/vision")
async def vision(req: VisionRequest, request: Request):
    content: list[dict[str, Any]] = [{"type": "text", "text": req.prompt}]
    content.append({"type": "image_url", "image_url": {"url": req.image}})
    inner = ChatRequest(
        messages=[{"role": "user", "content": content}],
        system=req.system,
        provider=req.provider,
        model=req.model,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        response_format=(
            ResponseFormat(type="json_schema", schema=req.schema_, name=req.schema_name, strict=True)
            if req.schema_
            else None
        ),
        agent=req.agent,
        session=req.session,
    )
    return await chat(inner, request)


@router.post("/v1/embed")
async def embed(req: EmbedRequest, request: Request):
    from glc import embedders as E

    state = request.app.state
    embedders = state.embedders
    if not embedders:
        raise HTTPException(503, "no embedding providers configured")
    if len(req.text) > E.MAX_INPUT_CHARS:
        raise HTTPException(
            413,
            f"text is {len(req.text)} chars; embed input is capped at "
            f"{E.MAX_INPUT_CHARS} chars (~{E.MAX_INPUT_CHARS // 4} tokens). Chunk and re-embed.",
        )
    t0 = time.time()
    try:
        name, result, attempts, latency = await E.embed_with_failover(
            embedders,
            req.text,
            req.task_type,
            explicit=req.provider,
        )
    except E.EmbedderError as e:
        latency = int((time.time() - t0) * 1000)
        db.log_call(
            provider=req.provider or "(any)",
            model="(none)",
            status="error",
            error=str(e)[:500],
            latency_ms=latency,
            prompt_chars=len(req.text),
            override=req.provider,
            call_role="embed",
        )
        if req.provider:
            if e.status == 429:
                raise HTTPException(429, f"{req.provider} rate-limited: {e}")
            if e.status == 400:
                raise HTTPException(400, str(e))
            raise HTTPException(502, f"{req.provider} embed failed: {e}")
        raise HTTPException(503, str(e))

    db.log_call(
        provider=name,
        model=result["model"],
        status="ok",
        latency_ms=latency,
        prompt_chars=len(req.text),
        override=req.provider,
        attempted=_attempts_str(attempts),
        call_role="embed",
        embed_dim=result["dim"],
    )
    return EmbedResponse(
        provider=name,
        model=result["model"],
        embedding=result["embedding"],
        dim=result["dim"],
        latency_ms=latency,
        attempted=attempts,
    ).model_dump()


@router.get("/v1/embedders")
async def list_embedders(request: Request):
    from glc import embedders as E

    state = request.app.state
    return {
        "order": state.embed_order,
        "models": {e.name: e.model for e in state.embedders},
        "fixed_dim": E.EMBED_DIM,
        "max_input_chars": E.MAX_INPUT_CHARS,
        "backoff_steps_s": E.BACKOFF_STEPS,
        "live": {e.name: e.state.snapshot() for e in state.embedders},
        "today": db.aggregate(call_role="embed"),
    }


@router.get("/v1/cost/by_agent")
async def cost_by_agent(session: str | None = None, agent: str | None = None):
    from glc import pricing as _pricing

    raw = db.by_agent(session=session)
    if agent:
        raw = {agent: raw.get(agent, [])}
    out: dict[str, list[dict]] = {}
    for ag, rows in raw.items():
        out[ag] = []
        for r in rows:
            r2 = dict(r)
            r2["dollars"] = _pricing.estimate_usd(r["provider"], r.get("in_tok") or 0, r.get("out_tok") or 0)
            out[ag].append(r2)
    return out


@router.get("/v1/providers")
async def list_providers(request: Request):
    r = request.app.state.router
    return {
        "order": r.order,
        "providers": list(r.providers.keys()),
        "shortcuts": SHORTCUTS,
        "limits": LIMITS,
        "models": {n: p.model for n, p in r.providers.items()},
    }


@router.get("/v1/capabilities")
async def capabilities(request: Request):
    r = request.app.state.router
    out = {}
    for name, p in r.providers.items():
        caps = dict(getattr(p, "capabilities", {}))
        caps = P.model_capabilities(name, p.model, caps)
        caps["model"] = p.model
        caps.update(
            {
                "max_ctx": LIMITS[name]["max_ctx"],
                "rpm": LIMITS[name]["rpm"],
                "rpd": LIMITS[name]["rpd"],
            }
        )
        out[name] = caps
    return out


@router.get("/v1/status")
async def status(request: Request):
    r = request.app.state.router
    return {
        "order": r.order,
        "live": r.all_status(),
        "today": db.aggregate(call_role="worker"),
        "limits": LIMITS,
    }


@router.get("/v1/routers")
async def routers(request: Request):
    rp = request.app.state.router_pool
    pol = _routing_policy()
    out = {
        "order": rp.order,
        "providers": list(rp.providers.keys()),
        "models": {n: p.model for n, p in rp.providers.items()},
        "live": rp.all_status(),
        "today": db.aggregate(call_role="router"),
        "limits": {k: LIMITS[k] for k in rp.providers},
        # v3 key, unchanged shape. routing.yaml's rings supersede it when the
        # policy loads; the fallback dict is reported when it does not.
        "tier_to_order": pol.tier_to_order() if pol else TIER_TO_ORDER,
    }
    if pol is not None:
        out["routing_policy"] = pol.describe()
    return out


@router.get("/v1/calls")
async def calls(limit: int = 100, provider: str | None = None, status: str | None = None):
    return db.recent(limit=limit, provider=provider, status=status)
