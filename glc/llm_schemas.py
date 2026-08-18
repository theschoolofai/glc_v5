"""Pydantic v2 request/response models for llm_gatewayV9."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolDef(BaseModel):
    """Canonical tool definition. Schema is JSON-Schema (typically from Pydantic)."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Optional opaque per-provider metadata (e.g. Gemini thoughtSignature)
    # that must be echoed back when sending the assistant turn.
    provider_meta: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")


class CacheableSystemBlock(BaseModel):
    text: str
    cache: bool = False


class ResponseFormat(BaseModel):
    type: Literal["json_schema", "json_object"] = "json_schema"
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    name: str = "out"
    strict: bool = True

    model_config = ConfigDict(populate_by_name=True)


class ChatRequest(BaseModel):
    """Backward-compatible request — every new field is optional."""

    messages: list[dict[str, Any]] | None = None
    prompt: str | None = None
    system: str | list[CacheableSystemBlock] | None = None
    provider: str | None = None
    model: str | None = None
    max_tokens: int = 2048
    temperature: float = 0.7
    stream: bool = False

    # New in V2:
    tools: list[ToolDef] | None = None
    tool_choice: str | dict[str, Any] | None = None  # "auto" | "none" | {name}
    cache_system: bool | None = None
    reasoning: Literal["off", "low", "medium", "high"] | None = None
    response_format: ResponseFormat | None = None

    # New in V3: when set, the gateway runs a router LLM first to pick a worker tier.
    # Role labels track which cognitive layer is asking. The worker is picked
    # from a tier-to-order table; router never sees system, tools, schemas.
    #
    # V4: widened from Literal["perception","memory","decision"] to a free-form
    # string, because the roles are declared in routing.yaml now and adding one
    # must not require editing this file. The three V3 values still work
    # unchanged; an unrecognised role resolves to routing.yaml's default_role
    # rather than 422-ing, so a typo degrades to sane routing.
    auto_route: str | None = None

    # New in V8: agent tag (which skill is calling) and session tag (which
    # flow-run). Used for cost-by-agent rollups and provider pinning via
    # agent_routing.yaml. Both are free-form strings; the gateway logs them
    # but does not validate them against any whitelist.
    agent: str | None = None
    session: str | None = None

    # ── New in V4 ──────────────────────────────────────────────────────────
    # The three attribution dimensions above agent+session. Same contract:
    # free-form, opaque, never validated against a whitelist, never used to
    # branch on a particular value. They exist so a bill can be split by who
    # incurred it, and so a budget can be hung on any of the five levels.
    tenant: str | None = None
    project: str | None = None
    user: str | None = None

    # Per-request semantic-cache opt-in. None means "use cache.yaml's
    # default_on"; True consults the response cache and may skip the provider
    # call entirely; False guarantees a fresh call.
    semantic_cache: bool | None = None

    # Submit at the provider's batch/async tier when it has one. Affects
    # pricing (typically -50%) and therefore budget projection.
    batch: bool = False

    # Cost/quality dial for this call, 0.0 = cheapest, 1.0 = best quality.
    # None uses routing.yaml's `selection.tradeoff`. Named after the industry
    # convention (OpenRouter's cost_quality_tradeoff) so it reads the same way.
    cost_quality_tradeoff: float | None = None

    # Opt out of cascade escalation for this call even when routing.yaml
    # enables it. None = follow config.
    escalate: bool | None = None


class RouterDecision(BaseModel):
    """What the router agent decided. Echoed back on the worker response so the
    agentic-world caller can see which model was picked and why.

    V4 widens `role` and `tier` from Literals to strings (roles and tiers are
    declared in routing.yaml) and adds the policy fields: what the classifier
    said, what the role's floor/ceiling turned it into, and what the cascade
    did. The V3 fields are untouched and still populated the same way.
    """

    role: str
    tier: str
    estimated_tokens: int
    router_provider: str
    router_model: str
    router_latency_ms: int
    chosen_worker_provider: str | None = None
    chosen_worker_model: str | None = None
    fallback_used: bool = False  # true if router LLM failed and tier was decided by token-count rule

    # ── V4 policy fields ───────────────────────────────────────────────────
    #: the tier the classifier produced, before the role policy was applied
    classified_tier: str | None = None
    #: true when the role's min_tier/max_tier overrode the classifier
    policy_clamped: bool = False
    #: human-readable "why this tier"
    policy_reason: str = ""
    #: how many cascade escalations fired
    escalations: int = 0
    #: the tier ladder actually walked, e.g. ["TINY", "LARGE"]
    tier_path: list[str] = Field(default_factory=list)
    #: structural confidence of the accepted answer, when a cascade ran
    confidence: float | None = None


class EmbedRequest(BaseModel):
    """Request for POST /v1/embed. The model is fixed per deployment (see
    README); only the text, task type, and an optional explicit provider
    are caller-controlled."""

    text: str
    task_type: Literal["retrieval_document", "retrieval_query"] = "retrieval_document"
    provider: str | None = None  # "ollama" | configured fallback name


class EmbedResponse(BaseModel):
    provider: str
    model: str
    embedding: list[float]
    dim: int
    latency_ms: int = 0
    attempted: list[dict[str, Any]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    provider: str
    model: str
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: Literal["tool_use", "end_turn", "max_tokens", "error"] = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    latency_ms: int = 0
    tool_call_dialect: Literal["native", "prompted_fallback", "none"] = "none"
    reasoning_applied: bool = False
    # New in V9: a provider's own reasoning channel, when it separates one out
    # (Ollama's message.thinking, an OpenAI-compat message.reasoning). These
    # tokens are generated and billed regardless of whether anyone reads this
    # field; capturing them costs nothing further. None when the provider did
    # not return one, or did not think.
    reasoning_text: str | None = None
    parsed: dict[str, Any] | None = None  # set when response_format used
    attempted: list[dict[str, Any]] = Field(default_factory=list)
    # New in V3: present only when auto_route was used
    router_decision: RouterDecision | None = None
    # New in V8: how many automatic retries fired before success (or final fail).
    retries: int = 0

    # ── New in V4. All additive; a V3 client ignores them. ─────────────────
    #: What this call cost, per-model, split by token class.
    cost: dict[str, Any] | None = None
    #: Budget verdict: projected cost, ceilings checked, remaining allowance.
    #: Present whenever a budget policy governed the principal.
    budget: dict[str, Any] | None = None
    #: Semantic-cache outcome. `hit: true` means no provider was contacted.
    cache: dict[str, Any] | None = None
    #: OTel trace/span ids, so a caller can jump straight to the waterfall.
    trace: dict[str, Any] | None = None


class BatchChatRequest(BaseModel):
    """V8 batch endpoint. The gateway dispatches the inner calls with
    bounded parallelism so providers' rate limits are respected centrally."""

    calls: list[ChatRequest]
    max_concurrency: int = 4


class VisionRequest(BaseModel):
    """V9: typed shim for single-image vision calls. Lower-ceremony than
    /v1/chat for the set-of-marks loop — callers send one image, one prompt,
    and (optionally) a JSON schema for typed output, and the gateway forces
    routing to a vision-capable provider.

    Accepts either a data: URL (base64) or an http(s) URL for `image`.
    The gateway pre-resolves http URLs the same way /v1/chat does.
    """

    image: str = Field(description="data: URL or http(s) URL of the image")
    prompt: str
    system: str | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    schema_name: str = "out"
    model: str | None = None
    provider: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.0
    agent: str | None = None
    session: str | None = None
    # V4: same attribution dimensions as ChatRequest, forwarded verbatim.
    tenant: str | None = None
    project: str | None = None
    user: str | None = None

    model_config = ConfigDict(populate_by_name=True)


class BudgetSetRequest(BaseModel):
    """POST /v1/budget — arm or move a ceiling at runtime.

    `principal` is "<dimension>:<value-or-glob>" over the five attribution
    dimensions (tenant, project, user, agent, session). A runtime entry shadows
    a same-named entry in budgets.yaml, so an operator can clamp a runaway
    session without a deploy, and `limit_usd: 0` refuses everything for that
    principal immediately.
    """

    principal: str = Field(description='e.g. "session:run-42" or "tenant:*"')
    limit_usd: float = Field(ge=0)
    period: Literal["minute", "hour", "day", "month", "lifetime"] | None = None
    include_errors: bool | None = None
