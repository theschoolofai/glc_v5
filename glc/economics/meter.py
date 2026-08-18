"""Attribution: who spent what.

v3's ledger answered "which agent, which session". That is two of the five
dimensions a FinOps report actually needs; the other three — tenant, project,
user — did not exist, so a multi-tenant deployment could report its total bill
and nothing else.

This module is the single write path for a metered call. It:

1. prices the call through `economics.pricing` (per model, not per provider),
2. writes one ledger row carrying all five principal dimensions,
3. hands the same numbers to the OTel span, so the trace and the ledger can
   never disagree about what a call cost.

The invariant it exists to enforce is *no unmetered spend*: a provider call's
result is not returned to the caller until `record()` has written its row.

Nothing here knows anything about what the agent was doing. A `Principal` is
five opaque strings supplied by the caller; the gateway never invents,
validates against, or special-cases any particular value.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from glc import db
from glc.economics import pricing as _pricing

#: The attribution dimensions, coarsest to finest. Structural (these are
#: ledger columns); the values in them are entirely caller-supplied.
DIMENSIONS: tuple[str, ...] = db.PRINCIPAL_DIMENSIONS


@dataclass(frozen=True)
class Principal:
    """The five-part identity a call is billed to. Every part optional."""

    tenant: str | None = None
    project: str | None = None
    user: str | None = None
    agent: str | None = None
    session: str | None = None

    @classmethod
    def from_request(cls, req) -> Principal:
        """Pull whichever dimensions a request object happens to carry."""
        return cls(**{d: getattr(req, d, None) for d in DIMENSIONS})

    @classmethod
    def parse(cls, spec: str) -> Principal:
        """`"agent:researcher"` -> Principal(agent="researcher")."""
        dim, _, value = spec.partition(":")
        dim = dim.strip()
        if dim not in DIMENSIONS:
            raise ValueError(f"unknown principal dimension {dim!r}; expected one of {list(DIMENSIONS)}")
        if not value:
            raise ValueError(f"principal {spec!r} has no value after the colon")
        return cls(**{dim: value})

    def as_dict(self) -> dict[str, str | None]:
        return {d: getattr(self, d) for d in DIMENSIONS}

    def present(self) -> dict[str, str]:
        """Only the dimensions that were actually supplied."""
        return {d: v for d, v in self.as_dict().items() if v}

    def keys(self) -> list[str]:
        """Canonical `dimension:value` strings for the dimensions present."""
        return [f"{d}:{v}" for d, v in self.present().items()]

    def with_defaults(self, **kw) -> Principal:
        """Fill only the empty dimensions (used for a configured default tenant)."""
        patch = {d: v for d, v in kw.items() if d in DIMENSIONS and v and not getattr(self, d)}
        return replace(self, **patch) if patch else self

    def __bool__(self) -> bool:
        return bool(self.present())


@dataclass
class Usage:
    """Token counts as the provider reported them."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens

    @classmethod
    def from_provider_result(cls, result: dict) -> Usage:
        return cls(
            input_tokens=int(result.get("input_tokens") or 0),
            output_tokens=int(result.get("output_tokens") or 0),
            cache_read_tokens=int(result.get("cache_read_input_tokens") or 0),
            cache_write_tokens=int(result.get("cache_creation_input_tokens") or 0),
        )


@dataclass
class MeterRecord:
    """What was written. Returned so the span and the response can echo it."""

    row_id: int
    provider: str
    model: str
    usd: float
    price_source: str
    usage: Usage
    principal: Principal
    cost_breakdown: dict = field(default_factory=dict)
    tokens_saved: int = 0
    usd_saved: float = 0.0

    def as_dict(self) -> dict:
        return {
            "row_id": self.row_id,
            "provider": self.provider,
            "model": self.model,
            "usd": round(self.usd, 8),
            "price_source": self.price_source,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "cache_read_tokens": self.usage.cache_read_tokens,
            "cache_write_tokens": self.usage.cache_write_tokens,
            "tokens_saved": self.tokens_saved,
            "usd_saved": round(self.usd_saved, 8),
            "principal": self.principal.present(),
            "cost": self.cost_breakdown,
        }


class Meter:
    """Writes priced ledger rows. Stateless apart from the pricing table."""

    def __init__(self, table: _pricing.PricingTable | None = None):
        self._table = table

    @property
    def table(self) -> _pricing.PricingTable:
        return self._table or _pricing.load_pricing()

    def price(
        self,
        provider: str,
        model: str | None,
        usage: Usage,
        batch: bool = False,
    ) -> _pricing.CostBreakdown:
        return self.table.cost(
            provider,
            model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            batch=batch,
        )

    def record(
        self,
        provider: str,
        model: str | None,
        principal: Principal | None = None,
        usage: Usage | None = None,
        batch: bool = False,
        tokens_saved: int = 0,
        usd_saved: float = 0.0,
        reservation_id: int | None = None,
        **ledger_fields,
    ) -> MeterRecord:
        """Price the call and append exactly one ledger row.

        `ledger_fields` is forwarded verbatim to `db.log_call`/`db.settle_call`,
        so every v3 field (latency_ms, status, error, attempted,
        router_decision, …) is available without this signature having to
        enumerate them.

        `reservation_id`, when given, is the hold `BudgetController.admit()`
        reserved for this call: the row is written as a settlement and the
        hold is released, atomically, instead of just being inserted — so
        the hold's worst-case figure stops counting once the real cost is
        known, without ever overwriting it.
        """
        principal = principal or Principal()
        usage = usage or Usage()
        cost = self.price(provider, model, usage, batch=batch)
        write = db.settle_call if reservation_id is not None else db.log_call
        write_kwargs = {"hold_id": reservation_id} if reservation_id is not None else {}
        row_id = write(
            provider=provider,
            model=model or "",
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_create_tokens=usage.cache_write_tokens,
            usd=cost.total_usd,
            price_source=cost.price_source,
            tokens_saved=tokens_saved,
            usd_saved=usd_saved,
            **principal.as_dict(),
            **ledger_fields,
            **write_kwargs,
        )
        return MeterRecord(
            row_id=row_id,
            provider=provider,
            model=model or "",
            usd=cost.total_usd,
            price_source=cost.price_source,
            usage=usage,
            principal=principal,
            cost_breakdown=cost.as_dict(),
            tokens_saved=tokens_saved,
            usd_saved=usd_saved,
        )

    # ── reporting ────────────────────────────────────────────────────────────

    def rollup(
        self,
        dimension: str = "agent",
        since: float | None = None,
        value: str | None = None,
        group_by_provider: bool = True,
    ) -> list[dict]:
        return db.by_principal(
            dimension=dimension, since=since, value=value, group_by_provider=group_by_provider
        )

    def rollup_all(self, since: float | None = None) -> dict[str, list[dict]]:
        """Every dimension at once — the `/v1/cost/by_principal` payload."""
        return {d: self.rollup(d, since=since) for d in DIMENSIONS}

    def spend(self, dimension: str, value: str, since: float = 0.0) -> float:
        return db.spend_usd(dimension, value, since=since)


_meter: Meter | None = None


def get_meter() -> Meter:
    global _meter
    if _meter is None:
        _meter = Meter()
    return _meter


def reset_meter() -> None:
    """Drop the singleton so a reloaded pricing table is picked up."""
    global _meter
    _meter = None
