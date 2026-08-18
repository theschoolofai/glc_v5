"""The hard controller.

glc_v3 had throughput caps (RPM, RPD, TPM) and no spend cap at all: a runaway
loop on a paid model was rate-limited into taking a while and then billed in
full. The 2026 incident reports put a number on the gap — >$47,000 over one
weekend from a single looping agent — which is why denial-of-wallet is filed
under security here and not under cost reporting.

Two properties make this a controller rather than a suggestion:

**It runs before the provider call.** `admit()` is checked against a *projected
worst-case* cost. Deciding after the fact only tells you how much you have
already lost.

**It lives in code, not in the prompt.** TALE (arXiv 2026) measured what
happens when you state a budget in the context — "token elasticity", models
overshooting tight budgets — and CostBench found agents consistently exceed
budgets and adapt badly mid-task. So a budget the model can reason about is a
budget the model can reason its way out of.

On breach the call is **refused** with a structured 402-style envelope carrying
the limit, the spend, the projection and the shortfall. It is never silently
truncated, downgraded, or dropped: the caller is the only party that knows
whether a cheaper answer is an acceptable answer.

Limits come from `budgets.yaml` and from a runtime override table written by
`POST /v1/budget`. Spend comes from the `calls` ledger. There is no third
place, so there is nothing to keep in sync.
"""

from __future__ import annotations

import fnmatch
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from glc import config as _config
from glc import db
from glc.economics.meter import DIMENSIONS, Principal

VALID_PERIODS = ("minute", "hour", "day", "month", "lifetime")

#: Machine-readable refusal code. Callers switch on this, not on the message.
BUDGET_EXCEEDED = "budget_exceeded"


class BudgetConfigError(ValueError):
    """budgets.yaml is malformed. Raised loudly — a typo must not read as
    "no budget"."""


@dataclass(frozen=True)
class BudgetPolicy:
    """One ceiling on one principal pattern."""

    principal: str  # "<dimension>:<glob>"
    dimension: str
    pattern: str
    limit_usd: float
    period: str = "day"
    include_errors: bool = True
    source: str = "budgets.yaml"

    @property
    def is_glob(self) -> bool:
        return any(ch in self.pattern for ch in "*?[")

    def matches(self, dimension: str, value: str) -> bool:
        return dimension == self.dimension and fnmatch.fnmatch(value, self.pattern)

    def as_dict(self) -> dict:
        return {
            "principal": self.principal,
            "dimension": self.dimension,
            "pattern": self.pattern,
            "limit_usd": self.limit_usd,
            "period": self.period,
            "include_errors": self.include_errors,
            "source": self.source,
	    "is_glob": self.is_glob,
        }


@dataclass
class BudgetStatus:
    """Where one concrete principal stands against one policy."""

    principal: str
    dimension: str
    value: str
    limit_usd: float
    spent_usd: float
    period: str
    period_start: float
    source: str = "budgets.yaml"

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.limit_usd - self.spent_usd)

    @property
    def fraction_used(self) -> float:
        return 1.0 if self.limit_usd <= 0 else min(1.0, self.spent_usd / self.limit_usd)

    def as_dict(self) -> dict:
        return {
            "principal": self.principal,
            "dimension": self.dimension,
            "value": self.value,
            "limit_usd": round(self.limit_usd, 8),
            "spent_usd": round(self.spent_usd, 8),
            "remaining_usd": round(self.remaining_usd, 8),
            "fraction_used": round(self.fraction_used, 6),
            "period": self.period,
            "period_start": self.period_start,
            "source": self.source,
        }


@dataclass
class Admission:
    """The verdict. `allowed` is the whole answer; the rest is the receipt."""

    allowed: bool
    projected_usd: float
    checked: list[BudgetStatus] = field(default_factory=list)
    breached: BudgetStatus | None = None
    warnings: list[BudgetStatus] = field(default_factory=list)
    reason: str = ""
    enforced: bool = True

    @property
    def shortfall_usd(self) -> float:
        if not self.breached:
            return 0.0
        return max(0.0, self.projected_usd - self.breached.remaining_usd)

    def envelope(self) -> dict:
        """The 402 body. Structured, numeric, and actionable — a caller can
        read this and decide to shrink max_tokens, pick a cheaper tier, raise
        the ceiling, or stop."""
        b = self.breached
        return {
            "error": BUDGET_EXCEEDED,
            "code": BUDGET_EXCEEDED,
            "message": self.reason,
            "principal": b.principal if b else None,
            "dimension": b.dimension if b else None,
            "period": b.period if b else None,
            "period_start": b.period_start if b else None,
            "limit_usd": round(b.limit_usd, 8) if b else None,
            "spent_usd": round(b.spent_usd, 8) if b else None,
            "remaining_usd": round(b.remaining_usd, 8) if b else None,
            "projected_usd": round(self.projected_usd, 8),
            "shortfall_usd": round(self.shortfall_usd, 8),
            "checked": [s.as_dict() for s in self.checked],
            "hint": (
                "The call was refused before any provider was contacted, so nothing was "
                "billed. Lower max_tokens, route to a cheaper tier, or raise the ceiling "
                "with POST /v1/budget."
            ),
        }

    def as_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "enforced": self.enforced,
            "projected_usd": round(self.projected_usd, 8),
            "reason": self.reason,
            "checked": [s.as_dict() for s in self.checked],
            "warnings": [s.as_dict() for s in self.warnings],
            "breached": self.breached.as_dict() if self.breached else None,
        }


# ── runtime override store ──────────────────────────────────────────────────

_OVERRIDE_DDL = """CREATE TABLE IF NOT EXISTS budget_limits (
    principal TEXT PRIMARY KEY,
    dimension TEXT NOT NULL,
    pattern   TEXT NOT NULL,
    limit_usd REAL NOT NULL,
    period    TEXT NOT NULL DEFAULT 'day',
    include_errors INTEGER NOT NULL DEFAULT 1,
    updated_at REAL NOT NULL
)"""


def init_store() -> None:
    """Create the runtime-override table. Idempotent."""
    with db.conn() as c:
        c.execute(_OVERRIDE_DDL)


def _parse_principal(spec: str) -> tuple[str, str]:
    dim, _, pattern = str(spec).partition(":")
    dim = dim.strip()
    pattern = pattern.strip()
    if dim not in DIMENSIONS:
        raise BudgetConfigError(
            f"principal {spec!r}: unknown dimension {dim!r}; expected one of {list(DIMENSIONS)}"
        )
    if not pattern:
        raise BudgetConfigError(f"principal {spec!r}: nothing after the colon")
    return dim, pattern


def _policy_from_entry(entry: dict, defaults: dict, source: str) -> BudgetPolicy:
    if not isinstance(entry, dict):
        raise BudgetConfigError(f"budget policy {entry!r} must be a mapping")
    spec = entry.get("principal")
    if not spec:
        raise BudgetConfigError(f"budget policy {entry!r} is missing `principal`")
    if "limit_usd" not in entry:
        raise BudgetConfigError(f"budget policy {spec!r} is missing `limit_usd`")
    try:
        limit = float(entry["limit_usd"])
    except (TypeError, ValueError) as e:
        raise BudgetConfigError(
            f"budget policy {spec!r}: limit_usd={entry['limit_usd']!r} is not a number"
        ) from e
    if limit < 0:
        raise BudgetConfigError(f"budget policy {spec!r}: limit_usd must be >= 0")
    period = str(entry.get("period", defaults.get("period", "day")))
    if period not in VALID_PERIODS:
        raise BudgetConfigError(f"budget policy {spec!r}: period {period!r} not in {list(VALID_PERIODS)}")
    dim, pattern = _parse_principal(spec)
    return BudgetPolicy(
        principal=f"{dim}:{pattern}",
        dimension=dim,
        pattern=pattern,
        limit_usd=limit,
        period=period,
        include_errors=bool(entry.get("include_errors", defaults.get("include_errors", True))),
        source=source,
    )


class BudgetController:
    """Loads policies, projects cost, admits or refuses."""

    def __init__(self, path: Path | str | None = None, load_overrides: bool = True):
        self.path = Path(path) if path else _config.budgets_yaml_path()
        self._lock = threading.Lock()
        self.defaults: dict = {}
        self.projection: dict = {}
        self.alerts: dict = {}
        self.file_policies: list[BudgetPolicy] = []
        self.load_overrides = load_overrides
        self.reload()

    # ── config ───────────────────────────────────────────────────────────────

    def reload(self) -> None:
        data = _config.load_yaml(self.path)
        defaults = data.get("defaults") or {}
        projection = data.get("projection") or {}
        # Check the type before the `or []` fallback: a `policies: {}` typo must
        # be an error, not a silent "no budget".
        entries = data.get("policies")
        if entries is not None and not isinstance(entries, list):
            raise BudgetConfigError(
                f"budgets.yaml: `policies` must be a list of mappings, got {type(entries).__name__}"
            )
        entries = entries or []
        policies = [_policy_from_entry(e, defaults, str(self.path)) for e in entries]
        with self._lock:
            self.defaults = defaults
            self.projection = projection
            self.alerts = data.get("alerts") or {}
            self.file_policies = policies

    @property
    def enabled(self) -> bool:
        return bool(self.defaults.get("enabled", True))

    def _overrides(self) -> list[BudgetPolicy]:
        if not self.load_overrides:
            return []
        try:
            with db.conn() as c:
                c.execute(_OVERRIDE_DDL)
                rows = c.execute("SELECT * FROM budget_limits").fetchall()
        except sqlite3.Error:  # pragma: no cover - DB unavailable
            return []
        return [
            BudgetPolicy(
                principal=r["principal"],
                dimension=r["dimension"],
                pattern=r["pattern"],
                limit_usd=float(r["limit_usd"]),
                period=r["period"],
                include_errors=bool(r["include_errors"]),
                source="runtime",
            )
            for r in rows
        ]

    def policies(self) -> list[BudgetPolicy]:
        """File policies plus runtime overrides. A runtime entry with the same
        principal string replaces the file entry — that is what makes
        `POST /v1/budget` authoritative without rewriting anyone's YAML."""
        with self._lock:
            merged = {p.principal: p for p in self.file_policies}
        for p in self._overrides():
            merged[p.principal] = p
        return list(merged.values())

    # ── the runtime write path (POST /v1/budget) ──────────────────────────────

    def set_limit(
        self,
        principal: str,
        limit_usd: float,
        period: str | None = None,
        include_errors: bool | None = None,
    ) -> BudgetPolicy:
        entry = {"principal": principal, "limit_usd": limit_usd}
        if period is not None:
            entry["period"] = period
        if include_errors is not None:
            entry["include_errors"] = include_errors
        pol = _policy_from_entry(entry, self.defaults, "runtime")
        with db.conn() as c:
            c.execute(_OVERRIDE_DDL)
            c.execute(
                """INSERT INTO budget_limits
                       (principal, dimension, pattern, limit_usd, period, include_errors, updated_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(principal) DO UPDATE SET
                       limit_usd=excluded.limit_usd,
                       period=excluded.period,
                       include_errors=excluded.include_errors,
                       updated_at=excluded.updated_at""",
                (
                    pol.principal,
                    pol.dimension,
                    pol.pattern,
                    pol.limit_usd,
                    pol.period,
                    1 if pol.include_errors else 0,
                    time.time(),
                ),
            )
        return pol

    def clear_limit(self, principal: str) -> bool:
        dim, pattern = _parse_principal(principal)
        with db.conn() as c:
            c.execute(_OVERRIDE_DDL)
            cur = c.execute("DELETE FROM budget_limits WHERE principal=?", (f"{dim}:{pattern}",))
            return cur.rowcount > 0

    # ── matching ─────────────────────────────────────────────────────────────

    @staticmethod
    def _specificity(pol: BudgetPolicy) -> tuple[int, int]:
        """Exact value beats a glob; a longer glob beats a shorter one."""
        return (0 if pol.is_glob else 1, len(pol.pattern))

    def policy_for(self, dimension: str, value: str) -> BudgetPolicy | None:
        """The single governing policy for one concrete principal."""
        matches = [p for p in self.policies() if p.matches(dimension, value)]
        if not matches:
            return None
        return max(matches, key=self._specificity)

    def status_for(self, dimension: str, value: str) -> BudgetStatus | None:
        pol = self.policy_for(dimension, value)
        if pol is None:
            return None
        return self._status(pol, value)

    def _status(self, pol: BudgetPolicy, value: str) -> BudgetStatus:
        start = db._period_start(pol.period)
        spent = db.spend_usd(pol.dimension, value, since=start, include_errors=pol.include_errors)
        return BudgetStatus(
            principal=f"{pol.dimension}:{value}",
            dimension=pol.dimension,
            value=value,
            limit_usd=pol.limit_usd,
            spent_usd=spent,
            period=pol.period,
            period_start=start,
            source=pol.source,
        )

    def statuses(self, principal: Principal) -> list[BudgetStatus]:
        """One status per governed dimension the principal actually carries."""
        out = []
        for dim, value in principal.present().items():
            st = self.status_for(dim, value)
            if st is not None:
                out.append(st)
        return out

    # ── projection ───────────────────────────────────────────────────────────

    def project(
        self,
        provider: str,
        model: str | None,
        est_input_tokens: int,
        max_output_tokens: int,
        batch: bool = False,
    ) -> float:
        from glc.economics import pricing as _pricing

        mode = str(self.projection.get("output_tokens", "max_tokens"))
        if mode == "ratio":
            ratio = float(self.projection.get("output_ratio", 1.0))
            out_tokens = int(max(0, est_input_tokens) * ratio)
        else:
            out_tokens = int(max_output_tokens)
        usd = _pricing.project_usd(
            provider,
            model,
            max(0, int(est_input_tokens)),
            out_tokens,
            batch=batch,
            safety_factor=float(self.projection.get("safety_factor", 1.0)),
        )
        return max(usd, float(self.projection.get("min_usd", 0.0)))

    # ── the gate ─────────────────────────────────────────────────────────────

    def admit(
        self,
        principal: Principal,
        projected_usd: float,
    ) -> Admission:
        """Decide whether a call costing at most `projected_usd` may proceed.

        Returns a verdict rather than raising, so the caller (which knows
        whether it is inside a cascade and could retry at a cheaper tier) picks
        the response.
        """
        if not self.enabled:
            return Admission(
                allowed=True,
                projected_usd=projected_usd,
                enforced=False,
                reason="budget enforcement disabled in budgets.yaml",
            )
        statuses = self.statuses(principal)
        if not statuses:
            allow_unmatched = str(self.defaults.get("unmatched", "allow")) == "allow"
            return Admission(
                allowed=allow_unmatched,
                projected_usd=projected_usd,
                reason=(
                    "no budget policy matches this principal"
                    if allow_unmatched
                    else "no budget policy matches this principal and defaults.unmatched=deny"
                ),
            )
        warn_at = float(self.alerts.get("warn_at_fraction", 1.1))
        warnings = []
        breached = None
        for st in statuses:
            if st.spent_usd + projected_usd > st.limit_usd:
                # Report the tightest breach, so the caller sees the binding
                # constraint rather than whichever dimension happened to be first.
                if breached is None or st.remaining_usd < breached.remaining_usd:
                    breached = st
            elif st.fraction_used >= warn_at:
                warnings.append(st)
        if breached is not None:
            return Admission(
                allowed=False,
                projected_usd=projected_usd,
                checked=statuses,
                breached=breached,
                warnings=warnings,
                reason=(
                    f"{breached.principal} would exceed its {breached.period} budget: "
                    f"spent ${breached.spent_usd:.6f} of ${breached.limit_usd:.6f}, "
                    f"this call projects ${projected_usd:.6f} "
                    f"(${breached.remaining_usd:.6f} remaining)"
                ),
            )
        return Admission(
            allowed=True,
            projected_usd=projected_usd,
            checked=statuses,
            warnings=warnings,
            reason="within budget",
        )

    def affordable_usd(self, principal: Principal) -> float | None:
        """Tightest remaining allowance, or None when ungoverned.

        The routing policy uses this to prefer a tier the principal can still
        pay for, which is a *hint*. `admit()` remains the decision.
        """
        statuses = self.statuses(principal)
        if not statuses:
            return None
        return min(s.remaining_usd for s in statuses)

    def describe(self) -> dict:
        return {
            "path": str(self.path),
            "enabled": self.enabled,
            "defaults": self.defaults,
            "projection": self.projection,
            "alerts": self.alerts,
            "policies": [p.as_dict() for p in self.policies()],
        }


_controller: BudgetController | None = None
_ctl_lock = threading.Lock()


def get_controller() -> BudgetController:
    global _controller
    with _ctl_lock:
        if _controller is None:
            _controller = BudgetController()
        return _controller


def reload_controller(path: Path | str | None = None) -> BudgetController:
    global _controller
    with _ctl_lock:
        _controller = BudgetController(path)
        return _controller


def reset_controller() -> None:
    """Drop the singleton (tests, config-dir switches)."""
    global _controller
    with _ctl_lock:
        _controller = None
