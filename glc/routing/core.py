"""Capability-aware router. Same RPM/RPD bookkeeping as V1, but now it can
skip providers that lack a requested capability (tools/reasoning/structured/caching)."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict, deque

log = logging.getLogger(__name__)

# Free-tier ceilings, which is the right default: a key with no billing behind
# it is the common case, and guessing high would turn every call into a 429.
#
# They are only defaults. A paid key is allowed considerably more, and a ceiling
# that cannot be raised without editing this file throttles a caller who is
# paying for more than it grants — silently, because being under your own
# provider's limit looks identical to being fast enough.
DEFAULT_LIMITS = {
    "ollama": {"rpm": 9999, "rpd": 9999999, "tpm": 99999999, "cooldown": 0, "max_ctx": 32000},
    "cerebras": {
        "rpm": 30,
        "rpd": 9999,
        "tpm": 60000,
        "cooldown": 2,
        "max_ctx": 8000,
        "tokens_per_day": 1_000_000,
    },
    "groq": {"rpm": 30, "rpd": 1000, "tpm": 6000, "cooldown": 2, "max_ctx": 100000},
    "nvidia": {"rpm": 40, "rpd": 9999, "tpm": 100000, "cooldown": 2, "max_ctx": 100000},
    "gemini": {"rpm": 15, "rpd": 1000, "tpm": 250000, "cooldown": 4, "max_ctx": 1000000},
    "openrouter": {"rpm": 20, "rpd": 50, "tpm": 99999999, "cooldown": 3, "max_ctx": 100000},
    "github": {"rpm": 10, "rpd": 50, "tpm": 99999999, "cooldown": 6, "max_ctx": 8000},
}

# `cooldown` is a delay in seconds and is meaningfully fractional; the rest are
# counts. Parsing them all as int would floor a 0.5s cooldown to 0.
_FRACTIONAL_FIELDS = frozenset({"cooldown"})


def apply_env_overrides(limits: dict, environ: dict | None = None) -> dict:
    """Raise (or lower) a provider ceiling from the environment, in place.

    ``GLC_LIMIT_<PROVIDER>_<FIELD>``, for example ``GLC_LIMIT_GEMINI_RPM=150``.

    Only providers and fields that already exist are touched. An unknown name is
    ignored rather than created: inventing a provider here would hand the router
    a candidate it cannot call, and inventing a field would add a ceiling nothing
    enforces. A malformed or negative value leaves the default in place and logs,
    because the failure mode of accepting it is a gateway that refuses every
    request while looking correctly configured.
    """
    env = os.environ if environ is None else environ
    for key, raw in env.items():
        if not key.startswith("GLC_LIMIT_"):
            continue
        remainder = key[len("GLC_LIMIT_"):].lower()
        # Provider and field names both contain underscores (gemini_1,
        # tokens_per_day), so split on the longest provider that matches.
        provider = next(
            (name for name in sorted(limits, key=len, reverse=True)
             if remainder.startswith(name + "_")),
            None,
        )
        if provider is None:
            log.warning("ignoring %s: no such provider", key)
            continue
        field = remainder[len(provider) + 1:]
        if field not in limits[provider]:
            log.warning("ignoring %s: %s has no %r ceiling", key, provider, field)
            continue
        text = (raw or "").strip()
        if not text:
            continue
        try:
            value = float(text) if field in _FRACTIONAL_FIELDS else int(text)
        except ValueError:
            log.warning("ignoring %s=%r: not a number", key, raw)
            continue
        if value < 0:
            log.warning("ignoring %s=%r: a ceiling cannot be negative", key, raw)
            continue
        limits[provider][field] = value
    return limits


LIMITS = apply_env_overrides(
    {name: dict(values) for name, values in DEFAULT_LIMITS.items()}
)

# One Google AI Studio key is one independently-metered provider.  The graph
# scheduler requests the logical name ``gemini``; this router owns expansion,
# cooldown and failover so no caller can accidentally build a second key pool.
MAX_GEMINI_KEYS = 16

SHORTCUTS = {
    "g": "gemini",
    "gem": "gemini",
    "gemini": "gemini",
    "n": "nvidia",
    "nv": "nvidia",
    "nvidia": "nvidia",
    "o": "ollama",
    "oll": "ollama",
    "ollama": "ollama",
    "gr": "groq",
    "groq": "groq",
    "c": "cerebras",
    "cer": "cerebras",
    "cerebras": "cerebras",
    "or": "openrouter",
    "opr": "openrouter",
    "openrouter": "openrouter",
    "gh": "github",
    "ghb": "github",
    "github": "github",
}


def resolve(name):
    if not name:
        return None
    return SHORTCUTS.get(name.lower())


class RateState:
    def __init__(self):
        self.calls_minute = deque()
        self.tokens_minute = deque()
        self.calls_today = 0
        self.tokens_today = 0
        self.day_start = self._day_start()
        self.last_call = 0.0
        self.unavailable_until = 0.0
        self.unavailable_reason = ""

    @staticmethod
    def _day_start():
        now = time.time()
        return now - (now % 86400)

    def gc(self):
        now = time.time()
        if now - self.day_start >= 86400:
            self.calls_today = 0
            self.tokens_today = 0
            self.day_start = self._day_start()
        cutoff = now - 60
        while self.calls_minute and self.calls_minute[0] < cutoff:
            self.calls_minute.popleft()
        while self.tokens_minute and self.tokens_minute[0][0] < cutoff:
            self.tokens_minute.popleft()

    def can_use(self, limits, est_tokens=0):
        self.gc()
        now = time.time()
        if now < self.unavailable_until:
            return False, f"backoff: {self.unavailable_reason} ({self.unavailable_until - now:.0f}s left)"
        wait = limits["cooldown"] - (now - self.last_call)
        if wait > 0:
            return False, f"cooldown ({wait:.1f}s)"
        if len(self.calls_minute) >= limits["rpm"]:
            return False, "RPM limit"
        if self.calls_today >= limits["rpd"]:
            return False, "RPD limit"
        tpm = sum(t for _, t in self.tokens_minute)
        if tpm + est_tokens > limits["tpm"]:
            return False, "TPM limit"
        if "tokens_per_day" in limits and self.tokens_today + est_tokens > limits["tokens_per_day"]:
            return False, "daily token cap"
        return True, None

    def record(self, tokens):
        now = time.time()
        self.calls_minute.append(now)
        self.tokens_minute.append((now, tokens))
        self.calls_today += 1
        self.tokens_today += tokens
        self.last_call = now

    def mark_unavailable(self, seconds: float, reason: str):
        self.unavailable_until = time.time() + seconds
        self.unavailable_reason = reason

    def snapshot(self, limits):
        self.gc()
        now = time.time()
        tpm = sum(t for _, t in self.tokens_minute)
        return {
            "rpm_used": len(self.calls_minute),
            "rpm_limit": limits["rpm"],
            "rpd_used": self.calls_today,
            "rpd_limit": limits["rpd"],
            "tpm_used": tpm,
            "tpm_limit": limits["tpm"],
            "tokens_today": self.tokens_today,
            "tokens_per_day": limits.get("tokens_per_day"),
            "cooldown_remaining": max(0, limits["cooldown"] - (now - self.last_call))
            if self.last_call
            else 0,
            "last_call": self.last_call,
            "backoff_remaining": max(0, self.unavailable_until - now),
            "backoff_reason": self.unavailable_reason if now < self.unavailable_until else "",
        }


class Router:
    def __init__(self, providers: dict, order: list[str]):
        self.providers = providers
        self.order = [p for p in order if p in providers or self._pool_of(p)]
        self.state = defaultdict(RateState)
        self.lock = asyncio.Lock()

    def _pool_of(self, base: str) -> list[str]:
        return [p for p in self.providers if p.startswith(base + "_")]

    def expand(self, names: list[str]) -> list[str]:
        """Expand a logical provider (for example ``gemini``) to its live,
        individually-metered instances while preserving order and removing
        duplicates."""
        out: list[str] = []
        seen: set[str] = set()
        for name in names:
            base = resolve(name) or name
            instances = [base] if base in self.providers else self._pool_of(base)
            for instance in instances:
                if instance not in seen:
                    seen.add(instance)
                    out.append(instance)
        return out

    def candidates(self, override=None):
        if override:
            return self.expand([override])
        return self.expand(self.order)

    def pick(self, est_tokens, candidates, required_caps: list[str] | None = None):
        attempts = []
        for name in candidates:
            limits = LIMITS[name]
            prov = self.providers[name]
            caps = getattr(prov, "capabilities", {})
            if required_caps:
                missing = [c for c in required_caps if not caps.get(c)]
                if missing:
                    attempts.append({"provider": name, "reason": f"skipped:no_{missing[0]}"})
                    continue
            if est_tokens > limits["max_ctx"]:
                attempts.append(
                    {"provider": name, "reason": f"prompt {est_tokens} > max_ctx {limits['max_ctx']}"}
                )
                continue
            ok, why = self.state[name].can_use(limits, est_tokens)
            if ok:
                return name, attempts
            attempts.append({"provider": name, "reason": why})
        return None, attempts

    def all_status(self):
        out = {}
        for name in self.providers:
            out[name] = self.state[name].snapshot(LIMITS[name])
            out[name]["model"] = self.providers[name].model
            out[name]["capabilities"] = getattr(self.providers[name], "capabilities", {})
        return out


# -----------------------------------------------------------------------------
# V3 Router pool — separate failover ring for routing-decision LLM calls.
# Same rate-state machinery, separate state dict so router quotas never compete
# with worker quotas (provider keys are shared but providers meter per-model).
# -----------------------------------------------------------------------------

DEFAULT_ROUTER_ORDER = ["cerebras", "groq", "nvidia", "github"]


class RouterPool:
    """Failover ring for router-LLM calls. Mirrors `Router` but for the
    Perception/Memory/Decision routing classifiers. Each call is logged with
    a call_role marker (router_perception | router_memory | router_decision)
    so the dashboard can show router activity separately from worker activity.
    """

    def __init__(self, providers: dict, order: list[str]):
        self.providers = providers
        self.order = [p for p in order if p in providers]
        self.state = defaultdict(RateState)
        self.lock = asyncio.Lock()

    def candidates(self):
        return list(self.order)

    def pick(self, est_tokens=400):
        """Pick first available router provider. Caps require nothing — router
        LLMs only need to emit one word, no tools/reasoning/structured needed."""
        attempts = []
        for name in self.candidates():
            limits = LIMITS[name]
            ok, why = self.state[name].can_use(limits, est_tokens)
            if ok:
                return name, attempts
            attempts.append({"provider": name, "reason": why})
        return None, attempts

    def all_status(self):
        out = {}
        for name in self.providers:
            out[name] = self.state[name].snapshot(LIMITS[name])
            out[name]["model"] = self.providers[name].model
        return out
