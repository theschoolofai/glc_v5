"""Declarative policy engine.

evaluate(tool_call, context) -> PolicyVerdict
  - first matching rule wins
  - ties resolve to deny
  - default allow when trust_level == 'owner_paired' and no rule matches
  - default deny otherwise

Hot reload on SIGHUP (process-level handler installed by main.py). Malformed
yaml is rejected: the engine falls back to a deny-everything safe-default
config and logs a warning so the gateway boots in a known-safe state.
"""

from __future__ import annotations

import fnmatch
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from glc.policy.schemas import PolicyConfig, PolicyRule, PolicyVerdict

_SAFE_DEFAULT = PolicyConfig(
    rules=[
        PolicyRule(
            tool="*",
            trust_level="*",
            action="deny",
            reason="policy.yaml unreadable — falling back to deny-everything",
        )
    ]
)


@lru_cache(maxsize=512)
def _double_star_regex(pattern: str) -> re.Pattern[str]:
    """Translate a `**` glob, keeping every other fnmatch metacharacter.

    `re.escape(pattern).replace(...)` cannot do this. It escapes the whole
    pattern first, so only the substitutions it names survive: `?` and `[...]`
    stay escaped and match themselves. A rule written as `~/[Dd]ocuments/**`
    then matches nothing at all — and a `deny` that matches nothing is not
    inert, because `evaluate()` falls through to default-allow.

    So the pattern is scanned rather than escaped. The one thing that differs
    from `fnmatch` is deliberate and is the reason this branch exists:

        `**`  crosses a path separator
        `*`   does not
        `?`   one character, not a separator
        `[…]` fnmatch's character class, `!` or `^` to negate

    `(?s:…)` and `\\Z` mirror `fnmatch.translate`, so a value carrying a
    newline is judged the same way by both halves of `_matches_glob`.
    """
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        char = pattern[i]
        if char == "*":
            i += 1
            if i < n and pattern[i] == "*":
                while i < n and pattern[i] == "*":
                    i += 1
                out.append(".*")
            else:
                out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
            i += 1
        elif char == "[":
            end = i + 1
            if end < n and pattern[end] in "!^":
                end += 1
            if end < n and pattern[end] == "]":
                end += 1
            while end < n and pattern[end] != "]":
                end += 1
            if end >= n:  # unterminated class: fnmatch treats "[" as a literal
                out.append(re.escape(char))
                i += 1
            else:
                body = pattern[i + 1 : end].replace("\\", r"\\")
                out.append("[" + ("^" + body[1:] if body[:1] in ("!", "^") else body) + "]")
                i = end + 1
        else:
            out.append(re.escape(char))
            i += 1
    return re.compile("(?s:" + "".join(out) + r")\Z")


def _matches_glob(value: Any, pattern: str) -> bool:
    if not isinstance(value, str):
        return False
    # fnmatch has no separator-aware `**`, so that form gets its own translator.
    if "**" in pattern:
        return bool(_double_star_regex(pattern).match(value))
    return fnmatch.fnmatch(value, pattern)


def _matches_condition(condition: dict[str, Any], params: dict[str, Any]) -> bool:
    for key, expected in condition.items():
        if key.endswith("_glob"):
            target = key[: -len("_glob")]
            if not _matches_glob(params.get(target), expected):
                return False
        elif key.endswith("_regex"):
            target = key[: -len("_regex")]
            val = params.get(target)
            if not isinstance(val, str) or not re.search(expected, val):
                return False
        elif key.endswith("_in"):
            target = key[: -len("_in")]
            if params.get(target) not in (expected or []):
                return False
        elif key == "command_matches":
            cmd = params.get("command", "")
            if not isinstance(cmd, str):
                return False
            patterns = expected if isinstance(expected, list) else [expected]
            if not any(p in cmd for p in patterns):
                return False
        elif key == "recipient_type":
            if params.get("recipient_type") != expected:
                return False
        elif isinstance(expected, list):
            if params.get(key) not in expected:
                return False
        else:
            if params.get(key) != expected:
                return False
    return True


class PolicyEngine:
    def __init__(self, config: PolicyConfig):
        self.config = config
        self._lock = threading.Lock()

    @classmethod
    def from_yaml(cls, path: Path | str) -> PolicyEngine:
        p = Path(path)
        if not p.exists():
            return cls(_SAFE_DEFAULT)
        try:
            raw = yaml.safe_load(p.read_text()) or {}
            cfg = PolicyConfig.model_validate(raw)
        except Exception as e:  # pragma: no cover
            print(f"[glc.policy] malformed {p}: {e!r} — using deny-everything safe default")
            cfg = _SAFE_DEFAULT
        return cls(cfg)

    def evaluate(self, tool_call: dict[str, Any], context: dict[str, Any]) -> PolicyVerdict:
        """tool_call = {'name': 'email.send', 'arguments': {...}}
        context   = {'channel': 'telegram', 'trust_level': 'owner_paired',
                     'channel_user_id': '...'}"""
        tool = tool_call.get("name", "")
        params = tool_call.get("arguments") or {}
        channel = context.get("channel", "")
        trust_level = context.get("trust_level", "untrusted")

        with self._lock:
            rules = list(self.config.rules)

        deny_match: tuple[int, PolicyRule] | None = None
        first_match: tuple[int, PolicyRule] | None = None
        for i, rule in enumerate(rules):
            if rule.tool != "*" and rule.tool != tool:
                continue
            if rule.channel != "*" and rule.channel != channel:
                continue
            if rule.trust_level != "*" and rule.trust_level != trust_level:
                continue
            if rule.condition and not _matches_condition(rule.condition, params):
                continue
            if first_match is None:
                first_match = (i, rule)
            if rule.action == "deny" and deny_match is None:
                deny_match = (i, rule)

        if deny_match is not None:
            i, r = deny_match
            return PolicyVerdict(action="deny", reason=r.reason or "denied by policy", matched_rule_index=i)
        if first_match is not None:
            i, r = first_match
            return PolicyVerdict(
                action=r.action, reason=r.reason or f"matched rule #{i}", matched_rule_index=i
            )
        if trust_level == "owner_paired":
            return PolicyVerdict(action="allow", reason="default-allow for owner_paired")
        return PolicyVerdict(action="deny", reason=f"default-deny for trust_level={trust_level}")

    def reload(self, path: Path | str) -> None:
        new = PolicyEngine.from_yaml(path)
        with self._lock:
            self.config = new.config


# Module-level singleton, lazily constructed from config.policy_yaml_path().
_engine: PolicyEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> PolicyEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            from glc.config import policy_yaml_path

            _engine = PolicyEngine.from_yaml(policy_yaml_path())
    return _engine


def reload_engine() -> None:
    from glc.config import policy_yaml_path

    eng = get_engine()
    eng.reload(policy_yaml_path())


def evaluate(tool_call: dict[str, Any], context: dict[str, Any]) -> PolicyVerdict:
    return get_engine().evaluate(tool_call, context)
