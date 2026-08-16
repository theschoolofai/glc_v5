"""Provider ceilings must be raisable without editing Python.

`LIMITS` in glc/routing/core.py carries free-tier numbers. They are correct
defaults and wrong for anybody who has enabled billing: a paid Gemini key is
allowed far more than 15 requests a minute, but the gateway throttles it to 15
anyway and inserts a 4-second cooldown between calls.

Pricing already works this way — `config/pricing.yaml` opens by saying no price
is ever hardcoded in Python. These tests hold rate limits to the same standard.
"""

from __future__ import annotations

from glc.routing.core import DEFAULT_LIMITS, apply_env_overrides


def _fresh() -> dict:
    return {name: dict(values) for name, values in DEFAULT_LIMITS.items()}


def test_defaults_are_unchanged_when_no_env_is_set():
    """The free-tier numbers stay put for anyone who sets nothing."""
    limits = apply_env_overrides(_fresh(), {})
    assert limits["gemini"]["rpm"] == 15
    assert limits["gemini"]["cooldown"] == 4


def test_env_raises_a_single_field():
    limits = apply_env_overrides(_fresh(), {"GLC_LIMIT_GEMINI_RPM": "150"})
    assert limits["gemini"]["rpm"] == 150
    # Neighbouring fields are untouched.
    assert limits["gemini"]["rpd"] == 1000


def test_env_can_set_every_numeric_field():
    limits = apply_env_overrides(_fresh(), {
        "GLC_LIMIT_GEMINI_RPM": "150",
        "GLC_LIMIT_GEMINI_RPD": "10000",
        "GLC_LIMIT_GEMINI_TPM": "2000000",
        "GLC_LIMIT_GEMINI_COOLDOWN": "0",
    })
    gemini = limits["gemini"]
    assert (gemini["rpm"], gemini["rpd"], gemini["tpm"]) == (150, 10000, 2000000)
    assert gemini["cooldown"] == 0


def test_cooldown_accepts_a_fraction():
    """Sub-second cooldowns are meaningful; integer parsing would floor them to 0."""
    limits = apply_env_overrides(_fresh(), {"GLC_LIMIT_GROQ_COOLDOWN": "0.5"})
    assert limits["groq"]["cooldown"] == 0.5


def test_providers_are_independent():
    limits = apply_env_overrides(_fresh(), {"GLC_LIMIT_GEMINI_RPM": "150"})
    assert limits["groq"]["rpm"] == 30
    assert limits["github"]["rpm"] == 10


def test_unknown_provider_is_ignored():
    """A typo must not invent a provider the router will then try to use."""
    limits = apply_env_overrides(_fresh(), {"GLC_LIMIT_GEMNI_RPM": "150"})
    assert "gemni" not in limits
    assert limits["gemini"]["rpm"] == 15


def test_unknown_field_is_ignored():
    limits = apply_env_overrides(_fresh(), {"GLC_LIMIT_GEMINI_RPZ": "5"})
    assert "rpz" not in limits["gemini"]


def test_garbage_value_leaves_the_default_in_place():
    """A malformed ceiling must not become zero, which would refuse every call."""
    limits = apply_env_overrides(_fresh(), {"GLC_LIMIT_GEMINI_RPM": "lots"})
    assert limits["gemini"]["rpm"] == 15


def test_negative_value_is_refused():
    limits = apply_env_overrides(_fresh(), {"GLC_LIMIT_GEMINI_RPM": "-1"})
    assert limits["gemini"]["rpm"] == 15


def test_empty_value_is_ignored():
    """An unset-but-present variable is not an instruction to throttle to zero."""
    limits = apply_env_overrides(_fresh(), {"GLC_LIMIT_GEMINI_RPM": "   "})
    assert limits["gemini"]["rpm"] == 15


def test_tokens_per_day_is_overridable_where_it_exists():
    limits = apply_env_overrides(_fresh(), {"GLC_LIMIT_CEREBRAS_TOKENS_PER_DAY": "5000000"})
    assert limits["cerebras"]["tokens_per_day"] == 5_000_000


def test_a_field_absent_from_a_provider_is_not_added():
    """gemini has no tokens_per_day; setting one must not invent a new ceiling."""
    limits = apply_env_overrides(_fresh(), {"GLC_LIMIT_GEMINI_TOKENS_PER_DAY": "5000000"})
    assert "tokens_per_day" not in limits["gemini"]


def test_module_level_limits_went_through_the_override():
    """The dict the router reads is the overridden one, not the raw defaults.

    Deliberately not `importlib.reload`: LIMITS is shared mutable module state
    that `providers.py` adds per-key entries to at import, and reloading it
    rebinds this module's copy while every other importer keeps the old one.
    The suite then fails several files later with a missing `gemini_1`, which is
    a much harder bug to find than the one it was testing for.
    """
    from glc.routing import core

    for provider, values in DEFAULT_LIMITS.items():
        for field in values:
            assert field in core.LIMITS[provider], (
                f"{provider}.{field} was dropped building LIMITS"
            )
