"""Which thinking dial a Gemini model takes, decided by version not by substring.

Newer Gemini models think by default. If the gateway does not recognise one as a
thinking model it sends no thinkingConfig, the reasoning budget stays uncapped,
and a call with a modest maxOutputTokens can spend the whole allowance in the
reasoning channel and return empty content — a fully billed non-answer. That is
the same failure `S17Code/config/tiers.yaml` records for gpt-oss-120b.

The detector matched the literal strings "3-flash" and "3.1-flash", so every
later minor version fell through to None: gemini-3.5-flash, gemini-3.6-flash and
gemini-3.7-flash were all treated as non-thinking models, which is how a planner
call came back with no JSON object in it.
"""
from __future__ import annotations

import pytest

from glc.providers import _gemini_supports_thinking, _gemini_thinking_knob


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # Families that already worked, which must keep working.
        ("gemini-2.5-flash", "budget"),
        ("gemini-2.5-pro", "level"),
        ("gemini-3-flash-preview", "level"),
        ("gemini-3.1-pro-preview", "level"),
        # Lite never thinks, at any version.
        ("gemini-2.5-flash-lite", None),
        ("gemini-3.1-flash-lite", None),
        ("gemini-3.5-flash-lite", None),
        # The regression: minor versions past .1 were silently non-thinking.
        ("gemini-3.5-flash", "level"),
        ("gemini-3.6-flash", "level"),
        ("gemini-3.7-flash", "level"),
        ("gemini-3.5-pro", "level"),
        # A major version nobody has written a branch for yet. Capping the
        # reasoning of an unknown thinking model and letting the existing 400
        # retry strip the field is the safe direction; leaving it uncapped is
        # the failure this test exists for.
        ("gemini-4-flash", "level"),
        ("gemini-4.2-pro", "level"),
        # Not Gemini at all.
        ("openai/gpt-oss-120b", None),
        ("", None),
    ],
)
def test_the_knob_follows_the_version(model: str, expected: str | None) -> None:
    assert _gemini_thinking_knob(model) == expected


def test_support_agrees_with_the_knob() -> None:
    for model in ("gemini-3.5-flash", "gemini-3.7-flash", "gemini-4-flash"):
        assert _gemini_supports_thinking(model) is True
    for model in ("gemini-3.5-flash-lite", "openai/gpt-oss-120b"):
        assert _gemini_supports_thinking(model) is False


@pytest.mark.parametrize(
    ("model", "reasoning", "expected"),
    [
        # "off" must be said. Omitting thinkingConfig leaves the reasoning
        # channel uncapped, drawing on the same maxOutputTokens as the answer.
        # Measured on gemini-3.5-flash at maxOutputTokens 1600: unset spends 234
        # thought tokens, minimal spends 0.
        ("gemini-3.5-flash", "off", {"thinkingLevel": "minimal"}),
        ("gemini-3.7-flash", "off", {"thinkingLevel": "minimal"}),
        ("gemini-2.5-pro", "off", {"thinkingLevel": "minimal"}),
        ("gemini-2.5-flash", "off", {"thinkingBudget": 0}),
        # A model that never thinks needs no dial at all.
        ("gemini-3.1-flash-lite", "off", None),
        ("openai/gpt-oss-120b", "off", None),
        # The dial's other settings keep their existing meaning.
        ("gemini-3.5-flash", "high", {"thinkingLevel": "high"}),
        ("gemini-2.5-flash", "medium", {"thinkingBudget": 8192}),
        ("gemini-2.5-flash", "low", {"thinkingBudget": 2048}),
        # Unset means unset: leave the provider's own default alone.
        ("gemini-3.5-flash", None, None),
        ("gemini-3.5-flash", "", None),
        # An unrecognised setting must not raise. _GEMINI_BUDGETS has no key for
        # it, and a KeyError here would take out the whole call.
        ("gemini-2.5-flash", "extreme", None),
    ],
)
def test_off_is_configured_rather_than_omitted(model, reasoning, expected) -> None:
    from glc.providers import _gemini_thinking_config

    assert _gemini_thinking_config(model, reasoning) == expected


def test_thinking_level_off_is_never_sent_as_a_level() -> None:
    """The API rejects thinkingLevel: "off" as an invalid enum value."""
    from glc.providers import _gemini_thinking_config

    for model in ("gemini-3.5-flash", "gemini-3.7-flash", "gemini-2.5-pro"):
        config = _gemini_thinking_config(model, "off")
        assert config is not None
        assert config.get("thinkingLevel") != "off"


def test_a_two_digit_minor_is_not_read_as_a_smaller_number() -> None:
    """3.10 must not compare as 3.1 in a way that changes the answer.

    Both are >= 3 so both take a level, but the parse must not raise or fall
    through to None on a version string it has not seen.
    """
    assert _gemini_thinking_knob("gemini-3.10-flash") == "level"
