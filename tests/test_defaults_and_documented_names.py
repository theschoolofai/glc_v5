"""Defaults and documented names, both of which had gone quietly stale.

Two bug classes meet here, and neither one errors when it is wrong:

  * A hardcoded default that upstream has since retired. `gemini-2.5-flash` is
    still returned by ListModels, so nothing in the config reads as wrong, but
    `generateContent` answers 404 "no longer available to new users".
  * Documentation naming an environment variable that nothing reads. Follow it
    and the feature fails closed without saying why.

Both are cheap to assert and expensive to discover.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from glc import providers

REPO = Path(__file__).resolve().parents[1]


class TestGeminiDefaultModel:
    def test_the_default_is_not_the_retired_model(self):
        source = (REPO / "glc" / "providers.py").read_text(encoding="utf-8")

        assert 'os.getenv("GEMINI_MODEL", "gemini-2.5-flash")' not in source, (
            "gemini-2.5-flash 404s on generateContent for new accounts")

    def test_the_default_is_recorded_next_to_its_reason(self):
        """A bare literal is how this went stale the first time."""
        source = (REPO / "glc" / "providers.py").read_text(encoding="utf-8")
        default = re.search(r'os\.getenv\("GEMINI_MODEL", "([^"]+)"\)', source)

        assert default, "the GEMINI_MODEL default should stay greppable"
        assert "no longer available to new users" in source, (
            "keep the note saying why 2.5-flash was abandoned")


class TestThinkingKnobDoesNotGoStaleEachRelease:
    """The ladder named 3 and 3.1 only.

    3.5 and 3.7 shipped afterwards and fell through to None — "no thinking
    support" — so raising the default model would have silently turned
    reasoning off across the whole pool. That is the failure mode that punishes
    you for upgrading, so it is matched now rather than enumerated.
    """

    @pytest.mark.parametrize("model", [
        "gemini-3-flash", "gemini-3.1-flash", "gemini-3.5-flash", "gemini-3.7-flash",
        "gemini-3-pro", "gemini-3.1-pro", "gemini-3.5-pro",
    ])
    def test_every_gemini_3x_flash_or_pro_supports_thinking_level(self, model):
        assert providers._gemini_thinking_knob(model) == "level"

    def test_the_configured_default_still_supports_reasoning(self):
        """The specific regression: bumping the default must not cost thinking."""
        source = (REPO / "glc" / "providers.py").read_text(encoding="utf-8")
        default = re.search(r'os\.getenv\("GEMINI_MODEL", "([^"]+)"\)', source).group(1)

        assert providers._gemini_thinking_knob(default) is not None, (
            f"the default model {default} reports no thinking support")

    @pytest.mark.parametrize("model", ["gemini-3-flash-lite", "gemini-3.5-flash-lite"])
    def test_lite_models_are_still_excluded(self, model):
        """Guard against the new pattern being too greedy."""
        assert providers._gemini_thinking_knob(model) is None

    def test_2_5_flash_keeps_its_budget_knob(self):
        """Older API, different field. The fix must not collapse the two."""
        assert providers._gemini_thinking_knob("gemini-2.5-flash") == "budget"

    def test_a_non_gemini_model_is_unaffected(self):
        assert providers._gemini_thinking_knob("claude-opus-5") is None


class TestDocumentedEnvNamesAreTheOnesRead:
    """An incomplete S16 -> S17 rename, guessed at rather than read.

    glc_v5 kept the S16-era vocabulary for the whole bridge; coding_agent
    renamed itself and its docs then guessed the gateway's variable by prefix
    substitution. Both sides guessed, in opposite directions, and a prefix that
    names a repo was treated as if it named a variable.
    """

    @staticmethod
    def _assigned(text: str) -> set[str]:
        """Only names the file actually tells you to SET.

        Deliberately not every name mentioned: prose may legitimately name the
        agent's variable, or name a dead one in order to warn you off it.
        """
        return set(re.findall(r"^([A-Z][A-Z0-9_]+)=", text, re.MULTILINE))

    def test_every_bridge_variable_it_tells_you_to_set_is_one_it_reads(self):
        assigned = self._assigned((REPO / ".env.example").read_text(encoding="utf-8"))
        bridge = {n for n in assigned if "S16" in n or "S17" in n}
        code = (REPO / "glc" / "channels" / "agent_bridge.py").read_text(encoding="utf-8")
        code += (REPO / "glc" / "routes" / "channels.py").read_text(encoding="utf-8")

        assert bridge, ".env.example should document the bridge variables"
        for name in bridge:
            assert name in code, f"{name} is documented as settable but read nowhere"

    def test_the_name_read_for_the_bridge_is_documented_somewhere(self):
        code = (REPO / "glc" / "routes" / "channels.py").read_text(encoding="utf-8")
        read = set(re.findall(r'os\.getenv\("(GLC_S1[67]_[A-Z_]+)"', code))
        documented = (REPO / ".env.example").read_text(encoding="utf-8")

        for name in read:
            assert name in documented, f"{name} is read but documented nowhere"
