"""Tests verifying OpenAI provider registration and routing config."""
import os
import pytest
from unittest.mock import patch


def test_openai_shortcut_exists():
    """The routing shortcuts must include 'openai'."""
    from glc.routing.core import SHORTCUTS
    assert "openai" in SHORTCUTS, "'openai' shortcut is missing from SHORTCUTS"
    assert SHORTCUTS["openai"] == "openai"


def test_oai_shortcut_exists():
    """The 'oai' abbreviation must resolve to 'openai'."""
    from glc.routing.core import SHORTCUTS
    assert "oai" in SHORTCUTS, "'oai' shortcut is missing from SHORTCUTS"
    assert SHORTCUTS["oai"] == "openai"


@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-for-unit-test"})
def test_openai_provider_registered_when_key_present():
    """When OPENAI_API_KEY is set, build_providers must include 'openai'."""
    from glc.providers import build_providers
    providers = build_providers(cache_store=None)
    assert "openai" in providers, "OpenAI provider not registered despite OPENAI_API_KEY being set"
