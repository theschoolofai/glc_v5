"""Operator errors on the live Gmail send path (no mock client)."""

from __future__ import annotations

from pathlib import Path

import pytest

from glc.channels.catalogue.gmail.adapter import Adapter

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_google_auth_for_gmail_send():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "google-auth" in text
    assert "google-auth-oauthlib" in text
    assert "google-api-python-client" in text


def test_live_gmail_missing_token_explains_auth_setup(monkeypatch, tmp_path):
    monkeypatch.setenv("GMAIL_TOKEN_PATH", str(tmp_path / "missing-token.json"))
    adapter = Adapter()
    with pytest.raises(RuntimeError, match="gmail.auth_setup"):
        adapter._get_client()
