"""Meta's hub.* verification handshake must fail closed.

`GET /v1/channels/{name}/webhook` answers Meta's subscription challenge by
comparing `hub.verify_token` against `<CHANNEL>_VERIFY_TOKEN`. When that variable
is unset the expected value is the empty string, and `compare_digest("", "")` is
True — so an unconfigured install echoes the challenge to anyone who asks with an
empty token, which is what a fresh checkout looks like.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_verify_token(monkeypatch):
    """The state a fresh checkout is in: the variable was never set."""
    monkeypatch.delenv("WHATSAPP_VERIFY_TOKEN", raising=False)


def test_unset_token_does_not_echo_the_challenge(app_client):
    """The bug: with no token configured, an empty token is accepted."""
    r = app_client.get(
        "/v1/channels/whatsapp/webhook",
        params={"hub.mode": "subscribe", "hub.verify_token": "", "hub.challenge": "PWNED"},
    )
    assert r.status_code == 403, (
        "an unconfigured install echoed Meta's challenge to an empty verify token, "
        "letting anyone subscribe this endpoint to their own app"
    )
    assert "PWNED" not in r.text


def test_unset_token_rejects_a_supplied_token(app_client):
    r = app_client.get(
        "/v1/channels/whatsapp/webhook",
        params={"hub.mode": "subscribe", "hub.verify_token": "guess", "hub.challenge": "x"},
    )
    assert r.status_code == 403


def test_configured_token_still_verifies(monkeypatch, app_client):
    """The fix must not break the path Meta actually uses."""
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "s3cret-verify-token")
    r = app_client.get(
        "/v1/channels/whatsapp/webhook",
        params={"hub.mode": "subscribe",
                "hub.verify_token": "s3cret-verify-token",
                "hub.challenge": "1158201444"},
    )
    assert r.status_code == 200
    assert r.text == "1158201444"


def test_configured_token_rejects_a_wrong_one(monkeypatch, app_client):
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "s3cret-verify-token")
    r = app_client.get(
        "/v1/channels/whatsapp/webhook",
        params={"hub.mode": "subscribe", "hub.verify_token": "wrong", "hub.challenge": "x"},
    )
    assert r.status_code == 403


def test_channel_without_meta_verification_is_still_404(app_client):
    """Unchanged behaviour: only meta_hub channels have this handshake."""
    r = app_client.get(
        "/v1/channels/telegram/webhook",
        params={"hub.mode": "subscribe", "hub.verify_token": "", "hub.challenge": "x"},
    )
    assert r.status_code == 404
