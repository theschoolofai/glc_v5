"""Test: Meta webhook verify token must reject empty tokens.

Bug: When the channel's verify-token env var is unset (common during
initial setup), `os.environ.get(…, "")` returns an empty string.
`hmac.compare_digest("", "")` is True, so any caller sending
`hub.verify_token=` can register their own webhook URL and hijack
all future messages for that channel.

Fix: Reject verification outright when the expected token is empty.
"""

import pytest


def test_empty_verify_token_is_rejected(app_client, monkeypatch):
    """An empty hub.verify_token must NOT pass when the env var is unset."""
    # Ensure no verify token is configured (simulates first deployment)
    monkeypatch.delenv("WHATSAPP_VERIFY_TOKEN", raising=False)

    # Attacker sends an empty verify_token — before the fix this returns 200
    resp = app_client.get(
        "/v1/channels/whatsapp/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "",
            "hub.challenge": "hijacked",
        },
    )
    # Must be 403, NOT 200
    assert resp.status_code == 403, (
        f"Expected 403 but got {resp.status_code}. "
        f"An empty verify_token was accepted — webhook can be hijacked!"
    )
    # The challenge value must NOT appear in the response body
    assert "hijacked" not in resp.text


def test_correct_verify_token_still_works(app_client, monkeypatch):
    """A correctly configured verify token must still pass."""
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "my-secret-token")

    resp = app_client.get(
        "/v1/channels/whatsapp/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "my-secret-token",
            "hub.challenge": "challenge-accepted",
        },
    )
    assert resp.status_code == 200
    assert resp.text == "challenge-accepted"


def test_wrong_verify_token_is_rejected(app_client, monkeypatch):
    """A wrong verify token must be rejected."""
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "correct-token")

    resp = app_client.get(
        "/v1/channels/whatsapp/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong-token",
            "hub.challenge": "should-not-pass",
        },
    )
    assert resp.status_code == 403
