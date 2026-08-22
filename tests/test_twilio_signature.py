"""Twilio HMACs the public URL it posted to, not the internal ASGI URL.

The documented deployment (`twilio_sms/server.py`) sits behind an ngrok
tunnel: Twilio posts to `https://<id>.ngrok.io/webhooks/twilio_sms`, ngrok
forwards it in-process as plain `http://` on the bound port, and
`str(request.url)` reconstructs that *internal* URL -- not the public one
Twilio actually signed. The scheme mismatch alone (`https` vs `http`) is
enough to make every genuine inbound message fail signature verification.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from glc.channels.catalogue.twilio_sms.webhook import (
    WEBHOOK_PATH,
    build_app,
    compute_signature,
)
from glc.channels.envelope import ChannelMessage

AUTH_TOKEN = "test_token_abc123"
PUBLIC_BASE = "https://abcd1234.ngrok.io"


class FakeAdapter:
    async def on_message(self, form):
        return ChannelMessage(
            channel="twilio_sms",
            channel_user_id=form.get("From", ""),
            user_handle=form.get("From", ""),
            text=form.get("Body") or None,
            trust_level="owner_paired",
            arrived_at=datetime.now(UTC),
        )


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", AUTH_TOKEN)
    monkeypatch.delenv("GLC_TWILIO_SKIP_SIG", raising=False)
    monkeypatch.delenv("GLC_PUBLIC_BASE", raising=False)


def _client_and_seen():
    seen: list[ChannelMessage] = []

    async def handle_message(msg):
        seen.append(msg)

    app = build_app(FakeAdapter(), handle_message)
    return TestClient(app), seen


def test_a_genuine_request_behind_the_documented_ngrok_tunnel_is_accepted(monkeypatch):
    """Twilio signed the public https URL it dialed (GLC_PUBLIC_BASE, per
    server.py's own documented deployment), not the internal ASGI URL this
    process sees. A real webhook must not be rejected as forged."""
    monkeypatch.setenv("GLC_PUBLIC_BASE", PUBLIC_BASE)
    client, seen = _client_and_seen()
    form = {"From": "+19999999999", "To": "+15555550100", "Body": "hi", "NumMedia": "0"}
    sig = compute_signature(AUTH_TOKEN, f"{PUBLIC_BASE}{WEBHOOK_PATH}", form)

    resp = client.post(WEBHOOK_PATH, data=form, headers={"X-Twilio-Signature": sig})

    assert resp.status_code == 200
    assert len(seen) == 1


def test_without_a_configured_public_base_the_internal_url_still_works():
    """No GLC_PUBLIC_BASE (local dev, no tunnel): unchanged behaviour."""
    client, seen = _client_and_seen()
    form = {"From": "+19999999999", "To": "+15555550100", "Body": "hi", "NumMedia": "0"}
    sig = compute_signature(AUTH_TOKEN, f"http://testserver{WEBHOOK_PATH}", form)

    resp = client.post(WEBHOOK_PATH, data=form, headers={"X-Twilio-Signature": sig})

    assert resp.status_code == 200
    assert len(seen) == 1


def test_a_signature_for_the_wrong_host_is_still_rejected(monkeypatch):
    """The fix must not widen acceptance to any signature at all."""
    monkeypatch.setenv("GLC_PUBLIC_BASE", PUBLIC_BASE)
    client, seen = _client_and_seen()
    form = {"From": "+19999999999", "Body": "hi", "NumMedia": "0"}
    sig = compute_signature(AUTH_TOKEN, f"https://attacker.example{WEBHOOK_PATH}", form)

    resp = client.post(WEBHOOK_PATH, data=form, headers={"X-Twilio-Signature": sig})

    assert resp.status_code == 403
    assert seen == []
