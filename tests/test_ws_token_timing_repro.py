"""Reproduces: the channel WebSocket auth check compares the presented
token with plain `!=` instead of the constant-time comparison every other
auth path in this same file uses (see glc/routes/channels.py lines 57,
176, 261 for the compare_digest pattern this endpoint skips).

A non-constant-time comparison leaks how many leading characters matched
via response timing, letting an attacker recover the installation token
byte-by-byte instead of needing to guess it whole. This is the exact
"a token check that passes when it should not" pattern the session calls
out, on the one endpoint every channel bridge (Telegram, Discord, IMAP,
webhook, Twilio...) authenticates against.
"""
from __future__ import annotations

from unittest.mock import patch


def test_channel_ws_auth_uses_constant_time_comparison(app_client, install_token):
    """glc.routes.channels.channel_ws must authenticate the presented
    token via hmac.compare_digest, not Python's `!=`, exactly like every
    other auth check in the same module already does."""
    with patch("glc.routes.channels.hmac.compare_digest", wraps=__import__("hmac").compare_digest) as spy:
        try:
            with app_client.websocket_connect(f"/v1/channels/telegram?token={install_token}"):
                pass
        except Exception:
            pass  # connection lifecycle isn't what's under test here
        assert spy.called, (
            "channel_ws accepted/rejected a WebSocket connection without ever "
            "calling hmac.compare_digest — the token check is not constant-time."
        )
