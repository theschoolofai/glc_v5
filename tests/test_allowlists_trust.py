"""Allowlist enforcement and trust-level classification."""

from __future__ import annotations

from datetime import UTC, datetime

from glc.channels.envelope import ChannelMessage, ChannelReply
from glc.security.allowlists import allowed
from glc.security.pairing import get_pairing_store
from glc.security.trust_level import classify

# Default channels.yaml ships every channel except webui as
# `enabled: false` — this is a security default for fresh installs.
# These tests exercise allowlist policy on an enabled channel.


def test_owner_in_dm_is_allowed():
    ok, _ = allowed("webui", "owner-1", owner_ids=["owner-1"], is_public_channel=False, was_mentioned=False)
    assert ok


def test_unknown_sender_in_dm_is_denied_by_default():
    ok, why = allowed(
        "webui", "stranger-1", owner_ids=["owner-1"], is_public_channel=False, was_mentioned=False
    )
    assert ok is False
    assert "allowed_senders" in why


def test_owner_in_public_without_mention_is_denied():
    ok, why = allowed("webui", "owner-1", owner_ids=["owner-1"], is_public_channel=True, was_mentioned=False)
    assert ok is False
    assert "mention" in why.lower()


def test_owner_in_public_with_mention_is_allowed():
    ok, _ = allowed("webui", "owner-1", owner_ids=["owner-1"], is_public_channel=True, was_mentioned=True)
    assert ok


def test_disabled_channel_blocks_owner():
    # telegram defaults to enabled=false; even the owner cannot reach it
    # until the operator enables the channel in channels.yaml.
    ok, why = allowed("telegram", "owner-1", owner_ids=["owner-1"])
    assert ok is False
    assert "disabled" in why


def test_disabled_channel_blocked(monkeypatch):
    import glc.security.allowlists as al

    def fake_load_channels():
        return {"channels": {"telegram": {"enabled": False}}}

    monkeypatch.setattr(al, "load_channels", fake_load_channels)
    ok, why = allowed("telegram", "owner-1", owner_ids=["owner-1"])
    assert ok is False
    assert "disabled" in why


def test_trust_level_unknown_is_untrusted():
    assert classify("telegram", "no-such-user") == "untrusted"


def test_trust_level_owner_paired():
    get_pairing_store().force_pair_owner("matrix", "owner-1", "owner")
    assert classify("matrix", "owner-1") == "owner_paired"


def test_trust_level_user_paired():
    store = get_pairing_store()
    code, _ = store.issue_code("slack", "U1", "user")
    store.confirm_code(code)
    assert classify("slack", "U1") == "user_paired"


# ── the allowlist on a live adapter connection ──────────────────────────────
# An adapter WebSocket outlives any number of pairing changes, so the owner set
# the allowlist is checked against has to be read per message, not per
# connection. `_verified_identity` already re-reads the trust level per message;
# these assert the admission decision keeps up with it.


class _RecordingBridge:
    """Stands in for S16. Records whatever GLC decided to forward."""

    def __init__(self) -> None:
        self.messages: list[ChannelMessage] = []

    async def handle(self, message: ChannelMessage) -> ChannelReply:
        self.messages.append(message)
        return ChannelReply(
            channel=message.channel,
            channel_user_id=message.channel_user_id,
            text=f"S16 answered {message.text}",
        )


def _envelope(text: str) -> str:
    return ChannelMessage(
        channel="webui",
        channel_user_id="42",
        user_handle="owner",
        text=text,
        trust_level="owner_paired",
        arrived_at=datetime.now(UTC),
    ).model_dump_json()


def test_revoking_an_owner_takes_effect_on_a_live_connection(app_client, install_token):
    """Revocation is a control; it has to hold without an adapter restart."""
    pairings = get_pairing_store()
    pairings.force_pair_owner("webui", "42", "owner")
    bridge = _RecordingBridge()
    app_client.app.state.agent_bridge = bridge

    with app_client.websocket_connect(f"/v1/channels/webui?token={install_token}") as ws:
        ws.send_text(_envelope("before revocation"))
        assert ws.receive_json()["text"] == "S16 answered before revocation"

        assert pairings.revoke("webui", "42") is True

        ws.send_text(_envelope("after revocation"))
        refused = ws.receive_json()

    assert refused["error"].startswith("dropped:")
    assert [m.text for m in bridge.messages] == ["before revocation"]


def test_pairing_an_owner_takes_effect_on_a_live_connection(app_client, install_token):
    """The mirror case: a fresh pairing must not wait for a reconnect either."""
    bridge = _RecordingBridge()
    app_client.app.state.agent_bridge = bridge

    with app_client.websocket_connect(f"/v1/channels/webui?token={install_token}") as ws:
        ws.send_text(_envelope("before pairing"))
        assert ws.receive_json()["error"].startswith("dropped:")

        get_pairing_store().force_pair_owner("webui", "42", "owner")

        ws.send_text(_envelope("after pairing"))
        assert ws.receive_json()["text"] == "S16 answered after pairing"

    assert [m.text for m in bridge.messages] == ["after pairing"]
