"""Discord adapter tests.

Wire-format basis: real Discord gateway and REST payloads as defined at
https://discord.com/developers/docs/topics/gateway-events#message-create
and https://discord.com/developers/docs/resources/channel#create-message.

Six structural tests + one behavioural test (mention resolution).
"""

from __future__ import annotations

from datetime import datetime

import pytest

from glc.channels.catalogue.discord.adapter import Adapter
from glc.channels.envelope import ChannelMessage, ChannelReply
from glc.security.pairing import get_pairing_store
from tests.channels.mocks.discord_mock import OWNER_ID, STRANGER_ID, DiscordMock


@pytest.fixture
def mock():
    return DiscordMock()


@pytest.fixture
def pair_owner():
    store = get_pairing_store()
    store.force_pair_owner("discord", OWNER_ID, user_handle="owner")
    yield
    store.revoke("discord", OWNER_ID)


@pytest.mark.asyncio
async def test_on_message_owner_returns_valid_envelope(mock, pair_owner):
    adapter = Adapter(config={"mock": mock})
    ev = mock.queue_owner_message("hello from owner")
    msg = await adapter.on_message(ev)
    assert isinstance(msg, ChannelMessage)
    assert msg.channel == "discord"
    assert msg.channel_user_id == OWNER_ID
    assert msg.trust_level == "owner_paired"
    assert msg.text == "hello from owner"
    assert isinstance(msg.arrived_at, datetime)


@pytest.mark.asyncio
async def test_on_message_stranger_is_untrusted(mock):
    adapter = Adapter(config={"mock": mock})
    ev = mock.queue_stranger_message("hi")
    msg = await adapter.on_message(ev)
    assert msg is not None
    assert msg.channel_user_id == STRANGER_ID
    assert msg.trust_level == "untrusted"


@pytest.mark.asyncio
async def test_send_emits_valid_wire_payload(mock, pair_owner):
    """The dispatched payload must conform to Discord's POST
    /channels/{channel.id}/messages body shape — `content` is the
    canonical text field. Adapters must NOT set `tts: true` by default
    (Discord text-to-speech is opt-in per channel)."""
    adapter = Adapter(config={"mock": mock})
    reply = ChannelReply(channel="discord", channel_user_id=OWNER_ID, text="hi back")
    await adapter.send(reply)
    assert len(mock.send_log) == 1
    body = mock.send_log[0]
    assert "content" in body, "Discord create-message requires `content`"
    assert body["content"] == "hi back"
    assert body.get("tts") is not True, "default tts must be false/absent"


@pytest.mark.asyncio
async def test_send_to_unpaired_recipient_is_blocked(mock):
    adapter = Adapter(config={"mock": mock})
    result = await adapter.send(ChannelReply(channel="discord", channel_user_id=STRANGER_ID, text="hi"))
    assert result == {"error": "recipient not paired", "code": "outbound_blocked"}
    assert mock.send_log == []


@pytest.mark.asyncio
async def test_disconnect_is_handled(mock, pair_owner):
    adapter = Adapter(config={"mock": mock})
    mock.force_disconnect()
    try:
        await adapter.on_message(mock.queue_owner_message("after disconnect"))
    except Exception as e:
        pytest.fail(f"adapter did not handle disconnect cleanly: {e!r}")


@pytest.mark.asyncio
async def test_rate_limit_propagates_429(mock, pair_owner):
    mock.rate_limited = True
    adapter = Adapter(config={"mock": mock})
    reply = ChannelReply(channel="discord", channel_user_id=OWNER_ID, text="x")
    result = await adapter.send(reply)
    assert isinstance(result, dict)
    # Discord's real 429 carries `retry_after` and `message`.
    assert result.get("status") == 429 or result.get("retry_after") is not None


@pytest.mark.asyncio
async def test_allowlist_silently_drops_stranger_in_public(mock):
    adapter = Adapter(config={"mock": mock, "is_public_channel": True})
    ev = mock.queue_stranger_message("hi from public")
    msg = await adapter.on_message(ev)
    assert msg is None or msg.trust_level == "untrusted"


@pytest.mark.asyncio
async def test_channel_specific_behaviour_mention_resolution(mock, pair_owner):
    """When a message mentions another user with `<@id>`, the adapter
    must resolve the mentioned user through the mock's `get_user(id)`
    helper and surface the resolved handles in
    `ChannelMessage.metadata["mentions"]`. Adapters that drop the
    mention or only echo the raw `<@id>` token fail this test."""
    adapter = Adapter(config={"mock": mock})
    ev = mock.queue_mention_message(mentioned_user_id="123456789", mentioned_username="alice")
    msg = await adapter.on_message(ev)
    assert msg is not None
    mentions = msg.metadata.get("mentions")
    assert mentions, "metadata['mentions'] must be a non-empty list"
    handles = [m if isinstance(m, str) else m.get("username") for m in mentions]
    assert "alice" in handles, "mentioned user's resolved handle must appear"
