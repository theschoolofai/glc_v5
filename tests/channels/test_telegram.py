"""Telegram adapter tests.

Six structural tests (locked by name) plus one channel-specific
behavioural test. The behavioural test is the load-bearing rubric for
graded submission; the structural tests are the envelope contract.

Wire-format basis: real Telegram Bot API payloads as defined at
https://core.telegram.org/bots/api — `getUpdates` for inbound,
`sendMessage` for outbound, `getFile` for the photo-attachment path.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from glc.channels.catalogue.telegram.adapter import Adapter
from glc.channels.envelope import ChannelMessage, ChannelReply
from glc.security.pairing import get_pairing_store
from tests.channels.mocks.telegram_mock import OWNER_ID, STRANGER_ID, TelegramMock


@pytest.fixture
def mock():
    return TelegramMock()


@pytest.fixture
def pair_owner():
    store = get_pairing_store()
    store.force_pair_owner("telegram", OWNER_ID, user_handle="owner")
    yield
    store.revoke("telegram", OWNER_ID)


# ── Structural tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_message_owner_returns_valid_envelope(mock, pair_owner):
    adapter = Adapter(config={"mock": mock})
    update = mock.queue_owner_message("hello from owner")
    msg = await adapter.on_message(update)
    assert isinstance(msg, ChannelMessage)
    assert msg.channel == "telegram"
    assert msg.channel_user_id == OWNER_ID
    assert msg.trust_level == "owner_paired"
    assert msg.text == "hello from owner"
    assert isinstance(msg.arrived_at, datetime)


@pytest.mark.asyncio
async def test_on_message_stranger_is_untrusted(mock):
    adapter = Adapter(config={"mock": mock})
    update = mock.queue_stranger_message("hi")
    msg = await adapter.on_message(update)
    assert msg is not None
    assert msg.channel_user_id == STRANGER_ID
    assert msg.trust_level == "untrusted"


@pytest.mark.asyncio
async def test_send_emits_valid_wire_payload(mock, pair_owner):
    """The dispatched payload must conform to Telegram's sendMessage
    request shape — `chat_id` (int or str) and `text` (str), not
    arbitrary JSON the adapter invented."""
    adapter = Adapter(config={"mock": mock})
    reply = ChannelReply(channel="telegram", channel_user_id=OWNER_ID, text="hi back")
    await adapter.send(reply)
    assert len(mock.send_log) == 1
    body = mock.send_log[0]
    assert "chat_id" in body, "Telegram sendMessage requires chat_id"
    assert "text" in body, "Telegram sendMessage requires text"
    assert body["text"] == "hi back"
    assert str(body["chat_id"]) == OWNER_ID
    assert "parse_mode" not in body, (
        "agent replies are plain text; MarkdownV2 rejects unescaped '!' and similar"
    )


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
    reply = ChannelReply(channel="telegram", channel_user_id=OWNER_ID, text="x")
    result = await adapter.send(reply)
    # Telegram's real shape: {"ok": False, "error_code": 429, "parameters": {"retry_after": N}}.
    assert isinstance(result, dict)
    assert result.get("error_code") == 429 or result.get("status") == 429


@pytest.mark.asyncio
async def test_allowlist_silently_drops_stranger_in_public(mock):
    adapter = Adapter(config={"mock": mock, "is_public_channel": True})
    update = mock.queue_stranger_message("hi from public group")
    msg = await adapter.on_message(update)
    assert msg is None or msg.trust_level == "untrusted"


# ── Channel-specific behavioural test ───────────────────────────────


@pytest.mark.asyncio
async def test_channel_specific_behaviour_photo_attachment(mock, pair_owner):
    """A Telegram photo Update requires two steps. First, the adapter
    parses the `photo` array (an array of PhotoSize objects with
    `file_id`s for each rendered size). Second, it calls
    `getFile(file_id)` to resolve the largest size to a `file_path`.
    The Attachment.ref must encode the resolved file_path — adapters
    that store the raw file_id without resolving fail this test.

    https://core.telegram.org/bots/api#photosize
    https://core.telegram.org/bots/api#getfile"""
    adapter = Adapter(config={"mock": mock})
    update = mock.queue_photo_message(file_id="AgADBAADREALPHOTO")
    msg = await adapter.on_message(update)
    assert msg is not None
    assert isinstance(msg, ChannelMessage)
    assert len(msg.attachments) >= 1, "photo Update must produce at least one Attachment"
    img = next((a for a in msg.attachments if a.kind == "image"), None)
    assert img is not None, "photo Attachment must have kind='image'"
    assert "photos/file_AgADBAADREALPHOTO" in img.ref, (
        "Attachment.ref should encode the resolved file_path from getFile, not the raw file_id"
    )
