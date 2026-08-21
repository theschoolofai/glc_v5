"""IMAP/SMTP adapter tests.

Wire-format basis: RFC 5322 message format + RFC 2045 MIME multipart.

Six structural tests + one behavioural test (PDF attachment extraction
to the artifact store). The rate-limit shape here uses an SMTP 421
(service unavailable) instead of HTTP 429 — that is the realistic
back-pressure signal SMTP servers emit.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from glc.channels.catalogue.imap.adapter import Adapter
from glc.channels.envelope import ChannelMessage, ChannelReply
from glc.security.pairing import get_pairing_store
from tests.channels.mocks.imap_mock import OWNER_ID, STRANGER_ID, ImapMock


@pytest.fixture
def mock():
    return ImapMock()


@pytest.fixture
def pair_owner():
    store = get_pairing_store()
    store.force_pair_owner("imap", OWNER_ID, user_handle="owner")
    yield
    store.revoke("imap", OWNER_ID)


@pytest.mark.asyncio
async def test_on_message_owner_returns_valid_envelope(mock, pair_owner):
    adapter = Adapter(config={"mock": mock})
    ev = mock.queue_owner_message("hello from owner")
    msg = await adapter.on_message(ev)
    assert isinstance(msg, ChannelMessage)
    assert msg.channel == "imap"
    assert msg.channel_user_id == OWNER_ID
    assert msg.trust_level == "owner_paired"
    assert "hello from owner" in (msg.text or "")
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
    """SMTP send shape: `{from, to, raw}`. `raw` is RFC 822 bytes — it
    must include valid `From`, `To`, and `Subject` headers, otherwise
    most SMTP relays will reject it as spam."""
    adapter = Adapter(config={"mock": mock})
    reply = ChannelReply(channel="imap", channel_user_id=OWNER_ID, text="hi back")
    await adapter.send(reply)
    assert len(mock.send_log) == 1
    body = mock.send_log[0]
    assert body.get("to") == OWNER_ID
    assert "raw" in body
    raw = body["raw"] if isinstance(body["raw"], bytes) else body["raw"].encode()
    assert b"From:" in raw
    assert b"To:" in raw
    assert b"Subject:" in raw
    assert b"hi back" in raw


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
    """SMTP signals back-pressure with 421 (service unavailable). The
    adapter surfaces this to callers as a structured error code."""
    mock.rate_limited = True
    adapter = Adapter(config={"mock": mock})
    reply = ChannelReply(channel="imap", channel_user_id=OWNER_ID, text="x")
    result = await adapter.send(reply)
    assert isinstance(result, dict)
    assert result.get("status") in (421, 429), "SMTP back-pressure uses 421; the adapter may normalise to 429"


@pytest.mark.asyncio
async def test_allowlist_silently_drops_stranger_in_public(mock):
    adapter = Adapter(config={"mock": mock, "is_public_channel": True})
    ev = mock.queue_stranger_message("hi from public")
    msg = await adapter.on_message(ev)
    assert msg is None or msg.trust_level == "untrusted"


@pytest.mark.asyncio
async def test_channel_specific_behaviour_pdf_attachment_to_artifact(mock, pair_owner):
    """A multipart/mixed message with a base64-encoded PDF attachment
    must be parsed into:
      - text body from the text/plain part
      - one Attachment of kind 'file' with mime 'application/pdf' and
        ref of the form `art:<sha>`
    The attachment bytes must land in the artifact store (the mock's
    `artifact_store` dict is the observable side).

    Adapters that emit the raw PDF bytes inline in ChannelMessage.text
    will flood the agent's context with binary garbage."""
    adapter = Adapter(config={"mock": mock})
    ev = mock.queue_pdf_attachment_message(body="see attached file")
    msg = await adapter.on_message(ev)
    assert msg is not None
    assert "see attached" in (msg.text or "")
    pdf = next((a for a in msg.attachments if a.kind == "file" and a.mime == "application/pdf"), None)
    assert pdf is not None, "PDF attachment must produce kind='file', mime='application/pdf'"
    assert pdf.ref.startswith("art:"), (
        "Attachment.ref must be an artifact handle, not raw bytes or a temp path"
    )
    sha = pdf.ref.removeprefix("art:")
    assert sha in mock.artifact_store, "PDF bytes must land in the artifact store"


def _raw(message_id: str, body: str, *, in_reply_to: str = "", references: str = "") -> bytes:
    headers = [f"From: {OWNER_ID}", "To: bot@example.com", "Subject: ping",
               f"Message-ID: {message_id}"]
    if in_reply_to:
        headers.append(f"In-Reply-To: {in_reply_to}")
    if references:
        headers.append(f"References: {references}")
    return ("\r\n".join(headers) + "\r\n\r\n" + body).encode()


@pytest.mark.asyncio
async def test_two_mails_in_one_conversation_share_a_thread_id(mock, pair_owner):
    """`thread_id` identified the message, not the conversation it belongs to.

    The adapter used `Message-ID`, which is unique per message, so every reply
    arrived as a brand new thread and nothing downstream could tell that two
    mails belong together. A reply answering a question the agent parked on
    therefore matched no waiting conversation and was read as a fresh request.

    RFC 5322 already carries the conversation: the first entry of `References`
    is its root, `In-Reply-To` is the parent when no chain was kept, and a mail
    that starts a thread is its own root.
    """
    adapter = Adapter(config={"mock": mock})

    opening = await adapter.on_message({"uid": 1, "raw": _raw("<root@example.com>", "may I?")})
    reply = await adapter.on_message({"uid": 2, "raw": _raw(
        "<later@example.com>", "yes, go ahead",
        in_reply_to="<root@example.com>", references="<root@example.com>")})

    assert opening.thread_id == "<root@example.com>", "a mail that starts a thread is its own root"
    assert reply.thread_id == opening.thread_id, "a reply belongs to the conversation it answers"
