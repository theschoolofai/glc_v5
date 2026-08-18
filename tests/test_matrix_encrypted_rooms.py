"""An unreadable Matrix message must be reported, not silently discarded.

Element enables end-to-end encryption on direct messages by default. In an
encrypted room a user's message arrives as `m.room.encrypted`, not
`m.room.message`, so the adapter returns None and the message is dropped.

Nothing anywhere reported it. Four messages were delivered to the homeserver and
none reached the bridge; the user saw a bot that was plainly online and simply
never answered. A silent skip and a dead process look identical from outside,
which is the whole failure mode this warning exists to break.

Decrypting is out of scope: that needs a full olm/megolm session, device
verification and key storage. Saying so out loud costs one log line and turns an
invisible failure into a diagnosable one.
"""

from __future__ import annotations

from glc.channels.catalogue.matrix.adapter import Adapter as MatrixAdapter


def _sync(event: dict) -> dict:
    return {"rooms": {"join": {"!room:example.org": {"timeline": {"events": [event]}}}}}


class TestEncryptedRooms:
    async def test_an_encrypted_event_is_ignored_but_reported(self, caplog) -> None:
        adapter = MatrixAdapter()
        sync = _sync({"type": "m.room.encrypted", "sender": "@someone:example.org",
                      "content": {"algorithm": "m.megolm.v1.aes-sha2"}})

        with caplog.at_level("WARNING"):
            assert await adapter.on_message(sync) is None

        assert any("encrypted" in record.message.lower() for record in caplog.records), \
            "an unreadable message must not be discarded silently"

    async def test_a_plaintext_message_is_still_delivered(self, caplog) -> None:
        # The warning must not fire on the path that works.
        adapter = MatrixAdapter()
        sync = _sync({"type": "m.room.message", "sender": "@someone:example.org",
                      "content": {"msgtype": "m.text", "body": "hello"},
                      "event_id": "$1", "origin_server_ts": 1})

        with caplog.at_level("WARNING"):
            envelope = await adapter.on_message(sync)

        assert envelope is not None
        assert envelope.text == "hello"
        assert not any("encrypted" in record.message.lower() for record in caplog.records)
