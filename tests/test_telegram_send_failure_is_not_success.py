"""A send that did not happen must not return something shaped like a receipt.

`Adapter.send` ends with:

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        return payload

The caller gets back the request it asked to have sent. `send()`'s contract is to
"translate a ChannelReply into a native wire-format payload and dispatch it,
returning whatever the native API returns" - so a payload-shaped dict is
indistinguishable from a successful dispatch. The gateway's
`/v1/channels/{name}/send` then answers `200 {"accepted": true}`, and the message
was never sent.

Observed 10 Aug 2026: the gateway process was started before TELEGRAM_BOT_TOKEN
was present in its environment. Every send returned

    200 {"accepted":true,"channel":"telegram",
         "adapter_result":{"chat_id":...,"text":"...","parse_mode":"MarkdownV2"}}

and nothing arrived. Nothing in that response says so.

An unconfigured channel is a configuration error, and it is exactly the kind that
must be loud: Section 16 asks that silence and death never look the same, and this
is the delivery layer's version of that. A missing credential should fail closed
and visibly, not be reported as delivery.
"""

from __future__ import annotations

import asyncio

import pytest

from glc.channels.catalogue.telegram.adapter import Adapter
from glc.channels.envelope import ChannelReply

REPLY = ChannelReply(channel="telegram", channel_user_id="4242", text="did this arrive?")


def test_missing_token_raises_instead_of_returning_the_payload(monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    adapter = Adapter()
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        asyncio.run(adapter.send(REPLY))


def test_missing_token_never_returns_a_dispatch_shaped_result(monkeypatch) -> None:
    """The precise failure: the return value looked like a sent message."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    adapter = Adapter()
    try:
        result = asyncio.run(adapter.send(REPLY))
    except RuntimeError:
        return  # correct: refused loudly
    pytest.fail(
        "send() returned "
        f"{result!r} for a message it never dispatched; a caller cannot tell this "
        "from a delivered message, and the gateway reports accepted: true"
    )


def test_an_empty_token_is_treated_as_missing(monkeypatch) -> None:
    """A blank env var is the common shape of "not configured"."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "   ")
    adapter = Adapter()
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        asyncio.run(adapter.send(REPLY))


def test_mock_mode_is_untouched(monkeypatch) -> None:
    """Configured tests must keep working: only the unconfigured path changes."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    class _Mock:
        def __init__(self) -> None:
            self.sent: list[dict] = []

        async def send(self, payload: dict) -> dict:
            self.sent.append(payload)
            return {"ok": True, "result": {"message_id": 7}}

    mock = _Mock()
    result = asyncio.run(Adapter(config={"mock": mock}).send(REPLY))
    assert result == {"ok": True, "result": {"message_id": 7}}
    assert mock.sent and mock.sent[0]["chat_id"] == 4242
