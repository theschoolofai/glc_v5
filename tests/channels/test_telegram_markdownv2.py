"""Regression tests: the Telegram adapter declares parse_mode MarkdownV2 but must
not hand Telegram text that MarkdownV2 rejects.

Before the fix, `send()` set `parse_mode: "MarkdownV2"` and passed the reply text
through untouched. Telegram then answered 400 for any text containing one of its
reserved characters:

    Bad Request: can't parse entities: Character '.' is reserved and must be
    escaped with the preceding '\\'

Assistant replies are ordinary prose, so nearly every one contains '.' or '-'.
Verified live against api.telegram.org on 9 Aug 2026:
  "Outbound check three works"        -> delivered, message_id=3
  "This sentence ends with a period." -> 400, not delivered
"""

from __future__ import annotations

import asyncio

import pytest

from glc.channels.catalogue.telegram.adapter import Adapter
from glc.channels.envelope import ChannelReply

# Telegram's documented MarkdownV2 reserved set.
RESERVED = r"_*[]()~`>#+-=|{}.!"


class _CapturingMock:
    """Stands in for the Bot API and records the payload it was handed."""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def send(self, payload: dict) -> dict:
        self.payloads.append(payload)
        return {"ok": True, "result": {"message_id": 1}}


def _send(text: str) -> dict:
    mock = _CapturingMock()
    adapter = Adapter(config={"mock": mock})
    asyncio.run(adapter.send(ChannelReply(channel="telegram", channel_user_id="42", text=text)))
    return mock.payloads[-1]


@pytest.mark.parametrize("char", list(RESERVED))
def test_every_reserved_character_is_escaped(char: str) -> None:
    """Each reserved character must reach Telegram backslash-escaped."""
    payload = _send(f"a{char}b")
    assert payload["text"] == f"a\\{char}b", (
        f"reserved character {char!r} was sent unescaped as {payload['text']!r}; "
        "Telegram rejects this with 400 can't parse entities"
    )


def test_ordinary_prose_survives() -> None:
    """The exact shape of message an assistant sends all day."""
    payload = _send("This sentence ends with a period.")
    assert payload["text"] == "This sentence ends with a period\\."


def test_hyphen_in_prose_is_escaped() -> None:
    """The first failure seen live was a hyphen, not a period."""
    payload = _send("S16 outbound check 2 - the outbound path is live.")
    assert "\\-" in payload["text"]
    assert payload["text"].endswith("live\\.")


def test_parse_mode_still_declared() -> None:
    """The fix escapes the text rather than silently dropping formatting support."""
    assert _send("hello").get("parse_mode") == "MarkdownV2"


def test_backslash_itself_is_escaped_first() -> None:
    """A literal backslash must not be able to escape the escape."""
    payload = _send(r"a\b")
    assert payload["text"] == r"a\\b"


def test_empty_and_none_text_do_not_raise() -> None:
    assert _send("")["text"] == ""
    mock = _CapturingMock()
    adapter = Adapter(config={"mock": mock})
    asyncio.run(adapter.send(ChannelReply(channel="telegram", channel_user_id="42", text=None)))
    assert mock.payloads[-1]["text"] == ""
