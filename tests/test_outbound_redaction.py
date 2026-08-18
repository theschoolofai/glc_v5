"""A reply must not carry the operator's filesystem out to a channel.

The leak this pins was real and was delivered: a Telegram reply to the owner
contained the full path of a sandbox file, including the Windows username. A
channel is a third-party service and a sent message cannot be recalled, so the
disclosure is permanent the moment it is delivered.

The tests assert at the envelope, not at an adapter, because that is where the
guarantee has to hold: every outbound route in the gateway converges on
``ChannelReply``.
"""

from __future__ import annotations

import pytest

from glc.channels.envelope import ChannelMessage, ChannelReply
from glc.security.redaction import redact_local_paths

LEAKED = (
    "Based on your reminders list "
    "(`file://C:\\Users\\alice\\GitHub\\executive_assistant\\event_based_agents"
    "\\sandbox\\reminders.txt`) and today's date (Thursday, August 13, 2026), "
    "here is what is due this week"
)


class TestTheRealLeak:
    """The exact message that was delivered, in the exact shape it took."""

    def test_the_username_does_not_survive(self) -> None:
        cleaned = redact_local_paths(LEAKED)
        assert "alice" not in cleaned
        assert "C:\\" not in cleaned
        assert "file://" not in cleaned

    def test_the_sentence_still_reads(self) -> None:
        # A hole where the path was would be worse than the path for anyone
        # trying to follow the answer, so the basename is kept.
        cleaned = redact_local_paths(LEAKED)
        assert "reminders.txt" in cleaned
        assert cleaned.startswith("Based on your reminders list")
        assert "due this week" in cleaned

    def test_it_is_stripped_by_constructing_a_reply(self) -> None:
        # Not by calling the helper: the guarantee is that you cannot build an
        # outbound reply carrying a path, however you build it.
        reply = ChannelReply(channel="telegram", channel_user_id="8715135896", text=LEAKED)
        assert "alice" not in (reply.text or "")

    def test_it_is_stripped_when_parsed_from_json(self) -> None:
        # The agent bridge uses model_validate, and the proactive /send endpoint
        # parses a request body. Both must be covered, not just __init__.
        reply = ChannelReply.model_validate(
            {"channel": "telegram", "channel_user_id": "8715135896", "text": LEAKED})
        assert "alice" not in (reply.text or "")


class TestPathShapes:
    """Each shape a host path can take on the way out."""

    @pytest.mark.parametrize("text", [
        "see (file://C:\\Users\\alice\\sandbox\\reminders.txt) for details",
        "see `C:\\Users\\alice\\sandbox\\reminders.txt` for details",
        "see file:///C:/Users/alice/sandbox/reminders.txt for details",
        "see C:/Users/alice/sandbox/reminders.txt for details",
        "see /home/alice/sandbox/reminders.txt for details",
        "see /Users/alice/sandbox/reminders.txt for details",
        "see \\\\fileserver\\share\\reminders.txt for details",
    ])
    def test_no_username_survives(self, text: str) -> None:
        cleaned = redact_local_paths(text)
        assert "alice" not in cleaned, text
        assert cleaned.startswith("see ") and cleaned.endswith(" for details"), \
            "surrounding prose must be preserved"

    def test_several_paths_in_one_message(self) -> None:
        cleaned = redact_local_paths(
            "compared C:\\Users\\alice\\a.txt with /home/alice/b.txt")
        assert "alice" not in cleaned
        assert "a.txt" in cleaned and "b.txt" in cleaned


class TestItDoesNotMangleOrdinaryText:
    """A redactor that eats real content is worse than the leak it prevents."""

    def test_an_http_link_is_left_intact(self) -> None:
        # Deliberately a URL whose own path would match the POSIX-home pattern.
        text = "full report at https://example.com/Users/alice/report.html today"
        assert redact_local_paths(text) == text

    def test_an_https_link_with_a_windows_looking_segment_survives(self) -> None:
        text = "see https://example.com/docs/C:/legacy for the old scheme"
        assert redact_local_paths(text) == text

    @pytest.mark.parametrize("text", [
        "Ratio was 3:1, the meeting is at 10:30, see section C.",
        "Dentist appointment at 6:30pm with Dr Rao.",
        "The DevSummit talk is 30 minutes on 14 November in Bangalore.",
        "Reply with A: renew for twelve months, or B: six months at more rent.",
        "Use the art:abc123 handle, or the mxc://matrix.org/xyz reference.",
        # The agent now cites sandbox:// instead of file://. If the net ate that
        # too, the producer-side fix would have made answers worse, not safer.
        "Based on sandbox://reminders.txt, the dentist is today.",
        "Artifact written to artifact://run-1a2d/invite.ics for you.",
    ])
    def test_prose_is_untouched(self, text: str) -> None:
        assert redact_local_paths(text) == text

    def test_none_and_empty_pass_through(self) -> None:
        assert redact_local_paths(None) is None
        assert redact_local_paths("") == ""


class TestScope:
    """The net is on the way out, and only on the way out."""

    def test_an_inbound_message_is_not_rewritten(self) -> None:
        # An owner may legitimately send a path to the agent; rewriting what a
        # human typed would corrupt the request rather than protect anything.
        from datetime import UTC, datetime

        message = ChannelMessage(
            channel="telegram", channel_user_id="8715135896", user_handle="owner",
            text="please read C:\\Users\\alice\\notes.txt", trust_level="owner_paired",
            arrived_at=datetime.now(UTC))
        assert "alice" in (message.text or ""), \
            "inbound text is the user's own words and must arrive verbatim"
