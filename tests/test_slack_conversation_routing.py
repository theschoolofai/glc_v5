"""A Slack reply must reach the conversation its message arrived in.

`ChannelReply` carries no metadata field and forbids extras, so the
`slack_channel_id` the adapter records on the way in cannot travel back out.
Without somewhere to keep it, `send()` fell through to a literal `"C01CHAN"` -
the CHANNEL_ID constant from the test mock - on every production call.

These tests construct the adapter with no config at all, which is the production
path. Every test in the shipped `tests/channels/test_slack.py` passes
`config={"mock": mock}`, so the branch that reads a real conversation id is the
only one they ever exercise.
"""

from __future__ import annotations

from glc.channels.catalogue.slack.adapter import Adapter as SlackAdapter
from glc.channels.envelope import ChannelReply


def _event(user: str, channel: str) -> dict:
    return {"event": {"type": "message", "user": user, "text": "hello",
                      "channel": channel, "ts": "1700000000.000001"}}


class TestConversationRouting:
    async def test_reply_goes_to_the_conversation_the_message_came_from(self) -> None:
        adapter = SlackAdapter()
        await adapter.on_message(_event("U123ABC", "D999REAL"))

        body = await adapter.send(ChannelReply(channel="slack",
                                               channel_user_id="U123ABC",
                                               text="answer"))
        assert body["channel"] == "D999REAL"
        assert body["channel"] != "C01CHAN", "the placeholder must not reach Slack"

    async def test_each_sender_keeps_their_own_conversation(self) -> None:
        adapter = SlackAdapter()
        await adapter.on_message(_event("U111", "D111"))
        await adapter.on_message(_event("U222", "D222"))

        first = await adapter.send(ChannelReply(channel="slack", channel_user_id="U111", text="a"))
        second = await adapter.send(ChannelReply(channel="slack", channel_user_id="U222", text="b"))
        assert first["channel"] == "D111"
        assert second["channel"] == "D222"

    async def test_an_unknown_sender_falls_back_rather_than_failing(self) -> None:
        """No remembered conversation behaves exactly as before the fix.

        Deliberately a fallback and not a raise: the change adds a working path
        without introducing a new failure mode for callers that never had one.
        """
        adapter = SlackAdapter()
        body = await adapter.send(ChannelReply(channel="slack",
                                               channel_user_id="UNEVERSEEN",
                                               text="answer"))
        assert body["channel"] == "C01CHAN"
