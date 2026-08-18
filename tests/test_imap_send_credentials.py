"""The imap adapter must be able to send when nobody hands it a config.

`POST /v1/channels/{name}/send` builds its adapter with
`registry.instantiate(name)` and passes no config, which is the path an
autonomous run uses when it calls `send_channel_message`. The adapter read its
SMTP settings only from `self.config`, defaulting each to an empty string, so on
that path there was no host to connect to and the send could never succeed.

The failure was invisible from every other angle. Inbound polling worked, and
replies dispatched by the bridge worked, because the bridge constructs its own
adapter with credentials from the environment. Only the proactive path was dead.

Telegram already falls back to `TELEGRAM_BOT_TOKEN` for exactly this reason, so
the fix makes the two channels resolve credentials the same way instead of each
having its own rule.
"""

from __future__ import annotations

import pytest

from glc.channels.catalogue.imap.adapter import Adapter as ImapAdapter
from glc.channels.envelope import ChannelReply


class Recorder:
    """Stands in for SmtpSender, capturing how it was constructed."""

    made: list[dict] = []

    def __init__(self, **kwargs) -> None:
        Recorder.made.append(kwargs)

    def send(self, *, to: str, raw_bytes: bytes) -> dict:
        return {"status": 250, "to": to}


@pytest.fixture(autouse=True)
def _recorder(monkeypatch):
    Recorder.made = []
    monkeypatch.setattr("glc.channels.catalogue.imap.adapter.SmtpSender", Recorder)
    for name in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD",
                 "IMAP_USER", "IMAP_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    return Recorder


def _reply() -> ChannelReply:
    return ChannelReply(channel="imap", channel_user_id="client@example.invalid",
                        text="Monday at 12:00 works.")


class TestTheProactivePath:
    async def test_credentials_come_from_the_environment_when_no_config_is_given(
            self, monkeypatch) -> None:
        monkeypatch.setenv("SMTP_HOST", "smtp.example.invalid")
        monkeypatch.setenv("SMTP_PORT", "2525")
        monkeypatch.setenv("SMTP_USER", "bot@example.invalid")
        monkeypatch.setenv("SMTP_PASSWORD", "secret")

        await ImapAdapter().send(_reply())

        made = Recorder.made[-1]
        assert made["host"] == "smtp.example.invalid"
        assert made["port"] == 2525
        assert made["user"] == "bot@example.invalid"
        assert made["password"] == "secret"

    async def test_imap_credentials_serve_as_the_second_choice(self, monkeypatch) -> None:
        # One mailbox is read and sent from, so configuring it once is enough.
        monkeypatch.setenv("SMTP_HOST", "smtp.example.invalid")
        monkeypatch.setenv("IMAP_USER", "bot@example.invalid")
        monkeypatch.setenv("IMAP_PASSWORD", "secret")

        await ImapAdapter().send(_reply())

        made = Recorder.made[-1]
        assert made["user"] == "bot@example.invalid"
        assert made["password"] == "secret"

    async def test_the_port_defaults_when_nothing_says_otherwise(self, monkeypatch) -> None:
        monkeypatch.setenv("SMTP_HOST", "smtp.example.invalid")
        await ImapAdapter().send(_reply())
        assert Recorder.made[-1]["port"] == 587


class TestAnExplicitConfigStillWins:
    async def test_config_takes_precedence_over_the_environment(self, monkeypatch) -> None:
        """A bridge passes credentials explicitly; that must not change."""
        monkeypatch.setenv("SMTP_HOST", "environment.example.invalid")
        monkeypatch.setenv("SMTP_USER", "environment@example.invalid")

        adapter = ImapAdapter(config={"smtp_host": "configured.example.invalid",
                                      "smtp_port": 1025,
                                      "smtp_user": "configured@example.invalid",
                                      "smtp_password": "from-config"})
        await adapter.send(_reply())

        made = Recorder.made[-1]
        assert made["host"] == "configured.example.invalid"
        assert made["port"] == 1025
        assert made["user"] == "configured@example.invalid"
        assert made["password"] == "from-config"

    async def test_with_neither_it_is_empty_rather_than_wrong(self) -> None:
        # No host configured anywhere is a real deployment error; it must not
        # silently acquire one.
        await ImapAdapter().send(_reply())
        assert Recorder.made[-1]["host"] == ""
