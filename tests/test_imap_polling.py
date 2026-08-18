"""A long-lived IMAP poller must see mail that arrives after it connected.

The failure this pins was silent and inverted, which is what made it hard to
believe. Mail sent while the poller was *stopped* arrived the moment it next
started, because connecting performs a fresh SELECT. Mail sent while it was
*running* was never delivered at all. So every quick test passed and the only
configuration that failed was the one it runs in.

The cause is that a selected mailbox does not refresh itself. A server reports
newly arrived messages with an untagged EXISTS, and issues one during a command
that permits it. Polling SEARCH on a stale selection keeps returning the set
that existed at SELECT time.
"""

from __future__ import annotations

import pytest

from glc.channels.catalogue.imap.connection import ImapConnection


class FakeIMAP:
    """An imaplib-shaped double whose view only updates on NOOP.

    That is the behaviour under test: `delivered` is what the server holds,
    `_visible` is what this connection has been told about, and only NOOP
    moves one to the other.
    """

    def __init__(self) -> None:
        self.delivered: dict[int, bytes] = {}
        self._visible: dict[int, bytes] = {}
        self.calls: list[str] = []

    def deliver(self, uid: int, raw: bytes) -> None:
        """New mail arrives at the server, after SELECT."""
        self.delivered[uid] = raw

    def noop(self):
        self.calls.append("noop")
        self._visible = dict(self.delivered)
        return "OK", [b""]

    def search(self, charset, *criteria):
        self.calls.append("search")
        return "OK", [b" ".join(str(uid).encode() for uid in sorted(self._visible))]

    def fetch(self, message_set, parts):
        self.calls.append("fetch")
        uid = int(message_set)
        return "OK", [(b"header", self._visible[uid])]

    def store(self, message_set, command, flags):
        self.calls.append("store")
        return "OK", [b""]


@pytest.fixture
def connection():
    link = ImapConnection("imap.invalid", 993, "user", "password")
    fake = FakeIMAP()
    link._conn = fake
    return link, fake


def test_mail_arriving_after_connect_is_delivered(connection) -> None:
    link, fake = connection
    fake.noop()                      # the SELECT-time view: empty
    fake.deliver(41, b"From: someone@example.invalid\r\n\r\nhello")

    messages = link.fetch_unseen()

    assert [m["uid"] for m in messages] == [41], (
        "mail that arrived after SELECT must still be found; without a refresh "
        "the poller is blind for the life of the connection")
    assert messages[0]["raw"] == b"From: someone@example.invalid\r\n\r\nhello"


def test_the_refresh_happens_before_the_search(connection) -> None:
    link, fake = connection
    fake.deliver(7, b"raw")
    link.fetch_unseen()
    assert fake.calls.index("noop") < fake.calls.index("search"), \
        "refreshing after the search would still miss this poll's mail"


def test_an_empty_mailbox_is_not_an_error(connection) -> None:
    link, fake = connection
    assert link.fetch_unseen() == []


def test_a_dead_connection_raises_rather_than_reporting_no_mail(connection) -> None:
    """Silence and emptiness must not look the same to the caller.

    The poll loop reconnects when fetching raises. If a broken connection
    returned "no messages" instead, the bridge would sit in a tight healthy
    loop over a socket that will never deliver anything again.
    """
    link, fake = connection

    def broken():
        raise OSError("connection reset")

    fake.noop = broken
    with pytest.raises(OSError):
        link.fetch_unseen()


def test_mock_mode_is_unaffected() -> None:
    class Mock:
        inbound_events = [{"uid": 1, "raw": b"x"}]

    link = ImapConnection("imap.invalid", 993, "u", "p", mock=Mock())
    assert link.fetch_unseen() == [{"uid": 1, "raw": b"x"}]
