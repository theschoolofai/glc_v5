"""Reading a mailbox must not change it.

`ImapConnection.fetch_unseen` fetches each message with:

    self._conn.fetch(str(uid), "(RFC822)")

Under RFC 3501, `FETCH ... RFC822` is a data item that implicitly sets the
`\\Seen` flag on the server. `BODY.PEEK[]` exists precisely to read a message
without doing that. So merely polling the inbox marks every message it reads as
read, on the human's real mailbox.

Three consequences:

1. A visible side effect nobody asked for. Section 15 says to start read-only and
   "let it tell you what it would have done"; a poller that silently marks a
   person's unread mail as read is not read-only.
2. `mark_seen()` becomes unreachable as an intentional decision. The class offers
   an explicit way to flag a message, implying the caller chooses - but the flag
   was already set by the read, so the choice does not exist.
3. Any caller-level "do not touch the server" option is silently defeated,
   because the flag is set inside the fetch it depends on.

Observed 10 Aug 2026 against a live Gmail mailbox: a single sweep with server-side
marking deliberately disabled still left the fetched message flagged, and it no
longer matched `SEARCH UNSEEN` on the next sweep.
"""

from __future__ import annotations

from glc.channels.catalogue.imap.connection import ImapConnection

RAW = b"From: someone@example.com\r\nSubject: hello\r\n\r\nbody\r\n"


class _FakeImap:
    """Records the commands issued, and which data items were requested."""

    def __init__(self) -> None:
        self.fetch_items: list[str] = []
        self.store_calls: list[tuple[str, str, str]] = []

    def search(self, charset, *criteria):
        return "OK", [b"7"]

    def fetch(self, message_set, message_parts):
        self.fetch_items.append(message_parts)
        return "OK", [(b"7 (RFC822 {%d}" % len(RAW), RAW), b")"]

    def store(self, message_set, command, flags):
        self.store_calls.append((message_set, command, flags))
        return "OK", [b""]


def _connection() -> tuple[ImapConnection, _FakeImap]:
    conn = ImapConnection(host="imap.example.com", port=993, user="me", password="pw")
    fake = _FakeImap()
    conn._conn = fake
    return conn, fake


def test_fetch_uses_peek_so_the_server_flag_is_untouched() -> None:
    conn, fake = _connection()
    conn.fetch_unseen()

    assert fake.fetch_items, "no FETCH was issued"
    requested = " ".join(fake.fetch_items).upper()
    assert "PEEK" in requested, (
        f"fetch requested {requested!r}; a bare RFC822 fetch implicitly sets "
        "\\Seen on the server, so polling silently marks the user's mail read"
    )


def test_fetch_does_not_request_the_seen_setting_data_item() -> None:
    conn, fake = _connection()
    conn.fetch_unseen()
    requested = " ".join(fake.fetch_items).upper()
    assert "RFC822" not in requested.replace("RFC822.SIZE", ""), (
        f"fetch requested {requested!r}, which RFC 3501 defines as implicitly "
        "setting \\Seen"
    )


def test_fetch_still_returns_the_message_bytes() -> None:
    """The fix must not change what callers get back."""
    conn, _ = _connection()
    messages = conn.fetch_unseen()
    assert len(messages) == 1
    assert messages[0]["uid"] == 7
    assert messages[0]["raw"] == RAW


def test_marking_read_remains_an_explicit_caller_decision() -> None:
    conn, fake = _connection()
    conn.fetch_unseen()
    assert fake.store_calls == [], "fetch_unseen must not flag anything by itself"

    conn.mark_seen(7)
    assert fake.store_calls == [("7", "+FLAGS", "\\Seen")]
