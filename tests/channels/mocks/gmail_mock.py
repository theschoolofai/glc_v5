"""Mock-API fake for Gmail with Pub/Sub push.

Wire-format source:
  https://developers.google.com/gmail/api/guides/push
  https://developers.google.com/gmail/api/reference/rest/v1/users.history/list
  https://developers.google.com/gmail/api/reference/rest/v1/users.messages/get
  https://developers.google.com/gmail/api/reference/rest/v1/users.messages/send

Inbound: a Pub/Sub push POST body. `message.data` is base64 of
`{"emailAddress":"...","historyId":N}`. The adapter then calls
`users.history.list(startHistoryId=N)` to get the new message ids,
then `users.messages.get(id, format="full")` for each, then parses the
multipart MIME body to extract text/plain.

Outbound: `users.messages.send` body `{"raw": "<base64url-rfc822>"}`.

Helpers
-------
queue_owner_message(text)              → registers an inbound RFC 822
                                         message from owner and emits
                                         a Pub/Sub push for its history_id
queue_stranger_message(text)           → same, from stranger
register_message(message_id, raw)      → seed the messages store for
                                         direct lookup
history_list(start_history_id)         → mirrors users.history.list
messages_get(message_id)               → mirrors users.messages.get
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Any

OWNER_EMAIL = "owner@example.com"
STRANGER_EMAIL = "stranger@example.com"
OWNER_ID = OWNER_EMAIL
STRANGER_ID = STRANGER_EMAIL

BOT_EMAIL = "bot@example.com"


def _build_multipart(*, from_addr: str, to_addr: str, subject: str, text_body: str, html_body: str) -> bytes:
    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg["Date"] = "Wed, 17 Jun 2026 12:00:00 +0000"
    msg["Message-ID"] = f"<{from_addr.replace('@', '-at-')}@mail.example>"
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")
    return bytes(msg)


def _pubsub_push(*, email_address: str, history_id: int, message_id: str) -> dict[str, Any]:
    inner = {"emailAddress": email_address, "historyId": history_id}
    data = base64.b64encode(json.dumps(inner).encode()).decode()
    return {
        "message": {
            "data": data,
            "messageId": f"pubsub-{message_id}",
            "publishTime": "2026-06-17T12:00:00Z",
            "attributes": {"emailAddress": email_address},
        },
        "subscription": "projects/glc/subscriptions/gmail-watch",
    }


@dataclass
class GmailMock:
    inbound_events: list[dict[str, Any]] = field(default_factory=list)
    send_log: list[dict[str, Any]] = field(default_factory=list)
    rate_limited: bool = False
    _disconnect_pending: bool = False
    _next_history: int = 1000
    _next_msg: int = 5000
    _messages: dict[str, dict[str, Any]] = field(default_factory=dict)
    _history: dict[int, list[str]] = field(default_factory=dict)

    def _h(self) -> int:
        self._next_history += 1
        return self._next_history

    def _m(self) -> str:
        self._next_msg += 1
        return f"msg-{self._next_msg}"

    def register_message(self, message_id: str, raw_bytes: bytes, from_addr: str, history_id: int) -> None:
        # Gmail's API stores `raw` as base64url (no padding).
        raw_b64url = base64.urlsafe_b64encode(raw_bytes).decode().rstrip("=")
        self._messages[message_id] = {
            "id": message_id,
            "threadId": "thread-" + message_id,
            "historyId": str(history_id),
            "snippet": "...",
            "raw": raw_b64url,
        }
        self._history.setdefault(history_id, []).append(message_id)

    def _seed_message(self, *, from_addr: str, text: str) -> tuple[str, int]:
        msg_id = self._m()
        history_id = self._h()
        raw = _build_multipart(
            from_addr=from_addr,
            to_addr=BOT_EMAIL,
            subject="ping",
            text_body=text,
            html_body=f"<p>{text}</p><p>--<br>(html part the adapter must ignore)</p>",
        )
        self.register_message(msg_id, raw, from_addr, history_id)
        return msg_id, history_id

    def queue_owner_message(self, text: str = "hello") -> dict[str, Any]:
        msg_id, history_id = self._seed_message(from_addr=OWNER_EMAIL, text=text)
        ev = _pubsub_push(email_address=BOT_EMAIL, history_id=history_id, message_id=msg_id)
        self.inbound_events.append(ev)
        return ev

    def queue_stranger_message(self, text: str = "ping") -> dict[str, Any]:
        msg_id, history_id = self._seed_message(from_addr=STRANGER_EMAIL, text=text)
        ev = _pubsub_push(email_address=BOT_EMAIL, history_id=history_id, message_id=msg_id)
        self.inbound_events.append(ev)
        return ev

    def history_list(self, start_history_id: int) -> dict[str, Any]:
        new_msgs = self._history.get(start_history_id) or []
        return {
            "history": [
                {
                    "id": str(start_history_id),
                    "messagesAdded": [{"message": {"id": m, "threadId": "thread-" + m}} for m in new_msgs],
                }
            ],
            "historyId": str(start_history_id),
        }

    def messages_get(self, message_id: str) -> dict[str, Any]:
        msg = self._messages.get(message_id)
        if msg is None:
            raise KeyError(f"unknown message id: {message_id}")
        return dict(msg)

    async def send(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.rate_limited:
            return {"error": {"code": 429, "message": "User-rate limit exceeded"}, "status": 429}
        self.send_log.append(payload)
        return {"id": self._m(), "threadId": "thread-out", "labelIds": ["SENT"]}

    def force_disconnect(self) -> None:
        self._disconnect_pending = True

    def pop_disconnect(self) -> bool:
        was = self._disconnect_pending
        self._disconnect_pending = False
        return was
