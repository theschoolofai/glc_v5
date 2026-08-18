"""Stateless SMTP sender with STARTTLS and SMTP 421 → 429 normalisation.

Design: stateless (open-per-send)
----------------------------------
Zoho Mail (and most SMTP servers) silently close idle connections after
approximately 5 minutes. Keeping a persistent connection open in a
long-running process causes stale-socket errors on the first send after
an idle period. Opening a fresh SMTP session per send avoids this
entirely at negligible cost (TLS handshake ~50 ms on LAN, ~200 ms WAN).

SMTP back-pressure
------------------
SMTP servers signal transient unavailability with 4xx response codes.
421 ("Service not available, try later") is the canonical back-pressure
signal. This module normalises any SMTP 421 to the dict
    {"status": 429, "error": "<smtp message>"}
so callers can apply standard rate-limit handling without knowing SMTP
response codes.

All other SMTP errors are re-raised to the caller.
"""

from __future__ import annotations

import logging
import re
import smtplib
import socket
import uuid
from contextlib import contextmanager
from typing import Any

log = logging.getLogger(__name__)

#: A hostname smtplib may put in an EHLO: letters, digits, hyphens and dots.
_EHLO_SAFE = re.compile(r"^[A-Za-z0-9.-]+$")


def _ehlo_name() -> str:
    """The name to announce in EHLO, guaranteed to be legal.

    smtplib defaults to ``socket.getfqdn()``, which is not always a valid
    hostname. On Windows it can come back with control characters embedded --
    e.g. ``'LAPTOP-TGJ7B0SF.\\x08\\x08\\x04\\x04'`` -- and a server that
    receives that rejects the EHLO with 501. Because the rejected EHLO is also
    what advertises the server's extensions, the next call fails as
    ``SMTPNotSupportedError: STARTTLS extension not supported by server``,
    which points at the server rather than at the name we sent it.

    A validated name with no domain suffix is handled the way smtplib's own
    default (no ``local_hostname`` at all) already handles it: promoted to a
    bracketed address literal (RFC 5321 4.1.3) rather than sent bare, so this
    only adds a safety net for the corrupted case -- it never trades away
    behaviour smtplib already got right for the merely-dotless one.
    """
    try:
        candidate = socket.getfqdn().strip()
    except Exception:  # noqa: BLE001 - name resolution must never break sending
        candidate = ""
    if candidate and _EHLO_SAFE.match(candidate):
        if "." in candidate:
            return candidate
        try:
            return f"[{socket.gethostbyname(socket.gethostname())}]"
        except socket.gaierror:
            return "[127.0.0.1]"
    return "localhost"


class SmtpSender:
    """Stateless SMTP sender.

    Usage:
        sender = SmtpSender(host="smtp.zoho.in", port=587,
                            user="bot@your-domain.com", password="<app-password>",
                            bot_from="bot@your-domain.com")
        result = sender.send(to="user@example.com", raw_bytes=msg_bytes)
        # {"status": 250, "message_id": "<...>"}
        # {"status": 429, "error": "..."} on SMTP 421
    """

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        bot_from: str,
    ) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.bot_from = bot_from

    @contextmanager
    def _session(self):
        """Open an SMTP session with EHLO → STARTTLS → AUTH, then close."""
        smtp = smtplib.SMTP(self.host, self.port, local_hostname=_ehlo_name(), timeout=30)
        try:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(self.user, self.password)
            yield smtp
        finally:
            try:
                smtp.quit()
            except Exception:
                pass

    def send(self, to: str, raw_bytes: bytes) -> dict[str, Any]:
        """Send pre-built RFC 5322 bytes via SMTP STARTTLS.

        Returns:
            {"status": 250, "message_id": "<...>"}  — on success
            {"status": 429, "error": "..."}          — on SMTP 421 (try later)

        Raises smtplib.SMTPException for all other SMTP errors.
        """
        msg_id = f"<{uuid.uuid4().hex}@glc>"
        try:
            with self._session() as smtp:
                smtp.sendmail(self.bot_from, to, raw_bytes)
            log.info("[SMTP ] Delivered to %s — 250 OK", to)
            return {"status": 250, "message_id": msg_id}
        except smtplib.SMTPResponseException as exc:
            if exc.smtp_code == 421:
                log.warning("[SMTP ] 421 back-pressure sending to %s: %s", to, exc)
                return {"status": 429, "error": str(exc)}
            raise
