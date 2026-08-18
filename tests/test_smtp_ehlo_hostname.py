"""SMTP EHLO name derivation: socket.getfqdn() is not always a legal hostname.

Windows in particular can hand one back with control characters embedded --
observed live: 'LAPTOP-TGJ7B0SF.\\x08\\x08\\x04\\x04'. smtplib passes whatever
socket.getfqdn() returns straight into the EHLO command unless told otherwise,
and Gmail rejects an EHLO carrying control characters with 501 -- which also
means STARTTLS is never advertised, so the very next call fails as
SMTPNotSupportedError, pointing at the wrong layer entirely.
"""

from __future__ import annotations

from glc.channels.catalogue.imap import smtp_sender as S

# ── _ehlo_name() ────────────────────────────────────────────────────────────

def test_a_clean_fqdn_is_used_as_is(monkeypatch):
    monkeypatch.setattr(S.socket, "getfqdn", lambda: "mail.example.com")
    assert S._ehlo_name() == "mail.example.com"


def test_a_fqdn_with_control_characters_falls_back_to_localhost(monkeypatch):
    # The exact shape observed live on Windows.
    monkeypatch.setattr(S.socket, "getfqdn", lambda: "LAPTOP-TGJ7B0SF.\x08\x08\x04\x04")
    assert S._ehlo_name() == "localhost"


def test_an_empty_fqdn_falls_back_to_localhost(monkeypatch):
    monkeypatch.setattr(S.socket, "getfqdn", lambda: "")
    assert S._ehlo_name() == "localhost"


def test_a_failing_lookup_falls_back_to_localhost_rather_than_raising(monkeypatch):
    def boom():
        raise OSError("name resolution unavailable")

    monkeypatch.setattr(S.socket, "getfqdn", boom)
    assert S._ehlo_name() == "localhost"


def test_a_dotless_safe_hostname_is_promoted_to_an_ip_literal(monkeypatch):
    # smtplib's own default (when no local_hostname is given at all) treats a
    # dotless name as not a real fqdn and promotes it to a domain literal --
    # RFC 5321 4.1.3 -- rather than sending it bare. Matching that here means
    # this fix only adds a safety net for the corrupted case; it never trades
    # away behaviour smtplib already got right for the merely-dotless one.
    monkeypatch.setattr(S.socket, "getfqdn", lambda: "labhost")
    monkeypatch.setattr(S.socket, "gethostbyname", lambda _name: "192.168.1.5")
    assert S._ehlo_name() == "[192.168.1.5]"


def test_a_dotless_hostname_falls_back_to_the_loopback_literal_if_resolution_fails(monkeypatch):
    def boom(_name):
        raise S.socket.gaierror("no address associated with hostname")

    monkeypatch.setattr(S.socket, "getfqdn", lambda: "labhost")
    monkeypatch.setattr(S.socket, "gethostbyname", boom)
    assert S._ehlo_name() == "[127.0.0.1]"


# ── SmtpSender._session() actually uses it ──────────────────────────────────

def test_the_session_passes_the_safe_name_as_local_hostname(monkeypatch):
    captured: dict = {}

    class FakeSmtp:
        def __init__(self, host, port, local_hostname=None, timeout=None):
            captured["host"] = host
            captured["port"] = port
            captured["local_hostname"] = local_hostname

        def ehlo(self):
            pass

        def starttls(self):
            pass

        def login(self, user, password):
            pass

        def quit(self):
            pass

    monkeypatch.setattr(S.socket, "getfqdn", lambda: "LAPTOP-TGJ7B0SF.\x08\x08\x04\x04")
    monkeypatch.setattr(S.smtplib, "SMTP", FakeSmtp)

    sender = S.SmtpSender(host="smtp.example.com", port=587, user="u", password="p",
                          bot_from="bot@example.com")
    with sender._session():
        pass

    assert captured["host"] == "smtp.example.com"
    assert captured["port"] == 587
    assert captured["local_hostname"] == "localhost"
