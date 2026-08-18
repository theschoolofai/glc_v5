"""Pairing flow — code generation, expiry, validation."""

from __future__ import annotations

import time

from glc.security.pairing import CODE_TTL_SECONDS, PairingStore


def test_issue_code_is_six_digits():
    store = PairingStore()
    code, exp = store.issue_code("telegram", "42", "me")
    assert len(code) == 6 and code.isdigit()
    assert exp > time.time()


def test_confirm_creates_pairing():
    store = PairingStore()
    code, _ = store.issue_code("telegram", "42", "me", requested_trust_level="user_paired")
    rec = store.confirm_code(code)
    assert rec is not None
    assert rec.trust_level == "user_paired"
    assert store.lookup("telegram", "42") is not None


def test_expired_code_is_rejected(monkeypatch):
    store = PairingStore()
    code, _ = store.issue_code("telegram", "42", "me")
    # Move time forward past the TTL
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + CODE_TTL_SECONDS + 1)
    assert store.confirm_code(code) is None


def test_unknown_code_returns_none():
    store = PairingStore()
    assert store.confirm_code("000000") is None


def test_owner_paired_classification():
    store = PairingStore()
    rec = store.force_pair_owner("webui", "owner-1")
    assert rec.trust_level == "owner_paired"
    found = store.lookup("webui", "owner-1")
    assert found is not None
    assert found.trust_level == "owner_paired"


def test_owners_only_returns_owner_paired():
    store = PairingStore()
    store.force_pair_owner("telegram", "owner-1")
    code, _ = store.issue_code("telegram", "user-1", requested_trust_level="user_paired")
    store.confirm_code(code)
    owners = store.owners(channel="telegram")
    assert len(owners) == 1
    assert owners[0].channel_user_id == "owner-1"


def test_revoke_removes_pairing():
    store = PairingStore()
    store.force_pair_owner("matrix", "owner-1")
    assert store.revoke("matrix", "owner-1") is True
    assert store.lookup("matrix", "owner-1") is None


def test_code_collision_replaces_pending():
    store = PairingStore()
    code1, _ = store.issue_code("slack", "U1")
    code2, _ = store.issue_code("slack", "U1")
    # Two requests for the same user: latest pending wins (sane UX —
    # the user shouldn't have to remember which old code is live).
    if code1 != code2:
        assert store.confirm_code(code2) is not None


# ------------------------------------------------ email identities are case-insensitive

def test_owner_pairing_survives_a_different_case_on_a_later_message():
    """gmail/imap adapters extract the bare address straight off the From:
    header with no lowercasing (see gmail/adapter.py:_extract_email,
    imap/mime_parser.py:_strip_display_name), and real mail clients/MTAs are
    not consistent about local-part case even for the same real inbox. An
    owner paired as "Owner@Gmail.com" must still classify as owner_paired
    when a later message arrives as "owner@gmail.com" — RFC 5321 permits
    case-sensitive local parts in theory, but no real provider (Gmail
    included) treats them that way, and a pairing store that does is a
    trust boundary that silently drops a real owner to untrusted."""
    store = PairingStore()
    store.force_pair_owner("gmail", "Owner@Gmail.com", "owner")
    assert store.lookup("gmail", "owner@gmail.com") is not None
    assert store.lookup("gmail", "OWNER@GMAIL.COM") is not None
    assert store.lookup("gmail", "owner@gmail.com").trust_level == "owner_paired"


def test_pairing_code_flow_is_also_case_insensitive():
    store = PairingStore()
    code, _ = store.issue_code("imap", "User.Name@Example.com", "them",
                               requested_trust_level="user_paired")
    store.confirm_code(code)
    assert store.lookup("imap", "user.name@example.com") is not None


def test_revoke_is_case_insensitive_too():
    store = PairingStore()
    store.force_pair_owner("gmail", "Owner@Gmail.com")
    assert store.revoke("gmail", "owner@gmail.com") is True
    assert store.lookup("gmail", "Owner@Gmail.com") is None
