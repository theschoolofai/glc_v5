"""Credential files must be readable only by the account that wrote them.

This gateway is the only process that holds provider keys, and it persists two
things worth stealing: every saved channel credential, and the install token that
authenticates `/v1/control/*`. Both docstrings promised "owner-only permissions"
while, on Windows, nothing of the sort was happening.

The trap is that `os.chmod(path, 0o600)` there SUCCEEDS and changes nothing that
matters, so the surrounding `except OSError` never fires — the guard looks like
handling and is not. Separately, `stat().st_mode` on Windows is synthesised from
the read-only attribute and reports 0o666 even for a file whose ACL grants
exactly one account, so a POSIX-bit assertion cannot detect success there either.
These tests therefore assert the invariant through `config.owner_only`, which
asks the platform-appropriate question.
"""
from __future__ import annotations

import os

import pytest

from glc import config


@pytest.fixture()
def secret(tmp_path):
    path = tmp_path / "credential.json"
    path.write_text('{"token": "sensitive"}')
    return path


class TestRestrictToOwner:
    def test_a_freshly_written_file_is_not_owner_only(self, secret):
        """The premise. If this were already restricted the rest proves nothing."""
        assert config.owner_only(secret) is False

    def test_restricting_makes_it_owner_only(self, secret):
        assert config.restrict_to_owner(secret) is True
        assert config.owner_only(secret) is True

    def test_a_missing_file_reports_failure_rather_than_raising(self, tmp_path):
        """A gateway that will not start is worse than one that warns."""
        absent = tmp_path / "never-written.json"

        assert config.restrict_to_owner(absent) is False
        assert config.owner_only(absent) is False

    @pytest.mark.parametrize("name", [
        "name with spaces.json",
        "weird (1).json",
        "unicode-ü-名.json",
    ])
    def test_awkward_paths_survive(self, tmp_path, name):
        """The unicode case is a regression test for the fix itself.

        On Windows the helper shells out to icacls, which echoes the path back.
        Decoding that with the locale codec (cp1252 here) raised
        UnicodeDecodeError inside subprocess's reader THREAD for any non-ASCII
        path: an unhandled traceback on a background thread, stdout left as
        None, and the caller then failing on the None rather than on anything to
        do with permissions. GLC_CONFIG_DIR sits under a user's home directory,
        and plenty of people have an accented name in that path.
        """
        path = tmp_path / name
        path.write_text("x")

        assert config.restrict_to_owner(path) is True
        assert config.owner_only(path) is True


class TestPersistedCredentials:
    def test_the_install_token_is_restricted(self, tmp_path, monkeypatch):
        """This token authenticates every /v1/control/* request."""
        monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)

        token = config.get_or_create_install_token()

        assert token
        assert config.owner_only(config.install_token_path()) is True

    def test_saved_channel_secrets_are_restricted(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
        from glc.channels import setup

        setup.update("telegram", {"TELEGRAM_BOT_TOKEN": "not-for-anyone-else"}, enabled=True)

        stored = tmp_path / "channel_secrets.json"
        assert "not-for-anyone-else" in stored.read_text()
        assert config.owner_only(stored) is True

    def test_the_temp_file_does_not_outlive_the_save(self, tmp_path, monkeypatch):
        """The write goes through a .tmp then renames. A leftover temp copy would
        be a second, unprotected copy of the same secrets."""
        monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
        from glc.channels import setup

        setup.update("telegram", {"TELEGRAM_BOT_TOKEN": "secret"}, enabled=True)

        assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.skipif(os.name != "nt", reason="the POSIX signal is only wrong on Windows")
def test_stat_mode_is_not_a_usable_signal_on_windows():
    """Documents WHY these tests do not assert st_mode, so nobody reinstates it.

    A file whose ACL grants exactly one account still reports 0o666 here. The
    original assertion in test_channel_setup.py could therefore never pass on
    Windows, even against a correctly protected file — which is why a real
    security bug was readable as a path quirk for so long.
    """
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "c.json"
    path.write_text("x")
    config.restrict_to_owner(path)

    assert config.owner_only(path) is True
    assert path.stat().st_mode & 0o077 != 0
