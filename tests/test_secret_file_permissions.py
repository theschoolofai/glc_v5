"""Owner-only has to be a fact about the file, not a call that returned.

`os.chmod(path, 0o600)` is the POSIX way to say it. On Windows the same call
maps only to the read-only attribute and returns successfully whatever happens,
so the guarantee in glc/channels/setup.py's docstring held on one platform and
was silently absent on the other.
"""
from __future__ import annotations

import os
import subprocess
import sys

from glc import config


def test_restrict_to_owner_makes_the_file_owner_only(tmp_path) -> None:
    secret = tmp_path / "install_token"
    secret.write_text("a-token")

    assert config.restrict_to_owner(secret) is True
    assert config.owner_only(secret) is True


def test_a_file_another_account_can_read_is_not_reported_as_owner_only(tmp_path) -> None:
    """The check must be able to say no, or it is not a check.

    Reproduce what the parent directory grants under C:\\ProgramData: read for
    BUILTIN\\Users, which is every account on the machine. On POSIX the same
    condition is the group/other bits, set explicitly so the result does not
    depend on the umask this suite happens to run under.
    """
    plain = tmp_path / "not_restricted"
    plain.write_text("a-token")
    if sys.platform == "win32":
        subprocess.run(  # nosec B603: fixed argv, no shell
            ["icacls", str(plain), "/grant", "*S-1-5-32-545:(R)"],
            capture_output=True, check=True, timeout=15,
        )
    else:
        os.chmod(plain, 0o644)

    assert config.owner_only(plain) is False


def test_owner_only_survives_a_rewrite(tmp_path) -> None:
    """_save writes a temp file and replaces the target; both must stay closed."""
    secret = tmp_path / "channel_secrets.json"
    secret.write_text("{}")
    config.restrict_to_owner(secret)

    replacement = tmp_path / "channel_secrets.json.tmp"
    replacement.write_text('{"channels": {}}')
    config.restrict_to_owner(replacement)
    replacement.replace(secret)
    config.restrict_to_owner(secret)

    assert config.owner_only(secret) is True
