"""Shared owner-only-permission assertion for credential-file tests.

`st_mode` is a meaningful POSIX check but doesn't reflect real ACL state on
Windows, even after a correct `icacls`-based fix (see
`glc.security.file_permissions.restrict_to_owner`). So Windows gets its own
check here: parse `icacls <path>` (no modification flags) and assert only
expected principals appear. Kept in one place so the parsing logic — which
is the fragile part, since `icacls`'s text output can vary by locale — isn't
duplicated and doesn't drift between the tests that use it.
"""

from __future__ import annotations

import getpass
import platform
import subprocess
from pathlib import Path

# The current user is always expected. SYSTEM and Administrators are
# standard/expected too (Windows itself may retain them); broad groups are
# never expected on a file meant to be owner-only.
_ALLOWED_PRINCIPAL_SUFFIXES = {"system", "administrators"}
_FORBIDDEN_PRINCIPAL_SUFFIXES = {"everyone", "users", "authenticated users"}


def _icacls_principals(path: Path) -> set[str]:
    result = subprocess.run(
        ["icacls", str(path)], capture_output=True, text=True, check=True
    )
    principals: set[str] = set()
    for index, line in enumerate(result.stdout.splitlines()):
        text = line.replace(str(path), "", 1) if index == 0 else line
        text = text.strip()
        if not text or text.lower().startswith("successfully processed"):
            continue
        principal = text.split(":", 1)[0].strip()
        if principal:
            principals.add(principal)
    return principals


def assert_restricted_to_owner(path: Path) -> None:
    """Assert `path` is readable/writable only by its owner (+ SYSTEM/Admins)."""
    if platform.system() == "Windows":
        principals = _icacls_principals(path)
        current_user = getpass.getuser().lower()
        for principal in principals:
            local_name = principal.lower().rsplit("\\", 1)[-1]
            assert local_name not in _FORBIDDEN_PRINCIPAL_SUFFIXES, (
                f"{path} grants access to broad group {principal!r}; full ACL: {principals}"
            )
            assert local_name == current_user or local_name in _ALLOWED_PRINCIPAL_SUFFIXES, (
                f"{path} grants access to unexpected principal {principal!r}; full ACL: {principals}"
            )
    else:
        assert path.stat().st_mode & 0o077 == 0
