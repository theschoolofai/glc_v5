"""Restrict a freshly-written credential file to the current user.

POSIX honours `chmod`'s owner/group/other bits directly. Windows does not:
`os.chmod` there only toggles the read-only DOS attribute and leaves the
file's NTFS ACL (inherited from its parent folder, typically far more
permissive) untouched, so a bare `os.chmod(path, 0o600)` on Windows
silently protects nothing. `icacls` ships with every Windows install and is
the dependency-free way to actually strip that inherited ACL and grant
access to the current user alone.
"""

from __future__ import annotations

import getpass
import logging
import os
import platform
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def restrict_to_owner(path: Path) -> None:
    """Best-effort: make `path` readable/writable only by the current user.

    Failure is logged rather than swallowed — silently doing nothing is the
    original bug this replaces, and it hid the one signal an operator would
    have had that a credential file wasn't actually protected.
    """
    if platform.system() == "Windows":
        user = getpass.getuser()
        result = subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:F"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(
                "could not restrict %s to its owner: %s", path, result.stderr.strip()
            )
    else:
        try:
            os.chmod(path, 0o600)
        except OSError as error:
            logger.warning("could not restrict %s to its owner: %s", path, error)
