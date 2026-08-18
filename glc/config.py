"""Loads channels.yaml and policy.yaml. Resolves user-config directory.

The default config lives in `~/.glc/`. Override with GLC_CONFIG_DIR for
tests and CI. The directory is created on import if missing.
"""

from __future__ import annotations

import getpass
import os
import subprocess  # nosec B404: fixed argv, no shell, Windows ACL tooling only
import sys
from pathlib import Path

import yaml

DEFAULT_DIR = Path(os.path.expanduser("~/.glc"))
CONFIG_DIR = Path(os.getenv("GLC_CONFIG_DIR", str(DEFAULT_DIR)))
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Packaged defaults shipped with glc (under the policy/ subpackage).
PACKAGED_POLICY = Path(__file__).parent / "policy" / "policy.yaml"
PACKAGED_CHANNELS = Path(__file__).parent / "channels.yaml"

# v4 config surface. Same two-level resolution as policy.yaml: a packaged
# default ships with the wheel, and a same-named file in CONFIG_DIR wins.
# Every one of these can also be pointed at an arbitrary path with an env var,
# which is what the proof harness and the tests use.
PACKAGED_PRICING = Path(__file__).parent / "economics" / "pricing.yaml"
PACKAGED_BUDGETS = Path(__file__).parent / "economics" / "budgets.yaml"
PACKAGED_ROUTING = Path(__file__).parent / "routing" / "routing.yaml"
PACKAGED_CACHE = Path(__file__).parent / "cache" / "cache.yaml"


def policy_yaml_path() -> Path:
    user = CONFIG_DIR / "policy.yaml"
    return user if user.exists() else PACKAGED_POLICY


def channels_yaml_path() -> Path:
    user = CONFIG_DIR / "channels.yaml"
    return user if user.exists() else PACKAGED_CHANNELS


def _resolve(env_var: str, filename: str, packaged: Path) -> Path:
    """env var override → CONFIG_DIR/<filename> → packaged default."""
    override = os.getenv(env_var)
    if override:
        return Path(override).expanduser()
    user = CONFIG_DIR / filename
    return user if user.exists() else packaged


def pricing_yaml_path() -> Path:
    return _resolve("GLC_PRICING_YAML", "pricing.yaml", PACKAGED_PRICING)


def budgets_yaml_path() -> Path:
    return _resolve("GLC_BUDGETS_YAML", "budgets.yaml", PACKAGED_BUDGETS)


def routing_yaml_path() -> Path:
    return _resolve("GLC_ROUTING_YAML", "routing.yaml", PACKAGED_ROUTING)


def cache_yaml_path() -> Path:
    return _resolve("GLC_CACHE_YAML", "cache.yaml", PACKAGED_CACHE)


def load_yaml(path: Path, default: dict | None = None) -> dict:
    """Read a config file, returning `default` when it is absent or empty.

    Parse errors are raised, not swallowed: a typo in a budget ceiling must not
    silently degrade into "no budget".
    """
    if not path.exists():
        return dict(default or {})
    data = yaml.safe_load(path.read_text())
    if data is None:
        return dict(default or {})
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level, got {type(data).__name__}")
    return data


def load_channels() -> dict:
    p = channels_yaml_path()
    if not p.exists():
        return {"channels": {}}
    return yaml.safe_load(p.read_text()) or {"channels": {}}


# 0o600 keeps a file from the group and from everyone else. It does not keep it
# from root, who reads anything on the machine. The faithful Windows equivalent
# therefore keeps SYSTEM and the local Administrators group — removing them
# would be stricter than the POSIX behaviour being ported and would break a
# service or a backup running under either. What must go is BUILTIN\Users and
# anything else inherited from the parent directory.
WELL_KNOWN_SYSTEM = "*S-1-5-18"
WELL_KNOWN_ADMINISTRATORS = "*S-1-5-32-544"

# icacls prints resolved display names, which are localised. Matching them by
# name is best-effort: on a non-English Windows an unrecognised name is treated
# as foreign, so owner_only errs towards reporting False rather than towards
# claiming a guarantee it cannot verify.
_ROOT_EQUIVALENT = {"system", "administrators", "owner rights", "s-1-5-18", "s-1-5-32-544"}


def _current_account() -> str:
    """The account name icacls should grant, without the domain prefix."""
    return getpass.getuser().split("\\")[-1]


def _windows_principals(path: Path) -> set[str] | None:
    """Every principal holding an ACE on the file, or None if icacls failed."""
    try:
        result = subprocess.run(  # nosec B603: fixed argv, no shell
            ["icacls", str(path)], capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    principals: set[str] = set()
    for raw in result.stdout.splitlines():
        line = raw.strip()
        if line.startswith(str(path)):
            line = line[len(str(path)):].strip()
        if ":(" not in line:
            continue
        principals.add(line.split(":(")[0].strip())
    return principals


def restrict_to_owner(path: Path) -> bool:
    """Make a secret file readable by its owner alone. Returns whether it took.

    ``os.chmod`` expresses POSIX mode bits. On Windows it only maps to the
    read-only attribute and it returns *successfully* either way, so
    ``chmod(0o600)`` there reports that it worked and changes nothing — the file
    keeps whatever ACL it inherited. Under a user profile that is usually
    harmless; anywhere under ``C:\\ProgramData`` the inherited ACL grants
    ``BUILTIN\\Users`` read access, which is every account on the machine.
    """
    if sys.platform != "win32":
        try:
            os.chmod(path, 0o600)
        except OSError:
            return False
        return True
    try:
        result = subprocess.run(  # nosec B603: fixed argv, no shell
            ["icacls", str(path), "/inheritance:r",
             "/grant:r", f"{_current_account()}:F",
             "/grant:r", f"{WELL_KNOWN_SYSTEM}:F",
             "/grant:r", f"{WELL_KNOWN_ADMINISTRATORS}:F"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def owner_only(path: Path) -> bool:
    """Whether the file is in fact reachable by nobody but its owner.

    Separate from ``restrict_to_owner`` on purpose: the bug this replaces was a
    call that claimed success, so the property is worth checking rather than
    assuming. Suitable for a preflight check as well as for tests.
    """
    if sys.platform != "win32":
        return path.stat().st_mode & 0o077 == 0
    principals = _windows_principals(path)
    if principals is None:
        return False
    account = _current_account().casefold()
    def allowed(principal: str) -> bool:
        tail = principal.split("\\")[-1].casefold()
        return tail == account or tail in _ROOT_EQUIVALENT
    return bool(principals) and all(allowed(p) for p in principals)


def install_token_path() -> Path:
    return CONFIG_DIR / "install_token"


def get_or_create_install_token() -> str:
    """Per-installation token used to authenticate WS adapter connections
    and /v1/control/* requests. Generated once and persisted to disk."""
    p = install_token_path()
    if p.exists():
        return p.read_text().strip()
    import secrets

    tok = secrets.token_urlsafe(32)
    p.write_text(tok)
    restrict_to_owner(p)
    return tok
