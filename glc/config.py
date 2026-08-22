"""Loads channels.yaml and policy.yaml. Resolves user-config directory.

The default config lives in `~/.glc/`. Override with GLC_CONFIG_DIR for
tests and CI. The directory is created on import if missing.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

import yaml

log = logging.getLogger("glc.config")

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


def _icacls(*args: str) -> subprocess.CompletedProcess[str]:
    """Run icacls, decoding defensively.

    `text=True` decodes with the locale codec, which is cp1252 here — and icacls
    echoes the path back, so a single accented or CJK character in a config
    directory raised UnicodeDecodeError inside subprocess's reader THREAD. That
    surfaces as an unhandled traceback on a background thread and leaves stdout
    as None, so the caller then fails on the None rather than on anything to do
    with permissions. Decoding with replacement keeps the output's structure,
    which is all either caller reads.
    """
    return subprocess.run(                                       # noqa: S603
        ["icacls", *args], capture_output=True, timeout=15, check=False,
        text=True, encoding="utf-8", errors="replace")


def restrict_to_owner(path: Path) -> bool:
    """Make `path` readable only by the account that owns it. Returns success.

    `os.chmod(path, 0o600)` is the obvious thing and, on Windows, does nothing
    that matters: the call SUCCEEDS, so a surrounding `except OSError` never
    fires, but only the read-only attribute is touched — the POSIX permission
    bits are ignored and the file keeps whatever the parent directory's ACL
    gave it. Measured on Windows 11: mode is 0o100666 both before and after the
    chmod. Every credential this gateway persists — channel tokens, and the
    install token that authenticates /v1/control/* — was written with a
    protection that silently did not exist, while the docstrings promised
    "owner-only permissions".

    So on Windows we set a real ACL instead: break inheritance and grant the
    current user alone. `icacls` ships with the OS, which is why it is used in
    preference to adding a pywin32 dependency for two calls.

    Reports failure rather than raising. A gateway that will not start is worse
    than one that starts and tells you a file could not be locked down — but it
    must TELL you, which is the part that was missing.
    """
    if os.name != "nt":
        try:
            os.chmod(path, 0o600)
            return path.stat().st_mode & 0o077 == 0
        except OSError:
            log.warning("could not restrict permissions on %s", path)
            return False

    user = os.environ.get("USERNAME") or os.environ.get("USER")
    if not user:
        log.warning("cannot restrict %s: no USERNAME in the environment", path)
        return False
    try:
        done = _icacls(str(path), "/inheritance:r", "/grant:r", f"{user}:F")
    except (OSError, subprocess.SubprocessError) as error:
        log.warning("could not restrict %s: %s", path, error)
        return False
    if done.returncode != 0:
        log.warning("could not restrict %s: icacls exited %d: %s",
                    path, done.returncode, (done.stderr or done.stdout or "").strip()[:200])
        return False
    return True


def owner_only(path: Path) -> bool:
    """Is `path` actually restricted to its owner right now?

    Deliberately platform-specific, because the SIGNAL is platform-specific.
    `stat().st_mode` on Windows is synthesised from the read-only attribute and
    reports 0o666 even for a file whose ACL grants exactly one account — so a
    POSIX-bit assertion there is not merely unreliable, it can never pass. Ask
    the ACL instead.
    """
    if os.name != "nt":
        return path.stat().st_mode & 0o077 == 0
    try:
        listed = _icacls(str(path))
    except (OSError, subprocess.SubprocessError):
        return False
    if listed.returncode != 0:
        return False
    # icacls prints "<path> ACCOUNT:(perms)" then one indented line per extra
    # entry. Exactly one grant means inheritance was broken and nobody else was
    # added back. Counting the STRUCTURE rather than reading the account names
    # is what keeps this correct even when a name did not survive decoding.
    grants = re.findall(r"[^\s:]+\\[^\s:]+:\([^)]*\)", listed.stdout or "")
    return len(grants) == 1


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
    # This token authenticates every /v1/control/* request and every WS adapter
    # connection. It is the one file here whose exposure hands over the gateway.
    if not restrict_to_owner(p):
        log.warning("install token at %s could not be restricted to this account — "
                    "anyone able to read it can drive the control plane", p)
    return tok
