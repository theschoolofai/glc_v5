"""Loads channels.yaml and policy.yaml. Resolves user-config directory.

The default config lives in `~/.glc/`. Override with GLC_CONFIG_DIR for
tests and CI. The directory is created on import if missing.
"""

from __future__ import annotations

import os
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


def install_token_path() -> Path:
    return CONFIG_DIR / "install_token"


def get_or_create_install_token() -> str:
    """Per-installation token used to authenticate WS adapter connections
    and /v1/control/* requests. Generated once and persisted to disk."""
    p = install_token_path()
    if p.exists():
        return p.read_text().strip()
    import secrets

    # Deferred: glc.security's package init imports glc.config (via
    # allowlists.py), so importing this at module level would be circular.
    from glc.security.file_permissions import restrict_to_owner

    tok = secrets.token_urlsafe(32)
    p.write_text(tok)
    restrict_to_owner(p)
    return tok
