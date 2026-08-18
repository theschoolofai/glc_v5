"""Shared fixtures.

Each test session gets a fresh isolated config/db dir so user state at
~/.glc/ is never touched. Per-test, the audit / pairing / gateway DBs
are rolled fresh.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_glc_state(monkeypatch, tmp_path):
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    monkeypatch.setenv("GLC_CONFIG_DIR", str(cfg))
    monkeypatch.setenv("GLC_AUDIT_DB", str(tmp_path / "audit.sqlite"))
    monkeypatch.setenv("GLC_PAIRING_DB", str(tmp_path / "pairings.sqlite"))
    monkeypatch.setenv("GLC_GATEWAY_DB", str(tmp_path / "gateway.sqlite"))
    monkeypatch.setenv("GLC_PRESENCE_DB", str(tmp_path / "channel_presence.sqlite"))

    # Reset singletons that cache config-dir at first access.
    import glc.config as _cfg

    _cfg.CONFIG_DIR = cfg
    import glc.security.pairing as _p

    _p._singleton = None
    import glc.security.rate_limits as _r

    _r._limiter = None
    import glc.policy.engine as _e

    _e._engine = None
    import glc.audit.store as _a

    _a._singleton = None
    import glc.channels.presence as _pres

    _pres._schema_ready = False

    # `glc.db.DB_PATH` is bound from the environment at *import* time, so
    # setting GLC_GATEWAY_DB above only isolates the ledger for a module that
    # has not been imported yet — which, once anything has touched glc.db, is
    # nothing. Without this the whole suite writes into the developer's real
    # ~/.glc/gateway.sqlite. Patch the bound value too.
    import glc.db as _db

    monkeypatch.setattr(_db, "DB_PATH", str(tmp_path / "gateway.sqlite"))

    # v4 singletons hold config read against the previous CONFIG_DIR.
    import glc.economics.budget as _b

    _b.reset_controller()
    import glc.economics.meter as _m

    _m.reset_meter()
    import glc.economics.pricing as _pr

    _pr.reload_pricing()
    import glc.routing.policy as _rp

    _rp.reset_policy()
    import glc.telemetry.otel as _o

    _o.reset_telemetry()
    yield


@pytest.fixture
def app_client():
    """TestClient pointed at a freshly-booted glc.main:app."""
    from fastapi.testclient import TestClient

    import glc.main as m

    with TestClient(m.app) as c:
        yield c


@pytest.fixture
def install_token(app_client):
    """Returns the per-installation token created during boot."""
    from glc.config import install_token_path

    return install_token_path().read_text().strip()
