"""Empty install / Meta verify tokens must fail closed, not authenticate nobody.

A blank ``~/.glc/install_token`` or an unset ``*_VERIFY_TOKEN`` used to make
``"" == ""`` / ``compare_digest("", "")`` succeed, which opens the channel
WebSocket, control plane, and Meta hub subscribe challenge to anyone.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


def test_empty_install_token_file_is_regenerated(tmp_path, monkeypatch):
    from glc import config as cfg

    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    blank = tmp_path / "install_token"
    blank.write_text("   \n")

    token = cfg.get_or_create_install_token()
    assert token
    assert blank.read_text().strip() == token


def test_control_plane_rejects_empty_bearer_when_token_blank(tmp_path, monkeypatch):
    """Even if regeneration were bypassed, empty presented must not pass."""
    from glc import config as cfg

    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cfg, "get_or_create_install_token", lambda: "")

    import glc.main as m

    with TestClient(m.app) as client:
        r = client.post(
            "/v1/control/pair",
            headers={"Authorization": "Bearer "},
            json={"channel": "telegram", "channel_user_id": "1"},
        )
    assert r.status_code in (403, 503)


def test_channel_websocket_rejects_empty_token_pair(tmp_path, monkeypatch):
    from glc import config as cfg

    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cfg, "get_or_create_install_token", lambda: "")

    import glc.main as m

    with TestClient(m.app) as client:
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect("/v1/channels/telegram?token="):
                pass
    assert excinfo.value.code == 1008


def test_meta_hub_verify_fails_closed_when_verify_token_unset(app_client, monkeypatch):
    monkeypatch.delenv("WHATSAPP_VERIFY_TOKEN", raising=False)
    r = app_client.get(
        "/v1/channels/whatsapp/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "",
            "hub.challenge": "challenge-token",
        },
    )
    assert r.status_code == 403


def test_meta_hub_verify_accepts_configured_token(app_client, monkeypatch):
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "expected-secret")
    r = app_client.get(
        "/v1/channels/whatsapp/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "expected-secret",
            "hub.challenge": "challenge-token",
        },
    )
    assert r.status_code == 200
    assert r.text == "challenge-token"
