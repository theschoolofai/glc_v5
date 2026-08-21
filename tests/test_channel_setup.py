from __future__ import annotations

import json
import os


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_channels_page_and_safe_catalogue(app_client, install_token):
    page = app_client.get("/channels")
    assert page.status_code == 200
    assert "GLC v5" in page.text

    denied = app_client.get("/v1/channel-admin/catalogue")
    assert denied.status_code == 401

    result = app_client.get("/v1/channel-admin/catalogue", headers=_auth(install_token))
    assert result.status_code == 200
    names = {entry["name"] for entry in result.json()["channels"]}
    assert {"gmail", "telegram", "slack", "whatsapp", "webui"} <= names
    telegram = next(entry for entry in result.json()["channels"] if entry["name"] == "telegram")
    assert telegram["webhook_url"] is None
    assert telegram["docs"] == "https://core.telegram.org/bots"
    assert all("value" not in field for field in telegram["fields"])


def test_secret_save_never_round_trips_and_requires_restart(app_client, install_token, tmp_path):
    result = app_client.put(
        "/v1/channel-admin/telegram",
        headers=_auth(install_token),
        json={"enabled": True, "values": {"TELEGRAM_BOT_TOKEN": "not-for-the-browser", "TELEGRAM_OWNER_ID": "42"}},
    )
    assert result.status_code == 200
    assert result.json()["restart_required"] is True
    raw = (tmp_path / "cfg" / "channel_secrets.json").read_text()
    assert "not-for-the-browser" in raw
    assert (tmp_path / "cfg" / "channel_secrets.json").stat().st_mode & 0o077 == 0

    catalogue = app_client.get("/v1/channel-admin/catalogue", headers=_auth(install_token))
    assert "not-for-the-browser" not in catalogue.text
    telegram = next(entry for entry in catalogue.json()["channels"] if entry["name"] == "telegram")
    assert telegram["enabled"] is True
    assert telegram["configured"] is True
    assert all(set(field) <= {"key", "label", "secret", "configured"} for field in telegram["fields"])


def test_adapter_contracts_are_reflected_in_cards(app_client, install_token):
    catalogue = app_client.get("/v1/channel-admin/catalogue", headers=_auth(install_token)).json()["channels"]
    by_name = {entry["name"]: entry for entry in catalogue}

    teams = by_name["teams"]
    assert [field["key"] for field in teams["fields"]] == [
        "TEAMS_APP_ID",
        "TEAMS_APP_PASSWORD",
        "TEAMS_TENANT_ID",
    ]
    for guide_only in ("discord", "imap", "line", "local_mic", "matrix", "signal", "slack", "webui"):
        assert by_name[guide_only]["can_configure"] is False
        assert by_name[guide_only]["fields"] == []
    assert by_name["webhook"]["webhook_url"].endswith("/v1/channels/webhook/webhook")
    assert by_name["whatsapp"]["verification"] == "meta_hub"
    assert by_name["whatsapp"]["webhook_url"].endswith("/v1/channels/whatsapp/webhook")


def test_channel_admin_does_not_accept_s16_bridge_token(app_client, install_token, monkeypatch):
    monkeypatch.setenv("GLC_S16_BRIDGE_TOKEN", "agent-only-token")
    denied = app_client.get("/v1/channel-admin/catalogue", headers=_auth("agent-only-token"))
    assert denied.status_code == 401
    assert app_client.get("/v1/channel-admin/catalogue", headers=_auth(install_token)).status_code == 200


def test_only_meta_get_verification_is_exposed(app_client):
    assert app_client.get("/v1/channels/telegram/webhook").status_code == 404
    assert app_client.get("/v1/channels/whatsapp/webhook").status_code == 403


def test_meta_verification_fails_closed_when_token_unset(app_client, monkeypatch):
    """With WHATSAPP_VERIFY_TOKEN unset, an empty verify_token must not pass.

    hmac.compare_digest('', '') is True, so an empty token against an unset
    env var would complete the Meta hub handshake. The check must fail closed
    when no token is configured.
    """
    monkeypatch.delenv("WHATSAPP_VERIFY_TOKEN", raising=False)
    resp = app_client.get(
        "/v1/channels/whatsapp/webhook",
        params={"hub.mode": "subscribe", "hub.verify_token": "", "hub.challenge": "challenge-123"},
    )
    assert resp.status_code == 403


def test_rejects_unknown_channel_and_unknown_setting(app_client, install_token):
    missing = app_client.put("/v1/channel-admin/not-real", headers=_auth(install_token), json={"enabled": True})
    assert missing.status_code == 404

    result = app_client.put(
        "/v1/channel-admin/telegram",
        headers=_auth(install_token),
        json={"values": {"NOT_A_REAL_SECRET": "discarded"}},
    )
    assert result.status_code == 200
    from glc.channels import setup

    assert "NOT_A_REAL_SECRET" not in json.dumps(setup.configured("telegram"))


def test_only_exact_adapter_env_fields_are_applied(monkeypatch):
    from glc.channels import setup

    monkeypatch.delenv("TEAMS_APP_ID", raising=False)
    monkeypatch.delenv("MICROSOFT_APP_ID", raising=False)
    setup.update(
        "teams",
        {"TEAMS_APP_ID": "actual-adapter-key", "MICROSOFT_APP_ID": "discarded"},
        enabled=True,
    )
    setup.apply_saved_environment()
    assert os.environ["TEAMS_APP_ID"] == "actual-adapter-key"
    assert "MICROSOFT_APP_ID" not in os.environ
