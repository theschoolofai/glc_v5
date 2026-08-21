"""The economics surface is operator control-plane: it arms and disarms the
budget ceiling (the denial-of-wallet protection) and purges the cache. Like
/v1/control/* and /v1/channel-admin/*, the mutating routes must require the
installation token. An unauthenticated caller must not be able to close a
budget, remove a ceiling, or purge the cache."""

from __future__ import annotations


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_post_budget_requires_token(app_client):
    resp = app_client.post("/v1/budget", json={"principal": "agent:a", "limit_usd": 0.0})
    assert resp.status_code == 401


def test_delete_budget_requires_token(app_client):
    resp = app_client.delete("/v1/budget/agent:a")
    assert resp.status_code == 401


def test_cache_purge_requires_token(app_client):
    resp = app_client.post("/v1/cache/purge")
    assert resp.status_code == 401


def test_post_budget_succeeds_with_token(app_client, install_token):
    resp = app_client.post(
        "/v1/budget",
        headers=_auth(install_token),
        json={"principal": "agent:a", "limit_usd": 0.0},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_delete_budget_succeeds_with_token(app_client, install_token):
    resp = app_client.delete("/v1/budget/agent:a", headers=_auth(install_token))
    assert resp.status_code == 200
    assert resp.json()["ok"] is True