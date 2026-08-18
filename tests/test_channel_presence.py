"""Regression test for a cross-container channel-connectivity bug.

The gateway can autoscale to several containers behind the same URL (e.g.
on Modal). `/v1/channels` used to answer from `app.state.registered_channels`,
a plain Python list private to whichever process happened to hold a given
channel's WebSocket. A request that landed on a *different* container --
which never saw that WebSocket connect -- reported the channel as
disconnected even while it was live elsewhere.

That silent inconsistency has a real, user-visible failure mode: a "send a
message" capability that checks `/v1/channels` before sending would find the
channel "disconnected" and refuse to send, even though the channel's bridge
was live and had round-tripped a real message minutes earlier -- because the
refusing container simply never had that WebSocket in its own memory.

This test stands up two independent FastAPI apps -- two separate
`app.state` objects, the way two autoscaled containers never share process
memory -- both wired the same way `glc.main` wires the channels router.
A WebSocket connects on container A; `/v1/channels` is then queried on
container B. Before the fix (connectivity read from `app.state`), this
assertion fails. After the fix (connectivity read from `glc.channels.
presence`, a SQLite-backed store shared across containers via the
persistent volume in production), it passes.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

import glc.routes.channels as channel_routes
from glc.channels.envelope import ChannelMessage, ChannelReply
from glc.config import get_or_create_install_token


class _StubBridge:
    async def handle(self, message: ChannelMessage) -> ChannelReply:
        return ChannelReply(
            channel=message.channel,
            channel_user_id=message.channel_user_id,
            text="ack",
            thread_id=message.thread_id,
        )


def _container(monkeypatch) -> FastAPI:
    """One gateway container: its own FastAPI app, its own `app.state`,
    wired with the real channels router -- nothing shared with any other
    container except whatever each backing store resolves to on disk."""
    app = FastAPI()
    app.include_router(channel_routes.router)
    app.state.agent_bridge = _StubBridge()
    # Allowlist/policy checks are exercised by other tests; this one is
    # only about cross-container connectivity visibility.
    monkeypatch.setattr(channel_routes, "allowed", lambda *a, **k: (True, "test"))
    return app


def test_channel_connectivity_is_visible_across_independent_containers(monkeypatch):
    token = get_or_create_install_token()
    container_a = _container(monkeypatch)
    container_b = _container(monkeypatch)

    with TestClient(container_a) as client_a:
        with client_a.websocket_connect(f"/v1/channels/matrix?token={token}") as socket:
            envelope = ChannelMessage(
                channel="matrix",
                channel_user_id="user-1",
                user_handle="student",
                text="hello",
                trust_level="owner_paired",
                arrived_at=datetime.now(UTC),
            )
            socket.send_text(envelope.model_dump_json())
            socket.receive_json()

    # A second, independent process. If it had to answer from its own
    # memory alone, it would have no idea container A ever connected.
    with TestClient(container_b) as client_b:
        catalogue = client_b.get("/v1/channels").json()["channels"]

    matrix_entry = next(item for item in catalogue if item["name"] == "matrix")
    assert matrix_entry["connected"] is True, (
        "container B reported 'matrix' disconnected even though container A's "
        "WebSocket for it is live -- connectivity must be a shared fact, not "
        "one process's private memory"
    )
