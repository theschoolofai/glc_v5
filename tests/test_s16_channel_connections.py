from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest

from glc.channels import registry
from glc.channels.agent_bridge import S16AgentBridge
from glc.channels.envelope import ChannelMessage, ChannelReply
from glc.routes.channels import _verified_identity


class RecordingBridge:
    def __init__(self) -> None:
        self.messages: list[ChannelMessage] = []

    async def handle(self, message: ChannelMessage) -> ChannelReply:
        self.messages.append(message)
        return ChannelReply(
            channel=message.channel,
            channel_user_id=message.channel_user_id,
            text=f"S16 answered {message.text}",
            thread_id=message.thread_id,
        )


def test_gateway_replaces_client_claimed_owner_identity_with_pairing_state():
    claimed = ChannelMessage(
        channel="telegram", channel_user_id="attacker", user_handle="attacker", text="hello",
        trust_level="owner_paired", arrived_at=datetime.now(UTC),
    )
    unpaired = SimpleNamespace(lookup=lambda _channel, _user: None)
    checked = _verified_identity(claimed, unpaired)
    assert checked.trust_level == "untrusted"
    assert checked.metadata["glc_principal_id"] == "telegram:attacker"

    owner_store = SimpleNamespace(
        lookup=lambda _channel, _user: SimpleNamespace(trust_level="owner_paired")
    )
    checked_owner = _verified_identity(claimed, owner_store)
    assert checked_owner.trust_level == "owner_paired"
    assert checked_owner.metadata["glc_principal_id"] == "installation-owner"


@pytest.mark.asyncio
async def test_http_bridge_preserves_the_shared_envelope_contract():
    seen = {}

    async def answer(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        body = __import__("json").loads(request.content)
        return httpx.Response(200, json={
            "channel": body["channel"],
            "channel_user_id": body["channel_user_id"],
            "text": "agent result",
            "attachments": [],
            "voice_audio_ref": None,
            "thread_id": body["thread_id"],
        })

    client = httpx.AsyncClient(transport=httpx.MockTransport(answer), base_url="http://s16")
    bridge = S16AgentBridge("http://s16", token="shared", client=client)
    reply = await bridge.handle(ChannelMessage(
        channel="telegram", channel_user_id="42", user_handle="rohan", text="hello",
        thread_id="topic-7", trust_level="owner_paired", arrived_at=datetime.now(UTC),
    ))
    assert reply.text == "agent result" and reply.thread_id == "topic-7"
    assert seen == {"path": "/v1/agent/channel-messages", "auth": "Bearer shared"}
    await client.aclose()


def test_every_discovered_channel_uses_the_same_s16_websocket_connection(
    app_client, install_token, monkeypatch
):
    import glc.routes.channels as channel_routes

    monkeypatch.setattr(channel_routes, "allowed", lambda *args, **kwargs: (True, "enabled for proof"))
    bridge = RecordingBridge()
    app_client.app.state.agent_bridge = bridge
    names = registry.list_channels()
    assert names

    for index, name in enumerate(names):
        envelope = ChannelMessage(
            channel=name,
            channel_user_id=f"user-{index}",
            user_handle=f"student-{index}",
            text=f"real request through {name}",
            thread_id=f"thread-{index}",
            trust_level="owner_paired",
            arrived_at=datetime.now(UTC),
        )
        with app_client.websocket_connect(f"/v1/channels/{name}?token={install_token}") as socket:
            socket.send_text(envelope.model_dump_json())
            reply = socket.receive_json()
        assert reply.get("channel") == name, (name, reply)
        assert reply["text"] == f"S16 answered real request through {name}"
        assert reply["thread_id"] == f"thread-{index}"

    assert [message.channel for message in bridge.messages] == names
    catalogue = app_client.get("/v1/channels").json()["channels"]
    assert [item["name"] for item in catalogue] == names
    assert all(item["connected"] for item in catalogue)


def test_proactive_send_is_registry_driven_and_authenticated(app_client, monkeypatch):
    sent = []

    class Adapter:
        async def send(self, reply):
            sent.append(reply)
            return {"provider_message_id": "msg-9"}

    monkeypatch.setenv("GLC_S16_BRIDGE_TOKEN", "shared")
    monkeypatch.setattr(registry, "instantiate", lambda name: Adapter() if name == "matrix" else None)
    response = app_client.post(
        "/v1/channels/matrix/send",
        headers={"Authorization": "Bearer shared"},
        json={"channel": "matrix", "channel_user_id": "!room:example.org", "text": "Build passed"},
    )
    assert response.status_code == 200
    assert response.json()["adapter_result"]["provider_message_id"] == "msg-9"
    assert sent[0].channel_user_id == "!room:example.org"


def test_proactive_send_accepts_s17_bridge_token_name(app_client, monkeypatch):
    sent = []

    class Adapter:
        async def send(self, reply):
            sent.append(reply)
            return {"provider_message_id": "msg-17"}

    monkeypatch.delenv("GLC_S16_BRIDGE_TOKEN", raising=False)
    monkeypatch.setenv("GLC_S17_BRIDGE_TOKEN", "shared")
    monkeypatch.setattr(registry, "instantiate", lambda name: Adapter() if name == "matrix" else None)
    response = app_client.post(
        "/v1/channels/matrix/send",
        headers={"Authorization": "Bearer shared"},
        json={"channel": "matrix", "channel_user_id": "!room:example.org", "text": "Build passed"},
    )
    assert response.status_code == 200
    assert response.json()["adapter_result"]["provider_message_id"] == "msg-17"
    assert sent[0].channel_user_id == "!room:example.org"


def test_proactive_send_rejects_missing_authority(app_client):
    response = app_client.post(
        "/v1/channels/telegram/send",
        json={"channel": "telegram", "channel_user_id": "42", "text": "hello"},
    )
    assert response.status_code == 401
