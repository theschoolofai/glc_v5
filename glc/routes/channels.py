"""WS /v1/channels/{name} — adapter control plane.

Adapters connect over WebSocket and exchange JSON-serialised
ChannelMessage and ChannelReply envelopes. The connection is gated by
the installation token presented in the Authorization header (Sec-Websocket
clients can pass it as a query string fallback, ?token=...).

This endpoint is the contract surface adapters speak to. The gateway
processes incoming messages through the rate limiter, allowlist and audit
boundary, then forwards the shared envelope to S16Code. No channel-specific
agent integration lives here: every discovered adapter uses the same bridge.
"""

from __future__ import annotations

import hmac
import json
import os

from fastapi import APIRouter, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from glc.audit import append as audit_append
from glc.channels import registry
from glc.channels.agent_bridge import AgentBridgeError
from glc.channels.envelope import ChannelMessage, ChannelReply
from glc.channels.setup import CHANNEL_SPECS, public_catalogue, update
from glc.config import get_or_create_install_token, load_channels
from glc.security.allowlists import allowed
from glc.security.pairing import get_pairing_store
from glc.security.rate_limits import get_rate_limiter

router = APIRouter()


def _channel_cfg(name: str) -> tuple[dict, dict]:
    cfg = load_channels() or {}
    return (cfg.get("defaults") or {}), ((cfg.get("channels") or {}).get(name) or {})


def _derive_gate(name: str, env: ChannelMessage) -> tuple[bool, bool]:
    """Return the (is_public_channel, was_mentioned) pair to gate on.

    Both values used to be read straight out of ``env.metadata``, which the
    caller controls. In ``allowed()`` they only ever make the check stricter,
    so the caller's lever is to *weaken* it: claim ``is_public_channel=False``
    and the public-channel mention requirement never applies, or claim
    ``was_mentioned=True`` and it is satisfied without a mention.

    So neither claim may relax the gate:

    * ``is_public`` is true if the config says so OR the caller admits it.
      Admitting a public channel is only ever stricter, so it is safe to
      honour; denying one is not, so config wins.
    * ``was_mentioned`` is derived from the channel's ``mention_tokens`` when
      any are configured, and the caller's claim is ignored outright. Where no
      tokens are configured there is nothing to verify against, so the claim
      is used as before rather than hard-denying a working channel. Configure
      ``mention_tokens`` to close that last gap.
    """
    defaults, ch = _channel_cfg(name)
    md = env.metadata or {}
    claimed_public = bool(md.get("is_public_channel", False))
    claimed_mention = bool(md.get("was_mentioned", False))

    configured_public = bool(ch.get("is_public", defaults.get("is_public", False)))
    is_public = configured_public or claimed_public

    tokens = ch.get("mention_tokens", defaults.get("mention_tokens", [])) or []
    if tokens:
        text = env.text or ""
        was_mentioned = any(tok and tok in text for tok in tokens)
    else:
        was_mentioned = claimed_mention

    if tokens and claimed_mention != was_mentioned:
        audit_append(
            channel=name,
            channel_user_id=env.channel_user_id,
            trust_level=env.trust_level,
            event_type="mention_claim_ignored",
            result={
                "claimed_was_mentioned": claimed_mention,
                "server_was_mentioned": was_mentioned,
                "claimed_is_public_channel": claimed_public,
                "server_is_public_channel": is_public,
            },
        )
    return is_public, was_mentioned


class ChannelSettingsUpdate(BaseModel):
    """Only non-empty fields replace saved values; a blank clears one value."""

    enabled: bool | None = None
    values: dict[str, str] = Field(default_factory=dict)


def _bridge(state):
    bridge = getattr(state, "agent_bridge", None)
    if bridge is None:
        raise AgentBridgeError("S16 agent bridge is not configured")
    return bridge


def _authorised_channel_control(authorization: str | None) -> bool:
    presented = (authorization or "").removeprefix("Bearer ").strip()
    candidates = {get_or_create_install_token()}
    bridge_token = os.getenv("GLC_S16_BRIDGE_TOKEN", "").strip()
    if bridge_token:
        candidates.add(bridge_token)
    return bool(presented) and any(hmac.compare_digest(presented, expected) for expected in candidates)


def _verified_identity(message: ChannelMessage, pairings) -> ChannelMessage:
    """Replace adapter/client identity claims with gateway-owned pairing state."""
    record = pairings.lookup(message.channel, message.channel_user_id)
    trust = record.trust_level if record is not None else "untrusted"
    principal = ("installation-owner" if trust == "owner_paired"
                 else f"{message.channel}:{message.channel_user_id}")
    metadata = {**message.metadata, "glc_principal_id": principal}
    return message.model_copy(update={"trust_level": trust, "metadata": metadata})


@router.websocket("/v1/channels/{name}")
async def channel_ws(websocket: WebSocket, name: str, token: str | None = Query(default=None)):
    header_auth = websocket.headers.get("authorization") or websocket.headers.get("Authorization")
    presented = None
    if header_auth and header_auth.startswith("Bearer "):
        presented = header_auth.removeprefix("Bearer ").strip()
    elif token:
        presented = token
    expected = get_or_create_install_token()
    if presented != expected:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    state = websocket.app.state
    registered = list(getattr(state, "registered_channels", []))
    if name not in registered:
        registered.append(name)
        state.registered_channels = registered

    limiter = get_rate_limiter()
    pairings = get_pairing_store()
    owners = [p.channel_user_id for p in pairings.owners(channel=name)]

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
                env = ChannelMessage.model_validate(payload)
            except Exception as e:
                await websocket.send_text(json.dumps({"error": f"invalid envelope: {e}"}))
                continue

            if env.channel != name:
                await websocket.send_text(json.dumps({"error": "envelope channel does not match route"}))
                continue
            env = _verified_identity(env, pairings)

            gate_public, gate_mentioned = _derive_gate(env.channel, env)
            ok, why = allowed(
                env.channel,
                env.channel_user_id,
                owner_ids=owners,
                is_public_channel=gate_public,
                was_mentioned=gate_mentioned,
            )
            if not ok:
                audit_append(
                    channel=env.channel,
                    channel_user_id=env.channel_user_id,
                    trust_level=env.trust_level,
                    event_type="allowlist_drop",
                    result={"reason": why},
                )
                await websocket.send_text(json.dumps({"error": f"dropped: {why}"}))
                continue

            ok, why = limiter.check_message(env.channel, env.channel_user_id)
            if not ok:
                audit_append(
                    channel=env.channel,
                    channel_user_id=env.channel_user_id,
                    trust_level=env.trust_level,
                    event_type="rate_limit",
                    result={"reason": why},
                )
                await websocket.send_text(json.dumps({"status": 429, "error": why}))
                continue

            audit_append(
                channel=env.channel,
                channel_user_id=env.channel_user_id,
                trust_level=env.trust_level,
                event_type="inbound_message",
                params={"text": env.text, "thread_id": env.thread_id},
            )

            try:
                reply = await _bridge(state).handle(env)
            except AgentBridgeError as error:
                audit_append(
                    channel=env.channel,
                    channel_user_id=env.channel_user_id,
                    trust_level=env.trust_level,
                    event_type="agent_bridge_error",
                    result={"error": str(error)},
                )
                await websocket.send_text(json.dumps({"status": 503, "error": str(error)}))
                continue
            await websocket.send_text(reply.model_dump_json())
    except WebSocketDisconnect:
        return


@router.get("/v1/channels/{name}/webhook")
async def channel_webhook_verify(name: str, request: Request):
    # The hub.* GET challenge is Meta's protocol. Do not pretend Telegram,
    # Slack, Teams, or other providers can use it just because they share a
    # route shape.
    if CHANNEL_SPECS.get(name, {}).get("verification") != "meta_hub":
        raise HTTPException(status_code=404, detail="this channel has no GLC-native webhook verification")
    params = dict(request.query_params)
    mode = params.get("hub.mode", "")
    token = params.get("hub.verify_token", "")
    challenge = params.get("hub.challenge", "")
    expected = os.environ.get(f"{name.upper()}_VERIFY_TOKEN", "")
    # Fail closed when no token is configured: compare_digest('', '') is True,
    # so an empty token against an unset env var would pass the handshake.
    if not expected:
        raise HTTPException(status_code=403)
    if mode == "subscribe" and hmac.compare_digest(token, expected):
        return PlainTextResponse(challenge)
    raise HTTPException(status_code=403)


@router.post("/v1/channels/{name}/webhook")
async def channel_webhook(name: str, request: Request):
    try:
        adapter = registry.instantiate(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown channel: {name}") from None

    raw = {
        "raw_body": await request.body(),
        "headers": dict(request.headers),
    }
    msg = await adapter.on_message(raw)
    if msg is None:
        return {"status": "ok"}

    limiter = get_rate_limiter()
    pairings = get_pairing_store()
    msg = _verified_identity(msg, pairings)
    owners = [p.channel_user_id for p in pairings.owners(channel=name)]

    gate_public, gate_mentioned = _derive_gate(msg.channel, msg)
    ok, why = allowed(
        msg.channel,
        msg.channel_user_id,
        owner_ids=owners,
        is_public_channel=gate_public,
        was_mentioned=gate_mentioned,
    )
    if not ok:
        audit_append(
            channel=msg.channel,
            channel_user_id=msg.channel_user_id,
            trust_level=msg.trust_level,
            event_type="allowlist_drop",
            result={"reason": why},
        )
        return {"status": "ok"}

    ok, why = limiter.check_message(msg.channel, msg.channel_user_id)
    if not ok:
        audit_append(
            channel=msg.channel,
            channel_user_id=msg.channel_user_id,
            trust_level=msg.trust_level,
            event_type="rate_limit",
            result={"reason": why},
        )
        return JSONResponse(status_code=429, content={"error": why})

    audit_append(
        channel=msg.channel,
        channel_user_id=msg.channel_user_id,
        trust_level=msg.trust_level,
        event_type="inbound_message",
        params={"text": msg.text, "thread_id": msg.thread_id, "provider": msg.metadata.get("provider")},
    )

    try:
        reply = await _bridge(request.app.state).handle(msg)
    except AgentBridgeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    await adapter.send(reply)
    return {"status": "ok"}


@router.get("/v1/channels")
async def channel_catalogue(request: Request):
    """The dynamic channel surface S16 can inspect without a hard-coded list."""
    connected = set(getattr(request.app.state, "registered_channels", []))
    return {
        "channels": [
            {"name": name, "connected": name in connected}
            for name in registry.list_channels()
        ]
    }


def _require_admin(authorization: str | None) -> None:
    # S16's bridge token can send an already-authorised outbound message, but
    # it must never gain access to the human setup/credential surface.
    presented = (authorization or "").removeprefix("Bearer ").strip()
    if not presented or not hmac.compare_digest(presented, get_or_create_install_token()):
        raise HTTPException(status_code=401, detail="enter this installation's control token")


@router.get("/v1/channel-admin/catalogue")
async def channel_admin_catalogue(request: Request, authorization: str | None = Header(default=None)):
    """Safe setup view: values are represented solely by presence booleans."""
    _require_admin(authorization)
    base = os.getenv("GLC_PUBLIC_BASE", str(request.base_url).rstrip("/"))
    return {"channels": public_catalogue(base, registry.list_channels())}


@router.put("/v1/channel-admin/{name}")
async def update_channel_settings(
    name: str,
    settings: ChannelSettingsUpdate,
    authorization: str | None = Header(default=None),
):
    _require_admin(authorization)
    if name not in CHANNEL_SPECS or name not in registry.list_channels():
        raise HTTPException(status_code=404, detail=f"unknown channel: {name}")
    update(name, settings.values, settings.enabled)
    # The process environment is intentionally not changed here. A restart is
    # required before a legacy adapter consumes new credentials, preventing a
    # half-reconfigured live connection.
    return {"ok": True, "restart_required": True}


@router.post("/v1/channels/{name}/send")
async def send_channel_message(
    name: str,
    reply: ChannelReply,
    request: Request,
    authorization: str | None = Header(default=None),
):
    """Dispatch one proactive message through any registered adapter.

    This is intentionally one generic capability. Provider-specific addressing
    remains in the adapter and the model cannot invent an unregistered channel.
    """
    if not _authorised_channel_control(authorization):
        raise HTTPException(status_code=401, detail="invalid channel bridge token")
    if reply.channel != name:
        raise HTTPException(status_code=422, detail="body channel does not match route")
    try:
        adapter = registry.instantiate(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown channel: {name}") from None
    try:
        result = await adapter.send(reply)
    except Exception as error:
        audit_append(
            channel=name,
            channel_user_id=reply.channel_user_id,
            trust_level="owner_paired",
            event_type="outbound_message_failed",
            params={"thread_id": reply.thread_id},
            result={"error": f"{type(error).__name__}: {error}"},
        )
        raise HTTPException(status_code=502, detail=f"channel send failed: {type(error).__name__}: {error}") from error
    audit_append(
        channel=name,
        channel_user_id=reply.channel_user_id,
        trust_level="owner_paired",
        event_type="outbound_message",
        params={"thread_id": reply.thread_id},
        result={"adapter_result": result},
    )
    return {"accepted": True, "channel": name, "adapter_result": result}
