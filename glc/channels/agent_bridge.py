"""The one HTTP seam between GLC channel envelopes and S16Code.

Adapters remain owned by GLC.  S16 never imports Telegram, Gmail, Twilio, or
any other provider SDK; it receives the common ChannelMessage contract and
returns the common ChannelReply contract.  Consequently every adapter in the
registry, including adapters added later, gets the same agent connection.
"""

from __future__ import annotations

import os

import httpx

from glc.channels.envelope import ChannelMessage, ChannelReply


class AgentBridgeError(RuntimeError):
    """S16 could not accept or answer a channel message."""


def configured_bridge_tokens() -> frozenset[str]:
    """Both env names are valid; S17 is the current name, S16 is the legacy alias."""
    return frozenset(
        token
        for token in (
            os.getenv("GLC_S17_BRIDGE_TOKEN", "").strip(),
            os.getenv("GLC_S16_BRIDGE_TOKEN", "").strip(),
        )
        if token
    )


def configured_bridge_token() -> str:
    """Token GLC presents when calling S17. Prefer the current env name."""
    return os.getenv("GLC_S17_BRIDGE_TOKEN", "").strip() or os.getenv("GLC_S16_BRIDGE_TOKEN", "").strip()


class S16AgentBridge:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        token: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv("S17_BASE_URL", "").strip()
            or os.getenv("S16_BASE_URL", "http://127.0.0.1:8113")
        ).rstrip("/")
        self.token = token if token is not None else configured_bridge_token()
        timeout = os.getenv("GLC_S17_TIMEOUT") or os.getenv("GLC_S16_TIMEOUT") or "180"
        self._client = client or httpx.AsyncClient(timeout=float(timeout))
        self._owns_client = client is None

    async def handle(self, message: ChannelMessage) -> ChannelReply:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        try:
            response = await self._client.post(
                f"{self.base_url}/v1/agent/channel-messages",
                json=message.model_dump(mode="json"),
                headers=headers,
            )
        except httpx.HTTPError as error:
            raise AgentBridgeError(f"S16 is unavailable: {type(error).__name__}: {error}") from error
        if response.status_code >= 400:
            raise AgentBridgeError(
                f"S16 channel bridge returned {response.status_code}: {response.text[:500]}"
            )
        try:
            return ChannelReply.model_validate(response.json())
        except Exception as error:
            raise AgentBridgeError(f"S16 returned an invalid ChannelReply: {error}") from error

    async def health(self) -> dict:
        try:
            response = await self._client.get(f"{self.base_url}/healthz", timeout=3)
            response.raise_for_status()
            return {"ok": True, "base_url": self.base_url, "service": response.json()}
        except Exception as error:
            return {"ok": False, "base_url": self.base_url, "error": f"{type(error).__name__}: {error}"}

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
