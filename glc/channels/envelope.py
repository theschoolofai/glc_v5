"""Channel envelope — the typed contract between adapters and the agent
runtime. Lecture §8 documents the shape. Adapters import these types and
the test suite asserts adapter output conforms.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from glc.security.redaction import redact_local_paths

TrustLevel = Literal["owner_paired", "user_paired", "untrusted"]


class Attachment(BaseModel):
    """A non-text payload attached to a message. `ref` is either an
    art:... handle that the gateway can resolve to bytes via the artifact
    store, or an external URL the receiving side can fetch."""

    kind: Literal["image", "audio", "video", "file", "location"]
    ref: str
    mime: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ChannelMessage(BaseModel):
    channel: str
    channel_user_id: str
    user_handle: str
    text: str | None = None
    attachments: list[Attachment] = Field(default_factory=list)
    voice_audio_ref: str | None = None
    thread_id: str | None = None
    trust_level: TrustLevel
    arrived_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ChannelReply(BaseModel):
    channel: str
    channel_user_id: str
    text: str | None = None
    attachments: list[Attachment] = Field(default_factory=list)
    voice_audio_ref: str | None = None
    thread_id: str | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("text", mode="after")
    @classmethod
    def _strip_host_paths(cls, value: str | None) -> str | None:
        """No reply leaves this process carrying a host filesystem path.

        Here rather than in an adapter because every outbound route converges on
        this type: the agent bridge validates one, and the proactive
        ``POST /v1/channels/{name}/send`` endpoint that carries approval
        questions parses one from its request body. An adapter-level guard would
        have to be written fourteen times and would still miss the fifteenth.
        """
        return redact_local_paths(value)
