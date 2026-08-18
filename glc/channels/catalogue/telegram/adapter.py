"""Telegram Bot API channel adapter.

Group G16: Implement on_message and send against the mock-API fake and real Telegram Bot API.
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from typing import Any

import httpx

from glc.channels.base import ChannelAdapter
from glc.channels.envelope import Attachment, ChannelMessage, ChannelReply
from glc.security.allowlists import allowed
from glc.security.pairing import get_pairing_store
from glc.security.trust_level import classify

from .schemas import TelegramUpdate

# Telegram rejects any MarkdownV2 message containing an unescaped reserved
# character. A full stop is reserved, so an ordinary English sentence fails to
# send. The envelope carries plain prose with no formatting contract, so the
# text is escaped rather than interpreted.
# https://core.telegram.org/bots/api#markdownv2-style
_MARKDOWN_V2_RESERVED = set("_*[]()~`>#+-=|{}.!\\")


def escape_markdown_v2(text: str) -> str:
    """Escape every MarkdownV2 reserved character in plain prose."""
    return "".join("\\" + char if char in _MARKDOWN_V2_RESERVED else char for char in text)


# Agent replies arrive as ordinary markdown: **bold**, ### headings, * bullets.
# Telegram's MarkdownV2 is a different dialect, so escaping alone delivers the
# message but shows the syntax literally. These convert the common constructs
# into MarkdownV2 before escaping the rest.
_FENCE_RE = re.compile(r"```(\w*)\n(.*?)```", re.S)
_CODE_RE = re.compile(r"`([^`\n]+)`")
# Deliberately NOT re.S. A model routinely emits an unbalanced ** at the start
# of a heading line; allowing bold to span newlines lets that stray marker pair
# with one several lines later, bolding the whole block and leaving a literal
# ** behind. Bold is confined to a single line.
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", re.M)
_BULLET_RE = re.compile(r"^(\s*)[*+-]\s+", re.M)
# [label](url). A file:// target is not a link Telegram can open, and rendering
# it exposes a local filesystem path, so those collapse to their label.
_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((\S+?)\)")
_PLACEHOLDER = "\x00"


def markdown_to_telegram(text: str) -> str:
    """Render common markdown as Telegram MarkdownV2, escaping everything else.

    Escaping alone is enough to deliver a message, which is the bug this
    adapter had. It is not enough to read well: an agent answer full of
    ``**bold**`` and ``###`` arrives with the syntax visible. Bold and code are
    translated into the MarkdownV2 spelling, headings become bold, and bullets
    are normalised, before the remaining reserved characters are escaped.
    """
    if not text:
        return ""

    kept: list[tuple[str, str]] = []

    def stash(opener: str, body: str, closer: str) -> str:
        kept.append((opener, body, closer))
        return f"{_PLACEHOLDER}{len(kept) - 1}{_PLACEHOLDER}"

    # Code must survive untouched apart from the two characters Telegram
    # requires escaping even inside a code span.
    def keep_fence(match: re.Match[str]) -> str:
        body = match.group(2).replace("\\", "\\\\").replace("`", "\\`")
        return stash("```\n", body, "```")

    def keep_code(match: re.Match[str]) -> str:
        body = match.group(1).replace("\\", "\\\\").replace("`", "\\`")
        return stash("`", body, "`")

    text = _FENCE_RE.sub(keep_fence, text)
    text = _CODE_RE.sub(keep_code, text)

    text = _HEADING_RE.sub(lambda m: f"**{m.group(1)}**", text)
    text = _BULLET_RE.sub(lambda m: f"{m.group(1)}- ", text)
    def keep_link(match: re.Match[str]) -> str:
        label, url = match.group(1), match.group(2)
        if not url.startswith(("http://", "https://")):
            # Not something Telegram can open, and a file:// target would put a
            # local filesystem path in the chat. Keep the label only.
            return label
        # Inside a MarkdownV2 link target only ) and \ are special.
        safe = url.replace("\\", "\\\\").replace(")", "\\)")
        return stash("[", f"{escape_markdown_v2(label)}]({safe}", ")")

    text = _LINK_RE.sub(keep_link, text)

    def keep_bold(match: re.Match[str]) -> str:
        return stash("*", escape_markdown_v2(match.group(1)), "*")

    text = _BOLD_RE.sub(keep_bold, text)
    # Anything still spelled ** is an unmatched marker the model emitted. It
    # carries no meaning, so drop it rather than escaping it into view.
    text = text.replace("**", "")
    text = escape_markdown_v2(text)

    # Escaping mangled the placeholders, so match them loosely on restore.
    for index, (opener, body, closer) in enumerate(kept):
        marker = re.compile(re.escape(_PLACEHOLDER) + r"\\?" + str(index) + re.escape(_PLACEHOLDER))
        text = marker.sub(lambda _m, o=opener, b=body, c=closer: f"{o}{b}{c}", text, count=1)
    return text


class Adapter(ChannelAdapter):
    name = "telegram"

    async def on_message(self, raw: Any) -> ChannelMessage | None:  # type: ignore[override]
        mock = self.config.get("mock")
        if mock is not None:
            if hasattr(mock, "pop_disconnect") and mock.pop_disconnect():
                return ChannelMessage(
                    channel=self.name,
                    channel_user_id="",
                    user_handle="",
                    text="disconnected",
                    trust_level="untrusted",
                    arrived_at=datetime.now(UTC),
                )

        update = TelegramUpdate.model_validate(raw)

        if update.message is None:
            return None

        message = update.message

        # Extract user information from "from" block, falling back to "chat" block
        sender = message.from_

        if sender is None:
            channel_user_id = str(message.chat.id)
            user_handle = message.chat.username or ""
        else:
            channel_user_id = str(sender.id)
            user_handle = sender.username or ""

        # Get handle/username
        if not user_handle:
            store = get_pairing_store()
            rec = store.lookup(self.name, channel_user_id)
            user_handle = rec.user_handle if rec else channel_user_id

        # Classify trust level
        trust_level = classify(self.name, channel_user_id)

        # Allowlist check for stranger in public channel
        if self.config.get("is_public_channel"):
            owners = [o.channel_user_id for o in get_pairing_store().owners(self.name)]
            is_allowed, _ = allowed(
                channel=self.name,
                channel_user_id=channel_user_id,
                owner_ids=owners,
                is_public_channel=True,
                was_mentioned=bool(self.config.get("was_mentioned", False)),
            )
            if not is_allowed:
                return None

        # Parse text and photo attachments
        text = message.text or message.caption

        attachments: list[Attachment] = []
        photo = message.photo
        if photo:
            # Find the largest photo size
            largest = max(
                photo,
                key=lambda p: p.file_size or (p.width * p.height),
            )

            file_id = largest.file_id
            if file_id:
                ref = ""
                if mock is not None:
                    try:
                        file_info = mock.get_file(file_id)
                        ref = file_info.get("file_path", "")
                    except Exception:
                        pass
                else:
                    token = os.getenv("TELEGRAM_BOT_TOKEN")
                    if token:
                        try:
                            async with httpx.AsyncClient() as client:
                                resp = await client.get(
                                    f"https://api.telegram.org/bot{token}/getFile",
                                    params={"file_id": file_id},
                                    timeout=10.0,
                                )
                                if resp.status_code == 200:
                                    res_json = resp.json()
                                    if res_json.get("ok"):
                                        file_path = res_json["result"].get("file_path", "")
                                        ref = f"https://api.telegram.org/file/bot{token}/{file_path}"
                        except Exception:
                            pass

                if ref:
                    attachments.append(
                        Attachment(
                            kind="image",
                            ref=ref,
                            mime="image/jpeg",
                        )
                    )

        # Arrived at
        try:
            arrived_at = datetime.fromtimestamp(float(message.date or 0), UTC)
        except (ValueError, TypeError):
            arrived_at = datetime.now(UTC)

        metadata = {
            "is_public_channel": self.config.get("is_public_channel", False),
            "was_mentioned": bool(self.config.get("was_mentioned", False)),
        }

        return ChannelMessage(
            channel=self.name,
            channel_user_id=channel_user_id,
            user_handle=user_handle,
            text=text,
            attachments=attachments,
            trust_level=trust_level,
            arrived_at=arrived_at,
            metadata=metadata,
        )

    async def send(self, reply: ChannelReply) -> Any:
        # Build sendMessage payload
        payload = {
            "chat_id": int(reply.channel_user_id)
            if reply.channel_user_id.isdigit()
            else reply.channel_user_id,
            "text": markdown_to_telegram(reply.text or ""),
            "parse_mode": "MarkdownV2",
        }

        if reply.thread_id:
            payload["message_thread_id"] = (
                int(reply.thread_id) if reply.thread_id.isdigit() else reply.thread_id
            )

        # In mock mode, call mock.send
        mock = self.config.get("mock")
        if mock is not None:
            return await mock.send(payload)

        # Real Telegram send logic
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            return payload

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json=payload,
                timeout=10.0,
            )
            # Propagate 429
            if resp.status_code == 429:
                return {
                    "ok": False,
                    "error_code": 429,
                    "status": 429,
                    "description": "Too Many Requests",
                    "parameters": resp.json().get("parameters", {}),
                }
            return resp.json()
