"""Strip host filesystem paths out of text on its way to a channel.

A channel is a third-party service. Telegram, Discord, Slack and Matrix all
store what is sent to them on infrastructure the operator does not control, and
a message is not recallable once delivered. An absolute path is therefore not a
cosmetic blemish in an answer; it discloses the operator's username, home
directory and installation layout to an outside party.

This is the net, not the fix. The real fix is that a capability result should
never carry a host path in the first place, because a path the model can read
is a path the model can quote. But the net covers three things no producer-side
fix can reach:

* text already written into durable memory before the producer was fixed, which
  replays through recall,
* text authored by the planner rather than by a capability, such as an approval
  question, which reaches a channel without passing through the agent's reply
  path at all,
* any capability added later by someone who has not read this comment.

Applied at the ``ChannelReply`` envelope so it holds for every adapter at once.
Redacting inside one adapter would protect one channel and quietly leave the
other thirteen exposed.
"""

from __future__ import annotations

import re

# http(s) URLs are stashed before any path pattern runs and restored afterwards.
# Without this, a perfectly legitimate link such as
# https://example.com/Users/alice/report is mangled, because the POSIX-home
# pattern matches inside its path component.
_HTTP = re.compile(r"https?://\S+")

# Ordered most specific first. Each stops at whitespace or a closing delimiter
# so a trailing quote, bracket or backtick is not swallowed into the match.
_TERMINATOR = r"[^\s`'\")\]}>]"

_PATTERNS = (
    # file://C:\..., file:///C:/..., file:/C:/...
    re.compile(rf"(?i)file:/{{0,3}}[A-Za-z]:[\\/]{_TERMINATOR}*"),
    # file:///home/..., any other file: URI
    re.compile(rf"(?i)file://{_TERMINATOR}+"),
    # UNC share: \\server\share\...
    re.compile(rf"\\\\[A-Za-z0-9._-]+\\{_TERMINATOR}*"),
    # Bare Windows path: C:\... or C:/...
    # A single letter then a colon then a separator. "10:30" cannot match (a
    # digit is not [A-Za-z]) and neither can "mxc://" or "art:x" (no word
    # boundary before the final letter of a multi-letter scheme).
    re.compile(rf"(?i)\b[A-Za-z]:[\\/]{_TERMINATOR}*"),
    # POSIX home directories, which carry the username just as a drive path does.
    re.compile(rf"(?<![\w.])/(?:home|Users)/{_TERMINATOR}+"),
)

_SEPARATORS = re.compile(r"[\\/]")


def _label(match: re.Match[str]) -> str:
    """Replace a path with its basename only.

    The basename is kept deliberately. "reminders.txt" is the part that carries
    meaning for the reader and it names nothing about the machine, so an answer
    stays intelligible instead of becoming a sentence with a hole in it.
    """
    name = _SEPARATORS.split(match.group(0))[-1].strip()
    return f"[local file: {name}]" if name else "[local file]"


def redact_local_paths(text: str | None) -> str | None:
    """Return `text` with any host filesystem path reduced to its basename.

    None and empty strings pass through untouched, so this is safe to apply as
    a field validator on an optional field.
    """
    if not text:
        return text

    held: list[str] = []

    def stash(match: re.Match[str]) -> str:
        held.append(match.group(0))
        return f"\x00{len(held) - 1}\x00"

    out = _HTTP.sub(stash, text)
    for pattern in _PATTERNS:
        out = pattern.sub(_label, out)
    for index, original in enumerate(held):
        out = out.replace(f"\x00{index}\x00", original)
    return out
