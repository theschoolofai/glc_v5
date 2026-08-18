"""Cross-container record of which channels have connected at least once.

`registered_channels` used to live only on `app.state` -- in-process memory.
The gateway can autoscale to several Modal containers, so a `list_channels`
call landing on a container whose process never held a given channel's
WebSocket reported it as not connected, even while that channel's bridge was
live and connected on a different container. This store makes "has this
channel connected" a fact in shared SQLite (the persistent volume in
production) instead of a per-process fact, so every container sees the same
answer.

Once a channel connects it stays marked connected, matching the previous
in-memory behaviour (`registered_channels` was append-only, never pruned on
disconnect).
"""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

DEFAULT_DIR = Path(os.path.expanduser("~/.glc"))


def _resolve_path() -> str:
    """Resolve at call time, not import time, so tests that swap the env
    var see the change."""
    return os.getenv("GLC_PRESENCE_DB", str(DEFAULT_DIR / "channel_presence.sqlite"))


@contextmanager
def _conn():
    p = _resolve_path()
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p, isolation_level=None)  # autocommit; each write flushes
    c.row_factory = sqlite3.Row
    try:
        yield c
    finally:
        c.close()


_schema_ready = False


def _ensure_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    with _conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS channel_presence (
                channel TEXT PRIMARY KEY,
                first_connected_at REAL NOT NULL
            )"""
        )
    _schema_ready = True


def mark_connected(channel: str) -> None:
    _ensure_schema()
    with _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO channel_presence (channel, first_connected_at) VALUES (?,?)",
            (channel, time.time()),
        )


def connected_channels() -> set[str]:
    _ensure_schema()
    with _conn() as c:
        return {r["channel"] for r in c.execute("SELECT channel FROM channel_presence").fetchall()}
