"""Attribution: the five principal dimensions, the additive migration, and the
guarantee that a v3 ledger row keeps its meaning."""

from __future__ import annotations

import sqlite3

import pytest

from glc import db
from glc.economics import meter as M
from glc.economics import pricing as P


@pytest.fixture(autouse=True)
def _fresh():
    db.init()
    P.reload_pricing()
    M.reset_meter()
    yield


# ── the schema migration ────────────────────────────────────────────────────


def test_v4_columns_exist_after_init():
    with db.conn() as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(calls)").fetchall()}
    assert set(db.V4_COLUMNS) <= cols
    # every v3 column is still there
    for v3col in ("agent", "session", "call_role", "router_decision", "retries", "embed_dim"):
        assert v3col in cols


def test_migration_is_additive_over_a_real_v3_table(monkeypatch, tmp_path):
    """Create the exact v3 `calls` table, insert a v3 row, migrate, and prove
    the row survives and every v3 query still answers."""
    path = tmp_path / "v3.sqlite"
    monkeypatch.setattr(db, "DB_PATH", str(path))
    v3_ddl = """CREATE TABLE calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
        provider TEXT NOT NULL, model TEXT NOT NULL,
        input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0,
        cache_create_tokens INTEGER DEFAULT 0, cache_read_tokens INTEGER DEFAULT 0,
        latency_ms INTEGER DEFAULT 0, status TEXT, error TEXT,
        prompt_chars INTEGER DEFAULT 0, response_chars INTEGER DEFAULT 0,
        override TEXT, attempted TEXT, tool_calls INTEGER DEFAULT 0,
        reasoning_applied INTEGER DEFAULT 0, tool_dialect TEXT,
        call_role TEXT DEFAULT 'worker', router_decision TEXT, embed_dim INTEGER,
        agent TEXT, session TEXT, retries INTEGER DEFAULT 0)"""
    raw = sqlite3.connect(str(path))
    raw.execute(v3_ddl)
    raw.execute(
        "INSERT INTO calls (ts, provider, model, input_tokens, output_tokens, status, agent, session)"
        " VALUES (1000.0, 'groq', 'old-model', 100, 50, 'ok', 'legacy-agent', 'legacy-session')"
    )
    raw.commit()
    raw.close()

    added = db.migrate()
    assert set(added) == set(db.V4_COLUMNS) | set(db.V5_COLUMNS)

    # v3 row is intact and still readable by the v3 query
    rows = db.recent(limit=10)
    assert len(rows) == 1
    assert rows[0]["agent"] == "legacy-agent"
    assert rows[0]["input_tokens"] == 100
    # new columns default benignly rather than to garbage
    assert rows[0]["usd"] == 0
    assert rows[0]["tenant"] is None

    by_agent = db.by_agent(since=0)
    assert by_agent["legacy-agent"][0]["in_tok"] == 100

    # and migrating again is a no-op
    assert db.migrate() == []


def test_migrate_on_a_v4_table_adds_nothing():
    assert db.migrate() == []


# ── recording ───────────────────────────────────────────────────────────────


def test_record_prices_the_call_and_writes_all_five_dimensions():
    m = M.get_meter()
    p = M.Principal(tenant="t1", project="proj-a", user="u-7", agent="researcher", session="run-42")
    rec = m.record(
        provider="gemini_1",
        model="gemini-3.1-flash-lite",
        principal=p,
        usage=M.Usage(input_tokens=1_000_000, output_tokens=1_000_000),
        latency_ms=120,
        status="ok",
    )
    assert rec.usd == pytest.approx(1.75)
    assert rec.price_source == "model"
    assert rec.row_id > 0

    row = db.recent(limit=1)[0]
    assert row["tenant"] == "t1"
    assert row["project"] == "proj-a"
    assert row["user"] == "u-7"
    assert row["agent"] == "researcher"
    assert row["session"] == "run-42"
    assert row["usd"] == pytest.approx(1.75)


def test_spend_is_derived_from_the_ledger_not_a_counter():
    m = M.get_meter()
    p = M.Principal(user="u-1")
    for _ in range(3):
        m.record(
            provider="gemini_1",
            model="gemini-3.1-flash-lite",
            principal=p,
            usage=M.Usage(input_tokens=1_000_000),
        )
    assert db.spend_usd("user", "u-1", since=0) == pytest.approx(0.75)
    # a different principal is untouched
    assert db.spend_usd("user", "u-2", since=0) == 0.0


def test_rollup_reports_dollars_per_dimension():
    m = M.get_meter()
    m.record(
        provider="groq",
        model="openai/gpt-oss-120b",
        principal=M.Principal(tenant="acme", agent="a1"),
        usage=M.Usage(input_tokens=1_000_000, output_tokens=1_000_000),
    )
    m.record(
        provider="groq",
        model="openai/gpt-oss-120b",
        principal=M.Principal(tenant="acme", agent="a2"),
        usage=M.Usage(input_tokens=1_000_000),
    )
    by_tenant = m.rollup("tenant", since=0, group_by_provider=False)
    assert len(by_tenant) == 1
    assert by_tenant[0]["principal"] == "tenant:acme"
    assert by_tenant[0]["calls"] == 2
    assert by_tenant[0]["dollars"] == pytest.approx(0.15 + 0.75 + 0.15)

    by_agent = {r["principal_value"]: r for r in m.rollup("agent", since=0)}
    assert set(by_agent) == {"a1", "a2"}
    assert by_agent["a1"]["dollars"] == pytest.approx(0.90)


def test_rollup_all_covers_every_dimension():
    out = M.get_meter().rollup_all(since=0)
    assert list(out) == list(M.DIMENSIONS) == ["tenant", "project", "user", "agent", "session"]


def test_unknown_dimension_is_rejected():
    with pytest.raises(ValueError):
        db.by_principal(dimension="favourite_colour")
    with pytest.raises(ValueError):
        db.spend_usd("favourite_colour", "blue")


# ── Principal ───────────────────────────────────────────────────────────────


def test_principal_parse_roundtrip():
    p = M.Principal.parse("session:run-42")
    assert p.session == "run-42"
    assert p.keys() == ["session:run-42"]


def test_principal_parse_rejects_unknown_dimension():
    with pytest.raises(ValueError):
        M.Principal.parse("nonsense:x")
    with pytest.raises(ValueError):
        M.Principal.parse("agent:")


def test_principal_from_request_pulls_only_what_exists():
    class Req:
        agent = "a"
        session = "s"
        tenant = None

    p = M.Principal.from_request(Req())
    assert p.present() == {"agent": "a", "session": "s"}
    assert p.keys() == ["agent:a", "session:s"]


def test_empty_principal_is_falsy():
    assert not M.Principal()
    assert M.Principal(user="u")


def test_with_defaults_only_fills_gaps():
    p = M.Principal(agent="a").with_defaults(tenant="default-tenant", agent="ignored")
    assert p.tenant == "default-tenant"
    assert p.agent == "a"


def test_usage_from_provider_result_maps_the_cache_columns():
    u = M.Usage.from_provider_result(
        {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 2,
        }
    )
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens) == (10, 5, 3, 2)
    assert u.total == 20


def test_log_call_still_accepts_only_v3_arguments():
    """Nothing in v3's call signature became mandatory."""
    row_id = db.log_call(provider="groq", model="m", input_tokens=1, output_tokens=2)
    assert row_id > 0
    row = db.recent(limit=1)[0]
    assert row["usd"] == 0
    assert row["call_role"] == "worker"
