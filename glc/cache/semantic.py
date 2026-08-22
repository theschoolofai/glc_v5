"""Semantic response cache: embed the request, and on a close enough match
skip the provider call entirely.

The economics are different from prompt caching in kind, not degree. A prompt
cache read is billed at 0.1x-0.25x of the input rate on a call that still runs,
so the ceiling on the saving is roughly the input share of one call. A semantic
hit returns a stored response and the provider is never contacted, so the saving
is every token of that call — and the latency goes to roughly the cost of one
embedding.

The price is that matching is fuzzy. A byte-exact prefix cache cannot serve you
the wrong answer; a cosine match at 0.93 absolutely can, and does so
confidently. Hence: a high default threshold, an exact-match namespace so two
prompts must at least share model and system prompt before similarity is even
consulted, a temperature ceiling, and opt-in-by-default.

Storage is a table in the same SQLite file as the ledger, so hits survive a
restart and the hit-rate statistics are real history rather than
uptime-since-boot. Embeddings go in as JSON floats — at this scale a linear scan
over a namespace is far cheaper than the operational cost of a vector index,
and the honest ceiling (a few thousand entries) is what `max_entries` is for.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from glc import config as _config
from glc import db

_DDL = """CREATE TABLE IF NOT EXISTS semantic_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    last_hit_at REAL,
    namespace TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT '',
    query_text TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    embedding TEXT NOT NULL,
    response_json TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    usd REAL DEFAULT 0,
    hits INTEGER DEFAULT 0
)"""

_STATS_DDL = """CREATE TABLE IF NOT EXISTS semantic_cache_stats (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    lookups INTEGER DEFAULT 0,
    hits INTEGER DEFAULT 0,
    misses INTEGER DEFAULT 0,
    skipped INTEGER DEFAULT 0,
    stores INTEGER DEFAULT 0,
    embed_failures INTEGER DEFAULT 0,
    tokens_saved INTEGER DEFAULT 0,
    usd_saved REAL DEFAULT 0,
    best_similarity_missed REAL DEFAULT 0
)"""


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Returns 0.0 for mismatched or degenerate vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


@dataclass
class SemanticCacheConfig:
    enabled: bool = True
    default_on: bool = False
    threshold: float = 0.95
    ttl_seconds: int = 3600
    max_entries: int = 5000
    namespace_fields: list[str] = field(
        default_factory=lambda: [
            "model", "system", "tools", "response_format", "temperature",
            "provider", "max_tokens", "auto_route",
        ]
    )
    max_temperature: float = 0.3
    skip_when_tools: bool = True
    skip_when_stream: bool = True
    scope_dimensions: list[str] = field(default_factory=lambda: ["tenant"])
    task_type: str = "retrieval_query"
    path: str = ""

    @classmethod
    def load(cls, path: Path | str | None = None) -> SemanticCacheConfig:
        p = Path(path) if path else _config.cache_yaml_path()
        data = _config.load_yaml(p)
        sem = data.get("semantic") or {}
        known = {f for f in cls.__dataclass_fields__ if f != "path"}
        return cls(path=str(p), **{k: v for k, v in sem.items() if k in known})

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class CacheLookup:
    """The result of consulting the cache."""

    hit: bool = False
    similarity: float = 0.0
    entry_id: int | None = None
    response: dict | None = None
    provider: str | None = None
    model: str | None = None
    tokens_saved: int = 0
    usd_saved: float = 0.0
    skipped_reason: str = ""
    best_similarity: float = 0.0

    def as_dict(self) -> dict:
        return {
            "hit": self.hit,
            "similarity": round(self.similarity, 6),
            "best_similarity": round(self.best_similarity, 6),
            "entry_id": self.entry_id,
            "cached_provider": self.provider,
            "cached_model": self.model,
            "tokens_saved": self.tokens_saved,
            "usd_saved": round(self.usd_saved, 8),
            "skipped_reason": self.skipped_reason,
        }


class SemanticCache:
    """Embedding-similarity response cache backed by the gateway's SQLite file.

    `embed_fn` is an async callable `(text, task_type) -> list[float]`. It is
    injected rather than imported so the cache can be tested without an
    embedding provider, and so a deployment can point it at any embedder.
    """

    def __init__(self, config: SemanticCacheConfig | None = None, embed_fn=None):
        self.config = config or SemanticCacheConfig.load()
        self.embed_fn = embed_fn
        self._lock = threading.Lock()
        self.init_store()

    # ── storage ──────────────────────────────────────────────────────────────

    def init_store(self) -> None:
        with db.conn() as c:
            c.execute(_DDL)
            c.execute(_STATS_DDL)
            c.execute("CREATE INDEX IF NOT EXISTS idx_sem_ns ON semantic_cache(namespace, created_at DESC)")
            c.execute("INSERT OR IGNORE INTO semantic_cache_stats (id) VALUES (1)")

    def _bump(self, **deltas) -> None:
        if not deltas:
            return
        sets = ", ".join(f"{k} = {k} + ?" for k in deltas)
        with db.conn() as c:
            c.execute(_STATS_DDL)
            c.execute("INSERT OR IGNORE INTO semantic_cache_stats (id) VALUES (1)")
            c.execute(f"UPDATE semantic_cache_stats SET {sets} WHERE id = 1", tuple(deltas.values()))

    def _set_max(self, column: str, value: float) -> None:
        with db.conn() as c:
            c.execute(
                f"UPDATE semantic_cache_stats SET {column} = MAX({column}, ?) WHERE id = 1",
                (float(value),),
            )

    # ── keying ───────────────────────────────────────────────────────────────

    def namespace(self, request_fields: dict) -> str:
        """Hash of the fields that must match exactly before similarity counts.

        Two prompts are only comparable when they would be sent to the same
        model with the same system prompt and the same tools. Everything in
        `namespace_fields` is folded into this hash, so a cached answer can
        never leak across those boundaries no matter how similar the text is.
        """
        payload = {
            k: request_fields.get(k)
            for k in self.config.namespace_fields
            if request_fields.get(k) is not None
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def scope(self, principal) -> str:
        """Which principal dimensions must match for a hit to be permitted."""
        dims = self.config.scope_dimensions or []
        if not dims:
            return ""
        present = principal.present() if hasattr(principal, "present") else dict(principal or {})
        return json.dumps({d: present.get(d) for d in dims}, sort_keys=True)

    # ── admission ────────────────────────────────────────────────────────────

    def should_consult(
        self,
        opt_in: bool | None = None,
        temperature: float = 0.0,
        has_tools: bool = False,
        stream: bool = False,
    ) -> tuple[bool, str]:
        """Whether this request is even eligible for the cache."""
        if not self.config.enabled:
            return False, "semantic cache disabled in cache.yaml"
        on = self.config.default_on if opt_in is None else bool(opt_in)
        if not on:
            return False, "not opted in (set semantic_cache: true or default_on in cache.yaml)"
        if self.config.skip_when_stream and stream:
            return False, "streaming responses are not cached"
        if self.config.skip_when_tools and has_tools:
            return False, "requests carrying tool definitions are not cached"
        if temperature > self.config.max_temperature:
            return False, (
                f"temperature {temperature} above max_temperature "
                f"{self.config.max_temperature}: caller asked for variety"
            )
        return True, ""

    # ── lookup ───────────────────────────────────────────────────────────────

    async def embed(self, text: str) -> list[float] | None:
        if self.embed_fn is None:
            return None
        try:
            vec = await self.embed_fn(text, self.config.task_type)
            return list(vec) if vec else None
        except Exception:
            self._bump(embed_failures=1)
            return None

    def _live_rows(self, namespace: str, scope: str) -> list[dict]:
        now = time.time()
        args: list = [namespace, scope]
        q = "SELECT * FROM semantic_cache WHERE namespace=? AND scope=?"
        if self.config.ttl_seconds:
            q += " AND created_at >= ?"
            args.append(now - self.config.ttl_seconds)
        with db.conn() as c:
            c.execute(_DDL)
            return [dict(r) for r in c.execute(q, args).fetchall()]

    async def lookup(
        self,
        query_text: str,
        request_fields: dict,
        principal=None,
        opt_in: bool | None = None,
        temperature: float = 0.0,
        has_tools: bool = False,
        stream: bool = False,
    ) -> CacheLookup:
        """Consult the cache. A hit means the provider call must not happen."""
        ok, why = self.should_consult(
            opt_in=opt_in, temperature=temperature, has_tools=has_tools, stream=stream
        )
        if not ok:
            self._bump(skipped=1)
            return CacheLookup(skipped_reason=why)

        vec = await self.embed(query_text)
        if vec is None:
            self._bump(skipped=1)
            return CacheLookup(skipped_reason="no embedding available; cache not consulted")

        ns = self.namespace(request_fields)
        sc = self.scope(principal)
        rows = self._live_rows(ns, sc)
        self._bump(lookups=1)

        best: dict | None = None
        best_sim = 0.0
        for r in rows:
            try:
                stored = json.loads(r["embedding"])
            except (TypeError, ValueError):
                continue
            sim = cosine(vec, stored)
            if sim > best_sim:
                best_sim, best = sim, r

        if best is not None and best_sim >= self.config.threshold:
            response = json.loads(best["response_json"])
            tokens = int(best["input_tokens"] or 0) + int(best["output_tokens"] or 0)
            usd = float(best["usd"] or 0.0)
            with db.conn() as c:
                c.execute(
                    "UPDATE semantic_cache SET hits = hits + 1, last_hit_at = ? WHERE id = ?",
                    (time.time(), best["id"]),
                )
            self._bump(hits=1, tokens_saved=tokens, usd_saved=usd)
            return CacheLookup(
                hit=True,
                similarity=best_sim,
                best_similarity=best_sim,
                entry_id=int(best["id"]),
                response=response,
                provider=best["provider"],
                model=best["model"],
                tokens_saved=tokens,
                usd_saved=usd,
            )

        self._bump(misses=1)
        if best_sim:
            self._set_max("best_similarity_missed", best_sim)
        return CacheLookup(hit=False, best_similarity=best_sim)

    # ── store ────────────────────────────────────────────────────────────────

    async def store(
        self,
        query_text: str,
        request_fields: dict,
        response: dict,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        usd: float = 0.0,
        principal=None,
        embedding: list[float] | None = None,
    ) -> int | None:
        """Remember a response so a similar future request can skip the call."""
        if not self.config.enabled:
            return None
        vec = embedding if embedding is not None else await self.embed(query_text)
        if vec is None:
            return None
        ns = self.namespace(request_fields)
        sc = self.scope(principal)
        qh = hashlib.sha256(query_text.encode()).hexdigest()
        with db.conn() as c:
            c.execute(_DDL)
            cur = c.execute(
                """INSERT INTO semantic_cache
                     (created_at, namespace, scope, query_text, query_hash, embedding,
                      response_json, provider, model, input_tokens, output_tokens, usd, hits)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0)""",
                (
                    time.time(),
                    ns,
                    sc,
                    query_text[:8000],
                    qh,
                    json.dumps([float(x) for x in vec]),
                    json.dumps(response, default=str),
                    provider,
                    model,
                    int(input_tokens or 0),
                    int(output_tokens or 0),
                    float(usd or 0.0),
                ),
            )
            row_id = int(cur.lastrowid or 0)
            if self.config.max_entries:
                c.execute(
                    """DELETE FROM semantic_cache WHERE id IN (
                           SELECT id FROM semantic_cache ORDER BY created_at DESC
                           LIMIT -1 OFFSET ?)""",
                    (self.config.max_entries,),
                )
        self._bump(stores=1)
        return row_id

    # ── maintenance and reporting ────────────────────────────────────────────

    def purge(self, expired_only: bool = True) -> int:
        with db.conn() as c:
            c.execute(_DDL)
            if expired_only and self.config.ttl_seconds:
                cur = c.execute(
                    "DELETE FROM semantic_cache WHERE created_at < ?",
                    (time.time() - self.config.ttl_seconds,),
                )
            elif expired_only:
                return 0
            else:
                cur = c.execute("DELETE FROM semantic_cache")
            return cur.rowcount

    def stats(self) -> dict:
        with db.conn() as c:
            c.execute(_DDL)
            c.execute(_STATS_DDL)
            c.execute("INSERT OR IGNORE INTO semantic_cache_stats (id) VALUES (1)")
            s = dict(c.execute("SELECT * FROM semantic_cache_stats WHERE id=1").fetchone())
            e = dict(
                c.execute(
                    "SELECT COUNT(*) AS entries, COALESCE(SUM(hits),0) AS entry_hits, "
                    "MIN(created_at) AS oldest, MAX(created_at) AS newest FROM semantic_cache"
                ).fetchone()
            )
        lookups = int(s.get("lookups") or 0)
        hits = int(s.get("hits") or 0)
        s.pop("id", None)
        return {
            **s,
            **e,
            "hit_rate": round(hits / lookups, 6) if lookups else 0.0,
            "tokens_saved": int(s.get("tokens_saved") or 0),
            "usd_saved": round(float(s.get("usd_saved") or 0.0), 8),
            "config": self.config.as_dict(),
        }


def load_cache_config(path: Path | str | None = None) -> SemanticCacheConfig:
    return SemanticCacheConfig.load(path)


def build_semantic_cache(embedders=None, config_path: Path | str | None = None) -> SemanticCache:
    """Wire the cache to the gateway's own embedder failover ring.

    The cache does not get its own embedding provider: it goes through the same
    ring `/v1/embed` uses, so it inherits the rate-state, the backoff and the
    768-dim pinning, and its embedding calls show up in the ledger like any
    other.
    """
    embed_fn = None
    if embedders:

        async def embed_fn(text: str, task_type: str):  # noqa: F811
            from glc import embedders as E

            _name, result, _attempts, _latency = await E.embed_with_failover(
                embedders, text[: E.MAX_INPUT_CHARS], task_type
            )
            return result["embedding"]

    return SemanticCache(config=SemanticCacheConfig.load(config_path), embed_fn=embed_fn)
