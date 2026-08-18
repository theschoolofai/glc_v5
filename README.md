# glc_v5

`glc_v5` is the local Gateway for LLMs and Channels used by EAG V3. It owns provider credentials, model routing, rate limits, cost and audit records, voice, and channel adapters. Agent runtimes call it over HTTP; they do not import its provider code or read its keys.

v5 keeps every v4 model and economics contract, and adds one channel-to-agent
connection. Every enabled adapter converts its native payload into the same
`ChannelMessage`; GLC verifies the sender, forwards that envelope to S16, and
returns S16's `ChannelReply` through the same adapter.

## Channel setup UI

Open `http://127.0.0.1:8111/channels` and enter the installation control token
from `~/.glc/install_token`. The page covers every registered adapter and
returns only whether a value is saved — never the secret itself. It offers
saveable fields only when the existing adapter consumes those exact values;
adapters that need an injected external client/transport are guide-only.
Settings are stored locally in `~/.glc/channel_secrets.json` with owner-only
permissions and are loaded on the next GLC restart. Existing environment
variables win.

“Configured” means local values are present; it does **not** claim the
provider has been authenticated or live-tested. Gmail presently uses its local
OAuth helper rather than a browser callback, and several adapters still need a
provider bridge/polling process. The generated ingress endpoint is shown only
for the two adapters whose current native protocol GLC actually handles:
`webhook` and Meta WhatsApp (including its Meta `hub.*` verification GET).
Set `GLC_PUBLIC_BASE` before using either externally, then perform a real
inbound test before enabling autonomous actions.

## Requirements

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/)
- At least one configured model provider
- Ollama only if you want local generation or gateway embeddings

## Install and run

```bash
uv sync
cp .env.example .env
# Edit .env locally. Never commit it.
uv run glc serve
```

The gateway listens on `http://127.0.0.1:8111` by default.

- Dashboard: `http://127.0.0.1:8111/`
- Help: `http://127.0.0.1:8111/help`
- OpenAPI: `http://127.0.0.1:8111/docs`
- Health: `http://127.0.0.1:8111/healthz`

## Multiple Gemini keys

Number the keys in `.env`:

```dotenv
GEMINI_API_KEY_1=replace-me
GEMINI_API_KEY_2=replace-me
GEMINI_API_KEY_3=replace-me
```

The gateway registers them as independently metered providers such as `gemini_1`, `gemini_2`, and `gemini_3`. A caller can request the logical provider `gemini`; the router selects an available numbered slot. The dashboard shows each slot separately.

## Smoke test

```bash
curl -s http://127.0.0.1:8111/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Say hello in one line."}],
    "provider": "gemini",
    "max_tokens": 80,
    "temperature": 0
  }'
```

The response reports the actual provider slot and model used.

## Connect every channel to S16

Run GLC and S16 as separate services. Put the same private bridge token in both
local `.env` files:

```dotenv
# glc_v5/.env
S16_BASE_URL=http://127.0.0.1:8113
GLC_S16_BRIDGE_TOKEN=replace-with-a-long-random-local-token

# S16Code/.env
GLC_BASE_URL=http://127.0.0.1:8111
S16_CHANNEL_BRIDGE_TOKEN=replace-with-a-long-random-local-token
```

`GET /v1/channels` is the live catalogue. There is no second list of Telegram,
Gmail, Slack, or future adapter names in S16. Inbound adapter WebSockets all use
the same bridge; proactive work uses `POST /v1/channels/{name}/send`. A newly
installed adapter is therefore discovered without changing agent code.

GLC recomputes trust from its pairing store before forwarding a message. It
never accepts a client-supplied owner identity. The bridge token authenticates
GLC to S16; it is not a user token and must not be committed.

## What v4 adds

Five modules and four config files. Nothing here is required: with the shipped
configuration the gateway behaves exactly like `glc_v3`, and every feature is
armed by editing YAML rather than Python.

| Module | Does |
|---|---|
| `glc/economics/pricing.py` | Per-**model** pricing from `pricing.yaml`, with cache-read/cache-write and batch multipliers as data. v3's per-provider table survives as the fallback, so an unpriced model reports what v3 reported. |
| `glc/economics/meter.py` | Attribution across five dimensions — **tenant, project, user**, agent, session. One priced ledger row per call. |
| `glc/economics/budget.py` | The **hard controller**. Admission on a projected worst-case cost *before* the provider is called; breach → HTTP 402 with the numbers. |
| `glc/telemetry/otel.py` | OTel `chat` spans with `gen_ai.*` attributes plus computed cost. Content capture **off** by default. |
| `glc/routing/policy.py` | Role→tier policy, cost/quality candidate ordering, a servable `HUGE` tier, cascade escalation. |
| `glc/cache/semantic.py` | Response cache: embed the request, cosine-match, and on a hit **skip the provider call entirely**. |

### Config, not code

| File | Packaged default | Override with |
|---|---|---|
| `pricing.yaml` | `glc/economics/pricing.yaml` | `~/.glc/pricing.yaml` or `GLC_PRICING_YAML` |
| `budgets.yaml` | `glc/economics/budgets.yaml` (ships **empty** — nothing is refused) | `~/.glc/budgets.yaml` or `GLC_BUDGETS_YAML` |
| `routing.yaml` | `glc/routing/routing.yaml` | `~/.glc/routing.yaml` or `GLC_ROUTING_YAML` |
| `cache.yaml` | `glc/cache/cache.yaml` (semantic cache ships **opt-in**) | `~/.glc/cache.yaml` or `GLC_CACHE_YAML` |

Adding a model, a role, a tier or a budget is an edit to one of those files.
There is no Python change and no list of names inside the library. A malformed
budget or routing file raises rather than silently degrading into "no budget" —
`GET /v1/budget` and `app.state.config_errors` report what failed.

### New endpoints

```
GET    /v1/budget                 every loaded policy
GET    /v1/budget/{principal}     limit, spend, remaining — e.g. /v1/budget/session:run-42
POST   /v1/budget                 arm or move a ceiling (install token; limit_usd: 0 = stop now)
DELETE /v1/budget/{principal}     drop a runtime override (install token)
GET    /v1/cost/by_principal      five-dimension rollup (superset of /v1/cost/by_agent)
GET    /v1/cache/stats            hit rate, tokens and dollars saved
GET    /v1/pricing                the resolved price table, or one (provider, model)
GET    /v1/telemetry              tracer state and exporters
POST   /v1/cache/purge            drop expired (or all) cache entries (install token)
```

`POST /v1/chat` keeps its contract. New **optional** request fields:
`tenant`, `project`, `user`, `semantic_cache`, `batch`, `cost_quality_tradeoff`,
`escalate`. New response fields `cost`, `budget`, `cache`, `trace` are `null`
unless the corresponding feature ran.

### Arming the budget controller

```bash
# ceiling for one run, enforced before any provider is contacted
curl -s -X POST http://127.0.0.1:8111/v1/budget \
  -H "Authorization: Bearer $(cat ~/.glc/install_token)" \
  -H 'Content-Type: application/json' \
  -d '{"principal": "session:run-42", "limit_usd": 0.50, "period": "lifetime"}'

# a call whose projected cost does not fit gets HTTP 402 and is never sent
curl -s -X POST http://127.0.0.1:8111/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "...", "provider": "gemini", "max_tokens": 2048, "session": "run-42"}'
```

Budgets are enforced in code, not stated in a prompt. Token elasticity means a
model asked to stay under a limit does not, so the ceiling lives where the model
cannot argue with it.

### Tracing

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 uv run glc     # export to Jaeger
GLC_OTEL_CONSOLE=1 uv run glc                                    # dump spans to stdout
GLC_OTEL_CAPTURE_CONTENT=1 ...                                   # attach prompts (PII — off by default)
```

With no endpoint set, spans are a no-op and no collector is needed.

### Ledger migration

The `calls` table gains twelve nullable columns (`tenant`, `project`, `user`,
`usd`, `cache_hit`, …) through `ALTER TABLE ADD COLUMN`. A v3 database is
upgraded in place on boot: existing rows keep their meaning and every v3 query
still answers. Spend is read back out of this ledger rather than tracked in a
parallel counter, so it survives a restart and cannot drift from what was billed.

## Relationship to Session 16

`glc_v5` is a dependency of `S16Code`, not its parent project. The ownership boundary is deliberate:

| `glc_v5` owns | `S16Code` owns |
|---|---|
| Keys, providers and models | Live task graph |
| Routing, quotas and costs | Memory and semantic indexing |
| Channels and voice | A2A discovery and delegation |
| `/v1/chat` | `/v1/agent/*` |

`glc_v5` exposes only the narrow channel bridge into S16; it does not own the
agent graph. S16 does not import adapters or provider credentials.

## Development

```bash
uv run ruff check .
uv run pytest -q
```

Never commit `.env`, API keys, local databases, audit records, pairing state, or user memory.

## License

MIT. See `LICENSE`.
