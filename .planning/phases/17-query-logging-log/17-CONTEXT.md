# Phase 17: Query Logging (LOG) - Context

**Gathered:** 2026-06-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Log every real `/chat` USER query to Cloud SQL so it becomes the **demand-side
learning signal** for the v2.3 adaptive data loop. This is foundational plumbing:
Phase 16 thin-sliced LOG out (its falsifier hardcoded a gap and never touched a log
table); Phase 18 (GAP) is the consumer that mines this table for under-served
neighborhood/cuisine demand.

**Owns:** the new `user_query_log` table (migration) + the write on `/chat`.
**Delivers:** a populated demand-signal table downstream agents can mine.

**Critical framing — why this phase exists at all:** the existing `place_query_hits`
table looks like query logging but is **ingestion-side telemetry** — it records which
*Google Places API ingest query* returned which `place_id` at which rank during a
`scripts/ingest_places_sf.py` run (it's `place_id`-PK-FK, per-page, per-rank). It is
NOT user demand. The whole v2.3 premise is "learn from real USER queries (not
ingestion hits)." Phase 17 creates the real user-demand source that `place_query_hits`
was never about.

**Boundary guard:** Phase 17 ONLY creates + populates `user_query_log`. It does NOT
modify `scripts/coverage_agent.py` or its `recent_query` CTE (which today wrongly reads
demand diversity from `place_query_hits`). Repointing the miner at `user_query_log` is
Phase 18 (GAP) scope — see Deferred Ideas.

</domain>

<decisions>
## Implementation Decisions

### Write path & latency (the `/chat` endpoint is "the one thing that must work")
- **D-01:** **Fire-and-forget after the response is computed.** Schedule the INSERT via
  FastAPI `BackgroundTasks` (the idiomatic mechanism; `asyncio.create_task` is the
  fallback if a `BackgroundTasks` param can't be threaded cleanly) so it runs AFTER the
  `ChatResponse` is returned and adds **0ms** to user-perceived latency. Logging is a
  side-channel, never part of the response contract. (Inline `await asyncio.to_thread(...)`
  was rejected: it puts a DB round-trip on every `/chat` critical path for a signal the
  user doesn't need synchronously. Batching/buffering was rejected as premature +
  shutdown-lossy on Cloud Run.)
  - **Planner note (paths):** `/chat` has THREE turn-producing exits — the main graph
    path (`app/main.py:800`), `_decline_path` (~626), and the accept early-return inside
    `_try_accept_path`. The log should fire for a genuine user turn on whichever paths
    are in scope (at minimum the main graph path; planner decides whether the
    closure-decision early-returns also count as logged user demand — they are short
    "accept/decline/alternative" replies, arguably low demand-signal). Be explicit in the
    plan about which paths log.
  - **Planner note (mechanism):** `BackgroundTasks` is **not currently used anywhere**
    in `app/` — this phase introduces the pattern. It's standard FastAPI (add a
    `background_tasks: BackgroundTasks` param to `chat()`).
  - **Planner note (DB layer):** the app DB layer is **sync psycopg2** via a
    `ThreadedConnectionPool` (`app/db_pool.py`) borrowed through `app/db.py:get_conn()`.
    The background log function is a plain sync function using `get_conn()`; FastAPI runs
    `BackgroundTasks` in a threadpool, so no `asyncio.to_thread` wrapper is needed inside
    it. (Contrast: the agent graph offloads sync tool calls via `asyncio.to_thread` at
    `app/agent/graph.py:511` because it runs inside the async graph — different context.)

### What each record captures
- **D-02:** **Raw message + already-extracted slots + lightweight metadata.** Schema:
  ```
  user_query_log:
    id                      uuid pk    (uuid_generate_v4(), matches place_query_hits style)
    message                 text not null     -- raw user query, verbatim
    requested_primary_types text[]            -- the extracted_types /chat already computed
    num_stops               int null          -- explicit_num_stops_from_conversation(...)
    rag_label               text              -- active model/version label (request.app.state.rag_label)
    created_at              timestamptz not null default now()
    session_id              text null         -- OPTIONAL turn/session marker (see note)
  ```
  Rationale: reuse the structured slots `/chat` **already extracts for free** before the
  graph runs (`extracted_types` at `app/main.py:724`; `explicit_num_stops_from_conversation(req.history, req.message)`
  at `app/main.py:786`), so Phase 18's miner gets structured demand without re-running
  slot NLP. Stops short of logging the full reply / committed place_ids (rejected as
  over-capture: largest PII+storage surface, overlaps the MLflow tracing already on
  `/chat` via `trace_request`). Raw-message-only was rejected: it pushes slot extraction
  downstream that `/chat` has already done.
  - **Planner note (pass slots as args):** because the write is fire-and-forget (D-01),
    the log function must receive `message`, `extracted_types`, `num_stops`, and
    `rag_label` as **arguments captured in the handler** (they live inside the
    `with trace_request(...)` block) — it must not re-derive them.
  - **Planner note (`session_id`):** `/chat` is stateless; `conversation_state` is opaque
    and there is **no server-side session id today**. Leave `session_id` nullable; the
    planner MAY derive a cheap per-turn marker (e.g. a request-scoped uuid or the
    existing `trace_id`) or leave it null. Do not invent session infrastructure — that's
    out of scope.

### New table vs extend existing
- **D-03:** **Brand-new `user_query_log` table; Phase 17 creates + populates ONLY.** A
  user query has no `place_id` at log time, so it physically cannot extend the
  `place_id`-PK-FK `place_query_hits`. Add via a new Alembic migration
  (`make migration MSG="..."`; current head is `e0cd7069bc8f`). Phase 17 does **not**
  repoint `coverage_agent.py`'s `recent_query` CTE — that rewiring (and the whole miner
  rework) is Phase 18, which reworks the miner regardless; doing it now risks doing it
  twice and bleeds GAP scope into LOG. Keeps Phase 17 a zero-behavior-change foundational
  slice.

### Failure & privacy posture
- **D-04:** **Store raw message verbatim — no PII scrubbing, no retention cap — and
  document the decision.** The loop's entire value is mining real demand text, and SF
  place queries ("date night in North Beach, 3 stops, under $$$") are intrinsically
  low-PII; scrubbing would degrade the exact signal Phase 18 needs. Document the
  raw-store decision in this CONTEXT + a comment on the migration, noting this is a
  **private capstone DB, not a public service**. (Retention window + write-time scrubbing
  both rejected as premature for capstone volume / Phase 19+ concerns.)
  - **Failure posture (settled by D-01):** because the write is fire-and-forget after the
    response, a logging failure **cannot affect the user's reply**. The background log
    function must still **swallow all exceptions and emit a `logger.warning(..., exc_info=True)`**
    so an unhandled error never produces a noisy stack trace or (in the
    `asyncio.create_task` fallback) an unawaited-exception warning. Fail-open, fail-quiet.

### Claude's Discretion
- Exact SQL column types / index choices (e.g. whether to index `created_at` for the
  miner's time-window queries — likely yes, mirroring `place_query_hits`' indexes).
- Internal helper decomposition + unit-test seams for the log-write function (must be
  unit-testable with a mocked connection, independent of a live `/chat` call).
- Whether to derive a per-turn `session_id` marker or leave it null (per D-02 note).
- Which of the closure-decision early-return paths (if any beyond the main graph path)
  also log a user turn (per D-01 paths note).
- Migration revision id / filename (auto-generated by `make migration`).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope (read first)
- `.planning/ROADMAP.md` § "Phase 17: Query Logging (LOG)" — the one-line goal: log
  `/chat` user queries to Cloud SQL as the loop's learning signal. (NOTE: there is no
  v2.3-REQUIREMENTS.md and no formal LOG-01..05 requirement text exists anywhere — the
  "LOG-01..05" codes in `16-CONTEXT.md`/`16-DISCUSSION-LOG.md` are forward-references
  only. This CONTEXT's decisions are the authoritative spec for Phase 17.)
- `.planning/phases/16-loop-falsifier/16-CONTEXT.md` § Deferred Ideas — establishes LOG
  as the deferred "`/chat` user-query logging to Cloud SQL → Phase 17" slice and the
  "LOG table is the eventual source of the gap" framing.
- `.planning/PROJECT.md` § "Current State" — v2.3 milestone goal + the sandbox-not-prod
  invariant (LOG writes go to the REAL prod Cloud SQL `/chat` DB; the sandbox invariant
  is a Phase 16/falsifier concern, not Phase 17 — Phase 17 logs live prod traffic).

### The `/chat` write site
- `app/main.py:662` → `async def chat(req: ChatRequest, request: Request)` — the endpoint
  to instrument. Slots are available at `:724` (`extracted_types`) and `:786`
  (`explicit_num_stops_from_conversation(req.history, req.message)`); `rag_label` at
  `:668`. Response built at `:800` (main path), `:614`/`:654` (closure early-returns).
- `app/main.py:215` → `class ChatRequest` / `:227` `class ChatResponse` — request/response
  contracts (no schema change needed for LOG; it's a side-channel write).

### DB access + migration mechanism (compose, don't reinvent)
- `app/db.py:37` → `get_conn()` context manager — the sanctioned pooled-connection borrow
  for non-async code (the background log fn uses this).
- `app/db_pool.py` → `ThreadedConnectionPool` lifecycle (`get_connection` / `return_connection`
  / `close_db_pool`). Sync psycopg2.
- `app/agent/graph.py:511` — reference ONLY for the "sync DB in async context →
  `asyncio.to_thread`" note in D-01 (NOT the pattern Phase 17 uses; BackgroundTasks
  already runs in a threadpool).
- `alembic/versions/2026_05_08_1000-34679f77f726_add_places_ingest_query_proposals.py` —
  closest analog migration for a small new table (uuid pk option, text columns,
  `server_default NOW()`, a status index). Mirror its style.
- `alembic/versions/2026_05_14_1200-e0cd7069bc8f_add_place_relations.py` — current head
  revision (`e0cd7069bc8f`); new migration's `down_revision`.
- `CLAUDE.md` § Architecture (Alembic) — `make migration MSG="..."` to create,
  `make migrate` to apply; DB URL via `app.db_url.resolve_alembic_database_url()`.

### The wrong-source table this phase replaces (context, do not modify)
- `scripts/db/init.sql:65` → `place_query_hits` schema — the INGESTION-side table
  (`place_id` FK, `field_mode`, `page_number`, `rank_in_page`). Shows what LOG is NOT.
- `scripts/ingest_places_sf.py:471` (CREATE) + `:562` (INSERT) — proves `place_query_hits`
  is written by the INGEST script, not by `/chat`.
- `scripts/coverage_agent.py:53` `gather_stats()` + `:86`–`:100` `recent_query_diversity`
  CTE — the consumer that today reads demand diversity from `place_query_hits` (the wrong
  source). Phase 18 repoints this at `user_query_log`; Phase 17 leaves it untouched.

### Memory cross-refs (machine-specific gotchas)
- `project_local_postgres_port_collision` — Postgres.app squats host 5432; project uses
  5433 for both prod + sandbox URLs (relevant if testing the migration locally).
- `project_prod_alembic_w7_divergence` — prod alembic was stamped to an unmerged W7
  migration historically; verify head alignment before assuming `alembic upgrade` is
  clean against prod Cloud SQL.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`get_conn()` (`app/db.py`)** — pooled sync connection borrow with rollback-on-return
  safety; the background log function uses this directly.
- **`places_ingest_query_proposals` migration** — copy-paste-shaped analog for the new
  `user_query_log` table (uuid pk via `uuid_generate_v4()`, text cols, `NOW()` default,
  index). `uuid-ossp`/`uuid_generate_v4` is already enabled (see `place_query_hits`).
- **Already-extracted slots in `/chat`** — `extracted_types` (list[str]) + `num_stops`
  (int|None) are computed before the graph runs and can be handed to the logger for free.

### Established Patterns
- **Sync psycopg2 + ThreadedConnectionPool** is the ONLY DB access pattern in this
  codebase (no asyncpg). The log write follows it.
- **Migrations via Alembic** (`make migration` / `make migrate`); never hand-edit schema.
- **`uuid PRIMARY KEY DEFAULT uuid_generate_v4()`** is the established pk style for
  append-only event tables (`place_query_hits`).

### Integration Points / MISSING (build new)
- **`user_query_log` table** — net-new (Alembic migration).
- **`BackgroundTasks` usage** — net-new to `app/`; introduce in `chat()`.
- **A small `log_user_query(...)` write function** — net-new; unit-testable with a mocked
  connection, called from the background task.

</code_context>

<specifics>
## Specific Ideas

- Schema shape locked in D-02 (see the column list there). The user explicitly endorsed
  the "raw message + extracted slots + metadata" shape over both the minimal and the
  full-output variants.
- Honesty framing for the essay/interview (parallels the Phase 16 honesty note): "Phase
  17 turns the loop's learning signal from *ingestion hits* (`place_query_hits` — what
  our own Google Places ingest queries returned) into *real user demand*
  (`user_query_log` — what users actually asked `/chat` for). Phase 18 mines the demand
  table for under-served neighborhood/cuisine gaps."

</specifics>

<deferred>
## Deferred Ideas

- **Repoint `coverage_agent.gather_stats` `recent_query` CTE from `place_query_hits` →
  `user_query_log`** → **Phase 18 (GAP).** This is the payoff of Phase 17, but it's part
  of the miner rework, not the logging slice. Phase 17 deliberately leaves the miner
  untouched (D-03) so it ships zero behavior change to the existing loop.
- **Neighborhood as a structured slot.** Phase 18's miner buckets demand by **neighborhood
  AND cuisine**. `/chat` extracts `requested_primary_types` (≈ cuisine/category) but does
  NOT extract neighborhood — so the cuisine side of demand is captured structured in
  `user_query_log`, while the neighborhood side lives only in the raw `message` text.
  This is a defensible thin-slice (the raw text is retained, so the signal is not lost —
  just not pre-structured). Whether to add structured neighborhood extraction is a
  Phase 18 decision (the miner can NLP it from `message`, or `/chat` intake could be
  extended). **Flagged so Phase 18 is not surprised.**
- **Retention window / PII scrubbing / cleanup job** → Phase 19+ if ever. Rejected for
  Phase 17 (D-04).
- **Logging the full reply + committed place_ids per turn** → out of scope; overlaps
  MLflow `trace_request` tracing already on `/chat`. Rejected in D-02.
- **Server-side session/turn infrastructure** → out of scope; `/chat` stays stateless,
  `session_id` is nullable (D-02 note).

</deferred>

---

*Phase: 17-query-logging-log*
*Context gathered: 2026-06-16*
