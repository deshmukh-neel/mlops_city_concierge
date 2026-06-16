# Phase 17: Query Logging (LOG) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-16
**Phase:** 17-query-logging-log
**Areas discussed:** Write path & latency, What each record captures, New table vs extend existing, Failure & privacy posture

---

## Write path & latency

| Option | Description | Selected |
|--------|-------------|----------|
| Fire-and-forget after response | Schedule INSERT via FastAPI BackgroundTasks AFTER the response is computed; 0ms added latency; logging is a side-channel | ✓ |
| Inline via asyncio.to_thread | Await the INSERT in-handler (mirrors graph.py:511); simplest/most consistent but +1 DB round-trip on every /chat critical path | |
| Batched / buffered | Accumulate in memory, flush in batches; lowest DB load but shutdown-lossy on Cloud Run + premature for capstone volume | |

**User's choice:** Fire-and-forget after response (D-01)
**Notes:** Logging must never touch the user's critical path on the one endpoint that must always work. Planner notes captured in CONTEXT: BackgroundTasks is new to the app (introduce the pattern); there are 3 turn-producing exits (main graph path + 2 closure early-returns) and the plan must be explicit about which log; the bg function uses sync `get_conn()` (FastAPI runs BackgroundTasks in a threadpool, no asyncio.to_thread needed).

---

## What each record captures

| Option | Description | Selected |
|--------|-------------|----------|
| Raw message + extracted slots + metadata | message + already-extracted requested_primary_types + num_stops + rag_label + created_at + nullable session_id; reuses slots /chat already computes | ✓ |
| Raw message + timestamp only | Smallest schema; pushes slot extraction downstream (duplicates /chat intake) | |
| Everything incl. reply + committed place_ids | Richest, but logs full agent output (largest PII/storage surface) and overlaps MLflow tracing | |

**User's choice:** Raw message + extracted slots + metadata (D-02)
**Notes:** Reuse `extracted_types` (main.py:724) + `explicit_num_stops_from_conversation` (main.py:786) that /chat computes for free, so Phase 18's miner gets structured demand without re-NLP. Because the write is fire-and-forget, slots must be passed as args captured in the handler. `session_id` left nullable — /chat is stateless, no server-side session today; planner may derive a cheap turn marker or leave null (don't build session infra).

---

## New table vs extend existing

| Option | Description | Selected |
|--------|-------------|----------|
| New table, leave miner rewiring to Phase 18 | Create user_query_log; Phase 17 only creates + populates; repointing coverage_agent's recent_query CTE is Phase 18 scope | ✓ |
| New table + repoint miner now | Also repoint coverage_agent CTE in Phase 17; changes existing-loop behavior in a plumbing phase, risks doing it twice | |
| Different name | Discuss naming (chat_query_log / user_queries / demand_log) | |

**User's choice:** New table, leave miner rewiring to Phase 18 (D-03)
**Notes:** A user query has no place_id at log time, so it can't extend the place_id-PK-FK `place_query_hits`. New `user_query_log` table via Alembic (head e0cd7069bc8f). Phase 17 ships zero behavior change to the existing loop; Phase 18 reworks the miner regardless. Key framing surfaced during scout: `place_query_hits` is INGESTION-side telemetry (written by ingest_places_sf.py), not user demand — that's exactly the gap v2.3 closes.

---

## Failure & privacy posture

| Option | Description | Selected |
|--------|-------------|----------|
| Store raw, no scrubbing, document it | Verbatim store, no scrubbing/retention; document raw-store + private-capstone-DB; bg task swallows + warns | ✓ |
| Store raw + retention window | Verbatim + documented deletion window; adds a cleanup mechanism (Phase 19+ scope) | |
| Light scrubbing on write | Strip emails/phones before store; adds write-path step, risks mangling legit query text | |

**User's choice:** Store raw, no scrubbing, document it (D-04)
**Notes:** SF place queries are intrinsically low-PII; scrubbing would degrade the exact demand signal Phase 18 mines. Failure posture is settled by D-01 (fire-and-forget → a log failure can't affect the reply); the bg function still must swallow all exceptions + `logger.warning(exc_info=True)` so nothing noisy escapes. Document raw-store in CONTEXT + the migration comment, noting it's a private capstone DB.

---

## Claude's Discretion

- Exact SQL column types / index choices (likely index `created_at` for the miner's time-window queries, mirroring place_query_hits).
- Internal helper decomposition + unit-test seams for the log-write function (mock-connection unit-testable).
- Whether to derive a per-turn session_id marker or leave it null.
- Which closure-decision early-return paths (beyond the main graph path) also log a user turn.
- Migration revision id / filename (auto-generated by `make migration`).

## Deferred Ideas

- Repoint coverage_agent's `recent_query` CTE from place_query_hits → user_query_log → Phase 18 (GAP).
- Structured neighborhood extraction — miner buckets by neighborhood + cuisine, but /chat extracts only cuisine-ish `requested_primary_types`; neighborhood lives in raw message text. Flagged so Phase 18 isn't surprised.
- Retention window / PII scrubbing / cleanup job → Phase 19+ if ever.
- Logging full reply + committed place_ids → out of scope (overlaps MLflow tracing).
- Server-side session/turn infrastructure → out of scope; /chat stays stateless.
