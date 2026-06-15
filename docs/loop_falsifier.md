# Loop Falsifier Runbook (FALSIFY-01)

Phase 16 / D-08 through D-12. This document explains how to choose the gap,
regenerate frozen paraphrases once (and why they stay frozen), run the gate,
and interpret the result.

---

## What the falsifier proves

The loop falsifier is a **mechanism falsifier**: it proves "the adaptive-data loop
can add places that weren't there and make them retrievable."

It does NOT prove "it finds a subtle under-served corner of a populated dataset" —
that is Phase 19 (GAP/METRIC production scope). The empty-sandbox baseline makes the
before/after signal unambiguous.

---

## Chosen gap

**GAP = ("Outer Sunset", "vietnamese")**
**SEED_QUERY = "vietnamese restaurants in Outer Sunset San Francisco"**

### Smoke-test rationale

Candidate buckets evaluated:

| Bucket | Neighborhood | Cuisine | Rationale |
|--------|-------------|---------|-----------|
| **Chosen** | Outer Sunset | vietnamese | Well-established Vietnamese restaurant cluster (Irving Street corridor); Google Places returns a healthy count (~10-20 places) for this search. High confidence of non-zero after-snapshot. |
| Alternate | Bayview | filipino | Reasonable count expected but less-known cluster; chosen as backup if primary fails. |
| Alternate | Excelsior | salvadoran | Smaller footprint; may return too few places for a convincing hit@5 signal. |

The Outer Sunset/vietnamese bucket was selected because it is a well-known, dense
cluster that reliably returns multiple results from a single `searchText` call — giving
the after-snapshot a real chance of hitting top-5 for at least one paraphrase.

**To swap the gap:** change the single `GAP = (...)` constant in `scripts/loop_falsifier.py`
and regenerate the paraphrases (see below). The rest of the orchestrator is gap-agnostic.

---

## Seed isolation and cost (CRITICAL)

Without the orchestrator's seed-isolation pre-mark, `scripts/ingest_places_sf.py`
(line 936) would concatenate the inserted pending proposal with its full 3,410-query
static catalog, capped at `MAX_API_CALLS = 2000`. That would:

1. Issue up to **2,000 paid Google Places API calls** (not 1).
2. Make the DB-diff the entire 2,000-row ingest — after-hit@k would pass by
   retrieving **ANY** newly-embedded place, not one from the chosen gap.
3. **Invalidate FALSIFY-01(c)+(d)**: the metric would be trivially satisfiable.

The orchestrator's `premark_seed_isolation()` function prevents this:
- UPSERTs every static catalog query EXCEPT `SEED_QUERY` as `status='completed'`
  in `places_ingest_query_checkpoints`.
- Inserts `SEED_QUERY` as the only pending proposal.
- Clears any stale pending proposals from prior runs (`status='rejected'`).

The `SKIP_COMPLETED_QUERIES` path in the ingest script then skips all pre-marked
queries, leaving only the gap query (~1 paid Google Places call). The DB-diff is
therefore exactly the gap's new places, and the metric is honest.

**If you see hundreds of new_place_count in the run output, the pre-mark FAILED.**
Stop and investigate — do not accept the gate result.

---

## Frozen paraphrases (D-06)

Paraphrases are **LLM-generated once, then committed** to `configs/falsifier_paraphrases.json`.

The gate reads this file and **never regenerates** — deterministic pass/fail, and the
paraphrases are frozen BEFORE any ingest (non-circularity by construction).

### Why frozen?

- Live regeneration each run would make the gate non-deterministic.
- If paraphrases were generated AFTER ingest, they could be influenced by knowledge
  of what was ingested (circular metric).
- Frozen-before-ingest + exact-string non-circularity check (D-07) ensures honesty.

### To regenerate paraphrases (one-time, after changing the gap)

```python
# Run this script ONCE offline, NOT at gate time
from app.agent.critique.vibe import make_judge

judge = make_judge()
if judge is None:
    raise RuntimeError("No API key for the judge model")

prompt = """Generate exactly 5 distinct rephrasings of the search intent:
'vietnamese restaurants in Outer Sunset San Francisco'
Each rephrasing must express the same intent (finding Vietnamese food in the
Outer Sunset neighborhood of SF) but use different words and phrasing.
Do NOT repeat the original string. Return only the 5 strings as a JSON array."""

from langchain_core.messages import HumanMessage
response = judge.invoke([HumanMessage(content=prompt)])
print(response.content)
# Paste the 5 strings into configs/falsifier_paraphrases.json under "paraphrases"
# Update "seed_query" to match the new SEED_QUERY constant
# Commit the result — it is now frozen
```

**Then commit `configs/falsifier_paraphrases.json` before running the gate.**

---

## Running the gate

### Prerequisites

1. **Provision the sandbox** (if not already done):
   ```bash
   export SANDBOX_DATABASE_URL=postgresql://postgres:cityconcierge@127.0.0.1:5433/city_concierge_sandbox
   make sandbox-provision  # idempotent; see docs/sandbox_db.md
   ```
   Confirm: `SELECT count(*) FROM places_raw` = 0 in city_concierge_sandbox.

2. **Export required env vars**:
   ```bash
   export SANDBOX_DATABASE_URL=postgresql://postgres:cityconcierge@127.0.0.1:5433/city_concierge_sandbox
   export GOOGLE_PLACES_API_KEY=<your-key>
   export OPENAI_API_KEY=<your-key>
   ```

3. **Confirm your prod DATABASE_URL is DIFFERENT from the sandbox** — the guard will exit 2
   if they match (correct behavior; do not bypass).

### Run

```bash
make loop-falsifier
```

Watch the staged output:
1. `[prod-safety] PASS` — prod-safety guard confirmed sandbox != prod.
2. `[non-circularity] PASS` — no paraphrase is byte-identical to the seed.
3. `[seed-isolation]` — pre-marked N catalog queries as completed.
4. `[before-snapshot]` — hit@5 = 0/5 = 0.000 (verified 0.0: sandbox is empty).
   The orchestrator asserts `places_raw` and `place_embeddings_v2` are both empty
   before running any probes. Exit 2 if not.
5. `[ingest]` — real Google Places `searchText` call for the chosen gap (~1 call).
6. `[embed-v2]` — embeddings generated for the new places.
7. `[db-diff]` — small new_place_count (consistent with 1 gap query, NOT thousands).
8. `[after-snapshot]` — hit@5 = M/5 (M >= 1 for a successful gate run).
9. `[gate] delta = +X.XXX` — the strictly-positive delta.
10. `loop_falsifier: VERDICT = PASS` (exit 0) or `FAIL` (exit 1).

---

## Exit codes (D-09)

| Code | Meaning |
|------|---------|
| `0` | PASS — strictly-positive before→after hit@k delta. The loop works. |
| `1` | FAIL — non-positive delta. Expected falsifier outcome; re-scopes v2.3 milestone. Do NOT loosen the gate. |
| `2` | INFRA error — precondition not met: SANDBOX_DATABASE_URL unset/collides with prod, paraphrase file missing/wrong count, non-circularity violation, embed produced zero new rows, subprocess failed, MLflow logging failed, sandbox was not clean (before_hit_rate != 0). |

**A FAIL (exit 1) is a valid scientific result**, not a bug. If the delta is not strictly
positive, the loop did not successfully add retrievable places for the chosen gap. Capture
the before/after snapshots from MLflow and investigate whether:
- The ingest found places but embed-v2 did not embed them.
- The paraphrases were semantically too distant from the ingested places.
- The k=5 retrieval window was too tight.

Do NOT loosen the gate (e.g. by accepting delta=0) — that would make the falsifier meaningless.

---

## Empty-sandbox baseline honesty note

The sandbox starts **empty (zero rows)**. This makes the `before hit@k = 0/N` baseline
trivial — the "gap" is not discovered from real user demand data; it is hardcoded.

**This is intentional for Phase 16**: the goal is to prove the *mechanism* can add places
that weren't there and make them retrievable. The production loop runs against the live
populated dataset (Phases 17–19, where the LOG and GAP phases mine real user queries).

Defensible framing: *"Phase 16 was a mechanism falsifier — empty by design so the
before/after signal is unambiguous; the production loop runs against the live populated
dataset (Phases 17–19)."*

---

## Prod-safety guarantee (D-12)

The falsifier hard-asserts:
1. `SANDBOX_DATABASE_URL` is set and non-empty.
2. The sandbox URL's normalized `(host, dbname)` differs from the prod URL's `(host, dbname)`.
3. For Cloud SQL socket URLs, the instance connection name also differs.

It resolves the prod URL from `{**dotenv_values(".env"), **os.environ}` (with the sandbox
keys popped) — so prod sitting unexported in `.env` is still compared. A prod-safety
violation exits `2` BEFORE any destructive operation (ingest subprocess).

Belt-and-suspenders: after coercing `os.environ["DATABASE_URL"] = sandbox_url`, the
orchestrator asserts the in-process resolved target equals the sandbox — catching a stale
`lru_cache` from a settings import that happened before the injection.

---

## MLflow artifacts (FALSIFY-01e)

Under the `coverage_agent` experiment, each run logs:

| Artifact / Metric | Description |
|-------------------|-------------|
| `frozen_paraphrases.json` | The 5 paraphrases + seed_query from the committed file |
| `before_snapshot.json` | Per-paraphrase top-K lists + hit_rate before ingest |
| `after_snapshot.json` | Per-paraphrase top-K lists + hit_rate after ingest + embed |
| `db_diff_place_ids.json` | place_ids present after ingest but absent before |
| `before_hit_rate` | Metric: 0.0 for empty sandbox (verified by in-process row-count assertion before probing) |
| `after_hit_rate` | Metric: M/N hit rate after ingest |
| `hit_rate_delta` | Metric: after - before (strictly positive -> PASS) |
| `new_place_count` | Metric: number of new places_raw rows |
| `embed_added_count` | Metric: number of new place_embeddings_v2 rows |

A failure to log ANY artifact exits `EXIT_INFRA(2)` — durable evidence is mandatory.

To view results: `make mlflow-tunnel` then open `http://localhost:5050`.
