# Post-Deploy Demand Backfill Runbook

**Purpose:** Close the adaptive-data loop against *real production demand* — detect places
real users ask for that aren't in our DB, and backfill them into prod Cloud SQL.

**Status:** MANUAL, human-gated operational procedure. The full automated "real demand →
auto-backfill prod" capability is a **future phase** (see [Automation: why it's a phase, not a
runbook](#automation-why-its-a-phase-not-a-runbook)), not something to wire up today.

> **Read first:** [docs/loop_runner.md](loop_runner.md) (the sandbox measurement loop) and
> [docs/loop_falsifier.md](loop_falsifier.md). This runbook is the **prod-apply** counterpart —
> the loop *proves* a gap is worth filling in a sandbox; this runbook *applies* a vetted gap to prod.

---

## Why this is a POST-DEPLOY step (and can't be earlier)

The loop runs on **real user demand**, read from the prod `user_query_log` table. That signal
does not exist until real users are hitting `/chat` in production. The dependency chain is strict:

```
deploy → real users ask questions → /chat logs them to prod user_query_log
       → demand accumulates over a meaningful window (days/weeks)
       → gap-mine finds genuine demand×supply gaps → pending proposals
       → HUMAN reviews proposals → prod ingest backfills the missing places
```

Running any of this before real traffic accumulates operates on an empty (or fixture-only)
demand log — there is nothing real to find. This is **not** a deploy-day checklist item; it is a
**recurring operational cadence** you run periodically once the app is live and has traffic.

---

## Prerequisites (verify before running)

1. **App is deployed and serving `/chat`.** Query logging is wired (`app/query_log.py:log_user_query`,
   called from `app/main.py`). Confirm rows are actually landing:
   ```bash
   # Read-only against prod. Replace with the real prod Cloud SQL URL.
   psql "$PROD_DATABASE_URL" -tAc "SELECT COUNT(*), MAX(created_at) FROM user_query_log;"
   ```
   If the count is ~0 or only fixture rows, STOP — there is no real demand yet. Wait for traffic.

2. **A meaningful demand window has accumulated.** A handful of queries is noise. Aim for at least
   a few hundred real `/chat` queries before trusting a gap signal.

3. **⚠️ KNOWN BLOCKER — gap-detector neighborhood-blindness (unfixed).** Per
   [docs/loop_runner.md](loop_runner.md) ("structural finding"), the Google Places
   `searchText` "in {neighborhood}" query does **not** partition by neighborhood — every cuisine
   returns the same ~citywide set in every neighborhood. So **per-(neighborhood, cuisine) gaps are
   not real gaps** and will produce false proposals. Until this is fixed (a future-phase task),
   treat any per-neighborhood proposal with strong skepticism and prefer **citywide-cuisine**
   gaps (a cuisine genuinely near-absent across all of SF). Do NOT auto-apply per-neighborhood
   proposals.

---

## The manual procedure

All steps assume two distinct DB URLs:
- `PROD_DATABASE_URL` — the Cloud SQL prod instance (the real `places_raw` / `user_query_log`).
- A throwaway sandbox URL for the *measurement* step (see loop_runner.md), if you want to prove
  the gap before applying it.

### Step 1 — Mine real prod demand (read prod, write proposals)

`gap-mine` reads demand from `DEMAND_DATABASE_URL` (a **read-only** prod connection) and writes
`pending` proposals. By default it reads the sandbox; set `DEMAND_DATABASE_URL` to read prod demand.

**Dry-run first** (no writes — always start here):
```bash
DEMAND_DATABASE_URL="$PROD_READONLY_DATABASE_URL" make gap-mine-dry
```
Review the printed gaps. Are they plausible? Are they per-neighborhood (suspect — see blocker #3)
or citywide-cuisine (more trustworthy)?

**Then the real mine** (writes `pending` proposals to the *sandbox/proposals* store, NOT prod
`places_raw`):
```bash
DEMAND_DATABASE_URL="$PROD_READONLY_DATABASE_URL" make gap-mine
```

### Step 2 — (Recommended) Prove the gap in the sandbox loop

Before spending a prod ingest, validate the proposed gap actually lifts retrieval using the
sandbox loop ([docs/loop_runner.md](loop_runner.md)). A gap that doesn't pass the loop (no
positive before→after hit@k delta) is not worth ingesting into prod. **Note:** per the known
blocker, per-neighborhood gaps will not pass — this is expected, not a loop failure.

### Step 3 — HUMAN REVIEW the pending proposals (the gate)

There is **no automated approval gate** today — `make ingest-places` consumes *all* `pending`
proposals blindly and flips them to `applied` (`scripts/ingest_places_sf.py:935`). So the human
review IS the gate. Inspect before applying:
```bash
psql "$PROD_DATABASE_URL" -tAc \
  "SELECT query_text, status, rationale FROM places_ingest_query_proposals WHERE status='pending';"
```
**Delete or leave-non-pending any proposal you do NOT want ingested** (e.g. per-neighborhood
false gaps, junk from weird queries). Only `pending` rows get applied.

### Step 4 — Backfill prod (⚠️ REAL PRODUCTION WRITE)

This pulls Google Places data and upserts into prod `places_raw`, consuming the vetted `pending`
proposals. It costs Google Places API calls and OpenAI embedding calls, and mutates prod.

```bash
# DATABASE_URL must point at PROD here. Upsert (ON CONFLICT) — adds/refreshes, does not wipe.
DATABASE_URL="$PROD_DATABASE_URL" make ingest-places
DATABASE_URL="$PROD_DATABASE_URL" make embed-places
```
`ingest-places` prepends pending proposals to the seed list and flips them to `applied` after.

### Step 5 — Verify

```bash
psql "$PROD_DATABASE_URL" -tAc "SELECT COUNT(*) FROM places_raw;"
psql "$PROD_DATABASE_URL" -tAc "SELECT COUNT(*) FROM place_embeddings_v2;"
psql "$PROD_DATABASE_URL" -tAc \
  "SELECT status, COUNT(*) FROM places_ingest_query_proposals GROUP BY status;"
```
Confirm the new places are present, embedded, and the applied proposals flipped to `applied`.

---

## Cadence

This is a **recurring ops task**, not a one-time step. A reasonable rhythm once live:
- Weekly/bi-weekly: `gap-mine-dry` against prod demand to eyeball emerging gaps.
- When a real, citywide gap appears and validates in the loop: review → `ingest-places`.

---

## Automation: why it's a PHASE, not a runbook

Turning this manual cadence into a hands-off "real demand auto-backfills prod" loop is a **future
engineering phase** (candidate v2.4), not a config tweak. The manual chain is fully built, but
automation needs four things that do **not** exist today:

1. **An approval gate.** `ingest-places` consumes *all* `pending` proposals unreviewed and
   auto-flips them to `applied`. Unattended, that ingests junk from any weird query into prod.
   Needs a `pending → approved → applied` status flow with an explicit approval step.
2. **A scheduler.** Only `ci.yml` / `docker.yml` / `terraform-*.yml` workflows exist — no cron.
   Automation needs a scheduled job (GitHub Actions cron, Cloud Scheduler, etc.) to run gap-mine
   and (gated) ingest.
3. **The neighborhood-blindness fix.** The gap detector produces false per-neighborhood gaps
   (see blocker #3 / loop_runner.md). Automating on top of a broken detector backfills garbage.
   Needs a coarser, valid gap construct (citywide-absent cuisine) or a real neighborhood filter.
4. **Unattended prod-write safety.** The loop's prod-safety guards protect the *sandbox* path; the
   prod `ingest-places` path is designed to be human-run. An automated path needs its own guards
   (budget caps, rate limits, rollback, alerting).

Until those exist, this stays **manual and human-gated**. Do not script Step 4 to run unattended.

---

*Created post-Phase-19. The loop machinery (Phases 16–19) is plumbing-verified; this runbook is
the human-operated prod-apply procedure that the loop was designed to feed. See
[.planning/ROADMAP.md](../.planning/ROADMAP.md) v2.3 milestone.*
