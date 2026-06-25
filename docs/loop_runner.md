# Loop Runner Runbook (Phase 19 / LOOP-01..03 + METRIC)

Phase 19 / D-01 through D-08. This document explains the gate's purpose, how to
choose a demand mode, how to run `make loop`, how to interpret the result, and the
known limitation that blocks a calibrated quality floor.

---

## What the loop runner proves

The loop runner is the **productionized loop gate**: it takes a real (or seeded-fixture)
demand signal, mines the top gap, ingests and embeds the gap's places into a populated
sandbox baseline, and measures a before→after hit@k lift.

Unlike the Phase 16 loop falsifier (which starts from an **empty sandbox** with a
hardcoded gap and a trivial before-baseline of 0), the loop runner:

- Uses **real demand** (from `user_query_log`) or a **seeded-sandbox fixture** (see below).
- Runs against a **populated baseline** — new places must compete against an existing corpus.
- Measures a **before→after hit@k delta** against the v2-diff target set (new embedded
  place_ids only — before=0 by construction since those IDs did not exist pre-ingest).

**Operator-run gate (D-06):** `make loop` requires live Google Places + OpenAI keys and
a provisioned sandbox DB. CI does NOT run the live loop — CI unit-tests only the pure
decision logic (floor handling, cold-start, one-gap handoff, stale-proposal rejection,
exit-code mapping) at zero API cost. This mirrors Phase 16's gate-operator-run /
core-unit-tested split exactly.

---

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | PASS — strictly-positive before→after hit@k delta AND after_hit@k >= floor. |
| `1` | FAIL — delta non-positive, or after_hit@k below the calibrated floor. |
| `2` | INFRA error — precondition failure: SANDBOX_DATABASE_URL unset or collides with prod; embedding-table mismatch; ingest/embed subprocess failed; v2-diff empty after ingest; non-circularity violation; gap-handoff mismatch; MLflow logging failed. |

A FAIL (exit 1) is a valid scientific result. If the delta is not strictly positive,
the loop did not successfully add retrievable places for the chosen gap. Capture the
before/after snapshots from MLflow and investigate. Do NOT loosen the gate.

An INFRA error (exit 2) means a precondition failed — investigate and fix before
treating the result as a gate outcome.

---

## Env prerequisites (three-key guard)

`make loop` enforces a three-key guard before running `scripts/loop_runner.py`:

```bash
export SANDBOX_DATABASE_URL=postgresql://postgres:cityconcierge@127.0.0.1:5433/city_concierge_sandbox
export GOOGLE_PLACES_API_KEY=<your-key>        # Google Cloud Console → APIs & Services → Credentials
export OPENAI_API_KEY=<your-key>               # OpenAI dashboard → API keys
```

If any of the three is unset, `make loop` prints `ERROR: <VAR> is not set.` and exits 1
without running the runner.

---

## TWO demand modes

### Mode A — REAL DEMAND (set DEMAND_DATABASE_URL)

```bash
export DEMAND_DATABASE_URL=<prod read-only URL>   # e.g. the Cloud SQL prod instance URL
make loop
```

When `DEMAND_DATABASE_URL` is set, `gap_mine_main` reads demand from the prod
`user_query_log` table directly (read-only). No seed step is needed. This is the
intended steady-state mode once prod `user_query_log` is populated with real user queries.

### Mode B — FIXTURE DEMAND (leave DEMAND_DATABASE_URL unset, run seed_demand_log.py first)

When `DEMAND_DATABASE_URL` is NOT set, the miner reads demand from the **sandbox**
`user_query_log`. An empty sandbox `user_query_log` causes the miner to cold-start
(exit 0, no-op) — it does NOT automatically use a fixture. You MUST seed the sandbox
demand log BEFORE running `make loop`:

```bash
# 1. Seed the sandbox demand log with the hardcoded fixture bucket (Outer Sunset, vietnamese)
DATABASE_URL="$SANDBOX_DATABASE_URL" poetry run python scripts/seed_demand_log.py

# 2. Then run the loop
make loop
```

`scripts/seed_demand_log.py` calls `assert_sandbox_write_target` before any INSERT, so
it is safe — it refuses to write to a non-sandbox database.

**Critical preflight (LOOP_GAP_NAMESPACE contract):** When using the fixture mode, provision
the populated baseline with the MATCHING gap bucket BEFORE running `make loop`:

```bash
export LOOP_GAP_NEIGHBORHOOD="Outer Sunset"
export LOOP_GAP_CUISINE="vietnamese"
make sandbox-provision-populated   # provisions the baseline excluding this bucket
```

If `LOOP_GAP_NEIGHBORHOOD` / `LOOP_GAP_CUISINE` are set and the miner chooses a
different bucket, `loop_runner.py` exits EXIT_INFRA (baseline/miner gap mismatch).

**IMPORTANT — operator cleanup between runs:** A FAILED run leaves the proposal with the
mined gap in `places_ingest_query_proposals` (any status). The miner's `ON CONFLICT DO
NOTHING` dedup treats this as already-covered and will NOT re-mine the same gap on a
subsequent run. Before re-running, delete the stale proposal:

```sql
DELETE FROM places_ingest_query_proposals
WHERE query_text = '<gap seed query>';
```

Or re-provision the sandbox baseline (DROP + re-provision), which truncates all proposal
and ingest tables and is the idempotent clean reset.

---

## Honesty caveats

### D-01 — Fixture vs real demand

When `DEMAND_DATABASE_URL` is unset and `scripts/seed_demand_log.py` was used, the run
is a **FIXTURE** — the demand signal is a hardcoded `(Outer Sunset, vietnamese)` bucket,
not real user queries. The loop output and MLflow run record the `fixture_mode=True`
flag for auditability. Do NOT present a fixture run as evidence of real-demand coverage.

A real-demand run requires a live prod `user_query_log` with enough diverse queries for
the miner to surface a genuine demand×supply gap.

### D-02 — Gap attribution is audited, not guaranteed

`scripts/ingest_places_sf.py` persists every Google Text Search result for the gap seed
query with **no neighborhood/cuisine post-filter**. The statement "these new rows are
from the gap bucket" is an **audited/logged observation** based on the miner's seed
query — it is NOT a guarantee of clean seed isolation. Inspect `new_place_count` and
the MLflow `db_diff_v2_place_ids.json` artifact to verify the diff is plausible for a
single gap query (~5–25 places), not hundreds (which would signal pre-mark failure or
a provisioning error). If `new_place_count` is suspiciously large, investigate before
accepting the gate result.

---

## PARAPHRASE_PROVIDER / PARAPHRASE_MODEL override

The runner generates N paraphrases of the mined gap intent using
`PARAPHRASE_PROVIDER` / `PARAPHRASE_MODEL` env vars (default: the same judge provider
and model used by `make_judge` in `app/agent/critique/vibe`). After the fix in commit
`387f1b3`, Gemini block-content responses are unwrapped correctly and the default
Gemini provider works. OpenAI `gpt-4o-mini` is also confirmed working:

```bash
export PARAPHRASE_PROVIDER=openai
export PARAPHRASE_MODEL=gpt-4o-mini
make loop
```

---

## Running the gate

### Full example (fixture mode)

```bash
# 1. Export required env vars
export SANDBOX_DATABASE_URL=postgresql://postgres:cityconcierge@127.0.0.1:5433/city_concierge_sandbox
export GOOGLE_PLACES_API_KEY=<your-key>
export OPENAI_API_KEY=<your-key>

# 2. Provision the populated baseline (excluding the fixture gap bucket)
export LOOP_GAP_NEIGHBORHOOD="Outer Sunset"
export LOOP_GAP_CUISINE="vietnamese"
make sandbox-provision-populated

# 3. Seed the sandbox demand log (FIXTURE — not real demand, D-01)
DATABASE_URL="$SANDBOX_DATABASE_URL" poetry run python scripts/seed_demand_log.py

# 4. Run the operator gate
make loop
```

Watch the staged output:
1. `[prod-safety] PASS` — sandbox != prod confirmed.
2. `[resolved-target]` — in-process DATABASE_URL confirmed to point at sandbox.
3. `[embedding-table] confirmed: 'place_embeddings_v2'` — scorer uses the right view.
4. `[gap-handoff]` — stale pending cleared, gap-mine run, one new proposal found.
5. `[paraphrase-gen]` — N paraphrases generated and frozen to disk.
6. `[non-circularity] PASS` — none of the paraphrases is byte-identical to the seed.
7. `[before-snapshot]` — probe N paraphrases with k=5 BEFORE ingest.
8. `[ingest]` — real Google Places `searchText` call for the chosen gap.
9. `[embed-v2]` — embeddings generated for the new places.
10. `[db-diff]` — new_place_count and embed_added_count printed.
11. `[after-snapshot]` — probe N paraphrases with k=5 AFTER ingest.
12. `[gate]` — delta and floor printed; VERDICT = PASS (exit 0) or FAIL (exit 1).

---

## Floor calibration — DEFERRED (known limitation)

### Current state: FLOOR = 0.0 (uncalibrated, strict-positive-delta only)

The committed `FLOOR` in `app/loop/falsifier_core.py` is `0.0`. This means the gate
passes on ANY strictly-positive before→after hit@k delta. No quality bar (e.g. >= 0.4
after_hit@k) is enforced beyond delta > 0.

### Why calibration is BLOCKED (corpus structural finding)

The operator attempted a calibration run on 2026-06-20 using three fixture gap buckets:
- `(Outer Sunset, vietnamese)`
- `(Inner Sunset, nepalese)`
- `(Outer Richmond, korean)`

A structural finding blocked positive-lift measurement: **a per-(neighborhood, cuisine)
supply gap is NOT zeroable in the SF corpus with the current provisioning approach.**

The populated baseline provisioning excludes the gap bucket's seed query, but gap
places still leak into the before-snapshot via ~20 OTHER neighborhoods' cuisine queries.
For example, `vietnamese/Outer Sunset` had ~20 pair-supply places, but only ~1 was
UNSHARED — the other 19 already appeared in results from `vietnamese restaurants in
{Inner Sunset, Cow Hollow, Chinatown, Marina, SOMA, ...}`. SF is too geographically
small for the Places API `"in {neighborhood}"` filter to partition restaurants by
neighborhood.

This means `after_hit@k` cannot lift above `before_hit@k` for these buckets, so the
gate fails by construction even on a successful ingest. Changing the exclusion rules in
`provision_sandbox.sh` would NOT fix this — it would require excluding the cuisine
citywide, which defeats the purpose of a neighborhood-specific gap.

### What WAS verified (plumbing-verified status)

Despite the calibration deferral, two real bugs were found and fixed during the
operator run:

- **commit `dbf9b1a`** — fixed metric target-set asymmetry: `before_hit@k` was scored
  against `before_v2_ids` (all pre-existing IDs, inflating before to ~1.0 → delta ~-1.0)
  instead of `new_v2_ids` (the v2-diff target, which correctly gives before=0 by
  construction per D-03). The fix makes `before_hit@k = 0.0` by construction as designed.

- **commit `387f1b3`** — fixed Gemini block-content crash: the Gemini model returns
  paraphrase content as a list of typed blocks `[{'type': 'text', 'text': '<json>'}]`
  instead of a plain string; the runner now unwraps the text field from block-type
  content before JSON parsing.

The full gap-mine→ingest→embed→score→gate pipeline runs end-to-end and these bugs are
confirmed fixed. The gate is plumbing-verified.

**The PRE-FIX MLflow run `42fdf99657714f1ca17e849ddf0ce787`** (`loop-runner-Outer Sunset-vietnamese`)
recorded `delta=-1.000`. This run documents the metric target-set bug, NOT a valid
calibration outcome. Do not interpret this run as evidence that the loop failed to
ingest retrievable places.

### What a real calibration requires

A positive-lift calibration needs a **cuisine that is genuinely absent or near-absent
citywide** in the SF corpus — not just absent in one neighborhood while present in 19
others. Candidate constructs:

- A very niche cuisine with few SF Places API results across all neighborhoods.
- A synthetic gap constructed by deliberately excluding an entire cuisine from the
  provisioned baseline.

This is explicitly deferred. When a suitable gap construct is identified:

1. Provision the baseline excluding that cuisine citywide.
2. Run `make loop` with `fixture_mode=True` (seed the matching demand).
3. Record `after_hit@k` from the passing MLflow run.
4. Set `FLOOR` in `app/loop/falsifier_core.py` to the highest defensible value
   at-or-below the observed `after_hit@k` (D-05 — do NOT set above observed).
5. Update this section with the justifying run (MLflow run id, gap bucket, observed
   after_hit@k, chosen FLOOR, date).

---

## MLflow artifacts

Under the `coverage_agent` experiment, each run logs:

| Artifact / Metric | Description |
|-------------------|-------------|
| `frozen_paraphrases_runner.json` | N paraphrases + seed_query + generation_prompt + model + timestamp |
| `before_snapshot.json` | Per-paraphrase top-K lists + hit_rate before ingest |
| `after_snapshot.json` | Per-paraphrase top-K lists + hit_rate + recall after ingest |
| `db_diff_v2_place_ids.json` | place_ids in place_embeddings_v2 after ingest but absent before |
| `before_hit_at_k` | Metric: 0.0 by construction (new v2 IDs did not exist before ingest) |
| `after_hit_at_k` | Metric: hit@k rate after ingest |
| `hit_rate_delta` | Metric: after - before (strictly positive → PASS on floor=0.0) |
| `recall_at_k` | Metric: fraction of new embedded places found across paraphrases' top-k |
| `new_place_count` | Metric: number of new places_raw rows |
| `embed_added_count` | Metric: number of new place_embeddings_v2 rows |
| `floor` | Param: the floor value in effect for this run |
| `fixture_mode` | Param: True = fixture demand (seed_demand_log.py used), False = real demand |

To view results: `make mlflow-tunnel` then open `http://localhost:5050`.
