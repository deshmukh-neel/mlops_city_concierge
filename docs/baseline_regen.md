# Baseline Regen Runbook (BASE-01 / D-11-08)

This runbook documents the ordered procedure for regenerating `configs/eval_baselines/*.json`
under live-infra conditions. It produces the honest n=5 measurements that replace the
fail-open-saturated v2.0 baselines.

**When to run this:** Whenever a Wave-0 harness fix changes measurement semantics, after a
provider adapter ships, or at the start of a new baseline era. The runbook is designed to make
a second pass cheap in case a fix slips after an initial run — but the plan ordering (all
measurement-semantics fixes land in Waves 0/1 before Wave 2) is intended to make that
unnecessary.

> Numbers and gate values live only in `configs/eval_gates.yaml`.
> Do not duplicate them in this document.

---

## Preconditions

All three must be satisfied before proceeding. The 2026-06-05T21:14:30Z disaster was caused
by an exhausted OpenAI embeddings quota that poisoned every cell with fail-open 1.0 scores —
every precondition below is a direct guard against its recurrence.

### 1. OpenAI embeddings sanity probe

Run a single `semantic_search` call and confirm it returns results, **not a 429**.

```bash
# Quick sanity check — expects JSON output with at least one result, not an error:
APP_ENV=eval poetry run python -c "
import asyncio
from app.tools.retrieval import semantic_search
results = asyncio.run(semantic_search('ramen Mission District', k=1))
assert results, 'embeddings probe returned no results — check quota'
print('embeddings OK:', results[0].get('name', '?'))
"
```

A 429 response here means OpenAI embeddings quota is exhausted. **Do not proceed** — every
matrix cell will silently record poisoned 1.0 scores. Top up the quota or rotate the key,
then re-run the probe until it returns results.

### 2. DB reachable

The retrieval path requires a live Postgres/pgvector instance.

**Cloud SQL (production):**
```bash
# Start the proxy for instance mlops-491820:us-central1:mlops--city-concierge
# (note: instance name has DOUBLE-DASH; DB inside uses single-dash mlops-city-concierge)
cloud-sql-proxy --port 5433 mlops-491820:us-central1:mlops--city-concierge &
```

**Or local Docker Postgres:**
```bash
make db-up
```

**Verify DB connectivity:**
```bash
psql "$DATABASE_URL" -c "SELECT 1"
# Expected: " ?column? \n----------\n        1"
```

### 3. All four provider API keys live

Export all four before running any matrix step:

```bash
export OPENAI_API_KEY=...      # OpenAI dashboard — also covers shared embeddings quota
export DEEPSEEK_API_KEY=...    # DeepSeek dashboard
export ANTHROPIC_API_KEY=...   # Anthropic console
export GEMINI_API_KEY=...      # Google AI Studio (gemini logged-not-gated)
```

---

## Steps

Execute in order. Do not skip or reorder — each step depends on the previous one being clean.

### Step 1 — Probe providers (mandatory pre-matrix)

```bash
make probe-providers
```

This is the D-10-14 mandatory pre-matrix step. It runs live probes for all four providers and
writes redacted fixtures to `tests/fixtures/provider_payloads/`. If any gated provider
(openai, anthropic) fails here, investigate before proceeding to the matrix runs. A probe
failure on gemini or deepseek is non-blocking (logged-not-gated) but should be noted.

### Step 2 — Snapshot current baselines (audit trail)

```bash
make snapshot-baselines
```

This creates `configs/eval_baselines/_snapshots/*.pre-phase11.json` copies of the three
canonical baseline JSONs (D-11-09). These snapshots are append-only and preserve the
pre-regen numbers as an auditable historical floor. Verify all three files exist after:

```bash
ls configs/eval_baselines/_snapshots/*.pre-phase11.json
# Expected: omakase_mission_open_ended.pre-phase11.json
#           refinement_cheaper.pre-phase11.json
#           late_night_closure_cascade.pre-phase11.json
```

### Step 3 — Run the omakase matrix at n=5

```bash
APP_ENV=eval make eval-matrix RUNS=5
```

This runs all providers in `configs/eval_matrix.yaml` against `omakase_mission_open_ended`
at n=5. The matrix writes per-cell JSON files and a `summary.json` to
`eval_reports/{timestamp}/`. Note the timestamp for Step 5.

Per-family thinking/temperature policies stay exactly as shipped in `app/llm_factory.py`
— no tuning. The `feedback_temp1_reasoning_off_all_models` policy and its documented
Claude/Gemini/DeepSeek-reasoner carve-outs stand; do not override them for the regen run.

### Step 4 — Run the refinement matrix at n=5

```bash
APP_ENV=eval make eval-matrix-refinement RUNS=5
```

This runs all providers in `configs/eval_matrix_refinement.yaml` against
`refinement_cheaper` at n=5. This replaces the n=1 SHIPPED-WITH-GAP anthropic cell and
records the first-ever gemini n=5 cell (or its deferral if gemini errors).

### Step 5 — Write baselines from both summaries

Run once for each matrix's `summary.json`:

```bash
# Omakase matrix:
make write-baselines SUMMARY=eval_reports/{omakase_ts}/summary.json RUNS=5

# Refinement matrix:
make write-baselines SUMMARY=eval_reports/{refinement_ts}/summary.json RUNS=5
```

Replace `{omakase_ts}` and `{refinement_ts}` with the actual timestamps from Steps 3 and 4.

The tool (D-11-07) reads the summary and writes/updates `configs/eval_baselines/*.json`.
It **refuses** cells with `n_scored < n_requested` (D-10-03) and scenarios with
`baseline_eligible: false`. It stamps `generated_at` / `generated_by` mechanically.

After writing, verify `committed_itinerary_rate` is present in the regenerated scorers blocks
(D-11-02 proof):

```bash
python -c "
import json
for f in ['configs/eval_baselines/omakase_mission_open_ended.json',
          'configs/eval_baselines/refinement_cheaper.json']:
    d = json.load(open(f))
    has_rate = any('committed_itinerary_rate' in c.get('scorers', {})
                   for c in d['providers'].values())
    print(f'{f}: committed_itinerary_rate present = {has_rate}')
"
```

### Step 6 — Run the baselines-mode gate check

```bash
make eval-gates-check-baselines
```

This runs the same code path that CI uses (D-11-15). It reads the committed
`configs/eval_baselines/*.json` and checks them against `configs/eval_gates.yaml`.

Expected result: exit 0 (an `ASPIRATIONAL miss` line for `openai/gpt-5-mini` on stdout is
acceptable and does **not** change the exit code — that gap is a known v2.2 target, not a
regen failure). **ANY non-zero exit is a stop condition:** 1 = hard-gate violation (see
"gpt-4o-mini `committed_itinerary_rate` below 0.8" below), 2 = infra failure (missing or
empty baselines dir, malformed JSON).

### Step 7 — Commit baselines and snapshots together

Commit the regenerated baseline JSONs and their pre-phase11 snapshots in a single focused
commit (separate from any gate-config or code changes per `feedback_small_focused_commits`):

```bash
git add configs/eval_baselines/omakase_mission_open_ended.json
git add configs/eval_baselines/refinement_cheaper.json
git add configs/eval_baselines/_snapshots/omakase_mission_open_ended.pre-phase11.json
git add configs/eval_baselines/_snapshots/refinement_cheaper.pre-phase11.json
git add configs/eval_baselines/_snapshots/late_night_closure_cascade.pre-phase11.json
git commit -m "chore(11-08): regenerate baselines at n=5 with committed_itinerary_rate (D-11-09/10)"
```

---

## Failure branches

### Gemini cells error

`write_baselines.py` refuses cells with `n_scored < n_requested`. If gemini provider cells
error during the matrix run, the writer will refuse to write them and print a REFUSED line to
stderr.

Per D-11-11, gemini is logged-not-gated — errored gemini cells **do not block** BASE-01.
Record the deferral in `tests/unit/test_eval_matrix.py`
`DEFERREDBASELINE_CELLS["eval_matrix_refinement.yaml"]`:

```python
DEFERREDBASELINE_CELLS = {
    "eval_matrix_refinement.yaml": {
        "gemini/gemini-3.1-pro-preview",  # D-11-11: deferred — gemini errored at regen
    },
    ...
}
```

Add a matching comment in `configs/eval_matrix_refinement.yaml` citing D-11-11. Retry once
after verifying the `GEMINI_API_KEY` is live and the embeddings probe still passes. If gemini
continues to error after one retry, proceed without it.

### Gated provider errors (openai)

If `openai/gpt-4o-mini` cells error, **do not proceed**. This is the primary gated
family — an errored cell is not acceptable for the committed baseline. Rerun the affected
matrix until those cells complete cleanly. Check:

1. The embeddings sanity probe still returns results (Step 1 precondition may have expired).
2. The relevant API key is still exported (`echo $OPENAI_API_KEY`).
3. The DB is still reachable (`psql "$DATABASE_URL" -c "SELECT 1"`).

### Anthropic deferral (D-11-20)

`anthropic/claude-sonnet-4-6` was demoted to `logged` (same treatment as gemini) on
2026-06-11 because all 5 omakase cells returned HTTP 400 "credit balance too low". This
is a billing-side blocker — the code and wiring are correct. `write_baselines.py` will
REFUSE the anthropic cells (n_scored=0 < n_requested=5 per D-10-03), which is the
correct documented outcome.

**Do not re-run the matrix on depleted Anthropic credits** — the error is deterministic
and burns no useful compute. Proceed without the anthropic cell; record the deferral in
`DEFERREDBASELINE_CELLS` in `tests/unit/test_eval_matrix.py`.

To promote anthropic back to an active gate when billing is restored, see
`docs/eval_gates.md § Anthropic deferral (2026-06-11)` for the step-by-step promotion
path.

### Gemini deferral (D-12-09)

`gemini/gemini-3.1-pro-preview` n=5 baseline is **deferred as a v2.2 user budget
decision** (D-12-09, 2026-06-11) — no quota or billing top-up; same treatment as the
anthropic ANCH-01 deferral. Gemini stays `logged-not-gated` with its
`DEFERREDBASELINE_CELLS["eval_matrix_refinement.yaml"]` entry intact.

This is **measurement debt, not unknown risk**: the single scored gemini run already hit
`committed_itinerary_rate 1.0` — first evidence the Phase-9 provider adapter fixed the
Gemini loop issue. The comparison floor for v2.2 Phase 13 judging is the matrix minus
BOTH deferred cells (anthropic AND gemini); every other (non-deferred) cell is honest n=5.

**Do not run `write_baselines.py` for gemini cells until quota/billing allows** — the
gemini cells in `eval_matrix_refinement.yaml` will be refused by `write_baselines.py`
when they error (n_scored < n_requested per D-10-03), which is the correct outcome.

**Promotion path:** when quota/budget allows:

1. Verify `GEMINI_API_KEY` is live: run the embeddings probe (Step 1) and confirm it passes.
2. Run `APP_ENV=eval make eval-matrix-refinement RUNS=5` — gemini cells should complete.
3. Run `make write-baselines SUMMARY=eval_reports/{refinement_ts}/summary.json RUNS=5`.
4. Confirm `committed_itinerary_rate` is present in `configs/eval_baselines/refinement_cheaper.json`
   for the `gemini/gemini-3.1-pro-preview` provider.
5. Remove `"gemini/gemini-3.1-pro-preview"` from
   `DEFERREDBASELINE_CELLS["eval_matrix_refinement.yaml"]` in
   `tests/unit/test_eval_matrix.py` — run the parity test to confirm `missing == deferred`
   still holds (it should now equal the empty set for the refinement matrix).
6. Edit `configs/eval_gates.yaml`: set the gemini family entry to `status: active` (or
   `aspirational` if data warrants), add a hard-gate block with the measured floor, add
   a D-ID rationale.
7. Verify `make eval-gates-check-baselines` passes, then commit and open a PR.

See `docs/eval_gates.md § Gemini deferral (2026-06-11)` for gate-semantics context.

### gpt-4o-mini committed_itinerary_rate below 0.8

If `openai/gpt-4o-mini`'s `committed_itinerary_rate` median is below 0.8 on the honest regen,
**STOP. Do not write baselines. Do not commit.**

This is a real anchor regression, not noise. The current gate (D-10-07) is `>= 0.8`. A miss
here means the Phase-9 adapters or a Wave-0 harness fix has broken the primary anchor.
Investigate before committing any regen results.

Steps to diagnose:
1. Run a single `APP_ENV=eval make eval-matrix RUNS=1` to confirm the miss is repeatable.
2. Check `git log --oneline app/agent/ app/llm_factory.py` for any Wave-0 changes that
   might have altered decisiveness behavior (not just measurement semantics).
3. Open a blocker in the phase plan before merging the PR.

---

## What is NOT regenerated

`late_night_closure_cascade` is **not regenerated** (D-10-09/10 standing). Its canonical
baseline JSON (`configs/eval_baselines/late_night_closure_cascade.json`) retains its
`_observations` annotation describing the legacy-threading quarantine. The scenario stays
runnable as a diagnostic:

```bash
APP_ENV=eval make eval-matrix SCENARIOS=late_night_closure_cascade RUNS=5
```

but its results are never written back to the canonical baseline JSON. The pre-phase11
snapshot is taken (Step 2) for audit completeness, but the canonical file is not touched
after the snapshot.

---

## References

- D-11-08 — this runbook decision
- D-11-09 — pre-regen snapshot requirement
- D-11-10 — regen scope (omakase + refinement; late_night excluded)
- D-11-11 — gemini failure branch / deferral policy
- D-10-03 — write_baselines refusal rule (n_scored < n_requested)
- D-10-07 — gate values (gpt-4o-mini >= 0.8 active; gpt-5-mini >= 0.6 aspirational)
- D-10-09 / D-10-10 — late_night quarantine + annotate-not-regenerate
- D-10-14 — probe-providers as mandatory pre-matrix step
- `configs/eval_gates.yaml` — gate values (single source of truth)
- `docs/eval_gates.md` — gate semantics
- `configs/eval_baselines/_snapshots/README.md` — snapshot lifecycle and naming convention
