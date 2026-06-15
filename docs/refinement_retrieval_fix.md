# Refinement Retrieval Fix — Root Cause + Resolution (2026-06-15)

## TL;DR

The long-standing `refinement_cheaper = 0.000 committed_itinerary_rate` was **two
stacked retrieval/tool-contract bugs**, NOT a data gap and NOT (for the anchor) a
model-capability gap. Fixing both took the production anchor from **0.0 → 0.8** on
refinement at n=5, with no anchor regression on omakase. The reasoning model
(gpt-5-mini) stayed 0/5 — isolating its failure as pure decisiveness (ARCH-FUT-01
territory), not retrieval.

## Investigation (evidence-first)

Scenario: "date night dinner-then-drinks-then-dessert in Hayes Valley, 3 stops"
→ requested slots: Restaurant + Cocktail Bar + Dessert Shop.

1. **Data is NOT the gap.** Near Hayes Valley: 19+ Restaurants, 4 Cocktail Bars,
   4 Wine Bars, Bakeries/Ice Cream Shops. Citywide: 77 Cocktail Bars, 24 "Dessert
   Shop" primary_types, 215 places with `dessert_shop` in types[]. Candidates exist.

2. **pgvector / retrieval mechanics are NOT the gap.** A well-formed query with the
   family filter returns 5–10 viable candidates per slot (top cosine 0.66). The
   retrieval layer works when handed good inputs.

3. **Bug A — the family filter never bound.** Live trace: the agent issues
   `semantic_search` with `family=None` on every call. `_inject_primary_type_family`
   only fired when the model emitted `slot_index`, which gpt-4o-mini and the
   reasoning models routinely DON'T. Thin/unfiltered queries ("drinks in Hayes
   Valley") top out at ~0.52–0.55 → 0 viable per slot → gate never satisfied.

4. **Bug B — viability matched primary_type EXACTLY.** Even with the family filter
   retrieving real dessert venues, `all_slots_viable` required the hit's
   `primary_type` to equal the requested string literally. Dessert venues are typed
   "Bakery"/"Ice Cream Shop" — **zero** literal "Dessert Shop" near Hayes Valley —
   so the dessert slot was permanently unsatisfiable. Restaurant similarly (most are
   "Italian Restaurant"/"Pizza Restaurant", not bare "Restaurant").

## Fixes

- **Fix #1 (`427864a`)** `family_from_query()` in `app/tools/filters.py`: when the
  model omits `slot_index`, infer the slot family from the query text — but ONLY a
  family the user actually requested. Wired into `_inject_primary_type_family`
  (`app/agent/graph.py`) as the fallback after the existing slot_index path.

- **Fix #2 (`16fa3cc`)** `requested_type_for_hit()` in `app/agent/viability.py`:
  match a hit to a requested slot by FAMILY, not exact primary_type (Bakery satisfies
  a Dessert Shop slot). Applied to `all_slots_viable` + `best_viable_candidate_per_slot`
  and the D-13-03 twin sites in `scripts/eval_agent.py` (single source of truth kept).

## Measured result (real eval harness, n=5, temp=1.0)

| Model | Scenario | Baseline | Post-fix |
|-------|----------|----------|----------|
| gpt-4o-mini | refinement_cheaper | 0.0 | **0.8 (4/5)** |
| gpt-4o-mini | omakase | 1.0 | 0.8 (4/5)* |
| gpt-5-mini | refinement_cheaper | 0.0 | 0.0 (0/5) |
| gpt-5-mini | omakase | 1.0 | 0.8 (4/5)* |

*one stochastic miss at n=5; median 1.0, within the documented "0.8 absorbs one miss" band.

## What remains (decisiveness, not retrieval)

gpt-5-mini retrieves fine and the viability gate is satisfiable, but it writes a
3-stop itinerary in prose and won't call `commit_itinerary`. The retrieval fixes
that lifted the anchor did NOT move gpt-5-mini — cleanly isolating its gap as
decisiveness (the v2.2 honest-null finding; ARCH-FUT-01 trigger). The forced-commit
path is configured for step 6 but the agent terminates ~step 4, so it never fires.

## Not done here (deliberate stop)

- No prod-default flip (e.g. forced-commit-on-viable) — that changes anchor behavior
  and is the deferred D-15-07 decision.
- Baselines NOT regenerated. The committed `refinement_cheaper` baseline (0.0) is now
  STALE for gpt-4o-mini — a future baseline regen against this fixed code should
  record the ~0.8 honest flag-off rate. (Left for a deliberate regen pass.)
