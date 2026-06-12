---
phase: 13-decisiveness-experiment-arms
plan: "08"
subsystem: agent/viability + agent/graph + docs
tags: [cr-01, forced-commit, placehit, synthesizer, tdd, docs]
dependency_graph:
  requires: []
  provides: [DEC-02 operative forced-commit synthesizer, CR-01 regression tests, A2 honest verdict annotation]
  affects: [app/agent/viability.py, app/agent/graph.py, tests/unit/test_viability.py, tests/unit/test_graph_forced_commit.py, docs/decisiveness_arm_verdicts.md]
tech_stack:
  added: []
  patterns: [TDD red/green, model_dump(mode='json') for Pydantic→dict conversion]
key_files:
  created: [tests/unit/test_graph_forced_commit.py (extended)]
  modified:
    - app/agent/viability.py
    - app/agent/graph.py
    - tests/unit/test_viability.py
    - tests/unit/test_graph_forced_commit.py
    - docs/decisiveness_arm_verdicts.md
decisions:
  - "CR-01 fix (a): viability.py typed path uses BaseModel.model_dump(mode='json') for PlaceHit — eliminates empty-dict {} placeholder that discarded all Pydantic model hits"
  - "CR-01 fix (b): graph.py synthesizer builds explicit commit-shaped dicts per candidate with synthesized rationale string — satisfies Stop.rationale required field"
  - "IN-04 removed: step_count no-op from model_copy update dict in forced-commit branch"
  - "A2 Phase-14 retry disposition: reserved as Phase-14/15 candidate; D-13-02 four-run live cap consumed in Phase 13; honest null result (0.500 model-initiated) stands"
metrics:
  duration: "4m"
  completed: "2026-06-12"
  tasks: 3
  files: 5
---

# Phase 13 Plan 08: CR-01 Forced-Commit Synthesizer Fix Summary

**One-liner:** Fix two independent CR-01 defects in the A2 forced-commit synthesizer (PlaceHit→dict conversion + missing rationale field) and add a non-mocked regression test that fails on the pre-fix code; annotate the A2 verdict with the synthesis bug and Phase-14 retry disposition.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 RED | PlaceHit typed-path regression tests (failing) | eb4a377 | tests/unit/test_viability.py |
| 1 GREEN | Fix typed-path PlaceHit→dict via model_dump | c8ef28d | app/agent/viability.py |
| 2 RED | Non-mocked synthesizer regression test (failing) | df405b8 | tests/unit/test_graph_forced_commit.py |
| 2 GREEN | Fix synthesizer: build commit-shaped stops with rationale, drop IN-04 | 56e12e9 | app/agent/graph.py |
| 3 | Annotate A2 verdict with CR-01 finding | 41f4fc2 | docs/decisiveness_arm_verdicts.md |

## What Was Built

### Task 1: viability.py typed-path fix (CR-01a)

The `best_viable_candidate_per_slot` typed path had `hit_dict = hit if isinstance(hit, dict) else {}` (line 216). In production, `semantic_search` stores `PlaceHit` Pydantic models directly in scratch — not dicts — so every typed candidate was silently converted to `{}`. The empty dict had no `place_id`, so candidates were filtered out before reaching `commit_stops`.

**Fix:** Three-branch conversion on both typed and untyped paths:
- `isinstance(hit, dict)` → `dict(hit)`
- `isinstance(hit, BaseModel)` → `hit.model_dump(mode='json')` (JSON-safe, preserves all fields)
- else → `continue` (skip unusable shapes — no `{}` placeholder)

Also fixed the untyped path's legacy `{k: getattr(hit, k, None) for k in dir(hit) if not k.startswith("_")}` conversion (captured bound methods, not JSON-safe) to use the same `model_dump(mode='json')` branch.

**Tests added (4 new tests in test_viability.py):**
- `test_best_viable_candidate_per_slot_placehit_typed_path_returns_populated_dict` — asserts populated entry with correct fields for real PlaceHit; FAILS on pre-fix code
- `test_best_viable_candidate_per_slot_dict_path_unchanged` — byte-compatibility for dict hits
- `test_best_viable_candidate_per_slot_unknown_shape_is_skipped` — unknown shapes yield None (not {})
- `test_best_viable_candidate_per_slot_placehit_place_id_propagated` — regression guard: place_id from PlaceHit == returned dict's place_id; FAILS on pre-fix code

### Task 2: graph.py synthesizer fix (CR-01b) + non-mocked test

The forced-commit branch passed raw candidate dicts straight to `commit_stops`. Even after Task 1, those dicts lacked `rationale` — and `Stop.rationale` is REQUIRED with no default — so `Stop(**raw)` raised `ValidationError` and `commit_stops` rejected every synthesized stop.

**Fix:** Replaced `raw_stops = [c for c in candidates if c is not None]` with explicit commit-shaped dict construction per candidate:
- `place_id`, `name`, `primary_type`, `source` from the candidate dict
- Synthesized `rationale` string: `"Best available match for requested {ptype} slot (forced commit at step N; cosine similarity X.XXX)."`
- Skips candidates lacking a truthy `place_id` (WR-07 admission consistency)

Also removed the no-op `"step_count": state.step_count` from the `model_copy` update dict (IN-04).

**Test added (non-mocked regression test):**
`test_forced_commit_synthesizer_real_placehit_shapes` in `test_graph_forced_commit.py`:
- Scratch contains real `PlaceHit` objects (not dicts)
- `best_viable_candidate_per_slot` and `commit_stops` are NOT patched
- Only DB boundary (`get_details_many`), `critique_final_with_stops`, `route_legs`, `swap_closed_stops` are mocked
- Asserts `commit_forced=True` and `stops` non-empty
- FAILS on pre-fix synthesizer (graph loops to recursion limit)

### Task 3: A2 verdict annotation

Added CR-01 annotation block to `docs/decisiveness_arm_verdicts.md` immediately after the "Key finding — FORCED_COMMIT_STEP=6 mechanism NEVER FIRED" paragraph. The annotation states:
- (a) The mechanism was INOPERATIVE due to CR-01 synthesis bug (both defects described)
- (b) forced=0 is over-determined by the bug — not solely gate non-satisfaction
- (c) The 0.500 gpt-5-mini result STANDS as entirely model-initiated
- (d) The forced mechanism is UNTESTED at n=5
- (e) Phase-14 A2 retry is a Phase-14/15 candidate (not Phase-13 re-run; D-13-02 cap consumed)

Updated closing verdict sentence from "the mechanism's viability gate was not satisfied" to "forced=0 is over-determined by the CR-01 synthesis bug; whether the gate would have been satisfied on the fixed synthesizer is unknown."

The honest null result ("No arm cleared the INST-05 falsifier bar") is preserved and unaltered.

## Deviations from Plan

None — plan executed exactly as written.

## Threat Flags

None — no new network endpoints, auth paths, or trust boundaries introduced. The synthesizer routes through existing `commit_stops` grounding check (T-13-08-01 already accepted; place_id grounding unchanged).

## Known Stubs

None.

## Verification Results

- `poetry run pytest tests/unit/test_viability.py tests/unit/test_graph_forced_commit.py -q` — 34 passed
- `poetry run pytest tests/unit/test_critique_checks.py::test_prompt_02_grep_gate_no_behavioral_phrases_in_prompts -q` — 1 passed (Phase-7 grep gate green)
- `make test` — 1350 passed, 53 skipped, 17 warnings (full suite clean)
- `grep -n "model_dump" app/agent/viability.py` — 2 matches (typed and untyped paths)
- `grep -n "else {}" app/agent/viability.py` — 0 matches (pre-fix line gone)
- `grep -n "rationale" app/agent/graph.py` — synthesized rationale string present in forced-commit branch
- A2 verdict annotation: all four required phrases present ("synthesis bug", "untested at n=5", "model-initiated", "No arm cleared")

## Self-Check: PASSED

Files verified:
- app/agent/viability.py — exists, contains model_dump on both paths, no `else {}`
- app/agent/graph.py — exists, contains rationale synthesis, no IN-04 no-op
- tests/unit/test_viability.py — 4 new tests added; all pass
- tests/unit/test_graph_forced_commit.py — 1 new non-mocked test added; passes
- docs/decisiveness_arm_verdicts.md — annotation block present with all required phrases

Commits verified: eb4a377, c8ef28d, df405b8, 56e12e9, 41f4fc2 — all present in git log.
