---
phase: 13-decisiveness-experiment-arms
verified: 2026-06-12T12:00:00Z
status: gaps_found
score: 4/5 must-haves verified
overrides_applied: 0
gaps:
  - truth: "The parallel-tool-execution arm (DEC-04) runs all tool calls within one act() plan step concurrently with results order-stable — measurable gpt-4o-mini latency reduction at n=5 recorded in run JSON"
    status: partial
    reason: "Concurrency and order-stability are implemented and verified by tests. INST-04 tool_exec_seconds ARE recorded in run JSON (absolute values confirmed in A3 run files). However the 'measurable latency reduction' half of the criterion cannot be computed: the Phase-12 comparison-floor run dirs predate the INST-04 step_telemetry instrumentation (all Phase-12 runs have tool_exec_seconds=None), so no valid baseline exists for reduction measurement. The A3 arm verdict honestly documents this as UNMEASURABLE but that means the success criterion's reduction deliverable is unachieved."
    artifacts:
      - path: "docs/decisiveness_arm_verdicts.md"
        issue: "A3 latency table shows 'N/A — no telemetry in Phase-12 runs' for baseline; delta column is 'unmeasurable' across all cells. No latency reduction is recorded, only absolute tool_exec_seconds from the arm run."
    missing:
      - "A flag-off (sequential, same-phase) baseline run with INST-04 step_telemetry to provide a valid comparison point for the A3 arm's latency reduction claim"
      - "Alternatively, annotate the roadmap success criterion as modified by the discovered constraint (Phase-12 runs lack step_telemetry) so the criterion is respecified as 'absolute latency recorded for future baseline use'"

critical_findings:
  - id: CR-01
    severity: blocker_for_future_phases
    component: "app/agent/viability.py:216, app/agent/graph.py:619-639"
    description: "The A2 forced-commit synthesizer can never produce a valid commit in production. Two independent defects: (a) the typed path converts every non-dict PlaceHit Pydantic model to an empty dict ('hit_dict = hit if isinstance(hit, dict) else {}' at viability.py:216), and (b) even with a correct hit dict, commit_stops requires a 'rationale' field with no default (state.py:198) which PlaceHit lacks. Net effect: raw_stops is [{},...] -> commit_stops rejects all -> committed_stops is empty -> forced branch silently falls through with commit_forced=False. The live A2 run's forced=0 is consistent with this bug. The A2 verdict's causal claim ('gate conditions were not satisfied') is unverifiable. The unit test (test_forced_commit_triggers_at_step_n) masks this by mocking best_viable_candidate_per_slot AND commit_stops with pre-formed correct shapes."
    verdict_impact: "Null result (no arm cleared INST-05 bar) STANDS regardless of CR-01 because forced mechanism never contributed positively. BUT the A2 causal explanation is wrong/unverifiable, and any Phase 14 decision about whether to retry A2 with a working synthesizer rests on bad evidence."
    required_action_before_phase_14: "Fix CR-01 per REVIEW recommendations before any A2 re-run. Annotate the A2 section of docs/decisiveness_arm_verdicts.md with: 'mechanism was inoperative due to synthesis bug; 0.500 model-initiated improvement stands; forced mechanism is untested at n=5.'"

  - id: CR-02
    severity: blocker_for_future_phases
    component: "scripts/eval_falsifier.py:205, tests/unit/test_eval_falsifier.py:951"
    description: "_commit_split_from_run_dir reads data.get('deterministic') at top level of per-run JSON files, but the real EvalRunReport shape nests deterministic under queries[i]. Confirmed by reading an actual A2 run file: top-level keys are ['aggregate', 'chat_model', 'eval_queries_path', 'llm_provider', 'queries', 'query_count'] — no top-level 'deterministic'. Running the buggy code against real A2 gpt-5-mini runs returns (0, 0); correct logic (iterating queries[i]) returns (4, 0). The committed verdict doc carries the correct (4/10) hand-computed split in tables but the pasted falsifier output shows '(model-initiated 0/0, forced 0/0)' — inconsistent numbers within the same document. The unit tests pass because the fixture writer encodes the bug (writes deterministic at top level, matching what the code reads)."
    verdict_impact: "The D-13-04 honesty contract ('the verdict MUST report the model-initiated vs forced split') is fulfilled by the hand-computed tables in the doc, but the tool that is supposed to enforce this contract (eval_falsifier.py) is silently broken. Any future operator running eval-falsifier-arm will get 0/0 in all split annotations."
    required_action_before_phase_14: "Fix CR-02 per REVIEW recommendations. Fix test fixture to write real EvalRunReport shape. The existing verdict doc tables are numerically correct but must note that the falsifier output '0/0' is a tool bug, not a real measurement."
---

# Phase 13: Decisiveness Experiment Arms — Verification Report

**Phase Goal:** Four coupled experiment arms are implemented, run at n=5 temp=1.0 against the Phase-12 comparison floor, and their verdicts are documented — revealing whether any arm clears the falsifier bar or all plateau below it.
**Verified:** 2026-06-12T12:00:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | DEC-01 viability-contract arm ships without touching Phase-7 CI grep gate sections | VERIFIED | `test_prompt_02_grep_gate_no_behavioral_phrases_in_prompts` PASSES; `rule8_viability_addendum` appended AFTER SYSTEM_PROMPT, not inside; none of the six D-07-04 forbidden phrases appear in prompts.py |
| 2 | DEC-02 forced-commit is a graph-level mechanism confirmed by a unit test on a mock that never calls commit_itinerary | VERIFIED (with caveat) | `test_forced_commit_triggers_at_step_n` PASSES — mock LLM emitting only semantic_search triggers commit_forced=True. Caveat: test mocks `all_slots_viable`, `best_viable_candidate_per_slot`, AND `commit_stops`, so it cannot observe CR-01 (synthesizer silently fails on real PlaceHit shapes in production) |
| 3 | DEC-04 parallel tool execution runs concurrently with order-stable results; measurable latency reduction at n=5 recorded in run JSON | PARTIAL FAIL | Concurrency confirmed: `asyncio.gather` in graph.py, order-stability test passes. INST-04 `tool_exec_seconds` IS in A3 run JSON. BUT "measurable reduction" is UNMEASURABLE: Phase-12 comparison-floor run dirs have `step_telemetry: None` (field added in Phase 13 plan 13-01). No baseline exists for delta computation. Verdict doc honest about this. |
| 4 | DEC-03 critique-recalibration is co-tuned with DEC-01; threshold direction and low_similarity scoping decision documented before threshold code lands | VERIFIED | `docs/decisiveness_dec03_decision.md` exists (2026-06-12) with Decision 1 (threshold direction: lower below 0.55 via env override) and Decision 2 (scope low_similarity to pre-candidate steps). Commit `4d3c0ec` lands the doc; `18b918f` lands the code — doc-before-code ordering confirmed in git log |
| 5 | DEC-05 arm-verdict document records per-arm n=5 numbers for gpt-5-mini, deepseek-reasoner, gpt-4o-mini anchor and explicitly states which arm cleared INST-05 bar or records honest null result | VERIFIED (with caveat) | `docs/decisiveness_arm_verdicts.md` exists with per-arm tables for all three models. Per-arm correct split numbers confirmed against actual run JSON data (A1: mi=1/10, A2: mi=4/10, A3: mi=5/10). Closing line: "No arm cleared the INST-05 falsifier bar. All arms plateaued below gpt-5-mini >= 0.6." Caveat: pasted falsifier output shows "0/0" throughout (CR-02 tool bug); correct numbers exist in tables but inconsistency within doc is a documentation integrity issue |

**Score:** 4/5 truths verified (SC-3 is a partial fail; SC-2 and SC-5 verified with named caveats)

---

## Critical Findings (from Code Review CR-01 and CR-02)

These were identified in the co-submitted `13-REVIEW.md` and independently confirmed during verification.

### CR-01: A2 Forced-Commit Synthesizer Dead Code in Production

**File:** `app/agent/viability.py:216`, `app/agent/graph.py:619-639`

Confirmed by reading the code: `hit_dict = hit if isinstance(hit, dict) else {}` at `viability.py:216`. Real `semantic_search` hits are `PlaceHit` Pydantic models stored verbatim in scratch. This line converts every real hit to `{}`. Additionally, `commit_stops` requires a `rationale` field (`state.py:198`) which `PlaceHit` lacks, so even a correct hit dict would be rejected.

The live A2 run (30 episodes, forced=0) is explained by this bug. The verdict doc's causal claim ("gate conditions were not satisfied") is unverifiable from the code as shipped.

**Why the test does not catch this:** `test_forced_commit_triggers_at_step_n` at lines 242-272 patches `best_viable_candidate_per_slot` to return a pre-formed dict with all required fields including `rationale` — bypassing both defects. The test proves graph-level wiring fires but not that the synthesizer works end-to-end.

**Verdict impact:** The null result (no arm cleared INST-05) STANDS. But the reason A2's forced mechanism produced `forced=0` is CR-01, not "gate not satisfied", and this distinction matters for whether to retry A2 in Phase 14.

### CR-02: Falsifier Split Reader Always Returns 0/0

**File:** `scripts/eval_falsifier.py:205`

Confirmed by reading both the code and an actual run file:
- Code reads: `det = data.get("deterministic") or {}`
- Real run file top-level keys: `['aggregate', 'chat_model', 'eval_queries_path', 'llm_provider', 'queries', 'query_count']` — no top-level `deterministic`.
- `deterministic` is at `data["queries"][i]["deterministic"]`.

Simulated both buggy and correct logic against A2 gpt-5-mini run files:
- Buggy (as coded): `model_initiated=0, forced=0`
- Correct (iterating queries[i]): `model_initiated=4, forced=0`

The verdict doc carries correct split numbers in tables (hand-computed, confirmed accurate) but pasted falsifier output shows `(model-initiated 0/0, forced 0/0)` — creating an internal inconsistency in the doc.

The unit test fixture at `test_eval_falsifier.py:951` writes `{"deterministic": {...}}` at top level, matching what the buggy code reads. The tests pass because the fixture encodes the bug.

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `app/agent/viability.py` | Shared viability predicate (all_slots_viable, best_viable_candidate_per_slot) | VERIFIED | File exists, exports both functions, 240+ lines. Typed path bug (CR-01) exists at line 216 but predicate logic otherwise correct |
| `app/agent/state.py` | commit_forced + forced_commit_step fields | VERIFIED | Lines 297-312 confirm both fields with correct defaults (False/None) |
| `scripts/eval_agent.py` | arm_flags + commit_forced/forced_commit_step in DeterministicEvalResult | VERIFIED | Lines 149-152 confirm all three fields; make_error_record has safe defaults at lines 256-258; arm_flags assembled at lines 928-937 |
| `app/agent/graph.py` | Arm flag reads at build time + A2 forced-commit branch + A3 parallel act() + A1 prompt wiring | VERIFIED | FORCED_COMMIT_STEP, VIABILITY_CONTRACT_ENABLED, PARALLEL_TOOL_EXECUTION_ENABLED all present; all_slots_viable imported and used in critique(); asyncio.gather present for A3 |
| `app/agent/prompts.py` | rule8_viability_addendum (additive, flag-gated) | VERIFIED | Lines 19-47 implement addendum; flag-off returns "" (byte-identical); Phase-7 grep gate passes |
| `app/agent/revision.py` | LOW_SIMILARITY_THRESHOLD env-overridable + low_similarity scoping | VERIFIED | Lines 27-36 confirm env-override; lines 207-216 confirm DEC-03 flag-gated scoping |
| `configs/eval_matrix_arm.yaml` | 3-provider x 2-scenario arm matrix config | VERIFIED | 3 entries (gpt-4o-mini, gpt-5-mini, deepseek-reasoner); 2 scenarios (omakase_mission_open_ended, refinement_cheaper); anthropic/gemini excluded |
| `scripts/eval_falsifier.py` | --matrix-config flag + split reader | VERIFIED (CR-02 bug present) | --matrix-config flag at line 265; _commit_split_from_run_dir at line 179; split reader reads wrong JSON shape (confirmed) |
| `Makefile` | eval-matrix-arm + eval-falsifier-arm targets | VERIFIED | Lines 226-248 confirm both targets with correct flag doc and POETRY_RUN prefix |
| `docs/decisiveness_arm_verdicts.md` | Per-arm verdict sections with DEC-05 required field set | VERIFIED (with CR-02 caveat) | 395-line document with A1/A2/A3/A4 sections and Closing Verdict; run dirs recorded; per-model tables complete; correct split numbers in tables (hand-computed) but falsifier print shows 0/0 |
| `docs/decisiveness_dec03_decision.md` | DEC-03 decision before threshold-touching code | VERIFIED | Commit 4d3c0ec (doc) precedes 18b918f (code) in git log; both decisions documented |
| `tests/unit/test_graph_forced_commit.py` | D-13-04 required test: mock never commits → forced fires | VERIFIED (mocked) | test_forced_commit_triggers_at_step_n passes; 6 total tests pass |
| `tests/unit/test_graph_parallel_tools.py` | Order-stability + INST-04 timing on parallel path | VERIFIED | 5 tests pass including order-stability and tool_exec_seconds recording |
| `.planning/REQUIREMENTS.md` | DEC-01..05 traceability flipped to Complete | VERIFIED | All 5 DEC checkboxes [x]; traceability table shows Phase 13 Complete for DEC-01..05 |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `app/agent/graph.py critique()` | `app.agent.viability.all_slots_viable` | forced-commit gate | WIRED | Import at line 46; usage at line 617 of graph.py |
| `app/agent/graph.py forced-commit branch` | `app.agent.commit.commit_stops` | synthetic commit routed through normal path | WIRED (CR-01 breaks production path) | Code path exists; commit_stops called at line 625; but synthesizer produces empty/rejected stops |
| `app/agent/graph.py plan()` | `app.agent.prompts.rule8_viability_addendum` | flag-gated addendum appended at build time | WIRED | `_viability_prompt_addendum` set at line 312; appended at line 324 |
| `scripts/eval_falsifier.py split reader` | `run JSON deterministic.commit_forced` | per-run JSON read | BROKEN (CR-02) | Reads `data.get("deterministic")` at top level; real shape nests under `queries[i]`; always returns 0/0 |
| `docs/decisiveness_arm_verdicts.md` | eval_reports/ run dirs | recorded paths + verbatim falsifier output | WIRED | All 6 run dirs (smoke + full per arm) exist and listed in doc; falsifier output pasted verbatim |

---

## Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `docs/decisiveness_arm_verdicts.md` tables | split counts (mi, forced) | Hand-computed from run JSON queries[i].deterministic | YES — confirmed against actual data | VERIFIED (hand-computed tables accurate) |
| `docs/decisiveness_arm_verdicts.md` pasted falsifier output | split annotation "(model-initiated X/Y, forced X/Y)" | eval_falsifier._commit_split_from_run_dir | NO — always 0/0 due to CR-02 | HOLLOW (CR-02) |
| `eval_reports/2026-06-12T*/` run JSONs | step_telemetry.tool_exec_seconds | asyncio timing in graph.py A3 parallel path | YES — confirmed in A3 run files | FLOWING |

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Phase-7 PROMPT-02 grep gate stays green | `pytest tests/unit/test_critique_checks.py::test_prompt_02_grep_gate_no_behavioral_phrases_in_prompts` | PASSED | PASS |
| DEC-02 mock-never-commits test fires forced commit | `pytest tests/unit/test_graph_forced_commit.py::test_forced_commit_triggers_at_step_n` | PASSED (6/6 forced commit tests) | PASS (mocked path only) |
| DEC-04 parallel order-stability + INST-04 timing | `pytest tests/unit/test_graph_parallel_tools.py` | PASSED (5/5 tests) | PASS |
| CR-02: falsifier split reader produces 0/0 on real run data | Python simulation against eval_reports/2026-06-12T07-27-03Z | `buggy: mi=0, forced=0` vs `correct: mi=4, forced=0` | FAIL (CR-02 confirmed) |
| CR-01: viability.py typed path discards PlaceHit hits | Code inspection at viability.py:216 | `hit_dict = hit if isinstance(hit, dict) else {}` — non-dict PlaceHit becomes `{}` | FAIL (CR-01 confirmed) |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| DEC-01 | 13-02, 13-04 | Viability-contract arm: explicit viability definition in commit precondition without violating Phase-7 grep gate | SATISFIED | prompts.py rule8_viability_addendum; PROMPT-02 test passes |
| DEC-02 | 13-01, 13-04 | Forced-commit-at-step-N arm: graph-level, model-independent mechanism | SATISFIED WITH KNOWN BUG | graph.py A2 branch exists; unit test confirms trigger on mock; CR-01 means production synthesizer is dead code |
| DEC-03 | 13-03 | Critique-recalibration co-tuned with DEC-01; threshold + scoping documented before code | SATISFIED | dec03_decision.md predates code commit; both decisions present |
| DEC-04 | 13-04, 13-05 | Parallel tool execution in act() with order-stable results | SATISFIED (latency reduction measurement gap) | asyncio.gather in graph.py; order-stability test passes; tool_exec_seconds in run JSON; reduction vs baseline unmeasurable |
| DEC-05 | 13-06, 13-07 | Arm verdicts documented with per-arm n=5 numbers for all three models; explicit cleared-or-null | SATISFIED WITH CR-02 CAVEAT | docs/decisiveness_arm_verdicts.md complete; correct numbers in tables; closing line unambiguous; falsifier tool print incorrect (0/0) |

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `app/agent/viability.py` | 216 | `hit if isinstance(hit, dict) else {}` — silent Pydantic→empty conversion | BLOCKER (CR-01) | Forced-commit synthesizer produces empty stops in production; mechanism never fires |
| `scripts/eval_falsifier.py` | 205 | `data.get("deterministic")` reads wrong JSON level | BLOCKER (CR-02) | Split reader always returns 0/0; falsifier output in verdict doc is wrong |
| `tests/unit/test_eval_falsifier.py` | 951 | Fixture writes `{"deterministic": {...}}` at top level | BLOCKER (CR-02) | Fixture encodes the bug; tests pass while masking CR-02 |
| `tests/unit/test_graph_forced_commit.py` | 242-272 | Mocks `best_viable_candidate_per_slot` AND `commit_stops` | WARNING | Masks CR-01; test cannot detect synthesizer failure on real PlaceHit shapes |
| `app/agent/revision.py` | 35-37 | `_VIABILITY_CONTRACT_ENABLED` read at import time (vs build time in graph.py) | WARNING (WR-02) | Flag split-read risk: monkeypatch env changes between import and graph-build can desync DEC-01 and DEC-03 |
| `app/agent/revision.py` (et al.) | Multiple | Truthy-env-flag parsing duplicated 6 times | WARNING (WR-09) | DRY violation per CLAUDE.md; desync risk between copies |

---

## Human Verification Required

### 1. CR-01 Production Impact Assessment

**Test:** Run a live A2 arm run after applying the CR-01 fix (convert PlaceHit via model_dump in best_viable_candidate_per_slot, and add rationale field in the synthesizer), then compare commit_forced counts to the Phase-13 baseline (forced=0).
**Expected:** forced > 0 when all slots have viable candidates at step 6.
**Why human:** Requires live API spend; can't verify synthesizer fix without real PlaceHit objects from actual semantic_search calls.

### 2. A2 Verdict Annotation

**Test:** Read the amended A2 section of docs/decisiveness_arm_verdicts.md after CR-01 annotation is added, and confirm the note: (a) states mechanism was inoperative due to synthesis bug, (b) preserves the 0.500 model-initiated finding, (c) explicitly flags that forced mechanism is untested at n=5, (d) states whether Phase 14 reserves a slot for A2 retry.
**Expected:** Annotation is unambiguous; Phase-14 entry gate input is honest.
**Why human:** Requires editorial judgment on how to characterize the bug's impact without invalidating the broader null result.

### 3. A3 Latency Criterion Respecification

**Test:** Run a flag-off (control) arm run with the same arm matrix config but PARALLEL_TOOL_EXECUTION_ENABLED unset to obtain a sequential baseline with step_telemetry. Compare tool_exec_seconds to the A3 arm run.
**Expected:** Measurable delta exists; if positive (A3 faster), SC-3 is retroactively satisfied.
**Why human:** Requires API spend and operator decision on whether this is in-scope for Phase 13 or deferred to Phase 15.

---

## Gaps Summary

**SC-3 partial fail (DEC-04 latency reduction):** The parallel tool execution mechanism is correctly implemented and tested. INST-04 `tool_exec_seconds` IS recorded in every A3 run JSON. However, the success criterion requires "measurable latency reduction" — a reduction requires a before/after comparison. The Phase-12 comparison-floor run dirs (the designated "before") were produced before Phase 13 added step_telemetry instrumentation, so their `tool_exec_seconds` is `None`. The reduction is structurally unmeasurable against the specified baseline. The verdict doc documents this honestly as UNMEASURABLE. The gap is the absence of a same-phase control (flag-off) run that would provide a valid sequential baseline.

**CR-01 and CR-02 are blockers for PHASE 14 integrity, not for the Phase 13 null result:** The honest null result ("no arm cleared INST-05") survives both bugs — the forced mechanism producing 0 forced commits is consistent with both CR-01 and genuine gate non-satisfaction, and the commit rates as measured are accurate. However:
- CR-01 means any Phase-14 decision to retry A2 rests on a false causal explanation
- CR-02 means the D-13-04 honesty contract's enforcement tool is broken for all future arm runs

These must be fixed before Phase 14 runs begin. They do not require reopening Phase 13 if the verdict doc is annotated per WR-10.

---

_Verified: 2026-06-12T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
