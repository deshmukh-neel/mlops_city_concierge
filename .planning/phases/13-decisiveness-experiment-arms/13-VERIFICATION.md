---
phase: 13-decisiveness-experiment-arms
verified: 2026-06-12T18:30:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 4/5
  gaps_closed:
    - "SC-3 partial fail — ROADMAP criterion 3 respecified to absolute tool_exec_seconds for future-baseline use; constraint annotated in ROADMAP.md and docs/decisiveness_arm_verdicts.md A3 section (Plan 13-10)"
    - "CR-01 blocker — forced-commit synthesizer dead code fixed: viability.py typed path uses model_dump(mode='json') for PlaceHit; graph.py synthesizer builds commit-shaped stops with required rationale; non-mocked regression test added; A2 verdict annotated (Plan 13-08)"
    - "CR-02 blocker — falsifier split reader fixed to iterate queries[i].deterministic; fixture rewritten to real EvalRunReport shape; regression tests added; verdict doc annotated that pasted 0/0 was tool bug (Plan 13-09)"
  gaps_remaining: []
  regressions: []
---

# Phase 13: Decisiveness Experiment Arms — Re-Verification Report

**Phase Goal:** Four coupled experiment arms are implemented, run at n=5 temp=1.0 against the Phase-12 comparison floor, and their verdicts are documented — revealing whether any arm clears the falsifier bar or all plateau below it.
**Verified:** 2026-06-12T18:30:00Z
**Status:** passed
**Re-verification:** Yes — after gap closure (Plans 13-08, 13-09, 13-10)

---

## Re-verification Context

The prior verification (2026-06-12T12:00:00Z) returned `gaps_found` 4/5 with three gaps requiring closure:

1. **SC-3 partial fail** — "measurable latency reduction" was structurally unmeasurable; Phase-12 comparison-floor runs predate INST-04 step_telemetry.
2. **CR-01** — forced-commit synthesizer dead code in production (PlaceHit→{} silent conversion + missing required rationale field).
3. **CR-02** — falsifier split reader reading `deterministic` at wrong JSON level, always returning 0/0.

Gap plans 13-08, 13-09, 13-10 addressed all three. A fresh code review (13-REVIEW.md, 2026-06-12T17:54Z) confirmed all four prior findings (CR-01, CR-02, WR-02, WR-09) are fixed, 0 critical findings. Full test suite reported by orchestrator: 1376 passed, 53 skipped.

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | DEC-01 viability-contract arm ships without touching Phase-7 CI grep gate sections | VERIFIED | `test_prompt_02_grep_gate_no_behavioral_phrases_in_prompts` passes (confirmed live run). `rule8_viability_addendum` appended after SYSTEM_PROMPT; no Phase-7 forbidden phrases in prompts.py. Grep gate passes after all gap-closure plans. |
| 2 | DEC-02 forced-commit is a graph-level mechanism confirmed by a unit test on a mock that never calls commit_itinerary | VERIFIED | `test_forced_commit_triggers_at_step_n` passes (mocked path). `test_forced_commit_synthesizer_real_placehit_shapes` passes (non-mocked regression on real PlaceHit shapes, CR-01 repaired — commit_forced=True, stops non-empty). graph.py synthesizer now builds commit-shaped dicts with synthesized rationale at lines 637–643; viability.py typed path uses `model_dump(mode='json')` at lines 185, 222. |
| 3 | DEC-04 parallel tool execution runs concurrently with order-stable results; AND absolute gpt-4o-mini tool_exec_seconds at n=5 recorded in run JSON for future-baseline use | VERIFIED | Criterion respecified (Plan 13-10) — the "measurable reduction delta" replaced with "absolute latency recorded for future-baseline use" due to structurally discovered constraint (Phase-12 floors predate INST-04 step_telemetry). ROADMAP.md line 91 contains the respecified criterion with constraint annotation. `docs/decisiveness_arm_verdicts.md` A3 section echoes respecification and names the absolute tool_exec_seconds values as the future-baseline artifact. asyncio.gather in graph.py provides concurrency; order-stability test passes. |
| 4 | DEC-03 critique-recalibration is co-tuned with DEC-01; threshold direction and low_similarity scoping decision documented before threshold code lands | VERIFIED | `docs/decisiveness_dec03_decision.md` exists with both decisions documented. Commit ordering (doc before code) confirmed in git log. WR-02 co-tuning split-read risk closed by Plan 13-10: `_VIABILITY_CONTRACT_ENABLED` module constant removed; `revision.py` now reads flag live via `env_flag("VIABILITY_CONTRACT_ENABLED")` per call (line 203), eliminating the import-time/build-time desync hazard. |
| 5 | DEC-05 arm-verdict document records per-arm n=5 numbers for gpt-5-mini, deepseek-reasoner, gpt-4o-mini anchor and explicitly states which arm cleared INST-05 bar or records honest null result | VERIFIED | `docs/decisiveness_arm_verdicts.md` contains per-arm tables for all three models. Closing verdict: "No arm cleared the INST-05 falsifier bar" present at 2 locations. Pasted 0/0 falsifier output annotated as CR-02 tool bug (Plan 13-09); hand-computed table numbers affirmed correct; pasted 0/0 lines preserved as historical record (confirmed: 9 occurrences). A2 section annotated with CR-01 synthesis bug and Phase-14 retry disposition. |

**Score:** 5/5 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `app/agent/viability.py` | Typed-path PlaceHit→dict via model_dump(mode='json') | VERIFIED | Lines 185, 222: `hit.model_dump(mode="json")` on both typed and untyped paths; `else {}` anti-pattern gone (grep confirms 0 matches) |
| `app/agent/graph.py` | Synthesized rationale in forced-commit branch + env_flag calls at build time | VERIFIED | Lines 619, 637–643: synthesized rationale string. Lines 306–307: `env_flag("VIABILITY_CONTRACT_ENABLED")` and `env_flag("PARALLEL_TOOL_EXECUTION_ENABLED")` at build time |
| `tests/unit/test_graph_forced_commit.py` | Non-mocked regression test on real PlaceHit shapes | VERIFIED | `test_forced_commit_synthesizer_real_placehit_shapes` at line 343: imports real PlaceHit, puts in scratch, does NOT patch best_viable_candidate_per_slot or commit_stops, asserts commit_forced=True and stops non-empty |
| `scripts/eval_falsifier.py` | Split reader iterating queries[i].deterministic | VERIFIED | Line 207: `for query in data.get("queries") or []:`; `data.get("deterministic")` top-level read gone (grep confirms 0 matches) |
| `tests/unit/test_eval_falsifier.py` | Fixture using real EvalRunReport shape + CR-02 regression tests | VERIFIED | `_write_run_file` rebuilt to use EvalRunReport/QueryEvalResult/DeterministicEvalResult dataclasses. `test_cr02_real_shape_returns_nonzero_counts` and `test_cr02_old_top_level_shape_returns_zeros` at lines 1198, 1232 |
| `app/config.py` | `env_flag(name)` truthy-set helper | VERIFIED | Line 14: `def env_flag(name: str) -> bool` with truthy set `{"1", "true", "yes", "on"}` |
| `app/agent/revision.py` | Live env_flag read; `_VIABILITY_CONTRACT_ENABLED` constant removed | VERIFIED | Line 203: `env_flag("VIABILITY_CONTRACT_ENABLED")` live per call. grep for `_VIABILITY_CONTRACT_ENABLED` in app/ and scripts/ returns 0 matches. |
| `scripts/eval_agent.py` | env_flag used in arm_flags assembly | VERIFIED | Lines 929, 931, 1173: `env_flag` called for all three boolean arm flags |
| `tests/unit/test_config.py` | env_flag truthiness tests | VERIFIED | `test_env_flag_truthy_values`, `test_env_flag_falsy_values`, `test_env_flag_unset_returns_false` at lines 123, 133, 139 |
| `.planning/ROADMAP.md` | SC-3 respecified with constraint annotation | VERIFIED | Line 91: criterion requires "absolute gpt-4o-mini tool-execution latency at n=5 (INST-04 `tool_exec_seconds`, summed per run) is recorded in run JSON for future-baseline use" with discovered-constraint parenthetical |
| `docs/decisiveness_arm_verdicts.md` | All required annotations (CR-01, CR-02, SC-3 echo, null result) | VERIFIED | "synthesis bug" at line 219; "untested" at line 237 (as "UNTESTED at n=5"); "tool bug" at line 191 (A2 CR-02 annotation); "CR-02" annotation in A1/A2/A3; "future baseline" / "future-baseline use" at lines 370, 372; "No arm cleared" at 2 locations |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `viability.py best_viable_candidate_per_slot typed path` | `PlaceHit.model_dump(mode='json')` | isinstance(hit, BaseModel) branch | VERIFIED | Lines 185, 222 in viability.py; both typed and untyped paths now JSON-safe |
| `graph.py forced-commit synthesizer` | `commit_stops` | commit-shaped dicts with synthesized rationale | VERIFIED | Lines 637–643: explicit rationale field built; commit_stops accepts them; regression test confirms end-to-end |
| `eval_falsifier._commit_split_from_run_dir` | `queries[i].deterministic` | per-query iteration | VERIFIED | Line 207: iterates `data.get("queries") or []`; reads `query.get("deterministic")` per entry |
| `revision.py _diagnose_last_tool_result` | `env_flag("VIABILITY_CONTRACT_ENABLED")` | live call per invocation | VERIFIED | Line 203: live read; import-time constant eliminated; DEC-01 and DEC-03 now desync-proof |
| `eval_agent.py arm_flags` | `env_flag` (app.config) | imported and called | VERIFIED | Line 41 import; lines 929, 931, 1173 usage — same parser as graph.py build-time reads |

---

## Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `docs/decisiveness_arm_verdicts.md` tables | split counts (mi, forced) | Hand-computed from run JSON queries[i].deterministic | YES — confirmed against actual data | VERIFIED (tables correct) |
| `docs/decisiveness_arm_verdicts.md` pasted falsifier output | "(model-initiated 0/0, forced 0/0)" | eval_falsifier (now fixed) | HISTORICAL RECORD — annotated as CR-02 tool bug | VERIFIED (annotation present; tool now fixed; 9 preserved 0/0 lines are historical record) |
| `eval_reports/2026-06-12T*/` run JSONs | step_telemetry.tool_exec_seconds | asyncio timing in graph.py A3 parallel path | YES — absolute values in run JSON | FLOWING (future-baseline artifact per respecified SC-3) |

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Phase-7 grep gate stays green after all gap-closure plans | `pytest tests/unit/test_critique_checks.py::test_prompt_02_grep_gate_no_behavioral_phrases_in_prompts -q` | 1 passed | PASS |
| CR-01 regression tests + non-mocked synthesizer test | `pytest tests/unit/test_viability.py tests/unit/test_graph_forced_commit.py -q` | 34 passed (includes new tests) | PASS |
| CR-02 regression tests (real EvalRunReport shape) | `pytest tests/unit/test_eval_falsifier.py -q` | 55 passed (includes new CR-02 tests) | PASS |
| env_flag DRY helper tests | `pytest tests/unit/test_config.py -q` | 31 passed (includes 3 new env_flag tests) | PASS |
| Combined gap-closure suite | `pytest tests/unit/test_viability.py tests/unit/test_graph_forced_commit.py tests/unit/test_eval_falsifier.py tests/unit/test_config.py -q` | 120 passed | PASS |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| DEC-01 | 13-02, 13-04 | Viability-contract arm: explicit viability definition without violating Phase-7 grep gate | SATISFIED | prompts.py rule8_viability_addendum; PROMPT-02 test passes; WR-02 co-tuning split-read closed by 13-10 |
| DEC-02 | 13-01, 13-04, 13-08 | Forced-commit-at-step-N arm: graph-level, model-independent mechanism | SATISFIED | CR-01 fixed by 13-08; synthesizer now produces valid commit stops; non-mocked regression test pins this; A2 verdict annotated |
| DEC-03 | 13-03 | Critique-recalibration co-tuned with DEC-01; documented before code | SATISFIED | dec03_decision.md predates code commit; env_flag live read closes WR-02 co-tuning hazard |
| DEC-04 | 13-04, 13-05, 13-10 | Parallel tool execution with order-stable results; absolute latency recorded for future-baseline use | SATISFIED | asyncio.gather in graph.py; order-stability test passes; tool_exec_seconds in A3 run JSON; SC-3 respecified per locked D-13-02 constraint |
| DEC-05 | 13-06, 13-07, 13-08, 13-09 | Arm verdicts documented with per-arm n=5 numbers; explicit cleared-or-null statement | SATISFIED | docs/decisiveness_arm_verdicts.md complete; honest null result preserved; CR-01/CR-02 annotations present; pasted 0/0 flagged as tool bug |

---

## Anti-Patterns Found (Re-Verification — Advisory Warnings from 13-REVIEW.md)

The code review (13-REVIEW.md, 2026-06-12T17:54Z) found 0 critical findings after gap closure. Five advisory warnings remain. None are blockers for phase completion:

| File | Pattern | Severity | Assessment |
|------|---------|----------|------------|
| `scripts/eval_falsifier.py` | WR-01: `_commit_split_from_run_dir` "never raises" contract broken for valid-JSON non-dict top-level artifacts (AttributeError on list/string) | WARNING | Does not affect correctness of the fixed reader for real run artifacts; affects resilience to malformed files. Advisory — acceptable for Phase 14 entry. |
| `scripts/eval_falsifier.py` | WR-02: Split denominator is total commits, not episodes — diverges from D-13-04 documented format | WARNING | Numerators correct; denominators print "of N commits" not "of N episodes". Hand-computed tables in verdict doc use episodes. Informational inconsistency; does not invalidate Phase 13 verdict. |
| `scripts/eval_falsifier.py` | WR-03: Split computed over different scenario universe than the rates it annotates | WARNING | `scenario_ids` parameter exists and is tested but not wired in main(). Annotation may include out-of-scope scenarios. Does not affect the honest null result finding. |
| `app/agent/prompts.py` | WR-04: "Rule 8" viability addendum renders ~120 lines from rule 8, after REVISION_GUIDANCE — naming misleads | WARNING | A1 arm may have underperformed partly due to placement. Informational for Phase 14 retry. Naming correctable without code change. |
| `app/agent/revision.py` | WR-05: Comment claims DEC-01/DEC-03 pick up env changes at the same time; live vs build-time difference remains | WARNING | Build-time (graph.py) vs call-time (revision.py) semantics differ but are practically identical under export-then-run eval workflow. WR-02 hazard closed; this is a comment accuracy issue only. |

No `TBD`, `FIXME`, or `XXX` debt markers found in files modified by the gap-closure plans.

---

## Human Verification Required

None — all prior human verification items from the initial verification are resolved:

1. **CR-01 Production Impact Assessment** — resolved by Plan 13-08: synthesizer is now operative (non-mocked regression test proves commit_forced=True on real PlaceHit shapes); A2 verdict annotated that forced mechanism was untested at n=5.
2. **A2 Verdict Annotation** — resolved by Plan 13-08: annotation is unambiguous, preserves 0.500 model-initiated finding, states forced mechanism untested at n=5, and states Phase-14 A2 retry disposition (reserved as Phase-14/15 candidate per D-13-02 cap).
3. **A3 Latency Criterion Respecification** — resolved by Plan 13-10 (zero-spend path): criterion respecified at ROADMAP level; no additional live run required.

---

## Gaps Summary

No gaps. All three prior gaps are closed:

**SC-3** — ROADMAP criterion 3 respecified from "measurable latency reduction" to "absolute tool_exec_seconds recorded for future-baseline use." The respecification is consistent with locked decision D-13-02 (four-run cap fully consumed). The discovered constraint (Phase-12 floors lack step_telemetry) is named, not hidden, in both ROADMAP.md and the A3 verdict section.

**CR-01** — Forced-commit synthesizer is now operative end-to-end: viability.py typed path uses `model_dump(mode='json')` for PlaceHit objects; graph.py synthesizer builds commit-shaped stops with required rationale field. A non-mocked regression test (`test_forced_commit_synthesizer_real_placehit_shapes`) asserts commit_forced=True on real PlaceHit objects — would fail on the pre-fix code. A2 verdict is annotated with the synthesis bug, the model-initiated 0.500 finding stands, forced mechanism is flagged untested at n=5, and Phase-14 A2 retry disposition is stated.

**CR-02** — Falsifier split reader iterates `queries[i].deterministic` (not top-level). Test fixture rebuilt to use real EvalRunReport dataclasses from `scripts.eval_agent` (shape cannot drift silently). Regression test `test_cr02_real_shape_returns_nonzero_counts` returns (2,2) on real shape and would return (0,0) on old buggy reader. Verdict doc annotates pasted 0/0 lines as CR-02 tool bug; hand-computed tables affirmed correct; historical 0/0 lines preserved.

**Honest null result intact:** "No arm cleared the INST-05 falsifier bar" appears at 2 locations in `docs/decisiveness_arm_verdicts.md` and is unaltered across all gap-closure plans. Phase 14 entry gate (DEC-05 verdict: all arms plateau below INST-05 bar) is OPEN.

---

_Verified: 2026-06-12T18:30:00Z_
_Verifier: Claude (gsd-verifier)_
_Re-verification: Yes — after gap closure (Plans 13-08, 13-09, 13-10)_
