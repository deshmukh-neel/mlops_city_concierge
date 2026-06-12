---
phase: 14-richer-state-replay-conditional
plan: "04"
subsystem: eval-harness
tags: [replay, live-eval, falsifier, arm-verdicts, honesty-contract]
dependency_graph:
  requires: [14-01, 14-02, 14-03]
  provides:
    - "R1 measured verdict (eval_reports/2026-06-12T20-00-05Z)"
    - "R2 measured verdict (eval_reports/2026-06-12T20-58-32Z)"
    - "docs/replay_arm_verdicts.md R1 + R2 sections filled"
  affects: [14-05]
tech_stack:
  added: []
  patterns:
    - "Smoke-first live-spend contract (D-14-02): n=1 arm_flags verification before every n=5"
    - "Three-delta verdict tables: pooled rate + delta vs flag-off floor + delta vs A2 (0.500)"
    - "Honest ERRORED recording: provider 400s recorded verbatim, never as zeros (D-11-14)"
key_files:
  created: []
  modified:
    - docs/replay_arm_verdicts.md
decisions:
  - "Flag-off floor reuses Phase-13 plateau numbers (A1 run dir 2026-06-12T06-25-52Z) — no live control run spent; byte-identity verified in scripted mode (PASS)"
  - "R1 (REPLAY_MULTI_MESSAGE_ENABLED=1): gpt-5-mini 0.500 median-weighted — exactly matches A2, delta vs A2 = 0.000; positive signal vs floor (+0.500) but does NOT clear the 0.6 bar"
  - "R2 (REPLAY_CONTENT_BLOCKS_ENABLED=1): EXPECTED-NULL REFUTED for gpt-5-mini — 10/10 episodes deterministic 400 'No tool output found for function call'; str() collapse was load-bearing protection for Responses-API list content; NEGATIVE signal, cannot qualify R3"
  - "EXPECTED-NULL CONFIRMED for str-content models: anchor held median 1.000/1.000 in both arms; deepseek at plateau (raw 1/10 both arms)"
  - "Anchor non-regression confirmed for BOTH arms by the INST-05 median criterion"
metrics:
  duration: ~95min (dominated by two live n=5 matrix runs ~50min each wall time)
  completed: "2026-06-12"
  tasks_completed: 3
  files_created: 0
---

# Phase 14 Plan 04: Run R1 + R2 Judged Arms Summary

**One-liner:** R1 (multi-message replay) measured at 0.500 — identical to A2, no incremental signal; R2 (content-block preservation) catastrophically breaks gpt-5-mini (10/10 provider 400s) because Responses-API list content embeds function_call items the pruner's tool outputs no longer answer — the str() collapse was load-bearing, refuting the EXPECTED-NULL audit prediction for that cell.

## What Was Built

### Task 1: Flag-off byte-identity verification + floor selection (commit 36f9c97)

- Scripted-mode smoke (`eval_reports/2026-06-12T19-49-39Z`, no live keys, no flags set) confirmed `deterministic.arm_flags` carries both new replay keys as `False` alongside all four Phase-13 DEC keys — byte-identity PASS.
- Flag-off floor fixed as the recorded Phase-13 plateau (A1 full run `eval_reports/2026-06-12T06-25-52Z`): gpt-5-mini 0.000, anchor 1.000, deepseek 0.000. No live control run spent, preserving the ≤4-run cap.

### Task 2: R1 — REPLAY_MULTI_MESSAGE_ENABLED=1, pure (commit aeaf832)

- Smoke `eval_reports/2026-06-12T19-50-32Z`: arm_flags `{replay_multi_message: True, replay_content_blocks: False, DEC flags off}` — confounded-run guard PASS.
- Full n=5 `eval_reports/2026-06-12T20-00-05Z`: gpt-5-mini **0.500** median-weighted (model-initiated 4/10, forced 0/10; omakase 1.000, refinement 0.000), anchor **1.000** (10/10, non-regression PASS), deepseek **0.100** raw (1/10, informational). Falsifier exit **1 (FAIL)**.
- Verdict: R1 delta vs floor +0.500, delta vs A2 **±0.000** — multi-message replay reproduces but does not exceed the best DEC arm; the omakase/refinement asymmetry persists unchanged. Human-verified checkpoint approved.

### Task 3: R2 — REPLAY_CONTENT_BLOCKS_ENABLED=1, pure (commit e4aac9e)

- Smoke `eval_reports/2026-06-12T20-53-05Z`: arm_flags `{replay_content_blocks: True, replay_multi_message: False, DEC flags off}` — guard PASS.
- Full n=5 `eval_reports/2026-06-12T20-58-32Z`: **gpt-5-mini ERRORED 10/10 episodes** with deterministic `400 BadRequestError: "No tool output found for function call call_..."`. Anchor 1.000 median-weighted (model-initiated 7/10, non-regression PASS by median criterion), deepseek 0.100 raw. Falsifier exit **1 (FAIL)** — gpt-5-mini "N/A (no evaluable cells)".
- NOT infrastructure failure: gpt-4o-mini + deepseek completed normally in the same run; DB and both API keys healthy throughout.
- Root cause (verified in code): gpt-5 family routes through `OpenAIReasoningChatModel` with `use_responses_api=True` (`app/llm_factory.py:360`); Responses-API `AIMessage.content` is a content-block LIST embedding function-call items. Preserving the list verbatim re-sends `function_call` items whose ToolMessage outputs `_prune_for_llm` dropped → OpenAI rejects. The `AIMessage(content=m.content, ...)` constructor strips the `.tool_calls` attribute but NOT the function-call state inside the content list — a second channel D-14-06 did not account for.
- Criterion-2 reconciliation recorded in the doc: `str()` collapse caused NO observable loss anywhere, was a no-op for the two str-content models (as predicted), and was load-bearing protection for gpt-5-mini (prediction refuted). R2 signal classification: **NEGATIVE** — cannot qualify the R3 combo under D-14-01 (the formal R3 decision belongs to Plan 14-05).

## Run Budget

| Slot | Arm | Run dirs | Status |
|------|-----|----------|--------|
| 1 of 4 | R1 | smoke 2026-06-12T19-50-32Z, full 2026-06-12T20-00-05Z | SPENT |
| 2 of 4 | R2 | smoke 2026-06-12T20-53-05Z, full 2026-06-12T20-58-32Z | SPENT |
| 3 of 4 | R3 (conditional) | — | decision pending Plan 14-05 (R2 negative → precondition fails) |
| 4 of 4 | discretionary valve | — | reserved |

## Deviations from Plan

### Process deviations (documented for Wave 4 / future phases)

**1. Smoke gate checked arm_flags but not error cells — R2 full spend proceeded past a visible failure**
- **Found during:** Task 3 post-run analysis
- **Issue:** The R2 n=1 smoke ALREADY contained both gpt-5-mini 400 error cells (`status=error`). The D-14-02 smoke contract verifies only the `arm_flags` dict, which passed, so the full n=5 was spent on a run where gpt-5-mini was guaranteed to error.
- **Mitigating value:** The full run produced the anchor + deepseek measurements and upgraded the gpt-5-mini failure from n=1 anecdote to deterministic-across-10-episodes evidence.
- **Learning:** Future smoke contracts should inspect smoke error cells in addition to arm_flags before authorizing a full spend. Recorded as a process note in the R2 section of the verdict doc.

**2. [Rule 1 - correction] D-14-05 audit half-(b) openai row was wrong for gpt-5-mini**
- **Found during:** Task 3 root-cause analysis
- **Issue:** The audit examined `OpenAIReasoningAdapter` and concluded "content shape: str → NO-OP", but missed that the chat-model FACTORY (`app/llm_factory.py`) routes the gpt-5 family onto the Responses API where content IS a list. The audit looked at the adapter, not the factory.
- **Fix:** Post-run annotation blockquote added to the R2 Evidence Audit section (Phase-13 CR-01/CR-02 annotation precedent — original text preserved, correction appended).
- **Files modified:** docs/replay_arm_verdicts.md
- **Commit:** e4aac9e
- **Learning:** Structural audits of message shape must trace the full construction path (factory → chat model → adapter), not just the adapter layer.

### Auto-fixed Issues

None beyond the audit-correction annotation above — no code changes were required by this plan (verdict-doc-only plan, as specified in frontmatter `files_modified`).

## Verification

- R1 + R2 sections of docs/replay_arm_verdicts.md filled with real run dirs, verified smoke arm_flags, three-delta per-model tables, verbatim falsifier output, and closing verdicts: DONE
- Each full run preceded by a smoke whose arm_flags matched the intended pure-arm config (DEC flags off): DONE (both guards PASS)
- Anchor non-regression checked for both arms: CONFIRMED both (median criterion; R2 raw-count movement honestly noted, not flagged)
- ≤4-run cap respected: R1 + R2 = 2 of 4
- Plan automated checks: `grep REPLAY_MULTI_MESSAGE_ENABLED docs/replay_arm_verdicts.md` PASS, `grep "R2 Evidence Audit" docs/replay_arm_verdicts.md` PASS

## Handoff to Plan 14-05

- **R3 qualification evidence:** R1 positive (+0.500 vs floor) but R2 NEGATIVE (catastrophic error, no commit measurement). D-14-01 requires BOTH arms positive — R3 precondition appears unsatisfied; 14-05 records the formal decision.
- **Discretionary 4th-run valve evidence:** R1 (best replay arm, 0.500) and A2 (0.500) each show positive signal independently — the D-14-01 escalation-valve precondition MAY be satisfiable; 14-05 evaluates whether an R1+FORCED_COMMIT_STEP=6 stack run is recommended before declaring plateau.
- **ARCH-FUT-01 inputs:** R1 delta vs A2 = 0.000 and R2 negative — cumulative evidence chain for the plateau recommendation is now complete pending 14-05's closing verdict.

## Known Stubs

The remaining `[fill]` slots in docs/replay_arm_verdicts.md (R3 section, Closing Verdict, ARCH-FUT-01 Evaluation, Explicit Closing Line, Phase-15 Consequence) are INTENTIONAL — they are Plan 14-05's deliverable per the phase structure, not omissions of this plan.

## Self-Check: PASSED

- docs/replay_arm_verdicts.md exists and contains both run-dir paths: FOUND
- Commit 36f9c97 (Task 1): FOUND
- Commit aeaf832 (Task 2): FOUND
- Commit e4aac9e (Task 3): FOUND
- eval_reports/2026-06-12T20-00-05Z (R1 full, 31 files): FOUND
- eval_reports/2026-06-12T20-58-32Z (R2 full, 31 files): FOUND
