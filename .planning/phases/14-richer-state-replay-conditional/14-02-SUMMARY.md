---
phase: 14-richer-state-replay-conditional
plan: "02"
subsystem: eval-harness
tags: [evidence-audit, replay, documentation, testing]
dependency_graph:
  requires: [14-01]
  provides: [docs/replay_arm_verdicts.md, scripts/audit_list_content_aimessages.py]
  affects: [14-03, 14-04, 14-05]
tech_stack:
  added: []
  patterns:
    - "Zero-spend audit pattern: half (a) run-dir scan + half (b) structural analysis"
    - "Verdict doc scaffold mirrors DEC-05 structure with labeled [fill] placeholders"
key_files:
  created:
    - scripts/audit_list_content_aimessages.py
    - docs/replay_arm_verdicts.md
    - tests/unit/test_audit_list_content.py
  modified: []
decisions:
  - "D-14-05: R2 EXPECTED-NULL on gpt-5-mini/gpt-4o-mini/deepseek-reasoner — str() collapse is a NO-OP for all three RUN models; only AnthropicAdapter uses list-content, and it is deferred (D-12-09)"
  - "Verdict doc cross-links (not appends to) docs/decisiveness_arm_verdicts.md — Phase-13 record stays immutable"
  - "ARCH-FUT-01 Evaluation section filled as USER CHECKPOINT scaffold per D-14-08"
metrics:
  duration: ~15min
  completed: "2026-06-12"
  tasks_completed: 2
  files_created: 3
---

# Phase 14 Plan 02: Evidence Audit and Verdict Scaffold Summary

**One-liner:** Zero-spend REPLAY-02 evidence audit (D-14-05) derives EXPECTED-NULL verdict from adapter structure; docs/replay_arm_verdicts.md scaffolded mirroring DEC-05 with R2 audit filled and all live-run slots as labeled placeholders.

## What Was Built

### Task 1: Zero-spend list-content audit script + unit tests

`scripts/audit_list_content_aimessages.py` implements the D-14-05 evidence audit in two complementary halves:

**Half (a) — Run-dir scan:** Iterates Phase-12/13 arm run dirs (defaults to the three n=5 arm runs referenced in `docs/decisiveness_arm_verdicts.md`) and reports the persisted EvalRunReport shape. Key finding confirmed: `queries[i].deterministic.tool_calls` is an integer count with no serialized AIMessage `.content` or `.additional_kwargs`. The run JSONs are structurally insufficient to directly answer "did an AIMessage carry list content pre-cutoff." This is reported as an explicit finding, not a crash.

**Half (b) — Structural adapter analysis:** Derives the ground truth from `app/agent/adapters/*.py`. All four Phase-9 adapters are classified:
- `OpenAIReasoningAdapter` (gpt-5-mini, gpt-4o-mini): `AIMessage.content` is a plain string; reasoning state lives in `additional_kwargs["reasoning_content"]`. `str()` collapse is a NO-OP.
- `DeepSeekReasonerAdapter` (deepseek-reasoner): same shape. `str()` collapse is a NO-OP.
- `AnthropicAdapter`: uses a `list[dict]` content block list (thinking + text blocks). `str()` collapse would be LOSSY — but Anthropic is deferred (D-12-09) and not in the run matrix.
- `GeminiAdapter`: string content, deferred.

**Overall verdict printed by the script:** R2 EXPECTED-NULL on all three tested cells. The script exits 0.

`tests/unit/test_audit_list_content.py` provides 14 unit tests covering: run-dir scan with fixture JSON (correct shape detected, no crash), missing dir, malformed JSON, surprise message content flag, structural analysis (all run models str-content, expected-null verdict, anthropic is the only list-content adapter, anthropic deferred), and `main()` exit 0.

### Task 2: Scaffold docs/replay_arm_verdicts.md

`docs/replay_arm_verdicts.md` mirrors the DEC-05 structure exactly:
- INST-05 Falsifier Definition with Phase-14 comparison points (flag-off floor 0.000, best DEC arm A2 0.500)
- Run Budget Contract (≤4 full runs, R1 + R2 + conditional R3 + discretionary valve)
- R1 section (REPLAY_MULTI_MESSAGE_ENABLED=1) with three delta columns per model and labeled [fill] placeholders
- R2 section (REPLAY_CONTENT_BLOCKS_ENABLED=1) with the same structure PLUS the R2 Evidence Audit subsection fully filled with the Task 1 conclusion
- R3 Conditional section with D-14-01 qualification conditions
- Closing Verdict with Per-Arm Summary Table (three delta columns), ARCH-FUT-01 Evaluation scaffold (USER CHECKPOINT per D-14-08), Explicit Closing Line slot, and Phase-15 Consequence slot

The doc cross-links `docs/decisiveness_arm_verdicts.md` (not appended to). The Phase-13 record was not modified (`git diff docs/decisiveness_arm_verdicts.md` is empty).

## Decisions Made

- D-14-05 confirmed: str() collapse at `graph.py:232` was a NO-OP for all three RUN model cells. Run JSONs lack message traces; structural analysis is the definitive source. R2 is EXPECTED-NULL but still runs (criterion 2 requires a measured delta).
- Verdict doc uses the DEC-05 exact structural mirror: INST-05 falsifier, run budget, per-arm sections, closing verdict with ARCH-FUT-01 section and explicit closing line.
- ARCH-FUT-01 Evaluation section: scaffolded as a USER CHECKPOINT (D-14-08) with three required parts ((a) cumulative evidence chain, (b) contingency restatement, (c) written recommendation) marked as fill-after-live-runs.

## Deviations from Plan

None — plan executed exactly as written. The plan's ground-truth note that "run JSONs do NOT contain full message traces" was confirmed correct by half (a) scan.

## Self-Check

### Created files exist:
- `scripts/audit_list_content_aimessages.py`: FOUND
- `docs/replay_arm_verdicts.md`: FOUND
- `tests/unit/test_audit_list_content.py`: FOUND

### Commits exist:
- `95deab2`: feat(14-02): REPLAY-02 zero-spend evidence audit script + unit tests — FOUND
- `4678d6f`: docs(14-02): scaffold docs/replay_arm_verdicts.md with filled R2 evidence audit — FOUND

### Verification commands passed:
- `poetry run pytest tests/unit/test_audit_list_content.py -q`: 14 passed
- `poetry run python scripts/audit_list_content_aimessages.py --help`: exit 0
- `make lint`: All checks passed
- `make typecheck`: Success: no issues found in 40 source files

## Self-Check: PASSED
