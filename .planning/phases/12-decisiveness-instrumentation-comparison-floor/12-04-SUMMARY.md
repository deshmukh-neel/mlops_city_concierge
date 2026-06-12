---
phase: 12-decisiveness-instrumentation-comparison-floor
plan: "04"
subsystem: eval-harness
tags: [bookkeeping, deferred-cells, comparison-floor, docs]
dependency_graph:
  requires: []
  provides:
    - "D-12-09 gemini deferral recorded in code + docs + planning"
    - "ANCH-02 satisfied as deferred-with-note"
    - "ANCH-03 satisfied (reinterpreted): non-deferred cells honest n=5 confirmed by parity test"
  affects:
    - "tests/unit/test_eval_matrix.py (_DEFERRED_BASELINE_CELLS)"
    - "docs/baseline_regen.md (Gemini deferral section)"
    - "docs/eval_gates.md (Gemini deferral section)"
    - ".planning/ROADMAP.md (Phase 12 success criterion 4)"
    - ".planning/REQUIREMENTS.md (ANCH-02, ANCH-03)"
tech_stack:
  added: []
  patterns:
    - "_DEFERRED_BASELINE_CELLS deferred-cell guard with decision-ID rationale comments"
    - "Symmetric docs: deferral sections in both baseline_regen.md and eval_gates.md"
key_files:
  created: []
  modified:
    - "tests/unit/test_eval_matrix.py"
    - "docs/baseline_regen.md"
    - "docs/eval_gates.md"
    - ".planning/ROADMAP.md"
    - ".planning/REQUIREMENTS.md"
decisions:
  - "D-12-09 honored: gemini n=5 baseline deferred at user budget decision (no quota/billing top-up); joins anthropic as second deferred cell"
  - "ANCH-02 satisfied as deferred-with-note (single scored run hit 1.0 = measurement debt, not unknown risk)"
  - "ANCH-03 reinterpreted: comparison floor = matrix minus BOTH anthropic AND gemini; parity test confirms non-deferred cells honest n=5"
  - "No gemini baseline generation shipped per 12-CONTEXT.md prohibition"
metrics:
  duration: "3 minutes"
  completed: "2026-06-12"
  tasks_completed: 3
  files_modified: 5
---

# Phase 12 Plan 04: Comparison Floor Deferral Bookkeeping Summary

**One-liner:** Gemini n=5 baseline deferred as v2.2 user budget decision (D-12-09), recorded consistently across code + docs + planning; parity test confirms non-deferred comparison floor is honest n=5.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Record D-12-09 gemini deferral in _DEFERRED_BASELINE_CELLS | 9294145 | tests/unit/test_eval_matrix.py |
| 2 | Document gemini deferral symmetrically in runbook and gates docs | cce26ec | docs/baseline_regen.md, docs/eval_gates.md |
| 3 | Amend ROADMAP and REQUIREMENTS to match D-12-09 | a82461e | .planning/ROADMAP.md, .planning/REQUIREMENTS.md |

## What Was Done

This plan records the D-12-09 user decision (2026-06-11: no gemini quota/billing top-up) consistently across all bookkeeping surfaces, and verifies the v2.2 comparison floor is intact.

**Task 1** extended the `_DEFERRED_BASELINE_CELLS` gemini comment in `test_eval_matrix.py` to cite D-12-09 as a v2.2 user-decision deferral (not only the transient Phase-11 regen error D-11-11). Both deferred cells (gemini + anthropic) remain in the guard. The parity test passes: `missing == deferred` holds with `{gemini}` for the refinement matrix and `{anthropic}` for the default matrix — ANCH-03 comparison floor confirmed.

**Task 2** added symmetric Gemini deferral sections to both docs:
- `docs/baseline_regen.md`: new `### Gemini deferral (D-12-09)` failure-branch section mirroring the Anthropic deferral section, with full promotion path.
- `docs/eval_gates.md`: new `## Gemini deferral (2026-06-11)` section parallel to `## Anthropic deferral (2026-06-11)`, citing D-12-09, logged-not-gated status, and promotion path.

**Task 3** amended the planning docs:
- ROADMAP.md Phase 12: summary bullet, goal line, success criterion 4, and external-dependency note now reflect gemini deferred (D-12-09); comparison floor = matrix minus BOTH deferred cells; ANCH-02 no longer a phase-completion requirement.
- REQUIREMENTS.md: ANCH-02 marked complete as deferred-with-note (D-12-09); ANCH-03 marked complete (reinterpreted per D-12-09); traceability table updated; footer updated.

## Deviations from Plan

None — plan executed exactly as written.

## Success Criteria Verification

- ANCH-02 satisfied as deferred-with-note (D-12-09): gemini joins anthropic in deferred-cell status, logged-not-gated, with the deferral recorded in code + docs + planning. ✓
- ANCH-03 satisfied (reinterpreted per D-12-09): every non-deferred matrix cell is honest n=5 (parity test: 68 passed); both deferred cells retain their entries with notes. ✓
- No gemini-baseline generation shipped (CONTEXT prohibition honored); promotion path preserved for when budget allows. ✓
- `grep -n "D-12-09"` matches across all 5 bookkeeping files. ✓

## Self-Check: PASSED

Files created/modified exist and commits verified:
- tests/unit/test_eval_matrix.py: modified ✓
- docs/baseline_regen.md: modified ✓
- docs/eval_gates.md: modified ✓
- .planning/ROADMAP.md: modified ✓
- .planning/REQUIREMENTS.md: modified ✓
- Commit 9294145: exists ✓
- Commit cce26ec: exists ✓
- Commit a82461e: exists ✓
