# Phase 5: Rationale-Stop Alignment Fix - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-27
**Phase:** 05-rationale-stop-alignment-fix
**Areas discussed:** None (user fast-tracked to plan)

---

## Discussion mode

| Option | Description | Selected |
|--------|-------------|----------|
| Skip discussion, go to plan | Roadmap was right — well-scoped. Pick deterministic replacement and overwrite placeholder before `summarize_stops`. Jump to `/gsd:plan-phase 5`. | ✓ |
| Where to fix it | Choose between swap.py:238, commit.py::enrich_stops_with_booking, or summarize_stops as the mutation point. | |
| Replacement content | Deterministic template / per-candidate LLM / inherit closed stop's rationale / empty. | |
| Eval gating | Reuse Phase 4's gate / add new dedicated swap scenario / finally gate `late_night_closure_cascade`. | |

**User's choice:** Skip discussion, go to plan.

**Notes:**
- ROADMAP.md Phase 5 entry explicitly says `Discuss-phase: Not needed` — user's selection confirms that call.
- The pre-question codebase scout had already located the placeholder source
  (`app/agent/swap.py:238`), the bleed path (`enrich_stops_with_booking` does
  not touch `rationale`; `summarize_stops` renders it as-is), and the existing
  scorer that already catches the bug (`rationale_stop_alignment` docstring at
  `app/agent/critique/checks.py:336-341`). The recommended path
  (inherit closed stop's rationale, with `None` as the fallback) and a
  regression test shape were captured in CONTEXT.md so the planner can act
  without re-discovering them.
- The three areas the user declined to discuss are still surfaced in
  CONTEXT.md as Claude's-discretion items for the planner.

---

## Claude's Discretion

Captured in `05-CONTEXT.md` § Claude's Discretion. Summary:
- Exact mutation site (swap.py / commit.py / summarize_stops) — planner picks
  based on test blast radius; D-05-02 names the inherit-from-closed-stop
  approach as the recommended starting point.
- Regression test shape (unit + functional minimum per
  `feedback_test_layering.md`).
- Whether to add a defensive `rationale_stop_alignment` runtime guard
  post-swap (nice-to-have, not required).

## Deferred Ideas

Captured in `05-CONTEXT.md` § Deferred:
- Closure-cascade gating decision (D-04-12 follow-up) — punted forward.
- `rationale_stop_alignment` metric redefinition (Phase 4 flag from D-04-14).
- LLM-generated swap rationales — rejected for latency.
- Phase 6 (REF-01..04) — out of scope.
