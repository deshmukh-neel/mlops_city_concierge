# Phase 6: Minimal-Edit Refinement - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-02
**Phase:** 6-Minimal-Edit Refinement
**Areas discussed:** Where to inject the plan, Plan format in the prompt, Eval-harness threading parity, Byte-equal scorer + merge gate

---

## Where to inject the plan

| Option | Description | Selected |
|--------|-------------|----------|
| /chat handler + extend ConversationState | Add `committed_stops: list[Stop]` to ConversationState (round-trip via frontend opaquely, same as closure_context today). In /chat, when committed_stops is non-empty AND the message looks like a refinement (deterministic regex pre-check), prepend a HumanMessage carrying the structured plan before threading req.history. Mirrors Phase 4's hybrid intake pattern. Single injection site, backwards-compatible. | ✓ |
| Graph-layer injection (Phase 4 mirror) | Add `prior_committed_stops` to ItineraryState, mirror Phase 4's _inject_primary_type_family pattern — inject at the start of plan node. Still requires ConversationState to carry the snapshot upstream (same wire change as Option A), so the only difference is injection site. | |
| SYSTEM_PROMPT {current_plan} template | Add `{current_plan}` variable to SYSTEM_PROMPT (mirrors {current_datetime}). Computed from ConversationState.committed_stops; renders as empty/'no prior plan' on turn 1. Risk: changes SYSTEM_PROMPT on EVERY turn (P10 risk), unless gated entirely behind the flag. | |
| Always-inject (no detection) | Whenever ConversationState.committed_stops is non-empty, inject the structured plan — no regex/intent check. Relies on SYSTEM_PROMPT to instruct when to preserve vs replace. | |

**User's choice:** /chat handler + extend ConversationState
**Notes:** Captured as D-06-01/02/03 in CONTEXT.md. The detection question (always-on vs regex vs LLM) was folded into D-06-03 as "deterministic regex in input_parsing.py" mirroring Phase 4's `has_slot_structure` pattern.

---

## Plan format in the prompt

| Option | Description | Selected |
|--------|-------------|----------|
| Hybrid: prose preamble + JSON block | One HumanMessage with: short natural-language preamble ("The following stops are committed. For any stop you do not change, return the exact same place_id byte-for-byte.") + a fenced JSON block listing each stop's place_id, name, primary_type, arrival_time. Preamble carries the instruction; JSON carries the ground truth. Robust across providers (P7). | ✓ |
| Pure structured JSON field | HumanMessage content = single JSON object with `current_plan` field. Maximally machine-readable; model must infer from system prompt what to do with it. | |
| Natural-language summary only | HumanMessage content = prose summary ("Currently committed: 1) Kaiseki Yuzu (place_id ChIJ_abc) at 7pm — keep. 2) Drinks bar (place_id ChIJ_def) at 8:30pm — REPLACE."). Risk: model may paraphrase place_ids. | |
| Let Claude decide | Defer to planner / researcher. | |

**User's choice:** Hybrid: prose preamble + JSON block
**Notes:** Captured as D-06-04 in CONTEXT.md. Exact preamble wording deferred to planner per "Claude's Discretion" with the constraint that an unambiguous phrase like "byte-for-byte" or "EXACT SAME place_id" MUST appear.

---

## Eval-harness threading parity

| Option | Description | Selected |
|--------|-------------|----------|
| Opt-in threading_mode per case | Add `EvalQuery.threading_mode: Literal['legacy', 'prod'] = 'legacy'`. `evaluate_multi_turn_case` branches on it: `legacy` keeps full state.messages threading; `prod` rebuilds messages per turn from text history + the new injected structured-plan HumanMessage. Phase 6 marks `refinement_cheaper` as `threading_mode: prod`, re-baselines; late_night_closure_cascade stays `legacy`. | ✓ |
| Rewrite evaluate_multi_turn_case to match prod | Make prod-shape threading the only behavior. Honest, but invalidates `refinement_cheaper` AND `late_night_closure_cascade` baselines simultaneously; conflates signal. | |
| Accept the mismatch + add a functional test | Keep eval threading as-is. Document: `refinement_cheaper` baseline measures eval-shape; gate stays on that number. Add a new pytest functional test for the prod-shape behavior. | |
| Both: opt-in threading_mode AND functional test | Combine A + C: double coverage. Costs the most planning effort. | |

**User's choice:** Opt-in threading_mode per case
**Notes:** Captured as D-06-05/06/07 in CONTEXT.md. Re-baseline of `refinement_cheaper` lands in the same PR as the implementation, sequenced after the flag is verified locally.

---

## Byte-equal scorer + merge gate

| Option | Description | Selected |
|--------|-------------|----------|
| Scratch-cached scorer + strict 1.0 + default OFF | Scorer `refinement_minimal_edit(state) -> float` reads `state.scratch['prior_committed_stops']`. Merge gate: strict 1.0 on openai/gpt-4o-mini `refinement_cheaper` cell. REF-04 first-turn no-regression on existing scorers against `omakase_mission_open_ended`. Flag default OFF (ship dark). | ✓ |
| Two-arg scorer + delta floor + default OFF | Scorer signature `(state, prior_state) -> float`. Eval runner + baseline storage adapts to two-argument shape. Merge gate: `+0.5` delta vs pre-Phase-6 baseline. | |
| Pytest-only + no eval scorer + default ON | No new scorer. Single functional test asserts byte-equal place_ids on preserved stops. Merge gate = pytest passes. Flag default ON. | |
| Let Claude decide | Defer to planner. | |

**User's choice:** Scratch-cached scorer + strict 1.0 + default OFF
**Notes:** Captured as D-06-08/09/10 in CONTEXT.md. Per-cell env override for `eval_matrix.py` is a new field added in this phase (D-06-10).

---

## Claude's Discretion

Deferred to planner / researcher:

- Exact regex patterns for the refinement pre-check in `input_parsing.py`
- Exact JSON schema inside the structured-plan HumanMessage (minimal viable fields)
- Exact wording of the preamble (must include "byte-for-byte" or equivalent)
- Whether to add a new `RevisionReason` literal `"minimal_edit_violated"` for the post-commit critique loop
- Whether to surface a `slot_index` arg on `commit_itinerary` for explicit target marking
- Test layering: at minimum unit + functional + one integration assertion per new surface

## Deferred Ideas

- Migrate all multi-turn eval scenarios to `threading_mode: prod` (v2.1)
- New `RevisionReason` literal `"minimal_edit_violated"` for post-commit critique (revisit if eval reveals the failure mode)
- `slot_index` arg on `commit_itinerary` (belt-and-braces for v2.1)
- Flip `REFINEMENT_STRUCTURED_PLAN_ENABLED` default ON (follow-up PR after Phase 6 merges + prod observation)
- DSPy-style automated prompt tuning of the preamble (v2.1+)
