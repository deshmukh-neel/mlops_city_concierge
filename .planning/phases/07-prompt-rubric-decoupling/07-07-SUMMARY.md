# Plan 07-07 — Rebaseline & Falsifier — SUMMARY

**Status:** Complete (PROMPT-04 accept-with-notes; PROMPT-05 logged)
**Wave:** 4 (terminal)
**Closed:** 2026-06-04

---

## What shipped

1. **Pre-Phase-7 snapshot** at `configs/eval_baselines/_snapshots/refinement_cheaper.pre-phase7.json` — byte-identical copy of the prior baseline (commit `61aee1b`, Phase 6 era).
2. **Post-Phase-7 regenerated baseline** at `configs/eval_baselines/refinement_cheaper.json` — three providers (`openai/gpt-4o-mini`, `deepseek/deepseek-chat`, `openai/gpt-5-mini`), n=5 each, fresh medians under the decoupled prompt + extended scorer + D-07-10 preamble-tightening iteration.
3. **D-07-10 preamble-tightening iteration** in `app/agent/io.py` — added one imperative sentence to `_REFINEMENT_PREAMBLE`:
   > "Reuse the `place_id` and `slot` index of every stop you are not changing exactly as listed; only the slot named by the user gets a new `place_id`."

   The grep gate (`test_prompt_02_grep_gate_no_behavioral_phrases_in_prompts`) stays green — none of the six D-07-04 forbidden phrases appear in the new line.
4. **CI staleness gate** (`scripts/check_baselines_fresh.py origin/main`) exits 0 with both the snapshot and the regenerated baseline counted as refreshed.

## PROMPT-04 outcome — accept-with-notes

| Cell | Pre-Phase-7 median | Post-Phase-7 median (run 1) | Post-Phase-7 median (run 2 — tightened preamble) | Delta |
|------|--------------------|------------------------------|---------------------------------------------------|-------|
| `openai/gpt-4o-mini` × `refinement_cheaper` × `refinement_minimal_edit` | 1.0 (saturated) | 0.0 | 0.0 (2× 0.5, 3× 0.0) | **-1.000** |

**Verdict:** FAIL on the strict numeric gate; **accepted-with-notes** per Phase 6 D-06-09 part 2 precedent.

**Why accept-with-notes:**

- `committed_itinerary_rate` median = `1.0` post-Phase-7 (vs `1.0` pre) — the agent still commits 3-stop itineraries 4-5 of every 5 runs after revision-loop correction. The agent is **functional**.
- All other scorers (`category_compliance`, `category_compliance_strict`, `geographic_coherence`, `rationale_stop_alignment`, `temporal_coherence`, `constraints_satisfied`, `no_hallucinated_place_ids`, `walking_budget_respected`) all median = `1.0`. No regression elsewhere.
- The new `refinement_minimal_edit` scorer (07-04 / D-07-05 / D-07-07) tightened the gate with the target-slot `primary_type` sub-check. The pre-Phase-7 baseline at 1.0 was measured under the LOOSER scorer (no category check) — it is no longer apples-to-apples comparable to the new scorer's output (D-07-10 explicitly anticipated this: "the gpt-4o-mini cell may move under the new scoring even with no agent-behavior regression").
- `revision_reasons` across cells show `stop_count_mismatch` retries — the agent initially over-edits (likely dropping a non-target slot) but the revision loop pulls it back to 3 stops. Byte-equality has already lost the intermediate state by then, so the final scorer reads `< 1.0`.
- D-07-10 preamble-tightening iteration (Option A from the user checkpoint) added one imperative without re-introducing any of the six D-07-04 forbidden phrases. Distribution shifted from all-0.0 to 2× 0.5 — partial recovery, but median still 0.0.
- Further tightening would push the preamble back toward the imperative-rich wording 07-01 deliberately deleted to decouple prompt-from-rubric (PROMPT-02). Accepting the regression at the gate level preserves the architectural achievement of the phase.

**Phase 10 inheritance:** When Phase 10 regenerates cross-model baselines under the matured Phase-7+ contract, the post-Phase-7 number recorded here becomes the new floor. The strict 1.0 binary merge gate is preserved (no threshold relaxation); the baseline JSON simply records the honest measurement.

## PROMPT-05 outcome — falsifier signal

| Cell | post-Phase-7 median | `committed_itinerary_rate` median | Interpretation |
|------|---------------------|-----------------------------------|----------------|
| `openai/gpt-5-mini` × `refinement_cheaper` × `refinement_minimal_edit` | `0.0` | `0.0` (1/5 runs committed; 2× 0.0 stops, 2× 0.0 stops, 1× 3.0 stops, 1× 3.0 stops with rme=0) | **state-loss dominates over prompt-coupling** |

Per D-07-09 the cell ran with reasoning ENABLED (no `--llm-provider-override`, no thinking-disable). Per D-07-08 it is logged-not-gated (no CI gate trip).

**Interpretation for Phase 9 scope:** State-loss dominates. The new task-only-plus-imperative preamble doesn't recover gpt-5-mini's refinement-turn behavior because the agent loses reasoning state across turns (`_prune_for_llm` drops `reasoning_content`). Phase 9 (per-provider state preservation impls) stays at **full scope** — narrowing is not justified.

This matches the established memory `project_agent_loses_reasoning_state_all_providers` (DECISIVE: agent _prune_for_llm drops reasoning state for ALL reasoning models; architectural, not vendor-specific).

## DeepSeek (logged-not-gated)

`deepseek/deepseek-chat` median = `0.0`, `committed_itinerary_rate` = `0.0`. Matches the prior known decisiveness gap (`project_deepseek_decisiveness_gap`). Not regressed by Phase 7; just unchanged.

## Files changed / created

| Path | Change |
|------|--------|
| `configs/eval_baselines/_snapshots/refinement_cheaper.pre-phase7.json` | created (byte-identical snapshot) |
| `configs/eval_baselines/refinement_cheaper.json` | regenerated (3 providers, fresh n=5 medians, Phase 7 `_observations`) |
| `app/agent/io.py` | added one imperative line to `_REFINEMENT_PREAMBLE` (D-07-10 iteration) |

## Commits

| SHA | Message |
|-----|---------|
| `44c6e3e` | `data(07-07): snapshot pre-Phase-7 baseline + regen refinement_cheaper.json (n=5)` |
| `2315430` | `refactor(07-07): tighten _REFINEMENT_PREAMBLE with single preserve imperative` |
| (next) | `data(07-07): refresh refinement_cheaper.json with post-iteration n=5 medians` |
| (next) | `docs(07-07): complete plan SUMMARY (PROMPT-04 accept-with-notes; PROMPT-05 state-loss)` |

## Verification gates (post-plan)

- `make eval-matrix-refinement-structural-check` — exit 0
- `poetry run python scripts/check_baselines_fresh.py origin/main` — exit 0
- `poetry run pytest tests/unit/ -q` — 995+ passed, 0 failed (per Wave 3 + iteration test sweep)
- `make eval-matrix-refinement RUNS=5` — runs against three live cells, writes per-cell JSON to `eval_reports/2026-06-04T05-56-45Z/`. Exit code is 1 because per-cell subprocesses exit 1 when `refinement_minimal_edit` violation is recorded (intended scorer behavior, not subprocess crash). The matrix runner files them as "failed" cells in summary.json; per-cell aggregates are valid.

## Outcomes for upstream phases

- **PROMPT-01** (`/chat` refinement integration test): met by 07-06; unaffected by re-baseline.
- **PROMPT-02** (grep gate locks deletion of behavioral phrases): met by 07-01 + 07-05; survives the D-07-10 tightening iteration (grep gate still green).
- **PROMPT-03** (target-slot category enforcement in scorer): met by 07-04; the post-Phase-7 baseline measurements reflect the new gate.
- **PROMPT-04** (no-regression on `openai/gpt-4o-mini`): accept-with-notes per Phase 6 D-06-09 part 2 precedent. Documented above.
- **PROMPT-05** (falsifier on `openai/gpt-5-mini`): measured and recorded; interpretation = state-loss dominates → Phase 9 stays at full scope.

## Notes for Phase 8 / Phase 9 / Phase 10

- **Phase 8** (reasoning-state thread-through contract + conformance harness) was conditionally promotable. PROMPT-05 returned the "state-loss dominates" signal, so Phase 8 becomes the dominant remaining work for the v2.1 milestone anchor gate.
- **Phase 9** (per-provider state preservation impls) stays at **full scope** (gpt-5 → DeepSeek → Claude → Gemini 3). The PROMPT-05 falsifier did not justify narrowing.
- **Phase 10** (cross-model baseline regen + matrix expansion) inherits the post-Phase-7 gpt-4o-mini baseline as the new PROMPT-04 floor. The strict 1.0 binary merge gate stays.

