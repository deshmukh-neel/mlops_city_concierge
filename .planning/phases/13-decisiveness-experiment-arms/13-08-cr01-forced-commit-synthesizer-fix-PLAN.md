---
phase: 13-decisiveness-experiment-arms
plan: 08
type: execute
wave: 1
gap_closure: true
depends_on: []
files_modified:
  - app/agent/viability.py
  - app/agent/graph.py
  - tests/unit/test_graph_forced_commit.py
  - docs/decisiveness_arm_verdicts.md
autonomous: true
requirements: [DEC-02]
must_haves:
  truths:
    - "best_viable_candidate_per_slot returns populated dicts (with place_id, name, primary_type, similarity, source) for real PlaceHit Pydantic models on the typed path — never {} for a viable hit"
    - "The A2 forced-commit synthesizer produces commit-shaped stops carrying a non-empty rationale so commit_stops accepts them; commit_forced=True flows to final state when every slot has a viable candidate at step N"
    - "A non-mocked regression test exercises the real synthesizer (real best_viable_candidate_per_slot + real commit_stops over PlaceHit objects in scratch, DB mocked only at get_details_many) and asserts commit_forced is True with non-empty stops — and fails on the pre-fix code"
    - "The A2 section of docs/decisiveness_arm_verdicts.md is annotated that the forced mechanism was inoperative due to the synthesis bug; the 0.500 model-initiated finding stands; the forced mechanism is untested at n=5"
  artifacts:
    - path: "app/agent/viability.py"
      provides: "Typed-path PlaceHit→dict conversion via model_dump(mode=json)"
      contains: "model_dump"
    - path: "tests/unit/test_graph_forced_commit.py"
      provides: "Non-mocked synthesizer regression test on real PlaceHit shapes"
      contains: "PlaceHit"
    - path: "docs/decisiveness_arm_verdicts.md"
      provides: "A2 CR-01 annotation (mechanism inoperative; finding stands; forced untested)"
      contains: "synthesis bug"
  key_links:
    - from: "app/agent/graph.py forced-commit synthesizer"
      to: "app.agent.commit.commit_stops"
      via: "commit-shaped raw_stops with synthesized rationale"
      pattern: "rationale"
    - from: "app/agent/viability.py best_viable_candidate_per_slot typed path"
      to: "PlaceHit Pydantic model"
      via: "model_dump(mode='json') conversion"
      pattern: "model_dump"
---

<objective>
Fix CR-01: the A2 forced-commit synthesizer is dead code in production because two
independent defects make every synthesized stop empty or rejected. Repair both, then
add a regression test that exercises the REAL synthesis path on REAL PlaceHit shapes
(no mocking of best_viable_candidate_per_slot or commit_stops), and annotate the A2
verdict section so the Phase-14 entry decision rests on honest evidence.

Purpose: CR-01 is a blocker for Phase 14 (per 13-VERIFICATION.md). The A2 causal claim
("gate conditions were not satisfied") is unverifiable as shipped — forced=0 is
over-determined by the bug. The honest null result STANDS, but the A2 mechanism must be
operative (or its inoperativeness documented) before Phase 14 reads the verdict doc to
decide whether to retry A2.

Output: working forced-commit synthesizer (DEC-02), a non-mocked regression test that
fails on the old code, and an annotated A2 verdict section. No flags ship enabled by
default — A2 stays behind FORCED_COMMIT_STEP (unset/0 = off); promotion is Phase 15.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/13-decisiveness-experiment-arms/13-CONTEXT.md
@.planning/phases/13-decisiveness-experiment-arms/13-REVIEW.md
@.planning/phases/13-decisiveness-experiment-arms/13-VERIFICATION.md
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Fix the typed-path PlaceHit→dict conversion in viability.py</name>
  <files>app/agent/viability.py, tests/unit/test_viability.py</files>
  <read_first>
    - app/agent/viability.py (full file — note line 216 typed-path `hit_dict = hit if isinstance(hit, dict) else {}` and the untyped-path conversion at lines 180-184 for the contrasting behavior)
    - app/tools/retrieval.py lines 32-43 (PlaceHit model: place_id:str, name:str, primary_type:str|None, source:str, similarity:float — a Pydantic BaseModel; NO rationale field)
    - tests/unit/test_viability.py (existing test structure, fixtures, and assertion style for best_viable_candidate_per_slot)
    - .planning/phases/13-decisiveness-experiment-arms/13-REVIEW.md CR-01 section (the recommended fix uses isinstance(hit, BaseModel) → hit.model_dump(mode="json"))
  </read_first>
  <behavior>
    - Test 1: best_viable_candidate_per_slot over a typed scenario where scratch carries real PlaceHit objects (not dicts) returns entries that each contain a non-empty place_id, name, primary_type, similarity, and source — NOT {} .
    - Test 2: best_viable_candidate_per_slot still returns the same populated dicts when scratch carries plain dict hits (untyped/legacy path unchanged — byte-compatible).
    - Test 3: a hit whose shape is neither dict nor BaseModel is skipped (no {} placeholder appended for it).
    - Test 4 (regression guard on old behavior): assert the returned dict for a PlaceHit input has place_id == the PlaceHit's place_id (this assertion fails on the pre-fix `else {}` code).
  </behavior>
  <action>
    In `app/agent/viability.py`, the TYPED path of `best_viable_candidate_per_slot` (currently line 216) does `hit_dict = hit if isinstance(hit, dict) else {}`, which converts every non-dict hit to an empty dict. In production, `semantic_search` stores `PlaceHit` Pydantic models (`app/tools/retrieval.py:32`) verbatim in scratch, so every typed candidate becomes `{}` with no `place_id`. Replace this single-line conversion with a three-branch conversion that mirrors what the untyped path already attempts: if `isinstance(hit, dict)` use `dict(hit)`; elif `isinstance(hit, BaseModel)` use `hit.model_dump(mode="json")`; else `continue` (skip the unusable shape — do NOT append a `{}` placeholder). Import `BaseModel` from `pydantic` at module top. Apply the SAME `model_dump(mode="json")` conversion to the untyped path's object branch (currently lines 180-184 builds a dict via `{k: getattr(hit, k, None) for k in dir(hit) ...}` which captures bound methods and is not JSON-safe — IN-01/CR-01(a)) so both paths use the one JSON-safe conversion. The conversion must preserve at least `place_id`, `name`, `primary_type`, `similarity`, and `source` keys from the PlaceHit. Add the four behavior tests above to `tests/unit/test_viability.py` constructing real PlaceHit instances in scratch.
  </action>
  <verify>
    <automated>cd "/Users/pnhek/usf msds/msds-603-mlops/mlops_city_concierge" && poetry run pytest tests/unit/test_viability.py -x -q</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "model_dump" app/agent/viability.py` returns at least one match on the typed path region.
    - `grep -n "else {}" app/agent/viability.py` returns NO match in best_viable_candidate_per_slot (the `hit if isinstance(hit, dict) else {}` line is gone).
    - A new test in tests/unit/test_viability.py constructs a real `PlaceHit` (imported from app.tools.retrieval), puts it in scratch, calls best_viable_candidate_per_slot, and asserts the returned entry's `place_id` equals the PlaceHit's place_id (non-empty). This test FAILS when run against the pre-fix `else {}` line and PASSES after the fix.
    - The untyped/dict path tests still pass unchanged (no behavior change for dict hits).
    - `poetry run pytest tests/unit/test_viability.py -q` exits 0.
  </acceptance_criteria>
  <done>best_viable_candidate_per_slot returns populated, JSON-safe dicts for real PlaceHit models on both typed and untyped paths; a regression test pins place_id propagation and fails on the old code.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Build commit-shaped stops with rationale in the graph synthesizer + non-mocked regression test</name>
  <files>app/agent/graph.py, tests/unit/test_graph_forced_commit.py</files>
  <read_first>
    - app/agent/graph.py lines 600-640 (the forced-commit branch in critique(): `raw_stops = [c for c in candidates if c is not None]` then `committed_stops, _payload = commit_stops(state, raw_stops)`; the `if committed_stops:` gate, the critique_final_with_stops call, and the commit_forced/forced_commit_step telemetry merge)
    - app/agent/commit.py lines 45-73 (commit_stops: rejects a raw stop unless place_id is grounded in scratch AND Stop(**raw) succeeds; rejection reason "invalid stop: ..." when a required field is missing)
    - app/agent/state.py lines 190-204 (Stop model: rationale:str and source:str are REQUIRED with no default; place_id, name also required)
    - app/agent/viability.py (best_viable_candidate_per_slot output shape after Task 1: dicts with place_id, name, primary_type, similarity, source — but NO rationale)
    - tests/unit/test_graph_forced_commit.py (full file: helpers _make_mock_llm_semantic_search_only, _state_with_viable_scratch, _make_committed_stop, _run_graph_sync, _VALID_PLACE_ID; and test_forced_commit_triggers_at_step_n lines 220-296 which mocks best_viable_candidate_per_slot AND commit_stops — the masking test)
    - app/tools/retrieval.py lines 32-43 (PlaceHit fields — for building real PlaceHit objects in the new test's scratch)
  </read_first>
  <action>
    In `app/agent/graph.py`, the forced-commit synthesizer (lines 619-639) currently passes the raw candidate dicts straight to `commit_stops` via `raw_stops = [c for c in candidates if c is not None]`. Even after Task 1 those dicts lack a `rationale` (PlaceHit has none) and `Stop(**raw)` rejects them because `Stop.rationale` is required with no default (`app/agent/state.py:198`). Replace the `raw_stops` construction with one that builds an explicit commit-shaped dict per candidate carrying: `place_id` (from candidate), `name` (candidate name or empty string), `primary_type` (candidate primary_type), `source` (candidate source or "google_places"), and a synthesized `rationale` string that names the forced-commit provenance and the cosine — e.g. a sentence stating it is the best available match for the requested slot at the candidate's similarity. Only include candidates that are not None AND have a truthy `place_id` (skip None/place_id-less entries — WR-07 admission consistency). Keep the rest of the branch (commit_stops call, the `if committed_stops:` gate, critique_final_with_stops on the state.model_copy, and the commit_forced=True / forced_commit_step telemetry merge) unchanged. Also remove the no-op `"step_count": state.step_count` from the model_copy update dict (IN-04 — it writes the value it already has).

    THEN add a NON-MOCKED regression test to `tests/unit/test_graph_forced_commit.py` named test_forced_commit_synthesizer_real_placehit_shapes (or similar). It must: build a state whose scratch["semantic_search"] result list contains REAL PlaceHit objects (imported from app.tools.retrieval) — at least one viable PlaceHit per requested primary_type, each with similarity >= LOW_SIMILARITY_THRESHOLD and a grounded place_id; set FORCED_COMMIT_STEP so the gate fires; NOT patch best_viable_candidate_per_slot and NOT patch commit_stops (exercise the real synthesis path end-to-end); mock ONLY the DB boundary (get_details_many — patch where commit.py imports it — so enrichment does not hit a real DB) and critique_final_with_stops / route_legs / swap_closed_stops as the existing tests do to avoid live DB/LLM; assert the final state has commit_forced is True and stops is non-empty. This test MUST fail on the pre-fix synthesizer (empty/rejected stops → commit_forced stays False).
  </action>
  <verify>
    <automated>cd "/Users/pnhek/usf msds/msds-603-mlops/mlops_city_concierge" && poetry run pytest tests/unit/test_graph_forced_commit.py -x -q</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "rationale" app/agent/graph.py` shows a synthesized rationale string built in the forced-commit branch's raw_stops construction.
    - `grep -n "step_count.*state.step_count" app/agent/graph.py` no longer matches inside the forced-commit model_copy update (IN-04 removed).
    - A new test in tests/unit/test_graph_forced_commit.py imports PlaceHit from app.tools.retrieval, puts real PlaceHit objects in scratch, does NOT patch best_viable_candidate_per_slot or commit_stops, and asserts final commit_forced is True and stops non-empty.
    - The new test FAILS when run against the pre-fix graph.py synthesizer (demonstrable: stops empty / commit_forced False) and PASSES after the fix.
    - The pre-existing test_forced_commit_triggers_at_step_n still passes (mocked path unchanged).
    - `poetry run pytest tests/unit/test_graph_forced_commit.py -q` exits 0.
  </acceptance_criteria>
  <done>The synthesizer builds commit-shaped stops with rationale that commit_stops accepts; a non-mocked test pins commit_forced=True on real PlaceHit shapes and fails on the old code, so the CR-01 bug class cannot recur silently.</done>
</task>

<task type="auto">
  <name>Task 3: Annotate the A2 verdict section with the CR-01 finding</name>
  <files>docs/decisiveness_arm_verdicts.md</files>
  <read_first>
    - docs/decisiveness_arm_verdicts.md lines 116-202 (the full A2 section: Honesty Contract, Per-model results, the "Key finding — FORCED_COMMIT_STEP=6 mechanism NEVER FIRED" paragraph at 180-189, and the A2 Closing verdict at 191-200 — note the existing claim "the mechanism's viability gate was not satisfied")
    - .planning/phases/13-decisiveness-experiment-arms/13-VERIFICATION.md (CR-01 required_action_before_phase_14: annotation must state mechanism was inoperative due to synthesis bug; 0.500 model-initiated finding stands; forced mechanism untested at n=5)
    - .planning/phases/13-decisiveness-experiment-arms/13-REVIEW.md WR-10 (the A2 causal claim is unsupported; decide explicitly whether to reserve a Phase-14 A2 retry)
  </read_first>
  <action>
    Add a clearly-labeled annotation block to the A2 section of `docs/decisiveness_arm_verdicts.md` (immediately after the "Key finding — FORCED_COMMIT_STEP=6 mechanism NEVER FIRED" paragraph, lines ~180-189, and reflected in the A2 Closing verdict). The annotation MUST state, in plain language, all four points: (a) the forced-commit synthesizer was INOPERATIVE in the n=5 A2 run due to a synthesis bug (CR-01: viability.py typed path discarded PlaceHit models to {}, and synthesized stops lacked the required rationale field so commit_stops rejected them) — so forced=0 is explained by the bug, not solely by gate non-satisfaction; (b) the 0.500 gpt-5-mini result STANDS as an entirely MODEL-INITIATED commit rate (the bug only affected the forced path, never the model's own commits); (c) the forced mechanism is UNTESTED at n=5 — its effect on commit rate is unknown until a re-run on the fixed synthesizer; (d) state explicitly whether Phase 14 reserves a slot to re-test A2 with the working synthesizer (recommend: A2 re-test is a Phase-14/15 candidate, NOT a Phase-13 re-run, because the D-13-02 four-run live cap is already consumed). Do NOT soften or invalidate the honest null result ("No arm cleared the INST-05 falsifier bar") — the annotation qualifies the A2 causal explanation only. Update the misleading sentence "the mechanism's viability gate was not satisfied" to "forced=0 is over-determined by the CR-01 synthesis bug; whether the gate would have been satisfied on the fixed synthesizer is unknown."
  </action>
  <verify>
    <automated>cd "/Users/pnhek/usf msds/msds-603-mlops/mlops_city_concierge" && grep -qi "synthesis bug" docs/decisiveness_arm_verdicts.md && grep -qi "untested at n=5" docs/decisiveness_arm_verdicts.md && grep -qi "model-initiated" docs/decisiveness_arm_verdicts.md && grep -qi "No arm cleared" docs/decisiveness_arm_verdicts.md && echo OK</automated>
  </verify>
  <acceptance_criteria>
    - The A2 section contains an annotation that uses the phrase "synthesis bug" and references CR-01.
    - The annotation states the 0.500 result stands as model-initiated (phrase "model-initiated" present in the A2 annotation context).
    - The annotation states the forced mechanism is "untested at n=5".
    - The annotation states whether Phase 14 reserves an A2 retry slot (explicit yes/no with rationale).
    - The honest null line "No arm cleared the INST-05 falsifier bar" is still present and unaltered in the Closing Verdict section (`grep -c "No arm cleared" docs/decisiveness_arm_verdicts.md` >= 1).
    - The Phase-7 grep gate is untouched (no prompt edits): `poetry run pytest tests/unit/test_critique_checks.py::test_prompt_02_grep_gate_no_behavioral_phrases_in_prompts -q` exits 0.
  </acceptance_criteria>
  <done>The A2 verdict section honestly records that the forced mechanism was inoperative due to CR-01, preserves the model-initiated 0.500 finding, flags the forced mechanism as untested at n=5, and states the Phase-14 A2-retry disposition — without softening the null result.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| (none new) | This plan repairs an internal arm mechanism behind an off-by-default env flag (FORCED_COMMIT_STEP). No new external input, network, or auth surface is introduced. The synthesizer routes through the EXISTING commit_stops grounding check (place_id must be seen via a prior tool result), so synthesized stops cannot inject ungrounded place_ids. |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-13-08-01 | Tampering | forced synthesizer injects an ungrounded place_id into a committed plan | accept | commit_stops already rejects any place_id not grounded in scratch (`app/agent/commit.py:62`); the synthesizer reuses scratch hits only — no new path bypasses grounding |
| T-13-08-02 | Repudiation | A2 verdict causal claim misleads the Phase-14 entry decision | mitigate | Task 3 annotation records the bug's effect explicitly; honest null result preserved |
| T-13-08-SC | Tampering | npm/pip/cargo installs | mitigate | No package installs in this plan; slopcheck N/A — no new dependencies added |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_viability.py tests/unit/test_graph_forced_commit.py -q` exits 0, including the two new non-mocked regression tests.
- The Phase-7 grep gate stays green: `poetry run pytest tests/unit/test_critique_checks.py::test_prompt_02_grep_gate_no_behavioral_phrases_in_prompts -q` exits 0.
- Full suite mandatory before merge (DB-pool contamination risk with real-graph tests): `make test` passes.
- A2 verdict section annotated; honest null result intact.
- No flags enabled by default (FORCED_COMMIT_STEP unset/0 = off); no baselines written (`git status configs/eval_baselines/` clean).
</verification>

<success_criteria>
- CR-01 closed: both defects fixed (typed-path conversion + synthesized rationale); the synthesizer produces accepted commit stops on real PlaceHit shapes.
- The bug class cannot recur silently: a non-mocked regression test asserts commit_forced=True on real PlaceHit objects and fails on the pre-fix code.
- The A2 verdict is honest: mechanism inoperative due to synthesis bug, 0.500 model-initiated finding stands, forced mechanism untested at n=5, Phase-14 retry disposition stated.
</success_criteria>

<output>
Create `.planning/phases/13-decisiveness-experiment-arms/13-08-SUMMARY.md` when done.
</output>
