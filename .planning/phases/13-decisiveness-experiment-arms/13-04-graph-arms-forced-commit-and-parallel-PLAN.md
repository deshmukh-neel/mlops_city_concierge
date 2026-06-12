---
phase: 13-decisiveness-experiment-arms
plan: 04
type: execute
wave: 2
depends_on: ["13-01", "13-02"]
files_modified:
  - app/agent/graph.py
  - tests/unit/test_graph_forced_commit.py
  - tests/unit/test_graph_parallel_tools.py
autonomous: true
requirements: [DEC-01, DEC-02, DEC-04]
must_haves:
  truths:
    - "Three arm flags are read once at graph-build time (FORCED_COMMIT_STEP int, VIABILITY_CONTRACT_ENABLED bool, PARALLEL_TOOL_EXECUTION_ENABLED bool) and closed over the graph's inner functions"
    - "A2: at step N (FORCED_COMMIT_STEP, default 6 when on, unset/0 = off), if the model has not committed AND every requested stop has a viable candidate, the graph synthesizes a commit_itinerary from best-so-far candidates and routes it through the NORMAL commit path; if any slot lacks a viable candidate it does NOT force and falls through to max-steps"
    - "A2 sets state.commit_forced=True and forced_commit_step=N when it fires; the forced commit produces a real committed itinerary (place_id validation + critique_final_with_stops + finalize-on-commit), distinct from short_circuit_max_steps which finalizes WITHOUT a commit"
    - "A3: when PARALLEL_TOOL_EXECUTION_ENABLED, all tool calls within one act() step run concurrently via asyncio.gather with results appended in ORIGINAL tool_call order; the commit branch still short-circuits correctly; INST-04 tool_execution timing captures the reduction"
    - "A1 wiring: when VIABILITY_CONTRACT_ENABLED, the rule-8 viability addendum (from plan 13-02) is appended to the system prompt at graph-build time"
    - "With all three flags off, graph behavior is byte-identical to the current sequential, non-forced, base-prompt path"
  artifacts:
    - path: "app/agent/graph.py"
      provides: "Arm-flag reads + A2 forced-commit branch + A3 parallel act() + A1 prompt wiring"
      contains: "FORCED_COMMIT_STEP"
    - path: "tests/unit/test_graph_forced_commit.py"
      provides: "D-13-04 required test: mock model that never commits triggers the forced commit"
      contains: "FORCED_COMMIT_STEP"
    - path: "tests/unit/test_graph_parallel_tools.py"
      provides: "A3 order-stability + commit-branch interaction tests"
      contains: "PARALLEL_TOOL_EXECUTION_ENABLED"
  key_links:
    - from: "app/agent/graph.py critique() forced-commit branch"
      to: "app.agent.viability.all_slots_viable + best_viable_candidate_per_slot"
      via: "viability gate + best-so-far candidate selection for the synthetic commit"
      pattern: "all_slots_viable"
    - from: "app/agent/graph.py forced commit"
      to: "app.agent.commit.commit_stops + app.agent.revision.critique_final_with_stops"
      via: "synthetic commit routed through the normal commit path"
      pattern: "commit_stops"
    - from: "app/agent/graph.py plan()"
      to: "app.agent.prompts.rule8_viability_addendum"
      via: "flag-gated additive prompt addendum appended at graph-build time"
      pattern: "rule8_viability_addendum"
---

<objective>
Wire all three graph-level arms into app/agent/graph.py: A2 (forced-commit-at-
step-N), A3 (parallel tool execution in act()), and A1's prompt-injection point
(append the viability addendum built in plan 13-02). All three read env flags
ONCE at graph-build time and close them over the inner functions, so each eval
run with a given env produces a graph with the right arm behavior. With all flags
off, the path is byte-identical to today.

Purpose: D-13-03 (forced-commit mechanism + viability gate + best-so-far
synthesis through the normal commit path), D-13-04 (commit_forced/forced_commit_step
honesty telemetry + the required "mock never commits → forced commit fires" test),
D-13-08 (concurrency inside one act() step, order-stable results, commit-branch
care), D-13-06 (additive prompt wiring). This is the only plan that touches
graph.py, so it owns that file exclusively (no Wave-2 file overlap).

Output: arm-flag reads + A2 branch + A3 gather + A1 addendum wiring in graph.py,
plus two new graph unit-test files.
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
@.planning/phases/13-decisiveness-experiment-arms/13-PATTERNS.md
@.planning/phases/13-decisiveness-experiment-arms/13-01-SUMMARY.md
@.planning/phases/13-decisiveness-experiment-arms/13-02-SUMMARY.md
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Read arm flags at graph-build time and wire A1 prompt addendum</name>
  <files>app/agent/graph.py</files>
  <read_first>
    - app/agent/graph.py lines 251-360 (build_agent_graph signature + the plan() closure where the system prompt is assembled at lines 286-296; adapter/judge_llm graph-build-time resolution pattern at 273-282)
    - app/main.py lines 753-763 (truthy-set env-flag parsing precedent — copy verbatim for the two bool flags; FORCED_COMMIT_STEP is int(os.environ.get("FORCED_COMMIT_STEP","0") or "0"))
    - app/agent/prompts.py rule8_viability_addendum (the additive builder from plan 13-02 — call with the VIABILITY_CONTRACT_ENABLED bool, append after SYSTEM_PROMPT.format(...))
    - .planning/phases/13-decisiveness-experiment-arms/13-PATTERNS.md "app/agent/graph.py — A2 forced-commit branch + A3 parallel tool execution" (env-flag reading pattern: read inside build_agent_graph before the inner defs) and "app/agent/prompts.py" flag-conditional prompt pattern
  </read_first>
  <behavior>
    - build_agent_graph reads FORCED_COMMIT_STEP (int, default 0), VIABILITY_CONTRACT_ENABLED (bool), PARALLEL_TOOL_EXECUTION_ENABLED (bool) once, before defining plan/act/critique.
    - With VIABILITY_CONTRACT_ENABLED set, plan() appends rule8_viability_addendum(True) to the system message; with it unset, the system message is byte-identical to today.
  </behavior>
  <action>
    In `app/agent/graph.py` `build_agent_graph`, before the inner function defs (alongside the adapter/judge_llm resolution at ~273-282), read the three arm flags into local closure variables: `_forced_commit_step = int(os.environ.get("FORCED_COMMIT_STEP", "0") or "0")`, `_viability_contract_enabled` (truthy-set parse of VIABILITY_CONTRACT_ENABLED), `_parallel_tool_execution_enabled` (truthy-set parse of PARALLEL_TOOL_EXECUTION_ENABLED). Add `import os` if not present. In `plan()`, where the SystemMessage is built (lines 286-296), append `rule8_viability_addendum(_viability_contract_enabled)` (imported from app.agent.prompts) to the system prompt string AFTER `SYSTEM_PROMPT.format(...) + _constraints_context(state)`. Keep flag-off byte-identity: when `_viability_contract_enabled` is False the addendum returns "" so the prompt is unchanged. This task is wiring only — the A2 branch and A3 gather land in Tasks 2 and 3.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_graph_forced_commit.py -v -k "flag_reads or prompt_addendum"</automated>
  </verify>
  <acceptance_criteria>
    - build_agent_graph resolves the three flags once at build time (greppable: `FORCED_COMMIT_STEP`, `VIABILITY_CONTRACT_ENABLED`, `PARALLEL_TOOL_EXECUTION_ENABLED` in graph.py).
    - A test builds the graph with VIABILITY_CONTRACT_ENABLED=1 and asserts the assembled system prompt contains "cosine similarity"; with it unset, asserts it does not (flag-off byte-identity).
    - `make typecheck` passes.
  </acceptance_criteria>
  <done>Arm flags resolve at graph-build time; A1 prompt addendum is wired and flag-gated; flag-off prompt is byte-identical.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: A2 forced-commit-at-step-N branch through the normal commit path</name>
  <files>app/agent/graph.py, tests/unit/test_graph_forced_commit.py</files>
  <read_first>
    - app/agent/graph.py lines 361-487 (act() — the commit branch at 374-390, the update dict at 459-487; the forced-commit fields commit_forced/forced_commit_step extend this update)
    - app/agent/graph.py lines 489-522 (critique() — the max_steps short-circuit at 496 is where the A2 branch slots in BEFORE; committed_this_step detection at 511-514 shows how to read commit scratch)
    - app/agent/commit.py lines 45-73 (commit_stops — the normal commit path the synthetic commit routes through; place_id validation against grounded scratch)
    - app/agent/revision.py lines 314-391 (short_circuit_max_steps — finalizes WITHOUT a commit/caveat; critique_final_with_stops — the post-commit gauntlet the forced commit must run through)
    - app/agent/viability.py (all_slots_viable + best_viable_candidate_per_slot from plan 13-01)
    - app/agent/state.py commit_forced/forced_commit_step fields (from plan 13-01)
    - .planning/phases/13-decisiveness-experiment-arms/13-PATTERNS.md "A2: forced-commit branch in critique()" + "tests/unit/test_graph_forced_commit.py" (mock-model test pattern + state fixtures)
  </read_first>
  <behavior>
    - With FORCED_COMMIT_STEP=N (>0) and a model that never calls commit_itinerary: when state.step_count reaches N AND all_slots_viable(state) is True AND no commit has happened, the graph synthesizes a commit_itinerary from best_viable_candidate_per_slot, routes it through commit_stops, sets commit_forced=True and forced_commit_step=N, runs critique_final_with_stops, and finalizes.
    - If at step N any slot lacks a viable candidate, the forced commit does NOT fire: the graph falls through to the existing max-steps path and commit_forced stays False (the skip is observable: forced_commit_step is None).
    - The synthesized commit only uses place_ids already grounded in scratch (commit_stops rejects ungrounded ids — best_viable_candidate_per_slot returns hits seen via tool results, so they validate).
    - FORCED_COMMIT_STEP unset/0 disables the branch entirely — byte-identical to today.
    - REQUIRED (D-13-04): a mock LLM that always emits a non-commit tool call triggers the forced commit and ends the run with state.commit_forced True.
  </behavior>
  <action>
    Add the A2 forced-commit branch to `critique()` in `app/agent/graph.py`, placed BEFORE the `if state.step_count >= max_steps` check (line 496). Fire when `_forced_commit_step > 0 AND state.step_count >= _forced_commit_step AND not state.stops AND all_slots_viable(state, LOW_SIMILARITY_THRESHOLD)`. When firing: build a synthetic stops list via `best_viable_candidate_per_slot(state, LOW_SIMILARITY_THRESHOLD)` (each entry must carry the grounded place_id, name, rationale, source — minimum fields commit_stops + Stop require), call `commit_stops(state, synthetic_stops)` to validate + enrich, capturing the returned `committed_stops`. CRITICAL state-threading: `critique()` returns an update dict and does NOT mutate `state` in place, so `state.stops` is still empty at this point — pass `state.model_copy(update={"stops": committed_stops})` as the state argument to `critique_final_with_stops(...)` so `itinerary_violations` sees the newly committed plan (calling it with the un-updated `state` would make violations vacuously empty and run the post-commit gauntlet on uncommitted state). Merge any revision hints returned by that call into the final update dict along with `commit_forced=True`, `forced_commit_step=state.step_count`, and the committed `stops`. If `all_slots_viable` is False at step N, do NOT force — fall through to the existing `short_circuit_max_steps` path (which finalizes WITHOUT a commit), leaving commit_forced False. Import `all_slots_viable` + `best_viable_candidate_per_slot` from `app.agent.viability` and `LOW_SIMILARITY_THRESHOLD` from `app.agent.revision`. Keep the default `FORCED_COMMIT_STEP` for the arm at 6 (documented in a comment; max_steps=8, so step 6 leaves headroom) — but the code reads whatever env value is set; do NOT hardcode 6 in the firing condition. CRITICAL: the synthetic stops must reuse the JSON-safe candidate dicts from viability (no Pydantic in tool-call-style args). Write `tests/unit/test_graph_forced_commit.py` with the D-13-04 required test (mock model never commits → forced commit fires at N) plus a no-viable-slot test (forced commit skipped, commit_forced stays False).
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_graph_forced_commit.py -v</automated>
  </verify>
  <acceptance_criteria>
    - test_graph_forced_commit.py has a test where a mock LLM emitting only semantic_search calls triggers the forced commit at FORCED_COMMIT_STEP and the final state has commit_forced True and forced_commit_step == the configured step.
    - A test asserts that when a slot has no viable candidate at step N, the forced commit does NOT fire (commit_forced False, forced_commit_step None) and the run finalizes via the max-steps path.
    - A test asserts FORCED_COMMIT_STEP unset leaves commit_forced False and the run behaves as today.
    - The forced commit produces a non-empty state.stops (real committed itinerary), distinct from short_circuit_max_steps' caveat-only finalize.
    - `grep -n "6" app/agent/graph.py` shows no hardcoded 6 in the firing condition (env-driven).
  </acceptance_criteria>
  <done>A2 forced-commit fires model-independently at step N when every slot is viable, routes through the normal commit path, sets honesty telemetry, and skips when any slot lacks a viable candidate.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: A3 parallel tool execution in act() with order-stable results</name>
  <files>app/agent/graph.py, tests/unit/test_graph_parallel_tools.py</files>
  <read_first>
    - app/agent/graph.py lines 361-487 (act() — the sequential `for tc in ai.tool_calls:` loop at 374-451, the commit short-circuit at 375-390, the scratch_updates assembly, the _tool_start/_tool_elapsed INST-04 timing at 371/453, and the telemetry patch at 467-486)
    - .planning/phases/13-decisiveness-experiment-arms/13-PATTERNS.md "A3: parallel tool execution in act()" (gather-with-index pattern; _exec_one returns a tuple; re-assemble in ORIGINAL order) and "Asyncio.gather ordering guarantee" (gather preserves input order)
    - app/agent/commit.py commit_stops (the commit branch must stay correct under parallelism — D-13-08: commit short-circuits the step today)
  </read_first>
  <behavior>
    - With PARALLEL_TOOL_EXECUTION_ENABLED set: all tool calls in one act() step run concurrently (asyncio.gather); ToolMessages and scratch entries are appended in the ORIGINAL tool_call order regardless of completion order; tool_calls_this_step and INST-04 tool_exec_seconds remain correct.
    - The commit_itinerary branch still produces committed_stops correctly (commit is handled in the per-call body, order-preserved).
    - With the flag unset: the sequential loop runs unchanged — byte-identical results and scratch ordering to today.
    - A test with two tool calls whose mocked execution completes out of order asserts the appended results are in original tool_call order.
  </behavior>
  <action>
    Refactor the per-tool-call body of `act()` in `app/agent/graph.py` into an inner async helper `_exec_one(tc)` returning a tuple `(ToolMessage, scratch_name | None, scratch_entry | None, committed_stops | None, was_commit: bool)` — preserving ALL existing per-call logic (commit branch, unknown-tool branch, closure/primary_type_family injection, asyncio.to_thread tool.invoke). When `_parallel_tool_execution_enabled` is True, run `results = await asyncio.gather(*[_exec_one(tc) for tc in ai.tool_calls])` and re-assemble new_messages/scratch_updates/committed_stops by iterating `results` IN ORDER (gather preserves input order — no sort needed, D-13-08). When False, keep the existing sequential loop EXACTLY as-is (do not route the off path through _exec_one if that risks any behavior change — prefer an explicit `if/else` with the sequential branch untouched). Preserve the INST-04 `_tool_start`/`_tool_elapsed` timing wrapping the whole step and the telemetry-patch block (lines 467-486) unchanged. CRITICAL (D-13-08): the commit branch must still set committed_stops; scratch entries must keep their `step`/`id` keys for the viability + rule8 readers; results MUST be appended in original order. Write `tests/unit/test_graph_parallel_tools.py` asserting order-stability with out-of-order mock completion and correct commit handling under the parallel path.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_graph_parallel_tools.py -v</automated>
  </verify>
  <acceptance_criteria>
    - A test with two non-commit tool calls whose mocks resolve out of completion order asserts new_messages tool results and scratch entries are in ORIGINAL tool_call order under PARALLEL_TOOL_EXECUTION_ENABLED=1.
    - A test asserts the commit_itinerary branch under the parallel path still yields committed stops.
    - A test asserts that with the flag unset, the act() result (messages, scratch, telemetry) is identical to a sequential reference run.
    - INST-04 tool_exec_seconds is still recorded in step_telemetry on the parallel path.
  </acceptance_criteria>
  <done>A3 runs tool calls in one act() step concurrently with order-stable results and correct commit handling; flag-off is byte-identical sequential behavior; INST-04 timing intact.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| arm env flags → graph behavior | Operator-controlled in eval; never set by this phase in prod (default OFF) |
| model tool-call args → concurrent execution | Untrusted model output drives concurrent tool.invoke calls |
| synthetic forced-commit stops → commit_stops | Graph-generated stops must still pass place_id grounding validation |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-13-04-01 | Tampering | forced commit injecting ungrounded place_ids | mitigate | synthetic stops are built from best_viable_candidate_per_slot (hits already seen via tool results) and routed through commit_stops, which rejects any place_id not in grounded scratch |
| T-13-04-02 | Elevation of Privilege | forced commit bypassing quality gates | mitigate | the synthetic commit runs through critique_final_with_stops (same deterministic gauntlet as a model-driven commit) — it cannot ship a plan that fails hard checks; D-13-04 honesty split is reported in the verdict (plan 13-05/06) |
| T-13-04-03 | Tampering | concurrency race corrupting scratch/result ordering | mitigate | asyncio.gather preserves input order; results re-assembled in original tool_call order; DB pool behavior verified via mandatory full `make test` (D-13-08 pool-contamination risk) |
| T-13-04-04 | Denial of Service | unbounded concurrency on a step with many tool calls | accept | tool_calls per step are bounded by the model's single AIMessage (small, single-digit); no fan-out amplification beyond what sequential already executes |
| T-13-04-SC | Tampering | npm/pip/cargo installs | mitigate | No package installs in this plan (asyncio is stdlib); slopcheck N/A |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_graph_forced_commit.py tests/unit/test_graph_parallel_tools.py -v` passes.
- MANDATORY: full `make test` passes (D-13-08 + project memory full_suite_db_pool_contamination — real-graph changes leak a live DB pool if not run full-suite; never trust changed-file-only runs here).
- `make typecheck` and `make lint` pass.
- With all three flags off, a reference eval/graph run is byte-identical to current main (no behavioral change → preserves [skip-baseline] eligibility ONLY if flag-off path is provably unchanged; otherwise refresh baselines per stale-baseline gate note in 13-CONTEXT.md).
</verification>

<success_criteria>
- A2 forced-commit is graph-level, model-independent, gated on full-slot viability, routed through the normal commit path, with commit_forced/forced_commit_step honesty telemetry (roadmap criterion 2 — confirmed by the mock-never-commits test).
- A3 parallel tool execution is order-stable with correct commit handling and INST-04 timing (roadmap criterion 3 instrumentation in place).
- A1 prompt addendum is wired and flag-gated (roadmap criterion 1 — additive, grep gate green).
- All three flags off → byte-identical to current behavior.
</success_criteria>

<output>
Create `.planning/phases/13-decisiveness-experiment-arms/13-04-SUMMARY.md` when done.
</output>
