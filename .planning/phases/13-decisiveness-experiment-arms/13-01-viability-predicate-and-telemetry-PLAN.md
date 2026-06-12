---
phase: 13-decisiveness-experiment-arms
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - app/agent/viability.py
  - app/agent/state.py
  - scripts/eval_agent.py
  - tests/unit/test_viability.py
  - tests/unit/test_eval_agent.py
autonomous: true
requirements: [DEC-02, DEC-05]
must_haves:
  truths:
    - "A single viability predicate decides 'does every requested stop have a viable candidate' from an ItineraryState, importable by both the graph and the eval harness without a circular import"
    - "ItineraryState carries commit_forced (bool) and forced_commit_step (int|None) fields, both JSON-safe primitives, defaulting to the no-force state"
    - "Each eval run JSON self-describes its arm config via an arm_flags dict and carries commit_forced/forced_commit_step telemetry"
    - "make_error_record produces records with safe defaults for every new field (commit_forced=False, forced_commit_step=None, arm_flags={})"
  artifacts:
    - path: "app/agent/viability.py"
      provides: "Shared viability predicate (all_slots_viable) — single source of truth for DEC-02 forced-commit gate and DEC-03 scoping"
      min_lines: 30
    - path: "app/agent/state.py"
      provides: "commit_forced + forced_commit_step fields on ItineraryState"
      contains: "commit_forced"
    - path: "scripts/eval_agent.py"
      provides: "commit_forced/forced_commit_step/arm_flags on DeterministicEvalResult"
      contains: "arm_flags"
  key_links:
    - from: "app/agent/viability.py"
      to: "scripts.eval_agent.rule8_met_per_step_from_state"
      via: "shared semantic-search viability semantics (cosine >= threshold + matching primary_type)"
      pattern: "viability_threshold"
    - from: "scripts/eval_agent.py query_result_from_state"
      to: "ItineraryState.commit_forced"
      via: "getattr forwarding into DeterministicEvalResult"
      pattern: "commit_forced"
---

<objective>
Build the shared foundation every arm consumes: one viability predicate (so the
DEC-02 forced-commit gate and the DEC-03 critique scoping share a single source
of truth for "every requested stop has a viable candidate"), the two state
telemetry fields the forced-commit mechanism writes, and the run-JSON
self-description (`arm_flags`) plus forced-commit telemetry every arm run needs
to be unambiguous.

Purpose: D-13-03 requires the forced-commit gate to use the SAME viability
definition the rule-8 harness machinery uses; D-13-04 requires `commit_forced` /
`forced_commit_step` telemetry (NOT new scorers); D-13-05 requires every eval run
JSON to self-describe its arm config. This plan lands all of that with zero
changes to graph.py, prompts.py, or revision.py, so the graph/prompt/critique
plans can build on a stable contract.

Output: `app/agent/viability.py` (new shared predicate), two new JSON-safe state
fields, three new `DeterministicEvalResult` fields wired through
`query_result_from_state` and `make_error_record`, and unit tests.
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
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Extract the shared viability predicate into app/agent/viability.py</name>
  <files>app/agent/viability.py, tests/unit/test_viability.py</files>
  <read_first>
    - scripts/eval_agent.py lines 555-735 (viable_candidates_per_step_from_state + rule8_met_per_step_from_state — the existing semantic-search-only viability semantics this predicate must match exactly: cosine >= viability_threshold AND primary_type in requested_types, multiset coverage per WR-02, semantic_search scratch only per WR-01)
    - app/agent/state.py lines 256-294 (ItineraryState fields: scratch, constraints, requested_primary_types, num_stops)
    - app/agent/revision.py line 21 (LOW_SIMILARITY_THRESHOLD = 0.55 — the threshold constant; import it, never hardcode 0.55)
    - tests/unit/test_eval_agent.py lines 308-420 (ItineraryState scratch-fixture pattern: scratch={"semantic_search": [{"step": 0, "args": {...}, "result": [...], "id": "tc0"}]})
  </read_first>
  <behavior>
    - all_slots_viable(state, threshold) returns True when, cumulatively across all semantic_search scratch steps, every requested slot has >=1 distinct viable candidate (cosine >= threshold AND primary_type in requested_primary_types when set; distinct-place_id multiset coverage matching rule8 semantics).
    - When requested_primary_types is empty, the target is constraints.num_stops (or 1 if unset) distinct viable place_ids.
    - best_viable_candidate_per_slot(state, threshold) returns an ordered list — one entry per requested slot — of the highest-cosine viable hit for that slot (used by the forced-commit synthesizer in plan 13-04). Returns None for a slot with no viable candidate.
    - Malformed/empty scratch returns False (all_slots_viable) / empty (best_viable_candidate_per_slot) without raising.
    - Predicate result agrees with the LAST element of rule8_met_per_step_from_state for the same state+threshold+requested_types (regression guard against drift).
  </behavior>
  <action>
    Create `app/agent/viability.py` exposing `all_slots_viable(state: ItineraryState, threshold: float) -> bool` and `best_viable_candidate_per_slot(state: ItineraryState, threshold: float) -> list[dict[str, Any] | None]`. Reuse the EXACT semantic-search-only viability semantics from `scripts/eval_agent.py` `rule8_met_per_step_from_state` (cosine >= threshold AND `primary_type` in `requested_primary_types`; multiset coverage keyed on distinct `place_id`; empty-types fallback targets `constraints.num_stops or 1`). Read `LOW_SIMILARITY_THRESHOLD` default from `app.agent.revision`. CRITICAL — this module lives under `app/agent/` (NOT `scripts/`) so `app/agent/graph.py` and `app/agent/revision.py` can import it WITHOUT a circular import (scripts.eval_agent imports app.agent.*, so the predicate cannot live there per PATTERNS.md "Viability definition: single source of truth"). Use `value_from_hit`-style accessors that handle both Pydantic hits (getattr) and dict hits (.get) — mirror scripts/eval_agent.py value_from_hit. Guard every scratch read with isinstance checks. Do NOT scan `nearby` scratch (WR-01: nearby hardcodes similarity=0.0). All returned candidate dicts must be plain JSON-safe dicts (no Pydantic) for the downstream forced-commit synthesizer.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_viability.py -v</automated>
  </verify>
  <acceptance_criteria>
    - `app/agent/viability.py` exports `all_slots_viable` and `best_viable_candidate_per_slot`.
    - `python -c "import app.agent.viability"` succeeds and `python -c "import app.agent.graph"` still succeeds (no circular import introduced).
    - test_viability.py asserts all_slots_viable returns True for a state where each requested_primary_type has a distinct viable hit, and False when one slot has none.
    - test_viability.py asserts all_slots_viable(state, t) == rule8_met_per_step_from_state(state, viable, types, t)[-1] for at least one shared fixture (drift guard).
    - best_viable_candidate_per_slot returns the highest-cosine hit per slot and None for an uncovered slot; every returned entry is a plain dict (assert not isinstance(entry, BaseModel)).
  </acceptance_criteria>
  <done>Shared viability predicate exists under app/agent/, importable by graph and revision without circular import, semantics provably match rule8_met_per_step_from_state.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Add commit_forced / forced_commit_step fields to ItineraryState</name>
  <files>app/agent/state.py, tests/unit/test_viability.py</files>
  <read_first>
    - app/agent/state.py lines 256-294 (ItineraryState — add the two fields next to step_telemetry, following the JSON-safe plain-primitive contract)
    - .planning/phases/13-decisiveness-experiment-arms/13-PATTERNS.md "JSON-safe state fields" section (commit_forced: bool = False; forced_commit_step: int | None = None — plain primitives only, never Pydantic)
  </read_first>
  <behavior>
    - A freshly constructed ItineraryState has commit_forced == False and forced_commit_step is None.
    - Both fields serialize cleanly through json.dumps(state.model_dump(mode="json")) (JSON-safe invariant — the same constraint step_telemetry holds).
  </behavior>
  <action>
    Add two fields to `ItineraryState` in `app/agent/state.py`, placed after `step_telemetry`: `commit_forced: bool = Field(default=False, ...)` and `forced_commit_step: int | None = Field(default=None, ...)`. Docstring each: `commit_forced` is True only when the D-13-03 forced-commit branch synthesized a `commit_itinerary` call; `forced_commit_step` is the step index of that forced commit (else None). Both are plain primitives (JSON-safe invariant — see the step_telemetry field docstring and the project memory `aimessage_tool_call_args_json_safe`). Do NOT add any Literal/enum/Pydantic-instance types.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_viability.py -v -k state_fields</automated>
  </verify>
  <acceptance_criteria>
    - `ItineraryState().commit_forced is False` and `ItineraryState().forced_commit_step is None`.
    - `json.dumps(ItineraryState().model_dump(mode="json"))` does not raise (JSON-safe).
    - mypy app/ passes (no new type errors from the field additions).
  </acceptance_criteria>
  <done>ItineraryState carries the two forced-commit telemetry fields with safe defaults; default-path behavior (flags off) is byte-identical because both default to the no-force state.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Thread commit_forced / forced_commit_step / arm_flags into eval run JSON</name>
  <files>scripts/eval_agent.py, tests/unit/test_eval_agent.py</files>
  <read_first>
    - scripts/eval_agent.py lines 120-148 (DeterministicEvalResult dataclass — append new fields after viability_threshold)
    - scripts/eval_agent.py lines 190-250 (make_error_record — every new field needs a safe default in the error path)
    - scripts/eval_agent.py lines 857-919 (query_result_from_state — construct the DeterministicEvalResult; read state.commit_forced/forced_commit_step via getattr and assemble arm_flags from env)
    - .planning/phases/13-decisiveness-experiment-arms/13-PATTERNS.md "scripts/eval_agent.py — extend for A2 telemetry fields" (exact field shapes + arm_flags dict keys)
    - app/main.py lines 753-763 (truthy-set env-flag parsing precedent: os.environ.get(NAME, "").strip().lower() in {"1","true","yes","on"})
  </read_first>
  <behavior>
    - A scored run JSON's deterministic block contains commit_forced (bool), forced_commit_step (int|None), and arm_flags (dict with keys viability_contract, forced_commit_step, parallel_tool, viability_threshold_override).
    - arm_flags reflects the env at run time: with all arm env vars unset, viability_contract is False, forced_commit_step is 0, parallel_tool is False, viability_threshold_override is None.
    - An error record (make_error_record) carries commit_forced=False, forced_commit_step=None, arm_flags={} and serializes via asdict() without raising.
  </behavior>
  <action>
    In `scripts/eval_agent.py`, append three fields to the `DeterministicEvalResult` dataclass after `viability_threshold`: `commit_forced: bool`, `forced_commit_step: int | None`, and `arm_flags: dict[str, Any]`. In `query_result_from_state`, populate them: `commit_forced = getattr(state, "commit_forced", False)`, `forced_commit_step = getattr(state, "forced_commit_step", None)`, and build `arm_flags` with keys `viability_contract` (bool from `VIABILITY_CONTRACT_ENABLED`), `forced_commit_step` (int from `FORCED_COMMIT_STEP`, "0" default), `parallel_tool` (bool from `PARALLEL_TOOL_EXECUTION_ENABLED`), `viability_threshold_override` (`os.environ.get("LOW_SIMILARITY_THRESHOLD_OVERRIDE") or None`). Use the truthy-set parser from app/main.py:758 verbatim for the bool flags. In `make_error_record`, add `commit_forced=False, forced_commit_step=None, arm_flags={}` to the DeterministicEvalResult constructor so the error path stays serializable (Phase-12 make_error_record backward-compat pattern). Do NOT add new scorers (anti-scope, D-13-04) — these are telemetry fields only; do NOT register them in DETERMINISTIC_CHECKS.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_eval_agent.py -v -k "arm_flags or forced or error_record"</automated>
  </verify>
  <acceptance_criteria>
    - DeterministicEvalResult has commit_forced, forced_commit_step, arm_flags fields.
    - A test asserts query_result_from_state on a plain ItineraryState (no env set) yields arm_flags == {"viability_contract": False, "forced_commit_step": 0, "parallel_tool": False, "viability_threshold_override": None}.
    - A test sets FORCED_COMMIT_STEP=6 + VIABILITY_CONTRACT_ENABLED=1 via monkeypatch and asserts arm_flags reflects them.
    - make_error_record(...) result serializes via dataclasses.asdict without raising and carries the three new defaults.
    - `grep -c "DETERMINISTIC_CHECKS\[" scripts/eval_agent.py` shows no new scorer registration for these fields (telemetry only).
  </acceptance_criteria>
  <done>Every eval run JSON self-describes its arm config and carries forced-commit telemetry; error records keep safe defaults; no new scorers added.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| eval env vars → graph/harness behavior | Arm flags are read from process env (operator-controlled in eval; never set in prod by this phase) |
| LLM tool-call args → state.scratch → viability predicate | Untrusted model output reaches the viability reader |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-13-01-01 | Tampering | viability predicate reading model-supplied scratch hits | mitigate | isinstance-guard every read; non-numeric similarity / missing place_id degrade to "not viable" rather than raising (mirrors rule8_met_per_step_from_state) |
| T-13-01-02 | Information Disclosure | arm_flags in run JSON | accept | arm_flags carries only flag booleans/ints + threshold override string — no secrets, no PII; run JSONs are local eval artifacts |
| T-13-01-03 | Denial of Service | malformed scratch with huge result lists | accept | eval-only path, bounded by tool result sizes already capped upstream in retrieval; no new unbounded loop introduced |
| T-13-01-SC | Tampering | npm/pip/cargo installs | mitigate | No new package installs in this plan (pure stdlib + existing deps); slopcheck N/A |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_viability.py tests/unit/test_eval_agent.py -v` passes.
- `poetry run python -c "import app.agent.graph; import app.agent.revision; import app.agent.viability"` succeeds (no circular import).
- `make typecheck` (mypy app/) passes.
- `make lint` passes (ruff, line-length 100).
- Default-path behavior with all arm flags unset is byte-identical: new state fields default to no-force, arm_flags reflects "all off".
</verification>

<success_criteria>
- Shared viability predicate exists under app/agent/, provably matches rule8_met_per_step_from_state semantics.
- ItineraryState carries commit_forced + forced_commit_step (JSON-safe).
- Eval run JSON self-describes arm config (arm_flags) and carries forced-commit telemetry; error records keep safe defaults.
- No new scorers registered (D-13-04 anti-scope honored).
</success_criteria>

<output>
Create `.planning/phases/13-decisiveness-experiment-arms/13-01-SUMMARY.md` when done.
</output>
