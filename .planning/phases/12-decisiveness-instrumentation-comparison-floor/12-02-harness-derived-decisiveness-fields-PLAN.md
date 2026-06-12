---
phase: 12-decisiveness-instrumentation-comparison-floor
plan: 02
type: execute
wave: 2
depends_on: ["12-01"]
files_modified:
  - scripts/eval_agent.py
  - tests/unit/test_eval_agent.py
autonomous: true
requirements: [INST-01, INST-02, INST-03]
must_haves:
  truths:
    - "Each run JSON records first_commit_call_step: the step index of the first commit_itinerary call, or null if never"
    - "Each run JSON records per-step viable-candidate counts using the imported LOW_SIMILARITY_THRESHOLD (not a hardcoded 0.55)"
    - "Each run JSON records, per step, whether every requested stop already had >=1 viable candidate (rule-8 precondition met) plus the steps where it was met but the model kept searching"
    - "Each run JSON records viability_threshold so the run is self-describing when Phase 13 lowers the threshold"
    - "step_telemetry from ItineraryState is forwarded verbatim into the run JSON (no field filtering on the asdict write path)"
  artifacts:
    - path: "scripts/eval_agent.py"
      provides: "first_commit_call_step_from_state, viable_candidates_per_step_from_state, rule8_met_per_step_from_state helpers + extended DeterministicEvalResult"
      contains: "first_commit_call_step"
    - path: "tests/unit/test_eval_agent.py"
      provides: "unit tests for the three derived-field helpers"
      contains: "first_commit_call_step_from_state"
  key_links:
    - from: "scripts/eval_agent.py query_result_from_state"
      to: "ItineraryState.scratch['commit_itinerary'] step indices"
      via: "first_commit_call_step_from_state"
      pattern: "first_commit_call_step"
    - from: "scripts/eval_agent.py viability helper"
      to: "app/agent/revision.LOW_SIMILARITY_THRESHOLD"
      via: "import + per-hit similarity >= threshold check"
      pattern: "LOW_SIMILARITY_THRESHOLD"
    - from: "app/agent/state.py ItineraryState.step_telemetry (Plan 12-01)"
      to: "scripts/eval_agent.py DeterministicEvalResult.step_telemetry"
      via: "query_result_from_state forwards list(state.step_telemetry); asdict() write path serializes it verbatim into the run JSON"
      pattern: "step_telemetry"
---

<objective>
Compute the three derived (judgmental) decisiveness fields harness-side in
`scripts/eval_agent.py` and write them as first-class fields into each run JSON
(INST-01/02/03). Per D-12-01 these are eval semantics — they must NOT live in prod graph
code; they are computed from message history + scratch + the `step_telemetry` field that
Plan 12-01 added to `ItineraryState`.

- INST-01 (D-12-03): `first_commit_call_step` — step index of the first actual
  `commit_itinerary` tool call (null if never), the objective cross-provider-comparable
  primary metric. A secondary `first_commit_mention_step` is recorded as null by default
  (heuristic, opaque-reasoning-safe — never used for gating).
- INST-02 (D-12-04): per-step viable-candidate counts. Viable = hit `similarity` >=
  `LOW_SIMILARITY_THRESHOLD` (imported from `app/agent/revision.py`, never hardcoded
  0.55) AND `primary_type` matches a requested stop type. The threshold value is recorded
  as `viability_threshold` so runs self-describe.
- INST-03 (D-12-05): per-step boolean — did every requested stop already have >=1 viable
  candidate — plus a `rule8_met_but_kept_searching_steps` summary (the decisiveness gap).

Purpose: gives Phase 13 the objective per-step decisiveness signals to diff before/after
each arm. Field names are stable and self-describing per the CONTEXT specifics.

Output: three pure helper functions, an extended `DeterministicEvalResult`, wired derived
fields in `query_result_from_state`, safe defaults in `make_error_record`, and unit tests.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/12-decisiveness-instrumentation-comparison-floor/12-CONTEXT.md
@.planning/phases/12-decisiveness-instrumentation-comparison-floor/12-PATTERNS.md
@.planning/phases/12-decisiveness-instrumentation-comparison-floor/12-01-SUMMARY.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add three pure derived-field helpers + extend DeterministicEvalResult</name>
  <files>scripts/eval_agent.py</files>
  <read_first>
    - scripts/eval_agent.py (the `DeterministicEvalResult` dataclass at lines 118-130;
      the existing scratch-reading helpers `count_tool_calls` line 404,
      `tool_names_from_state` line 413, `value_from_hit` line 422, `contexts_from_state`
      line 448, `revision_reasons_from_state` line 485 — these show the pure-function +
      `value_from_hit(hit, field)` + `state.scratch.get(tool_name)` patterns to follow)
    - app/agent/revision.py (line 21: `LOW_SIMILARITY_THRESHOLD = 0.55` — the constant to
      import, never hardcode)
    - app/tools/retrieval.py (the `PlaceHit` model at line 32: it has `primary_type: str |
      None` and `similarity: float`; `nearby` returns `similarity=0.0`. Hits in scratch may
      be PlaceHit objects OR dicts after serialization — use `value_from_hit` which handles
      both)
    - app/agent/graph.py (act() lines 340-415: commit_itinerary scratch entries are dicts
      with a `"step"` key written at `state.step_count` — NOT step_count+1 — see lines 351
      and 412; semantic_search/nearby entries have `result` = list of hits and a `"step"`
      key, also written at `state.step_count`. STEP-INDEX CONTRACT: scratch entries and the
      Plan 12-01 step_telemetry entries are BOTH keyed by `state.step_count` (the pre-
      increment counter), so the i-th telemetry entry, the i-th viable_candidates_per_step
      element, and a `commit_itinerary` entry with `"step"==i` all refer to the same plan
      step. Document this alignment in each new helper's docstring.)
    - .planning/phases/12-decisiveness-instrumentation-comparison-floor/12-PATTERNS.md
      (section "scripts/eval_agent.py (modify ...)" — the exact helper signatures, the
      DeterministicEvalResult field additions, and the make_error_record default stubs)
  </read_first>
  <action>
    Add `from app.agent.revision import LOW_SIMILARITY_THRESHOLD` to the import block
    (D-12-04 — import, never write 0.55). Extend the `DeterministicEvalResult` dataclass
    (after `revision_reasons`) with these typed fields: `first_commit_call_step: int |
    None`, `first_commit_mention_step: int | None`, `viable_candidates_per_step: list[int]`,
    `rule8_met_per_step: list[bool]`, `rule8_met_but_kept_searching_steps: list[int]`,
    `step_telemetry: list[dict[str, Any]]`, `viability_threshold: float`. Add three pure
    helper functions taking `ItineraryState`:
    (1) `first_commit_call_step_from_state(state) -> int | None` — read
    `state.scratch.get("commit_itinerary")`; if it is a non-empty list of dicts with a
    `"step"` key, return `min(step values)`; else None.
    (2) `viable_candidates_per_step_from_state(state, viability_threshold, requested_types)
    -> list[int]`. CONTRACT (resolve the per-step-vs-cumulative ambiguity decisively):
    this helper returns PER-STEP (non-cumulative) counts — element i is the number of
    viable candidates found in the scratch entries whose `"step"` key equals i, NOT a
    running total. Document this explicitly in the docstring ("per-step, not cumulative").
    For each plan step (one int per `"step"` index in step order), count hits across
    `semantic_search` and `nearby` scratch entries (grouped by their `"step"` key) whose
    `value_from_hit(hit, "similarity")` is a number >= `viability_threshold` AND whose
    `value_from_hit(hit, "primary_type")` is in `requested_types` (or `requested_types` is
    empty → no type constraint, count on cosine alone). The cumulative accumulation lives
    entirely inside rule8 (next helper), NOT here.
    (3) `rule8_met_per_step_from_state(state, viable_per_step, requested_types) ->
    list[bool]`. CONTRACT: rule8 performs the CUMULATIVE accumulation internally. Because
    per-type coverage cannot be reconstructed from the flat per-step int list, rule8 MUST
    re-read the `semantic_search`/`nearby` scratch entries itself, accumulating the SET of
    requested `primary_type`s that have had >=1 viable hit across steps 0..i (inclusive).
    Element i is True iff, cumulatively up to and including step i, every requested type in
    `requested_types` has at least one viable candidate (per D-12-05). The
    `viable_per_step` argument is passed for the empty-`requested_types` fallback only: when
    `requested_types` is empty, use the count of distinct requested slots from
    `state.constraints.num_stops` if set against the cumulative sum of `viable_per_step`
    (element i True iff `sum(viable_per_step[0..i]) >= num_stops`); if `num_stops` is also
    unset, element i is True iff the cumulative sum of `viable_per_step[0..i] >= 1` ("at
    least one viable candidate exists"). Document both the cumulative-set semantics and the
    fallback in the docstring. Guard every read with isinstance checks like the existing
    helpers (scratch entries may be malformed).
  </action>
  <verify>
    <automated>poetry run python -c "from scripts.eval_agent import first_commit_call_step_from_state, viable_candidates_per_step_from_state, rule8_met_per_step_from_state, DeterministicEvalResult; import dataclasses; names = {f.name for f in dataclasses.fields(DeterministicEvalResult)}; assert {'first_commit_call_step','first_commit_mention_step','viable_candidates_per_step','rule8_met_per_step','rule8_met_but_kept_searching_steps','step_telemetry','viability_threshold'} <= names"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "from app.agent.revision import LOW_SIMILARITY_THRESHOLD" scripts/eval_agent.py` matches (imported, not hardcoded)
    - `grep -c "0.55" scripts/eval_agent.py` does not increase from baseline (no hardcoded threshold introduced; verify the file has no new literal 0.55 in viability logic)
    - `DeterministicEvalResult` has all seven new fields (verified by the dataclasses.fields assertion in verify)
    - The three helpers exist and `first_commit_call_step_from_state(ItineraryState())` returns None (no commit)
    - `viable_candidates_per_step_from_state` docstring states the list is PER-STEP (non-cumulative); `rule8_met_per_step_from_state` docstring states it accumulates cumulatively (set of covered requested types across steps 0..i) — the per-step-vs-cumulative contract is unambiguous
    - All three helpers guard scratch reads with isinstance checks (no KeyError/TypeError on a malformed/empty state)
  </acceptance_criteria>
  <done>Three pure helpers plus seven new DeterministicEvalResult fields exist; threshold is imported, never hardcoded; viable_candidates_per_step is per-step and rule8 accumulates cumulatively (documented in both docstrings).</done>
</task>

<task type="auto">
  <name>Task 2: Wire derived fields into query_result_from_state + make_error_record defaults + verify end-to-end serialization</name>
  <files>scripts/eval_agent.py</files>
  <read_first>
    - scripts/eval_agent.py (`query_result_from_state` at lines 609-641 — the current
      DeterministicEvalResult constructor call to extend; `make_error_record` at lines
      173-230 — the `empty_checks` stub pattern that every new field must mirror with a safe
      default; CRITICAL the write path: `report_to_dict` at line 1177 calls `asdict(report)`
      (line 1179) and the run JSON is written via `json.dumps(report_to_dict(report), ...)`
      at lines 1232-1233 — confirm NO field is dropped/filtered between the dataclass and
      the file write, so `step_telemetry` lands in the JSON verbatim)
    - app/agent/state.py (confirm `step_telemetry` field exists from Plan 12-01, and
      `state.constraints.requested_primary_types` is the requested-types source — line 94)
    - .planning/phases/12-decisiveness-instrumentation-comparison-floor/12-PATTERNS.md
      (the `query_result_from_state` extension block and the make_error_record backward-compat
      stubs)
  </read_first>
  <action>
    In `query_result_from_state`: before constructing `DeterministicEvalResult`, compute
    `threshold = LOW_SIMILARITY_THRESHOLD`, `requested_types =
    list(state.constraints.requested_primary_types)`, `viable_per_step =
    viable_candidates_per_step_from_state(state, threshold, requested_types)`,
    `rule8_per_step = rule8_met_per_step_from_state(state, viable_per_step,
    requested_types)`, then derive `rule8_met_steps = [i for i, met in
    enumerate(rule8_per_step) if met]`, `commit_steps = {e["step"] for e in
    state.scratch.get("commit_itinerary", []) if isinstance(e, dict) and "step" in e}`, and
    `rule8_kept_searching = [s for s in rule8_met_steps if s not in commit_steps]`. Pass all
    seven new fields into the `DeterministicEvalResult(...)` call:
    `first_commit_call_step=first_commit_call_step_from_state(state)`,
    `first_commit_mention_step=None` (opaque-reasoning-safe default per D-12-03),
    `viable_candidates_per_step=viable_per_step`, `rule8_met_per_step=rule8_per_step`,
    `rule8_met_but_kept_searching_steps=rule8_kept_searching`,
    `step_telemetry=list(state.step_telemetry)`, `viability_threshold=threshold`. In
    `make_error_record`, add safe defaults to its `DeterministicEvalResult(...)` call:
    `first_commit_call_step=None`, `first_commit_mention_step=None`,
    `viable_candidates_per_step=[]`, `rule8_met_per_step=[]`,
    `rule8_met_but_kept_searching_steps=[]`, `step_telemetry=[]`,
    `viability_threshold=LOW_SIMILARITY_THRESHOLD`. Both constructors must remain
    `asdict()`-clean (no non-serializable values). Do NOT add any post-`asdict` filtering or
    field allow-list in `report_to_dict`/the write path — `step_telemetry` (and every new
    field) must reach the run JSON verbatim (INST-04 success criterion: readable in the run
    JSON without post-processing, D-12-02).
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_eval_agent.py -x -q</automated>
  </verify>
  <acceptance_criteria>
    - `query_result_from_state` passes all seven new fields to DeterministicEvalResult (grep shows first_commit_call_step=, viability_threshold=, step_telemetry= in the function body)
    - `make_error_record` passes safe defaults for all seven new fields (None/[]/threshold)
    - Inspect the write path: `report_to_dict` (line ~1177) calls `asdict(report)` and the file write (lines ~1232-1233) dumps that dict — confirm by reading these lines that NO field allow-list / pop / filter is applied to the deterministic block, so step_telemetry serializes verbatim (`grep -n "asdict(report)" scripts/eval_agent.py` matches and no `del`/`pop`/key-allowlist sits between asdict and json.dumps on the report path)
    - A run JSON dict produced from a state carrying step_telemetry (e.g. build a QueryEvalResult via query_result_from_state on a state with a non-empty `step_telemetry`, then `asdict(...)` it) contains a `step_telemetry` key with the same entries (round-trip: `json.dumps` succeeds and the deterministic block exposes step_telemetry without post-processing)
    - `poetry run python -c "from dataclasses import asdict; from scripts.eval_agent import make_error_record, EvalQuery"` plus building an error record and calling `asdict()` on it succeeds and `json.dumps` of the result works
    - The run-JSON dict produced from a state with a committed itinerary contains a numeric `first_commit_call_step` and a `viability_threshold` equal to LOW_SIMILARITY_THRESHOLD
    - `poetry run pytest tests/unit/test_eval_agent.py -q` passes
  </acceptance_criteria>
  <done>Both record-assembly paths populate the seven INST fields; the asdict/json write path is verified to forward step_telemetry verbatim into the run JSON with no filtering (INST-04 / D-12-02).</done>
</task>

<task type="auto">
  <name>Task 3: Unit tests for the three derived-field helpers</name>
  <files>tests/unit/test_eval_agent.py</files>
  <read_first>
    - tests/unit/test_eval_agent.py (lines 1-61: the import block from scripts.eval_agent
      and the `eval_case()` fixture style; follow it for the new imports and a minimal
      ItineraryState fixture builder)
    - scripts/eval_agent.py (the three helpers + their exact viability/rule-8 semantics from
      Task 1, to write assertions that match the implementation — note viable_per_step is
      PER-STEP, rule8 accumulates cumulatively)
    - .planning/phases/12-decisiveness-instrumentation-comparison-floor/12-PATTERNS.md
      (the `_state_with_commit_at_step` fixture pattern and the assertion examples)
  </read_first>
  <action>
    Add the three helpers to the `from scripts.eval_agent import (...)` block. Add a small
    fixture builder `_state_with_commit_at_step(step)` returning an `ItineraryState` whose
    `scratch["commit_itinerary"]` is `[{"step": step, "args": {}, "result": {}, "id":
    "tc1"}]`. Tests for `first_commit_call_step_from_state`: returns the step index for a
    single commit; returns the MIN when multiple commit entries exist; returns None on an
    empty `ItineraryState()`; returns None when scratch has a non-list/malformed
    commit_itinerary value. Tests for `viable_candidates_per_step_from_state`: build a state
    with a `semantic_search` scratch entry whose `result` is a list of dicts with
    `similarity` and `primary_type` keys, assert the per-step count includes only hits with
    `similarity >= LOW_SIMILARITY_THRESHOLD` AND matching `primary_type`; assert the list is
    PER-STEP (an entry at step 0 and another at step 1 yield two independent counts, not a
    running total); assert that with an empty `requested_types` the count falls back to
    cosine-only; assert a `nearby` entry with `similarity=0.0` contributes zero viable
    candidates. Tests for `rule8_met_per_step_from_state`: with two requested types and viable
    candidates for both accumulated across steps, the per-step boolean becomes True once BOTH
    types have been covered cumulatively (False on the step where only the first type is
    covered, True on the later step that covers the second — proves cumulative accumulation);
    with only one type covered, it stays False. Assert results are JSON-safe (`json.dumps` of
    each helper output succeeds).
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_eval_agent.py -x -q -k "commit_call_step or viable or rule8"</automated>
  </verify>
  <acceptance_criteria>
    - New imports for the three helpers are added to the existing `from scripts.eval_agent import` block
    - A test asserts `first_commit_call_step_from_state` returns the MIN step across multiple commit entries
    - A test asserts a `similarity=0.0` (nearby-style) hit contributes 0 viable candidates
    - A test asserts viability respects both `similarity >= LOW_SIMILARITY_THRESHOLD` AND primary_type match
    - A test asserts viable_candidates_per_step is PER-STEP (independent counts at step 0 and step 1, not cumulative)
    - A test asserts `rule8_met_per_step_from_state` flips False→True only once BOTH requested types are covered cumulatively (proves cumulative accumulation), and stays False when only one of two requested types has a viable candidate
    - `poetry run pytest tests/unit/test_eval_agent.py -q` passes with no new failures
  </acceptance_criteria>
  <done>Unit tests cover the commit-step, viability (per-step), and rule-8 (cumulative) helpers including the cosine/type/empty-type edge cases.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| eval harness ← prod graph state | eval semantics (viability judgments) must stay in eval_agent.py, never in prod graph code (D-12-01) |
| run JSON consumer ← derived fields | Phase 13 diffs these fields; they must be objective and self-describing (record the threshold used) |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-12-04 | Tampering | hardcoded threshold drift | mitigate | import LOW_SIMILARITY_THRESHOLD; Task 1 acceptance gate forbids a new 0.55 literal; viability_threshold recorded in every run for auditability |
| T-12-05 | Repudiation | non-comparable commit metric | mitigate | first_commit_call_step is the objective tool-call signal (D-12-03); first_commit_mention_step defaults null and is documented heuristic, never gated |
| T-12-06 | Information disclosure | derived fields in run JSON | accept | run JSONs already record contexts/place data; the new fields are counts/steps/floats — no new sensitive data class |
| T-12-11 | Tampering | step_telemetry silently dropped on write path | mitigate | Task 2 verifies asdict→json.dumps forwards step_telemetry verbatim (no allow-list/pop between dataclass and file); round-trip test asserts presence in the JSON |
| T-12-SC | Tampering | npm/pip/cargo installs | mitigate | no new package installs (imports an existing in-repo constant + stdlib); no slopcheck needed |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_eval_agent.py -q` passes
- `make lint` passes (ruff E,F,I,N,UP,B,SIM per CLAUDE.md) on the modified files
- `grep -n "LOW_SIMILARITY_THRESHOLD" scripts/eval_agent.py` shows the import is used (no hardcoded 0.55 in viability logic)
- A run JSON built from a committed state contains numeric first_commit_call_step, a viable_candidates_per_step list, a rule8_met_per_step list, viability_threshold == LOW_SIMILARITY_THRESHOLD, and step_telemetry forwarded verbatim — all readable directly (no post-processing, D-12-02)
- `make_error_record` output is asdict/json-safe with the seven new fields stubbed
</verification>

<success_criteria>
- INST-01/02/03 satisfied: each run JSON directly exposes steps-to-first-commit, per-step viable-candidate counts, and the rule-8 precondition met/kept-searching signal
- INST-04 closed end-to-end: step_telemetry from ItineraryState reaches the run JSON verbatim through the asdict write path (verified, no filtering)
- All eval semantics live harness-side; prod graph code is untouched (D-12-01)
- Run JSON is self-describing (viability_threshold recorded) so Phase 13 threshold changes are traceable
</success_criteria>

<output>
Create `.planning/phases/12-decisiveness-instrumentation-comparison-floor/12-02-SUMMARY.md` when done
</output>
</content>
</invoke>
