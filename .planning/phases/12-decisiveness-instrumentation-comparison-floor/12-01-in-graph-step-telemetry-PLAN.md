---
phase: 12-decisiveness-instrumentation-comparison-floor
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - app/agent/state.py
  - app/agent/graph.py
  - tests/unit/test_agent_graph.py
autonomous: true
requirements: [INST-04]
must_haves:
  truths:
    - "Each completed eval run's JSON contains a step_telemetry list with one entry per plan step"
    - "Each step_telemetry entry records the LLM-call wall time and the tool-execution wall time for that step separately"
    - "step_telemetry survives the LangGraph state reducer between plan() and act() without crashing the next plan step"
  artifacts:
    - path: "app/agent/state.py"
      provides: "step_telemetry field on ItineraryState"
      contains: "step_telemetry"
    - path: "app/agent/graph.py"
      provides: "in-graph timing hooks in plan() and act()"
      contains: "step_telemetry"
  key_links:
    - from: "app/agent/graph.py plan()"
      to: "ItineraryState.step_telemetry"
      via: "return {'step_telemetry': ...} update"
      pattern: "step_telemetry"
    - from: "app/agent/graph.py act()"
      to: "ItineraryState.step_telemetry"
      via: "patch current-step entry with tool_exec_seconds + tool_calls_this_step"
      pattern: "tool_exec_seconds"
---

<objective>
Add always-on in-graph per-step timing telemetry to the agent loop (INST-04). The graph
records, for each plan step, the LLM-call wall time and the sequential tool-execution
wall time as a `step_telemetry` list on `ItineraryState`. This is the only INST signal
that genuinely cannot be reconstructed post-hoc (D-12-02), so it must be captured live
in `plan()`/`act()`.

Purpose: gives Phase 13 the latency-decomposition data (LLM call time vs tool time per
step) needed to judge whether decisiveness arms reduce step-count and per-turn latency.
Per D-12-01 this is cheap enough to stay on in prod; only raw timings/counts live in the
graph — no eval semantics.

Output: `step_telemetry` field on `ItineraryState`, timing hooks in `plan()` and `act()`,
unit tests proving per-step entries are produced and JSON-safe.
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
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add step_telemetry field to ItineraryState</name>
  <files>app/agent/state.py</files>
  <read_first>
    - app/agent/state.py (the ItineraryState class at lines 256-279 — see the existing
      field pattern: revision_hints, revision_counts, closure_context all use
      `Field(default_factory=...)`; model_config = ConfigDict(arbitrary_types_allowed=True))
    - .planning/phases/12-decisiveness-instrumentation-comparison-floor/12-PATTERNS.md
      (section "app/agent/state.py (modify — add step_telemetry field)" — the exact field
      definition to copy and the JSON-safe constraint)
  </read_first>
  <action>
    Append a `step_telemetry` field to `ItineraryState` immediately after the
    `closure_context` field (line 277), before `model_config`. Type it
    `list[dict[str, Any]]` with `Field(default_factory=list)` and a description naming the
    four primitive keys each entry carries: `step` (int), `llm_call_seconds` (float),
    `tool_exec_seconds` (float), `tool_calls_this_step` (int). Cite D-12-01 in the field
    description (always-on; cheap enough for prod). Do NOT add any new import — `Any` and
    `Field` are already imported in this module. Every value written into a step_telemetry
    entry MUST be a plain Python primitive (int/float/str) — never a Pydantic model — per
    the `aimessage_tool_call_args_json_safe` incident, because non-JSON-safe values in
    state crash the next plan() step when the AIMessage is re-serialized.
  </action>
  <verify>
    <automated>poetry run python -c "from app.agent.state import ItineraryState; s = ItineraryState(); assert s.step_telemetry == []; import json; json.dumps(s.step_telemetry)"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "step_telemetry" app/agent/state.py` returns the field definition inside the
      ItineraryState class (after closure_context, before model_config)
    - The field type annotation is `list[dict[str, Any]]` with `Field(default_factory=list)`
    - A fresh `ItineraryState()` has `step_telemetry == []` and `json.dumps(state.step_telemetry)` succeeds (proves JSON-safe default)
    - The field description string contains the literal token `D-12-01` and names all four keys: step, llm_call_seconds, tool_exec_seconds, tool_calls_this_step
  </acceptance_criteria>
  <done>ItineraryState carries a JSON-safe step_telemetry list field defaulting to [].</done>
</task>

<task type="auto">
  <name>Task 2: Add timing hooks to plan() and act() in graph.py</name>
  <files>app/agent/graph.py</files>
  <read_first>
    - app/agent/graph.py (the plan() closure at lines 283-329 and act() at lines 331-428 —
      note: plan() wraps `ai = await llm_with_tools.ainvoke(messages_for_llm)` at line 314
      and returns `{"messages": new_messages}` at line 329; act() loops `for tc in
      ai.tool_calls:` at line 340, executes tools via `await asyncio.to_thread(tool.invoke,
      effective_args)` at line 402, and assembles `update` at lines 421-427 returning it at
      428. CRITICAL existing comment at 371-378: never reassign tc["args"]. STEP-INDEX
      CONTRACT for Plan 12-02: act() writes scratch entries with `state.step_count` (NOT
      step_count+1 — see lines 351 and 412); the telemetry entry MUST use the SAME
      `state.step_count` so scratch entries and telemetry entries share one step index.)
    - app/agent/state.py (confirm the step_telemetry field shape from Task 1)
    - .planning/phases/12-decisiveness-instrumentation-comparison-floor/12-PATTERNS.md
      (section "app/agent/graph.py (modify — timing hooks)" — the exact plan() and act()
      timing-hook patterns, including the act() patch-the-current-step logic)
  </read_first>
  <action>
    Add `import time` to the top of graph.py only if it is not already imported (check the
    existing import block first; do not add a duplicate). In `plan()`: wrap the existing
    `await llm_with_tools.ainvoke(messages_for_llm)` call (line 314) with
    `time.monotonic()` before/after to compute `llm_call_seconds`; change the return at
    line 329 to also include a `step_telemetry` key whose value is
    `[*state.step_telemetry, {...}]` appending one entry `{"step": state.step_count,
    "llm_call_seconds": <elapsed>, "tool_exec_seconds": 0.0, "tool_calls_this_step": 0}`.
    In `act()`: time the entire `for tc in ai.tool_calls:` loop with `time.monotonic()`
    before/after (one wall-clock span covering sequential tool execution), and count
    tool calls executed into `tool_calls_this_step`. Before returning `update` (line 428),
    PATCH the telemetry: copy `state.step_telemetry` into a local list; if the last entry's
    `"step"` equals `state.step_count` (the entry plan() wrote for this step), replace it
    with a copy that sets `tool_exec_seconds` to the measured elapsed and
    `tool_calls_this_step` to the count; otherwise append a fresh entry with
    `llm_call_seconds` 0.0. Add the resulting list to `update` under key `step_telemetry`.
    Use `state.step_count` (pre-increment) for the telemetry `"step"` value so it aligns
    with the scratch entries act() writes at `state.step_count` (the Plan 12-02 step-index
    contract). All written values MUST be plain int/float (D-12-01 JSON-safe constraint). Do
    NOT alter the existing tool-execution body, the `effective_args` computation, or the
    `committed_stops`/`merged_scratch` assembly — only add timing measurement and the
    telemetry key on the returned update dicts.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_agent_graph.py -x -q -k "telemetry or step"</automated>
  </verify>
  <acceptance_criteria>
    - `grep -n "import time" app/agent/graph.py` shows time is importable (existing or added once — not duplicated)
    - `grep -c "step_telemetry" app/agent/graph.py` is >= 2 (plan() and act() each reference it)
    - `grep -n "time.monotonic" app/agent/graph.py` shows at least two monotonic spans (one in plan around ainvoke, one in act around the tool loop)
    - The plan() return dict and the act() update dict each include a `step_telemetry` key
    - The telemetry entry's `"step"` uses `state.step_count` (pre-increment), matching the scratch step index written at lines 351/412
    - The existing line `result: Any = await asyncio.to_thread(tool.invoke, effective_args)` is unchanged (`grep -n "asyncio.to_thread(tool.invoke" app/agent/graph.py` still matches)
    - `tc["args"]` is never reassigned (the existing belt-and-suspenders comment block at lines 371-398 and the `effective_args` local are intact)
  </acceptance_criteria>
  <done>plan() records LLM-call wall time and act() records tool-execution wall time + tool count into step_telemetry, one entry per plan step, all values JSON-safe, keyed by the same state.step_count as the scratch entries.</done>
</task>

<task type="auto">
  <name>Task 3: Unit-test per-step telemetry production through the graph</name>
  <files>tests/unit/test_agent_graph.py</files>
  <read_first>
    - tests/unit/test_agent_graph.py (the `_ScriptedLLM` class at line 78, `_make_fake()`
      helper at line 103, and the multi-step analog tests `test_graph_executes_tool_and_continues`
      at line 263 and `test_graph_finalizes_on_commit_even_if_llm_keeps_calling_tools` at line
      330 — these build the graph via `build_agent_graph(fake, max_steps=4)`, drive it with
      `await graph.ainvoke(ItineraryState(messages=[HumanMessage(...)]))`, and mock
      `app.agent.tools._semantic_search` and `app.agent.revision.itinerary_violations`)
    - app/agent/graph.py (the plan()/act() telemetry code from Task 2, to know the exact
      keys to assert on)
    - .planning/STATE.md (note: full-suite DB-pool contamination risk — real-graph tests
      need itinerary_violations mocked; the analog tests already do this — copy that mocking)
  </read_first>
  <action>
    Add async unit test(s) modeled on `test_graph_executes_tool_and_continues` (line 263):
    script a `_ScriptedLLM` with a tool-call step (e.g. a semantic_search call) followed by
    a commit/finalize step via `_make_fake([...])`, build with
    `build_agent_graph(fake, max_steps=4)`, monkeypatch
    `app.agent.tools._semantic_search` to a list and mocker.patch
    `app.agent.revision.itinerary_violations` to `[]` (same as the analog tests), and run
    `await graph.ainvoke(...)`. Read `step_telemetry` off the resulting state
    (`out["step_telemetry"]` or the returned ItineraryState). Assert: (1) it is a non-empty
    list of dicts; (2) every entry's key set is exactly `{"step", "llm_call_seconds",
    "tool_exec_seconds", "tool_calls_this_step"}`; (3) all values are plain int/float
    (assert `isinstance(v, (int, float))` and that count fields are ints, not bool);
    (4) `json.dumps(state.step_telemetry)` succeeds; (5) at least one entry has
    `tool_calls_this_step >= 1` and `tool_exec_seconds >= 0.0`. Do NOT assert absolute
    timing magnitudes (non-deterministic) — only presence, key set, types, and JSON-safety.
  </action>
  <verify>
    <automated>poetry run pytest tests/unit/test_agent_graph.py -x -q -k "telemetry or step_telemetry"</automated>
  </verify>
  <acceptance_criteria>
    - New test(s) in tests/unit/test_agent_graph.py assert step_telemetry is a non-empty list after a multi-step run
    - A test asserts every entry's key set equals {step, llm_call_seconds, tool_exec_seconds, tool_calls_this_step}
    - A test calls `json.dumps(state.step_telemetry)` and it succeeds (JSON-safety guard)
    - A test asserts at least one entry has tool_calls_this_step >= 1
    - `poetry run pytest tests/unit/test_agent_graph.py -q` passes with no new failures
  </acceptance_criteria>
  <done>Unit tests prove the graph produces one JSON-safe telemetry entry per plan step with correct keys and types.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| graph state → next plan() step re-serialization | step_telemetry values are re-serialized into the AIMessage/state on the next step; non-JSON-safe values crash the loop (prior incident) |
| prod runtime ← always-on telemetry | D-12-01 keeps this in prod code; must not leak eval semantics or measurable latency |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-12-01 | Tampering | step_telemetry dict values | mitigate | enforce plain-primitive values only (int/float); Task 3 asserts json.dumps succeeds and isinstance checks pass |
| T-12-02 | Denial of Service | always-on prod timing hooks | accept | only two time.monotonic() reads + small dict append per step — negligible; D-12-01 records this is cheap enough for prod |
| T-12-03 | Information disclosure | telemetry contents | accept | only wall-clock floats + counts recorded; no PII, no place data, no user text in step_telemetry |
| T-12-SC | Tampering | npm/pip/cargo installs | mitigate | no new package installs in this plan (uses stdlib `time`); no slopcheck needed |
</threat_model>

<verification>
- `poetry run pytest tests/unit/test_agent_graph.py -q` passes
- `make lint` passes (ruff E,F,I,N,UP,B,SIM per CLAUDE.md) on app/agent/state.py, app/agent/graph.py, and tests/unit/test_agent_graph.py
- `grep -n "step_telemetry" app/agent/state.py app/agent/graph.py` shows the field plus plan() and act() references
- A fresh ItineraryState serializes its step_telemetry via json.dumps with no error
- The existing tool-execution body (asyncio.to_thread(tool.invoke, effective_args)) and the never-reassign-tc["args"] invariant are unchanged
</verification>

<success_criteria>
- INST-04 satisfied: each plan step contributes one step_telemetry entry with separate LLM-call and tool-execution wall times — the per-turn latency decomposition Phase 13 will diff
- Telemetry is always-on in graph code (prod-safe), JSON-safe, and adds no eval semantics
- All new and existing graph unit tests pass
</success_criteria>

<output>
Create `.planning/phases/12-decisiveness-instrumentation-comparison-floor/12-01-SUMMARY.md` when done
</output>
</content>
