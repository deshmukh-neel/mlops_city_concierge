# Phase 4: Category Compliance Fix - Context

**Gathered:** 2026-05-22
**Status:** Ready for planning

<domain>
## Phase Boundary

When the user names category slots ("omakase, then drinks, then dessert" / "dinner-then-drinks-then-dessert"), the agent's retrieval tool calls pass `SearchFilters.primary_type_family` for each slot, and rationales describe the committed place's actual category — not the user's requested category. Validated against the eval harness on the gpt-4o-mini production anchor.

Concretely, Phase 4 delivers:

- `requested_primary_types: list[str]` populated on `UserConstraints` via a hybrid pipeline (deterministic slot-indicator pre-check + gated structured-output LLM intake call). Free-text queries pay zero added latency; slot-structured queries pay one extra LLM call.
- `slot_index: int | None` optional arg added to `semantic_search` and `nearby` retrieval tools so the agent can declare which slot a tool call is for. Backward-compatible (None = no slot mapping).
- Graph-layer rewrite in `app/agent/graph.py::act()`: when `requested_primary_types` is non-empty AND a retrieval tool_call carries `slot_index`, the graph injects `primary_type_family = family_of(requested_primary_types[slot_index])` into the tool_call's `filters` before execution. Mirrors the existing `_inject_closure_exclusions` pattern.
- SYSTEM_PROMPT additions: (a) tell the model to pass `slot_index` on per-slot retrieval calls when slots are named; (b) tighten step 6 ("JUSTIFY every stop") so the rationale describes the committed place's actual `primary_type`, not the user's requested category.
- New `category_compliance_strict` scorer in `checks.py` alongside the existing family-level `category_compliance`: matches `primary_type` keywords (e.g., "omakase" → `primary_type in {"Sushi Restaurant", "Japanese Restaurant"}`) so within-family drift (Pizza Restaurant on a sushi slot, both family=restaurant) is measurable.
- New `rationale_misaligned` revision-hint type in `app/agent/revision.py`: post-commit, run `rationale_stop_alignment` (existing Phase 3 scorer); if score < 1.0, emit a hint asking the model to rewrite the offending rationale. Two-retry budget per hint per the existing REVISION_GUIDANCE pattern.
- Eval YAML updates: `omakase_mission_open_ended` and `refinement_cheaper` cases get explicit `requested_primary_types` in their expected_constraints so the eval scorer fires.

Scope anchor: 4 requirements (CAT-01 through CAT-04) from `.planning/REQUIREMENTS.md`, plus RAT-01 folded in from Phase 5 (rationale-stop alignment on refinement turns is the same root cause as CAT-02; solving them together is cleaner than artificial split). Phase 5 narrows to RAT-02 only (closure-swap placeholder bleed) — a structurally different bug.

</domain>

<spec_lock>
**No SPEC.md exists for Phase 4** (the `/gsd:spec-phase` flow was not run). Requirements come from REQUIREMENTS.md CAT-01..CAT-04 and the ROADMAP.md Phase 4 success criteria. Phase 5 scope change (RAT-01 → Phase 4) is captured here and must be reflected in REQUIREMENTS.md traceability + ROADMAP.md before Phase 5 begins.
</spec_lock>

<decisions>
## Implementation Decisions

### Slot extraction source (Gray Area #1)

- **D-04-01:** Slot extraction uses a **hybrid pipeline**: (a) eval YAML carries `requested_primary_types` in expected_constraints (needed regardless), (b) deterministic regex pre-check in `app/agent/input_parsing.py` detects slot-indicator phrases (comma-separated category lists, "then", "followed by", numbered structure), (c) only when the pre-check fires, a gated structured-output LLM intake call extracts per-slot Google `primary_type` values into `UserConstraints.requested_primary_types`. Free-text queries skip the LLM call entirely — zero latency tax on the common case.
- **D-04-02:** The intake LLM uses the **same model as the planning LLM** (Phase 2's `RAG_MODEL_OVERRIDE`-resolved model, falling back to the MLflow production alias). Reasons: keeps `RAG_MODEL_OVERRIDE` meaningful for end-to-end testing of candidate models; consistent prompt-cache behavior under MLflow logging; modern OpenAI models do structured-output extraction reliably at any tier so a smaller-cheaper-model swap saves negligible cost.
- **D-04-03:** Intake output schema: a small Pydantic model with `requested_primary_types: list[str]` matching the existing `UserConstraints` field shape. Validated against `family_of()` — entries that don't map to a known family are dropped with a debug log, never raised. The agent continues with whatever slots successfully extracted (fail-open).

### Enforcement approach (Gray Area #2)

- **D-04-04:** Graph-layer enforcement, not prompt enforcement. In `app/agent/graph.py::act()`, before executing a retrieval tool call (`semantic_search`, `nearby`), if `state.constraints.requested_primary_types` is non-empty AND the tool_call carries a `slot_index`, the graph injects `primary_type_family = family_of(requested_primary_types[slot_index])` into the tool_call's `filters` arg. Mirrors the existing `_inject_closure_exclusions` pattern. Sidesteps the critique-loop ↔ commit conflict (see `project_critique_commit_conflict.md`) because the model never sees the slot filter as something to argue with.
- **D-04-05:** Retrieval tools (`semantic_search`, `nearby`) get an optional `slot_index: int | None = None` arg. Backward-compatible — free-text queries leave it None, and the graph-layer injection only fires when `slot_index is not None` AND `requested_primary_types` is non-empty. The model is prompted to pass `slot_index = i` when retrieving for stop i in a slot-structured query.
- **D-04-06:** If the model fails to pass `slot_index` despite a slot-structured query (i.e., `requested_primary_types` is non-empty but `slot_index is None`), the graph does NOT inject the filter — the model's behavior is what's measured. This is a deliberate trust boundary: the graph supplements when the model cooperates, but doesn't override the model's judgment. The eval scorer catches non-cooperation as a category-compliance failure.

### Rationale category guarantee (Gray Area #3 / CAT-02 + RAT-01)

- **D-04-07:** Prompt + scorer-in-the-loop. SYSTEM_PROMPT step 6 ("JUSTIFY every stop") gets one new line: "Your rationale MUST describe the actual `primary_type` of the committed place from the tool result, NOT the category the user asked for. Never claim a stop offers omakase if its `primary_type` is not Sushi Restaurant or similar."
- **D-04-08:** After `commit_itinerary`, run `rationale_stop_alignment` (existing Phase 3 scorer in `checks.py:256`). If score < 1.0, emit a new `rationale_misaligned` `RevisionHint` (added to `RevisionReason` Literal in `state.py:22`) with the offending stop index and the failure type ("missing_name_or_family_keyword"). The model gets one chance to rewrite via the existing REVISION_GUIDANCE pattern; after that, the agent ships the misaligned rationale rather than infinite-loop. The eval scorer catches it as a metric.
- **D-04-09:** Phase 5 scope reduced from RAT-01 + RAT-02 to **RAT-02 only** (closure-swap placeholder bleed: "Walking-distance alternative for X" reaching the user). RAT-01 (refinement-turn rationale integrity) is structurally the same as CAT-02 and folds into Phase 4. Must be reflected in REQUIREMENTS.md traceability table + ROADMAP.md Phase 5 entry before Phase 5 begins.

### Grading + gates (Gray Area #4)

- **D-04-10:** Add `category_compliance_strict` scorer in `app/agent/critique/checks.py` alongside the existing family-level `category_compliance`. Strict scorer matches `primary_type` keywords (lookup table mapping common requested-type keywords → expected Google `primary_type` strings; e.g., "omakase" → `{"Sushi Restaurant", "Japanese Restaurant"}`). Both scorers run; baselines record both; merge gate targets the strict variant for the +0.3 improvement bar (family scorer used as a regression guard).
- **D-04-11:** Cross-provider gate: DeepSeek tracked in the matrix output (`eval_matrix.py` continues to run it) but **exempt from the merge gate**. Rationale: project memory `project_deepseek_decisiveness_gap.md` documents DeepSeek 0/9 on real-provider runs; its baseline scorers read 1.0 via fail-open on empty stops; a +0.3 delta against a 1.0 floor is mathematically impossible. Gate enforces on `openai/gpt-4o-mini` only.
- **D-04-12:** `late_night_closure_cascade` scenario tracked in matrix output but **exempt from Phase 4 merge gate**. Rationale: scenario doesn't exercise category compliance (no slot structure in the query) and the eval-harness multi-turn threading mismatch (see `project_eval_multi_turn_threading_bug.md`) makes its baseline measure a non-prod scenario. Phase 5 (RAT-02) will decide whether to fix the harness, scope the scenario out long-term, or accept the caveat.
- **D-04-13:** **Phase 4 merge gate** (locked):
  - On scenarios `omakase_mission_open_ended` and `refinement_cheaper`, provider `openai/gpt-4o-mini`, with `requested_primary_types` populated in expected_constraints:
    - `category_compliance_strict` median ≥ baseline + 0.3 absolute (CAT-03 equivalent on the strict variant)
    - `category_compliance` (family-level) median ≥ baseline + 0.0 (regression guard)
    - `rationale_stop_alignment` median ≥ baseline + 0.2 absolute (was Phase 5 RAT-03; owned by Phase 4 since RAT-01 folded in)
    - No regression (median ≥ baseline) on `geographic_coherence`, `walking_budget_respected`, `temporal_coherence`, `constraints_satisfied`, `no_hallucinated_place_ids`
  - DeepSeek scorers logged but not gated.
  - `late_night_closure_cascade` logged but not gated.

- **D-04-14 (2026-05-22, locked during 04-07 merge-gate checkpoint):** When the pre-Phase-4 baseline median for `rationale_stop_alignment` is ≥ 0.95 (saturated via fail-open on non-convergence), Gate 3 is reinterpreted as an **absolute floor of 0.8** instead of a `+0.2 delta`. Rationale: same class of missing-key issue D-04-13 already resolved for `category_compliance_strict` — when the old baseline reflects scorer fail-open rather than measured quality, a delta is undefined. The 0.8 floor preserves the spirit of "no regression on a meaningful threshold" without requiring impossible-by-construction headroom.

  Phase 4 measured result: 1.000 on both gated scenarios. Gate 3 PASSES under D-04-14.

  RAT-03 metric definition is flagged for revisit in a future phase — the saturated baseline reflects scorer abstention, not "rationale-alignment quality on committed itineraries". This is logged in [[phase4-d-04-14-locked]] project memory.

### Claude's Discretion
- Exact regex patterns for the slot-indicator pre-check in `input_parsing.py` — pick patterns that catch the three target eval scenarios and a few common paraphrases ("dinner, drinks, dessert" / "X then Y then Z" / numbered lists). Documented in plan.
- Exact intake LLM prompt — structured-output template, few-shot if needed for reliability, must be small enough to not dominate latency. Plan-level decision.
- Exact `category_compliance_strict` keyword → primary_type mapping — start with the 3 target scenario categories (omakase, dinner-drinks-dessert pattern) and expand only if scoring proves brittle. Documented in plan + linked to `_PRIMARY_TYPE_FAMILIES` source of truth in `app/tools/filters.py`.
- Whether the new `slot_index` arg gets prose documentation in the tool's docstring (so the model sees it as a normal arg) vs. a separate SYSTEM_PROMPT directive — planner decides based on which is more discoverable; both can coexist.
- `rationale_misaligned` revision-hint exact wording — follow the existing REVISION_GUIDANCE pattern (one-sentence problem + suggested action).
- Baseline re-run strategy: after Phase 4 lands, the existing baselines (commit `40fe3fd`) become stale because they record category_compliance=1.0 via abstain (D-03), not via real category enforcement. Plan must include a "re-baseline after Phase 4 is functional but before merge gate is enforced" step. Sequencing: implement → re-baseline → verify gate would pass with new floor → merge.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents (researcher, planner) MUST read these before planning or implementing.**

### Phase scope and requirements
- `.planning/ROADMAP.md` § "Phase 4: Category Compliance Fix" — success criteria, branch name, requirement list
- `.planning/REQUIREMENTS.md` § "Category Compliance Fix (Phase 3)" — CAT-01..CAT-04 (note: REQUIREMENTS.md numbers this section "Phase 3" pre-reshuffle; it's the same work as roadmap Phase 4)
- `.planning/REQUIREMENTS.md` § "Rationale-Stop Alignment Fix" — RAT-01 folds into Phase 4 per D-04-09; RAT-02 remains in Phase 5; RAT-03 metric is owned by Phase 4's merge gate
- `.planning/PROJECT.md` § "Current Milestone: v2.0 Production Readiness" + "Key Decisions" — milestone-level locked decisions
- `.planning/phases/03-eval-harness-extension/03-CONTEXT.md` § D-01..D-04 — Phase 3 locked the `requested_primary_types` schema and scorer contract that Phase 4 builds on
- `.planning/research/PITFALLS.md` (if present; else inlined in SUMMARY.md) — P7 (single-model overfit), P9 (stale baselines), P10 (refinement first-turn regression)

### Baselines and eval infrastructure (extend, don't break)
- `configs/eval_baselines/omakase_mission_open_ended.json` — current floor; will need re-baselining after Phase 4 implementation lands and before merge gate enforced
- `configs/eval_baselines/refinement_cheaper.json` — current floor; same re-baseline note
- `configs/eval_baselines/late_night_closure_cascade.json` — tracked but not gated for Phase 4
- `configs/eval_queries.yaml` — `omakase_mission_open_ended` (line 376) and `refinement_cheaper` (line 388) need `requested_primary_types` added to expected_constraints
- `scripts/eval_agent.py` + `scripts/eval_matrix.py` — runners that consume the new scorer and new YAML fields
- `eval_reports/2026-05-22T19-00-11Z/summary.json` — the pre-Phase-4 baseline run, source of the committed baselines

### Agent state, tools, and graph (the surfaces Phase 4 changes)
- `app/agent/state.py` § `UserConstraints.requested_primary_types` (line 90) — already exists from Phase 3; Phase 4 wires the producer
- `app/agent/state.py` § `RevisionReason` (line 22) — add `"rationale_misaligned"` literal
- `app/agent/state.py` § `Stop` (lines 148-162) — `primary_type`, `rationale`, `name` are the fields scorers and rationale-guarantee logic read
- `app/agent/tools.py` § `semantic_search` (line 35) and `nearby` (line 49) — add optional `slot_index: int | None = None` arg
- `app/agent/graph.py` § `act()` (~line 300) — graph-layer filter injection lives here, next to existing patterns
- `app/agent/input_parsing.py` — slot-indicator pre-check lives here, deliberately conservative per its docstring
- `app/agent/prompts.py` § `SYSTEM_PROMPT` — step 6 (rationale) gets one new line; new directive on `slot_index` usage
- `app/agent/revision.py` § `_diagnose_last_tool_result` (line 130) and surrounding — `rationale_misaligned` hint emission lives here
- `app/agent/critique/checks.py` § `category_compliance` (line 218) and `rationale_stop_alignment` (line 256) — existing scorers; add `category_compliance_strict` here; both new+old register in `CRITIQUE_THRESHOLDS` and `DETERMINISTIC_CHECKS`

### Filter family logic (Phase 2 carry-over)
- `app/tools/filters.py` § `SearchFilters.primary_type_family` (line 74) — the field the graph injects
- `app/tools/filters.py` § `family_of()` and `_PRIMARY_TYPE_FAMILIES` (lines ~167-258) — coarse-family lookup used by both family-level scorer and the new strict scorer's fallback
- `app/tools/filters.py` § `compile_filters()` (line 302) — already handles `primary_type_family` in the SQL clause; no change needed downstream of injection

### Model selection (Phase 2 carry-over)
- `app/main.py` § `load_registered_rag_chain` + `_parse_model_override` — Phase 2's `RAG_MODEL_OVERRIDE` env var; the intake LLM uses this same resolution path
- `app/llm_factory.py` § `build_chat_model(provider, model, temperature)` — what the intake call constructs
- `README.md` § Model Override section — may need a one-line note that intake LLM also respects override

### Production /chat plumbing
- `app/main.py:597-607` — where `ItineraryState` is constructed for a `/chat` request; this is where slot extraction must happen so `requested_primary_types` is populated before the graph runs
- `app/agent/io.py::messages_from_history` — relevant if Phase 4's plan touches multi-turn shape (memory says it doesn't, but flagged for awareness)

### Project memory — must-read before planning
- `project_eval_multi_turn_threading_bug.md` — why `late_night_closure_cascade` is exempt from Phase 4 gate
- `project_deepseek_decisiveness_gap.md` — why DeepSeek is exempt from Phase 4 gate
- `project_critique_commit_conflict.md` — why Phase 4 uses graph-layer enforcement (D-04-04), not prompt-only

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `family_of()` in `app/tools/filters.py:269` — already used by Phase 3's `category_compliance` and `rationale_stop_alignment`; reused by Phase 4's intake validation (D-04-03) and the new strict scorer
- `_PRIMARY_TYPE_FAMILIES` in `app/tools/filters.py:167` — single source of truth for the family ↔ primary_type mapping; the new strict scorer's keyword table is a layer ON TOP of this, not a replacement
- `_inject_closure_exclusions` in `app/agent/graph.py` (per memory `aimessage_tool_call_args_json_safe.md`) — the exact pattern Phase 4's `_inject_primary_type_family` mirrors; both run in `act()` before tool execution, both rewrite tool_call args without mutating shared state
- `RevisionHint` + revision-loop dispatch in `app/agent/revision.py` — adding a new `RevisionReason` literal + a new emission site is a small, well-typed extension
- Existing scorer pattern `(state: ItineraryState) -> float` returning 0.0-1.0; threshold lives in `CRITIQUE_THRESHOLDS`; auto-runs via `itinerary_violations` — `category_compliance_strict` follows this exactly
- `EvalQuery.expected_constraints` (Pydantic, in `app/eval/config.py`) — already supports per-case overrides; adding `requested_primary_types` here is a small field addition
- Phase 2's `RAG_MODEL_OVERRIDE` + `load_registered_rag_chain` — Phase 4's intake LLM resolution path

### Established Patterns
- Slot ordering assumption: the agent prompt + critique structure currently treats tool calls within an AIMessage as "for stop K, in order." The new `slot_index` arg makes this explicit and avoids the unstated invariant. Critical to set this up cleanly because the graph-layer filter injection depends on the index being right.
- Graph-layer rewrites of tool_call args are a known pattern on this codebase (`_inject_closure_exclusions`), with one critical constraint per memory `aimessage_tool_call_args_json_safe.md`: never reassign `tc["args"]` — use a local `effective_args` dict that gets passed to the tool executor. Phase 4 MUST follow this pattern.
- Tests follow the layering memory: unit + smoke + functional + integration per module. New surfaces (intake LLM, slot_index arg, strict scorer, rationale_misaligned hint, graph injection) each need at least unit + functional; intake LLM also needs an integration test that pins a known input → known structured output via the scripted-LLM provider (per memory `feedback_test_layering.md`).
- JSON serialization: every test that touches `tool_call_args` asserts `json.dumps(args)` does not raise. Phase 3 EVAL-08 enforces this. Phase 4's graph injection must preserve this — if `effective_args` contains a Pydantic instance, the next plan step crashes per memory `aimessage_tool_call_args_json_safe.md`.

### Integration Points
- `app/main.py:597-607` — the slot-extraction hybrid pipeline (D-04-01) hooks in here, BEFORE `graph.ainvoke`. Order: (1) deterministic pre-check on the user message; (2) if pre-check fires, await intake LLM call; (3) build `UserConstraints` with `requested_primary_types` populated; (4) construct `ItineraryState` and invoke graph. Pre-check failures fall through with `requested_primary_types=[]` — graceful degradation to existing free-text behavior.
- `scripts/eval_agent.py::evaluate_cases` — when `EvalQuery.expected_constraints.requested_primary_types` is set, the runner constructs `UserConstraints(requested_primary_types=[...])` directly instead of going through the intake pipeline. This bypasses intake for deterministic eval. Mirrors the existing pattern where eval bypasses production parsers for reproducibility.
- `Makefile` — no new targets needed; Phase 3's `eval-agent` and `eval-matrix` work unchanged once scorers are added.
- `configs/eval_queries.yaml` — modify `omakase_mission_open_ended` and `refinement_cheaper` to add `requested_primary_types`; this is a 2-line YAML change per case but cascades into baseline staleness (D-04-13 sequencing note).
- `configs/eval_baselines/*.json` — must be re-baselined after Phase 4 implementation completes and before the merge gate is enforced (D-04 sequencing in Claude's Discretion).

</code_context>

<specifics>
## Specific Ideas

- **Slot indicator pre-check regex** (D-04-01) — starting set, expandable per Claude's Discretion:
  - Comma-separated category list near a planning verb: `(plan|schedule|book|do).*?\b(\w+(?:,\s*\w+){1,})\b`
  - "X then Y [then Z]" pattern: `\b\w+\s+(?:then|followed by)\s+\w+`
  - Numbered structure: `\b1\..*?2\.` (already exists in agent prompts test set for stop-count parsing — extend to slot category)
  - Explicit slot vocabulary words: `\b(dinner|drinks|dessert|brunch|lunch|breakfast|cocktails|coffee|nightcap)\b` appearing 2+ times in proximity

- **Intake LLM prompt shape** (D-04-02 / D-04-03), suggested:
  ```
  Extract the user's per-slot category structure. The user's message is:
  "{user_message}"

  If the user named distinct slots (e.g., "dinner, drinks, dessert" or
  "omakase then ramen"), return a list of Google primary_type values, one
  per slot in order. If the message is free-text or has no clear slot
  structure, return [].

  Output shape: {"requested_primary_types": ["restaurant", "bar", "dessert_shop"]}

  Use this vocabulary (Google primary_type values):
  - Restaurants: restaurant, japanese_restaurant, sushi_restaurant, ...
  - Bars: bar, cocktail_bar, wine_bar, ...
  - Dessert: dessert_shop, bakery, ice_cream_shop, cafe, ...
  - Cafes: cafe, coffee_shop, tea_house
  ```
  Few-shot if structured output drifts. Validate against `family_of()` — drop unmappable values.

- **`category_compliance_strict` keyword table** (D-04-10), starting set:
  ```python
  _STRICT_TYPE_KEYWORDS: dict[str, frozenset[str]] = {
      "omakase": frozenset({"Sushi Restaurant", "Japanese Restaurant", "Fine Dining Restaurant"}),
      "sushi": frozenset({"Sushi Restaurant", "Japanese Restaurant"}),
      "ramen": frozenset({"Ramen Restaurant", "Japanese Restaurant"}),
      "tacos": frozenset({"Mexican Restaurant", "Restaurant"}),
      "cocktails": frozenset({"Cocktail Bar", "Bar"}),
      "dessert": frozenset({"Dessert Shop", "Bakery", "Ice Cream Shop"}),
      # ... starting set; expand only when needed
  }
  ```
  Scorer matches if requested keyword's expected set ∩ {committed_stop.primary_type} is non-empty. Falls back to family-level on unmapped keywords.

- **Phase 4 plan sequencing** (suggested order; planner to confirm):
  1. Add `slot_index` arg to `semantic_search` and `nearby` tools (backward-compatible)
  2. Add `category_compliance_strict` scorer + `_STRICT_TYPE_KEYWORDS` table; register in `CRITIQUE_THRESHOLDS` + `DETERMINISTIC_CHECKS`
  3. Update eval YAML to add `requested_primary_types` to the two target cases
  4. Graph-layer injection (`_inject_primary_type_family`) in `act()`, mirroring `_inject_closure_exclusions`
  5. SYSTEM_PROMPT additions (slot_index directive + rationale step 6 tightening)
  6. `rationale_misaligned` revision-hint type + emission site in `revision.py`
  7. Slot-indicator pre-check in `input_parsing.py` + intake LLM in `app/main.py` request setup
  8. Re-baseline: `APP_ENV=eval make eval-matrix RUNS=3`, post-process into baselines, commit
  9. Verify merge gate would pass with new baseline floor; if not, iterate on the prompt or filter logic; if yes, the implementation is gate-ready

- **Re-baseline timing matters.** D-04-13's gate is "median ≥ baseline + 0.3" against the post-Phase-4 baselines, not the current ones. The current baselines (commit `40fe3fd`) are pre-Phase-4 noise (category scorer abstains at 1.0). They get overwritten by step 8 before the gate is enforced.

</specifics>

<deferred>
## Deferred Ideas

- **DeepSeek decisiveness investigation.** Memory documents the 0/9 gap. Possible deep dive: raw LangGraph message trace from inside the agent loop (not eval reports), comparison of DeepSeek's tool-call format vs gpt-4o-mini's, examination of how the deepseek LangChain adapter passes tool schemas. Defer to v2.1 backlog unless DeepSeek becomes load-bearing for a future milestone.

- **Eval-harness multi-turn threading fix.** Memory `project_eval_multi_turn_threading_bug.md` documents that `evaluate_multi_turn_case` threads full message history including tool calls/results, while prod `/chat` threads only HumanMessage+AIMessage text. Phase 5 (RAT-02) will likely need to decide between (a) rewrite `evaluate_multi_turn_case` to mirror prod, (b) flag affected scenarios as eval-only shape, or (c) accept the caveat. Not Phase 4's call.

- **Update REQUIREMENTS.md + ROADMAP.md to reflect Phase 5 scope narrowing.** D-04-09 reduced Phase 5 from RAT-01 + RAT-02 to RAT-02 only. Documentation TODO that should be done as part of the Phase 4 PR (small doc-bookkeeping commit) so the project state matches reality.

- **`category_compliance_strict` keyword table expansion.** Starting set in <specifics> covers the 3 target eval scenarios. Adding new query patterns to the eval suite later will require expanding the keyword table. Defer to natural growth.

- **N-of-M pass rate (REQUIREMENTS NOJ-01).** Statistical rigor under temperature variance. Project memory `feedback_temp1_reasoning_off_all_models.md` says always temp=1.0. Deferred to v2.1 if Phase 4-6 PR gates show false-fail variance at N=3.

- **LLM-as-judge scorer for rationale quality (REQUIREMENTS JUD-01).** Add only if `category_compliance_strict` + `rationale_stop_alignment` together miss too many real failures. Requires a different model family for judge vs. agent (P3 bias mitigation). v2.1 if needed.

- **Per-PR ephemeral MLflow aliases (REQUIREMENTS OVR-07).** Phase 2's `RAG_MODEL_OVERRIDE` is sufficient for Phase 4 intake-LLM testing. Defer unless multi-PR concurrent testing emerges as a need.

- **Investigating other providers (Gemini 3.5 Flash, Kimi, Claude Haiku) for the intake LLM specifically.** Same-model-as-planner (D-04-02) is the locked decision; structured-output is well-supported on every OpenAI tier. Different intake provider would add complexity for marginal cost savings. Revisit only if intake cost becomes measurable in MLflow.

- **Closure-cascade scenario inclusion in Phase 4-6 gates.** Tracked but not gated for Phase 4 (D-04-12). Phase 5 (RAT-02) makes the long-term call.

- **Streaming / retry / rate-limit policies.** v2.0 out-of-scope per PROJECT.md.

</deferred>

---

*Phase: 04-Category Compliance Fix*
*Context gathered: 2026-05-22*
