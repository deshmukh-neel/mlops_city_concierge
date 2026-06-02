# Phase 6: Minimal-Edit Refinement - Context

**Gathered:** 2026-06-02
**Status:** Ready for planning

<domain>
## Phase Boundary

When the user issues a refinement turn ("make stop 2 cheaper", "swap stop 1
for something different", "stop 3 instead a dessert spot"), the agent must
edit ONLY the targeted stop — every other committed stop's `place_id` is
preserved byte-equal between turn 1 and turn 2, and only timing on stops
downstream of the edit shifts. The fix passes the committed plan
**structurally** to revision prompts (instead of the current text-only
history threading) behind a feature flag, with eval coverage that proves
first-turn behavior does not regress.

Concretely, Phase 6 delivers:

- `ConversationState.committed_stops: list[Stop]` round-tripped through
  `/chat`, mirroring how `closure_context` is round-tripped today (opaque
  to the frontend, validated server-side).
- In the `/chat` handler, when `committed_stops` is non-empty AND a
  deterministic refinement pre-check matches the incoming message
  ("make stop N", "stop N cheaper", "swap", "instead", "different"),
  prepend a single `HumanMessage` carrying a hybrid structured-plan
  payload (one-line prose preamble + fenced JSON block listing each
  committed stop's `place_id`, `name`, `primary_type`, `arrival_time`).
- A feature flag `REFINEMENT_STRUCTURED_PLAN_ENABLED` (env var, default
  OFF) gating the injection so first-turn behavior cannot drift until
  eval confirms no regression.
- A new deterministic scorer `refinement_minimal_edit` in
  `app/agent/critique/checks.py` that reads
  `state.scratch["prior_committed_stops"]` (populated by the eval runner
  after turn 1) and returns `preserved_byte_equal_count /
  expected_preserved_count`.
- An opt-in `EvalQuery.threading_mode: Literal["legacy", "prod"]` field
  + a `evaluate_multi_turn_case` branch that rebuilds messages per turn
  from text history + the injected structured-plan HumanMessage when set
  to `"prod"`. The `refinement_cheaper` scenario is migrated to
  `threading_mode: prod` and re-baselined; other multi-turn scenarios
  (e.g. `late_night_closure_cascade`) stay on `legacy` per D-04-12.
- A merge gate (D-06-09) enforcing `refinement_minimal_edit == 1.0` on
  the `refinement_cheaper` openai/gpt-4o-mini cell, plus REF-04
  first-turn no-regression on existing scorers against the
  `omakase_mission_open_ended` baseline.

Scope anchor: 4 requirements (REF-01 through REF-04) from
`.planning/REQUIREMENTS.md`. No scope folding from other phases; no
deferred items pulled forward.

</domain>

<decisions>
## Implementation Decisions

### Injection site (Gray Area #1)

- **D-06-01:** **`/chat` handler injection + extended `ConversationState`.**
  Add `committed_stops: list[Stop]` to `ConversationState` (Pydantic;
  defaulted to `[]` so legacy payloads decode unchanged). In `app/main.py`'s
  `chat()` handler, after the existing intake block (~line 731) and before
  constructing `ItineraryState.messages`, run a deterministic refinement
  pre-check on `req.message`. When the pre-check fires AND
  `incoming.committed_stops` is non-empty AND
  `REFINEMENT_STRUCTURED_PLAN_ENABLED` is true, prepend a single
  `HumanMessage` carrying the hybrid structured-plan payload (see D-06-04).
  Rationale: mirrors Phase 4's hybrid intake pattern (D-04-01..03); single
  injection site keeps the wire shape readable; backwards-compatible
  because legacy `conversation_state` payloads without
  `committed_stops` decode to an empty list and never trigger the
  injection.
- **D-06-02:** **Frontend round-trip is opaque** — same contract as
  `closure_context`. Backend stamps `conversation_state` in
  `_build_outbound_state` (extending it to include `committed_stops`
  derived from `final_state.stops` on every `/chat` response). Frontend
  treats the entire payload as opaque state in a `useRef` (matches the
  existing pattern documented in PR #94). No new frontend types or
  fields beyond the existing pass-through.
- **D-06-03:** **Refinement detection = deterministic regex pre-check**,
  not LLM intent-classifier. Living in `app/agent/input_parsing.py`
  alongside `has_slot_structure`. Pattern coverage: "make stop \d",
  "stop \d (cheaper|different|fancier|earlier|later)", "swap stop \d",
  "(instead|different) for stop \d", plus a few common paraphrases.
  Conservative on purpose — false negatives (missing a refinement) are
  cheaper than false positives (injecting the plan when the user changed
  topic). Documented + tested by the planner.

### Plan format in the prompt (Gray Area #2 / REF-02)

- **D-06-04:** **Hybrid format — prose preamble + fenced JSON block.**
  Single `HumanMessage` content shaped as:
  ```
  The following itinerary is currently committed. For any stop the
  user does not explicitly ask you to change, you MUST return the
  EXACT SAME `place_id` byte-for-byte when you call
  `commit_itinerary` next. Only the stop the user asks to change
  (and downstream timing) may differ. The user's next message
  follows immediately.

  ```json
  {{
    "current_plan": [
      {{"slot": 0, "place_id": "ChIJ...", "name": "...",
        "primary_type": "...", "arrival_time": "2026-05-21T19:00:00-07:00"}},
      ...
    ]
  }}
  ```
  ```
  Preamble carries the instruction; JSON carries the ground truth.
  Rationale: cross-provider robustness (P7) — JSON-attached-to-HumanMessage
  is well-handled by every provider in the matrix, and the prose preamble
  removes ambiguity about what the model should DO with the JSON
  (preserve `place_id` byte-equal). Pure JSON risks paraphrase on smaller
  models; pure natural-language risks place_id loss.

### Eval-harness threading parity (Gray Area #3)

- **D-06-05:** **Opt-in `EvalQuery.threading_mode: Literal["legacy", "prod"]`,
  default `"legacy"`.** `evaluate_multi_turn_case` in
  `scripts/eval_agent.py` branches: `legacy` keeps the current
  threading (state.messages threaded directly across turns,
  preserving existing baselines for every non-Phase-6 multi-turn
  scenario); `prod` rebuilds `ItineraryState.messages` per turn from
  the synthesized text history of prior assistant prose +
  `messages_from_history`-equivalent re-construction + the new
  structured-plan HumanMessage (mirrors `/chat` exactly).
- **D-06-06:** `configs/eval_queries.yaml` marks **only** the
  `refinement_cheaper` scenario as `threading_mode: prod`. Every other
  multi-turn scenario (today: `late_night_closure_cascade`) stays
  `legacy`. Rationale: `late_night_closure_cascade` is already gate-exempt
  per D-04-12 specifically because of this threading mismatch; flipping
  it now would conflate Phase 6's signal with the cascade scenario's
  pre-existing eval-shape dependency. v2.1 owns whether to migrate
  everything.
- **D-06-07:** **Re-baseline `refinement_cheaper` in the same PR as the
  implementation lands.** The existing baseline file
  (`configs/eval_baselines/refinement_cheaper.json`, all-1.0 via
  fail-open under the eval-shape threading) is invalidated the moment
  `threading_mode: prod` flips. Sequencing: implement → set flag ON
  locally → re-baseline → verify gate passes with new floor → merge.
  Same pattern Phase 4 used for `category_compliance_strict` (D-04
  discretion notes).

### Scorer + merge gate (Gray Area #4 / REF-01, REF-03, REF-04)

- **D-06-08:** **New deterministic scorer
  `refinement_minimal_edit(state) -> float` in
  `app/agent/critique/checks.py`.** Reads
  `state.scratch["prior_committed_stops"]` (a list of
  `{slot, place_id}` dicts, populated by `evaluate_multi_turn_case`
  immediately after turn 1's commit succeeds, before invoking turn 2)
  and `state.scratch["refinement_target_slot"]` (the slot the user
  asked to edit; populated by the same eval-runner step from
  `EvalQuery.expected_refinement.target_slot` — new YAML field). Returns
  `count(slot for slot in committed_stops if slot != target_slot AND
  current.place_id == prior.place_id) / count(slot for slot in
  committed_stops if slot != target_slot)`. 1.0 = every non-target
  stop preserved byte-equal. Partial credit for diagnostic value;
  merge gate enforces strict 1.0.
- **D-06-09:** **Phase 6 merge gate** (locked):
  - On scenario `refinement_cheaper`, provider `openai/gpt-4o-mini`,
    `threading_mode: prod`, with `REFINEMENT_STRUCTURED_PLAN_ENABLED=true`
    in CI for refinement scenarios:
    - `refinement_minimal_edit` median **== 1.0** (strict; REF-01 is
      binary by contract).
    - No regression (median ≥ baseline) on `category_compliance_strict`,
      `rationale_stop_alignment`, `geographic_coherence`,
      `walking_budget_respected`, `temporal_coherence`.
  - REF-04 first-turn no-regression: on `omakase_mission_open_ended`
    (single-turn) with `REFINEMENT_STRUCTURED_PLAN_ENABLED=false`,
    every Phase 4 scorer median ≥ Phase 4 post-merge baseline. This
    cell already runs in CI; Phase 6 adds the explicit assertion.
  - DeepSeek scorers logged but not gated (D-04-11 convention).
  - `late_night_closure_cascade` logged but not gated (D-04-12
    convention).
- **D-06-10:** **`REFINEMENT_STRUCTURED_PLAN_ENABLED` defaults OFF** at
  ship time. Env var read inside `chat()` per-request (OVR-05 pattern —
  never at module-load time; tests use `monkeypatch.setenv`). CI flips
  it on for refinement scenarios via the matrix YAML's per-cell env
  override (extend `EvalMatrixConfig.MatrixEntry` with optional
  `env: dict[str, str] | None = None`). Follow-up PR (post-Phase-6
  merge, post-prod observation) flips the default ON.

### Claude's Discretion (planner-level)

- Exact regex patterns for the refinement pre-check in
  `input_parsing.py` — pick patterns that catch `refinement_cheaper`
  ("make stop 2 cheaper") plus a few common paraphrases ("swap stop 1",
  "stop 3 instead", "different for stop 2"). Conservative; document +
  test.
- Exact JSON schema inside the structured-plan HumanMessage — minimal
  viable fields (`slot`, `place_id`, `name`, `primary_type`,
  `arrival_time`). Add fields only if eval shows the model needs them
  to disambiguate which stop is which.
- Exact wording of the preamble — must include "byte-for-byte" or
  equivalent unambiguous phrase about `place_id` preservation. Planner
  picks final wording.
- Whether to add a new `RevisionReason` literal like
  `"minimal_edit_violated"` so the post-commit critique loop can catch
  a model that drops the structured plan and re-plans from scratch.
  Nice-to-have; not on the merge gate. Planner decides based on
  whether early eval runs surface this failure mode.
- Whether to surface a `slot_index` arg on `commit_itinerary` so the
  model can explicitly mark which stop is the refinement target (mirror
  of Phase 4 D-04-05 pattern). Nice-to-have; the deterministic regex
  already extracts the target slot.
- Per-test layering per `feedback_test_layering.md`: at minimum a unit
  test for the pre-check regex, a unit test for `refinement_minimal_edit`
  scorer math, a functional test for the `/chat` handler injection
  branch (using scripted-LLM), and an integration assertion that the
  `refinement_cheaper` `threading_mode: prod` cell actually scores
  1.0 with the flag ON.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents (researcher, planner) MUST read these before planning or implementing.**

### Phase scope and requirements
- `.planning/ROADMAP.md` § "Phase 6: Minimal-Edit Refinement" — success criteria, branch name (`feature/v2-minimal-refinement`), discuss-phase mandate calling out the JSON-vs-natural-language decision
- `.planning/REQUIREMENTS.md` § "Minimal-Edit Refinement (Phase 6)" — REF-01..REF-04 (sole in-scope requirements)
- `.planning/PROJECT.md` § "Current Milestone: v2.0 Production Readiness" — milestone-level context; this phase closes out the v2.0 behavior-bug slate
- `.planning/research/PITFALLS.md` § P7 (single-model overfit), § P10 (refinement-turn fix breaks first-turn) — explicit mitigations live in REF-03 and REF-04

### Prior-phase decisions that constrain this phase
- `.planning/phases/04-category-compliance-fix/04-CONTEXT.md` § D-04-01..D-04-04 (hybrid intake pipeline pattern, graph-layer rewrites) — Phase 6 mirrors the hybrid pattern in `/chat`
- `.planning/phases/04-category-compliance-fix/04-CONTEXT.md` § D-04-11, D-04-12, D-04-13, D-04-14 — merge-gate conventions (openai/gpt-4o-mini only, DeepSeek + late_night_closure_cascade exempt, strict-floor reinterpretation when baseline is fail-open-saturated)
- `.planning/phases/03-eval-harness-extension/03-CONTEXT.md` — `EvalQuery.turns` shape that `threading_mode` extends; matrix YAML schema
- `.planning/phases/02-model-override/02-CONTEXT.md` (if present; else PROJECT.md "Key Decisions") — OVR-05 env-var-read-inside-function pattern that `REFINEMENT_STRUCTURED_PLAN_ENABLED` follows

### Baselines and eval infrastructure (extend, don't break)
- `configs/eval_baselines/refinement_cheaper.json` — current floor (all-1.0 via fail-open under legacy threading); MUST be re-baselined under `threading_mode: prod` in the same PR as Phase 6 lands
- `configs/eval_baselines/omakase_mission_open_ended.json` — REF-04 first-turn no-regression floor (do not re-baseline; Phase 4 set it)
- `configs/eval_queries.yaml` — `refinement_cheaper` (line 389) gets `threading_mode: prod` + `expected_refinement.target_slot: 1` (new field); `late_night_closure_cascade` stays `threading_mode: legacy`
- `configs/eval_matrix.yaml` — adds per-cell env override for the refinement row (REF-04: first-turn cells run with flag OFF)
- `scripts/eval_agent.py` § `evaluate_multi_turn_case` (line 557) — branches on `threading_mode`
- `scripts/eval_matrix.py` — consumes the new env-override field

### Production /chat plumbing (the surfaces Phase 6 changes)
- `app/main.py:218-228` § `ConversationState` Pydantic model — add `committed_stops: list[Stop] = Field(default_factory=list)`
- `app/main.py:498-606` § closure-path helpers + `_build_outbound_state` — extend to stamp `committed_stops` from `final_state.stops` on outbound payload
- `app/main.py:650-757` § `chat()` handler — new pre-`ItineraryState`-construction block reads `REFINEMENT_STRUCTURED_PLAN_ENABLED`, runs the refinement pre-check, prepends the structured-plan HumanMessage
- `app/agent/io.py:27-35` § `messages_from_history` — NOT modified (the structured-plan HumanMessage is prepended in `chat()`, not built inside `messages_from_history`, so `messages_from_history` stays a pure text-history mapper)
- `app/agent/input_parsing.py` — new `is_refinement_request(message: str) -> tuple[bool, int | None]` helper (returns matched + target slot index if extractable from the message)
- `app/agent/prompts.py` § `SYSTEM_PROMPT` — minor docstring addendum noting that a structured-plan HumanMessage may precede the user's turn-2 message; the model preserves `place_id` byte-equal for any stop not explicitly named. Do NOT add a `{current_plan}` template variable (D-06-04 routes the plan through a HumanMessage instead).

### Agent state and scorers
- `app/agent/state.py` § `Stop` (lines 148-162) — `place_id`, `arrival_time`, `name`, `primary_type` are the fields the structured plan carries
- `app/agent/state.py` § `ItineraryState.scratch` (existing dict) — eval runner stamps `prior_committed_stops` + `refinement_target_slot` here for the scorer
- `app/agent/critique/checks.py` § new `refinement_minimal_edit` scorer + `CRITIQUE_THRESHOLDS` + `DETERMINISTIC_CHECKS` registration
- `app/eval/config.py` § `EvalQuery` — add `turns_metadata: list[TurnMetadata] | None = None` (or extend the existing `turns: list[str]` to `turns: list[str | TurnMetadata]`; planner decides) with `target_slot: int` field; add `threading_mode: Literal["legacy", "prod"] = "legacy"`
- `app/eval/config.py` § `EvalMatrixConfig.MatrixEntry` — add `env: dict[str, str] | None = None` for per-cell env override

### Project memory — must-read before planning
- `project_eval_multi_turn_threading_bug.md` — the core driver of D-06-05/06; explains why prod and eval thread differently today and which scenarios are sensitive
- `project_aimessage_tool_call_args_json_safe.md` — the structured-plan HumanMessage content must serialize cleanly through the LangChain message pipeline; do not put Pydantic objects in `.content`
- `project_critique_commit_conflict.md` — Phase 6's preamble wording must NOT pull against `commit_itinerary`'s decisiveness directive; the preamble talks about WHICH stops to preserve, not WHETHER to commit
- `project_finalize_on_commit_fix.md` — Phase 6 changes do not alter the commit-termination path; agent still terminates on successful commit
- `feedback_temp1_reasoning_off_all_models.md` — eval runs temp=1.0 and reasoning-off uniformly; no per-cell tuning

### Reference docs
- `.github/copilot-instructions.md` / `AGENTS.md` / `CLAUDE.md` — keep all three in sync if the README "Refinement turns" section ships in this PR

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ConversationState` round-trip pattern (`app/main.py:218`, `_build_outbound_state`, frontend `useRef`) — established by closure_context in PR #94; Phase 6's `committed_stops` field plugs into the same flow with no new architectural surface
- `has_slot_structure` + `_intake_bind_kwargs` + `_SLOT_INTAKE_PROMPT_TEMPLATE` (Phase 4) — deterministic pre-check + gated LLM pattern. Phase 6's refinement pre-check follows the deterministic half of this pattern; no LLM call needed (regex is sufficient for the refinement-detection task — much narrower than slot extraction)
- `state.scratch` (`app/agent/state.py:ItineraryState.scratch`) — already used by eval runner to inject synthetic tool-error entries (memory `multi_turn_runner` pattern in `evaluate_multi_turn_case` line 630); Phase 6 reuses it for `prior_committed_stops` + `refinement_target_slot`
- `EvalQuery.turns` (Phase 3, EVAL-03) — adding `threading_mode` is a small Pydantic field addition; `turns_metadata` (or in-place extension) follows the same pattern
- `RAG_MODEL_OVERRIDE` (Phase 2) — Phase 6 does not add a new model-resolution path; the agent under refinement uses the same loaded model
- `MatrixEntry` Pydantic in `eval_matrix.py` (Phase 3, EVAL-04) — adding `env: dict[str, str] | None` for per-cell override is a small, backwards-compatible field addition

### Established Patterns
- **Hybrid pre-check + gated computation** (Phase 4 D-04-01..03) — deterministic pre-check FIRST so common cases pay zero added latency; LLM call (or in Phase 6's case, prompt injection) only when the pre-check fires
- **Round-trip-state-opaquely-via-conversation_state** (PR #94 closure_context) — frontend treats payload as opaque; backend builds + validates; Phase 6 follows this exactly
- **Env var read inside function, not at module-load time** (OVR-05) — `REFINEMENT_STRUCTURED_PLAN_ENABLED` is read inside `chat()` per-request so tests use `monkeypatch.setenv` and never `os.environ[...]`. Avoids the alias-caching-style pitfall on per-request flags
- **Scorer pattern `(state) -> float`** registered in `CRITIQUE_THRESHOLDS` + `DETERMINISTIC_CHECKS` — `refinement_minimal_edit` follows this; auto-runs via `itinerary_violations`
- **Strict-floor merge gate when baseline is saturated via fail-open** (D-04-14 reinterpretation) — D-06-09 applies the same logic: REF-01 is binary, so the merge gate is strict 1.0 rather than `+delta`
- **Per-cell env override pattern** — does not yet exist in `eval_matrix.py`; Phase 6 introduces it. Designed so future phases can run any cell with a custom flag without forking the runner

### Integration Points
- `/chat` handler (`app/main.py:650-757`) — single existing place where `ItineraryState.messages` is constructed; Phase 6's structured-plan HumanMessage is prepended here
- `evaluate_multi_turn_case` (`scripts/eval_agent.py:557`) — single existing place where multi-turn message threading happens; Phase 6 branches on `threading_mode`
- `_build_outbound_state` (`app/main.py` near closure helpers) — single existing place where ConversationState is built for the response; extended to stamp `committed_stops`
- `app/agent/critique/checks.py` registration tables — single existing place where scorers are wired into the matrix; `refinement_minimal_edit` plugs in here

</code_context>

<specifics>
## Specific Ideas

- The structured-plan HumanMessage preamble MUST contain an unambiguous
  phrase like "byte-for-byte" or "EXACT SAME `place_id`" so even
  paraphrase-prone models retain the contract. Wording will be finalized
  by the planner.
- Test-layering preference (`feedback_test_layering.md`) applies: every
  new surface (pre-check, scorer, threading_mode branch, /chat injection
  branch) gets unit + functional + at least one integration assertion.
- Small focused commits (`feedback_small_focused_commits.md`) — the
  planner should split this into roughly 5-7 plans (ConversationState
  field, refinement pre-check, /chat injection block, scorer,
  threading_mode branch, baseline + matrix YAML, README + sync docs).

</specifics>

<deferred>
## Deferred Ideas

- **Migrate all multi-turn eval scenarios to `threading_mode: prod`** —
  v2.1 work. Phase 6 only migrates `refinement_cheaper`. Migrating
  `late_night_closure_cascade` requires resolving the cascade-specific
  eval-shape dependency documented in `project_eval_multi_turn_threading_bug.md`
  (separate investigation).
- **New `RevisionReason` literal `"minimal_edit_violated"`** for the
  post-commit critique loop to catch a model that drops the structured
  plan and re-plans from scratch. Phase 6 ships without it; revisit if
  eval reveals this failure mode is common.
- **`slot_index` arg on `commit_itinerary`** so the model explicitly
  marks the refinement target. Phase 6 derives the target slot
  deterministically from the user's message; the explicit arg is a
  belt-and-braces add for v2.1 if needed.
- **Flip `REFINEMENT_STRUCTURED_PLAN_ENABLED` default ON** — follow-up
  PR after Phase 6 merges and prod observation confirms no first-turn
  regression. Not in Phase 6 scope.
- **DSPy-style automated prompt tuning of the preamble** — v2.1+. The
  Phase 6 preamble is hand-written; tuning it across providers is a
  separate research effort.

</deferred>

---

*Phase: 6-Minimal-Edit Refinement*
*Context gathered: 2026-06-02*
