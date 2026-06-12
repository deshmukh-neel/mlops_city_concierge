# Phase 13: Decisiveness Experiment Arms - Context

**Gathered:** 2026-06-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Four coupled experiment arms (DEC-01..04) are implemented behind flags, run at n=5
temp=1.0 against the Phase-12 comparison floor, and judged via the Phase-12 falsifier
(`make eval-falsifier`, exit-code 0/1/2 contract). The DEC-05 verdict document records
per-arm numbers for gpt-5-mini, deepseek-reasoner, and the gpt-4o-mini anchor — naming
the winning arm or an honest null result. Phase 14's conditional entry gate and Phase
15's promotion both read that document.

Nothing ships enabled-by-default to prod in this phase — prod promotion is Phase 15.

</domain>

<decisions>
## Implementation Decisions

User delegated all four discussed areas to Claude ("do ur best"); D-13-01..09 are
Claude's calls. Planner may refine names/details but must preserve the substance.

### Arm structure & run budget
- **D-13-01: Three judged arms, not four independent ones.**
  - **A1 = viability contract + critique recalibration** (DEC-01 + DEC-03 as ONE
    co-tuned arm — the `critique-loop-and-commit-tool-conflict` memory locks them
    together; never tune either in isolation).
  - **A2 = forced-commit-at-step-N** (DEC-02), graph-level and model-independent.
  - **A3 = parallel tool execution** (DEC-04) — the latency arm; judged on measured
    latency reduction + zero scorer regression, NOT on commit rate.
  - **A4 (conditional combo, A1+A2):** run ONLY if neither A1 nor A2 alone clears the
    falsifier but both show positive signal. Hard cap: ≤4 full live matrix runs total.
- **D-13-02: Per-arm run budget.** Each judged arm runs 3 models (gpt-5-mini,
  gpt-4o-mini anchor, deepseek-reasoner) × 2 scenarios × n=5, temp=1.0, sequential
  (D-11-14). The arm scenario universe is **omakase_mission_open_ended +
  refinement_cheaper** — the default `configs/eval_matrix.yaml` is omakase-only, and an
  omakase-only falsifier is vacuous per D-12-08 (gpt-5-mini already 1.0 there). Use a
  committed arm matrix config (or a recorded `SCENARIOS=` override) so the falsifier
  run-dir mode pools both scenarios. `late_night_closure_cascade` stays quarantined
  (D-10-09); anthropic/gemini cells stay deferred (D-12-09) and are NOT run.
  Smoke n=1 sanity run per arm before committing to the full n=5 spend. No billing
  top-ups: if quota/budget dies mid-arm, record the partial honestly in the verdict doc
  and stop — never write partial-cell baselines (D-11-14).

### Forced-commit arm (DEC-02)
- **D-13-03: Mechanism.** At step N (env `FORCED_COMMIT_STEP`, default 6 when arm is
  on, unset/0 = disabled; max_steps is 8), if the model has not called
  `commit_itinerary` AND every requested stop has ≥1 viable candidate (Phase-12
  viability definition: cosine ≥ `LOW_SIMILARITY_THRESHOLD` + matching `primary_type`
  — same semantics as the rule-8 harness machinery, computed graph-side from scratch
  entries), the graph constructs a synthetic `commit_itinerary` call from best-so-far
  (highest-cosine viable candidate per slot) and routes it through the NORMAL commit
  path — place_id validation, `critique_final_with_stops`, finalize-on-commit. If any
  slot lacks a viable candidate at step N, do NOT force: fall through to the existing
  max-steps path and record the skip. This is distinct from `short_circuit_max_steps`,
  which finalizes WITHOUT a commit (caveat text) at the ceiling; forced-commit produces
  a real committed itinerary, earlier, through the commit machinery.
- **D-13-04: Honesty contract — forced commits cannot silently game the falsifier.**
  Every run JSON records `commit_forced: bool` and `forced_commit_step` (telemetry
  fields, NOT new scorers — anti-scope). `committed_itinerary_rate` counts forced
  commits (product-honest: the user receives a committed plan), BUT (a) the DEC-05
  verdict MUST report the model-initiated vs forced split per model, (b) the A2 verdict
  line must carry that split annotation explicitly, and (c) A2 only counts as clearing
  the falsifier if quality scorers (category_compliance etc.) hold ≥ baseline and the
  anchor is behaviorally unchanged (gpt-4o-mini commits before step 6 on its own, so
  any anchor change is a red flag). The required DEC-02 unit test: a mock model that
  never calls `commit_itinerary` triggers the forced commit.

### Arm toggling & prod surface
- **D-13-05: All arms behind env flags, default OFF; prod untouched in Phase 13.**
  - A1: `VIABILITY_CONTRACT_ENABLED` — one flag selects BOTH the rule-8 viability text
    variant AND the critique recalibration (sharing one flag mechanically enforces
    co-tuning). The threshold value itself comes from a `LOW_SIMILARITY_THRESHOLD` env
    override (code default stays 0.55).
  - A2: `FORCED_COMMIT_STEP` (int; unset/0 = off).
  - A3: `PARALLEL_TOOL_EXECUTION_ENABLED`.
  - Flag parsing follows the `REFINEMENT_STRUCTURED_PLAN_ENABLED` precedent
    (`app/main.py:753-758` truthy-set). Planner decides the read point (per-request vs
    `build_agent_graph` time), but every eval run JSON must self-describe its arm
    config (mirroring the D-12-04 `viability_threshold` field precedent) so run dirs
    are unambiguous. Winners promote to defaults in Phase 15, not here.

### Viability contract + critique recalibration (DEC-01 + DEC-03)
- **D-13-06: DEC-01 is additive-only to the prompt.** Rule 8 gains an explicit
  viability definition ("a result with cosine ≥ {threshold} and matching primary_type
  IS viable — do not keep searching past it"). Both flag states must keep every pinned
  phrase green: `test_system_prompt_has_decisive_commit_contract` ("one viable option",
  "do not keep"), all other `tests/unit/test_agent_prompts.py` locks, and the
  `tests/unit/test_agent_io.py` forbidden-phrase preamble gate (no behavioral-rubric
  language: "byte-for-byte", "SAME primary_type", etc.). The Phase-7 CI grep gate
  staying green is a merge condition (roadmap success criterion 1).
- **D-13-07: DEC-03 candidate changes (inside A1 only):** (a) lower
  `LOW_SIMILARITY_THRESHOLD` below 0.55 — direction locked by seed finding 2 (it
  already IS 0.55; "set to 0.55" is a no-op), and/or (b) scope `low_similarity` hints
  to pre-candidate steps only — stop firing once every requested stop has a viable
  candidate (the same rule-8 machinery DEC-02 uses). Per roadmap criterion 4, the
  chosen direction AND the scoping decision must be documented in the plan BEFORE any
  threshold change lands. Runs stay self-describing via the recorded
  `viability_threshold` field.

### Parallel tool execution (DEC-04)
- **D-13-08: Concurrency inside one `act()` plan step only.** All tool calls within a
  step run concurrently (asyncio.gather or equivalent); results are appended in the
  ORIGINAL tool_call order regardless of completion order (roadmap criterion 3:
  order-stable). Scratch/telemetry writes stay deterministic. DB pool behavior must be
  verified under concurrency — known full-suite pool-contamination risk; full
  `make test` is mandatory, not just the changed files. The INST-04
  `tool_execution_seconds` telemetry captures the reduction; the measurable gpt-4o-mini
  latency reduction at n=5 is recorded in run JSON. The `commit_itinerary` branch and
  ordinary tool branch interactions need care (commit short-circuits the step today).

### Verdict document (DEC-05)
- **D-13-09: `docs/decisiveness_arm_verdicts.md`.** Per arm: flag config, committed
  matrix config + run-dir paths, per-model pooled commit rate with per-scenario
  breakdown, model-initiated vs forced split (A2), latency decomposition deltas (A3),
  falsifier verdict line + exit code, and one explicit closing line: which arm (if any)
  cleared the INST-05 bar — or the honest null result. This doc is the Phase-14
  conditional entry gate input and the Phase-15 promotion input.

### Claude's Discretion
User delegated run budget, forced-commit semantics/honesty, toggling strategy, and
combination/verdict structure entirely ("u got it fable, do ur best king"). All
D-13-01..09 are Claude's calls; planner may refine exact env-var names, file names,
and the default N, but must preserve: co-tuned A1, forced-commit transparency split,
flags-off-by-default prod surface, two-scenario pooling universe, and the ≤4 live
matrix run cap.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Milestone framing (settled — do not re-derive)
- `.planning/milestones/v2.2-MILESTONE-SEED.md` — load-bearing findings 1-6 (critique
  loop NOT the gpt-5-mini driver; threshold already 0.55; finalize-on-commit works;
  latency = call-count × per-call cost; replay gaps are Phase-14 material)
- `.planning/REQUIREMENTS.md` — DEC-01..05 definitions + anti-scope
- `.planning/ROADMAP.md` — Phase 13 success criteria 1-5
- `.planning/phases/12-decisiveness-instrumentation-comparison-floor/12-CONTEXT.md` —
  D-12-01..09 (telemetry capture, viability semantics, falsifier pooling, deferrals)

### Falsifier + eval harness (judging machinery — consume, don't duplicate)
- `scripts/eval_falsifier.py` — exit-code 0/1/2 contract; run-dir mode; zero-overlap
  exit-2 guard (arm run dirs must overlap the baseline scenario universe)
- `scripts/eval_agent.py` — INST field assembly (`first_commit_call_step`,
  `viable_candidates_per_step`, `rule8_met_per_step`, `viability_threshold`);
  `commit_forced` / `forced_commit_step` join this record
- `scripts/eval_matrix.py` + `configs/eval_matrix.yaml` — cell fan-out; default
  scenarios are omakase-only (arm configs must add refinement_cheaper)
- `configs/eval_gates.yaml` + `scripts/check_eval_gates.py` — Phase-10 honest gates,
  D-11-20 omakase/refinement split rationale
- `scripts/write_baselines.py` + `configs/eval_baselines/` — baselines ONLY via this
  tool (D-11-14); `_DEFERRED_BASELINE_CELLS` (anthropic + gemini stay deferred)
- `docs/baseline_regen.md` — deferred-cell promotion path
- `docs/eval_gates.md` — gate semantics documentation

### Agent loop (implementation target)
- `app/agent/graph.py` — plan/act loop; sequential tool execution (~340-460, DEC-04
  target); `step_telemetry` hooks (DEC-02 fields extend these); max_steps=8 routing
  (~496, DEC-02 inserts before this); commit branch short-circuit inside `act()`
- `app/agent/prompts.py` — SYSTEM_PROMPT rule 8 (~lines 151-161, DEC-01 target);
  REVISION_GUIDANCE low_similarity bullet (DEC-03 scoping interacts)
- `app/agent/revision.py` — `LOW_SIMILARITY_THRESHOLD` (line 21), `_diagnose_one`
  low_similarity firing (~122-167, DEC-03 scoping target)
- `app/agent/io.py` — `_REFINEMENT_PREAMBLE` (line 103); must NOT gain behavioral text
- `app/main.py:753-758` — env-flag parsing precedent (`REFINEMENT_STRUCTURED_PLAN_ENABLED`)

### Prompt-lock gates (DEC-01 hard constraints)
- `tests/unit/test_agent_prompts.py` — `test_system_prompt_has_decisive_commit_contract`
  + sibling locks; both flag states must pass
- `tests/unit/test_agent_io.py` — `test_preamble_describes_task_not_behavior`
  forbidden-phrase list (D-07-04 grep gate)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase-12 telemetry: `step_telemetry` on `ItineraryState` + INST helpers in
  `eval_agent.py` — DEC-02's `commit_forced`/`forced_commit_step` and DEC-04's latency
  deltas extend these existing fields, same assembly path
- Rule-8 viability machinery (`rule8_met_per_step_from_state`, viable-candidate
  counting) — DEC-02's "every stop has a viable candidate" gate and DEC-03's
  pre-candidate scoping reuse the same definition; keep one source of truth
- `eval_falsifier.py` exit codes — DEC-05 verdicts consume them mechanically
- `short_circuit_max_steps` + `critique_final_with_stops` routing — DEC-02 slots in as
  a new branch before the max-steps check, reusing the commit path

### Established Patterns
- Env-flag pattern: truthy-set parsing per `REFINEMENT_STRUCTURED_PLAN_ENABLED`
- JSON-safe state: plain int/float/str only in `ItineraryState` fields (Pydantic in
  tool-call args crashed plan() once — never again)
- Test layering: unit/mock + smoke + functional per new module; full `make test`
  required (DB-pool contamination risk with real-graph tests)
- Stale-baseline CI gate: `app/agent/` changes trip it; behavior-changing arm code
  merged flagged-OFF warrants `[skip-baseline]` ONLY if default-path behavior is
  byte-identical with flags off — otherwise refresh baselines
- Bookkeeping commits (ROADMAP/REQUIREMENTS status flips) go straight to main

### Integration Points
- `act()` tool-execution loop (`graph.py` ~340-460) — DEC-04 gather + DEC-02 forced
  branch both land here
- `route` after act (`graph.py` ~492-500) — DEC-02 fires before the max_steps check
- `SYSTEM_PROMPT` rule 8 + `REVISION_GUIDANCE` — DEC-01 variant text
- `query_result_from_state` (`eval_agent.py`) — new arm fields join the run record

</code_context>

<specifics>
## Specific Ideas

- Falsifier already prints per-scenario breakdowns — arm verdicts should paste those
  directly rather than recompute
- Arm run dirs must satisfy the falsifier's zero-overlap exit-2 guard: scenario IDs in
  arm matrix configs must match committed baseline scenario IDs exactly
- A2's verdict line format suggestion: `commit_rate 0.8 (model-initiated 0.4, forced
  0.4)` — the split visible at a glance

</specifics>

<deferred>
## Deferred Ideas

- **Best-combination arm beyond A1+A2** (e.g., A1+A2+A3 stacked) — only the single A4
  combo is sanctioned this phase; further stacking is Phase-15 material if promotion
  wants it
- **Richer state replay** — Phase 14, conditional on all arms plateauing below the
  falsifier bar (entry gate reads `docs/decisiveness_arm_verdicts.md`)
- **Anthropic/gemini n=5 baselines** — remain deferred (D-12-09); arm runs exclude
  those cells entirely

</deferred>

---

*Phase: 13-Decisiveness Experiment Arms*
*Context gathered: 2026-06-12*
