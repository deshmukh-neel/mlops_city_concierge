# Phase 12: Decisiveness Instrumentation + Comparison Floor - Context

**Gathered:** 2026-06-11
**Status:** Ready for planning

<domain>
## Phase Boundary

The eval harness emits per-run decisiveness telemetry (INST-01..04) and the milestone
falsifier becomes a single executable report (INST-05). This is measurement
infrastructure — every Phase 13 experiment arm is judged against what this phase builds.

**Scope change at discuss time (user decision 2026-06-11):** ANCH-02 (gemini n=5
baseline) is DEFERRED for now — user declined quota/billing top-up, same treatment as
the anthropic ANCH-01 deferral. Phase 12 completes on INST-01..05 alone. The v2.2
comparison floor is the matrix minus BOTH deferred cells (anthropic + gemini); both keep
their `_DEFERRED_BASELINE_CELLS` entries with deferral notes. If quota resolves at no
cost mid-milestone, the gemini baseline can run opportunistically, but nothing blocks
on it. ROADMAP.md success criterion 4 and the ANCH-02/ANCH-03 wording in
REQUIREMENTS.md predate this decision — this CONTEXT.md supersedes them; planner should
NOT create a gemini-baseline plan as a phase-completion requirement.

</domain>

<decisions>
## Implementation Decisions

### Telemetry capture point (INST-01..04)
- **D-12-01: Hybrid capture, minimal prod surface.** Per-step facts only the graph can
  observe — per-step LLM-call wall time, per-step tool-execution wall time, tool calls
  per step — are recorded in-graph into `ItineraryState` (a lightweight `step_telemetry`
  list of timestamps/counts appended in `plan()`/`act()`). It is cheap enough to stay
  always-on in prod. Derived/judgmental metrics — steps-to-first-commit-consideration,
  viable-candidate counts, rule-8 precondition flags — are computed harness-side in
  `scripts/eval_agent.py` from message history + `step_telemetry` at result-assembly
  time, and written as first-class fields into the run JSON. Eval semantics (viability
  judgments) must NOT live in prod graph code.
- **D-12-02: "No post-processing" interpretation.** The success criterion "readable in
  the run JSON without post-processing" means the *consumer* of a run JSON reads the
  fields directly; it does not prohibit `eval_agent.py` computing them while assembling
  the result. INST-04 per-step latency decomposition genuinely cannot be reconstructed
  post-hoc — that is why it needs in-graph timing (D-12-01).

### Commit-consideration + viability semantics (INST-01/02/03)
- **D-12-03: Record both signals; strict is primary.** Run JSON gets
  `first_commit_call_step` (step index of the first actual `commit_itinerary` tool
  call; null if never) as the primary, objective, cross-provider-comparable INST-01
  metric. A best-effort secondary `first_commit_mention_step` (first step whose visible
  content/reasoning text references committing) is recorded where reasoning text is
  visible, null where it is opaque (gpt-5 encrypted reasoning, Gemini signatures) —
  documented as heuristic, never used for gating.
- **D-12-04: Viability bar reuses the prod constant, parameterized in output.** Viable
  = cosine ≥ `LOW_SIMILARITY_THRESHOLD` (imported from `app/agent/revision.py`, do NOT
  hardcode 0.55) AND `primary_type` matches the requested stop category. Every run JSON
  records the threshold value used (e.g., `viability_threshold` field) so runs are
  self-describing when Phase 13 DEC-03 lowers the threshold — telemetry follows
  automatically.
- **D-12-05: Rule-8 precondition flag (INST-03).** At each step where the model kept
  searching, compute whether every requested stop already had ≥1 viable candidate
  (per D-12-04 definition). Emit the per-step boolean sequence plus a summary field
  (e.g., `rule8_met_but_kept_searching_steps`) in the run JSON.

### Falsifier report mechanics (INST-05)
- **D-12-06: Report over existing artifacts, never a live run.** `make eval-falsifier`
  reads an eval_reports matrix run dir (latest by default, `--run-dir` to override) and
  does NOT fan out live API calls. Workflow stays two-step: `make eval-matrix` (existing
  target, expensive, live) → `make eval-falsifier` (cheap, repeatable report).
- **D-12-07: New thin script reusing existing machinery.** Implement as
  `scripts/eval_falsifier.py` (or similar), reusing `eval_matrix.py` cell-JSON
  aggregation helpers and `check_eval_gates.py` baseline/gate-reading machinery rather
  than duplicating either. Output: per-model numbers + explicit PASS/FAIL line; exit
  code 0/1 so Phase 13 can consume it mechanically.
- **D-12-08: Falsifier commit-rate scope = pooled across scored scenario cells.**
  gpt-5-mini currently splits sharply by scenario (omakase committed_itinerary_rate
  median 1.0 vs refinement 0.0 per eval_gates.yaml D-11-20 rationale), so scenario
  scope decides pass/fail. Decision: the ≥0.6 bar applies to the committed-itinerary
  rate pooled across ALL scored scenario cells in `configs/eval_matrix.yaml` for the
  target model, with a per-scenario breakdown always printed. An omakase-only falsifier
  would already pass at baseline and make the milestone vacuous; pooling keeps it
  meaningful (~0.5 today). Anchor non-regression (gpt-4o-mini ≥ honest baseline) is
  checked per-metric against `scripts/eval_baselines/` via the existing gate machinery.
  Flag for researcher/planner: validate pooling against the actual matrix config cell
  structure; if the planner finds the pooled-n semantics conflict with how baselines
  are keyed, surface it in the plan rather than silently re-interpreting.

### Comparison floor / gemini (ANCH-02, ANCH-03)
- **D-12-09: Gemini n=5 baseline DEFERRED (user decision, 2026-06-11).** No quota or
  billing top-up for now. Gemini joins anthropic in `_DEFERRED_BASELINE_CELLS` with a
  deferral note; both stay logged-not-gated. ANCH-03 is reinterpreted as: all matrix
  cells except the two deferred cells (anthropic, gemini) are honest n=5 — which is
  already true after Phase 11 for the openai pair + deepseek pair. Phase 12 ships no
  gemini-baseline plan; revisit when quota/budget allows (promotion path stays in
  `docs/baseline_regen.md`).

### Claude's Discretion
User delegated telemetry capture point, commit-consideration semantics, and falsifier
mechanics entirely to Claude ("u decide for me"). D-12-01..08 are Claude's calls —
planner may refine details (exact field names, file names) but must preserve the
decisions' substance: hybrid capture, strict-primary commit signal, imported threshold
constant, artifact-reading falsifier with pooled scope, exit-code contract.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Milestone framing (settled — do not re-derive)
- `.planning/milestones/v2.2-MILESTONE-SEED.md` — load-bearing findings from the
  2026-06-10 harness analysis (critique loop not the gpt-5-mini driver;
  LOW_SIMILARITY_THRESHOLD already 0.55; finalize-on-commit works; latency =
  call-count × per-call cost); decisions 1-3 already made
- `.planning/REQUIREMENTS.md` — INST-01..05 / ANCH-02..03 definitions, anti-scope table
- `.planning/ROADMAP.md` — Phase 12 success criteria (criterion 4 superseded by D-12-09)

### Eval harness (instrumentation target)
- `scripts/eval_agent.py` — per-run record assembly (`QueryEvalResult`,
  `query_result_from_state`, `count_tool_calls`, `revision_reasons_from_state`); where
  INST-01/02/03 derived fields get computed and written
- `scripts/eval_matrix.py` — cell fan-out, `aggregate_cell_jsons`, run-dir resolution;
  helpers the falsifier report reuses
- `scripts/check_eval_gates.py` + `configs/eval_gates.yaml` — gate schema (D-10-08),
  current per-family statuses, D-11-20 rationales with the omakase/refinement split data
- `scripts/write_baselines.py` + `scripts/eval_baselines/` — honest-baseline write path
  (D-11-14: refuses partial/quarantined); `_DEFERRED_BASELINE_CELLS` lives in this flow
- `configs/eval_matrix.yaml` — the scored cells that define falsifier pooling scope
- `docs/baseline_regen.md` — deferred-cell promotion path (anthropic, now also gemini)

### Agent loop (in-graph telemetry target)
- `app/agent/graph.py` — plan/act loop (one LLM call per plan step; sequential tool
  execution ~lines 340-402; reasoning-state replay 307-312; `_prune_for_llm` str()
  collapse line 227); where `step_telemetry` recording hooks in
- `app/agent/revision.py` — `LOW_SIMILARITY_THRESHOLD` (line ~21) — the viability
  constant to import, never hardcode

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `eval_agent.py` already records `tool_calls`, `latency_seconds` (total only),
  `revision_reasons`, `tool_names` per run — new INST fields extend this record, same
  assembly path
- `eval_matrix.py` `aggregate_cell_jsons` + `_scorer_means_from_cell` +
  `_parse_cell_filename` — the falsifier report should consume these, not reimplement
- `check_eval_gates.py` already evaluates `committed_itinerary_rate` hard gates against
  `eval_gates.yaml` — the anchor non-regression half of the falsifier is close to free
- `Makefile` eval targets (`eval-matrix`, `eval-gates-check` pattern) — `eval-falsifier`
  follows the same shape

### Established Patterns
- Run JSONs per cell in timestamped `eval_reports/` dirs; aggregation reads cell files
  by filename convention — new telemetry fields must serialize into the same cell JSONs
- Gates are single-source-of-truth in `configs/eval_gates.yaml` with decision-ID
  rationales — falsifier thresholds (0.6, anchor floor) should reference INST-05, not
  duplicate gate YAML semantics
- Tests: unit/mock + smoke + functional layering expected for new modules (user's
  standing test-layering preference); conftest patches env so tests never hit real
  services; full `make test` required (DB-pool contamination risk with real-graph tests)

### Integration Points
- `ItineraryState` (graph state) — new `step_telemetry` field; must stay JSON-safe
  (prior incident: Pydantic objects in tool-call args crashed the next plan step)
- `plan()`/`act()` in `app/agent/graph.py` — timing hooks around the LLM call and the
  sequential tool-execution block
- `query_result_from_state` in `eval_agent.py` — where derived INST fields join the
  run record

</code_context>

<specifics>
## Specific Ideas

- Falsifier output should answer the milestone question in one line per model: did
  gpt-5-mini hit ≥0.6 pooled commit rate at n=5, did gpt-4o-mini hold ≥ its honest
  baseline — PASS/FAIL with the numbers, exit code 0/1.
- Phase 13 DEC arms will diff telemetry before/after — field names should be stable and
  self-describing (record `viability_threshold` in every run).

</specifics>

<deferred>
## Deferred Ideas

- **ANCH-02 gemini n=5 baseline** — deferred at discuss time (no quota/billing top-up;
  user decision 2026-06-11). Joins ANCH-01 anthropic in deferred-cell status. Revisit
  when budget allows; single scored gemini run already hit commit-rate 1.0 (first
  evidence the Phase-9 adapter fixed it), so this is measurement debt, not unknown risk.
- **ROADMAP.md / REQUIREMENTS.md bookkeeping** — Phase 12 success criterion 4 and
  ANCH-02/03 wording need a docs-only amendment to reflect the gemini deferral (user's
  convention: bookkeeping commits go straight to main).

</deferred>

---

*Phase: 12-Decisiveness Instrumentation + Comparison Floor*
*Context gathered: 2026-06-11*
