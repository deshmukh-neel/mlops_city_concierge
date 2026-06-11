# Phase 10: Eval Harness Honesty - Context

**Gathered:** 2026-06-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 10 makes the eval harness honest: infrastructure failures become ERROR records
instead of fake scores, only prod-shaped conversation behavior gets measured, merge
gates are re-derived to values the honest anchor data can actually satisfy, and the
synthetic-vs-live test gap gets a standing mitigation (live-probe gate + recorded
real-wire fixtures). Phase 10 ships **mechanism, not measurements** — the wholesale
baseline regen (including anthropic n=5 + gemini first n=5) is Phase 11 BASE-01..04.

**In scope (EVAL-01..06):**
- Error-status semantics in `scripts/eval_agent.py` + `scripts/eval_matrix.py` +
  `app/agent/critique/checks.py` interplay — closes the three fail-open paths proven by
  `eval_reports/2026-06-05T21-14-30Z/` (EVAL-01).
- `late_night_closure_cascade` threading decision: quarantine from baselines/gates
  (EVAL-02, decided below).
- Gate re-derivation + machine-readable gate file + executable gate-check Make target
  (EVAL-03).
- Baseline↔matrix parity verification/extension (EVAL-04 — initial test shipped PR #104).
- Per-provider live-probe Make target + checked-in real-wire fixtures + redaction
  (EVAL-05).
- Sync/async test-debt closure: gpt-5 factory dispatch tests, `ScriptedChatModel`
  ainvoke coverage, `vibe_check` blocking-call resolution (EVAL-06).

**Out of scope:**
- Running any live n=5 matrix or regenerating any baseline JSON — Phase 11 BASE-01.
- Adding cells to `configs/eval_matrix.yaml` (cross-model expansion) — Phase 11 BASE-02.
- Promoting any gate (incl. the conformance harness marker) to CI — Phase 11 BASE-03
  per D-09-10/D-08-14 precedent (local empirical, CI structural).
- Staleness-check extension for new baselines — Phase 11 BASE-04.
- Anything that changes agent behavior: prompts, critique thresholds
  (`LOW_SIMILARITY_THRESHOLD`), commit contract, `_prune_for_llm`, adapters — v2.2
  decisiveness territory (see `.planning/v2.2-MILESTONE-SEED.md`). The ONLY agent-code
  edit allowed is the EVAL-06 `vibe_check` async fix, which is behavior-preserving.
- New scorers or scorer-rubric changes beyond the error-path handling.

</domain>

<decisions>
## Implementation Decisions

User delegated all four discussed areas to Claude's recommendations (2026-06-10,
"I'll just go with whatever u recommend"). The decisions below are therefore locked
with the same force as user-stated ones.

### ERROR-Run Semantics (EVAL-01)

- **D-10-01: Errored runs produce a run JSON with `"status": "error"`** — never a
  scored record, never silence. Schema: `status: "ok" | "error"` on every per-run JSON;
  error runs add `error: {stage: "turn0" | "turnN" | "setup", type: "<ExceptionClass>",
  message: "<truncated>"}`. Scored fields (checks, scores) are absent/null on error
  runs. Rationale: auditability — the 21-14-30Z post-mortem was only possible because
  *something* was written per run.
- **D-10-02: Any exception during a turn = ERROR; no exception-type allowlist.**
  Model failures are not exceptions — they are low scores on completed runs. Exceptions
  in this harness are infra/config by definition (429 quota, DB down, 400 config bug).
  The recorded `stage` + `type` enable later classification; an allowlist would just be
  a second thing to keep honest. The current `partial_state` scoring-on-exception path
  in `_run_prod_threading` (`scripts/eval_agent.py:839-863`) is REMOVED — exceptions
  never reach scorers.
- **D-10-03: Cell validity rule — a cell with any errored run is
  `INVALID_FOR_BASELINE`.** `summary.json` gains per-cell `n_scored` / `n_errored` and
  a top-level `errors` array; medians over the scored subset are still reported
  (diagnostic value) but flagged invalid. Any future baseline writer (Phase 11) MUST
  refuse cells with `n_scored < n_requested`. Matrix exit code: non-zero when any
  errors occurred, with error count distinct from score-violation count in output.
- **D-10-04: Acceptance test = the 21-14-30Z replay.** A unit/functional test simulates
  the failure conditions (turn-0 LLM exception; turn-1 exception; retrieval-only
  exception) and asserts all produce ERROR records, zero scored cells, and that the
  former Branch-1-abstain-1.0 / prior-vs-itself-1.0 / retrieval-0.0 asymmetry outcomes
  are gone. The scorer's Branch-1 abstain (`checks.py:488-491`) is retained ONLY for
  its legitimate purpose (non-refinement scenarios where `refinement_context` is
  genuinely absent on a *completed* run).

### Gate Derivation & Storage (EVAL-03)

- **D-10-05: Single source of truth is machine-readable —
  `configs/eval_gates.yaml`.** Consumed by a new `scripts/check_eval_gates.py`
  (Make target `eval-gates-check`) that takes a matrix `summary.json` and exits
  non-zero on any hard-gate violation. `docs/eval_gates.md` explains semantics and
  links to the YAML; it never duplicates numbers. Rationale: docs-with-hardcoded-
  Makefile numbers is how the unsatisfiable strict-1.0 gate fossilized.
- **D-10-06: Gate shape — extend the D-09-02 two-part pattern to ALL families.**
  Hard gates ride on `committed_itinerary_rate` floors; `refinement_minimal_edit`
  medians are advisory/logged everywhere until v2.2's decisiveness work earns more.
  The Phase-6-era strict `refinement_minimal_edit == 1.0` anchor gate is **formally
  retired** (it was a fail-open-baseline artifact; the honest anchor sits at median
  0.0 / max 0.5 post-D-07-05/D-07-07).
- **D-10-07: Provisional hard-gate values (from existing honest data; Phase 11
  re-ratifies after fresh baselines):**
  - `openai/gpt-4o-mini` (anchor): commit_rate ≥ 0.8 (observed 1.0 in every clean
    matrix; 0.8 absorbs one stochastic miss at n=5 without going flaky).
  - `openai/gpt-5-mini`: commit_rate ≥ 0.6 — D-09-02 Part A retained verbatim;
    currently FAILS at 0.4 and is expected to fail until v2.2; the gate file marks it
    `status: aspirational` so `eval-gates-check` reports it distinctly rather than
    hard-failing Phase-10/11 work on a known gap.
  - `anthropic/claude-sonnet-4-6`: commit_rate ≥ 0.8 `status: provisional-n1` until
    Phase 11's n=5 lands.
  - `deepseek/deepseek-reasoner`, `deepseek/deepseek-chat`,
    `gemini/gemini-3.1-pro-preview`: logged-not-gated.
  - Non-regression: every cell's advisory medians are compared against its baseline
    JSON cell; a drop beyond tolerance is a WARN (not hard-fail) in Phase 10, with
    hard-fail promotion decided in Phase 11 BASE-03.
- **D-10-08:** The gate YAML records, per cell: `hard` (metric, op, value), `advisory`
  list, `status` (active | aspirational | provisional-n1 | logged), and a `rationale`
  one-liner with the D-ID. This is what `docs/eval_gates.md` renders.

### late_night Threading Fate (EVAL-02)

- **D-10-09: Quarantine now; migrate later (NOT in Phase 10).**
  `late_night_closure_cascade` keeps `threading_mode: legacy` and gains an explicit
  `baseline_eligible: false` flag (scenario YAML/config + honored by the matrix
  runner and any baseline tooling). It stays runnable as a diagnostic; it is excluded
  from Phase 11 regen and from all gates. Rationale: migrating it to prod threading
  changes what the scenario measures — the closure-cascade turn-2 scorers were
  designed against the full-tool-history shape (memory
  `project_eval_multi_turn_threading_bug`), and redesigning the scenario is scope
  creep on a harness-honesty phase. The migration/redesign is deferred (see
  `<deferred>`). `omakase_mission_open_ended` is single-turn and unaffected.
- **D-10-10:** The quarantine decision is recorded in three places: the scenario
  config comment, `configs/eval_gates.yaml` (cell `status: quarantined-legacy-threading`),
  and `docs/eval_gates.md`. The late_night baseline JSON is NOT regenerated and NOT
  deleted — it is annotated (`_observations`) as legacy-threading-shaped, measurement
  not comparable to prod.

### Live-Probe & Fixtures (EVAL-05)

- **D-10-11: Full-fidelity probes that produce checked-in fixtures.** One generalized
  `scripts/probe_provider_capture.py --provider {openai|deepseek|anthropic|gemini}`
  (generalizing `scripts/probe_gpt5_capture.py`) makes one tool-call-shaped request
  per provider, then writes a redacted AIMessage dump (additional_kwargs keys+values
  relevant to reasoning state, content shape, response_metadata, library version) to
  `tests/fixtures/provider_payloads/{provider}.json`.
- **D-10-12: Fixtures AUGMENT, never replace, the hand-written synthetic cases.**
  Adapter unit tests gain parametrized cases that load the real-wire fixtures; the
  existing synthetic dicts stay (they document the contract and run without files).
  This closes the exact gap class that produced the Gemini lcgg key-shape miss
  (D-09-09) and the 4 live-only Anthropic bugs.
- **D-10-13: Redaction is mandatory and tested.** Extend the existing probe redaction
  beyond `sk-` prefixes (Phase 9 review finding IN-04, deferred there — folded in
  here): redact any string matching common API-key shapes AND any env-var-sourced
  secret values; a unit test feeds a fake leaked key through the probe writer and
  asserts redaction. Probe fixtures are reviewed before commit like any code.
- **D-10-14: Probes are manual + documented, not scheduled.** `make probe-providers`
  is the documented MANDATORY pre-matrix step (runbook + Makefile comment); no CI/cron
  (no live keys in CI per D-09-10). Cost ~$0.01/provider/run.

### Sync/Async Test Debt (EVAL-06)

- **D-10-15:** Factory-level tests for the gpt-5 dispatch branch: `build_chat_model`
  with a `gpt-5-*` model returns `OpenAIReasoningChatModel` with
  `use_responses_api=True`; with `gpt-4o-mini` returns plain `ChatOpenAI`
  (`app/llm_factory.py:350-362` — currently ZERO tests reference
  `use_responses_api` / `_is_openai_reasoning_model`).
- **D-10-16:** `ScriptedChatModel` gets exercised via `ainvoke` (either an explicit
  `_agenerate` override or a test proving the BaseChatModel executor fallback works —
  planner verifies which; the graph only ever calls `ainvoke`).
- **D-10-17:** `vibe_check`'s sync `judge_llm.invoke` (`app/agent/critique/vibe.py:78`)
  inside the sync `critique` node: planner FIRST verifies whether LangGraph runs sync
  nodes in an executor under `ainvoke` (in which case the event loop is not blocked
  and a doc-comment suffices); only if it genuinely blocks does Phase 10 make the
  judge call non-blocking. Behavior must be byte-identical either way — this is the
  only agent-code touch permitted in the phase.

### Claude's Discretion

- Exact run-JSON error schema field names; summary.json error-array shape.
- `configs/eval_gates.yaml` schema details (as long as D-10-08's fields are present).
- Whether `check_eval_gates.py` is a new script or a mode of `eval_matrix.py`.
- Probe script structure, fixture JSON layout, and how parametrized fixture-loading
  tests are organized.
- Whether EVAL-04 needs any code beyond verifying the PR #104 test (likely none —
  a verification task in the plan is sufficient).
- Plan/wave decomposition (suggested seams: EVAL-01 runner+scorer error path;
  EVAL-03+02 gates file + quarantine; EVAL-05 probes+fixtures; EVAL-06+04 test debt).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Definition
- `.planning/ROADMAP.md` §Phase 10 — Goal, EVAL-01..06 success criteria (re-scoped
  2026-06-10; success criteria are unusually concrete — treat as acceptance tests).
- `.planning/REQUIREMENTS.md` §Eval Harness Honesty (Phase 10) — EVAL-01..06 verbatim.
  NOTE: this file is intentionally NOT git-tracked; it exists in the working tree.
- `.planning/v2.2-MILESTONE-SEED.md` — What is explicitly NOT this phase (decisiveness,
  critique tuning, prune/replay changes). Read to avoid scope bleed.

### Evidence Base (why each EVAL exists)
- `eval_reports/2026-06-05T21-14-30Z/` — The fail-open proof: all 25 cells errored
  (OpenAI 429 / anthropic temp-400), medians read 1.0. EVAL-01's acceptance scenario.
- `eval_reports/2026-06-05T20-29-56Z/` — The clean matrix the current baseline was cut
  from; source of the honest anchor numbers behind D-10-07.
- `.planning/phases/09-per-provider-state-preservation-implementations/09-PROV-01-BLOCKER.md`
  — D-09-02 gate re-scope rationale; the per-run evidence tables.
- `.planning/phases/09-.../09-REVIEW.md` — IN-04 (redaction gap, folded into D-10-13);
  WR-01 pattern (silent no-op capture) that EVAL-05 fixtures guard against.

### Edit Targets (EVAL-01)
- `scripts/eval_agent.py` — `_run_prod_threading` exception paths (`:839-863` —
  partial_state scoring REMOVED per D-10-02), `_run_legacy_threading` (`:652-730`),
  `score_checks` (`:475-495` — None-score swallowing), per-run JSON writer, exit-code
  logic (`:1112`).
- `scripts/eval_matrix.py` — `run_matrix` subprocess loop (`:360-441`), summary
  aggregation (`:671-677` + median computation `eval_agent.py:1038-1044`), structural
  check (`:547-637` — extend to validate the new error-schema fields).
- `app/agent/critique/checks.py` — `refinement_minimal_edit` (`:355-577`); Branch-1
  abstain (`:488-491`) retained for completed non-refinement runs only; Branch-2
  fail-loud (`:496-500`) unchanged for completed runs.

### Edit Targets (EVAL-02/03)
- `configs/eval_matrix_refinement.yaml` — gate-status comments migrate to pointing at
  the new gates YAML (comments stay as breadcrumbs; numbers live in YAML only).
- `configs/eval_matrix.yaml` + the late_night scenario config — `baseline_eligible:
  false` quarantine flag (D-10-09/10).
- `configs/eval_baselines/late_night_closure_cascade.json` — `_observations`
  annotation only (no regen).
- NEW `configs/eval_gates.yaml`, NEW `scripts/check_eval_gates.py` (or eval_matrix
  mode), NEW `docs/eval_gates.md`, Makefile target `eval-gates-check`.

### Edit Targets (EVAL-05/06)
- `scripts/probe_gpt5_capture.py` — generalize to `probe_provider_capture.py`
  (also fixes 09-REVIEW IN-03's hardcoded phase-09 output path).
- NEW `tests/fixtures/provider_payloads/{provider}.json` + parametrized loader tests
  in `tests/unit/test_adapters.py`.
- `app/llm_factory.py:350-362` (gpt-5 dispatch — tests only), `:277-298`
  (`ScriptedChatModel` — possible `_agenerate` override per D-10-16).
- `app/agent/critique/vibe.py:78` — conditional per D-10-17.
- `tests/unit/test_eval_matrix.py` — PR #104 parity test
  (`test_baseline_provider_cells_match_matrix_entries`) is the EVAL-04 anchor; verify,
  extend only if new matrix files appear.

### Prior Decisions That Bind This Phase
- `.planning/phases/09-.../09-CONTEXT.md` — D-09-10 (local empirical / CI structural
  split — Phase 10 does NOT add live CI), D-09-02 (two-part gate shape extended by
  D-10-06).
- `.planning/phases/08-.../08-CONTEXT.md` — D-08-14 (conformance marker quarantine —
  promotion is Phase 11), D-08-15 (byte-identity regression pattern to imitate for
  behavior-preserving changes).
- `CLAUDE.md` — `make eval-matrix-refinement-structural-check` stays the CI hard gate;
  human checkpoint remains the empirical gate until Phase 11.

### Project Memories (verified current 2026-06-10)
- `phase10-rescope-plan` — the re-scope decision record + corrections to older memories.
- `project_eval_multi_turn_threading_bug` — why late_night is quarantined not migrated.
- `project_phase6_d_06_09_root_cause` — fail-open baseline history (gate-realism
  context for D-10-06/07).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **PR #104 parity test** (`tests/unit/test_eval_matrix.py`,
  `_DEFERRED_BASELINE_CELLS` pattern) — EVAL-04 is mostly done; the deferred-cells
  dict is the template for any further "documented deferral" enforcement.
- **`scripts/probe_gpt5_capture.py`** — working probe skeleton (auth, message shape,
  dump format) to generalize for EVAL-05.
- **`scripts/check_baselines_fresh.py`** — exit-code + CI-integration pattern for
  `check_eval_gates.py` to imitate.
- **Structural-check mode** (`eval_matrix.py:547-637`) — established "validate shape
  without live calls" pattern; extend for error-schema validation.
- **`ScriptedChatModel`** — the no-API-key test vehicle for end-to-end error-path
  tests (a scripted LLM that raises on demand exercises EVAL-01 without live calls).

### Established Patterns
- Local empirical gate / CI structural gate split (D-06-10, D-09-10) — unchanged.
- Machine-readable config in `configs/` consumed by `scripts/` with a Make target
  wrapper — gates YAML follows.
- Test layering (memory `feedback_test_layering`): EVAL-01 needs unit (scorer branch),
  functional (scripted-LLM end-to-end error run), and the 21-14-30Z replay acceptance.
- Small focused commits; plan-per-seam (suggested seams in Claude's Discretion).

### Integration Points
- `eval_agent.py` per-run JSON → `eval_matrix.py` aggregation → `summary.json` →
  (new) `check_eval_gates.py` — the error-status field must thread through all four.
- Baseline writer does not exist yet as a discrete tool (baselines were hand-rolled
  from summary.json) — D-10-03's "refuse n_scored < n" rule lands wherever Phase 11
  builds it; Phase 10 records the rule in the gates doc and summary flags.

</code_context>

<specifics>
## Specific Ideas

- The `status: aspirational` gate state (D-10-07) exists so `eval-gates-check` can
  report "gpt-5-mini still below its target" without failing the build — keeping the
  D-09-02 Part A gate visible without making Phase 10/11 un-shippable on a known v2.2
  gap.
- Error-record stage values: `setup` (graph/LLM construction), `turn0`, `turnN`
  (N≥1) — granular enough to distinguish "embeddings died mid-refinement" from
  "provider key invalid".
- The 21-14-30Z replay test should be cheap: simulate with `ScriptedChatModel`-style
  raising stubs, not by re-running live calls.

</specifics>

<deferred>
## Deferred Ideas

- **late_night prod-threading migration / scenario redesign** — quarantined in Phase 10
  (D-10-09); redesigning the closure-cascade scenario for prod threading (so its
  turn-2 scorers fire under text-only history) is its own piece of work — candidate
  for Phase 11 stretch or v2.2.
- **Parallel matrix execution** (`ProcessPoolExecutor` over cells) — explicitly out;
  revisit after EVAL-01 lands (rate-limit storms must become ERROR records first).
  Candidate for Phase 11 (regen wall-clock is ~62 min sequential).
- **CI promotion of gates / conformance marker / live-provider secrets** — Phase 11
  BASE-03 by standing decision (D-09-10, D-08-14).
- **Hard-fail non-regression deltas** (advisory→hard promotion of median-drop checks)
  — Phase 11 BASE-03 decides after fresh baselines exist (D-10-07).
- **Baseline-writer tool** (summary.json → baseline JSON with n_scored enforcement) —
  Phase 11 BASE-01 builds it; Phase 10 only specifies the refusal rule (D-10-03).
- **Anything decisiveness-related** — `.planning/v2.2-MILESTONE-SEED.md` is the
  canonical parking lot (critique thresholds, commit contract, prune/replay richness,
  step-count latency).

</deferred>

---

*Phase: 10-eval-harness-honesty*
*Context gathered: 2026-06-10*
