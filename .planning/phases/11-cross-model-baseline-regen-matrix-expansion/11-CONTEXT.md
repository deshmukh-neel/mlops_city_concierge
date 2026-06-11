# Phase 11: Cross-Model Baseline Regen + Matrix Expansion - Context

**Gathered:** 2026-06-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 11 turns Phase 10's honest *mechanisms* into honest *measurements* and locks them
into CI. All eval baselines are regenerated under DB-up conditions with the Phase-7
prompt and Phase-9 adapters active (including the carried-forward anthropic n=5
re-measure and gemini first-ever n=5); the cross-model matrix gains `gpt-5-mini`,
`claude-sonnet-4-6`, and `deepseek-reasoner`; the per-family merge gates from
`configs/eval_gates.yaml` become CI-enforceable; and the staleness check covers the
new surface. Phase 10's deferred review findings (WR-05..WR-12) land here because
they change scorer/exit-code semantics the regen will permanently bake in.

**In scope (BASE-01..04 + Phase-10 carry-overs):**
- Wave-0 harness pre-fixes that change recorded numbers: `committed_itinerary_rate`
  threading into matrix cell scorers, WR-06, WR-08, WR-09, WR-12 (BASE-01 precondition).
- Baseline-writer tool enforcing the D-10-03 refusal rule + `docs/baseline_regen.md`
  runbook + pre-regen snapshots + the live n=5 regen itself (BASE-01).
- `configs/eval_matrix.yaml` cross-model expansion (3 new entries, flag-OFF) (BASE-02).
- CI gate enforcement without live keys: gates-vs-committed-baselines check, synthetic-
  regression proof test, WR-07 exit-code contract, WR-05 advisory implementation,
  WR-11 structural-check fix, conformance-marker CI promotion, gate re-ratification
  from fresh n=5 data (BASE-03).
- Staleness watch-set extension + dry-run test (BASE-04).
- WR-10 fixture-fidelity fix (non-blocking, any wave).

**Out of scope:**
- Anything that changes agent behavior: prompts, critique thresholds, commit contract,
  `_prune_for_llm`, adapter logic — v2.2 decisiveness territory
  (`.planning/v2.2-MILESTONE-SEED.md`). The Wave-0 scorer fixes change *measurement*
  semantics only, never agent behavior.
- Making the gpt-5-mini aspirational gate PASS — known v2.2 gap; Phase 11 only keeps
  it visible (`status: aspirational`).
- `late_night_closure_cascade` regen or prod-threading redesign (D-10-09/10 standing).
- Parallel matrix execution (`ProcessPoolExecutor` over cells) — deferred again.
- Promoting Gemini 3 into the cross-model prod matrix (PROV-FUT-01).
- Advisory→hard promotion of `refinement_minimal_edit` medians — v2.2.
- New scorers beyond the WR-12 semantics fix.

</domain>

<decisions>
## Implementation Decisions

User delegated all four discussed areas to Claude's recommendations (2026-06-11,
"ill let u decide on whats best"). Decisions below are locked with the same force
as user-stated ones, per the Phase-10 precedent.

### Wave-0 Pre-Regen Harness Fixes (sequencing)

- **D-11-01: A Wave-0 of measurement-semantics fixes lands BEFORE any live regen
  run.** Members: D-11-02 (commit-rate threading), WR-06, WR-08, WR-09, WR-12.
  Rationale: each changes numbers that BASE-01 writes into committed baselines; fixing
  them after regen means regenerating twice. Live regen is the LAST wave of the phase.
- **D-11-02: `committed_itinerary_rate` is threaded into each matrix cell's scorers
  block in `summary.json`.** `scripts/eval_agent.py:1092` already computes it; the
  matrix aggregation must carry it so `check_eval_gates.py` hard gates flip from
  "not-evaluable" (T-10-04-01) to enforced. This is the keystone — every BASE-03 hard
  gate rides on this metric.
- **D-11-03: WR-12 — `category_compliance` ABSTAINS (returns None, excluded from
  aggregation) on zero committed stops.** Not 1.0 (current fail-open contradiction of
  its own docstring), not 0.0 (double-penalty). Decisiveness failure is already the
  hard-gated `committed_itinerary_rate` signal; category compliance measures only what
  was committed. Docstring updated + unit tests for the zero-stop branch. Same
  abstain-vs-penalize logic as the Branch-1 abstain retained in D-10-04.
- **D-11-04: WR-09 — an all-errored cell (`n_scored == 0`) publishes `None` (or
  omits) `deterministic_pass_rate` / `tool_success_rate`, never 1.0**, with
  `cell_valid: false`. Aligns the derived-rate fields with D-10-03 cell-validity.
- **D-11-05: WR-08 — prod-threading scratch keys (`prior_committed_stops`,
  `prior_stops_obj`) are excluded from tool-call counting** before regen, so
  `tool_calls_mean` doesn't bake in phantom calls.
- **D-11-06: WR-06 — the single-turn eval path gets `make_error_record` (stage
  `"setup"`) like the multi-turn paths**, so one transient failure no longer aborts a
  whole run silently.
- **WR-10** (probe fixtures stringify `additional_kwargs` values, so adapter fixture
  tests never see real-typed bytes/dict payloads) is in scope but non-blocking — any
  wave; it affects test fidelity, not recorded baselines.

### BASE-01 Regen Mechanics

- **D-11-07: Baselines are written by a discrete tool (`scripts/write_baselines.py`,
  name at planner's discretion), never hand-rolled again.** It reads a matrix
  `summary.json` and writes/updates baseline JSON cells. It REFUSES: cells with
  `n_scored < n_requested` (D-10-03 rule, recorded in Phase 10, built here) and
  scenarios with `baseline_eligible: false`. It stamps `generated_at` /
  `generated_by` mechanically.
- **D-11-08: Runbook at `docs/baseline_regen.md`** (BASE-01 names it). Ordered steps:
  (1) preconditions — all 4 provider keys, cloud-sql-proxy / DB-up verification, an
  embeddings sanity probe (the 21-14-30Z failure was an exhausted OpenAI embeddings
  quota poisoning every cell regardless of LLM provider); (2) `make probe-providers`
  (D-10-14 mandatory pre-matrix step); (3) snapshot per D-11-09; (4) matrix runs at
  RUNS=5 for both `configs/eval_matrix.yaml` and `configs/eval_matrix_refinement.yaml`;
  (5) `write_baselines` (refuses invalid cells); (6) `make eval-gates-check` against
  the fresh summary; (7) commit baselines + snapshots together.
- **D-11-09: Pre-regen snapshots** of all three baseline JSONs to
  `configs/eval_baselines/_snapshots/*.pre-phase11.json` (existing `_snapshots/`
  pattern; the fail-open-saturated v2.0 numbers stay auditable after replacement).
- **D-11-10: Regen scope:** `omakase_mission_open_ended` across the expanded
  cross-model matrix (D-11-11) and `refinement_cheaper` across all six refinement
  entries — including the anthropic n=5 re-measure (replacing the n=1
  SHIPPED-WITH-GAP cell) and the gemini first-ever n=5. `late_night_closure_cascade`
  is NOT regenerated (D-10-09/10 standing; its `_observations` annotation stays).
  n=5 everywhere; per-family thinking/temperature policies stay exactly as shipped in
  `app/llm_factory.py` (no tuning — `feedback_temp1_reasoning_off_all_models` + its
  documented Claude/Gemini/DeepSeek-reasoner carve-outs stand).
- **D-11-11 (gemini failure branch): gemini is logged-not-gated, so errored gemini
  cells do NOT block BASE-01.** If gemini cells error at regen time, the writer
  refuses them per D-11-07, the gap is recorded as a documented deferral in the
  EVAL-04 parity test's `_DEFERRED_BASELINE_CELLS` dict, and the runbook notes the
  retry procedure. Gated families (gpt-4o-mini, anthropic) erroring DOES block —
  rerun until clean per the runbook.

### BASE-02 Matrix Expansion

- **D-11-12: `configs/eval_matrix.yaml` gains `openai/gpt-5-mini`,
  `anthropic/claude-sonnet-4-6`, `deepseek/deepseek-reasoner` as first-turn flag-OFF
  entries**, alongside the existing `openai/gpt-4o-mini` + `deepseek/deepseek-chat`.
  Gemini stays OUT of the cross-model matrix (PROV-04 experimental standing — "absent
  from the prod matrix"); gemini coverage lives only in
  `configs/eval_matrix_refinement.yaml` as logged.
- **D-11-13: `late_night_closure_cascade` is removed from the default
  `eval_matrix.yaml` scenarios list.** It stays runnable explicitly (eval-agent
  SCENARIOS param) as a diagnostic; the YAML comment documents both the D-10-09
  quarantine and the cost rationale (5 providers × 5 runs of a baseline-ineligible
  scenario is pure burn). This sharpens, not contradicts, D-10-09 "stays runnable".
- **D-11-14: Execution stays sequential.** Parallel matrix execution remains deferred
  — rate-limit storms are exactly what poisoned 21-14-30Z, and regen is a per-change
  event, not a hot path. The runbook may document per-provider chunked invocations for
  wall-clock management; no executor-pool code in Phase 11.
- **BASE-02 acceptance** ("all four providers without errors") interpreted under
  EVAL-01 semantics: the final regen summary shows `n_errored == 0` for every gated
  or logged cell of the four cross-model providers.

### BASE-03 CI Enforcement & Gate Re-Ratification

- **D-11-15: CI stays live-key-free (D-09-10 standing). CI gate enforcement is three
  mechanical pieces:** (a) `check_eval_gates.py` gains a mode that evaluates the
  committed `configs/eval_baselines/*.json` (the canonical committed empirical
  record) so a committed regression below a hard gate fails CI; (b) a unit test
  proves the `gpt-5-mini × refinement_cheaper` anchor gate fires (exit 1) on a
  synthetic regressed summary — the BASE-03 "fires on synthetic regression"
  criterion; (c) a `ci.yml` step runs the baselines-mode gate check via a named
  Makefile target alongside the existing structural checks. Whether baselines-mode
  synthesizes a summary-shaped dict or reads baseline JSON natively is planner's
  discretion.
- **D-11-16: WR-07 exit-code contract:** `eval_agent` exits 0 = clean, 1 =
  model-behavior violations, 2 = infra failure (extends the WR-04 precedent that
  classified structural defects as infra exit 2); `run_matrix` reports and exits
  distinguishing violation-cells from error-cells. Folded into BASE-03 because gate
  wiring consumes these exit codes.
- **D-11-17: WR-05 — advisory gate entries are IMPLEMENTED (report-only WARN), not
  deleted.** `configs/eval_gates.yaml` and `docs/eval_gates.md` already promise
  advisory reporting; the unresolvable metric name (`refinement_minimal_edit_median`)
  gets a resolution path in the checker. Advisory misses never affect exit codes.
- **D-11-18: WR-11 — structural-check "Check 6" exercises the real
  `make_error_record` schema** (build the record via the real function, validate the
  real output) instead of the current tautology.
- **D-11-19: The `reasoning_conformance` pytest marker is promoted to required CI.**
  It is mock-driven (no live keys), and the fixture-loading cases already SKIP
  gracefully when fixtures are absent (D-10-12). This resolves the D-08-14 promotion
  decision deferred to this phase.
- **D-11-20: Gate re-ratification from fresh n=5 data:**
  `anthropic/claude-sonnet-4-6` moves `provisional-n1 → active` with its value
  re-derived under the D-10-07 "absorb one stochastic miss at n=5" rule;
  `openai/gpt-5-mini` STAYS `aspirational` (expected to fail until v2.2 — Phase 11
  must remain shippable on this known gap); `deepseek/*` and `gemini/*` stay
  `logged`. Advisory medians stay advisory. Every status/value change gets a
  rationale line with a D-11 ID in `configs/eval_gates.yaml`.

### BASE-04 Staleness Extension

- **D-11-21: `scripts/check_baselines_fresh.py` watch-set extends beyond
  `app/agent/` to include `app/llm_factory.py` and `configs/eval_matrix*.yaml`.**
  Verified gap: only `AGENT_PREFIX = "app/agent/"` is watched today, yet provider
  branches, thinking policies, and temperature clamps live in `app/llm_factory.py`
  and directly change measured behavior. A dry-run test simulates an agent-loop
  change without baseline refresh and asserts exit 1 (the BASE-04 criterion).
  Scorers (`app/agent/critique/`) are already covered by the existing prefix.

### Claude's Discretion

- Script/tool names, Makefile target names, runbook section structure.
- Baselines-mode implementation shape in `check_eval_gates.py` (synthesized summary
  vs native baseline reading).
- IN-01..IN-06 info items from 10-REVIEW.md: fold trivial ones in opportunistically;
  none are required for the phase gates.
- Plan/wave decomposition. Suggested seams: **Wave 0** = D-11-01..06 harness pre-fixes
  (no live calls); **Wave 1** = matrix expansion + writer tool + CI/gates/staleness
  wiring + WR-10 (no live calls); **Wave 2** = live regen + gate re-ratification +
  runbook finalization (live calls, DB-up, LAST so everything it bakes is final).
- Exact `n_requested` plumbing for the writer's refusal check (summary already
  carries `n_scored`/`n_errored` per Phase 10).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Definition
- `.planning/ROADMAP.md` §Phase 11 — Goal, BASE-01..04 success criteria, and the
  WR-05..12 carry-over block (verbatim scope of the Phase-10 review debt).
- `.planning/REQUIREMENTS.md` §Cross-Model Baseline Regen + Matrix Expansion —
  BASE-01..04 verbatim. NOTE: intentionally NOT git-tracked; exists in working tree.
- `.planning/phases/10-eval-harness-honesty/10-REVIEW.md` — WR-05..WR-12 + IN-01..06
  full finding text (ROADMAP carries summaries; the review has file:line specifics).
- `.planning/v2.2-MILESTONE-SEED.md` — what is explicitly NOT this phase
  (decisiveness, critique tuning, prune/replay changes). Read to avoid scope bleed.

### Phase-10 Mechanisms This Phase Consumes
- `configs/eval_gates.yaml` — single source of truth for gates (D-10-05); the NOTE
  block documents the commit-rate not-evaluable condition D-11-02 closes; statuses
  re-ratified per D-11-20.
- `scripts/check_eval_gates.py` — gate checker; gains baselines mode (D-11-15);
  advisory implementation (D-11-17); exit codes 0/1/2 already match
  check_baselines_fresh.
- `docs/eval_gates.md` — gate semantics doc; renders the YAML; update alongside
  re-ratification (never duplicate numbers).
- `scripts/eval_agent.py` — `committed_itinerary_rate` computed at `:1092`;
  error-record machinery (`make_error_record`), single-turn WR-06 target, exit-code
  WR-07 target.
- `scripts/eval_matrix.py` — cell aggregation (`aggregate_cell_jsons`), structural
  check (WR-11 target), run_matrix exit semantics (WR-07).
- `app/agent/critique/checks.py` — `category_compliance` zero-stop branch (WR-12 /
  D-11-03).
- `scripts/probe_provider_capture.py` + `tests/fixtures/provider_payloads/` —
  probe/fixture machinery; WR-10 stringification fix target.
- `scripts/check_baselines_fresh.py` — staleness gate; watch-set extension target
  (D-11-21); also the exit-code + CI-integration pattern to imitate.

### Edit Targets (configs & CI)
- `configs/eval_matrix.yaml` — 3 new entries + late_night scenario removal
  (D-11-12/13).
- `configs/eval_matrix_refinement.yaml` — unchanged entries; comments may need
  gate-status breadcrumb updates after re-ratification.
- `configs/eval_baselines/{omakase_mission_open_ended,refinement_cheaper}.json` —
  regen targets; `late_night_closure_cascade.json` untouched.
- `configs/eval_baselines/_snapshots/` — pre-phase11 snapshots (D-11-09).
- `.github/workflows/ci.yml` — gates-check step (D-11-15c), conformance-marker
  promotion (D-11-19); note `lint-baselines` + `eval-matrix` jobs' existing shapes;
  eval-matrix installs must NOT use `--no-root` (memory `ci_no_root_eval_matrix`).
- `Makefile` — new targets: baselines-mode gate check, write-baselines wrapper;
  existing `eval-matrix`, `eval-matrix-refinement`, `probe-providers`,
  `eval-gates-check` reused.
- NEW: `scripts/write_baselines.py` (D-11-07), `docs/baseline_regen.md` (D-11-08).

### Prior Decisions That Bind This Phase
- `.planning/phases/10-eval-harness-honesty/10-CONTEXT.md` — D-10-03 (refusal rule
  the writer implements), D-10-05..08 (gate file semantics), D-10-09/10 (late_night
  quarantine — do not regen), D-10-12 (fixtures augment, never replace), D-10-14
  (probes manual, no CI/cron).
- `.planning/phases/09-per-provider-state-preservation-implementations/09-CONTEXT.md`
  — D-09-10 (local empirical / CI structural split — D-11-15 honors it), D-09-06
  (Claude thinking carve-out — regen does not tune), D-09-02 (two-part anchor gate).
- `.planning/phases/08-reasoning-state-thread-through-contract-conformance-harness/08-CONTEXT.md`
  — D-08-14 (conformance marker quarantine — resolved by D-11-19).
- `CLAUDE.md` — `make eval-matrix-refinement-structural-check` stays a CI hard gate;
  user merges PRs themselves.

### Evidence Base
- `eval_reports/2026-06-05T21-14-30Z/` — the fail-open disaster; D-11-08's
  preconditions exist to prevent its recurrence.
- `eval_reports/2026-06-05T20-29-56Z/` — the clean matrix behind current honest
  anchor numbers; comparison point for the regen.
- `configs/eval_baselines/refinement_cheaper.json` `_observations` (anthropic cell) —
  the n=1 SHIPPED-WITH-GAP record the anthropic n=5 re-measure replaces.

### Project Memories (verified current 2026-06-11)
- `project_phase10_rescope_plan` — why BASE-01..04 live here.
- `project_phase4_d_04_14_locked` + `project_phase6_d_06_09_root_cause` — the
  fail-open-saturated baseline history BASE-01 finally replaces (named in BASE-01).
- `project_ci_no_root_eval_matrix` — CI install constraint for eval-matrix jobs.
- `feedback_temp1_reasoning_off_all_models` — regen runs policies as-shipped, no
  tuning (with D-09-06 carve-outs).
- `feedback_user_merges_prs`, `feedback_small_focused_commits`,
  `feedback_test_layering` — process constraints.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`check_baselines_fresh.py`** — exit-code contract (0/1/2) + CI job shape that the
  baselines-mode gate check and the dry-run staleness test imitate.
- **`check_eval_gates.py`** — already walks nested `scenarios->providers` (CR-01
  fix), validates status vocabulary at load (WR-03), and reports not-evaluable
  distinctly — the baselines mode is an input-source extension, not a rewrite.
- **`_DEFERRED_BASELINE_CELLS`** in `tests/unit/test_eval_matrix.py` (PR #104) — the
  documented-deferral mechanism for any gemini regen failure (D-11-11).
- **`_snapshots/` pattern** (`configs/eval_baselines/_snapshots/` + README) — exact
  template for pre-phase11 snapshots.
- **`make_error_record` + `n_scored`/`n_errored`/`cell_valid` threading** (Phase 10)
  — the writer's refusal check reads these fields directly from summary.json.
- **`ScriptedChatModel`** — no-API-key vehicle for the synthetic-regression gate test
  and the WR-06 single-turn error-path test.
- **`make probe-providers`** fixtures — WR-10 fix improves their type fidelity.

### Established Patterns
- Machine-readable config in `configs/` + `scripts/` checker + Make target + CI step
  — every new enforcement piece follows this chain.
- Local empirical gate / CI structural gate split (D-09-10) — D-11-15 keeps CI on
  committed artifacts + synthetic tests only.
- Snapshot-then-regen (Phase 7 precedent, `_snapshots/` dir).
- Test layering (`feedback_test_layering`): each Wave-0 scorer fix needs unit +
  functional (scripted-LLM) coverage; the staleness extension needs the dry-run test.
- Small focused commits; live-regen results committed separately from code.

### Integration Points
- `eval_agent.py` per-run JSON → `eval_matrix.py` `aggregate_cell_jsons` →
  `summary.json` → `write_baselines.py` (NEW) → `configs/eval_baselines/*.json` →
  `check_eval_gates.py` baselines mode (NEW) → CI. The commit-rate field (D-11-02)
  must survive every hop.
- `check_baselines_fresh.py` watch-set ↔ the files regen actually depends on
  (D-11-21 closes the `app/llm_factory.py` gap).
- Branch: `gsd/phase-11-cross-model-baseline-regen-matrix-expansion` off main
  `cc1407c` (post-PR #105; Phase 10 fully merged, working tree verified identical).

### External Preconditions (verify before Wave 2 live regen)
- OpenAI embeddings quota topped up (done 2026-06-10; runbook re-verifies via sanity
  probe).
- Cloud SQL reachable via cloud-sql-proxy :5433 (or local Postgres) — instance is
  `mlops--city-concierge` (double dash), DB inside is single-dash.
- All 4 provider keys live (`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`,
  `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`).

</code_context>

<specifics>
## Specific Ideas

- The regen is the LAST wave on purpose: every Wave-0/Wave-1 change that touches
  measurement semantics lands first so baselines are written exactly once. If a
  Wave-0 fix slips after regen, the regen repeats — the runbook makes a second pass
  cheap, but the plan ordering should make it unnecessary.
- `check_eval_gates.py` baselines mode doubles as the local pre-commit sanity step in
  the runbook (step 6) — same code path CI runs, so "passes locally" means "passes CI".
- Expected post-regen gate picture (sanity expectations, not gates): gpt-4o-mini
  commit_rate 1.0 (gate ≥ 0.8 passes); gpt-5-mini ~0.4 (aspirational ≥ 0.6 reported,
  non-blocking); anthropic re-ratified from its n=5; deepseek/gemini logged. If
  gpt-4o-mini's commit rate drops below 0.8 on honest regen, STOP and investigate
  before committing baselines — that would be a real anchor regression, not noise.

</specifics>

<deferred>
## Deferred Ideas

- **Parallel matrix execution** (`ProcessPoolExecutor` over cells) — deferred again
  (D-11-14); revisit in v2.2 if regen cadence increases.
- **`late_night_closure_cascade` prod-threading redesign** — v2.2 candidate
  (D-10-09 standing; D-11-13 only trims it from the default matrix run).
- **Advisory→hard promotion** of `refinement_minimal_edit` median gates — v2.2,
  after decisiveness work moves the medians off 0.0.
- **Gemini 3 promotion to gated / prod matrix** — PROV-FUT-01; needs the critique-loop
  fix (`LOW_SIMILARITY_THRESHOLD` interplay) that remains v2.2 scope.
- **gpt-5-mini aspirational gate to PASS** — the v2.2 decisiveness milestone's
  empirical target; Phase 11 keeps it visible, not green.
- **IN-01..IN-06** (10-REVIEW info items) — opportunistic only; whatever isn't folded
  carries to the v2.1 milestone audit.
- **o-series / Kimi adapters** — PROV-FUT-02/03, unchanged.

</deferred>

---

*Phase: 11-cross-model-baseline-regen-matrix-expansion*
*Context gathered: 2026-06-11*
